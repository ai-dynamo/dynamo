// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV transfer hint attached to a booked selection.

use super::*;
use crate::plugins::worker_selection::experimental::{
    KvTransferAction, KvTransferInput, KvTransferPolicy, KvTransferSource,
};
use parking_lot::Mutex;

/// Pick the best router-hint source for `target`: a same-role worker (or
/// cache owner) holding a longer root-aligned prefix than the target's own
/// `target_cached_prefix_blocks`, with a non-empty control endpoint.
/// Whether any worker in the published partition snapshot advertises a router
/// hint worker type (worker-level metadata, so one rank suffices).
pub(super) fn hint_capable_partition(configs: &HashMap<WorkerId, SelectionWorkerConfig>) -> bool {
    configs.values().any(|config| {
        config
            .router_hint_worker_type
            .as_deref()
            .is_some_and(|worker_type| !worker_type.is_empty())
    })
}

pub(super) fn transfer_hint_for_selection(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    target: WorkerWithDpRank,
    target_cached_prefix_blocks: u32,
    candidates: Option<&KvTransferCandidates>,
    transfer_policy: Option<&Mutex<Box<dyn KvTransferPolicy>>>,
) -> Option<KvSourceLocationsPayload> {
    let candidates = candidates?;
    let target_config = configs.get(&target.worker_id)?;
    let target_metadata = target_config.kv_hint_transfer_metadata_for_dp_rank(target.dp_rank)?;

    let prefix_blocks_to_beat = usize::try_from(target_cached_prefix_blocks).unwrap_or(usize::MAX);
    let is_eligible_source = |source| match source {
        KvTransferCandidateSource::Worker(worker) => {
            worker != target
                && configs.get(&worker.worker_id).is_some_and(|config| {
                    config.kv_event_source_mode.as_deref() != Some("state_agent_v2")
                        && config
                            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)
                            .is_some_and(|source_metadata| {
                                source_metadata.worker_type == target_metadata.worker_type
                                    && source_metadata
                                        .source_control_endpoint
                                        .is_some_and(|endpoint| !endpoint.is_empty())
                            })
                })
        }
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.router_hint_source(owner))
            .is_some_and(|source| {
                source.attached_worker != Some(target)
                    && source.metadata.worker_type == target_metadata.worker_type
                    && !source.metadata.source_control_endpoint.is_empty()
            }),
    };
    let (source, block_hashes) = match transfer_policy {
        None => candidates.best_source(prefix_blocks_to_beat, is_eligible_source)?,
        Some(transfer_policy) => {
            let (source, prefix_blocks) = policy_source(
                transfer_policy,
                configs,
                target,
                target_cached_prefix_blocks,
                candidates,
                is_eligible_source,
            )?;
            (
                source,
                candidates.block_hashes.get(..prefix_blocks)?.to_vec(),
            )
        }
    };
    let source_control_endpoint = match source {
        KvTransferCandidateSource::Worker(worker) => configs
            .get(&worker.worker_id)?
            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)?
            .source_control_endpoint?
            .to_string(),
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()?
            .router_hint_source(owner)?
            .metadata
            .source_control_endpoint
            .clone(),
    };
    if block_hashes.is_empty() {
        return None;
    }
    Some(KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes,
    })
}

/// Let the partition's fetch-hint policy choose among every eligible source. Sources are offered
/// longest prefix first, with the same tie-break as [`KvTransferCandidates::best_source`], so
/// index 0 is the router's default choice.
fn policy_source(
    transfer_policy: &Mutex<Box<dyn KvTransferPolicy>>,
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    target: WorkerWithDpRank,
    local_prefix_blocks: u32,
    candidates: &KvTransferCandidates,
    mut is_eligible_source: impl FnMut(KvTransferCandidateSource) -> bool,
) -> Option<(KvTransferCandidateSource, usize)> {
    let prefix_blocks_to_beat = usize::try_from(local_prefix_blocks).unwrap_or(usize::MAX);
    let mut sources: Vec<KvTransferSource> = candidates
        .owner_prefix_blocks
        .iter()
        .copied()
        .filter(|(source, blocks)| {
            *blocks > prefix_blocks_to_beat
                && *blocks <= candidates.block_hashes.len()
                && is_eligible_source(*source)
        })
        .map(|(source, prefix_blocks)| KvTransferSource {
            source,
            worker: match source {
                KvTransferCandidateSource::Worker(worker) => Some(worker),
                KvTransferCandidateSource::CacheOwner(owner) => candidates
                    .routing_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.router_hint_source(owner))
                    .and_then(|source| source.attached_worker),
            },
            prefix_blocks,
        })
        .collect();
    if sources.is_empty() {
        return None;
    }
    sources.sort_unstable_by(|left, right| {
        right
            .prefix_blocks
            .cmp(&left.prefix_blocks)
            .then_with(|| left.source.cmp(&right.source))
    });
    let action = transfer_policy.lock().decide(KvTransferInput {
        target,
        local_prefix_blocks,
        sources: &sources,
        workers: configs,
    });
    let chosen = match action {
        KvTransferAction::Skip => None,
        KvTransferAction::FetchFrom(index) => sources.get(index),
        KvTransferAction::Default => sources.first(),
    };
    tracing::debug!(
        worker_id = target.worker_id,
        dp_rank = target.dp_rank,
        local_prefix_blocks,
        source_count = sources.len(),
        chosen_prefix_blocks = chosen.map(|source| source.prefix_blocks),
        ?action,
        "KV transfer policy decision"
    );
    if chosen.is_none() && action != KvTransferAction::Skip {
        tracing::warn!(
            ?action,
            source_count = sources.len(),
            "KV transfer policy chose a source out of range; attaching no hint"
        );
    }
    chosen.map(|source| (source.source, source.prefix_blocks))
}

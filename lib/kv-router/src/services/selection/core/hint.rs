// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV transfer hint attached to a booked selection.

use super::*;
use crate::kv_hints::KvTransferSourceChoice;

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

/// The fetch payload for the source worker selection chose (see
/// [`crate::kv_hints::KvTransferSourceSummary`]). The endpoint comes from the
/// current snapshot, so a source that left since selection gets no hint.
pub(super) fn transfer_hint_payload(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    source: KvTransferSourceChoice,
    candidates: &KvTransferCandidates,
) -> Option<KvSourceLocationsPayload> {
    let source_control_endpoint = match source.source {
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
    let block_hashes = candidates.block_hashes.get(..source.prefix_blocks)?;
    if source_control_endpoint.is_empty() || block_hashes.is_empty() {
        return None;
    }
    Some(KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes: block_hashes.to_vec(),
    })
}

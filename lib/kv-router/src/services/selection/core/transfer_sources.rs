// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advisory transfer-source lookup, independent of execution admission.

use super::*;
use crate::protocols::KvTransferEnforcement;

/// Inventory and endpoint protocol must agree. Host is the existing CPU path;
/// GPU uses only producer-published GpuDirect blocks and a GPU-only pin endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferSourceMemoryKind {
    Host,
    Gpu,
}

/// Inputs to a read-only lookup on the donor partition. The target is already
/// admitted elsewhere. The explicit domain certifies identical model weights,
/// tokenizer, KV layout and rank mapping; topology labels do not certify this.
pub struct TransferSourceQuery<'a> {
    pub model_name: &'a str,
    pub routing_group: &'a str,
    pub prompt: PromptView<'a>,
    pub target: WorkerWithDpRank,
    pub target_worker_type: &'a str,
    pub memory_kind: TransferSourceMemoryKind,
    /// Explicit target support for kv.fetch_gpu@1.0, never inferred from CPU support.
    pub target_gpu_direct: bool,
    pub target_transfer_domain: Option<&'a str>,
    /// Independent topology-island policy of the already admitted target.
    pub target_topology_domains: Option<&'a HashMap<String, String>>,
    pub target_kv_transfer_domain: Option<&'a str>,
    pub target_kv_transfer_enforcement: Option<KvTransferEnforcement>,
    pub target_block_size: u32,
    pub target_is_eagle: bool,
    /// A donor must hold strictly more complete blocks than this baseline.
    pub minimum_prefix_blocks: usize,
    pub deadline: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferSourceResult {
    pub source: KvTransferCandidateSource,
    pub memory_kind: TransferSourceMemoryKind,
    /// Full root-aligned prefix, including blocks already cached at the target.
    pub payload: KvSourceLocationsPayload,
}

/// Filter rooted inventory using certified endpoint, role and topology facts.
/// This pure eligibility check does not admit or dispatch execution.
pub fn transfer_source_for_target(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    query: &TransferSourceQuery<'_>,
    candidates: &KvTransferCandidates,
) -> Option<TransferSourceResult> {
    if query.memory_kind == TransferSourceMemoryKind::Gpu && !query.target_gpu_direct {
        return None;
    }
    let domain = query
        .target_transfer_domain
        .filter(|domain| !domain.trim().is_empty())?;
    let source_config = |worker: WorkerWithDpRank| {
        if worker == query.target {
            return None;
        }
        let config = configs.get(&worker.worker_id)?;
        let end = config
            .data_parallel_start_rank
            .saturating_add(config.data_parallel_size);
        if !(config.data_parallel_start_rank..end).contains(&worker.dp_rank)
            || config.router_hint_transfer_domain() != Some(domain)
            || config.is_eagle.unwrap_or(false) != query.target_is_eagle
        {
            return None;
        }
        let metadata = match query.memory_kind {
            TransferSourceMemoryKind::Host => {
                config.kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)
            }
            TransferSourceMemoryKind::Gpu => {
                config.kv_hint_gpu_transfer_metadata_for_dp_rank(worker.dp_rank)
            }
        }?;
        (metadata.worker_type == "decode").then_some(config)
    };
    let endpoint = |source| -> Option<&str> {
        match source {
            KvTransferCandidateSource::Worker(worker) => {
                let config = source_config(worker)?;
                if config.kv_event_source_mode.as_deref() == Some("state_agent_v2") {
                    return None;
                }
                match query.memory_kind {
                    TransferSourceMemoryKind::Host => {
                        config.kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)?
                    }
                    TransferSourceMemoryKind::Gpu => {
                        config.kv_hint_gpu_transfer_metadata_for_dp_rank(worker.dp_rank)?
                    }
                }
                .source_control_endpoint
            }
            KvTransferCandidateSource::CacheOwner(owner) => {
                // Persistent CPU owner metadata does not certify a live GPU lease service.
                if query.memory_kind == TransferSourceMemoryKind::Gpu {
                    return None;
                }
                let source = candidates
                    .routing_snapshot
                    .as_ref()?
                    .router_hint_source(owner)?;
                // Detached persistent owners have no certified layout metadata
                // in this protocol revision. Fail closed until that is added.
                source_config(source.attached_worker?)?;
                (source.metadata.worker_type == "decode")
                    .then_some(source.metadata.source_control_endpoint.as_str())
            }
        }
        .filter(|endpoint| !endpoint.trim().is_empty())
    };
    let (source, block_hashes) =
        candidates.best_ranked_source(query.minimum_prefix_blocks, |source| {
            endpoint(source)?;
            hint::transfer_source_topology_rank(
                configs,
                query.target_topology_domains,
                query.target_kv_transfer_domain,
                query.target_kv_transfer_enforcement,
                candidates,
                source,
            )
        })?;
    Some(TransferSourceResult {
        source,
        memory_kind: query.memory_kind,
        payload: KvSourceLocationsPayload {
            source_control_endpoint: endpoint(source)?.to_string(),
            block_hashes,
        },
    })
}

#[cfg(test)]
mod tests;

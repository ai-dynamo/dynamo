// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advisory transfer-source lookup, independent of execution admission.

use super::*;
use crate::protocols::KvTransferEnforcement;

pub(super) const MAX_IN_FLIGHT_TRANSFER_SOURCE_LOOKUPS: usize = 8;

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

impl SelectionCore {
    /// Ready retained GPU blocks already owned by the admitted target. This
    /// advisory count changes cost estimates only, never execution overlap or bookings.
    pub async fn query_local_gpu_direct_prefix(
        &self,
        query: &TransferSourceQuery<'_>,
    ) -> Result<Option<usize>, SelectionError> {
        self.ensure_running()?;
        let key = RoutingPartitionId::new(query.model_name, query.routing_group);
        let Some(partition) = self.partition(&key) else {
            return Ok(None);
        };
        let result = partition.query_local_gpu_direct_prefix(query).await;
        self.ensure_running()?;
        result
    }

    /// Query inventory without selecting, queuing, booking or dispatching work.
    /// Missing partitions and expired advisory deadlines yield no donor.
    pub async fn query_transfer_sources(
        &self,
        query: TransferSourceQuery<'_>,
    ) -> Result<Option<TransferSourceResult>, SelectionError> {
        self.ensure_running()?;
        let key = RoutingPartitionId::new(query.model_name, query.routing_group);
        let Some(partition) = self.partition(&key) else {
            return Ok(None);
        };
        let result = partition.query_transfer_sources(query).await;
        self.ensure_running()?;
        result
    }
}

impl SelectionPartition {
    pub async fn query_transfer_sources(
        &self,
        query: TransferSourceQuery<'_>,
    ) -> Result<Option<TransferSourceResult>, SelectionError> {
        let Some(candidates) = self.query_transfer_inventory(&query).await? else {
            return Ok(None);
        };
        // Re-read membership after inventory lookup; no departed endpoint is authoritative.
        let result = transfer_source_for_target(&self.0.workers_tx.borrow(), &query, &candidates);
        Ok((Instant::now() < query.deadline)
            .then_some(result)
            .flatten())
    }

    async fn query_local_gpu_direct_prefix(
        &self,
        query: &TransferSourceQuery<'_>,
    ) -> Result<Option<usize>, SelectionError> {
        if query.memory_kind != TransferSourceMemoryKind::Gpu || !query.target_gpu_direct {
            return Ok(None);
        }
        let target = {
            let workers = self.0.workers_tx.borrow();
            let Some(target) = workers.get(&query.target.worker_id) else {
                return Ok(None);
            };
            let end = target
                .data_parallel_start_rank
                .saturating_add(target.data_parallel_size);
            if !(target.data_parallel_start_rank..end).contains(&query.target.dp_rank)
                || target.router_hint_transfer_domain() != query.target_transfer_domain
                || target
                    .kv_hint_gpu_transfer_metadata_for_dp_rank(query.target.dp_rank)
                    .is_none_or(|metadata| metadata.worker_type != "prefill")
            {
                return Ok(None);
            }
            target.clone()
        };
        let Some(candidates) = self.query_transfer_inventory(query).await? else {
            return Ok(None);
        };
        if Instant::now() >= query.deadline
            || self.0.workers_tx.borrow().get(&query.target.worker_id) != Some(&target)
        {
            return Ok(None);
        }
        let blocks = candidates
            .owner_prefix_blocks
            .iter()
            .find_map(|(source, blocks)| {
                (*source == KvTransferCandidateSource::Worker(query.target)).then_some(*blocks)
            })
            .unwrap_or(0);
        Ok((blocks <= candidates.block_hashes.len()).then_some(blocks))
    }

    async fn query_transfer_inventory(
        &self,
        query: &TransferSourceQuery<'_>,
    ) -> Result<Option<KvTransferCandidates>, SelectionError> {
        let entry = &self.0;
        if Instant::now() >= query.deadline
            || entry.key.model_name != query.model_name
            || entry.key.routing_group != query.routing_group
            || entry.block_size != query.target_block_size
            || entry.is_eagle != query.target_is_eagle
            || query
                .prompt
                .is_eagle
                .is_some_and(|value| value != entry.is_eagle)
            || query.target_worker_type != "prefill"
            || (query.memory_kind == TransferSourceMemoryKind::Gpu && !query.target_gpu_direct)
            || query
                .target_transfer_domain
                .is_none_or(|domain| domain.trim().is_empty())
            || !entry.indexer.supports_kv_transfer_chain_retention()
        {
            return Ok(None);
        }
        // Admission is immediate and independent of execution admission. Keep
        // the permit inside the blocking closure so expiry/cancellation cannot
        // free capacity while its abandoned walk is still queued or running.
        let Ok(permit) = entry.transfer_source_slots.clone().try_acquire_owned() else {
            return Ok(None);
        };
        let hashes = query
            .prompt
            .block_hashes_for_indexer(entry.block_size, entry.is_eagle)?;
        if Instant::now() >= query.deadline {
            return Ok(None);
        }
        let lower_tier = match &entry.indexer {
            Indexer::Single { lower_tier, .. } | Indexer::Concurrent { lower_tier, .. } => {
                lower_tier.clone()
            }
            Indexer::Remote { .. } | Indexer::None => return Ok(None),
        };
        let memory_kind = query.memory_kind;
        // Retained GPU inventory is independent of native execution Device
        // events. Both walks stay off the executor and hold the permit through
        // abandoned/expired work; neither walk can acquire a GPU memory lease.
        let lookup = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            match memory_kind {
                TransferSourceMemoryKind::Host => lower_tier.host_transfer_candidates(&hashes),
                TransferSourceMemoryKind::Gpu => lower_tier.gpu_direct_transfer_candidates(&hashes),
            }
        });
        let candidates = match tokio::time::timeout_at(query.deadline.into(), lookup).await {
            Ok(Ok(candidates)) => candidates,
            Ok(Err(error)) => return Err(SelectionError::Internal(error.to_string())),
            Err(_) => return Ok(None),
        };
        if Instant::now() >= query.deadline {
            return Ok(None);
        }
        // A completed empty walk is an observed zero; timeout/unsupported input
        // above remains None so callers cannot treat unknown target residency as cold.
        Ok(Some(candidates.unwrap_or(KvTransferCandidates {
            block_hashes: Vec::new(),
            owner_prefix_blocks: Vec::new(),
            routing_snapshot: None,
        })))
    }
}

fn transfer_source_for_target(
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

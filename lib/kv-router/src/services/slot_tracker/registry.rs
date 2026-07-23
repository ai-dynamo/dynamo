// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dashmap::DashMap;
use dynamo_tokens::SequenceHash;
use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::identity::RoutingPartitionId;
use crate::protocols::{PrefillLoadHint, WorkerId, WorkerWithDpRank};
use crate::scheduling::PotentialLoad;
use crate::sequences::topology::{WorkerDpRange, WorkerTopologyError};
use crate::sequences::{
    ActiveSequencesMultiWorker, PrefillTokenDeltas, ReplicaWorkerPolicy, SequenceError,
    SequenceRequest,
};

use crate::services::common::replica_sync::{
    ReplicaSyncConfig, ScopedReplicaEvent, ScopedSequencePublisher, setup_scoped_replica_sync,
};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WorkerInfo {
    pub worker_id: WorkerId,
    pub model_name: String,
    pub routing_group: String,
    pub block_size: u32,
    pub dp_start: u32,
    pub dp_size: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ActiveLoadInfo {
    pub model_name: String,
    pub routing_group: String,
    pub worker_id: WorkerId,
    pub dp_rank: u32,
    pub active_prefill_tokens: usize,
    pub active_decode_blocks: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("block_size must be greater than 0")]
    InvalidBlockSize,

    #[error("dp_size must be greater than 0")]
    InvalidDpSize,

    #[error("dp range overflows u32: start={dp_start} size={dp_size}")]
    InvalidDpRange { dp_start: u32, dp_size: u32 },

    #[error(
        "block_size mismatch for model={model_name} routing_group={routing_group}: existing={existing}, requested={requested}"
    )]
    BlockSizeMismatch {
        model_name: String,
        routing_group: String,
        existing: u32,
        requested: u32,
    },

    #[error(
        "worker {worker_id} already registered for model={model_name} routing_group={routing_group}"
    )]
    DuplicateWorker {
        worker_id: WorkerId,
        model_name: String,
        routing_group: String,
    },

    #[error("worker {worker_id} not found for model={model_name} routing_group={routing_group}")]
    WorkerNotFound {
        worker_id: WorkerId,
        model_name: String,
        routing_group: String,
    },

    #[error("no slot tracker for model={model_name} routing_group={routing_group}")]
    TrackerNotFound {
        model_name: String,
        routing_group: String,
    },
}

struct TrackerEntry {
    tracker: Arc<ActiveSequencesMultiWorker<ScopedSequencePublisher>>,
    pub block_size: u32,
    lifecycle_lock: Mutex<()>,
    replica_tx: Option<mpsc::Sender<crate::protocols::ActiveSequenceEvent>>,
    cancel_token: CancellationToken,
}

impl TrackerEntry {
    fn new(
        key: &RoutingPartitionId,
        block_size: u32,
        root_cancel_token: &CancellationToken,
        replica_config: Option<&ReplicaSyncConfig>,
    ) -> Arc<Self> {
        let cancel_token = root_cancel_token.child_token();
        let scoped_replica_sync = setup_scoped_replica_sync(replica_config, key, block_size);
        let tracker = Arc::new(ActiveSequencesMultiWorker::new_with_replica_worker_policy(
            scoped_replica_sync.publisher,
            block_size as usize,
            Default::default(),
            scoped_replica_sync.enabled,
            scoped_replica_sync.process_id,
            "standalone",
            ReplicaWorkerPolicy::RequireRegistered,
        ));
        let replica_tx = scoped_replica_sync.channel.map(|(replica_tx, subscriber)| {
            tracker.start_replica_sync(subscriber, cancel_token.clone());
            replica_tx
        });
        tracker.start_periodic_force_expiry_across_all_workers(cancel_token.clone());
        Arc::new(Self {
            tracker,
            block_size,
            lifecycle_lock: Mutex::new(()),
            replica_tx,
            cancel_token,
        })
    }
}

pub struct SlotTrackerRegistry {
    trackers: DashMap<RoutingPartitionId, Arc<TrackerEntry>>,
    root_cancel_token: CancellationToken,
    replica_config: Option<ReplicaSyncConfig>,
}

impl SlotTrackerRegistry {
    pub fn new(root_cancel_token: CancellationToken) -> Self {
        Self {
            trackers: DashMap::new(),
            root_cancel_token,
            replica_config: None,
        }
    }

    pub(crate) fn new_with_replica_sync(
        root_cancel_token: CancellationToken,
        replica_config: ReplicaSyncConfig,
    ) -> Self {
        Self {
            trackers: DashMap::new(),
            root_cancel_token,
            replica_config: Some(replica_config),
        }
    }

    pub fn register(
        &self,
        key: RoutingPartitionId,
        worker_id: WorkerId,
        block_size: u32,
        dp_start: u32,
        dp_size: u32,
    ) -> Result<(), RegistryError> {
        validate_block_size(block_size)?;
        let range = WorkerDpRange::new(worker_id, dp_start, dp_size)
            .validate()
            .map_err(|error| topology_error(&key, error))?;

        loop {
            let entry = self
                .trackers
                .entry(key.clone())
                .or_insert_with(|| {
                    TrackerEntry::new(
                        &key,
                        block_size,
                        &self.root_cancel_token,
                        self.replica_config.as_ref(),
                    )
                })
                .clone();

            let _lifecycle = entry.lifecycle_lock.lock();
            if !self.is_attached(&key, &entry) {
                continue;
            }
            if entry.block_size != block_size {
                return Err(RegistryError::BlockSizeMismatch {
                    model_name: key.model_name,
                    routing_group: key.routing_group,
                    existing: entry.block_size,
                    requested: block_size,
                });
            }

            entry
                .tracker
                .register_worker(range)
                .map_err(|error| topology_error(&key, error))?;
            return Ok(());
        }
    }

    pub fn unregister(
        &self,
        key: &RoutingPartitionId,
        worker_id: WorkerId,
    ) -> Result<(), RegistryError> {
        loop {
            let Some(entry) = self
                .trackers
                .get(key)
                .map(|entry| Arc::clone(entry.value()))
            else {
                return Err(RegistryError::WorkerNotFound {
                    worker_id,
                    model_name: key.model_name.clone(),
                    routing_group: key.routing_group.clone(),
                });
            };

            let _lifecycle = entry.lifecycle_lock.lock();
            if !self.is_attached(key, &entry) {
                continue;
            }

            entry
                .tracker
                .unregister_worker(worker_id)
                .map_err(|error| topology_error(key, error))?;
            if !entry.tracker.has_registered_workers()
                && self
                    .trackers
                    .remove_if(key, |_, current| Arc::ptr_eq(current, &entry))
                    .is_some()
            {
                entry.cancel_token.cancel();
            }
            return Ok(());
        }
    }

    pub fn list_workers(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<WorkerInfo> {
        let mut workers = Vec::new();
        for entry in &self.trackers {
            let key = entry.key();
            if !matches_filters(key, model_name, routing_group) {
                continue;
            }
            for range in entry.value().tracker.worker_ranges() {
                workers.push(WorkerInfo {
                    worker_id: range.worker_id,
                    model_name: key.model_name.clone(),
                    routing_group: key.routing_group.clone(),
                    block_size: entry.value().block_size,
                    dp_start: range.dp_start,
                    dp_size: range.dp_size,
                });
            }
        }
        workers.sort_by(|a, b| {
            (&a.model_name, &a.routing_group, a.worker_id).cmp(&(
                &b.model_name,
                &b.routing_group,
                b.worker_id,
            ))
        });
        workers
    }

    pub fn add_request(
        &self,
        key: &RoutingPartitionId,
        request_id: String,
        worker: WorkerWithDpRank,
        sequence_hashes: Vec<SequenceHash>,
        new_isl_tokens: usize,
    ) -> Result<(), ServiceError> {
        let entry = self.entry(key)?;
        let prefill_load_hint = (new_isl_tokens > 0).then_some(PrefillLoadHint {
            initial_effective_prefill_tokens: new_isl_tokens,
            expected_prefill_duration: None,
        });
        entry.tracker.add_request_if_registered(
            SequenceRequest {
                request_id,
                token_sequence: Some(sequence_hashes),
                track_prefill_tokens: prefill_load_hint.is_some(),
                expected_output_tokens: None,
                prefill_load_hint,
                worker,
                lora_name: None,
            },
            Instant::now(),
        )?;
        Ok(())
    }

    pub fn mark_prefill_completed(
        &self,
        key: &RoutingPartitionId,
        request_id: &str,
    ) -> Result<(), ServiceError> {
        let entry = self.entry(key)?;
        entry
            .tracker
            .mark_prefill_completed(&request_id.to_string(), Instant::now())?;
        Ok(())
    }

    pub fn free(&self, key: &RoutingPartitionId, request_id: &str) -> Result<(), ServiceError> {
        let entry = self.entry(key)?;
        entry
            .tracker
            .free(&request_id.to_string(), Instant::now())?;
        Ok(())
    }

    pub fn list_loads(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<ActiveLoadInfo> {
        let mut loads = Vec::new();
        for entry in &self.trackers {
            let key = entry.key();
            if !matches_filters(key, model_name, routing_group) {
                continue;
            }
            let (decode_blocks, prefill_tokens, _) = entry
                .value()
                .tracker
                .potential_blocks_and_tokens::<false>(None, &PrefillTokenDeltas::none());
            let mut workers: FxHashSet<_> = decode_blocks.keys().copied().collect();
            workers.extend(prefill_tokens.keys().copied());
            for worker in workers {
                loads.push(ActiveLoadInfo {
                    model_name: key.model_name.clone(),
                    routing_group: key.routing_group.clone(),
                    worker_id: worker.worker_id,
                    dp_rank: worker.dp_rank,
                    active_prefill_tokens: prefill_tokens.get(&worker).copied().unwrap_or(0),
                    active_decode_blocks: decode_blocks.get(&worker).copied().unwrap_or(0),
                });
            }
        }
        loads.sort_by(|a, b| {
            (&a.model_name, &a.routing_group, a.worker_id, a.dp_rank).cmp(&(
                &b.model_name,
                &b.routing_group,
                b.worker_id,
                b.dp_rank,
            ))
        });
        loads
    }

    pub fn potential_loads(
        &self,
        key: &RoutingPartitionId,
        sequence_hashes: &[SequenceHash],
        new_isl_tokens: usize,
    ) -> Result<Vec<PotentialLoad>, RegistryError> {
        let entry = self.entry(key)?;
        let (decode_blocks, prefill_tokens, active_requests) =
            entry.tracker.potential_blocks_and_tokens::<true>(
                Some(sequence_hashes),
                &PrefillTokenDeltas::uniform(new_isl_tokens),
            );
        let active_requests = active_requests.expect("active request projection should be present");
        Ok(decode_blocks
            .into_iter()
            .map(|(worker, potential_decode_blocks)| PotentialLoad {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                potential_prefill_tokens: prefill_tokens.get(&worker).copied().unwrap_or(0),
                potential_decode_blocks,
                active_requests: active_requests.get(&worker).copied().unwrap_or(0),
            })
            .collect())
    }

    pub(crate) fn dispatch_replica_event(&self, envelope: ScopedReplicaEvent) {
        let (key, block_size, event) = envelope.into_parts();
        if self
            .replica_config
            .as_ref()
            .is_some_and(|config| config.is_self_event(&event))
        {
            return;
        }

        let Some(entry) = self
            .trackers
            .get(&key)
            .map(|entry| Arc::clone(entry.value()))
        else {
            tracing::trace!(
                model_name = %key.model_name,
                routing_group = %key.routing_group,
                "Dropping replica event for unknown slot tracker"
            );
            return;
        };
        if entry.block_size != block_size {
            tracing::debug!(
                model_name = %key.model_name,
                routing_group = %key.routing_group,
                expected_block_size = entry.block_size,
                received_block_size = block_size,
                "Dropping replica event with mismatched block size"
            );
            return;
        }
        let Some(replica_tx) = &entry.replica_tx else {
            return;
        };
        match replica_tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::trace!(
                    model_name = %key.model_name,
                    routing_group = %key.routing_group,
                    request_id = %event.request_id,
                    "Replica subscriber channel full; dropping event"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!(
                    model_name = %key.model_name,
                    routing_group = %key.routing_group,
                    "Replica subscriber channel closed; dropping event"
                );
            }
        }
    }

    fn entry(&self, key: &RoutingPartitionId) -> Result<Arc<TrackerEntry>, RegistryError> {
        self.trackers
            .get(key)
            .map(|entry| Arc::clone(entry.value()))
            .ok_or_else(|| RegistryError::TrackerNotFound {
                model_name: key.model_name.clone(),
                routing_group: key.routing_group.clone(),
            })
    }

    fn is_attached(&self, key: &RoutingPartitionId, entry: &Arc<TrackerEntry>) -> bool {
        self.trackers
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current.value(), entry))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error(transparent)]
    Registry(#[from] RegistryError),

    #[error(transparent)]
    Sequence(#[from] SequenceError),
}

fn validate_block_size(block_size: u32) -> Result<(), RegistryError> {
    if block_size == 0 {
        return Err(RegistryError::InvalidBlockSize);
    }
    Ok(())
}

fn topology_error(key: &RoutingPartitionId, error: WorkerTopologyError) -> RegistryError {
    match error {
        WorkerTopologyError::InvalidDpSize { .. } => RegistryError::InvalidDpSize,
        WorkerTopologyError::InvalidDpRange {
            dp_start, dp_size, ..
        } => RegistryError::InvalidDpRange { dp_start, dp_size },
        WorkerTopologyError::DuplicateWorker { worker_id } => RegistryError::DuplicateWorker {
            worker_id,
            model_name: key.model_name.clone(),
            routing_group: key.routing_group.clone(),
        },
        WorkerTopologyError::WorkerNotFound { worker_id } => RegistryError::WorkerNotFound {
            worker_id,
            model_name: key.model_name.clone(),
            routing_group: key.routing_group.clone(),
        },
    }
}

fn matches_filters(
    key: &RoutingPartitionId,
    model_name: Option<&str>,
    routing_group: Option<&str>,
) -> bool {
    model_name.is_none_or(|model_name| key.model_name == model_name)
        && routing_group.is_none_or(|routing_group| key.routing_group == routing_group)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{ActiveSequenceEvent, ActiveSequenceEventData};

    fn registry() -> SlotTrackerRegistry {
        SlotTrackerRegistry::new(CancellationToken::new())
    }

    fn key(routing_group: &str) -> RoutingPartitionId {
        RoutingPartitionId::new("model", routing_group)
    }

    fn replica_event(
        routing_group: &str,
        block_size: u32,
        worker: WorkerWithDpRank,
        router_id: u64,
    ) -> ScopedReplicaEvent {
        ScopedReplicaEvent {
            model_name: "model".to_string(),
            routing_group: routing_group.to_string(),
            block_size,
            event: ActiveSequenceEvent {
                request_id: "replica-request".to_string(),
                worker,
                data: ActiveSequenceEventData::AddRequest {
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                },
                router_id,
                lora_name: None,
            },
        }
    }

    #[tokio::test]
    async fn register_validates_ranges() {
        let registry = registry();
        assert!(matches!(
            registry.register(key("default"), 1, 0, 0, 1),
            Err(RegistryError::InvalidBlockSize)
        ));
        assert!(matches!(
            registry.register(key("default"), 1, 16, 0, 0),
            Err(RegistryError::InvalidDpSize)
        ));
        assert!(matches!(
            registry.register(key("default"), 1, 16, u32::MAX, 1),
            Err(RegistryError::InvalidDpRange { .. })
        ));
    }

    #[tokio::test]
    async fn register_expands_dp_range_and_enforces_tracker_invariants() {
        let registry = registry();
        let key = key("default");
        registry.register(key.clone(), 1, 16, 2, 2).unwrap();

        assert_eq!(
            registry.list_loads(None, None),
            vec![
                ActiveLoadInfo {
                    model_name: "model".to_string(),
                    routing_group: "default".to_string(),
                    worker_id: 1,
                    dp_rank: 2,
                    active_prefill_tokens: 0,
                    active_decode_blocks: 0,
                },
                ActiveLoadInfo {
                    model_name: "model".to_string(),
                    routing_group: "default".to_string(),
                    worker_id: 1,
                    dp_rank: 3,
                    active_prefill_tokens: 0,
                    active_decode_blocks: 0,
                },
            ]
        );
        assert!(matches!(
            registry.register(key.clone(), 1, 16, 0, 1),
            Err(RegistryError::DuplicateWorker { .. })
        ));
        assert!(matches!(
            registry.register(key, 2, 32, 0, 1),
            Err(RegistryError::BlockSizeMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn unregister_busy_worker_removes_tracker_and_readd_starts_empty() {
        let registry = registry();
        let key = key("default");
        registry.register(key.clone(), 1, 16, 0, 1).unwrap();
        registry
            .add_request(
                &key,
                "req-1".to_string(),
                WorkerWithDpRank::new(1, 0),
                vec![1, 2],
                8,
            )
            .unwrap();

        registry.unregister(&key, 1).unwrap();
        assert!(registry.list_workers(None, None).is_empty());

        registry.register(key.clone(), 1, 16, 0, 1).unwrap();
        assert_eq!(
            registry.list_loads(None, None),
            vec![ActiveLoadInfo {
                model_name: "model".to_string(),
                routing_group: "default".to_string(),
                worker_id: 1,
                dp_rank: 0,
                active_prefill_tokens: 0,
                active_decode_blocks: 0,
            }]
        );
    }

    #[tokio::test]
    async fn concurrent_last_worker_removal_and_registration_keep_fresh_entry() {
        let registry = Arc::new(registry());
        let key = key("default");

        for _ in 0..64 {
            registry.register(key.clone(), 1, 16, 0, 1).unwrap();
            let barrier = Arc::new(tokio::sync::Barrier::new(3));

            let remove_registry = Arc::clone(&registry);
            let remove_key = key.clone();
            let remove_barrier = Arc::clone(&barrier);
            let remove = tokio::spawn(async move {
                remove_barrier.wait().await;
                remove_registry.unregister(&remove_key, 1)
            });

            let add_registry = Arc::clone(&registry);
            let add_key = key.clone();
            let add_barrier = Arc::clone(&barrier);
            let add = tokio::spawn(async move {
                add_barrier.wait().await;
                add_registry.register(add_key, 2, 16, 0, 1)
            });

            barrier.wait().await;
            remove.await.unwrap().unwrap();
            add.await.unwrap().unwrap();

            let workers = registry.list_workers(None, None);
            assert_eq!(workers.len(), 1);
            assert_eq!(workers[0].worker_id, 2);
            registry.unregister(&key, 2).unwrap();
        }
    }

    #[tokio::test]
    async fn replica_dispatch_rejects_self_and_requires_matching_registered_worker() {
        let (outbound_tx, _outbound_rx) = mpsc::channel(1);
        let cancel_token = CancellationToken::new();
        let registry = SlotTrackerRegistry::new_with_replica_sync(
            cancel_token.clone(),
            ReplicaSyncConfig::new(7, outbound_tx, cancel_token),
        );
        let key = key("default");
        registry.register(key.clone(), 1, 16, 0, 1).unwrap();

        registry.dispatch_replica_event(replica_event(
            "default",
            16,
            WorkerWithDpRank::new(1, 1),
            8,
        ));
        registry.dispatch_replica_event(replica_event(
            "default",
            32,
            WorkerWithDpRank::new(1, 0),
            8,
        ));
        registry.dispatch_replica_event(replica_event(
            "default",
            16,
            WorkerWithDpRank::new(1, 0),
            7,
        ));
        registry.dispatch_replica_event(replica_event("other", 16, WorkerWithDpRank::new(1, 0), 8));
        tokio::task::yield_now().await;
        assert_eq!(registry.list_loads(None, None)[0].active_decode_blocks, 0);

        registry.dispatch_replica_event(replica_event(
            "default",
            16,
            WorkerWithDpRank::new(1, 0),
            8,
        ));
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if registry.list_loads(None, None)[0].active_decode_blocks == 3 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn worker_ids_are_scoped_by_model_and_routing_group() {
        let registry = registry();
        registry.register(key("a"), 1, 16, 0, 1).unwrap();
        registry.register(key("b"), 1, 16, 0, 1).unwrap();
        assert_eq!(registry.list_workers(None, None).len(), 2);
    }

    #[tokio::test]
    async fn lifecycle_preserves_arrival_order_and_strict_admission() {
        let registry = registry();
        let key = key("default");
        registry.register(key.clone(), 1, 16, 0, 1).unwrap();

        registry.free(&key, "early-free").unwrap();
        assert!(matches!(
            registry.mark_prefill_completed(&key, "early-complete"),
            Err(ServiceError::Sequence(
                SequenceError::RequestNotFound { .. }
            ))
        ));
        assert!(matches!(
            registry.add_request(
                &key,
                "bad-rank".to_string(),
                WorkerWithDpRank::new(1, 1),
                vec![1],
                0,
            ),
            Err(ServiceError::Sequence(SequenceError::WorkerNotFound { .. }))
        ));

        registry
            .add_request(
                &key,
                "early-free".to_string(),
                WorkerWithDpRank::new(1, 0),
                vec![1, 2],
                8,
            )
            .unwrap();
        assert!(matches!(
            registry.add_request(
                &key,
                "early-free".to_string(),
                WorkerWithDpRank::new(1, 0),
                vec![1, 2],
                8,
            ),
            Err(ServiceError::Sequence(
                SequenceError::DuplicateRequest { .. }
            ))
        ));
        registry.mark_prefill_completed(&key, "early-free").unwrap();
        registry.mark_prefill_completed(&key, "early-free").unwrap();
        registry.free(&key, "early-free").unwrap();
        registry.free(&key, "early-free").unwrap();

        registry
            .add_request(
                &key,
                "early-complete".to_string(),
                WorkerWithDpRank::new(1, 0),
                vec![3],
                8,
            )
            .unwrap();
        assert_eq!(registry.list_loads(None, None)[0].active_prefill_tokens, 8);
        registry.free(&key, "early-complete").unwrap();

        assert_eq!(registry.list_loads(None, None)[0].active_decode_blocks, 0);
    }

    #[tokio::test]
    async fn potential_loads_reuse_active_hashes_and_add_uniform_prefill() {
        let registry = registry();
        let key = key("default");
        registry.register(key.clone(), 1, 16, 0, 1).unwrap();
        registry
            .add_request(
                &key,
                "req-1".to_string(),
                WorkerWithDpRank::new(1, 0),
                vec![1, 2, 3],
                8,
            )
            .unwrap();

        let loads = registry.potential_loads(&key, &[1, 2, 4], 5).unwrap();
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0].worker_id, 1);
        assert_eq!(loads[0].dp_rank, 0);
        assert_eq!(loads[0].potential_prefill_tokens, 13);
        assert_eq!(loads[0].potential_decode_blocks, 4);
        assert_eq!(loads[0].active_requests, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn periodic_expiry_clears_requests_older_than_300_seconds() {
        let registry = registry();
        let key = key("default");
        registry.register(key.clone(), 1, 16, 0, 1).unwrap();
        registry
            .add_request(
                &key,
                "req-1".to_string(),
                WorkerWithDpRank::new(1, 0),
                vec![1, 2, 3],
                8,
            )
            .unwrap();

        tokio::time::advance(std::time::Duration::from_secs(331)).await;
        tokio::task::yield_now().await;

        assert_eq!(registry.list_loads(None, None)[0].active_decode_blocks, 0);
    }
}

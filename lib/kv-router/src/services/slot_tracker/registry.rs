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

use crate::protocols::{PrefillLoadHint, WorkerId, WorkerWithDpRank};
use crate::scheduling::PotentialLoad;
use crate::sequences::topology::{WorkerDpRange, WorkerTopologyError};
use crate::sequences::{
    ActiveSequencesMultiWorker, PrefillTokenDeltas, ReplicaWorkerPolicy, SequenceError,
    SequenceRequest,
};

use super::replica_sync::{
    ChannelSequenceSubscriber, REPLICA_EVENT_CHANNEL_CAPACITY, ReplicaEventSender,
    ScopedSequencePublisher, SlotReplicaEvent,
};

fn default_tenant() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TrackerKey {
    pub model_name: String,
    pub tenant_id: String,
}

impl TrackerKey {
    pub fn new(model_name: String, tenant_id: Option<String>) -> Self {
        Self {
            model_name,
            tenant_id: tenant_id.unwrap_or_else(default_tenant),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WorkerInfo {
    pub worker_id: WorkerId,
    pub model_name: String,
    pub tenant_id: String,
    pub block_size: u32,
    pub dp_start: u32,
    pub dp_size: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ActiveLoadInfo {
    pub model_name: String,
    pub tenant_id: String,
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
        "block_size mismatch for model={model_name} tenant={tenant_id}: existing={existing}, requested={requested}"
    )]
    BlockSizeMismatch {
        model_name: String,
        tenant_id: String,
        existing: u32,
        requested: u32,
    },

    #[error("worker {worker_id} already registered for model={model_name} tenant={tenant_id}")]
    DuplicateWorker {
        worker_id: WorkerId,
        model_name: String,
        tenant_id: String,
    },

    #[error("worker {worker_id} not found for model={model_name} tenant={tenant_id}")]
    WorkerNotFound {
        worker_id: WorkerId,
        model_name: String,
        tenant_id: String,
    },

    #[error("no slot tracker for model={model_name} tenant={tenant_id}")]
    TrackerNotFound {
        model_name: String,
        tenant_id: String,
    },
}

#[derive(Clone)]
struct RegistryReplicaConfig {
    process_id: u64,
    outbound_tx: ReplicaEventSender,
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
        key: &TrackerKey,
        block_size: u32,
        root_cancel_token: &CancellationToken,
        replica_config: Option<&RegistryReplicaConfig>,
    ) -> Arc<Self> {
        let cancel_token = root_cancel_token.child_token();
        let (publisher, replica_sync, router_id, replica_channel) =
            if let Some(replica_config) = replica_config {
                let (replica_tx, replica_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
                (
                    ScopedSequencePublisher::enabled(
                        Arc::from(key.model_name.as_str()),
                        Arc::from(key.tenant_id.as_str()),
                        block_size,
                        replica_config.outbound_tx.clone(),
                    ),
                    true,
                    replica_config.process_id,
                    Some((replica_tx, replica_rx)),
                )
            } else {
                (ScopedSequencePublisher::disabled(), false, 0, None)
            };
        let tracker = Arc::new(ActiveSequencesMultiWorker::new_with_replica_worker_policy(
            publisher,
            block_size as usize,
            Default::default(),
            replica_sync,
            router_id,
            "standalone",
            ReplicaWorkerPolicy::RequireRegistered,
        ));
        let replica_tx = replica_channel.map(|(replica_tx, replica_rx)| {
            tracker.start_replica_sync(
                ChannelSequenceSubscriber::new(replica_rx),
                cancel_token.clone(),
            );
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
    trackers: DashMap<TrackerKey, Arc<TrackerEntry>>,
    root_cancel_token: CancellationToken,
    replica_config: Option<RegistryReplicaConfig>,
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
        process_id: u64,
        outbound_tx: ReplicaEventSender,
    ) -> Self {
        Self {
            trackers: DashMap::new(),
            root_cancel_token,
            replica_config: Some(RegistryReplicaConfig {
                process_id,
                outbound_tx,
            }),
        }
    }

    pub fn register(
        &self,
        key: TrackerKey,
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
                    tenant_id: key.tenant_id,
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

    pub fn unregister(&self, key: &TrackerKey, worker_id: WorkerId) -> Result<(), RegistryError> {
        loop {
            let Some(entry) = self
                .trackers
                .get(key)
                .map(|entry| Arc::clone(entry.value()))
            else {
                return Err(RegistryError::WorkerNotFound {
                    worker_id,
                    model_name: key.model_name.clone(),
                    tenant_id: key.tenant_id.clone(),
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
        tenant_id: Option<&str>,
    ) -> Vec<WorkerInfo> {
        let mut workers = Vec::new();
        for entry in &self.trackers {
            let key = entry.key();
            if !matches_filters(key, model_name, tenant_id) {
                continue;
            }
            for range in entry.value().tracker.worker_ranges() {
                workers.push(WorkerInfo {
                    worker_id: range.worker_id,
                    model_name: key.model_name.clone(),
                    tenant_id: key.tenant_id.clone(),
                    block_size: entry.value().block_size,
                    dp_start: range.dp_start,
                    dp_size: range.dp_size,
                });
            }
        }
        workers.sort_by(|a, b| {
            (&a.model_name, &a.tenant_id, a.worker_id).cmp(&(
                &b.model_name,
                &b.tenant_id,
                b.worker_id,
            ))
        });
        workers
    }

    pub fn add_request(
        &self,
        key: &TrackerKey,
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
        key: &TrackerKey,
        request_id: &str,
    ) -> Result<(), ServiceError> {
        let entry = self.entry(key)?;
        entry
            .tracker
            .mark_prefill_completed(&request_id.to_string(), Instant::now())?;
        Ok(())
    }

    pub fn free(&self, key: &TrackerKey, request_id: &str) -> Result<(), ServiceError> {
        let entry = self.entry(key)?;
        entry
            .tracker
            .free(&request_id.to_string(), Instant::now())?;
        Ok(())
    }

    pub fn list_loads(
        &self,
        model_name: Option<&str>,
        tenant_id: Option<&str>,
    ) -> Vec<ActiveLoadInfo> {
        let mut loads = Vec::new();
        for entry in &self.trackers {
            let key = entry.key();
            if !matches_filters(key, model_name, tenant_id) {
                continue;
            }
            let (decode_blocks, prefill_tokens) = entry
                .value()
                .tracker
                .potential_blocks_and_tokens(None, &PrefillTokenDeltas::none());
            let mut workers: FxHashSet<_> = decode_blocks.keys().copied().collect();
            workers.extend(prefill_tokens.keys().copied());
            for worker in workers {
                loads.push(ActiveLoadInfo {
                    model_name: key.model_name.clone(),
                    tenant_id: key.tenant_id.clone(),
                    worker_id: worker.worker_id,
                    dp_rank: worker.dp_rank,
                    active_prefill_tokens: prefill_tokens.get(&worker).copied().unwrap_or(0),
                    active_decode_blocks: decode_blocks.get(&worker).copied().unwrap_or(0),
                });
            }
        }
        loads.sort_by(|a, b| {
            (&a.model_name, &a.tenant_id, a.worker_id, a.dp_rank).cmp(&(
                &b.model_name,
                &b.tenant_id,
                b.worker_id,
                b.dp_rank,
            ))
        });
        loads
    }

    pub fn potential_loads(
        &self,
        key: &TrackerKey,
        sequence_hashes: &[SequenceHash],
        new_isl_tokens: usize,
    ) -> Result<Vec<PotentialLoad>, RegistryError> {
        let entry = self.entry(key)?;
        let (decode_blocks, prefill_tokens) = entry.tracker.potential_blocks_and_tokens(
            Some(sequence_hashes),
            &PrefillTokenDeltas::uniform(new_isl_tokens),
        );
        let mut workers: FxHashSet<_> = decode_blocks.keys().copied().collect();
        workers.extend(prefill_tokens.keys().copied());
        Ok(workers
            .into_iter()
            .map(|worker| PotentialLoad {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                potential_prefill_tokens: prefill_tokens.get(&worker).copied().unwrap_or(0),
                potential_decode_blocks: decode_blocks.get(&worker).copied().unwrap_or(0),
            })
            .collect())
    }

    pub(crate) fn dispatch_replica_event(&self, envelope: SlotReplicaEvent) {
        if self
            .replica_config
            .as_ref()
            .is_some_and(|config| envelope.event.router_id == config.process_id)
        {
            return;
        }

        let key = TrackerKey::new(envelope.model_name, Some(envelope.tenant_id));
        let Some(entry) = self
            .trackers
            .get(&key)
            .map(|entry| Arc::clone(entry.value()))
        else {
            tracing::trace!(
                model_name = %key.model_name,
                tenant_id = %key.tenant_id,
                "Dropping replica event for unknown slot tracker"
            );
            return;
        };
        if entry.block_size != envelope.block_size {
            tracing::debug!(
                model_name = %key.model_name,
                tenant_id = %key.tenant_id,
                expected_block_size = entry.block_size,
                received_block_size = envelope.block_size,
                "Dropping replica event with mismatched block size"
            );
            return;
        }
        let Some(replica_tx) = &entry.replica_tx else {
            return;
        };
        match replica_tx.try_send(envelope.event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::trace!(
                    model_name = %key.model_name,
                    tenant_id = %key.tenant_id,
                    request_id = %event.request_id,
                    "Replica subscriber channel full; dropping event"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!(
                    model_name = %key.model_name,
                    tenant_id = %key.tenant_id,
                    "Replica subscriber channel closed; dropping event"
                );
            }
        }
    }

    fn entry(&self, key: &TrackerKey) -> Result<Arc<TrackerEntry>, RegistryError> {
        self.trackers
            .get(key)
            .map(|entry| Arc::clone(entry.value()))
            .ok_or_else(|| RegistryError::TrackerNotFound {
                model_name: key.model_name.clone(),
                tenant_id: key.tenant_id.clone(),
            })
    }

    fn is_attached(&self, key: &TrackerKey, entry: &Arc<TrackerEntry>) -> bool {
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

fn topology_error(key: &TrackerKey, error: WorkerTopologyError) -> RegistryError {
    match error {
        WorkerTopologyError::InvalidDpSize { .. } => RegistryError::InvalidDpSize,
        WorkerTopologyError::InvalidDpRange {
            dp_start, dp_size, ..
        } => RegistryError::InvalidDpRange { dp_start, dp_size },
        WorkerTopologyError::DuplicateWorker { worker_id } => RegistryError::DuplicateWorker {
            worker_id,
            model_name: key.model_name.clone(),
            tenant_id: key.tenant_id.clone(),
        },
        WorkerTopologyError::WorkerNotFound { worker_id } => RegistryError::WorkerNotFound {
            worker_id,
            model_name: key.model_name.clone(),
            tenant_id: key.tenant_id.clone(),
        },
    }
}

fn matches_filters(key: &TrackerKey, model_name: Option<&str>, tenant_id: Option<&str>) -> bool {
    model_name.is_none_or(|model_name| key.model_name == model_name)
        && tenant_id.is_none_or(|tenant_id| key.tenant_id == tenant_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{ActiveSequenceEvent, ActiveSequenceEventData};

    fn registry() -> SlotTrackerRegistry {
        SlotTrackerRegistry::new(CancellationToken::new())
    }

    fn key(tenant_id: &str) -> TrackerKey {
        TrackerKey::new("model".to_string(), Some(tenant_id.to_string()))
    }

    fn replica_event(
        tenant_id: &str,
        block_size: u32,
        worker: WorkerWithDpRank,
        router_id: u64,
    ) -> SlotReplicaEvent {
        SlotReplicaEvent {
            model_name: "model".to_string(),
            tenant_id: tenant_id.to_string(),
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
                    tenant_id: "default".to_string(),
                    worker_id: 1,
                    dp_rank: 2,
                    active_prefill_tokens: 0,
                    active_decode_blocks: 0,
                },
                ActiveLoadInfo {
                    model_name: "model".to_string(),
                    tenant_id: "default".to_string(),
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
                tenant_id: "default".to_string(),
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
    async fn replica_dispatch_requires_registered_worker_and_matching_configuration() {
        let (outbound_tx, _outbound_rx) = mpsc::channel(1);
        let registry =
            SlotTrackerRegistry::new_with_replica_sync(CancellationToken::new(), 7, outbound_tx);
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
    async fn recreated_tracker_uses_a_fresh_replica_channel() {
        let (outbound_tx, _outbound_rx) = mpsc::channel(1);
        let registry =
            SlotTrackerRegistry::new_with_replica_sync(CancellationToken::new(), 7, outbound_tx);
        let key = key("default");
        registry.register(key.clone(), 1, 16, 0, 1).unwrap();
        let first = registry.entry(&key).unwrap();

        registry.unregister(&key, 1).unwrap();
        registry.register(key.clone(), 1, 16, 0, 1).unwrap();
        let second = registry.entry(&key).unwrap();

        assert!(!Arc::ptr_eq(&first, &second));
        assert!(
            !first
                .replica_tx
                .as_ref()
                .unwrap()
                .same_channel(second.replica_tx.as_ref().unwrap())
        );
    }

    #[tokio::test]
    async fn worker_ids_are_scoped_by_model_and_tenant() {
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

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::{self, Future};
use std::sync::Arc;

use dashmap::DashMap;
use dynamo_tokens::SequenceHash;
use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use serde::Serialize;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::protocols::{
    ActiveLoad, ActiveSequenceEvent, PrefillLoadHint, WorkerId, WorkerWithDpRank,
};
use crate::scheduling::PotentialLoad;
use crate::sequences::{
    ActiveSequencesMultiWorker, PrefillTokenDeltas, SequenceError, SequencePublisher,
    SequenceRequest,
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

pub struct StandaloneSequencePublisher;

impl SequencePublisher for StandaloneSequencePublisher {
    fn publish_event(
        &self,
        _event: &ActiveSequenceEvent,
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        future::ready(Ok(()))
    }

    fn publish_load(&self, _load: ActiveLoad) {}

    fn observe_load(
        &self,
        _worker: &WorkerWithDpRank,
        _worker_type: &str,
        _blocks: usize,
        _tokens: usize,
    ) {
    }
}

pub struct TrackerEntry {
    pub tracker: Arc<ActiveSequencesMultiWorker<StandaloneSequencePublisher>>,
    pub block_size: u32,
    worker_ranges: Mutex<HashMap<WorkerId, (u32, u32)>>,
    cancel_token: CancellationToken,
}

impl TrackerEntry {
    fn new(block_size: u32, root_cancel_token: &CancellationToken) -> Arc<Self> {
        let cancel_token = root_cancel_token.child_token();
        let tracker = Arc::new(ActiveSequencesMultiWorker::new(
            StandaloneSequencePublisher,
            block_size as usize,
            HashMap::new(),
            false,
            0,
            "standalone",
        ));
        tracker.start_periodic_force_expiry_across_all_workers(cancel_token.clone());
        Arc::new(Self {
            tracker,
            block_size,
            worker_ranges: Mutex::new(HashMap::new()),
            cancel_token,
        })
    }
}

pub struct SlotTrackerRegistry {
    trackers: DashMap<TrackerKey, Arc<TrackerEntry>>,
    root_cancel_token: CancellationToken,
}

impl SlotTrackerRegistry {
    pub fn new(root_cancel_token: CancellationToken) -> Self {
        Self {
            trackers: DashMap::new(),
            root_cancel_token,
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
        validate_registration(block_size, dp_start, dp_size)?;

        loop {
            let entry = self
                .trackers
                .entry(key.clone())
                .or_insert_with(|| TrackerEntry::new(block_size, &self.root_cancel_token))
                .clone();

            if entry.block_size != block_size {
                return Err(RegistryError::BlockSizeMismatch {
                    model_name: key.model_name,
                    tenant_id: key.tenant_id,
                    existing: entry.block_size,
                    requested: block_size,
                });
            }

            let mut worker_ranges = entry.worker_ranges.lock();
            if !self.is_attached(&key, &entry) {
                continue;
            }
            if worker_ranges.contains_key(&worker_id) {
                return Err(RegistryError::DuplicateWorker {
                    worker_id,
                    model_name: key.model_name,
                    tenant_id: key.tenant_id,
                });
            }

            worker_ranges.insert(worker_id, (dp_start, dp_size));
            entry.tracker.update_workers(&worker_ranges);
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

            let mut worker_ranges = entry.worker_ranges.lock();
            if !self.is_attached(key, &entry) {
                continue;
            }
            if worker_ranges.remove(&worker_id).is_none() {
                return Err(RegistryError::WorkerNotFound {
                    worker_id,
                    model_name: key.model_name.clone(),
                    tenant_id: key.tenant_id.clone(),
                });
            }

            entry.tracker.update_workers(&worker_ranges);
            if worker_ranges.is_empty()
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
            for (&worker_id, &(dp_start, dp_size)) in entry.value().worker_ranges.lock().iter() {
                workers.push(WorkerInfo {
                    worker_id,
                    model_name: key.model_name.clone(),
                    tenant_id: key.tenant_id.clone(),
                    block_size: entry.value().block_size,
                    dp_start,
                    dp_size,
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

fn validate_registration(
    block_size: u32,
    dp_start: u32,
    dp_size: u32,
) -> Result<(), RegistryError> {
    if block_size == 0 {
        return Err(RegistryError::InvalidBlockSize);
    }
    if dp_size == 0 {
        return Err(RegistryError::InvalidDpSize);
    }
    if dp_start.checked_add(dp_size).is_none() {
        return Err(RegistryError::InvalidDpRange { dp_start, dp_size });
    }
    Ok(())
}

fn matches_filters(key: &TrackerKey, model_name: Option<&str>, tenant_id: Option<&str>) -> bool {
    model_name.is_none_or(|model_name| key.model_name == model_name)
        && tenant_id.is_none_or(|tenant_id| key.tenant_id == tenant_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> SlotTrackerRegistry {
        SlotTrackerRegistry::new(CancellationToken::new())
    }

    fn key(tenant_id: &str) -> TrackerKey {
        TrackerKey::new("model".to_string(), Some(tenant_id.to_string()))
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

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local shared G4 object-store simulation for mock KVBM offload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Result, bail};
use kvbm_engine::SequenceHash;
use kvbm_engine::offload::PendingTracker;

use super::bandwidth_sharing_model::TransferId;
use super::config::KvbmOffloadConfig;
use super::worker::{
    CompletedTransfer, DrainResult, SharedDrainCounts, TransferDirection, TransferState,
};

/// Object metadata tracked by the mock G4 tier.
#[derive(Debug, Clone, Copy)]
pub(crate) struct G4ObjectRecord {
    pub(crate) size_bytes: usize,
}

#[derive(Debug)]
struct PendingG4Put {
    keys: Vec<SequenceHash>,
    size_bytes: usize,
}

/// Shared process-local G4 resource. All mock workers in the process share
/// the object registry and the G2<->G4 processor-sharing model.
pub(crate) struct SharedG4Store {
    config: KvbmOffloadConfig,
    objects: Mutex<HashMap<SequenceHash, G4ObjectRecord>>,
    pending_puts: Mutex<HashMap<TransferId, PendingG4Put>>,
    state: Arc<Mutex<TransferState>>,
    pending_tracker: Arc<PendingTracker>,
    pending_owner_drains: Mutex<HashMap<u64, SharedDrainCounts>>,
}

static SHARED_G4_STORE: OnceLock<Mutex<Option<Weak<SharedG4Store>>>> = OnceLock::new();

#[cfg(test)]
static SHARED_G4_TEST_LOCK: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();

impl SharedG4Store {
    pub(crate) fn get_or_create(config: &KvbmOffloadConfig) -> Result<Option<Arc<Self>>> {
        if !config.enable_g4_storage {
            return Ok(None);
        }

        let store_slot = SHARED_G4_STORE.get_or_init(|| Mutex::new(None));
        let mut store_slot = store_slot
            .lock()
            .expect("shared G4 store registry poisoned");

        if let Some(existing) = store_slot.as_ref().and_then(Weak::upgrade) {
            existing.validate_config(config)?;
            return Ok(Some(existing));
        }

        *store_slot = None;
        let store = Arc::new(Self {
            config: config.clone(),
            objects: Mutex::new(HashMap::new()),
            pending_puts: Mutex::new(HashMap::new()),
            state: Arc::new(Mutex::new(TransferState::new(
                config.bandwidth_g2_to_g4_gbps,
                config.bandwidth_g4_to_g2_gbps,
            ))),
            pending_tracker: Arc::new(PendingTracker::new()),
            pending_owner_drains: Mutex::new(HashMap::new()),
        });
        *store_slot = Some(Arc::downgrade(&store));
        Ok(Some(store))
    }

    fn validate_config(&self, config: &KvbmOffloadConfig) -> Result<()> {
        if self.config.enable_g4_storage != config.enable_g4_storage
            || self.config.block_size_tokens != config.block_size_tokens
            || self.config.block_size_bytes.unwrap_or(0) != config.block_size_bytes.unwrap_or(0)
            || self.config.bandwidth_g2_to_g4_gbps != config.bandwidth_g2_to_g4_gbps
            || self.config.bandwidth_g4_to_g2_gbps != config.bandwidth_g4_to_g2_gbps
        {
            bail!("process-local shared G4 store already exists with a different shape/bandwidth");
        }
        Ok(())
    }

    pub(crate) fn transfer_state(&self) -> Arc<Mutex<TransferState>> {
        self.state.clone()
    }

    pub(crate) fn pending_tracker(&self) -> Arc<PendingTracker> {
        self.pending_tracker.clone()
    }

    pub(crate) fn has_object(&self, hash: &SequenceHash) -> Option<usize> {
        let objects = self.objects.lock().expect("shared G4 object map poisoned");
        objects.get(hash).map(|record| record.size_bytes)
    }

    pub(crate) fn insert_object(&self, hash: SequenceHash, size_bytes: usize) {
        let mut objects = self.objects.lock().expect("shared G4 object map poisoned");
        objects.insert(hash, G4ObjectRecord { size_bytes });
    }

    pub(crate) fn register_pending_put(
        &self,
        transfer_id: TransferId,
        keys: Vec<SequenceHash>,
        size_bytes: usize,
    ) {
        let mut pending = self
            .pending_puts
            .lock()
            .expect("shared G4 pending put map poisoned");
        pending.insert(transfer_id, PendingG4Put { keys, size_bytes });
    }

    #[cfg(test)]
    pub(crate) fn object_count(&self) -> usize {
        let objects = self.objects.lock().expect("shared G4 object map poisoned");
        objects.len()
    }

    pub(crate) fn drain_completions(&self, now_ms: f64, owner_id: u64) -> SharedDrainCounts {
        let mut state = self.state.lock().expect("shared G4 state poisoned");
        let drained = state.drain_completions(now_ms, "shared-g4");
        drop(state);

        self.record_drained(drained, Some(owner_id))
    }

    pub(crate) fn drain_completions_to_pending(&self, now_ms: f64) {
        let mut state = self.state.lock().expect("shared G4 state poisoned");
        let drained = state.drain_completions(now_ms, "shared-g4");
        drop(state);

        self.record_drained(drained, None);
    }

    fn record_drained(&self, drained: DrainResult, owner_id: Option<u64>) -> SharedDrainCounts {
        self.publish_completed_puts(&drained.completed);

        let mut owner_result = SharedDrainCounts::default();
        let mut pending = self
            .pending_owner_drains
            .lock()
            .expect("shared G4 owner drain map poisoned");
        if let Some(owner_id) = owner_id
            && let Some(record) = pending.remove(&owner_id)
        {
            owner_result.add_deferred_record(record);
        }
        for (owner, counts) in drained.by_owner {
            let record = SharedDrainCounts {
                counts,
                deferred_onboard_blocks: 0,
                offload_registration_baseline: None,
            };
            if Some(owner) == owner_id {
                owner_result.add_record(record);
            } else {
                pending.entry(owner).or_default().add_record(record);
            }
        }
        owner_result
    }

    fn publish_completed_puts(&self, completed: &[CompletedTransfer]) {
        let mut puts = Vec::new();
        {
            let mut pending = self
                .pending_puts
                .lock()
                .expect("shared G4 pending put map poisoned");
            for transfer in completed {
                if transfer.direction == TransferDirection::G2ToG4
                    && let Some(pending_put) = pending.remove(&transfer.id)
                {
                    puts.push(pending_put);
                }
            }
        }

        if puts.is_empty() {
            return;
        }

        let mut objects = self.objects.lock().expect("shared G4 object map poisoned");
        for pending_put in puts {
            for key in pending_put.keys {
                objects.insert(
                    key,
                    G4ObjectRecord {
                        size_bytes: pending_put.size_bytes,
                    },
                );
            }
        }
    }

    pub(crate) fn earliest_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G4 state poisoned");
        state.earliest_finish()
    }

    pub(crate) fn earliest_offload_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G4 state poisoned");
        state.earliest_offload_finish()
    }

    pub(crate) fn earliest_onboard_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G4 state poisoned");
        state.earliest_onboard_finish()
    }

    #[cfg(test)]
    fn reset_for_tests() {
        let store_slot = SHARED_G4_STORE.get_or_init(|| Mutex::new(None));
        let mut store_slot = store_slot
            .lock()
            .expect("shared G4 store registry poisoned");
        *store_slot = None;
    }
}

#[cfg(test)]
pub(crate) async fn shared_g4_test_guard() -> tokio::sync::OwnedMutexGuard<()> {
    let guard = SHARED_G4_TEST_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
        .lock_owned()
        .await;
    SharedG4Store::reset_for_tests();
    guard
}

#[cfg(test)]
pub(crate) fn shared_g4_test_guard_blocking() -> tokio::sync::OwnedMutexGuard<()> {
    let guard = SHARED_G4_TEST_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
        .blocking_lock_owned();
    SharedG4Store::reset_for_tests();
    guard
}

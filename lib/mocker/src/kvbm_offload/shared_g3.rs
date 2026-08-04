// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local shared G3 resource for mock KVBM offload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Result, bail};
use kvbm_engine::G3;
use kvbm_engine::offload::PendingTracker;
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_logical::pools::BlockDuplicationPolicy;
use kvbm_logical::registry::BlockRegistry;

use super::KvbmDriveMode;
use super::capacity_reservation::CapacityReservations;
use super::config::KvbmOffloadConfig;
use super::worker::{
    CompletedTransfer, CompletionAction, DeferredOwnerDrain, DrainResult, SharedDrainCounts,
    TransferDirection, TransferState, fire_completion_actions_with,
};

struct DeferredOwnerCompletions {
    actions: Vec<CompletionAction>,
    deadline_ms: f64,
    offload_registration_baseline: Option<u64>,
}

/// Shared process-local G3 resource. All mock workers in the process share
/// this block manager and this G2↔G3 PS model.
pub(crate) struct SharedG3Pool {
    config: KvbmOffloadConfig,
    drive_mode: KvbmDriveMode,
    manager: Arc<BlockManager<G3>>,
    state: Arc<Mutex<TransferState>>,
    pending_tracker: Arc<PendingTracker>,
    capacity_reservations: Arc<CapacityReservations>,
    pending_owner_drains: Mutex<HashMap<u64, DeferredOwnerDrain>>,
    pending_owner_completions: Mutex<HashMap<u64, DeferredOwnerCompletions>>,
}

static SHARED_G3_POOL: OnceLock<Mutex<Option<Weak<SharedG3Pool>>>> = OnceLock::new();

#[cfg(test)]
static SHARED_G3_TEST_LOCK: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();

impl SharedG3Pool {
    #[cfg(test)]
    pub(crate) fn get_or_create(config: &KvbmOffloadConfig) -> Result<Option<Arc<Self>>> {
        Self::get_or_create_with_mode(config, KvbmDriveMode::Live)
    }

    pub(crate) fn get_or_create_with_mode(
        config: &KvbmOffloadConfig,
        drive_mode: KvbmDriveMode,
    ) -> Result<Option<Arc<Self>>> {
        let Some(block_count) = config.num_g3_blocks else {
            return Ok(None);
        };
        let pool_slot = SHARED_G3_POOL.get_or_init(|| Mutex::new(None));
        let mut pool_slot = pool_slot.lock().expect("shared G3 pool registry poisoned");

        if let Some(existing) = pool_slot.as_ref().and_then(Weak::upgrade) {
            existing.validate_config(config, drive_mode)?;
            return Ok(Some(existing));
        }

        *pool_slot = None;
        let registry = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker())
            .build();
        let manager = Arc::new(
            BlockManager::<G3>::builder()
                .block_count(block_count)
                .block_size(config.block_size_tokens)
                .registry(registry)
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .with_lineage_backend()
                .build()
                .expect("BlockManager<G3> should build with valid config"),
        );
        let pool = Arc::new(Self {
            config: config.clone(),
            drive_mode,
            manager,
            state: Arc::new(Mutex::new(TransferState::new(
                config.bandwidth_g2_to_g3_gbps,
                config.bandwidth_g3_to_g2_gbps,
            ))),
            pending_tracker: Arc::new(PendingTracker::new()),
            capacity_reservations: Arc::new(CapacityReservations::default()),
            pending_owner_drains: Mutex::new(HashMap::new()),
            pending_owner_completions: Mutex::new(HashMap::new()),
        });
        *pool_slot = Some(Arc::downgrade(&pool));
        Ok(Some(pool))
    }

    fn validate_config(&self, config: &KvbmOffloadConfig, drive_mode: KvbmDriveMode) -> Result<()> {
        if self.config.num_g3_blocks != config.num_g3_blocks
            || self.config.block_size_tokens != config.block_size_tokens
            || self.config.block_size_bytes.unwrap_or(0) != config.block_size_bytes.unwrap_or(0)
            || self.config.bandwidth_g2_to_g3_gbps != config.bandwidth_g2_to_g3_gbps
            || self.config.bandwidth_g3_to_g2_gbps != config.bandwidth_g3_to_g2_gbps
            || self.drive_mode != drive_mode
        {
            bail!(
                "process-local shared G3 pool already exists with a different shape, bandwidth, or drive mode"
            );
        }
        Ok(())
    }

    pub(crate) fn manager(&self) -> Arc<BlockManager<G3>> {
        self.manager.clone()
    }

    pub(crate) fn transfer_state(&self) -> Arc<Mutex<TransferState>> {
        self.state.clone()
    }

    pub(crate) fn pending_tracker(&self) -> Arc<PendingTracker> {
        self.pending_tracker.clone()
    }

    pub(crate) fn capacity_reservations(&self) -> Arc<CapacityReservations> {
        self.capacity_reservations.clone()
    }

    pub(crate) fn release_capacity_reservations(&self, blocks: usize) {
        self.capacity_reservations.release(blocks);
    }

    pub(crate) fn drain_completions(&self, now_ms: f64, owner_id: u64) -> SharedDrainCounts {
        let registrations_before = self.manager.metrics().snapshot().registrations;
        let mut state = self.state.lock().expect("shared G3 state poisoned");
        let drained = state.drain_completions(now_ms, "shared-g3");
        drop(state);

        self.record_drained(drained, registrations_before, Some(owner_id), now_ms)
    }

    pub(crate) fn drain_completions_to_pending(&self, now_ms: f64) {
        let registrations_before = self.manager.metrics().snapshot().registrations;
        let mut state = self.state.lock().expect("shared G3 state poisoned");
        let drained = state.drain_completions(now_ms, "shared-g3");
        drop(state);

        self.record_drained(drained, registrations_before, None, now_ms);
    }

    pub(crate) fn drain_completions_ordered(
        &self,
        now_ms: f64,
        owner_id: u64,
        after_pipeline_completion: impl FnMut(CompletedTransfer),
    ) -> SharedDrainCounts {
        let registrations_before = self.manager.metrics().snapshot().registrations;
        self.defer_due_completions(now_ms, registrations_before);
        let pending = self
            .pending_owner_completions
            .lock()
            .expect("shared G3 ordered completion map poisoned")
            .remove(&owner_id);
        let Some(pending) = pending else {
            return SharedDrainCounts::default();
        };
        let result = fire_completion_actions_with(pending.actions, after_pipeline_completion);
        SharedDrainCounts {
            counts: result.by_owner.get(&owner_id).copied().unwrap_or_default(),
            // Ordered actions fire only at this owner's boundary, so onboard
            // blocks are current rather than already-published deferred work.
            deferred_onboard_blocks: 0,
            offload_registration_baseline: pending.offload_registration_baseline,
        }
    }

    pub(crate) fn defer_completions_ordered(&self, now_ms: f64) {
        let registrations_before = self.manager.metrics().snapshot().registrations;
        self.defer_due_completions(now_ms, registrations_before);
    }

    fn defer_due_completions(&self, now_ms: f64, registrations_before: u64) {
        let actions = {
            let mut state = self.state.lock().expect("shared G3 state poisoned");
            state.take_completion_actions(now_ms, "shared-g3")
        };
        if actions.is_empty() {
            return;
        }

        let mut by_owner: HashMap<u64, Vec<CompletionAction>> = HashMap::new();
        for action in actions {
            let owner_id = action
                .owner_id()
                .expect("shared G3 completion must have an owner");
            by_owner.entry(owner_id).or_default().push(action);
        }

        let mut pending = self
            .pending_owner_completions
            .lock()
            .expect("shared G3 ordered completion map poisoned");
        for (owner_id, actions) in by_owner {
            let has_offload = actions
                .iter()
                .any(|action| action.completed_transfer().direction == TransferDirection::G2ToG3);
            match pending.entry(owner_id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let pending = entry.get_mut();
                    pending.actions.extend(actions);
                    pending.deadline_ms = pending.deadline_ms.min(now_ms);
                    if has_offload {
                        pending.offload_registration_baseline = Some(
                            pending
                                .offload_registration_baseline
                                .map_or(registrations_before, |baseline| {
                                    baseline.min(registrations_before)
                                }),
                        );
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(DeferredOwnerCompletions {
                        actions,
                        deadline_ms: now_ms,
                        offload_registration_baseline: has_offload.then_some(registrations_before),
                    });
                }
            }
        }
    }

    fn record_drained(
        &self,
        drained: DrainResult,
        registrations_before: u64,
        owner_id: Option<u64>,
        now_ms: f64,
    ) -> SharedDrainCounts {
        let mut owner_result = SharedDrainCounts::default();
        let mut pending = self
            .pending_owner_drains
            .lock()
            .expect("shared G3 owner drain map poisoned");
        if let Some(owner_id) = owner_id
            && let Some(record) = pending.remove(&owner_id)
        {
            owner_result.add_deferred_record(record.counts);
        }
        for (owner, counts) in drained.by_owner {
            let record = SharedDrainCounts {
                counts,
                deferred_onboard_blocks: 0,
                offload_registration_baseline: (counts.offload_blocks > 0)
                    .then_some(registrations_before),
            };
            if Some(owner) == owner_id {
                owner_result.add_record(record);
            } else {
                pending
                    .entry(owner)
                    .and_modify(|pending| {
                        pending.counts.add_record(record);
                        pending.deadline_ms = pending.deadline_ms.min(now_ms);
                    })
                    .or_insert(DeferredOwnerDrain {
                        counts: record,
                        deadline_ms: now_ms,
                    });
            }
        }
        owner_result
    }

    pub(crate) fn earliest_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G3 state poisoned");
        state.earliest_finish()
    }

    pub(crate) fn pending_owner_deadline(&self, owner_id: u64) -> Option<f64> {
        let drained = self
            .pending_owner_drains
            .lock()
            .expect("shared G3 owner drain map poisoned")
            .get(&owner_id)
            .map(|pending| pending.deadline_ms);
        let completions = self
            .pending_owner_completions
            .lock()
            .expect("shared G3 ordered completion map poisoned")
            .get(&owner_id)
            .map(|pending| pending.deadline_ms);
        drained.into_iter().chain(completions).reduce(f64::min)
    }

    pub(crate) fn earliest_offload_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G3 state poisoned");
        state.earliest_offload_finish()
    }

    pub(crate) fn earliest_onboard_finish(&self) -> Option<f64> {
        let state = self.state.lock().expect("shared G3 state poisoned");
        state.earliest_onboard_finish()
    }

    #[cfg(test)]
    fn reset_for_tests() {
        let pool_slot = SHARED_G3_POOL.get_or_init(|| Mutex::new(None));
        let mut pool_slot = pool_slot.lock().expect("shared G3 pool registry poisoned");
        *pool_slot = None;
    }
}

#[cfg(test)]
pub(crate) async fn shared_g3_test_guard() -> tokio::sync::OwnedMutexGuard<()> {
    let guard = SHARED_G3_TEST_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
        .lock_owned()
        .await;
    SharedG3Pool::reset_for_tests();
    guard
}

#[cfg(test)]
pub(crate) fn shared_g3_test_guard_blocking() -> tokio::sync::OwnedMutexGuard<()> {
    let guard = SHARED_G3_TEST_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
        .blocking_lock_owned();
    SharedG3Pool::reset_for_tests();
    guard
}

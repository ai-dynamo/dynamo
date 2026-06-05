// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load-aware worker selection over a dynamic prefill fleet.
//!
//! Each worker carries its own per-worker concurrency semaphore plus
//! mutable counters (`inflight`, `load_net_new`). [`Selector::pick`]
//! awaits until some worker has free capacity, then atomically acquires
//! that worker's permit and increments its counters, returning a
//! [`ChargeGuard`] whose `Drop` decrements the counters and wakes the
//! next waiter via [`tokio::sync::Notify`].
//!
//! Continuous fleet membership: [`Selector::add_worker`] and
//! [`Selector::remove_worker`] are safe to call at any time; in-flight
//! requests against a removed worker keep their charge against the
//! detached slot until they finish.

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use velo_ext::InstanceId;

use super::execution::PrefillExecutionBackend;
use crate::protocol::PrefillRequest;

/// Selector tuning held immutably for the lifetime of a [`Selector`].
#[derive(Debug, Clone, Copy)]
pub struct SelectorConfig {
    /// Maximum concurrent in-flight requests per worker.
    pub per_worker_concurrency: u32,
    /// Block size used for the `net_new` computation
    /// (`tokens - hashes * block_size`).
    pub block_size: usize,
}

impl SelectorConfig {
    /// Compute the worst-case net-new prefill token count for a request:
    /// `token_ids.len() - num_provided_tokens`, floored at zero. The
    /// wire carries the FULL prompt in `token_ids` and `num_provided_tokens`
    /// (DNPT) is decode's total commitment from absolute position 0; the
    /// difference is the suffix prefill must actually compute.
    pub fn net_new(&self, req: &PrefillRequest) -> u64 {
        (req.token_ids.len() as u64).saturating_sub(req.num_provided_tokens as u64)
    }
}

/// Mutable per-worker counters protected by [`WorkerSlot::counters`].
#[derive(Debug, Default, Clone, Copy)]
pub struct WorkerCounters {
    /// Number of requests currently in flight on this worker.
    pub inflight: u32,
    /// Sum of `net_new` tokens across the in-flight requests on this
    /// worker.
    pub load_net_new: u64,
}

/// One prefill worker the dispatcher can route to.
pub struct WorkerSlot {
    /// Worker's hub registration id (used for logging and as the
    /// addressing key for the velo backend).
    pub instance_id: InstanceId,
    /// Execution backend bound to this worker.
    pub backend: Arc<dyn PrefillExecutionBackend>,
    /// Per-worker concurrency cap. Acquired in [`Selector::pick`]; the
    /// returned [`OwnedSemaphorePermit`] is held by the dispatcher task
    /// until execution completes.
    permits: Arc<Semaphore>,
    /// Mutable counters used by the selector's sort key.
    counters: Mutex<WorkerCounters>,
}

impl WorkerSlot {
    /// Snapshot of the current counters.
    pub fn counters(&self) -> WorkerCounters {
        *self.counters.lock()
    }
}

impl std::fmt::Debug for WorkerSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.counters();
        f.debug_struct("WorkerSlot")
            .field("instance_id", &self.instance_id)
            .field("backend", &self.backend.label())
            .field("inflight", &c.inflight)
            .field("load_net_new", &c.load_net_new)
            .finish()
    }
}

/// Selector over a dynamic fleet of [`WorkerSlot`]s.
pub struct Selector {
    config: SelectorConfig,
    state: RwLock<Vec<Arc<WorkerSlot>>>,
    /// Signaled when a worker is added or a [`ChargeGuard`] is dropped.
    /// Used by [`Selector::pick`] to wake when the fleet may now have
    /// capacity.
    notify: Arc<Notify>,
}

impl std::fmt::Debug for Selector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read();
        f.debug_struct("Selector")
            .field("worker_count", &state.len())
            .field(
                "per_worker_concurrency",
                &self.config.per_worker_concurrency,
            )
            .field("block_size", &self.config.block_size)
            .finish()
    }
}

impl Selector {
    /// Construct an empty selector. Workers must be added via
    /// [`Self::add_worker`] before [`Self::pick`] can succeed.
    pub fn new(config: SelectorConfig) -> Arc<Self> {
        assert!(
            config.per_worker_concurrency >= 1,
            "per_worker_concurrency must be >= 1"
        );
        Arc::new(Self {
            config,
            state: RwLock::new(Vec::new()),
            notify: Arc::new(Notify::new()),
        })
    }

    /// Selector configuration (immutable for the selector's lifetime).
    pub fn config(&self) -> SelectorConfig {
        self.config
    }

    /// Add a worker to the fleet. Idempotent: a re-add for the same
    /// `instance_id` is treated as a replace (the old slot is detached;
    /// in-flight tasks against it keep their charge but their permit
    /// release no longer affects the new slot). Returns `true` if the
    /// worker was newly added, `false` if it replaced an existing slot.
    pub fn add_worker(
        &self,
        instance_id: InstanceId,
        backend: Arc<dyn PrefillExecutionBackend>,
    ) -> bool {
        let slot = Arc::new(WorkerSlot {
            instance_id,
            backend,
            permits: Arc::new(Semaphore::new(self.config.per_worker_concurrency as usize)),
            counters: Mutex::new(WorkerCounters::default()),
        });
        let mut state = self.state.write();
        let existing_idx = state.iter().position(|s| s.instance_id == instance_id);
        let newly_added = existing_idx.is_none();
        if let Some(idx) = existing_idx {
            state[idx] = slot;
        } else {
            state.push(slot);
        }
        drop(state);
        // Wake any waiter parked because the fleet was empty / at cap.
        self.notify.notify_waiters();
        newly_added
    }

    /// Remove a worker from the fleet by id. In-flight tasks against
    /// the removed slot keep their charge until completion; the slot
    /// itself becomes unreachable for new picks.
    pub fn remove_worker(&self, instance_id: InstanceId) {
        let mut state = self.state.write();
        if let Some(idx) = state.iter().position(|s| s.instance_id == instance_id) {
            state.remove(idx);
        }
    }

    /// Snapshot of the fleet membership. Cheap clone of `Arc`s.
    pub fn snapshot(&self) -> Vec<Arc<WorkerSlot>> {
        self.state.read().clone()
    }

    /// Number of workers currently registered.
    pub fn worker_count(&self) -> usize {
        self.state.read().len()
    }

    /// Number of permits currently available across the whole fleet —
    /// the sum of every worker's free per-worker semaphore slots.
    pub fn available_permits(&self) -> u32 {
        self.state
            .read()
            .iter()
            .map(|s| s.permits.available_permits() as u32)
            .sum()
    }

    /// Total permit capacity across the whole fleet — the denominator for the
    /// circuit breaker's free-capacity fraction
    /// (`available_permits / total_permits`). Equals
    /// `worker_count * per_worker_concurrency`; `0` when the fleet is empty
    /// (callers must guard the division — an empty fleet has no spare
    /// capacity to disaggregate into).
    pub fn total_permits(&self) -> u32 {
        self.state.read().len() as u32 * self.config.per_worker_concurrency
    }

    /// Pick a worker for `net_new` tokens of new prefill work.
    ///
    /// Blocks until at least one worker has a free per-worker permit.
    /// On success, charges the chosen worker's counters and returns the
    /// slot, its acquired permit (the dispatcher task holds it until
    /// execution completes), and a [`ChargeGuard`] whose `Drop`
    /// decrements the counters and wakes the next waiter.
    ///
    /// If the fleet is empty, this call parks indefinitely on
    /// [`Self::add_worker`].
    pub async fn pick(self: &Arc<Self>, net_new: u64) -> PickedSlot {
        loop {
            // Arm the wait *before* the capacity check so we never miss
            // a notification that happens between check-and-park.
            let notified = self.notify.notified();
            if let Some(picked) = self.try_pick(net_new) {
                return picked;
            }
            notified.await;
        }
    }

    /// Non-blocking variant of [`Self::pick`]. Returns `None` if no
    /// worker currently has free capacity.
    pub fn try_pick(self: &Arc<Self>, net_new: u64) -> Option<PickedSlot> {
        let workers = self.state.read().clone();
        if workers.is_empty() {
            return None;
        }

        // Sort indices by (load_net_new, inflight, original index) to
        // mirror the branch's deterministic tiebreak.
        let mut order: Vec<(usize, WorkerCounters)> = workers
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.counters()))
            .collect();
        order.sort_by_key(|(i, c)| (c.load_net_new, c.inflight, *i));

        for (idx, _c) in order {
            let slot = Arc::clone(&workers[idx]);
            if let Ok(permit) = Arc::clone(&slot.permits).try_acquire_owned() {
                // Charge counters under the slot's own lock.
                {
                    let mut counters = slot.counters.lock();
                    counters.inflight = counters.inflight.saturating_add(1);
                    counters.load_net_new = counters.load_net_new.saturating_add(net_new);
                }
                let guard = ChargeGuard {
                    slot: Arc::clone(&slot),
                    net_new,
                    notify: Arc::clone(&self.notify),
                };
                return Some(PickedSlot {
                    slot,
                    permit,
                    guard,
                });
            }
        }
        None
    }
}

/// One picked worker, with the bits the dispatcher task needs to hold
/// across execution.
pub struct PickedSlot {
    /// The chosen worker.
    pub slot: Arc<WorkerSlot>,
    /// Per-worker permit; the dispatcher task holds this until
    /// `backend.execute()` returns. Drop releases the slot for the next
    /// pick.
    pub permit: OwnedSemaphorePermit,
    /// Counter guard; drop decrements `inflight` and `load_net_new`
    /// and wakes the next waiter.
    pub guard: ChargeGuard,
}

/// Decrements `inflight` and `load_net_new` for a charged worker when
/// dropped, and wakes any [`Selector::pick`] waiter via the selector's
/// `Notify`.
pub struct ChargeGuard {
    slot: Arc<WorkerSlot>,
    net_new: u64,
    notify: Arc<Notify>,
}

impl Drop for ChargeGuard {
    fn drop(&mut self) {
        {
            let mut counters = self.slot.counters.lock();
            counters.inflight = counters.inflight.saturating_sub(1);
            counters.load_net_new = counters.load_net_new.saturating_sub(self.net_new);
        }
        self.notify.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::prefill_router::execution::PrefillExecutionBackend;
    use anyhow::Result;
    use async_trait::async_trait;
    use kvbm_protocols::disagg::DISAGG_PROTOCOL_VERSION;
    use std::time::Duration;

    use super::super::dispatcher::DispatchOutcome;

    struct StubBackend {
        id: InstanceId,
    }

    #[async_trait]
    impl PrefillExecutionBackend for StubBackend {
        fn instance_id(&self) -> InstanceId {
            self.id
        }
        fn label(&self) -> &'static str {
            "stub"
        }
        async fn execute(&self, _req: PrefillRequest) -> Result<DispatchOutcome> {
            Ok(DispatchOutcome::Accepted)
        }
    }

    fn cfg() -> SelectorConfig {
        SelectorConfig {
            per_worker_concurrency: 4,
            block_size: 16,
        }
    }

    fn add(sel: &Arc<Selector>) -> InstanceId {
        let id = InstanceId::new_v4();
        sel.add_worker(id, Arc::new(StubBackend { id }));
        id
    }

    fn make_request(n_tokens: usize, n_hashes: usize) -> PrefillRequest {
        use kvbm_protocols::disagg::KvHashingRequestEnvelope;
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: "r".to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            token_ids: vec![0u32; n_tokens],
            num_provided_tokens: n_hashes * 16,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: None,
        }
    }

    #[test]
    fn net_new_subtracts_cached_blocks() {
        let c = cfg();
        let req = make_request(100, 2);
        // 100 - 2 * 16 = 68
        assert_eq!(c.net_new(&req), 68);
    }

    #[test]
    fn net_new_floors_at_zero_when_cache_exceeds_tokens() {
        let c = cfg();
        let req = make_request(10, 5);
        assert_eq!(c.net_new(&req), 0);
    }

    #[tokio::test]
    async fn pick_picks_lowest_load() {
        let sel = Selector::new(cfg());
        let a = add(&sel);
        let b = add(&sel);
        let c = add(&sel);

        // Manually set per-worker counters to mimic the branch's test fleet.
        for slot in sel.snapshot() {
            let mut counters = slot.counters.lock();
            if slot.instance_id == a {
                *counters = WorkerCounters {
                    inflight: 2,
                    load_net_new: 500,
                };
            } else if slot.instance_id == b {
                *counters = WorkerCounters {
                    inflight: 1,
                    load_net_new: 100,
                };
            } else if slot.instance_id == c {
                *counters = WorkerCounters {
                    inflight: 1,
                    load_net_new: 300,
                };
            }
        }

        let picked = sel.try_pick(50).expect("should pick");
        assert_eq!(picked.slot.instance_id, b);
    }

    #[tokio::test]
    async fn try_pick_skips_workers_at_concurrency_cap() {
        let sel = Selector::new(SelectorConfig {
            per_worker_concurrency: 2,
            block_size: 16,
        });
        let a = add(&sel);
        let b = add(&sel);

        // Saturate worker `a` by draining its permits.
        let a_slot = sel
            .snapshot()
            .into_iter()
            .find(|s| s.instance_id == a)
            .unwrap();
        let _p1 = Arc::clone(&a_slot.permits).try_acquire_owned().unwrap();
        let _p2 = Arc::clone(&a_slot.permits).try_acquire_owned().unwrap();
        // Give `a` lower load so it would be preferred by sort.
        a_slot.counters.lock().load_net_new = 100;
        sel.snapshot()
            .into_iter()
            .find(|s| s.instance_id == b)
            .unwrap()
            .counters
            .lock()
            .load_net_new = 999;

        let picked = sel.try_pick(10).expect("b should still be available");
        assert_eq!(picked.slot.instance_id, b);
    }

    #[tokio::test]
    async fn try_pick_breaks_ties_by_inflight_then_insertion_order() {
        let sel = Selector::new(cfg());
        let _a = add(&sel);
        let b = add(&sel);
        let _c = add(&sel);

        for slot in sel.snapshot() {
            let mut counters = slot.counters.lock();
            counters.load_net_new = 100;
            counters.inflight = if slot.instance_id == b { 1 } else { 2 };
        }
        // Lower a.inflight to 1 so `a` and `b` tie on (load, inflight),
        // breaking by index (a < b).
        let a_slot = sel.snapshot().into_iter().next().unwrap();
        a_slot.counters.lock().inflight = 1;

        let picked = sel.try_pick(10).expect("pick");
        // a < b in insertion order, equal counters → a wins.
        assert_eq!(picked.slot.instance_id, a_slot.instance_id);

        // Drop the prior pick's permit + guard first, so `a` starts the
        // next round with all permits free, then drain them so a is at
        // cap and `b` is the only candidate.
        drop(picked);
        let _drain: Vec<_> = (0..cfg().per_worker_concurrency)
            .map(|_| Arc::clone(&a_slot.permits).try_acquire_owned().unwrap())
            .collect();
        let picked2 = sel.try_pick(10).expect("pick again");
        assert_eq!(picked2.slot.instance_id, b);
    }

    #[tokio::test]
    async fn try_pick_returns_none_when_empty() {
        let sel = Selector::new(cfg());
        assert!(sel.try_pick(10).is_none());
    }

    #[tokio::test]
    async fn try_pick_returns_none_when_all_at_cap() {
        let sel = Selector::new(SelectorConfig {
            per_worker_concurrency: 1,
            block_size: 16,
        });
        add(&sel);
        let first = sel.try_pick(10).expect("first");
        // Hold the permit; nothing else available.
        assert!(sel.try_pick(10).is_none());
        drop(first);
        assert!(sel.try_pick(10).is_some());
    }

    #[tokio::test]
    async fn charge_guard_decrements_on_drop() {
        let sel = Selector::new(cfg());
        let id = add(&sel);
        let picked = sel.try_pick(123).expect("pick");
        let slot_arc = sel
            .snapshot()
            .into_iter()
            .find(|s| s.instance_id == id)
            .unwrap();
        let c = slot_arc.counters();
        assert_eq!(c.inflight, 1);
        assert_eq!(c.load_net_new, 123);
        drop(picked);
        let c = slot_arc.counters();
        assert_eq!(c.inflight, 0);
        assert_eq!(c.load_net_new, 0);
    }

    #[tokio::test]
    async fn pick_unblocks_when_worker_added() {
        let sel = Selector::new(cfg());
        let sel_clone = Arc::clone(&sel);
        let waiter = tokio::spawn(async move { sel_clone.pick(10).await });
        // No workers yet → waiter parked.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());
        add(&sel);
        let picked = tokio::time::timeout(Duration::from_millis(500), waiter)
            .await
            .expect("pick should complete")
            .unwrap();
        assert!(picked.slot.backend.label() == "stub");
    }

    #[tokio::test]
    async fn pick_unblocks_when_charge_dropped() {
        let sel = Selector::new(SelectorConfig {
            per_worker_concurrency: 1,
            block_size: 16,
        });
        add(&sel);
        let first = sel.try_pick(10).expect("first");
        let sel_clone = Arc::clone(&sel);
        let waiter = tokio::spawn(async move { sel_clone.pick(10).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "should block until first drops");
        drop(first);
        let picked = tokio::time::timeout(Duration::from_millis(500), waiter)
            .await
            .expect("pick should complete")
            .unwrap();
        assert_eq!(picked.slot.counters().inflight, 1);
    }

    #[tokio::test]
    async fn remove_worker_detaches_from_future_picks() {
        let sel = Selector::new(cfg());
        let a = add(&sel);
        let b = add(&sel);
        sel.remove_worker(a);
        let picked = sel.try_pick(10).expect("pick");
        assert_eq!(picked.slot.instance_id, b);
        assert_eq!(sel.worker_count(), 1);
    }

    #[tokio::test]
    async fn total_permits_is_workers_times_concurrency() {
        let sel = Selector::new(cfg()); // per_worker_concurrency = 4
        assert_eq!(sel.total_permits(), 0, "empty fleet has zero total permits");
        let a = add(&sel);
        assert_eq!(sel.total_permits(), 4);
        add(&sel);
        assert_eq!(sel.total_permits(), 8);
        // Draining permits does not change the TOTAL (only available).
        let a_slot = sel
            .snapshot()
            .into_iter()
            .find(|s| s.instance_id == a)
            .unwrap();
        let _p = Arc::clone(&a_slot.permits).try_acquire_owned().unwrap();
        assert_eq!(sel.total_permits(), 8);
        assert_eq!(sel.available_permits(), 7);
        // Removing a worker drops its capacity from the total.
        sel.remove_worker(a);
        assert_eq!(sel.total_permits(), 4);
    }

    #[tokio::test]
    async fn add_worker_replace_returns_false() {
        let sel = Selector::new(cfg());
        let id = InstanceId::new_v4();
        assert!(sel.add_worker(id, Arc::new(StubBackend { id })));
        assert!(!sel.add_worker(id, Arc::new(StubBackend { id })));
        assert_eq!(sel.worker_count(), 1);
    }
}

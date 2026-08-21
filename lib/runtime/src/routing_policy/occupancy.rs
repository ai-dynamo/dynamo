// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use dashmap::{DashMap, mapref::entry::Entry};

use super::{CandidateView, RouteContext, RouteDecision, RoutePicker, RoutePolicy};
use crate::{component::Endpoint, traits::DistributedRuntimeProvider};

/// Load-aware policies backed by request occupancy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyPolicy {
    PowerOfTwoChoices,
    LeastLoaded,
}

impl OccupancyPolicy {
    fn route_policy(self) -> RoutePolicy {
        match self {
            Self::PowerOfTwoChoices => RoutePolicy::PowerOfTwoChoices,
            Self::LeastLoaded => RoutePolicy::LeastLoaded,
        }
    }
}

/// The result of an atomic load-aware selection and reservation.
pub struct OccupancySelection {
    worker_id: u64,
    candidate_count: usize,
    load: u64,
    reservation: OccupancyReservation,
}

impl OccupancySelection {
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Selected worker load after this request was admitted.
    pub fn load(&self) -> u64 {
        self.load
    }

    pub fn into_reservation(self) -> OccupancyReservation {
        self.reservation
    }
}

/// Shared O(1) per-worker request occupancy.
///
/// Discovery controls eligibility separately from accounting. Removing a worker
/// marks it absent immediately, but a counter with live reservations remains until
/// its final reservation releases. Re-adding the same worker ID therefore sees the
/// retained load instead of starting from zero.
#[derive(Debug, Default)]
pub struct RoutingOccupancyState {
    counts: DashMap<u64, Arc<AtomicU64>>,
    discovered: parking_lot::RwLock<HashSet<u64>>,
    admission_lock: parking_lot::Mutex<()>,
}

impl RoutingOccupancyState {
    fn increment_locked(&self, worker_id: u64) -> Arc<AtomicU64> {
        let count = self
            .counts
            .entry(worker_id)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        count.fetch_add(1, Ordering::Relaxed);
        count
    }

    pub(crate) fn increment(&self, worker_id: u64) -> Arc<AtomicU64> {
        let _admission = self.admission_lock.lock();
        self.increment_locked(worker_id)
    }

    pub(crate) async fn select_exact_min_and_increment(&self, worker_ids: &[u64]) -> Option<u64> {
        let picker = RoutePicker::new(RoutePolicy::LeastLoaded);
        self.select_and_admit(
            &picker,
            CandidateView::Workers(worker_ids),
            RouteContext::default(),
        )
        .map(|(decision, _)| decision.target.worker_id)
    }

    pub(crate) fn peek(
        &self,
        picker: &RoutePicker,
        candidates: CandidateView<'_>,
        context: RouteContext,
    ) -> Option<RouteDecision> {
        picker.peek(candidates, context, |id| self.load(id))
    }

    pub(crate) fn select_and_admit(
        &self,
        picker: &RoutePicker,
        candidates: CandidateView<'_>,
        context: RouteContext,
    ) -> Option<(RouteDecision, Option<Arc<AtomicU64>>)> {
        let _admission = self.admission_lock.lock();
        let decision = picker.select(candidates, context, |id| self.load(id))?;
        let counter = match decision.admission {
            super::AdmissionKind::None => None,
            super::AdmissionKind::Occupancy => {
                Some(self.increment_locked(decision.target.worker_id))
            }
        };
        Some((decision, counter))
    }

    /// Atomically select and reserve a worker for a load-aware policy.
    pub fn select_and_reserve(
        self: &Arc<Self>,
        policy: OccupancyPolicy,
        candidates: &[u64],
    ) -> Option<OccupancySelection> {
        let _admission = self.admission_lock.lock();
        let decision = RoutePicker::new(policy.route_policy()).select(
            CandidateView::Workers(candidates),
            RouteContext::default(),
            |id| self.load(id),
        )?;
        let worker_id = decision.target.worker_id;
        let counter = self.increment_locked(worker_id);
        let load = counter.load(Ordering::Relaxed);
        Some(OccupancySelection {
            worker_id,
            candidate_count: candidates.len(),
            load,
            reservation: OccupancyReservation::from_counter(Arc::clone(self), worker_id, counter),
        })
    }

    /// Reserve an explicitly selected worker.
    pub fn reserve(self: &Arc<Self>, worker_id: u64) -> OccupancyReservation {
        let counter = self.increment(worker_id);
        OccupancyReservation::from_counter(Arc::clone(self), worker_id, counter)
    }

    /// Peek a load-aware selection without admitting work.
    pub fn peek_policy(&self, policy: OccupancyPolicy, candidates: &[u64]) -> Option<u64> {
        RoutePicker::new(policy.route_policy())
            .peek(
                CandidateView::Workers(candidates),
                RouteContext::default(),
                |id| self.load(id),
            )
            .map(|decision| decision.target.worker_id)
    }

    fn decrement_locked(&self, worker_id: u64, counter: &Arc<AtomicU64>) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(1))
        });

        if counter.load(Ordering::Relaxed) != 0 || self.discovered.read().contains(&worker_id) {
            return;
        }

        if let Entry::Occupied(entry) = self.counts.entry(worker_id)
            && Arc::ptr_eq(entry.get(), counter)
            && entry.get().load(Ordering::Relaxed) == 0
        {
            entry.remove();
        }
    }

    pub(crate) fn release(&self, worker_id: u64, counter: &Arc<AtomicU64>) {
        let _admission = self.admission_lock.lock();
        self.decrement_locked(worker_id, counter);
    }

    pub(crate) fn decrement(&self, worker_id: u64) {
        let _admission = self.admission_lock.lock();
        let counter = self.counts.get(&worker_id).map(|entry| entry.clone());
        if let Some(counter) = counter {
            self.decrement_locked(worker_id, &counter);
        }
    }

    pub fn load(&self, worker_id: u64) -> u64 {
        self.counts
            .get(&worker_id)
            .map(|count| count.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Reconcile discovery eligibility without discarding guard-owned accounting.
    pub(crate) fn retain(&self, worker_ids: &[u64]) {
        let _admission = self.admission_lock.lock();
        let mut live = self.discovered.write();
        live.clear();
        live.extend(worker_ids.iter().copied());
        self.counts.retain(|worker_id, count| {
            live.contains(worker_id) || count.load(Ordering::Relaxed) != 0
        });
    }

    #[cfg(test)]
    pub(crate) fn contains_slot(&self, worker_id: u64) -> bool {
        self.counts.contains_key(&worker_id)
    }
}

/// One guard-owned occupancy booking.
pub struct OccupancyReservation {
    state: Arc<RoutingOccupancyState>,
    worker_id: u64,
    counter: Arc<AtomicU64>,
}

impl OccupancyReservation {
    pub(crate) fn from_counter(
        state: Arc<RoutingOccupancyState>,
        worker_id: u64,
        counter: Arc<AtomicU64>,
    ) -> Self {
        Self {
            state,
            worker_id,
            counter,
        }
    }

    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn load(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Move this booking to the worker selected by transport fallback.
    pub fn retarget(&mut self, worker_id: u64) -> u64 {
        if self.worker_id == worker_id {
            return self.load();
        }

        let _admission = self.state.admission_lock.lock();
        let next = self.state.increment_locked(worker_id);
        self.state.decrement_locked(self.worker_id, &self.counter);
        self.worker_id = worker_id;
        self.counter = next;
        self.load()
    }
}

impl Drop for OccupancyReservation {
    fn drop(&mut self) {
        self.state.release(self.worker_id, &self.counter);
    }
}

/// Get or create the shared routing occupancy state for an endpoint.
pub(crate) async fn get_or_create_routing_occupancy_state(
    endpoint: &Endpoint,
) -> Arc<RoutingOccupancyState> {
    let drt = endpoint.drt();
    let registry = drt.routing_occupancy_states();
    let mut registry = registry.lock().await;

    if let Some(weak) = registry.get(endpoint) {
        if let Some(state) = weak.upgrade() {
            return state;
        }
        registry.remove(endpoint);
    }

    let state = Arc::new(RoutingOccupancyState::default());
    registry.insert(endpoint.clone(), Arc::downgrade(&state));
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_worker_keeps_live_reservation_across_same_id_readd() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[7]);
        let old = state.reserve(7);

        state.retain(&[]);
        assert_eq!(state.load(7), 1);
        assert!(state.contains_slot(7));

        state.retain(&[7]);
        let new = state.reserve(7);
        assert_eq!(state.load(7), 2);

        drop(old);
        assert_eq!(state.load(7), 1);
        drop(new);
        assert_eq!(state.load(7), 0);
    }

    #[test]
    fn absent_worker_slot_is_removed_after_final_release() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[7]);
        let reservation = state.reserve(7);

        state.retain(&[]);
        assert!(state.contains_slot(7));
        drop(reservation);

        assert_eq!(state.load(7), 0);
        assert!(!state.contains_slot(7));
    }

    #[test]
    fn retarget_moves_exactly_one_booking() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[1, 2]);
        let mut reservation = state.reserve(1);

        assert_eq!(reservation.retarget(2), 1);
        assert_eq!(state.load(1), 0);
        assert_eq!(state.load(2), 1);

        drop(reservation);
        assert_eq!(state.load(2), 0);
    }

    #[test]
    fn concurrent_selection_and_reservation_stays_balanced() {
        let state = Arc::new(RoutingOccupancyState::default());
        state.retain(&[1, 2, 3]);
        let threads = (0..90)
            .map(|_| {
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    state
                        .select_and_reserve(OccupancyPolicy::LeastLoaded, &[1, 2, 3])
                        .unwrap()
                        .into_reservation()
                })
            })
            .collect::<Vec<_>>();
        let reservations = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!([state.load(1), state.load(2), state.load(3)], [30, 30, 30]);
        drop(reservations);
        assert_eq!([state.load(1), state.load(2), state.load(3)], [0, 0, 0]);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache-free worker selection shared by `PushRouter` and higher-level routing hosts.
//!
//! Topology changes rebuild the candidate table on the discovery path. Request selection then
//! performs a fixed number of vector/hash lookups: one for round-robin and random, or two load
//! probes for P2C.

#[cfg(test)]
use std::cell::Cell;
use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::Stream;
use parking_lot::Mutex;

use super::fast::FastPicker;
use crate::{
    component::{RoutingInstances, RoutingInstancesObserver},
    error::{DynamoError, ErrorType},
    pipeline::{ManyOut, PipelineError, ResponseStream},
    protocols::maybe_error::MaybeError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuiltinRoutingPolicy {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
}

impl BuiltinRoutingPolicy {
    fn tracks_occupancy(self) -> bool {
        matches!(self, Self::PowerOfTwoChoices)
    }
}

#[derive(Clone, Copy, Default)]
struct WorkerLoad {
    active: u64,
    generation: u64,
}

#[derive(Default)]
struct PickerState {
    candidates: Vec<u64>,
    routable: HashSet<u64>,
    loads: HashMap<u64, WorkerLoad>,
    next_generation: u64,
    #[cfg(test)]
    selection_probes: Cell<usize>,
}

impl PickerState {
    fn reconcile(&mut self, routing_instances: &RoutingInstances, tracks_occupancy: bool) {
        self.routable.clear();
        self.routable
            .extend(routing_instances.routable_ids().iter().copied());
        self.candidates.clear();
        self.candidates
            .extend_from_slice(routing_instances.free_ids());
        if !tracks_occupancy {
            self.loads.clear();
            return;
        }

        let mut live = routing_instances
            .discovered_ids()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        live.extend(routing_instances.routable_ids().iter().copied());
        self.loads.retain(|worker_id, _| live.contains(worker_id));
        for worker_id in live {
            if self.loads.contains_key(&worker_id) {
                continue;
            }
            self.next_generation = self.next_generation.wrapping_add(1);
            self.loads.insert(
                worker_id,
                WorkerLoad {
                    active: 0,
                    generation: self.next_generation,
                },
            );
        }
    }

    fn load(&self, worker_id: u64) -> u64 {
        self.loads.get(&worker_id).map_or(0, |load| load.active)
    }

    #[inline(always)]
    fn candidate(&self, index: usize) -> u64 {
        #[cfg(test)]
        self.selection_probes
            .set(self.selection_probes.get().saturating_add(1));
        self.candidates[index]
    }

    fn increment(&mut self, worker_id: u64) -> Option<(u64, u64)> {
        let next_generation = &mut self.next_generation;
        let worker = self.loads.entry(worker_id).or_insert_with(|| {
            *next_generation = next_generation.wrapping_add(1);
            WorkerLoad {
                active: 0,
                generation: *next_generation,
            }
        });
        let next = worker.active.checked_add(1)?;
        worker.active = next;
        Some((worker.generation, next))
    }

    fn decrement(&mut self, worker_id: u64, generation: u64) {
        let Some(worker) = self.loads.get_mut(&worker_id) else {
            return;
        };
        if worker.generation != generation || worker.active == 0 {
            return;
        }
        worker.active -= 1;
    }
}

struct SharedPickerState(Mutex<PickerState>, bool);

impl SharedPickerState {
    fn new(tracks_occupancy: bool) -> Self {
        Self(Mutex::new(PickerState::default()), tracks_occupancy)
    }
}

impl RoutingInstancesObserver for SharedPickerState {
    fn update(&self, routing_instances: &RoutingInstances) {
        self.0.lock().reconcile(routing_instances, self.1);
    }
}

pub struct BuiltinWorkerPicker {
    policy: BuiltinRoutingPolicy,
    picker: FastPicker,
    state: Arc<SharedPickerState>,
}

impl BuiltinWorkerPicker {
    pub(crate) fn new(
        client: &crate::component::Client,
        policy: BuiltinRoutingPolicy,
    ) -> Arc<Self> {
        if !policy.tracks_occupancy() {
            return Self::create(client, policy);
        }

        client.get_or_create_p2c_worker_picker(|| Self::create(client, policy))
    }

    fn create(client: &crate::component::Client, policy: BuiltinRoutingPolicy) -> Arc<Self> {
        let state = Arc::new(SharedPickerState::new(policy.tracks_occupancy()));
        client.observe_routing_instances(state.clone());
        Arc::new(Self {
            policy,
            picker: FastPicker::new(),
            state,
        })
    }

    pub fn policy(&self) -> BuiltinRoutingPolicy {
        self.policy
    }

    pub fn select(&self) -> anyhow::Result<BuiltinWorkerReservation> {
        self.choose(true)
    }

    pub fn peek(&self) -> Option<u64> {
        self.choose(false)
            .ok()
            .map(|reservation| reservation.worker_id)
    }

    pub fn reserve_exact(&self, worker_id: u64) -> anyhow::Result<BuiltinWorkerReservation> {
        let mut state = self.state.0.lock();
        if !state.routable.contains(&worker_id) {
            anyhow::bail!("instance_id={worker_id} is not routable");
        }
        let tracking = self
            .policy
            .tracks_occupancy()
            .then(|| state.increment(worker_id))
            .flatten();
        let load = tracking.map_or_else(|| state.load(worker_id), |(_, load)| load);
        Ok(BuiltinWorkerReservation {
            state: self.policy.tracks_occupancy().then(|| self.state.clone()),
            worker_id,
            generation: tracking.map(|(generation, _)| generation),
            candidate_count: state.candidates.len(),
            load,
        })
    }

    fn choose(&self, commit: bool) -> anyhow::Result<BuiltinWorkerReservation> {
        let mut state = self.state.0.lock();
        let candidate_count = state.candidates.len();
        let worker_id = match self.policy {
            BuiltinRoutingPolicy::RoundRobin => self
                .picker
                .round_robin_index(candidate_count, commit)
                .map(|index| state.candidate(index)),
            BuiltinRoutingPolicy::Random => {
                FastPicker::random_index(candidate_count).map(|index| state.candidate(index))
            }
            BuiltinRoutingPolicy::PowerOfTwoChoices => {
                FastPicker::power_of_two_choices_index(candidate_count, |index| {
                    state.load(state.candidate(index))
                })
                .map(|index| state.candidate(index))
            }
        }
        .ok_or_else(|| empty_pool_error(&state))?;

        let tracking = (commit && self.policy.tracks_occupancy())
            .then(|| state.increment(worker_id))
            .flatten();
        let load = tracking.map_or_else(|| state.load(worker_id), |(_, load)| load);
        Ok(BuiltinWorkerReservation {
            state: tracking.map(|_| self.state.clone()),
            worker_id,
            generation: tracking.map(|(generation, _)| generation),
            candidate_count,
            load,
        })
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn occupancy_for_test(&self, worker_id: u64) -> u64 {
        self.state.0.lock().load(worker_id)
    }
}

fn empty_pool_error(state: &PickerState) -> anyhow::Error {
    if !state.routable.is_empty() {
        let cause = PipelineError::ServiceOverloaded(
            "All workers are busy, please retry later".to_string(),
        );
        return DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message("All workers are busy, please retry later")
            .cause(cause)
            .build()
            .into();
    }
    DynamoError::builder()
        .error_type(ErrorType::Unavailable)
        .message("No workers available")
        .build()
        .into()
}

pub struct BuiltinWorkerReservation {
    state: Option<Arc<SharedPickerState>>,
    worker_id: u64,
    generation: Option<u64>,
    candidate_count: usize,
    load: u64,
}

impl BuiltinWorkerReservation {
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    pub fn load(&self) -> u64 {
        self.load
    }

    pub fn retarget(&mut self, worker_id: u64) {
        if self.worker_id == worker_id {
            return;
        }
        let Some(state) = self.state.as_ref() else {
            self.worker_id = worker_id;
            return;
        };
        let mut picker_state = state.0.lock();
        if let Some(generation) = self.generation.take() {
            picker_state.decrement(self.worker_id, generation);
        }
        let tracking = picker_state.increment(worker_id);
        self.generation = tracking.map(|(generation, _)| generation);
        self.worker_id = worker_id;
        self.load = tracking.map_or_else(|| picker_state.load(worker_id), |(_, load)| load);
    }

    pub fn into_tracked_stream<U: crate::pipeline::Data + MaybeError>(
        self,
        stream: ManyOut<U>,
    ) -> ManyOut<U> {
        if self.state.is_none() {
            return stream;
        }
        let context = stream.context();
        ResponseStream::new(
            Box::pin(BuiltinTrackedStream {
                inner: stream,
                reservation: Some(self),
            }),
            context,
        )
    }

    fn release(&mut self) {
        let (Some(state), Some(generation)) = (self.state.as_ref(), self.generation.take()) else {
            return;
        };
        state.0.lock().decrement(self.worker_id, generation);
    }
}

impl Drop for BuiltinWorkerReservation {
    fn drop(&mut self) {
        self.release();
    }
}

struct BuiltinTrackedStream<U: crate::pipeline::Data + MaybeError> {
    inner: ManyOut<U>,
    reservation: Option<BuiltinWorkerReservation>,
}

impl<U: crate::pipeline::Data + MaybeError> Stream for BuiltinTrackedStream<U> {
    type Item = U;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = self.inner.as_mut().poll_next(cx);
        if matches!(&result, Poll::Ready(None))
            || matches!(&result, Poll::Ready(Some(item)) if item.err().is_some())
        {
            self.reservation.take();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_workers(
        policy: BuiltinRoutingPolicy,
        worker_count: usize,
    ) -> Arc<SharedPickerState> {
        let state = Arc::new(SharedPickerState::new(policy.tracks_occupancy()));
        let ids = (0..worker_count as u64).collect::<Vec<_>>();
        state
            .0
            .lock()
            .reconcile(&RoutingInstances::new(ids), state.1);
        state
    }

    fn picker(policy: BuiltinRoutingPolicy, worker_count: usize) -> BuiltinWorkerPicker {
        BuiltinWorkerPicker {
            policy,
            picker: FastPicker::new(),
            state: state_with_workers(policy, worker_count),
        }
    }

    #[test]
    fn all_policies_use_constant_candidate_probes() {
        for policy in [
            BuiltinRoutingPolicy::RoundRobin,
            BuiltinRoutingPolicy::Random,
            BuiltinRoutingPolicy::PowerOfTwoChoices,
        ] {
            let picker = picker(policy, 65_536);
            let reservation = picker.select().unwrap();
            assert!(reservation.worker_id() < 65_536);
            assert!(
                picker.state.0.lock().selection_probes.get() <= 3,
                "{policy:?} selection must not scan the candidate table"
            );
        }
    }

    #[test]
    fn static_policies_do_not_initialize_load_tracking() {
        for policy in [
            BuiltinRoutingPolicy::RoundRobin,
            BuiltinRoutingPolicy::Random,
        ] {
            let picker = picker(policy, 1_024);
            let state = picker.state.0.lock();
            assert!(state.loads.is_empty());
        }
    }

    #[test]
    fn p2c_initializes_counts() {
        let picker = picker(BuiltinRoutingPolicy::PowerOfTwoChoices, 1_024);
        let state = picker.state.0.lock();
        assert_eq!(state.loads.len(), 1_024);
    }

    #[test]
    fn old_reservation_does_not_release_readded_worker() {
        let picker = picker(BuiltinRoutingPolicy::PowerOfTwoChoices, 1);
        let reservation = picker.select().unwrap();
        let empty = RoutingInstances::new(Vec::new());
        picker.state.update(&empty);
        let readded = RoutingInstances::new(vec![0]);
        picker.state.update(&readded);
        let current = picker.select().unwrap();
        drop(reservation);
        assert_eq!(picker.occupancy_for_test(0), 1);
        drop(current);
        assert_eq!(picker.occupancy_for_test(0), 0);
    }
}

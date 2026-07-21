// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::marker::PhantomData;

use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::{
    EngineEventBatch, Placement, PlacementDecision, PlacementEffects, PlacementPolicy,
    RequestIdentity, WorkerTopology,
};

#[derive(Debug)]
pub(in crate::replay::offline) struct AggregatedRoundRobin {
    next_worker: usize,
    next_rank_by_worker: FxHashMap<usize, u32>,
    dp_size: u32,
}

#[derive(Debug)]
pub(in crate::replay) struct AggregatedRoundRobinPlacement<Events: EngineEventBatch> {
    counter: AggregatedRoundRobin,
    workers: BTreeMap<usize, Vec<usize>>,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> AggregatedRoundRobinPlacement<Events> {
    pub(in crate::replay) fn new(dp_size: u32, workers: Vec<WorkerTopology>) -> Self {
        Self {
            counter: AggregatedRoundRobin::new(dp_size),
            workers: workers
                .into_iter()
                .map(|worker| (worker.worker_id, worker.scheduler_ids))
                .collect(),
            events: PhantomData,
        }
    }

    #[cfg(test)]
    pub(in crate::replay::offline) fn tracked_workers(&self) -> &FxHashMap<usize, u32> {
        self.counter.tracked_workers()
    }
}

impl<Request, Events> PlacementPolicy<Request> for AggregatedRoundRobinPlacement<Events>
where
    Request: RequestIdentity,
    Events: EngineEventBatch,
{
    type Metadata = ();
    type Observation = Events;

    #[inline]
    fn place(
        &mut self,
        request: &Request,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> Result<PlacementEffects> {
        let request_id = request
            .request_id()
            .ok_or_else(|| anyhow!("round-robin placement requires a request UUID"))?;
        let scheduler_id = self
            .counter
            .next(self.workers.keys().copied(), |worker_id, rank| {
                self.workers
                    .get(&worker_id)
                    .and_then(|ranks| ranks.get(rank as usize))
                    .copied()
            });
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                reported_overlap_tokens: 0,
                planner_cache_sample: None,
            }),
            released: Vec::new(),
        })
    }

    #[inline]
    fn observe(&mut self, _observation: Events, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    #[inline]
    fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
        false
    }

    #[inline]
    fn request_terminal(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    #[inline]
    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.insert(worker.worker_id, worker.scheduler_ids);
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        self.counter.worker_removed(worker.worker_id);
        Ok(Vec::new())
    }

    #[inline]
    fn topology_settled(&mut self, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

#[derive(Debug)]
pub(in crate::replay) struct PoolRoundRobinPlacement<Events: EngineEventBatch> {
    next: usize,
    workers: BTreeMap<usize, Vec<usize>>,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> PoolRoundRobinPlacement<Events> {
    pub(in crate::replay) fn new(workers: Vec<WorkerTopology>) -> Self {
        Self {
            next: 0,
            workers: workers
                .into_iter()
                .map(|worker| (worker.worker_id, worker.scheduler_ids))
                .collect(),
            events: PhantomData,
        }
    }
}

impl<Request, Events> PlacementPolicy<Request> for PoolRoundRobinPlacement<Events>
where
    Request: RequestIdentity,
    Events: EngineEventBatch,
{
    type Metadata = ();
    type Observation = Events;

    fn place(
        &mut self,
        request: &Request,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> Result<PlacementEffects> {
        let request_id = request
            .request_id()
            .ok_or_else(|| anyhow!("round-robin placement requires a request UUID"))?;
        let active_count = self.workers.values().map(Vec::len).sum::<usize>();
        let index = self.next % active_count;
        let scheduler_id = self
            .workers
            .values()
            .flat_map(|ranks| ranks.iter().copied())
            .nth(index)
            .expect("active round-robin pool must contain a scheduler");
        self.next = index + 1;
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                reported_overlap_tokens: 0,
                planner_cache_sample: None,
            }),
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _observation: Events, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
        false
    }

    fn request_terminal(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.insert(worker.worker_id, worker.scheduler_ids);
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

impl AggregatedRoundRobin {
    pub(in crate::replay::offline) fn new(dp_size: u32) -> Self {
        Self {
            next_worker: 0,
            next_rank_by_worker: FxHashMap::default(),
            dp_size: dp_size.max(1),
        }
    }

    pub(in crate::replay::offline) fn next(
        &mut self,
        mut active_workers: impl ExactSizeIterator<Item = usize>,
        rank_id: impl FnOnce(usize, u32) -> Option<usize>,
    ) -> usize {
        debug_assert!(
            active_workers.len() > 0,
            "no active workers for round-robin"
        );
        let index = self.next_worker % active_workers.len();
        self.next_worker = index + 1;
        let worker_id = active_workers
            .nth(index)
            .expect("active round-robin worker must exist at the selected index");
        let next_rank = self.next_rank_by_worker.entry(worker_id).or_default();
        let rank = *next_rank % self.dp_size;
        *next_rank = rank + 1;
        rank_id(worker_id, rank).expect("active worker must contain every configured DP rank")
    }

    pub(in crate::replay::offline) fn worker_removed(&mut self, worker_id: usize) {
        self.next_rank_by_worker.remove(&worker_id);
    }

    #[cfg(test)]
    pub(in crate::replay::offline) fn tracked_workers(&self) -> &FxHashMap<usize, u32> {
        &self.next_rank_by_worker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestRequest(Uuid);

    impl RequestIdentity for TestRequest {
        fn request_id(&self) -> Option<Uuid> {
            Some(self.0)
        }
    }

    fn scheduler_id(policy: &mut PoolRoundRobinPlacement<()>, ordinal: u128) -> usize {
        let effects = PlacementPolicy::<TestRequest>::place(
            policy,
            &TestRequest(Uuid::from_u128(ordinal)),
            (),
            None,
            0.0,
        )
        .unwrap();
        let PlacementDecision::Immediate(placement) = effects.decision else {
            panic!("round-robin placement must be immediate");
        };
        placement.scheduler_id
    }

    #[test]
    fn pool_rotation_preserves_position_after_topology_change() {
        let mut policy = PoolRoundRobinPlacement::<()>::new(vec![
            WorkerTopology {
                worker_id: 0,
                scheduler_ids: vec![10],
            },
            WorkerTopology {
                worker_id: 1,
                scheduler_ids: vec![11],
            },
            WorkerTopology {
                worker_id: 2,
                scheduler_ids: vec![12],
            },
        ]);

        assert_eq!(
            (1..=4)
                .map(|ordinal| scheduler_id(&mut policy, ordinal))
                .collect::<Vec<_>>(),
            vec![10, 11, 12, 10]
        );
        PlacementPolicy::<TestRequest>::worker_draining(
            &mut policy,
            WorkerTopology {
                worker_id: 2,
                scheduler_ids: vec![12],
            },
            0.0,
        )
        .unwrap();

        assert_eq!(scheduler_id(&mut policy, 5), 11);
    }
}

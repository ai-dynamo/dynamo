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
pub struct AggregatedRoundRobin {
    next_worker: usize,
    next_rank_by_worker: FxHashMap<usize, u32>,
    dp_size: u32,
}

#[derive(Debug)]
pub struct AggregatedRoundRobinPlacement<Events: EngineEventBatch> {
    counter: AggregatedRoundRobin,
    workers: BTreeMap<usize, Vec<usize>>,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> AggregatedRoundRobinPlacement<Events> {
    pub fn new(dp_size: u32, workers: Vec<WorkerTopology>) -> Self {
        let mut counter = AggregatedRoundRobin::new(dp_size);
        for worker in &workers {
            counter.worker_ready(worker.worker_id);
        }
        Self {
            counter,
            workers: workers
                .into_iter()
                .map(|worker| (worker.worker_id, worker.scheduler_ids))
                .collect(),
            events: PhantomData,
        }
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
        let scheduler_id = self.counter.next(
            self.workers.keys().copied(),
            request.preferred_dp_rank(),
            |worker_id, rank| {
                self.workers
                    .get(&worker_id)
                    .and_then(|ranks| ranks.get(rank as usize))
                    .copied()
            },
        )?;
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                reported_overlap_tokens: 0,
                cache_sample: None,
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
        self.counter.worker_ready(worker.worker_id);
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
pub struct PoolRoundRobinPlacement<Events: EngineEventBatch> {
    next: usize,
    workers: BTreeMap<usize, Vec<usize>>,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> PoolRoundRobinPlacement<Events> {
    pub fn new(workers: Vec<WorkerTopology>) -> Self {
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
        if active_count == 0 {
            return Err(anyhow!("no active workers for round-robin placement"));
        }
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
                cache_sample: None,
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
    pub fn new(dp_size: u32) -> Self {
        Self {
            next_worker: 0,
            next_rank_by_worker: FxHashMap::default(),
            dp_size: dp_size.max(1),
        }
    }

    pub(crate) fn next(
        &mut self,
        mut active_workers: impl ExactSizeIterator<Item = usize>,
        preferred_rank: Option<u32>,
        rank_id: impl FnOnce(usize, u32) -> Option<usize>,
    ) -> Result<usize> {
        if active_workers.len() == 0 {
            return Err(anyhow!("no active workers for round-robin placement"));
        }
        let index = self.next_worker % active_workers.len();
        self.next_worker = index + 1;
        let worker_id = active_workers
            .nth(index)
            .expect("active round-robin worker must exist at the selected index");
        let rank = match preferred_rank {
            Some(rank) if rank >= self.dp_size => {
                return Err(anyhow!(
                    "preferred attention-DP rank {rank} is out of range for dp_size {}",
                    self.dp_size
                ));
            }
            Some(rank) => rank,
            None => {
                let next_rank = self.next_rank_by_worker.entry(worker_id).or_default();
                let rank = *next_rank % self.dp_size;
                *next_rank = rank + 1;
                rank
            }
        };
        rank_id(worker_id, rank).ok_or_else(|| {
            anyhow!("logical worker {worker_id} does not expose preferred attention-DP rank {rank}")
        })
    }

    pub(crate) fn worker_removed(&mut self, worker_id: usize) {
        self.next_rank_by_worker.remove(&worker_id);
    }

    fn worker_ready(&mut self, worker_id: usize) {
        self.next_rank_by_worker.entry(worker_id).or_default();
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

    #[derive(Debug)]
    struct RankedTestRequest {
        id: Uuid,
        preferred_dp_rank: u32,
    }

    impl RequestIdentity for RankedTestRequest {
        fn request_id(&self) -> Option<Uuid> {
            Some(self.id)
        }

        fn preferred_dp_rank(&self) -> Option<u32> {
            Some(self.preferred_dp_rank)
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

    #[test]
    fn empty_pool_returns_an_error_instead_of_dividing_by_zero() {
        let mut policy = PoolRoundRobinPlacement::<()>::new(Vec::new());
        let error = PlacementPolicy::<TestRequest>::place(
            &mut policy,
            &TestRequest(Uuid::from_u128(1)),
            (),
            None,
            0.0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("no active workers"));
    }

    #[test]
    fn aggregated_round_robin_honors_authored_dp_rank_within_each_worker() {
        let mut policy = AggregatedRoundRobinPlacement::<()>::new(
            2,
            vec![
                WorkerTopology {
                    worker_id: 0,
                    scheduler_ids: vec![10, 11],
                },
                WorkerTopology {
                    worker_id: 1,
                    scheduler_ids: vec![20, 21],
                },
            ],
        );

        let place = |policy: &mut AggregatedRoundRobinPlacement<()>, ordinal, rank| {
            let effects = PlacementPolicy::<RankedTestRequest>::place(
                policy,
                &RankedTestRequest {
                    id: Uuid::from_u128(ordinal),
                    preferred_dp_rank: rank,
                },
                (),
                None,
                0.0,
            )?;
            let PlacementDecision::Immediate(placement) = effects.decision else {
                panic!("round-robin placement must be immediate");
            };
            Ok::<_, anyhow::Error>(placement.scheduler_id)
        };

        assert_eq!(place(&mut policy, 1, 1).unwrap(), 11);
        assert_eq!(place(&mut policy, 2, 1).unwrap(), 21);
        assert!(
            place(&mut policy, 3, 2)
                .unwrap_err()
                .to_string()
                .contains("out of range")
        );
    }
}

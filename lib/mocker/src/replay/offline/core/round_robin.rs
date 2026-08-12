// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::marker::PhantomData;

use anyhow::{Result, anyhow, bail};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::{
    EngineEventBatch, Placement, PlacementDecision, PlacementEffects, PlacementPolicy,
    PlacementRequest, WorkerTopology,
};

#[derive(Debug)]
struct RoundRobinWorker {
    scheduler_ids: Vec<usize>,
    taints: BTreeSet<String>,
}

#[derive(Debug)]
pub(in crate::replay::offline) struct AggregatedRoundRobin {
    next_worker: usize,
    next_rank_by_worker: FxHashMap<usize, u32>,
    dp_size: u32,
}

#[derive(Debug)]
pub(in crate::replay) struct AggregatedRoundRobinPlacement<Events: EngineEventBatch> {
    counter: AggregatedRoundRobin,
    workers: BTreeMap<usize, RoundRobinWorker>,
    configured_taints: BTreeMap<usize, BTreeSet<String>>,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> AggregatedRoundRobinPlacement<Events> {
    pub(in crate::replay) fn with_taints(
        dp_size: u32,
        workers: Vec<WorkerTopology>,
        worker_taints: &[HashSet<String>],
    ) -> Self {
        let mut counter = AggregatedRoundRobin::new(dp_size);
        for worker in &workers {
            counter.worker_ready(worker.worker_id);
        }
        let configured_taints = configured_taints(worker_taints);
        Self {
            counter,
            workers: workers
                .into_iter()
                .map(|worker| {
                    let taints = configured_taints
                        .get(&worker.worker_id)
                        .cloned()
                        .unwrap_or_default();
                    (
                        worker.worker_id,
                        RoundRobinWorker {
                            scheduler_ids: worker.scheduler_ids,
                            taints,
                        },
                    )
                })
                .collect(),
            configured_taints,
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
    Request: PlacementRequest,
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
        let required_taints = request.required_taints();
        let eligible_workers = self
            .workers
            .iter()
            .filter(|(_, worker)| {
                required_taints
                    .iter()
                    .all(|required| worker.taints.contains(required))
            })
            .map(|(&worker_id, _)| worker_id)
            .collect::<Vec<_>>();
        if eligible_workers.is_empty() {
            bail!(
                "round-robin placement has no active worker satisfying required taints {required_taints:?}"
            );
        }
        let scheduler_id = self
            .counter
            .next(eligible_workers.into_iter(), |worker_id, rank| {
                self.workers
                    .get(&worker_id)
                    .and_then(|worker| worker.scheduler_ids.get(rank as usize))
                    .copied()
            });
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                reported_overlap_tokens: None,
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
        self.counter.worker_ready(worker.worker_id);
        let taints = self
            .configured_taints
            .get(&worker.worker_id)
            .cloned()
            .unwrap_or_default();
        self.workers.insert(
            worker.worker_id,
            RoundRobinWorker {
                scheduler_ids: worker.scheduler_ids,
                taints,
            },
        );
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
    workers: BTreeMap<usize, RoundRobinWorker>,
    configured_taints: BTreeMap<usize, BTreeSet<String>>,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> PoolRoundRobinPlacement<Events> {
    pub(in crate::replay) fn with_taints(
        workers: Vec<WorkerTopology>,
        worker_taints: &[HashSet<String>],
    ) -> Self {
        let configured_taints = configured_taints(worker_taints);
        Self {
            next: 0,
            workers: workers
                .into_iter()
                .map(|worker| {
                    let taints = configured_taints
                        .get(&worker.worker_id)
                        .cloned()
                        .unwrap_or_default();
                    (
                        worker.worker_id,
                        RoundRobinWorker {
                            scheduler_ids: worker.scheduler_ids,
                            taints,
                        },
                    )
                })
                .collect(),
            configured_taints,
            events: PhantomData,
        }
    }
}

impl<Request, Events> PlacementPolicy<Request> for PoolRoundRobinPlacement<Events>
where
    Request: PlacementRequest,
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
        let required_taints = request.required_taints();
        let eligible_schedulers = self
            .workers
            .values()
            .filter(|worker| {
                required_taints
                    .iter()
                    .all(|required| worker.taints.contains(required))
            })
            .flat_map(|worker| worker.scheduler_ids.iter().copied())
            .collect::<Vec<_>>();
        if eligible_schedulers.is_empty() {
            bail!(
                "round-robin placement has no active worker satisfying required taints {required_taints:?}"
            );
        }
        let index = self.next % eligible_schedulers.len();
        let scheduler_id = eligible_schedulers
            .into_iter()
            .nth(index)
            .expect("validated eligible round-robin pool must contain a scheduler");
        self.next = index + 1;
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                reported_overlap_tokens: None,
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
        let taints = self
            .configured_taints
            .get(&worker.worker_id)
            .cloned()
            .unwrap_or_default();
        self.workers.insert(
            worker.worker_id,
            RoundRobinWorker {
                scheduler_ids: worker.scheduler_ids,
                taints,
            },
        );
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

fn configured_taints(worker_taints: &[HashSet<String>]) -> BTreeMap<usize, BTreeSet<String>> {
    worker_taints
        .iter()
        .enumerate()
        .map(|(worker_id, taints)| (worker_id, taints.iter().cloned().collect()))
        .collect()
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

    fn worker_ready(&mut self, worker_id: usize) {
        self.next_rank_by_worker.entry(worker_id).or_default();
    }

    #[cfg(test)]
    pub(in crate::replay::offline) fn tracked_workers(&self) -> &FxHashMap<usize, u32> {
        &self.next_rank_by_worker
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::offline::core::RequestIdentity;

    #[derive(Debug)]
    struct TestRequest(Uuid);

    impl RequestIdentity for TestRequest {
        fn request_id(&self) -> Option<Uuid> {
            Some(self.0)
        }
    }

    impl PlacementRequest for TestRequest {
        fn required_taints(&self) -> BTreeSet<String> {
            BTreeSet::new()
        }
    }

    #[derive(Debug)]
    struct ConstrainedTestRequest {
        id: Uuid,
        required_taints: BTreeSet<String>,
    }

    impl RequestIdentity for ConstrainedTestRequest {
        fn request_id(&self) -> Option<Uuid> {
            Some(self.id)
        }
    }

    impl PlacementRequest for ConstrainedTestRequest {
        fn required_taints(&self) -> BTreeSet<String> {
            self.required_taints.clone()
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
        let mut policy = PoolRoundRobinPlacement::<()>::with_taints(
            vec![
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
            ],
            &[],
        );

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

    fn constrained_request(ordinal: u128, taint: &str) -> ConstrainedTestRequest {
        ConstrainedTestRequest {
            id: Uuid::from_u128(ordinal),
            required_taints: BTreeSet::from([taint.to_string()]),
        }
    }

    fn immediate_scheduler<Policy>(
        policy: &mut Policy,
        request: &ConstrainedTestRequest,
    ) -> Result<usize>
    where
        Policy: PlacementPolicy<ConstrainedTestRequest, Metadata = (), Observation = ()>,
    {
        let effects = policy.place(request, (), None, 0.0)?;
        let PlacementDecision::Immediate(placement) = effects.decision else {
            panic!("round-robin placement must be immediate");
        };
        Ok(placement.scheduler_id)
    }

    #[test]
    fn aggregated_round_robin_enforces_taints_without_advancing_on_failure() -> Result<()> {
        let workers = vec![
            WorkerTopology {
                worker_id: 0,
                scheduler_ids: vec![10],
            },
            WorkerTopology {
                worker_id: 1,
                scheduler_ids: vec![11],
            },
        ];
        let taints = vec![
            HashSet::from(["general".to_string()]),
            HashSet::from(["trusted".to_string()]),
        ];
        let mut policy = AggregatedRoundRobinPlacement::<()>::with_taints(1, workers, &taints);

        let missing = constrained_request(10, "missing");
        assert!(
            immediate_scheduler(&mut policy, &missing)
                .unwrap_err()
                .to_string()
                .contains("no active worker")
        );
        // A failed placement is recoverable and does not advance RR state.
        assert!(immediate_scheduler(&mut policy, &missing).is_err());
        assert_eq!(
            immediate_scheduler(&mut policy, &constrained_request(11, "trusted"))?,
            11
        );
        Ok(())
    }

    #[test]
    fn pool_round_robin_enforces_taints_and_keeps_request_recoverable() -> Result<()> {
        let workers = vec![
            WorkerTopology {
                worker_id: 0,
                scheduler_ids: vec![10],
            },
            WorkerTopology {
                worker_id: 1,
                scheduler_ids: vec![11],
            },
        ];
        let taints = vec![
            HashSet::from(["general".to_string()]),
            HashSet::from(["trusted".to_string()]),
        ];
        let mut policy = PoolRoundRobinPlacement::<()>::with_taints(workers, &taints);

        let missing = constrained_request(20, "missing");
        assert!(immediate_scheduler(&mut policy, &missing).is_err());
        assert!(immediate_scheduler(&mut policy, &missing).is_err());
        assert_eq!(
            immediate_scheduler(&mut policy, &constrained_request(21, "trusted"))?,
            11
        );
        Ok(())
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit, controller-owned placement for interactive offline replay.

use std::collections::{BTreeMap, BTreeSet, VecDeque, hash_map::Entry};

use anyhow::{Result, anyhow, bail};
use rustc_hash::{FxHashMap, FxHashSet};
use uuid::Uuid;

use super::{
    EngineEventBatch, Placement, PlacementDecision, PlacementEffects, PlacementPolicy,
    PlacementRequest, WorkerTopology,
};
use crate::replay::offline::agg::AggregatedPlacement;
#[cfg(test)]
use crate::replay::offline::components::OfflineRouterSnapshot;
use crate::replay::offline::topology::{
    DEFAULT_REPLAY_POOL_ID, PoolRouter, ResolvedPoolWorker, WorkerTarget,
};

/// Placement policy whose decisions are supplied explicitly by the caller.
///
/// Requests with a pre-registered target route immediately. All other
/// requests remain pending until `assign` is called at the current virtual
/// timestamp. Pending requests survive worker lifecycle changes.
#[derive(Debug)]
pub(in crate::replay) struct ExternalPlacement<Events: EngineEventBatch> {
    workers: BTreeMap<(String, usize), (Vec<usize>, std::collections::BTreeSet<String>)>,
    pool_workers: BTreeMap<String, Vec<WorkerTarget>>,
    pool_routers: BTreeMap<String, PoolRouter>,
    pool_next: BTreeMap<String, usize>,
    preassigned: FxHashMap<Uuid, WorkerTarget>,
    constraints: FxHashMap<Uuid, BTreeSet<String>>,
    pending_order: VecDeque<Uuid>,
    pending: FxHashSet<Uuid>,
    events: std::marker::PhantomData<Events>,
}

impl<Events: EngineEventBatch> ExternalPlacement<Events> {
    pub(in crate::replay) fn new(
        workers: Vec<WorkerTopology>,
        worker_taints: &[std::collections::HashSet<String>],
    ) -> Self {
        let resolved = workers
            .iter()
            .enumerate()
            .map(|(worker_idx, worker)| ResolvedPoolWorker {
                target: WorkerTarget::default_pool(worker.worker_id, 0),
                engine_args: crate::common::protocols::MockEngineArgs::default(),
                tags: Default::default(),
                taints: worker_taints
                    .get(worker_idx)
                    .map(|taints| taints.iter().cloned().collect())
                    .unwrap_or_default(),
                capabilities: Default::default(),
                active: true,
                draining: false,
            })
            .collect();
        Self::new_pooled(
            workers,
            resolved,
            vec![(DEFAULT_REPLAY_POOL_ID.to_string(), PoolRouter::RoundRobin)],
        )
    }

    pub(in crate::replay::offline) fn new_pooled(
        workers: Vec<WorkerTopology>,
        resolved: Vec<ResolvedPoolWorker>,
        pool_routers: Vec<(String, PoolRouter)>,
    ) -> Self {
        debug_assert_eq!(workers.len(), resolved.len());
        let mut target_workers = BTreeMap::new();
        let mut pool_workers: BTreeMap<String, Vec<WorkerTarget>> = BTreeMap::new();
        for (worker, resolved) in workers.into_iter().zip(resolved) {
            if !resolved.active || resolved.draining {
                continue;
            }
            pool_workers
                .entry(resolved.target.pool_id.clone())
                .or_default()
                .push(resolved.target.clone());
            target_workers.insert(
                (resolved.target.pool_id, resolved.target.worker_id),
                (worker.scheduler_ids, resolved.taints),
            );
        }
        for workers in pool_workers.values_mut() {
            workers.sort_by_key(|worker| worker.worker_id);
        }
        Self {
            workers: target_workers,
            pool_workers,
            pool_routers: pool_routers.into_iter().collect(),
            pool_next: BTreeMap::new(),
            preassigned: FxHashMap::default(),
            constraints: FxHashMap::default(),
            pending_order: VecDeque::new(),
            pending: FxHashSet::default(),
            events: std::marker::PhantomData,
        }
    }

    fn resolve_target(&self, target: &WorkerTarget) -> Result<usize> {
        if !self.pool_workers.contains_key(&target.pool_id) {
            bail!(
                "interactive replay pool {:?} is unavailable",
                target.pool_id
            );
        }
        let (scheduler_ids, _) = self
            .workers
            .get(&(target.pool_id.clone(), target.worker_id))
            .ok_or_else(|| {
                anyhow!(
                    "interactive replay worker {} is not a member of pool {:?} or is unavailable",
                    target.worker_id,
                    target.pool_id
                )
            })?;
        scheduler_ids.get(target.dp_rank).copied().ok_or_else(|| {
            anyhow!(
                "interactive replay worker {} has no active DP rank {} (active ranks: {})",
                target.worker_id,
                target.dp_rank,
                scheduler_ids.len()
            )
        })
    }

    fn validate_required_taints(
        &self,
        required_taints: &BTreeSet<String>,
        target: &WorkerTarget,
    ) -> Result<()> {
        if required_taints.is_empty() {
            return Ok(());
        }
        let (_, taints) = self
            .workers
            .get(&(target.pool_id.clone(), target.worker_id))
            .ok_or_else(|| anyhow!("interactive replay target disappeared during validation"))?;
        if required_taints
            .iter()
            .all(|required| taints.contains(required))
        {
            Ok(())
        } else {
            bail!(
                "interactive replay worker {} in pool {:?} does not satisfy required taints {:?}",
                target.worker_id,
                target.pool_id,
                required_taints
            )
        }
    }

    fn validate_constraints(&self, request_id: Uuid, target: &WorkerTarget) -> Result<()> {
        let Some(required_taints) = self.constraints.get(&request_id) else {
            return Ok(());
        };
        self.validate_required_taints(required_taints, target)
    }

    pub(in crate::replay) fn preassign(
        &mut self,
        request_id: Uuid,
        target: WorkerTarget,
        required_taints: BTreeSet<String>,
    ) -> Result<()> {
        self.resolve_target(&target)?;
        self.validate_required_taints(&required_taints, &target)?;
        match self.preassigned.entry(request_id) {
            Entry::Vacant(entry) => {
                entry.insert(target);
            }
            Entry::Occupied(_) => {
                bail!("interactive replay request {request_id} already has a placement target");
            }
        }
        self.constraints.insert(request_id, required_taints);
        Ok(())
    }

    pub(in crate::replay) fn assign(
        &mut self,
        request_id: Uuid,
        target: &WorkerTarget,
    ) -> Result<Placement> {
        if !self.pending.contains(&request_id) {
            bail!("interactive replay request {request_id} is not awaiting placement");
        }
        let scheduler_id = self.resolve_target(target).and_then(|scheduler_id| {
            self.validate_constraints(request_id, target)?;
            Ok(scheduler_id)
        })?;
        self.pending.remove(&request_id);
        self.constraints.remove(&request_id);
        self.pending_order
            .retain(|candidate| *candidate != request_id);
        Ok(Placement {
            request_id,
            scheduler_id,
            // The runtime decorates external placements from the selected
            // worker's committed scheduler state immediately before dispatch.
            reported_overlap_tokens: None,
            planner_cache_sample: None,
        })
    }

    pub(in crate::replay) fn assign_pool(
        &mut self,
        request_id: Uuid,
        pool_id: &str,
    ) -> Result<(Placement, WorkerTarget)> {
        if !self.pending.contains(&request_id) {
            bail!("interactive replay request {request_id} is not awaiting placement");
        }
        if self.pool_routers.get(pool_id) != Some(&PoolRouter::RoundRobin) {
            bail!("interactive replay pool {pool_id:?} has no supported internal router");
        }
        let workers = self
            .pool_workers
            .get(pool_id)
            .ok_or_else(|| anyhow!("interactive replay pool {pool_id:?} is unavailable"))?;
        let eligible_count = workers
            .iter()
            .filter(|target| self.validate_constraints(request_id, target).is_ok())
            .count();
        if eligible_count == 0 {
            bail!(
                "interactive replay pool {pool_id:?} has no worker eligible for request {request_id}"
            );
        }
        let cursor = self.pool_next.get(pool_id).copied().unwrap_or_default();
        let selected = cursor % eligible_count;
        let target = workers
            .iter()
            .filter(|target| self.validate_constraints(request_id, target).is_ok())
            .nth(selected)
            .expect("eligible worker count and selected rank diverged")
            .clone();
        self.pool_next
            .insert(pool_id.to_string(), (cursor + 1) % eligible_count);
        let placement = self.assign(request_id, &target)?;
        Ok((placement, target))
    }

    pub(in crate::replay) fn pending_ids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.pending_order.iter().copied()
    }

    pub(in crate::replay) fn is_pending(&self, request_id: Uuid) -> bool {
        self.pending.contains(&request_id)
    }

    pub(in crate::replay) fn cancel(&mut self, request_id: Uuid) -> bool {
        self.preassigned.remove(&request_id);
        self.constraints.remove(&request_id);
        let removed = self.pending.remove(&request_id);
        if removed {
            self.pending_order
                .retain(|candidate| *candidate != request_id);
        }
        removed
    }

    fn register_ready_target(
        &mut self,
        worker: WorkerTopology,
        target: &WorkerTarget,
    ) -> Result<()> {
        anyhow::ensure!(
            target.dp_rank == 0,
            "interactive replay lifecycle target must identify logical worker rank zero, got {target:?}"
        );
        let key = (target.pool_id.clone(), target.worker_id);
        anyhow::ensure!(
            !self.workers.contains_key(&key),
            "interactive replay lifecycle target {target:?} collides with an available worker"
        );
        self.pool_routers
            .entry(target.pool_id.clone())
            .or_insert(PoolRouter::RoundRobin);
        let pool_workers = self.pool_workers.entry(target.pool_id.clone()).or_default();
        pool_workers.push(target.clone());
        pool_workers.sort_by_key(|worker| worker.worker_id);
        self.workers
            .insert(key, (worker.scheduler_ids, Default::default()));
        Ok(())
    }

    fn unregister_target(&mut self, target: &WorkerTarget) {
        self.workers
            .remove(&(target.pool_id.clone(), target.worker_id));
        if let Some(workers) = self.pool_workers.get_mut(&target.pool_id) {
            workers.retain(|candidate| candidate.worker_id != target.worker_id);
        }
    }
}

impl<Events, Request> PlacementPolicy<Request> for ExternalPlacement<Events>
where
    Events: EngineEventBatch,
    Request: PlacementRequest,
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
            .ok_or_else(|| anyhow!("external placement requires a request UUID"))?;
        let required_taints = request.required_taints();
        if let Some(preassigned) = self.constraints.get(&request_id) {
            if preassigned != &required_taints {
                bail!(
                    "interactive replay request {request_id} routing constraints changed after authored preassignment"
                );
            }
        } else {
            self.constraints.insert(request_id, required_taints);
        }
        let decision = if let Some(target) = self.preassigned.remove(&request_id) {
            let scheduler_id = self.resolve_target(&target).and_then(|scheduler_id| {
                self.validate_constraints(request_id, &target)?;
                Ok(scheduler_id)
            })?;
            self.constraints.remove(&request_id);
            PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                // The runtime decorates authored preassignments from the
                // selected worker's committed scheduler state.
                reported_overlap_tokens: None,
                planner_cache_sample: None,
            })
        } else {
            if !self.pending.insert(request_id) {
                bail!("interactive replay request {request_id} is already awaiting placement");
            }
            self.pending_order.push_back(request_id);
            PlacementDecision::Queued
        };
        Ok(PlacementEffects {
            decision,
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _observation: Events, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.cancel(request_id)
    }

    fn request_terminal(&mut self, request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        self.cancel(request_id);
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn next_pending_request_id(&self) -> Option<Uuid> {
        self.pending_order.front().copied()
    }

    fn worker_ready(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        let target = WorkerTarget::default_pool(worker.worker_id, 0);
        self.register_ready_target(worker, &target)?;
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.unregister_target(&WorkerTarget::default_pool(worker.worker_id, 0));
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.unregister_target(&WorkerTarget::default_pool(worker.worker_id, 0));
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

impl<Events: EngineEventBatch> AggregatedPlacement<Events, ()> for ExternalPlacement<Events> {
    fn worker_ready_authored(
        &mut self,
        worker: WorkerTopology,
        target: &WorkerTarget,
        _now_ms: f64,
    ) -> Result<Vec<Placement>> {
        self.register_ready_target(worker, target)?;
        Ok(Vec::new())
    }

    fn worker_draining_authored(
        &mut self,
        _worker: WorkerTopology,
        target: &WorkerTarget,
        _now_ms: f64,
    ) -> Result<Vec<Placement>> {
        self.unregister_target(target);
        Ok(Vec::new())
    }

    fn worker_removed_authored(
        &mut self,
        _worker: WorkerTopology,
        target: &WorkerTarget,
        _now_ms: f64,
    ) -> Result<Vec<Placement>> {
        self.unregister_target(target);
        Ok(Vec::new())
    }

    #[cfg(test)]
    fn is_router(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn debug_router_snapshot(&self, _now_ms: f64) -> Option<OfflineRouterSnapshot> {
        None
    }
}

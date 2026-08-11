// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit, controller-owned placement for interactive offline replay.

use std::collections::{BTreeMap, VecDeque, hash_map::Entry};

use anyhow::{Result, anyhow, bail};
use rustc_hash::{FxHashMap, FxHashSet};
use uuid::Uuid;

use super::{
    EngineEventBatch, Placement, PlacementDecision, PlacementEffects, PlacementPolicy,
    RequestIdentity, WorkerTopology,
};
use crate::replay::offline::agg::AggregatedPlacement;
#[cfg(test)]
use crate::replay::offline::components::OfflineRouterSnapshot;
use crate::replay::offline::interactive::WorkerTarget;

/// Placement policy whose decisions are supplied explicitly by the caller.
///
/// Requests with a pre-registered target route immediately. All other
/// requests remain pending until `assign` is called at the current virtual
/// timestamp. Pending requests survive worker lifecycle changes.
#[derive(Debug)]
pub(in crate::replay) struct ExternalPlacement<Events: EngineEventBatch> {
    workers: BTreeMap<usize, Vec<usize>>,
    preassigned: FxHashMap<Uuid, WorkerTarget>,
    pending_order: VecDeque<Uuid>,
    pending: FxHashSet<Uuid>,
    events: std::marker::PhantomData<Events>,
}

impl<Events: EngineEventBatch> ExternalPlacement<Events> {
    pub(in crate::replay) fn new(workers: Vec<WorkerTopology>) -> Self {
        Self {
            workers: workers
                .into_iter()
                .map(|worker| (worker.worker_id, worker.scheduler_ids))
                .collect(),
            preassigned: FxHashMap::default(),
            pending_order: VecDeque::new(),
            pending: FxHashSet::default(),
            events: std::marker::PhantomData,
        }
    }

    fn resolve_target(&self, target: WorkerTarget) -> Result<usize> {
        let scheduler_ids = self.workers.get(&target.worker_id).ok_or_else(|| {
            anyhow!(
                "interactive replay worker {} is unavailable or draining",
                target.worker_id
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

    pub(in crate::replay) fn preassign(
        &mut self,
        request_id: Uuid,
        target: WorkerTarget,
    ) -> Result<()> {
        self.resolve_target(target)?;
        match self.preassigned.entry(request_id) {
            Entry::Vacant(entry) => {
                entry.insert(target);
            }
            Entry::Occupied(_) => {
                bail!("interactive replay request {request_id} already has a placement target");
            }
        }
        Ok(())
    }

    pub(in crate::replay) fn assign(
        &mut self,
        request_id: Uuid,
        target: WorkerTarget,
    ) -> Result<Placement> {
        if !self.pending.remove(&request_id) {
            bail!("interactive replay request {request_id} is not awaiting placement");
        }
        let scheduler_id = match self.resolve_target(target) {
            Ok(scheduler_id) => scheduler_id,
            Err(error) => {
                self.pending.insert(request_id);
                return Err(error);
            }
        };
        self.pending_order
            .retain(|candidate| *candidate != request_id);
        Ok(Placement {
            request_id,
            scheduler_id,
            reported_overlap_tokens: 0,
            planner_cache_sample: None,
        })
    }

    pub(in crate::replay) fn pending_ids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.pending_order.iter().copied()
    }

    pub(in crate::replay) fn cancel(&mut self, request_id: Uuid) -> bool {
        self.preassigned.remove(&request_id);
        let removed = self.pending.remove(&request_id);
        if removed {
            self.pending_order
                .retain(|candidate| *candidate != request_id);
        }
        removed
    }
}

impl<Request, Events> PlacementPolicy<Request> for ExternalPlacement<Events>
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
            .ok_or_else(|| anyhow!("external placement requires a request UUID"))?;
        let decision = if let Some(target) = self.preassigned.remove(&request_id) {
            PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id: self.resolve_target(target)?,
                reported_overlap_tokens: 0,
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

impl<Events: EngineEventBatch> AggregatedPlacement<Events, ()> for ExternalPlacement<Events> {
    #[cfg(test)]
    fn is_router(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn debug_router_snapshot(&self, _now_ms: f64) -> Option<OfflineRouterSnapshot> {
        None
    }
}

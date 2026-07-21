// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-neutral contracts for offline replay.
//!
//! Runtime adapters and concrete policies live one level up so this directory
//! can later become a standalone crate without deployment-specific APIs.

use anyhow::Result;
use uuid::Uuid;

pub(crate) mod round_robin;

pub(crate) trait RequestIdentity {
    fn request_id(&self) -> Option<Uuid>;
}

#[derive(Debug)]
pub(crate) struct ReadyArrival<Request, Metadata> {
    pub(in crate::replay::offline) request: Request,
    pub(in crate::replay::offline) arrival_time_ms: f64,
    pub(in crate::replay::offline) metadata: Metadata,
    pub(in crate::replay::offline) session_id: Option<String>,
    pub(in crate::replay::offline) turn_index: Option<usize>,
}

pub(crate) trait AdmissionSource {
    type Request;
    type Metadata;

    fn next_ready_time_ms(&mut self) -> Option<f64>;
    fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>>;
    fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()>;
    fn on_terminal(&mut self, request_id: Uuid, now_ms: f64, rejected: bool) -> Result<()>;
    fn is_drained(&self) -> bool;
    fn total_requests(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannerCacheSample {
    pub(in crate::replay::offline) overlap_blocks: u32,
    pub(in crate::replay::offline) isl_blocks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    pub(in crate::replay::offline) request_id: Uuid,
    pub(in crate::replay::offline) scheduler_id: usize,
    pub(in crate::replay::offline) reported_overlap_tokens: usize,
    pub(in crate::replay::offline) planner_cache_sample: Option<PlannerCacheSample>,
}

#[derive(Debug)]
pub(crate) enum PlacementDecision {
    Immediate(Placement),
    Queued,
}

#[derive(Debug)]
pub(crate) struct PlacementEffects {
    pub(crate) decision: PlacementDecision,
    pub(crate) released: Vec<Placement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkerTopology {
    pub(crate) worker_id: usize,
    pub(crate) scheduler_ids: Vec<usize>,
}

pub(crate) trait PlacementPolicy<Request> {
    type Metadata;
    type Observation;

    fn place(
        &mut self,
        request: &Request,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects>;
    fn observe(&mut self, observation: Self::Observation, now_ms: f64) -> Result<Vec<Placement>>;
    fn cancel_pending(&mut self, request_id: Uuid) -> bool;
    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>>;
    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>>;
    fn pending_count(&self) -> usize;
    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>>;
    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct EngineProgress {
    pub(in crate::replay::offline) made_progress: bool,
    pub(in crate::replay::offline) had_raw_observations: bool,
}

pub(crate) trait EngineEventBatch: Default {
    fn is_empty(&self) -> bool;
    fn append(&mut self, other: Self);
}

impl EngineEventBatch for () {
    #[inline]
    fn is_empty(&self) -> bool {
        true
    }

    #[inline]
    fn append(&mut self, _other: Self) {}
}

pub(crate) trait EngineObservation<Raw> {
    type Batch: EngineEventBatch;

    const CAPTURE_RAW: bool;

    fn observe(raw: Raw) -> Self::Batch;
}

#[derive(Debug, Default)]
pub(crate) struct NoEngineEvents;

impl<Raw> EngineObservation<Raw> for NoEngineEvents {
    type Batch = ();

    const CAPTURE_RAW: bool = false;

    #[inline]
    fn observe(_raw: Raw) -> Self::Batch {}
}

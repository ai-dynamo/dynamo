// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-neutral contracts for offline replay.
//!
//! Runtime adapters and concrete policies live one level up so this directory
//! can later become a standalone crate without deployment-specific APIs.

use anyhow::Result;
use uuid::Uuid;

pub mod round_robin;

pub trait RequestIdentity {
    fn request_id(&self) -> Option<Uuid>;

    /// An authored attention-DP preference. Policies that do not provide
    /// rank-affinity semantics may ignore it.
    fn preferred_dp_rank(&self) -> Option<u32> {
        None
    }
}

#[derive(Debug)]
pub struct ReadyArrival<Request, Metadata> {
    pub request: Request,
    pub arrival_time_ms: f64,
    pub metadata: Metadata,
    pub session_id: Option<String>,
    pub turn_index: Option<usize>,
}

pub trait AdmissionSource {
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
pub struct PlacementCacheSample {
    pub overlap_blocks: u32,
    pub isl_blocks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub request_id: Uuid,
    pub scheduler_id: usize,
    pub reported_overlap_tokens: usize,
    pub cache_sample: Option<PlacementCacheSample>,
}

#[derive(Debug)]
pub enum PlacementDecision {
    Immediate(Placement),
    Queued,
}

#[derive(Debug)]
pub struct PlacementEffects {
    pub decision: PlacementDecision,
    pub released: Vec<Placement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerTopology {
    pub worker_id: usize,
    pub scheduler_ids: Vec<usize>,
}

pub trait PlacementPolicy<Request> {
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
    pub(crate) made_progress: bool,
    pub(crate) had_raw_observations: bool,
}

pub trait EngineEventBatch: Default {
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

#[derive(Debug, Default)]
pub struct NoEngineEvents;

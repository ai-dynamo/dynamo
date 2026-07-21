// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::{Result, bail};
#[cfg(test)]
use rustc_hash::FxHashMap;
#[cfg(test)]
use uuid::Uuid;

use super::{KvRouterPlacement, ReplayKvRouterConfig};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::WorkloadDriver;
use crate::replay::offline::agg::{AggRuntimeImpl, AggregatedPlacement};
#[cfg(test)]
use crate::replay::offline::components::OfflineRouterSnapshot;
use crate::replay::offline::components::{
    AdmissionQueue, KvReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation, ReplayMode,
};
#[cfg(test)]
use crate::replay::offline::core::round_robin::AggregatedRoundRobinPlacement;
use crate::replay::offline::core::{EngineEventBatch, WorkerTopology};
#[cfg(test)]
use crate::replay::offline::core::{Placement, PlacementEffects, PlacementPolicy};
use crate::replay::offline::extensions::kv_events::{RouterEventBatch, RouterEventObservation};
use crate::replay::{ReplayPrefillLoadEstimator, ReplayRouterMode};

pub(in crate::replay) trait ConfiguredAggregatedPlacement<Events, Metadata>:
    AggregatedPlacement<Events, Metadata>
where
    Events: EngineEventBatch,
    Metadata: ReplayAdmissionMetadata,
{
    fn create(
        args: &MockEngineArgs,
        router_config: Option<ReplayKvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self>;
}

impl AggregatedPlacement<RouterEventBatch, KvReplayMetadata> for KvRouterPlacement {
    #[cfg(test)]
    #[inline]
    fn is_router(&self) -> bool {
        true
    }

    #[cfg(test)]
    fn debug_router_snapshot(&self, now_ms: f64) -> Option<OfflineRouterSnapshot> {
        Some(self.debug_snapshot(now_ms))
    }
}

impl ConfiguredAggregatedPlacement<RouterEventBatch, KvReplayMetadata> for KvRouterPlacement {
    fn create(
        args: &MockEngineArgs,
        router_config: Option<ReplayKvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
        _topology: Vec<WorkerTopology>,
    ) -> Result<Self> {
        if router_mode != ReplayRouterMode::KvRouter {
            bail!("KV replay composition received round-robin mode");
        }
        Self::new(args, router_config, prefill_load_estimator, num_workers)
    }
}

#[cfg(test)]
// The adaptive test adapter is not instantiated in production hot paths.
#[allow(clippy::large_enum_variant)]
pub(in crate::replay) enum AdaptiveAggPlacement {
    RoundRobin(AggregatedRoundRobinPlacement<RouterEventBatch>),
    Kv(KvRouterPlacement),
}

#[cfg(test)]
impl AdaptiveAggPlacement {
    pub(in crate::replay::offline) fn tracked_round_robin_workers(&self) -> &FxHashMap<usize, u32> {
        let Self::RoundRobin(policy) = self else {
            panic!("expected round-robin placement");
        };
        policy.tracked_workers()
    }
}

#[cfg(test)]
impl PlacementPolicy<DirectRequest> for AdaptiveAggPlacement {
    type Metadata = KvReplayMetadata;
    type Observation = RouterEventBatch;

    fn place(
        &mut self,
        request: &DirectRequest,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        match self {
            Self::RoundRobin(policy) => policy.place(request, (), session_id, now_ms),
            Self::Kv(policy) => policy.place(request, metadata, session_id, now_ms),
        }
    }

    fn observe(&mut self, observation: Self::Observation, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::observe(policy, observation, now_ms)
            }
            Self::Kv(policy) => policy.observe(observation, now_ms),
        }
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::cancel_pending(policy, request_id)
            }
            Self::Kv(policy) => policy.cancel_pending(request_id),
        }
    }

    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::request_terminal(policy, request_id, now_ms)
            }
            Self::Kv(policy) => policy.request_terminal(request_id, now_ms),
        }
    }

    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::prefill_completed(policy, request_id, now_ms)
            }
            Self::Kv(policy) => policy.prefill_completed(request_id, now_ms),
        }
    }

    fn pending_count(&self) -> usize {
        match self {
            Self::RoundRobin(policy) => PlacementPolicy::<DirectRequest>::pending_count(policy),
            Self::Kv(policy) => policy.pending_count(),
        }
    }

    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::worker_ready(policy, worker, now_ms)
            }
            Self::Kv(policy) => policy.worker_ready(worker, now_ms),
        }
    }

    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::worker_draining(policy, worker, now_ms)
            }
            Self::Kv(policy) => policy.worker_draining(worker, now_ms),
        }
    }

    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::worker_removed(policy, worker, now_ms)
            }
            Self::Kv(policy) => policy.worker_removed(worker, now_ms),
        }
    }

    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<DirectRequest>::topology_settled(policy, now_ms)
            }
            Self::Kv(policy) => policy.topology_settled(now_ms),
        }
    }
}

#[cfg(test)]
impl AggregatedPlacement<RouterEventBatch, KvReplayMetadata> for AdaptiveAggPlacement {
    fn is_router(&self) -> bool {
        matches!(self, Self::Kv(_))
    }

    fn debug_router_snapshot(&self, now_ms: f64) -> Option<OfflineRouterSnapshot> {
        match self {
            Self::RoundRobin(_) => None,
            Self::Kv(policy) => Some(policy.debug_snapshot(now_ms)),
        }
    }
}

#[cfg(test)]
impl ConfiguredAggregatedPlacement<RouterEventBatch, KvReplayMetadata> for AdaptiveAggPlacement {
    fn create(
        args: &MockEngineArgs,
        router_config: Option<ReplayKvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self> {
        match router_mode {
            ReplayRouterMode::RoundRobin => Ok(Self::RoundRobin(
                AggregatedRoundRobinPlacement::new(args.dp_size, topology),
            )),
            ReplayRouterMode::KvRouter => Ok(Self::Kv(KvRouterPlacement::new(
                args,
                router_config,
                prefill_load_estimator,
                num_workers,
            )?)),
        }
    }
}

#[cfg(not(test))]
type KvAggRuntime = AggRuntimeImpl<KvRouterPlacement, RouterEventObservation, KvReplayMetadata>;

#[cfg(test)]
pub(in crate::replay) type AggRuntime =
    AggRuntimeImpl<AdaptiveAggPlacement, RouterEventObservation, KvReplayMetadata>;
#[cfg(not(test))]
pub(in crate::replay) type AggRuntime = KvAggRuntime;

impl<PlacementPolicyImpl, Observation, Metadata>
    AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: AggregatedPlacement<Observation::Batch, Metadata>,
{
    /// Create a KV-configurable aggregated runtime seeded from an explicit request queue.
    pub(in crate::replay) fn new(
        args: &MockEngineArgs,
        router_config: Option<ReplayKvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self>
    where
        PlacementPolicyImpl: ConfiguredAggregatedPlacement<Observation::Batch, Metadata>,
    {
        Self::new_with_source(
            args,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::<Metadata>::new_requests(pending, mode),
            num_workers,
            router_mode,
        )
    }

    /// Create a KV-configurable aggregated runtime backed by a workload driver.
    pub(in crate::replay) fn new_workload(
        args: &MockEngineArgs,
        router_config: Option<ReplayKvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        driver: WorkloadDriver,
        num_workers: usize,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self>
    where
        PlacementPolicyImpl: ConfiguredAggregatedPlacement<Observation::Batch, Metadata>,
    {
        Self::new_with_source(
            args,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::<Metadata>::new_workload(driver, mode),
            num_workers,
            router_mode,
        )
    }

    fn new_with_source(
        args: &MockEngineArgs,
        router_config: Option<ReplayKvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        admission: AdmissionQueue<Metadata>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
    ) -> Result<Self>
    where
        PlacementPolicyImpl: ConfiguredAggregatedPlacement<Observation::Batch, Metadata>,
    {
        Self::new_composed(args, admission, num_workers, |args, topology| {
            PlacementPolicyImpl::create(
                args,
                router_config,
                prefill_load_estimator,
                num_workers,
                router_mode,
                topology,
            )
        })
    }
}

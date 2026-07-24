// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::{Result, bail};
use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};
#[cfg(test)]
use uuid::Uuid;

use super::KvRouterPlacement;
use crate::common::protocols::{DirectRequest, MockEngineArgs};
#[cfg(test)]
use crate::loadgen::ReplayRequestPayload;
use crate::loadgen::WorkloadDriver;
#[cfg(test)]
use crate::replay::offline::components::OfflineRouterSnapshot;
use crate::replay::offline::components::{
    AdmissionQueue, KvReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation, ReplayMode,
};
#[cfg(test)]
use crate::replay::offline::core::round_robin::PoolRoundRobinPlacement;
use crate::replay::offline::core::{EngineEventBatch, WorkerTopology};
#[cfg(test)]
use crate::replay::offline::core::{Placement, PlacementEffects, PlacementPolicy};
use crate::replay::offline::disagg::{DisaggRuntimeImpl, PoolPlacement};
use crate::replay::offline::extensions::kv_events::{RouterEventBatch, RouterEventObservation};
use crate::replay::{OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode};

pub(in crate::replay) trait ConfiguredPoolPlacement<Events, Metadata>:
    PoolPlacement<Events, Metadata>
where
    Events: EngineEventBatch,
    Metadata: ReplayAdmissionMetadata,
{
    fn create(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self>;
}

impl PoolPlacement<RouterEventBatch, KvReplayMetadata> for KvRouterPlacement {
    #[inline]
    fn is_router(&self) -> bool {
        true
    }
}

impl ConfiguredPoolPlacement<RouterEventBatch, KvReplayMetadata> for KvRouterPlacement {
    fn create(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
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
pub(in crate::replay) enum AdaptivePoolPlacement {
    RoundRobin(PoolRoundRobinPlacement<RouterEventBatch>),
    Kv(KvRouterPlacement),
}

#[cfg(test)]
impl AdaptivePoolPlacement {
    pub(in crate::replay::offline) fn debug_snapshot(&self, now_ms: f64) -> OfflineRouterSnapshot {
        match self {
            Self::RoundRobin(_) => panic!("expected KV router placement"),
            Self::Kv(policy) => policy.debug_snapshot(now_ms),
        }
    }
}

#[cfg(test)]
impl PlacementPolicy<ReplayRequestPayload> for AdaptivePoolPlacement {
    type Metadata = KvReplayMetadata;
    type Observation = RouterEventBatch;

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
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
                PlacementPolicy::<ReplayRequestPayload>::observe(policy, observation, now_ms)
            }
            Self::Kv(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::observe(policy, observation, now_ms)
            }
        }
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::cancel_pending(policy, request_id)
            }
            Self::Kv(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::cancel_pending(policy, request_id)
            }
        }
    }

    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => PlacementPolicy::<ReplayRequestPayload>::request_terminal(
                policy, request_id, now_ms,
            ),
            Self::Kv(policy) => PlacementPolicy::<ReplayRequestPayload>::request_terminal(
                policy, request_id, now_ms,
            ),
        }
    }

    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => PlacementPolicy::<ReplayRequestPayload>::prefill_completed(
                policy, request_id, now_ms,
            ),
            Self::Kv(policy) => PlacementPolicy::<ReplayRequestPayload>::prefill_completed(
                policy, request_id, now_ms,
            ),
        }
    }

    fn pending_count(&self) -> usize {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::pending_count(policy)
            }
            Self::Kv(policy) => PlacementPolicy::<ReplayRequestPayload>::pending_count(policy),
        }
    }

    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::worker_ready(policy, worker, now_ms)
            }
            Self::Kv(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::worker_ready(policy, worker, now_ms)
            }
        }
    }

    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::worker_draining(policy, worker, now_ms)
            }
            Self::Kv(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::worker_draining(policy, worker, now_ms)
            }
        }
    }

    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::worker_removed(policy, worker, now_ms)
            }
            Self::Kv(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::worker_removed(policy, worker, now_ms)
            }
        }
    }

    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        match self {
            Self::RoundRobin(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::topology_settled(policy, now_ms)
            }
            Self::Kv(policy) => {
                PlacementPolicy::<ReplayRequestPayload>::topology_settled(policy, now_ms)
            }
        }
    }
}

#[cfg(test)]
impl PoolPlacement<RouterEventBatch, KvReplayMetadata> for AdaptivePoolPlacement {
    fn is_router(&self) -> bool {
        matches!(self, Self::Kv(_))
    }
}

#[cfg(test)]
impl ConfiguredPoolPlacement<RouterEventBatch, KvReplayMetadata> for AdaptivePoolPlacement {
    fn create(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        router_mode: ReplayRouterMode,
        topology: Vec<WorkerTopology>,
    ) -> Result<Self> {
        match router_mode {
            ReplayRouterMode::RoundRobin => {
                Ok(Self::RoundRobin(PoolRoundRobinPlacement::new(topology)))
            }
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
type KvDisaggRuntime =
    DisaggRuntimeImpl<KvRouterPlacement, RouterEventObservation, KvReplayMetadata>;

#[cfg(test)]
pub(in crate::replay) type DisaggRuntime =
    DisaggRuntimeImpl<AdaptivePoolPlacement, RouterEventObservation, KvReplayMetadata>;
#[cfg(not(test))]
pub(in crate::replay) type DisaggRuntime = KvDisaggRuntime;

impl<PlacementPolicyImpl, Observation, Metadata>
    DisaggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: PoolPlacement<Observation::Batch, Metadata>,
{
    /// Create a KV-configurable disaggregated runtime seeded from a request queue.
    pub(in crate::replay) fn new(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        pending: VecDeque<DirectRequest>,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self>
    where
        PlacementPolicyImpl: ConfiguredPoolPlacement<Observation::Batch, Metadata>,
    {
        Self::new_with_source(
            config,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::<Metadata>::new_requests(pending, mode),
            router_mode,
        )
    }

    /// Create a KV-configurable disaggregated runtime backed by a workload driver.
    pub(in crate::replay) fn new_workload(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        driver: WorkloadDriver,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self>
    where
        PlacementPolicyImpl: ConfiguredPoolPlacement<Observation::Batch, Metadata>,
    {
        Self::new_with_source(
            config,
            router_config,
            prefill_load_estimator,
            AdmissionQueue::<Metadata>::new_workload(driver, mode),
            router_mode,
        )
    }

    fn new_with_source(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        admission: AdmissionQueue<Metadata>,
        router_mode: ReplayRouterMode,
    ) -> Result<Self>
    where
        PlacementPolicyImpl: ConfiguredPoolPlacement<Observation::Batch, Metadata>,
    {
        let (prefill_router_config, decode_router_config) = match router_mode {
            ReplayRouterMode::RoundRobin => (None, None),
            ReplayRouterMode::KvRouter => (
                Some(derive_prefill_router_config(
                    &config.prefill_args,
                    router_config.clone(),
                )),
                Some(derive_decode_router_config(
                    &config.decode_args,
                    router_config,
                )),
            ),
        };

        let prefill_capture_kv =
            router_mode == ReplayRouterMode::KvRouter && Observation::CAPTURE_RAW;
        Self::new_composed(
            config,
            admission,
            prefill_capture_kv,
            false,
            false,
            |args, topology| {
                PlacementPolicyImpl::create(
                    args,
                    prefill_router_config,
                    prefill_load_estimator,
                    config.num_prefill_workers,
                    router_mode,
                    topology,
                )
            },
            |args, topology| {
                PlacementPolicyImpl::create(
                    args,
                    decode_router_config,
                    None,
                    config.num_decode_workers,
                    router_mode,
                    topology,
                )
            },
        )
    }
}

fn base_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = router_config.unwrap_or_default();
    if let Some(policy) = args.router_queue_policy {
        config.router_queue_policy = policy;
    }
    config
}

pub(in crate::replay::offline) fn derive_prefill_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = base_router_config(args, router_config);
    config.router_track_active_blocks = false;
    config
}

pub(in crate::replay::offline) fn derive_decode_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = base_router_config(args, router_config);
    config.overlap_score_credit = 0.0;
    config.router_assume_kv_reuse = false;
    config.router_track_prefill_tokens = false;
    config.router_prefill_load_model = RouterPrefillLoadModel::None;
    config
}

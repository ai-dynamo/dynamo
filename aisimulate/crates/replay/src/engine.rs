// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default generalized-engine construction for offline replay.
//!
//! This module contains configuration and conversion helpers only. Scheduler
//! state lives in `aisimulate-engine`; virtual time and worker lifecycle live
//! in the moved aggregated/disaggregated replay runtimes.

use std::num::NonZeroU32;
use std::sync::Arc;

use aisimulate_engine::generalized::EngineIdentity;
use aisimulate_engine::{Backend, Engine, EngineConfig, EngineFactory, TimingModel, WorkerType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::components::{AdmissionQueue, NoReplayMetadata, ReplayEngineObservation, ReplayMode};
use crate::core::EngineEventBatch;
use crate::core::round_robin::PoolRoundRobinPlacement;
use crate::disagg::DisaggRuntimeImpl;
use crate::error::runtime_error;
use crate::protocol::DirectRequest;
use crate::replay::OfflineDisaggReplayConfig;
use crate::{
    ReplayError, ReplayResult, ReplaySpec, ReplayTopology, Replayer, TraceSimulationReport,
    WorkerStage,
};

fn default_dp_size() -> u32 {
    1
}

fn default_tensor_parallel_size() -> u32 {
    1
}

/// Serializable execution-time descriptor for the AISimulate engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayEngineConfig {
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: u32,
    pub rank: EngineConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill: Option<ReplayRoleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<ReplayRoleConfig>,
}

impl Default for ReplayEngineConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tensor_parallel_size: 1,
            rank: EngineConfig::default(),
            prefill: None,
            decode: None,
        }
    }
}

/// Rank-group descriptor for one disaggregated role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayRoleConfig {
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: u32,
    pub rank: EngineConfig,
}

impl Default for ReplayRoleConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tensor_parallel_size: 1,
            rank: EngineConfig::default(),
        }
    }
}

impl ReplayEngineConfig {
    pub(crate) fn parse(value: &Value) -> ReplayResult<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        serde_json::from_value(value.clone()).map_err(|error| {
            ReplayError::InvalidSpec(format!("invalid native engine descriptor: {error}"))
        })
    }

    pub(crate) fn role(&self, stage: WorkerStage) -> ReplayRoleConfig {
        let mut role = match stage {
            WorkerStage::Aggregated => ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                rank: self.rank.clone(),
            },
            WorkerStage::Prefill => self.prefill.clone().unwrap_or_else(|| ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                rank: self.rank.clone(),
            }),
            WorkerStage::Decode => self.decode.clone().unwrap_or_else(|| ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                rank: self.rank.clone(),
            }),
        };
        role.rank.worker_type = match stage {
            WorkerStage::Aggregated => WorkerType::Aggregated,
            WorkerStage::Prefill => WorkerType::Prefill,
            WorkerStage::Decode => WorkerType::Decode,
        };
        role
    }

    pub(crate) fn validate_topology(&self, topology: &ReplayTopology) -> ReplayResult<()> {
        if matches!(topology, ReplayTopology::Disaggregated { .. }) {
            for stage in [WorkerStage::Prefill, WorkerStage::Decode] {
                let role = self.role(stage);
                if role.dp_size != 1 {
                    let role_name = match stage {
                        WorkerStage::Prefill => "prefill",
                        WorkerStage::Decode => "decode",
                        WorkerStage::Aggregated => unreachable!(),
                    };
                    return Err(ReplayError::InvalidSpec(format!(
                        "disaggregated replay requires {role_name} dp_size=1; attention-DP handoff identity is not implemented"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Reusable construction state for one worker role.
#[doc(hidden)]
#[derive(Clone)]
pub struct ReplayRoleFactory {
    factory: EngineFactory,
    dp_size: NonZeroU32,
    tensor_parallel_size: u32,
    backend: Backend,
}

impl ReplayRoleFactory {
    #[doc(hidden)]
    pub fn build(&self, worker_id: usize) -> ReplayResult<Engine> {
        let worker_id = u64::try_from(worker_id).map_err(|_| {
            ReplayError::Engine(format!(
                "worker id {worker_id} exceeds the native engine range"
            ))
        })?;
        self.factory
            .build(EngineIdentity::new(worker_id), self.dp_size)
            .map_err(engine_error)
    }

    #[doc(hidden)]
    pub fn dp_size(&self) -> u32 {
        self.dp_size.get()
    }

    #[doc(hidden)]
    pub fn gpus_per_worker(&self) -> ReplayResult<usize> {
        usize::try_from(self.dp_size.get())
            .ok()
            .and_then(|dp| {
                usize::try_from(self.tensor_parallel_size)
                    .ok()
                    .and_then(|tp| dp.checked_mul(tp))
            })
            .ok_or_else(|| ReplayError::InvalidSpec("engine GPU count overflows usize".into()))
    }

    #[doc(hidden)]
    pub fn backend(&self) -> Backend {
        self.backend
    }
}

/// Resolves built-in or Runner-provided timing once, then creates role factories.
#[derive(Clone, Default)]
pub struct ReplayEngineFactory {
    timing: Option<Arc<dyn TimingModel>>,
    prefill_timing: Option<Arc<dyn TimingModel>>,
    decode_timing: Option<Arc<dyn TimingModel>>,
}

impl ReplayEngineFactory {
    pub const fn new() -> Self {
        Self {
            timing: None,
            prefill_timing: None,
            decode_timing: None,
        }
    }

    pub fn with_timing_model(timing: Arc<dyn TimingModel>) -> Self {
        Self {
            timing: Some(timing),
            prefill_timing: None,
            decode_timing: None,
        }
    }

    pub fn with_optional_role_timing_models(
        prefill: Option<Arc<dyn TimingModel>>,
        decode: Option<Arc<dyn TimingModel>>,
    ) -> Self {
        Self {
            timing: None,
            prefill_timing: prefill,
            decode_timing: decode,
        }
    }

    #[doc(hidden)]
    pub fn role_factory(
        &self,
        config: &ReplayEngineConfig,
        stage: WorkerStage,
        emit_kv_events: bool,
    ) -> ReplayResult<ReplayRoleFactory> {
        let mut role = config.role(stage);
        role.rank.emit_kv_events = emit_kv_events;
        let dp_size = NonZeroU32::new(role.dp_size).ok_or_else(|| {
            ReplayError::InvalidSpec("native engine dp_size must be positive".into())
        })?;
        if role.tensor_parallel_size == 0 {
            return Err(ReplayError::InvalidSpec(
                "native tensor_parallel_size must be positive".into(),
            ));
        }
        let timing = match stage {
            WorkerStage::Aggregated => self.timing.as_ref(),
            WorkerStage::Prefill => self.prefill_timing.as_ref().or(self.timing.as_ref()),
            WorkerStage::Decode => self.decode_timing.as_ref().or(self.timing.as_ref()),
        };
        let backend = role.rank.backend;
        let factory = match timing {
            Some(timing) => EngineFactory::with_timing_model(role.rank, Arc::clone(timing)),
            None => EngineFactory::new(role.rank),
        }
        .map_err(engine_error)?;
        Ok(ReplayRoleFactory {
            factory,
            dp_size,
            tensor_parallel_size: role.tensor_parallel_size,
            backend,
        })
    }
}

pub fn run_engine_replay(spec: ReplaySpec) -> ReplayResult<TraceSimulationReport> {
    Replayer::new(spec, ReplayEngineFactory::new())?.run()
}

pub fn run_engine_replay_with_timing(
    spec: ReplaySpec,
    timing: Arc<dyn TimingModel>,
) -> ReplayResult<TraceSimulationReport> {
    Replayer::new(spec, ReplayEngineFactory::with_timing_model(timing))?.run()
}

pub fn run_engine_replay_with_optional_role_timing(
    spec: ReplaySpec,
    prefill: Option<Arc<dyn TimingModel>>,
    decode: Option<Arc<dyn TimingModel>>,
) -> ReplayResult<TraceSimulationReport> {
    Replayer::new(
        spec,
        ReplayEngineFactory::with_optional_role_timing_models(prefill, decode),
    )?
    .run()
}

#[derive(Debug, Default)]
struct KvEventBatch(Vec<aisimulate_engine::KvEvent>);

impl EngineEventBatch for KvEventBatch {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

#[derive(Debug, Default)]
struct KvEventObservation;

impl ReplayEngineObservation for KvEventObservation {
    type Batch = KvEventBatch;

    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn observe_engine_events(
        _stage: WorkerStage,
        _worker_id: usize,
        _dp_rank: u32,
        events: Vec<aisimulate_engine::KvEvent>,
    ) -> Self::Batch {
        KvEventBatch(events)
    }

    fn stored_hashes(batch: &Self::Batch) -> Vec<u64> {
        batch
            .0
            .iter()
            .flat_map(|event| match &event.data {
                aisimulate_engine::KvEventData::Stored(stored) => stored.blocks.as_slice(),
                aisimulate_engine::KvEventData::Removed { .. } => &[],
            })
            .map(|block| block.tokens_hash)
            .collect()
    }
}

/// Run the engine-neutral half of Dynamo's live/offline handoff conformance
/// fixture without importing or recompiling Replay implementation sources.
#[doc(hidden)]
pub fn run_engine_handoff_conformance(
    config: ReplayEngineConfig,
    factory: ReplayEngineFactory,
    request: DirectRequest,
) -> ReplayResult<crate::NormalizedHandoffConformance> {
    let prefill_factory = factory.role_factory(&config, WorkerStage::Prefill, true)?;
    let decode_factory = factory.role_factory(&config, WorkerStage::Decode, true)?;
    let backend = prefill_factory.backend();
    if backend != decode_factory.backend() {
        return Err(ReplayError::InvalidSpec(
            "handoff conformance requires matching prefill/decode backends".into(),
        ));
    }
    let runtime_config = OfflineDisaggReplayConfig {
        prefill_factory,
        decode_factory,
        prefill_startup_time_ms: None,
        decode_startup_time_ms: None,
        num_prefill_workers: 1,
        num_decode_workers: 1,
        handoff_latency_ms: 0.0,
    };
    DisaggRuntimeImpl::<
        PoolRoundRobinPlacement<KvEventBatch>,
        KvEventObservation,
        NoReplayMetadata,
    >::new_composed(
        &runtime_config,
        AdmissionQueue::new_requests(
            std::collections::VecDeque::from([request]),
            ReplayMode::Trace,
        ),
        true,
        |_, prefill_topology, _, decode_topology| {
            Ok((
                PoolRoundRobinPlacement::new(prefill_topology),
                PoolRoundRobinPlacement::new(decode_topology),
            ))
        },
    )
    .map_err(runtime_error)?
    .run_handoff_conformance(backend)
    .map_err(runtime_error)
}

fn engine_error(error: impl std::fmt::Display) -> ReplayError {
    ReplayError::Engine(error.to_string())
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted and modified from ai-dynamo/dynamo, Apache-2.0:
// https://github.com/ai-dynamo/dynamo/blob/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/lib/mocker/src/replay/offline/extensions/kv_router/composition.rs

use std::cell::RefCell;
use std::collections::HashSet;

use anyhow::{Context, Result, ensure};
use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};

use super::helpers::NativeRouterEngineArgs;
use super::observation::RouterEventObservation;
use super::router::{KvReplayMetadata, KvRouterPlacement};
use super::{
    NATIVE_DYNAMO_PROVIDER, NativeDynamoRouterConfig, ReplayPrefillLoadEstimator, RoutingEvidence,
};
use aisimulate_core::replay::{
    ReplayComposition, ReplayDeterminism, ReplayEngineConfig, ReplayError, ReplayResult,
    ReplayRoleConfig, ReplayScalingPolicy, ReplaySpec, WorkerTopology,
};

/// Native routing hosted by the same AISimulate executor as round-robin replay.
pub struct NativeDynamoReplayComposition {
    config: NativeDynamoRouterConfig,
    engine: RefCell<Option<ReplayEngineConfig>>,
    scaling: Option<Box<dyn ReplayScalingPolicy>>,
    scaling_enabled: bool,
    estimator: Option<ReplayPrefillLoadEstimator>,
    determinism: ReplayDeterminism,
    capture_decisions: bool,
    evidence: RoutingEvidence,
}

impl NativeDynamoReplayComposition {
    pub fn from_config(
        config: NativeDynamoRouterConfig,
        scaling: Option<Box<dyn ReplayScalingPolicy>>,
    ) -> Self {
        Self {
            config,
            engine: RefCell::new(None),
            scaling_enabled: scaling.is_some(),
            scaling,
            estimator: None,
            determinism: ReplayDeterminism::Random,
            capture_decisions: false,
            evidence: RoutingEvidence::default(),
        }
    }

    pub fn with_capture_decisions(mut self, capture: bool) -> Self {
        self.capture_decisions = capture;
        self
    }

    pub fn with_prefill_load_estimator(mut self, estimator: ReplayPrefillLoadEstimator) -> Self {
        self.estimator = Some(estimator);
        self
    }

    pub fn routing_evidence(&self) -> RoutingEvidence {
        self.evidence.clone()
    }

    fn placement(
        &self,
        args: &NativeRouterEngineArgs,
        config: Option<KvRouterConfig>,
        estimator: Option<ReplayPrefillLoadEstimator>,
        workers: usize,
        role: &'static str,
    ) -> Result<KvRouterPlacement> {
        KvRouterPlacement::new_with_selector_seed(
            args,
            config,
            estimator,
            workers,
            self.determinism.selector_seed(),
        )?
        .with_affinity(
            self.config.affinity.clone(),
            role,
            self.capture_decisions,
            self.evidence.clone(),
        )
    }
}

impl ReplayComposition for NativeDynamoReplayComposition {
    type Metadata = KvReplayMetadata;
    type Observation = RouterEventObservation;
    type AggregatedPlacement = KvRouterPlacement;
    type DisaggregatedPlacement = KvRouterPlacement;

    fn validate_spec(&self, spec: &ReplaySpec) -> ReplayResult<()> {
        let validate = || -> Result<ReplayEngineConfig> {
            ensure!(
                spec.adapters.placement.provider == NATIVE_DYNAMO_PROVIDER,
                "native Dynamo composition requires placement provider '{NATIVE_DYNAMO_PROVIDER}'"
            );
            let declared = NativeDynamoRouterConfig::from_value(&spec.adapters.placement.config)?;
            ensure!(
                serde_json::to_value(declared)? == serde_json::to_value(&self.config)?,
                "native Dynamo composition disagrees with the serialized placement configuration"
            );
            ensure!(
                spec.adapters.scaling.config.is_null(),
                "native routing received unused scaling configuration"
            );
            ensure!(
                if self.scaling_enabled {
                    !spec.adapters.scaling.provider.is_empty()
                        && spec.adapters.scaling.provider != "none"
                } else {
                    spec.adapters.scaling.provider == "none"
                },
                "scaling provider and supplied scaling policy must agree"
            );
            let engine: ReplayEngineConfig = if spec.engine.is_null() {
                Default::default()
            } else {
                serde_json::from_value(spec.engine.clone())?
            };
            validate_replay_router_config(self.config.router_config.as_ref())?;
            if self.config.affinity.is_some() {
                ensure!(
                    !self.scaling_enabled,
                    "conversation affinity requires static worker pools without a Planner"
                );
                ensure!(
                    self.estimator.is_none(),
                    "conversation affinity does not support a custom prefill-load estimator"
                );
                validate_affinity_router_config(self.config.router_config.as_ref())?;
                for rank in std::iter::once(&engine.rank)
                    .chain(engine.prefill.iter().map(|r| &r.rank))
                    .chain(engine.decode.iter().map(|r| &r.rank))
                {
                    ensure!(
                        rank.native_host_offload.is_none() && rank.g3_offload.is_none(),
                        "conversation affinity supports device KV cache only"
                    );
                    ensure!(
                        rank.aic_nextn.is_none(),
                        "conversation affinity does not support speculative decoding"
                    );
                    ensure!(
                        matches!(
                            rank.backend,
                            aisimulate_core::engine::Backend::Vllm
                                | aisimulate_core::engine::Backend::Sglang
                        ),
                        "conversation affinity supports vllm or sglang"
                    );
                }
            }
            Ok(engine)
        };
        *self.engine.borrow_mut() =
            Some(validate().map_err(|e| ReplayError::InvalidSpec(e.to_string()))?);
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<KvRouterPlacement> {
        let engine = self.engine.borrow();
        let engine = engine
            .as_ref()
            .context("native routing engine was not validated")?;
        let args = router_args(engine.dp_size, &engine.rank)?;
        validate_topology(&args, dp_size, &topology)?;
        self.placement(
            &args,
            self.config.router_config.clone(),
            self.estimator.clone(),
            topology.len(),
            "aggregated",
        )
    }

    fn create_disaggregated_placements(
        &mut self,
        prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> Result<(KvRouterPlacement, KvRouterPlacement)> {
        let engine = self.engine.borrow();
        let engine = engine
            .as_ref()
            .context("native routing engine was not validated")?;
        let fallback = ReplayRoleConfig {
            dp_size: engine.dp_size,
            tensor_parallel_size: engine.tensor_parallel_size,
            prefill_context_parallel_size: engine.prefill_context_parallel_size,
            num_gpu_blocks_is_explicit: engine.num_gpu_blocks_is_explicit,
            rank: engine.rank.clone(),
        };
        let prefill = engine.prefill.as_ref().unwrap_or(&fallback);
        let decode = engine.decode.as_ref().unwrap_or(&fallback);
        let prefill_args = router_args(prefill.dp_size, &prefill.rank)?;
        let decode_args = router_args(decode.dp_size, &decode.rank)?;
        validate_topology(&prefill_args, prefill_dp_size, &prefill_topology)?;
        validate_topology(&decode_args, decode_dp_size, &decode_topology)?;
        let mut prefill_config = self.config.router_config.clone().unwrap_or_default();
        prefill_config.router_track_active_blocks = false;
        let mut decode_config = self.config.router_config.clone().unwrap_or_default();
        decode_config.overlap_score_credit = 0.0;
        decode_config.router_assume_kv_reuse = false;
        decode_config.router_track_prefill_tokens = false;
        decode_config.router_prefill_load_model = RouterPrefillLoadModel::None;
        Ok((
            self.placement(
                &prefill_args,
                Some(prefill_config),
                self.estimator.clone(),
                prefill_topology.len(),
                "prefill",
            )?,
            self.placement(
                &decode_args,
                Some(decode_config),
                None,
                decode_topology.len(),
                "decode",
            )?,
        ))
    }

    fn take_scaling_policy(&mut self) -> Result<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(self.scaling.take())
    }

    fn set_determinism(&mut self, determinism: ReplayDeterminism) -> ReplayResult<()> {
        #[cfg(not(feature = "replay-bench"))]
        if determinism == ReplayDeterminism::CanonicalV1 {
            return Err(ReplayError::InvalidSpec(
                "canonical native routing requires the replay-bench feature".into(),
            ));
        }
        self.determinism = determinism;
        Ok(())
    }
}

fn router_args(
    dp_size: u32,
    rank: &aisimulate_core::engine::EngineConfig,
) -> Result<NativeRouterEngineArgs> {
    Ok(NativeRouterEngineArgs {
        dp_size,
        block_size: rank
            .block_size
            .try_into()
            .context("native router block size exceeds u32")?,
        num_gpu_blocks: rank.num_gpu_blocks,
        max_num_batched_tokens: Some(rank.max_num_batched_tokens),
    })
}

fn validate_topology(
    args: &NativeRouterEngineArgs,
    dp_size: u32,
    topology: &[WorkerTopology],
) -> Result<()> {
    ensure!(
        dp_size == args.dp_size.max(1),
        "native router and engine DP size disagree"
    );
    let mut ids = HashSet::new();
    for (worker_id, worker) in topology.iter().enumerate() {
        ensure!(
            worker.worker_id == worker_id,
            "native router worker IDs must be contiguous"
        );
        ensure!(
            worker.scheduler_ids.len() == dp_size as usize,
            "native router and worker rank count disagree"
        );
        for (rank, &id) in worker.scheduler_ids.iter().enumerate() {
            ensure!(
                id == worker_id * dp_size as usize + rank,
                "native router requires contiguous worker/rank scheduler IDs"
            );
            ensure!(
                ids.insert(id),
                "native router topology repeats a scheduler ID"
            );
        }
    }
    Ok(())
}

fn validate_affinity_router_config(config: Option<&KvRouterConfig>) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    let defaults = KvRouterConfig::default();
    ensure!(
        serde_json::to_value(config)? == serde_json::to_value(&defaults)?
            && config.router_approximate_cache_policy == defaults.router_approximate_cache_policy
            && config.router_prefill_policy.is_none()
            && config.router_decode_policy.is_none()
            && config.policy_model_name.is_none()
            && config.policy_config_cache.get().is_none(),
        "conversation affinity supports only the default native KV router configuration; queue, custom-policy, cache-tier, and prefill-load overrides require ordinary non-affinity replay"
    );
    Ok(())
}

fn validate_replay_router_config(config: Option<&KvRouterConfig>) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    let defaults = KvRouterConfig::default();
    let configured = serde_json::to_value(config)?;
    let baseline = serde_json::to_value(&defaults)?;
    // These are live-service facilities. Replay supplies synchronous local KV
    // events and a fixed topology, so accepting overrides would silently discard
    // the requested behavior. Native scoring, tracking and queue controls remain
    // available to ordinary routing.
    for field in [
        "use_kv_events",
        "router_replica_sync",
        "router_track_output_blocks",
        "router_ttl_secs",
        "router_event_threads",
        "skip_initial_worker_wait",
        "use_remote_indexer",
        "serve_indexer",
        "enable_session_prefix_index",
        "shared_cache_multiplier",
        "shared_cache_type",
        "router_predicted_ttl_secs",
        "host_cache_hit_weight",
        "disk_cache_hit_weight",
        "conditional_disagg_enabled",
        "conditional_disagg_policy",
        "conditional_disagg_eff_isl_threshold",
        "conditional_disagg_eff_isl_ratio_threshold",
        "conditional_disagg_prefill_busy_threshold",
        "conditional_disagg_decode_busy_threshold",
    ] {
        ensure!(
            configured.get(field) == baseline.get(field),
            "native replay does not support overriding router_config.{field}"
        );
    }
    ensure!(
        config.router_approximate_cache_policy == defaults.router_approximate_cache_policy,
        "native replay does not support approximate-cache policies"
    );
    Ok(())
}

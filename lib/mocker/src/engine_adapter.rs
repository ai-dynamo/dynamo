// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Dynamo-to-AISimulate engine configuration boundary.

use std::sync::Arc;

use aisim_core::engine::{
    EngineConfig, EngineFactory, FittedIterationTiming, TimingModel, TimingModelConfig, WorkerType,
};
use aisim_core::replay::{ReplayEngineConfig, ReplayEngineFactory, ReplayRoleConfig};
use aisim_core::{
    BackendKind, EstimationMode, ForwardPassRegressionWorkloadKind, ForwardPassWorkerType,
};
use anyhow::{Context, Result, anyhow, ensure};
use serde_json::Value;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::MockerConfig;

/// Fully materialized rank configuration and its optional process-local
/// external timing provider.
pub(crate) struct EngineComponents {
    pub(crate) args: MockerConfig,
    pub(crate) rank: EngineConfig,
    pub(crate) timing: Option<Arc<dyn TimingModel>>,
}

/// Normalize Dynamo mock-engine arguments and materialize the neutral engine
/// rank contract.
///
/// Attention-DP size remains a grouped-engine concern and is intentionally not
/// copied into [`EngineConfig`].
pub(crate) fn engine_components(
    args: MockerConfig,
    emit_kv_events: bool,
    emit_kv_token_ids: bool,
) -> Result<EngineComponents> {
    let args = args
        .normalized()
        .context("invalid Mocker engine arguments")?;
    let mut rank = args.engine.engine.clone();
    rank.emit_kv_events = emit_kv_events;
    rank.emit_kv_token_ids = emit_kv_token_ids;
    let timing = match args.perf_model.as_ref() {
        PerfModel::Polynomial => None,
        PerfModel::Fixed {
            prefill_ms,
            decode_ms,
        } => {
            rank.timing_model = TimingModelConfig::Fixed {
                prefill_ms: *prefill_ms,
                decode_ms: *decode_ms,
            };
            None
        }
        PerfModel::Interpolated { .. } | PerfModel::Ais { .. } => {
            rank.timing_model = TimingModelConfig::External {
                provider: "dynamo_perf_model".to_string(),
                config: Value::Null,
            };
            Some(Arc::new(DynamoPerfTimingModel {
                inner: Arc::clone(&args.perf_model),
            }) as Arc<dyn TimingModel>)
        }
    };
    Ok(EngineComponents { args, rank, timing })
}

/// Engine components for one offline Replay role. A fitted AIS estimator
/// (`fpm_regression` or `fpm_learned`) whose stores can price every pass the
/// role runs is priced by AISimulate's own iteration timing: one forward-pass
/// query per pass, mixed passes and per-request detail included, as
/// AISimulate's engine replay prices it. One that lacks a store the role
/// needs, such as a model tuned from profiler points that never mix prefill
/// and decode, keeps the callback, which prices a mixed pass as a prefill plus
/// a separate decode pass.
fn offline_replay_components(args: &MockerConfig) -> Result<EngineComponents> {
    let mut components = engine_components(args.clone(), false, false)?;
    let PerfModel::Ais { callback } = components.args.perf_model.as_ref() else {
        return Ok(components);
    };
    let Some(model) = callback.forward_pass_model() else {
        return Ok(components);
    };
    let Some(provenance) = model.provenance().filter(|provenance| {
        matches!(
            provenance.selected_estimation_mode,
            EstimationMode::FpmRegression | EstimationMode::FpmLearned
        )
    }) else {
        return Ok(components);
    };
    // `from_model` takes the role from the model's identity; AISimulate
    // leaves checking it against the engine's role to the embedder.
    let role = replay_role(components.args.worker_type);
    let model_role = provenance.config.worker_type;
    ensure!(
        model_role == role,
        "the fitted AIS model is for the {} role, but this engine is {}",
        role_name(model_role),
        role_name(role),
    );
    let required = iteration_stores(role, &provenance.config.backend, components.args.dp_size);
    let ready = model
        .readiness_report()
        .stores
        .iter()
        .all(|store| store.ready || !required.contains(&store.workload_kind));
    if ready {
        // The cause goes into the message itself: Python shows only the
        // outermost error of a chain.
        let timing = FittedIterationTiming::from_model(model).map_err(|error| {
            anyhow!(
                "invalid fitted AIS timing for the {} role: {error:#}",
                role_name(role)
            )
        })?;
        timing
            .require_replay_role(
                components.args.dp_size,
                components.args.decode_speedup_ratio,
            )
            .map_err(|error| {
                anyhow!(
                    "fitted AIS timing cannot serve the {} role: {error:#}",
                    role_name(role)
                )
            })?;
        components.timing = Some(Arc::new(timing));
    }
    Ok(components)
}

/// The workload stores `FittedIterationTiming` requires to price every pass a
/// `role` engine with `attention_dp` ranks runs, as AISimulate's
/// `require_iteration_stores` lists them: the role's pure stores, and for an
/// aggregated engine locally mixed passes unless the backend is SGLang, and
/// cross-rank passes when attention DP is above one.
fn iteration_stores(
    role: ForwardPassWorkerType,
    backend: &BackendKind,
    attention_dp: u32,
) -> Vec<ForwardPassRegressionWorkloadKind> {
    use ForwardPassRegressionWorkloadKind::*;
    match role {
        ForwardPassWorkerType::Prefill => vec![PurePrefill],
        ForwardPassWorkerType::Decode => vec![PureDecode],
        ForwardPassWorkerType::Aggregated => {
            let mut stores = vec![PurePrefill, PureDecode];
            if !matches!(backend, BackendKind::Sglang) {
                stores.push(ContainsLocallyMixed);
            }
            if attention_dp > 1 {
                stores.push(CrossRankAggregated);
            }
            stores
        }
    }
}

fn replay_role(worker_type: WorkerType) -> ForwardPassWorkerType {
    match worker_type {
        WorkerType::Aggregated => ForwardPassWorkerType::Aggregated,
        WorkerType::Prefill => ForwardPassWorkerType::Prefill,
        WorkerType::Decode => ForwardPassWorkerType::Decode,
    }
}

fn role_name(role: ForwardPassWorkerType) -> &'static str {
    match role {
        ForwardPassWorkerType::Aggregated => "aggregated",
        ForwardPassWorkerType::Prefill => "prefill",
        ForwardPassWorkerType::Decode => "decode",
    }
}

pub(crate) fn engine_factory(
    rank: EngineConfig,
    timing: Option<Arc<dyn TimingModel>>,
) -> Result<EngineFactory> {
    match timing {
        Some(timing) => EngineFactory::with_timing_model(rank, timing),
        None => EngineFactory::new(rank),
    }
}

fn replay_tensor_parallel_size(args: &MockerConfig) -> Result<u32> {
    u32::try_from(args.tensor_parallel_size)
        .context("Mocker tensor-parallel size exceeds the Replay contract")
}

/// Materialize the serializable engine descriptor and process-local timing
/// provider used by one aggregated Replay invocation.
pub(crate) fn aggregated_replay_setup(
    args: &MockerConfig,
) -> Result<(ReplayEngineConfig, ReplayEngineFactory)> {
    let components = offline_replay_components(args)?;
    let config = ReplayEngineConfig {
        kv_eviction_policy: Default::default(),
        dp_size: components.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&components.args)?,
        prefill_context_parallel_size: components.args.replay_prefill_context_parallel_size()?,
        // Mocker arguments reach this boundary with a concrete capacity,
        // whether authored directly or supplied by the Dynamo planner.
        num_gpu_blocks_is_explicit: Some(true),
        rank: components.rank,
        prefill: None,
        decode: None,
    };
    let factory = match components.timing {
        Some(timing) => ReplayEngineFactory::with_timing_model(timing),
        None => ReplayEngineFactory::new(),
    };
    Ok((config, factory))
}

/// Materialize role-specific descriptors and timing providers for one
/// disaggregated Replay invocation.
pub(crate) fn disaggregated_replay_setup(
    prefill_args: &MockerConfig,
    decode_args: &MockerConfig,
) -> Result<(ReplayEngineConfig, ReplayEngineFactory)> {
    let prefill = offline_replay_components(prefill_args)?;
    let decode = offline_replay_components(decode_args)?;
    let prefill_role = ReplayRoleConfig {
        dp_size: prefill.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&prefill.args)?,
        prefill_context_parallel_size: prefill.args.replay_prefill_context_parallel_size()?,
        num_gpu_blocks_is_explicit: Some(true),
        rank: prefill.rank,
    };
    let decode_role = ReplayRoleConfig {
        dp_size: decode.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&decode.args)?,
        prefill_context_parallel_size: decode.args.replay_prefill_context_parallel_size()?,
        num_gpu_blocks_is_explicit: Some(true),
        rank: decode.rank,
    };
    let config = ReplayEngineConfig {
        kv_eviction_policy: Default::default(),
        dp_size: prefill_role.dp_size,
        tensor_parallel_size: prefill_role.tensor_parallel_size,
        prefill_context_parallel_size: prefill_role.prefill_context_parallel_size,
        num_gpu_blocks_is_explicit: Some(true),
        rank: prefill_role.rank.clone(),
        prefill: Some(prefill_role),
        decode: Some(decode_role),
    };
    Ok((
        config,
        ReplayEngineFactory::with_optional_role_timing_models(prefill.timing, decode.timing),
    ))
}

struct DynamoPerfTimingModel {
    inner: Arc<PerfModel>,
}

impl TimingModel for DynamoPerfTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        self.inner
            .predict_prefill_time(batch_size, mean_isl, mean_prefix)
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        mean_context_length: usize,
        total_kv_tokens: usize,
    ) -> Result<f64> {
        self.inner.predict_decode_time(
            batch_size,
            active_kv_tokens,
            mean_context_length,
            total_kv_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use aisim_core::ForwardPassPerfModel;
    use aisim_core::engine::generalized::EngineIdentity;
    use aisim_core::replay::WorkerStage;
    use ndarray_interp::InterpolateError;

    use super::*;
    use crate::common::perf_model::{AisCallback, DecodeInterpolator, PrefillInterpolator};
    use crate::common::protocols::NativeHostOffloadConfig;
    use aisim_core::engine::generalized::SchedulerCommand;
    use aisim_core::engine::{Backend, Command, Engine, Request, SglangSchedulePolicy};

    struct EchoPrefill;

    impl PrefillInterpolator for EchoPrefill {
        fn interp(&self, x: f64) -> std::result::Result<f64, InterpolateError> {
            Ok(x)
        }
    }

    struct EchoDecode;

    impl DecodeInterpolator for EchoDecode {
        fn interp(&self, x: f64, y: f64) -> std::result::Result<f64, InterpolateError> {
            Ok(x + y)
        }
    }

    struct EchoAis;

    impl AisCallback for EchoAis {
        fn predict_prefill(
            &self,
            batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> Result<f64> {
            Ok(batch_size as f64)
        }

        fn predict_decode(&self, batch_size: usize, _isl: usize, _osl: usize) -> Result<f64> {
            Ok(batch_size as f64)
        }
    }

    #[test]
    fn replay_gpu_accounting_uses_each_roles_canonical_pcp() {
        let args_for = |role: &str, cp_size: Option<u32>| {
            let mut args = MockerConfig::from_value(serde_json::json!({
                "engine": {
                    "worker_type": role,
                    "timing_model": {
                        "type": "external", "provider": "ais",
                        "config": {
                            "model": "test-model", "system": "h200_sxm",
                            "backend": "vllm", "worker_type": role,
                            "tp": 2, "attention_dp": 3, "cp_size": cp_size, "dcp": 2
                        }
                    }
                }
            }))
            .unwrap();
            args.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(EchoAis)));
            args
        };
        for cp_size in [None, Some(4)] {
            let args = args_for("aggregated", cp_size);
            let (config, factory) = aggregated_replay_setup(&args).unwrap();
            let role = factory
                .role_factory(&config, WorkerStage::Aggregated, false)
                .unwrap();
            assert_eq!(
                role.gpus_per_worker().unwrap(),
                6 * cp_size.unwrap_or(1) as usize
            );
        }
        let (config, factory) =
            disaggregated_replay_setup(&args_for("prefill", Some(4)), &args_for("decode", Some(2)))
                .unwrap();
        assert_eq!(config.prefill_context_parallel_size, 4);
        for (stage, expected) in [(WorkerStage::Prefill, 24), (WorkerStage::Decode, 12)] {
            assert_eq!(
                factory
                    .role_factory(&config, stage, false)
                    .unwrap()
                    .gpus_per_worker()
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn vllm_defaults_materialize_once_at_the_shared_boundary() {
        let mut args = MockerConfig::from_value(serde_json::json!({})).unwrap();
        args.kv_transfer_bytes_per_token = Some(4096);
        args.kv_cache_bytes_per_token = Some(1024);
        let components = engine_components(args, true, true).unwrap();

        assert_eq!(components.args.block_size, 64);
        assert_eq!(components.rank.backend, Backend::Vllm);
        assert_eq!(components.rank.block_size, 64);
        assert_eq!(components.rank.kv_transfer_bytes_per_token, Some(4096));
        assert_eq!(components.rank.kv_cache_bytes_per_token, Some(1024));
        assert!(components.rank.emit_kv_events);
        assert!(components.rank.emit_kv_token_ids);
        assert_eq!(components.rank.timing_model, TimingModelConfig::Polynomial);
        assert!(components.timing.is_none());
    }

    #[test]
    fn replay_json_keeps_transfer_and_cache_geometry_independent() {
        let args: MockerConfig = serde_json::from_value(serde_json::json!({
            "engine": {
                "kv_transfer_bytes_per_token": 4096,
                "kv_cache_bytes_per_token": 1024,
            }
        }))
        .unwrap();
        let components = engine_components(args, false, false).unwrap();
        assert_eq!(components.rank.kv_transfer_bytes_per_token, Some(4096));
        assert_eq!(components.rank.kv_cache_bytes_per_token, Some(1024));
    }

    #[test]
    fn native_host_offload_preserves_explicit_cache_geometry() {
        let mut args = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "kv_transfer_bytes_per_token": 4096,
                "kv_cache_bytes_per_token": 2048,
                "native_host_offload": NativeHostOffloadConfig::new(8)
            }
        }))
        .unwrap();
        args.kv_transfer_bytes_per_token = Some(8192);
        let components = engine_components(args, false, false).unwrap();
        assert_eq!(components.rank.kv_cache_bytes_per_token, Some(2048));
        assert_eq!(components.rank.kv_transfer_bytes_per_token, Some(8192));
    }

    #[test]
    fn backend_specific_fields_match_the_engine_contract() {
        let sglang = MockerConfig::from_json_str(
            r#"{"engine": {"sglang": {"schedule_policy": "lpm", "max_prefill_tokens": 512, "chunked_prefill_size": 256, "clip_max_new_tokens": 128, "schedule_conservativeness": 0.5}, "backend": "sglang", "block_size": 8}}"#,
        )
        .unwrap();
        let components = engine_components(sglang, false, false).unwrap();
        assert_eq!(components.rank.backend, Backend::Sglang);
        assert_eq!(components.rank.block_size, 8);
        assert_eq!(
            components.rank.sglang.schedule_policy,
            SglangSchedulePolicy::Lpm
        );
        assert_eq!(components.rank.sglang.chunked_prefill_size, 256);

        let trtllm = MockerConfig::from_json_str(r#"{"engine": {"backend": "trtllm"}}"#).unwrap();
        let components = engine_components(trtllm, false, false).unwrap();
        assert_eq!(components.rank.backend, Backend::Trtllm);
        assert_eq!(components.rank.block_size, 32);
    }

    #[test]
    fn polynomial_builds_replay_and_live_engines_without_an_external_provider() {
        let args = MockerConfig::from_value(serde_json::json!({})).unwrap();
        let components = engine_components(args.clone(), false, false).unwrap();
        assert_eq!(components.rank.timing_model, TimingModelConfig::Polynomial);
        assert!(components.timing.is_none());

        engine_factory(components.rank, components.timing)
            .unwrap()
            .build(EngineIdentity::new(0), NonZeroU32::MIN)
            .unwrap();

        let (config, replay_factory) = aggregated_replay_setup(&args).unwrap();
        assert_eq!(config.rank.timing_model, TimingModelConfig::Polynomial);
        replay_factory
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap()
            .build(0)
            .unwrap();
    }

    #[test]
    fn public_fixed_timing_materializes_as_builtin_engine_timing() {
        let args = MockerConfig::from_json_str(
            r#"{"engine": {"timing_model": {"type": "fixed", "prefill_ms": 2.5, "decode_ms": 0.5}, "backend": "vllm"}}"#,
        )
        .unwrap();
        let components = engine_components(args, false, false).unwrap();

        assert_eq!(
            components.rank.timing_model,
            TimingModelConfig::Fixed {
                prefill_ms: 2.5,
                decode_ms: 0.5,
            }
        );
        assert!(components.timing.is_none());
    }

    #[test]
    fn npz_and_ais_models_use_the_external_timing_adapter() {
        let models = [
            PerfModel::Interpolated {
                prefill_interp: Arc::new(EchoPrefill),
                decode_interp: Arc::new(EchoDecode),
            },
            PerfModel::from_ais_callback(Arc::new(EchoAis)),
        ];

        for model in models {
            let mut args = MockerConfig::from_value(serde_json::json!({})).unwrap();
            args.perf_model = Arc::new(model);
            let components = engine_components(args, false, false).unwrap();

            assert!(matches!(
                components.rank.timing_model,
                TimingModelConfig::External { ref provider, .. }
                    if provider == "dynamo_perf_model"
            ));
            let timing = components.timing.as_ref().map(Arc::clone);
            assert!(timing.is_some());
            engine_factory(components.rank, timing)
                .unwrap()
                .build(EngineIdentity::new(0), NonZeroU32::MIN)
                .unwrap();
        }
    }

    #[test]
    fn disaggregated_roles_resolve_builtin_and_external_timing_independently() {
        let prefill_args = MockerConfig::from_value(serde_json::json!({})).unwrap();
        let mut decode_args = MockerConfig::from_value(serde_json::json!({})).unwrap();
        decode_args.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(EchoAis)));

        let (config, factory) = disaggregated_replay_setup(&prefill_args, &decode_args).unwrap();
        let prefill = config.prefill.as_ref().unwrap();
        let decode = config.decode.as_ref().unwrap();
        assert_eq!(prefill.rank.timing_model, TimingModelConfig::Polynomial);
        assert!(matches!(
            decode.rank.timing_model,
            TimingModelConfig::External { ref provider, .. }
                if provider == "dynamo_perf_model"
        ));

        factory
            .role_factory(&config, WorkerStage::Prefill, false)
            .unwrap()
            .build(0)
            .unwrap();
        factory
            .role_factory(&config, WorkerStage::Decode, false)
            .unwrap()
            .build(0)
            .unwrap();
    }

    /// A callback that prices every query at 1 ms and carries `model`.
    struct FittedEcho(Arc<ForwardPassPerfModel>);

    impl AisCallback for FittedEcho {
        fn predict_prefill(&self, _: usize, _: usize, _: usize) -> Result<f64> {
            Ok(1.0)
        }

        fn predict_decode(&self, _: usize, _: usize, _: usize) -> Result<f64> {
            Ok(1.0)
        }

        fn forward_pass_model(&self) -> Option<Arc<ForwardPassPerfModel>> {
            Some(Arc::clone(&self.0))
        }
    }

    /// Mocker arguments whose AIS callback carries an aggregated vLLM learned
    /// model: pure prefill 10 ms, pure decode 5 ms, and, when `mixed`, a pass
    /// that computes both 7 ms. The callback prices every query at 1 ms. The
    /// artifact lives in `dir`.
    fn fitted_ais_args(dir: &std::path::Path, mixed: bool) -> MockerConfig {
        let artifact = dir.join("learned.json");
        let store = |ms: f64| serde_json::json!({"baseline": ms, "trees": []});
        let mut stores = serde_json::json!({
            "pure_prefill": store(10.0),
            "pure_decode": store(5.0),
        });
        if mixed {
            stores["contains_locally_mixed"] = store(7.0);
        }
        std::fs::write(
            &artifact,
            serde_json::json!({
                "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
                "worker_type": "aggregated", "target": "ms",
                "features": ["num_decode_requests"], "metadata": {},
                "stores": stores,
            })
            .to_string(),
        )
        .unwrap();
        let config = serde_json::json!({
            "model": "test/model", "system": "test-gpu", "backend": "vllm",
            "backend_version": "test-version", "worker_type": "aggregated",
            "estimation_mode": "fpm_learned",
            "estimator_config": {
                "fpm_learned": {"artifact_path": artifact, "adaptation": "frozen"},
                "correction": {"enabled": false},
            },
        });
        let model =
            ForwardPassPerfModel::best_available(serde_json::from_value(config.clone()).unwrap())
                .unwrap();
        let mut args = MockerConfig::from_value(serde_json::json!({
            "engine": {"timing_model": {"type": "external", "provider": "ais", "config": config}}
        }))
        .unwrap();
        args.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(FittedEcho(
            Arc::new(model),
        ))));
        args
    }

    #[test]
    fn offline_replay_prices_a_fitted_ais_mixed_pass_as_one_query() {
        let dir = tempfile::tempdir().unwrap();
        let args = fitted_ais_args(dir.path(), true);

        let (config, factory) = aggregated_replay_setup(&args).unwrap();
        let mut engine = factory
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap()
            .build(0)
            .unwrap();
        let submit = |engine: &mut Engine, id: u128, now_ms: f64| {
            let request = Request {
                request_id: uuid::Uuid::from_u128(id),
                tokens: (0..40).collect(),
                max_output_tokens: 4,
                output_token_ids: None,
                session_id: None,
                cache_salt: None,
                images: Vec::new(),
                kv_hint: None,
            };
            engine
                .apply_command_effects(SchedulerCommand::new(0, Command::Submit(request)), now_ms)
                .unwrap();
        };
        // The second prompt arrives while the first decodes, so one pass mixes.
        submit(&mut engine, 1, 0.0);
        let (mut now_ms, mut mixed_passes) = (0.0, 0);
        for pass in 0.. {
            assert!(pass < 32, "the engine never idled");
            if pass == 1 {
                submit(&mut engine, 2, now_ms);
            }
            let Some(started) = engine.execute_pass(now_ms).unwrap() else {
                break;
            };
            let completed = engine
                .complete_pass(started.pass_id, started.end_ms)
                .unwrap();
            let work = &completed.effects.by_rank[0].effects.forward_pass_metrics;
            let expected_ms = match (work.num_prefill_requests > 0, work.num_decode_requests > 0) {
                (true, true) => {
                    mixed_passes += 1;
                    7.0
                }
                (true, false) => 10.0,
                (false, _) => 5.0,
            };
            assert_eq!(
                started.end_ms - now_ms,
                expected_ms,
                "pass {pass}: {work:?}"
            );
            now_ms = started.end_ms;
        }
        assert_eq!(mixed_passes, 1);
    }

    #[test]
    fn offline_replay_refuses_a_fitted_ais_model_with_a_decode_speedup() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = fitted_ais_args(dir.path(), true);
        args.decode_speedup_ratio = 2.0;

        // `{}` shows only the outermost error, as the Python binding does, so
        // the role and AISimulate's reason must both be in it.
        let error = aggregated_replay_setup(&args).err().unwrap().to_string();
        assert!(
            error.contains("fitted AIS timing cannot serve the aggregated role: ")
                && error.contains("decode_speedup_ratio"),
            "{error}"
        );
    }

    #[test]
    fn offline_replay_refuses_a_fitted_ais_model_built_for_another_role() {
        let dir = tempfile::tempdir().unwrap();
        let prefill_args = MockerConfig::from_value(serde_json::json!({
            "engine": {"worker_type": "prefill"}
        }))
        .unwrap();
        // The configured identity says decode, so the launch check passes,
        // but the callback's model was built for an aggregated engine.
        let mut decode_args = fitted_ais_args(dir.path(), true);
        decode_args.worker_type = WorkerType::Decode;
        let TimingModelConfig::External { config, .. } =
            &mut decode_args.engine.engine.timing_model
        else {
            panic!("the fixture configures an external AIS timing model");
        };
        config["worker_type"] = serde_json::json!("decode");

        let error = format!(
            "{:#}",
            disaggregated_replay_setup(&prefill_args, &decode_args)
                .err()
                .unwrap()
        );
        assert!(
            error.contains(
                "the fitted AIS model is for the aggregated role, but this engine is decode"
            ),
            "{error}"
        );
    }

    #[test]
    fn offline_replay_keeps_the_callback_for_a_fitted_ais_model_without_mixed_passes() {
        // Like a model tuned from profiler points: no locally mixed store, so
        // AISimulate's iteration timing cannot price an aggregated vLLM pass.
        let dir = tempfile::tempdir().unwrap();
        let args = fitted_ais_args(dir.path(), false);

        let (config, factory) = aggregated_replay_setup(&args).unwrap();
        let mut engine = factory
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap()
            .build(0)
            .unwrap();
        let request = Request {
            request_id: uuid::Uuid::from_u128(1),
            tokens: (0..40).collect(),
            max_output_tokens: 4,
            output_token_ids: None,
            session_id: None,
            cache_salt: None,
            images: Vec::new(),
            kv_hint: None,
        };
        engine
            .apply_command_effects(SchedulerCommand::new(0, Command::Submit(request)), 0.0)
            .unwrap();
        let prefill = engine.execute_pass(0.0).unwrap().unwrap();

        // The callback prices the prefill pass at 1 ms; the fitted store would
        // price it at 10 ms.
        assert_eq!(prefill.end_ms, 1.0);
    }
}

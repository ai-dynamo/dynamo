// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Dynamo-to-AISimulate engine configuration boundary.

use std::sync::Arc;

use aisimulate_engine::{
    Backend, EngineConfig, EngineFactory, PreemptionMode as EnginePreemptionMode, SglangConfig,
    SglangSchedulePolicy, TimingModel, TimingModelConfig, TransferTimingMode,
    WorkerType as EngineWorkerType,
};
use aisimulate_replay::{ReplayEngineConfig, ReplayEngineFactory, ReplayRoleConfig};
use anyhow::{Context, Result};
use serde_json::Value;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::{
    EngineType, KvTransferTimingMode, MockEngineArgs, PreemptionMode, WorkerType,
};

/// Fully materialized rank configuration and its optional process-local
/// external timing provider.
pub(crate) struct EngineComponents {
    pub(crate) args: MockEngineArgs,
    pub(crate) rank: EngineConfig,
    pub(crate) timing: Option<Arc<dyn TimingModel>>,
}

/// Normalize Dynamo mock-engine arguments and materialize the neutral engine
/// rank contract.
///
/// Attention-DP size remains a grouped-engine concern and is intentionally not
/// copied into [`EngineConfig`].
pub(crate) fn engine_components(
    args: MockEngineArgs,
    emit_kv_events: bool,
    emit_kv_token_ids: bool,
) -> Result<EngineComponents> {
    let args = args
        .normalized()
        .context("invalid Mocker engine arguments")?;
    let backend = match args.engine_type {
        EngineType::Vllm => Backend::Vllm,
        EngineType::Sglang => Backend::Sglang,
        EngineType::Trtllm => Backend::Trtllm,
    };
    let worker_type = match args.worker_type {
        WorkerType::Aggregated => EngineWorkerType::Aggregated,
        WorkerType::Prefill => EngineWorkerType::Prefill,
        WorkerType::Decode => EngineWorkerType::Decode,
    };
    let preemption_mode = match args.preemption_mode {
        PreemptionMode::Lifo => EnginePreemptionMode::Lifo,
        PreemptionMode::Fifo => EnginePreemptionMode::Fifo,
    };
    let kv_transfer_timing_mode = match args.kv_transfer_timing_mode {
        KvTransferTimingMode::FullPrompt => TransferTimingMode::FullPrompt,
        KvTransferTimingMode::DestinationMissing => TransferTimingMode::DestinationMissing,
    };
    let sglang_args = args.sglang.as_ref();
    let schedule_policy = match sglang_args.and_then(|sglang| sglang.schedule_policy.as_deref()) {
        Some("lpm") => SglangSchedulePolicy::Lpm,
        Some("fifo") | Some("fcfs") | None => SglangSchedulePolicy::Fifo,
        Some(other) => {
            tracing::warn!(
                schedule_policy = other,
                "unknown SGLang schedule policy; using FIFO"
            );
            SglangSchedulePolicy::Fifo
        }
    };
    let sglang = SglangConfig {
        schedule_policy,
        max_prefill_tokens: sglang_args
            .and_then(|sglang| sglang.max_prefill_tokens)
            .unwrap_or(16_384),
        chunked_prefill_size: sglang_args
            .and_then(|sglang| sglang.chunked_prefill_size)
            .unwrap_or(8_192),
        clip_max_new_tokens: sglang_args
            .and_then(|sglang| sglang.clip_max_new_tokens)
            .unwrap_or(4_096),
        schedule_conservativeness: sglang_args
            .and_then(|sglang| sglang.schedule_conservativeness)
            .unwrap_or(1.0),
    };
    let (timing_model, timing) = match args.perf_model.as_ref() {
        PerfModel::Polynomial => (TimingModelConfig::Polynomial, None),
        PerfModel::Interpolated { .. } | PerfModel::Aiconfigurator { .. } => (
            TimingModelConfig::External {
                provider: "dynamo_perf_model".to_string(),
                config: Value::Null,
            },
            Some(Arc::new(DynamoPerfTimingModel {
                inner: Arc::clone(&args.perf_model),
            }) as Arc<dyn TimingModel>),
        ),
    };
    let rank = EngineConfig {
        backend,
        num_gpu_blocks: args.num_gpu_blocks,
        block_size: args.block_size,
        max_model_len: args.max_model_len,
        max_num_seqs: args.max_num_seqs.unwrap_or(usize::MAX),
        max_num_batched_tokens: args.max_num_batched_tokens.unwrap_or(usize::MAX),
        enable_prefix_caching: args.enable_prefix_caching,
        enable_chunked_prefill: args.enable_chunked_prefill,
        speedup_ratio: args.speedup_ratio,
        decode_speedup_ratio: args.decode_speedup_ratio,
        aic_nextn: args.aic_nextn,
        aic_nextn_accept_rates: args.aic_nextn_accept_rates.clone(),
        aic_mtp_seed: args.aic_mtp_seed,
        worker_type,
        preemption_mode,
        emit_kv_events,
        emit_kv_token_ids,
        kv_bytes_per_token: args.kv_bytes_per_token,
        kv_transfer_bandwidth: args.kv_transfer_bandwidth,
        kv_transfer_timing_mode,
        timing_model,
        sglang,
        ..EngineConfig::for_backend(backend)
    };
    Ok(EngineComponents { args, rank, timing })
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

fn replay_tensor_parallel_size(args: &MockEngineArgs) -> Result<u32> {
    u32::try_from(args.aic_tp_size.unwrap_or(1))
        .context("Mocker tensor-parallel size exceeds the Replay contract")
}

/// Materialize the serializable engine descriptor and process-local timing
/// provider used by one aggregated Replay invocation.
pub(crate) fn aggregated_replay_setup(
    args: &MockEngineArgs,
) -> Result<(ReplayEngineConfig, ReplayEngineFactory)> {
    let components = engine_components(args.clone(), false, false)?;
    let config = ReplayEngineConfig {
        dp_size: components.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&components.args)?,
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
    prefill_args: &MockEngineArgs,
    decode_args: &MockEngineArgs,
) -> Result<(ReplayEngineConfig, ReplayEngineFactory)> {
    let prefill = engine_components(prefill_args.clone(), false, false)?;
    let decode = engine_components(decode_args.clone(), false, false)?;
    let prefill_role = ReplayRoleConfig {
        dp_size: prefill.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&prefill.args)?,
        rank: prefill.rank,
    };
    let decode_role = ReplayRoleConfig {
        dp_size: decode.args.dp_size,
        tensor_parallel_size: replay_tensor_parallel_size(&decode.args)?,
        rank: decode.rank,
    };
    let config = ReplayEngineConfig {
        dp_size: prefill_role.dp_size,
        tensor_parallel_size: prefill_role.tensor_parallel_size,
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

    use aisimulate_engine::generalized::EngineIdentity;
    use aisimulate_replay::WorkerStage;
    use ndarray_interp::InterpolateError;

    use super::*;
    use crate::common::perf_model::{AicCallback, DecodeInterpolator, PrefillInterpolator};
    use crate::common::protocols::{SglangArgs, TrtllmArgs};

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

    struct EchoAic;

    impl AicCallback for EchoAic {
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
    fn vllm_defaults_materialize_once_at_the_shared_boundary() {
        let args = MockEngineArgs::builder().build().unwrap();
        let components = engine_components(args, true, true).unwrap();

        assert_eq!(components.args.block_size, 64);
        assert_eq!(components.rank.backend, Backend::Vllm);
        assert_eq!(components.rank.block_size, 64);
        assert!(components.rank.emit_kv_events);
        assert!(components.rank.emit_kv_token_ids);
        assert_eq!(components.rank.timing_model, TimingModelConfig::Polynomial);
        assert!(components.timing.is_none());
    }

    #[test]
    fn backend_specific_fields_match_the_engine_contract() {
        let mut sglang = MockEngineArgs::builder().build().unwrap();
        sglang.engine_type = EngineType::Sglang;
        sglang.sglang = Some(SglangArgs {
            schedule_policy: Some("lpm".to_string()),
            page_size: Some(8),
            max_prefill_tokens: Some(512),
            chunked_prefill_size: Some(256),
            clip_max_new_tokens: Some(128),
            schedule_conservativeness: Some(0.5),
        });
        let components = engine_components(sglang, false, false).unwrap();
        assert_eq!(components.rank.backend, Backend::Sglang);
        assert_eq!(components.rank.block_size, 8);
        assert_eq!(
            components.rank.sglang.schedule_policy,
            SglangSchedulePolicy::Lpm
        );
        assert_eq!(components.rank.sglang.chunked_prefill_size, 256);

        let mut trtllm = MockEngineArgs::builder().build().unwrap();
        trtllm.engine_type = EngineType::Trtllm;
        trtllm.trtllm = Some(TrtllmArgs::default());
        let components = engine_components(trtllm, false, false).unwrap();
        assert_eq!(components.rank.backend, Backend::Trtllm);
        assert_eq!(components.rank.block_size, 32);
    }

    #[test]
    fn polynomial_builds_replay_and_live_engines_without_an_external_provider() {
        let args = MockEngineArgs::builder().build().unwrap();
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
    fn npz_and_aic_models_use_the_external_timing_adapter() {
        let models = [
            PerfModel::Interpolated {
                prefill_interp: Arc::new(EchoPrefill),
                decode_interp: Arc::new(EchoDecode),
            },
            PerfModel::from_aic_callback(Arc::new(EchoAic)),
        ];

        for model in models {
            let mut args = MockEngineArgs::builder().build().unwrap();
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
        let prefill_args = MockEngineArgs::builder().build().unwrap();
        let mut decode_args = MockEngineArgs::builder().build().unwrap();
        decode_args.perf_model = Arc::new(PerfModel::from_aic_callback(Arc::new(EchoAic)));

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
}

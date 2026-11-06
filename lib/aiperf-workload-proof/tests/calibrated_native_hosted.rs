// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preparatory calibrated GPU control for the native AIPerf-hosted Replayer.
//!
//! This proves *provider wiring*, not a full-scenario equivalence claim. Per
//! role (prefill and decode), the canonical H100 BF16 op-level silicon timing
//! model and the native KV-cache geometry from `new_with_geometry` are the one
//! timing source for both the native Replayer engines and the real Dynamo KV
//! Router launch config. The router's `MockerConfig` carries the same resolved
//! FPM config as its role and is given a canonical AIS callback, never a
//! `PerfModel::Fixed` placeholder.
//!
//! Scope limit: the shared `run_fixture` branch still uses AIPerf's builtin
//! tokenizer. This control therefore says nothing about tokenizer equivalence
//! for the configured model; the pinned model tokenizer and model metadata are
//! acquired by main before any full-scenario claim. Model identity is explicit
//! via `AISIM_MODEL` (default `Qwen/Qwen3-8B`) and construction fails, rather
//! than falling back, when the checked-in op-level silicon tables cannot serve
//! that identity.

#![cfg(feature = "calibrated-proof")]

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use aiperf_clock_runtime::cancellation::RunCancellation;
use aisimulate_core::engine::{CanonicalTimingModel, EngineConfig, TimingModel, TimingModelConfig};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayComposition, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRoleConfig, ReplayRuntimeInput, ReplaySpec, Replayer,
};
use aisimulate_core::{
    BackendKind, DataType, DatabaseMode,
    perfmodel::{
        EstimationMode, ForwardPassFallbackPolicy, ForwardPassPerfModelConfig,
        ForwardPassWorkerType,
        memory::{
            EstimateSource, KvCacheEstimateOptions, KvCacheEstimateRequest, KvCacheMemoryFraction,
        },
    },
};
use anyhow::{Context, Result, ensure};
use dynamo_mocker::common::perf_model::{AisCallback, PerfModel};
use dynamo_mocker::placement::{
    KvReplayComposition, KvReplayMetadata, MockerConfig, RouterEventObservation,
};
use serde_json::{Value, json};

// Shared with the runner's own tests; reused in place rather than copied.
#[path = "../../../../../../aisimulate/.worktrees/aiperf-owned-runner-proof/crates/tests/aiperf-workload-runner/tests/support/run_fixture.rs"]
mod run_fixture;

const DEFAULT_MODEL: &str = "Qwen/Qwen3-8B";
const SYSTEM: &str = "h100_sxm";
const BACKEND_VERSION: &str = "0.24.0";
const IDENTITY_SEED: [u8; 32] = [7; 32];
const PREFILL_WORKERS: usize = 2;
const DECODE_WORKERS: usize = 2;
/// Matches the fixture's authored hash/block size and the existing native
/// hosted proof. A different block size would make request KV identities and
/// router block accounting disagree.
const BLOCK_SIZE: usize = 8;
const MAX_NUM_SEQS: usize = 8;
const MAX_NUM_BATCHED_TOKENS: usize = 4096;
const FIXED_PREFILL_MS: f64 = 1.0;
const FIXED_DECODE_MS: f64 = 2.0;

fn configured_model() -> (String, &'static str) {
    match std::env::var("AISIM_MODEL") {
        Ok(model) => (model, "AISIM_MODEL"),
        Err(std::env::VarError::NotPresent) => (DEFAULT_MODEL.to_owned(), "default"),
        Err(error) => panic!("AISIM_MODEL is not readable: {error}"),
    }
}

fn systems_root() -> Result<PathBuf> {
    let root = std::env::var_os("AISIM_SYSTEMS_ROOT").context(
        "AISIM_SYSTEMS_ROOT must identify the pinned checked-in systems assets; no skip or synthetic fallback",
    )?;
    let root = PathBuf::from(root);
    let hardware = root.join(format!("{SYSTEM}.yaml"));
    ensure!(
        hardware.is_file(),
        "H100 hardware metadata is required at {}",
        hardware.display()
    );
    Ok(root)
}

fn memory_role(role: ForwardPassWorkerType) -> &'static str {
    match role {
        ForwardPassWorkerType::Prefill => "prefill",
        ForwardPassWorkerType::Decode => "decode",
        _ => "aggregated",
    }
}

/// A fully resolved, immutable per-role canonical FPM config. Callers must not
/// mutate it after construction: the timing model, native memory geometry and
/// router launch config all derive from this exact identity.
fn role_config(
    model: &str,
    systems: &Path,
    role: ForwardPassWorkerType,
) -> ForwardPassPerfModelConfig {
    let mut config = ForwardPassPerfModelConfig::new(model, SYSTEM, BackendKind::Vllm, role);
    config.backend_version = Some(BACKEND_VERSION.into());
    config.tp = 1;
    config.pp = 1;
    config.attention_dp = 1;
    config.gemm_quant_mode = Some("bfloat16".into());
    config.fmha_quant_mode = Some("bfloat16".into());
    config.kvcache_quant_mode = Some("bfloat16".into());
    config.estimation_mode = EstimationMode::OpLevel;
    config.fallback_policy = ForwardPassFallbackPolicy::Deny;
    config.database_mode = DatabaseMode::Silicon;
    config.strict_provenance = true;
    config.systems_paths = vec![systems.to_path_buf()];
    config.estimator_config.correction.enabled = false;
    config
}

struct RoleSetup {
    timing: Arc<CanonicalTimingModel>,
    rank: EngineConfig,
    router: MockerConfig,
    diagnostics: Value,
    geometry: Value,
}

/// Router-side adapter over the exact canonical estimator used by the engine.
/// The owning Mocker boundary has no canonical FPM constructor of its own: an
/// external AIS provider is materialized as a process-local `AisCallback`, so
/// the host must install the real model here instead of leaving Polynomial or
/// choosing `PerfModel::Fixed`.
struct CanonicalAisCallback {
    timing: Arc<CanonicalTimingModel>,
}

impl AisCallback for CanonicalAisCallback {
    fn predict_prefill(&self, batch_size: usize, isl: usize, prefix: usize) -> Result<f64> {
        self.timing.predict_prefill_ms(batch_size, isl, prefix)
    }

    fn predict_decode(
        &self,
        batch_size: usize,
        context_length: usize,
        _output_tokens: usize,
    ) -> Result<f64> {
        let total_kv_tokens = batch_size
            .checked_mul(context_length)
            .context("decode KV overflow")?;
        self.timing
            .predict_decode_ms(batch_size, total_kv_tokens, context_length, total_kv_tokens)
    }
}

fn canonical_role(model: &str, systems: &Path, role: ForwardPassWorkerType) -> Result<RoleSetup> {
    let config = role_config(model, systems, role);
    let memory = KvCacheEstimateRequest {
        engine: serde_json::from_value(json!({
            "schema_version": aisimulate_core::perfmodel::ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": model,
            "system_name": SYSTEM,
            "systems_path": systems,
            "backend": BackendKind::Vllm,
            "backend_version": BACKEND_VERSION,
            "forward_model": "op_level",
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 1,
            "kv_block_size": BLOCK_SIZE,
            "weight_dtype": DataType::Bfloat16,
            "activation_dtype": DataType::Bfloat16,
            "kv_cache_dtype": DataType::Bfloat16,
            "strict_provenance": true,
            "database_mode": DatabaseMode::Silicon,
            // Canonical geometry rejects a disaggregated role that silently
            // inherits the aggregated worker default.
            "worker_type": memory_role(role),
        }))
        .context("complete matching native memory engine identity")?,
        max_num_tokens: 4096,
        max_batch_size: MAX_NUM_SEQS,
        kv_cache_memory_fraction: KvCacheMemoryFraction::OfTotal(0.9),
        gpu_memory_capacity_bytes_override: None,
        cuda_graph_reserved_bytes: 0,
        tolerance_fraction: Some(0.05),
        options: KvCacheEstimateOptions {
            allow_naive_fallback: false,
            allow_hf_config_download: false,
            naive_kv_reservation: 0.8,
        },
    };
    let (timing, memory) =
        CanonicalTimingModel::new_with_geometry(config, memory).with_context(|| {
            format!(
                "calibrated {role:?} timing/memory for {model} on {SYSTEM} \
                 vllm {BACKEND_VERSION}; checked-in op-level silicon tables must \
                 serve this identity and deny fallback"
            )
        })?;

    let provenance = timing
        .provenance()
        .context("canonical construction omitted resolved provenance")?;
    ensure!(
        provenance.selected_estimation_mode == EstimationMode::OpLevel,
        "canonical {role:?} timing did not select op-level estimation"
    );
    ensure!(
        provenance.config.database_mode == DatabaseMode::Silicon,
        "canonical {role:?} timing did not select the silicon database"
    );
    ensure!(
        provenance.config.fallback_policy == ForwardPassFallbackPolicy::Deny,
        "canonical {role:?} timing must deny fallback"
    );
    ensure!(
        provenance.selection_failures.is_empty(),
        "no alternate estimator may qualify the {role:?} proof"
    );
    ensure!(
        memory.source == EstimateSource::Native,
        "canonical {role:?} geometry requires the native memory model"
    );
    ensure!(
        memory.total_kv_size_tokens > 0 && memory.kv_size_per_token_bytes > 0,
        "canonical {role:?} geometry is empty"
    );

    let diagnostics = serde_json::to_value(timing.diagnostics())?;
    ensure!(
        diagnostics["source"] == json!("aic"),
        "calibrated {role:?} timing must be served by the AIC provider: {diagnostics}"
    );

    let adjusted = memory
        .tolerance_adjusted
        .as_ref()
        .context("explicit tolerance geometry")?;
    let num_gpu_blocks = usize::try_from(adjusted.total_kv_size_tokens)? / BLOCK_SIZE;
    ensure!(
        num_gpu_blocks > 0,
        "native {role:?} geometry yields no KV blocks"
    );

    let timing_config = serde_json::to_value(&provenance.config)?;
    let rank = EngineConfig {
        // External AIS timing is priced by the role factory supplied below; the
        // rank still names the exact resolved config so the engine and router
        // cannot silently diverge.
        timing_model: TimingModelConfig::External {
            provider: "aic".into(),
            config: timing_config.clone(),
        },
        max_num_seqs: MAX_NUM_SEQS,
        max_num_batched_tokens: MAX_NUM_BATCHED_TOKENS,
        block_size: BLOCK_SIZE,
        num_gpu_blocks,
        kv_cache_bytes_per_token: Some(usize::try_from(memory.kv_size_per_token_bytes)?),
        enable_prefix_caching: true,
        // Keep the existing native-hosted disaggregated handoff path; chunked
        // prefill is not part of this preparatory control.
        enable_chunked_prefill: false,
        async_scheduling: Some(false),
        ..Default::default()
    };
    let router = canonical_router_config(&rank, &timing_config, timing.clone())?;

    Ok(RoleSetup {
        timing,
        rank,
        router,
        diagnostics,
        geometry: serde_json::to_value(&memory)?,
    })
}

/// Build the router launch config through its owning parser and require that it
/// carries the same canonical FPM config. The parser maps external AIS timing
/// to a callback slot; install the real canonical estimator there and fail if
/// the provider or the resulting slot is not what this proof requires.
fn canonical_router_config(
    rank: &EngineConfig,
    timing_config: &Value,
    timing: Arc<CanonicalTimingModel>,
) -> Result<MockerConfig> {
    let mut router = MockerConfig::from_value(json!({"engine": rank.clone()}))
        .context("canonical engine rank must be a valid Mocker launch config")?;
    ensure!(
        router.ais_perf_config() == Some(timing_config),
        "router MockerConfig must carry the same resolved canonical FPM config as the role engine"
    );
    ensure!(
        !matches!(router.perf_model.as_ref(), PerfModel::Fixed { .. }),
        "canonical external timing must never be priced by PerfModel::Fixed"
    );
    router.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(
        CanonicalAisCallback {
            timing: timing.clone(),
        },
    )));
    Ok(router)
}

fn native_run(frame: &str, model: &str) -> Result<aiperf_config::model::BenchmarkRun> {
    let mut run = aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes())
        .context("native strict codec")?
        .run;
    let item = run
        .cfg
        .models
        .as_mut()
        .context("fixture must name models")?
        .items
        .first_mut()
        .context("fixture must name at least one model")?;
    item.name = model.to_owned();
    Ok(run)
}

fn role_descriptor(setup: &RoleSetup) -> ReplayRoleConfig {
    ReplayRoleConfig {
        dp_size: 1,
        tensor_parallel_size: 1,
        prefill_context_parallel_size: 1,
        num_gpu_blocks_is_explicit: Some(true),
        rank: setup.rank.clone(),
    }
}

/// Disaggregated 2P2D spec with separate per-role geometry. Requests are
/// supplied by the hosted participant, not by the serializable spec.
fn spec(prefill: &RoleSetup, decode: &RoleSetup) -> Result<ReplaySpec> {
    serde_json::from_value(json!({
        "version": 1,
        "topology": {
            "kind": "disaggregated",
            "prefill": {"initial_workers": PREFILL_WORKERS},
            "decode": {"initial_workers": DECODE_WORKERS},
            "handoff_latency_ms": 0.0
        },
        "engine": ReplayEngineConfig {
            dp_size: 1,
            tensor_parallel_size: 1,
            prefill_context_parallel_size: 1,
            num_gpu_blocks_is_explicit: Some(true),
            rank: prefill.rank.clone(),
            prefill: Some(role_descriptor(prefill)),
            decode: Some(role_descriptor(decode)),
            ..Default::default()
        },
        "adapters": {"placement": {"provider": "dynamo_kv_router"}},
        "requests": []
    }))
    .context("strict disaggregated ReplaySpec")
}

/// Host the production AIPerf plans on the native Replayer event loop with the
/// real Dynamo KV Router composition and the per-role canonical factories.
fn host(
    run: &aiperf_config::model::BenchmarkRun,
    prefill: &RoleSetup,
    decode: &RoleSetup,
) -> Result<(ReplayReport, Value)> {
    let (participant, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        run,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            IDENTITY_SEED,
        )),
    )
    .context("shared native lowering hosts the production plans")?;
    let prefill_timing: Arc<dyn TimingModel> = prefill.timing.clone();
    let decode_timing: Arc<dyn TimingModel> = decode.timing.clone();
    let composition = KvReplayComposition::disaggregated(
        prefill.router.clone(),
        decode.router.clone(),
        PREFILL_WORKERS,
        DECODE_WORKERS,
        None,
        None,
        None,
    );
    let report = Replayer::with_composition(
        spec(prefill, decode)?,
        ReplayEngineFactory::with_optional_role_timing_models(
            Some(prefill_timing),
            Some(decode_timing),
        ),
        composition,
    )
    .context("real Dynamo KV composition accepts the disaggregated calibrated spec")?
    .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
    .with_capture_options(ReplayCaptureOptions {
        capture_per_request: true,
        determinism: ReplayDeterminism::CanonicalV1,
        ..Default::default()
    })
    .run()
    .context("native Replayer event loop hosts calibrated AIPerf over the real KV composition")?;
    Ok((report, handle.into_report().context("AIPerf phase report")?))
}

fn per_request_records(report: &ReplayReport) -> Vec<Value> {
    serde_json::to_value(&report.per_request)
        .expect("engine per-request records")
        .as_array()
        .expect("engine per-request records")
        .clone()
}

fn assert_calibrated_branch(report: &ReplayReport, hosted: &Value) {
    let rows = hosted["requests"].as_array().expect("hosted requests");
    assert_eq!(
        rows.len(),
        4,
        "the seeded strict frame has two parent and two child requests"
    );
    for row in rows {
        assert_eq!(row["status"], "settled");
        assert_eq!(row["terminal"], "completed");
        assert_eq!(row["output_token_ids"].as_array().expect("tokens").len(), 2);
    }
    let correlations: BTreeSet<_> = rows
        .iter()
        .map(|row| {
            row["request"]["correlation_id"]
                .as_str()
                .expect("correlation id")
        })
        .collect();
    assert_eq!(
        correlations.len(),
        2,
        "parent and child are distinct sessions"
    );

    let records = per_request_records(report);
    assert_eq!(records.len(), 4, "engine must retain every request");
    for record in &records {
        assert_eq!(record["terminal_status"], "completed");
        let history = record["routing_history"]
            .as_array()
            .expect("routing history");
        let pools: Vec<&str> = history
            .iter()
            .map(|route| route["pool"].as_str().expect("route pool"))
            .collect();
        assert_eq!(
            pools.iter().filter(|pool| **pool == "prefill").count(),
            1,
            "each request is placed on exactly one prefill worker: {history:?}"
        );
        assert_eq!(
            pools.iter().filter(|pool| **pool == "decode").count(),
            1,
            "each request is placed on exactly one decode worker: {history:?}"
        );
        assert!(
            record["prefill_worker_idx"]
                .as_u64()
                .expect("prefill worker")
                < PREFILL_WORKERS as u64
        );
        assert!(
            record["decode_worker_idx"].as_u64().expect("decode worker") < DECODE_WORKERS as u64
        );
        let ttft = record["ttft_ms"].as_f64().expect("ttft");
        assert!(ttft > 0.0, "calibrated TTFT must be positive: {ttft}");
        assert!(
            record["terminal_time_ms"].as_f64().expect("terminal")
                > record["arrival_time_ms"].as_f64().expect("arrival"),
            "calibrated terminal must follow arrival"
        );
    }
}

/// Positive causal control: the calibrated model is shape dependent on both
/// roles and cannot coincide with the fixed `1ms/2ms` per-pass constants for
/// every probed shape. This is the cheap substitute for an unavailable native
/// per-pass trace; the full run's raw per-request TTFTs are printed alongside.
fn assert_not_fixed(prefill: &RoleSetup, decode: &RoleSetup) -> Result<(f64, f64, f64, f64)> {
    let prefill_small = prefill.timing.predict_prefill_ms(1, 16, 0)?;
    let prefill_large = prefill.timing.predict_prefill_ms(1, 2048, 0)?;
    ensure!(
        prefill_small > 0.0 && prefill_large > 0.0,
        "calibrated prefill timings must be positive"
    );
    ensure!(
        prefill_small != prefill_large,
        "calibrated prefill must vary with prompt shape, not stay a fixed per-pass constant"
    );
    ensure!(
        prefill_small != FIXED_PREFILL_MS || prefill_large != FIXED_PREFILL_MS,
        "calibrated prefill is indistinguishable from the fixed {FIXED_PREFILL_MS}ms control"
    );

    let decode_small = decode.timing.predict_decode_ms(1, 32, 32, 32)?;
    let decode_large = decode.timing.predict_decode_ms(1, 4096, 4096, 4096)?;
    ensure!(
        decode_small > 0.0 && decode_large > 0.0,
        "calibrated decode timings must be positive"
    );
    ensure!(
        decode_small != decode_large,
        "calibrated decode must vary with context, not stay a fixed per-pass constant"
    );
    ensure!(
        decode_small != FIXED_DECODE_MS || decode_large != FIXED_DECODE_MS,
        "calibrated decode is indistinguishable from the fixed {FIXED_DECODE_MS}ms control"
    );
    Ok((prefill_small, prefill_large, decode_small, decode_large))
}

#[test]
fn calibrated_h100_fpm_drives_native_hosted_real_dynamo_kv_2p2d() {
    let (model, model_source) = configured_model();
    assert!(!model.trim().is_empty(), "AISIM_MODEL must not be empty");
    let systems = systems_root().expect("pinned systems assets");

    let prefill = canonical_role(&model, &systems, ForwardPassWorkerType::Prefill)
        .expect("prefill canonical op-level timing and native geometry");
    let decode = canonical_role(&model, &systems, ForwardPassWorkerType::Decode)
        .expect("decode canonical op-level timing and native geometry");
    let (prefill_small, prefill_large, decode_small, decode_large) =
        assert_not_fixed(&prefill, &decode).expect("fixed-vs-calibrated causal control");

    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&run_fixture::branch_request(directory.path()), &model)
        .expect("native strict run with the configured model identity");
    let (report, hosted) = host(&run, &prefill, &decode).expect("calibrated native proof");
    assert_calibrated_branch(&report, &hosted);

    let mut ttfts: Vec<f64> = per_request_records(&report)
        .iter()
        .map(|record| record["ttft_ms"].as_f64().expect("ttft"))
        .collect();
    ttfts.sort_by(f64::total_cmp);
    let ttft_nonconstant = ttfts
        .windows(2)
        .any(|pair| pair[0].to_bits() != pair[1].to_bits());

    println!(
        "CALIBRATED_NATIVE_HOSTED_RAW {}",
        serde_json::to_string_pretty(&json!({
            "proof": "calibrated-native-hosted-h100-op-level-bf16-silicon-deny-fallback",
            "scope": "preparatory GPU provider control; not a full-scenario or tokenizer-equivalence claim",
            "model": model,
            "model_source": model_source,
            "system": SYSTEM,
            "backend": "vllm",
            "backend_version": BACKEND_VERSION,
            "precision": "bfloat16",
            "tokenizer": serde_json::to_value(&run.cfg.tokenizer).expect("tokenizer descriptor"),
            "tokenizer_note": "fixture builtin tokenizer; main acquires the pinned model tokenizer before a full-scenario claim",
            "roles": {
                "prefill": {
                    "timing": prefill.diagnostics,
                    "native_kv_geometry": prefill.geometry,
                    "router_perf_model": "canonical_ais_callback",
                    "replay_rank": prefill.rank,
                },
                "decode": {
                    "timing": decode.diagnostics,
                    "native_kv_geometry": decode.geometry,
                    "router_perf_model": "canonical_ais_callback",
                    "replay_rank": decode.rank,
                },
            },
            "causal_control": {
                "fixed": {"prefill_ms": FIXED_PREFILL_MS, "decode_ms": FIXED_DECODE_MS},
                "calibrated": {
                    "prefill_16_ms": prefill_small,
                    "prefill_2048_ms": prefill_large,
                    "decode_32_ms": decode_small,
                    "decode_4096_ms": decode_large,
                },
            },
            "actual_ttft_ms": ttfts,
            "actual_ttft_nonconstant": ttft_nonconstant,
            "native_engine_report": report,
            "hosted_report": hosted,
        }))
        .expect("calibrated native artifact")
    );
}

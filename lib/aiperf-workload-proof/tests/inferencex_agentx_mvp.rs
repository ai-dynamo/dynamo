// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full InferenceX AgentX MVP scenario construction and end-to-end proof.
//!
//! This file builds the exact user scenario from the shared `run_fixture`
//! frame, replacing the inline branch dataset with the public, revision-pinned
//! `semianalysis_cc_traces_weka_062126` corpus without reducing it. The full
//! E2E then hosts the production AIPerf plans on the native AISim `Replayer`
//! event loop with:
//!
//! * canonical B200 op-level role timing with explicit FP8 weights/MoE/KV and BF16 attention;
//! * the real Dynamo KV Router composition produced by
//!   `KvReplayComposition::disaggregated`;
//! * the caller-owned `HostedTypedReport`, finalized into the normal native-v2
//!   report and GenAI-Perf v1.7 summary by
//!   `hosted_report::finalize_hosted_run_report`.
//!
//! Scenario invariants are sourced from
//! `aiperf_agentx::scenario::agentx()`:
//!
//! * timing mode `agentic_replay`, streaming, `ignore_eos`
//! * first-turn-prefix cache bust
//! * 10-second global system-idle cap
//! * 3600-second profile duration (the full default, not the 900-second lower
//!   bound and not the 1200-second `agentx_fast` diagnostic)
//! * concurrency 32, seed 42
//! * named `agentx` trajectory-start ratios 0.25..=0.75
//! * ten live warmup requests per lane, 1800s warmup grace, 300s tree-idle cap
//! * server/native usage counts, 30s stats cadence, 1s metric slices, no GPU telemetry
//! * no request or session caps, finite replay disabled
//! * no `unsafe_override`, all corpus sources retained
//! * direct trace-hash input with no text tokenizer; canonical model-config
//!   bytes, not tokenizer assets, establish the 1M-token context capacity
//!
//! The full E2E deliberately holds no Dynamo parity claim: the normal native
//! report is finalized with empty `ReportPairRunFacts`, and the raw native
//! backend evidence plus canonical role geometry are written separately as
//! `native_backend_report.json` with `dynamo_parity: false`. That file is
//! provenance, never substituted for typed parity facts, and it records its own
//! designation as not a frozen parity result together with the consumed canonical
//! model-config hashes and the runner executable identity that produced it.
//!
//! # Running the full E2E
//!
//! The full scenario needs external inputs and is `#[ignore]`d so ordinary
//! test sweeps stay fast. Run it explicitly once the inputs exist:
//!
//! ```text
//! cargo test -p dynamo-aiperf-workload-proof --features calibrated-proof \
//!     --test inferencex_agentx_mvp -- --ignored
//! ```
//!
//! Required environment:
//!
//! * `MVP_AISIM_CONFIG` — an authored canonical AISim YAML configuration. The
//!   owning compiler/factory resolves it; this proof verifies the accepted
//!   scenario's physical identity without deriving or overriding its values.
//! * `MVP_ARTIFACT_DIR` — a new directory. The test refuses to reuse an
//!   existing path and never uses a temporary directory, so the complete
//!   result survives.
//!
//! A separate `shared_agentx_legacy_polynomial_round_robin_comparison` test
//! consumes `MVP_COMPARISON_AISIM_CONFIG` through the owning AISim config path.
//! It uses polynomial timing, native round-robin placement and legacy client
//! counts, with explicit non-calibrated/TraceHashes-versus-Text limitations.
//! It never selects or weakens the accepted calibrated full-scenario admission.

#![cfg(feature = "calibrated-proof")]

use std::{
    fs::File,
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
    rc::Rc,
};

use aiperf_clock_runtime::cancellation::RunCancellation;
use aiperf_config::model::{
    BenchmarkRun,
    dataset::{CacheBust, CacheBustTarget, Dataset, PublicDataset, Sampling},
    phase::{Phase, PhaseCommon, PhaseKind as ConfigPhaseKind},
    transport::{AISimulateMode, Transport},
    workload_kind::workload_kind,
};
use aiperf_report_model::report_output::{
    ReportEndpointProfileIdentity, ReportPairRunFacts, ReportRunMetadata, ReportSeedReplayScope,
};
use aiperf_timing::PhaseKind as ReportPhaseKind;
use aisim_aiperf_workload_runner::{
    hosted_report::{FinalizedHostedReport, finalize_hosted_run_report},
    participant_adapter::{HostedTypedReport, native_hosted},
};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayDeterminism, ReplayReport, ReplayRuntimeInput, Replayer,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use dynamo_mocker::placement::KvReplayComposition;
use serde_json::{Value, json};

// Shared with the runner's own tests; reused in place rather than copied.
#[allow(dead_code)]
#[path = "../../../../../../aisimulate/.worktrees/aiperf-owned-runner-proof/crates/tests/aiperf-workload-runner/tests/support/run_fixture.rs"]
mod run_fixture;

// Shared calibrated role/spec/host control owned by main. This target uses the
// role/spec/geometry helpers directly; the other shared control entry points
// are exercised by their own standalone target.
#[allow(dead_code)]
#[path = "support/calibrated.rs"]
mod calibrated;

#[path = "support/reconcile_hosted.rs"]
mod reconcile_hosted;

fn init_debug_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(
                "warn,aisim_aiperf_workload_runner=debug,aiperf_engine_phase::agentic_replay=debug,aiperf_timing::runner=debug"
            ));
        let _ = tracing_subscriber::fmt().with_env_filter(filter)
            .with_ansi(false).with_target(true).with_thread_ids(true)
            .with_writer(std::io::stderr).try_init();
    });
}

/// `aiperf_agentx::scenario::AGENTX_SCENARIO`.
const AGENTX_SCENARIO: &str = "agentx";
/// `aiperf_agentx::scenario::agentx()` corpus alias.
const AGENTX_CORPUS: &str = "semianalysis_cc_traces_weka_062126";
/// The alias's pinned corpus revision, mirrored from the catalog entry.
const AGENTX_CORPUS_REVISION: &str = "23f152f6f0f9399a85901b89a6458def0ef16729";
/// Full scenario profile duration. Not 900 (the lower bound) and not 1200
/// (the `agentx_fast` diagnostic). Cross-checked against the scenario spec.
const PROFILE_SECONDS: f64 = 3600.0;
/// Scenario global system-idle cap. Cross-checked against the scenario spec.
const SYSTEM_IDLE_SECONDS: f64 = 10.0;
/// Scenario concurrency.
const CONCURRENCY: u32 = 32;
/// Scenario seed.
const SEED: u64 = 42;
/// Scenario default profiling grace period from the shared canonical plan.
const GRACE_PERIOD_SECONDS: f64 = 30.0;
/// Operator-selected full-MVP window; explicit values override scenario defaults.
const TRAJECTORY_START_MIN_RATIO: f64 = 0.25;
const TRAJECTORY_START_MAX_RATIO: f64 = 0.75;
/// Canonical role model identity. Same model on every role and on the run.
const MODEL: &str = "zai-org/GLM-5.2-FP8";
/// Canonical model maximum context, cross-checked against the consumed model
/// config. Both roles explicitly use FP8 weights/MoE with BF16 attention/KV.
const MODEL_MAX_CONTEXT_TOKENS: u64 = 1_048_576;
/// Complete root trees the full pinned corpus prepares.
const EXPECTED_PREPARED_ROOTS: usize = 393;
/// Normal native report schema version.
const NATIVE_REPORT_SCHEMA: &str = "2.0";
/// Normal GenAI-Perf summary schema version.
const SUMMARY_SCHEMA: &str = "1.7";
/// Historical summary name used when no export stem is authored.
const SUMMARY_ARTIFACT: &str = "profile_export_aiperf.json";
/// Separate raw-backend provenance written beside the normal artifacts.
const RAW_BACKEND_ARTIFACT: &str = "native_backend_report.json";

/// Build the exact InferenceX AgentX MVP run from the strict native fixture.
///
/// The base frame is the shared `run_fixture::request`, so every non-dataset
/// section keeps its fully materialized native representation. Scenario-owned
/// dataset, phase, transport, export, and replay policy members are replaced
/// from the current `agentx()` preset, including its full CLI-default bundle.
/// Concurrency and profiling grace remain explicit deployment inputs.
///
/// This call writes nothing into `directory` and needs no external tokenizer
/// assets, so it runs as a preflight before the artifact directory is created.
fn full_mvp_run(directory: &Path) -> Result<BenchmarkRun> {
    let scenario = aiperf_agentx::scenario::agentx();
    ensure!(
        scenario.name == AGENTX_SCENARIO,
        "scenario spec names {}, expected {AGENTX_SCENARIO}",
        scenario.name
    );
    ensure!(
        scenario.default_benchmark_duration_seconds == Some(PROFILE_SECONDS as i64),
        "scenario default duration must stay the full {PROFILE_SECONDS}s"
    );
    let profile_seconds = scenario
        .default_benchmark_duration_seconds
        .context("scenario must declare the full profile duration")?
        as f64;
    let system_idle_seconds = scenario
        .system_idle_gap_cap_seconds
        .context("scenario must declare the global system-idle cap")?;
    // Explicit ratios are honored by current Python origin/main's scenario
    // resolver; they are not invariant locks. Keep the requested 25-75 window
    // without changing the distinct named `agentx` preset's other defaults.
    let trajectory_min = TRAJECTORY_START_MIN_RATIO;
    let trajectory_max = TRAJECTORY_START_MAX_RATIO;
    ensure!(
        system_idle_seconds == SYSTEM_IDLE_SECONDS,
        "scenario system-idle cap changed to {system_idle_seconds}"
    );

    let (catalog_key, meta) = aiperf_config::model::public_catalog::lookup_entry(AGENTX_CORPUS)
        .with_context(|| format!("{AGENTX_CORPUS} must be in the public dataset catalog"))?;
    ensure!(
        catalog_key == AGENTX_CORPUS,
        "catalog normalized {AGENTX_CORPUS} to {catalog_key}"
    );
    ensure!(
        meta.format == "weka_trace",
        "{AGENTX_CORPUS} must load through the native weka_trace loader, got {}",
        meta.format
    );
    ensure!(
        meta.source["dataset"] == json!("semianalysisai/cc-traces-weka-062126"),
        "catalog source must be the public Semianalysis 062126 corpus: {}",
        meta.source
    );
    ensure!(
        meta.source["revision"] == json!(AGENTX_CORPUS_REVISION),
        "catalog source must stay pinned to {AGENTX_CORPUS_REVISION}: {}",
        meta.source
    );
    // The 062126 alias already is the with-subagents corpus. An option that
    // filtered sources would silently reduce the workload.
    ensure!(
        meta.options.is_empty(),
        "the full-corpus alias must not carry a source filter: {:?}",
        meta.options
    );

    let base = run_fixture::request(directory);
    let base_bytes = serde_json::to_vec(&base).context("strict base fixture")?;
    let mut run = aiperf_protocol::runner_v2::decode_run_request_v2(&base_bytes)
        .context("strict native base frame")?
        .run;

    run.random_seed = Some(SEED);
    run.cfg.scenario = Some(AGENTX_SCENARIO.to_owned());
    run.cfg.system_idle_gap_cap_seconds = Some(system_idle_seconds);
    run.cfg.finite_replay = Some(false);
    run.cfg.trajectory_start_min_ratio = Some(trajectory_min);
    run.cfg.trajectory_start_max_ratio = Some(trajectory_max);
    run.cfg.unsafe_override = false;
    run.cfg.agentx_fast = false;
    run.cfg.ignore_trace_delays = false;
    run.cfg
        .runtime
        .as_mut()
        .context("runtime policy")?
        .stats_interval = Some(30.0);
    run.cfg
        .metrics
        .as_mut()
        .context("metrics policy")?
        .slice_duration_seconds = Some(1.0);
    run.cfg
        .gpu_telemetry
        .as_mut()
        .context("GPU telemetry policy")?
        .enabled = false;
    // The full scenario executes the legacy WEKA agentic graph projection. The
    // workload id used by report metadata is then derived by the owning
    // `workload_kind` classifier, never spelled by hand.
    run.cfg.weka_engine = Some("legacy".to_owned());

    // The proof harness is hosted by the AISim off-line transport. Parse the
    // owning typed policy rather than constructing its private field set.
    run.cfg.transport = Some(
        serde_json::from_value(json!({"type": "aisimulate", "mode": "offline"}))
            .context("canonical aisimulate transport policy")?,
    );

    // The base proof fixture disables exports. The full scenario publishes the
    // normal native-v2 report and the normal schema-1.7 JSON summary.
    let export = run
        .cfg
        .export
        .as_mut()
        .context("base fixture must carry the export policy")?;
    export.genai_perf.enabled = true;
    export.genai_perf.json = true;
    export.genai_perf.csv = false;

    // Same model on every role and on the run.
    let item = run
        .cfg
        .models
        .as_mut()
        .context("base fixture must name models")?
        .items
        .first_mut()
        .context("base fixture must name at least one model")?;
    item.name = MODEL.to_owned();

    // TraceHashes consumes the authored hash IDs directly. The native wire
    // permits tokenizer:null; no local, remote, or builtin text tokenizer is used.
    run.cfg.tokenizer = None;

    let endpoint = run
        .cfg
        .endpoint
        .as_mut()
        .context("base fixture must carry the endpoint policy")?;
    endpoint.streaming = true;
    endpoint.use_server_token_count = true;
    endpoint
        .extra
        .insert("ignore_eos".to_owned(), Value::Bool(true));

    run.cfg.datasets = Some(vec![Dataset::Public(PublicDataset {
        system_prompt: None,
        context_mode: None,
        osl: None,
        name: catalog_key.to_owned(),
        format: meta.format.clone(),
        source: meta.source.clone(),
        options: meta.options.clone(),
        weka_nested_timestamp_basis: None,
        sampling: Sampling("sequential".to_owned()),
        // No entry/conversation cap: the complete pinned corpus is the input.
        entries: Some(393),
        entries_explicit: Some(false),
        random_seed: Some(SEED),
        prompts: None,
        synthesis: Some(json!({
            "speedup_ratio": 1.0,
            "prefix_len_multiplier": 1.0,
            "prefix_root_multiplier": 1,
            "prompt_len_multiplier": 1.0,
            "output_len_multiplier": 1.0,
            "cache_bust_target": "first_turn_prefix",
            "idle_gap_cap_seconds": 300.0,
        })),
        cache_bust: Some(CacheBust {
            target: CacheBustTarget::FirstTurnPrefix,
        }),
        prefetch_media_urls: false,
        end_conversation_on_truncation: None,
    })]);

    run.cfg.phases = Some(vec![Phase {
        common: PhaseCommon {
            name: "profiling".to_owned(),
            // Resolved from the canonical name.
            kind: None,
            exclude_from_results: false,
            seamless: false,
            // No request or session caps: the 3600-second profile is the only
            // stop condition, and finite replay stays off.
            requests: None,
            sessions: None,
            duration: Some(profile_seconds),
            prefill_concurrency: None,
            grace_period: Some(GRACE_PERIOD_SECONDS),
            concurrency_ramp: None,
            prefill_ramp: None,
            rate_ramp: None,
            cancellation: None,
            agentic_cache_warmup_duration: None,
            warmup_requests_per_lane: Some(10),
            agentic_warmup_grace_period: Some(1800.0),
            failed_request_threshold: Some(0.10),
            timing_mode: Some(scenario.timing_mode.clone()),
            adaptive_scale: None,
            rate_series: None,
        },
        kind: ConfigPhaseKind::Concurrency {
            concurrency: CONCURRENCY,
        },
    }]);

    // Re-enter through the strict native codec and the workload projection so
    // the mutated run is proven to be a valid direct-run frame, not merely a
    // typed struct.
    let frame = aiperf_protocol::runner_v2::RunRequestV2 {
        protocol_version: 2,
        run,
    };
    let encoded = serde_json::to_vec(&frame).context("encode exact MVP frame")?;
    let decoded = aiperf_protocol::runner_v2::decode_run_request_v2(&encoded)
        .context("exact MVP frame must pass the native strict codec")?;
    aiperf_engine_input::execution_input::factory_workload_input_resolved(&decoded.run)
        .context("exact MVP frame must project to a native workload input")?;
    assert_mvp_config(&decoded.run)?;
    Ok(decoded.run)
}

/// Assert the exact scenario invariants directly on the strict run.
pub fn assert_mvp_config(run: &BenchmarkRun) -> Result<()> {
    let scenario = aiperf_agentx::scenario::agentx();
    let profile_seconds = scenario
        .default_benchmark_duration_seconds
        .context("scenario must declare the full profile duration")?
        as f64;
    let system_idle_seconds = scenario
        .system_idle_gap_cap_seconds
        .context("scenario must declare the global system-idle cap")?;
    // Explicit ratios are honored by current Python origin/main's scenario
    // resolver; they are not invariant locks. Keep the requested 25-75 window
    // without changing the distinct named `agentx` preset's other defaults.
    let trajectory_min = TRAJECTORY_START_MIN_RATIO;
    let trajectory_max = TRAJECTORY_START_MAX_RATIO;

    ensure!(
        scenario.timing_mode == "agentic_replay",
        "scenario timing mode must be agentic_replay, got {}",
        scenario.timing_mode
    );
    ensure!(
        scenario.require_streaming && scenario.require_ignore_eos,
        "scenario must require streaming with ignore_eos"
    );
    ensure!(
        scenario.forbid_ignore_trace_delays && scenario.forbid_input_truncation,
        "scenario must forbid ignored trace delays and input truncation"
    );
    ensure!(
        scenario
            .require_loader
            .iter()
            .any(|loader| loader.as_str() == AGENTX_CORPUS),
        "scenario loader allowlist must include {AGENTX_CORPUS}"
    );
    ensure!(
        // The scenario spec carries the exchange-model enum, which is not a
        // direct dependency of this proof crate. Compare its owning serde
        // spelling instead of naming a transitive type.
        serde_json::to_value(scenario.require_cache_bust).ok() == Some(json!("first_turn_prefix")),
        "scenario must require the first-turn-prefix cache bust"
    );
    ensure!(
        scenario.minimum_profile_metric_coverage_ratio == Some(0.95),
        "scenario metric-coverage ratio changed to {:?}",
        scenario.minimum_profile_metric_coverage_ratio
    );

    let dataset = run
        .cfg
        .datasets
        .as_deref()
        .and_then(|rows| rows.first())
        .context("MVP run must carry one dataset")?;
    let Dataset::Public(public) = dataset else {
        bail!("MVP run must use the public corpus, not a private file dataset");
    };
    ensure!(
        public.name == AGENTX_CORPUS,
        "wrong corpus: {}",
        public.name
    );
    ensure!(
        public.format == "weka_trace",
        "wrong loader: {}",
        public.format
    );
    ensure!(
        public.source["dataset"] == json!("semianalysisai/cc-traces-weka-062126"),
        "wrong source repository: {}",
        public.source
    );
    ensure!(
        public.source["revision"] == json!(AGENTX_CORPUS_REVISION),
        "corpus revision is not pinned: {}",
        public.source
    );
    ensure!(
        public.options.is_empty(),
        "source filtering would reduce the corpus: {:?}",
        public.options
    );
    ensure!(
        public.entries == Some(393) && public.entries_explicit == Some(false),
        "agentx defaults must retain all 393 corpus roots"
    );
    ensure!(
        public.cache_bust
            == Some(CacheBust {
                target: CacheBustTarget::FirstTurnPrefix,
            }),
        "first-turn-prefix cache bust is required"
    );
    ensure!(
        public
            .synthesis
            .as_ref()
            .and_then(|value| value["cache_bust_target"].as_str())
            == Some("first_turn_prefix"),
        "synthesis cache bust must agree with the typed cache-bust policy"
    );
    ensure!(
        public.random_seed == Some(SEED),
        "dataset seed must be {SEED}"
    );

    let endpoint = run
        .cfg
        .endpoint
        .as_ref()
        .context("MVP run must carry the endpoint policy")?;
    ensure!(endpoint.streaming, "streaming must be enabled");
    ensure!(
        endpoint.use_server_token_count,
        "agentx uses authoritative native backend usage"
    );
    ensure!(
        run.cfg.runtime.as_ref().and_then(|x| x.stats_interval) == Some(30.0),
        "agentx stats cadence"
    );
    ensure!(
        run.cfg
            .metrics
            .as_ref()
            .and_then(|x| x.slice_duration_seconds)
            == Some(1.0),
        "agentx timeslices"
    );
    ensure!(
        run.cfg.gpu_telemetry.as_ref().is_some_and(|x| !x.enabled),
        "agentx disables real GPU telemetry"
    );
    ensure!(
        public
            .synthesis
            .as_ref()
            .and_then(|x| x["idle_gap_cap_seconds"].as_f64())
            == Some(300.0),
        "agentx tree idle cap"
    );
    ensure!(
        endpoint.extra.get("ignore_eos") == Some(&Value::Bool(true)),
        "ignore_eos must be enabled"
    );

    let transport = run
        .cfg
        .transport
        .as_ref()
        .context("MVP run must select a transport")?;
    ensure!(
        matches!(
            transport,
            Transport::AISimulate(config) if matches!(config.mode, AISimulateMode::Offline)
        ),
        "the MVP scenario must use the typed aisimulate transport in explicit offline mode, got {}",
        transport.canonical_id()
    );
    ensure!(
        workload_kind(&run.cfg).workload_id() == "graph",
        "the legacy WEKA projection must select the graph workload"
    );
    ensure!(
        run.cfg.weka_engine.as_deref() == Some("legacy"),
        "the full MVP uses the legacy WEKA graph projection"
    );

    let export = run
        .cfg
        .export
        .as_ref()
        .context("MVP run must carry the export policy")?;
    ensure!(
        export.genai_perf.enabled && export.genai_perf.json,
        "the full scenario writes the normal schema-1.7 summary"
    );

    let items = &run
        .cfg
        .models
        .as_ref()
        .context("MVP run must name models")?
        .items;
    ensure!(items.len() == 1, "MVP run names exactly one model");
    ensure!(items[0].name == MODEL, "wrong model: {}", items[0].name);

    let phases = run
        .cfg
        .phases
        .as_deref()
        .context("MVP run must name a phase")?;
    ensure!(phases.len() == 1, "MVP run has exactly one profiling phase");
    let phase = &phases[0];
    ensure!(phase.common.name == "profiling", "wrong phase name");
    ensure!(
        phase.common.warmup_requests_per_lane == Some(10),
        "agentx warmup pressure quota"
    );
    ensure!(
        phase.common.agentic_warmup_grace_period == Some(1800.0),
        "agentx warmup grace"
    );
    ensure!(
        phase.common.failed_request_threshold == Some(0.10),
        "agentx failed request threshold"
    );
    ensure!(
        phase.common.requests.is_none() && phase.common.sessions.is_none(),
        "no request or session cap is permitted"
    );
    ensure!(
        phase.common.duration == Some(profile_seconds),
        "profile duration must be the full {profile_seconds}s, got {:?}",
        phase.common.duration
    );
    ensure!(
        phase.common.grace_period == Some(GRACE_PERIOD_SECONDS),
        "profiling grace period must be the shared scenario default {GRACE_PERIOD_SECONDS}s, got {:?}",
        phase.common.grace_period
    );
    ensure!(
        phase.common.timing_mode.as_deref() == Some("agentic_replay"),
        "timing mode must be agentic_replay"
    );
    match &phase.kind {
        ConfigPhaseKind::Concurrency { concurrency } => ensure!(
            *concurrency == CONCURRENCY,
            "concurrency must be {CONCURRENCY}, got {concurrency}"
        ),
        other => bail!("MVP profiling phase must use concurrency arrivals, got {other:?}"),
    }

    ensure!(
        run.cfg.scenario.as_deref() == Some(AGENTX_SCENARIO),
        "scenario name must be {AGENTX_SCENARIO}"
    );
    ensure!(
        run.cfg.system_idle_gap_cap_seconds == Some(system_idle_seconds),
        "system idle cap must be {system_idle_seconds}s"
    );
    ensure!(
        run.cfg.finite_replay == Some(false),
        "finite replay must stay disabled"
    );
    ensure!(
        !run.cfg.agentx_fast,
        "agentx_fast would shorten the profile to 1200s"
    );
    ensure!(
        !run.cfg.unsafe_override,
        "unsafe_override is forbidden for this scenario"
    );
    ensure!(
        run.cfg.trajectory_start_min_ratio == Some(trajectory_min)
            && run.cfg.trajectory_start_max_ratio == Some(trajectory_max),
        "trajectory-start window must be {trajectory_min}..={trajectory_max}"
    );
    ensure!(run.random_seed == Some(SEED), "run seed must be {SEED}");
    ensure!(
        run.cfg.profiling_request_cap().is_none(),
        "no profiling request cap may truncate the corpus"
    );
    ensure!(
        run.cfg.tokenizer.is_none(),
        "trace-hash input must not require a text tokenizer policy"
    );
    Ok(())
}

/// Host the production plans on the native Replayer event loop with the real
/// Dynamo KV composition and the per-role canonical timing factories, retaining
/// the typed AIPerf report instead of a lossy JSON projection.
fn host_typed(
    run: &BenchmarkRun,
    prefill: &calibrated::RoleSetup,
    decode: &calibrated::RoleSetup,
    compact_evidence: bool,
) -> Result<(ReplayReport, HostedTypedReport)> {
    host_typed_using(run, compact_evidence, || {
        let composition = KvReplayComposition::disaggregated(
            prefill.router.clone(),
            decode.router.clone(),
            calibrated::PREFILL_WORKERS,
            calibrated::DECODE_WORKERS,
            None,
            None,
            None,
        );
        Replayer::with_composition(
            calibrated::spec(prefill, decode)?,
            prefill.factory.clone(),
            composition,
        )
        .context("real Dynamo KV composition accepts the disaggregated calibrated spec")
    })
}

/// Only the physical composition varies; preparation, phase futures, capture,
/// clock ownership and typed report collection stay in this one shared path.
fn host_typed_using<C: aisimulate_core::replay::ReplayComposition>(
    run: &BenchmarkRun,
    compact_evidence: bool,
    create_replayer: impl FnOnce() -> Result<Replayer<C>>,
) -> Result<(ReplayReport, HostedTypedReport)> {
    init_debug_tracing();
    eprintln!("MVP stage: preparing complete shared workload");
    let (participant, handle) = native_hosted(
        run,
        aisim_aiperf_workload_runner::participant_adapter::NativeInputProjection::TraceHashes,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            calibrated::IDENTITY_SEED,
        )),
    )
    .context("shared native lowering hosts the production plans")?;
    if run.cfg.scenario.as_deref() == Some(AGENTX_SCENARIO) {
        let lanes = handle.initial_lanes();
        let cuts = lanes.lanes.iter().map(|lane| {
            let tree = handle.source_tree_specs().iter()
                .find(|tree| tree.root == lane.root_id)
                .context("initial lane must retain its canonical source tree")?;
            let root_start = tree.recorded_intervals.iter()
                .filter(|turn| turn.template_id == lane.root_id)
                .filter_map(|turn| turn.timestamp_ms)
                .min_by(f64::total_cmp)
                .context("initial root must have a recorded source start")?;
            Ok(json!({"source_trace_id": lane.root_id, "absolute_t_star_ms": lane.t_star_ms,
                "recorded_root_start_ms": root_start, "relative_t_star_ms": lane.t_star_ms - root_start}))
        }).collect::<Result<Vec<_>>>()?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.artifact_dir.join("initial_snapshot_selections.json"))?;
        serde_json::to_writer_pretty(
            file,
            &json!({"seed": SEED,
            "sampler_draws": lanes.sampler_draws, "recycle_roots": lanes.recycle_roots,
            "selections": cuts}),
        )?;
    }
    // Complete metric records and backend facts remain retained. Repeated
    // prompt bodies/token arrays and per-token diagnostic JSON are explicitly
    // omitted from dispatch evidence; counts and BLAKE3 witnesses name inputs.
    if compact_evidence {
        handle.use_compact_dispatch_evidence()?;
    }
    eprintln!("MVP stage: prepared; entering native simulator event loop");
    let report = create_replayer()?
        .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
        .with_capture_options(ReplayCaptureOptions {
            capture_per_request: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        })
        .run()
        .context("native Replayer event loop hosts the full AIPerf scenario")?;
    eprintln!("MVP stage: native simulator drained; retaining typed report");
    let typed = handle
        .into_typed_report()
        .context("AIPerf typed phase report")?;
    Ok((report, typed))
}

/// Stream the running executable through BLAKE3 without loading it into memory.
fn executable_distribution_id() -> Result<String> {
    let executable = std::env::current_exe().context("locating the running executable")?;
    let mut file = File::open(&executable)
        .with_context(|| format!("opening executable {}", executable.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing executable {}", executable.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

/// Build the coordinator-owned metadata from the typed run. The transport and
/// workload ids come from their owning classifiers; the endpoint profile uses
/// the owning validated identity constructor.
fn report_run_metadata(run: &BenchmarkRun, harness: &str) -> Result<ReportRunMetadata> {
    let transport_policy = run
        .cfg
        .transport
        .as_ref()
        .context("MVP run must select a transport")?;
    ensure!(
        matches!(
            transport_policy,
            Transport::AISimulate(config) if matches!(config.mode, AISimulateMode::Offline)
        ),
        "the full scenario must use the typed aisimulate transport in explicit offline mode, got {}",
        transport_policy.canonical_id()
    );
    let transport = transport_policy.canonical_id().to_owned();
    let workload = workload_kind(&run.cfg).workload_id().to_owned();
    let endpoint = run
        .cfg
        .endpoint
        .as_ref()
        .context("MVP run must select an endpoint profile")?
        .endpoint_type
        .0
        .clone();
    let endpoint_profiles = vec![
        ReportEndpointProfileIdentity::new("default", endpoint)
            .map_err(|error| anyhow!("invalid endpoint profile identity: {error:?}"))?,
    ];
    let mut metadata = ReportRunMetadata::new(
        harness.to_owned(),
        transport,
        workload,
        Vec::new(),
        endpoint_profiles,
    )
    .map_err(|error| anyhow!("invalid report run metadata: {error:?}"))?;
    metadata.random_seed = run.random_seed;
    metadata.random_seed_replay_scope = Some(ReportSeedReplayScope::FreshMaterialization);
    Ok(metadata)
}

/// Preserve the raw native backend report, trace-hash input designation, the
/// runner executable, and canonical role provenance as a separate, explicitly
/// non-parity artifact. The normal report's typed pair facts stay empty, so
/// nothing here can be mistaken for `ReportDynamoRunInfo`.
fn write_raw_backend_artifact(
    artifact_dir: &Path,
    backend: &ReplayReport,
    typed: &HostedTypedReport,
    prefill: &calibrated::RoleSetup,
    decode: &calibrated::RoleSetup,
    harness: &str,
) -> Result<()> {
    let artifact = json!({
        "scope": "raw native AISim Replayer backend and canonical role provenance; \
                  this is not Dynamo parity evidence and is never substituted for \
                  the typed ReportDynamoRunInfo facts",
        "dynamo_parity": false,
        "dynamo_typed_facts_present": false,
        "dynamo_clock": Value::Null,
        "parity_designation": "not a frozen parity designation: this artifact is \
                               provenance for one host, one runner executable, and \
                               one trace-hash input projection, and is never \
                               compared against or substituted for frozen Dynamo \
                               parity facts",
        "harness": {
            "label": "inferencex_agentx_mvp::full_scenario",
            "executable_distribution_id": harness,
        },
        "input_representation": "trace_hashes",
        "text_tokenizer_used": false,
        "transport": "aisimulate",
        "workload": "graph",
        "model_max_context_tokens": MODEL_MAX_CONTEXT_TOKENS,
        "model_max_context_note": "calibration metadata only; the full pinned corpus is never filtered or truncated, and context overflow is classified by the backend",
        "limitations": [
            "no typed Dynamo topology, clock, capacity, or parity facts are supplied; this raw artifact does not claim full Dynamo parity",
            "the full pinned corpus remains unfiltered; long-context overflow is backend-owned and is judged by the normal submission verdict",
        ],
        "prepared_roots": typed.prepared_roots,
        "virtual_end_ns": typed.virtual_end_ns,
        "physical_configuration_source": "AISim CorePredictionConfig/compiler/Runner materializer and native physical factory",
        "prefill": {
            "timing": &prefill.diagnostics,
            "capacity_diagnostics": &prefill.geometry,
            "state_geometry": &prefill.state_geometry,
            "replay_rank": &prefill.rank,
            "transfer": &prefill.transfer,
            "model_config": &prefill.model_config,
        },
        "decode": {
            "timing": &decode.diagnostics,
            "capacity_diagnostics": &decode.geometry,
            "state_geometry": &decode.state_geometry,
            "replay_rank": &decode.rank,
            "transfer": &decode.transfer,
            "model_config": &decode.model_config,
        },
    });
    write_raw_backend_with_provenance(artifact_dir, backend, typed, &artifact)
}

fn write_raw_backend_with_provenance(
    artifact_dir: &Path,
    backend: &ReplayReport,
    typed: &HostedTypedReport,
    artifact: &Value,
) -> Result<()> {
    // Serialize borrowed large reports straight to the artifact. Building a
    // serde_json::Value and a second pretty-printed Vec duplicates every
    // per-request observation at full-scenario scale.
    #[derive(serde::Serialize)]
    struct RawArtifact<'a, T: serde::Serialize, R: serde::Serialize> {
        #[serde(flatten)]
        provenance: &'a Value,
        backend_report: &'a ReplayReport,
        // ReplayReport's owning Serialize deliberately emits aggregates only.
        // Retain its captured request evidence explicitly rather than assuming
        // serializing the report also serializes the population we reconciled.
        backend_requests: &'a [aisimulate_core::replay::PerRequestRecord],
        native_records: Vec<RawPhaseRecords<'a, R>>,
        dispatch_requests: &'a [Value],
        typed_phase_report: &'a T,
    }
    #[derive(serde::Serialize)]
    struct RawPhaseRecords<'a, R: serde::Serialize> {
        phase_index: usize,
        phase_name: &'a str,
        records: &'a R,
    }
    let path = artifact_dir.join(RAW_BACKEND_ARTIFACT);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("creating raw native backend artifact {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(
        &mut writer,
        &RawArtifact {
            provenance: artifact,
            backend_report: backend,
            backend_requests: &backend.per_request,
            native_records: typed
                .records
                .iter()
                .map(|phase| RawPhaseRecords {
                    phase_index: phase.phase_index,
                    phase_name: &phase.phase_name,
                    records: &phase.records,
                })
                .collect(),
            dispatch_requests: &typed.dispatch.requests,
            typed_phase_report: &typed.phased,
        },
    )
    .context("streaming raw native backend artifact")?;
    std::io::Write::flush(&mut writer).context("flushing raw native backend artifact")?;
    Ok(())
}

/// Assert the complete full-scenario evidence: exact corpus population, the
/// actual 3600-second profile span, branch spawn/join counters from the typed
/// phase report, metric coverage from the live accumulator, and the normal
/// native-v2/schema-1.7 artifacts.
fn assert_full_scenario(
    run: &BenchmarkRun,
    typed: &HostedTypedReport,
    finalized: &FinalizedHostedReport,
    artifact_dir: &Path,
) -> Result<()> {
    let scenario = aiperf_agentx::scenario::agentx();
    let profile_seconds = scenario
        .default_benchmark_duration_seconds
        .context("scenario must declare the full profile duration")?
        as f64;
    let required_ratio = scenario
        .minimum_profile_metric_coverage_ratio
        .context("scenario must declare the metric-coverage ratio")?;
    let profile_nanos = (profile_seconds * 1_000_000_000.0) as i64;

    ensure!(
        !run.cfg.unsafe_override && !run.cfg.agentx_fast,
        "the full scenario must not run through unsafe_override or agentx_fast"
    );
    ensure!(
        aiperf_engine_input::execution_input::scenario_locks_satisfied(&run.cfg),
        "scenario locks must hold without any unsafe override"
    );

    ensure!(
        typed.prepared_roots == EXPECTED_PREPARED_ROOTS,
        "the complete corpus must prepare all {EXPECTED_PREPARED_ROOTS} roots, got {}",
        typed.prepared_roots
    );

    let profiling_phase = typed
        .phased
        .phases
        .iter()
        .find(|phase| phase.kind == ReportPhaseKind::Profiling)
        .context("hosted phased report has no profiling phase")?;
    let start_ns = profiling_phase
        .start_ns
        .context("profiling phase must record its actual start")?;
    let end_ns = profiling_phase
        .requests_end_ns
        .context("profiling phase must record its actual end")?;
    ensure!(
        end_ns - start_ns >= profile_nanos,
        "profiling phase ran {}ns, expected at least the full {profile_seconds}s",
        end_ns - start_ns
    );
    // Context rejections, if any, are classified by the backend. The
    // scenario's own submission verdict (the
    // 1% failure threshold) is the acceptance contract, so this proof does not
    // assert a zero general request-error count.

    let profiling_report = typed
        .phased
        .reports
        .iter()
        .find(|report| report.kind == ReportPhaseKind::Profiling)
        .context("hosted phased report has no profiling performance report")?;
    let branch = profiling_report
        .report
        .branch_stats
        .context("AgentX profiling phase must expose branch spawn/join counters")?;
    ensure!(
        branch.children_spawned > 0,
        "the full AgentX profile must spawn branch children"
    );
    ensure!(
        branch.children_completed > 0,
        "the full AgentX profile must complete and join branch children"
    );
    ensure!(
        branch.children_spawned >= branch.children_completed,
        "completed branch children cannot exceed spawned children"
    );

    ensure!(
        !finalized.coverage.is_empty(),
        "the scenario's metric-duration coverage must be present"
    );
    for coverage in &finalized.coverage {
        ensure!(
            coverage.required_ratio.to_bits() == required_ratio.to_bits(),
            "coverage ratio {} disagrees with the scenario's {required_ratio}",
            coverage.required_ratio
        );
        ensure!(
            coverage.expected_duration_seconds >= profile_seconds,
            "coverage expected {}s, below the full {profile_seconds}s",
            coverage.expected_duration_seconds
        );
        ensure!(
            coverage.ttft_ratio >= required_ratio
                || coverage.inter_token_latency_ratio >= required_ratio,
            "phase {} coverage is below {required_ratio}: ttft={} itl={}",
            coverage.phase_name,
            coverage.ttft_ratio,
            coverage.inter_token_latency_ratio
        );
        ensure!(
            coverage.passed(),
            "phase {} did not pass its coverage check",
            coverage.phase_name
        );
    }
    ensure!(
        finalized.coverage_errors.is_empty(),
        "coverage plan validation produced errors: {:?}",
        finalized.coverage_errors
    );
    ensure!(
        finalized.coverage_failures.is_none(),
        "coverage failures were recorded: {:?}",
        finalized.coverage_failures
    );

    let stamped = finalized
        .native_report
        .run
        .run_metadata()
        .context("normal native report must be stamped with run metadata")?;
    ensure!(
        stamped.transport == "aisimulate",
        "native report transport must be aisimulate, got {}",
        stamped.transport
    );
    ensure!(
        stamped.workload == "graph",
        "native report workload must be graph, got {}",
        stamped.workload
    );
    ensure!(
        finalized.native_report.run.facts().dynamo.is_none(),
        "no Dynamo parity facts may be fabricated for this proof"
    );

    let report_path = artifact_dir.join(aiperf_config::model::artifacts::report_artifact_name(
        run.cfg.artifacts.as_ref(),
    ));
    let report: Value = serde_json::from_slice(
        &std::fs::read(&report_path)
            .with_context(|| format!("reading native report {}", report_path.display()))?,
    )
    .with_context(|| format!("parsing native report {}", report_path.display()))?;
    ensure!(
        report["schema_version"] == json!(NATIVE_REPORT_SCHEMA),
        "native report schema must be {NATIVE_REPORT_SCHEMA}: {report}"
    );
    ensure!(
        report["run"]["transport"] == json!("aisimulate"),
        "native report transport must be aisimulate: {report}"
    );
    ensure!(
        report["run"]["workload"] == json!("graph"),
        "native report workload must be graph: {report}"
    );

    let summary_path = artifact_dir.join(SUMMARY_ARTIFACT);
    let summary: Value = serde_json::from_slice(
        &std::fs::read(&summary_path)
            .with_context(|| format!("reading summary {}", summary_path.display()))?,
    )
    .with_context(|| format!("parsing summary {}", summary_path.display()))?;
    ensure!(
        summary["schema_version"] == json!(SUMMARY_SCHEMA),
        "GenAI-Perf summary schema must be {SUMMARY_SCHEMA}: {summary}"
    );
    ensure!(
        summary["metadata"]["submission_valid"] == json!(true),
        "the full scenario submission must be valid: {summary}"
    );
    // ScenarioSubmission's owning serializer omits an empty reason list.
    // Require absence or an empty array, never a malformed/nonempty value.
    if let Some(invalid_reasons) = summary["metadata"].get("submission_invalid_reasons") {
        ensure!(
            invalid_reasons.as_array().is_some_and(Vec::is_empty),
            "valid submission must carry no invalid reasons: {invalid_reasons:?}"
        );
    }

    Ok(())
}

/// Read and validate `MVP_ARTIFACT_DIR` without creating it. Creation waits for
/// every static check, so a failed preflight leaves no artifact directory.
fn mvp_artifact_path() -> Result<PathBuf> {
    let raw = std::env::var_os("MVP_ARTIFACT_DIR").context(
        "MVP_ARTIFACT_DIR must name a new, empty directory for the complete \
         full-scenario artifacts; the full E2E never uses a temporary directory \
         and refuses to overwrite an existing one",
    )?;
    let path = PathBuf::from(raw);
    let leaf = path
        .file_name()
        .context("MVP_ARTIFACT_DIR must name a directory leaf, not a filesystem root")?;
    ensure!(
        !leaf.is_empty(),
        "MVP_ARTIFACT_DIR must name a directory leaf"
    );
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        ensure!(
            parent.is_dir(),
            "MVP_ARTIFACT_DIR parent must already exist: {}",
            parent.display()
        );
    }
    Ok(path)
}

/// Create the fresh artifact directory. `create_dir` (never `create_dir_all`)
/// keeps the parent under the operator's control and refuses an existing leaf.
fn create_new_artifact_dir(path: &Path) -> Result<PathBuf> {
    match std::fs::create_dir(path) {
        Ok(()) => Ok(path.to_path_buf()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => bail!(
            "MVP_ARTIFACT_DIR already exists; refusing to overwrite full-scenario artifacts: {}",
            path.display()
        ),
        Err(error) => Err(anyhow!(
            "creating MVP_ARTIFACT_DIR {}: {error}",
            path.display()
        )),
    }
}

fn full_scenario_config() -> Result<Value> {
    let path = PathBuf::from(
        std::env::var_os("MVP_AISIM_CONFIG")
            .context("MVP_AISIM_CONFIG must identify an authored canonical AISim configuration")?,
    );
    let bytes = aiperf_fsio::artifact_file::read_no_follow_bounded(&path, 1 << 20)
        .context("bounded no-follow canonical AISim configuration")?;
    Ok(serde_yaml::from_slice(&bytes)?)
}

fn full_scenario_roles() -> Result<(calibrated::RoleSetup, calibrated::RoleSetup)> {
    full_scenario_roles_from_config(full_scenario_config()?)
}

fn full_scenario_roles_from_config(
    config: Value,
) -> Result<(calibrated::RoleSetup, calibrated::RoleSetup)> {
    let (prefill, decode) = calibrated::roles_from_config(config)?;
    let spec = calibrated::spec(&prefill, &decode)?;
    ensure!(
        matches!(&spec.topology,
        aisimulate_core::replay::ReplayTopology::Disaggregated { prefill, decode, .. }
            if prefill.initial_workers == 2 && decode.initial_workers == 2),
        "resolved physical deployment must match the accepted 2P2D scenario"
    );
    for (name, role) in [("prefill", &prefill), ("decode", &decode)] {
        let aisimulate_core::engine::TimingModelConfig::External { provider, config } =
            &role.rank.timing_model
        else {
            bail!("{name} must retain the canonical AIS timing identity");
        };
        ensure!(
            provider == "aic"
                && config["model"] == MODEL
                && config["system"] == "b200_sxm"
                && config["backend"] == "vllm"
                && config["backend_version"] == calibrated::BACKEND_VERSION
                && config["worker_type"] == name
                && config["tp"] == 8
                && config["pp"] == 1
                && config["attention_dp"] == 1,
            "{name} resolved model/hardware/role/topology differs from the accepted scenario"
        );
        ensure!(
            config["gemm_quant_mode"] == "fp8"
                && config["moe_quant_mode"] == "fp8"
                && config["kvcache_quant_mode"] == "fp8"
                && config["fmha_quant_mode"] == "bfloat16",
            "{name} resolved precision differs from the accepted scenario"
        );
        ensure!(
            config["estimation_mode"] == "op_level"
                && config["fallback_policy"] == "deny"
                && config["database_mode"] == "SILICON",
            "{name} must use the accepted canonical silicon op-level selection"
        );
        ensure!(
            role.rank.max_num_seqs == CONCURRENCY as usize,
            "{name} resolved sequence budget differs from the accepted scenario"
        );
        ensure!(
            role.rank.max_num_batched_tokens == 4096
                && role.rank.block_size == calibrated::BLOCK_SIZE
                && role.rank.max_model_len == Some(MODEL_MAX_CONTEXT_TOKENS as usize)
                && role.rank.enable_prefix_caching
                && role.rank.enable_chunked_prefill
                && role.rank.async_scheduling == Some(false),
            "{name} resolved scheduler/context differs from the accepted scenario"
        );
        let descriptor = calibrated::role_descriptor(role);
        ensure!(
            descriptor.dp_size == 1
                && descriptor.tensor_parallel_size == 8
                && descriptor.prefill_context_parallel_size == 1
                && descriptor.num_gpu_blocks_is_explicit == Some(false)
                && role.rank.num_gpu_blocks > 0,
            "{name} must retain the owner-inferred capacity and accepted rank topology"
        );
        ensure!(
            role.rank
                .kv_transfer_bytes_per_token
                .is_some_and(|bytes| bytes > 0)
                && role.rank.kv_transfer_bandwidth == Some(50.0)
                && role.rank.kv_transfer_timing_mode
                    == aisimulate_core::engine::TransferTimingMode::FullPrompt,
            "{name} must retain a compiler-resolved transfer footprint and accepted link"
        );
    }
    Ok((prefill, decode))
}

#[test]
#[ignore = "requires MVP_AISIM_CONFIG; negative physical-identity admission control, no corpus execution"]
fn calibrated_full_scenario_rejects_mismatched_physical_identity() -> Result<()> {
    let mut config = full_scenario_config()?;
    config["engine"]["workers"]["prefill"]["scheduler"]["max_sequences"] = json!(31);
    let result = full_scenario_roles_from_config(config);
    let error = match result {
        Ok(_) => bail!("changed authored physical identity was silently accepted"),
        Err(error) => error,
    };
    ensure!(
        format!("{error:#}").contains("sequence budget"),
        "negative control failed before its intended physical-identity assertion: {error:#}"
    );
    Ok(())
}

/// Separate functional comparison only. These are admission checks against
/// the pinned legacy control, not configuration lowering or GPU calibration.
fn validate_polynomial_comparison_spec(spec: &aisimulate_core::replay::ReplaySpec) -> Result<()> {
    use aisimulate_core::{
        engine::{Backend, PreemptionMode, TimingModelConfig, TransferTimingMode},
        replay::{ReplayEngineConfig, ReplayTopology, WorkerStage},
    };
    ensure!(
        matches!(&spec.topology, ReplayTopology::Disaggregated { prefill, decode, .. }
        if prefill.initial_workers == 2 && decode.initial_workers == 2),
        "polynomial comparison requires the legacy 2P2D topology"
    );
    ensure!(
        spec.adapters.placement.provider == "round_robin",
        "polynomial comparison must retain native round-robin placement, not Dynamo KV routing"
    );
    let engine: ReplayEngineConfig = serde_json::from_value(spec.engine.clone())?;
    for stage in [WorkerStage::Prefill, WorkerStage::Decode] {
        let role = engine.role(stage);
        let rank = role.rank;
        ensure!(
            role.dp_size == 1
                && role.tensor_parallel_size == 1
                && role.prefill_context_parallel_size == 1,
            "old polynomial factory has DP1/TP1 accounting, not calibrated TP8 GPUs"
        );
        ensure!(
            rank.timing_model == TimingModelConfig::Polynomial && rank.backend == Backend::Vllm,
            "separate comparison requires the owning built-in polynomial timing on vLLM"
        );
        ensure!(
            rank.block_size == 8
                && rank.num_gpu_blocks == 135335
                && rank.max_model_len == Some(1048576)
                && rank.max_num_batched_tokens == 4096
                && rank.max_num_seqs == 32
                && rank.enable_prefix_caching
                && rank.enable_chunked_prefill
                && rank.async_scheduling == Some(false),
            "polynomial comparison differs from the explicit legacy serving limits"
        );
        ensure!(
            rank.kv_transfer_bytes_per_token == Some(44928)
                && rank.kv_transfer_bandwidth == Some(50.0)
                && rank.kv_transfer_timing_mode == TransferTimingMode::FullPrompt,
            "polynomial comparison differs from the explicit legacy transfer assumption"
        );
        ensure!(
            rank.speedup_ratio == 1.0
                && rank.decode_speedup_ratio == 1.0
                && rank.prefill_schedule_interval == 1
                && rank.prefill_decode_interval == 0
                && rank.preemption_mode == PreemptionMode::Lifo,
            "polynomial comparison differs from pinned legacy scheduling defaults"
        );
        ensure!(
            rank.kv_cache_groups.is_empty()
                && rank.kv_cache_capacity_bytes.is_none()
                && rank.native_host_offload.is_none()
                && rank.g3_offload.is_none()
                && rank.kvcr.is_none()
                && rank.state_cache.is_none()
                && rank.prefix_match_unit.is_none()
                && rank.aic_nextn.is_none()
                && !rank.vision,
            "polynomial comparison must not add grouped/state/offload/speculative/vision physics"
        );
    }
    Ok(())
}

fn polynomial_comparison_prepared() -> Result<aisimulate_core::replay::PreparedReplay> {
    let path = PathBuf::from(std::env::var_os("MVP_COMPARISON_AISIM_CONFIG").context(
        "MVP_COMPARISON_AISIM_CONFIG must name the separate polynomial comparison YAML",
    )?);
    let bytes = aiperf_fsio::artifact_file::read_no_follow_bounded(&path, 1 << 20)?;
    let config: Value = serde_yaml::from_slice(&bytes)?;
    let prepared = aisimulate_core::replay::PreparedReplay::from_prediction_config(
        &serde_json::to_string(&config)?,
    )?;
    validate_polynomial_comparison_spec(&prepared.spec)?;
    Ok(prepared)
}

#[test]
#[ignore = "requires MVP_COMPARISON_AISIM_CONFIG; bounded polynomial/RR comparison admission"]
fn polynomial_comparison_rejects_nonmatching_identity() -> Result<()> {
    let prepared = polynomial_comparison_prepared()?;
    let mut changed = prepared.spec.clone();
    let mut engine: aisimulate_core::replay::ReplayEngineConfig =
        serde_json::from_value(changed.engine.clone())?;
    engine
        .prefill
        .as_mut()
        .context("prefill role")?
        .rank
        .timing_model = aisimulate_core::engine::TimingModelConfig::Fixed {
        prefill_ms: 1.0,
        decode_ms: 2.0,
    };
    changed.engine = serde_json::to_value(engine)?;
    ensure!(
        validate_polynomial_comparison_spec(&changed).is_err(),
        "fixed timing must not qualify as polynomial"
    );
    let mut changed = prepared.spec;
    let mut engine: aisimulate_core::replay::ReplayEngineConfig =
        serde_json::from_value(changed.engine.clone())?;
    engine
        .decode
        .as_mut()
        .context("decode role")?
        .rank
        .num_gpu_blocks += 1;
    changed.engine = serde_json::to_value(engine)?;
    ensure!(
        validate_polynomial_comparison_spec(&changed).is_err(),
        "changed capacity must not qualify as legacy-matched"
    );
    Ok(())
}

#[test]
#[ignore = "requires MVP_COMPARISON_AISIM_CONFIG; four-request polynomial/RR launch, no full corpus"]
fn polynomial_comparison_bounded_shared_workload() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let mut run = aiperf_protocol::runner_v2::decode_run_request_v2(
        run_fixture::branch_request(directory.path()).as_bytes(),
    )?
    .run;
    run.cfg.transport = Some(serde_json::from_value(
        json!({"type":"aisimulate","mode":"offline"}),
    )?);
    run.cfg
        .endpoint
        .as_mut()
        .context("fixture endpoint")?
        .use_server_token_count = false;
    let prepared = polynomial_comparison_prepared()?;
    let (backend, typed) = host_typed_using(&run, true, || {
        Replayer::new(prepared.spec, prepared.factory).context("native polynomial/RR composition")
    })?;
    ensure!(
        backend.per_request.len() == 4
            && backend.per_request.iter().all(|row| {
                row.terminal_status == aisimulate_core::replay::ReplayTerminalStatus::Completed
                    && row.output_length == 2
            }),
        "bounded comparison must complete all four shared requests with their two output tokens"
    );
    let _ = reconcile_hosted::reconcile(&backend, &typed)?;
    Ok(())
}

#[test]
#[ignore = "separate full shared AgentX polynomial/RR functional comparison; requires MVP_COMPARISON_AISIM_CONFIG and new MVP_ARTIFACT_DIR"]
fn shared_agentx_legacy_polynomial_round_robin_comparison() -> Result<()> {
    let artifact_path = mvp_artifact_path()?;
    let mut run = full_mvp_run(&artifact_path)?;
    // Verify the accepted shared workload first; only this explicitly labelled
    // comparison selects the legacy client-count policy afterward.
    assert_mvp_config(&run)?;
    run.cfg
        .endpoint
        .as_mut()
        .context("scenario endpoint")?
        .use_server_token_count = false;
    let prepared = polynomial_comparison_prepared()?;
    let physical_spec = prepared.spec.clone();
    let harness = executable_distribution_id()?;
    let metadata = report_run_metadata(&run, &harness)?;
    let artifact_dir = create_new_artifact_dir(&artifact_path)?;
    let (backend, typed) = host_typed_using(&run, true, || {
        Replayer::new(prepared.spec, prepared.factory).context("native polynomial/RR comparison")
    })?;
    let provenance = json!({
        "scope":"separate legacy-matched polynomial/RR functional comparison; not calibrated acceptance or parity",
        "comparison":"shared-agentx-polynomial-round-robin", "dynamo_parity":false,
        "gpu_calibration":false, "physical_gpu_count_claimed":false,
        "physical_configuration_source":"existing AISim canonical config/compiler/PreparedReplay",
        "physical_spec":physical_spec, "executable_distribution_id":harness,
        "input_representation":"trace_hashes", "legacy_input_representation":"text",
        "projection_difference_explicit":true, "text_tokenizer_used":false,
        "use_server_token_count":false, "prepared_roots":typed.prepared_roots,
        "virtual_end_ns":typed.virtual_end_ns,
        "limitations":["same polynomial source does not prove all-engine or workload parity",
            "explicit copied capacity and transfer assumptions are not independently GPU-calibrated",
            "legacy text preparation and shared TraceHashes preparation remain different",
            "client-count policy is a comparison exception; calibrated full proof keeps server counts"]
    });
    write_raw_backend_with_provenance(&artifact_dir, &backend, &typed, &provenance)?;
    let finalized = finalize_hosted_run_report(
        &run,
        &typed,
        metadata,
        ReportPairRunFacts::new(),
        &artifact_dir,
    )?;
    let reconciliation = reconcile_hosted::reconcile(&backend, &typed)?;
    std::fs::write(
        artifact_dir.join("request_reconciliation.json"),
        serde_json::to_vec_pretty(&reconciliation)?,
    )?;
    ensure!(
        typed.prepared_roots == EXPECTED_PREPARED_ROOTS && !backend.per_request.is_empty(),
        "comparison must retain the complete corpus and actual shared request population"
    );
    ensure!(
        finalized.native_report.run.facts().dynamo.is_none(),
        "no Dynamo parity facts may be fabricated"
    );
    ensure!(
        artifact_dir.join(SUMMARY_ARTIFACT).is_file(),
        "normal GenAI-Perf summary must be retained"
    );
    Ok(())
}

/// Option-matched speed control over AISim's owning WEKA importer and native
/// agentic driver, using the same physical factory and real KV-router as MVP.
/// This is not an oracle or a replacement workload implementation.
#[test]
#[ignore = "requires matched snapshot artifact, complete WEKA corpus and canonical models"]
fn canonical_weka_agentx_with_matched_options() -> Result<()> {
    use aisimulate_core::replay::loadgen::{
        AgenticProfileOptions, AgenticSnapshotOptions, AgenticSnapshotSelection,
        AgenticWarmupPolicy, WekaClient, WekaImportOptions, WorkloadDriver,
        load_weka_agentic_graph_with_options,
    };
    let started = std::time::Instant::now();
    let artifact_path = mvp_artifact_path()?;
    let (prefill, decode) = full_scenario_roles()?;
    let corpus =
        PathBuf::from(std::env::var_os("MVP_CORPUS_PATH").context("MVP_CORPUS_PATH required")?);
    let cuts_file = PathBuf::from(
        std::env::var_os("MVP_INITIAL_SELECTIONS").context("MVP_INITIAL_SELECTIONS required")?,
    );
    let cuts: Value = serde_json::from_slice(&std::fs::read(cuts_file)?)?;
    let tree_idle_cap_seconds = 300.0; // Named agentx preset tree-idle cap.
    let imported_at = std::time::Instant::now();
    let (graph, _timestamp_basis) = load_weka_agentic_graph_with_options(
        &corpus,
        None,
        WekaImportOptions {
            client: WekaClient::Agentx,
            ..Default::default()
        },
    )?;
    ensure!(
        graph.play_count() == EXPECTED_PREPARED_ROOTS,
        "official import must retain every source root"
    );
    let selections = cuts["selections"]
        .as_array()
        .context("source cuts")?
        .iter()
        .map(|cut| {
            let trace = cut["source_trace_id"].as_str().context("source trace")?;
            let suffix = format!(":play:{trace}");
            let mut matches = graph
                .nodes()
                .iter()
                .map(|node| node.play_id())
                .filter(|id| id.ends_with(&suffix));
            let play = matches.next().context("official source play missing")?;
            ensure!(
                matches.all(|id| id == play),
                "ambiguous official source play"
            );
            Ok(AgenticSnapshotSelection {
                source_play_id: play.into(),
                t_star_ms: cut["relative_t_star_ms"]
                    .as_f64()
                    .context("relative source cut")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        selections.len() == CONCURRENCY as usize,
        "one exact cut per lane required"
    );
    let prepared = graph.into_prepared_snapshots(
        CONCURRENCY as usize,
        AgenticSnapshotOptions {
            seed: SEED,
            initial_selections: Some(selections),
            corpus_cursor: Some(
                cuts["sampler_draws"]
                    .as_u64()
                    .context("initial sampler cursor")?,
            ),
        },
    )?;
    let mut driver = WorkloadDriver::new_agentic_warmup_with_policy(
        prepared,
        calibrated::BLOCK_SIZE,
        true,
        1.0,
        AgenticWarmupPolicy::LiveTrajectoryV1,
        SYSTEM_IDLE_SECONDS * 1000.0,
    )?;
    driver.enable_agentic_profile(AgenticProfileOptions {
        duration_seconds: PROFILE_SECONDS,
        response_grace_seconds: GRACE_PERIOD_SECONDS,
        cancel_drain_seconds: 10.0,
        tree_idle_cap_seconds,
        global_idle_cap_seconds: SYSTEM_IDLE_SECONDS,
    })?;
    let prepared_at = std::time::Instant::now();
    let composition = KvReplayComposition::disaggregated(
        prefill.router.clone(),
        decode.router.clone(),
        calibrated::PREFILL_WORKERS,
        calibrated::DECODE_WORKERS,
        None,
        None,
        None,
    );
    let report = Replayer::with_composition(
        calibrated::spec(&prefill, &decode)?,
        prefill.factory.clone(),
        composition,
    )?
    .with_runtime_input(ReplayRuntimeInput::Workload(driver))
    .with_capture_options(ReplayCaptureOptions {
        capture_per_request: true,
        determinism: ReplayDeterminism::CanonicalV1,
        ..Default::default()
    })
    .run()?;
    let finished_at = std::time::Instant::now();
    let dir = create_new_artifact_dir(&artifact_path)?;
    #[derive(serde::Serialize)]
    struct Evidence<'a> {
        backend_report: &'a ReplayReport,
        backend_requests: &'a [aisimulate_core::replay::PerRequestRecord],
        options: Value,
        host_seconds: Value,
    }
    let evidence = Evidence {
        backend_report: &report,
        backend_requests: &report.per_request,
        options: json!({"client":"agentx","snapshot":"exact shared source cuts",
            "warmup":"live_trajectory_v1","lanes":CONCURRENCY,"duration_seconds":PROFILE_SECONDS,
            "response_grace_seconds":GRACE_PERIOD_SECONDS,"tree_idle_cap_seconds":tree_idle_cap_seconds,
            "global_idle_cap_seconds":SYSTEM_IDLE_SECONDS,"placement":"same real Dynamo KV composition",
            "model":"same owner-prepared physical factory from full_scenario_roles",
            "capture_per_request":true,"determinism":"canonical_v1",
            "limitations":["native WEKA hash rekeying differs from shared first_turn_prefix representation",
                "native driver and shared workload are independently owned semantics; matching options is not parity"]}),
        host_seconds: json!({"model_construction":imported_at.duration_since(started).as_secs_f64(),
            "source_import_and_prepare":prepared_at.duration_since(imported_at).as_secs_f64(),
            "replay":finished_at.duration_since(prepared_at).as_secs_f64(),
            "before_serialization":finished_at.duration_since(started).as_secs_f64()}),
    };
    serde_json::to_writer(
        std::fs::File::create(dir.join("matched_canonical_weka.json"))?,
        &evidence,
    )?;
    ensure!(
        !report.per_request.is_empty(),
        "comparison must execute requests"
    );
    println!("MATCHED_CANONICAL_WEKA {}", evidence.host_seconds);
    Ok(())
}

#[test]
#[ignore = "requires MVP_AISIM_CONFIG and a new MVP_ARTIFACT_DIR; runs the full 3600s trace-hash scenario over the real Dynamo KV composition"]
fn inferencex_agentx_mvp_full_scenario_runs_on_native_dynamo_kv() -> Result<()> {
    // Static preflight first: the retained directory is created only after the
    // scenario frame, the system assets, and the canonical roles
    // are proven, so a failing check cannot leave a half-built result behind.
    let artifact_path = mvp_artifact_path()?;
    let run = full_mvp_run(&artifact_path).context("exact InferenceX AgentX MVP run")?;
    assert_mvp_config(&run)?;
    let harness = executable_distribution_id()?;

    let (prefill, decode) = full_scenario_roles()?;
    for role in [&prefill, &decode] {
        ensure!(
            role.rank.max_model_len == Some(usize::try_from(MODEL_MAX_CONTEXT_TOKENS)?)
                && role.model_config.max_model_len == usize::try_from(MODEL_MAX_CONTEXT_TOKENS)?,
            "canonical role context capacity disagrees with the consumed 1M model config"
        );
        ensure!(
            prefill.model_config.sha256 == decode.model_config.sha256,
            "canonical roles must consume the same pinned model config bytes"
        );
    }
    let metadata = report_run_metadata(&run, &harness)?;

    let artifact_dir = create_new_artifact_dir(&artifact_path)?;
    eprintln!("MVP stage: exact input and calibrated physical roles verified");

    let (backend, typed) = host_typed(&run, &prefill, &decode, true)?;
    // Retain complete physical evidence before any acceptance assertion can
    // fail, so a population or metric defect never erases the run's output.
    write_raw_backend_artifact(&artifact_dir, &backend, &typed, &prefill, &decode, &harness)?;

    ensure!(
        metadata.transport == "aisimulate",
        "metadata transport must be aisimulate"
    );
    ensure!(
        metadata.workload == "graph",
        "metadata workload must be graph"
    );
    // Ordinary scheduled/graph runs carry no backend parity facts. The raw
    // backend evidence is the separate, explicitly non-parity artifact above.
    let facts = ReportPairRunFacts::new();
    eprintln!("MVP stage: retaining normal artifacts before acceptance assertions");
    let finalized = finalize_hosted_run_report(&run, &typed, metadata, facts, &artifact_dir)
        .context("finalizing the normal hosted run report")?;
    let reconciliation = reconcile_hosted::reconcile(&backend, &typed)
        .context("reconciling every backend occurrence against native records")?;
    std::fs::write(
        artifact_dir.join("request_reconciliation.json"),
        serde_json::to_vec_pretty(&reconciliation)?,
    )?;
    eprintln!("MVP stage: request reconciliation passed");

    assert_full_scenario(&run, &typed, &finalized, &artifact_dir)?;
    eprintln!("MVP stage: full scenario assertions passed");
    Ok(())
}

#[test]
fn inferencex_agentx_mvp_run_is_the_exact_user_scenario() -> Result<()> {
    let directory = tempfile::tempdir().context("artifact directory")?;
    let run = full_mvp_run(directory.path()).context("exact InferenceX AgentX MVP run")?;
    assert_mvp_config(&run)?;
    Ok(())
}

/// Physical rejection must remain visible, not disappear from metric evidence.
/// This bounded fixture exercises prompt-at-model-limit rejection without
/// changing the full scenario's canonical model capacity.
#[test]
#[ignore = "requires MVP_AISIM_CONFIG; bounded calibrated rejection accounting control"]
fn calibrated_hosted_rejections_reconcile_every_request() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let run = aiperf_protocol::runner_v2::decode_run_request_v2(
        run_fixture::branch_request(directory.path()).as_bytes(),
    )?
    .run;
    let (mut prefill, mut decode) = full_scenario_roles()?;
    prefill.rank.max_model_len = Some(16);
    decode.rank.max_model_len = Some(16);
    let (backend, typed) = host_typed(&run, &prefill, &decode, true)?;
    ensure!(
        !backend.per_request.is_empty(),
        "rejection control is empty"
    );
    ensure!(
        backend
            .per_request
            .iter()
            .all(|row| row.terminal_status
                == aisimulate_core::replay::ReplayTerminalStatus::Rejected),
        "every control prompt must hit the explicit context limit"
    );
    let reconciled = reconcile_hosted::reconcile(&backend, &typed)?;
    ensure!(
        reconciled.native_records == backend.per_request.len(),
        "rejected native records were lost"
    );
    Ok(())
}

/// Small boundary control, never acceptance for the full corpus or duration.
/// Exercise the exact calibrated host, typed reconciliation, and artifact
/// finalizer before spending full-corpus resources. Deliberately use the
/// existing branch fixture's trace hashes, with no scenario designation.
#[test]
#[ignore = "requires MVP_AISIM_CONFIG; four-request calibrated trace-hash artifact control, not full MVP acceptance"]
fn calibrated_hosted_artifacts_reconcile_every_request() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let mut run = aiperf_protocol::runner_v2::decode_run_request_v2(
        run_fixture::branch_request(directory.path()).as_bytes(),
    )?
    .run;
    run.cfg.transport = Some(serde_json::from_value(json!({
        "type": "aisimulate", "mode": "offline"
    }))?);
    run.cfg.models.as_mut().context("fixture models")?.items[0].name = MODEL.to_owned();
    let export = run.cfg.export.as_mut().context("fixture export")?;
    export.genai_perf.enabled = true;
    export.genai_perf.json = true;
    let (prefill, decode) = full_scenario_roles()?;
    let (mut backend, typed) = host_typed(&run, &prefill, &decode, true)?;
    let (complete_backend, complete_typed) = host_typed(&run, &prefill, &decode, false)?;
    let mut compact_physical = serde_json::to_value(&backend)?;
    let mut complete_physical = serde_json::to_value(&complete_backend)?;
    // Host execution cost is real elapsed time (including cold FPM work), not
    // simulated physics. Keep it in raw outputs; exclude exactly that clock
    // and its two derived rates from this semantic equivalence assertion.
    for report in [&mut compact_physical, &mut complete_physical] {
        let fields = report.as_object_mut().context("backend report object")?;
        for field in [
            "wall_time_ms",
            "processed_tokens_per_s",
            "processed_output_tokens_per_s",
        ] {
            ensure!(
                fields.remove(field).is_some(),
                "host cost field {field} missing"
            );
        }
    }
    ensure!(
        compact_physical == complete_physical,
        "compact capture changed physical replay facts: compact={} complete={}",
        compact_physical,
        complete_physical
    );
    ensure!(
        serde_json::to_value(&typed.phased)? == serde_json::to_value(&complete_typed.phased)?,
        "compact capture changed native phase results: compact={} complete={}",
        serde_json::to_string(&typed.phased)?,
        serde_json::to_string(&complete_typed.phased)?
    );
    let compact_records: Vec<_> = typed
        .records
        .iter()
        .flat_map(|phase| &phase.records)
        .collect();
    let complete_records: Vec<_> = complete_typed
        .records
        .iter()
        .flat_map(|phase| &phase.records)
        .collect();
    ensure!(
        compact_records == complete_records,
        "compact capture changed native record facts"
    );
    for row in &typed.dispatch.requests {
        ensure!(
            row["capture"]["mode"] == json!("compact"),
            "compact evidence must be explicitly labelled"
        );
        ensure!(
            row["request"].get("body").is_none()
                && row.get("wire").is_none()
                && row.get("input_token_ids").is_none(),
            "compact evidence retained a full context copy"
        );
        ensure!(
            row["capture"]["wire_blake3"]
                .as_str()
                .is_some_and(|value| value.len() == 64),
            "compact evidence must identify actual wire bytes"
        );
        ensure!(
            row["observations"]
                .as_array()
                .context("observations")?
                .iter()
                .all(|event| event["event"] != json!("output_token_ids")),
            "compact evidence retained per-token diagnostic JSON"
        );
    }
    let reconciliation = reconcile_hosted::reconcile(&backend, &typed)?;
    ensure!(
        reconciliation.backend_requests == 4,
        "branch control must retain all four requests"
    );
    ensure!(
        reconciliation.native_records == 4,
        "all four native records must reconcile"
    );
    let metadata = report_run_metadata(&run, &executable_distribution_id()?)?;
    let finalized = finalize_hosted_run_report(
        &run,
        &typed,
        metadata,
        ReportPairRunFacts::new(),
        directory.path(),
    )?;
    ensure!(
        finalized.native_report.run.facts().dynamo.is_none(),
        "control must not invent parity facts"
    );
    let report: Value = serde_json::from_slice(&std::fs::read(directory.path().join(
        aiperf_config::model::artifacts::report_artifact_name(run.cfg.artifacts.as_ref()),
    ))?)?;
    ensure!(
        report["schema_version"] == json!(NATIVE_REPORT_SCHEMA),
        "native artifact schema mismatch"
    );
    let summary: Value =
        serde_json::from_slice(&std::fs::read(directory.path().join(SUMMARY_ARTIFACT))?)?;
    ensure!(
        summary["schema_version"] == json!(SUMMARY_SCHEMA),
        "summary artifact schema mismatch"
    );
    write_raw_backend_artifact(
        directory.path(),
        &backend,
        &typed,
        &prefill,
        &decode,
        &executable_distribution_id()?,
    )?;
    let raw: Value =
        serde_json::from_slice(&std::fs::read(directory.path().join(RAW_BACKEND_ARTIFACT))?)?;
    ensure!(
        raw["backend_report"] == serde_json::to_value(&backend)?,
        "streaming backend artifact lost facts"
    );
    ensure!(
        raw["typed_phase_report"] == serde_json::to_value(&typed.phased)?,
        "streaming phase artifact lost facts"
    );
    ensure!(
        raw["backend_requests"] == serde_json::to_value(&backend.per_request)?,
        "streaming backend artifact lost per-request facts"
    );
    ensure!(
        raw["dispatch_requests"] == serde_json::to_value(&typed.dispatch.requests)?,
        "streaming artifact lost dispatch evidence"
    );
    for (captured, phase) in raw["native_records"]
        .as_array()
        .context("raw native phases")?
        .iter()
        .zip(&typed.records)
    {
        ensure!(
            captured["records"] == serde_json::to_value(&phase.records)?,
            "streaming native artifact lost request facts"
        );
    }
    ensure!(
        raw["native_records"]
            .as_array()
            .context("raw native phases")?
            .len()
            == typed.records.len(),
        "streaming native artifact lost a phase"
    );
    ensure!(
        raw["prefill"]["transfer"]["link"]["calibration_manifest_sha256"]
            .as_str()
            .is_some_and(|hash| hash.len() == 64),
        "full artifact lost calibration provenance"
    );
    // A validator that only counts rows would miss this corruption.
    backend.per_request[0].terminal_time_ms += 1.0;
    ensure!(
        reconcile_hosted::reconcile(&backend, &typed).is_err(),
        "reconciliation must reject timestamp drift"
    );
    Ok(())
}

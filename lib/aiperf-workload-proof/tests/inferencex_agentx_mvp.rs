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
//! * canonical H100 BF16 op-level role timing for prefill and decode;
//! * the real Dynamo KV Router composition produced by
//!   `KvReplayComposition::disaggregated`;
//! * the caller-owned `HostedTypedReport`, finalized into the normal native-v2
//!   report and GenAI-Perf v1.7 summary by
//!   `hosted_report::finalize_hosted_run_report`.
//!
//! Scenario invariants are sourced from
//! `aiperf_agentx::scenario::inferencex_agentx_mvp()`:
//!
//! * timing mode `agentic_replay`, streaming, `ignore_eos`
//! * first-turn-prefix cache bust
//! * 10-second global system-idle cap
//! * 1800-second profile duration (the full default, not the 900-second lower
//!   bound and not the 1200-second `agentx_fast` diagnostic)
//! * concurrency 32, seed 42
//! * trajectory-start ratios 0.0..=1.0 (not the frozen parity preset)
//! * no request or session caps, finite replay disabled
//! * no `unsafe_override`, all corpus sources retained
//! * explicit local Qwen3-8B tokenizer from `MVP_TOKENIZER_ROOT`
//!
//! The full E2E deliberately holds no Dynamo parity claim: the normal native
//! report is finalized with empty `ReportPairRunFacts`, and the raw native
//! backend evidence plus canonical role geometry are written separately as
//! `native_backend_report.json` with `dynamo_parity: false`. That file is
//! provenance, never substituted for typed parity facts.
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
//! * `MVP_TOKENIZER_ROOT`, `MVP_TOKENIZER_REVISION` — the local pinned
//!   Qwen3-8B tokenizer.
//! * `AISIM_SYSTEMS_ROOT` — the checked-in H100 system assets.
//! * `MVP_ARTIFACT_DIR` — a new directory. The test refuses to reuse an
//!   existing path and never uses a temporary directory, so the complete
//!   result survives.

#![cfg(feature = "calibrated-proof")]

use std::{
    fs::File,
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use aiperf_clock_runtime::cancellation::RunCancellation;
use aiperf_config::model::{
    BenchmarkRun,
    dataset::{CacheBust, CacheBustTarget, Dataset, PublicDataset, Sampling},
    phase::{Phase, PhaseCommon, PhaseKind as ConfigPhaseKind},
    tokenizer::Tokenizer,
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
use aisimulate_core::{
    engine::TimingModel,
    perfmodel::ForwardPassWorkerType,
    replay::{
        ReplayCaptureOptions, ReplayDeterminism, ReplayEngineFactory, ReplayReport,
        ReplayRuntimeInput, Replayer,
    },
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

/// `aiperf_agentx::scenario::AGENTX_SCENARIO`.
const AGENTX_SCENARIO: &str = "inferencex-agentx-mvp";
/// `aiperf_agentx::scenario::inferencex_agentx_mvp()` corpus alias.
const AGENTX_CORPUS: &str = "semianalysis_cc_traces_weka_062126";
/// The alias's pinned corpus revision, mirrored from the catalog entry.
const AGENTX_CORPUS_REVISION: &str = "23f152f6f0f9399a85901b89a6458def0ef16729";
/// Full scenario profile duration. Not 900 (the lower bound) and not 1200
/// (the `agentx_fast` diagnostic). Cross-checked against the scenario spec.
const PROFILE_SECONDS: f64 = 1800.0;
/// Scenario global system-idle cap. Cross-checked against the scenario spec.
const SYSTEM_IDLE_SECONDS: f64 = 10.0;
/// Scenario concurrency.
const CONCURRENCY: u32 = 32;
/// Scenario seed.
const SEED: u64 = 42;
/// Scenario default profiling grace period from the shared canonical plan.
const GRACE_PERIOD_SECONDS: f64 = 30.0;
/// Scenario trajectory-start window.
const TRAJECTORY_START_MIN_RATIO: f64 = 0.0;
const TRAJECTORY_START_MAX_RATIO: f64 = 1.0;
/// Canonical role model identity. Same model on every role and on the run.
const MODEL: &str = "Qwen/Qwen3-8B";
/// Qwen3-8B's model maximum context. Calibration metadata only: the full
/// corpus stays unfiltered, and context overflow is owned by the backend.
const MODEL_MAX_CONTEXT_TOKENS: u64 = 40960;
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
/// from `inferencex_agentx_mvp()`; only the documented concurrency and grace
/// period are authored here because the scenario default does not carry them.
pub fn full_mvp_run(directory: &Path) -> Result<BenchmarkRun> {
    let tokenizer_root = std::env::var_os("MVP_TOKENIZER_ROOT").context(
        "MVP_TOKENIZER_ROOT must identify the local pinned Qwen3-8B tokenizer; \
         no builtin or remote fallback is permitted",
    )?;
    let tokenizer_root = PathBuf::from(tokenizer_root);
    ensure!(
        tokenizer_root.is_dir(),
        "MVP_TOKENIZER_ROOT is not a directory: {}",
        tokenizer_root.display()
    );
    let tokenizer_revision = std::env::var("MVP_TOKENIZER_REVISION")
        .context("MVP_TOKENIZER_REVISION must pin the Qwen3-8B tokenizer revision")?;
    ensure!(
        !tokenizer_revision.trim().is_empty(),
        "MVP_TOKENIZER_REVISION must not be empty"
    );

    let scenario = aiperf_agentx::scenario::inferencex_agentx_mvp();
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
    let trajectory_min = scenario
        .default_trajectory_start_min_ratio
        .context("scenario must declare the trajectory-start minimum")?;
    let trajectory_max = scenario
        .default_trajectory_start_max_ratio
        .context("scenario must declare the trajectory-start maximum")?;
    ensure!(
        system_idle_seconds == SYSTEM_IDLE_SECONDS,
        "scenario system-idle cap changed to {system_idle_seconds}"
    );
    ensure!(
        trajectory_min == TRAJECTORY_START_MIN_RATIO
            && trajectory_max == TRAJECTORY_START_MAX_RATIO,
        "scenario trajectory window changed to {trajectory_min}..={trajectory_max}"
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

    // Explicit local tokenizer; no builtin and no remote acquisition.
    run.cfg.tokenizer = Some(Tokenizer {
        name: Some(tokenizer_root.to_string_lossy().into_owned()),
        revision: tokenizer_revision,
        trust_remote_code: false,
        apply_chat_template: false,
        server_url: None,
        per_model: Default::default(),
    });

    let endpoint = run
        .cfg
        .endpoint
        .as_mut()
        .context("base fixture must carry the endpoint policy")?;
    endpoint.streaming = true;
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
        entries: None,
        entries_explicit: None,
        random_seed: Some(SEED),
        prompts: None,
        synthesis: Some(json!({
            "speedup_ratio": 1.0,
            "prefix_len_multiplier": 1.0,
            "prefix_root_multiplier": 1,
            "prompt_len_multiplier": 1.0,
            "output_len_multiplier": 1.0,
            "cache_bust_target": "first_turn_prefix",
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
            // No request or session caps: the 1800-second profile is the only
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
            warmup_requests_per_lane: None,
            agentic_warmup_grace_period: None,
            failed_request_threshold: None,
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
    let scenario = aiperf_agentx::scenario::inferencex_agentx_mvp();
    let profile_seconds = scenario
        .default_benchmark_duration_seconds
        .context("scenario must declare the full profile duration")?
        as f64;
    let system_idle_seconds = scenario
        .system_idle_gap_cap_seconds
        .context("scenario must declare the global system-idle cap")?;
    let trajectory_min = scenario
        .default_trajectory_start_min_ratio
        .context("scenario must declare the trajectory-start minimum")?;
    let trajectory_max = scenario
        .default_trajectory_start_max_ratio
        .context("scenario must declare the trajectory-start maximum")?;

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
        public.entries.is_none() && public.entries_explicit.is_none(),
        "no entry cap may reduce the corpus"
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
    let tokenizer = run
        .cfg
        .tokenizer
        .as_ref()
        .context("MVP run must carry the explicit tokenizer policy")?;
    ensure!(
        tokenizer.name.as_deref() != Some("builtin"),
        "the MVP run requires the explicit local tokenizer"
    );
    ensure!(
        tokenizer
            .name
            .as_deref()
            .is_some_and(|name| !name.trim().is_empty()),
        "the MVP tokenizer must name the local acquisition"
    );
    ensure!(
        !tokenizer.revision.trim().is_empty(),
        "the MVP tokenizer revision must be pinned"
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
) -> Result<(ReplayReport, HostedTypedReport)> {
    let (participant, handle) = native_hosted(
        run,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            calibrated::IDENTITY_SEED,
        )),
    )
    .context("shared native lowering hosts the production plans")?;
    let prefill_timing: Arc<dyn TimingModel> = prefill.timing.clone();
    let decode_timing: Arc<dyn TimingModel> = decode.timing.clone();
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
        calibrated::spec(prefill, decode)?,
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
    .context("native Replayer event loop hosts the full AIPerf scenario")?;
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
fn report_run_metadata(run: &BenchmarkRun) -> Result<ReportRunMetadata> {
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
        executable_distribution_id()?,
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

/// Preserve the raw native backend report and canonical role provenance as a
/// separate, explicitly non-parity artifact. The normal report's typed pair
/// facts stay empty, so nothing here can be mistaken for `ReportDynamoRunInfo`.
fn write_raw_backend_artifact(
    artifact_dir: &Path,
    backend: &ReplayReport,
    typed: &HostedTypedReport,
    prefill: &calibrated::RoleSetup,
    decode: &calibrated::RoleSetup,
) -> Result<()> {
    let artifact = json!({
        "scope": "raw native AISim Replayer backend and canonical role provenance; \
                  this is not Dynamo parity evidence and is never substituted for \
                  the typed ReportDynamoRunInfo facts",
        "dynamo_parity": false,
        "dynamo_typed_facts_present": false,
        "dynamo_clock": Value::Null,
        "transport": "aisimulate",
        "workload": "graph",
        "model_max_context_tokens": MODEL_MAX_CONTEXT_TOKENS,
        "model_max_context_note": "calibration metadata only; the full pinned corpus is never filtered or truncated, and context overflow is classified by the backend",
        "limitations": [
            "no typed Dynamo topology, clock, capacity, or parity facts are supplied; this raw artifact does not claim full Dynamo parity",
            "the full pinned corpus remains unfiltered; long-context overflow is backend-owned and is judged by the normal submission verdict",
        ],
        "backend_report": backend,
        "typed_phase_report": &typed.phased,
        "prepared_roots": typed.prepared_roots,
        "virtual_end_ns": typed.virtual_end_ns,
        "prefill": {
            "timing": &prefill.diagnostics,
            "native_kv_geometry": &prefill.geometry,
            "replay_rank": &prefill.rank,
        },
        "decode": {
            "timing": &decode.diagnostics,
            "native_kv_geometry": &decode.geometry,
            "replay_rank": &decode.rank,
        },
    });
    let path = artifact_dir.join(RAW_BACKEND_ARTIFACT);
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&artifact).context("raw native backend artifact")?,
    )
    .with_context(|| format!("writing raw native backend artifact {}", path.display()))?;
    Ok(())
}

/// Assert the complete full-scenario evidence: exact corpus population, the
/// actual 1800-second profile span, branch spawn/join counters from the typed
/// phase report, metric coverage from the live accumulator, and the normal
/// native-v2/schema-1.7 artifacts.
fn assert_full_scenario(
    run: &BenchmarkRun,
    typed: &HostedTypedReport,
    finalized: &FinalizedHostedReport,
    artifact_dir: &Path,
) -> Result<()> {
    let scenario = aiperf_agentx::scenario::inferencex_agentx_mvp();
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
    // Long-context rejections are part of the full Qwen scenario and are
    // classified by the backend. The scenario's own submission verdict (the
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
    let invalid_reasons = summary["metadata"]["submission_invalid_reasons"]
        .as_array()
        .context("summary must carry submission_invalid_reasons")?;
    ensure!(
        invalid_reasons.is_empty(),
        "valid submission must carry no invalid reasons: {invalid_reasons:?}"
    );

    Ok(())
}

/// Require a fresh artifact directory from `MVP_ARTIFACT_DIR`. The full result
/// is retained for the operator; no temporary directory is created or dropped.
fn mvp_artifact_dir() -> Result<PathBuf> {
    let raw = std::env::var_os("MVP_ARTIFACT_DIR").context(
        "MVP_ARTIFACT_DIR must name a new, empty directory for the complete \
         full-scenario artifacts; the full E2E never uses a temporary directory \
         and refuses to overwrite an existing one",
    )?;
    let directory = PathBuf::from(raw);
    match std::fs::create_dir(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => bail!(
            "MVP_ARTIFACT_DIR already exists; refusing to overwrite full-scenario artifacts: {}",
            directory.display()
        ),
        Err(error) => {
            return Err(anyhow!(
                "creating MVP_ARTIFACT_DIR {}: {error}",
                directory.display()
            ));
        }
    }
    Ok(directory)
}

#[test]
#[ignore = "requires MVP_TOKENIZER_ROOT, MVP_TOKENIZER_REVISION, AISIM_SYSTEMS_ROOT and a new MVP_ARTIFACT_DIR; runs the full 1800s scenario over the real Dynamo KV composition"]
fn inferencex_agentx_mvp_full_scenario_runs_on_native_dynamo_kv() -> Result<()> {
    let artifact_dir = mvp_artifact_dir()?;
    let run = full_mvp_run(&artifact_dir).context("exact InferenceX AgentX MVP run")?;
    assert_mvp_config(&run)?;

    let systems = calibrated::systems_root()?;
    let prefill = calibrated::canonical_role(
        MODEL,
        &systems,
        ForwardPassWorkerType::Prefill,
        usize::try_from(CONCURRENCY)?,
    )
    .context("prefill canonical op-level timing and native geometry")?;
    let decode = calibrated::canonical_role(
        MODEL,
        &systems,
        ForwardPassWorkerType::Decode,
        usize::try_from(CONCURRENCY)?,
    )
    .context("decode canonical op-level timing and native geometry")?;

    let (backend, typed) = host_typed(&run, &prefill, &decode)?;
    write_raw_backend_artifact(&artifact_dir, &backend, &typed, &prefill, &decode)?;

    let metadata = report_run_metadata(&run)?;
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
    let finalized = finalize_hosted_run_report(&run, &typed, metadata, facts, &artifact_dir)
        .context("finalizing the normal hosted run report")?;

    assert_full_scenario(&run, &typed, &finalized, &artifact_dir)?;
    Ok(())
}

#[test]
#[ignore = "requires MVP_TOKENIZER_ROOT and MVP_TOKENIZER_REVISION; no builtin tokenizer fallback"]
fn inferencex_agentx_mvp_run_is_the_exact_user_scenario() -> Result<()> {
    let directory = tempfile::tempdir().context("artifact directory")?;
    let run = full_mvp_run(directory.path()).context("exact InferenceX AgentX MVP run")?;
    assert_mvp_config(&run)?;
    Ok(())
}

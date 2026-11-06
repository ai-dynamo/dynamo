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
//! * explicit local Qwen3-8B tokenizer from `MVP_TOKENIZER_ROOT`, verified
//!   against the exact pinned acquisition at
//!   `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` by byte length
//!   and SHA-256 for `config.json`, `tokenizer.json`, and
//!   `tokenizer_config.json`
//!
//! The full E2E deliberately holds no Dynamo parity claim: the normal native
//! report is finalized with empty `ReportPairRunFacts`, and the raw native
//! backend evidence plus canonical role geometry are written separately as
//! `native_backend_report.json` with `dynamo_parity: false`. That file is
//! provenance, never substituted for typed parity facts, and it records its own
//! designation as not a frozen parity result together with the exact tokenizer
//! file hashes and the runner executable identity that produced it.
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
//!   Qwen3-8B tokenizer. The revision must be exactly
//!   `b968826d9c46dd6066d109eabc6255188de91218`, and the three files must
//!   match the byte lengths and SHA-256 values recorded in `TOKENIZER_PINS`.
//!   Files are opened without following links, read under a 12 MiB bound, and
//!   rejected when they change between inspection and open.
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

#[path = "support/reconcile_hosted.rs"]
mod reconcile_hosted;

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

/// Exact local tokenizer acquisition: `Qwen/Qwen3-8B` at this revision.
/// `MVP_TOKENIZER_REVISION` must spell it, and [`TOKENIZER_PINS`] must match
/// byte-for-byte, so identity never rests on a directory name alone.
const TOKENIZER_REVISION: &str = "b968826d9c46dd6066d109eabc6255188de91218";
/// Bound applied to every pinned tokenizer file. `tokenizer.json` is the
/// largest pinned file at 11,422,654 bytes and stays below this.
const MAX_TOKENIZER_FILE_BYTES: u64 = 12 * 1024 * 1024;

/// One pinned tokenizer file: name, exact byte length, and SHA-256.
struct TokenizerPin {
    name: &'static str,
    bytes: u64,
    sha256: &'static str,
}

/// The complete pinned Qwen3-8B tokenizer file set at [`TOKENIZER_REVISION`].
const TOKENIZER_PINS: [TokenizerPin; 3] = [
    TokenizerPin {
        name: "config.json",
        bytes: 728,
        sha256: "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    },
    TokenizerPin {
        name: "tokenizer.json",
        bytes: 11_422_654,
        sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    TokenizerPin {
        name: "tokenizer_config.json",
        bytes: 9_732,
        sha256: "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    },
];

/// Verified tokenizer identity, recorded in the raw backend artifact.
#[derive(serde::Serialize)]
struct TokenizerEvidence {
    repository: &'static str,
    revision: String,
    root: String,
    model_type: String,
    max_position_embeddings: u64,
    files: Vec<TokenizerFileEvidence>,
}

/// One verified tokenizer file as recorded in the raw backend artifact.
#[derive(serde::Serialize)]
struct TokenizerFileEvidence {
    name: &'static str,
    bytes: u64,
    sha256: String,
}

/// Build the exact InferenceX AgentX MVP run from the strict native fixture.
///
/// The base frame is the shared `run_fixture::request`, so every non-dataset
/// section keeps its fully materialized native representation. Scenario-owned
/// dataset, phase, transport, export, and replay policy members are replaced
/// from `inferencex_agentx_mvp()`; only the documented concurrency and grace
/// period are authored here because the scenario default does not carry them.
///
/// Also returns the verified tokenizer provenance. This call writes nothing
/// into `directory`, so it runs as a preflight before the artifact directory
/// is created.
fn full_mvp_run(directory: &Path) -> Result<(BenchmarkRun, TokenizerEvidence)> {
    let tokenizer_root = std::env::var_os("MVP_TOKENIZER_ROOT").context(
        "MVP_TOKENIZER_ROOT must identify the local pinned Qwen3-8B tokenizer; \
         no builtin or remote fallback is permitted",
    )?;
    let tokenizer = tokenizer_evidence(&PathBuf::from(tokenizer_root))?;

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
        name: Some(tokenizer.root.clone()),
        revision: tokenizer.revision.clone(),
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
    Ok((decoded.run, tokenizer))
}

/// Verify the local acquisition is exactly the pinned Qwen3-8B tokenizer and
/// return the provenance the raw artifact records. The exact file hashes, not
/// the directory name, establish identity.
fn tokenizer_evidence(root: &Path) -> Result<TokenizerEvidence> {
    let revision = std::env::var("MVP_TOKENIZER_REVISION")
        .context("MVP_TOKENIZER_REVISION must pin the Qwen3-8B tokenizer revision")?;
    ensure!(
        revision == TOKENIZER_REVISION,
        "MVP_TOKENIZER_REVISION is {revision}, expected exactly {TOKENIZER_REVISION}"
    );
    ensure!(
        root.is_dir(),
        "MVP_TOKENIZER_ROOT is not a directory: {}",
        root.display()
    );

    let mut files = Vec::with_capacity(TOKENIZER_PINS.len());
    let mut config = None;
    for pin in TOKENIZER_PINS {
        let bytes = read_pinned_file(root, &pin)?;
        let digest = sha256_hex(&bytes);
        ensure!(
            digest == pin.sha256,
            "pinned tokenizer file {} has SHA-256 {digest}, expected {}",
            pin.name,
            pin.sha256
        );
        if pin.name == "config.json" {
            config = Some(bytes);
        }
        files.push(TokenizerFileEvidence {
            name: pin.name,
            bytes: pin.bytes,
            sha256: digest,
        });
    }

    let config: Value = serde_json::from_slice(
        config
            .as_deref()
            .context("config.json must be part of the pinned file set")?,
    )
    .context("pinned config.json must parse as JSON")?;
    let model_type = config["model_type"]
        .as_str()
        .context("pinned config.json must declare model_type")?;
    ensure!(
        model_type == "qwen3",
        "pinned tokenizer must be Qwen3, got model_type {model_type}"
    );
    // The pinned `config.json` carries no model name; the verified file hashes
    // above are the identity, so no name is required or invented here.
    let max_position = config["max_position_embeddings"]
        .as_u64()
        .context("pinned config.json must declare max_position_embeddings")?;
    ensure!(
        max_position == MODEL_MAX_CONTEXT_TOKENS,
        "pinned Qwen3-8B max_position_embeddings is {max_position}, \
         expected {MODEL_MAX_CONTEXT_TOKENS}"
    );

    Ok(TokenizerEvidence {
        repository: MODEL,
        revision,
        root: root.to_string_lossy().into_owned(),
        model_type: model_type.to_owned(),
        max_position_embeddings: max_position,
        files,
    })
}

/// Read one pinned tokenizer file without following links, under the exact byte
/// bound, and prove the opened handle is the file that was inspected.
fn read_pinned_file(root: &Path, pin: &TokenizerPin) -> Result<Vec<u8>> {
    let path = root.join(pin.name);
    let inspected = std::fs::symlink_metadata(&path)
        .with_context(|| format!("inspecting pinned tokenizer file {}", path.display()))?;
    ensure!(
        !inspected.file_type().is_symlink(),
        "pinned tokenizer file {} must not be a symlink",
        path.display()
    );
    ensure!(
        inspected.is_file(),
        "pinned tokenizer file {} must be a regular file",
        path.display()
    );
    ensure!(
        inspected.len() == pin.bytes,
        "pinned tokenizer file {} is {} bytes, expected exactly {}",
        path.display(),
        inspected.len(),
        pin.bytes
    );
    ensure!(
        inspected.len() <= MAX_TOKENIZER_FILE_BYTES,
        "pinned tokenizer file {} exceeds the {MAX_TOKENIZER_FILE_BYTES}-byte bound",
        path.display()
    );

    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // `O_NOFOLLOW` from the Linux asm-generic `fcntl.h`. Spelling the
        // constant here keeps this proof crate free of a `libc` dependency for
        // a single flag; every other platform falls back to the stable-open
        // check below.
        options.custom_flags(0o400000);
    }
    let file = options
        .open(&path)
        .with_context(|| format!("opening pinned tokenizer file {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("inspecting open handle for {}", path.display()))?;
    ensure_stable_open(&path, &inspected, &opened)?;

    let mut bytes = Vec::with_capacity(pin.bytes as usize);
    let mut limited = file.take(MAX_TOKENIZER_FILE_BYTES + 1);
    limited
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading pinned tokenizer file {}", path.display()))?;
    ensure!(
        bytes.len() as u64 == pin.bytes,
        "pinned tokenizer file {} read {} bytes, expected exactly {}",
        path.display(),
        bytes.len(),
        pin.bytes
    );
    Ok(bytes)
}

/// Reject a file that was swapped between inspection and open. The source path
/// is operator-supplied, but the no-follow boundary still has to hold.
fn ensure_stable_open(
    path: &Path,
    inspected: &std::fs::Metadata,
    opened: &std::fs::Metadata,
) -> Result<()> {
    ensure!(
        opened.is_file() && opened.len() == inspected.len(),
        "pinned tokenizer file {} changed between inspection and open",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            opened.dev() == inspected.dev() && opened.ino() == inspected.ino(),
            "pinned tokenizer file {} was replaced between inspection and open",
            path.display()
        );
    }
    Ok(())
}

/// Lowercase hexadecimal SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
        tokenizer.revision == TOKENIZER_REVISION,
        "the MVP tokenizer revision must be exactly {TOKENIZER_REVISION}, got {}",
        tokenizer.revision
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
    // Complete metric records and backend facts remain retained. Repeated
    // prompt bodies/token arrays and per-token diagnostic JSON are explicitly
    // omitted from dispatch evidence; counts and BLAKE3 witnesses name inputs.
    if compact_evidence {
        handle.use_compact_dispatch_evidence()?;
    }
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
    eprintln!("MVP stage: prepared; entering native simulator event loop");
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

/// Preserve the raw native backend report, the verified tokenizer identity, the
/// runner executable, and canonical role provenance as a separate, explicitly
/// non-parity artifact. The normal report's typed pair facts stay empty, so
/// nothing here can be mistaken for `ReportDynamoRunInfo`.
fn write_raw_backend_artifact(
    artifact_dir: &Path,
    backend: &ReplayReport,
    typed: &HostedTypedReport,
    prefill: &calibrated::RoleSetup,
    decode: &calibrated::RoleSetup,
    tokenizer: &TokenizerEvidence,
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
                               one verified tokenizer acquisition, and is never \
                               compared against or substituted for frozen Dynamo \
                               parity facts",
        "harness": {
            "label": "inferencex_agentx_mvp::full_scenario",
            "executable_distribution_id": harness,
        },
        "tokenizer": tokenizer,
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
        "prefill": {
            "timing": &prefill.diagnostics,
            "native_kv_geometry": &prefill.geometry,
            "replay_rank": &prefill.rank,
            "transfer": &prefill.transfer,
            "model_config": &prefill.model_config,
        },
        "decode": {
            "timing": &decode.diagnostics,
            "native_kv_geometry": &decode.geometry,
            "replay_rank": &decode.rank,
            "transfer": &decode.transfer,
            "model_config": &decode.model_config,
        },
    });
    // Serialize borrowed large reports straight to the artifact. Building a
    // serde_json::Value and a second pretty-printed Vec duplicates every
    // per-request observation at full-scenario scale.
    #[derive(serde::Serialize)]
    struct RawArtifact<'a, T: serde::Serialize> {
        #[serde(flatten)]
        provenance: &'a Value,
        backend_report: &'a ReplayReport,
        typed_phase_report: &'a T,
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
            provenance: &artifact,
            backend_report: backend,
            typed_phase_report: &typed.phased,
        },
    )
    .context("streaming raw native backend artifact")?;
    std::io::Write::flush(&mut writer).context("flushing raw native backend artifact")?;
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

#[test]
#[ignore = "requires the exact pinned MVP_TOKENIZER_ROOT/MVP_TOKENIZER_REVISION acquisition, AISIM_SYSTEMS_ROOT and a new MVP_ARTIFACT_DIR; runs the full 1800s scenario over the real Dynamo KV composition"]
fn inferencex_agentx_mvp_full_scenario_runs_on_native_dynamo_kv() -> Result<()> {
    // Static preflight first: the retained directory is created only after the
    // tokenizer, the scenario frame, the system assets, and the canonical roles
    // are proven, so a failing check cannot leave a half-built result behind.
    let artifact_path = mvp_artifact_path()?;
    let (run, tokenizer) =
        full_mvp_run(&artifact_path).context("exact InferenceX AgentX MVP run")?;
    assert_mvp_config(&run)?;
    let harness = executable_distribution_id()?;

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
    for role in [&prefill, &decode] {
        ensure!(
            role.rank.max_model_len == Some(usize::try_from(tokenizer.max_position_embeddings)?),
            "canonical role context capacity disagrees with the verified tokenizer model config"
        );
        ensure!(
            tokenizer
                .files
                .iter()
                .any(|file| file.name == "config.json" && file.sha256 == role.model_config.sha256),
            "canonical role model config bytes disagree with the verified tokenizer acquisition"
        );
    }
    let metadata = report_run_metadata(&run, &harness)?;

    let artifact_dir = create_new_artifact_dir(&artifact_path)?;
    eprintln!("MVP stage: exact input and calibrated physical roles verified");

    let (backend, typed) = host_typed(&run, &prefill, &decode, true)?;
    // Retain complete physical evidence before any acceptance assertion can
    // fail, so a population or metric defect never erases the run's output.
    write_raw_backend_artifact(
        &artifact_dir,
        &backend,
        &typed,
        &prefill,
        &decode,
        &tokenizer,
        &harness,
    )?;

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
#[ignore = "requires the exact pinned MVP_TOKENIZER_ROOT/MVP_TOKENIZER_REVISION acquisition; no builtin tokenizer fallback"]
fn inferencex_agentx_mvp_run_is_the_exact_user_scenario() -> Result<()> {
    let directory = tempfile::tempdir().context("artifact directory")?;
    let (run, _tokenizer) =
        full_mvp_run(directory.path()).context("exact InferenceX AgentX MVP run")?;
    assert_mvp_config(&run)?;
    Ok(())
}

/// Physical rejection must remain visible, not disappear from metric evidence.
/// This bounded fixture exercises prompt-at-model-limit rejection without
/// changing the full scenario's canonical model capacity.
#[test]
#[ignore = "requires AISIM_SYSTEMS_ROOT; bounded calibrated rejection accounting control"]
fn calibrated_hosted_rejections_reconcile_every_request() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let run = aiperf_protocol::runner_v2::decode_run_request_v2(
        run_fixture::branch_request(directory.path()).as_bytes(),
    )?
    .run;
    let systems = calibrated::systems_root()?;
    let mut prefill = calibrated::canonical_role(
        MODEL,
        &systems,
        ForwardPassWorkerType::Prefill,
        CONCURRENCY as usize,
    )?;
    let mut decode = calibrated::canonical_role(
        MODEL,
        &systems,
        ForwardPassWorkerType::Decode,
        CONCURRENCY as usize,
    )?;
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
/// existing branch fixture and builtin tokenizer, with no scenario designation.
#[test]
#[ignore = "requires AISIM_SYSTEMS_ROOT and pinned MVP_TOKENIZER_ROOT/MVP_TOKENIZER_REVISION; four-request calibrated artifact control, not full MVP acceptance"]
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
    let systems = calibrated::systems_root()?;
    let prefill = calibrated::canonical_role(
        MODEL,
        &systems,
        ForwardPassWorkerType::Prefill,
        CONCURRENCY as usize,
    )?;
    let decode = calibrated::canonical_role(
        MODEL,
        &systems,
        ForwardPassWorkerType::Decode,
        CONCURRENCY as usize,
    )?;
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
    let (_, tokenizer) = full_mvp_run(directory.path())?;
    write_raw_backend_artifact(
        directory.path(),
        &backend,
        &typed,
        &prefill,
        &decode,
        &tokenizer,
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

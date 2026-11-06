// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real Dynamo KV Router placement hosted by the AISim-owned AIPerf workload
//! runner. Moved verbatim from the runner's `tests/disaggregated.rs` so the
//! runner never depends on Dynamo.

use aisimulate_core::{
    engine::{EngineConfig, TimingModelConfig},
    replay::{ReplayEngineConfig, ReplayEngineFactory, loadgen::SteppableDisagg},
};
use std::rc::Rc;

// Shared with the runner's own tests; reused in place rather than copied.
#[path = "../../../../../../aisimulate/.worktrees/aiperf-owned-runner-proof/crates/tests/aiperf-workload-runner/tests/support/run_fixture.rs"]
mod run_fixture;

#[test]
fn simulator_owned_agentx_uses_real_dynamo_kv_placement_for_both_pools() {
    use aisimulate_core::replay::{ReplayComposition, ReplayDeterminism};
    use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
    let directory = tempfile::tempdir().unwrap();
    let raw = run_fixture::branch_request(directory.path());
    let frame = aiperf_protocol::runner_v2::decode_run_request_v2(raw.as_bytes()).unwrap();
    let rank = EngineConfig {
        timing_model: TimingModelConfig::Fixed { prefill_ms: 1.0, decode_ms: 2.0 },
        block_size: 8, max_num_seqs: 64, enable_prefix_caching: true,
        enable_chunked_prefill: false, async_scheduling: Some(false),
        ..Default::default()
    };
    let mut prefill = MockerConfig::default();
    prefill.engine.engine = rank.clone();
    let decode = prefill.clone();
    let mut composition = KvReplayComposition::disaggregated(prefill, decode, 2, 2, None, None, None);
    composition.set_determinism(ReplayDeterminism::CanonicalV1).unwrap();
    let deployment = SteppableDisagg::<_, dynamo_mocker::placement::RouterEventObservation,
        dynamo_mocker::placement::KvReplayMetadata>::with_placements(
        ReplayEngineConfig { rank, ..Default::default() }, &ReplayEngineFactory::new(), 2, 2,
        |pdp, ptopology, ddp, dtopology| composition.create_disaggregated_placements(pdp, ptopology, ddp, dtopology),
    ).unwrap();
    let report = aisim_aiperf_workload_runner::execute_with_deployment(
        &frame.run, aiperf_clock_runtime::cancellation::RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Box::new(deployment), 64, [7; 32],
    ).expect("actual Dynamo KV placement over disaggregated physical engines");
    assert_eq!(report["engine_drained"], true);
    assert_eq!(report["backend_report"]["completed_requests"], 4);
    for record in report["backend_per_request"].as_array().unwrap() {
        assert!(record["routing_history"].as_array().unwrap().len() >= 2);
        assert!(record["prefill_worker_idx"].is_number());
        assert!(record["decode_worker_idx"].is_number());
        assert_eq!(record["terminal_status"], "completed");
    }
}

#[test]
fn real_kv_router_reuses_cached_prefix_on_the_same_prefill_worker() {
    use aisimulate_core::replay::{DirectRequest, ReplayComposition, ReplayDeterminism, loadgen::SteppableReplay};
    use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
    let rank = EngineConfig {
        timing_model: TimingModelConfig::Fixed { prefill_ms: 1.0, decode_ms: 1.0 },
        block_size: 8, max_num_seqs: 64, enable_prefix_caching: true,
        enable_chunked_prefill: false, async_scheduling: Some(false), ..Default::default()
    };
    let mut args = MockerConfig::default();
    args.engine.engine = rank.clone();
    let mut composition = KvReplayComposition::disaggregated(args.clone(), args, 2, 2, None, None, None);
    composition.set_determinism(ReplayDeterminism::CanonicalV1).unwrap();
    let mut deployment = SteppableDisagg::<_, dynamo_mocker::placement::RouterEventObservation,
        dynamo_mocker::placement::KvReplayMetadata>::with_placements(
        ReplayEngineConfig { rank, ..Default::default() }, &ReplayEngineFactory::new(), 2, 2,
        |pdp, pt, ddp, dt| composition.create_disaggregated_placements(pdp, pt, ddp, dt),
    ).unwrap();
    deployment.set_capture_per_request(true);
    let first = uuid::Uuid::from_u128(501);
    let second = uuid::Uuid::from_u128(502);
    for id in [first, second] {
        deployment.submit(DirectRequest { uuid: Some(id), tokens: (0..64).collect(), max_output_tokens: 2, ..Default::default() }).unwrap();
        for _ in 0..1000 {
            if deployment.is_idle() { break; }
            deployment.step().unwrap();
        }
        assert!(deployment.is_idle(), "actual physical work must drain");
    }
    let report = deployment.take_report(deployment.now_ms()).unwrap();
    let first_record = report.per_request.iter().find(|row| row.uuid == first.to_string()).unwrap();
    let second_record = report.per_request.iter().find(|row| row.uuid == second.to_string()).unwrap();
    assert_eq!(first_record.prefill_worker_idx, second_record.prefill_worker_idx);
    assert!(second_record.first_admission_reused_input_tokens > 0,
        "same-worker selection must reflect actual cache reuse, not only tie-breaking");
    assert_eq!(report.request_counts.completed_requests, 2);
}

/// Independent native pull-driver control for the dynamically-fed real router.
#[test]
fn real_kv_disaggregated_reports_match_native_pull_driver_bytes() {
    use aisimulate_core::replay::{DirectRequest, ReplayComposition, ReplayDeterminism,
        ReplayCaptureOptions, ReplaySpec, Replayer, loadgen::SteppableReplay};
    use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
    let rank = EngineConfig {
        timing_model: TimingModelConfig::Fixed { prefill_ms: 1.0, decode_ms: 1.0 },
        block_size: 8, max_num_seqs: 64, enable_prefix_caching: true,
        enable_chunked_prefill: false, async_scheduling: Some(false), ..Default::default()
    };
    let config = ReplayEngineConfig { rank: rank.clone(), ..Default::default() };
    let make_composition = || {
        let mut args = MockerConfig::default(); args.engine.engine = rank.clone();
        KvReplayComposition::disaggregated(args.clone(), args, 2, 2, None, None, None)
    };
    let spec: ReplaySpec = serde_json::from_value(serde_json::json!({
        "version":1,"topology":{"kind":"disaggregated",
            "prefill":{"initial_workers":2},"decode":{"initial_workers":2},"handoff_latency_ms":0.0},
        "engine": config,"adapters":{"placement":{"provider":"dynamo_kv_router"}},
        "requests": (0..4).map(|index| serde_json::json!({
            "id":format!("request-{index}"),"arrival_time_ms":index as f64 * 10.0,
            "input_tokens":64,"input_token_ids":(0..64).collect::<Vec<u32>>(),"output_tokens":2,
        })).collect::<Vec<_>>()
    })).unwrap();
    let capture = ReplayCaptureOptions { capture_per_request:true,
        determinism:ReplayDeterminism::CanonicalV1, ..Default::default() };
    let reference = Replayer::with_composition(spec, ReplayEngineFactory::new(), make_composition())
        .unwrap().with_capture_options(capture).run().unwrap();
    let mut composition = make_composition();
    composition.set_determinism(ReplayDeterminism::CanonicalV1).unwrap();
    let mut candidate = SteppableDisagg::<_, dynamo_mocker::placement::RouterEventObservation,
        dynamo_mocker::placement::KvReplayMetadata>::with_placements(
        config, &ReplayEngineFactory::new(), 2, 2,
        |pdp, pt, ddp, dt| composition.create_disaggregated_placements(pdp, pt, ddp, dt),
    ).unwrap();
    candidate.set_capture_per_request(true);
    for index in 0..4 {
        candidate.advance_now_ms(index as f64 * 10.0);
        candidate.submit(DirectRequest { uuid:Some(uuid::Uuid::from_u128(index+1)),
            arrival_timestamp_ms:Some(index as f64 * 10.0), tokens:(0..64).collect(),
            max_output_tokens:2, ..Default::default() }).unwrap();
        for _ in 0..1000 { if candidate.is_idle() { break; } candidate.step().unwrap(); }
        assert!(candidate.is_idle());
    }
    let actual = candidate.take_report(candidate.now_ms()).unwrap();
    // Native run measures host execution cost; the external driver receives
    // this stamp explicitly. Supply the same stamp, not an estimated cost.
    let actual = actual.with_wall_time_ms(reference.throughput.wall_time_ms);
    let reference_bytes = serde_json::to_vec(&reference).unwrap();
    let actual_bytes = serde_json::to_vec(&actual).unwrap();
    if reference_bytes != actual_bytes {
        let reference_value = serde_json::to_value(&reference).unwrap();
        let actual_value = serde_json::to_value(&actual).unwrap();
        for (key, value) in reference_value.as_object().unwrap() {
            if actual_value.get(key) != Some(value) {
                eprintln!("DIFF {key}: reference={value} candidate={}", actual_value[key]);
            }
        }
    }
    assert_eq!(reference_bytes, actual_bytes,
        "all aggregate, routing, handoff, cache and per-request fields must retain exact values");
}

/// Exact comparison of two JSON trees. Floats are compared by IEEE-754 bit
/// pattern (no tolerance); an integer never equals a float.
fn first_difference(path: &str, left: &serde_json::Value, right: &serde_json::Value) -> Option<String> {
    use serde_json::Value;
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => {
            let equal = if a.is_f64() || b.is_f64() {
                a.is_f64() && b.is_f64() && a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits)
            } else {
                a == b
            };
            (!equal).then(|| format!("{path}: capture={a} native={b}"))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path}: capture length {} native length {}", a.len(), b.len()));
            }
            a.iter().zip(b).enumerate()
                .find_map(|(index, (x, y))| first_difference(&format!("{path}[{index}]"), x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys().chain(b.keys()) {
                match (a.get(key), b.get(key)) {
                    (Some(x), Some(y)) => {
                        if let Some(found) = first_difference(&format!("{path}.{key}"), x, y) {
                            return Some(found);
                        }
                    }
                    _ => return Some(format!("{path}.{key}: present on one side only")),
                }
            }
            None
        }
        _ => (left != right).then(|| format!("{path}: capture={left} native={right}")),
    }
}

/// Strictly typed `deployment.json` of the installed-entrypoint capture.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedDeployment {
    engine: ReplayEngineConfig,
    prefill_workers: usize,
    decode_workers: usize,
    admission_capacity: usize,
    /// Omitted means the production default router configuration.
    #[serde(default)]
    router_config: Option<dynamo_mocker::placement::KvRouterConfig>,
}

/// Delegates to the real Dynamo KV composition but pins `CanonicalV1` selector
/// seeding. `Replayer::run` re-applies its capture determinism through
/// `set_determinism`; the capture stays `Random` so explicit request ids are
/// preserved, while this wrapper keeps the router selectors canonically seeded.
struct CanonicalSelectorComposition(dynamo_mocker::placement::KvReplayComposition);

impl aisimulate_core::replay::ReplayComposition for CanonicalSelectorComposition {
    type Metadata = <dynamo_mocker::placement::KvReplayComposition as aisimulate_core::replay::ReplayComposition>::Metadata;
    type Observation = <dynamo_mocker::placement::KvReplayComposition as aisimulate_core::replay::ReplayComposition>::Observation;
    type AggregatedPlacement = <dynamo_mocker::placement::KvReplayComposition as aisimulate_core::replay::ReplayComposition>::AggregatedPlacement;
    type DisaggregatedPlacement = <dynamo_mocker::placement::KvReplayComposition as aisimulate_core::replay::ReplayComposition>::DisaggregatedPlacement;

    fn validate_spec(&self, spec: &aisimulate_core::replay::ReplaySpec) -> aisimulate_core::replay::ReplayResult<()> {
        self.0.validate_spec(spec)
    }
    fn create_aggregated_placement(
        &mut self, dp_size: u32, topology: Vec<aisimulate_core::replay::WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        self.0.create_aggregated_placement(dp_size, topology)
    }
    fn create_disaggregated_placements(
        &mut self, prefill_dp_size: u32, prefill_topology: Vec<aisimulate_core::replay::WorkerTopology>,
        decode_dp_size: u32, decode_topology: Vec<aisimulate_core::replay::WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        self.0.create_disaggregated_placements(prefill_dp_size, prefill_topology, decode_dp_size, decode_topology)
    }
    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn aisimulate_core::replay::ReplayScalingPolicy>>> {
        self.0.take_scaling_policy()
    }
    fn set_determinism(&mut self, _requested: aisimulate_core::replay::ReplayDeterminism)
        -> aisimulate_core::replay::ReplayResult<()> {
        self.0.set_determinism(aisimulate_core::replay::ReplayDeterminism::CanonicalV1)
    }
}

/// Conditional parity: the INSTALLED Python-entrypoint capture versus the native
/// pull `Replayer` (real Dynamo KV Router composition), fed the same captured
/// requests (uuid, input_token_ids, max_output_tokens, start_ns) and the same
/// deployment. This does NOT prove original-AIPerf scheduling semantics: it
/// proves that, given identical captured arrivals and inputs, the native
/// replay policy reproduces the capture's backend report bit for bit.
#[test]
#[ignore = "requires explicit installed provider capture"]
fn installed_entrypoint_capture_matches_native_replayer_for_same_captured_requests() {
    use aisimulate_core::replay::{DirectRequest, ReplayCaptureOptions, ReplayDeterminism,
        ReplayRuntimeInput, ReplaySpec, Replayer};
    use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
    use serde_json::{Value, json};
    use std::collections::VecDeque;

    let directory = std::path::PathBuf::from(
        std::env::var_os("AIPERF_ENTRYPOINT_CAPTURE_DIR").expect("AIPERF_ENTRYPOINT_CAPTURE_DIR must name the capture directory"));
    let read = |name: &str| -> Vec<u8> {
        std::fs::read(directory.join(name)).unwrap_or_else(|error| panic!("reading {name}: {error}"))
    };
    let result: Value = serde_json::from_slice(&read("result.json")).expect("parsing result.json");
    let deployment: CapturedDeployment =
        serde_json::from_slice(&read("deployment.json")).expect("deployment.json is not a strict CapturedDeployment");

    // Exact captured requests, in issue order, with captured identities.
    let rows = result["requests"].as_array().expect("result.json `requests` array");
    assert!(!rows.is_empty(), "capture must not pass vacuously");
    let mut requests = VecDeque::new();
    let mut previous_start_ns = i64::MIN;
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row["issue_order"], index, "requests must be in issue order");
        // `start_ns` is the actual modeled submit time; `issue_ns` precedes the capacity wait.
        let start_ns = row["start_ns"].as_i64().expect("start_ns must be an integer");
        assert!(start_ns >= previous_start_ns, "start_ns must be non-decreasing in issue order");
        previous_start_ns = start_ns;
        let uuid: uuid::Uuid = row["request"]["uuid"].as_str().expect("request.uuid string")
            .parse().expect("request.uuid must be a UUID");
        let tokens: Vec<u32> = serde_json::from_value(row["input_token_ids"].clone()).expect("input_token_ids as u32");
        let max_output_tokens = usize::try_from(row["request"]["max_output_tokens"].as_u64().expect("request.max_output_tokens"))
            .expect("max_output_tokens fits usize");
        requests.push_back(DirectRequest {
            uuid: Some(uuid), tokens, max_output_tokens,
            arrival_timestamp_ms: Some(start_ns as f64 / 1_000_000.0),
            ..Default::default()
        });
    }

    let rank = deployment.engine.rank.clone();
    let mut args = MockerConfig::default();
    args.engine.engine = rank;
    let composition = CanonicalSelectorComposition(KvReplayComposition::disaggregated(
        args.clone(), args, deployment.prefill_workers, deployment.decode_workers,
        deployment.router_config, None, None));
    // Arrivals already include any capacity wait (start_ns); the cap is kept as
    // the source-side in-flight bound to preserve the captured deployment.
    let spec: ReplaySpec = serde_json::from_value(json!({
        "version":1,
        "topology":{"kind":"disaggregated",
            "prefill":{"initial_workers":deployment.prefill_workers},
            "decode":{"initial_workers":deployment.decode_workers},"handoff_latency_ms":0.0},
        "engine": deployment.engine, "adapters":{"placement":{"provider":"dynamo_kv_router"}},
        "max_in_flight": deployment.admission_capacity,
        "requests": [],
    })).unwrap();
    // Random keeps the supplied explicit ids untouched; the wrapper keeps the
    // router selectors canonically seeded as the capture's provider did.
    let capture = ReplayCaptureOptions { capture_per_request: true,
        determinism: ReplayDeterminism::Random, ..Default::default() };
    let native = Replayer::with_composition(spec, ReplayEngineFactory::new(), composition).unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(requests))
        .with_capture_options(capture)
        .run().unwrap();

    // The capture stamps the supplied virtual duration as top-level wall_time_ms.
    let captured_report = &result["backend_report"];
    let wall_time_ms = captured_report["wall_time_ms"].as_f64()
        .expect("capture backend_report.wall_time_ms");
    let native = native.with_wall_time_ms(wall_time_ms);
    // Aggregate serialization skips per_request; compare it separately. With
    // serde_json float_roundtrip the file parse and to_value agree bit for bit.
    let native_report: Value = serde_json::to_value(&native).unwrap();
    let native_per_request: Value = serde_json::to_value(&native.per_request).unwrap();
    let captured_per_request = &result["backend_per_request"];
    assert!(native_per_request.as_array().is_some_and(|rows| !rows.is_empty()), "native per_request must be populated");
    let per_request = first_difference("backend_per_request", captured_per_request, &native_per_request);
    let aggregate = first_difference("backend_report", captured_report, &native_report);
    assert!(per_request.is_none() && aggregate.is_none(),
        "native Replayer diverges from the installed-entrypoint capture (same captured arrivals/inputs): per_request={per_request:?} aggregate={aggregate:?}");
}

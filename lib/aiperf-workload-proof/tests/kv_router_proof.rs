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

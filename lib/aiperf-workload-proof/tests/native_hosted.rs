// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production AgentX hosted by AISim's native Replayer event loop over the
//! real Dynamo KV Router composition, disaggregated 2P2D.
//!
//! The AIPerf-owned workload is lowered by the shared runner and arrives as
//! `participant_adapter::native_hosted`. The Replayer owns logical time and
//! every engine decision; tokens and terminals return to the participant as
//! mailbox facts. No `drive_sim*` pump and no `SteppableReplay` is involved in
//! the proof itself.
//!
//! The clock-hosted `execute_with_deployment` run is retained only as an
//! oracle for identical inputs. It is not a candidate and cannot substitute
//! for the native proof; the raw oracle output is printed for inspection.

use aiperf_clock_runtime::cancellation::RunCancellation;
use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayComposition, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, Replayer,
    loadgen::SteppableDisagg,
};
use dynamo_mocker::placement::{
    KvReplayComposition, KvReplayMetadata, MockerConfig, RouterEventObservation,
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, rc::Rc};

// Shared with the runner's own tests; reused in place rather than copied.
#[path = "../../../../../../aisimulate/.worktrees/aiperf-owned-runner-proof/crates/tests/aiperf-workload-runner/tests/support/run_fixture.rs"]
mod run_fixture;

const IDENTITY_SEED: [u8; 32] = [7; 32];
const PREFILL_WORKERS: usize = 2;
const DECODE_WORKERS: usize = 2;
const ADMISSION_CAPACITY: usize = 64;

/// The same fixed single-rank model used by the existing KV proof and the
/// installed entrypoint capture.
fn rank() -> EngineConfig {
    EngineConfig {
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 1.0,
            decode_ms: 2.0,
        },
        block_size: 8,
        max_num_seqs: 64,
        enable_prefix_caching: true,
        enable_chunked_prefill: false,
        async_scheduling: Some(false),
        ..Default::default()
    }
}

fn native_run(frame: &str) -> aiperf_config::model::BenchmarkRun {
    aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes())
        .expect("native strict codec")
        .run
}

/// Disaggregated 2P2D spec for the real Dynamo KV composition. Requests are
/// supplied by the hosted participant, not by the serializable spec.
fn spec(rank: &EngineConfig) -> ReplaySpec {
    serde_json::from_value(json!({
        "version": 1,
        "topology": {
            "kind": "disaggregated",
            "prefill": {"initial_workers": PREFILL_WORKERS},
            "decode": {"initial_workers": DECODE_WORKERS},
            "handoff_latency_ms": 0.0
        },
        "engine": ReplayEngineConfig { rank: rank.clone(), ..Default::default() },
        "adapters": {"placement": {"provider": "dynamo_kv_router"}},
        "requests": []
    }))
    .expect("strict disaggregated ReplaySpec")
}

/// Build through the owning launch parser, as the actual provider does, so the
/// fixed timing model also selects `PerfModel::Fixed` for the router cost model.
fn composition(rank: &EngineConfig) -> KvReplayComposition {
    let args = MockerConfig::from_value(json!({"engine": rank.clone()}))
        .expect("fixed engine rank must be a valid mocker launch config");
    KvReplayComposition::disaggregated(
        args.clone(),
        args,
        PREFILL_WORKERS,
        DECODE_WORKERS,
        None,
        None,
        None,
    )
}

/// Host the production AIPerf plans on the native Replayer event loop.
fn host(run: &aiperf_config::model::BenchmarkRun, rank: &EngineConfig) -> (ReplayReport, Value) {
    let (participant, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        run,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            IDENTITY_SEED,
        )),
    )
    .expect("shared native lowering hosts the production plans");
    let report =
        Replayer::with_composition(spec(rank), ReplayEngineFactory::new(), composition(rank))
            .expect("real Dynamo KV composition accepts the disaggregated spec")
            .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
            .with_capture_options(ReplayCaptureOptions {
                capture_per_request: true,
                determinism: ReplayDeterminism::CanonicalV1,
                ..Default::default()
            })
            .run()
            .expect("native Replayer event loop hosts AIPerf over the real KV composition");
    (report, handle.into_report().expect("AIPerf phase report"))
}

/// Oracle only: the existing clock-hosted deployment path over the same
/// composition, seed and run. Never used as the proof candidate.
fn clock_hosted_oracle(run: &aiperf_config::model::BenchmarkRun, rank: &EngineConfig) -> Value {
    let mut composition = composition(rank);
    composition
        .set_determinism(ReplayDeterminism::CanonicalV1)
        .expect("canonical selector determinism");
    let deployment =
        SteppableDisagg::<_, RouterEventObservation, KvReplayMetadata>::with_placements(
            ReplayEngineConfig {
                rank: rank.clone(),
                ..Default::default()
            },
            &ReplayEngineFactory::new(),
            PREFILL_WORKERS,
            DECODE_WORKERS,
            |pdp, ptopology, ddp, dtopology| {
                composition.create_disaggregated_placements(pdp, ptopology, ddp, dtopology)
            },
        )
        .expect("disaggregated physical engines");
    aisim_aiperf_workload_runner::execute_with_deployment(
        run,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Box::new(deployment),
        ADMISSION_CAPACITY,
        IDENTITY_SEED,
    )
    .expect("clock-hosted oracle")
}

fn ns_close(ns: &Value, ms: f64) -> bool {
    (ns.as_i64().expect("ns value") as f64 / 1_000_000.0 - ms).abs() < 1e-6
}

/// Lineage, per-request time agreement and actual two-pool routing for the
/// seeded strict four-request branch frame.
fn assert_agentx_branch(report: &ReplayReport, hosted: &Value) {
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
    let (parents, children): (Vec<_>, Vec<_>) = rows
        .iter()
        .partition(|row| row["request"]["correlation_id"] == row["request"]["root_correlation_id"]);
    assert_eq!((parents.len(), children.len()), (2, 2));
    for child in &children {
        assert_eq!(
            child["request"]["parent_correlation_id"],
            parents[0]["request"]["correlation_id"]
        );
        assert!(child["issue_ns"].as_i64().unwrap() >= parents[0]["issue_ns"].as_i64().unwrap());
    }
    for group in [&parents, &children] {
        for pair in group.windows(2) {
            assert!(
                pair[1]["issue_ns"].as_i64().unwrap() >= pair[0]["end_ns"].as_i64().unwrap(),
                "a session's next turn is released only after its previous response"
            );
        }
    }

    // The Replayer's per-request time axis must agree with what AIPerf saw.
    let records = serde_json::to_value(&report.per_request).expect("engine records");
    let records = records.as_array().expect("engine per-request records");
    assert_eq!(records.len(), 4, "engine must retain every request");
    for row in rows {
        let uuid = row["request"]["uuid"].as_str().expect("request uuid");
        let record = records
            .iter()
            .find(|record| record["uuid"] == uuid)
            .unwrap_or_else(|| panic!("engine has no record for {uuid}"));
        assert!(ns_close(
            &row["start_ns"],
            record["arrival_time_ms"].as_f64().unwrap()
        ));
        assert!(ns_close(
            &row["end_ns"],
            record["terminal_time_ms"].as_f64().unwrap()
        ));
        assert!(ns_close(
            &row["first_content_ns"],
            record["first_token_ms"].as_f64().unwrap()
        ));
        assert!(ns_close(
            &json!(row["first_ttft_ns"].as_i64().unwrap()),
            record["ttft_ms"].as_f64().unwrap()
        ));
    }

    // Actual Dynamo KV routing: every request is placed on one prefill and one
    // decode pool, and the recorded worker indices are real 2P2D identities.
    for record in records {
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
    }
}

#[test]
fn native_replayer_hosts_production_agentx_over_real_dynamo_kv_2p2d() {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&run_fixture::branch_request(directory.path()));
    let (report, hosted) = host(&run, &rank());
    assert_agentx_branch(&report, &hosted);
}

/// Exact JSON comparison, including IEEE-754 bits rather than approximate or
/// numeric JSON equality. Every key and array member must be present on both sides.
fn assert_exact(actual: &Value, expected: &Value, path: &str) -> usize {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            assert_eq!(
                a.as_f64().unwrap().to_bits(),
                b.as_f64().unwrap().to_bits(),
                "{path}"
            );
            1
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: population");
            a.iter()
                .zip(b)
                .enumerate()
                .map(|(index, (a, b))| assert_exact(a, b, &format!("{path}/{index}")))
                .sum()
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path}: keys"
            );
            a.iter()
                .map(|(key, a)| assert_exact(a, &b[key], &format!("{path}/{key}")))
                .sum()
        }
        _ => {
            assert_eq!(actual, expected, "{path}");
            0
        }
    }
}

/// Exact comparison of two JSON trees. Floats are compared by IEEE-754 bit
/// pattern (no tolerance); an integer never equals a float.
fn first_difference(
    path: &str,
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> Option<String> {
    use serde_json::Value;
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => {
            let equal = if a.is_f64() || b.is_f64() {
                a.is_f64()
                    && b.is_f64()
                    && a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits)
            } else {
                a == b
            };
            (!equal).then(|| format!("{path}: oracle={a} native={b}"))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!(
                    "{path}: oracle length {} native length {}",
                    a.len(),
                    b.len()
                ));
            }
            a.iter()
                .zip(b)
                .enumerate()
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
        _ => (left != right).then(|| format!("{path}: oracle={left} native={right}")),
    }
}

/// Same inputs, same composition, same seed: compare the native-hosted backend
/// against the clock-hosted oracle. The participant contributes no admission
/// bound (engine `max_num_seqs` is the only capacity), while the oracle is given
/// `ADMISSION_CAPACITY`; the fixture submits four requests, so neither limit is
/// exercised. Host-processing wall time and its two derived rates are measured
/// by the native Replayer and stamped as virtual time by the oracle; both raw
/// stamps are printed and only those three fields are excluded.
#[test]
fn native_hosted_kv_backend_matches_clock_hosted_oracle_for_same_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let run = native_run(&run_fixture::branch_request(directory.path()));
    let rank = rank();
    let (report, hosted) = host(&run, &rank);
    let oracle = clock_hosted_oracle(&run, &rank);

    println!("NATIVE_HOSTED_RAW {hosted}");
    println!("CLOCK_HOSTED_RAW {oracle}");
    println!(
        "LIMITS native_participant_admission=none engine_max_num_seqs={} oracle_admission_capacity={ADMISSION_CAPACITY} fixture_requests=4",
        rank.max_num_seqs
    );
    println!(
        "WALL_STAMP native_wall_time_ms={} oracle_wall_time_ms={}",
        report.throughput.wall_time_ms, oracle["backend_report"]["wall_time_ms"]
    );

    assert_exact(&hosted["requests"], &oracle["requests"], "requests");
    assert_exact(
        &hosted["native_phase_report"],
        &oracle["native_phase_report"],
        "native_phase_report",
    );
    assert_exact(
        &hosted["virtual_end_ns"],
        &oracle["virtual_end_ns"],
        "virtual_end_ns",
    );

    let mut native_backend = serde_json::to_value(&report).expect("native backend report");
    let mut oracle_backend = oracle["backend_report"].clone();
    for key in [
        "wall_time_ms",
        "processed_tokens_per_s",
        "processed_output_tokens_per_s",
    ] {
        assert!(
            native_backend
                .as_object_mut()
                .unwrap()
                .remove(key)
                .is_some(),
            "native backend must retain {key}"
        );
        assert!(
            oracle_backend
                .as_object_mut()
                .unwrap()
                .remove(key)
                .is_some(),
            "oracle backend must retain {key}"
        );
    }
    let native_per_request =
        serde_json::to_value(&report.per_request).expect("native per-request records");
    let backend = first_difference("backend_report", &oracle_backend, &native_backend);
    let per_request = first_difference(
        "backend_per_request",
        &oracle["backend_per_request"],
        &native_per_request,
    );
    assert!(
        backend.is_none() && per_request.is_none(),
        "native hosting diverges from the clock-hosted oracle for identical inputs: backend={backend:?} per_request={per_request:?}"
    );
}

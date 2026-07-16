// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{OverlapScores, WorkerWithDpRank};

use super::offline::components::OfflineReplayRouter;
use super::{
    ReplayKvObservationMode, ReplayRouterMode, ReplaySessionAffinityMode, SlaThresholds,
    simulate_loaded_trace_with_router_mode_and_options, simulate_session_affinity_workload,
    simulate_session_affinity_workload_with_options,
};
use crate::{
    common::protocols::MockEngineArgs,
    loadgen::{SessionTrace, Trace, TurnTrace},
};

fn replay_args() -> MockEngineArgs {
    replay_args_with_speedup(1000.0)
}

fn replay_args_with_speedup(speedup_ratio: f64) -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(16)
        .num_gpu_blocks(4096)
        .max_num_batched_tokens(Some(4096))
        .max_num_seqs(Some(64))
        .enable_prefix_caching(true)
        .enable_chunked_prefill(true)
        .speedup_ratio(speedup_ratio)
        .build()
        .unwrap()
}

fn cumulative_trace(num_sessions: usize, turns_per_session: usize) -> Trace {
    let block_size = 16;
    let initial_blocks = 16;
    let added_blocks_per_turn = 4;
    let sessions = (0..num_sessions)
        .map(|session_idx| {
            let mut hashes = Vec::new();
            let turns = (0..turns_per_session)
                .map(|turn_idx| {
                    let target_blocks = initial_blocks + turn_idx * added_blocks_per_turn;
                    while hashes.len() < target_blocks {
                        let block_idx = hashes.len();
                        hashes.push((session_idx as u64 + 1) * 1_000_000 + block_idx as u64 + 1);
                    }
                    TurnTrace {
                        input_length: hashes.len() * block_size,
                        max_output_tokens: 16,
                        hash_ids: hashes.clone(),
                        delay_after_previous_ms: if turn_idx == 0 { 0.0 } else { 20.0 },
                        ..Default::default()
                    }
                })
                .collect();
            SessionTrace {
                session_id: format!("session-{session_idx}"),
                first_arrival_timestamp_ms: Some(session_idx as f64 * 0.5),
                turns,
            }
        })
        .collect();
    Trace {
        block_size,
        sessions,
    }
}

fn hrw_collision_trace(num_sessions: usize, turns_per_session: usize) -> Trace {
    let candidates = worker_rank_targets(4, 1);
    let target = candidates[0];
    let mut affinity =
        super::session_affinity::ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
    let session_ids = (0..)
        .map(|idx| format!("hrw-collision-{idx}"))
        .filter(|session_id| {
            affinity.preferred_target(Some(session_id), &candidates) == Some(target)
        })
        .take(num_sessions)
        .collect::<Vec<_>>();
    let mut trace = cumulative_trace(num_sessions, turns_per_session);
    for (session, session_id) in trace.sessions.iter_mut().zip(session_ids) {
        session.session_id = session_id;
    }
    trace
}

fn target_transition_rate(report: &super::TraceSimulationReport) -> f64 {
    let mut per_session = HashMap::<&str, Vec<(usize, usize)>>::new();
    for row in &report.per_request {
        if let (Some(session_id), Some(turn), Some(worker)) = (
            row.session_id.as_deref(),
            row.turn_index,
            row.decode_worker_idx,
        ) {
            per_session
                .entry(session_id)
                .or_default()
                .push((turn, worker));
        }
    }
    let mut transitions = 0;
    let mut opportunities = 0;
    for turns in per_session.values_mut() {
        turns.sort_unstable_by_key(|(turn, _)| *turn);
        for pair in turns.windows(2) {
            transitions += usize::from(pair[0].1 != pair[1].1);
            opportunities += 1;
        }
    }
    transitions as f64 / opportunities.max(1) as f64
}

fn run_affinity_for(trace: Trace, mode: ReplaySessionAffinityMode) -> super::TraceSimulationReport {
    simulate_session_affinity_workload(
        replay_args(),
        Some(KvRouterConfig::default()),
        trace,
        4,
        mode,
        true,
        SlaThresholds::default(),
    )
    .unwrap()
}

fn run_affinity_with_observation(
    trace: Trace,
    mode: ReplaySessionAffinityMode,
    observation: ReplayKvObservationMode,
) -> super::TraceSimulationReport {
    run_affinity_with_args_and_observation(replay_args(), trace, mode, observation)
}

fn run_affinity_with_args_and_observation(
    args: MockEngineArgs,
    trace: Trace,
    mode: ReplaySessionAffinityMode,
    observation: ReplayKvObservationMode,
) -> super::TraceSimulationReport {
    simulate_session_affinity_workload_with_options(
        args,
        Some(KvRouterConfig::default()),
        trace,
        4,
        mode,
        observation,
        true,
        SlaThresholds::default(),
    )
    .unwrap()
}

fn run_round_robin_for(trace: Trace) -> super::TraceSimulationReport {
    simulate_loaded_trace_with_router_mode_and_options(
        replay_args(),
        None,
        None,
        trace,
        4,
        1.0,
        ReplayRouterMode::RoundRobin,
        true,
        None,
        SlaThresholds::default(),
    )
    .unwrap()
}

fn run_affinity(mode: ReplaySessionAffinityMode) -> super::TraceSimulationReport {
    run_affinity_for(cumulative_trace(31, 6), mode)
}

fn reused_tokens(report: &super::TraceSimulationReport) -> usize {
    report
        .per_request
        .iter()
        .map(|row| row.reused_input_tokens)
        .sum()
}

fn worker_load_cv(report: &super::TraceSimulationReport, num_workers: usize) -> f64 {
    let mut counts = vec![0.0_f64; num_workers];
    for row in &report.per_request {
        if let Some(worker) = row.decode_worker_idx {
            counts[worker] += 1.0;
        }
    }
    let mean = counts.iter().sum::<f64>() / counts.len() as f64;
    let variance = counts
        .iter()
        .map(|count| (count - mean).powi(2))
        .sum::<f64>()
        / counts.len() as f64;
    variance.sqrt() / mean.max(f64::EPSILON)
}

fn scale_out_remaps(mode: ReplaySessionAffinityMode, sessions: usize) -> usize {
    let before = (0..8)
        .map(|worker_id| WorkerWithDpRank::new(worker_id, 0))
        .collect::<Vec<_>>();
    let after = (0..9)
        .map(|worker_id| WorkerWithDpRank::new(worker_id, 0))
        .collect::<Vec<_>>();
    let mut router = super::session_affinity::ReplaySessionAffinity::new(mode);
    (0..sessions)
        .filter(|idx| {
            let session = format!("session-{idx}");
            router.preferred_target(Some(&session), &before)
                != router.preferred_target(Some(&session), &after)
        })
        .count()
}

fn worker_rank_targets(worker_count: u64, ranks_per_worker: u32) -> Vec<WorkerWithDpRank> {
    (0..worker_count)
        .flat_map(|worker_id| {
            (0..ranks_per_worker).map(move |dp_rank| WorkerWithDpRank::new(worker_id, dp_rank))
        })
        .collect()
}

fn mapping_target_distribution(
    mode: ReplaySessionAffinityMode,
    sessions: usize,
    candidates: &[WorkerWithDpRank],
) -> (usize, f64) {
    let mut counts = BTreeMap::<WorkerWithDpRank, usize>::new();
    let mut router = super::session_affinity::ReplaySessionAffinity::new(mode);
    for idx in 0..sessions {
        let session = format!("session-{idx}");
        let target = router.preferred_target(Some(&session), candidates).unwrap();
        *counts.entry(target).or_default() += 1;
    }

    let mean = sessions as f64 / candidates.len() as f64;
    let variance = candidates
        .iter()
        .map(|target| {
            let count = *counts.get(target).unwrap_or(&0) as f64;
            (count - mean).powi(2)
        })
        .sum::<f64>()
        / candidates.len() as f64;
    (counts.len(), variance.sqrt() / mean)
}

#[derive(Debug)]
struct KvConflictMetrics {
    home_selection_rate: f64,
    mean_extra_uncached_tokens: f64,
    p95_extra_uncached_tokens: usize,
    divergent_snapshot_dispatch_agreement: f64,
}

fn kv_conflict_metrics(mode: ReplaySessionAffinityMode, sessions: usize) -> KvConflictMetrics {
    assert!(sessions > 0);
    let candidates = worker_rank_targets(8, 1);
    let block_size = 16_u32;
    let overlap_advantage_blocks = [0_u32, 4, 8, 16, 32];
    let mut strict =
        super::session_affinity::ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
    let mut left_router = super::session_affinity::ReplaySessionAffinity::new(mode);
    let mut right_router = super::session_affinity::ReplaySessionAffinity::new(mode);
    let mut home_selections = 0;
    let mut dispatch_agreements = 0;
    let mut extra_uncached_tokens = Vec::with_capacity(sessions);

    for idx in 0..sessions {
        let session = format!("kv-conflict-session-{idx}");
        let home = strict
            .preferred_target(Some(&session), &candidates)
            .unwrap();
        let home_index = candidates
            .iter()
            .position(|candidate| *candidate == home)
            .unwrap();
        let left_best = candidates[(home_index + 1) % candidates.len()];
        let right_best = candidates[(home_index + 2) % candidates.len()];
        let advantage_blocks = overlap_advantage_blocks[idx % overlap_advantage_blocks.len()];
        let home_overlap_blocks = 16_u32;
        let best_overlap_blocks = home_overlap_blocks + advantage_blocks;

        let mut left_overlap = OverlapScores::default();
        left_overlap.scores.insert(home, home_overlap_blocks);
        left_overlap.scores.insert(left_best, best_overlap_blocks);
        let mut right_overlap = OverlapScores::default();
        right_overlap.scores.insert(home, home_overlap_blocks);
        right_overlap.scores.insert(right_best, best_overlap_blocks);

        let left = left_router
            .preferred_target_with_overlap(Some(&session), &candidates, &left_overlap, block_size)
            .unwrap();
        let right = right_router
            .preferred_target_with_overlap(Some(&session), &candidates, &right_overlap, block_size)
            .unwrap();

        home_selections += usize::from(left == home);
        dispatch_agreements += usize::from(left == right);
        let selected_overlap = left_overlap.scores.get(&left).copied().unwrap_or(0);
        extra_uncached_tokens.push(
            ((best_overlap_blocks - selected_overlap) as usize).saturating_mul(block_size as usize),
        );
    }

    extra_uncached_tokens.sort_unstable();
    let p95_index = ((extra_uncached_tokens.len() - 1) as f64 * 0.95).round() as usize;
    KvConflictMetrics {
        home_selection_rate: home_selections as f64 / sessions as f64,
        mean_extra_uncached_tokens: extra_uncached_tokens.iter().sum::<usize>() as f64
            / sessions as f64,
        p95_extra_uncached_tokens: extra_uncached_tokens[p95_index],
        divergent_snapshot_dispatch_agreement: dispatch_agreements as f64 / sessions as f64,
    }
}

#[test]
fn cumulative_fixture_reuses_every_prior_block() {
    let trace = cumulative_trace(1, 4);
    let turns = &trace.sessions[0].turns;
    for pair in turns.windows(2) {
        assert!(pair[1].hash_ids.starts_with(&pair[0].hash_ids));
        assert!(pair[1].input_length > pair[0].input_length);
    }
}

#[test]
fn local_and_hrw_affinity_eliminate_session_target_transitions() {
    let local = run_affinity(ReplaySessionAffinityMode::Local);
    let hrw = run_affinity(ReplaySessionAffinityMode::Hrw);

    assert_eq!(target_transition_rate(&local), 0.0);
    assert_eq!(target_transition_rate(&hrw), 0.0);
}

#[test]
fn sticky_routing_improves_reuse_over_round_robin() {
    let trace = cumulative_trace(1, 6);
    let round_robin = run_round_robin_for(trace.clone());
    let local = run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Local);
    let modulo = run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Modulo);
    let hrw = run_affinity_for(trace, ReplaySessionAffinityMode::Hrw);

    for candidate in [&local, &modulo, &hrw] {
        assert!(
            candidate.prefix_cache_reused_ratio > round_robin.prefix_cache_reused_ratio,
            "affinity reuse {} must beat round-robin {}",
            candidate.prefix_cache_reused_ratio,
            round_robin.prefix_cache_reused_ratio,
        );
    }
}

#[test]
fn affinity_matches_ideal_single_router_kv_baseline() {
    let trace = cumulative_trace(31, 6);
    let baseline = run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Disabled);
    for candidate in [
        run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Local),
        run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Modulo),
        run_affinity_for(trace, ReplaySessionAffinityMode::Hrw),
    ] {
        assert_eq!(reused_tokens(&candidate), reused_tokens(&baseline));
        assert_eq!(
            candidate.prefix_cache_reused_ratio,
            baseline.prefix_cache_reused_ratio
        );
        assert!((candidate.latency.ttft.mean_ms - baseline.latency.ttft.mean_ms).abs() < 1.0e-12);
        assert!((candidate.latency.ttft.p95_ms - baseline.latency.ttft.p95_ms).abs() < 1.0e-12);
    }
}

#[test]
fn hybrid_matches_kv_baseline_with_complete_router_observation() {
    let trace = cumulative_trace(31, 6);
    let baseline = run_affinity_with_observation(
        trace.clone(),
        ReplaySessionAffinityMode::Disabled,
        ReplayKvObservationMode::Complete,
    );
    let hybrid = run_affinity_with_observation(
        trace,
        ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
            max_entries: 65_536,
            max_cost_regret_blocks: 0,
        },
        ReplayKvObservationMode::Complete,
    );

    assert_eq!(reused_tokens(&hybrid), reused_tokens(&baseline));
    assert_eq!(
        hybrid.prefix_cache_reused_ratio,
        baseline.prefix_cache_reused_ratio
    );
}

#[test]
fn hybrid_beats_kv_only_when_router_kv_events_are_dropped() {
    let trace = cumulative_trace(64, 8);
    let baseline = run_affinity_with_observation(
        trace.clone(),
        ReplaySessionAffinityMode::Disabled,
        ReplayKvObservationMode::DropAll,
    );
    let hybrid = run_affinity_with_observation(
        trace,
        ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
            max_entries: 65_536,
            max_cost_regret_blocks: 0,
        },
        ReplayKvObservationMode::DropAll,
    );

    assert!(
        reused_tokens(&hybrid) > reused_tokens(&baseline),
        "hybrid={} baseline={}",
        reused_tokens(&hybrid),
        reused_tokens(&baseline)
    );
    assert!(hybrid.latency.ttft.mean_ms < baseline.latency.ttft.mean_ms);
}

#[test]
fn hybrid_beats_kv_only_with_partial_router_kv_observation() {
    let trace = cumulative_trace(64, 8);
    let observation = ReplayKvObservationMode::DeterministicWorkerEventKeepPercent(50);
    let baseline = run_affinity_with_observation(
        trace.clone(),
        ReplaySessionAffinityMode::Disabled,
        observation,
    );
    let hybrid = run_affinity_with_observation(
        trace,
        ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
            max_entries: 65_536,
            max_cost_regret_blocks: 4,
        },
        observation,
    );

    assert!(reused_tokens(&hybrid) > reused_tokens(&baseline));
    assert!(hybrid.latency.ttft.mean_ms < baseline.latency.ttft.mean_ms);
}

#[test]
fn hybrid_cost_gate_beats_kv_only_under_saturated_hrw_collision() {
    let trace = hrw_collision_trace(64, 8);
    let observation = ReplayKvObservationMode::DeterministicWorkerEventKeepPercent(50);
    let baseline = run_affinity_with_args_and_observation(
        replay_args_with_speedup(1.0),
        trace.clone(),
        ReplaySessionAffinityMode::Disabled,
        observation,
    );
    let hybrid = run_affinity_with_args_and_observation(
        replay_args_with_speedup(1.0),
        trace,
        ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
            max_entries: 65_536,
            max_cost_regret_blocks: 16,
        },
        observation,
    );

    assert!(
        reused_tokens(&hybrid) > reused_tokens(&baseline),
        "hybrid_reused={} baseline_reused={}",
        reused_tokens(&hybrid),
        reused_tokens(&baseline),
    );
    assert!(
        hybrid.latency.ttft.mean_ms < baseline.latency.ttft.mean_ms,
        "hybrid_mean={}ms baseline_mean={}ms",
        hybrid.latency.ttft.mean_ms,
        baseline.latency.ttft.mean_ms,
    );
}

#[test]
fn affinity_replay_rejects_topology_mutation_instead_of_stranding_exact_lanes() {
    let mut router = OfflineReplayRouter::new_with_session_affinity(
        &replay_args(),
        Some(KvRouterConfig::default()),
        None,
        4,
        ReplaySessionAffinityMode::Hrw,
    )
    .unwrap();
    let error = router.remove_worker(0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires static worker topology")
    );
}

#[test]
fn deterministic_algorithms_distribute_sessions_across_worker_rank_targets() {
    let candidates = worker_rank_targets(3, 4);
    for mode in [
        ReplaySessionAffinityMode::Modulo,
        ReplaySessionAffinityMode::Hrw,
    ] {
        let (distinct_targets, load_cv) = mapping_target_distribution(mode, 100_000, &candidates);
        assert_eq!(distinct_targets, candidates.len());
        assert!(load_cv < 0.02, "mode={mode:?}, load_cv={load_cv}");
    }
}

#[test]
fn kv_overlap_budget_exposes_cache_regret_versus_dispatch_agreement_tradeoff() {
    let strict = kv_conflict_metrics(ReplaySessionAffinityMode::Hrw, 10_000);
    assert_eq!(strict.home_selection_rate, 1.0);
    assert_eq!(strict.mean_extra_uncached_tokens, 192.0);
    assert_eq!(strict.p95_extra_uncached_tokens, 512);
    assert_eq!(strict.divergent_snapshot_dispatch_agreement, 1.0);

    let expected = [
        (0, 0.2, 0.0, 0, 0.2),
        (64, 0.4, 12.8, 64, 0.4),
        (128, 0.6, 38.4, 128, 0.6),
        (256, 0.8, 89.6, 256, 0.8),
        (512, 1.0, 192.0, 512, 1.0),
    ];
    let mut previous_home_rate = 0.0;
    let mut previous_mean_regret = 0.0;
    let mut previous_agreement = 0.0;
    for (budget, expected_home, expected_mean, expected_p95, expected_agreement) in expected {
        let metrics = kv_conflict_metrics(
            ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: budget,
            },
            10_000,
        );
        assert!((metrics.home_selection_rate - expected_home).abs() < 1.0e-12);
        assert!((metrics.mean_extra_uncached_tokens - expected_mean).abs() < 1.0e-12);
        assert_eq!(metrics.p95_extra_uncached_tokens, expected_p95);
        assert!(metrics.p95_extra_uncached_tokens <= budget);
        assert!(
            (metrics.divergent_snapshot_dispatch_agreement - expected_agreement).abs() < 1.0e-12
        );
        assert!(metrics.home_selection_rate >= previous_home_rate);
        assert!(metrics.mean_extra_uncached_tokens >= previous_mean_regret);
        assert!(metrics.divergent_snapshot_dispatch_agreement >= previous_agreement);
        if budget < 512 {
            assert!(metrics.home_selection_rate < strict.home_selection_rate);
            assert!(
                metrics.divergent_snapshot_dispatch_agreement
                    < strict.divergent_snapshot_dispatch_agreement
            );
        }
        previous_home_rate = metrics.home_selection_rate;
        previous_mean_regret = metrics.mean_extra_uncached_tokens;
        previous_agreement = metrics.divergent_snapshot_dispatch_agreement;
    }
}

#[test]
#[ignore = "prints the reproducible A/B performance table"]
fn session_affinity_ab_report() {
    println!(
        "scenario,algorithm,reused_tokens,effective_prefill_tokens,reuse_ratio,target_transition_rate,worker_load_cv,mean_ttft_ms,p95_ttft_ms,p99_ttft_ms,request_throughput_rps"
    );
    for (scenario, num_sessions) in [("single_hot_session", 1), ("balanced_sessions", 31)] {
        let trace = cumulative_trace(num_sessions, 6);
        let results = [
            ("round_robin", run_round_robin_for(trace.clone())),
            (
                "kv_no_affinity",
                run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Disabled),
            ),
            (
                "local",
                run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Local),
            ),
            (
                "modulo",
                run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Modulo),
            ),
            (
                "hrw",
                run_affinity_for(trace.clone(), ReplaySessionAffinityMode::Hrw),
            ),
        ];
        for (algorithm, report) in results {
            let reused = reused_tokens(&report);
            let effective_prefill = report.request_counts.total_input_tokens - reused;
            println!(
                "{scenario},{algorithm},{reused},{effective_prefill},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                report.prefix_cache_reused_ratio,
                target_transition_rate(&report),
                worker_load_cv(&report, 4),
                report.latency.ttft.mean_ms,
                report.latency.ttft.p95_ms,
                report.latency.ttft.p99_ms,
                report.throughput.request_throughput_rps,
            );
        }
    }

    let mapping_sessions = 100_000;
    println!("\nmapping_algorithm,sessions,workers_before,workers_after,remaps,remap_ratio");
    for (algorithm, mode) in [
        ("modulo", ReplaySessionAffinityMode::Modulo),
        ("hrw", ReplaySessionAffinityMode::Hrw),
    ] {
        let remaps = scale_out_remaps(mode, mapping_sessions);
        println!(
            "{algorithm},{mapping_sessions},8,9,{remaps},{:.6}",
            remaps as f64 / mapping_sessions as f64
        );
    }

    let worker_count = 3;
    let ranks_per_worker = 4;
    let candidates = worker_rank_targets(worker_count, ranks_per_worker);
    println!(
        "\nmapping_algorithm,sessions,workers,ranks_per_worker,distinct_worker_rank_targets,worker_rank_load_cv"
    );
    for (algorithm, mode) in [
        ("modulo", ReplaySessionAffinityMode::Modulo),
        ("hrw", ReplaySessionAffinityMode::Hrw),
    ] {
        let (distinct_targets, load_cv) =
            mapping_target_distribution(mode, mapping_sessions, &candidates);
        println!(
            "{algorithm},{mapping_sessions},{worker_count},{ranks_per_worker},{distinct_targets},{load_cv:.6}"
        );
    }

    let conflict_sessions = 100_000;
    println!(
        "\npolicy,sessions,max_extra_uncached_tokens,home_selection_rate,mean_extra_uncached_tokens,p95_extra_uncached_tokens,divergent_snapshot_dispatch_agreement"
    );
    let conflict_policies = [
        ("strict_hrw", None, ReplaySessionAffinityMode::Hrw),
        (
            "kv_overlap_bounded_arrival_pin",
            Some(0),
            ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: 0,
            },
        ),
        (
            "kv_overlap_bounded_arrival_pin",
            Some(64),
            ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: 64,
            },
        ),
        (
            "kv_overlap_bounded_arrival_pin",
            Some(128),
            ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: 128,
            },
        ),
        (
            "kv_overlap_bounded_arrival_pin",
            Some(256),
            ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: 256,
            },
        ),
        (
            "kv_overlap_bounded_arrival_pin",
            Some(512),
            ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: 512,
            },
        ),
    ];
    for (policy, budget, mode) in conflict_policies {
        let metrics = kv_conflict_metrics(mode, conflict_sessions);
        let budget = budget.map_or_else(|| "n/a".to_string(), |value| value.to_string());
        println!(
            "{policy},{conflict_sessions},{budget},{:.6},{:.3},{},{:.6}",
            metrics.home_selection_rate,
            metrics.mean_extra_uncached_tokens,
            metrics.p95_extra_uncached_tokens,
            metrics.divergent_snapshot_dispatch_agreement,
        );
    }
}

#[test]
#[ignore = "prints the reproducible hybrid/KV-only A/B performance sweep"]
fn session_hybrid_ab_report() {
    println!(
        "worker_event_keep_percent,algorithm,cost_regret_budget_blocks,reused_tokens,effective_prefill_tokens,reuse_ratio,target_transition_rate,worker_load_cv,mean_ttft_ms,p95_ttft_ms,p99_ttft_ms,request_throughput_rps"
    );
    let trace = cumulative_trace(64, 8);
    for keep_percent in [100, 75, 50, 25, 0] {
        let observation = if keep_percent == 100 {
            ReplayKvObservationMode::Complete
        } else if keep_percent == 0 {
            ReplayKvObservationMode::DropAll
        } else {
            ReplayKvObservationMode::DeterministicWorkerEventKeepPercent(keep_percent)
        };
        let policies = [
            ("kv_only", None, ReplaySessionAffinityMode::Disabled),
            ("strict_local", None, ReplaySessionAffinityMode::Local),
            ("strict_hrw", None, ReplaySessionAffinityMode::Hrw),
            (
                "lru_hrw_kv_cost",
                Some(0),
                ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                    max_entries: 65_536,
                    max_cost_regret_blocks: 0,
                },
            ),
            (
                "lru_hrw_kv_cost",
                Some(4),
                ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                    max_entries: 65_536,
                    max_cost_regret_blocks: 4,
                },
            ),
            (
                "lru_hrw_kv_cost",
                Some(8),
                ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                    max_entries: 65_536,
                    max_cost_regret_blocks: 8,
                },
            ),
            (
                "lru_hrw_kv_cost",
                Some(16),
                ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                    max_entries: 65_536,
                    max_cost_regret_blocks: 16,
                },
            ),
        ];
        for (algorithm, budget, mode) in policies {
            let report = run_affinity_with_observation(trace.clone(), mode, observation);
            let reused = reused_tokens(&report);
            let effective_prefill = report.request_counts.total_input_tokens - reused;
            let budget = budget.map_or_else(|| "n/a".to_string(), |value| value.to_string());
            println!(
                "{keep_percent},{algorithm},{budget},{reused},{effective_prefill},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                report.prefix_cache_reused_ratio,
                target_transition_rate(&report),
                worker_load_cv(&report, 4),
                report.latency.ttft.mean_ms,
                report.latency.ttft.p95_ms,
                report.latency.ttft.p99_ms,
                report.throughput.request_throughput_rps,
            );
        }
    }

    println!(
        "\nload_scenario,algorithm,cost_regret_budget_blocks,reused_tokens,effective_prefill_tokens,reuse_ratio,target_transition_rate,worker_load_cv,mean_ttft_ms,p95_ttft_ms,p99_ttft_ms,request_throughput_rps"
    );
    let load_scenarios = [
        ("saturated_balanced", cumulative_trace(64, 8)),
        ("saturated_hrw_collision", hrw_collision_trace(64, 8)),
    ];
    let observation = ReplayKvObservationMode::DeterministicWorkerEventKeepPercent(50);
    let load_policies = [
        ("kv_only", None, ReplaySessionAffinityMode::Disabled),
        ("strict_local", None, ReplaySessionAffinityMode::Local),
        ("strict_hrw", None, ReplaySessionAffinityMode::Hrw),
        (
            "lru_hrw_kv_cost",
            Some(0),
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                max_entries: 65_536,
                max_cost_regret_blocks: 0,
            },
        ),
        (
            "lru_hrw_kv_cost",
            Some(4),
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                max_entries: 65_536,
                max_cost_regret_blocks: 4,
            },
        ),
        (
            "lru_hrw_kv_cost",
            Some(8),
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                max_entries: 65_536,
                max_cost_regret_blocks: 8,
            },
        ),
        (
            "lru_hrw_kv_cost",
            Some(16),
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                max_entries: 65_536,
                max_cost_regret_blocks: 16,
            },
        ),
    ];
    for (scenario, load_trace) in load_scenarios {
        for (algorithm, budget, mode) in load_policies {
            let report = run_affinity_with_args_and_observation(
                replay_args_with_speedup(1.0),
                load_trace.clone(),
                mode,
                observation,
            );
            let reused = reused_tokens(&report);
            let effective_prefill = report.request_counts.total_input_tokens - reused;
            let budget = budget.map_or_else(|| "n/a".to_string(), |value| value.to_string());
            println!(
                "{scenario}_50pct_visibility,{algorithm},{budget},{reused},{effective_prefill},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                report.prefix_cache_reused_ratio,
                target_transition_rate(&report),
                worker_load_cv(&report, 4),
                report.latency.ttft.mean_ms,
                report.latency.ttft.p95_ms,
                report.latency.ttft.p99_ms,
                report.throughput.request_throughput_rps,
            );
        }
    }
}

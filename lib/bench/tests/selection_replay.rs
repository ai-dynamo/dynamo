// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[path = "../kv_router/selection_replay/corpus.rs"]
mod corpus;
#[allow(dead_code)]
#[path = "../kv_router/selection_replay/engine.rs"]
mod engine;
#[allow(dead_code)]
#[path = "../kv_router/selection_replay/host.rs"]
mod host;
#[allow(dead_code)]
#[path = "../kv_router/selection_replay/kv_cache.rs"]
mod kv_cache;

use std::path::Path;
use std::sync::Arc;

use corpus::{Corpus, TraceFormat, WorkloadParams};
use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use engine::{Action, Lifecycle, Limits, Mode, RunConfig};
use host::{BenchHost, HostConfig, router_config, worker_request};
use kv_cache::{KvCaches, decode_block_hashes};

/// Slots a test request holds in a worker's cache.
type Held = Vec<u32>;

const MOONCAKE_TRACE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/mooncake_trace_1000.jsonl"
);

fn params(trace_block_size: usize, lanes: usize) -> WorkloadParams {
    WorkloadParams {
        trace_format: TraceFormat::Mooncake,
        trace_block_size: Some(trace_block_size),
        lanes,
        workers: 8,
        dp_size: 1,
        block_size: 16,
        ignore_trace_delays: false,
        warmup_ms: 1_000,
        window_ms: 4_000,
        // Small enough that the workers' caches evict during the window.
        num_gpu_blocks: 2_048,
        max_num_batched_tokens: 8_192,
        max_num_seqs: 64,
        engine_speedup: 10.0,
    }
}

fn replay(corpus: &Arc<Corpus>, mode: Mode, queue_threshold: Option<f64>) -> engine::RunResult {
    let params = &corpus.params;
    let workers: Vec<_> = (1..=params.workers as u64)
        .map(|id| worker_request(id, params.block_size, params.dp_size, 8_192, 2_048))
        .collect();
    let host_config = HostConfig {
        router: KvRouterConfig {
            router_replica_sync: true,
            router_queue_threshold: queue_threshold,
            ..KvRouterConfig::default()
        },
        block_size: params.block_size,
        busy: Default::default(),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let host = Arc::new(
        runtime
            .block_on(BenchHost::start(&host_config, &workers))
            .unwrap(),
    );
    let config = RunConfig {
        mode,
        // Also the slack saturation gives engine actions; keep it small.
        limits: Limits {
            engine_lag_us: 1_000,
            ..Limits::default()
        },
        issuer_cpus: Vec::new(),
        engine_issuers: 2,
        prep_threads: 1,
        perf_ctl_fifo: None,
        perf_ack_fifo: None,
    };
    let result = engine::run(corpus, &host, &runtime, &config).unwrap();
    runtime.block_on(host.shutdown());
    result
}

#[test]
fn replays_every_request_through_the_frontend_lifecycle() {
    let corpus =
        Arc::new(corpus::generate(Path::new(MOONCAKE_TRACE), &params(512, 32), 4).unwrap());
    assert!(!corpus.requests.is_empty());
    // Every window request finished inside the mocker's run, so none is missing.
    assert_eq!(corpus.omitted.unfinished, 0, "{:?}", corpus.omitted);
    for mode in [
        Mode::Fixed { scale: 20.0 },
        Mode::Saturation {
            admission_window: 32,
        },
    ] {
        let result = replay(&corpus, mode, None);
        assert_eq!(result.failed, 0, "{:?}", result.failure_samples);
        assert!(result.window_requests > 0);
        assert_eq!(result.admitted, result.window_requests);
        // Requests release as they complete: at the end each closed-loop lane holds
        // at most its one running request.
        assert!(
            result.booked_at_end <= corpus.lanes.len() as u64,
            "{mode:?}"
        );
        // Booked workers cache and publish what they run, the router then finds it,
        // and full caches evict.
        assert!(result.cache.stored_blocks > 0 && result.cache.decode_blocks > 0);
        assert!(result.mean_predicted_cached_blocks > 0.0, "{mode:?}");
        assert!(result.cache.evicted_blocks > 0, "{:?}", result.cache);
        // Every stored parent and removed block was where the index expected it.
        for outcome in ["stored/parent_block_not_found", "removed/block_not_found"] {
            assert_eq!(
                result
                    .index_events_applied
                    .get(outcome)
                    .copied()
                    .unwrap_or(0),
                0,
                "{outcome} in {mode:?}"
            );
        }
    }
}

#[test]
fn saturation_runs_on_while_the_router_queues() {
    // Queue whenever every rank has prefill in flight, with few admission slots, so
    // queued requests hold every slot while they wait on engine actions.
    let corpus =
        Arc::new(corpus::generate(Path::new(MOONCAKE_TRACE), &params(512, 32), 4).unwrap());
    let result = replay(
        &corpus,
        Mode::Saturation {
            admission_window: 4,
        },
        Some(0.0),
    );
    assert_eq!(result.failed, 0, "{:?}", result.failure_samples);
    assert_eq!(result.admitted + result.rejected, result.window_requests);
}

/// Three three-turn sessions, each turn extending the last, in a temporary Mooncake
/// file whose timestamps start at `start_ms`.
fn multi_turn_trace(name: &str, start_ms: u64) -> std::path::PathBuf {
    let trace = std::env::temp_dir().join(format!(
        "selection-replay-{name}-{}.jsonl",
        std::process::id()
    ));
    let mut rows = String::new();
    for session in 0..3 {
        for turn in 0..3u64 {
            let hash_ids: Vec<u64> = (0..=turn + 1).map(|id| session * 100 + id).collect();
            let timing = if turn == 0 {
                format!("\"timestamp\": {}", start_ms + session * 10)
            } else {
                "\"delay\": 5".to_string()
            };
            rows.push_str(&format!(
                "{{\"session_id\": \"s{session}\", {timing}, \"input_length\": {}, \
                 \"output_length\": 8, \"hash_ids\": {hash_ids:?}}}\n",
                hash_ids.len() * 32
            ));
        }
    }
    std::fs::write(&trace, rows).unwrap();
    trace
}

#[test]
fn lanes_reusing_a_play_share_no_blocks_and_corpora_round_trip() {
    let trace = multi_turn_trace("lanes", 0);

    // Six lanes over three plays: each play runs twice, on disjoint hash IDs.
    let mut corpus = corpus::generate(&trace, &params(32, 6), 2).unwrap();
    std::fs::remove_file(&trace).unwrap();
    assert_eq!(corpus.lanes.len(), 6);
    let first_copy = corpus
        .requests
        .iter()
        .find(|request| request.lane == 0 && request.turn == 0)
        .unwrap();
    let second_copy = corpus
        .requests
        .iter()
        .find(|request| request.lane == 3 && request.turn == 0)
        .unwrap();
    let first = corpus.prompt_tokens(first_copy).unwrap();
    let second = corpus.prompt_tokens(second_copy).unwrap();
    assert_eq!(first.len(), second.len());
    assert!(first.iter().all(|token| !second.contains(token)));

    let saved =
        std::env::temp_dir().join(format!("selection-replay-{}.corpus", std::process::id()));
    corpus.save(&saved).unwrap();
    let loaded = Corpus::load(&saved).unwrap();
    assert_eq!(loaded.digest, corpus.digest);
    assert_eq!(loaded.requests.len(), corpus.requests.len());

    // The digest comes from what was loaded, so a changed corpus can't keep the old one.
    corpus.requests[0].arrival_us += 1;
    corpus.save(&saved).unwrap();
    let changed = Corpus::load(&saved).unwrap();
    std::fs::remove_file(&saved).unwrap();
    assert_ne!(changed.digest, loaded.digest);
}

#[test]
fn mooncake_arrivals_are_relative_to_the_trace_start() {
    let generate = |start_ms| {
        let trace = multi_turn_trace(&format!("start-{start_ms}"), start_ms);
        let corpus = corpus::generate(&trace, &params(32, 3), 2).unwrap();
        std::fs::remove_file(&trace).unwrap();
        corpus
    };
    let from_zero = generate(0);
    assert!(!from_zero.requests.is_empty());
    assert_eq!(generate(1_000_000).digest, from_zero.digest);
}

#[test]
fn a_late_booking_applies_the_actions_already_due_in_order() {
    let mut lifecycle = Lifecycle::default();
    assert!(!lifecycle.due(Action::FirstOutput));
    assert!(!lifecycle.due(Action::OutputBlock(0)));
    assert!(!lifecycle.due(Action::OutputBlock(1)));
    assert_eq!(
        lifecycle.book(),
        [
            Action::FirstOutput,
            Action::OutputBlock(0),
            Action::OutputBlock(1)
        ]
    );
    // Once booked, each action applies as it comes due.
    assert!(lifecycle.due(Action::OutputBlock(2)));
    assert!(lifecycle.due(Action::Completion));

    // Booked after it completed: the whole lifecycle, completion last.
    let mut lifecycle = Lifecycle::default();
    for action in [
        Action::FirstOutput,
        Action::OutputBlock(0),
        Action::Completion,
    ] {
        lifecycle.due(action);
    }
    assert_eq!(
        lifecycle.book(),
        [
            Action::FirstOutput,
            Action::OutputBlock(0),
            Action::Completion
        ]
    );
}

#[test]
fn modes_reject_values_that_cannot_run() {
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Mode::Fixed { scale }.validate().is_err(), "{scale}");
    }
    assert!(Mode::Fixed { scale: 0.5 }.validate().is_ok());
    assert!(
        Mode::Saturation {
            admission_window: 0
        }
        .validate()
        .is_err()
    );
    assert!(
        Mode::Saturation {
            admission_window: 1
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn router_config_overrides_any_field_and_rejects_unknown_ones() {
    // `decode_active_request_weight` is left out of the serialized defaults.
    let config = router_config(Some(
        r#"{"decode_active_request_weight": 0.5, "router_queue_threshold": 0.1}"#,
    ))
    .unwrap();
    assert_eq!(config.decode_active_request_weight, 0.5);
    assert_eq!(config.router_queue_threshold, Some(0.1));
    assert!(config.router_replica_sync);
    assert!(router_config(Some(r#"{"no_such_field": 1}"#)).is_err());
    assert!(router_config(Some("[]")).is_err());
}

fn prompt(blocks: &[u64]) -> kv_cache::PromptBlocks {
    kv_cache::PromptBlocks {
        local: blocks.iter().map(|b| b + 1000).collect(),
        sequence: blocks.to_vec(),
    }
}

fn stored(event: &KvCacheEvent) -> (Option<u64>, Vec<u64>) {
    let KvCacheEventData::Stored(data) = &event.data else {
        panic!("expected a stored event, got {event:?}");
    };
    (
        data.parent_hash.map(|hash| hash.0),
        data.blocks.iter().map(|block| block.block_hash.0).collect(),
    )
}

fn removed(event: &KvCacheEvent) -> Vec<u64> {
    let KvCacheEventData::Removed(data) = &event.data else {
        panic!("expected a removed event, got {event:?}");
    };
    data.block_hashes.iter().map(|hash| hash.0).collect()
}

/// A prefill on `cache`, with the events it published, the blocks it found cached,
/// and the slots the request now holds.
fn prefill(caches: &KvCaches, cache: usize, blocks: &[u64]) -> (Vec<KvCacheEvent>, usize, Held) {
    let mut published = Vec::new();
    let mut held = Held::new();
    let cached = caches.prefill(cache, 0, &prompt(blocks), &mut held, |events| {
        published = events
    });
    (published, cached, held)
}

#[test]
fn a_cached_prefix_is_reused_and_only_the_suffix_stored() {
    let caches = KvCaches::new(1, 1, 16, 1);
    let (events, cached, held) = prefill(&caches, 0, &[1, 2]);
    assert_eq!(cached, 0);
    assert_eq!(stored(&events[0]), (None, vec![1, 2]));
    caches.release(0, &held);

    let (events, cached, held) = prefill(&caches, 0, &[1, 2, 3, 4]);
    assert_eq!(cached, 2);
    assert_eq!(held.len(), 4);
    assert_eq!(events.len(), 1);
    assert_eq!(stored(&events[0]), (Some(2), vec![3, 4]));
}

#[test]
fn released_tails_are_evicted_first_and_pinned_blocks_never() {
    let caches = KvCaches::new(1, 1, 3, 1);
    let (_, _, held) = prefill(&caches, 0, &[1, 2, 3]);
    caches.release(0, &held);
    // A running request pins block 1; the full cache must evict 3, then 2.
    prefill(&caches, 0, &[1]);
    let (events, _, _) = prefill(&caches, 0, &[7, 8]);
    assert_eq!(removed(&events[0]), vec![3, 2]);
    assert_eq!(stored(&events[1]), (None, vec![7, 8]));
    // Everything left is pinned: the next block goes over capacity.
    let (events, _, _) = prefill(&caches, 0, &[9]);
    assert_eq!(events.len(), 1);
    assert_eq!(caches.stats().over_capacity, 1);
    assert_eq!(caches.stats().evicted_blocks, 2);
}

#[test]
fn output_blocks_chain_after_the_prompt() {
    let caches = KvCaches::new(2, 1, 4, 1);
    let (_, _, mut held) = prefill(&caches, 1, &[5]);
    let mut events = Vec::new();
    let tail = caches.decode_block(1, 0, Some(5), (42, 0), &mut held, |published| {
        events = published
    });
    let (sequence, _) = decode_block_hashes(5, 42, 0);
    assert_eq!(tail, sequence);
    assert_eq!(stored(&events[0]), (Some(5), vec![sequence]));
    // Another request's output after the same prompt is a different block.
    assert_ne!(decode_block_hashes(5, 43, 0).0, sequence);
    assert_eq!(held.len(), 2);
    assert_eq!(caches.index(2, 0), 1);
}

#[test]
fn runs_within_a_trace_block_are_cached_and_evicted_together() {
    // Two blocks per trace block. A prompt ending mid trace block caches a short
    // run; a longer prompt reuses it and stores the rest of that trace block, then
    // whole trace blocks.
    let caches = KvCaches::new(1, 1, 6, 2);
    let (_, _, held) = prefill(&caches, 0, &[1]);
    caches.release(0, &held);
    let (events, cached, held) = prefill(&caches, 0, &[1, 2, 3, 4, 5]);
    assert_eq!(cached, 1);
    assert_eq!(stored(&events[0]), (Some(1), vec![2, 3, 4, 5]));
    // Runs [1], [2], [3, 4], [5].
    assert_eq!(held.len(), 4);
    caches.release(0, &held);

    // A prompt ending inside a cached run still hits every block it shares, and its
    // output chains after its own last block, not the run's.
    let (events, cached, mut held) = prefill(&caches, 0, &[1, 2, 3]);
    assert_eq!((events.len(), cached), (0, 3));
    let mut events = Vec::new();
    caches.decode_block(0, 0, Some(3), (7, 0), &mut held, |published| {
        events = published
    });
    assert_eq!(stored(&events[0]).0, Some(3));
    caches.release(0, &held);

    // Filling the cache evicts the least recently released runs whole, tail first.
    let (events, _, _) = prefill(&caches, 0, &[7, 8, 9, 10]);
    let output = decode_block_hashes(3, 7, 0).0;
    assert_eq!(removed(&events[0]), vec![5, output, 4, 3]);
    assert_eq!(stored(&events[1]), (None, vec![7, 8, 9, 10]));
}

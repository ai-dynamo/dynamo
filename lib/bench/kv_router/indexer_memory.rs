// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Memory harness for `SyncIndexer` backends: heap bytes per (rank, block) membership,
//! sampled uniformly across a Mooncake replay instead of once after a sweep.
//!
//! A counting global allocator tracks live requested bytes. The corpus is prepared like
//! `mooncake_bench` (each worker timeline normalized to the replay window, then merged in
//! time order), the baseline is taken before the backend is built, and the replay feeds
//! every event through a `ThreadPoolIndexer` (`--event-workers` lanes) while the replay
//! thread issues the trace's lookups. At each of `--samples` evenly spaced points the
//! harness flushes the lanes and records the backend's bytes against the memberships a
//! set-semantics model of the same events holds at that point. Event and lookup payloads
//! are freed by then, so the delta is the backend's own state, including any garbage it
//! has not reclaimed. After the last sample it runs the backend's cleanup task once more
//! and records an after-sweep point.
//!
//! The headline is the time-averaged bytes per membership: the mean over samples of
//! backend bytes divided by memberships. An unpaced replay never reaches wall-clock
//! maintenance such as CRTC's five-minute stale-leaf sweep, which gives the no-sweep
//! bound. `--sweeps N` runs the backend's cleanup task N times evenly across the replay
//! (12 is CRTC's cadence over the 60-minute trace in real time), and `--pace-ms` stretches
//! the replay over that many wall-clock milliseconds instead.
//!
//! Plugging in a backend: add an arm to `run_backend`.
//!
//! ```text
//! cargo bench -p dynamo-bench --no-default-features --features mooncake --bench indexer_memory -- \
//!   /path/to/mooncake_trace.jsonl --num-unique-inference-workers 128 \
//!   --trace-duplication-factor 20 --trace-length-factor 4 --backend crtc \
//!   --result-json-output memory_crtc.json
//! # 16 workers with 16-token pages:
//! cargo bench ... -- /path/to/mooncake_trace.jsonl --num-unique-inference-workers 16 --block-size 16
//! ```
//!
//! Build with `--features mooncake,indexer-memory-system-alloc` to count over the system
//! allocator instead of jemalloc. Counted bytes do not depend on the allocator; RSS does.

#[path = "crtc_reclaim_args.rs"]
mod crtc_reclaim_args;

use std::alloc::{GlobalAlloc, Layout};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use crtc_reclaim_args::CrtcReclaimArgs;
use dynamo_bench::kv_router_common::args::CommonArgs;
use dynamo_bench::kv_router_common::replay::{WorkerReplayArtifacts, generate_replay_artifacts};
use dynamo_kv_router::indexer::SyncIndexer;
use dynamo_kv_router::protocols::{KvCacheEventData, LocalBlockHash, RouterEvent, WorkerId};
use dynamo_kv_router::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer};
use serde::Serialize;

#[cfg(not(feature = "indexer-memory-system-alloc"))]
type Inner = tikv_jemallocator::Jemalloc;
#[cfg(not(feature = "indexer-memory-system-alloc"))]
const INNER: Inner = tikv_jemallocator::Jemalloc;
#[cfg(not(feature = "indexer-memory-system-alloc"))]
const ALLOCATOR: &str = "jemalloc";

#[cfg(feature = "indexer-memory-system-alloc")]
type Inner = std::alloc::System;
#[cfg(feature = "indexer-memory-system-alloc")]
const INNER: Inner = std::alloc::System;
#[cfg(feature = "indexer-memory-system-alloc")]
const ALLOCATOR: &str = "system";

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_BYTES: AtomicIsize = AtomicIsize::new(0);

/// Counts live requested bytes over the inner allocator.
struct Counting;

impl Counting {
    fn grow(bytes: usize) {
        let live = LIVE_BYTES.fetch_add(bytes as isize, Ordering::Relaxed) + bytes as isize;
        PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
    }

    fn shrink(bytes: usize) {
        LIVE_BYTES.fetch_sub(bytes as isize, Ordering::Relaxed);
    }
}

// SAFETY: every call forwards to the inner allocator with the caller's layout and
// pointer; the counters only observe sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded as received.
        let ptr = unsafe { INNER.alloc(layout) };
        if !ptr.is_null() {
            Self::grow(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded as received.
        let ptr = unsafe { INNER.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded as received.
        unsafe { INNER.dealloc(ptr, layout) };
        Self::shrink(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded as received.
        let new_ptr = unsafe { INNER.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            Self::shrink(layout.size());
            Self::grow(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn live_bytes() -> isize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Resident set size in bytes, from `/proc/self/statm`.
fn rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

/// Sequence hashes are already uniformly distributed.
#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<IdentityHasher>>;
type FastSet<K> = HashSet<K, BuildHasherDefault<IdentityHasher>>;

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Backend to measure: crtc.
    #[clap(long, default_value = "crtc")]
    backend: String,

    /// Event lanes of the `ThreadPoolIndexer`.
    #[clap(long, default_value = "8")]
    event_workers: usize,

    /// Evenly spaced measurement points across the replay.
    #[clap(long, default_value = "100")]
    samples: usize,

    /// Skip the trace's lookups; replay events only.
    #[clap(long)]
    skip_queries: bool,

    /// Stretch the replay over this many wall-clock milliseconds (0: unpaced).
    #[clap(long, default_value = "0")]
    pace_ms: u64,

    /// Also run the backend's cleanup task this many times, evenly spaced in replay time,
    /// right after the sample at that point. `12` matches CRTC's five-minute sweep over
    /// the 60-minute Mooncake trace replayed in real time.
    #[clap(long, default_value = "0")]
    sweeps: usize,

    /// JSON output path.
    #[clap(long, default_value = "indexer_memory.json")]
    result_json_output: String,

    #[clap(flatten)]
    crtc_reclaim: CrtcReclaimArgs,
}

enum Entry {
    Query(Vec<LocalBlockHash>),
    Event(RouterEvent),
}

/// The merged replay: entries in time order, with their position in `[0, 1]`.
struct Corpus {
    entries: Vec<(f64, Entry)>,
    /// `(memberships, unique blocks)` the event stream holds at each sample point.
    held: Vec<(u64, u64)>,
    skipped_non_gpu_events: u64,
}

/// Normalizes each worker timeline to `[0, 1]` and merges them in time order, keeping
/// each worker's own order.
fn build_corpus(
    artifacts: Vec<WorkerReplayArtifacts>,
    duplication: usize,
    samples: usize,
) -> Corpus {
    let num_trace_workers = artifacts.len();
    let mut skipped_non_gpu_events = 0;
    let mut keyed: Vec<((f64, WorkerId, usize), Entry)> = Vec::new();
    for replica in 0..duplication {
        for (base_worker, artifact) in artifacts.iter().enumerate() {
            let worker_id = (base_worker + replica * num_trace_workers) as WorkerId;
            let mut timeline: Vec<(u64, Entry)> = Vec::new();
            for request in &artifact.requests {
                timeline.push((
                    request.timestamp_us,
                    Entry::Query(
                        request
                            .replay_hashes
                            .local_block_hashes
                            .iter()
                            .copied()
                            .map(LocalBlockHash)
                            .collect(),
                    ),
                ));
            }
            for event in &artifact.kv_events {
                if !event.storage_tier.is_gpu() {
                    skipped_non_gpu_events += 1;
                    continue;
                }
                timeline.push((
                    event.timestamp_us,
                    Entry::Event(RouterEvent::with_storage_tier(
                        worker_id,
                        event.event.clone(),
                        event.storage_tier,
                    )),
                ));
            }
            // Queries precede events at equal timestamps, as in `mooncake_bench`.
            timeline.sort_by_key(|(ts, entry)| (*ts, matches!(entry, Entry::Event(_))));
            let Some(first) = timeline.first().map(|(ts, _)| *ts) else {
                continue;
            };
            let span = timeline
                .last()
                .map_or(1, |(ts, _)| ts.saturating_sub(first))
                .max(1) as f64;
            for (ordinal, (ts, entry)) in timeline.into_iter().enumerate() {
                let at = (ts - first) as f64 / span;
                keyed.push(((at, worker_id, ordinal), entry));
            }
        }
    }
    keyed.sort_by(|(a, _), (b, _)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let entries: Vec<(f64, Entry)> = keyed
        .into_iter()
        .map(|((at, _, _), entry)| (at, entry))
        .collect();

    // Memberships under set semantics at each sample point.
    let mut ranks: FastMap<(WorkerId, u32), FastSet<u64>> = FastMap::default();
    let mut owners: FastMap<u64, u32> = FastMap::default();
    let mut memberships = 0u64;
    let mut held = Vec::with_capacity(samples);
    let mut next = 0usize;
    let points = sample_points(samples);
    for (at, entry) in &entries {
        while next < points.len() && *at > points[next] {
            held.push((memberships, owners.len() as u64));
            next += 1;
        }
        let Entry::Event(event) = entry else {
            continue;
        };
        let rank = ranks
            .entry((event.worker_id, event.event.dp_rank))
            .or_default();
        match &event.event.data {
            KvCacheEventData::Stored(store) => {
                for block in &store.blocks {
                    if rank.insert(block.block_hash.0) {
                        memberships += 1;
                        *owners.entry(block.block_hash.0).or_default() += 1;
                    }
                }
            }
            KvCacheEventData::Removed(remove) => {
                for hash in &remove.block_hashes {
                    if rank.remove(&hash.0) {
                        memberships -= 1;
                        drop_owner(&mut owners, hash.0);
                    }
                }
            }
            KvCacheEventData::Cleared => {
                memberships -= rank.len() as u64;
                for hash in rank.drain() {
                    drop_owner(&mut owners, hash);
                }
            }
        }
    }
    while held.len() < points.len() {
        held.push((memberships, owners.len() as u64));
    }
    Corpus {
        entries,
        held,
        skipped_non_gpu_events,
    }
}

fn drop_owner(owners: &mut FastMap<u64, u32>, hash: u64) {
    if let Some(count) = owners.get_mut(&hash) {
        *count -= 1;
        if *count == 0 {
            owners.remove(&hash);
        }
    }
}

/// Sample positions in `(0, 1]`.
fn sample_points(samples: usize) -> Vec<f64> {
    (1..=samples).map(|k| k as f64 / samples as f64).collect()
}

#[derive(Serialize, Clone)]
struct Sample {
    at: f64,
    events: u64,
    queries: u64,
    memberships: u64,
    unique_blocks: u64,
    backend_bytes: i64,
    bytes_per_membership: f64,
    bytes_per_unique_block: f64,
    rss_bytes: Option<u64>,
}

#[derive(Serialize)]
struct Report {
    backend: String,
    allocator: &'static str,
    trace: Option<String>,
    num_unique_inference_workers: usize,
    inference_worker_duplication_factor: usize,
    trace_duplication_factor: usize,
    trace_length_factor: usize,
    block_size: u32,
    num_gpu_blocks: usize,
    event_workers: usize,
    queries: bool,
    pace_ms: u64,
    sweeps: usize,
    entries: usize,
    skipped_non_gpu_events: u64,
    /// Mean over samples of backend bytes / memberships.
    time_avg_bytes_per_membership: f64,
    /// Mean backend bytes over mean memberships.
    mean_bytes_over_mean_memberships: f64,
    time_avg_bytes_per_unique_block: f64,
    max_bytes_per_membership: f64,
    final_bytes_per_membership: f64,
    after_sweep_bytes_per_membership: f64,
    peak_backend_bytes: i64,
    mean_memberships: f64,
    replay_secs: f64,
    samples: Vec<Sample>,
    after_sweep: Sample,
    /// Backend settings and its structural and memory probes after the final sweep.
    backend_config: Option<String>,
    backend_probe: Option<String>,
    backend_timing_report: String,
}

fn ratio(bytes: i64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        bytes as f64 / count as f64
    }
}

fn replay<T: SyncIndexer>(
    args: &Args,
    corpus: Corpus,
    make: impl FnOnce() -> T,
    probe: impl FnOnce(&T) -> String,
) -> anyhow::Result<Report> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let points = sample_points(args.samples);
    let Corpus {
        entries,
        held,
        skipped_non_gpu_events,
    } = corpus;
    let entry_count = entries.len();

    let baseline = live_bytes();
    PEAK_BYTES.store(baseline, Ordering::Relaxed);
    let indexer = ThreadPoolIndexer::new(make(), args.event_workers, args.common.block_size);
    let sample = |at: f64, events: u64, queries: u64, (memberships, unique): (u64, u64)| {
        let backend_bytes = (live_bytes() - baseline) as i64;
        Sample {
            at,
            events,
            queries,
            memberships,
            unique_blocks: unique,
            backend_bytes,
            bytes_per_membership: ratio(backend_bytes, memberships),
            bytes_per_unique_block: ratio(backend_bytes, unique),
            rss_bytes: rss_bytes(),
        }
    };

    let started = Instant::now();
    let mut samples = Vec::with_capacity(points.len());
    let mut events = 0u64;
    let mut queries = 0u64;
    let mut next = 0usize;
    for (at, entry) in &entries {
        while next < points.len() && *at > points[next] {
            runtime.block_on(indexer.flush());
            samples.push(sample(points[next], events, queries, held[next]));
            next += 1;
            if next * args.sweeps / points.len() > (next - 1) * args.sweeps / points.len() {
                indexer.backend().run_cleanup_task();
                runtime.block_on(indexer.flush());
            }
            if args.pace_ms > 0 {
                let due = Duration::from_secs_f64(points[next - 1] * args.pace_ms as f64 / 1e3);
                if let Some(wait) = due.checked_sub(started.elapsed()) {
                    std::thread::sleep(wait);
                }
            }
        }
        match entry {
            Entry::Event(event) => {
                indexer.enqueue_event(event.clone())?;
                events += 1;
            }
            Entry::Query(hashes) if !args.skip_queries => {
                std::hint::black_box(indexer.backend().find_matches(hashes, false));
                queries += 1;
            }
            Entry::Query(_) => {}
        }
    }
    while next < points.len() {
        runtime.block_on(indexer.flush());
        samples.push(sample(points[next], events, queries, held[next]));
        next += 1;
    }
    let replay_secs = started.elapsed().as_secs_f64();

    indexer.backend().run_cleanup_task();
    runtime.block_on(indexer.flush());
    let after_sweep = sample(1.0, events, queries, *held.last().unwrap_or(&(0, 0)));
    let backend_probe = probe(indexer.backend());
    let backend_timing_report = indexer.backend().timing_report();
    let peak_backend_bytes = (PEAK_BYTES.load(Ordering::Relaxed) - baseline) as i64;

    let measured: Vec<&Sample> = samples.iter().filter(|s| s.memberships > 0).collect();
    let n = measured.len().max(1) as f64;
    let mean = |f: &dyn Fn(&Sample) -> f64| measured.iter().map(|s| f(s)).sum::<f64>() / n;
    let mean_bytes = mean(&|s| s.backend_bytes as f64);
    let mean_memberships = mean(&|s| s.memberships as f64);
    let report = Report {
        backend: args.backend.clone(),
        allocator: ALLOCATOR,
        trace: args.common.mooncake_trace_path.clone(),
        num_unique_inference_workers: args.common.num_unique_inference_workers,
        inference_worker_duplication_factor: args.common.inference_worker_duplication_factor,
        trace_duplication_factor: args.common.trace_duplication_factor,
        trace_length_factor: args.common.trace_length_factor,
        block_size: args.common.block_size,
        num_gpu_blocks: args.common.num_gpu_blocks,
        event_workers: args.event_workers,
        queries: !args.skip_queries,
        pace_ms: args.pace_ms,
        sweeps: args.sweeps,
        entries: entry_count,
        skipped_non_gpu_events,
        time_avg_bytes_per_membership: mean(&|s| s.bytes_per_membership),
        mean_bytes_over_mean_memberships: mean_bytes / mean_memberships.max(1.0),
        time_avg_bytes_per_unique_block: mean(&|s| s.bytes_per_unique_block),
        max_bytes_per_membership: measured
            .iter()
            .map(|s| s.bytes_per_membership)
            .fold(0.0, f64::max),
        final_bytes_per_membership: samples.last().map_or(0.0, |s| s.bytes_per_membership),
        after_sweep_bytes_per_membership: after_sweep.bytes_per_membership,
        peak_backend_bytes,
        mean_memberships,
        replay_secs,
        samples,
        after_sweep,
        backend_config: None,
        backend_probe: Some(backend_probe),
        backend_timing_report,
    };
    drop(indexer);
    drop(entries);
    Ok(report)
}

fn run_backend(args: &Args, corpus: Corpus) -> anyhow::Result<Report> {
    match args.backend.as_str() {
        "crtc" | "concurrent-radix-tree-compressed" => {
            let config = args.crtc_reclaim.config();
            let mut report = replay(
                args,
                corpus,
                || ConcurrentRadixTreeCompressed::with_reclaim_config(config),
                |tree| {
                    let memory = tree.probe_memory();
                    format!(
                        "edge_slack={:.3} {memory:?} {:?}",
                        memory.edge_slack(),
                        tree.probe_shape()
                    )
                },
            )?;
            report.backend_config = Some(format!("{config:?}"));
            Ok(report)
        }
        other => anyhow::bail!("unknown backend {other:?}; add an arm to run_backend"),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.samples > 0, "--samples must be at least 1");
    anyhow::ensure!(args.event_workers > 0, "--event-workers must be at least 1");
    let Some(path) = args.common.mooncake_trace_path.clone() else {
        anyhow::bail!("a Mooncake trace path is required");
    };

    let corpus = {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let traces = args.common.load_mooncake_trace(&path)?;
        let artifacts = runtime.block_on(generate_replay_artifacts(
            &traces,
            args.common.num_gpu_blocks,
            args.common.block_size,
            args.common.trace_simulation_duration_ms,
        ))?;
        drop(traces);
        build_corpus(
            artifacts,
            args.common.inference_worker_duplication_factor,
            args.samples,
        )
    };
    // Let preparation threads exit and free their caches before the baseline.
    std::thread::sleep(Duration::from_millis(500));

    let report = run_backend(&args, corpus)?;
    println!(
        "MEMORY backend={} allocator={} workers={}x{} trace_dup={} length={} block_size={} \
         event_workers={} queries={} pace_ms={} sweeps={} time_avg_B_per_membership={:.1} \
         mean_B_over_mean_memberships={:.1} max_B_per_membership={:.1} \
         final_B_per_membership={:.1} after_sweep_B_per_membership={:.1} \
         time_avg_B_per_unique_block={:.1} peak_backend_MiB={:.1} mean_memberships={:.0} \
         replay_secs={:.1}",
        report.backend,
        report.allocator,
        report.num_unique_inference_workers,
        report.inference_worker_duplication_factor,
        report.trace_duplication_factor,
        report.trace_length_factor,
        report.block_size,
        report.event_workers,
        report.queries,
        report.pace_ms,
        report.sweeps,
        report.time_avg_bytes_per_membership,
        report.mean_bytes_over_mean_memberships,
        report.max_bytes_per_membership,
        report.final_bytes_per_membership,
        report.after_sweep_bytes_per_membership,
        report.time_avg_bytes_per_unique_block,
        report.peak_backend_bytes as f64 / (1 << 20) as f64,
        report.mean_memberships,
        report.replay_secs,
    );
    std::fs::write(
        &args.result_json_output,
        serde_json::to_string_pretty(&report)?,
    )?;
    if let Some(config) = &report.backend_config {
        println!("backend_config: {config}");
    }
    if let Some(probe) = &report.backend_probe {
        println!("backend_probe: {probe}");
    }
    if !report.backend_timing_report.is_empty() {
        println!("{}", report.backend_timing_report);
    }
    println!("Memory result written to {}", args.result_json_output);
    Ok(())
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: phantom KV-event publishers.
//!
//! Hosts `--count` phantom workers. Each binds its own direct-ZMQ PUB socket and replays its
//! base's captured engine events (salted per phantom) in the production wire format: the
//! publisher pipeline's envelopes (`OfflinePublisherPipeline`), msgpack `Vec<RouterEvent>`
//! payloads in an `EventEnvelope`, and the 4-frame ZMQ multipart of `ZmqPubTransport`. The
//! warm-up prefix is paced by block rate; the timed section is paced open loop on the shared
//! wall-clock mapping. The serving indexer accepts the phantoms through
//! `DYN_EXPERIMENT_STATIC_KV_SOURCES` (see `--sources-out`).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use dynamo_e2e_indexer_tools::pipeline::{ProcessedBase, envelope_events, process_base};
use dynamo_e2e_indexer_tools::plan::{
    DEFAULT_WORKER_ID_BASE, PhantomLayout, TimeMap, natural_rates, resolve_speedup, unix_now_us,
};
use dynamo_e2e_indexer_tools::stats::{Counter, Latency, summarize_ms};
use dynamo_e2e_indexer_tools::stream::{EventLists, Manifest, Sections, read_base};
use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::transports::event_plane::{Codec, EventTransportTx, ZmqPubTransport};
use serde_json::json;
use tokio::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "EXPERIMENT ONLY: replay phantom KV event streams over direct ZMQ")]
struct Args {
    /// Phantom-stream directory (manifest.json + base files).
    #[arg(long)]
    streams: PathBuf,
    /// Phantoms across every publisher process (fixes the base, salt, and ID of each phantom).
    #[arg(long)]
    total_phantoms: u64,
    /// First global phantom index hosted here.
    #[arg(long, default_value_t = 0)]
    first_phantom: u64,
    /// Phantoms hosted here (one PUB socket each; keep below libzmq's 1023-socket limit).
    #[arg(long)]
    count: u64,
    #[arg(long, default_value_t = DEFAULT_WORKER_ID_BASE, value_parser = parse_u64)]
    worker_id_base: u64,
    #[arg(long, default_value_t = 0)]
    salt_seed: u64,
    #[arg(long, default_value = "0.0.0.0")]
    bind_host: String,
    /// Host the indexer connects to (written to --sources-out).
    #[arg(long)]
    advertise_host: String,
    /// Phantom `first_phantom + k` binds `base_port + k`.
    #[arg(long)]
    base_port: u16,
    #[arg(long)]
    speedup: Option<f64>,
    /// Aggregate timed write blocks/s (stored + removed) over all --total-phantoms.
    #[arg(long)]
    target_write_blocks_per_sec: Option<f64>,
    /// Warm-up write blocks/s for this process; 0 skips the warm-up (smoke tests only: the
    /// timed section then references blocks the indexer never saw).
    #[arg(long, default_value_t = 2.0e6)]
    warmup_blocks_per_sec: f64,
    /// Seconds between binding and the first warm-up send, so the indexer's SUB sockets have
    /// connected (a PUB socket drops messages sent before a subscriber joins).
    #[arg(long, default_value_t = 10.0)]
    warmup_delay_s: f64,
    /// Unix ms at which the timed section starts (shared with every publisher and driver).
    #[arg(long)]
    start_at_unix_ms: u64,
    /// Per-phantom timed-phase start delay, uniform in [0, spread).
    #[arg(long, default_value_t = 0)]
    start_spread_ms: u64,
    /// Stop the timed phase this long after --start-at-unix-ms (default: when exhausted).
    #[arg(long)]
    duration_s: Option<f64>,
    #[arg(long, default_value_t = 5.0)]
    report_interval_s: f64,
    /// Write the DYN_EXPERIMENT_STATIC_KV_SOURCES entry for this process here.
    #[arg(long)]
    sources_out: Option<PathBuf>,
    #[arg(long)]
    summary_out: Option<PathBuf>,
    /// Bases loaded and processed in parallel.
    #[arg(long, default_value_t = 8)]
    load_parallelism: usize,
}

fn parse_u64(value: &str) -> Result<u64, String> {
    match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|error| error.to_string())
}

#[derive(Default)]
struct Section {
    envelopes: Counter,
    events: Counter,
    stored_blocks: Counter,
    removed_blocks: Counter,
    bytes: Counter,
}

impl Section {
    fn record(&self, events: &[RouterEvent], bytes: usize) {
        let (stored, removed) = write_blocks(events);
        self.envelopes.add(1);
        self.events.add(events.len() as u64);
        self.stored_blocks.add(stored);
        self.removed_blocks.add(removed);
        self.bytes.add(bytes as u64);
    }

    fn interval(&self, seconds: f64) -> serde_json::Value {
        let stored = self.stored_blocks.take_interval() as f64;
        let removed = self.removed_blocks.take_interval() as f64;
        json!({
            "envelopes_per_s": self.envelopes.take_interval() as f64 / seconds,
            "events_per_s": self.events.take_interval() as f64 / seconds,
            "write_blocks_per_s": (stored + removed) / seconds,
            "stored_blocks_per_s": stored / seconds,
            "removed_blocks_per_s": removed / seconds,
            "mbytes_per_s": self.bytes.take_interval() as f64 / seconds / 1e6,
        })
    }

    fn totals(&self) -> serde_json::Value {
        json!({
            "envelopes": self.envelopes.total(),
            "events": self.events.total(),
            "stored_blocks": self.stored_blocks.total(),
            "removed_blocks": self.removed_blocks.total(),
            "bytes": self.bytes.total(),
        })
    }
}

#[derive(Default)]
struct Shared {
    warmup: Section,
    timed: Section,
    /// Timed-section send lateness versus the schedule.
    lag: Latency,
    send_errors: Counter,
    warming: AtomicU64,
    timed_running: AtomicU64,
    exhausted: AtomicU64,
    late_warmups: AtomicU64,
    first_timed_send_us: AtomicU64,
    last_timed_send_us: AtomicU64,
    stop: AtomicBool,
}

fn write_blocks(events: &[RouterEvent]) -> (u64, u64) {
    use dynamo_kv_router::protocols::KvCacheEventData;
    events
        .iter()
        .fold((0, 0), |(stored, removed), event| match &event.event.data {
            KvCacheEventData::Stored(store) => (stored + store.blocks.len() as u64, removed),
            KvCacheEventData::Removed(remove) => {
                (stored, removed + remove.block_hashes.len() as u64)
            }
            KvCacheEventData::Cleared => (stored, removed),
        })
}

struct Phantom {
    index: u64,
    worker_id: u64,
    salt_key: u64,
    delay_us: u64,
    base: Arc<ProcessedBase>,
    transport: ZmqPubTransport,
}

struct Pacing {
    map: TimeMap,
    warmup_blocks_per_sec_per_phantom: f64,
    warmup_start: Instant,
    stop_at_unix_us: Option<u64>,
    /// Unix microseconds corresponding to `epoch`, for converting deadlines to `Instant`s.
    epoch_unix_us: u64,
    epoch: Instant,
}

impl Pacing {
    fn instant_at(&self, unix_us: u64) -> Instant {
        self.epoch + Duration::from_micros(unix_us.saturating_sub(self.epoch_unix_us))
    }
}

async fn send(
    phantom: &Phantom,
    sequence: &mut u64,
    events: &[RouterEvent],
    codec: &Codec,
    section: &Section,
    shared: &Shared,
) {
    let result = async {
        let payload = codec.encode_payload(&events)?;
        let envelope = codec.encode_envelope_parts(
            phantom.worker_id,
            *sequence,
            unix_now_us() / 1000,
            KV_EVENT_SUBJECT,
            &payload,
        )?;
        let bytes = envelope.len();
        phantom
            .transport
            .publish(KV_EVENT_SUBJECT, envelope)
            .await?;
        anyhow::Ok(bytes)
    }
    .await;
    *sequence += 1;
    match result {
        Ok(bytes) => section.record(events, bytes),
        Err(error) => {
            if shared.send_errors.total() == 0 {
                tracing::error!(%error, worker_id = phantom.worker_id, "phantom publish failed");
            }
            shared.send_errors.add(1);
        }
    }
}

fn section_events(
    lists: &EventLists,
    list: usize,
    first_id: u64,
    phantom: &Phantom,
) -> Vec<RouterEvent> {
    envelope_events(lists, list, first_id, phantom.worker_id, phantom.salt_key)
}

async fn run_phantom(phantom: Phantom, pacing: Arc<Pacing>, shared: Arc<Shared>) {
    let codec = Codec::default();
    let mut sequence = 0u64;
    let base = phantom.base.clone();

    if pacing.warmup_blocks_per_sec_per_phantom > 0.0 {
        shared.warming.fetch_add(1, Ordering::Relaxed);
        let mut next = pacing.warmup_start;
        let first_id = base.warmup_first_event_id();
        for list in 0..base.warmup.lists() {
            if shared.stop.load(Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep_until(next).await;
            let events = section_events(&base.warmup, list, first_id, &phantom);
            let blocks: u64 = base
                .warmup
                .list_events(list)
                .map(|index| base.warmup.event_blocks(index))
                .sum();
            send(
                &phantom,
                &mut sequence,
                &events,
                &codec,
                &shared.warmup,
                &shared,
            )
            .await;
            next += Duration::from_secs_f64(
                blocks.max(1) as f64 / pacing.warmup_blocks_per_sec_per_phantom,
            );
        }
        shared.warming.fetch_sub(1, Ordering::Relaxed);
        if unix_now_us() > pacing.map.wall_us(pacing.map.t0_us, phantom.delay_us) {
            shared.late_warmups.fetch_add(1, Ordering::Relaxed);
        }
    }

    shared.timed_running.fetch_add(1, Ordering::Relaxed);
    let first_id = base.timed_first_event_id();
    for list in 0..base.timed.lists() {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        let deadline = pacing
            .map
            .wall_us(base.timed.list_ts_us[list], phantom.delay_us);
        if pacing.stop_at_unix_us.is_some_and(|stop| deadline >= stop) {
            break;
        }
        tokio::time::sleep_until(pacing.instant_at(deadline)).await;
        let now = unix_now_us();
        shared
            .lag
            .record(phantom.index as usize, now.saturating_sub(deadline));
        let events = section_events(&base.timed, list, first_id, &phantom);
        send(
            &phantom,
            &mut sequence,
            &events,
            &codec,
            &shared.timed,
            &shared,
        )
        .await;
        shared.first_timed_send_us.fetch_min(now, Ordering::Relaxed);
        shared.last_timed_send_us.fetch_max(now, Ordering::Relaxed);
    }
    shared.timed_running.fetch_sub(1, Ordering::Relaxed);
    if !shared.stop.load(Ordering::Relaxed) {
        shared.exhausted.fetch_add(1, Ordering::Relaxed);
    }
}

async fn load_bases(
    streams: &std::path::Path,
    manifest: &Manifest,
    bases: Vec<usize>,
    parallelism: usize,
) -> Result<Vec<(usize, Arc<ProcessedBase>)>> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(parallelism.max(1)));
    let mut tasks = Vec::new();
    for base in bases {
        let path = streams.join(&manifest.bases[base].file);
        let permit = semaphore.clone().acquire_owned().await?;
        tasks.push(tokio::task::spawn_blocking(
            move || -> Result<(usize, Arc<ProcessedBase>)> {
                let _permit = permit;
                let raw = read_base(
                    &path,
                    Sections {
                        events: true,
                        queries: false,
                    },
                )?;
                let processed = futures::executor::block_on(process_base(&raw))
                    .with_context(|| format!("processing {}", path.display()))?;
                Ok((base, Arc::new(processed)))
            },
        ));
    }
    let mut loaded = Vec::new();
    for task in tasks {
        loaded.push(task.await??);
    }
    Ok(loaded)
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = Args::parse();
    let manifest = Manifest::read(&args.streams)?;
    let layout = PhantomLayout {
        total: args.total_phantoms,
        worker_id_base: args.worker_id_base,
        salt_seed: args.salt_seed,
    };
    layout.validate(&manifest)?;
    ensure!(args.count > 0, "--count must be positive");
    ensure!(
        args.first_phantom + args.count <= args.total_phantoms,
        "phantoms {}..{} exceed --total-phantoms {}",
        args.first_phantom,
        args.first_phantom + args.count,
        args.total_phantoms
    );
    ensure!(
        u64::from(args.base_port) + args.count - 1 <= u64::from(u16::MAX),
        "ports run past 65535"
    );
    let speedup = resolve_speedup(
        &manifest,
        &layout,
        args.speedup,
        args.target_write_blocks_per_sec,
    )?;
    let map = TimeMap {
        start_at_unix_us: args.start_at_unix_ms * 1000,
        t0_us: manifest.t0_us,
        speedup,
    };
    let phantoms = args.first_phantom..args.first_phantom + args.count;
    let local_rates = natural_rates(&manifest, &layout, phantoms.clone());

    let mut bases: Vec<usize> = phantoms
        .clone()
        .map(|phantom| layout.base_of(phantom, manifest.bases.len()))
        .collect();
    bases.dedup();
    let started = std::time::Instant::now();
    let loaded = load_bases(&args.streams, &manifest, bases, args.load_parallelism).await?;
    let load_s = started.elapsed().as_secs_f64();

    let mut phantom_states = Vec::with_capacity(args.count as usize);
    for (offset, index) in phantoms.clone().enumerate() {
        let base_index = layout.base_of(index, manifest.bases.len());
        let base = loaded
            .iter()
            .find(|(loaded_base, _)| *loaded_base == base_index)
            .map(|(_, base)| base.clone())
            .expect("every needed base is loaded");
        let endpoint = format!(
            "tcp://{}:{}",
            args.bind_host,
            u64::from(args.base_port) + offset as u64
        );
        let (transport, bound) = ZmqPubTransport::bind(&endpoint, KV_EVENT_SUBJECT)
            .await
            .with_context(|| format!("binding phantom {index} at {endpoint}"))?;
        tracing::debug!(phantom = index, %bound, "phantom bound");
        phantom_states.push(Phantom {
            index,
            worker_id: layout.worker_id(index),
            salt_key: layout.salt_key(index),
            delay_us: layout.start_delay_us(index, args.start_spread_ms * 1000),
            base,
            transport,
        });
    }
    let sources_entry = format!(
        "{}+{}@tcp://{}:{}",
        layout.worker_id(args.first_phantom),
        args.count,
        args.advertise_host,
        args.base_port
    );
    if let Some(path) = &args.sources_out {
        std::fs::write(path, format!("{sources_entry}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
    }

    let plan = json!({
        "kind": "plan",
        "phantoms": [args.first_phantom, args.first_phantom + args.count],
        "total_phantoms": args.total_phantoms,
        "sources_entry": sources_entry,
        "speedup": speedup,
        "start_at_unix_ms": args.start_at_unix_ms,
        "coverage_s": map.coverage_s(&manifest),
        "expected_timed_write_blocks_per_s": local_rates.write_blocks * speedup,
        "expected_timed_events_per_s": local_rates.events * speedup,
        "warmup_write_blocks": local_rates.warmup_write_blocks,
        "warmup_blocks_per_sec": args.warmup_blocks_per_sec,
        "load_s": load_s,
        "bases_loaded": loaded.len(),
    });
    println!("{plan}");

    let shared = Arc::new(Shared {
        first_timed_send_us: AtomicU64::new(u64::MAX),
        ..Shared::default()
    });
    let pacing = Arc::new(Pacing {
        map,
        warmup_blocks_per_sec_per_phantom: args.warmup_blocks_per_sec / args.count as f64,
        warmup_start: Instant::now() + Duration::from_secs_f64(args.warmup_delay_s),
        stop_at_unix_us: args
            .duration_s
            .map(|seconds| map.start_at_unix_us + (seconds * 1e6) as u64),
        epoch_unix_us: unix_now_us(),
        epoch: Instant::now(),
    });

    let mut handles = Vec::with_capacity(phantom_states.len());
    for phantom in phantom_states {
        handles.push(tokio::spawn(run_phantom(
            phantom,
            pacing.clone(),
            shared.clone(),
        )));
    }

    let reporter = {
        let shared = shared.clone();
        let interval = Duration::from_secs_f64(args.report_interval_s);
        tokio::spawn(async move {
            let mut last = std::time::Instant::now();
            loop {
                tokio::time::sleep(interval).await;
                let seconds = last.elapsed().as_secs_f64();
                last = std::time::Instant::now();
                let line = json!({
                    "kind": "interval",
                    "t_unix_ms": unix_now_us() / 1000,
                    "warming": shared.warming.load(Ordering::Relaxed),
                    "timed_running": shared.timed_running.load(Ordering::Relaxed),
                    "exhausted": shared.exhausted.load(Ordering::Relaxed),
                    "warmup": shared.warmup.interval(seconds),
                    "timed": shared.timed.interval(seconds),
                    "timed_lag": summarize_ms(&shared.lag.take_interval()),
                    "send_errors": shared.send_errors.total(),
                });
                println!("{line}");
            }
        })
    };

    let all_done = futures::future::join_all(handles);
    tokio::select! {
        _ = all_done => {}
        _ = tokio::signal::ctrl_c() => {
            shared.stop.store(true, Ordering::Relaxed);
            tracing::warn!("interrupted; stopping phantoms");
        }
    }
    reporter.abort();

    let first = shared.first_timed_send_us.load(Ordering::Relaxed);
    let last = shared.last_timed_send_us.load(Ordering::Relaxed);
    let timed_s = if first == u64::MAX {
        0.0
    } else {
        (last.saturating_sub(first)) as f64 / 1e6
    };
    let timed_totals = shared.timed.totals();
    let timed_write_blocks =
        shared.timed.stored_blocks.total() + shared.timed.removed_blocks.total();
    let summary = json!({
        "kind": "summary",
        "plan": plan,
        "warmup": shared.warmup.totals(),
        "timed": timed_totals,
        "timed_wall_s": timed_s,
        "achieved_timed_write_blocks_per_s": if timed_s > 0.0 { timed_write_blocks as f64 / timed_s } else { 0.0 },
        "timed_lag": summarize_ms(&shared.lag.cumulative()),
        "late_warmups": shared.late_warmups.load(Ordering::Relaxed),
        "exhausted": shared.exhausted.load(Ordering::Relaxed),
        "send_errors": shared.send_errors.total(),
    });
    println!("{summary}");
    if let Some(path) = &args.summary_out {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}

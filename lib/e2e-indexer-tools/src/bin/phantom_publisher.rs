// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: phantom KV-event publishers.
//!
//! Hosts `--count` phantom workers. Each binds its own ZMQ XPUB socket and replays its base's
//! captured engine events (salted per phantom) in a format-identical copy of the production
//! wire: the publisher pipeline's envelopes (`OfflinePublisherPipeline`), msgpack
//! `Vec<RouterEvent>` payloads in an `EventEnvelope`, and the 4-frame multipart of
//! `ZmqPubTransport`. The serving indexer accepts the phantoms through
//! `DYN_EXPERIMENT_STATIC_KV_SOURCES` (see `--sources-out`).
//!
//! Nothing is sent until the indexer has subscribed to every phantom socket of this process
//! (the XPUB socket reports subscriptions; `--subscribe-timeout-s` fails the run). The warm-up
//! prefix is then paced by block rate and the timed section open loop on the shared wall-clock
//! mapping. Sends that hit the high-water mark are dropped, as a PUB socket would, but counted.
//!
//! The summary lists, per phantom, what it planned and what it sent (events, write blocks, last
//! event ID), so `delivery_check` can require exact delivery per phantom. `finished_unix_ms` is
//! stamped after the ZMQ context terminated, i.e. after the sockets' linger flush.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use dynamo_e2e_indexer_tools::pipeline::{
    ProcessedBase, WriteTotals, envelope_events, prefix_totals, process_base,
};
use dynamo_e2e_indexer_tools::plan::{
    DEFAULT_WORKER_ID_BASE, PhantomLayout, TimeMap, natural_rates, resolve_speedup, unix_now_us,
};
use dynamo_e2e_indexer_tools::stats::{Counter, Latency, summarize_ms};
use dynamo_e2e_indexer_tools::stream::{EventLists, Manifest, Sections, read_base};
use dynamo_e2e_indexer_tools::xpub::{self, PhantomSocket, SendOutcome};
use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
use dynamo_runtime::transports::event_plane::Codec;
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
    /// Phantoms hosted here (one XPUB socket each; keep below libzmq's 1023-socket limit).
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
    /// Pace each phantom's warm-up in proportion to its warm-up write blocks, so every phantom
    /// finishes together after (this process's warm-up write blocks) / --warmup-blocks-per-sec.
    /// The default gives every phantom the same rate, so the largest base sets the tail.
    #[arg(long)]
    warmup_proportional: bool,
    /// Fail if the indexer has not subscribed to every phantom socket within this many seconds.
    #[arg(long, default_value_t = 600.0)]
    subscribe_timeout_s: f64,
    /// Seconds between the last subscription and the first warm-up send.
    #[arg(long, default_value_t = 1.0)]
    warmup_delay_s: f64,
    /// Unix ms at which the timed section starts (shared with every publisher and driver). The
    /// warm-up must finish before it (see `late_warmups`).
    #[arg(long)]
    start_at_unix_ms: u64,
    /// Per-phantom timed-phase start delay, uniform in [0, spread).
    #[arg(long, default_value_t = 0)]
    start_spread_ms: u64,
    /// Stop the timed phase this long after --start-at-unix-ms (default: when exhausted).
    #[arg(long)]
    duration_s: Option<f64>,
    /// Refuse to run when the planned timed window removes fewer blocks than this fraction of
    /// the blocks it stores (the capture had not reached steady-state eviction).
    #[arg(long, default_value_t = 0.8)]
    min_timed_remove_ratio: f64,
    /// Run despite a timed remove ratio below --min-timed-remove-ratio (smoke tests only).
    #[arg(long)]
    allow_low_eviction: bool,
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
    dropped_envelopes: Counter,
    dropped_events: Counter,
    dropped_write_blocks: Counter,
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

    fn record_drop(&self, events: &[RouterEvent]) {
        let (stored, removed) = write_blocks(events);
        self.dropped_envelopes.add(1);
        self.dropped_events.add(events.len() as u64);
        self.dropped_write_blocks.add(stored + removed);
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
            "hwm_dropped_envelopes": self.dropped_envelopes.take_interval(),
        })
    }

    fn sent(&self) -> WriteTotals {
        WriteTotals {
            events: self.events.total(),
            stored_blocks: self.stored_blocks.total(),
            removed_blocks: self.removed_blocks.total(),
        }
    }

    fn totals(&self) -> serde_json::Value {
        json!({
            "envelopes": self.envelopes.total(),
            "events": self.events.total(),
            "stored_blocks": self.stored_blocks.total(),
            "removed_blocks": self.removed_blocks.total(),
            "write_blocks": self.sent().write_blocks(),
            "bytes": self.bytes.total(),
            "hwm_dropped": {
                "envelopes": self.dropped_envelopes.total(),
                "events": self.dropped_events.total(),
                "write_blocks": self.dropped_write_blocks.total(),
            },
        })
    }
}

/// One phantom's successful sends; written only by its own task.
#[derive(Default)]
struct PhantomTally {
    events: AtomicU64,
    write_blocks: AtomicU64,
    last_event_id: AtomicU64,
}

#[derive(Default)]
struct Shared {
    warmup: Section,
    timed: Section,
    /// Indexed by the phantom's offset in this process.
    per_phantom: Vec<PhantomTally>,
    /// Timed-section lateness versus the schedule, measured once the send returns.
    lag: Latency,
    send_errors: Counter,
    warming: AtomicU64,
    timed_running: AtomicU64,
    exhausted: AtomicU64,
    late_warmups: AtomicU64,
    /// Phantoms whose subscriber reconnected (or left) after the gate.
    resubscribed: AtomicU64,
    unsubscribed: AtomicU64,
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
    /// Offset in this process (index into `Shared::per_phantom`).
    slot: usize,
    worker_id: u64,
    salt_key: u64,
    delay_us: u64,
    base: Arc<ProcessedBase>,
    /// Timed lists this phantom sends: those whose deadline falls before the stop.
    timed_end: usize,
    /// Everything it sends when nothing is dropped: the warm-up (unless skipped) and the timed
    /// lists before `timed_end`. Event IDs run contiguously from 1 over both.
    planned: WriteTotals,
    /// Warm-up write blocks this phantom sends (0 when the warm-up is skipped).
    warmup_write_blocks: u64,
    /// This phantom's warm-up pacing, in write blocks per second.
    warmup_rate: f64,
    socket: PhantomSocket,
}

struct Pacing {
    map: TimeMap,
    warmup_blocks_per_sec_per_phantom: f64,
    warmup_start: Instant,
    /// Unix microseconds corresponding to `epoch`, for converting deadlines to `Instant`s.
    epoch_unix_us: u64,
    epoch: Instant,
}

impl Pacing {
    fn instant_at(&self, unix_us: u64) -> Instant {
        self.epoch + Duration::from_micros(unix_us.saturating_sub(self.epoch_unix_us))
    }
}

fn send(
    phantom: &Phantom,
    sequence: &mut u64,
    events: &[RouterEvent],
    codec: &Codec,
    section: &Section,
    shared: &Shared,
) {
    let result = (|| {
        let payload = codec.encode_payload(&events)?;
        let envelope = codec.encode_envelope_parts(
            phantom.worker_id,
            *sequence,
            unix_now_us() / 1000,
            KV_EVENT_SUBJECT,
            &payload,
        )?;
        let bytes = envelope.len();
        let outcome =
            phantom
                .socket
                .send(KV_EVENT_SUBJECT, phantom.worker_id, *sequence, envelope)?;
        anyhow::Ok((outcome, bytes))
    })();
    *sequence += 1;
    match result {
        Ok((SendOutcome::Sent, bytes)) => {
            section.record(events, bytes);
            let (stored, removed) = write_blocks(events);
            let tally = &shared.per_phantom[phantom.slot];
            tally
                .events
                .fetch_add(events.len() as u64, Ordering::Relaxed);
            tally
                .write_blocks
                .fetch_add(stored + removed, Ordering::Relaxed);
            if let Some(last) = events.last() {
                tally
                    .last_event_id
                    .store(last.event.event_id, Ordering::Relaxed);
            }
        }
        Ok((SendOutcome::HwmDrop, _)) => section.record_drop(events),
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

async fn run_phantom(mut phantom: Phantom, pacing: Arc<Pacing>, shared: Arc<Shared>) {
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
            );
            next += Duration::from_secs_f64(blocks.max(1) as f64 / phantom.warmup_rate);
        }
        shared.warming.fetch_sub(1, Ordering::Relaxed);
        // Warm-up traffic after the timed start would load the measured window.
        if unix_now_us() > pacing.map.start_at_unix_us {
            shared.late_warmups.fetch_add(1, Ordering::Relaxed);
        }
    }

    shared.timed_running.fetch_add(1, Ordering::Relaxed);
    let first_id = base.timed_first_event_id();
    for list in 0..phantom.timed_end {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        let deadline = pacing
            .map
            .wall_us(base.timed.list_ts_us[list], phantom.delay_us);
        tokio::time::sleep_until(pacing.instant_at(deadline)).await;
        let events = section_events(&base.timed, list, first_id, &phantom);
        send(
            &phantom,
            &mut sequence,
            &events,
            &codec,
            &shared.timed,
            &shared,
        );
        let sent_us = unix_now_us();
        shared
            .lag
            .record(phantom.index as usize, sent_us.saturating_sub(deadline));
        shared
            .first_timed_send_us
            .fetch_min(sent_us, Ordering::Relaxed);
        shared
            .last_timed_send_us
            .fetch_max(sent_us, Ordering::Relaxed);
    }
    shared.timed_running.fetch_sub(1, Ordering::Relaxed);
    if !shared.stop.load(Ordering::Relaxed) {
        shared.exhausted.fetch_add(1, Ordering::Relaxed);
    }
    if phantom.socket.poll_subscriptions().is_ok() {
        if phantom.socket.subscribes() > 1 {
            shared.resubscribed.fetch_add(1, Ordering::Relaxed);
        }
        if phantom.socket.unsubscribes() > 0 {
            shared.unsubscribed.fetch_add(1, Ordering::Relaxed);
        }
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

/// What this process will send: every warm-up list, and each phantom's timed lists before the
/// stop. Fills in each phantom's `timed_end`.
fn plan_sends(
    phantoms: &mut [Phantom],
    map: &TimeMap,
    stop_at_unix_us: Option<u64>,
    warmup: bool,
) -> (WriteTotals, WriteTotals) {
    let mut planned_warmup = WriteTotals::default();
    let mut planned_timed = WriteTotals::default();
    let mut start = 0;
    while start < phantoms.len() {
        let base = phantoms[start].base.clone();
        let end = start
            + phantoms[start..]
                .iter()
                .take_while(|phantom| Arc::ptr_eq(&phantom.base, &base))
                .count();
        let group = &mut phantoms[start..end];
        for phantom in group.iter_mut() {
            phantom.timed_end = match stop_at_unix_us {
                Some(stop) => base
                    .timed
                    .list_ts_us
                    .partition_point(|&ts| map.wall_us(ts, phantom.delay_us) < stop),
                None => base.timed.lists(),
            };
        }
        let prefixes: Vec<usize> = group.iter().map(|phantom| phantom.timed_end).collect();
        let warmup_totals = if warmup {
            prefix_totals(&base.warmup, &[base.warmup.lists()])[0]
        } else {
            WriteTotals::default()
        };
        for (phantom, timed) in group.iter_mut().zip(prefix_totals(&base.timed, &prefixes)) {
            planned_timed.add(timed);
            planned_warmup.add(warmup_totals);
            phantom.planned = warmup_totals;
            phantom.planned.add(timed);
            phantom.warmup_write_blocks = warmup_totals.write_blocks();
        }
        start = end;
    }
    (planned_warmup, planned_timed)
}

/// Sets each phantom's warm-up rate and returns the expected warm-up duration in seconds (the
/// slowest phantom's). Uniform pacing gives each phantom `rate / phantoms`; proportional pacing
/// gives it `rate * its blocks / all blocks`, so every phantom finishes together.
fn set_warmup_rates(phantoms: &mut [Phantom], rate: f64, proportional: bool) -> f64 {
    if rate <= 0.0 || phantoms.is_empty() {
        return 0.0;
    }
    let total: u64 = phantoms.iter().map(|phantom| phantom.warmup_write_blocks).sum();
    let uniform = rate / phantoms.len() as f64;
    let mut expected_s: f64 = 0.0;
    for phantom in phantoms.iter_mut() {
        phantom.warmup_rate = if proportional && total > 0 && phantom.warmup_write_blocks > 0 {
            rate * phantom.warmup_write_blocks as f64 / total as f64
        } else {
            uniform
        };
        expected_s = expected_s.max(phantom.warmup_write_blocks as f64 / phantom.warmup_rate);
    }
    expected_s
}

/// Wait until the indexer has subscribed to every phantom socket; returns per-phantom
/// subscription latencies after binding.
async fn subscription_gate(
    phantoms: &mut [Phantom],
    timeout: Duration,
    report_interval: Duration,
) -> Result<Latency> {
    let started = Instant::now();
    let latency = Latency::default();
    let mut pending: Vec<usize> = (0..phantoms.len()).collect();
    let mut next_report = started + report_interval;
    loop {
        let mut still_pending = Vec::with_capacity(pending.len());
        for &index in &pending {
            if phantoms[index].socket.poll_subscriptions()? {
                latency.record(index, started.elapsed().as_micros() as u64);
            } else {
                still_pending.push(index);
            }
        }
        pending = still_pending;
        if pending.is_empty() {
            return Ok(latency);
        }
        let now = Instant::now();
        if now >= started + timeout {
            let missing: Vec<u64> = pending
                .iter()
                .take(8)
                .map(|&index| phantoms[index].worker_id)
                .collect();
            println!(
                "{}",
                json!({
                    "kind": "error",
                    "error": "subscribe_timeout",
                    "timeout_s": timeout.as_secs_f64(),
                    "pending": pending.len(),
                    "pending_worker_ids": missing,
                })
            );
            bail!(
                "the indexer did not subscribe to {} of {} phantom sockets within {:.0} s (first missing worker IDs {missing:?}); is DYN_EXPERIMENT_STATIC_KV_SOURCES set from this plan?",
                pending.len(),
                phantoms.len(),
                timeout.as_secs_f64()
            );
        }
        if now >= next_report {
            next_report = now + report_interval;
            println!(
                "{}",
                json!({
                    "kind": "subscribe_gate",
                    "subscribed": phantoms.len() - pending.len(),
                    "pending": pending.len(),
                    "elapsed_s": started.elapsed().as_secs_f64(),
                })
            );
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            _ = tokio::signal::ctrl_c() => bail!("interrupted while waiting for subscriptions"),
        }
    }
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
    let stop_at_unix_us = args
        .duration_s
        .map(|seconds| map.start_at_unix_us + (seconds * 1e6) as u64);
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

    let context = xpub::context()?;
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
        let (socket, bound) = PhantomSocket::bind(&context, &endpoint)
            .with_context(|| format!("binding phantom {index} at {endpoint}"))?;
        tracing::debug!(phantom = index, %bound, "phantom bound");
        phantom_states.push(Phantom {
            index,
            slot: offset,
            worker_id: layout.worker_id(index),
            salt_key: layout.salt_key(index),
            delay_us: layout.start_delay_us(index, args.start_spread_ms * 1000),
            base,
            timed_end: 0,
            planned: WriteTotals::default(),
            warmup_write_blocks: 0,
            warmup_rate: 0.0,
            socket,
        });
    }
    let warmup = args.warmup_blocks_per_sec > 0.0;
    let (planned_warmup, planned_timed) =
        plan_sends(&mut phantom_states, &map, stop_at_unix_us, warmup);
    let warmup_expected_s = set_warmup_rates(
        &mut phantom_states,
        args.warmup_blocks_per_sec,
        args.warmup_proportional,
    );
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

    let timed_remove_ratio = planned_timed.remove_ratio();
    let plan = json!({
        "kind": "plan",
        "phantoms": [args.first_phantom, args.first_phantom + args.count],
        "total_phantoms": args.total_phantoms,
        "sources_entry": sources_entry,
        "speedup": speedup,
        "start_at_unix_ms": args.start_at_unix_ms,
        "duration_s": args.duration_s,
        "coverage_s": map.coverage_s(&manifest),
        "expected_timed_write_blocks_per_s": local_rates.write_blocks * speedup,
        "expected_timed_events_per_s": local_rates.events * speedup,
        "warmup_blocks_per_sec": args.warmup_blocks_per_sec,
        "warmup_pacing": if args.warmup_proportional { "proportional" } else { "uniform" },
        "warmup_expected_s": warmup_expected_s,
        "planned": {
            "warmup": planned_warmup.json(),
            "timed": planned_timed.json(),
            "timed_remove_ratio": timed_remove_ratio,
        },
        "load_s": load_s,
        "bases_loaded": loaded.len(),
    });
    println!("{plan}");
    if timed_remove_ratio < args.min_timed_remove_ratio {
        let message = format!(
            "the planned timed window removes {timed_remove_ratio:.3} of the blocks it stores, below --min-timed-remove-ratio {}; the capture has not reached steady-state eviction (see the exporter's eviction report)",
            args.min_timed_remove_ratio
        );
        if !args.allow_low_eviction {
            bail!(message);
        }
        tracing::warn!("{message}; continuing because of --allow-low-eviction");
    }

    let latency = subscription_gate(
        &mut phantom_states,
        Duration::from_secs_f64(args.subscribe_timeout_s),
        Duration::from_secs_f64(args.report_interval_s),
    )
    .await?;
    let subscribe_latency = summarize_ms(&latency.cumulative());
    println!(
        "{}",
        json!({
            "kind": "subscribed",
            "t_unix_ms": unix_now_us() / 1000,
            "phantoms": phantom_states.len(),
            "subscribe_latency": subscribe_latency,
        })
    );

    let planned_per_phantom: Vec<(u64, WriteTotals)> = phantom_states
        .iter()
        .map(|phantom| (phantom.worker_id, phantom.planned))
        .collect();
    let shared = Arc::new(Shared {
        per_phantom: (0..phantom_states.len())
            .map(|_| PhantomTally::default())
            .collect(),
        first_timed_send_us: AtomicU64::new(u64::MAX),
        ..Shared::default()
    });
    let pacing = Arc::new(Pacing {
        map,
        warmup_blocks_per_sec_per_phantom: args.warmup_blocks_per_sec / args.count as f64,
        warmup_start: Instant::now() + Duration::from_secs_f64(args.warmup_delay_s),
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

    let aborts: Vec<_> = handles.iter().map(|handle| handle.abort_handle()).collect();
    let all_done = futures::future::join_all(handles);
    tokio::pin!(all_done);
    tokio::select! {
        _ = &mut all_done => {}
        _ = tokio::signal::ctrl_c() => {
            shared.stop.store(true, Ordering::Relaxed);
            tracing::warn!("interrupted; stopping phantoms");
            // A phantom may sleep until a distant deadline: cancel it so its socket closes.
            for abort in &aborts {
                abort.abort();
            }
            all_done.await;
        }
    }
    reporter.abort();
    let sends_done_unix_ms = unix_now_us() / 1000;
    // Every socket is closed; terminating the context waits for their linger flush, so the
    // indexer has been handed every message before `finished_unix_ms`.
    tokio::task::spawn_blocking(move || drop(context)).await?;

    let first = shared.first_timed_send_us.load(Ordering::Relaxed);
    let last = shared.last_timed_send_us.load(Ordering::Relaxed);
    let timed_s = if first == u64::MAX {
        0.0
    } else {
        (last.saturating_sub(first)) as f64 / 1e6
    };
    let timed_write_blocks = shared.timed.sent().write_blocks();
    let mut sent = shared.warmup.sent();
    sent.add(shared.timed.sent());
    let per_phantom_rows: Vec<[u64; 7]> = planned_per_phantom
        .iter()
        .zip(&shared.per_phantom)
        .map(|((worker_id, planned), tally)| {
            [
                *worker_id,
                planned.events,
                planned.write_blocks(),
                // Event IDs run contiguously from 1 through the warm-up and the timed lists.
                if warmup { planned.events } else { 0 },
                tally.events.load(Ordering::Relaxed),
                tally.write_blocks.load(Ordering::Relaxed),
                tally.last_event_id.load(Ordering::Relaxed),
            ]
        })
        .collect();
    let summary = json!({
        "kind": "summary",
        "sends_done_unix_ms": sends_done_unix_ms,
        "finished_unix_ms": unix_now_us() / 1000,
        "stop_at_unix_ms": stop_at_unix_us.map(|stop| stop / 1000),
        "first_timed_send_unix_ms": (first != u64::MAX).then_some(first / 1000),
        "last_timed_send_unix_ms": (first != u64::MAX).then_some(last / 1000),
        "plan": plan,
        "subscribe_latency": subscribe_latency,
        "warmup": shared.warmup.totals(),
        "timed": shared.timed.totals(),
        "sent": sent.json(),
        "hwm_dropped_envelopes": shared.warmup.dropped_envelopes.total()
            + shared.timed.dropped_envelopes.total(),
        "timed_wall_s": timed_s,
        "achieved_timed_write_blocks_per_s": if timed_s > 0.0 { timed_write_blocks as f64 / timed_s } else { 0.0 },
        "timed_lag": summarize_ms(&shared.lag.cumulative()),
        "late_warmups": shared.late_warmups.load(Ordering::Relaxed),
        "exhausted": shared.exhausted.load(Ordering::Relaxed),
        "resubscribed_phantoms": shared.resubscribed.load(Ordering::Relaxed),
        "unsubscribed_phantoms": shared.unsubscribed.load(Ordering::Relaxed),
        "send_errors": shared.send_errors.total(),
        "interrupted": shared.stop.load(Ordering::Relaxed),
        "per_phantom_columns": [
            "worker_id",
            "planned_events",
            "planned_write_blocks",
            "planned_last_event_id",
            "sent_events",
            "sent_write_blocks",
            "last_sent_event_id",
        ],
        "per_phantom": per_phantom_rows,
    });
    let mut line = summary.clone();
    line.as_object_mut()
        .expect("the summary is an object")
        .remove("per_phantom");
    println!("{line}");
    if let Some(path) = &args.summary_out {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}

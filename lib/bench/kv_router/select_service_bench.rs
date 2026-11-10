// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed-loop load generator for the standalone selection service
//! (`python -m dynamo.select_service`, or `gr_core_host` for the Global Router full-KV
//! benchmark).
//!
//! In-process live Mocker engines stand in for workers. With `--event-transport zmq` each
//! publishes vLLM-wire KV events on its own ZMQ PUB socket; with `--event-transport velo`
//! the process acts as one pool relay and streams each worker's router events to
//! `gr_core_host` over velo (see `gr_relay/pool.rs`). Either way the service indexes the
//! cache state its own routing decisions create. Several processes can drive one service:
//! each sends its own workers as `allowed_worker_ids` and replays its own session copies
//! (`--copy-base`). Session slots replay a Mooncake-format trace turn by turn through
//! either the Ray select-then-reserve flow or `select_and_reserve`, submit the turn to the
//! selected engine, and drive the reservation lifecycle (`prefill_complete`, one
//! `output_block` per generated block, `DELETE`) from the engine's output stream.
//!
//! The run moves through warmup, measure and profile phases and touches
//! `<out>/phase.<name>` at each start so an outside profiler can align with them.
//! Per-window and per-phase results go to `<out>/windows.jsonl` and `<out>/summary.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail, ensure};
use clap::{Parser, ValueEnum};
use dynamo_kv_router::protocols::KvCacheEventData;
use dynamo_mocker::common::protocols::{
    DirectRequest, KvEventPublishers, MockerConfig, RawKvEvent, RawKvEventSink,
};
use dynamo_mocker::live::{LiveEngine, LiveEngineConfig, RequestOutputBuffering};
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use uuid::Uuid;

#[path = "select_service/workload.rs"]
mod workload;

#[path = "gr_relay/common.rs"]
mod relay_common;
#[path = "gr_relay/peer_file.rs"]
mod relay_peer;
#[path = "gr_relay/pool.rs"]
mod relay_pool;

use relay_common::VeloOptions;
use relay_pool::PoolRelay;
use workload::{Turn, Workload};

const MAX_LOGGED_ERRORS: u64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Flow {
    /// `POST /select`, then `POST /reservations` replaying the cached selection.
    Ray,
    /// Atomic `POST /select_and_reserve`.
    SelectAndReserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum EventTransport {
    /// One raw vLLM ZMQ PUB per engine, read by the service's ZMQ ingress.
    Zmq,
    /// This process is one pool relay: router events over one velo stream per worker.
    Velo,
}

#[derive(Debug, Clone, Parser, Serialize)]
#[command(about = "Closed-loop session load against the standalone selection service")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8092")]
    service_url: String,
    #[arg(long, value_enum, default_value_t = EventTransport::Zmq)]
    event_transport: EventTransport,
    /// `--event-transport velo`: the core's peer file (`gr_core_host --peer-file`).
    #[arg(long)]
    peer_file: Option<PathBuf>,
    /// `--event-transport velo`: this relay's velo listen address (the node's NIC).
    #[arg(long)]
    velo_bind: Option<std::net::SocketAddr>,
    #[arg(long, default_value_t = 4)]
    velo_lanes: u16,
    #[arg(long, default_value_t = 2)]
    velo_linger_ms: u64,
    /// Worker threads of the relay runtime (`gr-relay`).
    #[arg(long, default_value_t = 4)]
    relay_threads: usize,
    /// Do not restrict selections to this process's workers. Only correct when one
    /// process drives the service.
    #[arg(long)]
    no_allowed_workers: bool,
    /// Session copies start at this index, so processes sharing a service replay
    /// distinct sessions (only `--shared-prefix-blocks` stays shared).
    #[arg(long, default_value_t = 0)]
    copy_base: u64,
    /// Mooncake-format trace with `session_id`; rows of a session are replayed in file order.
    #[arg(long)]
    trace: PathBuf,
    /// Tokens represented by one trace hash id.
    #[arg(long, default_value_t = 64)]
    trace_block_size: usize,
    /// Leading hash ids shared by every row (a global system prompt). They stay identical
    /// across session copies; every other id is shifted per copy.
    #[arg(long, default_value_t = 0)]
    shared_prefix_blocks: usize,
    #[arg(long, value_enum, default_value_t = Flow::Ray)]
    flow: Flow,
    #[arg(long, default_value_t = 1024)]
    workers: usize,
    #[arg(long, default_value_t = 1)]
    worker_id_base: u64,
    #[arg(long, default_value_t = 20_000)]
    zmq_base_port: u16,
    /// Bind a replay ROUTER per engine at this base port and register it as the worker's
    /// `replay_endpoint`, as Ray Serve LLM's vLLM KV-events config does.
    #[arg(long)]
    zmq_replay_base_port: Option<u16>,
    /// Host the service uses to reach the engines' ZMQ publishers.
    #[arg(long, default_value = "127.0.0.1")]
    zmq_host: String,
    #[arg(long, default_value_t = 16)]
    block_size: u32,
    #[arg(long, default_value_t = 65_536)]
    num_gpu_blocks: usize,
    #[arg(long, default_value_t = 8_192)]
    max_num_batched_tokens: usize,
    #[arg(long)]
    max_num_seqs: Option<usize>,
    /// Mocker engine speedup; 1.0 is real time.
    #[arg(long, default_value_t = 1.0)]
    speedup: f64,
    /// Concurrent sessions (closed loop: one turn in flight per session).
    #[arg(long, default_value_t = 8_192)]
    concurrency: usize,
    /// Spread session starts over this many seconds.
    #[arg(long, default_value_t = 30)]
    ramp_secs: u64,
    #[arg(long, default_value = "default")]
    model_name: String,
    /// Do not send `session_id` with selections.
    #[arg(long)]
    no_session_id: bool,
    /// Send no `output_block` calls; the service learns decode blocks from KV events,
    /// as a Global Router would.
    #[arg(long)]
    no_decode_progress: bool,
    /// Start every session at its first turn instead of a random turn for each slot's
    /// first session.
    #[arg(long)]
    no_random_start: bool,
    /// Generated tokens per `output_block` call; defaults to the block size.
    #[arg(long)]
    output_block_tokens: Option<usize>,
    #[arg(long, default_value_t = 240)]
    warmup_secs: u64,
    #[arg(long, default_value_t = 120)]
    measure_secs: u64,
    #[arg(long, default_value_t = 60)]
    profile_secs: u64,
    #[arg(long, default_value_t = 10)]
    window_secs: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Fail the run when errors exceed this fraction of calls in measure and profile.
    #[arg(long, default_value_t = 1e-4)]
    max_error_fraction: f64,
    #[arg(long, default_value_t = 300)]
    ready_timeout_secs: u64,
    /// Touched once this process's workers are registered (and relay streams open).
    #[arg(long)]
    ready_file: Option<PathBuf>,
    /// Wait for this file before starting load, so processes sharing a service
    /// start together.
    #[arg(long)]
    start_file: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
}

impl Args {
    fn replay_endpoint(&self, worker_index: usize) -> Option<String> {
        self.zmq_replay_base_port
            .map(|base| format!("tcp://{}:{}", self.zmq_host, base as usize + worker_index))
    }
}

// ---------------------------------------------------------------------------
// Selection API client bodies

#[derive(Serialize)]
struct WorkerBody<'a> {
    worker_id: u64,
    model_name: &'a str,
    endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_events_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    replay_endpoint: Option<String>,
    block_size: u32,
    max_num_batched_tokens: u64,
    total_kv_blocks: u64,
}

#[derive(Serialize)]
struct SelectBody<'a> {
    model_name: &'a str,
    selection_id: &'a str,
    token_ids: &'a [u32],
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_worker_ids: Option<&'a [u64]>,
}

#[derive(Serialize)]
struct ReserveBody<'a> {
    model_name: &'a str,
    selection_id: &'a str,
}

#[derive(Deserialize)]
struct SelectReply {
    worker_id: u64,
    overlap: OverlapReply,
}

#[derive(Deserialize)]
struct OverlapReply {
    gpu: u64,
}

#[derive(Deserialize)]
struct ReadyReply {
    schedulable_workers: usize,
}

// ---------------------------------------------------------------------------
// Stats

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoint {
    Select,
    Reserve,
    SelectAndReserve,
    PrefillComplete,
    OutputBlock,
    Free,
}

const ENDPOINTS: [Endpoint; 6] = [
    Endpoint::Select,
    Endpoint::Reserve,
    Endpoint::SelectAndReserve,
    Endpoint::PrefillComplete,
    Endpoint::OutputBlock,
    Endpoint::Free,
];

impl Endpoint {
    fn name(self) -> &'static str {
        match self {
            Endpoint::Select => "select",
            Endpoint::Reserve => "reserve",
            Endpoint::SelectAndReserve => "select_and_reserve",
            Endpoint::PrefillComplete => "prefill_complete",
            Endpoint::OutputBlock => "output_block",
            Endpoint::Free => "free",
        }
    }
}

#[derive(Default)]
struct EndpointStats {
    latencies_us: Mutex<Vec<u32>>,
    /// HTTP status of failed calls; 0 is a transport error.
    errors: Mutex<BTreeMap<u16, u64>>,
}

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        #[derive(Default)]
        struct Counters { $($name: AtomicU64,)* }

        const COUNTER_NAMES: &[&str] = &[$(stringify!($name),)*];

        impl Counters {
            fn snapshot(&self) -> Vec<u64> {
                vec![$(self.$name.load(Ordering::Relaxed),)*]
            }
        }
    };
}

counters!(
    started,
    completed,
    failed,
    rejected,
    prompt_tokens,
    predicted_hit_tokens,
    engine_cached_tokens,
    output_tokens,
    stored_blocks,
    removed_blocks,
    cleared_events,
    lifecycle_events,
    select_body_bytes,
);

#[derive(Default)]
struct Stats {
    endpoints: [EndpointStats; ENDPOINTS.len()],
    counters: Counters,
    logged_errors: AtomicU64,
}

impl Stats {
    fn add(&self, counter: impl Fn(&Counters) -> &AtomicU64, value: u64) {
        counter(&self.counters).fetch_add(value, Ordering::Relaxed);
    }

    fn record(&self, endpoint: Endpoint, elapsed: Duration, error: Option<u16>) {
        let stats = &self.endpoints[endpoint as usize];
        stats
            .latencies_us
            .lock()
            .unwrap()
            .push(elapsed.as_micros().min(u32::MAX as u128) as u32);
        if let Some(status) = error {
            *stats.errors.lock().unwrap().entry(status).or_default() += 1;
        }
    }
}

/// Counts published KV events before forwarding them to the ZMQ publisher or the relay.
struct CountingSink {
    inner: Box<dyn RawKvEventSink>,
    stats: Arc<Stats>,
}

impl CountingSink {
    fn count(&self, event: &RawKvEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(store) => self
                .stats
                .add(|c| &c.stored_blocks, store.blocks.len() as u64),
            KvCacheEventData::Removed(remove) => self
                .stats
                .add(|c| &c.removed_blocks, remove.block_hashes.len() as u64),
            KvCacheEventData::Cleared => self.stats.add(|c| &c.cleared_events, 1),
        }
    }
}

impl RawKvEventSink for CountingSink {
    fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
        self.count(&event);
        self.inner.publish(event)
    }

    fn publish_batch(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        events.iter().for_each(|event| self.count(event));
        self.inner.publish_batch(events)
    }
}

// ---------------------------------------------------------------------------
// Phases and reporting

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Phase {
    Warmup = 0,
    Measure = 1,
    Profile = 2,
    Done = 3,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Warmup => "warmup",
            Phase::Measure => "measure",
            Phase::Profile => "profile",
            Phase::Done => "done",
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            0 => Phase::Warmup,
            1 => Phase::Measure,
            2 => Phase::Profile,
            _ => Phase::Done,
        }
    }
}

struct PhaseAgg {
    secs: f64,
    counters: Vec<u64>,
    latencies_us: Vec<Vec<u32>>,
    errors: Vec<BTreeMap<u16, u64>>,
}

impl Default for PhaseAgg {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseAgg {
    fn new() -> Self {
        Self {
            secs: 0.0,
            counters: vec![0; COUNTER_NAMES.len()],
            latencies_us: vec![Vec::new(); ENDPOINTS.len()],
            errors: vec![BTreeMap::new(); ENDPOINTS.len()],
        }
    }

    fn merge(&mut self, other: &PhaseAgg) {
        self.secs += other.secs;
        for (total, value) in self.counters.iter_mut().zip(&other.counters) {
            *total += value;
        }
        for (total, values) in self.latencies_us.iter_mut().zip(&other.latencies_us) {
            total.extend_from_slice(values);
        }
        for (total, errors) in self.errors.iter_mut().zip(&other.errors) {
            for (status, count) in errors {
                *total.entry(*status).or_default() += count;
            }
        }
    }

    fn calls_and_errors(&self) -> (u64, u64) {
        let calls = self.latencies_us.iter().map(|l| l.len() as u64).sum();
        let errors = self.errors.iter().flat_map(|e| e.values()).sum();
        (calls, errors)
    }

    fn report(&mut self) -> Value {
        let secs = self.secs.max(1e-9);
        let counter = |name: &str| {
            let idx = COUNTER_NAMES
                .iter()
                .position(|n| *n == name)
                .expect("counter");
            self.counters[idx]
        };
        let ratio = |num: u64, den: u64| {
            if den == 0 {
                0.0
            } else {
                num as f64 / den as f64
            }
        };
        let mut endpoints = serde_json::Map::new();
        for (idx, endpoint) in ENDPOINTS.iter().enumerate() {
            let latencies = &mut self.latencies_us[idx];
            if latencies.is_empty() {
                continue;
            }
            latencies.sort_unstable();
            let pct = |q: f64| {
                let rank = ((latencies.len() as f64 * q).ceil() as usize).clamp(1, latencies.len());
                latencies[rank - 1] as f64 / 1000.0
            };
            endpoints.insert(
                endpoint.name().to_string(),
                json!({
                    "calls": latencies.len(),
                    "calls_per_s": latencies.len() as f64 / secs,
                    "p50_ms": pct(0.50),
                    "p90_ms": pct(0.90),
                    "p99_ms": pct(0.99),
                    "max_ms": pct(1.0),
                    "errors": self.errors[idx],
                }),
            );
        }
        let (calls, errors) = self.calls_and_errors();
        json!({
            "secs": self.secs,
            "requests_per_s": counter("completed") as f64 / secs,
            "http_calls_per_s": calls as f64 / secs,
            "error_fraction": ratio(errors, calls),
            "prompt_tokens_per_s": counter("prompt_tokens") as f64 / secs,
            "output_tokens_per_s": counter("output_tokens") as f64 / secs,
            "predicted_hit_ratio": ratio(counter("predicted_hit_tokens"), counter("prompt_tokens")),
            "engine_hit_ratio": ratio(counter("engine_cached_tokens"), counter("prompt_tokens")),
            "kv_stored_blocks_per_s": counter("stored_blocks") as f64 / secs,
            "kv_removed_blocks_per_s": counter("removed_blocks") as f64 / secs,
            "endpoints": endpoints,
            "counters": COUNTER_NAMES
                .iter()
                .zip(&self.counters)
                .map(|(name, value)| (name.to_string(), json!(value)))
                .collect::<serde_json::Map<_, _>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// Load

struct Ctx {
    args: Args,
    workload: Workload,
    engines: Vec<LiveEngine>,
    http: reqwest::Client,
    /// This process's workers, sent as `allowed_worker_ids` on every selection.
    allowed_workers: Option<Vec<u64>>,
    relay_counters: Option<Arc<relay_pool::RelayCounters>>,
    stats: Arc<Stats>,
    phase: AtomicU8,
    next_session: AtomicU64,
    output_block_tokens: usize,
}

enum Lifecycle {
    PrefillComplete,
    /// A generated block boundary.
    OutputBlock,
    Free,
}

impl Ctx {
    fn phase(&self) -> Phase {
        Phase::from_u8(self.phase.load(Ordering::Relaxed))
    }

    async fn call(
        &self,
        endpoint: Endpoint,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Option<Vec<u8>> {
        let started = Instant::now();
        let mut request = self
            .http
            .request(method, format!("{}{path}", self.args.service_url));
        if let Some(body) = body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body);
        }
        let result = match request.send().await {
            Ok(response) => {
                let status = response.status();
                match response.bytes().await {
                    Ok(bytes) if status.is_success() => Ok(bytes.to_vec()),
                    Ok(bytes) => Err((
                        status.as_u16(),
                        String::from_utf8_lossy(&bytes).into_owned(),
                    )),
                    Err(error) => Err((0, error.to_string())),
                }
            }
            Err(error) => Err((0, error.to_string())),
        };
        self.stats.record(
            endpoint,
            started.elapsed(),
            result.as_ref().err().map(|(status, _)| *status),
        );
        match result {
            Ok(bytes) => Some(bytes),
            Err((status, message)) => {
                if self.stats.logged_errors.fetch_add(1, Ordering::Relaxed) < MAX_LOGGED_ERRORS {
                    eprintln!("{} {path} failed ({status}): {message}", endpoint.name());
                }
                None
            }
        }
    }

    async fn select(
        &self,
        selection_id: &str,
        tokens: &[u32],
        session_id: &str,
    ) -> Option<SelectReply> {
        let body = serde_json::to_vec(&SelectBody {
            model_name: &self.args.model_name,
            selection_id,
            token_ids: tokens,
            session_id: (!self.args.no_session_id).then_some(session_id),
            allowed_worker_ids: self.allowed_workers.as_deref(),
        })
        .expect("select body serializes");
        self.stats.add(|c| &c.select_body_bytes, body.len() as u64);
        let reply = match self.args.flow {
            Flow::Ray => {
                let reply = self
                    .call(Endpoint::Select, Method::POST, "/select", Some(body))
                    .await?;
                let reserve = serde_json::to_vec(&ReserveBody {
                    model_name: &self.args.model_name,
                    selection_id,
                })
                .expect("reservation body serializes");
                self.call(
                    Endpoint::Reserve,
                    Method::POST,
                    "/reservations",
                    Some(reserve),
                )
                .await?;
                reply
            }
            Flow::SelectAndReserve => {
                self.call(
                    Endpoint::SelectAndReserve,
                    Method::POST,
                    "/select_and_reserve",
                    Some(body),
                )
                .await?
            }
        };
        match serde_json::from_slice(&reply) {
            Ok(reply) => Some(reply),
            Err(error) => {
                eprintln!("unparseable selection reply: {error}");
                None
            }
        }
    }

    async fn run_turn(self: &Arc<Self>, session_id: &str, copy: u64, turn: &Turn) {
        let tokens = self.workload.tokens(turn, copy);
        let prompt_tokens = tokens.len() as u64;
        let uuid = Uuid::new_v4();
        let selection_id = uuid.to_string();
        self.stats.add(|c| &c.started, 1);
        let Some(reply) = self.select(&selection_id, &tokens, session_id).await else {
            self.stats.add(|c| &c.failed, 1);
            return;
        };
        let Some(engine) = reply
            .worker_id
            .checked_sub(self.args.worker_id_base)
            .and_then(|idx| self.engines.get(idx as usize))
        else {
            eprintln!("selection returned unknown worker {}", reply.worker_id);
            self.stats.add(|c| &c.failed, 1);
            return;
        };
        self.stats.add(|c| &c.prompt_tokens, prompt_tokens);
        self.stats.add(
            |c| &c.predicted_hit_tokens,
            reply.overlap.gpu.min(prompt_tokens),
        );

        // Lifecycle calls run in order on their own task so a slow call never stalls the
        // engine's output stream.
        let (lifecycle_tx, mut lifecycle_rx) = mpsc::unbounded_channel();
        let lifecycle = {
            let ctx = Arc::clone(self);
            let base = format!("/reservations/{selection_id}");
            let output_block_path = format!("{base}/output_block");
            let prefill_complete_path = format!("{base}/prefill_complete");
            tokio::spawn(async move {
                while let Some(call) = lifecycle_rx.recv().await {
                    let (endpoint, method, path, body) = match call {
                        Lifecycle::PrefillComplete => (
                            Endpoint::PrefillComplete,
                            Method::POST,
                            &prefill_complete_path,
                            None,
                        ),
                        Lifecycle::OutputBlock => (
                            Endpoint::OutputBlock,
                            Method::POST,
                            &output_block_path,
                            Some(b"{}".to_vec()),
                        ),
                        Lifecycle::Free => (Endpoint::Free, Method::DELETE, &base, None),
                    };
                    ctx.call(endpoint, method, path, body).await;
                }
            })
        };

        let request = DirectRequest {
            tokens,
            max_output_tokens: turn.output_length,
            uuid: Some(uuid),
            ..Default::default()
        };
        match engine.submit(request).await {
            Ok(mut live) => {
                let mut generated = 0usize;
                let mut first = true;
                while let Some(signal) = live.recv().await {
                    if first {
                        first = false;
                        let _ = lifecycle_tx.send(Lifecycle::PrefillComplete);
                        if let Some(cached) = signal.cached_tokens {
                            self.stats.add(|c| &c.engine_cached_tokens, cached as u64);
                        }
                    }
                    if signal.token_id.is_some() {
                        generated += 1;
                        if !self.args.no_decode_progress
                            && generated.is_multiple_of(self.output_block_tokens)
                        {
                            let _ = lifecycle_tx.send(Lifecycle::OutputBlock);
                        }
                    }
                    if signal.completed {
                        if signal.rejected {
                            self.stats.add(|c| &c.rejected, 1);
                        }
                        break;
                    }
                }
                self.stats.add(|c| &c.output_tokens, generated as u64);
            }
            Err(error) => eprintln!("engine submit failed: {error:#}"),
        }
        let _ = lifecycle_tx.send(Lifecycle::Free);
        drop(lifecycle_tx);
        let _ = lifecycle.await;
        self.stats.add(|c| &c.completed, 1);
    }

    async fn run_slot(self: Arc<Self>, slot: usize) {
        let ramp = Duration::from_secs(self.args.ramp_secs);
        tokio::time::sleep(ramp.mul_f64(slot as f64 / self.args.concurrency as f64)).await;
        let mut rng =
            StdRng::seed_from_u64(self.args.seed ^ (slot as u64).wrapping_mul(0x9E37_79B9));
        let templates = self.workload.sessions.len() as u64;
        let mut first_session = true;
        while self.phase() != Phase::Done {
            let n = self.next_session.fetch_add(1, Ordering::Relaxed);
            let template = &self.workload.sessions[(n % templates) as usize];
            let copy = self.args.copy_base + n / templates;
            let start = if first_session && !self.args.no_random_start {
                rng.random_range(0..template.turns.len())
            } else {
                0
            };
            first_session = false;
            let session_id = format!("{}-{copy}", template.id);
            for turn in &template.turns[start..] {
                if self.phase() == Phase::Done {
                    return;
                }
                self.run_turn(&session_id, copy, turn).await;
            }
        }
    }
}

fn mocker_config(args: &Args) -> anyhow::Result<MockerConfig> {
    let mut engine = json!({
        "block_size": args.block_size,
        "num_gpu_blocks": args.num_gpu_blocks,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "speedup_ratio": args.speedup,
        "enable_prefix_caching": true,
    });
    if let Some(max_num_seqs) = args.max_num_seqs {
        engine["max_num_seqs"] = json!(max_num_seqs);
    }
    MockerConfig::from_value(json!({ "engine": engine }))
}

async fn register_workers(ctx: &Ctx) -> anyhow::Result<()> {
    let args = &ctx.args;
    for chunk in (0..args.workers).collect::<Vec<_>>().chunks(64) {
        let calls = chunk.iter().map(|&idx| {
            let worker_id = args.worker_id_base + idx as u64;
            let body = serde_json::to_vec(&WorkerBody {
                worker_id,
                model_name: &args.model_name,
                endpoint: format!("mock://worker-{worker_id}"),
                kv_events_endpoint: (args.event_transport == EventTransport::Zmq).then(|| {
                    format!(
                        "tcp://{}:{}",
                        args.zmq_host,
                        args.zmq_base_port as usize + idx
                    )
                }),
                replay_endpoint: match args.event_transport {
                    EventTransport::Zmq => args.replay_endpoint(idx),
                    EventTransport::Velo => None,
                },
                block_size: args.block_size,
                max_num_batched_tokens: args.max_num_batched_tokens as u64,
                total_kv_blocks: args.num_gpu_blocks as u64,
            })
            .expect("worker body serializes");
            let url = format!("{}/workers", args.service_url);
            async move {
                let response = ctx
                    .http
                    .post(url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body)
                    .send()
                    .await?;
                let status = response.status();
                if !status.is_success() {
                    bail!(
                        "POST /workers {worker_id} -> {status}: {}",
                        response.text().await?
                    );
                }
                Ok(())
            }
        });
        futures_util::future::try_join_all(calls).await?;
    }
    let deadline = Instant::now() + Duration::from_secs(args.ready_timeout_secs);
    loop {
        let response = ctx
            .http
            .get(format!("{}/ready", args.service_url))
            .send()
            .await;
        if let Ok(response) = response
            && let Ok(ready) = response.json::<ReadyReply>().await
            && ready.schedulable_workers >= args.workers
        {
            return Ok(());
        }
        ensure!(
            Instant::now() < deadline,
            "service not ready with {} workers",
            args.workers
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn touch_phase(out: &Path, phase: Phase) -> anyhow::Result<()> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
    std::fs::write(
        out.join(format!("phase.{}", phase.name())),
        format!("{now:.3}\n"),
    )?;
    Ok(())
}

/// Advance phases on window boundaries, write one line per window, and return the
/// per-phase aggregates once the profile phase ends.
async fn report(ctx: Arc<Ctx>) -> anyhow::Result<BTreeMap<&'static str, PhaseAgg>> {
    use std::io::Write;

    let args = &ctx.args;
    let window = Duration::from_secs(args.window_secs);
    let phase_windows = [
        (Phase::Warmup, args.warmup_secs / args.window_secs),
        (Phase::Measure, args.measure_secs / args.window_secs),
        (Phase::Profile, args.profile_secs / args.window_secs),
    ];
    let mut windows_file =
        std::io::BufWriter::new(std::fs::File::create(args.out.join("windows.jsonl"))?);
    let mut phases: BTreeMap<&'static str, PhaseAgg> = BTreeMap::new();
    let run_start = Instant::now();
    let mut previous = ctx.stats.counters.snapshot();
    let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + window, window);
    let mut last_tick = Instant::now();
    for (phase, windows) in phase_windows {
        ctx.phase.store(phase as u8, Ordering::Relaxed);
        touch_phase(&args.out, phase)?;
        for _ in 0..windows {
            ticker.tick().await;
            let now = Instant::now();
            let current = ctx.stats.counters.snapshot();
            let mut agg = PhaseAgg::new();
            agg.secs = (now - last_tick).as_secs_f64();
            last_tick = now;
            agg.counters = current.iter().zip(&previous).map(|(c, p)| c - p).collect();
            previous = current;
            for (idx, stats) in ctx.stats.endpoints.iter().enumerate() {
                agg.latencies_us[idx] = std::mem::take(&mut *stats.latencies_us.lock().unwrap());
                agg.errors[idx] = std::mem::take(&mut *stats.errors.lock().unwrap());
            }
            phases.entry(phase.name()).or_default().merge(&agg);
            let counters = ctx.stats.counters.snapshot();
            let inflight = counters[0] - counters[1] - counters[2];
            let mut line = agg.report();
            line["t"] = json!(run_start.elapsed().as_secs_f64());
            line["phase"] = json!(phase.name());
            line["inflight"] = json!(inflight);
            if let Some(relay) = &ctx.relay_counters {
                line["relay"] = relay.snapshot();
            }
            serde_json::to_writer(&mut windows_file, &line)?;
            windows_file.write_all(b"\n")?;
            windows_file.flush()?;
            let first_hop = match args.flow {
                Flow::Ray => "select",
                Flow::SelectAndReserve => "select_and_reserve",
            };
            eprintln!(
                "[{:>6.0}s {:<7}] req/s {:>8.1} http/s {:>9.1} inflight {:>6} {first_hop} p50/p99 {:>7.2}/{:>8.2} ms err {:.1e} hit pred/eng {:.3}/{:.3} kv store/rm {:>9.0}/{:>9.0} blk/s",
                run_start.elapsed().as_secs_f64(),
                phase.name(),
                line["requests_per_s"].as_f64().unwrap_or(0.0),
                line["http_calls_per_s"].as_f64().unwrap_or(0.0),
                inflight,
                line["endpoints"][first_hop]["p50_ms"]
                    .as_f64()
                    .unwrap_or(0.0),
                line["endpoints"][first_hop]["p99_ms"]
                    .as_f64()
                    .unwrap_or(0.0),
                line["error_fraction"].as_f64().unwrap_or(0.0),
                line["predicted_hit_ratio"].as_f64().unwrap_or(0.0),
                line["engine_hit_ratio"].as_f64().unwrap_or(0.0),
                line["kv_stored_blocks_per_s"].as_f64().unwrap_or(0.0),
                line["kv_removed_blocks_per_s"].as_f64().unwrap_or(0.0),
            );
        }
    }
    ctx.phase.store(Phase::Done as u8, Ordering::Relaxed);
    touch_phase(&args.out, Phase::Done)?;
    Ok(phases)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(args.window_secs > 0, "--window-secs must be positive");
    for (name, secs) in [
        ("warmup", args.warmup_secs),
        ("measure", args.measure_secs),
        ("profile", args.profile_secs),
    ] {
        ensure!(
            secs % args.window_secs == 0,
            "--{name}-secs must be a multiple of --window-secs"
        );
    }
    ensure!(
        args.workers > 0 && args.concurrency > 0,
        "workers and concurrency must be positive"
    );
    std::fs::create_dir_all(&args.out)?;
    std::fs::write(
        args.out.join("args.json"),
        serde_json::to_vec_pretty(&args)?,
    )?;

    let loaded = Instant::now();
    let workload = Workload::load(
        &args.trace,
        args.trace_block_size,
        args.shared_prefix_blocks,
    )?;
    eprintln!(
        "loaded {} sessions / {} turns in {:.1}s",
        workload.sessions.len(),
        workload
            .sessions
            .iter()
            .map(|s| s.turns.len())
            .sum::<usize>(),
        loaded.elapsed().as_secs_f64()
    );

    let stats = Arc::new(Stats::default());
    let config = mocker_config(&args)?;
    let worker_ids: Vec<u64> = (0..args.workers)
        .map(|idx| args.worker_id_base + idx as u64)
        .collect();
    let (mut relay, mut sinks): (Option<PoolRelay>, Vec<Box<dyn RawKvEventSink>>) = match args
        .event_transport
    {
        EventTransport::Zmq => {
            let mut sinks: Vec<Box<dyn RawKvEventSink>> = Vec::with_capacity(args.workers);
            for idx in 0..args.workers {
                let port =
                    u16::try_from(args.zmq_base_port as usize + idx).context("ZMQ port range")?;
                let replay_port = args
                    .zmq_replay_base_port
                    .map(|base| u16::try_from(base as usize + idx).context("ZMQ replay port range"))
                    .transpose()?;
                sinks.push(Box::new(ZmqKvEventSink::bind(
                    Some(port),
                    replay_port,
                    0,
                    args.block_size,
                )?));
            }
            (None, sinks)
        }
        EventTransport::Velo => {
            let bind = args
                .velo_bind
                .context("--event-transport velo needs --velo-bind")?;
            ensure!(
                args.peer_file.is_some(),
                "--event-transport velo needs --peer-file"
            );
            let (relay, sinks) = PoolRelay::start(
                VeloOptions {
                    bind,
                    lanes: args.velo_lanes,
                    linger_ms: args.velo_linger_ms,
                },
                args.relay_threads,
                &worker_ids,
                args.block_size,
            )
            .await?;
            let sinks = sinks
                .into_iter()
                .map(|sink| Box::new(sink) as Box<dyn RawKvEventSink>)
                .collect();
            (Some(relay), sinks)
        }
    };
    let mut engines = Vec::with_capacity(args.workers);
    for inner in sinks.drain(..) {
        let sink = CountingSink {
            inner,
            stats: Arc::clone(&stats),
        };
        engines.push(LiveEngine::start_with_config_and_request_output_buffering(
            config.clone(),
            0,
            LiveEngineConfig {
                kv_event_publishers: KvEventPublishers::new(None, Some(Arc::new(sink))),
                ..Default::default()
            },
            RequestOutputBuffering::FullResponse,
        )?);
    }

    let http = reqwest::Client::builder()
        .pool_max_idle_per_host(usize::MAX)
        .tcp_nodelay(true)
        .timeout(Duration::from_secs(120))
        .build()?;
    let output_block_tokens = args
        .output_block_tokens
        .unwrap_or(args.block_size as usize)
        .max(1);
    let allowed_workers = (!args.no_allowed_workers).then_some(worker_ids);
    let ctx = Arc::new(Ctx {
        args,
        workload,
        engines,
        http,
        allowed_workers,
        relay_counters: relay.as_ref().map(|relay| Arc::clone(relay.counters())),
        stats,
        phase: AtomicU8::new(Phase::Warmup as u8),
        next_session: AtomicU64::new(0),
        output_block_tokens,
    });
    register_workers(&ctx).await?;
    if let Some(relay) = relay.as_mut() {
        let peer_file = ctx.args.peer_file.as_deref().expect("checked above");
        relay
            .open(peer_file, Duration::from_secs(ctx.args.ready_timeout_secs))
            .await?;
        eprintln!("velo relay streams open for {} workers", ctx.args.workers);
    } else {
        // Let the service's SUB sockets finish connecting before the first events.
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    eprintln!("{} workers registered and ready", ctx.args.workers);
    if let Some(path) = &ctx.args.ready_file {
        std::fs::write(path, b"ready\n")?;
    }
    if let Some(path) = &ctx.args.start_file {
        eprintln!("waiting for {}", path.display());
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    for slot in 0..ctx.args.concurrency {
        tokio::spawn(Arc::clone(&ctx).run_slot(slot));
    }
    let mut phases = report(Arc::clone(&ctx)).await?;

    let mut gated = PhaseAgg::new();
    for name in ["measure", "profile"] {
        if let Some(agg) = phases.get(name) {
            gated.merge(agg);
        }
    }
    let (calls, errors) = gated.calls_and_errors();
    let error_fraction = if calls == 0 {
        1.0
    } else {
        errors as f64 / calls as f64
    };
    let summary = json!({
        "args": &ctx.args,
        "phases": phases
            .iter_mut()
            .map(|(name, agg)| (name.to_string(), agg.report()))
            .collect::<serde_json::Map<_, _>>(),
        "gated_error_fraction": error_fraction,
    });
    std::fs::write(
        ctx.args.out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    eprintln!(
        "summary written to {}",
        ctx.args.out.join("summary.json").display()
    );
    // In-flight sessions are abandoned; the run script stops the service.
    if error_fraction > ctx.args.max_error_fraction {
        eprintln!(
            "error fraction {error_fraction:.2e} exceeds {:.2e}",
            ctx.args.max_error_fraction
        );
        std::process::exit(2);
    }
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::services::selection::{
        ReservationRequest, SelectAndReserveRequest, SelectRequest, WorkerRequest,
    };

    #[test]
    fn client_bodies_parse_as_server_requests() {
        let tokens = [7u32, 8, 9];
        let select = serde_json::to_vec(&SelectBody {
            model_name: "m",
            selection_id: "sel-1",
            token_ids: &tokens,
            session_id: Some("s-1"),
            allowed_worker_ids: Some(&[3, 4]),
        })
        .unwrap();
        let parsed: SelectRequest = serde_json::from_slice(&select).unwrap();
        assert_eq!(parsed.selection_id.as_deref(), Some("sel-1"));
        assert_eq!(parsed.prompt.token_ids.as_deref(), Some(&tokens[..]));
        assert_eq!(parsed.session_id.as_deref(), Some("s-1"));
        assert_eq!(parsed.allowed_worker_ids.map(|ids| ids.len()), Some(2));
        let parsed: SelectAndReserveRequest = serde_json::from_slice(&select).unwrap();
        assert_eq!(parsed.model_name, "m");

        let reserve = serde_json::to_vec(&ReserveBody {
            model_name: "m",
            selection_id: "sel-1",
        })
        .unwrap();
        let parsed: ReservationRequest = serde_json::from_slice(&reserve).unwrap();
        assert_eq!(parsed.selection_id, "sel-1");
        assert!(
            parsed.worker_id.is_none(),
            "must use the cached-replay form"
        );

        let worker = serde_json::to_vec(&WorkerBody {
            worker_id: 3,
            model_name: "m",
            endpoint: "mock://worker-3".to_string(),
            kv_events_endpoint: Some("tcp://127.0.0.1:20002".to_string()),
            replay_endpoint: Some("tcp://127.0.0.1:21002".to_string()),
            block_size: 16,
            max_num_batched_tokens: 8192,
            total_kv_blocks: 1024,
        })
        .unwrap();
        let parsed: WorkerRequest = serde_json::from_slice(&worker).unwrap();
        assert_eq!(parsed.worker_id, 3);
        assert_eq!(
            parsed.kv_events_endpoint.as_deref(),
            Some("tcp://127.0.0.1:20002")
        );
        assert_eq!(parsed.total_kv_blocks, Some(1024));
        assert_eq!(
            parsed.replay_endpoint.as_deref(),
            Some("tcp://127.0.0.1:21002")
        );
    }
}

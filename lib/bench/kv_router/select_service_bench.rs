// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed-loop load generator for the standalone selection service
//! (`python -m dynamo.select_service`).
//!
//! In-process live Mocker engines stand in for workers: each publishes vLLM-wire KV events
//! on its own ZMQ PUB socket, so the service indexes the cache state its own routing
//! decisions create. Session slots replay a Mooncake-format trace turn by turn through
//! either the Ray select-then-reserve flow or `select_and_reserve`, submit the turn to the
//! selected engine, and drive the reservation lifecycle (`prefill_complete`, one
//! `output_block` per generated block, `DELETE`) from the engine's output stream.
//!
//! The run moves through warmup, measure and profile phases and touches
//! `<out>/phase.<name>` at each start so an outside profiler can align with them.
//! Per-window and per-phase results go to `<out>/windows.jsonl` and `<out>/summary.json`.
//!
//! `--flow ingress` replaces HTTP with `select_service_ray_ingress.py`, which hosts the
//! service in process the way a Ray Serve LLM ingress replica does.

use std::collections::{BTreeMap, HashMap};
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

#[path = "select_service/ingress.rs"]
mod ingress;

use ingress::{IngressClient, LifecycleEvent, WorkerRegistration};

const VOCAB_SIZE: u64 = 151_936;
const MAX_LOGGED_ERRORS: u64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Flow {
    /// `POST /select`, then `POST /reservations` replaying the cached selection.
    Ray,
    /// Atomic `POST /select_and_reserve`.
    SelectAndReserve,
    /// In-process `select_and_reserve` plus batched lifecycle events through the Ray-style
    /// ingress hosts at `--ingress-socket`.
    Ingress,
}

#[derive(Debug, Clone, Parser, Serialize)]
#[command(about = "Closed-loop session load against the standalone selection service")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8092")]
    service_url: String,
    /// Unix socket of a `select_service_ray_ingress.py` replica; required by `--flow
    /// ingress`. Repeat for several replicas: each request goes to one at random and every
    /// lifecycle event goes to all of them, as Ray Serve LLM routes and broadcasts.
    #[arg(long)]
    ingress_socket: Vec<PathBuf>,
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
    /// `--flow ingress`: send no decode-progress events, like Ray Serve LLM's default
    /// (`RAY_SERVE_LLM_ENABLE_DECODE_BLOCK_PROGRESS=0`).
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
// Trace

#[derive(Debug)]
struct Turn {
    hash_ids: Vec<u64>,
    input_length: usize,
    output_length: usize,
}

#[derive(Debug)]
struct SessionTemplate {
    id: String,
    turns: Vec<Turn>,
}

#[derive(Debug)]
struct Workload {
    sessions: Vec<SessionTemplate>,
    /// Hash-id shift per session copy; larger than every id in the trace.
    copy_stride: u64,
    shared_prefix_blocks: usize,
    trace_block_size: usize,
}

#[derive(Debug, Deserialize)]
struct TraceRow {
    session_id: Option<String>,
    input_length: usize,
    output_length: usize,
    hash_ids: Vec<u64>,
}

impl Workload {
    fn load(
        path: &Path,
        trace_block_size: usize,
        shared_prefix_blocks: usize,
    ) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read trace {}", path.display()))?;
        let rows = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(idx, line)| {
                serde_json::from_str::<TraceRow>(line)
                    .with_context(|| format!("trace line {} is not a Mooncake row", idx + 1))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Self::from_rows(rows, trace_block_size, shared_prefix_blocks)
    }

    fn from_rows(
        rows: Vec<TraceRow>,
        trace_block_size: usize,
        shared_prefix_blocks: usize,
    ) -> anyhow::Result<Self> {
        ensure!(trace_block_size > 0, "trace_block_size must be positive");
        ensure!(!rows.is_empty(), "trace has no rows");
        let shared_prefix = rows[0]
            .hash_ids
            .get(..shared_prefix_blocks)
            .map(<[u64]>::to_vec);
        let Some(shared_prefix) = shared_prefix else {
            bail!("first row is shorter than --shared-prefix-blocks {shared_prefix_blocks}");
        };
        let mut max_id = 0u64;
        let mut order = Vec::new();
        let mut sessions: HashMap<String, Vec<Turn>> = HashMap::new();
        for (idx, row) in rows.into_iter().enumerate() {
            ensure!(
                row.hash_ids.starts_with(&shared_prefix),
                "row {} does not start with the {shared_prefix_blocks} shared prefix ids",
                idx + 1
            );
            max_id = max_id.max(row.hash_ids.iter().copied().max().unwrap_or(0));
            let session = row.session_id.unwrap_or_else(|| format!("row-{idx}"));
            let turns = sessions.entry(session.clone()).or_insert_with(|| {
                order.push(session);
                Vec::new()
            });
            turns.push(Turn {
                hash_ids: row.hash_ids,
                input_length: row.input_length,
                output_length: row.output_length.max(1),
            });
        }
        let sessions = order
            .into_iter()
            .map(|id| {
                let turns = sessions.remove(&id).expect("session recorded in order");
                SessionTemplate { id, turns }
            })
            .collect();
        Ok(Self {
            sessions,
            copy_stride: max_id + 1,
            shared_prefix_blocks,
            trace_block_size,
        })
    }

    /// Prompt tokens of `turn` in session copy `copy`. A token block depends only on its
    /// (shifted) hash id, so equal ids give equal blocks across sessions and turns.
    fn tokens(&self, turn: &Turn, copy: u64) -> Vec<u32> {
        let len = turn
            .input_length
            .min(turn.hash_ids.len() * self.trace_block_size);
        let mut tokens = Vec::with_capacity(len);
        'blocks: for (position, &id) in turn.hash_ids.iter().enumerate() {
            let id = if position < self.shared_prefix_blocks {
                id
            } else {
                id + copy * self.copy_stride
            };
            let mut state = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..self.trace_block_size {
                if tokens.len() == len {
                    break 'blocks;
                }
                tokens.push((splitmix64(&mut state) % VOCAB_SIZE) as u32);
            }
        }
        tokens
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Selection API client bodies

#[derive(Serialize)]
struct WorkerBody<'a> {
    worker_id: u64,
    model_name: &'a str,
    endpoint: String,
    kv_events_endpoint: String,
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

/// Counts published KV events before forwarding them to the ZMQ publisher.
struct CountingSink {
    inner: ZmqKvEventSink,
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
    ingress: Vec<IngressClient>,
    stats: Arc<Stats>,
    phase: AtomicU8,
    next_session: AtomicU64,
    output_block_tokens: usize,
}

enum Lifecycle {
    PrefillComplete,
    /// Cumulative generated tokens at the block boundary.
    OutputBlock(u32),
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
        max_tokens: usize,
    ) -> Option<SelectReply> {
        if !self.ingress.is_empty() {
            let ingress = &self.ingress[rand::random_range(0..self.ingress.len())];
            let started = Instant::now();
            let session_id = (!self.args.no_session_id).then_some(session_id);
            let max_tokens = u32::try_from(max_tokens).unwrap_or(u32::MAX);
            let result = ingress
                .select(selection_id, session_id, max_tokens, tokens)
                .await;
            self.stats.record(
                Endpoint::SelectAndReserve,
                started.elapsed(),
                result.as_ref().err().map(|_| 0),
            );
            return match result {
                Ok(selected) => Some(SelectReply {
                    worker_id: self.args.worker_id_base + u64::from(selected.worker_index),
                    overlap: OverlapReply {
                        gpu: u64::from(selected.gpu_overlap_tokens),
                    },
                }),
                Err(error) => {
                    if self.stats.logged_errors.fetch_add(1, Ordering::Relaxed) < MAX_LOGGED_ERRORS
                    {
                        eprintln!("ingress select failed: {error}");
                    }
                    None
                }
            };
        }
        let body = serde_json::to_vec(&SelectBody {
            model_name: &self.args.model_name,
            selection_id,
            token_ids: tokens,
            session_id: (!self.args.no_session_id).then_some(session_id),
        })
        .expect("select body serializes");
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
            Flow::SelectAndReserve | Flow::Ingress => {
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
        let Some(reply) = self
            .select(&selection_id, &tokens, session_id, turn.output_length)
            .await
        else {
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
            let selection_id_owned = selection_id.clone();
            let base = format!("/reservations/{selection_id}");
            let output_block_path = format!("{base}/output_block");
            let prefill_complete_path = format!("{base}/prefill_complete");
            tokio::spawn(async move {
                while let Some(call) = lifecycle_rx.recv().await {
                    if !ctx.ingress.is_empty() {
                        let id = selection_id_owned.clone();
                        let event = match call {
                            Lifecycle::PrefillComplete => LifecycleEvent::PrefillComplete(id),
                            Lifecycle::OutputBlock(_) if ctx.args.no_decode_progress => continue,
                            Lifecycle::OutputBlock(tokens) => {
                                LifecycleEvent::DecodeProgress(id, tokens)
                            }
                            Lifecycle::Free => LifecycleEvent::Completed(id),
                        };
                        for ingress in &ctx.ingress {
                            ingress.lifecycle(event.clone());
                        }
                        ctx.stats.add(|c| &c.lifecycle_events, 1);
                        continue;
                    }
                    let (endpoint, method, path, body) = match call {
                        Lifecycle::PrefillComplete => (
                            Endpoint::PrefillComplete,
                            Method::POST,
                            &prefill_complete_path,
                            None,
                        ),
                        Lifecycle::OutputBlock(_) => (
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
                        if generated.is_multiple_of(self.output_block_tokens) {
                            let tokens = u32::try_from(generated).unwrap_or(u32::MAX);
                            let _ = lifecycle_tx.send(Lifecycle::OutputBlock(tokens));
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
            let copy = n / templates;
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
                kv_events_endpoint: format!(
                    "tcp://{}:{}",
                    args.zmq_host,
                    args.zmq_base_port as usize + idx
                ),
                replay_endpoint: args.replay_endpoint(idx),
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
            serde_json::to_writer(&mut windows_file, &line)?;
            windows_file.write_all(b"\n")?;
            windows_file.flush()?;
            let first_hop = match args.flow {
                Flow::Ray => "select",
                Flow::SelectAndReserve | Flow::Ingress => "select_and_reserve",
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
    let mut engines = Vec::with_capacity(args.workers);
    for idx in 0..args.workers {
        let port = u16::try_from(args.zmq_base_port as usize + idx).context("ZMQ port range")?;
        let replay_port = args
            .zmq_replay_base_port
            .map(|base| u16::try_from(base as usize + idx).context("ZMQ replay port range"))
            .transpose()?;
        let sink = ZmqKvEventSink::bind(Some(port), replay_port, 0, args.block_size)?;
        let sink = CountingSink {
            inner: sink,
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
    let ingress = match args.flow {
        Flow::Ingress => {
            ensure!(
                !args.ingress_socket.is_empty(),
                "--flow ingress needs --ingress-socket"
            );
            futures_util::future::try_join_all(args.ingress_socket.iter().map(|path| {
                IngressClient::connect(path, Duration::from_secs(args.ready_timeout_secs))
            }))
            .await?
        }
        _ => Vec::new(),
    };
    let ctx = Arc::new(Ctx {
        args,
        workload,
        engines,
        http,
        ingress,
        stats,
        phase: AtomicU8::new(Phase::Warmup as u8),
        next_session: AtomicU64::new(0),
        output_block_tokens,
    });
    if !ctx.ingress.is_empty() {
        let workers = (0..ctx.args.workers)
            .map(|idx| WorkerRegistration {
                worker_id: ctx.args.worker_id_base + idx as u64,
                block_size: ctx.args.block_size,
                max_num_batched_tokens: ctx.args.max_num_batched_tokens as u32,
                kv_events_endpoint: format!(
                    "tcp://{}:{}",
                    ctx.args.zmq_host,
                    ctx.args.zmq_base_port as usize + idx
                ),
                replay_endpoint: ctx.args.replay_endpoint(idx),
            })
            .collect::<Vec<_>>();
        futures_util::future::try_join_all(
            ctx.ingress.iter().map(|ingress| ingress.register(&workers)),
        )
        .await?;
    } else {
        register_workers(&ctx).await?;
    }
    // Let the service's SUB sockets finish connecting before the first events.
    tokio::time::sleep(Duration::from_secs(2)).await;
    eprintln!("{} workers registered and ready", ctx.args.workers);

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

    fn row(session: &str, hash_ids: Vec<u64>, input_length: usize) -> TraceRow {
        TraceRow {
            session_id: Some(session.to_string()),
            input_length,
            output_length: 4,
            hash_ids,
        }
    }

    #[test]
    fn client_bodies_parse_as_server_requests() {
        let tokens = [7u32, 8, 9];
        let select = serde_json::to_vec(&SelectBody {
            model_name: "m",
            selection_id: "sel-1",
            token_ids: &tokens,
            session_id: Some("s-1"),
        })
        .unwrap();
        let parsed: SelectRequest = serde_json::from_slice(&select).unwrap();
        assert_eq!(parsed.selection_id.as_deref(), Some("sel-1"));
        assert_eq!(parsed.prompt.token_ids.as_deref(), Some(&tokens[..]));
        assert_eq!(parsed.session_id.as_deref(), Some("s-1"));
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
            kv_events_endpoint: "tcp://127.0.0.1:20002".to_string(),
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

    #[test]
    fn copies_share_only_the_prefix() {
        let workload = Workload::from_rows(
            vec![row("a", vec![100, 1, 2], 10), row("b", vec![100, 3], 8)],
            4,
            1,
        )
        .unwrap();
        let turn = &workload.sessions[0].turns[0];
        let copy0 = workload.tokens(turn, 0);
        let copy1 = workload.tokens(turn, 1);
        assert_eq!(copy0.len(), 10, "truncated to input_length");
        assert_eq!(copy0[..4], copy1[..4], "shared prefix block is identical");
        assert_ne!(copy0[4..8], copy1[4..8], "per-copy blocks differ");

        let mismatched = Workload::from_rows(
            vec![row("a", vec![100, 1], 8), row("b", vec![101, 1], 8)],
            4,
            1,
        );
        assert!(mismatched.is_err(), "rows must share the declared prefix");
    }
}

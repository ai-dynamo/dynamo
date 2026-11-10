// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed-loop load for the Global Router pool benchmark.
//!
//! Each session slot replays Mooncake-format sessions turn by turn: it asks the
//! Global Router (`gr_core_host`) which pool to use (`POST /route`, the prompt as
//! little-endian `u32` token ids), then sends the turn to that pool's
//! `dynamo.frontend` as a non-streaming `/v1/completions` with the prompt as token
//! ids. The pool's own KV router picks the worker.
//!
//! The run moves through warmup, measure and profile phases and writes
//! `<out>/phase.<name>` (unix time) at each start; per-window lines go to
//! `<out>/windows.jsonl` and per-phase totals to `<out>/summary.json`.

#[path = "select_service/workload.rs"]
mod workload;

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, ensure};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use workload::{Turn, Workload};

const MAX_LOGGED_ERRORS: u64 = 20;

#[derive(Debug, Clone, Parser, Serialize)]
#[command(about = "Closed-loop sessions routed by the Global Router to pool frontends")]
struct Args {
    /// Base URL of `gr_core_host`.
    #[arg(long)]
    gr_url: String,
    /// JSON array of pool frontend base URLs, indexed by the pool the router returns.
    #[arg(long)]
    frontends: PathBuf,
    /// Model name the pool frontends serve.
    #[arg(long)]
    model: String,
    #[arg(long)]
    trace: PathBuf,
    /// Tokens represented by one trace hash id.
    #[arg(long, default_value_t = 64)]
    trace_block_size: usize,
    /// Leading hash ids shared by every row (a global system prompt).
    #[arg(long, default_value_t = 0)]
    shared_prefix_blocks: usize,
    /// KV block size the router hashes with; turns its matched blocks into tokens.
    #[arg(long, default_value_t = 128)]
    block_size: u64,
    /// Concurrent sessions (one turn in flight each).
    #[arg(long)]
    sessions: usize,
    /// Session copies start here, so processes sharing a fleet replay distinct sessions.
    #[arg(long, default_value_t = 0)]
    copy_base: u64,
    #[arg(long, default_value_t = 30)]
    ramp_secs: u64,
    #[arg(long, default_value_t = 120)]
    warmup_secs: u64,
    #[arg(long, default_value_t = 120)]
    measure_secs: u64,
    #[arg(long, default_value_t = 60)]
    profile_secs: u64,
    #[arg(long, default_value_t = 10)]
    window_secs: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Wait for this file before starting load.
    #[arg(long)]
    start_file: Option<PathBuf>,
    #[arg(long, default_value_t = 600)]
    request_timeout_secs: u64,
    #[arg(long, default_value_t = 1e-4)]
    max_error_fraction: f64,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Deserialize)]
struct RouteReply {
    pool: u32,
    matched_blocks: u32,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct CompletionReply {
    #[serde(default)]
    usage: Option<Usage>,
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
    route_errors,
    completion_errors,
    prompt_tokens,
    output_tokens,
    predicted_tokens,
    cached_tokens,
    cached_reports,
    route_body_bytes,
);

#[derive(Default)]
struct Stats {
    counters: Counters,
    route_us: Mutex<Vec<u32>>,
    completion_ms: Mutex<Vec<u32>>,
    logged_errors: AtomicU64,
}

impl Stats {
    fn add(&self, counter: impl Fn(&Counters) -> &AtomicU64, value: u64) {
        counter(&self.counters).fetch_add(value, Ordering::Relaxed);
    }

    fn log_error(&self, what: &str, error: impl std::fmt::Display) {
        if self.logged_errors.fetch_add(1, Ordering::Relaxed) < MAX_LOGGED_ERRORS {
            eprintln!("{what}: {error}");
        }
    }
}

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
}

struct Ctx {
    args: Args,
    workload: Workload,
    frontends: Vec<String>,
    http: reqwest::Client,
    stats: Stats,
    phase: AtomicU8,
    next_session: AtomicU64,
}

impl Ctx {
    fn done(&self) -> bool {
        self.phase.load(Ordering::Relaxed) == Phase::Done as u8
    }

    async fn run_turn(&self, turn: &Turn, copy: u64) {
        let tokens = self.workload.tokens(turn, copy);
        let prompt_tokens = tokens.len() as u64;
        self.stats.add(|c| &c.started, 1);
        let body: Vec<u8> = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        self.stats.add(|c| &c.route_body_bytes, body.len() as u64);

        let started = Instant::now();
        let route = async {
            let response = self
                .http
                .post(format!("{}/route", self.args.gr_url))
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(body)
                .send()
                .await?
                .error_for_status()?;
            anyhow::Ok(response.json::<RouteReply>().await?)
        }
        .await;
        self.stats
            .route_us
            .lock()
            .unwrap()
            .push(started.elapsed().as_micros().min(u32::MAX as u128) as u32);
        let route = match route {
            Ok(route) => route,
            Err(error) => {
                self.stats.add(|c| &c.route_errors, 1);
                self.stats.log_error("route failed", format!("{error:#}"));
                return;
            }
        };
        let Some(frontend) = self.frontends.get(route.pool as usize) else {
            self.stats.add(|c| &c.route_errors, 1);
            self.stats
                .log_error("route failed", format!("unknown pool {}", route.pool));
            return;
        };
        let predicted_tokens =
            (u64::from(route.matched_blocks) * self.args.block_size).min(prompt_tokens);

        let started = Instant::now();
        let completion = async {
            let response = self
                .http
                .post(format!("{frontend}/v1/completions"))
                .json(&json!({
                    "model": self.args.model,
                    "prompt": tokens,
                    "max_tokens": turn.output_length,
                    "ignore_eos": true,
                    "stream": false,
                }))
                .send()
                .await?
                .error_for_status()?;
            anyhow::Ok(response.json::<CompletionReply>().await?)
        }
        .await;
        self.stats
            .completion_ms
            .lock()
            .unwrap()
            .push(started.elapsed().as_millis().min(u32::MAX as u128) as u32);
        match completion {
            Ok(reply) => {
                let usage = reply.usage.unwrap_or_default();
                // Counted with the completion, so the predicted and cached ratios share a
                // denominator.
                self.stats.add(|c| &c.completed, 1);
                self.stats.add(|c| &c.prompt_tokens, prompt_tokens);
                self.stats.add(|c| &c.predicted_tokens, predicted_tokens);
                self.stats
                    .add(|c| &c.output_tokens, usage.completion_tokens);
                if let Some(cached) = usage.prompt_tokens_details.and_then(|d| d.cached_tokens) {
                    self.stats.add(|c| &c.cached_tokens, cached);
                    self.stats.add(|c| &c.cached_reports, 1);
                }
            }
            Err(error) => {
                self.stats.add(|c| &c.completion_errors, 1);
                self.stats
                    .log_error("completion failed", format!("{error:#}"));
            }
        }
    }

    async fn run_slot(self: Arc<Self>, slot: usize) {
        let ramp = Duration::from_secs(self.args.ramp_secs);
        tokio::time::sleep(ramp.mul_f64(slot as f64 / self.args.sessions as f64)).await;
        let mut rng =
            StdRng::seed_from_u64(self.args.seed ^ (slot as u64).wrapping_mul(0x9E37_79B9));
        let templates = self.workload.sessions.len() as u64;
        let mut first_session = true;
        while !self.done() {
            let n = self.next_session.fetch_add(1, Ordering::Relaxed);
            let template = &self.workload.sessions[(n % templates) as usize];
            let copy = self.args.copy_base + n / templates;
            let start = if first_session {
                rng.random_range(0..template.turns.len())
            } else {
                0
            };
            first_session = false;
            for turn in &template.turns[start..] {
                if self.done() {
                    return;
                }
                self.run_turn(turn, copy).await;
            }
        }
    }
}

fn percentile(sorted: &[u32], q: f64) -> Option<u32> {
    (!sorted.is_empty()).then(|| {
        let rank = ((sorted.len() as f64 * q).ceil() as usize).clamp(1, sorted.len());
        sorted[rank - 1]
    })
}

#[derive(Default)]
struct Window {
    secs: f64,
    counters: Vec<u64>,
    route_us: Vec<u32>,
    completion_ms: Vec<u32>,
}

impl Window {
    fn merge(&mut self, other: &Window) {
        self.secs += other.secs;
        if self.counters.is_empty() {
            self.counters = vec![0; COUNTER_NAMES.len()];
        }
        for (total, value) in self.counters.iter_mut().zip(&other.counters) {
            *total += value;
        }
        self.route_us.extend_from_slice(&other.route_us);
        self.completion_ms.extend_from_slice(&other.completion_ms);
    }

    fn report(&mut self) -> Value {
        let secs = self.secs.max(1e-9);
        let counter = |name: &str| {
            COUNTER_NAMES
                .iter()
                .position(|n| *n == name)
                .and_then(|idx| self.counters.get(idx).copied())
                .unwrap_or(0)
        };
        let ratio = |num: u64, den: u64| {
            if den == 0 {
                0.0
            } else {
                num as f64 / den as f64
            }
        };
        self.route_us.sort_unstable();
        self.completion_ms.sort_unstable();
        let errors = counter("route_errors") + counter("completion_errors");
        json!({
            "secs": self.secs,
            "requests_per_s": counter("completed") as f64 / secs,
            "routes_per_s": self.route_us.len() as f64 / secs,
            "error_fraction": ratio(errors, counter("started")),
            "route_p50_ms": percentile(&self.route_us, 0.5).map(|us| us as f64 / 1e3),
            "route_p99_ms": percentile(&self.route_us, 0.99).map(|us| us as f64 / 1e3),
            "completion_p50_s": percentile(&self.completion_ms, 0.5).map(|ms| ms as f64 / 1e3),
            "completion_p99_s": percentile(&self.completion_ms, 0.99).map(|ms| ms as f64 / 1e3),
            "prompt_tokens_per_s": counter("prompt_tokens") as f64 / secs,
            "output_tokens_per_s": counter("output_tokens") as f64 / secs,
            "predicted_hit_ratio": ratio(counter("predicted_tokens"), counter("prompt_tokens")),
            "cached_hit_ratio": (counter("cached_reports") > 0)
                .then(|| ratio(counter("cached_tokens"), counter("prompt_tokens"))),
            "route_body_bytes_per_s": counter("route_body_bytes") as f64 / secs,
            "counters": COUNTER_NAMES
                .iter()
                .zip(&self.counters)
                .map(|(name, value)| (name.to_string(), json!(value)))
                .collect::<serde_json::Map<_, _>>(),
        })
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

async fn report(ctx: Arc<Ctx>) -> anyhow::Result<BTreeMap<&'static str, Window>> {
    let args = &ctx.args;
    let window = Duration::from_secs(args.window_secs);
    let mut lines = std::io::BufWriter::new(std::fs::File::create(args.out.join("windows.jsonl"))?);
    let mut phases: BTreeMap<&'static str, Window> = BTreeMap::new();
    let mut previous = ctx.stats.counters.snapshot();
    let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + window, window);
    let mut last = Instant::now();
    let run_start = Instant::now();
    for (phase, secs) in [
        (Phase::Warmup, args.warmup_secs),
        (Phase::Measure, args.measure_secs),
        (Phase::Profile, args.profile_secs),
    ] {
        ctx.phase.store(phase as u8, Ordering::Relaxed);
        std::fs::write(
            args.out.join(format!("phase.{}", phase.name())),
            format!("{:.3}\n", unix_now()),
        )?;
        for _ in 0..secs / args.window_secs {
            ticker.tick().await;
            let now = Instant::now();
            let current = ctx.stats.counters.snapshot();
            let mut w = Window {
                secs: (now - last).as_secs_f64(),
                counters: current.iter().zip(&previous).map(|(c, p)| c - p).collect(),
                route_us: std::mem::take(&mut *ctx.stats.route_us.lock().unwrap()),
                completion_ms: std::mem::take(&mut *ctx.stats.completion_ms.lock().unwrap()),
            };
            last = now;
            phases.entry(phase.name()).or_default().merge(&w);
            let inflight = current[0] - current[1] - current[2] - current[3];
            previous = current;
            let mut line = w.report();
            line["t"] = json!(unix_now());
            line["phase"] = json!(phase.name());
            line["inflight"] = json!(inflight);
            serde_json::to_writer(&mut lines, &line)?;
            lines.write_all(b"\n")?;
            lines.flush()?;
            eprintln!(
                "[{:>6.0}s {:<7}] req/s {:>8.1} inflight {:>6} route p50/p99 {:>6.2}/{:>7.2} ms completion p50 {:>6.2} s err {:.1e} hit pred/cached {:.3}/{}",
                run_start.elapsed().as_secs_f64(),
                phase.name(),
                line["requests_per_s"].as_f64().unwrap_or(0.0),
                inflight,
                line["route_p50_ms"].as_f64().unwrap_or(0.0),
                line["route_p99_ms"].as_f64().unwrap_or(0.0),
                line["completion_p50_s"].as_f64().unwrap_or(0.0),
                line["error_fraction"].as_f64().unwrap_or(0.0),
                line["predicted_hit_ratio"].as_f64().unwrap_or(0.0),
                line["cached_hit_ratio"]
                    .as_f64()
                    .map_or("-".to_string(), |r| format!("{r:.3}")),
            );
        }
    }
    ctx.phase.store(Phase::Done as u8, Ordering::Relaxed);
    std::fs::write(args.out.join("phase.done"), format!("{:.3}\n", unix_now()))?;
    Ok(phases)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure!(
        args.window_secs > 0 && args.sessions > 0,
        "window and sessions must be positive"
    );
    std::fs::create_dir_all(&args.out)?;
    std::fs::write(
        args.out.join("args.json"),
        serde_json::to_vec_pretty(&args)?,
    )?;
    let frontends: Vec<String> = serde_json::from_slice(
        &std::fs::read(&args.frontends)
            .with_context(|| format!("reading {}", args.frontends.display()))?,
    )?;
    ensure!(!frontends.is_empty(), "no pool frontends");
    let workload = Workload::load(
        &args.trace,
        args.trace_block_size,
        args.shared_prefix_blocks,
    )?;
    eprintln!(
        "loaded {} sessions; {} pools; {} session slots",
        workload.sessions.len(),
        frontends.len(),
        args.sessions
    );
    let http = reqwest::Client::builder()
        .pool_max_idle_per_host(usize::MAX)
        .tcp_nodelay(true)
        .timeout(Duration::from_secs(args.request_timeout_secs))
        .build()?;
    if let Some(path) = &args.start_file {
        eprintln!("waiting for {}", path.display());
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    let ctx = Arc::new(Ctx {
        args,
        workload,
        frontends,
        http,
        stats: Stats::default(),
        phase: AtomicU8::new(Phase::Warmup as u8),
        next_session: AtomicU64::new(0),
    });
    for slot in 0..ctx.args.sessions {
        tokio::spawn(Arc::clone(&ctx).run_slot(slot));
    }
    let mut phases = report(Arc::clone(&ctx)).await?;
    let mut gated = Window::default();
    for name in ["measure", "profile"] {
        if let Some(window) = phases.get(name) {
            gated.merge(window);
        }
    }
    let gated_report = gated.report();
    let error_fraction = gated_report["error_fraction"].as_f64().unwrap_or(1.0);
    let summary = json!({
        "args": &ctx.args,
        "phases": phases
            .iter_mut()
            .map(|(name, window)| (name.to_string(), window.report()))
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
    if error_fraction > ctx.args.max_error_fraction {
        eprintln!(
            "error fraction {error_fraction:.2e} exceeds {:.2e}",
            ctx.args.max_error_fraction
        );
        std::process::exit(2);
    }
    std::process::exit(0);
}

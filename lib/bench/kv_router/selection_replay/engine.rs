// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replays a corpus against a [`BenchHost`].
//!
//! Each request goes through the frontend's call sequence: `run_selection` with a
//! lease and the prompt's tokens, `commit`, the worker-origin and frontend
//! prefill-complete marks at its first output, and the acknowledged release at
//! completion. Its KV events come from the worker the router booked: the prompt's
//! new blocks at first output, one block per `block_size` output tokens during
//! decode, and evictions as the worker's cache fills (see `kv_cache`). They reach
//! the index through `try_apply_event`, the frontend's ingestion call.
//!
//! Fixed mode maps trace time to wall time with a scale. Saturation mode instead
//! advances trace time only as fast as the router admits, with at most
//! `admission_window` admissions in flight. Engine actions may trail a request's
//! arrival by `Limits::engine_lag_us`, as a real router hears of them late, so a
//! turn's release can race the next turn's arrival as it does in a frontend.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use dynamo_bench::kv_router_common::clock::BenchmarkClock;
use dynamo_bench::kv_router_common::issuer::pin_current_thread;
use dynamo_bench::kv_router_common::results::PercentileStats;
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventData, KvCacheEvent, RouterEvent, StorageTier,
};
use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;
use dynamo_kv_router::services::selection::{
    SelectionAdmission, SelectionOperation, SelectionOutcome, SessionBinding,
};
use futures_util::FutureExt;
use serde::Serialize;

use super::corpus::Corpus;
use super::host::BenchHost;
use super::kv_cache::{CacheStats, KvCaches, PromptBlocks};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Wall time = trace time / scale.
    Fixed { scale: f64 },
    /// Trace time advances as fast as the router admits.
    Saturation { admission_window: usize },
}

/// When a run counts as keeping up.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Limits {
    /// Fixed mode: the issue lag p99 allowed for requests, in µs.
    pub request_lag_us: u64,
    /// How far engine actions may trail their trace time, in µs: the issue lag p99
    /// allowed in fixed mode, and how far saturation lets requests run ahead of them.
    /// A real router hears of engine actions a scheduler step late.
    pub engine_lag_us: u64,
    /// KV events the index may fall behind during the window.
    pub index_backlog: u64,
    /// Fixed mode: the admission latency p99 allowed, in ms.
    pub admission_p99_ms: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            request_lag_us: 50,
            engine_lag_us: 10_000,
            index_backlog: 20_000,
            admission_p99_ms: 20.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub mode: Mode,
    pub limits: Limits,
    /// One CPU per bench thread, in order: the request issuer, the sampler, the
    /// engine issuers, prompt preparation. Empty leaves them unpinned.
    pub issuer_cpus: Vec<usize>,
    pub engine_issuers: usize,
    pub prep_threads: usize,
    pub perf_ctl_fifo: Option<PathBuf>,
    pub perf_ack_fifo: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct InFlight {
    pub median: u64,
    pub max: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunResult {
    pub mode: Mode,
    /// Requests that arrived in the measured window, and those the router booked.
    pub window_requests: usize,
    pub admitted: usize,
    pub queue_rejected: usize,
    pub failed: usize,
    pub failure_samples: Vec<String>,
    pub achieved_rps: f64,
    pub kept_up: bool,
    /// Why the run did not keep up; empty when it did.
    pub failure_reasons: Vec<String>,
    /// Deadline (fixed) or issue (saturation) to booking, in ns.
    pub admission_latency_ns: Option<PercentileStats<u64>>,
    /// Fixed mode: how late window requests and engine actions were issued, in ns.
    pub request_issue_lag_ns: Option<PercentileStats<u64>>,
    pub engine_issue_lag_ns: Option<PercentileStats<u64>>,
    /// Saturation: the share of the window the request issuer waited on the bench's
    /// own threads, prompt preparation and engine actions, rather than on the router.
    pub harness_wait_share: Option<f64>,
    /// Saturation: how often trace time moved to the next engine action because every
    /// admission in the window was waiting on one, as a queueing router's do.
    pub stall_advances: u64,
    /// Bookings in flight through the window, and at the end of the run.
    pub booked_in_flight: InFlight,
    pub booked_at_end: u64,
    /// The index's applied events by `type/outcome` at the end of the run.
    pub index_events_applied: BTreeMap<String, u64>,
    /// The largest gap during the window between events handed to the index and
    /// events it had applied, and the gap as the window closed. Beyond
    /// `Limits::index_backlog`, the router routed on a stale index: it could not
    /// take this load.
    pub max_index_backlog: u64,
    pub index_backlog_at_window_end: u64,
    pub index_kept_up: bool,
    /// The workers' caches over the whole run.
    pub cache: CacheStats,
    /// Prompt blocks found cached on the booked worker, over all prompt blocks.
    pub prompt_hit_rate: f64,
    /// For window requests: the cached prompt blocks the router predicted on its
    /// pick, those the worker's cache held at first output, and their mean gap: the
    /// index's lag plus any cache changes between the pick and the first output.
    pub mean_predicted_cached_blocks: f64,
    pub mean_cached_blocks: f64,
    pub mean_cached_blocks_error: f64,
    pub window_wall_ns: u64,
}

const UNSET: u64 = u64::MAX;
/// Requests issue this long after engine actions due at the same trace time.
const TIE_NS: u64 = 1_000;
/// Saturation results are invalid if the bench's own threads, not the router, held
/// back more of the window than this.
const MAX_HARNESS_WAIT_SHARE: f64 = 0.05;
/// Prompts prepared ahead of their arrival.
const PREP_AHEAD: usize = 256;
/// Fixed mode: threads sleep until this close to a deadline, then spin.
const SPIN_NS: u64 = 75_000;
/// How often an idle engine issuer takes in finished admissions (fixed mode).
const ADMISSION_POLL_NS: u64 = 100_000;
/// Saturation: with the admission window full and no admission finishing for this
/// long, trace time moves to the next engine action.
const STALL_NS: u64 = 10_000_000;
/// How long the end of a run waits for admissions and lifecycle tasks.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Default)]
struct Slot {
    booking: OnceLock<SchedulerBookingDescriptor>,
    issued_ns: AtomicU64,
    /// When the router returned a booking; `UNSET` if it did not.
    booked_ns: AtomicU64,
    /// Prompt blocks the router predicted cached on its pick, and those the cache held.
    predicted_cached_blocks: AtomicU64,
    cached_blocks: AtomicU64,
    /// The request's place in its worker's cache, from booking to release.
    held: Mutex<Option<Held>>,
}

struct Held {
    worker_id: u64,
    dp_rank: u32,
    /// The prompt's block hashes, until prefill.
    prompt: Option<PromptBlocks>,
    /// The cache entries the request pins, prompt first.
    entries: Vec<u32>,
    /// The last block of the request's sequence so far, after prefill: its next
    /// output block chains after it.
    tail: Option<u64>,
}

struct Shared {
    corpus: Arc<Corpus>,
    host: Arc<BenchHost>,
    handle: tokio::runtime::Handle,
    clock: BenchmarkClock,
    mode: Mode,
    start_ns: u64,
    slots: Vec<Slot>,
    /// Trace time the issuers may advance to (saturation mode), in µs.
    virtual_us: AtomicU64,
    /// Each engine issuer's next unprocessed trace time; `UNSET` when done.
    issuer_next_us: Vec<AtomicU64>,
    /// Each engine issuer's finished admissions: the request, and whether it was booked.
    admissions: Vec<Sender<(usize, bool)>>,
    /// Saturation: trace µs requests may run ahead of engine actions.
    engine_slack_us: u64,
    /// The request issuer has stopped issuing.
    requests_stopped: AtomicBool,
    /// The run is over, or failed: every wait gives up.
    stop: AtomicBool,
    /// Admissions finished, booked or not.
    finished: AtomicU64,
    outstanding: AtomicUsize,
    booked_in_flight: AtomicU64,
    pending_tasks: AtomicUsize,
    queue_rejected: AtomicUsize,
    failed: AtomicUsize,
    events_issued: AtomicU64,
    stall_advances: AtomicU64,
    caches: KvCaches,
    failure_samples: Mutex<Vec<String>>,
    window_start_us: u64,
    window_end_us: u64,
}

impl Shared {
    fn deadline_ns(&self, trace_us: u64) -> u64 {
        match self.mode {
            Mode::Fixed { scale } => self.start_ns + (trace_us as f64 * 1_000.0 / scale) as u64,
            Mode::Saturation { .. } => 0,
        }
    }

    fn in_window(&self, trace_us: u64) -> bool {
        (self.window_start_us..self.window_end_us).contains(&trace_us)
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn fail(&self, message: String) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        let mut samples = self.failure_samples.lock().unwrap();
        if samples.len() < 16 {
            samples.push(message);
        }
    }
}

/// Runs `on_exit` when dropped, including while a panic unwinds, so one bench
/// thread failing releases every other thread's wait instead of hanging the run.
struct OnExit<F: FnMut()>(F);

impl<F: FnMut()> Drop for OnExit<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

fn backoff(spins: &mut u32) {
    *spins += 1;
    if *spins < 64 {
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

fn pin(cpu: Option<usize>) -> Result<()> {
    pin_current_thread(cpu).with_context(|| format!("pin a bench thread to CPU {cpu:?}"))
}

/// Replay `corpus` against `host`, which must run on `runtime`.
pub fn run(
    corpus: &Arc<Corpus>,
    host: &Arc<BenchHost>,
    runtime: &tokio::runtime::Runtime,
    config: &RunConfig,
) -> Result<RunResult> {
    let params = &corpus.params;
    let engine_issuers = config.engine_issuers.max(1);
    let prep_threads = config.prep_threads.max(1);
    let threads = 2 + engine_issuers + prep_threads;
    ensure!(
        config.issuer_cpus.is_empty() || config.issuer_cpus.len() >= threads,
        "--issuer-cpus needs {threads} CPUs: the request issuer, the sampler, \
         {engine_issuers} engine issuers and {prep_threads} prep threads"
    );
    let clock = BenchmarkClock::new(SPIN_NS)?;
    // Blocks per trace block, when trace blocks hold whole blocks.
    let block_size = params.block_size as usize;
    let run_blocks = if corpus.trace_block_size.is_multiple_of(block_size) {
        corpus.trace_block_size / block_size
    } else {
        1
    };
    let (admission_senders, admission_receivers): (Vec<_>, Vec<_>) =
        (0..engine_issuers).map(|_| channel()).unzip();
    let shared = Arc::new(Shared {
        corpus: Arc::clone(corpus),
        host: Arc::clone(host),
        handle: runtime.handle().clone(),
        start_ns: clock.now_ns() + 50_000_000,
        clock,
        mode: config.mode,
        slots: (0..corpus.requests.len())
            .map(|_| Slot::default())
            .collect(),
        virtual_us: AtomicU64::new(0),
        issuer_next_us: (0..engine_issuers).map(|_| AtomicU64::new(0)).collect(),
        admissions: admission_senders,
        engine_slack_us: config.limits.engine_lag_us,
        requests_stopped: AtomicBool::new(false),
        stop: AtomicBool::new(false),
        finished: AtomicU64::new(0),
        outstanding: AtomicUsize::new(0),
        booked_in_flight: AtomicU64::new(0),
        pending_tasks: AtomicUsize::new(0),
        queue_rejected: AtomicUsize::new(0),
        failed: AtomicUsize::new(0),
        events_issued: AtomicU64::new(0),
        stall_advances: AtomicU64::new(0),
        caches: KvCaches::new(
            params.workers,
            params.dp_size,
            params.num_gpu_blocks,
            run_blocks,
        ),
        failure_samples: Mutex::new(Vec::new()),
        window_start_us: params.warmup_ms * 1_000,
        window_end_us: (params.warmup_ms + params.window_ms) * 1_000,
    });
    for slot in &shared.slots {
        slot.issued_ns.store(UNSET, Ordering::Relaxed);
        slot.booked_ns.store(UNSET, Ordering::Relaxed);
        slot.predicted_cached_blocks.store(UNSET, Ordering::Relaxed);
        slot.cached_blocks.store(UNSET, Ordering::Relaxed);
    }

    let mut cpus = config.issuer_cpus.iter().copied();
    let request_cpu = cpus.next();
    let sampler_cpu = cpus.next();
    let engine_cpus: Vec<_> = (0..engine_issuers).map(|_| cpus.next()).collect();
    let prep_cpus: Vec<_> = (0..prep_threads).map(|_| cpus.next()).collect();
    let (prep_senders, prep_receivers): (Vec<_>, Vec<_>) = (0..prep_threads)
        .map(|_| sync_channel::<PreparedPrompt>((PREP_AHEAD / prep_threads).max(1)))
        .unzip();

    let mut engine_lags = Vec::new();
    let mut request_lags = Vec::new();
    let mut harness_wait_ns = 0;
    let mut samples = Vec::new();
    std::thread::scope(|scope| -> Result<()> {
        let shared = &shared;
        // However this body exits, every bench thread's wait ends.
        let _stop = OnExit(|| shared.stop.store(true, Ordering::Release));
        for (index, (sender, cpu)) in prep_senders.into_iter().zip(prep_cpus).enumerate() {
            scope.spawn(move || {
                if let Err(error) = pin(cpu) {
                    let _ = sender.send(Err(error));
                    return;
                }
                for request in shared
                    .corpus
                    .requests
                    .iter()
                    .skip(index)
                    .step_by(prep_threads)
                {
                    let block_size = shared.corpus.params.block_size;
                    let prepared = shared.corpus.prompt_tokens(request).map(|tokens| {
                        let prompt = PromptBlocks::from_tokens(&tokens, block_size);
                        (tokens, prompt)
                    });
                    if sender.send(prepared).is_err() {
                        return;
                    }
                }
            });
        }
        let engines: Vec<_> = engine_cpus
            .into_iter()
            .zip(admission_receivers)
            .enumerate()
            .map(|(issuer, (cpu, admissions))| {
                scope.spawn(move || {
                    let _done = OnExit(|| {
                        shared.issuer_next_us[issuer].store(UNSET, Ordering::Release);
                        if std::thread::panicking() {
                            shared.stop.store(true, Ordering::Release);
                        }
                    });
                    pin(cpu)?;
                    Ok::<_, anyhow::Error>(issue_engine(shared, issuer, engine_issuers, admissions))
                })
            })
            .collect();
        let sampler = scope.spawn(move || {
            pin(sampler_cpu)?;
            sample(shared, config)
        });
        let requests = scope.spawn(move || {
            // Let the engine issuers run out the timeline.
            let _done = OnExit(|| {
                shared.requests_stopped.store(true, Ordering::Release);
                shared.virtual_us.store(u64::MAX, Ordering::Release);
            });
            pin(request_cpu)?;
            issue_requests(shared, prep_receivers)
        });

        (request_lags, harness_wait_ns) = requests.join().expect("request issuer panicked")?;
        for issuer in engines {
            engine_lags.extend(issuer.join().expect("engine issuer panicked")?);
        }
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        while shared.outstanding.load(Ordering::Acquire) > 0
            || shared.pending_tasks.load(Ordering::Acquire) > 0
        {
            if Instant::now() >= deadline {
                shared.fail("admissions or lifecycle tasks still running after the drain".into());
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        shared.stop.store(true, Ordering::Release);
        samples = sampler.join().expect("sampler panicked")?;
        Ok(())
    })?;

    Ok(analyze(
        &shared,
        config,
        request_lags,
        engine_lags,
        harness_wait_ns,
        samples,
    ))
}

/// A request's prompt tokens and their block hashes, prepared off the router's CPUs.
type PreparedPrompt = Result<(Vec<u32>, PromptBlocks)>;

/// Issues each request at its arrival. Returns the window requests' issue lags
/// (fixed mode) and the time they waited on the bench's own threads (saturation).
fn issue_requests(
    shared: &Arc<Shared>,
    prep: Vec<Receiver<PreparedPrompt>>,
) -> Result<(Vec<u64>, u64)> {
    let mut lags = Vec::new();
    let mut harness_wait_ns = 0;
    for (index, request) in shared.corpus.requests.iter().enumerate() {
        let waited_from = shared.clock.now_ns();
        let (tokens, prompt) = prep[index % prep.len()]
            .recv()
            .context("prompt preparation stopped")??;
        match shared.mode {
            Mode::Fixed { .. } => {
                // A turn arrives as the previous one completes; let that release go first.
                let deadline = shared.deadline_ns(request.arrival_us) + TIE_NS;
                shared.clock.wait_until(deadline);
                if shared.in_window(request.arrival_us) {
                    lags.push(shared.clock.now_ns().saturating_sub(deadline));
                }
            }
            Mode::Saturation { admission_window } => {
                shared
                    .virtual_us
                    .fetch_max(request.arrival_us, Ordering::AcqRel);
                wait_for_engines(shared, request.arrival_us)?;
                if shared.in_window(request.arrival_us) {
                    harness_wait_ns += shared.clock.now_ns() - waited_from;
                }
                wait_for_admission_slot(shared, admission_window)?;
            }
        }
        if shared.stopped() {
            bail!("the run stopped");
        }
        shared.slots[index]
            .issued_ns
            .store(shared.clock.now_ns(), Ordering::Release);
        shared.outstanding.fetch_add(1, Ordering::AcqRel);
        shared
            .handle
            .spawn(admit(Arc::clone(shared), index, tokens, prompt));
    }
    Ok((lags, harness_wait_ns))
}

/// Saturation: wait until the engine issuers have applied everything due more than
/// the slack before `arrival_us`.
fn wait_for_engines(shared: &Shared, arrival_us: u64) -> Result<()> {
    let due_by = arrival_us.saturating_sub(shared.engine_slack_us);
    let mut spins = 0u32;
    while shared
        .issuer_next_us
        .iter()
        .any(|next| next.load(Ordering::Acquire) <= due_by)
    {
        if shared.stopped() {
            bail!("the run stopped");
        }
        backoff(&mut spins);
    }
    Ok(())
}

/// Saturation: wait for room in the admission window. A router that queues
/// requests until prefills complete can hold every slot waiting on an engine
/// action; trace time then moves to the next one, as it would in a deployment.
fn wait_for_admission_slot(shared: &Shared, admission_window: usize) -> Result<()> {
    let mut spins = 0u32;
    let mut finished = shared.finished.load(Ordering::Acquire);
    let mut idle_since = shared.clock.now_ns();
    while shared.outstanding.load(Ordering::Acquire) >= admission_window {
        if shared.stopped() {
            bail!("the run stopped");
        }
        let now = shared.clock.now_ns();
        let latest = shared.finished.load(Ordering::Acquire);
        if latest != finished {
            finished = latest;
            idle_since = now;
        } else if now - idle_since > STALL_NS {
            let next = shared
                .issuer_next_us
                .iter()
                .map(|next| next.load(Ordering::Acquire))
                .min()
                .unwrap_or(UNSET);
            if next != UNSET && next > shared.virtual_us.fetch_max(next, Ordering::AcqRel) {
                shared.stall_advances.fetch_add(1, Ordering::Relaxed);
            }
            idle_since = now;
        }
        backoff(&mut spins);
    }
    Ok(())
}

/// What an engine does for a request, in the order it happens at equal trace times.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Action {
    FirstOutput,
    OutputBlock(u32),
    Completion,
}

/// What the booked workers do, in trace order: prefill at a request's first output,
/// one output block per `block_size` output tokens spread evenly until completion,
/// and release at completion. Requests are split across issuers, so each request's
/// actions stay in order.
fn issue_engine(
    shared: &Arc<Shared>,
    issuer: usize,
    issuers: usize,
    admissions: Receiver<(usize, bool)>,
) -> Vec<u64> {
    let block_size = u64::from(shared.corpus.params.block_size);
    let output_block_due = |index: usize, k: u32| {
        let request = &shared.corpus.requests[index];
        let (Some(first), Some(done)) = (request.first_output_us, request.completed_us) else {
            return None;
        };
        let blocks = u64::from(request.output_length) / block_size;
        let k = u64::from(k);
        (k < blocks).then(|| first + (done - first) * (k + 1) / (blocks + 1))
    };
    let requests = (issuer..shared.corpus.requests.len()).step_by(issuers);
    let mut engine = Engine {
        shared,
        issuer,
        issuers,
        admissions,
        states: vec![0; requests.len()],
    };
    let mut timeline = BinaryHeap::new();
    for index in requests {
        let request = &shared.corpus.requests[index];
        if let Some(at) = request.first_output_us {
            timeline.push(Reverse((at, Action::FirstOutput, index)));
        }
        if let Some(at) = output_block_due(index, 0) {
            timeline.push(Reverse((at, Action::OutputBlock(0), index)));
        }
        if let Some(at) = request.completed_us {
            timeline.push(Reverse((at, Action::Completion, index)));
        }
    }
    let mut lags = Vec::new();
    while let Some(Reverse((at, action, index))) = timeline.pop() {
        let Some(lag) = engine.wait_until(at) else {
            return lags;
        };
        if shared.in_window(at) {
            lags.extend(lag);
        }
        match action {
            Action::FirstOutput => engine.reach(index, FIRST_DUE),
            Action::Completion => engine.reach(index, DONE_DUE),
            Action::OutputBlock(k) => {
                output_block(shared, index, k);
                if let Some(next) = output_block_due(index, k + 1) {
                    timeline.push(Reverse((next, Action::OutputBlock(k + 1), index)));
                }
            }
        }
    }
    shared.issuer_next_us[issuer].store(UNSET, Ordering::Release);
    engine.finish();
    lags
}

const FIRST_DUE: u8 = 1;
const DONE_DUE: u8 = 2;
const BOOKED: u8 = 4;

/// One engine issuer's requests: which lifecycle points each has reached, and
/// whether it is booked. Admission tasks only report bookings here, so all cache
/// work runs on this thread, off the router's runtime, and in each request's order.
struct Engine<'a> {
    shared: &'a Arc<Shared>,
    issuer: usize,
    issuers: usize,
    admissions: Receiver<(usize, bool)>,
    /// `FIRST_DUE`, `DONE_DUE` and `BOOKED`, by request / issuers.
    states: Vec<u8>,
}

impl Engine<'_> {
    /// A request reached its first output or completion; act if it is booked.
    fn reach(&mut self, index: usize, due: u8) {
        let state = &mut self.states[index / self.issuers];
        *state |= due;
        if *state & BOOKED != 0 {
            self.act(index, due);
        }
    }

    /// Take in finished admissions, and act on what each booking already reached.
    fn take_admissions(&mut self) {
        while let Ok((index, booked)) = self.admissions.try_recv() {
            if !booked {
                continue;
            }
            let state = &mut self.states[index / self.issuers];
            *state |= BOOKED;
            let reached = *state;
            for due in [FIRST_DUE, DONE_DUE] {
                if reached & due != 0 {
                    self.act(index, due);
                }
            }
        }
    }

    /// Update the worker's cache, then send the frontend's marks or release.
    fn act(&self, index: usize, due: u8) {
        let shared = self.shared;
        if due == FIRST_DUE {
            prefill_cache(shared, index);
        } else {
            release_cache(shared, index);
        }
        shared.pending_tasks.fetch_add(1, Ordering::AcqRel);
        let task_shared = Arc::clone(shared);
        shared.handle.spawn(async move {
            if due == FIRST_DUE {
                mark_prefill_complete(&task_shared, index).await;
            } else {
                release(&task_shared, index).await;
            }
            task_shared.pending_tasks.fetch_sub(1, Ordering::AcqRel);
        });
    }

    /// Wait until `trace_us` is due, taking in admissions meanwhile. Returns the
    /// issue lag in ns (fixed mode), or `None` once the run has stopped.
    fn wait_until(&mut self, trace_us: u64) -> Option<Option<u64>> {
        let shared = self.shared;
        shared.issuer_next_us[self.issuer].store(trace_us, Ordering::Release);
        self.take_admissions();
        match shared.mode {
            Mode::Fixed { .. } => {
                let deadline = shared.deadline_ns(trace_us);
                while shared.clock.now_ns() + ADMISSION_POLL_NS < deadline {
                    if shared.stopped() {
                        return None;
                    }
                    std::thread::sleep(Duration::from_nanos(ADMISSION_POLL_NS));
                    self.take_admissions();
                }
                shared.clock.wait_until(deadline);
                Some(Some(shared.clock.now_ns().saturating_sub(deadline)))
            }
            Mode::Saturation { .. } => {
                let mut spins = 0u32;
                while shared.virtual_us.load(Ordering::Acquire) < trace_us {
                    if shared.stopped() {
                        return None;
                    }
                    backoff(&mut spins);
                    self.take_admissions();
                }
                Some(None)
            }
        }
    }

    /// After the timeline, requests booked after their completion still release.
    fn finish(&mut self) {
        let mut spins = 0u32;
        // Admissions report here before they stop counting as outstanding.
        let shared = self.shared;
        let settled = || {
            shared.requests_stopped.load(Ordering::Acquire)
                && shared.outstanding.load(Ordering::Acquire) == 0
        };
        while !settled() && !shared.stopped() {
            self.take_admissions();
            backoff(&mut spins);
        }
        self.take_admissions();
    }
}

/// Store a booked request's prompt on its worker's cache, once.
fn prefill_cache(shared: &Shared, index: usize) {
    let slot = &shared.slots[index];
    let mut held = slot.held.lock().unwrap();
    let Some(held) = held.as_mut() else {
        return;
    };
    let Some(prompt) = held.prompt.take() else {
        return;
    };
    let worker_id = held.worker_id;
    let cached = shared.caches.prefill(
        shared.caches.index(worker_id, held.dp_rank),
        held.dp_rank,
        &prompt,
        &mut held.entries,
        |events| publish(shared, worker_id, events),
    );
    held.tail = prompt.sequence.last().copied();
    slot.cached_blocks.store(cached as u64, Ordering::Release);
}

/// Store output block `k` of a request on its worker's cache, once its prompt is there.
fn output_block(shared: &Shared, index: usize, k: u32) {
    let mut held = shared.slots[index].held.lock().unwrap();
    let Some(held) = held.as_mut().filter(|held| held.prompt.is_none()) else {
        return;
    };
    let worker_id = held.worker_id;
    let tail = shared.caches.decode_block(
        shared.caches.index(worker_id, held.dp_rank),
        held.dp_rank,
        held.tail,
        (index as u64, k),
        &mut held.entries,
        |events| publish(shared, worker_id, events),
    );
    held.tail = Some(tail);
}

/// Unpin a finished request's blocks in its worker's cache.
fn release_cache(shared: &Shared, index: usize) {
    if let Some(held) = shared.slots[index].held.lock().unwrap().take() {
        let cache = shared.caches.index(held.worker_id, held.dp_rank);
        shared.caches.release(cache, &held.entries);
    }
}

/// Hand a worker's events to the index. The concurrent indexer enqueues device
/// events without waiting, so this runs under the worker's cache lock and keeps
/// the index in the order the cache changed.
fn publish(shared: &Shared, worker_id: u64, events: Vec<KvCacheEvent>) {
    for event in events {
        let event = RouterEvent::with_storage_tier(worker_id, event, StorageTier::Device);
        match shared.host.indexer.try_apply_event(event).now_or_never() {
            Some(Ok(())) => {}
            Some(Err(error)) => shared.fail(format!("apply event on worker {worker_id}: {error}")),
            None => shared.fail(format!(
                "event on worker {worker_id} was not enqueued at once"
            )),
        }
        shared.events_issued.fetch_add(1, Ordering::Relaxed);
    }
}

struct Sample {
    trace_us: u64,
    booked_in_flight: u64,
    /// KV events handed to the index, and those it has applied: the gap is its backlog.
    events_issued: u64,
    events_applied: u64,
}

/// Samples the run every 100 ms and brackets the measured window for
/// `perf record --control`.
fn sample(shared: &Shared, config: &RunConfig) -> Result<Vec<Sample>> {
    let mut perf = PerfControl::open(config)?;
    let mut samples = Vec::new();
    let (mut opened, mut closed) = (false, false);
    let mut next_sample = shared.clock.now_ns();
    loop {
        let now = shared.clock.now_ns();
        let trace_us = match shared.mode {
            Mode::Fixed { scale } => {
                ((now.saturating_sub(shared.start_ns)) as f64 * scale / 1_000.0) as u64
            }
            Mode::Saturation { .. } => shared.virtual_us.load(Ordering::Acquire),
        };
        if !opened && trace_us >= shared.window_start_us && trace_us != u64::MAX {
            opened = true;
            perf.send("enable")?;
        }
        if opened && !closed && trace_us >= shared.window_end_us {
            closed = true;
            perf.send("disable")?;
        }
        if now >= next_sample {
            next_sample = now + 100_000_000;
            samples.push(Sample {
                trace_us,
                booked_in_flight: shared.booked_in_flight.load(Ordering::Relaxed),
                events_issued: shared.events_issued.load(Ordering::Relaxed),
                events_applied: shared.host.index_events_applied().values().sum(),
            });
        }
        if shared.stopped() {
            if opened && !closed {
                perf.send("disable")?;
            }
            return Ok(samples);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `perf record --control fifo:CTL,ACK`.
struct PerfControl {
    control: Option<std::fs::File>,
    ack: Option<std::io::BufReader<std::fs::File>>,
}

impl PerfControl {
    fn open(config: &RunConfig) -> Result<Self> {
        let control = config
            .perf_ctl_fifo
            .as_ref()
            .map(|path| std::fs::OpenOptions::new().write(true).open(path))
            .transpose()
            .context("open perf control fifo")?;
        let ack = config
            .perf_ack_fifo
            .as_ref()
            .map(std::fs::File::open)
            .transpose()
            .context("open perf ack fifo")?
            .map(std::io::BufReader::new);
        Ok(Self { control, ack })
    }

    fn send(&mut self, command: &str) -> Result<()> {
        use std::io::{BufRead, Write};
        if let Some(control) = &mut self.control {
            writeln!(control, "{command}")?;
            control.flush()?;
            if let Some(ack) = &mut self.ack {
                let mut line = String::new();
                ack.read_line(&mut line)?;
            }
        }
        Ok(())
    }
}

async fn admit(shared: Arc<Shared>, index: usize, tokens: Vec<u32>, prompt: PromptBlocks) {
    let slot = &shared.slots[index];
    let request_id = format!("req-{index}");
    let run = Box::pin(
        shared
            .host
            .service
            .core()
            .run_selection(SelectionOperation {
                key: shared.host.partition.key().clone(),
                prompt: dynamo_kv_router::services::selection::PromptView {
                    token_ids: Some(&tokens),
                    mm_routing_info: None,
                    block_mm_infos: None,
                    block_hashes: None,
                    sequence_hashes: None,
                    isl_tokens: None,
                    lora_name: None,
                    cache_namespace: None,
                    is_eagle: Some(false),
                },
                router_config_override: None,
                expected_output_tokens: None,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_context: None,
                session: SessionBinding::None,
                affinity_target: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: Default::default(),
                admission: SelectionAdmission::Lease { request_id },
                track_active_blocks: shared.host.track_active_blocks,
                return_routing_hashes: false,
                replay_id: None,
            }),
    )
    .await;
    let returned_ns = shared.clock.now_ns();
    drop(tokens);
    let mut booked = false;
    match run.result {
        Ok(SelectionOutcome::Selected(selected)) => match selected.booking {
            Some(booking) => {
                let worker = selected.response.best_worker;
                let _ = slot.booking.set(booking.commit());
                slot.booked_ns.store(returned_ns, Ordering::Release);
                slot.predicted_cached_blocks.store(
                    (selected.response.cached_tokens / shared.corpus.params.block_size as usize)
                        as u64,
                    Ordering::Release,
                );
                *slot.held.lock().unwrap() = Some(Held {
                    worker_id: worker.worker_id,
                    dp_rank: worker.dp_rank,
                    prompt: Some(prompt),
                    entries: Vec::new(),
                    tail: None,
                });
                shared.booked_in_flight.fetch_add(1, Ordering::Relaxed);
                booked = true;
            }
            None => shared.fail(format!(
                "request {index}: lease admission returned no booking"
            )),
        },
        Ok(SelectionOutcome::QueueRejected { .. }) => {
            shared.queue_rejected.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => shared.fail(format!("request {index}: {error}")),
    }
    // The request's engine issuer does its cache work and lifecycle actions.
    let engines = &shared.admissions;
    let _ = engines[index % engines.len()].send((index, booked));
    shared.finished.fetch_add(1, Ordering::Release);
    shared.outstanding.fetch_sub(1, Ordering::AcqRel);
}

/// The worker's completion mark arrives on the replica channel, then the frontend
/// marks the booking itself on the first output tokens.
async fn mark_prefill_complete(shared: &Shared, index: usize) {
    let booking = shared.slots[index].booking.get().expect("booked");
    let _ = shared.host.inbound.try_send(ActiveSequenceEvent {
        request_id: booking.request_id.clone(),
        worker: booking.worker,
        data: ActiveSequenceEventData::MarkPrefillCompleted,
        router_id: booking.worker.worker_id,
        lora_name: None,
    });
    if let Err(error) = shared
        .host
        .partition
        .scheduler()
        .mark_prefill_completed_if_booking(booking)
        .await
    {
        shared.fail(format!(
            "mark prefill complete {}: {error}",
            booking.request_id
        ));
    }
}

/// The frontend's lease `finish`: an acknowledged release of the booking.
async fn release(shared: &Shared, index: usize) {
    let booking = shared.slots[index].booking.get().expect("booked").clone();
    let request_id = booking.request_id.clone();
    if let Err(error) = shared
        .host
        .partition
        .scheduler()
        .booking_cleanup()
        .enqueue_acknowledged(booking)
        .wait()
        .await
    {
        shared.fail(format!("release {request_id}: {error}"));
    }
    shared.booked_in_flight.fetch_sub(1, Ordering::Relaxed);
}

fn percentiles(values: Vec<u64>) -> Option<PercentileStats<u64>> {
    PercentileStats::from_values(values, |value| value as f64)
}

fn analyze(
    shared: &Shared,
    config: &RunConfig,
    request_lags: Vec<u64>,
    engine_lags: Vec<u64>,
    harness_wait_ns: u64,
    samples: Vec<Sample>,
) -> RunResult {
    let limits = &config.limits;
    let mut admission = Vec::new();
    let (mut window_requests, mut admitted) = (0, 0);
    let (mut compared, mut predicted_sum, mut cached_sum, mut error_sum) =
        (0usize, 0u64, 0u64, 0u64);
    let (mut first_issued, mut last_booked) = (u64::MAX, 0u64);
    for (request, slot) in shared.corpus.requests.iter().zip(&shared.slots) {
        if !shared.in_window(request.arrival_us) {
            continue;
        }
        window_requests += 1;
        let booked = slot.booked_ns.load(Ordering::Acquire);
        if booked == UNSET {
            continue;
        }
        admitted += 1;
        let issued = slot.issued_ns.load(Ordering::Acquire);
        first_issued = first_issued.min(issued);
        last_booked = last_booked.max(booked);
        let reference = match shared.mode {
            Mode::Fixed { .. } => shared.deadline_ns(request.arrival_us),
            Mode::Saturation { .. } => issued,
        };
        admission.push(booked.saturating_sub(reference));
        let predicted = slot.predicted_cached_blocks.load(Ordering::Acquire);
        let cached = slot.cached_blocks.load(Ordering::Acquire);
        if predicted != UNSET && cached != UNSET {
            compared += 1;
            predicted_sum += predicted;
            cached_sum += cached;
            error_sum += predicted.abs_diff(cached);
        }
    }

    // Fixed mode measures against the window's deadlines; saturation from the
    // first window request issued to the last booked.
    let window_wall_ns = match shared.mode {
        Mode::Fixed { .. } => {
            let start = shared.deadline_ns(shared.window_start_us);
            let span = shared.deadline_ns(shared.window_end_us) - start;
            last_booked.saturating_sub(start).max(span)
        }
        Mode::Saturation { .. } => last_booked.saturating_sub(first_issued),
    };
    let achieved_rps = admitted as f64 * 1e9 / window_wall_ns.max(1) as f64;

    let failed = shared.failed.load(Ordering::Relaxed);
    let request_issue_lag_ns = percentiles(request_lags);
    let engine_issue_lag_ns = percentiles(engine_lags);
    let admission_latency_ns = percentiles(admission);
    let harness_wait_share = matches!(shared.mode, Mode::Saturation { .. })
        .then(|| harness_wait_ns as f64 / window_wall_ns.max(1) as f64);
    let p99 = |stats: Option<PercentileStats<u64>>| stats.map_or(0, |stats| stats.p99);

    let mut failure_reasons = Vec::new();
    if failed > 0 {
        failure_reasons.push(format!("{failed} failures"));
    }
    if admitted < window_requests {
        failure_reasons.push(format!(
            "{} of {window_requests} window requests not booked",
            window_requests - admitted
        ));
    }
    if let Mode::Fixed { .. } = shared.mode {
        for (name, lag, limit_us) in [
            ("request", request_issue_lag_ns, limits.request_lag_us),
            ("engine", engine_issue_lag_ns, limits.engine_lag_us),
        ] {
            if p99(lag) > limit_us * 1_000 {
                failure_reasons.push(format!("{name} issue lag p99 {} ns", p99(lag)));
            }
        }
        let admission_p99_ms = p99(admission_latency_ns) as f64 / 1e6;
        if admission_p99_ms > limits.admission_p99_ms {
            failure_reasons.push(format!("admission p99 {admission_p99_ms:.2} ms"));
        }
        let start = shared.deadline_ns(shared.window_start_us);
        let span = shared.deadline_ns(shared.window_end_us) - start;
        if last_booked > start + span + span / 10 {
            failure_reasons.push("window admissions ran past 1.1x the window".to_string());
        }
    }
    if let Some(share) = harness_wait_share
        && share > MAX_HARNESS_WAIT_SHARE
    {
        failure_reasons.push(format!(
            "the bench's own threads held back {:.0}% of the window",
            share * 100.0
        ));
    }

    let window_samples: Vec<&Sample> = samples
        .iter()
        .filter(|sample| shared.in_window(sample.trace_us))
        .collect();
    let backlog = |sample: &&Sample| sample.events_issued.saturating_sub(sample.events_applied);
    let max_index_backlog = window_samples.iter().map(backlog).max().unwrap_or(0);
    let index_kept_up = max_index_backlog <= limits.index_backlog;
    if !index_kept_up {
        failure_reasons.push(format!("the index fell {max_index_backlog} events behind"));
    }
    let mut in_flight: Vec<u64> = window_samples
        .iter()
        .map(|sample| sample.booked_in_flight)
        .collect();
    in_flight.sort_unstable();

    let cache = shared.caches.stats();
    RunResult {
        mode: shared.mode,
        window_requests,
        admitted,
        queue_rejected: shared.queue_rejected.load(Ordering::Relaxed),
        failed,
        failure_samples: shared.failure_samples.lock().unwrap().clone(),
        achieved_rps,
        kept_up: failure_reasons.is_empty(),
        failure_reasons,
        admission_latency_ns,
        request_issue_lag_ns,
        engine_issue_lag_ns,
        harness_wait_share,
        stall_advances: shared.stall_advances.load(Ordering::Relaxed),
        booked_in_flight: InFlight {
            median: in_flight.get(in_flight.len() / 2).copied().unwrap_or(0),
            max: in_flight.last().copied().unwrap_or(0),
        },
        booked_at_end: shared.booked_in_flight.load(Ordering::Acquire),
        index_events_applied: shared.host.index_events_applied(),
        max_index_backlog,
        index_backlog_at_window_end: window_samples.last().map_or(0, backlog),
        index_kept_up,
        cache,
        prompt_hit_rate: cache.prompt_hits as f64 / cache.prompt_blocks.max(1) as f64,
        mean_predicted_cached_blocks: predicted_sum as f64 / compared.max(1) as f64,
        mean_cached_blocks: cached_sum as f64 / compared.max(1) as f64,
        mean_cached_blocks_error: error_sum as f64 / compared.max(1) as f64,
        window_wall_ns,
    }
}

/// Requests per second the corpus generates inside the window, in trace time.
pub fn corpus_window_rate(corpus: &Corpus) -> Result<f64> {
    let params = &corpus.params;
    let start = params.warmup_ms * 1_000;
    let end = start + params.window_ms * 1_000;
    let count = corpus
        .requests
        .iter()
        .filter(|request| request.arrival_us >= start && request.arrival_us < end)
        .count();
    if count == 0 {
        bail!("the corpus has no requests in its measured window");
    }
    Ok(count as f64 * 1_000.0 / params.window_ms as f64)
}

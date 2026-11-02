// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The tokio runtime the EPP runs on, and the signals that show when it stalls.
//!
//! A gateway multiplexes many ext_proc streams onto each HTTP/2 connection,
//! and one task drives all of a connection's reads and writes. When another
//! task holds a worker thread for tens of milliseconds, that connection task
//! can wait the whole time: in the busy worker's LIFO slot, which other
//! workers cannot steal from, or behind the I/O driver when no other worker
//! takes it over. Every stream on the connection then stops in both
//! directions. These signals show such stalls from inside the process:
//!
//! * `dynamo_epp_event_loop_delay_seconds` times a 10 ms sleep. The sleep runs
//!   late only when no worker services the timer driver, so it catches stalls
//!   of the whole runtime, not a single trapped task.
//! * With `DYN_EPP_TASK_METRICS=1`, task poll hooks record each poll and each
//!   schedule delay of at least `DYN_EPP_TASK_METRICS_MIN_US` (default 1 ms),
//!   by task kind, in `dynamo_epp_task_poll_seconds` and
//!   `dynamo_epp_task_schedule_delay_seconds`. A slow poll names the task
//!   that held a worker; a schedule delay names a task that waited for one.
//!   Each of 20 ms or more is also logged as "EPP runtime stall".

use std::cell::Cell;
use std::future::Future;
use std::panic::Location;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use tokio::runtime::TaskMeta;

use crate::log_budget::LogBudget;
use crate::metrics;

/// Set to `1` or `true` to record slow task polls and schedule delays.
pub const TASK_METRICS_ENV: &str = "DYN_EPP_TASK_METRICS";

/// The shortest poll or schedule delay the task metrics record, in
/// microseconds.
pub const TASK_METRICS_MIN_US_ENV: &str = "DYN_EPP_TASK_METRICS_MIN_US";

const DEFAULT_TASK_METRICS_MIN: Duration = Duration::from_millis(1);

/// Polls, schedule delays and event-loop delays at least this long are
/// logged as "EPP runtime stall".
const STALL_LOG_THRESHOLD: Duration = Duration::from_millis(20);

/// Stall lines per second, at most.
static STALL_LOG_BUDGET: LogBudget = LogBudget::new(10);

/// Stall events the hooks can queue before the event-loop watcher logs them.
/// Events beyond it are dropped from the log but still counted in the
/// histograms.
const STALL_QUEUE: usize = 256;

const EVENT_LOOP_TICK: Duration = Duration::from_millis(10);

/// What a task does, as far as stalls go. Labels the task metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskKind {
    /// One ext_proc stream: reads the gateway's messages, picks a worker and
    /// relays the response.
    ExtProcStream,
    /// One gateway connection: TLS and every HTTP/2 frame of every stream
    /// on it.
    H2Connection,
    /// hyper's task for each HTTP/2 request, which hands an ext_proc
    /// stream's responses to its connection. Requests to the health and
    /// metrics servers run here too.
    H2Stream,
    /// The 1 Hz scan for prefill workers.
    PrefillWatcher,
    /// Anything else.
    Other,
}

impl TaskKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            TaskKind::ExtProcStream => "ext_proc_stream",
            TaskKind::H2Connection => "h2_connection",
            TaskKind::H2Stream => "h2_stream",
            TaskKind::PrefillWatcher => "prefill_watcher",
            TaskKind::Other => "other",
        }
    }

    /// Where [`spawn`] last spawned a task of this kind.
    fn spawn_site(self) -> Option<&'static OnceLock<&'static Location<'static>>> {
        static EXT_PROC_STREAM: OnceLock<&'static Location<'static>> = OnceLock::new();
        static H2_CONNECTION: OnceLock<&'static Location<'static>> = OnceLock::new();
        static PREFILL_WATCHER: OnceLock<&'static Location<'static>> = OnceLock::new();
        match self {
            TaskKind::ExtProcStream => Some(&EXT_PROC_STREAM),
            TaskKind::H2Connection => Some(&H2_CONNECTION),
            TaskKind::PrefillWatcher => Some(&PREFILL_WATCHER),
            TaskKind::H2Stream | TaskKind::Other => None,
        }
    }

    /// The kind of a task spawned at `spawned_at`. hyper spawns its
    /// per-request tasks from hyper-util's executor.
    fn of(spawned_at: &Location<'_>) -> Self {
        for kind in [
            TaskKind::ExtProcStream,
            TaskKind::H2Connection,
            TaskKind::PrefillWatcher,
        ] {
            if kind
                .spawn_site()
                .and_then(OnceLock::get)
                .is_some_and(|site| same_place(site, spawned_at))
            {
                return kind;
            }
        }
        if spawned_at.file().contains("hyper-util") {
            TaskKind::H2Stream
        } else {
            TaskKind::Other
        }
    }
}

fn same_place(a: &Location<'_>, b: &Location<'_>) -> bool {
    a.line() == b.line() && a.column() == b.column() && a.file() == b.file()
}

/// `tokio::spawn`, remembering where tasks of `kind` are spawned so that the
/// task metrics can label them.
#[track_caller]
pub(crate) fn spawn<F>(kind: TaskKind, future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    if let Some(site) = kind.spawn_site() {
        let here = Location::caller();
        site.get_or_init(|| here);
    }
    // `tokio::spawn` is `#[track_caller]` too, so the task records this
    // function's caller, the same location remembered above.
    tokio::spawn(future)
}

/// Run CPU-bound work that scales with the request body without holding an
/// async worker, as the frontend's preprocessor does. On the multi-thread
/// runtime `block_in_place` hands this worker's queue, including its LIFO
/// slot, to another thread first, so a gateway connection task woken by this
/// request is not left waiting for it. Runs inline on the current-thread
/// runtime, where `block_in_place` is not available.
pub(crate) fn off_runtime<T>(f: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current().map(|handle| handle.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(f),
        _ => f(),
    }
}

/// Build the multi-thread runtime the EPP runs on: the runtime
/// `#[tokio::main]` builds, plus the task hooks when `DYN_EPP_TASK_METRICS`
/// asks for them. A binary that wraps [`crate::run`] can build its runtime
/// here to get the task metrics too.
pub fn build_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    build_runtime_with(task_metrics_min_from_env())
}

fn task_metrics_min_from_env() -> Option<Duration> {
    let enabled = std::env::var(TASK_METRICS_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    });
    enabled.then(|| {
        std::env::var(TASK_METRICS_MIN_US_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map_or(DEFAULT_TASK_METRICS_MIN, Duration::from_micros)
    })
}

/// The task metrics' threshold, once a runtime with the hooks is built.
static TASK_METRICS_MIN: OnceLock<Duration> = OnceLock::new();

/// Stall events from the hooks, until the event-loop watcher takes them.
static STALL_EVENTS: Mutex<Option<Receiver<Stall>>> = Mutex::new(None);

fn build_runtime_with(
    task_metrics_min: Option<Duration>,
) -> std::io::Result<tokio::runtime::Runtime> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    if let Some(min) = task_metrics_min {
        let (stalls, events) = std::sync::mpsc::sync_channel(STALL_QUEUE);
        *STALL_EVENTS.lock().unwrap_or_else(PoisonError::into_inner) = Some(events);
        let _ = TASK_METRICS_MIN.set(min);
        let hooks = std::sync::Arc::new(TaskHooks { min, stalls });
        let before = std::sync::Arc::clone(&hooks);
        builder
            .track_task_schedule_latency()
            .on_before_task_poll(move |meta| before.before_poll(meta))
            .on_after_task_poll(move |meta| hooks.after_poll(meta));
    }
    builder.build()
}

thread_local! {
    /// When the task on this worker thread started its current poll.
    static POLL_STARTED: Cell<Option<Instant>> = const { Cell::new(None) };
}

struct TaskHooks {
    min: Duration,
    stalls: SyncSender<Stall>,
}

impl TaskHooks {
    fn before_poll(&self, meta: &TaskMeta<'_>) {
        if let Some(delay) = meta.schedule_latency()
            && delay >= self.min
        {
            let kind = TaskKind::of(meta.spawned_at());
            metrics::observe_task_schedule_delay(kind, delay);
            self.report(StallKind::ScheduleDelay, kind, meta, delay);
        }
        POLL_STARTED.set(Some(Instant::now()));
    }

    fn after_poll(&self, meta: &TaskMeta<'_>) {
        let Some(started) = POLL_STARTED.take() else {
            return;
        };
        let poll = started.elapsed();
        if poll >= self.min {
            let kind = TaskKind::of(meta.spawned_at());
            metrics::observe_task_poll(kind, poll);
            self.report(StallKind::SlowPoll, kind, meta, poll);
        }
    }

    /// Queue a stall for the event-loop watcher to log. Never blocks and
    /// never logs here: a hook that wrote to stdout could stall the worker
    /// it runs on.
    fn report(&self, kind: StallKind, task: TaskKind, meta: &TaskMeta<'_>, duration: Duration) {
        if duration >= STALL_LOG_THRESHOLD {
            let _ = self.stalls.try_send(Stall {
                kind,
                task,
                spawned_at: meta.spawned_at(),
                task_id: meta.id(),
                duration,
            });
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum StallKind {
    SlowPoll,
    ScheduleDelay,
}

impl StallKind {
    fn label(self) -> &'static str {
        match self {
            StallKind::SlowPoll => "slow_poll",
            StallKind::ScheduleDelay => "schedule_delay",
        }
    }
}

struct Stall {
    kind: StallKind,
    task: TaskKind,
    spawned_at: &'static Location<'static>,
    task_id: tokio::task::Id,
    duration: Duration,
}

/// Time a 10 ms sleep forever, recording how late each one wakes, and log
/// the stalls the task hooks queue. Runs next to the metrics server.
pub(crate) async fn watch_event_loop() {
    let stalls = STALL_EVENTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(min) = TASK_METRICS_MIN.get() {
        tracing::info!(
            min_us = min.as_micros(),
            "Recording slow task polls and schedule delays ({TASK_METRICS_ENV})"
        );
    }
    loop {
        let slept = Instant::now();
        tokio::time::sleep(EVENT_LOOP_TICK).await;
        let late = slept.elapsed().saturating_sub(EVENT_LOOP_TICK);
        metrics::observe_event_loop_delay(late);
        if late >= STALL_LOG_THRESHOLD
            && let Some(suppressed) = STALL_LOG_BUDGET.take()
        {
            tracing::warn!(
                kind = "event_loop",
                duration_ms = millis(late),
                threshold_ms = millis(STALL_LOG_THRESHOLD),
                suppressed,
                "EPP runtime stall"
            );
        }
        if let Some(stalls) = &stalls {
            for stall in stalls.try_iter() {
                log_stall(&stall);
            }
        }
    }
}

fn log_stall(stall: &Stall) {
    let Some(suppressed) = STALL_LOG_BUDGET.take() else {
        return;
    };
    tracing::warn!(
        kind = stall.kind.label(),
        task = stall.task.label(),
        spawned_at = %stall.spawned_at,
        task_id = %stall.task_id,
        duration_ms = millis(stall.duration),
        threshold_ms = millis(STALL_LOG_THRESHOLD),
        suppressed,
        "EPP runtime stall"
    );
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each kind is remembered at its first spawn site, so every test spawns
    /// its prefill-watcher stand-in from this one line.
    fn spawn_watcher<F>(future: F) -> tokio::task::JoinHandle<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        spawn(TaskKind::PrefillWatcher, future)
    }

    /// A task that holds its worker for 5 ms is recorded as a slow poll of
    /// its kind, and a task spawned some other way is labeled `other`.
    #[test]
    fn hooks_record_slow_polls_by_task_kind() {
        metrics::set_served_model("test-model");
        let runtime = build_runtime_with(Some(Duration::from_millis(1))).expect("runtime builds");
        let watcher_before = metrics::task_polls_recorded(TaskKind::PrefillWatcher);
        let other_before = metrics::task_polls_recorded(TaskKind::Other);

        runtime.block_on(async {
            spawn_watcher(async {
                std::thread::sleep(Duration::from_millis(5));
            })
            .await
            .expect("task runs");
            tokio::spawn(async {
                std::thread::sleep(Duration::from_millis(5));
            })
            .await
            .expect("task runs");
        });

        assert!(metrics::task_polls_recorded(TaskKind::PrefillWatcher) > watcher_before);
        assert!(metrics::task_polls_recorded(TaskKind::Other) > other_before);
    }

    #[test]
    fn task_kinds_come_from_the_remembered_spawn_site() {
        let runtime = build_runtime_with(None).expect("runtime builds");
        runtime.block_on(async {
            spawn_watcher(async {}).await.expect("task runs");
        });
        let site = TaskKind::PrefillWatcher
            .spawn_site()
            .and_then(OnceLock::get)
            .copied()
            .expect("spawn remembers its caller");
        assert_eq!(TaskKind::of(site), TaskKind::PrefillWatcher);
        assert_eq!(TaskKind::of(Location::caller()), TaskKind::Other);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn off_runtime_runs_the_work_on_the_multi_thread_runtime() {
        assert_eq!(off_runtime(|| 6 * 7), 42);
    }

    #[tokio::test]
    async fn off_runtime_runs_inline_on_the_current_thread_runtime() {
        assert_eq!(off_runtime(|| 6 * 7), 42);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn event_loop_watcher_records_its_delay() {
        metrics::set_served_model("test-model");
        let before = metrics::event_loop_delays_recorded();
        let watcher = tokio::spawn(watch_event_loop());
        tokio::time::sleep(Duration::from_millis(60)).await;
        watcher.abort();
        assert!(metrics::event_loop_delays_recorded() > before);
    }
}

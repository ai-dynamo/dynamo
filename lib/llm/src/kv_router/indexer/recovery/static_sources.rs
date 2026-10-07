// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: static direct-ZMQ KV sources for the e2e indexer-contention campaign.
//!
//! [`STATIC_KV_SOURCES_ENV`] lists KV event publishers that the direct-ZMQ ingress subscribes to
//! without discovery and whose events it accepts without the serving-membership check. Entries
//! are separated by commas or whitespace:
//!
//! - `<worker_id>@<zmq endpoint>`: one source;
//! - `<first_worker_id>+<count>@tcp://<host>:<first_port>`: `count` sources, worker
//!   `first_worker_id + k` publishing on port `first_port + k`.
//!
//! A value starting with `@` names a file of entries (`#` starts a comment). A static source's
//! publisher ID equals its worker ID and its DP rank is 0. Static sources are live-only (no
//! recovery target) and exist only in the ingress's private membership view, so they never
//! become serving workers and are never routable. A static worker or publisher ID that collides
//! with a discovered source is skipped and logged.
//!
//! Static sources require an explicit `DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB`, so both arms of a
//! comparison state their SUB fan-in instead of inheriting the default.
//!
//! Delivery accounting: every event the live path admits to the indexer queue from a static
//! source is counted per source (events, stored and removed blocks, first and last event ID),
//! together with gap resets (`ResetDegraded`), other rank resets after the source had indexed
//! events, and envelopes dropped because the source was inactive. A reporter logs the totals
//! every [`REPORT_INTERVAL_ENV`] seconds and optionally rewrites them as JSON to
//! [`ACCOUNTING_OUT_ENV`]. It adds a `final` report when the subscriber is cancelled, which a
//! signal-terminated router can skip, so read the file three intervals after the last publisher
//! exits. The file also lists every static source's totals, so a checker can compare each
//! phantom's delivery with what its publisher planned.
//!
//! With [`TIMED_START_ENV`] (and optionally [`TIMED_END_ENV`]) the reporter marks the timed
//! window: at each instant it reads the totals, then times a FIFO barrier through the indexer's
//! event queues (`drain_ms`). Delivery inside the window and a drained indexer at both edges can
//! then be checked: admission feeds unbounded queues, so admitted is not applied.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use dynamo_kv_router::protocols::{KvCacheEventData, RouterEvent, WorkerWithDpRank};
use serde::Serialize;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::direct_zmq_sub_pool::{ENDPOINTS_PER_SUB_ENV, endpoints_per_sub_from_env};
use crate::discovery::{
    KvEventSource, KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus,
};
use crate::kv_router::Indexer;

pub(crate) const STATIC_KV_SOURCES_ENV: &str = "DYN_EXPERIMENT_STATIC_KV_SOURCES";
pub(crate) const ACCOUNTING_OUT_ENV: &str = "DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT";
pub(crate) const REPORT_INTERVAL_ENV: &str = "DYN_EXPERIMENT_STATIC_KV_REPORT_S";
pub(crate) const TIMED_START_ENV: &str = "DYN_EXPERIMENT_STATIC_KV_TIMED_START_UNIX_MS";
pub(crate) const TIMED_END_ENV: &str = "DYN_EXPERIMENT_STATIC_KV_TIMED_END_UNIX_MS";
const DEFAULT_REPORT_INTERVAL_S: f64 = 10.0;
const MAX_REPORTED_ANOMALIES: usize = 16;
/// A barrier still queued after this long is reported as a timeout (the arm fell far behind).
const DRAIN_PROBE_TIMEOUT: Duration = Duration::from_secs(600);
/// Columns of each row in a report's `sources` list.
const SOURCE_COLUMNS: [&str; 6] = [
    "worker_id",
    "events",
    "write_blocks",
    "first_event_id",
    "last_event_id",
    "gap_resets",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StaticKvSource {
    pub(crate) worker_id: u64,
    pub(crate) endpoint: String,
}

impl StaticKvSource {
    pub(crate) fn publisher_id(&self) -> u64 {
        self.worker_id
    }
}

/// Parse [`STATIC_KV_SOURCES_ENV`]; unset or empty yields no sources.
pub(crate) fn static_kv_sources_from_env() -> Result<Arc<[StaticKvSource]>> {
    let Some(value) = std::env::var_os(STATIC_KV_SOURCES_ENV) else {
        return Ok(Arc::from([]));
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{STATIC_KV_SOURCES_ENV} must be valid UTF-8"))?;
    let spec = match value.trim().strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("reading {STATIC_KV_SOURCES_ENV} file {path}"))?,
        None => value,
    };
    let sources =
        parse_static_sources(&spec).with_context(|| format!("parsing {STATIC_KV_SOURCES_ENV}"))?;
    if sources.is_empty() {
        return Ok(Arc::from([]));
    }
    ensure!(
        std::env::var_os(ENDPOINTS_PER_SUB_ENV).is_some(),
        "{STATIC_KV_SOURCES_ENV} requires an explicit {ENDPOINTS_PER_SUB_ENV} (use phantom_plan's value, identical in both arms)"
    );
    tracing::warn!(
        count = sources.len(),
        env = STATIC_KV_SOURCES_ENV,
        endpoints_per_sub = endpoints_per_sub_from_env()?,
        "EXPERIMENT: accepting static direct-ZMQ KV sources without serving membership"
    );
    Ok(sources.into())
}

fn parse_static_sources(spec: &str) -> Result<Vec<StaticKvSource>> {
    let mut sources = Vec::new();
    let mut workers = HashSet::new();
    for line in spec.lines() {
        let line = line.split('#').next().unwrap_or_default();
        for entry in line
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|entry| !entry.is_empty())
        {
            for source in parse_entry(entry)? {
                ensure!(
                    workers.insert(source.worker_id),
                    "worker {} is listed twice",
                    source.worker_id
                );
                sources.push(source);
            }
        }
    }
    Ok(sources)
}

fn parse_entry(entry: &str) -> Result<Vec<StaticKvSource>> {
    let (ids, endpoint) = entry
        .split_once('@')
        .with_context(|| format!("entry {entry:?} is not <worker_id>@<endpoint>"))?;
    ensure!(
        !endpoint.is_empty(),
        "entry {entry:?} has an empty endpoint"
    );
    let Some((first, count)) = ids.split_once('+') else {
        let worker_id = parse_u64(ids, entry)?;
        return Ok(vec![StaticKvSource {
            worker_id,
            endpoint: endpoint.to_string(),
        }]);
    };
    let first = parse_u64(first, entry)?;
    let count = parse_u64(count, entry)?;
    let (prefix, port) = endpoint
        .rsplit_once(':')
        .filter(|(prefix, _)| prefix.starts_with("tcp://"))
        .with_context(|| format!("range entry {entry:?} needs a tcp://<host>:<port> endpoint"))?;
    let first_port = port
        .parse::<u16>()
        .with_context(|| format!("range entry {entry:?} has an invalid port"))?;
    ensure!(count > 0, "range entry {entry:?} has a zero count");
    if u64::from(first_port) + count - 1 > u64::from(u16::MAX) {
        bail!("range entry {entry:?} runs past port 65535");
    }
    (0..count)
        .map(|offset| {
            let worker_id = first
                .checked_add(offset)
                .with_context(|| format!("range entry {entry:?} overflows the worker ID"))?;
            Ok(StaticKvSource {
                worker_id,
                endpoint: format!("{prefix}:{}", u64::from(first_port) + offset),
            })
        })
        .collect()
}

fn parse_u64(value: &str, entry: &str) -> Result<u64> {
    let value = value.trim();
    let parsed = match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse::<u64>(),
    };
    parsed.with_context(|| format!("entry {entry:?} has an invalid number {value:?}"))
}

/// Derive a membership channel that adds the static sources as active live-only sources.
pub(crate) fn with_static_sources(
    membership: KvSourceMembershipWatch,
    sources: Arc<[StaticKvSource]>,
    cancel: CancellationToken,
) -> KvSourceMembershipWatch {
    let mut source = membership.clone();
    let mut reported = HashSet::new();
    let initial = augment_view(source.borrow().clone(), &sources, &mut reported);
    let (tx, rx) = watch::channel(initial);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                changed = source.changed() => if changed.is_err() { break; },
            }
            let view = augment_view(source.borrow_and_update().clone(), &sources, &mut reported);
            tx.send_replace(view);
        }
    });
    membership.with_receiver(rx)
}

fn augment_view(
    mut view: KvSourceMembershipView,
    sources: &[StaticKvSource],
    reported: &mut HashSet<u64>,
) -> KvSourceMembershipView {
    let Some(kv_state_endpoint) = view.resolved_kv_state_endpoint().cloned() else {
        return view;
    };
    let discovered_workers: HashSet<u64> =
        view.sources.keys().map(|worker| worker.worker_id).collect();
    let discovered_publishers: HashSet<u64> = view
        .sources
        .values()
        .filter_map(|status| status.active_source().map(|source| source.publisher_id))
        .collect();
    let mut added = HashMap::with_capacity(sources.len());
    for source in sources {
        if discovered_workers.contains(&source.worker_id)
            || discovered_publishers.contains(&source.publisher_id())
        {
            if reported.insert(source.worker_id) {
                tracing::error!(
                    worker_id = source.worker_id,
                    "EXPERIMENT: static KV source collides with a discovered source; skipping it"
                );
            }
            continue;
        }
        let worker = WorkerWithDpRank::new(source.worker_id, 0);
        added.insert(
            worker,
            KvSourceStatus::ActiveLiveOnly(KvEventSource {
                kv_state_endpoint: kv_state_endpoint.clone(),
                worker,
                publisher_id: source.publisher_id(),
                recovery_target: None,
            }),
        );
    }
    for (worker, status) in added {
        view.kv_event_publishing_enabled
            .insert(worker.worker_id, Some(true));
        view.recovery_expected.insert(worker, false);
        view.sources.insert(worker, status);
    }
    view
}

static ACCOUNTING: OnceLock<Arc<StaticSourceAccounting>> = OnceLock::new();

/// One static source's admitted totals. Its envelopes are handled serially by its own source
/// task, so a cache line per source keeps the hot path free of cross-source contention.
#[repr(align(64))]
#[derive(Debug, Default)]
struct SourceAccount {
    first_event_id: AtomicU64,
    last_event_id: AtomicU64,
    events: AtomicU64,
    stored_blocks: AtomicU64,
    removed_blocks: AtomicU64,
    gap_resets: AtomicU64,
    rank_resets: AtomicU64,
    dropped_events: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(crate) struct DeliveredTotals {
    pub(crate) events: u64,
    pub(crate) stored_blocks: u64,
    pub(crate) removed_blocks: u64,
    pub(crate) write_blocks: u64,
}

impl DeliveredTotals {
    fn minus(self, earlier: Self) -> Self {
        Self {
            events: self.events.saturating_sub(earlier.events),
            stored_blocks: self.stored_blocks.saturating_sub(earlier.stored_blocks),
            removed_blocks: self.removed_blocks.saturating_sub(earlier.removed_blocks),
            write_blocks: self.write_blocks.saturating_sub(earlier.write_blocks),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AnomalousSource {
    worker_id: u64,
    first_event_id: u64,
    last_event_id: u64,
    events: u64,
    gap_resets: u64,
    rank_resets: u64,
    dropped_events: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(crate) struct AccountingSnapshot {
    pub(crate) totals: DeliveredTotals,
    pub(crate) sources_with_events: u64,
    /// Sources whose first admitted event was not event 1: their prefix was lost before the
    /// indexer applied it (live-only sources treat the first event they see as initial).
    pub(crate) sources_first_event_late: u64,
    pub(crate) gap_resets: u64,
    pub(crate) rank_resets: u64,
    pub(crate) dropped_events: u64,
    anomalous_sources: Vec<AnomalousSource>,
}

/// Experiment-only delivery accounting for static KV sources (identical in both arms).
pub(crate) struct StaticSourceAccounting {
    worker_ids: Box<[u64]>,
    slots: HashMap<u64, usize>,
    accounts: Box<[SourceAccount]>,
    endpoints_per_sub: usize,
    /// Cancelled at shutdown; teardown fences after it are not counted as rank resets.
    cancel: CancellationToken,
}

impl StaticSourceAccounting {
    fn new(
        sources: &[StaticKvSource],
        endpoints_per_sub: usize,
        cancel: CancellationToken,
    ) -> Self {
        let worker_ids: Box<[u64]> = sources.iter().map(|source| source.worker_id).collect();
        let slots = worker_ids
            .iter()
            .enumerate()
            .map(|(slot, &worker_id)| (worker_id, slot))
            .collect();
        Self {
            accounts: worker_ids
                .iter()
                .map(|_| SourceAccount::default())
                .collect(),
            worker_ids,
            slots,
            endpoints_per_sub,
            cancel,
        }
    }

    fn account(&self, publisher_id: u64) -> Option<usize> {
        self.slots.get(&publisher_id).copied()
    }

    fn record_rank_reset(&self, publisher_id: u64) {
        let Some(slot) = self.account(publisher_id) else {
            return;
        };
        let account = &self.accounts[slot];
        if self.cancel.is_cancelled() || account.last_event_id.load(Ordering::Relaxed) == 0 {
            return;
        }
        account.rank_resets.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            worker_id = self.worker_ids[slot],
            "EXPERIMENT: a static KV source's rank was reset after it had indexed events"
        );
    }

    fn totals(&self) -> DeliveredTotals {
        let mut totals = DeliveredTotals::default();
        for account in self.accounts.iter() {
            totals.events += account.events.load(Ordering::Relaxed);
            totals.stored_blocks += account.stored_blocks.load(Ordering::Relaxed);
            totals.removed_blocks += account.removed_blocks.load(Ordering::Relaxed);
        }
        totals.write_blocks = totals.stored_blocks + totals.removed_blocks;
        totals
    }

    fn snapshot(&self) -> AccountingSnapshot {
        let mut snapshot = AccountingSnapshot {
            totals: self.totals(),
            ..AccountingSnapshot::default()
        };
        for (account, &worker_id) in self.accounts.iter().zip(self.worker_ids.iter()) {
            let source = AnomalousSource {
                worker_id,
                first_event_id: account.first_event_id.load(Ordering::Relaxed),
                last_event_id: account.last_event_id.load(Ordering::Relaxed),
                events: account.events.load(Ordering::Relaxed),
                gap_resets: account.gap_resets.load(Ordering::Relaxed),
                rank_resets: account.rank_resets.load(Ordering::Relaxed),
                dropped_events: account.dropped_events.load(Ordering::Relaxed),
            };
            snapshot.sources_with_events += u64::from(source.events > 0);
            let late = source.events > 0 && source.first_event_id != 1;
            snapshot.sources_first_event_late += u64::from(late);
            snapshot.gap_resets += source.gap_resets;
            snapshot.rank_resets += source.rank_resets;
            snapshot.dropped_events += source.dropped_events;
            let anomalous = late
                || source.gap_resets > 0
                || source.rank_resets > 0
                || source.dropped_events > 0;
            if anomalous && snapshot.anomalous_sources.len() < MAX_REPORTED_ANOMALIES {
                snapshot.anomalous_sources.push(source);
            }
        }
        snapshot
    }

    /// One row per static source, in [`SOURCE_COLUMNS`] order.
    fn source_rows(&self) -> Vec<[u64; 6]> {
        self.accounts
            .iter()
            .zip(self.worker_ids.iter())
            .map(|(account, &worker_id)| {
                [
                    worker_id,
                    account.events.load(Ordering::Relaxed),
                    account.stored_blocks.load(Ordering::Relaxed)
                        + account.removed_blocks.load(Ordering::Relaxed),
                    account.first_event_id.load(Ordering::Relaxed),
                    account.last_event_id.load(Ordering::Relaxed),
                    account.gap_resets.load(Ordering::Relaxed),
                ]
            })
            .collect()
    }

    /// The report the reporter logs; [`Self::file_report`] adds the per-source rows.
    fn report(&self, kind: &str, interval_s: f64, window: &TimedWindow) -> serde_json::Value {
        let snapshot = self.snapshot();
        let warmup = window.start.as_ref().map(|start| start.totals);
        // The timed window's delivery: up to the end mark once taken, otherwise so far.
        let timed = window.start.as_ref().map(|start| {
            let until = window
                .end
                .as_ref()
                .map_or(snapshot.totals, |end| end.totals);
            until.minus(start.totals)
        });
        let after_end = window
            .end
            .as_ref()
            .map(|end| snapshot.totals.minus(end.totals));
        serde_json::json!({
            "kind": kind,
            "t_unix_ms": unix_now_ms(),
            "report_interval_s": interval_s,
            "static_sources": self.worker_ids.len(),
            "endpoints_per_sub": self.endpoints_per_sub,
            "timed_start_unix_ms": window.start_at_unix_ms,
            "timed_end_unix_ms": window.end_at_unix_ms,
            "start": window.start,
            "end": window.end,
            "warmup": warmup,
            "timed": timed,
            "after_end": after_end,
            "accounting": snapshot,
        })
    }

    fn file_report(&self, mut report: serde_json::Value) -> serde_json::Value {
        report["source_columns"] = serde_json::json!(SOURCE_COLUMNS);
        report["sources"] = serde_json::json!(self.source_rows());
        report
    }
}

/// The admitted totals at one edge of the timed window, and how long the indexer's event
/// queues took to complete everything queued before it.
#[derive(Debug, Clone, Serialize)]
struct WindowMark {
    /// The instant the environment asked for.
    requested_unix_ms: u64,
    /// When the totals were actually read (the reporter runs on a loaded runtime).
    taken_unix_ms: u64,
    totals: DeliveredTotals,
    /// Milliseconds for a FIFO barrier to pass every event-thread queue: about zero when the
    /// indexer kept up, the backlog's drain time when it did not.
    drain_ms: Option<f64>,
    drain_error: Option<String>,
}

#[derive(Debug, Default)]
struct TimedWindow {
    start_at_unix_ms: Option<u64>,
    end_at_unix_ms: Option<u64>,
    start: Option<WindowMark>,
    end: Option<WindowMark>,
}

async fn take_mark(
    accounting: &StaticSourceAccounting,
    indexer: &Indexer,
    requested_unix_ms: u64,
) -> WindowMark {
    let taken_unix_ms = unix_now_ms();
    let totals = accounting.totals();
    let (drain_ms, drain_error) = match probe_drain(indexer).await {
        Ok(ms) => (Some(ms), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };
    WindowMark {
        requested_unix_ms,
        taken_unix_ms,
        totals,
        drain_ms,
        drain_error,
    }
}

/// Time a FIFO barrier through every event-thread queue of the indexer.
async fn probe_drain(indexer: &Indexer) -> Result<f64> {
    let Indexer::Concurrent { primary, .. } = indexer else {
        bail!("only the thread-pool indexer (router event threads > 1) can be drain-probed");
    };
    let started = std::time::Instant::now();
    tokio::time::timeout(DRAIN_PROBE_TIMEOUT, primary.flush_and_wait())
        .await
        .with_context(|| format!("still draining after {DRAIN_PROBE_TIMEOUT:?}"))??;
    Ok(started.elapsed().as_secs_f64() * 1e3)
}

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

fn unix_ms_from_env(name: &str) -> Result<Option<u64>> {
    std::env::var(name)
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()
        .with_context(|| format!("{name} must be unix milliseconds"))
}

/// Install the process-wide accounting for `sources` and start its reporter.
pub(crate) fn start_static_source_accounting(
    sources: &[StaticKvSource],
    indexer: Indexer,
    cancel: CancellationToken,
) -> Result<()> {
    let interval_s = match std::env::var(REPORT_INTERVAL_ENV) {
        Ok(value) => value
            .parse::<f64>()
            .ok()
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .with_context(|| format!("{REPORT_INTERVAL_ENV} must be a positive number"))?,
        Err(_) => DEFAULT_REPORT_INTERVAL_S,
    };
    let timed_start_unix_ms = unix_ms_from_env(TIMED_START_ENV)?;
    let timed_end_unix_ms = unix_ms_from_env(TIMED_END_ENV)?;
    if let Some(end) = timed_end_unix_ms {
        ensure!(
            timed_start_unix_ms.is_some_and(|start| start < end),
            "{TIMED_END_ENV} needs an earlier {TIMED_START_ENV}"
        );
    }
    let out = std::env::var_os(ACCOUNTING_OUT_ENV).map(PathBuf::from);
    let accounting = Arc::new(StaticSourceAccounting::new(
        sources,
        endpoints_per_sub_from_env()?,
        cancel.clone(),
    ));
    if ACCOUNTING.set(accounting.clone()).is_err() {
        tracing::warn!(
            "EXPERIMENT: static KV source accounting already runs in this process; not starting another"
        );
        return Ok(());
    }
    tokio::spawn(run_reporter(
        accounting,
        indexer,
        interval_s,
        TimedWindow {
            start_at_unix_ms: timed_start_unix_ms,
            end_at_unix_ms: timed_end_unix_ms,
            ..TimedWindow::default()
        },
        out,
        cancel,
    ));
    Ok(())
}

/// A sleep until `at_unix_ms`, or `None` (logged) when that instant passed before startup.
fn sleep_until_unix_ms(
    at_unix_ms: Option<u64>,
    now_ms: u64,
    env: &str,
) -> Option<std::pin::Pin<Box<tokio::time::Sleep>>> {
    let at = at_unix_ms?;
    if at <= now_ms {
        tracing::error!(
            requested_unix_ms = at,
            now_unix_ms = now_ms,
            env,
            "EXPERIMENT: a timed-window edge passed before the indexer started; it is not marked"
        );
        return None;
    }
    Some(Box::pin(tokio::time::sleep(Duration::from_millis(
        at - now_ms,
    ))))
}

async fn run_reporter(
    accounting: Arc<StaticSourceAccounting>,
    indexer: Indexer,
    interval_s: f64,
    mut window: TimedWindow,
    out: Option<PathBuf>,
    cancel: CancellationToken,
) {
    let now_ms = unix_now_ms();
    let mut start_sleep = sleep_until_unix_ms(window.start_at_unix_ms, now_ms, TIMED_START_ENV);
    let mut end_sleep = sleep_until_unix_ms(window.end_at_unix_ms, now_ms, TIMED_END_ENV);
    if start_sleep.is_none() {
        end_sleep = None;
    }
    let mut ticker = tokio::time::interval(Duration::from_secs_f64(interval_s));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        let kind = tokio::select! {
            biased;
            _ = cancel.cancelled() => "final",
            _ = async { start_sleep.as_mut().expect("guarded").await }, if start_sleep.is_some() => {
                start_sleep = None;
                let at = window.start_at_unix_ms.expect("a start sleep has a start");
                window.start = Some(take_mark(&accounting, &indexer, at).await);
                "split"
            }
            _ = async { end_sleep.as_mut().expect("guarded").await },
                if end_sleep.is_some() && start_sleep.is_none() => {
                end_sleep = None;
                let at = window.end_at_unix_ms.expect("an end sleep has an end");
                window.end = Some(take_mark(&accounting, &indexer, at).await);
                "end"
            }
            _ = ticker.tick() => "interval",
        };
        let report = accounting.report(kind, interval_s, &window);
        tracing::warn!(report = %report, "EXPERIMENT static KV source accounting");
        if let Some(path) = &out
            && let Err(error) = write_report(path, &accounting.file_report(report))
        {
            tracing::error!(%error, path = %path.display(), "EXPERIMENT: failed to write the static KV source accounting");
        }
        if kind == "final" {
            break;
        }
    }
}

fn write_report(path: &Path, report: &serde_json::Value) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec(report)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Count a rank reset of a static source after it had indexed events (outside shutdown).
pub(crate) fn record_static_rank_reset(publisher_id: u64) {
    if let Some(accounting) = ACCOUNTING.get() {
        accounting.record_rank_reset(publisher_id);
    }
}

/// Stored and removed blocks one event writes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WrittenBlocks {
    stored: u64,
    removed: u64,
}

/// Accumulates one envelope's outcome for a static source and publishes it on drop, so the
/// live path touches the shared counters once per envelope. A no-op for other sources.
pub(crate) struct StaticBatchTally {
    target: Option<(&'static StaticSourceAccounting, usize)>,
    first_event_id: u64,
    last_event_id: u64,
    events: u64,
    stored_blocks: u64,
    removed_blocks: u64,
    gap_resets: u64,
    dropped_events: u64,
}

impl StaticBatchTally {
    pub(crate) fn new(publisher_id: u64) -> Self {
        Self::with(
            ACCOUNTING.get().map(|accounting| &**accounting),
            publisher_id,
        )
    }

    fn with(accounting: Option<&'static StaticSourceAccounting>, publisher_id: u64) -> Self {
        Self {
            target: accounting.and_then(|accounting| {
                accounting
                    .account(publisher_id)
                    .map(|slot| (accounting, slot))
            }),
            first_event_id: 0,
            last_event_id: 0,
            events: 0,
            stored_blocks: 0,
            removed_blocks: 0,
            gap_resets: 0,
            dropped_events: 0,
        }
    }

    pub(crate) fn written(&self, event: &RouterEvent) -> WrittenBlocks {
        if self.target.is_none() {
            return WrittenBlocks::default();
        }
        match &event.event.data {
            KvCacheEventData::Stored(store) => WrittenBlocks {
                stored: store.blocks.len() as u64,
                removed: 0,
            },
            KvCacheEventData::Removed(remove) => WrittenBlocks {
                stored: 0,
                removed: remove.block_hashes.len() as u64,
            },
            KvCacheEventData::Cleared => WrittenBlocks::default(),
        }
    }

    pub(crate) fn admitted(&mut self, event_id: u64, written: WrittenBlocks) {
        if self.target.is_none() {
            return;
        }
        if self.events == 0 {
            self.first_event_id = event_id;
        }
        self.last_event_id = event_id;
        self.events += 1;
        self.stored_blocks += written.stored;
        self.removed_blocks += written.removed;
    }

    pub(crate) fn gap_reset(&mut self) {
        self.gap_resets += 1;
    }

    pub(crate) fn dropped(&mut self, events: usize) {
        self.dropped_events += events as u64;
    }
}

impl Drop for StaticBatchTally {
    fn drop(&mut self) {
        let Some((accounting, slot)) = self.target else {
            return;
        };
        let account = &accounting.accounts[slot];
        if self.events > 0 {
            let first_seen = account
                .first_event_id
                .compare_exchange(0, self.first_event_id, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok();
            if first_seen && self.first_event_id != 1 {
                tracing::warn!(
                    worker_id = accounting.worker_ids[slot],
                    first_event_id = self.first_event_id,
                    "EXPERIMENT: a static KV source's first admitted event is not event 1; its prefix was lost before the indexer applied it"
                );
            }
            account
                .last_event_id
                .store(self.last_event_id, Ordering::Relaxed);
            account.events.fetch_add(self.events, Ordering::Relaxed);
            account
                .stored_blocks
                .fetch_add(self.stored_blocks, Ordering::Relaxed);
            account
                .removed_blocks
                .fetch_add(self.removed_blocks, Ordering::Relaxed);
        }
        if self.gap_resets > 0 {
            account
                .gap_resets
                .fetch_add(self.gap_resets, Ordering::Relaxed);
        }
        if self.dropped_events > 0 {
            account
                .dropped_events
                .fetch_add(self.dropped_events, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use dynamo_kv_router::indexer::WorkerKvQueryResponse;
    use dynamo_kv_router::protocols::{
        DpRank, ExternalSequenceBlockHash, KvCacheEvent, KvCacheRemoveData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, WorkerId,
    };
    use dynamo_runtime::component::Instance;
    use dynamo_runtime::protocols::EndpointId;

    use super::super::target::{RecoveryResetReason, RecoveryTarget};
    use super::super::worker_query::WorkerQueryClient;
    use super::super::worker_query_transport::WorkerQueryTransport;
    use super::*;
    use crate::discovery::{KvStateEndpointResolution, PublisherId};

    fn endpoint() -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: "backend".to_string(),
            name: "generate".to_string(),
        }
    }

    fn view_with(worker_id: u64, publisher_id: u64) -> KvSourceMembershipView {
        let worker = WorkerWithDpRank::new(worker_id, 0);
        KvSourceMembershipView {
            serving_endpoint: endpoint(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint()),
            sources: HashMap::from([(
                worker,
                KvSourceStatus::ActiveLiveOnly(KvEventSource {
                    kv_state_endpoint: endpoint(),
                    worker,
                    publisher_id,
                    recovery_target: None,
                }),
            )]),
            kv_event_publishing_enabled: HashMap::from([(worker_id, Some(true))]),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::from([(worker, false)]),
        }
    }

    #[test]
    fn parses_singles_ranges_and_comments() {
        let sources = parse_static_sources(
            "7@tcp://10.0.0.1:5000, 0x10+3@tcp://host-a:6000\n# comment\n100@tcp://h:1 # tail",
        )
        .unwrap();
        let ids: Vec<_> = sources.iter().map(|source| source.worker_id).collect();
        assert_eq!(ids, vec![7, 16, 17, 18, 100]);
        assert_eq!(sources[1].endpoint, "tcp://host-a:6000");
        assert_eq!(sources[3].endpoint, "tcp://host-a:6002");
        assert_eq!(sources[3].publisher_id(), 18);
    }

    #[test]
    fn rejects_duplicates_and_bad_ranges() {
        assert!(parse_static_sources("1@tcp://a:1 1@tcp://b:2").is_err());
        assert!(parse_static_sources("1+2@tcp://a:65535").is_err());
        assert!(parse_static_sources("1+0@tcp://a:1").is_err());
        assert!(parse_static_sources("1+2@ipc://sock").is_err());
        assert!(parse_static_sources("tcp://a:1").is_err());
    }

    #[test]
    fn augment_adds_live_only_sources_and_skips_collisions() {
        let sources = parse_static_sources("5+3@tcp://h:7000").unwrap();
        // Worker 5 is discovered; publisher 7 belongs to a discovered source.
        let mut view = view_with(5, 99);
        let other = WorkerWithDpRank::new(42, 0);
        view.sources.insert(
            other,
            KvSourceStatus::ActiveLiveOnly(KvEventSource {
                kv_state_endpoint: endpoint(),
                worker: other,
                publisher_id: 7,
                recovery_target: None,
            }),
        );
        let mut reported = HashSet::new();
        let augmented = augment_view(view, &sources, &mut reported);

        assert_eq!(reported, HashSet::from([5, 7]));
        let phantom = WorkerWithDpRank::new(6, 0);
        let Some(KvSourceStatus::ActiveLiveOnly(source)) = augmented.sources.get(&phantom) else {
            panic!("static worker 6 should be an active live-only source");
        };
        assert_eq!(source.publisher_id, 6);
        assert_eq!(source.kv_state_endpoint, endpoint());
        assert_eq!(augmented.kv_event_publishing_enabled(6), Some(true));
        assert_eq!(augmented.recovery_expected(&phantom), Some(false));
        // The discovered source for worker 5 is untouched.
        let Some(KvSourceStatus::ActiveLiveOnly(source)) =
            augmented.sources.get(&WorkerWithDpRank::new(5, 0))
        else {
            panic!("discovered worker 5 should remain");
        };
        assert_eq!(source.publisher_id, 99);
        assert_eq!(augmented.sources.len(), 3);
    }

    #[test]
    fn ambiguous_endpoint_adds_nothing() {
        let sources = parse_static_sources("5@tcp://h:7000").unwrap();
        let mut view = view_with(1, 1);
        view.endpoint_resolution = KvStateEndpointResolution::Ambiguous {
            endpoints: vec![endpoint()],
        };
        let augmented = augment_view(view.clone(), &sources, &mut HashSet::new());
        assert_eq!(augmented, view);
    }
    fn store(worker_id: u64, event_id: u64, blocks: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: (0..blocks)
                        .map(|block| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(event_id * 1000 + block),
                            tokens_hash: LocalBlockHash(block),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    fn remove(worker_id: u64, event_id: u64, blocks: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: (0..blocks).map(ExternalSequenceBlockHash).collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    fn admit(tally: &mut StaticBatchTally, event: &RouterEvent) {
        let written = tally.written(event);
        tally.admitted(event.event.event_id, written);
    }

    #[test]
    fn tallies_publish_admissions_late_prefixes_and_anomalies() {
        let sources = parse_static_sources("10+3@tcp://h:7000").unwrap();
        let accounting: &'static StaticSourceAccounting = Box::leak(Box::new(
            StaticSourceAccounting::new(&sources, 4, CancellationToken::new()),
        ));

        let mut tally = StaticBatchTally::with(Some(accounting), 10);
        admit(&mut tally, &store(10, 1, 3));
        admit(&mut tally, &remove(10, 2, 2));
        drop(tally);
        let mut tally = StaticBatchTally::with(Some(accounting), 10);
        tally.gap_reset();
        admit(&mut tally, &store(10, 5, 1));
        drop(tally);
        // Worker 11 lost its prefix: the first event the indexer applied is 4.
        let mut tally = StaticBatchTally::with(Some(accounting), 11);
        admit(&mut tally, &store(11, 4, 2));
        drop(tally);
        StaticBatchTally::with(Some(accounting), 12).dropped(7);
        // Not a static source: nothing is recorded.
        admit(
            &mut StaticBatchTally::with(Some(accounting), 99),
            &store(99, 1, 5),
        );

        let snapshot = accounting.snapshot();
        assert_eq!(
            snapshot.totals,
            DeliveredTotals {
                events: 4,
                stored_blocks: 6,
                removed_blocks: 2,
                write_blocks: 8,
            }
        );
        assert_eq!(snapshot.sources_with_events, 2);
        assert_eq!(snapshot.sources_first_event_late, 1);
        assert_eq!(snapshot.gap_resets, 1);
        assert_eq!(snapshot.dropped_events, 7);
        let anomalous: Vec<_> = snapshot
            .anomalous_sources
            .iter()
            .map(|source| source.worker_id)
            .collect();
        assert_eq!(anomalous, vec![10, 11, 12]);
        assert_eq!(snapshot.anomalous_sources[0].first_event_id, 1);
        assert_eq!(snapshot.anomalous_sources[0].last_event_id, 5);

        // Resets count only after the source indexed events and only before shutdown.
        accounting.record_rank_reset(12);
        accounting.record_rank_reset(10);
        assert_eq!(accounting.snapshot().rank_resets, 1);
        accounting.cancel.cancel();
        accounting.record_rank_reset(10);
        assert_eq!(accounting.snapshot().rank_resets, 1);

        let mark = |write_blocks| WindowMark {
            requested_unix_ms: 1,
            taken_unix_ms: 2,
            totals: DeliveredTotals {
                events: 1,
                stored_blocks: write_blocks,
                removed_blocks: 0,
                write_blocks,
            },
            drain_ms: Some(0.5),
            drain_error: None,
        };
        let mut window = TimedWindow {
            start_at_unix_ms: Some(1),
            end_at_unix_ms: Some(3),
            start: Some(mark(3)),
            end: None,
        };
        let report = accounting.report("split", 10.0, &window);
        assert_eq!(
            report["timed"]["write_blocks"], 5,
            "running until the end mark"
        );
        assert_eq!(report["accounting"]["totals"]["events"], 4);
        assert_eq!(report["endpoints_per_sub"], 4);
        assert!(report["after_end"].is_null());
        assert!(report.get("sources").is_none(), "the log line stays small");

        window.end = Some(mark(6));
        let report = accounting.file_report(accounting.report("end", 10.0, &window));
        assert_eq!(report["timed"]["write_blocks"], 3, "start to end mark");
        assert_eq!(report["after_end"]["write_blocks"], 2);
        assert_eq!(report["end"]["drain_ms"], 0.5);
        assert_eq!(report["source_columns"][2], "write_blocks");
        let row10 = &report["sources"][0];
        assert_eq!(row10[0], 10);
        assert_eq!((row10[1].clone(), row10[2].clone()), (3.into(), 6.into()));
        assert_eq!((row10[3].clone(), row10[4].clone()), (1.into(), 5.into()));
        assert_eq!(
            report["sources"][1][3], 4,
            "worker 11's first admitted event"
        );
    }

    #[tokio::test]
    async fn drain_probe_times_the_thread_pool_and_rejects_other_indexers() {
        use dynamo_kv_router::ConcurrentRadixTreeCompressed;
        use dynamo_kv_router::indexer::{LowerTierIndexers, ThreadPoolIndexer};

        let primary = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            2,
            16,
        ));
        let indexer = Indexer::Concurrent {
            primary,
            lower_tier: LowerTierIndexers::new_with_metrics(2, 16, None),
            approx: None,
            primary_records_routing_decisions: false,
            session_updates: None,
        };
        let drain_ms = probe_drain(&indexer).await.unwrap();
        assert!((0.0..DRAIN_PROBE_TIMEOUT.as_secs_f64() * 1e3).contains(&drain_ms));
        assert!(probe_drain(&Indexer::None).await.is_err());
    }

    #[derive(Clone, Default)]
    struct AcceptingTarget;

    impl RecoveryTarget for AcceptingTarget {
        async fn admit_event(&self, _: PublisherId, _: RouterEvent) -> anyhow::Result<()> {
            Ok(())
        }

        async fn replace_rank(
            &self,
            _: PublisherId,
            _: WorkerId,
            _: DpRank,
            _: Vec<RouterEvent>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn reset_rank(
            &self,
            _: PublisherId,
            _: WorkerId,
            _: DpRank,
            _: RecoveryResetReason,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct NoRecoveryTransport;

    #[async_trait]
    impl WorkerQueryTransport for NoRecoveryTransport {
        async fn query_worker(
            &self,
            _: WorkerId,
            _: DpRank,
            _: Instance,
            _: Option<u64>,
            _: Option<u64>,
        ) -> Result<WorkerKvQueryResponse> {
            bail!("static sources are live-only")
        }
    }

    /// The only test that installs the process-wide accounting; its worker IDs are disjoint from
    /// every other test's publishers.
    #[tokio::test]
    async fn live_path_counts_static_admissions_gap_resets_and_inactive_drops() {
        let first = 0x7E00_0000_0000_0000u64;
        let sources = parse_static_sources(&format!("{first}+2@tcp://h:7000")).unwrap();
        let accounting = Arc::new(StaticSourceAccounting::new(
            &sources,
            1,
            CancellationToken::new(),
        ));
        assert!(ACCOUNTING.set(accounting.clone()).is_ok());

        let mut view = view_with(1, 1);
        view.sources.clear();
        let view = augment_view(view, &sources, &mut HashSet::new());
        let (_tx, rx) = watch::channel(view);
        let client = WorkerQueryClient::new_target_for_test(
            AcceptingTarget,
            rx,
            Arc::new(NoRecoveryTransport),
        );
        // Before activation the envelope is dropped (and counted).
        client
            .handle_live_batch(first, vec![store(first, 1, 2)])
            .await;
        client.sync_membership().await;

        client
            .handle_live_batch(first, vec![store(first, 1, 2), remove(first, 2, 1)])
            .await;
        // Event 3 never arrives: a live-only gap resets the rank and applies event 4.
        client
            .handle_live_batch(first, vec![store(first, 4, 1)])
            .await;

        let snapshot = accounting.snapshot();
        assert_eq!(
            snapshot.totals,
            DeliveredTotals {
                events: 3,
                stored_blocks: 3,
                removed_blocks: 1,
                write_blocks: 4,
            }
        );
        assert_eq!(snapshot.gap_resets, 1);
        assert_eq!(
            snapshot.rank_resets, 1,
            "the gap reset discarded indexed state"
        );
        assert_eq!(snapshot.dropped_events, 1);
        assert_eq!(snapshot.sources_with_events, 1);
        assert_eq!(snapshot.sources_first_event_late, 0);
    }
}

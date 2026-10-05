// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-scaling diagnostics for the Mooncake open-loop replay: per-thread CPU accounting,
//! process memory, workload shape per inference worker, a do-nothing indexer backend that
//! bounds the harness ceiling, and an independent reference index for score checks.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::{
    KvIndexerMetrics, SyncIndexer, WorkerLookupStats, WorkerObservationState, WorkerTask,
};
use dynamo_kv_router::protocols::{KvCacheEventData, OverlapScores, RouterEvent, WorkerWithDpRank};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;

use super::mooncake_shared::{PreparedMooncakeBenchmark, WorkerTraceEntry};

pub const THREAD_NAME_EVENT: &str = "mc-event";
pub const THREAD_NAME_TOKIO: &str = "mc-tokio";
pub const THREAD_NAME_QUERY_ISSUER: &str = "mc-qissuer";
pub const THREAD_NAME_EVENT_ISSUER: &str = "mc-eissuer";

/// Set the calling thread's kernel name. Threads spawned afterwards inherit it, which is how
/// the indexer's event threads (spawned inside `dynamo-kv-router`) get a recognizable name.
pub fn set_current_thread_name(name: &str) {
    #[cfg(target_os = "linux")]
    {
        let mut bytes = name.as_bytes().iter().copied().take(15).collect::<Vec<_>>();
        bytes.push(0);
        unsafe {
            libc::prctl(libc::PR_SET_NAME, bytes.as_ptr() as libc::c_ulong, 0, 0, 0);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = name;
}

/// The calling thread's kernel name, or an empty string when unavailable.
pub fn current_thread_name() -> String {
    std::fs::read_to_string("/proc/thread-self/comm")
        .map(|name| name.trim().to_string())
        .unwrap_or_default()
}

#[derive(Clone, Debug)]
pub struct ThreadCpuSample {
    pub tid: i64,
    pub name: String,
    pub cpu_ns: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ProcessThreadSnapshot {
    pub taken_ns: u64,
    pub threads: Vec<ThreadCpuSample>,
    pub source: &'static str,
}

fn thread_role(tid: i64, pid: i64, name: &str) -> &'static str {
    if tid == pid {
        return "main";
    }
    match name {
        THREAD_NAME_EVENT => "indexer_event",
        THREAD_NAME_TOKIO => "tokio_runtime",
        THREAD_NAME_QUERY_ISSUER => "query_issuer",
        _ if name.starts_with(THREAD_NAME_EVENT_ISSUER) => "event_issuer",
        _ => "other",
    }
}

#[cfg(target_os = "linux")]
fn clock_ticks_per_sec() -> u64 {
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks > 0 { ticks as u64 } else { 100 }
}

/// Per-thread CPU time from `schedstat` (ns resolution), falling back to `stat` utime+stime.
#[cfg(target_os = "linux")]
fn read_thread_cpu_ns(task_dir: &std::path::Path) -> Option<(u64, &'static str)> {
    if let Ok(schedstat) = std::fs::read_to_string(task_dir.join("schedstat"))
        && let Some(run_ns) = schedstat
            .split_whitespace()
            .next()
            .and_then(|v| v.parse().ok())
    {
        return Some((run_ns, "schedstat"));
    }
    let stat = std::fs::read_to_string(task_dir.join("stat")).ok()?;
    let after_comm = &stat[stat.rfind(')')? + 1..];
    let fields = after_comm.split_whitespace().collect::<Vec<_>>();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    let ns_per_tick = 1_000_000_000 / clock_ticks_per_sec();
    Some(((utime + stime) * ns_per_tick, "stat_ticks"))
}

pub fn snapshot_process_threads(taken_ns: u64) -> ProcessThreadSnapshot {
    #[cfg(target_os = "linux")]
    {
        let mut snapshot = ProcessThreadSnapshot {
            taken_ns,
            threads: Vec::new(),
            source: "unavailable",
        };
        let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
            return snapshot;
        };
        for entry in entries.flatten() {
            let Some(tid) = entry.file_name().to_str().and_then(|v| v.parse().ok()) else {
                continue;
            };
            let path = entry.path();
            let Some((cpu_ns, source)) = read_thread_cpu_ns(&path) else {
                continue;
            };
            let name = std::fs::read_to_string(path.join("comm"))
                .map(|v| v.trim().to_string())
                .unwrap_or_default();
            snapshot.source = source;
            snapshot.threads.push(ThreadCpuSample { tid, name, cpu_ns });
        }
        snapshot
    }
    #[cfg(not(target_os = "linux"))]
    ProcessThreadSnapshot {
        taken_ns,
        threads: Vec::new(),
        source: "unavailable",
    }
}

/// Secondary diagnostic: CPU consumed per thread role between two process snapshots.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ThreadCpuReport {
    pub source: &'static str,
    /// Time between the two snapshots. It brackets the measured window plus the 20 ms
    /// pre-start gap and the post-window drain, harvest, and flush.
    pub sample_interval_ns: u64,
    pub threads_at_window_start: usize,
    pub threads_by_role_at_window_start: BTreeMap<&'static str, usize>,
    pub cpu_ns_by_role: BTreeMap<&'static str, u64>,
    /// `cpu_ns_by_role / sample_interval_ns`: average busy cores per role.
    pub busy_cores_by_role: BTreeMap<&'static str, f64>,
}

pub fn thread_cpu_report(
    before: &ProcessThreadSnapshot,
    after: &ProcessThreadSnapshot,
) -> ThreadCpuReport {
    let pid = i64::from(std::process::id());
    let baseline = before
        .threads
        .iter()
        .map(|thread| (thread.tid, thread.cpu_ns))
        .collect::<HashMap<_, _>>();
    let mut report = ThreadCpuReport {
        source: after.source,
        sample_interval_ns: after.taken_ns.saturating_sub(before.taken_ns),
        threads_at_window_start: before.threads.len(),
        ..ThreadCpuReport::default()
    };
    for thread in &before.threads {
        *report
            .threads_by_role_at_window_start
            .entry(thread_role(thread.tid, pid, &thread.name))
            .or_default() += 1;
    }
    for thread in &after.threads {
        let delta = thread
            .cpu_ns
            .saturating_sub(baseline.get(&thread.tid).copied().unwrap_or(0));
        *report
            .cpu_ns_by_role
            .entry(thread_role(thread.tid, pid, &thread.name))
            .or_default() += delta;
    }
    let interval = report.sample_interval_ns.max(1) as f64;
    report.busy_cores_by_role = report
        .cpu_ns_by_role
        .iter()
        .map(|(role, cpu_ns)| (*role, *cpu_ns as f64 / interval))
        .collect();
    report
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ProcessMemory {
    pub vm_rss_kb: u64,
    pub vm_hwm_kb: u64,
}

pub fn process_memory() -> ProcessMemory {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return ProcessMemory::default();
    };
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    };
    ProcessMemory {
        vm_rss_kb: field("VmRSS:"),
        vm_hwm_kb: field("VmHWM:"),
    }
}

/// Wall time of each untimed preparation phase, in milliseconds.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PrepTimings {
    pub trace_load_ms: f64,
    pub simulation_ms: f64,
    pub merge_and_rescale_ms: f64,
    pub corpus_and_dispatch_ms: f64,
    pub quiescence_ms: f64,
}

/// Distribution summary of one non-negative integer quantity.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CountSummary {
    pub count: usize,
    pub mean: f64,
    pub min: u64,
    pub p50: u64,
    pub p99: u64,
    pub max: u64,
}

pub fn count_summary(mut values: Vec<u64>) -> CountSummary {
    if values.is_empty() {
        return CountSummary::default();
    }
    values.sort_unstable();
    let rank = |numerator: usize| {
        let rank = (values.len() * numerator).div_ceil(100).max(1);
        values[rank - 1]
    };
    CountSummary {
        count: values.len(),
        mean: values.iter().sum::<u64>() as f64 / values.len() as f64,
        min: values[0],
        p50: rank(50),
        p99: rank(99),
        max: values[values.len() - 1],
    }
}

/// Per-inference-worker workload shape, used to show that weak scaling holds the per-worker
/// load constant across worker counts.
#[derive(Clone, Debug, Default, Serialize)]
pub struct WorkloadDiagnostics {
    pub workers: usize,
    pub num_gpu_blocks: usize,
    pub requests_per_worker: CountSummary,
    pub request_blocks_per_worker: CountSummary,
    pub stored_blocks_per_worker: CountSummary,
    pub removed_blocks_per_worker: CountSummary,
    pub block_ops_per_worker: CountSummary,
    /// Stored blocks per worker divided by the per-worker cache capacity.
    pub mean_cache_turnover: f64,
    pub fraction_of_workers_that_evict: f64,
    /// Sum over workers of (stored - removed) blocks at the end of the replay.
    pub end_residency_blocks: u64,
    pub end_residency_fraction_of_capacity: f64,
    pub total_block_ops: u64,
}

pub fn workload_diagnostics(
    prepared: &PreparedMooncakeBenchmark,
    num_gpu_blocks: usize,
) -> WorkloadDiagnostics {
    let workers = prepared.worker_traces.len();
    let mut requests = Vec::with_capacity(workers);
    let mut request_blocks = Vec::with_capacity(workers);
    let mut stored = Vec::with_capacity(workers);
    let mut removed = Vec::with_capacity(workers);
    let mut block_ops = Vec::with_capacity(workers);
    for trace in prepared.worker_traces.iter() {
        let (mut reqs, mut req_blocks, mut st, mut rm) = (0u64, 0u64, 0u64, 0u64);
        for entry in trace {
            match &entry.entry {
                WorkerTraceEntry::Request(hashes) => {
                    reqs += 1;
                    req_blocks += hashes.len() as u64;
                }
                WorkerTraceEntry::Event { event, .. } => match &event.data {
                    KvCacheEventData::Stored(store) => st += store.blocks.len() as u64,
                    KvCacheEventData::Removed(remove) => rm += remove.block_hashes.len() as u64,
                    KvCacheEventData::Cleared => {}
                },
            }
        }
        requests.push(reqs);
        request_blocks.push(req_blocks);
        stored.push(st);
        removed.push(rm);
        block_ops.push(req_blocks + st + rm);
    }
    let evicting = removed.iter().filter(|&&rm| rm > 0).count();
    let end_residency = stored
        .iter()
        .zip(&removed)
        .map(|(st, rm)| st.saturating_sub(*rm))
        .sum::<u64>();
    let capacity = (workers * num_gpu_blocks).max(1) as f64;
    let total_block_ops = block_ops.iter().sum::<u64>();
    let stored_summary = count_summary(stored);
    WorkloadDiagnostics {
        workers,
        num_gpu_blocks,
        requests_per_worker: count_summary(requests),
        request_blocks_per_worker: count_summary(request_blocks),
        mean_cache_turnover: stored_summary.mean / num_gpu_blocks.max(1) as f64,
        stored_blocks_per_worker: stored_summary,
        removed_blocks_per_worker: count_summary(removed),
        block_ops_per_worker: count_summary(block_ops),
        fraction_of_workers_that_evict: evicting as f64 / workers.max(1) as f64,
        end_residency_blocks: end_residency,
        end_residency_fraction_of_capacity: end_residency as f64 / capacity,
        total_block_ops,
    }
}

/// Indexer backend that accepts every event and answers every lookup with no matches. It runs
/// the same event threads, observation bookkeeping, query lanes, and issuers as a real backend,
/// so its sustained ceiling bounds what the harness itself can drive.
#[derive(Default)]
pub struct NullIndexer {
    events: AtomicU64,
}

impl NullIndexer {
    pub fn events_seen(&self) -> u64 {
        self.events.load(Ordering::Relaxed)
    }
}

impl SyncIndexer for NullIndexer {
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        _metrics: Option<std::sync::Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let mut observation = WorkerObservationState::default();
        let mut events = 0u64;
        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(_) => events += 1,
                WorkerTask::EventWithAck { resp, .. } => {
                    events += 1;
                    let _ = resp.send(true);
                }
                WorkerTask::ApproximateLru(task) => drop(task),
                WorkerTask::InstallObservation { writer, resp } => {
                    observation.install(writer, resp)
                }
                WorkerTask::ObservedEvent {
                    event,
                    correlation_id,
                } => {
                    drop(event);
                    events += 1;
                    observation.record(correlation_id, true);
                }
                WorkerTask::SealObservation(resp) => observation.seal(resp),
                WorkerTask::HarvestObservation(resp) => observation.harvest(resp),
                WorkerTask::Anchor { .. }
                | WorkerTask::RemoveWorkerDpRank { .. }
                | WorkerTask::CleanupStaleChildren => {}
                WorkerTask::RemoveWorker { resp, .. } => {
                    let _ = resp.send(());
                }
                WorkerTask::DumpEvents(sender) => {
                    let _ = sender.send(Ok(Vec::new()));
                }
                WorkerTask::Stats(sender) => {
                    let _ = sender.send(WorkerLookupStats::from_worker_block_counts(
                        std::iter::empty(),
                    ));
                }
                WorkerTask::ContainsWorkerBlock { resp, .. } => {
                    let _ = resp.send(false);
                }
                WorkerTask::Flush(sender) => {
                    self.events
                        .fetch_add(std::mem::take(&mut events), Ordering::Relaxed);
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => break,
            }
        }
        self.events.fetch_add(events, Ordering::Relaxed);
        Ok(())
    }

    fn find_matches(&self, _sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
        OverlapScores::new()
    }

    fn supports_event_dump(&self) -> bool {
        false
    }

    fn timing_report(&self) -> String {
        format!("Null backend: events seen = {}", self.events_seen())
    }
}

/// Exact-prefix reference index used to check stack scores at scale.
///
/// It is deliberately naive: a map from (parent sequence hash, local hash) to the stored
/// sequence hash, plus the set of ranks holding each stored block. A rank's score for a
/// query is the length of the longest query prefix whose every block that rank holds.
#[derive(Default)]
pub struct ReferenceIndex {
    children: FxHashMap<(Option<u64>, u64), u64>,
    nodes: FxHashMap<u64, ReferenceNode>,
    rank_blocks: FxHashMap<WorkerWithDpRank, FxHashSet<u64>>,
    /// Stores whose sequence hash was already linked under a different (parent, local) key.
    pub inconsistent_links: u64,
}

struct ReferenceNode {
    key: (Option<u64>, u64),
    holders: FxHashSet<WorkerWithDpRank>,
}

impl ReferenceIndex {
    pub fn apply(&mut self, event: &RouterEvent) {
        let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        match &event.event.data {
            KvCacheEventData::Stored(store) => {
                let mut parent = store.parent_hash.map(|hash| hash.0);
                for block in &store.blocks {
                    let hash = block.block_hash.0;
                    let key = (parent, block.tokens_hash.0);
                    let node = self.nodes.entry(hash).or_insert_with(|| ReferenceNode {
                        key,
                        holders: FxHashSet::default(),
                    });
                    if node.key != key {
                        self.inconsistent_links += 1;
                    }
                    node.holders.insert(rank);
                    self.children.insert(node.key, hash);
                    self.rank_blocks.entry(rank).or_default().insert(hash);
                    parent = Some(hash);
                }
            }
            KvCacheEventData::Removed(remove) => {
                for hash in &remove.block_hashes {
                    if let Some(blocks) = self.rank_blocks.get_mut(&rank) {
                        blocks.remove(&hash.0);
                    }
                    self.release(rank, hash.0);
                }
            }
            KvCacheEventData::Cleared => {
                let blocks = self.rank_blocks.remove(&rank).unwrap_or_default();
                for hash in blocks {
                    self.release(rank, hash);
                }
            }
        }
    }

    fn release(&mut self, rank: WorkerWithDpRank, hash: u64) {
        let Some(node) = self.nodes.get_mut(&hash) else {
            return;
        };
        node.holders.remove(&rank);
        if !node.holders.is_empty() {
            return;
        }
        let key = node.key;
        self.nodes.remove(&hash);
        if self.children.get(&key) == Some(&hash) {
            self.children.remove(&key);
        }
    }

    pub fn scores(&self, query: &[LocalBlockHash]) -> BTreeMap<WorkerWithDpRank, u32> {
        let mut scores = BTreeMap::new();
        let mut active = Vec::<WorkerWithDpRank>::new();
        let mut parent = None;
        let mut matched = 0u32;
        for (depth, local) in query.iter().enumerate() {
            let Some(hash) = self.children.get(&(parent, local.0)).copied() else {
                break;
            };
            let Some(node) = self.nodes.get(&hash) else {
                break;
            };
            if depth == 0 {
                active.extend(node.holders.iter().copied());
            } else {
                active.retain(|rank| {
                    let keep = node.holders.contains(rank);
                    if !keep {
                        scores.insert(*rank, matched);
                    }
                    keep
                });
            }
            if active.is_empty() {
                break;
            }
            matched += 1;
            parent = Some(hash);
        }
        for rank in active {
            scores.insert(rank, matched);
        }
        scores
    }

    pub fn resident_blocks(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ScoreMismatch {
    pub query_ordinal: usize,
    pub operation_id: u32,
    pub query_len: usize,
    pub expected_ranks: usize,
    pub actual_ranks: usize,
    pub first_difference: String,
}

/// Result of the quiescent score check (not a timed run).
#[derive(Clone, Debug, Default, Serialize)]
pub struct CorrectnessReport {
    pub mode: &'static str,
    pub backend: String,
    pub workers: usize,
    pub total_queries: usize,
    pub stride: usize,
    pub checked_queries: usize,
    pub mismatches: usize,
    pub mismatch_examples: Vec<ScoreMismatch>,
    pub checked_matched_ranks: CountSummary,
    pub checked_queries_with_more_than_256_matches: usize,
    pub events_applied: usize,
    pub reference_inconsistent_links: u64,
    pub reference_resident_blocks_at_end: usize,
    pub registration_failures: usize,
    pub elapsed_ms: f64,
    pub pass: bool,
    /// Agentic corpus provenance and validity checks (agentic workload only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agentic: Option<serde_json::Value>,
}

pub fn compare_scores(
    expected: &BTreeMap<WorkerWithDpRank, u32>,
    actual: &OverlapScores,
) -> Option<String> {
    let actual = actual
        .scores
        .iter()
        .filter(|(_, score)| **score > 0)
        .map(|(rank, score)| (*rank, *score))
        .collect::<BTreeMap<_, _>>();
    if &actual == expected {
        return None;
    }
    let difference = expected
        .iter()
        .find(|(rank, score)| actual.get(rank) != Some(score))
        .map(|(rank, score)| {
            format!(
                "rank {}:{} expected {score} actual {:?}",
                rank.worker_id,
                rank.dp_rank,
                actual.get(rank)
            )
        })
        .or_else(|| {
            actual
                .iter()
                .find(|(rank, _)| !expected.contains_key(rank))
                .map(|(rank, score)| {
                    format!(
                        "rank {}:{} unexpected actual {score}",
                        rank.worker_id, rank.dp_rank
                    )
                })
        })
        .unwrap_or_default();
    Some(difference)
}

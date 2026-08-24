// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::{Parser, ValueEnum};
use dynamo_bench::kv_router_common::args::CommonArgs;
use dynamo_bench::kv_router_common::issuer::{pin_current_thread, pin_current_thread_to_cpus};
use dynamo_bench::kv_router_common::replay::{
    WorkerReplayArtifacts, generate_replay_artifacts, process_mooncake_trace,
};
use dynamo_kv_router::indexer::{
    ApproximateAcquireMode, ApproximateLruBlock, ApproximateLruLease, ApproximateLruRequestId,
    ApproximateLruStats, ApproximateRetentionConfig, KvIndexerInterface, KvIndexerMetrics,
};
use dynamo_kv_router::protocols::{LocalBlockHash, WorkerWithDpRank};
use dynamo_kv_router::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer, approx::PruneConfig};
use dynamo_tokens::SequenceHash;
use serde::Serialize;
use tokio::sync::{Notify, oneshot};

const RESULT_SCHEMA_VERSION: u32 = 1;
const EMPTY_OPERATION_ID: u32 = u32::MAX;
const START_LEAD_NS: u64 = 20_000_000;
const RANK_INCARNATION: u64 = 1;

type ApproximateIndexer = ThreadPoolIndexer<ConcurrentRadixTreeCompressed>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Policy {
    Ttl,
    Lru,
}

#[derive(Parser, Debug)]
#[clap(
    version,
    about = "Direct prompt-lifecycle benchmark for approximate TTL and capacity-bounded LRU"
)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Approximate cache-retention policy to exercise.
    #[clap(long, value_enum, default_value = "ttl")]
    policy: Policy,

    /// Number of persistent request-attempt operation lanes.
    #[clap(long, default_value = "128")]
    operation_lanes: usize,

    /// Number of sticky mutation workers owned by the production indexer.
    #[clap(long, default_value = "4")]
    num_event_workers: usize,

    /// Per-rank LRU capacity. Defaults to --num-gpu-blocks.
    #[clap(long)]
    capacity_blocks: Option<usize>,

    /// Retention TTL for the TTL arm and LRU fallback path.
    #[clap(long, default_value = "3600")]
    ttl_secs: u64,

    /// Busy-spin interval after the deadline issuer sleeps.
    #[clap(long, default_value = "75")]
    issuer_spin_us: u64,

    /// Optional CPU for the absolute-deadline issuer thread.
    #[clap(long)]
    issuer_cpu: Option<usize>,

    /// Optional CPU list/ranges for Tokio and indexer work, for example 2-15,18.
    #[clap(long)]
    backend_cpus: Option<String>,

    /// Untimed allocator quiescence before backend construction.
    #[clap(long, default_value_t = default_quiescence_ms())]
    pre_run_quiescence_ms: u64,

    /// Watchdog applied to each acknowledged indexer command and final fence.
    #[clap(long, default_value = "30")]
    command_timeout_secs: u64,

    /// Fail unless the LRU arm performs at least one physical eviction.
    #[clap(long)]
    require_eviction: bool,

    /// JSON output path for one benchmark result.
    #[clap(long, default_value = "approximate_lru_result.json")]
    result_json_output: String,
}

const fn default_quiescence_ms() -> u64 {
    if cfg!(target_os = "linux") { 5_000 } else { 0 }
}

#[derive(Clone, Debug)]
struct SourceRequest {
    admission_timestamp_us: u64,
    terminal_timestamp_us: u64,
    local_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<SequenceHash>,
    private_blocks: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum OperationKind {
    Acquire,
    Release,
}

#[derive(Clone, Copy, Debug)]
struct LogicalOperation {
    id: u32,
    deadline_ns: u64,
    worker: WorkerWithDpRank,
    request_index: usize,
    lane: usize,
    kind: OperationKind,
}

struct PreparedRequest {
    lru_request_id: ApproximateLruRequestId,
    lookup_hashes: Vec<LocalBlockHash>,
    local_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<SequenceHash>,
    lru_blocks: Vec<ApproximateLruBlock>,
    private_blocks: usize,
}

struct ScheduledOperation {
    id: u32,
    deadline_ns: u64,
    lane: usize,
}

struct LanePayloadSlot {
    id: u32,
    payload: Option<LanePayload>,
}

enum LanePayload {
    Acquire {
        worker: WorkerWithDpRank,
        request_slot: usize,
        request: PreparedRequest,
    },
    Release {
        request_slot: usize,
    },
}

struct PreparedTrial {
    dispatch: Vec<ScheduledOperation>,
    lane_payloads: Vec<Box<[LanePayloadSlot]>>,
    lane_request_counts: Vec<usize>,
    #[cfg(test)]
    #[allow(dead_code)]
    operation_kinds: Box<[OperationKind]>,
    #[cfg(test)]
    #[allow(dead_code)]
    operation_requests: Box<[usize]>,
    workers: Vec<WorkerWithDpRank>,
    requests: usize,
    complete_prompt_blocks: usize,
    private_prompt_blocks: usize,
    unique_worker_blocks: usize,
    benchmark_duration_ns: u64,
}

impl PreparedTrial {
    fn page_touch_untimed(&self) {
        let mut checksum = self.benchmark_duration_ns;
        for operation in &self.dispatch {
            checksum ^= operation.deadline_ns ^ u64::from(operation.id) ^ operation.lane as u64;
        }
        for lane in &self.lane_payloads {
            for slot in lane.iter() {
                checksum ^= u64::from(slot.id);
                if let Some(LanePayload::Acquire {
                    request, worker, ..
                }) = &slot.payload
                {
                    checksum ^= worker.worker_id ^ u64::from(worker.dp_rank);
                    checksum ^= request.private_blocks as u64;
                    for hash in &request.local_hashes {
                        checksum ^= hash.0;
                    }
                    for hash in &request.lookup_hashes {
                        checksum ^= hash.0;
                    }
                    for hash in &request.sequence_hashes {
                        checksum ^= *hash;
                    }
                    for block in &request.lru_blocks {
                        checksum ^= block.local_hash.0 ^ block.sequence_hash;
                    }
                }
            }
        }
        black_box(checksum);
    }
}

fn terminal_timestamp(
    request_timestamp_us: u64,
    signals: impl IntoIterator<Item = (u64, bool)>,
) -> anyhow::Result<u64> {
    let terminals = signals
        .into_iter()
        .filter_map(|(timestamp_us, completed)| completed.then_some(timestamp_us))
        .collect::<Vec<_>>();
    match terminals.as_slice() {
        [terminal] if *terminal >= request_timestamp_us => Ok(*terminal),
        [terminal] => anyhow::bail!(
            "request terminal timestamp {terminal} precedes admission {request_timestamp_us}"
        ),
        [] => anyhow::bail!("request has no terminal output signal"),
        _ => anyhow::bail!("request has more than one terminal output signal"),
    }
}

fn requests_from_artifact(
    artifact: WorkerReplayArtifacts,
    block_size: u32,
) -> anyhow::Result<Vec<SourceRequest>> {
    let mut signals = HashMap::new();
    for timed in artifact.output_signals {
        signals
            .entry(timed.signal.uuid)
            .or_insert_with(Vec::new)
            .push((timed.timestamp_us, timed.signal.completed));
    }

    artifact
        .requests
        .into_iter()
        .map(|request| {
            let terminal_timestamp_us = terminal_timestamp(
                request.timestamp_us,
                signals.remove(&request.uuid).unwrap_or_default(),
            )?;
            let local_hashes = request
                .replay_hashes
                .local_block_hashes
                .into_iter()
                .map(LocalBlockHash)
                .collect::<Vec<_>>();
            let sequence_hashes = request.replay_hashes.sequence_hashes;
            if local_hashes.len() != sequence_hashes.len() {
                anyhow::bail!(
                    "request {} has {} local hashes but {} sequence hashes",
                    request.uuid,
                    local_hashes.len(),
                    sequence_hashes.len()
                );
            }
            let complete_tokens = local_hashes
                .len()
                .checked_mul(block_size as usize)
                .context("complete prompt-token count overflow")?;
            Ok(SourceRequest {
                admission_timestamp_us: request.timestamp_us,
                terminal_timestamp_us,
                local_hashes,
                sequence_hashes,
                private_blocks: usize::from(request.input_length > complete_tokens),
            })
        })
        .collect()
}

fn prepare_trial(
    worker_requests: Vec<Vec<SourceRequest>>,
    benchmark_duration_ms: u64,
    worker_duplication: usize,
    operation_lanes: usize,
) -> anyhow::Result<PreparedTrial> {
    if worker_requests.is_empty() {
        anyhow::bail!("approximate corpus has no worker timelines");
    }
    if benchmark_duration_ms == 0 || worker_duplication == 0 || operation_lanes == 0 {
        anyhow::bail!(
            "benchmark duration, worker duplication, and operation lanes must be positive"
        );
    }

    let mut requests = Vec::new();
    let mut operations = Vec::new();
    let mut workers = Vec::new();
    let mut complete_prompt_blocks = 0usize;
    let mut private_prompt_blocks = 0usize;
    let mut unique_worker_blocks = HashSet::new();
    let target_us = u128::from(benchmark_duration_ms) * 1_000;
    let source_worker_count = worker_requests.len();

    for replica in 0..worker_duplication {
        for (source_worker, source_requests) in worker_requests.iter().enumerate() {
            if source_requests.is_empty() {
                continue;
            }
            let worker_id = replica
                .checked_mul(source_worker_count)
                .and_then(|base| base.checked_add(source_worker))
                .context("worker ID overflow")? as u64;
            let worker = WorkerWithDpRank::from_worker_id(worker_id);
            workers.push(worker);
            let first = source_requests
                .iter()
                .map(|request| request.admission_timestamp_us)
                .min()
                .context("worker trace has no admission")?;
            let last = source_requests
                .iter()
                .map(|request| request.terminal_timestamp_us)
                .max()
                .unwrap_or(first);
            let span = last.saturating_sub(first).max(1);

            for source in source_requests {
                let request_index = requests.len();
                let lane = request_index % operation_lanes;
                let attempt_value = u64::try_from(request_index)?.saturating_add(1);
                let lru_blocks = source
                    .local_hashes
                    .iter()
                    .copied()
                    .zip(source.sequence_hashes.iter().copied())
                    .map(|(local_hash, sequence_hash)| ApproximateLruBlock {
                        local_hash,
                        sequence_hash,
                    })
                    .collect::<Vec<_>>();
                for hash in &source.sequence_hashes {
                    unique_worker_blocks.insert((worker, *hash));
                }
                complete_prompt_blocks = complete_prompt_blocks
                    .checked_add(source.local_hashes.len())
                    .context("complete prompt-block count overflow")?;
                private_prompt_blocks = private_prompt_blocks
                    .checked_add(source.private_blocks)
                    .context("private prompt-block count overflow")?;
                requests.push(PreparedRequest {
                    lru_request_id: ApproximateLruRequestId::for_benchmark(attempt_value),
                    lookup_hashes: source.local_hashes.clone(),
                    local_hashes: source.local_hashes.clone(),
                    sequence_hashes: source.sequence_hashes.clone(),
                    lru_blocks,
                    private_blocks: source.private_blocks,
                });
                for (kind, timestamp_us) in [
                    (OperationKind::Acquire, source.admission_timestamp_us),
                    (OperationKind::Release, source.terminal_timestamp_us),
                ] {
                    let relative = timestamp_us.saturating_sub(first);
                    let scaled_us = u128::from(relative) * target_us / u128::from(span);
                    operations.push(LogicalOperation {
                        id: 0,
                        deadline_ns: (scaled_us.min(u128::from(u64::MAX)) as u64)
                            .saturating_mul(1_000),
                        worker,
                        request_index,
                        lane,
                        kind,
                    });
                }
            }
        }
    }
    workers.sort_unstable();
    workers.dedup();
    if requests.is_empty() {
        anyhow::bail!("approximate corpus has no requests");
    }
    if operations.len() >= EMPTY_OPERATION_ID as usize {
        anyhow::bail!("approximate corpus exceeds the u32 operation-ID space");
    }
    operations.sort_by_key(|operation| {
        (
            operation.deadline_ns,
            match operation.kind {
                OperationKind::Acquire => 0,
                OperationKind::Release => 1,
            },
            operation.worker,
            operation.request_index,
        )
    });
    for (id, operation) in operations.iter_mut().enumerate() {
        operation.id = id as u32;
    }

    let mut lane_request_counts = vec![0usize; operation_lanes];
    let mut request_slots = vec![0usize; requests.len()];
    for (request_index, request_slot) in request_slots.iter_mut().enumerate() {
        let lane = request_index % operation_lanes;
        *request_slot = lane_request_counts[lane];
        lane_request_counts[lane] += 1;
    }
    let mut lane_payloads = (0..operation_lanes).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut dispatch = Vec::with_capacity(operations.len());
    #[cfg(test)]
    let mut operation_kinds = Vec::with_capacity(operations.len());
    #[cfg(test)]
    let mut operation_requests = Vec::with_capacity(operations.len());
    let mut requests = requests.into_iter().map(Some).collect::<Vec<_>>();
    for operation in operations {
        let request_slot = request_slots[operation.request_index];
        let payload = match operation.kind {
            OperationKind::Acquire => LanePayload::Acquire {
                worker: operation.worker,
                request_slot,
                request: requests[operation.request_index]
                    .take()
                    .context("request has more than one acquire")?,
            },
            OperationKind::Release => LanePayload::Release { request_slot },
        };
        dispatch.push(ScheduledOperation {
            id: operation.id,
            deadline_ns: operation.deadline_ns,
            lane: operation.lane,
        });
        lane_payloads[operation.lane].push(LanePayloadSlot {
            id: operation.id,
            payload: Some(payload),
        });
        #[cfg(test)]
        operation_kinds.push(operation.kind);
        #[cfg(test)]
        operation_requests.push(operation.request_index);
    }

    Ok(PreparedTrial {
        dispatch,
        lane_payloads: lane_payloads
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect(),
        lane_request_counts,
        #[cfg(test)]
        operation_kinds: operation_kinds.into_boxed_slice(),
        #[cfg(test)]
        operation_requests: operation_requests.into_boxed_slice(),
        workers,
        requests: request_slots.len(),
        complete_prompt_blocks,
        private_prompt_blocks,
        unique_worker_blocks: unique_worker_blocks.len(),
        benchmark_duration_ns: benchmark_duration_ms.saturating_mul(1_000_000),
    })
}

struct OperationLane {
    slots: Box<[AtomicU32]>,
    published: AtomicUsize,
    consumed: AtomicUsize,
    closed: AtomicBool,
    notify: Notify,
}

impl OperationLane {
    fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| AtomicU32::new(EMPTY_OPERATION_ID))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            published: AtomicUsize::new(0),
            consumed: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_one();
    }

    fn depth(&self) -> usize {
        self.published
            .load(Ordering::Acquire)
            .saturating_sub(self.consumed.load(Ordering::Acquire))
    }
}

#[derive(Debug)]
struct CompletionRecord {
    id: u32,
    mode: Option<ApproximateAcquireMode>,
    failure: Option<String>,
}

struct LaneResult {
    completions: Vec<CompletionRecord>,
}

struct IssuerOutput {
    issued_operations: usize,
    producer_stop_ns: u64,
    cpu_ns: Option<u64>,
    failure: Option<String>,
}

async fn control_plane_timeout<T>(
    timeout: Duration,
    future: impl Future<Output = T>,
) -> anyhow::Result<T> {
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| anyhow::anyhow!("indexer command exceeded {timeout:?}"))
}

async fn execute_payload(
    indexer: &ApproximateIndexer,
    policy: Policy,
    payload: LanePayload,
    leases: &mut [Option<ApproximateLruLease>],
) -> (Option<ApproximateAcquireMode>, Option<String>) {
    match payload {
        LanePayload::Acquire {
            worker,
            request_slot,
            request,
        } => {
            let lookup_result = indexer.find_matches(request.lookup_hashes).await;
            if let Err(error) = lookup_result {
                return (None, Some(format!("lookup: {error:#}")));
            }
            match policy {
                Policy::Ttl => {
                    let result = indexer
                        .process_routing_decision_hash_slices(
                            worker,
                            &request.local_hashes,
                            &request.sequence_hashes,
                        )
                        .await;
                    (
                        None,
                        result.err().map(|error| format!("TTL insert: {error:#}")),
                    )
                }
                Policy::Lru => {
                    let Some(lease) = indexer.begin_approximate_lru_request(
                        worker,
                        RANK_INCARNATION,
                        request.lru_request_id,
                    ) else {
                        return (None, Some("LRU lease unavailable".to_string()));
                    };
                    let result = lease
                        .acquire(request.lru_blocks, request.private_blocks)
                        .await;
                    match result {
                        Ok(mode) => {
                            leases[request_slot] = Some(lease);
                            (Some(mode), None)
                        }
                        Err(error) => (None, Some(format!("LRU acquire: {error:#}"))),
                    }
                }
            }
        }
        LanePayload::Release { request_slot } => match policy {
            Policy::Ttl => (None, None),
            Policy::Lru => {
                let Some(lease) = leases.get_mut(request_slot).and_then(Option::take) else {
                    return (None, Some("release has no acquired lease".to_string()));
                };
                let result = lease.finish().await;
                (
                    None,
                    result.err().map(|error| format!("LRU release: {error:#}")),
                )
            }
        },
    }
}

async fn lane_worker(
    indexer: Arc<ApproximateIndexer>,
    policy: Policy,
    lane: Arc<OperationLane>,
    mut payloads: Box<[LanePayloadSlot]>,
    request_count: usize,
    ready: oneshot::Sender<()>,
) -> LaneResult {
    let mut leases = vec![None; request_count];
    let mut completions = Vec::with_capacity(payloads.len());
    let mut consumed = 0usize;
    let _ = ready.send(());

    loop {
        let published = lane.published.load(Ordering::Acquire);
        while consumed < published {
            let id = lane.slots[consumed].load(Ordering::Relaxed);
            let (mode, failure) = match payloads.get_mut(consumed) {
                Some(slot) if slot.id == id => match slot.payload.take() {
                    Some(payload) => {
                        execute_payload(indexer.as_ref(), policy, payload, &mut leases).await
                    }
                    None => (None, Some("missing lane payload".to_string())),
                },
                _ => (None, Some("lane payload order mismatch".to_string())),
            };
            completions.push(CompletionRecord { id, mode, failure });
            consumed += 1;
            lane.consumed.store(consumed, Ordering::Release);
        }

        if lane.closed.load(Ordering::Acquire) && consumed == lane.published.load(Ordering::Acquire)
        {
            break;
        }
        let notified = lane.notify.notified();
        if consumed == lane.published.load(Ordering::Acquire)
            && !lane.closed.load(Ordering::Acquire)
        {
            notified.await;
        }
    }
    drop(leases);
    LaneResult { completions }
}

fn issue_operations(
    dispatch: Vec<ScheduledOperation>,
    lanes: Vec<Arc<OperationLane>>,
    clock: Arc<BenchmarkClock>,
    start_signal: Arc<AtomicU64>,
    ready: mpsc::SyncSender<()>,
    issuer_cpu: Option<usize>,
) -> IssuerOutput {
    let cpu_start = thread_cpu_time_ns();
    let mut lane_cursors = vec![0usize; lanes.len()];
    let mut issued_operations = 0usize;
    let mut failure = pin_current_thread(issuer_cpu)
        .err()
        .map(|error| format!("issuer affinity: {error}"));
    if ready.send(()).is_err() {
        failure.get_or_insert_with(|| "issuer ready receiver dropped".to_string());
    }
    while failure.is_none() && start_signal.load(Ordering::Acquire) == 0 {
        std::hint::spin_loop();
    }
    let start_ns = start_signal.load(Ordering::Acquire);

    for operation in dispatch {
        if failure.is_some() {
            break;
        }
        clock.wait_until(start_ns.saturating_add(operation.deadline_ns));
        let Some(lane) = lanes.get(operation.lane) else {
            failure = Some("issuer lane out of range".to_string());
            break;
        };
        let cursor = lane_cursors[operation.lane];
        let Some(slot) = lane.slots.get(cursor) else {
            failure = Some("issuer fixed lane overflow".to_string());
            break;
        };
        slot.store(operation.id, Ordering::Relaxed);
        lane_cursors[operation.lane] += 1;
        lane.published
            .store(lane_cursors[operation.lane], Ordering::Release);
        lane.notify.notify_one();
        issued_operations += 1;
    }

    IssuerOutput {
        issued_operations,
        producer_stop_ns: clock.now_ns(),
        cpu_ns: cpu_start
            .and_then(|start| thread_cpu_time_ns().map(|end| end.saturating_sub(start))),
        failure,
    }
}

#[derive(Debug, Serialize)]
struct LruStatsResult {
    ranks: usize,
    fallback_ranks: usize,
    resident_blocks: usize,
    active_blocks: usize,
    inactive_blocks: usize,
    private_blocks: usize,
    leases: usize,
    overcapacity_blocks: usize,
    requests: u64,
    request_messages: u64,
    fallback_activations: u64,
    eviction_batches: u64,
    evicted_blocks: u64,
    terminal_mutation_queue_depth: usize,
    mean_mutation_wait_ns: f64,
}

impl From<ApproximateLruStats> for LruStatsResult {
    fn from(stats: ApproximateLruStats) -> Self {
        Self {
            ranks: stats.ranks,
            fallback_ranks: stats.fallback_ranks,
            resident_blocks: stats.resident_blocks,
            active_blocks: stats.active_blocks,
            inactive_blocks: stats.inactive_blocks,
            private_blocks: stats.private_blocks,
            leases: stats.leases,
            overcapacity_blocks: stats.overcapacity_blocks,
            requests: stats.requests,
            request_messages: stats.request_messages,
            fallback_activations: stats.fallback_activations,
            eviction_batches: stats.eviction_batches,
            evicted_blocks: stats.evicted_blocks,
            terminal_mutation_queue_depth: stats.mutation_queue_depth,
            mean_mutation_wait_ns: if stats.mutation_wait_samples == 0 {
                0.0
            } else {
                stats.mutation_wait_ns as f64 / stats.mutation_wait_samples as f64
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ResultJson {
    schema_version: u32,
    scope: &'static str,
    policy: Policy,
    timer: &'static str,
    benchmark_duration_ms: u64,
    operation_lanes: usize,
    num_event_workers: usize,
    total_workers: usize,
    capacity_blocks_per_rank: Option<usize>,
    total_requests: usize,
    total_operations: usize,
    complete_prompt_blocks: usize,
    private_prompt_blocks: usize,
    unique_worker_blocks: usize,
    offered_operations_per_sec: f64,
    achieved_operations_per_sec: f64,
    process_cpu_ns: Option<u64>,
    issuer_cpu_ns: Option<u64>,
    process_cpu_ns_per_request: Option<f64>,
    total_wall_ns: u64,
    queue_depth_at_stop: Vec<usize>,
    issue_span_ns: u64,
    drain_ns: u64,
    acquired_lru: usize,
    acquired_ttl_fallback: usize,
    acquired_ignored: usize,
    final_radix_blocks: usize,
    lru_stats: Option<LruStatsResult>,
    generator_valid: bool,
    kept_up: bool,
    failure_reasons: Vec<String>,
}

async fn run_trial(
    trial: PreparedTrial,
    args: &Args,
    backend_cpus: Option<&[usize]>,
) -> anyhow::Result<ResultJson> {
    trial.page_touch_untimed();
    quiesce(args.pre_run_quiescence_ms).await;
    if let Some(cpus) = backend_cpus {
        pin_current_thread_to_cpus(cpus).context("pinning benchmark backend CPUs")?;
    }

    let fallback = PruneConfig {
        ttl: Duration::from_secs(args.ttl_secs),
    };
    let retention = match args.policy {
        Policy::Ttl => ApproximateRetentionConfig::Ttl(fallback),
        Policy::Lru => ApproximateRetentionConfig::Lru {
            fallback_ttl: fallback,
        },
    };
    let indexer = Arc::new(
        ApproximateIndexer::new_with_metrics_and_approximate_retention(
            ConcurrentRadixTreeCompressed::new(),
            args.num_event_workers,
            args.common.block_size,
            Some(Arc::new(KvIndexerMetrics::new_unregistered())),
            Some(retention),
        ),
    );
    let timeout = Duration::from_secs(args.command_timeout_secs);
    let capacity = args.capacity_blocks.unwrap_or(args.common.num_gpu_blocks);
    if args.policy == Policy::Lru {
        for &worker in &trial.workers {
            control_plane_timeout(
                timeout,
                indexer.set_approximate_lru_capacity(worker, RANK_INCARNATION, Some(capacity)),
            )
            .await??;
        }
        let registered = control_plane_timeout(timeout, indexer.approximate_lru_stats()).await??;
        if registered.ranks != trial.workers.len() || registered.fallback_ranks != 0 {
            anyhow::bail!(
                "LRU registration fence found ranks={} fallback_ranks={}, expected {} LRU ranks",
                registered.ranks,
                registered.fallback_ranks,
                trial.workers.len()
            );
        }
    }

    let clock = Arc::new(BenchmarkClock::new(
        args.issuer_spin_us.saturating_mul(1_000),
    )?);
    let lanes = trial
        .lane_payloads
        .iter()
        .map(|payloads| Arc::new(OperationLane::new(payloads.len())))
        .collect::<Vec<_>>();
    let mut ready_receivers = Vec::with_capacity(lanes.len());
    let mut tasks = Vec::with_capacity(lanes.len());
    for ((lane, payloads), request_count) in lanes
        .iter()
        .zip(trial.lane_payloads)
        .zip(&trial.lane_request_counts)
    {
        let (ready_tx, ready_rx) = oneshot::channel();
        ready_receivers.push(ready_rx);
        tasks.push(tokio::spawn(lane_worker(
            Arc::clone(&indexer),
            args.policy,
            Arc::clone(lane),
            payloads,
            *request_count,
            ready_tx,
        )));
    }
    for ready in ready_receivers {
        ready.await.context("operation lane exited before ready")?;
    }

    let operation_count = trial.dispatch.len();
    let start_signal = Arc::new(AtomicU64::new(0));
    let (issuer_ready_tx, issuer_ready_rx) = mpsc::sync_channel(0);
    let issuer_handle = std::thread::spawn({
        let issuer_lanes = lanes.iter().map(Arc::clone).collect();
        let clock = Arc::clone(&clock);
        let start_signal = Arc::clone(&start_signal);
        let dispatch = trial.dispatch;
        let issuer_cpu = args.issuer_cpu;
        move || {
            issue_operations(
                dispatch,
                issuer_lanes,
                clock,
                start_signal,
                issuer_ready_tx,
                issuer_cpu,
            )
        }
    });
    issuer_ready_rx
        .recv()
        .context("issuer exited before becoming ready")?;
    let process_cpu_start = process_cpu_time_ns();
    let start_ns = clock.now_ns().saturating_add(START_LEAD_NS);
    start_signal.store(start_ns, Ordering::Release);
    let issuer = issuer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("issuer panicked"))?;
    let queue_depth_at_stop = lanes.iter().map(|lane| lane.depth()).collect::<Vec<_>>();
    for lane in &lanes {
        lane.close();
    }
    let mut lane_results = Vec::with_capacity(tasks.len());
    let mut join_failures = Vec::new();
    let drain_timeout =
        Duration::from_millis(args.common.benchmark_duration_ms).saturating_add(timeout);
    let drain_result = tokio::time::timeout(drain_timeout, async {
        for task in &mut tasks {
            match task.await {
                Ok(result) => lane_results.push(result),
                Err(error) => join_failures.push(format!("lane task: {error}")),
            }
        }
    })
    .await;
    if drain_result.is_err() {
        join_failures.push(format!("lane drain exceeded {drain_timeout:?}"));
        for task in &tasks {
            task.abort();
        }
        for task in tasks {
            let _ = task.await;
        }
    }
    let end_ns = clock.now_ns();
    let process_cpu_ns = process_cpu_start
        .and_then(|start| process_cpu_time_ns().map(|end| end.saturating_sub(start)));

    let final_stats = if args.policy == Policy::Lru {
        Some(control_plane_timeout(timeout, indexer.approximate_lru_stats()).await??)
    } else {
        None
    };
    let final_radix_blocks = control_plane_timeout(timeout, indexer.shard_sizes())
        .await?
        .into_iter()
        .map(|snapshot| snapshot.block_count)
        .sum();

    let mut completions = std::iter::repeat_with(|| None)
        .take(operation_count)
        .collect::<Vec<_>>();
    let mut failure_reasons = join_failures;
    for (lane_id, result) in lane_results.into_iter().enumerate() {
        for completion in result.completions {
            let Some(slot) = completions.get_mut(completion.id as usize) else {
                push_failure(&mut failure_reasons, "completion_id_out_of_range");
                continue;
            };
            if slot.replace(completion).is_some() {
                push_failure(&mut failure_reasons, "duplicate_completion");
            }
        }
        let expected = lanes[lane_id].slots.len();
        let actual = lanes[lane_id].consumed.load(Ordering::Acquire);
        if expected != actual {
            push_failure(&mut failure_reasons, "lane_incomplete");
        }
    }
    if let Some(failure) = issuer.failure {
        failure_reasons.push(failure);
    }

    let mut acquired_lru = 0usize;
    let mut acquired_ttl_fallback = 0usize;
    let mut acquired_ignored = 0usize;
    for completion in &completions {
        let Some(completion) = completion.as_ref() else {
            push_failure(&mut failure_reasons, "missing_completion");
            continue;
        };
        if let Some(failure) = &completion.failure {
            failure_reasons.push(failure.clone());
        }
        match completion.mode {
            Some(ApproximateAcquireMode::Lru) => acquired_lru += 1,
            Some(ApproximateAcquireMode::TtlFallback) => acquired_ttl_fallback += 1,
            Some(ApproximateAcquireMode::Ignored) => acquired_ignored += 1,
            None => {}
        }
    }

    let issue_span_ns = issuer.producer_stop_ns.saturating_sub(start_ns);
    let drain_ns = end_ns.saturating_sub(issuer.producer_stop_ns);
    if issue_span_ns > trial.benchmark_duration_ns.saturating_mul(101) / 100 {
        push_failure(&mut failure_reasons, "issue_span_exceeded");
    }
    if issuer.issued_operations != operation_count {
        push_failure(&mut failure_reasons, "incomplete_issue");
    }
    if args.policy == Policy::Lru && acquired_lru != trial.requests {
        push_failure(&mut failure_reasons, "not_all_acquires_used_lru");
    }
    if let Some(stats) = final_stats {
        if stats.ranks != trial.workers.len()
            || stats.fallback_ranks != 0
            || stats.active_blocks != 0
            || stats.private_blocks != 0
            || stats.leases != 0
            || stats.overcapacity_blocks != 0
            || stats.resident_blocks > capacity.saturating_mul(trial.workers.len())
        {
            push_failure(&mut failure_reasons, "final_lru_invariant");
        }
        if args.require_eviction && stats.evicted_blocks == 0 {
            push_failure(&mut failure_reasons, "required_eviction_missing");
        }
    } else if args.require_eviction {
        push_failure(&mut failure_reasons, "require_eviction_needs_lru_policy");
    }
    failure_reasons.sort();
    failure_reasons.dedup();
    let generator_valid = failure_reasons.is_empty();
    let total_ns = end_ns.saturating_sub(start_ns);
    let kept_up =
        generator_valid && total_ns <= trial.benchmark_duration_ns.saturating_mul(110) / 100;
    let offered_seconds = (trial.benchmark_duration_ns as f64 / 1e9).max(f64::EPSILON);
    let achieved_seconds = (total_ns as f64 / 1e9).max(f64::EPSILON);

    Ok(ResultJson {
        schema_version: RESULT_SCHEMA_VERSION,
        scope: "prompt_lifecycle_only_no_output_materialization",
        policy: args.policy,
        timer: clock.timer_name(),
        benchmark_duration_ms: trial.benchmark_duration_ns / 1_000_000,
        operation_lanes: args.operation_lanes,
        num_event_workers: args.num_event_workers,
        total_workers: trial.workers.len(),
        capacity_blocks_per_rank: (args.policy == Policy::Lru).then_some(capacity),
        total_requests: trial.requests,
        total_operations: operation_count,
        complete_prompt_blocks: trial.complete_prompt_blocks,
        private_prompt_blocks: trial.private_prompt_blocks,
        unique_worker_blocks: trial.unique_worker_blocks,
        offered_operations_per_sec: operation_count as f64 / offered_seconds,
        achieved_operations_per_sec: operation_count as f64 / achieved_seconds,
        process_cpu_ns,
        issuer_cpu_ns: issuer.cpu_ns,
        process_cpu_ns_per_request: process_cpu_ns
            .map(|cpu| cpu as f64 / trial.requests.max(1) as f64),
        total_wall_ns: total_ns,
        queue_depth_at_stop,
        issue_span_ns,
        drain_ns,
        acquired_lru,
        acquired_ttl_fallback,
        acquired_ignored,
        final_radix_blocks,
        lru_stats: final_stats.map(Into::into),
        generator_valid,
        kept_up,
        failure_reasons,
    })
}

fn validate_args(args: &Args, backend_cpus: Option<&[usize]>) -> anyhow::Result<()> {
    if args.common.test || args.common.sweep {
        anyhow::bail!("approximate_lru_bench does not support --test or --sweep");
    }
    if args.operation_lanes == 0 || args.num_event_workers == 0 {
        anyhow::bail!("--operation-lanes and --num-event-workers must be positive");
    }
    if args.common.block_size == 0
        || args.common.benchmark_duration_ms == 0
        || args.command_timeout_secs == 0
    {
        anyhow::bail!("block size, benchmark duration, and command timeout must be positive");
    }
    let capacity = args.capacity_blocks.unwrap_or(args.common.num_gpu_blocks);
    if args.policy == Policy::Lru && capacity == 0 {
        anyhow::bail!("LRU capacity must be positive");
    }
    if args.issuer_cpu.is_some() && backend_cpus.is_none() {
        anyhow::bail!("--backend-cpus is required when --issuer-cpu is set");
    }
    if let (Some(issuer), Some(cpus)) = (args.issuer_cpu, backend_cpus)
        && cpus.contains(&issuer)
    {
        anyhow::bail!("--issuer-cpu must be disjoint from --backend-cpus");
    }
    if args.require_eviction && args.policy != Policy::Lru {
        anyhow::bail!("--require-eviction requires --policy lru");
    }
    Ok(())
}

async fn async_main(args: Args, backend_cpus: Option<Vec<usize>>) -> anyhow::Result<()> {
    let trace_path = args
        .common
        .mooncake_trace_path
        .as_deref()
        .context("mooncake trace path is required")?;
    let traces = process_mooncake_trace(
        trace_path,
        args.common.block_size,
        args.common.trace_length_factor,
        args.common.trace_duplication_factor,
        args.common.num_unique_inference_workers,
        args.common.seed,
    )?;
    let artifacts = generate_replay_artifacts(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;
    drop(traces);
    let worker_requests = artifacts
        .into_iter()
        .map(|artifact| requests_from_artifact(artifact, args.common.block_size))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let trial = prepare_trial(
        worker_requests,
        args.common.benchmark_duration_ms,
        args.common.inference_worker_duplication_factor,
        args.operation_lanes,
    )?;
    let capacity = args.capacity_blocks.unwrap_or(args.common.num_gpu_blocks);
    if args.require_eviction
        && trial.unique_worker_blocks <= capacity.saturating_mul(trial.workers.len())
    {
        anyhow::bail!(
            "eviction smoke is invalid: {} unique worker-blocks do not exceed aggregate capacity {}",
            trial.unique_worker_blocks,
            capacity.saturating_mul(trial.workers.len())
        );
    }

    let result = run_trial(trial, &args, backend_cpus.as_deref()).await?;
    std::fs::write(
        &args.result_json_output,
        serde_json::to_string_pretty(&result)?,
    )?;
    println!(
        "policy={:?} requests={} achieved={:.0} ops/s CPU/request={:.0}ns valid={} kept_up={}",
        result.policy,
        result.total_requests,
        result.achieved_operations_per_sec,
        result.process_cpu_ns_per_request.unwrap_or_default(),
        result.generator_valid,
        result.kept_up,
    );
    println!(
        "Approximate indexer result written to {}",
        args.result_json_output
    );
    if !result.generator_valid {
        anyhow::bail!(
            "approximate benchmark failed validation: {}",
            result.failure_reasons.join(", ")
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let backend_cpus = args
        .backend_cpus
        .as_deref()
        .map(parse_cpu_list)
        .transpose()?;
    validate_args(&args, backend_cpus.as_deref())?;
    if let Some(cpus) = backend_cpus.as_deref() {
        pin_current_thread_to_cpus(cpus).context("pinning runtime construction CPUs")?;
    }
    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    if let Some(cpus) = backend_cpus.as_deref() {
        runtime.worker_threads(cpus.len());
    }
    runtime.build()?.block_on(async_main(args, backend_cpus))
}

fn parse_cpu_list(value: &str) -> anyhow::Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if let Some((start, end)) = part.split_once('-') {
            let start = start.parse::<usize>()?;
            let end = end.parse::<usize>()?;
            if start > end {
                anyhow::bail!("invalid descending CPU range {part}");
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(part.parse::<usize>()?);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    if cpus.is_empty() {
        anyhow::bail!("CPU list is empty");
    }
    Ok(cpus)
}

async fn quiesce(milliseconds: u64) {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
    if milliseconds > 0 {
        tokio::time::sleep(Duration::from_millis(milliseconds)).await;
    }
}

fn push_failure(failures: &mut Vec<String>, failure: &str) {
    if !failures.iter().any(|existing| existing == failure) {
        failures.push(failure.to_string());
    }
}

struct BenchmarkClock {
    epoch: Instant,
    spin_ns: u64,
    #[cfg(target_os = "linux")]
    monotonic_epoch_ns: u64,
}

impl BenchmarkClock {
    fn new(spin_ns: u64) -> anyhow::Result<Self> {
        #[cfg(target_os = "linux")]
        let monotonic_epoch_ns = clock_time_ns(libc::CLOCK_MONOTONIC)?;
        Ok(Self {
            epoch: Instant::now(),
            spin_ns,
            #[cfg(target_os = "linux")]
            monotonic_epoch_ns,
        })
    }

    fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    fn wait_until(&self, target_ns: u64) {
        let sleep_target = target_ns.saturating_sub(self.spin_ns);
        let now = self.now_ns();
        #[cfg(target_os = "linux")]
        if sleep_target > now {
            sleep_until_monotonic(self.monotonic_epoch_ns.saturating_add(sleep_target));
        }
        #[cfg(not(target_os = "linux"))]
        if sleep_target > now {
            std::thread::sleep(Duration::from_nanos(sleep_target - now));
        }
        while self.now_ns() < target_ns {
            std::hint::spin_loop();
        }
    }

    fn timer_name(&self) -> &'static str {
        if cfg!(target_os = "linux") {
            "clock_nanosleep_monotonic_absolute"
        } else {
            "portable_sleep_spin_non_authoritative"
        }
    }
}

#[cfg(target_os = "linux")]
fn clock_time_ns(clock_id: libc::clockid_t) -> anyhow::Result<u64> {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { libc::clock_gettime(clock_id, &mut timestamp) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok((timestamp.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(timestamp.tv_nsec as u64))
}

#[cfg(target_os = "linux")]
fn process_cpu_time_ns() -> Option<u64> {
    clock_time_ns(libc::CLOCK_PROCESS_CPUTIME_ID).ok()
}

#[cfg(not(target_os = "linux"))]
fn process_cpu_time_ns() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn thread_cpu_time_ns() -> Option<u64> {
    clock_time_ns(libc::CLOCK_THREAD_CPUTIME_ID).ok()
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_time_ns() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn sleep_until_monotonic(target_ns: u64) {
    let request = libc::timespec {
        tv_sec: (target_ns / 1_000_000_000) as libc::time_t,
        tv_nsec: (target_ns % 1_000_000_000) as libc::c_long,
    };
    loop {
        let rc = unsafe {
            libc::clock_nanosleep(
                libc::CLOCK_MONOTONIC,
                libc::TIMER_ABSTIME,
                &request,
                std::ptr::null_mut(),
            )
        };
        if rc == 0 || rc != libc::EINTR {
            return;
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;

    fn request(admission: u64, terminal: u64, hash: u64) -> SourceRequest {
        SourceRequest {
            admission_timestamp_us: admission,
            terminal_timestamp_us: terminal,
            local_hashes: vec![LocalBlockHash(hash)],
            sequence_hashes: vec![hash],
            private_blocks: 0,
        }
    }

    fn args(capacity_blocks: usize, require_eviction: bool) -> Args {
        Args {
            common: CommonArgs {
                mooncake_trace_path: None,
                test: false,
                num_gpu_blocks: 16,
                block_size: 4,
                trace_simulation_duration_ms: None,
                benchmark_duration_ms: 1_000,
                num_unique_inference_workers: 1,
                inference_worker_duplication_factor: 1,
                trace_length_factor: 1,
                trace_duplication_factor: 1,
                seed: 42,
                sweep: false,
                sweep_min_ms: 1,
                sweep_max_ms: 1,
                sweep_steps: 1,
                bench: false,
                sequence_logs: false,
            },
            policy: Policy::Lru,
            operation_lanes: 2,
            num_event_workers: 1,
            capacity_blocks: Some(capacity_blocks),
            ttl_secs: 3_600,
            issuer_spin_us: 0,
            issuer_cpu: None,
            backend_cpus: None,
            pre_run_quiescence_ms: 0,
            command_timeout_secs: 5,
            require_eviction,
            result_json_output: String::new(),
        }
    }

    #[test]
    fn lifecycle_uses_admission_and_terminal_timestamps() {
        assert_eq!(
            terminal_timestamp(10, [(100, false), (200, true)]).unwrap(),
            200
        );
        let trial = prepare_trial(vec![vec![request(10, 200, 1)]], 1_000, 1, 2).unwrap();
        assert_eq!(trial.dispatch[0].deadline_ns, 0);
        assert_eq!(trial.dispatch[1].deadline_ns, 1_000_000_000);
    }

    #[test]
    fn request_attempts_keep_acquire_and_release_on_the_same_lane() {
        let trial = prepare_trial(
            vec![vec![request(0, 20, 1), request(10, 30, 2)]],
            1_000,
            1,
            2,
        )
        .unwrap();
        let mut lanes = HashMap::new();
        for operation in trial.dispatch {
            let request = trial.operation_requests[operation.id as usize];
            match trial.operation_kinds[operation.id as usize] {
                OperationKind::Acquire => {
                    lanes.insert(request, operation.lane);
                }
                OperationKind::Release => {
                    assert_eq!(lanes.get(&request), Some(&operation.lane));
                }
            }
        }
        assert_ne!(trial.lane_request_counts[0], 0);
        assert_ne!(trial.lane_request_counts[1], 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pressure_trial_evicts_and_drains_request_state() {
        let trial = prepare_trial(
            vec![vec![request(0, 10, 1), request(20, 30, 2)]],
            1_000,
            1,
            2,
        )
        .unwrap();
        let result = run_trial(trial, &args(1, true), None).await.unwrap();

        assert!(result.generator_valid, "{:?}", result.failure_reasons);
        let stats = result.lru_stats.unwrap();
        assert!(stats.evicted_blocks > 0);
        assert_eq!(stats.leases, 0);
        assert_eq!(stats.active_blocks, 0);
        assert_eq!(stats.private_blocks, 0);
        assert!(stats.resident_blocks <= 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shared_prefix_reuse_keeps_one_radix_membership() {
        let trial = prepare_trial(
            vec![vec![request(0, 40, 7), request(10, 30, 7)]],
            1_000,
            1,
            2,
        )
        .unwrap();
        let result = run_trial(trial, &args(4, false), None).await.unwrap();

        assert!(result.generator_valid, "{:?}", result.failure_reasons);
        assert_eq!(result.final_radix_blocks, 1);
        let stats = result.lru_stats.unwrap();
        assert_eq!(stats.leases, 0);
        assert_eq!(stats.active_blocks, 0);
        assert_eq!(stats.inactive_blocks, stats.resident_blocks);
    }
}

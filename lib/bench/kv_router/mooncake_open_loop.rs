// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
#[cfg(not(target_os = "linux"))]
use std::time::Duration;
use std::time::Instant;

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::{
    ObservationError, SyncIndexer, ThreadPoolIndexer, ThreadPoolObservationPlan,
};
use dynamo_kv_router::protocols::{KvCacheEventData, RouterEvent};
use serde::Serialize;
use tokio::sync::{Notify, oneshot};

use super::mooncake_shared::{MooncakeTraceTotals, PreparedMooncakeBenchmark, WorkerTraceEntry};

const RESULT_SCHEMA_VERSION: u32 = 1;
const EMPTY_OPERATION_ID: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct OpenLoopConfig {
    pub query_lanes: usize,
    pub issuer_threads: usize,
    pub spin_us: u64,
    pub issue_lag_diagnostic_threshold_us: u64,
    pub pre_run_quiescence_ms: u64,
    pub issuer_cpus: Vec<usize>,
    pub query_issuer_cpu: Option<usize>,
    pub backend_cpus: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum OperationKind {
    #[default]
    Query,
    Event,
}

#[derive(Clone, Copy, Debug, Default)]
struct QuerySpec {
    start: u32,
    len: u32,
    valid: bool,
}

#[derive(Debug)]
struct QueryCorpus {
    hashes: Box<[LocalBlockHash]>,
    specs: Box<[QuerySpec]>,
}

#[derive(Debug)]
pub(crate) enum MooncakeOperationPayload {
    Query,
    Event(RouterEvent),
}

#[derive(Debug)]
pub(crate) struct MooncakeLogicalOperation {
    pub(crate) id: u32,
    pub(crate) deadline_ns: u64,
    pub(crate) worker_id: u64,
    pub(crate) source_ordinal: usize,
    pub(crate) payload: MooncakeOperationPayload,
}

impl MooncakeLogicalOperation {
    fn priority(&self) -> u8 {
        match self.payload {
            MooncakeOperationPayload::Query => 0,
            MooncakeOperationPayload::Event(_) => 1,
        }
    }
}

#[derive(Debug)]
pub(crate) struct PreparedMooncakeCorpus {
    pub(crate) operations: Vec<MooncakeLogicalOperation>,
    query_corpus: QueryCorpus,
    expected_events_by_worker: Vec<(u64, usize)>,
    totals: MooncakeTraceTotals,
    benchmark_duration_ns: u64,
    block_size: u32,
}

#[cfg(test)]
#[allow(dead_code)]
impl PreparedMooncakeCorpus {
    pub(crate) fn query_hashes(&self, operation_id: u32) -> anyhow::Result<&[LocalBlockHash]> {
        let spec = self
            .query_corpus
            .specs
            .get(operation_id as usize)
            .filter(|spec| spec.valid)
            .ok_or_else(|| anyhow::anyhow!("operation {operation_id} is not a prepared query"))?;
        let start = spec.start as usize;
        let end = start
            .checked_add(spec.len as usize)
            .ok_or_else(|| anyhow::anyhow!("query {operation_id} hash range overflow"))?;
        self.query_corpus.hashes.get(start..end).ok_or_else(|| {
            anyhow::anyhow!(
                "query {operation_id} hash range {start}..{end} exceeds the flattened slab"
            )
        })
    }

    pub(crate) fn test_block_totals(&self) -> (usize, usize) {
        (self.totals.request_blocks, self.totals.event_blocks())
    }
}

#[derive(Debug)]
enum DispatchPayload {
    Query { lane: usize },
    Event(RouterEvent),
}

#[derive(Debug)]
struct DispatchEntry {
    id: u32,
    deadline_ns: u64,
    deadline_group: u32,
    worker_id: u64,
    payload: DispatchPayload,
}

#[derive(Debug)]
pub struct PreparedOpenLoopTrial {
    dispatch: Vec<DispatchEntry>,
    query_corpus: Arc<QueryCorpus>,
    operation_workers: Box<[u64]>,
    lane_capacities: Vec<usize>,
    expected_events_by_worker: Vec<(u64, usize)>,
    deadline_query_counts: Vec<u32>,
    totals: MooncakeTraceTotals,
    benchmark_duration_ns: u64,
    block_size: u32,
}

impl PreparedOpenLoopTrial {
    fn page_touch_untimed(&self) {
        let mut checksum = self.benchmark_duration_ns;
        checksum ^= u64::from(self.block_size);
        for hash in &self.query_corpus.hashes {
            checksum ^= hash.0;
        }
        for spec in &self.query_corpus.specs {
            checksum ^=
                u64::from(spec.start) ^ u64::from(spec.len) ^ u64::from(u8::from(spec.valid));
        }
        for entry in &self.dispatch {
            checksum ^= entry.deadline_ns
                ^ entry.worker_id
                ^ u64::from(entry.id)
                ^ u64::from(entry.deadline_group);
            match &entry.payload {
                DispatchPayload::Query { lane } => checksum ^= *lane as u64,
                DispatchPayload::Event(event) => {
                    checksum ^= event.worker_id ^ event.event.event_id;
                    match &event.event.data {
                        KvCacheEventData::Stored(store) => {
                            for block in &store.blocks {
                                checksum ^= block.block_hash.0 ^ block.tokens_hash.0;
                            }
                        }
                        KvCacheEventData::Removed(remove) => {
                            for hash in &remove.block_hashes {
                                checksum ^= hash.0;
                            }
                        }
                        KvCacheEventData::Cleared => {}
                    }
                }
            }
        }
        for &worker_id in &self.operation_workers {
            checksum ^= worker_id;
        }
        for &capacity in &self.lane_capacities {
            checksum ^= capacity as u64;
        }
        for &(worker_id, event_count) in &self.expected_events_by_worker {
            checksum ^= worker_id ^ event_count as u64;
        }
        for &query_count in &self.deadline_query_counts {
            checksum ^= u64::from(query_count);
        }
        checksum ^= self.totals.requests as u64
            ^ self.totals.stored_events as u64
            ^ self.totals.removed_events as u64
            ^ self.totals.cleared_events as u64
            ^ self.totals.request_blocks as u64
            ^ self.totals.stored_blocks as u64
            ^ self.totals.removed_blocks as u64;
        black_box(checksum);
    }
}

pub(crate) fn prepare_mooncake_corpus(
    prepared: PreparedMooncakeBenchmark,
    inference_worker_duplication_factor: usize,
) -> anyhow::Result<PreparedMooncakeCorpus> {
    let num_trace_workers = prepared.worker_traces.len();
    let estimated_ops = prepared
        .worker_traces
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .checked_mul(inference_worker_duplication_factor)
        .ok_or_else(|| anyhow::anyhow!("open-loop operation count overflow"))?;
    if estimated_ops >= EMPTY_OPERATION_ID as usize {
        anyhow::bail!("open-loop corpus exceeds the u32 operation-ID space");
    }

    let mut operations = Vec::with_capacity(estimated_ops);
    let mut hash_slab = Vec::with_capacity(
        prepared
            .totals
            .request_blocks
            .saturating_mul(inference_worker_duplication_factor),
    );
    let mut events_by_worker = BTreeMap::<u64, usize>::new();

    for replica in 0..inference_worker_duplication_factor {
        for (base_worker, trace) in prepared.worker_traces.iter().enumerate() {
            let worker_id = (base_worker + replica * num_trace_workers) as u64;
            events_by_worker.entry(worker_id).or_default();
            for (source_ordinal, entry) in trace.iter().enumerate() {
                let deadline_ns = entry.timestamp_us.saturating_mul(1_000);
                let (payload, query_spec) = match &entry.entry {
                    WorkerTraceEntry::Request(hashes) => {
                        let start = u32::try_from(hash_slab.len())?;
                        let len = u32::try_from(hashes.len())?;
                        hash_slab.extend_from_slice(hashes);
                        (
                            MooncakeOperationPayload::Query,
                            QuerySpec {
                                start,
                                len,
                                valid: true,
                            },
                        )
                    }
                    WorkerTraceEntry::Event {
                        event,
                        storage_tier,
                    } => {
                        if !storage_tier.is_gpu() {
                            anyhow::bail!(
                                "open-loop replay encountered non-GPU event for worker {worker_id}"
                            );
                        }
                        *events_by_worker.entry(worker_id).or_default() += 1;
                        (
                            MooncakeOperationPayload::Event(RouterEvent::with_storage_tier(
                                worker_id,
                                event.clone(),
                                *storage_tier,
                            )),
                            QuerySpec::default(),
                        )
                    }
                };
                operations.push((
                    MooncakeLogicalOperation {
                        id: 0,
                        deadline_ns,
                        worker_id,
                        source_ordinal,
                        payload,
                    },
                    query_spec,
                ));
            }
        }
    }

    operations.sort_by_key(|(entry, _)| {
        (
            entry.deadline_ns,
            entry.priority(),
            entry.worker_id,
            entry.source_ordinal,
        )
    });

    let mut specs = vec![QuerySpec::default(); operations.len()].into_boxed_slice();
    for (id, (entry, query_spec)) in operations.iter_mut().enumerate() {
        entry.id = id as u32;
        if query_spec.valid {
            specs[id] = *query_spec;
        }
    }
    let operations = operations
        .into_iter()
        .map(|(entry, _)| entry)
        .collect::<Vec<_>>();

    Ok(PreparedMooncakeCorpus {
        operations,
        query_corpus: QueryCorpus {
            hashes: hash_slab.into_boxed_slice(),
            specs,
        },
        expected_events_by_worker: events_by_worker.into_iter().collect(),
        totals: prepared
            .totals
            .expanded(inference_worker_duplication_factor),
        benchmark_duration_ns: prepared.benchmark_duration_ms.saturating_mul(1_000_000),
        block_size: prepared.block_size,
    })
}

pub(crate) fn prepare_open_loop_trial(
    corpus: PreparedMooncakeCorpus,
    query_lanes: usize,
) -> anyhow::Result<PreparedOpenLoopTrial> {
    if query_lanes == 0 {
        anyhow::bail!("open-loop query lane count must be positive");
    }
    if query_lanes > u16::MAX as usize {
        anyhow::bail!("open-loop query lane count exceeds the u16 queue-ID space");
    }

    let PreparedMooncakeCorpus {
        operations,
        query_corpus,
        expected_events_by_worker,
        totals,
        benchmark_duration_ns,
        block_size,
    } = corpus;
    let mut dispatch = Vec::with_capacity(operations.len());
    let mut operation_workers = Vec::with_capacity(operations.len());
    let mut lane_capacities = vec![0usize; query_lanes];
    let mut deadline_query_counts = Vec::<u32>::new();
    let mut previous_deadline = None;

    for operation in operations {
        operation_workers.push(operation.worker_id);
        if previous_deadline != Some(operation.deadline_ns) {
            previous_deadline = Some(operation.deadline_ns);
            deadline_query_counts.push(0);
        }
        let deadline_group = deadline_query_counts.len() - 1;
        let payload = match operation.payload {
            MooncakeOperationPayload::Query => {
                let lane = operation.worker_id as usize % query_lanes;
                lane_capacities[lane] += 1;
                deadline_query_counts[deadline_group] =
                    deadline_query_counts[deadline_group].saturating_add(1);
                DispatchPayload::Query { lane }
            }
            MooncakeOperationPayload::Event(event) => DispatchPayload::Event(event),
        };
        dispatch.push(DispatchEntry {
            id: operation.id,
            deadline_ns: operation.deadline_ns,
            deadline_group: deadline_group as u32,
            worker_id: operation.worker_id,
            payload,
        });
    }

    Ok(PreparedOpenLoopTrial {
        dispatch,
        query_corpus: Arc::new(query_corpus),
        operation_workers: operation_workers.into_boxed_slice(),
        lane_capacities,
        expected_events_by_worker,
        deadline_query_counts,
        totals,
        benchmark_duration_ns,
        block_size,
    })
}

struct QueryLane {
    slots: Box<[AtomicU32]>,
    published: AtomicUsize,
    closed: AtomicBool,
    notify: Notify,
}

impl QueryLane {
    fn new(capacity: usize) -> Self {
        let slots = (0..capacity)
            .map(|_| AtomicU32::new(EMPTY_OPERATION_ID))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            published: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    fn write(&self, cursor: usize, id: u32) -> Result<(), IssuerFailure> {
        let Some(slot) = self.slots.get(cursor) else {
            return Err(IssuerFailure::QueryLaneOverflow);
        };
        slot.store(id, Ordering::Relaxed);
        Ok(())
    }

    fn publish(&self, count: usize) {
        self.published.store(count, Ordering::Release);
        self.notify.notify_one();
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_one();
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct QueryCompletion {
    id: u32,
    started_ns: u64,
    finished_ns: u64,
    success: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum QueryLaneFailure {
    #[default]
    None,
    InvalidOperationId,
    InvalidHashRange,
    MissingPublishedId,
    CompletionOverflow,
}

struct QueryLaneResult {
    completions: Box<[QueryCompletion]>,
    written: usize,
    drain_ns: u64,
    failure: QueryLaneFailure,
}

async fn query_lane_worker<T: SyncIndexer>(
    indexer: Arc<ThreadPoolIndexer<T>>,
    lane: Arc<QueryLane>,
    corpus: Arc<QueryCorpus>,
    epoch: Instant,
    ready: oneshot::Sender<()>,
) -> QueryLaneResult {
    let mut completions = vec![QueryCompletion::default(); lane.slots.len()].into_boxed_slice();
    let mut consumed = 0usize;
    let mut failure = QueryLaneFailure::None;
    let _ = ready.send(());
    loop {
        let published = lane.published.load(Ordering::Acquire);
        while consumed < published {
            let id = lane.slots[consumed].load(Ordering::Relaxed);
            if id == EMPTY_OPERATION_ID {
                failure = QueryLaneFailure::MissingPublishedId;
                break;
            }
            let Some(spec) = corpus.specs.get(id as usize).copied() else {
                failure = QueryLaneFailure::InvalidOperationId;
                break;
            };
            let start = spec.start as usize;
            let end = start.saturating_add(spec.len as usize);
            let Some(hashes) = corpus.hashes.get(start..end).filter(|_| spec.valid) else {
                failure = QueryLaneFailure::InvalidHashRange;
                break;
            };
            let Some(slot) = completions.get_mut(consumed) else {
                failure = QueryLaneFailure::CompletionOverflow;
                break;
            };
            let started_ns = elapsed_ns(epoch);
            let output = indexer.backend().find_matches(hashes, false);
            black_box(output);
            let finished_ns = elapsed_ns(epoch);
            *slot = QueryCompletion {
                id,
                started_ns,
                finished_ns,
                success: true,
            };
            consumed += 1;
        }

        if failure != QueryLaneFailure::None {
            break;
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

    QueryLaneResult {
        completions,
        written: consumed,
        drain_ns: elapsed_ns(epoch),
        failure,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct IssueRecord {
    scheduled_ns: u64,
    accepted_ns: u64,
    queue_id: u16,
    kind: OperationKind,
    accepted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IssuerFailure {
    QueryLaneOverflow,
    Observation,
    Affinity,
    DispatchMismatch,
    PeerFailed,
}

#[derive(Clone, Copy, Debug)]
struct LocalIssueRecord {
    id: u32,
    record: IssueRecord,
}

impl Default for LocalIssueRecord {
    fn default() -> Self {
        Self {
            id: EMPTY_OPERATION_ID,
            record: IssueRecord::default(),
        }
    }
}

struct IssuerOutput {
    records: Box<[LocalIssueRecord]>,
    written: usize,
    issuer_cpu_ns: u64,
    failure: Option<IssuerFailure>,
}

struct IssuerStorage {
    records: Box<[LocalIssueRecord]>,
    written: usize,
    lane_cursors: Box<[usize]>,
    touched_flags: Box<[bool]>,
    touched_lanes: Vec<usize>,
}

impl IssuerStorage {
    fn new(operation_count: usize, lane_count: usize) -> Self {
        Self {
            records: vec![LocalIssueRecord::default(); operation_count].into_boxed_slice(),
            written: 0,
            lane_cursors: vec![0; lane_count].into_boxed_slice(),
            touched_flags: vec![false; lane_count].into_boxed_slice(),
            touched_lanes: Vec::with_capacity(lane_count),
        }
    }

    fn record(&mut self, id: u32, record: IssueRecord) -> Result<(), IssuerFailure> {
        let Some(slot) = self.records.get_mut(self.written) else {
            return Err(IssuerFailure::DispatchMismatch);
        };
        *slot = LocalIssueRecord { id, record };
        self.written += 1;
        Ok(())
    }
}

struct IssuerAnalysisInput {
    records: Box<[IssueRecord]>,
    producer_stop_ns: u64,
    issuer_cpu_ns: u64,
    failure: Option<IssuerFailure>,
}

struct QueryIssuerContext<'a> {
    lanes: &'a [Arc<QueryLane>],
    deadline_ready: &'a [AtomicBool],
    peer_failed: &'a AtomicBool,
    clock: &'a BenchmarkClock,
    start_ns: u64,
}

struct EventIssuerContext<'a, T: SyncIndexer> {
    indexer: &'a ThreadPoolIndexer<T>,
    deadline_ready: &'a [AtomicBool],
    peer_failed: &'a AtomicBool,
    clock: &'a BenchmarkClock,
    start_ns: u64,
}

fn issue_queries(
    context: QueryIssuerContext<'_>,
    dispatch: Vec<DispatchEntry>,
    mut storage: IssuerStorage,
    initial_failure: Option<IssuerFailure>,
) -> IssuerOutput {
    let cpu_started = thread_cpu_time_ns();
    let mut failure = initial_failure;
    let mut entries = dispatch.into_iter().peekable();

    while failure.is_none() {
        let Some(first) = entries.peek() else {
            break;
        };
        let deadline_ns = first.deadline_ns;
        let deadline_group = first.deadline_group as usize;
        context
            .clock
            .wait_until(context.start_ns.saturating_add(deadline_ns));
        storage.touched_lanes.clear();

        while entries
            .peek()
            .is_some_and(|entry| entry.deadline_ns == deadline_ns)
        {
            let Some(entry) = entries.next() else {
                failure = Some(IssuerFailure::DispatchMismatch);
                break;
            };
            let scheduled_ns = context.start_ns.saturating_add(deadline_ns);
            match entry.payload {
                DispatchPayload::Query { lane, .. } => {
                    if let Err(error) =
                        context.lanes[lane].write(storage.lane_cursors[lane], entry.id)
                    {
                        failure = Some(error);
                        break;
                    }
                    storage.lane_cursors[lane] += 1;
                    if !storage.touched_flags[lane] {
                        storage.touched_flags[lane] = true;
                        storage.touched_lanes.push(lane);
                    }
                    let record = IssueRecord {
                        scheduled_ns,
                        accepted_ns: context.clock.now_ns(),
                        queue_id: lane as u16,
                        kind: OperationKind::Query,
                        accepted: true,
                    };
                    if let Err(error) = storage.record(entry.id, record) {
                        failure = Some(error);
                        break;
                    }
                }
                DispatchPayload::Event(_) => failure = Some(IssuerFailure::DispatchMismatch),
            }
        }
        publish_touched(
            context.lanes,
            &storage.lane_cursors,
            &mut storage.touched_flags,
            &storage.touched_lanes,
        );
        if failure.is_some() {
            context.peer_failed.store(true, Ordering::Release);
            break;
        }
        let Some(ready) = context.deadline_ready.get(deadline_group) else {
            failure = Some(IssuerFailure::DispatchMismatch);
            break;
        };
        ready.store(true, Ordering::Release);
    }

    if failure.is_some() {
        context.peer_failed.store(true, Ordering::Release);
    }
    IssuerOutput {
        records: storage.records,
        written: storage.written,
        issuer_cpu_ns: thread_cpu_time_ns().saturating_sub(cpu_started),
        failure,
    }
}

fn issue_events<T: SyncIndexer>(
    context: EventIssuerContext<'_, T>,
    dispatch: Vec<DispatchEntry>,
    mut storage: IssuerStorage,
    initial_failure: Option<IssuerFailure>,
) -> IssuerOutput {
    let cpu_started = thread_cpu_time_ns();
    let mut failure = initial_failure;
    let mut entries = dispatch.into_iter().peekable();

    while failure.is_none() {
        let Some(first) = entries.peek() else {
            break;
        };
        let deadline_ns = first.deadline_ns;
        let deadline_group = first.deadline_group as usize;
        context
            .clock
            .wait_until(context.start_ns.saturating_add(deadline_ns));
        if !wait_for_deadline_queries(
            context.deadline_ready.get(deadline_group),
            context.peer_failed,
        ) {
            failure = Some(IssuerFailure::PeerFailed);
            break;
        }

        while entries
            .peek()
            .is_some_and(|entry| entry.deadline_ns == deadline_ns)
        {
            let Some(entry) = entries.next() else {
                failure = Some(IssuerFailure::DispatchMismatch);
                break;
            };
            let scheduled_ns = context.start_ns.saturating_add(deadline_ns);
            match entry.payload {
                DispatchPayload::Query { .. } => {
                    failure = Some(IssuerFailure::DispatchMismatch);
                    break;
                }
                DispatchPayload::Event(event) => {
                    let enqueue_result = enqueue_event(&context, event, entry.id);
                    match enqueue_result {
                        Ok((event_worker, accepted_ns)) => {
                            let record = IssueRecord {
                                scheduled_ns,
                                accepted_ns,
                                queue_id: event_worker as u16,
                                kind: OperationKind::Event,
                                accepted: true,
                            };
                            if let Err(error) = storage.record(entry.id, record) {
                                failure = Some(error);
                                break;
                            }
                        }
                        Err(_) => {
                            failure = Some(IssuerFailure::Observation);
                            break;
                        }
                    }
                }
            }
        }
    }

    if failure.is_some() {
        context.peer_failed.store(true, Ordering::Release);
    }
    IssuerOutput {
        records: storage.records,
        written: storage.written,
        issuer_cpu_ns: thread_cpu_time_ns().saturating_sub(cpu_started),
        failure,
    }
}

fn enqueue_event<T: SyncIndexer>(
    context: &EventIssuerContext<'_, T>,
    event: RouterEvent,
    id: u32,
) -> Result<(usize, u64), ()> {
    context
        .indexer
        .enqueue_active_observation_owned(event, id, context.clock.epoch())
        .map(|receipt| (receipt.event_worker, receipt.accepted_ns))
        .map_err(|_| ())
}

fn wait_for_deadline_queries(
    deadline_ready: Option<&AtomicBool>,
    peer_failed: &AtomicBool,
) -> bool {
    let Some(ready) = deadline_ready else {
        return false;
    };
    while !ready.load(Ordering::Acquire) {
        if peer_failed.load(Ordering::Acquire) {
            return false;
        }
        std::hint::spin_loop();
    }
    true
}

fn publish_touched(
    lanes: &[Arc<QueryLane>],
    lane_cursors: &[usize],
    touched_flags: &mut [bool],
    touched_lanes: &[usize],
) {
    for &lane in touched_lanes {
        lanes[lane].publish(lane_cursors[lane]);
        touched_flags[lane] = false;
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Distribution {
    pub p50_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub max_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct OpenLoopResult {
    pub schema_version: u32,
    pub backend: String,
    pub timer: &'static str,
    pub benchmark_duration_ms: u64,
    pub block_size: u32,
    pub pre_run_quiescence_ms: u64,
    pub query_lanes: usize,
    pub issuer_threads: usize,
    pub event_workers: usize,
    pub issuer_cpus: Vec<usize>,
    pub query_issuer_cpu: Option<usize>,
    pub backend_cpus: Vec<usize>,
    pub total_requests: usize,
    pub total_events: usize,
    pub total_stored_events: usize,
    pub total_removed_events: usize,
    pub total_cleared_events: usize,
    pub total_request_blocks: usize,
    pub total_stored_blocks: usize,
    pub total_removed_blocks: usize,
    pub total_logical_ops: usize,
    pub total_block_ops: usize,
    pub offered_logical_ops_per_sec: f64,
    pub actual_issue_logical_ops_per_sec: f64,
    pub achieved_logical_ops_per_sec: f64,
    pub offered_block_ops_per_sec: f64,
    pub actual_issue_block_ops_per_sec: f64,
    pub achieved_block_ops_per_sec: f64,
    pub read_issue_lag: Distribution,
    pub update_issue_lag: Distribution,
    pub generator_gate: &'static str,
    pub query_queue_wait: Distribution,
    pub query_service: Distribution,
    pub query_scheduled_to_finished: Distribution,
    pub update_accepted_to_finished: Distribution,
    pub update_scheduled_to_finished: Distribution,
    pub delayed_reads: usize,
    pub delayed_updates: usize,
    pub queue_depth_at_stop: Vec<usize>,
    pub queued_queries_at_stop: usize,
    pub maximum_query_queue_depth: usize,
    pub outstanding_updates_at_stop: usize,
    pub maximum_outstanding_updates: usize,
    pub post_acceptance_completion_races: usize,
    pub issuer_cpu_ns: u64,
    pub issue_span_ns: u64,
    pub drain_ns: u64,
    pub generator_valid: bool,
    pub kept_up: bool,
    pub failure_reasons: Vec<String>,
}

fn partition_dispatch(
    dispatch: Vec<DispatchEntry>,
    event_issuer_count: usize,
) -> anyhow::Result<(Vec<DispatchEntry>, Vec<Vec<DispatchEntry>>)> {
    if event_issuer_count == 0 {
        anyhow::bail!("event-issuer count must be positive");
    }
    let logical_workers = dispatch
        .iter()
        .map(|entry| entry.worker_id as usize)
        .max()
        .map_or(0, |worker| worker + 1);
    let workers_per_issuer = logical_workers.div_ceil(event_issuer_count).max(1);
    let event_shard_for =
        |worker_id: u64| (worker_id as usize / workers_per_issuer).min(event_issuer_count - 1);
    let mut query_count = 0usize;
    let mut event_counts = vec![0usize; event_issuer_count];
    for entry in &dispatch {
        match entry.payload {
            DispatchPayload::Query { .. } => query_count += 1,
            DispatchPayload::Event(_) => event_counts[event_shard_for(entry.worker_id)] += 1,
        }
    }
    let mut queries = Vec::with_capacity(query_count);
    let mut event_shards = event_counts
        .into_iter()
        .map(Vec::with_capacity)
        .collect::<Vec<_>>();
    for entry in dispatch {
        match entry.payload {
            DispatchPayload::Query { .. } => queries.push(entry),
            DispatchPayload::Event(_) => {
                event_shards[event_shard_for(entry.worker_id)].push(entry);
            }
        }
    }
    Ok((queries, event_shards))
}

fn aggregate_issuer_outputs(
    outputs: Vec<IssuerOutput>,
    operation_count: usize,
    producer_stop_ns: u64,
) -> IssuerAnalysisInput {
    let mut records = vec![IssueRecord::default(); operation_count].into_boxed_slice();
    let mut issuer_cpu_ns = 0u64;
    let mut failure = None;
    for output in outputs {
        issuer_cpu_ns = issuer_cpu_ns.saturating_add(output.issuer_cpu_ns);
        failure = failure.or(output.failure);
        for local in output.records[..output.written].iter().copied() {
            let Some(slot) = records.get_mut(local.id as usize) else {
                failure = failure.or(Some(IssuerFailure::DispatchMismatch));
                continue;
            };
            if slot.accepted {
                failure = failure.or(Some(IssuerFailure::DispatchMismatch));
                continue;
            }
            *slot = local.record;
        }
    }
    IssuerAnalysisInput {
        records,
        producer_stop_ns,
        issuer_cpu_ns,
        failure,
    }
}

fn deadline_readiness(query_counts: Vec<u32>) -> Box<[AtomicBool]> {
    query_counts
        .into_iter()
        .map(|count| AtomicBool::new(count == 0))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

pub async fn run_open_loop<T: SyncIndexer>(
    backend_name: &str,
    indexer: Arc<ThreadPoolIndexer<T>>,
    trial: PreparedOpenLoopTrial,
    config: OpenLoopConfig,
) -> anyhow::Result<OpenLoopResult> {
    trial.page_touch_untimed();
    let clock = BenchmarkClock::new(config.spin_us.saturating_mul(1_000))?;
    let epoch = clock.epoch();

    for spec in trial
        .query_corpus
        .specs
        .iter()
        .filter(|spec| spec.valid)
        .take(128)
    {
        let start = spec.start as usize;
        let end = start + spec.len as usize;
        black_box(
            indexer
                .backend()
                .find_matches(&trial.query_corpus.hashes[start..end], false),
        );
    }

    let lanes = trial
        .lane_capacities
        .iter()
        .map(|&capacity| Arc::new(QueryLane::new(capacity)))
        .collect::<Vec<_>>();
    let mut ready_receivers = Vec::with_capacity(lanes.len());
    let mut query_tasks = Vec::with_capacity(lanes.len());
    for lane in &lanes {
        let (ready_tx, ready_rx) = oneshot::channel();
        ready_receivers.push(ready_rx);
        query_tasks.push(tokio::spawn(query_lane_worker(
            Arc::clone(&indexer),
            Arc::clone(lane),
            Arc::clone(&trial.query_corpus),
            epoch,
            ready_tx,
        )));
    }
    for receiver in ready_receivers {
        receiver.await?;
    }

    let issuer_count = config.issuer_threads;
    let operation_count = trial.dispatch.len();
    let (query_dispatch, event_dispatch_shards) = partition_dispatch(trial.dispatch, issuer_count)?;
    let query_storage = IssuerStorage::new(query_dispatch.len(), lanes.len());
    let issuer_storages = event_dispatch_shards
        .iter()
        .map(|shard| IssuerStorage::new(shard.len(), lanes.len()))
        .collect::<Vec<_>>();
    let deadline_ready = deadline_readiness(trial.deadline_query_counts);
    let observation = indexer
        .begin_observation(ThreadPoolObservationPlan {
            epoch,
            expected_events_by_worker: trial.expected_events_by_worker,
        })
        .await?;
    let peer_failed = AtomicBool::new(false);
    let start_signal = AtomicU64::new(0);
    let ready_barrier = Barrier::new(issuer_count + 2);
    let start_barrier = Barrier::new(issuer_count + 2);
    let (start_ns, issuer_outputs) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(issuer_count + 1);
        let query_clock = &clock;
        let query_lanes = &lanes;
        let query_deadline_ready = deadline_ready.as_ref();
        let query_peer_failed = &peer_failed;
        let query_ready_barrier = &ready_barrier;
        let query_start_barrier = &start_barrier;
        let query_start_signal = &start_signal;
        let query_cpu = config.query_issuer_cpu;
        handles.push(scope.spawn(move || {
            let mut initial_failure = pin_current_thread(query_cpu)
                .err()
                .map(|_| IssuerFailure::Affinity);
            if initial_failure.is_some() {
                query_peer_failed.store(true, Ordering::Release);
            }
            query_ready_barrier.wait();
            query_start_barrier.wait();
            if initial_failure.is_none() && query_peer_failed.load(Ordering::Acquire) {
                initial_failure = Some(IssuerFailure::PeerFailed);
            }
            issue_queries(
                QueryIssuerContext {
                    lanes: query_lanes,
                    deadline_ready: query_deadline_ready,
                    peer_failed: query_peer_failed,
                    clock: query_clock,
                    start_ns: query_start_signal.load(Ordering::Acquire),
                },
                query_dispatch,
                query_storage,
                initial_failure,
            )
        }));
        for (issuer_idx, (dispatch, storage)) in event_dispatch_shards
            .into_iter()
            .zip(issuer_storages)
            .enumerate()
        {
            let issuer_cpu = config.issuer_cpus.get(issuer_idx).copied();
            let issuer_clock = &clock;
            let issuer_indexer = indexer.as_ref();
            let issuer_deadline_ready = deadline_ready.as_ref();
            let issuer_peer_failed = &peer_failed;
            let issuer_ready_barrier = &ready_barrier;
            let issuer_start_barrier = &start_barrier;
            let issuer_start_signal = &start_signal;
            handles.push(scope.spawn(move || {
                let mut initial_failure = pin_current_thread(issuer_cpu)
                    .err()
                    .map(|_| IssuerFailure::Affinity);
                if initial_failure.is_some() {
                    issuer_peer_failed.store(true, Ordering::Release);
                }
                issuer_ready_barrier.wait();
                issuer_start_barrier.wait();
                if initial_failure.is_none() && issuer_peer_failed.load(Ordering::Acquire) {
                    initial_failure = Some(IssuerFailure::PeerFailed);
                }
                issue_events(
                    EventIssuerContext {
                        indexer: issuer_indexer,
                        deadline_ready: issuer_deadline_ready,
                        peer_failed: issuer_peer_failed,
                        clock: issuer_clock,
                        start_ns: issuer_start_signal.load(Ordering::Acquire),
                    },
                    dispatch,
                    storage,
                    initial_failure,
                )
            }));
        }

        ready_barrier.wait();
        let start_ns = clock.now_ns().saturating_add(20_000_000);
        start_signal.store(start_ns, Ordering::Release);
        start_barrier.wait();
        let mut outputs = Vec::with_capacity(issuer_count + 1);
        for handle in handles {
            outputs.push(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("open-loop issuer panicked"))?,
            );
        }
        Ok::<_, anyhow::Error>((start_ns, outputs))
    })?;

    let producer_stop_ns = clock.now_ns();
    let drain = observation.close_observed_producers();
    let issuer_analysis =
        aggregate_issuer_outputs(issuer_outputs, operation_count, producer_stop_ns);

    for lane in &lanes {
        lane.close();
    }
    let seal_future = drain.seal();
    let query_future = async {
        let mut results = Vec::with_capacity(query_tasks.len());
        for task in query_tasks {
            results.push(task.await?);
        }
        Ok::<_, tokio::task::JoinError>(results)
    };
    let (sealed, query_results) = tokio::join!(seal_future, query_future);
    let sealed = sealed?;
    let query_results = query_results?;
    let query_drain_ns = query_results
        .iter()
        .map(|result| result.drain_ns)
        .max()
        .unwrap_or(start_ns);
    let end_ns = query_drain_ns.max(sealed.latest_seal_ns());
    let snapshot = sealed.harvest().await?;

    Ok(analyze_result(
        backend_name,
        &clock,
        start_ns,
        end_ns,
        trial.totals,
        trial.benchmark_duration_ns,
        trial.block_size,
        &trial.operation_workers,
        config,
        issuer_analysis,
        query_results,
        snapshot,
    ))
}

#[allow(clippy::too_many_arguments)]
fn analyze_result(
    backend_name: &str,
    clock: &BenchmarkClock,
    start_ns: u64,
    end_ns: u64,
    totals: MooncakeTraceTotals,
    benchmark_duration_ns: u64,
    block_size: u32,
    operation_workers: &[u64],
    config: OpenLoopConfig,
    issuer: IssuerAnalysisInput,
    query_results: Vec<QueryLaneResult>,
    snapshot: dynamo_kv_router::indexer::ThreadPoolObservationSnapshot,
) -> OpenLoopResult {
    let IssuerAnalysisInput {
        records,
        producer_stop_ns,
        issuer_cpu_ns,
        failure: issuer_failure,
    } = issuer;
    let mut failure_reasons = Vec::new();
    if let Some(failure) = issuer_failure {
        failure_reasons.push(format!("issuer_{failure:?}"));
    }

    let mut expected_queries_by_lane = vec![Vec::new(); config.query_lanes];
    let mut expected_events_by_worker = BTreeMap::<u64, Vec<u32>>::new();
    for (id, record) in records
        .iter()
        .enumerate()
        .filter(|(_, record)| record.accepted)
    {
        match record.kind {
            OperationKind::Query => {
                if let Some(expected) = expected_queries_by_lane.get_mut(record.queue_id as usize) {
                    expected.push(id as u32);
                }
            }
            OperationKind::Event => {
                let worker_id = operation_workers[id];
                expected_events_by_worker
                    .entry(worker_id)
                    .or_default()
                    .push(id as u32);
            }
        }
    }

    let mut query_completions = vec![None; records.len()];
    for (lane_idx, result) in query_results.into_iter().enumerate() {
        if result.failure != QueryLaneFailure::None {
            failure_reasons.push(format!("query_lane_{:?}", result.failure));
        }
        let actual_ids = result.completions[..result.written]
            .iter()
            .map(|completion| completion.id)
            .collect::<Vec<_>>();
        if expected_queries_by_lane.get(lane_idx) != Some(&actual_ids) {
            failure_reasons.push(format!("query_lane_order_{lane_idx}"));
        }
        for completion in result.completions[..result.written].iter().copied() {
            let slot = &mut query_completions[completion.id as usize];
            if slot.replace(completion).is_some() {
                failure_reasons.push("duplicate_query_completion".to_string());
            }
        }
    }

    let mut event_completions = vec![None; records.len()];
    let mut actual_events_by_worker = BTreeMap::<u64, Vec<u32>>::new();
    for (worker_idx, buffer) in snapshot.buffers.iter().enumerate() {
        if buffer.overflowed() {
            failure_reasons.push("event_completion_overflow".to_string());
        }
        if snapshot.seals[worker_idx].written != buffer.records().len() {
            failure_reasons.push(format!("event_worker_seal_count_{worker_idx}"));
        }
        for completion in buffer.records().iter().copied() {
            let id = completion.correlation_id as usize;
            let Some(record) = records.get(id) else {
                failure_reasons.push("event_completion_id_out_of_range".to_string());
                continue;
            };
            if record.kind != OperationKind::Event {
                failure_reasons.push(format!("non_event_completion_{id}"));
                continue;
            }
            if record.queue_id as usize != worker_idx {
                failure_reasons.push(format!("event_completion_queue_{id}"));
            }
            actual_events_by_worker
                .entry(operation_workers[id])
                .or_default()
                .push(completion.correlation_id);
            let slot = &mut event_completions[id];
            if slot.replace(completion).is_some() {
                failure_reasons.push("duplicate_event_completion".to_string());
            }
        }
    }
    for (worker_id, expected) in &expected_events_by_worker {
        if actual_events_by_worker.get(worker_id) != Some(expected) {
            failure_reasons.push(format!("event_worker_fifo_{worker_id}"));
        }
    }

    let mut read_issue_lag = Vec::with_capacity(totals.requests);
    let mut update_issue_lag = Vec::with_capacity(totals.events());
    let mut query_queue_wait = Vec::with_capacity(totals.requests);
    let mut query_service = Vec::with_capacity(totals.requests);
    let mut query_end_to_end = Vec::with_capacity(totals.requests);
    let mut update_accepted_to_finished = Vec::with_capacity(totals.events());
    let mut update_end_to_end = Vec::with_capacity(totals.events());
    let mut delayed_reads = 0usize;
    let mut delayed_updates = 0usize;
    let mut post_acceptance_completion_races = 0usize;
    let tolerance_ns = config
        .issue_lag_diagnostic_threshold_us
        .saturating_mul(1_000);
    let mut accepted_query_edges = Vec::with_capacity(totals.requests * 2);
    let mut accepted_update_edges = Vec::with_capacity(totals.events() * 2);

    for (id, record) in records.iter().enumerate() {
        if !record.accepted {
            failure_reasons.push(format!("unissued_operation_{id}"));
            continue;
        }
        let issue_lag = record.accepted_ns.saturating_sub(record.scheduled_ns);
        match record.kind {
            OperationKind::Query => {
                read_issue_lag.push(issue_lag);
                delayed_reads += usize::from(issue_lag > tolerance_ns);
                let Some(completion) = query_completions[id] else {
                    failure_reasons.push(format!("missing_query_completion_{id}"));
                    continue;
                };
                if !completion.success {
                    failure_reasons.push(format!("failed_query_{id}"));
                }
                query_queue_wait.push(completion.started_ns.saturating_sub(record.accepted_ns));
                query_service.push(completion.finished_ns.saturating_sub(completion.started_ns));
                query_end_to_end.push(completion.finished_ns.saturating_sub(record.scheduled_ns));
                accepted_query_edges.push((record.accepted_ns, 1i8));
                accepted_query_edges.push((completion.started_ns.max(record.accepted_ns), -1i8));
            }
            OperationKind::Event => {
                if record.queue_id == u16::MAX {
                    failure_reasons.push(format!("missing_event_worker_{id}"));
                }
                update_issue_lag.push(issue_lag);
                delayed_updates += usize::from(issue_lag > tolerance_ns);
                let Some(completion) = event_completions[id] else {
                    failure_reasons.push(format!("missing_event_completion_{id}"));
                    continue;
                };
                if !completion.success {
                    failure_reasons.push(format!("failed_event_{id}"));
                }
                if completion.finished_ns < record.accepted_ns {
                    post_acceptance_completion_races += 1;
                }
                update_accepted_to_finished
                    .push(completion.finished_ns.saturating_sub(record.accepted_ns));
                update_end_to_end.push(completion.finished_ns.saturating_sub(record.scheduled_ns));
                accepted_update_edges.push((record.accepted_ns, 1i8));
                accepted_update_edges.push((completion.finished_ns.max(record.accepted_ns), -1i8));
            }
        }
    }

    let maximum_query_queue_depth = maximum_depth(&mut accepted_query_edges);
    let maximum_outstanding = maximum_depth(&mut accepted_update_edges);
    let queued_queries_at_stop = records
        .iter()
        .enumerate()
        .filter(|(_, record)| record.kind == OperationKind::Query && record.accepted)
        .filter(|(id, record)| {
            query_completions[*id].is_some_and(|completion| {
                record.accepted_ns <= producer_stop_ns && completion.started_ns > producer_stop_ns
            })
        })
        .count();
    let outstanding_at_stop = event_completions
        .iter()
        .flatten()
        .filter(|completion| completion.finished_ns > producer_stop_ns)
        .count();

    let issue_span_ns = records
        .iter()
        .filter(|record| record.accepted)
        .map(|record| record.accepted_ns)
        .max()
        .unwrap_or(start_ns)
        .saturating_sub(start_ns);
    let drain_ns = end_ns.saturating_sub(producer_stop_ns);
    let elapsed_ns = end_ns.saturating_sub(start_ns).max(1);
    let total_logical_ops = totals.requests + totals.events();
    let total_block_ops = totals.total_block_ops();
    let offered_seconds = benchmark_duration_ns.max(1) as f64 / 1e9;
    let achieved_seconds = elapsed_ns as f64 / 1e9;
    let issue_seconds = issue_span_ns.max(1) as f64 / 1e9;

    let issue_span_valid = issue_span_ns <= benchmark_duration_ns.saturating_mul(101) / 100;
    let generator_valid = failure_reasons.is_empty() && issue_span_valid;
    let kept_up = generator_valid && elapsed_ns <= benchmark_duration_ns.saturating_mul(110) / 100;

    OpenLoopResult {
        schema_version: RESULT_SCHEMA_VERSION,
        backend: backend_name.to_string(),
        timer: clock.timer_name(),
        benchmark_duration_ms: benchmark_duration_ns / 1_000_000,
        block_size,
        pre_run_quiescence_ms: config.pre_run_quiescence_ms,
        query_lanes: config.query_lanes,
        issuer_threads: config.issuer_threads,
        event_workers: snapshot.buffers.len(),
        issuer_cpus: config.issuer_cpus,
        query_issuer_cpu: config.query_issuer_cpu,
        backend_cpus: config.backend_cpus,
        total_requests: totals.requests,
        total_events: totals.events(),
        total_stored_events: totals.stored_events,
        total_removed_events: totals.removed_events,
        total_cleared_events: totals.cleared_events,
        total_request_blocks: totals.request_blocks,
        total_stored_blocks: totals.stored_blocks,
        total_removed_blocks: totals.removed_blocks,
        total_logical_ops,
        total_block_ops,
        offered_logical_ops_per_sec: total_logical_ops as f64 / offered_seconds,
        actual_issue_logical_ops_per_sec: total_logical_ops as f64 / issue_seconds,
        achieved_logical_ops_per_sec: total_logical_ops as f64 / achieved_seconds,
        offered_block_ops_per_sec: total_block_ops as f64 / offered_seconds,
        actual_issue_block_ops_per_sec: total_block_ops as f64 / issue_seconds,
        achieved_block_ops_per_sec: total_block_ops as f64 / achieved_seconds,
        read_issue_lag: distribution(read_issue_lag),
        update_issue_lag: distribution(update_issue_lag),
        generator_gate: "issue_span_exact_completion",
        query_queue_wait: distribution(query_queue_wait),
        query_service: distribution(query_service),
        query_scheduled_to_finished: distribution(query_end_to_end),
        update_accepted_to_finished: distribution(update_accepted_to_finished),
        update_scheduled_to_finished: distribution(update_end_to_end),
        delayed_reads,
        delayed_updates,
        queue_depth_at_stop: snapshot.queue_depth_at_stop,
        queued_queries_at_stop,
        maximum_query_queue_depth,
        outstanding_updates_at_stop: outstanding_at_stop,
        maximum_outstanding_updates: maximum_outstanding,
        post_acceptance_completion_races,
        issuer_cpu_ns,
        issue_span_ns,
        drain_ns,
        generator_valid,
        kept_up,
        failure_reasons,
    }
}

fn maximum_depth(edges: &mut [(u64, i8)]) -> usize {
    edges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| right.1.cmp(&left.1)));
    let mut depth = 0isize;
    let mut maximum = 0isize;
    for &(_, delta) in edges.iter() {
        depth += delta as isize;
        maximum = maximum.max(depth);
    }
    maximum.max(0) as usize
}

fn distribution(mut values: Vec<u64>) -> Distribution {
    if values.is_empty() {
        return Distribution::default();
    }
    values.sort_unstable();
    Distribution {
        p50_ns: nearest_rank(&values, 50, 100),
        p99_ns: nearest_rank(&values, 99, 100),
        p999_ns: nearest_rank(&values, 999, 1_000),
        max_ns: values.last().copied().unwrap_or_default(),
    }
}

fn nearest_rank(values: &[u64], numerator: usize, denominator: usize) -> u64 {
    let rank = values
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .max(1);
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn elapsed_ns(epoch: Instant) -> u64 {
    epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64
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
        let monotonic_epoch_ns = monotonic_now_ns()?;
        Ok(Self {
            epoch: Instant::now(),
            spin_ns,
            #[cfg(target_os = "linux")]
            monotonic_epoch_ns,
        })
    }

    fn epoch(&self) -> Instant {
        self.epoch
    }

    fn now_ns(&self) -> u64 {
        elapsed_ns(self.epoch)
    }

    fn wait_until(&self, target_ns: u64) {
        let sleep_target = target_ns.saturating_sub(self.spin_ns);
        let now = self.now_ns();
        #[cfg(target_os = "linux")]
        if sleep_target > now {
            sleep_until_monotonic(self.monotonic_epoch_ns.saturating_add(sleep_target));
        }
        #[cfg(not(target_os = "linux"))]
        {
            if sleep_target > now {
                std::thread::sleep(Duration::from_nanos(sleep_target - now));
            }
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
fn monotonic_now_ns() -> anyhow::Result<u64> {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok((timestamp.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(timestamp.tv_nsec as u64))
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
        if rc == 0 {
            return;
        }
        if rc != libc::EINTR {
            return;
        }
    }
}

fn thread_cpu_time_ns() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let mut timestamp = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut timestamp) };
        if rc == 0 {
            return (timestamp.tv_sec as u64)
                .saturating_mul(1_000_000_000)
                .saturating_add(timestamp.tv_nsec as u64);
        }
    }
    0
}

fn pin_current_thread(cpu: Option<usize>) -> Result<(), ()> {
    let Some(cpu) = cpu else {
        return Ok(());
    };
    #[cfg(target_os = "linux")]
    {
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        unsafe {
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
        }
        let rc = unsafe {
            libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set as *const libc::cpu_set_t,
            )
        };
        if rc != 0 {
            return Err(());
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
    Ok(())
}

pub fn parse_cpu_list(value: &str) -> anyhow::Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let Some((start, end)) = part.split_once('-') else {
            cpus.push(part.parse()?);
            continue;
        };
        let start: usize = start.parse()?;
        let end: usize = end.parse()?;
        if start > end {
            anyhow::bail!("invalid descending CPU range {part}");
        }
        cpus.extend(start..=end);
    }
    cpus.sort_unstable();
    cpus.dedup();
    if cpus.is_empty() {
        anyhow::bail!("CPU list must not be empty");
    }
    Ok(cpus)
}

pub fn validate_cpu_partition(
    issuer_cpus: &[usize],
    query_issuer_cpu: Option<usize>,
    backend_cpus: &[usize],
) -> anyhow::Result<()> {
    if backend_cpus.is_empty() {
        if !issuer_cpus.is_empty() || query_issuer_cpu.is_some() {
            anyhow::bail!("--backend-cpus is required when generator CPUs are specified");
        }
        return Ok(());
    }
    if issuer_cpus.iter().any(|cpu| backend_cpus.contains(cpu)) {
        anyhow::bail!("an issuer CPU overlaps the backend CPU set");
    }
    if query_issuer_cpu.is_some_and(|cpu| backend_cpus.contains(&cpu)) {
        anyhow::bail!("query-issuer CPU overlaps the backend CPU set");
    }
    if query_issuer_cpu.is_some_and(|cpu| issuer_cpus.contains(&cpu)) {
        anyhow::bail!("query and event issuer CPUs must be distinct");
    }
    let mut unique_issuers = issuer_cpus.to_vec();
    unique_issuers.sort_unstable();
    unique_issuers.dedup();
    if unique_issuers.len() != issuer_cpus.len() {
        anyhow::bail!("issuer CPUs must be unique");
    }

    #[cfg(target_os = "linux")]
    {
        let allowed = allowed_cpu_set()?;
        let mut selected_cores = BTreeMap::new();
        for cpu in issuer_cpus
            .iter()
            .copied()
            .chain(query_issuer_cpu)
            .chain(backend_cpus.iter().copied())
        {
            if !allowed.contains(&cpu) {
                anyhow::bail!("CPU {cpu} is outside the process affinity mask");
            }
            if let Some(core) = physical_core(cpu)?
                && let Some(existing) = selected_cores.insert(core, cpu)
            {
                anyhow::bail!(
                    "CPUs {existing} and {cpu} are SMT siblings; select one logical CPU per physical core"
                );
            }
        }
    }
    Ok(())
}

pub fn pin_current_thread_to_cpus(cpus: &[usize]) -> anyhow::Result<()> {
    if cpus.is_empty() {
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        unsafe {
            libc::CPU_ZERO(&mut set);
            for &cpu in cpus {
                libc::CPU_SET(cpu, &mut set);
            }
        }
        let rc = unsafe {
            libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set as *const libc::cpu_set_t,
            )
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn allowed_cpu_set() -> anyhow::Result<Vec<usize>> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let rc = unsafe {
        libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut set as *mut libc::cpu_set_t,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
        .collect())
}

#[cfg(target_os = "linux")]
fn physical_core(cpu: usize) -> anyhow::Result<Option<(u32, u32)>> {
    let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
    let package_path = format!("{topology}/physical_package_id");
    let core_path = format!("{topology}/core_id");
    if !std::path::Path::new(&package_path).exists() || !std::path::Path::new(&core_path).exists() {
        return Ok(None);
    }
    let package = std::fs::read_to_string(package_path)?.trim().parse()?;
    let core = std::fs::read_to_string(core_path)?.trim().parse()?;
    Ok(Some((package, core)))
}

impl From<ObservationError> for IssuerFailure {
    fn from(_: ObservationError) -> Self {
        Self::Observation
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn partition_assigns_contiguous_worker_groups_to_write_issuers() -> anyhow::Result<()> {
        let mut dispatch = (0..128u32)
            .map(|id| DispatchEntry {
                id,
                deadline_ns: u64::from(id),
                deadline_group: id,
                worker_id: u64::from(id),
                payload: DispatchPayload::Query { lane: id as usize },
            })
            .collect::<Vec<_>>();
        dispatch.extend((0..128u32).map(|id| DispatchEntry {
            id: id + 128,
            deadline_ns: u64::from(id),
            deadline_group: id,
            worker_id: u64::from(id),
            payload: DispatchPayload::Event(RouterEvent::new(
                u64::from(id),
                dynamo_kv_router::protocols::KvCacheEvent {
                    event_id: u64::from(id),
                    data: KvCacheEventData::Cleared,
                    dp_rank: 0,
                },
            )),
        }));

        let (queries, event_shards) = partition_dispatch(dispatch, 8)?;

        assert_eq!(queries.len(), 128);
        assert_eq!(event_shards.len(), 8);
        assert!(event_shards.iter().all(|shard| !shard.is_empty()));
        assert_eq!(event_shards.iter().map(Vec::len).sum::<usize>(), 128);
        for (issuer, shard) in event_shards.iter().enumerate() {
            let expected = issuer * 16..(issuer + 1) * 16;
            assert!(
                shard
                    .iter()
                    .all(|entry| expected.contains(&(entry.worker_id as usize)))
            );
        }
        Ok(())
    }
}

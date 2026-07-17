// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
#[cfg(not(target_os = "linux"))]
use std::time::Duration;
use std::time::Instant;

use dynamo_bench::kv_router_common::replay::NoopSequencePublisher;
use dynamo_bench::kv_router_common::trace_gen::WorkerTimelines;
use dynamo_kv_router::protocols::{PrefillLoadHint, WorkerWithDpRank};
use dynamo_kv_router::{
    ActiveSequenceStride, ActiveSequencesMultiWorker, SequenceError, SequenceRequest,
    WorkerLoadProjection,
};
use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::Serialize;
use tokio::sync::{Notify, oneshot};

use super::active_sequences_shared::{SequenceTrace, SequenceTraceEntry};

const RESULT_SCHEMA_VERSION: u32 = 1;
const EMPTY_OPERATION_ID: u32 = u32::MAX;
const START_LEAD_NS: u64 = 20_000_000;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveSequencesRunConfig {
    pub(crate) operation_lanes: usize,
    pub(crate) spin_us: u64,
    pub(crate) issue_lag_diagnostic_threshold_us: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActiveOperationKind {
    ProjectAndAdd,
    PrefillComplete,
    Free,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ActiveSequencesTotals {
    pub(crate) adds: usize,
    pub(crate) prefill_completes: usize,
    pub(crate) frees: usize,
    pub(crate) input_blocks: usize,
    pub(crate) projection_block_visits: usize,
    pub(crate) add_registration_block_visits: usize,
    pub(crate) free_release_block_visits: usize,
}

impl ActiveSequencesTotals {
    fn logical_operations(self) -> usize {
        self.adds + self.prefill_completes + self.frees
    }

    fn logical_block_visits(self) -> usize {
        self.projection_block_visits
            + self.add_registration_block_visits
            + self.free_release_block_visits
    }
}

#[derive(Debug)]
struct PreparedRequest {
    request_id: String,
    hash_start: u32,
    hash_len: u32,
    isl: usize,
    output_length: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveLogicalOperation {
    pub(crate) id: u32,
    pub(crate) deadline_ns: u64,
    pub(crate) worker_id: u64,
    pub(crate) source_ordinal: usize,
    pub(crate) kind: ActiveOperationKind,
    request_index: u32,
}

#[derive(Debug)]
pub(crate) struct PreparedActiveSequencesCorpus {
    pub(crate) operations: Vec<ActiveLogicalOperation>,
    hashes: Box<[SequenceHash]>,
    requests: Box<[PreparedRequest]>,
    expected_operations_by_worker: Vec<(u64, usize)>,
    totals: ActiveSequencesTotals,
    benchmark_duration_ns: u64,
    block_size: u32,
    total_workers: usize,
}

impl PreparedActiveSequencesCorpus {
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn operation_hashes(&self, operation_id: u32) -> anyhow::Result<&[SequenceHash]> {
        let operation = self
            .operations
            .get(operation_id as usize)
            .filter(|operation| operation.kind == ActiveOperationKind::ProjectAndAdd)
            .ok_or_else(|| anyhow::anyhow!("operation {operation_id} is not a prepared add"))?;
        let request = &self.requests[operation.request_index as usize];
        let start = request.hash_start as usize;
        let end = start
            .checked_add(request.hash_len as usize)
            .ok_or_else(|| anyhow::anyhow!("operation {operation_id} hash range overflow"))?;
        self.hashes.get(start..end).ok_or_else(|| {
            anyhow::anyhow!(
                "operation {operation_id} hash range {start}..{end} exceeds the flattened slab"
            )
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn totals(&self) -> ActiveSequencesTotals {
        self.totals
    }
}

#[derive(Clone, Copy)]
struct LifecycleState {
    request_index: u32,
    prefill_completed: bool,
    freed: bool,
}

pub(crate) fn prepare_active_sequences_corpus(
    traces: Vec<Vec<SequenceTrace>>,
    block_size: u32,
    benchmark_duration_ms: u64,
    inference_worker_duplication_factor: usize,
) -> anyhow::Result<PreparedActiveSequencesCorpus> {
    if block_size == 0 {
        anyhow::bail!("Active Sequences block size must be positive");
    }
    if inference_worker_duplication_factor == 0 {
        anyhow::bail!("Active Sequences worker duplication factor must be positive");
    }
    if benchmark_duration_ms == 0 {
        anyhow::bail!("Active Sequences replay window must be positive");
    }

    for (worker_id, worker_trace) in traces.iter().enumerate() {
        for (source_ordinal, pair) in worker_trace.windows(2).enumerate() {
            if pair[1].timestamp_us < pair[0].timestamp_us {
                anyhow::bail!(
                    "worker {worker_id} Active Sequences timestamps are not ordered at source indices {} and {}: prev={}, curr={}",
                    source_ordinal,
                    source_ordinal + 1,
                    pair[0].timestamp_us,
                    pair[1].timestamp_us,
                );
            }
        }
    }

    let scaled = WorkerTimelines::new(traces).into_rescaled_from_first(
        benchmark_duration_ms,
        |entry| entry.timestamp_us,
        |entry, timestamp_us| SequenceTrace {
            entry: entry.entry,
            timestamp_us,
        },
    );
    let num_trace_workers = scaled.len();
    if num_trace_workers == 0 {
        anyhow::bail!("Active Sequences corpus has no worker timelines");
    }

    let total_workers = num_trace_workers
        .checked_mul(inference_worker_duplication_factor)
        .ok_or_else(|| anyhow::anyhow!("Active Sequences worker count overflow"))?;
    let estimated_operations = scaled
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .checked_mul(inference_worker_duplication_factor)
        .ok_or_else(|| anyhow::anyhow!("Active Sequences operation count overflow"))?;
    if estimated_operations >= EMPTY_OPERATION_ID as usize {
        anyhow::bail!("Active Sequences corpus exceeds the u32 operation-ID space");
    }
    if estimated_operations == 0 {
        anyhow::bail!("Active Sequences corpus has no lifecycle operations");
    }

    let mut operations = Vec::with_capacity(estimated_operations);
    let mut requests = Vec::new();
    let mut hashes = Vec::new();
    let mut expected_operations_by_worker = Vec::with_capacity(total_workers);
    let mut totals = ActiveSequencesTotals::default();

    for replica in 0..inference_worker_duplication_factor {
        for (trace_idx, worker_trace) in scaled.iter().enumerate() {
            let worker_id = (replica * num_trace_workers + trace_idx) as u64;
            let mut lifecycle = HashMap::<String, LifecycleState>::new();
            expected_operations_by_worker.push((worker_id, worker_trace.len()));

            for (source_ordinal, trace_entry) in worker_trace.iter().enumerate() {
                let deadline_ns = trace_entry.timestamp_us.saturating_mul(1_000);
                let (kind, request_index) = match &trace_entry.entry {
                    SequenceTraceEntry::Add {
                        request_id,
                        block_hashes,
                        isl,
                        output_length,
                    } => {
                        if lifecycle.contains_key(request_id) {
                            anyhow::bail!(
                                "worker {worker_id} request {request_id} has more than one Add"
                            );
                        }
                        let hash_start = u32::try_from(hashes.len())?;
                        let hash_len = u32::try_from(block_hashes.len())?;
                        hashes.extend_from_slice(block_hashes);
                        let request_index = u32::try_from(requests.len())?;
                        requests.push(PreparedRequest {
                            request_id: format!("{worker_id}:{request_id}"),
                            hash_start,
                            hash_len,
                            isl: *isl,
                            output_length: u32::try_from(*output_length).map_err(|_| {
                                anyhow::anyhow!(
                                    "worker {worker_id} request {request_id} output length exceeds u32"
                                )
                            })?,
                        });
                        lifecycle.insert(
                            request_id.clone(),
                            LifecycleState {
                                request_index,
                                prefill_completed: false,
                                freed: false,
                            },
                        );
                        totals.adds += 1;
                        totals.input_blocks += block_hashes.len();
                        totals.projection_block_visits += block_hashes.len();
                        totals.add_registration_block_visits += block_hashes.len();
                        (ActiveOperationKind::ProjectAndAdd, request_index)
                    }
                    SequenceTraceEntry::PrefillComplete { request_id } => {
                        let state = lifecycle.get_mut(request_id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "worker {worker_id} request {request_id} completes prefill before Add"
                            )
                        })?;
                        if state.freed {
                            anyhow::bail!(
                                "worker {worker_id} request {request_id} completes prefill after Free"
                            );
                        }
                        if state.prefill_completed {
                            anyhow::bail!(
                                "worker {worker_id} request {request_id} completes prefill more than once"
                            );
                        }
                        state.prefill_completed = true;
                        totals.prefill_completes += 1;
                        (ActiveOperationKind::PrefillComplete, state.request_index)
                    }
                    SequenceTraceEntry::Free { request_id } => {
                        let state = lifecycle.get_mut(request_id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "worker {worker_id} request {request_id} is freed before Add"
                            )
                        })?;
                        if state.freed {
                            anyhow::bail!(
                                "worker {worker_id} request {request_id} is freed more than once"
                            );
                        }
                        state.freed = true;
                        let request = &requests[state.request_index as usize];
                        totals.frees += 1;
                        totals.free_release_block_visits += request.hash_len as usize;
                        (ActiveOperationKind::Free, state.request_index)
                    }
                };
                operations.push(ActiveLogicalOperation {
                    id: 0,
                    deadline_ns,
                    worker_id,
                    source_ordinal,
                    kind,
                    request_index,
                });
            }

            let unfinished = lifecycle
                .iter()
                .filter_map(|(request_id, state)| (!state.freed).then_some(request_id.as_str()))
                .collect::<Vec<_>>();
            if !unfinished.is_empty() {
                anyhow::bail!(
                    "worker {worker_id} has {} requests without a final Free: {}",
                    unfinished.len(),
                    unfinished.join(", ")
                );
            }
        }
    }

    operations.sort_by_key(|operation| {
        (
            operation.deadline_ns,
            operation.worker_id,
            operation.source_ordinal,
        )
    });
    for (id, operation) in operations.iter_mut().enumerate() {
        operation.id = id as u32;
    }

    if operations.len() != totals.logical_operations() {
        anyhow::bail!("Active Sequences logical-operation accounting mismatch");
    }

    Ok(PreparedActiveSequencesCorpus {
        operations,
        hashes: hashes.into_boxed_slice(),
        requests: requests.into_boxed_slice(),
        expected_operations_by_worker,
        totals,
        benchmark_duration_ns: benchmark_duration_ms.saturating_mul(1_000_000),
        block_size,
        total_workers,
    })
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
    ProjectAndAdd(SequenceRequest),
    PrefillComplete(u32),
    Free(u32),
}

pub(crate) struct PreparedActiveSequencesTrial {
    dispatch: Vec<ScheduledOperation>,
    lane_payloads: Vec<Box<[LanePayloadSlot]>>,
    lane_request_ids: Vec<Box<[String]>>,
    lane_capacities: Vec<usize>,
    operation_workers: Box<[u64]>,
    operation_kinds: Box<[ActiveOperationKind]>,
    expected_operations_by_worker: Vec<(u64, usize)>,
    totals: ActiveSequencesTotals,
    benchmark_duration_ns: u64,
    block_size: u32,
    total_workers: usize,
}

impl PreparedActiveSequencesTrial {
    fn page_touch_untimed(&self) {
        let mut checksum = self.benchmark_duration_ns ^ u64::from(self.block_size);
        for operation in &self.dispatch {
            checksum ^= operation.deadline_ns ^ u64::from(operation.id) ^ operation.lane as u64;
        }
        for lane in &self.lane_payloads {
            for slot in lane.iter() {
                checksum ^= u64::from(slot.id);
                match slot.payload.as_ref() {
                    Some(LanePayload::ProjectAndAdd(request)) => {
                        for byte in request.request_id.as_bytes() {
                            checksum ^= u64::from(*byte);
                        }
                        if let Some(hashes) = request.token_sequence.as_ref() {
                            for hash in hashes.as_slice() {
                                checksum ^= *hash;
                            }
                        }
                    }
                    Some(LanePayload::PrefillComplete(request_index))
                    | Some(LanePayload::Free(request_index)) => {
                        checksum ^= u64::from(*request_index)
                    }
                    None => checksum ^= u64::MAX,
                }
            }
        }
        for request_ids in &self.lane_request_ids {
            for request_id in request_ids.iter() {
                for byte in request_id.as_bytes() {
                    checksum ^= u64::from(*byte);
                }
            }
        }
        for (&worker_id, &kind) in self.operation_workers.iter().zip(&self.operation_kinds) {
            checksum ^= worker_id ^ kind as u64;
        }
        for &(worker_id, count) in &self.expected_operations_by_worker {
            checksum ^= worker_id ^ count as u64;
        }
        checksum ^= self.totals.logical_operations() as u64
            ^ self.totals.logical_block_visits() as u64
            ^ self.total_workers as u64;
        black_box(checksum);
    }
}

pub(crate) fn prepare_active_sequences_trial(
    corpus: PreparedActiveSequencesCorpus,
    operation_lanes: usize,
) -> anyhow::Result<PreparedActiveSequencesTrial> {
    if operation_lanes == 0 {
        anyhow::bail!("Active Sequences operation-lane count must be positive");
    }
    if operation_lanes > u16::MAX as usize {
        anyhow::bail!("Active Sequences operation-lane count exceeds u16");
    }

    let PreparedActiveSequencesCorpus {
        operations,
        hashes,
        requests,
        expected_operations_by_worker,
        totals,
        benchmark_duration_ns,
        block_size,
        total_workers,
    } = corpus;
    let mut lane_capacities = vec![0usize; operation_lanes];
    for operation in &operations {
        lane_capacities[operation.worker_id as usize % operation_lanes] += 1;
    }
    let mut lane_payloads = lane_capacities
        .iter()
        .map(|&capacity| Vec::with_capacity(capacity))
        .collect::<Vec<_>>();
    let mut lane_request_ids = (0..operation_lanes)
        .map(|_| Vec::<String>::new())
        .collect::<Vec<_>>();
    let mut request_lane_indices = vec![None; requests.len()];
    for operation in operations
        .iter()
        .filter(|operation| operation.kind == ActiveOperationKind::ProjectAndAdd)
    {
        let lane = operation.worker_id as usize % operation_lanes;
        let request = &requests[operation.request_index as usize];
        let lane_request_index = u32::try_from(lane_request_ids[lane].len())?;
        lane_request_ids[lane].push(request.request_id.clone());
        request_lane_indices[operation.request_index as usize] = Some((lane, lane_request_index));
    }
    let mut dispatch = Vec::with_capacity(operations.len());
    let mut operation_workers = Vec::with_capacity(operations.len());
    let mut operation_kinds = Vec::with_capacity(operations.len());

    for operation in operations {
        let request = requests
            .get(operation.request_index as usize)
            .ok_or_else(|| anyhow::anyhow!("operation {} has an invalid request", operation.id))?;
        let lane = operation.worker_id as usize % operation_lanes;
        let (request_lane, lane_request_index) = request_lane_indices
            .get(operation.request_index as usize)
            .and_then(|entry| *entry)
            .ok_or_else(|| {
                anyhow::anyhow!("operation {} has no prepared request ID", operation.id)
            })?;
        if request_lane != lane {
            anyhow::bail!(
                "operation {} request is assigned to another lane",
                operation.id
            );
        }
        let payload = match operation.kind {
            ActiveOperationKind::ProjectAndAdd => {
                let start = request.hash_start as usize;
                let end = start
                    .checked_add(request.hash_len as usize)
                    .ok_or_else(|| {
                        anyhow::anyhow!("operation {} hash range overflow", operation.id)
                    })?;
                let token_sequence = hashes
                    .get(start..end)
                    .ok_or_else(|| {
                        anyhow::anyhow!("operation {} hash range exceeds the slab", operation.id)
                    })?
                    .to_vec();
                LanePayload::ProjectAndAdd(SequenceRequest {
                    request_id: request.request_id.clone(),
                    token_sequence: Some(ActiveSequenceStride::ONE.sample_dense(token_sequence)),
                    track_prefill_tokens: true,
                    expected_output_tokens: Some(request.output_length),
                    prefill_load_hint: Some(PrefillLoadHint {
                        initial_effective_prefill_tokens: request.isl,
                        expected_prefill_duration: None,
                    }),
                    worker: WorkerWithDpRank::from_worker_id(operation.worker_id),
                    lora_name: None,
                })
            }
            ActiveOperationKind::PrefillComplete => {
                LanePayload::PrefillComplete(lane_request_index)
            }
            ActiveOperationKind::Free => LanePayload::Free(lane_request_index),
        };
        dispatch.push(ScheduledOperation {
            id: operation.id,
            deadline_ns: operation.deadline_ns,
            lane,
        });
        lane_payloads[lane].push(LanePayloadSlot {
            id: operation.id,
            payload: Some(payload),
        });
        operation_workers.push(operation.worker_id);
        operation_kinds.push(operation.kind);
    }

    if lane_payloads
        .iter()
        .zip(&lane_capacities)
        .any(|(payloads, &capacity)| payloads.len() != capacity)
    {
        anyhow::bail!("Active Sequences lane-capacity accounting mismatch");
    }

    Ok(PreparedActiveSequencesTrial {
        dispatch,
        lane_payloads: lane_payloads
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect(),
        lane_request_ids: lane_request_ids
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect(),
        lane_capacities,
        operation_workers: operation_workers.into_boxed_slice(),
        operation_kinds: operation_kinds.into_boxed_slice(),
        expected_operations_by_worker,
        totals,
        benchmark_duration_ns,
        block_size,
        total_workers,
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

    fn write(&self, cursor: usize, id: u32) -> Result<(), IssuerFailure> {
        let Some(slot) = self.slots.get(cursor) else {
            return Err(IssuerFailure::LaneOverflow);
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

    fn depth(&self) -> usize {
        self.published
            .load(Ordering::Acquire)
            .saturating_sub(self.consumed.load(Ordering::Acquire))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum CompletionFailure {
    #[default]
    None,
    MissingPayload,
    WorkerNotFound,
    DuplicateRequest,
    RequestNotFound,
    ReplicaSyncPublishFailed,
    InvalidRequestIndex,
}

impl CompletionFailure {
    fn from_sequence_error(error: SequenceError) -> Self {
        match error {
            SequenceError::WorkerNotFound { .. } => Self::WorkerNotFound,
            SequenceError::DuplicateRequest { .. } => Self::DuplicateRequest,
            SequenceError::RequestNotFound { .. } => Self::RequestNotFound,
            SequenceError::ReplicaSyncPublishFailed(_) => Self::ReplicaSyncPublishFailed,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CompletionRecord {
    id: u32,
    started_ns: u64,
    project_finished_ns: u64,
    finished_ns: u64,
    projection_count: u64,
    projection_inspected: u64,
    projection_digest: u64,
    failure: CompletionFailure,
}

pub(crate) fn summarize_worker_projections(
    projections: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
) -> (u64, u64) {
    let mut inspected = 0u64;
    let mut digest = 0u64;
    for (worker, projection) in projections {
        let entry = mix64(worker.worker_id)
            ^ mix64(u64::from(worker.dp_rank)).rotate_left(7)
            ^ mix64(projection.active_prefill_tokens as u64).rotate_left(13)
            ^ mix64(projection.active_decode_blocks as u64).rotate_left(29)
            ^ mix64(projection.additional_active_blocks as u64).rotate_left(43);
        digest ^= mix64(entry);
        inspected += 1;
    }
    (inspected, digest)
}

pub(crate) fn accumulate_projection_digest(
    aggregate: u64,
    operation_id: u32,
    projection_digest: u64,
) -> u64 {
    aggregate ^ mix64(projection_digest ^ u64::from(operation_id))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LaneFailure {
    #[default]
    None,
    MissingPublishedId,
    PayloadOrder,
    CompletionOverflow,
}

struct LaneResult {
    completions: Box<[CompletionRecord]>,
    written: usize,
    drain_ns: u64,
    failure: LaneFailure,
}

fn execute_payload(
    sequences: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    clock: &BenchmarkClock,
    id: u32,
    request_ids: &[String],
    payload: Option<LanePayload>,
) -> CompletionRecord {
    let started_ns = clock.now_ns();
    let Some(payload) = payload else {
        return CompletionRecord {
            id,
            started_ns,
            finished_ns: clock.now_ns(),
            failure: CompletionFailure::MissingPayload,
            ..CompletionRecord::default()
        };
    };
    let decay_now = tokio::time::Instant::now();

    match payload {
        LanePayload::ProjectAndAdd(request) => {
            let projections =
                sequences.project_worker_loads(request.token_sequence.as_ref(), decay_now);
            let projection_count = projections.len() as u64;
            let (projection_inspected, projection_digest) =
                summarize_worker_projections(&projections);
            drop(black_box(projections));
            let project_finished_ns = clock.now_ns();
            let failure = sequences
                .add_request(request, decay_now)
                .err()
                .map(CompletionFailure::from_sequence_error)
                .unwrap_or_default();
            CompletionRecord {
                id,
                started_ns,
                project_finished_ns,
                finished_ns: clock.now_ns(),
                projection_count,
                projection_inspected,
                projection_digest,
                failure,
            }
        }
        LanePayload::PrefillComplete(request_index) => {
            let Some(request_id) = request_ids.get(request_index as usize) else {
                return CompletionRecord {
                    id,
                    started_ns,
                    finished_ns: clock.now_ns(),
                    failure: CompletionFailure::InvalidRequestIndex,
                    ..CompletionRecord::default()
                };
            };
            let failure = sequences
                .mark_prefill_completed(request_id, decay_now)
                .err()
                .map(CompletionFailure::from_sequence_error)
                .unwrap_or_default();
            CompletionRecord {
                id,
                started_ns,
                finished_ns: clock.now_ns(),
                failure,
                ..CompletionRecord::default()
            }
        }
        LanePayload::Free(request_index) => {
            let Some(request_id) = request_ids.get(request_index as usize) else {
                return CompletionRecord {
                    id,
                    started_ns,
                    finished_ns: clock.now_ns(),
                    failure: CompletionFailure::InvalidRequestIndex,
                    ..CompletionRecord::default()
                };
            };
            let failure = sequences
                .free(request_id, decay_now)
                .err()
                .map(CompletionFailure::from_sequence_error)
                .unwrap_or_default();
            CompletionRecord {
                id,
                started_ns,
                finished_ns: clock.now_ns(),
                failure,
                ..CompletionRecord::default()
            }
        }
    }
}

async fn operation_lane_worker(
    sequences: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    lane: Arc<OperationLane>,
    mut payloads: Box<[LanePayloadSlot]>,
    request_ids: Box<[String]>,
    clock: Arc<BenchmarkClock>,
    ready: oneshot::Sender<()>,
) -> LaneResult {
    let mut completions = vec![CompletionRecord::default(); payloads.len()].into_boxed_slice();
    let mut consumed = 0usize;
    let mut failure = LaneFailure::None;
    let _ = ready.send(());

    loop {
        let published = lane.published.load(Ordering::Acquire);
        while consumed < published {
            let id = lane.slots[consumed].load(Ordering::Relaxed);
            if id == EMPTY_OPERATION_ID {
                failure = LaneFailure::MissingPublishedId;
                break;
            }
            let Some(payload_slot) = payloads.get_mut(consumed) else {
                failure = LaneFailure::PayloadOrder;
                break;
            };
            if payload_slot.id != id {
                failure = LaneFailure::PayloadOrder;
                break;
            }
            let Some(completion_slot) = completions.get_mut(consumed) else {
                failure = LaneFailure::CompletionOverflow;
                break;
            };
            *completion_slot = execute_payload(
                sequences.as_ref(),
                clock.as_ref(),
                id,
                &request_ids,
                payload_slot.payload.take(),
            );
            consumed += 1;
            lane.consumed.store(consumed, Ordering::Release);
        }

        if failure != LaneFailure::None {
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

    LaneResult {
        completions,
        written: consumed,
        drain_ns: clock.now_ns(),
        failure,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct IssueRecord {
    scheduled_ns: u64,
    accepted_ns: u64,
    lane: u16,
    accepted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IssuerFailure {
    LaneOverflow,
    DuplicateOperation,
    InvalidLane,
    DispatchMismatch,
}

struct IssuerOutput {
    records: Box<[IssueRecord]>,
    producer_stop_ns: u64,
    failure: Option<IssuerFailure>,
}

fn issue_operations(
    dispatch: Vec<ScheduledOperation>,
    lanes: Vec<Arc<OperationLane>>,
    operation_count: usize,
    clock: Arc<BenchmarkClock>,
    start_signal: Arc<AtomicU64>,
    ready: mpsc::SyncSender<()>,
) -> IssuerOutput {
    let mut records = vec![IssueRecord::default(); operation_count].into_boxed_slice();
    let mut lane_cursors = vec![0usize; lanes.len()].into_boxed_slice();
    let mut touched_flags = vec![false; lanes.len()].into_boxed_slice();
    let mut touched_lanes = Vec::with_capacity(lanes.len());
    let mut failure = None;
    if ready.send(()).is_err() {
        return IssuerOutput {
            records,
            producer_stop_ns: clock.now_ns(),
            failure: Some(IssuerFailure::DispatchMismatch),
        };
    }
    while start_signal.load(Ordering::Acquire) == 0 {
        std::hint::spin_loop();
    }
    let start_ns = start_signal.load(Ordering::Acquire);
    let mut operations = dispatch.into_iter().peekable();

    while failure.is_none() {
        let Some(first) = operations.peek() else {
            break;
        };
        let deadline_ns = first.deadline_ns;
        clock.wait_until(start_ns.saturating_add(deadline_ns));
        touched_lanes.clear();

        while operations
            .peek()
            .is_some_and(|operation| operation.deadline_ns == deadline_ns)
        {
            let Some(operation) = operations.next() else {
                failure = Some(IssuerFailure::DispatchMismatch);
                break;
            };
            let Some(lane) = lanes.get(operation.lane) else {
                failure = Some(IssuerFailure::InvalidLane);
                break;
            };
            if let Err(error) = lane.write(lane_cursors[operation.lane], operation.id) {
                failure = Some(error);
                break;
            }
            lane_cursors[operation.lane] += 1;
            if !touched_flags[operation.lane] {
                touched_flags[operation.lane] = true;
                touched_lanes.push(operation.lane);
            }
            let Some(record) = records.get_mut(operation.id as usize) else {
                failure = Some(IssuerFailure::DuplicateOperation);
                break;
            };
            if record.accepted {
                failure = Some(IssuerFailure::DuplicateOperation);
                break;
            }
            *record = IssueRecord {
                scheduled_ns: start_ns.saturating_add(deadline_ns),
                accepted_ns: clock.now_ns(),
                lane: operation.lane as u16,
                accepted: true,
            };
        }

        for &lane_id in &touched_lanes {
            lanes[lane_id].publish(lane_cursors[lane_id]);
            touched_flags[lane_id] = false;
        }
    }

    IssuerOutput {
        records,
        producer_stop_ns: clock.now_ns(),
        failure,
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Distribution {
    pub(crate) p50_ns: u64,
    pub(crate) p99_ns: u64,
    pub(crate) p999_ns: u64,
    pub(crate) max_ns: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct ActiveSequencesResult {
    pub(crate) schema_version: u32,
    pub(crate) timer: &'static str,
    pub(crate) benchmark_duration_ms: u64,
    pub(crate) block_size: u32,
    pub(crate) operation_lanes: usize,
    pub(crate) total_workers: usize,
    pub(crate) total_adds: usize,
    pub(crate) total_prefill_completes: usize,
    pub(crate) total_frees: usize,
    pub(crate) total_logical_operations: usize,
    pub(crate) total_input_blocks: usize,
    pub(crate) projection_block_visits: usize,
    pub(crate) add_registration_block_visits: usize,
    pub(crate) free_release_block_visits: usize,
    pub(crate) total_logical_block_visits: usize,
    pub(crate) worker_projections_produced: u64,
    pub(crate) worker_projections_inspected: u64,
    pub(crate) worker_projection_digest: u64,
    pub(crate) offered_logical_ops_per_sec: f64,
    pub(crate) actual_issue_logical_ops_per_sec: f64,
    pub(crate) achieved_logical_ops_per_sec: f64,
    pub(crate) offered_logical_block_visits_per_sec: f64,
    pub(crate) actual_issue_logical_block_visits_per_sec: f64,
    pub(crate) achieved_logical_block_visits_per_sec: f64,
    pub(crate) issue_lag: Distribution,
    pub(crate) queue_wait: Distribution,
    pub(crate) scheduled_to_completed: Distribution,
    pub(crate) project_service: Distribution,
    pub(crate) add_service: Distribution,
    pub(crate) project_and_add_service: Distribution,
    pub(crate) prefill_complete_service: Distribution,
    pub(crate) free_service: Distribution,
    pub(crate) delayed_operations: usize,
    pub(crate) queue_depth_at_stop: Vec<usize>,
    pub(crate) outstanding_at_stop: usize,
    pub(crate) maximum_queue_depth: usize,
    pub(crate) issue_span_ns: u64,
    pub(crate) drain_ns: u64,
    pub(crate) issued_operations: usize,
    pub(crate) completed_operations: usize,
    pub(crate) successful_operations: usize,
    pub(crate) exact_ids: bool,
    pub(crate) per_worker_fifo: bool,
    pub(crate) zero_fixed_buffer_overflow: bool,
    pub(crate) issue_span_valid: bool,
    pub(crate) final_state_empty: bool,
    pub(crate) generator_valid: bool,
    pub(crate) kept_up: bool,
    pub(crate) failure_reasons: Vec<String>,
}

pub(crate) async fn run_active_sequences_benchmark(
    corpus: PreparedActiveSequencesCorpus,
    config: ActiveSequencesRunConfig,
) -> anyhow::Result<ActiveSequencesResult> {
    let trial = prepare_active_sequences_trial(corpus, config.operation_lanes)?;
    run_active_sequences_trial(trial, config).await
}

async fn run_active_sequences_trial(
    trial: PreparedActiveSequencesTrial,
    config: ActiveSequencesRunConfig,
) -> anyhow::Result<ActiveSequencesResult> {
    if config.operation_lanes != trial.lane_payloads.len() {
        anyhow::bail!("Active Sequences run configuration does not match the prepared schedule");
    }
    trial.page_touch_untimed();

    let dp_range: HashMap<u64, (u32, u32)> = (0..trial.total_workers as u64)
        .map(|worker_id| (worker_id, (0, 1)))
        .collect();
    let sequences = Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
        NoopSequencePublisher,
        trial.block_size as usize,
        dp_range,
        false,
        0,
        "bench",
    ));
    let clock = Arc::new(BenchmarkClock::new(config.spin_us.saturating_mul(1_000))?);
    let lanes = trial
        .lane_capacities
        .iter()
        .map(|&capacity| Arc::new(OperationLane::new(capacity)))
        .collect::<Vec<_>>();

    let mut ready_receivers = Vec::with_capacity(lanes.len());
    let mut lane_tasks = Vec::with_capacity(lanes.len());
    for ((lane, payloads), request_ids) in lanes
        .iter()
        .zip(trial.lane_payloads)
        .zip(trial.lane_request_ids)
    {
        let (ready_tx, ready_rx) = oneshot::channel();
        ready_receivers.push(ready_rx);
        lane_tasks.push(tokio::spawn(operation_lane_worker(
            Arc::clone(&sequences),
            Arc::clone(lane),
            payloads,
            request_ids,
            Arc::clone(&clock),
            ready_tx,
        )));
    }
    for receiver in ready_receivers {
        receiver.await?;
    }

    let operation_count = trial.operation_workers.len();
    let start_signal = Arc::new(AtomicU64::new(0));
    let (issuer_ready_tx, issuer_ready_rx) = mpsc::sync_channel(0);
    let issuer_handle = std::thread::spawn({
        let issuer_lanes = lanes.iter().map(Arc::clone).collect::<Vec<_>>();
        let clock = Arc::clone(&clock);
        let start_signal = Arc::clone(&start_signal);
        move || {
            issue_operations(
                trial.dispatch,
                issuer_lanes,
                operation_count,
                clock,
                start_signal,
                issuer_ready_tx,
            )
        }
    });

    if issuer_ready_rx.recv().is_err() {
        for lane in &lanes {
            lane.close();
        }
        for task in lane_tasks {
            let _ = task.await;
        }
        let _ = issuer_handle.join();
        anyhow::bail!("Active Sequences issuer exited before becoming ready");
    }
    let start_ns = clock.now_ns().saturating_add(START_LEAD_NS);
    start_signal.store(start_ns, Ordering::Release);
    let issuer = match issuer_handle.join() {
        Ok(issuer) => issuer,
        Err(_) => {
            for lane in &lanes {
                lane.close();
            }
            for task in lane_tasks {
                let _ = task.await;
            }
            anyhow::bail!("Active Sequences issuer panicked");
        }
    };
    let queue_depth_at_stop = lanes.iter().map(|lane| lane.depth()).collect::<Vec<_>>();
    for lane in &lanes {
        lane.close();
    }

    let mut lane_results = Vec::with_capacity(lane_tasks.len());
    for task in lane_tasks {
        lane_results.push(task.await?);
    }
    let end_ns = lane_results
        .iter()
        .map(|result| result.drain_ns)
        .max()
        .unwrap_or(start_ns);

    let obvious_empty = sequences.active_blocks().values().all(|&count| count == 0)
        && sequences
            .active_tokens(tokio::time::Instant::now())
            .values()
            .all(|&count| count == 0)
        && sequences
            .active_request_counts()
            .values()
            .all(|&count| count == 0)
        && sequences.get_active_lora_counts().is_empty();
    let all_lane_methods_succeeded = lane_results.iter().all(|result| {
        result.failure == LaneFailure::None
            && result.completions[..result.written]
                .iter()
                .all(|completion| completion.failure == CompletionFailure::None)
    });
    if all_lane_methods_succeeded && obvious_empty {
        sequences.assert_completely_drained(tokio::time::Instant::now());
    }

    Ok(analyze_result(
        &clock,
        start_ns,
        end_ns,
        &trial.operation_workers,
        &trial.operation_kinds,
        trial.expected_operations_by_worker,
        trial.lane_capacities,
        trial.totals,
        trial.benchmark_duration_ns,
        trial.block_size,
        trial.total_workers,
        config,
        issuer,
        lane_results,
        queue_depth_at_stop,
        obvious_empty,
    ))
}

#[allow(clippy::too_many_arguments)]
fn analyze_result(
    clock: &BenchmarkClock,
    start_ns: u64,
    end_ns: u64,
    operation_workers: &[u64],
    operation_kinds: &[ActiveOperationKind],
    expected_operations_by_worker: Vec<(u64, usize)>,
    lane_capacities: Vec<usize>,
    totals: ActiveSequencesTotals,
    benchmark_duration_ns: u64,
    block_size: u32,
    total_workers: usize,
    config: ActiveSequencesRunConfig,
    issuer: IssuerOutput,
    lane_results: Vec<LaneResult>,
    queue_depth_at_stop: Vec<usize>,
    final_state_empty: bool,
) -> ActiveSequencesResult {
    let mut failure_reasons = Vec::new();
    let mut exact_ids = true;
    let mut per_worker_fifo = true;
    let mut zero_fixed_buffer_overflow = true;
    if let Some(failure) = issuer.failure {
        exact_ids = false;
        push_failure(&mut failure_reasons, format!("issuer_{failure:?}"));
        if failure == IssuerFailure::LaneOverflow {
            zero_fixed_buffer_overflow = false;
        }
    }
    if !final_state_empty {
        push_failure(&mut failure_reasons, "final_state_not_empty".to_string());
    }

    let mut expected_by_lane = vec![Vec::new(); config.operation_lanes];
    let mut expected_by_worker = BTreeMap::<u64, Vec<u32>>::new();
    for (id, record) in issuer.records.iter().enumerate() {
        if !record.accepted {
            continue;
        }
        if let Some(expected) = expected_by_lane.get_mut(record.lane as usize) {
            expected.push(id as u32);
        } else {
            exact_ids = false;
            push_failure(&mut failure_reasons, "issue_lane_out_of_range".to_string());
        }
        if let Some(&worker_id) = operation_workers.get(id) {
            expected_by_worker
                .entry(worker_id)
                .or_default()
                .push(id as u32);
        }
    }

    let mut completions = vec![None; issuer.records.len()];
    let mut actual_by_worker = BTreeMap::<u64, Vec<u32>>::new();
    for (lane_id, result) in lane_results.into_iter().enumerate() {
        if result.failure != LaneFailure::None {
            if result.failure == LaneFailure::CompletionOverflow {
                zero_fixed_buffer_overflow = false;
            }
            push_failure(
                &mut failure_reasons,
                format!("lane_{lane_id}_{:?}", result.failure),
            );
        }
        if result.written != lane_capacities[lane_id] {
            exact_ids = false;
            push_failure(&mut failure_reasons, format!("lane_{lane_id}_incomplete"));
        }
        let actual_ids = result.completions[..result.written]
            .iter()
            .map(|completion| completion.id)
            .collect::<Vec<_>>();
        if expected_by_lane.get(lane_id) != Some(&actual_ids) {
            exact_ids = false;
            push_failure(&mut failure_reasons, format!("lane_{lane_id}_order"));
        }
        for completion in result.completions[..result.written].iter().copied() {
            let id = completion.id as usize;
            let Some(slot) = completions.get_mut(id) else {
                exact_ids = false;
                push_failure(
                    &mut failure_reasons,
                    "completion_id_out_of_range".to_string(),
                );
                continue;
            };
            if slot.replace(completion).is_some() {
                exact_ids = false;
                push_failure(&mut failure_reasons, "duplicate_completion".to_string());
            }
            if let Some(&worker_id) = operation_workers.get(id) {
                actual_by_worker
                    .entry(worker_id)
                    .or_default()
                    .push(completion.id);
            }
        }
    }

    for (worker_id, expected_count) in expected_operations_by_worker {
        let expected = expected_by_worker.get(&worker_id).map_or(0, Vec::len);
        if expected != expected_count {
            exact_ids = false;
            push_failure(
                &mut failure_reasons,
                format!("worker_{worker_id}_issue_count"),
            );
        }
        if actual_by_worker.get(&worker_id) != expected_by_worker.get(&worker_id) {
            per_worker_fifo = false;
            push_failure(&mut failure_reasons, format!("worker_{worker_id}_fifo"));
        }
    }

    let mut issue_lag = Vec::with_capacity(totals.logical_operations());
    let mut queue_wait = Vec::with_capacity(totals.logical_operations());
    let mut scheduled_to_completed = Vec::with_capacity(totals.logical_operations());
    let mut project_service = Vec::with_capacity(totals.adds);
    let mut add_service = Vec::with_capacity(totals.adds);
    let mut project_and_add_service = Vec::with_capacity(totals.adds);
    let mut prefill_complete_service = Vec::with_capacity(totals.prefill_completes);
    let mut free_service = Vec::with_capacity(totals.frees);
    let mut delayed_operations = 0usize;
    let mut worker_projections_produced = 0u64;
    let mut worker_projections_inspected = 0u64;
    let mut worker_projection_digest = 0u64;
    let mut successful_operations = 0usize;
    let mut outstanding_at_stop = 0usize;
    let mut queue_edges = lane_capacities
        .iter()
        .map(|&capacity| Vec::with_capacity(capacity.saturating_mul(2)))
        .collect::<Vec<_>>();
    let diagnostic_threshold_ns = config
        .issue_lag_diagnostic_threshold_us
        .saturating_mul(1_000);

    for id in 0..issuer.records.len() {
        let issue = issuer.records[id];
        if !issue.accepted {
            exact_ids = false;
            push_failure(&mut failure_reasons, "missing_issue".to_string());
            continue;
        }
        let Some(completion) = completions[id] else {
            exact_ids = false;
            push_failure(&mut failure_reasons, "missing_completion".to_string());
            continue;
        };
        if completion.failure != CompletionFailure::None {
            push_failure(
                &mut failure_reasons,
                format!("method_{:?}", completion.failure),
            );
        } else {
            successful_operations += 1;
        }

        let lag = issue.accepted_ns.saturating_sub(issue.scheduled_ns);
        issue_lag.push(lag);
        delayed_operations += usize::from(lag > diagnostic_threshold_ns);
        queue_wait.push(completion.started_ns.saturating_sub(issue.accepted_ns));
        scheduled_to_completed.push(completion.finished_ns.saturating_sub(issue.scheduled_ns));
        let lane = issue.lane as usize;
        if let Some(edges) = queue_edges.get_mut(lane) {
            edges.push((issue.accepted_ns, 1));
            edges.push((completion.started_ns, -1));
        }
        if issue.accepted_ns <= issuer.producer_stop_ns
            && completion.finished_ns > issuer.producer_stop_ns
        {
            outstanding_at_stop += 1;
        }

        match operation_kinds[id] {
            ActiveOperationKind::ProjectAndAdd => {
                if completion.projection_count != completion.projection_inspected {
                    push_failure(
                        &mut failure_reasons,
                        "projection_inspection_incomplete".to_string(),
                    );
                }
                project_service.push(
                    completion
                        .project_finished_ns
                        .saturating_sub(completion.started_ns),
                );
                add_service.push(
                    completion
                        .finished_ns
                        .saturating_sub(completion.project_finished_ns),
                );
                project_and_add_service
                    .push(completion.finished_ns.saturating_sub(completion.started_ns));
                worker_projections_produced =
                    worker_projections_produced.saturating_add(completion.projection_count);
                worker_projections_inspected =
                    worker_projections_inspected.saturating_add(completion.projection_inspected);
                worker_projection_digest = accumulate_projection_digest(
                    worker_projection_digest,
                    id as u32,
                    completion.projection_digest,
                );
            }
            ActiveOperationKind::PrefillComplete => prefill_complete_service
                .push(completion.finished_ns.saturating_sub(completion.started_ns)),
            ActiveOperationKind::Free => {
                free_service.push(completion.finished_ns.saturating_sub(completion.started_ns));
            }
        }
    }

    let maximum_queue_depth = queue_edges
        .iter_mut()
        .map(|edges| maximum_depth(edges))
        .max()
        .unwrap_or(0);
    let issue_span_ns = issuer.producer_stop_ns.saturating_sub(start_ns);
    let drain_ns = end_ns.saturating_sub(issuer.producer_stop_ns);
    let total_duration_ns = end_ns.saturating_sub(start_ns);
    let issue_limit_ns = benchmark_duration_ns.saturating_mul(101) / 100;
    let issue_span_valid = issue_span_ns <= issue_limit_ns;
    if !issue_span_valid {
        push_failure(&mut failure_reasons, "issue_span_exceeded".to_string());
    }
    let issued_operations = issuer
        .records
        .iter()
        .filter(|record| record.accepted)
        .count();
    let completed_operations = completions
        .iter()
        .filter(|completion| completion.is_some())
        .count();
    exact_ids &= issued_operations == totals.logical_operations()
        && completed_operations == totals.logical_operations();
    if !exact_ids {
        push_failure(&mut failure_reasons, "exact_id_validation".to_string());
    }
    if !zero_fixed_buffer_overflow {
        push_failure(&mut failure_reasons, "fixed_buffer_overflow".to_string());
    }
    if successful_operations != totals.logical_operations() {
        push_failure(
            &mut failure_reasons,
            "operation_completion_failure".to_string(),
        );
    }
    let generator_valid = failure_reasons.is_empty();
    let kept_up =
        generator_valid && total_duration_ns <= benchmark_duration_ns.saturating_mul(110) / 100;
    let offered_seconds = (benchmark_duration_ns as f64 / 1e9).max(f64::EPSILON);
    let issue_seconds = (issue_span_ns as f64 / 1e9).max(f64::EPSILON);
    let achieved_seconds = (total_duration_ns as f64 / 1e9).max(f64::EPSILON);

    ActiveSequencesResult {
        schema_version: RESULT_SCHEMA_VERSION,
        timer: clock.timer_name(),
        benchmark_duration_ms: benchmark_duration_ns / 1_000_000,
        block_size,
        operation_lanes: config.operation_lanes,
        total_workers,
        total_adds: totals.adds,
        total_prefill_completes: totals.prefill_completes,
        total_frees: totals.frees,
        total_logical_operations: totals.logical_operations(),
        total_input_blocks: totals.input_blocks,
        projection_block_visits: totals.projection_block_visits,
        add_registration_block_visits: totals.add_registration_block_visits,
        free_release_block_visits: totals.free_release_block_visits,
        total_logical_block_visits: totals.logical_block_visits(),
        worker_projections_produced,
        worker_projections_inspected,
        worker_projection_digest,
        offered_logical_ops_per_sec: totals.logical_operations() as f64 / offered_seconds,
        actual_issue_logical_ops_per_sec: totals.logical_operations() as f64 / issue_seconds,
        achieved_logical_ops_per_sec: totals.logical_operations() as f64 / achieved_seconds,
        offered_logical_block_visits_per_sec: totals.logical_block_visits() as f64
            / offered_seconds,
        actual_issue_logical_block_visits_per_sec: totals.logical_block_visits() as f64
            / issue_seconds,
        achieved_logical_block_visits_per_sec: totals.logical_block_visits() as f64
            / achieved_seconds,
        issue_lag: distribution(issue_lag),
        queue_wait: distribution(queue_wait),
        scheduled_to_completed: distribution(scheduled_to_completed),
        project_service: distribution(project_service),
        add_service: distribution(add_service),
        project_and_add_service: distribution(project_and_add_service),
        prefill_complete_service: distribution(prefill_complete_service),
        free_service: distribution(free_service),
        delayed_operations,
        queue_depth_at_stop,
        outstanding_at_stop,
        maximum_queue_depth,
        issue_span_ns,
        drain_ns,
        issued_operations,
        completed_operations,
        successful_operations,
        exact_ids,
        per_worker_fifo,
        zero_fixed_buffer_overflow,
        issue_span_valid,
        final_state_empty,
        generator_valid,
        kept_up,
        failure_reasons,
    }
}

fn push_failure(failures: &mut Vec<String>, failure: String) {
    if !failures.contains(&failure) {
        failures.push(failure);
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

#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::{
        ActiveOperationKind, ActiveSequencesRunConfig, BenchmarkClock, CompletionFailure,
        IssuerFailure, LanePayload, LanePayloadSlot, OperationLane,
        prepare_active_sequences_corpus, run_active_sequences_benchmark,
    };
    use crate::active_sequences_shared::{SequenceTrace, SequenceTraceEntry};
    use dynamo_bench::kv_router_common::replay::NoopSequencePublisher;
    use dynamo_kv_router::protocols::{PrefillLoadHint, WorkerWithDpRank};
    use dynamo_kv_router::{ActiveSequenceStride, ActiveSequencesMultiWorker, SequenceRequest};

    fn add(request_id: &str, hashes: &[u64], timestamp_us: u64) -> SequenceTrace {
        SequenceTrace {
            entry: SequenceTraceEntry::Add {
                request_id: request_id.to_string(),
                block_hashes: hashes.to_vec(),
                isl: hashes.len() * 128,
                output_length: 8,
            },
            timestamp_us,
        }
    }

    fn prefill(request_id: &str, timestamp_us: u64) -> SequenceTrace {
        SequenceTrace {
            entry: SequenceTraceEntry::PrefillComplete {
                request_id: request_id.to_string(),
            },
            timestamp_us,
        }
    }

    fn free(request_id: &str, timestamp_us: u64) -> SequenceTrace {
        SequenceTrace {
            entry: SequenceTraceEntry::Free {
                request_id: request_id.to_string(),
            },
            timestamp_us,
        }
    }

    fn request(request_id: &str, hashes: &[u64]) -> SequenceRequest {
        SequenceRequest {
            request_id: request_id.to_string(),
            token_sequence: Some(ActiveSequenceStride::ONE.sample_dense(hashes.to_vec())),
            track_prefill_tokens: true,
            expected_output_tokens: Some(8),
            prefill_load_hint: Some(PrefillLoadHint {
                initial_effective_prefill_tokens: hashes.len() * 128,
                expected_prefill_duration: None,
            }),
            worker: WorkerWithDpRank::from_worker_id(0),
            lora_name: None,
        }
    }

    #[test]
    fn preparation_normalizes_each_worker_and_preserves_equal_deadline_source_order() {
        let traces = vec![
            vec![add("a", &[1, 2], 10), prefill("a", 10), free("a", 20)],
            vec![add("b", &[3], 1_000), prefill("b", 1_000), free("b", 1_010)],
        ];
        let corpus = prepare_active_sequences_corpus(traces, 128, 1_000, 1).unwrap();

        for worker_id in 0..2 {
            let worker_ops = corpus
                .operations
                .iter()
                .filter(|operation| operation.worker_id == worker_id)
                .collect::<Vec<_>>();
            assert_eq!(worker_ops.first().unwrap().deadline_ns, 0);
            assert_eq!(worker_ops.last().unwrap().deadline_ns, 1_000_000_000);
            assert_eq!(
                worker_ops
                    .iter()
                    .take(2)
                    .map(|operation| operation.kind)
                    .collect::<Vec<_>>(),
                [
                    ActiveOperationKind::ProjectAndAdd,
                    ActiveOperationKind::PrefillComplete,
                ]
            );
            assert_eq!(worker_ops[0].source_ordinal, 0);
            assert_eq!(worker_ops[1].source_ordinal, 1);
        }
    }

    #[test]
    fn preparation_flattens_hashes_and_counts_logical_block_visits() {
        let traces = vec![vec![
            add("a", &[1, 2], 0),
            prefill("a", 0),
            add("b", &[3], 0),
            prefill("b", 0),
            free("a", 1),
            free("b", 1),
        ]];
        let corpus = prepare_active_sequences_corpus(traces, 128, 100, 1).unwrap();
        let add_operations = corpus
            .operations
            .iter()
            .filter(|operation| operation.kind == ActiveOperationKind::ProjectAndAdd)
            .map(|operation| {
                (
                    operation.source_ordinal,
                    corpus.operation_hashes(operation.id).unwrap().to_vec(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(add_operations, [(0, vec![1, 2]), (2, vec![3]),]);
        let totals = corpus.totals();
        assert_eq!(totals.adds, 2);
        assert_eq!(totals.prefill_completes, 2);
        assert_eq!(totals.frees, 2);
        assert_eq!(totals.input_blocks, 3);
        assert_eq!(totals.projection_block_visits, 3);
        assert_eq!(totals.add_registration_block_visits, 3);
        assert_eq!(totals.free_release_block_visits, 3);
        assert_eq!(totals.logical_block_visits(), 9);
    }

    #[test]
    fn preparation_rejects_invalid_lifecycle() {
        let error = prepare_active_sequences_corpus(vec![vec![free("missing", 0)]], 128, 100, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("freed before Add"));

        let error =
            prepare_active_sequences_corpus(vec![vec![add("unfinished", &[1], 0)]], 128, 100, 1)
                .unwrap_err()
                .to_string();
        assert!(error.contains("without a final Free"));

        let error = prepare_active_sequences_corpus(
            vec![vec![add("unordered", &[1], 20), free("unordered", 10)]],
            128,
            100,
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("timestamps are not ordered"));

        let error = prepare_active_sequences_corpus(
            vec![vec![add("zero", &[1], 0), free("zero", 1)]],
            128,
            0,
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("replay window must be positive"));
    }

    #[test]
    fn fixed_lane_reports_overflow() {
        let lane = OperationLane::new(1);
        assert!(lane.write(0, 1).is_ok());
        assert_eq!(lane.write(1, 2), Err(IssuerFailure::LaneOverflow));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fixed_lane_does_not_lose_publish_before_wait() {
        let lane = Arc::new(OperationLane::new(1));
        lane.write(0, 7).unwrap();
        lane.publish(1);
        lane.close();
        let sequences = Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
            NoopSequencePublisher,
            128,
            HashMap::from([(0, (0, 1))]),
            false,
            0,
            "bench",
        ));
        let clock = Arc::new(BenchmarkClock::new(0).unwrap());
        let (ready_tx, _ready_rx) = tokio::sync::oneshot::channel();
        let result = super::operation_lane_worker(
            sequences,
            lane,
            vec![LanePayloadSlot {
                id: 7,
                payload: Some(LanePayload::Free(0)),
            }]
            .into_boxed_slice(),
            vec!["missing".to_string()].into_boxed_slice(),
            clock,
            ready_tx,
        )
        .await;
        assert_eq!(result.written, 1);
        assert_eq!(result.failure, super::LaneFailure::None);
    }

    #[test]
    fn direct_executor_records_sequence_errors() {
        let sequences = ActiveSequencesMultiWorker::new_without_expiry(
            NoopSequencePublisher,
            128,
            HashMap::from([(0, (0, 1))]),
            false,
            0,
            "bench",
        );
        let now = tokio::time::Instant::now();
        sequences
            .add_request(request("duplicate", &[1]), now)
            .unwrap();
        let clock = BenchmarkClock::new(0).unwrap();
        let completion = super::execute_payload(
            &sequences,
            &clock,
            0,
            &[],
            Some(LanePayload::ProjectAndAdd(request("duplicate", &[1]))),
        );
        assert_eq!(completion.failure, CompletionFailure::DuplicateRequest);
        sequences.free(&"duplicate".to_string(), now).unwrap();
        sequences.assert_completely_drained(now);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn small_replay_has_exact_ids_timed_drain_and_stable_json_shape() {
        let corpus = prepare_active_sequences_corpus(
            vec![vec![add("a", &[1, 2], 10), prefill("a", 10), free("a", 10)]],
            128,
            100,
            1,
        )
        .unwrap();
        let result = run_active_sequences_benchmark(
            corpus,
            ActiveSequencesRunConfig {
                operation_lanes: 1,
                spin_us: 50,
                issue_lag_diagnostic_threshold_us: 250,
            },
        )
        .await
        .unwrap();
        assert!(result.generator_valid, "{:?}", result.failure_reasons);
        assert!(result.final_state_empty);
        assert!(result.exact_ids);
        assert!(result.per_worker_fifo);
        assert!(result.zero_fixed_buffer_overflow);
        assert!(result.issue_span_valid);
        assert_eq!(result.total_logical_operations, 3);
        assert_eq!(result.issued_operations, 3);
        assert_eq!(result.completed_operations, 3);
        assert_eq!(result.successful_operations, 3);
        assert_eq!(result.total_logical_block_visits, 6);
        assert_eq!(result.worker_projections_produced, 1);
        assert_eq!(result.worker_projections_inspected, 1);
        assert_eq!(result.queue_depth_at_stop.len(), 1);
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["total_adds"], 1);
        assert_eq!(json["total_prefill_completes"], 1);
        assert_eq!(json["total_frees"], 1);
        assert_eq!(json["failure_reasons"], serde_json::json!([]));
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Context, bail};
use crossbeam_queue::ArrayQueue;
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::agg::{AggRuntimeImpl, AggRuntimeStats, AggregatedPlacement};
use super::components::{
    AdmissionQueue, ExecutedGroupEpoch, IsolatedWorkerGroup, ReplayAdmissionMetadata,
    ReplayEngineObservation, TrafficAccumulator, WorkerCompletionAugment, WorkerCompletionMutation,
};
use super::core::{
    AdmissionSource as CoreAdmissionSource, EngineEventBatch, EngineProgress, Placement,
    PlacementDecision, ReadyArrival,
};
use super::events::{SimulationWorkerStage, WorkerCompletionPayload};
use super::evidence::{KvIngestBoundary, WorkerPool, lifecycle_capture_active};
use super::progress::ReplayProgress;
use super::state::AggRequestState;
use crate::common::protocols::{DirectRequest, ForwardPassSnapshot, OutputSignal};
use crate::loadgen::ReplayRequestPayload;
use crate::replay::{ReplayRequestPool, ReplayTerminalStatus, TraceCollector};
use crate::scheduler::{AdmissionEvent, RouterEventVisibility};

const EPOCH_CHUNK_CAPACITY: usize = 64;
const MAX_STARTS_PER_CHUNK: usize = (EPOCH_CHUNK_CAPACITY - 1) / 2;
const READY_CHUNK_DEPTH: usize = 2;
const CHUNKS_PER_GROUP: usize = READY_CHUNK_DEPTH + 1;

struct CoordinatorAffinityGuard {
    #[cfg(target_os = "linux")]
    original_cpus: Vec<usize>,
}

impl Drop for CoordinatorAffinityGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if !self.original_cpus.is_empty() {
            let _ = set_current_thread_affinity(&self.original_cpus);
        }
    }
}

fn reserve_dedicated_cpus(lane_count: usize) -> (CoordinatorAffinityGuard, Vec<Option<usize>>) {
    #[cfg(target_os = "linux")]
    {
        let Ok(original_cpus) = current_thread_affinity() else {
            return (
                CoordinatorAffinityGuard {
                    original_cpus: Vec::new(),
                },
                vec![None; lane_count],
            );
        };
        if original_cpus.len() <= lane_count {
            return (
                CoordinatorAffinityGuard {
                    original_cpus: Vec::new(),
                },
                vec![None; lane_count],
            );
        }
        if set_current_thread_affinity(&original_cpus[..1]).is_err() {
            return (
                CoordinatorAffinityGuard {
                    original_cpus: Vec::new(),
                },
                vec![None; lane_count],
            );
        }
        let lane_cpus = original_cpus[1..=lane_count]
            .iter()
            .copied()
            .map(Some)
            .collect();
        (CoordinatorAffinityGuard { original_cpus }, lane_cpus)
    }

    #[cfg(not(target_os = "linux"))]
    {
        (CoordinatorAffinityGuard {}, vec![None; lane_count])
    }
}

#[cfg(target_os = "linux")]
fn current_thread_affinity() -> std::io::Result<Vec<usize>> {
    // SAFETY: the zeroed value is a valid empty CPU set for sched_getaffinity.
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    // SAFETY: set points to writable storage of the supplied size.
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&set), &mut set) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let cpus = (0..libc::CPU_SETSIZE as usize)
        // SAFETY: every checked index is below CPU_SETSIZE.
        .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
        .collect();
    Ok(cpus)
}

#[cfg(target_os = "linux")]
fn set_current_thread_affinity(cpus: &[usize]) -> std::io::Result<()> {
    // SAFETY: the zeroed value is a valid empty CPU set.
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    for &cpu in cpus {
        if cpu >= libc::CPU_SETSIZE as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("CPU {cpu} exceeds CPU_SETSIZE"),
            ));
        }
        // SAFETY: cpu is checked against CPU_SETSIZE above.
        unsafe { libc::CPU_SET(cpu, &mut set) };
    }
    // SAFETY: set points to readable storage of the supplied size.
    if unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ConservativeEpochId {
    pub(super) worker_id: usize,
    pub(super) ordinal: u64,
}

pub(super) struct ConservativeEpochStart<Events: EngineEventBatch> {
    pub(super) id: ConservativeEpochId,
    pub(super) start_ms: f64,
    pub(super) end_ms: f64,
    pub(super) admissions: Vec<AdmissionEvent>,
    pub(super) pass_start_events: Events,
    pub(super) payloads: Vec<WorkerCompletionPayload<Events>>,
}

pub(super) struct ConservativeEpochCompletion<Events: EngineEventBatch> {
    pub(super) id: ConservativeEpochId,
    pub(super) at_ms: f64,
    pub(super) augments: Vec<WorkerCompletionAugment<Events>>,
}

pub(super) enum ConservativeAction<Events: EngineEventBatch> {
    Start(ConservativeEpochStart<Events>),
    Complete(ConservativeEpochCompletion<Events>),
}

impl<Events: EngineEventBatch> ConservativeAction<Events> {
    pub(super) fn at_ms(&self) -> f64 {
        match self {
            Self::Start(start) => start.start_ms,
            Self::Complete(completion) => completion.at_ms,
        }
    }
}

struct PendingEpoch {
    id: ConservativeEpochId,
    end_ms: f64,
    mutations: Vec<WorkerCompletionMutation>,
}

#[derive(Clone, Copy, Default)]
struct GroupWindowState {
    tail_completion_ms: Option<f64>,
    next_start_ms: Option<f64>,
    in_flight: usize,
    drained: bool,
    covered: bool,
}

struct GroupChunk<Events: EngineEventBatch> {
    generation: u64,
    actions: VecDeque<ConservativeAction<Events>>,
    state: GroupWindowState,
}

impl<Events: EngineEventBatch> Default for GroupChunk<Events> {
    fn default() -> Self {
        Self {
            generation: 0,
            actions: VecDeque::with_capacity(EPOCH_CHUNK_CAPACITY),
            state: GroupWindowState::default(),
        }
    }
}

struct GroupStream<Events: EngineEventBatch> {
    ready: ArrayQueue<GroupChunk<Events>>,
    recycle: ArrayQueue<GroupChunk<Events>>,
    queued_actions: AtomicUsize,
}

impl<Events: EngineEventBatch> GroupStream<Events> {
    fn new() -> Self {
        let stream = Self {
            ready: ArrayQueue::new(READY_CHUNK_DEPTH),
            recycle: ArrayQueue::new(CHUNKS_PER_GROUP),
            queued_actions: AtomicUsize::new(0),
        };
        for _ in 0..CHUNKS_PER_GROUP {
            stream
                .recycle
                .push(GroupChunk::default())
                .unwrap_or_else(|_| unreachable!("new recycle queue has exact capacity"));
        }
        stream
    }
}

/// A lane writes its slots before publishing completion with release ordering.
/// The coordinator reads and drains them only after the matching acquire load.
struct SharedSlot<T>(UnsafeCell<T>);

impl<T> SharedSlot<T> {
    fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    fn as_mut_ptr(&self) -> *mut T {
        self.0.get()
    }
}

// SAFETY: access is serialized by each lane's generation/completion atomics.
unsafe impl<T: Send> Send for SharedSlot<T> {}
// SAFETY: access is serialized by each lane's generation/completion atomics.
unsafe impl<T: Send> Sync for SharedSlot<T> {}

type SharedGroupStream<Events> = Arc<GroupStream<Events>>;

struct DispatchCommand {
    rank_id: usize,
    request: DirectRequest,
}

struct LaneGroup<Observation: ReplayEngineObservation> {
    state: IsolatedWorkerGroup,
    pending: Option<PendingEpoch>,
    next_ordinal: u64,
    cursor_ms: f64,
    stream: SharedGroupStream<Observation::Batch>,
    marker: std::marker::PhantomData<Observation>,
}

impl<Observation: ReplayEngineObservation> LaneGroup<Observation> {
    fn new(state: IsolatedWorkerGroup, stream: SharedGroupStream<Observation::Batch>) -> Self {
        Self {
            state,
            pending: None,
            next_ordinal: 0,
            cursor_ms: 0.0,
            stream,
            marker: std::marker::PhantomData,
        }
    }

    fn dispatch(&mut self, rank_id: usize, request: DirectRequest) -> anyhow::Result<()> {
        self.state.dispatch(rank_id, request)
    }

    fn run_window(
        &mut self,
        now_ms: f64,
        horizon_ms: f64,
        max_starts: Option<usize>,
        actions: &mut VecDeque<ConservativeAction<Observation::Batch>>,
    ) -> anyhow::Result<GroupWindowState> {
        debug_assert!(actions.is_empty());
        let mut cursor_ms = self.cursor_ms.max(now_ms);
        let mut starts = 0;

        loop {
            if let Some(pending) = self.pending.take() {
                if pending.end_ms > horizon_ms {
                    self.pending = Some(pending);
                    break;
                }
                cursor_ms = pending.end_ms;
                let mut augments = Vec::with_capacity(pending.mutations.len());
                for mutation in pending.mutations {
                    augments.push(
                        self.state
                            .apply_completion_mutation::<Observation>(mutation)?,
                    );
                }
                actions.push_back(ConservativeAction::Complete(ConservativeEpochCompletion {
                    id: pending.id,
                    at_ms: pending.end_ms,
                    augments,
                }));
            }

            if cursor_ms >= horizon_ms
                || max_starts.is_some_and(|limit| starts >= limit)
                || !self.state.is_ready()
            {
                break;
            }

            let epoch = self.state.execute_epoch(cursor_ms)?;
            let start = lower_epoch::<Observation>(
                ConservativeEpochId {
                    worker_id: self.state.worker_id(),
                    ordinal: self.next_ordinal,
                },
                epoch,
            );
            self.next_ordinal = self
                .next_ordinal
                .checked_add(1)
                .expect("conservative replay epoch ordinal overflow");
            starts += 1;

            let Some((start, mutations)) = start else {
                break;
            };
            self.state
                .mark_epoch_started_from_times(start.start_ms, start.end_ms, &mutations);
            let end_ms = start.end_ms;
            let id = start.id;
            actions.push_back(ConservativeAction::Start(start));
            if end_ms > cursor_ms {
                self.pending = Some(PendingEpoch {
                    id,
                    end_ms,
                    mutations,
                });
                continue;
            }

            let mut augments = Vec::with_capacity(mutations.len());
            for mutation in mutations {
                augments.push(
                    self.state
                        .apply_completion_mutation::<Observation>(mutation)?,
                );
            }
            actions.push_back(ConservativeAction::Complete(ConservativeEpochCompletion {
                id,
                at_ms: cursor_ms,
                augments,
            }));
        }

        self.cursor_ms = cursor_ms;
        let stopped_at_limit = max_starts.is_some_and(|limit| starts >= limit);
        let covered = self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.end_ms > horizon_ms)
            || cursor_ms >= horizon_ms
            || !self.state.is_ready()
            || !stopped_at_limit;
        Ok(GroupWindowState {
            tail_completion_ms: self.pending.as_ref().map(|pending| pending.end_ms),
            next_start_ms: (!covered).then_some(cursor_ms),
            in_flight: self.state.in_flight(),
            drained: self.state.is_drained(),
            covered,
        })
    }

    fn try_publish_chunk(
        &mut self,
        generation: u64,
        now_ms: f64,
        horizon_ms: f64,
        max_starts: Option<usize>,
        continuous: bool,
    ) -> anyhow::Result<Option<bool>> {
        if self.stream.ready.is_full() {
            return Ok(None);
        }
        let Some(mut chunk) = self.stream.recycle.pop() else {
            return Ok(None);
        };
        debug_assert!(chunk.actions.is_empty());
        let mut state = match self.run_window(now_ms, horizon_ms, max_starts, &mut chunk.actions) {
            Ok(state) => state,
            Err(error) => {
                self.stream
                    .recycle
                    .push(chunk)
                    .unwrap_or_else(|_| unreachable!("producer owns a recycled chunk"));
                return Err(error);
            }
        };
        if !continuous {
            state.next_start_ms = None;
            state.covered = true;
        }
        chunk.generation = generation;
        chunk.state = state;
        let action_count = chunk.actions.len();
        self.stream
            .queued_actions
            .fetch_add(action_count, Ordering::Relaxed);
        self.stream
            .ready
            .push(chunk)
            .unwrap_or_else(|_| unreachable!("single producer checked ready capacity"));
        Ok(Some(state.covered))
    }
}

fn lower_epoch<Observation: ReplayEngineObservation>(
    id: ConservativeEpochId,
    epoch: ExecutedGroupEpoch,
) -> Option<(
    ConservativeEpochStart<Observation::Batch>,
    Vec<WorkerCompletionMutation>,
)> {
    let start_ms = epoch.start_ms();
    let end_ms = epoch.end_ms();
    let completion_capacity = epoch.completion_capacity();
    let wall_time_secs = epoch.wall_time_secs();
    let mut admissions = Vec::new();
    let mut pass_start_events = Observation::Batch::default();
    let mut payloads = Vec::with_capacity(completion_capacity);
    let mut mutations = Vec::with_capacity(completion_capacity);
    let mut any_effect = false;

    for rank in epoch.ranks {
        let Some(mut executed) = rank.executed else {
            if end_ms > start_ms {
                payloads.push(WorkerCompletionPayload {
                    stage: SimulationWorkerStage::Aggregated,
                    worker_idx: rank.rank_id,
                    completed_requests: 0,
                    output_signals: Vec::new(),
                    lifecycle_events: Vec::new(),
                    engine_events: Observation::Batch::default(),
                    progress: EngineProgress::default(),
                    fpm: Some(ForwardPassSnapshot {
                        wall_time_secs,
                        ..Default::default()
                    }),
                    accept_length_output_tokens: 0,
                    accept_length_decode_forwards: 0,
                });
                mutations.push(WorkerCompletionMutation {
                    rank_id: rank.rank_id,
                    completed_requests: 0,
                });
                any_effect = true;
            }
            continue;
        };

        if let Some(fpm) = executed.fpm.as_mut() {
            fpm.wall_time_secs = wall_time_secs;
        }
        let admitted_requests = !executed.admissions.is_empty();
        let had_raw_observations = !executed.kv_events.is_empty();
        let published_pass_start_kv = executed.router_event_visibility
            == RouterEventVisibility::PassStart
            && had_raw_observations;
        let fpm_has_scheduled_work = executed
            .fpm
            .as_ref()
            .is_some_and(|fpm| fpm.num_prefill_requests > 0 || fpm.num_decode_requests > 0);
        let made_progress = admitted_requests
            || published_pass_start_kv
            || executed.completed_requests > 0
            || !executed.output_signals.is_empty()
            || !executed.lifecycle_events.is_empty()
            || had_raw_observations
            || fpm_has_scheduled_work;
        let mut observed_events = Observation::take_pass_events(&mut executed);
        admissions.append(&mut executed.admissions);
        let completion_events =
            if executed.router_event_visibility == RouterEventVisibility::PassStart {
                pass_start_events.append(observed_events);
                Observation::Batch::default()
            } else {
                std::mem::take(&mut observed_events)
            };
        let completed_requests = executed.completed_requests;
        payloads.push(WorkerCompletionPayload {
            stage: SimulationWorkerStage::Aggregated,
            worker_idx: rank.rank_id,
            completed_requests,
            output_signals: executed.output_signals,
            lifecycle_events: executed.lifecycle_events,
            engine_events: completion_events,
            progress: EngineProgress {
                made_progress,
                had_raw_observations,
            },
            fpm: executed.fpm,
            accept_length_output_tokens: executed.accept_length_output_tokens,
            accept_length_decode_forwards: executed.accept_length_decode_forwards,
        });
        mutations.push(WorkerCompletionMutation {
            rank_id: rank.rank_id,
            completed_requests,
        });
        any_effect |= made_progress;
    }

    if !any_effect && end_ms <= start_ms {
        return None;
    }
    Some((
        ConservativeEpochStart {
            id,
            start_ms,
            end_ms,
            admissions,
            pass_start_events,
            payloads,
        },
        mutations,
    ))
}

struct LaneCommandData {
    now_ms: f64,
    horizon_ms: f64,
    max_starts: Option<usize>,
    continuous: bool,
    all_workers: bool,
    worker_ids: Vec<usize>,
    dispatches: Vec<DispatchCommand>,
}

impl Default for LaneCommandData {
    fn default() -> Self {
        Self {
            now_ms: 0.0,
            horizon_ms: 0.0,
            max_starts: None,
            continuous: false,
            all_workers: true,
            worker_ids: Vec::new(),
            dispatches: Vec::new(),
        }
    }
}

struct LaneControl {
    generation: AtomicU64,
    completed_generation: AtomicU64,
    shutdown: AtomicBool,
    failed: AtomicBool,
    command: SharedSlot<LaneCommandData>,
    busy_secs: SharedSlot<f64>,
    queue_stall_secs: SharedSlot<f64>,
    fatal_error: Mutex<Option<anyhow::Error>>,
}

impl LaneControl {
    fn new() -> Self {
        Self {
            generation: AtomicU64::new(0),
            completed_generation: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            command: SharedSlot::new(LaneCommandData::default()),
            busy_secs: SharedSlot::new(0.0),
            queue_stall_secs: SharedSlot::new(0.0),
            fatal_error: Mutex::new(None),
        }
    }
}

fn run_lane<Observation>(
    groups: &mut [LaneGroup<Observation>],
    control: &LaneControl,
) -> anyhow::Result<()>
where
    Observation: ReplayEngineObservation + Send + 'static,
{
    let mut observed_generation = 0;
    let mut selected = vec![false; groups.len()];
    loop {
        let generation = loop {
            if control.shutdown.load(Ordering::Acquire) {
                return Ok(());
            }
            let generation = control.generation.load(Ordering::Acquire);
            if generation != observed_generation {
                break generation;
            }
            std::hint::spin_loop();
        };
        if control.shutdown.load(Ordering::Acquire) {
            return Ok(());
        }

        // SAFETY: the coordinator published this command before the generation
        // release store and does not touch it until this lane reports completion.
        let command = unsafe { &mut *control.command.as_mut_ptr() };
        for dispatch in command.dispatches.drain(..) {
            let group = groups
                .iter_mut()
                .find(|group| {
                    group
                        .state
                        .rank_identities()
                        .iter()
                        .any(|(candidate, _)| *candidate == dispatch.rank_id)
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "conservative lane received unknown rank {}",
                        dispatch.rank_id
                    )
                })?;
            group.dispatch(dispatch.rank_id, dispatch.request)?;
        }

        let mut remaining = 0;
        for (index, group) in groups.iter().enumerate() {
            selected[index] =
                command.all_workers || command.worker_ids.contains(&group.state.worker_id());
            remaining += usize::from(selected[index]);
        }
        let mut stall_start: Option<std::time::Instant> = None;
        while remaining > 0 {
            if control.shutdown.load(Ordering::Acquire) {
                return Ok(());
            }
            let mut made_progress = false;
            for (index, group) in groups.iter_mut().enumerate() {
                if !selected[index] {
                    continue;
                }
                let busy_start = std::time::Instant::now();
                let published = group.try_publish_chunk(
                    generation,
                    command.now_ms,
                    command.horizon_ms,
                    command.max_starts,
                    command.continuous,
                )?;
                if let Some(covered) = published {
                    // SAFETY: this lane is the only writer. The coordinator reads
                    // the counter after the final completion-generation acquire.
                    *unsafe { &mut *control.busy_secs.as_mut_ptr() } +=
                        busy_start.elapsed().as_secs_f64();
                    made_progress = true;
                    if covered {
                        selected[index] = false;
                        remaining -= 1;
                    }
                }
            }
            if made_progress {
                if let Some(start) = stall_start.take() {
                    // SAFETY: this lane is the only writer.
                    *unsafe { &mut *control.queue_stall_secs.as_mut_ptr() } +=
                        start.elapsed().as_secs_f64();
                }
            } else {
                stall_start.get_or_insert_with(std::time::Instant::now);
                std::hint::spin_loop();
            }
        }
        if let Some(start) = stall_start.take() {
            // SAFETY: this lane is the only writer.
            *unsafe { &mut *control.queue_stall_secs.as_mut_ptr() } +=
                start.elapsed().as_secs_f64();
        }
        observed_generation = generation;
        control
            .completed_generation
            .store(generation, Ordering::Release);
    }
}

struct LaneHandle {
    control: Arc<LaneControl>,
    thread: Option<JoinHandle<()>>,
}

pub(super) struct ConservativeEngine<Observation: ReplayEngineObservation> {
    lanes: Vec<LaneHandle>,
    rank_to_lane: Vec<usize>,
    group_to_lane: Vec<usize>,
    rank_identity: Vec<(usize, u32)>,
    group_streams: Vec<SharedGroupStream<Observation::Batch>>,
    current_chunks: Vec<Option<GroupChunk<Observation::Batch>>>,
    group_states: Vec<GroupWindowState>,
    selected_groups: Vec<bool>,
    next_generation: u64,
    _coordinator_affinity: CoordinatorAffinityGuard,
}

impl<Observation> ConservativeEngine<Observation>
where
    Observation: ReplayEngineObservation + Send + 'static,
{
    pub(super) fn new(groups: Vec<IsolatedWorkerGroup>, lane_count: usize) -> anyhow::Result<Self> {
        if groups.is_empty() {
            bail!("conservative replay requires at least one logical worker");
        }
        let lane_count = lane_count.clamp(1, groups.len());
        let (coordinator_affinity, lane_cpus) = reserve_dedicated_cpus(lane_count);
        let group_streams = (0..groups.len())
            .map(|_| Arc::new(GroupStream::new()))
            .collect::<Vec<_>>();
        let mut groups_by_lane = (0..lane_count).map(|_| Vec::new()).collect::<Vec<_>>();
        let max_rank_id = groups
            .iter()
            .flat_map(IsolatedWorkerGroup::rank_identities)
            .map(|(rank_id, _)| rank_id)
            .max()
            .unwrap_or(0);
        let mut rank_to_lane = vec![usize::MAX; max_rank_id + 1];
        let mut rank_identity = vec![(usize::MAX, u32::MAX); max_rank_id + 1];
        let mut group_to_lane = vec![usize::MAX; groups.len()];
        for group in groups {
            let worker_id = group.worker_id();
            let lane_id = worker_id % lane_count;
            group_to_lane[worker_id] = lane_id;
            for (rank_id, dp_rank) in group.rank_identities() {
                rank_to_lane[rank_id] = lane_id;
                rank_identity[rank_id] = (worker_id, dp_rank);
            }
            groups_by_lane[lane_id].push(LaneGroup::<Observation>::new(
                group,
                Arc::clone(&group_streams[worker_id]),
            ));
        }

        let mut lanes: Vec<LaneHandle> = Vec::with_capacity(lane_count);
        for (lane_id, mut groups) in groups_by_lane.into_iter().enumerate() {
            let control = Arc::new(LaneControl::new());
            let lane_control = Arc::clone(&control);
            let lane_cpu = lane_cpus[lane_id];
            let spawn_result = thread::Builder::new()
                .name(format!("offline-replay-lane-{lane_id}"))
                .spawn(move || {
                    #[cfg(target_os = "linux")]
                    if let Some(cpu) = lane_cpu
                        && let Err(error) = set_current_thread_affinity(&[cpu])
                    {
                        *lane_control
                            .fatal_error
                            .lock()
                            .expect("lane error mutex poisoned") = Some(anyhow::anyhow!(
                            "failed to pin conservative replay lane {lane_id} to CPU {cpu}: {error}"
                        ));
                        lane_control.failed.store(true, Ordering::Release);
                        lane_control
                            .completed_generation
                            .store(u64::MAX, Ordering::Release);
                        return;
                    }
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_lane::<Observation>(&mut groups, &lane_control)
                    }));
                    let error = match outcome {
                        Ok(Ok(())) => return,
                        Ok(Err(error)) => error,
                        Err(payload) => {
                            let message = payload
                                .downcast_ref::<&str>()
                                .map(|value| (*value).to_string())
                                .or_else(|| payload.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown panic payload".to_string());
                            anyhow::anyhow!(
                                "conservative replay lane {lane_id} panicked: {message}"
                            )
                        }
                    };
                    *lane_control
                        .fatal_error
                        .lock()
                        .expect("lane error mutex poisoned") = Some(error);
                    lane_control.failed.store(true, Ordering::Release);
                    lane_control
                        .completed_generation
                        .store(u64::MAX, Ordering::Release);
                });
            let thread = match spawn_result {
                Ok(thread) => thread,
                Err(error) => {
                    for lane in &lanes {
                        lane.control.shutdown.store(true, Ordering::Release);
                        lane.control.generation.fetch_add(1, Ordering::Release);
                    }
                    for lane in &mut lanes {
                        if let Some(thread) = lane.thread.take() {
                            let _ = thread.join();
                        }
                    }
                    return Err(error).with_context(|| {
                        format!("failed to spawn conservative replay lane {lane_id}")
                    });
                }
            };
            lanes.push(LaneHandle {
                control,
                thread: Some(thread),
            });
        }

        let worker_count = group_streams.len();
        Ok(Self {
            lanes,
            rank_to_lane,
            group_to_lane,
            rank_identity,
            group_streams,
            current_chunks: (0..worker_count).map(|_| None).collect(),
            group_states: vec![GroupWindowState::default(); worker_count],
            selected_groups: vec![false; worker_count],
            next_generation: 0,
            _coordinator_affinity: coordinator_affinity,
        })
    }

    pub(super) fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    pub(super) fn worker_count(&self) -> usize {
        self.group_to_lane.len()
    }

    fn lane_busy_secs(&self) -> Vec<f64> {
        self.lanes
            .iter()
            .map(|lane| {
                // SAFETY: this is read only after the final completed generation.
                *unsafe { &mut *lane.control.busy_secs.as_mut_ptr() }
            })
            .collect()
    }

    fn queue_stall_secs(&self) -> f64 {
        self.lanes
            .iter()
            .map(|lane| {
                // SAFETY: this is read only after the final completed generation.
                *unsafe { &mut *lane.control.queue_stall_secs.as_mut_ptr() }
            })
            .sum()
    }

    pub(super) fn rank_identity(&self, rank_id: usize) -> Option<(usize, u32)> {
        self.rank_identity
            .get(rank_id)
            .copied()
            .filter(|(worker_id, _)| *worker_id != usize::MAX)
    }

    pub(super) fn dispatch(
        &mut self,
        rank_id: usize,
        request: DirectRequest,
    ) -> anyhow::Result<()> {
        let lane_id = *self
            .rank_to_lane
            .get(rank_id)
            .filter(|lane_id| **lane_id != usize::MAX)
            .ok_or_else(|| anyhow::anyhow!("offline replay selected unknown rank {rank_id}"))?;
        let control = &self.lanes[lane_id].control;
        if control.generation.load(Ordering::Acquire)
            != control.completed_generation.load(Ordering::Acquire)
        {
            bail!("conservative replay dispatched work while a lane was executing");
        }
        // SAFETY: the lane is idle between completed and next published generation.
        unsafe { &mut *control.command.as_mut_ptr() }
            .dispatches
            .push(DispatchCommand { rank_id, request });
        Ok(())
    }

    pub(super) fn run_window(&mut self, now_ms: f64, horizon_ms: f64) -> anyhow::Result<()> {
        self.start_all(now_ms, horizon_ms, Some(MAX_STARTS_PER_CHUNK), true)
    }

    pub(super) fn settle_completions(&mut self, now_ms: f64) -> anyhow::Result<()> {
        self.start_all(now_ms, now_ms, Some(0), false)
    }

    pub(super) fn run_group_step(&mut self, worker_id: usize, now_ms: f64) -> anyhow::Result<()> {
        self.group_to_lane
            .get(worker_id)
            .filter(|lane_id| **lane_id != usize::MAX)
            .ok_or_else(|| anyhow::anyhow!("offline replay selected unknown worker {worker_id}"))?;
        self.start_selected(
            now_ms,
            next_nonnegative_float(now_ms),
            Some(1),
            false,
            &[worker_id],
        )
    }

    fn start_all(
        &mut self,
        now_ms: f64,
        horizon_ms: f64,
        max_starts: Option<usize>,
        continuous: bool,
    ) -> anyhow::Result<()> {
        self.start_on_lanes(now_ms, horizon_ms, max_starts, continuous, true, &[])
    }

    fn start_selected(
        &mut self,
        now_ms: f64,
        horizon_ms: f64,
        max_starts: Option<usize>,
        continuous: bool,
        worker_ids: &[usize],
    ) -> anyhow::Result<()> {
        self.start_on_lanes(
            now_ms, horizon_ms, max_starts, continuous, false, worker_ids,
        )
    }

    fn start_on_lanes(
        &mut self,
        now_ms: f64,
        horizon_ms: f64,
        max_starts: Option<usize>,
        continuous: bool,
        all_workers: bool,
        worker_ids: &[usize],
    ) -> anyhow::Result<()> {
        if horizon_ms < now_ms {
            bail!("conservative replay horizon {horizon_ms} precedes current time {now_ms}");
        }
        if self.current_chunks.iter().any(Option::is_some)
            || self
                .group_streams
                .iter()
                .any(|stream| !stream.ready.is_empty())
        {
            bail!("conservative replay started a window before draining its prior streams");
        }
        let generation = self
            .next_generation
            .checked_add(1)
            .expect("conservative replay generation overflow");
        self.next_generation = generation;
        self.selected_groups.fill(false);
        if all_workers {
            self.selected_groups.fill(true);
        } else {
            for &worker_id in worker_ids {
                let selected = self.selected_groups.get_mut(worker_id).ok_or_else(|| {
                    anyhow::anyhow!("offline replay selected unknown worker {worker_id}")
                })?;
                *selected = true;
            }
        }
        for worker_id in 0..self.worker_count() {
            if self.selected_groups[worker_id] {
                self.group_states[worker_id].next_start_ms = Some(now_ms);
                self.group_states[worker_id].covered = false;
            }
        }
        for lane_id in 0..self.lanes.len() {
            if !all_workers
                && !worker_ids
                    .iter()
                    .any(|worker_id| self.group_to_lane[*worker_id] == lane_id)
            {
                continue;
            }
            let control = &self.lanes[lane_id].control;
            if control.generation.load(Ordering::Acquire)
                != control.completed_generation.load(Ordering::Acquire)
            {
                bail!("conservative replay lane {lane_id} did not settle its prior window");
            }
            // SAFETY: the prior generation completed and the next is not published yet.
            let command = unsafe { &mut *control.command.as_mut_ptr() };
            command.now_ms = now_ms;
            command.horizon_ms = horizon_ms;
            command.max_starts = max_starts;
            command.continuous = continuous;
            command.all_workers = all_workers;
            command.worker_ids.clear();
            if !all_workers {
                command.worker_ids.extend(
                    worker_ids
                        .iter()
                        .copied()
                        .filter(|worker_id| self.group_to_lane[*worker_id] == lane_id),
                );
            }
            control.generation.store(generation, Ordering::Release);
        }
        Ok(())
    }

    fn group_state(&self, worker_id: usize) -> GroupWindowState {
        self.group_states[worker_id]
    }

    fn pop_action(&mut self, worker_id: usize) -> Option<ConservativeAction<Observation::Batch>> {
        let action = self.current_chunks[worker_id].as_mut()?.actions.pop_front();
        if self.current_chunks[worker_id]
            .as_ref()
            .is_some_and(|chunk| chunk.actions.is_empty())
        {
            let chunk = self.current_chunks[worker_id]
                .take()
                .expect("empty current chunk must exist");
            self.group_streams[worker_id]
                .recycle
                .push(chunk)
                .unwrap_or_else(|_| unreachable!("coordinator owns a drained chunk"));
        }
        action
    }

    fn front_action_time(&self, worker_id: usize) -> Option<f64> {
        self.current_chunks[worker_id]
            .as_ref()?
            .actions
            .front()
            .map(ConservativeAction::at_ms)
    }

    fn front_completion_id(&self, worker_id: usize, at_ms: f64) -> Option<ConservativeEpochId> {
        match self.current_chunks[worker_id]
            .as_ref()
            .and_then(|chunk| chunk.actions.front())
        {
            Some(ConservativeAction::Complete(completion))
                if completion.at_ms.to_bits() == at_ms.to_bits() =>
            {
                Some(completion.id)
            }
            _ => None,
        }
    }

    fn current_action_count(&self, target_worker: Option<usize>) -> usize {
        self.current_chunks
            .iter()
            .enumerate()
            .filter(|(worker_id, _)| target_worker.is_none_or(|target| target == *worker_id))
            .map(|(_, chunk)| chunk.as_ref().map_or(0, |chunk| chunk.actions.len()))
            .sum()
    }

    fn buffered_actions(&self, target_worker: Option<usize>) -> usize {
        (0..self.worker_count())
            .filter(|worker_id| target_worker.is_none_or(|target| target == *worker_id))
            .map(|worker_id| {
                self.current_chunks[worker_id]
                    .as_ref()
                    .map_or(0, |chunk| chunk.actions.len())
                    + self.group_streams[worker_id]
                        .queued_actions
                        .load(Ordering::Relaxed)
            })
            .sum()
    }

    fn next_unseen_start(&self, worker_id: usize) -> Option<f64> {
        self.group_state(worker_id).next_start_ms
    }

    fn try_load_chunks(&mut self, target_worker: Option<usize>) -> anyhow::Result<usize> {
        let mut loaded = 0;
        for worker_id in 0..self.worker_count() {
            if !self.selected_groups[worker_id]
                || target_worker.is_some_and(|target| target != worker_id)
                || self.current_chunks[worker_id].is_some()
            {
                continue;
            }
            let stream = &self.group_streams[worker_id];
            if stream.ready.is_empty() {
                continue;
            }
            while let Some(chunk) = stream.ready.pop() {
                if chunk.generation != self.next_generation {
                    bail!(
                        "conservative worker {worker_id} published generation {} while generation {} was active",
                        chunk.generation,
                        self.next_generation
                    );
                }
                let action_count = chunk.actions.len();
                self.group_streams[worker_id]
                    .queued_actions
                    .fetch_sub(action_count, Ordering::Relaxed);
                self.group_states[worker_id] = chunk.state;
                loaded += 1;
                if chunk.actions.is_empty() {
                    self.group_streams[worker_id]
                        .recycle
                        .push(chunk)
                        .unwrap_or_else(|_| unreachable!("coordinator owns a loaded chunk"));
                    continue;
                }
                self.current_chunks[worker_id] = Some(chunk);
                break;
            }
        }
        Ok(loaded)
    }

    fn streams_drained(&self, target_worker: Option<usize>) -> bool {
        (0..self.worker_count())
            .filter(|worker_id| {
                self.selected_groups[*worker_id]
                    && target_worker.is_none_or(|target| target == *worker_id)
            })
            .all(|worker_id| {
                self.group_states[worker_id].covered
                    && self.current_chunks[worker_id].is_none()
                    && self.group_streams[worker_id].ready.is_empty()
            })
    }

    fn selected_lane(&self, lane_id: usize) -> bool {
        self.selected_groups
            .iter()
            .enumerate()
            .any(|(worker_id, selected)| *selected && self.group_to_lane[worker_id] == lane_id)
    }

    fn check_selected_lanes(&mut self) -> anyhow::Result<()> {
        for lane_id in 0..self.lanes.len() {
            if !self.selected_lane(lane_id) {
                continue;
            }
            let lane = &self.lanes[lane_id];
            if lane.control.failed.load(Ordering::Acquire) {
                if let Some(error) = lane
                    .control
                    .fatal_error
                    .lock()
                    .expect("lane error mutex poisoned")
                    .take()
                {
                    return Err(error);
                }
                bail!("conservative replay lane {lane_id} failed without an error");
            }
        }
        Ok(())
    }

    fn wait_for_stream_progress(&mut self, target_worker: Option<usize>) -> anyhow::Result<usize> {
        let mut spins = 0_u8;
        loop {
            let loaded = self.try_load_chunks(target_worker)?;
            if loaded > 0 || self.streams_drained(target_worker) {
                return Ok(loaded);
            }
            spins = spins.wrapping_add(1);
            if spins == 0 {
                self.check_selected_lanes()?;
            }
            std::hint::spin_loop();
        }
    }

    fn wait_current_window_complete(&mut self) -> anyhow::Result<()> {
        let mut spins = 0_u8;
        loop {
            let mut complete = true;
            for lane_id in 0..self.lanes.len() {
                if self.selected_lane(lane_id)
                    && self.lanes[lane_id]
                        .control
                        .completed_generation
                        .load(Ordering::Acquire)
                        != self.next_generation
                {
                    complete = false;
                    break;
                }
            }
            if complete {
                self.check_selected_lanes()?;
                return Ok(());
            }
            spins = spins.wrapping_add(1);
            if spins == 0 {
                self.check_selected_lanes()?;
            }
            std::hint::spin_loop();
        }
    }
}

fn next_nonnegative_float(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value >= 0.0);
    f64::from_bits(value.to_bits() + 1)
}

struct PendingGlobalEpoch<Events: EngineEventBatch> {
    id: ConservativeEpochId,
    seq_no: u64,
    end_ms: f64,
    payloads: Vec<WorkerCompletionPayload<Events>>,
}

#[derive(Clone, Copy)]
struct MergeKey {
    at_ms: f64,
    phase: u8,
    tie: u64,
}

impl MergeKey {
    fn start(at_ms: f64, worker_id: usize) -> Self {
        Self {
            at_ms,
            phase: 1,
            tie: worker_id as u64,
        }
    }

    fn completion(at_ms: f64, seq_no: u64) -> Self {
        Self {
            at_ms,
            phase: 0,
            tie: seq_no,
        }
    }

    fn cmp(self, other: Self) -> std::cmp::Ordering {
        self.at_ms
            .total_cmp(&other.at_ms)
            .then_with(|| self.phase.cmp(&other.phase))
            .then_with(|| self.tie.cmp(&other.tie))
    }
}

#[derive(Default)]
struct ConservativeRunStats {
    safe_windows: u64,
    window_chunks: u64,
    serial_fallback_steps: u64,
    epochs: u64,
    max_epochs_per_window: u64,
    peak_buffered_effects: usize,
    coordinator_wait_secs: f64,
    merge_secs: f64,
}

struct ConservativeAggRuntime<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation + Send + 'static,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: AggregatedPlacement<Observation::Batch, Metadata>,
{
    now_ms: f64,
    next_event_seq: u64,
    admission: AdmissionQueue<Metadata>,
    requests: FxHashMap<Uuid, AggRequestState>,
    engine: ConservativeEngine<Observation>,
    collector: TraceCollector,
    placement: PlacementPolicyImpl,
    progress: ReplayProgress,
    runtime_stats: AggRuntimeStats,
    traffic: TrafficAccumulator,
    max_sim_time_ms: Option<f64>,
    total_in_flight: usize,
    pending_epochs: Vec<Option<PendingGlobalEpoch<Observation::Batch>>>,
    tail_completion_ms: Vec<Option<f64>>,
    group_drained: Vec<bool>,
    stats: ConservativeRunStats,
    safe_merge: bool,
    #[cfg(test)]
    worker_active_requests: Vec<Vec<Uuid>>,
}

pub(super) fn run_aggregated_conservative<PlacementPolicyImpl, Observation, Metadata>(
    runtime: AggRuntimeImpl<PlacementPolicyImpl, Observation, Metadata>,
    lane_count: usize,
) -> anyhow::Result<(TraceCollector, AggRuntimeStats)>
where
    Observation: ReplayEngineObservation + Send + 'static,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: AggregatedPlacement<Observation::Batch, Metadata>,
{
    if runtime.scaling_policy.is_some() {
        bail!("conservative replay does not support dynamic scaling or planning");
    }
    if runtime.collect_fpm {
        bail!("conservative replay does not support lifecycle evidence capture");
    }
    if lifecycle_capture_active() {
        bail!("conservative replay does not support lifecycle-evidence capture");
    }
    if !runtime.events.is_empty() {
        bail!("conservative replay requires an empty global event queue at startup");
    }
    runtime.admission.validate_conservative_windows()?;
    runtime.engine.validate_conservative_windows()?;
    if let Some(cap_ms) = runtime.max_sim_time_ms
        && (!cap_ms.is_finite() || cap_ms < 0.0)
    {
        bail!("max_sim_time_ms must be a finite, non-negative value; got {cap_ms}");
    }

    let AggRuntimeImpl {
        now_ms,
        dp_size: _,
        next_event_seq,
        next_scaling_tick_ordinal: _,
        admission,
        requests,
        engine,
        mut collector,
        events: _,
        placement,
        progress,
        stats: runtime_stats,
        fpm_buffer: _,
        traffic,
        max_sim_time_ms,
        scaling_policy: _,
        collect_fpm: _,
        #[cfg(test)]
        worker_active_requests,
        #[cfg(test)]
            stepped: _,
    } = runtime;
    let available_parallelism = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let background_token_folding = available_parallelism >= 3;
    let lane_count = if background_token_folding {
        lane_count.min(available_parallelism - 2).max(1)
    } else {
        lane_count.min(available_parallelism.saturating_sub(1).max(1))
    };
    if background_token_folding {
        collector.enable_background_token_timeline_folding();
    }
    let total_in_flight = engine.in_flight();
    let groups = engine.into_isolated_groups()?;
    let worker_count = groups.len();
    let engine = ConservativeEngine::<Observation>::new(groups, lane_count)?;
    let actual_lanes = engine.lane_count();
    tracing::info!(
        configured_lanes = lane_count,
        actual_lanes,
        worker_count,
        "enabled conservative aggregated replay windows"
    );
    let mut runtime = ConservativeAggRuntime {
        now_ms,
        next_event_seq,
        admission,
        requests,
        engine,
        collector,
        placement,
        progress,
        runtime_stats,
        traffic,
        max_sim_time_ms,
        total_in_flight,
        pending_epochs: (0..worker_count).map(|_| None).collect(),
        tail_completion_ms: vec![None; worker_count],
        group_drained: vec![false; worker_count],
        stats: ConservativeRunStats::default(),
        safe_merge: false,
        #[cfg(test)]
        worker_active_requests,
    };
    let run_start = std::time::Instant::now();
    runtime.run()?;
    let run_secs = run_start.elapsed().as_secs_f64();
    runtime.progress.finish();
    let lane_busy_secs = runtime.engine.lane_busy_secs();
    let lane_busy_total_secs = lane_busy_secs.iter().sum::<f64>();
    let lane_busy_min_secs = lane_busy_secs
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let lane_busy_max_secs = lane_busy_secs
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let lane_utilization = if run_secs > 0.0 {
        lane_busy_total_secs / (run_secs * actual_lanes as f64)
    } else {
        0.0
    };
    tracing::info!(
        lanes = actual_lanes,
        safe_windows = runtime.stats.safe_windows,
        window_chunks = runtime.stats.window_chunks,
        serial_fallback_steps = runtime.stats.serial_fallback_steps,
        epochs = runtime.stats.epochs,
        max_epochs_per_window = runtime.stats.max_epochs_per_window,
        peak_buffered_effects = runtime.stats.peak_buffered_effects,
        coordinator_wait_secs = runtime.stats.coordinator_wait_secs,
        merge_secs = runtime.stats.merge_secs,
        queue_stall_secs = runtime.engine.queue_stall_secs(),
        lane_busy_total_secs,
        lane_busy_min_secs,
        lane_busy_max_secs,
        lane_utilization,
        run_secs,
        "conservative aggregated replay statistics"
    );
    eprintln!(
        "conservative_replay_stats lanes={actual_lanes} safe_windows={} window_chunks={} \
         serial_fallback_steps={} epochs={} max_epochs_per_window={} peak_buffered_effects={} \
         coordinator_wait_secs={:.6} merge_secs={:.6} queue_stall_secs={:.6} \
         lane_busy_total_secs={lane_busy_total_secs:.6} lane_busy_min_secs={lane_busy_min_secs:.6} \
         lane_busy_max_secs={lane_busy_max_secs:.6} lane_utilization={lane_utilization:.6} \
         run_secs={run_secs:.6}",
        runtime.stats.safe_windows,
        runtime.stats.window_chunks,
        runtime.stats.serial_fallback_steps,
        runtime.stats.epochs,
        runtime.stats.max_epochs_per_window,
        runtime.stats.peak_buffered_effects,
        runtime.stats.coordinator_wait_secs,
        runtime.stats.merge_secs,
        runtime.engine.queue_stall_secs(),
    );
    Ok((runtime.collector, runtime.runtime_stats))
}

impl<PlacementPolicyImpl, Observation, Metadata>
    ConservativeAggRuntime<PlacementPolicyImpl, Observation, Metadata>
where
    Observation: ReplayEngineObservation + Send + 'static,
    Metadata: ReplayAdmissionMetadata,
    PlacementPolicyImpl: AggregatedPlacement<Observation::Batch, Metadata>,
{
    fn run(&mut self) -> anyhow::Result<()> {
        self.release_ready_arrivals()?;
        self.drive_serial_starts_if_needed()?;

        while !self.is_done() {
            if self.placement.pending_count() > 0 {
                if self.run_serial_fallback_step()? {
                    break;
                }
                continue;
            }

            let next_arrival_ms = CoreAdmissionSource::next_ready_time_ms(&mut self.admission);
            let horizon_ms = match (next_arrival_ms, self.max_sim_time_ms) {
                (Some(arrival), Some(cap)) => arrival.min(cap),
                (Some(arrival), None) => arrival,
                (None, Some(cap)) => cap,
                (None, None) => f64::INFINITY,
            };
            if horizon_ms < self.now_ms {
                bail!(
                    "conservative replay horizon {horizon_ms} precedes current time {}",
                    self.now_ms
                );
            }

            self.stats.safe_windows += 1;
            let epochs_before = self.stats.epochs;
            let window_start_ms = self.now_ms;
            self.engine.run_window(window_start_ms, horizon_ms)?;
            self.drain_started_window(None, true)?;
            self.stats.max_epochs_per_window = self
                .stats
                .max_epochs_per_window
                .max(self.stats.epochs - epochs_before);

            if horizon_ms.is_finite() {
                self.advance_now_ms(horizon_ms);
            }
            if self.max_sim_time_ms == Some(horizon_ms)
                && next_arrival_ms.is_none_or(|arrival| arrival > horizon_ms)
            {
                break;
            }
            self.release_ready_arrivals()?;
            self.drive_serial_starts_if_needed()?;
            if !horizon_ms.is_finite() && !self.is_done() {
                bail!(
                    "conservative replay reached a dead end with {} in-flight requests",
                    self.total_in_flight + self.placement.pending_count()
                );
            }
        }
        Ok(())
    }

    fn is_done(&mut self) -> bool {
        self.total_in_flight == 0
            && self.placement.pending_count() == 0
            && CoreAdmissionSource::is_drained(&self.admission)
            && self.group_drained.iter().all(|drained| *drained)
    }

    fn earliest_tail(&self) -> Option<f64> {
        self.tail_completion_ms
            .iter()
            .flatten()
            .copied()
            .min_by(f64::total_cmp)
    }

    /// Return `true` when the soft simulation cap stops further stepping.
    fn run_serial_fallback_step(&mut self) -> anyhow::Result<bool> {
        self.stats.serial_fallback_steps += 1;
        let next_arrival = CoreAdmissionSource::next_ready_time_ms(&mut self.admission);
        let next_time = match (self.earliest_tail(), next_arrival) {
            (Some(completion), Some(arrival)) => completion.min(arrival),
            (Some(completion), None) => completion,
            (None, Some(arrival)) => arrival,
            (None, None) => {
                bail!(
                    "conservative replay reached a dead end with {} in-flight requests",
                    self.total_in_flight + self.placement.pending_count()
                )
            }
        };
        if let Some(cap_ms) = self.max_sim_time_ms
            && next_time > cap_ms
        {
            self.advance_now_ms(cap_ms);
            return Ok(true);
        }
        self.advance_now_ms(next_time);
        self.engine.settle_completions(self.now_ms)?;
        self.drain_started_window(None, false)?;
        self.release_ready_arrivals()?;
        self.drive_serial_starts_if_needed()?;
        Ok(false)
    }

    fn drive_serial_starts_if_needed(&mut self) -> anyhow::Result<()> {
        if self.placement.pending_count() == 0 {
            return Ok(());
        }
        loop {
            let mut made_progress = false;
            for worker_id in 0..self.engine.worker_count() {
                self.engine.run_group_step(worker_id, self.now_ms)?;
                let applied = self.drain_started_window(Some(worker_id), false)?;
                if applied > 0 {
                    made_progress = true;
                    break;
                }
            }
            if !made_progress {
                return Ok(());
            }
        }
    }

    fn drain_started_window(
        &mut self,
        target_worker: Option<usize>,
        concurrent_merge: bool,
    ) -> anyhow::Result<usize> {
        if !concurrent_merge {
            let wait_start = std::time::Instant::now();
            self.engine.wait_current_window_complete()?;
            self.stats.coordinator_wait_secs += wait_start.elapsed().as_secs_f64();
        }

        let mut applied = 0;
        loop {
            applied += self.merge_group_slots(target_worker)?;
            if self.engine.streams_drained(target_worker) {
                break;
            }
            let wait_start = std::time::Instant::now();
            let loaded = self.engine.wait_for_stream_progress(target_worker)?;
            self.stats.coordinator_wait_secs += wait_start.elapsed().as_secs_f64();
            self.stats.window_chunks += loaded as u64;
        }

        if concurrent_merge {
            let wait_start = std::time::Instant::now();
            self.engine.wait_current_window_complete()?;
            self.stats.coordinator_wait_secs += wait_start.elapsed().as_secs_f64();
        }
        Ok(applied)
    }

    fn merge_group_slots(&mut self, target_worker: Option<usize>) -> anyhow::Result<usize> {
        let merge_start = std::time::Instant::now();
        for worker_id in 0..self.engine.worker_count() {
            if target_worker.is_none_or(|target| target == worker_id) {
                self.update_group_state(worker_id);
            }
        }
        let buffered = self.engine.buffered_actions(target_worker);
        self.stats.peak_buffered_effects = self.stats.peak_buffered_effects.max(buffered);
        let mut remaining = self.engine.current_action_count(target_worker);
        let mut applied = 0;
        let frontier = target_worker.is_none().then(|| {
            (0..self.engine.worker_count())
                .filter_map(|worker_id| {
                    self.engine
                        .next_unseen_start(worker_id)
                        .map(|at_ms| MergeKey::start(at_ms, worker_id))
                })
                .min_by(|left, right| left.cmp(*right))
        });
        let frontier = frontier.flatten();

        while remaining > 0 {
            let min_time = (0..self.engine.worker_count())
                .filter(|worker_id| target_worker.is_none_or(|target| target == *worker_id))
                .filter_map(|worker_id| self.engine.front_action_time(worker_id))
                .min_by(f64::total_cmp)
                .expect("non-empty conservative slots must have a minimum time");
            let mut selected = None;
            let mut selected_seq = u64::MAX;
            for worker_id in 0..self.engine.worker_count() {
                if target_worker.is_some_and(|target| target != worker_id) {
                    continue;
                }
                if let Some(id) = self.engine.front_completion_id(worker_id, min_time) {
                    let seq_no = self.pending_epoch(id)?.seq_no;
                    if seq_no < selected_seq {
                        selected = Some(worker_id);
                        selected_seq = seq_no;
                    }
                }
            }
            if selected.is_none() {
                selected = (0..self.engine.worker_count())
                    .filter(|worker_id| target_worker.is_none_or(|target| target == *worker_id))
                    .filter(|worker_id| {
                        self.engine
                            .front_action_time(*worker_id)
                            .is_some_and(|at_ms| at_ms.to_bits() == min_time.to_bits())
                    })
                    .min();
            }
            let worker_id = selected.expect("conservative merge could not select a ready action");
            let key = self
                .front_action_key(worker_id)?
                .expect("selected conservative action must have an ordering key");
            if frontier.is_some_and(|frontier| key.cmp(frontier).is_ge()) {
                break;
            }
            let action = self
                .engine
                .pop_action(worker_id)
                .expect("selected conservative action must exist");
            self.advance_now_ms(action.at_ms());
            match action {
                ConservativeAction::Start(start) => self.apply_epoch_start(start)?,
                ConservativeAction::Complete(completion) => {
                    self.apply_epoch_completion(completion)?
                }
            }
            remaining -= 1;
            applied += 1;
        }
        self.stats.merge_secs += merge_start.elapsed().as_secs_f64();
        Ok(applied)
    }

    fn front_action_key(&self, worker_id: usize) -> anyhow::Result<Option<MergeKey>> {
        let Some(at_ms) = self.engine.front_action_time(worker_id) else {
            return Ok(None);
        };
        if let Some(id) = self.engine.front_completion_id(worker_id, at_ms) {
            return Ok(Some(MergeKey::completion(
                at_ms,
                self.pending_epoch(id)?.seq_no,
            )));
        }
        Ok(Some(MergeKey::start(at_ms, worker_id)))
    }

    fn update_group_state(&mut self, worker_id: usize) {
        let state = self.engine.group_state(worker_id);
        self.tail_completion_ms[worker_id] = state.tail_completion_ms;
        self.group_drained[worker_id] = state.drained;
        debug_assert!(state.in_flight <= self.total_in_flight);
    }

    fn pending_epoch(
        &self,
        id: ConservativeEpochId,
    ) -> anyhow::Result<&PendingGlobalEpoch<Observation::Batch>> {
        self.pending_epochs
            .get(id.worker_id)
            .and_then(Option::as_ref)
            .filter(|pending| pending.id == id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "conservative replay completion references unknown epoch {}:{}",
                    id.worker_id,
                    id.ordinal
                )
            })
    }

    fn apply_epoch_start(
        &mut self,
        start: ConservativeEpochStart<Observation::Batch>,
    ) -> anyhow::Result<()> {
        if self.pending_epochs[start.id.worker_id].is_some() {
            bail!(
                "conservative worker {} started an epoch with an outstanding completion",
                start.id.worker_id
            );
        }
        for admission in &start.admissions {
            self.collector.on_admit(
                admission.uuid,
                start.start_ms,
                admission.reused_input_tokens,
            );
            self.collector.on_pool_admission(
                admission.uuid,
                ReplayRequestPool::Agg,
                start.start_ms,
                admission.reused_input_tokens,
            );
        }
        for payload in &start.payloads {
            self.collector.on_output_signals(
                &payload.output_signals,
                start.end_ms,
                payload.accept_length_output_tokens > payload.accept_length_decode_forwards,
            );
        }
        self.apply_observations_without_release(
            start.pass_start_events,
            KvIngestBoundary::PassStart,
        )?;
        let payload_count =
            u64::try_from(start.payloads.len()).expect("completion payload count must fit in u64");
        if payload_count == 0 {
            bail!("conservative replay started an epoch without completion payloads");
        }
        let seq_no = self.next_event_seq;
        self.next_event_seq = self
            .next_event_seq
            .checked_add(payload_count)
            .expect("offline replay event sequence overflow");
        self.pending_epochs[start.id.worker_id] = Some(PendingGlobalEpoch {
            id: start.id,
            seq_no,
            end_ms: start.end_ms,
            payloads: start.payloads,
        });
        self.stats.epochs += 1;
        Ok(())
    }

    fn apply_epoch_completion(
        &mut self,
        completion: ConservativeEpochCompletion<Observation::Batch>,
    ) -> anyhow::Result<()> {
        let mut pending = self.pending_epochs[completion.id.worker_id]
            .take()
            .filter(|pending| pending.id == completion.id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "conservative replay completion references unknown epoch {}:{}",
                    completion.id.worker_id,
                    completion.id.ordinal
                )
            })?;
        if pending.end_ms.to_bits() != completion.at_ms.to_bits() {
            bail!("conservative replay completion time does not match its epoch");
        }
        for augment in completion.augments {
            let payload = pending
                .payloads
                .iter_mut()
                .find(|payload| payload.worker_idx == augment.rank_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "conservative replay completion augment references unknown rank {}",
                        augment.rank_id
                    )
                })?;
            payload.lifecycle_events.extend(augment.lifecycle_events);
            payload.engine_events.append(augment.engine_events);
            payload.progress.had_raw_observations |= augment.had_raw_observations;
            payload.progress.made_progress |= augment.had_raw_observations;
        }
        for payload in pending.payloads {
            self.total_in_flight = self
                .total_in_flight
                .checked_sub(payload.completed_requests)
                .expect("conservative replay completed more requests than it owned");
            self.process_completed_pass(payload)?;
        }
        Ok(())
    }

    fn apply_observations_without_release(
        &mut self,
        events: Observation::Batch,
        boundary: KvIngestBoundary,
    ) -> anyhow::Result<()> {
        Observation::record_ingestion(&events, WorkerPool::Agg, boundary, self.now_ms)?;
        let placements = self.placement.observe(events, self.now_ms)?;
        if self.safe_merge && !placements.is_empty() {
            bail!("conservative replay crossed a router causality boundary inside a safe window");
        }
        self.dispatch_placements(placements)
    }

    fn process_completed_pass(
        &mut self,
        payload: WorkerCompletionPayload<Observation::Batch>,
    ) -> anyhow::Result<()> {
        self.apply_observations_without_release(payload.engine_events, KvIngestBoundary::PassEnd)?;
        self.traffic.on_accept_length_sample(
            payload.accept_length_output_tokens,
            payload.accept_length_decode_forwards,
        );
        for signal in payload.output_signals {
            self.process_output_signal(signal)?;
        }
        Ok(())
    }

    fn process_output_signal(&mut self, signal: OutputSignal) -> anyhow::Result<()> {
        if let Some(token_id) = signal.token_id {
            CoreAdmissionSource::on_output_token(&mut self.admission, signal.uuid, token_id)?;
        }
        if signal.completed {
            let status = if signal.rejected {
                ReplayTerminalStatus::Rejected
            } else {
                ReplayTerminalStatus::Completed
            };
            self.collector.on_terminal(signal.uuid, self.now_ms, status);
            #[cfg(test)]
            self.remove_active_request(signal.uuid);
            let placements = self.placement.request_terminal(signal.uuid, self.now_ms)?;
            if self.safe_merge && !placements.is_empty() {
                bail!(
                    "conservative replay terminal completion released queued work inside a safe window"
                );
            }
            let removed_state = self.requests.remove(&signal.uuid).ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?;
            if !signal.rejected {
                let latencies = self.collector.request_latencies(signal.uuid);
                let actual_output_tokens = self
                    .collector
                    .actual_output_length(signal.uuid)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "offline replay missing collector state for {}",
                            signal.uuid
                        )
                    })?;
                self.traffic.on_completion(
                    removed_state.input_tokens,
                    actual_output_tokens,
                    latencies,
                );
            }
            CoreAdmissionSource::on_terminal(
                &mut self.admission,
                signal.uuid,
                self.now_ms,
                signal.rejected,
            )?;
            self.progress.inc_completed();
            self.dispatch_placements(placements)?;
            return Ok(());
        }
        let state = self.requests.get_mut(&signal.uuid).ok_or_else(|| {
            anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
        })?;
        if state.prefill_completed {
            return Ok(());
        }
        state.prefill_completed = true;
        let placements = self.placement.prefill_completed(signal.uuid, self.now_ms)?;
        if self.safe_merge && !placements.is_empty() {
            bail!(
                "conservative replay prefill completion released queued work inside a safe window"
            );
        }
        self.dispatch_placements(placements)
    }

    fn release_ready_arrivals(&mut self) -> anyhow::Result<()> {
        let cluster_in_flight = self.total_in_flight + self.placement.pending_count();
        for ready in self
            .admission
            .drain_ready_compact(self.now_ms, cluster_in_flight)?
        {
            let ReadyArrival {
                request,
                arrival_time_ms,
                metadata,
                session_id,
                turn_index,
            } = ready;
            let session_metadata = session_id.clone().zip(turn_index);
            let uuid = self.assign_request(request, arrival_time_ms, metadata, session_id)?;
            if let Some((session_id, turn_index)) = session_metadata {
                self.collector
                    .on_session_metadata(uuid, session_id, turn_index);
            }
        }
        Ok(())
    }

    fn assign_request(
        &mut self,
        mut request: ReplayRequestPayload,
        arrival_time_ms: f64,
        metadata: Metadata,
        session_id: Option<String>,
    ) -> anyhow::Result<Uuid> {
        let uuid = request.metadata().uuid.unwrap_or_else(Uuid::new_v4);
        let input_length = request.input_length();
        let output_length = request.metadata().effective_max_output_tokens();
        request.metadata_mut().uuid = Some(uuid);
        self.collector
            .on_arrival(uuid, arrival_time_ms, input_length, output_length);
        self.traffic.on_arrival();
        let effects = self
            .placement
            .place(&request, metadata, session_id, self.now_ms)?;
        match effects.decision {
            PlacementDecision::Immediate(placement) => {
                if placement.request_id != uuid {
                    bail!(
                        "offline placement returned request {} while placing {uuid}",
                        placement.request_id
                    );
                }
                self.record_placement(placement);
                let (logical_worker_id, dp_rank) = self
                    .engine
                    .rank_identity(placement.scheduler_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "offline replay placement references unknown scheduler {}",
                            placement.scheduler_id
                        )
                    })?;
                self.collector.on_route_immediate(
                    uuid,
                    ReplayRequestPool::Agg,
                    logical_worker_id,
                    placement.scheduler_id,
                    dp_rank,
                    placement.reported_overlap_tokens,
                );
                self.requests.insert(
                    uuid,
                    AggRequestState::new_running(input_length, output_length),
                );
                self.dispatch_to_worker(
                    request.into_direct_request(),
                    uuid,
                    placement.scheduler_id,
                )?;
            }
            PlacementDecision::Queued => {
                self.collector
                    .on_route_queued(uuid, ReplayRequestPool::Agg, self.now_ms);
                self.requests
                    .insert(uuid, AggRequestState::new_queued(request));
            }
        }
        self.dispatch_placements(effects.released)?;
        Ok(uuid)
    }

    fn record_placement(&mut self, placement: Placement) {
        if let Some(sample) = placement.planner_cache_sample {
            self.traffic
                .on_admission(sample.overlap_blocks, sample.isl_blocks);
        }
    }

    fn dispatch_placements(&mut self, placements: Vec<Placement>) -> anyhow::Result<()> {
        for placement in placements {
            self.record_placement(placement);
            let uuid = placement.request_id;
            let (logical_worker_id, dp_rank) = self
                .engine
                .rank_identity(placement.scheduler_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "offline replay placement references unknown scheduler {}",
                        placement.scheduler_id
                    )
                })?;
            self.collector.on_route_released(
                uuid,
                ReplayRequestPool::Agg,
                self.now_ms,
                logical_worker_id,
                placement.scheduler_id,
                dp_rank,
                placement.reported_overlap_tokens,
            );
            let request = self
                .requests
                .get_mut(&uuid)
                .ok_or_else(|| {
                    anyhow::anyhow!("offline replay missing queued request state for {uuid}")
                })?
                .take_queued_request(uuid)?;
            self.dispatch_to_worker(request, uuid, placement.scheduler_id)?;
        }
        Ok(())
    }

    fn dispatch_to_worker(
        &mut self,
        request: DirectRequest,
        uuid: Uuid,
        rank_id: usize,
    ) -> anyhow::Result<()> {
        self.engine.dispatch(rank_id, request)?;
        self.total_in_flight = self
            .total_in_flight
            .checked_add(1)
            .expect("conservative replay in-flight count overflow");
        self.collector.on_decode_assigned(uuid, rank_id);
        #[cfg(test)]
        self.worker_active_requests[rank_id].push(uuid);
        Ok(())
    }

    fn advance_now_ms(&mut self, new_now_ms: f64) {
        let dt_ms = (new_now_ms - self.now_ms).max(0.0);
        if dt_ms > 0.0 {
            let decode_worker_seconds = self.engine.worker_count() as f64 * dt_ms / 1000.0;
            self.collector
                .add_worker_seconds(0.0, decode_worker_seconds);
        }
        self.now_ms = new_now_ms;
    }

    #[cfg(test)]
    fn remove_active_request(&mut self, uuid: Uuid) {
        for active_requests in &mut self.worker_active_requests {
            if let Some(position) = active_requests
                .iter()
                .position(|candidate| *candidate == uuid)
            {
                active_requests.remove(position);
                return;
            }
        }
    }
}

impl<Observation: ReplayEngineObservation> Drop for ConservativeEngine<Observation> {
    fn drop(&mut self) {
        for lane in &self.lanes {
            lane.control.shutdown.store(true, Ordering::Release);
            lane.control.generation.fetch_add(1, Ordering::Release);
        }
        for lane in &mut self.lanes {
            if let Some(thread) = lane.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

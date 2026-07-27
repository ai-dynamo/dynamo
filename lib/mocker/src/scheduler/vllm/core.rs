// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use dynamo_tokens::blocks::UniqueBlock;
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::common::handoff::HandoffId;
use crate::common::protocols::{
    DirectRequest, KvEventPublishers, MockEngineArgs, MoveBlock, OutputSignal, PreemptionMode,
    PrefillCost, WorkerType,
};
use crate::common::sequence::ActiveSequence;
use crate::common::speculative::{SpeculativeDecodeSampler, normalize_conditional_accept_rates};
use crate::common::utils::{compute_prefill_handoff_delay_ms, prefill_handoff_transfer_timing};
use crate::kv_manager::G1Manager;
#[cfg(feature = "kvbm-offload")]
use crate::kv_manager::kvbm_backend::SwapInRegistrationBlock;
use crate::kv_manager::{DestinationReservation, G1Acquire, OffloadDependency};
#[cfg(feature = "kvbm-offload")]
use crate::kvbm_offload::coordinator::SwapInTerminal;
use crate::replay::TraceCollector;
use crate::scheduler::vllm::policy::{self, AdmissionDecision};
use crate::scheduler::{
    ActiveHandoffRequests, AdmissionEvent, AdmissionInvariant, AdmissionStage,
    CapturedRouterEventBuffer, DestinationHolds, EnginePassResult, ForwardPassSnapshot,
    MockerMetrics, PendingDestinations, RemovedSource, RouterEventVisibility, SchedulerCommand,
    SchedulerCommandEffects, SchedulerCommandResult, SchedulerLifecycleEvent, SourceCompletion,
    SourceHolds, accept_length_sample, build_fpm_snapshot, capture_router_event_sink,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestStatus {
    WaitingForRemoteKv,
    Waiting,
    Running,
    Preempted,
}

pub(crate) struct VllmRequestState {
    pub(crate) sequence: ActiveSequence,
    pub(crate) status: RequestStatus,
    pub(crate) num_computed_tokens: usize,
    pub(crate) num_preemptions: usize,
    pub(crate) offload_dependency: Option<OffloadDependency>,
}

impl VllmRequestState {
    fn prompt_is_prebuilt(&self) -> bool {
        self.num_computed_tokens >= self.sequence.num_input_tokens()
            && self.sequence.num_allocated_tokens() >= self.sequence.num_input_tokens()
    }

    fn debug_assert_invariants(&self, _uuid: Uuid) {
        #[cfg(debug_assertions)]
        {
            let uuid = _uuid;
            let seq_len = self.sequence.len();
            let allocated = self.sequence.num_allocated_tokens();
            debug_assert!(
                self.num_computed_tokens <= seq_len,
                "request {uuid} computed {} tokens but sequence length is {seq_len}",
                self.num_computed_tokens
            );
            debug_assert!(
                allocated <= seq_len,
                "request {uuid} allocated {allocated} tokens but sequence length is {seq_len}"
            );
        }
    }

    fn debug_assert_progress(&self, _uuid: Uuid) {
        #[cfg(debug_assertions)]
        {
            let uuid = _uuid;
            self.debug_assert_invariants(uuid);
            let allocated = self.sequence.num_allocated_tokens();
            debug_assert!(
                allocated >= self.num_computed_tokens,
                "request {uuid} allocated {allocated} tokens but computed {}",
                self.num_computed_tokens
            );
        }
    }
}

#[derive(Default)]
pub(crate) struct SchedulerState {
    pub(crate) waiting: VecDeque<Uuid>,
    waiting_members: FxHashSet<Uuid>,
    pub(crate) running: VecDeque<Uuid>,
    running_members: FxHashSet<Uuid>,
    pub(crate) requests: FxHashMap<Uuid, VllmRequestState>,
    pub(crate) preemptions_total: u64,
}

pub(super) struct PreemptedRequest {
    uuid: Uuid,
    signals: Vec<MoveBlock>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ScheduledWork {
    total_tokens: usize,
    prompt_tokens: usize,
    prefix_tokens: usize,
    terminal_after_schedule: bool,
    /// Full prompt length, captured at schedule time for FPM variance calculation.
    prompt_len: usize,
    /// Total sequence length (prompt + generated) at schedule time, used for
    /// decode KV context in FPM. Captured here because completed requests are
    /// removed from state before `compute_fpm` runs.
    sequence_len: usize,
}

enum ScheduleOutcome {
    Scheduled {
        tokens_used: usize,
        admission: Option<AdmissionEvent>,
    },
    Blocked,
    /// This request cannot progress yet, but later running requests may.
    DependencyBlocked,
    CurrentPreempted,
}

impl SchedulerState {
    pub(crate) fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn request_sequence_len(&self, uuid: Uuid) -> usize {
        self.requests
            .get(&uuid)
            .map(|request| request.sequence.len())
            .unwrap_or_default()
    }

    fn push_waiting(&mut self, uuid: Uuid) {
        if !self.waiting_members.insert(uuid) {
            return;
        }
        self.waiting.push_back(uuid);
    }

    fn insert_waiting(&mut self, uuid: Uuid, request: VllmRequestState) {
        debug_assert!(!self.requests.contains_key(&uuid));
        self.requests.insert(uuid, request);
        self.push_waiting(uuid);
    }

    fn prepend_waiting(&mut self, uuid: Uuid) {
        if !self.waiting_members.insert(uuid) {
            return;
        }
        self.waiting.push_front(uuid);
    }

    /// Remove `uuid` from the waiting queue and from the
    /// `waiting_members` set. Shared between `transition_to_running`
    /// (which then promotes to running) and the offload admission
    /// hook's parking path (which keeps the request in `Waiting`
    /// status while parked on a swap-in).
    fn remove_from_waiting(&mut self, uuid: Uuid) {
        if let Some(position) = self.waiting.iter().position(|waiting| *waiting == uuid) {
            self.waiting.remove(position);
        }
        self.waiting_members.remove(&uuid);
    }

    fn next_waiting_uuid(&mut self, prefer_materialized: bool) -> Option<Uuid> {
        loop {
            let uuid = *self.waiting.front()?;
            if self.waiting_members.contains(&uuid)
                && self
                    .requests
                    .get(&uuid)
                    .is_some_and(|request| request.status != RequestStatus::Running)
            {
                break;
            }
            self.waiting.pop_front();
            self.waiting_members.remove(&uuid);
        }

        if prefer_materialized {
            return self.waiting.iter().copied().find(|uuid| {
                self.waiting_members.contains(uuid)
                    && self
                        .requests
                        .get(uuid)
                        .is_some_and(VllmRequestState::prompt_is_prebuilt)
            });
        }
        self.waiting.front().copied()
    }

    #[cfg_attr(feature = "profile", inline(never))]
    fn compact_running(&mut self) {
        let mut compacted = VecDeque::with_capacity(self.running.len());
        while let Some(uuid) = self.running.pop_front() {
            let is_running = self.running_members.contains(&uuid)
                && self
                    .requests
                    .get(&uuid)
                    .is_some_and(|request| request.status == RequestStatus::Running);
            if is_running {
                compacted.push_back(uuid);
                continue;
            }
            self.running_members.remove(&uuid);
        }
        self.running = compacted;
    }

    fn transition_to_running(&mut self, uuid: Uuid) {
        self.remove_from_waiting(uuid);
        if self.running_members.insert(uuid) {
            self.running.push_back(uuid);
        }
        if let Some(request) = self.requests.get_mut(&uuid) {
            request.status = RequestStatus::Running;
        }
    }

    pub(crate) fn complete(&mut self, uuid: &Uuid) {
        self.take_completed(uuid);
    }

    pub(crate) fn take_completed(&mut self, uuid: &Uuid) -> Option<VllmRequestState> {
        self.waiting_members.remove(uuid);
        self.running_members.remove(uuid);
        self.requests.remove(uuid)
    }

    pub(crate) fn running_sequence_mut(&mut self, uuid: Uuid) -> Option<&mut ActiveSequence> {
        if !self.running_members.contains(&uuid) {
            return None;
        }
        self.requests
            .get_mut(&uuid)
            .map(|request| &mut request.sequence)
    }

    pub(super) fn preempt_uuid(&mut self, uuid: Uuid) -> Option<PreemptedRequest> {
        let is_running = self.running_members.contains(&uuid)
            && self
                .requests
                .get(&uuid)
                .is_some_and(|request| request.status == RequestStatus::Running);
        if !is_running {
            return None;
        }
        let position = self
            .running
            .iter()
            .position(|candidate| *candidate == uuid)?;
        self.running.remove(position);
        self.running_members.remove(&uuid);
        let request = self.requests.get_mut(&uuid)?;
        request.status = RequestStatus::Preempted;
        request.offload_dependency = None;
        request.num_computed_tokens = 0;
        request.num_preemptions += 1;
        self.preemptions_total += 1;
        let signals = request.sequence.reset_with_signal();
        request.debug_assert_invariants(uuid);
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                request.sequence.num_allocated_tokens(),
                0,
                "preempted request {uuid} should release all allocated KV"
            );
        }
        self.prepend_waiting(uuid);
        Some(PreemptedRequest { uuid, signals })
    }

    #[cfg(test)]
    pub(super) fn insert_running_for_test(&mut self, uuid: Uuid) {
        self.running_members.insert(uuid);
        self.running.push_back(uuid);
    }

    fn debug_assert_ready_to_decode(&self, _uuid: Uuid) {
        #[cfg(debug_assertions)]
        {
            let uuid = _uuid;
            let Some(request) = self.requests.get(&uuid) else {
                return;
            };
            let seq_len = request.sequence.len();
            if request.num_computed_tokens < seq_len {
                return;
            }
            let allocated = request.sequence.num_allocated_tokens();
            debug_assert_eq!(
                allocated, seq_len,
                "request {uuid} is decode-ready but allocated {allocated} tokens for sequence length {seq_len}"
            );
        }
    }

    fn debug_assert_invariants(&self) {
        #[cfg(debug_assertions)]
        {
            let mut seen = std::collections::HashSet::new();
            for uuid in &self.waiting_members {
                debug_assert!(
                    seen.insert(*uuid),
                    "request {uuid} appears multiple times across waiting/running queues"
                );
                let request = self
                    .requests
                    .get(uuid)
                    .expect("waiting request missing from state map");
                debug_assert!(
                    request.status != RequestStatus::Running,
                    "request {uuid} is queued in waiting but marked Running"
                );
                request.debug_assert_invariants(*uuid);
            }
            for uuid in &self.running_members {
                debug_assert!(
                    seen.insert(*uuid),
                    "request {uuid} appears multiple times across waiting/running queues"
                );
                let request = self
                    .requests
                    .get(uuid)
                    .expect("running request missing from state map");
                debug_assert_eq!(
                    request.status,
                    RequestStatus::Running,
                    "request {uuid} is queued in running but marked {:?}",
                    request.status
                );
                request.debug_assert_invariants(*uuid);
            }
            debug_assert!(
                self.waiting.len() >= self.waiting_members.len(),
                "waiting queue dropped live membership entries"
            );
            debug_assert!(
                self.running.len() >= self.running_members.len(),
                "running queue dropped live membership entries"
            );
        }
    }
}

/// A request parked on a pending G2→G1 swap-in. The scheduler holds
/// one per deferred request and polls `handle.is_complete()` each pass
/// via [`VllmCore::tick_and_promote_swap_ins`]; on completion the
/// request becomes promotable.
///
/// `skip_blocks` is the number of full prefix blocks that were already
/// cached in G1 at park time; the swap-in covers the next
/// `handle.block_count()` blocks starting at that offset. We need this
/// to register the right slice of the request's PLHs into G1 inactive
/// after the transfer completes. The coordinator keeps that cached prefix
/// resident until the suffix publishes Device-tier Stored events.
#[cfg(feature = "kvbm-offload")]
pub(crate) struct AwaitingSwapIn {
    pub(crate) uuid: Uuid,
    pub(crate) handle: crate::kvbm_offload::SwapInHandle,
    pub(crate) skip_blocks: usize,
}

#[cfg(feature = "kvbm-offload")]
enum SwapInAdmissionAttempt {
    NoHit,
    Parked,
    BlockedOnG1Offload,
}

pub(crate) struct VllmCore {
    pub(super) args: MockEngineArgs,
    dp_rank: u32,
    pub(super) state: SchedulerState,
    pub(super) kv_manager: G1Manager,
    speculative_sampler: Option<SpeculativeDecodeSampler>,
    kv_event_buffer: Option<CapturedRouterEventBuffer>,
    source_holds: SourceHolds<HeldVllmPrefill>,
    pending_destinations: PendingDestinations<VllmRequestState>,
    destination_holds: DestinationHolds<ReservedVllmDecode>,
    active_destination_handoffs: ActiveHandoffRequests,
    capacity_generation: u64,
    #[cfg(test)]
    destination_reservation_attempts: usize,
    lifecycle_events: Vec<SchedulerLifecycleEvent>,

    /// Requests parked on pending G2→G1 swap-ins. Populated by the
    /// admission path when a request's remaining prefix matches G2 only
    /// (not active, not inactive in G1); drained at pass entry by
    /// [`Self::tick_and_promote_swap_ins`] once the associated
    /// [`SwapInHandle`](crate::kvbm_offload::SwapInHandle) reports
    /// complete. Lives on core (not engine) so the engine stays
    /// request-agnostic — engine hands out opaque handles, core owns the
    /// uuid↔handle mapping.
    #[cfg(feature = "kvbm-offload")]
    pub(super) requests_awaiting_swap_in: Vec<AwaitingSwapIn>,
}

struct HeldVllmPrefill {
    request_id: Uuid,
    request: VllmRequestState,
    deferred_deref: Vec<MoveBlock>,
}

struct ReservedVllmDecode {
    request: VllmRequestState,
    kv: DestinationReservation,
}

impl ReservedVllmDecode {
    fn activate(self, kv_manager: &mut G1Manager) -> VllmRequestState {
        let Self { mut request, kv } = self;
        kv_manager.activate_destination(kv);
        let prompt_len = request.sequence.num_input_tokens();
        request.sequence.commit_allocation(prompt_len);
        request.num_computed_tokens = prompt_len;
        request.status = RequestStatus::Waiting;
        request
    }

    fn cancel(self, kv_manager: &mut G1Manager) {
        let Self { request: _, kv } = self;
        kv_manager.cancel_destination(kv);
    }
}

struct VllmTerminalEffects {
    immediate: Vec<MoveBlock>,
    cleanup: Vec<MoveBlock>,
}

impl VllmCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        Self::new_internal(args, 0, 0, None, KvEventPublishers::default())
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        Self::new_with_worker_rank(args, worker_id, 0, worker_id, true)
    }

    pub(crate) fn new_with_worker_rank(
        args: MockEngineArgs,
        worker_id: WorkerId,
        dp_rank: u32,
        seed_offset: u64,
        capture_kv_events: bool,
    ) -> Self {
        let (buffer, publishers) = if capture_kv_events {
            let (buffer, sink) = capture_router_event_sink(worker_id);
            (Some(buffer), KvEventPublishers::new(Some(sink), None))
        } else {
            (None, KvEventPublishers::default())
        };
        Self::new_internal(args, dp_rank, seed_offset, buffer, publishers)
    }

    pub(super) fn new_with_sink(
        args: MockEngineArgs,
        dp_rank: u32,
        kv_event_publishers: KvEventPublishers,
    ) -> Self {
        Self::new_internal(args, dp_rank, u64::from(dp_rank), None, kv_event_publishers)
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        seed_offset: u64,
        kv_event_buffer: Option<CapturedRouterEventBuffer>,
        kv_event_publishers: KvEventPublishers,
    ) -> Self {
        let args = args.normalized().expect("invalid MockEngineArgs");
        let kv_event_publishers = if args.enable_prefix_caching {
            kv_event_publishers
        } else {
            KvEventPublishers::default()
        };
        let speculative_sampler = args.aic_nextn.map(|nextn| {
            let rates =
                normalize_conditional_accept_rates(nextn, args.aic_nextn_accept_rates.as_deref())
                    .expect("normalized MTP acceptance rates");
            SpeculativeDecodeSampler::new(rates, args.aic_mtp_seed.wrapping_add(seed_offset))
        });
        Self {
            kv_manager: G1Manager::new_with_backend_and_caching(
                args.num_gpu_blocks,
                args.block_size,
                kv_event_publishers,
                dp_rank,
                args.g1_backend,
                args.enable_prefix_caching,
            ),
            args,
            dp_rank,
            state: SchedulerState::default(),
            speculative_sampler,
            kv_event_buffer,
            source_holds: SourceHolds::default(),
            pending_destinations: PendingDestinations::default(),
            destination_holds: DestinationHolds::default(),
            active_destination_handoffs: ActiveHandoffRequests::default(),
            capacity_generation: 0,
            #[cfg(test)]
            destination_reservation_attempts: 0,
            lifecycle_events: Vec::new(),
            #[cfg(feature = "kvbm-offload")]
            requests_awaiting_swap_in: Vec::new(),
        }
    }

    /// Wire a live-mode (`ClockSource::Real`) offload engine onto this
    /// core's G1 manager. No-op when `args.kv_bytes_per_token` is
    /// unset. Caller must be inside an ambient tokio runtime.
    #[cfg(feature = "kvbm-offload")]
    pub(crate) async fn init_offload_live(&mut self) -> anyhow::Result<()> {
        crate::scheduler::init_kvbm_live(&self.args, &mut self.kv_manager).await?;
        Ok(())
    }

    /// Wire an offline-mode (`ClockSource::Virtual`) offload engine
    /// onto this core's G1 manager. No-op when
    /// `args.kv_bytes_per_token` is unset. Sync entry — owns the
    /// internal tokio runtime via `attach_runtime`.
    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn init_offload_offline(&mut self) -> anyhow::Result<()> {
        crate::scheduler::init_kvbm_offline(&self.args, &mut self.kv_manager)?;
        Ok(())
    }

    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        match self
            .apply_command(SchedulerCommand::Submit(request))
            .expect("ordinary request ID must be unique")
        {
            SchedulerCommandResult::Submitted(uuid) => uuid,
            _ => unreachable!("submit command must return a request ID"),
        }
    }

    pub(crate) fn apply_command(
        &mut self,
        command: SchedulerCommand,
    ) -> anyhow::Result<SchedulerCommandResult> {
        Ok(self.apply_command_effects(command, true)?.result)
    }

    pub(crate) fn apply_command_effects(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        self.apply_command_effects_at(command, allow_destination_admission, None)
    }

    pub(super) fn apply_command_effects_at(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
        reservation_now_ms: Option<f64>,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        match command {
            SchedulerCommand::Submit(mut request) => {
                let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
                request.uuid = Some(uuid);
                self.validate_request_id(uuid)?;
                Ok(SchedulerCommandEffects::new(
                    SchedulerCommandResult::Submitted(self.submit(request)?),
                ))
            }
            SchedulerCommand::CancelRequest { request_id } => {
                let result = if self.state.requests.contains_key(&request_id) {
                    self.drop_request(request_id);
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                if allow_destination_admission {
                    Ok(self.effects_after_capacity_change(result, reservation_now_ms))
                } else {
                    Ok(SchedulerCommandEffects::new(result))
                }
            }
            SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                mut request,
            } => {
                let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
                request.uuid = Some(uuid);
                self.validate_request_id(uuid)?;
                self.source_holds.register(uuid, handoff_id)?;
                let submitted = self
                    .submit(request)
                    .expect("prevalidated handoff request must submit");
                Ok(SchedulerCommandEffects::new(
                    SchedulerCommandResult::Submitted(submitted),
                ))
            }
            SchedulerCommand::ReleaseSource { handoff_id } => {
                let result = if self.release_source(handoff_id) {
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                Ok(self.effects_after_capacity_change(result, reservation_now_ms))
            }
            SchedulerCommand::CancelSource { handoff_id } => {
                let result = if self.cancel_source(handoff_id) {
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                Ok(self.effects_after_capacity_change(result, reservation_now_ms))
            }
            SchedulerCommand::ReserveDestination {
                handoff_id,
                mut request,
            } => {
                if !policy::supports_destination_reservation(self.args.scheduling_policy()) {
                    anyhow::bail!("destination reservation is not supported for TRT-LLM");
                }
                let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
                request.uuid = Some(uuid);
                self.validate_request_id(uuid)?;
                self.pending_destinations.validate(uuid, handoff_id)?;
                self.destination_holds.validate(uuid, handoff_id)?;
                if self
                    .active_destination_handoffs
                    .contains_handoff(handoff_id)
                {
                    anyhow::bail!("destination handoff {handoff_id:?} is already active");
                }
                let request = self.make_request_state(request, RequestStatus::WaitingForRemoteKv);
                if request.sequence.current_known_blocks() > self.args.num_gpu_blocks {
                    anyhow::bail!("destination prompt exceeds the KV pool capacity");
                }
                self.pending_destinations.insert(uuid, handoff_id, request);
                let mut effects =
                    SchedulerCommandEffects::new(SchedulerCommandResult::DestinationAccepted {
                        request_id: uuid,
                    });
                if allow_destination_admission {
                    effects
                        .lifecycle_events
                        .extend(self.retry_pending_destinations_at(reservation_now_ms));
                }
                Ok(effects)
            }
            SchedulerCommand::ActivateDestination { handoff_id } => {
                let Some((uuid, reservation)) = self.destination_holds.remove(handoff_id) else {
                    return Ok(SchedulerCommandEffects::new(SchedulerCommandResult::Noop));
                };
                let active_before = self.kv_manager.num_active_blocks();
                let request = reservation.activate(&mut self.kv_manager);
                self.active_destination_handoffs.insert(handoff_id, uuid);
                self.state.insert_waiting(uuid, request);
                if self.kv_manager.num_active_blocks() < active_before {
                    self.bump_capacity_generation();
                }
                Ok(self.effects_after_capacity_change(
                    SchedulerCommandResult::Applied,
                    reservation_now_ms,
                ))
            }
            SchedulerCommand::CancelDestination { handoff_id } => {
                if self.pending_destinations.remove(handoff_id).is_some() {
                    self.bump_capacity_generation();
                    return Ok(self.effects_after_capacity_change(
                        SchedulerCommandResult::Applied,
                        reservation_now_ms,
                    ));
                }
                if let Some((_, reservation)) = self.destination_holds.remove(handoff_id) {
                    reservation.cancel(&mut self.kv_manager);
                    self.bump_capacity_generation();
                    return Ok(self.effects_after_capacity_change(
                        SchedulerCommandResult::Applied,
                        reservation_now_ms,
                    ));
                }
                let Some(request_id) = self.active_destination_handoffs.remove_handoff(handoff_id)
                else {
                    return Ok(SchedulerCommandEffects::new(SchedulerCommandResult::Noop));
                };
                self.drop_request(request_id);
                Ok(self.effects_after_capacity_change(
                    SchedulerCommandResult::Applied,
                    reservation_now_ms,
                ))
            }
        }
    }

    fn effects_after_capacity_change(
        &mut self,
        result: SchedulerCommandResult,
        reservation_now_ms: Option<f64>,
    ) -> SchedulerCommandEffects {
        let mut effects = SchedulerCommandEffects::new(result);
        if result == SchedulerCommandResult::Applied {
            effects
                .lifecycle_events
                .extend(self.retry_pending_destinations_at(reservation_now_ms));
        }
        effects
    }

    pub(crate) fn retry_pending_destinations(&mut self) -> Vec<SchedulerLifecycleEvent> {
        self.retry_pending_destinations_at(None)
    }

    pub(super) fn retry_pending_destinations_at(
        &mut self,
        reservation_now_ms: Option<f64>,
    ) -> Vec<SchedulerLifecycleEvent> {
        let generation = self.capacity_generation;
        let max_num_running = self.args.max_num_seqs.unwrap_or(usize::MAX);
        if self.state.running_members.len() >= max_num_running {
            self.pending_destinations.mark_front_attempted(generation);
            return Vec::new();
        }

        let Some((_, request_id, request)) = self.pending_destinations.front_due_mut(generation)
        else {
            return Vec::new();
        };
        if let Some(dependency) = request.offload_dependency {
            request.offload_dependency = self.kv_manager.refresh_offload_dependency(dependency);
            if request.offload_dependency.is_some() {
                return Vec::new();
            }
        }
        #[cfg(test)]
        {
            self.destination_reservation_attempts += 1;
        }
        let reservation = self.kv_manager.reserve_destination_at(
            request_id,
            &request.sequence,
            reservation_now_ms,
        );
        let kv = match reservation {
            G1Acquire::Ready(kv) => kv,
            G1Acquire::BlockedOnOffload {
                offload_id,
                deadline_ms,
            } => {
                request.offload_dependency = Some(OffloadDependency {
                    offload_id,
                    deadline_ms,
                });
                self.pending_destinations.mark_front_attempted(generation);
                return Vec::new();
            }
            G1Acquire::CapacityExhausted => {
                self.pending_destinations.mark_front_attempted(generation);
                return Vec::new();
            }
            G1Acquire::RetryNow { .. } => {
                panic!("destination reservation must consume bounded RetryNow internally")
            }
        };
        self.pending_destinations.mark_front_attempted(generation);
        let transferable_prompt_tokens = kv.transferable_prompt_tokens(self.args.block_size);
        let (handoff_id, request_id, request) = self
            .pending_destinations
            .pop_front()
            .expect("attempted pending destination must remain at the head");
        self.destination_holds
            .insert(request_id, handoff_id, ReservedVllmDecode { request, kv });
        vec![SchedulerLifecycleEvent::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        }]
    }

    fn validate_request_id(&self, uuid: Uuid) -> anyhow::Result<()> {
        #[cfg(feature = "kvbm-offload")]
        let awaiting_swap_in = self
            .requests_awaiting_swap_in
            .iter()
            .any(|request| request.uuid == uuid);
        #[cfg(not(feature = "kvbm-offload"))]
        let awaiting_swap_in = false;

        if self.state.requests.contains_key(&uuid)
            || self.source_holds.contains_request(uuid)
            || self.pending_destinations.contains_request(uuid)
            || self.destination_holds.contains_request(uuid)
            || self.active_destination_handoffs.contains_request(uuid)
            || awaiting_swap_in
        {
            anyhow::bail!("request {uuid} is already active");
        }
        Ok(())
    }

    fn submit(&mut self, mut request: DirectRequest) -> anyhow::Result<Uuid> {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        request.uuid = Some(uuid);
        if self.state.requests.contains_key(&uuid) {
            anyhow::bail!("request {uuid} is already active");
        }
        let request = self.make_request_state(request, RequestStatus::Waiting);
        self.state.insert_waiting(uuid, request);
        if let Some(request) = self.state.requests.get(&uuid) {
            request.debug_assert_progress(uuid);
        }
        Ok(uuid)
    }

    fn make_request_state(
        &self,
        request: DirectRequest,
        status: RequestStatus,
    ) -> VllmRequestState {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        let mut max_output_tokens = request.max_output_tokens;
        let planned_output_ids = request.output_token_ids;
        if let Some(planned_output_ids) = planned_output_ids.as_ref()
            && planned_output_ids.len() != max_output_tokens
        {
            tracing::warn!(
                %uuid,
                requested = max_output_tokens,
                planned = planned_output_ids.len(),
                "planned output token count differs from max_output_tokens; using planned count"
            );
            max_output_tokens = planned_output_ids.len();
        }
        if let Some(clamped) = policy::normalize_max_output_tokens(
            self.args.scheduling_policy(),
            request.tokens.len(),
            max_output_tokens,
            self.args.num_gpu_blocks,
            self.args.block_size,
        ) {
            if clamped != max_output_tokens {
                tracing::warn!(%uuid, requested = max_output_tokens, clamped,
                    "clamped TRT-LLM max_output_tokens to KV-pool capacity");
            }
            max_output_tokens = clamped;
        }
        // The `None` case (a TRT-LLM prompt alone leaves no decode room) is
        // unchanged here. The waiting-admission policy owns terminal rejection
        // because that path can emit the lifecycle signal.
        let sequence = ActiveSequence::new_with_planned_output_ids(
            request.tokens,
            max_output_tokens,
            Some(self.args.block_size),
            self.args.enable_prefix_caching,
            self.args.zmq_kv_events_port.is_some(),
            planned_output_ids,
        );
        VllmRequestState {
            sequence,
            status,
            num_computed_tokens: 0,
            num_preemptions: 0,
            offload_dependency: None,
        }
    }

    fn release_source(&mut self, handoff_id: HandoffId) -> bool {
        match self.source_holds.remove(handoff_id) {
            RemovedSource::Held(payload) => {
                self.cleanup_completed_prefill(payload);
                self.bump_capacity_generation();
                true
            }
            RemovedSource::Pending { .. } => true,
            RemovedSource::Missing => false,
        }
    }

    fn cancel_source(&mut self, handoff_id: HandoffId) -> bool {
        match self.source_holds.remove(handoff_id) {
            RemovedSource::Held(payload) => {
                self.cleanup_completed_prefill(payload);
                self.bump_capacity_generation();
                true
            }
            RemovedSource::Pending { request_id } => {
                self.drop_request(request_id);
                true
            }
            RemovedSource::Missing => false,
        }
    }

    fn complete_source(&mut self, uuid: Uuid, deferred_deref: Vec<MoveBlock>) {
        let transfer_timing = self.state.requests.get(&uuid).map(|request| {
            prefill_handoff_transfer_timing(
                request.sequence.num_input_tokens(),
                self.args.kv_transfer_bandwidth,
                self.args.kv_bytes_per_token,
                self.args.kv_transfer_timing_mode,
            )
        });
        let request = self
            .state
            .take_completed(&uuid)
            .expect("completed request must remain scheduler-owned");
        let payload = HeldVllmPrefill {
            request_id: uuid,
            request,
            deferred_deref,
        };
        match self.source_holds.complete_source(uuid, payload) {
            SourceCompletion::Release(payload) => {
                self.cleanup_completed_prefill(payload);
            }
            SourceCompletion::Held { handoff_id } => {
                self.lifecycle_events
                    .push(SchedulerLifecycleEvent::SourceHeld {
                        handoff_id,
                        request_id: uuid,
                        transfer_timing: transfer_timing
                            .expect("completed source request must retain transfer timing"),
                    });
            }
        }
        self.active_destination_handoffs.remove_request(uuid);
        // Completion always releases a vLLM runnable slot, even when source KV
        // remains held for handoff.
        self.bump_capacity_generation();
    }

    fn cleanup_completed_prefill(&mut self, payload: HeldVllmPrefill) {
        let HeldVllmPrefill {
            request_id,
            request,
            deferred_deref,
        } = payload;
        for signal in deferred_deref {
            assert!(
                matches!(
                    self.kv_manager.process_for_request(request_id, &signal, 0),
                    G1Acquire::Ready(_)
                ),
                "terminal prefill cleanup must be infallible"
            );
        }
        drop(request);
    }

    #[cfg(test)]
    pub(crate) fn source_is_held(&self, handoff_id: HandoffId) -> bool {
        self.source_holds.is_held(handoff_id)
    }

    #[cfg(test)]
    pub(crate) fn source_is_registered(&self, handoff_id: HandoffId) -> bool {
        self.source_holds.is_registered(handoff_id)
    }

    #[cfg(test)]
    pub(crate) fn destination_reservation_attempts(&self) -> usize {
        self.destination_reservation_attempts
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn is_drained(&self) -> bool {
        if !self.is_empty()
            || !self.source_holds.is_empty()
            || !self.pending_destinations.is_empty()
            || !self.destination_holds.is_empty()
            || !self.active_destination_handoffs.is_empty()
        {
            return false;
        }
        #[cfg(feature = "kvbm-offload")]
        {
            self.requests_awaiting_swap_in.is_empty()
                && self.kv_manager.earliest_offload_deadline().is_none()
        }
        #[cfg(not(feature = "kvbm-offload"))]
        {
            true
        }
    }

    #[cfg(test)]
    pub(crate) fn destination_is_held(&self, handoff_id: HandoffId) -> bool {
        self.destination_holds.contains(handoff_id)
            || self.pending_destinations.contains_handoff(handoff_id)
    }

    #[cfg(test)]
    pub(crate) fn destination_block_count(&self, handoff_id: HandoffId) -> usize {
        self.destination_holds
            .get(handoff_id)
            .map(|reservation| reservation.kv.len())
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn request_block_count(&self, uuid: Uuid) -> usize {
        self.state
            .requests
            .get(&uuid)
            .map(|request| self.kv_manager.request_block_count(uuid, &request.sequence))
            .unwrap_or(0)
    }

    pub(crate) fn drain_kv_events(&self) -> Vec<dynamo_kv_router::protocols::RouterEvent> {
        self.kv_event_buffer
            .as_ref()
            .map(CapturedRouterEventBuffer::drain)
            .unwrap_or_default()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.state.requests.len()
    }

    fn bump_capacity_generation(&mut self) {
        self.capacity_generation = self
            .capacity_generation
            .checked_add(1)
            .expect("destination capacity generation overflow");
    }

    /// Read-only view of the scheduler state for policy tests that assert on
    /// queue membership.
    #[cfg(test)]
    pub(crate) fn state(&self) -> &SchedulerState {
        &self.state
    }

    pub(crate) fn mocker_metrics(&self) -> MockerMetrics {
        let preactivation_destinations =
            self.pending_destinations.len() + self.destination_holds.len();
        MockerMetrics::from_parts(
            self.dp_rank,
            self.kv_manager.num_active_blocks() as u64,
            self.args.num_gpu_blocks as u64,
            self.state.running_members.len() as u64,
            (self.state.waiting_members.len() + preactivation_destinations) as u64,
            self.state.preemptions_total,
            0,
            0,
        )
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.execute_pass_internal(Some(collector), now_ms, None)
    }

    pub(crate) fn execute_hidden_pass(&mut self, now_ms: f64) -> EnginePassResult {
        self.execute_pass_internal(None, now_ms, None)
    }

    /// Drive the offload engine forward to `now_ms` and promote any
    /// parked swap-ins whose transfers just completed.
    #[cfg(feature = "kvbm-offload")]
    fn tick_and_promote_swap_ins(&mut self, now_ms: f64) {
        let active_before = self.kv_manager.num_active_blocks();
        self.kv_manager.tick_offload_engine(now_ms);
        let awaiting = std::mem::take(&mut self.requests_awaiting_swap_in);
        let mut completed = Vec::new();
        let mut pending = Vec::with_capacity(awaiting.len());

        for aws in awaiting {
            match aws.handle.terminal() {
                SwapInTerminal::Pending => pending.push(aws),
                SwapInTerminal::Completed => completed.push(aws),
                SwapInTerminal::Cancelled => {}
                SwapInTerminal::Failed(context) => {
                    panic!("mocker swap-in failed for request {}: {context}", aws.uuid)
                }
            }
        }

        self.requests_awaiting_swap_in = pending;
        // Completed swap-ins are ready to run immediately: put them back at
        // the front so unrelated cold requests cannot evict the freshly
        // onboarded inactive blocks first. Iterate in reverse because each
        // completion prepends to the queue; this preserves completion order.
        for aws in completed.into_iter().rev() {
            self.complete_swap_in(aws);
        }
        if self.kv_manager.num_active_blocks() < active_before {
            self.bump_capacity_generation();
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn tick_offload_only(
        &mut self,
        now_ms: f64,
    ) -> crate::scheduler::OffloadTickEffects {
        self.tick_offload(now_ms, true)
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn tick_offload_transport_only(
        &mut self,
        now_ms: f64,
    ) -> crate::scheduler::OffloadTickEffects {
        self.tick_offload(now_ms, false)
    }

    #[cfg(feature = "kvbm-offload")]
    fn tick_offload(
        &mut self,
        now_ms: f64,
        allow_destination_admission: bool,
    ) -> crate::scheduler::OffloadTickEffects {
        self.tick_and_promote_swap_ins(now_ms);
        let lifecycle_events = if allow_destination_admission {
            self.retry_pending_destinations()
        } else {
            Vec::new()
        };
        let kv_events = self
            .kv_event_buffer
            .as_ref()
            .map(CapturedRouterEventBuffer::drain)
            .unwrap_or_default();
        crate::scheduler::OffloadTickEffects {
            kv_events,
            lifecycle_events,
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn earliest_offload_deadline(&self) -> Option<f64> {
        self.kv_manager.earliest_offload_deadline()
    }

    /// Register the onboard'd PLHs into G1 inactive (so the request's
    /// next `process_use` sees `InactiveHit`) and re-queue the request at
    /// the front for admission. The swap-in covers
    /// `[skip_blocks .. skip_blocks + count]` of the request's block
    /// sequence — we skip the G1-cached prefix the request already had
    /// and register only the uncached-remainder blocks that the engine
    /// actually onboarded from G2. `register_completed_swap_in` consumes
    /// the coordinator lease, registers its retained G1 destinations, and
    /// releases the remaining swap-in resources.
    #[cfg(feature = "kvbm-offload")]
    fn complete_swap_in(&mut self, aws: AwaitingSwapIn) {
        let count = aws.handle.block_count();
        let skip = aws.skip_blocks;
        let entries: Vec<_> = {
            let request = self
                .state
                .requests
                .get(&aws.uuid)
                .expect("swap-in completed for known request");
            let unique = request.sequence.unique_blocks();
            let plhs = request.sequence.positional_lineage_hashes();
            let local_hashes = request.sequence.block_hashes();
            let token_ids = request.sequence.block_token_ids();
            unique
                .iter()
                .zip(plhs.iter())
                .enumerate()
                .skip(skip)
                .take(count)
                .filter_map(|(idx, (block, plh))| match block {
                    UniqueBlock::FullBlock(seq_hash) => Some(SwapInRegistrationBlock {
                        seq_hash: *seq_hash,
                        plh: *plh,
                        local_hash: local_hashes.get(idx).copied(),
                        token_ids: token_ids.get(idx).cloned(),
                    }),
                    UniqueBlock::PartialBlock(_) => None,
                })
                .collect()
        };
        let parent_hash = if skip == 0 {
            None
        } else {
            let request = self
                .state
                .requests
                .get(&aws.uuid)
                .expect("swap-in completed for known request");
            match request.sequence.unique_blocks().get(skip - 1) {
                Some(UniqueBlock::FullBlock(seq_hash)) => Some(*seq_hash),
                _ => None,
            }
        };
        let entries_len = entries.len();
        let outcome =
            self.kv_manager
                .register_completed_swap_in(aws.handle.id(), entries, parent_hash);
        debug_assert_eq!(
            outcome.consumed_entries, entries_len,
            "reserved destination slots should cover every swapped-in block"
        );
        self.state.prepend_waiting(aws.uuid);
    }

    /// Admission-side hook: park a request on a G2 swap-in covering its
    /// **uncached remainder prefix** (the run of full-block PLHs after
    /// whatever G1 already has cached). Returns `true` when parked.
    ///
    /// The gate is deliberately wider than "cold only": a request whose
    /// first N blocks hit G1 can still benefit from a G2 onboard of
    /// blocks N, N+1, ... — that's exactly the "evicted-then-recalled"
    /// pattern in workloads with prefix sharing (mooncake_trace etc.).
    /// By passing only the uncached-suffix PLHs to the engine, we avoid
    /// redundantly re-onboarding the G1-cached prefix.
    #[cfg(feature = "kvbm-offload")]
    fn try_park_for_swap_in(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        g1_cached_tokens: usize,
    ) -> SwapInAdmissionAttempt {
        use crate::kv_manager::kvbm_backend::BatchSwapInOutcome;
        if !self.kv_manager.has_offload_engine() {
            return SwapInAdmissionAttempt::NoHit;
        }
        let request = self
            .state
            .requests
            .get(&uuid)
            .expect("try_park_for_swap_in: uuid in waiting queue but missing from state");
        if !matches!(request.status, RequestStatus::Waiting) {
            return SwapInAdmissionAttempt::NoHit;
        }
        // Lower-tier lookup starts after the physical G1 prefix. MTP may
        // recompute the final matched block, but that changes compute
        // accounting rather than which blocks are resident in G1.
        let block_size = request.sequence.block_size();
        let skip_blocks = g1_cached_tokens / block_size;
        let plhs = request.sequence.positional_lineage_hashes();
        tracing::trace!(
            %uuid,
            now_ms,
            cached_tokens = g1_cached_tokens,
            skip_blocks,
            plhs_len = plhs.len(),
            "kvbm-offload: swap-in admission probe"
        );
        if skip_blocks >= plhs.len() {
            tracing::trace!(
                %uuid,
                now_ms,
                skip_blocks,
                plhs_len = plhs.len(),
                "kvbm-offload: swap-in skipped; G1 prefix covers request"
            );
            return SwapInAdmissionAttempt::NoHit;
        }
        let remaining_plhs = &plhs[skip_blocks..];
        if remaining_plhs.is_empty() {
            return SwapInAdmissionAttempt::NoHit;
        }
        let prefix_pins = match self.kv_manager.try_pin_g1_prefix(&plhs[..skip_blocks]) {
            Some(pins) => pins,
            None => {
                tracing::trace!(
                    %uuid,
                    now_ms,
                    skip_blocks,
                    "kvbm-offload: swap-in skipped; failed to pin G1 prefix"
                );
                return SwapInAdmissionAttempt::NoHit;
            }
        };
        let handle =
            match self
                .kv_manager
                .try_batch_swap_in(remaining_plhs, prefix_pins, Some(now_ms))
            {
                BatchSwapInOutcome::Scheduled { handle } => {
                    tracing::debug!(
                        %uuid,
                        now_ms,
                        skip_blocks,
                        remaining_blocks = remaining_plhs.len(),
                        "kvbm-offload: swap-in admission parked"
                    );
                    handle
                }
                BatchSwapInOutcome::BlockedOnG1Offload(dependency) => {
                    tracing::debug!(
                        %uuid,
                        now_ms,
                        offload_id = ?dependency.offload_id,
                        deadline_ms = ?dependency.deadline_ms,
                        skip_blocks,
                        remaining_blocks = remaining_plhs.len(),
                        "kvbm-offload: swap-in blocked on G1 offload"
                    );
                    let request = self
                        .state
                        .requests
                        .get_mut(&uuid)
                        .expect("swap-in dependency request must remain active");
                    request.offload_dependency = Some(dependency);
                    return SwapInAdmissionAttempt::BlockedOnG1Offload;
                }
                BatchSwapInOutcome::NoHits => {
                    tracing::trace!(
                        %uuid,
                        now_ms,
                        skip_blocks,
                        remaining_blocks = remaining_plhs.len(),
                        "kvbm-offload: swap-in lower-tier miss"
                    );
                    return SwapInAdmissionAttempt::NoHit;
                }
            };
        self.state.remove_from_waiting(uuid);
        self.requests_awaiting_swap_in.push(AwaitingSwapIn {
            uuid,
            handle,
            skip_blocks,
        });
        SwapInAdmissionAttempt::Parked
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn execute_pass_internal(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        now_ms: f64,
        admission_tx: Option<&mpsc::UnboundedSender<AdmissionEvent>>,
    ) -> EnginePassResult {
        let requests_before = self.state.requests.len();
        #[cfg(feature = "kvbm-offload")]
        self.tick_and_promote_swap_ins(now_ms);
        self.state.compact_running();
        let mut token_budget = self.args.max_num_batched_tokens.unwrap_or(usize::MAX);
        let mut scheduled = FxHashMap::default();
        scheduled.reserve(
            self.state
                .running
                .len()
                .saturating_add(self.state.waiting.len().min(16)),
        );
        let mut batch_count = 0usize;
        let mut batch_total_isl = 0usize;
        let mut batch_total_prefix = 0usize;
        let mut admissions = Vec::with_capacity(self.state.waiting.len().min(16));
        let mut preempted_any = false;

        let mut req_index = 0usize;
        while req_index < self.state.running.len() && token_budget > 0 {
            let uuid = self.state.running[req_index];
            match self.schedule_request(
                uuid,
                false,
                None,
                &mut token_budget,
                &mut scheduled,
                &mut batch_count,
                &mut batch_total_isl,
                &mut batch_total_prefix,
                &mut preempted_any,
            ) {
                ScheduleOutcome::Scheduled { admission, .. } => {
                    if let Some(admission) = admission {
                        if let Some(collector) = collector.as_deref_mut() {
                            collector.on_admit(
                                admission.uuid,
                                now_ms,
                                admission.reused_input_tokens,
                            );
                        }
                        if let Some(admission_tx) = admission_tx {
                            let _ = admission_tx.send(admission.clone());
                        }
                        admissions.push(admission);
                    }
                    req_index += 1;
                }
                ScheduleOutcome::DependencyBlocked => req_index += 1,
                ScheduleOutcome::Blocked => break,
                ScheduleOutcome::CurrentPreempted => {}
            }
        }

        let max_num_running = self.args.max_num_seqs.unwrap_or(usize::MAX);
        let scheduling_policy = self.args.scheduling_policy();
        let admission = AdmissionInvariant::new(self.pending_destinations.has_pending());
        let mut rejected_uuids: Vec<Uuid> = Vec::new();
        while !preempted_any && self.state.running.len() < max_num_running {
            let prefer_materialized = matches!(
                admission.stage_for(false),
                AdmissionStage::PendingDestinationHead
            );
            let Some(uuid) = self.state.next_waiting_uuid(prefer_materialized) else {
                break;
            };
            if self.refresh_request_offload_dependency(uuid).is_some() {
                break;
            }
            let decision = {
                let request = self
                    .state
                    .requests
                    .get(&uuid)
                    .expect("waiting request missing from state");
                let running_seqs = self
                    .state
                    .running
                    .iter()
                    .filter_map(|running_uuid| self.state.requests.get(running_uuid))
                    .map(|request| &request.sequence);
                if policy::should_reject_for_model_len(
                    scheduling_policy,
                    &request.sequence,
                    self.args.max_model_len,
                ) {
                    AdmissionDecision::Reject
                } else {
                    let prompt_is_prebuilt = request.prompt_is_prebuilt();
                    match admission.stage_for(prompt_is_prebuilt) {
                        AdmissionStage::Materialized => AdmissionDecision::Admit {
                            prefill_cost: PrefillCost {
                                new_blocks: 0,
                                new_tokens: 0,
                                cached_tokens: request.sequence.num_input_tokens(),
                                active_cached_tokens: request.sequence.num_input_tokens(),
                            },
                            g1_cached_tokens: request.sequence.num_input_tokens(),
                        },
                        AdmissionStage::PendingDestinationHead => break,
                        AdmissionStage::FreshKv => {
                            let is_fresh = request.status == RequestStatus::Waiting;
                            policy::decide_waiting_admission(
                                policy::WaitingAdmissionConfig {
                                    policy: scheduling_policy,
                                    num_gpu_blocks: self.args.num_gpu_blocks,
                                    block_size: self.args.block_size,
                                    mtp_enabled: self.args.aic_nextn.is_some(),
                                },
                                &request.sequence,
                                is_fresh,
                                running_seqs,
                                &self.kv_manager,
                            )
                        }
                    }
                }
            };
            let (prefill_cost, _g1_cached_tokens) = match decision {
                AdmissionDecision::Admit {
                    prefill_cost,
                    g1_cached_tokens,
                } => (prefill_cost, g1_cached_tokens),
                AdmissionDecision::Wait => {
                    break;
                }
                AdmissionDecision::Reject => {
                    tracing::warn!(
                        %uuid,
                        ?scheduling_policy,
                        prompt_tokens = self
                            .state
                            .requests
                            .get(&uuid)
                            .map(|request| request.sequence.num_input_tokens()),
                        max_model_len = self.args.max_model_len,
                        num_gpu_blocks = self.args.num_gpu_blocks,
                        "rejecting request that exceeds a worker admission limit"
                    );
                    rejected_uuids.push(uuid);
                    self.drop_request(uuid);
                    continue;
                }
            };
            #[cfg(feature = "kvbm-offload")]
            match self.try_park_for_swap_in(uuid, now_ms, _g1_cached_tokens) {
                SwapInAdmissionAttempt::Parked => continue,
                SwapInAdmissionAttempt::BlockedOnG1Offload => break,
                SwapInAdmissionAttempt::NoHit => {}
            }
            match self.schedule_request(
                uuid,
                true,
                Some(&prefill_cost),
                &mut token_budget,
                &mut scheduled,
                &mut batch_count,
                &mut batch_total_isl,
                &mut batch_total_prefix,
                &mut preempted_any,
            ) {
                ScheduleOutcome::Scheduled {
                    admission,
                    tokens_used,
                } => {
                    if let Some(admission) = admission {
                        if let Some(collector) = collector.as_deref_mut() {
                            collector.on_admit(
                                admission.uuid,
                                now_ms,
                                admission.reused_input_tokens,
                            );
                        }
                        if let Some(admission_tx) = admission_tx {
                            let _ = admission_tx.send(admission.clone());
                        }
                        admissions.push(admission);
                    }
                    if tokens_used == 0 && token_budget == 0 {
                        break;
                    }
                }
                ScheduleOutcome::Blocked
                | ScheduleOutcome::DependencyBlocked
                | ScheduleOutcome::CurrentPreempted => break,
            }
        }

        let prefill_time =
            predict_prefill_duration(batch_count, batch_total_isl, batch_total_prefix, &self.args);
        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let (decode_time, mut output_signals) = self.emit_ready_tokens(collector, decode_start_ms);
        // Emit the terminal signals for the requests the gate rejected above
        // (see the gate comment for why this can't be done inline).
        for uuid in rejected_uuids {
            output_signals.push(OutputSignal {
                uuid,
                token_id: None,
                completed: true,
                rejected: true,
                handoff_delay_ms: None,
            });
        }
        #[cfg_attr(not(feature = "kvbm-offload"), allow(unused_mut))]
        let mut end_ms = decode_start_ms + decode_time.as_secs_f64() * 1000.0;

        // Stall-advance for pending offload work: if the pass did no
        // model work but either (a) requests are parked on G2→G1 swap-ins
        // or (b) G1 source slots are quarantined behind a G1→G2 offload,
        // advance virtual time to the earliest offload-engine deadline.
        // Without this the offline replay can spin forever at the same
        // `current_time_ms`: `execute_pass` returns `end_ms == now_ms`,
        // but the worker still has blocked requests or pending source-slot
        // releases, so `is_done()` never triggers.
        #[cfg(feature = "kvbm-offload")]
        if end_ms <= now_ms
            && let Some(deadline) = self.kv_manager.earliest_offload_deadline()
        {
            end_ms = deadline.max(now_ms);
        }

        let fpm = self.compute_fpm(&scheduled, (end_ms - now_ms) / 1000.0);
        let (accept_length_output_tokens, accept_length_decode_forwards) =
            accept_length_sample(&output_signals);
        self.state.debug_assert_invariants();
        EnginePassResult {
            end_ms,
            completed_requests: requests_before.saturating_sub(self.state.requests.len()),
            output_signals,
            admissions,
            lifecycle_events: std::mem::take(&mut self.lifecycle_events),
            mocker_metrics: self.mocker_metrics(),
            router_event_visibility: RouterEventVisibility::PassStart,
            kv_events: self
                .kv_event_buffer
                .as_ref()
                .map(CapturedRouterEventBuffer::drain)
                .unwrap_or_default(),
            fpm: Some(fpm),
            accept_length_output_tokens,
            accept_length_decode_forwards,
        }
    }

    pub(super) fn drop_request(&mut self, uuid: Uuid) {
        let active_blocks_before = self.kv_manager.num_active_blocks();
        #[cfg(feature = "kvbm-offload")]
        {
            let awaiting = std::mem::take(&mut self.requests_awaiting_swap_in);
            for request in awaiting {
                if request.uuid == uuid {
                    assert!(
                        self.kv_manager.cancel_swap_in(request.handle.id()),
                        "parked swap-in must retain a coordinator lease"
                    );
                } else {
                    self.requests_awaiting_swap_in.push(request);
                }
            }
        }

        let Some(request) = self.state.requests.get(&uuid) else {
            if self.kv_manager.num_active_blocks() < active_blocks_before {
                self.bump_capacity_generation();
            }
            return;
        };
        let capacity_improved = request.sequence.num_allocated_tokens() > 0
            || self.state.running_members.contains(&uuid)
            || self.kv_manager.num_active_blocks() < active_blocks_before;
        for signal in request.sequence.free_signal() {
            assert!(
                matches!(
                    self.kv_manager.process_for_request(uuid, &signal, 0),
                    G1Acquire::Ready(_)
                ),
                "request drop cleanup must be infallible"
            );
        }
        self.source_holds.remove_request(uuid);
        self.active_destination_handoffs.remove_request(uuid);
        self.state.complete(&uuid);
        if capacity_improved {
            self.bump_capacity_generation();
        }
    }

    /// Preempt a running request under the active scheduling policy.
    ///
    /// Under vLLM semantics this evicts a running request on KV pressure. Under
    /// TRT-LLM `GUARANTEED_NO_EVICT` preemption must never happen — the capacity
    /// gate reserves blocks for every admitted request up front — so reaching
    /// this path is reported as a hard error and nothing is evicted.
    pub(super) fn policy_preempt(&mut self) -> Option<PreemptedRequest> {
        if !policy::allows_preemption(self.args.scheduling_policy()) {
            policy::report_no_preemption_violation();
            return None;
        }
        let running_len = self.state.running.len();
        let mut selected = None;
        for offset in 0..running_len {
            let index = match self.args.preemption_mode {
                PreemptionMode::Fifo => offset,
                PreemptionMode::Lifo => running_len - offset - 1,
            };
            let Some(uuid) = self.state.running.get(index).copied() else {
                continue;
            };
            let is_running = self.state.running_members.contains(&uuid)
                && self
                    .state
                    .requests
                    .get(&uuid)
                    .is_some_and(|request| request.status == RequestStatus::Running);
            if !is_running || self.refresh_request_offload_dependency(uuid).is_some() {
                continue;
            }
            selected = Some(uuid);
            break;
        }
        let preempted = selected.and_then(|uuid| self.state.preempt_uuid(uuid));
        if let Some(preempted) = preempted.as_ref() {
            self.bump_capacity_generation();
            tracing::debug!(
                worker_id = self.dp_rank,
                request_id = %preempted.uuid,
                preemptions_total = self.state.preemptions_total,
                "vLLM scheduler preempted and requeued request"
            );
        }
        preempted
    }

    fn refresh_request_offload_dependency(&mut self, uuid: Uuid) -> Option<OffloadDependency> {
        let dependency = self
            .state
            .requests
            .get(&uuid)
            .and_then(|request| request.offload_dependency)?;
        let refreshed = self.kv_manager.refresh_offload_dependency(dependency);
        if let Some(request) = self.state.requests.get_mut(&uuid) {
            request.offload_dependency = refreshed;
        }
        refreshed
    }

    /// Compute a forward pass metrics snapshot from the just-completed pass.
    ///
    /// `scheduled` contains the work items that were scheduled in this iteration.
    /// Per-request metadata (prompt_len, sequence_len) is captured in `ScheduledWork`
    /// at schedule time, so this method does not depend on `self.state.requests` for
    /// scheduled requests — completed requests may have already been removed.
    /// Queue metrics are derived from `self.state.waiting` at the moment of the call.
    #[cfg_attr(feature = "profile", inline(never))]
    fn compute_fpm(
        &self,
        scheduled: &FxHashMap<Uuid, ScheduledWork>,
        wall_time_secs: f64,
    ) -> ForwardPassSnapshot {
        let scheduled_prefills = scheduled.values().filter_map(|work| {
            (work.prompt_tokens > 0).then_some((
                work.prompt_len as u64,
                work.prefix_tokens as u64,
                work.total_tokens as u64,
            ))
        });

        let scheduled_decodes = scheduled.values().filter_map(|work| {
            (work.prompt_tokens == 0 && !work.terminal_after_schedule)
                .then_some(work.sequence_len as u64)
        });

        let queued_prefills = self.state.waiting.iter().filter_map(|uuid| {
            let request = self.state.requests.get(uuid)?;
            (matches!(request.status, RequestStatus::Waiting)
                && !self.active_destination_handoffs.contains_request(*uuid))
            .then_some(request.sequence.num_input_tokens() as u64)
        });

        let ordinary_queued_decodes = self.state.waiting.iter().filter_map(|uuid| {
            let request = self.state.requests.get(uuid)?;
            if self.active_destination_handoffs.contains_request(*uuid) {
                return Some(request.sequence.num_input_tokens() as u64);
            }
            matches!(request.status, RequestStatus::Preempted).then_some(
                (request.sequence.num_input_tokens() + request.sequence.generated_tokens()) as u64,
            )
        });
        let preactivation_decodes = self
            .pending_destinations
            .payloads()
            .map(|request| request.sequence.num_input_tokens() as u64)
            .chain(
                self.destination_holds
                    .payloads()
                    .map(|reservation| reservation.request.sequence.num_input_tokens() as u64),
            );
        let queued_decodes = ordinary_queued_decodes.chain(preactivation_decodes);

        build_fpm_snapshot(
            scheduled_prefills,
            scheduled_decodes,
            queued_prefills,
            queued_decodes,
            wall_time_secs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(feature = "profile", inline(never))]
    fn schedule_request(
        &mut self,
        uuid: Uuid,
        from_waiting: bool,
        prefill_cost: Option<&PrefillCost>,
        token_budget: &mut usize,
        scheduled: &mut FxHashMap<Uuid, ScheduledWork>,
        batch_count: &mut usize,
        batch_total_isl: &mut usize,
        batch_total_prefix: &mut usize,
        preempted_any: &mut bool,
    ) -> ScheduleOutcome {
        let request = self
            .state
            .requests
            .get(&uuid)
            .unwrap_or_else(|| panic!("schedule_request: {uuid} missing from state.requests"));
        request.debug_assert_invariants(uuid);
        let cached_prefix_tokens = if request.num_computed_tokens == 0 {
            prefill_cost
                .map(|cost| cost.cached_tokens)
                .unwrap_or_else(|| {
                    policy::apply_prefix_recompute(
                        self.args.scheduling_policy(),
                        request.sequence.len(),
                        self.args.block_size,
                        self.args.aic_nextn.is_some(),
                        !policy::generation_complete(&request.sequence, self.args.max_model_len),
                        self.kv_manager.get_prefill_cost(&request.sequence),
                    )
                    .cached_tokens
                })
        } else {
            0
        };
        let effective_computed_before = request.num_computed_tokens + cached_prefix_tokens;
        let prompt_len = request.sequence.num_input_tokens();
        let prompt_before = effective_computed_before.min(prompt_len);
        let remaining_known_tokens = request
            .sequence
            .len()
            .saturating_sub(effective_computed_before);
        let prompt_remaining = prompt_len.saturating_sub(prompt_before);
        if prompt_remaining > 0
            && !self.args.enable_chunked_prefill
            && prompt_remaining > *token_budget
        {
            return ScheduleOutcome::Blocked;
        }

        let desired_tokens = remaining_known_tokens.min(*token_budget);
        if desired_tokens == 0 && remaining_known_tokens > 0 {
            return ScheduleOutcome::Blocked;
        }

        let desired_computed_after = effective_computed_before + desired_tokens;
        let mut actual_computed_after = desired_computed_after;

        if self.refresh_request_offload_dependency(uuid).is_some() {
            return ScheduleOutcome::DependencyBlocked;
        }

        loop {
            let allocation = {
                let request = self.state.requests.get_mut(&uuid).unwrap_or_else(|| {
                    panic!("schedule_request: {uuid} removed mid-pass (alloc prep)")
                });
                let allocation_target = desired_computed_after;
                let prev_allocated_tokens = request.sequence.num_allocated_tokens();
                if allocation_target <= prev_allocated_tokens {
                    request.num_computed_tokens = actual_computed_after;
                    None
                } else {
                    let maybe_signal = request.sequence.prepare_allocation(allocation_target);
                    Some((allocation_target, maybe_signal))
                }
            };
            let Some((allocation_target, maybe_signal)) = allocation else {
                break;
            };
            let Some(signal) = maybe_signal else {
                let request = self.state.requests.get_mut(&uuid).unwrap_or_else(|| {
                    panic!("schedule_request: {uuid} removed mid-pass (commit no-signal)")
                });
                request.sequence.commit_allocation(allocation_target);
                request.num_computed_tokens = actual_computed_after;
                break;
            };

            match self.kv_manager.process_for_request(
                uuid,
                &signal,
                cached_prefix_tokens / self.args.block_size,
            ) {
                G1Acquire::Ready(allocated) => {
                    let expected = match &signal {
                        MoveBlock::Use(blocks, ..) => blocks.len(),
                        _ => unreachable!(),
                    };
                    assert_eq!(allocated, expected, "Use commit must be all-or-nothing");
                    let request = self.state.requests.get_mut(&uuid).unwrap_or_else(|| {
                        panic!("schedule_request: {uuid} removed mid-pass (post-process commit)")
                    });
                    request.sequence.commit_allocation(allocation_target);
                    request.num_computed_tokens = actual_computed_after;
                    request.offload_dependency = None;
                    break;
                }
                G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => {
                    let request = self.state.requests.get_mut(&uuid).unwrap_or_else(|| {
                        panic!("schedule_request: {uuid} removed while attaching dependency")
                    });
                    request.offload_dependency = Some(OffloadDependency {
                        offload_id,
                        deadline_ms,
                    });
                    return ScheduleOutcome::DependencyBlocked;
                }
                G1Acquire::RetryNow { .. } => {
                    panic!("process_use must consume its bounded RetryNow internally")
                }
                G1Acquire::CapacityExhausted => {}
            }

            let Some(preempted) = self.policy_preempt() else {
                actual_computed_after = effective_computed_before;
                break;
            };
            for signal in preempted.signals {
                assert!(
                    matches!(
                        self.kv_manager
                            .process_for_request(preempted.uuid, &signal, 0),
                        G1Acquire::Ready(_)
                    ),
                    "preemption cleanup must be infallible"
                );
            }
            *preempted_any = true;
            if let Some(undone) = scheduled.remove(&preempted.uuid) {
                *token_budget += undone.total_tokens;
                if undone.prompt_tokens > 0 && self.args.worker_type != WorkerType::Decode {
                    *batch_count = batch_count.saturating_sub(1);
                    *batch_total_isl =
                        batch_total_isl.saturating_sub(undone.prefix_tokens + undone.prompt_tokens);
                    *batch_total_prefix = batch_total_prefix.saturating_sub(undone.prefix_tokens);
                }
            }
            if preempted.uuid == uuid {
                return ScheduleOutcome::CurrentPreempted;
            }
        }

        if let Some(request) = self.state.requests.get(&uuid) {
            request.debug_assert_invariants(uuid);
        }
        let tokens_used = actual_computed_after.saturating_sub(effective_computed_before);
        if tokens_used == 0 && actual_computed_after < self.state.request_sequence_len(uuid) {
            return ScheduleOutcome::Blocked;
        }

        // vLLM's allocate_slots() caches full blocks through this request's
        // newly scheduled token boundary before scheduling the next request.
        // Keep over-allocated future blocks owner-scoped, while making blocks
        // completed by this scheduling decision visible in deterministic
        // scheduler order (including to later requests in the same pass).
        {
            let request = self.state.requests.get_mut(&uuid).unwrap_or_else(|| {
                panic!("schedule_request: {uuid} removed before prefix finalization")
            });
            self.kv_manager.finalize_computed_prefix(
                uuid,
                effective_computed_before,
                actual_computed_after,
                &mut request.sequence,
            );
        }

        let prompt_after = actual_computed_after.min(prompt_len);
        let prompt_tokens = prompt_after.saturating_sub(prompt_before);
        let sequence_len = self
            .state
            .requests
            .get(&uuid)
            .map(|r| r.sequence.len())
            .unwrap_or(0);
        let terminal_after_schedule = self.state.requests.get(&uuid).is_some_and(|request| {
            policy::generation_complete(&request.sequence, self.args.max_model_len)
        });
        scheduled.insert(
            uuid,
            ScheduledWork {
                total_tokens: tokens_used,
                prompt_tokens,
                prefix_tokens: prompt_before,
                terminal_after_schedule,
                prompt_len,
                sequence_len,
            },
        );
        if prompt_tokens > 0 && self.args.worker_type != WorkerType::Decode {
            *batch_count += 1;
            *batch_total_isl += prompt_before + prompt_tokens;
            *batch_total_prefix += prompt_before;
        }

        if from_waiting {
            self.state.transition_to_running(uuid);
        }
        *token_budget = token_budget.saturating_sub(tokens_used);

        let admission = if from_waiting {
            Some(AdmissionEvent {
                uuid,
                reused_input_tokens: cached_prefix_tokens,
            })
        } else {
            None
        };
        ScheduleOutcome::Scheduled {
            tokens_used,
            admission,
        }
    }

    #[cfg_attr(feature = "profile", inline(never))]
    fn emit_ready_tokens(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        decode_start_ms: f64,
    ) -> (Duration, Vec<OutputSignal>) {
        let mut ready = Vec::with_capacity(self.state.running.len());
        let mut already_complete = Vec::new();
        let mut total_length = 0usize;
        for uuid in self.state.running.iter().copied() {
            let Some(request) = self.state.requests.get(&uuid) else {
                continue;
            };
            if request.num_computed_tokens < request.sequence.len() {
                continue;
            }
            if policy::generation_complete(&request.sequence, self.args.max_model_len) {
                let handoff_delay_ms = compute_prefill_handoff_delay_ms(
                    self.args.worker_type,
                    true,
                    request.sequence.num_input_tokens(),
                    self.args.kv_transfer_bandwidth,
                    self.args.kv_bytes_per_token,
                );
                let effects = split_terminal_effects(request.sequence.terminal_signals());
                debug_assert!(effects.immediate.is_empty());
                already_complete.push((uuid, handoff_delay_ms, effects.cleanup));
                continue;
            }
            ready.push(uuid);
            total_length += request.sequence.len();
        }

        // Requests already terminal after prefill must release their running slots
        // without manufacturing an output token.
        let mut output_signals = Vec::with_capacity(already_complete.len() + ready.len());
        for (uuid, handoff_delay_ms, cleanup) in already_complete {
            self.complete_source(uuid, cleanup);
            output_signals.push(OutputSignal {
                uuid,
                token_id: None,
                completed: true,
                rejected: false,
                handoff_delay_ms,
            });
        }

        if ready.is_empty() {
            if !output_signals.is_empty() {
                self.state.compact_running();
            }
            return (Duration::ZERO, output_signals);
        }

        if self.speculative_sampler.is_some() {
            if output_signals.is_empty() {
                return self.emit_speculative_ready_tokens(ready, collector, decode_start_ms);
            }

            self.state.compact_running();
            let (decode_time, mut speculative_signals) =
                self.emit_speculative_ready_tokens(ready, collector, decode_start_ms);
            output_signals.append(&mut speculative_signals);
            return (decode_time, output_signals);
        }

        // For prefill workers, the first decode token is produced as part of
        // the prefill forward pass — no separate decode iteration needed.
        let (decode_time, decode_end_ms) = if self.args.worker_type == WorkerType::Prefill {
            (Duration::ZERO, decode_start_ms)
        } else {
            let active_kv_tokens = self.kv_manager.num_active_blocks() * self.args.block_size;
            let total_kv_tokens = self.args.num_gpu_blocks * self.args.block_size;
            let context_length = total_length / ready.len();
            let decode_ms = self.args.perf_model.predict_decode_time(
                ready.len(),
                active_kv_tokens,
                context_length,
                total_kv_tokens,
            );
            let dt = scale_decode_time(decode_ms, &self.args);
            (dt, decode_start_ms + dt.as_secs_f64() * 1000.0)
        };

        let mut running_changed = !output_signals.is_empty();
        for uuid in ready {
            let mut emitted = false;
            let mut emitted_token_id = None;
            let mut completed = false;
            let mut deferred_deref = Vec::new();
            loop {
                if self.refresh_request_offload_dependency(uuid).is_some() {
                    break;
                }
                self.state.debug_assert_ready_to_decode(uuid);
                let Some(sequence) = self.state.running_sequence_mut(uuid) else {
                    break;
                };
                let (token_id, mut signals) = sequence.generate_token();
                completed = policy::generation_complete(sequence, self.args.max_model_len);
                // A sampled token is dangling until the next scheduling
                // iteration. Upstream vLLM calls allocate_slots() for it only
                // when that iteration schedules the token for computation, so
                // generation never acquires its newly crossed block.
                signals.retain(|signal| !matches!(signal, MoveBlock::Use(..)));
                if completed {
                    // A terminal dangling token is never computed. Preserve
                    // any preceding full-block promotion, then replace the
                    // logical terminal Deref with cleanup for only the
                    // footprint allocated before this sample.
                    signals.retain(|signal| !matches!(signal, MoveBlock::Deref(_)));
                    signals.extend(sequence.free_signal());
                }
                let effects = if completed {
                    split_terminal_effects(signals)
                } else {
                    VllmTerminalEffects {
                        immediate: signals,
                        cleanup: Vec::new(),
                    }
                };
                let signal_outcome = if effects.immediate.is_empty() {
                    G1Acquire::Ready(())
                } else {
                    process_signals(&mut self.kv_manager, uuid, &effects.immediate)
                };
                match signal_outcome {
                    G1Acquire::Ready(()) => {
                        emitted = true;
                        emitted_token_id = Some(token_id);
                        deferred_deref = effects.cleanup;
                        break;
                    }
                    G1Acquire::BlockedOnOffload {
                        offload_id,
                        deadline_ms,
                    } => {
                        sequence.pop();
                        let request = self
                            .state
                            .requests
                            .get_mut(&uuid)
                            .expect("decode dependency request must remain active");
                        request.offload_dependency = Some(OffloadDependency {
                            offload_id,
                            deadline_ms,
                        });
                        break;
                    }
                    G1Acquire::RetryNow { .. } => {
                        panic!("decode Use must consume its bounded RetryNow internally")
                    }
                    G1Acquire::CapacityExhausted => {}
                }
                sequence.pop();

                let Some(preempted) = self.policy_preempt() else {
                    break;
                };
                running_changed = true;
                for signal in preempted.signals {
                    assert!(
                        matches!(
                            self.kv_manager
                                .process_for_request(preempted.uuid, &signal, 0),
                            G1Acquire::Ready(_)
                        ),
                        "decode preemption cleanup must be infallible"
                    );
                }
                if preempted.uuid == uuid {
                    break;
                }
            }
            if !emitted {
                continue;
            }

            let handoff_delay_ms = self.state.requests.get(&uuid).and_then(|request| {
                request.debug_assert_progress(uuid);
                compute_prefill_handoff_delay_ms(
                    self.args.worker_type,
                    completed,
                    request.sequence.num_input_tokens(),
                    self.args.kv_transfer_bandwidth,
                    self.args.kv_bytes_per_token,
                )
            });
            let output_signal = OutputSignal {
                uuid,
                token_id: emitted_token_id,
                completed,
                rejected: false,
                handoff_delay_ms,
            };
            if completed {
                self.complete_source(uuid, deferred_deref);
                running_changed = true;
            }
            if let Some(collector) = collector.as_deref_mut() {
                collector.on_token(uuid, decode_end_ms);
            }
            output_signals.push(output_signal);
        }

        if output_signals.is_empty() {
            if running_changed {
                self.state.compact_running();
            }
            return (Duration::ZERO, output_signals);
        }

        if running_changed {
            self.state.compact_running();
        }
        (decode_time, output_signals)
    }

    fn emit_speculative_ready_tokens(
        &mut self,
        mut ready: Vec<Uuid>,
        collector: Option<&mut TraceCollector>,
        decode_start_ms: f64,
    ) -> (Duration, Vec<OutputSignal>) {
        let max_burst = if self.args.worker_type == WorkerType::Prefill {
            1
        } else {
            self.args
                .aic_nextn
                .expect("speculative sampler requires nextn")
                + 1
        };
        for uuid in ready.iter().copied() {
            if self.refresh_request_offload_dependency(uuid).is_some() {
                return (Duration::ZERO, Vec::new());
            }
        }
        let mut running_changed = false;
        let mut reservation = loop {
            let required_blocks = ready
                .iter()
                .filter_map(|uuid| self.state.requests.get(uuid))
                .map(|request| {
                    let remaining = policy::remaining_generation_tokens(
                        &request.sequence,
                        self.args.max_model_len,
                    );
                    let burst = max_burst.min(remaining);
                    let current_blocks = request.sequence.len().div_ceil(self.args.block_size);
                    let target_blocks =
                        (request.sequence.len() + burst).div_ceil(self.args.block_size);
                    target_blocks.saturating_sub(current_blocks)
                })
                .sum();

            match self.kv_manager.reserve_decode_blocks(required_blocks) {
                G1Acquire::Ready(reservation) => break reservation,
                G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                } => {
                    let dependency = Some(OffloadDependency {
                        offload_id,
                        deadline_ms,
                    });
                    for uuid in &ready {
                        let request = self
                            .state
                            .requests
                            .get_mut(uuid)
                            .expect("speculative dependency request must remain active");
                        request.offload_dependency = dependency;
                    }
                    return (Duration::ZERO, Vec::new());
                }
                G1Acquire::RetryNow { .. } => {
                    panic!("speculative reservation must consume bounded RetryNow internally")
                }
                G1Acquire::CapacityExhausted => {}
            }

            let Some(preempted) = self.policy_preempt() else {
                if running_changed {
                    self.state.compact_running();
                }
                return (Duration::ZERO, Vec::new());
            };
            running_changed = true;
            for signal in preempted.signals {
                assert!(
                    matches!(
                        self.kv_manager
                            .process_for_request(preempted.uuid, &signal, 0),
                        G1Acquire::Ready(_)
                    ),
                    "speculative preemption cleanup must be infallible"
                );
            }

            ready.clear();
            for uuid in self.state.running.iter().copied() {
                let Some(request) = self.state.requests.get(&uuid) else {
                    continue;
                };
                if request.num_computed_tokens == request.sequence.len()
                    && !policy::generation_complete(&request.sequence, self.args.max_model_len)
                {
                    ready.push(uuid);
                }
            }
            if ready.is_empty() {
                self.state.compact_running();
                return (Duration::ZERO, Vec::new());
            }
        };

        let total_length = ready
            .iter()
            .filter_map(|uuid| self.state.requests.get(uuid))
            .map(|request| request.sequence.len())
            .sum::<usize>();
        let (decode_time, decode_end_ms) = if self.args.worker_type == WorkerType::Prefill {
            (Duration::ZERO, decode_start_ms)
        } else {
            let active_kv_tokens = self
                .kv_manager
                .num_active_blocks()
                .saturating_sub(reservation.len())
                * self.args.block_size;
            let total_kv_tokens = self.args.num_gpu_blocks * self.args.block_size;
            let context_length = total_length / ready.len();
            let decode_ms = self.args.perf_model.predict_decode_time(
                ready.len(),
                active_kv_tokens,
                context_length,
                total_kv_tokens,
            );
            let duration = scale_decode_time(decode_ms, &self.args);
            (duration, decode_start_ms + duration.as_secs_f64() * 1000.0)
        };

        let sampled_bursts = {
            let sampler = self
                .speculative_sampler
                .as_mut()
                .expect("speculative sampler checked above");
            ready
                .iter()
                .map(|uuid| {
                    let request = self
                        .state
                        .requests
                        .get(uuid)
                        .expect("ready request must remain active");
                    let remaining = policy::remaining_generation_tokens(
                        &request.sequence,
                        self.args.max_model_len,
                    );
                    let burst = if self.args.worker_type == WorkerType::Prefill {
                        remaining.min(1)
                    } else {
                        sampler.sample_output_tokens(remaining)
                    };
                    (*uuid, burst)
                })
                .collect::<Vec<_>>()
        };

        let mut output_signals =
            Vec::with_capacity(sampled_bursts.iter().map(|(_, burst)| *burst).sum());
        for (uuid, burst) in sampled_bursts {
            let mut completed = false;
            let mut deferred_deref = Vec::new();
            for _ in 0..burst {
                let (token_id, signals, is_complete) = {
                    let request = self
                        .state
                        .requests
                        .get_mut(&uuid)
                        .expect("sampled request must remain active");
                    let (token_id, mut signals) = request.sequence.generate_token();
                    let is_complete =
                        policy::generation_complete(&request.sequence, self.args.max_model_len);
                    if is_complete
                        && request.sequence.generated_tokens()
                            < request.sequence.max_output_tokens()
                    {
                        signals.extend(request.sequence.terminal_signals());
                    }
                    (token_id, signals, is_complete)
                };
                let effects = if is_complete {
                    split_terminal_effects(signals)
                } else {
                    VllmTerminalEffects {
                        immediate: signals,
                        cleanup: Vec::new(),
                    }
                };
                for signal in &effects.immediate {
                    self.kv_manager.process_decode_signal_for_request(
                        uuid,
                        signal,
                        &mut reservation,
                    );
                }

                let prompt_tokens = {
                    let request = self
                        .state
                        .requests
                        .get_mut(&uuid)
                        .expect("sampled request must remain active");
                    if !is_complete {
                        request.sequence.commit_allocation(request.sequence.len());
                    }
                    request.sequence.num_input_tokens()
                };
                output_signals.push(OutputSignal {
                    uuid,
                    token_id: Some(token_id),
                    completed: is_complete,
                    rejected: false,
                    handoff_delay_ms: compute_prefill_handoff_delay_ms(
                        self.args.worker_type,
                        is_complete,
                        prompt_tokens,
                        self.args.kv_transfer_bandwidth,
                        self.args.kv_bytes_per_token,
                    ),
                });
                if is_complete {
                    completed = true;
                    deferred_deref = effects.cleanup;
                    break;
                }
            }

            if completed {
                self.complete_source(uuid, deferred_deref);
                running_changed = true;
                continue;
            }

            let request = self
                .state
                .requests
                .get_mut(&uuid)
                .expect("nonterminal sampled request must remain active");
            request.num_computed_tokens = request.sequence.len().saturating_sub(1);
            request.debug_assert_progress(uuid);
            debug_assert_eq!(
                request.sequence.len() - request.num_computed_tokens,
                1,
                "nonterminal speculative decode must leave exactly one dangling token"
            );
        }

        self.kv_manager.release_decode_reservation(reservation);

        if let Some(collector) = collector {
            for signal in &output_signals {
                collector.on_token(signal.uuid, decode_end_ms);
            }
        }

        if running_changed {
            self.state.compact_running();
        }
        (decode_time, output_signals)
    }
}

fn predict_prefill_duration(
    batch_count: usize,
    batch_total_isl: usize,
    batch_total_prefix: usize,
    args: &MockEngineArgs,
) -> Duration {
    if batch_count == 0 || args.worker_type == WorkerType::Decode {
        return Duration::ZERO;
    }

    let mean_isl = batch_total_isl / batch_count;
    let mean_prefix = batch_total_prefix / batch_count;
    let prefill_ms = args
        .perf_model
        .predict_prefill_time(batch_count, mean_isl, mean_prefix);
    let total_time = Duration::from_secs_f64(prefill_ms / 1000.0);
    if args.speedup_ratio <= 0.0 || total_time <= Duration::ZERO {
        return total_time;
    }
    Duration::from_secs_f64(total_time.as_secs_f64() / args.speedup_ratio)
}

fn scale_decode_time(decode_ms: f64, args: &MockEngineArgs) -> Duration {
    let unscaled = Duration::from_secs_f64(decode_ms / 1000.0);
    let effective_ratio = args.speedup_ratio * args.decode_speedup_ratio;
    if effective_ratio <= 0.0 || unscaled <= Duration::ZERO {
        return unscaled;
    }
    Duration::from_secs_f64(unscaled.as_secs_f64() / effective_ratio)
}

fn split_terminal_effects(signals: Vec<MoveBlock>) -> VllmTerminalEffects {
    let (cleanup, immediate) = signals
        .into_iter()
        .partition(|signal| matches!(signal, MoveBlock::Deref(_)));
    VllmTerminalEffects { immediate, cleanup }
}

fn process_signals(
    kv_manager: &mut G1Manager,
    owner: Uuid,
    signals: &[MoveBlock],
) -> G1Acquire<()> {
    for signal in signals {
        match kv_manager.process_for_request(owner, signal, 0) {
            G1Acquire::Ready(_) => continue,
            G1Acquire::BlockedOnOffload {
                offload_id,
                deadline_ms,
            } => {
                validate_decode_allocation_failure(signal);
                return G1Acquire::BlockedOnOffload {
                    offload_id,
                    deadline_ms,
                };
            }
            G1Acquire::CapacityExhausted => {
                validate_decode_allocation_failure(signal);
                return G1Acquire::CapacityExhausted;
            }
            G1Acquire::RetryNow { .. } => {
                panic!("process_use must consume its bounded RetryNow internally")
            }
        }
    }
    G1Acquire::Ready(())
}

fn validate_decode_allocation_failure(signal: &MoveBlock) {
    let MoveBlock::Use(blocks, ..) = signal else {
        panic!("Failed signal is invalid. Expected decode allocation failure, got {signal:?}");
    };
    if blocks.len() != 1 {
        panic!(
            "Failed signal is invalid. Tried to allocate {} blocks during decode.",
            blocks.len()
        );
    }
    if !matches!(blocks[0], UniqueBlock::PartialBlock(_)) {
        panic!("Failed signal is invalid. Decode allocation must use a partial block.");
    }
}

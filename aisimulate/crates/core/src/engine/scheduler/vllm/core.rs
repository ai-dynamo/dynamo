// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use rustc_hash::{FxHashMap, FxHashSet};
use uuid::Uuid;

use crate::engine::common::protocols::{
    DirectRequest, KvEventPublishers, MockEngineArgs, OutputSignal, PreemptionMode, PrefillCost,
    SchedulingPolicy, WorkerType,
};
use crate::engine::common::speculative::{
    SpeculativeDecodeSampler, normalize_conditional_accept_rates,
};
use crate::engine::common::utils::{
    compute_prefill_handoff_delay_ms, prefill_handoff_transfer_timing,
};
use crate::engine::kv_manager::G1Manager;
use crate::engine::kv_manager::{DestinationReservation, G1Acquire};
#[cfg(test)]
use crate::engine::scheduler::accept_length_sample;
use crate::engine::scheduler::vllm::policy::{self, AdmissionDecision, PolicySequence};
use crate::engine::scheduler::vllm::request::RequestKvState;
use crate::engine::scheduler::{
    ActiveHandoffRequests, AdmissionEvent, AdmissionInvariant, AdmissionStage,
    CapturedKvEventBuffer, DestinationHolds, EnginePassResult, ForwardPassSnapshot,
    KvEventVisibility, MockerMetrics, PendingDestinations, RemovedSource, SchedulerCommand,
    SchedulerCommandEffects, SchedulerCommandResult, SchedulerLifecycleEvent, SourceCompletion,
    SourceHolds, build_fpm_snapshot, capture_kv_event_sink,
};
use crate::engine::trace::TraceCollector;
use crate::engine::{HandoffId, PressureEvent, PressureKind, PressureState, modeled_duration_ms};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestStatus {
    WaitingForRemoteKv,
    Waiting,
    Running,
    Preempted,
}

pub(crate) struct VllmRequestState {
    pub(crate) sequence: RequestKvState,
    pub(crate) status: RequestStatus,
    pub(crate) num_computed_tokens: usize,
    pub(crate) num_preemptions: usize,
    /// Prefix tokens found cached at first admission (set once: a preempted
    /// request re-probes against a cache warmed by its own blocks, which
    /// would inflate the value).
    pub(crate) cached_prefix_tokens: Option<usize>,
    /// Whether the admission cache truth was already attached to a signal.
    pub(crate) cached_tokens_signaled: bool,
}

impl VllmRequestState {
    /// Admission cache truth rides the request's first signal only.
    fn take_cached_tokens_for_signal(&mut self) -> Option<usize> {
        if self.cached_tokens_signaled {
            None
        } else {
            self.cached_tokens_signaled = true;
            self.cached_prefix_tokens
        }
    }

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

    pub(crate) fn take_completed(&mut self, uuid: &Uuid) -> Option<VllmRequestState> {
        self.waiting_members.remove(uuid);
        self.running_members.remove(uuid);
        self.requests.remove(uuid)
    }

    pub(crate) fn running_sequence_mut(&mut self, uuid: Uuid) -> Option<&mut RequestKvState> {
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
        request.num_computed_tokens = 0;
        request.num_preemptions += 1;
        self.preemptions_total += 1;
        request.debug_assert_invariants(uuid);
        self.prepend_waiting(uuid);
        Some(PreemptedRequest { uuid })
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

pub(crate) struct VllmCore {
    pub(super) args: MockEngineArgs,
    dp_rank: u32,
    pub(super) state: SchedulerState,
    pub(super) kv_manager: G1Manager,
    speculative_sampler: Option<SpeculativeDecodeSampler>,
    kv_event_buffer: Option<CapturedKvEventBuffer>,
    source_holds: SourceHolds<HeldVllmPrefill>,
    pending_destinations: PendingDestinations<VllmRequestState>,
    destination_holds: DestinationHolds<ReservedVllmDecode>,
    active_destination_handoffs: ActiveHandoffRequests,
    capacity_generation: u64,
    #[cfg(test)]
    destination_reservation_attempts: usize,
    lifecycle_events: Vec<SchedulerLifecycleEvent>,
    pressure_events: Vec<PressureEvent>,
    retain_local_hashes: bool,
    emit_token_ids: bool,
}

struct HeldVllmPrefill {
    request_id: Uuid,
    request: VllmRequestState,
}

struct ReservedVllmDecode {
    request: VllmRequestState,
    kv: DestinationReservation,
}

impl ReservedVllmDecode {
    fn activate(self, kv_manager: &mut G1Manager) -> VllmRequestState {
        let Self { mut request, kv } = self;
        let prompt_len = request.sequence.num_input_tokens();
        let owner = request.sequence.lease.owner();
        kv_manager.activate_native_destination(
            owner,
            &request.sequence.sequence,
            &mut request.sequence.lease,
            kv,
        );
        request.num_computed_tokens = prompt_len;
        request.status = RequestStatus::Waiting;
        request
    }

    fn cancel(self, kv_manager: &mut G1Manager) {
        let Self { request: _, kv } = self;
        kv_manager.cancel_destination(kv);
    }
}

impl VllmCore {
    #[cfg(test)]
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        Self::new_internal(args, 0, 0, None, KvEventPublishers::default())
    }

    #[cfg(test)]
    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: u64) -> Self {
        Self::new_with_worker_rank(args, worker_id, 0, worker_id, true)
    }

    pub(crate) fn new_with_worker_rank(
        args: MockEngineArgs,
        _worker_id: u64,
        dp_rank: u32,
        seed_offset: u64,
        capture_kv_events: bool,
    ) -> Self {
        let (buffer, publishers) = if capture_kv_events {
            let (buffer, sink) = capture_kv_event_sink();
            (Some(buffer), KvEventPublishers::new(Some(sink)))
        } else {
            (None, KvEventPublishers::default())
        };
        Self::new_internal(args, dp_rank, seed_offset, buffer, publishers)
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        seed_offset: u64,
        kv_event_buffer: Option<CapturedKvEventBuffer>,
        kv_event_publishers: KvEventPublishers,
    ) -> Self {
        let kv_event_publishers = if args.enable_prefix_caching {
            kv_event_publishers
        } else {
            KvEventPublishers::default()
        };
        let retain_local_hashes = !kv_event_publishers.is_empty() || args.emit_kv_events;
        let emit_token_ids = args.emit_kv_token_ids;
        let speculative_sampler = args.aic_nextn.map(|nextn| {
            let rates =
                normalize_conditional_accept_rates(nextn, args.aic_nextn_accept_rates.as_deref())
                    .expect("normalized MTP acceptance rates");
            SpeculativeDecodeSampler::new(rates, args.aic_mtp_seed.wrapping_add(seed_offset))
        });
        Self {
            kv_manager: G1Manager::new_with_caching(
                args.num_gpu_blocks,
                args.block_size,
                kv_event_publishers,
                dp_rank,
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
            pressure_events: Vec::new(),
            retain_local_hashes,
            emit_token_ids,
        }
    }

    #[cfg(test)]
    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        match self
            .apply_command(SchedulerCommand::Submit(request))
            .expect("ordinary request ID must be unique")
        {
            SchedulerCommandResult::Submitted(uuid) => uuid,
            _ => unreachable!("submit command must return a request ID"),
        }
    }

    #[cfg(test)]
    pub(crate) fn request_uses_flat_tokens(&self, uuid: Uuid) -> bool {
        self.state
            .requests
            .get(&uuid)
            .is_some_and(|request| request.sequence.uses_flat_tokens())
    }

    #[cfg(test)]
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
                let retired = self.state.requests.contains_key(&request_id);
                let result = if retired {
                    self.drop_request(request_id);
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                let effects = if allow_destination_admission {
                    self.effects_after_capacity_change(result, reservation_now_ms)
                } else {
                    SchedulerCommandEffects::new(result)
                };
                Ok(if retired {
                    effects.retire(request_id)
                } else {
                    effects
                })
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
                let (applied, retired) = self.release_source(handoff_id);
                let result = if applied {
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                let effects = self.effects_after_capacity_change(result, reservation_now_ms);
                Ok(if let Some(request_id) = retired {
                    effects.retire(request_id)
                } else {
                    effects
                })
            }
            SchedulerCommand::CancelSource { handoff_id } => {
                let (applied, retired) = self.cancel_source(handoff_id);
                let result = if applied {
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                let effects = self.effects_after_capacity_change(result, reservation_now_ms);
                Ok(if let Some(request_id) = retired {
                    effects.retire(request_id)
                } else {
                    effects
                })
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
                if let Some((request_id, _)) = self.pending_destinations.remove(handoff_id) {
                    self.bump_capacity_generation();
                    return Ok(self
                        .effects_after_capacity_change(
                            SchedulerCommandResult::Applied,
                            reservation_now_ms,
                        )
                        .retire(request_id));
                }
                if let Some((request_id, reservation)) = self.destination_holds.remove(handoff_id) {
                    reservation.cancel(&mut self.kv_manager);
                    self.bump_capacity_generation();
                    return Ok(self
                        .effects_after_capacity_change(
                            SchedulerCommandResult::Applied,
                            reservation_now_ms,
                        )
                        .retire(request_id));
                }
                let Some(request_id) = self.active_destination_handoffs.remove_handoff(handoff_id)
                else {
                    return Ok(SchedulerCommandEffects::new(SchedulerCommandResult::Noop));
                };
                self.drop_request(request_id);
                Ok(self
                    .effects_after_capacity_change(
                        SchedulerCommandResult::Applied,
                        reservation_now_ms,
                    )
                    .retire(request_id))
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
        #[cfg(test)]
        {
            self.destination_reservation_attempts += 1;
        }
        let reservation = self.kv_manager.reserve_native_destination_at(
            request_id,
            &request.sequence.sequence,
            &request.sequence.lease,
            reservation_now_ms,
        );
        let kv = match reservation {
            G1Acquire::Ready(kv) => kv,
            G1Acquire::CapacityExhausted => {
                self.pending_destinations.mark_front_attempted(generation);
                return Vec::new();
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
        if self.state.requests.contains_key(&uuid)
            || self.source_holds.contains_request(uuid)
            || self.pending_destinations.contains_request(uuid)
            || self.destination_holds.contains_request(uuid)
            || self.active_destination_handoffs.contains_request(uuid)
        {
            anyhow::bail!("request {uuid} is already active");
        }
        Ok(())
    }

    /// Output tokens worth reserving storage for when materializing a request.
    ///
    /// This is only an allocation hint. The sequence retains the request's
    /// logical `max_output_tokens`; scheduling separately enforces the model
    /// and physical KV limits.
    fn output_capacity_hint(&self, prompt_len: usize, max_output_tokens: usize) -> usize {
        let kv_remaining = self
            .args
            .num_gpu_blocks
            .saturating_mul(self.args.block_size)
            .saturating_sub(prompt_len);
        let model_remaining = if self.args.scheduling_policy() == SchedulingPolicy::Vllm {
            self.args
                .max_model_len
                .map(|limit| limit.saturating_sub(prompt_len))
                .unwrap_or(usize::MAX)
        } else {
            usize::MAX
        };
        max_output_tokens.min(kv_remaining).min(model_remaining)
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
        let prompt_len = request.tokens.len();
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
            prompt_len,
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
        let output_capacity_hint = self.output_capacity_hint(prompt_len, max_output_tokens);
        let sequence = RequestKvState::native(
            uuid,
            request.tokens,
            max_output_tokens,
            output_capacity_hint,
            self.args.block_size,
            self.args.enable_prefix_caching,
            self.retain_local_hashes,
            self.emit_token_ids,
            planned_output_ids,
        );
        VllmRequestState {
            sequence,
            status,
            num_computed_tokens: 0,
            num_preemptions: 0,
            cached_prefix_tokens: None,
            cached_tokens_signaled: false,
        }
    }

    fn release_source(&mut self, handoff_id: HandoffId) -> (bool, Option<Uuid>) {
        match self.source_holds.remove(handoff_id) {
            RemovedSource::Held(payload) => {
                let request_id = payload.request_id;
                self.cleanup_completed_prefill(payload);
                self.bump_capacity_generation();
                (true, Some(request_id))
            }
            RemovedSource::Pending { .. } => (true, None),
            RemovedSource::Missing => (false, None),
        }
    }

    fn cancel_source(&mut self, handoff_id: HandoffId) -> (bool, Option<Uuid>) {
        match self.source_holds.remove(handoff_id) {
            RemovedSource::Held(payload) => {
                let request_id = payload.request_id;
                self.cleanup_completed_prefill(payload);
                self.bump_capacity_generation();
                (true, Some(request_id))
            }
            RemovedSource::Pending { request_id } => {
                self.drop_request(request_id);
                (true, Some(request_id))
            }
            RemovedSource::Missing => (false, None),
        }
    }

    fn complete_source(&mut self, uuid: Uuid) {
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
        } = payload;
        self.kv_manager
            .finish_native(request_id, request.sequence.lease);
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
        true
    }

    pub(crate) fn waiting_for_external_command(&self) -> bool {
        self.is_empty() && !self.is_drained()
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
            .map(|request| request.sequence.lease.resident_block_count())
            .unwrap_or(0)
    }

    pub(crate) fn drain_kv_events(&self) -> Vec<crate::engine::KvEvent> {
        self.kv_event_buffer
            .as_ref()
            .map(CapturedKvEventBuffer::drain)
            .unwrap_or_default()
    }

    #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.try_execute_pass(collector, now_ms)
            .expect("vLLM scheduler pass failed")
    }

    #[cfg(test)]
    pub(crate) fn try_execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> anyhow::Result<EnginePassResult> {
        self.execute_pass_internal(Some(collector), now_ms)
    }

    pub(crate) fn try_execute_hidden_pass(
        &mut self,
        now_ms: f64,
    ) -> anyhow::Result<EnginePassResult> {
        self.execute_pass_internal(None, now_ms)
    }

    pub(super) fn execute_pass_internal(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        now_ms: f64,
    ) -> anyhow::Result<EnginePassResult> {
        #[cfg(test)]
        let requests_before = self.state.requests.len();
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
                now_ms,
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
                        admissions.push(admission);
                    }
                    req_index += 1;
                }
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
            let prefill_cost = match decision {
                AdmissionDecision::Admit { prefill_cost } => prefill_cost,
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
                now_ms,
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
                        admissions.push(admission);
                    }
                    if tokens_used == 0 && token_budget == 0 {
                        break;
                    }
                }
                ScheduleOutcome::Blocked | ScheduleOutcome::CurrentPreempted => break,
            }
        }

        let prefill_time =
            predict_prefill_duration(batch_count, batch_total_isl, batch_total_prefix, &self.args)?;
        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let (decode_time, mut output_signals) =
            self.emit_ready_tokens(collector, decode_start_ms, now_ms)?;
        // Emit the terminal signals for the requests the gate rejected above
        // (see the gate comment for why this can't be done inline).
        for uuid in rejected_uuids {
            output_signals.push(OutputSignal {
                uuid,
                token_id: None,
                completed: true,
                rejected: true,
                handoff_delay_ms: None,
                cached_tokens: None,
            });
        }
        let end_ms = decode_start_ms + decode_time.as_secs_f64() * 1000.0;

        let fpm = self.compute_fpm(&scheduled, (end_ms - now_ms) / 1000.0);
        #[cfg(test)]
        let (accept_length_output_tokens, accept_length_decode_forwards) =
            accept_length_sample(&output_signals);
        self.state.debug_assert_invariants();
        Ok(EnginePassResult {
            end_ms,
            same_timestamp_retry: crate::engine::generalized::SameTimestampRetry::NotApplicable,
            #[cfg(test)]
            completed_requests: requests_before.saturating_sub(self.state.requests.len()),
            output_signals,
            admissions,
            pressure_events: std::mem::take(&mut self.pressure_events),
            lifecycle_events: std::mem::take(&mut self.lifecycle_events),
            mocker_metrics: self.mocker_metrics(),
            // vLLM and TensorRT-LLM share this core. Publish completed-block
            // mutations only when the pass completes so a request arriving
            // mid-pass cannot route against partially materialized KV state.
            kv_event_visibility: KvEventVisibility::PassEnd,
            kv_events: self
                .kv_event_buffer
                .as_ref()
                .map(CapturedKvEventBuffer::drain)
                .unwrap_or_default(),
            fpm: Some(fpm),
            #[cfg(test)]
            accept_length_output_tokens,
            #[cfg(test)]
            accept_length_decode_forwards,
        })
    }

    pub(super) fn drop_request(&mut self, uuid: Uuid) {
        let active_blocks_before = self.kv_manager.num_active_blocks();
        let Some(request) = self.state.requests.get(&uuid) else {
            if self.kv_manager.num_active_blocks() < active_blocks_before {
                self.bump_capacity_generation();
            }
            return;
        };
        let capacity_improved = request.sequence.num_allocated_tokens() > 0
            || self.state.running_members.contains(&uuid)
            || self.kv_manager.num_active_blocks() < active_blocks_before;
        self.source_holds.remove_request(uuid);
        self.active_destination_handoffs.remove_request(uuid);
        let request = self
            .state
            .take_completed(&uuid)
            .expect("request drop must remove the scheduler-owned request");
        self.kv_manager.finish_native(uuid, request.sequence.lease);
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
    pub(super) fn policy_preempt(&mut self, at_ms: f64) -> Option<PreemptedRequest> {
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
            if !is_running {
                continue;
            }
            selected = Some(uuid);
            break;
        }
        let selected = selected?;
        let state_before = self.pressure_state();
        let request_active_blocks_before = self
            .state
            .requests
            .get(&selected)
            .map(|request| {
                request
                    .sequence
                    .num_allocated_tokens()
                    .div_ceil(self.args.block_size)
            })
            .unwrap_or_default();
        let preempted = self.state.preempt_uuid(selected);
        if let Some(preempted) = preempted.as_ref()
            && let Some(request) = self.state.requests.get_mut(&preempted.uuid)
        {
            self.kv_manager
                .preempt_native(preempted.uuid, &mut request.sequence.lease);
        }
        if let Some(preempted) = preempted.as_ref() {
            debug_assert_eq!(
                self.state.requests[&preempted.uuid]
                    .sequence
                    .num_allocated_tokens(),
                0,
                "preempted request {} should release all allocated KV",
                preempted.uuid
            );
            self.bump_capacity_generation();
            tracing::debug!(
                worker_id = self.dp_rank,
                request_id = %preempted.uuid,
                preemptions_total = self.state.preemptions_total,
                "vLLM scheduler preempted and requeued request"
            );
            let state_after = self.pressure_state();
            self.pressure_events.push(PressureEvent {
                at_ms,
                kind: PressureKind::VllmPreemption,
                request_id: preempted.uuid,
                state_before,
                state_after,
                request_active_blocks_before,
                logical_available_blocks_before: None,
                required_blocks_before: None,
            });
        }
        preempted
    }

    fn pressure_state(&self) -> PressureState {
        PressureState {
            running_requests: self.state.running_members.len(),
            waiting_requests: Some(self.state.waiting_members.len()),
            active_blocks: self.kv_manager.num_active_blocks(),
        }
    }

    /// Compute a forward pass metrics snapshot from the just-completed pass.
    ///
    /// `scheduled` contains the work items that were scheduled in this iteration.
    /// Per-request metadata (prompt_len, sequence_len) is captured in `ScheduledWork`
    /// at schedule time, so this method does not depend on `self.state.requests` for
    /// scheduled requests — completed requests may have already been removed.
    /// Queue metrics are derived from `self.state.waiting` at the moment of the call.
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
        pressure_at_ms: f64,
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
                        request.sequence.prefill_cost(&self.kv_manager),
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

        loop {
            let allocation_outcome = {
                let kv_manager = &mut self.kv_manager;
                let request = self.state.requests.get_mut(&uuid).unwrap_or_else(|| {
                    panic!("schedule_request: {uuid} removed mid-pass (allocation)")
                });
                let allocation_target = desired_computed_after;
                let prev_allocated_tokens = request.sequence.num_allocated_tokens();
                if allocation_target <= prev_allocated_tokens {
                    request.num_computed_tokens = actual_computed_after;
                    G1Acquire::Ready(0)
                } else {
                    let outcome = kv_manager.allocate_native(
                        uuid,
                        &mut request.sequence.lease,
                        allocation_target,
                        cached_prefix_tokens / self.args.block_size,
                    );
                    match outcome {
                        G1Acquire::Ready(_) => {
                            request.num_computed_tokens = actual_computed_after;
                        }
                        G1Acquire::CapacityExhausted => {}
                    }
                    outcome
                }
            };

            match allocation_outcome {
                G1Acquire::Ready(_) => break,
                G1Acquire::CapacityExhausted => {}
            }

            let Some(preempted) = self.policy_preempt(pressure_at_ms) else {
                actual_computed_after = effective_computed_before;
                break;
            };
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
            self.kv_manager.finalize_native_computed_prefix(
                uuid,
                effective_computed_before,
                actual_computed_after,
                &mut request.sequence.sequence,
                &mut request.sequence.lease,
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
            self.state
                .requests
                .get_mut(&uuid)
                .unwrap_or_else(|| panic!("schedule_request: {uuid} removed mid-pass (admission)"))
                .cached_prefix_tokens
                .get_or_insert(cached_prefix_tokens);
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

    fn emit_ready_tokens(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        decode_start_ms: f64,
        pressure_at_ms: f64,
    ) -> anyhow::Result<(Duration, Vec<OutputSignal>)> {
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
                already_complete.push((uuid, handoff_delay_ms));
                continue;
            }
            ready.push(uuid);
            total_length += request.sequence.len();
        }

        // Requests already terminal after prefill must release their running slots
        // without manufacturing an output token.
        let mut output_signals = Vec::with_capacity(already_complete.len() + ready.len());
        for (uuid, handoff_delay_ms) in already_complete {
            // The request's only signal; read before complete_source drops the state.
            let cached_tokens = self
                .state
                .requests
                .get_mut(&uuid)
                .and_then(VllmRequestState::take_cached_tokens_for_signal);
            self.complete_source(uuid);
            output_signals.push(OutputSignal {
                uuid,
                token_id: None,
                completed: true,
                rejected: false,
                handoff_delay_ms,
                cached_tokens,
            });
        }

        if ready.is_empty() {
            if !output_signals.is_empty() {
                self.state.compact_running();
            }
            return Ok((Duration::ZERO, output_signals));
        }

        if self.speculative_sampler.is_some() {
            if output_signals.is_empty() {
                return self.emit_speculative_ready_tokens(
                    ready,
                    collector,
                    decode_start_ms,
                    pressure_at_ms,
                );
            }

            self.state.compact_running();
            let (decode_time, mut speculative_signals) = self.emit_speculative_ready_tokens(
                ready,
                collector,
                decode_start_ms,
                pressure_at_ms,
            )?;
            output_signals.append(&mut speculative_signals);
            return Ok((decode_time, output_signals));
        }

        // For prefill workers, the first decode token is produced as part of
        // the prefill forward pass — no separate decode iteration needed.
        let (decode_time, decode_end_ms) = if self.args.worker_type == WorkerType::Prefill {
            (Duration::ZERO, decode_start_ms)
        } else {
            let total_kv_tokens = self.args.num_gpu_blocks * self.args.block_size;
            let active_kv_tokens = total_length;
            let context_length = total_length / ready.len();
            let decode_ms = self.args.perf_model.predict_decode_time(
                ready.len(),
                active_kv_tokens,
                context_length,
                total_kv_tokens,
            )?;
            let dt = scale_decode_time(decode_ms, &self.args)?;
            (dt, decode_start_ms + dt.as_secs_f64() * 1000.0)
        };

        let mut running_changed = !output_signals.is_empty();
        for uuid in ready {
            self.state.debug_assert_ready_to_decode(uuid);
            let Some(sequence) = self.state.running_sequence_mut(uuid) else {
                continue;
            };
            // Native G1 allocates the token about to be computed at the next
            // scheduler pass, matching vLLM's dangling-sample boundary.
            let token_id = sequence.generate_token();
            let completed = policy::generation_complete(sequence, self.args.max_model_len);

            let worker_type = self.args.worker_type;
            let kv_transfer_bandwidth = self.args.kv_transfer_bandwidth;
            let kv_bytes_per_token = self.args.kv_bytes_per_token;
            let (handoff_delay_ms, cached_tokens) = match self.state.requests.get_mut(&uuid) {
                Some(request) => {
                    request.debug_assert_progress(uuid);
                    let handoff_delay_ms = compute_prefill_handoff_delay_ms(
                        worker_type,
                        completed,
                        request.sequence.num_input_tokens(),
                        kv_transfer_bandwidth,
                        kv_bytes_per_token,
                    );
                    (handoff_delay_ms, request.take_cached_tokens_for_signal())
                }
                None => (None, None),
            };
            let output_signal = OutputSignal {
                uuid,
                token_id: Some(token_id),
                completed,
                rejected: false,
                handoff_delay_ms,
                cached_tokens,
            };
            if completed {
                self.complete_source(uuid);
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
            return Ok((Duration::ZERO, output_signals));
        }

        if running_changed {
            self.state.compact_running();
        }
        Ok((decode_time, output_signals))
    }

    fn emit_speculative_ready_tokens(
        &mut self,
        mut ready: Vec<Uuid>,
        collector: Option<&mut TraceCollector>,
        decode_start_ms: f64,
        pressure_at_ms: f64,
    ) -> anyhow::Result<(Duration, Vec<OutputSignal>)> {
        let max_burst = if self.args.worker_type == WorkerType::Prefill {
            1
        } else {
            self.args
                .aic_nextn
                .expect("speculative sampler requires nextn")
                + 1
        };
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
                G1Acquire::CapacityExhausted => {}
            }

            let Some(_preempted) = self.policy_preempt(pressure_at_ms) else {
                if running_changed {
                    self.state.compact_running();
                }
                return Ok((Duration::ZERO, Vec::new()));
            };
            running_changed = true;

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
                return Ok((Duration::ZERO, Vec::new()));
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
            let total_kv_tokens = self.args.num_gpu_blocks * self.args.block_size;
            let active_kv_tokens = total_length;
            let context_length = total_length / ready.len();
            let decode_ms = self.args.perf_model.predict_decode_time(
                ready.len(),
                active_kv_tokens,
                context_length,
                total_kv_tokens,
            )?;
            let duration = scale_decode_time(decode_ms, &self.args)?;
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
            for _ in 0..burst {
                let (token_id, is_complete) = {
                    let kv_manager = &mut self.kv_manager;
                    let request = self
                        .state
                        .requests
                        .get_mut(&uuid)
                        .expect("sampled request must remain active");
                    let len = request.sequence.sequence.len();
                    kv_manager.finalize_native_computed_prefix(
                        uuid,
                        len.saturating_sub(1),
                        len,
                        &mut request.sequence.sequence,
                        &mut request.sequence.lease,
                    );
                    let token_id = request.sequence.generate_token();
                    let is_complete =
                        policy::generation_complete(&request.sequence, self.args.max_model_len);
                    (token_id, is_complete)
                };
                if let Some(request) = self.state.requests.get_mut(&uuid)
                    && request.sequence.lease.allocated_tokens() < request.sequence.sequence.len()
                {
                    self.kv_manager.use_native_decode_reservation(
                        uuid,
                        &mut request.sequence.lease,
                        request.sequence.sequence.len(),
                        &mut reservation,
                    );
                }

                let (prompt_tokens, cached_tokens) = {
                    let request = self
                        .state
                        .requests
                        .get_mut(&uuid)
                        .expect("sampled request must remain active");
                    (
                        request.sequence.num_input_tokens(),
                        request.take_cached_tokens_for_signal(),
                    )
                };
                output_signals.push(OutputSignal {
                    uuid,
                    token_id: Some(token_id),
                    completed: is_complete,
                    rejected: false,
                    cached_tokens,
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
                    break;
                }
            }

            if completed {
                self.complete_source(uuid);
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
        Ok((decode_time, output_signals))
    }
}

fn predict_prefill_duration(
    batch_count: usize,
    batch_total_isl: usize,
    batch_total_prefix: usize,
    args: &MockEngineArgs,
) -> anyhow::Result<Duration> {
    if batch_count == 0 || args.worker_type == WorkerType::Decode {
        return Ok(Duration::ZERO);
    }

    let mean_isl = batch_total_isl / batch_count;
    let mean_prefix = batch_total_prefix / batch_count;
    let prefill_ms = args
        .perf_model
        .predict_prefill_time(batch_count, mean_isl, mean_prefix)?;
    let modeled_ms = modeled_duration_ms(prefill_ms, args.speedup_ratio)?;
    Ok(Duration::from_secs_f64(modeled_ms / 1_000.0))
}

fn scale_decode_time(decode_ms: f64, args: &MockEngineArgs) -> anyhow::Result<Duration> {
    let effective_ratio = args.speedup_ratio * args.decode_speedup_ratio;
    let modeled_ms = modeled_duration_ms(decode_ms, effective_ratio)?;
    Ok(Duration::from_secs_f64(modeled_ms / 1_000.0))
}

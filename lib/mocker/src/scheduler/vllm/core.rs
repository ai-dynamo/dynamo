// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use dynamo_tokens::blocks::UniqueBlock;
#[cfg(feature = "kvbm-offload")]
use kvbm_logical::MutableBlock;
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::mpsc;
use uuid::Uuid;

#[cfg(feature = "kvbm-offload")]
use crate::common::protocols::G1;
use crate::common::protocols::{
    DirectRequest, KvEventPublishers, MockEngineArgs, MoveBlock, OutputSignal, PreemptionMode,
    WorkerType,
};
use crate::common::sequence::ActiveSequence;
use crate::common::utils::compute_prefill_handoff_delay_ms;
use crate::kv_manager::KvManager;
use crate::replay::TraceCollector;
use crate::scheduler::{
    AdmissionEvent, CapturedRouterEventBuffer, EnginePassResult, ForwardPassSnapshot,
    RouterEventVisibility, build_fpm_snapshot, capture_router_event_sink,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestStatus {
    Waiting,
    Running,
    Preempted,
}

pub(crate) struct VllmRequestState {
    pub(crate) sequence: ActiveSequence,
    pub(crate) status: RequestStatus,
    pub(crate) num_computed_tokens: usize,
    pub(crate) num_preemptions: usize,
}

#[derive(Default)]
pub(crate) struct SchedulerState {
    pub(crate) waiting: VecDeque<Uuid>,
    waiting_members: FxHashSet<Uuid>,
    pub(crate) running: VecDeque<Uuid>,
    running_members: FxHashSet<Uuid>,
    pub(crate) requests: FxHashMap<Uuid, VllmRequestState>,
}

struct PreemptedRequest {
    uuid: Uuid,
    signals: Vec<MoveBlock>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ScheduledWork {
    total_tokens: usize,
    prompt_tokens: usize,
    prefix_tokens: usize,
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

    fn push_waiting(&mut self, uuid: Uuid) {
        if !self.waiting_members.insert(uuid) {
            return;
        }
        self.waiting.push_back(uuid);
    }

    fn prepend_waiting(&mut self, uuid: Uuid) {
        if !self.waiting_members.insert(uuid) {
            return;
        }
        self.waiting.push_front(uuid);
    }

    /// Remove `uuid` from the waiting queue (front-only) and from the
    /// `waiting_members` set. Shared between `transition_to_running`
    /// (which then promotes to running) and the offload admission
    /// hook's parking path (which keeps the request in `Waiting`
    /// status while parked on a swap-in).
    fn remove_from_waiting(&mut self, uuid: Uuid) {
        if self.waiting.front().copied() == Some(uuid) {
            self.waiting.pop_front();
        }
        self.waiting_members.remove(&uuid);
    }

    fn next_waiting_uuid(&mut self) -> Option<Uuid> {
        loop {
            let uuid = *self.waiting.front()?;
            let Some(request) = self.requests.get(&uuid) else {
                self.waiting.pop_front();
                self.waiting_members.remove(&uuid);
                continue;
            };
            if self.waiting_members.contains(&uuid) && request.status != RequestStatus::Running {
                return Some(uuid);
            }
            self.waiting.pop_front();
            self.waiting_members.remove(&uuid);
        }
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

    pub(crate) fn complete(&mut self, uuid: &Uuid) {
        self.waiting_members.remove(uuid);
        self.running_members.remove(uuid);
        self.requests.remove(uuid);
    }

    pub(crate) fn running_sequence_mut(&mut self, uuid: Uuid) -> Option<&mut ActiveSequence> {
        if !self.running_members.contains(&uuid) {
            return None;
        }
        self.requests
            .get_mut(&uuid)
            .map(|request| &mut request.sequence)
    }

    fn preempt(&mut self, mode: PreemptionMode) -> Option<PreemptedRequest> {
        let uuid = loop {
            let candidate = match mode {
                PreemptionMode::Lifo => self.running.pop_back(),
                PreemptionMode::Fifo => self.running.pop_front(),
            }?;
            let is_running = self.running_members.contains(&candidate)
                && self
                    .requests
                    .get(&candidate)
                    .is_some_and(|request| request.status == RequestStatus::Running);
            if is_running {
                break candidate;
            }
            self.running_members.remove(&candidate);
        };
        self.running_members.remove(&uuid);
        let request = self.requests.get_mut(&uuid)?;
        request.status = RequestStatus::Preempted;
        request.num_computed_tokens = 0;
        request.num_preemptions += 1;
        let signals = request.sequence.reset_with_signal();
        debug_assert_vllm_request_invariants(uuid, request);
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
/// after the transfer completes.
#[cfg(feature = "kvbm-offload")]
pub(crate) struct AwaitingSwapIn {
    pub(crate) uuid: Uuid,
    pub(crate) handle: crate::kvbm_offload::SwapInHandle,
    pub(crate) destination_slots: Vec<MutableBlock<G1>>,
    pub(crate) skip_blocks: usize,
}

#[cfg(feature = "kvbm-offload")]
enum SwapInAdmissionAttempt {
    NoHit,
    Parked,
    BlockedOnG1Offload,
}

pub(crate) struct VllmCore {
    args: MockEngineArgs,
    pub(super) state: SchedulerState,
    pub(super) kv_manager: KvManager,
    kv_event_buffer: Option<CapturedRouterEventBuffer>,

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

impl VllmCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        Self::new_internal(args, 0, None, KvEventPublishers::default())
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        let (buffer, sink) = capture_router_event_sink(worker_id);
        Self::new_internal(
            args,
            0,
            Some(buffer),
            KvEventPublishers::new(Some(sink), None),
        )
    }

    pub(super) fn new_with_sink(
        args: MockEngineArgs,
        dp_rank: u32,
        kv_event_publishers: KvEventPublishers,
    ) -> Self {
        Self::new_internal(args, dp_rank, None, kv_event_publishers)
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        kv_event_buffer: Option<CapturedRouterEventBuffer>,
        kv_event_publishers: KvEventPublishers,
    ) -> Self {
        let args = args.normalized().expect("invalid MockEngineArgs");
        Self {
            kv_manager: KvManager::new_with_event_sink(
                args.num_gpu_blocks,
                args.block_size,
                kv_event_publishers,
                dp_rank,
            ),
            args,
            state: SchedulerState::default(),
            kv_event_buffer,
            #[cfg(feature = "kvbm-offload")]
            requests_awaiting_swap_in: Vec::new(),
        }
    }

    /// Wire a live-mode (`ClockSource::Real`) offload engine onto this
    /// core's `KvManager`. No-op when `args.kv_bytes_per_token` is
    /// unset. Caller must be inside an ambient tokio runtime.
    #[cfg(feature = "kvbm-offload")]
    pub(crate) async fn init_offload_live(&mut self) -> anyhow::Result<()> {
        crate::scheduler::init_kvbm_live(&self.args, &mut self.kv_manager).await?;
        Ok(())
    }

    /// Wire an offline-mode (`ClockSource::Virtual`) offload engine
    /// onto this core's `KvManager`. No-op when
    /// `args.kv_bytes_per_token` is unset. Sync entry — owns the
    /// internal tokio runtime via `attach_runtime`.
    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn init_offload_offline(&mut self) -> anyhow::Result<()> {
        crate::scheduler::init_kvbm_offline(&self.args, &mut self.kv_manager)?;
        Ok(())
    }

    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        let sequence = ActiveSequence::new(
            request.tokens,
            request.max_output_tokens,
            Some(self.args.block_size),
            self.args.enable_prefix_caching,
            self.args.zmq_kv_events_port.is_some(),
        );
        self.state.requests.insert(
            uuid,
            VllmRequestState {
                sequence,
                status: RequestStatus::Waiting,
                num_computed_tokens: 0,
                num_preemptions: 0,
            },
        );
        self.state.push_waiting(uuid);
        if let Some(request) = self.state.requests.get(&uuid) {
            debug_assert_vllm_request_progress(uuid, request);
        }
        uuid
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.state.requests.len()
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
        self.kv_manager.tick_offload_engine(now_ms);
        let awaiting = std::mem::take(&mut self.requests_awaiting_swap_in);
        let mut completed = Vec::new();
        let mut pending = Vec::with_capacity(awaiting.len());

        for aws in awaiting {
            if aws.handle.is_complete() {
                completed.push(aws);
            } else {
                pending.push(aws);
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
    }

    /// Register the onboard'd PLHs into G1 inactive (so the request's
    /// next `process_use` sees `InactiveHit`) and re-queue the request at
    /// the front for admission. The swap-in covers
    /// `[skip_blocks .. skip_blocks + count]` of the request's block
    /// sequence — we skip the G1-cached prefix the request already had
    /// and register only the uncached-remainder blocks that the engine
    /// actually onboarded from G2. `aws` drops at the end →
    /// `SwapInHandle` drops → pinned G2 blocks release to kvbm-engine's
    /// inactive pool.
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
            unique
                .iter()
                .zip(plhs.iter())
                .zip(local_hashes.iter())
                .skip(skip)
                .take(count)
                .filter_map(|((block, plh), local)| match block {
                    UniqueBlock::FullBlock(seq_hash) => Some((*seq_hash, *plh, *local)),
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
        let outcome = self.kv_manager.register_swapped_in_blocks(
            &entries,
            parent_hash,
            aws.destination_slots,
        );
        debug_assert_eq!(
            outcome.consumed_entries,
            entries.len(),
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
    fn try_park_for_swap_in(&mut self, uuid: Uuid, now_ms: f64) -> SwapInAdmissionAttempt {
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
        let cost = self.kv_manager.get_prefill_cost(&request.sequence);
        let block_size = request.sequence.block_size();
        let skip_blocks = cost.cached_tokens / block_size;
        let plhs = request.sequence.positional_lineage_hashes();
        if skip_blocks >= plhs.len() {
            return SwapInAdmissionAttempt::NoHit;
        }
        let remaining_plhs = &plhs[skip_blocks..];
        if remaining_plhs.is_empty() {
            return SwapInAdmissionAttempt::NoHit;
        }
        let (handle, destination_slots) = match self
            .kv_manager
            .try_batch_swap_in(remaining_plhs, Some(now_ms))
        {
            BatchSwapInOutcome::Scheduled {
                handle,
                destination_slots,
            } => (handle, destination_slots),
            BatchSwapInOutcome::BlockedOnG1Offload => {
                return SwapInAdmissionAttempt::BlockedOnG1Offload;
            }
            BatchSwapInOutcome::NoHits => return SwapInAdmissionAttempt::NoHit,
        };
        self.state.remove_from_waiting(uuid);
        self.requests_awaiting_swap_in.push(AwaitingSwapIn {
            uuid,
            handle,
            destination_slots,
            skip_blocks,
        });
        SwapInAdmissionAttempt::Parked
    }

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
        let mut batch_count = 0usize;
        let mut batch_total_isl = 0usize;
        let mut batch_total_prefix = 0usize;
        let mut admissions = Vec::new();
        let mut preempted_any = false;

        let mut req_index = 0usize;
        while req_index < self.state.running.len() && token_budget > 0 {
            let uuid = self.state.running[req_index];
            match self.schedule_request(
                uuid,
                false,
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
                ScheduleOutcome::Blocked => break,
                ScheduleOutcome::CurrentPreempted => {}
            }
        }

        let max_num_running = self.args.max_num_seqs.unwrap_or(usize::MAX);
        while !preempted_any && self.state.running.len() < max_num_running {
            let Some(uuid) = self.state.next_waiting_uuid() else {
                break;
            };
            #[cfg(feature = "kvbm-offload")]
            match self.try_park_for_swap_in(uuid, now_ms) {
                SwapInAdmissionAttempt::Parked => continue,
                SwapInAdmissionAttempt::BlockedOnG1Offload => break,
                SwapInAdmissionAttempt::NoHit => {}
            }
            match self.schedule_request(
                uuid,
                true,
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
                ScheduleOutcome::Blocked | ScheduleOutcome::CurrentPreempted => break,
            }
        }

        let prefill_time =
            predict_prefill_duration(batch_count, batch_total_isl, batch_total_prefix, &self.args);
        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let (decode_time, output_signals) = self.emit_ready_tokens(collector, decode_start_ms);
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
        if end_ms <= now_ms {
            if let Some(deadline) = self.kv_manager.earliest_offload_deadline() {
                end_ms = deadline.max(now_ms);
            }
        }

        let fpm = self.compute_fpm(&scheduled, (end_ms - now_ms) / 1000.0);

        debug_assert_vllm_scheduler_state(&self.state);
        EnginePassResult {
            end_ms,
            completed_requests: requests_before.saturating_sub(self.state.requests.len()),
            output_signals,
            admissions,
            active_decode_blocks: self.kv_manager.num_active_blocks() as u64,
            router_event_visibility: RouterEventVisibility::PassStart,
            kv_events: self
                .kv_event_buffer
                .as_ref()
                .map(CapturedRouterEventBuffer::drain)
                .unwrap_or_default(),
            fpm: Some(fpm),
        }
    }

    pub(super) fn drop_request(&mut self, uuid: Uuid) {
        let Some(request) = self.state.requests.get(&uuid) else {
            return;
        };
        for signal in request.sequence.free_signal() {
            self.kv_manager.process(&signal);
        }
        self.state.complete(&uuid);
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

        let scheduled_decodes = scheduled
            .values()
            .filter_map(|work| (work.prompt_tokens == 0).then_some(work.sequence_len as u64));

        let queued_prefills = self.state.waiting.iter().filter_map(|uuid| {
            let request = self.state.requests.get(uuid)?;
            matches!(request.status, RequestStatus::Waiting)
                .then_some(request.sequence.num_input_tokens() as u64)
        });

        let queued_decodes = self.state.waiting.iter().filter_map(|uuid| {
            let request = self.state.requests.get(uuid)?;
            matches!(request.status, RequestStatus::Preempted).then_some(
                (request.sequence.num_input_tokens() + request.sequence.generated_tokens()) as u64,
            )
        });

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
        debug_assert_vllm_request_invariants(uuid, request);
        let cached_prefix_tokens = if request.num_computed_tokens == 0 {
            self.kv_manager
                .get_prefill_cost(&request.sequence)
                .cached_tokens
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
                    Some((allocation_target, prev_allocated_tokens, maybe_signal))
                }
            };
            let Some((allocation_target, prev_allocated_tokens, maybe_signal)) = allocation else {
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

            let expected = match &signal {
                MoveBlock::Use(blocks, ..) => blocks.len(),
                _ => unreachable!(),
            };
            let allocated = self.kv_manager.process(&signal);
            let (_committed_tokens, current_computed_tokens) = {
                let request = self.state.requests.get_mut(&uuid).unwrap_or_else(|| {
                    panic!("schedule_request: {uuid} removed mid-pass (post-process commit)")
                });
                let committed_tokens = if allocated == expected {
                    allocation_target
                } else {
                    let prev_blocks = prev_allocated_tokens
                        .div_ceil(request.sequence.block_size())
                        .min(request.sequence.unique_blocks().len());
                    (prev_blocks + allocated) * request.sequence.block_size()
                };
                request
                    .sequence
                    .commit_allocation(committed_tokens.min(allocation_target));
                request.num_computed_tokens = actual_computed_after.min(committed_tokens);
                (committed_tokens, request.num_computed_tokens)
            };
            if allocated == expected {
                break;
            }

            let Some(preempted) = self.state.preempt(self.args.preemption_mode) else {
                actual_computed_after = current_computed_tokens;
                break;
            };
            for signal in preempted.signals {
                self.kv_manager.process(&signal);
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
            debug_assert_vllm_request_invariants(uuid, request);
        }
        let tokens_used = actual_computed_after.saturating_sub(effective_computed_before);
        if tokens_used == 0
            && actual_computed_after < request_sequence_len(&self.state.requests, uuid)
        {
            return ScheduleOutcome::Blocked;
        }

        let prompt_after = actual_computed_after.min(prompt_len);
        let prompt_tokens = prompt_after.saturating_sub(prompt_before);
        let sequence_len = self
            .state
            .requests
            .get(&uuid)
            .map(|r| r.sequence.len())
            .unwrap_or(0);
        scheduled.insert(
            uuid,
            ScheduledWork {
                total_tokens: tokens_used,
                prompt_tokens,
                prefix_tokens: prompt_before,
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

    fn emit_ready_tokens(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        decode_start_ms: f64,
    ) -> (Duration, Vec<OutputSignal>) {
        let ready = self
            .state
            .running
            .iter()
            .copied()
            .filter(|uuid| {
                let Some(request) = self.state.requests.get(uuid) else {
                    return false;
                };
                request.num_computed_tokens >= request.sequence.len()
                    && request.sequence.generated_tokens() < request.sequence.max_output_tokens()
            })
            .collect::<Vec<_>>();
        if ready.is_empty() {
            return (Duration::ZERO, Vec::new());
        }

        // For prefill workers, the first decode token is produced as part of
        // the prefill forward pass — no separate decode iteration needed.
        let (decode_time, decode_end_ms) = if self.args.worker_type == WorkerType::Prefill {
            (Duration::ZERO, decode_start_ms)
        } else {
            let active_kv_tokens = self.kv_manager.num_active_blocks() * self.args.block_size;
            let total_kv_tokens = self.args.num_gpu_blocks * self.args.block_size;
            let total_length = ready
                .iter()
                .filter_map(|uuid| self.state.requests.get(uuid))
                .map(|request| request.sequence.len())
                .sum::<usize>();
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

        let mut output_signals = Vec::with_capacity(ready.len());
        for uuid in ready {
            let mut emitted = false;
            let mut completed = false;
            loop {
                debug_assert_vllm_ready_to_decode(&self.state.requests, uuid);
                let Some(sequence) = self.state.running_sequence_mut(uuid) else {
                    break;
                };
                let signals = sequence.generate();
                if process_signals(&mut self.kv_manager, &signals) {
                    if sequence.generated_tokens() < sequence.max_output_tokens() {
                        sequence.commit_allocation(sequence.len());
                    }
                    emitted = true;
                    completed = sequence.generated_tokens() >= sequence.max_output_tokens();
                    break;
                }
                sequence.pop();

                let Some(preempted) = self.state.preempt(self.args.preemption_mode) else {
                    break;
                };
                for signal in preempted.signals {
                    self.kv_manager.process(&signal);
                }
                if preempted.uuid == uuid {
                    break;
                }
            }
            if !emitted {
                continue;
            }

            if let Some(collector) = collector.as_deref_mut() {
                collector.on_token(uuid, decode_end_ms);
            }
            if let Some(request) = self.state.requests.get(&uuid) {
                debug_assert_vllm_request_progress(uuid, request);
                let handoff_delay_ms = compute_prefill_handoff_delay_ms(
                    self.args.worker_type,
                    completed,
                    request.sequence.num_input_tokens(),
                    self.args.kv_transfer_bandwidth,
                    self.args.kv_bytes_per_token,
                );
                output_signals.push(OutputSignal {
                    uuid,
                    completed,
                    handoff_delay_ms,
                });
            } else {
                output_signals.push(OutputSignal {
                    uuid,
                    completed,
                    handoff_delay_ms: None,
                });
            }
            if completed {
                self.state.complete(&uuid);
            }
        }

        if output_signals.is_empty() {
            return (Duration::ZERO, output_signals);
        }

        self.state.compact_running();
        (decode_time, output_signals)
    }
}

fn request_sequence_len(requests: &FxHashMap<Uuid, VllmRequestState>, uuid: Uuid) -> usize {
    requests
        .get(&uuid)
        .map(|request| request.sequence.len())
        .unwrap_or_default()
}

fn debug_assert_vllm_request_invariants(_uuid: Uuid, _request: &VllmRequestState) {
    #[cfg(debug_assertions)]
    {
        let uuid = _uuid;
        let request = _request;
        let seq_len = request.sequence.len();
        let allocated = request.sequence.num_allocated_tokens();
        debug_assert!(
            request.num_computed_tokens <= seq_len,
            "request {uuid} computed {} tokens but sequence length is {seq_len}",
            request.num_computed_tokens
        );
        debug_assert!(
            allocated <= seq_len,
            "request {uuid} allocated {allocated} tokens but sequence length is {seq_len}"
        );
    }
}

fn debug_assert_vllm_request_progress(_uuid: Uuid, _request: &VllmRequestState) {
    #[cfg(debug_assertions)]
    {
        let uuid = _uuid;
        let request = _request;
        debug_assert_vllm_request_invariants(uuid, request);
        let allocated = request.sequence.num_allocated_tokens();
        debug_assert!(
            allocated >= request.num_computed_tokens,
            "request {uuid} allocated {allocated} tokens but computed {}",
            request.num_computed_tokens
        );
    }
}

fn debug_assert_vllm_ready_to_decode(_requests: &FxHashMap<Uuid, VllmRequestState>, _uuid: Uuid) {
    #[cfg(debug_assertions)]
    {
        let requests = _requests;
        let uuid = _uuid;
        let Some(request) = requests.get(&uuid) else {
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

fn debug_assert_vllm_scheduler_state(_state: &SchedulerState) {
    #[cfg(debug_assertions)]
    {
        let state = _state;
        let mut seen = std::collections::HashSet::new();
        for uuid in &state.waiting_members {
            debug_assert!(
                seen.insert(*uuid),
                "request {uuid} appears multiple times across waiting/running queues"
            );
            let request = state
                .requests
                .get(uuid)
                .expect("waiting request missing from state map");
            debug_assert!(
                request.status != RequestStatus::Running,
                "request {uuid} is queued in waiting but marked Running"
            );
            debug_assert_vllm_request_invariants(*uuid, request);
        }
        for uuid in &state.running_members {
            debug_assert!(
                seen.insert(*uuid),
                "request {uuid} appears multiple times across waiting/running queues"
            );
            let request = state
                .requests
                .get(uuid)
                .expect("running request missing from state map");
            debug_assert_eq!(
                request.status,
                RequestStatus::Running,
                "request {uuid} is queued in running but marked {:?}",
                request.status
            );
            debug_assert_vllm_request_invariants(*uuid, request);
        }
        debug_assert!(
            state.waiting.len() >= state.waiting_members.len(),
            "waiting queue dropped live membership entries"
        );
        debug_assert!(
            state.running.len() >= state.running_members.len(),
            "running queue dropped live membership entries"
        );
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

fn process_signals(kv_manager: &mut KvManager, signals: &[MoveBlock]) -> bool {
    for signal in signals {
        if kv_manager.process(signal) > 0 {
            continue;
        }

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
        return false;
    }
    true
}

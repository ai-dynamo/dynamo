// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use uuid::Uuid;

use crate::common::handoff::HandoffId;
use crate::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs, WorkerType};
use crate::common::speculative::{SpeculativeDecodeSampler, normalize_conditional_accept_rates};
use crate::common::utils::prefill_handoff_transfer_timing;
use crate::kv_manager::SglangKvManager;
use crate::kv_manager::sglang_backend::SglangDestinationReservation;
use crate::replay::TraceCollector;

use super::config::SglangConfig;
use super::decode::{
    cache_materialized_prefix, cleanup_completed_request, simulate_decode_step_with_sampler,
};
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::SglangRequest;
use crate::scheduler::{
    ActiveHandoffRequests, AdmissionInvariant, AdmissionStage, CapturedRouterEventBuffer,
    DestinationHolds, EnginePassResult, MockerMetrics, PendingDestinations, RemovedSource,
    RouterEventVisibility, SchedulerCommand, SchedulerCommandEffects, SchedulerCommandResult,
    SchedulerLifecycleEvent, SourceCompletion, SourceHolds, accept_length_sample,
    build_fpm_snapshot, capture_router_event_sink,
};

pub(crate) struct SglangCore {
    pub(super) config: SglangConfig,
    dp_rank: u32,
    pub(super) waiting: VecDeque<SglangRequest>,
    prebuilt_ready: VecDeque<SglangRequest>,
    pub(super) running: Vec<SglangRequest>,
    pub(super) new_token_ratio: f64,
    pub(super) kv_manager: SglangKvManager,
    speculative_sampler: Option<SpeculativeDecodeSampler>,
    kv_event_buffer: Option<CapturedRouterEventBuffer>,
    source_holds: SourceHolds<HeldSglangPrefill>,
    pending_destinations: PendingDestinations<SglangRequest>,
    destination_holds: DestinationHolds<ReservedSglangDecode>,
    active_destination_handoffs: ActiveHandoffRequests,
    capacity_generation: u64,
    #[cfg(test)]
    destination_reservation_attempts: usize,
    lifecycle_events: Vec<SchedulerLifecycleEvent>,
}

struct HeldSglangPrefill {
    request: SglangRequest,
}

struct ReservedSglangDecode {
    request: SglangRequest,
    kv: SglangDestinationReservation,
}

impl ReservedSglangDecode {
    fn activate(self, kv_manager: &mut SglangKvManager, block_size: usize) -> SglangRequest {
        let Self { mut request, kv } = self;
        let allocated_tokens = kv.allocated_tokens;
        let alloc = kv_manager.activate_destination(kv, request.prompt_tokens());
        request.kv_lease = alloc.lease;
        request.materialized_tokens = request.prompt_len();
        request.allocated_tokens = allocated_tokens;
        request.debug_assert_invariants(block_size);
        request
    }

    fn cancel(self, kv_manager: &mut SglangKvManager) {
        let Self { request: _, kv } = self;
        kv_manager.cancel_destination(kv);
    }
}

impl SglangCore {
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
        let config = SglangConfig::from_args(&args);
        let total_tokens = args.num_gpu_blocks * args.block_size;
        let speculative_sampler = args.aic_nextn.map(|nextn| {
            let rates =
                normalize_conditional_accept_rates(nextn, args.aic_nextn_accept_rates.as_deref())
                    .expect("normalized MTP acceptance rates");
            SpeculativeDecodeSampler::new(rates, args.aic_mtp_seed.wrapping_add(seed_offset))
        });

        Self {
            config,
            dp_rank,
            waiting: VecDeque::new(),
            prebuilt_ready: VecDeque::new(),
            running: Vec::new(),
            new_token_ratio: SglangConfig::from_args(&args).init_new_token_ratio,
            kv_manager: SglangKvManager::new(
                total_tokens,
                args.block_size,
                kv_event_publishers,
                dp_rank,
            ),
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
        }
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
                let result = if self.cancel_active_request(request_id) {
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                if allow_destination_admission {
                    Ok(self.effects_after_capacity_change(result))
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
                Ok(self.effects_after_capacity_change(result))
            }
            SchedulerCommand::CancelSource { handoff_id } => {
                let result = if self.cancel_source(handoff_id) {
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                };
                Ok(self.effects_after_capacity_change(result))
            }
            SchedulerCommand::ReserveDestination {
                handoff_id,
                mut request,
            } => {
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
                let request = SglangRequest::from(request);
                let prompt_footprint = request
                    .prompt_len()
                    .div_ceil(self.config.block_size)
                    .saturating_mul(self.config.block_size);
                if prompt_footprint > self.config.total_kv_tokens {
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
                        .extend(self.retry_pending_destinations());
                }
                Ok(effects)
            }
            SchedulerCommand::ActivateDestination { handoff_id } => {
                let Some((_, reservation)) = self.destination_holds.remove(handoff_id) else {
                    return Ok(SchedulerCommandEffects::new(SchedulerCommandResult::Noop));
                };
                let available_before = self.kv_manager.cache().available_tokens();
                let request = reservation.activate(&mut self.kv_manager, self.config.block_size);
                self.active_destination_handoffs
                    .insert(handoff_id, request.uuid);
                self.prebuilt_ready.push_back(request);
                if self.kv_manager.cache().available_tokens() > available_before {
                    self.bump_capacity_generation();
                }
                Ok(self.effects_after_capacity_change(SchedulerCommandResult::Applied))
            }
            SchedulerCommand::CancelDestination { handoff_id } => {
                if self.pending_destinations.remove(handoff_id).is_some() {
                    self.bump_capacity_generation();
                    return Ok(self.effects_after_capacity_change(SchedulerCommandResult::Applied));
                }
                if let Some((_, reservation)) = self.destination_holds.remove(handoff_id) {
                    reservation.cancel(&mut self.kv_manager);
                    self.bump_capacity_generation();
                    return Ok(self.effects_after_capacity_change(SchedulerCommandResult::Applied));
                }
                let Some(request_id) = self.active_destination_handoffs.remove_handoff(handoff_id)
                else {
                    return Ok(SchedulerCommandEffects::new(SchedulerCommandResult::Noop));
                };
                self.cancel_active_request(request_id);
                Ok(self.effects_after_capacity_change(SchedulerCommandResult::Applied))
            }
        }
    }

    fn effects_after_capacity_change(
        &mut self,
        result: SchedulerCommandResult,
    ) -> SchedulerCommandEffects {
        let mut effects = SchedulerCommandEffects::new(result);
        if result == SchedulerCommandResult::Applied {
            effects
                .lifecycle_events
                .extend(self.retry_pending_destinations());
        }
        effects
    }

    pub(crate) fn retry_pending_destinations(&mut self) -> Vec<SchedulerLifecycleEvent> {
        let generation = self.capacity_generation;
        let Some((_, _, request)) = self.pending_destinations.front_due(generation) else {
            return Vec::new();
        };
        // TODO(disagg): Real SGLang also preserves logical decode headroom
        // (`num_reserved_decode_tokens`, default 512, plus a one-request
        // completion guard). This foundation physically reserves only the
        // page-rounded incoming prompt footprint.
        #[cfg(test)]
        {
            self.destination_reservation_attempts += 1;
        }
        let reservation = self.kv_manager.reserve_destination(request.prompt_tokens());
        self.pending_destinations.mark_front_attempted(generation);
        let Some(kv) = reservation else {
            return Vec::new();
        };
        let transferable_prompt_tokens = kv.transferable_prompt_tokens();
        let (handoff_id, request_id, request) = self
            .pending_destinations
            .pop_front()
            .expect("attempted pending destination must remain at the head");
        self.destination_holds
            .insert(request_id, handoff_id, ReservedSglangDecode { request, kv });
        vec![SchedulerLifecycleEvent::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        }]
    }

    fn validate_request_id(&self, uuid: Uuid) -> anyhow::Result<()> {
        if self.request_is_active(uuid)
            || self.source_holds.contains_request(uuid)
            || self.pending_destinations.contains_request(uuid)
            || self.destination_holds.contains_request(uuid)
            || self.active_destination_handoffs.contains_request(uuid)
        {
            anyhow::bail!("request {uuid} is already active");
        }
        Ok(())
    }

    fn request_is_active(&self, uuid: Uuid) -> bool {
        self.waiting.iter().any(|request| request.uuid == uuid)
            || self
                .prebuilt_ready
                .iter()
                .any(|request| request.uuid == uuid)
            || self.running.iter().any(|request| request.uuid == uuid)
    }

    fn submit(&mut self, request: DirectRequest) -> anyhow::Result<Uuid> {
        let request = SglangRequest::from(request);
        if self.request_is_active(request.uuid) {
            anyhow::bail!("request {} is already active", request.uuid);
        }
        request.debug_assert_invariants(self.config.block_size);
        let uuid = request.uuid;
        self.waiting.push_back(request);
        Ok(uuid)
    }

    fn complete_source(&mut self, request: SglangRequest) {
        let uuid = request.uuid;
        let transfer_timing = prefill_handoff_transfer_timing(
            request.prompt_len(),
            self.config.kv_transfer_bandwidth,
            self.config.kv_bytes_per_token,
            self.config.kv_transfer_timing_mode,
        );
        let payload = HeldSglangPrefill { request };
        let released = match self.source_holds.complete_source(uuid, payload) {
            SourceCompletion::Release(payload) => {
                self.cleanup_completed_prefill(payload);
                true
            }
            SourceCompletion::Held { handoff_id } => {
                self.lifecycle_events
                    .push(SchedulerLifecycleEvent::SourceHeld {
                        handoff_id,
                        request_id: uuid,
                        transfer_timing,
                    });
                false
            }
        };
        self.active_destination_handoffs.remove_request(uuid);
        if released {
            self.bump_capacity_generation();
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
                self.cancel_active_request(request_id);
                true
            }
            RemovedSource::Missing => false,
        }
    }

    fn cancel_active_request(&mut self, request_id: Uuid) -> bool {
        let request = if let Some(index) = self
            .waiting
            .iter()
            .position(|request| request.uuid == request_id)
        {
            self.waiting.remove(index)
        } else if let Some(index) = self
            .prebuilt_ready
            .iter()
            .position(|request| request.uuid == request_id)
        {
            self.prebuilt_ready.remove(index)
        } else if let Some(index) = self
            .running
            .iter()
            .position(|request| request.uuid == request_id)
        {
            Some(self.running.remove(index))
        } else {
            None
        };
        let Some(mut request) = request else {
            return false;
        };
        let capacity_improved = self.kv_manager.abort(std::mem::take(&mut request.kv_lease));
        self.source_holds.remove_request(request_id);
        self.active_destination_handoffs.remove_request(request_id);
        if capacity_improved {
            self.bump_capacity_generation();
        }
        true
    }

    fn cleanup_completed_prefill(&mut self, payload: HeldSglangPrefill) {
        let mut request = payload.request;
        cleanup_completed_request(&mut request, &mut self.kv_manager, self.config.block_size);
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
        self.waiting.is_empty() && self.prebuilt_ready.is_empty() && self.running.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn is_drained(&self) -> bool {
        self.is_empty()
            && self.source_holds.is_empty()
            && self.pending_destinations.is_empty()
            && self.destination_holds.is_empty()
            && self.active_destination_handoffs.is_empty()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.waiting.len() + self.prebuilt_ready.len() + self.running.len()
    }

    pub(crate) fn mocker_metrics(&self) -> MockerMetrics {
        self.mocker_metrics_with_cache(0, 0)
    }

    fn mocker_metrics_with_cache(
        &self,
        sglang_cache_hit_tokens: u64,
        sglang_cache_total_tokens: u64,
    ) -> MockerMetrics {
        let preactivation_destinations =
            self.pending_destinations.len() + self.destination_holds.len();
        MockerMetrics::from_parts(
            self.dp_rank,
            self.active_kv_blocks(),
            self.config.total_kv_tokens.div_ceil(self.config.block_size) as u64,
            self.running.len() as u64,
            (self.waiting.len() + self.prebuilt_ready.len() + preactivation_destinations) as u64,
            0,
            sglang_cache_hit_tokens,
            sglang_cache_total_tokens,
        )
    }

    #[cfg(test)]
    pub(crate) fn destination_is_held(&self, handoff_id: HandoffId) -> bool {
        self.destination_holds.contains(handoff_id)
            || self.pending_destinations.contains_handoff(handoff_id)
    }

    #[cfg(test)]
    pub(crate) fn destination_indices(&self, handoff_id: HandoffId) -> Vec<usize> {
        self.destination_holds
            .get(handoff_id)
            .map(|reservation| reservation.kv.indices())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(super) fn prebuilt_request(&self, uuid: Uuid) -> Option<&SglangRequest> {
        self.prebuilt_ready
            .iter()
            .find(|request| request.uuid == uuid)
    }

    fn bump_capacity_generation(&mut self) {
        self.capacity_generation = self
            .capacity_generation
            .checked_add(1)
            .expect("destination capacity generation overflow");
    }

    pub(crate) fn drain_kv_events(&self) -> Vec<dynamo_kv_router::protocols::RouterEvent> {
        self.kv_event_buffer
            .as_ref()
            .map(CapturedRouterEventBuffer::drain)
            .unwrap_or_default()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.execute_pass_internal(Some(collector), now_ms)
    }

    pub(crate) fn execute_hidden_pass(&mut self, now_ms: f64) -> EnginePassResult {
        self.execute_pass_internal(None, now_ms)
    }

    pub(super) fn execute_pass_internal(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        now_ms: f64,
    ) -> EnginePassResult {
        let mut admissions = self.promote_prebuilt_ready();
        let materialized_waiting = !self.prebuilt_ready.is_empty();
        apply_schedule_policy(&mut self.waiting, &self.kv_manager, &self.config);

        let admission = AdmissionInvariant::new(self.pending_destinations.has_pending());
        let mut admit = match admission.stage_for(materialized_waiting) {
            AdmissionStage::Materialized | AdmissionStage::PendingDestinationHead => {
                Default::default()
            }
            AdmissionStage::FreshKv => get_new_batch_prefill(
                &mut self.waiting,
                &mut self.kv_manager,
                &self.config,
                self.new_token_ratio,
                &self.running,
            ),
        };

        if admit.oom {
            self.new_token_ratio = self.config.init_new_token_ratio;
        }

        admissions.append(&mut admit.admissions);
        for admission in &admissions {
            if let Some(collector) = collector.as_deref_mut() {
                collector.on_admit(admission.uuid, now_ms, admission.reused_input_tokens);
            }
        }

        // Capture per-request prefill FPM data before dispersing can_run.
        let prefill_fpm = admit.prefill_fpm;

        let batch_size = admit.can_run.len();
        let mean_isl = admit.total_isl.checked_div(batch_size).unwrap_or(0);
        let mean_prefix = admit.total_prefix.checked_div(batch_size).unwrap_or(0);
        let prefill_time =
            simulate_prefill_duration(batch_size, mean_isl, mean_prefix, &self.config, true);

        for mut req in admit.can_run {
            if req.materialized_tokens < req.current_sequence_len() {
                cache_materialized_prefix(&mut req, &mut self.kv_manager, &self.config);
                self.waiting.push_front(req);
            } else {
                self.running.push(req);
            }
        }

        // Capture scheduled decode data before the decode step modifies running.
        let scheduled_decode_lens: Vec<u64> = self
            .running
            .iter()
            .filter(|req| req.remaining_output_tokens() > 0)
            .map(|req| req.current_sequence_len() as u64)
            .collect();

        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let mut decode = simulate_decode_step_with_sampler(
            &mut self.running,
            &mut self.kv_manager,
            &self.config,
            self.speculative_sampler.as_mut(),
            decode_start_ms,
            true,
        );

        for request in decode.completed_requests.drain(..) {
            self.complete_source(request);
        }

        if let Some(collector) = collector {
            for signal in &decode.output_signals {
                if signal.token_id.is_some() {
                    collector.on_token(signal.uuid, decode.end_ms);
                }
            }
        }

        for req in decode.requests.drain(..).rev() {
            self.waiting.push_front(req);
        }

        if decode.retracted_any {
            self.new_token_ratio = self.config.init_new_token_ratio;
            self.bump_capacity_generation();
        }
        self.new_token_ratio = (self.new_token_ratio - self.config.new_token_ratio_decay_step)
            .max(self.config.min_new_token_ratio);

        // Build FPM snapshot now that all state has settled.
        let sglang_cache_hit_tokens = prefill_fpm
            .iter()
            .map(|item| item.prefix_tokens as u64)
            .sum::<u64>();
        let sglang_cache_total_tokens = prefill_fpm
            .iter()
            .map(|item| (item.prefix_tokens + item.tokens_computed) as u64)
            .sum::<u64>();
        let queued_prefills = self
            .waiting
            .iter()
            .filter(|request| {
                request.output_len() == 0
                    && !self
                        .active_destination_handoffs
                        .contains_request(request.uuid)
            })
            .map(|request| request.prompt_len() as u64);
        let ordinary_queued_decodes = self
            .waiting
            .iter()
            .filter(|request| {
                request.output_len() > 0
                    || self
                        .active_destination_handoffs
                        .contains_request(request.uuid)
            })
            .map(|request| request.current_sequence_len() as u64)
            .chain(
                self.prebuilt_ready
                    .iter()
                    .map(|request| request.current_sequence_len() as u64),
            );
        let preactivation_decodes = self
            .pending_destinations
            .payloads()
            .map(|request| request.prompt_len() as u64)
            .chain(
                self.destination_holds
                    .payloads()
                    .map(|reservation| reservation.request.prompt_len() as u64),
            );
        let fpm = build_fpm_snapshot(
            prefill_fpm
                .iter()
                .filter(|p| p.tokens_computed > 0)
                .map(|p| {
                    (
                        p.prompt_len as u64,
                        p.prefix_tokens as u64,
                        p.tokens_computed as u64,
                    )
                }),
            scheduled_decode_lens.into_iter(),
            queued_prefills,
            ordinary_queued_decodes.chain(preactivation_decodes),
            (decode.end_ms - now_ms) / 1000.0,
        );

        let (accept_length_output_tokens, accept_length_decode_forwards) =
            accept_length_sample(&decode.output_signals);
        debug_assert_sglang_scheduler_state(&self.waiting, &self.running, self.config.block_size);
        EnginePassResult {
            end_ms: decode.end_ms,
            completed_requests: decode
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count(),
            output_signals: decode.output_signals,
            admissions,
            lifecycle_events: std::mem::take(&mut self.lifecycle_events),
            mocker_metrics: self
                .mocker_metrics_with_cache(sglang_cache_hit_tokens, sglang_cache_total_tokens),
            router_event_visibility: RouterEventVisibility::PassEnd,
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

    fn active_kv_blocks(&self) -> u64 {
        let actual_used =
            self.kv_manager.cache().total_tokens() - self.kv_manager.cache().available_tokens();
        actual_used.div_ceil(self.config.block_size) as u64
    }

    fn promote_prebuilt_ready(&mut self) -> Vec<crate::scheduler::AdmissionEvent> {
        let mut admissions = Vec::new();
        while self.running.len() < self.config.max_running_requests {
            let Some(request) = self.prebuilt_ready.pop_front() else {
                break;
            };
            admissions.push(crate::scheduler::AdmissionEvent {
                uuid: request.uuid,
                reused_input_tokens: 0,
            });
            self.running.push(request);
        }
        admissions
    }
}

fn simulate_prefill_duration(
    batch_size: usize,
    mean_isl: usize,
    mean_prefix: usize,
    config: &SglangConfig,
    apply_speedup: bool,
) -> Duration {
    if batch_size == 0 || config.worker_type == WorkerType::Decode {
        return Duration::ZERO;
    }

    let prefill_time = config
        .perf_model
        .predict_prefill_time(batch_size, mean_isl, mean_prefix);
    let total_time = Duration::from_secs_f64(prefill_time / 1000.0);

    if !apply_speedup || config.speedup_ratio <= 0.0 || total_time <= Duration::ZERO {
        return total_time;
    }

    Duration::from_secs_f64(total_time.as_secs_f64() / config.speedup_ratio)
}

fn debug_assert_sglang_scheduler_state(
    _waiting: &VecDeque<SglangRequest>,
    _running: &[SglangRequest],
    _block_size: usize,
) {
    #[cfg(debug_assertions)]
    {
        let waiting = _waiting;
        let running = _running;
        let block_size = _block_size;
        let mut seen = std::collections::HashSet::new();
        for req in waiting {
            debug_assert!(
                seen.insert(req.uuid),
                "request {} appears multiple times across waiting/running queues",
                req.uuid
            );
            req.debug_assert_invariants(block_size);
        }
        for req in running {
            debug_assert!(
                seen.insert(req.uuid),
                "request {} appears multiple times across waiting/running queues",
                req.uuid
            );
            req.debug_assert_invariants(block_size);
        }
    }
}

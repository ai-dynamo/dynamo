// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use uuid::Uuid;

use crate::common::handoff::HandoffId;
use crate::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs, WorkerType};
use crate::common::speculative::{SpeculativeDecodeSampler, normalize_conditional_accept_rates};
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
    CapturedRouterEventBuffer, DestinationHolds, EnginePassResult, MockerMetrics, RemovedSource,
    RouterEventVisibility, SchedulerCommand, SchedulerCommandResult, SourceCompletion, SourceHolds,
    accept_length_sample, build_fpm_snapshot, capture_router_event_sink,
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
    destination_holds: DestinationHolds<ReservedSglangDecode>,
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
        let prompt_tokens = request.prompt_tokens.clone();
        let alloc = kv_manager.activate_destination(kv, &prompt_tokens);
        request.last_node = Some(alloc.last_node);
        request.kv_indices = alloc.kv_indices;
        request.materialized_tokens = request.prompt_len();
        request.cached_tokens = request.page_aligned_materialized_tokens(block_size);
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

    pub(crate) fn new_with_worker_id(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        Self::new_internal(args, 0, worker_id, None, KvEventPublishers::default())
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        let (buffer, sink) = capture_router_event_sink(worker_id);
        Self::new_internal(
            args,
            0,
            worker_id,
            Some(buffer),
            KvEventPublishers::new(Some(sink), None),
        )
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
        worker_id: WorkerId,
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
            SpeculativeDecodeSampler::new(rates, args.aic_mtp_seed.wrapping_add(worker_id))
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
            destination_holds: DestinationHolds::default(),
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
        match command {
            SchedulerCommand::Submit(mut request) => {
                let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
                request.uuid = Some(uuid);
                self.validate_request_id(uuid)?;
                Ok(SchedulerCommandResult::Submitted(self.submit(request)?))
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
                Ok(SchedulerCommandResult::Submitted(submitted))
            }
            SchedulerCommand::ReleaseSource { handoff_id }
            | SchedulerCommand::CancelSource { handoff_id } => {
                Ok(if self.remove_source(handoff_id) {
                    SchedulerCommandResult::Applied
                } else {
                    SchedulerCommandResult::Noop
                })
            }
            SchedulerCommand::ReserveDestination {
                handoff_id,
                mut request,
            } => {
                let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
                request.uuid = Some(uuid);
                self.validate_request_id(uuid)?;
                self.destination_holds.validate(uuid, handoff_id)?;
                let request = SglangRequest::from(request);
                let Some(kv) = self.kv_manager.reserve_destination(&request.prompt_tokens) else {
                    return Ok(SchedulerCommandResult::DestinationUnavailable);
                };
                self.destination_holds.insert(
                    uuid,
                    handoff_id,
                    ReservedSglangDecode { request, kv },
                );
                Ok(SchedulerCommandResult::DestinationReserved { request_id: uuid })
            }
            SchedulerCommand::ActivateDestination { handoff_id } => {
                let Some((_, reservation)) = self.destination_holds.remove(handoff_id) else {
                    return Ok(SchedulerCommandResult::Noop);
                };
                let request = reservation.activate(&mut self.kv_manager, self.config.block_size);
                self.prebuilt_ready.push_back(request);
                Ok(SchedulerCommandResult::Applied)
            }
            SchedulerCommand::CancelDestination { handoff_id } => {
                let Some((_, reservation)) = self.destination_holds.remove(handoff_id) else {
                    return Ok(SchedulerCommandResult::Noop);
                };
                reservation.cancel(&mut self.kv_manager);
                Ok(SchedulerCommandResult::Applied)
            }
        }
    }

    fn validate_request_id(&self, uuid: Uuid) -> anyhow::Result<()> {
        if self.request_is_active(uuid)
            || self.source_holds.contains_request(uuid)
            || self.destination_holds.contains_request(uuid)
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
        let payload = HeldSglangPrefill { request };
        if let SourceCompletion::Release(payload) = self.source_holds.complete_source(uuid, payload)
        {
            self.cleanup_completed_prefill(payload);
        }
    }

    fn remove_source(&mut self, handoff_id: HandoffId) -> bool {
        match self.source_holds.remove(handoff_id) {
            RemovedSource::Held(payload) => {
                self.cleanup_completed_prefill(payload);
                true
            }
            RemovedSource::Pending => true,
            RemovedSource::Missing => false,
        }
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

    pub(crate) fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.prebuilt_ready.is_empty() && self.running.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn is_drained(&self) -> bool {
        self.is_empty() && self.source_holds.is_empty() && self.destination_holds.is_empty()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.waiting.len() + self.prebuilt_ready.len() + self.running.len()
    }

    #[cfg(test)]
    pub(crate) fn destination_is_held(&self, handoff_id: HandoffId) -> bool {
        self.destination_holds.contains(handoff_id)
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

    #[cfg(test)]
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
        apply_schedule_policy(&mut self.waiting, &self.kv_manager, &self.config);

        let mut admit = get_new_batch_prefill(
            &mut self.waiting,
            &mut self.kv_manager,
            &self.config,
            self.new_token_ratio,
            &self.running,
        );

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
        let mean_isl = if batch_size > 0 {
            admit.total_isl / batch_size
        } else {
            0
        };
        let mean_prefix = if batch_size > 0 {
            admit.total_prefix / batch_size
        } else {
            0
        };
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
                collector.on_token(signal.uuid, decode.end_ms);
            }
        }

        for req in decode.requests.drain(..).rev() {
            self.waiting.push_front(req);
        }

        if decode.retracted_any {
            self.new_token_ratio = self.config.init_new_token_ratio;
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
        let fpm = build_fpm_snapshot(
            prefill_fpm.iter().map(|p| {
                (
                    p.prompt_len as u64,
                    p.prefix_tokens as u64,
                    p.tokens_computed as u64,
                )
            }),
            scheduled_decode_lens.into_iter(),
            self.waiting
                .iter()
                .filter(|req| req.output_len() == 0)
                .map(|req| req.prompt_len() as u64),
            self.waiting
                .iter()
                .filter(|req| req.output_len() > 0)
                .map(|req| req.current_sequence_len() as u64),
            (decode.end_ms - now_ms) / 1000.0,
        );

        let (accept_length_output_tokens, accept_length_decode_forwards) =
            accept_length_sample(&decode.output_signals);
        debug_assert_sglang_scheduler_state(&self.waiting, &self.running, self.config.block_size);
        let active_decode_blocks = self.active_kv_blocks();
        EnginePassResult {
            end_ms: decode.end_ms,
            completed_requests: decode
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count(),
            output_signals: decode.output_signals,
            admissions,
            mocker_metrics: MockerMetrics::from_parts(
                self.dp_rank,
                active_decode_blocks,
                self.config.total_kv_tokens.div_ceil(self.config.block_size) as u64,
                self.running.len() as u64,
                (self.waiting.len() + self.prebuilt_ready.len()) as u64,
                0,
                sglang_cache_hit_tokens,
                sglang_cache_total_tokens,
            ),
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
        let active_reserved = self
            .waiting
            .iter()
            .map(SglangRequest::extra_reserved_tokens)
            .sum::<usize>()
            + self
                .prebuilt_ready
                .iter()
                .map(SglangRequest::extra_reserved_tokens)
                .sum::<usize>()
            + self
                .running
                .iter()
                .map(SglangRequest::extra_reserved_tokens)
                .sum::<usize>();
        let actual_used =
            self.kv_manager.cache().total_tokens() - self.kv_manager.cache().available_tokens();
        (actual_used + active_reserved).div_ceil(self.config.block_size) as u64
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

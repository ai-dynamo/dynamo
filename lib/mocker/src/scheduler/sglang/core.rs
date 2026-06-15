// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use uuid::Uuid;

use crate::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs, WorkerType};
use crate::common::speculative::{SpeculativeDecodeSampler, normalize_conditional_accept_rates};
use crate::kv_manager::SglangKvManager;
use crate::replay::TraceCollector;

use super::config::SglangConfig;
use super::decode::{cache_materialized_prefix, simulate_decode_step_with_sampler};
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::SglangRequest;
use crate::scheduler::{
    CapturedRouterEventBuffer, EnginePassResult, MockerMetrics, RouterEventVisibility,
    accept_length_sample, build_fpm_snapshot, capture_router_event_sink,
};

pub(crate) struct SglangCore {
    pub(super) config: SglangConfig,
    dp_rank: u32,
    pub(super) waiting: VecDeque<SglangRequest>,
    pub(super) running: Vec<SglangRequest>,
    pub(super) new_token_ratio: f64,
    pub(super) kv_manager: SglangKvManager,
    speculative_sampler: Option<SpeculativeDecodeSampler>,
    kv_event_buffer: Option<CapturedRouterEventBuffer>,
    /// completed disagg-prefill requests whose KV (kv_indices +
    /// last_node) is *pinned* — retained, not freed — modeling NIXL behavior
    /// where prefill KV stays resident until the matching decode pulls the
    /// transfer. The retained kv_indices keep those slots out of the free pool,
    /// so `active_kv_blocks()` stays elevated (the cascade). Released via
    /// [`SglangCore::release_pinned`].
    pinned: std::collections::HashMap<Uuid, SglangRequest>,
    /// Sum of the token footprints (kv_indices) of the currently pinned
    /// (stranded) prefills. Pinned KV must still occupy the pool for the
    /// cascade (counted in `active_kv_blocks()` used by the admission path),
    /// but it must NOT inflate the *router-facing* load metric
    /// (`active_decode_blocks`): a stranded prefill is awaiting handoff, not
    /// actively decoding, so it should not divert overlap/load routing of
    /// unrelated requests. The router-facing metric discounts this amount.
    pinned_token_footprint: usize,
    /// Replay: release pins on a modeled virtual-time deadline instead
    /// of a live decode event. See the vLLM core for the rationale.
    time_based_pin_release: bool,
    /// uuid -> virtual-time (ms) deadline for time-based pin release.
    pinned_release_at: std::collections::HashMap<Uuid, f64>,
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
            pinned: std::collections::HashMap::new(),
            pinned_token_footprint: 0,
            time_based_pin_release: false,
            pinned_release_at: std::collections::HashMap::new(),
        }
    }

    /// switch this core to modeled time-based pin release (replay).
    pub(crate) fn set_time_based_pin_release(&mut self, enabled: bool) {
        self.time_based_pin_release = enabled;
    }

    /// Replay: release any time-based pins whose deadline has passed.
    fn release_due_pins(&mut self, now_ms: f64) {
        if self.pinned_release_at.is_empty() {
            return;
        }
        let due: Vec<Uuid> = self
            .pinned_release_at
            .iter()
            .filter(|&(_, &deadline)| deadline <= now_ms)
            .map(|(&uuid, _)| uuid)
            .collect();
        for uuid in due {
            self.pinned_release_at.remove(&uuid);
            self.release_pinned(uuid);
        }
    }

    /// Replay: earliest modeled pin-release deadline, if any.
    pub(crate) fn earliest_pin_deadline(&self) -> Option<f64> {
        self.pinned_release_at
            .values()
            .copied()
            .fold(None, |acc, d| Some(acc.map_or(d, |a: f64| a.min(d))))
    }

    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        // Live release: a release-pin control message frees the pinned
        // KV for `uuid` rather than enqueueing a new request.
        if request.release_pin {
            let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
            self.release_pinned(uuid);
            return uuid;
        }
        let request = SglangRequest::from(request);
        request.debug_assert_invariants(self.config.block_size);
        let uuid = request.uuid;
        self.waiting.push_back(request);
        uuid
    }

    /// release a pinned (stranded) prefill's KV. Frees the retained
    /// kv_indices and last_node back to the KV manager and drops the pinned
    /// state. Returns `true` if a pin was held. Idempotent.
    pub(crate) fn release_pinned(&mut self, uuid: Uuid) -> bool {
        let Some(mut req) = self.pinned.remove(&uuid) else {
            return false;
        };
        self.pinned_token_footprint = self
            .pinned_token_footprint
            .saturating_sub(req.kv_indices.len());
        if !req.kv_indices.is_empty() {
            self.kv_manager.free_indices(&req.kv_indices);
        }
        if let Some(last_node) = req.last_node.take() {
            self.kv_manager.free_request(last_node);
        }
        true
    }

    /// Number of currently pinned (stranded) prefill requests. Test/forensics.
    #[cfg(test)]
    pub(crate) fn num_pinned(&self) -> usize {
        self.pinned.len()
    }

    /// True pool occupancy in blocks (capacity / admission view, INCLUDING
    /// pinned/stranded KV). Distinct from the router-facing
    /// `MockerMetrics::active_decode_blocks`, which discounts pinned KV.
    #[cfg(test)]
    pub(crate) fn pool_active_blocks(&self) -> u64 {
        self.active_kv_blocks()
    }

    pub(crate) fn is_empty(&self) -> bool {
        // a worker holding pinned (stranded) KV is not idle.
        self.waiting.is_empty() && self.running.is_empty() && self.pinned.is_empty()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.waiting.len() + self.running.len() + self.pinned.len()
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
        // Replay: release modeled-time strands whose deadline elapsed.
        self.release_due_pins(now_ms);
        apply_schedule_policy(&mut self.waiting, &self.kv_manager, &self.config);

        let admit = get_new_batch_prefill(
            &mut self.waiting,
            &mut self.kv_manager,
            &self.config,
            self.new_token_ratio,
            &self.running,
        );

        if admit.oom {
            self.new_token_ratio = self.config.init_new_token_ratio;
        }

        for admission in &admit.admissions {
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

        if let Some(collector) = collector {
            for signal in &decode.output_signals {
                collector.on_token(signal.uuid, decode.end_ms);
            }
        }

        for req in decode.requests.drain(..).rev() {
            self.waiting.push_front(req);
        }

        // stranded prefills — retain their (unfreed) KV in the pinned
        // set until the matching decode releases it.
        for req in decode.pinned.drain(..) {
            // Replay: schedule a modeled time-based release at decode_end +
            // handoff/transfer delay (the offline path can't observe the live
            // decode pickup).
            if self.time_based_pin_release {
                let handoff = decode
                    .output_signals
                    .iter()
                    .find(|s| s.uuid == req.uuid)
                    .and_then(|s| s.handoff_delay_ms)
                    .unwrap_or(0.0);
                self.pinned_release_at
                    .insert(req.uuid, decode.end_ms + handoff);
            }
            self.pinned_token_footprint += req.kv_indices.len();
            self.pinned.insert(req.uuid, req);
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
        // Router-facing load discounts stranded-prefill KV: the pinned tokens
        // still occupy the pool (cascade) via the KV manager's reservation that
        // the admission path reads, but an awaiting-handoff prefill is not
        // active load and must not divert overlap/load routing of unrelated
        // requests.
        let pinned_blocks =
            (self.pinned_token_footprint as u64).div_ceil(self.config.block_size as u64);
        let active_decode_blocks = self.active_kv_blocks().saturating_sub(pinned_blocks);
        EnginePassResult {
            end_ms: decode.end_ms,
            completed_requests: decode
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count(),
            output_signals: decode.output_signals,
            admissions: admit.admissions,
            mocker_metrics: {
                let mut metrics = MockerMetrics::from_parts(
                    self.dp_rank,
                    active_decode_blocks,
                    self.config.total_kv_tokens.div_ceil(self.config.block_size) as u64,
                    self.running.len() as u64,
                    self.waiting.len() as u64,
                    0,
                    sglang_cache_hit_tokens,
                    sglang_cache_total_tokens,
                );
                // Expose the seq-slot cap for the decode-side admission wait.
                metrics.max_num_seqs = self.config.max_num_seqs;
                metrics
            },
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
                .running
                .iter()
                .map(SglangRequest::extra_reserved_tokens)
                .sum::<usize>();
        let actual_used =
            self.kv_manager.cache().total_tokens() - self.kv_manager.cache().available_tokens();
        (actual_used + active_reserved).div_ceil(self.config.block_size) as u64
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

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use uuid::Uuid;

use crate::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs, WorkerType};
use crate::kv_manager::SglangKvManager;
use crate::replay::TraceCollector;

use super::config::SglangConfig;
use super::decode::{cache_materialized_prefix, simulate_decode_step};
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::{SglangRequest, direct_to_sglang};
use crate::scheduler::{
    CapturedRouterEventBuffer, EnginePassResult, RouterEventVisibility, capture_router_event_sink,
};

pub(crate) struct SglangCore {
    pub(super) config: SglangConfig,
    pub(super) waiting: VecDeque<SglangRequest>,
    pub(super) running: Vec<SglangRequest>,
    pub(super) new_token_ratio: f64,
    pub(super) kv_manager: SglangKvManager,
    kv_event_buffer: Option<CapturedRouterEventBuffer>,
}

impl SglangCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        Self::new_internal(args, 0, None, KvEventPublishers::default())
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        let (buffer, sink) = capture_router_event_sink(worker_id);
        Self::new_internal(
            args,
            worker_id as u32,
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
        let config = SglangConfig::from_args(&args);
        let total_tokens = args.num_gpu_blocks * args.block_size;

        Self {
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            new_token_ratio: SglangConfig::from_args(&args).init_new_token_ratio,
            kv_manager: SglangKvManager::new(
                total_tokens,
                args.block_size,
                kv_event_publishers,
                dp_rank,
            ),
            kv_event_buffer,
        }
    }

    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        let request = direct_to_sglang(request);
        request.debug_assert_invariants(self.config.block_size);
        let uuid = request.uuid;
        self.waiting.push_back(request);
        uuid
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.execute_pass_internal(Some(collector), now_ms)
    }

    pub(super) fn execute_pass_internal(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        now_ms: f64,
    ) -> EnginePassResult {
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

        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let mut decode = simulate_decode_step(
            &mut self.running,
            &mut self.kv_manager,
            &self.config,
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

        if decode.retracted_any {
            self.new_token_ratio = self.config.init_new_token_ratio;
        }
        self.new_token_ratio = (self.new_token_ratio - self.config.new_token_ratio_decay_step)
            .max(self.config.min_new_token_ratio);

        debug_assert_sglang_scheduler_state(&self.waiting, &self.running, self.config.block_size);
        EnginePassResult {
            end_ms: decode.end_ms,
            completed_requests: decode
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count(),
            output_signals: decode.output_signals,
            admissions: admit.admissions,
            active_decode_blocks: self.active_kv_blocks(),
            router_event_visibility: RouterEventVisibility::PassEnd,
            kv_events: self
                .kv_event_buffer
                .as_ref()
                .map(CapturedRouterEventBuffer::drain)
                .unwrap_or_default(),
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
    waiting: &VecDeque<SglangRequest>,
    running: &[SglangRequest],
    block_size: usize,
) {
    #[cfg(debug_assertions)]
    {
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

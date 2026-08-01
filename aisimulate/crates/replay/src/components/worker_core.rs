// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::MockEngineArgs;
use crate::replay::TraceCollector;
use crate::scheduler::{EngineCore, EnginePassResult, SglangCore, VllmCore};

fn record_pass(collector: &mut TraceCollector, pass: &EnginePassResult, now_ms: f64) {
    collector.on_scheduler_pass(pass, now_ms, Some(pass.token_completion_ms));
}

pub(crate) struct ReplayWorkerCore {
    core: EngineCore,
}

impl ReplayWorkerCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        let core = match args.engine_type {
            crate::common::protocols::EngineType::Vllm
            | crate::common::protocols::EngineType::Trtllm => {
                let mut core = VllmCore::new(args);
                Self::init_offload_vllm(&mut core);
                EngineCore::Vllm(core)
            }
            crate::common::protocols::EngineType::Sglang => {
                EngineCore::Sglang(SglangCore::new(args))
            }
        };
        Self { core }
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: u64) -> Self {
        let core = match args.engine_type {
            crate::common::protocols::EngineType::Vllm
            | crate::common::protocols::EngineType::Trtllm => {
                let mut core = VllmCore::new_with_kv_capture(args, worker_id);
                Self::init_offload_vllm(&mut core);
                EngineCore::Vllm(core)
            }
            crate::common::protocols::EngineType::Sglang => {
                EngineCore::Sglang(SglangCore::new_with_kv_capture(args, worker_id))
            }
        };
        Self { core }
    }

    #[cfg(feature = "kvbm-offload")]
    fn init_offload_vllm(core: &mut VllmCore) {
        if let Err(e) = core.init_offload_offline() {
            tracing::error!("kvbm-offload single-worker offline init failed: {e}");
        }
    }

    #[cfg(not(feature = "kvbm-offload"))]
    fn init_offload_vllm(_core: &mut VllmCore) {}

    pub(crate) fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    pub(crate) fn receive(
        &mut self,
        request: crate::common::protocols::DirectRequest,
    ) -> uuid::Uuid {
        self.core.receive(request)
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.core.num_requests()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> anyhow::Result<EnginePassResult> {
        let pass = self.core.try_execute_pass(now_ms)?;
        record_pass(collector, &pass, now_ms);
        Ok(pass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::{ForwardPassSnapshot, OutputSignal};
    use crate::scheduler::vllm::MockerMetrics;
    use crate::scheduler::{AdmissionEvent, RouterEventVisibility};
    use uuid::Uuid;

    #[test]
    fn single_worker_records_rank_local_token_completion() {
        let uuid = Uuid::from_u128(1);
        let mut collector = TraceCollector::default();
        collector.on_arrival(uuid, 0.0, 4, 1);
        let pass = EnginePassResult {
            end_ms: 20.0,
            token_completion_ms: 5.0,
            completed_requests: 1,
            output_signals: vec![OutputSignal {
                uuid,
                token_id: Some(1),
                completed: true,
                rejected: false,
                cached_tokens: None,
                handoff_delay_ms: None,
            }],
            admissions: vec![AdmissionEvent {
                uuid,
                reused_input_tokens: 0,
            }],
            lifecycle_events: Vec::new(),
            mocker_metrics: MockerMetrics::default(),
            router_event_visibility: RouterEventVisibility::PassEnd,
            kv_events: Vec::new(),
            fpm: Some(ForwardPassSnapshot::default()),
            accept_length_output_tokens: 1,
            accept_length_decode_forwards: 1,
        };

        record_pass(&mut collector, &pass, 0.0);

        assert_eq!(collector.request_latencies(uuid), Some((5.0, 0.0)));
    }
}

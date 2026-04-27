// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo LLM integration helpers for agent trace records.

use crate::protocols::common::timing::RequestTracker;
use dynamo_agents::trace::{AgentRequestMetrics, WorkerInfo};

pub(crate) fn request_metrics(
    request_id: String,
    model: String,
    tracker: Option<&RequestTracker>,
) -> AgentRequestMetrics {
    let timing = tracker.map(RequestTracker::get_timing_info);
    let worker = tracker.and_then(|tracker| {
        tracker.get_worker_info().map(|worker| WorkerInfo {
            prefill_worker_id: worker.prefill_worker_id,
            prefill_dp_rank: worker.prefill_dp_rank,
            decode_worker_id: worker.decode_worker_id,
            decode_dp_rank: worker.decode_dp_rank,
        })
    });

    AgentRequestMetrics {
        request_id,
        model,
        input_tokens: tracker.and_then(|tracker| tracker.isl_tokens().map(|v| v as u64)),
        output_tokens: tracker.map(RequestTracker::osl_tokens),
        cached_tokens: tracker.and_then(|tracker| tracker.cached_tokens().map(|v| v as u64)),
        request_received_ms: timing
            .as_ref()
            .map(|timing| timing.request_received_ms)
            .unwrap_or(0),
        ttft_ms: timing.as_ref().and_then(|timing| timing.ttft_ms),
        total_time_ms: timing.as_ref().and_then(|timing| timing.total_time_ms),
        queue_depth: timing
            .as_ref()
            .and_then(|timing| timing.router_queue_depth.map(|v| v as u64)),
        worker,
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use crate::protocols::common::timing::{RequestTracker, WORKER_TYPE_DECODE};

    use super::request_metrics;

    #[test]
    fn test_request_metrics_from_tracker() {
        let tracker = RequestTracker::new();
        tracker.record_isl(128, Some(32));
        tracker.record_osl(5);
        tracker.record_router_queue_depth(3);
        tracker.record_worker(17, Some(2), WORKER_TYPE_DECODE);
        tracker.record_first_token();
        thread::sleep(Duration::from_millis(1));
        tracker.record_finish();

        let metrics = request_metrics(
            "req-1".to_string(),
            "test-model".to_string(),
            Some(&tracker),
        );

        assert_eq!(metrics.request_id, "req-1");
        assert_eq!(metrics.model, "test-model");
        assert_eq!(metrics.input_tokens, Some(128));
        assert_eq!(metrics.output_tokens, Some(5));
        assert_eq!(metrics.cached_tokens, Some(32));
        assert!(metrics.request_received_ms > 0);
        assert!(metrics.ttft_ms.is_some());
        assert!(metrics.total_time_ms.is_some());
        assert_eq!(metrics.queue_depth, Some(3));
        let worker = metrics.worker.expect("worker info should be set");
        assert_eq!(worker.prefill_worker_id, Some(17));
        assert_eq!(worker.prefill_dp_rank, Some(2));
        assert_eq!(worker.decode_worker_id, Some(17));
        assert_eq!(worker.decode_dp_rank, Some(2));
    }

    #[test]
    fn test_request_metrics_without_tracker_is_partial() {
        let metrics = request_metrics("req-1".to_string(), "test-model".to_string(), None);

        assert_eq!(metrics.request_id, "req-1");
        assert_eq!(metrics.model, "test-model");
        assert_eq!(metrics.input_tokens, None);
        assert_eq!(metrics.output_tokens, None);
        assert_eq!(metrics.cached_tokens, None);
        assert_eq!(metrics.request_received_ms, 0);
        assert_eq!(metrics.ttft_ms, None);
        assert_eq!(metrics.total_time_ms, None);
        assert_eq!(metrics.queue_depth, None);
        assert!(metrics.worker.is_none());
    }
}

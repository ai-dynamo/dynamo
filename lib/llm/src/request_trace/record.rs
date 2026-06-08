// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{SystemTime, UNIX_EPOCH};

use crate::agents::trace::AgentReplayMetrics;
use crate::protocols::common::timing::RequestTracker;

use super::{
    RequestTraceEventType, RequestTraceMetrics, RequestTraceRecord, RequestTraceSchema, publish,
};

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

pub(crate) fn emit_request_end(
    request_id: String,
    tracker: &RequestTracker,
    replay: AgentReplayMetrics,
) {
    let request_received_ms = tracker.request_received_epoch_ms();
    let event_time_unix_ms = tracker
        .total_time_ms()
        .map_or_else(unix_time_ms, |elapsed| {
            request_received_ms.saturating_add(elapsed.max(0.0).round() as u64)
        });

    publish(RequestTraceRecord {
        schema: RequestTraceSchema::V1,
        event_type: RequestTraceEventType::RequestEnd,
        event_time_unix_ms,
        request: RequestTraceMetrics {
            request_id,
            request_received_ms,
            output_tokens: tracker.osl_tokens(),
            replay,
        },
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_trace::BUS;

    #[tokio::test]
    async fn emits_tracker_timing_lengths_and_hashes() {
        BUS.init(16);
        let mut rx = BUS.subscribe();
        let tracker = RequestTracker::new();
        tracker.record_osl(7);
        tracker.record_finish();

        emit_request_end(
            "req-1".to_string(),
            &tracker,
            AgentReplayMetrics {
                trace_block_size: 2,
                input_length: 3,
                input_sequence_hashes: vec![11, 22],
            },
        );

        let record = loop {
            let record = rx.recv().await.unwrap();
            if record.request.request_id == "req-1" {
                break record;
            }
        };
        assert_eq!(record.request.request_id, "req-1");
        assert_eq!(record.request.output_tokens, 7);
        assert_eq!(
            record.request.request_received_ms,
            tracker.request_received_epoch_ms()
        );
        assert!(record.event_time_unix_ms >= record.request.request_received_ms);
        assert_eq!(record.request.replay.input_length, 3);
    }
}

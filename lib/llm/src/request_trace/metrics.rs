// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for request-trace sampling decisions.
//!
//! These counters are deliberately recorded before the trace bus fans a record
//! out to sinks. They measure intentional sampling only; queue pressure and
//! sink delivery are separate concerns.

use std::sync::LazyLock;

use dynamo_runtime::metrics::prometheus_names::{frontend_service, name_prefix};
use prometheus::{IntCounterVec, Opts, Registry};

use super::RequestTraceEventType;

const DECISION_LABEL: &str = "decision";
const EVENT_TYPE_LABEL: &str = "event_type";

struct RequestTraceSamplingMetrics {
    records_total: IntCounterVec,
}

impl RequestTraceSamplingMetrics {
    fn new() -> Result<Self, prometheus::Error> {
        Ok(Self {
            records_total: IntCounterVec::new(
                Opts::new(
                    format!(
                        "{}_{}",
                        name_prefix::FRONTEND,
                        frontend_service::REQUEST_TRACE_SAMPLING_RECORDS_TOTAL
                    ),
                    "Request trace records evaluated by the pre-fan-out sampling decision",
                ),
                &[DECISION_LABEL, EVENT_TYPE_LABEL],
            )?,
        })
    }

    fn record(&self, event_type: RequestTraceEventType, retained: bool) {
        let decision = if retained { "retained" } else { "dropped" };
        self.records_total
            .with_label_values(&[decision, event_type.as_str()])
            .inc();
    }

    fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.records_total.clone()))
    }
}

static REQUEST_TRACE_SAMPLING_METRICS: LazyLock<RequestTraceSamplingMetrics> =
    LazyLock::new(|| {
        RequestTraceSamplingMetrics::new().expect("failed to create request trace sampling metrics")
    });

pub(crate) fn record_sampling_decision(event_type: RequestTraceEventType, retained: bool) {
    REQUEST_TRACE_SAMPLING_METRICS.record(event_type, retained);
}

/// Register request-trace sampling metrics with the frontend's local registry.
pub(crate) fn register_request_trace_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    REQUEST_TRACE_SAMPLING_METRICS.register(registry)
}

#[cfg(test)]
mod tests {
    use prometheus::{Encoder, Registry, TextEncoder};

    use super::*;

    #[test]
    fn records_bounded_sampling_decisions() {
        let metrics = RequestTraceSamplingMetrics::new().unwrap();
        metrics.record(RequestTraceEventType::RequestEnd, true);
        metrics.record(RequestTraceEventType::ToolError, false);

        let registry = Registry::new();
        metrics.register(&registry).unwrap();
        let mut output = Vec::new();
        TextEncoder::new()
            .encode(&registry.gather(), &mut output)
            .unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains(
            "dynamo_frontend_request_trace_sampling_records_total{decision=\"retained\",event_type=\"request_end\"} 1"
        ));
        assert!(output.contains(
            "dynamo_frontend_request_trace_sampling_records_total{decision=\"dropped\",event_type=\"tool_error\"} 1"
        ));
    }
}

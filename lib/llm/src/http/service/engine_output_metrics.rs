// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend metrics and request span shared by routes that serve the raw
//! engine output stream instead of the OpenAI postprocessor pipeline.

use std::sync::Arc;

use super::metrics::{HttpQueueGuard, ResponseMetricCollector};
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::{Annotated, common::llm_backend::LLMEngineOutput};

pub(super) fn generate_dispatch_span(request_id: &str, model: &str) -> tracing::Span {
    tracing::info_span!(
        target: "request_span",
        "generate",
        request_id = %request_id,
        model = %model,
        input_tokens = tracing::field::Empty,
        output_tokens = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
        avg_itl_ms = tracing::field::Empty,
        prefill_worker_id = tracing::field::Empty,
        decode_worker_id = tracing::field::Empty,
    )
}

/// Metrics adapter for raw engine streams served by token-in/token-out routes
/// (vLLM `/inference/v1/generate` and native SGLang `/generate`).
///
/// Unlike the OpenAI text endpoints, these routes deliberately bypass the
/// tokenizer/postprocessor pipeline that emits `LLMMetricAnnotation`. Their
/// token IDs are already rendered, so observe the same response metrics from
/// the raw token deltas while leaving tokenizer and media metrics untouched.
pub(super) struct GenerateMetricCollector {
    response: ResponseMetricCollector,
    http_queue: Option<HttpQueueGuard>,
    tracker: Arc<RequestTracker>,
    input_tokens: usize,
    output_tokens: usize,
    worker_info_observed: bool,
}

impl GenerateMetricCollector {
    pub(super) fn new(
        metrics: Arc<super::metrics::Metrics>,
        model: &str,
        tracker: Arc<RequestTracker>,
        input_tokens: usize,
    ) -> Self {
        Self {
            response: metrics.clone().create_response_collector(model),
            http_queue: Some(metrics.create_http_queue_guard(model)),
            tracker,
            input_tokens,
            output_tokens: 0,
            worker_info_observed: false,
        }
    }

    /// Copy routed worker labels once, after the tracker has populated them.
    fn observe_worker_info(&mut self) {
        if self.worker_info_observed {
            return;
        }

        let Some(worker) = self.tracker.get_worker_info() else {
            return;
        };
        self.response.set_worker_info(
            worker.prefill_worker_id,
            worker.prefill_dp_rank,
            self.tracker.prefill_worker_type().map(String::from),
            worker.decode_worker_id,
            worker.decode_dp_rank,
            self.tracker.decode_worker_type().map(String::from),
        );
        self.worker_info_observed = true;
    }

    pub(super) fn observe(&mut self, annotated: &mut Annotated<LLMEngineOutput>) {
        let Some(output) = annotated.data.as_mut() else {
            return;
        };

        if let Some(routing_data) = output.routing_data.take() {
            if let Some(timing) = routing_data.timing {
                self.tracker.set_external_timing(timing);
            }
            if let Some(worker_id) = routing_data.worker_id {
                self.tracker.set_external_worker_info(worker_id);
            }
            if let Some(token_ids) = routing_data.token_ids {
                self.tracker.set_external_query_token_ids(token_ids);
            }
        }

        self.observe_worker_info();

        let cached_tokens = output
            .completion_usage
            .as_ref()
            // A migrated attempt includes already-delivered output tokens in
            // its prompt. Ignore that attempt-local usage for this logical
            // request and let the RequestTracker fallback run on drop.
            .filter(|usage| usage.prompt_tokens as usize == self.input_tokens)
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .and_then(|details| details.cached_tokens)
            .map(|tokens| tokens as usize);
        self.response.observe_cached_tokens(cached_tokens);

        // RetryManager appends delivered tokens to the retried request and only
        // yields newly generated deltas, so this remains exact across migration.
        let chunk_tokens = output.token_ids.len();
        self.output_tokens += chunk_tokens;
        self.response.observe_current_osl(self.output_tokens);
        if self.response.is_first_token()
            && chunk_tokens > 0
            && let Some(guard) = self.http_queue.take()
        {
            drop(guard);
        }
        self.response
            .observe_response(self.input_tokens, chunk_tokens);
    }
}

impl Drop for GenerateMetricCollector {
    fn drop(&mut self) {
        // Matching backend usage is authoritative when present. The response
        // collector latches it during streaming; this logical-request router
        // estimate fills missing or migration-expanded attempt usage.
        self.response
            .observe_cached_tokens(self.tracker.cached_tokens());
    }
}

#[cfg(test)]
pub(super) mod test_support {
    /// Find the series of metric family `name` carrying all `labels`.
    pub(crate) fn metric_value<'a>(
        families: &'a [prometheus::proto::MetricFamily],
        name: &str,
        labels: &[(&str, &str)],
    ) -> &'a prometheus::proto::Metric {
        let family = families
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing metric family {name}"));
        family
            .get_metric()
            .iter()
            .find(|metric| {
                labels.iter().all(|(expected_name, expected_value)| {
                    metric.get_label().iter().any(|label| {
                        label.name() == *expected_name && label.value() == *expected_value
                    })
                })
            })
            .unwrap_or_else(|| panic!("missing {name} series with labels {labels:?}"))
    }
}

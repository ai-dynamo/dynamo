// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, LazyLock};

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::Event},
    routing::get,
};
use dynamo_http_server::metrics::{HttpQueueGuard, Metrics, Registry, ResponseMetricCollector};
use dynamo_runtime::metrics::prometheus_names::frontend_service;
use prometheus::{Encoder, IntGauge, IntGaugeVec, Opts};
use serde::Serialize;

use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::metrics::{ANNOTATION_LLM_METRICS, LLMMetricAnnotation};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

use super::RouteDoc;

// ---------------------------------------------------------------------------
// LoRA allocation metrics (updated by LoraController each tick)
// ---------------------------------------------------------------------------

pub static LORA_REPLICA_FACTOR_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_REPLICA_FACTOR),
            "Number of replicas allocated for a LoRA adapter",
        ),
        &["lora"],
    )
    .expect("Failed to create lora_replica_factor gauge")
});

pub static LORA_IS_ACTIVE_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_IS_ACTIVE),
            "Whether a LoRA adapter is active (1) or inactive (0)",
        ),
        &["lora"],
    )
    .expect("Failed to create lora_is_active gauge")
});

pub static LORA_RAW_ARRIVAL_COUNT_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::LORA_RAW_ARRIVAL_COUNT
            ),
            "Raw arrival count (windowed rate counter) for a LoRA adapter",
        ),
        &["lora"],
    )
    .expect("Failed to create lora_raw_arrival_count gauge")
});

pub static LORA_ESTIMATED_LOAD_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_ESTIMATED_LOAD),
            "Estimated load (windowed request count) for a LoRA adapter",
        ),
        &["lora"],
    )
    .expect("Failed to create lora_estimated_load gauge")
});

pub static LORA_ACTIVE_REQUESTS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_ACTIVE_REQUESTS),
            "Number of in-flight requests for a LoRA adapter",
        ),
        &["lora"],
    )
    .expect("Failed to create lora_active_requests gauge")
});

pub static LORA_CHURN_LOADS_GAUGE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        format!(
            "dynamo_frontend_{}",
            frontend_service::LORA_CHURN_LOADS_TOTAL
        ),
        "Total LoRA loads (new placements) this tick",
    )
    .expect("Failed to create lora_churn_loads gauge")
});

pub static LORA_CHURN_UNLOADS_GAUGE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        format!(
            "dynamo_frontend_{}",
            frontend_service::LORA_CHURN_UNLOADS_TOTAL
        ),
        "Total LoRA unloads (removed placements) this tick",
    )
    .expect("Failed to create lora_churn_unloads gauge")
});

pub static LORA_OVERFLOW_COUNT_GAUGE: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        format!("dynamo_frontend_{}", frontend_service::LORA_OVERFLOW_COUNT),
        "MCF solver overflow count (unplaceable replicas)",
    )
    .expect("Failed to create lora_overflow_count gauge")
});

pub fn register_lora_allocation_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(LORA_REPLICA_FACTOR_GAUGE.clone()))?;
    registry.register(Box::new(LORA_IS_ACTIVE_GAUGE.clone()))?;
    registry.register(Box::new(LORA_RAW_ARRIVAL_COUNT_GAUGE.clone()))?;
    registry.register(Box::new(LORA_ESTIMATED_LOAD_GAUGE.clone()))?;
    registry.register(Box::new(LORA_ACTIVE_REQUESTS_GAUGE.clone()))?;
    registry.register(Box::new(LORA_CHURN_LOADS_GAUGE.clone()))?;
    registry.register(Box::new(LORA_CHURN_UNLOADS_GAUGE.clone()))?;
    registry.register(Box::new(LORA_OVERFLOW_COUNT_GAUGE.clone()))?;
    Ok(())
}

/// State for the metrics endpoint, including the optional runtime registry tree.
struct MetricsHandlerState {
    registry: Arc<Registry>,
    drt_metrics: Option<dynamo_runtime::metrics::MetricsRegistry>,
}

/// Adapt worker runtime configuration into scalar HTTP metric updates.
pub fn update_runtime_config_metrics(
    metrics: &Metrics,
    model_name: &str,
    runtime_config: &ModelRuntimeConfig,
) {
    metrics.update_runtime_config_metrics_raw(
        model_name,
        runtime_config.total_kv_blocks,
        runtime_config.max_num_seqs,
        runtime_config.max_num_batched_tokens,
    );
}

/// Adapt a model deployment card into plane-independent HTTP metric updates.
pub fn update_metrics_from_mdc(
    metrics: &Metrics,
    card: &ModelDeploymentCard,
) -> anyhow::Result<()> {
    update_runtime_config_metrics(metrics, &card.display_name, &card.runtime_config);
    metrics.update_deployment_metrics(
        &card.display_name,
        card.effective_context_length(),
        card.kv_cache_block_size,
        card.migration_limit,
    );
    tracing::debug!(model = %card.display_name, "Successfully updated MDC metrics");
    Ok(())
}

/// Process streaming metrics for annotated responses
///
/// This function handles metrics collection and http_queue_guard management for streaming responses.
/// It observes the current output sequence length, drops the http_queue_guard on the first token,
/// and records response metrics.
pub fn process_response_and_observe_metrics<T>(
    annotated: &crate::types::Annotated<T>,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) {
    if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(annotated) {
        observe_llm_metrics(&metrics, response_collector, http_queue_guard);
    }
}

/// Process streaming metrics for chat-derived responses.
///
/// The typed metrics field is the hot path. Annotation parsing remains as a compatibility
/// fallback for legacy/generated annotation frames.
pub fn process_chat_response_and_observe_metrics(
    annotated: &crate::types::Annotated<NvCreateChatCompletionStreamResponse>,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) {
    if let Some(metrics) = annotated
        .data
        .as_ref()
        .and_then(|data| data.llm_metrics.as_ref())
    {
        observe_llm_metrics(metrics, response_collector, http_queue_guard);
    } else {
        process_response_and_observe_metrics(annotated, response_collector, http_queue_guard);
    }
}

/// Event converter wrapper for streaming responses
pub struct EventConverter<T>(pub crate::types::Annotated<T>);

impl<T> From<crate::types::Annotated<T>> for EventConverter<T> {
    fn from(annotated: crate::types::Annotated<T>) -> Self {
        EventConverter(annotated)
    }
}

fn sse_json_data<T: Serialize>(event: Event, data: &T) -> Result<Event, axum::Error> {
    // serde_json escapes literal LF/CR in string content, so the resulting JSON
    // is one SSE data line and can avoid Axum json_data's per-write filtering.
    let json = serde_json::to_string(data).map_err(axum::Error::new)?;
    Ok(event.data(json))
}

fn observe_llm_metrics(
    metrics: &LLMMetricAnnotation,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) {
    response_collector.observe_current_osl(metrics.output_tokens);
    response_collector.observe_cached_tokens(metrics.cached_tokens);
    response_collector.observe_multimodal_counts(
        metrics.image_count,
        metrics.video_count,
        metrics.audio_count,
    );
    response_collector.observe_tokenize_latencies(
        metrics.tokenize_latency,
        metrics.detokenize_total_latency,
        metrics.detokenize_count,
    );
    response_collector.set_worker_info(
        metrics.prefill_worker_id,
        metrics.prefill_dp_rank,
        metrics.prefill_worker_type.as_deref(),
        metrics.decode_worker_id,
        metrics.decode_dp_rank,
        metrics.decode_worker_type.as_deref(),
    );

    if response_collector.is_first_token()
        && metrics.chunk_tokens > 0
        && let Some(guard) = http_queue_guard.take()
    {
        drop(guard);
    }

    response_collector.observe_response(metrics.input_tokens, metrics.chunk_tokens);
}

fn observe_annotation_metrics<T>(
    annotated: &crate::types::Annotated<T>,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) -> bool {
    if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(annotated) {
        observe_llm_metrics(&metrics, response_collector, http_queue_guard);
        true
    } else {
        false
    }
}

fn annotated_to_sse_event<T: Serialize>(
    annotated: crate::types::Annotated<T>,
) -> Result<Option<Event>, axum::Error> {
    let mut event = Event::default();

    if let Some(ref data) = annotated.data {
        event = sse_json_data(event, data)?;
    }

    if let Some(ref msg) = annotated.event {
        if msg == "error" {
            let error_message = if let Some(ref dynamo_err) = annotated.error
                && !dynamo_err.message().is_empty()
            {
                dynamo_err.message().to_string()
            } else if let Some(ref comments) = annotated.comment {
                let joined = comments.join(" -- ");
                if joined.trim().is_empty() {
                    "unspecified error".to_string()
                } else {
                    joined
                }
            } else {
                "unspecified error".to_string()
            };
            return Err(axum::Error::new(error_message));
        }
        event = event.event(msg);
    }

    if let Some(comments) = annotated.comment {
        for comment in comments {
            // Axum's Event::comment() panics on \n / \r
            event = event.comment(comment.replace(['\n', '\r'], " "));
        }
    }

    // Filter out metrics annotation events (events without SSE data payload)
    if annotated.data.is_none() && annotated.event.is_none() {
        Ok(None)
    } else {
        Ok(Some(event))
    }
}

/// Process streaming response with event conversion for SSE
///
/// This function handles metrics collection, http_queue_guard management, and converts
/// annotated responses to SSE events for streaming responses.
///
/// Returns None for metrics annotation events (events without SSE data payload).
pub fn process_response_using_event_converter_and_observe_metrics<T: Serialize>(
    annotated: EventConverter<T>,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) -> Result<Option<Event>, axum::Error> {
    let mut annotated = annotated.0;

    if observe_annotation_metrics(&annotated, response_collector, http_queue_guard) {
        // Preserve the previous SSE behavior: observe legacy metrics annotation
        // frames internally, then strip them before building the outbound event.
        if annotated.event.as_deref() == Some(ANNOTATION_LLM_METRICS) {
            annotated.event = None;
            annotated.comment = None;
        }
    }

    // ANNOTATION_PAYLOAD_USAGE is payload-only and must never reach the client SSE
    // stream (the payload DeltaAggregator consumed its usage upstream).
    if annotated.event.as_deref() == Some(crate::preprocessor::ANNOTATION_PAYLOAD_USAGE) {
        annotated.event = None;
        annotated.comment = None;
        annotated.data = None;
    }
    annotated_to_sse_event(annotated)
}

/// Process chat-derived streaming responses with typed metrics before SSE conversion.
///
/// If the typed field is absent, this falls back to the same legacy annotation
/// frame parsing and stripping behavior as `process_response_using_event_converter_and_observe_metrics`.
pub fn process_chat_response_using_event_converter_and_observe_metrics(
    annotated: EventConverter<NvCreateChatCompletionStreamResponse>,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) -> Result<Option<Event>, axum::Error> {
    let mut annotated = annotated.0;

    if let Some(metrics) = annotated
        .data
        .as_ref()
        .and_then(|data| data.llm_metrics.as_ref())
    {
        observe_llm_metrics(metrics, response_collector, http_queue_guard);
    } else if observe_annotation_metrics(&annotated, response_collector, http_queue_guard)
        && annotated.event.as_deref() == Some(ANNOTATION_LLM_METRICS)
    {
        // Legacy compatibility path: annotation frames were never emitted as
        // client-visible SSE payloads after metrics collection.
        annotated.event = None;
        annotated.comment = None;
    }

    // ANNOTATION_PAYLOAD_USAGE is payload-only: the payload DeltaAggregator already
    // consumed its usage upstream, so strip the whole chunk; it must never
    // reach the client SSE stream.
    if annotated.event.as_deref() == Some(crate::preprocessor::ANNOTATION_PAYLOAD_USAGE) {
        annotated.event = None;
        annotated.comment = None;
        annotated.data = None;
    }
    annotated_to_sse_event(annotated)
}

/// Create a new router with optional DRT metrics integration.
///
/// When `drt_metrics` is provided, the `/metrics` handler will also include
/// metrics from the DRT's registry tree (anything created via `metrics().create*()`).
pub fn router(
    registry: Registry,
    path: Option<String>,
    drt_metrics: Option<dynamo_runtime::metrics::MetricsRegistry>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/metrics".to_string());
    let doc = RouteDoc::new(axum::http::Method::GET, &path);

    let metrics_state = MetricsHandlerState {
        registry: Arc::new(registry),
        drt_metrics,
    };

    let route = Router::new()
        .route(&path, get(handler_metrics))
        .with_state(Arc::new(metrics_state));
    (vec![doc], route)
}

/// Unified metrics handler.
///
/// Gathers from the local HTTP-service registry first, then appends any
/// metrics from the DRT's registry tree (if configured).
async fn handler_metrics(State(state): State<Arc<MetricsHandlerState>>) -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = state.registry.gather();
    let mut buffer = vec![];
    if encoder.encode(&metric_families, &mut buffer).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics",
        )
            .into_response();
    }

    let mut metrics = match String::from_utf8(buffer) {
        Ok(metrics) => metrics,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics",
            )
                .into_response();
        }
    };

    // Append DRT registry tree metrics (anything created via metrics().create*()).
    if let Some(ref drt_metrics) = state.drt_metrics {
        match drt_metrics.prometheus_expfmt_combined() {
            Ok(drt_text) => {
                if !drt_text.is_empty() {
                    if !metrics.is_empty() && !metrics.ends_with('\n') {
                        metrics.push('\n');
                    }
                    metrics.push_str(&drt_text);
                }
            }
            Err(e) => {
                tracing::warn!("Failed to gather DRT metrics: {}", e);
            }
        }
    }

    (StatusCode::OK, metrics).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_metrics_annotation_event_handling() {
        use crate::protocols::common::metrics::LLMMetricAnnotation;
        use crate::types::Annotated;

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let expected_tokenizer_metric_name = "dynamo_frontend_tokenizer_latency_ms";
        let mut collector = metrics.clone().create_response_collector(model);

        // Create a metrics annotation event (event without SSE data payload)
        let mut annotated = Annotated::<
            crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        > {
            id: None,
            data: None,
            event: Some(ANNOTATION_LLM_METRICS.to_string()),
            comment: None,
            error: None,
        };

        // Add metrics annotation with cached_tokens
        let llm_metrics = LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens: 20,
            chunk_tokens: 5,
            cached_tokens: Some(15),
            prefill_worker_id: None,
            prefill_dp_rank: None,
            prefill_worker_type: None,
            decode_worker_id: None,
            decode_dp_rank: None,
            decode_worker_type: None,
            tokenize_latency: Some(Duration::from_millis(8)),
            detokenize_total_latency: Some(Duration::from_micros(100)),
            detokenize_count: Some(2),
            ..Default::default()
        };

        let annotation = llm_metrics.to_annotation::<()>().unwrap();
        annotated.event = annotation.event;
        annotated.comment = annotation.comment;

        // Process the event
        let mut http_queue_guard = None;
        let result = process_response_using_event_converter_and_observe_metrics(
            EventConverter::from(annotated),
            &mut collector,
            &mut http_queue_guard,
        );

        // Should return Ok(None) for metrics annotation events
        assert!(matches!(result, Ok(None)));

        // Drop collector so the detokenize observation fires in Drop
        drop(collector);

        // Should have observed the cached tokens from the metrics annotation event
        let metric_families = registry.gather();
        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_metric_name)
            .expect("histogram should be registered");
        assert_eq!(
            histogram_family.get_metric()[0]
                .get_histogram()
                .get_sample_count(),
            1
        );

        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_tokenizer_metric_name)
            .expect("histogram should be registered");

        // Find the tokenize and detokenize observations by label
        let tokenize_metric = histogram_family
            .get_metric()
            .iter()
            .find(|m| m.get_label().iter().any(|l| l.value() == "tokenize"))
            .expect("tokenize metric should exist");
        assert_eq!(tokenize_metric.get_histogram().get_sample_count(), 1);
        // 8ms
        assert!(
            (tokenize_metric.get_histogram().get_sample_sum() - 8.0).abs() < 0.001,
            "tokenize latency should be 8.0ms"
        );

        let detokenize_metric = histogram_family
            .get_metric()
            .iter()
            .find(|m| m.get_label().iter().any(|l| l.value() == "detokenize"))
            .expect("detokenize metric should exist");
        assert_eq!(detokenize_metric.get_histogram().get_sample_count(), 1);
        // Average: 100us total / 2 samples = 50us = 0.05ms
        assert!(
            (detokenize_metric.get_histogram().get_sample_sum() - 0.05).abs() < 0.001,
            "detokenize average latency should be 0.05ms, got {}",
            detokenize_metric.get_histogram().get_sample_sum()
        );
    }

    #[tokio::test]
    async fn metrics_annotations_stripped_from_client_sse() {
        // PR #9390: two metric annotations must never leak to the client SSE.
        // (1) Per-chunk `llm_metrics` rides on a content chunk: the content delta
        //     must reach the client, but the metrics event/comment must be stripped.
        // (2) The audit-only `audit_usage` chunk must be dropped entirely (the audit
        //     DeltaAggregator already consumed its usage upstream).
        use crate::protocols::common::metrics::{ANNOTATION_PAYLOAD_USAGE, LLMMetricAnnotation};
        use crate::types::Annotated;
        use axum::response::IntoResponse;
        use axum::response::sse::Sse;

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let mut collector = metrics.clone().create_response_collector("test-model");

        let llm_metrics = LLMMetricAnnotation {
            input_tokens: 7,
            output_tokens: 3,
            chunk_tokens: 1,
            cached_tokens: Some(2),
            prefill_worker_id: None,
            prefill_dp_rank: None,
            prefill_worker_type: None,
            decode_worker_id: None,
            decode_dp_rank: None,
            decode_worker_type: None,
            tokenize_latency: None,
            detokenize_total_latency: None,
            detokenize_count: None,
            ..Default::default()
        };
        let metrics_comment = llm_metrics.to_annotation::<()>().unwrap().comment;
        let metrics_json = metrics_comment.as_ref().expect("metrics comment present")[0].clone();

        // (1) Per-chunk metrics on a content chunk (event = llm_metrics).
        let content: crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse =
            serde_json::from_value(serde_json::json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "hello"}}]
            }))
            .unwrap();
        let per_chunk = Annotated {
            id: None,
            data: Some(content),
            event: Some(ANNOTATION_LLM_METRICS.to_string()),
            comment: metrics_comment.clone(),
            error: None,
        };

        let mut http_queue_guard = None;
        let event = process_response_using_event_converter_and_observe_metrics(
            EventConverter::from(per_chunk),
            &mut collector,
            &mut http_queue_guard,
        )
        .expect("conversion ok")
        .expect("content chunk should yield a client event");

        let sse = Sse::new(futures::stream::once(async move {
            Ok::<_, std::convert::Infallible>(event)
        }));
        let body = sse.into_response().into_body();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let wire = String::from_utf8_lossy(&bytes);

        assert!(
            wire.contains("hello"),
            "content delta should reach the client: {wire}"
        );
        assert!(
            !wire.contains(&metrics_json)
                && !wire.contains("chunk_tokens")
                && !wire.contains("input_tokens"),
            "internal metrics leaked to client SSE: {wire}"
        );

        // (2) Payload-only usage chunk (event = payload_usage, carries usage data).
        let usage: crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse =
            serde_json::from_value(serde_json::json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1,
                "model": "test-model", "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            }))
            .unwrap();
        let payload_usage = Annotated {
            id: None,
            data: Some(usage),
            event: Some(ANNOTATION_PAYLOAD_USAGE.to_string()),
            comment: metrics_comment,
            error: None,
        };

        let mut http_queue_guard = None;
        let result = process_response_using_event_converter_and_observe_metrics(
            EventConverter::from(payload_usage),
            &mut collector,
            &mut http_queue_guard,
        )
        .expect("conversion ok");
        assert!(
            result.is_none(),
            "payload_usage chunk must not be forwarded to the client SSE stream"
        );
    }

    #[test]
    fn test_chat_typed_metrics_fast_path_observes_without_annotation() {
        use crate::protocols::common::metrics::LLMMetricAnnotation;
        use dynamo_http_server::metrics::{WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL};

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let expected_tokenizer_metric_name = "dynamo_frontend_tokenizer_latency_ms";
        let mut collector = metrics.clone().create_response_collector(model);
        let mut annotated = make_chat_stream_annotated("hello");
        annotated.data.as_mut().unwrap().llm_metrics = Some(LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens: 20,
            chunk_tokens: 5,
            cached_tokens: Some(15),
            prefill_worker_id: Some(11),
            prefill_dp_rank: Some(1),
            prefill_worker_type: Some(WORKER_TYPE_PREFILL.to_string()),
            decode_worker_id: Some(22),
            decode_dp_rank: Some(2),
            decode_worker_type: Some(WORKER_TYPE_DECODE.to_string()),
            tokenize_latency: Some(Duration::from_millis(8)),
            detokenize_total_latency: Some(Duration::from_micros(100)),
            detokenize_count: Some(2),
            ..Default::default()
        });

        assert!(annotated.event.is_none());
        assert!(annotated.comment.is_none());

        let mut http_queue_guard = Some(metrics.clone().create_http_queue_guard(model));
        let result = process_chat_response_using_event_converter_and_observe_metrics(
            EventConverter::from(annotated),
            &mut collector,
            &mut http_queue_guard,
        );

        assert!(
            http_queue_guard.is_none(),
            "first positive chunk should release HTTP queue guard"
        );
        let event = result
            .unwrap()
            .expect("typed metrics data chunk should still produce an SSE event");
        let json = extract_sse_data_json(event);
        assert!(
            json.get("llm_metrics").is_none(),
            "typed metrics must be skipped on the SSE wire"
        );

        drop(collector);

        let metric_families = registry.gather();
        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_metric_name)
            .expect("histogram should be registered");
        assert_eq!(
            histogram_family.get_metric()[0]
                .get_histogram()
                .get_sample_count(),
            1
        );

        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_tokenizer_metric_name)
            .expect("histogram should be registered");
        let tokenize_metric = histogram_family
            .get_metric()
            .iter()
            .find(|m| m.get_label().iter().any(|l| l.value() == "tokenize"))
            .expect("tokenize metric should exist");
        assert_eq!(tokenize_metric.get_histogram().get_sample_count(), 1);
        assert!(
            (tokenize_metric.get_histogram().get_sample_sum() - 8.0).abs() < 0.001,
            "tokenize latency should be 8.0ms"
        );
    }

    #[test]
    fn test_chat_stream_typed_metrics_are_skipped_in_json() {
        use crate::protocols::common::metrics::LLMMetricAnnotation;

        let without_metrics = make_chat_stream_annotated("hello").data.unwrap();
        let mut with_metrics = without_metrics.clone();
        with_metrics.llm_metrics = Some(LLMMetricAnnotation {
            input_tokens: 1,
            output_tokens: 2,
            chunk_tokens: 1,
            cached_tokens: Some(1),
            prefill_worker_id: None,
            prefill_dp_rank: None,
            prefill_worker_type: None,
            decode_worker_id: None,
            decode_dp_rank: None,
            decode_worker_type: None,
            tokenize_latency: None,
            detokenize_total_latency: None,
            detokenize_count: None,
            ..Default::default()
        });

        let without_json = serde_json::to_value(&without_metrics).unwrap();
        let with_json = serde_json::to_value(&with_metrics).unwrap();

        assert_eq!(with_json, without_json);
        assert!(with_json.get("llm_metrics").is_none());
    }

    #[test]
    fn test_non_streaming_path_observes_cached_tokens() {
        use crate::protocols::common::metrics::LLMMetricAnnotation;
        use crate::types::Annotated;

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let expected_tokenizer_metric_name = "dynamo_frontend_tokenizer_latency_ms";
        let mut collector = metrics.clone().create_response_collector(model);

        // Create a metrics annotation event
        let mut annotated = Annotated::<
            crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        > {
            id: None,
            data: None,
            event: Some(ANNOTATION_LLM_METRICS.to_string()),
            comment: None,
            error: None,
        };

        let llm_metrics = LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens: 20,
            chunk_tokens: 5,
            cached_tokens: Some(15),
            prefill_worker_id: None,
            prefill_dp_rank: None,
            prefill_worker_type: None,
            decode_worker_id: None,
            decode_dp_rank: None,
            decode_worker_type: None,
            tokenize_latency: Some(Duration::from_millis(8)),
            detokenize_total_latency: Some(Duration::from_micros(100)),
            detokenize_count: Some(2),
            ..Default::default()
        };

        let annotation = llm_metrics.to_annotation::<()>().unwrap();
        annotated.event = annotation.event;
        annotated.comment = annotation.comment;

        // Process via the non-streaming metrics hook
        let mut http_queue_guard = None;
        process_response_and_observe_metrics(&annotated, &mut collector, &mut http_queue_guard);

        // Drop collector so the detokenize observation fires in Drop
        drop(collector);

        // Should have observed the cached tokens from the metrics annotation event
        let metric_families = registry.gather();
        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_metric_name)
            .expect("histogram should be registered");
        assert_eq!(
            histogram_family.get_metric()[0]
                .get_histogram()
                .get_sample_count(),
            1
        );

        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_tokenizer_metric_name)
            .expect("histogram should be registered");

        // Find the tokenize and detokenize observations by label
        let tokenize_metric = histogram_family
            .get_metric()
            .iter()
            .find(|m| m.get_label().iter().any(|l| l.value() == "tokenize"))
            .expect("tokenize metric should exist");
        assert_eq!(tokenize_metric.get_histogram().get_sample_count(), 1);

        let detokenize_metric = histogram_family
            .get_metric()
            .iter()
            .find(|m| m.get_label().iter().any(|l| l.value() == "detokenize"))
            .expect("detokenize metric should exist");
        assert_eq!(detokenize_metric.get_histogram().get_sample_count(), 1);
        // Average: 100us total / 2 samples = 50us = 0.05ms
        assert!(
            (detokenize_metric.get_histogram().get_sample_sum() - 0.05).abs() < 0.001,
            "detokenize average latency should be 0.05ms, got {}",
            detokenize_metric.get_histogram().get_sample_sum()
        );
    }

    fn run_event_converter(
        annotated: crate::types::Annotated<String>,
    ) -> Result<Option<Event>, axum::Error> {
        let metrics = Arc::new(Metrics::new());
        let mut collector = metrics.create_response_collector("test-model");
        let mut http_queue_guard: Option<HttpQueueGuard> = None;
        process_response_using_event_converter_and_observe_metrics(
            EventConverter::from(annotated),
            &mut collector,
            &mut http_queue_guard,
        )
    }

    fn run_chat_stream_event_converter(
        annotated: crate::types::Annotated<
            crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >,
    ) -> Result<Option<Event>, axum::Error> {
        let metrics = Arc::new(Metrics::new());
        let mut collector = metrics.create_response_collector("test-model");
        let mut http_queue_guard: Option<HttpQueueGuard> = None;
        process_chat_response_using_event_converter_and_observe_metrics(
            EventConverter::from(annotated),
            &mut collector,
            &mut http_queue_guard,
        )
    }

    fn make_chat_stream_annotated(
        content: &str,
    ) -> crate::types::Annotated<
        crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
    > {
        use dynamo_protocols::types::{
            ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
            CreateChatCompletionStreamResponse,
        };

        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: Some(ChatCompletionMessageContent::Text(content.to_string())),
                function_call: None,
                tool_calls: None,
                role: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        };

        crate::types::Annotated {
            id: Some("test-id".to_string()),
            data: Some(
                crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse {
                    inner: CreateChatCompletionStreamResponse {
                        id: "test-id".to_string(),
                        choices: vec![choice],
                        created: 0,
                        model: "test-model".to_string(),
                        system_fingerprint: None,
                        object: "chat.completion.chunk".to_string(),
                        usage: None,
                        service_tier: None,
                    },
                    nvext: None,
                    llm_metrics: None,
                },
            ),
            event: None,
            comment: None,
            error: None,
        }
    }

    fn extract_sse_data_json(event: Event) -> serde_json::Value {
        let body = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(async move {
                use axum::{
                    body::to_bytes,
                    response::{IntoResponse, Sse},
                };
                let stream = futures::stream::iter(vec![Ok::<Event, axum::Error>(event)]);
                let response = Sse::new(stream).into_response();
                let bytes = to_bytes(response.into_body(), 1 << 20)
                    .await
                    .expect("body bytes");
                String::from_utf8(bytes.to_vec()).expect("utf8 body")
            });

        let data = body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("SSE data line");
        serde_json::from_str(data).unwrap_or_else(|e| {
            panic!("failed to parse JSON from SSE data: {e}\nbody: {body}\ndata: {data}")
        })
    }

    fn error_annotated(
        error: Option<dynamo_runtime::error::DynamoError>,
        comment: Option<Vec<String>>,
    ) -> crate::types::Annotated<String> {
        crate::types::Annotated {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment,
            error,
        }
    }

    #[test]
    fn test_error_event_uses_dynamo_error_message() {
        use dynamo_runtime::error::DynamoError;
        let result = run_event_converter(error_annotated(
            Some(DynamoError::msg("image load failed: 403 Forbidden")),
            None,
        ));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("403 Forbidden"));
    }

    #[test]
    fn test_error_event_falls_back_to_comment() {
        let result =
            run_event_converter(error_annotated(None, Some(vec!["connection lost".into()])));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("connection lost"));
    }

    #[test]
    fn test_error_event_unspecified_when_no_message() {
        let result = run_event_converter(error_annotated(None, None));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "unspecified error");
    }

    #[test]
    fn test_error_event_empty_comment_falls_through() {
        let result = run_event_converter(error_annotated(None, Some(vec!["".into()])));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "unspecified error");
    }

    #[test]
    fn test_event_converter_serializes_chat_stream_response_as_json_data() {
        let event = run_chat_stream_event_converter(make_chat_stream_annotated("hello"))
            .unwrap()
            .expect("chat stream response should produce an SSE event");

        let json = extract_sse_data_json(event);
        assert_eq!(json["id"], "test-id");
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["content"], "hello");
    }

    #[test]
    fn test_event_converter_json_data_round_trips_escaped_content() {
        let content = "line1\nline2\r\"quoted\" \\\\ slash 你好 🚀";
        let event = run_chat_stream_event_converter(make_chat_stream_annotated(content))
            .unwrap()
            .expect("chat stream response should produce an SSE event");

        let json = extract_sse_data_json(event);
        assert_eq!(json["choices"][0]["delta"]["content"], content);
    }

    #[test]
    fn test_comment_newlines_sanitized() {
        let annotated = crate::types::Annotated::<String> {
            data: Some("test".to_string()),
            id: None,
            event: Some("metrics".to_string()),
            comment: Some(vec!["line1\nline2\r\nline3".into()]),
            error: None,
        };
        let event = run_event_converter(annotated)
            .unwrap()
            .expect("data event with comment should be returned");
        let debug = format!("{:?}", event);
        assert!(
            debug.contains(": line1 line2  line3\\n"),
            "comment newlines should be replaced before Event::comment: {debug}"
        );
    }
}

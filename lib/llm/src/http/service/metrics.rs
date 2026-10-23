// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::Event},
    routing::get,
};
use dynamo_runtime::metrics::prometheus_names::{
    frontend_service, name_prefix, sanitize_frontend_prometheus_prefix,
};
use prometheus::{
    Encoder, IntGaugeVec, Opts,
    core::{Collector, Desc},
    proto::{Gauge, LabelPair, Metric, MetricFamily, MetricType},
};
use serde::Serialize;
use std::sync::{Arc, LazyLock};

use crate::protocols::{
    common::metrics::{ANNOTATION_LLM_METRICS, LLMMetricAnnotation},
    openai::chat_completions::NvCreateChatCompletionStreamResponse,
};
use crate::reasoning_field::{ReasoningField, RoutedReasoning};
use crate::{
    discovery::ModelManager, local_model::runtime_config::ModelRuntimeConfig,
    model_card::ModelDeploymentCard,
};

use super::RouteDoc;
use dynamo_http_server::metrics::{
    HttpQueueGuard, Metrics, Registry, ResponseMetricCollector, ServingLoadCollector,
};

const MODEL_READY_HELP: &str = "Whether the frontend can route at least one inference request for the model (1 = ready, 0 = not ready)";

fn model_ready_metric_name(metrics_prefix: Option<&str>) -> String {
    let prefix =
        sanitize_frontend_prometheus_prefix(metrics_prefix.unwrap_or(name_prefix::FRONTEND));
    format!("{}_{}", prefix, frontend_service::MODEL_READY)
}

/// Collects current model readiness directly from the frontend's routing catalog.
///
/// This is evaluated at scrape time so worker registration and removal are
/// reflected without maintaining a second, potentially stale readiness state.
struct ModelReadyCollector {
    manager: Arc<ModelManager>,
    desc: Desc,
    metric_name: String,
}

impl ModelReadyCollector {
    fn new(
        manager: Arc<ModelManager>,
        metrics_prefix: Option<String>,
    ) -> Result<Self, prometheus::Error> {
        let metric_name = model_ready_metric_name(metrics_prefix.as_deref());
        let desc = Desc::new(
            metric_name.clone(),
            MODEL_READY_HELP.to_string(),
            vec!["model".to_string()],
            Default::default(),
        )?;
        Ok(Self {
            manager,
            desc,
            metric_name,
        })
    }
}

impl Collector for ModelReadyCollector {
    fn desc(&self) -> Vec<&Desc> {
        vec![&self.desc]
    }

    fn collect(&self) -> Vec<MetricFamily> {
        let readiness = self.manager.registered_model_readiness();
        if readiness.is_empty() {
            return Vec::new();
        }

        let mut metrics = Vec::with_capacity(readiness.len());
        for (model, ready) in readiness {
            let mut model_label = LabelPair::default();
            model_label.set_name("model".to_string());
            model_label.set_value(model);

            let mut gauge = Gauge::default();
            gauge.set_value(if ready { 1.0 } else { 0.0 });

            let mut metric = Metric::default();
            metric.set_label(vec![model_label]);
            metric.set_gauge(gauge);
            metrics.push(metric);
        }

        let mut family = MetricFamily::default();
        family.set_name(self.metric_name.clone());
        family.set_help(MODEL_READY_HELP.to_string());
        family.set_field_type(MetricType::GAUGE);
        family.set_metric(metrics);
        vec![family]
    }
}

pub fn register_model_ready_metric(
    registry: &Registry,
    manager: Arc<ModelManager>,
    metrics_prefix: Option<String>,
) -> Result<(), prometheus::Error> {
    registry.register(Box::new(ModelReadyCollector::new(manager, metrics_prefix)?))
}

// ---------------------------------------------------------------------------
// LoRA allocation metrics (updated by LoraController each tick)
// ---------------------------------------------------------------------------

pub static LORA_REPLICA_FACTOR_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_REPLICA_FACTOR),
            "Number of replicas allocated for a LoRA adapter",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_replica_factor gauge")
});

pub static LORA_IS_ACTIVE_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_IS_ACTIVE),
            "Whether a LoRA adapter is active (1) or inactive (0)",
        ),
        &["endpoint", "lora"],
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
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_raw_arrival_count gauge")
});

pub static LORA_ESTIMATED_LOAD_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_ESTIMATED_LOAD),
            "Estimated load (windowed request count) for a LoRA adapter",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_estimated_load gauge")
});

pub static LORA_ACTIVE_REQUESTS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_ACTIVE_REQUESTS),
            "Number of in-flight requests for a LoRA adapter",
        ),
        &["endpoint", "lora"],
    )
    .expect("Failed to create lora_active_requests gauge")
});

pub static LORA_CHURN_LOADS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::LORA_CHURN_LOADS_TOTAL
            ),
            "Total LoRA loads (new placements) this tick",
        ),
        &["endpoint"],
    )
    .expect("Failed to create lora_churn_loads gauge")
});

pub static LORA_CHURN_UNLOADS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::LORA_CHURN_UNLOADS_TOTAL
            ),
            "Total LoRA unloads (removed placements) this tick",
        ),
        &["endpoint"],
    )
    .expect("Failed to create lora_churn_unloads gauge")
});

pub static LORA_OVERFLOW_COUNT_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!("dynamo_frontend_{}", frontend_service::LORA_OVERFLOW_COUNT),
            "MCF solver overflow count (unplaceable replicas)",
        ),
        &["endpoint"],
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

struct MetricsHandlerState {
    registry: Arc<Registry>,
    drt_metrics: Option<dynamo_runtime::metrics::MetricsRegistry>,
}

struct ModelServingLoadCollector {
    manager: Option<Arc<ModelManager>>,
    metrics: ServingLoadCollector,
}

impl Collector for ModelServingLoadCollector {
    fn desc(&self) -> Vec<&Desc> {
        self.metrics.desc()
    }
    fn collect(&self) -> Vec<MetricFamily> {
        if let Some(manager) = &self.manager {
            for view in manager.committed_model_views() {
                self.metrics.initialize_model(&view.name);
            }
        }
        self.metrics.collect()
    }
}

pub(super) fn register_with_model_manager(
    metrics: &Metrics,
    registry: &Registry,
    manager: Option<Arc<ModelManager>>,
) -> Result<(), prometheus::Error> {
    metrics.register_with_serving_load_collector(
        registry,
        Box::new(ModelServingLoadCollector {
            manager,
            metrics: metrics.serving_load_collector(),
        }),
    )
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
    response_collector.observe_multimodal_metrics(
        metrics.image_count,
        metrics.video_count,
        metrics.audio_count,
        metrics.image_tokens,
    );
    response_collector.observe_tokenize_latencies(
        metrics.tokenize_latency,
        metrics.detokenize_total_latency,
        metrics.detokenize_count,
    );
    response_collector.set_worker_info_from_parts(
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
            let error_message = if let Some(ref dynamo_err) = annotated.error {
                format!(
                    "semantic stream error: class={} reason={}",
                    dynamo_err.class(),
                    dynamo_err.reason()
                )
            } else {
                "backend stream error".to_string()
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
    reasoning_field: ReasoningField,
) -> Result<Option<Event>, axum::Error> {
    let mut annotated = annotated.0;
    if let Some(data) = annotated.data.as_mut() {
        data.tool_call_completion.clear();
    }

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

    // Route reasoning at the SSE boundary. Internal representation stays
    // `reasoning_content`.
    let annotated =
        annotated.map_data(|response| Ok(RoutedReasoning::new(response, reasoning_field)));
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
    use crate::discovery::UNKNOWN_METRIC_MODEL;
    use dynamo_http_server::metrics::*;
    use std::time::Duration;
    const STATE_AWAITING_FIRST_TOKEN: &str = "awaiting_first_token";
    const STATE_GENERATING: &str = "generating";
    fn response_collector(metrics: &Arc<Metrics>, model: &str) -> ResponseMetricCollector {
        ResponseMetricCollector::new(
            metrics.clone(),
            model.to_string(),
            Arc::new(RequestMetrics::new(metrics, model)),
        )
    }

    fn model_ready_value_with_name(
        registry: &Registry,
        metric_name: &str,
        model: &str,
    ) -> Option<f64> {
        registry
            .gather()
            .into_iter()
            .find(|family| family.name() == metric_name)?
            .get_metric()
            .iter()
            .find(|metric| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == "model" && label.value() == model)
            })
            .map(|metric| metric.get_gauge().value())
    }

    fn model_ready_value(registry: &Registry, model: &str) -> Option<f64> {
        model_ready_value_with_name(registry, &model_ready_metric_name(None), model)
    }

    #[test]
    fn model_ready_metric_tracks_live_routing_catalog() {
        let manager = Arc::new(ModelManager::new());
        let registry = Registry::new();
        let metrics = Metrics::new();
        let load_ready = || {
            metrics
                .frontend_load(
                    0,
                    true,
                    manager
                        .committed_model_views()
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                )
                .models[0]
                .is_ready
        };
        register_model_ready_metric(&registry, manager.clone(), None).unwrap();

        assert_eq!(model_ready_value(&registry, "test-model"), None);

        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
        let mut worker_set = crate::discovery::WorkerSet::new(
            "watched".to_string(),
            "watched-mdc".to_string(),
            card,
        );
        let (worker_tx, worker_rx) = tokio::sync::watch::channel(Vec::new());
        worker_set.set_instance_watcher(worker_rx);
        worker_set.chat_engine = Some(Arc::new(crate::engines::StreamingEngineAdapter::new(
            crate::engines::make_echo_engine(),
        )));
        assert!(manager.add_worker_set("test-model", "watched", worker_set));
        assert_eq!(model_ready_value(&registry, "test-model"), Some(0.0));
        assert!(!load_ready());

        let custom_registry = Registry::new();
        register_model_ready_metric(
            &custom_registry,
            manager.clone(),
            Some("custom_frontend".to_string()),
        )
        .unwrap();
        assert_eq!(
            model_ready_value_with_name(
                &custom_registry,
                "custom_frontend_model_ready",
                "test-model"
            ),
            Some(0.0)
        );
        assert_eq!(model_ready_value(&custom_registry, "test-model"), None);

        worker_tx.send(vec![1]).unwrap();
        assert_eq!(model_ready_value(&registry, "test-model"), Some(1.0));
        assert!(load_ready());
        let draining = metrics.frontend_load(
            0,
            false,
            manager
                .committed_model_views()
                .into_iter()
                .map(Into::into)
                .collect(),
        );
        assert!(!draining.is_serving_ready);
        assert!(draining.models[0].is_ready);

        worker_tx.send(Vec::new()).unwrap();
        assert_eq!(model_ready_value(&registry, "test-model"), Some(0.0));
        assert!(!load_ready());
    }

    #[test]
    fn model_ready_metric_omits_alias_names() {
        let manager = Arc::new(ModelManager::new());
        let registry = Registry::new();
        register_model_ready_metric(&registry, manager.clone(), None).unwrap();

        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
        let mut worker_set = crate::discovery::WorkerSet::new(
            "watched".to_string(),
            "watched-mdc".to_string(),
            card,
        );
        let (worker_tx, worker_rx) = tokio::sync::watch::channel(Vec::new());
        worker_set.set_instance_watcher(worker_rx);
        worker_set.chat_engine = Some(Arc::new(crate::engines::StreamingEngineAdapter::new(
            crate::engines::make_echo_engine(),
        )));
        let worker_set = Arc::new(worker_set);

        // `register_alias` refuses a name that is already a live primary, so it must
        // run before the alias name gains its own WorkerSet.
        assert!(manager.add_worker_set_arc("test-model", "watched", worker_set.clone()));
        assert!(manager.register_alias("test-model-alias", "test-model"));
        assert!(manager.add_worker_set_arc("test-model-alias", "watched", worker_set));

        worker_tx.send(vec![1]).unwrap();

        assert_eq!(model_ready_value(&registry, "test-model"), Some(1.0));
        assert_eq!(model_ready_value(&registry, "test-model-alias"), None);

        let series = registry
            .gather()
            .into_iter()
            .find(|family| family.name() == model_ready_metric_name(None))
            .map(|family| family.get_metric().len());
        assert_eq!(series, Some(1));
    }

    /// A chunk whose delta renders as nothing is still an accounting event: the
    /// engine generated those tokens. The chat SSE loop drops such chunks before
    /// forwarding them (multi-byte token assembly, and the Nemotron
    /// force_nonempty_content deferral, both produce them), so it observes their
    /// metrics explicitly before discarding. This pins the helper it calls: an
    /// empty delta carrying `llm_metrics` must still count.
    #[test]
    fn test_empty_delta_with_metrics_is_still_counted() {
        use crate::protocols::common::metrics::LLMMetricAnnotation;
        use dynamo_protocols::types::{
            ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
        };

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let model = "test-model";
        let mut collector = response_collector(&metrics, model);
        let mut guard = None;

        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: None,
                content: None,
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        };
        let data = NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "test".to_string(),
                choices: vec![choice],
                created: 0,
                model: model.to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
            prompt_logprobs: None,
            llm_metrics: Some(LLMMetricAnnotation {
                input_tokens: 10,
                output_tokens: 4,
                chunk_tokens: 4,
                ..Default::default()
            }),
            tool_call_completion: Vec::new(),
        };
        let annotated = crate::types::Annotated::from_data(data);

        process_chat_response_and_observe_metrics(&annotated, &mut collector, &mut guard);

        assert_eq!(
            metrics
                .frontend_load(
                    0,
                    true,
                    vec![dynamo_http_server::load::ModelLoadView {
                        name: model.into(),
                        is_ready: true,
                        aliases: vec![],
                    }]
                )
                .models[0]
                .totals
                .output_tokens,
            4,
            "tokens on a non-renderable chunk must still be counted"
        );
    }

    #[test]
    fn serving_load_metrics_exist_on_the_first_scrape_after_model_registration() {
        let metrics = Metrics::new();
        let manager = Arc::new(ModelManager::new());
        let registry = Registry::new();
        register_with_model_manager(&metrics, &registry, Some(manager.clone())).unwrap();
        let children = |model: &str| {
            registry
                .gather()
                .into_iter()
                .flat_map(|family| {
                    family
                        .get_metric()
                        .iter()
                        .filter(|sample| {
                            sample
                                .get_label()
                                .iter()
                                .any(|label| label.name() == "model" && label.value() == model)
                        })
                        .map(|sample| {
                            let state = sample
                                .get_label()
                                .iter()
                                .find(|label| label.name() == "state")
                                .map(|label| label.value())
                                .unwrap_or("");
                            assert_eq!(sample.get_gauge().value(), 0.0);
                            assert_eq!(sample.get_counter().value(), 0.0);
                            (family.name().to_string(), state.to_string())
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<std::collections::BTreeSet<_>>()
        };
        let expected = [
            (
                "dynamo_frontend_awaiting_first_token_requests".to_string(),
                String::new(),
            ),
            (
                "dynamo_frontend_inflight_input_tokens".to_string(),
                STATE_AWAITING_FIRST_TOKEN.to_string(),
            ),
            (
                "dynamo_frontend_inflight_input_tokens".to_string(),
                STATE_GENERATING.to_string(),
            ),
            (
                "dynamo_frontend_input_tokens_total".to_string(),
                String::new(),
            ),
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(children(UNKNOWN_METRIC_MODEL), expected);
        assert!(children("registered-model").is_empty());
        manager
            .add_chat_completions_model(
                "registered-model",
                "checksum",
                Arc::new(crate::engines::StreamingEngineAdapter::new(
                    crate::engines::make_echo_engine(),
                )),
            )
            .unwrap();
        assert_eq!(children("registered-model"), expected);
    }

    #[test]
    fn test_metrics_annotation_event_handling() {
        use crate::{preprocessor::LLMMetricAnnotation, types::Annotated};

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let expected_tokenizer_metric_name = "dynamo_frontend_tokenizer_latency_ms";
        let mut collector = response_collector(&metrics, model);

        // Create a metrics annotation event (event without SSE data payload)
        let mut annotated = Annotated::<
            crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        > {
            id: None,
            data: None,
            event: Some(crate::preprocessor::ANNOTATION_LLM_METRICS.to_string()),
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
        // (2) The payload-only `payload_usage` chunk must be dropped entirely (the payload
        //     DeltaAggregator already consumed its usage upstream).
        use crate::preprocessor::{ANNOTATION_PAYLOAD_USAGE, LLMMetricAnnotation};
        use crate::types::Annotated;
        use axum::response::IntoResponse;
        use axum::response::sse::Sse;

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let mut collector = response_collector(&metrics, "test-model");

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
            event: Some(crate::preprocessor::ANNOTATION_LLM_METRICS.to_string()),
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

        // A choice-less Dynamo metadata frame is client-visible.
        let metadata: crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse =
            serde_json::from_value(serde_json::json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1,
                "model": "test-model", "choices": [],
                "nvext": {"engine_data": {"prompt_token_ids": [1, 2]}}
            }))
            .unwrap();
        let metadata = Annotated {
            id: None,
            data: Some(metadata),
            event: None,
            comment: None,
            error: None,
        };
        let mut http_queue_guard = None;
        let event = process_chat_response_using_event_converter_and_observe_metrics(
            EventConverter::from(metadata),
            &mut collector,
            &mut http_queue_guard,
            ReasoningField::default(),
        )
        .expect("conversion ok")
        .expect("nvext chunk should yield a client event");
        let sse = Sse::new(futures::stream::once(async move {
            Ok::<_, std::convert::Infallible>(event)
        }));
        let body = sse.into_response().into_body();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let wire = String::from_utf8_lossy(&bytes);
        assert!(
            wire.contains("engine_data") && wire.contains("prompt_token_ids"),
            "nvext metadata did not reach client SSE: {wire}"
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
        let result = process_chat_response_using_event_converter_and_observe_metrics(
            EventConverter::from(payload_usage),
            &mut collector,
            &mut http_queue_guard,
            ReasoningField::default(),
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

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let expected_tokenizer_metric_name = "dynamo_frontend_tokenizer_latency_ms";
        let mut collector = response_collector(&metrics, model);
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

        annotated.data.as_mut().unwrap().tool_call_completion = vec![
            crate::protocols::openai::chat_completions::ToolCallCompletion {
                choice_index: 0,
                tool_index: 0,
                complete: true,
            },
        ];
        let transported: crate::types::Annotated<NvCreateChatCompletionStreamResponse> =
            serde_json::from_value(serde_json::to_value(&annotated).unwrap()).unwrap();
        assert_eq!(
            transported.data.as_ref().unwrap().tool_call_completion,
            annotated.data.as_ref().unwrap().tool_call_completion
        );

        let mut http_queue_guard = Some(metrics.clone().create_http_queue_guard(model));
        let result = process_chat_response_using_event_converter_and_observe_metrics(
            EventConverter::from(annotated),
            &mut collector,
            &mut http_queue_guard,
            ReasoningField::default(),
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
        assert!(json.get("tool_call_completion").is_none());

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
            image_tokens: Some(300),
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

        let mut inbound_json = without_json;
        inbound_json["llm_metrics"] = serde_json::json!({
            "input_tokens": 1,
            "output_tokens": 2,
            "chunk_tokens": 1,
            "cached_tokens": 1,
            "image_tokens": 300
        });
        let inbound: NvCreateChatCompletionStreamResponse =
            serde_json::from_value(inbound_json).unwrap();
        assert_eq!(inbound.llm_metrics, with_metrics.llm_metrics);
    }

    #[test]
    fn test_non_streaming_path_observes_cached_tokens() {
        use crate::{preprocessor::LLMMetricAnnotation, types::Annotated};

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let expected_tokenizer_metric_name = "dynamo_frontend_tokenizer_latency_ms";
        let mut collector = response_collector(&metrics, model);

        // Create a metrics annotation event
        let mut annotated = Annotated::<
            crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        > {
            id: None,
            data: None,
            event: Some(crate::preprocessor::ANNOTATION_LLM_METRICS.to_string()),
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
        let mut collector = response_collector(&metrics, "test-model");
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
        let mut collector = response_collector(&metrics, "test-model");
        let mut http_queue_guard: Option<HttpQueueGuard> = None;
        process_chat_response_using_event_converter_and_observe_metrics(
            EventConverter::from(annotated),
            &mut collector,
            &mut http_queue_guard,
            ReasoningField::default(),
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
                    prompt_logprobs: None,
                    llm_metrics: None,
                    tool_call_completion: Vec::new(),
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
    fn test_error_event_uses_semantic_identity_without_diagnostic() {
        use dynamo_runtime::error::DynamoError;
        let result = run_event_converter(error_annotated(
            Some(DynamoError::msg("image load failed: 403 Forbidden")),
            None,
        ));
        assert!(result.is_err());
        let message = result.unwrap_err().to_string();
        assert!(message.contains("class=Internal"));
        assert!(message.contains("reason=runtime.unclassified"));
        assert!(!message.contains("403 Forbidden"));
    }

    #[test]
    fn test_error_event_does_not_expose_comment() {
        let result =
            run_event_converter(error_annotated(None, Some(vec!["connection lost".into()])));
        assert!(result.is_err());
        let message = result.unwrap_err().to_string();
        assert_eq!(message, "backend stream error");
        assert!(!message.contains("connection lost"));
    }

    #[test]
    fn test_error_event_without_identity_is_generic() {
        let result = run_event_converter(error_annotated(None, None));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "backend stream error");
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

    /// Request payload capture must be transparent: the collector, the client
    /// response and the handler's error path behave identically with capture on
    /// (`scan_aggregate_with_future`) and off (#11349).
    mod capture_transparency {
        use super::*;
        use crate::http::service::openai::check_for_backend_error;
        use crate::http::service::service_v2::BackendErrorCheck;
        use crate::preprocessor::OpenAIPreprocessor;
        use crate::protocols::common::llm_backend::{BackendOutput, FinishReason};
        use crate::protocols::common::metrics::ANNOTATION_PAYLOAD_USAGE;
        use crate::protocols::openai::ParsingOptions;
        use crate::protocols::openai::chat_completions::aggregator::ChatCompletionAggregator;
        use crate::protocols::openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
        };
        use crate::request_trace::payload_stream::scan_aggregate_with_future;
        use crate::types::Annotated;
        use dynamo_runtime::engine::AsyncEngineContext;
        use futures::{Stream, StreamExt};

        const MODEL: &str = "test-model";
        const REQUEST_ID: &str = "test-id";
        const INPUT_TOKENS: usize = 7;
        const TAIL_CACHED_TOKENS: usize = 2;

        #[derive(Debug)]
        struct TestContext;

        #[async_trait::async_trait]
        impl AsyncEngineContext for TestContext {
            fn id(&self) -> &str {
                REQUEST_ID
            }
            fn stop_generating(&self) {}
            fn is_stopped(&self) -> bool {
                false
            }
            fn is_killed(&self) -> bool {
                false
            }
            async fn stopped(&self) {}
            async fn killed(&self) {}
            fn stop(&self) {}
            fn kill(&self) {}
            fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {}
        }

        fn backend_output(
            text: &str,
            token_ids: Vec<u32>,
            finish_reason: Option<FinishReason>,
        ) -> BackendOutput {
            BackendOutput {
                token_ids,
                tokens: vec![],
                text: Some(text.to_string()),
                cum_log_probs: None,
                log_probs: None,
                top_logprobs: None,
                finish_reason,
                stop_reason: None,
                index: Some(0),
                completion_usage: None,
                disaggregated_params: None,
                encoder_result: None,
                worker_trace_link: None,
                engine_data: None,
                routing_data: None,
                jailed_text: None,
            }
        }

        /// A backend that answers "Hello world" in two chunks and reports cached
        /// prompt tokens on the final one.
        fn backend_success() -> Vec<Annotated<BackendOutput>> {
            let mut last = backend_output("world", vec![2, 3], Some(FinishReason::Stop));
            last.completion_usage = Some(dynamo_protocols::types::CompletionUsage {
                prompt_tokens: INPUT_TOKENS as u32,
                completion_tokens: 3,
                total_tokens: (INPUT_TOKENS + 3) as u32,
                prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                    cached_tokens: Some(TAIL_CACHED_TOKENS as u32),
                    ..Default::default()
                }),
                ..Default::default()
            });
            vec![
                Annotated::from_data(backend_output("Hello ", vec![1], None)),
                Annotated::from_data(last),
            ]
        }

        /// What the frontend hands the non-streaming chat handler: the real
        /// preprocessor output for `outputs`, with payload capture off or on.
        /// The two differ in shape (capture on ends with a `payload_usage`
        /// chunk), which is exactly what the handler chain must not notice.
        fn preprocessed(
            capture: bool,
            outputs: Vec<Annotated<BackendOutput>>,
        ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static
        {
            let request = NvCreateChatCompletionRequest {
                inner: dynamo_protocols::types::CreateChatCompletionRequest {
                    model: MODEL.to_string(),
                    messages: vec![dynamo_protocols::types::ChatCompletionRequestMessage::User(
                        dynamo_protocols::types::ChatCompletionRequestUserMessage {
                            content:
                                dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                                    "Hello".to_string(),
                                ),
                            name: None,
                        },
                    )],
                    ..Default::default()
                },
                ..Default::default()
            };
            // The non-streaming handler forces usage on before preprocessing
            // (`force_include_usage`), so the client always gets a usage block.
            let mut request = request;
            request.inner.stream_options =
                Some(dynamo_protocols::types::ChatCompletionStreamOptions {
                    include_usage: true,
                    continuous_usage_stats: false,
                });
            let mut generator = request.response_generator(REQUEST_ID.to_string());
            generator.update_isl(INPUT_TOKENS as u32);
            OpenAIPreprocessor::transform_postprocessor_stream(
                futures::stream::iter(outputs),
                Box::new(generator),
                Arc::new(TestContext),
                capture,
                false,
                None,
                Default::default(),
            )
        }

        /// Run `outputs` through the handler chain with capture off, and with
        /// capture on through the scan; return both registries and results plus
        /// the capture record future.
        async fn both_legs(
            outputs: Vec<Annotated<BackendOutput>>,
        ) -> (
            (Registry, Result<NvCreateChatCompletionResponse, Rejected>),
            (Registry, Result<NvCreateChatCompletionResponse, Rejected>),
            impl std::future::Future<Output = crate::request_trace::payload_stream::PayloadOutcome>,
        ) {
            let plain = observe_and_aggregate(preprocessed(false, outputs.clone())).await;
            let (captured, future) = scan_aggregate_with_future(
                Box::pin(preprocessed(true, outputs)),
                crate::protocols::openai::ParsingOptions::default(),
            );
            let capture = observe_and_aggregate(captured).await;
            (plain, capture, future)
        }

        fn chat_chunk(
            content: Option<&str>,
            finish: Option<dynamo_protocols::types::FinishReason>,
            llm_metrics: Option<LLMMetricAnnotation>,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
                role: Some(dynamo_protocols::types::Role::Assistant),
                content: content.map(|c| {
                    dynamo_protocols::types::ChatCompletionMessageContent::Text(c.to_string())
                }),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            };
            let choice = dynamo_protocols::types::ChatChoiceStream {
                index: 0,
                delta,
                finish_reason: finish,
                logprobs: None,
            };
            let response = NvCreateChatCompletionStreamResponse {
                inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                    id: REQUEST_ID.to_string(),
                    choices: vec![choice],
                    created: 0,
                    model: MODEL.to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                    service_tier: None,
                },
                nvext: None,
                llm_metrics,
                prompt_logprobs: None,
                tool_call_completion: Vec::new(),
            };
            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
                error: None,
            }
        }

        /// Per-chunk metrics as the preprocessor attaches them to content
        /// chunks: no `cached_tokens`, which only the usage tail carries.
        fn chunk_metrics(chunk_tokens: usize, output_tokens: usize) -> LLMMetricAnnotation {
            LLMMetricAnnotation {
                input_tokens: INPUT_TOKENS,
                output_tokens,
                chunk_tokens,
                cached_tokens: None,
                ..Default::default()
            }
        }

        /// The tool-call-jail tail shape: usage data plus cumulative metrics as a
        /// `payload_usage` annotation. The tag is set explicitly because
        /// `to_annotation` always emits `llm_metrics`.
        fn payload_usage_tail(
            output_tokens: usize,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            let tail_metrics = LLMMetricAnnotation {
                cached_tokens: Some(TAIL_CACHED_TOKENS),
                ..chunk_metrics(0, output_tokens)
            };
            let annotation = tail_metrics.to_annotation::<()>().unwrap();
            let mut tail = chat_chunk(None, None, None);
            {
                let data = tail.data.as_mut().unwrap();
                data.inner.choices = vec![];
                data.inner.usage = Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens: INPUT_TOKENS as u32,
                    completion_tokens: output_tokens as u32,
                    total_tokens: (INPUT_TOKENS + output_tokens) as u32,
                    ..Default::default()
                });
            }
            tail.event = Some(ANNOTATION_PAYLOAD_USAGE.to_string());
            tail.comment = annotation.comment;
            tail
        }

        /// What the collector wrote, minus durations: TTFT and ITL sums are
        /// wall-clock and cannot be compared across runs, so only their
        /// sample counts are part of the signature.
        #[derive(Debug, PartialEq)]
        struct MetricSignature {
            output_tokens_total: u64,
            /// (sample count, sample sum)
            isl: (u64, u64),
            osl: (u64, u64),
            cached_tokens: (u64, u64),
            ttft_samples: u64,
            itl_samples: u64,
        }

        fn signature(registry: &Registry) -> MetricSignature {
            let families = registry.gather();
            let histogram = |name: &str| -> (u64, u64) {
                families
                    .iter()
                    .find(|mf| mf.name() == name)
                    .map(|mf| {
                        let h = mf.get_metric()[0].get_histogram();
                        (h.get_sample_count(), h.get_sample_sum() as u64)
                    })
                    .unwrap_or((0, 0))
            };
            let counter = |name: &str| -> u64 {
                families
                    .iter()
                    .find(|mf| mf.name() == name)
                    .map(|mf| mf.get_metric()[0].get_counter().value() as u64)
                    .unwrap_or(0)
            };
            MetricSignature {
                output_tokens_total: counter("dynamo_frontend_output_tokens_total"),
                isl: histogram("dynamo_frontend_input_sequence_tokens"),
                osl: histogram("dynamo_frontend_output_sequence_tokens"),
                cached_tokens: histogram("dynamo_frontend_cached_tokens"),
                ttft_samples: histogram("dynamo_frontend_time_to_first_token_seconds").0,
                itl_samples: histogram("dynamo_frontend_inter_token_latency_seconds").0,
            }
        }

        /// Where the non-streaming handler chain rejected the stream, if it did.
        #[derive(Debug, PartialEq)]
        enum Rejected {
            Preflight,
            Aggregation,
        }

        /// Drive `stream` through the non-streaming HTTP handler's chain (observe,
        /// backend-error preflight, aggregate) against a private registry.
        /// Consuming the stream drops the collector, which flushes ITL and OSL.
        async fn observe_and_aggregate(
            stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
        ) -> (Registry, Result<NvCreateChatCompletionResponse, Rejected>) {
            let metrics = Arc::new(Metrics::new_with_prefix(None));
            let registry = Registry::new();
            metrics.register(&registry).unwrap();
            let mut collector = response_collector(&metrics, MODEL);
            let mut http_queue_guard = None;

            let observed = stream.inspect(move |response| {
                process_chat_response_and_observe_metrics(
                    response,
                    &mut collector,
                    &mut http_queue_guard,
                );
            });
            let checked =
                match check_for_backend_error(observed, BackendErrorCheck::UntilFirstEvent).await {
                    Ok(checked) => checked,
                    Err(_) => return (registry, Err(Rejected::Preflight)),
                };
            let response = NvCreateChatCompletionResponse::from_annotated_stream(
                checked,
                ParsingOptions::default(),
            )
            .await
            .map_err(|_| Rejected::Aggregation);
            (registry, response)
        }

        fn ok(
            result: Result<NvCreateChatCompletionResponse, Rejected>,
        ) -> NvCreateChatCompletionResponse {
            result.expect("production-shaped stream must aggregate")
        }

        /// The invariant #11349 is about: the same backend output, through the
        /// real preprocessor with capture off and on, gives the collector the
        /// same numbers and the client the same response.
        #[tokio::test]
        async fn test_capture_does_not_change_metrics_or_response() {
            let ((plain_registry, plain), (capture_registry, capture), future) =
                both_legs(backend_success()).await;

            let signature_plain = signature(&plain_registry);
            assert_eq!(
                signature(&capture_registry),
                signature_plain,
                "capture changed the collector output"
            );
            // Not agreeing zeros: the real request.
            assert_eq!(
                signature_plain,
                MetricSignature {
                    output_tokens_total: 3,
                    isl: (1, INPUT_TOKENS as u64),
                    osl: (1, 3),
                    cached_tokens: (1, TAIL_CACHED_TOKENS as u64),
                    ttft_samples: 1,
                    itl_samples: 2,
                }
            );

            let (mut plain, mut capture) = (ok(plain), ok(capture));
            // `created` is wall-clock per generator; everything else must match.
            plain.inner.created = 0;
            capture.inner.created = 0;
            assert_eq!(plain, capture, "capture changed the client response");
            assert_eq!(plain.inner.model, MODEL);
            assert_eq!(
                plain.inner.choices[0].message.content.as_ref().unwrap(),
                &dynamo_protocols::types::ChatCompletionMessageContent::Text(
                    "Hello world".to_string()
                )
            );

            let outcome = future.await;
            assert!(outcome.drop_reason.is_none());
            let record = outcome.response.expect("capture must produce a record");
            assert_eq!(record.inner.model, MODEL);
            assert_eq!(record.inner.usage.as_ref().unwrap().completion_tokens, 3);
        }

        /// A chunk carrying both typed `llm_metrics` and a `payload_usage`
        /// annotation (tool-call jail shape) is observed once, typed form
        /// winning, with capture on or off.
        #[tokio::test]
        async fn test_capture_observes_dual_carrier_chunk_once() {
            let dual_carrier = || {
                let mut tail = payload_usage_tail(3);
                tail.data.as_mut().unwrap().llm_metrics = Some(chunk_metrics(5, 5));
                vec![tail]
            };
            let (plain_registry, _) =
                observe_and_aggregate(futures::stream::iter(dual_carrier())).await;
            let (captured, _future) = scan_aggregate_with_future(
                futures::stream::iter(dual_carrier()),
                crate::protocols::openai::ParsingOptions::default(),
            );
            let (capture_registry, _) = observe_and_aggregate(captured).await;

            let expected = MetricSignature {
                output_tokens_total: 5,
                isl: (1, INPUT_TOKENS as u64),
                osl: (1, 5),
                cached_tokens: (0, 0),
                ttft_samples: 1,
                itl_samples: 0,
            };
            assert_eq!(signature(&plain_registry), expected);
            assert_eq!(signature(&capture_registry), expected);
        }

        /// A backend error after content is surfaced to the client as an error
        /// (not an empty success) with capture on, exactly as with capture off,
        /// and the metrics observed before the error agree.
        #[tokio::test]
        async fn test_capture_surfaces_mid_stream_error_identically() {
            let outputs = vec![
                Annotated::from_data(backend_output("Hello ", vec![1], None)),
                Annotated::<BackendOutput>::from_error("invalid sampling parameter"),
            ];
            let ((plain_registry, plain), (capture_registry, capture), future) =
                both_legs(outputs).await;

            assert_eq!(plain.unwrap_err(), Rejected::Aggregation);
            assert_eq!(
                capture.unwrap_err(),
                Rejected::Aggregation,
                "capture must not turn an error into a success"
            );
            assert_eq!(signature(&capture_registry), signature(&plain_registry));

            let outcome = future.await;
            assert!(
                outcome
                    .drop_reason
                    .as_deref()
                    .unwrap()
                    .contains("invalid sampling parameter")
            );
            assert!(
                outcome.response.is_some(),
                "the pre-error prefix is recorded"
            );
        }

        /// A backend error before any data chunk is rejected by the preflight,
        /// with capture on or off. Without a leading frame nothing is observed;
        /// with a leading metrics frame that frame is observed (the observer runs
        /// ahead of the preflight, as on the streaming path) — identically in
        /// both modes.
        #[tokio::test]
        async fn test_capture_rejects_leading_error_identically() {
            // Legacy engines send a data-less `llm_metrics` frame; the
            // preprocessor forwards it untouched.
            let frame = || {
                chunk_metrics(1, 1)
                    .to_annotation::<BackendOutput>()
                    .unwrap()
            };
            let error = || Annotated::<BackendOutput>::from_error("backend failed");
            let nothing = MetricSignature {
                output_tokens_total: 0,
                isl: (0, 0),
                osl: (0, 0),
                cached_tokens: (0, 0),
                ttft_samples: 0,
                itl_samples: 0,
            };
            let one_frame = MetricSignature {
                output_tokens_total: 1,
                isl: (1, INPUT_TOKENS as u64),
                osl: (1, 1),
                cached_tokens: (0, 0),
                ttft_samples: 1,
                itl_samples: 0,
            };
            for (outputs, expected) in [
                (vec![error()], nothing),
                (vec![frame(), error()], one_frame),
            ] {
                let ((plain_registry, plain), (capture_registry, capture), future) =
                    both_legs(outputs).await;

                assert_eq!(plain.unwrap_err(), Rejected::Preflight);
                assert_eq!(capture.unwrap_err(), Rejected::Preflight);
                assert_eq!(signature(&plain_registry), expected);
                assert_eq!(signature(&capture_registry), expected);
                assert!(
                    future
                        .await
                        .drop_reason
                        .as_deref()
                        .unwrap()
                        .contains("backend failed")
                );
            }
        }
    }
}

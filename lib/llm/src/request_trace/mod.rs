// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod agent_context;
pub mod config;
mod integration;
mod otel_sink;
pub mod payload;
pub(crate) mod payload_stream;
mod record;
mod replay;
#[cfg(feature = "request-trace-s3")]
mod s3_sink;
pub mod sink;
mod tool_relay;
pub mod types;

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use dynamo_runtime::DistributedRuntime;

use crate::protocols::openai::responses::ToolNameMap;
use crate::telemetry::bus::TelemetryBus;
use dynamo_model_card::local_model::LocalModel;

pub use agent_context::SharedFinishReasonMetadata;
pub(crate) use agent_context::{
    AgentContextTraceState, build_agent_context_trace_state, into_owned_replay_metrics,
    record_backend_finish_reason_metadata, record_chat_finish_reason_metadata,
    record_completion_finish_reason_metadata, record_llm_metric_tokens, request_metrics,
    request_metrics_from_agent_state, start_request_trace_tool_event_ingest,
};
pub use config::{
    RequestTraceFileFormat, RequestTracePolicy, RequestTraceRecordKind, RequestTraceSinkKind,
    is_enabled, policy,
};
pub(crate) use integration::{
    build_request_end_trace_state, finish_reason_metadata_handle, wrap_chat_request_end_stream,
    wrap_completion_request_end_stream,
};
pub(crate) use record::{emit_router_selection, publish_tool_record, validate_tool_record};
pub(crate) use replay::replay_metrics;
pub use sink::{ActiveInput, TraceShutdownReport, shutdown_workers};
pub use types::{
    ChoiceFinishReasonMetadata, FinishReasonMetadata, RequestReplayMetrics,
    RequestTraceEventSource, RequestTraceEventType, RequestTraceMetrics, RequestTracePayload,
    RequestTraceRecord, RequestTraceRouterSelection, RequestTraceSchema, RequestTraceToolEvent,
    RequestTraceToolEventIngress, RequestTraceToolStatus, RequestTraceWorkerInfo, ToolCallMetadata,
};

static BUS: TelemetryBus<RequestTraceRecord> = TelemetryBus::new();

pub const DEFAULT_TOOL_EVENTS_TOPIC: &str = "agent-tool-events";
pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "request_trace.x_request_id";

/// Context key for an endpoint that gives clients other tool-call IDs than the
/// backend's; its value is the endpoint's [`ClientToolCallIds`].
pub const CLIENT_TOOL_CALL_IDS_CONTEXT_KEY: &str = "request_trace.client_tool_call_ids";

/// How an endpoint turns a backend tool-call ID into the ID its client
/// receives, so `request_end` records the client's ID. Called as
/// `derive(chat_completion_id, backend_id, occurrence)`, where `occurrence`
/// is how many earlier tool calls in the response, suppressed or not, carried
/// the same backend ID. Each call's ID depends only on its own identity, so
/// the endpoint and the trace agree whatever either one filters out.
#[derive(Clone, Copy)]
pub struct ClientToolCallIds(pub fn(&str, &str, u32) -> String);

/// Context key for an endpoint that gives clients other tool names than the
/// backend's; its value is the endpoint's [`ClientToolNames`].
pub const CLIENT_TOOL_NAMES_CONTEXT_KEY: &str = "request_trace.client_tool_names";

/// The request's Responses tool-name map, so `request_end` records each tool
/// call's public leaf name and namespace as the client receives them, not the
/// backend's qualified name or collision alias.
#[derive(Clone)]
pub struct ClientToolNames(pub Arc<ToolNameMap>);

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        config::mark_capture_inactive();
        return Ok(());
    }

    config::mark_capture_inactive();

    if policy.tool_events_zmq_endpoint.is_some()
        && policy.emit_tool_records()
        && policy.sinks.is_empty()
    {
        tracing::warn!(
            tool_events_zmq_endpoint = ?policy.tool_events_zmq_endpoint,
            "request trace tool events are enabled but no local trace sinks are configured; set DYN_REQUEST_TRACE_SINKS to write local trace records"
        );
    }

    BUS.init(policy.capacity);
    sink::spawn_workers_from_env(shutdown).await?;
    config::mark_capture_active();
    tracing::info!(
        capacity = policy.capacity,
        sinks = ?policy.sink_names(),
        file_format = policy.file_format.as_str(),
        records = ?policy.record_names(),
        "Request trace initialized"
    );
    Ok(())
}

pub(crate) async fn start_tool_event_ingest_from_policy(
    drt: DistributedRuntime,
    local_model: &LocalModel,
) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled || !policy.emit_tool_records() {
        return Ok(());
    }

    start_request_trace_tool_event_ingest(
        drt,
        local_model,
        policy.tool_events_zmq_endpoint.clone(),
        policy.tool_events_zmq_topic.clone(),
    )
    .await
}

pub fn publish(record: RequestTraceRecord) {
    BUS.publish(record);
}

/// The router-selection capture spec for an embedded router: `Some` only when
/// request tracing is initialized in this process with `router_selection`
/// records enabled, so a process without a trace bus never pays for capture.
/// Resolved once at router construction.
pub fn router_selection_spec() -> Option<dynamo_worker_selection::host::SelectionTraceSpec> {
    let policy = policy();
    (config::capture_enabled() && policy.emit_router_selection_records()).then_some(
        dynamo_worker_selection::host::SelectionTraceSpec {
            top_k: policy.router_selection_top_k,
        },
    )
}

pub fn subscribe() -> tokio::sync::broadcast::Receiver<RequestTraceRecord> {
    BUS.subscribe()
}

#[cfg(test)]
pub(crate) fn init_bus_for_test(capacity: usize) {
    BUS.init(capacity);
}

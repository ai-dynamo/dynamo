// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod agent_context;
pub mod config;
mod integration;
mod otel_sink;
mod record;
mod replay;
pub mod sink;
mod tool_relay;
pub mod types;

use tokio_util::sync::CancellationToken;

use dynamo_runtime::DistributedRuntime;

use crate::local_model::LocalModel;
use crate::telemetry::bus::TelemetryBus;

pub use agent_context::SharedFinishReasonMetadata;
pub(crate) use agent_context::{
    AgentContextTraceState, build_agent_context_trace_state, into_owned_replay_metrics,
    record_backend_finish_reason_metadata, record_chat_finish_reason_metadata,
    record_completion_finish_reason_metadata, record_llm_metric_tokens, request_metrics,
    request_metrics_from_agent_state, start_request_trace_tool_event_ingest,
};
pub(crate) use config::capture_enabled;
pub use config::{
    RequestTraceFileCompression, RequestTraceFileFormat, RequestTracePolicy, RequestTraceSinkKind,
    is_enabled, policy,
};
pub(crate) use integration::{
    build_request_end_trace_state, finish_reason_metadata_handle, wrap_chat_request_end_stream,
    wrap_completion_request_end_stream,
};
pub(crate) use record::{emit_request_payload, publish_tool_record, validate_tool_record};
pub(crate) use replay::replay_metrics;
pub use types::{
    ChoiceFinishReasonMetadata, FinishReasonMetadata, RequestReplayMetrics,
    RequestTraceEventSource, RequestTraceEventType, RequestTraceMetrics, RequestTracePayload,
    RequestTraceRecord, RequestTraceSchema, RequestTraceToolEvent, RequestTraceToolEventIngress,
    RequestTraceToolStatus, RequestTraceWorkerInfo, ToolCallMetadata,
};

static BUS: TelemetryBus<RequestTraceRecord> = TelemetryBus::new();

pub const DEFAULT_TOOL_EVENTS_TOPIC: &str = "agent-tool-events";
pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "request_trace.x_request_id";

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        config::mark_capture_inactive();
        return Ok(());
    }

    config::mark_capture_inactive();

    if policy.tool_events_zmq_endpoint.is_some() && policy.sinks.is_empty() {
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
        file_compression = policy.file_compression.as_str(),
        include_request_response = policy.include_request_response,
        "Request trace initialized"
    );
    Ok(())
}

pub(crate) async fn start_tool_event_ingest_from_policy(
    drt: DistributedRuntime,
    local_model: &LocalModel,
) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
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

pub fn subscribe() -> tokio::sync::broadcast::Receiver<RequestTraceRecord> {
    BUS.subscribe()
}

#[cfg(test)]
pub(crate) fn init_bus_for_test(capacity: usize) {
    BUS.init(capacity);
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
mod integration;
mod record;
mod relay;
mod replay;
pub mod sink;
pub mod types;

use tokio_util::sync::CancellationToken;

use crate::telemetry::bus::TelemetryBus;

pub use config::{AgentTracePolicy, is_enabled, policy};
pub(crate) use integration::{
    record_llm_metric_tokens, request_metrics, start_tool_event_ingest_from_policy,
};
pub(crate) use record::validate_tool_record;
pub use record::{emit_request_end, publish_tool_record};
pub use relay::AgentToolEventRelay;
pub(crate) use replay::request_replay_metrics;
pub use types::{
    AgentReplayMetrics, AgentRequestMetrics, AgentToolEvent, AgentToolStatus, AgentTraceRecord,
    TraceEventSource, TraceEventType, TraceSchema, WorkerInfo,
};

pub const DEFAULT_TOOL_EVENTS_TOPIC: &str = "agent-tool-events";
pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "agent_trace.x_request_id";

static BUS: TelemetryBus<AgentTraceRecord> = TelemetryBus::new();

pub async fn init_from_env() -> anyhow::Result<()> {
    init_from_env_with_shutdown(CancellationToken::new()).await
}

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    if policy.tool_events_zmq_endpoint.is_some() && policy.sinks.is_empty() {
        tracing::warn!(
            tool_events_zmq_endpoint = ?policy.tool_events_zmq_endpoint,
            "agent trace tool events are enabled but no local trace sinks are configured; set DYN_AGENT_TRACE_SINKS to write local trace records"
        );
    }

    BUS.init(policy.capacity);
    sink::spawn_workers_from_env(shutdown).await?;

    tracing::info!(
        capacity = policy.capacity,
        sinks = ?policy.sinks,
        "Agent trace initialized"
    );
    Ok(())
}

pub fn publish(rec: AgentTraceRecord) {
    BUS.publish(rec);
}

pub fn subscribe() -> tokio::sync::broadcast::Receiver<AgentTraceRecord> {
    BUS.subscribe()
}

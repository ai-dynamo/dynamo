// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tokio_util::sync::CancellationToken;

use dynamo_runtime::DistributedRuntime;

use crate::local_model::LocalModel;
use crate::telemetry::bus::TelemetryBus;

use super::{RequestTraceRecord, policy, sink, start_request_trace_tool_event_ingest};

pub(crate) static BUS: TelemetryBus<RequestTraceRecord> = TelemetryBus::new();

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    if policy.tool_events_zmq_endpoint.is_some() && policy.sinks.is_empty() {
        tracing::warn!(
            tool_events_zmq_endpoint = ?policy.tool_events_zmq_endpoint,
            "request trace tool events are enabled but no local trace sinks are configured; set DYN_REQUEST_TRACE_SINKS to write local trace records"
        );
    }

    BUS.init(policy.capacity);
    sink::spawn_workers_from_env(shutdown).await?;
    tracing::info!(
        capacity = policy.capacity,
        sinks = ?policy.sinks,
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

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
mod integration;
mod record;
pub mod sink;
pub mod types;

use tokio_util::sync::CancellationToken;

use crate::telemetry::bus::TelemetryBus;

pub use config::{RequestTracePolicy, is_enabled, policy};
pub(crate) use integration::{
    build_request_end_trace_state, finish_reason_metadata_handle, wrap_chat_request_end_stream,
    wrap_completion_request_end_stream,
};
pub use types::{
    RequestTraceEventType, RequestTraceMetrics, RequestTraceRecord, RequestTraceSchema,
};

static BUS: TelemetryBus<RequestTraceRecord> = TelemetryBus::new();

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
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

pub fn publish(record: RequestTraceRecord) {
    BUS.publish(record);
}

pub fn subscribe() -> tokio::sync::broadcast::Receiver<RequestTraceRecord> {
    BUS.subscribe()
}

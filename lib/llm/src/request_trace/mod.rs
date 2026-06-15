// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod agent_context;
pub mod config;
mod integration;
mod record;
mod replay;
mod runtime;
pub mod sink;
mod tool_relay;
pub mod types;

pub use agent_context::SharedFinishReasonMetadata;
pub(crate) use agent_context::{
    AgentContextTraceState, build_agent_context_trace_state, into_owned_replay_metrics,
    record_backend_finish_reason_metadata, record_chat_finish_reason_metadata,
    record_completion_finish_reason_metadata, record_llm_metric_tokens, request_metrics,
    request_metrics_from_agent_state, start_request_trace_tool_event_ingest,
};
pub use config::{RequestTracePolicy, is_enabled, policy};
pub(crate) use integration::{
    build_request_end_trace_state, finish_reason_metadata_handle, wrap_chat_request_end_stream,
    wrap_completion_request_end_stream,
};
pub(crate) use record::{publish_tool_record, validate_tool_record};
pub(crate) use replay::replay_metrics;
#[cfg(test)]
pub(crate) use runtime::BUS;
pub(crate) use runtime::start_tool_event_ingest_from_policy;
pub use runtime::{init_from_env_with_shutdown, publish, subscribe};
pub use types::{
    ChoiceFinishReasonMetadata, FinishReasonMetadata, RequestReplayMetrics,
    RequestTraceEventSource, RequestTraceEventType, RequestTraceMetrics, RequestTraceRecord,
    RequestTraceSchema, RequestTraceToolEvent, RequestTraceToolStatus, RequestTraceWorkerInfo,
    ToolCallMetadata,
};

pub const DEFAULT_TOOL_EVENTS_TOPIC: &str = "agent-tool-events";
pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "request_trace.x_request_id";

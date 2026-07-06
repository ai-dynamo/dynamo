// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime adapter for HTTP disconnect handling.

use std::sync::Arc;

use axum::response::sse::Event;
use dynamo_http_server::disconnect::{
    ConnectionHandle, StreamTerminal, backend_stream_timeout, monitor_for_disconnects_with_timeout,
};
use dynamo_http_server::error::SanitizedError;
use dynamo_http_server::metrics::InflightGuard;
use dynamo_runtime::engine::AsyncEngineContext;
use futures::Stream;

fn terminal_events(terminal: StreamTerminal) -> [Option<Event>; 2] {
    match terminal {
        StreamTerminal::Completed => [Some(Event::default().data("[DONE]")), None],
        StreamTerminal::Error => {
            let error = SanitizedError::Internal;
            [
                Some(
                    Event::default().data(
                        serde_json::json!({
                            "error": {
                                "message": error.to_string(),
                                "type": error.openai_type_slug(),
                                "code": error.status().as_u16(),
                            }
                        })
                        .to_string(),
                    ),
                ),
                Some(Event::default().data("[DONE]")),
            ]
        }
    }
}

/// Monitor an HTTP SSE response using the runtime-configured inactivity timeout.
pub fn monitor_for_disconnects(
    stream: impl Stream<Item = Result<Event, axum::Error>>,
    context: Arc<dyn AsyncEngineContext>,
    inflight_guard: InflightGuard,
    stream_handle: ConnectionHandle,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    monitor_for_disconnects_with_timeout(
        stream,
        context,
        inflight_guard,
        stream_handle,
        backend_stream_timeout(),
        terminal_events,
    )
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::realtime::{
    EventType, MaxOutputTokens, RealtimeAPIError, RealtimeClientEvent, RealtimeResponse,
    RealtimeResponseStatus, RealtimeServerEvent, RealtimeServerEventError,
};
use dynamo_runtime::protocols::annotated::Annotated;

pub const ECHO_AUDIO_DELTA_CHUNK_LEN: usize = 64;

pub fn annotated_event(frame: u64, event: RealtimeServerEvent) -> Annotated<RealtimeServerEvent> {
    Annotated {
        id: Some(frame.to_string()),
        ..Annotated::from_data(event)
    }
}

pub fn unsupported_event(
    frame: u64,
    session_id: &str,
    event: &RealtimeClientEvent,
) -> Annotated<RealtimeServerEvent> {
    annotated_event(
        frame,
        RealtimeServerEvent::Error(RealtimeServerEventError {
            event_id: format!("event_{session_id}_{frame}"),
            error: RealtimeAPIError {
                r#type: "invalid_request_error".to_string(),
                code: Some("echo_backend_unsupported".to_string()),
                message: format!(
                    "voice-agent echo backend does not support client event {}",
                    event.event_type()
                ),
                param: None,
                event_id: None,
            },
        }),
    )
}

pub fn echo_response(id: &str, status: RealtimeResponseStatus) -> RealtimeResponse {
    RealtimeResponse {
        audio: None,
        conversation_id: None,
        id: id.to_string(),
        max_output_tokens: MaxOutputTokens::Inf,
        metadata: None,
        object: "realtime.response".to_string(),
        output: Vec::new(),
        output_modalities: vec!["audio".to_string()],
        status,
        status_details: None,
        usage: None,
    }
}

pub fn server_event_finishes_turn(chunk: &Annotated<RealtimeServerEvent>) -> bool {
    matches!(
        chunk.data.as_ref(),
        Some(
            RealtimeServerEvent::SessionUpdated(_)
                | RealtimeServerEvent::ResponseDone(_)
                | RealtimeServerEvent::Error(_)
        )
    )
}

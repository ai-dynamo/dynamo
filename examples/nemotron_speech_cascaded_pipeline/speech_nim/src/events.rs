// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI realtime event projection. No Speech NIM transport types cross this module.

use serde_json::{Value, json};

pub fn id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn event(kind: &str, mut fields: Value) -> Value {
    fields["type"] = kind.into();
    fields["event_id"] = id("event").into();
    fields
}

pub fn invalid(code: &str, message: impl Into<String>, client_id: &Value) -> Value {
    event(
        "error",
        json!({"error": {
            "type": "invalid_request_error", "code": code,
            "message": message.into(), "event_id": client_id,
        }}),
    )
}

pub fn updated(session: Value) -> Value {
    event("session.updated", json!({"session": session}))
}

pub fn committed(item_id: &str) -> Value {
    event(
        "input_audio_buffer.committed",
        json!({"previous_item_id": null, "item_id": item_id}),
    )
}

pub fn cleared() -> Value {
    event("input_audio_buffer.cleared", json!({}))
}

pub fn delta(item_id: &str, delta: &str) -> Value {
    event(
        "conversation.item.input_audio_transcription.delta",
        json!({
            "item_id": item_id, "content_index": 0, "delta": delta, "logprobs": null,
        }),
    )
}

pub fn completed(item_id: &str, transcript: &str) -> Value {
    event(
        "conversation.item.input_audio_transcription.completed",
        json!({
            "item_id": item_id, "content_index": 0, "transcript": transcript, "logprobs": null,
            "usage": {"type": "tokens", "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "input_token_details": {"audio_tokens": 0, "text_tokens": 0}},
        }),
    )
}

pub fn failed(item_id: &str, message: &str) -> Value {
    event(
        "conversation.item.input_audio_transcription.failed",
        json!({
            "item_id": item_id, "content_index": 0,
            "error": {"type": "server_error", "code": "transcription_error", "message": message},
        }),
    )
}

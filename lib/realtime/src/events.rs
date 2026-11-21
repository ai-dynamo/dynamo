// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use serde_json::{Value, json};
use uuid::Uuid;

pub(crate) fn id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

pub(crate) fn event(kind: &str, mut fields: Value) -> Value {
    fields["type"] = kind.into();
    fields["event_id"] = id("event").into();
    fields
}

pub(crate) fn invalid(code: &str, message: &str, client: &Value) -> Value {
    event(
        "error",
        json!({"error": {
            "type": "invalid_request_error", "code": code, "message": message,
            "event_id": client.get("event_id"),
        }}),
    )
}

pub(crate) fn transcription_failed(item: &str) -> Value {
    event(
        "conversation.item.input_audio_transcription.failed",
        json!({
            "item_id": item, "content_index": 0,
            "error": {"type": "server_error", "code": "transcription_error",
                      "message": "Upstream transcription failed"}
        }),
    )
}

pub(crate) fn omni_failed(response: &str, modalities: &Value) -> Value {
    event(
        "response.done",
        json!({"response": {
            "id": response, "object": "realtime.response", "status": "failed",
            "output": [], "output_modalities": modalities, "max_output_tokens": "inf",
            "status_details": {"type": "failed", "error": {
                "type": "server_error", "code": "upstream_error"
            }}
        }}),
    )
}

pub(crate) fn asr_output(
    upstream: Value,
    item: &str,
    audio_bytes: usize,
) -> Result<Option<Value>, Error> {
    let invalid = || Error::Protocol("malformed transcription event".into());
    Ok(match upstream["type"].as_str().ok_or_else(invalid)? {
        "transcription.delta" => {
            let delta = upstream["delta"].as_str().ok_or_else(invalid)?;
            Some(event(
                "conversation.item.input_audio_transcription.delta",
                json!({"item_id": item, "content_index": 0, "delta": delta, "logprobs": null}),
            ))
        }
        "transcription.done" => {
            let text = upstream["text"].as_str().ok_or_else(invalid)?;
            let mut completed = event(
                "conversation.item.input_audio_transcription.completed",
                json!({
                    "item_id": item, "content_index": 0, "transcript": text, "logprobs": null,
                }),
            );
            if !upstream["usage"].is_null() {
                let count = |key: &str| {
                    upstream["usage"][key]
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or_else(invalid)
                };
                let input = count("prompt_tokens")?;
                let output = count("completion_tokens")?;
                let total = input.checked_add(output).ok_or_else(invalid)?;
                completed["usage"] = json!({"type": "tokens", "input_tokens": input, "output_tokens": output,
                          "total_tokens": total,
                          "input_token_details": {"audio_tokens": input, "text_tokens": 0}});
            } else {
                // Public completion events require usage. Duration is measurable
                // even when upstream omits token accounting.
                completed["usage"] = json!({"type": "duration",
                    "seconds": audio_bytes as f64 / (2 * crate::audio::OPENAI_PCM_RATE) as f64});
            }
            Some(completed)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_usage_reports_audio_duration_and_malformed_text_is_rejected() {
        let output = asr_output(
            json!({"type": "transcription.done", "text": "hello"}),
            "item",
            24_000,
        )
        .unwrap()
        .unwrap();
        assert_eq!(output["usage"], json!({"type": "duration", "seconds": 0.5}));
        serde_json::from_value::<dynamo_protocols::types::realtime::RealtimeServerEvent>(output)
            .unwrap();
        assert!(asr_output(json!({"type": "transcription.done"}), "item", 0).is_err());
    }
}

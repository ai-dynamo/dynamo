// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Deployment validation around Switchyard's OpenAI Chat decoder.
use anyhow::{Result, bail, ensure};
use protocol::{ContentBlock, Metadata, Request};
use serde_json::Value;
use switchyard_translation::{
    DeterministicIdPolicy, LossyConversionPolicy, PreservationPolicy, TranslationPolicy,
    codecs::{FormatCodec, openai_chat::OpenAiChatCodec},
};

fn validate_text(content: Option<&Value>) -> Result<()> {
    match content {
        None | Some(Value::Null | Value::String(_)) => Ok(()),
        Some(Value::Array(parts)) => {
            for part in parts {
                ensure!(
                    part.get("type").and_then(Value::as_str) == Some("text")
                        && part.get("text").is_some_and(Value::is_string),
                    "only text content is supported by this model catalog"
                );
            }
            Ok(())
        }
        _ => bail!("message content must be text, text parts, or null"),
    }
}

pub fn decode(raw: &Value, headers: &http::HeaderMap) -> Result<Request> {
    validate_deployment(raw)?;
    let policy = TranslationPolicy {
        // Forward the original body; no IR-to-wire conversion or retained body copy is needed.
        preservation: PreservationPolicy::Disabled,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        deterministic_ids: DeterministicIdPolicy::Preserve,
        ..Default::default()
    };
    let mut llm_request = OpenAiChatCodec.decode_request(raw, &policy)?.request;
    // SDK 0.3.0 drops tool is_error; restore this routing signal.
    // Remove this projection when the SDK decoder preserves tool error flags.
    let tool_messages = raw
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|message| message["role"] == "tool");
    let results = llm_request
        .messages
        .iter_mut()
        .flat_map(|message| &mut message.content)
        .filter_map(|block| match block {
            ContentBlock::ToolResult(result) => Some(result),
            _ => None,
        });
    for (message, result) in tool_messages.zip(results) {
        result.is_error = message.get("is_error").and_then(Value::as_bool);
    }
    Ok(Request {
        llm_request,
        raw_request: None,
        metadata: Some(Metadata::from_headers(headers)),
    })
}

fn validate_deployment(raw: &Value) -> Result<()> {
    if let Some(nvext) = raw.get("nvext").filter(|v| !v.is_null()) {
        ensure!(nvext.is_object(), "nvext must be an object");
        for reserved in [
            "token_data",
            "backend_instance_id",
            "prefill_worker_id",
            "decode_worker_id",
            "dp_rank",
            "prefill_dp_rank",
            "routing_constraints",
            "router",
            "use_raw_prompt",
            "metadata_upload",
        ] {
            ensure!(
                nvext.get(reserved).is_none_or(Value::is_null),
                "nvext contains a reserved worker or preprocessing control"
            );
        }
    }
    if let Some(messages) = raw.get("messages").and_then(Value::as_array) {
        for message in messages {
            validate_text(message.get("content"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decodes_chat_controls_and_preserves_tool_error_flags() {
        let raw = json!({
            "model": "auto", "temperature": 0.2, "max_tokens": 64,
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "Bash", "arguments": "{\"command\":\"pytest\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "failed", "is_error": true},
                {"role": "developer", "content": "Check the result"},
                {"role": "tool", "tool_call_id": "compacted", "content": "passed", "is_error": false}
            ],
            "tools": [{"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}}],
            "tool_choice": "required"
        });
        let mut headers = http::HeaderMap::new();
        headers.insert("x-switchyard-session-id", "session-1".parse().unwrap());
        let request = decode(&raw, &headers).unwrap();
        let ir = request.llm_request;
        assert_eq!(ir.instructions.len(), 2);
        assert_eq!(ir.sampling.temperature, Some(0.2));
        assert_eq!(ir.output.max_output_tokens, Some(64));
        assert_eq!(ir.tool_choice, Some(protocol::ToolChoice::Required));
        assert_eq!(ir.tools[0].name, "Bash");
        let ContentBlock::ToolCall(call) = &ir.messages[0].content[0] else {
            panic!("missing tool call")
        };
        assert_eq!(call.arguments, json!({"command": "pytest"}));
        for (index, flag) in [(1, true), (2, false)] {
            let ContentBlock::ToolResult(result) = &ir.messages[index].content[0] else {
                panic!("missing tool result")
            };
            assert_eq!(result.is_error, Some(flag));
        }
        assert!(ir.preservation.requests.is_empty());
        assert!(request.raw_request.is_none());
        assert_eq!(
            request.metadata.unwrap().session_id.as_deref(),
            Some("session-1")
        );
    }

    #[test]
    fn rejects_reserved_controls_and_unsupported_media() {
        for extra in [
            json!({"nvext": {"backend_instance_id": 42}}),
            json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}]}]}),
        ] {
            let mut raw =
                json!({"model": "auto", "messages": [{"role": "user", "content": "hello"}]});
            raw.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            assert!(
                decode(&raw, &http::HeaderMap::new()).is_err(),
                "accepted {raw}"
            );
        }
    }
}

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
use std::collections::HashSet;

use anyhow::{Result, bail, ensure};
use protocol::{ContentBlock, Metadata, Request, Role};
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
    validate(raw)?;
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
    let tool_messages = raw["messages"]
        .as_array()
        .unwrap()
        .iter()
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

fn validate(raw: &Value) -> Result<()> {
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
    let messages = raw
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("messages must be an array"))?;
    ensure!(!messages.is_empty(), "messages must not be empty");
    ensure!(
        raw.get("stream").is_none_or(Value::is_boolean),
        "stream must be a boolean"
    );
    let mut pending = HashSet::new();
    for message in messages {
        let role = match message.get("role").and_then(Value::as_str) {
            Some("system") => Role::System,
            Some("developer") => Role::Developer,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            _ => bail!("unsupported or missing message role"),
        };
        validate_text(message.get("content"))?;
        if let Some(calls) = message.get("tool_calls") {
            ensure!(role == Role::Assistant, "tool_calls require assistant role");
            let calls = calls
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("tool_calls must be an array"))?;
            for call in calls {
                ensure!(
                    call.get("type").and_then(Value::as_str) == Some("function"),
                    "only function tool calls are supported"
                );
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("tool call needs an id"))?;
                ensure!(
                    pending.insert(id.to_owned()),
                    "duplicate outstanding tool call id"
                );
                let function = call
                    .get("function")
                    .ok_or_else(|| anyhow::anyhow!("tool call needs function"))?;
                function
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("tool call needs name"))?;
                function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("tool call arguments must be a JSON string"))?;
            }
        }
        if role == Role::Tool {
            let id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("tool result needs tool_call_id"))?;
            pending.remove(id);
            message
                .get("is_error")
                .map(|v| {
                    v.as_bool()
                        .ok_or_else(|| anyhow::anyhow!("is_error must be boolean"))
                })
                .transpose()?;
        }
    }
    if let Some(tools) = raw.get("tools") {
        for tool in tools
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tools must be an array"))?
        {
            ensure!(
                tool.get("type").and_then(Value::as_str) == Some("function"),
                "only function tools are supported"
            );
            let f = tool
                .get("function")
                .ok_or_else(|| anyhow::anyhow!("tool needs function"))?;
            f.get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool needs name"))?;
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
    fn rejects_invalid_or_reserved_input_before_permissive_codec_normalization() {
        for extra in [
            json!({"stream": "yes"}),
            json!({"nvext": {"backend_instance_id": 42}}),
            json!({"tools": {"type": "function"}}),
            json!({"tools": [{"type": "function", "function": {}}]}),
            json!({"messages": [{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": "{}"}}]}]}),
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

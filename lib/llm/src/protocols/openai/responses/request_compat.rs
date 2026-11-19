// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalize coding-agent extensions before typed Responses admission.

use std::collections::{HashMap, HashSet};

use dynamo_protocols::types::{
    ChatCompletionRequestMessage as Message, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestToolMessageContent as ToolContent,
    ChatCompletionRequestToolMessageContentPart as ToolPart, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContentPart as UserPart,
};
use serde_json::Value;

use super::ResponsesConversionError;

pub(crate) fn normalize_input(body: &mut Value) -> Result<(), ResponsesConversionError> {
    if let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("custom_tool_call_output") {
                item["type"] = Value::String("function_call_output".into());
            }
        }
    }
    hoist_additional_tools(body)
}

fn hoist_additional_tools(body: &mut Value) -> Result<(), ResponsesConversionError> {
    let Some(items) = body.get("input").and_then(Value::as_array) else {
        return Ok(());
    };
    if !items
        .iter()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("additional_tools"))
    {
        return Ok(());
    }
    if body
        .get("tools")
        .is_some_and(|tools| !tools.is_null() && !tools.is_array())
    {
        return Err(ResponsesConversionError::InvalidArgument(
            "tools must be an array".into(),
        ));
    }
    if items.iter().any(|item| {
        item.get("type").and_then(Value::as_str) == Some("additional_tools")
            && !item.get("tools").is_some_and(Value::is_array)
    }) {
        return Err(ResponsesConversionError::InvalidArgument(
            "additional_tools input must contain a tools array".into(),
        ));
    }
    let Value::Array(items) = body["input"].take() else {
        unreachable!("input array was validated");
    };
    let mut tools = match body["tools"].take() {
        Value::Array(tools) => tools,
        _ => Vec::new(),
    };
    let mut input = Vec::with_capacity(items.len());
    for mut item in items {
        if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
            let Value::Array(additional) = item["tools"].take() else {
                unreachable!("additional tools array was validated");
            };
            tools.extend(additional);
        } else {
            input.push(item);
        }
    }
    body["input"] = Value::Array(input);
    body["tools"] = Value::Array(tools);
    Ok(())
}

/// Routing hint only; this does not authenticate a session or reconstruct history.
pub(crate) fn body_affinity(body: &Value) -> Option<&str> {
    [
        body.pointer("/conversation_params/conversation_id"),
        body.pointer("/client_metadata/thread_id"),
        body.pointer("/client_metadata/session_id"),
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_str)
    .map(str::trim)
    .find(|value| !value.is_empty())
}

/// Kimi numbers results within the most recent assistant call batch.
/// Merge split results within a run, then retain unmatched output as user notes.
pub(super) fn kimi_history(messages: Vec<Message>) -> Vec<Message> {
    if !messages
        .iter()
        .any(|message| matches!(message, Message::Tool(_)))
    {
        return messages;
    }

    let mut merged = Vec::with_capacity(messages.len());
    let mut run: HashMap<String, usize> = HashMap::new();
    for message in messages {
        if let Message::Tool(tool) = message {
            if let Some(index) = run.get(&tool.tool_call_id).copied()
                && let Message::Tool(first) = &mut merged[index]
            {
                merge_tool_content(&mut first.content, tool.content);
                continue;
            }
            run.insert(tool.tool_call_id.clone(), merged.len());
            merged.push(Message::Tool(tool));
        } else {
            run.clear();
            merged.push(message);
        }
    }

    let mut settled = Vec::with_capacity(merged.len());
    let mut notes = Vec::new();
    let mut calls = HashSet::new();
    for message in merged {
        if let Message::Tool(tool) = message {
            if calls.contains(&tool.tool_call_id) {
                settled.push(Message::Tool(tool));
            } else {
                let id = tool.tool_call_id;
                let mut parts = vec![UserPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: format!(
                            "[Output of tool call '{id}', which matches no tool call before it:]"
                        ),
                        prompt_cache_breakpoint: None,
                    },
                )];
                parts.extend(tool_parts(tool.content).into_iter().map(|part| match part {
                    ToolPart::Text(text) => UserPart::Text(text),
                    ToolPart::ImageUrl(image) => UserPart::ImageUrl(image),
                    ToolPart::VideoUrl(video) => UserPart::VideoUrl(video),
                    ToolPart::AudioUrl(audio) => UserPart::AudioUrl(audio),
                    ToolPart::Custom(custom) => UserPart::Custom(custom),
                }));
                notes.push(Message::User(ChatCompletionRequestUserMessage {
                    content: parts.into(),
                    name: None,
                }));
            }
        } else {
            settled.append(&mut notes);
            calls.clear();
            if let Message::Assistant(assistant) = &message {
                calls.extend(
                    assistant
                        .tool_calls
                        .iter()
                        .flatten()
                        .map(|call| call.id.clone()),
                );
            }
            settled.push(message);
        }
    }
    settled.append(&mut notes);
    settled
}

fn merge_tool_content(first: &mut ToolContent, next: ToolContent) {
    match (first, next) {
        (ToolContent::Text(first), ToolContent::Text(next)) => {
            if !first.is_empty() && !first.ends_with('\n') {
                first.push('\n');
            }
            first.push_str(&next);
        }
        (ToolContent::Array(parts), next) => parts.extend(tool_parts(next)),
        (first, next) => {
            let mut parts = tool_parts(std::mem::take(first));
            parts.extend(tool_parts(next));
            *first = ToolContent::Array(parts);
        }
    }
}

fn tool_parts(content: ToolContent) -> Vec<ToolPart> {
    match content {
        ToolContent::Array(parts) => parts,
        ToolContent::Text(text) => vec![ToolPart::Text(
            ChatCompletionRequestMessageContentPartText {
                text,
                prompt_cache_breakpoint: None,
            },
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::responses::NvCreateResponse;
    use serde_json::json;

    #[test]
    fn hoisted_namespaces_pass_typed_validation() {
        let mut body = json!({"model":"m", "input":[
            {"type":"additional_tools", "tools":[{"type":"namespace", "name":"functions", "tools":[
                {"type":"function", "name":"exec", "parameters":{"type":"object"}}
            ]}]}, {"role":"user", "content":"run pwd"}
        ]});
        hoist_additional_tools(&mut body).unwrap();
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        let request: NvCreateResponse = serde_json::from_value(body).unwrap();
        assert_eq!(request.inner.tools.unwrap().len(), 1);
    }

    #[test]
    fn malformed_declarations_fail_without_mutating_history() {
        let mut body = json!({"input":[{"type":"additional_tools","tools":{}}]});
        let original = body.clone();
        assert!(hoist_additional_tools(&mut body).is_err());
        assert_eq!(body, original);
    }

    #[test]
    fn kimi_split_results_do_not_shift_call_indices_or_lose_orphans() {
        let messages: Vec<super::super::ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role":"assistant","content":null,"tool_calls":[
                    {"type":"function","id":"a","function":{"name":"one","arguments":"{}"}},
                    {"type":"function","id":"b","function":{"name":"two","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"a","content":"first"},
                {"role":"tool","tool_call_id":"missing","content":"orphan"},
                {"role":"tool","tool_call_id":"b","content":"second"},
                {"role":"tool","tool_call_id":"a","content":"continued"}
            ]))
            .unwrap();
        let normalized = kimi_history(messages);
        let value = serde_json::to_value(&normalized).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 4);
        assert_eq!(value[1]["content"], "first\ncontinued");
        assert_eq!(value[2]["tool_call_id"], "b");
        assert_eq!(value[3]["role"], "user");
        assert_eq!(value[3]["content"][1]["text"], "orphan");
        assert_eq!(kimi_history(normalized.clone()), normalized);
    }

    #[test]
    fn kimi_user_message_ends_the_tool_call_batch() {
        let messages: Vec<super::super::ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role":"assistant","content":null,"tool_calls":[
                    {"type":"function","id":"a","function":{"name":"one","arguments":"{}"}}
                ]},
                {"role":"user","content":"new turn"},
                {"role":"tool","tool_call_id":"a","content":"late result"}
            ]))
            .unwrap();
        let normalized = kimi_history(messages);
        let value = serde_json::to_value(&normalized).unwrap();
        assert_eq!(value[2]["role"], "user");
        assert_eq!(value[2]["content"][1]["text"], "late result");
        assert_eq!(kimi_history(normalized.clone()), normalized);
    }

    #[test]
    fn affinity_uses_first_nonempty_string() {
        let body = json!({"conversation_params":{"conversation_id":" parent "},"client_metadata":{"thread_id":"child"}});
        assert_eq!(body_affinity(&body), Some("parent"));
        assert_eq!(
            body_affinity(
                &json!({"conversation_params":{"conversation_id":" "},"client_metadata":{"thread_id":"child"}})
            ),
            Some("child")
        );
        for cache_key in [
            json!("support-bot-v3"),
            json!("x".repeat(257)),
            json!(false),
        ] {
            assert_eq!(body_affinity(&json!({"prompt_cache_key":cache_key})), None);
        }
    }
}

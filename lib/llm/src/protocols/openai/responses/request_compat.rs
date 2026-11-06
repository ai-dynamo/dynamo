// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalize coding-agent extensions before typed Responses admission.

use serde_json::Value;

use super::ResponsesConversionError;

pub(crate) fn hoist_additional_tools(body: &mut Value) -> Result<(), ResponsesConversionError> {
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
    let mut tools = body
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut input = Vec::with_capacity(items.len());
    for item in items {
        if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
            let Some(additional) = item.get("tools").and_then(Value::as_array) else {
                return Err(ResponsesConversionError::InvalidArgument(
                    "additional_tools input must contain a tools array".into(),
                ));
            };
            tools.extend(additional.iter().cloned());
        } else {
            input.push(item.clone());
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
        body.get("prompt_cache_key"),
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
pub(super) fn kimi_history(
    messages: &[super::ChatCompletionRequestMessage],
) -> anyhow::Result<Vec<super::ChatCompletionRequestMessage>> {
    use std::collections::{HashMap, HashSet};
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    let mut run: HashMap<String, usize> = HashMap::new();
    for message in messages {
        let message = serde_json::to_value(message)?;
        if message["role"] != "tool" {
            run.clear();
        } else if let Some(id) = message["tool_call_id"].as_str() {
            if let Some(index) = run.get(id).copied() {
                let first = &mut merged[index]["content"];
                if let (Some(a), Some(b)) = (first.as_str(), message["content"].as_str()) {
                    let separator = if a.is_empty() || a.ends_with('\n') {
                        ""
                    } else {
                        "\n"
                    };
                    *first = Value::String(format!("{a}{separator}{b}"));
                } else {
                    let mut parts = content_parts(first);
                    parts.extend(content_parts(&message["content"]));
                    *first = Value::Array(parts);
                }
                continue;
            }
            run.insert(id.to_string(), merged.len());
        }
        merged.push(message);
    }
    let mut settled = Vec::with_capacity(merged.len());
    let mut notes = Vec::new();
    let mut calls: HashSet<String> = HashSet::new();
    for message in merged {
        if message["role"] == "tool" {
            let id = message["tool_call_id"].as_str().unwrap_or_default();
            if calls.contains(id) {
                settled.push(message);
            } else {
                let mut parts = vec![serde_json::json!({"type":"text", "text":format!(
                    "[Output of tool call '{id}', which matches no tool call before it:]"
                )})];
                parts.extend(content_parts(&message["content"]));
                notes.push(serde_json::json!({"role":"user", "content":parts}));
            }
            continue;
        }
        settled.append(&mut notes);
        if message["role"] == "assistant" {
            calls = message["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|call| call["id"].as_str().map(str::to_owned))
                .collect();
        }
        settled.push(message);
    }
    settled.append(&mut notes);
    Ok(serde_json::from_value(Value::Array(settled))?)
}

fn content_parts(content: &Value) -> Vec<Value> {
    content.as_array().cloned().unwrap_or_else(|| {
        vec![serde_json::json!({"type":"text", "text":content.as_str().unwrap_or_default()})]
    })
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
        let normalized = kimi_history(&messages).unwrap();
        let value = serde_json::to_value(&normalized).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 4);
        assert_eq!(value[1]["content"], "first\ncontinued");
        assert_eq!(value[2]["tool_call_id"], "b");
        assert_eq!(value[3]["role"], "user");
        assert_eq!(value[3]["content"][1]["text"], "orphan");
        assert_eq!(kimi_history(&normalized).unwrap(), normalized);
    }

    #[test]
    fn affinity_uses_first_nonempty_string() {
        let body = json!({"prompt_cache_key":" parent ","client_metadata":{"thread_id":"child"}});
        assert_eq!(body_affinity(&body), Some("parent"));
        assert_eq!(
            body_affinity(&json!({"prompt_cache_key":" ","client_metadata":{"thread_id":"child"}})),
            Some("child")
        );
        assert_eq!(body_affinity(&json!({"prompt_cache_key":false})), None);
    }
}

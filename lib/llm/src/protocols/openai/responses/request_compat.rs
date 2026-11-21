// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalize coding-agent extensions before typed Responses admission.

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

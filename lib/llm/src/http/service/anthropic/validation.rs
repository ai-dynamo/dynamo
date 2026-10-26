// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, value::RawValue};

/// Validate wire fields before the shared protocol types discard their shape
/// (notably system block types and malformed untagged tool-choice variants).
pub(super) struct ValidatedRequest<T>(pub T);

impl<'de, T: DeserializeOwned> Deserialize<'de> for ValidatedRequest<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Axum's JSON extractor buffers the body, so borrow its original JSON.
        // Parsing T from a Value would silently discard duplicate object keys.
        let raw = <&RawValue>::deserialize(deserializer)?;
        let value = serde_json::from_str(raw.get()).map_err(serde::de::Error::custom)?;
        validate(&value).map_err(serde::de::Error::custom)?;
        drop(value);
        serde_json::from_str(raw.get())
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

fn validate(request: &Value) -> Result<(), String> {
    if let Some(system) = request.get("system").filter(|value| !value.is_null()) {
        match system {
            Value::String(_) => {}
            Value::Array(blocks) => {
                for (index, block) in blocks.iter().enumerate() {
                    if block.get("type").and_then(Value::as_str) != Some("text") {
                        return Err(format!("system[{index}].type: must be text"));
                    }
                    if block
                        .get("text")
                        .and_then(Value::as_str)
                        .is_none_or(|text| text.is_empty())
                    {
                        return Err(format!("system[{index}].text: must be a non-empty string"));
                    }
                }
            }
            _ => return Err("system: must be a string or an array of text blocks".into()),
        }
    }

    if let Some(choice) = request.get("tool_choice").filter(|value| !value.is_null()) {
        match choice.get("type").and_then(Value::as_str) {
            Some("tool") => {
                if choice
                    .get("name")
                    .and_then(Value::as_str)
                    .is_none_or(|name| name.is_empty())
                {
                    return Err("tool_choice.name: must be a non-empty string for type tool".into());
                }
            }
            Some("auto" | "any" | "none") => {
                if choice.get("name").is_some() {
                    return Err("tool_choice.name: only allowed for type tool".into());
                }
            }
            _ => return Err("tool_choice.type: must be auto, any, none, or tool".into()),
        }
        if let Some(parallel) = choice.get("disable_parallel_tool_use") {
            if choice["type"] == "none" {
                return Err(
                    "tool_choice.disable_parallel_tool_use: not allowed for type none".into(),
                );
            }
            if !parallel.is_boolean() {
                return Err("tool_choice.disable_parallel_tool_use: must be a boolean".into());
            }
        }
    }

    if let Some(metadata) = request.get("metadata").filter(|value| !value.is_null()) {
        if !metadata.is_object() {
            return Err("metadata: must be an object".into());
        }
        if let Some(user_id) = metadata.get("user_id").filter(|value| !value.is_null())
            && user_id
                .as_str()
                .is_none_or(|id| id.chars().take(513).count() > 512)
        {
            return Err("metadata.user_id: must be a string of at most 512 characters".into());
        }
    }

    if let Some(thinking) = request.get("thinking").filter(|value| !value.is_null()) {
        match thinking.get("type").and_then(Value::as_str) {
            Some("enabled") => {
                let budget = thinking
                    .get("budget_tokens")
                    .and_then(Value::as_u64)
                    .filter(|budget| *budget >= 1024)
                    .ok_or("thinking.budget_tokens: must be an integer of at least 1024 for type enabled")?;
                if let Some(max_tokens) = request.get("max_tokens").and_then(Value::as_u64)
                    && budget >= max_tokens
                {
                    return Err("thinking.budget_tokens: must be less than max_tokens".into());
                }
            }
            Some("disabled" | "adaptive") => {
                if thinking.get("budget_tokens").is_some() {
                    return Err("thinking.budget_tokens: only allowed for type enabled".into());
                }
            }
            _ => return Err("thinking.type: must be enabled, disabled, or adaptive".into()),
        }
    }
    Ok(())
}

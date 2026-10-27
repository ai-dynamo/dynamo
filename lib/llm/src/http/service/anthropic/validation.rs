// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, marker::PhantomData};

use axum::{Json, http::HeaderMap};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, value::RawValue};

/// Retain the original JSON until header-aware admission. Typed deserialization
/// must read the original bytes, since a Value loses duplicate object keys.
pub(super) struct ValidatedRequest<T> {
    raw: Box<RawValue>,
    marker: PhantomData<T>,
}

impl<'de, T> Deserialize<'de> for ValidatedRequest<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            raw: Box::<RawValue>::deserialize(deserializer)?,
            marker: PhantomData,
        })
    }
}

impl<T: DeserializeOwned> ValidatedRequest<T> {
    fn fields(&self) -> Result<HashMap<String, &RawValue>, String> {
        serde_json::from_str(self.raw.get()).map_err(|error| error.to_string())
    }

    /// Best-effort labels only; admission below still rejects malformed fields.
    pub(super) fn identity(&self) -> (String, bool) {
        let fields = self.fields().unwrap_or_default();
        let model = fields
            .get("model")
            .and_then(|raw| serde_json::from_str(raw.get()).ok())
            .unwrap_or_default();
        let stream = fields
            .get("stream")
            .and_then(|raw| serde_json::from_str(raw.get()).ok())
            .unwrap_or(false);
        (model, stream)
    }

    pub(super) fn into_request(self, headers: &HeaderMap) -> Result<T, String> {
        let fields = self.fields()?;
        // Inspect only fields whose shape the shared types would discard. Unknown
        // fields keep the typed parser's permissive behavior and numeric range.
        let mut values = serde_json::Map::new();
        for name in [
            "system",
            "tool_choice",
            "metadata",
            "thinking",
            "max_tokens",
            "top_k",
        ] {
            if let Some(raw) = fields.get(name) {
                values.insert(
                    name.into(),
                    serde_json::from_str(raw.get()).map_err(|error| format!("{name}: {error}"))?,
                );
            }
        }
        validate(&Value::Object(values), headers)?;
        // Axum retains typed field paths in diagnostics, including nested ones.
        Json::<T>::from_bytes(self.raw.get().as_bytes())
            .map(|Json(request)| request)
            .map_err(|error| error.body_text())
    }
}

fn has_beta(headers: &HeaderMap, beta: &str) -> bool {
    headers
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|value| value.trim() == beta)
}

fn validate(request: &Value, headers: &HeaderMap) -> Result<(), String> {
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
                    && !has_beta(headers, "interleaved-thinking-2025-05-14")
                {
                    return Err("thinking.budget_tokens: must be less than max_tokens".into());
                }
            }
            Some("disabled" | "adaptive" | "between_tools") => {
                if thinking.get("budget_tokens").is_some() {
                    return Err("thinking.budget_tokens: only allowed for type enabled".into());
                }
            }
            _ => {
                return Err(
                    "thinking.type: must be enabled, disabled, adaptive, or between_tools".into(),
                );
            }
        }
    }
    if request
        .get("top_k")
        .and_then(Value::as_u64)
        .is_some_and(|value| value > i32::MAX as u64)
    {
        return Err("top_k: must be at most 2147483647".into());
    }
    Ok(())
}

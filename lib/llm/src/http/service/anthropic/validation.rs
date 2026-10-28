// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, marker::PhantomData};

use axum::{
    Json,
    extract::{FromRequest, Request, rejection::MissingJsonContentType},
    http::{HeaderMap, header},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde_json::{Value, value::RawValue};

/// Retain the ingress buffer and small admission fields. Borrowed raw fields
/// avoid copying messages, images, or tool schemas before typed deserialization.
pub(super) struct ValidatedRequest<T> {
    body: Bytes,
    messages: Option<Bytes>,
    model: String,
    stream: bool,
    fields: Result<Value, String>,
    marker: PhantomData<T>,
}

impl<S: Send + Sync, T: DeserializeOwned + Send> FromRequest<S> for ValidatedRequest<T> {
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let is_json = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<mime::Mime>().ok())
            .is_some_and(|mime| {
                mime.type_() == mime::APPLICATION
                    && (mime.subtype() == mime::JSON || mime.suffix() == Some(mime::JSON))
            });
        if !is_json {
            return Err(MissingJsonContentType::default().into_response());
        }
        let body = Bytes::from_request(request, state)
            .await
            .map_err(IntoResponse::into_response)?;
        Ok(Self::from_bytes(body))
    }
}

impl<T: DeserializeOwned> ValidatedRequest<T> {
    pub(super) fn from_bytes(body: Bytes) -> Self {
        let mut messages = None;
        let mut model = String::new();
        let mut stream = false;
        let fields = (|| {
            let raw: HashMap<String, &RawValue> =
                serde_json::from_slice(&body).map_err(|error| error.to_string())?;
            messages = raw
                .get("messages")
                .map(|raw| body.slice_ref(raw.get().as_bytes()));
            model = raw
                .get("model")
                .and_then(|raw| serde_json::from_str(raw.get()).ok())
                .unwrap_or_default();
            stream = raw
                .get("stream")
                .and_then(|raw| serde_json::from_str(raw.get()).ok())
                .unwrap_or(false);
            let mut values = serde_json::Map::new();
            for (name, value) in raw {
                let value = if matches!(
                    name.as_str(),
                    "system" | "tool_choice" | "metadata" | "thinking" | "max_tokens" | "top_k"
                ) {
                    serde_json::from_str(value.get()).map_err(|error| format!("{name}: {error}"))?
                } else {
                    // Presence is sufficient for version checks. Unknown values
                    // retain the typed parser's permissive numeric range.
                    Value::Null
                };
                values.insert(name, value);
            }
            Ok(Value::Object(values))
        })();
        Self {
            body,
            messages,
            model,
            stream,
            fields,
            marker: PhantomData,
        }
    }

    /// Best-effort labels; validation and typed parsing run inside the handler's metrics guard.
    pub(super) fn identity(&self) -> (String, bool) {
        (self.model.clone(), self.stream)
    }

    pub(super) fn into_request(self, headers: &HeaderMap) -> Result<T, String> {
        let fields = self.fields?;
        let mut versions = headers.get_all("anthropic-version").iter();
        let version = versions.next();
        if versions.next().is_some() {
            return Err("anthropic-version: supply exactly one version header".into());
        }
        match version.map(|value| value.to_str()) {
            None | Some(Ok("2023-06-01")) => {}
            Some(Ok("2026-10-21")) => {
                if let Some(messages) = &self.messages {
                    reject_user_citations(messages)?;
                }
                if let Some(parameter) = ["temperature", "top_p", "top_k"]
                    .into_iter()
                    .find(|parameter| fields.get(*parameter).is_some())
                {
                    return Err(format!(
                        "{parameter}: not allowed with anthropic-version 2026-10-21; remove the parameter or use 2023-06-01"
                    ));
                }
            }
            _ => {
                return Err(
                    "anthropic-version: supported versions are 2023-06-01 and 2026-10-21".into(),
                );
            }
        }

        validate(&fields, headers)?;
        drop(fields);
        // Read the original bytes, preserving duplicate-field rejection and Axum's
        // nested field paths. Do not reconstruct the request through a Value.
        Json::<T>::from_bytes(&self.body)
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

// Inspect citations before tool-result conversion can discard them. Skip text,
// image data, and tool inputs instead of materializing a second message tree.
fn reject_user_citations(messages: &[u8]) -> Result<(), String> {
    use serde::{
        Deserialize,
        de::{self, MapAccess, SeqAccess, Visitor},
    };

    #[derive(Default)]
    struct CitationScan {
        cited_blocks: bool,
        cited_users: bool,
    }

    impl<'de> Deserialize<'de> for CitationScan {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct ScanVisitor;
            impl<'de> Visitor<'de> for ScanVisitor {
                type Value = CitationScan;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("message content")
                }
                fn visit_str<E: de::Error>(self, _: &str) -> Result<Self::Value, E> {
                    Ok(CitationScan::default())
                }
                fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                    Ok(CitationScan::default())
                }
                fn visit_bool<E: de::Error>(self, _: bool) -> Result<Self::Value, E> {
                    Ok(CitationScan::default())
                }
                fn visit_i64<E: de::Error>(self, _: i64) -> Result<Self::Value, E> {
                    Ok(CitationScan::default())
                }
                fn visit_u64<E: de::Error>(self, _: u64) -> Result<Self::Value, E> {
                    Ok(CitationScan::default())
                }
                fn visit_f64<E: de::Error>(self, _: f64) -> Result<Self::Value, E> {
                    Ok(CitationScan::default())
                }
                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    let mut scan = CitationScan::default();
                    while let Some(child) = seq.next_element::<CitationScan>()? {
                        scan.cited_blocks |= child.cited_blocks;
                        scan.cited_users |= child.cited_users;
                    }
                    Ok(scan)
                }
                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                    let mut role = None;
                    let mut kind = None;
                    let mut text = false;
                    let mut citations = false;
                    let mut content = CitationScan::default();
                    while let Some(key) = map.next_key::<String>()? {
                        match key.as_str() {
                            "role" | "type" => {
                                let raw = map.next_value::<&RawValue>()?;
                                let value = serde_json::from_str::<String>(raw.get()).ok();
                                if key == "role" {
                                    role = value;
                                } else {
                                    kind = value;
                                }
                            }
                            "text" => {
                                text = map.next_value::<&RawValue>()?.get().starts_with('"');
                            }
                            "citations" => {
                                let raw = map.next_value::<&RawValue>()?;
                                citations = raw
                                    .get()
                                    .strip_prefix('[')
                                    .is_some_and(|tail| tail.trim_start() != "]");
                            }
                            "content" => {
                                content = map.next_value()?;
                            }
                            _ => {
                                map.next_value::<de::IgnoredAny>()?;
                            }
                        }
                    }
                    // Tool results also accept an implicit text block with no type.
                    let is_text = kind.as_deref() == Some("text") || (kind.is_none() && text);
                    let is_container = matches!(
                        kind.as_deref(),
                        Some("search_result" | "tool_result" | "web_search_tool_result")
                    );
                    Ok(CitationScan {
                        cited_blocks: (is_text && citations)
                            || (is_container && content.cited_blocks),
                        cited_users: role.as_deref() == Some("user") && content.cited_blocks,
                    })
                }
            }
            deserializer.deserialize_any(ScanVisitor)
        }
    }

    // Leave malformed-message diagnostics to the original typed parser below.
    if serde_json::from_slice::<CitationScan>(messages).is_ok_and(|scan| scan.cited_users) {
        return Err("messages: non-empty citations on user text blocks are not allowed with anthropic-version 2026-10-21".into());
    }
    Ok(())
}

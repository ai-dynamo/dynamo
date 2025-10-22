// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request context extraction

use axum::http::HeaderMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteMode {
    Auto,
    Force,
    Shadow,
    Off,
}

#[derive(Debug, Clone)]
pub struct RequestCtx {
    pub route_mode: RouteMode,
    pub explicit_model: Option<String>,
    pub is_stream: bool,
    pub project_id: Option<String>,
    pub user_id: Option<String>,
}

impl RequestCtx {
    pub fn from_headers_and_model(headers: &HeaderMap, model_in_body: Option<&str>, is_stream: bool) -> Self {
        let route_mode = headers
            .get("x-dynamo-routing")
            .and_then(|v| v.to_str().ok())
            .map(|s| match s.to_ascii_lowercase().as_str() {
                "force" => RouteMode::Force,
                "shadow" => RouteMode::Shadow,
                "off" => RouteMode::Off,
                _ => RouteMode::Auto,
            })
            .unwrap_or(RouteMode::Auto);

        let project_id = headers.get("x-project-id").and_then(|v| v.to_str().ok()).map(|s| s.to_string());
        let user_id = headers.get("x-user-id").and_then(|v| v.to_str().ok()).map(|s| s.to_string());

        Self {
            route_mode,
            explicit_model: model_in_body.map(|s| s.to_string()),
            is_stream,
            project_id,
            user_id,
        }
    }
}

/// Best-effort text extraction from OpenAI-like request bodies.
/// - Chat: joins `messages[].content` (string or array parts with type/text)
/// - Legacy: `prompt` (string or array of strings)
/// - Responses API / embeddings: tries `input` if string
pub fn extract_text_for_classification(body: &serde_json::Value) -> String {
    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        let mut acc = String::new();
        for msg in messages {
            if let Some(content) = msg.get("content") {
                match content {
                    serde_json::Value::String(s) => {
                        acc.push_str(s);
                        acc.push('\n');
                    }
                    serde_json::Value::Array(parts) => {
                        for p in parts {
                            if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                                acc.push_str(t);
                                acc.push('\n');
                            } else if let Some(t) = p.get("content").and_then(|x| x.as_str()) {
                                acc.push_str(t);
                                acc.push('\n');
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if !acc.is_empty() { return acc; }
    }
    if let Some(prompt) = body.get("prompt") {
        if let Some(s) = prompt.as_str() { return s.to_string(); }
        if let Some(arr) = prompt.as_array() {
            return arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join("\n");
        }
    }
    if let Some(input) = body.get("input") {
        if let Some(s) = input.as_str() { return s.to_string(); }
    }
    String::new()
}


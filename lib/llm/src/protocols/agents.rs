// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coding-agent request metadata recognized at Dynamo's HTTP boundary.

use axum::http::HeaderMap;

pub(crate) const HEADER_CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";
pub(crate) const HEADER_CLAUDE_CODE_AGENT_ID: &str = "x-claude-code-agent-id";
pub(crate) const HEADER_CODEX_SESSION_ID: &str = "session-id";
pub(crate) const HEADER_OPENCODE_SESSION_ID: &str = "x-session-id";
pub(crate) const HEADER_DYNAMO_SESSION_ID: &str = "x-dynamo-session-id";
pub(crate) const HEADER_DYNAMO_TRAJECTORY_ID: &str = "x-dynamo-trajectory-id";
pub(crate) const HEADER_DYNAMO_PARENT_TRAJECTORY_ID: &str = "x-dynamo-parent-trajectory-id";
pub(crate) const HEADER_DYNAMO_TRAJECTORY_FINAL: &str = "x-dynamo-trajectory-final";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentContextHeaderValues {
    pub(crate) trajectory_id: String,
    pub(crate) parent_trajectory_id: Option<String>,
    pub(crate) trajectory_final: Option<bool>,
}

fn header_value(headers: &HeaderMap, header_name: &str) -> Option<String> {
    let value = headers.get(header_name)?.to_str().ok()?.trim();
    (!value.is_empty()).then(|| value.to_string())
}

pub(crate) fn agent_context_header_values(headers: &HeaderMap) -> Option<AgentContextHeaderValues> {
    let trajectory_id = header_value(headers, HEADER_DYNAMO_TRAJECTORY_ID)
        .or_else(|| header_value(headers, HEADER_CLAUDE_CODE_AGENT_ID))?;
    let trajectory_final = header_bool(headers, HEADER_DYNAMO_TRAJECTORY_FINAL);
    Some(AgentContextHeaderValues {
        trajectory_id,
        parent_trajectory_id: header_value(headers, HEADER_DYNAMO_PARENT_TRAJECTORY_ID),
        trajectory_final,
    })
}

pub(crate) fn session_affinity_header_value(headers: &HeaderMap) -> Option<String> {
    [
        HEADER_DYNAMO_SESSION_ID,
        HEADER_CLAUDE_CODE_SESSION_ID,
        HEADER_CODEX_SESSION_ID,
        HEADER_OPENCODE_SESSION_ID,
    ]
    .into_iter()
    .find_map(|header| header_value(headers, header))
}

fn header_bool(headers: &HeaderMap, header_name: &str) -> Option<bool> {
    let value = header_value(headers, header_name)?;
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

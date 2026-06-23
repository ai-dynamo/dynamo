// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coding-agent request metadata recognized at Dynamo's HTTP boundary.

use axum::http::HeaderMap;

pub(crate) const HEADER_CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";
pub(crate) const HEADER_CLAUDE_CODE_AGENT_ID: &str = "x-claude-code-agent-id";
pub(crate) const HEADER_CODEX_SESSION_ID: &str = "session-id";
pub(crate) const HEADER_OPENCODE_SESSION_ID: &str = "x-session-id";
pub(crate) const HEADER_OPENCODE_PARENT_SESSION_ID: &str = "x-parent-session-id";
pub(crate) const HEADER_DYNAMO_TRAJECTORY_ID: &str = "x-dynamo-trajectory-id";
pub(crate) const HEADER_DYNAMO_PARENT_TRAJECTORY_ID: &str = "x-dynamo-parent-trajectory-id";
pub(crate) const HEADER_DYNAMO_TRAJECTORY_FINAL: &str = "x-dynamo-trajectory-final";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentHeaderMapping {
    session_header: &'static str,
    trajectory_header: Option<&'static str>,
    parent_session_header: Option<&'static str>,
    infer_parent_from_session_for_child: bool,
}

const AGENT_HEADER_MAPPINGS: &[AgentHeaderMapping] = &[
    AgentHeaderMapping {
        session_header: HEADER_CLAUDE_CODE_SESSION_ID,
        trajectory_header: Some(HEADER_CLAUDE_CODE_AGENT_ID),
        parent_session_header: None,
        infer_parent_from_session_for_child: true,
    },
    AgentHeaderMapping {
        session_header: HEADER_CODEX_SESSION_ID,
        trajectory_header: None,
        parent_session_header: None,
        infer_parent_from_session_for_child: false,
    },
    AgentHeaderMapping {
        session_header: HEADER_OPENCODE_SESSION_ID,
        trajectory_header: None,
        parent_session_header: Some(HEADER_OPENCODE_PARENT_SESSION_ID),
        infer_parent_from_session_for_child: false,
    },
];

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
    let trajectory_final = header_bool(headers, HEADER_DYNAMO_TRAJECTORY_FINAL);

    for mapping in AGENT_HEADER_MAPPINGS {
        let Some(session_id) = header_value(headers, mapping.session_header) else {
            continue;
        };
        let trajectory_id = mapping
            .trajectory_header
            .and_then(|trajectory_header| header_value(headers, trajectory_header))
            .unwrap_or_else(|| session_id.clone());
        let parent_trajectory_id = mapping
            .parent_session_header
            .and_then(|parent_header| header_value(headers, parent_header))
            .or_else(|| {
                (mapping.infer_parent_from_session_for_child && trajectory_id != session_id)
                    .then(|| session_id.clone())
            });

        return Some(AgentContextHeaderValues {
            trajectory_id,
            parent_trajectory_id,
            trajectory_final,
        });
    }
    header_value(headers, HEADER_DYNAMO_TRAJECTORY_ID).map(|trajectory_id| {
        AgentContextHeaderValues {
            trajectory_id,
            parent_trajectory_id: header_value(headers, HEADER_DYNAMO_PARENT_TRAJECTORY_ID),
            trajectory_final,
        }
    })
}

fn header_bool(headers: &HeaderMap, header_name: &str) -> Option<bool> {
    let value = header_value(headers, header_name)?;
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

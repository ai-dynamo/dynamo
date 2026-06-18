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
pub(crate) const DEFAULT_DYNAMO_SESSION_TYPE_ID: &str = "dynamo";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentHeaderMapping {
    session_type_id: &'static str,
    session_header: &'static str,
    trajectory_header: Option<&'static str>,
    parent_session_header: Option<&'static str>,
    infer_parent_from_session_for_child: bool,
}

const AGENT_HEADER_MAPPINGS: &[AgentHeaderMapping] = &[
    AgentHeaderMapping {
        session_type_id: "claude_code",
        session_header: HEADER_CLAUDE_CODE_SESSION_ID,
        trajectory_header: Some(HEADER_CLAUDE_CODE_AGENT_ID),
        parent_session_header: None,
        infer_parent_from_session_for_child: true,
    },
    AgentHeaderMapping {
        session_type_id: "codex",
        session_header: HEADER_CODEX_SESSION_ID,
        trajectory_header: None,
        parent_session_header: None,
        infer_parent_from_session_for_child: false,
    },
    AgentHeaderMapping {
        session_type_id: "opencode",
        session_header: HEADER_OPENCODE_SESSION_ID,
        trajectory_header: None,
        parent_session_header: Some(HEADER_OPENCODE_PARENT_SESSION_ID),
        infer_parent_from_session_for_child: false,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentContextHeaderValues {
    pub(crate) session_type_id: &'static str,
    pub(crate) session_id: Option<String>,
    pub(crate) trajectory_id: String,
    pub(crate) parent_trajectory_id: Option<String>,
}

fn header_value(headers: &HeaderMap, header_name: &str) -> Option<String> {
    let value = headers.get(header_name)?.to_str().ok()?.trim();
    (!value.is_empty()).then(|| value.to_string())
}

pub(crate) fn agent_context_header_values(headers: &HeaderMap) -> Option<AgentContextHeaderValues> {
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
            session_type_id: mapping.session_type_id,
            session_id: Some(session_id),
            trajectory_id,
            parent_trajectory_id,
        });
    }
    header_value(headers, HEADER_DYNAMO_TRAJECTORY_ID).map(|trajectory_id| {
        AgentContextHeaderValues {
            session_type_id: DEFAULT_DYNAMO_SESSION_TYPE_ID,
            session_id: None,
            trajectory_id,
            parent_trajectory_id: None,
        }
    })
}

pub(crate) fn has_agent_headers(headers: &HeaderMap) -> bool {
    headers.contains_key(HEADER_DYNAMO_TRAJECTORY_ID)
        || AGENT_HEADER_MAPPINGS.iter().any(|mapping| {
            headers.contains_key(mapping.session_header)
                || mapping
                    .trajectory_header
                    .is_some_and(|trajectory_header| headers.contains_key(trajectory_header))
                || mapping
                    .parent_session_header
                    .is_some_and(|parent_header| headers.contains_key(parent_header))
        })
}

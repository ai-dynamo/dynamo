// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coding-agent request metadata recognized at Dynamo's HTTP boundary.

use axum::http::HeaderMap;

pub(crate) const HEADER_CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";
pub(crate) const HEADER_CODEX_SESSION_ID: &str = "session-id";
pub(crate) const HEADER_OPENCODE_SESSION_ID: &str = "x-session-id";
pub(crate) const HEADER_OPENCODE_PARENT_SESSION_ID: &str = "x-parent-session-id";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentHeaderMapping {
    session_type_id: &'static str,
    session_header: &'static str,
    parent_session_header: Option<&'static str>,
}

const AGENT_HEADER_MAPPINGS: &[AgentHeaderMapping] = &[
    AgentHeaderMapping {
        session_type_id: "claude_code",
        session_header: HEADER_CLAUDE_CODE_SESSION_ID,
        parent_session_header: None,
    },
    AgentHeaderMapping {
        session_type_id: "codex",
        session_header: HEADER_CODEX_SESSION_ID,
        parent_session_header: None,
    },
    AgentHeaderMapping {
        session_type_id: "opencode",
        session_header: HEADER_OPENCODE_SESSION_ID,
        parent_session_header: Some(HEADER_OPENCODE_PARENT_SESSION_ID),
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentContextHeaderValues {
    pub(crate) session_type_id: &'static str,
    pub(crate) session_id: String,
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
        let parent_trajectory_id = mapping
            .parent_session_header
            .and_then(|parent_header| header_value(headers, parent_header));

        return Some(AgentContextHeaderValues {
            session_type_id: mapping.session_type_id,
            session_id,
            parent_trajectory_id,
        });
    }
    None
}

pub(crate) fn has_agent_headers(headers: &HeaderMap) -> bool {
    AGENT_HEADER_MAPPINGS.iter().any(|mapping| {
        headers.contains_key(mapping.session_header)
            || mapping
                .parent_session_header
                .is_some_and(|parent_header| headers.contains_key(parent_header))
    })
}

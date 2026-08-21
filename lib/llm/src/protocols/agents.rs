// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coding-agent request metadata recognized at Dynamo's HTTP boundary.

use axum::http::HeaderMap;
use serde::Deserialize;

use crate::protocols::common::extensions::AgentCompaction;

pub(crate) const HEADER_CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";
pub(crate) const HEADER_CLAUDE_CODE_AGENT_ID: &str = "x-claude-code-agent-id";
pub(crate) const HEADER_CLAUDE_CODE_PARENT_AGENT_ID: &str = "x-claude-code-parent-agent-id";
pub(crate) const HEADER_CODEX_THREAD_ID: &str = "thread-id";
pub(crate) const HEADER_CODEX_PARENT_THREAD_ID: &str = "x-codex-parent-thread-id";
pub(crate) const HEADER_CODEX_TURN_METADATA: &str = "x-codex-turn-metadata";
pub(crate) const HEADER_DEEPSEEK_HARNESS_SESSION_ID: &str = "x-deepseek-harness-session-id";
pub(crate) const HEADER_DEEPSEEK_HARNESS_COMPACT: &str = "x-deepseek-harness-compact";
pub(crate) const HEADER_OPENCODE_SESSION_ID: &str = "x-session-id";
pub(crate) const HEADER_OPENCODE_PARENT_SESSION_ID: &str = "x-parent-session-id";
pub(crate) const HEADER_DYNAMO_SESSION_ID: &str = "x-dynamo-session-id";
pub(crate) const HEADER_DYNAMO_PARENT_SESSION_ID: &str = "x-dynamo-parent-session-id";
pub(crate) const HEADER_DYNAMO_SESSION_FINAL: &str = "x-dynamo-session-final";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentHeaderMapping {
    root_session_header: &'static str,
    child_session_header: Option<&'static str>,
    parent_session_header: Option<&'static str>,
    infer_parent_from_session_for_child: bool,
}

const AGENT_HEADER_MAPPINGS: &[AgentHeaderMapping] = &[
    AgentHeaderMapping {
        root_session_header: HEADER_CLAUDE_CODE_SESSION_ID,
        child_session_header: Some(HEADER_CLAUDE_CODE_AGENT_ID),
        parent_session_header: Some(HEADER_CLAUDE_CODE_PARENT_AGENT_ID),
        infer_parent_from_session_for_child: true,
    },
    AgentHeaderMapping {
        root_session_header: HEADER_CODEX_THREAD_ID,
        child_session_header: None,
        parent_session_header: Some(HEADER_CODEX_PARENT_THREAD_ID),
        infer_parent_from_session_for_child: false,
    },
    AgentHeaderMapping {
        root_session_header: HEADER_DEEPSEEK_HARNESS_SESSION_ID,
        child_session_header: None,
        parent_session_header: None,
        infer_parent_from_session_for_child: false,
    },
    AgentHeaderMapping {
        root_session_header: HEADER_OPENCODE_SESSION_ID,
        child_session_header: None,
        parent_session_header: Some(HEADER_OPENCODE_PARENT_SESSION_ID),
        infer_parent_from_session_for_child: false,
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentContextHeaderValues {
    pub(crate) session_id: String,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) session_final: Option<bool>,
    pub(crate) compaction: Option<AgentCompaction>,
}

#[derive(Deserialize)]
struct CodexTurnMetadata {
    request_kind: Option<String>,
    compaction: Option<AgentCompaction>,
}

fn borrowed_header_value<'a>(headers: &'a HeaderMap, header_name: &str) -> Option<&'a str> {
    let value = headers.get(header_name)?.to_str().ok()?.trim();
    (!value.is_empty()).then_some(value)
}

pub(crate) fn agent_context_header_values(headers: &HeaderMap) -> Option<AgentContextHeaderValues> {
    let session_final = header_bool(headers, HEADER_DYNAMO_SESSION_FINAL);

    if let Some(session_id) = borrowed_header_value(headers, HEADER_DYNAMO_SESSION_ID) {
        return Some(AgentContextHeaderValues {
            parent_session_id: borrowed_header_value(headers, HEADER_DYNAMO_PARENT_SESSION_ID)
                .filter(|parent_session_id| *parent_session_id != session_id)
                .map(str::to_owned),
            session_id: session_id.to_owned(),
            session_final,
            compaction: borrowed_header_value(headers, HEADER_CODEX_THREAD_ID)
                .and_then(|_| codex_compaction_header_value(headers)),
        });
    }

    for mapping in AGENT_HEADER_MAPPINGS {
        let Some(root_session_id) = borrowed_header_value(headers, mapping.root_session_header)
        else {
            continue;
        };
        let session_id = mapping
            .child_session_header
            .and_then(|child_session_header| borrowed_header_value(headers, child_session_header))
            .unwrap_or(root_session_id);
        let parent_session_id = mapping
            .parent_session_header
            .and_then(|parent_header| borrowed_header_value(headers, parent_header))
            .filter(|parent_session_id| *parent_session_id != session_id)
            .filter(|_| {
                !mapping.infer_parent_from_session_for_child || session_id != root_session_id
            })
            .or_else(|| {
                (mapping.infer_parent_from_session_for_child && session_id != root_session_id)
                    .then_some(root_session_id)
            })
            .map(str::to_owned);
        return Some(AgentContextHeaderValues {
            session_id: session_id.to_owned(),
            parent_session_id,
            session_final,
            compaction: match mapping.root_session_header {
                HEADER_CODEX_THREAD_ID => codex_compaction_header_value(headers),
                HEADER_DEEPSEEK_HARNESS_SESSION_ID => deepseek_compaction_header_value(headers),
                _ => None,
            },
        });
    }
    None
}

fn codex_compaction_header_value(headers: &HeaderMap) -> Option<AgentCompaction> {
    let raw = borrowed_header_value(headers, HEADER_CODEX_TURN_METADATA)?;
    let metadata: CodexTurnMetadata = serde_json::from_str(raw).ok()?;
    (metadata.request_kind.as_deref() == Some("compaction"))
        .then(|| metadata.compaction.unwrap_or_default())
}

fn deepseek_compaction_header_value(headers: &HeaderMap) -> Option<AgentCompaction> {
    borrowed_header_value(headers, HEADER_DEEPSEEK_HARNESS_SESSION_ID)?;
    (borrowed_header_value(headers, HEADER_DEEPSEEK_HARNESS_COMPACT) == Some("1"))
        .then(AgentCompaction::default)
}

pub(crate) fn session_affinity_header_value(headers: &HeaderMap) -> Option<String> {
    if let Some(session_id) = borrowed_header_value(headers, HEADER_DYNAMO_SESSION_ID) {
        return Some(session_id.to_owned());
    }
    for mapping in AGENT_HEADER_MAPPINGS {
        let Some(root_session_id) = borrowed_header_value(headers, mapping.root_session_header)
        else {
            continue;
        };
        let session_id = mapping
            .child_session_header
            .and_then(|child_session_header| borrowed_header_value(headers, child_session_header))
            .unwrap_or(root_session_id);
        return Some(session_id.to_owned());
    }
    None
}

fn header_bool(headers: &HeaderMap, header_name: &str) -> Option<bool> {
    let value = borrowed_header_value(headers, header_name)?;
    dynamo_runtime::config::parse_bool_opt(value)
}

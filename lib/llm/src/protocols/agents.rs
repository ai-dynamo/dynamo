// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coding-agent request metadata recognized at Dynamo's HTTP boundary.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::http::HeaderMap;

pub(crate) const HEADER_CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";
pub(crate) const HEADER_CLAUDE_CODE_AGENT_ID: &str = "x-claude-code-agent-id";
pub(crate) const HEADER_CLAUDE_CODE_PARENT_AGENT_ID: &str = "x-claude-code-parent-agent-id";
pub(crate) const HEADER_CODEX_THREAD_ID: &str = "thread-id";
pub(crate) const HEADER_CODEX_PARENT_THREAD_ID: &str = "x-codex-parent-thread-id";
pub(crate) const HEADER_OPENCODE_SESSION_ID: &str = "x-session-id";
pub(crate) const HEADER_OPENCODE_PARENT_SESSION_ID: &str = "x-parent-session-id";
pub const HEADER_DYNAMO_SESSION_ID: &str = "x-dynamo-session-id";
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
    pub(crate) agent_headers: Arc<BTreeMap<String, Vec<String>>>,
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
            agent_headers: capture_agent_headers(headers),
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
            agent_headers: capture_agent_headers(headers),
        });
    }
    None
}

const MAX_AGENT_HEADER_VALUES: usize = 64;
const MAX_AGENT_HEADER_VALUE_BYTES: usize = 16 * 1024;
const MAX_AGENT_HEADER_BYTES: usize = 32 * 1024;

fn is_agent_header(name: &str) -> bool {
    name.starts_with("x-claude-code-")
        || name.starts_with("x-codex-")
        || matches!(
            name,
            "session-id"
                | HEADER_CODEX_THREAD_ID
                | "x-openai-subagent"
                | "x-openai-memgen-request"
                | HEADER_OPENCODE_SESSION_ID
                | HEADER_OPENCODE_PARENT_SESSION_ID
        )
}

fn capture_agent_headers(headers: &HeaderMap) -> Arc<BTreeMap<String, Vec<String>>> {
    let mut captured = BTreeMap::<String, Vec<String>>::new();
    let mut count = 0;
    let mut bytes = 0;
    for (name, value) in headers {
        if !is_agent_header(name.as_str()) || value.is_sensitive() {
            continue;
        }
        if name == "session-id" && borrowed_header_value(headers, HEADER_CODEX_THREAD_ID).is_none()
        {
            continue;
        }
        let Ok(value) = value.to_str() else {
            continue;
        };
        let size = name.as_str().len() + value.len();
        if value.len() > MAX_AGENT_HEADER_VALUE_BYTES
            || count >= MAX_AGENT_HEADER_VALUES
            || bytes + size > MAX_AGENT_HEADER_BYTES
        {
            continue;
        }
        captured
            .entry(name.as_str().to_owned())
            .or_default()
            .push(value.to_owned());
        count += 1;
        bytes += size;
    }
    Arc::new(captured)
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderName, HeaderValue};

    #[test]
    fn captures_open_families_and_repeated_values_verbatim() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_CLAUDE_CODE_SESSION_ID, "root".parse().unwrap());
        headers.append(
            "X-Claude-Code-Future".parse::<HeaderName>().unwrap(),
            "  unknown  ".parse().unwrap(),
        );
        headers.append("x-claude-code-future", "second,third".parse().unwrap());
        headers.append("x-claude-code-future", HeaderValue::from_static(""));
        headers.insert("x-codex-future", "{invalid json".parse().unwrap());
        headers.insert(
            "x-claude-code-prev-tool-durations",
            "Bash=742;Bash=9;custom%3Btool=2".parse().unwrap(),
        );
        headers.insert(
            "x-claude-code-compaction",
            "future-trigger".parse().unwrap(),
        );
        headers.insert(
            "x-claude-code-context-compacted",
            "reactive".parse().unwrap(),
        );
        headers.insert("x-claude-code-prompt-id", "prompt-1".parse().unwrap());
        let captured = agent_context_header_values(&headers).unwrap();
        assert_eq!(
            captured.agent_headers["x-claude-code-future"],
            ["  unknown  ", "second,third", ""]
        );
        for (name, value) in &headers {
            assert!(
                captured.agent_headers[name.as_str()].contains(&value.to_str().unwrap().to_owned())
            );
        }
    }

    #[test]
    fn capture_excludes_unrelated_sensitive_and_non_text_values() {
        let mut headers = HeaderMap::new();
        for name in [
            "authorization",
            "cookie",
            "x-api-key",
            "anthropic-beta",
            "x-unrelated",
        ] {
            headers.insert(name, "private".parse().unwrap());
        }
        let mut sensitive = HeaderValue::from_static("private");
        sensitive.set_sensitive(true);
        headers.insert("x-codex-sensitive", sensitive);
        headers.insert("x-codex-binary", HeaderValue::from_bytes(&[0xff]).unwrap());
        assert!(capture_agent_headers(&headers).is_empty());

        for name in [
            "session-id",
            "thread-id",
            "x-openai-subagent",
            "x-openai-memgen-request",
            "x-session-id",
            "x-parent-session-id",
        ] {
            headers.insert(name, "value".parse().unwrap());
        }
        assert_eq!(capture_agent_headers(&headers).len(), 6);
    }

    #[test]
    fn capture_session_id_requires_usable_thread_id() {
        for thread_id in [
            None,
            Some(HeaderValue::from_static("")),
            Some(HeaderValue::from_static(" \t ")),
            Some(HeaderValue::from_bytes(&[0xff]).unwrap()),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(HEADER_CLAUDE_CODE_SESSION_ID, "root".parse().unwrap());
            headers.insert("session-id", "unrelated".parse().unwrap());
            headers.insert(HEADER_OPENCODE_PARENT_SESSION_ID, "parent".parse().unwrap());
            if let Some(thread_id) = thread_id {
                headers.insert(HEADER_CODEX_THREAD_ID, thread_id);
            }
            let context = agent_context_header_values(&headers).unwrap();
            assert_eq!(context.session_id, "root");
            assert!(!context.agent_headers.contains_key("session-id"));
            assert_eq!(
                context.agent_headers[HEADER_OPENCODE_PARENT_SESSION_ID],
                ["parent"]
            );
        }
    }

    #[test]
    fn capture_session_id_preserves_values_with_identity_override() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_CODEX_THREAD_ID, " thread ".parse().unwrap());
        for value in [" cache-session ", "", "other-session"] {
            headers.append("session-id", HeaderValue::from_static(value));
        }
        let context = agent_context_header_values(&headers).unwrap();
        assert_eq!(context.session_id, "thread");
        assert_eq!(
            context.agent_headers["session-id"],
            [" cache-session ", "", "other-session"]
        );

        headers.insert(HEADER_DYNAMO_SESSION_ID, "override".parse().unwrap());
        let overridden = agent_context_header_values(&headers).unwrap();
        assert_eq!(overridden.session_id, "override");
        assert_eq!(overridden.agent_headers, context.agent_headers);
    }

    #[test]
    fn capture_limits_omit_whole_values_without_changing_identity() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_CODEX_THREAD_ID, "thread".parse().unwrap());
        headers.insert(
            "x-codex-huge",
            "x".repeat(MAX_AGENT_HEADER_VALUE_BYTES + 1)
                .parse()
                .unwrap(),
        );
        let context = agent_context_header_values(&headers).unwrap();
        assert_eq!(context.session_id, "thread");
        assert!(!context.agent_headers.contains_key("x-codex-huge"));

        let mut repeated = HeaderMap::new();
        for _ in 0..MAX_AGENT_HEADER_VALUES + 1 {
            repeated.append("x-codex-repeated", HeaderValue::from_static("value"));
        }
        assert_eq!(
            capture_agent_headers(&repeated)["x-codex-repeated"].len(),
            MAX_AGENT_HEADER_VALUES
        );

        let mut large = HeaderMap::new();
        for _ in 0..3 {
            large.append(
                "x-codex-large",
                "x".repeat(MAX_AGENT_HEADER_VALUE_BYTES).parse().unwrap(),
            );
        }
        let captured = capture_agent_headers(&large);
        assert_eq!(captured["x-codex-large"].len(), 1);
        assert_eq!(
            captured["x-codex-large"][0].len(),
            MAX_AGENT_HEADER_VALUE_BYTES
        );
    }
}

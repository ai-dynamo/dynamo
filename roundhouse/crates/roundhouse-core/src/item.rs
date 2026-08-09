// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical conversation items.
//!
//! This is the provider-neutral form the session stores. Both the local Dynamo
//! path and the frontier path render *from* this; neither renders *to* it. That
//! direction matters: it is what lets one session be served by an OSS model on
//! one turn and a frontier model on the next without the history changing shape.

use serde::{Deserialize, Serialize};

use crate::ids::ResponseId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// The payload of an item.
///
/// Deliberately small for the walking skeleton: text plus the two tool shapes
/// an agentic loop cannot do without. Images and audio slot in as further
/// variants without disturbing the session or routing layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ItemContent {
    Text {
        text: String,
    },
    ToolCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        call_id: String,
        output: String,
    },
}

impl ItemContent {
    /// The text used for prompt rendering and token accounting.
    ///
    /// Tool calls and results are flattened to a stable textual form so the
    /// token buffer stays append-only. The exact rendering is a placeholder
    /// pending per-model chat templates; what matters here is determinism,
    /// because a rendering that varies between turns would invalidate every
    /// cached block after the first divergence.
    pub fn render(&self) -> String {
        match self {
            ItemContent::Text { text } => text.clone(),
            ItemContent::ToolCall {
                call_id,
                name,
                arguments,
            } => format!("<tool_call id=\"{call_id}\" name=\"{name}\">{arguments}</tool_call>"),
            ItemContent::ToolResult { call_id, output } => {
                format!("<tool_result id=\"{call_id}\">{output}</tool_result>")
            }
        }
    }
}

/// One entry in the canonical conversation log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub role: Role,
    pub content: ItemContent,
    /// Set on assistant items so `previous_response_id` can resolve to the
    /// exact prefix a client is continuing from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<ResponseId>,
}

impl Item {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: ItemContent::Text { text: text.into() },
            response_id: None,
        }
    }

    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: ItemContent::Text { text: text.into() },
            response_id: None,
        }
    }

    pub fn assistant_text(text: impl Into<String>, response_id: ResponseId) -> Self {
        Self {
            role: Role::Assistant,
            content: ItemContent::Text { text: text.into() },
            response_id: Some(response_id),
        }
    }

    /// Deterministic prompt rendering for a single item.
    pub fn render(&self) -> String {
        format!("<|{}|>{}", self.role.as_str(), self.content.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendering_is_stable_across_calls() {
        let item = Item::user_text("hello");
        assert_eq!(item.render(), item.render());
        assert_eq!(item.render(), "<|user|>hello");
    }

    #[test]
    fn tool_items_render_deterministically() {
        let call = Item {
            role: Role::Assistant,
            content: ItemContent::ToolCall {
                call_id: "c1".into(),
                name: "grep".into(),
                arguments: "{\"q\":\"x\"}".into(),
            },
            response_id: None,
        };
        assert_eq!(call.render(), call.render());
        assert!(call.render().contains("name=\"grep\""));
    }
}

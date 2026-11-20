// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request session metadata supplied by hosts and read by the stages' views.

use std::collections::BTreeMap;
use std::sync::Arc;

/// The event that caused an agent request to enter worker selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerSelectionInputTrigger {
    /// A user message started the turn.
    UserMessage,
    /// A tool result continued the turn.
    ToolResult,
    /// Another event caused the request.
    Other,
}

/// Session metadata supplied to a custom worker-selection policy.
///
/// The internal request protocol supplies these values. Optional values remain
/// absent when the request does not include them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionContext {
    session_id: String,
    parent_session_id: Option<String>,
    session_final: Option<bool>,
    input_trigger: Option<WorkerSelectionInputTrigger>,
    agent_headers: Option<Arc<BTreeMap<String, Vec<String>>>>,
}

impl SessionContext {
    /// Create the session metadata available to worker selection.
    pub fn new(
        session_id: String,
        parent_session_id: Option<String>,
        session_final: Option<bool>,
        input_trigger: Option<WorkerSelectionInputTrigger>,
    ) -> Self {
        Self {
            session_id,
            parent_session_id,
            session_final,
            input_trigger,
            agent_headers: None,
        }
    }

    /// Return the stable reasoning or tool-session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the parent session identifier for a subagent request.
    pub fn parent_session_id(&self) -> Option<&str> {
        self.parent_session_id.as_deref()
    }

    /// Return the optional terminal marker for this session.
    ///
    /// `Some(true)` marks the session as final, `Some(false)` explicitly marks
    /// it as continuing, and `None` means the caller supplied no marker.
    pub fn session_final(&self) -> Option<bool> {
        self.session_final
    }

    /// Return the event that caused this request, when supplied.
    pub fn input_trigger(&self) -> Option<WorkerSelectionInputTrigger> {
        self.input_trigger
    }

    /// Attach request-scoped opaque headers captured by the protocol ingress.
    /// The map is shared with the request envelope without copying its values.
    pub fn with_agent_headers(mut self, headers: Arc<BTreeMap<String, Vec<String>>>) -> Self {
        self.agent_headers = Some(headers);
        self
    }

    /// Return raw coding-agent observations for this request, without normalization.
    ///
    /// HTTP ingress lowercases names and preserves repeated text values in order.
    /// It omits sensitive/non-text values and values exceeding its capture limits
    /// (64 values, 16 KiB/value, 32 KiB including names). An empty map or missing
    /// key means no observation was captured, not that a lifecycle event did not
    /// occur. Values are untrusted; plugins own parsing and harness semantics.
    /// This request-level input needs no worker signal group and is never weighted
    /// or interpreted by the default policy.
    pub fn agent_headers(&self) -> &BTreeMap<String, Vec<String>> {
        static EMPTY: BTreeMap<String, Vec<String>> = BTreeMap::new();
        self.agent_headers.as_deref().unwrap_or(&EMPTY)
    }
}

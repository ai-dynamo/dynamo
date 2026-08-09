// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The session event log.
//!
//! Every observable thing a session does becomes an event with a monotonic
//! `seq`. A client that reconnects presents the last `seq` it saw and the
//! server replays forward from there, which is the same mechanism the Responses
//! API exposes as `starting_after`. Because the log is the source of truth
//! rather than a side-channel, a partially generated response survives the
//! death of the process that was generating it.

use serde::{Deserialize, Serialize};

use crate::ids::{ResponseId, SessionId, TurnId};
use crate::item::Item;
use crate::routing::DecisionRecord;

/// Token accounting for one completed model call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    /// Portion of `input_tokens` served from a prefix cache.
    ///
    /// Locally this is derived from the scheduler's cache credit; for frontier
    /// providers it is whatever the provider reports. It is the number the
    /// whole design exists to maximize.
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Why a response stopped short.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteReason {
    /// The owning process lost its lease or died mid-generation.
    ///
    /// The partial output is already durable in the log, so a successor can
    /// resume from it rather than restarting the turn.
    OwnerLost,
    MaxOutputTokens,
    ClientCancelled,
    UpstreamError,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEventKind {
    SessionCreated {
        model_policy: String,
    },
    /// A turn was admitted. Carries the client's idempotency key.
    TurnStarted {
        turn_id: TurnId,
        response_id: ResponseId,
    },
    /// An item was committed to the canonical conversation.
    ItemAppended {
        item: Item,
    },
    /// The routing layer chose a target. Emitted before any bytes are produced
    /// so the audit trail records the decision even if execution then fails.
    Routed {
        response_id: ResponseId,
        decision: DecisionRecord,
    },
    OutputTextDelta {
        response_id: ResponseId,
        text: String,
    },
    ResponseCompleted {
        response_id: ResponseId,
        usage: Usage,
    },
    ResponseIncomplete {
        response_id: ResponseId,
        reason: IncompleteReason,
        usage: Usage,
    },
    /// A turn was re-sent after reconnect and served from the existing result.
    TurnDeduplicated {
        turn_id: TurnId,
        response_id: ResponseId,
    },
    Error {
        message: String,
    },
}

/// A sealed log entry. `seq` is assigned by the store on append and is
/// contiguous and strictly increasing within a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEvent {
    pub seq: u64,
    pub session_id: SessionId,
    pub at_ms: u64,
    #[serde(flatten)]
    pub kind: SessionEventKind,
}

impl SessionEvent {
    /// The response this event belongs to, if any.
    ///
    /// Used to project the session-wide log down to the per-response view the
    /// Responses API exposes.
    pub fn response_id(&self) -> Option<&ResponseId> {
        match &self.kind {
            SessionEventKind::TurnStarted { response_id, .. }
            | SessionEventKind::Routed { response_id, .. }
            | SessionEventKind::OutputTextDelta { response_id, .. }
            | SessionEventKind::ResponseCompleted { response_id, .. }
            | SessionEventKind::ResponseIncomplete { response_id, .. }
            | SessionEventKind::TurnDeduplicated { response_id, .. } => Some(response_id),
            SessionEventKind::SessionCreated { .. }
            | SessionEventKind::ItemAppended { .. }
            | SessionEventKind::Error { .. } => None,
        }
    }

    /// Whether this event ends its response.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.kind,
            SessionEventKind::ResponseCompleted { .. }
                | SessionEventKind::ResponseIncomplete { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_events_are_recognized() {
        let done = SessionEvent {
            seq: 4,
            session_id: SessionId::new("s"),
            at_ms: 0,
            kind: SessionEventKind::ResponseCompleted {
                response_id: ResponseId::new("r"),
                usage: Usage::default(),
            },
        };
        assert!(done.is_terminal());

        let delta = SessionEvent {
            seq: 3,
            session_id: SessionId::new("s"),
            at_ms: 0,
            kind: SessionEventKind::OutputTextDelta {
                response_id: ResponseId::new("r"),
                text: "hi".into(),
            },
        };
        assert!(!delta.is_terminal());
        assert_eq!(delta.response_id(), Some(&ResponseId::new("r")));
    }
}

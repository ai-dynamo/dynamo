// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Per-deployment selector for the reasoning field
// name emitted on chat completion responses.
//
// Reads the DYN_REASONING_FIELD_NAME environment variable at first access
// and caches the result. Values:
//   - "reasoning"          → emit `message.reasoning` only, no `reasoning_content`
//   - "reasoning_content"  → emit `message.reasoning_content` only, no `reasoning`
//                           (this is the default, preserving upstream behavior)
//
// Called by the response envelope constructor in
// `protocols/openai/chat_completions/aggregator.rs` and `delta.rs`, and by
// the streaming delta rebuilder in `audit/stream.rs`. The struct fields for
// both names exist on `ChatCompletionResponseMessage` and
// `ChatCompletionStreamResponseDelta` (see lib/protocols/src/types/chat.rs);
// this helper decides which one gets populated at build time.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningField {
    /// Emit `message.reasoning` on responses.
    Reasoning,
    /// Emit `message.reasoning_content` on responses (upstream default).
    ReasoningContent,
}

pub fn reasoning_field_selection() -> ReasoningField {
    static FIELD: OnceLock<ReasoningField> = OnceLock::new();
    *FIELD.get_or_init(|| {
        match std::env::var("DYN_REASONING_FIELD_NAME").as_deref() {
            Ok("reasoning") => ReasoningField::Reasoning,
            Ok("reasoning_content") | Err(_) => ReasoningField::ReasoningContent,
            Ok(other) => {
                tracing::warn!(
                    "DYN_REASONING_FIELD_NAME={other:?} is not \"reasoning\" or \"reasoning_content\"; defaulting to reasoning_content"
                );
                ReasoningField::ReasoningContent
            }
        }
    })
}

/// The pair of wire fields that populate `ChatCompletionResponseMessage`
/// and `ChatCompletionStreamResponseDelta`. Exactly one is `Some(source)`
/// and the other is `None` — [`route_reasoning`] picks which per the
/// `DYN_REASONING_FIELD_NAME` env var. Named struct (rather than a raw
/// tuple) so call-site destructures document what each field carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedReasoning {
    /// Populated only when `DYN_REASONING_FIELD_NAME=reasoning`.
    pub reasoning: Option<String>,
    /// Populated only when `DYN_REASONING_FIELD_NAME=reasoning_content` (default).
    pub reasoning_content: Option<String>,
}

/// Route `source` to the wire field chosen by `DYN_REASONING_FIELD_NAME`.
/// Return shape lets emit sites use struct-literal shorthand:
///
///     let RoutedReasoning { reasoning, reasoning_content } = route_reasoning(source);
///     ChatCompletionResponseMessage { reasoning, reasoning_content, ..other_fields }
#[inline]
pub fn route_reasoning(source: Option<String>) -> RoutedReasoning {
    match reasoning_field_selection() {
        ReasoningField::Reasoning => RoutedReasoning {
            reasoning: source,
            reasoning_content: None,
        },
        ReasoningField::ReasoningContent => RoutedReasoning {
            reasoning: None,
            reasoning_content: source,
        },
    }
}

/// Merge any reasoning already present on a streaming delta with `new_content`
/// and route the combined value to the wire field chosen by
/// `DYN_REASONING_FIELD_NAME`. Idempotent — callable at every live-streaming
/// reasoning-write site (reasoning-parser output, jail late-release, etc.) so
/// the outbound SSE emits a single field regardless of which internal stage
/// last touched the delta.
///
/// Pass `new_content = None` to just re-route whatever is already on the
/// delta (useful as a final normalize before serialization).
pub fn merge_and_route_stream_delta_reasoning(
    delta: &mut dynamo_protocols::types::ChatCompletionStreamResponseDelta,
    new_content: Option<String>,
) {
    let existing = match (delta.reasoning_content.take(), delta.reasoning.take()) {
        (Some(mut a), Some(b)) => {
            a.push_str(&b);
            Some(a)
        }
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    let combined = match (existing, new_content) {
        (Some(mut a), Some(b)) => {
            a.push_str(&b);
            Some(a)
        }
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    let RoutedReasoning {
        reasoning,
        reasoning_content,
    } = route_reasoning(combined);
    delta.reasoning = reasoning;
    delta.reasoning_content = reasoning_content;
}

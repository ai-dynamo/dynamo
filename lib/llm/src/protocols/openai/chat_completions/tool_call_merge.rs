// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared merge policy for streamed tool-call chunks.
//!
//! Both the non-streaming aggregator
//! ([`super::aggregator`]) and the streaming `tool_call_dispatch`
//! side channel (`crate::http::service::openai`) have to reassemble a single
//! logical tool call out of the per-index delta chunks a producer emits. They
//! must agree on that policy exactly, otherwise the typed dispatch event and
//! the aggregated response could disagree about the same tool call.

use dynamo_protocols::types::ChatCompletionMessageToolCallChunk;

/// Identity field on which two chunks for the same `index` disagreed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCallIdentityField {
    Id,
    Type,
    Name,
}

impl ToolCallIdentityField {
    /// Field name for the streaming `tool_call_dispatch` fail-closed log line.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::Type => "type",
            Self::Name => "function.name",
        }
    }
}

/// Result of merging one incoming chunk into an accumulator.
///
/// The merge itself is infallible: the first *non-empty* value of each identity
/// field wins and argument fragments are always concatenated. The outcome is
/// *reporting only* — a caller that ignores it still gets the merged chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCallMergeOutcome {
    /// The incoming chunk was consistent with what had already accumulated.
    Merged,
    /// The incoming chunk carried a *different* non-empty value for an
    /// identity field that had already been established. First-wins was
    /// applied and the incoming value discarded; the field is named so the
    /// caller can log or fail-closed as it sees fit.
    IdentityConflict { field: ToolCallIdentityField },
}

/// Merge an incoming chunk into the per-index accumulator.
///
/// #8640: the prior implementation required `id`, `name`, and `arguments`
/// all on the same chunk, and thus the argument-fragment deltas were dropped
/// and the client saw `arguments: ""`.
///
/// The fix here merges by `index` across deltas: the first non-empty `id`,
/// `type`, and `function.name` win; `function.arguments` are concatenated
/// across fragments. An empty identity value is treated as absent rather than
/// established, because consumers test identity by presence alone. This matches
/// the OpenAI streaming spec and vLLM/SGLang hermes emission:
///
/// * delta 1: `{index, id, type, function: { name }}`
/// * delta 2..N: `{index, function: { arguments: "<fragment>" }}`
///
/// Returns the first identity conflict observed, if any. Conflicts do not
/// change what is merged; see [`ToolCallMergeOutcome`].
pub(crate) fn merge_tool_call_chunk(
    existing: &mut ChatCompletionMessageToolCallChunk,
    incoming: ChatCompletionMessageToolCallChunk,
) -> ToolCallMergeOutcome {
    let mut conflict: Option<ToolCallIdentityField> = None;

    // Record the first conflicting field only; later fields are still merged
    // first-wins regardless.
    let mut note = |field: ToolCallIdentityField| {
        if conflict.is_none() {
            conflict = Some(field);
        }
    };

    // An empty incoming value is a producer no-op rather than a disagreement,
    // and an empty accumulated value is not yet an identity: consumers test
    // only for presence, so letting `Some("")` stand would dispatch a call
    // whose id and name are blank and suppress the real ones.
    if let Some(id) = incoming.id.filter(|id| !id.is_empty()) {
        match existing.id.as_deref() {
            None | Some("") => existing.id = Some(id),
            Some(current) if current != id => note(ToolCallIdentityField::Id),
            Some(_) => {}
        }
    }
    match (&existing.r#type, incoming.r#type) {
        (None, Some(ty)) => existing.r#type = Some(ty),
        // Forward-compat only: `FunctionType` has one variant today, so this
        // arm cannot fire. Kept so a second variant is not silently unreported.
        (Some(current), Some(ty)) if *current != ty => note(ToolCallIdentityField::Type),
        _ => {}
    }

    let Some(mut incoming_fn) = incoming.function else {
        return outcome(conflict);
    };
    match &mut existing.function {
        None => {
            // Adopting the first `function` wholesale would smuggle in an empty
            // name that the arm below would have refused.
            if incoming_fn.name.as_deref() == Some("") {
                incoming_fn.name = None;
            }
            existing.function = Some(incoming_fn);
        }
        Some(existing_fn) => {
            if let Some(name) = incoming_fn.name.filter(|name| !name.is_empty()) {
                match existing_fn.name.as_deref() {
                    None | Some("") => existing_fn.name = Some(name),
                    Some(current) if current != name => note(ToolCallIdentityField::Name),
                    Some(_) => {}
                }
            }
            if let Some(args_fragment) = incoming_fn.arguments {
                existing_fn
                    .arguments
                    .get_or_insert_with(String::new)
                    .push_str(&args_fragment);
            }
        }
    }

    outcome(conflict)
}

fn outcome(conflict: Option<ToolCallIdentityField>) -> ToolCallMergeOutcome {
    match conflict {
        None => ToolCallMergeOutcome::Merged,
        Some(field) => ToolCallMergeOutcome::IdentityConflict { field },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{FunctionCallStream, FunctionType};

    fn chunk(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> ChatCompletionMessageToolCallChunk {
        ChatCompletionMessageToolCallChunk {
            index,
            id: id.map(str::to_string),
            r#type: id.map(|_| FunctionType::Function),
            function: Some(FunctionCallStream {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
        }
    }

    #[test]
    fn concatenates_argument_fragments_and_reports_merged() {
        let mut acc = chunk(0, Some("call-1"), Some("calculator"), Some("{\"a\""));
        assert_eq!(
            merge_tool_call_chunk(&mut acc, chunk(0, None, None, Some(":1}"))),
            ToolCallMergeOutcome::Merged
        );
        let f = acc.function.unwrap();
        assert_eq!(f.arguments.as_deref(), Some("{\"a\":1}"));
        assert_eq!(f.name.as_deref(), Some("calculator"));
    }

    #[test]
    fn identity_is_first_wins_and_conflict_is_reported() {
        let mut acc = chunk(0, Some("call-1"), Some("calculator"), Some("{}"));
        let outcome =
            merge_tool_call_chunk(&mut acc, chunk(0, Some("call-2"), Some("other"), None));
        // First conflicting field wins the report; `id` is checked first.
        assert_eq!(
            outcome,
            ToolCallMergeOutcome::IdentityConflict {
                field: ToolCallIdentityField::Id
            }
        );
        assert_eq!(acc.id.as_deref(), Some("call-1"));
        assert_eq!(acc.function.unwrap().name.as_deref(), Some("calculator"));
    }

    #[test]
    fn empty_later_identity_value_is_not_a_conflict() {
        let mut acc = chunk(0, Some("call-1"), Some("calculator"), None);
        assert_eq!(
            merge_tool_call_chunk(&mut acc, chunk(0, Some(""), Some(""), Some("{}"))),
            ToolCallMergeOutcome::Merged
        );
        assert_eq!(acc.id.as_deref(), Some("call-1"));
    }

    #[test]
    fn empty_established_identity_value_is_replaced_by_the_real_one() {
        // An empty opener carries no identity, so the real id and name that
        // follow it fill the accumulator instead of being discarded.
        let mut acc = chunk(0, Some(""), Some(""), None);
        assert_eq!(
            merge_tool_call_chunk(&mut acc, chunk(0, Some("call-1"), Some("calculator"), None)),
            ToolCallMergeOutcome::Merged
        );
        assert_eq!(acc.id.as_deref(), Some("call-1"));
        assert_eq!(acc.function.unwrap().name.as_deref(), Some("calculator"));
    }

    #[test]
    fn an_only_ever_empty_identity_never_becomes_established() {
        // Consumers test identity by presence alone, so an empty value must
        // leave the accumulator without one rather than pass as an identity.
        let mut acc = chunk(0, None, None, None);
        assert_eq!(
            merge_tool_call_chunk(&mut acc, chunk(0, Some(""), Some(""), Some("{}"))),
            ToolCallMergeOutcome::Merged
        );
        assert_eq!(acc.id, None);
        assert_eq!(acc.function.unwrap().name, None);
    }

    #[test]
    fn name_conflict_is_named_when_id_agrees() {
        let mut acc = chunk(0, Some("call-1"), Some("calculator"), None);
        assert_eq!(
            merge_tool_call_chunk(&mut acc, chunk(0, Some("call-1"), Some("other"), None)),
            ToolCallMergeOutcome::IdentityConflict {
                field: ToolCallIdentityField::Name
            }
        );
    }
}

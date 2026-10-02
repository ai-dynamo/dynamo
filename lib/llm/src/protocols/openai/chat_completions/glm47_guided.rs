// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use dynamo_parsers_v2::GuidedJsonCursor;
use dynamo_protocols::types::{
    ChatCompletionMessageToolCallChunk, FunctionCallStream, FunctionType,
};

use crate::protocols::openai::GuidedToolConstraint;

use super::tool_parser_v2::parse_complete_guided_json;

#[derive(Default)]
enum Mode {
    #[default]
    Undecided,
    Native,
    Json,
}

pub(crate) struct GuidedState {
    constraint: GuidedToolConstraint,
    streaming: bool,
    mode: Mode,
    payload: String,
    cursor: GuidedJsonCursor,
    indices: BTreeMap<usize, u32>,
}

#[derive(Default)]
pub(crate) struct GuidedOutput {
    pub(crate) native: Option<String>,
    pub(crate) calls: Vec<ChatCompletionMessageToolCallChunk>,
    pub(crate) dropped: bool,
}

impl GuidedState {
    pub(crate) fn new(constraint: GuidedToolConstraint, streaming: bool) -> Self {
        let cursor = match &constraint {
            GuidedToolConstraint::GuidedJsonNamed { tool_name } => {
                GuidedJsonCursor::named(tool_name)
            }
            _ => GuidedJsonCursor::new(),
        };
        Self {
            constraint,
            streaming,
            mode: Mode::Undecided,
            payload: String::new(),
            cursor,
            indices: BTreeMap::new(),
        }
    }

    pub(crate) fn push(&mut self, text: &str, next_call: &mut u32) -> GuidedOutput {
        if matches!(self.mode, Mode::Native) {
            return GuidedOutput {
                native: Some(text.to_string()),
                ..Default::default()
            };
        }
        self.payload.push_str(text);
        if matches!(self.mode, Mode::Undecided) {
            self.mode = match self.payload.trim_start().chars().next() {
                Some('{' | '[') => Mode::Json,
                Some(_) => Mode::Native,
                None => return GuidedOutput::default(),
            };
            if matches!(self.mode, Mode::Native) {
                return GuidedOutput {
                    native: Some(std::mem::take(&mut self.payload)),
                    ..Default::default()
                };
            }
        }
        if !self.streaming {
            return GuidedOutput::default();
        }
        let mut deltas = Vec::new();
        self.cursor.advance(&self.payload, &mut deltas);
        let calls = deltas
            .into_iter()
            .map(|delta| {
                let index = *self.indices.entry(delta.tool_index).or_insert_with(|| {
                    let index = *next_call;
                    *next_call = next_call.saturating_add(1);
                    index
                });
                let first = delta.name.is_some();
                ChatCompletionMessageToolCallChunk {
                    index,
                    id: first.then(|| format!("call-{}", uuid::Uuid::new_v4())),
                    r#type: first.then_some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: delta.name,
                        arguments: Some(delta.arguments),
                    }),
                }
            })
            .collect();
        GuidedOutput {
            calls,
            ..Default::default()
        }
    }

    pub(crate) fn finish(&mut self, next_call: &mut u32) -> GuidedOutput {
        match self.mode {
            Mode::Native => return GuidedOutput::default(),
            Mode::Undecided => {
                return GuidedOutput {
                    native: Some(std::mem::take(&mut self.payload)),
                    ..Default::default()
                };
            }
            Mode::Json => {}
        }
        if matches!(self.constraint, GuidedToolConstraint::GuidedJsonRequired)
            && let Ok(payload) = serde_json::from_str::<serde_json::Value>(&self.payload)
        {
            let envelopes = payload
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_else(|| std::slice::from_ref(&payload));
            if envelopes.iter().any(|envelope| {
                envelope.get("arguments").is_some() && envelope.get("parameters").is_some()
            }) {
                self.payload.clear();
                return GuidedOutput {
                    dropped: true,
                    ..Default::default()
                };
            }
        }
        let parsed = parse_complete_guided_json(&self.payload, &self.constraint);
        let mut output = GuidedOutput::default();
        match parsed {
            Ok(calls) if !calls.is_empty() => {
                for (source_index, call) in calls.into_iter().enumerate() {
                    let arguments_are_object =
                        serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                            .is_ok_and(|arguments| arguments.is_object());
                    if !arguments_are_object
                        || self.cursor.committed().iter().any(|committed| {
                            committed.index == source_index
                                && (committed.ambiguous || committed.name != call.function.name)
                        })
                    {
                        output.dropped = true;
                        continue;
                    }
                    if self.indices.contains_key(&source_index) {
                        continue;
                    }
                    let index = *next_call;
                    *next_call = next_call.saturating_add(1);
                    output.calls.push(ChatCompletionMessageToolCallChunk {
                        index,
                        id: Some(call.id),
                        r#type: Some(FunctionType::Function),
                        function: Some(FunctionCallStream {
                            name: Some(call.function.name),
                            arguments: Some(call.function.arguments),
                        }),
                    });
                }
            }
            Ok(_) => output.dropped = true,
            Err(error) => {
                tracing::debug!(error = %error, "dropping invalid guided tool payload");
                output.dropped = true;
            }
        }
        self.payload.clear();
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_and_named_keep_early_release_and_rollback() {
        for (constraint, first, last) in [
            (
                GuidedToolConstraint::GuidedJsonRequired,
                "[{\"name\":\"weather\",\"arguments\":{\"city\":\"Pa",
                "ris\"}}]",
            ),
            (
                GuidedToolConstraint::GuidedJsonNamed {
                    tool_name: "weather".to_string(),
                },
                "{\"city\":\"Pa",
                "ris\"}",
            ),
        ] {
            for streaming in [false, true] {
                let mut state = GuidedState::new(constraint.clone(), streaming);
                let mut index = 3;
                let first = state.push(first, &mut index);
                assert_eq!(!first.calls.is_empty(), streaming);
                let last = state.push(last, &mut index);
                let finish = state.finish(&mut index);
                assert!(!finish.dropped);
                let calls: Vec<_> = first
                    .calls
                    .into_iter()
                    .chain(last.calls)
                    .chain(finish.calls)
                    .collect();
                assert!(calls.iter().all(|call| call.index == 3));
                assert_eq!(calls.iter().filter(|call| call.id.is_some()).count(), 1);
                let arguments: String = calls
                    .iter()
                    .filter_map(|call| call.function.as_ref())
                    .filter_map(|function| function.arguments.as_deref())
                    .collect();
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&arguments).unwrap()["city"],
                    "Paris"
                );
            }
        }
    }

    #[test]
    fn malformed_json_does_not_become_native_fallback() {
        let mut state = GuidedState::new(GuidedToolConstraint::GuidedJsonRequired, true);
        let mut index = 0;
        let first = state.push(
            "[{\"name\":\"weather\",\"arguments\":{\"literal\":\"<tool_call>",
            &mut index,
        );
        assert!(!first.calls.is_empty());
        assert!(first.native.is_none());
        let finish = state.finish(&mut index);
        assert!(finish.dropped);
        assert!(finish.native.is_none());
        assert!(finish.calls.is_empty());
    }

    #[test]
    fn native_fallback_preserves_leading_whitespace_and_literal_quotes() {
        let mut state = GuidedState::new(GuidedToolConstraint::GuidedJsonRequired, true);
        let mut index = 0;
        assert!(state.push(" \n", &mut index).native.is_none());
        let output = state.push("\"literal <tool_call>weather</tool_call>", &mut index);
        assert_eq!(
            output.native.as_deref(),
            Some(" \n\"literal <tool_call>weather</tool_call>")
        );
        assert!(output.calls.is_empty());
        assert!(!state.finish(&mut index).dropped);
    }

    #[test]
    fn ambiguous_argument_aliases_are_not_committed_on_rollback() {
        for streaming in [false, true] {
            let mut state = GuidedState::new(GuidedToolConstraint::GuidedJsonRequired, streaming);
            let mut index = 0;
            state.push(
                "[{\"name\":\"weather\",\"arguments\":{},\"parameters\":{}}]",
                &mut index,
            );
            let finish = state.finish(&mut index);
            assert!(finish.dropped);
            assert!(finish.calls.is_empty());
        }
    }

    #[test]
    fn changed_committed_name_is_not_reported_as_a_valid_call() {
        let mut state = GuidedState::new(GuidedToolConstraint::GuidedJsonRequired, true);
        let mut index = 0;
        let first = state.push("[{\"name\":\"weather\",\"arguments\":{}", &mut index);
        assert_eq!(
            first.calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("weather")
        );
        state.push(",\"name\":\"search\"}]", &mut index);
        let finish = state.finish(&mut index);
        assert!(finish.dropped);
        assert!(finish.calls.is_empty());
    }
}

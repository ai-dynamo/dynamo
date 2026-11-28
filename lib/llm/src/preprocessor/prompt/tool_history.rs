// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Project tool results for renderers that associate them by position.

use std::collections::HashMap;

use dynamo_protocols::types::{
    ChatCompletionRequestMessage as Message, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestToolMessageContent as ToolContent,
    ChatCompletionRequestToolMessageContentPart as ToolPart, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContentPart as UserPart,
};
use dynamo_renderer::RenderedPrompt;

use super::is_kimi_k3_prompt_hint;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PromptToolHistory {
    #[default]
    AsProvided,
    Positional,
}

impl PromptToolHistory {
    pub(crate) fn prepare_request(
        self,
        request: &mut NvCreateChatCompletionRequest,
    ) -> ToolResultIndices {
        if self != Self::Positional
            || !request
                .inner
                .messages
                .iter()
                .any(|message| matches!(message, Message::Tool(_)))
        {
            return ToolResultIndices::default();
        }
        let mut phases = request
            .responses_tools
            .as_mut()
            .map(|tools| &mut tools.history_phases)
            .filter(|phases| !phases.is_empty());
        let assistant_phases = phases.as_deref_mut().map(|phases| {
            request
                .inner
                .messages
                .iter()
                .enumerate()
                .filter(|(_, message)| matches!(message, Message::Assistant(_)))
                .map(|(index, _)| phases.remove(&index))
                .collect::<Vec<_>>()
        });
        let (messages, indices) =
            positional_tool_history(std::mem::take(&mut request.inner.messages));
        request.inner.messages = messages;
        if let (Some(phases), Some(assistant_phases)) = (phases, assistant_phases) {
            *phases = request
                .inner
                .messages
                .iter()
                .enumerate()
                .filter_map(|(index, message)| {
                    matches!(message, Message::Assistant(_)).then_some(index)
                })
                .zip(assistant_phases)
                .filter_map(|(index, phase)| phase.map(|phase| (index, phase)))
                .collect();
        }
        indices
    }
}

#[derive(Debug, Default)]
pub(crate) struct ToolResultIndices {
    result_count: usize,
    corrections: HashMap<usize, usize>,
}

impl ToolResultIndices {
    fn append(&mut self, matched: &mut Vec<(usize, Message)>, settled: &mut Vec<Message>) {
        matched.sort_unstable_by_key(|(index, _)| *index);
        for (ordinal, (index, message)) in matched.drain(..).enumerate() {
            if ordinal != index {
                self.corrections.insert(self.result_count, index + 1);
            }
            self.result_count += 1;
            settled.push(message);
        }
    }

    pub(crate) fn apply(&self, prompt: RenderedPrompt) -> anyhow::Result<RenderedPrompt> {
        let Some(segments) = prompt.segments().filter(|_| !self.corrections.is_empty()) else {
            return Ok(prompt);
        };
        // In the pinned XTML renderer, only the tag delimiters are special
        // tokens. The index is the last attribute of a tool message header.
        let mut previous_control = None;
        let mut updated = segments.to_vec();
        let mut result_count = 0;
        for (end, control) in segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| segment.allow_special)
        {
            let Some(start) = previous_control.replace(end) else {
                continue;
            };
            if segments[start].text != "<|open|>" || control.text != "<|sep|>" {
                continue;
            }
            let header = &segments[start + 1..end];
            if !header
                .iter()
                .take(5)
                .map(|segment| segment.text.as_str())
                .eq(["message", " role", "=\"", "tool", "\""])
            {
                continue;
            }
            let ordinal = result_count;
            result_count += 1;
            let Some(index) = self.corrections.get(&ordinal) else {
                continue;
            };
            let [.., key, equals, _, quote] = header else {
                anyhow::bail!("positional tool result is missing its index");
            };
            anyhow::ensure!(
                key.text == " index" && equals.text == "=\"" && quote.text == "\"",
                "positional tool result is missing its index"
            );
            updated[end - 2].text = index.to_string();
        }
        anyhow::ensure!(
            result_count == self.result_count,
            "positional prompt rendered an unexpected number of tool results"
        );
        Ok(RenderedPrompt::segmented_with_pending(
            updated,
            prompt.pending_segments(),
        ))
    }
}

impl NvCreateChatCompletionRequest {
    /// Alternate history for an in-process positional renderer selected by Python.
    pub fn python_positional_tool_history(
        &self,
        runtime_config: &ModelRuntimeConfig,
    ) -> anyhow::Result<Option<Vec<Message>>> {
        let can_select_positional_renderer = [
            Some(self.inner.model.as_str()),
            runtime_config.tool_call_parser.as_deref(),
            runtime_config.reasoning_parser.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(is_kimi_k3_prompt_hint);
        if !can_select_positional_renderer
            || !self.inner.messages.iter().any(|message| match message {
                Message::Tool(_) => true,
                Message::Assistant(assistant) => {
                    self.responses_tools.is_some()
                        && assistant
                            .tool_calls
                            .as_ref()
                            .is_some_and(|calls| !calls.is_empty())
                }
                _ => false,
            })
        {
            return Ok(None);
        }
        let messages = if self.responses_tools.is_some() {
            let mut request = self.clone();
            request.normalize_responses_tools_for_renderer(false, PromptToolHistory::Positional)?;
            request.inner.messages
        } else {
            self.inner.messages.clone()
        };
        Ok(Some(positional_tool_history(messages).0))
    }
}

/// Coalesce chunks for the same call without crossing a non-tool message.
fn merge_split_tool_results(messages: Vec<Message>) -> Vec<Message> {
    let mut merged = Vec::with_capacity(messages.len());
    let mut run: HashMap<String, usize> = HashMap::new();
    for message in messages {
        if let Message::Tool(tool) = message {
            if let Some(index) = run.get(&tool.tool_call_id).copied()
                && let Message::Tool(first) = &mut merged[index]
            {
                merge_tool_content(&mut first.content, tool.content);
                continue;
            }
            run.insert(tool.tool_call_id.clone(), merged.len());
            merged.push(Message::Tool(tool));
        } else {
            run.clear();
            merged.push(message);
        }
    }
    merged
}

/// Positional renderers number results relative to the latest assistant batch.
fn positional_tool_history(messages: Vec<Message>) -> (Vec<Message>, ToolResultIndices) {
    if !messages
        .iter()
        .any(|message| matches!(message, Message::Tool(_)))
    {
        return (messages, ToolResultIndices::default());
    }

    let merged = merge_split_tool_results(messages);
    let mut settled = Vec::with_capacity(merged.len());
    let mut notes = Vec::new();
    let mut calls = HashMap::new();
    let mut matched = Vec::new();
    let mut indices = ToolResultIndices::default();
    for message in merged {
        if let Message::Tool(tool) = message {
            if let Some(&index) = calls.get(&tool.tool_call_id) {
                matched.push((index, Message::Tool(tool)));
            } else {
                let id = tool.tool_call_id;
                let mut parts = vec![UserPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: format!(
                            "[Output of tool call '{id}', which matches no tool call before it:]"
                        ),
                        prompt_cache_breakpoint: None,
                    },
                )];
                parts.extend(tool_parts(tool.content).into_iter().map(|part| match part {
                    ToolPart::Text(text) => UserPart::Text(text),
                    ToolPart::ImageUrl(image) => UserPart::ImageUrl(image),
                    ToolPart::VideoUrl(video) => UserPart::VideoUrl(video),
                    ToolPart::AudioUrl(audio) => UserPart::AudioUrl(audio),
                    ToolPart::Custom(custom) => UserPart::Custom(custom),
                }));
                notes.push(Message::User(ChatCompletionRequestUserMessage {
                    content: parts.into(),
                    name: None,
                }));
            }
        } else {
            indices.append(&mut matched, &mut settled);
            settled.append(&mut notes);
            calls.clear();
            if let Message::Assistant(assistant) = &message {
                calls.extend(
                    assistant
                        .tool_calls
                        .iter()
                        .flatten()
                        .enumerate()
                        .map(|(index, call)| (call.id.clone(), index)),
                );
            }
            settled.push(message);
        }
    }
    indices.append(&mut matched, &mut settled);
    settled.append(&mut notes);
    (settled, indices)
}

fn merge_tool_content(first: &mut ToolContent, next: ToolContent) {
    match (first, next) {
        (ToolContent::Text(first), ToolContent::Text(next)) => {
            if !first.is_empty() && !first.ends_with('\n') {
                first.push('\n');
            }
            first.push_str(&next);
        }
        (ToolContent::Array(parts), next) => parts.extend(tool_parts(next)),
        (first, next) => {
            let mut parts = tool_parts(std::mem::take(first));
            parts.extend(tool_parts(next));
            *first = ToolContent::Array(parts);
        }
    }
}

fn tool_parts(content: ToolContent) -> Vec<ToolPart> {
    match content {
        ToolContent::Array(parts) => parts,
        ToolContent::Text(text) => vec![ToolPart::Text(
            ChatCompletionRequestMessageContentPartText {
                text,
                prompt_cache_breakpoint: None,
            },
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_results_do_not_shift_call_indices_or_lose_orphans() {
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({"model":"team/Kimi--K3","messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"type":"function","id":"a","function":{"name":"one","arguments":"{}"}},
                    {"type":"function","id":"b","function":{"name":"two","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"b","content":"second"},
                {"role":"tool","tool_call_id":"missing","content":"orphan"},
                {"role":"tool","tool_call_id":"a","content":"first"},
                {"role":"tool","tool_call_id":"a","content":"continued"}
            ]}))
            .unwrap();
        let normalized = request
            .python_positional_tool_history(&Default::default())
            .unwrap()
            .unwrap();
        PromptToolHistory::Positional.prepare_request(&mut request);
        assert_eq!(request.inner.messages, normalized);
        let value = serde_json::to_value(&normalized).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 4);
        assert_eq!(value[1]["content"], "first\ncontinued");
        assert_eq!(value[2]["tool_call_id"], "b");
        assert_eq!(value[3]["role"], "user");
        assert_eq!(value[3]["content"][1]["text"], "orphan");
        assert_eq!(positional_tool_history(normalized.clone()).0, normalized);
    }

    #[test]
    fn user_message_ends_the_tool_call_batch() {
        let messages: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role":"assistant","content":null,"tool_calls":[
                {"type":"function","id":"a","function":{"name":"one","arguments":"{}"}}
            ]},
            {"role":"user","content":"new turn"},
            {"role":"tool","tool_call_id":"a","content":"late result"}
        ]))
        .unwrap();
        let (normalized, _) = positional_tool_history(messages);
        let value = serde_json::to_value(&normalized).unwrap();
        assert_eq!(value[2]["role"], "user");
        assert_eq!(value[2]["content"][1]["text"], "late result");
        assert_eq!(positional_tool_history(normalized.clone()).0, normalized);
    }
}

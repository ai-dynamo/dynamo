// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
    ChatCompletionResponseContentPart, ChatCompletionResponseContentPartText,
    ChatCompletionStreamResponseDelta, ChatCompletionToolChoiceOption, FinishReason,
    FunctionCallStream, FunctionType,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use super::NvCreateChatCompletionStreamResponse;
use super::glm47_framing::{Glm47Frame, Glm47Framer, parse_block};
use super::glm47_guided::{GuidedOutput, GuidedState};
use crate::protocols::openai::GuidedToolConstraint;

#[derive(Default)]
struct ChoiceState {
    framer: Glm47Framer,
    next_call: u32,
    has_calls: bool,
    dropped_call: bool,
    terminal_emitted: bool,
    worker_calls: BTreeMap<u32, WorkerCall>,
    guided: Option<GuidedState>,
}

#[derive(Default)]
struct WorkerCall {
    output_index: Option<u32>,
    allowed: Option<bool>,
    pending: Vec<ChatCompletionMessageToolCallChunk>,
}

impl ChoiceState {
    fn append_guided(
        &mut self,
        output: GuidedOutput,
        choice: &mut ChatChoiceStream,
        tools: Option<&[ToolDefinition]>,
        forced_name: Option<&str>,
    ) {
        self.dropped_call |= output.dropped;
        if !output.calls.is_empty() {
            self.has_calls = true;
            choice
                .delta
                .tool_calls
                .get_or_insert_with(Vec::new)
                .extend(output.calls);
        }
        if let Some(native) = output.native {
            let frames = self.framer.push(&native);
            self.append_frames(frames, choice, tools, forced_name);
        }
    }
    fn map_worker_calls(&mut self, choice: &mut ChatChoiceStream, forced_name: Option<&str>) {
        let mut output = Vec::new();
        for call in choice.delta.tool_calls.take().unwrap_or_default() {
            let state = self.worker_calls.entry(call.index).or_default();
            if let Some(forced_name) = forced_name {
                if let Some(name) = call
                    .function
                    .as_ref()
                    .and_then(|function| function.name.as_deref())
                {
                    state.allowed = Some(name == forced_name);
                }
            } else {
                state.allowed = Some(true);
            }
            state.pending.push(call);
            match state.allowed {
                Some(true) => {
                    let index = *state.output_index.get_or_insert_with(|| {
                        let index = self.next_call;
                        self.next_call = self.next_call.saturating_add(1);
                        index
                    });
                    output.extend(state.pending.drain(..).map(|mut call| {
                        call.index = index;
                        call
                    }));
                    self.has_calls = true;
                }
                Some(false) => state.pending.clear(),
                None => {}
            }
        }
        choice.delta.tool_calls = (!output.is_empty()).then_some(output);
    }

    fn append_frames(
        &mut self,
        frames: Vec<Glm47Frame>,
        choice: &mut ChatChoiceStream,
        tools: Option<&[ToolDefinition]>,
        forced_name: Option<&str>,
    ) {
        let mut text = String::new();
        for frame in frames {
            match frame {
                Glm47Frame::Text(content) => text.push_str(&content),
                Glm47Frame::ToolBlock(block) => match parse_block(&block, tools) {
                    Ok((calls, content)) if !calls.is_empty() => {
                        text.push_str(content.as_deref().unwrap_or_default());
                        for call in calls {
                            if forced_name.is_some_and(|name| name != call.function.name) {
                                tracing::warn!(
                                    choice_index = choice.index,
                                    "dropping native call that does not match the named tool"
                                );
                                continue;
                            }
                            choice.delta.tool_calls.get_or_insert_with(Vec::new).push(
                                ChatCompletionMessageToolCallChunk {
                                    index: self.next_call,
                                    id: Some(call.id),
                                    r#type: Some(FunctionType::Function),
                                    function: Some(FunctionCallStream {
                                        name: Some(call.function.name),
                                        arguments: Some(call.function.arguments),
                                    }),
                                },
                            );
                            self.next_call = self.next_call.saturating_add(1);
                            self.has_calls = true;
                        }
                    }
                    Ok(_) => self.dropped_call = true,
                    Err(error) => {
                        self.dropped_call = true;
                        tracing::debug!(error = %error, choice_index = choice.index, "dropping malformed native tool call");
                    }
                },
            }
        }
        if !text.is_empty() {
            match choice.delta.content.as_mut() {
                Some(ChatCompletionMessageContent::Text(content)) => content.push_str(&text),
                None => choice.delta.content = Some(ChatCompletionMessageContent::Text(text)),
                Some(ChatCompletionMessageContent::Parts(parts)) => parts.insert(
                    0,
                    ChatCompletionResponseContentPart::Text(
                        ChatCompletionResponseContentPartText { text },
                    ),
                ),
            }
        }
    }

    fn finish(
        &mut self,
        choice: &mut ChatChoiceStream,
        tools: Option<&[ToolDefinition]>,
        forced_name: Option<&str>,
    ) {
        if let Some(guided) = self.guided.as_mut() {
            let output = guided.finish(&mut self.next_call);
            self.append_guided(output, choice, tools, forced_name);
        }
        let finish = self.framer.finish();
        self.append_frames(finish.frames, choice, tools, forced_name);
        choice.finish_reason = Some(match choice.finish_reason {
            Some(FinishReason::Length) => FinishReason::Length,
            Some(FinishReason::ContentFilter) => FinishReason::ContentFilter,
            reason @ (Some(FinishReason::Stop) | Some(FinishReason::ToolCalls) | None) => {
                if finish.incomplete_tool_call || self.dropped_call {
                    tracing::warn!(
                        choice_index = choice.index,
                        why = "dropped_tool_output_reported_as_length",
                        "reporting incomplete or invalid tool output as length"
                    );
                    FinishReason::Length
                } else if self.has_calls {
                    FinishReason::ToolCalls
                } else {
                    match reason {
                        Some(FinishReason::ToolCalls) | None => FinishReason::Stop,
                        Some(reason) => reason,
                    }
                }
            }
            Some(other) => other,
        });
        self.terminal_emitted = true;
    }
}

/// Native GLM calls are decoded only after framing has excluded quoted prose.
/// `Length` also signals an unfinished native call at EOS, not only a token limit.
#[cfg(test)]
pub(crate) fn apply_stream<S>(
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    tools: Option<Vec<ToolDefinition>>,
    stream: S,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    apply_stream_with_guided_constraint(tool_choice, tools, false, false, stream)
}

pub(crate) fn apply_stream_with_guided_constraint<S>(
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    tools: Option<Vec<ToolDefinition>>,
    uses_guided_json: bool,
    guided_streaming: bool,
    stream: S,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    async_stream::stream! {
        let forced_name = match &tool_choice {
            Some(ChatCompletionToolChoiceOption::Named(named)) => Some(named.function.name.as_str()),
            _ => None,
        };
        let mut states = BTreeMap::<u32, ChoiceState>::new();
        let constraint = if uses_guided_json {
            match &tool_choice {
                Some(ChatCompletionToolChoiceOption::Required) => Some(GuidedToolConstraint::GuidedJsonRequired),
                Some(ChatCompletionToolChoiceOption::Named(named)) => Some(GuidedToolConstraint::GuidedJsonNamed { tool_name: named.function.name.clone() }),
                _ => None,
            }
        } else {
            None
        };
        let mut template = None;
        tokio::pin!(stream);
        while let Some(mut response) = stream.next().await {
            if response.is_error() {
                yield response;
                return;
            }
            if let Some(data) = response.data.as_mut() {
                if template.is_none() && !data.inner.choices.is_empty() {
                    let mut inner = data.inner.clone();
                    inner.choices.clear();
                    inner.usage = None;
                    template = Some(inner);
                }
                let mut choices = Vec::with_capacity(data.inner.choices.len());
                for mut choice in std::mem::take(&mut data.inner.choices) {
                    let state = states.entry(choice.index).or_insert_with(|| ChoiceState {
                        guided: constraint.clone().map(|constraint| GuidedState::new(constraint, guided_streaming)),
                        ..Default::default()
                    });
                    if state.terminal_emitted {
                        continue;
                    }
                    state.map_worker_calls(&mut choice, forced_name);
                    match choice.delta.content.take() {
                        Some(ChatCompletionMessageContent::Text(text)) => {
                            if let Some(guided) = state.guided.as_mut() {
                                let output = guided.push(&text, &mut state.next_call);
                                state.append_guided(output, &mut choice, tools.as_deref(), forced_name);
                            } else {
                                let frames = state.framer.push(&text);
                                state.append_frames(frames, &mut choice, tools.as_deref(), forced_name);
                            }
                        }
                        Some(parts @ ChatCompletionMessageContent::Parts(_)) => {
                            choice.delta.content = Some(parts);
                            let frames = state.framer.flush_prose();
                            state.append_frames(frames, &mut choice, tools.as_deref(), forced_name);
                        }
                        None => {}
                    }
                    if choice.finish_reason.is_some() {
                        state.finish(&mut choice, tools.as_deref(), forced_name);
                    }
                    choices.push(choice);
                }
                data.inner.choices = choices;
            }
            // Even a buffered content chunk carries token metadata and annotations.
            yield response;
        }
        if let Some(mut inner) = template {
            for (index, state) in &mut states {
                if state.terminal_emitted {
                    continue;
                }
                let mut choice = ChatChoiceStream {
                    index: *index,
                    delta: ChatCompletionStreamResponseDelta {
                        role: None,
                        content: None,
                        tool_calls: None,
                        function_call: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                };
                state.finish(&mut choice, tools.as_deref(), forced_name);
                inner.choices.push(choice);
            }
            if !inner.choices.is_empty() {
                yield Annotated {
                    data: Some(NvCreateChatCompletionStreamResponse {
                        inner,
                        nvext: None,
                        llm_metrics: None,
                    }),
                    id: None,
                    event: None,
                    comment: None,
                    error: None,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::ChatCompletionResponseContentPart;
    use futures::stream;
    use serde_json::{Value, json};

    fn chunk(
        delta: Value,
        terminal: Option<&str>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        Annotated::from_data(NvCreateChatCompletionStreamResponse {
            inner: serde_json::from_value(json!({
                "id": "test", "model": "glm", "created": 1, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": terminal}]
            }))
            .unwrap(),
            nvext: None,
            llm_metrics: None,
        })
    }

    fn choices(
        output: &[Annotated<NvCreateChatCompletionStreamResponse>],
    ) -> impl Iterator<Item = &ChatChoiceStream> {
        output
            .iter()
            .filter_map(|response| response.data.as_ref())
            .flat_map(|response| &response.inner.choices)
    }

    fn content_sequence(output: &[Annotated<NvCreateChatCompletionStreamResponse>]) -> Vec<String> {
        let mut sequence = Vec::new();
        for choice in choices(output) {
            match choice.delta.content.as_ref() {
                Some(ChatCompletionMessageContent::Text(text)) => sequence.push(text.clone()),
                Some(ChatCompletionMessageContent::Parts(parts)) => {
                    sequence.extend(parts.iter().map(|part| match part {
                        ChatCompletionResponseContentPart::Text(text) => text.text.clone(),
                        ChatCompletionResponseContentPart::ImageUrl(_) => "image".to_string(),
                        other => panic!("unexpected content part: {other:?}"),
                    }));
                }
                None => {}
            }
        }
        sequence
    }

    #[tokio::test]
    async fn mixed_native_and_worker_calls_have_distinct_stable_indices() {
        let input = vec![
            chunk(json!({"content": "<tool_call>weather</tool_call>"}), None),
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "worker-call", "type": "function", "function": {"name": "search", "arguments": "{\"q\":"}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"Dynamo\"}"}}]}),
                Some("stop"),
            ),
        ];
        let output: Vec<_> = apply_stream(None, None, stream::iter(input))
            .collect()
            .await;
        let calls: Vec<_> = choices(&output)
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();
        assert_eq!(
            calls.iter().map(|call| call.index).collect::<Vec<_>>(),
            vec![0, 1, 1]
        );
        assert_eq!(calls[1].id.as_deref(), Some("worker-call"));
        assert_eq!(
            choices(&output)
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls]
        );
    }

    #[tokio::test]
    async fn named_choice_rejects_later_argument_fragments() {
        let named =
            serde_json::from_value(json!({"type": "function", "function": {"name": "weather"}}))
                .unwrap();
        let input = vec![
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "wrong", "type": "function", "function": {"name": "search", "arguments": "{"}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"q\":\"x\"}"}}]}),
                Some("stop"),
            ),
        ];
        let output: Vec<_> = apply_stream(Some(named), None, stream::iter(input))
            .collect()
            .await;
        assert!(choices(&output).all(|choice| choice.delta.tool_calls.is_none()));
        assert_eq!(
            choices(&output)
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::Stop]
        );
    }

    #[tokio::test]
    async fn terminal_parts_do_not_drop_buffered_safe_prose() {
        let input = vec![
            chunk(json!({"content": "Hello"}), None),
            chunk(
                json!({"content": [{"type": "image_url", "image_url": {"url": "image"}}]}),
                Some("stop"),
            ),
        ];
        let output: Vec<_> = apply_stream(None, None, stream::iter(input))
            .collect()
            .await;
        assert_eq!(content_sequence(&output), vec!["Hello", "image"]);
        assert_eq!(
            choices(&output)
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::Stop]
        );
    }

    #[tokio::test]
    async fn nonterminal_parts_do_not_reorder_buffered_prose() {
        let input = vec![
            chunk(json!({"content": "Hello"}), None),
            chunk(
                json!({"content": [{"type": "image_url", "image_url": {"url": "image"}}]}),
                None,
            ),
            chunk(json!({"content": " world"}), Some("stop")),
        ];
        let output: Vec<_> = apply_stream(None, None, stream::iter(input))
            .collect()
            .await;
        assert_eq!(content_sequence(&output), vec!["Hello", "image", " world"]);
    }
}

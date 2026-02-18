// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Anthropic Messages API SSE events.
//!
//! The event sequence follows the Anthropic streaming spec:
//! `message_start` -> `content_block_start` -> N x `content_block_delta` ->
//! `content_block_stop` -> `message_delta` -> `message_stop`

use std::collections::HashSet;

use axum::response::sse::Event;
use dynamo_async_openai::types::ChatCompletionMessageContent;
use uuid::Uuid;

use super::types::{
    AnthropicDelta, AnthropicErrorBody, AnthropicMessageDeltaBody, AnthropicMessageResponse,
    AnthropicResponseContentBlock, AnthropicStopReason, AnthropicStreamEvent, AnthropicUsage,
};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// State machine that converts a chat completion stream into Anthropic SSE events.
pub struct AnthropicStreamConverter {
    model: String,
    message_id: String,
    // Text tracking
    text_block_started: bool,
    text_block_closed: bool,
    text_block_index: u32,
    // Token usage (from engine)
    input_token_count: u32,
    output_token_count: u32,
    // Tool call tracking
    tool_call_states: Vec<ToolCallState>,
    tool_calls_sent: HashSet<String>,
    // Block index counter
    next_block_index: u32,
    // Stop reason
    stop_reason: Option<AnthropicStopReason>,
}

struct ToolCallState {
    id: String,
    name: String,
    accumulated_args: String,
    block_index: u32,
    started: bool,
}

impl AnthropicStreamConverter {
    pub fn new(model: String) -> Self {
        Self {
            model,
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            text_block_started: false,
            text_block_closed: false,
            text_block_index: 0,
            input_token_count: 0,
            output_token_count: 0,
            tool_call_states: Vec::new(),
            tool_calls_sent: HashSet::new(),
            next_block_index: 0,
            stop_reason: None,
        }
    }

    /// Emit the initial `message_start` event.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let message = AnthropicMessageResponse {
            id: self.message_id.clone(),
            object_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: self.model.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 0,
                output_tokens: 0,
            },
        };

        let event = AnthropicStreamEvent::MessageStart { message };
        vec![make_sse_event("message_start", &event)]
    }

    /// Process a single chat completion stream chunk and return zero or more SSE events.
    pub fn process_chunk(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();

        // Capture real token usage from engine when available (typically on the final chunk).
        if let Some(usage) = &chunk.usage {
            self.input_token_count = usage.prompt_tokens;
            self.output_token_count = usage.completion_tokens;
        }

        for choice in &chunk.choices {
            let delta = &choice.delta;

            // Track finish reason
            if let Some(ref fr) = choice.finish_reason {
                self.stop_reason = Some(match fr {
                    dynamo_async_openai::types::FinishReason::Stop => AnthropicStopReason::EndTurn,
                    dynamo_async_openai::types::FinishReason::Length => {
                        AnthropicStopReason::MaxTokens
                    }
                    dynamo_async_openai::types::FinishReason::ToolCalls => {
                        AnthropicStopReason::ToolUse
                    }
                    dynamo_async_openai::types::FinishReason::ContentFilter => {
                        AnthropicStopReason::EndTurn
                    }
                    dynamo_async_openai::types::FinishReason::FunctionCall => {
                        AnthropicStopReason::ToolUse
                    }
                });
            }

            // Handle text content deltas
            let content_text = match &delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            };

            if let Some(text) = content_text
                && !text.is_empty()
            {
                // Emit content_block_start on first text
                if !self.text_block_started {
                    self.text_block_started = true;
                    self.text_block_index = self.next_block_index;
                    self.next_block_index += 1;

                    let block_start = AnthropicStreamEvent::ContentBlockStart {
                        index: self.text_block_index,
                        content_block: AnthropicResponseContentBlock::Text {
                            text: String::new(),
                        },
                    };
                    events.push(make_sse_event("content_block_start", &block_start));
                }

                // Emit text delta
                let block_delta = AnthropicStreamEvent::ContentBlockDelta {
                    index: self.text_block_index,
                    delta: AnthropicDelta::TextDelta {
                        text: text.to_string(),
                    },
                };
                events.push(make_sse_event("content_block_delta", &block_delta));
            }

            // Handle tool call deltas
            if let Some(tool_calls) = &delta.tool_calls {
                // Close the text block before opening any tool blocks.
                // Anthropic streaming spec requires each block to be closed
                // (content_block_stop) before the next block starts.
                if self.text_block_started && !self.text_block_closed {
                    self.text_block_closed = true;
                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.text_block_index,
                    };
                    events.push(make_sse_event("content_block_stop", &block_stop));
                }

                for tc in tool_calls {
                    let tc_index = tc.index as usize;

                    // Ensure we have state for this tool call index
                    while self.tool_call_states.len() <= tc_index {
                        let block_index = self.next_block_index;
                        self.next_block_index += 1;
                        self.tool_call_states.push(ToolCallState {
                            id: String::new(),
                            name: String::new(),
                            accumulated_args: String::new(),
                            block_index,
                            started: false,
                        });
                    }

                    // Update id and name if provided
                    if let Some(id) = &tc.id {
                        self.tool_call_states[tc_index].id = id.clone();
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            self.tool_call_states[tc_index].name = name.clone();
                        }
                        if let Some(args) = &func.arguments {
                            // Emit content_block_start on first delta for this tool call
                            if !self.tool_call_states[tc_index].started {
                                let tc_id = self.tool_call_states[tc_index].id.clone();

                                // Dedup guard: skip if we've already emitted this tool call ID
                                if !tc_id.is_empty() && self.tool_calls_sent.contains(&tc_id) {
                                    continue;
                                }

                                self.tool_call_states[tc_index].started = true;
                                let block_index = self.tool_call_states[tc_index].block_index;
                                let tc_name = self.tool_call_states[tc_index].name.clone();

                                if !tc_id.is_empty() {
                                    self.tool_calls_sent.insert(tc_id.clone());
                                }

                                let block_start = AnthropicStreamEvent::ContentBlockStart {
                                    index: block_index,
                                    content_block: AnthropicResponseContentBlock::ToolUse {
                                        id: tc_id,
                                        name: tc_name,
                                        input: serde_json::json!({}),
                                    },
                                };
                                events.push(make_sse_event("content_block_start", &block_start));
                            }

                            self.tool_call_states[tc_index]
                                .accumulated_args
                                .push_str(args);

                            let block_index = self.tool_call_states[tc_index].block_index;
                            let block_delta = AnthropicStreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: AnthropicDelta::InputJsonDelta {
                                    partial_json: args.clone(),
                                },
                            };
                            events.push(make_sse_event("content_block_delta", &block_delta));
                        }
                    }
                }
            }
        }

        events
    }

    /// Emit the final events when the stream ends.
    pub fn emit_end_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();

        // Close text block if started and not already closed mid-stream
        if self.text_block_started && !self.text_block_closed {
            let block_stop = AnthropicStreamEvent::ContentBlockStop {
                index: self.text_block_index,
            };
            events.push(make_sse_event("content_block_stop", &block_stop));
        }

        // Close tool call blocks
        for tc in &self.tool_call_states {
            if tc.started {
                let block_stop = AnthropicStreamEvent::ContentBlockStop {
                    index: tc.block_index,
                };
                events.push(make_sse_event("content_block_stop", &block_stop));
            }
        }

        // Emit message_delta with stop_reason and real token usage from engine
        let message_delta = AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: self.stop_reason.clone(),
                stop_sequence: None,
            },
            usage: AnthropicUsage {
                input_tokens: self.input_token_count,
                output_tokens: self.output_token_count,
            },
        };
        events.push(make_sse_event("message_delta", &message_delta));

        // Emit message_stop
        let message_stop = AnthropicStreamEvent::MessageStop {};
        events.push(make_sse_event("message_stop", &message_stop));

        events
    }

    /// Emit error events when the stream ends due to a backend error.
    pub fn emit_error_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let error_event = AnthropicStreamEvent::Error {
            error: AnthropicErrorBody {
                error_type: "api_error".to_string(),
                message: "An internal error occurred during generation.".to_string(),
            },
        };
        vec![make_sse_event("error", &error_event)]
    }
}

fn make_sse_event(event_type: &str, event: &AnthropicStreamEvent) -> Result<Event, anyhow::Error> {
    let data = serde_json::to_string(event)?;
    Ok(Event::default().event(event_type).data(data))
}

/// A tagged event for testing: the event type string paired with the
/// serialized stream event. This avoids needing to parse `axum::sse::Event`
/// (which doesn't implement `Display`).
#[cfg(test)]
#[derive(Debug)]
struct TaggedEvent {
    event_type: String,
    data: AnthropicStreamEvent,
}

#[cfg(test)]
fn make_tagged_event(event_type: &str, event: &AnthropicStreamEvent) -> TaggedEvent {
    TaggedEvent {
        event_type: event_type.to_string(),
        data: event.clone(),
    }
}

#[cfg(test)]
impl AnthropicStreamConverter {
    /// Like `process_chunk` but returns tagged events for test assertions.
    fn process_chunk_tagged(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<TaggedEvent> {
        let mut events = Vec::new();

        if let Some(usage) = &chunk.usage {
            self.input_token_count = usage.prompt_tokens;
            self.output_token_count = usage.completion_tokens;
        }

        for choice in &chunk.choices {
            let delta = &choice.delta;

            if let Some(ref fr) = choice.finish_reason {
                self.stop_reason = Some(match fr {
                    dynamo_async_openai::types::FinishReason::Stop => AnthropicStopReason::EndTurn,
                    dynamo_async_openai::types::FinishReason::Length => {
                        AnthropicStopReason::MaxTokens
                    }
                    dynamo_async_openai::types::FinishReason::ToolCalls => {
                        AnthropicStopReason::ToolUse
                    }
                    dynamo_async_openai::types::FinishReason::ContentFilter => {
                        AnthropicStopReason::EndTurn
                    }
                    dynamo_async_openai::types::FinishReason::FunctionCall => {
                        AnthropicStopReason::ToolUse
                    }
                });
            }

            let content_text = match &delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            };

            if let Some(text) = content_text
                && !text.is_empty()
            {
                if !self.text_block_started {
                    self.text_block_started = true;
                    self.text_block_index = self.next_block_index;
                    self.next_block_index += 1;

                    let ev = AnthropicStreamEvent::ContentBlockStart {
                        index: self.text_block_index,
                        content_block: AnthropicResponseContentBlock::Text {
                            text: String::new(),
                        },
                    };
                    events.push(make_tagged_event("content_block_start", &ev));
                }

                self.output_token_count += 1;
                let ev = AnthropicStreamEvent::ContentBlockDelta {
                    index: self.text_block_index,
                    delta: AnthropicDelta::TextDelta {
                        text: text.to_string(),
                    },
                };
                events.push(make_tagged_event("content_block_delta", &ev));
            }

            if let Some(tool_calls) = &delta.tool_calls {
                if self.text_block_started && !self.text_block_closed {
                    self.text_block_closed = true;
                    let ev = AnthropicStreamEvent::ContentBlockStop {
                        index: self.text_block_index,
                    };
                    events.push(make_tagged_event("content_block_stop", &ev));
                }

                for tc in tool_calls {
                    let tc_index = tc.index as usize;
                    while self.tool_call_states.len() <= tc_index {
                        let block_index = self.next_block_index;
                        self.next_block_index += 1;
                        self.tool_call_states.push(ToolCallState {
                            id: String::new(),
                            name: String::new(),
                            accumulated_args: String::new(),
                            block_index,
                            started: false,
                        });
                    }
                    if let Some(id) = &tc.id {
                        self.tool_call_states[tc_index].id = id.clone();
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            self.tool_call_states[tc_index].name = name.clone();
                        }
                        if let Some(args) = &func.arguments {
                            if !self.tool_call_states[tc_index].started {
                                let tc_id = self.tool_call_states[tc_index].id.clone();
                                if !tc_id.is_empty() && self.tool_calls_sent.contains(&tc_id) {
                                    continue;
                                }
                                self.tool_call_states[tc_index].started = true;
                                let block_index = self.tool_call_states[tc_index].block_index;
                                let tc_name = self.tool_call_states[tc_index].name.clone();
                                if !tc_id.is_empty() {
                                    self.tool_calls_sent.insert(tc_id.clone());
                                }
                                let ev = AnthropicStreamEvent::ContentBlockStart {
                                    index: block_index,
                                    content_block: AnthropicResponseContentBlock::ToolUse {
                                        id: tc_id,
                                        name: tc_name,
                                        input: serde_json::json!({}),
                                    },
                                };
                                events.push(make_tagged_event("content_block_start", &ev));
                            }
                            self.tool_call_states[tc_index]
                                .accumulated_args
                                .push_str(args);
                            let block_index = self.tool_call_states[tc_index].block_index;
                            let ev = AnthropicStreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: AnthropicDelta::InputJsonDelta {
                                    partial_json: args.clone(),
                                },
                            };
                            events.push(make_tagged_event("content_block_delta", &ev));
                        }
                    }
                }
            }
        }

        events
    }

    /// Like `emit_end_events` but returns tagged events for test assertions.
    fn emit_end_events_tagged(&mut self) -> Vec<TaggedEvent> {
        let mut events = Vec::new();

        if self.text_block_started && !self.text_block_closed {
            let ev = AnthropicStreamEvent::ContentBlockStop {
                index: self.text_block_index,
            };
            events.push(make_tagged_event("content_block_stop", &ev));
        }

        for tc in &self.tool_call_states {
            if tc.started {
                let ev = AnthropicStreamEvent::ContentBlockStop {
                    index: tc.block_index,
                };
                events.push(make_tagged_event("content_block_stop", &ev));
            }
        }

        let ev = AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: self.stop_reason.clone(),
                stop_sequence: None,
            },
            usage: AnthropicUsage {
                input_tokens: self.input_token_count,
                output_tokens: self.output_token_count,
            },
        };
        events.push(make_tagged_event("message_delta", &ev));

        let ev = AnthropicStreamEvent::MessageStop {};
        events.push(make_tagged_event("message_stop", &ev));

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_async_openai::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionStreamResponseDelta, ChatCompletionToolType, FunctionCallStream,
    };

    fn text_chunk(text: &str) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            id: "chat-1".into(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some(ChatCompletionMessageContent::Text(text.into())),
                    function_call: None,
                    tool_calls: None,
                    role: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                stop_reason: None,
                logprobs: None,
            }],
            created: 0,
            model: "test".into(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".into(),
            usage: None,
            nvext: None,
        }
    }

    fn tool_call_chunk(
        tc_index: u32,
        id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            id: "chat-1".into(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: None,
                    function_call: None,
                    tool_calls: Some(vec![ChatCompletionMessageToolCallChunk {
                        index: tc_index,
                        id: id.map(String::from),
                        r#type: Some(ChatCompletionToolType::Function),
                        function: Some(FunctionCallStream {
                            name: name.map(String::from),
                            arguments: args.map(String::from),
                        }),
                    }]),
                    role: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                stop_reason: None,
                logprobs: None,
            }],
            created: 0,
            model: "test".into(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".into(),
            usage: None,
            nvext: None,
        }
    }

    fn event_types(events: &[TaggedEvent]) -> Vec<&str> {
        events.iter().map(|e| e.event_type.as_str()).collect()
    }

    /// Regression test: text block must be closed (content_block_stop)
    /// before the tool_use block starts (content_block_start).
    ///
    /// Without this fix, the text block stop was batched at the end,
    /// causing Claude Code's streaming parser to receive out-of-order
    /// events and fail to execute tool calls ("Error editing file").
    #[test]
    fn test_text_block_stops_before_tool_block_starts() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        // Stream some text
        let text_events = conv.process_chunk_tagged(&text_chunk("I'll edit the file."));
        assert_eq!(
            event_types(&text_events),
            vec!["content_block_start", "content_block_delta"]
        );

        // Stream a tool call — text block must close first
        let tool_events = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Edit"),
            Some("{\"file_path\":\"/tmp/test.txt\"}"),
        ));

        assert_eq!(
            event_types(&tool_events),
            vec![
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ],
            "text block must be closed before tool block starts"
        );

        // Verify indices: stop=0 (text), start=1 (tool)
        match &tool_events[0].data {
            AnthropicStreamEvent::ContentBlockStop { index } => assert_eq!(*index, 0),
            other => panic!("expected ContentBlockStop, got {other:?}"),
        }
        match &tool_events[1].data {
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(*index, 1);
                match content_block {
                    AnthropicResponseContentBlock::ToolUse { name, .. } => {
                        assert_eq!(name, "Edit");
                    }
                    other => panic!("expected ToolUse, got {other:?}"),
                }
            }
            other => panic!("expected ContentBlockStart, got {other:?}"),
        }

        // End events should NOT duplicate the text block stop
        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["content_block_stop", "message_delta", "message_stop"],
            "only tool block stop in end events (text already closed)"
        );
        match &end_events[0].data {
            AnthropicStreamEvent::ContentBlockStop { index } => assert_eq!(*index, 1),
            other => panic!("expected tool stop at index 1, got {other:?}"),
        }
    }

    /// Tool-only response (no preceding text): no spurious stop events.
    #[test]
    fn test_tool_only_response_no_text_block() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        let tool_events = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/test.txt\"}"),
        ));
        assert_eq!(
            event_types(&tool_events),
            vec!["content_block_start", "content_block_delta"]
        );

        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
    }

    /// Text-only response: stop emitted in end events (no early close).
    #[test]
    fn test_text_only_response_stop_in_end_events() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        conv.process_chunk_tagged(&text_chunk("Hello world"));

        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
        match &end_events[0].data {
            AnthropicStreamEvent::ContentBlockStop { index } => assert_eq!(*index, 0),
            other => panic!("expected text stop at index 0, got {other:?}"),
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Anthropic Messages API SSE events.
//!
//! The event sequence follows the Anthropic streaming spec:
//! `message_start` -> `content_block_start` -> N x `content_block_delta` ->
//! `content_block_stop` -> `message_delta` -> `message_stop`

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
    text_block_index: u32,
    output_token_count: u32,
    // Tool call tracking
    tool_call_states: Vec<ToolCallState>,
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
            text_block_index: 0,
            output_token_count: 0,
            tool_call_states: Vec::new(),
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
                self.output_token_count += 1; // Rough estimate
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
                                self.tool_call_states[tc_index].started = true;
                                let block_index = self.tool_call_states[tc_index].block_index;
                                let tc_id = self.tool_call_states[tc_index].id.clone();
                                let tc_name = self.tool_call_states[tc_index].name.clone();

                                let block_start = AnthropicStreamEvent::ContentBlockStart {
                                    index: block_index,
                                    content_block: AnthropicResponseContentBlock::ToolUse {
                                        id: tc_id,
                                        name: tc_name,
                                        input: serde_json::json!({}),
                                    },
                                };
                                events
                                    .push(make_sse_event("content_block_start", &block_start));
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

        // Close text block if started
        if self.text_block_started {
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

        // Emit message_delta with stop_reason
        let message_delta = AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: self.stop_reason.clone(),
                stop_sequence: None,
            },
            usage: AnthropicUsage {
                input_tokens: 0,
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

fn make_sse_event(
    event_type: &str,
    event: &AnthropicStreamEvent,
) -> Result<Event, anyhow::Error> {
    let data = serde_json::to_string(event)?;
    Ok(Event::default().event(event_type).data(data))
}

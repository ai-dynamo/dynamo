// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Anthropic Messages API SSE events.
//!
//! The event sequence follows the Anthropic streaming spec:
//! `message_start` -> `content_block_start` -> N x `content_block_delta` ->
//! `content_block_stop` -> `message_delta` -> `message_stop`

use std::collections::HashSet;

use axum::response::sse::Event;
use dynamo_protocols::types::ChatCompletionMessageContent;
use uuid::Uuid;

use super::types::{
    AnthropicDelta, AnthropicErrorBody, AnthropicMessageDeltaBody, AnthropicMessageResponse,
    AnthropicResponseContentBlock, AnthropicStopReason, AnthropicStreamEvent, AnthropicUsage,
};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::protocols::unified::AnthropicContext;

/// State machine that converts a chat completion stream into Anthropic SSE events.
pub struct AnthropicStreamConverter {
    model: String,
    message_id: String,
    /// Preserved Anthropic-specific request context for faithful response reconstruction.
    api_context: Option<AnthropicContext>,
    // Thinking/reasoning tracking
    thinking_block_started: bool,
    thinking_block_closed: bool,
    thinking_block_index: u32,
    // Text tracking
    text_block_started: bool,
    text_block_closed: bool,
    text_block_index: u32,
    // Token usage (from engine)
    input_token_count: u32,
    output_token_count: u32,
    cached_token_count: Option<u32>,
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
    /// Set when `content_block_stop` has already been emitted inline
    /// (complete tool call detected mid-stream). Prevents duplicate stop in `emit_end_events()`.
    stopped: bool,
}

impl AnthropicStreamConverter {
    pub fn new(model: String) -> Self {
        Self {
            model,
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            api_context: None,
            thinking_block_started: false,
            thinking_block_closed: false,
            thinking_block_index: 0,
            text_block_started: false,
            text_block_closed: false,
            text_block_index: 0,
            input_token_count: 0,
            output_token_count: 0,
            cached_token_count: None,
            tool_call_states: Vec::new(),
            tool_calls_sent: HashSet::new(),
            next_block_index: 0,
            stop_reason: None,
        }
    }

    /// Create a converter seeded with the original Anthropic request context.
    /// This allows the response stream to carry forward metadata that was lost
    /// during the Anthropic-to-OpenAI request conversion.
    pub fn with_context(model: String, context: AnthropicContext) -> Self {
        let mut converter = Self::new(model);
        converter.api_context = Some(context);
        converter
    }

    /// Emit the initial `message_start` event.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        // TODO: When AnthropicMessageResponse gains a `service_tier` field,
        // populate it from `self.api_context` (if the original request specified one).
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
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
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
        if let Some(usage) = &chunk.inner.usage {
            self.input_token_count = usage.prompt_tokens;
            self.output_token_count = usage.completion_tokens;
            self.cached_token_count = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens);
        }

        for choice in &chunk.inner.choices {
            let delta = &choice.delta;

            // Track finish reason
            if let Some(ref fr) = choice.finish_reason {
                self.stop_reason = Some(match fr {
                    dynamo_protocols::types::FinishReason::Stop => AnthropicStopReason::EndTurn,
                    dynamo_protocols::types::FinishReason::Length => AnthropicStopReason::MaxTokens,
                    dynamo_protocols::types::FinishReason::ToolCalls => {
                        AnthropicStopReason::ToolUse
                    }
                    dynamo_protocols::types::FinishReason::ContentFilter => {
                        AnthropicStopReason::EndTurn
                    }
                    dynamo_protocols::types::FinishReason::FunctionCall => {
                        AnthropicStopReason::ToolUse
                    }
                });
            }

            // Handle reasoning/thinking content deltas
            if let Some(ref reasoning) = delta.reasoning_content
                && !reasoning.is_empty()
            {
                // Emit content_block_start on first thinking token
                if !self.thinking_block_started {
                    self.thinking_block_started = true;
                    self.thinking_block_index = self.next_block_index;
                    self.next_block_index += 1;

                    let block_start = AnthropicStreamEvent::ContentBlockStart {
                        index: self.thinking_block_index,
                        content_block: AnthropicResponseContentBlock::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                    };
                    events.push(make_sse_event("content_block_start", &block_start));
                }

                // Emit thinking delta
                let block_delta = AnthropicStreamEvent::ContentBlockDelta {
                    index: self.thinking_block_index,
                    delta: AnthropicDelta::ThinkingDelta {
                        thinking: reasoning.clone(),
                    },
                };
                events.push(make_sse_event("content_block_delta", &block_delta));
            }

            // Handle text content deltas
            let content_text = match &delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            };

            if let Some(text) = content_text
                && !text.is_empty()
            {
                // Close thinking block before text starts (Anthropic spec: thinking → text → tool_use)
                if self.thinking_block_started && !self.thinking_block_closed {
                    self.thinking_block_closed = true;
                    // Emit signature delta to close the thinking block.
                    // The engine doesn't produce Anthropic-style cryptographic signatures,
                    // so we use "erased" (the standard placeholder per the Anthropic spec).
                    // When `api_context` is available and the original request had
                    // `thinking.thinking_type == "enabled"`, this is expected — the backend
                    // simply doesn't generate real signatures. If/when the backend starts
                    // returning real signatures, we can use the context to validate or
                    // pass them through instead of hardcoding "erased".
                    let sig_delta = AnthropicStreamEvent::ContentBlockDelta {
                        index: self.thinking_block_index,
                        delta: AnthropicDelta::SignatureDelta {
                            signature: "erased".to_string(),
                        },
                    };
                    events.push(make_sse_event("content_block_delta", &sig_delta));

                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(make_sse_event("content_block_stop", &block_stop));
                }

                // Emit content_block_start on first text
                if !self.text_block_started {
                    self.text_block_started = true;
                    self.text_block_index = self.next_block_index;
                    self.next_block_index += 1;

                    let block_start = AnthropicStreamEvent::ContentBlockStart {
                        index: self.text_block_index,
                        content_block: AnthropicResponseContentBlock::Text {
                            text: String::new(),
                            citations: None,
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
                // Close thinking block before tool blocks (if text never appeared)
                if self.thinking_block_started && !self.thinking_block_closed {
                    self.thinking_block_closed = true;
                    let sig_delta = AnthropicStreamEvent::ContentBlockDelta {
                        index: self.thinking_block_index,
                        delta: AnthropicDelta::SignatureDelta {
                            signature: "erased".to_string(),
                        },
                    };
                    events.push(make_sse_event("content_block_delta", &sig_delta));
                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(make_sse_event("content_block_stop", &block_stop));
                }

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
                            stopped: false,
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

                            // Emit content_block_stop immediately if the tool call arrived
                            // complete in a single chunk (id + name + args all present).
                            // Dynamo backends emit complete tool calls, so this fires on the
                            // same chunk — no need to wait for finish_reason.
                            if tc.id.is_some()
                                && func.name.is_some()
                                && !self.tool_call_states[tc_index].stopped
                            {
                                self.tool_call_states[tc_index].stopped = true;
                                let block_stop =
                                    AnthropicStreamEvent::ContentBlockStop { index: block_index };
                                events.push(make_sse_event("content_block_stop", &block_stop));
                            }
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

        // Close thinking block if started and not already closed mid-stream
        if self.thinking_block_started && !self.thinking_block_closed {
            self.thinking_block_closed = true;
            let sig_delta = AnthropicStreamEvent::ContentBlockDelta {
                index: self.thinking_block_index,
                delta: AnthropicDelta::SignatureDelta {
                    signature: "erased".to_string(),
                },
            };
            events.push(make_sse_event("content_block_delta", &sig_delta));
            let block_stop = AnthropicStreamEvent::ContentBlockStop {
                index: self.thinking_block_index,
            };
            events.push(make_sse_event("content_block_stop", &block_stop));
        }

        // Close text block if started and not already closed mid-stream
        if self.text_block_started && !self.text_block_closed {
            let block_stop = AnthropicStreamEvent::ContentBlockStop {
                index: self.text_block_index,
            };
            events.push(make_sse_event("content_block_stop", &block_stop));
        }

        // Close tool call blocks (skip any already stopped inline)
        for tc in &self.tool_call_states {
            if tc.started && !tc.stopped {
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
                cache_creation_input_tokens: None,
                cache_read_input_tokens: self.cached_token_count,
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

        if let Some(usage) = &chunk.inner.usage {
            self.input_token_count = usage.prompt_tokens;
            self.output_token_count = usage.completion_tokens;
            self.cached_token_count = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens);
        }

        for choice in &chunk.inner.choices {
            let delta = &choice.delta;

            if let Some(ref fr) = choice.finish_reason {
                self.stop_reason = Some(match fr {
                    dynamo_protocols::types::FinishReason::Stop => AnthropicStopReason::EndTurn,
                    dynamo_protocols::types::FinishReason::Length => AnthropicStopReason::MaxTokens,
                    dynamo_protocols::types::FinishReason::ToolCalls => {
                        AnthropicStopReason::ToolUse
                    }
                    dynamo_protocols::types::FinishReason::ContentFilter => {
                        AnthropicStopReason::EndTurn
                    }
                    dynamo_protocols::types::FinishReason::FunctionCall => {
                        AnthropicStopReason::ToolUse
                    }
                });
            }

            // Handle reasoning/thinking content deltas
            if let Some(ref reasoning) = delta.reasoning_content
                && !reasoning.is_empty()
            {
                if !self.thinking_block_started {
                    self.thinking_block_started = true;
                    self.thinking_block_index = self.next_block_index;
                    self.next_block_index += 1;

                    let ev = AnthropicStreamEvent::ContentBlockStart {
                        index: self.thinking_block_index,
                        content_block: AnthropicResponseContentBlock::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                    };
                    events.push(make_tagged_event("content_block_start", &ev));
                }

                let ev = AnthropicStreamEvent::ContentBlockDelta {
                    index: self.thinking_block_index,
                    delta: AnthropicDelta::ThinkingDelta {
                        thinking: reasoning.clone(),
                    },
                };
                events.push(make_tagged_event("content_block_delta", &ev));
            }

            let content_text = match &delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            };

            if let Some(text) = content_text
                && !text.is_empty()
            {
                // Close thinking block before text starts
                if self.thinking_block_started && !self.thinking_block_closed {
                    self.thinking_block_closed = true;
                    let ev = AnthropicStreamEvent::ContentBlockDelta {
                        index: self.thinking_block_index,
                        delta: AnthropicDelta::SignatureDelta {
                            signature: "erased".to_string(),
                        },
                    };
                    events.push(make_tagged_event("content_block_delta", &ev));
                    let ev = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(make_tagged_event("content_block_stop", &ev));
                }

                if !self.text_block_started {
                    self.text_block_started = true;
                    self.text_block_index = self.next_block_index;
                    self.next_block_index += 1;

                    let ev = AnthropicStreamEvent::ContentBlockStart {
                        index: self.text_block_index,
                        content_block: AnthropicResponseContentBlock::Text {
                            text: String::new(),
                            citations: None,
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
                // Close thinking block before tool blocks
                if self.thinking_block_started && !self.thinking_block_closed {
                    self.thinking_block_closed = true;
                    let ev = AnthropicStreamEvent::ContentBlockDelta {
                        index: self.thinking_block_index,
                        delta: AnthropicDelta::SignatureDelta {
                            signature: "erased".to_string(),
                        },
                    };
                    events.push(make_tagged_event("content_block_delta", &ev));
                    let ev = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(make_tagged_event("content_block_stop", &ev));
                }

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
                            stopped: false,
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

                            // Emit content_block_stop immediately if the tool call arrived
                            // complete in a single chunk (id + name + args all present).
                            // Dynamo backends emit complete tool calls, so this fires on the
                            // same chunk — no need to wait for finish_reason.
                            if tc.id.is_some()
                                && func.name.is_some()
                                && !self.tool_call_states[tc_index].stopped
                            {
                                self.tool_call_states[tc_index].stopped = true;
                                let ev =
                                    AnthropicStreamEvent::ContentBlockStop { index: block_index };
                                events.push(make_tagged_event("content_block_stop", &ev));
                            }
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

        // Close thinking block if not already closed
        if self.thinking_block_started && !self.thinking_block_closed {
            self.thinking_block_closed = true;
            let ev = AnthropicStreamEvent::ContentBlockDelta {
                index: self.thinking_block_index,
                delta: AnthropicDelta::SignatureDelta {
                    signature: "erased".to_string(),
                },
            };
            events.push(make_tagged_event("content_block_delta", &ev));
            let ev = AnthropicStreamEvent::ContentBlockStop {
                index: self.thinking_block_index,
            };
            events.push(make_tagged_event("content_block_stop", &ev));
        }

        if self.text_block_started && !self.text_block_closed {
            let ev = AnthropicStreamEvent::ContentBlockStop {
                index: self.text_block_index,
            };
            events.push(make_tagged_event("content_block_stop", &ev));
        }

        // Skip already-stopped tool call blocks
        for tc in &self.tool_call_states {
            if tc.started && !tc.stopped {
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
                cache_creation_input_tokens: None,
                cache_read_input_tokens: self.cached_token_count,
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
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionStreamResponseDelta, ChatCompletionToolType, FunctionCallStream,
    };

    fn text_chunk(text: &str) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
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
            },
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
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
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
            },
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
                "content_block_delta",
                "content_block_stop",
            ],
            "text block must be closed before tool block starts; complete tool call stopped inline"
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

        // End events should NOT duplicate either stop (both already emitted inline)
        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["message_delta", "message_stop"],
            "no block stops in end events (both text and tool already closed inline)"
        );
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
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ],
            "complete tool call emits stop inline"
        );

        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["message_delta", "message_stop"],
            "no block stop in end events (already stopped inline)"
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

    fn reasoning_chunk(text: &str) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chat-1".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: None,
                        function_call: None,
                        tool_calls: None,
                        role: None,
                        refusal: None,
                        reasoning_content: Some(text.into()),
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
            },
            nvext: None,
        }
    }

    /// Full reasoning flow: thinking → text → tool_use.
    /// Verifies block ordering (thinking=0, text=1, tool=2) and that each
    /// block is properly closed before the next one starts.
    #[test]
    fn test_thinking_text_then_tool_call() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        // 1. Reasoning tokens → thinking block starts
        let ev = conv.process_chunk_tagged(&reasoning_chunk("Let me think..."));
        assert_eq!(
            event_types(&ev),
            vec!["content_block_start", "content_block_delta"]
        );
        assert!(matches!(
            &ev[0].data,
            AnthropicStreamEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicResponseContentBlock::Thinking { .. }
            }
        ));

        // 2. Text arrives → thinking block closes (signature + stop), text block opens
        let ev = conv.process_chunk_tagged(&text_chunk("Hello!"));
        assert_eq!(
            event_types(&ev),
            vec![
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ]
        );
        assert!(matches!(
            &ev[1].data,
            AnthropicStreamEvent::ContentBlockStop { index: 0 }
        ));
        assert!(matches!(
            &ev[2].data,
            AnthropicStreamEvent::ContentBlockStart { index: 1, .. }
        ));

        // 3. Tool call → text block closes, tool block opens at index 2.
        //    Because the tool call arrives complete (id + name + args in one
        //    chunk), inline dispatch also emits content_block_stop immediately.
        let ev = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/test.txt\"}"),
        ));
        assert_eq!(
            event_types(&ev),
            vec![
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ]
        );
        assert!(matches!(
            &ev[0].data,
            AnthropicStreamEvent::ContentBlockStop { index: 1 }
        ));
        assert!(matches!(
            &ev[1].data,
            AnthropicStreamEvent::ContentBlockStart { index: 2, .. }
        ));
    }

    /// Thinking-only response (no text/tool follows): thinking block closed in end events.
    #[test]
    fn test_thinking_only_closed_in_end_events() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        conv.process_chunk_tagged(&reasoning_chunk("Deep thought..."));

        let ev = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&ev),
            vec![
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
    }

    /// Multiple tool calls: each gets inline content_block_stop.
    #[test]
    fn test_multiple_tool_calls_each_stopped_inline() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        let events1 = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/a.txt\"}"),
        ));
        assert_eq!(
            event_types(&events1),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ],
            "first tool call closed inline"
        );

        let events2 = conv.process_chunk_tagged(&tool_call_chunk(
            1,
            Some("call-2"),
            Some("Write"),
            Some("{\"path\":\"/tmp/b.txt\"}"),
        ));
        assert_eq!(
            event_types(&events2),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ],
            "second tool call closed inline"
        );

        // End events: no block stops (both already closed)
        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["message_delta", "message_stop"],
            "no block stops in end events"
        );
    }

    /// Verify that `with_context` stores the context and produces the same
    /// event structure as `new` — the context is carried for future enrichment.
    #[test]
    fn test_with_context_preserves_context() {
        use crate::protocols::unified::AnthropicContext;

        let ctx = AnthropicContext {
            service_tier: Some("priority".to_string()),
            ..Default::default()
        };
        let mut conv = AnthropicStreamConverter::with_context("test-model".into(), ctx);
        assert!(conv.api_context.is_some());
        assert_eq!(
            conv.api_context.as_ref().unwrap().service_tier.as_deref(),
            Some("priority")
        );

        // Should produce the same events as a regular converter
        let ev = conv.process_chunk_tagged(&text_chunk("Hello"));
        assert_eq!(
            event_types(&ev),
            vec!["content_block_start", "content_block_delta"]
        );

        let end = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end),
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
    }
}

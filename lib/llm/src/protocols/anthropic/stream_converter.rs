// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Anthropic Messages API SSE events.
//!
//! The event sequence follows the Anthropic streaming spec:
//! `message_start` -> `content_block_start` -> N x `content_block_delta` ->
//! `content_block_stop` -> `message_delta` -> `message_stop`

use std::collections::HashSet;

use axum::response::sse::Event;
use dynamo_protocols::types::{ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk};
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
    tool_blocks_flushed: bool,
    // Block index counter
    next_block_index: u32,
    // Stop reason
    stop_reason: Option<AnthropicStopReason>,
}

struct ToolCallState {
    id: String,
    name: String,
    argument_fragments: Vec<String>,
}

impl ToolCallState {
    /// A tool block is ready to flush once both required identity fields are
    /// present. Arguments are optional: a tool call with no parameters still
    /// emits, so `argument_fragments` is deliberately not part of this check.
    fn is_emit_ready(&self) -> bool {
        !self.id.is_empty() && !self.name.is_empty()
    }
}

impl AnthropicStreamConverter {
    pub fn new(model: String, estimated_input_tokens: u32) -> Self {
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
            input_token_count: estimated_input_tokens,
            output_token_count: 0,
            cached_token_count: None,
            tool_call_states: Vec::new(),
            tool_blocks_flushed: false,
            next_block_index: 0,
            stop_reason: None,
        }
    }

    /// Create a converter seeded with the original Anthropic request context.
    /// This allows the response stream to carry forward metadata that was lost
    /// during the Anthropic-to-OpenAI request conversion.
    pub fn with_context(
        model: String,
        estimated_input_tokens: u32,
        context: AnthropicContext,
    ) -> Self {
        let mut converter = Self::new(model, estimated_input_tokens);
        converter.api_context = Some(context);
        converter
    }

    /// Accumulate one streamed tool-call chunk into per-index state.
    ///
    /// Two distinct orderings matter here, and only the first is something a
    /// current backend actually produces:
    ///
    /// - Within a single call, the id/name and the argument fragments may arrive
    ///   in either order — arguments can begin before the chunk carrying the id
    ///   and name. We therefore record whichever fields are present on each chunk
    ///   and defer emitting the block until the identity is complete (see
    ///   `is_emit_ready`). This is the case the fixtures exercise.
    /// - Across parallel calls, the in-tree `dynamo-parsers-v2` parsers emit one
    ///   call at a time with a monotonically increasing `index` (call 0's chunks
    ///   all precede call 1's), so indices are never interleaved today. Indexing
    ///   `tool_call_states` by `tool_call.index` keeps each call's state separate
    ///   regardless, so interleaved indices would also be handled — but no current
    ///   parser emits them, so that path is defensive rather than exercised.
    fn record_tool_call(&mut self, tool_call: &ChatCompletionMessageToolCallChunk) {
        let tool_call_index = tool_call.index as usize;
        while self.tool_call_states.len() <= tool_call_index {
            self.tool_call_states.push(ToolCallState {
                id: String::new(),
                name: String::new(),
                argument_fragments: Vec::new(),
            });
        }

        let state = &mut self.tool_call_states[tool_call_index];
        if let Some(id) = &tool_call.id {
            state.id = id.clone();
        }
        if let Some(function) = &tool_call.function {
            if let Some(name) = &function.name {
                state.name = name.clone();
            }
            if let Some(arguments) = &function.arguments {
                state.argument_fragments.push(arguments.clone());
            }
        }
    }

    fn drain_buffered_tool_events(&mut self) -> Vec<(&'static str, AnthropicStreamEvent)> {
        if self.tool_blocks_flushed {
            return Vec::new();
        }
        self.tool_blocks_flushed = true;

        let mut events = Vec::new();
        let mut sent_tool_call_ids = HashSet::new();
        let mut block_index = self.next_block_index;

        for tool_call in &self.tool_call_states {
            if !tool_call.is_emit_ready() || !sent_tool_call_ids.insert(tool_call.id.clone()) {
                continue;
            }

            events.push((
                "content_block_start",
                AnthropicStreamEvent::ContentBlockStart {
                    index: block_index,
                    content_block: AnthropicResponseContentBlock::ToolUse {
                        id: tool_call.id.clone(),
                        name: tool_call.name.clone(),
                        input: serde_json::json!({}),
                    },
                },
            ));

            for arguments in &tool_call.argument_fragments {
                events.push((
                    "content_block_delta",
                    AnthropicStreamEvent::ContentBlockDelta {
                        index: block_index,
                        delta: AnthropicDelta::InputJsonDelta {
                            partial_json: arguments.clone(),
                        },
                    },
                ));
            }

            events.push((
                "content_block_stop",
                AnthropicStreamEvent::ContentBlockStop { index: block_index },
            ));
            block_index += 1;
        }

        self.next_block_index = block_index;
        events
    }

    fn append_buffered_tool_events(&mut self, events: &mut Vec<Result<Event, anyhow::Error>>) {
        for (event_type, event) in self.drain_buffered_tool_events() {
            events.push(make_sse_event(event_type, &event));
        }
    }

    /// Emit the initial `message_start` event.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::with_capacity(1);
        self.append_start_events(&mut events);
        events
    }

    /// Append the initial `message_start` event.
    pub fn append_start_events(&mut self, events: &mut Vec<Result<Event, anyhow::Error>>) {
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
                input_tokens: self.input_token_count,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let event = AnthropicStreamEvent::MessageStart { message };
        events.push(make_sse_event("message_start", &event));
    }

    /// Process a single chat completion stream chunk and return zero or more SSE events.
    pub fn process_chunk(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();
        self.append_chunk_events(chunk, &mut events);
        events
    }

    /// Process a single chat completion stream chunk and append zero or more SSE events.
    pub fn append_chunk_events(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
        events: &mut Vec<Result<Event, anyhow::Error>>,
    ) {
        // Capture token usage from engine when available (typically on the final chunk).
        // Only update output_token_count — input_token_count is set once from the
        // estimate in new() and must stay consistent between message_start and
        // message_delta to avoid Claude Code's token display jumping.
        if let Some(usage) = &chunk.inner.usage {
            self.output_token_count = usage.completion_tokens;
            self.cached_token_count = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens);
        }

        let mut should_flush_tool_blocks = false;
        for choice in &chunk.inner.choices {
            let delta = &choice.delta;

            // Track finish reason
            if let Some(ref fr) = choice.finish_reason {
                should_flush_tool_blocks |= matches!(
                    fr,
                    dynamo_protocols::types::FinishReason::ToolCalls
                        | dynamo_protocols::types::FinishReason::FunctionCall
                );
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

                for tool_call in tool_calls {
                    self.record_tool_call(tool_call);
                }
            }
        }

        // A tool-call finish reason is the first explicit guarantee that all
        // argument fragments in this choice are complete. `JailedStream` rewrites
        // `Stop` to `ToolCalls` after emitting tool-call chunks; interrupted
        // `Length`/`ContentFilter` streams use the EOF fallback below. Flush only
        // after every choice and delta in the terminal chunk has been recorded.
        if should_flush_tool_blocks {
            self.append_buffered_tool_events(events);
        }
    }

    /// Emit the final events when the stream ends.
    pub fn emit_end_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();
        self.append_end_events(&mut events);
        events
    }

    /// Append the final events when the stream ends.
    pub fn append_end_events(&mut self, events: &mut Vec<Result<Event, anyhow::Error>>) {
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

        // EOF remains a fallback for backends that omit a terminal finish reason.
        // If a finish chunk already flushed these blocks, this is a no-op.
        self.append_buffered_tool_events(events);

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
    }

    /// Emit error events when the stream ends due to a backend error.
    pub fn emit_error_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::with_capacity(1);
        self.append_error_events(&mut events);
        events
    }

    /// Append error events when the stream ends due to a backend error.
    pub fn append_error_events(&mut self, events: &mut Vec<Result<Event, anyhow::Error>>) {
        let error_event = AnthropicStreamEvent::Error {
            error: AnthropicErrorBody {
                error_type: "api_error".to_string(),
                message: "An internal error occurred during generation.".to_string(),
            },
        };
        events.push(make_sse_event("error", &error_event));
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
            self.output_token_count = usage.completion_tokens;
            self.cached_token_count = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens);
        }

        let mut should_flush_tool_blocks = false;
        for choice in &chunk.inner.choices {
            let delta = &choice.delta;

            if let Some(ref fr) = choice.finish_reason {
                should_flush_tool_blocks |= matches!(
                    fr,
                    dynamo_protocols::types::FinishReason::ToolCalls
                        | dynamo_protocols::types::FinishReason::FunctionCall
                );
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

                for tool_call in tool_calls {
                    self.record_tool_call(tool_call);
                }
            }
        }

        // Keep this test path aligned with `process_chunk`: normal tool-call
        // streams carry a tool-call finish reason, while interrupted streams use
        // the EOF fallback in `emit_end_events_tagged`.
        if should_flush_tool_blocks {
            for (event_type, event) in self.drain_buffered_tool_events() {
                events.push(make_tagged_event(event_type, &event));
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

        // EOF fallback; a finish-triggered drain leaves no events here.
        for (event_type, event) in self.drain_buffered_tool_events() {
            events.push(make_tagged_event(event_type, &event));
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
        ChatCompletionStreamResponseDelta, FinishReason, FunctionCallStream, FunctionType,
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
            llm_metrics: None,
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
                            r#type: Some(FunctionType::Function),
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
            llm_metrics: None,
        }
    }

    fn finish_chunk(finish_reason: FinishReason) -> NvCreateChatCompletionStreamResponse {
        let mut chunk = tool_call_chunk(0, None, None, None);
        chunk.inner.choices[0].delta.tool_calls = None;
        chunk.inner.choices[0].finish_reason = Some(finish_reason);
        chunk
    }

    fn event_types(events: &[TaggedEvent]) -> Vec<&str> {
        events.iter().map(|e| e.event_type.as_str()).collect()
    }

    #[test]
    fn test_append_events_reuse_caller_storage() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);
        let mut events = Vec::with_capacity(8);

        conv.append_start_events(&mut events);
        assert_eq!(events.len(), 1);
        assert!(events.iter().all(Result::is_ok));

        events.clear();
        let capacity = events.capacity();
        conv.append_chunk_events(&text_chunk("I'll edit the file."), &mut events);
        assert_eq!(events.len(), 2);
        assert_eq!(events.capacity(), capacity);
        assert!(events.iter().all(Result::is_ok));

        events.clear();
        conv.append_chunk_events(
            &tool_call_chunk(
                0,
                Some("call-1"),
                Some("Edit"),
                Some("{\"file_path\":\"/tmp/test.txt\"}"),
            ),
            &mut events,
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events.capacity(), capacity);
        assert!(events.iter().all(Result::is_ok));

        events.clear();
        conv.append_chunk_events(&finish_chunk(FinishReason::ToolCalls), &mut events);
        assert_eq!(events.len(), 3);
        assert_eq!(events.capacity(), capacity);
        assert!(events.iter().all(Result::is_ok));

        events.clear();
        conv.append_end_events(&mut events);
        assert_eq!(events.len(), 2);
        assert_eq!(events.capacity(), capacity);
        assert!(events.iter().all(Result::is_ok));

        events.clear();
        conv.append_error_events(&mut events);
        assert_eq!(events.len(), 1);
        assert_eq!(events.capacity(), capacity);
        assert!(events.iter().all(Result::is_ok));
    }

    /// Regression test: text block must be closed (content_block_stop)
    /// before the tool_use block starts (content_block_start).
    ///
    /// Without this fix, the text block stop was batched at the end,
    /// causing Claude Code's streaming parser to receive out-of-order
    /// events and fail to execute tool calls ("Error editing file").
    #[test]
    fn test_text_block_stops_before_tool_block_starts() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

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
            vec!["content_block_stop"],
            "text block must close before buffered tool output"
        );

        // Verify index 0 closes before buffered tool index 1 is emitted.
        match &tool_events[0].data {
            AnthropicStreamEvent::ContentBlockStop { index } => assert_eq!(*index, 0),
            other => panic!("expected ContentBlockStop, got {other:?}"),
        }

        let finish_events = conv.process_chunk_tagged(&finish_chunk(FinishReason::ToolCalls));
        assert_eq!(
            event_types(&finish_events),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ]
        );
        match &finish_events[0].data {
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
        assert_eq!(
            event_types(&conv.emit_end_events_tagged()),
            vec!["message_delta", "message_stop"]
        );
    }

    /// EOF remains a fallback when the backend omits a finish reason.
    #[test]
    fn test_tool_only_response_flushes_at_eof_without_finish_reason() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

        let tool_events = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/test.txt\"}"),
        ));
        assert!(tool_events.is_empty());

        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
    }

    #[test]
    fn test_fragmented_tool_arguments_flush_on_tool_calls_finish() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

        let first =
            conv.process_chunk_tagged(&tool_call_chunk(0, Some("call-1"), Some("Read"), Some("")));
        assert!(first.is_empty());

        let middle =
            conv.process_chunk_tagged(&tool_call_chunk(0, None, None, Some("{\"path\":\"/tmp")));
        assert!(middle.is_empty());

        let last = conv.process_chunk_tagged(&tool_call_chunk(0, None, None, Some("\"}")));
        assert!(last.is_empty());

        let finish = conv.process_chunk_tagged(&finish_chunk(FinishReason::ToolCalls));
        assert_eq!(
            event_types(&finish),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
            ]
        );

        assert_eq!(
            event_types(&conv.emit_end_events_tagged()),
            vec!["message_delta", "message_stop"],
            "EOF must not repeat finish-triggered tool events"
        );
    }

    #[test]
    fn test_id_and_name_only_tool_call_is_emitted() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

        let mut chunk = tool_call_chunk(0, Some("call-1"), Some("Read"), None);
        chunk.inner.choices[0].finish_reason = Some(FinishReason::FunctionCall);

        let finish = conv.process_chunk_tagged(&chunk);
        assert_eq!(
            event_types(&finish),
            vec!["content_block_start", "content_block_stop"]
        );
        assert!(matches!(
            &finish[0].data,
            AnthropicStreamEvent::ContentBlockStart {
                content_block: AnthropicResponseContentBlock::ToolUse { id, name, input },
                ..
            } if id == "call-1" && name == "Read" && input == &serde_json::json!({})
        ));
        assert_eq!(
            event_types(&conv.emit_end_events_tagged()),
            vec!["message_delta", "message_stop"]
        );
    }

    #[test]
    fn test_terminal_chunk_records_arguments_before_flushing() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);
        let mut chunk = tool_call_chunk(0, Some("call-1"), Some("Read"), Some("{}"));
        chunk.inner.choices[0].finish_reason = Some(FinishReason::ToolCalls);

        let finish = conv.process_chunk_tagged(&chunk);
        assert_eq!(
            event_types(&finish),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ]
        );
        assert!(matches!(
            &finish[1].data,
            AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicDelta::InputJsonDelta { partial_json },
                ..
            } if partial_json == "{}"
        ));
    }

    #[test]
    fn test_incomplete_tool_call_identity_is_not_emitted() {
        for chunk in [
            tool_call_chunk(0, Some("call-1"), None, None),
            tool_call_chunk(0, None, Some("Read"), None),
            tool_call_chunk(0, None, None, Some("{}")),
        ] {
            let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);
            assert!(conv.process_chunk_tagged(&chunk).is_empty());
            assert!(
                conv.process_chunk_tagged(&finish_chunk(FinishReason::ToolCalls))
                    .is_empty()
            );
            assert_eq!(
                event_types(&conv.emit_end_events_tagged()),
                vec!["message_delta", "message_stop"]
            );
        }
    }

    #[test]
    fn test_incomplete_tool_call_does_not_create_block_index_gap() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

        conv.process_chunk_tagged(&tool_call_chunk(0, Some("incomplete"), None, None));
        conv.process_chunk_tagged(&tool_call_chunk(
            1,
            Some("call-1"),
            Some("Read"),
            Some("{}"),
        ));

        let finish = conv.process_chunk_tagged(&finish_chunk(FinishReason::ToolCalls));
        assert!(matches!(
            &finish[0].data,
            AnthropicStreamEvent::ContentBlockStart { index: 0, .. }
        ));
        assert!(matches!(
            &finish[1].data,
            AnthropicStreamEvent::ContentBlockDelta { index: 0, .. }
        ));
        assert!(matches!(
            &finish[2].data,
            AnthropicStreamEvent::ContentBlockStop { index: 0 }
        ));
        assert_eq!(
            event_types(&conv.emit_end_events_tagged()),
            vec!["message_delta", "message_stop"]
        );
    }

    /// Text-only response: stop emitted in end events (no early close).
    #[test]
    fn test_text_only_response_stop_in_end_events() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

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
            llm_metrics: None,
        }
    }

    /// Full reasoning flow: thinking → text → tool_use.
    /// Verifies block ordering (thinking=0, text=1, tool=2) and that each
    /// block is properly closed before the next one starts.
    #[test]
    fn test_thinking_text_then_tool_call() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

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

        // 3. Tool call → text block closes; tool output is buffered.
        let ev = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/test.txt\"}"),
        ));
        assert_eq!(event_types(&ev), vec!["content_block_stop"]);
        assert!(matches!(
            &ev[0].data,
            AnthropicStreamEvent::ContentBlockStop { index: 1 }
        ));
        let end = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert!(matches!(
            &end[0].data,
            AnthropicStreamEvent::ContentBlockStart { index: 2, .. }
        ));
    }

    /// Thinking-only response (no text/tool follows): thinking block closed in end events.
    #[test]
    fn test_thinking_only_closed_in_end_events() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);
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

    /// Parallel tool calls flush as non-overlapping blocks at the finish signal.
    #[test]
    fn test_parallel_tool_calls_flush_sequentially_on_finish() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

        let events1 = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/a.txt\"}"),
        ));
        assert!(events1.is_empty());

        let events2 = conv.process_chunk_tagged(&tool_call_chunk(
            1,
            Some("call-2"),
            Some("Write"),
            Some("{\"path\":\"/tmp/b.txt\"}"),
        ));
        assert!(events2.is_empty());

        let finish_events = conv.process_chunk_tagged(&finish_chunk(FinishReason::ToolCalls));
        assert_eq!(
            event_types(&finish_events),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
        assert!(matches!(
            &finish_events[0].data,
            AnthropicStreamEvent::ContentBlockStart { index: 0, .. }
        ));
        assert!(matches!(
            &finish_events[2].data,
            AnthropicStreamEvent::ContentBlockStop { index: 0 }
        ));
        assert!(matches!(
            &finish_events[3].data,
            AnthropicStreamEvent::ContentBlockStart { index: 1, .. }
        ));
        assert!(matches!(
            &finish_events[5].data,
            AnthropicStreamEvent::ContentBlockStop { index: 1 }
        ));
        assert_eq!(
            event_types(&conv.emit_end_events_tagged()),
            vec!["message_delta", "message_stop"]
        );
    }

    #[test]
    fn test_duplicate_tool_call_id_is_emitted_once() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 0);

        conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/a.txt\"}"),
        ));
        conv.process_chunk_tagged(&tool_call_chunk(
            1,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/a.txt\"}"),
        ));

        let finish_events = conv.process_chunk_tagged(&finish_chunk(FinishReason::ToolCalls));
        assert_eq!(
            event_types(&finish_events),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
        assert_eq!(
            event_types(&conv.emit_end_events_tagged()),
            vec!["message_delta", "message_stop"]
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
        let mut conv = AnthropicStreamConverter::with_context("test-model".into(), 0, ctx);
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

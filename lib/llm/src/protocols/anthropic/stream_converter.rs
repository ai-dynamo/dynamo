// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Anthropic Messages API SSE events.
//!
//! The event sequence follows the Anthropic streaming spec:
//! `message_start` -> `content_block_start` -> N x `content_block_delta` ->
//! `content_block_stop` -> `message_delta` -> `message_stop`

use axum::response::sse::Event;
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk, CompletionUsage,
};
use uuid::Uuid;

use super::types::{
    AnthropicDelta, AnthropicErrorBody, AnthropicMessageDeltaBody, AnthropicMessageResponse,
    AnthropicResponseContentBlock, AnthropicStopReason, AnthropicStreamEvent, AnthropicUsage,
    completion_usage_to_anthropic, new_tool_use_id,
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
    // Starts with a frontend estimate and is replaced atomically when the
    // engine reports authoritative usage.
    usage: AnthropicUsage,
    // True once the backend has reported authoritative usage at least once
    // (via `include_usage`/`continuous_usage_stats`). Until then the converter
    // falls back to counting output deltas so that every `content_block_delta`
    // still carries a non-zero output-token estimate.
    saw_backend_usage: bool,
    // Tool call tracking
    tool_call_states: Vec<ToolCallState>,
    tool_blocks_flushed: bool,
    // Block index counter
    next_block_index: u32,
    // Stop reason
    stop_reason: Option<AnthropicStopReason>,
}

struct ToolCallState {
    /// The backend's original tool call ID (e.g. `call_abc123`). Never emitted
    /// to the client; used only to determine that a real call has arrived.
    backend_id: String,
    name: String,
    /// Each buffered argument fragment paired with the cumulative usage snapshot
    /// taken when its backend chunk was processed. Tool-call blocks are flushed
    /// lazily (at the finish reason or EOF), long after their chunks were seen,
    /// so serializing them against the converter's *current* usage would stamp
    /// every `input_json_delta` with the final token count. Snapshotting per
    /// fragment preserves the per-chunk cumulative count instead.
    argument_fragments: Vec<(String, AnthropicUsage)>,
}

impl ToolCallState {
    /// A tool block is ready to flush once both required identity fields are
    /// present. Arguments are optional: a tool call with no parameters still
    /// emits, so `argument_fragments` is deliberately not part of this check.
    fn is_emit_ready(&self) -> bool {
        !self.backend_id.is_empty() && !self.name.is_empty()
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
            usage: AnthropicUsage {
                input_tokens: estimated_input_tokens,
                ..Default::default()
            },
            saw_backend_usage: false,
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
    /// - Within a single call, the `backend_id`/`name` and the argument
    ///   fragments may arrive in either order — arguments can begin before the
    ///   chunk carrying the backend id and name. We therefore record whichever
    ///   fields are present on each chunk and defer emitting the block until the
    ///   identity is complete (see `is_emit_ready`). This is the case the
    ///   fixtures exercise.
    /// - Across parallel calls, the in-tree `dynamo-parsers-v2` parsers emit one
    ///   call at a time with a monotonically increasing `index` (call 0's chunks
    ///   all precede call 1's), so indices are never interleaved today. Indexing
    ///   `tool_call_states` by `tool_call.index` keeps each call's state separate
    ///   regardless, so interleaved indices would also be handled — but no current
    ///   parser emits them, so that path is defensive rather than exercised.
    fn record_tool_call(
        &mut self,
        tool_call: &ChatCompletionMessageToolCallChunk,
        usage_snapshot: &AnthropicUsage,
    ) {
        let tool_call_index = tool_call.index as usize;
        while self.tool_call_states.len() <= tool_call_index {
            self.tool_call_states.push(ToolCallState {
                backend_id: String::new(),
                name: String::new(),
                argument_fragments: Vec::new(),
            });
        }

        let state = &mut self.tool_call_states[tool_call_index];
        if let Some(id) = &tool_call.id {
            state.backend_id = id.clone();
        }
        if let Some(function) = &tool_call.function {
            if let Some(name) = &function.name {
                state.name = name.clone();
            }
            if let Some(arguments) = &function.arguments {
                state
                    .argument_fragments
                    .push((arguments.clone(), usage_snapshot.clone()));
            }
        }
    }

    /// Drain buffered tool blocks into native Anthropic events. Argument
    /// fragments retain the per-chunk usage snapshot captured when the backend
    /// produced them; this matters because the blocks are flushed lazily.
    fn drain_buffered_tool_events(&mut self) -> Vec<AnthropicStreamEvent> {
        if self.tool_blocks_flushed {
            return Vec::new();
        }
        self.tool_blocks_flushed = true;

        let mut events = Vec::new();
        let mut block_index = self.next_block_index;

        for tool_call in &self.tool_call_states {
            if !tool_call.is_emit_ready() {
                continue;
            }

            let emitted_id = new_tool_use_id();
            tracing::debug!(
                backend_id = %tool_call.backend_id,
                emitted_id = %emitted_id,
                "minting Anthropic tool_use id"
            );
            events.push(AnthropicStreamEvent::ContentBlockStart {
                index: block_index,
                content_block: AnthropicResponseContentBlock::ToolUse {
                    id: emitted_id,
                    name: tool_call.name.clone(),
                    input: serde_json::json!({}),
                },
            });

            for (arguments, usage_snapshot) in &tool_call.argument_fragments {
                events.push(AnthropicStreamEvent::ContentBlockDelta {
                    index: block_index,
                    delta: AnthropicDelta::InputJsonDelta {
                        partial_json: arguments.clone(),
                    },
                    usage: Some(usage_snapshot.clone()),
                });
            }

            events.push(AnthropicStreamEvent::ContentBlockStop { index: block_index });
            block_index += 1;
        }

        self.next_block_index = block_index;
        events
    }

    fn append_buffered_tool_events(&mut self, events: &mut Vec<AnthropicStreamEvent>) {
        events.extend(self.drain_buffered_tool_events());
    }

    fn record_usage(&mut self, usage: &CompletionUsage) {
        // Preserve the running output-token estimate if the backend reports a
        // lower value than we have already emitted. Backends that only send a
        // final usage chunk (`include_usage` without `continuous_usage_stats`)
        // would otherwise regress `output_tokens` on the terminal chunk.
        let running = self.usage.output_tokens;
        self.usage = completion_usage_to_anthropic(usage);
        self.usage.output_tokens = self.usage.output_tokens.max(running);
        self.saw_backend_usage = true;
    }

    /// Emit the initial `message_start` event.
    pub fn emit_start_events(&mut self) -> Vec<AnthropicStreamEvent> {
        let mut events = Vec::with_capacity(1);
        self.append_start_events(&mut events);
        events
    }

    /// Append the initial `message_start` event.
    pub fn append_start_events(&mut self, events: &mut Vec<AnthropicStreamEvent>) {
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
            usage: self.usage.clone(),
        };

        let event = AnthropicStreamEvent::MessageStart { message };
        events.push(event);
    }

    /// Process a single chat completion stream chunk and return zero or more SSE events.
    pub fn process_chunk(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<AnthropicStreamEvent> {
        let mut events = Vec::new();
        self.append_chunk_events(chunk, &mut events);
        events
    }

    /// Process a single chat completion stream chunk and append zero or more SSE events.
    pub fn append_chunk_events(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
        events: &mut Vec<AnthropicStreamEvent>,
    ) {
        // Replace the initial estimate when the engine reports authoritative
        // usage (typically on the final chunk). This also applies Anthropic's
        // non-overlapping cached-token accounting.
        if let Some(usage) = &chunk.inner.usage {
            self.record_usage(usage);
        }

        // Fallback output-token accounting for backends that never populate
        // per-chunk `usage` (e.g. the `ModelInput::Text` / PushRouter path).
        // Once the backend reports authoritative usage, `record_usage` owns the
        // count and this estimate is dropped. One token per content-bearing
        // chunk is the standard approximation for token-by-token streaming.
        //
        // This must run *before* the content_block_delta events below are
        // serialized so every token chunk carries a non-zero output-token count
        // (the acceptance requirement: usage present on 100% of token chunks).
        if !self.saw_backend_usage {
            let produced_output = chunk.inner.choices.iter().any(|choice| {
                choice
                    .delta
                    .reasoning_content
                    .as_ref()
                    .is_some_and(|r| !r.is_empty())
                    || matches!(
                        &choice.delta.content,
                        Some(ChatCompletionMessageContent::Text(t)) if !t.is_empty()
                    )
                    || choice
                        .delta
                        .tool_calls
                        .as_ref()
                        .is_some_and(|tool_calls| !tool_calls.is_empty())
            });
            if produced_output {
                self.usage.output_tokens += 1;
            }
        }

        // Snapshot the cumulative usage for this chunk so buffered tool-argument
        // fragments recorded below are stamped with the count as of *this* chunk,
        // not the final count at flush time. `self.usage` is stable for the rest
        // of this call (record_usage / the fallback above already ran).
        let usage_snapshot = self.usage.clone();

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
                    events.push(block_start);
                }

                // Emit thinking delta
                let block_delta = AnthropicStreamEvent::ContentBlockDelta {
                    index: self.thinking_block_index,
                    delta: AnthropicDelta::ThinkingDelta {
                        thinking: reasoning.clone(),
                    },
                    usage: Some(self.usage.clone()),
                };
                events.push(block_delta);
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
                        usage: Some(self.usage.clone()),
                    };
                    events.push(sig_delta);

                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(block_stop);
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
                    events.push(block_start);
                }

                // Emit text delta
                let block_delta = AnthropicStreamEvent::ContentBlockDelta {
                    index: self.text_block_index,
                    delta: AnthropicDelta::TextDelta {
                        text: text.to_string(),
                    },
                    usage: Some(self.usage.clone()),
                };
                events.push(block_delta);
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
                        usage: Some(self.usage.clone()),
                    };
                    events.push(sig_delta);
                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(block_stop);
                }

                // Close the text block before opening any tool blocks.
                // Anthropic streaming spec requires each block to be closed
                // (content_block_stop) before the next block starts.
                if self.text_block_started && !self.text_block_closed {
                    self.text_block_closed = true;
                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.text_block_index,
                    };
                    events.push(block_stop);
                }

                for tool_call in tool_calls {
                    self.record_tool_call(tool_call, &usage_snapshot);
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
    pub fn emit_end_events(&mut self) -> Vec<AnthropicStreamEvent> {
        let mut events = Vec::new();
        self.append_end_events(&mut events);
        events
    }

    /// Append the final events when the stream ends.
    pub fn append_end_events(&mut self, events: &mut Vec<AnthropicStreamEvent>) {
        // Close thinking block if started and not already closed mid-stream
        if self.thinking_block_started && !self.thinking_block_closed {
            self.thinking_block_closed = true;
            let sig_delta = AnthropicStreamEvent::ContentBlockDelta {
                index: self.thinking_block_index,
                delta: AnthropicDelta::SignatureDelta {
                    signature: "erased".to_string(),
                },
                usage: Some(self.usage.clone()),
            };
            events.push(sig_delta);
            let block_stop = AnthropicStreamEvent::ContentBlockStop {
                index: self.thinking_block_index,
            };
            events.push(block_stop);
        }

        // Close text block if started and not already closed mid-stream
        if self.text_block_started && !self.text_block_closed {
            let block_stop = AnthropicStreamEvent::ContentBlockStop {
                index: self.text_block_index,
            };
            events.push(block_stop);
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
            usage: self.usage.clone(),
        };
        events.push(message_delta);

        // Emit message_stop
        let message_stop = AnthropicStreamEvent::MessageStop {};
        events.push(message_stop);
    }

    /// Emit error events when the stream ends due to a backend error.
    pub fn emit_error_events(&mut self) -> Vec<AnthropicStreamEvent> {
        let mut events = Vec::with_capacity(1);
        self.append_error_events(&mut events);
        events
    }

    /// Append error events when the stream ends due to a backend error.
    pub fn append_error_events(&mut self, events: &mut Vec<AnthropicStreamEvent>) {
        let error_event = AnthropicStreamEvent::Error {
            error: AnthropicErrorBody {
                error_type: "api_error".to_string(),
                message: "An internal error occurred during generation.".to_string(),
            },
        };
        events.push(error_event);
    }
}

pub(crate) fn make_sse_event(event: &AnthropicStreamEvent) -> Result<Event, anyhow::Error> {
    let data = serde_json::to_string(&event_json_with_usage(event)?)?;
    Ok(Event::default().event(stream_event_type(event)).data(data))
}

pub(crate) fn stream_event_type(event: &AnthropicStreamEvent) -> &'static str {
    match event {
        AnthropicStreamEvent::MessageStart { .. } => "message_start",
        AnthropicStreamEvent::ContentBlockStart { .. } => "content_block_start",
        AnthropicStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        AnthropicStreamEvent::ContentBlockStop { .. } => "content_block_stop",
        AnthropicStreamEvent::MessageDelta { .. } => "message_delta",
        AnthropicStreamEvent::MessageStop {} => "message_stop",
        AnthropicStreamEvent::Ping {} => "ping",
        AnthropicStreamEvent::Error { .. } => "error",
    }
}

/// Serialize an Anthropic stream event to JSON, injecting a `usage` triple onto
/// `content_block_delta` events (and leaving every other event untouched).
///
/// The native event carries the cumulative Anthropic usage. Dynamo adds the
/// derived `total_tokens` field at its HTTP boundary so typed in-process users
/// do not depend on a wire-only extension.
///
/// `total_tokens` counts the *complete* prompt plus the output, i.e.
/// `input_tokens + cache_read_input_tokens + cache_creation_input_tokens +
/// output_tokens`. Anthropic reports the cached prefix separately from
/// `input_tokens` (`completion_usage_to_anthropic` subtracts it out), so the
/// cache fields must be added back here; otherwise a cache-hit request would
/// under-report and disagree with the `total_tokens` a proxy sees on the
/// equivalent `/v1/chat/completions` request (`prompt_tokens + completion_tokens`,
/// cached tokens included). Saturating arithmetic guards against overflow.
fn event_json_with_usage(event: &AnthropicStreamEvent) -> Result<serde_json::Value, anyhow::Error> {
    let mut value = serde_json::to_value(event)?;
    if let (
        AnthropicStreamEvent::ContentBlockDelta {
            usage: Some(usage), ..
        },
        serde_json::Value::Object(map),
    ) = (event, &mut value)
        && let Some(serde_json::Value::Object(usage_map)) = map.get_mut("usage")
    {
        let total = usage
            .input_tokens
            .saturating_add(usage.cache_read_input_tokens.unwrap_or(0))
            .saturating_add(usage.cache_creation_input_tokens.unwrap_or(0))
            .saturating_add(usage.output_tokens);
        usage_map.insert("total_tokens".to_string(), serde_json::json!(total));
    }
    Ok(value)
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
        self.process_chunk(chunk)
            .into_iter()
            .map(|event| make_tagged_event(stream_event_type(&event), &event))
            .collect()
    }

    /// Like `emit_end_events` but returns tagged events for test assertions.
    fn emit_end_events_tagged(&mut self) -> Vec<TaggedEvent> {
        self.emit_end_events()
            .into_iter()
            .map(|event| make_tagged_event(stream_event_type(&event), &event))
            .collect()
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

        events.clear();
        let capacity = events.capacity();
        conv.append_chunk_events(&text_chunk("I'll edit the file."), &mut events);
        // content_block_start + content_block_delta. The content_block_delta now
        // carries an injected per-chunk usage triple (see
        // test_content_block_delta_carries_usage_triple) rather than a separate
        // interim message_delta event.
        assert_eq!(events.len(), 2);
        assert_eq!(events.capacity(), capacity);

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

        events.clear();
        conv.append_chunk_events(&finish_chunk(FinishReason::ToolCalls), &mut events);
        assert_eq!(events.len(), 3);
        assert_eq!(events.capacity(), capacity);

        events.clear();
        conv.append_end_events(&mut events);
        assert_eq!(events.len(), 2);
        assert_eq!(events.capacity(), capacity);

        events.clear();
        conv.append_error_events(&mut events);
        assert_eq!(events.len(), 1);
        assert_eq!(events.capacity(), capacity);
    }

    /// A chunk carrying engine usage (typically the final chunk).
    fn usage_chunk(
        prompt_tokens: u32,
        cached_tokens: Option<u32>,
        completion_tokens: u32,
    ) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chat-1".into(),
                choices: vec![],
                created: 0,
                model: "test".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    prompt_tokens_details: cached_tokens.map(|c| {
                        dynamo_protocols::types::PromptTokensDetails {
                            audio_tokens: None,
                            cached_tokens: Some(c),
                        }
                    }),
                    completion_tokens_details: None,
                }),
            },
            nvext: None,
            llm_metrics: None,
        }
    }

    /// Streaming usage starts with the frontend estimate, then reconciles to
    /// the engine's total prompt tokens minus its cached-token count.
    #[test]
    fn test_streaming_input_tokens_reconciled_from_engine_usage() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 19);

        // `message_start` is emitted before backend usage is available.
        assert_eq!(conv.usage.input_tokens, 19);

        // Exercise the production chunk path rather than its tagged test mirror.
        let mut events = Vec::new();
        conv.append_chunk_events(&usage_chunk(12, Some(11), 5), &mut events);
        // A usage-only chunk carries no content deltas, so it emits no SSE
        // events; the reconciled usage is stamped onto subsequent token chunks
        // and the terminal message_delta.
        assert!(events.is_empty(), "usage-only chunk emits no SSE events");
        assert_eq!(conv.usage.input_tokens, 1);
        assert_eq!(conv.usage.cache_read_input_tokens, Some(11));
        assert_eq!(conv.usage.output_tokens, 5);

        let delta = conv.emit_end_events_tagged();
        let message_delta = delta
            .iter()
            .find(|e| e.event_type == "message_delta")
            .expect("message_delta present");
        match &message_delta.data {
            AnthropicStreamEvent::MessageDelta { usage, .. } => {
                assert_eq!(usage.input_tokens, 1);
                assert_eq!(usage.cache_read_input_tokens, Some(11));
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected MessageDelta, got {other:?}"),
        }
    }

    /// Backends that never populate per-chunk `usage` (the `ModelInput::Text`
    /// path) still get a running output-token estimate that advances *before*
    /// each token chunk is serialized, so every `content_block_delta` carries a
    /// non-zero usage triple even for a client that aborts mid-stream.
    #[test]
    fn test_fallback_counter_advances_before_each_token_chunk() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 7);
        let mut events = Vec::new();

        // First content chunk: content_block_start + content_block_delta. The
        // fallback advances output_tokens to 1 before the delta is serialized.
        conv.append_chunk_events(&text_chunk("Hello"), &mut events);
        assert_eq!(conv.usage.output_tokens, 1);
        assert_eq!(events.len(), 2);

        // Second content chunk on the open block: content_block_delta only
        // (output_tokens advanced to 2).
        events.clear();
        conv.append_chunk_events(&text_chunk(" world"), &mut events);
        assert_eq!(conv.usage.output_tokens, 2);
        assert_eq!(events.len(), 1);

        // The frontend input estimate is preserved until the backend reports.
        assert_eq!(conv.usage.input_tokens, 7);
        assert!(!conv.saw_backend_usage);
    }

    /// The root fix: `content_block_delta` events carry an injected `usage`
    /// triple (input/output/total) — mirroring OpenAI `continuous_usage_stats`
    /// — while non-delta events are left untouched.
    #[test]
    fn test_content_block_delta_carries_usage_triple() {
        let usage = AnthropicUsage {
            input_tokens: 7,
            output_tokens: 3,
            ..Default::default()
        };

        let delta = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicDelta::TextDelta {
                text: "hi".to_string(),
            },
            usage: Some(usage),
        };
        let value = event_json_with_usage(&delta).expect("serialize");
        let usage_obj = value
            .get("usage")
            .expect("content_block_delta must carry usage");
        assert_eq!(
            usage_obj.get("input_tokens").and_then(|v| v.as_u64()),
            Some(7)
        );
        assert_eq!(
            usage_obj.get("output_tokens").and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(
            usage_obj.get("total_tokens").and_then(|v| v.as_u64()),
            Some(10),
            "total_tokens must equal input + output"
        );

        // Cache-hit case: `total_tokens` must count the complete prompt
        // (visible input + cached prefix + cache writes) plus output, matching
        // the `prompt_tokens + completion_tokens` total a proxy sees on
        // `/v1/chat/completions`. Anthropic reports the cached prefix outside
        // `input_tokens`, so it must be added back into the total.
        let cached_usage = AnthropicUsage {
            input_tokens: 2,
            output_tokens: 3,
            cache_read_input_tokens: Some(9),
            cache_creation_input_tokens: Some(1),
        };
        let cached_delta = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicDelta::TextDelta {
                text: "hi".to_string(),
            },
            usage: Some(cached_usage),
        };
        let cached_value = event_json_with_usage(&cached_delta).expect("serialize");
        let cached_usage_obj = cached_value
            .get("usage")
            .expect("content_block_delta must carry usage");
        assert_eq!(
            cached_usage_obj
                .get("input_tokens")
                .and_then(|v| v.as_u64()),
            Some(2),
            "input_tokens stays the visible (non-cached) prompt count"
        );
        assert_eq!(
            cached_usage_obj
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64()),
            Some(9)
        );
        assert_eq!(
            cached_usage_obj
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_u64()),
            Some(1)
        );
        assert_eq!(
            cached_usage_obj
                .get("total_tokens")
                .and_then(|v| v.as_u64()),
            Some(15),
            "total_tokens must equal input + cache_read + cache_creation + output"
        );

        // A non-delta event carries no injected usage.
        let stop = AnthropicStreamEvent::ContentBlockStop { index: 0 };
        let stop_value = event_json_with_usage(&stop).expect("serialize");
        assert!(
            stop_value.get("usage").is_none(),
            "only content_block_delta gets injected usage"
        );
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
            } if id.starts_with("toolu_")
                && id.len() > "toolu_".len()
                && name == "Read"
                && input == &serde_json::json!({})
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

    /// Buffered tool-argument fragments must carry the usage snapshot from their
    /// own chunk, not the cumulative count at flush time. With no backend usage
    /// the fallback advances `output_tokens` by one per tool-call chunk, so two
    /// argument fragments serialize with `output_tokens` 1 and 2 respectively.
    #[test]
    fn test_buffered_tool_fragments_snapshot_per_chunk_usage() {
        let mut conv = AnthropicStreamConverter::new("test-model".into(), 5);
        let mut sink = Vec::new();

        conv.append_chunk_events(
            &tool_call_chunk(0, Some("call-1"), Some("Read"), Some("{\"path\":\"/a")),
            &mut sink,
        );
        conv.append_chunk_events(&tool_call_chunk(0, None, None, Some(".txt\"}")), &mut sink);
        // Tool blocks are buffered until a finish/EOF flush, so nothing is
        // emitted per chunk.
        assert!(
            sink.is_empty(),
            "tool blocks are buffered, not emitted per chunk"
        );
        // Fallback counted one token per tool-call chunk.
        assert_eq!(conv.usage.output_tokens, 2);

        let fragment_outputs: Vec<u32> = conv
            .drain_buffered_tool_events()
            .iter()
            .filter_map(|event| match event {
                AnthropicStreamEvent::ContentBlockDelta {
                    delta: AnthropicDelta::InputJsonDelta { .. },
                    usage,
                    ..
                } => Some(
                    usage
                        .as_ref()
                        .expect("tool-argument fragment carries a usage snapshot")
                        .output_tokens,
                ),
                _ => None,
            })
            .collect();
        assert_eq!(
            fragment_outputs,
            vec![1, 2],
            "each fragment stamps the cumulative output count as of its own chunk"
        );
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

    /// Two tool calls that share a backend ID (different indices) must both be
    /// emitted. The old dedup keyed on backend ID, which was safe when the
    /// emitted ID also came from the backend — both blocks would have been
    /// identical and unroutable. Now that each block gets a freshly minted
    /// `toolu_` ID, collapsing them discards a distinct, routable call.
    #[test]
    fn test_shared_backend_id_emits_both_calls() {
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
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
        // Both emitted IDs must be Anthropic-native and distinct.
        let starts: Vec<_> = finish_events
            .iter()
            .filter(|e| e.event_type == "content_block_start")
            .collect();
        assert_eq!(starts.len(), 2);
        let ids: Vec<_> = starts
            .iter()
            .filter_map(|e| {
                if let AnthropicStreamEvent::ContentBlockStart {
                    content_block: AnthropicResponseContentBlock::ToolUse { id, .. },
                    ..
                } = &e.data
                {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        for id in &ids {
            assert!(
                id.starts_with("toolu_") && id.len() > "toolu_".len(),
                "emitted id must be Anthropic-native, got {id:?}"
            );
        }
        assert_ne!(ids[0], ids[1], "parallel calls must receive distinct ids");
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

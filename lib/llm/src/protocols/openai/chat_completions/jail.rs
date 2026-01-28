// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_stream::stream;
use dynamo_async_openai::types::{
    ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageToolCallChunk,
    ChatCompletionStreamResponseDelta, FinishReason, FunctionCallStream, Role,
};

use dynamo_parsers::tool_calling::parsers::get_tool_parser_map;
use dynamo_parsers::tool_calling::{
    detect_tool_call_start, find_tool_call_end_position, try_tool_call_parse_aggregate,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use std::collections::HashMap;
use uuid::Uuid;

use crate::utils::{MarkerMatcher, MatchResult};

use super::NvCreateChatCompletionStreamResponse;

/// Represents what a choice wants to emit after processing content
#[derive(Debug, Clone)]
pub enum ChoiceEmission {
    /// Pass through content unchanged (choice is not jailed)
    PassThrough(ChatChoiceStream),
    /// Emit parsed tool calls (choice finished jailing with tool calls)
    ToolCall(ChatChoiceStream),
    /// Emit accumulated content (choice finished jailing without tool calls)
    Content(ChatChoiceStream),
    /// Emit trailing content after tool call end (choice has trailing after unjail)
    Trailing(ChatChoiceStream),
}

impl ChoiceEmission {
    /// Extract the ChatChoiceStream from any emission type
    pub fn into_choice(self) -> ChatChoiceStream {
        match self {
            ChoiceEmission::PassThrough(choice) => choice,
            ChoiceEmission::ToolCall(choice) => choice,
            ChoiceEmission::Content(choice) => choice,
            ChoiceEmission::Trailing(choice) => choice,
        }
    }

    /// Get the choice index
    pub fn index(&self) -> u32 {
        match self {
            ChoiceEmission::PassThrough(choice) => choice.index,
            ChoiceEmission::ToolCall(choice) => choice.index,
            ChoiceEmission::Content(choice) => choice.index,
            ChoiceEmission::Trailing(choice) => choice.index,
        }
    }
}

/// Configuration for jail detection and parsing
#[derive(Debug, Clone)]
pub struct JailConfig<'a> {
    pub jail_start_sequences: &'a [String],
    pub jail_end_sequences: &'a [String],
    pub tool_call_parser: Option<&'a str>,
}

/// Jail activation mode
#[derive(Debug, Clone, PartialEq)]
pub enum JailMode {
    /// Traditional: wait for start marker, then jail
    MarkerBased,
    /// Immediate: start jailed from first token (for tool_choice)
    Immediate { format: ToolChoiceFormat },
}

/// Format for tool_choice immediate jail mode
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoiceFormat {
    /// tool_choice=named: expect single object {"location": "Paris", ...}
    SingleObject { tool_name: String },
    /// tool_choice=required: expect array [{name:"search", parameters:{...}}, ...]
    ArrayOfTools,
}

/// State tracking for an individual choice during jail processing
#[derive(Debug, Clone)]
struct ChoiceJailState {
    /// The choice index (0, 1, 2, ...)
    index: u32,
    /// Whether this choice is currently jailed
    is_jailed: bool,
    /// Accumulated content for this choice while jailed
    accumulated_content: String,
    /// Buffer for partial marker matches across chunks
    partial_match_buffer: String,
    /// Stream finish reason
    stream_finish_reason: Option<FinishReason>,
    /// Number of tool calls already emitted for this choice
    emitted_tool_calls_count: usize,
}

fn create_choice_stream(
    index: u32,
    role: Option<Role>,
    content: &str,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    finish_reason: Option<FinishReason>,
    stop_reason: Option<dynamo_async_openai::types::StopReason>,
    logprobs: Option<ChatChoiceLogprobs>,
) -> ChatChoiceStream {
    #[allow(deprecated)]
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role,
            content: Some(content.to_string()),
            tool_calls,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason,
        stop_reason,
        logprobs,
    }
}

impl ChoiceJailState {
    /// Create a new jail state for a choice
    fn new(index: u32, starts_jailed: bool) -> Self {
        Self {
            index,
            is_jailed: starts_jailed,
            accumulated_content: String::new(),
            partial_match_buffer: String::new(),
            stream_finish_reason: None,
            emitted_tool_calls_count: 0,
        }
    }

    /// Add content to this choice's accumulation
    fn accumulate(&mut self, content: &str) {
        if self.is_jailed {
            self.accumulated_content.push_str(content);
        }
    }

    /// End jailing and return the accumulated content
    fn end_jail(&mut self) -> String {
        self.is_jailed = false;
        std::mem::take(&mut self.accumulated_content)
    }

    /// Process incoming content and return what should be emitted (if anything)
    async fn process_content(
        &mut self,
        choice: &ChatChoiceStream,
        content: &str,
        jail_stream: &JailedStream,
    ) -> Vec<ChoiceEmission> {
        let mut emissions = Vec::new();
        if !self.is_jailed {
            // Use the marker matcher to detect complete/partial markers
            let match_result = jail_stream
                .marker_matcher
                .process_chunk(content, &self.partial_match_buffer);

            match match_result {
                MatchResult::Complete {
                    prefix,
                    marker,
                    suffix,
                    ..
                } => {
                    // Emit prefix if any
                    if !prefix.is_empty() {
                        #[allow(deprecated)]
                        let prefix_choice = create_choice_stream(
                            choice.index,
                            choice.delta.role,
                            &prefix,
                            None,
                            choice.finish_reason,
                            None,
                            choice.logprobs.clone(),
                        );
                        emissions.push(ChoiceEmission::PassThrough(prefix_choice));
                    }

                    // Build the potential full content
                    let full_content = format!("{}{}", marker, suffix);

                    // Check if this already contains the end marker
                    let (should_end, split_pos) = jail_stream.should_end_jail(&full_content).await;

                    if should_end {
                        // Complete tool call found in this chunk
                        tracing::debug!(
                            "Choice {} complete tool call detected in single chunk",
                            choice.index
                        );

                        let (jailed_part, trailing_part) = full_content.split_at(split_pos);

                        // Create the tool call choice
                        let tool_choice = jail_stream
                            .create_tool_call_choice(
                                choice.index,
                                jailed_part,
                                choice,
                                self.emitted_tool_calls_count,
                            )
                            .await;

                        if tool_choice.delta.tool_calls.is_some() {
                            if let Some(ref tool_calls) = tool_choice.delta.tool_calls {
                                self.emitted_tool_calls_count += tool_calls.len();
                            }
                            emissions.push(ChoiceEmission::ToolCall(tool_choice));
                        } else {
                            emissions.push(ChoiceEmission::Content(tool_choice));
                        }

                        // Handle trailing content if any
                        if !trailing_part.is_empty() {
                            #[allow(deprecated)]
                            let trailing_choice = create_choice_stream(
                                choice.index,
                                choice.delta.role,
                                trailing_part,
                                None,
                                choice.finish_reason,
                                None,
                                choice.logprobs.clone(),
                            );
                            emissions.push(ChoiceEmission::Trailing(trailing_choice));
                        }
                    } else {
                        // Start jailing with the marker and suffix
                        tracing::debug!(
                            "Choice {} start marker '{}' detected, starting jail",
                            choice.index,
                            marker
                        );
                        self.is_jailed = true;
                        self.accumulated_content = full_content;
                    }

                    self.partial_match_buffer.clear();
                }

                MatchResult::Partial {
                    prefix,
                    partial,
                    possible_patterns,
                } => {
                    // Emit the safe prefix
                    if !prefix.is_empty() {
                        #[allow(deprecated)]
                        let prefix_choice = create_choice_stream(
                            choice.index,
                            choice.delta.role,
                            &prefix,
                            None,
                            choice.finish_reason,
                            None,
                            choice.logprobs.clone(),
                        );
                        emissions.push(ChoiceEmission::PassThrough(prefix_choice));
                    }

                    // Hold the partial for next chunk
                    self.partial_match_buffer = partial;

                    tracing::trace!(
                        "Choice {} holding partial '{}' for patterns: {:?}",
                        choice.index,
                        self.partial_match_buffer,
                        possible_patterns
                    );
                }

                MatchResult::None { content } => {
                    // Check if this content (combined with partial buffer) should start jailing
                    let combined_content = if self.partial_match_buffer.is_empty() {
                        content.clone()
                    } else {
                        format!("{}{}", self.partial_match_buffer, content)
                    };

                    if jail_stream.should_start_jail(&combined_content) {
                        // Start jailing with the combined content
                        tracing::debug!(
                            "Choice {} tool call start detected via parser, starting jail",
                            choice.index
                        );
                        self.is_jailed = true;
                        self.accumulated_content = combined_content;
                        self.partial_match_buffer.clear();
                    } else {
                        // No markers - emit everything
                        if !content.is_empty() {
                            #[allow(deprecated)]
                            let pass_through_choice = create_choice_stream(
                                choice.index,
                                choice.delta.role,
                                &content,
                                None,
                                choice.finish_reason,
                                None,
                                choice.logprobs.clone(),
                            );
                            emissions.push(ChoiceEmission::PassThrough(pass_through_choice));
                        }
                        self.partial_match_buffer.clear();
                    }
                }
            }
        } else {
            // Already jailed - accumulate and check for unjail
            self.accumulate(content);

            let (should_end, split_pos) =
                jail_stream.should_end_jail(&self.accumulated_content).await;

            if should_end {
                tracing::debug!(
                    "Choice {} jail exit detected, releasing accumulated content",
                    choice.index
                );

                // Split the content
                let (jailed_part, trailing_part) = self.accumulated_content.split_at(split_pos);

                // Create the unjailed choice
                let unjailed_choice = jail_stream
                    .create_tool_call_choice(
                        choice.index,
                        jailed_part,
                        choice,
                        self.emitted_tool_calls_count,
                    )
                    .await;

                // Determine emission type based on whether tool calls were parsed
                if unjailed_choice.delta.tool_calls.is_some() {
                    if let Some(ref tool_calls) = unjailed_choice.delta.tool_calls {
                        self.emitted_tool_calls_count += tool_calls.len();
                    }
                    emissions.push(ChoiceEmission::ToolCall(unjailed_choice));
                } else {
                    emissions.push(ChoiceEmission::Content(unjailed_choice));
                }

                // Handle trailing content if any
                if !trailing_part.is_empty() {
                    #[allow(deprecated)]
                    let trailing_choice = create_choice_stream(
                        choice.index,
                        choice.delta.role,
                        trailing_part,
                        None,
                        choice.finish_reason,
                        None,
                        choice.logprobs.clone(),
                    );
                    emissions.push(ChoiceEmission::Trailing(trailing_choice));
                }

                // End jailing
                self.end_jail();
            }
            // If not unjailing, don't emit anything (still accumulating)
        }
        emissions
    }

    /// Finalize any remaining content when stream ends
    async fn finalize(&mut self, jail_stream: &JailedStream) -> Option<ChoiceEmission> {
        if self.is_jailed && !self.accumulated_content.is_empty() {
            tracing::debug!(
                "Choice {} stream ended while jailed, releasing accumulated content",
                self.index
            );

            // Create a dummy choice for the method call
            #[allow(deprecated)]
            let dummy_choice = create_choice_stream(
                self.index,
                Some(Role::Assistant),
                &self.accumulated_content,
                None,
                self.stream_finish_reason, // For the accumulated content, assign the original stream finish reason, otherwise it will get lost
                None,
                None,
            );

            let final_choice = jail_stream
                .create_tool_call_choice(
                    self.index,
                    &self.accumulated_content,
                    &dummy_choice,
                    self.emitted_tool_calls_count,
                )
                .await;

            if let Some(ref tool_calls) = final_choice.delta.tool_calls {
                self.emitted_tool_calls_count += tool_calls.len();
            }

            // End jailing
            self.end_jail();

            // Determine emission type
            if final_choice.delta.tool_calls.is_some() {
                Some(ChoiceEmission::ToolCall(final_choice))
            } else {
                Some(ChoiceEmission::Content(final_choice))
            }
        } else {
            None
        }
    }
}

/// Collection of choice jail states with deterministic ordering
#[derive(Debug, Clone)]
struct ChoiceJailStateCollection {
    /// Vec of states, always kept sorted by choice index for deterministic iteration
    states: Vec<ChoiceJailState>,
}

impl ChoiceJailStateCollection {
    /// Create a new empty collection
    fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Get or create state for a choice index
    fn get_or_create_state(&mut self, index: u32, starts_jailed: bool) -> &mut ChoiceJailState {
        // Find the position where this index should be
        match self.states.binary_search_by_key(&index, |s| s.index) {
            Ok(pos) => {
                // Found existing state
                &mut self.states[pos]
            }
            Err(insert_pos) => {
                // Need to create new state
                let new_state = ChoiceJailState::new(index, starts_jailed);
                self.states.insert(insert_pos, new_state);
                &mut self.states[insert_pos]
            }
        }
    }
}

/// Emission mode for handling multiple choices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmissionMode {
    /// Pack multiple choices in the same chunk (default, matches original behavior)
    #[default]
    Packed,
    /// Emit one choice per chunk for OpenAI compatibility
    SingleChoicePerChunk,
}

/// A stream transformer that can "jail" tokens based on configurable start/end sequences
/// When jailed, tokens are accumulated rather than yielded immediately
/// When the jail ends (via end sequence or stream completion), accumulated content is processed and released
pub struct JailedStream {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    tool_definitions: Option<Vec<dynamo_parsers::tool_calling::ToolDefinition>>,
    emission_mode: EmissionMode,
    marker_matcher: MarkerMatcher,
    jail_mode: JailMode,
}

impl JailedStream {
    /// Create a new builder for configuring a JailedStream
    pub fn builder() -> JailedStreamBuilder {
        JailedStreamBuilder::new()
    }

    /// Apply jail stream transformation with finish_reason fix
    /// This is a convenience method that applies both apply() and fix_finish_reason()
    pub fn apply_with_finish_reason<S>(
        self,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        let jail_mode = self.jail_mode.clone();
        let jailed_stream = self.apply(stream);
        JailedStream::fix_finish_reason(jailed_stream, jail_mode)
    }

    /// Apply the jail transformation to a stream of chat completion responses
    /// Consumes self and returns the transformed stream
    pub fn apply<S>(
        self,
        stream: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        // Use the stream! macro for cleaner async stream processing
        stream! {
            // State variables - clean architecture with choice state collection
            let mut choice_states = ChoiceJailStateCollection::new();
            // Track Annotated metadata for preservation
            let mut last_annotated_id: Option<String> = None;
            let mut last_annotated_event: Option<String> = None;
            let mut last_annotated_comment: Option<Vec<String>> = None;

            // Pin the stream for iteration (stack pinning is more efficient)
            tokio::pin!(stream);


            // Process each item in the stream
            while let Some(response) = stream.next().await {
                if let Some(chat_response) = response.data.as_ref() {
                    let mut all_emissions = Vec::new();

                    if chat_response.choices.is_empty() {
                        // No choices processed (e.g., usage-only chunk)
                        // Pass through as-is to preserve usage and other metadata
                        yield response;
                        continue;
                    }

                    // Process each choice independently using the new architecture
                    for choice in &chat_response.choices {
                        if let Some(ref content) = choice.delta.content {
                            let starts_jailed = matches!(self.jail_mode, JailMode::Immediate { .. });
                            let choice_state = choice_states.get_or_create_state(choice.index, starts_jailed);

                            // Store metadata when any choice becomes jailed (first time only)
                            if !choice_state.is_jailed && self.should_start_jail(content)
                                && last_annotated_id.is_none() {
                                    last_annotated_id = response.id.clone();
                                    last_annotated_event = response.event.clone();
                                    last_annotated_comment = response.comment.clone();
                                }

                            // Track actual stream finish reason in the choice state
                            choice_state.stream_finish_reason = choice.finish_reason;

                            // Process this choice and get emissions
                            let emissions = choice_state.process_content(choice, content, &self).await;
                            all_emissions.extend(emissions);
                        } else {
                            // Handle choices without content (e.g., final chunks with finish_reason)
                            // Only filter out if this choice was ever jailed and lacks role
                            // (to avoid aggregator issues with deltas missing role after unjail)
                            let choice_state = choice_states.get_or_create_state(choice.index, false);
                            let was_ever_jailed = !choice_state.accumulated_content.is_empty() || choice_state.is_jailed;

                            let should_emit = choice.delta.role.is_some()
                                || choice.delta.tool_calls.is_some()
                                || !was_ever_jailed; // Always pass through if never jailed

                            if should_emit {
                                let pass_through_choice = ChatChoiceStream {
                                    index: choice.index,
                                    delta: choice.delta.clone(),
                                    finish_reason: choice.finish_reason,
                                    stop_reason: choice.stop_reason.clone(),
                                    logprobs: choice.logprobs.clone(),
                                };
                                all_emissions.push(ChoiceEmission::PassThrough(pass_through_choice));
                            }
                        }
                    }

                    // Emit all results based on emission mode
                    if !all_emissions.is_empty() {
                        // Group emissions by type for proper ordering and separation
                        let mut tool_content_emissions = Vec::new();
                        let mut trailing_emissions = Vec::new();
                        let mut passthrough_emissions = Vec::new();

                        for emission in all_emissions {
                            match emission {
                                ChoiceEmission::PassThrough(_) => passthrough_emissions.push(emission),
                                ChoiceEmission::ToolCall(_) | ChoiceEmission::Content(_) => {
                                    tool_content_emissions.push(emission);
                                }
                                ChoiceEmission::Trailing(_) => {
                                    trailing_emissions.push(emission);
                                }
                            }
                        }

                        // Emit tool calls and content with preserved metadata
                        if !tool_content_emissions.is_empty() {
                            let preserved_metadata = (
                                last_annotated_id.clone(),
                                last_annotated_event.clone(),
                                last_annotated_comment.clone(),
                            );
                            let responses = self.emit_choice_emissions(tool_content_emissions, chat_response, preserved_metadata);
                            for emitted_response in responses {
                                yield emitted_response;
                            }
                        }

                        // Emit trailing content separately (always as individual chunks)
                        if !trailing_emissions.is_empty() {
                            let preserved_metadata = (
                                last_annotated_id.clone(),
                                last_annotated_event.clone(),
                                last_annotated_comment.clone(),
                            );
                            let responses = self.emit_choice_emissions(trailing_emissions, chat_response, preserved_metadata);
                            for emitted_response in responses {
                                yield emitted_response;
                            }
                        }

                        // Emit pass-through content with current metadata
                        if !passthrough_emissions.is_empty() {
                            let current_metadata = (response.id.clone(), response.event.clone(), response.comment.clone());
                            let responses = self.emit_choice_emissions(passthrough_emissions, chat_response, current_metadata);
                            for emitted_response in responses {
                                yield emitted_response;
                            }
                        }
                    }
                } else {
                    // No response data, pass through as-is
                    yield response;
                }
            }

            // Stream ended - finalize any remaining jailed choices
            let mut final_emissions = Vec::new();
            for state in choice_states.states.iter_mut() {
                if let Some(emission) = state.finalize(&self).await {
                    final_emissions.push(emission);
                }
            }

            if !final_emissions.is_empty() {
                tracing::debug!("Stream ended while jailed, releasing accumulated content");
                // Create a dummy response for finalization
                let dummy_response = NvCreateChatCompletionStreamResponse {
                    id: "stream-end".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 0,
                    model: "unknown".to_string(),
                    choices: Vec::new(),
                    usage: None,
                    service_tier: None,
                    system_fingerprint: None,
                    nvext: None,
                };

                let final_metadata = (last_annotated_id, last_annotated_event, last_annotated_comment);
                let responses = self.emit_choice_emissions(final_emissions, &dummy_response, final_metadata);
                for emitted_response in responses {
                    yield emitted_response;
                }
            }
        }
    }

    /// Emit choice emissions based on the configured emission mode
    fn emit_choice_emissions(
        &self,
        emissions: Vec<ChoiceEmission>,
        base_response: &NvCreateChatCompletionStreamResponse,
        annotated_metadata: (Option<String>, Option<String>, Option<Vec<String>>),
    ) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        if emissions.is_empty() {
            return Vec::new();
        }

        let (id, event, comment) = annotated_metadata;

        match self.emission_mode {
            EmissionMode::Packed => {
                // Pack all choices into a single response
                let mut response = base_response.clone();
                response.choices = emissions.into_iter().map(|e| e.into_choice()).collect();

                vec![Annotated {
                    data: Some(response),
                    id,
                    event,
                    comment,
                }]
            }
            EmissionMode::SingleChoicePerChunk => {
                // Emit each choice in a separate response
                emissions
                    .into_iter()
                    .map(|emission| {
                        let mut response = base_response.clone();
                        response.choices = vec![emission.into_choice()];

                        Annotated {
                            data: Some(response),
                            id: id.clone(),
                            event: event.clone(),
                            comment: comment.clone(),
                        }
                    })
                    .collect()
            }
        }
    }

    /// Check if content matches any jail start patterns
    fn should_start_jail(&self, content: &str) -> bool {
        // Path 1: Check configured start sequences
        let sequence_match = !self.jail_start_sequences.is_empty()
            && self
                .jail_start_sequences
                .iter()
                .any(|seq| content.contains(seq));

        // Path 2: Check for tool call start pattern
        let tool_call_match = self.tool_call_parser.is_some()
            && detect_tool_call_start(content, self.tool_call_parser.as_deref()).unwrap_or(false);

        tracing::debug!(
            "should_start_jail: content={:?}, sequence_match={}, tool_call_match={}, sequences={:?}",
            content,
            sequence_match,
            tool_call_match,
            self.jail_start_sequences
        );

        sequence_match || tool_call_match
    }

    /// Check if accumulated content should end jail
    async fn should_end_jail(&self, accumulated_content: &str) -> (bool, usize) {
        match &self.jail_mode {
            JailMode::MarkerBased => {
                // Path 1: End sequence detected
                let end_marker_info = if !self.jail_end_sequences.is_empty() {
                    self.jail_end_sequences.iter().find_map(|seq| {
                        accumulated_content
                            .find(seq)
                            .map(|pos| (pos + seq.len(), seq.clone()))
                    })
                } else {
                    None
                };

                // Path 2: Complete tool call(s) can be parsed (early exit)
                let early_exit = self.should_exit_jail_early(accumulated_content).await;

                if let Some((end_pos, _)) = end_marker_info {
                    (true, end_pos)
                } else if early_exit {
                    // For early exit, find where the complete tool call ends
                    if let Some(parser) = &self.tool_call_parser {
                        let tools_slice = self.tool_definitions.as_deref();
                        if let Ok((_, _)) = try_tool_call_parse_aggregate(
                            accumulated_content,
                            Some(parser),
                            tools_slice,
                        )
                        .await
                        {
                            let split_pos =
                                find_tool_call_end_position(accumulated_content, Some(parser));
                            (true, split_pos)
                        } else {
                            (false, accumulated_content.len())
                        }
                    } else {
                        (false, accumulated_content.len())
                    }
                } else {
                    (false, accumulated_content.len())
                }
            }
            JailMode::Immediate { format } => {
                // For tool_choice, check if we have valid complete JSON
                match format {
                    ToolChoiceFormat::SingleObject { .. } => {
                        // Expect single object: {"location": "Paris", "unit": "celsius"}
                        if let Ok(value) =
                            serde_json::from_str::<serde_json::Value>(accumulated_content)
                            && value.is_object()
                        {
                            return (true, accumulated_content.len());
                        }
                        (false, accumulated_content.len())
                    }
                    ToolChoiceFormat::ArrayOfTools => {
                        // Expect array: [{"name":"search","parameters":{...}}, ...]
                        if let Ok(value) =
                            serde_json::from_str::<serde_json::Value>(accumulated_content)
                            && let Some(arr) = value.as_array()
                            && !arr.is_empty()
                        {
                            return (true, accumulated_content.len());
                        }
                        (false, accumulated_content.len())
                    }
                }
            }
        }
    }

    /// Parse tool calls from accumulated content and create choice
    async fn create_tool_call_choice(
        &self,
        choice_index: u32,
        accumulated_content: &str,
        base_choice: &ChatChoiceStream,
        tool_call_offset: usize,
    ) -> ChatChoiceStream {
        match &self.jail_mode {
            JailMode::MarkerBased => {
                // Traditional marker-based tool call parsing
                let tools_slice = self.tool_definitions.as_deref();
                if let Ok((tool_calls, normal_text)) = try_tool_call_parse_aggregate(
                    accumulated_content,
                    self.tool_call_parser.as_deref(),
                    tools_slice,
                )
                .await
                    && !tool_calls.is_empty()
                {
                    // Convert to streaming format
                    let tool_call_chunks: Vec<ChatCompletionMessageToolCallChunk> = tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(idx, tool_call)| ChatCompletionMessageToolCallChunk {
                            index: (tool_call_offset + idx) as u32,
                            id: Some(tool_call.id),
                            r#type: Some(tool_call.r#type),
                            function: Some(FunctionCallStream {
                                name: Some(tool_call.function.name),
                                arguments: Some(tool_call.function.arguments),
                            }),
                        })
                        .collect();
                    // Create choice with tool calls
                    let choice = create_choice_stream(
                        choice_index,
                        Some(Role::Assistant),
                        normal_text.as_deref().unwrap_or(""),
                        Some(tool_call_chunks),
                        None,
                        None,
                        None,
                    );
                    return choice;
                }

                // No tool calls found or parsing failed, return content choice
                create_choice_stream(
                    choice_index,
                    Some(Role::Assistant),
                    accumulated_content,
                    None,
                    base_choice.finish_reason,
                    base_choice.stop_reason.clone(),
                    base_choice.logprobs.clone(),
                )
            }
            JailMode::Immediate { format } => {
                // tool_choice mode: parse JSON and convert to tool calls
                match self.parse_tool_choice_json(accumulated_content, format) {
                    Ok(tool_call_chunks) if !tool_call_chunks.is_empty() => create_choice_stream(
                        choice_index,
                        Some(Role::Assistant),
                        "",
                        Some(tool_call_chunks),
                        base_choice.finish_reason,
                        None,
                        base_choice.logprobs.clone(),
                    ),
                    Ok(_) | Err(_) => {
                        // Parsing failed, return as content
                        create_choice_stream(
                            choice_index,
                            Some(Role::Assistant),
                            accumulated_content,
                            None,
                            base_choice.finish_reason,
                            base_choice.stop_reason.clone(),
                            base_choice.logprobs.clone(),
                        )
                    }
                }
            }
        }
    }

    /// Helper to create a ChatCompletionMessageToolCallChunk
    fn create_tool_call_chunk(
        index: u32,
        name: String,
        arguments: String,
    ) -> ChatCompletionMessageToolCallChunk {
        ChatCompletionMessageToolCallChunk {
            index,
            id: Some(format!("call-{}", Uuid::new_v4())),
            r#type: Some(dynamo_async_openai::types::ChatCompletionToolType::Function),
            function: Some(FunctionCallStream {
                name: Some(name),
                arguments: Some(arguments),
            }),
        }
    }

    /// Parse tool_choice JSON output into tool call chunks
    fn parse_tool_choice_json(
        &self,
        json_content: &str,
        format: &ToolChoiceFormat,
    ) -> anyhow::Result<Vec<ChatCompletionMessageToolCallChunk>> {
        let parsed = serde_json::from_str::<serde_json::Value>(json_content)?;

        match format {
            ToolChoiceFormat::SingleObject { tool_name } => {
                // For named tool choice: JSON is the parameters object
                if parsed.is_object() {
                    Ok(vec![Self::create_tool_call_chunk(
                        0,
                        tool_name.clone(),
                        json_content.to_string(),
                    )])
                } else {
                    Ok(vec![])
                }
            }
            ToolChoiceFormat::ArrayOfTools => {
                // For required tool choice: JSON is array of {name, parameters}
                if let Some(array) = parsed.as_array() {
                    let chunks: Vec<ChatCompletionMessageToolCallChunk> = array
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, entry)| {
                            let name = entry.get("name")?.as_str()?.to_string();
                            let parameters = entry.get("parameters")?;
                            let args = serde_json::to_string(parameters).ok()?;
                            Some(Self::create_tool_call_chunk(idx as u32, name, args))
                        })
                        .collect();
                    Ok(chunks)
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Check if accumulated content contains complete tool calls that can be parsed
    /// Returns true if we should exit the jail early
    async fn should_exit_jail_early(&self, accumulated: &str) -> bool {
        if let Some(ref parser) = self.tool_call_parser {
            // Try to parse - if successful and we have complete tool calls, exit early
            let tools_slice = self.tool_definitions.as_deref();
            match try_tool_call_parse_aggregate(accumulated, Some(parser), tools_slice).await {
                Ok((tool_calls, _normal_text)) => {
                    let result = !tool_calls.is_empty();
                    return result;
                }
                Err(_e) => {}
            }
        }
        false
    }

    /// Post-processor that sets finish_reason to ToolCalls when tool calls were emitted
    /// This should be called after apply() to fix the finish_reason for tool call chunks
    fn fix_finish_reason<S>(
        input_stream: S,
        jail_mode: JailMode,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    {
        stream! {
            tokio::pin!(input_stream);
            let mut has_tool_calls_per_choice: HashMap<u32, bool> = HashMap::new();

            while let Some(mut response) = input_stream.next().await {
                // Track if any choice emitted tool calls
                if let Some(ref data) = response.data {
                    for choice in &data.choices {
                        if choice.delta.tool_calls.is_some() {
                            has_tool_calls_per_choice.insert(choice.index, true);
                        }
                    }
                }

                // Fix finish_reason based on jail mode and whether tool calls were emitted
                if let Some(ref mut data) = response.data {
                    for choice in &mut data.choices {
                        if let Some(finish) = choice.finish_reason {
                            // Only modify Stop finish reason, preserve Length/ContentFilter
                            if finish == FinishReason::Stop {
                                let has_tool_calls = has_tool_calls_per_choice.get(&choice.index).copied().unwrap_or(false);

                                match &jail_mode {
                                    JailMode::MarkerBased => {
                                        // Traditional: if tool calls emitted, change to ToolCalls
                                        if has_tool_calls {
                                            choice.finish_reason = Some(FinishReason::ToolCalls);
                                        }
                                    }
                                    JailMode::Immediate { format } => {
                                        // tool_choice mode: apply specific finish_reason logic
                                        match format {
                                            ToolChoiceFormat::SingleObject { .. } => {
                                                // Named tool choice: keep Stop
                                                // (already Stop, no change needed)
                                            }
                                            ToolChoiceFormat::ArrayOfTools => {
                                                // Required tool choice: change to ToolCalls
                                                if has_tool_calls {
                                                    choice.finish_reason = Some(FinishReason::ToolCalls);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Length and ContentFilter are preserved as-is
                        }
                    }
                }

                yield response;
            }
        }
    }
}

/// Builder for configuring a JailedStream
pub struct JailedStreamBuilder {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    tool_definitions: Option<Vec<dynamo_parsers::tool_calling::ToolDefinition>>,
    emission_mode: EmissionMode,
    jail_mode: JailMode,
}

impl JailedStreamBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            jail_start_sequences: Vec::new(),
            jail_end_sequences: Vec::new(),
            tool_call_parser: None,
            tool_definitions: None,
            emission_mode: EmissionMode::default(),
            jail_mode: JailMode::MarkerBased,
        }
    }

    /// Add a sequence that triggers jailing when detected
    pub fn jail_start_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.jail_start_sequences.push(sequence.into());
        self
    }

    /// Add multiple sequences that trigger jailing when detected
    pub fn jail_start_sequences(
        mut self,
        sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.jail_start_sequences
            .extend(sequences.into_iter().map(Into::into));
        self
    }

    /// Add a sequence that ends jailing when detected
    pub fn jail_end_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.jail_end_sequences.push(sequence.into());
        self
    }

    /// Add multiple sequences that end jailing when detected
    pub fn jail_end_sequences(
        mut self,
        sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.jail_end_sequences
            .extend(sequences.into_iter().map(Into::into));
        self
    }

    /// Set the tool call parser to use for detection and parsing
    pub fn tool_call_parser(mut self, parser: impl Into<String>) -> Self {
        self.tool_call_parser = Some(parser.into());
        self
    }

    /// Set the tool definitions for runtime validation and parsing
    pub fn tool_definitions(
        mut self,
        tools: Vec<dynamo_parsers::tool_calling::ToolDefinition>,
    ) -> Self {
        self.tool_definitions = Some(tools);
        self
    }

    /// Set the emission mode for handling multiple choices
    pub fn emission_mode(mut self, mode: EmissionMode) -> Self {
        self.emission_mode = mode;
        self
    }

    /// Enable single choice per chunk emission for OpenAI compatibility
    pub fn single_choice_per_chunk(mut self) -> Self {
        self.emission_mode = EmissionMode::SingleChoicePerChunk;
        self
    }

    /// Enable packed emission mode (multiple choices per chunk)
    pub fn packed_emission(mut self) -> Self {
        self.emission_mode = EmissionMode::Packed;
        self
    }

    /// Enable immediate jail mode for tool_choice=named
    pub fn tool_choice_named(mut self, tool_name: String) -> Self {
        self.jail_mode = JailMode::Immediate {
            format: ToolChoiceFormat::SingleObject { tool_name },
        };
        self
    }

    /// Enable immediate jail mode for tool_choice=required
    pub fn tool_choice_required(mut self) -> Self {
        self.jail_mode = JailMode::Immediate {
            format: ToolChoiceFormat::ArrayOfTools,
        };
        self
    }

    /// Build the configured JailedStream
    pub fn build(mut self) -> JailedStream {
        // Auto-populate jail sequences from parser config if not manually configured
        if let Some(ref parser_name) = self.tool_call_parser {
            let parser_map = get_tool_parser_map();
            if let Some(config) = parser_map.get(parser_name.as_str()) {
                // Auto-populate start sequences if none configured
                if self.jail_start_sequences.is_empty() {
                    self.jail_start_sequences = config.parser_config.tool_call_start_tokens();
                }

                // Auto-populate end sequences if none configured
                if self.jail_end_sequences.is_empty() {
                    self.jail_end_sequences = config
                        .parser_config
                        .tool_call_end_tokens()
                        .iter()
                        .filter(|&s| !s.is_empty())
                        .cloned()
                        .collect();
                }
            }
        }

        // Collect all possible marker patterns for the MarkerMatcher
        let mut all_patterns = Vec::new();

        // Add configured start sequences (now auto-populated if needed)
        all_patterns.extend(self.jail_start_sequences.clone());

        // Add patterns from tool call parser if configured (for redundancy)
        if let Some(ref parser_name) = self.tool_call_parser {
            let parser_map = get_tool_parser_map();
            if let Some(config) = parser_map.get(parser_name.as_str()) {
                // Add start tokens from the parser config
                all_patterns.extend(config.parser_config.tool_call_start_tokens());
            }
        }

        // Add common tool call markers to ensure we detect all formats
        // Only include these when a specific parser is NOT configured,
        // to avoid unexpected false positives for explicit formats
        if self.tool_call_parser.is_none() {
            let common_markers = vec![
                "<TOOLCALL>".to_string(),     // nemotron_deci format
                "<tool_call>".to_string(),    // hermes format
                "[TOOL_CALLS]".to_string(),   // mistral format
                "<|python_tag|>".to_string(), // llama3_json format
                "functools[".to_string(),     // phi4 format
                // Add JSON start patterns for Mistral-style tool calls
                "[{".to_string(),
                "{".to_string(),
                // Note: Harmony parser uses JSON patterns, covered by "{" above
            ];
            for marker in common_markers {
                if !all_patterns.contains(&marker) {
                    all_patterns.push(marker);
                }
            }
        }

        // Create the marker matcher (fallback to empty patterns if none configured)
        let marker_matcher = if all_patterns.is_empty() {
            // If no patterns, create a dummy matcher that never matches
            MarkerMatcher::new(vec!["__NEVER_MATCH__".to_string()])
                .expect("Failed to create dummy MarkerMatcher")
        } else {
            tracing::debug!("Creating MarkerMatcher with patterns: {:?}", all_patterns);
            MarkerMatcher::new(all_patterns)
                .expect("Failed to create MarkerMatcher with configured patterns")
        };

        JailedStream {
            jail_start_sequences: self.jail_start_sequences,
            jail_end_sequences: self.jail_end_sequences,
            tool_call_parser: self.tool_call_parser,
            tool_definitions: self.tool_definitions,
            emission_mode: self.emission_mode,
            marker_matcher,
            jail_mode: self.jail_mode,
        }
    }
}

impl Default for JailedStreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

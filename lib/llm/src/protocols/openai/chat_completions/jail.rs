// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_stream::stream;
use dynamo_async_openai::types::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    FinishReason, FunctionCallStream, Role,
};

use dynamo_parsers::tool_calling::parsers::get_tool_parser_map;
use dynamo_parsers::tool_calling::{detect_tool_call_start, try_tool_call_parse_aggregate};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

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
}

impl ChoiceJailState {
    /// Create a new jail state for a choice
    fn new(index: u32) -> Self {
        Self {
            index,
            is_jailed: false,
            accumulated_content: String::new(),
            partial_match_buffer: String::new(),
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
    fn process_content(
        &mut self,
        choice: &ChatChoiceStream,
        content: &str,
        jail_stream: &JailedStream,
    ) -> Vec<ChoiceEmission> {
        let mut emissions = Vec::new();

        if !self.is_jailed {
            // Use the marker matcher to detect complete/partial markers
            match jail_stream
                .marker_matcher
                .process_chunk(content, &self.partial_match_buffer)
            {
                MatchResult::Complete {
                    prefix,
                    marker,
                    suffix,
                    ..
                } => {
                    // Emit prefix if any
                    if !prefix.is_empty() {
                        #[allow(deprecated)]
                        let prefix_choice = ChatChoiceStream {
                            index: choice.index,
                            delta: ChatCompletionStreamResponseDelta {
                                role: choice.delta.role,
                                content: Some(prefix),
                                tool_calls: None,
                                function_call: None,
                                refusal: None,
                                reasoning_content: None,
                            },
                            finish_reason: None,
                            logprobs: choice.logprobs.clone(),
                        };
                        emissions.push(ChoiceEmission::PassThrough(prefix_choice));
                    }

                    // Build the potential full content
                    let full_content = format!("{}{}", marker, suffix);

                    // Check if this already contains the end marker
                    let (should_unjail, split_pos) = jail_stream.should_end_jail(&full_content);

                    if should_unjail {
                        // Complete tool call found in this chunk
                        tracing::debug!(
                            "Choice {} complete tool call detected in single chunk",
                            choice.index
                        );

                        let (jailed_part, trailing_part) = full_content.split_at(split_pos);

                        // Create the tool call choice
                        let tool_choice =
                            jail_stream.create_tool_call_choice(choice.index, jailed_part, choice);

                        if tool_choice.delta.tool_calls.is_some() {
                            emissions.push(ChoiceEmission::ToolCall(tool_choice));
                        } else {
                            emissions.push(ChoiceEmission::Content(tool_choice));
                        }

                        // Handle trailing content if any
                        if !trailing_part.is_empty() {
                            #[allow(deprecated)]
                            let trailing_choice = ChatChoiceStream {
                                index: choice.index,
                                delta: ChatCompletionStreamResponseDelta {
                                    role: choice.delta.role,
                                    content: Some(trailing_part.to_string()),
                                    tool_calls: None,
                                    function_call: None,
                                    refusal: None,
                                    reasoning_content: None,
                                },
                                finish_reason: None,
                                logprobs: choice.logprobs.clone(),
                            };
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
                        let prefix_choice = ChatChoiceStream {
                            index: choice.index,
                            delta: ChatCompletionStreamResponseDelta {
                                role: choice.delta.role,
                                content: Some(prefix),
                                tool_calls: None,
                                function_call: None,
                                refusal: None,
                                reasoning_content: None,
                            },
                            finish_reason: None,
                            logprobs: choice.logprobs.clone(),
                        };
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
                    // No markers - emit everything
                    if !content.is_empty() {
                        #[allow(deprecated)]
                        let pass_through_choice = ChatChoiceStream {
                            index: choice.index,
                            delta: ChatCompletionStreamResponseDelta {
                                role: choice.delta.role,
                                content: Some(content),
                                tool_calls: None,
                                function_call: None,
                                refusal: None,
                                reasoning_content: None,
                            },
                            finish_reason: None,
                            logprobs: choice.logprobs.clone(),
                        };
                        emissions.push(ChoiceEmission::PassThrough(pass_through_choice));
                    }
                    self.partial_match_buffer.clear();
                }
            }
        } else {
            // Already jailed - accumulate and check for unjail
            self.accumulate(content);

            let (should_unjail, split_pos) = jail_stream.should_end_jail(&self.accumulated_content);

            if should_unjail {
                tracing::debug!(
                    "Choice {} jail exit detected, releasing accumulated content",
                    choice.index
                );

                // Split the content
                let (jailed_part, trailing_part) = self.accumulated_content.split_at(split_pos);

                // Create the unjailed choice
                let unjailed_choice =
                    jail_stream.create_tool_call_choice(choice.index, jailed_part, choice);

                // Determine emission type based on whether tool calls were parsed
                if unjailed_choice.delta.tool_calls.is_some() {
                    emissions.push(ChoiceEmission::ToolCall(unjailed_choice));
                } else {
                    emissions.push(ChoiceEmission::Content(unjailed_choice));
                }

                // Handle trailing content if any
                if !trailing_part.is_empty() {
                    #[allow(deprecated)]
                    let trailing_choice = ChatChoiceStream {
                        index: choice.index,
                        delta: ChatCompletionStreamResponseDelta {
                            role: choice.delta.role,
                            content: Some(trailing_part.to_string()),
                            tool_calls: None,
                            function_call: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                        logprobs: choice.logprobs.clone(),
                    };
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
    fn finalize(&mut self, jail_stream: &JailedStream) -> Option<ChoiceEmission> {
        if self.is_jailed && !self.accumulated_content.is_empty() {
            tracing::debug!(
                "Choice {} stream ended while jailed, releasing accumulated content",
                self.index
            );

            // Create a dummy choice for the method call
            #[allow(deprecated)]
            let dummy_choice = ChatChoiceStream {
                index: self.index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: None,
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            };

            let final_choice = jail_stream.create_tool_call_choice(
                self.index,
                &self.accumulated_content,
                &dummy_choice,
            );

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
    fn get_or_create_state(&mut self, index: u32) -> &mut ChoiceJailState {
        // Find the position where this index should be
        match self.states.binary_search_by_key(&index, |s| s.index) {
            Ok(pos) => {
                // Found existing state
                &mut self.states[pos]
            }
            Err(insert_pos) => {
                // Need to create new state
                let new_state = ChoiceJailState::new(index);
                self.states.insert(insert_pos, new_state);
                &mut self.states[insert_pos]
            }
        }
    }
}

/// Emission mode for handling multiple choices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmissionMode {
    /// Pack multiple choices in the same chunk (default, matches original behavior)
    Packed,
    /// Emit one choice per chunk for OpenAI compatibility
    SingleChoicePerChunk,
}

impl Default for EmissionMode {
    fn default() -> Self {
        Self::Packed
    }
}

/// A stream transformer that can "jail" tokens based on configurable start/end sequences
/// When jailed, tokens are accumulated rather than yielded immediately
/// When the jail ends (via end sequence or stream completion), accumulated content is processed and released
pub struct JailedStream {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    emission_mode: EmissionMode,
    marker_matcher: MarkerMatcher,
}

impl JailedStream {
    /// Create a new builder for configuring a JailedStream
    pub fn builder() -> JailedStreamBuilder {
        JailedStreamBuilder::new()
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

                    // Process each choice independently using the new architecture
                    for choice in &chat_response.choices {
                        if let Some(ref content) = choice.delta.content {
                            let choice_state = choice_states.get_or_create_state(choice.index);

                            // Store metadata when any choice becomes jailed (first time only)
                            if !choice_state.is_jailed && self.should_start_jail(content)
                                && last_annotated_id.is_none() {
                                    last_annotated_id = response.id.clone();
                                    last_annotated_event = response.event.clone();
                                    last_annotated_comment = response.comment.clone();
                                }

                            // Process this choice and get emissions
                            let emissions = choice_state.process_content(choice, content, &self);
                            all_emissions.extend(emissions);
                        } else {
                            // Handle choices without content (e.g., final chunks with finish_reason)
                            // These should always pass through
                            let pass_through_choice = ChatChoiceStream {
                                index: choice.index,
                                delta: choice.delta.clone(),
                                finish_reason: choice.finish_reason,
                                logprobs: choice.logprobs.clone(),
                            };
                            all_emissions.push(ChoiceEmission::PassThrough(pass_through_choice));
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
                if let Some(emission) = state.finalize(&self) {
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

        sequence_match || tool_call_match
    }

    /// Check if accumulated content should end jail
    fn should_end_jail(&self, accumulated_content: &str) -> (bool, usize) {
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
        let early_exit = self.should_exit_jail_early(accumulated_content);

        if let Some((end_pos, _)) = end_marker_info {
            (true, end_pos)
        } else if early_exit {
            // For early exit, find where the complete tool call ends
            if let Some(parser) = &self.tool_call_parser {
                if let Ok((_, _)) = try_tool_call_parse_aggregate(accumulated_content, Some(parser))
                {
                    let split_pos = self.find_tool_call_end_position(accumulated_content, parser);
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

    /// Parse tool calls from accumulated content and create choice
    fn create_tool_call_choice(
        &self,
        choice_index: u32,
        accumulated_content: &str,
        base_choice: &ChatChoiceStream,
    ) -> ChatChoiceStream {
        if let Ok((tool_calls, normal_text)) =
            try_tool_call_parse_aggregate(accumulated_content, self.tool_call_parser.as_deref())
            && !tool_calls.is_empty()
        {
            // Convert to streaming format
            let tool_call_chunks: Vec<ChatCompletionMessageToolCallChunk> = tool_calls
                .into_iter()
                .enumerate()
                .map(|(idx, tool_call)| ChatCompletionMessageToolCallChunk {
                    index: idx as u32,
                    id: Some(tool_call.id),
                    r#type: Some(tool_call.r#type),
                    function: Some(FunctionCallStream {
                        name: Some(tool_call.function.name),
                        arguments: Some(tool_call.function.arguments),
                    }),
                })
                .collect();

            // Create choice with tool calls
            #[allow(deprecated)]
            return ChatChoiceStream {
                index: choice_index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: normal_text.filter(|t| !t.is_empty()),
                    tool_calls: Some(tool_call_chunks),
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::ToolCalls),
                logprobs: None,
            };
        }

        // No tool calls found or parsing failed, return content choice
        #[allow(deprecated)]
        ChatChoiceStream {
            index: choice_index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(accumulated_content.to_string()),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: base_choice.logprobs.clone(),
        }
    }

    /// Check if accumulated content contains complete tool calls that can be parsed
    /// Returns true if we should exit the jail early
    fn should_exit_jail_early(&self, accumulated: &str) -> bool {
        if let Some(ref parser) = self.tool_call_parser {
            // Try to parse - if successful and we have complete tool calls, exit early
            if let Ok((tool_calls, _)) = try_tool_call_parse_aggregate(accumulated, Some(parser)) {
                return !tool_calls.is_empty();
            }
        }
        false
    }

    /// Find the exact position where the tool call ends for splitting content
    /// This handles the early exit case where we have trailing content after the tool call
    fn find_tool_call_end_position(&self, content: &str, parser: &str) -> usize {
        match parser {
            "hermes" => {
                // For Hermes, look for </tool_call> marker
                if let Some(pos) = content.find("</tool_call>") {
                    pos + "</tool_call>".len()
                } else {
                    content.len()
                }
            }
            "nemotron_deci" => {
                // For Nemotron, look for </TOOLCALL> marker
                if let Some(pos) = content.find("</TOOLCALL>") {
                    pos + "</TOOLCALL>".len()
                } else {
                    content.len()
                }
            }
            "mistral" => {
                // For Mistral, look for [/TOOL_CALLS] marker or end of JSON array
                if let Some(pos) = content.find("[/TOOL_CALLS]") {
                    pos + "[/TOOL_CALLS]".len()
                } else if let Some(pos) = content.rfind(']') {
                    // Find the last ] which should be the end of the tool calls array
                    pos + 1
                } else {
                    content.len()
                }
            }
            "phi4" => {
                // For Phi4, look for <|tool_call|> end marker
                if let Some(pos) = content.rfind("<|tool_call|>") {
                    // Look for the next occurrence after this position
                    if let Some(end_pos) = content[pos..].find(">") {
                        pos + end_pos + 1
                    } else {
                        content.len()
                    }
                } else {
                    content.len()
                }
            }
            "llama3_json" => {
                // For Llama3 JSON, there's no explicit end marker
                // The end is determined by complete JSON parsing
                // Return full content length to avoid early splitting
                content.len()
            }
            _ => {
                // Unknown parser, default to full content
                content.len()
            }
        }
    }
}

/// Builder for configuring a JailedStream
pub struct JailedStreamBuilder {
    jail_start_sequences: Vec<String>,
    jail_end_sequences: Vec<String>,
    tool_call_parser: Option<String>,
    emission_mode: EmissionMode,
}

impl JailedStreamBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            jail_start_sequences: Vec::new(),
            jail_end_sequences: Vec::new(),
            tool_call_parser: None,
            emission_mode: EmissionMode::default(),
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

    /// Build the configured JailedStream
    pub fn build(self) -> JailedStream {
        // Collect all possible marker patterns for the MarkerMatcher
        let mut all_patterns = Vec::new();

        // Add configured start sequences
        all_patterns.extend(self.jail_start_sequences.clone());

        // Add patterns from tool call parser if configured
        if let Some(ref parser_name) = self.tool_call_parser {
            let parser_map = get_tool_parser_map();
            if let Some(config) = parser_map.get(parser_name.as_str()) {
                // Add start tokens from the parser config
                all_patterns.extend(config.json.tool_call_start_tokens.clone());
            }
        }

        // Add common tool call markers to ensure we detect all formats
        let common_markers = vec![
            "<TOOLCALL>".to_string(),
            "<tool_call>".to_string(),
            "[TOOL_CALLS]".to_string(),
            "<|python_tag|>".to_string(),
            "functools[".to_string(),
            // Add JSON start patterns for Mistral-style tool calls
            "[{".to_string(),
            "{".to_string(),
        ];
        for marker in common_markers {
            if !all_patterns.contains(&marker) {
                all_patterns.push(marker);
            }
        }

        // Create the marker matcher (fallback to empty patterns if none configured)
        let marker_matcher = if all_patterns.is_empty() {
            // If no patterns, create a dummy matcher that never matches
            MarkerMatcher::new(vec!["__NEVER_MATCH__".to_string()])
                .expect("Failed to create dummy MarkerMatcher")
        } else {
            MarkerMatcher::new(all_patterns)
                .expect("Failed to create MarkerMatcher with configured patterns")
        };

        JailedStream {
            jail_start_sequences: self.jail_start_sequences,
            jail_end_sequences: self.jail_end_sequences,
            tool_call_parser: self.tool_call_parser,
            emission_mode: self.emission_mode,
            marker_matcher,
        }
    }
}

impl Default for JailedStreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::stream;

    // Test utilities module - shared test infrastructure
    pub(crate) mod test_utils {
        use super::*;

        /// Helper function to create a mock chat response chunk
        pub fn create_mock_response_chunk(
            content: String,
            index: u32,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some(content),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper function to create a final response chunk with finish reason
        pub fn create_final_response_chunk(
            index: u32,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper function to create a mock chat response chunk with metadata
        pub fn create_annotated_chunk(
            content: String,
            index: u32,
            id: Option<String>,
            event: Option<String>,
            comment: Option<Vec<String>>,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some(content),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            };

            Annotated {
                data: Some(response),
                id,
                event,
                comment,
            }
        }

        /// Helper function to create a multi-choice chunk
        pub fn create_multi_choice_chunk(
            choices_content: Vec<(String, u32)>, // (content, index)
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            let choices: Vec<ChatChoiceStream> = choices_content
                .into_iter()
                .map(|(content, index)| {
                    #[allow(deprecated)]
                    ChatChoiceStream {
                        index,
                        delta: ChatCompletionStreamResponseDelta {
                            role: Some(Role::Assistant),
                            content: Some(content),
                            tool_calls: None,
                            function_call: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }
                })
                .collect();

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices,
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper to assert content in a result
        pub fn assert_content(
            result: &Annotated<NvCreateChatCompletionStreamResponse>,
            expected: &str,
        ) {
            let content = result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .expect("Expected content in result");

            assert_eq!(
                content, expected,
                "Content mismatch: expected '{}', got '{}'",
                expected, content
            );
        }

        /// Helper to assert a tool call in a result
        pub fn assert_tool_call(
            result: &Annotated<NvCreateChatCompletionStreamResponse>,
            name: &str,
            args: serde_json::Value,
        ) {
            let tool_calls = result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .expect("Expected tool calls in result");

            assert!(!tool_calls.is_empty(), "Expected at least one tool call");

            let tool_call = &tool_calls[0];
            let function = tool_call
                .function
                .as_ref()
                .expect("Expected function in tool call");

            assert_eq!(
                function.name.as_deref(),
                Some(name),
                "Tool call name mismatch: expected '{}', got '{:?}'",
                name,
                function.name
            );

            if let Some(arguments_str) = &function.arguments {
                let parsed_args: serde_json::Value = serde_json::from_str(arguments_str)
                    .expect("Tool call arguments should be valid JSON");
                assert_eq!(
                    parsed_args, args,
                    "Tool call arguments mismatch: expected {}, got {}",
                    args, parsed_args
                );
            } else if !args.is_null() {
                panic!("Expected tool call arguments {} but got None", args);
            }
        }

        /// Helper to assert no content or tool calls (for accumulated chunks)
        pub fn assert_empty_emission(result: &Annotated<NvCreateChatCompletionStreamResponse>) {
            if let Some(data) = &result.data {
                if let Some(choice) = data.choices.first() {
                    assert!(
                        choice.delta.content.is_none()
                            || choice.delta.content.as_ref().unwrap().is_empty(),
                        "Expected no content but got: {:?}",
                        choice.delta.content
                    );
                    assert!(
                        choice.delta.tool_calls.is_none()
                            || choice.delta.tool_calls.as_ref().unwrap().is_empty(),
                        "Expected no tool calls but got: {:?}",
                        choice.delta.tool_calls
                    );
                }
            }
        }

        /// Helper to reconstruct all content from results
        pub fn reconstruct_content(
            results: &[Annotated<NvCreateChatCompletionStreamResponse>],
        ) -> String {
            results
                .iter()
                .filter_map(|r| {
                    r.data
                        .as_ref()
                        .and_then(|d| d.choices.first())
                        .and_then(|c| c.delta.content.as_ref())
                })
                .cloned()
                .collect::<Vec<_>>()
                .join("")
        }

        /// Helper to extract content from a single result (for negative assertions)
        pub fn extract_content(result: &Annotated<NvCreateChatCompletionStreamResponse>) -> String {
            result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .cloned()
                .unwrap_or_default()
        }

        /// Helper to check if result contains a tool call
        pub fn has_tool_call(result: &Annotated<NvCreateChatCompletionStreamResponse>) -> bool {
            result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        }

        /// Helper to check if result contains content
        pub fn has_content(result: &Annotated<NvCreateChatCompletionStreamResponse>) -> bool {
            result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| !content.is_empty())
                .unwrap_or(false)
        }
    }

    use serde_json::json;
    use test_utils::*;

    #[tokio::test]
    async fn test_jailed_stream_with_start_end_sequences() {
        // Create chunks with jail start/end markers
        let chunks = vec![
            create_mock_response_chunk("Hello ".to_string(), 0),
            create_mock_response_chunk("<jail>".to_string(), 0),
            create_mock_response_chunk("This is jailed ".to_string(), 0),
            create_mock_response_chunk("content".to_string(), 0),
            create_mock_response_chunk("</jail>".to_string(), 0),
            create_mock_response_chunk(" World".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with start/end sequences
        let jail = JailedStream::builder()
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // We should only get 3 chunks now:
        // 1. "Hello " (before jail)
        // 2. Accumulated jailed content when jail ends
        // 3. " World" (after jail)
        assert_eq!(results.len(), 3);

        // First chunk should pass through
        assert_eq!(
            results[0].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("Hello ")
        );

        // When jail ends, accumulated content should be released
        let unjailed_content = &results[1].data.as_ref().unwrap().choices[0].delta.content;
        assert!(unjailed_content.is_some());
        assert!(
            unjailed_content
                .as_ref()
                .unwrap()
                .contains("<jail>This is jailed content</jail>")
        );

        // Last chunk should pass through normally
        assert_eq!(
            results[2].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some(" World")
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_with_tool_calls() {
        // Create chunks representing a tool call
        let chunks = vec![
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with tool call parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have jailed the content and parsed tool calls at the end
        assert!(!results.is_empty());

        // Check if tool calls were parsed
        if let Some(last_result) = results.last()
            && let Some(ref response_data) = last_result.data
            && let Some(ref tool_calls) = response_data.choices[0].delta.tool_calls
        {
            assert!(!tool_calls.as_slice().is_empty());
            assert_eq!(
                tool_calls[0].function.as_ref().unwrap().name.as_deref(),
                Some("get_weather")
            );
        }
    }

    #[tokio::test]
    async fn test_jailed_stream_dual_entry_paths() {
        // Test that BOTH sequence AND tool call detection can trigger jail
        let chunks = vec![
            create_mock_response_chunk("Normal text ".to_string(), 0),
            create_mock_response_chunk("<jail><TOOLCALL>".to_string(), 0), // Both triggers
            create_mock_response_chunk("Jailed content".to_string(), 0),
            create_mock_response_chunk("</jail>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Configure with both sequences AND tool call parser
        let jail = JailedStream::builder()
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // First chunk should pass through
        assert_eq!(
            results[0].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("Normal text ")
        );

        // Jail should trigger and accumulate
        assert!(results.len() >= 2);
    }

    #[tokio::test]
    async fn test_jailed_stream_early_exit() {
        // Tests detection of complete tool call with unjail in same chunk as the end marker
        // Input: "<TOOLCALL>" + "[{\"name\": \"test\", " + "\"arguments\": {}}]" + "</TOOLCALL>More text"
        // Expected output: 2 chunks [ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"test\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {}}]".to_string(), 0),
            create_mock_response_chunk("</TOOLCALL>More text".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have exactly 2 chunks: tool call + trailing content
        assert_eq!(
            results.len(),
            2,
            "Should have tool call and trailing content"
        );

        // Verify exact output structure: [ToolCall(), Content()]
        test_utils::assert_tool_call(&results[0], "test", serde_json::json!({}));
        test_utils::assert_content(&results[1], "More text");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "More text");
    }

    #[tokio::test]
    async fn test_jailed_stream_no_jailing() {
        // Input chunks:
        // [0] "Hello "
        // [1] "World"
        // [2] [final chunk]
        //
        // Expected output (pass-through):
        // [0] Content("Hello ")
        // [1] Content("World")
        // [2] [final chunk]
        let chunks = vec![
            create_mock_response_chunk("Hello ".to_string(), 0),
            create_mock_response_chunk("World".to_string(), 0),
            create_final_response_chunk(0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with sequences that won't match
        let jail = JailedStream::builder()
            .jail_start_sequence("<NOTPRESENT>")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            3,
            "Should pass through all 3 chunks unchanged"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "Hello ");
        assert_content(&results[1], "World");
        // results[2] is the final chunk - no content to verify

        // === Verify negative assertions ===
        for (i, result) in results.iter().take(2).enumerate() {
            assert!(
                !has_tool_call(result),
                "Chunk {} should not contain tool calls when no patterns match",
                i
            );
        }

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "Hello World",
            "Content should pass through unchanged when no jailing occurs"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_hermes_parser() {
        // Tests Hermes format tool call parsing with <tool_call> markers
        // Input: "I'll help you with that. " + "<tool_call>{\"name\": \"search_web\", \"arguments\": {\"query\": \"weather today\"}}</tool_call>" + " Let me search for that."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("I'll help you with that. ".to_string(), 0),
            create_mock_response_chunk("<tool_call>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"search_web\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"query\": \"weather today\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
            create_mock_response_chunk(" Let me search for that.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "I'll help you with that. ");
        test_utils::assert_tool_call(
            &results[1],
            "search_web",
            serde_json::json!({"query": "weather today"}),
        );
        test_utils::assert_content(&results[2], " Let me search for that.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "I'll help you with that.  Let me search for that."
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_mistral_parser() {
        // Tests Mistral format tool call parsing with [{ pattern
        // Input: "Sure, I can help. " + "[{\"name\": \"calculate\", \"arguments\": {\"expression\": \"2+2\"}}]" + " The calculation is done."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Sure, I can help. ".to_string(), 0),
            create_mock_response_chunk("[{".to_string(), 0),
            create_mock_response_chunk("\"name\": \"calculate\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"expression\": \"2+2\"}".to_string(), 0),
            create_mock_response_chunk("}]".to_string(), 0),
            create_mock_response_chunk(" The calculation is done.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Mistral parser
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Sure, I can help. ");
        test_utils::assert_tool_call(
            &results[1],
            "calculate",
            serde_json::json!({"expression": "2+2"}),
        );
        test_utils::assert_content(&results[2], " The calculation is done.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "Sure, I can help.  The calculation is done.");
    }

    #[tokio::test]
    async fn test_jailed_stream_mistral_parser_with_tool_calls_marker() {
        // Tests Mistral format tool call parsing with explicit [TOOL_CALLS] marker
        // Input: "Let me check that for you. " + "[TOOL_CALLS][{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}]" + " Here's the time."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Let me check that for you. ".to_string(), 0),
            create_mock_response_chunk("[TOOL_CALLS]".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"get_time\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"timezone\": \"UTC\"}}]".to_string(), 0),
            create_mock_response_chunk(" Here's the time.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Mistral parser
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Let me check that for you. ");
        test_utils::assert_tool_call(
            &results[1],
            "get_time",
            serde_json::json!({"timezone": "UTC"}),
        );
        test_utils::assert_content(&results[2], " Here's the time.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Let me check that for you.  Here's the time."
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_phi4_parser() {
        // Tests Phi4 format tool call parsing with functools[ pattern
        // Input: "I'll analyze this data. " + "functools[{\"name\": \"analyze_data\", \"arguments\": {\"dataset\": \"sales_data\"}}]" + " Analysis complete."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("I'll analyze this data. ".to_string(), 0),
            create_mock_response_chunk("functools[".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"analyze_data\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"dataset\": \"sales_data\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk("]".to_string(), 0),
            create_mock_response_chunk(" Analysis complete.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Phi4 parser
        let jail = JailedStream::builder().tool_call_parser("phi4").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "I'll analyze this data. ");
        test_utils::assert_tool_call(
            &results[1],
            "analyze_data",
            serde_json::json!({"dataset": "sales_data"}),
        );
        test_utils::assert_content(&results[2], " Analysis complete.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "I'll analyze this data.  Analysis complete.");
    }

    #[tokio::test]
    async fn test_jailed_stream_llama3_json_parser() {
        // Tests Llama3 JSON format tool call parsing with <|python_tag|> pattern
        // Input: "Let me run some code. " + "<|python_tag|>{\"name\": \"execute_code\", \"arguments\": {\"code\": \"print('Hello')\"}}" + " Done executing."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Let me run some code. ".to_string(), 0),
            create_mock_response_chunk("<|python_tag|>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"execute_code\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"code\": \"print('Hello')\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk(" Done executing.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with llama3_json parser
        let jail = JailedStream::builder()
            .tool_call_parser("llama3_json")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Let me run some code. ");
        test_utils::assert_tool_call(
            &results[1],
            "execute_code",
            serde_json::json!({"code": "print('Hello')"}),
        );
        test_utils::assert_content(&results[2], " Done executing.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "Let me run some code.  Done executing.");
    }

    #[tokio::test]
    async fn test_jailed_stream_false_positive_json() {
        // Tests that JSON-like content doesn't trigger false positive tool call detection
        // Input: "I can explain JSON format. " + "Here's an example: { \"key\": \"value\" }" + " is a simple JSON object. " + "Hope that helps!"
        // Expected output: 4 chunks [Content(), Content(), Content(), Content()] - no jailing
        let chunks = vec![
            create_mock_response_chunk("I can explain JSON format. ".to_string(), 0),
            create_mock_response_chunk("Here's an example: { \"key\": \"value\" }".to_string(), 0),
            create_mock_response_chunk(" is a simple JSON object. ".to_string(), 0),
            create_mock_response_chunk("Hope that helps!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with mistral parser (which specifically looks for [{ or [TOOL_CALLS] patterns)
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // The "{" pattern triggers jailing, so some chunks get combined
        assert_eq!(
            results.len(),
            3,
            "Should handle {{ pattern jailing and combine chunks appropriately"
        );

        // Verify exact output structure: content chunks
        test_utils::assert_content(&results[0], "I can explain JSON format. ");
        test_utils::assert_content(&results[1], "Here's an example: ");
        test_utils::assert_content(
            &results[2],
            "{ \"key\": \"value\" } is a simple JSON object. Hope that helps!",
        );

        // Verify no tool calls were detected and all content preserved
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "I can explain JSON format. Here's an example: { \"key\": \"value\" } is a simple JSON object. Hope that helps!"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_malformed_tool_call() {
        // Tests graceful handling of malformed JSON within tool call markers
        // Input: "Let me call a function. " + "<TOOLCALL>[{\"name\": \"broken_func\", \"arguments\": {\"param\": incomplete</TOOLCALL>" + " Function call attempt finished."
        // Expected output: 3 chunks [Content(), Content(malformed), Content()] - parser fails gracefully
        let chunks = vec![
            create_mock_response_chunk("Let me call a function. ".to_string(), 0),
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"broken_func\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"param\": incomplete".to_string(), 0), // Malformed JSON
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Function call attempt finished.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with nemotron_deci parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Jailing combines the tool call content into fewer chunks
        assert_eq!(
            results.len(),
            2,
            "Should handle malformed JSON gracefully and jail appropriately"
        );

        // Verify exact output structure: [Content(), Content(complete jailed content)]
        test_utils::assert_content(&results[0], "Let me call a function. ");
        test_utils::assert_content(
            &results[1],
            "<TOOLCALL>[{\"name\": \"broken_func\", \"arguments\": {\"param\": incomplete</TOOLCALL> Function call attempt finished.",
        );

        // Verify malformed content is preserved as text (including markers when parsing fails)
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Let me call a function. <TOOLCALL>[{\"name\": \"broken_func\", \"arguments\": {\"param\": incomplete</TOOLCALL> Function call attempt finished."
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_partial_tool_call() {
        // Tests handling of incomplete tool call when stream ends abruptly
        // Input: "Starting function call. " + "<TOOLCALL>[{\"name\": \"incomplete_func\", \"arguments\": {" (no end marker)
        // Expected output: 2 chunks [Content(), Content(partial)] - partial accumulated content released on stream end
        let chunks = vec![
            create_mock_response_chunk("Starting function call. ".to_string(), 0),
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"incomplete_func\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {".to_string(), 0),
            // Stream ends abruptly without closing the tool call
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with nemotron_deci parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should handle partial tool call gracefully - releases accumulated content on stream end
        assert_eq!(
            results.len(),
            2,
            "Should handle partial tool call and release content"
        );

        // Verify exact output structure: [Content(), Content(accumulated partial)]
        test_utils::assert_content(&results[0], "Starting function call. ");
        test_utils::assert_content(
            &results[1],
            "<TOOLCALL>[{\"name\": \"incomplete_func\", \"arguments\": {",
        );

        // Verify partial content is preserved as text since no valid tool call could be parsed
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Starting function call. <TOOLCALL>[{\"name\": \"incomplete_func\", \"arguments\": {"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_empty_stream() {
        // Input chunks: []
        //
        // Expected output: []
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];
        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            0,
            "Empty stream should produce exactly 0 results"
        );

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "",
            "Empty stream should reconstruct to empty string"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_multiple_tool_calls() {
        // Input chunks: 9 chunks for 2 tool calls with content between
        //
        // Expected output:
        // [0] Content("I'll help with multiple tasks. ")
        // [1] ToolCall("get_weather", {"city": "NYC"})
        // [2] Content(" Now let me get the time. ")
        // [3] ToolCall("get_time", {"timezone": "EST"})
        // [4] Content(" Both tasks completed!")
        let chunks = vec![
            create_mock_response_chunk("I'll help with multiple tasks. ".to_string(), 0),
            // First tool call
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Now let me get the time. ".to_string(), 0),
            // Second tool call
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"EST\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Both tasks completed!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            5,
            "Should emit exactly 5 chunks as documented above"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "I'll help with multiple tasks. ");
        assert_tool_call(&results[1], "get_weather", json!({"city": "NYC"}));
        assert_content(&results[2], " Now let me get the time. ");
        assert_tool_call(&results[3], "get_time", json!({"timezone": "EST"}));
        assert_content(&results[4], " Both tasks completed!");

        // === Verify content reconstruction ===
        let expected_content =
            "I'll help with multiple tasks.  Now let me get the time.  Both tasks completed!";
        assert_eq!(
            reconstruct_content(&results),
            expected_content,
            "Content reconstruction should exclude tool calls and preserve text flow"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_tool_call_across_many_chunks() {
        // Tests extreme fragmentation: tool call split across 65 individual character chunks
        // Input: "I'll process your request. " + "<TOOLCALL>[{"name": "process_data", "arguments": {}}]</TOOLCALL>" + " Processing complete!"
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("I'll process your request. ".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk("T".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("C".to_string(), 0),
            create_mock_response_chunk("A".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk(">".to_string(), 0),
            create_mock_response_chunk("[".to_string(), 0),
            create_mock_response_chunk("{".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("n".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("m".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(":".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("p".to_string(), 0),
            create_mock_response_chunk("r".to_string(), 0),
            create_mock_response_chunk("o".to_string(), 0),
            create_mock_response_chunk("c".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("_".to_string(), 0),
            create_mock_response_chunk("d".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("t".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(",".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("r".to_string(), 0),
            create_mock_response_chunk("g".to_string(), 0),
            create_mock_response_chunk("u".to_string(), 0),
            create_mock_response_chunk("m".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("n".to_string(), 0),
            create_mock_response_chunk("t".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(":".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("{".to_string(), 0),
            create_mock_response_chunk("}".to_string(), 0),
            create_mock_response_chunk("}".to_string(), 0),
            create_mock_response_chunk("]".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk("/".to_string(), 0),
            create_mock_response_chunk("T".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("C".to_string(), 0),
            create_mock_response_chunk("A".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk(">".to_string(), 0),
            create_mock_response_chunk(" Processing complete!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should consolidate extreme fragmentation into 3 clean chunks
        // Input: "I'll process your request. " + 54-char tool call + " Processing complete!"
        // Expected output: [Content(), ToolCall(), Content()]
        assert_eq!(
            results.len(),
            3,
            "Should consolidate fragments into 3 chunks"
        );

        // Verify exact output structure
        test_utils::assert_content(&results[0], "I'll process your request. ");
        test_utils::assert_tool_call(&results[1], "process_data", serde_json::json!({}));
        test_utils::assert_content(&results[2], " Processing complete!");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "I'll process your request.  Processing complete!"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_preserves_metadata() {
        // Test metadata preservation through jail processing
        let test_id = Some("correlation-id-123".to_string());
        let test_event = Some("request-processing".to_string());
        let test_comment = Some(vec![
            "upstream-correlation".to_string(),
            "debug-info".to_string(),
        ]);

        // Create chunks with specific metadata for the jail trigger
        let chunks = vec![
            create_annotated_chunk(
                "I'll help you with that. ".to_string(),
                0,
                None, // No metadata on first chunk
                None,
                None,
            ),
            create_annotated_chunk(
                "<tool_call>".to_string(),
                0,
                test_id.clone(), // Metadata on jail trigger chunk
                test_event.clone(),
                test_comment.clone(),
            ),
            create_mock_response_chunk("{\"name\": \"search_web\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"query\": \"test\"}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
            create_mock_response_chunk(" Processing complete.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should get 3 chunks: before jail, tool call response, after jail
        assert!(
            results.len() >= 3,
            "Should have at least 3 chunks, got {}",
            results.len()
        );

        // Find the synthesized tool call response chunk
        let tool_call_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .map(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                    .unwrap_or(false)
            })
            .expect("Should have a tool call response chunk");

        // Verify metadata is preserved
        assert_eq!(
            tool_call_chunk.id, test_id,
            "ID should be preserved from jail trigger chunk"
        );
        assert_eq!(
            tool_call_chunk.event, test_event,
            "Event should be preserved from jail trigger chunk"
        );
        assert_eq!(
            tool_call_chunk.comment, test_comment,
            "Comment should be preserved from jail trigger chunk"
        );

        // Verify tool call was parsed correctly
        let tool_calls = &tool_call_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls;
        assert!(tool_calls.is_some(), "Should have tool calls");
        let tool_calls = tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1, "Should have exactly one tool call");
        assert_eq!(
            tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .name
                .as_ref()
                .unwrap(),
            "search_web"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_preserves_metadata_on_stream_end() {
        // Test metadata preservation when stream ends while jailed
        let test_id = Some("end-correlation-456".to_string());
        let test_event = Some("stream-termination".to_string());
        let test_comment = Some(vec!["incomplete-processing".to_string()]);

        // Create chunks that end while jailed (no explicit end marker)
        let chunks = vec![
            create_mock_response_chunk("Starting function call: ".to_string(), 0),
            create_annotated_chunk(
                "<tool_call>".to_string(), // This chunk triggers jail and has metadata
                0,
                test_id.clone(),
                test_event.clone(),
                test_comment.clone(),
            ),
            create_mock_response_chunk(
                "{\"name\": \"incomplete_call\"".to_string(), // No closing brace
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should get 2 chunks: first chunk passes through, stream end releases accumulated
        assert_eq!(results.len(), 2, "Should have exactly 2 chunks");

        // The second chunk is the accumulated content released when stream ended
        let accumulated_chunk = &results[1];

        // Verify metadata is preserved from the jail trigger
        assert_eq!(
            accumulated_chunk.id, test_id,
            "ID should be preserved when stream ends while jailed"
        );
        assert_eq!(
            accumulated_chunk.event, test_event,
            "Event should be preserved when stream ends while jailed"
        );
        assert_eq!(
            accumulated_chunk.comment, test_comment,
            "Comment should be preserved when stream ends while jailed"
        );

        // Verify accumulated content is returned
        let content = &accumulated_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .content;
        assert!(content.is_some(), "Should have accumulated content");
        let content = content.as_ref().unwrap();
        assert!(
            content.contains("<tool_call>"),
            "Should contain jail start marker in accumulated content"
        );
        assert!(
            content.contains("incomplete_call"),
            "Should contain accumulated incomplete content"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_metadata_edge_cases() {
        // Test edge cases: empty metadata, partial metadata, etc.
        let chunks = vec![
            create_annotated_chunk(
                "Text with ".to_string(),
                0,
                Some("".to_string()), // Empty string ID
                None,                 // No event
                Some(vec![]),         // Empty comment vector
            ),
            create_annotated_chunk(
                "<tool_call>".to_string(),
                0,
                None,                                 // No ID
                Some("partial-metadata".to_string()), // Only event
                None,                                 // No comment
            ),
            create_mock_response_chunk("{\"name\": \"test\", \"arguments\": {}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Find the tool call response
        let tool_call_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .map(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                    .unwrap_or(false)
            })
            .expect("Should have a tool call response chunk");

        // Verify partial metadata is preserved correctly
        assert_eq!(tool_call_chunk.id, None, "Should preserve None ID");
        assert_eq!(
            tool_call_chunk.event,
            Some("partial-metadata".to_string()),
            "Should preserve event"
        );
        assert_eq!(
            tool_call_chunk.comment, None,
            "Should preserve None comment"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_trailing_content_same_chunk() {
        // Input chunks:
        // [0] "I'll help you. "
        // [1] "<tool_call>"
        // [2] "{\"name\": \"search\", \"arguments\": {}}"
        // [3] "</tool_call>trailing text that should not be lost"
        //
        // Expected output:
        // [0] Content("I'll help you. ")
        // [1] ToolCall("search", {})
        // [2] Content("trailing text that should not be lost")
        let chunks = vec![
            create_mock_response_chunk("I'll help you. ".to_string(), 0),
            create_mock_response_chunk("<tool_call>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"search\", \"arguments\": {}}".to_string(), 0),
            // This chunk contains both the end marker AND trailing content
            create_mock_response_chunk(
                "</tool_call>trailing text that should not be lost".to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            3,
            "Should emit exactly 3 chunks as documented above"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "I'll help you. ");
        assert_tool_call(&results[1], "search", json!({}));
        assert_content(&results[2], "trailing text that should not be lost");

        // === Verify content reconstruction ===
        let expected_content = "I'll help you. trailing text that should not be lost";
        assert_eq!(
            reconstruct_content(&results),
            expected_content,
            "Content reconstruction should preserve initial and trailing text"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_early_exit_with_trailing() {
        // Tests early exit when complete tool call is detected in chunk that also contains trailing content
        // Input: "Starting task: " + "<tool_call>{\"name\": \"complete_task\", \"arguments\": {}}" + "</tool_call> Task completed successfully."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Starting task: ".to_string(), 0),
            create_mock_response_chunk(
                "<tool_call>{\"name\": \"complete_task\", \"arguments\": {}}".to_string(),
                0,
            ),
            // Early exit should happen here, but we also have trailing content
            create_mock_response_chunk("</tool_call> Task completed successfully.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have exactly 3 chunks: content + tool call + trailing
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Starting task: ");
        test_utils::assert_tool_call(&results[1], "complete_task", serde_json::json!({}));
        test_utils::assert_content(&results[2], " Task completed successfully.");

        // Verify content reconstruction excludes tool calls but preserves trailing
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Starting task:  Task completed successfully."
        );
    }

    #[tokio::test]
    async fn test_multiple_choices_independent_jailing() {
        // Test that different choices can jail and unjail independently
        // This test will FAIL with the current HashMap-based implementation
        let chunks = vec![
            // Chunk 1: All choices start normally
            create_multi_choice_chunk(vec![
                ("Starting task A. ".to_string(), 0),
                ("Starting task B. ".to_string(), 1),
                ("Starting task C. ".to_string(), 2),
            ]),
            // Chunk 2: Choice 0 starts tool call (gets jailed), others continue
            create_multi_choice_chunk(vec![
                ("<tool_call>".to_string(), 0),    // Choice 0 jailed
                ("Continuing B. ".to_string(), 1), // Choice 1 continues
                ("Continuing C. ".to_string(), 2), // Choice 2 continues
            ]),
            // Chunk 3: Choice 0 still jailed, Choice 2 starts tool call
            create_multi_choice_chunk(vec![
                ("{\"name\": \"tool_a\"".to_string(), 0), // Choice 0 still jailed
                ("More B content. ".to_string(), 1),      // Choice 1 continues
                ("<tool_call>".to_string(), 2),           // Choice 2 now jailed
            ]),
            // Chunk 4: Choice 0 finishes tool call, Choice 2 continues tool call
            create_multi_choice_chunk(vec![
                (", \"arguments\": {}}</tool_call>".to_string(), 0), // Choice 0 unjails
                ("Final B. ".to_string(), 1),                        // Choice 1 continues
                ("{\"name\": \"tool_c\", \"arguments\": {}}".to_string(), 2), // Choice 2 still jailed
            ]),
            // Chunk 5: Choice 2 finishes tool call
            create_multi_choice_chunk(vec![
                ("After tool A. ".to_string(), 0), // Choice 0 continues after unjail
                ("Done with B. ".to_string(), 1),  // Choice 1 continues
                ("</tool_call>".to_string(), 2),   // Choice 2 unjails
            ]),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // EXPECTED BEHAVIOR (will fail with current implementation):
        // - Choice 1 should stream continuously (never jailed)
        // - Choice 0 should jail from chunk 2 until chunk 4
        // - Choice 2 should jail from chunk 3 until chunk 5
        // - Each choice should emit independently

        // Verify choice 1 was never interrupted (should have ~5 chunks of content)
        let choice_1_chunks: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 1 && c.delta.content.is_some())
            .collect();

        assert!(
            choice_1_chunks.len() >= 4,
            "Choice 1 should have multiple continuous chunks, got {}",
            choice_1_chunks.len()
        );

        // Verify choice 0 has a tool call response
        let choice_0_tool_calls: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 0 && c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        assert!(
            !choice_0_tool_calls.is_empty(),
            "Choice 0 should have tool call response"
        );

        // Verify choice 2 has a tool call response
        let choice_2_tool_calls: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 2 && c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        assert!(
            !choice_2_tool_calls.is_empty(),
            "Choice 2 should have tool call response"
        );
    }

    #[tokio::test]
    async fn test_deterministic_choice_ordering() {
        // Test that choices are processed in deterministic order (0, 1, 2...)
        // This test will FAIL with the current HashMap implementation
        let chunks = vec![
            // All choices have tool calls that complete at the same time
            create_multi_choice_chunk(vec![
                (
                    "<tool_call>{\"name\": \"tool_0\", \"arguments\": {}}</tool_call>".to_string(),
                    0,
                ),
                (
                    "<tool_call>{\"name\": \"tool_1\", \"arguments\": {}}</tool_call>".to_string(),
                    1,
                ),
                (
                    "<tool_call>{\"name\": \"tool_2\", \"arguments\": {}}</tool_call>".to_string(),
                    2,
                ),
            ]),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Find all tool call responses
        let mut tool_call_responses: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        // Sort by the order they appear in the results
        // With HashMap, this order will be non-deterministic
        // With Vec, this should always be [0, 1, 2]
        tool_call_responses.sort_by_key(|c| c.index);

        assert_eq!(
            tool_call_responses.len(),
            3,
            "Should have 3 tool call responses"
        );

        // Run this test multiple times to verify determinism
        for run in 0..5 {
            let chunks = vec![create_multi_choice_chunk(vec![
                (
                    "<tool_call>{\"name\": \"tool_0\", \"arguments\": {}}</tool_call>".to_string(),
                    0,
                ),
                (
                    "<tool_call>{\"name\": \"tool_1\", \"arguments\": {}}</tool_call>".to_string(),
                    1,
                ),
                (
                    "<tool_call>{\"name\": \"tool_2\", \"arguments\": {}}</tool_call>".to_string(),
                    2,
                ),
            ])];

            let input_stream = stream::iter(chunks);
            let jail = JailedStream::builder().tool_call_parser("hermes").build();
            let jailed_stream = jail.apply(input_stream);
            let run_results: Vec<_> = jailed_stream.collect().await;

            let run_responses: Vec<_> = run_results
                .iter()
                .filter_map(|r| r.data.as_ref())
                .flat_map(|d| &d.choices)
                .filter(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                .collect();

            // The order should be consistent across runs
            // This will fail with HashMap due to non-deterministic iteration
            let indices: Vec<u32> = run_responses.iter().map(|c| c.index).collect();
            assert_eq!(
                indices,
                vec![0, 1, 2],
                "Choice processing order should be deterministic on run {}",
                run
            );
        }
    }

    #[tokio::test]
    async fn test_multiple_choices_usage_aggregation() {
        // Test that usage is correctly aggregated across multiple choices
        // This test demonstrates how usage should work with n>1

        // For now, this test just documents expected behavior
        // It will need to be expanded once usage aggregation is implemented

        let chunks = vec![create_multi_choice_chunk(vec![
            ("Response A with many tokens".to_string(), 0), // ~5 tokens
            ("Response B".to_string(), 1),                  // ~2 tokens
            ("Response C has even more tokens than A".to_string(), 2), // ~8 tokens
        ])];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // TODO: Once usage aggregation is implemented, verify:
        // - Usage chunk has choices: [] (empty array)
        // - completion_tokens = sum of all choices (~15 total)
        // - prompt_tokens counted once
        // - total_tokens = prompt_tokens + completion_tokens

        // For now, just verify we got some results
        assert!(!results.is_empty(), "Should have some results");
    }

    #[tokio::test]
    async fn test_partial_matching_false_positive_prevention() {
        // Input chunks:
        // [0] "n "
        // [1] "<"
        // [2] " 5"
        //
        // Expected output:
        // [0] Content("n ")
        // [1] Content("< 5")  // "<" held as partial, then combined with " 5" when pattern doesn't match
        let chunks = vec![
            create_mock_response_chunk("n ".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk(" 5".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Use nemotron parser which has <TOOLCALL> as a pattern
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            2,
            "Should emit exactly 2 chunks: 'n ' and '< 5'"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "n ");
        assert_content(&results[1], "< 5");

        // === Verify negative assertions ===
        // Verify NO tool calls were detected
        for (i, result) in results.iter().enumerate() {
            assert!(
                !has_tool_call(result),
                "Chunk {} should not contain tool calls in mathematical expression",
                i
            );
        }

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "n < 5",
            "Content reconstruction should preserve the complete mathematical expression"
        );
    }

    #[tokio::test]
    async fn test_partial_matching_suffix_detection() {
        // Input chunks:
        // [0] "text<TO"
        // [1] "OLCALL>[{\"name\": \"test\", \"arguments\": {}}]</TOOLCALL>"
        //
        // Expected output:
        // [0] Content("text")  // "<TO" held as partial
        // [1] ToolCall("test", {})
        let chunks = vec![
            create_mock_response_chunk("text<TO".to_string(), 0),
            create_mock_response_chunk(
                "OLCALL>[{\"name\": \"test\", \"arguments\": {}}]</TOOLCALL>".to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .jail_end_sequence("</TOOLCALL>")
            .build();

        let jailed_stream = jail.apply(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            2,
            "Should emit exactly 2 chunks: [0] 'text' content, [1] tool call"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "text");
        assert_tool_call(&results[1], "test", json!({}));

        // === Verify negative assertions ===
        // Verify '<' was not emitted in first chunk (held as partial)
        let first_content = extract_content(&results[0]);
        assert!(
            !first_content.contains('<'),
            "First chunk should not contain '<' as it's part of partial match '<TO'"
        );

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "text",
            "Content reconstruction should only include 'text' (tool call parsed separately)"
        );
    }
}

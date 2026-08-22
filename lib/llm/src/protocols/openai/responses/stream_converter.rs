// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Responses API SSE events.
//!
//! The event sequence follows the OpenAI Responses API streaming spec:
//! `response.created` -> `response.in_progress` -> `response.output_item.added` ->
//! `response.content_part.added` -> N x `response.output_text.delta` ->
//! `response.output_text.done` -> `response.content_part.done` ->
//! `response.output_item.done` -> `response.completed` -> `[DONE]`

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::Event;
use dynamo_protocols::types::responses::{
    AssistantRole, FunctionToolCall, IncompleteDetails, InputTokenDetails, Instructions,
    OutputContent, OutputItem, OutputMessage, OutputMessageContent, OutputStatus,
    OutputTextContent, OutputTokenDetails, ReasoningItem, Response, ResponseCompletedEvent,
    ResponseContentPartAddedEvent, ResponseContentPartDoneEvent, ResponseCreatedEvent,
    ResponseCustomToolCallInputDoneEvent, ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent, ResponseFunctionCallArgumentsDoneEvent,
    ResponseInProgressEvent, ResponseIncompleteEvent, ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent, ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent, ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent, ResponseStreamEvent, ResponseTextDeltaEvent,
    ResponseTextDoneEvent, ResponseTextParam, ResponseUsage, ServiceTier, Status, SummaryPart,
    SummaryTextContent, TextResponseFormatConfiguration, ToolChoiceOptions, ToolChoiceParam,
    Truncation,
};
use serde::{
    Serialize,
    ser::{SerializeMap, Serializer},
};
use uuid::Uuid;

use dynamo_protocols::types::{ChatCompletionMessageContent, FinishReason};

use super::{
    ResponseParams, ResponseToolOrigin, custom_input_from_synthetic_arguments,
    make_custom_tool_call_item,
};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::protocols::unified::ResponsesContext;

/// State machine that converts a chat completion stream into Responses API events.
pub struct ResponseStreamConverter {
    response_id: String,
    model: String,
    params: ResponseParams,
    /// Preserved Responses API-specific request context for faithful response reconstruction.
    api_context: Option<ResponsesContext>,
    created_at: u64,
    sequence_number: u64,
    // Text message tracking
    message_item_id: String,
    message_started: bool,
    message_output_index: u32,
    accumulated_text: String,
    // Reasoning summary tracking
    reasoning_item_id: String,
    reasoning_started: bool,
    reasoning_done: bool,
    reasoning_output_index: u32,
    reasoning_output_status: Option<OutputStatus>,
    accumulated_reasoning: String,
    // Function call tracking
    function_call_items: Vec<FunctionCallState>,
    // Output index counter
    next_output_index: u32,
    // Usage stats from the backend's final chunk
    usage: Option<ResponseUsage>,
    // The backend exhausted the output budget.
    output_limit_reached: bool,
}

struct FunctionCallState {
    item_id: String,
    call_id: String,
    name: String,
    namespace: Option<String>,
    origin: Option<ResponseToolOrigin>,
    accumulated_args: String,
    pending_arg_deltas: Vec<String>,
    output_index: Option<u32>,
    started: bool,
    done: bool,
}

impl FunctionCallState {
    fn has_identity(&self) -> bool {
        !self.call_id.is_empty() && !self.name.is_empty()
    }
}

impl ResponseStreamConverter {
    pub fn new(model: String, params: ResponseParams) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            response_id: format!("resp_{}", Uuid::new_v4().simple()),
            model,
            params,
            api_context: None,
            created_at,
            sequence_number: 0,
            message_item_id: format!("msg_{}", Uuid::new_v4().simple()),
            message_started: false,
            message_output_index: 0,
            accumulated_text: String::new(),
            reasoning_item_id: format!("rs_{}", Uuid::new_v4().simple()),
            reasoning_started: false,
            reasoning_done: false,
            reasoning_output_index: 0,
            reasoning_output_status: None,
            accumulated_reasoning: String::new(),
            function_call_items: Vec::new(),
            next_output_index: 0,
            usage: None,
            output_limit_reached: false,
        }
    }

    pub fn with_context(model: String, params: ResponseParams, context: ResponsesContext) -> Self {
        let mut converter = Self::new(model, params);
        converter.api_context = Some(context);
        converter
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    fn make_response(&self, status: Status, output: Vec<OutputItem>) -> Response {
        let is_incomplete = status == Status::Incomplete;
        let completed_at = if status == Status::Completed {
            Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            )
        } else {
            None
        };
        Response {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_at,
            completed_at,
            status,
            model: self.model.clone(),
            output,
            // Echo request params with spec-required defaults for omitted fields
            background: Some(false),
            metadata: Some(HashMap::new()),
            parallel_tool_calls: self.params.parallel_tool_calls.or(Some(true)),
            temperature: self.params.temperature.or(Some(1.0)),
            text: Some(self.params.text.clone().unwrap_or(ResponseTextParam {
                format: TextResponseFormatConfiguration::Text,
                verbosity: None,
            })),
            tool_choice: self
                .params
                .tool_choice
                .clone()
                .or(Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto))),
            tools: Some(
                self.params
                    .tools
                    .clone()
                    .map(super::normalize_tools)
                    .unwrap_or_default(),
            ),
            top_p: self.params.top_p.or(Some(1.0)),
            truncation: Some(self.params.truncation.unwrap_or(Truncation::Disabled)),
            // Nullable required fields
            billing: None,
            conversation: None,
            error: None,
            incomplete_details: is_incomplete.then(|| IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            }),
            instructions: self.params.instructions.clone().map(Instructions::Text),
            max_output_tokens: self.params.max_output_tokens,
            previous_response_id: self
                .api_context
                .as_ref()
                .and_then(|ctx| ctx.previous_response_id.clone()),
            prompt: None,
            prompt_cache_key: self.params.prompt_cache_key.clone(),
            prompt_cache_retention: self.params.prompt_cache_retention,
            reasoning: self.params.reasoning.clone(),
            safety_identifier: self.params.safety_identifier.clone(),
            service_tier: Some(self.params.service_tier.unwrap_or(ServiceTier::Auto)),
            top_logprobs: Some(0),
            usage: self.usage.clone(),
        }
    }

    /// Emit the initial lifecycle events: created + in_progress.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::with_capacity(2);
        self.append_start_events(&mut events);
        events
    }

    /// Append the initial lifecycle events: created + in_progress.
    pub fn append_start_events(&mut self, events: &mut Vec<Result<Event, anyhow::Error>>) {
        let created = ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress, vec![]),
        });
        events.push(self.make_sse_event(&created));

        let in_progress = ResponseStreamEvent::ResponseInProgress(ResponseInProgressEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress, vec![]),
        });
        events.push(self.make_sse_event(&in_progress));
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
        // Capture usage stats from the final chunk (sent when stream_options.include_usage=true)
        if let Some(ref u) = chunk.inner.usage {
            self.usage = Some(ResponseUsage {
                input_tokens: u.prompt_tokens,
                input_tokens_details: InputTokenDetails {
                    cached_tokens: u
                        .prompt_tokens_details
                        .as_ref()
                        .and_then(|d| d.cached_tokens)
                        .unwrap_or(0),
                },
                output_tokens: u.completion_tokens,
                output_tokens_details: OutputTokenDetails {
                    reasoning_tokens: u
                        .completion_tokens_details
                        .as_ref()
                        .and_then(|d| d.reasoning_tokens)
                        .unwrap_or(0),
                },
                total_tokens: u.total_tokens,
            });
        }

        let mut should_finish_function_calls = false;
        for choice in &chunk.inner.choices {
            let delta = &choice.delta;

            if choice.finish_reason == Some(FinishReason::Length) {
                self.output_limit_reached = true;
            }

            if let Some(reasoning) = delta.reasoning_content.as_deref()
                && !reasoning.is_empty()
                && !self.reasoning_done
                && self.params.reasoning_summary_requested()
            {
                self.accumulated_reasoning.push_str(reasoning);
                if !self.reasoning_started {
                    self.reasoning_started = true;
                    self.reasoning_output_index = self.next_output_index;
                    let output_index = self.reasoning_output_index;
                    self.next_output_index += 1;

                    let item_added = ResponseStreamEvent::ResponseOutputItemAdded(
                        ResponseOutputItemAddedEvent {
                            sequence_number: self.next_seq(),
                            output_index,
                            item: OutputItem::Reasoning(ReasoningItem {
                                id: Some(self.reasoning_item_id.clone()),
                                summary: vec![],
                                content: None,
                                encrypted_content: None,
                                status: Some(OutputStatus::InProgress),
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&item_added));

                    let part_added = ResponseStreamEvent::ResponseReasoningSummaryPartAdded(
                        ResponseReasoningSummaryPartAddedEvent {
                            sequence_number: self.next_seq(),
                            item_id: self.reasoning_item_id.clone(),
                            output_index,
                            summary_index: 0,
                            part: SummaryPart::SummaryText(SummaryTextContent {
                                text: String::new(),
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&part_added));
                }

                let reasoning_delta = ResponseStreamEvent::ResponseReasoningSummaryTextDelta(
                    ResponseReasoningSummaryTextDeltaEvent {
                        sequence_number: self.next_seq(),
                        item_id: self.reasoning_item_id.clone(),
                        output_index: self.reasoning_output_index,
                        summary_index: 0,
                        delta: reasoning.to_string(),
                    },
                );
                events.push(self.make_sse_event(&reasoning_delta));
            }

            // Handle text content deltas — extract text from the enum
            let content_text = match &delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                Some(ChatCompletionMessageContent::Parts(_)) => {
                    // Multimodal streaming not yet supported
                    None
                }
                None => None,
            };
            if let Some(content) = content_text
                && !content.is_empty()
            {
                // Starting the answer is an explicit reasoning phase boundary.
                // The reasoning item completed even when this same chunk also
                // reports that the answer exhausted the output budget.
                self.append_reasoning_done_events(events, OutputStatus::Completed);

                // Emit output_item.added + content_part.added on first text
                if !self.message_started {
                    self.message_started = true;
                    self.message_output_index = self.next_output_index;
                    let output_index = self.message_output_index;
                    self.next_output_index += 1;

                    let item_added = ResponseStreamEvent::ResponseOutputItemAdded(
                        ResponseOutputItemAddedEvent {
                            sequence_number: self.next_seq(),
                            output_index,
                            item: OutputItem::Message(OutputMessage {
                                id: self.message_item_id.clone(),
                                content: vec![],
                                role: AssistantRole::Assistant,
                                phase: None,
                                status: OutputStatus::InProgress,
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&item_added));

                    let part_added = ResponseStreamEvent::ResponseContentPartAdded(
                        ResponseContentPartAddedEvent {
                            sequence_number: self.next_seq(),
                            item_id: self.message_item_id.clone(),
                            output_index,
                            content_index: 0,
                            part: OutputContent::OutputText(OutputTextContent {
                                text: String::new(),
                                annotations: vec![],
                                logprobs: Some(vec![]),
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&part_added));
                }

                // Emit text delta
                self.accumulated_text.push_str(content);
                let text_delta =
                    ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
                        sequence_number: self.next_seq(),
                        item_id: self.message_item_id.clone(),
                        output_index: self.message_output_index,
                        content_index: 0,
                        delta: content.to_string(),
                        logprobs: Some(vec![]),
                    });
                events.push(self.make_sse_event(&text_delta));
            }

            // Handle tool call deltas
            if let Some(tool_calls) = &delta.tool_calls {
                if !tool_calls.is_empty() {
                    // Starting a tool call is also an explicit reasoning phase
                    // boundary, independent of this chunk's finish reason.
                    self.append_reasoning_done_events(events, OutputStatus::Completed);
                }
                for tc in tool_calls {
                    let tc_index = tc.index as usize;

                    // Start a new function call if we haven't seen this index
                    while self.function_call_items.len() <= tc_index {
                        self.function_call_items.push(FunctionCallState {
                            item_id: format!("fc_{}", Uuid::new_v4().simple()),
                            call_id: String::new(),
                            name: String::new(),
                            namespace: None,
                            origin: None,
                            accumulated_args: String::new(),
                            pending_arg_deltas: Vec::new(),
                            output_index: None,
                            started: false,
                            done: false,
                        });
                    }

                    // Update call_id and name if provided
                    if let Some(id) = &tc.id {
                        self.function_call_items[tc_index].call_id = id.clone();
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            self.function_call_items[tc_index].name = name.clone();
                            let origin = self.params.tool_origin(name);
                            self.function_call_items[tc_index].namespace = match &origin {
                                Some(ResponseToolOrigin::Function { namespace })
                                | Some(ResponseToolOrigin::Custom { namespace }) => {
                                    namespace.clone()
                                }
                                None => None,
                            };
                            self.function_call_items[tc_index].origin = origin;
                        }
                        if let Some(args) = &func.arguments {
                            self.function_call_items[tc_index]
                                .accumulated_args
                                .push_str(args);
                            self.function_call_items[tc_index]
                                .pending_arg_deltas
                                .push(args.clone());
                        }
                    }

                    // Within a single call, identity (id/name) and arguments can arrive in
                    // either order — arguments may begin before the identity chunk. Do not
                    // publish an output item with empty required fields; once identity is
                    // complete, publish the item and any argument fragments already received.
                    // Across parallel calls, the in-tree parsers emit one call at a time with
                    // a monotonically increasing index, so indices are not interleaved today;
                    // keying state by `tc_index` would handle interleaving too, but that path
                    // is defensive rather than exercised by any current backend.
                    let should_start = {
                        let state = &self.function_call_items[tc_index];
                        !state.started && state.has_identity()
                    };
                    let new_output_index = should_start.then(|| {
                        let output_index = self.next_output_index;
                        self.next_output_index += 1;
                        output_index
                    });
                    let (item_added, argument_target, argument_deltas) = {
                        let state = &mut self.function_call_items[tc_index];
                        let item_added = if let Some(output_index) = new_output_index {
                            state.started = true;
                            state.output_index = Some(output_index);
                            Some((
                                state.item_id.clone(),
                                state.call_id.clone(),
                                state.name.clone(),
                                state.namespace.clone(),
                                state.origin.clone(),
                                output_index,
                            ))
                        } else {
                            None
                        };
                        let argument_deltas = if state.started {
                            std::mem::take(&mut state.pending_arg_deltas)
                        } else {
                            Vec::new()
                        };
                        (
                            item_added,
                            state.output_index.map(|output_index| {
                                (state.item_id.clone(), output_index, state.origin.clone())
                            }),
                            argument_deltas,
                        )
                    };

                    if let Some((item_id, call_id, name, namespace, origin, output_index)) =
                        item_added
                    {
                        let item = match origin {
                            Some(ResponseToolOrigin::Custom { .. }) => {
                                OutputItem::CustomToolCall(make_custom_tool_call_item(
                                    item_id,
                                    call_id,
                                    namespace,
                                    name,
                                    String::new(),
                                ))
                            }
                            _ => OutputItem::FunctionCall(FunctionToolCall {
                                id: Some(item_id),
                                call_id,
                                namespace,
                                name,
                                arguments: String::new(),
                                status: Some(OutputStatus::InProgress),
                            }),
                        };
                        let item_added = ResponseStreamEvent::ResponseOutputItemAdded(
                            ResponseOutputItemAddedEvent {
                                sequence_number: self.next_seq(),
                                output_index,
                                item,
                            },
                        );
                        events.push(self.make_sse_event(&item_added));
                    }

                    if let Some((item_id, output_index, origin)) = argument_target {
                        if !matches!(origin, Some(ResponseToolOrigin::Custom { .. })) {
                            for delta in argument_deltas {
                                let args_delta =
                                    ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(
                                        ResponseFunctionCallArgumentsDeltaEvent {
                                            sequence_number: self.next_seq(),
                                            item_id: item_id.clone(),
                                            output_index,
                                            delta,
                                        },
                                    );
                                events.push(self.make_sse_event(&args_delta));
                            }
                        }
                    }
                }
            }

            // `JailedStream` rewrites `Stop` to `ToolCalls` after emitting
            // tool-call chunks. Interrupted `Length`/`ContentFilter` streams
            // retain their reason and use the EOF fallback in `append_end_events`.
            if choice.finish_reason == Some(FinishReason::ToolCalls)
                || choice.finish_reason == Some(FinishReason::FunctionCall)
            {
                should_finish_function_calls = true;
            }
        }

        if should_finish_function_calls {
            self.append_pending_function_call_done_events(events);
        }
    }

    fn append_reasoning_done_events(
        &mut self,
        events: &mut Vec<Result<Event, anyhow::Error>>,
        output_status: OutputStatus,
    ) {
        if self.reasoning_done {
            return;
        }
        self.reasoning_done = true;
        if !self.reasoning_started {
            return;
        }
        self.reasoning_output_status = Some(output_status);

        let text_done = ResponseStreamEvent::ResponseReasoningSummaryTextDone(
            ResponseReasoningSummaryTextDoneEvent {
                sequence_number: self.next_seq(),
                item_id: self.reasoning_item_id.clone(),
                output_index: self.reasoning_output_index,
                summary_index: 0,
                text: self.accumulated_reasoning.clone(),
            },
        );
        events.push(self.make_sse_event(&text_done));

        let summary = SummaryPart::SummaryText(SummaryTextContent {
            text: self.accumulated_reasoning.clone(),
        });
        let part_done = ResponseStreamEvent::ResponseReasoningSummaryPartDone(
            ResponseReasoningSummaryPartDoneEvent {
                sequence_number: self.next_seq(),
                item_id: self.reasoning_item_id.clone(),
                output_index: self.reasoning_output_index,
                summary_index: 0,
                part: summary.clone(),
            },
        );
        events.push(self.make_sse_event(&part_done));

        let item_done = ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
            sequence_number: self.next_seq(),
            output_index: self.reasoning_output_index,
            item: OutputItem::Reasoning(ReasoningItem {
                id: Some(self.reasoning_item_id.clone()),
                summary: vec![summary],
                content: None,
                encrypted_content: None,
                status: Some(output_status),
            }),
        });
        events.push(self.make_sse_event(&item_done));
    }

    fn append_pending_function_call_done_events(
        &mut self,
        events: &mut Vec<Result<Event, anyhow::Error>>,
    ) {
        let output_status = self.output_status();
        // `started` is set only after `has_identity()` observes both required
        // fields, matching Anthropic's `is_emit_ready()` identity requirement.
        let mut pending: Vec<_> = self
            .function_call_items
            .iter_mut()
            .filter(|fc| fc.started && !fc.done)
            .map(|fc| {
                fc.done = true;
                (
                    fc.item_id.clone(),
                    fc.call_id.clone(),
                    fc.name.clone(),
                    fc.namespace.clone(),
                    fc.origin.clone(),
                    fc.output_index
                        .expect("started function call is missing an output index"),
                    fc.accumulated_args.clone(),
                )
            })
            .collect();
        pending.sort_unstable_by_key(|(_, _, _, _, _, output_index, _)| *output_index);

        for (item_id, call_id, fc_name, namespace, origin, output_index, accumulated_args) in
            pending
        {
            if matches!(origin, Some(ResponseToolOrigin::Custom { .. })) {
                let input = custom_input_from_synthetic_arguments(&accumulated_args);
                let input_done = ResponseStreamEvent::ResponseCustomToolCallInputDone(
                    ResponseCustomToolCallInputDoneEvent {
                        sequence_number: self.next_seq(),
                        output_index,
                        item_id: item_id.clone(),
                        input: input.clone(),
                    },
                );
                events.push(self.make_sse_event(&input_done));

                let item_done =
                    ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                        sequence_number: self.next_seq(),
                        output_index,
                        item: OutputItem::CustomToolCall(make_custom_tool_call_item(
                            item_id, call_id, namespace, fc_name, input,
                        )),
                    });
                events.push(self.make_sse_event(&item_done));
                continue;
            }

            let args_done = ResponseStreamEvent::ResponseFunctionCallArgumentsDone(
                ResponseFunctionCallArgumentsDoneEvent {
                    sequence_number: self.next_seq(),
                    item_id: item_id.clone(),
                    output_index,
                    arguments: accumulated_args.clone(),
                    name: Some(fc_name.clone()),
                },
            );
            events.push(self.make_sse_event(&args_done));

            let item_done =
                ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                    sequence_number: self.next_seq(),
                    output_index,
                    item: OutputItem::FunctionCall(FunctionToolCall {
                        id: Some(item_id),
                        call_id,
                        namespace,
                        name: fc_name,
                        arguments: accumulated_args,
                        status: Some(output_status),
                    }),
                });
            events.push(self.make_sse_event(&item_done));
        }
    }

    fn output_status(&self) -> OutputStatus {
        if self.output_limit_reached {
            OutputStatus::Incomplete
        } else {
            OutputStatus::Completed
        }
    }

    fn terminal_status(&self) -> Status {
        if self.output_limit_reached {
            Status::Incomplete
        } else {
            Status::Completed
        }
    }

    fn completed_output(&self) -> Vec<OutputItem> {
        let output_status = self.output_status();
        let mut output = Vec::new();
        if self.reasoning_started {
            output.push((
                self.reasoning_output_index,
                OutputItem::Reasoning(ReasoningItem {
                    id: Some(self.reasoning_item_id.clone()),
                    summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                        text: self.accumulated_reasoning.clone(),
                    })],
                    content: None,
                    encrypted_content: None,
                    status: Some(self.reasoning_output_status.unwrap_or(output_status)),
                }),
            ));
        }
        if self.message_started {
            output.push((
                self.message_output_index,
                OutputItem::Message(OutputMessage {
                    id: self.message_item_id.clone(),
                    content: vec![OutputMessageContent::OutputText(OutputTextContent {
                        text: self.accumulated_text.clone(),
                        annotations: vec![],
                        logprobs: Some(vec![]),
                    })],
                    role: AssistantRole::Assistant,
                    phase: None,
                    status: output_status,
                }),
            ));
        }
        for function_call in &self.function_call_items {
            if let Some(output_index) = function_call.output_index {
                let item = match &function_call.origin {
                    Some(ResponseToolOrigin::Custom { .. }) => {
                        OutputItem::CustomToolCall(make_custom_tool_call_item(
                            function_call.item_id.clone(),
                            function_call.call_id.clone(),
                            function_call.namespace.clone(),
                            function_call.name.clone(),
                            custom_input_from_synthetic_arguments(&function_call.accumulated_args),
                        ))
                    }
                    _ => OutputItem::FunctionCall(FunctionToolCall {
                        id: Some(function_call.item_id.clone()),
                        call_id: function_call.call_id.clone(),
                        namespace: function_call.namespace.clone(),
                        name: function_call.name.clone(),
                        arguments: function_call.accumulated_args.clone(),
                        status: Some(output_status),
                    }),
                };
                output.push((output_index, item));
            }
        }
        output.sort_unstable_by_key(|(output_index, _)| *output_index);
        output.into_iter().map(|(_, item)| item).collect()
    }

    /// Emit remaining output completion events and `response.completed` at stream end.
    pub fn emit_end_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();
        self.append_end_events(&mut events);
        events
    }

    /// Append remaining output completion events and `response.completed` at stream end.
    pub fn append_end_events(&mut self, events: &mut Vec<Result<Event, anyhow::Error>>) {
        let output_status = self.output_status();
        // Without a later output item, the response finish reason determines
        // whether the still-open reasoning item completed or was truncated.
        self.append_reasoning_done_events(events, output_status);

        // Close text message if it was started
        if self.message_started {
            let text_done = ResponseStreamEvent::ResponseOutputTextDone(ResponseTextDoneEvent {
                sequence_number: self.next_seq(),
                item_id: self.message_item_id.clone(),
                output_index: self.message_output_index,
                content_index: 0,
                text: self.accumulated_text.clone(),
                logprobs: Some(vec![]),
            });
            events.push(self.make_sse_event(&text_done));

            let part_done =
                ResponseStreamEvent::ResponseContentPartDone(ResponseContentPartDoneEvent {
                    sequence_number: self.next_seq(),
                    item_id: self.message_item_id.clone(),
                    output_index: self.message_output_index,
                    content_index: 0,
                    part: OutputContent::OutputText(OutputTextContent {
                        text: self.accumulated_text.clone(),
                        annotations: vec![],
                        logprobs: Some(vec![]),
                    }),
                });
            events.push(self.make_sse_event(&part_done));

            let item_done =
                ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                    sequence_number: self.next_seq(),
                    output_index: self.message_output_index,
                    item: OutputItem::Message(OutputMessage {
                        id: self.message_item_id.clone(),
                        content: vec![OutputMessageContent::OutputText(OutputTextContent {
                            text: self.accumulated_text.clone(),
                            annotations: vec![],
                            logprobs: Some(vec![]),
                        })],
                        role: AssistantRole::Assistant,
                        phase: None,
                        status: output_status,
                    }),
                });
            events.push(self.make_sse_event(&item_done));
        }

        // Fallback for backends that end the transport without a finish-reason chunk.
        self.append_pending_function_call_done_events(events);

        let terminal_status = self.terminal_status();
        let response = self.make_response(terminal_status.clone(), self.completed_output());
        let terminal = if terminal_status == Status::Incomplete {
            ResponseStreamEvent::ResponseIncomplete(ResponseIncompleteEvent {
                sequence_number: self.next_seq(),
                response,
            })
        } else {
            ResponseStreamEvent::ResponseCompleted(ResponseCompletedEvent {
                sequence_number: self.next_seq(),
                response,
            })
        };
        events.push(self.make_sse_event(&terminal));
    }

    /// Emit error events when the stream ends due to a backend error.
    pub fn emit_error_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();
        self.append_error_events(&mut events);
        events
    }

    /// Append error events when the stream ends due to a backend error.
    pub fn append_error_events(&mut self, events: &mut Vec<Result<Event, anyhow::Error>>) {
        let failed = ResponseStreamEvent::ResponseFailed(ResponseFailedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::Failed, vec![]),
        });
        events.push(self.make_sse_event(&failed));
    }
}

impl ResponseStreamConverter {
    /// Serialize a stream event, patching any embedded `response` object to
    /// satisfy the OpenResponses schema. Takes `&self` so spec-required
    /// sampling params can be sourced from the originating request via
    /// `self.params` rather than hardcoded at each emit site.
    fn make_sse_event(&self, event: &ResponseStreamEvent) -> Result<Event, anyhow::Error> {
        let event_type = get_event_type(event);
        let data = self.serialize_event_data(event)?;
        Ok(Event::default().event(event_type).data(data))
    }

    fn serialize_event_data(
        &self,
        event: &ResponseStreamEvent,
    ) -> Result<String, serde_json::Error> {
        let spec = ResponseSpecFields {
            presence_penalty: self.params.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.params.frequency_penalty.unwrap_or(0.0),
            store: self.params.store.unwrap_or(false),
        };

        match event {
            ResponseStreamEvent::ResponseCreated(event) => {
                serde_json::to_string(&ResponseEventForSpec::new(
                    "response.created",
                    event.sequence_number,
                    &event.response,
                    spec,
                ))
            }
            ResponseStreamEvent::ResponseInProgress(event) => {
                serde_json::to_string(&ResponseEventForSpec::new(
                    "response.in_progress",
                    event.sequence_number,
                    &event.response,
                    spec,
                ))
            }
            ResponseStreamEvent::ResponseCompleted(event) => {
                serde_json::to_string(&ResponseEventForSpec::new(
                    "response.completed",
                    event.sequence_number,
                    &event.response,
                    spec,
                ))
            }
            ResponseStreamEvent::ResponseFailed(event) => {
                serde_json::to_string(&ResponseEventForSpec::new(
                    "response.failed",
                    event.sequence_number,
                    &event.response,
                    spec,
                ))
            }
            ResponseStreamEvent::ResponseIncomplete(event) => {
                serde_json::to_string(&ResponseEventForSpec::new(
                    "response.incomplete",
                    event.sequence_number,
                    &event.response,
                    spec,
                ))
            }
            ResponseStreamEvent::ResponseQueued(event) => {
                serde_json::to_string(&ResponseEventForSpec::new(
                    "response.queued",
                    event.sequence_number,
                    &event.response,
                    spec,
                ))
            }
            _ => serde_json::to_string(event),
        }
    }
}

#[derive(Clone, Copy)]
struct ResponseSpecFields {
    presence_penalty: f32,
    frequency_penalty: f32,
    store: bool,
}

struct ResponseEventForSpec<'a> {
    event_type: &'static str,
    sequence_number: u64,
    response: &'a Response,
    spec: ResponseSpecFields,
}

impl<'a> ResponseEventForSpec<'a> {
    fn new(
        event_type: &'static str,
        sequence_number: u64,
        response: &'a Response,
        spec: ResponseSpecFields,
    ) -> Self {
        Self {
            event_type,
            sequence_number,
            response,
            spec,
        }
    }
}

impl Serialize for ResponseEventForSpec<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("type", self.event_type)?;
        map.serialize_entry("sequence_number", &self.sequence_number)?;
        map.serialize_entry(
            "response",
            &ResponseForSpec {
                response: self.response,
                spec: self.spec,
            },
        )?;
        map.end()
    }
}

struct ResponseForSpec<'a> {
    response: &'a Response,
    spec: ResponseSpecFields,
}

// Mirrors async-openai's `Response` serialization while writing Dynamo's
// OpenResponses spec fields directly, avoiding a per-stream-event Value tree.
impl Serialize for ResponseForSpec<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let response = self.response;
        let mut map = serializer.serialize_map(None)?;

        serialize_optional_entry(&mut map, "background", &response.background)?;
        map.serialize_entry("billing", &response.billing)?;
        map.serialize_entry("conversation", &response.conversation)?;
        map.serialize_entry("created_at", &response.created_at)?;
        map.serialize_entry("completed_at", &response.completed_at)?;
        map.serialize_entry("error", &response.error)?;
        map.serialize_entry("id", &response.id)?;
        map.serialize_entry("incomplete_details", &response.incomplete_details)?;
        map.serialize_entry("instructions", &response.instructions)?;
        map.serialize_entry("max_output_tokens", &response.max_output_tokens)?;
        map.serialize_entry("max_tool_calls", &None::<u32>)?;
        serialize_optional_entry(&mut map, "metadata", &response.metadata)?;
        map.serialize_entry("model", &response.model)?;
        map.serialize_entry("object", &response.object)?;
        map.serialize_entry("output", &response.output)?;
        serialize_optional_entry(
            &mut map,
            "parallel_tool_calls",
            &response.parallel_tool_calls,
        )?;
        map.serialize_entry("previous_response_id", &response.previous_response_id)?;
        map.serialize_entry("prompt", &response.prompt)?;
        map.serialize_entry("prompt_cache_key", &response.prompt_cache_key)?;
        map.serialize_entry("prompt_cache_retention", &response.prompt_cache_retention)?;
        map.serialize_entry("reasoning", &response.reasoning)?;
        map.serialize_entry("safety_identifier", &response.safety_identifier)?;
        serialize_optional_entry(&mut map, "service_tier", &response.service_tier)?;
        map.serialize_entry("status", &response.status)?;
        serialize_optional_entry(&mut map, "temperature", &response.temperature)?;
        serialize_optional_entry(&mut map, "text", &response.text)?;
        serialize_optional_entry(&mut map, "tool_choice", &response.tool_choice)?;
        serialize_optional_entry(&mut map, "tools", &response.tools)?;
        serialize_optional_entry(&mut map, "top_logprobs", &response.top_logprobs)?;
        serialize_optional_entry(&mut map, "top_p", &response.top_p)?;
        serialize_optional_entry(&mut map, "truncation", &response.truncation)?;
        map.serialize_entry("usage", &response.usage)?;
        map.serialize_entry("presence_penalty", &self.spec.presence_penalty)?;
        map.serialize_entry("frequency_penalty", &self.spec.frequency_penalty)?;
        map.serialize_entry("store", &self.spec.store)?;

        map.end()
    }
}

fn serialize_optional_entry<S, T>(
    map: &mut S,
    key: &'static str,
    value: &Option<T>,
) -> Result<(), S::Error>
where
    S: SerializeMap,
    T: Serialize,
{
    if let Some(value) = value {
        map.serialize_entry(key, value)?;
    }
    Ok(())
}

fn get_event_type(event: &ResponseStreamEvent) -> &'static str {
    match event {
        ResponseStreamEvent::ResponseCreated(_) => "response.created",
        ResponseStreamEvent::ResponseInProgress(_) => "response.in_progress",
        ResponseStreamEvent::ResponseCompleted(_) => "response.completed",
        ResponseStreamEvent::ResponseFailed(_) => "response.failed",
        ResponseStreamEvent::ResponseIncomplete(_) => "response.incomplete",
        ResponseStreamEvent::ResponseQueued(_) => "response.queued",
        ResponseStreamEvent::ResponseOutputItemAdded(_) => "response.output_item.added",
        ResponseStreamEvent::ResponseOutputItemDone(_) => "response.output_item.done",
        ResponseStreamEvent::ResponseContentPartAdded(_) => "response.content_part.added",
        ResponseStreamEvent::ResponseContentPartDone(_) => "response.content_part.done",
        ResponseStreamEvent::ResponseOutputTextDelta(_) => "response.output_text.delta",
        ResponseStreamEvent::ResponseOutputTextDone(_) => "response.output_text.done",
        ResponseStreamEvent::ResponseRefusalDelta(_) => "response.refusal.delta",
        ResponseStreamEvent::ResponseRefusalDone(_) => "response.refusal.done",
        ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(_) => {
            "response.function_call_arguments.delta"
        }
        ResponseStreamEvent::ResponseFunctionCallArgumentsDone(_) => {
            "response.function_call_arguments.done"
        }
        ResponseStreamEvent::ResponseFileSearchCallInProgress(_) => {
            "response.file_search_call.in_progress"
        }
        ResponseStreamEvent::ResponseFileSearchCallSearching(_) => {
            "response.file_search_call.searching"
        }
        ResponseStreamEvent::ResponseFileSearchCallCompleted(_) => {
            "response.file_search_call.completed"
        }
        ResponseStreamEvent::ResponseWebSearchCallInProgress(_) => {
            "response.web_search_call.in_progress"
        }
        ResponseStreamEvent::ResponseWebSearchCallSearching(_) => {
            "response.web_search_call.searching"
        }
        ResponseStreamEvent::ResponseWebSearchCallCompleted(_) => {
            "response.web_search_call.completed"
        }
        ResponseStreamEvent::ResponseReasoningSummaryPartAdded(_) => {
            "response.reasoning_summary_part.added"
        }
        ResponseStreamEvent::ResponseReasoningSummaryPartDone(_) => {
            "response.reasoning_summary_part.done"
        }
        ResponseStreamEvent::ResponseReasoningSummaryTextDelta(_) => {
            "response.reasoning_summary_text.delta"
        }
        ResponseStreamEvent::ResponseReasoningSummaryTextDone(_) => {
            "response.reasoning_summary_text.done"
        }
        ResponseStreamEvent::ResponseReasoningTextDelta(_) => "response.reasoning_text.delta",
        ResponseStreamEvent::ResponseReasoningTextDone(_) => "response.reasoning_text.done",
        ResponseStreamEvent::ResponseImageGenerationCallCompleted(_) => {
            "response.image_generation_call.completed"
        }
        ResponseStreamEvent::ResponseImageGenerationCallGenerating(_) => {
            "response.image_generation_call.generating"
        }
        ResponseStreamEvent::ResponseImageGenerationCallInProgress(_) => {
            "response.image_generation_call.in_progress"
        }
        ResponseStreamEvent::ResponseImageGenerationCallPartialImage(_) => {
            "response.image_generation_call.partial_image"
        }
        ResponseStreamEvent::ResponseMCPCallArgumentsDelta(_) => {
            "response.mcp_call_arguments.delta"
        }
        ResponseStreamEvent::ResponseMCPCallArgumentsDone(_) => "response.mcp_call_arguments.done",
        ResponseStreamEvent::ResponseMCPCallCompleted(_) => "response.mcp_call.completed",
        ResponseStreamEvent::ResponseMCPCallFailed(_) => "response.mcp_call.failed",
        ResponseStreamEvent::ResponseMCPCallInProgress(_) => "response.mcp_call.in_progress",
        ResponseStreamEvent::ResponseMCPListToolsCompleted(_) => {
            "response.mcp_list_tools.completed"
        }
        ResponseStreamEvent::ResponseMCPListToolsFailed(_) => "response.mcp_list_tools.failed",
        ResponseStreamEvent::ResponseMCPListToolsInProgress(_) => {
            "response.mcp_list_tools.in_progress"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallInProgress(_) => {
            "response.code_interpreter_call.in_progress"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallInterpreting(_) => {
            "response.code_interpreter_call.interpreting"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallCompleted(_) => {
            "response.code_interpreter_call.completed"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallCodeDelta(_) => {
            "response.code_interpreter_call_code.delta"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallCodeDone(_) => {
            "response.code_interpreter_call_code.done"
        }
        ResponseStreamEvent::ResponseOutputTextAnnotationAdded(_) => {
            "response.output_text.annotation.added"
        }
        ResponseStreamEvent::ResponseCustomToolCallInputDelta(_) => {
            "response.custom_tool_call_input.delta"
        }
        ResponseStreamEvent::ResponseCustomToolCallInputDone(_) => {
            "response.custom_tool_call_input.done"
        }
        ResponseStreamEvent::ResponseError(_) => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::unified::ResponsesContext;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionStreamResponseDelta, FunctionCallStream, FunctionType,
    };

    fn default_params() -> ResponseParams {
        ResponseParams::default()
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

    fn finish_chunk(reason: FinishReason) -> NvCreateChatCompletionStreamResponse {
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
                        reasoning_content: None,
                    },
                    finish_reason: Some(reason),
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

    fn with_finish_reason(
        mut chunk: NvCreateChatCompletionStreamResponse,
        reason: FinishReason,
    ) -> NvCreateChatCompletionStreamResponse {
        chunk.inner.choices[0].finish_reason = Some(reason);
        chunk
    }

    /// Extract the SSE event type from a Result<Event, _>.
    fn event_type(event: &Result<Event, anyhow::Error>) -> String {
        let debug = format!("{:?}", event.as_ref().unwrap());
        // Event debug format: Event { ... event: "response.xxx" ... }
        // Parse the event type from the serialized SSE data
        if let Some(start) = debug.find("event: ") {
            let rest = &debug[start + 7..];
            if let Some(end) = rest.find("\\n") {
                return rest[..end].to_string();
            }
        }
        "unknown".to_string()
    }

    fn event_types(events: &[Result<Event, anyhow::Error>]) -> Vec<String> {
        events.iter().map(event_type).collect()
    }

    fn legacy_event_json(
        event: &ResponseStreamEvent,
        params: &ResponseParams,
    ) -> serde_json::Value {
        let mut value = serde_json::to_value(event).unwrap();
        if let serde_json::Value::Object(ref mut obj) = value
            && let Some(serde_json::Value::Object(inner)) = obj.get_mut("response")
        {
            super::super::patch_response_for_spec(
                inner,
                params.presence_penalty.unwrap_or(0.0),
                params.frequency_penalty.unwrap_or(0.0),
                params.store.unwrap_or(false),
            );
        }
        value
    }

    fn optimized_event_json(
        converter: &ResponseStreamConverter,
        event: &ResponseStreamEvent,
    ) -> serde_json::Value {
        serde_json::from_str(&converter.serialize_event_data(event).unwrap()).unwrap()
    }

    /// Parseable arguments remain open until an explicit tool-call finish reason.
    #[test]
    fn test_complete_tool_call_closes_on_finish_reason() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events(); // consume start events

        let events = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF\"}"),
        ));

        let types = event_types(&events);
        assert!(
            types.contains(&"response.output_item.added".to_string()),
            "should emit output_item.added: {types:?}"
        );
        assert!(
            types.contains(&"response.function_call_arguments.delta".to_string()),
            "should emit args delta: {types:?}"
        );
        assert!(!types.contains(&"response.function_call_arguments.done".to_string()));
        assert!(!types.contains(&"response.output_item.done".to_string()));

        let finish_types = event_types(&conv.process_chunk(&finish_chunk(FinishReason::ToolCalls)));
        assert_eq!(
            finish_types,
            vec![
                "response.function_call_arguments.done".to_string(),
                "response.output_item.done".to_string(),
            ]
        );

        let end_types = event_types(&conv.emit_end_events());
        assert!(!end_types.contains(&"response.function_call_arguments.done".to_string()));
        assert!(!end_types.contains(&"response.output_item.done".to_string()));
        assert!(end_types.contains(&"response.completed".to_string()));

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Completed);
        let OutputItem::FunctionCall(call) = &response.output[0] else {
            panic!("expected function call output");
        };
        assert_eq!(call.status, Some(OutputStatus::Completed));
    }

    #[test]
    fn test_function_call_finish_reason_closes_tool_call() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF\"}"),
        ));

        let finish_types =
            event_types(&conv.process_chunk(&finish_chunk(FinishReason::FunctionCall)));
        assert_eq!(
            finish_types,
            vec![
                "response.function_call_arguments.done".to_string(),
                "response.output_item.done".to_string(),
            ]
        );
    }

    #[test]
    fn test_function_call_preserves_namespace() {
        let params = ResponseParams {
            tools: Some(
                serde_json::from_value(serde_json::json!([{
                    "type": "namespace",
                    "name": "agents",
                    "description": "Subagent tools",
                    "tools": [{"type": "function", "name": "spawn_agent"}],
                }]))
                .unwrap(),
            ),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let added_events = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("spawn_agent"),
            Some(r#"{"agent_type":"worker"}"#),
        ));
        let done_events = conv.process_chunk(&finish_chunk(FinishReason::ToolCalls));

        for (events, expected_type) in [
            (&added_events, "response.output_item.added"),
            (&done_events, "response.output_item.done"),
        ] {
            let event = events
                .iter()
                .find(|event| event_type(event) == expected_type)
                .unwrap();
            assert!(
                format!("{event:?}").contains(r#"\"namespace\":\"agents\""#),
                "{expected_type} did not preserve the namespace"
            );
        }

        let OutputItem::FunctionCall(call) = &conv.completed_output()[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("agents"));
    }

    #[test]
    fn test_length_finish_reason_marks_open_tool_call_incomplete() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF"),
        ));
        let finish_events = conv.process_chunk(&finish_chunk(FinishReason::Length));
        assert!(finish_events.is_empty());

        let end_events = conv.emit_end_events();
        assert_eq!(
            event_types(&end_events),
            vec![
                "response.function_call_arguments.done".to_string(),
                "response.output_item.done".to_string(),
                "response.incomplete".to_string(),
            ]
        );

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        assert_eq!(
            response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str()),
            Some("max_output_tokens")
        );
        let OutputItem::FunctionCall(call) = &response.output[0] else {
            panic!("expected function call output");
        };
        assert_eq!(call.status, Some(OutputStatus::Incomplete));
    }

    #[test]
    fn test_length_finish_reason_emits_incomplete_terminal_response() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.process_chunk(&text_chunk("partial"));
        let _ = conv.process_chunk(&finish_chunk(FinishReason::Length));

        let end_events = conv.emit_end_events();
        assert_eq!(
            event_types(&end_events).last().map(String::as_str),
            Some("response.incomplete")
        );

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        assert_eq!(
            response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str()),
            Some("max_output_tokens")
        );
        assert_eq!(response.completed_at, None);
        let OutputItem::Message(message) = &response.output[0] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn test_length_finish_reason_marks_reasoning_item_incomplete() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("partial"));
        let _ = conv.process_chunk(&finish_chunk(FinishReason::Length));
        let _ = conv.emit_end_events();

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Incomplete));
    }

    #[test]
    fn test_completed_reasoning_stays_complete_when_text_is_truncated() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("complete reasoning"));
        let _ = conv.process_chunk(&text_chunk("partial answer"));
        let _ = conv.process_chunk(&finish_chunk(FinishReason::Length));
        let _ = conv.emit_end_events();

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Completed));
        let OutputItem::Message(message) = &response.output[1] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn test_same_chunk_text_and_length_complete_reasoning_only() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("complete reasoning"));

        let events = conv.process_chunk(&with_finish_reason(
            text_chunk("partial answer"),
            FinishReason::Length,
        ));

        assert_eq!(
            event_types(&events),
            vec![
                "response.reasoning_summary_text.done".to_string(),
                "response.reasoning_summary_part.done".to_string(),
                "response.output_item.done".to_string(),
                "response.output_item.added".to_string(),
                "response.content_part.added".to_string(),
                "response.output_text.delta".to_string(),
            ]
        );
        assert_eq!(conv.reasoning_output_status, Some(OutputStatus::Completed));

        let _ = conv.emit_end_events();
        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Completed));
        let OutputItem::Message(message) = &response.output[1] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn test_same_chunk_tool_call_and_length_complete_reasoning_only() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("complete reasoning"));

        let _ = conv.process_chunk(&with_finish_reason(
            tool_call_chunk(
                0,
                Some("call-1"),
                Some("get_weather"),
                Some("{\"city\":\"SF"),
            ),
            FinishReason::Length,
        ));

        assert_eq!(conv.reasoning_output_status, Some(OutputStatus::Completed));
        let _ = conv.emit_end_events();
        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Completed));
        let OutputItem::FunctionCall(function_call) = &response.output[1] else {
            panic!("expected function call output");
        };
        assert_eq!(function_call.status, Some(OutputStatus::Incomplete));
    }

    #[test]
    fn test_requested_reasoning_summary_streams_complete_event_sequence() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let reasoning_events = conv.process_chunk(&reasoning_chunk("thinking"));
        assert_eq!(
            event_types(&reasoning_events),
            vec![
                "response.output_item.added".to_string(),
                "response.reasoning_summary_part.added".to_string(),
                "response.reasoning_summary_text.delta".to_string(),
            ]
        );

        let text_events = conv.process_chunk(&text_chunk("answer"));
        assert_eq!(
            event_types(&text_events),
            vec![
                "response.reasoning_summary_text.done".to_string(),
                "response.reasoning_summary_part.done".to_string(),
                "response.output_item.done".to_string(),
                "response.output_item.added".to_string(),
                "response.content_part.added".to_string(),
                "response.output_text.delta".to_string(),
            ]
        );

        let response = conv.make_response(Status::Completed, conv.completed_output());
        assert_eq!(response.output.len(), 2);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output before message");
        };
        assert_eq!(
            reasoning.summary,
            vec![SummaryPart::SummaryText(SummaryTextContent {
                text: "thinking".to_string(),
            })]
        );
        assert!(matches!(response.output[1], OutputItem::Message(_)));
    }

    #[test]
    fn test_reasoning_without_requested_summary_emits_no_events() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());

        let events = conv.process_chunk(&reasoning_chunk("private reasoning"));

        assert!(events.is_empty());
        assert!(conv.completed_output().is_empty());
    }

    #[test]
    fn test_reasoning_summary_ignores_updates_after_completion() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let _ = conv.process_chunk(&reasoning_chunk("summary"));
        let _ = conv.process_chunk(&text_chunk("answer"));
        let late_events = conv.process_chunk(&reasoning_chunk(" must not be appended"));

        assert!(late_events.is_empty());
        let output = conv.completed_output();
        let OutputItem::Reasoning(reasoning) = &output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(
            reasoning.summary,
            vec![SummaryPart::SummaryText(SummaryTextContent {
                text: "summary".to_string(),
            })]
        );
    }

    #[test]
    fn test_reasoning_summary_finishes_before_tool_call() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let _ = conv.process_chunk(&reasoning_chunk("summary"));
        let tool_events =
            conv.process_chunk(&tool_call_chunk(0, Some("call-1"), Some("get_time"), None));
        assert_eq!(
            event_types(&tool_events),
            vec![
                "response.reasoning_summary_text.done".to_string(),
                "response.reasoning_summary_part.done".to_string(),
                "response.output_item.done".to_string(),
                "response.output_item.added".to_string(),
            ]
        );

        let late_events = conv.process_chunk(&reasoning_chunk(" must not be appended"));
        assert!(late_events.is_empty());
    }

    #[test]
    fn test_reasoning_summary_does_not_start_after_visible_output() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let _ = conv.process_chunk(&text_chunk("answer"));
        let late_events = conv.process_chunk(&reasoning_chunk("out of order"));

        assert!(late_events.is_empty());
        assert!(
            conv.completed_output()
                .iter()
                .all(|item| !matches!(item, OutputItem::Reasoning(_)))
        );
    }

    #[test]
    fn test_identity_only_tool_call_is_emitted_and_finished() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let start_types = event_types(&conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_time"),
            None,
        )));
        assert_eq!(start_types, vec!["response.output_item.added".to_string()]);

        let finish_types = event_types(&conv.process_chunk(&finish_chunk(FinishReason::ToolCalls)));
        assert_eq!(
            finish_types,
            vec![
                "response.function_call_arguments.done".to_string(),
                "response.output_item.done".to_string(),
            ]
        );
        assert_eq!(conv.function_call_items[0].accumulated_args, "");
    }

    #[test]
    fn test_arguments_wait_for_identity_before_events_are_published() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let argument_types =
            event_types(&conv.process_chunk(&tool_call_chunk(0, None, None, Some("{}"))));
        assert!(argument_types.is_empty());

        let identity_types = event_types(&conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_time"),
            None,
        )));
        assert_eq!(
            identity_types,
            vec![
                "response.output_item.added".to_string(),
                "response.function_call_arguments.delta".to_string(),
            ]
        );
        assert_eq!(conv.function_call_items[0].call_id, "call-1");
        assert_eq!(conv.function_call_items[0].name, "get_time");
        assert_eq!(conv.function_call_items[0].accumulated_args, "{}");
    }

    #[test]
    fn test_out_of_order_identity_preserves_assigned_output_order() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let incomplete = event_types(&conv.process_chunk(&tool_call_chunk(
            0,
            Some("incomplete"),
            None,
            Some("{}"),
        )));
        assert!(incomplete.is_empty());
        assert_eq!(conv.function_call_items[0].output_index, None);

        let valid = event_types(&conv.process_chunk(&tool_call_chunk(
            1,
            Some("call-1"),
            Some("get_time"),
            Some("{}"),
        )));
        assert_eq!(
            valid,
            vec![
                "response.output_item.added".to_string(),
                "response.function_call_arguments.delta".to_string(),
            ]
        );
        assert_eq!(conv.function_call_items[1].output_index, Some(0));

        let late_identity =
            event_types(&conv.process_chunk(&tool_call_chunk(0, None, Some("late_call"), None)));
        assert_eq!(
            late_identity,
            vec![
                "response.output_item.added".to_string(),
                "response.function_call_arguments.delta".to_string(),
            ]
        );
        assert_eq!(conv.function_call_items[0].output_index, Some(1));

        let finish = event_types(&conv.process_chunk(&finish_chunk(FinishReason::ToolCalls)));
        assert_eq!(
            finish,
            vec![
                "response.function_call_arguments.done".to_string(),
                "response.output_item.done".to_string(),
                "response.function_call_arguments.done".to_string(),
                "response.output_item.done".to_string(),
            ]
        );

        let names: Vec<_> = conv
            .completed_output()
            .into_iter()
            .map(|item| match item {
                OutputItem::FunctionCall(call) => call.name,
                other => panic!("expected function call, got {other:?}"),
            })
            .collect();
        assert_eq!(names, vec!["get_time", "late_call"]);
    }

    #[test]
    fn test_empty_initial_arguments_do_not_finish_function_call_early() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let first = event_types(&conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("read_file"),
            Some(""),
        )));
        assert_eq!(
            first,
            vec![
                "response.output_item.added".to_string(),
                "response.function_call_arguments.delta".to_string(),
            ]
        );

        let middle = event_types(&conv.process_chunk(&tool_call_chunk(
            0,
            None,
            None,
            Some("{\"path\":\"/tmp"),
        )));
        assert_eq!(
            middle,
            vec!["response.function_call_arguments.delta".to_string()]
        );

        let last = event_types(&conv.process_chunk(&tool_call_chunk(0, None, None, Some("\"}"))));
        assert_eq!(
            last,
            vec!["response.function_call_arguments.delta".to_string()]
        );

        let finish = event_types(&conv.process_chunk(&finish_chunk(FinishReason::ToolCalls)));
        assert_eq!(
            finish,
            vec![
                "response.function_call_arguments.done".to_string(),
                "response.output_item.done".to_string(),
            ]
        );

        let end = event_types(&conv.emit_end_events());
        assert!(!end.contains(&"response.function_call_arguments.done".to_string()));
        assert!(!end.contains(&"response.output_item.done".to_string()));
    }

    /// A tool-call finish reason closes every pending parallel call exactly once.
    #[test]
    fn test_multiple_tool_calls_each_close_on_finish_reason() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let events1 = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF\"}"),
        ));
        let types1 = event_types(&events1);
        assert!(!types1.contains(&"response.function_call_arguments.done".to_string()));

        let events2 = conv.process_chunk(&tool_call_chunk(
            1,
            Some("call-2"),
            Some("get_time"),
            Some("{\"tz\":\"PST\"}"),
        ));
        let types2 = event_types(&events2);
        assert!(!types2.contains(&"response.function_call_arguments.done".to_string()));

        let finish_types = event_types(&conv.process_chunk(&finish_chunk(FinishReason::ToolCalls)));
        let fc_done_count = finish_types
            .iter()
            .filter(|t| *t == "response.function_call_arguments.done")
            .count();
        let item_done_count = finish_types
            .iter()
            .filter(|t| *t == "response.output_item.done")
            .count();
        assert_eq!(fc_done_count, 2);
        assert_eq!(item_done_count, 2);

        let end_types = event_types(&conv.emit_end_events());
        assert_eq!(
            end_types
                .iter()
                .filter(|t| *t == "response.function_call_arguments.done")
                .count(),
            0,
            "finish-reason completion must not repeat at EOF: {end_types:?}"
        );
        assert_eq!(
            end_types
                .iter()
                .filter(|t| *t == "response.output_item.done")
                .count(),
            0,
            "finish-reason item completion must not repeat at EOF: {end_types:?}"
        );
    }

    #[test]
    fn test_tool_call_without_finish_reason_closes_at_stream_end() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let chunk_types = event_types(&conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF\"}"),
        )));
        assert!(!chunk_types.contains(&"response.function_call_arguments.done".to_string()));

        let end_types = event_types(&conv.emit_end_events());
        assert_eq!(
            end_types
                .iter()
                .filter(|t| *t == "response.function_call_arguments.done")
                .count(),
            1
        );
        assert_eq!(
            end_types
                .iter()
                .filter(|t| *t == "response.output_item.done")
                .count(),
            1
        );
    }

    #[test]
    fn test_custom_tool_call_stream_maps_to_custom_events() {
        let patch = "*** Begin Patch\n*** Update File: a.txt\n@@\n+hello\n*** End Patch";
        let args = serde_json::json!({ "input": patch }).to_string();
        let params = ResponseParams {
            tools: Some(
                serde_json::from_value(serde_json::json!([{
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch",
                    "format": {"type": "text"},
                }]))
                .unwrap(),
            ),
            ..Default::default()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.emit_start_events();

        let chunk_types = event_types(&conv.process_chunk(&tool_call_chunk(
            0,
            Some("call_patch"),
            Some("apply_patch"),
            Some(&args),
        )));
        assert!(chunk_types.contains(&"response.output_item.added".to_string()));
        assert!(!chunk_types.contains(&"response.function_call_arguments.delta".to_string()));
        assert!(!chunk_types.contains(&"response.custom_tool_call_input.done".to_string()));

        let finish_types = event_types(&conv.process_chunk(&finish_chunk(FinishReason::ToolCalls)));
        assert!(finish_types.contains(&"response.custom_tool_call_input.done".to_string()));
        assert!(finish_types.contains(&"response.output_item.done".to_string()));
        assert!(!finish_types.contains(&"response.function_call_arguments.done".to_string()));

        let output = conv.completed_output();
        let OutputItem::CustomToolCall(call) = &output[0] else {
            panic!("expected custom tool call");
        };
        assert_eq!(call.call_id, "call_patch");
        assert_eq!(call.name, "apply_patch");
        assert_eq!(call.input, patch);
    }

    /// Text-only response: no tool-related events at all.
    #[test]
    fn test_text_only_response_no_tool_events() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let events = conv.process_chunk(&text_chunk("Hello world"));
        let types = event_types(&events);
        assert!(
            !types.contains(&"response.function_call_arguments.done".to_string()),
            "no tool events in text-only: {types:?}"
        );

        let end_events = conv.emit_end_events();
        let end_types = event_types(&end_events);
        assert!(
            end_types.contains(&"response.output_text.done".to_string()),
            "text done in end events: {end_types:?}"
        );
        assert!(
            end_types.contains(&"response.completed".to_string()),
            "completed in end events: {end_types:?}"
        );
    }

    /// Text followed by tool call: both handled correctly.
    #[test]
    fn test_text_then_tool_call() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let text_events = conv.process_chunk(&text_chunk("Let me check that."));
        let text_types = event_types(&text_events);
        assert!(
            text_types.contains(&"response.output_item.added".to_string()),
            "text message started: {text_types:?}"
        );

        let tool_events = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("search"),
            Some("{\"q\":\"rust\"}"),
        ));
        let tool_types = event_types(&tool_events);
        assert!(!tool_types.contains(&"response.function_call_arguments.done".to_string()));
        assert!(!tool_types.contains(&"response.output_item.done".to_string()));

        let end_types = event_types(&conv.emit_end_events());
        assert!(end_types.contains(&"response.function_call_arguments.done".to_string()));
        assert!(end_types.contains(&"response.output_item.done".to_string()));
    }

    #[test]
    fn test_completed_output_keeps_tool_before_later_text() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();
        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("search"),
            Some("{}"),
        ));
        let _ = conv.process_chunk(&text_chunk("Searching."));

        let output = conv.completed_output();
        assert!(matches!(output[0], OutputItem::FunctionCall(_)));
        assert!(matches!(output[1], OutputItem::Message(_)));
    }

    /// Verify that `with_context` populates `previous_response_id`
    /// in the generated Response objects.
    #[test]
    fn test_with_context_enriches_response() {
        let ctx = ResponsesContext {
            previous_response_id: Some("resp_prev_123".to_string()),
            store: true,
            ..Default::default()
        };
        let params = ResponseParams::default();
        let mut conv = ResponseStreamConverter::with_context("test-model".into(), params, ctx);

        // Process one text chunk so there's output
        let _ = conv.emit_start_events();
        let _ = conv.process_chunk(&text_chunk("Hello"));
        let _end_events = conv.emit_end_events();

        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(
            response.previous_response_id.as_deref(),
            Some("resp_prev_123")
        );
    }

    /// Without context, previous_response_id is None.
    #[test]
    fn test_without_context_defaults() {
        let params = ResponseParams::default();
        let conv = ResponseStreamConverter::new("test-model".into(), params);

        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(response.previous_response_id, None);
    }

    #[test]
    fn test_stream_response_echoes_parallel_tool_calls() {
        let params = ResponseParams {
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let conv = ResponseStreamConverter::new("test-model".into(), params);

        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(response.parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_append_chunk_events_preserves_order() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let mut events = Vec::with_capacity(4);

        conv.append_chunk_events(&text_chunk("Hello"), &mut events);

        assert_eq!(
            event_types(&events),
            vec![
                "response.output_item.added".to_string(),
                "response.content_part.added".to_string(),
                "response.output_text.delta".to_string(),
            ]
        );

        events.clear();
        conv.append_chunk_events(
            &tool_call_chunk(0, Some("call-1"), Some("lookup"), Some("{\"q\":\"x\"}")),
            &mut events,
        );

        assert_eq!(
            event_types(&events),
            vec![
                "response.output_item.added".to_string(),
                "response.function_call_arguments.delta".to_string(),
            ]
        );
    }

    #[test]
    fn test_optimized_stream_event_serializer_matches_patched_json() {
        let params = ResponseParams {
            presence_penalty: Some(0.25),
            frequency_penalty: Some(0.5),
            store: Some(true),
            ..Default::default()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params.clone());

        let response_event = ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: conv.next_seq(),
            response: conv.make_response(Status::InProgress, vec![]),
        });
        let text_event = ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
            sequence_number: conv.next_seq(),
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 0,
            delta: "line\nquote\"slash\\ cjk 漢字 emoji 🚀".to_string(),
            logprobs: Some(vec![]),
        });
        let tool_event = ResponseStreamEvent::ResponseFunctionCallArgumentsDone(
            ResponseFunctionCallArgumentsDoneEvent {
                name: Some("lookup".to_string()),
                sequence_number: conv.next_seq(),
                item_id: "fc_1".to_string(),
                output_index: 1,
                arguments: "{\"q\":\"x\"}".to_string(),
            },
        );
        let completed_event = ResponseStreamEvent::ResponseCompleted(ResponseCompletedEvent {
            sequence_number: conv.next_seq(),
            response: conv.make_response(Status::Completed, vec![]),
        });

        for event in [&response_event, &text_event, &tool_event, &completed_event] {
            assert_eq!(
                optimized_event_json(&conv, event),
                legacy_event_json(event, &params)
            );
        }

        let response_json = optimized_event_json(&conv, &response_event);
        assert_eq!(response_json["response"]["presence_penalty"], 0.25);
        assert_eq!(response_json["response"]["frequency_penalty"], 0.5);
        assert_eq!(response_json["response"]["store"], true);
        assert!(response_json["response"]["max_tool_calls"].is_null());
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Responses API SSE events.
//!
//! The event sequence follows the OpenAI Responses API streaming spec:
//! `response.created` -> `response.in_progress` -> `response.output_item.added` ->
//! `response.content_part.added` -> N x `response.output_text.delta` ->
//! `response.output_text.done` -> `response.content_part.done` ->
//! `response.output_item.done` -> `response.completed` -> `[DONE]`

use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::Event;
use dynamo_async_openai::types::responses::{
    AssistantRole, FunctionToolCall, Instructions, OutputContent, OutputItem, OutputMessage,
    OutputMessageContent, OutputStatus, OutputTextContent, Response, ResponseCompletedEvent,
    ResponseContentPartAddedEvent, ResponseContentPartDoneEvent, ResponseCreatedEvent,
    ResponseFailedEvent, ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent, ResponseInProgressEvent, ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent, ResponseStreamEvent, ResponseTextDeltaEvent,
    ResponseTextDoneEvent, ResponseTextParam, ServiceTier, Status, TextResponseFormatConfiguration,
    ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use uuid::Uuid;

use dynamo_async_openai::types::ChatCompletionMessageContent;

use super::ResponseParams;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// State machine that converts a chat completion stream into Responses API events.
pub struct ResponseStreamConverter {
    response_id: String,
    model: String,
    params: ResponseParams,
    created_at: u64,
    sequence_number: u64,
    // Text message tracking
    message_item_id: String,
    message_started: bool,
    message_output_index: u32,
    accumulated_text: String,
    // Function call tracking
    function_call_items: Vec<FunctionCallState>,
    // Output index counter
    next_output_index: u32,
}

struct FunctionCallState {
    item_id: String,
    call_id: String,
    name: String,
    accumulated_args: String,
    output_index: u32,
    started: bool,
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
            created_at,
            sequence_number: 0,
            message_item_id: format!("msg_{}", Uuid::new_v4().simple()),
            message_started: false,
            message_output_index: 0,
            accumulated_text: String::new(),
            function_call_items: Vec::new(),
            next_output_index: 0,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    fn make_response(&self, status: Status, output: Vec<OutputItem>) -> Response {
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
            frequency_penalty: Some(0.0),
            metadata: Some(serde_json::Value::Object(Default::default())),
            parallel_tool_calls: Some(true),
            presence_penalty: Some(0.0),
            // store: false because this branch does not persist responses.
            store: self.params.store.or(Some(false)),
            temperature: self.params.temperature.or(Some(1.0)),
            text: Some(ResponseTextParam {
                format: TextResponseFormatConfiguration::Text,
                verbosity: None,
            }),
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
            truncation: Some(Truncation::Disabled),
            // Nullable required fields
            billing: None,
            conversation: None,
            error: None,
            incomplete_details: None,
            instructions: self.params.instructions.clone().map(Instructions::Text),
            max_output_tokens: self.params.max_output_tokens,
            max_tool_calls: None,
            previous_response_id: None,
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            safety_identifier: None,
            service_tier: Some(ServiceTier::Auto),
            top_logprobs: Some(0),
            usage: None,
        }
    }

    /// Emit the initial lifecycle events: created + in_progress.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::with_capacity(2);

        let created = ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress, vec![]),
        });
        events.push(make_sse_event(&created));

        let in_progress = ResponseStreamEvent::ResponseInProgress(ResponseInProgressEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress, vec![]),
        });
        events.push(make_sse_event(&in_progress));

        events
    }

    /// Process a single chat completion stream chunk and return zero or more SSE events.
    pub fn process_chunk(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();

        for choice in &chunk.choices {
            let delta = &choice.delta;

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
                                status: OutputStatus::InProgress,
                            }),
                        },
                    );
                    events.push(make_sse_event(&item_added));

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
                    events.push(make_sse_event(&part_added));
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
                events.push(make_sse_event(&text_delta));
            }

            // Handle tool call deltas
            if let Some(tool_calls) = &delta.tool_calls {
                for tc in tool_calls {
                    let tc_index = tc.index as usize;

                    // Start a new function call if we haven't seen this index
                    while self.function_call_items.len() <= tc_index {
                        let output_index = self.next_output_index;
                        self.next_output_index += 1;
                        self.function_call_items.push(FunctionCallState {
                            item_id: format!("fc_{}", Uuid::new_v4().simple()),
                            call_id: String::new(),
                            name: String::new(),
                            accumulated_args: String::new(),
                            output_index,
                            started: false,
                        });
                    }

                    // Update call_id and name if provided
                    if let Some(id) = &tc.id {
                        self.function_call_items[tc_index].call_id = id.clone();
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            self.function_call_items[tc_index].name = name.clone();
                        }
                        if let Some(args) = &func.arguments {
                            // Emit output_item.added on first delta for this function call
                            if !self.function_call_items[tc_index].started {
                                self.function_call_items[tc_index].started = true;
                                let item_id = self.function_call_items[tc_index].item_id.clone();
                                let call_id = self.function_call_items[tc_index].call_id.clone();
                                let fc_name = self.function_call_items[tc_index].name.clone();
                                let output_index = self.function_call_items[tc_index].output_index;
                                let seq = self.next_seq();
                                let item_added = ResponseStreamEvent::ResponseOutputItemAdded(
                                    ResponseOutputItemAddedEvent {
                                        sequence_number: seq,
                                        output_index,
                                        item: OutputItem::FunctionCall(FunctionToolCall {
                                            id: Some(item_id),
                                            call_id,
                                            name: fc_name,
                                            arguments: String::new(),
                                            status: Some(OutputStatus::InProgress),
                                        }),
                                    },
                                );
                                events.push(make_sse_event(&item_added));
                            }

                            self.function_call_items[tc_index]
                                .accumulated_args
                                .push_str(args);
                            let item_id = self.function_call_items[tc_index].item_id.clone();
                            let output_index = self.function_call_items[tc_index].output_index;
                            let seq = self.next_seq();
                            let args_delta =
                                ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(
                                    ResponseFunctionCallArgumentsDeltaEvent {
                                        sequence_number: seq,
                                        item_id,
                                        output_index,
                                        delta: args.clone(),
                                    },
                                );
                            events.push(make_sse_event(&args_delta));
                        }
                    }
                }
            }
        }

        events
    }

    /// Emit the final events when the stream ends: done events + completed.
    pub fn emit_end_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();

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
            events.push(make_sse_event(&text_done));

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
            events.push(make_sse_event(&part_done));

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
                        status: OutputStatus::Completed,
                    }),
                });
            events.push(make_sse_event(&item_done));
        }

        // Close any function call items - collect data first to avoid borrow conflicts
        let fc_data: Vec<_> = self
            .function_call_items
            .iter()
            .filter(|fc| fc.started)
            .map(|fc| {
                (
                    fc.item_id.clone(),
                    fc.call_id.clone(),
                    fc.name.clone(),
                    fc.output_index,
                    fc.accumulated_args.clone(),
                )
            })
            .collect();
        for (item_id, call_id, fc_name, output_index, accumulated_args) in fc_data {
            let args_done = ResponseStreamEvent::ResponseFunctionCallArgumentsDone(
                ResponseFunctionCallArgumentsDoneEvent {
                    sequence_number: self.next_seq(),
                    item_id: item_id.clone(),
                    output_index,
                    arguments: accumulated_args.clone(),
                    name: Some(fc_name.clone()),
                },
            );
            events.push(make_sse_event(&args_done));

            let item_done =
                ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                    sequence_number: self.next_seq(),
                    output_index,
                    item: OutputItem::FunctionCall(FunctionToolCall {
                        id: Some(item_id),
                        call_id,
                        name: fc_name,
                        arguments: accumulated_args,
                        status: Some(OutputStatus::Completed),
                    }),
                });
            events.push(make_sse_event(&item_done));
        }

        // Build the final output vector from accumulated state
        let mut output = Vec::new();
        if self.message_started {
            output.push(OutputItem::Message(OutputMessage {
                id: self.message_item_id.clone(),
                content: vec![OutputMessageContent::OutputText(OutputTextContent {
                    text: self.accumulated_text.clone(),
                    annotations: vec![],
                    logprobs: Some(vec![]),
                })],
                role: AssistantRole::Assistant,
                status: OutputStatus::Completed,
            }));
        }
        for fc in &self.function_call_items {
            if fc.started {
                output.push(OutputItem::FunctionCall(FunctionToolCall {
                    id: Some(fc.item_id.clone()),
                    call_id: fc.call_id.clone(),
                    name: fc.name.clone(),
                    arguments: fc.accumulated_args.clone(),
                    status: Some(OutputStatus::Completed),
                }));
            }
        }

        // Emit response.completed
        let completed = ResponseStreamEvent::ResponseCompleted(ResponseCompletedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::Completed, output),
        });
        events.push(make_sse_event(&completed));

        events
    }

    /// Emit error events when the stream ends due to a backend error.
    pub fn emit_error_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();

        let failed = ResponseStreamEvent::ResponseFailed(ResponseFailedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::Failed, vec![]),
        });
        events.push(make_sse_event(&failed));

        events
    }
}

fn make_sse_event(event: &ResponseStreamEvent) -> Result<Event, anyhow::Error> {
    let event_type = get_event_type(event);
    let data = serde_json::to_string(event)?;
    Ok(Event::default().event(event_type).data(data))
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

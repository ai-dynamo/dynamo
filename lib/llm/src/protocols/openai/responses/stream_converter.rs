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
use dynamo_protocols::types::responses::{
    AssistantRole, FunctionToolCall, InputTokenDetails, Instructions, OutputContent, OutputItem,
    OutputMessage, OutputMessageContent, OutputStatus, OutputTextContent, OutputTokenDetails,
    Response, ResponseCompletedEvent, ResponseContentPartAddedEvent, ResponseContentPartDoneEvent,
    ResponseCreatedEvent, ResponseFailedEvent, ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent, ResponseInProgressEvent, ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent, ResponseStreamEvent, ResponseTextDeltaEvent,
    ResponseTextDoneEvent, ResponseTextParam, ResponseUsage, ServiceTier, Status,
    TextResponseFormatConfiguration, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use uuid::Uuid;

use dynamo_protocols::types::ChatCompletionMessageContent;

use super::ResponseParams;
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
    // Function call tracking
    function_call_items: Vec<FunctionCallState>,
    // Output index counter
    next_output_index: u32,
    // Usage stats from the backend's final chunk
    usage: Option<ResponseUsage>,
}

struct FunctionCallState {
    item_id: String,
    call_id: String,
    name: String,
    accumulated_args: String,
    output_index: u32,
    started: bool,
    /// Set when done/item_done events have already been emitted inline
    /// (complete tool call detected mid-stream). Prevents duplicate in `emit_end_events()`.
    done: bool,
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
            function_call_items: Vec::new(),
            next_output_index: 0,
            usage: None,
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
            store: self
                .api_context
                .as_ref()
                .map(|ctx| ctx.store)
                .or(self.params.store)
                .or(Some(false)),
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
            incomplete_details: None,
            instructions: self.params.instructions.clone().map(Instructions::Text),
            max_output_tokens: self.params.max_output_tokens,
            max_tool_calls: None,
            previous_response_id: self
                .api_context
                .as_ref()
                .and_then(|ctx| ctx.previous_response_id.clone()),
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: self.params.reasoning.clone(),
            safety_identifier: None,
            service_tier: Some(self.params.service_tier.unwrap_or(ServiceTier::Auto)),
            top_logprobs: Some(0),
            usage: self.usage.clone(),
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

        for choice in &chunk.inner.choices {
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
                                id: Some(self.message_item_id.clone()),
                                content: vec![],
                                role: AssistantRole::Assistant,
                                status: Some(OutputStatus::InProgress),
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
                            let output_index = self.function_call_items[tc_index].output_index;
                            let is_complete = tc.id.is_some()
                                && func.name.is_some()
                                && !self.function_call_items[tc_index].done;

                            // Clone item_id once; reused by both args_delta and (if complete) done events.
                            let item_id = self.function_call_items[tc_index].item_id.clone();
                            let seq = self.next_seq();
                            let args_delta =
                                ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(
                                    ResponseFunctionCallArgumentsDeltaEvent {
                                        sequence_number: seq,
                                        item_id: item_id.clone(),
                                        output_index,
                                        delta: args.clone(),
                                    },
                                );
                            events.push(make_sse_event(&args_delta));

                            // Emit done + output_item.done immediately if the tool call
                            // arrived complete in a single chunk (id + name + args all present).
                            // Dynamo backends emit complete tool calls, so this fires on the
                            // same chunk — no need to wait for finish_reason.
                            if is_complete {
                                self.function_call_items[tc_index].done = true;
                                // Reuse item_id from above; capture remaining values before self.next_seq()
                                let fc_item_id = item_id;
                                let fc_call_id = self.function_call_items[tc_index].call_id.clone();
                                let fc_name = self.function_call_items[tc_index].name.clone();
                                let fc_args =
                                    self.function_call_items[tc_index].accumulated_args.clone();
                                let fc_output_index =
                                    self.function_call_items[tc_index].output_index;

                                let args_done =
                                    ResponseStreamEvent::ResponseFunctionCallArgumentsDone(
                                        ResponseFunctionCallArgumentsDoneEvent {
                                            sequence_number: self.next_seq(),
                                            item_id: fc_item_id.clone(),
                                            output_index: fc_output_index,
                                            arguments: fc_args.clone(),
                                            name: Some(fc_name.clone()),
                                        },
                                    );
                                events.push(make_sse_event(&args_done));

                                let item_done = ResponseStreamEvent::ResponseOutputItemDone(
                                    ResponseOutputItemDoneEvent {
                                        sequence_number: self.next_seq(),
                                        output_index: fc_output_index,
                                        item: OutputItem::FunctionCall(FunctionToolCall {
                                            id: Some(fc_item_id),
                                            call_id: fc_call_id,
                                            name: fc_name,
                                            arguments: fc_args,
                                            status: Some(OutputStatus::Completed),
                                        }),
                                    },
                                );
                                events.push(make_sse_event(&item_done));
                            }
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
                        id: Some(self.message_item_id.clone()),
                        content: vec![OutputMessageContent::OutputText(OutputTextContent {
                            text: self.accumulated_text.clone(),
                            annotations: vec![],
                            logprobs: Some(vec![]),
                        })],
                        role: AssistantRole::Assistant,
                        status: Some(OutputStatus::Completed),
                    }),
                });
            events.push(make_sse_event(&item_done));
        }

        // Close any function call items not already done inline
        let fc_data: Vec<_> = self
            .function_call_items
            .iter()
            .filter(|fc| fc.started && !fc.done)
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
                id: Some(self.message_item_id.clone()),
                content: vec![OutputMessageContent::OutputText(OutputTextContent {
                    text: self.accumulated_text.clone(),
                    annotations: vec![],
                    logprobs: Some(vec![]),
                })],
                role: AssistantRole::Assistant,
                status: Some(OutputStatus::Completed),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::unified::ResponsesContext;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionStreamResponseDelta, ChatCompletionToolType, FunctionCallStream,
    };

    fn default_params() -> ResponseParams {
        ResponseParams {
            model: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            store: None,
            tools: None,
            tool_choice: None,
            instructions: None,
            reasoning: None,
            text: None,
            service_tier: None,
            include: None,
            truncation: None,
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

    /// Complete tool call emits function_call_arguments.done + output_item.done inline.
    #[test]
    fn test_complete_tool_call_emits_done_inline() {
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
        assert!(
            types.contains(&"response.function_call_arguments.done".to_string()),
            "should emit args done inline: {types:?}"
        );
        assert!(
            types.contains(&"response.output_item.done".to_string()),
            "should emit output_item.done inline: {types:?}"
        );

        // End events should NOT duplicate the done events
        let end_types = event_types(&conv.emit_end_events());
        assert!(
            !end_types.contains(&"response.function_call_arguments.done".to_string()),
            "done should not be duplicated in end events: {end_types:?}"
        );
        assert!(
            !end_types.contains(&"response.output_item.done".to_string())
                || end_types
                    .iter()
                    .filter(|t| *t == "response.output_item.done")
                    .count()
                    == 0,
            "output_item.done for the tool should not appear in end events"
        );
    }

    /// Multiple tool calls each get their own inline done events.
    #[test]
    fn test_multiple_tool_calls_each_emit_done_inline() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let events1 = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF\"}"),
        ));
        let types1 = event_types(&events1);
        assert!(
            types1.contains(&"response.function_call_arguments.done".to_string()),
            "first tool call done inline: {types1:?}"
        );

        let events2 = conv.process_chunk(&tool_call_chunk(
            1,
            Some("call-2"),
            Some("get_time"),
            Some("{\"tz\":\"PST\"}"),
        ));
        let types2 = event_types(&events2);
        assert!(
            types2.contains(&"response.function_call_arguments.done".to_string()),
            "second tool call done inline: {types2:?}"
        );

        // End events should have no function call done events
        let end_types = event_types(&conv.emit_end_events());
        let fc_done_count = end_types
            .iter()
            .filter(|t| *t == "response.function_call_arguments.done")
            .count();
        assert_eq!(
            fc_done_count, 0,
            "no function_call_arguments.done in end events: {end_types:?}"
        );
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
        assert!(
            tool_types.contains(&"response.function_call_arguments.done".to_string()),
            "tool call done inline after text: {tool_types:?}"
        );
        assert!(
            tool_types.contains(&"response.output_item.done".to_string()),
            "output_item.done inline after text: {tool_types:?}"
        );
    }

    /// Verify that `with_context` populates `previous_response_id` and `store`
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

        // Verify the Response object carries the context values through
        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(
            response.previous_response_id.as_deref(),
            Some("resp_prev_123")
        );
        assert_eq!(response.store, Some(true));
    }

    /// Without context, previous_response_id is None and store defaults to false.
    #[test]
    fn test_without_context_defaults() {
        let params = ResponseParams::default();
        let conv = ResponseStreamConverter::new("test-model".into(), params);

        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(response.previous_response_id, None);
        assert_eq!(response.store, Some(false));
    }
}

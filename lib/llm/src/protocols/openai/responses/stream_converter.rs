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
    AssistantRole, FunctionToolCall, OutputContent, OutputItem, OutputMessage, OutputMessageContent,
    OutputStatus, OutputTextContent, Response, ResponseCompletedEvent, ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent, ResponseCreatedEvent, ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent, ResponseInProgressEvent, ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent, ResponseStreamEvent, ResponseTextDeltaEvent,
    ResponseTextDoneEvent, Status,
};
use uuid::Uuid;

use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// State machine that converts a chat completion stream into Responses API events.
pub struct ResponseStreamConverter {
    response_id: String,
    model: String,
    created_at: u64,
    sequence_number: u64,
    // Text message tracking
    message_item_id: String,
    message_started: bool,
    accumulated_text: String,
    // Function call tracking
    function_call_items: Vec<FunctionCallState>,
    current_fc_index: Option<usize>,
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
    pub fn new(model: String) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            response_id: format!("resp_{}", Uuid::new_v4().simple()),
            model,
            created_at,
            sequence_number: 0,
            message_item_id: format!("msg_{}", Uuid::new_v4().simple()),
            message_started: false,
            accumulated_text: String::new(),
            function_call_items: Vec::new(),
            current_fc_index: None,
            next_output_index: 0,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    fn make_response(&self, status: Status) -> Response {
        Response {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_at,
            status,
            model: self.model.clone(),
            output: vec![],
            background: None,
            billing: None,
            conversation: None,
            completed_at: None,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            metadata: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            safety_identifier: None,
            service_tier: None,
            temperature: None,
            text: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_p: None,
            truncation: None,
            usage: None,
        }
    }

    /// Emit the initial lifecycle events: created + in_progress.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::with_capacity(2);

        let created = ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress),
        });
        events.push(make_sse_event(&created));

        let in_progress = ResponseStreamEvent::ResponseInProgress(ResponseInProgressEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress),
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

            // Handle text content deltas
            if let Some(content) = &delta.content {
                if !content.is_empty() {
                    // Emit output_item.added + content_part.added on first text
                    if !self.message_started {
                        self.message_started = true;
                        let output_index = self.next_output_index;
                        self.next_output_index += 1;

                        let item_added =
                            ResponseStreamEvent::ResponseOutputItemAdded(ResponseOutputItemAddedEvent {
                                sequence_number: self.next_seq(),
                                output_index,
                                item: OutputItem::Message(OutputMessage {
                                    id: self.message_item_id.clone(),
                                    content: vec![],
                                    role: AssistantRole::Assistant,
                                    status: OutputStatus::InProgress,
                                }),
                            });
                        events.push(make_sse_event(&item_added));

                        let part_added =
                            ResponseStreamEvent::ResponseContentPartAdded(ResponseContentPartAddedEvent {
                                sequence_number: self.next_seq(),
                                item_id: self.message_item_id.clone(),
                                output_index,
                                content_index: 0,
                                part: OutputContent::OutputText(OutputTextContent {
                                    text: String::new(),
                                    annotations: vec![],
                                    logprobs: None,
                                }),
                            });
                        events.push(make_sse_event(&part_added));
                    }

                    // Emit text delta
                    self.accumulated_text.push_str(content);
                    let text_delta =
                        ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
                            sequence_number: self.next_seq(),
                            item_id: self.message_item_id.clone(),
                            output_index: 0,
                            content_index: 0,
                            delta: content.clone(),
                            logprobs: None,
                        });
                    events.push(make_sse_event(&text_delta));
                }
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
                                let item_id =
                                    self.function_call_items[tc_index].item_id.clone();
                                let call_id =
                                    self.function_call_items[tc_index].call_id.clone();
                                let fc_name =
                                    self.function_call_items[tc_index].name.clone();
                                let output_index =
                                    self.function_call_items[tc_index].output_index;
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
                            let item_id =
                                self.function_call_items[tc_index].item_id.clone();
                            let output_index =
                                self.function_call_items[tc_index].output_index;
                            let seq = self.next_seq();
                            let args_delta = ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(
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

                    self.current_fc_index = Some(tc_index);
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
                output_index: 0,
                content_index: 0,
                text: self.accumulated_text.clone(),
                logprobs: None,
            });
            events.push(make_sse_event(&text_done));

            let part_done = ResponseStreamEvent::ResponseContentPartDone(ResponseContentPartDoneEvent {
                sequence_number: self.next_seq(),
                item_id: self.message_item_id.clone(),
                output_index: 0,
                content_index: 0,
                part: OutputContent::OutputText(OutputTextContent {
                    text: self.accumulated_text.clone(),
                    annotations: vec![],
                    logprobs: None,
                }),
            });
            events.push(make_sse_event(&part_done));

            let item_done = ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                sequence_number: self.next_seq(),
                output_index: 0,
                item: OutputItem::Message(OutputMessage {
                    id: self.message_item_id.clone(),
                    content: vec![OutputMessageContent::OutputText(OutputTextContent {
                        text: self.accumulated_text.clone(),
                        annotations: vec![],
                        logprobs: None,
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

            let item_done = ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
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

        // Emit response.completed
        let completed = ResponseStreamEvent::ResponseCompleted(ResponseCompletedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::Completed),
        });
        events.push(make_sse_event(&completed));

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
        ResponseStreamEvent::ResponseContentPartAdded(_) => "response.content_part.added",
        ResponseStreamEvent::ResponseOutputTextDelta(_) => "response.output_text.delta",
        ResponseStreamEvent::ResponseOutputTextDone(_) => "response.output_text.done",
        ResponseStreamEvent::ResponseContentPartDone(_) => "response.content_part.done",
        ResponseStreamEvent::ResponseOutputItemDone(_) => "response.output_item.done",
        ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(_) => {
            "response.function_call_arguments.delta"
        }
        ResponseStreamEvent::ResponseFunctionCallArgumentsDone(_) => {
            "response.function_call_arguments.done"
        }
        _ => "unknown",
    }
}

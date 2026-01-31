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
    ContentPart, OutputItem, ResponseCompleted, ResponseContentPartAdded, ResponseContentPartDone,
    ResponseCreated, ResponseEvent, ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone, ResponseInProgress, ResponseMetadata,
    ResponseOutputItemAdded, ResponseOutputItemDone, ResponseOutputTextDelta,
    ResponseOutputTextDone, Status,
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

    fn make_metadata(&self, status: Status) -> ResponseMetadata {
        ResponseMetadata {
            id: self.response_id.clone(),
            object: Some("response".to_string()),
            created_at: self.created_at,
            status,
            model: Some(self.model.clone()),
            usage: None,
            error: None,
            incomplete_details: None,
            input: None,
            instructions: None,
            max_output_tokens: None,
            background: None,
            service_tier: None,
            top_logprobs: None,
            max_tool_calls: None,
            output: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            store: None,
            temperature: None,
            text: None,
            tool_choice: None,
            tools: None,
            top_p: None,
            truncation: None,
            user: None,
            metadata: None,
            prompt_cache_key: None,
            safety_identifier: None,
        }
    }

    /// Emit the initial lifecycle events: created + in_progress.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::with_capacity(2);

        let created = ResponseEvent::ResponseCreated(ResponseCreated {
            sequence_number: self.next_seq(),
            response: self.make_metadata(Status::InProgress),
        });
        events.push(make_sse_event(&created));

        let in_progress = ResponseEvent::ResponseInProgress(ResponseInProgress {
            sequence_number: self.next_seq(),
            response: self.make_metadata(Status::InProgress),
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
                            ResponseEvent::ResponseOutputItemAdded(ResponseOutputItemAdded {
                                sequence_number: self.next_seq(),
                                output_index,
                                item: OutputItem {
                                    id: self.message_item_id.clone(),
                                    item_type: "message".to_string(),
                                    status: Some("in_progress".to_string()),
                                    content: Some(vec![]),
                                    role: Some("assistant".to_string()),
                                    summary: None,
                                },
                            });
                        events.push(make_sse_event(&item_added));

                        let part_added =
                            ResponseEvent::ResponseContentPartAdded(ResponseContentPartAdded {
                                sequence_number: self.next_seq(),
                                item_id: self.message_item_id.clone(),
                                output_index,
                                content_index: 0,
                                part: ContentPart {
                                    part_type: "output_text".to_string(),
                                    text: Some(String::new()),
                                    annotations: Some(vec![]),
                                    logprobs: None,
                                },
                            });
                        events.push(make_sse_event(&part_added));
                    }

                    // Emit text delta
                    self.accumulated_text.push_str(content);
                    let text_delta =
                        ResponseEvent::ResponseOutputTextDelta(ResponseOutputTextDelta {
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
                                let output_index =
                                    self.function_call_items[tc_index].output_index;
                                let seq = self.next_seq();
                                let item_added = ResponseEvent::ResponseOutputItemAdded(
                                    ResponseOutputItemAdded {
                                        sequence_number: seq,
                                        output_index,
                                        item: OutputItem {
                                            id: item_id,
                                            item_type: "function_call".to_string(),
                                            status: Some("in_progress".to_string()),
                                            content: None,
                                            role: None,
                                            summary: None,
                                        },
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
                            let args_delta = ResponseEvent::ResponseFunctionCallArgumentsDelta(
                                ResponseFunctionCallArgumentsDelta {
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
            let text_done = ResponseEvent::ResponseOutputTextDone(ResponseOutputTextDone {
                sequence_number: self.next_seq(),
                item_id: self.message_item_id.clone(),
                output_index: 0,
                content_index: 0,
                text: self.accumulated_text.clone(),
                logprobs: None,
            });
            events.push(make_sse_event(&text_done));

            let part_done = ResponseEvent::ResponseContentPartDone(ResponseContentPartDone {
                sequence_number: self.next_seq(),
                item_id: self.message_item_id.clone(),
                output_index: 0,
                content_index: 0,
                part: ContentPart {
                    part_type: "output_text".to_string(),
                    text: Some(self.accumulated_text.clone()),
                    annotations: Some(vec![]),
                    logprobs: None,
                },
            });
            events.push(make_sse_event(&part_done));

            let item_done = ResponseEvent::ResponseOutputItemDone(ResponseOutputItemDone {
                sequence_number: self.next_seq(),
                output_index: 0,
                item: OutputItem {
                    id: self.message_item_id.clone(),
                    item_type: "message".to_string(),
                    status: Some("completed".to_string()),
                    content: Some(vec![ContentPart {
                        part_type: "output_text".to_string(),
                        text: Some(self.accumulated_text.clone()),
                        annotations: Some(vec![]),
                        logprobs: None,
                    }]),
                    role: Some("assistant".to_string()),
                    summary: None,
                },
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
                    fc.output_index,
                    fc.accumulated_args.clone(),
                )
            })
            .collect();
        for (item_id, output_index, accumulated_args) in fc_data {
            let args_done = ResponseEvent::ResponseFunctionCallArgumentsDone(
                ResponseFunctionCallArgumentsDone {
                    sequence_number: self.next_seq(),
                    item_id: item_id.clone(),
                    output_index,
                    arguments: accumulated_args,
                },
            );
            events.push(make_sse_event(&args_done));

            let item_done = ResponseEvent::ResponseOutputItemDone(ResponseOutputItemDone {
                sequence_number: self.next_seq(),
                output_index,
                item: OutputItem {
                    id: item_id,
                    item_type: "function_call".to_string(),
                    status: Some("completed".to_string()),
                    content: None,
                    role: None,
                    summary: None,
                },
            });
            events.push(make_sse_event(&item_done));
        }

        // Emit response.completed
        let completed = ResponseEvent::ResponseCompleted(ResponseCompleted {
            sequence_number: self.next_seq(),
            response: self.make_metadata(Status::Completed),
        });
        events.push(make_sse_event(&completed));

        events
    }
}

fn make_sse_event(event: &ResponseEvent) -> Result<Event, anyhow::Error> {
    let event_type = get_event_type(event);
    let data = serde_json::to_string(event)?;
    Ok(Event::default().event(event_type).data(data))
}

fn get_event_type(event: &ResponseEvent) -> &'static str {
    match event {
        ResponseEvent::ResponseCreated(_) => "response.created",
        ResponseEvent::ResponseInProgress(_) => "response.in_progress",
        ResponseEvent::ResponseCompleted(_) => "response.completed",
        ResponseEvent::ResponseFailed(_) => "response.failed",
        ResponseEvent::ResponseIncomplete(_) => "response.incomplete",
        ResponseEvent::ResponseQueued(_) => "response.queued",
        ResponseEvent::ResponseOutputItemAdded(_) => "response.output_item.added",
        ResponseEvent::ResponseContentPartAdded(_) => "response.content_part.added",
        ResponseEvent::ResponseOutputTextDelta(_) => "response.output_text.delta",
        ResponseEvent::ResponseOutputTextDone(_) => "response.output_text.done",
        ResponseEvent::ResponseContentPartDone(_) => "response.content_part.done",
        ResponseEvent::ResponseOutputItemDone(_) => "response.output_item.done",
        ResponseEvent::ResponseFunctionCallArgumentsDelta(_) => {
            "response.function_call_arguments.delta"
        }
        ResponseEvent::ResponseFunctionCallArgumentsDone(_) => {
            "response.function_call_arguments.done"
        }
        _ => "unknown",
    }
}

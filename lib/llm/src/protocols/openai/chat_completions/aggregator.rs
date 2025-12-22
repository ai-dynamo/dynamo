// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};
use std::collections::HashMap;

use super::{NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse};
use crate::protocols::{
    Annotated,
    codec::{Message, SseCodecError},
    convert_sse_stream,
    openai::ParsingOptions,
};

use dynamo_async_openai::types::StopReason;
use dynamo_runtime::engine::DataStream;

/// Aggregates a stream of [`NvCreateChatCompletionStreamResponse`]s into a single
/// [`NvCreateChatCompletionResponse`]. This struct accumulates incremental responses
/// from a streaming OpenAI API call into a complete final response.
pub struct DeltaAggregator {
    /// Unique identifier for the chat completion.
    id: String,
    /// Model name used for the chat completion.
    model: String,
    /// Timestamp (Unix epoch) indicating when the response was created.
    created: u32,
    /// Optional usage statistics for the completion request.
    usage: Option<dynamo_async_openai::types::CompletionUsage>,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Map of incremental response choices, keyed by index.
    choices: HashMap<u32, DeltaChoice>,
    /// Optional error message if an error occurs during aggregation.
    error: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<dynamo_async_openai::types::ServiceTierResponse>,
    /// Aggregated nvext field from stream responses
    nvext: Option<serde_json::Value>,
}

/// Represents the accumulated state of a single chat choice during streaming aggregation.
#[derive(Debug)]
struct DeltaChoice {
    /// The index of the choice in the completion.
    index: u32,
    /// The accumulated text content for the choice.
    text: String,
    /// The role associated with this message (e.g., `system`, `user`, `assistant`).
    role: Option<dynamo_async_openai::types::Role>,
    /// The reason the completion was finished (if applicable).
    finish_reason: Option<dynamo_async_openai::types::FinishReason>,
    /// The stop string or token that triggered the stop condition.
    stop_reason: Option<StopReason>,
    /// Optional log probabilities for the chat choice.
    logprobs: Option<dynamo_async_openai::types::ChatChoiceLogprobs>,
    // Optional tool calls for the chat choice.
    tool_calls: Option<Vec<dynamo_async_openai::types::ChatCompletionMessageToolCall>>,

    /// Optional reasoning content for the chat choice.
    reasoning_content: Option<String>,
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
    fn default() -> Self {
        Self::new()
    }
}

fn convert_tool_chunk_to_message_tool_call(
    chunk: &dynamo_async_openai::types::ChatCompletionMessageToolCallChunk,
) -> Option<dynamo_async_openai::types::ChatCompletionMessageToolCall> {
    // Convert ChatCompletionMessageToolCallChunk to ChatCompletionMessageToolCall
    if let (Some(id), Some(r#type), Some(function)) = (&chunk.id, &chunk.r#type, &chunk.function) {
        if let (Some(name), Some(arguments)) = (&function.name, &function.arguments) {
            Some(dynamo_async_openai::types::ChatCompletionMessageToolCall {
                id: id.clone(),
                r#type: r#type.clone(),
                function: dynamo_async_openai::types::FunctionCall {
                    name: name.clone(),
                    arguments: arguments.clone(),
                },
            })
        } else {
            None
        }
    } else {
        None
    }
}

impl DeltaAggregator {
    /// Creates a new, empty [`DeltaAggregator`] instance.
    pub fn new() -> Self {
        Self {
            id: "".to_string(),
            model: "".to_string(),
            created: 0,
            usage: None,
            system_fingerprint: None,
            choices: HashMap::new(),
            error: None,
            service_tier: None,
            nvext: None,
        }
    }

    /// Aggregates a stream of [`NvCreateChatCompletionStreamResponse`]s into a single
    /// [`NvCreateChatCompletionResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation is successful.
    /// * `Err(String)` if an error occurs during processing.
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
        _parsing_options: ParsingOptions,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                // Attempt to unwrap the delta, capturing any errors.
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none() && delta.data.is_some() {
                    // Extract the data payload from the delta.
                    let delta = delta.data.unwrap();
                    aggregator.id = delta.id;
                    aggregator.model = delta.model;
                    aggregator.created = delta.created;
                    aggregator.service_tier = delta.service_tier;

                    // Aggregate usage statistics if available.
                    if let Some(usage) = delta.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // Aggregate nvext field (take the last non-None value)
                    if delta.nvext.is_some() {
                        aggregator.nvext = delta.nvext;
                    }

                    // Aggregate choices incrementally.
                    for choice in delta.choices {
                        let state_choice =
                            aggregator
                                .choices
                                .entry(choice.index)
                                .or_insert(DeltaChoice {
                                    index: choice.index,
                                    text: "".to_string(),
                                    role: choice.delta.role,
                                    finish_reason: None,
                                    stop_reason: None,
                                    logprobs: None,
                                    tool_calls: None,
                                    reasoning_content: None,
                                });
                        // Append content if available.
                        if let Some(content) = &choice.delta.content {
                            state_choice.text.push_str(content);
                        }

                        if let Some(reasoning_content) = &choice.delta.reasoning_content {
                            state_choice
                                .reasoning_content
                                .get_or_insert_with(String::new)
                                .push_str(reasoning_content);
                        }

                        // Since one tool call is one chunk, we don't need to aggregate them
                        // We just need to convert the ChatCompletionMessageToolCallChunk to ChatCompletionMessageToolCall and append to the state_choice.tool_calls
                        if let Some(tool_calls) = &choice.delta.tool_calls
                            && !tool_calls.is_empty()
                        {
                            // Convert ChatCompletionMessageToolCallChunk to ChatCompletionMessageToolCall
                            let converted_tool_calls: Vec<
                                dynamo_async_openai::types::ChatCompletionMessageToolCall,
                            > = tool_calls
                                .iter()
                                .filter_map(convert_tool_chunk_to_message_tool_call)
                                .collect();

                            // Initialize and push the converted tool calls to state_choice.tool_calls
                            // Only set tool_calls to Some if there are actual tool calls
                            if !converted_tool_calls.is_empty() {
                                if let Some(existing_tool_calls) = &mut state_choice.tool_calls {
                                    existing_tool_calls.extend(converted_tool_calls);
                                } else {
                                    state_choice.tool_calls = Some(converted_tool_calls);
                                }
                            }
                        }

                        // Update finish reason if provided.
                        if let Some(finish_reason) = choice.finish_reason {
                            state_choice.finish_reason = Some(finish_reason);
                        }

                        // Update stop reason if provided.
                        if let Some(stop_reason) = choice.stop_reason {
                            state_choice.stop_reason = Some(stop_reason);
                        }

                        // Update logprobs
                        if let Some(logprobs) = &choice.logprobs {
                            let state_lps = state_choice.logprobs.get_or_insert(
                                dynamo_async_openai::types::ChatChoiceLogprobs {
                                    content: None,
                                    refusal: None,
                                },
                            );
                            if let Some(content_lps) = &logprobs.content {
                                state_lps
                                    .content
                                    .get_or_insert(Vec::new())
                                    .extend(content_lps.clone());
                            }
                            if let Some(refusal_lps) = &logprobs.refusal {
                                state_lps
                                    .refusal
                                    .get_or_insert(Vec::new())
                                    .extend(refusal_lps.clone());
                            }
                        }
                    }
                }
                aggregator
            })
            .await;

        // Return early if an error was encountered.
        if let Some(error) = aggregator.error {
            return Err(error);
        }

        // Extract aggregated choices and sort them by index.
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(dynamo_async_openai::types::ChatChoice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        // Construct the final response object.
        let response = NvCreateChatCompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "chat.completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
            service_tier: aggregator.service_tier,
            nvext: aggregator.nvext,
        };

        Ok(response)
    }
}

#[allow(deprecated)]
impl From<DeltaChoice> for dynamo_async_openai::types::ChatChoice {
    /// Converts a [`DeltaChoice`] into an [`dynamo_async_openai::types::ChatChoice`].
    ///
    /// # Note
    /// The `function_call` field is deprecated.
    fn from(delta: DeltaChoice) -> Self {
        // If tool calls are present and non-empty, finish reason should be ToolCalls
        let finish_reason = if delta
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            Some(dynamo_async_openai::types::FinishReason::ToolCalls)
        } else {
            delta.finish_reason
        };

        dynamo_async_openai::types::ChatChoice {
            message: dynamo_async_openai::types::ChatCompletionResponseMessage {
                role: delta.role.expect("delta should have a Role"),
                content: if delta.text.is_empty() {
                    None
                } else {
                    Some(delta.text)
                },
                tool_calls: delta.tool_calls,
                refusal: None,
                function_call: None,
                audio: None,
                reasoning_content: delta.reasoning_content,
            },
            index: delta.index,
            finish_reason,
            stop_reason: delta.stop_reason,
            logprobs: delta.logprobs,
        }
    }
}

/// Trait for aggregating chat completion responses from streams.
/// Setting this macro because our async functions are not used outside of the library
#[allow(async_fn_in_trait)]
pub trait ChatCompletionAggregator {
    /// Aggregates an annotated stream of chat completion responses into a final response.
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateChatCompletionResponse, String>;

    /// Converts an SSE stream into a [`NvCreateChatCompletionResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of SSE messages containing chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateChatCompletionResponse, String>;
}

impl ChatCompletionAggregator for dynamo_async_openai::types::CreateChatCompletionResponse {
    async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        DeltaAggregator::apply(stream, parsing_options).await
    }

    async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        let stream = convert_sse_stream::<NvCreateChatCompletionStreamResponse>(stream);
        NvCreateChatCompletionResponse::from_annotated_stream(stream, parsing_options).await
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use futures::stream;

    #[allow(deprecated)]
    fn create_test_delta(
        index: u32,
        text: &str,
        role: Option<dynamo_async_openai::types::Role>,
        finish_reason: Option<dynamo_async_openai::types::FinishReason>,
        logprob: Option<f32>,
        tool_calls: Option<&str>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        // ALLOW: function_call is deprecated

        let tool_calls: Option<serde_json::Value> =
            tool_calls.map(|tool_calls| serde_json::from_str(tool_calls).unwrap());

        let tool_call_chunks = if let Some(tool_calls) = tool_calls {
            Some(vec![
                dynamo_async_openai::types::ChatCompletionMessageToolCallChunk {
                    index: 0,
                    id: Some("test_id".to_string()),
                    r#type: Some(dynamo_async_openai::types::ChatCompletionToolType::Function),
                    function: Some(dynamo_async_openai::types::FunctionCallStream {
                        name: tool_calls["name"].as_str().map(|s| s.to_string()),
                        arguments: Some(serde_json::to_string(&tool_calls["arguments"]).unwrap()),
                    }),
                },
            ])
        } else {
            None
        };

        let delta = dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
            content: Some(text.to_string()),
            function_call: None,
            tool_calls: tool_call_chunks,
            role,
            refusal: None,
            reasoning_content: None,
        };
        let logprobs = logprob.map(|lp| dynamo_async_openai::types::ChatChoiceLogprobs {
            content: Some(vec![
                dynamo_async_openai::types::ChatCompletionTokenLogprob {
                    token: text.to_string(),
                    logprob: lp,
                    bytes: None,
                    top_logprobs: vec![],
                },
            ]),
            refusal: None,
        });
        let choice = dynamo_async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            stop_reason: None,
            logprobs,
        };

        let data = NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b-instruct".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
            nvext: None,
        };

        Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        // Create an empty stream
        let stream: DataStream<Annotated<NvCreateChatCompletionStreamResponse>> =
            Box::pin(stream::empty());

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify that the response is empty and has default values
        assert_eq!(response.id, "");
        assert_eq!(response.model, "");
        assert_eq!(response.created, 0);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 0);
        assert!(response.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta = create_test_delta(
            0,
            "Hello,",
            Some(dynamo_async_openai::types::Role::User),
            None,
            None,
            None,
        );

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.id, "test_id");
        assert_eq!(response.model, "meta/llama-3.1-8b-instruct");
        assert_eq!(response.created, 1234567890);
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello,");
        assert!(choice.finish_reason.is_none());
        assert_eq!(choice.message.role, dynamo_async_openai::types::Role::User);
        assert!(response.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 = create_test_delta(
            0,
            "Hello,",
            Some(dynamo_async_openai::types::Role::User),
            None,
            Some(-0.1),
            None,
        );
        let annotated_delta2 = create_test_delta(
            0,
            " world!",
            None,
            Some(dynamo_async_openai::types::FinishReason::Stop),
            Some(-0.2),
            None,
        );

        // Create a stream
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello, world!");
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice.message.role, dynamo_async_openai::types::Role::User);
        assert_eq!(
            choice
                .logprobs
                .as_ref()
                .unwrap()
                .content
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            choice.logprobs.as_ref().unwrap().content.as_ref().unwrap()[0].logprob,
            -0.1
        );
        assert_eq!(
            choice.logprobs.as_ref().unwrap().content.as_ref().unwrap()[1].logprob,
            -0.2
        );
    }

    #[tokio::test]
    async fn test_preserves_intermediate_whitespace_chunks() {
        // This validates behavior before/after removing trim_end():
        // If a whitespace-only chunk (" ") arrives between tokens, it must be preserved.
        // With trim_end(), that chunk was dropped, yielding "Helloworld" instead of "Hello world".

        let annotated_delta1 = create_test_delta(
            0,
            "Hello",
            Some(dynamo_async_openai::types::Role::User),
            None,
            None,
            None,
        );
        // A whitespace-only chunk
        let annotated_delta2 = create_test_delta(0, " ", None, None, None, None);
        let annotated_delta3 = create_test_delta(
            0,
            "world",
            None,
            Some(dynamo_async_openai::types::FinishReason::Stop),
            None,
            None,
        );

        let stream = Box::pin(stream::iter(vec![
            annotated_delta1,
            annotated_delta2,
            annotated_delta3,
        ]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_deref(), Some("Hello world"));
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice.message.role, dynamo_async_openai::types::Role::User);
    }

    #[allow(deprecated)]
    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        // ALLOW: function_call is deprecated
        let data = NvCreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "test_model".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![
                dynamo_async_openai::types::ChatChoiceStream {
                    index: 0,
                    delta: dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
                        role: Some(dynamo_async_openai::types::Role::Assistant),
                        content: Some("Choice 0".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_async_openai::types::FinishReason::Stop),
                    stop_reason: None,
                    logprobs: None,
                },
                dynamo_async_openai::types::ChatChoiceStream {
                    index: 1,
                    delta: dynamo_async_openai::types::ChatCompletionStreamResponseDelta {
                        role: Some(dynamo_async_openai::types::Role::Assistant),
                        content: Some("Choice 1".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_async_openai::types::FinishReason::Stop),
                    stop_reason: None,
                    logprobs: None,
                },
            ],
            object: "chat.completion".to_string(),
            nvext: None,
        };

        // Wrap it in Annotated and create a stream
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let mut response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.choices.len(), 2);
        response.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(choice0.message.content.as_ref().unwrap(), "Choice 0");
        assert_eq!(
            choice0.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::Stop)
        );
        assert_eq!(
            choice0.message.role,
            dynamo_async_openai::types::Role::Assistant
        );

        let choice1 = &response.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(choice1.message.content.as_ref().unwrap(), "Choice 1");
        assert_eq!(
            choice1.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::Stop)
        );
        assert_eq!(
            choice1.message.role,
            dynamo_async_openai::types::Role::Assistant
        );
    }

    #[tokio::test]
    async fn test_tool_calling_finish_reason_override_from_stop() {
        // Test that when tool calls are present but finish reason is Stop, it gets overridden to ToolCalls
        let tool_call_json =
            r#"{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}"#;

        let annotated_delta = create_test_delta(
            0,
            "I'll check the weather for you.",
            Some(dynamo_async_openai::types::Role::Assistant),
            Some(dynamo_async_openai::types::FinishReason::Stop), // Original finish reason is Stop
            None,
            Some(tool_call_json),
        );

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);

        // Most importantly, verify that finish reason was overridden to ToolCalls despite original being Stop
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_tool_calling_finish_reason_override_from_length() {
        // Test that when tool calls are present but finish reason is Length, it gets overridden to ToolCalls
        let tool_call_json = r#"{"name": "search", "arguments": {"query": "rust programming"}}"#;

        let annotated_delta = create_test_delta(
            0,
            "Let me search for that.",
            Some(dynamo_async_openai::types::Role::Assistant),
            Some(dynamo_async_openai::types::FinishReason::Length), // Original finish reason is Length
            None,
            Some(tool_call_json),
        );

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);

        // Verify that finish reason was overridden to ToolCalls despite original being Length
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_tool_calling_finish_reason_override_from_none() {
        // Test that when tool calls are present but finish reason is None, it gets set to ToolCalls
        let tool_call_json = r#"{"name": "calculate", "arguments": {"expression": "2+2"}}"#;

        let annotated_delta = create_test_delta(
            0,
            "I'll calculate that for you.",
            Some(dynamo_async_openai::types::Role::Assistant),
            None, // Original finish reason is None
            None,
            Some(tool_call_json),
        );

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);

        // Verify that finish reason was set to ToolCalls despite original being None
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_no_tool_calling_preserves_original_finish_reason() {
        // Test that when no tool calls are present, the original finish reason is preserved
        let annotated_delta = create_test_delta(
            0,
            "This is a regular response without tool calls.",
            Some(dynamo_async_openai::types::Role::Assistant),
            Some(dynamo_async_openai::types::FinishReason::Stop),
            None,
            None, // No tool calls
        );

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        // Verify no tool calls are present
        assert!(choice.message.tool_calls.is_none());

        // Verify that original finish reason (Stop) is preserved
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn test_empty_tool_calls_preserves_original_finish_reason() {
        // Test that when tool calls array is empty, the original finish reason is preserved
        // Create a delta with empty tool calls by modifying the create_test_delta output
        let mut annotated_delta = create_test_delta(
            0,
            "Response with empty tool calls array.",
            Some(dynamo_async_openai::types::Role::Assistant),
            Some(dynamo_async_openai::types::FinishReason::Length),
            None,
            None,
        );

        // Manually set empty tool calls array
        if let Some(ref mut data) = annotated_delta.data {
            data.choices[0].delta.tool_calls = Some(vec![]); // Empty tool calls array
        }

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        // Verify tool calls array is empty
        assert!(choice.message.tool_calls.is_none());

        // Verify that original finish reason (Length) is preserved since tool calls are empty
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::Length)
        );
    }

    #[tokio::test]
    async fn test_tool_calling_output() {
        // Simulate a delta with a tool call in the content
        let tool_call_json = r#"{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#;

        // Use create_test_delta to generate the annotated delta, then extract the inner delta for the test
        let annotated_delta = create_test_delta(
            0,
            "Hey Dude ! What's the weather in San Francisco in Fahrenheit?",
            Some(dynamo_async_openai::types::Role::Assistant),
            Some(dynamo_async_openai::types::FinishReason::ToolCalls),
            None,
            Some(tool_call_json),
        );
        let data = annotated_delta.data.unwrap();

        // Wrap it in Annotated and create a stream
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // There should be one choice
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        // The tool_calls field should be present and parsed
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);

        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.function.name, "get_weather");
        // The arguments should be a JSON string containing the expected keys
        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");

        assert_eq!(
            choice.message.content.as_ref().unwrap(),
            "Hey Dude ! What's the weather in San Francisco in Fahrenheit?"
        );

        // The finish_reason should be ToolCalls
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls)
        );
        assert_eq!(
            choice.message.role,
            dynamo_async_openai::types::Role::Assistant
        );
    }

    #[tokio::test]
    async fn test_tool_calling_finish_reason_override_from_stop_alternative() {
        // Test that when tool calls are present but finish reason is Stop, it gets overridden to ToolCalls
        let tool_call_json =
            r#"{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}"#;

        let annotated_delta = create_test_delta(
            0,
            "Getting weather for New York",
            Some(dynamo_async_openai::types::Role::Assistant),
            Some(dynamo_async_openai::types::FinishReason::Stop), // This should be overridden
            None,
            Some(tool_call_json),
        );

        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // There should be one choice
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];

        // The finish_reason should be ToolCalls, not Stop, because tool calls are present
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls)
        );

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }
}

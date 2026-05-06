// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};
use std::collections::{BTreeMap, HashMap};

use dynamo_parsers::tool_calling::try_tool_call_parse_aggregate;

use super::{NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse};
use crate::protocols::{
    Annotated,
    codec::{Message, SseCodecError},
    convert_sse_stream,
    openai::ParsingOptions,
};

use dynamo_protocols::types::{ChatCompletionMessageContent, StopReason};
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
    usage: Option<dynamo_protocols::types::CompletionUsage>,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Map of incremental response choices, keyed by index.
    choices: HashMap<u32, DeltaChoice>,
    /// Optional error message if an error occurs during aggregation.
    error: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<dynamo_protocols::types::ServiceTierResponse>,
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
    role: Option<dynamo_protocols::types::Role>,
    /// The reason the completion was finished (if applicable).
    finish_reason: Option<dynamo_protocols::types::FinishReason>,
    /// The stop string or token that triggered the stop condition.
    stop_reason: Option<StopReason>,
    /// Optional log probabilities for the chat choice.
    logprobs: Option<dynamo_protocols::types::ChatChoiceLogprobs>,
    // Tool-call chunks accumulated in the order they arrived from the stream,
    // keyed by `index` so chunks that carry only argument fragments can be
    // merged into the entry created by the initial (id + name) chunk.
    // BTreeMap preserves deterministic iteration order on the index dimension.
    // See [`merge_tool_call_chunk`] for per-field merge semantics.
    // #8640: replaces the old `Option<Vec<ChatCompletionMessageToolCall>>`
    // which required id/name/arguments to all be set on the same chunk.
    tool_call_chunks: BTreeMap<u32, dynamo_protocols::types::ChatCompletionMessageToolCallChunk>,
    // Optional tool calls for the chat choice, populated *after* fold either
    // by finalizing `tool_call_chunks` above, or by
    // `try_tool_call_parse_aggregate` running against `text` for producers
    // that put tool calls in content rather than as structured chunks.
    tool_calls: Option<Vec<dynamo_protocols::types::ChatCompletionMessageToolCall>>,

    /// Optional reasoning content for the chat choice.
    reasoning_content: Option<String>,

    /// Accumulated content parts for multimodal responses
    content_parts: Vec<dynamo_protocols::types::ChatCompletionResponseContentPart>,
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// Merge an incoming chunk into the per-index accumulator.
///
/// #8640: the prior implementation required `id`, `name`, and `arguments`
/// all on the same chunk, and thus the argument-fragment deltas were dropped
/// and the client saw `arguments: ""`.
///
/// The fix here merges by `index` across deltas: `id`, `type`, `function.name`
/// first-wins; `function.arguments` concatenated across fragments. This matches
/// the OpenAI streaming spec and vLLM/SGLang hermes emission:
///
/// * delta 1: `{index, id, type, function: { name }}`
/// * delta 2..N: `{index, function: { arguments: "<fragment>" }}`
fn merge_tool_call_chunk(
    existing: &mut dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
    incoming: dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
) {
    if existing.id.is_none()
        && let Some(id) = incoming.id
    {
        existing.id = Some(id);
    }
    if existing.r#type.is_none()
        && let Some(ty) = incoming.r#type
    {
        existing.r#type = Some(ty);
    }
    let Some(incoming_fn) = incoming.function else {
        return;
    };
    match &mut existing.function {
        None => existing.function = Some(incoming_fn),
        Some(existing_fn) => {
            if existing_fn.name.is_none()
                && let Some(name) = incoming_fn.name
            {
                existing_fn.name = Some(name);
            }
            if let Some(args_fragment) = incoming_fn.arguments {
                existing_fn
                    .arguments
                    .get_or_insert_with(String::new)
                    .push_str(&args_fragment);
            }
        }
    }
}

/// Convert a fully merged chunk (post-merge accumulator state) to a finalized
/// `ChatCompletionMessageToolCall`. Returns `None` only if `id` or
/// `function.name` never arrived across any chunk — those are required by the
/// final OpenAI response schema. Missing `arguments` is legal (empty-args
/// tool calls) and becomes `""`. A warning is logged on drop so a producer
/// bug in upstream (e.g. vLLM / SGLang emitting fragments without ever
/// establishing the id+name opener) doesn't silently eat a tool call the
/// way the pre-fix code did.
fn finalize_merged_tool_chunk(
    chunk: dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
) -> Option<dynamo_protocols::types::ChatCompletionMessageToolCall> {
    let index = chunk.index;
    let Some(id) = chunk.id else {
        tracing::warn!(
            tool_call_index = index,
            "dropping merged tool-call chunk: no `id` arrived across any delta"
        );
        return None;
    };
    let Some(function) = chunk.function else {
        tracing::warn!(
            tool_call_index = index,
            tool_call_id = %id,
            "dropping merged tool-call chunk: no `function` arrived across any delta"
        );
        return None;
    };
    let Some(name) = function.name else {
        tracing::warn!(
            tool_call_index = index,
            tool_call_id = %id,
            "dropping merged tool-call chunk: no `function.name` arrived across any delta"
        );
        return None;
    };
    Some(dynamo_protocols::types::ChatCompletionMessageToolCall {
        id,
        // Use the merged r#type if the stream carried one. Falls back to
        // `Function` — today the only variant in the OpenAI schema, but
        // threading the merged value keeps us forward-compat if variants
        // are added later and avoids dead state in `merge_tool_call_chunk`.
        r#type: chunk
            .r#type
            .unwrap_or(dynamo_protocols::types::FunctionType::Function),
        function: dynamo_protocols::types::FunctionCall {
            name,
            arguments: function.arguments.unwrap_or_default(),
        },
    })
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
        parsing_options: ParsingOptions,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        let mut aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                // Attempt to unwrap the delta, capturing any errors.
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none()
                    && let Some(delta) = delta.data
                {
                    aggregator.id = delta.inner.id;
                    aggregator.model = delta.inner.model;
                    aggregator.created = delta.inner.created;
                    aggregator.service_tier = delta.inner.service_tier;

                    // Aggregate usage statistics if available.
                    if let Some(usage) = delta.inner.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.inner.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // Aggregate nvext field (take the last non-None value)
                    if delta.nvext.is_some() {
                        aggregator.nvext = delta.nvext;
                    }

                    // Aggregate choices incrementally.
                    for choice in delta.inner.choices {
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
                                    tool_call_chunks: BTreeMap::new(),
                                    tool_calls: None,
                                    reasoning_content: None,
                                    content_parts: Vec::new(),
                                });
                        // Handle content based on type
                        if let Some(content) = &choice.delta.content {
                            match content {
                                ChatCompletionMessageContent::Text(text) => {
                                    state_choice.text.push_str(text);
                                }
                                ChatCompletionMessageContent::Parts(parts) => {
                                    state_choice.content_parts.extend(parts.clone());
                                }
                            }
                        }

                        if let Some(reasoning_content) = &choice.delta.reasoning_content {
                            state_choice
                                .reasoning_content
                                .get_or_insert_with(String::new)
                                .push_str(reasoning_content);
                        }

                        // #8640: streaming producers split a single tool call across
                        // multiple deltas (delta 1 = id + name; delta 2..N = argument
                        // fragments), so we merge chunks into a per-index accumulator
                        // here instead of treating each chunk as a complete tool call.
                        // Finalization to `tool_calls` happens after the fold.
                        if let Some(incoming_chunks) = choice.delta.tool_calls {
                            for chunk in incoming_chunks {
                                let entry = state_choice
                                    .tool_call_chunks
                                    .entry(chunk.index)
                                    .or_insert_with(|| {
                                        dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
                                            index: chunk.index,
                                            id: None,
                                            r#type: None,
                                            function: None,
                                        }
                                    });
                                merge_tool_call_chunk(entry, chunk);
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
                                dynamo_protocols::types::ChatChoiceLogprobs {
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

        // #8640: finalize the per-index tool-call chunk accumulator into the
        // choice's `tool_calls` vector. Chunks missing id or name across the
        // whole stream are dropped here (they're not a valid tool call in the
        // final schema), but chunks missing only `arguments` get defaulted to
        // "" — the old code dropped those entirely.
        for choice in aggregator.choices.values_mut() {
            if choice.tool_call_chunks.is_empty() {
                continue;
            }
            let finalized: Vec<_> = std::mem::take(&mut choice.tool_call_chunks)
                .into_values()
                .filter_map(finalize_merged_tool_chunk)
                .collect();
            // choice.tool_calls is always None at this point: or_insert
            // initializes it to None, try_tool_call_parse_aggregate runs
            // strictly after this loop. Unconditional assign is the only
            // reachable path; no merge-with-existing needed.
            if !finalized.is_empty() {
                choice.tool_calls = Some(finalized);
            }
        }

        if let Some(parser) = parsing_options.tool_call_parser.as_deref() {
            for choice in aggregator.choices.values_mut() {
                if choice
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                    || choice.text.is_empty()
                {
                    continue;
                }

                let (tool_calls, content) =
                    match try_tool_call_parse_aggregate(&choice.text, Some(parser), None).await {
                        Ok(result) => result,
                        Err(error) => {
                            tracing::debug!(
                                error = %error,
                                parser,
                                "failed to parse aggregated chat tool calls"
                            );
                            continue;
                        }
                    };

                if !tool_calls.is_empty() {
                    choice.tool_calls = Some(tool_calls);
                    choice.text = content.unwrap_or_default();
                }
            }
        }

        // Extract aggregated choices and sort them by index.
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(dynamo_protocols::types::ChatChoice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        // Construct the final response object.
        let response = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: aggregator.id,
                created: aggregator.created,
                usage: aggregator.usage,
                model: aggregator.model,
                object: "chat.completion".to_string(),
                system_fingerprint: aggregator.system_fingerprint,
                choices,
                service_tier: aggregator.service_tier,
            },
            nvext: aggregator.nvext,
        };

        Ok(response)
    }
}

#[allow(deprecated)]
impl From<DeltaChoice> for dynamo_protocols::types::ChatChoice {
    /// Converts a [`DeltaChoice`] into an [`dynamo_protocols::types::ChatChoice`].
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
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        } else {
            delta.finish_reason
        };

        // Determine content format based on what we accumulated
        let content = if !delta.content_parts.is_empty() {
            // Multimodal response with content parts
            Some(ChatCompletionMessageContent::Parts(delta.content_parts))
        } else if !delta.text.is_empty() {
            // Text-only response (backward compatible)
            Some(ChatCompletionMessageContent::Text(delta.text))
        } else {
            None
        };

        dynamo_protocols::types::ChatChoice {
            message: dynamo_protocols::types::ChatCompletionResponseMessage {
                role: delta.role.expect("delta should have a Role"),
                content,
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

impl ChatCompletionAggregator for NvCreateChatCompletionResponse {
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
    use crate::protocols::openai::token_to_utf8_bytes;
    use futures::stream;

    #[allow(deprecated)]
    fn create_test_delta(
        index: u32,
        text: &str,
        role: Option<dynamo_protocols::types::Role>,
        finish_reason: Option<dynamo_protocols::types::FinishReason>,
        logprob: Option<f32>,
        tool_calls: Option<&str>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        // ALLOW: function_call is deprecated

        let tool_calls: Option<serde_json::Value> =
            tool_calls.map(|tool_calls| serde_json::from_str(tool_calls).unwrap());

        let tool_call_chunks = if let Some(tool_calls) = tool_calls {
            Some(vec![
                dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
                    index: 0,
                    id: Some("test_id".to_string()),
                    r#type: Some(dynamo_protocols::types::FunctionType::Function),
                    function: Some(dynamo_protocols::types::FunctionCallStream {
                        name: tool_calls["name"].as_str().map(|s| s.to_string()),
                        arguments: Some(serde_json::to_string(&tool_calls["arguments"]).unwrap()),
                    }),
                },
            ])
        } else {
            None
        };

        let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
            content: Some(ChatCompletionMessageContent::Text(text.to_string())),
            function_call: None,
            tool_calls: tool_call_chunks,
            role,
            refusal: None,
            reasoning_content: None,
        };
        let logprobs = logprob.map(|lp| {
            let token = text.to_string();
            dynamo_protocols::types::ChatChoiceLogprobs {
                content: Some(vec![dynamo_protocols::types::ChatCompletionTokenLogprob {
                    token: token.clone(),
                    logprob: lp,
                    bytes: token_to_utf8_bytes(&token),
                    top_logprobs: vec![],
                }]),
                refusal: None,
            }
        });
        let choice = dynamo_protocols::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            stop_reason: None,
            logprobs,
        };

        let data = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test_id".to_string(),
                model: "meta/llama-3.1-8b-instruct".to_string(),
                created: 1234567890,
                service_tier: None,
                usage: None,
                system_fingerprint: None,
                choices: vec![choice],
                object: "chat.completion".to_string(),
            },
            nvext: None,
        };

        Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        }
    }

    /// Build a stream delta carrying a raw list of tool-call chunks (no content).
    /// Used by multi-chunk tests that mimic vLLM hermes' streaming emission:
    /// the first chunk carries `id` + `function.name` only, subsequent chunks
    /// carry `function.arguments` fragments with neither `id` nor `name`.
    fn create_test_delta_with_tool_chunks(
        index: u32,
        tool_chunks: Vec<dynamo_protocols::types::ChatCompletionMessageToolCallChunk>,
        finish_reason: Option<dynamo_protocols::types::FinishReason>,
        role: Option<dynamo_protocols::types::Role>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
            content: None,
            function_call: None,
            tool_calls: Some(tool_chunks),
            role,
            refusal: None,
            reasoning_content: None,
        };
        let choice = dynamo_protocols::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            stop_reason: None,
            logprobs: None,
        };
        let data = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test_id".to_string(),
                model: "meta/llama-3.1-8b-instruct".to_string(),
                created: 1234567890,
                service_tier: None,
                usage: None,
                system_fingerprint: None,
                choices: vec![choice],
                object: "chat.completion".to_string(),
            },
            nvext: None,
        };
        Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        }
    }

    /// Repro for [#8640](https://github.com/ai-dynamo/dynamo/issues/8640):
    /// vLLM hermes (and any spec-compliant OpenAI tool-call streaming producer)
    /// splits a single tool call across multiple deltas:
    ///   delta 1: `{index: 0, id: "tc1", type: function, function: {name: "get_weather"}}`
    ///   delta 2: `{index: 0, function: {arguments: "{\"city\":"}}`
    ///   delta 3: `{index: 0, function: {arguments: "\"Tokyo\"}"}}`
    /// The aggregated non-stream response must reconstruct
    /// `arguments = "{\"city\":\"Tokyo\"}"`. Before the fix,
    /// `convert_tool_chunk_to_message_tool_call` requires id/name/arguments all
    /// set on the *same* chunk and drops the argument-fragment chunks — so the
    /// client sees `arguments: ""`.
    #[tokio::test]
    async fn test_issue_8640_split_tool_call_arguments_reconstructed() {
        let name_chunk = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("tc1".to_string()),
            r#type: Some(dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: None,
            }),
        };
        let args_chunk_1 = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: None,
                arguments: Some("{\"city\":".to_string()),
            }),
        };
        let args_chunk_2 = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: None,
                arguments: Some("\"Tokyo\"}".to_string()),
            }),
        };

        let deltas = vec![
            create_test_delta_with_tool_chunks(
                0,
                vec![name_chunk],
                None,
                Some(dynamo_protocols::types::Role::Assistant),
            ),
            create_test_delta_with_tool_chunks(0, vec![args_chunk_1], None, None),
            create_test_delta_with_tool_chunks(
                0,
                vec![args_chunk_2],
                Some(dynamo_protocols::types::FinishReason::ToolCalls),
                None,
            ),
        ];
        let stream = Box::pin(stream::iter(deltas));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;
        assert!(result.is_ok(), "aggregation should not error");
        let response = result.unwrap();
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        let tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls should be Some after aggregation");
        assert_eq!(
            tool_calls.len(),
            1,
            "must produce exactly one aggregated tool_call, got {}",
            tool_calls.len()
        );
        let tc = &tool_calls[0];
        assert_eq!(tc.id, "tc1");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(
            tc.function.arguments, "{\"city\":\"Tokyo\"}",
            "#8640: arguments must be reconstructed from split fragments, \
             not dropped (got {:?})",
            tc.function.arguments
        );
    }

    /// Two parallel tool calls (index=0 and index=1), their chunks interleaved
    /// in emission order. Exercises that the per-index accumulator correctly
    /// keeps the two calls separate — not just that split args get merged
    /// within one call (which [`test_issue_8640_split_tool_call_arguments_reconstructed`]
    /// already covers). Related: [#8636](https://github.com/ai-dynamo/dynamo/issues/8636)
    /// is about the streaming path dropping the second call; the non-stream
    /// aggregator now handles the parallel case too, and this test pins it.
    #[tokio::test]
    async fn test_parallel_tool_calls_interleaved_chunks_aggregate_independently() {
        let make_name = |idx: u32, id: &str, name: &str| {
            dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
                index: idx,
                id: Some(id.to_string()),
                r#type: Some(dynamo_protocols::types::FunctionType::Function),
                function: Some(dynamo_protocols::types::FunctionCallStream {
                    name: Some(name.to_string()),
                    arguments: None,
                }),
            }
        };
        let make_args = |idx: u32, fragment: &str| {
            dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
                index: idx,
                id: None,
                r#type: None,
                function: Some(dynamo_protocols::types::FunctionCallStream {
                    name: None,
                    arguments: Some(fragment.to_string()),
                }),
            }
        };

        // Emission order mimics a hermes-style parser:
        //   open call 0 → open call 1 → args-frag-0a → args-frag-1a →
        //   args-frag-0b → args-frag-1b → finish
        let deltas = vec![
            create_test_delta_with_tool_chunks(
                0,
                vec![make_name(0, "tc0", "get_weather")],
                None,
                Some(dynamo_protocols::types::Role::Assistant),
            ),
            create_test_delta_with_tool_chunks(
                0,
                vec![make_name(1, "tc1", "get_time")],
                None,
                None,
            ),
            create_test_delta_with_tool_chunks(0, vec![make_args(0, "{\"city\":")], None, None),
            create_test_delta_with_tool_chunks(0, vec![make_args(1, "{\"tz\":")], None, None),
            create_test_delta_with_tool_chunks(0, vec![make_args(0, "\"Tokyo\"}")], None, None),
            create_test_delta_with_tool_chunks(
                0,
                vec![make_args(1, "\"JST\"}")],
                Some(dynamo_protocols::types::FinishReason::ToolCalls),
                None,
            ),
        ];
        let stream = Box::pin(stream::iter(deltas));

        let response = DeltaAggregator::apply(stream, ParsingOptions::default())
            .await
            .expect("aggregation should succeed");
        assert_eq!(response.inner.choices.len(), 1);
        let tool_calls = response.inner.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls should be Some");
        assert_eq!(tool_calls.len(), 2, "must produce both parallel tool calls");

        // BTreeMap iteration is index-ordered, so [0] is tc0, [1] is tc1.
        assert_eq!(tool_calls[0].id, "tc0");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Tokyo\"}");
        assert_eq!(tool_calls[1].id, "tc1");
        assert_eq!(tool_calls[1].function.name, "get_time");
        assert_eq!(tool_calls[1].function.arguments, "{\"tz\":\"JST\"}");
    }

    /// When fragment-only chunks arrive but no id/name ever establishes the
    /// call opener (producer bug), `finalize_merged_tool_chunk` drops the
    /// chunk with a warn! log instead of emitting a malformed tool call.
    /// This test just pins the "no panic, no phantom tool call" half — the
    /// warn! is observable via tracing subscriber in prod, not asserted here.
    #[tokio::test]
    async fn test_fragment_only_chunks_without_opener_drop_cleanly() {
        let args_only = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: None,
                arguments: Some("{\"orphaned\":true}".to_string()),
            }),
        };
        let deltas = vec![create_test_delta_with_tool_chunks(
            0,
            vec![args_only],
            Some(dynamo_protocols::types::FinishReason::Stop),
            Some(dynamo_protocols::types::Role::Assistant),
        )];
        let stream = Box::pin(stream::iter(deltas));

        let response = DeltaAggregator::apply(stream, ParsingOptions::default())
            .await
            .expect("aggregation should succeed even with dropped chunk");
        // Finalization only assigns `tool_calls` when the finalized vec is
        // non-empty, so the strict post-condition here is `None`. Tight
        // assertion catches a regression that flips to `Some(vec![])`.
        assert!(
            response.inner.choices[0].message.tool_calls.is_none(),
            "orphaned fragment must not produce a tool call (got {:?})",
            response.inner.choices[0].message.tool_calls,
        );
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
        assert_eq!(response.inner.id, "");
        assert_eq!(response.inner.model, "");
        assert_eq!(response.inner.created, 0);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 0);
        assert!(response.inner.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta = create_test_delta(
            0,
            "Hello,",
            Some(dynamo_protocols::types::Role::User),
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
        assert_eq!(response.inner.id, "test_id");
        assert_eq!(response.inner.model, "meta/llama-3.1-8b-instruct");
        assert_eq!(response.inner.created, 1234567890);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(
            choice.message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Hello,".to_string())
        );
        assert!(choice.finish_reason.is_none());
        assert_eq!(choice.message.role, dynamo_protocols::types::Role::User);
        assert!(response.inner.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 = create_test_delta(
            0,
            "Hello,",
            Some(dynamo_protocols::types::Role::User),
            None,
            Some(-0.1),
            None,
        );
        let annotated_delta2 = create_test_delta(
            0,
            " world!",
            None,
            Some(dynamo_protocols::types::FinishReason::Stop),
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
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(
            choice.message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Hello, world!".to_string())
        );
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::Stop)
        );
        assert_eq!(choice.message.role, dynamo_protocols::types::Role::User);
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
            Some(dynamo_protocols::types::Role::User),
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
            Some(dynamo_protocols::types::FinishReason::Stop),
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
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        assert_eq!(choice.index, 0);
        assert_eq!(
            choice.message.content.as_ref(),
            Some(&ChatCompletionMessageContent::Text(
                "Hello world".to_string()
            ))
        );
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::Stop)
        );
        assert_eq!(choice.message.role, dynamo_protocols::types::Role::User);
    }

    #[allow(deprecated)]
    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        // ALLOW: function_call is deprecated
        let data = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test_id".to_string(),
                model: "test_model".to_string(),
                created: 1234567890,
                service_tier: None,
                usage: None,
                system_fingerprint: None,
                choices: vec![
                    dynamo_protocols::types::ChatChoiceStream {
                        index: 0,
                        delta: dynamo_protocols::types::ChatCompletionStreamResponseDelta {
                            role: Some(dynamo_protocols::types::Role::Assistant),
                            content: Some(ChatCompletionMessageContent::Text(
                                "Choice 0".to_string(),
                            )),
                            function_call: None,
                            tool_calls: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
                        stop_reason: None,
                        logprobs: None,
                    },
                    dynamo_protocols::types::ChatChoiceStream {
                        index: 1,
                        delta: dynamo_protocols::types::ChatCompletionStreamResponseDelta {
                            role: Some(dynamo_protocols::types::Role::Assistant),
                            content: Some(ChatCompletionMessageContent::Text(
                                "Choice 1".to_string(),
                            )),
                            function_call: None,
                            tool_calls: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
                        stop_reason: None,
                        logprobs: None,
                    },
                ],
                object: "chat.completion".to_string(),
            },
            nvext: None,
        };

        // Wrap it in Annotated and create a stream
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let mut response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.choices.len(), 2);
        response.inner.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.inner.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(
            choice0.message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Choice 0".to_string())
        );
        assert_eq!(
            choice0.finish_reason,
            Some(dynamo_protocols::types::FinishReason::Stop)
        );
        assert_eq!(
            choice0.message.role,
            dynamo_protocols::types::Role::Assistant
        );

        let choice1 = &response.inner.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(
            choice1.message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Choice 1".to_string())
        );
        assert_eq!(
            choice1.finish_reason,
            Some(dynamo_protocols::types::FinishReason::Stop)
        );
        assert_eq!(
            choice1.message.role,
            dynamo_protocols::types::Role::Assistant
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
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Stop), // Original finish reason is Stop
            None,
            Some(tool_call_json),
        );

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].r#type,
            dynamo_protocols::types::FunctionType::Function
        );

        // Most importantly, verify that finish reason was overridden to ToolCalls despite original being Stop
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_tool_calling_finish_reason_override_from_length() {
        // Test that when tool calls are present but finish reason is Length, it gets overridden to ToolCalls
        let tool_call_json = r#"{"name": "search", "arguments": {"query": "rust programming"}}"#;

        let annotated_delta = create_test_delta(
            0,
            "Let me search for that.",
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Length), // Original finish reason is Length
            None,
            Some(tool_call_json),
        );

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].r#type,
            dynamo_protocols::types::FunctionType::Function
        );

        // Verify that finish reason was overridden to ToolCalls despite original being Length
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_tool_calling_finish_reason_override_from_none() {
        // Test that when tool calls are present but finish reason is None, it gets set to ToolCalls
        let tool_call_json = r#"{"name": "calculate", "arguments": {"expression": "2+2"}}"#;

        let annotated_delta = create_test_delta(
            0,
            "I'll calculate that for you.",
            Some(dynamo_protocols::types::Role::Assistant),
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
            error: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);

        // Verify that finish reason was set to ToolCalls despite original being None
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_no_tool_calling_preserves_original_finish_reason() {
        // Test that when no tool calls are present, the original finish reason is preserved
        let annotated_delta = create_test_delta(
            0,
            "This is a regular response without tool calls.",
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Stop),
            None,
            None, // No tool calls
        );

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        // Verify no tool calls are present
        assert!(choice.message.tool_calls.is_none());

        // Verify that original finish reason (Stop) is preserved
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn test_empty_tool_calls_preserves_original_finish_reason() {
        // Test that when tool calls array is empty, the original finish reason is preserved
        // Create a delta with empty tool calls by modifying the create_test_delta output
        let mut annotated_delta = create_test_delta(
            0,
            "Response with empty tool calls array.",
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Length),
            None,
            None,
        );

        // Manually set empty tool calls array
        if let Some(ref mut data) = annotated_delta.data {
            data.inner.choices[0].delta.tool_calls = Some(vec![]); // Empty tool calls array
        }

        let data = annotated_delta.data.unwrap();
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        // Verify tool calls array is empty
        assert!(choice.message.tool_calls.is_none());

        // Verify that original finish reason (Length) is preserved since tool calls are empty
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::Length)
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
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::ToolCalls),
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
            error: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // There should be one choice
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

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
            &ChatCompletionMessageContent::Text(
                "Hey Dude ! What's the weather in San Francisco in Fahrenheit?".to_string()
            )
        );

        // The finish_reason should be ToolCalls
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        );
        assert_eq!(
            choice.message.role,
            dynamo_protocols::types::Role::Assistant
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
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Stop), // This should be overridden
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
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];

        // The finish_reason should be ToolCalls, not Stop, because tool calls are present
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        );

        // Verify tool calls are present
        assert!(choice.message.tool_calls.is_some());
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[tokio::test]
    async fn test_parses_aggregated_tool_call_text_into_tool_calls() {
        let annotated_delta = create_test_delta(
            0,
            "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"SF\"}}\n</tool_call>",
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Stop),
            None,
            None,
        );

        let stream = Box::pin(stream::iter(vec![annotated_delta]));
        let result = DeltaAggregator::apply(
            stream,
            ParsingOptions::new(Some("hermes".to_string()), None),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        let choice = &response.inner.choices[0];
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        );
        assert_eq!(choice.message.content, None);
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].r#type,
            dynamo_protocols::types::FunctionType::Function
        );
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"location\":\"SF\"}");
    }

    #[tokio::test]
    async fn test_preserves_non_tool_content_when_parsing_aggregated_tool_calls() {
        let annotated_delta = create_test_delta(
            0,
            "hello\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"SF\"}}\n</tool_call>",
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Stop),
            None,
            None,
        );

        let stream = Box::pin(stream::iter(vec![annotated_delta]));
        let result = DeltaAggregator::apply(
            stream,
            ParsingOptions::new(Some("hermes".to_string()), None),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        let choice = &response.inner.choices[0];
        assert_eq!(
            choice.message.content,
            Some(ChatCompletionMessageContent::Text("hello".to_string()))
        );
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::ToolCalls)
        );
        assert_eq!(
            choice.message.tool_calls.as_ref().unwrap()[0].r#type,
            dynamo_protocols::types::FunctionType::Function
        );
    }

    #[test]
    fn test_reasoning_only_response_serializes_content_key_as_null() {
        // DGH-651: when a response carries reasoning_content but no text or
        // content parts, the `content` key must still be present in the
        // serialized JSON (as `null`) so clients can rely on it alongside
        // `reasoning_content`. Fixed by removing skip_serializing_if from
        // ChatCompletionResponseMessage.content.
        let delta = DeltaChoice {
            index: 0,
            text: String::new(),
            role: Some(dynamo_protocols::types::Role::Assistant),
            finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
            stop_reason: None,
            logprobs: None,
            tool_call_chunks: BTreeMap::new(),
            tool_calls: None,
            reasoning_content: Some("Analyzing the question.".to_string()),
            content_parts: vec![],
        };

        let choice: dynamo_protocols::types::ChatChoice = delta.into();

        assert!(choice.message.content.is_none());
        assert_eq!(
            choice.message.reasoning_content.as_deref(),
            Some("Analyzing the question.")
        );

        let json = serde_json::to_value(&choice.message).unwrap();
        assert_eq!(
            json.get("content"),
            Some(&serde_json::Value::Null),
            "content key must be serialized as null when absent"
        );
        assert!(json.get("reasoning_content").is_some());
    }
}

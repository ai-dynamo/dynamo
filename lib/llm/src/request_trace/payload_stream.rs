// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::oneshot;

use crate::protocols::openai::ParsingOptions;
use crate::protocols::openai::chat_completions::{
    DeltaAggregator, NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::protocols::annotated::Annotated;

use dynamo_protocols::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta};
use futures::StreamExt;

type PayloadStream =
    Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>;

/// What the aggregation of a response stream produced, for request payload capture.
///
/// `response` is the aggregated record when aggregation succeeded, or the partial
/// record recovered from the chunks that arrived before a failure. `drop_reason` is
/// set whenever the record is not a faithful copy of what the client received, and
/// is published verbatim as the record's `payload_drop_reason`. The two travel
/// together so the emit site can derive `payload_complete` from the data rather
/// than assuming it: a partial response is `Some` response *and* `Some` reason.
pub struct PayloadOutcome {
    pub response: Option<NvCreateChatCompletionResponse>,
    pub drop_reason: Option<String>,
}

impl PayloadOutcome {
    fn complete(response: NvCreateChatCompletionResponse) -> Self {
        Self {
            response: Some(response),
            drop_reason: None,
        }
    }

    fn dropped(
        response: Option<NvCreateChatCompletionResponse>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            response,
            drop_reason: Some(reason.into()),
        }
    }
}

/// The response stream ended without ever yielding a chunk.
const DROP_EMPTY_RESPONSE_STREAM: &str = "empty_response_stream";

/// The aggregation never reported an outcome at all: the SSE consumer dropped the
/// pass-through stream before end-of-stream, or the aggregation task was cancelled.
const DROP_CLIENT_CANCELLED: &str = "client_cancelled";

/// Build the `aggregation_failed` drop reason. The colon-delimited
/// `identifier:detail` shape matches the marker reasons `otel_sink.rs` already
/// publishes, so a consumer can parse one grammar across both producers.
fn aggregation_failed_reason(error: impl std::fmt::Display) -> String {
    format!("aggregation_failed:{error}")
}

/// Resolves to the aggregation outcome. On success it carries the final response and
/// no drop reason; otherwise it carries a drop reason naming why the record is
/// incomplete, plus whatever partial response was recovered. The caller emits the
/// single combined request payload record once, either way.
type PayloadFuture = Pin<Box<dyn std::future::Future<Output = PayloadOutcome> + Send>>;

/// Forwards transformed chunks unchanged; collects them for aggregation.
pub struct PassThroughWithAgg<S> {
    inner: S,
    chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    done_tx: Option<oneshot::Sender<PayloadOutcome>>,
}

impl<S> PassThroughWithAgg<S> {
    fn new(inner: S, tx: oneshot::Sender<PayloadOutcome>) -> Self {
        Self {
            inner,
            chunks: Vec::new(),
            done_tx: Some(tx),
        }
    }
}

impl<S> Stream for PassThroughWithAgg<S>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Unpin,
{
    type Item = Annotated<NvCreateChatCompletionStreamResponse>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(chunk)) => {
                // Store chunk for aggregation
                self.chunks.push(chunk.clone());
                // Forward the chunk unchanged downstream
                Poll::Ready(Some(chunk))
            }
            Poll::Ready(None) => {
                if let Some(tx) = self.done_tx.take() {
                    // Aggregate all collected chunks
                    let chunks = std::mem::take(&mut self.chunks);
                    if chunks.is_empty() {
                        tracing::debug!(
                            "request payload: empty response stream, no response to aggregate"
                        );
                        let _ = tx.send(PayloadOutcome::dropped(None, DROP_EMPTY_RESPONSE_STREAM));
                        return Poll::Ready(None);
                    }
                    let parsing_options = ParsingOptions::default();

                    tokio::spawn(async move {
                        let _ =
                            tx.send(aggregate_with_partial_recovery(chunks, parsing_options).await);
                    });
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Aggregate the buffered chunks, keeping whatever content arrived before an
/// error-tagged chunk.
///
/// `Annotated::into_data` returns `Err` for exactly the chunks `is_error()` matches,
/// and `DeltaAggregator::apply` drives the buffer through a `try_fold` that
/// short-circuits on the first of them and discards the accumulator. So a buffer
/// containing an error-tagged chunk can only aggregate to `Err`, and aggregating the
/// prefix before that chunk recovers the content the client already received. The
/// aggregator itself is deliberately left alone: its short-circuit is load-bearing on
/// the client-facing non-streaming path, where a typed backend error must surface as
/// an error rather than as a truncated success.
async fn aggregate_with_partial_recovery(
    mut chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    parsing_options: ParsingOptions,
) -> PayloadOutcome {
    let Some(error_at) = chunks.iter().position(|chunk| chunk.is_error()) else {
        return match DeltaAggregator::apply(futures::stream::iter(chunks), parsing_options).await {
            Ok(final_resp) => PayloadOutcome::complete(final_resp),
            Err(e) => {
                tracing::warn!("request payload: aggregation failed: {e}");
                PayloadOutcome::dropped(None, aggregation_failed_reason(e))
            }
        };
    };

    let error = match chunks[error_at].clone().into_data() {
        Err(error) => error.to_string(),
        // Unreachable: `is_error()` is the same predicate `into_data` errors on.
        // Kept so a future divergence still yields a well-formed reason.
        Ok(_) => "unknown error".to_string(),
    };
    tracing::warn!("request payload: aggregation failed: {error}");
    let reason = aggregation_failed_reason(&error);

    chunks.truncate(error_at);
    if chunks.is_empty() {
        return PayloadOutcome::dropped(None, reason);
    }
    let partial = DeltaAggregator::apply(futures::stream::iter(chunks), parsing_options)
        .await
        .ok();
    PayloadOutcome::dropped(partial, reason)
}

/// Return (pass-through stream, future -> aggregation outcome for request payload capture).
pub fn scan_aggregate_with_future<S>(stream: S) -> (PayloadStream, PayloadFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Unpin + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<PayloadOutcome>();
    let passthrough = PassThroughWithAgg::new(stream, tx);
    (
        Box::pin(passthrough),
        Box::pin(async move {
            match rx.await {
                Ok(outcome) => outcome,
                Err(_) => {
                    // tx dropped without sending: the SSE consumer dropped the
                    // passthrough stream before end-of-stream (client cancel), or the
                    // spawned aggregation task was cancelled before it reported.
                    // Aggregation failures report themselves and no longer land here.
                    tracing::debug!(
                        "request payload: response aggregation produced no outcome (client cancel)"
                    );
                    PayloadOutcome::dropped(None, DROP_CLIENT_CANCELLED)
                }
            }
        }),
    )
}

/// Collect all chunks, aggregate them, then emit a single final chunk (for non-streaming)
pub fn fold_aggregate_with_future<S>(stream: S) -> (PayloadStream, PayloadFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<PayloadOutcome>();

    let single_chunk_stream = async move {
        let chunks: Vec<_> = stream.collect().await;
        let chunks_stream = futures::stream::iter(chunks);
        let parsing_options = ParsingOptions::default();

        match DeltaAggregator::apply(chunks_stream, parsing_options).await {
            Ok(final_resp) => {
                let _ = tx.send(PayloadOutcome::complete(final_resp.clone()));
                final_response_to_one_chunk_stream(final_resp)
            }
            Err(e) => {
                tracing::warn!("fold aggregation failed: {e}");
                // Report the failure instead of dropping tx silently, so the record
                // carries why it has no response. The client still receives a
                // (best-effort) empty fallback chunk so the HTTP response shape stays
                // valid; the combined request payload record is emitted with
                // `response = None` and this drop reason.
                let _ = tx.send(PayloadOutcome::dropped(None, aggregation_failed_reason(&e)));
                let fallback = NvCreateChatCompletionResponse {
                    inner: dynamo_protocols::types::CreateChatCompletionResponse {
                        id: String::new(),
                        created: 0,
                        usage: None,
                        model: String::new(),
                        object: "chat.completion".to_string(),
                        system_fingerprint: None,
                        choices: vec![],
                        service_tier: None,
                    },
                    nvext: None,
                };
                final_response_to_one_chunk_stream(fallback)
            }
        }
    };

    let future = Box::pin(async move {
        match rx.await {
            Ok(outcome) => outcome,
            Err(_) => {
                tracing::debug!(
                    "request payload: fold response aggregation produced no outcome (client cancel)"
                );
                PayloadOutcome::dropped(None, DROP_CLIENT_CANCELLED)
            }
        }
    });

    (
        Box::pin(futures::stream::once(single_chunk_stream).flatten()),
        future,
    )
}

/// Convert a final (non-streaming) response into a single "final chunk" stream.
/// Put the entire final text/tool-calls into `delta` so downstream aggregate is a no-op.
pub fn final_response_to_one_chunk_stream(
    resp: NvCreateChatCompletionResponse,
) -> std::pin::Pin<
    Box<dyn futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>,
> {
    let mut choices: Vec<ChatChoiceStream> = Vec::with_capacity(resp.inner.choices.len());
    for (idx, ch) in resp.inner.choices.iter().enumerate() {
        // Convert FunctionCall to FunctionCallStream if present
        #[allow(deprecated)]
        let function_call = ch.message.function_call.as_ref().map(|fc| {
            dynamo_protocols::types::ChatCompletionStreamResponseDeltaFunctionCall {
                name: Some(fc.name.clone()),
                arguments: Some(fc.arguments.clone()),
            }
        });

        // Convert tool calls
        let tool_calls = ch.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(
                    |(i, call)| dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
                        index: i as u32,
                        id: Some(call.id.clone()),
                        r#type: Some(dynamo_protocols::types::FunctionType::Function),
                        function: Some(dynamo_protocols::types::FunctionCallStream {
                            name: Some(call.function.name.clone()),
                            arguments: Some(call.function.arguments.clone()),
                        }),
                    },
                )
                .collect()
        });

        #[allow(deprecated)]
        let delta = ChatCompletionStreamResponseDelta {
            role: Some(ch.message.role),
            content: ch.message.content.clone(),
            tool_calls,
            function_call,
            refusal: ch.message.refusal.clone(),
            reasoning_content: ch.message.reasoning_content.clone(),
        };

        let choice = ChatChoiceStream {
            index: idx as u32,
            delta,
            finish_reason: ch.finish_reason,
            logprobs: ch.logprobs.clone(),
        };
        choices.push(choice);
    }

    let chunk = NvCreateChatCompletionStreamResponse {
        inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
            id: resp.inner.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: resp.inner.created,
            model: resp.inner.model.clone(),
            system_fingerprint: resp.inner.system_fingerprint.clone(),
            service_tier: resp.inner.service_tier.clone(),
            choices,
            usage: resp.inner.usage.clone(),
        },
        nvext: resp.nvext.clone(),
        llm_metrics: None,
    };

    let annotated = Annotated {
        data: Some(chunk),
        id: None,
        event: None,
        comment: None,
        error: None,
    };
    Box::pin(futures::stream::once(async move { annotated }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
        FinishReason, FunctionCallStream, FunctionType, Role,
    };
    use futures::StreamExt;
    use futures::stream;

    /// Helper function to create a mock chat response chunk
    fn create_mock_chunk(
        content: String,
        index: u32,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(ChatCompletionMessageContent::Text(content)),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        };

        let response = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    /// Helper function to create a final response chunk with finish reason
    fn create_final_chunk(index: u32) -> Annotated<NvCreateChatCompletionStreamResponse> {
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
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn create_reasoning_chunk(
        reasoning_content: String,
        index: u32,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: None,
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: Some(reasoning_content),
            },
            finish_reason: None,
            logprobs: None,
        };

        let response = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn create_tool_call_chunk(
        tool_chunk: dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
        finish_reason: Option<FinishReason>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: None,
                content: None,
                tool_calls: Some(vec![tool_chunk]),
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason,
            logprobs: None,
        };

        let response = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    /// Helper to extract content from a chunk
    fn extract_content(chunk: &Annotated<NvCreateChatCompletionStreamResponse>) -> String {
        chunk
            .data
            .as_ref()
            .and_then(|d| d.inner.choices.first())
            .and_then(|c| c.delta.content.as_ref())
            .and_then(|content| match content {
                ChatCompletionMessageContent::Text(text) => Some(text.clone()),
                ChatCompletionMessageContent::Parts(_) => None,
            })
            .unwrap_or_default()
    }

    /// Helper to reconstruct all content from results
    fn reconstruct_content(results: &[Annotated<NvCreateChatCompletionStreamResponse>]) -> String {
        results
            .iter()
            .map(extract_content)
            .collect::<Vec<_>>()
            .join("")
    }

    #[tokio::test]
    async fn test_passthrough_forwards_chunks_unchanged() {
        // Input chunks should pass through exactly as-is
        let chunks = vec![
            create_mock_chunk("Hello ".to_string(), 0),
            create_mock_chunk("World".to_string(), 0),
            create_final_chunk(0),
        ];

        let input_stream = stream::iter(chunks.clone());
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;
        // Negative control for the drop-reason work: a clean stream must still
        // aggregate to a complete record with no reason attached.
        assert!(
            outcome.drop_reason.is_none(),
            "a fully successful aggregation must not carry a drop reason"
        );
        let final_resp = outcome
            .response
            .expect("aggregation should produce a record");

        // Verify chunk count
        assert_eq!(results.len(), 3, "Should pass through all chunks unchanged");

        // Verify content is identical
        assert_eq!(extract_content(&results[0]), "Hello ");
        assert_eq!(extract_content(&results[1]), "World");
        assert_eq!(extract_content(&results[2]), ""); // Final chunk has no content

        // Verify complete content reconstruction
        assert_eq!(reconstruct_content(&results), "Hello World");
        assert_eq!(
            final_resp.inner.choices[0]
                .message
                .content
                .as_ref()
                .unwrap(),
            &ChatCompletionMessageContent::Text("Hello World".to_string())
        );
    }

    #[tokio::test]
    async fn test_passthrough_aggregates_reasoning_content_and_tool_calls() {
        let name_chunk = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_weather".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: None,
            }),
        };
        let args_chunk = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(FunctionCallStream {
                name: None,
                arguments: Some("{\"city\":\"Tokyo\"}".to_string()),
            }),
        };
        let chunks = vec![
            create_reasoning_chunk("I should inspect the weather. ".to_string(), 0),
            create_mock_chunk("The weather is clear.".to_string(), 0),
            create_tool_call_chunk(name_chunk, None),
            create_tool_call_chunk(args_chunk, Some(FinishReason::ToolCalls)),
        ];

        let input_stream = stream::iter(chunks.clone());
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;
        assert!(
            outcome.drop_reason.is_none(),
            "a fully successful aggregation must not carry a drop reason"
        );
        let final_resp = outcome
            .response
            .expect("aggregation should produce a record");

        assert_eq!(results.len(), chunks.len());
        assert_eq!(
            final_resp.inner.choices[0]
                .message
                .reasoning_content
                .as_deref(),
            Some("I should inspect the weather. ")
        );
        assert_eq!(
            final_resp.inner.choices[0]
                .message
                .content
                .as_ref()
                .unwrap(),
            &ChatCompletionMessageContent::Text("The weather is clear.".to_string())
        );
        let tool_call = &final_resp.inner.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls should aggregate")[0];
        assert_eq!(tool_call.id, "call_weather");
        assert_eq!(tool_call.function.name, "get_weather");
        assert_eq!(tool_call.function.arguments, "{\"city\":\"Tokyo\"}");
    }

    #[tokio::test]
    async fn test_final_response_to_one_chunk_preserves_reasoning_and_tool_calls() {
        let response: NvCreateChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The weather is clear.",
                    "reasoning_content": "I should inspect the weather.",
                    "tool_calls": [{
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .expect("response parses");

        let chunks: Vec<_> = final_response_to_one_chunk_stream(response).collect().await;
        assert_eq!(chunks.len(), 1);
        let delta = &chunks[0].data.as_ref().unwrap().inner.choices[0].delta;

        assert_eq!(
            delta.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("The weather is clear.".to_string())
        );
        assert_eq!(
            delta.reasoning_content.as_deref(),
            Some("I should inspect the weather.")
        );
        let tool_call = &delta.tool_calls.as_ref().expect("tool calls preserved")[0];
        assert_eq!(tool_call.id.as_deref(), Some("call_weather"));
        assert_eq!(tool_call.r#type, Some(FunctionType::Function));
        let function = tool_call.function.as_ref().expect("function preserved");
        assert_eq!(function.name.as_deref(), Some("get_weather"));
        assert_eq!(function.arguments.as_deref(), Some("{\"city\":\"Tokyo\"}"));
    }

    #[tokio::test]
    async fn test_empty_stream_handling() {
        // Empty stream: the aggregator has nothing to apply, so the outcome carries
        // no response. It names itself rather than looking like a client cancel, so
        // the caller (preprocessor) emits the combined request payload record with
        // `response = None` and this specific drop reason.
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;

        assert_eq!(results.len(), 0, "Empty stream should produce no chunks");
        assert!(
            outcome.response.is_none(),
            "Empty stream should resolve request payload future to no response, not a fallback record"
        );
        assert_eq!(
            outcome.drop_reason.as_deref(),
            Some("empty_response_stream"),
            "an empty stream must be distinguishable from a client cancel"
        );
    }

    #[tokio::test]
    async fn test_single_chunk_stream() {
        // Single chunk should pass through and aggregate correctly
        let chunks = vec![create_mock_chunk("Single chunk".to_string(), 0)];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;
        assert!(
            outcome.drop_reason.is_none(),
            "a fully successful aggregation must not carry a drop reason"
        );
        let final_resp = outcome
            .response
            .expect("aggregation should produce a record");

        // Verify passthrough
        assert_eq!(results.len(), 1);
        assert_eq!(extract_content(&results[0]), "Single chunk");

        // Verify aggregation
        assert_eq!(final_resp.inner.object, "chat.completion");
    }

    #[tokio::test]
    async fn test_chunks_with_metadata_preserved() {
        // Test that metadata (id, event, comment) is preserved through passthrough
        let chunk_with_metadata = Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                    id: "test-id".to_string(),
                    choices: vec![{
                        #[allow(deprecated)]
                        ChatChoiceStream {
                            index: 0,
                            delta: ChatCompletionStreamResponseDelta {
                                role: Some(Role::Assistant),
                                content: Some(ChatCompletionMessageContent::Text(
                                    "Content".to_string(),
                                )),
                                tool_calls: None,
                                function_call: None,
                                refusal: None,
                                reasoning_content: None,
                            },
                            finish_reason: None,
                            logprobs: None,
                        }
                    }],
                    created: 1234567890,
                    model: "test-model".to_string(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                    service_tier: None,
                },
                nvext: None,
                llm_metrics: None,
            }),
            id: Some("correlation-123".to_string()),
            event: Some("test-event".to_string()),
            comment: Some(vec!["test-comment".to_string()]),
            error: None,
        };

        let input_stream = stream::iter(vec![chunk_with_metadata.clone()]);
        let (passthrough, _future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;

        // Verify metadata is preserved
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, Some("correlation-123".to_string()));
        assert_eq!(results[0].event, Some("test-event".to_string()));
        assert_eq!(results[0].comment, Some(vec!["test-comment".to_string()]));
    }

    #[tokio::test]
    async fn test_concurrent_futures() {
        // Test that multiple concurrent payload streams don't interfere. The
        // passthrough streams are dropped immediately (the `_` destructure), which
        // models a client cancel before the first poll — each future should
        // independently resolve to a cancelled outcome without crosstalk.
        let chunks1 = vec![create_mock_chunk("Stream 1".to_string(), 0)];
        let chunks2 = vec![create_mock_chunk("Stream 2".to_string(), 0)];

        let (_, future1) = scan_aggregate_with_future(stream::iter(chunks1));
        let (_, future2) = scan_aggregate_with_future(stream::iter(chunks2));

        let (outcome1, outcome2) = tokio::join!(future1, future2);

        assert!(outcome1.response.is_none());
        assert!(outcome2.response.is_none());
        assert_eq!(outcome1.drop_reason.as_deref(), Some("client_cancelled"));
        assert_eq!(outcome2.drop_reason.as_deref(), Some("client_cancelled"));
    }

    #[tokio::test]
    async fn error_chunk_mid_stream_keeps_partial_content_and_names_the_error() {
        // The regression this change exists for: the client saw "Hello " before the
        // backend errored, so the audit record must keep that text and say why it is
        // short, instead of silently claiming a complete empty record.
        let chunks = vec![
            create_mock_chunk("Hello ".to_string(), 0),
            Annotated::<NvCreateChatCompletionStreamResponse>::from_error(
                "invalid sampling parameter",
            ),
            create_mock_chunk("never delivered".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks.clone());
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;

        assert_eq!(
            results.len(),
            chunks.len(),
            "the error chunk must still reach the client unchanged"
        );

        let reason = outcome
            .drop_reason
            .as_deref()
            .expect("an errored stream must carry a drop reason");
        assert!(
            reason.starts_with("aggregation_failed:"),
            "reason should use the colon-delimited grammar, got {reason}"
        );
        assert!(
            reason.contains("invalid sampling parameter"),
            "reason should name the underlying error, got {reason}"
        );

        let partial = outcome
            .response
            .expect("content delivered before the error must be preserved");
        assert_eq!(
            partial.inner.choices[0].message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Hello ".to_string()),
            "only the pre-error prefix should be aggregated"
        );
    }

    #[tokio::test]
    async fn error_as_first_chunk_reports_reason_without_a_response() {
        // Nothing was delivered before the error, so there is no partial content to
        // keep — but the record must still say why it is empty.
        let chunks = vec![
            Annotated::<NvCreateChatCompletionStreamResponse>::from_error("backend unavailable"),
        ];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let _results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;

        assert!(
            outcome.response.is_none(),
            "no content arrived before the error, so there is nothing to preserve"
        );
        let reason = outcome
            .drop_reason
            .as_deref()
            .expect("an errored stream must carry a drop reason");
        assert!(
            reason.starts_with("aggregation_failed:"),
            "reason should use the colon-delimited grammar, got {reason}"
        );
        assert!(
            reason.contains("backend unavailable"),
            "reason should name the underlying error, got {reason}"
        );
    }
}

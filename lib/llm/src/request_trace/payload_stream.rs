// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::oneshot;

use crate::protocols::common::metrics::{
    ANNOTATION_LLM_METRICS, ANNOTATION_PAYLOAD_USAGE, LLMMetricAnnotation,
};
use crate::protocols::openai::ParsingOptions;
use crate::protocols::openai::chat_completions::{
    DeltaAggregator, NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::protocols::annotated::Annotated;

use dynamo_protocols::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta};
use futures::StreamExt;

type PayloadStream =
    Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>;

/// Resolves to `Some(final_response)` when aggregation succeeds, or `None` when the
/// client cancels mid-stream / the aggregator fails. The caller emits the single
/// combined request payload record once either way — with the response on `Some`, or
/// request-only (`response = None`) on `None`.
type PayloadFuture =
    Pin<Box<dyn std::future::Future<Output = Option<NvCreateChatCompletionResponse>> + Send>>;

/// Forwards transformed chunks unchanged; collects them for aggregation.
pub struct PassThroughWithAgg<S> {
    inner: S,
    chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    done_tx: Option<oneshot::Sender<NvCreateChatCompletionResponse>>,
}

impl<S> PassThroughWithAgg<S> {
    fn new(inner: S, tx: oneshot::Sender<NvCreateChatCompletionResponse>) -> Self {
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
                        drop(tx);
                        return Poll::Ready(None);
                    }
                    let chunks_stream = futures::stream::iter(chunks);
                    let parsing_options = ParsingOptions::default();

                    tokio::spawn(async move {
                        match DeltaAggregator::apply(chunks_stream, parsing_options).await {
                            Ok(final_resp) => {
                                let _ = tx.send(final_resp);
                            }
                            Err(e) => {
                                tracing::warn!("request payload: aggregation failed: {e}");
                            }
                        }
                    });
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Return (pass-through stream, future -> final aggregated response for request payload capture).
pub fn scan_aggregate_with_future<S>(stream: S) -> (PayloadStream, PayloadFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Unpin + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<NvCreateChatCompletionResponse>();
    let passthrough = PassThroughWithAgg::new(stream, tx);
    (
        Box::pin(passthrough),
        Box::pin(async move {
            match rx.await {
                Ok(resp) => Some(resp),
                Err(_) => {
                    // tx dropped without sending: either the SSE consumer dropped the
                    // passthrough stream before end-of-stream (client cancel) or the
                    // spawned `DeltaAggregator::apply` errored. Either way, the combined
                    // record is emitted with `response = None`.
                    tracing::debug!(
                        "request payload: response aggregation produced no record (client cancel or aggregation error)"
                    );
                    None
                }
            }
        }),
    )
}

/// Fold a non-streaming payload into one final client chunk while forwarding
/// metric annotations as they arrive.
///
/// Metrics must bypass the fold because downstream latency metrics depend on
/// observation time. Typed metrics are moved out of buffered chunks, while
/// event-tagged annotations are forwarded without their payload data.
///
/// At end of stream, the remaining client data is aggregated into one chunk.
/// The folded chunk carries no `llm_metrics` because they were already emitted
/// in-stream (#11349).
pub fn fold_aggregate_with_future<S>(stream: S) -> (PayloadStream, PayloadFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<NvCreateChatCompletionResponse>();

    let out = async_stream::stream! {
        let mut stream = std::pin::pin!(stream);
        let mut buffered: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = Vec::new();
        let mut forwarded_frames: usize = 0;
        let mut total_chunk_tokens: usize = 0;

        while let Some(mut chunk) = stream.next().await {
            // Move typed metrics out of the buffered chunk and forward them
            // immediately as an annotation frame.
            if let Some(metrics) = chunk.data.as_mut().and_then(|data| data.llm_metrics.take()) {
                total_chunk_tokens += metrics.chunk_tokens;
                forwarded_frames += 1;
                match metrics.to_annotation::<NvCreateChatCompletionStreamResponse>() {
                    Ok(frame) => yield frame,
                    Err(e) => {
                        tracing::warn!(
                            "request payload: failed to serialize metric annotation: {e}"
                        );
                    }
                }
            }

            // Forward the event annotation as a data-less shell, leaving any
            // payload data behind for aggregation.
            if matches!(
                chunk.event.as_deref(),
                Some(ANNOTATION_LLM_METRICS) | Some(ANNOTATION_PAYLOAD_USAGE)
            ) {
                if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(&chunk) {
                    total_chunk_tokens += metrics.chunk_tokens;
                }
                forwarded_frames += 1;
                yield Annotated {
                    data: None,
                    id: chunk.id.take(),
                    event: chunk.event.take(),
                    comment: chunk.comment.take(),
                    error: None,
                };
            }

            buffered.push(chunk);
        }

        tracing::debug!(
            forwarded_frames,
            total_chunk_tokens,
            "request payload: metric annotations forwarded ahead of the non-streaming fold"
        );

        let parsing_options = ParsingOptions::default();
        match DeltaAggregator::apply(futures::stream::iter(buffered), parsing_options).await {
            Ok(final_resp) => {
                let _ = tx.send(final_resp.clone());
                yield final_response_to_one_chunk(final_resp);
            }
            Err(e) => {
                tracing::warn!("fold aggregation failed: {e}");
                // Drop tx without sending so the request payload future resolves to None.
                // The client still receives a (best-effort) empty fallback chunk so
                // the HTTP response shape stays valid; the combined request payload record is
                // emitted with `response = None`.
                drop(tx);
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
                yield final_response_to_one_chunk(fallback);
            }
        }
    };

    let future = Box::pin(async move {
        match rx.await {
            Ok(resp) => Some(resp),
            Err(_) => {
                tracing::debug!(
                    "request payload: fold response aggregation produced no record (client cancel or aggregation error)"
                );
                None
            }
        }
    });

    (Box::pin(out), future)
}

/// Build the single client chunk for a folded non-streaming response.
///
/// The complete response is placed in one `delta`. `llm_metrics` remains
/// `None` because metrics were already forwarded before the fold; attaching
/// them here would double-observe them downstream (#11349).
fn final_response_to_one_chunk(
    resp: NvCreateChatCompletionResponse,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
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

    Annotated {
        data: Some(chunk),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
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
        let final_resp = future.await.expect("aggregation should produce a record");

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
        let final_resp = future.await.expect("aggregation should produce a record");

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

    #[test]
    fn test_final_response_to_one_chunk_preserves_reasoning_and_tool_calls() {
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

        let chunk = final_response_to_one_chunk(response);
        let delta = &chunk.data.as_ref().unwrap().inner.choices[0].delta;

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
        // Empty stream is treated the same as a client-cancel mid-stream: the
        // aggregator has nothing to apply, tx drops without sending, and the
        // future resolves to None. The caller (preprocessor) then emits the
        // combined request payload record with `response = None`.
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let final_resp = future.await;

        assert_eq!(results.len(), 0, "Empty stream should produce no chunks");
        assert!(
            final_resp.is_none(),
            "Empty stream should resolve request payload future to None, not a fallback record"
        );
    }

    #[tokio::test]
    async fn test_single_chunk_stream() {
        // Single chunk should pass through and aggregate correctly
        let chunks = vec![create_mock_chunk("Single chunk".to_string(), 0)];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let final_resp = future.await.expect("aggregation should produce a record");

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
        // independently resolve to None without crosstalk.
        let chunks1 = vec![create_mock_chunk("Stream 1".to_string(), 0)];
        let chunks2 = vec![create_mock_chunk("Stream 2".to_string(), 0)];

        let (_, future1) = scan_aggregate_with_future(stream::iter(chunks1));
        let (_, future2) = scan_aggregate_with_future(stream::iter(chunks2));

        let (resp1, resp2) = tokio::join!(future1, future2);

        assert!(resp1.is_none());
        assert!(resp2.is_none());
    }

    fn typed_metrics(chunk_tokens: usize, output_tokens: usize) -> LLMMetricAnnotation {
        LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens,
            chunk_tokens,
            cached_tokens: Some(4),
            ..Default::default()
        }
    }

    /// Production-shaped payload-usage tail: usage data plus its metric
    /// annotation, with `chunk_tokens = 0`.
    fn payload_usage_tail(output_tokens: usize) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let metrics = LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens,
            chunk_tokens: 0,
            cached_tokens: Some(4),
            ..Default::default()
        };
        let annotation = metrics.to_annotation::<()>().unwrap();

        let usage_chunk = NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens: 10,
                    completion_tokens: output_tokens as u32,
                    total_tokens: 10 + output_tokens as u32,
                    ..Default::default()
                }),
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
        };

        Annotated {
            data: Some(usage_chunk),
            id: None,
            event: annotation.event,
            comment: annotation.comment,
            error: None,
        }
    }

    #[tokio::test]
    async fn test_fold_forwards_metric_frames_before_folded_chunk() {
        let mut chunk1 = create_mock_chunk("Hello ".to_string(), 0);
        chunk1.data.as_mut().unwrap().llm_metrics = Some(typed_metrics(1, 1));
        let mut chunk2 = create_mock_chunk("World".to_string(), 0);
        chunk2.data.as_mut().unwrap().llm_metrics = Some(typed_metrics(2, 3));

        let chunks = vec![chunk1, chunk2, create_final_chunk(0), payload_usage_tail(3)];

        let (folded, future) = fold_aggregate_with_future(stream::iter(chunks));
        let results: Vec<_> = folded.collect().await;
        let final_resp = future.await.expect("aggregation should produce a record");

        // Two typed metric frames, one payload-usage shell, then one
        // folded client chunk.
        assert_eq!(results.len(), 4);

        for (frame, expected_chunk_tokens) in results[..2].iter().zip([1usize, 2]) {
            assert!(frame.data.is_none(), "metric frames must be data-less");
            let metrics = LLMMetricAnnotation::from_annotation(frame)
                .unwrap()
                .expect("metric frame must parse");
            assert_eq!(metrics.chunk_tokens, expected_chunk_tokens);
        }

        // The annotation is forwarded without payload data; usage remains
        // in the fold.
        let shell = &results[2];
        assert!(shell.data.is_none());
        let tail_metrics = LLMMetricAnnotation::from_annotation(shell)
            .unwrap()
            .expect("payload-usage shell must parse");
        assert_eq!(tail_metrics.chunk_tokens, 0);
        assert_eq!(tail_metrics.output_tokens, 3);

        // Exactly one folded client chunk; metrics must not be replayed on it.
        let folded_chunk = results[3].data.as_ref().expect("folded client chunk");
        assert!(folded_chunk.llm_metrics.is_none());
        assert_eq!(
            folded_chunk.inner.choices[0]
                .delta
                .content
                .as_ref()
                .unwrap(),
            &ChatCompletionMessageContent::Text("Hello World".to_string())
        );
        assert_eq!(
            folded_chunk.inner.usage.as_ref().unwrap().completion_tokens,
            3
        );

        assert_eq!(
            final_resp.inner.choices[0]
                .message
                .content
                .as_ref()
                .unwrap(),
            &ChatCompletionMessageContent::Text("Hello World".to_string())
        );
        assert_eq!(
            final_resp.inner.usage.as_ref().unwrap().completion_tokens,
            3
        );
    }

    #[tokio::test]
    async fn test_fold_without_metrics_leaves_llm_metrics_none() {
        let chunks = vec![
            create_mock_chunk("Hello".to_string(), 0),
            create_final_chunk(0),
        ];

        let (folded, future) = fold_aggregate_with_future(stream::iter(chunks));
        let results: Vec<_> = folded.collect().await;
        let _ = future.await.expect("aggregation should produce a record");

        assert_eq!(results.len(), 1);
        assert!(results[0].data.as_ref().unwrap().llm_metrics.is_none());
    }
}

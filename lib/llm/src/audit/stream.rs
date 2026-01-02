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

use dynamo_async_openai::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta};
use futures::StreamExt;

type AuditStream =
    Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>;
type AuditFuture =
    Pin<Box<dyn std::future::Future<Output = NvCreateChatCompletionResponse> + Send>>;

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
                    let chunks_stream = futures::stream::iter(chunks);
                    let parsing_options = ParsingOptions::default();

                    tokio::spawn(async move {
                        match DeltaAggregator::apply(chunks_stream, parsing_options).await {
                            Ok(final_resp) => {
                                let _ = tx.send(final_resp);
                            }
                            Err(e) => {
                                tracing::warn!("audit: aggregation failed: {e}");
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

/// Return (pass-through stream, future -> final aggregated response for audit).
pub fn scan_aggregate_with_future<S>(stream: S) -> (AuditStream, AuditFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Unpin + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<NvCreateChatCompletionResponse>();
    let passthrough = PassThroughWithAgg::new(stream, tx);
    (
        Box::pin(passthrough),
        Box::pin(async move {
            rx.await.unwrap_or_else(|_| {
                tracing::warn!("audit: aggregation future canceled/failed");
                // Return minimal response if aggregation failed
                NvCreateChatCompletionResponse {
                    id: String::new(),
                    created: 0,
                    usage: None,
                    model: String::new(),
                    object: "chat.completion".to_string(),
                    system_fingerprint: None,
                    choices: vec![],
                    service_tier: None,
                    nvext: None,
                }
            })
        }),
    )
}

/// Collect all chunks, aggregate them, then emit a single final chunk (for non-streaming)
pub fn fold_aggregate_with_future<S>(stream: S) -> (AuditStream, AuditFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<NvCreateChatCompletionResponse>();

    let single_chunk_stream = async move {
        let chunks: Vec<_> = stream.collect().await;
        let chunks_stream = futures::stream::iter(chunks);
        let parsing_options = ParsingOptions::default();

        match DeltaAggregator::apply(chunks_stream, parsing_options).await {
            Ok(final_resp) => {
                let _ = tx.send(final_resp.clone());
                final_response_to_one_chunk_stream(final_resp)
            }
            Err(e) => {
                tracing::warn!("fold aggregation failed: {e}");
                let fallback = NvCreateChatCompletionResponse {
                    id: String::new(),
                    created: 0,
                    usage: None,
                    model: String::new(),
                    object: "chat.completion".to_string(),
                    system_fingerprint: None,
                    choices: vec![],
                    service_tier: None,
                    nvext: None,
                };
                let _ = tx.send(fallback.clone());
                final_response_to_one_chunk_stream(fallback)
            }
        }
    };

    let future = Box::pin(async move {
        rx.await.unwrap_or_else(|_| {
            tracing::warn!("fold aggregation future canceled");
            NvCreateChatCompletionResponse {
                id: String::new(),
                created: 0,
                usage: None,
                model: String::new(),
                object: "chat.completion".to_string(),
                system_fingerprint: None,
                choices: vec![],
                service_tier: None,
                nvext: None,
            }
        })
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
    let mut choices: Vec<ChatChoiceStream> = Vec::with_capacity(resp.choices.len());
    for (idx, ch) in resp.choices.iter().enumerate() {
        // Convert FunctionCall to FunctionCallStream if present
        #[allow(deprecated)]
        let function_call = ch.message.function_call.as_ref().map(|fc| {
            dynamo_async_openai::types::FunctionCallStream {
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
                    |(i, call)| dynamo_async_openai::types::ChatCompletionMessageToolCallChunk {
                        index: i as u32,
                        id: Some(call.id.clone()),
                        r#type: Some(call.r#type.clone()),
                        function: Some(dynamo_async_openai::types::FunctionCallStream {
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
            stop_reason: ch.stop_reason.clone(),
            logprobs: ch.logprobs.clone(),
        };
        choices.push(choice);
    }

    let chunk = NvCreateChatCompletionStreamResponse {
        id: resp.id.clone(),
        object: "chat.completion.chunk".to_string(),
        created: resp.created,
        model: resp.model.clone(),
        system_fingerprint: resp.system_fingerprint.clone(),
        service_tier: resp.service_tier.clone(),
        choices,
        usage: resp.usage.clone(),
        nvext: resp.nvext.clone(),
    };

    let annotated = Annotated {
        data: Some(chunk),
        id: None,
        event: None,
        comment: None,
    };
    Box::pin(futures::stream::once(async move { annotated }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_async_openai::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, FinishReason, Role,
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
                content: Some(content),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            stop_reason: None,
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
            nvext: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
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
            stop_reason: None,
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
            nvext: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
        }
    }

    /// Helper to extract content from a chunk
    fn extract_content(chunk: &Annotated<NvCreateChatCompletionStreamResponse>) -> String {
        chunk
            .data
            .as_ref()
            .and_then(|d| d.choices.first())
            .and_then(|c| c.delta.content.as_ref())
            .cloned()
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
        let (passthrough, _future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;

        // Verify chunk count
        assert_eq!(results.len(), 3, "Should pass through all chunks unchanged");

        // Verify content is identical
        assert_eq!(extract_content(&results[0]), "Hello ");
        assert_eq!(extract_content(&results[1]), "World");
        assert_eq!(extract_content(&results[2]), ""); // Final chunk has no content

        // Verify complete content reconstruction
        assert_eq!(reconstruct_content(&results), "Hello World");
    }

    #[tokio::test]
    async fn test_empty_stream_handling() {
        // Empty stream should not panic and should provide fallback response
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let final_resp = future.await;

        // Verify empty passthrough
        assert_eq!(results.len(), 0, "Empty stream should produce no chunks");

        // Verify fallback response (aggregation will fail on empty stream)
        assert_eq!(final_resp.object, "chat.completion");
        // Should get fallback response, not panic
    }

    #[tokio::test]
    async fn test_single_chunk_stream() {
        // Single chunk should pass through and aggregate correctly
        let chunks = vec![create_mock_chunk("Single chunk".to_string(), 0)];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
        let results: Vec<_> = passthrough.collect().await;
        let final_resp = future.await;

        // Verify passthrough
        assert_eq!(results.len(), 1);
        assert_eq!(extract_content(&results[0]), "Single chunk");

        // Verify aggregation
        assert_eq!(final_resp.object, "chat.completion");
    }

    #[tokio::test]
    async fn test_chunks_with_metadata_preserved() {
        // Test that metadata (id, event, comment) is preserved through passthrough
        let chunk_with_metadata = Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![{
                    #[allow(deprecated)]
                    ChatChoiceStream {
                        index: 0,
                        delta: ChatCompletionStreamResponseDelta {
                            role: Some(Role::Assistant),
                            content: Some("Content".to_string()),
                            tool_calls: None,
                            function_call: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                        stop_reason: None,
                        logprobs: None,
                    }
                }],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
                nvext: None,
            }),
            id: Some("correlation-123".to_string()),
            event: Some("test-event".to_string()),
            comment: Some(vec!["test-comment".to_string()]),
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
        // Test that multiple concurrent audit streams don't interfere
        let chunks1 = vec![create_mock_chunk("Stream 1".to_string(), 0)];
        let chunks2 = vec![create_mock_chunk("Stream 2".to_string(), 0)];

        let (_, future1) = scan_aggregate_with_future(stream::iter(chunks1));
        let (_, future2) = scan_aggregate_with_future(stream::iter(chunks2));

        // Run both futures concurrently
        let (resp1, resp2) = tokio::join!(future1, future2);

        // Both should complete successfully
        assert_eq!(resp1.object, "chat.completion");
        assert_eq!(resp2.object, "chat.completion");
    }
}

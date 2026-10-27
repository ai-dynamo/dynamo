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

type PayloadStream =
    Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>;

/// Aggregated response for audit capture. Partial responses carry a `drop_reason`.
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

const DROP_EMPTY_RESPONSE_STREAM: &str = "empty_response_stream";

/// The aggregation never reported an outcome at all: something dropped the
/// pass-through stream before end-of-stream. A client disconnect does that, but so
/// does `http::service::disconnect` killing the engine context on a backend
/// inactivity timeout, and the two are indistinguishable from here. The reason names
/// what we observed rather than guessing at a cause.
const DROP_RESPONSE_STREAM_DROPPED: &str = "response_stream_dropped";

/// Build the `aggregation_failed` drop reason. The colon-delimited
/// `identifier:detail` shape matches the marker reasons `otel_sink.rs` already
/// publishes, so a consumer can parse one grammar across both producers.
fn aggregation_failed_reason(error: impl std::fmt::Display) -> String {
    format!("aggregation_failed:{error}")
}

type PayloadFuture = Pin<Box<dyn std::future::Future<Output = PayloadOutcome> + Send>>;

/// Forwards transformed chunks unchanged; collects them for aggregation.
pub struct PassThroughWithAgg<S> {
    inner: S,
    chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    done_tx: Option<oneshot::Sender<PayloadOutcome>>,
    parsing_options: ParsingOptions,
}

impl<S> PassThroughWithAgg<S> {
    fn new(inner: S, tx: oneshot::Sender<PayloadOutcome>, parsing_options: ParsingOptions) -> Self {
        Self {
            inner,
            chunks: Vec::new(),
            done_tx: Some(tx),
            parsing_options,
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
                self.chunks.push(chunk.clone());
                // Capture the prefix now: the SSE monitor drops this stream on error.
                if chunk.is_error()
                    && let Some(tx) = self.done_tx.take()
                {
                    let chunks = std::mem::take(&mut self.chunks);
                    let parsing_options = self.parsing_options.clone();
                    tokio::spawn(async move {
                        let _ =
                            tx.send(aggregate_with_partial_recovery(chunks, parsing_options).await);
                    });
                }
                Poll::Ready(Some(chunk))
            }
            Poll::Ready(None) => {
                if let Some(tx) = self.done_tx.take() {
                    let chunks = std::mem::take(&mut self.chunks);
                    if chunks.is_empty() {
                        tracing::debug!(
                            "request payload: empty response stream, no response to aggregate"
                        );
                        let _ = tx.send(PayloadOutcome::dropped(None, DROP_EMPTY_RESPONSE_STREAM));
                        return Poll::Ready(None);
                    }
                    let parsing_options = self.parsing_options.clone();

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
/// `DeltaAggregator` is deliberately left alone rather than taught to recover here:
/// its short-circuit on the first error-tagged chunk is load-bearing on the
/// client-facing non-streaming path, where a typed backend error must surface as an
/// error rather than as a truncated success.
async fn aggregate_with_partial_recovery(
    mut chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    parsing_options: ParsingOptions,
) -> PayloadOutcome {
    let Some(error_at) = chunks.iter().position(|chunk| chunk.is_error()) else {
        return match DeltaAggregator::apply_for_selected_unified_parser(
            futures::stream::iter(chunks),
            parsing_options,
        )
        .await
        {
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
    let partial = DeltaAggregator::apply_for_selected_unified_parser(
        futures::stream::iter(chunks),
        parsing_options,
    )
    .await
    .ok();
    PayloadOutcome::dropped(partial, reason)
}

pub fn scan_aggregate_with_future<S>(
    stream: S,
    parsing_options: ParsingOptions,
) -> (PayloadStream, PayloadFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Unpin + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<PayloadOutcome>();
    let passthrough = PassThroughWithAgg::new(stream, tx, parsing_options);
    (
        Box::pin(passthrough),
        Box::pin(async move {
            match rx.await {
                Ok(outcome) => outcome,
                Err(_) => {
                    // tx dropped without sending: the passthrough stream went away before
                    // end-of-stream, either because the client disconnected or because the
                    // engine context was killed under it.
                    tracing::debug!(
                        "request payload: response aggregation produced no outcome (stream dropped)"
                    );
                    PayloadOutcome::dropped(None, DROP_RESPONSE_STREAM_DROPPED)
                }
            }
        }),
    )
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
            prompt_logprobs: None,
            llm_metrics: None,
            tool_call_completion: Vec::new(),
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
            prompt_logprobs: None,
            llm_metrics: None,
            tool_call_completion: Vec::new(),
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
            prompt_logprobs: None,
            llm_metrics: None,
            tool_call_completion: Vec::new(),
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
            prompt_logprobs: None,
            llm_metrics: None,
            tool_call_completion: Vec::new(),
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
        let (passthrough, future) =
            scan_aggregate_with_future(input_stream, ParsingOptions::default());
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;
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
        let (passthrough, future) =
            scan_aggregate_with_future(input_stream, ParsingOptions::default());
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;
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

    fn call_without_explicit_tool_terminal(
        finish_reason: FinishReason,
    ) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        let name = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_weather".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: None,
            }),
        };
        let arguments = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(FunctionCallStream {
                name: None,
                arguments: Some("{\"city\":\"Tokyo\"}".to_string()),
            }),
        };
        vec![
            create_tool_call_chunk(name, None),
            create_tool_call_chunk(arguments, Some(finish_reason)),
        ]
    }

    fn unified_parser_options() -> ParsingOptions {
        ParsingOptions::new(Some("qwen3_coder".to_string()), Some("qwen3".to_string()))
    }

    #[tokio::test]
    async fn payload_capture_keeps_complete_backend_calls_at_eof() {
        let chunks = call_without_explicit_tool_terminal(FinishReason::Stop);
        let (passthrough, future) =
            scan_aggregate_with_future(stream::iter(chunks.clone()), unified_parser_options());
        let delivered = passthrough.collect::<Vec<_>>().await;
        let outcome = future.await;
        assert_eq!(delivered.len(), chunks.len());
        assert!(outcome.drop_reason.is_none());
        assert!(
            outcome
                .response
                .expect("complete output should be captured")
                .inner
                .choices[0]
                .message
                .tool_calls
                .is_some()
        );

        // Without a unified parser selected, the same structurally valid call is kept.
        let (passthrough, future) =
            scan_aggregate_with_future(stream::iter(chunks), ParsingOptions::default());
        let _ = passthrough.collect::<Vec<_>>().await;
        let outcome = future.await;
        assert!(
            outcome
                .response
                .expect("default aggregation should capture the response")
                .inner
                .choices[0]
                .message
                .tool_calls
                .is_some()
        );
    }

    #[tokio::test]
    async fn payload_capture_keeps_complete_backend_calls_in_error_prefix() {
        let mut chunks = call_without_explicit_tool_terminal(FinishReason::Stop);
        chunks
            .push(Annotated::<NvCreateChatCompletionStreamResponse>::from_error("backend failed"));
        let (passthrough, future) =
            scan_aggregate_with_future(stream::iter(chunks), unified_parser_options());
        let delivered = passthrough.take(3).collect::<Vec<_>>().await;
        let outcome = future.await;

        assert_eq!(delivered.len(), 3);
        assert!(delivered[2].is_error());
        assert!(
            outcome
                .drop_reason
                .as_deref()
                .is_some_and(|reason| reason.starts_with("aggregation_failed:"))
        );
        assert!(
            outcome
                .response
                .expect("the prefix should still be captured")
                .inner
                .choices[0]
                .message
                .tool_calls
                .is_some()
        );
    }

    #[tokio::test]
    async fn payload_capture_respects_request_tool_suppression() {
        let mut chunk = create_mock_chunk(
            "<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>"
                .to_string(),
            0,
        );
        chunk.data.as_mut().unwrap().inner.choices[0].finish_reason = Some(FinishReason::Stop);
        let options = ParsingOptions::new(Some("hermes".to_string()), None)
            .with_tool_call_parsing_enabled(false);
        let (passthrough, future) = scan_aggregate_with_future(stream::iter(vec![chunk]), options);

        let _ = passthrough.collect::<Vec<_>>().await;
        let outcome = future.await;
        assert!(outcome.drop_reason.is_none());
        assert!(
            outcome
                .response
                .expect("suppressed output should still be captured")
                .inner
                .choices[0]
                .message
                .tool_calls
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_empty_stream_handling() {
        // Empty stream: the aggregator has nothing to apply, so the outcome carries no
        // response, and names itself rather than looking like a dropped stream.
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) =
            scan_aggregate_with_future(input_stream, ParsingOptions::default());
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
            "an empty stream must be distinguishable from a dropped one"
        );
    }

    #[tokio::test]
    async fn test_single_chunk_stream() {
        // Single chunk should pass through and aggregate correctly
        let chunks = vec![create_mock_chunk("Single chunk".to_string(), 0)];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) =
            scan_aggregate_with_future(input_stream, ParsingOptions::default());
        let results: Vec<_> = passthrough.collect().await;
        let outcome = future.await;
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
                prompt_logprobs: None,
                llm_metrics: None,
                tool_call_completion: Vec::new(),
            }),
            id: Some("correlation-123".to_string()),
            event: Some("test-event".to_string()),
            comment: Some(vec!["test-comment".to_string()]),
            error: None,
        };

        let input_stream = stream::iter(vec![chunk_with_metadata.clone()]);
        let (passthrough, _future) =
            scan_aggregate_with_future(input_stream, ParsingOptions::default());
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
        // models the stream going away before the first poll — each future should
        // independently resolve to a dropped outcome without crosstalk.
        let chunks1 = vec![create_mock_chunk("Stream 1".to_string(), 0)];
        let chunks2 = vec![create_mock_chunk("Stream 2".to_string(), 0)];

        let (_, future1) =
            scan_aggregate_with_future(stream::iter(chunks1), ParsingOptions::default());
        let (_, future2) =
            scan_aggregate_with_future(stream::iter(chunks2), ParsingOptions::default());

        let (outcome1, outcome2) = tokio::join!(future1, future2);

        assert!(outcome1.response.is_none());
        assert!(outcome2.response.is_none());
        assert_eq!(
            outcome1.drop_reason.as_deref(),
            Some("response_stream_dropped")
        );
        assert_eq!(
            outcome2.drop_reason.as_deref(),
            Some("response_stream_dropped")
        );
    }

    #[tokio::test]
    async fn error_chunk_settles_the_outcome_without_reaching_end_of_stream() {
        // `http::service::disconnect` stops polling and drops the stream as soon as an
        // error reaches it, so end-of-stream never arrives on the streaming path. Model
        // that: take chunks up to and including the error, then drop the pass-through.
        let chunks = vec![
            create_mock_chunk("Hello ".to_string(), 0),
            Annotated::<NvCreateChatCompletionStreamResponse>::from_error(
                "invalid sampling parameter",
            ),
            create_mock_chunk("never polled".to_string(), 0),
        ];

        let (passthrough, future) =
            scan_aggregate_with_future(stream::iter(chunks), ParsingOptions::default());
        let delivered: Vec<_> = passthrough.take(2).collect().await;
        assert_eq!(delivered.len(), 2);
        // The error reaches the client unchanged; the handler surfaces it exactly
        // as it does with capture off.
        assert!(delivered[1].is_error());

        let outcome = future.await;

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
            "the record must name the backend error rather than report a dropped stream, got {reason}"
        );

        let partial = outcome
            .response
            .expect("content delivered before the error must survive the early drop");
        assert_eq!(
            partial.inner.choices[0].message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Hello ".to_string()),
        );
    }

    #[tokio::test]
    async fn error_as_first_chunk_reports_reason_without_a_response() {
        let chunks = vec![
            Annotated::<NvCreateChatCompletionStreamResponse>::from_error("backend unavailable"),
        ];

        let input_stream = stream::iter(chunks);
        let (passthrough, future) =
            scan_aggregate_with_future(input_stream, ParsingOptions::default());
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

    // Model-text fixtures cannot carry the internal parser completion evidence.
    #[tokio::test]
    async fn payload_capture_rejects_provisional_evidence_at_eof_and_error_prefix() {
        for error in [false, true] {
            let mut chunks = call_without_explicit_tool_terminal(FinishReason::Stop);
            chunks[0].data.as_mut().unwrap().tool_call_completion.push(
                crate::protocols::openai::chat_completions::ToolCallCompletion {
                    choice_index: 0,
                    tool_index: 0,
                    complete: false,
                },
            );
            if error {
                chunks.push(Annotated::from_error("backend failed"));
            }
            let (passthrough, future) =
                scan_aggregate_with_future(stream::iter(chunks), unified_parser_options());
            let delivered = passthrough.collect::<Vec<_>>().await;
            let outcome = future.await;
            assert_eq!(delivered.len(), if error { 3 } else { 2 });
            assert_eq!(outcome.drop_reason.is_some(), error);
            assert!(
                outcome.response.unwrap().inner.choices[0]
                    .message
                    .tool_calls
                    .is_none()
            );
        }
    }
}

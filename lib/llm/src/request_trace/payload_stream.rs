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
                self.chunks.push(chunk.clone());
                // Capture the prefix now: the SSE monitor drops this stream on error.
                if chunk.is_error()
                    && let Some(tx) = self.done_tx.take()
                {
                    let chunks = std::mem::take(&mut self.chunks);
                    let parsing_options = ParsingOptions::default();
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
/// `DeltaAggregator` is deliberately left alone rather than taught to recover here:
/// its short-circuit on the first error-tagged chunk is load-bearing on the
/// client-facing non-streaming path, where a typed backend error must surface as an
/// error rather than as a truncated success.
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

/// A well-formed response carrying no choices, sent to the client when aggregation
/// failed so the HTTP response shape stays valid.
fn empty_fallback_response() -> NvCreateChatCompletionResponse {
    NvCreateChatCompletionResponse {
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
    }
}

/// Fold a non-streaming payload into one final client chunk while forwarding
/// metrics as they arrive.
///
/// Metrics must bypass the fold because downstream latency metrics depend on
/// observation time. Typed metrics are moved out of buffered chunks and
/// forwarded on metric-only frames that keep them typed (no JSON round-trip),
/// while event-tagged annotations are forwarded without their payload data.
///
/// At end of stream, the remaining client data is aggregated into one chunk.
/// The folded chunk carries no `llm_metrics` because they were already emitted
/// in-stream (#11349).
pub fn fold_aggregate_with_future<S>(stream: S) -> (PayloadStream, PayloadFuture)
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<PayloadOutcome>();

    let out = async_stream::stream! {
        let mut stream = std::pin::pin!(stream);
        let mut buffered: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = Vec::new();
        let mut forwarded_frames: usize = 0;

        while let Some(mut chunk) = stream.next().await {
            // Each source chunk yields at most one observable metrics frame.
            // Typed metrics win over an event annotation on the same chunk,
            // the same precedence the collector applies to an unfolded
            // stream, so the fold cannot observe one chunk twice. The
            // tool-call jail produces such dual-carrier chunks: it attaches
            // typed `llm_metrics` to the payload-usage tail, which also
            // carries its own annotation.
            if let Some(metrics) = chunk.data.as_mut().and_then(|data| data.llm_metrics.take()) {
                // Forward typed metrics immediately, still typed, so the
                // collector observes them without a serialize/deserialize
                // round-trip.
                forwarded_frames += 1;
                yield typed_metric_frame(metrics);
            } else if matches!(
                chunk.event.as_deref(),
                Some(ANNOTATION_LLM_METRICS) | Some(ANNOTATION_PAYLOAD_USAGE)
            ) {
                // Forward the event annotation as a data-less shell, leaving
                // any payload data behind for aggregation. The annotation is
                // parsed once, by the collector.
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
            "request payload: metric frames forwarded ahead of the non-streaming fold"
        );

        let parsing_options = ParsingOptions::default();
        let outcome = aggregate_with_partial_recovery(buffered, parsing_options).await;

        // A dropped outcome may still carry the pre-error prefix for the record, but the
        // client gets the empty fallback either way: a truncated aggregation presented as
        // the whole answer is worse than an empty one the caller can recognize as such.
        let client_response = match (&outcome.drop_reason, &outcome.response) {
            (None, Some(complete)) => complete.clone(),
            _ => empty_fallback_response(),
        };
        let _ = tx.send(outcome);
        yield final_response_to_one_chunk(client_response);
    };

    let future = Box::pin(async move {
        match rx.await {
            Ok(outcome) => outcome,
            Err(_) => {
                tracing::debug!(
                    "request payload: fold response aggregation produced no outcome (stream dropped)"
                );
                PayloadOutcome::dropped(None, DROP_RESPONSE_STREAM_DROPPED)
            }
        }
    });

    (Box::pin(out), future)
}

/// Build a metric-only frame that carries `metrics` typed on `llm_metrics`.
///
/// The response envelope is an empty internal carrier: no choices, no usage,
/// so the HTTP aggregator folds it as a no-op, and the folded client chunk
/// that follows re-supplies `id`/`model`/`created`. `llm_metrics` is
/// `#[serde(skip)]`, so nothing here reaches the client wire format.
fn typed_metric_frame(
    metrics: LLMMetricAnnotation,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: String::new(),
                choices: vec![],
                created: 0,
                model: String::new(),
                system_fingerprint: None,
                object: String::new(),
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: Some(metrics),
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
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
        let (passthrough, future) = scan_aggregate_with_future(input_stream);
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
        // Empty stream: the aggregator has nothing to apply, so the outcome carries no
        // response, and names itself rather than looking like a dropped stream.
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
            "an empty stream must be distinguishable from a dropped one"
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
        // models the stream going away before the first poll — each future should
        // independently resolve to a dropped outcome without crosstalk.
        let chunks1 = vec![create_mock_chunk("Stream 1".to_string(), 0)];
        let chunks2 = vec![create_mock_chunk("Stream 2".to_string(), 0)];

        let (_, future1) = scan_aggregate_with_future(stream::iter(chunks1));
        let (_, future2) = scan_aggregate_with_future(stream::iter(chunks2));

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

        let (passthrough, future) = scan_aggregate_with_future(stream::iter(chunks));
        let delivered: Vec<_> = passthrough.take(2).collect().await;
        assert_eq!(delivered.len(), 2);

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

    #[tokio::test]
    async fn fold_success_delivers_and_records_the_complete_response() {
        let chunks = vec![
            create_mock_chunk("Hello ".to_string(), 0),
            create_mock_chunk("World".to_string(), 0),
            create_final_chunk(0),
        ];

        let (folded, future) = fold_aggregate_with_future(stream::iter(chunks));
        let delivered: Vec<_> = folded.collect().await;
        let outcome = future.await;

        assert_eq!(delivered.len(), 1, "the fold path emits a single chunk");
        assert_eq!(extract_content(&delivered[0]), "Hello World");
        let client_response = delivered[0]
            .data
            .as_ref()
            .expect("the successful chunk carries a body");
        assert_eq!(client_response.inner.choices.len(), 1);
        assert_eq!(
            client_response.inner.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );

        assert!(
            outcome.drop_reason.is_none(),
            "a successful fold must not carry a drop reason"
        );
        let recorded = outcome
            .response
            .expect("the complete response must reach the record");
        assert_eq!(recorded.inner.choices.len(), 1);
        assert_eq!(
            recorded.inner.choices[0].message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Hello World".to_string())
        );
        assert_eq!(
            recorded.inner.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[tokio::test]
    async fn fold_error_after_content_records_the_prefix_and_sends_the_fallback() {
        let chunks = vec![
            create_mock_chunk("Hello ".to_string(), 0),
            Annotated::<NvCreateChatCompletionStreamResponse>::from_error(
                "invalid sampling parameter",
            ),
        ];

        let (folded, future) = fold_aggregate_with_future(stream::iter(chunks));
        let delivered: Vec<_> = folded.collect().await;
        let outcome = future.await;

        assert_eq!(delivered.len(), 1, "the fold path emits a single chunk");
        assert!(
            delivered[0]
                .data
                .as_ref()
                .expect("the fallback chunk carries a body")
                .inner
                .choices
                .is_empty(),
            "the client gets the empty fallback, not the truncated aggregation"
        );

        let reason = outcome
            .drop_reason
            .as_deref()
            .expect("an errored fold must carry a drop reason");
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
            .expect("content collected before the error must reach the record");
        assert_eq!(
            partial.inner.choices[0].message.content.as_ref().unwrap(),
            &ChatCompletionMessageContent::Text("Hello ".to_string()),
            "only the pre-error prefix should be aggregated"
        );
    }

    /// Per-chunk metrics as the preprocessor attaches them to content
    /// chunks: no `cached_tokens`, which only the usage tail carries.
    fn typed_metrics(chunk_tokens: usize, output_tokens: usize) -> LLMMetricAnnotation {
        LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens,
            chunk_tokens,
            cached_tokens: None,
            ..Default::default()
        }
    }

    /// Serialized annotation comment for `metrics`, tag-agnostic.
    fn annotation_comment(metrics: &LLMMetricAnnotation) -> Option<Vec<String>> {
        metrics.to_annotation::<()>().unwrap().comment
    }

    /// Usage chunk shaped like the preprocessor's payload-only usage tail.
    fn usage_chunk(output_tokens: usize) -> NvCreateChatCompletionStreamResponse {
        NvCreateChatCompletionStreamResponse {
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
        }
    }

    /// Production-shaped payload-usage tail: usage data plus its metric
    /// annotation under the `payload_usage` tag, with `chunk_tokens = 0`.
    ///
    /// The tag is set explicitly: `to_annotation` always emits the
    /// `llm_metrics` tag, which is not what the preprocessor puts on this
    /// chunk when payload capture is on.
    fn payload_usage_tail(output_tokens: usize) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let metrics = LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens,
            chunk_tokens: 0,
            cached_tokens: Some(4),
            ..Default::default()
        };

        Annotated {
            data: Some(usage_chunk(output_tokens)),
            id: None,
            event: Some(ANNOTATION_PAYLOAD_USAGE.to_string()),
            comment: annotation_comment(&metrics),
            error: None,
        }
    }

    /// Legacy data-less metrics frame under the `llm_metrics` tag, as emitted
    /// by engines that report metrics as annotations rather than on the typed
    /// field.
    fn llm_metrics_event_frame(
        chunk_tokens: usize,
        output_tokens: usize,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        typed_metrics(chunk_tokens, output_tokens)
            .to_annotation::<NvCreateChatCompletionStreamResponse>()
            .unwrap()
    }

    /// Dual-carrier chunk: the payload-usage tail after the tool-call jail has
    /// attached typed `llm_metrics` to it, so typed metrics and the
    /// `payload_usage` annotation ride on the same chunk.
    fn dual_carrier_tail(
        typed_output_tokens: usize,
        annotation_output_tokens: usize,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let mut tail = payload_usage_tail(annotation_output_tokens);
        tail.data.as_mut().unwrap().llm_metrics =
            Some(typed_metrics(typed_output_tokens, typed_output_tokens));
        tail
    }

    #[tokio::test]
    async fn test_fold_forwards_metric_frames_before_folded_chunk() {
        let mut chunk1 = create_mock_chunk("Hello ".to_string(), 0);
        chunk1.data.as_mut().unwrap().llm_metrics = Some(typed_metrics(1, 1));
        let mut chunk2 = create_mock_chunk("World".to_string(), 0);
        chunk2.data.as_mut().unwrap().llm_metrics = Some(typed_metrics(2, 3));

        // A legacy data-less `llm_metrics` frame leads the stream, so both
        // annotation tags the fold forwards are exercised: dropping either
        // arm of the `matches!` changes the frame count below.
        let chunks = vec![
            llm_metrics_event_frame(0, 0),
            chunk1,
            chunk2,
            create_final_chunk(0),
            payload_usage_tail(3),
        ];

        let (folded, future) = fold_aggregate_with_future(stream::iter(chunks));
        let results: Vec<_> = folded.collect().await;
        let outcome = future.await;
        assert!(
            outcome.drop_reason.is_none(),
            "complete fold must not carry a drop reason"
        );
        let final_resp = outcome
            .response
            .expect("aggregation should produce a record");

        // One legacy shell, two typed metric frames, one payload-usage shell,
        // then one folded client chunk.
        assert_eq!(results.len(), 5);

        // The legacy annotation is forwarded under its own tag.
        let legacy = &results[0];
        assert!(legacy.data.is_none());
        assert_eq!(legacy.event.as_deref(), Some(ANNOTATION_LLM_METRICS));
        let legacy_metrics = LLMMetricAnnotation::from_annotation(legacy)
            .unwrap()
            .expect("legacy llm_metrics shell must parse");
        assert_eq!(legacy_metrics.input_tokens, 10);

        // Typed metrics stay typed: carried on `llm_metrics` of an empty
        // envelope, never re-encoded as an annotation.
        for (frame, expected_chunk_tokens) in results[1..3].iter().zip([1usize, 2]) {
            let data = frame
                .data
                .as_ref()
                .expect("typed metric frame must carry data");
            let metrics = data
                .llm_metrics
                .as_ref()
                .expect("typed metric frame must carry llm_metrics");
            assert_eq!(metrics.chunk_tokens, expected_chunk_tokens);
            assert!(data.inner.choices.is_empty());
            assert!(data.inner.usage.is_none());
            assert!(frame.event.is_none());
            assert!(frame.comment.is_none());
        }

        // The payload-usage annotation is forwarded without payload data and
        // under its production tag; usage remains in the fold.
        let shell = &results[3];
        assert!(shell.data.is_none());
        assert_eq!(shell.event.as_deref(), Some(ANNOTATION_PAYLOAD_USAGE));
        let tail_metrics = LLMMetricAnnotation::from_annotation(shell)
            .unwrap()
            .expect("payload-usage shell must parse");
        assert_eq!(tail_metrics.chunk_tokens, 0);
        assert_eq!(tail_metrics.output_tokens, 3);

        // Exactly one folded client chunk; metrics must not be replayed on it.
        let folded_chunk = results[4].data.as_ref().expect("folded client chunk");
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

        // The blank envelopes of the metric frames must not blank the
        // response identity: the folded chunk re-supplies it.
        assert_eq!(folded_chunk.inner.id, "test-id");
        assert_eq!(folded_chunk.inner.model, "test-model");
        assert_eq!(folded_chunk.inner.created, 1234567890);

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
        assert_eq!(final_resp.inner.id, "test-id");
        assert_eq!(final_resp.inner.model, "test-model");
        assert_eq!(final_resp.inner.created, 1234567890);
    }

    /// A chunk carrying both typed `llm_metrics` and a `payload_usage`
    /// annotation must yield exactly one observable metrics frame, with the
    /// typed form winning, as it does on the unfolded stream. Two frames
    /// would make the collector observe the chunk twice; since the second
    /// observation is the zero-token tail, the visible damage is the final
    /// OSL being overwritten with the annotation's value.
    #[tokio::test]
    async fn test_fold_dual_carrier_chunk_yields_one_metrics_frame() {
        let chunks = vec![
            create_mock_chunk("Hello".to_string(), 0),
            create_final_chunk(0),
            dual_carrier_tail(5, 3),
        ];

        let (folded, future) = fold_aggregate_with_future(stream::iter(chunks));
        let results: Vec<_> = folded.collect().await;
        let outcome = future.await;
        assert!(
            outcome.drop_reason.is_none(),
            "complete fold must not carry a drop reason"
        );
        let final_resp = outcome
            .response
            .expect("aggregation should produce a record");

        // One typed metric frame, then the folded client chunk.
        assert_eq!(results.len(), 2);

        let observable = results
            .iter()
            .filter(|frame| {
                let typed = frame
                    .data
                    .as_ref()
                    .is_some_and(|data| data.llm_metrics.is_some());
                let annotated = matches!(LLMMetricAnnotation::from_annotation(frame), Ok(Some(_)));
                typed || annotated
            })
            .count();
        assert_eq!(
            observable, 1,
            "a dual-carrier chunk must be observable once"
        );

        let typed = results[0]
            .data
            .as_ref()
            .and_then(|data| data.llm_metrics.as_ref())
            .expect("typed metrics must win over the annotation");
        assert_eq!(typed.output_tokens, 5);
        assert_eq!(typed.chunk_tokens, 5);
        assert!(results[0].event.is_none());
        assert!(results[0].comment.is_none());

        // The annotation is not forwarded, but its usage payload still folds.
        let folded_chunk = results[1].data.as_ref().expect("folded client chunk");
        assert!(folded_chunk.llm_metrics.is_none());
        assert!(results[1].event.is_none());
        assert_eq!(
            folded_chunk.inner.usage.as_ref().unwrap().completion_tokens,
            3
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
        let outcome = future.await;
        assert!(
            outcome.drop_reason.is_none(),
            "complete fold must not carry a drop reason"
        );
        assert!(
            outcome.response.is_some(),
            "aggregation should produce a record"
        );

        assert_eq!(results.len(), 1);
        assert!(results[0].data.as_ref().unwrap().llm_metrics.is_none());
    }
}

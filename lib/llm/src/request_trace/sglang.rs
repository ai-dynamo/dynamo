// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-trace adapter for native SGLang `/generate` responses.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use dynamo_runtime::engine::{
    AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, EngineStream,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::Stream;
use serde_json::Value;

use super::SharedFinishReasonMetadata;
use super::integration::{RequestEndTraceState, emit_request_end_trace_state};
use crate::discovery::GenerateTraceConfig;
use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use dynamo_runtime::pipeline::Context;

const NATIVE_RESPONSE_KEY: &str = "sglang_response";

pub(crate) struct SglangRequestTrace(SglangRequestTraceState);

enum SglangRequestTraceState {
    Disabled,
    Pending,
    Active(Box<ActiveSglangRequestTrace>),
}

struct ActiveSglangRequestTrace {
    request_end: RequestEndTraceState,
    tool_call_parser: Option<String>,
}

impl SglangRequestTrace {
    pub(crate) fn new() -> Self {
        if super::policy().emit_request_end_records() {
            Self(SglangRequestTraceState::Pending)
        } else {
            Self(SglangRequestTraceState::Disabled)
        }
    }

    pub(crate) fn needs_runtime_config(&self) -> bool {
        !matches!(self.0, SglangRequestTraceState::Disabled)
    }

    pub(crate) fn prepare(
        self,
        context: Context<PreprocessedRequest>,
        trace_config: Option<GenerateTraceConfig>,
    ) -> (Self, Context<PreprocessedRequest>) {
        if matches!(self.0, SglangRequestTraceState::Disabled) {
            return (self, context);
        }
        let Some(trace_config) = trace_config else {
            tracing::warn!(
                "native SGLang request trace skipped because runtime metadata is missing"
            );
            return (Self(SglangRequestTraceState::Disabled), context);
        };
        let (mut request, context) = context.into_parts();
        crate::preprocessor::attach_agent_context_from_context(&mut request, &context);
        let Some(request_end) = super::integration::build_new_request_end_trace_state(
            &mut request,
            &context,
            trace_config.kv_cache_block_size as usize,
        ) else {
            return (
                Self(SglangRequestTraceState::Disabled),
                context.map(|_| request),
            );
        };
        let tool_call_parser = request_end
            .finish_reason_metadata()
            .and(trace_config.tool_call_parser);
        (
            Self(SglangRequestTraceState::Active(Box::new(
                ActiveSglangRequestTrace {
                    request_end,
                    tool_call_parser,
                },
            ))),
            context.map(|_| request),
        )
    }

    pub(crate) fn wrap(
        self,
        stream: EngineStream<Annotated<LLMEngineOutput>>,
        request_id: String,
    ) -> EngineStream<Annotated<LLMEngineOutput>> {
        let active = match self.0 {
            SglangRequestTraceState::Active(active) => active,
            SglangRequestTraceState::Disabled | SglangRequestTraceState::Pending => return stream,
        };
        let engine_context = stream.context();
        Box::pin(SglangTraceStream::new(
            stream,
            engine_context,
            active.request_end,
            active.tool_call_parser,
            request_id,
        ))
    }
}

struct SglangTraceStream {
    inner: Option<EngineStream<Annotated<LLMEngineOutput>>>,
    engine_context: Arc<dyn AsyncEngineContext>,
    request_end: Option<RequestEndTraceState>,
    request_id: Option<String>,
    tracker: Arc<RequestTracker>,
    finish_reason_metadata: Option<SharedFinishReasonMetadata>,
    tool_call_parser: Option<String>,
    output_text: Option<String>,
    first_token_recorded: bool,
    observed_osl: usize,
    terminal_recorded: bool,
}

impl std::fmt::Debug for SglangTraceStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SglangTraceStream")
            .field("trace_active", &self.request_end.is_some())
            .field("terminal_recorded", &self.terminal_recorded)
            .finish_non_exhaustive()
    }
}

impl SglangTraceStream {
    fn new(
        stream: EngineStream<Annotated<LLMEngineOutput>>,
        engine_context: Arc<dyn AsyncEngineContext>,
        request_end: RequestEndTraceState,
        tool_call_parser: Option<String>,
        request_id: String,
    ) -> Self {
        let tracker = request_end.request_tracker();
        let finish_reason_metadata = request_end.finish_reason_metadata();
        Self {
            inner: Some(stream),
            engine_context,
            request_end: Some(request_end),
            request_id: Some(request_id),
            tracker,
            finish_reason_metadata,
            output_text: tool_call_parser.as_ref().map(|_| String::new()),
            tool_call_parser,
            first_token_recorded: false,
            observed_osl: 0,
            terminal_recorded: false,
        }
    }

    fn observe(&mut self, response: &Annotated<LLMEngineOutput>) {
        let Some(native_response) = native_response(response) else {
            return;
        };
        if let Some(output_ids) = native_response.get("output_ids").and_then(Value::as_array) {
            self.observed_osl = self.observed_osl.saturating_add(output_ids.len());
            if !self.first_token_recorded && !output_ids.is_empty() {
                self.tracker.record_first_token();
                self.first_token_recorded = true;
            }
        }
        if let Some(output_text) = self.output_text.as_mut()
            && let Some(text) = native_response.get("text").and_then(Value::as_str)
        {
            output_text.push_str(text);
        }

        let Some(meta_info) = native_response.get("meta_info") else {
            return;
        };
        if let Some(completion_tokens) = usize_field(meta_info, "completion_tokens") {
            self.observed_osl = completion_tokens;
        }
        let Some(finish_reason) = meta_info
            .get("finish_reason")
            .filter(|value| !value.is_null())
        else {
            return;
        };
        if self.terminal_recorded {
            return;
        }

        self.terminal_recorded = true;
        record_input_usage(&self.tracker, meta_info);
        if let Some(metadata) = self.finish_reason_metadata.as_ref() {
            record_finish_reason(metadata, finish_reason);
        }
    }

    fn take_tool_parse(&mut self) -> Option<(String, String, SharedFinishReasonMetadata)> {
        if !self.terminal_recorded {
            return None;
        }
        Some((
            self.output_text
                .take()
                .filter(|output| !output.is_empty())?,
            self.tool_call_parser.take()?,
            self.finish_reason_metadata.take()?,
        ))
    }

    fn finish(&mut self) {
        drop(self.inner.take());
        if self.request_end.is_none() {
            return;
        }
        // The router's RequestGuard writes its final metrics when the inner
        // stream is dropped. Record the native observed length afterward so
        // the two observers stay ordered without a global atomic RMW.
        self.tracker.record_osl(self.observed_osl);
        self.tracker.record_finish();
        let (Some(request_end), Some(request_id)) =
            (self.request_end.take(), self.request_id.take())
        else {
            return;
        };
        let Some((output_text, tool_call_parser, metadata)) = self.take_tool_parse() else {
            emit_request_end_trace_state(request_end, request_id);
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                "native SGLang tool metadata skipped because no Tokio runtime is available"
            );
            emit_request_end_trace_state(request_end, request_id);
            return;
        };
        runtime.spawn(async move {
            record_tool_calls(&metadata, &output_text, &tool_call_parser).await;
            emit_request_end_trace_state(request_end, request_id);
        });
    }
}

impl Drop for SglangTraceStream {
    fn drop(&mut self) {
        self.finish();
    }
}

impl Stream for SglangTraceStream {
    type Item = Annotated<LLMEngineOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(response)) => {
                this.observe(&response);
                Poll::Ready(Some(response))
            }
            Poll::Ready(None) => {
                this.finish();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl AsyncEngineContextProvider for SglangTraceStream {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.engine_context.clone()
    }
}

impl AsyncEngineStream<Annotated<LLMEngineOutput>> for SglangTraceStream {}

fn native_response(response: &Annotated<LLMEngineOutput>) -> Option<&Value> {
    response
        .data
        .as_ref()?
        .engine_data
        .as_ref()?
        .get(NATIVE_RESPONSE_KEY)
}

fn record_input_usage(
    tracker: &crate::protocols::common::timing::RequestTracker,
    meta_info: &Value,
) {
    let prompt_tokens = usize_field(meta_info, "prompt_tokens");
    let cached_tokens = usize_field(meta_info, "cached_tokens");
    if prompt_tokens.is_some() || cached_tokens.is_some() {
        tracker.record_isl(prompt_tokens.unwrap_or(0), cached_tokens);
    }
}

fn usize_field(value: &Value, field: &str) -> Option<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn record_finish_reason(metadata: &SharedFinishReasonMetadata, finish_reason: &Value) {
    let Some(finish_type) = finish_reason
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| finish_reason.as_str())
    else {
        return;
    };
    let stop_reason = finish_reason
        .get("matched")
        .and_then(|matched| match matched {
            Value::String(value) => {
                Some(dynamo_protocols::types::StopReason::String(value.clone()))
            }
            Value::Number(value) => value.as_i64().map(dynamo_protocols::types::StopReason::Int),
            _ => None,
        });
    metadata.record_backend_finish_reason(Some(0), Some(finish_type.to_string()), stop_reason);

    let normalized = match finish_type {
        "stop" | "eos" => Some(dynamo_protocols::types::FinishReason::Stop),
        "length" => Some(dynamo_protocols::types::FinishReason::Length),
        "content_filter" => Some(dynamo_protocols::types::FinishReason::ContentFilter),
        "tool_calls" => Some(dynamo_protocols::types::FinishReason::ToolCalls),
        "function_call" => Some(dynamo_protocols::types::FinishReason::FunctionCall),
        _ => None,
    };
    if let Some(normalized) = normalized {
        metadata.record_choice_finish_reason(0, normalized);
    }
}

async fn record_tool_calls(
    metadata: &SharedFinishReasonMetadata,
    output_text: &str,
    tool_call_parser: &str,
) {
    let tool_calls = match dynamo_parsers::tool_calling::try_tool_call_parse_aggregate_finalize(
        output_text,
        Some(tool_call_parser),
        None,
    )
    .await
    {
        Ok((tool_calls, _)) => tool_calls,
        Err(error) => {
            tracing::debug!(
                %error,
                parser = tool_call_parser,
                "failed to parse native SGLang output for request-trace tool metadata"
            );
            return;
        }
    };

    for (tool_call_index, tool_call) in tool_calls.iter().enumerate() {
        let Ok(tool_call_index) = u32::try_from(tool_call_index) else {
            tracing::warn!("too many native SGLang tool calls to represent in request trace");
            break;
        };
        metadata.record_tool_call(
            0,
            tool_call_index,
            Some(&tool_call.id),
            Some(&tool_call.function.name),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_protocols::types::{FinishReason, StopReason};
    use dynamo_runtime::engine::ResponseStream;
    use futures::StreamExt;

    use super::*;
    use crate::protocols::common::extensions::AgentContext;
    use crate::protocols::common::timing::RequestTracker;
    use crate::request_trace::{
        AgentContextTraceState, BUS, RequestReplayMetrics, request_metrics_from_agent_state,
    };

    fn engine_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>> + Send + 'static,
    ) -> EngineStream<Annotated<LLMEngineOutput>> {
        let context = Context::new(());
        ResponseStream::new(Box::pin(stream), context.context())
    }

    struct RecordOslOnDropStream {
        items: std::vec::IntoIter<Annotated<LLMEngineOutput>>,
        tracker: Arc<RequestTracker>,
        drop_osl: usize,
    }

    impl Stream for RecordOslOnDropStream {
        type Item = Annotated<LLMEngineOutput>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
        ) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.items.next())
        }
    }

    impl Drop for RecordOslOnDropStream {
        fn drop(&mut self) {
            self.tracker.record_osl(self.drop_osl);
        }
    }

    fn agent_state(
        finish_reason_metadata: SharedFinishReasonMetadata,
        tracker: Option<Arc<RequestTracker>>,
    ) -> AgentContextTraceState {
        AgentContextTraceState {
            agent_context: AgentContext {
                session_id: "rollout-1".to_string(),
                parent_session_id: None,
                session_final: None,
                kv_hints: None,
                input_trigger: None,
            },
            request_model: "test-model".to_string(),
            request_tracker: tracker,
            x_request_id: Some("rollout-call-1".to_string()),
            finish_reason_metadata,
        }
    }

    fn snapshot(metadata: SharedFinishReasonMetadata) -> super::super::FinishReasonMetadata {
        request_metrics_from_agent_state(agent_state(metadata, None), "req-sglang".to_string())
            .1
            .finish_reason_metadata
            .expect("finish metadata should be recorded")
    }

    #[test]
    fn normalizes_known_finish_reasons_only() {
        for (finish_type, expected) in [
            ("stop", Some(FinishReason::Stop)),
            ("length", Some(FinishReason::Length)),
            ("content_filter", Some(FinishReason::ContentFilter)),
            ("tool_calls", Some(FinishReason::ToolCalls)),
            ("function_call", Some(FinishReason::FunctionCall)),
            ("cancelled", None),
            ("abort", None),
        ] {
            let metadata = SharedFinishReasonMetadata::default();
            record_finish_reason(&metadata, &serde_json::json!({"type": finish_type}));

            let metadata = snapshot(metadata);
            assert_eq!(metadata.backend_finish_reason.as_deref(), Some(finish_type));
            assert_eq!(metadata.finish_reason, expected);
            assert_eq!(metadata.choices[0].finish_reason, expected);
        }
    }

    #[tokio::test]
    async fn parsed_tools_do_not_override_length() {
        let metadata = SharedFinishReasonMetadata::default();
        record_finish_reason(&metadata, &serde_json::json!({"type": "length"}));
        record_tool_calls(
            &metadata,
            r#"<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>"#,
            "qwen25",
        )
        .await;

        let metadata = snapshot(metadata);
        assert_eq!(metadata.finish_reason, Some(FinishReason::Length));
        assert_eq!(metadata.tool_calls.len(), 1);
        assert_eq!(metadata.tool_calls[0].name.as_deref(), Some("get_weather"));
    }

    #[tokio::test]
    async fn disabled_trace_returns_original_engine_stream() {
        let stream = engine_stream(futures::stream::empty());
        let original =
            (&*stream as *const dyn AsyncEngineStream<Annotated<LLMEngineOutput>>) as *const ();
        let wrapped = SglangRequestTrace(SglangRequestTraceState::Disabled)
            .wrap(stream, "unused-request-id".to_string());
        let returned =
            (&*wrapped as *const dyn AsyncEngineStream<Annotated<LLMEngineOutput>>) as *const ();

        assert_eq!(original, returned);
    }

    #[tokio::test]
    async fn stream_emits_agent_finish_and_usage_metadata() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let metadata = SharedFinishReasonMetadata::default();
        let state = RequestEndTraceState::new(
            Some(agent_state(metadata, Some(tracker.clone()))),
            tracker,
            Arc::new(RequestReplayMetrics {
                trace_block_size: 2,
                input_length: 3,
                input_sequence_hashes: vec![11, 22],
            }),
        );
        let stream = engine_stream(futures::stream::iter([
            Annotated::from_data(LLMEngineOutput {
                engine_data: Some(serde_json::json!({
                    "sglang_response": {
                        "text": "<think>Inspect the weather.</think><tool_call>{\"name\":\"get_",
                        "output_ids": [101],
                        "meta_info": {
                            "finish_reason": null,
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "cached_tokens": 1
                        }
                    }
                })),
                ..Default::default()
            }),
            Annotated::from_data(LLMEngineOutput {
                engine_data: Some(serde_json::json!({
                    "sglang_response": {
                        "text": "weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call>",
                        "output_ids": [102],
                        "meta_info": {
                            "finish_reason": {"type": "stop", "matched": "END"},
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "cached_tokens": 1
                        }
                    }
                })),
                ..Default::default()
            }),
        ]));

        let responses = SglangRequestTrace(SglangRequestTraceState::Active(Box::new(
            ActiveSglangRequestTrace {
                request_end: state,
                tool_call_parser: Some("qwen25".to_string()),
            },
        )))
        .wrap(stream, "req-sglang".to_string())
        .collect::<Vec<_>>()
        .await;
        assert_eq!(responses.len(), 2);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == "req-sglang")
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        let request = record.request.expect("request metrics");
        assert_eq!(request.x_request_id.as_deref(), Some("rollout-call-1"));
        assert_eq!(request.input_tokens, Some(3));
        assert_eq!(request.output_tokens, Some(2));
        assert_eq!(request.cached_tokens, Some(1));
        assert!(request.ttft_ms.is_some());
        assert!(request.total_time_ms.is_some());
        let finish = request.finish_reason_metadata.expect("finish metadata");
        assert_eq!(finish.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(finish.backend_finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            finish.stop_reason,
            Some(StopReason::String("END".to_string()))
        );
        assert_eq!(finish.tool_calls.len(), 1);
        assert!(finish.tool_calls[0].id.is_some());
        assert_eq!(finish.tool_calls[0].name.as_deref(), Some("get_weather"));
        assert!(
            serde_json::to_value(&finish.tool_calls[0])
                .unwrap()
                .get("arguments")
                .is_none()
        );
    }

    #[tokio::test]
    async fn drop_records_observed_length_after_inner_stream() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let state = RequestEndTraceState::new(
            None,
            tracker.clone(),
            Arc::new(RequestReplayMetrics {
                trace_block_size: 2,
                input_length: 3,
                input_sequence_hashes: vec![11, 22],
            }),
        );
        let response = Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({
                "sglang_response": {
                    "output_ids": [101, 102],
                    "meta_info": {
                        "finish_reason": null,
                        "completion_tokens": 2
                    }
                }
            })),
            ..Default::default()
        });
        let stream = engine_stream(RecordOslOnDropStream {
            items: vec![response].into_iter(),
            tracker,
            drop_osl: 1,
        });
        let mut wrapped = SglangRequestTrace(SglangRequestTraceState::Active(Box::new(
            ActiveSglangRequestTrace {
                request_end: state,
                tool_call_parser: None,
            },
        )))
        .wrap(stream, "req-drop-order".to_string());

        assert!(wrapped.next().await.is_some());
        drop(wrapped);

        let record = loop {
            let record = receiver.recv().await.unwrap();
            if record
                .request
                .as_ref()
                .is_some_and(|request| request.request_id == "req-drop-order")
            {
                break record;
            }
        };
        assert_eq!(record.request.unwrap().output_tokens, Some(2));
    }
}

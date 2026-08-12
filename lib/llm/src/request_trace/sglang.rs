// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-trace adapter for native SGLang `/generate` responses.

use std::pin::Pin;

use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use serde_json::Value;

use super::integration::{
    RequestEndTraceState, finish_reason_metadata_handle,
    wrap_request_end_stream as wrap_generic_request_end_stream,
};
use super::{SharedFinishReasonMetadata, record_llm_metric_tokens};
use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::openai::ParsingOptions;
use dynamo_runtime::pipeline::Context;

const NATIVE_RESPONSE_KEY: &str = "sglang_response";

pub(crate) struct SglangRequestTrace {
    request_end: Option<RequestEndTraceState>,
    parsing_options: ParsingOptions,
}

impl SglangRequestTrace {
    pub(crate) fn prepare(
        request: &mut PreprocessedRequest,
        context: &Context<()>,
        trace_block_size: u32,
        parsing_options: ParsingOptions,
    ) -> Self {
        crate::preprocessor::attach_agent_context_from_context(request, context);
        let tracker = super::policy()
            .emit_request_end_records()
            .then(|| std::sync::Arc::new(crate::protocols::common::timing::RequestTracker::new()));
        if let Some(tracker) = tracker.as_ref() {
            tracker.record_isl(request.token_ids.len(), None);
        }
        let request_end = super::integration::build_request_end_trace_state(
            request,
            &tracker,
            context,
            trace_block_size as usize,
        );
        request.tracker = tracker;
        Self {
            request_end,
            parsing_options,
        }
    }

    pub(crate) fn wrap<S>(
        self,
        stream: S,
        request_id: String,
    ) -> Pin<Box<dyn Stream<Item = Annotated<LLMEngineOutput>> + Send>>
    where
        S: Stream<Item = Annotated<LLMEngineOutput>> + Send + 'static,
    {
        let Self {
            request_end,
            parsing_options,
        } = self;
        let Some(state) = request_end.as_ref() else {
            return Box::pin(stream);
        };
        let finish_reason_metadata = finish_reason_metadata_handle(&request_end);
        let tracker = state.request_tracker();
        let tool_call_parser = finish_reason_metadata
            .as_ref()
            .and_then(|_| effective_tool_call_parser(parsing_options));

        let stream = async_stream::stream! {
            futures::pin_mut!(stream);
            let mut output_text = String::new();
            let mut tool_metadata_recorded = false;
            while let Some(response) = stream.next().await {
                let mut terminal = false;
                if let Some(native_response) = native_response(&response) {
                    if native_response
                        .get("output_ids")
                        .and_then(Value::as_array)
                        .is_some_and(|output_ids| !output_ids.is_empty())
                    {
                        tracker.record_first_token();
                    }

                    // Dynamo exposes only SGLang's incremental SSE mode, so each
                    // response text is a delta and can be appended directly.
                    if tool_call_parser.is_some()
                        && let Some(text) = native_response.get("text").and_then(Value::as_str)
                    {
                        output_text.push_str(text);
                    }

                    if let Some(meta_info) = native_response.get("meta_info") {
                        record_usage(&tracker, meta_info);
                        if let Some(finish_reason) = meta_info
                            .get("finish_reason")
                            .filter(|value| !value.is_null())
                        {
                            terminal = true;
                            tracker.record_finish();
                            if let Some(metadata) = finish_reason_metadata.as_ref() {
                                record_finish_reason(metadata, finish_reason);
                            }
                        }
                    }
                }

                if terminal
                    && !tool_metadata_recorded
                    && let (Some(metadata), Some(parser)) =
                        (finish_reason_metadata.as_ref(), tool_call_parser.as_deref())
                {
                    record_tool_calls(metadata, &output_text, parser).await;
                    tool_metadata_recorded = true;
                }
                yield response;
            }
        };
        wrap_generic_request_end_stream(Box::pin(stream), request_end, request_id)
    }
}

fn native_response(response: &Annotated<LLMEngineOutput>) -> Option<&Value> {
    response
        .data
        .as_ref()?
        .engine_data
        .as_ref()?
        .get(NATIVE_RESPONSE_KEY)
}

fn effective_tool_call_parser(parsing_options: ParsingOptions) -> Option<String> {
    parsing_options.tool_call_parser.or_else(|| {
        parsing_options
            .reasoning_parser
            .filter(|parser| matches!(parser.as_str(), "kimi_k3" | "kimi-k3"))
    })
}

fn record_usage(tracker: &crate::protocols::common::timing::RequestTracker, meta_info: &Value) {
    let prompt_tokens = usize_field(meta_info, "prompt_tokens");
    let completion_tokens = usize_field(meta_info, "completion_tokens");
    let cached_tokens = usize_field(meta_info, "cached_tokens");
    if prompt_tokens.is_none() && completion_tokens.is_none() && cached_tokens.is_none() {
        return;
    }

    record_llm_metric_tokens(
        Some(tracker),
        prompt_tokens,
        completion_tokens
            .unwrap_or_else(|| usize::try_from(tracker.osl_tokens()).unwrap_or(usize::MAX)),
        cached_tokens,
    );
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
        "stop" | "eos" | "cancelled" | "abort" => Some(dynamo_protocols::types::FinishReason::Stop),
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
    if output_text.is_empty() {
        return;
    }

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
    if !tool_calls.is_empty() {
        metadata.record_choice_finish_reason(0, dynamo_protocols::types::FinishReason::ToolCalls);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_protocols::types::{FinishReason, StopReason};
    use futures::StreamExt;

    use super::*;
    use crate::protocols::common::extensions::AgentContext;
    use crate::protocols::common::timing::RequestTracker;
    use crate::request_trace::{
        AgentContextTraceState, BUS, RequestReplayMetrics, request_metrics_from_agent_state,
    };

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
    fn records_native_finish_reason_metadata() {
        let metadata = SharedFinishReasonMetadata::default();
        record_finish_reason(&metadata, &serde_json::json!({"type": "tool_calls"}));

        let metadata = snapshot(metadata);
        assert_eq!(
            metadata.backend_finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert_eq!(metadata.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(metadata.choices.len(), 1);
        assert_eq!(
            metadata.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn parses_tool_metadata_without_arguments() {
        let metadata = SharedFinishReasonMetadata::default();
        record_tool_calls(
            &metadata,
            r#"<think>Inspect the weather.</think><tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>"#,
            "qwen25",
        )
        .await;

        let metadata = snapshot(metadata);
        assert_eq!(metadata.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(metadata.tool_calls.len(), 1);
        assert!(metadata.tool_calls[0].id.is_some());
        assert_eq!(metadata.tool_calls[0].name.as_deref(), Some("get_weather"));
        assert!(
            serde_json::to_value(&metadata.tool_calls[0])
                .unwrap()
                .get("arguments")
                .is_none()
        );
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
        let stream = futures::stream::iter([
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
        ]);

        let responses = SglangRequestTrace {
            request_end: Some(state),
            parsing_options: ParsingOptions::new(Some("qwen25".to_string()), None),
        }
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
    }
}

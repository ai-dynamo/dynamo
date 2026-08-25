// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-end tracing for native SGLang `/generate` streams.

use std::sync::Arc;

use dynamo_runtime::engine::{AsyncEngineContextProvider, EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use parking_lot::Mutex;

use super::integration::{RequestEndTraceState, emit_request_end_trace_state};
use crate::discovery::GenerateTraceConfig;
use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::pipeline::Context;

pub(crate) struct SglangRequestTrace(SglangRequestTraceState);

enum SglangRequestTraceState {
    Disabled,
    Pending,
    Active(Box<ActiveSglangRequestTrace>),
}

struct ActiveSglangRequestTrace {
    request_end: RequestEndTraceState,
    tool_trace: Option<ToolTraceConfig>,
}

struct ToolTraceConfig {
    parser: String,
    tokenizer: crate::tokenizers::Tokenizer,
}

#[derive(Default)]
struct ToolTraceObservation {
    output_ids: Vec<u32>,
    terminal: bool,
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
        let tool_trace = request_end
            .observer()
            .records_finish_reason_metadata()
            .then(|| {
                trace_config
                    .tool_call_parser
                    .zip(trace_config.tokenizer)
                    .map(|(parser, tokenizer)| ToolTraceConfig { parser, tokenizer })
            })
            .flatten();

        (
            Self(SglangRequestTraceState::Active(Box::new(
                ActiveSglangRequestTrace {
                    request_end,
                    tool_trace,
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

        let ActiveSglangRequestTrace {
            request_end,
            tool_trace,
        } = *active;
        let engine_context = stream.context();
        let observer = request_end.observer();
        let tool_observation = tool_trace
            .as_ref()
            .map(|_| Arc::new(Mutex::new(ToolTraceObservation::default())));
        let stream_observer = observer.clone();
        let stream_tool_observation = tool_observation.clone();
        let stream = stream.map(move |response| {
            if let Some(output) = response.data.as_ref() {
                let usage = output.completion_usage.as_ref();
                let output_tokens = usage.map_or_else(
                    || {
                        usize::try_from(stream_observer.tracker().osl_tokens())
                            .unwrap_or(usize::MAX)
                    },
                    |usage| usage.completion_tokens as usize,
                );
                let input_tokens = usage.map(|usage| usage.prompt_tokens as usize);
                let cached_tokens = usage
                    .and_then(|usage| usage.prompt_tokens_details.as_ref())
                    .and_then(|details| details.cached_tokens)
                    .map(|tokens| tokens as usize);
                stream_observer.observe_backend_chunk(
                    output.index,
                    output.finish_reason.as_ref(),
                    output.stop_reason.as_ref(),
                    input_tokens,
                    output_tokens,
                    cached_tokens,
                );

                if let Some(tool_observation) = stream_tool_observation.as_ref() {
                    let mut tool_observation = tool_observation.lock();
                    tool_observation
                        .output_ids
                        .extend_from_slice(&output.token_ids);
                    tool_observation.terminal |= output.finish_reason.is_some();
                }
                if let Some(finish_reason) = output.finish_reason.clone()
                    && stream_observer.observe_chat_finish_reason_from_backend(
                        output.index.unwrap_or(0),
                        finish_reason,
                    )
                {
                    stream_observer.tracker().record_finish();
                }
            }
            response
        });

        let (stream, done) = crate::telemetry::stream::notify_on_completion(Box::pin(stream));
        tokio::spawn(async move {
            done.await;
            if let (Some(tool_trace), Some(observation)) = (tool_trace, tool_observation) {
                let observation = Arc::try_unwrap(observation)
                    .map(Mutex::into_inner)
                    .unwrap_or_else(|shared| {
                        let mut state = shared.lock();
                        std::mem::take(&mut *state)
                    });
                if observation.terminal && !observation.output_ids.is_empty() {
                    record_tool_calls(observer.clone(), tool_trace, observation.output_ids).await;
                }
            }
            emit_request_end_trace_state(request_end, request_id);
        });
        ResponseStream::new(stream, engine_context)
    }
}

async fn record_tool_calls(
    observer: super::RequestEndTraceObserver,
    config: ToolTraceConfig,
    output_ids: Vec<u32>,
) {
    let decoded =
        tokio::task::spawn_blocking(move || config.tokenizer.decode(&output_ids, false)).await;
    let output_text = match decoded {
        Ok(Ok(output_text)) => output_text,
        Ok(Err(error)) => {
            tracing::debug!(%error, "failed to decode native SGLang tool trace");
            return;
        }
        Err(error) => {
            tracing::warn!(%error, "native SGLang tool decode task failed");
            return;
        }
    };
    let tool_calls = match dynamo_parsers::tool_calling::try_tool_call_parse_aggregate_finalize(
        output_text.as_str(),
        Some(&config.parser),
        None,
    )
    .await
    {
        Ok((tool_calls, _)) => tool_calls,
        Err(error) => {
            tracing::debug!(%error, parser = config.parser, "failed to parse native SGLang tool trace");
            return;
        }
    };

    for (tool_call_index, tool_call) in tool_calls.iter().enumerate() {
        let Ok(tool_call_index) = u32::try_from(tool_call_index) else {
            tracing::warn!("too many native SGLang tool calls to represent in request trace");
            break;
        };
        observer.observe_tool_call(
            0,
            tool_call_index,
            Some(&tool_call.id),
            Some(&tool_call.function.name),
        );
    }
    observer.reconcile_tool_call_finish_reason(0);
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context as TaskContext, Poll};
    use std::time::Duration;

    use dynamo_protocols::types::{FinishReason, StopReason};
    use dynamo_runtime::engine::ResponseStream;
    use futures::Stream;

    use super::*;
    use crate::protocols::common::FinishReason as BackendFinishReason;
    use crate::protocols::common::extensions::AgentContext;
    use crate::protocols::common::timing::RequestTracker;
    use crate::request_trace::{AgentContextTraceState, BUS, RequestReplayMetrics};

    struct TrackerDropStream {
        tracker: Arc<RequestTracker>,
    }

    impl Stream for TrackerDropStream {
        type Item = Annotated<LLMEngineOutput>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl Drop for TrackerDropStream {
        fn drop(&mut self) {
            self.tracker.record_osl(9);
            self.tracker.record_finish();
        }
    }

    fn engine_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>> + Send + 'static,
    ) -> EngineStream<Annotated<LLMEngineOutput>> {
        let context = Context::new(());
        ResponseStream::new(Box::pin(stream), context.context())
    }

    fn active_trace(
        tracker: Arc<RequestTracker>,
        tool_trace: Option<ToolTraceConfig>,
    ) -> SglangRequestTrace {
        let agent_context = AgentContext::builder()
            .session_id("rollout-1".to_string())
            .build()
            .unwrap();
        let state = RequestEndTraceState::new(
            Some(AgentContextTraceState {
                agent_context,
                request_model: "test-model".to_string(),
                request_tracker: Some(tracker.clone()),
                x_request_id: Some("rollout-call-1".to_string()),
                finish_reason_metadata: Default::default(),
            }),
            tracker,
            Arc::new(RequestReplayMetrics {
                trace_block_size: 2,
                input_length: 3,
                input_sequence_hashes: vec![11, 22],
            }),
        );
        SglangRequestTrace(SglangRequestTraceState::Active(Box::new(
            ActiveSglangRequestTrace {
                request_end: state,
                tool_trace,
            },
        )))
    }

    #[tokio::test]
    async fn disabled_trace_returns_original_engine_stream() {
        let stream = engine_stream(futures::stream::empty());
        let original =
            (&*stream as *const dyn dynamo_runtime::engine::AsyncEngineStream<_>) as *const ();
        let wrapped = SglangRequestTrace(SglangRequestTraceState::Disabled)
            .wrap(stream, "unused".to_string());
        let returned =
            (&*wrapped as *const dyn dynamo_runtime::engine::AsyncEngineStream<_>) as *const ();
        assert_eq!(original, returned);
    }

    #[tokio::test]
    async fn native_common_output_emits_chat_compatible_agent_metadata() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tokenizer = crate::tokenizers::Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
        ))
        .unwrap();
        let output_ids = tokenizer
            .encode(r#"<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>"#)
            .unwrap()
            .token_ids()
            .to_vec();
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_first_token();
        let output = LLMEngineOutput {
            token_ids: output_ids.clone(),
            finish_reason: Some(BackendFinishReason::Stop),
            stop_reason: Some(StopReason::String("END".to_string())),
            index: Some(0),
            completion_usage: Some(dynamo_protocols::types::CompletionUsage {
                prompt_tokens: 3,
                completion_tokens: output_ids.len() as u32,
                total_tokens: 3 + output_ids.len() as u32,
                prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                    cached_tokens: Some(1),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let stream = engine_stream(futures::stream::iter([Annotated::from_data(output)]));
        let wrapped = active_trace(
            tracker,
            Some(ToolTraceConfig {
                parser: "qwen25".to_string(),
                tokenizer,
            }),
        )
        .wrap(stream, "req-sglang".to_string());
        futures::pin_mut!(wrapped);
        while wrapped.next().await.is_some() {}

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
        let request = record.request.unwrap();
        assert_eq!(request.x_request_id.as_deref(), Some("rollout-call-1"));
        assert_eq!(request.model.as_deref(), Some("test-model"));
        assert_eq!(request.input_tokens, Some(3));
        assert_eq!(request.output_tokens, Some(output_ids.len() as u64));
        assert_eq!(request.cached_tokens, Some(1));
        let finish = request.finish_reason_metadata.unwrap();
        assert_eq!(finish.backend_finish_reason.as_deref(), Some("stop"));
        assert_eq!(finish.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(
            finish.stop_reason,
            Some(StopReason::String("END".to_string()))
        );
        assert_eq!(finish.tool_calls[0].name.as_deref(), Some("get_weather"));
    }

    #[tokio::test]
    async fn request_end_emits_after_inner_router_stream_drop() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let stream = engine_stream(TrackerDropStream {
            tracker: tracker.clone(),
        });
        let wrapped = active_trace(tracker, None).wrap(stream, "req-drop-order".to_string());
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == "req-drop-order")
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(record.request.unwrap().output_tokens, Some(9));
    }
}

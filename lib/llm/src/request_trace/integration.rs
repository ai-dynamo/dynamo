// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::pipeline::Context;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionStreamResponse, completions::NvCreateCompletionResponse,
};
use crate::request_trace::{
    AgentContextTraceState, RequestReplayMetrics, SharedFinishReasonMetadata,
};

struct RequestTraceRequestEndState {
    request_tracker: Arc<RequestTracker>,
    replay_metrics: Arc<RequestReplayMetrics>,
}

pub(crate) struct RequestEndTraceState {
    agent: Option<AgentContextTraceState>,
    request: RequestTraceRequestEndState,
}

fn request_trace_rejection(common_request: &PreprocessedRequest) -> Option<&'static str> {
    if common_request.prompt_embeds.is_some() {
        return Some("prompt embeddings are not supported");
    }
    if common_request.multi_modal_data.is_some() {
        return Some("multimodal inputs are not supported");
    }
    if common_request.sampling_options.n.unwrap_or(1) > 1 {
        return Some("multiple output choices are not supported");
    }
    if common_request.sampling_options.best_of.unwrap_or(1) > 1 {
        return Some("best_of greater than one is not supported");
    }
    None
}

fn shared_replay_metrics(
    token_ids: &[crate::protocols::TokenIdType],
    trace_block_size: usize,
) -> Option<Arc<RequestReplayMetrics>> {
    if trace_block_size == 0 {
        return None;
    }
    super::replay_metrics(token_ids, trace_block_size).map(Arc::new)
}

pub(crate) fn build_request_end_trace_state(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
) -> Option<RequestEndTraceState> {
    build_request_end_trace_state_for_policy(
        common_request,
        tracker,
        context,
        trace_block_size,
        super::is_enabled(),
    )
}

fn build_request_end_trace_state_for_policy(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
    request_trace_enabled: bool,
) -> Option<RequestEndTraceState> {
    let has_agent_context = common_request.agent_context.is_some();

    if !request_trace_enabled {
        return None;
    }

    let request_id = context.id();
    if let Some(reason) = request_trace_rejection(common_request) {
        tracing::warn!(
            %request_id,
            reason,
            "request trace skipped because the request cannot be represented as one Mooncake row"
        );
        return None;
    }

    let request_tracker = match tracker {
        Some(tracker) => tracker.clone(),
        None => {
            tracing::warn!(
                %request_id,
                "request trace skipped because the request tracker is unavailable"
            );
            return None;
        }
    };

    let replay_metrics = match shared_replay_metrics(&common_request.token_ids, trace_block_size) {
        Some(metrics) => metrics,
        None => {
            tracing::warn!(
                %request_id,
                "request trace skipped because the KV cache block size is unavailable"
            );
            return None;
        }
    };

    let agent = has_agent_context
        .then(|| super::build_agent_context_trace_state(common_request, tracker, context))
        .flatten();

    let request = RequestTraceRequestEndState {
        request_tracker,
        replay_metrics,
    };

    Some(RequestEndTraceState { agent, request })
}

pub(crate) fn finish_reason_metadata_handle(
    trace_state: &Option<RequestEndTraceState>,
) -> Option<SharedFinishReasonMetadata> {
    trace_state
        .as_ref()
        .and_then(|state| state.agent.as_ref())
        .map(|state| state.finish_reason_metadata.clone())
}

fn wrap_request_end_stream<Resp>(
    stream: Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>
where
    Resp: Send + 'static,
{
    let Some(trace_state) = trace_state else {
        return stream;
    };

    let (stream, done) = crate::telemetry::stream::notify_on_completion(stream);
    tokio::spawn(async move {
        done.await;
        let request_state = trace_state.request;
        if let Some(agent_state) = trace_state.agent {
            let (agent_context, mut metrics) =
                super::request_metrics_from_agent_state(agent_state, request_id.clone());
            metrics.replay = Some(super::into_owned_replay_metrics(
                request_state.replay_metrics,
            ));
            super::record::emit_agent_request_end(agent_context, metrics);
        } else {
            super::record::emit_request_end(
                request_id.clone(),
                &request_state.request_tracker,
                super::into_owned_replay_metrics(request_state.replay_metrics),
            );
        }
    });
    stream
}

pub(crate) fn wrap_chat_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state, request_id);
    };

    let stream = stream.map(move |response| {
        super::record_chat_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state, request_id)
}

pub(crate) fn wrap_completion_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state, request_id);
    };

    let stream = stream.map(move |response| {
        super::record_completion_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state, request_id)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::task::{Context as TaskContext, Poll};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::protocols::common::extensions::AgentContext;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use crate::request_trace::BUS;
    use crate::request_trace::RequestTraceEventSource;

    struct TrackerDropStream {
        tracker: Arc<RequestTracker>,
        dropped: Arc<AtomicBool>,
    }

    impl Stream for TrackerDropStream {
        type Item = Annotated<NvCreateCompletionResponse>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl Drop for TrackerDropStream {
        fn drop(&mut self) {
            self.tracker.record_osl(9);
            self.tracker.record_finish();
            self.dropped.store(true, Ordering::Release);
        }
    }

    fn preprocessed_request(sampling_options: SamplingOptions) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(sampling_options)
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    #[test]
    fn rejects_unsupported_request_shapes() {
        let mut multi_choice = preprocessed_request(SamplingOptions {
            n: Some(2),
            ..Default::default()
        });
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("multiple output choices are not supported")
        );

        multi_choice.sampling_options.n = Some(1);
        multi_choice.sampling_options.best_of = Some(2);
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("best_of greater than one is not supported")
        );

        multi_choice.sampling_options.best_of = Some(1);
        multi_choice.prompt_embeds = Some("embedding".to_string());
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("prompt embeddings are not supported")
        );

        multi_choice.prompt_embeds = None;
        multi_choice.multi_modal_data = Some(Default::default());
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("multimodal inputs are not supported")
        );
    }

    #[test]
    fn replay_hashing_requires_block_size() {
        assert!(shared_replay_metrics(&[1, 2, 3], 0).is_none());

        let replay = shared_replay_metrics(&[1, 2, 3], 2).unwrap();
        assert_eq!(replay.input_sequence_hashes.len(), 2);
    }

    #[test]
    fn long_isl_hashing_reports_mode_costs_without_threshold() {
        let token_ids = (0..131_072_u32).collect::<Vec<_>>();

        let started = Instant::now();
        let request_only = shared_replay_metrics(&token_ids, 64).unwrap();
        let request_elapsed = started.elapsed();

        let started = Instant::now();
        let repeated = shared_replay_metrics(&token_ids, 64).unwrap();
        let repeated_elapsed = started.elapsed();

        eprintln!(
            "long-ISL replay hashing: request_only={request_elapsed:?}, repeated={repeated_elapsed:?}"
        );
        assert_eq!(request_only.input_sequence_hashes.len(), 2_048);
        assert_eq!(
            request_only.input_sequence_hashes,
            repeated.input_sequence_hashes
        );
    }

    #[tokio::test]
    async fn cancellation_reads_tracker_after_inner_stream_drop() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let state = RequestEndTraceState {
            agent: None,
            request: RequestTraceRequestEndState {
                request_tracker: tracker.clone(),
                replay_metrics: Arc::new(RequestReplayMetrics {
                    trace_block_size: 2,
                    input_length: 2,
                    input_sequence_hashes: vec![11],
                }),
            },
        };
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped =
            wrap_request_end_stream(Box::pin(stream), Some(state), "req-drop".to_string());
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == "req-drop")
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        let request = record.request.as_ref().expect("request payload");
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(request.output_tokens, Some(9));
    }

    #[tokio::test]
    async fn agent_context_emits_enriched_request_trace_row() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let state = RequestEndTraceState {
            agent: Some(AgentContextTraceState {
                agent_context: AgentContext {
                    session_id: "root".to_string(),
                    parent_session_id: None,
                    session_final: None,
                    kv_hints: None,
                },
                request_model: "test-model".to_string(),
                request_tracker: Some(tracker.clone()),
                x_request_id: Some("llm-call-1".to_string()),
                finish_reason_metadata: SharedFinishReasonMetadata::default(),
            }),
            request: RequestTraceRequestEndState {
                request_tracker: tracker.clone(),
                replay_metrics: Arc::new(RequestReplayMetrics {
                    trace_block_size: 2,
                    input_length: 2,
                    input_sequence_hashes: vec![11],
                }),
            },
        };
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped =
            wrap_request_end_stream(Box::pin(stream), Some(state), "req-agent".to_string());
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == "req-agent")
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(record.event_source, Some(RequestTraceEventSource::Dynamo));
        assert_eq!(
            record
                .agent_context
                .as_ref()
                .expect("agent context")
                .session_id,
            "root"
        );
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.model.as_deref(), Some("test-model"));
        assert_eq!(request.x_request_id.as_deref(), Some("llm-call-1"));
        assert_eq!(request.output_tokens, Some(9));
        assert_eq!(
            request
                .replay
                .as_ref()
                .expect("replay metrics")
                .input_length,
            2
        );
    }

    #[test]
    fn agent_context_does_not_bypass_request_trace_eligibility() {
        let mut request = preprocessed_request(SamplingOptions {
            best_of: Some(2),
            ..Default::default()
        });
        request.agent_context = Some(AgentContext {
            session_id: "root".to_string(),
            parent_session_id: None,
            session_final: None,
            kv_hints: None,
        });
        let tracker = Some(Arc::new(RequestTracker::new()));
        let context = Context::new(());

        let state = build_request_end_trace_state_for_policy(&request, &tracker, &context, 2, true);

        assert!(state.is_none());
    }
}

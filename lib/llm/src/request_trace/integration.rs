// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::pipeline::Context;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use crate::agents::trace::{
    AgentReplayMetrics, AgentTraceRequestEndState, SharedFinishReasonMetadata,
};
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionStreamResponse, completions::NvCreateCompletionResponse,
};

struct RequestTraceRequestEndState {
    request_tracker: Arc<RequestTracker>,
    replay_metrics: Arc<AgentReplayMetrics>,
}

pub(crate) struct RequestEndTraceState {
    agent: Option<AgentTraceRequestEndState>,
    request: Option<RequestTraceRequestEndState>,
}

impl RequestEndTraceState {
    pub(crate) fn is_enabled(&self) -> bool {
        self.agent.is_some() || self.request.is_some()
    }
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
    request_trace_supported: bool,
    agent_replay_enabled: bool,
    token_ids: &[crate::protocols::TokenIdType],
    trace_block_size: usize,
) -> Option<Arc<AgentReplayMetrics>> {
    if trace_block_size == 0 || (!request_trace_supported && !agent_replay_enabled) {
        return None;
    }
    crate::agents::trace::replay_metrics(token_ids, trace_block_size).map(Arc::new)
}

pub(crate) fn build_request_end_trace_state(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
) -> Option<RequestEndTraceState> {
    let request_trace_enabled = super::is_enabled();
    let agent_trace_enabled =
        crate::agents::trace::is_enabled() && common_request.agent_context.is_some();

    if !request_trace_enabled && !agent_trace_enabled {
        return None;
    }

    let request_id = context.id();
    let request_trace_supported = request_trace_enabled
        && match request_trace_rejection(common_request) {
            Some(reason) => {
                tracing::warn!(
                    %request_id,
                    reason,
                    "request trace skipped because the request cannot be represented as one Mooncake row"
                );
                false
            }
            None => true,
        };
    let request_trace_supported = request_trace_supported
        && if tracker.is_none() {
            tracing::warn!(
                %request_id,
                "request trace skipped because the request tracker is unavailable"
            );
            false
        } else {
            true
        };
    let request_trace_supported = request_trace_supported
        && if trace_block_size == 0 {
            tracing::warn!(
                %request_id,
                "request trace skipped because the KV cache block size is unavailable"
            );
            false
        } else {
            true
        };

    let agent_replay_enabled =
        agent_trace_enabled && crate::agents::trace::policy().replay_hashes_enabled;
    if agent_replay_enabled && trace_block_size == 0 {
        tracing::warn!(
            %request_id,
            "agent trace replay hashes requested but model KV cache block size is unavailable"
        );
    }

    let replay_metrics = shared_replay_metrics(
        request_trace_supported,
        agent_replay_enabled,
        &common_request.token_ids,
        trace_block_size,
    );

    let agent = crate::agents::trace::build_agent_trace_request_end_state(
        common_request,
        tracker,
        context,
        agent_replay_enabled
            .then(|| replay_metrics.clone())
            .flatten(),
    );
    let request = request_trace_supported
        .then(|| {
            Some(RequestTraceRequestEndState {
                request_tracker: tracker.clone()?,
                replay_metrics: replay_metrics.clone()?,
            })
        })
        .flatten();

    let state = RequestEndTraceState { agent, request };
    state.is_enabled().then_some(state)
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
        if let Some(request_state) = trace_state.request {
            super::record::emit_request_end(
                request_id.clone(),
                &request_state.request_tracker,
                crate::agents::trace::into_owned_replay_metrics(request_state.replay_metrics),
            );
        }
        if let Some(agent_state) = trace_state.agent {
            crate::agents::trace::emit_agent_trace_request_end(agent_state, request_id);
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
        crate::agents::trace::record_chat_finish_reason_metadata(
            &finish_reason_metadata,
            &response,
        );
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
        crate::agents::trace::record_completion_finish_reason_metadata(
            &finish_reason_metadata,
            &response,
        );
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
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use crate::request_trace::BUS;

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
    fn replay_hashing_is_disabled_when_unused_and_shared_when_both_need_it() {
        assert!(shared_replay_metrics(false, false, &[1, 2, 3], 2).is_none());

        let replay = shared_replay_metrics(true, true, &[1, 2, 3], 2).unwrap();
        let request_replay = replay.clone();
        let agent_replay = replay.clone();
        assert!(Arc::ptr_eq(&request_replay, &agent_replay));
        assert_eq!(replay.input_sequence_hashes.len(), 2);
    }

    #[test]
    fn long_isl_hashing_reports_mode_costs_without_threshold() {
        let token_ids = (0..131_072_u32).collect::<Vec<_>>();

        let started = Instant::now();
        let disabled = shared_replay_metrics(false, false, &token_ids, 64);
        let disabled_elapsed = started.elapsed();

        let started = Instant::now();
        let request_only = shared_replay_metrics(true, false, &token_ids, 64).unwrap();
        let request_elapsed = started.elapsed();

        let started = Instant::now();
        let both = shared_replay_metrics(true, true, &token_ids, 64).unwrap();
        let both_elapsed = started.elapsed();

        eprintln!(
            "long-ISL replay hashing: disabled={disabled_elapsed:?}, request_only={request_elapsed:?}, both={both_elapsed:?}"
        );
        assert!(disabled.is_none());
        assert_eq!(request_only.input_sequence_hashes.len(), 2_048);
        assert_eq!(
            request_only.input_sequence_hashes,
            both.input_sequence_hashes
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
            request: Some(RequestTraceRequestEndState {
                request_tracker: tracker.clone(),
                replay_metrics: Arc::new(AgentReplayMetrics {
                    trace_block_size: 2,
                    input_length: 2,
                    input_sequence_hashes: vec![11],
                }),
            }),
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
                if record.request.request_id == "req-drop" {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(record.request.output_tokens, 9);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::engine::{AsyncEngineContext, AsyncEngineContextProvider};
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
use dynamo_kv_router::protocols::BlockHashOptions;

struct RequestTraceRequestEndState {
    request_tracker: Arc<RequestTracker>,
    replay_metrics: Arc<RequestReplayMetrics>,
    model: String,
    x_request_id: Option<String>,
}

pub(crate) struct RequestEndTraceState {
    agent: Option<AgentContextTraceState>,
    request: Option<RequestTraceRequestEndState>,
    request_id: String,
    request_context: Arc<dyn AsyncEngineContext>,
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

/// Which request records the policy enables.
#[derive(Clone, Copy)]
struct RequestRecords {
    start: bool,
    end: bool,
}

/// Start tracing a preprocessed request, before it is dispatched: publish its
/// `request_start` row now, and return the state that publishes `request_end`
/// when the response completes. `is_eagle` selects the router's EAGLE block
/// layout for the `request_start` hashes.
pub(crate) fn start_request_trace(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
    is_eagle: bool,
) -> Option<RequestEndTraceState> {
    let policy = super::policy();
    start_request_trace_for_policy(
        common_request,
        tracker,
        context,
        trace_block_size,
        is_eagle,
        RequestRecords {
            start: policy.emit_request_start_records(),
            end: policy.emit_request_end_records(),
        },
    )
}

fn start_request_trace_for_policy(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
    is_eagle: bool,
    records: RequestRecords,
) -> Option<RequestEndTraceState> {
    let has_agent_context = common_request.agent_context.is_some();

    if !records.start && !records.end {
        return None;
    }

    let request_id = context.id();
    if let Some(reason) = request_trace_rejection(common_request) {
        tracing::warn!(
            %request_id,
            reason,
            "request trace skipped because the request cannot be represented as one replay request"
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

    if trace_block_size == 0 {
        tracing::warn!(
            %request_id,
            "request trace skipped because the KV cache block size is unavailable"
        );
        return None;
    }

    let x_request_id = super::x_request_id_from_context(context);
    if records.start {
        // Hash as the router does for selection, so the keys match the
        // engine's KV events for salted and LoRA requests too.
        let routing = common_request.routing.as_ref();
        let options = BlockHashOptions {
            block_mm_infos: None,
            lora_name: routing.and_then(|routing| routing.lora_name.as_deref()),
            cache_namespace: routing.and_then(|routing| routing.cache_namespace.as_deref()),
            is_eagle: Some(is_eagle),
        };
        super::record::emit_request_start(
            request_id.to_string(),
            x_request_id.clone(),
            common_request.model.clone(),
            &request_tracker,
            common_request.token_ids.len(),
            super::routing_hashes(&common_request.token_ids, trace_block_size, options),
        );
    }
    if !records.end {
        return None;
    }

    let replay_metrics = shared_replay_metrics(&common_request.token_ids, trace_block_size)?;

    let agent = has_agent_context
        .then(|| super::build_agent_context_trace_state(common_request, tracker, context))
        .flatten();

    let request = RequestTraceRequestEndState {
        request_tracker,
        replay_metrics,
        model: common_request.model.clone(),
        x_request_id,
    };

    Some(RequestEndTraceState {
        agent,
        request: Some(request),
        request_id: request_id.to_string(),
        request_context: context.context(),
    })
}

impl RequestEndTraceState {
    fn emit(&mut self) {
        let Some(request_state) = self.request.take() else {
            return;
        };
        if let Some(agent_state) = self.agent.take() {
            let (agent_context, mut metrics) =
                super::request_metrics_from_agent_state(agent_state, self.request_id.clone());
            metrics.replay = Some(super::into_owned_replay_metrics(
                request_state.replay_metrics,
            ));
            super::record::emit_agent_request_end(agent_context, metrics);
        } else {
            super::record::emit_request_end(
                self.request_id.clone(),
                request_state.x_request_id,
                request_state.model,
                &request_state.request_tracker,
                super::into_owned_replay_metrics(request_state.replay_metrics),
            );
        }
    }
}

impl Drop for RequestEndTraceState {
    fn drop(&mut self) {
        if self.request_context.is_killed() {
            if let Some(request) = &self.request {
                request.request_tracker.record_finish_if_missing();
            }
            self.emit();
        }
    }
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
) -> Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>
where
    Resp: Send + 'static,
{
    let Some(mut trace_state) = trace_state else {
        return stream;
    };

    let (stream, done) = crate::telemetry::stream::notify_on_completion(stream);
    tokio::spawn(async move {
        done.await;
        trace_state.emit();
    });
    stream
}

pub(crate) fn wrap_chat_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state);
    };

    let stream = stream.map(move |response| {
        super::record_chat_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state)
}

pub(crate) fn wrap_completion_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state);
    };

    let stream = stream.map(move |response| {
        super::record_completion_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state)
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
    use crate::protocols::common::extensions::{AgentContext, InputTrigger};
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use crate::request_trace::BUS;
    use crate::request_trace::RequestTraceEventSource;
    use crate::request_trace::{RequestRoutingHashes, RequestTraceEventType};
    use dynamo_kv_router::protocols::{
        KvCacheEventData, LocalBlockHash, WorkerWithDpRank, compute_seq_hash_for_block,
    };
    use dynamo_kv_router::zmq_wire::{BlockHashValue, RawKvEvent, convert_event};
    use std::sync::atomic::AtomicU32;

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

    fn request_end_state(
        request_id: &str,
        tracker: Arc<RequestTracker>,
    ) -> (RequestEndTraceState, Arc<dyn AsyncEngineContext>) {
        let context = Context::new(()).context();
        (
            RequestEndTraceState {
                agent: None,
                request: Some(RequestTraceRequestEndState {
                    request_tracker: tracker,
                    replay_metrics: Arc::new(RequestReplayMetrics {
                        trace_block_size: 2,
                        input_length: 2,
                        input_sequence_hashes: vec![11],
                    }),
                    model: "test-model".to_string(),
                    x_request_id: None,
                }),
                request_id: request_id.to_string(),
                request_context: context.clone(),
            },
            context,
        )
    }

    fn drain_request_records(
        receiver: &mut tokio::sync::broadcast::Receiver<crate::request_trace::RequestTraceRecord>,
        request_id: &str,
    ) -> Vec<crate::request_trace::RequestTraceRecord> {
        let mut records = Vec::new();
        loop {
            match receiver.try_recv() {
                Ok(record)
                    if record
                        .request
                        .as_ref()
                        .is_some_and(|request| request.request_id == request_id) =>
                {
                    records.push(record);
                }
                Ok(_) | Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                | Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            }
        }
        records
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

    #[test]
    fn cancellation_before_response_stream_emits_request_end() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(2, None);
        let (state, context) = request_end_state("req-pre-stream-cancel", tracker);

        context.kill();
        drop(state);

        let records = drain_request_records(&mut receiver, "req-pre-stream-cancel");
        assert_eq!(records.len(), 1);
        let record = &records[0];
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.input_tokens, Some(2));
        assert_eq!(request.output_tokens, Some(0));
    }

    #[test]
    fn agent_cancellation_finalizes_missing_timing_and_preserves_existing_finish() {
        BUS.init(64);
        let mut receiver = BUS.subscribe();
        for already_finished in [false, true] {
            let request_id = format!("req-agent-cancel-{already_finished}");
            let tracker = Arc::new(RequestTracker::new());
            let (mut state, context) = request_end_state(&request_id, tracker.clone());
            state.agent = Some(AgentContextTraceState {
                agent_context: AgentContext {
                    session_id: "cancel-timing".to_string(),
                    parent_session_id: None,
                    session_final: None,
                    agent_headers: Default::default(),
                    input_trigger: None,
                },
                request_model: "test-model".to_string(),
                request_tracker: Some(tracker.clone()),
                x_request_id: None,
                finish_reason_metadata: Default::default(),
                client_tool_call_ids: None,
            });
            if already_finished {
                tracker.record_finish();
            }
            let original_total = tracker.total_time_ms();
            std::thread::sleep(Duration::from_millis(20));
            context.kill();
            drop(state);

            let records = drain_request_records(&mut receiver, &request_id);
            assert_eq!(records.len(), 1);
            let record = &records[0];
            let request = record.request.as_ref().unwrap();
            let total = request.total_time_ms.expect("cancellation finish timing");
            if let Some(original_total) = original_total {
                assert_eq!(total, original_total);
            } else {
                assert!(total >= 20.0);
            }
            assert_eq!(
                record.event_time_unix_ms,
                request.request_received_ms.unwrap() + total.round() as u64
            );
        }
    }

    #[test]
    fn early_error_without_cancellation_does_not_emit_request_end() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let (state, _context) =
            request_end_state("req-pre-stream-error", Arc::new(RequestTracker::new()));

        drop(state);

        assert!(drain_request_records(&mut receiver, "req-pre-stream-error").is_empty());
    }

    #[test]
    fn explicit_emit_followed_by_drop_emits_once() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let (mut state, context) =
            request_end_state("req-emit-once", Arc::new(RequestTracker::new()));

        state.emit();
        context.kill();
        drop(state);

        assert_eq!(
            drain_request_records(&mut receiver, "req-emit-once").len(),
            1
        );
    }

    #[tokio::test]
    async fn cancellation_after_response_stream_reads_tracker_after_inner_drop() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let (state, _context) = request_end_state("req-drop", tracker.clone());
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped = wrap_request_end_stream(Box::pin(stream), Some(state));
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
        let mut request = preprocessed_request(SamplingOptions::default());
        request.agent_context = Some(AgentContext {
            session_id: "root".to_string(),
            parent_session_id: None,
            session_final: None,
            agent_headers: std::collections::BTreeMap::from([(
                "x-codex-turn-metadata".into(),
                vec![r#"{"request_kind":"compaction","future":true}"#.into()],
            )])
            .into(),
            input_trigger: Some(InputTrigger::ToolResult),
        });
        let mut context = Context::new(());
        context.insert(
            crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
            "llm-call-1".to_string(),
        );
        let expected_request_id = context.id().to_string();
        let state = start_request_trace_for_policy(
            &request,
            &Some(tracker.clone()),
            &context,
            2,
            false,
            END_ONLY,
        )
        .unwrap();
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped = wrap_request_end_stream(Box::pin(stream), Some(state));
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == expected_request_id)
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(record.event_source, Some(RequestTraceEventSource::Dynamo));
        let agent_context = record.agent_context.as_ref().expect("agent context");
        assert_eq!(agent_context.session_id, "root");
        assert_eq!(agent_context.input_trigger, Some(InputTrigger::ToolResult));
        assert_eq!(
            agent_context.agent_headers["x-codex-turn-metadata"],
            [r#"{"request_kind":"compaction","future":true}"#]
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
            3
        );
    }

    const END_ONLY: RequestRecords = RequestRecords {
        start: false,
        end: true,
    };
    const START_AND_END: RequestRecords = RequestRecords {
        start: true,
        end: true,
    };
    const START_ONLY: RequestRecords = RequestRecords {
        start: true,
        end: false,
    };

    fn request_with_tokens(token_ids: Vec<u32>) -> PreprocessedRequest {
        let mut request = preprocessed_request(SamplingOptions::default());
        request.token_ids = Arc::new(token_ids);
        request
    }

    fn routing_hashes(record: &crate::request_trace::RequestTraceRecord) -> &RequestRoutingHashes {
        record
            .request
            .as_ref()
            .and_then(|request| request.routing_hashes.as_ref())
            .expect("request_start routing hashes")
    }

    /// `request_start` is published while the request is preprocessed, before
    /// anything is dispatched, and its hashes chain to the `request_end` hashes.
    #[test]
    fn request_start_is_published_before_dispatch_and_chains_to_request_end() {
        BUS.init(64);
        let mut receiver = BUS.subscribe();
        // 7 tokens at block size 2: three full blocks and a partial one, then an
        // exact multiple, where every request_end hash is a full block.
        for (token_count, full_blocks) in [(7_u32, 3_usize), (8, 4)] {
            let tracker = Arc::new(RequestTracker::new());
            let request = request_with_tokens((1..=token_count).collect());
            let mut context = Context::new(());
            context.insert(
                crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
                "client-req-2".to_string(),
            );
            let request_id = context.id().to_string();

            let mut state = start_request_trace_for_policy(
                &request,
                &Some(tracker.clone()),
                &context,
                2,
                false,
                START_AND_END,
            )
            .unwrap();

            // Only request_start exists before the response stream completes.
            let start_records = drain_request_records(&mut receiver, &request_id);
            assert_eq!(start_records.len(), 1);
            let start = &start_records[0];
            assert_eq!(start.event_type, RequestTraceEventType::RequestStart);

            tracker.record_finish();
            state.emit();
            let end_records = drain_request_records(&mut receiver, &request_id);
            assert_eq!(end_records.len(), 1);
            let end = &end_records[0];
            assert_eq!(end.event_type, RequestTraceEventType::RequestEnd);

            let hashes = routing_hashes(start);
            assert_eq!(hashes.block_size, 2);
            assert_eq!(hashes.local_block_hashes.len(), full_blocks);
            let local = hashes
                .local_block_hashes
                .iter()
                .map(|hash| LocalBlockHash(*hash))
                .collect::<Vec<_>>();
            assert_eq!(compute_seq_hash_for_block(&local), hashes.sequence_hashes);
            let end_hashes = &end
                .request
                .as_ref()
                .unwrap()
                .replay
                .as_ref()
                .unwrap()
                .input_sequence_hashes;
            assert_eq!(hashes.sequence_hashes, end_hashes[..full_blocks]);

            let start_request = start.request.as_ref().unwrap();
            let end_request = end.request.as_ref().unwrap();
            assert_eq!(start_request.x_request_id.as_deref(), Some("client-req-2"));
            assert_eq!(end_request.x_request_id.as_deref(), Some("client-req-2"));
            assert_eq!(start_request.model.as_deref(), Some("test-model"));
            assert_eq!(start_request.input_tokens, Some(u64::from(token_count)));
            assert_eq!(
                start_request.request_received_ms,
                end_request.request_received_ms
            );
            assert!(start_request.replay.is_none());
            assert!(end_request.routing_hashes.is_none());
        }
    }

    #[test]
    fn request_start_alone_publishes_without_request_end_state() {
        BUS.init(64);
        let mut receiver = BUS.subscribe();
        let request = request_with_tokens(vec![1, 2, 3, 4]);
        let context = Context::new(());
        let request_id = context.id().to_string();

        let state = start_request_trace_for_policy(
            &request,
            &Some(Arc::new(RequestTracker::new())),
            &context,
            2,
            false,
            START_ONLY,
        );

        assert!(state.is_none());
        let records = drain_request_records(&mut receiver, &request_id);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].event_type, RequestTraceEventType::RequestStart);
    }

    #[test]
    fn request_start_follows_request_trace_eligibility() {
        BUS.init(64);
        let mut receiver = BUS.subscribe();
        let request = preprocessed_request(SamplingOptions {
            n: Some(2),
            ..Default::default()
        });
        let context = Context::new(());
        let request_id = context.id().to_string();

        let state = start_request_trace_for_policy(
            &request,
            &Some(Arc::new(RequestTracker::new())),
            &context,
            2,
            false,
            START_AND_END,
        );

        assert!(state.is_none());
        assert!(drain_request_records(&mut receiver, &request_id).is_empty());
    }

    /// The keys the engine's KV events produce for this prompt, after the KV
    /// event publisher converts them: one `tokens_hash` per stored block.
    fn engine_event_keys(
        token_ids: &[u32],
        block_size: usize,
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
    ) -> Vec<LocalBlockHash> {
        let full_tokens = token_ids.len() / block_size * block_size;
        let raw = RawKvEvent::BlockStored {
            block_hashes: (0..full_tokens / block_size)
                .map(|index| BlockHashValue::Unsigned(1_000 + index as u64))
                .collect(),
            parent_block_hash: None,
            token_ids: token_ids[..full_tokens].to_vec(),
            block_size,
            medium: None,
            lora_name: lora_name.map(str::to_string),
            cache_namespace: cache_namespace.map(str::to_string),
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
            session_id: None,
        };
        let event = convert_event(
            raw,
            1,
            block_size as u32,
            WorkerWithDpRank::from_worker_id(1),
            &Arc::new(AtomicU32::new(0)),
            None,
            None,
        )
        .expect("stored event");
        match event.event.data {
            KvCacheEventData::Stored(stored) => stored
                .blocks
                .iter()
                .map(|block| block.tokens_hash)
                .collect(),
            other => panic!("expected a stored event, got {other:?}"),
        }
    }

    /// The router hashes a salted or LoRA request with that salt or LoRA name,
    /// and so does the engine. `request_start` must carry those keys; the
    /// default-option `request_end` hashes do not match the engine's.
    #[test]
    fn request_start_hashes_match_engine_keys_for_salted_and_lora_requests() {
        BUS.init(64);
        let mut receiver = BUS.subscribe();
        let token_ids = (1..=9_u32).collect::<Vec<_>>();
        for (lora_name, cache_namespace) in [
            (None, None),
            (None, Some("tenant-salt")),
            (Some("adapter-a"), None),
            (Some("adapter-a"), Some("tenant-salt")),
        ] {
            let mut request = request_with_tokens(token_ids.clone());
            if lora_name.is_some() || cache_namespace.is_some() {
                let routing = request.routing_mut();
                routing.lora_name = lora_name.map(str::to_string);
                routing.cache_namespace = cache_namespace.map(str::to_string);
            }
            let context = Context::new(());
            let request_id = context.id().to_string();

            let mut state = start_request_trace_for_policy(
                &request,
                &Some(Arc::new(RequestTracker::new())),
                &context,
                4,
                false,
                START_AND_END,
            )
            .unwrap();
            state.emit();
            let records = drain_request_records(&mut receiver, &request_id);
            assert_eq!(records.len(), 2);
            let hashes = routing_hashes(&records[0]);
            let end_hashes = records[1]
                .request
                .as_ref()
                .unwrap()
                .replay
                .as_ref()
                .unwrap()
                .input_sequence_hashes
                .clone();

            let engine_keys = engine_event_keys(&token_ids, 4, lora_name, cache_namespace);
            assert_eq!(engine_keys.len(), 2);
            assert_eq!(
                hashes.local_block_hashes,
                engine_keys.iter().map(|hash| hash.0).collect::<Vec<_>>(),
                "lora={lora_name:?} salt={cache_namespace:?}"
            );
            assert_eq!(
                hashes.sequence_hashes,
                compute_seq_hash_for_block(&engine_keys)
            );
            // request_end keeps its default-option hashes, which join the
            // engine's keys only for a plain request.
            let plain = lora_name.is_none() && cache_namespace.is_none();
            assert_eq!(hashes.sequence_hashes == end_hashes[..2], plain);
        }
    }

    /// The trace bus is bounded in records, so `request_start` adds one record
    /// per request; its size grows with the prompt. Prints the serialized size
    /// at a 128k-token prompt with 16-token blocks.
    #[test]
    fn request_start_record_size_at_long_isl() {
        let token_ids = (0..131_072_u32).collect::<Vec<_>>();
        let hashes = super::super::routing_hashes(&token_ids, 16, BlockHashOptions::default());
        assert_eq!(hashes.local_block_hashes.len(), 8_192);
        assert_eq!(hashes.sequence_hashes.len(), 8_192);
        let record = super::super::record::request_start_record(
            "req-long".to_string(),
            Some("client-long".to_string()),
            "test-model".to_string(),
            &RequestTracker::new(),
            token_ids.len(),
            hashes,
        );
        let json_bytes = serde_json::to_vec(&record).unwrap().len();
        let hash_bytes = 2 * 8_192 * std::mem::size_of::<u64>();
        eprintln!(
            "request_start at 131072 tokens, block size 16: {hash_bytes} bytes of hashes in memory, {json_bytes} bytes as JSON"
        );
        assert!(json_bytes > hash_bytes);
    }

    /// The frontend mints its own `request_id`, so the client's `x-request-id`
    /// is the only key a client can join on. A plain request must carry it.
    #[tokio::test]
    async fn plain_request_end_carries_x_request_id_and_model() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let request = preprocessed_request(SamplingOptions::default());
        let mut context = Context::new(());
        context.insert(
            crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
            "client-req-1".to_string(),
        );
        let request_id = context.id().to_string();
        let mut state = start_request_trace_for_policy(
            &request,
            &Some(tracker.clone()),
            &context,
            2,
            false,
            END_ONLY,
        )
        .unwrap();
        tracker.record_finish();
        state.emit();

        let records = drain_request_records(&mut receiver, &request_id);
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert!(record.agent_context.is_none());
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.x_request_id.as_deref(), Some("client-req-1"));
        assert_eq!(request.model.as_deref(), Some("test-model"));
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
            agent_headers: Default::default(),
            input_trigger: None,
        });
        let tracker = Some(Arc::new(RequestTracker::new()));
        let context = Context::new(());

        let state =
            start_request_trace_for_policy(&request, &tracker, &context, 2, false, END_ONLY);

        assert!(state.is_none());
    }
}

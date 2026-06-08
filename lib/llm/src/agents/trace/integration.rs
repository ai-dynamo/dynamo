// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo LLM integration helpers for agent trace records.

use std::collections::HashMap;
#[cfg(test)]
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::pipeline::Context;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::transports::event_plane::EventSubscriber;
#[cfg(test)]
use futures::{Stream, StreamExt};
use parking_lot::Mutex;

use crate::agents::context::AgentContext;
use crate::agents::trace::{
    AgentReplayMetrics, AgentRequestMetrics, AgentToolEventRelay, AgentTracePolicy,
    AgentTraceRecord, DEFAULT_TOOL_EVENTS_TOPIC, FinishReasonMetadata, ToolCallMetadata,
    WorkerInfo,
};
use crate::local_model::LocalModel;
use crate::protocols::common::FinishReason as BackendFinishReason;
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionStreamResponse, completions::NvCreateCompletionResponse,
};

#[derive(Clone, Debug, Default)]
pub struct SharedFinishReasonMetadata {
    state: Arc<Mutex<FinishReasonMetadataState>>,
}

#[derive(Debug, Default)]
struct FinishReasonMetadataState {
    metadata: FinishReasonMetadata,
    pending_tool_calls: HashMap<(u32, u32), PendingToolCallMetadata>,
    tool_call_positions: HashMap<(u32, u32), usize>,
}

#[derive(Debug, Default)]
struct PendingToolCallMetadata {
    id: Option<String>,
    name: Option<String>,
}

impl SharedFinishReasonMetadata {
    fn lock(&self) -> parking_lot::MutexGuard<'_, FinishReasonMetadataState> {
        self.state.lock()
    }

    #[cfg(feature = "agent-trace-bench")]
    #[doc(hidden)]
    pub fn record_tool_call_chunk_for_bench(
        &self,
        choice_index: u32,
        tool_call_index: u32,
        id: Option<&str>,
        name: Option<&str>,
    ) {
        self.lock()
            .record_tool_call_chunk(choice_index, tool_call_index, id, name);
    }

    #[cfg(feature = "agent-trace-bench")]
    #[doc(hidden)]
    pub fn snapshot_for_bench(&self) -> Option<FinishReasonMetadata> {
        self.lock().snapshot()
    }
}

impl FinishReasonMetadataState {
    fn record_backend_finish_reason(
        &mut self,
        choice_index: Option<u32>,
        backend_finish_reason: Option<String>,
        stop_reason: Option<dynamo_protocols::types::StopReason>,
    ) {
        if let Some(backend_finish_reason) = backend_finish_reason.as_ref() {
            self.metadata.backend_finish_reason = Some(backend_finish_reason.clone());
        }
        if let Some(stop_reason) = stop_reason.as_ref() {
            self.metadata.stop_reason = Some(stop_reason.clone());
        }
        if let Some(choice_index) = choice_index {
            self.metadata.record_choice_backend_finish_reason(
                choice_index,
                backend_finish_reason,
                stop_reason,
            );
        }
    }

    fn record_choice_finish_reason(
        &mut self,
        choice_index: u32,
        finish_reason: dynamo_protocols::types::FinishReason,
    ) {
        self.metadata.finish_reason = Some(finish_reason);
        self.metadata
            .record_choice_finish_reason(choice_index, finish_reason);
    }

    fn record_tool_call_chunk(
        &mut self,
        choice_index: u32,
        tool_call_index: u32,
        id: Option<&str>,
        name: Option<&str>,
    ) {
        if id.is_none() && name.is_none() {
            return;
        }

        let pending = self
            .pending_tool_calls
            .entry((choice_index, tool_call_index))
            .or_default();
        let mut changed = false;
        if let Some(id) = id
            && pending.id.as_deref() != Some(id)
        {
            pending.id = Some(id.to_string());
            changed = true;
        }
        if let Some(name) = name
            && pending.name.as_deref() != Some(name)
        {
            pending.name = Some(name.to_string());
            changed = true;
        }

        if !changed {
            return;
        }

        let key = (choice_index, tool_call_index);
        if let Some(position) = self.tool_call_positions.get(&key).copied() {
            let existing = &mut self.metadata.tool_calls[position];
            if existing.id.is_none() {
                existing.id = pending.id.clone();
            }
            if existing.name.is_none() {
                existing.name = pending.name.clone();
            }
        } else {
            let position = self.metadata.tool_calls.len();
            self.tool_call_positions.insert(key, position);
            self.metadata.tool_calls.push(ToolCallMetadata {
                choice_index,
                tool_call_index,
                id: pending.id.clone(),
                name: pending.name.clone(),
            });
        }
    }

    fn snapshot(&self) -> Option<FinishReasonMetadata> {
        (!self.metadata.is_empty()).then_some(self.metadata.clone())
    }
}

/// Record token counts needed by agent trace request-end records.
///
/// Callers should gate this once per request so the response path only pays a
/// cheap boolean branch for untraced requests.
pub(crate) fn record_llm_metric_tokens(
    tracker: Option<&RequestTracker>,
    input_tokens: Option<usize>,
    output_tokens: usize,
    cached_tokens: Option<usize>,
) {
    let Some(tracker) = tracker else {
        return;
    };

    // Usage-derived token counts arrive late in the response path. Earlier
    // router-side observations still win because RequestTracker stores them
    // with OnceLock.
    if input_tokens.is_some() || cached_tokens.is_some() {
        tracker.record_isl(input_tokens.unwrap_or(0), cached_tokens);
    }
    tracker.record_osl(output_tokens);
}

static TOOL_EVENT_INGEST_STARTED: AtomicBool = AtomicBool::new(false);
static TOOL_EVENT_RELAY_STARTED: AtomicBool = AtomicBool::new(false);

pub(crate) fn request_metrics(
    request_id: String,
    x_request_id: Option<String>,
    model: String,
    tracker: Option<&RequestTracker>,
) -> AgentRequestMetrics {
    let timing = tracker.map(RequestTracker::get_timing_info);
    let worker = tracker.and_then(|tracker| {
        tracker.get_worker_info().map(|worker| WorkerInfo {
            prefill_worker_id: worker.prefill_worker_id,
            prefill_dp_rank: worker.prefill_dp_rank,
            decode_worker_id: worker.decode_worker_id,
            decode_dp_rank: worker.decode_dp_rank,
        })
    });

    AgentRequestMetrics {
        request_id,
        x_request_id,
        model,
        input_tokens: tracker.and_then(|tracker| tracker.isl_tokens().map(|v| v as u64)),
        output_tokens: tracker.map(RequestTracker::osl_tokens),
        cached_tokens: tracker.and_then(|tracker| tracker.cached_tokens().map(|v| v as u64)),
        request_received_ms: timing.as_ref().map(|timing| timing.request_received_ms),
        prefill_wait_time_ms: timing
            .as_ref()
            .and_then(|timing| timing.prefill_wait_time_ms),
        prefill_time_ms: timing.as_ref().and_then(|timing| timing.prefill_time_ms),
        ttft_ms: timing.as_ref().and_then(|timing| timing.ttft_ms),
        total_time_ms: timing.as_ref().and_then(|timing| timing.total_time_ms),
        avg_itl_ms: tracker.and_then(RequestTracker::avg_itl_ms),
        kv_hit_rate: timing.as_ref().and_then(|timing| timing.kv_hit_rate),
        kv_transfer_estimated_latency_ms: timing
            .as_ref()
            .and_then(|timing| timing.kv_transfer_estimated_latency_ms),
        queue_depth: timing
            .as_ref()
            .and_then(|timing| timing.router_queue_depth.map(|v| v as u64)),
        worker,
        replay: None,
        finish_reason_metadata: None,
    }
}

pub(crate) struct AgentTraceRequestEndState {
    pub agent_context: AgentContext,
    pub request_model: String,
    pub request_tracker: Option<Arc<RequestTracker>>,
    pub x_request_id: Option<String>,
    pub replay_metrics: Option<Arc<AgentReplayMetrics>>,
    pub finish_reason_metadata: SharedFinishReasonMetadata,
}

pub(crate) fn build_agent_trace_request_end_state(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    replay_metrics: Option<Arc<AgentReplayMetrics>>,
) -> Option<AgentTraceRequestEndState> {
    if !super::is_enabled() {
        return None;
    }
    let agent_context = common_request.agent_context.clone()?;
    let x_request_id = dynamo_runtime::logging::get_distributed_tracing_context()
        .and_then(|c| c.x_request_id)
        .or_else(|| {
            context
                .get::<String>(super::X_REQUEST_ID_CONTEXT_KEY)
                .ok()
                .map(|v| v.as_ref().clone())
        });
    Some(AgentTraceRequestEndState {
        agent_context,
        request_model: common_request.model.clone(),
        request_tracker: tracker.clone(),
        x_request_id,
        replay_metrics,
        finish_reason_metadata: SharedFinishReasonMetadata::default(),
    })
}

#[cfg(test)]
pub(crate) fn finish_reason_metadata_handle(
    trace_state: &Option<AgentTraceRequestEndState>,
) -> Option<SharedFinishReasonMetadata> {
    trace_state
        .as_ref()
        .map(|state| state.finish_reason_metadata.clone())
}

pub(crate) fn record_backend_finish_reason_metadata(
    finish_reason_metadata: Option<&SharedFinishReasonMetadata>,
    choice_index: Option<u32>,
    finish_reason: Option<&BackendFinishReason>,
    stop_reason: Option<&dynamo_protocols::types::StopReason>,
) {
    if finish_reason.is_none() && stop_reason.is_none() {
        return;
    }
    let Some(finish_reason_metadata) = finish_reason_metadata else {
        return;
    };

    finish_reason_metadata.lock().record_backend_finish_reason(
        choice_index,
        finish_reason.map(ToString::to_string),
        stop_reason.cloned(),
    );
}

pub(crate) fn record_chat_finish_reason_metadata(
    finish_reason_metadata: &SharedFinishReasonMetadata,
    response: &Annotated<NvCreateChatCompletionStreamResponse>,
) {
    let Some(data) = response.data.as_ref() else {
        return;
    };

    let mut metadata = finish_reason_metadata.lock();
    for choice in &data.inner.choices {
        if let Some(finish_reason) = choice.finish_reason.as_ref() {
            metadata.record_choice_finish_reason(choice.index, *finish_reason);
        }

        let Some(tool_calls) = choice.delta.tool_calls.as_ref() else {
            continue;
        };
        for tool_call in tool_calls {
            let function = tool_call.function.as_ref();
            metadata.record_tool_call_chunk(
                choice.index,
                tool_call.index,
                tool_call.id.as_deref(),
                function.and_then(|function| function.name.as_deref()),
            );
        }
    }
}

fn completion_finish_reason_to_finish_reason(
    finish_reason: dynamo_protocols::types::CompletionFinishReason,
) -> dynamo_protocols::types::FinishReason {
    match finish_reason {
        dynamo_protocols::types::CompletionFinishReason::Stop => {
            dynamo_protocols::types::FinishReason::Stop
        }
        dynamo_protocols::types::CompletionFinishReason::Length => {
            dynamo_protocols::types::FinishReason::Length
        }
        dynamo_protocols::types::CompletionFinishReason::ContentFilter => {
            dynamo_protocols::types::FinishReason::ContentFilter
        }
    }
}

pub(crate) fn record_completion_finish_reason_metadata(
    finish_reason_metadata: &SharedFinishReasonMetadata,
    response: &Annotated<NvCreateCompletionResponse>,
) {
    let Some(data) = response.data.as_ref() else {
        return;
    };

    let mut metadata = finish_reason_metadata.lock();
    for choice in &data.inner.choices {
        if let Some(finish_reason) = choice.finish_reason {
            metadata.record_choice_finish_reason(
                choice.index,
                completion_finish_reason_to_finish_reason(finish_reason),
            );
        }
    }
}

fn snapshot_finish_reason_metadata(
    finish_reason_metadata: &SharedFinishReasonMetadata,
) -> Option<FinishReasonMetadata> {
    finish_reason_metadata.lock().snapshot()
}

pub(crate) fn emit_agent_trace_request_end(
    trace_state: AgentTraceRequestEndState,
    request_id: String,
) {
    let AgentTraceRequestEndState {
        agent_context,
        request_model,
        request_tracker,
        x_request_id,
        replay_metrics,
        finish_reason_metadata,
    } = trace_state;

    if request_tracker.is_none() {
        tracing::warn!(
            request_id,
            "agent_context present but request tracker is missing; emitting partial trace"
        );
    }
    let mut metrics = super::request_metrics(
        request_id,
        x_request_id,
        request_model,
        request_tracker.as_deref(),
    );
    metrics.replay = replay_metrics.map(into_owned_replay_metrics);
    metrics.finish_reason_metadata = snapshot_finish_reason_metadata(&finish_reason_metadata);
    super::emit_request_end(agent_context, metrics);
}

pub(crate) fn into_owned_replay_metrics(
    replay_metrics: Arc<AgentReplayMetrics>,
) -> AgentReplayMetrics {
    Arc::try_unwrap(replay_metrics).unwrap_or_else(|shared| shared.as_ref().clone())
}

#[cfg(test)]
pub(crate) fn wrap_agent_trace_request_end_stream<Resp>(
    stream: Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>,
    trace_state: Option<AgentTraceRequestEndState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>
where
    Resp: Send + 'static,
{
    let Some(trace_state) = trace_state else {
        return stream;
    };

    let (stream, done_fut) = crate::telemetry::stream::notify_on_completion(stream);
    tokio::spawn(async move {
        done_fut.await;
        emit_agent_trace_request_end(trace_state, request_id);
    });
    stream
}

#[cfg(test)]
pub(crate) fn wrap_agent_trace_chat_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    trace_state: Option<AgentTraceRequestEndState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_agent_trace_request_end_stream(stream, trace_state, request_id);
    };

    let stream = stream.map(move |response| {
        record_chat_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_agent_trace_request_end_stream(Box::pin(stream), trace_state, request_id)
}

#[cfg(test)]
pub(crate) fn wrap_agent_trace_completion_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>>,
    trace_state: Option<AgentTraceRequestEndState>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_agent_trace_request_end_stream(stream, trace_state, request_id);
    };

    let stream = stream.map(move |response| {
        record_completion_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_agent_trace_request_end_stream(Box::pin(stream), trace_state, request_id)
}

pub(crate) async fn start_tool_event_ingest_from_policy(
    drt: DistributedRuntime,
    local_model: &LocalModel,
) -> anyhow::Result<()> {
    let policy = super::policy();
    if policy.tool_events_zmq_endpoint.is_none() {
        return Ok(());
    }

    start_tool_event_relay_from_policy(drt.clone(), local_model, policy).await?;

    if TOOL_EVENT_INGEST_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        tracing::debug!("agent tool event ingest already started");
        return Ok(());
    }

    let namespace_name = tool_events_namespace(local_model);
    let mut subscriber = match async {
        let namespace = drt.namespace(namespace_name.clone())?;
        EventSubscriber::for_namespace(&namespace, DEFAULT_TOOL_EVENTS_TOPIC)
            .await
            .map(|sub| sub.typed::<AgentTraceRecord>())
    }
    .await
    {
        Ok(subscriber) => subscriber,
        Err(error) => {
            TOOL_EVENT_INGEST_STARTED.store(false, Ordering::Release);
            return Err(error);
        }
    };

    let shutdown = drt.child_token();
    drt.runtime().secondary().spawn(async move {
        tracing::info!(
            namespace = %namespace_name,
            topic = DEFAULT_TOOL_EVENTS_TOPIC,
            "Agent tool event ingest started"
        );
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    tracing::debug!("agent tool event ingest stopping");
                    break;
                }
                next = subscriber.next() => {
                    match next {
                        Some(Ok((_envelope, record))) => {
                            super::publish_tool_record(record);
                        }
                        Some(Err(error)) => {
                            tracing::warn!(%error, "agent tool event ingest failed to decode event");
                        }
                        None => {
                            tracing::warn!("agent tool event ingest stream ended");
                            break;
                        }
                    }
                }
            }
        }
        TOOL_EVENT_INGEST_STARTED.store(false, Ordering::Release);
        tracing::info!("agent tool event ingest stopped");
    });

    Ok(())
}

async fn start_tool_event_relay_from_policy(
    drt: DistributedRuntime,
    local_model: &LocalModel,
    policy: &AgentTracePolicy,
) -> anyhow::Result<()> {
    let Some(zmq_endpoint) = policy.tool_events_zmq_endpoint.clone() else {
        return Ok(());
    };
    if TOOL_EVENT_RELAY_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        tracing::debug!("agent tool event relay already started");
        return Ok(());
    }

    let namespace_name = tool_events_namespace(local_model);
    let relay = match async {
        let namespace = drt.namespace(namespace_name.clone())?;
        let component = namespace.component(local_model.endpoint_id().component.clone())?;
        AgentToolEventRelay::start(
            component,
            zmq_endpoint.clone(),
            policy.tool_events_zmq_topic.clone(),
            Some(namespace_name.clone()),
            Some(DEFAULT_TOOL_EVENTS_TOPIC.to_string()),
        )
        .await
    }
    .await
    {
        Ok(relay) => relay,
        Err(error) => {
            TOOL_EVENT_RELAY_STARTED.store(false, Ordering::Release);
            return Err(error);
        }
    };
    let shutdown = drt.child_token();
    drt.runtime().secondary().spawn(async move {
        tracing::info!(
            namespace = %namespace_name,
            topic = DEFAULT_TOOL_EVENTS_TOPIC,
            zmq_endpoint = %zmq_endpoint,
            "Agent tool event relay started"
        );
        shutdown.cancelled().await;
        relay.shutdown();
        TOOL_EVENT_RELAY_STARTED.store(false, Ordering::Release);
        tracing::info!("agent tool event relay stopped");
    });

    Ok(())
}

fn tool_events_namespace(local_model: &LocalModel) -> String {
    local_model
        .namespace()
        .map(str::to_string)
        .unwrap_or_else(|| local_model.endpoint_id().namespace.clone())
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;

    use crate::agents::context::AgentContext;
    use crate::agents::trace::TraceEventType;
    use crate::protocols::common::{
        self,
        timing::{RequestTracker, WORKER_TYPE_DECODE},
    };
    use crate::protocols::openai::{
        chat_completions::NvCreateChatCompletionStreamResponse,
        completions::NvCreateCompletionResponse,
    };
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
        Choice, CompletionFinishReason, CreateChatCompletionStreamResponse,
        CreateCompletionResponse, FinishReason, FunctionCallStream, StopReason,
    };

    use super::{
        AgentTraceRequestEndState, SharedFinishReasonMetadata,
        record_backend_finish_reason_metadata, request_metrics,
        wrap_agent_trace_chat_request_end_stream, wrap_agent_trace_completion_request_end_stream,
    };

    #[test]
    fn test_request_metrics_from_tracker() {
        let tracker = RequestTracker::new();
        tracker.record_isl(128, Some(32));
        tracker.record_kv_hit(4.0, 8);
        tracker.record_osl(5);
        tracker.record_router_queue_depth(3);
        tracker.record_worker(17, Some(2), WORKER_TYPE_DECODE);
        tracker.record_prefill_start();
        thread::sleep(Duration::from_millis(1));
        tracker.record_first_token();
        tracker.record_prefill_complete();
        thread::sleep(Duration::from_millis(1));
        tracker.record_decode_first_token();
        thread::sleep(Duration::from_millis(1));
        tracker.record_finish();

        let metrics = request_metrics(
            "req-1".to_string(),
            Some("llm-call-1".to_string()),
            "test-model".to_string(),
            Some(&tracker),
        );

        assert_eq!(metrics.request_id, "req-1");
        assert_eq!(metrics.x_request_id.as_deref(), Some("llm-call-1"));
        assert_eq!(metrics.model, "test-model");
        assert_eq!(metrics.input_tokens, Some(128));
        assert_eq!(metrics.output_tokens, Some(5));
        assert_eq!(metrics.cached_tokens, Some(32));
        assert!(metrics.request_received_ms.is_some_and(|ms| ms > 0));
        assert!(metrics.prefill_wait_time_ms.is_some());
        assert!(metrics.prefill_time_ms.is_some());
        assert!(metrics.ttft_ms.is_some());
        assert!(metrics.total_time_ms.is_some());
        assert!(metrics.avg_itl_ms.is_some());
        assert_eq!(metrics.kv_hit_rate, Some(0.5));
        assert!(metrics.kv_transfer_estimated_latency_ms.is_some());
        assert_eq!(metrics.queue_depth, Some(3));
        assert!(metrics.finish_reason_metadata.is_none());
        let worker = metrics.worker.expect("worker info should be set");
        assert_eq!(worker.prefill_worker_id, Some(17));
        assert_eq!(worker.prefill_dp_rank, Some(2));
        assert_eq!(worker.decode_worker_id, Some(17));
        assert_eq!(worker.decode_dp_rank, Some(2));
    }

    #[test]
    fn test_request_metrics_without_tracker_is_partial() {
        let metrics = request_metrics(
            "req-1".to_string(),
            Some("llm-call-1".to_string()),
            "test-model".to_string(),
            None,
        );

        assert_eq!(metrics.request_id, "req-1");
        assert_eq!(metrics.x_request_id.as_deref(), Some("llm-call-1"));
        assert_eq!(metrics.model, "test-model");
        assert_eq!(metrics.input_tokens, None);
        assert_eq!(metrics.output_tokens, None);
        assert_eq!(metrics.cached_tokens, None);
        assert_eq!(metrics.request_received_ms, None);
        assert_eq!(metrics.prefill_wait_time_ms, None);
        assert_eq!(metrics.prefill_time_ms, None);
        assert_eq!(metrics.ttft_ms, None);
        assert_eq!(metrics.total_time_ms, None);
        assert_eq!(metrics.avg_itl_ms, None);
        assert_eq!(metrics.kv_hit_rate, None);
        assert_eq!(metrics.kv_transfer_estimated_latency_ms, None);
        assert_eq!(metrics.queue_depth, None);
        assert!(metrics.finish_reason_metadata.is_none());
        assert!(metrics.worker.is_none());
    }

    #[tokio::test]
    async fn test_chat_request_end_records_finish_reason_metadata() {
        super::super::BUS.init(16);
        let mut rx = super::super::BUS.subscribe();

        let finish_reason_metadata = SharedFinishReasonMetadata::default();
        record_backend_finish_reason_metadata(
            Some(&finish_reason_metadata),
            Some(0),
            Some(&common::FinishReason::Stop),
            Some(&StopReason::String("END".to_string())),
        );

        let trace_state = AgentTraceRequestEndState {
            agent_context: AgentContext {
                session_type_id: "ms_agent".to_string(),
                session_id: "run-finish".to_string(),
                trajectory_id: "run-finish:agent".to_string(),
                parent_trajectory_id: None,
            },
            request_model: "test-model".to_string(),
            request_tracker: None,
            x_request_id: Some("llm-call-1".to_string()),
            replay_metrics: None,
            finish_reason_metadata,
        };

        let stream = futures::stream::iter(vec![
            Annotated::from_data(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "chatcmpl-1".to_string(),
                    choices: vec![ChatChoiceStream {
                        index: 0,
                        delta: ChatCompletionStreamResponseDelta {
                            content: None,
                            function_call: None,
                            tool_calls: Some(vec![ChatCompletionMessageToolCallChunk {
                                index: 0,
                                id: Some("call-1".to_string()),
                                r#type: None,
                                function: Some(FunctionCallStream {
                                    name: Some("web_search".to_string()),
                                    arguments: None,
                                }),
                            }]),
                            role: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                    created: 0,
                    model: "test-model".to_string(),
                    service_tier: None,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                },
                nvext: None,
            }),
            Annotated::from_data(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "chatcmpl-1".to_string(),
                    choices: vec![ChatChoiceStream {
                        index: 0,
                        delta: ChatCompletionStreamResponseDelta {
                            content: None,
                            function_call: None,
                            tool_calls: None,
                            role: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: Some(FinishReason::ToolCalls),
                        logprobs: None,
                    }],
                    created: 0,
                    model: "test-model".to_string(),
                    service_tier: None,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage: None,
                },
                nvext: None,
            }),
        ]);

        let wrapped = wrap_agent_trace_chat_request_end_stream(
            Box::pin(stream),
            Some(trace_state),
            "req-finish".to_string(),
        );
        let responses: Vec<_> = wrapped.collect().await;
        assert_eq!(responses.len(), 2);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = rx.recv().await.expect("trace record should publish");
                if record.event_type == TraceEventType::RequestEnd
                    && record
                        .request
                        .as_ref()
                        .is_some_and(|request| request.request_id == "req-finish")
                {
                    break record;
                }
            }
        })
        .await
        .expect("trace record for req-finish should publish");
        let request = record.request.expect("request metrics should be present");
        let metadata = request
            .finish_reason_metadata
            .expect("finish metadata should be recorded");
        assert_eq!(metadata.backend_finish_reason.as_deref(), Some("stop"));
        assert_eq!(metadata.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(
            metadata.stop_reason,
            Some(StopReason::String("END".to_string()))
        );
        assert_eq!(metadata.tool_calls.len(), 1);
        assert_eq!(metadata.tool_calls[0].choice_index, 0);
        assert_eq!(metadata.tool_calls[0].tool_call_index, 0);
        assert_eq!(metadata.tool_calls[0].id.as_deref(), Some("call-1"));
        assert_eq!(metadata.tool_calls[0].name.as_deref(), Some("web_search"));
        assert_eq!(metadata.choices.len(), 1);
        assert_eq!(metadata.choices[0].choice_index, 0);
        assert_eq!(
            metadata.choices[0].backend_finish_reason.as_deref(),
            Some("stop")
        );
        assert_eq!(
            metadata.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_completion_request_end_records_finish_reason_metadata() {
        super::super::BUS.init(16);
        let mut rx = super::super::BUS.subscribe();

        let finish_reason_metadata = SharedFinishReasonMetadata::default();
        record_backend_finish_reason_metadata(
            Some(&finish_reason_metadata),
            Some(0),
            Some(&common::FinishReason::Stop),
            Some(&StopReason::String("END".to_string())),
        );

        let trace_state = AgentTraceRequestEndState {
            agent_context: AgentContext {
                session_type_id: "ms_agent".to_string(),
                session_id: "run-completion-finish".to_string(),
                trajectory_id: "run-completion-finish:agent".to_string(),
                parent_trajectory_id: None,
            },
            request_model: "test-model".to_string(),
            request_tracker: None,
            x_request_id: Some("completion-call-1".to_string()),
            replay_metrics: None,
            finish_reason_metadata,
        };

        let stream =
            futures::stream::iter(vec![Annotated::from_data(NvCreateCompletionResponse {
                inner: CreateCompletionResponse {
                    id: "cmpl-1".to_string(),
                    object: "text_completion".to_string(),
                    created: 0,
                    model: "test-model".to_string(),
                    system_fingerprint: None,
                    choices: vec![Choice {
                        text: "".to_string(),
                        index: 0,
                        logprobs: None,
                        finish_reason: Some(CompletionFinishReason::Length),
                    }],
                    usage: None,
                },
                nvext: None,
            })]);

        let wrapped = wrap_agent_trace_completion_request_end_stream(
            Box::pin(stream),
            Some(trace_state),
            "req-completion-finish".to_string(),
        );
        let responses: Vec<_> = wrapped.collect().await;
        assert_eq!(responses.len(), 1);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = rx.recv().await.expect("trace record should publish");
                if record.event_type == TraceEventType::RequestEnd
                    && record
                        .request
                        .as_ref()
                        .is_some_and(|request| request.request_id == "req-completion-finish")
                {
                    break record;
                }
            }
        })
        .await
        .expect("trace record for req-completion-finish should publish");
        let request = record.request.expect("request metrics should be present");
        let metadata = request
            .finish_reason_metadata
            .expect("finish metadata should be recorded");
        assert_eq!(metadata.backend_finish_reason.as_deref(), Some("stop"));
        assert_eq!(metadata.finish_reason, Some(FinishReason::Length));
        assert_eq!(
            metadata.stop_reason,
            Some(StopReason::String("END".to_string()))
        );
        assert!(metadata.tool_calls.is_empty());
        assert_eq!(metadata.choices.len(), 1);
        assert_eq!(metadata.choices[0].choice_index, 0);
        assert_eq!(
            metadata.choices[0].backend_finish_reason.as_deref(),
            Some("stop")
        );
        assert_eq!(
            metadata.choices[0].finish_reason,
            Some(FinishReason::Length)
        );
    }
}

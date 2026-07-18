// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_kv_router::protocols::{TokensWithHashes, WorkerWithDpRank};
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        ResponseStream, SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tracing::Instrument;

use crate::{
    kv_router::{KvRouter, metrics::RouterRequestMetrics},
    preprocessor::PreprocessedRequest,
    protocols::common::{
        FinishReason,
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RoutingData},
    },
    session_affinity::{
        AffinityAcquire, AffinityCoordinator, AffinityTarget, affinity_id, explicit_target,
    },
};

mod cancellation;
mod request_guard;
mod selection;

use cancellation::cancel_on_stop;
use request_guard::RequestGuard;
use selection::{RoutingRequestParts, SelectionOptions, WorkerSelection};

const OUTPUT_REPLAY_ID_ANNOTATION_KEY: &str = "output_replay_id";
const OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY: &str = "output_replay_consumer";

fn is_cancelled(error: &Error) -> bool {
    match_error_chain(error.as_ref(), &[ErrorType::Cancelled], &[])
}

fn invalidate_on_non_cancellation(operation: &mut Option<AffinityAcquire>, error: &Error) {
    if is_cancelled(error) {
        return;
    }
    if let Some(operation) = operation.take() {
        operation.invalidate();
    }
}

fn monitor_response_stream(
    mut response_stream: ManyOut<Annotated<LLMEngineOutput>>,
    context: Arc<dyn AsyncEngineContext>,
    context_id: String,
    mut guard: RequestGuard,
) -> impl futures::Stream<Item = Annotated<LLMEngineOutput>> + Send {
    async_stream::stream! {
        // Keep one cancellation future alive for the whole response stream. Calling
        // `stopped()` for every item repeatedly clones and polls a watch receiver.
        let stopped = context.stopped();
        tokio::pin!(stopped);

        let mut failed = false;
        let completed = loop {
            tokio::select! {
                biased;

                _ = &mut stopped => {
                    tracing::debug!("Request {context_id} cancelled, ending stream");
                    break false;
                }

                item = response_stream.next() => {
                    let Some(item) = item else {
                        break true;
                    };
                    failed |= response_item_failed(&item);
                    guard.on_item(&item).await;
                    let completed_terminal = !failed && response_item_completed(&item);
                    if completed_terminal {
                        guard.mark_completed_terminal();
                    }
                    yield item;
                    if completed_terminal {
                        break true;
                    }
                }
            }
        };

        if completed && !failed {
            guard.finish().await;
        } else {
            guard.abort().await;
        }
    }
}

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
    request_metrics: Arc<RouterRequestMetrics>,
    affinity: Option<AffinityCoordinator>,
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
        session_affinity_ttl: Option<Duration>,
    ) -> Result<Self, Error> {
        let affinity = session_affinity_ttl
            .map(AffinityCoordinator::new)
            .transpose()?;

        Ok(Self::new_with_coordinator(inner, chooser, affinity))
    }

    pub(crate) fn new_with_coordinator(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
        affinity: Option<AffinityCoordinator>,
    ) -> Self {
        // Eagerly register router request metrics (as zeros) so they are
        // scrapeable before any requests arrive. Both the frontend pipeline
        // and the standalone router create KvPushRouter, so this covers both.
        let request_metrics =
            RouterRequestMetrics::from_component(chooser.client().endpoint.component());

        KvPushRouter {
            inner,
            chooser,
            request_metrics,
            affinity,
        }
    }

    async fn select_request(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        affinity_worker: Option<WorkerWithDpRank>,
    ) -> Result<WorkerSelection, Error> {
        let context_id = request.context().id().to_string();
        let policy_class = request.metadata().get("policy-class").cloned();
        let session_id = request
            .agent_context
            .as_ref()
            .map(|context| context.session_id.clone());
        let routing_parts = RoutingRequestParts::new(request);
        let request_context = request.context().clone();
        let selection_future = self
            .select_worker(
                &context_id,
                request,
                routing_parts,
                phase,
                is_query_only,
                SelectionOptions {
                    affinity_worker,
                    policy_class,
                    session_id,
                },
            )
            .instrument(tracing::info_span!("kv_router.select_worker"));

        cancel_on_stop(request_context.as_ref(), &context_id, selection_future).await?
    }

    async fn select_with_affinity(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
    ) -> Result<(WorkerSelection, Option<AffinityAcquire>), Error> {
        let Some(affinity) = self.affinity.as_ref() else {
            return Ok((
                self.select_request(request, phase, is_query_only, None)
                    .await?,
                None,
            ));
        };
        let Some(session_id) = affinity_id(request)? else {
            return Ok((
                self.select_request(request, phase, is_query_only, None)
                    .await?,
                None,
            ));
        };
        let explicit = explicit_target(request, phase)?;
        if is_query_only {
            let target = affinity.query_target(&session_id, explicit)?;
            let worker = target.and_then(affinity_worker);
            return Ok((
                self.select_request(request, phase, true, worker).await?,
                None,
            ));
        }

        let request_context = request.context();
        let operation = affinity
            .acquire_with_context(&session_id, explicit, request_context.as_ref())
            .await?;
        let worker = operation.target().and_then(affinity_worker);
        match self.select_request(request, phase, false, worker).await {
            Ok(selection) => Ok((selection, Some(operation))),
            Err(error) if is_cancelled(&error) => Err(error),
            Err(_) if operation.target().is_some() && explicit.is_none() => {
                operation.invalidate();
                let retry = affinity
                    .acquire_with_context(&session_id, None, request_context.as_ref())
                    .await?;
                let retry_worker = retry.target().and_then(affinity_worker);
                match self
                    .select_request(request, phase, false, retry_worker)
                    .await
                {
                    Ok(selection) => Ok((selection, Some(retry))),
                    Err(retry_error) => {
                        retry.invalidate();
                        Err(retry_error)
                    }
                }
            }
            Err(error) => {
                operation.invalidate();
                Err(error)
            }
        }
    }

    async fn track_selection(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        is_query_only: bool,
    ) -> Result<RequestGuard, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let routing_parts = RoutingRequestParts::new(request);
        let block_size = self.chooser.block_size() as usize;
        let mut guard = RequestGuard::new(
            self.chooser.clone(),
            self.request_metrics.clone(),
            context_id.clone(),
            request,
            selection.scheduler_tracked,
            selection.request_progress.take(),
            selection.admission_lease.take(),
        );

        let record_result: Result<(), Error> = async {
            if let Some(hashes) = selection.routing_hashes.as_ref()
                && let Some(tracker) = request.tracker.as_ref()
                && tracker.kv_cache_sol_tracking_enabled()
            {
                tracker.record_kv_cache_sol_prompt_hashes(hashes.clone());
            }

            if !is_query_only && self.chooser.indexer().records_routing_decisions() {
                let worker = WorkerWithDpRank::new(selection.instance_id, selection.dp_rank);
                let record_result = if let Some(hashes) = selection.routing_hashes.take() {
                    cancel_on_stop(
                        request_context.as_ref(),
                        &context_id,
                        self.chooser.record_routing_decision_hashes(hashes, worker),
                    )
                    .await?
                } else {
                    let lora_name = request.routing.as_ref().and_then(|r| r.lora_name.clone());
                    let mut tokens_with_hashes = TokensWithHashes::new(
                        routing_parts.token_ids.to_vec(),
                        self.chooser.block_size(),
                    )
                    .with_is_eagle(self.chooser.is_eagle());
                    if let Some(infos) = routing_parts.block_mm_infos {
                        tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                    }
                    if let Some(lora_name) = lora_name {
                        tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                    }
                    cancel_on_stop(
                        request_context.as_ref(),
                        &context_id,
                        self.chooser
                            .record_routing_decision(tokens_with_hashes, worker),
                    )
                    .await?
                };
                if let Err(error) = record_result {
                    tracing::warn!(
                        request_id = %context_id,
                        worker_id = selection.instance_id,
                        dp_rank = selection.dp_rank,
                        error = %error,
                        "Failed to record routing decision"
                    );
                }
            }

            if let Some(ref tracker) = request.tracker {
                let isl_blocks = routing_parts.token_ids.len().div_ceil(block_size);
                tracker.record_kv_hit(selection.effective_overlap_blocks, isl_blocks);
                tracker.record_isl(routing_parts.token_ids.len(), Some(selection.cached_tokens));
                tracker.record_worker(
                    selection.instance_id,
                    Some(selection.dp_rank),
                    self.chooser.worker_type(),
                );
                tracker.record_router_queue_depth(self.chooser.pending_count());
                if let Some(hit_rate) = tracker.kv_hit_rate() {
                    guard.request_metrics().kv_hit_rate.observe(hit_rate);
                }
            }
            guard
                .request_metrics()
                .input_sequence_tokens
                .observe(request.token_ids.len() as f64);
            Ok(())
        }
        .await;

        if let Err(error) = record_result {
            guard.abort().await;
            return Err(error);
        }
        Ok(guard)
    }

    async fn dispatch_selection(
        &self,
        request: SingleIn<PreprocessedRequest>,
        selection: WorkerSelection,
        mut guard: RequestGuard,
        exact: bool,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let phase = request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated);
        let phase_label = phase.to_string();
        guard.start_dispatch(&phase_label);
        self.warn_if_output_replay_annotation_ignored(&request, &selection);

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(selection.dp_rank);
        let updated_request = context.map(|_| backend_input);
        guard.record_prefill_start();

        let dispatch = async {
            if exact {
                self.inner
                    .dispatch_exact(updated_request, selection.instance_id)
                    .await
            } else {
                self.inner
                    .direct(updated_request, selection.instance_id)
                    .await
            }
        };
        let dispatch_result = cancel_on_stop(
            request_context.as_ref(),
            &context_id,
            dispatch.instrument(tracing::info_span!(
                "kv_router.route_request",
                request_id = %context_id,
                worker_id = selection.instance_id,
                dp_rank = selection.dp_rank,
                overlap_blocks = selection.overlap_amount,
                phase = ?phase,
            )),
        )
        .await
        .and_then(|result| result);
        let response_stream = match dispatch_result {
            Ok(stream) => stream,
            Err(error) => {
                guard.abort().await;
                return Err(error);
            }
        };

        guard.mark_dispatched().await;
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();
        let wrapped_stream = Box::pin(monitor_response_stream(
            response_stream,
            context_for_monitoring,
            context_id,
            guard,
        ));
        Ok(ResponseStream::new(wrapped_stream, stream_context))
    }

    fn warn_if_output_replay_annotation_ignored(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &WorkerSelection,
    ) {
        let Some(replay_key) = request.get_annotation_value(OUTPUT_REPLAY_ID_ANNOTATION_KEY) else {
            return;
        };
        let consumes_replay = self
            .chooser
            .workers_with_configs
            .borrow()
            .get(&selection.instance_id)
            .and_then(|config| {
                config
                    .get_engine_specific::<bool>(OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY)
                    .ok()
                    .flatten()
            })
            .unwrap_or(false);
        if consumes_replay {
            return;
        }

        tracing::warn!(
            replay_key,
            worker_id = selection.instance_id,
            dp_rank = selection.dp_rank,
            "request has output token replay annotation but selected worker has not declared replay-token consumption"
        );
    }

    pub(crate) async fn select_and_dispatch_prefill<M, F>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        let phase = RequestPhase::Prefill;
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        let (mut selection, mut operation) = self
            .select_with_affinity(&request, phase, is_query_only)
            .await?;
        let mut guard = match self
            .track_selection(&request, &mut selection, is_query_only)
            .await
        {
            Ok(guard) => guard,
            Err(error) => {
                invalidate_on_non_cancellation(&mut operation, &error);
                return Err(error);
            }
        };
        let selected_target = AffinityTarget {
            worker_id: selection.instance_id,
            dp_rank: Some(selection.dp_rank),
        };
        let metadata = match prepare(&mut request, selected_target) {
            Ok(metadata) => metadata,
            Err(error) => {
                guard.abort().await;
                invalidate_on_non_cancellation(&mut operation, &error);
                return Err(error);
            }
        };
        drop(route_guard);
        let stream = match self
            .dispatch_selection(request, selection, guard, true)
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                invalidate_on_non_cancellation(&mut operation, &error);
                return Err(error);
            }
        };
        let Some(operation) = operation else {
            return Ok((metadata, stream));
        };
        Ok((metadata, operation.into_stream(selected_target, stream)?))
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for KvPushRouter
{
    /// Generate method that handles KV-aware routing with three distinct behaviors:
    ///
    /// 1. **If `query_instance_id` annotation is set**:
    ///    - Returns the best matching worker ID without routing the request
    ///    - Does NOT update any router local states
    ///    - Response includes worker_instance_id and token_data annotations
    ///
    /// 2. **If a phase-specific worker or `backend_instance_id` is set in the request**:
    ///    - Query-only requests return that worker selection without state updates
    ///    - Requests route through the scheduler as an exact pin when dp_rank is resolved
    ///    - If dp_rank cannot be resolved, the request is rejected instead of treating rank 0 as a sentinel
    ///
    /// 3. **If neither are set (default behavior)**:
    ///    - Finds the best worker based on KV cache overlap
    ///    - Updates router states to track the request
    ///    - Routes to the selected worker
    ///
    /// The router state updates include tracking active sequences and managing
    /// prefill/completion lifecycle for proper KV cache management.
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        let phase = request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated);
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let (mut selection, mut operation) = self
            .select_with_affinity(&request, phase, is_query_only)
            .await?;
        if is_query_only {
            let routing_parts = RoutingRequestParts::new(&request);
            if let Some(ref tracker) = request.tracker {
                let isl_blocks = routing_parts
                    .token_ids
                    .len()
                    .div_ceil(self.chooser.block_size() as usize);
                tracker.record_kv_hit(selection.effective_overlap_blocks, isl_blocks);
                tracker.record_isl(routing_parts.token_ids.len(), Some(selection.cached_tokens));
                tracker.record_worker(
                    selection.instance_id,
                    Some(selection.dp_rank),
                    self.chooser.worker_type(),
                );
                tracker.record_router_queue_depth(self.chooser.pending_count());
            }
            self.request_metrics
                .input_sequence_tokens
                .observe(request.token_ids.len() as f64);
            let stream_context = request.context().clone();
            let worker_id_info = request
                .tracker
                .as_ref()
                .and_then(|tracker| tracker.get_worker_info());

            tracing::trace!(
                ?phase,
                worker_id = selection.instance_id,
                ?worker_id_info,
                "Returning worker selection (query-only mode)"
            );

            let output = LLMEngineOutput {
                routing_data: Some(RoutingData {
                    worker_id: worker_id_info,
                    token_ids: Some(request.token_ids.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let response = Annotated::from_data(output);
            let stream = stream::iter(vec![response]);
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        }

        let guard = match self.track_selection(&request, &mut selection, false).await {
            Ok(guard) => guard,
            Err(error) => {
                invalidate_on_non_cancellation(&mut operation, &error);
                return Err(error);
            }
        };
        drop(route_guard);
        let selected_target = AffinityTarget {
            worker_id: selection.instance_id,
            dp_rank: Some(selection.dp_rank),
        };
        let stream = match self
            .dispatch_selection(request, selection, guard, operation.is_some())
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                invalidate_on_non_cancellation(&mut operation, &error);
                return Err(error);
            }
        };
        match operation {
            Some(operation) => operation.into_stream(selected_target, stream),
            None => Ok(stream),
        }
    }
}

fn affinity_worker(target: AffinityTarget) -> Option<WorkerWithDpRank> {
    target
        .dp_rank
        .map(|rank| WorkerWithDpRank::new(target.worker_id, rank))
}

/// A direct routing wrapper for `RouterMode::Direct`.
///
/// This wraps a `PushRouter` and reads worker IDs from each request's routing hints,
/// then routes directly to the specified worker. Used when an external router
/// (e.g., EPP) handles worker selection.
pub struct DirectRoutingRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
}

impl DirectRoutingRouter {
    pub fn new(inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>) -> Self {
        DirectRoutingRouter { inner }
    }

    /// Extract worker ID from request routing hints.
    /// Returns an error if no worker ID is found (required in direct routing mode).
    fn get_worker_id(request: &PreprocessedRequest) -> Result<u64, Error> {
        let routing = request.routing.as_ref();
        let worker_id = routing.and_then(|r| r.decode_worker_id.or(r.backend_instance_id));

        worker_id.ok_or_else(|| {
            anyhow::anyhow!(
                "Worker ID required (--direct-route) but none found in request. \
                 Expected decode_worker_id or backend_instance_id to be set by external router (e.g., EPP)."
            )
        })
    }
}

fn response_item_failed(item: &Annotated<LLMEngineOutput>) -> bool {
    item.error.is_some()
        || item.event.as_deref() == Some("error")
        || item
            .data
            .as_ref()
            .and_then(|data| data.finish_reason.as_ref())
            .is_some_and(|reason| {
                matches!(reason, FinishReason::Error(_) | FinishReason::Cancelled)
            })
}

fn response_item_completed(item: &Annotated<LLMEngineOutput>) -> bool {
    !response_item_failed(item)
        && item
            .data
            .as_ref()
            .is_some_and(|data| data.finish_reason.is_some())
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for DirectRoutingRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let worker_id = Self::get_worker_id(&request)?;

        tracing::debug!(worker_id = worker_id, "Direct routing to specified worker");

        self.inner.direct(request, worker_id).await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        future,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use dynamo_kv_router::{
        ActiveSequencesMultiWorker, DefaultWorkerSelector, SequencePublisher,
        config::{KvRouterConfig, RouterQueuePolicy},
        protocols::{ActiveLoad, ActiveSequenceEvent, RoutingConstraints},
        scheduling::{
            AdmissionAction, AdmissionDecision, AdmissionEvent, AdmissionId, AdmissionRequest,
            LocalScheduler, NoopOverlapScoresRefresh, OverlapSignals,
            PolicyClassAdmissionStrategies, PolicyClassAdmissionStrategy, PolicyProfile,
            ScheduleMode, ScheduleRequest, WorkerPlacement,
        },
    };
    use dynamo_runtime::{
        CancellationToken, DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        error::{ErrorType, match_error_chain},
        pipeline::{AsyncEngineContext, Context, PushRouter, RouterMode, context::Controller},
    };
    use tokio::sync::watch;

    use super::*;
    use crate::{
        local_model::runtime_config::ModelRuntimeConfig,
        protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
    };

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    struct NoopSequencePublisher;

    impl SequencePublisher for NoopSequencePublisher {
        fn publish_event(
            &self,
            _event: &ActiveSequenceEvent,
        ) -> impl future::Future<Output = anyhow::Result<()>> + Send {
            future::ready(Ok(()))
        }

        fn publish_load(&self, _load: ActiveLoad) {}

        fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
    }

    struct RecordingAdmissionStrategy(Arc<Mutex<Vec<AdmissionEvent>>>);

    impl PolicyClassAdmissionStrategy for RecordingAdmissionStrategy {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Ready(WorkerPlacement::Any)
        }

        fn on_event(&mut self, event: AdmissionEvent) -> Vec<AdmissionAction> {
            self.0.lock().unwrap().push(event);
            Vec::new()
        }
    }

    #[test]
    fn response_item_failed_includes_typed_terminal_failures() {
        let mut output = LLMEngineOutput::default();
        assert!(!response_item_failed(&Annotated::from_data(output.clone())));

        output.finish_reason = Some(FinishReason::Error("decode failed".to_string()));
        assert!(response_item_failed(&Annotated::from_data(output.clone())));

        output.finish_reason = Some(FinishReason::Cancelled);
        assert!(response_item_failed(&Annotated::from_data(output.clone())));

        output.finish_reason = Some(FinishReason::Length);
        assert!(!response_item_failed(&Annotated::from_data(output)));
    }

    #[test]
    fn response_item_completed_requires_successful_terminal_reason() {
        for reason in [FinishReason::Stop, FinishReason::EoS, FinishReason::Length] {
            let output = LLMEngineOutput {
                finish_reason: Some(reason),
                ..Default::default()
            };
            assert!(response_item_completed(&Annotated::from_data(output)));
        }

        let output = LLMEngineOutput {
            finish_reason: Some(FinishReason::Error("decode failed".to_string())),
            ..Default::default()
        };
        assert!(!response_item_completed(&Annotated::from_data(output)));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn dropping_stream_after_terminal_item_reports_admission_completed() {
        let (router, runtime) = router(None).await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            16,
            HashMap::from([(7, (0, 1))]),
            false,
            0,
            "decode",
        ));
        let (_config_tx, config_rx) =
            watch::channel(HashMap::from([(7, ModelRuntimeConfig::default())]));
        let mut strategies = PolicyClassAdmissionStrategies::new();
        strategies.insert(
            "default".to_owned(),
            Box::new(RecordingAdmissionStrategy(Arc::clone(&events))),
        );
        let cancel = CancellationToken::new();
        let scheduler = LocalScheduler::new_with_policy_profile_and_admission_strategies(
            Arc::clone(&slots),
            config_rx,
            PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
            16,
            DefaultWorkerSelector::new(None, "decode"),
            None,
            None::<Arc<NoopOverlapScoresRefresh>>,
            None,
            Duration::from_secs(60),
            true,
            cancel.clone(),
            "decode",
            false,
            strategies,
        )
        .unwrap();

        for (index, finish_reason) in [FinishReason::Stop, FinishReason::EoS, FinishReason::Length]
            .into_iter()
            .enumerate()
        {
            let request_id = format!("terminal-drop-{index}");
            let mut response = scheduler
                .schedule_request(ScheduleRequest {
                    mode: ScheduleMode::TrackedWithAdmission {
                        request_id: request_id.clone(),
                    },
                    token_seq: Some(vec![1]),
                    block_hashes: None,
                    isl_tokens: 1,
                    lora_name: None,
                    expected_output_tokens: None,
                    pinned_worker: None,
                    allowed_worker_ids: None,
                    routing_constraints: RoutingConstraints::default(),
                    router_config_override: None,
                    priority_jump: 0.0,
                    strict_priority: 0,
                    policy_class: None,
                    session_id: None,
                    overlap: OverlapSignals::default(),
                    shared_cache_hits: None,
                })
                .await
                .unwrap();
            let worker = response.best_worker;
            let mut guard = RequestGuard::new(
                Arc::clone(&router.chooser),
                Arc::clone(&router.request_metrics),
                request_id.clone(),
                &request(),
                true,
                response.request_progress.take(),
                response.admission_lease.take(),
            );
            guard.mark_dispatched().await;

            let context = Context::new(()).context();
            let source = ResponseStream::new(
                Box::pin(stream::iter([Annotated::from_data(LLMEngineOutput {
                    finish_reason: Some(finish_reason.clone()),
                    ..Default::default()
                })])),
                Arc::clone(&context),
            );
            {
                let monitored = monitor_response_stream(source, context, request_id.clone(), guard);
                tokio::pin!(monitored);
                let item = monitored.next().await.unwrap();
                assert_eq!(
                    item.data.and_then(|output| output.finish_reason),
                    Some(finish_reason)
                );
            }

            let expected_len = (index + 1) * 2;
            tokio::time::timeout(Duration::from_secs(1), async {
                while events.lock().unwrap().len() < expected_len {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("terminal stream drop did not report admission completion");
            assert_eq!(
                &events.lock().unwrap()[index * 2..expected_len],
                [
                    AdmissionEvent::Dispatched {
                        id: AdmissionId::new(index as u64),
                        worker,
                    },
                    AdmissionEvent::Completed {
                        id: AdmissionId::new(index as u64),
                        context_tokens: 1,
                    },
                ]
            );
        }

        assert!(
            slots
                .active_request_counts()
                .values()
                .all(|count| *count == 0)
        );
        cancel.cancel();
        drop(router);
        runtime.shutdown();
    }

    async fn router(session_affinity_ttl: Option<Duration>) -> (KvPushRouter, Runtime) {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let component = distributed
            .namespace("affinity-selection-cancellation".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap();
        let endpoint = component.endpoint("generate");
        let client = endpoint.client().await.unwrap();
        let workers = HashMap::from([(7, ModelRuntimeConfig::default())]);
        let (_tx, workers) = watch::channel(workers);
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            ..Default::default()
        };
        let chooser = KvRouter::new(
            endpoint,
            client.clone(),
            workers,
            16,
            DefaultWorkerSelector::new(Some(config.clone()), "decode"),
            Some(config),
            None,
            "decode",
            None,
            false,
            None,
            None,
        )
        .await
        .unwrap();
        let inner = PushRouter::from_client(client, RouterMode::KV)
            .await
            .unwrap();
        let router = KvPushRouter::new(inner, Arc::new(chooser), session_affinity_ttl).unwrap();
        (router, runtime)
    }

    async fn track_request(
        router: &KvPushRouter,
        is_query_only: bool,
    ) -> (SingleIn<PreprocessedRequest>, WorkerSelection, RequestGuard) {
        let request = Context::new(request());
        let (mut selection, _) = router
            .select_with_affinity(&request, RequestPhase::Aggregated, is_query_only)
            .await
            .unwrap();
        let guard = router
            .track_selection(&request, &mut selection, is_query_only)
            .await
            .unwrap();
        (request, selection, guard)
    }

    #[tokio::test]
    async fn session_affinity_disabled_does_not_create_coordinator() {
        let (router, runtime) = router(None).await;
        assert!(router.affinity.is_none());

        drop(router);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn router_request_counters_follow_admission_and_completion_lifecycle() {
        let (router, runtime) = router(None).await;
        let metrics = router.request_metrics.clone();
        let started_before = metrics.requests_started_total().get();
        let completed_before = metrics.requests_total.get();

        let controller = Controller::new("pre-admission-cancellation".to_string());
        controller.stop();
        let cancelled_request = Context::with_controller(request(), controller);
        assert!(
            router
                .select_with_affinity(&cancelled_request, RequestPhase::Aggregated, false)
                .await
                .is_err()
        );
        assert_eq!(metrics.requests_started_total().get(), started_before);

        let (_, _, mut query_guard) = track_request(&router, true).await;
        query_guard.abort().await;
        drop(query_guard);
        assert_eq!(metrics.requests_started_total().get(), started_before);

        let (_, _, mut cancelled_guard) = track_request(&router, false).await;

        assert_eq!(metrics.requests_started_total().get(), started_before + 1);
        assert_eq!(metrics.requests_total.get(), completed_before);

        // Admission remains counted even when the request aborts before dispatch.
        cancelled_guard.abort().await;
        drop(cancelled_guard);
        assert_eq!(metrics.requests_started_total().get(), started_before + 1);
        assert_eq!(metrics.requests_total.get(), completed_before);

        let (failed_request, failed_selection, failed_dispatch_guard) =
            track_request(&router, false).await;
        assert!(
            router
                .dispatch_selection(
                    failed_request,
                    failed_selection,
                    failed_dispatch_guard,
                    true,
                )
                .await
                .is_err()
        );
        assert_eq!(metrics.requests_started_total().get(), started_before + 2);
        assert_eq!(metrics.requests_total.get(), completed_before);

        let (_, _, mut completed_guard) = track_request(&router, false).await;
        completed_guard.start_dispatch("aggregated");
        completed_guard.mark_dispatched().await;
        completed_guard.finish().await;
        drop(completed_guard);
        assert_eq!(metrics.requests_started_total().get(), started_before + 3);
        assert_eq!(metrics.requests_total.get(), completed_before + 1);

        drop(router);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn session_affinity_post_selection_cancellation_preserves_binding() {
        let (router, runtime) = router(Some(Duration::from_secs(10))).await;
        let affinity = router.affinity.as_ref().unwrap();
        let session_id = SessionAffinityId::new("cancelled-after-selection");
        let original_target = AffinityTarget {
            worker_id: 7,
            dp_rank: Some(0),
        };
        let AffinityAcquire::Initialize(initializer) =
            affinity.acquire(&session_id, None).await.unwrap()
        else {
            panic!("first request must initialize");
        };
        drop(initializer.commit(original_target).unwrap());

        let mut operation = Some(affinity.acquire(&session_id, None).await.unwrap());
        let cancellation = cancellation::cancelled_error("cancelled-after-selection-request");
        invalidate_on_non_cancellation(&mut operation, &cancellation);
        assert!(operation.is_some());
        drop(operation);
        assert_eq!(
            affinity.query_target(&session_id, None).unwrap(),
            Some(original_target)
        );

        let mut operation = Some(affinity.acquire(&session_id, None).await.unwrap());
        let failure = anyhow::anyhow!("dispatch failed");
        invalidate_on_non_cancellation(&mut operation, &failure);
        assert!(operation.is_none());
        assert_eq!(affinity.query_target(&session_id, None).unwrap(), None);

        drop(router);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn session_affinity_existing_selection_cancellation_preserves_binding_without_retry() {
        let (router, runtime) = router(Some(Duration::from_secs(10))).await;
        let session_id = SessionAffinityId::new("cancelled-selection");
        let original_target = AffinityTarget {
            worker_id: 7,
            dp_rank: Some(0),
        };
        let AffinityAcquire::Initialize(initializer) = router
            .affinity
            .as_ref()
            .unwrap()
            .acquire(&session_id, None)
            .await
            .unwrap()
        else {
            panic!("first request must initialize");
        };
        drop(initializer.commit(original_target).unwrap());

        let controller = Controller::new("cancelled-selection-request".to_string());
        controller.stop();
        let mut request = Context::with_controller(request(), controller);
        request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());

        let Err(error) = router
            .select_with_affinity(&request, RequestPhase::Aggregated, false)
            .await
        else {
            panic!("stopped request must return cancellation");
        };
        assert!(match_error_chain(
            error.as_ref(),
            &[ErrorType::Cancelled],
            &[]
        ));
        assert_eq!(
            router
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&session_id, None)
                .unwrap(),
            Some(original_target)
        );

        let AffinityAcquire::Bound { target, lease } = router
            .affinity
            .as_ref()
            .unwrap()
            .acquire(&session_id, None)
            .await
            .unwrap()
        else {
            panic!("cancellation must preserve the existing binding");
        };
        assert_eq!(target, original_target);
        drop(lease);

        drop(router);
        runtime.shutdown();
    }
}

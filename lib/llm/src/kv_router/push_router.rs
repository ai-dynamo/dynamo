// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_kv_router::{
    protocols::{TokensWithHashes, WorkerWithDpRank},
    selector::{WorkerInputs, WorkerSelector},
};
use dynamo_runtime::{
    error::{DynamoError, ErrorType, match_error_chain},
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        ResponseStream, RouterMode, SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tracing::Instrument;

use crate::{
    kv_router::{
        KvRouter, metrics::RouterRequestMetrics, scheduler::DefaultWorkerSelector,
        to_worker_selection_session_context,
    },
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::{
        FinishReason,
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RoutingData, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
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

fn route_target(worker: WorkerWithDpRank) -> AffinityTarget {
    AffinityTarget::new(worker.worker_id, Some(worker.dp_rank))
}

fn monitor_response_stream<Sel>(
    mut response_stream: ManyOut<Annotated<LLMEngineOutput>>,
    context: Arc<dyn AsyncEngineContext>,
    mut guard: RequestGuard<Sel>,
) -> impl futures::Stream<Item = Annotated<LLMEngineOutput>> + Send
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    async_stream::stream! {
        // Keep one cancellation future alive for the whole response stream. Calling
        // `stopped()` for every item repeatedly clones and polls a watch receiver.
        let stopped = context.stopped();
        tokio::pin!(stopped);

        let completed = loop {
            tokio::select! {
                biased;

                _ = &mut stopped => {
                    tracing::debug!(request_id = context.id(), "Request cancelled, ending stream");
                    break false;
                }

                item = response_stream.next() => {
                    let Some(item) = item else {
                        break true;
                    };
                    let item_failed = response_item_failed(&item);
                    guard.on_item(&item).await;
                    if item_failed {
                        guard.record_migration_failure(item.error.clone());
                        // Release the failed attempt before Migration can observe
                        // the item and start another one. This keeps serialized
                        // retries free of stale-cleanup ABA races.
                        guard.abort().await;
                        yield item;
                        break false;
                    }
                    yield item;
                }
            }
        };

        if completed {
            guard.finish().await;
        } else {
            guard.abort().await;
        }
    }
}

fn into_monitored_response<Sel>(
    response_stream: ManyOut<Annotated<LLMEngineOutput>>,
    guard: RequestGuard<Sel>,
) -> ManyOut<Annotated<LLMEngineOutput>>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    let stream_context = response_stream.context();
    let wrapped_stream = Box::pin(monitor_response_stream(
        response_stream,
        stream_context.clone(),
        guard,
    ));
    ResponseStream::new(wrapped_stream, stream_context)
}

/// First-party policies that need no KV-cache index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinRoutingPolicy {
    RoundRobin,
    Random,
}

impl BuiltinRoutingPolicy {
    pub fn from_router_mode(mode: RouterMode) -> Option<Self> {
        match mode {
            RouterMode::RoundRobin => Some(Self::RoundRobin),
            RouterMode::Random => Some(Self::Random),
            _ => None,
        }
    }

    pub const fn required_worker_inputs(self) -> WorkerInputs {
        WorkerInputs::NONE
    }
}

enum RoutingPlane<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    Kv(Arc<KvRouter<Sel>>),
    Builtin(BuiltinRoutingPolicy),
}

/// Owns request routing from worker selection through response cleanup.
///
/// [`PushRouter`] owns discovery, fault detection, and transport. [`KvRouter`]
/// owns optional KV candidate state. `RoutingHost` owns the common request
/// lifecycle regardless of which policy selected the worker.
pub struct RoutingHost<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    plane: RoutingPlane<Sel>,
    request_metrics: Arc<RouterRequestMetrics>,
    affinity: Option<AffinityCoordinator>,
}

/// Compatibility name for the KV-only host used by existing callers.
///
/// This alias remains supported through the Dynamo 1.x series. It may be
/// removed only in a 2.0.0 (or later) breaking release.
pub type KvPushRouter<Sel = DefaultWorkerSelector> = RoutingHost<Sel>;

impl<Sel> RoutingHost<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        kv_router: Arc<KvRouter<Sel>>,
        session_affinity_ttl: Option<Duration>,
    ) -> Result<Self, Error> {
        let affinity = session_affinity_ttl
            .map(AffinityCoordinator::new)
            .transpose()?;

        Ok(Self::new_with_coordinator(inner, kv_router, affinity))
    }

    pub(crate) fn new_with_coordinator(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        kv_router: Arc<KvRouter<Sel>>,
        affinity: Option<AffinityCoordinator>,
    ) -> Self {
        // Eagerly register router request metrics (as zeros) so they are
        // scrapeable before any requests arrive. Both the frontend pipeline
        // and the standalone router create RoutingHost, so this covers both.
        let request_metrics =
            RouterRequestMetrics::from_component(kv_router.client().endpoint.component());

        RoutingHost {
            inner,
            plane: RoutingPlane::Kv(kv_router),
            request_metrics,
            affinity,
        }
    }

    pub(crate) fn new_builtin(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<Self, Error> {
        let policy =
            BuiltinRoutingPolicy::from_router_mode(inner.router_mode()).ok_or_else(|| {
                anyhow::anyhow!(
                    "{:?} routing is not a stateless builtin policy",
                    inner.router_mode()
                )
            })?;
        debug_assert_eq!(policy.required_worker_inputs(), WorkerInputs::NONE);
        let request_metrics =
            RouterRequestMetrics::from_component(inner.client.endpoint.component());
        Ok(Self {
            inner,
            plane: RoutingPlane::Builtin(policy),
            request_metrics,
            affinity: None,
        })
    }

    pub fn required_worker_inputs(&self) -> WorkerInputs {
        match &self.plane {
            RoutingPlane::Kv(chooser) => chooser.required_worker_inputs(),
            RoutingPlane::Builtin(policy) => policy.required_worker_inputs(),
        }
    }

    /// The active KV-aware data plane.
    pub fn kv_router(&self) -> &Arc<KvRouter<Sel>> {
        self.kv_router_if_enabled()
            .expect("routing host has no KV capability")
    }

    pub(crate) fn kv_router_if_enabled(&self) -> Option<&Arc<KvRouter<Sel>>> {
        match &self.plane {
            RoutingPlane::Kv(chooser) => Some(chooser),
            RoutingPlane::Builtin(_) => None,
        }
    }

    pub(crate) fn peek_next_worker(&self) -> Option<u64> {
        match &self.plane {
            RoutingPlane::Builtin(_) => self.inner.peek_next_worker(),
            RoutingPlane::Kv(_) => None,
        }
    }

    pub(crate) fn query_affinity_worker(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
    ) -> Result<Option<WorkerWithDpRank>, Error> {
        let Some(affinity) = self.affinity.as_ref() else {
            return Ok(None);
        };
        let Some(session_id) = affinity_id(request)? else {
            return Ok(None);
        };
        let explicit = explicit_target(request, phase)?;
        let target = affinity.query_target(&session_id, explicit)?;
        Ok(target.and_then(affinity_worker))
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
        let session_context = request
            .agent_context
            .as_ref()
            .map(to_worker_selection_session_context);
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
                    session_context,
                },
            )
            .instrument(tracing::info_span!("kv_router.select_worker"));

        cancel_on_stop(request_context.as_ref(), selection_future).await?
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
    ) -> Result<RequestGuard<Sel>, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let routing_parts = RoutingRequestParts::new(request);
        let chooser = self.kv_router();
        let block_size = chooser.block_size() as usize;
        let selected_worker = selection.worker;
        let mut guard = RequestGuard::new_kv(
            Arc::clone(chooser),
            self.request_metrics.clone(),
            context_id.clone(),
            selected_worker,
            request,
            !is_query_only,
        );

        let record_result: Result<(), Error> = async {
            if !is_query_only && chooser.indexer().records_routing_decisions() {
                let worker = selected_worker;
                let record_result = if let Some(hashes) = selection.routing_hashes.take() {
                    cancel_on_stop(
                        request_context.as_ref(),
                        chooser.record_routing_decision_hashes(hashes, worker),
                    )
                    .await?
                } else {
                    let lora_name = request.routing.as_ref().and_then(|r| r.lora_name.clone());
                    let mut tokens_with_hashes = TokensWithHashes::new(
                        routing_parts.token_ids.to_vec(),
                        chooser.block_size(),
                    )
                    .with_is_eagle(chooser.is_eagle());
                    if let Some(infos) = routing_parts.block_mm_infos {
                        tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                    }
                    if let Some(lora_name) = lora_name {
                        tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                    }
                    cancel_on_stop(
                        request_context.as_ref(),
                        chooser.record_routing_decision(tokens_with_hashes, worker),
                    )
                    .await?
                };
                if let Err(error) = record_result {
                    tracing::warn!(
                        request_id = %context_id,
                        worker_id = selection.worker.worker_id,
                        dp_rank = selection.worker.dp_rank,
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
                    selection.worker.worker_id,
                    Some(selection.worker.dp_rank),
                    chooser.worker_type(),
                );
                tracker.record_router_queue_depth(chooser.pending_count());
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
        mut guard: RequestGuard<Sel>,
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
        backend_input.routing_mut().dp_rank = Some(selection.worker.dp_rank);
        let _ = backend_input
            .extra_args
            .as_mut()
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|args| args.get_mut("kv_transfer_params"))
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|params| params.remove("router_hint"));
        if let Some(router_hint) = selection.router_hint.as_ref()
            && let Err(error) = backend_input.attach_router_hint(router_hint)
        {
            tracing::warn!(
                request_id = %context_id,
                worker_id = selection.worker.worker_id,
                error = %error,
                "Failed to attach router_hint to backend request"
            );
        }
        let updated_request = context.map(|_| backend_input);
        guard.record_prefill_start();

        let dispatch = async {
            if exact {
                self.inner
                    .dispatch_exact(updated_request, selection.worker.worker_id)
                    .await
            } else {
                self.inner
                    .direct(updated_request, selection.worker.worker_id)
                    .await
            }
        };
        let dispatch_result = cancel_on_stop(
            request_context.as_ref(),
            dispatch.instrument(tracing::info_span!(
                "kv_router.route_request",
                request_id = %context_id,
                worker_id = selection.worker.worker_id,
                dp_rank = selection.worker.dp_rank,
                overlap_blocks = selection.overlap_amount,
                phase = ?phase,
            )),
        )
        .await
        .and_then(|result| result);
        let response_stream = match dispatch_result {
            Ok(stream) => stream,
            Err(error) => {
                let typed_error = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<DynamoError>().cloned());
                guard.record_migration_failure(typed_error);
                guard.abort().await;
                return Err(error);
            }
        };

        guard.mark_dispatched();
        Ok(into_monitored_response(response_stream, guard))
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
            .kv_router()
            .workers_with_configs
            .borrow()
            .get(&selection.worker.worker_id)
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
            worker_id = selection.worker.worker_id,
            dp_rank = selection.worker.dp_rank,
            "request has output token replay annotation but selected worker has not declared replay-token consumption"
        );
    }

    async fn select_and_dispatch_builtin<M, F>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        let RoutingPlane::Builtin(policy) = &self.plane else {
            unreachable!("builtin dispatch called for KV routing")
        };
        let policy = *policy;
        debug_assert_eq!(policy.required_worker_inputs(), WorkerInputs::NONE);

        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let explicit = explicit_target(&request, phase)?;
        let initial_worker = match explicit {
            Some(target) => target.worker_id,
            None => self.inner.select_next_worker().ok_or_else(|| {
                anyhow::anyhow!("no eligible worker available for {policy:?} routing")
            })?,
        };
        let mut guard: RequestGuard<Sel> =
            RequestGuard::new_stateless(self.request_metrics.clone(), initial_worker, &request);
        let tracker = request.tracker.clone();
        let request_context = request.context().clone();
        self.request_metrics
            .input_sequence_tokens
            .observe(request.token_ids.len() as f64);
        drop(route_guard);

        guard.start_dispatch(&phase_label);
        guard.record_prefill_start();
        let dispatch_result = if let Some(target) = explicit {
            request.routing_mut().dp_rank = target.dp_rank;
            let metadata = match prepare(&mut request, target) {
                Ok(metadata) => metadata,
                Err(error) => {
                    guard.abort().await;
                    return Err(error);
                }
            };
            cancel_on_stop(
                request_context.as_ref(),
                self.inner.dispatch_exact(request, target.worker_id),
            )
            .await
            .and_then(|result| result)
            .map(|stream| (metadata, target, stream))
        } else {
            cancel_on_stop(
                request_context.as_ref(),
                self.inner.dispatch_selected_untracked(
                    initial_worker,
                    request,
                    |request, worker_id| {
                        let target = AffinityTarget::new(worker_id, None);
                        request.routing_mut().dp_rank = None;
                        prepare(request, target).map(|metadata| (metadata, target))
                    },
                ),
            )
            .await
            .and_then(|result| result)
            .map(|((metadata, target), stream)| (metadata, target, stream))
        };

        let (metadata, target, response_stream) = match dispatch_result {
            Ok(result) => result,
            Err(error) => {
                let typed_error = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<DynamoError>().cloned());
                guard.record_migration_failure(typed_error);
                guard.abort().await;
                return Err(error);
            }
        };
        guard.set_stateless_worker(target.worker_id);
        if let Some(tracker) = tracker {
            let worker_type = if tracker.phase() == RequestPhase::Prefill {
                WORKER_TYPE_PREFILL
            } else {
                WORKER_TYPE_DECODE
            };
            tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
        }
        guard.mark_dispatched();
        Ok((metadata, into_monitored_response(response_stream, guard)))
    }

    async fn select_and_dispatch_kv_prefill<M, F>(
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
        let selected_target = route_target(selection.worker);
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

    pub(crate) async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        match &self.plane {
            RoutingPlane::Kv(_) => self.select_and_dispatch_kv_prefill(request, prepare).await,
            RoutingPlane::Builtin(_) => {
                self.select_and_dispatch_builtin(request, RequestPhase::Prefill, prepare)
                    .await
            }
        }
    }
}

#[async_trait]
impl<Sel> AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for RoutingHost<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
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
        if matches!(&self.plane, RoutingPlane::Builtin(_)) {
            let phase = request
                .tracker
                .as_ref()
                .map(|tracker| tracker.phase())
                .unwrap_or(RequestPhase::Aggregated);
            return self
                .select_and_dispatch_builtin(request, phase, |_, _| Ok(()))
                .await
                .map(|(_, stream)| stream);
        }

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
                    .div_ceil(self.kv_router().block_size() as usize);
                tracker.record_kv_hit(selection.effective_overlap_blocks, isl_blocks);
                tracker.record_isl(routing_parts.token_ids.len(), Some(selection.cached_tokens));
                tracker.record_worker(
                    selection.worker.worker_id,
                    Some(selection.worker.dp_rank),
                    self.kv_router().worker_type(),
                );
                tracker.record_router_queue_depth(self.kv_router().pending_count());
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
                worker_id = selection.worker.worker_id,
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
        let selected_target = route_target(selection.worker);
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
        collections::{HashMap, HashSet},
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use dynamo_kv_router::{
        DefaultWorkerSelector, WorkerSelectionPolicy, config::KvRouterConfig,
        protocols::RoutingConstraints,
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::Instance,
        discovery::EventTransportKind,
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        error::{ErrorType, match_error_chain},
        pipeline::{
            AddressedRequest, AsyncEngineContext, Context, ManyIn, Operator, PushRouter,
            RouterMode, ServerStreamingEngine, StreamingDispatch, context::Controller,
        },
        storage::kv::Selector,
    };
    use tokio::sync::watch;

    use super::*;
    use crate::{
        http::service::metrics::Metrics,
        local_model::runtime_config::ModelRuntimeConfig,
        migration::Migration,
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
    fn selector_state_remains_owned_by_the_scheduler_actor() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<RoutingHost<WorkerSelectionPolicy>>();
    }

    #[test]
    fn stateless_builtin_policies_request_no_optional_capabilities() {
        assert_eq!(
            BuiltinRoutingPolicy::RoundRobin.required_worker_inputs(),
            WorkerInputs::NONE
        );
        assert_eq!(
            BuiltinRoutingPolicy::Random.required_worker_inputs(),
            WorkerInputs::NONE
        );
    }

    #[tokio::test]
    async fn builtin_host_does_not_construct_kv_capabilities() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("builtin-capabilities".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let client = endpoint.client().await.unwrap();
        let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let host = RoutingHost::<DefaultWorkerSelector>::new_builtin(inner).unwrap();

        assert_eq!(host.required_worker_inputs(), WorkerInputs::NONE);

        drop(host);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn terminal_item_does_not_skip_transport_eof() {
        let (router, runtime) = router(None).await;
        let inputs = router.required_worker_inputs();
        assert!(inputs.contains(WorkerInputs::CACHE));
        assert!(inputs.contains(WorkerInputs::LOAD));
        let context = Context::new(()).context();
        let drained = Arc::new(AtomicBool::new(false));
        let source_drained = Arc::clone(&drained);
        let source = ResponseStream::new(
            Box::pin(async_stream::stream! {
                yield Annotated::from_data(LLMEngineOutput {
                    finish_reason: Some(FinishReason::Stop),
                    ..Default::default()
                });
                source_drained.store(true, Ordering::Release);
            }),
            Arc::clone(&context),
        );
        let guard = RequestGuard::new_kv(
            Arc::clone(router.kv_router()),
            Arc::clone(&router.request_metrics),
            "terminal-drain".to_string(),
            WorkerWithDpRank::from_worker_id(0),
            &request(),
            false,
        );
        let monitored = monitor_response_stream(source, context, guard);
        tokio::pin!(monitored);

        assert!(monitored.next().await.is_some());
        assert!(monitored.next().await.is_none());
        assert!(drained.load(Ordering::Acquire));

        drop(router);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn stream_failure_releases_booking_before_error_is_observable() {
        let (router, runtime) = router(None).await;
        let context_id = "stream-failure-cleanup".to_string();
        let failed_request =
            Context::with_id_and_metadata(request(), context_id.clone(), Default::default());
        let (mut failed_selection, _) = router
            .select_with_affinity(&failed_request, RequestPhase::Aggregated, false)
            .await
            .unwrap();
        let failed_worker = failed_selection.worker;
        let failed_guard = router
            .track_selection(&failed_request, &mut failed_selection, false)
            .await
            .unwrap();
        let failure = Annotated {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: None,
            error: Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("selected worker is overloaded")
                    .build(),
            ),
        };
        let source = ResponseStream::new(
            Box::pin(stream::once(async move { failure })),
            failed_request.context().clone(),
        );
        let monitored =
            monitor_response_stream(source, failed_request.context().clone(), failed_guard);
        tokio::pin!(monitored);

        let item = monitored.next().await.expect("failed item must be yielded");
        assert!(item.error.is_some());

        // The monitored stream is still suspended at its yield point. Rebooking
        // the same id on the same worker proves cleanup completed before the
        // failure became visible, rather than relying on EOF or Drop cleanup.
        let retry_request =
            Context::with_id_and_metadata(request(), context_id.clone(), Default::default());
        let (mut retry_selection, _) = router
            .select_with_affinity(&retry_request, RequestPhase::Aggregated, false)
            .await
            .unwrap();
        assert_eq!(retry_selection.worker, failed_worker);
        let mut retry_guard = router
            .track_selection(&retry_request, &mut retry_selection, false)
            .await
            .expect("same-worker booking must be released before yielding the error");
        retry_guard.abort().await;

        drop(router);
        runtime.shutdown();
    }

    async fn router(session_affinity_ttl: Option<Duration>) -> (RoutingHost, Runtime) {
        router_with_workers(session_affinity_ttl, &[7]).await
    }

    async fn router_with_workers(
        session_affinity_ttl: Option<Duration>,
        worker_ids: &[u64],
    ) -> (RoutingHost, Runtime) {
        let workers = worker_ids
            .iter()
            .copied()
            .map(|worker_id| (worker_id, ModelRuntimeConfig::default()))
            .collect();
        router_with_worker_configs(session_affinity_ttl, workers).await
    }

    async fn router_with_worker_configs(
        session_affinity_ttl: Option<Duration>,
        workers: HashMap<u64, ModelRuntimeConfig>,
    ) -> (RoutingHost, Runtime) {
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
            None,
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
        let router = RoutingHost::new(inner, Arc::new(chooser), session_affinity_ttl).unwrap();
        (router, runtime)
    }

    async fn track_request(
        router: &RoutingHost,
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

        let mut failed_input = request();
        failed_input.migration_state = Some(Default::default());
        let migration_state = failed_input.migration_state.clone().unwrap();
        let failed_request = Context::new(failed_input);
        let (mut failed_selection, _) = router
            .select_with_affinity(&failed_request, RequestPhase::Aggregated, false)
            .await
            .unwrap();
        let failed_worker = failed_selection.worker.worker_id;
        let failed_dispatch_guard = router
            .track_selection(&failed_request, &mut failed_selection, false)
            .await
            .unwrap();
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
        assert_eq!(migration_state.excluded_worker_ids(), vec![failed_worker]);
        assert_eq!(metrics.requests_started_total().get(), started_before + 2);
        assert_eq!(metrics.requests_total.get(), completed_before);

        let (_, _, mut completed_guard) = track_request(&router, false).await;
        completed_guard.start_dispatch("aggregated");
        completed_guard.mark_dispatched();
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

    #[tokio::test]
    async fn query_affinity_worker_returns_existing_binding_without_reserving() {
        let (router, runtime) = router(Some(Duration::from_secs(10))).await;
        let session_id = SessionAffinityId::new("query-existing-binding");
        let target = AffinityTarget {
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
        drop(initializer.commit(target).unwrap());

        let mut request = Context::new(request());
        request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());

        assert_eq!(
            router
                .query_affinity_worker(&request, RequestPhase::Prefill)
                .unwrap(),
            Some(WorkerWithDpRank::new(7, 0))
        );
        assert_eq!(
            router
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&session_id, None)
                .unwrap(),
            Some(target)
        );

        drop(router);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn migration_exclusion_rebinds_affinity_without_widening_or_escaping_hard_pins() {
        let mut constrained_worker = ModelRuntimeConfig::default();
        constrained_worker.taints.insert("retry-pool".to_string());
        let workers = HashMap::from([
            (7, constrained_worker),
            (8, ModelRuntimeConfig::default()),
            (9, ModelRuntimeConfig::default()),
        ]);
        let (router, runtime) =
            router_with_worker_configs(Some(Duration::from_secs(10)), workers).await;
        let session_id = SessionAffinityId::new("migration-exclusion");
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

        let mut retry_input = request();
        retry_input.routing_mut().allowed_worker_ids = Some(HashSet::from([7, 8]));
        retry_input.migration_state = Some(Default::default());
        retry_input
            .migration_state
            .as_ref()
            .unwrap()
            .record_failure(
                7,
                Some(
                    DynamoError::builder()
                        .error_type(ErrorType::WorkerOverloaded)
                        .message("worker 7 overloaded")
                        .build(),
                ),
            );
        let mut retry_request = Context::new(retry_input);
        retry_request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id);

        let (selection, operation) = router
            .select_with_affinity(&retry_request, RequestPhase::Aggregated, false)
            .await
            .unwrap();
        assert_eq!(selection.worker.worker_id, 8);
        router.kv_router().free(retry_request.id()).await.unwrap();
        drop(operation);

        let mut exhausted_input = request();
        exhausted_input.routing_mut().allowed_worker_ids = Some(HashSet::from([7, 10]));
        exhausted_input.migration_state = Some(Default::default());
        exhausted_input
            .migration_state
            .as_ref()
            .unwrap()
            .record_failure(
                7,
                Some(
                    DynamoError::builder()
                        .error_type(ErrorType::WorkerOverloaded)
                        .message("worker 7 overloaded")
                        .build(),
                ),
            );
        let exhausted_request = Context::new(exhausted_input);
        let Err(error) = router
            .select_with_affinity(&exhausted_request, RequestPhase::Aggregated, false)
            .await
        else {
            panic!("exhausting the constrained worker set must reject the retry");
        };
        assert!(match_error_chain(
            error.as_ref(),
            &[ErrorType::ResourceExhausted],
            &[]
        ));

        let mut constrained_input = request();
        constrained_input.routing_mut().routing_constraints = Some(RoutingConstraints {
            required_taints: HashSet::from(["retry-pool".to_string()]),
            ..Default::default()
        });
        constrained_input.migration_state = Some(Default::default());
        constrained_input
            .migration_state
            .as_ref()
            .unwrap()
            .record_failure(
                7,
                Some(
                    DynamoError::builder()
                        .error_type(ErrorType::WorkerOverloaded)
                        .message("worker 7 overloaded")
                        .build(),
                ),
            );
        let constrained_request = Context::new(constrained_input);
        let Err(error) = router
            .select_with_affinity(&constrained_request, RequestPhase::Aggregated, false)
            .await
        else {
            panic!("routing constraints must not be widened during retry");
        };
        assert!(match_error_chain(
            error.as_ref(),
            &[ErrorType::ResourceExhausted],
            &[]
        ));

        let mut pinned_input = request();
        let routing = pinned_input.routing_mut();
        routing.backend_instance_id = Some(7);
        routing.dp_rank = Some(0);
        pinned_input.migration_state = Some(Default::default());
        pinned_input
            .migration_state
            .as_ref()
            .unwrap()
            .record_failure(7, None);
        let pinned_request = Context::new(pinned_input);
        let (selection, _) = router
            .select_with_affinity(&pinned_request, RequestPhase::Aggregated, true)
            .await
            .unwrap();
        assert_eq!(selection.worker.worker_id, 7);

        drop(router);
        runtime.shutdown();
    }

    #[derive(Default)]
    struct RejectFirstDispatch {
        attempts: Mutex<Vec<(u64, Vec<u64>)>>,
    }

    #[async_trait]
    impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for RejectFirstDispatch {
        async fn generate(
            &self,
            request: SingleIn<AddressedRequest<PreprocessedRequest>>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let (addressed, context) = request.transfer(());
            let (request, _, instance) = addressed.into_parts();
            let worker_id = instance.expect("selected worker instance").id();
            let excluded_worker_ids = request
                .migration_state
                .as_ref()
                .map(|state| state.excluded_worker_ids())
                .unwrap_or_default();
            let attempt = {
                let mut attempts = self.attempts.lock().unwrap();
                attempts.push((worker_id, excluded_worker_ids));
                attempts.len()
            };

            if attempt == 1 {
                let output = Annotated {
                    data: None,
                    id: None,
                    event: Some("error".to_string()),
                    comment: None,
                    error: Some(
                        DynamoError::builder()
                            .error_type(ErrorType::WorkerOverloaded)
                            .message("selected worker is overloaded")
                            .build(),
                    ),
                };
                return Ok(ResponseStream::new(
                    Box::pin(stream::once(async move { output })),
                    context.context(),
                ));
            }

            let output = Annotated::from_data(LLMEngineOutput {
                token_ids: vec![2],
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            });
            Ok(ResponseStream::new(
                Box::pin(stream::once(async move { output })),
                context.context(),
            ))
        }

        async fn generate_bidirectional(
            &self,
            _instance: Instance,
            _address: String,
            _input: ManyIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            unreachable!("the KV router dispatches unary requests")
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn worker_overload_stream_migration_releases_and_reselects() {
        async fn shared_drt(runtime: Runtime, store_path: &std::path::Path) -> DistributedRuntime {
            DistributedRuntime::new(
                runtime,
                DistributedConfig {
                    discovery_backend: DiscoveryBackend::KvStore(Selector::File(
                        store_path.to_path_buf(),
                    )),
                    nats_config: None,
                    request_plane: RequestPlaneMode::Tcp,
                    event_transport_kind: EventTransportKind::Zmq,
                },
            )
            .await
            .unwrap()
        }

        let runtime = Runtime::from_current().unwrap();
        let store = tempfile::tempdir().unwrap();
        let router_drt = shared_drt(runtime.clone(), store.path()).await;
        let first_worker_drt = shared_drt(runtime.clone(), store.path()).await;
        let second_worker_drt = shared_drt(runtime.clone(), store.path()).await;
        let namespace = "worker-overload-migration";
        let endpoint_for = |drt: &DistributedRuntime| {
            drt.namespace(namespace.to_string())
                .unwrap()
                .component("workers".to_string())
                .unwrap()
                .endpoint("generate")
        };
        let first_worker_endpoint = endpoint_for(&first_worker_drt);
        let second_worker_endpoint = endpoint_for(&second_worker_drt);
        first_worker_endpoint
            .register_endpoint_instance()
            .await
            .unwrap();
        second_worker_endpoint
            .register_endpoint_instance()
            .await
            .unwrap();

        let endpoint = endpoint_for(&router_drt);
        let client = endpoint.client().await.unwrap();
        let instances = tokio::time::timeout(Duration::from_secs(5), async {
            let mut source = client.instance_source.as_ref().clone();
            loop {
                let instances = source.borrow_and_update().clone();
                if instances.len() == 2 {
                    return instances;
                }
                source.changed().await.unwrap();
            }
        })
        .await
        .expect("both workers must be discovered");
        let registered_ids = instances
            .into_iter()
            .map(|instance| instance.id())
            .collect::<HashSet<_>>();
        assert_eq!(registered_ids.len(), 2);

        let workers = registered_ids
            .iter()
            .copied()
            .map(|worker_id| (worker_id, ModelRuntimeConfig::default()))
            .collect::<HashMap<_, _>>();
        let (_workers_tx, workers) = watch::channel(workers);
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
            None,
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
        let dispatch = Arc::new(RejectFirstDispatch::default());
        let push_router =
            PushRouter::from_client_with_dispatch(client.clone(), RouterMode::KV, dispatch.clone())
                .await
                .unwrap();
        let chooser = Arc::new(chooser);
        let kv_router = Arc::new(RoutingHost::new(push_router, chooser.clone(), None).unwrap());
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            kv_router;
        let migration = Migration::new(1, None, "test".to_string(), Arc::new(Metrics::new()));

        let responses: Vec<_> = migration
            .generate(Context::new(request()), next)
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(responses.len(), 1);
        assert!(responses[0].error.is_none());
        assert_eq!(responses[0].data.as_ref().unwrap().token_ids, vec![2]);
        let attempts = {
            let attempts = dispatch.attempts.lock().unwrap();
            attempts.clone()
        };
        assert_eq!(attempts.len(), 2);
        let failed_worker = attempts[0].0;
        let retried_worker = attempts[1].0;
        assert_ne!(failed_worker, retried_worker);
        assert!(registered_ids.contains(&failed_worker));
        assert!(registered_ids.contains(&retried_worker));
        assert!(attempts[0].1.is_empty());
        assert_eq!(attempts[1].1, vec![failed_worker]);
        let loads = chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(
            loads.iter().all(|load| load.active_requests == 0),
            "all scheduler bookings must be released after migration: {loads:?}"
        );
        runtime.shutdown();
    }
}

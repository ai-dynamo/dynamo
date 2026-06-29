// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_kv_router::protocols::{TokensWithHashes, WorkerWithDpRank};
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tracing::Instrument;

use crate::{
    kv_router::{KvRouter, metrics::RouterRequestMetrics},
    preprocessor::PreprocessedRequest,
    protocols::common::{
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

use cancellation::{cancel_on_stop, cancelled_error};
use request_guard::RequestGuard;
use selection::{RoutingRequestParts, SelectionOptions, WorkerSelection};

const OUTPUT_REPLAY_ID_ANNOTATION_KEY: &str = "output_replay_id";
const OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY: &str = "output_replay_consumer";

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
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

        // Eagerly register router request metrics (as zeros) so they are
        // scrapeable before any requests arrive. Both the frontend pipeline
        // and the standalone router create KvPushRouter, so this covers both.
        RouterRequestMetrics::from_component(chooser.client().endpoint.component());

        Ok(KvPushRouter {
            inner,
            chooser,
            affinity,
        })
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
        let routing_parts = RoutingRequestParts::new(request);
        let request_context = request.context().clone();
        let mut selection_future = Box::pin(async {
            self.select_worker(
                &context_id,
                request,
                routing_parts,
                phase,
                is_query_only,
                SelectionOptions {
                    affinity_worker,
                    policy_class,
                },
            )
            .instrument(tracing::info_span!("kv_router.select_worker"))
            .await
        });
        let selection_result = tokio::select! {
            biased;

            _ = request_context.stopped() => None,
            result = &mut selection_future => Some(result),
        };
        drop(selection_future);

        match selection_result {
            Some(result) => result,
            None => {
                if !is_query_only && let Err(error) = self.chooser.free(&context_id).await {
                    tracing::warn!(
                        request_id = %context_id,
                        %error,
                        "Failed to free scheduler state after cancellation during worker selection"
                    );
                }
                Err(cancelled_error(&context_id))
            }
        }
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
            Err(error) if match_error_chain(error.as_ref(), &[ErrorType::Cancelled], &[]) => {
                Err(error)
            }
            Err(_) if operation.target().is_some() && explicit.is_none() => {
                operation.invalidate();
                let retry = affinity
                    .acquire_with_context(&session_id, None, request_context.as_ref())
                    .await?;
                match self.select_request(request, phase, false, None).await {
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
            context_id.clone(),
            request,
            selection.scheduler_tracked,
        );

        let record_result: Result<(), Error> = async {
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
        let mut response_stream = match dispatch_result {
            Ok(stream) => stream,
            Err(error) => {
                guard.abort().await;
                return Err(error);
            }
        };

        guard.mark_dispatched();
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();
        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut guard = guard;

            loop {
                tokio::select! {
                    biased;

                    _ = context_for_monitoring.stopped() => {
                        tracing::debug!("Request {context_id} cancelled, ending stream");
                        break;
                    }

                    item = response_stream.next() => {
                        let Some(item) = item else {
                            break;
                        };
                        guard.on_item(&item).await;
                        yield item;
                    }
                }
            }

            guard.finish().await;
        });
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
        F: FnOnce(&mut PreprocessedRequest, u64, Option<u32>) -> Result<M, Error>,
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
                if let Some(operation) = operation.take() {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        let metadata = match prepare(&mut request, selection.instance_id, Some(selection.dp_rank)) {
            Ok(metadata) => metadata,
            Err(error) => {
                guard.abort().await;
                if let Some(operation) = operation.take() {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        let selected_target = AffinityTarget {
            worker_id: selection.instance_id,
            dp_rank: Some(selection.dp_rank),
        };
        drop(route_guard);
        let stream = match self
            .dispatch_selection(request, selection, guard, true)
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                if let Some(operation) = operation.take() {
                    operation.invalidate();
                }
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
            RouterRequestMetrics::from_component(self.chooser.client().endpoint.component())
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
                if let Some(operation) = operation.take() {
                    operation.invalidate();
                }
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
                if let Some(operation) = operation.take() {
                    operation.invalidate();
                }
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
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use dynamo_kv_router::{DefaultWorkerSelector, config::KvRouterConfig};
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
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
        )
        .await
        .unwrap();
        let inner = PushRouter::from_client(client, RouterMode::KV)
            .await
            .unwrap();
        let router = KvPushRouter::new(inner, Arc::new(chooser), session_affinity_ttl).unwrap();
        (router, runtime)
    }

    #[tokio::test]
    async fn session_affinity_disabled_does_not_create_coordinator() {
        let (router, runtime) = router(None).await;
        assert!(router.affinity.is_none());

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

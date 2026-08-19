// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::{
    protocols::{TokensWithHashes, WorkerWithDpRank},
    selector::WorkerSelector,
};
use dynamo_runtime::{
    error::DynamoError,
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn},
    protocols::annotated::Annotated,
};
use futures::stream;
use tracing::Instrument;

use crate::{
    kv_router::to_worker_selection_session_context,
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RoutingData},
    },
    session_affinity::{AffinityAcquire, AffinityTarget, affinity_id, explicit_target},
};

use super::{
    OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY, OUTPUT_REPLAY_ID_ANNOTATION_KEY, RoutingHost,
    affinity_worker, cancel_on_stop, into_monitored_response, invalidate_on_non_cancellation,
    is_cancelled,
    kv_selection::{RoutingRequestParts, SelectionOptions, WorkerSelection},
    request_guard::RequestGuard,
    route_target,
};

impl<Sel> RoutingHost<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
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

    pub(super) async fn select_with_affinity(
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

    pub(super) async fn track_selection(
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

    pub(super) async fn dispatch_selection(
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
                self.push_router
                    .dispatch_exact(updated_request, selection.worker.worker_id)
                    .await
            } else {
                self.push_router
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

    pub(super) async fn select_and_dispatch_kv_prefill<M, F>(
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

    /// Selects and dispatches a KV-aware request, including query-only requests,
    /// explicit pins, state tracking, and affinity cleanup.
    pub(super) async fn generate_kv(
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

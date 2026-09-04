// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::kv_router::{FindBestMatchAdmission, routing_host::kv_selection::SelectionOutcome};

impl<Sel> RoutingHost<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    async fn select_request_outcome(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        affinity_target: Option<AffinityTarget>,
        planned_worker: Option<WorkerWithDpRank>,
        admission: FindBestMatchAdmission,
    ) -> Result<SelectionOutcome, Error> {
        let context_id = request.context().id().to_string();
        let policy_class = request.metadata().get("policy-class").cloned();
        let session_context = request
            .agent_context
            .as_ref()
            .map(to_worker_selection_session_context);
        let routing_parts = RoutingRequestParts::new(request);
        let request_context = request.context().clone();
        let selection_future = self
            .select_worker_outcome(
                &context_id,
                request,
                routing_parts,
                phase,
                is_query_only,
                SelectionOptions {
                    pinned_target: match self.session_affinity_mode {
                        SessionAffinityMode::Hard => affinity_target,
                        SessionAffinityMode::Soft => None,
                    },
                    affinity_target: match self.session_affinity_mode {
                        SessionAffinityMode::Hard => None,
                        SessionAffinityMode::Soft => affinity_target,
                    },
                    planned_worker,
                    policy_class,
                    session_context,
                    admission,
                },
            )
            .instrument(tracing::info_span!("kv_router.select_worker"));

        cancel_on_stop(request_context.as_ref(), selection_future).await?
    }

    async fn select_request(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        affinity_target: Option<AffinityTarget>,
    ) -> Result<WorkerSelection, Error> {
        self.select_request_outcome(
            request,
            phase,
            is_query_only,
            affinity_target,
            None,
            FindBestMatchAdmission::WithAdmission {
                track_lifecycle: true,
            },
        )
        .await?
        .into_result()
    }

    pub(super) async fn select_with_affinity(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
    ) -> Result<(WorkerSelection, Option<AffinityAcquire>), Error> {
        self.select_with_session_affinity(request, phase, is_query_only, |target| {
            self.select_request(request, phase, is_query_only, target)
        })
        .await
    }

    fn route_signals(&self, selection: &WorkerSelection) -> RoutePlanSignals {
        let total_kv_blocks = match selection.selected_worker_load {
            Some(load) => load
                .total_kv_blocks
                .and_then(|blocks| blocks.try_into().ok()),
            None => self
                .kv_router()
                .workers_with_configs
                .borrow()
                .get(&selection.worker.worker_id)
                .and_then(WorkerConfigLike::total_kv_blocks),
        };
        RoutePlanSignals {
            worker: selection.worker,
            overlap_blocks: selection.overlap_amount,
            cached_tokens: selection.cached_tokens,
            potential_decode_blocks: selection.potential_decode_blocks,
            total_kv_blocks,
        }
    }

    pub(crate) async fn preview_kv_route(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
    ) -> Result<RoutePreview, Error> {
        if self.kv_router_if_enabled().is_none() {
            return Err(anyhow::anyhow!("KV route previews require KV routing"));
        }

        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let (outcome, _) = self
            .select_with_session_affinity(request, phase, true, |target| {
                self.select_request_outcome(
                    request,
                    phase,
                    true,
                    target,
                    None,
                    FindBestMatchAdmission::WithoutAdmission,
                )
            })
            .await?;
        let selection = outcome.into_result()?;
        let signals = self.route_signals(&selection);
        drop(route_guard);
        Ok(RoutePreview {
            request_id: request.context().id().to_string(),
            phase,
            signals,
        })
    }

    pub(crate) async fn plan_kv_route_from_preview(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        preview: RoutePreview,
    ) -> Result<RoutePlan<Sel>, Error> {
        if self.kv_router_if_enabled().is_none() {
            return Err(anyhow::anyhow!("KV route plans require KV routing"));
        }
        if request.context().id() != preview.request_id {
            return Err(anyhow::anyhow!(
                "KV route preview belongs to request {}, not {}",
                preview.request_id,
                request.context().id(),
            ));
        }

        let phase = preview.phase;
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let planned_worker = preview.signals.worker;
        let (selection, affinity) = self
            .select_with_session_affinity(request, phase, false, |target| async move {
                self.select_request_outcome(
                    request,
                    phase,
                    false,
                    target,
                    Some(planned_worker),
                    FindBestMatchAdmission::WithAdmission {
                        track_lifecycle: true,
                    },
                )
                .await?
                .into_result()
            })
            .await?;
        let signals = self.route_signals(&selection);
        drop(route_guard);
        Ok(RoutePlan {
            signals,
            cleanup: KvRequestCleanup::new(
                Arc::clone(self.kv_router()),
                request.context().id().to_string(),
                selection.worker,
                selection.attempt,
            ),
            selection,
            affinity,
        })
    }

    pub(crate) async fn dispatch_kv_plan(
        &self,
        request: SingleIn<PreprocessedRequest>,
        plan: RoutePlan<Sel>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let RoutePlan {
            mut selection,
            cleanup,
            mut affinity,
            ..
        } = plan;
        let selected_target = route_target(selection.worker);
        let guard = match self
            .track_planned_selection(&request, &mut selection, cleanup)
            .await
        {
            Ok(guard) => guard,
            Err(error) => return Err(error),
        };
        let stream = match self.dispatch_selection(request, selection, guard).await {
            Ok(stream) => stream,
            Err(error) => {
                if self.session_affinity_mode == SessionAffinityMode::Hard
                    && !self.affinity_target_is_valid(selected_target)
                    && let Some(operation) = affinity.take()
                {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        match affinity {
            Some(affinity) => {
                affinity.into_stream(selected_target, stream, self.session_affinity_mode)
            }
            None => Ok(stream),
        }
    }

    pub(crate) async fn prefill_worker_busy(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        threshold: f64,
    ) -> Result<bool, Error> {
        if self.kv_router_if_enabled().is_none() {
            return Err(anyhow::anyhow!("prefill load probe requires KV routing"));
        }

        let (outcome, _) = self
            .select_with_session_affinity(request, RequestPhase::Prefill, true, |target| {
                self.select_request_outcome(
                    request,
                    RequestPhase::Prefill,
                    true,
                    target,
                    None,
                    FindBestMatchAdmission::WithoutAdmission,
                )
            })
            .await?;
        match outcome {
            SelectionOutcome::Routed(selection) => selection
                .selected_worker_load
                .map(|load| load.prefill_load_exceeds(threshold))
                .ok_or_else(|| anyhow::anyhow!("advisory prefill selection returned no load")),
            SelectionOutcome::QueueRejected(_) => Ok(true),
        }
    }

    pub(super) async fn track_selection(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        is_query_only: bool,
    ) -> Result<RequestGuard<Sel>, Error> {
        self.track_selection_with_cleanup(request, selection, is_query_only, None)
            .await
    }

    async fn track_planned_selection(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        cleanup: KvRequestCleanup<Sel>,
    ) -> Result<RequestGuard<Sel>, Error> {
        self.track_selection_with_cleanup(request, selection, false, Some(cleanup))
            .await
    }

    async fn track_selection_with_cleanup(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        is_query_only: bool,
        cleanup: Option<KvRequestCleanup<Sel>>,
    ) -> Result<RequestGuard<Sel>, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let routing_parts = RoutingRequestParts::new(request);
        let chooser = self.kv_router();
        let block_size = chooser.block_size() as usize;
        let selected_worker = selection.worker;
        let mut guard = match cleanup {
            Some(cleanup) => {
                RequestGuard::new_kv_with_cleanup(self.request_metrics.clone(), cleanup, request)
            }
            None => RequestGuard::new_kv(
                Arc::clone(chooser),
                self.request_metrics.clone(),
                context_id.clone(),
                selected_worker,
                selection.attempt,
                request,
            ),
        };

        let record_result: Result<(), Error> = async {
            if !is_query_only && chooser.indexer().records_routing_decisions() {
                let worker = selected_worker;
                let hashes = if let Some(hashes) = selection.routing_hashes.take() {
                    hashes
                } else {
                    let routing = request.routing.as_ref();
                    let mut tokens_with_hashes = TokensWithHashes::new(
                        routing_parts.token_ids.to_vec(),
                        chooser.block_size(),
                    )
                    .with_is_eagle(chooser.is_eagle());
                    if let Some(infos) = routing_parts.block_mm_infos {
                        tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                    }
                    if let Some(lora_name) = routing.and_then(|r| r.lora_name.clone()) {
                        tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                    }
                    if let Some(cache_namespace) = routing.and_then(|r| r.cache_namespace.clone()) {
                        tokens_with_hashes =
                            tokens_with_hashes.with_cache_namespace(cache_namespace);
                    }
                    let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
                    let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
                    dynamo_kv_router::indexer::RoutingDecisionHashes {
                        local_hashes,
                        sequence_hashes,
                    }
                };
                let record_result = if guard.has_approximate_lru() {
                    cancel_on_stop(
                        request_context.as_ref(),
                        guard.acquire_approximate_lru(hashes),
                    )
                    .await?
                } else {
                    cancel_on_stop(
                        request_context.as_ref(),
                        chooser.record_routing_decision_hashes(hashes, worker),
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
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let route_trace_context = get_route_trace_context(&request);
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

        let dispatch = self
            .inner
            .dispatch_kv_admitted(updated_request, selection.worker.worker_id);
        let route_span = tracing::info_span!(
            target: "request_span",
            "kv_router.route_request",
            otel.kind = "client",
            request_id = %context_id,
            worker_id = tracing::field::Empty,
            dp_rank = selection.worker.dp_rank,
            overlap_blocks = selection.overlap_amount,
            phase = ?phase,
            "request.attempt" = tracing::field::Empty,
            "request.outcome" = tracing::field::Empty,
            "migration.is_retry" = tracing::field::Empty,
            "migration.reason" = tracing::field::Empty,
            "migration.from_worker_id" = tracing::field::Empty,
            "migration.tokens_completed" = tracing::field::Empty,
            "cancellation.signal" = tracing::field::Empty,
            "error.type" = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
            otel.status_description = tracing::field::Empty,
        );
        record_route_span_start(
            &route_span,
            route_trace_context.as_deref(),
            selection.worker.worker_id,
        );
        let dispatch_result = cancel_on_stop(
            request_context.as_ref(),
            dispatch.instrument(route_span.clone()),
        )
        .await
        .and_then(|result| result);
        let response_stream = match dispatch_result {
            Ok(stream) => stream,
            Err(error) => {
                record_route_error(&route_span, error.as_ref());
                let typed_error = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<DynamoError>().cloned());
                guard.record_migration_failure(typed_error);
                guard.abort().await;
                return Err(error);
            }
        };

        guard.mark_dispatched();
        Ok(wrap_route_span(
            into_monitored_response(response_stream, guard),
            route_span,
        ))
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
            Err(error) => return Err(error),
        };
        let selected_target = route_target(selection.worker);
        let metadata = match prepare(&mut request, selected_target) {
            Ok(metadata) => metadata,
            Err(error) => {
                guard.abort().await;
                return Err(error);
            }
        };
        drop(route_guard);
        let stream = match self.dispatch_selection(request, selection, guard).await {
            Ok(stream) => stream,
            Err(error) => {
                if self.session_affinity_mode == SessionAffinityMode::Hard
                    && !self.affinity_target_is_valid(selected_target)
                    && let Some(operation) = operation.take()
                {
                    operation.invalidate();
                }
                return Err(error);
            }
        };
        let Some(operation) = operation else {
            return Ok((metadata, stream));
        };
        Ok((
            metadata,
            operation.into_stream(selected_target, stream, self.session_affinity_mode)?,
        ))
    }
}

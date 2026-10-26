// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The dispatch half of the request lifecycle, shared by every routing policy.
//!
//! Selection, the worker source and the booking stay per policy. Once a policy
//! has a worker and a [`RequestGuard`], it hands a [`DispatchPlan`] to
//! [`RoutingHost::dispatch_plan`], which owns everything up to the monitored
//! response stream. Only the transport call differs: [`PlannedRoute`] says which.

use dynamo_kv_router::kv_hints::KvHint;

use super::*;
use crate::protocols::common::timing::RequestTracker;

/// A selected route that the host is about to send.
pub(super) struct DispatchPlan {
    pub(super) guard: RequestGuard,
    pub(super) affinity: Option<Hold>,
    pub(super) phase: RequestPhase,
    pub(super) route: PlannedRoute,
}

pub(super) enum PlannedRoute {
    Kv(KvAdmission),
    Hosted(HostedDispatch),
}

/// A worker and DP rank admitted by the KV scheduler. Dispatch is exact: the
/// booking is for this worker, so there is no transport fallback.
pub(super) struct KvAdmission {
    pub(super) worker: WorkerWithDpRank,
    pub(super) kv_hint: Option<KvHint>,
    pub(super) overlap_blocks: u32,
    pub(super) worker_type: &'static str,
}

/// A worker picked by a built-in policy. `PushRouter` may still move the
/// request to another worker if the transport to this one fails.
pub(super) struct HostedDispatch {
    pub(super) transport: HostedTransport,
    pub(super) initial_worker: u64,
    /// The DP rank a soft session binding asked for; it applies to whichever
    /// worker transport resolves to.
    pub(super) soft_dp_rank: Option<u32>,
    pub(super) uses_occupancy: bool,
    pub(super) candidate_count: usize,
    pub(super) selected_occupancy: Option<u64>,
    pub(super) device_aware_telemetry: Option<DeviceAwareTelemetry>,
}

pub(super) enum HostedTransport {
    /// Direct mode without a session: the request-supplied target, keeping its
    /// DP rank only if transport stays on that worker.
    Direct(AffinityTarget),
    /// An exact target from the request or a hard session binding.
    Exact(AffinityTarget),
    /// An occupancy policy's reserved pick.
    Occupancy,
    /// A stateless pick, optionally limited to LoRA-eligible fallbacks.
    Within(Option<HashSet<u64>>),
}

impl HostedDispatch {
    fn target_for_worker(&self, worker_id: u64) -> AffinityTarget {
        AffinityTarget::new(worker_id, self.soft_dp_rank)
    }

    /// The target this dispatch is expected to reach, before any transport
    /// fallback.
    fn expected_target(&self) -> AffinityTarget {
        match &self.transport {
            HostedTransport::Direct(target) | HostedTransport::Exact(target) => *target,
            HostedTransport::Occupancy | HostedTransport::Within(_) => {
                self.target_for_worker(self.initial_worker)
            }
        }
    }
}

impl PlannedRoute {
    fn expected_target(&self) -> AffinityTarget {
        match self {
            Self::Kv(admission) => route_target(admission.worker),
            Self::Hosted(hosted) => hosted.expected_target(),
        }
    }

    fn dispatch_stage(&self) -> &'static str {
        match self {
            Self::Kv(_) => "kv.dispatch",
            Self::Hosted(hosted) => match hosted.transport {
                HostedTransport::Direct(_) => "builtin.dispatch_direct",
                HostedTransport::Exact(_) => "builtin.dispatch_exact",
                HostedTransport::Occupancy => "builtin.dispatch_occupancy",
                HostedTransport::Within(_) => "builtin.dispatch",
            },
        }
    }

    fn worker_type(&self, tracker: &RequestTracker) -> &'static str {
        match self {
            Self::Kv(admission) => admission.worker_type,
            Self::Hosted(_) if tracker.phase() == RequestPhase::Prefill => WORKER_TYPE_PREFILL,
            Self::Hosted(_) => WORKER_TYPE_DECODE,
        }
    }

    /// The route's client span.
    ///
    /// The host opens `kv_router.route_request` for KV routes. Hosted routes
    /// get `router.route_request` from `PushRouter`, which opens it only after
    /// transport has resolved the final worker; a span opened here would name
    /// the selected worker even when fallback moved the request elsewhere.
    fn route_span(&self, request_id: &str, phase: RequestPhase) -> tracing::Span {
        let Self::Kv(admission) = self else {
            return tracing::Span::none();
        };
        tracing::info_span!(
            target: "request_span",
            "kv_router.route_request",
            otel.kind = "client",
            request_id = %request_id,
            worker_id = tracing::field::Empty,
            dp_rank = admission.worker.dp_rank,
            overlap_blocks = admission.overlap_blocks,
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
        )
    }
}

impl RoutingHost {
    /// Send a selected route and hand back its monitored response stream.
    ///
    /// `prepare` runs once the final worker is known, and its metadata is
    /// returned with the stream. On failure the guard is released here: a
    /// migratable error hands a classifier lifecycle to the retry, and anything
    /// else aborts it. Only KV routes carry a classifier lifecycle, so a built-in
    /// route always aborts, which releases its occupancy reservation.
    pub(super) async fn dispatch_plan<M, F>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        plan: DispatchPlan,
        budget: &CleanupBudget,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        let DispatchPlan {
            mut guard,
            mut affinity,
            phase,
            route,
        } = plan;
        let request_context = request.context().clone();
        let staged_kv = StagedKv::for_request(request.content());
        let tracker = request.tracker.clone();
        let expected_target = route.expected_target();
        let stage = route.dispatch_stage();
        let worker_type = tracker.as_deref().map(|tracker| route.worker_type(tracker));
        let route_span = route.route_span(request_context.id(), phase);
        if let PlannedRoute::Kv(admission) = &route {
            record_route_span_start(
                &route_span,
                get_route_trace_context(&request).as_deref(),
                admission.worker.worker_id,
            );
        }

        self.request_metrics
            .input_sequence_tokens
            .with_label_values(&[request.phase().as_str(), &request.model])
            .observe(request.token_ids.len() as f64);
        guard.start_dispatch(&phase.to_string());
        guard.record_prefill_start(request.content());

        let guard_ref = &mut guard;
        let dispatch = async move {
            match route {
                PlannedRoute::Kv(admission) => {
                    let target = route_target(admission.worker);
                    request.routing_mut().dp_rank = Some(admission.worker.dp_rank);
                    request.kv_hint = admission.kv_hint;
                    let metadata = prepare(&mut request, target)?;
                    let stream = self
                        .inner
                        .dispatch_kv_admitted(request, admission.worker.worker_id)
                        .await?;
                    Ok((metadata, target, stream))
                }
                PlannedRoute::Hosted(hosted) => {
                    self.dispatch_hosted(request, hosted, guard_ref, prepare)
                        .await
                }
            }
        };
        let dispatch_result = await_with_cleanup_policy(
            request_context.as_ref(),
            phase,
            staged_kv,
            stage,
            budget,
            dispatch.instrument(route_span.clone()),
        )
        .await
        .and_then(|result| result);

        let (metadata, target, response_stream) = match dispatch_result {
            Ok(dispatched) => dispatched,
            Err(error) => {
                record_route_error(&route_span, error.as_ref());
                if self.session_affinity_mode == SessionAffinityMode::Hard
                    && !self.affinity_target_is_valid(expected_target)
                    && let Some(operation) = affinity.take()
                {
                    operation.invalidate();
                }
                let typed_error = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<DynamoError>().cloned());
                guard.record_migration_failure(typed_error);
                if !crate::migration::is_migratable(error.as_ref())
                    || !guard.release_for_retry().await
                {
                    guard.abort_with_error(Some(error.as_ref())).await;
                }
                return Err(error);
            }
        };

        // Recorded after dispatch, not at selection: the tracker keeps the first
        // worker it sees, and hosted transport can move the request, so only the
        // worker that took the request belongs there.
        if let (Some(tracker), Some(worker_type)) = (tracker, worker_type) {
            tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
        }
        guard.mark_dispatched();
        let stream = wrap_route_span(into_monitored_response(response_stream, guard), route_span);
        Ok((metadata, self.bind_affinity(affinity, target, stream)?))
    }

    /// The built-in transport call. `PushRouter` may resolve a different worker
    /// than the one selected; the guard follows it.
    async fn dispatch_hosted<M, F>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        hosted: HostedDispatch,
        guard: &mut RequestGuard,
        prepare: F,
    ) -> Result<(M, AffinityTarget, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        let (metadata, target, final_occupancy, response_stream) = match &hosted.transport {
            HostedTransport::Direct(target) => {
                let target = *target;
                self.inner
                    .direct_within_prepared(
                        request,
                        target.worker_id,
                        None,
                        |request, worker_id| {
                            let occupancy = guard.retarget_worker(worker_id);
                            let target = AffinityTarget::new(
                                worker_id,
                                target.dp_rank.filter(|_| worker_id == target.worker_id),
                            );
                            request.routing_mut().dp_rank = target.dp_rank;
                            prepare(request, target).map(|metadata| (metadata, target, occupancy))
                        },
                    )
                    .await
                    .map(|((metadata, target, occupancy), stream)| {
                        (metadata, target, occupancy, stream)
                    })?
            }
            HostedTransport::Exact(target) => {
                let target = *target;
                request.routing_mut().dp_rank = target.dp_rank;
                let metadata = prepare(&mut request, target)?;
                let stream = self.inner.dispatch_exact(request, target.worker_id).await?;
                (metadata, target, hosted.selected_occupancy, stream)
            }
            HostedTransport::Occupancy => self
                .inner
                .dispatch_preselected_prepared(
                    request,
                    hosted.initial_worker,
                    |request, worker_id| {
                        let occupancy = guard.retarget_worker(worker_id);
                        let target = hosted.target_for_worker(worker_id);
                        request.routing_mut().dp_rank = target.dp_rank;
                        prepare(request, target).map(|metadata| (metadata, target, occupancy))
                    },
                )
                .await
                .map(|((metadata, target, occupancy), stream)| {
                    (metadata, target, occupancy, stream)
                })?,
            HostedTransport::Within(fallback) => self
                .inner
                .direct_within_prepared(
                    request,
                    hosted.initial_worker,
                    fallback.as_ref(),
                    |request, worker_id| {
                        let occupancy = guard.retarget_worker(worker_id);
                        let target = hosted.target_for_worker(worker_id);
                        request.routing_mut().dp_rank = target.dp_rank;
                        prepare(request, target).map(|metadata| (metadata, target, occupancy))
                    },
                )
                .await
                .map(|((metadata, target, occupancy), stream)| {
                    (metadata, target, occupancy, stream)
                })?,
        };
        guard.retarget_worker(target.worker_id);
        if let Some(telemetry) = &hosted.device_aware_telemetry {
            let selection_survived_transport = target.worker_id == hosted.initial_worker;
            tracing::info!(
                router_mode = "device-aware-weighted",
                worker_id = target.worker_id,
                candidate_count = hosted.candidate_count,
                occupancy = ?final_occupancy,
                endpoint = %self.inner.client.endpoint.id(),
                is_cpu = ?selection_survived_transport.then_some(telemetry.is_cpu),
                embedding_cache_hit = ?selection_survived_transport
                    .then_some(telemetry.embedding_cache_hit),
                request_cache_keys = telemetry.request_cache_keys,
                transport_fallback = !selection_survived_transport,
                "Selected worker"
            );
        } else if hosted.uses_occupancy {
            tracing::info!(
                router_mode = self.inner.router_mode().telemetry_label(),
                worker_id = target.worker_id,
                candidate_count = hosted.candidate_count,
                occupancy = ?final_occupancy,
                transport_fallback = target.worker_id != hosted.initial_worker,
                "Selected worker"
            );
        }
        Ok((metadata, target, response_stream))
    }
}

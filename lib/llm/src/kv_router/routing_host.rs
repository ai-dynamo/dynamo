// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    future::{Future, ready},
    sync::Arc,
    time::Duration,
};

use dynamo_kv_router::{
    protocols::{TokensWithHashes, WorkerConfigLike, WorkerWithDpRank},
    scheduling::{AbortCause, KvSchedulerError},
    selector::{WorkerInputs, WorkerSelector},
};
use dynamo_runtime::{
    error::{DynamoError, ErrorType, match_error_chain},
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        ResponseStream, SingleIn, async_trait,
        network::egress::route_span::{
            get_route_trace_context, record_route_error, record_route_span_start, wrap_route_span,
        },
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tracing::Instrument;

use crate::{
    kv_router::{
        KvRouter, metrics::RouterRequestMetrics, request_lease::RequestAttemptRelease,
        to_worker_selection_session_context,
    },
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    protocols::common::{
        FinishReason,
        extensions::SessionAffinityId,
        llm_backend::LLMEngineOutput,
        preprocessor::owned_abort_error,
        timing::{RequestPhase, RoutingData, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
    },
    session_affinity::{
        AffinityCoordinator, AffinityTarget, Hold, SessionAffinityMode, affinity_id,
        explicit_target, from_table, invalid_argument, subagent_group_affinity_id,
    },
};

mod builder;
mod builtin;
mod cache_read;
mod cancellation;
mod kv;
mod kv_selection;
mod occupancy;
mod request_guard;

pub use builder::{LlmDispatch, MultimodalCache, RoutingHostBuilder};
use builtin::BuiltinWorkerSelector;
use cancellation::{CleanupBudget, DispatchCancellation, StagedKv, await_with_cleanup_policy};
use kv_selection::{RoutingRequestParts, SelectionOptions, WorkerSelection};
use occupancy::HostedOccupancy;
pub(crate) use request_guard::prompt_private_blocks;
use request_guard::{KvRequestCleanup, LoraLoadGuard, RequestGuard};

const OUTPUT_REPLAY_ID_ANNOTATION_KEY: &str = "output_replay_id";
const OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY: &str = "output_replay_consumer";

/// Bounds the wait for a worker's trailing typed error after a terminal frame.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) struct PrefillDispatch<M> {
    pub(crate) metadata: M,
    pub(crate) stream: ManyOut<Annotated<LLMEngineOutput>>,
    /// Early release for the prefill's scheduler booking, held by `stream`
    /// until prefill finishes. `None` when the routing policy books nothing.
    pub(crate) booking: Option<RequestAttemptRelease>,
}

pub(crate) fn is_cancelled(error: &Error) -> bool {
    match_error_chain(error.as_ref(), &[ErrorType::Cancelled], &[])
}

fn classification_failure(error: &Error) -> Option<&KvSchedulerError> {
    error.chain().find_map(|cause| {
        let error = cause.downcast_ref::<KvSchedulerError>()?;
        matches!(
            error,
            KvSchedulerError::RequestClassifierPanicked(_)
                | KvSchedulerError::RequestClassifierFailed(_)
                | KvSchedulerError::InvalidClassificationMetadata(_)
        )
        .then_some(error)
    })
}

fn classifier_abort_error(error: &KvSchedulerError) -> Arc<AbortCause> {
    match error {
        KvSchedulerError::RequestClassifierFailed(source) => Arc::clone(source),
        _ => owned_abort_error(error),
    }
}

/// The client-facing error for a classifier failure. A typed [`DynamoError`]
/// returned by the plugin is an intentional, client-visible decision (flow
/// control) and passes through with its status; everything else is sanitized
/// to hide classifier internals from the client and logged for the operator.
fn classifier_failure_response(request_id: &str, error: &KvSchedulerError) -> Error {
    if let KvSchedulerError::RequestClassifierFailed(source) = error {
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(source.as_ref());
        while let Some(current) = cause {
            if let Some(typed) = current.downcast_ref::<DynamoError>() {
                // A load-shedding classifier rejects by design, one request at
                // a time; that is not an operator error.
                tracing::debug!(
                    request_id = %request_id,
                    error = %typed,
                    "request classifier rejected request"
                );
                return crate::migration::ClassifierRejection(typed.clone()).into();
            }
            cause = current.source();
        }
    }
    // The client only sees the sanitized error below, so this log is the
    // operator's sole copy of the original failure.
    tracing::error!(request_id = %request_id, error = %error, "request classifier failed");
    DynamoError::builder()
        .error_type(ErrorType::Unknown)
        .message("request classifier failed")
        .build()
        .into()
}

fn route_target(worker: WorkerWithDpRank) -> AffinityTarget {
    AffinityTarget::new(worker.worker_id, Some(worker.dp_rank))
}

fn monitor_response_stream(
    mut response_stream: ManyOut<Annotated<LLMEngineOutput>>,
    context: Arc<dyn AsyncEngineContext>,
    mut guard: RequestGuard,
) -> impl futures::Stream<Item = Annotated<LLMEngineOutput>> + Send {
    async_stream::stream! {
        // Keep one cancellation future alive for the whole response stream. Calling
        // `stopped()` for every item repeatedly clones and polls a watch receiver.
        let stopped = context.stopped();
        tokio::pin!(stopped);

        // Migration acts on errors only; a shutting-down worker sends its error after the terminal frame.
        let mut drainable_terminal = false;
        let mut pending_terminal: Option<Annotated<LLMEngineOutput>> = None;
        // Armed only while draining: a worker that goes quiet without EOF must not hang us.
        let drain_deadline = tokio::time::sleep(Duration::ZERO);
        tokio::pin!(drain_deadline);

        let completed = loop {
            tokio::select! {
                biased;

                _ = &mut stopped => {
                    tracing::debug!(request_id = context.id(), "Request cancelled, ending stream");
                    // The client is gone, so the withheld frame has nowhere to go.
                    drop(pending_terminal.take());
                    break false;
                }

                item = response_stream.next() => {
                    let Some(item) = item else {
                        // EOF while draining means no trailing error is coming.
                        if drainable_terminal {
                            guard.record_migration_failure(None);
                        }
                        break !drainable_terminal;
                    };
                    let outcome = classify_response_item(&item);
                    guard.on_item(&item).await;
                    match outcome {
                        ResponseItemOutcome::Failed => {
                            // Supersedes the withheld frame: never end a request about to be retried.
                            drop(pending_terminal.take());
                            guard.record_migration_failure(item.error.clone());
                            // Release the failed attempt before Migration can observe
                            // the item and start another one. This keeps serialized
                            // retries free of stale-cleanup ABA races. A migratable
                            // failure hands the classifier lifecycle to the retry,
                            // exactly like a dispatch-time failure; anything else is
                            // terminal for the logical request and aborts it here.
                            let migratable = item
                                .error
                                .as_ref()
                                .is_some_and(|error| crate::migration::is_migratable(error));
                            if !migratable || !guard.release_for_retry().await {
                                guard
                                    .abort(item.error.as_ref().map(|e| e as &AbortCause))
                                    .await;
                            }
                            yield item;
                            break false;
                        }
                        ResponseItemOutcome::DrainableTerminal => {
                            // Armed once: re-arming per frame would let a flood of terminals
                            // postpone the deadline forever.
                            if !drainable_terminal {
                                drainable_terminal = true;
                                drain_deadline.as_mut().reset(tokio::time::Instant::now() + DRAIN_TIMEOUT);
                            }
                            // Only the newest terminal frame can be the last one.
                            if let Some(previous) = pending_terminal.replace(item) {
                                yield previous;
                            }
                            // `biased` polls this arm first, so an always-ready stream would
                            // otherwise starve the deadline below. Compare the clock rather than
                            // `is_elapsed()`: a `Sleep` that is never polled never reports elapsed.
                            if tokio::time::Instant::now() >= drain_deadline.deadline() {
                                guard.record_migration_failure(None);
                                break false;
                            }
                        }
                        ResponseItemOutcome::Healthy => {
                            // More data followed, so the withheld frame was not last after all.
                            drainable_terminal = false;
                            if let Some(previous) = pending_terminal.take() {
                                yield previous;
                            }
                            yield item;
                        }
                    }
                }

                // Last arm: a frame that is already available always beats an expired drain.
                _ = &mut drain_deadline, if drainable_terminal => {
                    tracing::debug!(
                        request_id = context.id(),
                        "Terminal frame was not followed by an error within {DRAIN_TIMEOUT:?}, ending stream"
                    );
                    guard.record_migration_failure(None);
                    break false;
                }
            }
        };

        if completed {
            guard.complete().await;
        } else {
            guard.abort(None).await;
        }
        // Released only now: the drain proved it was last, and the booking is already gone.
        if let Some(pending) = pending_terminal.take() {
            yield pending;
        }
    }
}

fn into_monitored_response(
    response_stream: ManyOut<Annotated<LLMEngineOutput>>,
    guard: RequestGuard,
) -> ManyOut<Annotated<LLMEngineOutput>> {
    let stream_context = response_stream.context();
    let wrapped_stream = Box::pin(monitor_response_stream(
        response_stream,
        stream_context.clone(),
        guard,
    ));
    ResponseStream::new(wrapped_stream, stream_context)
}

enum RoutingPolicy {
    Kv(Arc<KvRouter>),
    Builtin(BuiltinWorkerSelector),
    Direct,
    DeviceAwareWeighted,
}

struct LoraRouting {
    filter: Arc<LoraFilter>,
    load_estimator: Arc<LoadEstimator>,
    selector: BuiltinWorkerSelector,
}

struct LoraSelection {
    target: u64,
    allowed_fallback: HashSet<u64>,
    load_guard: LoraLoadGuard,
}

struct HostedSelection {
    initial_worker: u64,
    target_constraint: Option<AffinityTarget>,
    occupancy_reservation: Option<dynamo_runtime::pipeline::OccupancyReservation>,
    candidate_count: usize,
    selected_occupancy: Option<u64>,
    device_aware_telemetry: Option<DeviceAwareTelemetry>,
}

struct DeviceAwareTelemetry {
    is_cpu: bool,
    embedding_cache_hit: bool,
    request_cache_keys: usize,
}

/// Owns request routing from worker selection through response cleanup.
///
/// [`PushRouter`] owns discovery, fault detection, and transport. [`KvRouter`]
/// owns optional KV candidate state. `RoutingHost` owns the common request
/// lifecycle regardless of which policy selected the worker.
pub struct RoutingHost {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    policy: RoutingPolicy,
    request_metrics: Arc<RouterRequestMetrics>,
    affinity: Option<AffinityCoordinator>,
    session_affinity_mode: SessionAffinityMode,
    hosted_occupancy: Option<HostedOccupancy>,
    lora: Option<LoraRouting>,
    /// Retains the shared client, overload state, and cancellation subtree for this host.
    ///
    /// Required for every mode except KV, where the [`KvRouter`] owns the client.
    #[allow(dead_code)]
    routing_context: Option<Arc<crate::kv_router::RoutingLoadContext>>,
}

/// An admitted KV route awaiting dispatch: the selected worker, its booking,
/// and the affinity hold, until [`RoutingHost::dispatch_kv_plan`] sends it or
/// [`RoutePlan::abort`] gives it up.
pub(crate) struct RoutePlan {
    /// The phase the worker was selected for; dispatch tracks under it rather
    /// than re-reading the request.
    phase: RequestPhase,
    selection: WorkerSelection,
    cleanup: KvRequestCleanup,
    affinity: Option<Hold>,
    /// Carried forward from the [`RoutePreview`] this plan was admitted from, so
    /// preview, admission and dispatch draw on one budget instead of three.
    budget: CleanupBudget,
}

/// A KV route selected without scheduler admission.
pub(crate) struct RoutePreview {
    request_id: String,
    phase: RequestPhase,
    pub(crate) signals: RoutePlanSignals,
    /// Starts here because the conditional route's first stage is the preview.
    budget: CleanupBudget,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RoutePlanSignals {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) overlap_blocks: u32,
    pub(crate) cached_tokens: usize,
    pub(crate) potential_decode_blocks: u64,
    pub(crate) total_kv_blocks: Option<u64>,
}

impl RoutePreview {
    /// Starts the budget's clock and reports what is left, so a test can follow
    /// one budget across the real preview/plan/dispatch chain.
    #[cfg(test)]
    pub(crate) fn cleanup_budget_remaining(&self) -> std::time::Duration {
        self.budget.remaining()
    }
}

impl RoutePlanSignals {
    pub(crate) fn decode_load_exceeds(self, threshold: f64) -> Option<bool> {
        let total_kv_blocks = self.total_kv_blocks?;
        Some(self.potential_decode_blocks as f64 > threshold * total_kv_blocks as f64)
    }
}

impl RoutePlan {
    #[cfg(test)]
    pub(crate) fn cleanup_budget_remaining(&self) -> std::time::Duration {
        self.budget.remaining()
    }

    pub(crate) fn worker(&self) -> WorkerWithDpRank {
        self.selection.worker
    }

    /// Early release for the booking, taken before dispatch so a migration
    /// retry can free it while the prefill stream still runs.
    pub(crate) fn attempt_release(&self) -> Option<RequestAttemptRelease> {
        self.cleanup.attempt_release()
    }

    /// Give the route up before dispatch: abort the classifier lifecycle with
    /// `error` as its cause, release the booking, and drop the affinity hold.
    pub(crate) async fn abort(mut self, error: Option<&Error>) {
        if let Some(mut lifecycle) = self.selection.request_lifecycle.take() {
            lifecycle.abort(error.map(|error| owned_abort_error(error.as_ref())));
        }
        self.cleanup.finish().await;
    }
}

/// Compatibility name for the KV-only host used by existing callers.
///
/// This alias remains supported through the Dynamo 1.x series. It may be
/// removed only in a 2.0.0 (or later) breaking release.
pub type KvPushRouter = RoutingHost;

impl RoutingHost {
    pub fn required_worker_inputs(&self) -> WorkerInputs {
        match &self.policy {
            RoutingPolicy::Kv(chooser) => chooser.required_worker_inputs(),
            RoutingPolicy::Builtin(selector) => selector.required_worker_inputs(),
            RoutingPolicy::Direct => WorkerInputs::NONE,
            RoutingPolicy::DeviceAwareWeighted => WorkerInputs::OCCUPANCY,
        }
    }

    pub(crate) fn available_worker_ids(&self) -> HashSet<u64> {
        self.inner
            .client
            .available_instance_ids()
            .map(|ids| ids.as_ref().clone())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn occupancy_for_test(&self, worker_id: u64) -> u64 {
        self.inner.occupancy_for_test(worker_id)
    }

    /// The active KV-aware data plane.
    pub fn kv_router(&self) -> &Arc<KvRouter> {
        self.kv_router_if_enabled()
            .expect("routing host has no KV capability")
    }

    pub(crate) fn kv_router_if_enabled(&self) -> Option<&Arc<KvRouter>> {
        match &self.policy {
            RoutingPolicy::Kv(chooser) => Some(chooser),
            RoutingPolicy::Builtin(_)
            | RoutingPolicy::Direct
            | RoutingPolicy::DeviceAwareWeighted => None,
        }
    }

    pub(crate) fn peek_next_worker(&self) -> Option<u64> {
        match &self.policy {
            RoutingPolicy::Builtin(selector) => match &self.hosted_occupancy {
                Some(occupancy) => occupancy.peek(&self.inner, selector),
                None => self
                    .inner
                    .with_selectable_worker_ids(|ids| {
                        selector.peek_worker(
                            dynamo_kv_router::selector::WorkerSelectionInput::hosted(ids, None),
                        )
                    })
                    .ok()
                    .and_then(Result::ok),
            },
            RoutingPolicy::DeviceAwareWeighted => self.inner.peek_next_worker(),
            RoutingPolicy::Direct => None,
            RoutingPolicy::Kv(_) => None,
        }
    }

    /// The group key for a subagent: a request that carries a parent session id binds under the
    /// parent's group for decode, while prefill uses the parent's own binding to reuse its cache.
    /// An explicit per-request target stays on the request's own session so
    /// it cannot be rejected against, or rebind, the group.
    fn group_binding_id(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        explicit: Option<AffinityTarget>,
        phase: RequestPhase,
    ) -> Option<Arc<SessionAffinityId>> {
        if explicit.is_some() {
            return None;
        }
        let agent = request.content().agent_context.as_ref()?;
        let parent_session_id = agent.parent_session_id.as_deref()?;
        let binding_id = if phase == RequestPhase::Prefill {
            let parent = parent_session_id.trim();
            if parent.is_empty() || parent == agent.session_id.trim() {
                return None;
            }
            parent.to_owned()
        } else {
            subagent_group_affinity_id(parent_session_id)
        };
        Some(Arc::new(SessionAffinityId::new(binding_id)))
    }

    /// The worker accepted the send: tell the guard, then bind the held session
    /// to `dispatched_target` and park its lease in the guard, so the binding
    /// is held exactly as long as the request it belongs to. A commit that
    /// fails after a successful send stops the worker's generation, aborts the
    /// attempt with the cause, and fails the request.
    async fn accept_dispatch(
        &self,
        guard: &mut RequestGuard,
        hold: Option<Hold>,
        dispatched_target: AffinityTarget,
        stream_context: &Arc<dyn AsyncEngineContext>,
    ) -> Result<(), Error> {
        guard.on_dispatched();
        let (Some(hold), Some(affinity)) = (hold, self.affinity.as_ref()) else {
            return Ok(());
        };
        match affinity.commit(hold, dispatched_target) {
            Ok(lease) => {
                guard.hold_affinity_lease(lease);
                Ok(())
            }
            Err(error) => {
                // The worker is already generating for a request the client
                // will see fail; stop it instead of letting it run to the end.
                stream_context.stop_generating();
                guard.abort(Some(error.as_ref())).await;
                Err(error)
            }
        }
    }

    /// Check request-supplied targets only at initial selection, not after a route preview
    /// or when dispatching: a worker disappearing later is a service failure.
    fn validate_explicit_worker(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
    ) -> Result<(), Error> {
        let Some(routing) = request.routing.as_ref() else {
            return Ok(());
        };
        let target = match phase {
            RequestPhase::Prefill => routing
                .prefill_worker_id
                .map(|id| (id, "prefill_worker_id")),
            RequestPhase::Decode | RequestPhase::Aggregated => {
                routing.decode_worker_id.map(|id| (id, "decode_worker_id"))
            }
        }
        .or_else(|| {
            routing
                .backend_instance_id
                .map(|id| (id, "backend_instance_id"))
        });
        // Validate discovery membership, not health: unavailable workers are still known.
        if let Some((worker_id, field)) = target
            && !self.inner.client.is_instance_discovered(worker_id)
        {
            return Err(invalid_argument(format!(
                "nvext.{field}={worker_id} does not identify a known worker"
            )));
        }
        Ok(())
    }

    fn affinity_target_is_valid(&self, target: AffinityTarget) -> bool {
        if !self.inner.client.is_instance_discovered(target.worker_id) {
            return false;
        }
        let Some(kv_router) = self.kv_router_if_enabled() else {
            return true;
        };
        let workers = kv_router.workers_with_configs.borrow();
        let Some(config) = workers.get(&target.worker_id) else {
            return true;
        };
        let Some(dp_rank) = target.dp_rank else {
            return true;
        };
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        (start..end).contains(&dp_rank)
    }

    /// Take a session-affinity slot under the same cleanup policy as every other
    /// routing stage.
    ///
    /// `acquire_with_context` cancels its own wait as soon as the context stops,
    /// and it runs upstream of every other stage. A decode leg with staged KV
    /// would therefore die here — before any of the wrapped stages could let it
    /// through — whenever a concurrent request for the same session is still
    /// `Initializing`. On that path we wait through the stop instead, drawing on
    /// the request's shared budget so the wait is still bounded.
    #[allow(clippy::too_many_arguments)]
    async fn acquire_affinity_slot(
        &self,
        affinity: &AffinityCoordinator,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        context: &dyn AsyncEngineContext,
        phase: RequestPhase,
        staged_kv: StagedKv,
        budget: &CleanupBudget,
    ) -> Result<Hold, Error> {
        match DispatchCancellation::for_request(phase, staged_kv) {
            DispatchCancellation::CancelWhenStopped => {
                affinity
                    .acquire_with_context(session_id, requested_target, context)
                    .await
            }
            DispatchCancellation::DispatchWhenStopped => await_with_cleanup_policy(
                context,
                phase,
                staged_kv,
                "affinity.acquire",
                budget,
                affinity.acquire(session_id, requested_target),
            )
            .await
            .and_then(|result| result),
        }
    }

    async fn select_with_session_affinity<T, Select, SelectionFuture>(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        budget: &CleanupBudget,
        mut select: Select,
    ) -> Result<(T, Option<Hold>), Error>
    where
        Select: FnMut(Option<AffinityTarget>) -> SelectionFuture,
        SelectionFuture: Future<Output = Result<T, Error>>,
    {
        let staged_kv = StagedKv::for_request(request.content());
        let Some(affinity) = self.affinity.as_ref() else {
            return Ok((select(None).await?, None));
        };
        let Some(session_id) = affinity_id(request)? else {
            return Ok((select(None).await?, None));
        };
        let explicit = explicit_target(request.content(), phase)?;
        let session_id = self
            .group_binding_id(request, explicit, phase)
            .unwrap_or(session_id);
        if is_query_only {
            let target = affinity.query_target(&session_id, explicit)?;
            return Ok((select(target).await?, None));
        }

        let request_context = request.context();
        let reacquire = || {
            self.acquire_affinity_slot(
                affinity,
                &session_id,
                None,
                request_context.as_ref(),
                phase,
                staged_kv,
                budget,
            )
        };
        let mut operation = self
            .acquire_affinity_slot(
                affinity,
                &session_id,
                explicit,
                request_context.as_ref(),
                phase,
                staged_kv,
                budget,
            )
            .await?;
        if self.session_affinity_mode == SessionAffinityMode::Soft
            && explicit.is_none()
            && matches!(
                &self.policy,
                RoutingPolicy::Builtin(_) | RoutingPolicy::DeviceAwareWeighted
            )
            && operation
                .target()
                .is_some_and(|target| !self.inner.client.is_instance_discovered(target.worker_id))
        {
            operation.invalidate();
            operation = reacquire().await?;
        }
        let target = operation.target().map(from_table);
        match select(target).await {
            Ok(selection) => Ok((selection, Some(operation))),
            Err(error) if is_cancelled(&error) => Err(error),
            Err(_error)
                if self.session_affinity_mode == SessionAffinityMode::Hard
                    && explicit.is_none()
                    && target.is_some_and(|target| !self.affinity_target_is_valid(target)) =>
            {
                operation.invalidate();
                let retry = reacquire().await?;
                let selection = select(retry.target().map(from_table)).await?;
                Ok((selection, Some(retry)))
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<PrefillDispatch<M>, Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        match &self.policy {
            RoutingPolicy::Kv(_) => self.select_and_dispatch_kv_prefill(request, prepare).await,
            RoutingPolicy::Builtin(_)
            | RoutingPolicy::Direct
            | RoutingPolicy::DeviceAwareWeighted => {
                let (metadata, stream) = self
                    .select_and_dispatch_builtin(request, RequestPhase::Prefill, prepare)
                    .await?;
                Ok(PrefillDispatch {
                    metadata,
                    stream,
                    booking: None,
                })
            }
        }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for RoutingHost
{
    /// Generate a request through the selected routing plane.
    ///
    /// On the KV plane, `query_instance_id` performs an advisory selection:
    ///    - Returns the best matching worker ID without routing the request
    ///    - Does NOT update any router local states
    ///    - Response includes worker_instance_id and token_data annotations
    ///
    /// The built-in Random and RoundRobin plane has no KV query path: it selects a worker and
    /// dispatches the request. `query_instance_id` is therefore a KV-routing/disaggregation
    /// annotation, not a request-execution suppressor for those modes.
    ///
    /// On the KV plane, a phase-specific worker or `backend_instance_id`:
    ///    - Query-only requests return that worker selection without state updates
    ///    - Requests route through the scheduler as an exact pin when dp_rank is resolved
    ///    - If dp_rank cannot be resolved, the request is rejected instead of treating rank 0 as a sentinel
    ///
    /// Otherwise, KV routing:
    ///    - Finds the best worker based on KV cache overlap
    ///    - Updates router states to track the request
    ///    - Routes to the selected worker
    ///
    /// The router state updates include tracking active sequences and managing
    /// prefill/completion lifecycle for proper KV cache management.
    async fn generate(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        // One cleanup budget for this request's whole route through the host.
        let budget = CleanupBudget::default();
        let phase = request.phase();
        self.normalize_prefix_cache_read_control(&mut request, phase);
        if !matches!(&self.policy, RoutingPolicy::Kv(_)) {
            return self
                .select_and_dispatch_builtin(request, phase, |_, _| Ok(()))
                .await
                .map(|(_, stream)| stream);
        }

        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        if is_query_only {
            let phase_label = phase.to_string();
            let _route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
            let (selection, _) = self
                .select_with_affinity(&request, phase, true, &budget)
                .await?;
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
                    token_ids: Some(request.token_ids.as_ref().clone()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let response = Annotated::from_data(output);
            let stream = stream::iter(vec![response]);
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        }

        let plan = self.plan_kv_route(&request, phase, None, budget).await?;
        self.dispatch_kv_plan(request, plan).await
    }
}

enum ResponseItemOutcome {
    /// The stream is healthy and must keep running.
    Healthy,
    /// Terminal by finish reason only; withheld while the stream drains for a trailing error.
    DrainableTerminal,
    /// Terminal and carries the error itself. Yielded, and the stream ends.
    Failed,
}

fn classify_response_item(item: &Annotated<LLMEngineOutput>) -> ResponseItemOutcome {
    if item.error.is_some() || item.event.as_deref() == Some("error") {
        return ResponseItemOutcome::Failed;
    }
    let terminal = item
        .data
        .as_ref()
        .and_then(|data| data.finish_reason.as_ref())
        .is_some_and(|reason| matches!(reason, FinishReason::Error(_) | FinishReason::Cancelled));
    if terminal {
        ResponseItemOutcome::DrainableTerminal
    } else {
        ResponseItemOutcome::Healthy
    }
}

#[cfg(test)]
mod tests;

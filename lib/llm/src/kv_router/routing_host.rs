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
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    protocols::common::{
        FinishReason,
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RoutingData, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
    },
    session_affinity::{
        AffinityAcquire, AffinityCoordinator, AffinityTarget, affinity_id, explicit_target,
        invalid_argument,
    },
};

mod builtin;
mod cancellation;
mod kv;
mod kv_selection;
mod occupancy;
mod request_guard;

use builtin::BuiltinWorkerSelector;
use cancellation::cancel_on_stop;
use kv_selection::{RoutingRequestParts, SelectionOptions, WorkerSelection};
use occupancy::HostedOccupancy;
use request_guard::{LoraLoadGuard, RequestGuard};

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

enum RoutingPolicy<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    Kv(Arc<KvRouter<Sel>>),
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
pub struct RoutingHost<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    policy: RoutingPolicy<Sel>,
    request_metrics: Arc<RouterRequestMetrics>,
    affinity: Option<AffinityCoordinator>,
    hosted_occupancy: Option<HostedOccupancy>,
    lora: Option<LoraRouting>,
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
            policy: RoutingPolicy::Kv(kv_router),
            request_metrics,
            affinity,
            hosted_occupancy: None,
            lora: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_builtin(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<Self, Error> {
        Self::new_builtin_with_capabilities(inner, None, None)
    }

    pub(crate) fn new_builtin_with_coordinator(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        affinity: Option<AffinityCoordinator>,
    ) -> Result<Self, Error> {
        Self::new_builtin_with_capabilities(inner, affinity, None)
    }

    pub(crate) fn new_builtin_with_capabilities(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        affinity: Option<AffinityCoordinator>,
        lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
    ) -> Result<Self, Error> {
        if affinity.is_some() && lora.is_some() {
            anyhow::bail!("session affinity and LoRA filtering cannot both be enabled");
        }
        let policy = match inner.router_mode() {
            RouterMode::Direct => RoutingPolicy::Direct,
            RouterMode::DeviceAwareWeighted => RoutingPolicy::DeviceAwareWeighted,
            mode => {
                RoutingPolicy::Builtin(BuiltinWorkerSelector::new(mode).ok_or_else(|| {
                    anyhow::anyhow!("{mode:?} routing is not a first-party policy")
                })?)
            }
        };
        let required_worker_inputs = match &policy {
            RoutingPolicy::Builtin(selector) => selector.required_worker_inputs(),
            RoutingPolicy::DeviceAwareWeighted => WorkerInputs::OCCUPANCY,
            RoutingPolicy::Direct => WorkerInputs::NONE,
            RoutingPolicy::Kv(_) => unreachable!(),
        };
        let hosted_occupancy = matches!(&policy, RoutingPolicy::Builtin(_))
            .then_some(required_worker_inputs.contains(WorkerInputs::OCCUPANCY))
            .unwrap_or(false)
            .then(|| HostedOccupancy::new(&inner))
            .transpose()?;
        if lora.is_some()
            && !matches!(
                inner.router_mode(),
                RouterMode::RoundRobin | RouterMode::Random
            )
        {
            anyhow::bail!(
                "LoRA filtering is unsupported with {:?} routing",
                inner.router_mode()
            );
        }
        let lora_selector = lora.as_ref().map(|_| {
            BuiltinWorkerSelector::new(inner.router_mode())
                .expect("LoRA routing mode was validated above")
        });
        let request_metrics =
            RouterRequestMetrics::from_component(inner.client.endpoint.component());
        Ok(Self {
            inner,
            policy,
            request_metrics,
            affinity,
            hosted_occupancy,
            lora: lora
                .zip(lora_selector)
                .map(|((filter, load_estimator), selector)| LoraRouting {
                    filter,
                    load_estimator,
                    selector,
                }),
        })
    }

    pub fn required_worker_inputs(&self) -> WorkerInputs {
        match &self.policy {
            RoutingPolicy::Kv(chooser) => chooser.required_worker_inputs(),
            RoutingPolicy::Builtin(selector) => selector.required_worker_inputs(),
            RoutingPolicy::Direct => WorkerInputs::NONE,
            RoutingPolicy::DeviceAwareWeighted => WorkerInputs::OCCUPANCY,
        }
    }

    /// The active KV-aware data plane.
    pub fn kv_router(&self) -> &Arc<KvRouter<Sel>> {
        self.kv_router_if_enabled()
            .expect("routing host has no KV capability")
    }

    pub(crate) fn kv_router_if_enabled(&self) -> Option<&Arc<KvRouter<Sel>>> {
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

    pub(crate) fn query_affinity_target(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
    ) -> Result<Option<AffinityTarget>, Error> {
        let Some(affinity) = self.affinity.as_ref() else {
            return Ok(None);
        };
        let Some(session_id) = affinity_id(request)? else {
            return Ok(None);
        };
        let explicit = explicit_target(request, phase)?;
        affinity.query_target(&session_id, explicit)
    }

    fn affinity_target_requires_rebind(
        &self,
        request: &PreprocessedRequest,
        target: AffinityTarget,
    ) -> bool {
        if request
            .migration_state
            .as_ref()
            .is_some_and(|state| state.excluded_worker_ids().contains(&target.worker_id))
        {
            return true;
        }
        if !self
            .inner
            .client
            .instance_ids_avail()
            .contains(&target.worker_id)
        {
            return true;
        }
        let Some(kv_router) = self.kv_router_if_enabled() else {
            return false;
        };
        let workers = kv_router.workers_with_configs.borrow();
        let Some(config) = workers.get(&target.worker_id) else {
            return true;
        };
        let Some(dp_rank) = target.dp_rank else {
            return false;
        };
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        !(start..end).contains(&dp_rank)
    }

    async fn select_with_session_affinity<T, Select, SelectionFuture>(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        mut select: Select,
    ) -> Result<(T, Option<AffinityAcquire>), Error>
    where
        Select: FnMut(Option<AffinityTarget>) -> SelectionFuture,
        SelectionFuture: Future<Output = Result<T, Error>>,
    {
        let Some(affinity) = self.affinity.as_ref() else {
            return Ok((select(None).await?, None));
        };
        let Some(session_id) = affinity_id(request)? else {
            return Ok((select(None).await?, None));
        };
        let explicit = explicit_target(request, phase)?;
        if is_query_only {
            let target = affinity.query_target(&session_id, explicit)?;
            return Ok((select(target).await?, None));
        }

        let request_context = request.context();
        let operation = affinity
            .acquire_with_context(&session_id, explicit, request_context.as_ref())
            .await?;
        let target = operation.target();
        match select(target).await {
            Ok(selection) => Ok((selection, Some(operation))),
            Err(error) if is_cancelled(&error) => Err(error),
            Err(_)
                if explicit.is_none()
                    && target.is_some_and(|target| {
                        self.affinity_target_requires_rebind(request.content(), target)
                    }) =>
            {
                operation.invalidate();
                let retry = affinity
                    .acquire_with_context(&session_id, None, request_context.as_ref())
                    .await?;
                match select(retry.target()).await {
                    Ok(selection) => Ok((selection, Some(retry))),
                    Err(retry_error) => Err(retry_error),
                }
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        match &self.policy {
            RoutingPolicy::Kv(_) => self.select_and_dispatch_kv_prefill(request, prepare).await,
            RoutingPolicy::Builtin(_)
            | RoutingPolicy::Direct
            | RoutingPolicy::DeviceAwareWeighted => {
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
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        if !matches!(&self.policy, RoutingPolicy::Kv(_)) {
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

#[cfg(test)]
mod tests;

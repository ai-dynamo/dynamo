// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_kv_router::{
    protocols::WorkerWithDpRank,
    selector::{WorkerInputs, WorkerSelector},
};
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    pipeline::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        ResponseStream, RouterMode, SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::StreamExt;

use crate::{
    kv_router::{KvRouter, metrics::RouterRequestMetrics, scheduler::DefaultWorkerSelector},
    local_model::runtime_config::ModelRuntimeConfig,
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    protocols::common::{FinishReason, llm_backend::LLMEngineOutput, timing::RequestPhase},
    session_affinity::{
        AffinityAcquire, AffinityCoordinator, AffinityTarget, affinity_id, explicit_target,
    },
};

mod builtin;
mod cancellation;
mod kv;
mod kv_selection;
mod load;
mod request_guard;

use builtin::BuiltinRoutingPolicy;
use builtin::LoraRouting;
use cancellation::cancel_on_stop;
pub(crate) use load::RoutingLoadState;
use request_guard::RequestGuard;

const OUTPUT_REPLAY_ID_ANNOTATION_KEY: &str = "output_replay_id";
const OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY: &str = "output_replay_consumer";

pub(crate) fn builtin_policy_requires_load(mode: RouterMode) -> bool {
    BuiltinRoutingPolicy::from_router_mode(mode)
        .is_some_and(|policy| policy.required_worker_inputs().contains(WorkerInputs::LOAD))
}

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
    push_router: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    plane: RoutingPlane<Sel>,
    request_metrics: Arc<RouterRequestMetrics>,
    affinity: Option<AffinityCoordinator>,
    load_state: Option<Arc<RoutingLoadState>>,
    lora: Option<LoraRouting>,
}

impl<Sel> RoutingHost<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub fn new(
        push_router: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        kv_router: Arc<KvRouter<Sel>>,
        session_affinity_ttl: Option<Duration>,
    ) -> Result<Self, Error> {
        let affinity = session_affinity_ttl
            .map(AffinityCoordinator::new)
            .transpose()?;

        Ok(Self::new_with_coordinator(push_router, kv_router, affinity))
    }

    pub(crate) fn new_with_coordinator(
        push_router: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        kv_router: Arc<KvRouter<Sel>>,
        affinity: Option<AffinityCoordinator>,
    ) -> Self {
        // Eagerly register router request metrics (as zeros) so they are
        // scrapeable before any requests arrive. Both the frontend pipeline
        // and the standalone router create RoutingHost, so this covers both.
        let request_metrics =
            RouterRequestMetrics::from_component(kv_router.client().endpoint.component());

        RoutingHost {
            push_router,
            plane: RoutingPlane::Kv(kv_router),
            request_metrics,
            affinity,
            load_state: None,
            lora: None,
        }
    }

    pub(crate) fn new_builtin_with_coordinator(
        push_router: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        load_state: Option<Arc<RoutingLoadState>>,
        affinity: Option<AffinityCoordinator>,
    ) -> Result<Self, Error> {
        Self::new_builtin_with_capabilities(push_router, load_state, affinity, None)
    }

    pub(crate) fn new_builtin_with_capabilities(
        push_router: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        load_state: Option<Arc<RoutingLoadState>>,
        affinity: Option<AffinityCoordinator>,
        lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
    ) -> Result<Self, Error> {
        if affinity.is_some() && lora.is_some() {
            anyhow::bail!("session affinity and LoRA filtering cannot both be enabled");
        }
        let policy =
            BuiltinRoutingPolicy::from_router_mode(push_router.router_mode()).ok_or_else(|| {
                anyhow::anyhow!(
                    "{:?} routing is not a first-party builtin policy",
                    push_router.router_mode()
                )
            })?;
        let needs_load = policy.required_worker_inputs().contains(WorkerInputs::LOAD);
        anyhow::ensure!(
            needs_load == load_state.is_some(),
            "{policy:?} routing requires LOAD capability: {needs_load}"
        );
        if lora.is_some()
            && !matches!(
                policy,
                BuiltinRoutingPolicy::RoundRobin | BuiltinRoutingPolicy::Random
            )
        {
            anyhow::bail!("LoRA filtering is unsupported with {policy:?} routing");
        }
        let request_metrics =
            RouterRequestMetrics::from_component(push_router.client.endpoint.component());
        Ok(Self {
            push_router,
            plane: RoutingPlane::Builtin(policy),
            request_metrics,
            affinity,
            load_state,
            lora: lora.map(|(filter, load_estimator)| LoraRouting {
                filter,
                load_estimator,
            }),
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
            RoutingPlane::Builtin(_) => self.push_router.peek_next_worker(),
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
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        match &self.plane {
            RoutingPlane::Kv(_) => self.generate_kv(request).await,
            RoutingPlane::Builtin(_) => {
                let phase = request
                    .tracker
                    .as_ref()
                    .map(|tracker| tracker.phase())
                    .unwrap_or(RequestPhase::Aggregated);
                self.select_and_dispatch_builtin(request, phase, |_, _| Ok(()))
                    .await
                    .map(|(_, stream)| stream)
            }
        }
    }
}

fn affinity_worker(target: AffinityTarget) -> Option<WorkerWithDpRank> {
    target
        .dp_rank
        .map(|rank| WorkerWithDpRank::new(target.worker_id, rank))
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

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use dynamo_kv_router::selector::{WorkerInputs, WorkerSelector};
use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    error::{DynamoError, ErrorType},
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{Error, ManyOut, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};

use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
    },
    session_affinity::{AffinityAcquire, AffinityTarget, affinity_id, explicit_target},
};

use super::{
    RoutingHost, RoutingPlane, cancel_on_stop, into_monitored_response,
    invalidate_on_non_cancellation,
    request_guard::{LoraLoadGuard, RequestGuard},
};

/// First-party policies selected without a KV-cache index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BuiltinRoutingPolicy {
    Direct,
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    LeastLoaded,
    DeviceAwareWeighted,
}

impl BuiltinRoutingPolicy {
    pub(super) fn from_router_mode(mode: RouterMode) -> Option<Self> {
        match mode {
            RouterMode::Direct => Some(Self::Direct),
            RouterMode::RoundRobin => Some(Self::RoundRobin),
            RouterMode::Random => Some(Self::Random),
            RouterMode::PowerOfTwoChoices => Some(Self::PowerOfTwoChoices),
            RouterMode::LeastLoaded => Some(Self::LeastLoaded),
            RouterMode::DeviceAwareWeighted => Some(Self::DeviceAwareWeighted),
            RouterMode::KV => None,
        }
    }

    pub(super) const fn required_worker_inputs(self) -> WorkerInputs {
        match self {
            Self::Direct | Self::RoundRobin | Self::Random => WorkerInputs::NONE,
            Self::PowerOfTwoChoices | Self::LeastLoaded | Self::DeviceAwareWeighted => {
                WorkerInputs::LOAD
            }
        }
    }
}

pub(super) struct LoraRouting {
    pub(super) filter: Arc<LoraFilter>,
    pub(super) load_estimator: Arc<LoadEstimator>,
}

struct LoraSelection {
    target: u64,
    allowed_fallback: HashSet<u64>,
    load_guard: LoraLoadGuard,
}

impl<Sel> RoutingHost<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn select_lora_target(
        &self,
        request: &PreprocessedRequest,
    ) -> Result<Option<LoraSelection>, Error> {
        let Some(lora) = self.lora.as_ref() else {
            return Ok(None);
        };
        let Some(lora_name) = request
            .routing
            .as_ref()
            .and_then(|routing| routing.lora_name.clone())
        else {
            return Ok(None);
        };
        let load_guard = LoraLoadGuard::new(Arc::clone(&lora.load_estimator), lora_name.clone());
        let routable = self.push_router.client.instance_ids_avail();
        let candidates = lora
            .filter
            .filter_worker_ids_for_lora(Some(&lora_name), &routable);
        if candidates.is_empty() {
            anyhow::bail!("No workers available after LoRA filtering (lora={lora_name})");
        }

        let free = self
            .push_router
            .client
            .instance_ids_free()
            .into_iter()
            .collect::<HashSet<_>>();
        let candidates = candidates
            .into_iter()
            .filter(|worker_id| free.contains(worker_id))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(anyhow::anyhow!(
                DynamoError::builder()
                    .error_type(ErrorType::ResourceExhausted)
                    .message(format!(
                        "All eligible LoRA workers are overloaded (lora={lora_name})"
                    ))
                    .build()
            ));
        }
        let target = self.push_router.select_from_candidates(&candidates)?;
        tracing::debug!(
            lora = %lora_name,
            worker_id = target,
            candidates = candidates.len(),
            routable = routable.len(),
            free = free.len(),
            "LoRA-filtered router selected worker"
        );
        Ok(Some(LoraSelection {
            target,
            allowed_fallback: candidates.into_iter().collect(),
            load_guard,
        }))
    }

    async fn builtin_affinity_target(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        explicit: Option<AffinityTarget>,
        is_query_only: bool,
    ) -> Result<(Option<AffinityTarget>, Option<AffinityAcquire>), Error> {
        let Some(affinity) = self.affinity.as_ref() else {
            return Ok((explicit, None));
        };
        let Some(session_id) = affinity_id(request)? else {
            return Ok((explicit, None));
        };
        if is_query_only {
            let target = affinity.query_target(&session_id, explicit)?.or(explicit);
            return Ok((target, None));
        }

        let request_context = request.context();
        let operation = affinity
            .acquire_with_context(&session_id, explicit, request_context.as_ref())
            .await?;
        let Some(target) = operation.target() else {
            return Ok((explicit, Some(operation)));
        };
        if self
            .push_router
            .client
            .instance_ids_avail()
            .contains(&target.worker_id)
        {
            return Ok((Some(target), Some(operation)));
        }

        operation.invalidate();
        let retry = affinity
            .acquire_with_context(&session_id, explicit, request_context.as_ref())
            .await?;
        Ok((retry.target().or(explicit), Some(retry)))
    }

    pub(super) async fn select_and_dispatch_builtin<M, F>(
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
        let required_inputs = policy.required_worker_inputs();

        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let explicit = explicit_target(&request, phase)?;
        if policy == BuiltinRoutingPolicy::Direct && explicit.is_none() {
            anyhow::bail!("worker ID required for {phase} request in Direct routing mode");
        }
        let lora_selection = self.select_lora_target(request.content())?;
        let affinity_explicit = if lora_selection.is_some() {
            None
        } else {
            explicit
        };
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        let (pinned_target, mut operation) = self
            .builtin_affinity_target(&request, affinity_explicit, is_query_only)
            .await?;
        let (lora_target, lora_fallback, lora_load) = match lora_selection {
            Some(selection) => (
                Some(selection.target),
                Some(selection.allowed_fallback),
                Some(selection.load_guard),
            ),
            None => (None, None, None),
        };
        let (initial_worker, load_reservation) = if let Some(target) = lora_target {
            (target, None)
        } else if required_inputs.contains(WorkerInputs::LOAD) {
            let load_state = self
                .load_state
                .as_ref()
                .expect("LOAD policy must have routing load state");
            let reservation = load_state.select_and_reserve(
                &self.push_router,
                request.context().id(),
                request.content(),
                pinned_target.map(|target| (target.worker_id, target.dp_rank)),
            );
            match reservation {
                Ok(reservation) => (reservation.worker().worker_id, Some(reservation)),
                Err(error) => {
                    invalidate_on_non_cancellation(&mut operation, &error);
                    return Err(error);
                }
            }
        } else {
            let selection = self
                .push_router
                .select_policy_target(pinned_target.map(|target| target.worker_id))
                .map(|target| (target, None));
            match selection {
                Ok(selection) => selection,
                Err(error) => {
                    invalidate_on_non_cancellation(&mut operation, &error);
                    return Err(error);
                }
            }
        };
        let mut guard: RequestGuard<Sel> = RequestGuard::new_builtin(
            self.request_metrics.clone(),
            initial_worker,
            load_reservation,
            lora_load,
            &request,
        );
        let tracker = request.tracker.clone();
        let request_context = request.context().clone();
        self.request_metrics
            .input_sequence_tokens
            .observe(request.token_ids.len() as f64);
        drop(route_guard);

        guard.start_dispatch(&phase_label);
        guard.record_prefill_start();
        let dispatch_result = if let Some(allowed_fallback) = lora_fallback.as_ref() {
            cancel_on_stop(
                request_context.as_ref(),
                self.push_router.direct_within_prepared(
                    request,
                    initial_worker,
                    Some(allowed_fallback),
                    |request, worker_id| {
                        guard.retarget_worker(worker_id)?;
                        let target = AffinityTarget::new(worker_id, None);
                        request.routing_mut().dp_rank = None;
                        prepare(request, target).map(|metadata| (metadata, target))
                    },
                ),
            )
            .await
            .and_then(|result| result)
            .map(|((metadata, target), stream)| (metadata, target, stream))
        } else if let Some(target) = pinned_target {
            request.routing_mut().dp_rank = target.dp_rank;
            let metadata = match prepare(&mut request, target) {
                Ok(metadata) => metadata,
                Err(error) => {
                    guard.abort().await;
                    invalidate_on_non_cancellation(&mut operation, &error);
                    return Err(error);
                }
            };
            cancel_on_stop(
                request_context.as_ref(),
                self.push_router.dispatch_exact(request, target.worker_id),
            )
            .await
            .and_then(|result| result)
            .map(|stream| (metadata, target, stream))
        } else {
            cancel_on_stop(
                request_context.as_ref(),
                self.push_router.dispatch_selected_untracked(
                    initial_worker,
                    request,
                    |request, worker_id| {
                        guard.retarget_worker(worker_id)?;
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
                invalidate_on_non_cancellation(&mut operation, &error);
                return Err(error);
            }
        };
        guard.retarget_worker(target.worker_id)?;
        if let Some(tracker) = tracker {
            let worker_type = if tracker.phase() == RequestPhase::Prefill {
                WORKER_TYPE_PREFILL
            } else {
                WORKER_TYPE_DECODE
            };
            tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
        }
        guard.mark_dispatched();
        let stream = into_monitored_response(response_stream, guard);
        let Some(operation) = operation else {
            return Ok((metadata, stream));
        };
        Ok((metadata, operation.into_stream(target, stream)?))
    }
}

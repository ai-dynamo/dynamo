// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    protocols::{WorkerSelectionResult, WorkerWithDpRank},
    scheduling::KvSchedulerError,
    selector::{HostedSelectionInputs, WorkerInputs, WorkerSelectionInput, WorkerSelector},
};
use dynamo_runtime::pipeline::{BuiltinRoutePicker, RouterMode};

use crate::local_model::runtime_config::ModelRuntimeConfig;

/// First-party selector hosted directly by [`RoutingHost`](super::RoutingHost).
pub(super) struct BuiltinWorkerSelector {
    mode: RouterMode,
    picker: BuiltinRoutePicker,
}

impl BuiltinWorkerSelector {
    pub(super) fn new(mode: RouterMode) -> Option<Self> {
        let picker = match mode {
            RouterMode::RoundRobin => BuiltinRoutePicker::round_robin(),
            RouterMode::Random => BuiltinRoutePicker::random(),
            RouterMode::PowerOfTwoChoices => BuiltinRoutePicker::power_of_two_choices(),
            RouterMode::LeastLoaded => BuiltinRoutePicker::least_loaded(),
            _ => return None,
        };
        Some(Self { mode, picker })
    }

    pub(super) fn peek_worker(
        &self,
        input: WorkerSelectionInput<'_, ModelRuntimeConfig>,
    ) -> Result<u64, KvSchedulerError> {
        let (worker_ids, occupancy) = self.hosted_inputs(input)?;
        self.picker
            .peek_worker(worker_ids, |worker_id| {
                occupancy.map_or(0, |occupancy| occupancy(worker_id))
            })
            .ok_or(KvSchedulerError::NoEndpoints)
    }

    fn hosted_inputs<'a>(
        &self,
        input: WorkerSelectionInput<'a, ModelRuntimeConfig>,
    ) -> Result<HostedSelectionInputs<'a>, KvSchedulerError> {
        let (worker_ids, occupancy) = input.into_hosted()?;
        if self
            .required_worker_inputs()
            .contains(WorkerInputs::OCCUPANCY)
            && occupancy.is_none()
        {
            return Err(dynamo_kv_router::WorkerSelectionPolicyError::failed(
                "selector requires hosted OCCUPANCY input",
            )
            .into());
        }
        Ok((worker_ids, occupancy))
    }
}

impl WorkerSelector<ModelRuntimeConfig> for BuiltinWorkerSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        if self.mode.requires_occupancy() {
            WorkerInputs::OCCUPANCY
        } else {
            WorkerInputs::NONE
        }
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, ModelRuntimeConfig>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let (worker_ids, occupancy) = self.hosted_inputs(input)?;
        let worker_id = self
            .picker
            .select_worker(worker_ids, |worker_id| {
                occupancy.map_or(0, |occupancy| occupancy(worker_id))
            })
            .ok_or(KvSchedulerError::NoEndpoints)?;
        Ok(selection(worker_id))
    }
}

fn selection(worker_id: u64) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker: WorkerWithDpRank::from_worker_id(worker_id),
        required_blocks: 0,
        effective_overlap_blocks: 0.0,
        cached_tokens: 0,
        potential_decode_blocks: 0,
    }
}

use super::*;

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
        let routable = self.inner.client.instance_ids_avail();
        let candidates = lora
            .filter
            .filter_worker_ids_for_lora(Some(&lora_name), &routable);
        if candidates.is_empty() {
            anyhow::bail!("No workers available after LoRA filtering (lora={lora_name})");
        }

        let free = self
            .inner
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
        let target = lora
            .selector
            .select_worker(dynamo_kv_router::selector::WorkerSelectionInput::hosted(
                &candidates,
                None,
            ))?
            .worker
            .worker_id;
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

    fn select_hosted_worker(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        target_constraint: Option<AffinityTarget>,
    ) -> Result<HostedSelection, Error> {
        match &self.policy {
            RoutingPolicy::Kv(_) => unreachable!("hosted selection called for KV routing"),
            RoutingPolicy::Direct => {
                let target = target_constraint
                    .ok_or_else(|| anyhow::anyhow!("Direct routing requires an exact target"))?;
                Ok(HostedSelection {
                    initial_worker: target.worker_id,
                    target_constraint: Some(target),
                    occupancy_reservation: None,
                    candidate_count: 1,
                    selected_occupancy: None,
                    device_aware_telemetry: None,
                })
            }
            RoutingPolicy::Builtin(selector) => {
                if selector
                    .required_worker_inputs()
                    .contains(WorkerInputs::OCCUPANCY)
                {
                    let occupancy = self
                        .hosted_occupancy
                        .as_ref()
                        .expect("OCCUPANCY policy must have hosted occupancy state");
                    let selection = occupancy.select_and_reserve(
                        &self.inner,
                        selector,
                        target_constraint.map(|target| target.worker_id),
                    )?;
                    Ok(HostedSelection {
                        initial_worker: selection.worker_id,
                        target_constraint,
                        occupancy_reservation: Some(selection.reservation),
                        candidate_count: selection.candidate_count,
                        selected_occupancy: Some(selection.occupancy),
                        device_aware_telemetry: None,
                    })
                } else {
                    let worker_id = match target_constraint {
                        Some(target) => {
                            self.inner.ensure_routable(target.worker_id)?;
                            target.worker_id
                        }
                        None => {
                            self.inner
                                .with_selectable_worker_ids(|ids| {
                                    selector.select_worker(
                                        dynamo_kv_router::selector::WorkerSelectionInput::hosted(
                                            ids, None,
                                        ),
                                    )
                                })??
                                .worker
                                .worker_id
                        }
                    };
                    Ok(HostedSelection {
                        initial_worker: worker_id,
                        target_constraint,
                        occupancy_reservation: None,
                        candidate_count: 0,
                        selected_occupancy: None,
                        device_aware_telemetry: None,
                    })
                }
            }
            RoutingPolicy::DeviceAwareWeighted => {
                let selection = self.inner.select_device_aware_and_reserve(
                    request.content(),
                    target_constraint.map(|target| target.worker_id),
                )?;
                let initial_worker = selection.worker_id();
                let candidate_count = selection.candidate_count();
                let selected_occupancy = Some(selection.load());
                let device_aware_telemetry = Some(DeviceAwareTelemetry {
                    is_cpu: selection.is_cpu(),
                    embedding_cache_hit: selection.embedding_cache_hit(),
                    request_cache_keys: selection.request_cache_keys(),
                });
                Ok(HostedSelection {
                    initial_worker,
                    target_constraint,
                    occupancy_reservation: selection.into_reservation(),
                    candidate_count,
                    selected_occupancy,
                    device_aware_telemetry,
                })
            }
        }
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
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let explicit = explicit_target(&request, phase)?;
        let is_direct = matches!(&self.policy, RoutingPolicy::Direct);
        if is_direct && explicit.is_none() {
            return Err(invalid_argument(format!(
                "worker ID required for {phase} request in Direct routing mode"
            )));
        }
        let has_affinity_session = self.affinity.is_some() && affinity_id(&request)?.is_some();
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        let (lora_target, lora_fallback, lora_load) =
            match self.select_lora_target(request.content())? {
                Some(selection) => (
                    Some(selection.target),
                    Some(selection.allowed_fallback),
                    Some(selection.load_guard),
                ),
                None => (None, None, None),
            };
        let (selection, mut operation) = if let Some(target) = lora_target {
            (
                HostedSelection {
                    initial_worker: target,
                    target_constraint: None,
                    occupancy_reservation: None,
                    candidate_count: 0,
                    selected_occupancy: None,
                    device_aware_telemetry: None,
                },
                None,
            )
        } else {
            self.select_with_session_affinity(&request, phase, is_query_only, |target| {
                ready(self.select_hosted_worker(&request, target.or(explicit)))
            })
            .await?
        };
        let HostedSelection {
            initial_worker,
            target_constraint,
            occupancy_reservation,
            candidate_count,
            selected_occupancy,
            device_aware_telemetry,
        } = selection;
        let uses_occupancy = self
            .required_worker_inputs()
            .contains(WorkerInputs::OCCUPANCY);
        let mut guard: RequestGuard<Sel> = RequestGuard::new_builtin(
            self.request_metrics.clone(),
            initial_worker,
            occupancy_reservation,
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
        let dispatch_result = if is_direct && !has_affinity_session {
            let target = target_constraint.expect("Direct routing requires an explicit target");
            cancel_on_stop(
                request_context.as_ref(),
                self.inner.direct_within_prepared(
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
                ),
            )
            .await
            .and_then(|result| result)
            .map(|((metadata, target, occupancy), stream)| (metadata, target, occupancy, stream))
        } else if let Some(target) = target_constraint {
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
                self.inner.dispatch_exact(request, target.worker_id),
            )
            .await
            .and_then(|result| result)
            .map(|stream| (metadata, target, selected_occupancy, stream))
        } else if uses_occupancy {
            cancel_on_stop(
                request_context.as_ref(),
                self.inner.dispatch_preselected_prepared(
                    request,
                    initial_worker,
                    |request, worker_id| {
                        let occupancy = guard.retarget_worker(worker_id);
                        let target = AffinityTarget::worker(worker_id);
                        request.routing_mut().dp_rank = None;
                        prepare(request, target).map(|metadata| (metadata, target, occupancy))
                    },
                ),
            )
            .await
            .and_then(|result| result)
            .map(|((metadata, target, occupancy), stream)| (metadata, target, occupancy, stream))
        } else {
            cancel_on_stop(
                request_context.as_ref(),
                self.inner.direct_within_prepared(
                    request,
                    initial_worker,
                    lora_fallback.as_ref(),
                    |request, worker_id| {
                        let occupancy = guard.retarget_worker(worker_id);
                        let target = AffinityTarget::worker(worker_id);
                        request.routing_mut().dp_rank = None;
                        prepare(request, target).map(|metadata| (metadata, target, occupancy))
                    },
                ),
            )
            .await
            .and_then(|result| result)
            .map(|((metadata, target, occupancy), stream)| (metadata, target, occupancy, stream))
        };

        let (metadata, target, final_occupancy, response_stream) = match dispatch_result {
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
        guard.retarget_worker(target.worker_id);
        if let Some(telemetry) = device_aware_telemetry {
            let selection_survived_transport = target.worker_id == initial_worker;
            tracing::info!(
                router_mode = "device-aware-weighted",
                worker_id = target.worker_id,
                candidate_count,
                occupancy = ?final_occupancy,
                endpoint = %self.inner.client.endpoint.id(),
                is_cpu = ?selection_survived_transport.then_some(telemetry.is_cpu),
                embedding_cache_hit = ?selection_survived_transport
                    .then_some(telemetry.embedding_cache_hit),
                request_cache_keys = telemetry.request_cache_keys,
                transport_fallback = !selection_survived_transport,
                "Selected worker"
            );
        } else if uses_occupancy {
            tracing::info!(
                router_mode = self.inner.router_mode().telemetry_label(),
                worker_id = target.worker_id,
                candidate_count,
                occupancy = ?final_occupancy,
                transport_fallback = target.worker_id != initial_worker,
                "Selected worker"
            );
        }
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
        match operation {
            Some(operation) => Ok((metadata, operation.into_stream(target, stream)?)),
            None => Ok((metadata, stream)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn select(selector: &BuiltinWorkerSelector, worker_ids: &[u64]) -> u64 {
        selector
            .select_worker(WorkerSelectionInput::hosted(worker_ids, None))
            .unwrap()
            .worker
            .worker_id
    }

    #[test]
    fn round_robin_uses_hosted_selector_input() {
        let selector = BuiltinWorkerSelector::new(RouterMode::RoundRobin).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::NONE);
        assert_eq!(select(&selector, &[10, 20]), 10);
        assert_eq!(select(&selector, &[10, 20]), 20);
        assert_eq!(select(&selector, &[10, 20]), 10);
    }

    #[test]
    fn random_uses_hosted_selector_input() {
        let selector = BuiltinWorkerSelector::new(RouterMode::Random).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::NONE);
        for _ in 0..32 {
            assert!(matches!(select(&selector, &[10, 20]), 10 | 20));
        }
    }

    #[test]
    fn occupancy_policies_require_lazy_occupancy_input() {
        for mode in [RouterMode::PowerOfTwoChoices, RouterMode::LeastLoaded] {
            let selector = BuiltinWorkerSelector::new(mode).unwrap();
            assert_eq!(selector.required_worker_inputs(), WorkerInputs::OCCUPANCY);
            assert!(
                selector
                    .select_worker(WorkerSelectionInput::hosted(&[10, 20], None))
                    .is_err()
            );
        }
    }

    #[test]
    fn least_loaded_reads_hosted_occupancy() {
        let selector = BuiltinWorkerSelector::new(RouterMode::LeastLoaded).unwrap();
        let occupancy = |worker_id| if worker_id == 10 { 4 } else { 1 };
        let selected = selector
            .select_worker(WorkerSelectionInput::hosted(&[10, 20], Some(&occupancy)))
            .unwrap();
        assert_eq!(selected.worker.worker_id, 20);
    }

    #[test]
    fn power_of_two_choices_reads_only_two_occupancies() {
        let selector = BuiltinWorkerSelector::new(RouterMode::PowerOfTwoChoices).unwrap();
        let reads = Cell::new(0);
        let occupancy = |_| {
            reads.set(reads.get() + 1);
            0
        };

        selector
            .select_worker(WorkerSelectionInput::hosted(
                &[10, 20, 30, 40],
                Some(&occupancy),
            ))
            .unwrap();

        assert_eq!(reads.get(), 2);
    }
}

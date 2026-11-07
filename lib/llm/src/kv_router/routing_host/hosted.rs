// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::selector::HostedSelectionRequest;

/// The request metadata a hosted picker reads. The routing host tracks no KV block size, so
/// [`request_blocks`](dynamo_kv_router::plugins::worker_selection::WorkerSelectionContext::request_blocks) reads 0.
pub(super) fn hosted_request(request: &PreprocessedRequest) -> HostedSelectionRequest {
    HostedSelectionRequest::new(request.token_ids.len())
}

use super::*;

impl RoutingHost {
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
            .picker
            .select(&candidates, None, &hosted_request(request))?;
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

    /// The DP rank a hard session binding uses on `worker_id`: the bound rank, or one
    /// chosen from the worker's declared ranks by the binding key so every turn agrees.
    fn hard_affinity_dp_rank(
        &self,
        worker_id: u64,
        bound: Option<AffinityTarget>,
        binding_key: &str,
    ) -> Option<u32> {
        if let Some(dp_rank) = bound
            .filter(|bound| bound.worker_id == worker_id)
            .and_then(|bound| bound.dp_rank)
        {
            return Some(dp_rank);
        }
        let dp_ranks = self
            .routing_context
            .as_ref()?
            .monitor()?
            .declared_dp_ranks(worker_id)?;
        let dp_size = dp_ranks.len() as u64;
        (dp_size > 1).then(|| {
            dp_ranks.start + (xxhash_rust::xxh3::xxh3_64(binding_key.as_bytes()) % dp_size) as u32
        })
    }

    pub(super) fn select_hosted_worker(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        target_constraint: Option<AffinityTarget>,
        affinity_target: Option<AffinityTarget>,
    ) -> Result<HostedSelection, Error> {
        let preferred_worker = || match affinity_target {
            Some(target) => self.inner.with_selectable_worker_ids(|ids| {
                ids.binary_search(&target.worker_id)
                    .is_ok()
                    .then_some(target.worker_id)
            }),
            None => Ok(None),
        };
        match &self.policy {
            RoutingPolicy::Kv(_) => unreachable!("hosted selection called for KV routing"),
            RoutingPolicy::Direct => {
                let target = target_constraint.ok_or_else(|| {
                    invalid_argument("Direct routing requires an exact affinity or request target")
                })?;
                Ok(HostedSelection {
                    initial_worker: target.worker_id,
                    target_constraint: Some(target),
                    occupancy_reservation: None,
                    candidate_count: 1,
                    selected_occupancy: None,
                    device_aware_telemetry: None,
                })
            }
            RoutingPolicy::Hosted(picker) => {
                let preferred_worker = preferred_worker()?;
                if picker.inputs().contains(WorkerInputs::ACTIVE_REQUESTS) {
                    let occupancy = self
                        .hosted_occupancy
                        .as_ref()
                        .expect("ACTIVE_REQUESTS policy must have hosted occupancy state");
                    let selection = occupancy.select_and_reserve(
                        &self.inner,
                        picker,
                        request.content(),
                        target_constraint
                            .map(|target| target.worker_id)
                            .or(preferred_worker),
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
                        None => self.inner.with_selectable_worker_ids(|ids| {
                            if let Some(worker_id) = preferred_worker {
                                Ok(worker_id)
                            } else {
                                picker.select(ids, None, &hosted_request(request.content()))
                            }
                        })??,
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
                let preferred_worker = preferred_worker()?;
                let selection = self.inner.select_device_aware_and_reserve(
                    request.content(),
                    target_constraint
                        .map(|target| target.worker_id)
                        .or(preferred_worker),
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
        let explicit = explicit_target(request.content(), phase)?;
        let has_affinity_session = self.affinity.is_some() && affinity_id(&request)?.is_some();
        let is_direct = matches!(&self.policy, RoutingPolicy::Direct);
        if is_direct && explicit.is_none() && !has_affinity_session {
            return Err(invalid_argument(format!(
                "worker ID required for {phase} request in Direct routing mode"
            )));
        }
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        // Declared before worker selection because the session-affinity wait is
        // the first stage that can spend it.
        let budget = CleanupBudget::default();
        let staged_kv = StagedKv::for_request(request.content());
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
            self.validate_explicit_worker(request.content(), phase)?;
            self.select_with_session_affinity(&request, phase, is_query_only, &budget, |target| {
                let pinned_target = explicit.or(match self.session_affinity_mode {
                    SessionAffinityMode::Hard => target,
                    SessionAffinityMode::Soft if is_direct => target,
                    SessionAffinityMode::Soft => None,
                });
                let affinity_target = match (explicit, self.session_affinity_mode) {
                    (None, SessionAffinityMode::Soft) => target,
                    _ => None,
                };
                ready(self.select_hosted_worker(&request, pinned_target, affinity_target))
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
        let bound_target = operation.as_ref().and_then(Hold::target).map(from_table);
        let hard_binding_key = operation
            .as_ref()
            .filter(|_| self.session_affinity_mode == SessionAffinityMode::Hard)
            .map(Hold::session_id);
        let target_for_worker = |worker_id| {
            let dp_rank = match hard_binding_key {
                Some(binding_key) => {
                    self.hard_affinity_dp_rank(worker_id, bound_target, binding_key)
                }
                None => bound_target
                    .filter(|target| target.worker_id == worker_id)
                    .and_then(|target| target.dp_rank),
            };
            AffinityTarget::new(worker_id, dp_rank)
        };
        let target_constraint = target_constraint.map(|target| match target.dp_rank {
            None if hard_binding_key.is_some() => target_for_worker(target.worker_id),
            _ => target,
        });
        let uses_occupancy = self
            .required_worker_inputs()
            .contains(WorkerInputs::ACTIVE_REQUESTS);
        let mut guard: RequestGuard = RequestGuard::new_builtin(
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
            .with_label_values(&[request.phase().as_str(), &request.model])
            .observe(request.token_ids.len() as f64);
        drop(route_guard);

        guard.begin_dispatch(&phase_label, request.content());
        let dispatch_result = if is_direct && !has_affinity_session {
            let target = target_constraint.expect("Direct routing requires an explicit target");
            await_with_cleanup_policy(
                request_context.as_ref(),
                phase,
                staged_kv,
                "builtin.dispatch_direct",
                &budget,
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
                    guard.abort(Some(error.as_ref())).await;
                    return Err(error);
                }
            };
            await_with_cleanup_policy(
                request_context.as_ref(),
                phase,
                staged_kv,
                "builtin.dispatch_exact",
                &budget,
                self.inner.dispatch_exact(request, target.worker_id),
            )
            .await
            .and_then(|result| result)
            .map(|stream| (metadata, target, selected_occupancy, stream))
        } else if uses_occupancy {
            await_with_cleanup_policy(
                request_context.as_ref(),
                phase,
                staged_kv,
                "builtin.dispatch_occupancy",
                &budget,
                self.inner.dispatch_preselected_prepared(
                    request,
                    initial_worker,
                    |request, worker_id| {
                        let occupancy = guard.retarget_worker(worker_id);
                        let target = target_for_worker(worker_id);
                        request.routing_mut().dp_rank = target.dp_rank;
                        prepare(request, target).map(|metadata| (metadata, target, occupancy))
                    },
                ),
            )
            .await
            .and_then(|result| result)
            .map(|((metadata, target, occupancy), stream)| (metadata, target, occupancy, stream))
        } else {
            await_with_cleanup_policy(
                request_context.as_ref(),
                phase,
                staged_kv,
                "builtin.dispatch",
                &budget,
                self.inner.direct_within_prepared(
                    request,
                    initial_worker,
                    lora_fallback.as_ref(),
                    |request, worker_id| {
                        let occupancy = guard.retarget_worker(worker_id);
                        let target = target_for_worker(worker_id);
                        request.routing_mut().dp_rank = target.dp_rank;
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
                let expected_target =
                    target_constraint.unwrap_or_else(|| target_for_worker(initial_worker));
                if self.session_affinity_mode == SessionAffinityMode::Hard
                    && !self.affinity_target_is_valid(expected_target)
                    && let Some(operation) = operation.take()
                {
                    operation.invalidate();
                }
                let typed_error = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<DynamoError>().cloned());
                guard.record_migration_failure(typed_error);
                guard.abort(Some(error.as_ref())).await;
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
        } else if target_constraint.is_some() && (!is_direct || has_affinity_session) {
            // `dispatch_exact` (session-affinity hits and explicit targets) logs nothing, while
            // the other non-occupancy paths log from `PushRouter`; keep one line per request.
            tracing::info!(
                router_mode = self.inner.router_mode().telemetry_label(),
                worker_id = target.worker_id,
                dp_rank = ?target.dp_rank,
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
        self.accept_dispatch(
            &mut guard,
            operation.take(),
            target,
            &response_stream.context(),
        )
        .await?;
        Ok((metadata, into_monitored_response(response_stream, guard)))
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::selector::HostedWorkerPicker;
    use dynamo_runtime::pipeline::RouterMode;

    use super::super::builder::hosted_picker;
    use super::*;

    fn request() -> HostedSelectionRequest {
        HostedSelectionRequest::new(0)
    }

    fn select(picker: &HostedWorkerPicker, worker_ids: &[u64]) -> u64 {
        picker.select(worker_ids, None, &request()).unwrap()
    }

    #[test]
    fn round_robin_rotates_and_previews_do_not_advance() {
        let picker = hosted_picker(RouterMode::RoundRobin).unwrap();
        assert_eq!(picker.inputs(), WorkerInputs::NONE);
        let peek = |picker: &HostedWorkerPicker| {
            picker
                .select(&[10, 20], None, &request().with_preview(true))
                .unwrap()
        };
        assert_eq!(peek(&picker), 10);
        assert_eq!(peek(&picker), 10);
        assert_eq!(select(&picker, &[10, 20]), 10);
        assert_eq!(select(&picker, &[10, 20]), 20);
        assert_eq!(select(&picker, &[10, 20]), 10);
    }

    #[test]
    fn random_picks_an_eligible_worker() {
        let picker = hosted_picker(RouterMode::Random).unwrap();
        assert_eq!(picker.inputs(), WorkerInputs::NONE);
        for _ in 0..32 {
            assert!(matches!(select(&picker, &[10, 20]), 10 | 20));
        }
    }

    #[test]
    fn least_loaded_requires_active_requests() {
        let picker = hosted_picker(RouterMode::LeastLoaded).unwrap();
        assert_eq!(picker.inputs(), WorkerInputs::ACTIVE_REQUESTS);
        assert!(picker.select(&[10, 20], None, &request()).is_err());
    }

    #[test]
    fn least_loaded_reads_active_requests() {
        let picker = hosted_picker(RouterMode::LeastLoaded).unwrap();
        let active = |worker_id| if worker_id == 10 { 4 } else { 1 };
        let selected = picker.select(&[10, 20], Some(&active), &request()).unwrap();
        assert_eq!(selected, 20);
    }

    #[test]
    fn non_plugin_modes_have_no_hosted_policy() {
        for mode in [
            RouterMode::KV,
            RouterMode::Direct,
            RouterMode::DeviceAwareWeighted,
            RouterMode::PowerOfTwoChoices,
        ] {
            assert!(hosted_picker(mode).is_err());
        }
    }
}

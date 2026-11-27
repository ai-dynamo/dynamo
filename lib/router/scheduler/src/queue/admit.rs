// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One admission on the actor: select a worker, book it, respond.

use super::*;

struct SelectedWorkerForRequest {
    selection: WorkerSelectionResult,
    selection_trace: Option<SelectionTrace>,
    selected_worker_tiers: SelectedWorkerTierSnapshot,
    selected_worker_load: AdvisoryWorkerLoad,
    non_max_overlap_selection: Option<NonMaxOverlapSelection>,
}

impl SelectedWorkerForRequest {
    /// The response for this selection, which takes the request's fetch candidates with it.
    fn into_response(self, request: &mut SchedulingRequest) -> SchedulingResponse {
        let selection = self.selection;
        SchedulingResponse {
            best_worker: selection.worker,
            effective_overlap_blocks: selection.effective_overlap_blocks,
            cached_tokens: selection.cached_tokens,
            max_raw_cached_tokens: selection.max_raw_cached_tokens,
            selected_raw_cached_tokens: selection.selected_raw_cached_tokens,
            selected_worker_tiers: self.selected_worker_tiers,
            kv_copy_source: request.kv_copy_source(selection.worker, selection.kv_transfer_source),
            kv_transfer_candidates: request.kv_transfer_candidates.take(),
            potential_decode_blocks: selection.potential_decode_blocks,
            selection_trace: self.selection_trace,
            log_fields: selection.log_fields,
        }
    }
}

pub(super) fn non_max_overlap_selection<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    selected_worker: WorkerWithDpRank,
    selected_overlap_blocks: f64,
) -> Option<NonMaxOverlapSelection> {
    if eligibility.pinned_worker().is_some() {
        return None;
    }

    let mut highest_overlap = None;
    for (&worker, &overlap_blocks) in &request.overlap.effective_overlap_blocks {
        if overlap_blocks <= selected_overlap_blocks
            || eligibility.validate_worker_rank(workers, worker).is_err()
        {
            continue;
        }
        let is_better = highest_overlap.is_none_or(
            |(current_worker, current_overlap): (WorkerWithDpRank, f64)| {
                overlap_blocks > current_overlap
                    || (overlap_blocks == current_overlap && worker < current_worker)
            },
        );
        if is_better {
            highest_overlap = Some((worker, overlap_blocks));
        }
    }

    let (highest_overlap_worker, highest_overlap_blocks) = highest_overlap?;
    (highest_overlap_blocks > selected_overlap_blocks).then_some(NonMaxOverlapSelection {
        selected_worker,
        highest_overlap_worker,
        highest_overlap_blocks,
        selected_overlap_blocks,
    })
}

/// Highest raw cached-prefix tokens over workers eligible under `eligibility`, starting
/// from the selector's own maximum. Workers with no overlap contribute zero, so only the
/// workers present in the tier overlap maps are visited.
fn best_eligible_raw_cached_tokens<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    selector_max: usize,
    block_size: u32,
) -> usize {
    let tiers = &request.overlap.tier_overlap_blocks;
    tiers
        .device
        .keys()
        .chain(tiers.host_pinned.keys())
        .chain(tiers.disk.keys())
        .filter(|&&worker| eligibility.validate_worker_rank(workers, worker).is_ok())
        .map(|&worker| request.raw_cached_tokens_for(worker, block_size))
        .fold(selector_max, usize::max)
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    /// `admitted` selections may carry the installed selection trace; advisory
    /// probes never do, so they do not pay for capture.
    fn select_worker_for_request(
        &self,
        request: &SchedulingRequest,
        admitted: bool,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> Result<SelectedWorkerForRequest, KvSchedulerError> {
        self.eligibility.with(
            request,
            &self.workers_with_configs,
            &self.full_ranks,
            true,
            |workers, eligibility, _, available_worker_ids, overloaded_worker_ids| {
                let prepared = self.adopt_prepared_rows(
                    prepared,
                    request,
                    workers,
                    available_worker_ids,
                    overloaded_worker_ids,
                );
                let trace = if admitted {
                    self.selection_trace.get().copied()
                } else {
                    None
                };
                let progress = |worker| self.slots.decode_progress_for(worker);
                // Held for this selection, so the snapshot the policy reads borrows live counters.
                let resident_blocks = self
                    .resident_blocks
                    .get()
                    .map(ResidentBlockCountsHandle::load);
                self.selector
                    .select_worker_prepared(
                        WorkerSelectionInput::configured(
                            workers,
                            request,
                            eligibility,
                            self.block_size,
                        )
                        .with_resident_blocks(resident_blocks.as_deref().map(|counts| &**counts))
                        .with_decode_progress(Some(&progress)),
                        prepared,
                        trace,
                    )
                    .map(|(mut selection, selection_trace)| {
                        // A projection without overlap counts none of the request's blocks as
                        // new on any rank. Count them for the selected rank alone.
                        if !self.projects_overlap()
                            && let Some(token_seq) = request.token_seq.as_deref()
                            && request.worker_loads.contains_key(&selection.worker)
                        {
                            selection.potential_decode_blocks += self
                                .slots
                                .additional_active_blocks(token_seq, selection.worker);
                        }
                        // Measure affinity selections against every eligible worker so cache
                        // metrics still mean "best eligible worker". Explicit pins are not
                        // widened, nor are selectors whose own filters may reject workers.
                        let affinity_selected = request
                            .affinity_target
                            .is_some_and(|target| target.matches(selection.worker));
                        let metrics_eligibility = if affinity_selected
                            && !self.selector.applies_worker_filters()
                        {
                            let unconstrained = RoutingEligibility::new(
                                request.allowed_worker_ids.as_ref(),
                                overloaded_worker_ids.map(|ids| ids.as_ref()),
                                None,
                                &request.routing_constraints,
                            )
                            .with_available_workers(available_worker_ids.map(|ids| ids.as_ref()))
                            .with_full_worker_ranks(&self.full_ranks);
                            if let Some(max_raw_cached_tokens) =
                                &mut selection.max_raw_cached_tokens
                            {
                                *max_raw_cached_tokens = best_eligible_raw_cached_tokens(
                                    workers,
                                    request,
                                    unconstrained,
                                    *max_raw_cached_tokens,
                                    self.block_size,
                                );
                            }
                            unconstrained
                        } else {
                            eligibility
                        };
                        let non_max_overlap_selection = if request.mode.is_tracked()
                            && self.non_max_overlap_selection_observer.get().is_some()
                        {
                            non_max_overlap_selection(
                                workers,
                                request,
                                metrics_eligibility,
                                selection.worker,
                                selection.effective_overlap_blocks,
                            )
                        } else {
                            None
                        };
                        let config = workers
                            .get(&selection.worker.worker_id)
                            .expect("selected worker config must exist");
                        let selected_worker_tiers = request
                            .overlap
                            .selected_worker_tiers(selection.worker, config);
                        let worker_load = request.worker_load_for(selection.worker);
                        let capacity = WorkerCapacity::of(config);
                        let selected_worker_load = AdvisoryWorkerLoad {
                            active_prefill_tokens: worker_load.active_prefill_tokens,
                            prefill_token_capacity: capacity.prefill_budget() as usize,
                            // TODO(rank-aware-kv-capacity): resolve the selected DP rank and preserve
                            // capacity quality. Estimated fallbacks must not authorize load-based
                            // admission/bypass decisions that require an exact or conservative bound.
                            total_kv_blocks: capacity
                                .total_kv_blocks()
                                .map(|blocks| blocks as usize),
                        };
                        SelectedWorkerForRequest {
                            selection,
                            selection_trace,
                            selected_worker_tiers,
                            selected_worker_load,
                            non_max_overlap_selection,
                        }
                    })
            },
        )
    }

    pub(super) fn select_without_admission_inner(
        &mut self,
        request: SchedulingRequest,
        decay_now: Instant,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        self.with_projected_loads(request, decay_now, None, |actor, request, _| {
            let selected = actor.select_worker_for_request(request, false, None)?;
            Ok(AdvisorySchedulingResponse {
                selected_worker_load: selected.selected_worker_load,
                response: selected.into_response(request),
            })
        })
    }

    /// Run the full scheduling pipeline for a single request of the context's class:
    /// compute projected load -> select worker -> book tracked state -> respond.
    /// A request whose due time passes before selection, booking, or delivery is rejected
    /// instead.
    pub(super) fn admit_one(
        &mut self,
        context: AdmissionContext,
        request: SchedulingRequest,
        decay_now: Instant,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> Admission {
        if prepared.is_some() {
            self.refresh_workers_generation();
        }
        self.with_projected_loads(request, decay_now, prepared, |actor, request, prepared| {
            actor.admit_one_projected(context, request, prepared)
        })
    }

    pub(super) fn admit_one_projected(
        &self,
        context: AdmissionContext,
        request: &mut SchedulingRequest,
        prepared: Option<Box<PreparedAdmission>>,
    ) -> Admission {
        // The dispatch gate and load projection ran after the drain read the clock. Check the
        // due time again so a failing selection cannot hide an expiry.
        if context
            .due_at
            .is_some_and(|due_at| due_at <= Instant::now())
        {
            self.reject_due_time_passed(context.class_index, request);
            return Admission::Refused;
        }
        let mut selected = match self.select_worker_for_request(request, true, prepared) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                return Admission::Refused;
            }
        };

        let non_max_overlap_selection = selected.non_max_overlap_selection.take();
        let response = selected.into_response(request);

        if !request.mode.is_tracked() {
            if context
                .due_at
                .is_some_and(|due_at| due_at <= Instant::now())
            {
                self.reject_due_time_passed(context.class_index, request);
                return Admission::Refused;
            }
            return if request.respond(Ok(response)) {
                Admission::Selected
            } else {
                Admission::Refused
            };
        }

        let request_id = request
            .mode
            .tracked_request_id()
            .expect("tracked mode always has a request ID")
            .to_string();

        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            response.cached_tokens,
            request.track_prefill_tokens,
        );

        let sequence_request = SequenceRequest {
            request_id,
            token_sequence: request.token_seq.take(),
            track_prefill_tokens: request.track_prefill_tokens,
            expected_output_tokens: request.expected_output_tokens,
            prefill_load_hint,
            worker: response.best_worker,
            lora_name: request
                .accounting_attribution
                .accounting_name(request.lora_name.take()),
        };
        self.book_and_respond(
            context,
            request,
            sequence_request,
            response,
            non_max_overlap_selection,
        )
    }

    /// A closed receiver means the actor-owned request was abandoned before
    /// booking, so there is nothing to install. A request whose `due_at` has passed is
    /// rejected unbooked, judged on the same instant that would stamp its booking. Otherwise
    /// booking precedes the response: once delivery succeeds, the response channel no longer
    /// tracks request lifetime and the caller must install its RAII cleanup owner. If
    /// delivery loses that race, roll back the booking here.
    fn book_and_respond(
        &self,
        context: AdmissionContext,
        request: &mut SchedulingRequest,
        sequence_request: SequenceRequest,
        response: SchedulingResponse,
        non_max_overlap_selection: Option<NonMaxOverlapSelection>,
    ) -> Admission {
        let AdmissionContext {
            class_index,
            due_at,
            attempt_tx,
            lifecycle_transfer,
        } = context;
        if request.response_is_closed() {
            tracing::debug!(
                request_id = %sequence_request.request_id,
                "Skipping scheduler booking for cancelled request"
            );
            return Admission::Refused;
        }

        let admitted_at = Instant::now();
        if due_at.is_some_and(|due_at| due_at <= admitted_at) {
            self.reject_due_time_passed(class_index, request);
            return Admission::Refused;
        }
        let request_id = sequence_request.request_id.clone();
        let worker = sequence_request.worker;
        let attempt_id = match self
            .slots
            .add_request_admitted(sequence_request, admitted_at)
        {
            Ok(attempt_id) => attempt_id,
            Err(error) => {
                tracing::warn!(%request_id, %error, "Failed to book scheduler state");
                request.respond(Err(KvSchedulerError::BookingFailed(error.to_string())));
                return Admission::Refused;
            }
        };
        if let Some(transfer) = lifecycle_transfer {
            transfer.arm_booking(SchedulerBookingDescriptor {
                request_id: request_id.clone(),
                worker,
                attempt_id,
            });
        }
        if let Some(attempt_tx) = attempt_tx {
            let _ = attempt_tx.send(attempt_id);
        }
        if request.respond(Ok(response)) {
            if let Some(selection) = non_max_overlap_selection
                && let Some(observer) = self.non_max_overlap_selection_observer.get()
            {
                let observer = Arc::clone(observer);
                let _observer_task =
                    tokio::task::spawn_blocking(move || observer(&request_id, selection));
            }
            return Admission::Booked;
        }

        tracing::debug!(%request_id, "Rolling back undelivered scheduler booking");
        if let Err(error) =
            self.slots
                .free_if_booking(&request_id, worker, attempt_id, Instant::now())
        {
            tracing::error!(%request_id, %error, "Failed to roll back scheduler booking");
        }
        Admission::Refused
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        cached_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let effective_isl = effective_prefill_tokens(isl_tokens, cached_tokens);
        if effective_isl == 0 {
            return None;
        }
        let prefix = isl_tokens - effective_isl;

        let expected_prefill_duration = match &self.prefill_load_estimator {
            Some(estimator) => match estimator.predict_prefill_duration(1, effective_isl, prefix) {
                Ok(expected_prefill_duration) => Some(expected_prefill_duration),
                Err(error) => {
                    tracing::warn!(
                        effective_isl,
                        prefix,
                        "failed to predict prefill duration for active load tracking: {error}"
                    );
                    None
                }
            },
            None => None,
        };

        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: effective_isl,
            expected_prefill_duration,
        })
    }
}

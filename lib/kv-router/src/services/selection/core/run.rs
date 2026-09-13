// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One selection, end to end: the wire adapters, `run_selection`, the
//! lookups that feed the scheduler, and the session binding around it.
//!
//! # Booking lifecycle invariants
//!
//! A booking is `(request_id, worker, attempt_id)` in the partition scheduler.
//! Both booking paths, `run_selection_inner` here and `finalize_reservation`
//! in `core/reservations.rs`, hold it through these invariants; the named
//! tests in `core/tests.rs` pin each one.
//!
//! - A booking is released exactly once, by whichever owner holds it when the
//!   request ends (`dropped_selection_future_frees_its_booking`,
//!   `reservation_index_tracks_bookings_until_freed`).
//! - Release ownership moves in one direction: the scheduler actor books, hands
//!   the armed handle to this run, and the run hands it to the index row for
//!   `Book` (`claim.install`) or to the host for `Lease` (`Selected::booking`)
//!   (`dropped_selection_future_frees_its_booking`,
//!   `lease_admission_installs_no_index_row_and_records_nothing`).
//! - The session hold is taken before the booking is made
//!   (`session_worker_departing_after_the_hold_reinitializes_the_session`) and
//!   the binding is committed while the booking is held. The commit never
//!   waits: `commit_session` is a synchronous fn that re-binds only through
//!   `try_acquire`. A commit that finds another request initializing the
//!   session joins that initialization instead of waiting: it takes a lease
//!   that counts once the initializer commits
//!   (`failover_commit_behind_an_initializing_hold_keeps_a_lease`); a joiner
//!   that finishes first is uncounted again
//!   (`joined_lease_released_before_the_commit_is_not_counted`). When the
//!   table is full, or the join misses twice, it returns `Ok(None)` and
//!   the request routes unpinned rather than waiting on a request queued
//!   behind this booking. A rejected
//!   commit frees the booking
//!   (`two_phase_replay_rejects_a_worker_the_session_left`).
//! - A `Lease` admission installs no index row and records no routing hashes
//!   in the core; the host owns both
//!   (`lease_admission_installs_no_index_row_and_records_nothing`).
//! - Dropping the run at any await after the booking lands and before the
//!   index row is installed frees the booking and releases the id claim
//!   (`dropped_selection_future_frees_its_booking`,
//!   `dropped_book_selection_during_routing_record_frees_booking_and_claim`).

use super::hint::{hint_capable_partition, transfer_hint_for_selection};

/// `try_acquire` then `join_initializing` attempts before a commit that finds
/// the session initializing routes unpinned (see the module doc).
const JOIN_ATTEMPTS: usize = 2;
use super::reservations::Reservation;
use super::*;

/// Action id of the single `kv.fetch` action a selection's KV hint carries.
const KV_HINT_FETCH_ACTION_ID: &str = "a1";

pub(super) struct PreparedSelectionInputs {
    pub(super) block_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
    pub(super) isl_tokens: usize,
    pub(super) overlap: OverlapSignals,
    pub(super) shared_cache_hits: Option<SharedCacheHits>,
    pub(super) kv_transfer_candidates: Option<KvTransferCandidates>,
}

impl SelectionCore {
    pub async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        self.select_with_policy_class(req, None).await
    }

    pub async fn select_with_policy_class(
        &self,
        mut req: SelectRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        let session_context = req.take_session_context();
        let is_steerable = req.affinity_target.is_none() && req.pinned_worker.is_none();
        let session = session_binding(session_context.as_ref(), is_steerable, false);
        let admission = if req.advisory {
            SelectionAdmission::Advisory {
                request_id: req.selection_id.clone(),
            }
        } else {
            SelectionAdmission::Query {
                request_id: req.selection_id.clone(),
            }
        };
        let (selected, endpoint) = self
            .select_or_reject(SelectionOperation {
                key: RoutingPartitionId::new(req.model_name, req.routing_group),
                prompt: req.prompt.view(),
                router_config_override: req.router_config_override,
                expected_output_tokens: req.expected_output_tokens,
                priority_jump: req.priority_jump.unwrap_or_default(),
                strict_priority: req.strict_priority.unwrap_or(0),
                policy_class,
                session_context,
                session,
                affinity_target: req.affinity_target,
                pinned_worker: req.pinned_worker,
                allowed_worker_ids: req.allowed_worker_ids,
                routing_constraints: req.routing_constraints,
                admission,
                track_active_blocks: true,
                return_routing_hashes: false,
                replay_id: req.selection_id.clone(),
            })
            .await?;
        Ok(self.select_response(selected, endpoint, req.selection_id))
    }

    pub async fn select_and_reserve(
        &self,
        req: SelectAndReserveRequest,
    ) -> Result<SelectResponse, SelectionError> {
        self.select_and_reserve_with_policy_class(req, None).await
    }

    pub async fn select_and_reserve_with_policy_class(
        &self,
        mut req: SelectAndReserveRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        let session_context = req.take_session_context();
        let selection_id = req
            .selection_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let is_steerable = req.affinity_target.is_none() && req.pinned_worker.is_none();
        let session = session_binding(session_context.as_ref(), is_steerable, true);
        let (selected, endpoint) = self
            .select_or_reject(SelectionOperation {
                key: RoutingPartitionId::new(req.model_name, req.routing_group),
                prompt: req.prompt.view(),
                router_config_override: req.router_config_override,
                expected_output_tokens: req.expected_output_tokens,
                priority_jump: req.priority_jump.unwrap_or_default(),
                strict_priority: req.strict_priority.unwrap_or(0),
                policy_class,
                session_context,
                session,
                affinity_target: req.affinity_target,
                pinned_worker: req.pinned_worker,
                allowed_worker_ids: req.allowed_worker_ids,
                routing_constraints: req.routing_constraints,
                admission: SelectionAdmission::Book {
                    selection_id: selection_id.clone(),
                },
                track_active_blocks: true,
                return_routing_hashes: false,
                replay_id: None,
            })
            .await?;
        Ok(self.select_response(selected, endpoint, Some(selection_id)))
    }

    /// The wire shape of a selection; a queue rejection keeps its 503 body.
    async fn select_or_reject(
        &self,
        operation: SelectionOperation<'_>,
    ) -> Result<(Selected, String), SelectionError> {
        let mut selected = match self.run_selection(operation).await.result? {
            SelectionOutcome::Selected(selected) => selected,
            SelectionOutcome::QueueRejected { rejection } => {
                return Err(SelectionError::Scheduler(KvSchedulerError::QueueRejected(
                    rejection,
                )));
            }
        };
        // Wire admissions never book without an endpoint; see `Selected::endpoint`.
        let endpoint = selected.endpoint.take().ok_or_else(|| {
            SelectionError::Internal(format!(
                "selected worker {} is no longer schedulable",
                selected.response.best_worker.worker_id
            ))
        })?;
        Ok((selected, endpoint))
    }

    fn select_response(
        &self,
        selected: Selected,
        endpoint: String,
        selection_id: Option<String>,
    ) -> SelectResponse {
        let Selected {
            key,
            response,
            advisory_load,
            total_kv_blocks,
            endpoint: _,
            block_size,
            isl_tokens,
            sequence_hashes,
            track_prefill_tokens,
            effective_prefill_tokens,
            kv_hint,
            routing_hashes: _,
            shared_cache_hits: _,
            booking: _,
        } = selected;
        let booked = sequence_hashes.is_some();
        let potential_decode_blocks = response.potential_decode_blocks as u64;
        let decode_busy = self
            .kv_router_config
            .conditional_disagg_decode_busy_threshold
            .zip(total_kv_blocks)
            .map(|(threshold, total_kv_blocks)| {
                potential_decode_blocks as f64 > threshold * total_kv_blocks as f64
            });
        let worker_load = advisory_load.map(|load| SelectionWorkerLoad {
            active_prefill_tokens: load.active_prefill_tokens,
            prefill_token_capacity: load.prefill_token_capacity,
            total_kv_blocks,
            prefill_busy: self
                .kv_router_config
                .conditional_disagg_prefill_busy_threshold
                .map(|threshold| load.prefill_load_exceeds(threshold)),
        });
        SelectResponse {
            selection_id,
            sequence_hashes: sequence_hashes
                .map(|hashes| hashes.into_iter().map(|hash| hash as i64).collect()),
            isl_tokens: booked.then_some(isl_tokens),
            track_prefill_tokens: booked.then_some(track_prefill_tokens),
            model_name: key.model_name,
            routing_group: key.routing_group,
            worker_id: response.best_worker.worker_id,
            dp_rank: response.best_worker.dp_rank,
            endpoint,
            block_size,
            overlap: MooncakeOverlapSummary::from_selected_worker_tiers(
                &response.selected_worker_tiers,
                block_size,
            ),
            effective_prefill_tokens,
            potential_decode_blocks,
            decode_busy,
            worker_load,
            kv_hint,
        }
    }

    /// Run one selection: resolve the session, look the prompt up, schedule,
    /// and (for `Book`) install the reservation. Every host's selection goes
    /// through here.
    pub async fn run_selection(&self, operation: SelectionOperation<'_>) -> SelectionRun {
        let mut lookup = None;
        let result = self.run_selection_inner(operation, &mut lookup).await;
        SelectionRun { result, lookup }
    }

    async fn run_selection_inner(
        &self,
        operation: SelectionOperation<'_>,
        lookup: &mut Option<LookupTimings>,
    ) -> Result<SelectionOutcome, SelectionError> {
        let SelectionOperation {
            key,
            prompt,
            router_config_override,
            expected_output_tokens,
            priority_jump,
            strict_priority,
            policy_class,
            session_context,
            session,
            affinity_target,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            admission,
            track_active_blocks,
            return_routing_hashes,
            replay_id,
        } = operation;
        self.ensure_running()?;

        let entry = self.ready_entry(&key)?;
        let book = admission.is_booking();
        let claim = match &admission {
            SelectionAdmission::Book { selection_id } => {
                Some(self.claim_reservation(selection_id, &key)?)
            }
            _ => None,
        };

        // Session stickiness: a bound session steers selection (exclusive for
        // the default selector); a new session is bound to the worker booked.
        let table = entry.affinity.get();
        let mut affinity_hold = None;
        let managed_session = match (&session, table) {
            (SessionBinding::Managed { .. }, _) if claim.is_none() => {
                return Err(SelectionError::Internal(
                    "a managed session binding requires Book admission".to_string(),
                ));
            }
            (SessionBinding::Managed { session_id }, Some(table)) => Some((table, session_id)),
            _ => None,
        };
        let affinity_target = match (&session, table) {
            (SessionBinding::Managed { session_id }, Some(table)) => {
                affinity_hold = self.hold_session(table, session_id, &key).await?;
                affinity_hold.as_ref().and_then(Hold::target)
            }
            (SessionBinding::Query { session_id }, Some(table)) => table
                .query_target(session_id, None)
                .map_err(affinity_error)?,
            _ => affinity_target,
        };
        // Router hints are attached to bookings only, and only when a worker in
        // this partition can consume them and the indexer can retain the
        // matched chain (local, event-driven, no approximate writes).
        // Read from the published partition snapshot, the same view the hint
        // builder uses, rather than scanning the catalog per booking.
        let hint_capable_workers = book && hint_capable_partition(&entry.workers_tx.borrow());
        let retain_kv_transfer_chain =
            hint_capable_workers && entry.indexer.supports_kv_transfer_chain_retention();
        if hint_capable_workers && !retain_kv_transfer_chain {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    "router hints need a local event-driven indexer with no approximate side indexer; workers advertise hint capability but no hints will be attached"
                );
            });
        }
        let PreparedSelectionInputs {
            block_hashes,
            sequence_hashes,
            isl_tokens,
            overlap,
            shared_cache_hits,
            kv_transfer_candidates,
        } = self
            .prepare_selection_inputs(
                &entry,
                &prompt,
                track_active_blocks.then(|| {
                    self.kv_router_config
                        .assume_kv_reuse(router_config_override.as_ref())
                }),
                true,
                retain_kv_transfer_chain,
                lookup,
            )
            .await?;
        if tracing::enabled!(tracing::Level::DEBUG) {
            let local_hashes: Vec<u64> = block_hashes.iter().map(|hash| hash.0).collect();
            tracing::debug!(
                request_id = admission.request_id().unwrap_or_default(),
                isl_tokens,
                block_size = entry.block_size,
                num_blocks = local_hashes.len(),
                ?local_hashes,
                "[ROUTING_INPUT] request local hashes"
            );
        }
        let host_shared_cache_hits = shared_cache_hits.clone();
        // The queue lease frees a booking whose response is never consumed
        // (the caller dropped this future after the actor booked).
        let mode = match &admission {
            SelectionAdmission::Book { selection_id }
            | SelectionAdmission::Lease {
                request_id: selection_id,
            } => ScheduleMode::TrackedWithLifecycle {
                request_id: selection_id.clone(),
            },
            SelectionAdmission::Query { request_id } => ScheduleMode::QueryOnly {
                request_id: request_id.clone(),
            },
            SelectionAdmission::Advisory { request_id } => ScheduleMode::QueryOnly {
                request_id: request_id.clone(),
            },
        };
        let track_prefill_tokens = router_config_override
            .as_ref()
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.kv_router_config.router_track_prefill_tokens);
        // An unbooked selection with an id caches the booking inputs so a
        // follow-up `create_reservation` can replay them by that id.
        let cached_inputs = replay_id.filter(|_| !book).map(|id| {
            (
                id,
                sequence_hashes.clone(),
                prompt.lora_name.map(str::to_string),
                track_prefill_tokens,
                match &session {
                    SessionBinding::Query { session_id } => Some(session_id.clone()),
                    _ => None,
                },
            )
        });
        let allowed_worker_ids = match self.host.eligibility.lora_worker_filter.as_deref() {
            Some(filter) => narrow_allowed_worker_ids_by_lora(
                filter,
                prompt.lora_name,
                allowed_worker_ids,
                pinned_worker.as_ref(),
                || entry.workers_tx.borrow().keys().copied().collect(),
            ),
            None => allowed_worker_ids,
        };
        // Bookings (now, or later via the pending-selection cache) are recorded
        // into an approximate indexer; keep the public hashes for that.
        let routing_hashes = (entry.indexer.records_routing_decisions()
            && (claim.is_some() || cached_inputs.is_some()))
        .then(|| block_hashes.clone());
        // Only `Book` callers read the hashes back (the reservation response);
        // the frontend's `Lease` path discards them.
        let booked_sequence_hashes =
            matches!(admission, SelectionAdmission::Book { .. }).then(|| sequence_hashes.clone());
        let returned_routing_hashes = return_routing_hashes.then(|| block_hashes.clone());
        let schedule_request = ScheduleRequest {
            mode,
            token_seq: track_active_blocks.then_some(sequence_hashes),
            block_hashes: Some(block_hashes),
            isl_tokens,
            overlap,
            kv_transfer_candidates,
            retain_kv_transfer_chain,
            router_config_override,
            lora_name: prompt.lora_name.map(str::to_string),
            priority_jump,
            strict_priority,
            policy_class,
            session_context,
            expected_output_tokens,
            affinity_target,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            shared_cache_hits,
        };
        // `booking` guards the booking until it is installed below: any early
        // return or drop before then frees it.
        let scheduled = tokio::select! {
            biased;
            _ = self.cancel_token.cancelled() => {
                return Err(SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown));
            }
            result = async {
                if matches!(admission, SelectionAdmission::Advisory { .. }) {
                    entry
                        .scheduler
                        .select_without_admission(schedule_request)
                        .instrument(tracing::info_span!("kv_router.select_without_admission"))
                        .await
                        .map(|advisory| {
                            (advisory.response, Some(advisory.selected_worker_load), None)
                        })
                } else {
                    entry
                        .scheduler
                        .schedule_request_with_booking(schedule_request)
                        .instrument(tracing::info_span!("kv_router.schedule"))
                        .await
                        .map(|(admitted, booking)| (admitted.response, None, booking))
                }
            } => result,
        };
        let (response, advisory_load, booking) = match scheduled {
            Ok(scheduled) => scheduled,
            Err(KvSchedulerError::QueueRejected(rejection)) => {
                return Ok(SelectionOutcome::QueueRejected { rejection });
            }
            Err(error) => return Err(error.into()),
        };
        let endpoint = self
            .catalog
            .schedulable_endpoint(response.best_worker.worker_id, &key);
        if endpoint.is_none() && !matches!(admission, SelectionAdmission::Lease { .. }) {
            return Err(SelectionError::Internal(format!(
                "selected worker {} is no longer schedulable",
                response.best_worker.worker_id
            )));
        }
        let effective_prefill = effective_prefill_tokens(isl_tokens, response.cached_tokens);
        let total_kv_blocks = advisory_load
            .and_then(|load| load.total_kv_blocks.map(|blocks| blocks as u64))
            .or_else(|| {
                self.catalog
                    .total_kv_blocks(response.best_worker.worker_id, &key)
            });
        let kv_hint = if retain_kv_transfer_chain && response.kv_transfer_candidates.is_some() {
            transfer_hint_for_selection(
                &entry.workers_tx.borrow(),
                response.best_worker,
                response.target_cached_prefix_blocks,
                response.kv_transfer_candidates.as_ref(),
            )
            .map(|payload| {
                KvHint::new(
                    admission.request_id().unwrap_or_default(),
                    vec![KvHintAction::fetch(KV_HINT_FETCH_ACTION_ID, payload)],
                )
            })
        } else {
            None
        };

        // The routing hashes go to exactly one of: the reservation recorded
        // now, or the replay cache a later reservation records from.
        let mut routing_hashes = routing_hashes;
        let booking = if let Some(claim) = claim {
            let Some(booking) = booking else {
                return Err(SelectionError::Internal(
                    "booked selection has no booking handle".to_string(),
                ));
            };
            // A rejected affinity commit returns while the handle is still armed,
            // so the booking is freed and nothing below is recorded.
            let affinity_lease = match (affinity_hold, managed_session) {
                (Some(hold), Some((table, session_id))) => {
                    self.commit_session(table, hold, session_id, response.best_worker, &key)?
                }
                _ => None,
            };
            if let Some(hashes) = routing_hashes.take() {
                self.record_routing_decision(&entry, response.best_worker, hashes)
                    .await;
            }
            claim.install(Reservation {
                partition: key.clone(),
                booking: Some(booking.commit()),
                _affinity_lease: affinity_lease,
            });
            None
        } else {
            booking
        };

        if let Some((cache_id, sequence_hashes, lora_name, track_prefill_tokens, session_id)) =
            cached_inputs
        {
            self.selection_cache.insert(
                cache_id,
                PendingSelection {
                    key: key.clone(),
                    worker: response.best_worker,
                    sequence_hashes,
                    isl_tokens,
                    effective_prefill_tokens: effective_prefill,
                    expected_output_tokens,
                    track_prefill_tokens,
                    lora_name,
                    routing_hashes,
                    session_id,
                },
                Instant::now(),
            );
        }
        Ok(SelectionOutcome::Selected(Selected {
            key,
            response,
            advisory_load,
            total_kv_blocks,
            endpoint,
            block_size: entry.block_size,
            isl_tokens,
            sequence_hashes: booked_sequence_hashes,
            track_prefill_tokens,
            effective_prefill_tokens: effective_prefill,
            kv_hint,
            routing_hashes: returned_routing_hashes,
            shared_cache_hits: host_shared_cache_hits,
            booking,
        }))
    }

    /// Hold `session_id` for a booking, re-initializing a binding whose worker
    /// this partition can no longer schedule. `None` when the table is full: a
    /// router-side limit, not a client fault, so the request routes unpinned.
    pub(super) async fn hold_session(
        &self,
        table: &SessionAffinity,
        session_id: &str,
        key: &RoutingPartitionId,
    ) -> Result<Option<Hold>, SelectionError> {
        loop {
            let acquired = tokio::select! {
                _ = self.cancel_token.cancelled() => {
                    return Err(SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown));
                }
                result = table.acquire(session_id, None) => result,
            };
            match acquired {
                Ok(Hold::Bound { target, mut lease })
                    if !self.catalog.is_schedulable(target.worker_id, key) =>
                {
                    tracing::debug!(
                        session_id,
                        worker_id = target.worker_id,
                        "Session affinity target is not schedulable; re-initializing"
                    );
                    lease.invalidate();
                }
                Ok(hold) => return Ok(Some(hold)),
                Err(AffinityError::ResourceExhausted(_)) => {
                    tracing::debug!(
                        session_id,
                        "Affinity table full; routing without session affinity"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(affinity_error(error)),
            }
        }
    }

    /// Bind the held session to `dispatched`. A `Hard` rejection whose bound
    /// worker departed after [`Self::hold_session`] checked it is not a client
    /// fault: the session is re-initialized on the dispatched worker instead.
    pub(super) fn commit_session(
        &self,
        table: &SessionAffinity,
        hold: Hold,
        session_id: &str,
        dispatched: WorkerWithDpRank,
        key: &RoutingPartitionId,
    ) -> Result<Option<AffinityLease>, SelectionError> {
        let bound = hold.target();
        let dispatched = WorkerAffinityTarget::new(dispatched.worker_id, Some(dispatched.dp_rank));
        match table.commit(hold, dispatched) {
            Ok(lease) => Ok(Some(lease)),
            Err(error) => {
                let departed =
                    bound.is_some_and(|target| !self.catalog.is_schedulable(target.worker_id, key));
                if !departed {
                    return Err(affinity_error(error));
                }
                // `commit` invalidated the stale binding. Another request may
                // already be initializing the replacement: join it rather
                // than wait, so this booking holds a lease on the new
                // binding. The join misses only if that initialization
                // resolved between the two calls; one retry covers that, and
                // a booking never waits on a request queued behind it.
                for _ in 0..JOIN_ATTEMPTS {
                    match table.try_acquire(session_id, None) {
                        Ok(AcquireStep::Held(hold)) => {
                            return table
                                .commit(hold, dispatched)
                                .map(Some)
                                .map_err(affinity_error);
                        }
                        Ok(AcquireStep::Wait(_)) => {
                            if let Some(lease) = table.join_initializing(session_id) {
                                return Ok(Some(lease));
                            }
                        }
                        Err(AffinityError::ResourceExhausted(_)) => return Ok(None),
                        Err(error) => return Err(affinity_error(error)),
                    }
                }
                Ok(None)
            }
        }
    }

    /// Record a booked routing decision into the partition's approximate
    /// indexer (side or primary). The booking already landed, so a failure
    /// here only costs predicted cache credit; it is logged, not returned.
    pub(super) async fn record_routing_decision(
        &self,
        entry: &SelectionEntry,
        worker: WorkerWithDpRank,
        block_hashes: Vec<LocalBlockHash>,
    ) {
        if block_hashes.is_empty() {
            return;
        }
        if let Err(error) = entry
            .indexer
            .record_routing_decision_hashes(
                worker,
                RoutingDecisionHashes::from_local_hashes(block_hashes),
            )
            .await
        {
            tracing::warn!(
                %error,
                key = %entry.key,
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                "Failed to record routing decision into approximate indexer"
            );
        }
    }

    /// Normalize the prompt and gather cache signals. The indexer lookup and
    /// the optional shared-cache lookup run concurrently; the shared cache is
    /// consulted only when `query_shared_cache` is set, a shared cache is
    /// attached, and the prompt carries raw `token_ids`.
    /// `tracking_assume_kv_reuse` is `None` when the caller does not track
    /// active blocks.
    pub(super) async fn prepare_selection_inputs(
        &self,
        entry: &SelectionEntry,
        prompt: &PromptView<'_>,
        tracking_assume_kv_reuse: Option<bool>,
        query_shared_cache: bool,
        retain_kv_transfer_chain: bool,
        lookup: &mut Option<LookupTimings>,
    ) -> Result<PreparedSelectionInputs, SelectionError> {
        let normalized = prompt.normalize_for_selection(
            entry.block_size,
            entry.is_eagle,
            tracking_assume_kv_reuse.map(|assume_kv_reuse| TrackingHashInput {
                context: &self.tracking_hash,
                scope: tracking_scope(entry),
                assume_kv_reuse,
            }),
        )?;
        let indexer_lookup = async {
            let started = Instant::now();
            let tiered = if normalized.block_hashes.is_empty() {
                Ok(TieredMatchDetails::default())
            } else {
                entry
                    .indexer
                    .find_tiered_matches_ref_with_options(
                        &normalized.block_hashes,
                        LowerTierQueryOptions {
                            retain_kv_transfer_chain,
                        },
                    )
                    .instrument(tracing::info_span!("kv_router.find_matches"))
                    .await
                    .map_err(SelectionError::Indexer)
            };
            (tiered, started.elapsed())
        };
        let shared_cache = query_shared_cache
            .then_some(self.host.cache.shared.as_deref())
            .flatten()
            .zip(prompt.token_ids);
        // `(hits, duration, failed)`; all `None`/`false` without a shared cache.
        let shared_cache_lookup = async {
            let Some((shared_cache, tokens)) = shared_cache else {
                return (None, None, false);
            };
            let started = Instant::now();
            let result = shared_cache
                .check_blocks(tokens, entry.block_size, prompt.cache_namespace)
                .instrument(tracing::info_span!("kv_router.shared_cache_check"))
                .await;
            let elapsed = started.elapsed();
            match result {
                Ok(hits) => (Some(hits), Some(elapsed), false),
                Err(error) => {
                    tracing::warn!(%error, "Shared cache query failed, ignoring");
                    (None, Some(elapsed), true)
                }
            }
        };
        let lookups_started = Instant::now();
        let ((tiered, indexer), (shared_cache_hits, shared_cache, shared_cache_error)) =
            tokio::join!(indexer_lookup, shared_cache_lookup);
        let timings = LookupTimings {
            block_hashing: normalized.block_hashing,
            seq_hashing: normalized.seq_hashing,
            lookups: lookups_started.elapsed(),
            indexer,
            shared_cache,
            shared_cache_error,
        };
        // Recorded before `?` so a shared-cache failure is still accounted
        // when the index lookup fails.
        *lookup = Some(timings);
        let tiered = tiered?;
        let overlap =
            OverlapAnalysis::new(&self.kv_router_config, entry.block_size, &tiered).signals();
        let kv_transfer_candidates = retain_kv_transfer_chain
            .then(|| tiered.kv_transfer_candidates().cloned())
            .flatten();
        drop(tiered);
        Ok(PreparedSelectionInputs {
            block_hashes: normalized.block_hashes,
            sequence_hashes: normalized.sequence_hashes,
            isl_tokens: normalized.isl_tokens,
            overlap,
            shared_cache_hits,
            kv_transfer_candidates,
        })
    }
}

/// The core's use of the session table for a request: a session is steered
/// only when the caller did not pin or target a worker explicitly.
fn session_binding(
    context: Option<&SessionContext>,
    is_steerable: bool,
    is_booking: bool,
) -> SessionBinding {
    match context.filter(|_| is_steerable) {
        Some(context) if is_booking => SessionBinding::Managed {
            session_id: context.session_id().to_string(),
        },
        Some(context) => SessionBinding::Query {
            session_id: context.session_id().to_string(),
        },
        None => SessionBinding::None,
    }
}

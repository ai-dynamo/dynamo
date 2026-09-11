// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reservations: the selection-id index over scheduler bookings, the
//! `create_reservation` replay path, and the lifecycle calls that advance or
//! release them.

use super::*;

/// Resolved inputs for booking a reservation, shared by the cached and explicit
/// `create_reservation` paths.
struct ReservationBooking {
    key: RoutingPartitionId,
    selection_id: String,
    worker: WorkerWithDpRank,
    sequence_hashes: Vec<SequenceHash>,
    prefill_load_hint: Option<PrefillLoadHint>,
    expected_output_tokens: Option<u32>,
    track_prefill_tokens: bool,
    lora_name: Option<String>,
    /// Public block hashes to record into an approximate indexer once booked.
    routing_hashes: Option<Vec<LocalBlockHash>>,
    session_id: Option<String>,
}

/// `selection_id` -> the booking it holds.
///
/// Lifecycle calls (`prefill_complete`, `free`, `add_output_block`) arrive with
/// only a selection id; the index resolves them to one partition and one
/// scheduler booking, so they never touch a booking made by a later request
/// that reused the id. An id is claimed here before its booking is made and
/// installed once the booking is final; a claim (`booking == None`) rejects a
/// concurrent booking of the same id and is invisible to lifecycle calls.
/// Bookings mirrored from replica peers are indexed by
/// [`ReservationIndexObserver`].
pub(super) type ReservationIndex = RwLock<HashMap<String, Reservation>>;

pub(super) struct Reservation {
    pub(super) partition: RoutingPartitionId,
    pub(super) booking: Option<SchedulerBookingDescriptor>,
    pub(super) _affinity_lease: Option<AffinityLease>,
}

/// Exclusive ownership of a selection id while its booking is in flight.
/// Dropping the claim without `install` releases the id.
pub(super) struct ReservationClaim<'a> {
    pub(super) index: &'a ReservationIndex,
    pub(super) selection_id: String,
    pub(super) armed: bool,
}

impl ReservationClaim<'_> {
    pub(super) fn install(mut self, reservation: Reservation) {
        self.armed = false;
        self.index
            .write()
            .insert(std::mem::take(&mut self.selection_id), reservation);
    }
}

impl Drop for ReservationClaim<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut index = self.index.write();
        if index
            .get(&self.selection_id)
            .is_some_and(|reservation| reservation.booking.is_none())
        {
            index.remove(&self.selection_id);
        }
    }
}

/// Keeps the reservation index exact for bookings replicated from peers, which
/// never pass through this core's booking paths, and forwards to the host's
/// own observer.
pub(super) struct ReservationIndexObserver {
    pub(super) index: Arc<ReservationIndex>,
    pub(super) partition: RoutingPartitionId,
    pub(super) host: Option<Arc<dyn ReplicaRequestLeaseObserver>>,
}

impl ReplicaRequestLeaseObserver for ReservationIndexObserver {
    fn admitted(&self, booking: SchedulerBookingDescriptor) {
        {
            let mut index = self.index.write();
            // A peer only admits an id this partition's scheduler does not hold,
            // so an existing row for it is a stale booking (expired, not yet
            // swept) or a claim whose local booking will now fail; the mirror
            // replaces both. A row from another partition is left alone.
            let own_row = index
                .get(&booking.request_id)
                .is_none_or(|reservation| reservation.partition == self.partition);
            if own_row {
                index.insert(
                    booking.request_id.clone(),
                    Reservation {
                        partition: self.partition.clone(),
                        booking: Some(booking.clone()),
                        _affinity_lease: None,
                    },
                );
            }
        }
        if let Some(host) = &self.host {
            host.admitted(booking);
        }
    }

    fn progressed(&self, booking: &SchedulerBookingDescriptor) {
        if let Some(host) = &self.host {
            host.progressed(booking);
        }
    }

    fn completed(&self, booking: &SchedulerBookingDescriptor) {
        forget_reservation_if(&self.index, &self.partition, booking);
        if let Some(host) = &self.host {
            host.completed(booking);
        }
    }
}

/// Remove the index entry for `booking` only if it still describes it.
fn forget_reservation_if(
    index: &ReservationIndex,
    partition: &RoutingPartitionId,
    booking: &SchedulerBookingDescriptor,
) {
    let mut index = index.write();
    let removed = index
        .get(&booking.request_id)
        .is_some_and(|reservation| {
            reservation.partition == *partition && reservation.booking.as_ref() == Some(booking)
        })
        .then(|| index.remove(&booking.request_id));
    drop(index);
    // The reservation's session binding releases on drop (shard lock, replica
    // publish); keep that work outside the index write lock.
    drop(removed);
}

impl SelectionCore {
    pub async fn create_reservation(
        &self,
        req: ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        self.ensure_running()?;

        let key = RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone());

        // Explicit form: book on the given worker under selection_id, discarding
        // any cached selection for the id so a later replay can't book stale state.
        if let Some(worker_id) = req.worker_id {
            self.selection_cache.discard(&key, &req.selection_id);
            return self.reserve_explicit(key, worker_id, req).await;
        }

        // Replay form: peek, book, and consume only once the booking lands. A
        // failure leaves the entry for a retry; concurrent replays of the same
        // id collide at the scheduler, so they can't double-book.
        let Some((pending, generation)) =
            self.selection_cache
                .peek(&key, &req.selection_id, Instant::now())
        else {
            return Err(SelectionError::NotFound(format!(
                "no pending selection {} for {key} (expired, already used, \
                 or never selected)",
                req.selection_id
            )));
        };
        let response = match self.book_cached_selection(pending, &req).await {
            Ok(response) => response,
            // The session moved: the cached worker can never be booked for it.
            Err(SelectionError::BadRequest(message)) => {
                self.selection_cache.discard(&key, &req.selection_id);
                return Err(SelectionError::BadRequest(message));
            }
            Err(error) => return Err(error),
        };
        self.selection_cache
            .remove(&key, &req.selection_id, generation);
        Ok(response)
    }

    /// Book a reservation replaying what the matching `select` captured; request
    /// fields other than the ids are ignored.
    async fn book_cached_selection(
        &self,
        pending: Arc<PendingSelection>,
        req: &ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        let (entry, endpoint, prefill_load_hint) = self.resolve_cached_booking(&pending)?;
        let track_prefill_tokens = pending.track_prefill_tokens;
        self.finalize_reservation(
            entry,
            endpoint,
            ReservationBooking {
                key: pending.key.clone(),
                selection_id: req.selection_id.clone(),
                worker: pending.worker,
                sequence_hashes: pending.sequence_hashes.clone(),
                prefill_load_hint: track_prefill_tokens.then_some(prefill_load_hint),
                expected_output_tokens: pending.expected_output_tokens,
                track_prefill_tokens,
                lora_name: pending.lora_name.clone(),
                routing_hashes: pending.routing_hashes.clone(),
                session_id: pending.session_id.clone(),
            },
        )
        .await
    }

    /// Resolve everything a cached booking needs (ready entry, schedulable
    /// endpoint, prefill hint), so the only fallible step left in
    /// `finalize_reservation` is the scheduler call.
    fn resolve_cached_booking(
        &self,
        pending: &PendingSelection,
    ) -> Result<(Arc<SelectionEntry>, String, PrefillLoadHint), SelectionError> {
        let entry = self.ready_entry(&pending.key)?;
        // Validate the full worker/rank against current topology; a rank a PATCH
        // removed during the window is rejected (the entry stays for a retry).
        let endpoint = self
            .catalog
            .schedulable_worker_endpoint(pending.worker, &pending.key)
            .ok_or_else(|| {
                SelectionError::NotFound(format!(
                    "schedulable worker {} (dp_rank {}) not found for {}",
                    pending.worker.worker_id, pending.worker.dp_rank, pending.key
                ))
            })?;
        let prefill_load_hint = prefill_load_hint_from_effective_tokens(
            pending.isl_tokens,
            pending.effective_prefill_tokens,
        )
        .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
        Ok((entry, endpoint, prefill_load_hint))
    }

    fn schedulable_endpoint(
        &self,
        worker_id: WorkerId,
        key: &RoutingPartitionId,
    ) -> Result<String, SelectionError> {
        self.catalog
            .schedulable_endpoint(worker_id, key)
            .ok_or_else(|| {
                SelectionError::NotFound(format!(
                    "schedulable worker {worker_id} not found for {key}"
                ))
            })
    }

    /// Book a reservation from a self-contained request (explicit worker_id and prompt).
    async fn reserve_explicit(
        &self,
        key: RoutingPartitionId,
        worker_id: WorkerId,
        req: ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        let entry = self.ready_entry(&key)?;
        let normalized = req.prompt.view().normalize_for_reservation(
            entry.is_eagle,
            TrackingHashInput {
                context: &self.tracking_hash,
                scope: tracking_scope(&entry),
                assume_kv_reuse: self
                    .kv_router_config
                    .assume_kv_reuse(req.router_config_override.as_ref()),
            },
        )?;
        let prefill_load_hint = req
            .effective_prefill_tokens
            .map(|tokens| {
                prefill_load_hint_from_effective_tokens(normalized.isl_tokens, tokens)
                    .map_err(|error| SelectionError::BadRequest(error.to_string()))
            })
            .transpose()?;
        let worker = WorkerWithDpRank::new(worker_id, req.dp_rank.unwrap_or(0));
        let endpoint = self.schedulable_endpoint(worker.worker_id, &key)?;
        let track_prefill_tokens = req.track_prefill_tokens.unwrap_or_else(|| {
            req.effective_prefill_tokens.is_some()
                || req
                    .router_config_override
                    .as_ref()
                    .and_then(|cfg| cfg.track_prefill_tokens)
                    .unwrap_or(self.kv_router_config.router_track_prefill_tokens)
        });
        // Hash-only reservations (sequence hashes without block hashes) carry
        // nothing an indexer can key on; recording is skipped for them.
        let can_record = entry.indexer.records_routing_decisions()
            && (req.prompt.token_ids.is_some() || req.prompt.block_hashes.is_some());
        let routing_hashes = can_record
            .then(|| {
                req.prompt
                    .view()
                    .block_hashes_for_indexer(entry.block_size, entry.is_eagle)
            })
            .transpose()?;

        self.finalize_reservation(
            entry,
            endpoint,
            ReservationBooking {
                key,
                selection_id: req.selection_id,
                worker,
                sequence_hashes: normalized.sequence_hashes,
                prefill_load_hint,
                expected_output_tokens: req.expected_output_tokens,
                track_prefill_tokens,
                lora_name: req.prompt.lora_name,
                routing_hashes,
                session_id: None,
            },
        )
        .await
    }

    /// Register the booking with the scheduler. All fallible resolution happens
    /// in the caller; the scheduler add here is the last step that can fail, and
    /// the cached path leaves its selection in place (to retry) if it does.
    async fn finalize_reservation(
        &self,
        entry: Arc<SelectionEntry>,
        endpoint: String,
        booking: ReservationBooking,
    ) -> Result<ReservationResponse, SelectionError> {
        let ReservationBooking {
            key,
            selection_id,
            worker,
            sequence_hashes,
            prefill_load_hint,
            expected_output_tokens,
            track_prefill_tokens,
            lora_name,
            routing_hashes,
            session_id,
        } = booking;

        let claim = self.claim_reservation(&selection_id, &key)?;
        // Hold the session before booking: holding after would wait on an
        // initializer that may itself be waiting for this worker's capacity.
        let session = match (session_id.as_deref(), entry.affinity.get()) {
            (Some(session_id), Some(table)) => self
                .hold_session(table, session_id, &key)
                .await?
                .map(|hold| (table, session_id, hold)),
            _ => None,
        };
        // Strict booking: never lazily recreate a worker/rank removed since the
        // reservation was resolved. The handle frees the booking if this future
        // is dropped before `install`.
        let booking = entry
            .scheduler
            .add_request_if_registered_guarded(SequenceRequest {
                request_id: selection_id.clone(),
                token_sequence: Some(sequence_hashes),
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
                worker,
                lora_name,
            })?;
        let affinity_lease = match session {
            Some((table, session_id, hold)) => {
                self.commit_session(table, hold, session_id, worker, &key)?
            }
            None => None,
        };
        if let Some(hashes) = routing_hashes {
            self.record_routing_decision(&entry, worker, hashes).await;
        }
        claim.install(Reservation {
            partition: key.clone(),
            booking: Some(booking.commit()),
            _affinity_lease: affinity_lease,
        });

        Ok(ReservationResponse {
            selection_id,
            model_name: key.model_name,
            routing_group: key.routing_group,
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            endpoint,
        })
    }

    /// Take `selection_id` for a booking about to be made in `key`.
    pub(super) fn claim_reservation(
        &self,
        selection_id: &str,
        key: &RoutingPartitionId,
    ) -> Result<ReservationClaim<'_>, SelectionError> {
        let mut index = self.reservation_index.write();
        if index.contains_key(selection_id) {
            return Err(SelectionError::Conflict(format!(
                "selection {selection_id} is already reserved or being reserved"
            )));
        }
        index.insert(
            selection_id.to_string(),
            Reservation {
                partition: key.clone(),
                booking: None,
                _affinity_lease: None,
            },
        );
        Ok(ReservationClaim {
            index: &self.reservation_index,
            selection_id: selection_id.to_string(),
            armed: true,
        })
    }

    /// The partition and booking a lifecycle call on `selection_id` may touch.
    /// `None` for unknown ids and for ids whose booking is still in flight.
    pub(super) fn indexed_booking(
        &self,
        selection_id: &str,
    ) -> Option<(Arc<SelectionEntry>, SchedulerBookingDescriptor)> {
        // Release the index guard before taking `entries`: the sweep takes
        // `entries` then `reservation_index`, so nesting here would invert the
        // lock order.
        let (partition, booking) = {
            let index = self.reservation_index.read();
            let reservation = index.get(selection_id)?;
            (reservation.partition.clone(), reservation.booking.clone()?)
        };
        Some((self.entry(&partition)?, booking))
    }

    fn reservation_not_found(selection_id: &str) -> SelectionError {
        SelectionError::NotFound(format!("reservation {selection_id} not found"))
    }

    pub async fn prefill_complete(&self, selection_id: &str) -> Result<(), SelectionError> {
        let Some((entry, booking)) = self.indexed_booking(selection_id) else {
            return Err(Self::reservation_not_found(selection_id));
        };
        match entry
            .scheduler
            .mark_prefill_completed_if_booking(&booking)
            .await?
        {
            LifecycleMutationOutcome::Applied => Ok(()),
            // Already marked: still publish the ordered completion so peers
            // that missed the first event converge.
            LifecycleMutationOutcome::NoChange
                if entry
                    .scheduler
                    .publish_prefill_completed_if_booking(&booking) =>
            {
                Ok(())
            }
            LifecycleMutationOutcome::NoChange => {
                forget_reservation_if(&self.reservation_index, &entry.key, &booking);
                Err(Self::reservation_not_found(selection_id))
            }
        }
    }

    pub async fn free_reservation(&self, selection_id: &str) -> Result<(), SelectionError> {
        let Some((entry, booking)) = self.indexed_booking(selection_id) else {
            return Err(Self::reservation_not_found(selection_id));
        };
        let outcome = entry.scheduler.free_if_booking(&booking).await?;
        forget_reservation_if(&self.reservation_index, &entry.key, &booking);
        match outcome {
            LifecycleMutationOutcome::Applied => Ok(()),
            LifecycleMutationOutcome::NoChange => Err(Self::reservation_not_found(selection_id)),
        }
    }

    pub fn add_output_block(
        &self,
        selection_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SelectionError> {
        if let Some(frac) = decay_fraction
            && !(0.0..=1.0).contains(&frac)
        {
            return Err(SelectionError::BadRequest(
                "decay_fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        let Some((entry, booking)) = self.indexed_booking(selection_id) else {
            return Err(Self::reservation_not_found(selection_id));
        };
        match entry
            .scheduler
            .add_output_block_if_booking_sync(&booking, decay_fraction)?
        {
            LifecycleMutationOutcome::Applied => Ok(()),
            LifecycleMutationOutcome::NoChange => {
                forget_reservation_if(&self.reservation_index, &entry.key, &booking);
                Err(Self::reservation_not_found(selection_id))
            }
        }
    }
}

/// Drop index entries whose booking no longer exists in its partition
/// scheduler (expired by the periodic force-expiry, or freed through a path
/// that bypassed this core). Claims are left for their owner. Returns the
/// number of entries removed.
pub(super) fn sweep_reservation_index(
    entries: &SelectionEntries,
    index: &ReservationIndex,
) -> usize {
    let entries = entries.read();
    let mut index = index.write();
    let before = index.len();
    let mut removed = Vec::new();
    index.retain(|_, reservation| {
        let Some(booking) = &reservation.booking else {
            return true;
        };
        let live = entries
            .get(&reservation.partition)
            .and_then(|cell| cell.get())
            .is_some_and(|entry| entry.scheduler.has_booking(booking));
        if !live {
            removed.push(reservation._affinity_lease.take());
        }
        live
    });
    let swept = before - index.len();
    drop(index);
    drop(entries);
    // Session bindings release on drop (shard lock, replica publish); do that
    // after both locks are gone.
    drop(removed);
    swept
}

pub(super) fn spawn_reservation_index_sweep(
    entries: Arc<SelectionEntries>,
    index: Arc<ReservationIndex>,
    cancel_token: CancellationToken,
) {
    let period = crate::sequences::active_request_expiry_duration();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => break,
                _ = interval.tick() => {
                    let removed = sweep_reservation_index(&entries, &index);
                    if removed > 0 {
                        tracing::debug!(removed, "Swept stale selection reservation index entries");
                    }
                }
            }
        }
    });
}

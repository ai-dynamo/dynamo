// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reservation resolution, renewal, release, and drop behavior.

use super::*;

impl PendingValkeyReservation {
    /// Execute exactly one logical reservation.  On a lost response the
    /// module sees the same nonce on retry; on cancellation/drop the pending
    /// guard replays this request once more to find and release any commit.
    pub(crate) async fn resolve(mut self) -> Result<Option<ValkeyReservationLease>> {
        let grant = self.indexer.select_reservation(&self.request).await?;
        let Some(grant) = grant else {
            self.armed = false;
            return Ok(None);
        };

        self.armed = false;
        let lease = ValkeyReservationLease {
            inner: Arc::new(ReservationLeaseInner {
                indexer: self.indexer.clone(),
                request: self.request.clone(),
                grant,
                state: Mutex::new(ReservationLeaseState {
                    expires_at_ms: grant.expires_at_ms,
                    released: false,
                }),
                lifecycle: Mutex::new(()),
                renewal_cancel: CancellationToken::new(),
                renewal_started: AtomicBool::new(false),
                release_started: AtomicBool::new(false),
            }),
        };
        // Cover the interval between an acknowledged module mutation and
        // RequestGuard handoff. Calling this again from RequestGuard is an
        // idempotent no-op.
        lease.start_renewal();
        Ok(Some(lease))
    }
}

impl Drop for PendingValkeyReservation {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                "No Tokio runtime to reconcile a pending Valkey reservation; its lease will expire"
            );
            return;
        };
        let indexer = self.indexer.clone();
        let request = self.request.clone();
        let Ok(permit) = Arc::clone(&indexer.inner.admission_cleanup_permits).try_acquire_owned()
        else {
            tracing::debug!(
                maximum = MAX_PENDING_ADMISSION_CLEANUPS,
                "Skipping pending Valkey reservation reconciliation because the cleanup task limit is full; lease expiry remains the backstop"
            );
            return;
        };
        handle.spawn(async move {
            let _permit = permit;
            let cleanup_window = Duration::from_millis(request.lease_ms);
            match timeout(cleanup_window, indexer.select_reservation(&request)).await {
                Err(_) => {
                    tracing::debug!(
                        lease_ms = request.lease_ms,
                        "Pending Valkey reservation reconciliation reached its lease-sized deadline"
                    );
                }
                Ok(Ok(Some(grant))) => {
                    match timeout(
                        cleanup_window,
                        indexer.release_reservation(&request, grant.expires_at_ms),
                    )
                    .await
                    {
                        Ok(Ok(_)) => {}
                        Ok(Err(error)) => {
                            tracing::debug!(
                                error = %error,
                                worker_id = grant.worker.worker_id,
                                dp_rank = grant.worker.dp_rank,
                                "Failed to release reconciled Valkey reservation; lease expiry remains the backstop"
                            );
                        }
                        Err(_) => {
                            tracing::debug!(
                                worker_id = grant.worker.worker_id,
                                dp_rank = grant.worker.dp_rank,
                                lease_ms = request.lease_ms,
                                "Reconciled Valkey reservation release reached its lease-sized deadline"
                            );
                        }
                    }
                }
                Ok(Ok(None)) => {}
                Ok(Err(error)) => {
                    tracing::debug!(
                        error = %error,
                        "Failed to reconcile pending Valkey reservation; lease expiry remains the backstop"
                    );
                }
            }
        });
    }
}

impl ValkeyReservationLease {
    pub(crate) fn worker(&self) -> WorkerWithDpRank {
        self.inner.grant.worker
    }

    pub(crate) fn matched_blocks(&self) -> u32 {
        self.inner.grant.matched_blocks
    }

    pub(crate) fn active_reservations_at_grant(&self) -> u32 {
        self.inner.grant.active_reservations_at_grant
    }

    /// Start a low-rate renewal task after the lease has entered request
    /// lifetime ownership.  Selection itself stays short and cancellation-safe.
    pub(crate) fn start_renewal(&self) {
        if self
            .inner
            .renewal_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                "No Tokio runtime to renew Valkey reservation; lease expiry is the only failure backstop"
            );
            return;
        };
        let inner = Arc::clone(&self.inner);
        let (initial_delay, renewal_interval) =
            renewal_schedule(inner.request.lease_ms, inner.request.nonce);
        handle.spawn(async move {
            renew_reservation_loop(inner, initial_delay, renewal_interval).await
        });
    }

    pub(crate) async fn release(&mut self) -> Result<()> {
        release_reservation_lease(Arc::clone(&self.inner)).await
    }

    /// Release without making request completion wait for a degraded Valkey
    /// primary or replica. The bounded task retains the lease state machine
    /// for at most one lease window; module expiry remains the backstop.
    pub(crate) fn release_detached(mut self) {
        // This method transfers request-lifetime ownership immediately, but
        // the spawned release task may not be polled for a while under a
        // completion burst. Stop the renewal loop before scheduling that task
        // so completed requests cannot enqueue a late RENEW ahead of their
        // RELEASE. `release_reservation_lease` repeats this cancellation and
        // still owns the release_started CAS and the actual remote mutation.
        self.inner.renewal_cancel.cancel();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            self.inner.release_started.store(true, Ordering::Release);
            tracing::warn!(
                worker_id = self.inner.grant.worker.worker_id,
                dp_rank = self.inner.grant.worker.dp_rank,
                "No Tokio runtime available for detached Valkey reservation release; lease expiry remains the backstop"
            );
            return;
        };
        let Ok(permit) =
            Arc::clone(&self.inner.indexer.inner.admission_cleanup_permits).try_acquire_owned()
        else {
            self.inner.release_started.store(true, Ordering::Release);
            tracing::debug!(
                maximum = MAX_PENDING_ADMISSION_CLEANUPS,
                worker_id = self.inner.grant.worker.worker_id,
                dp_rank = self.inner.grant.worker.dp_rank,
                "Skipping detached Valkey reservation release because the cleanup task limit is full; lease expiry remains the backstop"
            );
            return;
        };
        handle.spawn(async move {
            let _permit = permit;
            if let Err(error) = self.release().await {
                tracing::debug!(
                    error = %error,
                    "Failed to release detached Valkey reservation; lease expiry remains the backstop"
                );
            }
        });
    }
}

impl Drop for ValkeyReservationLease {
    fn drop(&mut self) {
        self.inner.renewal_cancel.cancel();
        if self.inner.release_started.load(Ordering::Acquire) {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                worker_id = self.inner.grant.worker.worker_id,
                dp_rank = self.inner.grant.worker.dp_rank,
                "No Tokio runtime to release Valkey reservation; lease expiry remains the backstop"
            );
            return;
        };
        let Ok(permit) =
            Arc::clone(&self.inner.indexer.inner.admission_cleanup_permits).try_acquire_owned()
        else {
            self.inner.release_started.store(true, Ordering::Release);
            tracing::debug!(
                maximum = MAX_PENDING_ADMISSION_CLEANUPS,
                worker_id = self.inner.grant.worker.worker_id,
                dp_rank = self.inner.grant.worker.dp_rank,
                "Skipping dropped Valkey reservation release because the cleanup task limit is full; lease expiry remains the backstop"
            );
            return;
        };
        let inner = Arc::clone(&self.inner);
        handle.spawn(async move {
            let _permit = permit;
            if let Err(error) = release_reservation_lease(inner).await {
                tracing::debug!(
                    error = %error,
                    "Failed to release Valkey reservation from request guard drop; lease expiry remains the backstop"
                );
            }
        });
    }
}

pub(super) async fn release_reservation_lease(inner: Arc<ReservationLeaseInner>) -> Result<()> {
    inner.renewal_cancel.cancel();
    if inner
        .release_started
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(());
    }
    let cleanup_window = Duration::from_millis(inner.request.lease_ms);
    timeout(cleanup_window, async {
        let _lifecycle = inner.lifecycle.lock().await;
        let expected_expires_at_ms = {
            let state = inner.state.lock().await;
            if state.released {
                return Ok(());
            }
            state.expires_at_ms
        };
        let released = inner
            .indexer
            .release_reservation(&inner.request, expected_expires_at_ms)
            .await?;
        inner.state.lock().await.released = true;
        if !released {
            tracing::debug!(
                worker_id = inner.grant.worker.worker_id,
                dp_rank = inner.grant.worker.dp_rank,
                "Valkey reservation was already expired before release"
            );
        }
        Ok(())
    })
    .await
    .with_context(|| {
        format!(
            "Valkey reservation release exceeded its {} ms lease-sized cleanup deadline",
            inner.request.lease_ms
        )
    })?
}

pub(super) async fn renew_reservation_loop(
    inner: Arc<ReservationLeaseInner>,
    initial_delay: Duration,
    renewal_interval: Duration,
) {
    // Keep the nonce-derived phase after the first renewal.  Recomputing a
    // common delay after every reply would gradually synchronize an admission
    // burst again. This schedule changes only when we try RENEW; the module
    // still owns the deadline token and continues to reject lease shortening.
    let mut next_delay = initial_delay;
    loop {
        tokio::select! {
            _ = inner.renewal_cancel.cancelled() => return,
            _ = tokio::time::sleep(next_delay) => {}
        }
        next_delay = renewal_interval;

        let renewal_window = Duration::from_millis(inner.request.lease_ms);
        let renewal = timeout(renewal_window, async {
            let _lifecycle = inner.lifecycle.lock().await;
            // `tokio::select!` deliberately randomizes when both branches are
            // ready. A detached release can therefore cancel this task at the
            // exact instant its sleep fires. Recheck after serializing with a
            // possible release so no late RENEW starts after completion.
            let expected_expires_at_ms = {
                let state = inner.state.lock().await;
                if renewal_should_stop(&inner, &state) {
                    return Ok(false);
                }
                state.expires_at_ms
            };
            match inner
                .indexer
                .renew_reservation(&inner.request, expected_expires_at_ms)
                .await
            {
                Ok(Some(grant)) if grant.worker == inner.grant.worker => {
                    inner.state.lock().await.expires_at_ms = grant.expires_at_ms;
                    Ok(true)
                }
                Ok(Some(grant)) => {
                    tracing::error!(
                        expected_worker_id = inner.grant.worker.worker_id,
                        expected_dp_rank = inner.grant.worker.dp_rank,
                        received_worker_id = grant.worker.worker_id,
                        received_dp_rank = grant.worker.dp_rank,
                        "Valkey RENEW returned a different worker; stopping renewal"
                    );
                    Ok(false)
                }
                Ok(None) => {
                    tracing::warn!(
                        worker_id = inner.grant.worker.worker_id,
                        dp_rank = inner.grant.worker.dp_rank,
                        "Valkey reservation expired before it could be renewed"
                    );
                    Ok(false)
                }
                Err(error) => Err(error),
            }
        })
        .await;
        match renewal {
            Ok(Ok(true)) => {}
            Ok(Ok(false)) => return,
            Ok(Err(error)) => {
                tracing::warn!(
                    error = %error,
                    worker_id = inner.grant.worker.worker_id,
                    dp_rank = inner.grant.worker.dp_rank,
                    "Failed to renew Valkey reservation; stopping renewal"
                );
                return;
            }
            Err(_) => {
                tracing::warn!(
                    worker_id = inner.grant.worker.worker_id,
                    dp_rank = inner.grant.worker.dp_rank,
                    lease_ms = inner.request.lease_ms,
                    "Valkey reservation renewal reached its lease-sized deadline; stopping renewal"
                );
                return;
            }
        }
    }
}

pub(super) fn renewal_should_stop(
    inner: &ReservationLeaseInner,
    state: &ReservationLeaseState,
) -> bool {
    state.released
        || inner.renewal_cancel.is_cancelled()
        || inner.release_started.load(Ordering::Acquire)
}

/// Produce a stable per-reservation renewal phase without using a process
/// randomized hasher. The first attempt is in the first quarter of the lease,
/// leaving at least three quarters of the original deadline for a queued
/// lifecycle mutation or a retry. Subsequent attempts retain the existing
/// one-third-of-lease cadence.
pub(super) fn renewal_schedule(lease_ms: u64, nonce: ReservationNonce) -> (Duration, Duration) {
    let renewal_interval_ms = (lease_ms / RENEWAL_INTERVAL_DIVISOR).max(1);
    let earliest_ms = (lease_ms / RENEWAL_INITIAL_EARLIEST_DIVISOR).max(1);
    let latest_ms = (lease_ms / RENEWAL_INITIAL_LATEST_DIVISOR).max(earliest_ms);
    let jitter_span_ms = latest_ms - earliest_ms;
    let initial_delay_ms = earliest_ms + stable_reservation_hash(nonce) % (jitter_span_ms + 1);

    (
        Duration::from_millis(initial_delay_ms),
        Duration::from_millis(renewal_interval_ms),
    )
}

/// SplitMix64 finalizer. It is deterministic for a reservation nonce and has
/// enough diffusion to spread UUID-derived request identities across a small
/// renewal window.
pub(super) fn stable_reservation_hash(nonce: ReservationNonce) -> u64 {
    let mut value = nonce.client_nonce ^ nonce.request_nonce.rotate_left(29);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

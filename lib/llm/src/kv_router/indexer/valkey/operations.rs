// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mutation, lifecycle, and replication operations for the Valkey indexer.

use super::*;

impl ValkeyIndexer {
    pub(crate) async fn apply_event(&self, event: &RouterEvent) -> Result<()> {
        if matches!(&event.event.data, KvCacheEventData::Stored(store) if store.blocks.is_empty()) {
            return Ok(());
        }
        // A match is an affinity hint: a worker that has evicted the matched
        // blocks still handles the request correctly. Avoid invalidating the
        // hot cache on normal KV churn; the short absolute TTL bounds stale
        // ownership, and lifecycle operations invalidate eagerly below.
        self.apply_to_primary(&encode_event(event)).await
    }

    #[cfg(test)]
    pub(crate) async fn apply_event_owned(
        &self,
        event: &RouterEvent,
        owner_nonce: u64,
    ) -> Result<()> {
        self.apply_events_owned(std::slice::from_ref(event), owner_nonce)
            .await
    }

    pub(crate) async fn apply_events_owned(
        &self,
        events: &[RouterEvent],
        owner_nonce: u64,
    ) -> Result<()> {
        if owner_nonce == 0 {
            bail!("Valkey worker owner nonce must be nonzero");
        }
        let payloads = events
            .iter()
            .filter(|event| {
                !matches!(&event.event.data, KvCacheEventData::Stored(store) if store.blocks.is_empty())
            })
            .map(encode_event)
            .collect::<Vec<_>>();
        if payloads.is_empty() {
            return Ok(());
        }
        let owner = owner_nonce.to_be_bytes();
        let arguments = payloads
            .iter()
            .map(|payload| {
                [
                    b"DYNKV.APPLY_OWNED".as_slice(),
                    self.inner.index_key.as_slice(),
                    owner.as_slice(),
                    payload.as_slice(),
                ]
            })
            .collect::<Vec<_>>();
        let commands = arguments
            .iter()
            .map(|arguments| arguments.as_slice())
            .collect::<Vec<_>>();
        self.command_direct_event_batch_and_replicate(&commands)
            .await
    }

    pub(crate) async fn register_worker_lease(
        &self,
        worker_id: WorkerId,
        owner_nonce: u64,
        lease_ms: u64,
        gc_inspection_budget: u32,
        dp_ranks: &[DpRank],
    ) -> Result<()> {
        let worker = worker_id.to_be_bytes();
        validate_worker_ranks(dp_ranks)?;
        if gc_inspection_budget == 0 || gc_inspection_budget > MAX_GC_INSPECTION_BUDGET {
            bail!(
                "Valkey registration cleanup budget must be in 1..={MAX_GC_INSPECTION_BUDGET}; got {gc_inspection_budget}"
            );
        }
        let mut delay = REGISTRATION_CAS_RETRY_INITIAL_DELAY;
        let mut attempts = 0_u32;
        loop {
            if self.inner.cancellation_token.is_cancelled() {
                bail!("Valkey worker registration cancelled during shutdown");
            }
            let expected_generation = self.registration_generation(worker_id).await?;
            let payload = encode_worker_lease_registration(
                owner_nonce,
                lease_ms,
                expected_generation,
                dp_ranks,
            )?;
            let result = self
                .command_write_and_replicate(&[
                    b"DYNKV.REGISTER_WORKER_RANKS",
                    &self.inner.index_key,
                    &worker,
                    &payload,
                ])
                .await;
            match result {
                Ok(()) => return Ok(()),
                Err(error) if stale_registration_generation_error(&error) => {
                    if attempts == 0 || attempts.is_multiple_of(10) {
                        tracing::debug!(
                            worker_id,
                            owner_nonce,
                            expected_generation,
                            attempts,
                            retry_delay_ms = delay.as_millis(),
                            "Valkey worker registration generation raced another lifecycle transition; refreshing the CAS token"
                        );
                    }
                    attempts = attempts.saturating_add(1);
                    tokio::select! {
                        _ = self.inner.cancellation_token.cancelled() => {
                            bail!("Valkey worker registration cancelled during shutdown");
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                    delay = (delay * 2).min(REGISTRATION_CAS_RETRY_MAX_DELAY);
                }
                Err(error) if worker_cleanup_pending_error(&error) => {
                    if attempts == 0 || attempts.is_multiple_of(10) {
                        tracing::info!(
                            worker_id,
                            owner_nonce,
                            attempts,
                            gc_inspection_budget,
                            retry_delay_ms = delay.as_millis(),
                            "Valkey worker replacement is waiting for bounded cleanup; advancing one GC step"
                        );
                    }
                    let gc_result = tokio::select! {
                        _ = self.inner.cancellation_token.cancelled() => {
                            bail!("Valkey worker registration cancelled during cleanup");
                        }
                        result = self.gc_step(gc_inspection_budget) => result,
                    };
                    match gc_result {
                        Ok(stats) => {
                            tracing::debug!(
                                worker_id,
                                owner_nonce,
                                attempts,
                                gc_stats = ?stats,
                                "Advanced bounded Valkey cleanup for worker replacement"
                            );
                            // A scan advances the module's cooperative cursor even
                            // when this particular chunk has no reclaim transition.
                            // Keep productive cleanup responsive; back off only when
                            // the module had no record to inspect.
                            if stats[0] != 0 {
                                delay = REGISTRATION_CAS_RETRY_INITIAL_DELAY;
                            }
                        }
                        Err(gc_error)
                            if gc_error
                                .downcast_ref::<RespError>()
                                .is_some_and(retryable_write_error) =>
                        {
                            tracing::warn!(
                                worker_id,
                                owner_nonce,
                                attempts,
                                error = %gc_error,
                                "Transient Valkey cleanup-assist failure; retrying worker registration"
                            );
                        }
                        Err(gc_error) => {
                            return Err(gc_error).context(format!(
                                "failed to advance pending Valkey cleanup before registering worker {worker_id}"
                            ));
                        }
                    }
                    attempts = attempts.saturating_add(1);
                    tokio::select! {
                        _ = self.inner.cancellation_token.cancelled() => {
                            bail!("Valkey worker registration cancelled during cleanup");
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                    delay = (delay * 2).min(REGISTRATION_CAS_RETRY_MAX_DELAY);
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn registration_generation(&self, worker_id: WorkerId) -> Result<u64> {
        let worker = worker_id.to_be_bytes();
        let response = self
            .command_primary_read_with_retry_timeout(
                &[
                    b"DYNKV.REGISTRATION_GENERATION",
                    &self.inner.index_key,
                    &worker,
                ],
                REGISTRATION_READ_FAILOVER_TIMEOUT,
            )
            .await?;
        decode_u64_bulk(response, "DYNKV.REGISTRATION_GENERATION")
    }

    /// Run one bounded module lifecycle-GC step. This command deliberately
    /// does not retry an ambiguous mutation: replaying a scan is semantically
    /// safe, but it could reclaim another `inspection_budget` records and
    /// violate the caller's per-tick work bound. A later periodic tick resumes
    /// from the module's runtime cursor.
    pub(crate) async fn gc_step(&self, inspection_budget: u32) -> Result<[u64; 8]> {
        if inspection_budget == 0 || inspection_budget > MAX_GC_INSPECTION_BUDGET {
            bail!(
                "Valkey lifecycle GC inspection budget must be in 1..={MAX_GC_INSPECTION_BUDGET}; got {inspection_budget}"
            );
        }
        let budget = inspection_budget.to_be_bytes();
        let arguments = [
            b"DYNKV.GC".as_slice(),
            self.inner.index_key.as_slice(),
            b"CURRENT".as_slice(),
            budget.as_slice(),
        ];
        let response = if self.inner.replication_quorum == 0 {
            self.inner.primary.command_write(&arguments).await
        } else {
            self.inner
                .primary
                .command_write_and_wait(&arguments, self.inner.replication_quorum, false)
                .await
        }?;
        decode_gc_reply(response)
    }

    pub(crate) async fn renew_worker_lease(
        &self,
        worker_id: WorkerId,
        owner_nonce: u64,
        lease_ms: u64,
    ) -> Result<()> {
        let payload = encode_worker_lease_control(worker_id, owner_nonce, Some(lease_ms))?;
        self.command_write_and_replicate(&[
            b"DYNKV.RENEW_WORKER_LEASE",
            &self.inner.index_key,
            &payload,
        ])
        .await
    }

    pub(crate) async fn unregister_worker_lease(
        &self,
        worker_id: WorkerId,
        owner_nonce: u64,
    ) -> Result<()> {
        let payload = encode_worker_lease_control(worker_id, owner_nonce, None)?;
        self.command_write_and_replicate(&[
            b"DYNKV.UNREGISTER_WORKER",
            &self.inner.index_key,
            &payload,
        ])
        .await
    }

    /// Atomically replace a recovered rank only if no mutation raced the
    /// captured tree dump. A stale generation is not an error: the recovery
    /// coordinator must fetch a fresh dump and try again.
    pub(crate) async fn replace_rank_snapshot_if_current(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: &[RouterEvent],
    ) -> Result<bool> {
        let worker = worker_id.to_be_bytes();
        let rank = dp_rank.to_be_bytes();
        let expected_generation = self.rank_generation(worker_id, dp_rank).await?;
        let generation = expected_generation.to_be_bytes();
        let snapshot = encode_rank_snapshot(worker_id, dp_rank, events)?;
        let arguments = [
            b"DYNKV.REPLACE_RANK_IF_GENERATION".as_slice(),
            self.inner.index_key.as_slice(),
            worker.as_slice(),
            rank.as_slice(),
            generation.as_slice(),
            snapshot.as_slice(),
        ];

        let mut delay = WRITE_RETRY_INITIAL_DELAY;
        let mut attempts = 0_u32;
        loop {
            let result = if self.inner.replication_quorum == 0 {
                self.inner.primary.command_write(&arguments).await
            } else {
                self.inner
                    .primary
                    .command_write_and_wait(&arguments, self.inner.replication_quorum, attempts > 0)
                    .await
            };
            match result {
                Ok(response) => {
                    let _new_generation =
                        decode_u64_bulk(response, "DYNKV.REPLACE_RANK_IF_GENERATION")?;
                    self.inner.match_cache.invalidate_all();
                    return Ok(true);
                }
                Err(error) if stale_rank_generation_error(&error) => {
                    self.inner.match_cache.invalidate_all();
                    tracing::debug!(
                        worker_id,
                        dp_rank,
                        expected_generation,
                        attempts,
                        "Discarding a stale Valkey rank snapshot after a concurrent mutation"
                    );
                    return Ok(false);
                }
                Err(error) if retryable_write_error(&error) => {
                    if attempts == 0 || attempts.is_multiple_of(10) {
                        tracing::warn!(
                            worker_id,
                            dp_rank,
                            expected_generation,
                            attempts,
                            error = %error,
                            retry_delay_ms = delay.as_millis(),
                            "Valkey conditional rank replacement was not acknowledged; retrying the fenced command"
                        );
                    }
                    attempts = attempts.saturating_add(1);
                    tokio::select! {
                        _ = self.inner.cancellation_token.cancelled() => {
                            bail!("Valkey conditional rank replacement cancelled during shutdown");
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                    delay = (delay * 2).min(self.inner.primary.write_retry_max_delay());
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    async fn rank_generation(&self, worker_id: WorkerId, dp_rank: DpRank) -> Result<u64> {
        let worker = worker_id.to_be_bytes();
        let rank = dp_rank.to_be_bytes();
        let response = self
            .command_primary_read_with_retry(&[
                b"DYNKV.RANK_GENERATION",
                &self.inner.index_key,
                &worker,
                &rank,
            ])
            .await?;
        decode_u64_bulk(response, "DYNKV.RANK_GENERATION")
    }

    /// Reset one rank before replaying a worker tree dump. This is distinct
    /// from worker retirement: a reset accepts the dump's lower synthetic
    /// event IDs, while retirement rejects delayed events from a removed rank.
    pub(crate) async fn reset_worker(&self, worker_id: WorkerId, dp_rank: DpRank) -> Result<()> {
        self.inner.match_cache.invalidate_all();
        let worker = worker_id.to_be_bytes();
        let dp_rank = dp_rank.to_be_bytes();
        self.command_write_and_replicate(&[
            b"DYNKV.RESET_WORKER",
            &self.inner.index_key,
            &worker,
            &dp_rank,
        ])
        .await
    }

    pub(crate) async fn remove_worker_all(&self, worker_id: WorkerId) -> Result<()> {
        self.inner.match_cache.invalidate_all();
        let worker = worker_id.to_be_bytes();
        self.command_write_and_replicate(&[
            b"DYNKV.REMOVE_WORKER_ALL",
            &self.inner.index_key,
            &worker,
        ])
        .await
    }

    async fn apply_to_primary(&self, payload: &[u8]) -> Result<()> {
        self.command_write_and_replicate(&[b"DYNKV.APPLY", &self.inner.index_key, payload])
            .await
    }

    async fn command_write_and_replicate(&self, arguments: &[&[u8]]) -> Result<()> {
        let mut delay = WRITE_RETRY_INITIAL_DELAY;
        let mut attempts = 0_u32;
        loop {
            let result = if self.inner.replication_quorum == 0 {
                ensure_ok(self.inner.primary.command_write(arguments).await)
            } else {
                ensure_ok(
                    self.inner
                        .primary
                        .command_write_and_wait(
                            arguments,
                            self.inner.replication_quorum,
                            attempts > 0,
                        )
                        .await,
                )
            };
            match result {
                Ok(()) => return Ok(()),
                Err(error) if retryable_write_error(&error) => {
                    if attempts == 0 || attempts.is_multiple_of(10) {
                        tracing::warn!(
                            error = %error,
                            retry_delay_ms = delay.as_millis(),
                            attempts,
                            "Valkey mutation was not acknowledged; retaining ordered KV event for retry"
                        );
                    }
                    attempts = attempts.saturating_add(1);
                    tokio::select! {
                        _ = self.inner.cancellation_token.cancelled() => {
                            bail!("Valkey mutation cancelled during shutdown");
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                    delay = (delay * 2).min(self.inner.primary.write_retry_max_delay());
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    async fn command_direct_event_batch_and_replicate(&self, commands: &[&[&[u8]]]) -> Result<()> {
        let mut delay = WRITE_RETRY_INITIAL_DELAY;
        let mut attempts = 0_u32;
        loop {
            let result = if self.inner.replication_quorum == 0 {
                self.inner
                    .primary
                    .command_direct_event_batch(commands)
                    .await
            } else {
                self.inner
                    .primary
                    .command_direct_event_batch_and_wait(
                        commands,
                        self.inner.replication_quorum,
                        attempts > 0,
                    )
                    .await
            };
            match result {
                Ok(()) => return Ok(()),
                Err(error) if retryable_write_error(&error) => {
                    if attempts == 0 || attempts.is_multiple_of(10) {
                        tracing::warn!(
                            error = %error,
                            batch_events = commands.len(),
                            retry_delay_ms = delay.as_millis(),
                            attempts,
                            "Pipelined Valkey mutations were not acknowledged; retaining the entire ordered batch for retry"
                        );
                    }
                    attempts = attempts.saturating_add(1);
                    tokio::select! {
                        _ = self.inner.cancellation_token.cancelled() => {
                            bail!("Pipelined Valkey mutations cancelled during shutdown");
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                    delay = (delay * 2).min(self.inner.primary.write_retry_max_delay());
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Same retry and quorum behavior as worker event writes, but on an
    /// independent admission connection and with the command's binary reply
    /// preserved.  Retrying with the same nonce is safe; the retry barrier
    /// makes `WAIT` prove the ambiguous original mutation reached the replica.
    pub(super) async fn command_admission_and_replicate(
        &self,
        operation: AdmissionOperation,
        arguments: &[&[u8]],
    ) -> Result<RespValue> {
        let mut delay = WRITE_RETRY_INITIAL_DELAY;
        let mut attempts = 0_u32;
        loop {
            let result = if self.inner.replication_quorum == 0 {
                match operation {
                    AdmissionOperation::Select => {
                        self.inner.primary.command_admission_select(arguments).await
                    }
                    AdmissionOperation::Lifecycle => {
                        self.inner
                            .primary
                            .command_admission_lifecycle(arguments)
                            .await
                    }
                }
            } else {
                match operation {
                    AdmissionOperation::Select => {
                        self.inner
                            .primary
                            .command_admission_select_and_wait(
                                arguments,
                                self.inner.replication_quorum,
                                attempts > 0,
                            )
                            .await
                    }
                    AdmissionOperation::Lifecycle => {
                        self.inner
                            .primary
                            .command_admission_lifecycle_and_wait(
                                arguments,
                                self.inner.replication_quorum,
                                attempts > 0,
                            )
                            .await
                    }
                }
            };
            match result {
                Ok(response) => return Ok(response),
                Err(error) if retryable_write_error(&error) => {
                    if attempts == 0 || attempts.is_multiple_of(10) {
                        tracing::warn!(
                            error = %error,
                            retry_delay_ms = delay.as_millis(),
                            attempts,
                            "Valkey admission mutation was not acknowledged; retrying with the same nonce"
                        );
                    }
                    attempts = attempts.saturating_add(1);
                    tokio::select! {
                        _ = self.inner.cancellation_token.cancelled() => {
                            bail!("Valkey admission mutation cancelled during shutdown");
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                    delay = (delay * 2).min(self.inner.primary.write_retry_max_delay());
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

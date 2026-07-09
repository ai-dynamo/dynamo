// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Primary routing, pooled command lanes, failover, and replica barriers.

use super::*;

#[derive(Clone, Copy)]
struct PrimaryPoolConfig {
    readers: usize,
    direct_event_writers: usize,
    admission_select_writers: usize,
    admission_lifecycle_writers: usize,
}

impl PrimaryPoolConfig {
    fn frontend(readers: usize) -> Self {
        Self {
            readers: readers.max(1),
            direct_event_writers: 0,
            admission_select_writers: FRONTEND_ADMISSION_LANES,
            admission_lifecycle_writers: FRONTEND_ADMISSION_LANES,
        }
    }

    fn worker(direct_event_writers: usize) -> Self {
        Self {
            readers: 1,
            direct_event_writers: direct_event_writers.clamp(1, MAX_WORKER_DIRECT_EVENT_LANES),
            admission_select_writers: 0,
            admission_lifecycle_writers: 0,
        }
    }
}

pub(super) struct ValkeyPrimary {
    endpoint: StdRwLock<String>,
    /// Incremented when any lane proves its socket is attached to a demoted
    /// or otherwise non-primary backend. Every other lane lazily discards its
    /// old-generation socket before the next command.
    topology_generation: AtomicU64,
    sentinel: Option<ValkeySentinelConfig>,
    allow_degraded_writes: bool,
    /// Emit one operator-visible degraded-durability warning per topology
    /// generation. Per-command details remain available at debug level.
    degraded_warning_generation: AtomicU64,
    sentinel_refresh: Mutex<()>,
    writer: Mutex<Option<RespConnection>>,
    /// Direct worker `APPLY_OWNED` commands use a bounded lane set so
    /// independent DP-rank processors can publish concurrently. Lifecycle,
    /// recovery, and registration commands deliberately remain on `writer`.
    pub(super) direct_event_writers: Vec<Mutex<Option<RespConnection>>>,
    direct_event_cursor: AtomicUsize,
    /// `SELECT_RESERVE` has its own lanes so high request concurrency does
    /// not serialize every frontend through one mutation/`WAIT` round trip.
    /// Each lane keeps its lock from mutation through `WAIT`, preserving the
    /// acknowledgement contract for that command.
    pub(super) admission_select_writers: Vec<Mutex<Option<RespConnection>>>,
    /// Lease renewals and releases must remain responsive even while selects
    /// are saturated. They use an equally sized independent lane set and are
    /// serialized per lease by `ReservationLeaseInner::state`.
    pub(super) admission_lifecycle_writers: Vec<Mutex<Option<RespConnection>>>,
    admission_select_cursor: AtomicUsize,
    admission_lifecycle_cursor: AtomicUsize,
    readers: Vec<Mutex<Option<RespConnection>>>,
    reader_cursor: AtomicUsize,
}

impl ValkeyPrimary {
    pub(super) fn new(endpoint: String, connection_pool_size: usize) -> Self {
        Self::with_endpoint(endpoint, connection_pool_size, None, false)
    }

    pub(super) fn new_worker(endpoint: String, direct_event_pool_size: usize) -> Self {
        Self::from_pool_config(
            endpoint,
            PrimaryPoolConfig::worker(direct_event_pool_size),
            None,
            false,
        )
    }

    pub(super) async fn new_with_sentinel(
        sentinel: ValkeySentinelConfig,
        connection_pool_size: usize,
        allow_degraded_writes: bool,
    ) -> std::result::Result<Self, RespError> {
        let endpoint = sentinel.resolve_validated_primary().await?;
        Ok(Self::with_endpoint(
            endpoint,
            connection_pool_size,
            Some(sentinel),
            allow_degraded_writes,
        ))
    }

    pub(super) async fn new_worker_with_sentinel(
        sentinel: ValkeySentinelConfig,
        direct_event_pool_size: usize,
        allow_degraded_writes: bool,
    ) -> std::result::Result<Self, RespError> {
        let endpoint = sentinel.resolve_validated_primary().await?;
        Ok(Self::from_pool_config(
            endpoint,
            PrimaryPoolConfig::worker(direct_event_pool_size),
            Some(sentinel),
            allow_degraded_writes,
        ))
    }

    pub(super) fn with_endpoint(
        endpoint: String,
        connection_pool_size: usize,
        sentinel: Option<ValkeySentinelConfig>,
        allow_degraded_writes: bool,
    ) -> Self {
        Self::from_pool_config(
            endpoint,
            PrimaryPoolConfig::frontend(connection_pool_size),
            sentinel,
            allow_degraded_writes,
        )
    }

    fn from_pool_config(
        endpoint: String,
        pools: PrimaryPoolConfig,
        sentinel: Option<ValkeySentinelConfig>,
        allow_degraded_writes: bool,
    ) -> Self {
        Self {
            endpoint: StdRwLock::new(endpoint),
            topology_generation: AtomicU64::new(0),
            sentinel,
            allow_degraded_writes,
            degraded_warning_generation: AtomicU64::new(u64::MAX),
            sentinel_refresh: Mutex::new(()),
            writer: Mutex::new(None),
            direct_event_writers: (0..pools.direct_event_writers)
                .map(|_| Mutex::new(None))
                .collect(),
            direct_event_cursor: AtomicUsize::new(0),
            admission_select_writers: (0..pools.admission_select_writers)
                .map(|_| Mutex::new(None))
                .collect(),
            admission_lifecycle_writers: (0..pools.admission_lifecycle_writers)
                .map(|_| Mutex::new(None))
                .collect(),
            admission_select_cursor: AtomicUsize::new(0),
            admission_lifecycle_cursor: AtomicUsize::new(0),
            readers: (0..pools.readers).map(|_| Mutex::new(None)).collect(),
            reader_cursor: AtomicUsize::new(0),
        }
    }

    #[cfg(test)]
    fn connection_capacity(&self) -> usize {
        1 + self.readers.len()
            + self.direct_event_writers.len()
            + self.admission_select_writers.len()
            + self.admission_lifecycle_writers.len()
    }

    pub(super) async fn command_write(
        &self,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        self.command(&self.writer, arguments).await
    }

    pub(super) fn generation(&self) -> u64 {
        self.topology_generation.load(Ordering::Acquire)
    }

    pub(super) fn endpoint(&self) -> String {
        self.endpoint
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub(super) fn write_retry_max_delay(&self) -> Duration {
        if self.sentinel.is_some() {
            SENTINEL_WRITE_RETRY_MAX_DELAY
        } else {
            WRITE_RETRY_MAX_DELAY
        }
    }

    pub(super) fn replication_wait_timeout_ms(&self, connection_generation: u64) -> u64 {
        if self.allow_degraded_writes && self.sentinel.is_some() && connection_generation > 0 {
            DEGRADED_FAILOVER_WAIT_TIMEOUT_MS
        } else {
            REPLICATION_WAIT_TIMEOUT_MS
        }
    }

    pub(super) fn first_degraded_warning_for_generation(&self, generation: u64) -> bool {
        self.degraded_warning_generation
            .swap(generation, Ordering::AcqRel)
            != generation
    }

    pub(super) fn discard_stale_connection(&self, connection: &mut Option<RespConnection>) {
        let generation = self.generation();
        if connection
            .as_ref()
            .is_some_and(|connection| connection.topology_generation != generation)
        {
            *connection = None;
        }
    }

    pub(super) fn observe_topology_error(&self, connection_generation: u64, error: &RespError) {
        if !topology_error(error) {
            return;
        }
        // Serialize endpoint replacement and generation-only invalidation so
        // a reader can never observe a newly installed address with an old
        // generation after the route lock is released.
        let endpoint = self
            .endpoint
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(next_generation) = connection_generation.checked_add(1) else {
            tracing::error!(
                connection_generation,
                error = %error,
                "Valkey topology generation exhausted; cannot invalidate pooled connections"
            );
            return;
        };
        if self
            .topology_generation
            .compare_exchange(
                connection_generation,
                next_generation,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            tracing::warn!(
                endpoint = %*endpoint,
                connection_generation,
                next_generation,
                error = %error,
                "Valkey primary topology changed; invalidating every pooled connection"
            );
        }
    }

    pub(super) fn install_primary_endpoint(
        &self,
        connection_generation: u64,
        endpoint: String,
        error: &RespError,
    ) -> bool {
        let mut current = self
            .endpoint
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.generation() != connection_generation || *current == endpoint {
            return false;
        }
        let Some(next_generation) = connection_generation.checked_add(1) else {
            tracing::error!(
                connection_generation,
                error = %error,
                "Valkey topology generation exhausted; cannot install Sentinel primary"
            );
            return false;
        };
        let previous = std::mem::replace(&mut *current, endpoint);
        if self
            .topology_generation
            .compare_exchange(
                connection_generation,
                next_generation,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // Generation changes are serialized by the same route lock.
            // Restore the old endpoint rather than publishing a route without
            // an invalidation ticket if that invariant is ever violated.
            *current = previous;
            return false;
        }
        tracing::warn!(
            previous_endpoint = %previous,
            endpoint = %*current,
            connection_generation,
            next_generation,
            error = %error,
            "Valkey Sentinel elected a new primary; invalidating every pooled connection"
        );
        true
    }

    /// Refresh the failed route. Returns true only when this call's Sentinel
    /// quorum and ROLE check confirmed the exact endpoint/generation on which
    /// the failed mutation ran. Callers may use that proof for an explicitly
    /// configured degraded-durability decision.
    pub(super) async fn handle_primary_error(
        &self,
        connection_generation: u64,
        error: &RespError,
    ) -> bool {
        let Some(sentinel) = self.sentinel.as_ref() else {
            self.observe_topology_error(connection_generation, error);
            return false;
        };
        if !sentinel_refresh_error(error) {
            self.observe_topology_error(connection_generation, error);
            return false;
        }

        // Only one failed lane queries the Sentinel quorum. Other lanes wait
        // for its generation change and then reuse the resolved route.
        let _refresh = self.sentinel_refresh.lock().await;
        if self.generation() != connection_generation {
            return false;
        }
        match sentinel.resolve_validated_primary().await {
            Ok(endpoint) => {
                if endpoint == self.endpoint() {
                    if topology_error(error) {
                        // The address can front a newly promoted backend while
                        // this pooled socket remains attached to its demoted
                        // predecessor. Force every lane to reconnect.
                        self.observe_topology_error(connection_generation, error);
                        return false;
                    }
                    return self.generation() == connection_generation;
                }
                if !self.install_primary_endpoint(connection_generation, endpoint, error) {
                    // A demoted backend can remain at the same stable address
                    // while an operator updates its target. Preserve the old
                    // behavior and force every lane to reconnect in that case.
                    self.observe_topology_error(connection_generation, error);
                }
                false
            }
            Err(resolve_error) => {
                tracing::warn!(
                    error = %resolve_error,
                    failed_endpoint = %self.endpoint(),
                    connection_generation,
                    "Valkey Sentinel quorum did not resolve a replacement primary"
                );
                self.observe_topology_error(connection_generation, error);
                false
            }
        }
    }

    pub(super) async fn connect_current(&self) -> std::result::Result<RespConnection, RespError> {
        loop {
            let generation = self.generation();
            let endpoint = self.endpoint();
            let connection = match RespConnection::connect(&endpoint, generation).await {
                Ok(connection) => connection,
                Err(error) => {
                    self.handle_primary_error(generation, &error).await;
                    if generation != self.generation() {
                        continue;
                    }
                    return Err(error);
                }
            };
            if generation == self.generation() {
                return Ok(connection);
            }
            tracing::debug!(
                endpoint,
                generation,
                "Discarding Valkey connection opened during a topology generation change"
            );
        }
    }

    /// `WAIT` is scoped to the replication offset of one client connection.
    /// Keep the writer locked across the mutation and acknowledgement so the
    /// quorum result proves this exact module command reached a replica.
    pub(super) async fn command_write_and_wait(
        &self,
        arguments: &[&[u8]],
        replication_quorum: usize,
        force_replication_barrier: bool,
    ) -> std::result::Result<RespValue, RespError> {
        self.command_write_and_wait_on(
            &self.writer,
            arguments,
            replication_quorum,
            force_replication_barrier,
        )
        .await
    }

    pub(super) fn next_writer<'a>(
        writers: &'a [Mutex<Option<RespConnection>>],
        cursor: &AtomicUsize,
        lane_name: &str,
    ) -> std::result::Result<&'a Mutex<Option<RespConnection>>, RespError> {
        if writers.is_empty() {
            return Err(RespError::Protocol(format!(
                "Valkey {lane_name} lane is unavailable for this client role"
            )));
        }
        let index = cursor.fetch_add(1, Ordering::Relaxed) % writers.len();
        Ok(&writers[index])
    }

    pub(super) async fn command_admission_select(
        &self,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        self.command(
            Self::next_writer(
                &self.admission_select_writers,
                &self.admission_select_cursor,
                "admission select",
            )?,
            arguments,
        )
        .await
    }

    pub(super) async fn command_admission_select_and_wait(
        &self,
        arguments: &[&[u8]],
        replication_quorum: usize,
        force_replication_barrier: bool,
    ) -> std::result::Result<RespValue, RespError> {
        self.command_write_and_wait_on(
            Self::next_writer(
                &self.admission_select_writers,
                &self.admission_select_cursor,
                "admission select",
            )?,
            arguments,
            replication_quorum,
            force_replication_barrier,
        )
        .await
    }

    pub(super) async fn command_admission_lifecycle(
        &self,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        self.command(
            Self::next_writer(
                &self.admission_lifecycle_writers,
                &self.admission_lifecycle_cursor,
                "admission lifecycle",
            )?,
            arguments,
        )
        .await
    }

    pub(super) async fn command_admission_lifecycle_and_wait(
        &self,
        arguments: &[&[u8]],
        replication_quorum: usize,
        force_replication_barrier: bool,
    ) -> std::result::Result<RespValue, RespError> {
        self.command_write_and_wait_on(
            Self::next_writer(
                &self.admission_lifecycle_writers,
                &self.admission_lifecycle_cursor,
                "admission lifecycle",
            )?,
            arguments,
            replication_quorum,
            force_replication_barrier,
        )
        .await
    }

    pub(super) async fn command_direct_event_batch(
        &self,
        commands: &[&[&[u8]]],
    ) -> std::result::Result<(), RespError> {
        let writer = Self::next_writer(
            &self.direct_event_writers,
            &self.direct_event_cursor,
            "direct event",
        )?;
        let queued_at = Instant::now();
        let mut connection = writer.lock().await;
        tracing::trace!(
            valkey_connection_queue_us = queued_at.elapsed().as_micros(),
            valkey_pipeline_commands = commands.len(),
            "Acquired pipelined Valkey mutation lane"
        );
        self.discard_stale_connection(&mut connection);
        let mut active_connection = match connection.take() {
            Some(connection) => connection,
            None => self.connect_current().await?,
        };
        let responses = self
            .command_pipeline_once(&mut active_connection, commands)
            .await?;
        let result = validate_apply_responses(&responses).map(|_| ());
        // Every pipeline reply was consumed, so an unexpected complete reply
        // does not poison framing. Wire/server errors return above while the
        // slot is empty and therefore discard the socket.
        *connection = Some(active_connection);
        result
    }

    pub(super) async fn command_direct_event_batch_and_wait(
        &self,
        commands: &[&[&[u8]]],
        replication_quorum: usize,
        force_replication_barrier: bool,
    ) -> std::result::Result<(), RespError> {
        let Some(index_key) = commands
            .first()
            .and_then(|arguments| arguments.get(1))
            .copied()
        else {
            return Err(RespError::Protocol(
                "Valkey APPLY pipeline lacks a module key argument".to_string(),
            ));
        };
        if commands
            .iter()
            .any(|arguments| arguments.get(1).copied() != Some(index_key))
        {
            return Err(RespError::Protocol(
                "Valkey APPLY pipeline contains multiple module keys".to_string(),
            ));
        }

        let writer = Self::next_writer(
            &self.direct_event_writers,
            &self.direct_event_cursor,
            "direct event",
        )?;
        let queued_at = Instant::now();
        let mut connection = writer.lock().await;
        tracing::trace!(
            valkey_connection_queue_us = queued_at.elapsed().as_micros(),
            valkey_pipeline_commands = commands.len(),
            "Acquired pipelined Valkey mutation/WAIT lane"
        );
        self.discard_stale_connection(&mut connection);
        // Keep the slot empty throughout one full wire pipeline. Cancellation
        // at any point drops the socket and all unread replies instead of
        // returning ambiguous framing to the pool.
        let mut active_connection = match connection.take() {
            Some(connection) => connection,
            None => self.connect_current().await?,
        };
        let quorum = replication_quorum.to_string();
        let timeout_ms = self
            .replication_wait_timeout_ms(active_connection.topology_generation)
            .to_string();
        let barrier_arguments = [b"DYNKV.BARRIER".as_slice(), index_key];
        let wait_arguments = [b"WAIT".as_slice(), quorum.as_bytes(), timeout_ms.as_bytes()];
        let mut pipeline = Vec::with_capacity(commands.len() + 2);
        pipeline.extend_from_slice(commands);
        if force_replication_barrier {
            // A retry may run on a different event lane. One barrier after
            // every replayed idempotent APPLY gives this socket a replication
            // offset covering the entire ambiguous batch.
            pipeline.push(barrier_arguments.as_slice());
        }
        // WAIT is written in the same syscall batch after all APPLY commands
        // (and the retry barrier). Valkey executes this connection in order,
        // so one response-reading phase proves the whole batch reached quorum.
        pipeline.push(wait_arguments.as_slice());
        let responses = self
            .command_pipeline_once(&mut active_connection, &pipeline)
            .await?;
        let apply_count = commands.len();
        let command_generation = active_connection.topology_generation;
        let mut result = validate_apply_responses(&responses[..apply_count]).and_then(|_| {
            let mut next_response = apply_count;
            if force_replication_barrier {
                match &responses[next_response] {
                    RespValue::Simple(value) if value == "OK" => {}
                    other => {
                        return Err(RespError::Protocol(format!(
                            "unexpected Valkey replication barrier response: {}",
                            response_kind(other)
                        )));
                    }
                }
                next_response += 1;
            }
            let wait_response = &responses[next_response];
            match wait_response {
                RespValue::Integer(acknowledged) if *acknowledged >= replication_quorum as i64 => {
                    Ok(())
                }
                RespValue::Integer(acknowledged) => Err(RespError::Protocol(format!(
                    "Valkey replication quorum not met: expected {replication_quorum}, received {acknowledged}"
                ))),
                other => Err(RespError::Protocol(format!(
                    "unexpected Valkey WAIT response: {}",
                    response_kind(other)
                ))),
            }
        });
        let accept_degraded = if let Err(error) = &result {
            let confirmed = self.handle_primary_error(command_generation, error).await;
            self.allow_degraded_writes && confirmed && replication_quorum_error(error)
        } else {
            false
        };
        if accept_degraded {
            if self.first_degraded_warning_for_generation(command_generation) {
                tracing::warn!(
                    endpoint = %self.endpoint(),
                    command_generation,
                    wait_timeout_ms = %timeout_ms,
                    pipeline_commands = commands.len(),
                    "Valkey entered Sentinel-confirmed degraded durability without a replica acknowledgement"
                );
            } else {
                tracing::debug!(
                    endpoint = %self.endpoint(),
                    command_generation,
                    wait_timeout_ms = %timeout_ms,
                    pipeline_commands = commands.len(),
                    "Accepting Sentinel-confirmed Valkey event batch without a replica acknowledgement"
                );
            }
            result = Ok(());
        }
        // All replies were consumed before validation, so complete protocol
        // errors retain a framed socket. A '-' or partial response returned
        // above with the pool slot empty and discarded the connection.
        *connection = Some(active_connection);
        result
    }

    pub(super) async fn command_write_and_wait_on(
        &self,
        writer: &Mutex<Option<RespConnection>>,
        arguments: &[&[u8]],
        replication_quorum: usize,
        force_replication_barrier: bool,
    ) -> std::result::Result<RespValue, RespError> {
        let queued_at = Instant::now();
        let mut connection = writer.lock().await;
        tracing::trace!(
            valkey_connection_queue_us = queued_at.elapsed().as_micros(),
            "Acquired serialized Valkey mutation lane"
        );
        self.discard_stale_connection(&mut connection);
        // Keep the pool slot empty while any wire I/O is in flight. Dropping
        // this future then drops the socket instead of returning a connection
        // whose unread mutation, barrier, or WAIT reply could corrupt the next
        // command on this lane.
        let mut active_connection = match connection.take() {
            Some(connection) => connection,
            None => self.connect_current().await?,
        };
        // Do not reconnect-and-retry either half of this protocol. `WAIT` is
        // scoped to the replication offset of this TCP connection; a retry on
        // another connection could acknowledge an unrelated offset. Failing
        // closed leaves the normalized event available to the caller's retry
        // and recovery path instead of claiming a false HA acknowledgement.
        let mutation_response = self.command_once(&mut active_connection, arguments).await?;
        if matches!(&mutation_response, RespValue::Simple(value) if value == "NOOP")
            && !force_replication_barrier
        {
            *connection = Some(active_connection);
            return Ok(mutation_response);
        }

        if force_replication_barrier {
            let Some(index_key) = arguments.get(1).copied() else {
                *connection = Some(active_connection);
                return Err(RespError::Protocol(
                    "Valkey mutation lacks a module key argument".to_string(),
                ));
            };
            // An earlier APPLY may have reached the primary before the socket
            // or WAIT failed. Retrying it can return NOOP, which carries no
            // replication offset on this replacement connection. Replicate a
            // no-state-change barrier and WAIT for that offset instead: an
            // acknowledgement proves every preceding primary mutation,
            // including the ambiguous original APPLY, reached the standby.
            let barrier_response = self
                .command_once(&mut active_connection, &[b"DYNKV.BARRIER", index_key])
                .await?;
            match barrier_response {
                RespValue::Simple(value) if value == "OK" => {}
                other => {
                    *connection = Some(active_connection);
                    return Err(RespError::Protocol(format!(
                        "unexpected Valkey replication barrier response: {}",
                        response_kind(&other)
                    )));
                }
            }
        }
        let quorum = replication_quorum.to_string();
        let command_generation = active_connection.topology_generation;
        let timeout_ms = self
            .replication_wait_timeout_ms(command_generation)
            .to_string();
        let wait_response = self
            .command_once(
                &mut active_connection,
                &[b"WAIT", quorum.as_bytes(), timeout_ms.as_bytes()],
            )
            .await?;
        let result = match wait_response {
            RespValue::Integer(acknowledged) if acknowledged >= replication_quorum as i64 => {
                Ok(mutation_response)
            }
            RespValue::Integer(acknowledged) => {
                let error = RespError::Protocol(format!(
                    "Valkey replication quorum not met: expected {replication_quorum}, received {acknowledged}"
                ));
                let confirmed = self.handle_primary_error(command_generation, &error).await;
                if self.allow_degraded_writes && confirmed {
                    if self.first_degraded_warning_for_generation(command_generation) {
                        tracing::warn!(
                            endpoint = %self.endpoint(),
                            command_generation,
                            wait_timeout_ms = %timeout_ms,
                            acknowledged,
                            replication_quorum,
                            "Valkey entered Sentinel-confirmed degraded durability without a replica acknowledgement"
                        );
                    } else {
                        tracing::debug!(
                            endpoint = %self.endpoint(),
                            command_generation,
                            wait_timeout_ms = %timeout_ms,
                            acknowledged,
                            replication_quorum,
                            "Accepting Sentinel-confirmed Valkey mutation without a replica acknowledgement"
                        );
                    }
                    Ok(mutation_response)
                } else {
                    Err(error)
                }
            }
            other => {
                let error = RespError::Protocol(format!(
                    "unexpected Valkey WAIT response: {}",
                    response_kind(&other)
                ));
                self.handle_primary_error(command_generation, &error).await;
                Err(error)
            }
        };
        *connection = Some(active_connection);
        result
    }

    pub(super) async fn command_read(
        &self,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        let index = self.reader_cursor.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        self.command(&self.readers[index], arguments).await
    }

    pub(super) async fn command(
        &self,
        connection: &Mutex<Option<RespConnection>>,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        let mut connection = connection.lock().await;
        self.command_locked(&mut connection, arguments).await
    }

    pub(super) async fn command_locked(
        &self,
        connection: &mut Option<RespConnection>,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        self.discard_stale_connection(connection);
        // The mutex-owned slot must remain empty across every await. If this
        // future is cancelled, `active_connection` is dropped and a later
        // borrower reconnects instead of consuming an abandoned RESP reply.
        let mut active_connection = connection.take();
        for attempt in 0..2 {
            let mut current = match active_connection.take() {
                Some(connection) => connection,
                None => self.connect_current().await?,
            };
            let connection_generation = current.topology_generation;
            let result = current.command(arguments).await;
            match result {
                Ok(response) => {
                    *connection = Some(current);
                    return Ok(response);
                }
                Err(error @ RespError::Server(_)) => {
                    self.handle_primary_error(connection_generation, &error)
                        .await;
                    if !topology_error(&error) {
                        // A server error is a complete RESP response, so this
                        // socket remains framed correctly unless it also proves
                        // that the primary topology changed.
                        *connection = Some(current);
                    }
                    return Err(error);
                }
                Err(error) if attempt == 0 => {
                    self.handle_primary_error(connection_generation, &error)
                        .await;
                    tracing::debug!(endpoint = %self.endpoint(), error = %error, "reconnecting Valkey client");
                }
                Err(error) => {
                    self.handle_primary_error(connection_generation, &error)
                        .await;
                    return Err(error);
                }
            }
        }
        unreachable!("two attempts either return or fail")
    }

    pub(super) async fn command_once(
        &self,
        connection: &mut RespConnection,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        let connection_generation = connection.topology_generation;
        let result = connection.command(arguments).await;
        if let Err(error) = &result {
            self.handle_primary_error(connection_generation, error)
                .await;
        }
        result
    }

    pub(super) async fn command_pipeline_once(
        &self,
        connection: &mut RespConnection,
        commands: &[&[&[u8]]],
    ) -> std::result::Result<Vec<RespValue>, RespError> {
        let connection_generation = connection.topology_generation;
        let result = connection.command_pipeline(commands).await;
        if let Err(error) = &result {
            self.handle_primary_error(connection_generation, error)
                .await;
        }
        result
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn frontend_pool_does_not_allocate_worker_event_lanes() {
        let primary = ValkeyPrimary::new("127.0.0.1:6379".to_string(), 64);

        assert_eq!(primary.readers.len(), 64);
        assert!(primary.direct_event_writers.is_empty());
        assert_eq!(primary.admission_select_writers.len(), 4);
        assert_eq!(primary.admission_lifecycle_writers.len(), 4);
        assert_eq!(primary.connection_capacity(), 73);
    }

    #[test]
    fn worker_pool_allocates_only_worker_lane_families() {
        let primary = ValkeyPrimary::new_worker("127.0.0.1:6379".to_string(), 4);

        assert_eq!(primary.readers.len(), 1);
        assert_eq!(primary.direct_event_writers.len(), 4);
        assert!(primary.admission_select_writers.is_empty());
        assert!(primary.admission_lifecycle_writers.is_empty());
        assert_eq!(primary.connection_capacity(), 6);
    }
}

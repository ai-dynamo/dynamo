// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Valkey-backed device-tier KV index.
//!
//! The loadable `dynkv` Valkey module owns the persistent radix index.  Dynamo
//! keeps its existing scheduler and replica-sync protocol, so worker selection
//! retains the same admission, constraints, and active-load behavior as the
//! in-process router.  This client only externalizes prefix ownership.

use std::{
    collections::BTreeSet,
    mem::size_of,
    sync::{
        Arc, RwLock as StdRwLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use dynamo_kv_router::{
    indexer::{KvRouterError, MatchDetails},
    protocols::{
        DpRank, ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash, OverlapScores,
        RouterEvent, WorkerId, WorkerWithDpRank,
    },
};
use moka::future::Cache;
use rustc_hash::FxHashMap;
use tokio::{
    sync::{Mutex, OwnedSemaphorePermit, Semaphore},
    time::timeout,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[cfg(test)]
use std::io;
#[cfg(test)]
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

#[cfg(test)]
use crate::valkey_transport::validate_primary_role_response;
use crate::valkey_transport::{
    RespConnection, RespError, RespValue, ValkeySentinelConfig, parse_endpoint,
};

mod primary;
use primary::ValkeyPrimary;
mod config;
use config::PreparedValkeyIndexer;
mod errors;
use errors::*;
mod lease;
mod reservation;
use reservation::{
    PendingValkeyReservation, ReservationGrant, ReservationLeaseInner, ReservationLeaseState,
    ReservationLifecycleRequest, ReservationNonce, ReservationRequest,
};
pub(crate) use reservation::{ValkeyReservationCandidate, ValkeyReservationLease};
mod wire;
use wire::*;
mod operations;
mod wire_limits {
    include!(concat!(env!("OUT_DIR"), "/dynkv_limits.rs"));
}
use wire_limits::{
    DYNKV_MAX_ADMISSION_CANDIDATES, DYNKV_MAX_ADMISSION_DOMAIN_LENGTH,
    DYNKV_MAX_ADMISSION_LEASE_MS, DYNKV_MAX_MATCH_HASHES, DYNKV_MAX_REGISTRATION_RANKS,
};

const WIRE_VERSION: u8 = 1;
const EVENT_STORE: u8 = 1;
const EVENT_REMOVE: u8 = 2;
const EVENT_CLEAR: u8 = 3;
const ROOT_PARENT: u64 = u64::MAX;
const MATCH_ENTRY_BYTES: usize = 24;
/// Bound a synchronous replica acknowledgement so a failed standby makes the
/// indexer unavailable instead of silently degrading the HA guarantee.
const REPLICATION_WAIT_TIMEOUT_MS: u64 = 3_000;
/// Valkey parses WAIT's replica count as a non-negative signed `int`.
const MAX_REQUIRED_REPLICA_ACKS: u32 = i32::MAX as u32;
const WRITE_RETRY_INITIAL_DELAY: Duration = Duration::from_millis(50);
const WRITE_RETRY_MAX_DELAY: Duration = Duration::from_secs(5);
const SENTINEL_WRITE_RETRY_MAX_DELAY: Duration = Duration::from_millis(250);
const DEGRADED_FAILOVER_WAIT_TIMEOUT_MS: u64 = 100;
/// A small frontend cache avoids a primary round trip for repeated prompts.
/// Bound it by estimated retained bytes rather than entry count: a valid
/// request can contain tens of thousands of hashes and a match can contain
/// hundreds of worker records. Oversized keys bypass the cache entirely.
const MATCH_CACHE_MAX_BYTES: u64 = 16 * 1024 * 1024;
const MATCH_CACHE_MAX_KEY_HASHES: usize = 8 * 1024;
const MATCH_CACHE_TTL: Duration = Duration::from_secs(1);
/// Reads are affinity hints, but they must never stay pinned to a demoted
/// primary indefinitely. Retry through the externally managed stable primary
/// endpoint for a bounded window before reporting the indexer offline.
const PRIMARY_READ_FAILOVER_TIMEOUT: Duration = Duration::from_secs(5);
// Worker registration can coincide with thousands of lazy frontend
// connections and hundreds of worker-owned leases. Keep request-path reads
// fail-fast, but give the replay-safe startup generation query enough time to
// retry through that bounded connection burst.
const REGISTRATION_READ_FAILOVER_TIMEOUT: Duration = Duration::from_secs(30);
const PRIMARY_READ_RETRY_INITIAL_DELAY: Duration = Duration::from_millis(25);
const PRIMARY_READ_RETRY_MAX_DELAY: Duration = Duration::from_millis(250);
const ADMISSION_WIRE_VERSION: u8 = 1;
const WORKER_LEASED_REGISTRATION_WIRE_VERSION: u8 = 3;
const WORKER_LEASE_CONTROL_VERSION: u8 = 1;
pub(crate) const MAX_WORKER_RANKS: usize = DYNKV_MAX_REGISTRATION_RANKS;
const MAX_GC_INSPECTION_BUDGET: u32 = 1_048_576;
const REGISTRATION_CAS_RETRY_INITIAL_DELAY: Duration = Duration::from_millis(5);
const REGISTRATION_CAS_RETRY_MAX_DELAY: Duration = Duration::from_millis(250);
const ADMISSION_NO_CAPACITY: u8 = 0;
const ADMISSION_RESERVED: u8 = 1;
/// A frontend must have enough time to receive a successful reservation and
/// start its renewal task before the module can expire it. This is stricter
/// than the module's raw command bound by design.
const MIN_ADMISSION_LEASE_MS: u64 = 10_000;
const MAX_ADMISSION_PREFIX_HASHES: usize = DYNKV_MAX_MATCH_HASHES;
const MAX_ADMISSION_CANDIDATES: usize = DYNKV_MAX_ADMISSION_CANDIDATES;
const MAX_ADMISSION_DOMAIN_LENGTH: usize = DYNKV_MAX_ADMISSION_DOMAIN_LENGTH;
const MAX_ADMISSION_LEASE_MS: u64 = DYNKV_MAX_ADMISSION_LEASE_MS;
/// Limit per-frontend admission sockets independently of the match pool.  The
/// module serializes mutations itself, while too many client sockets merely
/// add contention at the primary.
const FRONTEND_ADMISSION_LANES: usize = 4;
/// Drop paths must never turn a Valkey outage into an unbounded queue of
/// per-request cleanup futures. Once this many best-effort cleanups are in
/// flight, later reservations rely on their module-owned lease expiry.
const MAX_PENDING_ADMISSION_CLEANUPS: usize = 4096;
/// Bound full selection requests retained only while reconciling an ambiguous
/// SELECT_RESERVE. Count-only bounds allow maximum-size requests to consume
/// multiple GiB per frontend.
const MAX_PENDING_ADMISSION_CLEANUP_BYTES: usize = 64 * 1024 * 1024;
/// Worker event streams are ordered independently per DP rank. Give those
/// streams independent sockets while bounding the primary-side connection
/// and command scheduling overhead for one worker process.
pub(crate) const DEFAULT_WORKER_DIRECT_EVENT_LANES: usize = 4;
pub(crate) const MAX_WORKER_DIRECT_EVENT_LANES: usize = 16;
/// Keep the steady-state renewal cadence at one third of a lease, but make
/// the first renewal both early and lease-specific.  A large admission burst
/// otherwise gives every reservation the same first wake-up and can starve
/// the lifecycle lanes long enough for the oldest leases to expire.
const RENEWAL_INTERVAL_DIVISOR: u64 = 3;
const RENEWAL_INITIAL_EARLIEST_DIVISOR: u64 = 16;
const RENEWAL_INITIAL_LATEST_DIVISOR: u64 = 4;

#[derive(Clone, Copy)]
enum AdmissionOperation {
    Select,
    Lifecycle,
}

struct ValkeyIndexerInner {
    index_key: Vec<u8>,
    primary: ValkeyPrimary,
    match_cache: Cache<Vec<LocalBlockHash>, MatchDetails>,
    match_cache_generation: AtomicU64,
    cancellation_token: CancellationToken,
    admission_cleanup_permits: Arc<Semaphore>,
    admission_cleanup_bytes: Arc<Semaphore>,
    /// Number of replicas required to acknowledge each mutation.  With a
    /// primary/replica pair this is one; a single-node development topology
    /// remains supported with zero.
    replication_quorum: usize,
}

fn match_cache_key_is_cacheable(block_hashes: &[LocalBlockHash]) -> bool {
    block_hashes.len() <= MATCH_CACHE_MAX_KEY_HASHES
}

fn match_cache_weight(key: &Vec<LocalBlockHash>, details: &MatchDetails) -> u32 {
    // Hash-table control bytes are implementation-specific. Charging one
    // machine word per slot keeps this estimate conservative without coupling
    // the cache to hashbrown internals.
    let score_entry_bytes = size_of::<WorkerWithDpRank>() + size_of::<u32>() + size_of::<usize>();
    let last_hash_entry_bytes =
        size_of::<WorkerWithDpRank>() + size_of::<ExternalSequenceBlockHash>() + size_of::<usize>();
    let bytes = size_of::<Vec<LocalBlockHash>>()
        .saturating_add(size_of::<MatchDetails>())
        .saturating_add(key.capacity().saturating_mul(size_of::<LocalBlockHash>()))
        .saturating_add(
            details
                .overlap_scores
                .scores
                .capacity()
                .saturating_mul(score_entry_bytes),
        )
        .saturating_add(
            details
                .overlap_scores
                .frequencies
                .capacity()
                .saturating_mul(size_of::<usize>()),
        )
        .saturating_add(
            details
                .last_matched_hashes
                .capacity()
                .saturating_mul(last_hash_entry_bytes),
        );
    u32::try_from(bytes).unwrap_or(u32::MAX).max(1)
}

/// Persistent device-tier KV index backed by `dynkv` on a Valkey primary.
///
/// [`ValkeyIndexer::new`] preserves the fixed endpoint behavior: the first
/// configured URL is the only client connection target. The opt-in Sentinel
/// constructor resolves a strict witness quorum and atomically invalidates all
/// pooled connections when that quorum reports a replacement primary. Neither
/// mode mirrors commands between data-node addresses.
#[derive(Clone)]
pub struct ValkeyIndexer {
    inner: Arc<ValkeyIndexerInner>,
}

fn validate_reservation_dimensions(
    domain_length: usize,
    prefix_hashes: usize,
    candidates: usize,
    lease_ms: u64,
) -> Result<()> {
    if domain_length == 0 || domain_length > MAX_ADMISSION_DOMAIN_LENGTH {
        bail!("Valkey admission domain must contain 1..={MAX_ADMISSION_DOMAIN_LENGTH} bytes");
    }
    if prefix_hashes > MAX_ADMISSION_PREFIX_HASHES {
        bail!("Valkey admission supports at most {MAX_ADMISSION_PREFIX_HASHES} prefix hashes");
    }
    if candidates == 0 || candidates > MAX_ADMISSION_CANDIDATES {
        bail!("Valkey admission requires 1..={MAX_ADMISSION_CANDIDATES} eligible candidates");
    }
    if !(MIN_ADMISSION_LEASE_MS..=MAX_ADMISSION_LEASE_MS).contains(&lease_ms) {
        bail!(
            "Valkey admission lease must be in {MIN_ADMISSION_LEASE_MS}..={MAX_ADMISSION_LEASE_MS} milliseconds"
        );
    }
    Ok(())
}

impl ValkeyIndexer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        urls: &str,
        connection_pool_size: u32,
        required_replica_acks: Option<u32>,
        namespace: &str,
        component: &str,
        index_scope: Option<&str>,
        _model_name: Option<&str>,
        block_size: u32,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        let prepared = PreparedValkeyIndexer::new(
            urls,
            connection_pool_size,
            required_replica_acks,
            namespace,
            component,
            index_scope,
            _model_name,
            block_size,
        )?;
        let primary = ValkeyPrimary::new(prepared.primary_endpoint.clone(), prepared.pool_size);
        Ok(Self::from_prepared(prepared, primary, cancellation_token))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_worker(
        urls: &str,
        direct_event_pool_size: u32,
        required_replica_acks: Option<u32>,
        namespace: &str,
        component: &str,
        index_scope: Option<&str>,
        block_size: u32,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        let prepared = PreparedValkeyIndexer::new(
            urls,
            direct_event_pool_size,
            required_replica_acks,
            namespace,
            component,
            index_scope,
            None,
            block_size,
        )?;
        let primary =
            ValkeyPrimary::new_worker(prepared.primary_endpoint.clone(), prepared.pool_size);
        Ok(Self::from_prepared(prepared, primary, cancellation_token))
    }

    /// Construct an indexer whose data-node route is elected by a strict
    /// quorum of independent Sentinel witnesses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn new_with_sentinel(
        urls: &str,
        connection_pool_size: u32,
        required_replica_acks: Option<u32>,
        namespace: &str,
        component: &str,
        index_scope: Option<&str>,
        model_name: Option<&str>,
        block_size: u32,
        cancellation_token: CancellationToken,
        sentinel: ValkeySentinelConfig,
        allow_degraded_writes: bool,
    ) -> Result<Self> {
        if allow_degraded_writes {
            sentinel.validate_degraded_writes()?;
        }
        let prepared = PreparedValkeyIndexer::new(
            urls,
            connection_pool_size,
            required_replica_acks,
            namespace,
            component,
            index_scope,
            model_name,
            block_size,
        )?;
        let primary =
            ValkeyPrimary::new_with_sentinel(sentinel, prepared.pool_size, allow_degraded_writes)
                .await?;
        Ok(Self::from_prepared(prepared, primary, cancellation_token))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn new_worker_with_sentinel(
        urls: &str,
        direct_event_pool_size: u32,
        required_replica_acks: Option<u32>,
        namespace: &str,
        component: &str,
        index_scope: Option<&str>,
        block_size: u32,
        cancellation_token: CancellationToken,
        sentinel: ValkeySentinelConfig,
        allow_degraded_writes: bool,
    ) -> Result<Self> {
        if allow_degraded_writes {
            sentinel.validate_degraded_writes()?;
        }
        let prepared = PreparedValkeyIndexer::new(
            urls,
            direct_event_pool_size,
            required_replica_acks,
            namespace,
            component,
            index_scope,
            None,
            block_size,
        )?;
        let primary = ValkeyPrimary::new_worker_with_sentinel(
            sentinel,
            prepared.pool_size,
            allow_degraded_writes,
        )
        .await?;
        Ok(Self::from_prepared(prepared, primary, cancellation_token))
    }

    fn from_prepared(
        prepared: PreparedValkeyIndexer,
        primary: ValkeyPrimary,
        cancellation_token: CancellationToken,
    ) -> Self {
        Self {
            inner: Arc::new(ValkeyIndexerInner {
                index_key: prepared.index_key,
                primary,
                match_cache: Cache::builder()
                    .max_capacity(MATCH_CACHE_MAX_BYTES)
                    .weigher(match_cache_weight)
                    .time_to_live(MATCH_CACHE_TTL)
                    .build(),
                match_cache_generation: AtomicU64::new(0),
                cancellation_token,
                admission_cleanup_permits: Arc::new(Semaphore::new(MAX_PENDING_ADMISSION_CLEANUPS)),
                admission_cleanup_bytes: Arc::new(Semaphore::new(
                    MAX_PENDING_ADMISSION_CLEANUP_BYTES,
                )),
                replication_quorum: prepared.replication_quorum,
            }),
        }
    }

    /// Prepare an atomic module-side select-and-reserve operation.  The
    /// returned pending guard must be kept alive through the first reply so a
    /// cancellation after the request reaches Valkey cannot leak capacity.
    pub(crate) fn begin_reservation(
        &self,
        domain: &str,
        block_hashes: &[LocalBlockHash],
        candidates: Vec<ValkeyReservationCandidate>,
        lease_ms: u64,
    ) -> Result<PendingValkeyReservation> {
        let domain = domain.as_bytes();
        validate_reservation_dimensions(
            domain.len(),
            block_hashes.len(),
            candidates.len(),
            lease_ms,
        )?;
        if candidates
            .iter()
            .any(|candidate| candidate.capacity == 0 || candidate.capacity == u32::MAX)
        {
            bail!("Valkey admission candidate capacity must be in 1..u32::MAX");
        }
        Ok(PendingValkeyReservation {
            indexer: self.clone(),
            request: ReservationRequest {
                domain: domain.to_vec(),
                nonce: ReservationNonce::random(),
                lease_ms,
                block_hashes: block_hashes.to_vec(),
                candidates,
            },
            armed: true,
        })
    }

    async fn select_reservation(
        &self,
        request: &ReservationRequest,
    ) -> Result<Option<ReservationGrant>> {
        let payload = encode_select_reserve(request)?;
        let response = self
            .command_admission_and_replicate(
                AdmissionOperation::Select,
                &[
                    b"DYNKV.SELECT_RESERVE".as_slice(),
                    self.inner.index_key.as_slice(),
                    payload.as_slice(),
                ],
            )
            .await?;
        let RespValue::Bulk(payload) = response else {
            bail!(
                "Valkey SELECT_RESERVE returned {} instead of a binary reservation reply",
                response_kind(&response)
            );
        };
        decode_reservation_reply(&payload, request.nonce)
    }

    async fn release_reservation(
        &self,
        request: &ReservationLifecycleRequest,
        expected_expires_at_ms: u64,
    ) -> Result<bool> {
        let payload = encode_release(request, expected_expires_at_ms);
        let response = match self
            .command_admission_and_replicate(
                AdmissionOperation::Lifecycle,
                &[
                    b"DYNKV.RELEASE".as_slice(),
                    self.inner.index_key.as_slice(),
                    payload.as_slice(),
                ],
            )
            .await
        {
            Ok(response) => response,
            // A stale release must not touch a renewed reservation.  It is
            // equivalent to an idempotent no-op for the old lease owner.
            Err(error) if reservation_expired_error(&error) => return Ok(false),
            Err(error) => return Err(error),
        };
        let RespValue::Bulk(payload) = response else {
            bail!(
                "Valkey RELEASE returned {} instead of a binary admission reply",
                response_kind(&response)
            );
        };
        decode_admission_status(&payload)
    }

    async fn renew_reservation(
        &self,
        request: &ReservationLifecycleRequest,
        expected_expires_at_ms: u64,
    ) -> Result<Option<ReservationGrant>> {
        let payload = encode_renew(request, expected_expires_at_ms);
        let response = match self
            .command_admission_and_replicate(
                AdmissionOperation::Lifecycle,
                &[
                    b"DYNKV.RENEW".as_slice(),
                    self.inner.index_key.as_slice(),
                    payload.as_slice(),
                ],
            )
            .await
        {
            Ok(response) => response,
            Err(error) if reservation_expired_error(&error) => return Ok(None),
            Err(error) => return Err(error),
        };
        let RespValue::Bulk(payload) = response else {
            bail!(
                "Valkey RENEW returned {} instead of a binary reservation reply",
                response_kind(&response)
            );
        };
        decode_reservation_reply(&payload, request.nonce)
    }

    fn synchronize_match_cache_generation(&self) {
        let generation = self.inner.primary.generation();
        let cached_generation = self
            .inner
            .match_cache_generation
            .swap(generation, Ordering::AcqRel);
        if cached_generation != generation {
            self.inner.match_cache.invalidate_all();
        }
    }

    async fn command_primary_read_with_retry(
        &self,
        arguments: &[&[u8]],
    ) -> std::result::Result<RespValue, RespError> {
        self.command_primary_read_with_retry_timeout(arguments, PRIMARY_READ_FAILOVER_TIMEOUT)
            .await
    }

    async fn command_primary_read_with_retry_timeout(
        &self,
        arguments: &[&[u8]],
        retry_timeout: Duration,
    ) -> std::result::Result<RespValue, RespError> {
        let retry = async {
            let mut delay = PRIMARY_READ_RETRY_INITIAL_DELAY;
            loop {
                match self.inner.primary.command_read(arguments).await {
                    Ok(response) => return Ok(response),
                    Err(error) if retryable_primary_read_error(&error) => {
                        tracing::debug!(
                            error = %error,
                            retry_delay_ms = delay.as_millis(),
                            "Valkey primary read unavailable during topology transition"
                        );
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(PRIMARY_READ_RETRY_MAX_DELAY);
                    }
                    Err(error) => return Err(error),
                }
            }
        };
        tokio::select! {
            _ = self.inner.cancellation_token.cancelled() => {
                Err(RespError::Protocol("Valkey primary read cancelled during shutdown".to_string()))
            }
            result = timeout(retry_timeout, retry) => {
                result.map_err(|_| RespError::Timeout)?
            }
        }
    }

    pub(super) async fn find_match_details(
        &self,
        block_hashes: &[LocalBlockHash],
    ) -> Result<MatchDetails, KvRouterError> {
        if block_hashes.is_empty() {
            return Ok(MatchDetails::new());
        }
        self.synchronize_match_cache_generation();
        let cache_key = match_cache_key_is_cacheable(block_hashes).then(|| block_hashes.to_vec());
        if let Some(cache_key) = cache_key.as_ref()
            && let Some(details) = self.inner.match_cache.get(cache_key).await
        {
            return Ok(details);
        }
        let request = encode_match(block_hashes);
        let response = self
            .command_primary_read_with_retry(&[
                b"DYNKV.MATCH_PRIMARY",
                &self.inner.index_key,
                &request,
            ])
            .await
            .map_err(|error| {
                tracing::warn!(error = %error, "Valkey prefix match failed");
                KvRouterError::IndexerOffline
            })?;
        let RespValue::Bulk(payload) = response else {
            tracing::warn!("Valkey prefix match returned a non-bulk reply");
            return Err(KvRouterError::IndexerOffline);
        };
        let details = decode_match(&payload).map_err(|error| {
            tracing::warn!(error = %error, "Valkey prefix match returned invalid data");
            KvRouterError::IndexerOffline
        })?;
        self.synchronize_match_cache_generation();
        if let Some(cache_key) = cache_key {
            self.inner
                .match_cache
                .insert(cache_key, details.clone())
                .await;
        }
        Ok(details)
    }
}

#[cfg(test)]
mod tests;

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ-based registry hub implementation.
//!
//! Uses ZMQ sockets for communication:
//! - ROUTER socket for query handling (DEALER/ROUTER pattern for async req/rep)
//! - SUB socket for receiving registrations (PUB/SUB pattern)

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tmq::{
    router::router,
    subscribe::subscribe,
    Context, Message, Multipart,
};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use super::super::config::RegistryHubConfig;
use super::super::protocol::{
    decode_can_offload, decode_match_sequence, decode_message_type,
    decode_register, encode_match_response, MessageType,
};
use super::super::storage::MokaStorage;
use super::super::traits::RegistryHub;
use super::super::types::HubStats;

/// ZMQ-based registry hub.
///
/// The hub is the single source of truth for the distributed registry.
/// It handles:
/// - Registration requests (PUB/SUB, fire-and-forget)
/// - Query requests (PUSH/PULL pattern with client ID for routing responses)
///
/// # Protocol
///
/// Queries use a simple request-response pattern:
/// - Client sends: [client_id (8 bytes), query_payload]
/// - Hub responds via separate push socket: [client_id, response_payload]
///
/// # Example
/// ```ignore
/// let config = RegistryHubConfig::default();
/// let hub = ZmqRegistryHub::new(config)?;
///
/// // Run until cancelled
/// let cancel = CancellationToken::new();
/// hub.serve(cancel).await?;
/// ```
pub struct ZmqRegistryHub {
    config: RegistryHubConfig,
    storage: Arc<MokaStorage>,
    stats: Arc<HubStatsAtomic>,
}

/// Atomic version of HubStats for thread-safe updates.
struct HubStatsAtomic {
    total_registered: AtomicU64,
    total_offload_queries: AtomicU64,
    total_match_queries: AtomicU64,
    dedup_hits: AtomicU64,
    dedup_total_checked: AtomicU64,
    unique_clients: AtomicU64,
    // Timing stats (in microseconds)
    total_query_time_us: AtomicU64,
    min_query_time_us: AtomicU64,
    max_query_time_us: AtomicU64,
    // Rolling window for throughput (queries in last interval)
    interval_queries: AtomicU64,
    interval_query_time_us: AtomicU64,
}

impl HubStatsAtomic {
    fn new() -> Self {
        Self {
            total_registered: AtomicU64::new(0),
            total_offload_queries: AtomicU64::new(0),
            total_match_queries: AtomicU64::new(0),
            dedup_hits: AtomicU64::new(0),
            dedup_total_checked: AtomicU64::new(0),
            unique_clients: AtomicU64::new(0),
            total_query_time_us: AtomicU64::new(0),
            min_query_time_us: AtomicU64::new(u64::MAX),
            max_query_time_us: AtomicU64::new(0),
            interval_queries: AtomicU64::new(0),
            interval_query_time_us: AtomicU64::new(0),
        }
    }

    fn record_query_time(&self, duration_us: u64) {
        self.total_query_time_us.fetch_add(duration_us, Ordering::Relaxed);
        self.interval_queries.fetch_add(1, Ordering::Relaxed);
        self.interval_query_time_us.fetch_add(duration_us, Ordering::Relaxed);

        // Update min (compare-and-swap loop)
        let mut current_min = self.min_query_time_us.load(Ordering::Relaxed);
        while duration_us < current_min {
            match self.min_query_time_us.compare_exchange_weak(
                current_min,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current_min = c,
            }
        }

        // Update max
        let mut current_max = self.max_query_time_us.load(Ordering::Relaxed);
        while duration_us > current_max {
            match self.max_query_time_us.compare_exchange_weak(
                current_max,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current_max = c,
            }
        }
    }

    fn reset_interval(&self) -> (u64, u64) {
        let queries = self.interval_queries.swap(0, Ordering::Relaxed);
        let time_us = self.interval_query_time_us.swap(0, Ordering::Relaxed);
        (queries, time_us)
    }

    fn to_hub_stats(&self) -> HubStats {
        HubStats {
            total_registered: self.total_registered.load(Ordering::Relaxed),
            total_offload_queries: self.total_offload_queries.load(Ordering::Relaxed),
            total_match_queries: self.total_match_queries.load(Ordering::Relaxed),
            dedup_hits: self.dedup_hits.load(Ordering::Relaxed),
            dedup_total_checked: self.dedup_total_checked.load(Ordering::Relaxed),
        }
    }

    fn avg_query_time_us(&self) -> f64 {
        let total_queries = self.total_offload_queries.load(Ordering::Relaxed)
            + self.total_match_queries.load(Ordering::Relaxed);
        if total_queries == 0 {
            return 0.0;
        }
        self.total_query_time_us.load(Ordering::Relaxed) as f64 / total_queries as f64
    }

    fn min_query_time_us(&self) -> u64 {
        let min = self.min_query_time_us.load(Ordering::Relaxed);
        if min == u64::MAX { 0 } else { min }
    }

    fn max_query_time_us(&self) -> u64 {
        self.max_query_time_us.load(Ordering::Relaxed)
    }
}

impl ZmqRegistryHub {
    /// Create a new ZMQ registry hub.
    pub fn new(config: RegistryHubConfig) -> Result<Self> {
        let storage = Arc::new(MokaStorage::with_lease_timeout(
            config.capacity,
            config.lease_timeout,
        ));
        let stats = Arc::new(HubStatsAtomic::new());

        Ok(Self {
            config,
            storage,
            stats,
        })
    }

    /// Handle a query message and return the response.
    fn handle_query(
        storage: &MokaStorage,
        stats: &HubStatsAtomic,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        use super::super::protocol::{encode_can_offload_response_v2, OffloadStatus};

        let msg_type = decode_message_type(data)?;

        match msg_type {
            MessageType::CanOffload => {
                let (bucket_id, hashes) = decode_can_offload(data)?;
                stats
                    .total_offload_queries
                    .fetch_add(1, Ordering::Relaxed);
                stats
                    .dedup_total_checked
                    .fetch_add(hashes.len() as u64, Ordering::Relaxed);

                // Log each queried hash
                for hash in &hashes {
                    info!(
                        target: "kvbm_distributed_registry",
                        bucket_id = bucket_id,
                        sequence_hash = %hash,
                        "HUB_CAN_OFFLOAD_QUERY: bucket={} hash={:#018x}",
                        bucket_id, hash
                    );
                }

                // Atomically claim leases for available hashes in this bucket
                let (granted, already_stored, already_leased) =
                    storage.try_claim_leases(bucket_id, &hashes);

                // Track dedup hits (stored + leased = work avoided)
                let dedup_count = (already_stored.len() + already_leased.len()) as u64;
                stats.dedup_hits.fetch_add(dedup_count, Ordering::Relaxed);

                // Build response in same order as request
                let statuses: Vec<OffloadStatus> = hashes
                    .iter()
                    .map(|hash| {
                        if granted.contains(hash) {
                            OffloadStatus::Granted
                        } else if already_leased.contains(hash) {
                            OffloadStatus::Leased
                        } else {
                            OffloadStatus::AlreadyStored
                        }
                    })
                    .collect();

                // Log with more detail at info level if there was any deduplication
                let total = hashes.len();
                let dedup_pct = if total > 0 {
                    ((already_stored.len() + already_leased.len()) as f64 / total as f64) * 100.0
                } else {
                    0.0
                };

                info!(
                    target: "kvbm_distributed_registry",
                    bucket_id = bucket_id,
                    total = total,
                    granted = granted.len(),
                    stored = already_stored.len(),
                    leased = already_leased.len(),
                    dedup_pct = format!("{:.1}", dedup_pct),
                    "HUB_CAN_OFFLOAD_RESULT: bucket={} {} blocks → {} granted, {} stored, {} leased ({:.1}% dedup)",
                    bucket_id,
                    total,
                    granted.len(),
                    already_stored.len(),
                    already_leased.len(),
                    dedup_pct
                );

                Some(encode_can_offload_response_v2(&statuses))
            }
            MessageType::MatchSequence => {
                let (bucket_id, hashes) = decode_match_sequence(data)?;
                stats
                    .total_match_queries
                    .fetch_add(1, Ordering::Relaxed);

                // Log each queried hash
                info!(
                    target: "kvbm_distributed_registry",
                    bucket_id = bucket_id,
                    num_hashes = hashes.len(),
                    first_hash = ?hashes.first().map(|h| format!("{:#018x}", h)),
                    "HUB_MATCH_QUERY: bucket={} count={} first_hash={:#018x?}",
                    bucket_id, hashes.len(), hashes.first()
                );

                let matched = storage.match_prefix(bucket_id, &hashes);

                // Log each matched hash
                for (hash, key) in &matched {
                    info!(
                        target: "kvbm_distributed_registry",
                        bucket_id = bucket_id,
                        sequence_hash = %hash,
                        object_key = %key,
                        "HUB_MATCH_HIT: bucket={} hash={:#018x} key={:#018x}",
                        bucket_id, hash, key
                    );
                }

                info!(
                    target: "kvbm_distributed_registry",
                    bucket_id = bucket_id,
                    queried = hashes.len(),
                    matched = matched.len(),
                    "HUB_MATCH_RESULT: bucket={} queried={} matched={}",
                    bucket_id, hashes.len(), matched.len()
                );

                Some(encode_match_response(&matched))
            }
            _ => {
                warn!("Unexpected message type on query socket: {:?}", msg_type);
                None
            }
        }
    }

    /// Run the query handler loop using ROUTER socket (DEALER/ROUTER pattern).
    ///
    /// Protocol:
    /// - ROUTER socket receives: [identity, query_data]
    /// - ROUTER socket sends back: [identity, response_data]
    ///
    /// The ROUTER socket automatically tracks client identities and routes
    /// responses to the correct client. This fixes the round-robin issue
    /// that occurred with PUSH/PULL pattern.
    ///
    /// This implementation uses a pipelined architecture:
    /// - Receiver task: pulls messages from socket, processes queries, sends to response channel
    /// - Sender task: pulls from response channel, sends to socket
    /// This allows query processing to happen in parallel with I/O.
    async fn run_query_handler(
        storage: Arc<MokaStorage>,
        stats: Arc<HubStatsAtomic>,
        query_addr: String,
        cancel: CancellationToken,
    ) -> Result<()> {
        use dashmap::DashSet;
        use tokio::sync::mpsc;

        let context = Context::new();

        // ROUTER socket for both receiving queries and sending responses
        let router_socket = router(&context).bind(&query_addr)?;
        // split() returns (SplitSink, SplitStream) - sink is for sending, stream is for receiving
        let (mut send_half, mut recv_half) = router_socket.split();

        info!(
            "Registry hub query handler: ROUTER on {} (pipelined)",
            query_addr
        );

        // Channel for responses (identity, response_data)
        let (tx, mut rx) = mpsc::channel::<(Vec<u8>, Vec<u8>)>(1024);

        // Track unique client IDs (thread-safe)
        let known_clients: Arc<DashSet<Vec<u8>>> = Arc::new(DashSet::new());

        // Spawn sender task
        let send_cancel = cancel.clone();
        let sender_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = send_cancel.cancelled() => {
                        break;
                    }
                    Some((identity, response)) = rx.recv() => {
                        let mut response_frames = VecDeque::new();
                        response_frames.push_back(Message::from(identity));
                        response_frames.push_back(Message::from(response));

                        if let Err(e) = send_half.send(Multipart(response_frames)).await {
                            error!("Failed to send response: {}", e);
                        }
                    }
                    else => break,
                }
            }
        });

        // Periodic stats logging
        let mut last_stats_log = std::time::Instant::now();
        let stats_interval = std::time::Duration::from_secs(10);

        // Main receive loop
        loop {
            // Check if we should log stats
            if last_stats_log.elapsed() >= stats_interval {
                last_stats_log = std::time::Instant::now();
                let (interval_queries, interval_time_us) = stats.reset_interval();
                let hub_stats = stats.to_hub_stats();

                let qps = interval_queries as f64 / stats_interval.as_secs_f64();
                let avg_interval_us = if interval_queries > 0 {
                    interval_time_us as f64 / interval_queries as f64
                } else {
                    0.0
                };

                info!(
                    "HUB_STATS: qps={:.1} | interval: {} queries, avg={:.0}µs | \
                     total: {} queries (offload={}, match={}) | \
                     avg={:.0}µs, min={}µs, max={}µs | \
                     storage={} entries, {:.1}% dedup | {} clients",
                    qps,
                    interval_queries,
                    avg_interval_us,
                    hub_stats.total_queries(),
                    hub_stats.total_offload_queries,
                    hub_stats.total_match_queries,
                    stats.avg_query_time_us(),
                    stats.min_query_time_us(),
                    stats.max_query_time_us(),
                    storage.len(),
                    hub_stats.dedup_ratio() * 100.0,
                    known_clients.len(),
                );
            }

            tokio::select! {
                _ = cancel.cancelled() => {
                    info!("Query handler shutting down");
                    break;
                }
                result = recv_half.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            let query_start = std::time::Instant::now();

                            // ROUTER receives: [identity, query_data]
                            let frames: Vec<_> = msg.iter().collect();
                            if frames.len() < 2 {
                                warn!("Query missing identity or data (got {} frames)", frames.len());
                                continue;
                            }

                            let identity = frames[0].to_vec();
                            let query_data = frames[1].to_vec();

                            // Track new clients
                            if !known_clients.contains(&identity) {
                                known_clients.insert(identity.clone());
                                stats.unique_clients.fetch_add(1, Ordering::Relaxed);
                                let identity_hex: String = identity
                                    .iter()
                                    .map(|b| format!("{:02x}", b))
                                    .collect();
                                info!("NEW_CLIENT: {} (total: {})", identity_hex, known_clients.len());
                            }

                            // Process query synchronously (fast - just hash lookups)
                            let response = Self::handle_query(&storage, &stats, &query_data)
                                .unwrap_or_default();

                            let query_time_us = query_start.elapsed().as_micros() as u64;
                            stats.record_query_time(query_time_us);

                            // Send response through channel (non-blocking)
                            if tx.send((identity, response)).await.is_err() {
                                error!("Response channel closed");
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!("Error receiving query: {}", e);
                        }
                        None => {
                            warn!("Query socket closed");
                            break;
                        }
                    }
                }
            }
        }

        // Wait for sender to finish
        drop(tx);
        let _ = sender_handle.await;

        Ok(())
    }

    /// Run the registration handler loop.
    async fn run_register_handler(
        storage: Arc<MokaStorage>,
        stats: Arc<HubStatsAtomic>,
        register_addr: String,
        cancel: CancellationToken,
    ) -> Result<()> {
        let context = Context::new();
        let mut sub_socket = subscribe(&context)
            .bind(&register_addr)?
            .subscribe(b"")?; // Subscribe to all messages

        info!(
            "Registry hub registration handler listening on {}",
            register_addr
        );

        let mut last_stats_log = std::time::Instant::now();
        let stats_interval = std::time::Duration::from_secs(30);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    info!("Registration handler shutting down");
                    break;
                }
                result = sub_socket.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            for frame in msg.iter() {
                                let data = frame.as_ref();
                                if decode_message_type(data) == Some(MessageType::Register) {
                                    if let Some((bucket_id, entries)) = decode_register(data) {
                                        let count = entries.len();
                                        let prev_total = stats.total_registered.fetch_add(count as u64, Ordering::Relaxed);

                                        // Log each registered hash
                                        for (hash, key) in &entries {
                                            info!(
                                                target: "kvbm_distributed_registry",
                                                bucket_id = bucket_id,
                                                sequence_hash = %hash,
                                                object_key = %key,
                                                "HUB_REGISTER: bucket={} hash={:#018x} key={:#018x}",
                                                bucket_id, hash, key
                                            );
                                        }

                                        // Insert entries (converts leases to permanent)
                                        storage.insert_batch(bucket_id, &entries);

                                        info!(
                                            target: "kvbm_distributed_registry",
                                            bucket_id = bucket_id,
                                            count = count,
                                            total_registered = prev_total + count as u64,
                                            storage_len = storage.len(),
                                            "HUB_REGISTER_BATCH: bucket={} {} blocks (total: {}, storage: {})",
                                            bucket_id,
                                            count,
                                            prev_total + count as u64,
                                            storage.len(),
                                        );

                                        // Periodic stats logging
                                        if last_stats_log.elapsed() > stats_interval {
                                            last_stats_log = std::time::Instant::now();
                                            let hub_stats = stats.to_hub_stats();
                                            info!(
                                                "Hub stats: {} entries, {} registered, {} queries, {:.1}% dedup rate",
                                                storage.len(),
                                                hub_stats.total_registered,
                                                hub_stats.total_queries(),
                                                hub_stats.dedup_ratio() * 100.0
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => {
                            error!("Error receiving registration: {}", e);
                        }
                        None => {
                            warn!("Registration socket closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl RegistryHub for ZmqRegistryHub {
    async fn serve(&self, cancel: CancellationToken) -> Result<()> {
        info!(
            "Starting ZMQ registry hub (capacity: {}, query: {}, register: {})",
            self.config.capacity, self.config.query_addr, self.config.register_addr
        );

        // Spawn query handler (uses ROUTER socket for bidirectional communication)
        let query_storage = self.storage.clone();
        let query_stats = self.stats.clone();
        let query_addr = self.config.query_addr.clone();
        let query_cancel = cancel.clone();
        let query_handle = tokio::spawn(async move {
            Self::run_query_handler(
                query_storage,
                query_stats,
                query_addr,
                query_cancel,
            )
            .await
        });

        // Spawn registration handler
        let register_storage = self.storage.clone();
        let register_stats = self.stats.clone();
        let register_addr = self.config.register_addr.clone();
        let register_cancel = cancel.clone();
        let register_handle = tokio::spawn(async move {
            Self::run_register_handler(
                register_storage,
                register_stats,
                register_addr,
                register_cancel,
            )
            .await
        });

        // Wait for cancellation or error
        tokio::select! {
            result = query_handle => {
                if let Err(e) = result {
                    error!("Query handler error: {}", e);
                }
            }
            result = register_handle => {
                if let Err(e) = result {
                    error!("Registration handler error: {}", e);
                }
            }
            _ = cancel.cancelled() => {
                info!("Registry hub received shutdown signal");
            }
        }

        let stats = self.stats();
        info!(
            "Registry hub shutdown. Final stats: {} entries, {} registered, {} queries \
             (offload={}, match={}), {:.1}% dedup, \
             avg={:.0}µs, min={}µs, max={}µs",
            self.len(),
            stats.total_registered,
            stats.total_queries(),
            stats.total_offload_queries,
            stats.total_match_queries,
            stats.dedup_ratio() * 100.0,
            self.stats.avg_query_time_us(),
            self.stats.min_query_time_us(),
            self.stats.max_query_time_us(),
        );

        Ok(())
    }

    fn len(&self) -> u64 {
        self.storage.len()
    }

    fn stats(&self) -> HubStats {
        self.stats.to_hub_stats()
    }

    fn capacity(&self) -> u64 {
        self.config.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_creation() {
        let config = RegistryHubConfig::with_capacity(1000);
        let hub = ZmqRegistryHub::new(config).unwrap();
        assert_eq!(hub.len(), 0);
        assert_eq!(hub.capacity(), 1000);
    }

    #[test]
    fn test_hub_stats_initial() {
        let config = RegistryHubConfig::with_capacity(1000);
        let hub = ZmqRegistryHub::new(config).unwrap();
        let stats = hub.stats();
        assert_eq!(stats.total_registered, 0);
        assert_eq!(stats.total_queries(), 0);
    }
}

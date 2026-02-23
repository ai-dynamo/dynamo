// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ-based registry hub with proper async handling.
//!
//! Uses separate tasks for queries and registrations to avoid
//! issues with tokio::select! and ZMQ streams.

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use tmq::{Context, Message, Multipart, pull, router};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use super::codec::{OffloadStatus, QueryType, RegistryCodec, ResponseType};
use super::key::RegistryKey;
use super::lease::LeaseManager;
use super::metadata::RegistryMetadata;
use super::metrics::{NoopRegistryMetricsSink, RegistryMetricsSink};
use super::storage::Storage;
use super::value::RegistryValue;
use super::zmq_transport::REQUEST_ID_SIZE;

/// Configuration for ZMQ hub.
#[derive(Clone, Debug)]
pub struct ZmqHubConfig {
    pub query_addr: String,
    pub pull_addr: String,
    pub capacity: u64,
    pub lease_ttl: Duration,
    pub lease_cleanup_interval: Duration,
}

impl ZmqHubConfig {
    pub fn new(query_addr: impl Into<String>, pull_addr: impl Into<String>) -> Self {
        Self {
            query_addr: query_addr.into(),
            pull_addr: pull_addr.into(),
            capacity: 100_000,
            lease_ttl: Duration::from_secs(30),
            lease_cleanup_interval: Duration::from_secs(5),
        }
    }

    pub fn with_ports(host: &str, query_port: u16, pull_port: u16) -> Self {
        Self::new(
            format!("tcp://{}:{}", host, query_port),
            format!("tcp://{}:{}", host, pull_port),
        )
    }

    pub fn bind_all(query_port: u16, pull_port: u16) -> Self {
        Self::with_ports("*", query_port, pull_port)
    }
}

impl Default for ZmqHubConfig {
    fn default() -> Self {
        Self::bind_all(5555, 5556)
    }
}

/// ZMQ-based registry hub.
pub struct ZmqHub<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V> + Send + Sync + 'static,
    C: RegistryCodec<K, V, M> + Send + Sync + 'static,
{
    config: ZmqHubConfig,
    storage: Arc<S>,
    codec: Arc<C>,
    lease_manager: Arc<LeaseManager<K>>,
    metrics_sink: Arc<dyn RegistryMetricsSink>,
    _phantom: PhantomData<(K, V, M)>,
}

impl<K, V, M, S, C> ZmqHub<K, V, M, S, C>
where
    K: RegistryKey,
    V: RegistryValue,
    M: RegistryMetadata,
    S: Storage<K, V> + Send + Sync + 'static,
    C: RegistryCodec<K, V, M> + Send + Sync + 'static,
{
    pub fn new(config: ZmqHubConfig, storage: S, codec: C) -> Self {
        Self::with_metrics_sink(config, storage, codec, Arc::new(NoopRegistryMetricsSink))
    }

    /// Create a hub with an explicit metrics sink plugin.
    pub fn with_metrics_sink(
        config: ZmqHubConfig,
        storage: S,
        codec: C,
        metrics_sink: Arc<dyn RegistryMetricsSink>,
    ) -> Self {
        Self {
            lease_manager: Arc::new(LeaseManager::new(config.lease_ttl)),
            config,
            storage: Arc::new(storage),
            codec: Arc::new(codec),
            metrics_sink,
            _phantom: PhantomData,
        }
    }

    /// Get reference to storage for seeding data.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Returns the configured metrics sink.
    pub fn metrics_sink(&self) -> Arc<dyn RegistryMetricsSink> {
        self.metrics_sink.clone()
    }

    /// Run the hub until cancelled.
    pub async fn serve(&self, cancel: CancellationToken) -> Result<()> {
        info!(
            query_addr = %self.config.query_addr,
            pull_addr = %self.config.pull_addr,
            "ZMQ hub starting"
        );

        // Spawn query handler on the current (main) tokio runtime.
        // This is the latency-sensitive path — must never be starved.
        let query_cancel = cancel.clone();
        let query_storage = self.storage.clone();
        let query_codec = self.codec.clone();
        let query_lease_manager = self.lease_manager.clone();
        let query_metrics_sink = self.metrics_sink.clone();
        let query_addr = self.config.query_addr.clone();

        let query_handle = tokio::spawn(async move {
            Self::run_query_handler(
                query_storage,
                query_codec,
                query_lease_manager,
                query_metrics_sink,
                query_addr,
                query_cancel,
            )
            .await
        });

        // Spawn registration handler on a DEDICATED tokio runtime.
        //
        // Registration floods (thousands of entries during inference) create a
        // tight loop in the PULL handler that starves the query handler for CPU.
        // By running PULL on its own runtime with its own OS threads, it can
        // never block query processing regardless of registration volume.
        let pull_cancel = cancel.clone();
        let pull_storage = self.storage.clone();
        let pull_codec = self.codec.clone();
        let pull_lease_manager = self.lease_manager.clone();
        let pull_metrics_sink = self.metrics_sink.clone();
        let pull_addr = self.config.pull_addr.clone();

        let pull_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("registry-pull")
            .enable_all()
            .build()
            .map_err(|e| anyhow!("Failed to create PULL runtime: {}", e))?;

        let pull_join = std::thread::spawn(move || {
            pull_runtime.block_on(async move {
                if let Err(e) = Self::run_pull_handler(
                    pull_storage,
                    pull_codec,
                    pull_lease_manager,
                    pull_metrics_sink,
                    pull_addr,
                    pull_cancel,
                )
                .await
                {
                    error!(error = %e, "Pull handler failed");
                }
            });
        });

        // Spawn periodic lease cleanup on the main runtime so expired leases
        // are observable and don't block new claimants.
        let lease_cleanup_manager = self.lease_manager.clone();
        let lease_cleanup_sink = self.metrics_sink.clone();
        let lease_cleanup_interval = self.config.lease_cleanup_interval;
        let lease_cleanup_cancel = cancel.clone();
        let lease_cleanup_handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(lease_cleanup_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = lease_cleanup_cancel.cancelled() => break,
                    _ = ticker.tick() => {
                        let expired = lease_cleanup_manager.cleanup_expired();
                        if expired > 0 {
                            lease_cleanup_sink.on_leases_expired(
                                expired,
                                lease_cleanup_manager.active_count(),
                            );
                        }
                    }
                }
            }
        });

        // Wait for query handler to finish or cancellation
        tokio::select! {
            result = query_handle => {
                if let Err(e) = result {
                    error!(error = %e, "Query handler panicked");
                }
            }
            _ = cancel.cancelled() => {
                info!("Hub received shutdown signal");
            }
        }

        let _ = lease_cleanup_handle.await;

        // Wait for PULL runtime thread to finish
        let _ = pull_join.join();

        info!(entries = self.storage.len(), "ZMQ hub stopped");
        Ok(())
    }

    /// Run the query handler (ROUTER socket).
    ///
    /// Uses a single loop: receive query → process → send response → repeat.
    /// No socket splitting, no channel, no sender task. This avoids tokio task
    /// scheduling delays that caused 20-30 second response latencies when the
    /// sender task was starved by concurrent PULL registration processing.
    async fn run_query_handler(
        storage: Arc<S>,
        codec: Arc<C>,
        lease_manager: Arc<LeaseManager<K>>,
        metrics_sink: Arc<dyn RegistryMetricsSink>,
        addr: String,
        cancel: CancellationToken,
    ) -> Result<()> {
        let context = Context::new();
        let mut router = router::router(&context)
            .bind(&addr)
            .map_err(|e| anyhow!("Failed to bind ROUTER to {}: {}", addr, e))?;

        info!(addr = %addr, "Query handler started (ROUTER)");

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("Query handler shutting down");
                    break;
                }
                result = router.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            let frames: Vec<_> = msg.iter().collect();
                            if frames.len() < 2 {
                                warn!(frames = frames.len(), "Invalid ROUTER message");
                                continue;
                            }

                            let identity = frames[0].to_vec();
                            let raw_data: &[u8] = &frames[frames.len() - 1];

                            // Extract the 4-byte request ID prefix that the client prepended.
                            // Echo it back in the response so the client can correlate
                            // responses to requests and discard stale ones.
                            if raw_data.len() < REQUEST_ID_SIZE {
                                warn!(
                                    data_len = raw_data.len(),
                                    "Query payload too short for request ID"
                                );
                                continue;
                            }
                            let request_id_bytes = &raw_data[..REQUEST_ID_SIZE];
                            let query_data = &raw_data[REQUEST_ID_SIZE..];

                            debug!(
                                identity_len = identity.len(),
                                data_len = query_data.len(),
                                request_id = u32::from_le_bytes(
                                    request_id_bytes.try_into().unwrap_or([0; 4])
                                ),
                                frames = frames.len(),
                                "Query request received"
                            );

                            let query_response = Self::handle_query(
                                &storage,
                                &codec,
                                &lease_manager,
                                metrics_sink.as_ref(),
                                Self::client_id_from_identity(&identity),
                                query_data,
                            );

                            // Prepend the request ID to the response.
                            let mut response = Vec::with_capacity(
                                REQUEST_ID_SIZE + query_response.len(),
                            );
                            response.extend_from_slice(request_id_bytes);
                            response.extend_from_slice(&query_response);

                            // Send response inline — no channel hop, no task scheduling delay.
                            let mut resp_frames = VecDeque::new();
                            resp_frames.push_back(Message::from(identity));
                            resp_frames.push_back(Message::from(response));

                            if let Err(e) = router.send(Multipart(resp_frames)).await {
                                error!(error = %e, "Failed to send response");
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "ROUTER receive error");
                        }
                        None => {
                            warn!("ROUTER socket closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Run the registration handler (PULL socket).
    async fn run_pull_handler(
        storage: Arc<S>,
        codec: Arc<C>,
        lease_manager: Arc<LeaseManager<K>>,
        metrics_sink: Arc<dyn RegistryMetricsSink>,
        addr: String,
        cancel: CancellationToken,
    ) -> Result<()> {
        let context = Context::new();
        let mut puller = pull::pull(&context)
            .bind(&addr)
            .map_err(|e| anyhow!("Failed to bind PULL to {}: {}", addr, e))?;

        info!(addr = %addr, "Pull handler started (PULL)");

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("Pull handler shutting down");
                    break;
                }
                result = puller.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            let frame_count = msg.len();
                            let total_bytes: usize = msg.iter().map(|f| f.len()).sum();
                            debug!(
                                frames = frame_count,
                                total_bytes = total_bytes,
                                "Registration request received"
                            );
                            for frame in msg.iter() {
                                Self::handle_registration(
                                    &storage,
                                    &codec,
                                    &lease_manager,
                                    metrics_sink.as_ref(),
                                    frame.as_ref(),
                                );
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "PULL receive error");
                        }
                        None => {
                            warn!("PULL socket closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a query and return response bytes.
    fn client_id_from_identity(identity: &[u8]) -> u64 {
        if identity.len() >= 8 {
            u64::from_le_bytes(identity[0..8].try_into().unwrap_or([0; 8]))
        } else {
            let mut hash = 0u64;
            for (i, b) in identity.iter().enumerate() {
                hash ^= (*b as u64) << ((i % 8) * 8);
            }
            hash
        }
    }

    /// Handle a query and return response bytes.
    fn handle_query(
        storage: &S,
        codec: &C,
        lease_manager: &LeaseManager<K>,
        metrics_sink: &dyn RegistryMetricsSink,
        client_id: u64,
        data: &[u8],
    ) -> Vec<u8> {
        let started = Instant::now();
        let mut response = Vec::new();

        let Some(query) = codec.decode_query(data) else {
            warn!("Failed to decode query");
            metrics_sink.on_query_decode_failure();
            return response;
        };

        match query {
            QueryType::CanOffload(keys) => {
                let statuses: Vec<_> = keys
                    .iter()
                    .map(|k| {
                        if storage.contains(k) {
                            OffloadStatus::AlreadyStored
                        } else if lease_manager.try_acquire(*k, client_id).is_some() {
                            OffloadStatus::Granted
                        } else {
                            OffloadStatus::Leased
                        }
                    })
                    .collect();

                let granted_count = statuses
                    .iter()
                    .filter(|s| **s == OffloadStatus::Granted)
                    .count();
                let already_stored_count = statuses
                    .iter()
                    .filter(|s| **s == OffloadStatus::AlreadyStored)
                    .count();
                let leased_count = statuses
                    .iter()
                    .filter(|s| **s == OffloadStatus::Leased)
                    .count();

                debug!(
                    query_type = "CanOffload",
                    keys = ?keys,
                    keys_count = keys.len(),
                    granted = granted_count,
                    already_stored = already_stored_count,
                    leased = leased_count,
                    storage_size = storage.len(),
                    "Query processed"
                );

                if let Err(e) =
                    codec.encode_response(&ResponseType::CanOffload(statuses), &mut response)
                {
                    warn!("Failed to encode response: {}", e);
                }
                metrics_sink.on_query_processed("can_offload", keys.len(), started.elapsed());
                metrics_sink.on_can_offload_result(
                    granted_count,
                    already_stored_count,
                    leased_count,
                    lease_manager.active_count(),
                );
            }
            QueryType::Match(keys) => {
                let entries: Vec<_> = keys
                    .iter()
                    .filter_map(|k| storage.get(k).map(|v| (*k, v, M::default())))
                    .collect();
                let hits = entries.len();
                let misses = keys.len().saturating_sub(hits);

                debug!(
                    query_type = "Match",
                    keys = ?keys,
                    requested = keys.len(),
                    matched = entries.len(),
                    matched_keys = ?entries.iter().map(|(k, _, _)| k).collect::<Vec<_>>(),
                    miss = keys.len() - entries.len(),
                    storage_size = storage.len(),
                    "Query processed"
                );

                if let Err(e) = codec.encode_response(&ResponseType::Match(entries), &mut response)
                {
                    warn!("Failed to encode response: {}", e);
                }
                metrics_sink.on_match_result(hits, misses);
                metrics_sink.on_query_processed("match", keys.len(), started.elapsed());
            }
            QueryType::Remove(keys) => {
                let mut removed_count = 0usize;
                for key in &keys {
                    if storage.remove(key).is_some() {
                        removed_count += 1;
                    }
                }

                debug!(
                    query_type = "Remove",
                    keys = ?keys,
                    requested = keys.len(),
                    removed = removed_count,
                    storage_size = storage.len(),
                    "Query processed"
                );

                if let Err(e) =
                    codec.encode_response(&ResponseType::Remove(removed_count), &mut response)
                {
                    warn!("Failed to encode response: {}", e);
                }
                metrics_sink.on_record_count_change(storage.len());
                metrics_sink.on_query_processed("remove", keys.len(), started.elapsed());
            }
            QueryType::Touch(keys) => {
                // Touch is a no-op for now - just acknowledge
                // Future: could update access timestamps for LRU eviction
                let touched_count = keys.len();
                debug!(
                    query_type = "Touch",
                    keys = ?keys,
                    keys_count = touched_count,
                    storage_size = storage.len(),
                    "Query processed (no-op)"
                );

                if let Err(e) =
                    codec.encode_response(&ResponseType::Touch(touched_count), &mut response)
                {
                    warn!("Failed to encode response: {}", e);
                }
                metrics_sink.on_query_processed("touch", keys.len(), started.elapsed());
            }
        }

        response
    }

    /// Handle a registration message.
    fn handle_registration(
        storage: &S,
        codec: &C,
        lease_manager: &LeaseManager<K>,
        metrics_sink: &dyn RegistryMetricsSink,
        data: &[u8],
    ) {
        let started = Instant::now();
        let Some(entries) = codec.decode_register(data) else {
            warn!(data_len = data.len(), "Failed to decode registration");
            return;
        };

        let count = entries.len();
        let prev_total = storage.len();

        let mut released_count = 0usize;
        let batch: Vec<(K, V)> = entries
            .into_iter()
            .map(|(key, value, _metadata)| {
                if lease_manager.release(&key) {
                    released_count += 1;
                }
                (key, value)
            })
            .collect();
        storage.insert_batch(batch);
        let new_total = storage.len();

        // Only log the batch summary, not individual entries. Per-entry logging
        // at debug level caused 20-30 second stalls during registration floods
        // by monopolizing the tokio runtime with I/O.
        debug!(
            entries_count = count,
            prev_total = prev_total,
            new_total = new_total,
            added = new_total - prev_total,
            data_bytes = data.len(),
            "Registration batch processed"
        );
        metrics_sink.on_register_batch(count, started.elapsed(), new_total);
        if released_count > 0 {
            metrics_sink.on_leases_released(released_count, lease_manager.active_count());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::distributed::registry::core::{
        BinaryCodec, HashMapStorage, NoMetadata,
    };
    use std::time::Duration;

    #[test]
    fn test_config_builder() {
        let config = ZmqHubConfig::bind_all(6000, 6001);
        assert_eq!(config.query_addr, "tcp://*:6000");
        assert_eq!(config.pull_addr, "tcp://*:6001");
    }

    #[tokio::test]
    #[ignore] // Requires ZMQ, run with: cargo test -- --ignored
    async fn test_zmq_hub_e2e() {
        use crate::block_manager::distributed::registry::core::{RegistryTransport, ZmqTransport};

        let port_base = 16555;
        let config = ZmqHubConfig::bind_all(port_base, port_base + 1);

        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        storage.insert(1, 100);
        storage.insert(2, 200);

        let hub = ZmqHub::new(config, storage, BinaryCodec::<u64, u64, NoMetadata>::new());
        let cancel = CancellationToken::new();

        // Start hub
        let hub_cancel = cancel.clone();
        let hub_handle = tokio::spawn(async move { hub.serve(hub_cancel).await });

        // Wait for hub to bind
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Connect client
        let transport = ZmqTransport::connect_to("localhost", port_base, port_base + 1)
            .expect("Failed to connect");

        // Test query
        let codec = BinaryCodec::<u64, u64, NoMetadata>::new();
        let mut buf = Vec::new();
        codec
            .encode_query(&QueryType::CanOffload(vec![1, 3]), &mut buf)
            .unwrap();

        let response = transport.request(&buf).await.expect("Request failed");
        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&response).unwrap();

        match decoded {
            ResponseType::CanOffload(statuses) => {
                assert_eq!(statuses[0], OffloadStatus::AlreadyStored);
                assert_eq!(statuses[1], OffloadStatus::Granted);
            }
            _ => panic!("Wrong response type"),
        }

        // Shutdown
        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(1), hub_handle).await;
    }
}

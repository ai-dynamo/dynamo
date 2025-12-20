// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ-based registry client implementation.
//!
//! Uses ZMQ sockets for communication:
//! - DEALER socket for queries (bidirectional with hub's ROUTER)
//! - PUB socket for registrations (PUB/SUB pattern)
//!
//! The DEALER/ROUTER pattern ensures responses are routed to the correct client
//! even when multiple clients are connected to the same hub.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use tmq::{dealer::dealer, publish::publish, Context, Message, Multipart};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, warn};

use super::super::config::RegistryClientConfig;
use super::super::protocol::{
    bucket_id_from_name, decode_can_offload_response_v2, decode_match_response, encode_can_offload,
    encode_match_sequence, encode_register, BucketId, OffloadStatus, SequenceHash,
};
use super::super::storage::MokaStorage;
use super::super::traits::DistributedRegistry;
use super::super::types::{ObjectKey, OffloadResult};

/// ZMQ-based registry client.
///
/// Connects to a registry hub for querying and registering entries.
/// Supports optional local caching to reduce network round-trips.
///
/// # Protocol
///
/// Queries use DEALER/ROUTER pattern for async request-response:
/// - Client DEALER sends: [query_data] (ZMQ adds identity automatically)
/// - Hub ROUTER receives: [identity, query_data]
/// - Hub ROUTER sends: [identity, response_data]
/// - Client DEALER receives: [response_data] (ZMQ strips identity)
///
/// This ensures responses are routed to the correct client even with
/// multiple clients connected to the same hub.
///
/// Registrations use PUB/SUB (fire-and-forget):
/// - Client PUB: [register_message]
///
/// # Example
/// ```ignore
/// let config = RegistryClientConfig::connect_to("leader", 5555, 5557);
/// let client = ZmqRegistryClient::connect(config).await?;
///
/// // Register hashes after storing to object
/// client.register("my-bucket", &[hash1, hash2]).await?;
///
/// // Check what can be offloaded
/// let result = client.can_offload("my-bucket", &hashes).await?;
/// ```
pub struct ZmqRegistryClient {
    config: RegistryClientConfig,
    // Unique client ID for logging/debugging
    client_id: u64,
    // Pre-computed bucket ID from config.bucket_name
    bucket_id: BucketId,
    // DEALER socket for bidirectional query/response with hub's ROUTER
    dealer_socket: TokioMutex<tmq::dealer::Dealer>,
    // PUB socket for registrations
    pub_socket: TokioMutex<tmq::publish::Publish>,
    // Pending registration batch: (bucket_id, hash, key)
    pending_batch: Mutex<Vec<(BucketId, SequenceHash, ObjectKey)>>,
    // Last batch flush time
    last_flush: Mutex<Instant>,
    // Optional local cache
    local_cache: Option<MokaStorage>,
}

// Counter for generating unique client IDs (for logging)
static CLIENT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

impl ZmqRegistryClient {
    /// Connect to a registry hub.
    pub async fn connect(config: RegistryClientConfig) -> Result<Self> {
        let context = Context::new();

        // Generate unique client ID (for logging/debugging)
        let client_id = CLIENT_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        // Pre-compute bucket ID from bucket name
        let bucket_id = bucket_id_from_name(&config.bucket_name);

        // Create DEALER socket for bidirectional communication with hub's ROUTER
        // DEALER/ROUTER pattern ensures responses go to the correct client
        let dealer_socket = dealer(&context).connect(&config.hub_query_addr)?;

        // Create PUB socket for registrations
        let pub_socket = publish(&context).connect(&config.hub_register_addr)?;

        // Create optional local cache
        let local_cache = if config.local_cache_capacity > 0 {
            Some(MokaStorage::new(config.local_cache_capacity))
        } else {
            None
        };

        debug!(
            "ZmqRegistryClient[{}] connected: query={} (DEALER), register={}, bucket={}",
            client_id, config.hub_query_addr, config.hub_register_addr, config.bucket_name
        );

        Ok(Self {
            config,
            client_id,
            bucket_id,
            dealer_socket: TokioMutex::new(dealer_socket),
            pub_socket: TokioMutex::new(pub_socket),
            pending_batch: Mutex::new(Vec::new()),
            last_flush: Mutex::new(Instant::now()),
            local_cache,
        })
    }

    /// Get the configured bucket name.
    pub fn bucket_name(&self) -> &str {
        &self.config.bucket_name
    }

    /// Get the pre-computed bucket ID.
    pub fn bucket_id(&self) -> BucketId {
        self.bucket_id
    }

    /// Send a query and wait for response.
    ///
    /// Uses DEALER/ROUTER pattern - DEALER sends [query_data] and receives [response_data].
    /// ZMQ handles identity routing automatically, so responses always go to the correct client.
    async fn query(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        tracing::debug!(
            client_id = self.client_id,
            data_len = data.len(),
            "Sending query to registry hub"
        );

        let start = std::time::Instant::now();

        // Lock the DEALER socket for the entire request-response cycle
        // This ensures our response comes back to us (not interleaved with other requests)
        let mut socket = self.dealer_socket.lock().await;

        // Send query - DEALER adds identity automatically
        let mut msg = VecDeque::new();
        msg.push_back(Message::from(data));
        socket.send(Multipart(msg)).await?;
        tracing::debug!(client_id = self.client_id, "Query sent, waiting for response");

        // Wait for response with timeout
        // DEALER receives just the response data (identity is stripped by ZMQ)
        let response = tokio::time::timeout(self.config.request_timeout, async {
            match socket.next().await {
                Some(Ok(msg)) => {
                    let frames: Vec<_> = msg.iter().collect();
                    if frames.is_empty() {
                        return Err(anyhow!("Empty response from hub"));
                    }
                    tracing::debug!(
                        client_id = self.client_id,
                        response_len = frames[0].len(),
                        elapsed_ms = start.elapsed().as_millis(),
                        "Received response"
                    );
                    Ok(frames[0].to_vec())
                }
                Some(Err(e)) => Err(anyhow!("Receive error: {}", e)),
                None => Err(anyhow!("Socket closed")),
            }
        })
        .await
        .map_err(|_| {
            tracing::warn!(
                client_id = self.client_id,
                elapsed_ms = start.elapsed().as_millis(),
                timeout_ms = self.config.request_timeout.as_millis(),
                "Query timed out waiting for response"
            );
            anyhow!("Request timed out after {}ms", start.elapsed().as_millis())
        })??;

        Ok(response)
    }

    /// Check if auto-flush is needed and flush if so.
    async fn maybe_auto_flush(&self) -> Result<()> {
        let should_flush = {
            let batch = self.pending_batch.lock();
            let last_flush = self.last_flush.lock();
            batch.len() >= self.config.batch_size
                || (!batch.is_empty() && last_flush.elapsed() >= self.config.batch_timeout)
        };

        if should_flush {
            self.flush().await?;
        }

        Ok(())
    }

    /// Add entries to local cache if enabled.
    fn cache_entries(&self, bucket_id: BucketId, entries: &[(SequenceHash, ObjectKey)]) {
        if let Some(cache) = &self.local_cache {
            for (hash, key) in entries {
                cache.insert(bucket_id, *hash, *key);
            }
        }
    }

    /// Check local cache for hash.
    fn check_cache(&self, bucket_id: BucketId, hash: SequenceHash) -> Option<ObjectKey> {
        self.local_cache.as_ref()?.get(bucket_id, hash)
    }
}

#[async_trait]
impl DistributedRegistry for ZmqRegistryClient {
    async fn register(&self, bucket_name: &str, sequence_hashes: &[SequenceHash]) -> Result<()> {
        // Use hash as key (common case for object storage)
        let entries: Vec<_> = sequence_hashes.iter().map(|&h| (h, h)).collect();
        self.register_with_keys(bucket_name, &entries).await
    }

    async fn register_with_keys(
        &self,
        bucket_name: &str,
        entries: &[(SequenceHash, ObjectKey)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let bucket_id = bucket_id_from_name(bucket_name);

        // Log each registration
        for (hash, key) in entries {
            tracing::debug!(
                target: "kvbm_distributed_registry",
                bucket = %bucket_name,
                sequence_hash = %hash,
                object_key = %key,
                "REGISTER: bucket={} hash={:#018x} key={:#018x}",
                bucket_name, hash, key
            );
        }

        // Add to local cache
        self.cache_entries(bucket_id, entries);

        // Add to pending batch
        {
            let mut batch = self.pending_batch.lock();
            for (hash, key) in entries {
                batch.push((bucket_id, *hash, *key));
            }
        }

        // Check if we should auto-flush
        self.maybe_auto_flush().await?;

        Ok(())
    }

    async fn can_offload(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<OffloadResult> {
        if hashes.is_empty() {
            return Ok(OffloadResult::new());
        }

        let bucket_id = bucket_id_from_name(bucket_name);

        // Log query
        tracing::debug!(
            target: "kvbm_distributed_registry",
            bucket = %bucket_name,
            num_hashes = hashes.len(),
            first_hash = ?hashes.first().map(|h| format!("{:#018x}", h)),
            "CAN_OFFLOAD_QUERY: bucket={} count={}",
            bucket_name, hashes.len()
        );

        // Check local cache first for already stored hashes
        // Note: We can't use local cache to skip the hub query because we need
        // to claim leases atomically. But we can use it to pre-populate results.
        let (cached_exists, need_check): (Vec<_>, Vec<_>) = if self.local_cache.is_some() {
            let mut cached = Vec::new();
            let mut check = Vec::new();
            for &hash in hashes {
                if self.check_cache(bucket_id, hash).is_some() {
                    cached.push(hash);
                } else {
                    check.push(hash);
                }
            }
            (cached, check)
        } else {
            (Vec::new(), hashes.to_vec())
        };

        // Query hub for uncached hashes to claim leases
        let mut result = OffloadResult::with_capacity(hashes.len(), 0);
        result.already_stored = cached_exists;

        if !need_check.is_empty() {
            let request = encode_can_offload(bucket_id, &need_check);
            let response = self.query(request).await?;

            let statuses = decode_can_offload_response_v2(&response)
                .ok_or_else(|| anyhow!("Invalid can_offload response"))?;

            for (hash, status) in need_check.iter().zip(statuses.iter()) {
                match status {
                    OffloadStatus::Granted => {
                        result.can_offload.push(*hash);
                    }
                    OffloadStatus::AlreadyStored => {
                        result.already_stored.push(*hash);
                        // Cache the existence (hash == key for object storage)
                        self.cache_entries(bucket_id, &[(*hash, *hash)]);
                    }
                    OffloadStatus::Leased => {
                        result.leased.push(*hash);
                        // Don't cache leased - it's temporary
                    }
                }
            }
        }

        tracing::debug!(
            target: "kvbm_distributed_registry",
            bucket = %bucket_name,
            granted = result.can_offload.len(),
            stored = result.already_stored.len(),
            leased = result.leased.len(),
            "CAN_OFFLOAD_RESULT: bucket={} granted={} stored={} leased={}",
            bucket_name, result.can_offload.len(), result.already_stored.len(), result.leased.len()
        );

        Ok(result)
    }

    async fn match_sequence_hashes(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<Vec<(SequenceHash, ObjectKey)>> {
        if hashes.is_empty() {
            return Ok(Vec::new());
        }

        let bucket_id = bucket_id_from_name(bucket_name);

        // Log the query
        tracing::debug!(
            target: "kvbm_distributed_registry",
            bucket = %bucket_name,
            num_hashes = hashes.len(),
            first_hash = ?hashes.first().map(|h| format!("{:#018x}", h)),
            "MATCH_QUERY: bucket={} count={} first_hash={:#018x?}",
            bucket_name, hashes.len(), hashes.first()
        );

        // Check local cache for prefix
        if let Some(cache) = &self.local_cache {
            let cached_prefix: Vec<_> = hashes
                .iter()
                .map_while(|&hash| cache.get(bucket_id, hash).map(|key| (hash, key)))
                .collect();

            // If we have a full prefix match in cache, return it
            // Otherwise, we need to query the hub to be sure
            if cached_prefix.len() == hashes.len() {
                tracing::info!(
                    target: "kvbm_distributed_registry",
                    bucket = %bucket_name,
                    matched = cached_prefix.len(),
                    source = "cache",
                    "MATCH_RESULT: bucket={} matched={} (cache hit)",
                    bucket_name, cached_prefix.len()
                );
                return Ok(cached_prefix);
            }
        }

        // Query hub
        let request = encode_match_sequence(bucket_id, hashes);
        let response = self.query(request).await?;

        let matched = decode_match_response(&response)
            .ok_or_else(|| anyhow!("Invalid match_sequence response"))?;

        // Log each matched entry
        for (hash, key) in &matched {
            tracing::debug!(
                target: "kvbm_distributed_registry",
                bucket = %bucket_name,
                sequence_hash = %hash,
                object_key = %key,
                "MATCH_HIT: bucket={} hash={:#018x} key={:#018x}",
                bucket_name, hash, key
            );
        }

        // Cache results
        self.cache_entries(bucket_id, &matched);

        tracing::debug!(
            target: "kvbm_distributed_registry",
            bucket = %bucket_name,
            matched = matched.len(),
            source = "hub",
            "MATCH_RESULT: bucket={} matched={} (hub query)",
            bucket_name, matched.len()
        );

        Ok(matched)
    }

    async fn flush(&self) -> Result<()> {
        let entries = {
            let mut batch = self.pending_batch.lock();
            let mut last_flush = self.last_flush.lock();
            *last_flush = Instant::now();
            std::mem::take(&mut *batch)
        };

        if entries.is_empty() {
            return Ok(());
        }

        // Group entries by bucket_id for efficient encoding
        use std::collections::HashMap;
        let mut by_bucket: HashMap<BucketId, Vec<(SequenceHash, ObjectKey)>> = HashMap::new();
        for (bucket_id, hash, key) in entries {
            by_bucket.entry(bucket_id).or_default().push((hash, key));
        }

        let mut socket = self.pub_socket.lock().await;

        for (bucket_id, bucket_entries) in by_bucket {
            let encoded = encode_register(bucket_id, &bucket_entries);
            let mut msg = VecDeque::new();
            msg.push_back(Message::from(encoded));
            socket.send(Multipart(msg)).await?;
            debug!(
                "Flushed {} registration entries for bucket {}",
                bucket_entries.len(),
                bucket_id
            );
        }

        Ok(())
    }
}

// ============================================================================
// CONVENIENCE METHODS (use configured bucket)
// ============================================================================

impl ZmqRegistryClient {
    /// Match sequence hashes using the configured bucket.
    ///
    /// This is a convenience method that uses the bucket from `DYN_KVBM_OBJECT_BUCKET`.
    pub async fn match_hashes(&self, hashes: &[SequenceHash]) -> Result<Vec<(SequenceHash, ObjectKey)>> {
        self.match_sequence_hashes(&self.config.bucket_name, hashes).await
    }

    /// Check what can be offloaded using the configured bucket.
    ///
    /// This is a convenience method that uses the bucket from `DYN_KVBM_OBJECT_BUCKET`.
    pub async fn check_can_offload(&self, hashes: &[SequenceHash]) -> Result<OffloadResult> {
        self.can_offload(&self.config.bucket_name, hashes).await
    }

    /// Register entries using the configured bucket.
    ///
    /// This is a convenience method that uses the bucket from `DYN_KVBM_OBJECT_BUCKET`.
    pub async fn register_hashes(&self, hashes: &[SequenceHash]) -> Result<()> {
        self.register(&self.config.bucket_name, hashes).await
    }

    /// Register entries with keys using the configured bucket.
    pub async fn register_hashes_with_keys(&self, entries: &[(SequenceHash, ObjectKey)]) -> Result<()> {
        self.register_with_keys(&self.config.bucket_name, entries).await
    }
}

// ============================================================================
// BLOCKING METHODS (for use from sync contexts like vLLM connector)
// ============================================================================

impl ZmqRegistryClient {
    /// Blocking version of `match_hashes` for sync contexts.
    ///
    /// Uses the configured bucket from `DYN_KVBM_OBJECT_BUCKET`.
    ///
    /// This method can be called from synchronous code. It detects whether
    /// we're inside a tokio runtime and handles accordingly:
    /// - Inside runtime: uses `block_in_place` to avoid blocking the executor
    /// - Outside runtime: creates a temporary runtime
    ///
    /// # Safety
    /// This is safe to call from sync code that may or may not be inside
    /// a tokio runtime. However, avoid calling from single-threaded runtimes.
    pub fn match_hashes_blocking(
        &self,
        hashes: &[SequenceHash],
    ) -> Result<Vec<(SequenceHash, ObjectKey)>> {
        self.match_sequence_hashes_blocking(&self.config.bucket_name, hashes)
    }

    /// Blocking version of `match_sequence_hashes` for sync contexts.
    ///
    /// This method can be called from synchronous code. It detects whether
    /// we're inside a tokio runtime and handles accordingly:
    /// - Inside runtime: uses `block_in_place` to avoid blocking the executor
    /// - Outside runtime: creates a temporary runtime
    ///
    /// # Safety
    /// This is safe to call from sync code that may or may not be inside
    /// a tokio runtime. However, avoid calling from single-threaded runtimes.
    pub fn match_sequence_hashes_blocking(
        &self,
        bucket_name: &str,
        hashes: &[SequenceHash],
    ) -> Result<Vec<(SequenceHash, ObjectKey)>> {
        // Try to get a handle to an existing runtime
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // We're inside a tokio runtime - use block_in_place
            // This yields the current task's timeslice while blocking
            tokio::task::block_in_place(|| {
                handle.block_on(self.match_sequence_hashes(bucket_name, hashes))
            })
        } else {
            // Not in a runtime - create a temporary one
            // This is less efficient but necessary for pure sync contexts
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(self.match_sequence_hashes(bucket_name, hashes))
        }
    }
}

impl Drop for ZmqRegistryClient {
    fn drop(&mut self) {
        // Try to flush any pending registrations
        let batch = self.pending_batch.lock();
        if !batch.is_empty() {
            warn!(
                "ZmqRegistryClient dropped with {} pending registrations",
                batch.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require a running hub.
    // These are unit tests for the client logic.

    #[test]
    fn test_client_config_builder() {
        let config = RegistryClientConfig::connect_to("leader", 5555, 5557)
            .with_local_cache(10_000)
            .with_batch_size(50);

        assert_eq!(config.hub_query_addr, "tcp://leader:5555");
        assert_eq!(config.hub_register_addr, "tcp://leader:5557");
        assert_eq!(config.local_cache_capacity, 10_000);
        assert_eq!(config.batch_size, 50);
    }
}

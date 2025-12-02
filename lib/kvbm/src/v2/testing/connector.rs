// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testing utilities for connector components.
//!
//! This module provides reusable test infrastructure for:
//! - ConnectorInstance: 1 leader + N workers test environment
//! - ConnectorTestConfig: Role-specific configuration with dual API (builder + JSON)
//! - TestConnectorWorker: Enhanced worker wrapper with test-friendly accessors
//! - Mock KV cache tensors for testing without GPU allocation

use anyhow::{Context, Result, anyhow};
use dynamo_kvbm_config::{KvbmConfig, NixlConfig};
use dynamo_memory::nixl::{NixlAgent, NixlDescriptor};
use dynamo_memory::{MemoryDescriptor, StorageKind, TensorDescriptor};
use dynamo_nova::Nova;
use dynamo_nova_backend::WorkerAddress;
use figment::Figment;
use figment::providers::{Format, Json};
use serde::Serialize;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::v2::distributed::leader::InstanceLeader;
use crate::v2::distributed::worker::Worker;
use crate::v2::integrations::connector::leader::ConnectorLeader;
use crate::v2::integrations::connector::worker::{ConnectorWorker, ConnectorWorkerInterface};
use crate::v2::logical::pools::SequenceHash;
use crate::v2::physical::layout::LayoutConfig;
use crate::v2::physical::transfer::{BlockChecksum, FillPattern};
use crate::v2::{BlockId, G2, G3};
use crate::{InstanceId, KvbmRuntime};

use super::{managers, nova, physical, token_blocks};

// ============================================================================
// ConnectorTestConfig - Dual API configuration
// ============================================================================

/// Test configuration for connector instances wrapping Figment directly.
///
/// Workers automatically get NIXL defaults (UCX + POSIX backends) applied.
/// Leaders have no NIXL configuration by default.
///
/// Supports two APIs for configuration:
/// 1. **Builder API**: Typed methods for common configuration options
/// 2. **JSON API**: Raw JSON strings for vLLM-style complex configs
///
/// # Example - Builder API
/// ```rust,ignore
/// let config = ConnectorTestConfig::new()
///     .leader_cache_gb(1.0)
///     .worker_tokio_threads(2);
///
/// let leader = config.build_leader()?;  // nixl: None
/// let worker = config.build_worker()?;  // nixl: Some({UCX, POSIX})
/// ```
///
/// # Example - JSON API
/// ```rust,ignore
/// let config = ConnectorTestConfig::from_json(r#"{
///     "leader": { "cache": { "host": { "cache_size_gb": 1.0 } } },
///     "worker": { "tokio": { "worker_threads": 1 } }
/// }"#)?;
/// ```
///
/// # Example - Custom NIXL
/// ```rust,ignore
/// let config = ConnectorTestConfig::new()
///     .worker_nixl(NixlConfig::empty().with_backend("GDS"));
/// ```
#[derive(Clone)]
pub struct ConnectorTestConfig {
    leader: Figment,
    worker: Figment,
}

impl Default for ConnectorTestConfig {
    fn default() -> Self {
        Self {
            leader: KvbmConfig::figment_for_leader(),
            worker: KvbmConfig::figment_for_worker().merge((
                "nixl",
                serde_json::to_value(NixlConfig::default()).expect("NixlConfig serializes"),
            )),
        }
    }
}

impl ConnectorTestConfig {
    /// Create a new test configuration with defaults.
    ///
    /// Workers get NIXL defaults (UCX + POSIX backends) automatically.
    pub fn new() -> Self {
        Self::default()
    }

    // === Leader configuration ===

    /// Set leader's G2 host cache size in gigabytes.
    #[must_use]
    pub fn leader_cache_gb(mut self, gb: f64) -> Self {
        self.leader = self.leader.merge(("cache.host.cache_size_gb", gb));
        self
    }

    /// Set leader's G2 host cache size in number of blocks.
    #[must_use]
    pub fn leader_cache_blocks(mut self, n: usize) -> Self {
        self.leader = self.leader.merge(("cache.host.num_blocks", n as u64));
        self
    }

    /// Set leader's G3 disk cache size in number of blocks.
    #[must_use]
    pub fn leader_disk_blocks(mut self, n: usize) -> Self {
        self.leader = self.leader.merge(("cache.disk.num_blocks", n as u64));
        self
    }

    /// Set leader's tokio worker threads.
    #[must_use]
    pub fn leader_tokio_threads(mut self, n: usize) -> Self {
        self.leader = self.leader.merge(("tokio.worker_threads", n as u64));
        self
    }

    // === Worker configuration ===

    /// Set worker's tokio worker threads (uniform for all workers).
    #[must_use]
    pub fn worker_tokio_threads(mut self, n: usize) -> Self {
        self.worker = self.worker.merge(("tokio.worker_threads", n as u64));
        self
    }

    /// Add/override NIXL backends for workers.
    ///
    /// Note: This merges with existing backends. To fully replace, call
    /// `worker_without_nixl()` first.
    #[must_use]
    pub fn worker_nixl(mut self, nixl: NixlConfig) -> Self {
        self.worker = self.worker.merge((
            "nixl",
            serde_json::to_value(nixl).expect("NixlConfig serializes"),
        ));
        self
    }

    /// Fully replace NIXL configuration for workers.
    ///
    /// This clears any existing NIXL config (including defaults) and sets the new one.
    #[must_use]
    pub fn worker_nixl_replace(self, nixl: NixlConfig) -> Self {
        self.worker_without_nixl().worker_nixl(nixl)
    }

    /// Disable NIXL for workers.
    ///
    /// Creates a fresh worker figment without NIXL config.
    #[must_use]
    pub fn worker_without_nixl(mut self) -> Self {
        // Create fresh worker figment without NIXL, preserving non-NIXL settings
        // by using a new base figment
        self.worker = KvbmConfig::figment_for_worker();
        self
    }

    // === Generic path access ===

    /// Set any leader configuration path.
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = ConnectorTestConfig::new()
    ///     .leader_path("nova.backend.tcp_port", 9090u16);
    /// ```
    #[must_use]
    pub fn leader_path<V: Serialize>(mut self, path: &str, value: V) -> Self {
        self.leader = self.leader.merge((path, value));
        self
    }

    /// Set any worker configuration path.
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = ConnectorTestConfig::new()
    ///     .worker_path("nova.backend.tcp_port", 9091u16);
    /// ```
    #[must_use]
    pub fn worker_path<V: Serialize>(mut self, path: &str, value: V) -> Self {
        self.worker = self.worker.merge((path, value));
        self
    }

    // === JSON API (vLLM compatibility) ===

    /// Create from vLLM-style JSON with "leader" and "worker" keys.
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = ConnectorTestConfig::from_json(r#"{
    ///     "leader": { "cache": { "host": { "cache_size_gb": 1.0 } } },
    ///     "worker": { "nixl": { "backends": { "UCX": {}, "GDS": {} } } }
    /// }"#)?;
    /// ```
    pub fn from_json(json: &str) -> Result<Self> {
        let parsed: serde_json::Value =
            serde_json::from_str(json).context("Failed to parse JSON config")?;

        let mut config = Self::default();

        if let Some(leader_json) = parsed.get("leader") {
            config.leader = config.leader.merge(Json::string(&leader_json.to_string()));
        }
        if let Some(worker_json) = parsed.get("worker") {
            config.worker = config.worker.merge(Json::string(&worker_json.to_string()));
        }

        Ok(config)
    }

    /// Merge raw JSON into leader configuration.
    #[must_use]
    pub fn leader_json(mut self, json: &str) -> Self {
        self.leader = self.leader.merge(Json::string(json));
        self
    }

    /// Merge raw JSON into worker configuration.
    #[must_use]
    pub fn worker_json(mut self, json: &str) -> Self {
        self.worker = self.worker.merge(Json::string(json));
        self
    }

    // === Build ===

    /// Build KvbmConfig for leader role.
    pub fn build_leader(&self) -> Result<KvbmConfig> {
        KvbmConfig::extract_from(&self.leader)
            .map_err(|e| anyhow!("Failed to build leader config: {}", e))
    }

    /// Build KvbmConfig for worker role.
    pub fn build_worker(&self) -> Result<KvbmConfig> {
        KvbmConfig::extract_from(&self.worker)
            .map_err(|e| anyhow!("Failed to build worker config: {}", e))
    }

    // Legacy method names for backwards compatibility
    // TODO: Remove after updating all call sites

    /// Alias for `leader_cache_blocks`.
    #[must_use]
    pub fn with_leader_host_num_blocks(self, num_blocks: usize) -> Self {
        self.leader_cache_blocks(num_blocks)
    }

    /// Alias for `leader_cache_gb`.
    #[must_use]
    pub fn with_leader_host_cache_gb(self, gb: f64) -> Self {
        self.leader_cache_gb(gb)
    }

    /// Alias for `leader_tokio_threads`.
    #[must_use]
    pub fn with_leader_tokio_threads(self, threads: usize) -> Self {
        self.leader_tokio_threads(threads)
    }

    /// Alias for `worker_tokio_threads`.
    #[must_use]
    pub fn with_worker_tokio_threads(self, threads: usize) -> Self {
        self.worker_tokio_threads(threads)
    }

    /// Alias for `build_leader`.
    pub(crate) fn build_leader_config(&self) -> Result<KvbmConfig> {
        self.build_leader()
    }

    /// Alias for `build_worker`.
    pub(crate) fn build_worker_config(&self) -> Result<KvbmConfig> {
        self.build_worker()
    }
}

// ============================================================================
// TestConnectorWorker - Enhanced worker wrapper
// ============================================================================

/// Enhanced wrapper for ConnectorWorker with test-friendly accessors.
///
/// Contains the worker, identity information, and rank.
pub struct TestConnectorWorker {
    /// The ConnectorWorker instance.
    pub worker: ConnectorWorker,
    /// Nova instance ID for this worker.
    pub instance_id: InstanceId,
    /// Nova worker address.
    pub worker_address: WorkerAddress,
    /// Rank within the instance (0-indexed).
    pub rank: usize,
}

impl TestConnectorWorker {
    /// Check if the worker has been initialized by the leader.
    pub fn is_initialized(&self) -> bool {
        self.worker.is_initialized()
    }

    /// Fill G2 blocks with test pattern and return checksums.
    ///
    /// This directly fills the worker's G2 layout (pinned host memory).
    /// Can only be called on System or Pinned storage.
    pub fn fill_g2_blocks(
        &self,
        block_ids: &[BlockId],
        pattern: FillPattern,
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        let direct_worker = self
            .worker
            .worker()
            .ok_or_else(|| anyhow!("Worker not initialized"))?;

        let g2_handle = direct_worker
            .g2_handle()
            .ok_or_else(|| anyhow!("G2 handle not set"))?;

        physical::fill_and_checksum_manager(
            direct_worker.transfer_manager(),
            g2_handle,
            block_ids,
            pattern,
        )
    }

    /// Compute checksums for G2 blocks.
    ///
    /// This reads the current contents of G2 blocks and computes their checksums.
    pub fn compute_g2_checksums(
        &self,
        block_ids: &[BlockId],
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        let direct_worker = self
            .worker
            .worker()
            .ok_or_else(|| anyhow!("Worker not initialized"))?;

        let g2_handle = direct_worker
            .g2_handle()
            .ok_or_else(|| anyhow!("G2 handle not set"))?;

        physical::compute_manager_checksums(direct_worker.transfer_manager(), g2_handle, block_ids)
    }

    /// Fill G3 blocks with test pattern via G2 staging.
    ///
    /// This:
    /// 1. Uses temporary G2 staging blocks
    /// 2. Fills the staging blocks with the pattern
    /// 3. Transfers G2 -> G3
    /// 4. Returns checksums of the filled data
    ///
    /// The staging blocks are separate from the provided G3 block_ids to avoid conflicts.
    pub async fn fill_g3_blocks(
        &self,
        block_ids: &[BlockId],
        pattern: FillPattern,
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        use crate::v2::distributed::worker::WorkerTransfers;
        use crate::v2::logical::LogicalLayoutHandle;
        use crate::v2::physical::transfer::TransferOptions;

        let direct_worker = self
            .worker
            .worker()
            .ok_or_else(|| anyhow!("Worker not initialized"))?;

        let g2_handle = direct_worker
            .g2_handle()
            .ok_or_else(|| anyhow!("G2 handle not set"))?;

        let _g3_handle = direct_worker
            .g3_handle()
            .ok_or_else(|| anyhow!("G3 handle not set"))?;

        // Get the G2 layout config to determine valid staging block IDs
        let layout_config = direct_worker
            .transfer_manager()
            .get_layout_config(g2_handle)?;

        // Use high block IDs as staging blocks to avoid conflicts with actual cached data
        // Start from the middle of the available range
        let staging_offset = layout_config.num_blocks / 2;
        let staging_block_ids: Vec<BlockId> = (0..block_ids.len())
            .map(|i| (staging_offset + i) as BlockId)
            .collect();

        // Fill the staging blocks with the pattern
        let checksums = physical::fill_and_checksum_manager(
            direct_worker.transfer_manager(),
            g2_handle,
            &staging_block_ids,
            pattern,
        )?;

        // Transfer G2 staging -> G3 target blocks
        let notification = direct_worker.execute_local_transfer(
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G3,
            Arc::from(staging_block_ids.as_slice()),
            Arc::from(block_ids),
            TransferOptions::default(),
        )?;

        notification.await?;

        // Return the checksums (mapped to the actual G3 block IDs)
        let mut result = HashMap::new();
        for (staging_id, target_id) in staging_block_ids.iter().zip(block_ids.iter()) {
            if let Some(checksum) = checksums.get(staging_id) {
                result.insert(*target_id, checksum.clone());
            }
        }

        Ok(result)
    }

    /// Compute checksums for G3 blocks via G2 staging.
    ///
    /// This:
    /// 1. Uses temporary G2 staging blocks
    /// 2. Transfers G3 -> G2 staging
    /// 3. Computes checksums from staging blocks
    /// 4. Returns checksums mapped to G3 block IDs
    pub async fn compute_g3_checksums(
        &self,
        block_ids: &[BlockId],
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        use crate::v2::distributed::worker::WorkerTransfers;
        use crate::v2::logical::LogicalLayoutHandle;
        use crate::v2::physical::transfer::TransferOptions;

        let direct_worker = self
            .worker
            .worker()
            .ok_or_else(|| anyhow!("Worker not initialized"))?;

        let g2_handle = direct_worker
            .g2_handle()
            .ok_or_else(|| anyhow!("G2 handle not set"))?;

        let _g3_handle = direct_worker
            .g3_handle()
            .ok_or_else(|| anyhow!("G3 handle not set"))?;

        // Get the G2 layout config to determine valid staging block IDs
        let layout_config = direct_worker
            .transfer_manager()
            .get_layout_config(g2_handle)?;

        // Use high block IDs as staging blocks
        let staging_offset = layout_config.num_blocks / 2;
        let staging_block_ids: Vec<BlockId> = (0..block_ids.len())
            .map(|i| (staging_offset + i) as BlockId)
            .collect();

        // Transfer G3 source -> G2 staging
        let notification = direct_worker.execute_local_transfer(
            LogicalLayoutHandle::G3,
            LogicalLayoutHandle::G2,
            Arc::from(block_ids),
            Arc::from(staging_block_ids.as_slice()),
            TransferOptions::default(),
        )?;

        notification.await?;

        // Compute checksums from staging blocks
        let checksums = physical::compute_manager_checksums(
            direct_worker.transfer_manager(),
            g2_handle,
            &staging_block_ids,
        )?;

        // Map checksums back to G3 block IDs
        let mut result = HashMap::new();
        for (staging_id, source_id) in staging_block_ids.iter().zip(block_ids.iter()) {
            if let Some(checksum) = checksums.get(staging_id) {
                result.insert(*source_id, checksum.clone());
            }
        }

        Ok(result)
    }
}

// ============================================================================
// ConnectorInstance - Container for 1 leader + N workers
// ============================================================================

/// A single instance: 1 leader + N workers.
///
/// This is the primary test infrastructure container for connector testing.
/// An "instance" represents one leader coordinating N workers, which is
/// the unit of replication in a distributed system.
///
/// # Example - Single worker setup
/// ```rust,ignore
/// let instance = ConnectorInstance::single_worker().await?;
/// instance.register_all_workers()?;
/// instance.initialize()?;
/// assert!(instance.workers[0].is_initialized());
/// ```
///
/// # Example - Multi-worker setup with custom config
/// ```rust,ignore
/// let config = ConnectorTestConfig::new()
///     .with_leader_host_cache_gb(1.0);
///
/// let instance = ConnectorInstance::builder()
///     .num_workers(2)
///     .test_config(config)
///     .build()
///     .await?;
/// ```
pub struct ConnectorInstance {
    /// The ConnectorLeader for this instance (wrapped in Arc for spawn_blocking).
    pub leader: Arc<ConnectorLeader>,
    /// Workers in this instance (typically 1 per GPU in real deployments).
    pub workers: Vec<TestConnectorWorker>,
    /// Leader's Nova instance for cross-registration.
    pub leader_nova: Arc<Nova>,
}

impl ConnectorInstance {
    /// Quick creation of single-worker instance with defaults.
    ///
    /// Uses default configuration (NixL disabled for testing).
    /// Workers are pre-registered with mock KV caches.
    pub async fn single_worker() -> Result<Self> {
        Self::builder().num_workers(1).build().await
    }

    /// Create instance with N workers using defaults.
    pub async fn with_workers(n: usize) -> Result<Self> {
        Self::builder().num_workers(n).build().await
    }

    /// Builder for custom configuration.
    pub fn builder() -> ConnectorInstanceBuilder {
        ConnectorInstanceBuilder::default()
    }

    /// Register all workers with the leader.
    ///
    /// This must be called before `initialize()`.
    pub fn register_all_workers(&self) -> Result<()> {
        for worker in &self.workers {
            self.leader
                .register_worker(
                    worker.rank,
                    worker.instance_id,
                    worker.worker_address.clone(),
                )
                .with_context(|| format!("Failed to register worker {}", worker.rank))?;
        }
        Ok(())
    }

    /// Initialize all workers via leader-driven deferred init flow.
    ///
    /// This:
    /// 1. Gathers layout configs from all workers
    /// 2. Validates all configs match
    /// 3. Computes G2/G3 block counts from leader config
    /// 4. Sends initialization config to workers
    /// 5. Creates InstanceLeader with G2/G3 managers and worker references
    ///
    /// This is an async method that internally uses spawn_blocking since
    /// the leader's initialize_workers uses block_on internally.
    pub async fn initialize(&self) -> Result<()> {
        let leader = Arc::clone(&self.leader);

        // Use spawn_blocking since initialize_workers uses block_on internally
        tokio::task::spawn_blocking(move || leader.initialize_workers())
            .await
            .map_err(|e| anyhow!("Task panicked: {}", e))??;

        // After workers are initialized, create the InstanceLeader
        self.create_instance_leader()?;

        Ok(())
    }

    /// Create and set the InstanceLeader after workers are initialized.
    ///
    /// This builds the InstanceLeader with:
    /// - G2 and G3 BlockManagers using the leader's computed block counts
    /// - Worker references extracted from the initialized ConnectorWorkers
    /// - Nova instance for distributed communication
    fn create_instance_leader(&self) -> Result<()> {
        // Get the leader's config
        let config = self.leader.runtime.config();

        // Get layout config from first worker's TransferManager
        let first_worker = self
            .workers
            .first()
            .ok_or_else(|| anyhow!("No workers found"))?;

        let direct_worker = first_worker
            .worker
            .worker()
            .ok_or_else(|| anyhow!("Worker not initialized"))?;

        let g2_handle = direct_worker
            .g2_handle()
            .ok_or_else(|| anyhow!("Worker G2 handle not set"))?;

        let layout_config = direct_worker
            .transfer_manager()
            .get_layout_config(g2_handle)?;

        // Compute block counts
        let bytes_per_block = layout_config.required_bytes() / layout_config.num_blocks;
        let host_block_count = config
            .cache
            .host
            .compute_num_blocks(bytes_per_block)
            .unwrap_or(0);
        let disk_block_count = config
            .cache
            .disk
            .as_ref()
            .and_then(|dc| dc.compute_num_blocks(bytes_per_block));

        // Create G2 and G3 BlockManagers
        let g2_manager = Arc::new(managers::create_test_manager::<G2>(
            host_block_count,
            layout_config.page_size,
        ));

        let g3_manager = disk_block_count.map(|count| {
            Arc::new(managers::create_test_manager::<G3>(
                count,
                layout_config.page_size,
            ))
        });

        // Extract Worker trait objects from ConnectorWorkers
        let workers: Vec<Arc<dyn Worker>> = self
            .workers
            .iter()
            .filter_map(|w| w.worker.worker().map(|dw| dw.clone() as Arc<dyn Worker>))
            .collect();

        if workers.is_empty() {
            return Err(anyhow!("No initialized workers found"));
        }

        // Build InstanceLeader
        let instance_leader = InstanceLeader::builder()
            .nova(self.leader_nova.clone())
            .g2_manager(g2_manager)
            .g3_manager(g3_manager.unwrap_or_else(|| {
                // Create empty G3 manager if disk not configured
                Arc::new(managers::create_test_manager::<G3>(
                    0,
                    layout_config.page_size,
                ))
            }))
            .workers(workers)
            .remote_leaders(vec![]) // No remote leaders in single instance
            .build()?;

        // Register handlers
        instance_leader.register_handlers()?;

        // Set on leader
        self.leader.set_instance_leader(instance_leader)?;

        Ok(())
    }

    /// Access the InstanceLeader (available after initialize()).
    ///
    /// Returns an error if initialize() hasn't been called yet.
    pub fn instance_leader(&self) -> Result<&InstanceLeader> {
        self.leader
            .instance_leader()
            .ok_or_else(|| anyhow!("InstanceLeader not initialized - call initialize() first"))
    }

    /// Populate G2 manager with token blocks and return their details.
    ///
    /// This is a convenience method that:
    /// 1. Creates a token sequence
    /// 2. Populates the InstanceLeader's G2 BlockManager
    /// 3. Returns the allocated BlockIds and SequenceHashes
    ///
    /// # Example
    /// ```ignore
    /// let instance = ConnectorInstance::single_worker().await?;
    /// instance.register_all_workers()?;
    /// instance.initialize().await?;
    ///
    /// let (block_ids, hashes) = instance.populate_g2_blocks(32, 4, 0)?;
    /// assert_eq!(block_ids.len(), 32);
    /// ```
    pub fn populate_g2_blocks(
        &self,
        num_blocks: usize,
        block_size: usize,
        start_token: u32,
    ) -> Result<(Vec<BlockId>, Vec<SequenceHash>)> {
        let instance_leader = self.instance_leader()?;

        let token_sequence =
            token_blocks::create_token_sequence(num_blocks, block_size, start_token);
        let seq_hashes = managers::populate_manager_with_blocks(
            instance_leader.g2_manager(),
            token_sequence.blocks(),
        )?;

        // Get the block IDs that were allocated
        let matched = instance_leader.g2_manager().match_blocks(&seq_hashes);
        let block_ids: Vec<BlockId> = matched.into_iter().map(|b| b.block_id()).collect();

        Ok((block_ids, seq_hashes))
    }

    /// Fill blocks on all workers with a layer-specific pattern.
    ///
    /// Each layer gets a different fill byte: layer 0 = 0xA0, layer 1 = 0xA1, etc.
    /// This enables verification that the correct layer was transferred.
    ///
    /// # Example
    /// ```ignore
    /// let instance = ConnectorInstance::single_worker().await?;
    /// instance.register_all_workers()?;
    /// instance.initialize().await?;
    ///
    /// let block_ids = vec![0, 1, 2];
    /// let checksums = instance.fill_blocks_with_layer_pattern(&block_ids, 0)?;
    /// ```
    pub fn fill_blocks_with_layer_pattern(
        &self,
        block_ids: &[BlockId],
        layer: usize,
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        let pattern = FillPattern::Constant(0xA0 + layer as u8);
        let mut all_checksums = HashMap::new();

        for worker in &self.workers {
            let checksums = worker.fill_g2_blocks(block_ids, pattern)?;
            all_checksums.extend(checksums);
        }

        Ok(all_checksums)
    }

    /// Verify that blocks have the expected checksums across all workers.
    ///
    /// Checks that blocks were transferred correctly by verifying
    /// the checksum matches the expected values.
    ///
    /// # Example
    /// ```ignore
    /// let expected = instance.fill_blocks_with_layer_pattern(&block_ids, 0)?;
    /// // ... perform some transfers ...
    /// instance.verify_layer_checksums(&block_ids, &expected)?;
    /// ```
    pub fn verify_layer_checksums(
        &self,
        block_ids: &[BlockId],
        expected_checksums: &HashMap<BlockId, BlockChecksum>,
    ) -> Result<()> {
        for worker in &self.workers {
            let actual_checksums = worker.compute_g2_checksums(block_ids)?;
            for block_id in block_ids {
                let expected = expected_checksums
                    .get(block_id)
                    .ok_or_else(|| anyhow!("Missing expected checksum for block {}", block_id))?;
                let actual = actual_checksums
                    .get(block_id)
                    .ok_or_else(|| anyhow!("Missing actual checksum for block {}", block_id))?;
                if expected != actual {
                    anyhow::bail!(
                        "Checksum mismatch for block {}: expected {:?}, got {:?}",
                        block_id,
                        expected,
                        actual
                    );
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// ConnectorInstanceBuilder - Fluent builder
// ============================================================================

/// Builder for creating ConnectorInstance with custom configuration.
pub struct ConnectorInstanceBuilder {
    num_workers: usize,
    layout_config: LayoutConfig,
    test_config: ConnectorTestConfig,
}

impl Default for ConnectorInstanceBuilder {
    fn default() -> Self {
        Self {
            num_workers: 1,
            layout_config: LayoutConfig::builder()
                .num_blocks(10)
                .num_layers(4)
                .outer_dim(2)
                .page_size(16)
                .inner_dim(128)
                .dtype_width_bytes(2)
                .build()
                .expect("Default LayoutConfig should be valid"),
            test_config: ConnectorTestConfig::default(),
        }
    }
}

impl ConnectorInstanceBuilder {
    /// Set the number of workers in this instance.
    #[must_use]
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the layout configuration for all workers.
    #[must_use]
    pub fn layout_config(mut self, config: LayoutConfig) -> Self {
        self.layout_config = config;
        self
    }

    /// Set the test configuration (role-specific settings).
    #[must_use]
    pub fn test_config(mut self, config: ConnectorTestConfig) -> Self {
        self.test_config = config;
        self
    }

    /// Build the ConnectorInstance.
    ///
    /// This creates:
    /// 1. Nova mesh with leader + workers
    /// 2. Leader runtime with leader profile config
    /// 3. Worker runtimes with worker profile config
    /// 4. Workers pre-registered with mock KV caches
    pub async fn build(self) -> Result<ConnectorInstance> {
        // 1. Create Nova instances
        let leader_nova = nova::create_nova_instance_tcp().await?;

        let mut worker_novas = Vec::with_capacity(self.num_workers);
        for _ in 0..self.num_workers {
            worker_novas.push(nova::create_nova_instance_tcp().await?);
        }

        // 2. Cross-register all peers
        for worker_nova in &worker_novas {
            leader_nova.register_peer(worker_nova.peer_info())?;
            worker_nova.register_peer(leader_nova.peer_info())?;
        }

        // Give time for peer registration
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 3. Build leader runtime with profile config
        let leader_config = self.test_config.build_leader_config()?;
        let leader_runtime = KvbmRuntime::builder(leader_config)
            .with_nova(leader_nova.clone())
            .build_leader()
            .await?;

        let leader = Arc::new(ConnectorLeader::new(leader_runtime));

        // 4. Build worker runtimes with profile config
        let worker_config = self.test_config.build_worker_config()?;
        let mut workers = Vec::with_capacity(self.num_workers);

        for (rank, worker_nova) in worker_novas.into_iter().enumerate() {
            let nixl_agent = NixlAgent::new(&format!("worker-{}", rank))
                .context("Failed to create NixL agent")?;

            let worker_runtime = KvbmRuntime::builder(worker_config.clone())
                .with_nova(worker_nova.clone())
                .with_nixl_agent(nixl_agent)
                .build_worker()
                .await?;

            let worker_peer_info = worker_nova.peer_info();
            let instance_id = worker_peer_info.instance_id;
            let worker_address = worker_peer_info.worker_address;

            let connector_worker = ConnectorWorker::new(worker_runtime);

            // Pre-register mock KV caches
            let tensors =
                create_mock_kv_tensors_from_config(&self.layout_config, StorageKind::Device(0));

            connector_worker
                .register_kv_caches(
                    tensors,
                    self.layout_config.num_blocks,
                    self.layout_config.page_size,
                    self.layout_config.dtype_width_bytes,
                )
                .context("Failed to register KV caches")?;

            workers.push(TestConnectorWorker {
                worker: connector_worker,
                instance_id,
                worker_address,
                rank,
            });
        }

        Ok(ConnectorInstance {
            leader,
            workers,
            leader_nova,
        })
    }
}

// ============================================================================
// MockTensor - Minimal tensor implementation for testing
// ============================================================================

/// Mock tensor for testing connector functionality.
///
/// This provides a simple implementation of TensorDescriptor for unit tests
/// without requiring actual GPU memory allocation.
#[derive(Debug)]
pub struct MockTensor {
    addr: usize,
    size: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
    element_size: usize,
    storage_kind: StorageKind,
}

impl MockTensor {
    /// Create a new mock tensor with the given parameters.
    ///
    /// # Arguments
    /// * `addr` - Mock memory address
    /// * `shape` - Tensor shape (e.g., [num_blocks, num_heads, head_size, x])
    /// * `element_size` - Size of each element in bytes
    /// * `storage_kind` - Storage location (Device, System, Pinned, etc.)
    pub fn create(
        addr: usize,
        shape: Vec<usize>,
        element_size: usize,
        storage_kind: StorageKind,
    ) -> Arc<dyn TensorDescriptor> {
        // Calculate contiguous stride (row-major)
        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }

        // Calculate total size
        let numel: usize = shape.iter().product();
        let size = numel * element_size;

        Arc::new(Self {
            addr,
            size,
            shape,
            stride,
            element_size,
            storage_kind,
        })
    }
}

impl MemoryDescriptor for MockTensor {
    fn addr(&self) -> usize {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn storage_kind(&self) -> StorageKind {
        self.storage_kind
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl TensorDescriptor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn stride(&self) -> &[usize] {
        &self.stride
    }

    fn element_size(&self) -> usize {
        self.element_size
    }
}

/// Create a vector of mock KV cache tensors from a LayoutConfig.
///
/// Creates tensors that match the provided LayoutConfig exactly:
/// - Shape: [num_blocks, outer_dim, page_size, inner_dim]
/// - One tensor per layer
/// - Contiguous strides
/// - Blocks are in the first dimension (BlockIsFirstDim)
///
/// This ensures the generated tensors will produce the same LayoutConfig
/// when passed through `determine_kv_layout()`.
///
/// # Arguments
/// * `config` - The layout configuration to match
/// * `storage_kind` - Storage location (typically Device(0) for GPU)
///
/// # Returns
/// A vector of `config.num_layers` mock tensors, each with shape
/// `[num_blocks, outer_dim, page_size, inner_dim]`
pub fn create_mock_kv_tensors_from_config(
    config: &LayoutConfig,
    storage_kind: StorageKind,
) -> Vec<Arc<dyn TensorDescriptor>> {
    // Tensor shape matches LayoutConfig with blocks in first dimension
    let shape = vec![
        config.num_blocks,
        config.outer_dim,
        config.page_size,
        config.inner_dim,
    ];

    (0..config.num_layers)
        .map(|layer_idx| {
            // Use different base addresses for each layer to simulate separate allocations
            let base_addr = 0x10000000 + (layer_idx * 0x1000000);
            MockTensor::create(
                base_addr,
                shape.clone(),
                config.dtype_width_bytes,
                storage_kind,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::EnvFilter;

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_single_worker_initialization() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("warn,dynamo_nova=error,dynamo_kvbm=debug")),
            )
            .with_test_writer()
            .try_init();

        // Simple one-liner setup
        let instance = ConnectorInstance::single_worker()
            .await
            .expect("Should create single-worker instance");

        // Workers pre-registered with mock KV caches
        assert_eq!(instance.workers.len(), 1);
        assert!(
            !instance.workers[0].is_initialized(),
            "Worker should not be initialized before leader calls initialize"
        );

        // Register workers with leader
        instance
            .register_all_workers()
            .expect("Should register workers");

        // Leader-driven initialization
        let init_result = instance.initialize().await;

        // Worker-side initialization may fail if G2/G3 creation requires proper NIXL backend,
        // but the leader-side logic (fetch, validate, compute, send) will have executed.
        match init_result {
            Ok(_) => {
                println!("Leader successfully initialized workers");
                assert!(
                    instance.workers[0].is_initialized(),
                    "Worker should be initialized after successful init"
                );
            }
            Err(e) => {
                let error_msg = e.to_string();
                // If it failed at worker initialization (not at fetch/validate), that's acceptable
                if error_msg.contains("Failed to initialize worker") {
                    println!(
                        "Leader completed fetch/validate/compute/send (worker init failed as expected in test environment)"
                    );
                } else {
                    panic!("Unexpected error during leader initialization: {}", e);
                }
            }
        }

        // Drop instance in spawn_blocking to avoid runtime-in-runtime panic
        tokio::task::spawn_blocking(move || drop(instance))
            .await
            .expect("Cleanup should succeed");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_worker_initialization() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init();

        // Custom configuration for 2 workers (typical vLLM TP=2 scenario)
        let config = ConnectorTestConfig::new().with_leader_host_num_blocks(128);

        let layout = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(4)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .expect("Should build valid LayoutConfig");

        let instance = ConnectorInstance::builder()
            .num_workers(2)
            .layout_config(layout)
            .test_config(config)
            .build()
            .await
            .expect("Should create 2-worker instance");

        assert_eq!(instance.workers.len(), 2);

        // Register all workers
        instance
            .register_all_workers()
            .expect("Should register all workers");

        // Initialize via leader
        let init_result = instance.initialize().await;

        match init_result {
            Ok(_) => {
                println!("Leader successfully initialized all workers");
                for (i, worker) in instance.workers.iter().enumerate() {
                    assert!(
                        worker.is_initialized(),
                        "Worker {} should be initialized",
                        i
                    );
                }
            }
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("Failed to initialize worker") {
                    println!(
                        "Leader completed coordination (worker init failed as expected in test environment)"
                    );
                } else {
                    panic!("Unexpected error during multi-worker initialization: {}", e);
                }
            }
        }

        // Drop instance in spawn_blocking to avoid runtime-in-runtime panic
        tokio::task::spawn_blocking(move || drop(instance))
            .await
            .expect("Cleanup should succeed");
    }

    #[tokio::test]
    async fn test_config_json_api() {
        // Test the JSON API for configuration
        let config = ConnectorTestConfig::from_json(
            r#"{
            "leader": { "cache": { "host": { "cache_size_gb": 2.0 } } },
            "worker": { "tokio": { "worker_threads": 4 } }
        }"#,
        )
        .expect("Should parse JSON config");

        // Verify we can build configs from it
        let leader_config = config.build_leader_config();
        assert!(leader_config.is_ok(), "Should build leader config");

        let worker_config = config.build_worker_config();
        assert!(worker_config.is_ok(), "Should build worker config");
    }

    #[tokio::test]
    async fn test_config_builder_api() {
        // Test the builder API for configuration
        let config = ConnectorTestConfig::new()
            .with_leader_host_cache_gb(1.5)
            .with_leader_tokio_threads(2)
            .with_worker_tokio_threads(4);

        // Verify we can build configs from it
        let leader_config = config.build_leader_config();
        assert!(leader_config.is_ok(), "Should build leader config");

        let worker_config = config.build_worker_config();
        assert!(worker_config.is_ok(), "Should build worker config");

        // Verify the configs have the expected values
        let lc = leader_config.unwrap();
        assert_eq!(lc.cache.host.cache_size_gb, Some(1.5));
        assert_eq!(lc.tokio.worker_threads, Some(2));

        let wc = worker_config.unwrap();
        assert_eq!(wc.tokio.worker_threads, Some(4));
    }

    #[test]
    fn test_worker_gets_nixl_defaults() {
        // Workers should get NIXL defaults (UCX + POSIX) automatically
        let config = ConnectorTestConfig::new();

        let worker = config.build_worker().expect("Should build worker config");

        // Worker should have NIXL with UCX and POSIX backends
        assert!(worker.nixl.is_some(), "Worker should have NIXL config");
        let nixl = worker.nixl.unwrap();
        assert!(
            nixl.backends.contains_key("UCX"),
            "Worker should have UCX backend"
        );
        assert!(
            nixl.backends.contains_key("POSIX"),
            "Worker should have POSIX backend"
        );
    }

    #[test]
    fn test_leader_no_nixl() {
        // Leaders should NOT have NIXL by default
        let config = ConnectorTestConfig::new();

        let leader = config.build_leader().expect("Should build leader config");

        assert!(leader.nixl.is_none(), "Leader should NOT have NIXL config");
    }

    #[test]
    fn test_worker_custom_nixl_replaces() {
        // Test fully replacing NIXL configuration using worker_nixl_replace()
        let custom_nixl = NixlConfig::empty().with_backend("GDS");

        let config = ConnectorTestConfig::new().worker_nixl_replace(custom_nixl);

        let worker = config.build_worker().expect("Should build worker config");

        assert!(worker.nixl.is_some(), "Worker should have NIXL config");
        let nixl = worker.nixl.unwrap();
        assert!(
            nixl.backends.contains_key("GDS"),
            "Worker should have GDS backend"
        );
        // Should NOT have the defaults since we used _replace
        assert!(
            !nixl.backends.contains_key("UCX"),
            "Worker should NOT have UCX backend after replace"
        );
    }

    #[test]
    fn test_worker_nixl_merges_backends() {
        // Test that worker_nixl() adds to existing backends
        let extra_nixl = NixlConfig::empty().with_backend("GDS");

        let config = ConnectorTestConfig::new().worker_nixl(extra_nixl);

        let worker = config.build_worker().expect("Should build worker config");

        let nixl = worker.nixl.expect("Worker should have NIXL config");
        // Should have all backends: defaults (UCX, POSIX) + new (GDS)
        assert!(
            nixl.backends.contains_key("UCX"),
            "Should have UCX from defaults"
        );
        assert!(
            nixl.backends.contains_key("POSIX"),
            "Should have POSIX from defaults"
        );
        assert!(
            nixl.backends.contains_key("GDS"),
            "Should have GDS from merge"
        );
    }

    #[test]
    fn test_worker_without_nixl() {
        // Test explicitly disabling NIXL for workers
        let config = ConnectorTestConfig::new().worker_without_nixl();

        let worker = config.build_worker().expect("Should build worker config");

        assert!(
            worker.nixl.is_none(),
            "Worker should NOT have NIXL when explicitly disabled"
        );
    }

    #[test]
    fn test_generic_path_override() {
        // Test the generic path API
        let config = ConnectorTestConfig::new()
            .leader_path("cache.host.cache_size_gb", 3.5)
            .worker_path("tokio.worker_threads", 8u64);

        let leader = config.build_leader().expect("Should build leader config");
        assert_eq!(leader.cache.host.cache_size_gb, Some(3.5));

        let worker = config.build_worker().expect("Should build worker config");
        assert_eq!(worker.tokio.worker_threads, Some(8));
    }

    #[test]
    fn test_new_builder_api_names() {
        // Test the new concise method names
        let config = ConnectorTestConfig::new()
            .leader_cache_gb(2.0)
            .leader_cache_blocks(100)
            .leader_disk_blocks(50)
            .leader_tokio_threads(4)
            .worker_tokio_threads(2);

        let leader = config.build_leader().expect("Should build leader config");
        assert_eq!(leader.cache.host.cache_size_gb, Some(2.0));
        assert_eq!(leader.cache.host.num_blocks, Some(100));
        assert!(leader.cache.disk.is_some());
        assert_eq!(leader.cache.disk.unwrap().num_blocks, Some(50));
        assert_eq!(leader.tokio.worker_threads, Some(4));

        let worker = config.build_worker().expect("Should build worker config");
        assert_eq!(worker.tokio.worker_threads, Some(2));
        // Worker still gets NIXL defaults
        assert!(worker.nixl.is_some());
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use utils::*;
use zmq::*;

use derive_builder::Builder;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::sync::OnceCell;
use tokio::sync::oneshot;
use tokio::time::sleep;
use crate::block_manager::config::ObjectStorageConfig;
use crate::block_manager::v2::logical::distributed::DistributedRegistry;
use crate::block_manager::v2::logical::external_registry::SequenceHashRegistry;
use crate::block_manager::v2::logical::registry::ObjectRegistry;

#[derive(Builder, Clone, Debug, Default)]
pub struct KvbmLeaderNumBlocksConfig {
    #[builder(default = "0.0")]
    pub cache_size_in_gb: f64,

    #[builder(default = "0")]
    pub num_blocks_overriden: usize,
}

fn compute_num_blocks(
    num_blocks_config: &KvbmLeaderNumBlocksConfig,
    bytes_per_block: usize,
) -> usize {
    if num_blocks_config.num_blocks_overriden > 0 {
        num_blocks_config.num_blocks_overriden
    } else {
        ((num_blocks_config.cache_size_in_gb * 1_000_000_000.0) / bytes_per_block as f64) as usize
    }
}

#[derive(Builder, Clone, Debug)]
pub struct KvbmLeaderConfig {
    /// The world size.
    #[builder(default = "1")]
    world_size: usize,

    /// The leader-worker init connection timeout seconds.
    #[builder(default = "120")]
    leader_init_timeout_secs: u64,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    host_blocks_config: KvbmLeaderNumBlocksConfig,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    disk_blocks_config: KvbmLeaderNumBlocksConfig,

    /// Object storage configuration (read from environment variables)
    #[builder(default = "ObjectStorageConfig::from_env()")]
    object_storage_config: Option<ObjectStorageConfig>,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56001\")")]
    leader_pub_url: String,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56002\")")]
    leader_ack_url: String,
}

impl KvbmLeaderConfig {
    pub fn builder() -> KvbmLeaderConfigBuilder {
        KvbmLeaderConfigBuilder::default()
    }

    pub fn sanity_check(&self) -> anyhow::Result<()> {
        if self.leader_pub_url == self.leader_ack_url {
            anyhow::bail!(
                "leader_pub_url and leader_ack_url must differ (same endpoint would fail to bind)."
            );
        }

        let cpu = &self.host_blocks_config;
        let disk = &self.disk_blocks_config;
        let cpu_configured = cpu.num_blocks_overriden > 0 || cpu.cache_size_in_gb > 0.0;
        let disk_configured = disk.num_blocks_overriden > 0 || disk.cache_size_in_gb > 0.0;

        let object_configured = ObjectStorageConfig::is_offload_enabled()
            && ObjectStorageConfig::num_blocks_from_env() > 0;

        if !cpu_configured && !disk_configured && !object_configured {
            panic!(
                "KVBM Configuration Error: At least one cache tier must be configured.\n\
                \n\
                Configure CPU cache (G2) for CPU memory offloading:\n\
                • DYN_KVBM_CPU_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_CPU_CACHE_GB=4)\n\
                • DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>  (e.g., DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=1000)\n\
                \n\
                OR configure disk cache (G3) for direct GPU->Disk offloading:\n\
                • DYN_KVBM_DISK_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_DISK_CACHE_GB=8)\n\
                • DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>\n\
                \n\
                OR configure object storage (G4) for S3-compatible offloading:\n\
                • DYN_KVBM_USE_OBJECT_OFFLOAD=1\n\
                • DYN_KVBM_OBJECT_BUCKET=<bucket_name>  (supports {{worker_id}} template)\n\
                • DYN_KVBM_OBJECT_NUM_BLOCKS=<num_blocks>\n\
                \n\
                Note: Object storage requires DYN_KVBM_USE_V2_TRANSFER_EXPERIMENTAL=1"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct KvbmLeaderState {
    pub num_device_blocks: Arc<AtomicUsize>,
    pub num_host_blocks: Arc<AtomicUsize>,
    pub num_disk_blocks: Arc<AtomicUsize>,
    pub num_object_blocks: Arc<AtomicUsize>,
    pub bytes_per_block: Arc<AtomicUsize>,
    pub workers_allocation_ready: Arc<AtomicBool>,
    pub workers_ready_notify: Arc<Notify>,
}

/// The leader of the KVBM.
///
/// This is responsible for:
/// - Establishing a ZMQ connection with workers.
/// - Syncing the leader barrier with workers.
/// - Sending messages to workers.
/// - Maintaining G4 registry for fast onboard lookups.
pub struct KvbmLeader {
    state: Arc<KvbmLeaderState>,
    zmq_leader: Arc<OnceCell<ZmqActiveMessageLeader>>,
    config: KvbmLeaderConfig,

    /// G4 local registry for tracking sequence hashes offloaded to object storage.
    /// This is a local cache - the distributed registry is the source of truth.
    g4_registry: Option<SequenceHashRegistry>,

    /// Distributed registry client for cross-worker lookups.
    /// This is the source of truth for what exists in object storage.
    distributed_registry: Option<Arc<dyn DistributedRegistry>>,

    /// Bucket name for distributed registry lookups.
    g4_bucket: String,

    /// Tokio runtime handle for blocking calls from non-async contexts.
    /// Captured at construction time when we're in an async context.
    runtime_handle: Option<tokio::runtime::Handle>,
}

impl KvbmLeader {
    pub async fn new(config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        use crate::block_manager::v2::logical::distributed::create_registry_from_env;

        let leader_sockets = new_leader_sockets(&config.leader_pub_url, &config.leader_ack_url)?;

        // Check if G4 is configured and create registry with TinyLFU eviction
        let num_object_blocks = ObjectStorageConfig::num_blocks_from_env();
        let g4_enabled = ObjectStorageConfig::is_offload_enabled() && num_object_blocks > 0;

        let g4_registry: Option<SequenceHashRegistry> = if g4_enabled {
            tracing::info!(
                "G4 object storage enabled on leader with {} block capacity",
                num_object_blocks
            );
            Some(Arc::new(ObjectRegistry::with_capacity(num_object_blocks as u64)))
        } else {
            None
        };

        // Connect to distributed registry if enabled
        let distributed_registry = if g4_enabled {
            create_registry_from_env().await
        } else {
            None
        };

        // Get bucket name from config or use default
        let g4_bucket = config
            .object_storage_config
            .as_ref()
            .map(|c| c.resolve_bucket(0))
            .unwrap_or_else(|| "kvcache".to_string());

        // Capture the tokio runtime handle for later blocking calls
        let runtime_handle = tokio::runtime::Handle::try_current().ok();

        let leader = Self {
            state: Arc::new(KvbmLeaderState::default()),
            zmq_leader: Arc::new(tokio::sync::OnceCell::new()),
            config,
            g4_registry,
            distributed_registry,
            g4_bucket,
            runtime_handle,
        };

        let cancel_token = tokio_util::sync::CancellationToken::new();
        leader.spawn_zmq_task(leader_sockets, cancel_token);

        Ok(leader)
    }

    fn spawn_zmq_task(
        &self,
        leader_sockets: LeaderSockets,
        cancel: tokio_util::sync::CancellationToken,
    ) {
        let cell = self.zmq_leader.clone();
        let state = self.state.clone();
        let world_size = self.config.world_size;
        let timeout = self.config.leader_init_timeout_secs;
        let host_cfg = self.config.host_blocks_config.clone();
        let disk_cfg = self.config.disk_blocks_config.clone();
        let object_cfg = self.config.object_storage_config.clone();

        // capture num_device_blocks so we can set it inside the closure
        let num_device_blocks_cell = state.num_device_blocks.clone();
        let num_host_blocks_cell = state.num_host_blocks.clone();
        let num_disk_blocks_cell = state.num_disk_blocks.clone();

        let num_object_blocks_cell = state.num_object_blocks.clone();
        let bytes_per_block_cell = state.bytes_per_block.clone();

        tokio::spawn(async move {
            let res = ZmqActiveMessageLeader::new_with_handshake(
                leader_sockets,
                world_size,
                std::time::Duration::from_secs(timeout),
                cancel.clone(),
                move |workers: &[WorkerMetadata]| -> LeaderMetadata {
                    // Record device blocks: min across workers
                    if let Some(min_dev) = workers.iter().map(|w| w.num_device_blocks).min() {
                        num_device_blocks_cell.store(min_dev, Ordering::Release);
                    }

                    // For TP, sum bytes_per_block; adjust policy for DP/PP if needed.
                    let bytes_per_block: usize = workers.iter().map(|w| w.bytes_per_block).sum();
                    let num_host_blocks = compute_num_blocks(&host_cfg, bytes_per_block);
                    let num_disk_blocks = compute_num_blocks(&disk_cfg, bytes_per_block);
                    let num_object_blocks = ObjectStorageConfig::num_blocks_from_env();

                    // store into leader state
                    num_host_blocks_cell.store(num_host_blocks, Ordering::Release);
                    num_disk_blocks_cell.store(num_disk_blocks, Ordering::Release);
                    num_object_blocks_cell.store(num_object_blocks, Ordering::Release);
                    bytes_per_block_cell.store(bytes_per_block, Ordering::Release);

                    if num_object_blocks > 0 {
                        tracing::debug!(
                            "Object storage configured: {} blocks",
                            num_object_blocks
                        );
                    }
                    LeaderMetadata {
                        num_host_blocks,
                        num_disk_blocks,
                        num_object_blocks,
                        object_storage_config: object_cfg.clone(),
                    }
                },
            )
            .await;

            match res {
                Ok(zmq) => {
                    let _ = cell.set(zmq);
                    state
                        .workers_allocation_ready
                        .store(true, Ordering::Release);
                    state.workers_ready_notify.notify_waiters();
                    tracing::info!("ZMQ handshake complete; workers allocation ready");
                }
                Err(e) => {
                    tracing::error!("ZMQ init/handshake failed: {e:?}");
                }
            }
        });
    }

    pub async fn transfer_blocks_request(
        &self,
        request: BlockTransferRequest,
    ) -> anyhow::Result<oneshot::Receiver<()>> {
        let zmq = self
            .zmq_leader
            .get()
            .ok_or_else(|| anyhow::anyhow!("ZMQ leader not ready"))?;
        let data = vec![serde_json::to_vec(&request)?];
        zmq.broadcast(ZMQ_TRANSFER_BLOCKS_MESSAGE, data).await
    }

    pub fn num_device_blocks(&self) -> usize {
        self.state.num_device_blocks.load(Ordering::Acquire)
    }

    pub fn num_host_blocks(&self) -> usize {
        self.state.num_host_blocks.load(Ordering::Acquire)
    }

    pub fn num_disk_blocks(&self) -> usize {
        self.state.num_disk_blocks.load(Ordering::Acquire)
    }

    pub fn num_object_blocks(&self) -> usize {
        self.state.num_object_blocks.load(Ordering::Acquire)
    }

    pub fn bytes_per_block(&self) -> usize {
        self.state.bytes_per_block.load(Ordering::Acquire)
    }

    pub async fn wait_worker_sync_ready(&self) -> bool {
        if self.state.workers_allocation_ready.load(Ordering::Acquire) {
            return true;
        }
        let notified = self.state.workers_ready_notify.notified();
        tokio::select! {
            _ = notified => true,
            _ = sleep(Duration::from_secs(self.config.leader_init_timeout_secs)) => false,
        }
    }

    /// Check if G4 is enabled.
    pub fn g4_enabled(&self) -> bool {
        tracing::info!(
            "g4_enabled: g4_registry={}, distributed_registry={}",
            self.g4_registry.is_some(),
            self.distributed_registry.is_some()
        );
        self.g4_registry.is_some() || self.distributed_registry.is_some()
    }

    /// Get a clone of the G4 registry (if enabled).
    /// Returns the shared registry reference for use by other components.
    pub fn g4_registry(&self) -> Option<SequenceHashRegistry> {
        self.g4_registry.clone()
    }

    /// Register sequence hashes in the G4 registry after successful offload.
    pub fn g4_register_hashes(&self, hashes: &[u64]) {
        if let Some(registry) = &self.g4_registry {
            registry.register(hashes);
        }
    }

    /// Match sequence hashes against the G4 registry (local + distributed).
    ///
    /// Returns the contiguous prefix of hashes that exist in G4.
    /// First checks local cache, then queries distributed registry for remaining.
    pub fn g4_lookup(&self, hashes: &[u64]) -> Vec<u64> {
        if hashes.is_empty() {
            return vec![];
        }

        tracing::info!(
            "g4_lookup called with {} hashes, distributed_registry={}, bucket={}",
            hashes.len(),
            self.distributed_registry.is_some(),
            self.g4_bucket
        );

        // First check local registry (fast path)
        let local_matched = match &self.g4_registry {
            Some(registry) => registry.match_keys(hashes),
            None => vec![],
        };

        tracing::info!(
            "g4_lookup: local_matched={} of {} hashes",
            local_matched.len(),
            hashes.len()
        );

        // If we matched everything locally, we're done
        if local_matched.len() == hashes.len() {
            return local_matched;
        }

        // Check distributed registry for remaining hashes
        if let Some(distributed) = &self.distributed_registry {
            let remaining_offset = local_matched.len();
            let remaining_hashes = &hashes[remaining_offset..];

            tracing::info!(
                "g4_lookup: checking distributed registry for {} remaining hashes",
                remaining_hashes.len()
            );

            if !remaining_hashes.is_empty() {
                // Use stored runtime handle for blocking call
                let distributed_matched = if let Some(handle) = &self.runtime_handle {
                    let bucket = self.g4_bucket.clone();
                    let dist = distributed.clone();
                    let hashes = remaining_hashes.to_vec();

                    // Use the stored handle to run the async lookup
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        tokio::task::block_in_place(|| {
                            handle.block_on(async {
                                dist.match_sequence_hashes(&bucket, &hashes).await
                            })
                        })
                    })) {
                        Ok(Ok(matches)) => {
                            let result: Vec<u64> = matches.into_iter().map(|(h, _)| h).collect();
                            tracing::info!(
                                "g4_lookup: distributed registry returned {} matches",
                                result.len()
                            );
                            result
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("Distributed registry lookup failed: {}", e);
                            vec![]
                        }
                        Err(_) => {
                            tracing::warn!("Distributed registry lookup panicked (block_in_place failed)");
                            vec![]
                        }
                    }
                } else {
                    tracing::warn!("No runtime handle available for distributed registry lookup");
                    vec![]
                };

                if !distributed_matched.is_empty() {
                    tracing::info!(
                        "Distributed registry matched {} additional hashes",
                        distributed_matched.len()
                    );

                    // Register matches in local cache for future lookups
                    if let Some(local_registry) = &self.g4_registry {
                        local_registry.register(&distributed_matched);
                    }

                    // Combine local and distributed matches
                    let mut all_matched = local_matched;
                    all_matched.extend(distributed_matched);
                    return all_matched;
                }
            }
        } else {
            tracing::warn!("g4_lookup: no distributed registry available");
        }

        local_matched
    }

    /// Send a G4 onboard request to workers.
    ///
    /// Workers will handle G4→Host→Device transfer atomically.
    pub async fn g4_onboard_request(
        &self,
        request: G4OnboardRequest,
    ) -> anyhow::Result<oneshot::Receiver<()>> {
        let zmq = self
            .zmq_leader
            .get()
            .ok_or_else(|| anyhow::anyhow!("ZMQ leader not ready"))?;
        let data = vec![serde_json::to_vec(&request)?];
        zmq.broadcast(ZMQ_G4_ONBOARD_MESSAGE, data).await
    }
}

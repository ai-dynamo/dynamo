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
use super::registry::{DistributedRegistry, ObjectRegistry, RemoteKeyBuilder, SequenceHashRegistry};
use crate::block_manager::block::transfer::remote::RemoteKey;

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
                Optionally set DYN_KVBM_USE_V2_TRANSFER_EXPERIMENTAL=1 for experimental handler."
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

pub struct KvbmLeader {
    state: Arc<KvbmLeaderState>,
    zmq_leader: Arc<OnceCell<ZmqActiveMessageLeader>>,
    config: KvbmLeaderConfig,
    g4_registry: Option<SequenceHashRegistry>,
    distributed_registry: Option<Arc<dyn DistributedRegistry>>,
    key_builder: Option<RemoteKeyBuilder>,
    worker_id: u32,
    runtime_handle: Option<tokio::runtime::Handle>,
}

impl KvbmLeader {
    pub async fn new(config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        use super::registry::create_registry_from_env;

        let leader_sockets = new_leader_sockets(&config.leader_pub_url, &config.leader_ack_url)?;

        // Check if G4 is configured and create registry with TinyLFU eviction
        let num_object_blocks = ObjectStorageConfig::num_blocks_from_env();
        let g4_enabled = ObjectStorageConfig::is_offload_enabled() && num_object_blocks > 0;

        let g4_registry: Option<SequenceHashRegistry> = if g4_enabled {
            tracing::debug!(
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

        let key_builder = if g4_enabled {
            config.object_storage_config.as_ref().map(|obj_cfg| {
                RemoteKeyBuilder::from_object_config(
                    obj_cfg.bucket_template.clone(),
                    config.world_size as u32,
                )
            })
        } else {
            None
        };

        let worker_id = 0u32;

        // Capture the tokio runtime handle for later blocking calls
        let runtime_handle = tokio::runtime::Handle::try_current().ok();

        let leader = Self {
            state: Arc::new(KvbmLeaderState::default()),
            zmq_leader: Arc::new(tokio::sync::OnceCell::new()),
            config,
            g4_registry,
            distributed_registry,
            key_builder,
            worker_id,
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

    pub fn g4_enabled(&self) -> bool {
        self.g4_registry.is_some() || self.distributed_registry.is_some()
    }

    pub fn g4_registry(&self) -> Option<SequenceHashRegistry> {
        self.g4_registry.clone()
    }

    pub fn g4_register_hashes(&self, hashes: &[u64]) {
        if hashes.is_empty() {
            return;
        }

        if let Some(registry) = &self.g4_registry {
            registry.register(hashes);
            tracing::debug!("Registered {} hashes in local G4 cache", hashes.len());
        }

        let (distributed, builder, handle) = match (
            &self.distributed_registry,
            &self.key_builder,
            &self.runtime_handle,
        ) {
            (Some(d), Some(b), Some(h)) => (d.clone(), b, h),
            _ => return,
        };

        let keys: Vec<RemoteKey> = hashes
            .iter()
            .map(|&hash| builder.build_for_worker(self.worker_id, hash))
            .collect();
        let num_keys = keys.len();

        handle.spawn(async move {
            if let Err(e) = distributed.register_remote_keys(&keys).await {
                tracing::warn!(
                    "Failed to register {} RemoteKeys in distributed registry: {}",
                    num_keys,
                    e
                );
            }
        });
    }

    pub async fn g4_filter_for_offload(&self, hashes: &[u64]) -> Vec<u64> {
        if hashes.is_empty() {
            return vec![];
        }

        let need_check: Vec<u64> = match &self.g4_registry {
            Some(registry) => {
                let existing: std::collections::HashSet<_> =
                    registry.match_keys(hashes).into_iter().collect();
                hashes
                    .iter()
                    .filter(|h| !existing.contains(h))
                    .copied()
                    .collect()
            }
            None => hashes.to_vec(),
        };

        if need_check.is_empty() {
            return vec![];
        }

        let (distributed, builder) = match (&self.distributed_registry, &self.key_builder) {
            (Some(d), Some(b)) => (d, b),
            _ => return need_check,
        };

        let bucket = builder.location_for_worker(self.worker_id);
        match distributed.can_offload(&bucket, &need_check).await {
            Ok(result) => result.can_offload,
            Err(e) => {
                tracing::warn!("Distributed registry can_offload failed: {}", e);
                need_check
            }
        }
    }

    pub fn g4_lookup(&self, hashes: &[u64]) -> Vec<u64> {
        self.g4_lookup_remote_keys(hashes)
            .into_iter()
            .filter_map(|key| key.sequence_hash())
            .collect()
    }

    pub fn g4_lookup_remote_keys(&self, hashes: &[u64]) -> Vec<RemoteKey> {
        if hashes.is_empty() {
            return vec![];
        }

        let local_matched: Vec<RemoteKey> = match (&self.g4_registry, &self.key_builder) {
            (Some(registry), Some(builder)) => {
                let local_hashes = registry.match_keys(hashes);
                local_hashes
                    .iter()
                    .map(|&hash| builder.build_for_worker(self.worker_id, hash))
                    .collect()
            }
            _ => vec![],
        };

        if local_matched.len() == hashes.len() {
            return local_matched;
        }

        let (key_builder, distributed, handle) = match (
            &self.key_builder,
            &self.distributed_registry,
            &self.runtime_handle,
        ) {
            (Some(kb), Some(d), Some(h)) => (kb, d, h),
            _ => return local_matched,
        };

        let remaining_hashes = &hashes[local_matched.len()..];
        if remaining_hashes.is_empty() {
            return local_matched;
        }

        let candidate_keys: Vec<RemoteKey> = remaining_hashes
            .iter()
            .flat_map(|&hash| key_builder.build_all(hash))
            .collect();

        let dist = distributed.clone();
        let distributed_matched = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::task::block_in_place(|| {
                handle.block_on(async { dist.match_remote_keys(&candidate_keys).await })
            })
        })) {
            Ok(Ok(matches)) => matches,
            Ok(Err(e)) => {
                tracing::warn!("Distributed registry lookup failed: {}", e);
                return local_matched;
            }
            Err(_) => {
                tracing::warn!("Distributed registry lookup panicked (block_in_place failed)");
                return local_matched;
            }
        };

        if distributed_matched.is_empty() {
            return local_matched;
        }

        let mut seen_hashes = std::collections::HashSet::new();
        let mut unique_matches: Vec<RemoteKey> = Vec::new();

        for key in distributed_matched {
            if let Some(hash) = key.sequence_hash() {
                if seen_hashes.insert(hash) {
                    unique_matches.push(key);
                }
            }
        }

        if let Some(local_registry) = &self.g4_registry {
            let local_bucket = key_builder.location_for_worker(self.worker_id);
            let hashes_to_cache: Vec<u64> = unique_matches
                .iter()
                .filter(|key| key.location() == local_bucket)
                .filter_map(|key| key.sequence_hash())
                .collect();

            if !hashes_to_cache.is_empty() {
                local_registry.register(&hashes_to_cache);
            }
        }

        let mut all_matched = local_matched;
        all_matched.extend(unique_matches);
        all_matched
    }

    pub fn key_builder(&self) -> Option<&RemoteKeyBuilder> {
        self.key_builder.as_ref()
    }

    pub fn worker_id(&self) -> u32 {
        self.worker_id
    }
}

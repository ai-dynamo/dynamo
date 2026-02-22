// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::registry::{NoMetadata, PositionalKey, Registry};
use super::remote::PositionalRemoteHandle;
use super::*;
use crate::block_manager::block::transfer::remote::RemoteKey;

use utils::*;
use zmq::*;

use derive_builder::Builder;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::sync::OnceCell;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::sleep;

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
    /// The worker rank (0-indexed).
    /// Used as worker_id in positional registry keys.
    #[builder(default = "0")]
    pub rank: usize,

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
        if !cpu_configured && !disk_configured {
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
                Note: If only disk cache is configured, KVBM will offload directly from GPU (G1) to Disk (G3), bypassing CPU memory (G2)."
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
    pub workers_allocation_ready: Arc<AtomicBool>,
    pub workers_ready_notify: Arc<Notify>,
}

struct MatchPrefixRequest {
    handle: Option<PositionalRemoteHandle>,
    keys: Vec<PositionalKey>,
    reply: oneshot::Sender<Vec<(PositionalKey, RemoteKey, NoMetadata)>>,
}

/// The leader of the KVBM.
///
/// This is responsible for:
/// - Establishing a ZMQ connection with workers.
/// - Syncing the leader barrier with workers.
/// - Sending messages to workers.
pub struct KvbmLeader {
    state: Arc<KvbmLeaderState>,
    zmq_leader: Arc<OnceCell<ZmqActiveMessageLeader>>,
    config: KvbmLeaderConfig,
    /// Handle for remote registry operations (e.g., G4 object storage).
    /// Uses channels to avoid blocking Tokio worker threads.
    remote_handle: RwLock<Option<PositionalRemoteHandle>>,
    /// Leader-local worker queue for async match-prefix lookups.
    /// This avoids per-request tokio::spawn in synchronous caller paths.
    match_prefix_tx: mpsc::UnboundedSender<MatchPrefixRequest>,
    /// Tracks request IDs with failed G4 transfers for explicit failure detection.
    failed_g4_requests: RwLock<HashSet<String>>,
}

impl KvbmLeader {
    pub async fn new(config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        let leader_sockets = new_leader_sockets(&config.leader_pub_url, &config.leader_ack_url)?;
        let (match_prefix_tx, match_prefix_rx) = mpsc::unbounded_channel();

        let leader = Self {
            state: Arc::new(KvbmLeaderState::default()),
            zmq_leader: Arc::new(tokio::sync::OnceCell::new()),
            config,
            remote_handle: RwLock::new(None),
            match_prefix_tx,
            failed_g4_requests: RwLock::new(HashSet::new()),
        };

        let cancel_token = tokio_util::sync::CancellationToken::new();
        leader.spawn_zmq_task(leader_sockets, cancel_token);
        leader.spawn_match_prefix_worker(match_prefix_rx);

        Ok(leader)
    }

    fn spawn_match_prefix_worker(&self, mut rx: mpsc::UnboundedReceiver<MatchPrefixRequest>) {
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let result = if let Some(handle) = req.handle {
                    handle.match_prefix(req.keys).await
                } else {
                    vec![]
                };
                let _ = req.reply.send(result);
            }
            tracing::debug!("KvbmLeader match-prefix worker shutting down");
        });
    }

    /// Set the remote registry by spawning a RemoteHandle task.
    ///
    /// The registry should be created externally and passed in.
    /// This spawns a background task that handles registry operations via channels,
    /// avoiding blocking of Tokio worker threads.
    ///
    /// Returns `true` if set successfully, `false` if already set.
    pub fn set_remote_registry(
        &self,
        registry: Arc<dyn Registry<PositionalKey, RemoteKey, NoMetadata> + Send + Sync>,
    ) -> bool {
        let mut guard = self.remote_handle.write();
        if guard.is_some() {
            tracing::warn!("Remote registry already set");
            return false;
        }

        let handle = PositionalRemoteHandle::spawn(registry);
        *guard = Some(handle);
        tracing::info!("Remote registry enabled via RemoteHandle");
        true
    }

    /// Get the remote handle if available.
    pub fn remote_handle(&self) -> Option<PositionalRemoteHandle> {
        self.remote_handle.read().clone()
    }

    /// Check if remote registry is enabled.
    pub fn remote_registry_enabled(&self) -> bool {
        self.remote_handle.read().is_some()
    }

    /// Schedule an async match-prefix query against the remote registry.
    ///
    /// This keeps synchronous caller paths (e.g., vLLM connector slot logic)
    /// non-blocking without extending the registry handle API surface.
    pub fn schedule_match_prefix(
        &self,
        keys: Vec<PositionalKey>,
    ) -> oneshot::Receiver<Vec<(PositionalKey, RemoteKey, NoMetadata)>> {
        let (tx, rx) = oneshot::channel();

        if keys.is_empty() {
            let _ = tx.send(vec![]);
            return rx;
        }

        let req = MatchPrefixRequest {
            handle: self.remote_handle(),
            keys,
            reply: tx,
        };

        if let Err(err) = self.match_prefix_tx.send(req) {
            tracing::warn!(
                "schedule_match_prefix: worker queue closed; returning empty lookup result"
            );
            let _ = err.0.reply.send(vec![]);
        }

        rx
    }

    /// Get the worker ID used for registry operations.
    pub fn worker_id(&self) -> u64 {
        self.config.rank as u64
    }

    /// Get the world size (number of TP ranks).
    ///
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get the remote storage configuration based on environment variables.
    ///
    /// Returns `Some(config)` if remote storage is configured, `None` otherwise.
    /// This mirrors the logic in `create_remote_context` in worker.rs.
    ///
    /// Environment variables:
    /// - `DYN_KVBM_REMOTE_STORAGE_TYPE`: "object", "disk", or "auto" (default: "auto")
    /// - `DYN_KVBM_OBJECT_BUCKET` or `AWS_DEFAULT_BUCKET`: Bucket name for object storage
    /// - `DYN_KVBM_REMOTE_DISK_PATH`: Base path for disk storage
    /// - `DYN_KVBM_REMOTE_DISK_USE_GDS`: Enable GPU Direct Storage for disk (default: true)
    pub fn remote_storage_config(
        &self,
    ) -> Option<crate::block_manager::config::RemoteStorageConfig> {
        use crate::block_manager::config::RemoteStorageConfig;

        let worker_id = self.config.rank;
        let storage_type = std::env::var("DYN_KVBM_REMOTE_STORAGE_TYPE")
            .unwrap_or_else(|_| "auto".to_string())
            .to_lowercase();

        // Get object storage config
        let bucket = std::env::var("DYN_KVBM_OBJECT_BUCKET")
            .or_else(|_| std::env::var("AWS_DEFAULT_BUCKET"))
            .ok()
            .map(|b| b.replace("{worker_id}", &worker_id.to_string()));

        let object_endpoint = std::env::var("DYN_KVBM_OBJECT_ENDPOINT")
            .or_else(|_| std::env::var("AWS_ENDPOINT_URL"))
            .or_else(|_| std::env::var("AWS_ENDPOINT_OVERRIDE"))
            .ok();

        let object_region = std::env::var("DYN_KVBM_OBJECT_REGION")
            .or_else(|_| std::env::var("AWS_REGION"))
            .ok();

        // Get disk storage config
        let disk_path = std::env::var("DYN_KVBM_REMOTE_DISK_PATH")
            .ok()
            .map(|p| p.replace("{worker_id}", &worker_id.to_string()));

        let disk_use_gds = std::env::var("DYN_KVBM_REMOTE_DISK_USE_GDS")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true);
        let disk_gds_reads_only = std::env::var("DYN_KVBM_REMOTE_DISK_GDS_READS_ONLY")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        use crate::block_manager::config::{
            DISK_FLAG_GDS_READ, DISK_FLAG_GDS_WRITE, DISK_FLAGS_POSIX_BOTH,
        };
        let disk_flags = if !disk_use_gds {
            DISK_FLAGS_POSIX_BOTH
        } else if disk_gds_reads_only {
            DISK_FLAG_GDS_READ
        } else {
            DISK_FLAG_GDS_WRITE | DISK_FLAG_GDS_READ
        };

        match storage_type.as_str() {
            "disk" => disk_path.map(|path| RemoteStorageConfig::Disk {
                base_path: path,
                transfer_flags: disk_flags,
            }),
            "object" => Some(RemoteStorageConfig::Object {
                default_bucket: bucket,
                endpoint: object_endpoint,
                region: object_region,
            }),
            _ => match (&bucket, &disk_path) {
                (Some(_), Some(_)) | (Some(_), None) => Some(RemoteStorageConfig::Object {
                    default_bucket: bucket,
                    endpoint: object_endpoint,
                    region: object_region,
                }),
                (None, Some(path)) => Some(RemoteStorageConfig::Disk {
                    base_path: path.clone(),
                    transfer_flags: disk_flags,
                }),
                (None, None) => None,
            },
        }
    }

    /// Send a remote transfer request to the workers.
    /// Used for both G4 onboard (object -> device) and offload (device -> object).
    pub async fn remote_transfer_request(
        &self,
        request: RemoteTransferRequest,
    ) -> anyhow::Result<oneshot::Receiver<()>> {
        let zmq = self
            .zmq_leader
            .get()
            .ok_or_else(|| anyhow::anyhow!("ZMQ leader not ready"))?;
        let data = vec![serde_json::to_vec(&request)?];
        zmq.broadcast(ZMQ_REMOTE_TRANSFER_MESSAGE, data).await
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

        // capture num_device_blocks so we can set it inside the closure
        let num_device_blocks_cell = state.num_device_blocks.clone();
        let num_host_blocks_cell = state.num_host_blocks.clone();
        let num_disk_blocks_cell = state.num_disk_blocks.clone();

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

                    // store into leader state
                    num_host_blocks_cell.store(num_host_blocks, Ordering::Release);
                    num_disk_blocks_cell.store(num_disk_blocks, Ordering::Release);

                    LeaderMetadata {
                        num_host_blocks,
                        num_disk_blocks,
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

    /// Mark a request as having a failed G4 transfer.
    pub fn mark_g4_failed(&self, request_id: &str) {
        self.failed_g4_requests
            .write()
            .insert(request_id.to_string());
    }

    /// Check if a request has a failed G4 transfer.
    pub fn has_g4_failed(&self, request_id: &str) -> bool {
        self.failed_g4_requests.read().contains(request_id)
    }

    /// Clear the G4 failure flag for a request (called after recovery).
    pub fn clear_g4_failed(&self, request_id: &str) {
        self.failed_g4_requests.write().remove(request_id);
    }
}

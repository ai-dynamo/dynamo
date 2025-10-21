// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use dynamo_runtime::DistributedRuntime;
use utils::*;
use zmq::*;

use dynamo_runtime::utils::leader_worker_barrier::LeaderBarrier;

use anyhow::{Context, anyhow};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::sync::OnceCell;
use tokio::sync::oneshot;
use tokio::time::sleep;

/// Data that is sent to workers over ETCD to establish a ZMQ connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvbmLeaderData {
    pub pub_url: String,
    pub ack_url: String,
    pub num_host_blocks: usize,
    pub num_disk_blocks: usize,
}

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
    /// The barrier id to use for syncing with workers.
    #[builder(default = "String::from(\"kvbm\")")]
    barrier_id_prefix: String,

    /// The world size.
    #[builder(default = "1")]
    world_size: usize,

    /// The leader-worker init connection timeout seconds.
    #[builder(default = "120")]
    leader_init_timeout_secs: u64,

    #[builder(setter(strip_option))]
    drt: Option<DistributedRuntime>,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    host_blocks_config: KvbmLeaderNumBlocksConfig,

    #[builder(default = "KvbmLeaderNumBlocksConfig::default()")]
    disk_blocks_config: KvbmLeaderNumBlocksConfig,
}

impl KvbmLeaderConfig {
    pub fn builder() -> KvbmLeaderConfigBuilder {
        KvbmLeaderConfigBuilder::default()
    }

    pub fn sanity_check(&self) -> anyhow::Result<()> {
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
    //readiness flags
    workers_sync_ready: Arc<AtomicBool>,
    workers_sync_ready_notify: Arc<Notify>,
    workers_sync_done: Arc<AtomicBool>,
}

impl KvbmLeader {
    pub async fn new(mut config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        let drt = match config.drt.take() {
            Some(dtr) => dtr,
            None => {
                anyhow::bail!("No distributed runtime provided");
            }
        };

        let leader_sockets = new_leader_sockets("tcp://127.0.0.1")?;

        let leader = Self {
            state: Arc::new(KvbmLeaderState::default()),
            zmq_leader: Arc::new(tokio::sync::OnceCell::new()),
            config,
            workers_sync_ready: Arc::new(AtomicBool::new(false)),
            workers_sync_ready_notify: Arc::new(Notify::new()),
            workers_sync_done: Arc::new(AtomicBool::new(false)),
        };

        let cancel_token = tokio_util::sync::CancellationToken::new();

        // The leader_sockets struct cannot be cloned,
        // so we use a tuple to "struct" the two urls
        let leader_urls = (
            leader_sockets.pub_url.clone(),
            leader_sockets.ack_url.clone(),
        );
        leader.spawn_barrier_task(drt, leader_urls);
        leader.spawn_zmq_task(leader_sockets, cancel_token);

        Ok(leader)
    }

    fn spawn_barrier_task(&self, drt: DistributedRuntime, leader_urls: (String, String)) {
        let state = self.state.clone();
        let leader_config = self.config.clone();
        let ready = Arc::clone(&self.workers_sync_ready);
        let notify = Arc::clone(&self.workers_sync_ready_notify);
        let done = Arc::clone(&self.workers_sync_done);

        tokio::spawn(async move {
            match KvbmLeader::run_barrier_sync(drt, leader_urls, leader_config).await {
                Ok((num_device_blocks, num_host_blocks, num_disk_blocks)) => {
                    // write back results
                    state
                        .num_device_blocks
                        .store(num_device_blocks, Ordering::Release);
                    state
                        .num_host_blocks
                        .store(num_host_blocks, Ordering::Release);
                    state
                        .num_disk_blocks
                        .store(num_disk_blocks, Ordering::Release);
                    ready.store(true, Ordering::Release);
                    done.store(true, Ordering::Release);
                    notify.notify_waiters();
                }
                Err(e) => {
                    tracing::error!("Barrier sync failed: {e:?}");
                    done.store(true, Ordering::Release);
                    notify.notify_waiters();
                }
            }
        });
    }

    async fn run_barrier_sync(
        drt: DistributedRuntime,
        leader_urls: (String, String),
        leader_config: KvbmLeaderConfig,
    ) -> anyhow::Result<(usize, usize, usize)> {
        let barrier_id_worker_to_leader =
            format!("{}{}", leader_config.barrier_id_prefix, "-worker-to-leader");
        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            leader_config.world_size,
            barrier_id_worker_to_leader
        );

        // Build our leader barrier and publish the data.
        // TODO: Use a separate timeout parameter from the ZMQ connection timeout
        let worker_to_leader_barrier: LeaderBarrier<(), worker::KvbmWorkerData> =
            LeaderBarrier::new(
                barrier_id_worker_to_leader.clone(),
                leader_config.world_size,
                Some(Duration::from_secs(leader_config.leader_init_timeout_secs)),
            );

        let worker_data = worker_to_leader_barrier
            .sync(&drt, &())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to sync worker to leader barrier: {:?}", e))?;

        let num_device_blocks = worker_data
            .values()
            .map(|data| data.num_device_blocks)
            .min()
            .unwrap();

        // TODO: this works for TP, need to redefine bytes_per_block when we enable the DP/PP
        let bytes_per_block: usize = worker_data.values().map(|d| d.bytes_per_block).sum();

        assert!(
            bytes_per_block > 0,
            "bytes_per_block must be greater than 0"
        );

        tracing::info!(
            "Worker to leader barrier synced with {} workers",
            leader_config.world_size
        );
        tracing::debug!("Worker data: {:?}", worker_data);

        let num_host_blocks =
            compute_num_blocks(&leader_config.host_blocks_config, bytes_per_block);
        let num_disk_blocks =
            compute_num_blocks(&leader_config.disk_blocks_config, bytes_per_block);

        // Start the second sync to transfer num_host_blocks and num_disk_blocks to worker
        let barrier_id_leader_to_worker =
            format!("{}{}", leader_config.barrier_id_prefix, "-leader-to-worker");
        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            leader_config.world_size,
            barrier_id_leader_to_worker
        );

        let (leader_pub_url, leader_ack_url) = leader_urls;
        let zmq_data_leader_to_worker = Arc::new(KvbmLeaderData {
            pub_url: leader_pub_url,
            ack_url: leader_ack_url,
            num_host_blocks,
            num_disk_blocks,
        });

        let leader_to_worker_barrier: LeaderBarrier<KvbmLeaderData, ()> = LeaderBarrier::new(
            barrier_id_leader_to_worker.clone(),
            leader_config.world_size,
            Some(Duration::from_secs(leader_config.leader_init_timeout_secs)),
        );

        let _worker_data = leader_to_worker_barrier
            .sync(&drt, zmq_data_leader_to_worker.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to sync leader to worker barrier: {:?}", e))?;

        tracing::info!(
            "Worker to leader barrier synced with {} workers",
            leader_config.world_size
        );
        Ok((num_device_blocks, num_host_blocks, num_disk_blocks))
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

        tokio::spawn(async move {
            let res = ZmqActiveMessageLeader::new(
                leader_sockets,
                world_size,
                std::time::Duration::from_secs(timeout),
                cancel,
            )
            .await;

            match res {
                Ok(zmq) => {
                    let _ = cell.set(zmq);
                    // mark ready
                    state
                        .workers_allocation_ready
                        .store(true, Ordering::Release);
                    state.workers_ready_notify.notify_waiters();
                }
                Err(e) => {
                    tracing::error!("ZMQ init failed: {e:?}");
                }
            }
        });
    }

    // This is supposed to be used in non-blocking leader initialization
    pub fn spawn_leader_readiness_barrier(&self, drt: DistributedRuntime) {
        let timeout_secs = self.config.leader_init_timeout_secs;
        let state = self.state.clone();
        let leader_config = self.config.clone();
        let handle = drt.runtime().primary();
        handle.spawn(async move {
            if !state.workers_allocation_ready.load(Ordering::Acquire) {
                // Wait until ZMQ marks ready or we time out.
                let waited = tokio::time::timeout(
                    Duration::from_secs(timeout_secs),
                    state.workers_ready_notify.notified(),
                )
                .await;
                if waited.is_err() {
                    tracing::error!(
                        "leader readiness barrier wait timed out after {timeout_secs} seconds"
                    );
                    return;
                }
                // Double-check the flag (Acquire) after wakeup.
                if !state.workers_allocation_ready.load(Ordering::Acquire) {
                    tracing::error!("leader readiness notify fired but flag not set; aborting");
                    return;
                }
            }

            match KvbmLeader::run_leader_readiness(drt, leader_config).await {
                Ok(()) => {
                    tracing::info!("leader readiness barrier synced!");
                }
                Err(e) => {
                    tracing::error!("leader readiness barrier failed: {e:?}");
                }
            }
        });
    }

    // This is supposed to be used in blocking leader initialization
    pub fn run_leader_readiness_barrier_blocking(
        &self,
        drt: DistributedRuntime,
    ) -> anyhow::Result<()> {
        let state = self.state.clone();
        let timeout_secs = self.config.leader_init_timeout_secs;
        let leader_config = self.config.clone();

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async move {
                    // Create the future *before* checking the flag to avoid a lost-notify race.
                    let notified = state.workers_ready_notify.notified();

                    if !state.workers_allocation_ready.load(Ordering::Acquire) {
                        // Wait (with timeout) until ZMQ task marks ready.
                        tokio::time::timeout(Duration::from_secs(timeout_secs), notified)
                            .await
                            .map_err(|_| anyhow!("timed out waiting for workers_allocation_ready after {timeout_secs} seconds"))?;

                        // Double-check after wake to ensure the flag is actually set.
                        if !state.workers_allocation_ready.load(Ordering::Acquire) {
                            return Err(anyhow!(
                                "notified but workers_allocation_ready is still false"
                            ));
                        }
                    }

                    KvbmLeader::run_leader_readiness(drt, leader_config).await
                })
                .context("leader readiness barrier failed")
        })
    }

    async fn run_leader_readiness(
        drt: DistributedRuntime,
        leader_config: KvbmLeaderConfig,
    ) -> anyhow::Result<()> {
        let barrier_id_leader_ready =
            format!("{}{}", leader_config.barrier_id_prefix, "-leader-ready");
        tracing::info!(
            "Syncing leader readiness barrier with {} workers on barrier id {}",
            leader_config.world_size,
            barrier_id_leader_ready
        );

        let leader_readiness_barrier: LeaderBarrier<(), ()> = LeaderBarrier::new(
            barrier_id_leader_ready.clone(),
            leader_config.world_size,
            Some(Duration::from_secs(leader_config.leader_init_timeout_secs)),
        );

        let _ = leader_readiness_barrier
            .sync(&drt, &())
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to sync leader readiness barrier on leader: {:?}", e)
            })?;

        Ok(())
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

    pub fn is_worker_sync_ready(&self) -> bool {
        self.workers_sync_ready.load(Ordering::Acquire)
    }

    pub fn is_worker_sync_done(&self) -> bool {
        self.workers_sync_done.load(Ordering::Acquire)
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
        if self.is_worker_sync_ready() {
            return true;
        }
        if self.is_worker_sync_done() {
            return false;
        }

        let notified = self.workers_sync_ready_notify.notified();
        if self.is_worker_sync_ready() {
            return true;
        }
        if self.is_worker_sync_done() {
            return false;
        }

        // bounded wait
        tokio::select! {
            _ = notified => {
                self.is_worker_sync_ready()
            }
            _ = sleep(Duration::from_secs(self.config.leader_init_timeout_secs)) => false,
        }
    }
}

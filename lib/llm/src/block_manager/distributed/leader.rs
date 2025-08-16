// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use dynamo_runtime::DistributedRuntime;
use utils::*;
use zmq::*;

use dynamo_runtime::utils::leader_worker_barrier::LeaderBarrier;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

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

    #[builder(default = false)]
    pub is_overriden: bool,

    #[builder(default = "0")]
    pub num_blocks_overriden: usize
}

fn compute_num_blocks(num_blocks_config: &KvbmLeaderNumBlocksConfig, bytes_per_block: usize) -> usize {
    if num_blocks_config.is_overriden {
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
}

/// The leader of the KVBM.
///
/// This is responsible for:
/// - Establishing a ZMQ connection with workers.
/// - Syncing the leader barrier with workers.
/// - Sending messages to workers.
pub struct KvbmLeader {
    num_device_blocks: usize,
    num_host_blocks: usize,
    num_disk_blocks: usize,
    zmq_leader: ZmqActiveMessageLeader,
    config: KvbmLeaderConfig,
}

impl KvbmLeader {
    pub async fn new(mut config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        let drt = match config.drt.take() {
            Some(dtr) => dtr,
            None => {
                anyhow::bail!("No distributed runtime provided");
            }
        };

        let barrier_id_worker_to_leader = format!("{}{}", config.barrier_id_prefix, "-worker-to-leader");
        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            config.world_size,
            barrier_id_worker_to_leader
        );

        let leader_sockets = new_leader_sockets("tcp://127.0.0.1")?;

        let zmq_data_worker_to_leader: Arc<KvbmLeaderData> = Arc::new(KvbmLeaderData {
            pub_url: leader_sockets.pub_url.clone(),
            ack_url: leader_sockets.ack_url.clone(),
            num_host_blocks: 0, // doesn't matter for worker to leader sync
            num_disk_blocks: 0, // doesn't matter for worker to leader sync
        });

        // Build our leader barrier and publish the data.
        // TODO: Use a separate timeout parameter from the ZMQ connection timeout
        let worker_to_leader_barrier: LeaderBarrier<KvbmLeaderData, worker::KvbmWorkerData> =
            LeaderBarrier::new(
                barrier_id_worker_to_leader.clone(),
                config.world_size,
                Some(Duration::from_secs(config.leader_init_timeout_secs)),
            );

        let worker_data = worker_to_leader_barrier
            .sync(&drt, zmq_data_worker_to_leader.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to sync worker to leader barrier: {:?}", e))?;

        let num_device_blocks = worker_data
            .values()
            .map(|data| data.num_device_blocks)
            .min()
            .unwrap();

        let bytes_per_block = worker_data
            .values()
            .map(|data| data.bytes_per_block)
            .sum();

        assert!(bytes_per_block > 0, "bytes_per_block must be greater than 0");

        tracing::info!("Worker to leader barrier synced with {} workers", config.world_size);
        tracing::debug!("Worker data: {:?}", worker_data);

        let num_host_blocks = compute_num_blocks(&config.host_blocks_config, bytes_per_block);
        let num_disk_blocks = compute_num_blocks(&config.disk_blocks_config, bytes_per_block);

        // Start the second sync to transfer num_host_blocks and num_disk_blocks to worker
        let barrier_id_leader_to_worker = format!("{}{}", config.barrier_id_prefix, "-leader-to-worker");
        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            config.world_size,
            barrier_id_leader_to_worker
        );

        let zmq_data_leader_to_worker = Arc::new(KvbmLeaderData {
            pub_url: leader_sockets.pub_url.clone(),
            ack_url: leader_sockets.ack_url.clone(),
            num_host_blocks,
            num_disk_blocks,
        });

        let leader_to_worker_barrier: LeaderBarrier<KvbmLeaderData, worker::KvbmWorkerData> =
            LeaderBarrier::new(
                barrier_id_leader_to_worker.clone(),
                config.world_size,
                Some(Duration::from_secs(config.leader_init_timeout_secs)),
            );

        let _worker_data = leader_to_worker_barrier
            .sync(&drt, zmq_data_leader_to_worker.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to sync leader to worker barrier: {:?}", e))?;

        tracing::info!("Worker to leader barrier synced with {} workers", config.world_size);

        // Now, create our active message leader.
        // This also blocks until a ZMQ connection has been established.
        let cancel_token = CancellationToken::new();
        let zmq_leader = ZmqActiveMessageLeader::new(
            leader_sockets,
            config.world_size,
            Duration::from_secs(config.leader_init_timeout_secs),
            cancel_token.clone(),
        )
        .await?;

        Ok(Self {
            num_device_blocks,
            num_host_blocks,
            num_disk_blocks,
            zmq_leader,
            config,
        })
    }

    pub async fn transfer_blocks_request(
        &self,
        request: BlockTransferRequest,
    ) -> anyhow::Result<oneshot::Receiver<()>> {
        let data = vec![serde_json::to_vec(&request)?];
        self.zmq_leader
            .broadcast(ZMQ_TRANSFER_BLOCKS_MESSAGE, data)
            .await
    }

    pub fn num_device_blocks(&self) -> usize {
        self.num_device_blocks
    }

    pub fn num_host_blocks(&self) -> usize {
        self.num_host_blocks
    }

    pub fn num_disk_blocks(&self) -> usize {
        self.num_disk_blocks
    }
}

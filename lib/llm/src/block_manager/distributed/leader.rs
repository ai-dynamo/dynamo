// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use utils::*;
use zmq::*;

use dynamo_runtime::utils::leader_worker_barrier::LeaderBarrier;
use dynamo_runtime::{DistributedRuntime, Runtime};

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

const INIT_TIMEOUT_SECS: u64 = 120;

/// Data that is sent to workers over ETCD to establish a ZMQ connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvbmLeaderData {
    pub pub_url: String,
    pub ack_url: String,
    pub num_host_blocks: usize,
    pub num_disk_blocks: usize,
}

#[derive(Builder, Clone, Debug)]
pub struct KvbmLeaderConfig {
    #[builder(default = "0")]
    num_host_blocks: usize,

    #[builder(default = "0")]
    num_disk_blocks: usize,

    /// The barrier id to use for syncing with workers.
    #[builder(default = "String::from(\"kvbm\")")]
    barrier_id: String,

    /// The world size.
    #[builder(default = "1")]
    world_size: usize,
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
    _worker_data: Arc<HashMap<String, ()>>, // TODO: Replace with KvbmLeaderData
    zmq_leader: ZmqActiveMessageLeader,
    config: KvbmLeaderConfig,
}

impl KvbmLeader {
    pub async fn new(config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        let runtime = Runtime::from_current()?;

        let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            config.world_size,
            config.barrier_id
        );

        let leader_sockets = new_leader_sockets("tcp://127.0.0.1")?;

        let zmq_data = Arc::new(KvbmLeaderData {
            pub_url: leader_sockets.pub_url.clone(),
            ack_url: leader_sockets.ack_url.clone(),
            num_host_blocks: config.num_host_blocks,
            num_disk_blocks: config.num_disk_blocks,
        });

        // Build our leader barrier and publish the data.
        let leader_barrier: LeaderBarrier<KvbmLeaderData, ()> = LeaderBarrier::new(
            config.barrier_id.clone(),
            config.world_size,
            Some(Duration::from_secs(INIT_TIMEOUT_SECS)),
        );

        let worker_data = leader_barrier
            .sync(&drt, zmq_data.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to sync leader barrier: {:?}", e))?;

        tracing::info!("Leader barrier synced with {} workers", config.world_size);
        tracing::debug!("Worker data: {:?}", worker_data);

        // Now, create our active message leader.
        // This also blocks until a ZMQ connection has been established.
        let cancel_token = CancellationToken::new();
        let zmq_leader = ZmqActiveMessageLeader::new(
            leader_sockets,
            config.world_size,
            Duration::from_secs(INIT_TIMEOUT_SECS),
            cancel_token.clone(),
        )
        .await?;

        Ok(Self {
            _worker_data: Arc::new(worker_data),
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

    pub fn num_host_blocks(&self) -> usize {
        self.config.num_host_blocks
    }

    pub fn num_disk_blocks(&self) -> usize {
        self.config.num_disk_blocks
    }
}

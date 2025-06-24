// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use utils::*;
use zmq::*;

use dynamo_runtime::{utils::leader_worker_barrier::LeaderBarrier, DistributedRuntime};

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Data that is sent to workers over ETCD to establish a ZMQ connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvbmLeaderData {
    pub zmq_url: String,
    pub broadcast_port: usize,
    pub ack_port: usize,
    pub num_host_blocks: usize,
    pub num_disk_blocks: usize,
}

fn compute_num_blocks(env_var: &str, bytes_per_block: usize) -> usize {
    let cache_size_gb = std::env::var(env_var)
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(0);
    (cache_size_gb * 1_000_000_000) / bytes_per_block
}

#[derive(Builder, Clone, Debug)]
pub struct KvbmLeaderConfig {
    /// Amount of bytes within a full kv cache block (summed across all ranks).
    bytes_per_block: usize,

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
    _drt: DistributedRuntime,
    // The DistributedRuntime only stores a handle, so we need to keep the runtime around.
    _runtime: tokio::runtime::Runtime,
    _zmq_leader: ZmqActiveMessageLeader,
}

impl KvbmLeader {
    pub fn new(config: KvbmLeaderConfig) -> anyhow::Result<Self> {
        let (drt, runtime) = build_drt()?;

        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            config.world_size,
            config.barrier_id
        );

        let num_host_blocks = compute_num_blocks("DYNAMO_KVBM_CPU_CACHE", config.bytes_per_block);
        let num_disk_blocks = compute_num_blocks("DYNAMO_KVBM_DISK_CACHE", config.bytes_per_block);

        // TODO: For now, just hardcode localhost.
        let zmq_data = Arc::new(KvbmLeaderData {
            zmq_url: "127.0.0.1".to_string(),
            broadcast_port: 5555,
            ack_port: 5556,
            num_host_blocks,
            num_disk_blocks,
        });

        // Build our leader barrier and publish the data.
        let leader_barrier: LeaderBarrier<KvbmLeaderData, ()> = LeaderBarrier::new(
            config.barrier_id,
            config.world_size,
            Some(Duration::from_secs(30)),
        );

        let drt_clone = drt.clone();
        let zmq_data_clone = zmq_data.clone();

        // Block leader initialization (and vLLM) until all workers have come online.
        drt.runtime()
            .primary()
            .block_on(async move {
                leader_barrier
                    .sync(&drt_clone, zmq_data_clone.as_ref())
                    .await
            })
            .map_err(|e| anyhow::anyhow!("Failed to sync leader barrier: {:?}", e))?;

        tracing::info!("Leader barrier synced with {} workers", config.world_size);

        // Now, create our active message leader.
        // This also blocks until a ZMQ connection has been established.
        let zmq_leader = drt.runtime().primary().block_on(async move {
            let cancel_token = CancellationToken::new();
            ZmqActiveMessageLeader::new(
                &zmq_data.zmq_url,
                zmq_data.broadcast_port,
                zmq_data.ack_port,
                config.world_size,
                Duration::from_secs(30),
                cancel_token.clone(),
            )
            .await
        })?;

        Ok(Self {
            _drt: drt,
            _runtime: runtime,
            _zmq_leader: zmq_leader,
        })
    }
}

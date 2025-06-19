// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use utils::*;
use zmq::*;

use dynamo_runtime::{utils::leader_worker_barrier::LeaderBarrier, DistributedRuntime};

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
    pub fn new(
        barrier_id: String,
        bytes_per_block: usize,
        world_size: usize,
    ) -> anyhow::Result<Self> {
        let (drt, runtime) = build_drt()?;

        tracing::info!(
            "Syncing leader barrier with {} workers on barrier id {}",
            world_size,
            barrier_id
        );

        let num_host_blocks = compute_num_blocks("DYNAMO_KVBM_CPU_CACHE", bytes_per_block);
        let num_disk_blocks = compute_num_blocks("DYNAMO_KVBM_DISK_CACHE", bytes_per_block);

        // TODO: For now, just hardcode localhost.
        let zmq_data = Arc::new(KvbmLeaderData {
            zmq_url: "127.0.0.1".to_string(),
            broadcast_port: 5555,
            ack_port: 5556,
            num_host_blocks,
            num_disk_blocks,
        });

        // Build our leader barrier and publish the data.
        let leader_barrier =
            LeaderBarrier::new(barrier_id, world_size, Some(Duration::from_secs(30)));

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

        tracing::info!("Leader barrier synced with {} workers", world_size);

        // Now, create our active message leader.
        // This also blocks until a ZMQ connection has been established.
        let zmq_leader = drt.runtime().primary().block_on(async move {
            let cancel_token = CancellationToken::new();
            ZmqActiveMessageLeader::new(
                &zmq_data.zmq_url,
                zmq_data.broadcast_port,
                zmq_data.ack_port,
                world_size,
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

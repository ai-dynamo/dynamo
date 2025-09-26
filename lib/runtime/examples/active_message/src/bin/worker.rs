// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
};
use std::env;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use active_message_example::ComputeHandler;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting Worker");

    let leader_endpoint =
        env::var("LEADER_ENDPOINT").unwrap_or_else(|_| "tcp://127.0.0.1:5555".to_string());

    // Get worker rank from environment (optional for MPI/torch.distributed compatibility)
    let worker_rank = env::var("RANK")
        .ok()
        .and_then(|r| r.parse::<usize>().ok());

    if let Some(rank) = worker_rank {
        info!("Worker rank: {}", rank);
    }

    let cancel_token = CancellationToken::new();

    let manager =
        ZmqActiveMessageManager::new("tcp://0.0.0.0:0".to_string(), cancel_token.clone()).await?;

    let client = manager.zmq_client();

    info!("Worker listening on: {}", client.endpoint());
    info!("Worker instance ID: {}", client.instance_id());

    let handler = Arc::new(ComputeHandler);
    manager.register_handler(handler, None).await?;

    info!("Registered compute handler");

    // Discover leader instance ID (in real usage, this might be configured differently)
    let leader_instance_id = uuid::Uuid::new_v4();
    let leader_peer = PeerInfo::new(leader_instance_id, leader_endpoint.clone());

    client.connect_to_peer(leader_peer).await?;

    info!("Connected to leader at {}", leader_endpoint);

    // Join cohort with optional rank
    match client.join_cohort(leader_instance_id, worker_rank).await {
        Ok(response) => {
            if response.accepted {
                info!("Successfully joined cohort at position: {:?}", response.position);
                if let Some(rank) = response.expected_rank {
                    info!("Assigned rank: {}", rank);
                }
            } else {
                warn!("Failed to join cohort: {:?}", response.reason);
                return Ok(());
            }
        }
        Err(e) => {
            warn!("Error joining cohort: {}", e);
            return Ok(());
        }
    }

    cancel_token.cancelled().await;

    info!("Worker shutting down");
    manager.shutdown().await?;

    Ok(())
}

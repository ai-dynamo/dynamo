// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use dynamo_runtime::active_message::{
    client::ActiveMessageClient,
    manager::ActiveMessageManager,
    zmq::{
        cohort::{LeaderWorkerCohort, LeaderWorkerCohortConfig, LeaderWorkerCohortConfigBuilder, CohortType, CohortFailurePolicy},
        builtin_handlers::{JoinCohortHandler, RemoveServiceHandler, RequestShutdownHandler},
        ZmqActiveMessageManager,
    },
};
use std::{sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing::info;

use active_message_example::ComputeRequest;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting Leader");

    let cancel_token = CancellationToken::new();

    let manager =
        ZmqActiveMessageManager::new("tcp://0.0.0.0:5555".to_string(), cancel_token.clone())
            .await?;

    let client = manager.zmq_client();

    info!("Leader listening on: {}", client.endpoint());
    info!("Leader instance ID: {}", client.instance_id());

    // Create cohort with 2 expected workers
    let cohort_config = LeaderWorkerCohortConfigBuilder::default()
        .leader_instance(client.instance_id())
        .client(client.clone())
        .cohort_type(CohortType::FixedSize(2))
        .failure_policy(CohortFailurePolicy::TerminateAll)
        .build()?;

    let cohort = Arc::new(LeaderWorkerCohort::from_config(cohort_config));

    // Register cohort handlers
    let join_handler = Arc::new(JoinCohortHandler::new(cohort.clone()));
    manager.register_handler(join_handler, None).await?;

    let remove_handler = Arc::new(RemoveServiceHandler::new(cohort.clone()));
    manager.register_handler(remove_handler, None).await?;

    let shutdown_handler = Arc::new(RequestShutdownHandler::new(
        manager.manager_state(),
        cancel_token.clone(),
    ));
    manager.register_handler(shutdown_handler, None).await?;

    info!("Waiting for 2 workers to join cohort...");

    // Wait for workers to join the cohort
    let mut attempts = 0;
    let max_attempts = 30; // 30 seconds with 1-second intervals

    loop {
        if cohort.is_full().await {
            break;
        }

        attempts += 1;
        if attempts >= max_attempts {
            info!("Cohort not ready - timeout waiting for workers");
            return Ok(());
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    info!("Cohort ready! Waiting for workers to register compute handler...");

    // Wait for compute handler on all workers
    let workers_ready = cohort
        .await_handler_on_all_workers("compute", Some(Duration::from_secs(30)))
        .await
        .is_ok();

    if !workers_ready {
        info!("Workers do not have compute handler - skipping broadcast");
    } else {
        let request = ComputeRequest {
            x: 10,
            y: 20,
            operation: "add".to_string(),
        };

        let payload = Bytes::from(serde_json::to_vec(&request)?);

        info!("Broadcasting compute request to workers by rank");
        cohort.broadcast_by_rank("compute", payload).await?;
    }

    tokio::time::sleep(Duration::from_secs(2)).await;

    info!("Leader initiating graceful shutdown...");
    cohort.initiate_graceful_shutdown().await?;

    info!("Leader shutting down");
    manager.shutdown().await?;
    cancel_token.cancel();

    Ok(())
}

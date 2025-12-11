// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active message communication for scheduler leader.
//!
//! This module handles leader-side communication, including:
//! - Creating and managing the active message manager
//! - Accepting worker registrations via join_cohort
//! - Managing the leader-worker cohort

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use dynamo_am::{
    ActiveMessageClient, InstanceId,
    cohort::{CohortType, LeaderWorkerCohort},
    manager::ActiveMessageManager,
    zmq::ZmqActiveMessageManager,
};

/// Leader communication state and cohort management.
pub struct LeaderCommsState {
    /// Active message manager for the leader
    manager: Option<Arc<ZmqActiveMessageManager>>,
    /// Active message client
    client: Option<Arc<dyn ActiveMessageClient>>,
    /// Leader endpoint address
    leader_address: Option<String>,
    /// Leader-worker cohort (once created)
    cohort: Arc<RwLock<Option<Arc<LeaderWorkerCohort>>>>,
    /// Expected number of workers
    expected_workers: usize,
    /// Cancellation token for graceful shutdown
    cancel_token: CancellationToken,
}

impl LeaderCommsState {
    /// Create a new leader comms state.
    ///
    /// Reads DYN_SCHEDULER_LEADER_ADDRESS from environment to determine bind address.
    /// If not set, leader will not initialize active message communication.
    ///
    /// Also reads DYN_SCHEDULER_EXPECTED_WORKERS for cohort size (defaults to 1).
    pub fn new() -> Self {
        let leader_address = std::env::var("DYN_SCHEDULER_LEADER_ADDRESS").ok();
        let expected_workers = std::env::var("DYN_SCHEDULER_EXPECTED_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1);

        if let Some(ref addr) = leader_address {
            info!(
                "Leader comms: Will bind to {} and expect {} workers",
                addr, expected_workers
            );
        } else {
            debug!("Leader comms: DYN_SCHEDULER_LEADER_ADDRESS not set, operating without AM");
        }

        Self {
            manager: None,
            client: None,
            leader_address,
            cohort: Arc::new(RwLock::new(None)),
            expected_workers,
            cancel_token: CancellationToken::new(),
        }
    }

    /// Initialize the leader's active message manager and start listening.
    ///
    /// This must be called before workers can connect.
    pub async fn initialize(&mut self) -> Result<()> {
        let Some(addr) = self.leader_address.clone() else {
            debug!("Leader comms: No address configured, skipping initialization");
            return Ok(());
        };

        // Create AM manager for the leader
        let manager = ZmqActiveMessageManager::new(addr.clone(), self.cancel_token.clone())
            .await
            .context("Failed to create leader AM manager")?;

        let client = manager.client();
        info!("Leader comms: Listening on {}", client.endpoint());
        info!("Leader comms: Instance ID: {}", client.instance_id());

        // Store manager and client
        self.manager = Some(Arc::new(manager));
        let client_arc = client;
        self.client = Some(client_arc.clone());

        // Create cohort with expected number of workers
        // The cohort will automatically handle join_cohort calls from workers
        let cohort_type = CohortType::FixedSize(self.expected_workers);
        let cohort = Arc::new(LeaderWorkerCohort::new(
            client_arc as Arc<dyn ActiveMessageClient>,
            cohort_type,
        ));

        *self.cohort.write().await = Some(cohort);

        info!(
            "Leader comms: Cohort created, waiting for {} workers",
            self.expected_workers
        );

        Ok(())
    }

    /// Check if the cohort is complete (all expected workers have joined).
    pub async fn is_cohort_complete(&self) -> bool {
        if let Some(cohort) = self.cohort.read().await.as_ref() {
            cohort.is_cohort_complete().await
        } else {
            false
        }
    }

    /// Get the current cohort size.
    pub async fn cohort_size(&self) -> usize {
        if let Some(cohort) = self.cohort.read().await.as_ref() {
            cohort.worker_count().await
        } else {
            0
        }
    }

    /// Wait for all expected workers to join the cohort.
    ///
    /// Returns when the cohort is complete or timeout occurs.
    pub async fn wait_for_cohort_complete(&self, timeout: Duration) -> Result<()> {
        let cohort = self
            .cohort
            .read()
            .await
            .clone()
            .context("Cohort not initialized")?;

        info!("Leader comms: Waiting for cohort to complete...");

        let start = std::time::Instant::now();
        while !cohort.is_cohort_complete().await {
            if start.elapsed() > timeout {
                return Err(anyhow::anyhow!(
                    "Timeout waiting for cohort completion ({} / {} workers)",
                    cohort.worker_count().await,
                    self.expected_workers
                ));
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        info!(
            "Leader comms: Cohort complete with {} workers",
            cohort.worker_count().await
        );

        Ok(())
    }

    /// Get the leader's instance ID.
    pub fn instance_id(&self) -> Option<InstanceId> {
        self.client.as_ref().map(|c| c.instance_id())
    }

    /// Get the leader's endpoint address.
    pub fn endpoint(&self) -> Option<String> {
        self.client.as_ref().map(|c| c.endpoint().to_string())
    }

    /// Shutdown the leader communication.
    pub async fn shutdown(&self) -> Result<()> {
        self.cancel_token.cancel();

        if let Some(manager) = &self.manager {
            manager
                .shutdown()
                .await
                .context("Failed to shutdown AM manager")?;
        }

        Ok(())
    }
}

impl Default for LeaderCommsState {
    fn default() -> Self {
        Self::new()
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use derive_builder::Builder;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::active_message::{
    client::{ActiveMessageClient, PeerInfo},
    handler::InstanceId,
};

use super::client::ZmqActiveMessageClient;

#[derive(Debug, Clone, PartialEq)]
pub enum CohortType {
    FixedSize(usize), // Maximum number of workers
                      // Unbounded variant excluded per requirements
}

#[derive(Debug, Clone, PartialEq)]
pub enum CohortFailurePolicy {
    TerminateAll, // If any member fails, terminate entire cohort
                  // Other policies can be added later
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    instance_id: InstanceId,
    rank: Option<usize>, // Worker-reported rank (e.g., from torch.distributed)
    joined_at: Instant,
    last_heartbeat: Arc<RwLock<Instant>>,
}

impl WorkerInfo {
    pub fn new(instance_id: InstanceId, rank: Option<usize>) -> Self {
        let now = Instant::now();
        Self {
            instance_id,
            rank,
            joined_at: now,
            last_heartbeat: Arc::new(RwLock::new(now)),
        }
    }
}

#[derive(Debug)]
struct CohortState {
    leader_instance: InstanceId,
    workers: HashMap<InstanceId, WorkerInfo>,
    workers_by_rank: BTreeMap<usize, InstanceId>, // For rank-ordered operations
    cohort_type: CohortType,
    failure_policy: CohortFailurePolicy,
    registration_closed: bool,
    requires_ranks: Option<bool>, // None=unknown, Some(true)=all must have ranks, Some(false)=none have ranks
    client: Arc<ZmqActiveMessageClient>,
    cancel_token: CancellationToken,
    heartbeat_interval: Duration,
    heartbeat_timeout: Duration,
}

#[derive(Builder, Debug)]
#[builder(setter(into))]
pub struct LeaderWorkerCohortConfig {
    leader_instance: InstanceId,
    client: Arc<ZmqActiveMessageClient>,
    cohort_type: CohortType,

    #[builder(default = "CohortFailurePolicy::TerminateAll")]
    failure_policy: CohortFailurePolicy,

    #[builder(default = "Duration::from_secs(30)")] // 30 second heartbeat
    heartbeat_interval: Duration,

    #[builder(default = "Duration::from_secs(120)")] // 2 minute timeout (4 missed heartbeats)
    heartbeat_timeout: Duration,

    #[builder(default = "CancellationToken::new()")]
    cancel_token: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct LeaderWorkerCohort {
    state: Arc<RwLock<CohortState>>,
}

impl LeaderWorkerCohort {
    // Builder pattern constructor
    pub fn builder() -> LeaderWorkerCohortConfigBuilder {
        LeaderWorkerCohortConfigBuilder::default()
    }

    // Create from config
    pub fn from_config(config: LeaderWorkerCohortConfig) -> Self {
        let state = CohortState {
            leader_instance: config.leader_instance,
            workers: HashMap::new(),
            workers_by_rank: BTreeMap::new(),
            cohort_type: config.cohort_type,
            failure_policy: config.failure_policy,
            registration_closed: false,
            requires_ranks: None,
            client: config.client,
            cancel_token: config.cancel_token,
            heartbeat_interval: config.heartbeat_interval,
            heartbeat_timeout: config.heartbeat_timeout,
        };

        Self {
            state: Arc::new(RwLock::new(state)),
        }
    }

    // Legacy constructor for backward compatibility
    #[deprecated(note = "Use builder() pattern instead")]
    pub fn new(
        leader_instance: InstanceId,
        worker_instances: Vec<InstanceId>,
        client: Arc<ZmqActiveMessageClient>,
    ) -> Self {
        let config = LeaderWorkerCohortConfig {
            leader_instance,
            client,
            cohort_type: CohortType::FixedSize(worker_instances.len()),
            failure_policy: CohortFailurePolicy::TerminateAll,
            heartbeat_interval: Duration::from_secs(30),
            heartbeat_timeout: Duration::from_secs(120),
            cancel_token: CancellationToken::new(),
        };

        let cohort = Self::from_config(config);

        // Add the pre-specified workers for backward compatibility
        tokio::spawn({
            let cohort = cohort.clone();
            async move {
                for worker_id in worker_instances {
                    let _ = cohort.add_worker(worker_id, None).await;
                }
            }
        });

        cohort
    }

    // Worker management methods
    pub async fn validate_and_add_worker(
        &self,
        worker_id: InstanceId,
        rank: Option<usize>,
    ) -> Result<usize> {
        let mut state = self.state.write().await;

        // Check cohort size
        match state.cohort_type {
            CohortType::FixedSize(max) if state.workers.len() >= max => {
                return Err(anyhow::anyhow!(
                    "Cohort is full ({}/{} workers)",
                    state.workers.len(),
                    max
                ));
            }
            _ => {}
        }

        // Validate rank consistency
        match (state.requires_ranks, rank) {
            (None, Some(_)) => {
                // First worker has rank, all must have ranks
                state.requires_ranks = Some(true);
            }
            (None, None) => {
                // First worker has no rank, none should have ranks
                state.requires_ranks = Some(false);
            }
            (Some(true), None) => {
                return Err(anyhow::anyhow!(
                    "All workers must report ranks (torch.distributed or MPI)"
                ));
            }
            (Some(false), Some(_)) => {
                return Err(anyhow::anyhow!(
                    "No workers should report ranks (first worker didn't)"
                ));
            }
            (Some(true), Some(r)) => {
                // Check for rank conflicts
                if state.workers_by_rank.contains_key(&r) {
                    return Err(anyhow::anyhow!(
                        "Rank {} already taken by another worker",
                        r
                    ));
                }
                // Check rank is within expected range
                let CohortType::FixedSize(max) = state.cohort_type;
                if r >= max {
                    return Err(anyhow::anyhow!(
                        "Rank {} exceeds max cohort size {}",
                        r,
                        max
                    ));
                }
            }
            _ => {} // Valid combinations
        }

        // Add worker
        let position = state.workers.len();
        let info = WorkerInfo::new(worker_id, rank);

        if let Some(r) = rank {
            state.workers_by_rank.insert(r, worker_id);
        }
        state.workers.insert(worker_id, info);

        // Check if cohort is complete and validate rank contiguity when needed
        let CohortType::FixedSize(max_size) = state.cohort_type;
        let is_cohort_full = state.workers.len() == max_size;

        if is_cohort_full && state.requires_ranks == Some(true) {
            // Validate contiguous rank assignment now that cohort is full
            if state.workers_by_rank.len() != max_size {
                // Remove the just-added worker before returning error
                state.workers.remove(&worker_id);
                if let Some(r) = rank {
                    state.workers_by_rank.remove(&r);
                }
                return Err(anyhow::anyhow!(
                    "Incomplete rank assignment: have {} ranks, need {} for full cohort",
                    state.workers_by_rank.len(),
                    max_size
                ));
            }

            // Check all ranks [0, max_size) are present
            for expected_rank in 0..max_size {
                if !state.workers_by_rank.contains_key(&expected_rank) {
                    // Remove the just-added worker before returning error
                    state.workers.remove(&worker_id);
                    if let Some(r) = rank {
                        state.workers_by_rank.remove(&r);
                    }
                    return Err(anyhow::anyhow!(
                        "Missing rank {} in contiguous sequence [0, {}). Cannot complete cohort.",
                        expected_rank,
                        max_size
                    ));
                }
            }

            info!("Cohort is complete with contiguous ranks [0, {})", max_size);
        }

        info!(
            "Added worker {} to cohort (rank: {:?}, position: {})",
            worker_id, rank, position
        );

        Ok(position)
    }

    // Add worker (wrapper for validate_and_add_worker)
    pub async fn add_worker(&self, worker_id: InstanceId, rank: Option<usize>) -> Result<usize> {
        self.validate_and_add_worker(worker_id, rank).await
    }

    // Remove worker (called by RemoveServiceHandler)
    pub async fn remove_worker(&self, worker_id: InstanceId) -> Result<bool> {
        let mut state = self.state.write().await;

        if let Some(info) = state.workers.remove(&worker_id) {
            if let Some(rank) = info.rank {
                state.workers_by_rank.remove(&rank);
            }
            info!(
                "Removed worker {} from cohort (rank: {:?})",
                worker_id, info.rank
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // Check if cohort is full
    pub async fn is_full(&self) -> bool {
        let state = self.state.read().await;
        match state.cohort_type {
            CohortType::FixedSize(max) => state.workers.len() >= max,
        }
    }

    // Get worker count
    pub async fn worker_count(&self) -> usize {
        let state = self.state.read().await;
        state.workers.len()
    }

    // Check if cohort uses ranks
    pub async fn has_ranks(&self) -> bool {
        let state = self.state.read().await;
        state.requires_ranks == Some(true)
    }

    // Get workers in rank order (if ranks are used)
    pub async fn get_workers_by_rank(&self) -> Vec<(usize, InstanceId)> {
        let state = self.state.read().await;
        state
            .workers_by_rank
            .iter()
            .map(|(rank, id)| (*rank, *id))
            .collect()
    }

    // Get all workers as list of IDs
    pub async fn get_worker_ids(&self) -> Vec<InstanceId> {
        let state = self.state.read().await;
        state.workers.keys().cloned().collect()
    }

    // Update worker heartbeat
    pub async fn update_worker_heartbeat(&self, worker_id: InstanceId) -> Result<()> {
        let state = self.state.read().await;
        if let Some(info) = state.workers.get(&worker_id) {
            *info.last_heartbeat.write().await = Instant::now();
            Ok(())
        } else {
            Err(anyhow::anyhow!("Worker {} not found in cohort", worker_id))
        }
    }

    // Check if all ranks form contiguous sequence [0, N) when ranks are required
    pub async fn validate_ranks_complete(&self) -> Result<()> {
        let state = self.state.read().await;

        // Only validate if ranks are required
        if state.requires_ranks != Some(true) {
            return Ok(());
        }

        let CohortType::FixedSize(max_size) = state.cohort_type;

        // Check that we have exactly the right number of ranks
        if state.workers_by_rank.len() != max_size {
            return Err(anyhow::anyhow!(
                "Incomplete rank assignment: have {} ranks, need {}",
                state.workers_by_rank.len(),
                max_size
            ));
        }

        // Check that all ranks [0, max_size) are present
        for expected_rank in 0..max_size {
            if !state.workers_by_rank.contains_key(&expected_rank) {
                return Err(anyhow::anyhow!(
                    "Missing rank {} in contiguous sequence [0, {})",
                    expected_rank,
                    max_size
                ));
            }
        }

        Ok(())
    }

    // Get missing ranks in the sequence [0, N)
    pub async fn get_missing_ranks(&self) -> Vec<usize> {
        let state = self.state.read().await;

        // Only applies when ranks are required
        if state.requires_ranks != Some(true) {
            return vec![];
        }

        let CohortType::FixedSize(max_size) = state.cohort_type;
        let mut missing = Vec::new();

        for expected_rank in 0..max_size {
            if !state.workers_by_rank.contains_key(&expected_rank) {
                missing.push(expected_rank);
            }
        }

        missing
    }

    // Check if cohort is complete (size + ranks if required)
    pub async fn is_cohort_complete(&self) -> bool {
        let state = self.state.read().await;
        let CohortType::FixedSize(max_size) = state.cohort_type;

        // Must have correct number of workers
        if state.workers.len() != max_size {
            return false;
        }

        // If ranks are required, must have contiguous sequence
        if state.requires_ranks == Some(true) {
            if state.workers_by_rank.len() != max_size {
                return false;
            }

            // Check all ranks [0, max_size) are present
            for expected_rank in 0..max_size {
                if !state.workers_by_rank.contains_key(&expected_rank) {
                    return false;
                }
            }
        }

        true
    }

    pub async fn await_handler_on_all_workers(
        &self,
        handler: &str,
        timeout: Option<Duration>,
    ) -> Result<()> {
        let state = self.state.read().await;
        let workers: Vec<InstanceId> = state.workers.keys().cloned().collect();
        let client = state.client.clone();
        drop(state);

        let mut tasks = Vec::new();

        for worker_id in workers {
            let client = client.clone();
            let handler = handler.to_string();

            tasks.push(tokio::spawn(async move {
                client.await_handler(worker_id, &handler, timeout).await
            }));
        }

        for (idx, task) in tasks.into_iter().enumerate() {
            match task.await {
                Ok(Ok(true)) => {
                    debug!("Worker {} has handler '{}'", idx, handler);
                }
                Ok(Ok(false)) => {
                    warn!("Worker {} does not have handler '{}'", idx, handler);
                }
                Ok(Err(e)) => {
                    warn!(
                        "Worker {} failed to confirm handler '{}': {}",
                        idx, handler, e
                    );
                    return Err(e);
                }
                Err(e) => {
                    warn!("Worker {} task panicked: {}", idx, e);
                    return Err(anyhow::anyhow!("Worker task panicked: {}", e));
                }
            }
        }

        Ok(())
    }

    pub async fn broadcast_to_workers(&self, handler: &str, payload: Bytes) -> Result<()> {
        let state = self.state.read().await;
        let workers: Vec<InstanceId> = state.workers.keys().cloned().collect();
        let client = state.client.clone();
        drop(state);

        for worker_id in workers {
            if let Err(e) = client
                .send_message(worker_id, handler, payload.clone())
                .await
            {
                warn!("Failed to send to worker {}: {}", worker_id, e);
            }
        }

        Ok(())
    }

    // Broadcast to workers in rank order
    pub async fn broadcast_by_rank(&self, handler: &str, payload: Bytes) -> Result<()> {
        let workers = self.get_workers_by_rank().await;
        let state = self.state.read().await;
        let client = state.client.clone();
        drop(state);

        for (rank, worker_id) in workers {
            debug!("Sending to worker rank {}: {}", rank, worker_id);
            if let Err(e) = client
                .send_message(worker_id, handler, payload.clone())
                .await
            {
                warn!(
                    "Failed to send to worker {} (rank {}): {}",
                    worker_id, rank, e
                );
            }
        }

        Ok(())
    }

    pub async fn broadcast_to_workers_with_acks(
        &self,
        handler: &str,
        payload: Bytes,
        timeout: Duration,
    ) -> Result<()> {
        let state = self.state.read().await;
        let workers: Vec<InstanceId> = state.workers.keys().cloned().collect();
        let client = state.client.clone();
        drop(state);

        let mut ack_receivers = Vec::new();

        for worker_id in workers {
            let ack_id = Uuid::new_v4();
            let ack_rx = client.register_ack(ack_id, timeout).await?;
            ack_receivers.push((worker_id, ack_rx));

            let mut payload_with_ack = serde_json::from_slice::<serde_json::Value>(&payload)?;
            if let serde_json::Value::Object(ref mut map) = payload_with_ack {
                map.insert(
                    "_ack_id".to_string(),
                    serde_json::Value::String(ack_id.to_string()),
                );
            }

            let modified_payload = Bytes::from(serde_json::to_vec(&payload_with_ack)?);
            client
                .send_message(worker_id, handler, modified_payload)
                .await?;
        }

        for (worker_id, ack_rx) in ack_receivers {
            match tokio::time::timeout(timeout, ack_rx).await {
                Ok(Ok(Ok(()))) => {
                    debug!("Received ACK from worker {}", worker_id);
                }
                Ok(Ok(Err(error))) => {
                    warn!("Received NACK from worker {}: {}", worker_id, error);
                }
                Ok(Err(_)) => {
                    warn!("ACK channel closed for worker {}", worker_id);
                }
                Err(_) => {
                    warn!("Timeout waiting for ACK from worker {}", worker_id);
                }
            }
        }

        Ok(())
    }

    pub async fn list_workers(&self) -> Vec<InstanceId> {
        let state = self.state.read().await;
        state.workers.keys().cloned().collect()
    }

    pub fn leader_instance(&self) -> InstanceId {
        let state = self.state.blocking_read();
        state.leader_instance
    }

    // Start monitoring (called after creation)
    pub async fn start_monitoring(self: Arc<Self>) {
        tokio::spawn(self.clone().heartbeat_monitor_loop());
    }

    // Heartbeat monitoring loop for leader
    async fn heartbeat_monitor_loop(self: Arc<Self>) {
        let state = self.state.read().await;
        let heartbeat_interval = state.heartbeat_interval;
        let heartbeat_timeout = state.heartbeat_timeout;
        let cancel_token = state.cancel_token.clone();
        drop(state);

        let mut interval = tokio::time::interval(heartbeat_interval); // 30 seconds

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let now = Instant::now();
                    let mut failed_workers = Vec::new();

                    // Check all worker heartbeats
                    let state = self.state.read().await;
                    for (worker_id, info) in &state.workers {
                        let last_hb = *info.last_heartbeat.read().await;

                        if now.duration_since(last_hb) > heartbeat_timeout { // 2 minutes
                            warn!("Worker {} (rank: {:?}) missed 4 heartbeats, marking as failed",
                                  worker_id, info.rank);
                            failed_workers.push(*worker_id);
                        } else {
                            // Send health check
                            tokio::spawn({
                                let client = state.client.clone();
                                let worker_id = *worker_id;
                                let last_heartbeat = info.last_heartbeat.clone();
                                async move {
                                    if client.health_check(worker_id).await.is_ok() {
                                        *last_heartbeat.write().await = Instant::now();
                                    }
                                }
                            });
                        }
                    }
                    drop(state);

                    // Handle failures
                    if let Some(failed_worker) = failed_workers.into_iter().next() {
                        self.handle_worker_failure(failed_worker).await;
                        return; // Exit loop after handling failure
                    }
                }
                _ = cancel_token.cancelled() => {
                    info!("Heartbeat monitor cancelled");
                    break;
                }
            }
        }
    }

    // Handle worker failure
    async fn handle_worker_failure(&self, failed_worker: InstanceId) {
        let state = self.state.read().await;
        let failure_policy = state.failure_policy.clone();
        drop(state);

        match failure_policy {
            CohortFailurePolicy::TerminateAll => {
                error!(
                    "Worker {} failed, initiating cohort termination",
                    failed_worker
                );
                // Initiate graceful shutdown of all workers
                if let Err(e) = self.initiate_graceful_shutdown().await {
                    error!("Failed to initiate graceful shutdown: {}", e);
                }
                // Cancel own token
                let state = self.state.read().await;
                state.cancel_token.cancel();
            }
        }
    }

    // Graceful shutdown flow
    pub async fn initiate_graceful_shutdown(&self) -> Result<()> {
        info!("Initiating graceful cohort shutdown");

        let state = self.state.read().await;
        let client = state.client.clone();

        // Send shutdown requests in rank order (if ranks exist)
        let workers = if state.requires_ranks == Some(true) {
            drop(state);
            self.get_workers_by_rank()
                .await
                .into_iter()
                .map(|(_, id)| id)
                .collect()
        } else {
            let workers: Vec<InstanceId> = state.workers.keys().cloned().collect();
            drop(state);
            workers
        };

        // 1. Send _request_shutdown to all workers
        for worker_id in &workers {
            debug!("Sending shutdown request to worker {}", worker_id);
            if let Err(e) = client
                .system_message("_request_shutdown")
                .fire_and_forget(*worker_id)
                .await
            {
                warn!(
                    "Failed to send shutdown request to worker {}: {}",
                    worker_id, e
                );
            }
        }

        // 2. Wait for all workers to remove their service
        let timeout = Duration::from_secs(60); // Give workers time to drain
        let deadline = Instant::now() + timeout;

        while !self.state.read().await.workers.is_empty() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        let remaining = self.state.read().await.workers.len();
        if remaining > 0 {
            warn!(
                "{} workers did not shutdown gracefully within timeout",
                remaining
            );
        } else {
            info!("All workers shutdown gracefully");
        }

        Ok(())
    }
}

// Worker-side heartbeat monitoring function (to be called by worker)
pub async fn monitor_leader_heartbeat(
    leader_id: InstanceId,
    client: Arc<dyn ActiveMessageClient>,
    cancel_token: CancellationToken,
    worker_rank: Option<usize>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    let mut consecutive_failures = 0;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                match client.send_message(leader_id, "_health_check", Bytes::new()).await {
                    Ok(_) => {
                        debug!("Leader heartbeat successful (rank: {:?})", worker_rank);
                        consecutive_failures = 0;
                    }
                    Err(e) => {
                        consecutive_failures += 1;
                        warn!("Leader heartbeat failed ({}/4): {}", consecutive_failures, e);

                        if consecutive_failures >= 4 {
                            error!("Leader failed after 4 heartbeat attempts (2 minutes), shutting down");
                            cancel_token.cancel();
                            return;
                        }
                    }
                }
            }
            _ = cancel_token.cancelled() => {
                info!("Leader monitor cancelled");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_message::client::ActiveMessageClient;
    use std::{collections::HashMap, sync::Arc, time::Duration};
    use tokio::sync::oneshot;
    use uuid::Uuid;

    // Test utilities and helpers
    mod test_helpers {
        use super::*;
        use async_trait::async_trait;
        use bytes::Bytes;

        #[derive(Debug, Clone)]
        pub struct MockActiveMessageClient {
            pub instance_id: InstanceId,
            pub endpoint: String,
            pub sent_messages: Arc<tokio::sync::RwLock<Vec<(InstanceId, String, Bytes)>>>,
        }

        impl MockActiveMessageClient {
            pub fn new(instance_id: InstanceId, endpoint: String) -> Self {
                Self {
                    instance_id,
                    endpoint,
                    sent_messages: Arc::new(tokio::sync::RwLock::new(Vec::new())),
                }
            }

            pub async fn get_sent_messages(&self) -> Vec<(InstanceId, String, Bytes)> {
                self.sent_messages.read().await.clone()
            }
        }

        #[async_trait]
        impl ActiveMessageClient for MockActiveMessageClient {
            fn instance_id(&self) -> InstanceId {
                self.instance_id
            }

            fn endpoint(&self) -> &str {
                &self.endpoint
            }

            async fn send_message(
                &self,
                target: InstanceId,
                handler: &str,
                payload: Bytes,
            ) -> anyhow::Result<()> {
                self.sent_messages
                    .write()
                    .await
                    .push((target, handler.to_string(), payload));
                Ok(())
            }

            async fn broadcast_message(
                &self,
                _handler: &str,
                _payload: Bytes,
            ) -> anyhow::Result<()> {
                Ok(())
            }

            async fn list_peers(
                &self,
            ) -> anyhow::Result<Vec<crate::active_message::client::PeerInfo>> {
                Ok(vec![])
            }

            async fn connect_to_peer(
                &self,
                _peer: crate::active_message::client::PeerInfo,
            ) -> anyhow::Result<()> {
                Ok(())
            }

            async fn disconnect_from_peer(&self, _instance_id: InstanceId) -> anyhow::Result<()> {
                Ok(())
            }

            async fn await_handler(
                &self,
                _instance_id: InstanceId,
                _handler: &str,
                _timeout: Option<Duration>,
            ) -> anyhow::Result<bool> {
                Ok(true)
            }

            async fn list_handlers(&self, _instance_id: InstanceId) -> anyhow::Result<Vec<String>> {
                Ok(vec![])
            }

            async fn send_raw_message(
                &self,
                _target: InstanceId,
                _message: crate::active_message::handler::ActiveMessage,
            ) -> anyhow::Result<()> {
                Ok(())
            }

            async fn register_acceptance(
                &self,
                _message_id: Uuid,
                _sender: oneshot::Sender<()>,
            ) -> anyhow::Result<()> {
                Ok(())
            }

            async fn register_response(
                &self,
                _message_id: Uuid,
                _sender: oneshot::Sender<Bytes>,
            ) -> anyhow::Result<()> {
                Ok(())
            }

            async fn register_ack(
                &self,
                _ack_id: Uuid,
                _timeout: Duration,
            ) -> anyhow::Result<oneshot::Receiver<Result<(), String>>> {
                let (_tx, rx) = oneshot::channel();
                Ok(rx)
            }
        }

        pub fn test_cohort_builder() -> LeaderWorkerCohortConfig {
            let leader_id = Uuid::new_v4();
            let client = Arc::new(
                crate::active_message::zmq::client::ZmqActiveMessageClient::new(
                    leader_id,
                    "test://localhost:0".to_string(),
                ),
            );

            LeaderWorkerCohortConfig {
                leader_instance: leader_id,
                client,
                cohort_type: CohortType::FixedSize(3),
                failure_policy: CohortFailurePolicy::TerminateAll,
                heartbeat_interval: Duration::from_secs(30),
                heartbeat_timeout: Duration::from_secs(120),
                cancel_token: tokio_util::sync::CancellationToken::new(),
            }
        }

        pub fn create_mock_workers(
            count: usize,
            with_ranks: bool,
        ) -> Vec<(InstanceId, Option<usize>)> {
            (0..count)
                .map(|i| {
                    let id = Uuid::new_v4();
                    let rank = if with_ranks { Some(i) } else { None };
                    (id, rank)
                })
                .collect()
        }

        pub async fn assert_cohort_state(
            cohort: &LeaderWorkerCohort,
            expected_count: usize,
            expected_has_ranks: bool,
            expected_complete: bool,
        ) {
            assert_eq!(cohort.worker_count().await, expected_count);
            assert_eq!(cohort.has_ranks().await, expected_has_ranks);
            assert_eq!(cohort.is_cohort_complete().await, expected_complete);
        }
    }

    use test_helpers::*;

    // Test data structures for parameterized tests
    #[derive(Debug)]
    struct CohortTestCase {
        name: &'static str,
        cohort_size: usize,
        worker_ranks: Vec<Option<usize>>,
        expected_result: ExpectedResult,
    }

    #[derive(Debug)]
    enum ExpectedResult {
        Success { worker_count: usize },
        RankError(&'static str),
        CohortFull,
        MissingRank(usize),
    }

    // Test case data for rank validation
    fn get_rank_test_cases() -> Vec<CohortTestCase> {
        vec![
            CohortTestCase {
                name: "valid_contiguous_ranks",
                cohort_size: 3,
                worker_ranks: vec![Some(0), Some(1), Some(2)],
                expected_result: ExpectedResult::Success { worker_count: 3 },
            },
            CohortTestCase {
                name: "valid_no_ranks",
                cohort_size: 3,
                worker_ranks: vec![None, None, None],
                expected_result: ExpectedResult::Success { worker_count: 3 },
            },
            CohortTestCase {
                name: "missing_middle_rank",
                cohort_size: 3,
                worker_ranks: vec![Some(0), Some(2)],
                expected_result: ExpectedResult::MissingRank(1),
            },
            CohortTestCase {
                name: "missing_first_rank",
                cohort_size: 3,
                worker_ranks: vec![Some(1), Some(2)],
                expected_result: ExpectedResult::MissingRank(0),
            },
            CohortTestCase {
                name: "duplicate_rank",
                cohort_size: 3,
                worker_ranks: vec![Some(0), Some(0)],
                expected_result: ExpectedResult::RankError("already taken"),
            },
            CohortTestCase {
                name: "rank_exceeds_size",
                cohort_size: 3,
                worker_ranks: vec![Some(0), Some(3)],
                expected_result: ExpectedResult::RankError("exceeds max cohort size"),
            },
            CohortTestCase {
                name: "mixed_rank_presence",
                cohort_size: 3,
                worker_ranks: vec![Some(0), None],
                expected_result: ExpectedResult::RankError("must report ranks"),
            },
        ]
    }

    // Helper function to run parameterized test cases
    async fn run_cohort_test_case(test_case: &CohortTestCase) {
        let mut config = test_cohort_builder();
        config.cohort_type = CohortType::FixedSize(test_case.cohort_size);
        let cohort = LeaderWorkerCohort::from_config(config);

        let mut last_result = Ok(0);
        for (_idx, rank) in test_case.worker_ranks.iter().enumerate() {
            let worker_id = Uuid::new_v4();
            last_result = cohort.validate_and_add_worker(worker_id, *rank).await;

            // If we expect an error on this specific addition, check it
            if let ExpectedResult::RankError(expected_msg) = &test_case.expected_result {
                if last_result.is_err() {
                    assert!(
                        last_result
                            .as_ref()
                            .unwrap_err()
                            .to_string()
                            .contains(expected_msg),
                        "Expected error containing '{}', got: {:?}",
                        expected_msg,
                        last_result
                    );
                    return; // Test passed
                }
            }
        }

        // Check final result
        match &test_case.expected_result {
            ExpectedResult::Success { worker_count } => {
                assert!(
                    last_result.is_ok(),
                    "Expected success, got: {:?}",
                    last_result
                );
                assert_eq!(cohort.worker_count().await, *worker_count);
                assert!(cohort.is_cohort_complete().await);
            }
            ExpectedResult::MissingRank(missing) => {
                let missing_ranks = cohort.get_missing_ranks().await;
                assert!(
                    missing_ranks.contains(missing),
                    "Expected missing rank {}, got missing ranks: {:?}",
                    missing,
                    missing_ranks
                );
            }
            ExpectedResult::CohortFull => {
                assert!(last_result.is_err());
                assert!(
                    last_result
                        .unwrap_err()
                        .to_string()
                        .contains("Cohort is full")
                );
            }
            ExpectedResult::RankError(_) => {
                // Already handled above during iteration
            }
        }
    }

    // Basic functionality tests
    #[tokio::test]
    async fn test_cohort_creation_with_builder() {
        let config = test_cohort_builder();
        let cohort = LeaderWorkerCohort::from_config(config);

        assert_eq!(cohort.worker_count().await, 0);
        assert!(!cohort.has_ranks().await);
        assert!(!cohort.is_full().await);
        assert!(!cohort.is_cohort_complete().await);
    }

    #[tokio::test]
    async fn test_fixed_size_enforcement() {
        let mut config = test_cohort_builder();
        config.cohort_type = CohortType::FixedSize(2);
        let cohort = LeaderWorkerCohort::from_config(config);

        // Add first two workers
        let worker1 = Uuid::new_v4();
        let worker2 = Uuid::new_v4();
        let worker3 = Uuid::new_v4();

        assert!(cohort.validate_and_add_worker(worker1, None).await.is_ok());
        assert!(cohort.validate_and_add_worker(worker2, None).await.is_ok());

        // Third worker should be rejected
        let result = cohort.validate_and_add_worker(worker3, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cohort is full"));
    }

    #[tokio::test]
    async fn test_worker_addition_and_removal() {
        let config = test_cohort_builder();
        let cohort = LeaderWorkerCohort::from_config(config);

        let worker_id = Uuid::new_v4();

        // Add worker
        let position = cohort
            .validate_and_add_worker(worker_id, Some(0))
            .await
            .unwrap();
        assert_eq!(position, 0);
        assert_eq!(cohort.worker_count().await, 1);
        assert!(cohort.has_ranks().await);

        // Remove worker
        let removed = cohort.remove_worker(worker_id).await.unwrap();
        assert!(removed);
        assert_eq!(cohort.worker_count().await, 0);

        // Remove non-existent worker
        let removed = cohort.remove_worker(Uuid::new_v4()).await.unwrap();
        assert!(!removed);
    }

    // Individual rank validation tests (avoiding macro complexity)
    #[tokio::test]
    async fn test_rank_validation_valid_contiguous_ranks() {
        let test_cases = get_rank_test_cases();
        let test_case = &test_cases[0]; // valid_contiguous_ranks
        run_cohort_test_case(test_case).await;
    }

    #[tokio::test]
    async fn test_rank_validation_valid_no_ranks() {
        let test_cases = get_rank_test_cases();
        let test_case = &test_cases[1]; // valid_no_ranks
        run_cohort_test_case(test_case).await;
    }

    #[tokio::test]
    async fn test_rank_validation_missing_middle_rank() {
        let test_cases = get_rank_test_cases();
        let test_case = &test_cases[2]; // missing_middle_rank
        run_cohort_test_case(test_case).await;
    }

    #[tokio::test]
    async fn test_rank_validation_missing_first_rank() {
        let test_cases = get_rank_test_cases();
        let test_case = &test_cases[3]; // missing_first_rank
        run_cohort_test_case(test_case).await;
    }

    #[tokio::test]
    async fn test_rank_validation_duplicate_rank() {
        let test_cases = get_rank_test_cases();
        let test_case = &test_cases[4]; // duplicate_rank
        run_cohort_test_case(test_case).await;
    }

    #[tokio::test]
    async fn test_rank_validation_rank_exceeds_size() {
        let test_cases = get_rank_test_cases();
        let test_case = &test_cases[5]; // rank_exceeds_size
        run_cohort_test_case(test_case).await;
    }

    #[tokio::test]
    async fn test_rank_validation_mixed_rank_presence() {
        let test_cases = get_rank_test_cases();
        let test_case = &test_cases[6]; // mixed_rank_presence
        run_cohort_test_case(test_case).await;
    }

    #[tokio::test]
    async fn test_is_full_vs_is_complete() {
        let mut config = test_cohort_builder();
        config.cohort_type = CohortType::FixedSize(2);
        let cohort = LeaderWorkerCohort::from_config(config);

        // Add workers with missing rank
        let worker1 = Uuid::new_v4();
        let worker2 = Uuid::new_v4();

        // Both have ranks but not contiguous
        cohort
            .validate_and_add_worker(worker1, Some(0))
            .await
            .unwrap();
        cohort
            .validate_and_add_worker(worker2, Some(1))
            .await
            .unwrap();

        assert!(cohort.is_full().await);
        assert!(cohort.is_cohort_complete().await); // Should be complete with ranks [0, 1]
    }

    #[tokio::test]
    async fn test_get_missing_ranks() {
        let mut config = test_cohort_builder();
        config.cohort_type = CohortType::FixedSize(4);
        let cohort = LeaderWorkerCohort::from_config(config);

        // Add workers with ranks 0 and 2, missing 1 and 3
        let worker1 = Uuid::new_v4();
        let worker2 = Uuid::new_v4();

        cohort
            .validate_and_add_worker(worker1, Some(0))
            .await
            .unwrap();
        cohort
            .validate_and_add_worker(worker2, Some(2))
            .await
            .unwrap();

        let missing = cohort.get_missing_ranks().await;
        assert_eq!(missing, vec![1, 3]);
    }

    #[tokio::test]
    async fn test_workers_by_rank_ordering() {
        let config = test_cohort_builder();
        let cohort = LeaderWorkerCohort::from_config(config);

        let workers = create_mock_workers(3, true);

        // Add workers in non-sequential order
        cohort
            .validate_and_add_worker(workers[2].0, workers[2].1)
            .await
            .unwrap(); // rank 2
        cohort
            .validate_and_add_worker(workers[0].0, workers[0].1)
            .await
            .unwrap(); // rank 0
        cohort
            .validate_and_add_worker(workers[1].0, workers[1].1)
            .await
            .unwrap(); // rank 1

        let by_rank = cohort.get_workers_by_rank().await;
        assert_eq!(by_rank.len(), 3);

        // Should be sorted by rank
        assert_eq!(by_rank[0].0, 0); // rank 0
        assert_eq!(by_rank[1].0, 1); // rank 1
        assert_eq!(by_rank[2].0, 2); // rank 2

        assert_eq!(by_rank[0].1, workers[0].0); // worker with rank 0
        assert_eq!(by_rank[1].1, workers[1].0); // worker with rank 1
        assert_eq!(by_rank[2].1, workers[2].0); // worker with rank 2
    }

    #[tokio::test]
    async fn test_heartbeat_update() {
        let config = test_cohort_builder();
        let cohort = LeaderWorkerCohort::from_config(config);

        let worker_id = Uuid::new_v4();

        // Should fail for non-existent worker
        let result = cohort.update_worker_heartbeat(worker_id).await;
        assert!(result.is_err());

        // Add worker and update heartbeat
        cohort
            .validate_and_add_worker(worker_id, None)
            .await
            .unwrap();
        let result = cohort.update_worker_heartbeat(worker_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_broadcasting() {
        let config = test_cohort_builder();
        let cohort = LeaderWorkerCohort::from_config(config);

        let workers = create_mock_workers(2, true);

        // Add workers
        for (worker_id, rank) in &workers {
            cohort
                .validate_and_add_worker(*worker_id, *rank)
                .await
                .unwrap();
        }

        let payload = Bytes::from("test_message");

        // Test broadcast to all workers
        let result = cohort
            .broadcast_to_workers("test_handler", payload.clone())
            .await;
        assert!(result.is_ok());

        // Test broadcast by rank
        let result = cohort.broadcast_by_rank("test_handler", payload).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cohort_completeness_validation() {
        let config = test_cohort_builder();
        let cohort = LeaderWorkerCohort::from_config(config);

        // Test validate_ranks_complete with no ranks required
        let result = cohort.validate_ranks_complete().await;
        assert!(result.is_ok());

        // Add workers with ranks
        let worker1 = Uuid::new_v4();
        let worker2 = Uuid::new_v4();

        cohort
            .validate_and_add_worker(worker1, Some(0))
            .await
            .unwrap();
        cohort
            .validate_and_add_worker(worker2, Some(2))
            .await
            .unwrap(); // Missing rank 1

        // Should fail validation due to incomplete rank assignment
        let result = cohort.validate_ranks_complete().await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Incomplete rank assignment: have 2 ranks, need 3")
        );

        // Add missing rank
        let worker3 = Uuid::new_v4();
        cohort
            .validate_and_add_worker(worker3, Some(1))
            .await
            .unwrap();

        // Should now pass validation
        let result = cohort.validate_ranks_complete().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_graceful_shutdown() {
        let config = test_cohort_builder();
        let cohort = LeaderWorkerCohort::from_config(config);

        let workers = create_mock_workers(2, false);

        // Add workers
        for (worker_id, rank) in &workers {
            cohort
                .validate_and_add_worker(*worker_id, *rank)
                .await
                .unwrap();
        }

        // Test graceful shutdown
        let result = cohort.initiate_graceful_shutdown().await;
        assert!(result.is_ok());
    }
}

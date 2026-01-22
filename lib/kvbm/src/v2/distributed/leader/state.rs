// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LeaderState - Coordination layer for managing workers.
//!
//! This module provides the leader's coordination state, including:
//! - Worker registration and rank mapping
//! - Remote leader tracking for cross-leader transfers
//! - Routing strategies for asymmetric TP configurations

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use dynamo_nova::Nova;

use crate::physical::manager::SerializedLayout;
use crate::v2::InstanceId;
use crate::v2::distributed::worker::{CoordinatedWorker, Worker};

/// Info about a remote leader and its workers.
#[derive(Debug)]
pub struct RemoteLeaderInfo {
    /// Instance ID of the remote leader process
    pub instance_id: InstanceId,
    /// Number of workers under the remote leader
    pub worker_count: usize,
    /// Cached metadata from remote workers (rank-ordered)
    pub worker_metadata: Vec<SerializedLayout>,
}

/// Leader coordination state - owns workers and routing logic.
///
/// LeaderState manages:
/// - Registration of workers during handshake phase
/// - Coordination with remote leaders for cross-leader transfers
/// - Routing strategies for asymmetric TP configurations
pub struct LeaderState {
    /// This leader's instance ID
    instance_id: InstanceId,

    /// Nova runtime for RPC
    nova: Arc<Nova>,

    /// Workers under this leader (rank-ordered)
    workers: Vec<CoordinatedWorker>,

    /// Known remote leaders (by their instance ID)
    remote_leaders: RwLock<HashMap<InstanceId, RemoteLeaderInfo>>,
}

impl LeaderState {
    /// Create a new LeaderState.
    ///
    /// # Arguments
    /// * `instance_id` - This leader's unique identifier
    /// * `nova` - Nova runtime for RPC communication
    pub fn new(instance_id: InstanceId, nova: Arc<Nova>) -> Self {
        Self {
            instance_id,
            nova,
            workers: Vec::new(),
            remote_leaders: RwLock::new(HashMap::new()),
        }
    }

    /// Get this leader's instance ID.
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// Get the Nova runtime.
    pub fn nova(&self) -> &Arc<Nova> {
        &self.nova
    }

    /// Register a worker during the handshake phase.
    ///
    /// Workers should be registered in rank order (0, 1, 2, ...).
    ///
    /// # Arguments
    /// * `rank` - The worker's rank (0-indexed)
    /// * `host_instance` - Instance ID of the process hosting this worker
    /// * `worker` - The Worker implementation (DirectWorker or NovaWorkerClient)
    pub fn register_worker(
        &mut self,
        rank: usize,
        host_instance: InstanceId,
        worker: Box<dyn Worker>,
    ) {
        let coordinated = CoordinatedWorker::new(worker, rank, host_instance);

        // Ensure rank-ordered insertion
        if rank >= self.workers.len() {
            // Extend with placeholder capacity
            self.workers.resize_with(rank + 1, || {
                panic!("Gap in worker ranks - this shouldn't happen")
            });
        }
        self.workers[rank] = coordinated;
    }

    /// Number of workers under this leader.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Get a worker by rank.
    pub fn worker(&self, rank: usize) -> Option<&CoordinatedWorker> {
        self.workers.get(rank)
    }

    /// Get a mutable worker by rank.
    pub fn worker_mut(&mut self, rank: usize) -> Option<&mut CoordinatedWorker> {
        self.workers.get_mut(rank)
    }

    /// Iterate over all workers.
    pub fn workers(&self) -> impl Iterator<Item = &CoordinatedWorker> {
        self.workers.iter()
    }

    /// Connect to a remote leader and distribute its worker metadata to our workers.
    ///
    /// This implements the routing strategy for cross-leader transfers:
    /// - 1:1 mapping when TP sizes match
    /// - Many-to-one when local TP > remote TP
    /// - One-to-many when local TP < remote TP
    ///
    /// # Arguments
    /// * `remote_leader_id` - Instance ID of the remote leader
    /// * `remote_metadata` - Metadata from each remote worker (rank-ordered)
    pub async fn import_remote_leader(
        &self,
        remote_leader_id: InstanceId,
        remote_metadata: Vec<SerializedLayout>,
    ) -> Result<()> {
        let remote_count = remote_metadata.len();
        let local_count = self.workers.len();

        tracing::info!(
            local_count,
            remote_count,
            %remote_leader_id,
            "Importing remote leader metadata"
        );

        // Store remote leader info
        {
            let mut leaders = self.remote_leaders.write().unwrap();
            leaders.insert(
                remote_leader_id,
                RemoteLeaderInfo {
                    instance_id: remote_leader_id,
                    worker_count: remote_count,
                    worker_metadata: remote_metadata.clone(),
                },
            );
        }

        // Distribute metadata based on routing strategy
        for (local_rank, worker) in self.workers.iter().enumerate() {
            let target_remote_ranks = route_local_to_remote(local_rank, local_count, remote_count);

            for remote_rank in target_remote_ranks {
                tracing::debug!(
                    local_rank,
                    remote_rank,
                    %remote_leader_id,
                    "Importing remote metadata for local worker"
                );

                worker
                    .import_remote_metadata(
                        remote_leader_id,
                        remote_rank,
                        remote_metadata[remote_rank].clone(),
                    )
                    .await?;
            }
        }

        Ok(())
    }

    /// Export this leader's workers' metadata for another leader to import.
    ///
    /// Returns metadata from each worker in rank order.
    pub async fn export_worker_metadata(&self) -> Result<Vec<SerializedLayout>> {
        let mut metadata = Vec::with_capacity(self.workers.len());

        for worker in &self.workers {
            let response = worker.inner().export_metadata()?;
            metadata.push(response.await?);
        }

        Ok(metadata)
    }

    /// Check if we have imported metadata from a remote leader.
    pub fn has_remote_leader(&self, remote_leader_id: InstanceId) -> bool {
        self.remote_leaders
            .read()
            .unwrap()
            .contains_key(&remote_leader_id)
    }

    /// Get info about a remote leader if known.
    pub fn remote_leader_info(&self, remote_leader_id: InstanceId) -> Option<RemoteLeaderInfo> {
        self.remote_leaders
            .read()
            .unwrap()
            .get(&remote_leader_id)
            .map(|info| RemoteLeaderInfo {
                instance_id: info.instance_id,
                worker_count: info.worker_count,
                worker_metadata: info.worker_metadata.clone(),
            })
    }
}

/// Routing strategy: which local ranks receive from which remote ranks.
///
/// This function determines how metadata/transfers are routed when
/// the local and remote TP sizes differ.
///
/// # Examples
/// - TP=4 local, TP=4 remote: 1:1 mapping (rank 0→0, 1→1, 2→2, 3→3)
/// - TP=4 local, TP=2 remote: 0→0, 1→0, 2→1, 3→1 (many-to-one)
/// - TP=2 local, TP=4 remote: 0→[0,1], 1→[2,3] (one-to-many)
pub fn route_local_to_remote(
    local_rank: usize,
    local_count: usize,
    remote_count: usize,
) -> Vec<usize> {
    if local_count == remote_count {
        // 1:1 mapping
        vec![local_rank]
    } else if local_count > remote_count {
        // Many local → few remote: multiple locals share a remote
        vec![local_rank % remote_count]
    } else {
        // Few local → many remote: each local gets multiple remotes
        let remotes_per_local = remote_count / local_count;
        let start = local_rank * remotes_per_local;
        (start..start + remotes_per_local).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_1_to_1() {
        // Same TP size
        assert_eq!(route_local_to_remote(0, 4, 4), vec![0]);
        assert_eq!(route_local_to_remote(1, 4, 4), vec![1]);
        assert_eq!(route_local_to_remote(2, 4, 4), vec![2]);
        assert_eq!(route_local_to_remote(3, 4, 4), vec![3]);
    }

    #[test]
    fn test_route_many_to_one() {
        // Local TP=4, Remote TP=2
        assert_eq!(route_local_to_remote(0, 4, 2), vec![0]);
        assert_eq!(route_local_to_remote(1, 4, 2), vec![1]);
        assert_eq!(route_local_to_remote(2, 4, 2), vec![0]);
        assert_eq!(route_local_to_remote(3, 4, 2), vec![1]);
    }

    #[test]
    fn test_route_one_to_many() {
        // Local TP=2, Remote TP=4
        assert_eq!(route_local_to_remote(0, 2, 4), vec![0, 1]);
        assert_eq!(route_local_to_remote(1, 2, 4), vec![2, 3]);
    }

    #[test]
    fn test_route_4_to_8() {
        // Local TP=4, Remote TP=8
        assert_eq!(route_local_to_remote(0, 4, 8), vec![0, 1]);
        assert_eq!(route_local_to_remote(1, 4, 8), vec![2, 3]);
        assert_eq!(route_local_to_remote(2, 4, 8), vec![4, 5]);
        assert_eq!(route_local_to_remote(3, 4, 8), vec![6, 7]);
    }
}

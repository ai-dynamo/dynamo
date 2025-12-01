// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-side service for coordinating with workers via Nova RPC.
//!
//! This service is separate from `KvbmRuntime` because the runtime is shared
//! infrastructure while this service provides leader-specific coordination operations.

use std::sync::Arc;

use anyhow::Result;
use dynamo_nova::Nova;

use crate::InstanceId;
use crate::v2::distributed::worker::{LeaderLayoutConfig, WorkerLayoutResponse};

/// Leader-side client for coordinating with workers via Nova RPC.
///
/// This client wraps a Nova instance and provides methods for leader-driven
/// initialization and coordination with workers. It sends RPCs to worker instances
/// during the initialization phase to configure their layouts.
///
/// # Example
///
/// ```ignore
/// let client = WorkerClient::new(nova);
///
/// // Send configure_layouts RPC to trigger deferred initialization
/// let response = client.configure_layouts(worker_instance_id, config).await?;
/// ```
pub struct WorkerClient {
    nova: Arc<Nova>,
}

impl WorkerClient {
    /// Create a new `WorkerClient` from a Nova instance.
    ///
    /// # Arguments
    /// * `nova` - The Nova instance for RPC communication
    pub fn new(nova: Arc<Nova>) -> Self {
        Self { nova }
    }

    /// Get a reference to the underlying Nova instance.
    pub fn nova(&self) -> &Arc<Nova> {
        &self.nova
    }

    /// Send a `configure_layouts` RPC to a worker to trigger deferred initialization.
    ///
    /// This is called by the leader after collecting handshake metadata from all workers.
    /// It triggers the worker to complete NIXL registration and create G1/G2/G3 layouts.
    ///
    /// # Arguments
    /// * `instance_id` - The target worker's instance ID
    /// * `config` - Configuration specifying G2/G3 block counts and backend options
    ///
    /// # Returns
    /// `WorkerLayoutResponse` containing the worker's metadata after initialization
    ///
    /// # Errors
    /// Returns an error if the RPC fails or the worker returns an error
    pub async fn configure_layouts(
        &self,
        instance_id: InstanceId,
        config: LeaderLayoutConfig,
    ) -> Result<WorkerLayoutResponse> {
        tracing::info!(
            target_instance = %instance_id,
            host_block_count = config.host_block_count,
            disk_block_count = ?config.disk_block_count,
            "Sending configure_layouts RPC to worker"
        );

        let response = self
            .nova
            .typed_unary::<WorkerLayoutResponse>("kvbm.connector.configure_layouts")?
            .payload(config)?
            .instance(instance_id)
            .send()
            .await?;

        tracing::info!(
            target_instance = %instance_id,
            created_layouts = ?response.created_layouts,
            "Worker completed initialization"
        );

        Ok(response)
    }
}

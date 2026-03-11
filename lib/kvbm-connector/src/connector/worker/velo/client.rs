// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client for calling Velo services registered on ConnectorWorker.

use anyhow::Result;
use velo::Messenger;
use std::sync::Arc;

use kvbm_physical::layout::LayoutConfig;
use kvbm_common::BlockId;
use kvbm_engine::InstanceId;
use kvbm_engine::worker::{LeaderLayoutConfig, WorkerLayoutResponse};

use super::*;

/// Client for communicating with a remote ConnectorWorker via Velo.
///
/// This client is generally used by the leader or a mock leader to communicate with the worker.
///
/// This client provides methods for:
/// - Triggering deferred initialization via `initialize()`
/// - Marking onboarding/offloading operations as complete
#[derive(Clone)]
pub struct ConnectorWorkerClient {
    messenger: Arc<Messenger>,
    remote: InstanceId,
}

impl ConnectorWorkerClient {
    /// Create a new ConnectorWorkerClient for communicating with a remote worker.
    ///
    /// # Arguments
    /// * `messenger` - Local Velo Messenger instance for sending messages
    /// * `remote` - Remote worker's instance ID
    pub fn new(messenger: Arc<Messenger>, remote: InstanceId) -> Self {
        Self { messenger, remote }
    }

    /// Initialize the remote worker with leader-provided configuration.
    ///
    /// This calls the `kvbm.connector.worker.initialize` handler on the remote worker.
    /// The worker will complete NIXL registration and create G2/G3 layouts based on
    /// the provided configuration.
    ///
    /// # Arguments
    /// * `config` - Leader-provided configuration specifying block counts and backends
    ///
    /// # Returns
    /// A typed unary result that resolves to the worker's response with updated metadata
    pub fn initialize(
        &self,
        config: LeaderLayoutConfig,
    ) -> Result<velo::TypedUnaryResult<WorkerLayoutResponse>> {
        let awaiter = self
            .messenger
            .typed_unary::<WorkerLayoutResponse>(INITIALIZE_HANDLER)?
            .payload(config)?
            .instance(self.remote)
            .send();

        Ok(awaiter)
    }

    /// Notify the remote worker that onboarding is complete for a request.
    ///
    /// This calls the `kvbm.connector.worker.onboard_complete` handler.
    /// The worker will add the request_id to its finished_onboarding set.
    ///
    /// # Arguments
    /// * `request_id` - The request that finished onboarding
    pub async fn mark_onboarding_complete(&self, request_id: String) -> Result<()> {
        let message = OnboardCompleteMessage { request_id };

        self.messenger
            .unary(ONBOARD_COMPLETE_HANDLER)?
            .payload(message)?
            .instance(self.remote)
            .send()
            .await?;

        Ok(())
    }

    /// Notify the remote worker that onboarding failed for specific blocks.
    ///
    /// This calls the `kvbm.connector.worker.failed_onboard` handler.
    /// The worker will add the block_ids to its failed_onboarding set.
    ///
    /// # Arguments
    /// * `request_id` - The request that failed onboarding
    /// * `block_ids` - The block IDs that failed to load
    pub async fn mark_failed_onboarding(
        &self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> Result<()> {
        let message = FailedOnboardMessage {
            request_id,
            block_ids,
        };
        self.messenger
            .unary(FAILED_ONBOARD_HANDLER)?
            .payload(message)?
            .instance(self.remote)
            .send()
            .await?;

        Ok(())
    }

    /// Notify the remote worker that offloading is complete for a request.
    ///
    /// This calls the `kvbm.connector.worker.offload_complete` handler.
    /// The worker will add the request_id to its finished_offloading set.
    ///
    /// # Arguments
    /// * `request_id` - The request that finished offloading
    pub async fn mark_offloading_complete(&self, request_id: String) -> Result<()> {
        let message = OffloadCompleteMessage { request_id };

        self.messenger
            .unary(OFFLOAD_COMPLETE_HANDLER)?
            .payload(message)?
            .instance(self.remote)
            .send()
            .await?;

        Ok(())
    }

    /// Get the layout configuration from the remote worker.
    ///
    /// This calls the `kvbm.connector.worker.get_layout_config` handler on the remote worker.
    ///
    /// # Returns
    /// A typed unary result that resolves to the layout configuration
    pub fn get_layout_config(&self) -> Result<velo::TypedUnaryResult<LayoutConfig>> {
        let awaiter = self
            .messenger
            .typed_unary::<LayoutConfig>(GET_LAYOUT_CONFIG_HANDLER)?
            .instance(self.remote)
            .send();
        Ok(awaiter)
    }
}

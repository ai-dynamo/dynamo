// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use futures::future::try_join_all;

impl WorkerClients {
    pub async fn mark_onboarding_complete(&self, request_id: String) -> Result<()> {
        let futures = self
            .worker_connector_clients
            .iter()
            .map(|client| client.mark_onboarding_complete(request_id.clone()));
        try_join_all(futures).await?;
        Ok(())
    }

    #[expect(dead_code)] // Will be used when offload integration is complete
    pub async fn mark_offloading_complete(&self, request_id: String) -> Result<()> {
        let futures = self
            .worker_connector_clients
            .iter()
            .map(|client| client.mark_offloading_complete(request_id.clone()));
        try_join_all(futures).await?;
        Ok(())
    }

    pub async fn mark_failed_onboarding(
        &self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> Result<()> {
        let futures = self
            .worker_connector_clients
            .iter()
            .map(|client| client.mark_failed_onboarding(request_id.clone(), block_ids.clone()));
        try_join_all(futures).await?;
        Ok(())
    }
}

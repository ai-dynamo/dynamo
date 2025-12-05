// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Context, Result};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::traits::events::EventPublisher;
use tokio::sync::watch;

use crate::kv_router::WORKER_KV_INDEXER_QUERY_SUBJECT;
use crate::kv_router::indexer::{WorkerKvQueryRequest, WorkerKvQueryResponse};
use crate::kv_router::protocols::WorkerId;

/// Router-side client for querying worker local KV indexers
///
/// Performs request/reply communication with workers via NATS.
/// (Only queries workers that have `enable_local_indexer=true` in their MDC user_data)
/// The client is spawned by KvRouter; it watches same discovery stream as the router.
pub struct WorkerQueryClient {
    component: Component,
    /// Watch receiver for enable_local_indexer state per worker
    local_indexer_rx: watch::Receiver<HashMap<WorkerId, bool>>,
}

impl WorkerQueryClient {
    /// Create a new WorkerQueryClient with a watch receiver for local indexer states
    pub fn new(
        component: Component,
        local_indexer_rx: watch::Receiver<HashMap<WorkerId, bool>>,
    ) -> Self {
        Self {
            component,
            local_indexer_rx,
        }
    }

    /// Check if a worker has local indexer enabled
    pub fn has_local_indexer(&self, worker_id: WorkerId) -> bool {
        self.local_indexer_rx
            .borrow()
            .get(&worker_id)
            .copied()
            .unwrap_or(false)
    }

    /// Query a specific worker's local KV indexer and return its buffered events.
    /// Returns an error if the worker does not have enable_local_indexer=true.
    pub async fn query_worker(&self, worker_id: WorkerId) -> Result<WorkerKvQueryResponse> {
        // Check if worker has local indexer enabled
        if !self.has_local_indexer(worker_id) {
            anyhow::bail!(
                "Worker {} does not have local indexer enabled (enable_local_indexer=false or not set in MDC user_data)",
                worker_id
            );
        }

        // Match worker's subscribe format
        let subject_str = format!("{}.{}", WORKER_KV_INDEXER_QUERY_SUBJECT, worker_id); // see publisher.rs/start_worker_kv_query_service()
        let subject = format!("{}.{}", self.component.subject(), subject_str);

        tracing::debug!(
            "Router sending query request to worker {} on NATS subject: {}",
            worker_id,
            subject
        );

        // Create and serialize request
        let request = WorkerKvQueryRequest { worker_id };
        let request_bytes =
            serde_json::to_vec(&request).context("Failed to serialize WorkerKvQueryRequest")?;

        // Send NATS request with timeout using DRT helper
        let timeout = tokio::time::Duration::from_secs(1);
        let response_msg = self
            .component
            .drt()
            .kv_router_nats_request(subject.clone(), request_bytes.into(), timeout)
            .await
            .with_context(|| {
                format!(
                    "Failed to send request to worker {} on subject {}",
                    worker_id, subject
                )
            })?;

        // Deserialize response
        let response: WorkerKvQueryResponse = serde_json::from_slice(&response_msg.payload)
            .context("Failed to deserialize WorkerKvQueryResponse")?;

        Ok(response)
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use dynamo_runtime::component::Namespace;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::traits::events::EventPublisher;

use crate::kv_router::WORKER_KV_INDEXER_QUERY_SUBJECT;
use crate::kv_router::protocols::{WorkerId, WorkerKvQueryRequest, WorkerKvQueryResponse};

/// Router-side client for querying worker local KV indexers
///
/// Uses the namespace abstraction for clean request/reply communication
/// with workers via NATS.
pub struct WorkerQueryClient {
    namespace: Namespace,
}

impl WorkerQueryClient {
    pub fn new(namespace: Namespace) -> Self {
        Self { namespace }
    }

    /// Query a specific worker's local KV indexer and return its buffered events
    pub async fn query_worker(&self, worker_id: WorkerId) -> Result<WorkerKvQueryResponse> {
        // Match worker's subscribe format: namespace.{namespace_name}.{SUBJECT}.{worker_id}
        let subject = format!(
            "{}.{}.{}",
            self.namespace.subject(),
            WORKER_KV_INDEXER_QUERY_SUBJECT,
            worker_id
        );

        tracing::info!(
            "Router sending request to worker {} on NATS subject: {}",
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
            .namespace
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

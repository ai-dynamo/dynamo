// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dynamo_kv_router::{
    indexer::{
        KvStateAgentIdentity, KvStateAgentStatus, WorkerKvQueryKind, WorkerKvQueryRequest,
        WorkerKvQueryResponse,
    },
    protocols::{DpRank, WorkerId},
};
use dynamo_runtime::{
    component::{Component, Instance},
    pipeline::{AddressedPushRouter, AddressedRequest, AsyncEngine, ManyOut, SingleIn},
    protocols::maybe_error::MaybeError,
};
use futures::StreamExt;

#[async_trait]
pub(crate) trait WorkerQueryTransport: Send + Sync {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse>;
}

pub(crate) struct RuntimeWorkerQueryTransport {
    addressed: Arc<AddressedPushRouter>,
}

impl RuntimeWorkerQueryTransport {
    pub(crate) async fn new(component: &Component) -> Result<Self> {
        Ok(Self {
            addressed: AddressedPushRouter::from_runtime_provider(component).await?,
        })
    }

    pub(crate) async fn query_status(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        target: Instance,
        expected: KvStateAgentIdentity,
        expected_attachment_generation: Option<u64>,
        timeout: Duration,
    ) -> Result<KvStateAgentStatus> {
        let request = WorkerKvQueryRequest {
            worker_id,
            dp_rank,
            start_event_id: None,
            end_event_id: None,
            supports_tree_dump_failed: true,
            kind: WorkerKvQueryKind::Status {
                expected,
                expected_attachment_generation,
            },
        };
        let response = tokio::time::timeout(timeout, self.query(target, request))
            .await
            .context("KV state-agent status handshake timed out")??;
        match response {
            WorkerKvQueryResponse::Status(status) => Ok(status),
            WorkerKvQueryResponse::Error(message) => {
                anyhow::bail!("KV state-agent status query rejected: {message}")
            }
            _ => anyhow::bail!("unexpected non-status response to a KV state-agent status query"),
        }
    }

    async fn query(
        &self,
        instance: Instance,
        request: WorkerKvQueryRequest,
    ) -> Result<WorkerKvQueryResponse> {
        let instance_id = instance.instance_id;
        let endpoint_name = instance.endpoint.clone();
        let addressed_request =
            SingleIn::new(request).map(|req| AddressedRequest::for_instance(req, instance));
        let mut stream: ManyOut<WorkerKvQueryResponse> = self
            .addressed
            .generate(addressed_request)
            .await
            .with_context(|| {
                format!(
                    "Failed to send worker KV query via endpoint {endpoint_name} instance {instance_id}"
                )
            })?;
        let response = stream
            .next()
            .await
            .context("Worker KV query returned an empty response stream")?;
        if let Some(err) = response.err() {
            return Err(err).context("Worker KV query response error");
        }
        Ok(response)
    }
}

#[async_trait]
impl WorkerQueryTransport for RuntimeWorkerQueryTransport {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        let request = WorkerKvQueryRequest {
            worker_id,
            dp_rank,
            start_event_id,
            end_event_id,
            supports_tree_dump_failed: true,
            kind: WorkerKvQueryKind::Recovery,
        };
        self.query(target, request).await.with_context(|| {
            format!("worker KV recovery query failed for worker {worker_id} rank {dp_rank}")
        })
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dynamo_kv_router::{
    indexer::{WorkerKvQueryRequest, WorkerKvQueryResponse},
    protocols::{DpRank, WorkerId},
};
use dynamo_runtime::{
    component::{Component, Instance},
    discovery::EndpointInstanceId,
    pipeline::{AddressedPushRouter, AddressedRequest, AsyncEngine, ManyOut, SingleIn},
    protocols::maybe_error::MaybeError,
};
use futures::StreamExt;

#[async_trait]
pub(super) trait WorkerQueryTransport: Send + Sync {
    async fn query_worker(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        target: Instance,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse>;

    async fn cancel_instance_streams(&self, _endpoint_id: &EndpointInstanceId) -> usize {
        0
    }

    async fn clear_instance_tombstone(&self, _endpoint_id: &EndpointInstanceId) {}
}

pub(super) struct RuntimeWorkerQueryTransport {
    addressed: Arc<AddressedPushRouter>,
}

impl RuntimeWorkerQueryTransport {
    pub(super) async fn new(component: &Component) -> Result<Self> {
        Ok(Self {
            addressed: AddressedPushRouter::from_runtime_provider(component).await?,
        })
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
        };
        let instance = target;
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
                    "Failed to send worker KV query to worker {worker_id} dp_rank {dp_rank} \
                     via endpoint {endpoint_name} instance {instance_id}"
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

    async fn cancel_instance_streams(&self, endpoint_id: &EndpointInstanceId) -> usize {
        self.addressed.cancel_instance_streams(endpoint_id).await
    }

    async fn clear_instance_tombstone(&self, endpoint_id: &EndpointInstanceId) {
        self.addressed.clear_instance_tombstone(endpoint_id).await;
    }
}

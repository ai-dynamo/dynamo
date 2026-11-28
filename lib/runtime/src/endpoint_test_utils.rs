// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::component::Endpoint;
use crate::discovery::{
    Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, DiscoveryStream,
    EndpointInstanceId, MockDiscovery,
};
use crate::pipeline::network::PushWorkHandler;
use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

pub(crate) struct NoopHandler;

#[async_trait]
impl PushWorkHandler for NoopHandler {
    async fn handle_payload(
        &self,
        _payload: bytes::Bytes,
        _request_id: Option<String>,
    ) -> std::result::Result<(), crate::pipeline::PipelineError> {
        Ok(())
    }

    fn add_metrics(&self, _endpoint: &Endpoint, _labels: Option<&[(&str, &str)]>) -> Result<()> {
        Ok(())
    }
}

pub(crate) struct UnavailableUnregisterDiscovery {
    pub(crate) inner: MockDiscovery,
    pub(crate) hanging: bool,
    pub(crate) attempted: tokio::sync::Notify,
}

#[async_trait]
impl Discovery for UnavailableUnregisterDiscovery {
    fn instance_id(&self) -> u64 {
        self.inner.instance_id()
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        self.inner.register_internal(spec).await
    }

    async fn unregister(&self, _instance: DiscoveryInstance) -> Result<()> {
        self.attempted.notify_one();
        if self.hanging {
            std::future::pending().await
        } else {
            anyhow::bail!("discovery unavailable")
        }
    }

    async fn get_endpoint(&self, id: &EndpointInstanceId) -> Result<Option<DiscoveryInstance>> {
        self.inner.get_endpoint(id).await
    }

    async fn list(&self, _query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        anyhow::bail!("endpoint lease acquisition must use a point lookup")
    }

    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream> {
        self.inner.list_and_watch(query, cancel_token).await
    }
}

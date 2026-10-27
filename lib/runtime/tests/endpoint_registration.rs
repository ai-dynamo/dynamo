// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Endpoint,
    discovery::DiscoveryQuery,
    distributed::DistributedConfig,
    pipeline::{PipelineError, network::PushWorkHandler},
};

struct IdleHandler;

#[async_trait]
impl PushWorkHandler for IdleHandler {
    async fn handle_payload(
        &self,
        _payload: Bytes,
        _request_id: Option<String>,
    ) -> Result<(), PipelineError> {
        Ok(())
    }

    fn add_metrics(&self, _endpoint: &Endpoint, _labels: Option<&[(&str, &str)]>) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn listener_startup_can_withhold_discovery_until_explicit_registration() -> Result<()> {
    let runtime =
        DistributedRuntime::new(Runtime::from_current()?, DistributedConfig::process_local())
            .await?;
    let namespace = format!("admission_{}", uuid::Uuid::new_v4().simple());
    let endpoint = runtime
        .namespace(&namespace)?
        .component("worker")?
        .endpoint("generate");
    let query = DiscoveryQuery::Endpoint {
        namespace,
        component: "worker".into(),
        endpoint: "generate".into(),
    };

    let started = endpoint
        .endpoint_builder()
        .handler(Arc::new(IdleHandler))
        .start_without_registration()
        .await?;
    assert!(runtime.discovery().list(query.clone()).await?.is_empty());

    endpoint.register_endpoint_instance().await?;
    assert_eq!(runtime.discovery().list(query.clone()).await?.len(), 1);
    endpoint.unregister_endpoint_instance().await?;
    assert!(runtime.discovery().list(query.clone()).await?.is_empty());

    started.shutdown().await?;
    assert!(runtime.discovery().list(query).await?.is_empty());
    Ok(())
}

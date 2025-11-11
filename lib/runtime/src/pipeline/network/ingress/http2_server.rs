// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 Request Plane Server Adapter
//!
//! Wraps the existing HttpEndpoint to implement the RequestPlaneServer trait.

use super::http_endpoint::SharedHttpServer;
use super::unified_server::{InstanceInfo, RequestPlaneServer, ServerStats};
use super::*;
use anyhow::Result;
use async_trait::async_trait;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// HTTP/2 server adapter for the unified interface
pub struct Http2RequestServer {
    cancellation_token: CancellationToken,
    server: Option<Arc<SharedHttpServer>>,
}

impl Http2RequestServer {
    pub fn new(cancellation_token: CancellationToken) -> Self {
        Self {
            cancellation_token,
            server: None,
        }
    }
}

#[async_trait]
impl RequestPlaneServer for Http2RequestServer {
    async fn start(
        &self,
        bind_addr: SocketAddr,
        handler: Arc<dyn PushWorkHandler>,
        instance_info: InstanceInfo,
    ) -> Result<()> {
        let server = SharedHttpServer::new(bind_addr, self.cancellation_token.clone());

        // For single endpoint, register it with a default subject
        let subject = format!(
            "namespace.{}.component.{}.endpoint.{}.instance.{}",
            instance_info.namespace,
            instance_info.component_name,
            instance_info.endpoint_name,
            instance_info.instance_id
        );

        server
            .register_endpoint(
                subject,
                handler,
                instance_info.instance_id,
                instance_info.namespace.clone(),
                instance_info.component_name.clone(),
                instance_info.endpoint_name.clone(),
                instance_info.system_health.clone(),
            )
            .await?;

        server.start().await
    }

    async fn register_endpoint(
        &self,
        subject: String,
        handler: Arc<dyn PushWorkHandler>,
        instance_info: InstanceInfo,
    ) -> Result<()> {
        if let Some(server) = &self.server {
            server
                .register_endpoint(
                    subject,
                    handler,
                    instance_info.instance_id,
                    instance_info.namespace,
                    instance_info.component_name,
                    instance_info.endpoint_name,
                    instance_info.system_health,
                )
                .await
        } else {
            anyhow::bail!("HTTP server not started yet. Call start() first.")
        }
    }

    async fn unregister_endpoint(&self, subject: &str, endpoint_name: &str) -> Result<()> {
        if let Some(server) = &self.server {
            server.unregister_endpoint(subject, endpoint_name).await;
            Ok(())
        } else {
            Ok(())
        }
    }

    async fn stop(&self) -> Result<()> {
        self.cancellation_token.cancel();
        Ok(())
    }

    fn transport_name(&self) -> &'static str {
        "http2"
    }

    fn supports_multiplexing(&self) -> bool {
        true
    }
}

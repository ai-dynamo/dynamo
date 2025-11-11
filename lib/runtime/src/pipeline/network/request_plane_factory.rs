// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request Plane Factory
//!
//! Factory for creating request plane clients and servers based on configuration.
//! This provides a unified way to instantiate the appropriate transport implementation.

use crate::config::RequestPlaneMode;
use crate::traits::DistributedRuntimeProvider;
use anyhow::Result;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use super::egress::unified_client::RequestPlaneClient;
use super::ingress::unified_server::RequestPlaneServer;

/// Factory for creating request plane clients and servers
pub struct RequestPlane;

impl RequestPlane {
    // Create a request plane client based on the mode
    pub async fn create_client(
        mode: RequestPlaneMode,
        drt: &impl DistributedRuntimeProvider,
    ) -> Result<Arc<dyn RequestPlaneClient>> {
        match mode {
            RequestPlaneMode::Nats => {
                // NATS client support - requires NATS infrastructure
                let _nats = drt.drt().nats_client().ok_or_else(|| {
                    anyhow::anyhow!(
                        "NATS client not available. Please ensure NATS is running and accessible."
                    )
                })?;

                // Note: NatsRequestClient needs to be implemented or imported from the appropriate module
                anyhow::bail!("NATS request client not yet implemented in this configuration")
            }
            RequestPlaneMode::Http => {
                use super::egress::http_router::HttpRequestClient;

                let http_client = HttpRequestClient::from_env()?;
                Ok(Arc::new(http_client))
            }
            RequestPlaneMode::Tcp => {
                use super::egress::tcp_client::TcpRequestClient;

                let client = TcpRequestClient::from_env()?;
                Ok(Arc::new(client))
            }
        }
    }


    // Create a request plane server based on the mode
    pub async fn create_server(
        mode: RequestPlaneMode,
        cancellation_token: CancellationToken,
    ) -> Result<Arc<dyn RequestPlaneServer>> {
        match mode {
            RequestPlaneMode::Nats => {
                // NATS server support - requires NATS infrastructure
                // Note: NatsRequestServer needs to be implemented or imported from the appropriate module
                anyhow::bail!("NATS request server not yet implemented in this configuration")
            }
            RequestPlaneMode::Http => {
                use super::ingress::http2_server::Http2RequestServer;

                Ok(Arc::new(Http2RequestServer::new(cancellation_token)))
            }
            RequestPlaneMode::Tcp => {
                use super::ingress::tcp_server::TcpRequestServer;

                Ok(Arc::new(TcpRequestServer::new(cancellation_token)))
            }
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example demonstrating HTTP request plane usage
//!
//! This example shows how to use the HTTP/2 request plane instead of NATS
//! for distributing requests between routers and workers.
//!
//! # Running the example
//!
//! 1. Start etcd:
//!    ```bash
//!    etcd
//!    ```
//!
//! 2. Run the example with HTTP mode:
//!    ```bash
//!    DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo
//!    ```
//!
//! # Configuration
//!
//! The following environment variables control the HTTP request plane:
//!
//! - `DYN_REQUEST_PLANE`: Set to "http" to enable HTTP mode (default: "nats")
//! - `DYN_HTTP_RPC_HOST`: HTTP server bind address (default: "0.0.0.0")
//! - `DYN_HTTP_RPC_PORT`: HTTP server port (default: 8081)
//! - `DYN_HTTP_RPC_ROOT_PATH`: HTTP RPC root path (default: "/v1/dynamo")
//! - `DYN_HTTP_REQUEST_TIMEOUT`: HTTP request timeout in seconds (default: 5)

use anyhow::Result;
use dynamo_runtime::{
    DistributedRuntime,
    config::RequestPlaneMode,
    pipeline::{
        AsyncEngine, Data, Error, ManyOut, ServiceEngine, SingleIn,
        network::Ingress,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EchoRequest {
    message: String,
}

impl Data for EchoRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EchoResponse {
    message: String,
    worker_id: String,
}

impl Data for EchoResponse {}

/// Simple echo engine that returns the input message
struct EchoEngine {
    worker_id: String,
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<EchoRequest>, ManyOut<EchoResponse>, Error> for EchoEngine {
    async fn generate(&self, request: SingleIn<EchoRequest>) -> Result<ManyOut<EchoResponse>, Error> {
        let (req, context) = request.transfer(());

        // Simulate some processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let response = EchoResponse {
            message: req.message.clone(),
            worker_id: self.worker_id.clone(),
        };

        let stream = tokio_stream::once(response);
        Ok(ManyOut::new(Box::pin(stream), context.context()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Check request plane mode
    let mode = RequestPlaneMode::from_env();
    tracing::info!("Request plane mode: {}", mode);

    if mode.is_nats() {
        tracing::warn!(
            "Running in NATS mode. Set DYN_REQUEST_PLANE=http to use HTTP mode."
        );
    } else {
        tracing::info!("✓ Running in HTTP/2 mode");
        tracing::info!(
            "HTTP RPC endpoint: http://{}:{}{}",
            std::env::var("DYN_HTTP_RPC_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            std::env::var("DYN_HTTP_RPC_PORT").unwrap_or_else(|_| "8081".to_string()),
            std::env::var("DYN_HTTP_RPC_ROOT_PATH").unwrap_or_else(|_| "/v1/dynamo".to_string())
        );
    }

    // Create distributed runtime
    let drt = DistributedRuntime::builder()
        .namespace("example")?
        .build()
        .await?;

    // Create worker component
    let worker_component = drt.component("echo-worker")?;

    // Create service
    let service = worker_component.service_builder().create().await?;

    // Create echo engine
    let echo_engine = Arc::new(EchoEngine {
        worker_id: "worker-1".to_string(),
    });

    // Create service engine and ingress
    let service_engine = ServiceEngine::new(echo_engine);
    let ingress = Ingress::for_engine(service_engine)?;

    // Create endpoint
    let endpoint = service.endpoint("echo").endpoint_builder().handler(ingress);

    // Start endpoint in background
    let endpoint_task = tokio::spawn(async move {
        endpoint.start().await
    });

    // Give the server time to start
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Create client component
    let client_component = drt.component("echo-client")?;
    let client = client_component.endpoint("echo").client().await?;

    // Create router
    use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};
    let router = PushRouter::<EchoRequest, EchoResponse>::from_client(
        client,
        RouterMode::RoundRobin,
    )
    .await?;

    // Send requests
    tracing::info!("Sending requests...");
    for i in 0..5 {
        let request = EchoRequest {
            message: format!("Hello from request {}", i),
        };

        let request = SingleIn::new(request);
        let mut stream = router.generate(request).await?;

        while let Some(response) = stream.next().await {
            tracing::info!(
                "Received response: {:?} from worker: {}",
                response.message,
                response.worker_id
            );
        }
    }

    tracing::info!("All requests completed successfully!");

    // Shutdown
    drt.shutdown().await?;
    endpoint_task.abort();

    Ok(())
}


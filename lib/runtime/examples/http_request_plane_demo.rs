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
//! 2. Run the server with HTTP mode:
//!    ```bash
//!    DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- server
//!    ```
//!
//! 3. In another terminal, run the client:
//!    ```bash
//!    DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- client
//!    ```
//!
//! 4. Or test with curl (shown in server output)
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

use dynamo_runtime::{
    DistributedRuntime, Result, Runtime, Worker,
    config::RequestPlaneMode,
    engine::{AsyncEngine, AsyncEngineContextProvider},
    logging,
    pipeline::{
        Error, ManyOut, ResponseStream, SingleIn,
        network::Ingress,
        network::egress::push_router::{PushRouter, RouterMode},
    },
    protocols::maybe_error::MaybeError,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EchoRequest {
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EchoResponse {
    message: String,
    worker_id: String,
}

impl MaybeError for EchoResponse {
    fn err(&self) -> Option<anyhow::Error> {
        None
    }

    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self {
            message: format!("Error: {}", err),
            worker_id: "error".to_string(),
        }
    }
}

/// Simple echo engine that returns the input message
struct EchoEngine {
    worker_id: String,
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<EchoRequest>, ManyOut<EchoResponse>, Error> for EchoEngine {
    async fn generate(
        &self,
        request: SingleIn<EchoRequest>,
    ) -> Result<ManyOut<EchoResponse>, Error> {
        let (req, context) = request.transfer(());
        let engine_ctx = context.context();

        // Simulate some processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let response = EchoResponse {
            message: req.message.clone(),
            worker_id: self.worker_id.clone(),
        };

        let stream = tokio_stream::once(response);
        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
    }
}

fn main() -> Result<()> {
    logging::init();

    // Check if running as server or client
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("server");

    let worker = Worker::from_settings()?;

    match mode {
        "client" => worker.execute(client_app),
        "server" | _ => worker.execute(server_app),
    }
}

async fn server_app(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    // Check request plane mode
    let mode = RequestPlaneMode::from_env();
    tracing::info!("Request plane mode: {}", mode);

    if mode.is_nats() {
        tracing::warn!("Running in NATS mode. Set DYN_REQUEST_PLANE=http to use HTTP mode.");
    } else {
        let host = std::env::var("DYN_HTTP_RPC_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = std::env::var("DYN_HTTP_RPC_PORT").unwrap_or_else(|_| "8081".to_string());
        let root_path =
            std::env::var("DYN_HTTP_RPC_ROOT_PATH").unwrap_or_else(|_| "/v1/dynamo".to_string());

        tracing::info!("✓ Running in HTTP/2 mode");
        tracing::info!("HTTP RPC endpoint: http://{}:{}{}", host, port, root_path);

        // Note about curl testing
        tracing::info!("\n📋 To test the HTTP endpoint:");
        tracing::info!("   Run the client: cargo run --example http_request_plane_demo -- client");
        tracing::info!(
            "\n   Note: Direct curl testing requires encoding the request with TwoPartCodec,"
        );
        tracing::info!(
            "   which includes control headers + request payload. Use the client instead.\n"
        );
    }

    // Create echo engine
    let echo_engine = Arc::new(EchoEngine {
        worker_id: "worker-1".to_string(),
    });

    // Create ingress
    let ingress = Ingress::for_engine(echo_engine)?;

    // Start service and endpoint
    drt.namespace("example")?
        .component("echo-worker")?
        .service_builder()
        .create()
        .await?
        .endpoint("echo")
        .endpoint_builder()
        .handler(ingress)
        .start()
        .await
}

async fn client_app(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    // Check request plane mode
    let mode = RequestPlaneMode::from_env();
    tracing::info!("Client starting with request plane mode: {}", mode);

    // Create client
    let client = drt
        .namespace("example")?
        .component("echo-worker")?
        .endpoint("echo")
        .client()
        .await?;

    // Wait for worker instances to be available
    tracing::info!("Waiting for worker instances...");
    client.wait_for_instances().await?;
    tracing::info!("✓ Worker instances available");

    // Create router
    let router =
        PushRouter::<EchoRequest, EchoResponse>::from_client(client, RouterMode::RoundRobin)
            .await?;

    // Send multiple requests
    tracing::info!("\n🚀 Sending requests...\n");

    for i in 0..5 {
        let request = EchoRequest {
            message: format!("Hello from request #{}", i + 1),
        };

        tracing::info!("→ Sending: {:?}", request.message);

        let request = SingleIn::new(request);
        let mut stream = router.generate(request).await?;

        while let Some(response) = stream.next().await {
            tracing::info!(
                "← Received: message='{}' from worker='{}'",
                response.message,
                response.worker_id
            );
        }
    }

    tracing::info!("\n✅ All requests completed successfully!\n");

    // Shutdown
    runtime.shutdown();

    Ok(())
}

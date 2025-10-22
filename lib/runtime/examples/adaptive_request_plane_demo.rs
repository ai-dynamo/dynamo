// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example demonstrating adaptive request plane usage
//!
//! This example shows how to use the adaptive request plane that automatically
//! discovers services from etcd and adapts to their transport types (HTTP or NATS)
//! without relying on the DYN_REQUEST_PLANE environment variable.
//!
//! # Running the example
//!
//! 1. Start etcd:
//!    ```bash
//!    etcd
//!    ```
//!
//! 2. Run a worker with HTTP mode:
//!    ```bash
//!    DYN_REQUEST_PLANE=http cargo run --example adaptive_request_plane_demo -- server
//!    ```
//!
//! 3. In another terminal, run the adaptive client (no env var needed):
//!    ```bash
//!    cargo run --example adaptive_request_plane_demo -- client
//!    ```
//!
//! 4. The client will automatically discover the HTTP transport and use it
//!
//! # Key Features
//!
//! - **Workers are explicit**: They know what transport they want to provide and register accordingly
//! - **Clients are adaptive**: They automatically adapt to whatever transports are available
//! - **No environment variables needed for clients**: Discovery is automatic via etcd
//!
//! # Example Flow
//!
//! 1. Worker starts with `DYN_REQUEST_PLANE=http` → registers HTTP endpoint in etcd
//! 2. Frontend starts without `DYN_REQUEST_PLANE` → discovers HTTP endpoint from etcd → creates HTTP client
//! 3. Frontend routes requests using HTTP transport to the worker's HTTP endpoint

use dynamo_runtime::{
    DistributedRuntime, Result, Runtime, Worker,
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
    transport_used: String,
}

impl MaybeError for EchoResponse {
    fn err(&self) -> Option<anyhow::Error> {
        None
    }
    
    fn from_err(_: Box<dyn std::error::Error + Send + Sync + 'static>) -> Self {
        EchoResponse {
            message: "Error occurred".to_string(),
            worker_id: "error".to_string(),
            transport_used: "error".to_string(),
        }
    }
}

struct EchoEngine {
    worker_id: String,
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<EchoRequest>, ManyOut<EchoResponse>, Error> for EchoEngine {
    async fn generate(
        &self,
        request: SingleIn<EchoRequest>,
    ) -> Result<ManyOut<EchoResponse>, Error> {
        let (req, ctx) = request.into_parts();
        
        // Determine transport type from context or default
        let transport_used = "adaptive".to_string();
        
        let response = EchoResponse {
            message: format!("Echo: {}", req.message),
            worker_id: self.worker_id.clone(),
            transport_used,
        };

        let stream = tokio_stream::once(Ok(response));
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    logging::init();

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

    // Check request plane mode - workers are still explicit about their transport
    use dynamo_runtime::config::RequestPlaneMode;
    let mode = RequestPlaneMode::from_env();
    tracing::info!("Worker request plane mode: {}", mode);

    if mode.is_nats() {
        tracing::info!("✓ Worker running in NATS mode");
        tracing::info!("Worker will register NATS transport in etcd");
    } else {
        let host = dynamo_runtime::utils::get_http_rpc_host_from_env();
        let port = std::env::var("DYN_HTTP_RPC_PORT").unwrap_or_else(|_| "8081".to_string());
        let root_path =
            std::env::var("DYN_HTTP_RPC_ROOT_PATH").unwrap_or_else(|_| "/v1/rpc".to_string());

        tracing::info!("✓ Worker running in HTTP/2 mode");
        tracing::info!("HTTP RPC endpoint: http://{}:{}{}", host, port, root_path);
        tracing::info!("Worker will register HTTP transport in etcd");
    }

    // Create echo engine
    let echo_engine = Arc::new(EchoEngine {
        worker_id: "adaptive-worker-1".to_string(),
    });

    // Create ingress
    let ingress = Ingress::for_engine(echo_engine)?;

    // Start service and endpoint
    drt.namespace("adaptive-example")?
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

    tracing::info!("🔍 Starting adaptive client - no DYN_REQUEST_PLANE needed!");
    tracing::info!("Client will discover services from etcd and adapt to their transports");

    // Create client for the echo service - this will discover services dynamically
    let client = drt
        .namespace("adaptive-example")?
        .component("echo-worker")?
        .client("echo")
        .await?;

    tracing::info!("✓ Client created, waiting for service discovery...");

    // Wait for instances to be discovered
    let instances = client.wait_for_instances().await?;
    tracing::info!("🎯 Discovered {} instances:", instances.len());

    // Since all instances of a component use the same transport, just show the first one
    if let Some(first_instance) = instances.first() {
        match &first_instance.transport {
            dynamo_runtime::component::TransportType::HttpTcp { http_endpoint } => {
                tracing::info!("  - Component transport: HTTP (example endpoint: {})", http_endpoint);
            }
            dynamo_runtime::component::TransportType::NatsTcp(subject) => {
                tracing::info!("  - Component transport: NATS (example subject: {})", subject);
            }
        }
        tracing::info!("  - All {} instances use the same transport type", instances.len());
    }

    // Create adaptive push router - this will automatically use the discovered transports
    let router = PushRouter::<EchoRequest, EchoResponse>::adaptive_from_client(
        client,
        RouterMode::RoundRobin,
    ).await?;

    tracing::info!("🚀 Adaptive router created, sending test requests...");

    // Send some test requests
    for i in 1..=5 {
        let request = SingleIn::new(EchoRequest {
            message: format!("Adaptive test message {}", i),
        });

        match router.round_robin(request).await {
            Ok(mut response_stream) => {
                while let Some(response) = response_stream.next().await {
                    match response {
                        Ok(echo_response) => {
                            tracing::info!(
                                "✅ Response {}: {} (worker: {}, transport: {})",
                                i,
                                echo_response.message,
                                echo_response.worker_id,
                                echo_response.transport_used
                            );
                        }
                        Err(e) => {
                            tracing::error!("❌ Error in response {}: {}", i, e);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("❌ Failed to send request {}: {}", i, e);
            }
        }

        // Small delay between requests
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    tracing::info!("🎉 Adaptive client demo completed!");
    tracing::info!("Key points:");
    tracing::info!("  - Worker was explicit about its transport (via DYN_REQUEST_PLANE)");
    tracing::info!("  - Client was adaptive (no environment variable needed)");
    tracing::info!("  - Client automatically discovered and used the worker's transport");

    Ok(())
}

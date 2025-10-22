// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example demonstrating deadlock prevention in bidirectional communication
//!
//! This example shows how the adaptive request plane prevents deadlocks when
//! components A and B need to communicate with each other by ensuring all
//! instances of a component use the same transport type.
//!
//! # Scenario
//! - Component A needs to call Component B
//! - Component B needs to call Component A (as part of processing A's request)
//! - Both use the same transport type (HTTP) → No deadlock risk
//!
//! # Running the example
//!
//! 1. Start etcd:
//!    ```bash
//!    etcd
//!    ```
//!
//! 2. Start Component A (HTTP):
//!    ```bash
//!    DYN_REQUEST_PLANE=http cargo run --example bidirectional_communication_demo -- component-a
//!    ```
//!
//! 3. Start Component B (HTTP):
//!    ```bash
//!    DYN_REQUEST_PLANE=http cargo run --example bidirectional_communication_demo -- component-b
//!    ```
//!
//! 4. Run the client to trigger bidirectional communication:
//!    ```bash
//!    cargo run --example bidirectional_communication_demo -- client
//!    ```

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
struct ComponentARequest {
    message: String,
    depth: u32, // To prevent infinite recursion
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComponentBRequest {
    message: String,
    depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComponentResponse {
    message: String,
    component: String,
    transport_used: String,
    call_chain: Vec<String>,
}

impl MaybeError for ComponentResponse {
    fn err(&self) -> Option<anyhow::Error> {
        None
    }
}

/// Component A Engine - calls Component B during processing
struct ComponentAEngine {
    component_b_router: Option<Arc<PushRouter<ComponentBRequest, ComponentResponse>>>,
}

impl ComponentAEngine {
    fn new() -> Self {
        Self {
            component_b_router: None,
        }
    }

    fn set_component_b_router(&mut self, router: Arc<PushRouter<ComponentBRequest, ComponentResponse>>) {
        self.component_b_router = Some(router);
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<ComponentARequest>, ManyOut<ComponentResponse>, Error> for ComponentAEngine {
    async fn generate(
        &self,
        request: SingleIn<ComponentARequest>,
    ) -> Result<ManyOut<ComponentResponse>, Error> {
        let (req, ctx) = request.into_parts();
        
        tracing::info!("Component A processing: {} (depth: {})", req.message, req.depth);
        
        let mut call_chain = vec!["Component-A".to_string()];
        
        // If depth allows, call Component B
        if req.depth > 0 && self.component_b_router.is_some() {
            tracing::info!("Component A calling Component B...");
            
            let b_request = SingleIn::new(ComponentBRequest {
                message: format!("From A: {}", req.message),
                depth: req.depth - 1,
            });
            
            if let Some(router) = &self.component_b_router {
                match router.round_robin(b_request).await {
                    Ok(mut response_stream) => {
                        if let Some(Ok(b_response)) = response_stream.next().await {
                            tracing::info!("Component A received response from B: {}", b_response.message);
                            call_chain.extend(b_response.call_chain);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Component A failed to call Component B: {}", e);
                    }
                }
            }
        }

        let response = ComponentResponse {
            message: format!("A processed: {}", req.message),
            component: "Component-A".to_string(),
            transport_used: "adaptive".to_string(),
            call_chain,
        };

        let stream = tokio_stream::once(Ok(response));
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

/// Component B Engine - calls Component A during processing
struct ComponentBEngine {
    component_a_router: Option<Arc<PushRouter<ComponentARequest, ComponentResponse>>>,
}

impl ComponentBEngine {
    fn new() -> Self {
        Self {
            component_a_router: None,
        }
    }

    fn set_component_a_router(&mut self, router: Arc<PushRouter<ComponentARequest, ComponentResponse>>) {
        self.component_a_router = Some(router);
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<ComponentBRequest>, ManyOut<ComponentResponse>, Error> for ComponentBEngine {
    async fn generate(
        &self,
        request: SingleIn<ComponentBRequest>,
    ) -> Result<ManyOut<ComponentResponse>, Error> {
        let (req, ctx) = request.into_parts();
        
        tracing::info!("Component B processing: {} (depth: {})", req.message, req.depth);
        
        let mut call_chain = vec!["Component-B".to_string()];
        
        // If depth allows, call Component A
        if req.depth > 0 && self.component_a_router.is_some() {
            tracing::info!("Component B calling Component A...");
            
            let a_request = SingleIn::new(ComponentARequest {
                message: format!("From B: {}", req.message),
                depth: req.depth - 1,
            });
            
            if let Some(router) = &self.component_a_router {
                match router.round_robin(a_request).await {
                    Ok(mut response_stream) => {
                        if let Some(Ok(a_response)) = response_stream.next().await {
                            tracing::info!("Component B received response from A: {}", a_response.message);
                            call_chain.extend(a_response.call_chain);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Component B failed to call Component A: {}", e);
                    }
                }
            }
        }

        let response = ComponentResponse {
            message: format!("B processed: {}", req.message),
            component: "Component-B".to_string(),
            transport_used: "adaptive".to_string(),
            call_chain,
        };

        let stream = tokio_stream::once(Ok(response));
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    logging::init();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("component-a");

    let worker = Worker::new().await?;

    match mode {
        "component-a" => worker.execute(component_a_app),
        "component-b" => worker.execute(component_b_app),
        "client" => worker.execute(client_app),
        _ => {
            tracing::error!("Usage: {} [component-a|component-b|client]", args[0]);
            std::process::exit(1);
        }
    }
}

async fn component_a_app(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    tracing::info!("🚀 Starting Component A");
    
    // Check transport mode
    use dynamo_runtime::config::RequestPlaneMode;
    let mode = RequestPlaneMode::from_env();
    tracing::info!("Component A using transport: {}", mode);

    // Create Component A engine
    let engine = Arc::new(ComponentAEngine::new());

    // Create ingress for Component A
    let ingress = Ingress::for_engine(engine)?;

    // Start Component A service
    drt.namespace("bidirectional-demo")?
        .component("component-a")?
        .service_builder()
        .create()
        .await?
        .endpoint("process")
        .endpoint_builder()
        .handler(ingress)
        .start()
        .await
}

async fn component_b_app(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    tracing::info!("🚀 Starting Component B");
    
    // Check transport mode
    use dynamo_runtime::config::RequestPlaneMode;
    let mode = RequestPlaneMode::from_env();
    tracing::info!("Component B using transport: {}", mode);

    // Create Component B engine
    let engine = Arc::new(ComponentBEngine::new());

    // Create ingress for Component B
    let ingress = Ingress::for_engine(engine)?;

    // Start Component B service
    drt.namespace("bidirectional-demo")?
        .component("component-b")?
        .service_builder()
        .create()
        .await?
        .endpoint("process")
        .endpoint_builder()
        .handler(ingress)
        .start()
        .await
}

async fn client_app(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    tracing::info!("🔍 Starting bidirectional communication client");
    tracing::info!("This demonstrates deadlock prevention with consistent transports");

    // Create client for Component A
    let client_a = drt
        .namespace("bidirectional-demo")?
        .component("component-a")?
        .client("process")
        .await?;

    // Wait for Component A instances
    let instances_a = client_a.wait_for_instances().await?;
    tracing::info!("✓ Discovered Component A: {} instances", instances_a.len());

    // Create adaptive router for Component A
    let router_a = PushRouter::<ComponentARequest, ComponentResponse>::adaptive_from_client(
        client_a,
        RouterMode::RoundRobin,
    ).await?;

    tracing::info!("🎯 Triggering bidirectional communication...");
    tracing::info!("Component A will call Component B, which will call back to Component A");

    // Send request to Component A with depth=2 to trigger bidirectional calls
    let request = SingleIn::new(ComponentARequest {
        message: "Initial bidirectional test".to_string(),
        depth: 2,
    });

    match router_a.round_robin(request).await {
        Ok(mut response_stream) => {
            while let Some(response) = response_stream.next().await {
                match response {
                    Ok(resp) => {
                        tracing::info!("✅ Final response: {}", resp.message);
                        tracing::info!("📞 Call chain: {:?}", resp.call_chain);
                        tracing::info!("🎉 Bidirectional communication completed successfully!");
                        tracing::info!("💡 No deadlocks occurred because both components use the same transport");
                    }
                    Err(e) => {
                        tracing::error!("❌ Error in response: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            tracing::error!("❌ Failed to send request: {}", e);
        }
    }

    Ok(())
}

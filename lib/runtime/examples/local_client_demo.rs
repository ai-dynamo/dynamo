// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example demonstrating LocalClient functionality

use dynamo_runtime::component::LocalClient;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, async_trait};
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::pipeline::{Context, ManyOut, ResponseStream, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
use futures::StreamExt;
use std::sync::Arc;

/// Simple test engine that echoes strings
struct SimpleEchoEngine;

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error> for SimpleEchoEngine {
    async fn generate(
        &self,
        request: SingleIn<String>,
    ) -> Result<ManyOut<Annotated<String>>, anyhow::Error> {
        println!("Engine received: {}", *request);

        let response = Annotated {
            data: Some(format!("Echo: {}", *request)),
            id: None,
            event: None,
            comment: None,
        };

        let context = request.context();

        // Create a simple stream that yields the response once
        let stream = futures::stream::once(async move { response });
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== LocalClient Demo ===\n");

    // Create runtime and DRT
    println!("1. Creating runtime...");
    let runtime = Runtime::from_current()?;

    let config = DistributedConfig {
        etcd_config: Default::default(),
        nats_config: Default::default(),
        is_static: true,
    };

    let drt = DistributedRuntime::new(runtime, config).await?;
    println!("   ✓ Runtime created\n");

    // Create namespace, component, and endpoint
    println!("2. Creating endpoint structure...");
    let namespace = drt.namespace("demo")?;
    let component = namespace.component("echo-service")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("echo");
    println!("   ✓ Created endpoint: demo/echo-service/echo\n");

    // Create and register an engine
    println!("3. Creating and registering engine...");
    let engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error>> =
        Arc::new(SimpleEchoEngine);

    // Wrap the engine in an Ingress to make it a PushWorkHandler
    let ingress = Ingress::for_engine(engine)?;

    // Create the endpoint instance with the ingress as handler (setup phase)
    let _endpoint_instance = endpoint
        .endpoint_builder()
        .handler(ingress)
        .create()
        .await?;
    println!("   ✓ Engine registered automatically during endpoint creation\n");

    // Create a LocalClient using the endpoint's convenience method
    println!("4. Creating LocalClient...");
    let local_client: LocalClient<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error> =
        endpoint.local_client().await?;
    println!("   ✓ LocalClient created successfully\n");

    // Demonstrate local client usage
    println!("5. Testing LocalClient invocation...");
    println!("   (This bypasses all network layers and invokes the engine directly)");

    // Create a request with context
    let request = Context::new("Hello from LocalClient!".to_string());

    // Generate response using the local client
    let mut response_stream = local_client.generate(request).await?;
    let response = response_stream.next().await.expect("Expected response");

    println!("   Request: Hello from LocalClient!");
    if let Some(data) = &response.data {
        println!("   Response: {}", data);
    }
    println!();

    // Show the benefits
    println!("6. LocalClient Benefits:");
    println!("   ✓ No network overhead");
    println!("   ✓ No etcd watching required");
    println!("   ✓ No instance discovery needed");
    println!("   ✓ Direct in-process engine invocation");
    println!("   ✓ Perfect for testing and local development\n");

    println!("Demo completed successfully!");
    Ok(())
}

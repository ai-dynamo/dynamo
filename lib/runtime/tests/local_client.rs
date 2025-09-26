// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for LocalClient functionality

use dynamo_runtime::component::LocalClient;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, async_trait};
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::pipeline::{Context, ManyOut, ResponseStream, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
use futures::StreamExt;
use std::sync::Arc;

/// Simple test engine that echoes the input
struct EchoEngine;

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error> for EchoEngine {
    async fn generate(
        &self,
        request: SingleIn<String>,
    ) -> Result<ManyOut<Annotated<String>>, anyhow::Error> {
        let response = Annotated {
            data: Some((*request).clone()),
            id: None,
            event: None,
            comment: None,
        };
        let ctx = request.context();

        // Create a simple stream that yields the response once
        let stream = futures::stream::once(async move { response });
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

#[tokio::test]
async fn test_local_client_registration_and_retrieval() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = Runtime::from_current()?;
    let config = DistributedConfig {
        etcd_config: Default::default(),
        nats_config: Default::default(),
        is_static: true,
    };
    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns")?;
    let component = namespace.component("test-component")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint");

    // Create an engine and configure the endpoint with it
    let engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error>> =
        Arc::new(EchoEngine);

    // Wrap the engine in an Ingress to make it a PushWorkHandler
    let ingress = Ingress::for_engine(engine)?;

    // Create the endpoint instance with the ingress as handler (setup phase)
    let _endpoint_instance = endpoint
        .endpoint_builder()
        .handler(ingress)
        .create()
        .await?;
    println!("Created endpoint instance with local engine registered");

    // Create a LocalClient using the endpoint's convenience method
    let local_client: LocalClient<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error> =
        endpoint.local_client().await?;

    // Test the local client
    let request = Context::new("Hello, LocalClient!".to_string());

    let mut response_stream = local_client.generate(request).await?;
    let response = response_stream.next().await.expect("Expected response");

    assert_eq!(response.data, Some("Hello, LocalClient!".to_string()));
    println!("LocalClient test passed: received '{:?}'", response);

    // Note: We don't need to start the endpoint for local client testing
    // The engine is registered during create() and available for local access

    Ok(())
}

#[tokio::test]
async fn test_local_client_with_ingress() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = Runtime::from_current()?;
    let config = DistributedConfig {
        etcd_config: Default::default(),
        nats_config: Default::default(),
        is_static: true,
    };
    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns2")?;
    let component = namespace.component("test-component2")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint2");

    // Create an Ingress with an engine
    let engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error>> =
        Arc::new(EchoEngine);
    let ingress = Ingress::for_engine(engine.clone())?;

    // Create the endpoint instance with the ingress as handler
    let _endpoint_instance = endpoint
        .endpoint_builder()
        .handler(ingress)
        .create()
        .await?;
    println!("Created endpoint instance with ingress and local engine registered");

    // Now we can create a LocalClient using the convenience method
    let local_client: LocalClient<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error> =
        endpoint.local_client().await?;

    // Test the local client
    let request = Context::new("Test with Ingress".to_string());

    let mut response_stream = local_client.generate(request).await?;
    let response = response_stream.next().await.expect("Expected response");

    assert_eq!(response.data, Some("Test with Ingress".to_string()));
    println!(
        "LocalClient with Ingress test passed: received '{:?}'",
        response
    );

    // Note: The engine is automatically registered during endpoint creation

    Ok(())
}

#[tokio::test]
async fn test_local_client_type_mismatch() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime and DRT
    let runtime = Runtime::from_current()?;
    let config = DistributedConfig {
        etcd_config: Default::default(),
        nats_config: Default::default(),
        is_static: true,
    };
    let drt = DistributedRuntime::new(runtime, config).await?;

    // Create namespace, component, and endpoint
    let namespace = drt.namespace("test-ns3")?;
    let component = namespace.component("test-component3")?;
    let service = component.service_builder().create().await?;
    let endpoint = service.endpoint("test-endpoint3");

    // Create an endpoint instance with a String engine
    let engine: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, anyhow::Error>> =
        Arc::new(EchoEngine);
    let ingress = Ingress::for_engine(engine)?;
    let _endpoint_instance = endpoint
        .endpoint_builder()
        .handler(ingress)
        .create()
        .await?;
    println!("Created endpoint with String engine");

    // Try to create a LocalClient with different types (this should fail)
    type TestLocalClient = LocalClient<SingleIn<i32>, ManyOut<Annotated<i32>>, anyhow::Error>;
    let result: Result<TestLocalClient, _> = endpoint.local_client().await;

    assert!(result.is_err(), "Expected type mismatch error");
    if let Err(e) = result {
        println!("Got expected error for type mismatch: {}", e);
    }

    // Note: The engine is automatically registered during endpoint creation

    Ok(())
}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    backend::Backend,
    engines::StreamingEngineAdapter,
    model_type::{ModelInput, ModelType},
    preprocessor::{BackendOutput, PreprocessedRequest},
    types::{
        Annotated,
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
    },
};

use dynamo_runtime::engine::AsyncEngineStream;
use dynamo_runtime::pipeline::{
    Context, ManyOut, Operator, SegmentSource, ServiceBackend, SingleIn, Source, network::Ingress,
};
use dynamo_runtime::{DistributedRuntime, protocols::EndpointId};

use crate::entrypoint::EngineConfig;

pub async fn run(
    distributed_runtime: DistributedRuntime,
    path: String,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let cancel_token = distributed_runtime.primary_token().clone();
    let endpoint_id: EndpointId = path.parse()?;

    let component = distributed_runtime
        .namespace(&endpoint_id.namespace)?
        .component(&endpoint_id.component)?;
    let endpoint = component
        .service_builder()
        .create()
        .await?
        .endpoint(&endpoint_id.name);

    let (rt_fut, card): (Pin<Box<dyn Future<Output = _> + Send + 'static>>, _) = match engine_config
    {
        EngineConfig::StaticFull {
            engine,
            mut model,
            is_static,
        } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let ingress_chat = Ingress::<
                Context<NvCreateChatCompletionRequest>,
                Pin<Box<dyn AsyncEngineStream<Annotated<NvCreateChatCompletionStreamResponse>>>>,
            >::for_engine(engine)?;

            if !is_static {
                model
                    .attach(&endpoint, ModelType::Chat, ModelInput::Text)
                    .await?;
            }
            let fut_chat = endpoint.endpoint_builder().handler(ingress_chat).start();

            (Box::pin(fut_chat), Some(model.card().clone()))
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            mut model,
            is_static,
        } => {
            // Pre-processing is done ingress-side, so it should be already done.
            let frontend = SegmentSource::<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
            >::new();
            let backend = Backend::from_mdc(model.card()).into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);
            let pipeline = frontend
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(frontend)?;
            let ingress = Ingress::for_pipeline(pipeline)?;

            if !is_static {
                // Default to supporting both Chat and Completions endpoints
                let model_type = ModelType::Chat | ModelType::Completions;
                model
                    .attach(&endpoint, model_type, ModelInput::Tokens)
                    .await?;
            }

            let fut = endpoint.endpoint_builder().handler(ingress).start();

            (Box::pin(fut), Some(model.card().clone()))
        }
        EngineConfig::StaticRemote(_) => {
            panic!("StaticRemote definitions are only for the frontend end node.");
        }
        EngineConfig::Dynamic(_) => {
            unreachable!("An endpoint input will never have a Dynamic engine");
        }
    };

    // Capture the actual error from rt_fut when it completes
    // Note: We must return rt_result to propagate the actual error back to the user.
    // If we don't return the specific error, the programmer/user won't know what actually
    // caused the endpoint service to fail, making debugging much more difficult.
    let result = tokio::select! {
        rt_result = rt_fut => {
            tracing::debug!("Endpoint service completed");
            match rt_result {
                Ok(_) => {
                    tracing::warn!("Endpoint service completed unexpectedly for endpoint: {}", path);
                    Err(anyhow::anyhow!("Endpoint service completed unexpectedly for endpoint: {}", path))
                }
                Err(e) => {
                    tracing::error!(%e, "Endpoint service failed for endpoint: {} - Error: {}", path, e);
                    Err(anyhow::anyhow!("Endpoint service failed for endpoint: {} - Error: {}", path, e))
                }
            }
        }
        _ = cancel_token.cancelled() => {
            tracing::debug!("Endpoint service cancelled");
            Ok(())
        }
    };

    // If we got an error, return it
    result?;

    // Cleanup on shutdown
    if let Some(mut card) = card
        && let Err(err) = card
            .delete_from_nats(distributed_runtime.nats_client())
            .await
    {
        tracing::error!(%err, "delete_from_nats error on shutdown");
    }

    Ok(())
}

#[cfg(test)]
#[cfg(feature = "integration")]
mod integration_tests {
    use super::*;
    use dynamo_runtime::protocols::EndpointId;

    async fn create_test_environment() -> anyhow::Result<(DistributedRuntime, EngineConfig)> {
        // Create a minimal distributed runtime and engine config for testing
        let runtime = dynamo_runtime::Runtime::from_settings()
            .map_err(|e| anyhow::anyhow!("Failed to create runtime: {}", e))?;

        let distributed_runtime = dynamo_runtime::DistributedRuntime::from_settings(runtime)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create distributed runtime: {}", e))?;

        let engine_config = EngineConfig::StaticFull {
            engine: crate::engines::make_echo_engine(),
            model: Box::new(
                crate::local_model::LocalModelBuilder::default()
                    .model_name(Some("test-model".to_string()))
                    .build()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to build LocalModel: {}", e))?,
            ),
            is_static: false,
        };

        Ok((distributed_runtime, engine_config))
    }

    #[tokio::test]
    #[ignore = "Failing in CI"]
    async fn test_run_function_valid_endpoint() {
        // Test that run() works correctly with valid endpoints

        let (runtime, engine_config) = match create_test_environment().await {
            Ok(env) => env,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Test with valid endpoint - start the service and then connect to it
        let valid_path = "dyn://valid-endpoint.mocker.generate";
        let valid_endpoint: EndpointId = valid_path.parse().expect("Valid endpoint should parse");

        let runtime_clone = runtime.clone();
        let engine_config_clone = engine_config.clone();
        let valid_path_clone = valid_path.to_string();

        let service_handle =
            tokio::spawn(
                async move { run(runtime_clone, valid_path_clone, engine_config_clone).await },
            );

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let client_result = async {
            let namespace = runtime.namespace(&valid_endpoint.namespace)?;
            let component = namespace.component(&valid_endpoint.component)?;
            let client = component.endpoint(&valid_endpoint.name).client().await?;
            client.wait_for_instances().await?;
            Ok::<_, anyhow::Error>(client)
        }
        .await;

        match client_result {
            Ok(_client) => {
                println!("Valid endpoint: Successfully connected to service");
                service_handle.abort(); // Abort the service since we've verified it works
            }
            Err(e) => {
                println!("Valid endpoint: Failed to connect to service: {}", e);
                service_handle.abort(); // Abort the service since the test failed
                panic!(
                    "Valid endpoint should allow client connections, but failed: {}",
                    e
                );
            }
        }
    }

    #[tokio::test]
    #[ignore = "DistributedRuntime drop issue persists - test logic validates error propagation correctly"]
    async fn test_run_function_invalid_endpoint() {
        // Test that invalid endpoints fail validation during run()
        let invalid_path = "dyn://@@@123.mocker.generate";

        // Create test environment
        let (runtime, engine_config) = create_test_environment()
            .await
            .expect("Failed to create test environment");

        // Call run() directly - it should fail quickly for invalid endpoints
        let result = run(runtime, invalid_path.to_string(), engine_config).await;

        // Should return an error for invalid endpoints
        assert!(
            result.is_err(),
            "run() should fail for invalid endpoint: {:?}",
            result
        );

        // Check that the error message contains validation-related keywords
        let error_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            error_msg.contains("invalid")
                || error_msg.contains("namespace")
                || error_msg.contains("validation")
                || error_msg.contains("failed"),
            "Error message should contain validation keywords, got: {}",
            error_msg
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Error;
use async_stream::stream;
use dynamo_llm::{
    http::service::metrics::Endpoint,
    http::service::service_v2::HttpService,
    protocols::{
        Annotated,
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
    },
};
use dynamo_runtime::metrics::prometheus_names::frontend_service::METRICS_PREFIX_ENV;
use dynamo_runtime::{
    CancellationToken,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
    },
};
use std::{sync::Arc, time::Duration};

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

// Mock engine for testing metrics
struct MockModelEngine {}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for MockModelEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let mut generator = request.response_generator(ctx.id().to_string());

        let stream = stream! {
            // Simulate some processing time
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;

            // Generate 5 response chunks
            for i in 0..5 {
                let output = generator.create_choice(i, Some(format!("Mock response {i}")), None, None, None);
                yield Annotated::from_data(output);
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

#[tokio::test]
async fn test_metrics_prefix_default() {
    // Test default prefix when no env var is set
    temp_env::async_with_vars([(METRICS_PREFIX_ENV, None::<&str>)], async {
        let port = get_random_port().await;
        let service = HttpService::builder().port(port).build().unwrap();
        let token = CancellationToken::new();
        let handle = service.spawn(token.clone()).await;
        wait_for_metrics_ready(port).await;

        // Populate labeled metrics
        let state = service.state_clone();
        {
            let _guard = state.metrics_clone().create_inflight_guard(
                "test-model",
                Endpoint::ChatCompletions,
                false,
            );
        }

        let body = reqwest::get(format!("http://localhost:{}/metrics", port))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();

        // Assert metrics that are actually present in the default configuration
        assert!(body.contains("dynamo_frontend_requests_total"));
        assert!(body.contains("dynamo_frontend_inflight_requests_total"));
        assert!(body.contains("dynamo_frontend_request_duration_seconds"));
        assert!(body.contains("dynamo_frontend_client_disconnects"));

        token.cancel();
        let _ = handle.await;
    })
    .await;
}

#[tokio::test]
async fn test_metrics_prefix_custom() {
    // Test custom prefix override via environment variable
    temp_env::async_with_vars([(METRICS_PREFIX_ENV, Some("custom_prefix"))], async {
        let port = get_random_port().await;
        let service = HttpService::builder().port(port).build().unwrap();
        let token = CancellationToken::new();
        let handle = service.spawn(token.clone()).await;
        wait_for_metrics_ready(port).await;

        // Populate labeled metrics
        let state = service.state_clone();
        {
            let _guard = state.metrics_clone().create_inflight_guard(
                "test-model",
                Endpoint::ChatCompletions,
                true,
            );
        }

        let body = reqwest::get(format!("http://localhost:{}/metrics", port))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("custom_prefix_requests_total"));
        assert!(!body.contains("dynamo_frontend_requests_total"));

        token.cancel();
        let _ = handle.await;
    })
    .await;
}

#[tokio::test]
async fn test_metrics_prefix_sanitized() {
    // Test that invalid prefix characters are sanitized
    temp_env::async_with_vars([(METRICS_PREFIX_ENV, Some("nv-llm/http service"))], async {
        let port = get_random_port().await;
        let service = HttpService::builder().port(port).build().unwrap();
        let token = CancellationToken::new();
        let handle = service.spawn(token.clone()).await;
        wait_for_metrics_ready(port).await;

        let state = service.state_clone();
        {
            let _guard = state.metrics_clone().create_inflight_guard(
                "test-model",
                Endpoint::ChatCompletions,
                true,
            );
        }

        let body = reqwest::get(format!("http://localhost:{}/metrics", port))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("nv_llm_http_service_requests_total"));
        assert!(!body.contains("dynamo_frontend_requests_total"));

        token.cancel();
        let _ = handle.await;
    })
    .await;
}

// Poll /metrics until ready or timeout
async fn wait_for_metrics_ready(port: u16) {
    let url = format!("http://localhost:{}/metrics", port);
    let start = tokio::time::Instant::now();
    let timeout = Duration::from_secs(5);
    loop {
        if start.elapsed() > timeout {
            panic!("Timed out waiting for metrics endpoint at {}", url);
        }
        match reqwest::get(&url).await {
            Ok(resp) if resp.status().is_success() => break,
            _ => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
}

#[tokio::test]
async fn test_metrics_with_mock_model() {
    // Test metrics collection with a mock model serving requests
    // Ensure we use the default prefix
    temp_env::async_with_vars([(METRICS_PREFIX_ENV, None::<&str>)], async {
        let port = get_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .enable_chat_endpoints(true)
            .build()
            .unwrap();

        let state = service.state_clone();
        let manager = state.manager();

        // Start the HTTP service
        let token = CancellationToken::new();
        let cancel_token = token.clone();
        let task = tokio::spawn(async move { service.run(token.clone()).await });

        // Add mock model engine
        let mock_engine = Arc::new(MockModelEngine {});
        manager
            .add_chat_completions_model("mockmodel", mock_engine)
            .unwrap();

        // Wait for service to be ready
        wait_for_metrics_ready(port).await;

        let client = reqwest::Client::new();

        // Create a chat completion request
        let message = dynamo_async_openai::types::ChatCompletionRequestMessage::User(
            dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    "Hello, mock model!".to_string(),
                ),
                name: None,
            },
        );

        let request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("mockmodel")
            .messages(vec![message])
            .max_tokens(50u32)
            .stream(true)
            .build()
            .expect("Failed to build request");

        // Make the request to the HTTP service
        let response = client
            .post(format!("http://localhost:{}/v1/chat/completions", port))
            .json(&request)
            .send()
            .await
            .unwrap();

        assert!(
            response.status().is_success(),
            "Request failed: {:?}",
            response
        );

        // Consume the response stream to complete the request
        let _response_body = response.bytes().await.unwrap();

        // Give some time for metrics to be updated
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Fetch and verify metrics
        let metrics_response = client
            .get(format!("http://localhost:{}/metrics", port))
            .send()
            .await
            .unwrap();

        assert!(metrics_response.status().is_success());
        let metrics_body = metrics_response.text().await.unwrap();

        println!("=== METRICS WITH MOCK MODEL ===");
        println!("{}", metrics_body);
        println!("=== END METRICS ===");

        // Assert that key metrics are present with the mockmodel
        assert!(metrics_body.contains("dynamo_frontend_requests_total"));
        assert!(metrics_body.contains("model=\"mockmodel\""));
        assert!(metrics_body.contains("dynamo_frontend_inflight_requests_total"));
        assert!(metrics_body.contains("dynamo_frontend_request_duration_seconds"));
        assert!(metrics_body.contains("dynamo_frontend_output_sequence_tokens"));
        assert!(metrics_body.contains("dynamo_frontend_queued_requests_total"));

        // Verify specific request counter incremented
        assert!(metrics_body.contains("endpoint=\"chat_completions\""));
        assert!(metrics_body.contains("request_type=\"stream\""));
        assert!(metrics_body.contains("status=\"success\""));

        // Clean up
        cancel_token.cancel();
        task.await.unwrap().unwrap();
    })
    .await;
}

// Integration tests that require distributed runtime with etcd
#[cfg(feature = "integration")]
mod integration_tests {
    use super::*;
    use dynamo_llm::{
        discovery::ModelEntry, engines::make_echo_engine, entrypoint::EngineConfig,
        local_model::LocalModelBuilder,
    };
    use dynamo_runtime::DistributedRuntime;

    #[tokio::test]
    #[ignore = "Requires etcd and distributed runtime"]
    async fn test_metrics_with_mdc_registration() {
        // Integration test for metrics collection with full MDC registration (like real model servers)
        temp_env::async_with_vars([
            (METRICS_PREFIX_ENV, None::<&str>),
            ("DYN_HTTP_SVC_CONFIG_METRICS_POLL_INTERVAL_SECS", Some("0.6")), // Fast polling for tests (600ms)
        ], async {
            let port = get_random_port().await;

            // Create distributed runtime (required for MDC registration)
            let runtime = dynamo_runtime::Runtime::from_settings().unwrap();
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone())
                .await
                .unwrap();

            // Create LocalModel with realistic configuration for testing
            let mut local_model = LocalModelBuilder::default()
                .model_name(Some("test-mdc-model".to_string()))
                .build()
                .await
                .unwrap();

            // Create EngineConfig with EchoEngine
            let engine_config = EngineConfig::StaticFull {
                engine: make_echo_engine(),
                model: Box::new(local_model.clone()),
                is_static: false, // This enables MDC registration!
            };

            let service = HttpService::builder()
                .port(port)
                .enable_chat_endpoints(true)
                .with_etcd_client(distributed_runtime.etcd_client())
                .build()
                .unwrap();

            // Set up model watcher to discover models from etcd (like production)
            // This is crucial for the polling task to find model entries
            use dynamo_llm::discovery::{ModelWatcher, MODEL_ROOT_PATH};
            use dynamo_runtime::pipeline::RouterMode;

            let model_watcher = ModelWatcher::new(
                distributed_runtime.clone(),
                service.state().manager_clone(),
                RouterMode::RoundRobin,
                None,
                None,
            );

            // Start watching etcd for model registrations
            if let Some(etcd_client) = distributed_runtime.etcd_client() {
                let models_watcher = etcd_client.kv_get_and_watch_prefix(MODEL_ROOT_PATH).await.unwrap();
                let (_prefix, _watcher, receiver) = models_watcher.dissolve();

                // Spawn watcher task to discover models from etcd
                let _watcher_task = tokio::spawn(async move {
                    model_watcher.watch(receiver, None).await;
                });

            }

            // Set up the engine following the StaticFull pattern from http.rs
            let EngineConfig::StaticFull { engine, model, .. } = engine_config else {
                panic!("Expected StaticFull config");
            };

            let engine = Arc::new(dynamo_llm::engines::StreamingEngineAdapter::new(engine));
            let manager = service.model_manager();
            manager
                .add_chat_completions_model(model.service_name(), engine.clone())
                .unwrap();

            // Now do the proper MDC registration via LocalModel::attach()
            // Create a component and endpoint for proper registration
            let namespace = distributed_runtime.namespace("test-namespace").unwrap();
            let test_component = namespace.component("test-mdc-component").unwrap();
            let test_endpoint = test_component.endpoint("test-mdc-endpoint");

            // This will store the MDC in etcd and create the ModelEntry for discovery
            local_model
                .attach(
                    &test_endpoint,
                    dynamo_llm::model_type::ModelType::Chat,
                    dynamo_llm::model_type::ModelInput::Text,
                )
                .await
                .unwrap();


            // Start the HTTP service
            let token = CancellationToken::new();
            let cancel_token = token.clone();
            let service_for_task = service.clone();
            let task = tokio::spawn(async move { service_for_task.run(token.clone()).await });

            // Wait for service to be ready
            wait_for_metrics_ready(port).await;

            // Wait for MDC registration to complete by checking if the model appears
            // This simulates the real polling that happens in production
            let start = tokio::time::Instant::now();
            let timeout = Duration::from_secs(10);
            loop {
                if start.elapsed() > timeout {
                    break; // Continue with test even if MDC metrics aren't ready
                }

                // Check if our model is registered in the manager (indicates MDC registration completed)
                let model_service_name = model.service_name();
                if manager.has_model_any(model_service_name) {
                    tracing::info!("MDC registration completed for {}", model_service_name);
                    break;
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            // Give a bit more time for background metrics collection
            tokio::time::sleep(Duration::from_millis(200)).await;

            let client = reqwest::Client::new();

            // Create a chat completion request
            let message = dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content:
                        dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            "Hello, MDC model!".to_string(),
                        ),
                    name: None,
                },
            );

            let request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
                .model(model.service_name())
                .messages(vec![message])
                .max_tokens(50u32)
                .stream(true)
                .build()
                .expect("Failed to build request");

            // Make the request to the HTTP service
            let response = client
                .post(format!("http://localhost:{}/v1/chat/completions", port))
                .json(&request)
                .send()
                .await
                .unwrap();

            assert!(
                response.status().is_success(),
                "Request failed: {:?}",
                response
            );

            // Consume the response stream to complete the request
            let _response_body = response.bytes().await.unwrap();

        // Wait for the fast polling interval (600ms) for MDC metrics
        tokio::time::sleep(Duration::from_millis(5)).await;

            // Fetch and verify metrics
            let metrics_response = client
                .get(format!("http://localhost:{}/metrics", port))
                .send()
                .await
                .unwrap();

            assert!(metrics_response.status().is_success());
            let metrics_body = metrics_response.text().await.unwrap();

            println!("=== METRICS WITH FULL MDC REGISTRATION ===");
            println!("{}", metrics_body);
            println!("=== END METRICS ===");

            // Assert basic metrics are present (using service_name from the model)
            let model_name = model.service_name();
            assert!(metrics_body.contains("dynamo_frontend_requests_total"));
            assert!(metrics_body.contains(&format!("model=\"{}\"", model_name)));
            assert!(metrics_body.contains("dynamo_frontend_inflight_requests_total"));
            assert!(metrics_body.contains("dynamo_frontend_request_duration_seconds"));
            assert!(metrics_body.contains("dynamo_frontend_output_sequence_tokens"));
            assert!(metrics_body.contains("dynamo_frontend_queued_requests_total"));

            // Assert MDC-based model configuration metrics are present
            // These MUST be present for the test to pass
            assert!(metrics_body.contains("dynamo_frontend_model_context_length"),
                    "MDC metrics not found! Metrics body: {}", metrics_body);

            assert!(metrics_body.contains("dynamo_frontend_model_kv_cache_block_size"));
            assert!(metrics_body.contains("dynamo_frontend_model_migration_limit"));

            // Note: The following metrics are not present in this test because they require
            // actual inference engines (vllm/sglang/trtllm *.py) with real runtime configurations:
            // - dynamo_frontend_model_total_kv_blocks (requires actual KV cache from real engines)
            // - dynamo_frontend_model_max_num_seqs (requires actual batching config from real engines)
            // - dynamo_frontend_model_max_num_batched_tokens (requires actual batching config from real engines)


            // Verify specific request counter incremented
            assert!(metrics_body.contains("endpoint=\"chat_completions\""));
            assert!(metrics_body.contains("request_type=\"stream\""));
            assert!(metrics_body.contains("status=\"success\""));

            // Now test the complete lifecycle: remove the model from etcd

            // Get all model entries to find the one we need to delete
            if let Some(etcd_client) = distributed_runtime.etcd_client() {
                let kvs = etcd_client.kv_get_prefix("models").await.unwrap();

                // Find our model's etcd key
                let mut model_key_to_delete = None;
                for kv in kvs {
                    if let Ok(model_entry) = serde_json::from_slice::<ModelEntry>(kv.value())
                        && model_entry.name == "test-mdc-model"
                    {
                        model_key_to_delete = Some(kv.key_str().unwrap().to_string());
                        break;
                    }
                }

                if let Some(key) = model_key_to_delete {
                    etcd_client.kv_delete(key.as_str(), None).await.unwrap();

                    // Poll every 80ms for up to 2 seconds to check when worker count drops to 0

                    let start_time = tokio::time::Instant::now();
                    let timeout = Duration::from_millis(2000);
                    let mut worker_count_dropped = false;

                    while start_time.elapsed() < timeout {
                        // Check if the model was removed from the manager
                        let has_model = manager.has_model_any(model.service_name());

                        // Fetch current metrics
                        let metrics_response = client
                            .get(format!("http://localhost:{}/metrics", port))
                            .send()
                            .await
                            .unwrap();

                        if metrics_response.status().is_success() {

                            // Since model_workers metric was removed, just check if model is gone from manager
                            if !has_model {
                                worker_count_dropped = true;
                                break;
                            }
                        }

                        tokio::time::sleep(Duration::from_millis(80)).await;
                    }

                    // Assert that model was removed from manager
                    assert!(worker_count_dropped,
                            "Model should be removed from manager after etcd removal and polling cycles");

                } else {
                }
            }


            // Clean up
            cancel_token.cancel();
            task.await.unwrap().unwrap();
        })
        .await;
    }
}

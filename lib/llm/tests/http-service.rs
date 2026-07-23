// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Error;
use async_stream::stream;
use dynamo_llm::protocols::{
    Annotated,
    codec::SseLineCodec,
    convert_sse_stream,
    openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_llm::{
    http::service::{
        Metrics,
        error::HttpError,
        metrics::{Endpoint, ErrorType, RequestType, Status},
        service_v2::HttpService,
    },
    model_card::ModelDeploymentCard,
};
use dynamo_runtime::metrics::prometheus_names::{frontend_service, name_prefix};
use dynamo_runtime::{
    CancellationToken,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
    },
};
use futures::StreamExt;
use prometheus::{Registry, proto::MetricType};
use reqwest::StatusCode;
use std::{io::Cursor, sync::Arc};
use tokio::time::timeout;
use tokio_util::codec::FramedRead;

#[path = "common/ports.rs"]
mod ports;
use ports::bind_random_port;

struct CounterEngine {}

// Add a new long-running test engine
struct LongRunningEngine {
    delay_ms: u64,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl LongRunningEngine {
    fn new(delay_ms: u64) -> Self {
        Self {
            delay_ms,
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    fn was_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for CounterEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
        #[allow(deprecated)]
        let max_tokens = request.inner.max_tokens.unwrap_or(0) as u64;

        // let generator = NvCreateChatCompletionStreamResponse::generator(request.model.clone());
        let mut generator = request.response_generator(ctx.id().to_string());

        let stream = stream! {
            tokio::time::sleep(std::time::Duration::from_millis(max_tokens)).await;
            for i in 0..10 {
                let output = generator.create_choice(i, Some(format!("choice {i}")), None, None);

                yield Annotated::from_data(output);
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for LongRunningEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (_request, context) = request.transfer(());
        let ctx = context.context();

        tracing::info!(
            "LongRunningEngine: Starting generation with {}ms delay",
            self.delay_ms
        );

        let cancelled_flag = self.cancelled.clone();
        let delay_ms = self.delay_ms;

        let ctx_clone = ctx.clone();
        let stream = async_stream::stream! {

            // the stream can be dropped or it can be cancelled
            // either way we consider this a cancellation
            cancelled_flag.store(true, std::sync::atomic::Ordering::SeqCst);

            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {
                    // the stream went to completion
                    cancelled_flag.store(false, std::sync::atomic::Ordering::SeqCst);

                }
                _ = ctx_clone.stopped() => {
                    cancelled_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }

            yield Annotated::<NvCreateChatCompletionStreamResponse>::from_annotation("event.dynamo.test.sentinel", &"DONE".to_string()).expect("Failed to create annotated response");
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

struct AlwaysFailEngine {}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for AlwaysFailEngine
{
    async fn generate(
        &self,
        _request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        Err(HttpError {
            code: 403,
            message: "Always fail".to_string(),
        })?
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        Error,
    > for AlwaysFailEngine
{
    async fn generate(
        &self,
        _request: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        Err(HttpError {
            code: 401,
            message: "Always fail".to_string(),
        })?
    }
}

fn compare_counter(
    metrics: &Metrics,
    model: &str,
    endpoint: &Endpoint,
    request_type: &RequestType,
    status: &Status,
    error_type: &ErrorType,
    expected: u64,
) {
    assert_eq!(
        metrics.get_request_counter(model, endpoint, request_type, status, error_type),
        expected,
        "model: {}, endpoint: {:?}, request_type: {:?}, status: {:?}, error_type: {:?}",
        model,
        endpoint.as_str(),
        request_type.as_str(),
        status.as_str(),
        error_type.as_str()
    );
}

fn compute_index(endpoint: &Endpoint, request_type: &RequestType, status: &Status) -> usize {
    let endpoint = match endpoint {
        Endpoint::Completions => 0,
        Endpoint::ChatCompletions => 1,
        Endpoint::Embeddings => todo!(),
        Endpoint::Responses => todo!(),
        Endpoint::AnthropicMessages => todo!(),
        Endpoint::Tensor => todo!(),
        Endpoint::Images => todo!(),
        Endpoint::Videos => todo!(),
        Endpoint::Audios => todo!(),
        Endpoint::Generate => todo!(),
    };

    let request_type = match request_type {
        RequestType::Unary => 0,
        RequestType::Stream => 1,
    };

    let status = match status {
        Status::Success => 0,
        Status::Error => 1,
    };

    endpoint * 4 + request_type * 2 + status
}

fn compare_counters(metrics: &Metrics, model: &str, expected: &[u64; 8]) {
    for endpoint in &[Endpoint::Completions, Endpoint::ChatCompletions] {
        for request_type in &[RequestType::Unary, RequestType::Stream] {
            for status in &[Status::Success, Status::Error] {
                let index = compute_index(endpoint, request_type, status);
                let error_type = match status {
                    Status::Success => &ErrorType::None,
                    Status::Error => &ErrorType::Validation, // Test engines return 4xx errors
                };
                compare_counter(
                    metrics,
                    model,
                    endpoint,
                    request_type,
                    status,
                    error_type,
                    expected[index],
                );
            }
        }
    }
}

fn inc_counter(
    endpoint: Endpoint,
    request_type: RequestType,
    status: Status,
    expected: &mut [u64; 8],
) {
    let index = compute_index(&endpoint, &request_type, &status);
    expected[index] += 1;
}

#[allow(deprecated)]
#[tokio::test]
async fn test_http_service() {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task =
        tokio::spawn(async move { service.run_with_listener(token.clone(), listener).await });

    // Wait for the service to be ready before proceeding
    wait_for_service_ready(port).await;

    let registry = Registry::new();

    // TODO: Shouldn't this test know the card before it registers a model?
    let card = ModelDeploymentCard::with_name_only("foo");
    let counter = Arc::new(CounterEngine {});
    let result = manager.add_chat_completions_model("foo", card.mdcsum(), counter);
    assert!(result.is_ok());

    let failure = Arc::new(AlwaysFailEngine {});
    let card = ModelDeploymentCard::with_name_only("bar");
    let result = manager.add_chat_completions_model("bar", card.mdcsum(), failure.clone());
    assert!(result.is_ok());

    let result = manager.add_completions_model("bar", card.mdcsum(), failure);
    assert!(result.is_ok());

    let metrics = state.metrics_clone();
    metrics.register(&registry).unwrap();

    let mut foo_counters = [0u64; 8];
    let mut bar_counters = [0u64; 8];

    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);

    let client = reqwest::Client::new();

    let message = dynamo_protocols::types::ChatCompletionRequestMessage::User(
        dynamo_protocols::types::ChatCompletionRequestUserMessage {
            content: dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                "hi".to_string(),
            ),
            name: None,
        },
    );

    let mut request = dynamo_protocols::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![message])
        .build()
        .expect("Failed to build request");

    // let mut request = ChatCompletionRequest::builder()
    //     .model("foo")
    //     .add_user_message("hi")
    //     .build()
    //     .unwrap();

    // ==== ChatCompletions / Stream / Success ====
    request.stream = Some(true);

    // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
    request.max_tokens = Some(3000);

    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "{:?}", response);

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    assert_eq!(metrics.get_inflight_count("foo"), 1);

    // process byte stream
    let _ = response.bytes().await.unwrap();

    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Stream,
        Status::Success,
        &mut foo_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);

    // check registry and look or the request duration histogram
    let families = registry.gather();
    let histogram_metric_family = families
        .into_iter()
        .find(|m| {
            m.get_name()
                == format!(
                    "{}_{}",
                    name_prefix::FRONTEND,
                    frontend_service::REQUEST_DURATION_SECONDS
                )
        })
        .expect("Histogram metric not found");

    assert_eq!(
        histogram_metric_family.get_field_type(),
        MetricType::HISTOGRAM
    );

    let histogram_metric = histogram_metric_family.get_metric();

    assert_eq!(histogram_metric.len(), 1); // We have one metric with label model

    let metric = &histogram_metric[0];
    let histogram = metric.get_histogram();

    let buckets = histogram.get_bucket();

    let mut found = false;
    let mut expected_count = 0;
    for bucket_idx in 1..buckets.len() {
        if buckets[bucket_idx].get_upper_bound() >= 2.5
            && buckets[bucket_idx - 1].get_upper_bound() < 2.5
        {
            found = true;
            assert_eq!(
                buckets[bucket_idx].get_cumulative_count(),
                1,
                "Observation should be counted in the bucket containing 2.5"
            );
            expected_count = 1;
        } else {
            assert_eq!(
                buckets[bucket_idx].get_cumulative_count(),
                expected_count,
                "No observations should be in this bucket"
            );
        }
    }

    assert!(found, "The expected bucket was not found");
    // ==== ChatCompletions / Stream / Success ====

    // ==== ChatCompletions / Unary / Success ====
    request.stream = Some(false);

    // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
    request.max_tokens = Some(0);

    let future = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(&request)
        .send();

    let response = future.await.unwrap();

    assert!(response.status().is_success(), "{:?}", response);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Unary,
        Status::Success,
        &mut foo_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== ChatCompletions / Unary / Success ====

    // ==== ChatCompletions / Stream / Error ====
    request.model = "bar".to_string();

    // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
    request.max_tokens = Some(0);
    request.stream = Some(true);

    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Stream,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== ChatCompletions / Stream / Error ====

    // ==== ChatCompletions / Unary / Error ====
    request.stream = Some(false);

    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Unary,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== ChatCompletions / Unary / Error ====

    // ==== Completions / Unary / Error ====
    let mut request = dynamo_protocols::types::CreateCompletionRequestArgs::default()
        .model("bar")
        .prompt("hi")
        .build()
        .unwrap();

    let response = client
        .post(format!("http://localhost:{}/v1/completions", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    inc_counter(
        Endpoint::Completions,
        RequestType::Unary,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== Completions / Unary / Error ====

    // ==== Completions / Stream / Error ====
    request.stream = Some(true);

    let response = client
        .post(format!("http://localhost:{}/v1/completions", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    inc_counter(
        Endpoint::Completions,
        RequestType::Stream,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== Completions / Stream / Error ====

    // =========== Test Invalid Request ===========
    // send a completion request to a chat endpoint
    request.stream = Some(false);

    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{:?}", response);

    // =========== Query /metrics endpoint ===========
    let response = client
        .get(format!("http://localhost:{}/metrics", port))
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "{:?}", response);
    println!("{}", response.text().await.unwrap());

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

// === HTTP Client Tests ===

/// Wait for the HTTP service to be ready by checking its health endpoint
async fn wait_for_service_ready(port: u16) {
    let start = tokio::time::Instant::now();
    let timeout = tokio::time::Duration::from_secs(5);
    loop {
        match reqwest::get(&format!("http://localhost:{}/health", port)).await {
            Ok(_) => break,
            Err(_) if start.elapsed() < timeout => {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
            Err(e) => panic!("Service failed to start within timeout: {}", e),
        }
    }
}

#[tokio::test]
async fn test_batch_api_skeleton_routes_return_not_implemented() {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .enable_batch_endpoints(true)
        .build()
        .unwrap();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run_with_listener(token, listener).await });
    wait_for_service_ready(port).await;

    let client = reqwest::Client::new();
    let base = format!("http://localhost:{port}");

    let response = client
        .post(format!("{base}/v1/files"))
        .body("{\"custom_id\":\"r1\"}\n")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["code"], 501);
    assert_eq!(
        body["message"],
        "Batch file storage is not implemented yet."
    );

    let response = client
        .post(format!("{base}/v1/batches"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body("not valid JSON")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["code"], 501);
    assert_eq!(
        body["message"],
        "Batch job lifecycle persistence is not implemented yet."
    );

    let response = client
        .get(format!("{base}/v1/batches/batch-123"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["code"], 501);
    assert_eq!(
        body["message"],
        "Batch job lifecycle persistence is not implemented yet."
    );

    let response = client
        .get(format!("{base}/v1/files/file-123/content"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(
        body["message"],
        "Batch output file retrieval is not implemented yet."
    );

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

// NOTE: BYOT (Bring Your Own Type) client tests were removed during the
// upstream async-openai migration. They depended on the forked
// dynamo_protocols::config and http::client modules which no longer exist.
// TODO: Rewrite these tests using the upstream async-openai client.
#[tokio::test]
async fn test_client_disconnect_cancellation_unary() {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .port(port)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run_with_listener(token, listener).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Create a long-running engine (10 seconds)
    let card = ModelDeploymentCard::with_name_only("slow-model");
    let long_running_engine = Arc::new(LongRunningEngine::new(10_000));
    manager
        .add_chat_completions_model("slow-model", card.mdcsum(), long_running_engine.clone())
        .unwrap();

    let client = reqwest::Client::new();

    let message = dynamo_protocols::types::ChatCompletionRequestMessage::User(
        dynamo_protocols::types::ChatCompletionRequestUserMessage {
            content: dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                "This will take a long time".to_string(),
            ),
            name: None,
        },
    );

    let request = dynamo_protocols::types::CreateChatCompletionRequestArgs::default()
        .model("slow-model")
        .messages(vec![message])
        .stream(false) // Test unary response
        .build()
        .expect("Failed to build request");

    // Start the request and cancel it after 1 second
    let start_time = std::time::Instant::now();

    let request_future = async {
        client
            .post(format!("http://localhost:{}/v1/chat/completions", port))
            .json(&request)
            .send()
            .await
    };

    // Use timeout to simulate client disconnect after 1 second
    let result = timeout(std::time::Duration::from_millis(1000), request_future).await;

    let elapsed = start_time.elapsed();

    // The request should timeout (simulating client disconnect)
    assert!(result.is_err(), "Request should have timed out");

    // Give the service a moment to detect the disconnect and propagate cancellation
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Verify the engine was cancelled
    assert!(
        long_running_engine.was_cancelled(),
        "Engine should have been cancelled due to client disconnect"
    );

    // Verify cancellation happened quickly (within 2 seconds, not the full 10 seconds)
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "Cancellation should have propagated quickly, took {:?}",
        elapsed
    );

    tracing::info!(
        "✅ Client disconnect test passed! Request cancelled in {:?}, engine detected cancellation",
        elapsed
    );

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_client_disconnect_cancellation_streaming() {
    dynamo_runtime::logging::init();

    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .port(port)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run_with_listener(token, listener).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Create a long-running engine (10 seconds)
    let card = ModelDeploymentCard::with_name_only("slow-stream-model");
    let long_running_engine = Arc::new(LongRunningEngine::new(10_000));
    manager
        .add_chat_completions_model(
            "slow-stream-model",
            card.mdcsum(),
            long_running_engine.clone(),
        )
        .unwrap();

    let client = reqwest::Client::new();

    let message = dynamo_protocols::types::ChatCompletionRequestMessage::User(
        dynamo_protocols::types::ChatCompletionRequestUserMessage {
            content: dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                "This will stream for a long time".to_string(),
            ),
            name: None,
        },
    );

    let request = dynamo_protocols::types::CreateChatCompletionRequestArgs::default()
        .model("slow-stream-model")
        .messages(vec![message])
        .stream(true) // Test streaming response
        .build()
        .expect("Failed to build request");

    // Start the request and cancel it after 1 second
    let start_time = std::time::Instant::now();

    let request_future = async {
        let response = client
            .post(format!("http://localhost:{}/v1/chat/completions", port))
            .json(&request)
            .send()
            .await
            .unwrap();

        // Start reading the stream, then drop it to simulate client disconnect
        let mut stream = response.bytes_stream();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Read one chunk then drop the stream (simulating client disconnect)
        let _ = StreamExt::next(&mut stream).await;
        // Stream gets dropped here when function exits
    };

    // Use timeout to simulate the streaming request timing out
    let _result = timeout(std::time::Duration::from_millis(1500), request_future).await;

    let elapsed = start_time.elapsed();

    // Give the service time to detect the disconnect
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Verify the engine was cancelled
    assert!(
        long_running_engine.was_cancelled(),
        "Engine should have been cancelled due to streaming client disconnect"
    );

    // Verify cancellation happened reasonably quickly
    assert!(
        elapsed < std::time::Duration::from_secs(3),
        "Stream cancellation should have propagated reasonably quickly, took {:?}",
        elapsed
    );

    tracing::info!(
        "✅ Streaming client disconnect test passed! Stream cancelled in {:?}, engine detected cancellation",
        elapsed
    );

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_request_id_annotation() {
    // TODO(ryan): make better fixtures, this is too much to test sometime so simple
    dynamo_runtime::logging::init();

    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .port(port)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run_with_listener(token, listener).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Add a counter engine for this test
    let card = ModelDeploymentCard::with_name_only("test-model");
    let counter_engine = Arc::new(CounterEngine {});
    manager
        .add_chat_completions_model("test-model", card.mdcsum(), counter_engine)
        .unwrap();

    // Create reqwest client directly
    let client = reqwest::Client::new();

    // Generate a UUID for the request ID
    let request_uuid = uuid::Uuid::new_v4();

    // Create the request JSON directly
    let request_json = serde_json::json!({
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "Test request with annotation"
            }
        ],
        "stream": true,
        "max_tokens": 50,
        "nvext": {
            "annotations": ["request_id"]
        }
    });

    // Make the streaming request with custom header
    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .header("x-dynamo-request-id", request_uuid.to_string())
        .json(&request_json)
        .send()
        .await
        .expect("Request should succeed");

    assert!(
        response.status().is_success(),
        "Response should be successful"
    );

    // Collect the entire response body as bytes first
    let body_bytes = response
        .bytes()
        .await
        .expect("Failed to read response body");
    let body_text = String::from_utf8_lossy(&body_bytes);

    // Create a cursor from the text and use SseLineCodec to parse it
    let cursor = Cursor::new(body_text.to_string());
    let framed = FramedRead::new(cursor, SseLineCodec::new());
    let annotated_stream = convert_sse_stream::<NvCreateChatCompletionStreamResponse>(framed);

    // Look for the annotation in the stream
    let mut found_request_id_annotation = false;
    let mut received_request_id = None;

    // Process the annotated stream and look for the request_id annotation
    let mut annotated_stream = std::pin::pin!(annotated_stream);
    while let Some(annotated_response) = annotated_stream.next().await {
        // Check if this is a request_id annotation
        if let Some(event) = &annotated_response.event
            && event == "request_id"
        {
            found_request_id_annotation = true;
            // Extract the request ID from the annotation
            if let Some(comments) = &annotated_response.comment
                && let Some(comment) = comments.first()
            {
                // The comment contains a JSON-encoded string, so we need to parse it
                if let Ok(parsed_value) = serde_json::from_str::<String>(comment) {
                    received_request_id = Some(parsed_value);
                } else {
                    // Fallback: remove quotes manually if JSON parsing fails
                    received_request_id = Some(comment.trim_matches('"').to_string());
                }
            }
            break;
        }
    }

    // Verify we found the annotation
    assert!(
        found_request_id_annotation,
        "Should have received request_id annotation in the stream"
    );

    // Verify the request ID matches what we sent
    assert!(
        received_request_id.is_some(),
        "Should have received the request ID in the annotation"
    );

    let received_uuid_str = received_request_id.unwrap();
    assert_eq!(
        received_uuid_str,
        request_uuid.to_string(),
        "Received request ID should match the one we sent: expected {}, got {}",
        request_uuid,
        received_uuid_str
    );

    tracing::info!(
        "✅ Request ID annotation test passed! Sent UUID: {}, Received: {}",
        request_uuid,
        received_uuid_str
    );

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

/// Exercises the per-model readiness sub-resource `GET /v1/models/{model}/ready`
/// (Mechanism 4) end-to-end through the real router, including:
///   - the endpoint returns the structured readiness body (not the OpenAI
///     retrieve object),
///   - the old `/readiness` path is retired (404),
///   - a model literally named `.../ready` shadows the sub-resource (exact
///     model match wins), and
///   - an unknown model with a `/ready` suffix is a 404.
#[tokio::test]
async fn test_model_ready_endpoint() {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .enable_chat_endpoints(true)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run_with_listener(token, listener).await });
    wait_for_service_ready(port).await;

    // A normal, ready in-process model.
    let card = ModelDeploymentCard::with_name_only("foo");
    manager
        .add_chat_completions_model("foo", card.mdcsum(), Arc::new(CounterEngine {}))
        .unwrap();

    // A model whose *name* ends in `/ready` — must never be shadowed by the
    // readiness sub-resource (exact-match precedence in `get_model_openai`).
    let shadow_card = ModelDeploymentCard::with_name_only("shadow/ready");
    manager
        .add_chat_completions_model(
            "shadow/ready",
            shadow_card.mdcsum(),
            Arc::new(CounterEngine {}),
        )
        .unwrap();

    let client = reqwest::Client::new();
    let base = format!("http://localhost:{}/v1/models", port);

    // 1. `/ready` returns the structured readiness body, not the retrieve object.
    let resp = client
        .get(format!("{base}/foo/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "/foo/ready should be 200");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["model"], "foo",
        "readiness body carries the model name"
    );
    assert!(
        body.get("namespaces").is_some(),
        "readiness body has a namespaces map, got: {body}"
    );
    assert!(
        body.get("object").is_none(),
        "readiness body must not be the OpenAI retrieve object, got: {body}"
    );

    // 2. The old `/readiness` path is retired — 404.
    let resp = client
        .get(format!("{base}/foo/readiness"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "old /readiness path must be gone"
    );

    // 3. A model literally named `shadow/ready` resolves to the retrieve object,
    //    NOT the readiness sub-resource of a model named `shadow`.
    let resp = client
        .get(format!("{base}/shadow/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "/shadow/ready should be 200");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["object"], "model",
        "exact model match wins over the /ready sub-resource, got: {body}"
    );
    assert_eq!(body["id"], "shadow/ready");

    // 4. Unknown model with a `/ready` suffix is a 404 (no base model to gate).
    let resp = client
        .get(format!("{base}/ghost/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "/ready on an unknown model is 404"
    );

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

/// Regression: exact-match precedence must hold for a *non-displayable* model
/// whose ID ends in `/ready`. Such a model is absent from `model_display_names()`,
/// so keying the exact-match check off the displayable set (the earlier bug)
/// would fall through to the `/ready` sub-resource and return a sibling `foo`'s
/// readiness — shadowing the registered `foo/ready`. Exact match must win for
/// *any* registered model, displayable or not.
#[tokio::test]
async fn test_model_ready_endpoint_non_displayable_shadow() {
    use dynamo_llm::discovery::WorkerSet;
    use dynamo_llm::worker_type::WorkerType;

    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .enable_chat_endpoints(true)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run_with_listener(token, listener).await });
    wait_for_service_ready(port).await;

    // Base model `foo`: a normal, ready in-process model.
    let foo = ModelDeploymentCard::with_name_only("foo");
    manager
        .add_chat_completions_model("foo", foo.mdcsum(), Arc::new(CounterEngine {}))
        .unwrap();

    // `foo/ready`: registered but NOT displayable (no serving engine) and NOT
    // ready (decode worker type whose prefill peer is absent).
    let mut card = ModelDeploymentCard::with_name_only("foo/ready");
    card.worker_type = Some(WorkerType::Decode);
    card.needs = vec![vec![WorkerType::Prefill]];
    let ws = WorkerSet::new(
        "__nd_foo_ready".to_string(),
        card.mdcsum().to_string(),
        card,
    );
    manager.add_worker_set("foo/ready", "__nd_foo_ready", ws);

    // `GET /v1/models/foo/ready` must resolve to the registered `foo/ready`
    // model (exact match wins), NOT the readiness sub-resource of `foo`. Since
    // `foo/ready` is registered-but-not-ready, its gated retrieve returns 503 —
    // crucially *not* a 200 readiness body for `foo` (the pre-fix behavior).
    let resp = reqwest::Client::new()
        .get(format!("http://localhost:{}/v1/models/foo/ready", port))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "exact match on registered (non-displayable) foo/ready must hit its gated retrieve (503), not foo's readiness (200)"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body.get("namespaces").is_none(),
        "must not be foo's readiness body, got: {body}"
    );

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

/// With nvext disabled, a request asking for response `extra_fields` must not
/// produce any `nvext` field in the response.
#[tokio::test]
async fn test_nvext_disabled_strips_request_and_response() {
    dynamo_runtime::logging::init();

    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_nvext(false)
        .port(port)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run_with_listener(token, listener).await });
    wait_for_service_ready(port).await;

    let card = ModelDeploymentCard::with_name_only("test-model");
    manager
        .add_chat_completions_model("test-model", card.mdcsum(), Arc::new(CounterEngine {}))
        .unwrap();

    let response = reqwest::Client::new()
        .post(format!("http://localhost:{port}/v1/chat/completions"))
        .header("x-dynamo-worker-instance-id", "42")
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 1,
            "nvext": {
                "extra_fields": ["worker_id", "timing", "engine_data"],
                "backend_instance_id": 99
            }
        }))
        .send()
        .await
        .expect("request should succeed");
    assert!(response.status().is_success());

    let body = response.text().await.expect("read body");
    assert!(
        !body.contains("\"nvext\""),
        "nvext gate off: response must not contain an `nvext` field, got: {body}"
    );

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

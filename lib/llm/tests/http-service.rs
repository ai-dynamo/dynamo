// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Error;
use async_stream::stream;
use dynamo_async_openai::config::OpenAIConfig;
use dynamo_llm::http::{
    client::{
        GenericBYOTClient, HttpClientConfig, HttpRequestContext, NvCustomClient, PureOpenAIClient,
    },
    service::{
        Metrics,
        error::HttpError,
        metrics::{Endpoint, RequestType, Status},
        service_v2::HttpService,
    },
};
use dynamo_llm::protocols::{
    Annotated,
    codec::SseLineCodec,
    convert_sse_stream,
    openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_runtime::metrics::prometheus_names::{frontend_service, name_prefix};
use dynamo_runtime::{
    CancellationToken,
    engine::AsyncEngineContext,
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
use ports::get_random_port;

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
                let output = generator.create_choice(i,Some(format!("choice {i}")), None, None);

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
    expected: u64,
) {
    assert_eq!(
        metrics.get_request_counter(model, endpoint, request_type, status),
        expected,
        "model: {}, endpoint: {:?}, request_type: {:?}, status: {:?}",
        model,
        endpoint.as_str(),
        request_type.as_str(),
        status.as_str()
    );
}

fn compute_index(endpoint: &Endpoint, request_type: &RequestType, status: &Status) -> usize {
    let endpoint = match endpoint {
        Endpoint::Completions => 0,
        Endpoint::ChatCompletions => 1,
        Endpoint::Embeddings => todo!(),
        Endpoint::Responses => todo!(),
        Endpoint::Tensor => todo!(),
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
                compare_counter(
                    metrics,
                    model,
                    endpoint,
                    request_type,
                    status,
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
    let port = get_random_port().await;
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
    let task = tokio::spawn(async move { service.run(token.clone()).await });

    let registry = Registry::new();

    let counter = Arc::new(CounterEngine {});
    let result = manager.add_chat_completions_model("foo", counter);
    assert!(result.is_ok());

    let failure = Arc::new(AlwaysFailEngine {});
    let result = manager.add_chat_completions_model("bar", failure.clone());
    assert!(result.is_ok());

    let result = manager.add_completions_model("bar", failure);
    assert!(result.is_ok());

    let metrics = state.metrics_clone();
    metrics.register(&registry).unwrap();

    let mut foo_counters = [0u64; 8];
    let mut bar_counters = [0u64; 8];

    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);

    let client = reqwest::Client::new();

    let message = dynamo_async_openai::types::ChatCompletionRequestMessage::User(
        dynamo_async_openai::types::ChatCompletionRequestUserMessage {
            content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                "hi".to_string(),
            ),
            name: None,
        },
    );

    let mut request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
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

    for bucket in buckets {
        let upper_bound = bucket.get_upper_bound();
        let cumulative_count = bucket.get_cumulative_count();

        println!(
            "Bucket upper bound: {}, count: {}",
            upper_bound, cumulative_count
        );

        // Since our observation is 2.5, it should fall into the bucket with upper bound 4.0
        if upper_bound >= 4.0 {
            assert_eq!(
                cumulative_count, 1,
                "Observation should be counted in the 4.0 bucket"
            );
            found = true;
        } else {
            assert_eq!(
                cumulative_count, 0,
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
    let mut request = dynamo_async_openai::types::CreateCompletionRequestArgs::default()
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

async fn service_with_engines() -> (HttpService, Arc<CounterEngine>, Arc<AlwaysFailEngine>, u16) {
    let port = get_random_port().await;
    let service = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .port(port)
        .build()
        .unwrap();
    let manager = service.model_manager();

    let counter = Arc::new(CounterEngine {});
    let failure = Arc::new(AlwaysFailEngine {});

    manager
        .add_chat_completions_model("foo", counter.clone())
        .unwrap();
    manager
        .add_chat_completions_model("bar", failure.clone())
        .unwrap();
    manager
        .add_completions_model("bar", failure.clone())
        .unwrap();

    (service, counter, failure, port)
}

fn pure_openai_client(port: u16) -> PureOpenAIClient {
    let config = HttpClientConfig {
        openai_config: OpenAIConfig::new().with_api_base(format!("http://localhost:{}/v1", port)),
        verbose: false,
    };
    PureOpenAIClient::new(config)
}

fn nv_custom_client(port: u16) -> NvCustomClient {
    let config = HttpClientConfig {
        openai_config: OpenAIConfig::new().with_api_base(format!("http://localhost:{}/v1", port)),
        verbose: false,
    };
    NvCustomClient::new(config)
}

fn generic_byot_client(port: u16) -> GenericBYOTClient {
    let config = HttpClientConfig {
        openai_config: OpenAIConfig::new().with_api_base(format!("http://localhost:{}/v1", port)),
        verbose: false,
    };
    GenericBYOTClient::new(config)
}

#[tokio::test]
async fn test_pure_openai_client() {
    let (service, _counter, _failure, port) = service_with_engines().await;
    let pure_openai_client = pure_openai_client(port);

    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Test successful streaming request
    let request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content:
                        dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            "Hi".to_string(),
                        ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let result = pure_openai_client.chat_stream(request).await;
    assert!(result.is_ok(), "PureOpenAI client should succeed");

    let (mut stream, _context) = result.unwrap().dissolve();
    let mut count = 0;
    while let Some(response) = stream.next().await {
        count += 1;
        assert!(response.is_ok(), "Response should be ok");
        if count >= 3 {
            break; // Don't consume entire stream
        }
    }
    assert!(count > 0, "Should receive at least one response");

    // Test error case with invalid model
    let request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("bar") // This model will fail
        .messages(vec![
            dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content:
                        dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            "Hi".to_string(),
                        ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let result = pure_openai_client.chat_stream(request).await;
    assert!(
        result.is_ok(),
        "Client should return stream even for failing model"
    );

    let (mut stream, _context) = result.unwrap().dissolve();
    if let Some(response) = stream.next().await {
        assert!(
            response.is_err(),
            "Response should be error for failing model"
        );
    }

    // Test context management
    let ctx = HttpRequestContext::new();
    let request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content:
                        dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            "Hi".to_string(),
                        ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let result = pure_openai_client
        .chat_stream_with_context(request, ctx.clone())
        .await;
    assert!(result.is_ok(), "Context-based request should succeed");

    let (_stream, context) = result.unwrap().dissolve();
    assert_eq!(context.id(), ctx.id(), "Context ID should match");

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_nv_custom_client() {
    let (service, _counter, _failure, port) = service_with_engines().await;
    let nv_custom_client = nv_custom_client(port);

    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Test successful streaming request
    let inner_request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content:
                        dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            "Hi".to_string(),
                        ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let request = NvCreateChatCompletionRequest {
        inner: inner_request,
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
    };

    let result = nv_custom_client.chat_stream(request).await;
    assert!(result.is_ok(), "NvCustom client should succeed");

    let (mut stream, _context) = result.unwrap().dissolve();
    let mut count = 0;
    while let Some(response) = stream.next().await {
        count += 1;
        assert!(response.is_ok(), "Response should be ok");
        if count >= 3 {
            break; // Don't consume entire stream
        }
    }
    assert!(count > 0, "Should receive at least one response");

    // Test error case with invalid model
    let inner_request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("bar") // This model will fail
        .messages(vec![
            dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content:
                        dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            "Hi".to_string(),
                        ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let request = NvCreateChatCompletionRequest {
        inner: inner_request,
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
    };

    let result = nv_custom_client.chat_stream(request).await;
    assert!(
        result.is_ok(),
        "Client should return stream even for failing model"
    );

    let (mut stream, _context) = result.unwrap().dissolve();
    if let Some(response) = stream.next().await {
        assert!(
            response.is_err(),
            "Response should be error for failing model"
        );
    }

    // Test context management
    let ctx = HttpRequestContext::new();
    let inner_request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content:
                        dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                            "Hi".to_string(),
                        ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let request = NvCreateChatCompletionRequest {
        inner: inner_request,
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
    };

    let result = nv_custom_client
        .chat_stream_with_context(request, ctx.clone())
        .await;
    assert!(result.is_ok(), "Context-based request should succeed");

    let (_stream, context) = result.unwrap().dissolve();
    assert_eq!(context.id(), ctx.id(), "Context ID should match");

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_generic_byot_client() {
    let (service, _counter, _failure, port) = service_with_engines().await;
    let generic_byot_client = generic_byot_client(port);

    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Test successful streaming request
    let request = serde_json::json!({
        "model": "foo",
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "stream": true,
        "max_tokens": 50
    });

    let result = generic_byot_client.chat_stream(request).await;
    assert!(result.is_ok(), "GenericBYOT client should succeed");

    let (mut stream, _context) = result.unwrap().dissolve();
    let mut count = 0;
    while let Some(response) = stream.next().await {
        println!("Response: {:?}", response);
        count += 1;
        assert!(response.is_ok(), "Response should be ok");
        if count >= 3 {
            break; // Don't consume entire stream
        }
    }
    assert!(count > 0, "Should receive at least one response");

    // Test error case with invalid model
    let request = serde_json::json!({
        "model": "bar", // This model will fail
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "stream": true,
        "max_tokens": 50
    });

    let result = generic_byot_client.chat_stream(request).await;
    assert!(
        result.is_ok(),
        "Client should return stream even for failing model"
    );

    let (mut stream, _context) = result.unwrap().dissolve();
    if let Some(response) = stream.next().await {
        assert!(
            response.is_err(),
            "Response should be error for failing model"
        );
    }

    // Test context management
    let ctx = HttpRequestContext::new();
    let request = serde_json::json!({
        "model": "foo",
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "stream": true,
        "max_tokens": 50
    });

    let result = generic_byot_client
        .chat_stream_with_context(request, ctx.clone())
        .await;
    assert!(result.is_ok(), "Context-based request should succeed");

    let (_stream, context) = result.unwrap().dissolve();
    assert_eq!(context.id(), ctx.id(), "Context ID should match");

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_client_disconnect_cancellation_unary() {
    let port = get_random_port().await;
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
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Create a long-running engine (10 seconds)
    let long_running_engine = Arc::new(LongRunningEngine::new(10_000));
    manager
        .add_chat_completions_model("slow-model", long_running_engine.clone())
        .unwrap();

    let client = reqwest::Client::new();

    let message = dynamo_async_openai::types::ChatCompletionRequestMessage::User(
        dynamo_async_openai::types::ChatCompletionRequestUserMessage {
            content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                "This will take a long time".to_string(),
            ),
            name: None,
        },
    );

    let request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
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

    let port = get_random_port().await;
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
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Create a long-running engine (10 seconds)
    let long_running_engine = Arc::new(LongRunningEngine::new(10_000));
    manager
        .add_chat_completions_model("slow-stream-model", long_running_engine.clone())
        .unwrap();

    let client = reqwest::Client::new();

    let message = dynamo_async_openai::types::ChatCompletionRequestMessage::User(
        dynamo_async_openai::types::ChatCompletionRequestUserMessage {
            content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                "This will stream for a long time".to_string(),
            ),
            name: None,
        },
    );

    let request = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
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

    let port = get_random_port().await;
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
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(port).await;

    // Add a counter engine for this test
    let counter_engine = Arc::new(CounterEngine {});
    manager
        .add_chat_completions_model("test-model", counter_engine)
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

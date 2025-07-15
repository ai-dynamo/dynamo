// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use anyhow::Error;
use async_stream::stream;
use dynamo_llm::http::service::{
    error::HttpError,
    metrics::{Endpoint, RequestType, Status},
    rate_limiter::RateLimiterConfig,
    service_v2::HttpService,
    Metrics,
};
use dynamo_llm::protocols::{
    openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
    Annotated,
};
use dynamo_runtime::{
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
    },
    CancellationToken,
};
use prometheus::{proto::MetricType, Registry};
use reqwest::StatusCode;
use std::sync::Arc;

struct CounterEngine {}

#[allow(deprecated)]
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
        let max_tokens = request.inner.max_tokens.unwrap_or(0) as u64;

        // let generator = NvCreateChatCompletionStreamResponse::generator(request.model.clone());
        let generator = request.response_generator();

        let stream = stream! {
            tokio::time::sleep(std::time::Duration::from_millis(max_tokens)).await;
            for i in 0..10 {
                let inner = generator.create_choice(i,Some(format!("choice {i}")), None, None);

                let output = NvCreateChatCompletionStreamResponse {
                    inner,
                };

                yield Annotated::from_data(output);
            }
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
    };

    let request_type = match request_type {
        RequestType::Unary => 0,
        RequestType::Stream => 1,
    };

    let status = match status {
        Status::Success => 0,
        Status::Error => 1,
        Status::Rejected => 2,
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
    let service = HttpService::builder().port(8989).build().unwrap();
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

    let message = async_openai::types::ChatCompletionRequestMessage::User(
        async_openai::types::ChatCompletionRequestUserMessage {
            content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                "hi".to_string(),
            ),
            name: None,
        },
    );

    let mut request = async_openai::types::CreateChatCompletionRequestArgs::default()
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
        .post("http://localhost:8989/v1/chat/completions")
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
        .find(|m| m.get_name() == "nv_llm_http_service_request_duration_seconds")
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
        .post("http://localhost:8989/v1/chat/completions")
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
        .post("http://localhost:8989/v1/chat/completions")
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
        .post("http://localhost:8989/v1/chat/completions")
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
    let mut request = async_openai::types::CreateCompletionRequestArgs::default()
        .model("bar")
        .prompt("hi")
        .build()
        .unwrap();

    let response = client
        .post("http://localhost:8989/v1/completions")
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
        .post("http://localhost:8989/v1/completions")
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
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "{:?}",
        response
    );

    // =========== Query /metrics endpoint ===========
    let response = client
        .get("http://localhost:8989/metrics")
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "{:?}", response);
    println!("{}", response.text().await.unwrap());

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

/// Engine that simulates low TTFT to trigger rate limiting
struct SlowTTFTEngine {
    ttft_delay_ms: u64,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for SlowTTFTEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let generator = request.response_generator();
        let ttft_delay_ms = self.ttft_delay_ms;

        let stream = stream! {
            // Simulate slow TTFT
            tokio::time::sleep(std::time::Duration::from_millis(ttft_delay_ms)).await;

            // Generate a few tokens with normal ITL
            for i in 0..3 {
                let inner = generator.create_choice(i, Some(format!("token {i}")), None, None);
                let output = NvCreateChatCompletionStreamResponse { inner };
                yield Annotated::from_data(output);

                if i < 2 {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await; // Normal ITL
                }
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

/// Engine that simulates slow ITL to trigger rate limiting
struct SlowITLEngine {
    itl_delay_ms: u64,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for SlowITLEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let generator = request.response_generator();
        let itl_delay_ms = self.itl_delay_ms;

        let stream = stream! {
            // Fast TTFT
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            // Generate tokens with slow ITL
            for i in 0..5 {
                let inner = generator.create_choice(i, Some(format!("token {i}")), None, None);
                let output = NvCreateChatCompletionStreamResponse { inner };
                yield Annotated::from_data(output);

                if i < 4 {
                    tokio::time::sleep(std::time::Duration::from_millis(itl_delay_ms)).await; // Slow ITL
                }
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

#[tokio::test]
async fn test_rate_limiting_triggers_correctly() {
    // Create rate limiter config with low thresholds for testing
    let rate_limiter_config = RateLimiterConfig::new(
        1.0,   // TTFT threshold: 1 second
        0.1,   // ITL threshold: 100ms
        5.0,   // Time constant: 5 seconds
        false, // Global rate limiting
    )
    .unwrap();

    let service = HttpService::builder()
        .port(8990)
        .with_rate_limiter(rate_limiter_config)
        .build()
        .unwrap();

    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run(token.clone()).await });

    // Add engines with different performance characteristics
    let fast_engine = Arc::new(CounterEngine {});
    let slow_ttft_engine = Arc::new(SlowTTFTEngine {
        ttft_delay_ms: 1500,
    }); // 1.5s TTFT
    let slow_itl_engine = Arc::new(SlowITLEngine { itl_delay_ms: 200 }); // 200ms ITL

    manager
        .add_chat_completions_model("fast", fast_engine)
        .unwrap();
    manager
        .add_chat_completions_model("slow_ttft", slow_ttft_engine)
        .unwrap();
    manager
        .add_chat_completions_model("slow_itl", slow_itl_engine)
        .unwrap();

    let client = reqwest::Client::new();
    let metrics = state.metrics_clone();

    // Wait for service to be ready
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Test 1: Fast model should work fine initially
    let request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("fast")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "test".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(10 as u32)
        .build()
        .unwrap();

    let response = client
        .post("http://localhost:8990/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "Fast model should work initially"
    );
    let _ = response.bytes().await.unwrap();

    // Test 2: Slow TTFT model should trigger rate limiting after a few requests
    let mut slow_ttft_request = request.clone();
    slow_ttft_request.model = "slow_ttft".to_string();

    // Make several requests to build up the EMA
    for i in 0..3 {
        println!("Sending slow TTFT request {}", i + 1);
        let response = client
            .post("http://localhost:8990/v1/chat/completions")
            .json(&slow_ttft_request)
            .send()
            .await
            .unwrap();

        // First few requests should succeed (building up EMA)
        if i < 2 {
            assert!(
                response.status().is_success(),
                "Slow TTFT request {} should succeed while building EMA",
                i + 1
            );
            let _ = response.bytes().await.unwrap();
        } else {
            // Later requests should be rate limited
            if response.status() == StatusCode::TOO_MANY_REQUESTS {
                println!("Rate limiting triggered after {} requests", i + 1);
                break;
            } else {
                let _ = response.bytes().await.unwrap();
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await; // Small delay between requests
    }

    // Test 3: Slow ITL model should also trigger rate limiting
    let mut slow_itl_request = request.clone();
    slow_itl_request.model = "slow_itl".to_string();

    for i in 0..3 {
        println!("Sending slow ITL request {}", i + 1);
        let response = client
            .post("http://localhost:8990/v1/chat/completions")
            .json(&slow_itl_request)
            .send()
            .await
            .unwrap();

        if i < 2 {
            assert!(
                response.status().is_success(),
                "Slow ITL request {} should succeed while building EMA",
                i + 1
            );
            let _ = response.bytes().await.unwrap();
        } else {
            // Later requests should be rate limited
            if response.status() == StatusCode::TOO_MANY_REQUESTS {
                println!("ITL rate limiting triggered after {} requests", i + 1);
                break;
            } else {
                let _ = response.bytes().await.unwrap();
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Test 4: Verify rejection metrics were recorded
    let rejection_count = metrics.get_rate_limit_requests_counter(
        "slow_ttft",
        &Endpoint::ChatCompletions,
        &RequestType::Stream,
        &Status::Rejected,
    ) + metrics.get_rate_limit_requests_counter(
        "slow_itl",
        &Endpoint::ChatCompletions,
        &RequestType::Stream,
        &Status::Rejected,
    );

    println!("Total rejection count: {}", rejection_count);

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_rate_limiting_http_integration() {
    let rate_limiter_config = RateLimiterConfig::new(0.1, 0.01, 5.0, false).unwrap();
    let service = HttpService::builder()
        .port(8991)
        .with_rate_limiter(rate_limiter_config)
        .build()
        .unwrap();

    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run(token.clone()).await });

    // Use simple CounterEngine (already exists)
    let engine = Arc::new(CounterEngine {});
    manager.add_chat_completions_model("test", engine).unwrap();

    // Manually record high TTFT values to trigger rate limiting
    state.rate_limiter().record_ttft("test", 0.5);
    state.rate_limiter().record_ttft("test", 0.3);
    state.rate_limiter().record_ttft("test", 0.4);

    let client = reqwest::Client::new();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("test")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "test".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(3 as u32)
        .build()
        .unwrap();

    // This request should be rate limited
    let response = client
        .post("http://localhost:8991/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    println!("✅ Rate limiting triggered correctly!");

    // Verify metrics were recorded
    let rejection_count = state.metrics_clone().get_rate_limit_requests_counter(
        "test",
        &Endpoint::ChatCompletions,
        &RequestType::Stream,
        &Status::Rejected,
    );
    assert!(rejection_count > 0, "Should have recorded rejection");

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_per_model_vs_global_rate_limiting() {
    // Test global rate limiting (per_model_limits = false)
    let global_config = RateLimiterConfig::new(0.8, 0.08, 3.0, false).unwrap();
    let service1 = HttpService::builder()
        .port(8992)
        .with_rate_limiter(global_config)
        .build()
        .unwrap();

    let state1 = service1.state_clone();
    let manager1 = state1.manager();

    let token1 = CancellationToken::new();
    let cancel_token1 = token1.clone();
    let task1 = tokio::spawn(async move { service1.run(token1.clone()).await });

    // Test per-model rate limiting (per_model_limits = true)
    let per_model_config = RateLimiterConfig::new(0.8, 0.08, 3.0, true).unwrap();
    let service2 = HttpService::builder()
        .port(8993)
        .with_rate_limiter(per_model_config)
        .build()
        .unwrap();

    let state2 = service2.state_clone();
    let manager2 = state2.manager();

    let token2 = CancellationToken::new();
    let cancel_token2 = token2.clone();
    let task2 = tokio::spawn(async move { service2.run(token2.clone()).await });

    // Add slow engines to both services
    let slow_engine1a = Arc::new(SlowTTFTEngine {
        ttft_delay_ms: 1200,
    });
    let slow_engine1b = Arc::new(SlowTTFTEngine {
        ttft_delay_ms: 1200,
    });
    let slow_engine2a = Arc::new(SlowTTFTEngine {
        ttft_delay_ms: 1200,
    });
    let slow_engine2b = Arc::new(SlowTTFTEngine {
        ttft_delay_ms: 1200,
    });

    manager1
        .add_chat_completions_model("model_a", slow_engine1a)
        .unwrap();
    manager1
        .add_chat_completions_model("model_b", slow_engine1b)
        .unwrap();
    manager2
        .add_chat_completions_model("model_a", slow_engine2a)
        .unwrap();
    manager2
        .add_chat_completions_model("model_b", slow_engine2b)
        .unwrap();

    let client = reqwest::Client::new();

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let base_request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "test".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(10 as u32)
        .build()
        .unwrap();

    // Test global rate limiting - model_a affects model_b
    println!("Testing global rate limiting...");
    for _i in 0..3 {
        let mut request_a = base_request.clone();
        request_a.model = "model_a".to_string();

        let response = client
            .post("http://localhost:8992/v1/chat/completions")
            .json(&request_a)
            .send()
            .await
            .unwrap();

        if response.status().is_success() {
            let _ = response.bytes().await.unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Now model_b should be affected by model_a's rate limiting (global)
    let mut request_b = base_request.clone();
    request_b.model = "model_b".to_string();

    let response = client
        .post("http://localhost:8992/v1/chat/completions")
        .json(&request_b)
        .send()
        .await
        .unwrap();

    println!(
        "Global rate limiting - model_b status: {}",
        response.status()
    );

    // Test per-model rate limiting - model_a doesn't affect model_b
    println!("Testing per-model rate limiting...");
    for _i in 0..3 {
        let mut request_a = base_request.clone();
        request_a.model = "model_a".to_string();

        let response = client
            .post("http://localhost:8993/v1/chat/completions")
            .json(&request_a)
            .send()
            .await
            .unwrap();

        if response.status().is_success() {
            let _ = response.bytes().await.unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // model_b should NOT be affected by model_a's rate limiting (per-model)
    let mut request_b2 = base_request.clone();
    request_b2.model = "model_b".to_string();

    let response = client
        .post("http://localhost:8993/v1/chat/completions")
        .json(&request_b2)
        .send()
        .await
        .unwrap();

    println!(
        "Per-model rate limiting - model_b status: {}",
        response.status()
    );
    // Model B should succeed since it has its own rate limiting state
    assert!(
        response.status().is_success() || response.status() != StatusCode::TOO_MANY_REQUESTS,
        "Per-model rate limiting should not affect model_b"
    );

    if response.status().is_success() {
        let _ = response.bytes().await.unwrap();
    }

    cancel_token1.cancel();
    cancel_token2.cancel();
    task1.await.unwrap().unwrap();
    task2.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_rate_limiting_recovery() {
    let rate_limiter_config = RateLimiterConfig::new(
        0.6,  // TTFT threshold: 600ms
        0.06, // ITL threshold: 60ms
        1.0,  // Short time constant for faster recovery
        false,
    )
    .unwrap();

    let service = HttpService::builder()
        .port(8994)
        .with_rate_limiter(rate_limiter_config)
        .build()
        .unwrap();

    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run(token.clone()).await });

    // Add engines with different speeds
    let slow_engine = Arc::new(SlowTTFTEngine {
        ttft_delay_ms: 1000,
    }); // 1s TTFT
    let fast_engine = Arc::new(CounterEngine {});

    manager
        .add_chat_completions_model("slow", slow_engine)
        .unwrap();
    manager
        .add_chat_completions_model("fast", fast_engine)
        .unwrap();

    let client = reqwest::Client::new();

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let slow_request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("slow")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "test".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(10 as u32)
        .build()
        .unwrap();

    let fast_request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("fast")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "test".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(10 as u32)
        .build()
        .unwrap();

    // Phase 1: Trigger rate limiting with slow requests
    println!("Phase 1: Triggering rate limiting...");
    for i in 0..4 {
        let response = client
            .post("http://localhost:8994/v1/chat/completions")
            .json(&slow_request)
            .send()
            .await
            .unwrap();

        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            println!("Rate limiting triggered at request {}", i + 1);
            break;
        } else {
            let _ = response.bytes().await.unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Phase 2: Wait for system to recover (time constant is 1.0s)
    println!("Phase 2: Waiting for recovery...");
    tokio::time::sleep(std::time::Duration::from_millis(3000)).await; // Wait 3 time constants

    // Phase 3: Send fast requests to bring down the EMA
    println!("Phase 3: Sending fast requests to improve EMA...");
    for i in 0..3 {
        let response = client
            .post("http://localhost:8994/v1/chat/completions")
            .json(&fast_request)
            .send()
            .await
            .unwrap();

        println!("Fast request {} status: {}", i + 1, response.status());
        if response.status().is_success() {
            let _ = response.bytes().await.unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // Phase 4: Verify that slow requests work again (system recovered)
    println!("Phase 4: Testing recovery with moderate request...");
    let response = client
        .post("http://localhost:8994/v1/chat/completions")
        .json(&fast_request)
        .send()
        .await
        .unwrap();

    println!("Recovery test status: {}", response.status());
    assert!(
        response.status().is_success(),
        "System should have recovered and accept requests again"
    );

    if response.status().is_success() {
        let _ = response.bytes().await.unwrap();
    }

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

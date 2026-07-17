// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the frontend admission gates from DEP #9755:
//! `--rejection-frontend-request-concurrency-limit`,
//! `--rejection-frontend-runtime-task-limit`, and
//! `--rejection-frontend-request-plane-connection-limit`.

use anyhow::Error;
use async_stream::stream;
use dynamo_llm::discovery::UNKNOWN_METRIC_MODEL;
use dynamo_llm::frontend_config::AdmissionGateConfig;
use dynamo_llm::http::service::metrics::Endpoint;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::{
    Annotated,
    openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_runtime::metrics::prometheus_names::frontend_service::admission_gate;
use dynamo_runtime::metrics::request_plane::REQUEST_PLANE_INFLIGHT;
use dynamo_runtime::{
    CancellationToken,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
    },
};
use reqwest::StatusCode;
use std::sync::Arc;
use tokio::sync::Barrier;

#[path = "common/ports.rs"]
mod ports;
use ports::bind_random_port;

/// Sleeps for the request's `max_tokens` milliseconds before yielding one
/// choice, so a test can hold a request in flight for a controlled duration.
struct DelayEngine {}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for DelayEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
        #[allow(deprecated)]
        let delay_ms = request.inner.max_tokens.unwrap_or(0) as u64;
        let mut generator = request.response_generator(ctx.id().to_string());

        let stream = stream! {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            let output = generator.create_choice(0, Some("done".to_string()), None, None);
            yield Annotated::from_data(output);
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

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

struct GateTestService {
    port: u16,
    state: Arc<dynamo_llm::http::service::service_v2::State>,
    cancel_token: CancellationToken,
    task: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl Drop for GateTestService {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        self.task.abort();
    }
}

/// Simulate one live request-plane stream and guarantee cleanup if a test
/// assertion fails before the pressure is released explicitly.
struct RequestPlanePressureGuard;

impl RequestPlanePressureGuard {
    fn new() -> Self {
        REQUEST_PLANE_INFLIGHT.inc();
        Self
    }
}

impl Drop for RequestPlanePressureGuard {
    fn drop(&mut self) {
        REQUEST_PLANE_INFLIGHT.dec();
    }
}

/// Start an HTTP service with the given gate config and the chat models
/// `slow` and `fast`, both backed by [`DelayEngine`].
async fn start_gate_service(config: AdmissionGateConfig) -> GateTestService {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .enable_chat_endpoints(true)
        .enable_anthropic_endpoints(true)
        .admission_gate_config(config)
        .build()
        .unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    for model in ["slow", "fast"] {
        let card = ModelDeploymentCard::with_name_only(model);
        manager
            .add_chat_completions_model(model, card.mdcsum(), Arc::new(DelayEngine {}))
            .unwrap();
    }

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task =
        tokio::spawn(async move { service.run_with_listener(token.clone(), listener).await });
    wait_for_service_ready(port).await;

    GateTestService {
        port,
        state,
        cancel_token,
        task,
    }
}

fn chat_request(model: &str, delay_ms: u32) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": delay_ms,
    })
}

async fn post_chat(port: u16, body: &serde_json::Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(body)
        .send()
        .await
        .unwrap()
}

async fn post_count_tokens(port: u16) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("http://localhost:{port}/v1/messages/count_tokens"))
        .json(&serde_json::json!({
            "model": "fast",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap()
}

async fn wait_for_inflight(
    metrics: &dynamo_llm::http::service::Metrics,
    model: &str,
    expected: i64,
) {
    let start = tokio::time::Instant::now();
    while metrics.get_inflight_count(model) != expected {
        assert!(
            start.elapsed() < tokio::time::Duration::from_secs(2),
            "inflight count for {model} did not reach {expected}"
        );
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

async fn hold_request(
    port: u16,
    metrics: &dynamo_llm::http::service::Metrics,
    model: &str,
) -> tokio::task::JoinHandle<reqwest::Response> {
    let request_model = model.to_string();
    let holder =
        tokio::spawn(async move { post_chat(port, &chat_request(&request_model, 2000)).await });
    wait_for_inflight(metrics, model, 1).await;
    holder
}

async fn finish_held_request(
    holder: tokio::task::JoinHandle<reqwest::Response>,
    metrics: &dynamo_llm::http::service::Metrics,
    model: &str,
) {
    assert_eq!(holder.await.unwrap().status(), StatusCode::OK);
    wait_for_inflight(metrics, model, 0).await;
}

#[tokio::test]
async fn test_concurrency_gate_rejects_over_limit_per_model() {
    let svc = start_gate_service(
        AdmissionGateConfig::new(Some(1), None, None).expect("valid admission gate config"),
    )
    .await;
    let metrics = svc.state.metrics_clone();
    let holder = hold_request(svc.port, &metrics, "slow").await;

    // Second request for the same model exceeds limit=1 -> 503 with the knob name.
    let rejected = post_chat(svc.port, &chat_request("slow", 1)).await;
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = rejected.text().await.unwrap();
    assert!(
        body.contains("rejection-frontend-request-concurrency-limit"),
        "rejection body should name the knob: {body}"
    );
    assert!(
        body.contains("frontend-global"),
        "rejection body should identify the global limit source: {body}"
    );
    assert_eq!(
        metrics.get_admission_rejection_count(admission_gate::REQUEST_CONCURRENCY, "slow"),
        1
    );

    // Per-model isolation: a different served model is still admitted.
    let ok = post_chat(svc.port, &chat_request("fast", 1)).await;
    assert_eq!(ok.status(), StatusCode::OK);

    // Unregistered models are exempt from the gate and keep their 404.
    let unknown = post_chat(svc.port, &chat_request("no-such-model", 1)).await;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);

    // A literal request for the bounded unknown-model metrics sentinel must
    // also retain the normal 404, even when its shared inflight count exceeds
    // the configured gate limit.
    let sentinel_holder = metrics.clone().create_inflight_guard(
        UNKNOWN_METRIC_MODEL,
        Endpoint::ChatCompletions,
        false,
        "sentinel-collision-test",
    );
    let sentinel = post_chat(svc.port, &chat_request(UNKNOWN_METRIC_MODEL, 1)).await;
    assert_eq!(sentinel.status(), StatusCode::NOT_FOUND);
    drop(sentinel_holder);

    finish_held_request(holder, &metrics, "slow").await;
    let after = post_chat(svc.port, &chat_request("slow", 1)).await;
    assert_eq!(after.status(), StatusCode::OK);
}

/// Register a chat model whose MDC carries a per-model concurrency override.
fn add_model_with_override(manager: &dynamo_llm::discovery::ModelManager, model: &str, limit: u64) {
    let mut card = ModelDeploymentCard::with_name_only(model);
    card.rejection_frontend_request_concurrency_limit = Some(limit);
    let mdcsum = card.mdcsum().to_string();
    manager
        .add_chat_completions_model_with_card(model, &mdcsum, card, Arc::new(DelayEngine {}))
        .unwrap();
}

/// Hold one slow request on `model` in flight and assert the next request to
/// the same model is rejected (or admitted, per `expect_reject`).
async fn assert_second_request(
    port: u16,
    metrics: &dynamo_llm::http::service::Metrics,
    model: &str,
    expect_reject: bool,
) -> String {
    let holder = hold_request(port, metrics, model).await;
    let second = post_chat(port, &chat_request(model, 1)).await;
    let status = second.status();
    let body = second.text().await.unwrap();
    let expected = if expect_reject {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    assert_eq!(status, expected, "unexpected status for {model}: {body}");
    finish_held_request(holder, metrics, model).await;
    body
}

#[tokio::test]
async fn test_concurrency_gate_mdc_override_activates_without_global_default() {
    // No global limit: only the model with an MDC override is gated. The
    // uncapped model also verifies that gates are disabled by default.
    let svc = start_gate_service(
        AdmissionGateConfig::new(None, None, None).expect("valid admission gate config"),
    )
    .await;
    let manager = svc.state.manager();
    add_model_with_override(manager, "capped", 1);
    let metrics = svc.state.metrics_clone();

    let rejection = assert_second_request(svc.port, &metrics, "capped", true).await;
    assert!(
        rejection.contains("per-model MDC override"),
        "rejection body should identify the MDC limit source: {rejection}"
    );
    let _ = assert_second_request(svc.port, &metrics, "fast", false).await;
    assert_eq!(
        metrics.get_admission_rejection_count(admission_gate::REQUEST_CONCURRENCY, "capped"),
        1
    );
}

#[tokio::test]
async fn test_concurrency_gate_mdc_override_wins_over_global_default() {
    // Global limit 1, but the MDC override raises "roomy" to 3: the override
    // takes precedence for that model while other models keep the global.
    let svc = start_gate_service(
        AdmissionGateConfig::new(Some(1), None, None).expect("valid admission gate config"),
    )
    .await;
    let manager = svc.state.manager();
    add_model_with_override(manager, "roomy", 3);
    let metrics = svc.state.metrics_clone();

    let _ = assert_second_request(svc.port, &metrics, "roomy", false).await;
    let rejection = assert_second_request(svc.port, &metrics, "fast", true).await;
    assert!(
        rejection.contains("frontend-global"),
        "rejection body should identify the global limit source: {rejection}"
    );
}

#[tokio::test]
async fn test_runtime_task_gate_rejects_all_inference_but_not_system_routes() {
    // A live tokio runtime always has far more than one alive task, so
    // limit=1 rejects every inference request.
    let svc = start_gate_service(
        AdmissionGateConfig::new(None, Some(1), None).expect("valid admission gate config"),
    )
    .await;
    let rejected = post_chat(svc.port, &chat_request("fast", 1)).await;
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = rejected.text().await.unwrap();
    assert!(
        body.contains("rejection-frontend-runtime-task-limit"),
        "rejection body should name the knob: {body}"
    );
    assert_eq!(
        svc.state
            .metrics_clone()
            .get_admission_rejection_count(admission_gate::RUNTIME_TASK, ""),
        1
    );

    // count_tokens does not open a request-plane stream, but it is still an
    // inference route and remains subject to the runtime-task pressure gate.
    let count_tokens = post_count_tokens(svc.port).await;
    assert_eq!(count_tokens.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        svc.state
            .metrics_clone()
            .get_admission_rejection_count(admission_gate::RUNTIME_TASK, ""),
        2
    );

    // System routes bypass the inference admission middleware.
    let health = reqwest::get(format!("http://localhost:{}/health", svc.port))
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
}

#[tokio::test]
#[serial_test::serial]
async fn test_request_plane_gate_rejects_at_observed_threshold() {
    // REQUEST_PLANE_INFLIGHT is a process-global gauge; no request-plane
    // traffic exists in this test binary, so simulate pressure directly.
    let svc = start_gate_service(
        AdmissionGateConfig::new(None, None, Some(1)).expect("valid admission gate config"),
    )
    .await;
    let pressure = RequestPlanePressureGuard::new();

    let rejected = post_chat(svc.port, &chat_request("fast", 1)).await;
    let status = rejected.status();
    let body = rejected.text().await.unwrap();

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        body.contains("rejection-frontend-request-plane-connection-limit"),
        "rejection body should name the knob: {body}"
    );
    assert_eq!(
        svc.state
            .metrics_clone()
            .get_admission_rejection_count(admission_gate::REQUEST_PLANE_CONNECTION, ""),
        1
    );

    // With the pressure gone the same request is admitted.
    drop(pressure);
    let ok = post_chat(svc.port, &chat_request("fast", 1)).await;
    assert_eq!(ok.status(), StatusCode::OK);
}

#[tokio::test]
#[serial_test::serial]
async fn test_request_plane_gate_skips_local_count_tokens() {
    let svc = start_gate_service(
        AdmissionGateConfig::new(None, None, Some(1)).expect("valid admission gate config"),
    )
    .await;
    let _pressure = RequestPlanePressureGuard::new();

    // This route estimates tokens locally and never opens a request-plane
    // stream, so request-plane pressure must not reject it.
    let count_tokens = post_count_tokens(svc.port).await;
    assert_eq!(count_tokens.status(), StatusCode::OK);
    assert_eq!(
        svc.state
            .metrics_clone()
            .get_admission_rejection_count(admission_gate::REQUEST_PLANE_CONNECTION, ""),
        0
    );

    // A route that dispatches to a worker is still rejected by the same gate.
    let chat = post_chat(svc.port, &chat_request("fast", 1)).await;
    assert_eq!(chat.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        svc.state
            .metrics_clone()
            .get_admission_rejection_count(admission_gate::REQUEST_PLANE_CONNECTION, ""),
        1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn test_request_plane_counter_returns_to_zero_after_concurrent_completion() {
    const REQUEST_COUNT: usize = 100;

    assert_eq!(REQUEST_PLANE_INFLIGHT.get(), 0.0);
    let admitted = Arc::new(Barrier::new(REQUEST_COUNT + 1));
    let complete = Arc::new(Barrier::new(REQUEST_COUNT + 1));
    let mut tasks = Vec::with_capacity(REQUEST_COUNT);

    for _ in 0..REQUEST_COUNT {
        let admitted = admitted.clone();
        let complete = complete.clone();
        tasks.push(tokio::spawn(async move {
            let _inflight = RequestPlanePressureGuard::new();
            admitted.wait().await;
            complete.wait().await;
        }));
    }

    // All increments have completed before any guard is allowed to drop.
    admitted.wait().await;
    let peak = REQUEST_PLANE_INFLIGHT.get();

    // Release all completions together and wait for every atomic decrement.
    complete.wait().await;
    for task in tasks {
        task.await.unwrap();
    }
    assert_eq!(peak, REQUEST_COUNT as f64);
    assert_eq!(REQUEST_PLANE_INFLIGHT.get(), 0.0);
}

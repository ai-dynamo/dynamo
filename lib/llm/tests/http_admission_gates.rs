// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the frontend admission gates (DIS-2186):
//! `--rejection-frontend-request-concurrency-limit`,
//! `--rejection-frontend-runtime-task-limit`, and
//! `--rejection-frontend-request-plane-connection-limit`.

use anyhow::Error;
use async_stream::stream;
use dynamo_llm::frontend_config::AdmissionGateConfig;
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

/// Start an HTTP service with the given gate config and the chat models
/// `slow` and `fast`, both backed by [`DelayEngine`].
async fn start_gate_service(config: AdmissionGateConfig) -> GateTestService {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .enable_chat_endpoints(true)
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

#[tokio::test]
async fn test_concurrency_gate_rejects_over_limit_per_model() {
    let svc = start_gate_service(AdmissionGateConfig::new(Some(1), None, None)).await;
    let port = svc.port;
    let metrics = svc.state.metrics_clone();

    // Hold one slow request in flight (non-streaming, engine sleeps 2s).
    let holder = tokio::spawn(async move { post_chat(port, &chat_request("slow", 2000)).await });
    // Wait until the held request is actually counted in flight.
    let start = tokio::time::Instant::now();
    while metrics.get_inflight_count("slow") < 1 {
        assert!(
            start.elapsed() < tokio::time::Duration::from_secs(2),
            "held request never became inflight"
        );
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Second request for the same model exceeds limit=1 -> 503 with the knob name.
    let rejected = post_chat(port, &chat_request("slow", 1)).await;
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = rejected.text().await.unwrap();
    assert!(
        body.contains("rejection-frontend-request-concurrency-limit"),
        "rejection body should name the knob: {body}"
    );
    assert_eq!(
        metrics.get_admission_rejection_count(admission_gate::REQUEST_CONCURRENCY, "slow"),
        1
    );

    // Per-model isolation: a different served model is still admitted.
    let ok = post_chat(port, &chat_request("fast", 1)).await;
    assert_eq!(ok.status(), StatusCode::OK);

    // Unregistered models are exempt from the gate and keep their 404.
    let unknown = post_chat(port, &chat_request("no-such-model", 1)).await;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);

    // Once the held request finishes, the model admits again. The inflight
    // guard drops when the response body is done, which can trail the client
    // seeing the status line, so wait for the gauge to drain.
    let held = holder.await.unwrap();
    assert_eq!(held.status(), StatusCode::OK);
    let start = tokio::time::Instant::now();
    while metrics.get_inflight_count("slow") > 0 {
        assert!(
            start.elapsed() < tokio::time::Duration::from_secs(2),
            "held request never released its inflight permit"
        );
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    let after = post_chat(port, &chat_request("slow", 1)).await;
    assert_eq!(after.status(), StatusCode::OK);

    svc.cancel_token.cancel();
    svc.task.abort();
}

#[tokio::test]
async fn test_concurrency_gate_disabled_by_default_admits_parallel_requests() {
    let svc = start_gate_service(AdmissionGateConfig::new(None, None, None)).await;
    let port = svc.port;

    let body = chat_request("slow", 500);
    let (a, b) = tokio::join!(post_chat(port, &body), post_chat(port, &body));
    assert_eq!(a.status(), StatusCode::OK);
    assert_eq!(b.status(), StatusCode::OK);

    svc.cancel_token.cancel();
    svc.task.abort();
}

#[tokio::test]
async fn test_runtime_task_gate_rejects_all_inference_but_not_system_routes() {
    // A live tokio runtime always has far more than one alive task, so
    // limit=1 rejects every inference request.
    let svc = start_gate_service(AdmissionGateConfig::new(None, Some(1), None)).await;
    let port = svc.port;

    let rejected = post_chat(port, &chat_request("fast", 1)).await;
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

    // System routes bypass the inference admission middleware.
    let health = reqwest::get(format!("http://localhost:{}/health", port))
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    svc.cancel_token.cancel();
    svc.task.abort();
}

#[tokio::test]
async fn test_runtime_task_gate_admits_under_generous_limit() {
    let svc = start_gate_service(AdmissionGateConfig::new(None, Some(1_000_000), None)).await;
    let port = svc.port;

    let ok = post_chat(port, &chat_request("fast", 1)).await;
    assert_eq!(ok.status(), StatusCode::OK);

    svc.cancel_token.cancel();
    svc.task.abort();
}

#[tokio::test]
#[serial_test::serial]
async fn test_request_plane_gate_tracks_global_inflight_gauge() {
    // REQUEST_PLANE_INFLIGHT is a process-global gauge; no request-plane
    // traffic exists in this test binary, so simulate pressure directly.
    let svc = start_gate_service(AdmissionGateConfig::new(None, None, Some(1))).await;
    let port = svc.port;

    REQUEST_PLANE_INFLIGHT.inc();
    REQUEST_PLANE_INFLIGHT.inc();

    let rejected = post_chat(port, &chat_request("fast", 1)).await;
    let status = rejected.status();
    let body = rejected.text().await.unwrap();

    REQUEST_PLANE_INFLIGHT.dec();
    REQUEST_PLANE_INFLIGHT.dec();

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
    let ok = post_chat(port, &chat_request("fast", 1)).await;
    assert_eq!(ok.status(), StatusCode::OK);

    svc.cancel_token.cancel();
    svc.task.abort();
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use axum::{Json, Router, routing::post};
use dynamo_kv_router::services::selection::{CatalogReconciler, WorkerRequest};
use prometheus::{Encoder, Registry, TextEncoder};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::pod_discovery::RawWorker;
use crate::proto::envoy::config::core::v3::{HeaderMap, HeaderValue};
use crate::proto::envoy::service::ext_proc::v3::{
    self as ext_proc, ProcessingRequest, external_processor_client::ExternalProcessorClient,
    processing_request::Request, processing_response::Response,
};

async fn exchange(
    client: &mut ExternalProcessorClient<tonic::transport::Channel>,
    body: &[u8],
    response: &[u8],
) -> Vec<ext_proc::ProcessingResponse> {
    let requests = vec![
        Request::RequestHeaders(ext_proc::HttpHeaders {
            headers: Some(HeaderMap {
                headers: vec![HeaderValue {
                    key: "content-type".into(),
                    raw_value: b"application/json".to_vec(),
                    ..Default::default()
                }],
            }),
            end_of_stream: false,
        }),
        Request::RequestBody(ext_proc::HttpBody {
            body: body.to_vec(),
            end_of_stream: true,
        }),
        Request::ResponseBody(ext_proc::HttpBody {
            body: response.to_vec(),
            end_of_stream: true,
        }),
    ];
    let mut stream = client
        .process(tokio_stream::iter(requests.into_iter().map(|request| {
            ProcessingRequest {
                request: Some(request),
                ..Default::default()
            }
        })))
        .await
        .unwrap()
        .into_inner();
    let mut responses = Vec::new();
    while let Some(response) = stream.message().await.unwrap() {
        responses.push(response);
    }
    responses
}

#[tokio::test]
async fn standalone_router_metrics_follow_the_wire_observation_paths() {
    // Exercise the real renderer client, selector, picker and ext_proc server.
    // Only Kubernetes discovery and model inference are replaced by fixtures.
    let renderer = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let renderer_addr = renderer.local_addr().unwrap();
    let cancel = CancellationToken::new();
    let renderer_cancel = cancel.clone();
    let renderer_task = tokio::spawn(async move {
        let app = Router::new().route(
            "/v1/chat/completions/render",
            post(|| async { Json(serde_json::json!({"token_ids": [1, 2, 3, 4]})) }),
        );
        axum::serve(renderer, app)
            .with_graceful_shutdown(renderer_cancel.cancelled_owned())
            .await
            .unwrap();
    });
    let kv_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let kv_addr = kv_listener.local_addr().unwrap();
    let cfg = EppStandaloneConfig {
        selector_threads: 1,
        peer_replication: None,
        inference_pool_name: "test-pool".into(),
        namespace: "test-ns".into(),
        model_name: "served-model".into(),
        tokenizer_service_url: format!("http://{renderer_addr}"),
        renderer_protocol: RendererProtocol::VllmRender,
        tokenizer_max_response_bytes: 1024,
        tokenization_timeout_ms: 5000,
        block_size: 16,
        data_parallel_size: 1,
        kv_event_port_stride: 1,
        kv_event_port: kv_addr.port(),
        replay_port: None,
        total_kv_blocks: Some(1000),
        max_num_batched_tokens: Some(8192),
        max_inflight_requests: 1,
        session_affinity_ttl_secs: None,
    };
    let selector = Arc::new(
        Selector::new(&cfg, WorkerSelectionPolicyRegistry::default())
            .await
            .unwrap(),
    );
    let kv_endpoints = HashMap::from([(0, format!("tcp://{kv_addr}"))]);
    let worker = RawWorker {
        worker_id: 1,
        pod_name: "worker".into(),
        pod_ip: "127.0.0.1".into(),
        http_endpoint: format!("http://{renderer_addr}"),
        kv_events_endpoints: kv_endpoints.clone(),
        replay_endpoint: None,
    };
    CatalogReconciler::new(selector.service.core().clone())
        .apply(&[WorkerRequest {
            worker_id: 1,
            model_name: cfg.model_name.clone(),
            endpoint: Some(worker.http_endpoint.clone()),
            block_size: Some(16),
            data_parallel_start_rank: Some(0),
            data_parallel_size: Some(1),
            kv_events_endpoints: kv_endpoints,
            ..Default::default()
        }])
        .await
        .unwrap();
    assert!(selector.any_ready().await);
    let (reflector, _changes) = PodDiscovery::for_test(vec![worker]);
    let adapter = TopologyAdapter::spawn(
        reflector.clone(),
        selector.clone(),
        RegistrationDefaults::from_config(&cfg),
    );
    let registry = Registry::new();
    let metrics = RouterRequestMetrics::from_registry(
        &registry,
        "dynamo_component",
        &[
            ("model", &cfg.model_name),
            ("inference_pool", &cfg.inference_pool_name),
        ],
    )
    .unwrap();
    let router = Arc::new(EppRouter {
        renderer: RenderClient::Vllm(
            VllmRenderClient::new(&cfg.tokenizer_service_url, Duration::from_secs(5), 1024)
                .unwrap(),
        ),
        reflector: Arc::new(reflector),
        selector,
        _adapter: adapter,
        reflector_ready: Arc::new(AtomicBool::new(true)),
        model_name: cfg.model_name,
        request_metrics: metrics.clone(),
        inflight: Arc::new(Semaphore::new(1)),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = crate::ExtProcServer::new(router.clone()).into_service();
    let server_cancel = cancel.clone();
    let server_task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(server)
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                server_cancel.cancelled_owned(),
            )
            .await
            .unwrap();
    });
    let mut client = ExternalProcessorClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    // A request-controlled model must never become a metric label.
    let body = br#"{"model":"untrusted-name","messages":[{"role":"user","content":"hello"}]}"#;
    for response in [
        br#"{"usage":{"completion_tokens":5}}"#.as_slice(),
        br#"{"usage":{"completion_tokens":0}}"#,
        br#"{"choices":[]}"#,
        br#"{"usage":{"prompt_tokens":4}}"#,
    ] {
        let responses = exchange(&mut client, body, response).await;
        assert!(
            responses
                .iter()
                .any(|r| matches!(r.response, Some(Response::RequestBody(_))))
        );
        assert!(
            !responses
                .iter()
                .any(|r| matches!(r.response, Some(Response::ImmediateResponse(_))))
        );
    }
    assert_eq!(metrics.requests_started_total.get(), 4);
    assert_eq!(metrics.input_sequence_tokens.get_sample_count(), 4);
    assert_eq!(metrics.input_sequence_tokens.get_sample_sum(), 16.0);
    assert_eq!(metrics.output_sequence_tokens.get_sample_count(), 2);
    assert_eq!(metrics.output_sequence_tokens.get_sample_sum(), 5.0);

    let permit = router.inflight.acquire().await.unwrap();
    let rejected = exchange(&mut client, body, b"").await;
    assert!(
        rejected
            .iter()
            .any(|r| matches!(r.response, Some(Response::ImmediateResponse(_))))
    );
    drop(permit);
    let rejected = exchange(&mut client, b"invalid json", b"").await;
    assert!(
        rejected
            .iter()
            .any(|r| matches!(r.response, Some(Response::ImmediateResponse(_))))
    );
    assert_eq!(metrics.requests_started_total.get(), 4);
    assert_eq!(metrics.input_sequence_tokens.get_sample_count(), 4);
    assert_eq!(metrics.output_sequence_tokens.get_sample_count(), 2);

    // Byte chunks and terminal callbacks are not router token timing or a
    // scheduler-level completion signal. Do not invent those observations.
    assert_eq!(metrics.time_to_first_token_seconds.get_sample_count(), 0);
    assert_eq!(metrics.inter_token_latency_seconds.get_sample_count(), 0);
    assert_eq!(metrics.requests_total.get(), 0);
    let mut encoded = Vec::new();
    TextEncoder::new()
        .encode(&registry.gather(), &mut encoded)
        .unwrap();
    let text = String::from_utf8(encoded).unwrap();
    assert!(text.contains("model=\"served-model\""));
    assert!(text.contains("inference_pool=\"test-pool\""));
    assert!(!text.contains("untrusted-name"));
    assert!(!text.contains("dynamo_namespace"));
    drop(client);
    cancel.cancel();
    server_task.await.unwrap();
    renderer_task.await.unwrap();
}

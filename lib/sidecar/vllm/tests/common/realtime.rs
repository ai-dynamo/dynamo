// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Json, Router,
    extract::{State, WebSocketUpgrade, ws::Message},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use base64::{Engine, prelude::BASE64_STANDARD};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Endpoint,
    distributed::{DiscoveryBackend, DistributedConfig},
    pipeline::network::{
        ResponsePlaneMode,
        egress::push_router::{PushRouter, RouterMode},
    },
    protocols::annotated::Annotated,
    storage::kv,
};
use serde_json::{Value, json};
use tokio_util::task::AbortOnDropHandle;

pub const MODEL: &str = "test-asr";
pub const DEADLINE: Duration = Duration::from_secs(30);

pub fn configure_process() {
    // Each integration binary has one test; configure global runtime settings
    // before its Tokio runtime starts.
    unsafe {
        for (key, _) in std::env::vars().filter(|(key, _)| {
            key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
        }) {
            std::env::remove_var(key);
        }
        std::env::set_var("DYN_SYSTEM_HOST", "127.0.0.1");
        std::env::set_var("DYN_SYSTEM_PORT", "0");
        std::env::set_var("DYN_ENABLE_OTEL", "false");
        std::env::set_var("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0");
        std::env::set_var("NO_PROXY", "*");
        std::env::set_var("no_proxy", "*");
    }
}

#[derive(Clone)]
struct UpstreamState {
    healthy: Arc<AtomicBool>,
    probes: Arc<AtomicUsize>,
}

pub struct Upstream {
    pub url: String,
    pub healthy: Arc<AtomicBool>,
    pub probes: Arc<AtomicUsize>,
    _task: AbortOnDropHandle<()>,
}

pub async fn start_upstream() -> Upstream {
    let state = UpstreamState {
        healthy: Arc::new(AtomicBool::new(true)),
        probes: Arc::new(AtomicUsize::new(0)),
    };
    let app = Router::new()
        .route(
            "/health",
            get(|State(state): State<UpstreamState>| async move {
                state.probes.fetch_add(1, Ordering::SeqCst);
                if state.healthy.load(Ordering::SeqCst) {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }),
        )
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data": [{"id": MODEL}]})) }),
        )
        .route("/v1/realtime", get(upstream))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = AbortOnDropHandle::new(tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    }));
    Upstream {
        url,
        healthy: state.healthy,
        probes: state.probes,
        _task: task,
    }
}

async fn upstream(
    State(state): State<UpstreamState>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    if !state.healthy.load(Ordering::SeqCst) {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    ws.on_upgrade(|mut socket| async move {
        socket
            .send(Message::Text(
                json!({"type": "session.created", "id": "upstream"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        // Startup and active health probes close after session.created.
        while let Some(Ok(Message::Text(text))) = socket.recv().await {
            let request: Value = serde_json::from_str(&text).unwrap();
            let event = match request["type"].as_str() {
                Some("session.update") => {
                    assert_eq!(request["model"], MODEL);
                    continue;
                }
                Some("input_audio_buffer.append") => {
                    assert!(
                        !BASE64_STANDARD
                            .decode(request["audio"].as_str().unwrap())
                            .unwrap()
                            .is_empty()
                    );
                    json!({"type": "transcription.delta", "delta": "hello"})
                }
                Some("input_audio_buffer.commit") if request["final"] == true => {
                    json!({"type": "transcription.done", "text": "hello"})
                }
                Some("input_audio_buffer.commit") => continue,
                other => panic!("unexpected upstream request: {other:?}"),
            };
            socket
                .send(Message::Text(event.to_string().into()))
                .await
                .unwrap();
        }
    })
}

pub async fn runtime(discovery: &Path) -> (Runtime, DistributedRuntime) {
    let runtime = Runtime::from_current().unwrap();
    let mut config = DistributedConfig::process_local();
    config.discovery_backend = DiscoveryBackend::KvStore(kv::Selector::File(discovery.into()));
    config.response_plane = Some(ResponsePlaneMode::Tcp);
    let drt = DistributedRuntime::new(runtime.clone(), config)
        .await
        .unwrap();
    (runtime, drt)
}

pub fn endpoint(drt: &DistributedRuntime) -> Endpoint {
    drt.namespace("media-test")
        .unwrap()
        .component("asr")
        .unwrap()
        .endpoint("generate")
}

pub async fn router(endpoint: &Endpoint) -> PushRouter<Value, Annotated<Value>> {
    let client = endpoint.client().await.unwrap();
    tokio::time::timeout(DEADLINE, client.wait_for_instances())
        .await
        .unwrap()
        .unwrap();
    PushRouter::from_client_with_options(
        client,
        RouterMode::RoundRobin,
        dynamo_llm::kv_router::plugins::push_router_options(),
    )
    .await
    .unwrap()
}

pub fn session_update() -> Value {
    json!({"type": "session.update", "session": {
        "type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": MODEL}, "turn_detection": null
        }}
    }})
}

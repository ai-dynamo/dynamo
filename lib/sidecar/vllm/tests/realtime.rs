// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the sidecar executable through discovery and the bidirectional
//! request plane. The upstream is a deterministic vLLM transcription service.

use std::{process::Command, time::Duration};

use axum::{
    Json, Router,
    extract::{WebSocketUpgrade, ws::Message},
    routing::get,
};
use base64::{Engine, prelude::BASE64_STANDARD};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::{DiscoveryBackend, DistributedConfig},
    engine::AsyncEngine,
    pipeline::{
        RequestStream,
        context::Context,
        network::{
            ResponsePlaneMode,
            egress::push_router::{PushRouter, RouterMode},
        },
    },
    protocols::annotated::Annotated,
    storage::kv,
};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio_stream::wrappers::ReceiverStream;

const MODEL: &str = "test-asr";
const DEADLINE: Duration = Duration::from_secs(30);

struct Sidecar(std::process::Child);

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn upstream(ws: WebSocketUpgrade) -> axum::response::Response {
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

#[test]
fn asr_executable_streams_through_discovery_and_tcp_before_commit() {
    // This binary contains one test. Configure its process-global runtime
    // before starting any threads, independently of developer settings.
    unsafe {
        for (key, _) in std::env::vars().filter(|(key, _)| {
            key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
        }) {
            std::env::remove_var(key);
        }
        std::env::set_var("DYN_SYSTEM_HOST", "127.0.0.1");
        std::env::set_var("DYN_SYSTEM_PORT", "0");
        std::env::set_var("DYN_ENABLE_OTEL", "false");
    }
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(round_trip());
}

async fn round_trip() {
    let app = Router::new()
        .route("/health", get(|| async { "healthy" }))
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data": [{"id": MODEL}]})) }),
        )
        .route("/v1/realtime", get(upstream));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let discovery = tempfile::tempdir().unwrap();
    let logs = tempfile::NamedTempFile::new().unwrap();
    let mut child = Sidecar(
        Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-sidecar"))
            .args([
                "realtime",
                "--protocol",
                "asr",
                "--http-endpoint",
                &upstream_url,
                "--model",
                MODEL,
                "--namespace",
                "media-test",
                "--component",
                "asr",
                "--discovery-backend",
                "file",
                "--request-plane",
                "tcp",
                "--response-plane",
                "tcp",
                "--event-plane",
                "zmq",
            ])
            .env("DYN_FILE_KV", discovery.path())
            .stderr(logs.as_file().try_clone().unwrap())
            .spawn()
            .unwrap(),
    );
    let runtime = Runtime::from_current().unwrap();
    let mut config = DistributedConfig::process_local();
    config.discovery_backend =
        DiscoveryBackend::KvStore(kv::Selector::File(discovery.path().into()));
    config.response_plane = Some(ResponsePlaneMode::Tcp);
    let drt = DistributedRuntime::new(runtime.clone(), config)
        .await
        .unwrap();
    let endpoint = drt
        .namespace("media-test")
        .unwrap()
        .component("asr")
        .unwrap()
        .endpoint("generate");
    let client = endpoint.client().await.unwrap();
    tokio::time::timeout(DEADLINE, client.wait_for_instances())
        .await
        .unwrap_or_else(|error| {
            panic!(
                "sidecar did not register: {error}\n{}",
                std::fs::read_to_string(logs.path()).unwrap()
            )
        })
        .unwrap();
    let router = PushRouter::<Value, Annotated<Value>>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
    let (input, receiver) = tokio::sync::mpsc::channel(4);
    input
        .send(json!({"type": "session.update", "session": {
            "type": "transcription", "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": MODEL}, "turn_detection": null
            }}
        }}))
        .await
        .unwrap();
    let mut output = tokio::time::timeout(
        DEADLINE,
        router.generate(Context::new(RequestStream::new(
            ReceiverStream::new(receiver).boxed(),
        ))),
    )
    .await
    .unwrap()
    .unwrap();
    let updated = tokio::time::timeout(DEADLINE, output.next())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.data.unwrap()["type"], "session.updated");
    input
        .send(json!({"type": "input_audio_buffer.append",
            "audio": BASE64_STANDARD.encode(vec![0u8; 9600])}))
        .await
        .unwrap();
    let delta = tokio::time::timeout(DEADLINE, output.next())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(delta.data.unwrap()["delta"], "hello");
    input
        .send(json!({"type": "input_audio_buffer.commit"}))
        .await
        .unwrap();
    drop(input);
    let tail: Vec<_> = tokio::time::timeout(DEADLINE, output.collect())
        .await
        .unwrap();
    assert!(tail.iter().all(|event| event.data.is_some()));
    let completed = tail.last().unwrap().data.as_ref().unwrap();
    assert_eq!(
        completed["type"],
        "conversation.item.input_audio_transcription.completed"
    );
    assert_eq!(completed["transcript"], "hello");

    assert!(
        Command::new("kill")
            .args(["-TERM", &child.0.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let status = tokio::time::timeout(DEADLINE, async {
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("sidecar shuts down after SIGTERM");
    assert!(status.success());
    runtime.shutdown();
    server.abort();
}

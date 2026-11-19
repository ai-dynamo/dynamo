// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the sidecar executable through discovery and the bidirectional
//! request plane. The upstream is a deterministic vLLM transcription service.

#[path = "common/realtime.rs"]
mod common;

use std::{process::Command, sync::atomic::Ordering, time::Duration};

use base64::{Engine, prelude::BASE64_STANDARD};
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{RequestStream, context::Context},
};
use futures::StreamExt;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;

use common::{DEADLINE, MODEL};

struct Sidecar(std::process::Child);

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn asr_executable_streams_through_discovery_and_tcp_before_commit() {
    common::configure_process();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(round_trip());
}

async fn round_trip() {
    let upstream = common::start_upstream().await;
    let discovery = tempfile::tempdir().unwrap();
    let logs = tempfile::NamedTempFile::new().unwrap();
    let mut child = Sidecar(
        Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-sidecar"))
            .args([
                "realtime",
                "--protocol",
                "asr",
                "--http-endpoint",
                &upstream.url,
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
    let (runtime, drt) = common::runtime(discovery.path()).await;
    let endpoint = common::endpoint(&drt);
    let router = common::router(&endpoint).await;
    assert!(upstream.healthy.load(Ordering::SeqCst));
    assert!(upstream.probes.load(Ordering::SeqCst) > 0);
    let (input, receiver) = tokio::sync::mpsc::channel(4);
    input.send(common::session_update()).await.unwrap();
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
}

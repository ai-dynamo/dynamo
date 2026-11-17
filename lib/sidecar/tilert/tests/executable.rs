// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::{Command, Stdio};
use std::time::Duration;

fn sidecar() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dynamo-tilert-sidecar"));
    for (key, _) in std::env::vars().filter(|(key, _)| {
        key.starts_with("DYN_")
            || key.starts_with("TILERT_")
            || key.starts_with("ETCD_")
            || key.starts_with("NATS_")
    }) {
        command.env_remove(key);
    }
    command
}

#[test]
fn help_names_the_decode_endpoint_variable() {
    let output = sidecar()
        .arg("--help")
        .output()
        .expect("run dynamo-tilert-sidecar --help");
    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    assert!(stdout.contains("TILERT_DECODE_HTTP_ENDPOINT"), "{stdout}");
}

/// Unsupported roles and flags fail before the runtime is configured;
/// accepted settings reach runtime validation.
#[test]
fn unsupported_arguments_fail_before_runtime_configuration() {
    for (extra, expect) in [
        (
            &["--model-path", "model", "--disaggregation-mode", "prefill"][..],
            "run it in aggregated mode",
        ),
        (
            &["--model-path", "model", "--route-to-encoder"][..],
            "route-to-encoder is not supported",
        ),
        (
            &["--model-path", "model", "--dyn-structural-tag", "true"][..],
            "--dyn-structural-tag is not supported",
        ),
        (&["--model-path", " "][..], "model-path must not be empty"),
        (
            &["--model-path", "model"][..],
            "num_worker_threads: Validation error",
        ),
    ] {
        let output = sidecar()
            .args(["--grpc-endpoint", "http://127.0.0.1:1"])
            .args(["--decode-http-endpoint", "http://127.0.0.1:2"])
            .args(["--context-length", "8"])
            .args(extra)
            .env("DYN_RUNTIME_NUM_WORKER_THREADS", "0")
            .output()
            .expect("run dynamo-tilert-sidecar");
        assert!(!output.status.success(), "{extra:?} must fail");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(expect), "{extra:?}: {stderr}");
    }
}

struct Sidecar(std::process::Child);

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Engines can take many minutes to load, and the sidecar waits for them up
/// to its startup deadline. A shutdown must not wait that long.
#[tokio::test]
async fn sigterm_stops_a_sidecar_that_is_waiting_for_its_engines() {
    let blackhole = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let prefill = format!("http://{}", blackhole.local_addr().unwrap());
    let discovery = tempfile::tempdir().unwrap();
    let mut child = Sidecar(
        sidecar()
            .args(["--grpc-endpoint", &prefill])
            .args(["--decode-http-endpoint", "http://127.0.0.1:2"])
            .args(["--model-path", "model", "--context-length", "8"])
            .args(["--discovery-backend", "file", "--event-plane", "zmq"])
            .args(["--request-plane", "tcp", "--response-plane", "tcp"])
            .env("DYN_SYSTEM_HOST", "127.0.0.1")
            .env("DYN_SYSTEM_PORT", "0")
            .env("DYN_FILE_KV", discovery.path())
            .env("DYN_ENABLE_OTEL", "false")
            .stderr(Stdio::null())
            .spawn()
            .expect("start sidecar"),
    );
    // The sidecar dials the prefill only once its bootstrap is under way.
    tokio::time::timeout(Duration::from_secs(15), blackhole.accept())
        .await
        .expect("the sidecar starts waiting for its engines")
        .unwrap();

    let killed = Command::new("kill")
        .args(["-TERM", &child.0.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let status = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("SIGTERM ends the wait for the engines");
    assert!(status.success(), "sidecar shutdown failed: {status}");
}

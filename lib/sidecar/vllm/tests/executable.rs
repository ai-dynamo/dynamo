// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

#[test]
fn executable_exposes_native_grpc_configuration() {
    let output = Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-sidecar"))
        .arg("--help")
        .output()
        .expect("run dynamo-vllm-sidecar --help");

    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    for flag in [
        "--grpc-endpoint",
        "--grpc-connections",
        "--disaggregation-mode",
        "--grpc-connect-attempt-timeout-secs",
        "--grpc-retry-interval-secs",
        "--grpc-startup-deadline-secs",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in help output");
    }
    for env in [
        "DYN_SIDECAR_GRPC_ENDPOINT",
        "DYN_SIDECAR_GRPC_CONNECTIONS",
        "DYN_SIDECAR_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS",
        "DYN_SIDECAR_GRPC_RETRY_INTERVAL_SECS",
        "DYN_SIDECAR_GRPC_STARTUP_DEADLINE_SECS",
    ] {
        assert!(stdout.contains(env), "missing {env} in help output");
    }
}

struct Sidecar(std::process::Child);

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn sidecar(engine: u16, logs: &std::fs::File) -> Sidecar {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-sidecar"));
    // Keep these subprocess tests independent of the developer's runtime settings.
    for (key, _) in std::env::vars().filter(|(key, _)| {
        key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
    }) {
        command.env_remove(key);
    }
    Sidecar(
        command
            .args([
                "--grpc-endpoint",
                &format!("http://127.0.0.1:{engine}"),
                "--grpc-startup-deadline-secs",
                "60",
            ])
            .env("DYN_SYSTEM_HOST", "127.0.0.1")
            .env("DYN_SYSTEM_PORT", "0")
            .env("DYN_LOG", "info")
            .env("DYN_LOGGING_CONSOLE_FORMAT", "jsonl")
            .stderr(logs.try_clone().unwrap())
            .env("DYN_DISCOVERY_BACKEND", "mem")
            .env("DYN_REQUEST_PLANE", "tcp")
            .env("DYN_EVENT_PLANE", "zmq")
            .env("DYN_ENABLE_OTEL", "false")
            .spawn()
            .expect("start sidecar"),
    )
}

async fn wait_status(child: &mut Sidecar, client: &reqwest::Client, url: &str, expected: u16) {
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        loop {
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "sidecar exited before {url} returned {expected}"
            );
            if let Ok(response) = client.get(url).send().await
                && response.status().as_u16() == expected
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("probe response within startup deadline");
}

async fn terminate(child: &mut Sidecar) {
    assert!(
        Command::new("kill")
            .args(["-TERM", &child.0.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let status = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("SIGTERM cancels initialization");
    assert!(status.success(), "sidecar shutdown failed: {status}");
}

#[tokio::test]
async fn probes_work_before_engine_is_available() {
    let blackhole = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable_port = blackhole.local_addr().unwrap().port();
    let logs = tempfile::NamedTempFile::new().unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap();

    // No engine: runtime readiness must pass without metadata or registration.
    let mut child = sidecar(unavailable_port, logs.as_file());
    let (_engine_connection, _) =
        tokio::time::timeout(std::time::Duration::from_secs(15), blackhole.accept())
            .await
            .expect("sidecar starts engine bootstrap")
            .unwrap();
    // DRT binds before engine bootstrap. Read the allocated port from its log;
    // the OS owns the reservation continuously, with no test/child bind race.
    let log = std::fs::read_to_string(logs.path()).unwrap();
    let address = log
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find_map(|event| {
            event["message"]
                .as_str()?
                .strip_prefix("[spawn_system_status_server] system status server bound to: ")?
                .parse::<std::net::SocketAddr>()
                .ok()
        })
        .unwrap_or_else(|| panic!("missing bound address in sidecar log: {log}"));
    assert_ne!(address.port(), 0);
    let base = format!("http://{address}");
    wait_status(&mut child, &client, &format!("{base}/live"), 200).await;
    wait_status(&mut child, &client, &format!("{base}/health"), 200).await;
    wait_status(&mut child, &client, &format!("{base}/metrics"), 200).await;
    terminate(&mut child).await;
}

#[test]
fn invalid_arguments_fail_before_runtime_configuration() {
    let overflowing_deadline = u64::MAX.to_string();
    for (args, message) in [
        (
            ["--dyn-tool-call-parser", "hermes"],
            "vLLM gRPC does not preserve",
        ),
        (
            ["--vllm-http-endpoint", "http://localhost:8000?invalid=true"],
            "must not include a query or fragment",
        ),
        (
            [
                "--grpc-startup-deadline-secs",
                overflowing_deadline.as_str(),
            ],
            "exceeds the supported monotonic clock range",
        ),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-sidecar"));
        for (key, _) in std::env::vars().filter(|(key, _)| {
            key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
        }) {
            command.env_remove(key);
        }
        // A runtime configuration error must not mask a local argument error.
        let output = command
            .args(["--grpc-endpoint", "http://127.0.0.1:0"])
            .args(args)
            .env("DYN_DISCOVERY_BACKEND", "invalid-backend")
            .output()
            .unwrap();
        assert!(!output.status.success(), "{args:?} unexpectedly succeeded");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(message), "{args:?}: {stderr}");
    }
}

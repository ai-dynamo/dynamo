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

fn probe_port() -> u16 {
    // DYN_SYSTEM_PORT is currently an i16, so ephemeral OS ports can be too high.
    (18000..32000)
        .find(|port| std::net::TcpListener::bind(("127.0.0.1", *port)).is_ok())
        .expect("a free probe port")
}

fn sidecar(port: u16, discovery: &str, engine: u16, etcd: u16) -> Sidecar {
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
            .env("DYN_SYSTEM_PORT", port.to_string())
            .env("DYN_DISCOVERY_BACKEND", discovery)
            .env("DYN_REQUEST_PLANE", "tcp")
            .env("DYN_EVENT_PLANE", "zmq")
            .env("DYN_ENABLE_OTEL", "false")
            .env("ETCD_ENDPOINTS", format!("http://127.0.0.1:{etcd}"))
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
async fn probes_work_before_engine_and_discovery_are_available() {
    let blackhole = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let unavailable_port = blackhole.local_addr().unwrap().port();
    let port = probe_port();
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap();

    // No engine: runtime readiness must pass without metadata or registration.
    let mut child = sidecar(port, "mem", unavailable_port, unavailable_port);
    wait_status(&mut child, &client, &format!("{base}/live"), 200).await;
    wait_status(&mut child, &client, &format!("{base}/health"), 200).await;
    wait_status(&mut child, &client, &format!("{base}/metrics"), 200).await;
    terminate(&mut child).await;

    // An unresponsive discovery server must not prevent liveness or termination.
    let mut child = sidecar(port, "etcd", unavailable_port, unavailable_port);
    wait_status(&mut child, &client, &format!("{base}/live"), 200).await;
    wait_status(&mut child, &client, &format!("{base}/health"), 503).await;
    terminate(&mut child).await;
}

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

// A bound-but-silent engine must not prevent Kubernetes from starting the main
// container. Exercise the real executable, including signals and DRT ownership.
#[tokio::test]
async fn probes_are_available_before_engine_discovery_and_sigterm_exits() {
    use std::net::TcpListener;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    struct Child(std::process::Child);
    impl Drop for Child {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    // RuntimeConfig currently stores system_port as i16. Select a free dynamic
    // port in its supported range, retaining the reservation until launch.
    let reservation = (0..100)
        .find_map(|_| {
            let ephemeral = TcpListener::bind("127.0.0.1:0").unwrap();
            let candidate = 16384 + ephemeral.local_addr().unwrap().port() % 16384;
            TcpListener::bind(("127.0.0.1", candidate)).ok()
        })
        .expect("reserve an available system port");
    let address = reservation.local_addr().unwrap();
    let engine = TcpListener::bind("127.0.0.1:0").unwrap();
    let engine_address = engine.local_addr().unwrap();
    drop(reservation);
    let mut child = Child(
        Command::new(env!("CARGO_BIN_EXE_dynamo-vllm-sidecar"))
            .args([
                "--grpc-endpoint",
                &engine_address.to_string(),
                "--grpc-startup-deadline-secs",
                "60",
            ])
            .env("DYN_SYSTEM_HOST", "127.0.0.1")
            .env("DYN_SYSTEM_PORT", address.port().to_string())
            .env("DYN_DISCOVERY_BACKEND", "mem")
            .env("DYN_REQUEST_PLANE", "tcp")
            .env("DYN_EVENT_PLANE", "zmq")
            .env("DYN_SYSTEM_HEALTH_PATH", "/health")
            .env("DYN_SYSTEM_LIVE_PATH", "/live")
            .env("DYN_HEALTH_CHECK_ENABLED", "false")
            .env("DYN_COMPUTE_THREADS", "0")
            .env("DYN_RUNTIME_NUM_WORKER_THREADS", "2")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "sidecar exited during discovery"
        );
        if let Ok(response) = client.get(format!("http://{address}/health")).send().await
            && response.status() == 200
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "runtime probes unavailable while engine discovery is blocked"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        client
            .get(format!("http://{address}/live"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        client
            .get(format!("http://{address}/metrics"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert!(
        Command::new("kill")
            .args(["-TERM", &child.0.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(
                status.success(),
                "sidecar must exit cleanly on SIGTERM: {status}"
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "SIGTERM did not cancel engine discovery"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    TcpListener::bind(address).expect("sidecar releases its HTTP listener");
}

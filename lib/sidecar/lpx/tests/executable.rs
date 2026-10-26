// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::Command;
use std::time::{Duration, Instant};

fn sidecar() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dynamo-lpx-sidecar"));
    for (key, _) in std::env::vars().filter(|(key, _)| {
        key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
    }) {
        command.env_remove(key);
    }
    command.env_remove("DYNAMO_RANK");
    command
}

/// `--served-model-name` also reads `DYN_SERVED_MODEL_NAME`. Every other flag
/// and variable this sidecar adds is exercised by a behaviour test.
#[test]
fn help_names_dyn_served_model_name() {
    let output = sidecar()
        .arg("--help")
        .output()
        .expect("run dynamo-lpx-sidecar --help");
    assert!(
        output.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("help output is UTF-8");
    assert!(stdout.contains("DYN_SERVED_MODEL_NAME"), "{stdout}");
}

/// Unsupported roles and flags fail before the runtime is configured;
/// accepted settings reach runtime validation.
#[test]
fn unsupported_arguments_fail_before_runtime_configuration() {
    for (extra, expect) in [
        (
            &["--model-path", "model", "--disaggregation-mode", "prefill"][..],
            "does not serve --disaggregation-mode prefill",
        ),
        (
            &["--model-path", "model", "--route-to-encoder"][..],
            "route-to-encoder is not supported",
        ),
        (
            &["--model-path", "model", "--dyn-structural-tag", "true"][..],
            "--dyn-structural-tag is not supported",
        ),
        (
            &["--model-path", "model", "--dyn-structural-tag", "{}"][..],
            "--dyn-structural-tag is not supported",
        ),
        (
            &["--model-path", "model"][..],
            "num_worker_threads: Validation error",
        ),
        (
            &["--model-path", "model", "--dyn-structural-tag", "false"][..],
            "num_worker_threads: Validation error",
        ),
        (&["--model-path", " "][..], "model-path must not be empty"),
    ] {
        let output = sidecar()
            .args(["--grpc-endpoint", "http://127.0.0.1:1"])
            .args(extra)
            .env("DYN_RUNTIME_NUM_WORKER_THREADS", "0")
            .output()
            .expect("run dynamo-lpx-sidecar");
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

/// Sends one `GET` and returns the status code, the content type and the body.
fn get(address: SocketAddr, path: &str) -> (u16, Option<String>, String) {
    let mut stream = TcpStream::connect(address).expect("connect to the sidecar");
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: sidecar\r\nConnection: close\r\n\r\n"
    )
    .expect("send the request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read the response");
    let (head, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("no complete response in {response:?}"));
    let status = head
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status code in {response:?}"));
    let content_type = head.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("content-type")
            .then(|| value.trim().to_string())
    });
    (status, content_type, body.to_string())
}

/// Waits for the JSON log event with `message`, failing if the sidecar exits.
fn wait_for_log(child: &mut Sidecar, logs: &std::path::Path, message: &str) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let log = std::fs::read_to_string(logs).expect("read the sidecar log");
        if let Some(event) = log
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|event| event["message"] == message)
        {
            return event;
        }
        if let Some(status) = child.0.try_wait().expect("poll the sidecar") {
            panic!("sidecar exited with {status} before logging {message:?}: {log}");
        }
        assert!(
            Instant::now() < deadline,
            "no {message:?} in the sidecar log: {log}"
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Only rank 0 of a multi-Pod engine serves. Any other rank answers its
/// probes and metrics scrapes, dials neither the engine nor a Dynamo
/// dependency, and exits cleanly on SIGTERM. It starts with the serving
/// arguments rank 0 would take, and with ones rank 0 would reject.
#[test]
fn a_nonzero_node_rank_stands_by() {
    // The engine, etcd and NATS all point at one listener that no one may dial.
    let peer = TcpListener::bind("127.0.0.1:0").expect("bind the peer listener");
    peer.set_nonblocking(true)
        .expect("nonblocking peer listener");
    let peer_address = peer.local_addr().expect("peer address");
    let logs = tempfile::NamedTempFile::new().expect("log file");
    let mut child = Sidecar(
        sidecar()
            .args([
                "--grpc-endpoint",
                &format!("http://{peer_address}"),
                "--model-path",
                "model-source",
                // An agent template written for the TensorRT-LLM sidecar
                // passes it.
                "--served-model-name",
                "llama32-1b",
                // Rank 0 would refuse this role before connecting.
                "--disaggregation-mode",
                "prefill",
                "--discovery-backend",
                "etcd",
                "--request-plane",
                "nats",
                "--event-plane",
                "nats",
            ])
            // A launcher may pass the rank through the environment instead of --node-rank.
            .env("DYNAMO_RANK", "1")
            .env("ETCD_ENDPOINTS", format!("http://{peer_address}"))
            .env("NATS_SERVER", format!("nats://{peer_address}"))
            .env("DYN_SYSTEM_HOST", "127.0.0.1")
            .env("DYN_SYSTEM_PORT", "0")
            // The probe paths are the runtime's, as on rank 0.
            .env("DYN_SYSTEM_HEALTH_PATH", "/ready")
            .env("DYN_LOG", "info")
            .env("DYN_LOGGING_CONSOLE_FORMAT", "jsonl")
            .stderr(logs.reopen().expect("reopen the log file"))
            .spawn()
            .expect("start the sidecar"),
    );

    let event = wait_for_log(
        &mut child,
        logs.path(),
        "Sidecar standing by; only rank 0 serves",
    );
    let address: SocketAddr = event["address"]
        .as_str()
        .and_then(|address| address.parse().ok())
        .unwrap_or_else(|| panic!("no bound address in {event}"));
    for path in ["/ready", "/live"] {
        assert_eq!(get(address, path).0, 200, "{path}");
    }
    // A scraped standby must not show up as a down target.
    let (status, content_type, body) = get(address, "/metrics");
    assert_eq!(status, 200);
    assert_eq!(content_type.as_deref(), Some("text/plain; charset=utf-8"));
    assert!(body.is_empty(), "standby metrics must be empty: {body}");
    assert_eq!(get(address, "/metadata").0, 404);
    match peer.accept() {
        Err(error) if error.kind() == ErrorKind::WouldBlock => {}
        accepted => panic!("a standby dialed out: {accepted:?}"),
    }

    assert!(
        Command::new("kill")
            .args(["-TERM", &child.0.id().to_string()])
            .status()
            .expect("run kill")
            .success()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.0.try_wait().expect("poll the sidecar") {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "SIGTERM did not stop the standby"
        );
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(status.success(), "standby shutdown failed: {status}");
}

/// Kubernetes resolves a rank bound to a missing Pod label to an empty value.
/// That must fail at startup, not serve as a second rank 0.
#[test]
fn an_empty_node_rank_fails() {
    let output = sidecar()
        .args([
            "--grpc-endpoint",
            "http://127.0.0.1:1",
            "--model-path",
            "model-source",
        ])
        .env("DYNAMO_RANK", "")
        .output()
        .expect("run the sidecar");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--node-rank"), "{stderr}");
}

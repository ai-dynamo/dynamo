// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
    process::{Child, Command},
    time::Duration,
};

const BINARY: &str = env!("CARGO_BIN_EXE_dynamo-speech-nim-sidecar");
const DISPATCHER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../dynamo-sidecar");

fn command() -> Command {
    let mut command = Command::new(BINARY);
    // Tests must not inherit local cluster, logging, or worker settings.
    for (key, _) in std::env::vars().filter(|(key, _)| {
        key.starts_with("DYN_") || key.starts_with("ETCD_") || key.starts_with("NATS_")
    }) {
        command.env_remove(key);
    }
    command
}

#[test]
fn dispatcher_exposes_both_speech_modes_and_shared_configuration() {
    let bin = tempfile::tempdir().unwrap();
    symlink(BINARY, bin.path().join("dynamo-speech-nim-sidecar")).unwrap();
    for mode in ["asr", "tts"] {
        let output = Command::new("/bin/sh")
            .args([DISPATCHER, "speech-nim", mode, "--help"])
            .env("PATH", bin.path())
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let help = String::from_utf8(output.stdout).unwrap();
        for flag in [
            "--grpc-endpoint",
            "--grpc-connections",
            "--grpc-startup-deadline-secs",
            "--discovery-backend",
            "--request-plane",
            "--response-plane",
            "--event-plane",
            "--model-name",
        ] {
            assert!(help.contains(flag), "{mode}: missing {flag}");
        }
        assert!(help.contains(if mode == "asr" {
            "--max-concurrent-turns"
        } else {
            "--sample-rate-hz"
        }));
    }
}

#[test]
fn dispatcher_preserves_engine_selection_arguments_and_exit_status() {
    let bin = tempfile::tempdir().unwrap();
    for engine in ["vllm", "sglang", "trtllm", "lpx", "speech-nim"] {
        let path = bin.path().join(format!("dynamo-{engine}-sidecar"));
        std::fs::write(
            &path,
            format!("#!/bin/sh\nprintf '%s\\n' '{engine}' \"$@\"\nexit 37\n"),
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let output = Command::new("/bin/sh")
            .args([DISPATCHER, engine, "argument with spaces", "", "--help"])
            .env("PATH", bin.path())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(37));
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("{engine}\nargument with spaces\n\n--help\n")
        );
    }
    for args in [vec![], vec!["unknown-engine"]] {
        let output = Command::new("/bin/sh")
            .arg(DISPATCHER)
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
    }
}

#[test]
fn invalid_speech_configuration_fails_before_runtime_startup() {
    for mode in ["asr", "tts"] {
        let output = command()
            .args([mode, "--grpc-endpoint", "10.0.0.1:50051"])
            .env("DYN_DISCOVERY_BACKEND", "invalid-backend")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("use a loopback address"));
    }
}

struct Sidecar(Child);

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn probe_address(child: &mut Sidecar, logs: &Path) -> std::net::SocketAddr {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            assert!(child.0.try_wait().unwrap().is_none(), "sidecar exited");
            let log = std::fs::read_to_string(logs).unwrap();
            if let Some(address) = log
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .find_map(|event| {
                    event["message"]
                        .as_str()?
                        .strip_prefix(
                            "[spawn_system_status_server] system status server bound to: ",
                        )?
                        .parse()
                        .ok()
                })
            {
                return address;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("system probe address")
}

#[tokio::test]
async fn probes_and_sigterm_work_while_nim_initialization_is_pending() {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    for mode in ["asr", "tts"] {
        // Accept TCP but never complete a gRPC handshake.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let discovery = tempfile::tempdir().unwrap();
        let logs = tempfile::NamedTempFile::new().unwrap();
        let mut child = Sidecar(
            command()
                .args([
                    mode,
                    "--grpc-endpoint",
                    &endpoint,
                    "--grpc-startup-deadline-secs",
                    "60",
                    "--discovery-backend",
                    "file",
                    "--request-plane",
                    "tcp",
                    "--response-plane",
                    "tcp",
                    "--event-plane",
                    "zmq",
                ])
                .env("DYN_SYSTEM_HOST", "127.0.0.1")
                .env("DYN_SYSTEM_PORT", "0")
                .env("DYN_FILE_KV", discovery.path())
                .env("DYN_ENABLE_OTEL", "false")
                .env("DYN_LOG", "info")
                .env("DYN_LOGGING_CONSOLE_FORMAT", "jsonl")
                .env("DYN_RUNTIME_NUM_WORKER_THREADS", "2")
                .env("DYN_COMPUTE_THREADS", "0")
                // The CLI overrides must take effect before runtime startup.
                .env("DYN_DISCOVERY_BACKEND", "invalid-backend")
                .env("DYN_REQUEST_PLANE", "invalid-transport")
                .env("DYN_RESPONSE_PLANE", "invalid-transport")
                .env("DYN_EVENT_PLANE", "invalid-transport")
                .stderr(logs.as_file().try_clone().unwrap())
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        );
        let _connection = tokio::time::timeout(Duration::from_secs(15), listener.accept())
            .await
            .expect("sidecar connects to NIM")
            .unwrap();
        let address = probe_address(&mut child, logs.path()).await;
        for probe in ["live", "health", "metrics"] {
            let response = client
                .get(format!("http://{address}/{probe}"))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), 200, "{mode}: /{probe}");
        }
        assert!(
            Command::new("kill")
                .args(["-TERM", &child.0.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        let status = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("SIGTERM cancels NIM initialization");
        assert!(status.success(), "{mode}: {status}");
    }
}

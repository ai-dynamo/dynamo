// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// SIGTERM stops the sidecar while it is still retrying its connection to an unreachable Triton.
#[test]
fn sigterm_stops_startup_while_connecting() {
    // A port that was just free, so connection attempts are refused.
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("reserve a local port")
        .port();
    let mut sidecar = Command::new(env!("CARGO_BIN_EXE_dynamo-triton-sidecar"))
        .args(["--grpc", &format!("127.0.0.1:{port}")])
        .args(["--discovery-backend", "mem"])
        .env("DYN_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start dynamo-triton-sidecar");

    // Wait until the sidecar is retrying the connection.
    let stderr = sidecar.stderr.take().expect("piped stderr");
    let (retrying, retrying_rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if line.contains("connection attempt failed") {
                let _ = retrying.send(());
            }
        }
    });
    if retrying_rx.recv_timeout(Duration::from_secs(60)).is_err() {
        sidecar.kill().ok();
        panic!("sidecar never retried its Triton connection");
    }

    let status = Command::new("kill")
        .args(["-TERM", &sidecar.id().to_string()])
        .status()
        .expect("run kill");
    assert!(status.success(), "kill -TERM failed");
    let signalled = Instant::now();
    let exit = loop {
        if let Some(exit) = sidecar.try_wait().expect("poll sidecar") {
            break exit;
        }
        if signalled.elapsed() > Duration::from_secs(15) {
            sidecar.kill().ok();
            sidecar.wait().ok();
            panic!("sidecar still running 15s after SIGTERM");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(exit.success(), "sidecar exited with {exit} after SIGTERM");
}

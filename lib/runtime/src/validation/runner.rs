// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs one gate to a verdict. `Err` carries the reason shown on `/health`.

use std::process::Stdio;

use tokio::io::AsyncReadExt;

use super::{GateCheck, GateSpec};

/// How much of a failed check's stderr, or a check runner's response, is kept
/// as its reason.
const REASON_TAIL_BYTES: usize = 2048;

/// Wait between attempts while a check runner is not reachable yet.
const URL_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

pub(super) async fn run_gate(gate: &GateSpec, env: &[(&str, String)]) -> Result<(), String> {
    let run = async {
        match &gate.check {
            GateCheck::Exec(argv) => run_exec(&gate.name, argv, env).await,
            GateCheck::Url(url) => run_url(&gate.name, url, env).await,
        }
    };
    tokio::time::timeout(gate.timeout, run)
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "no verdict within timeout_secs={}",
                gate.timeout.as_secs()
            ))
        })
}

/// Run the command; exit status 0 is a pass. The child is killed if the
/// deadline drops this future.
async fn run_exec(gate: &str, argv: &[String], env: &[(&str, String)]) -> Result<(), String> {
    let mut child = tokio::process::Command::new(&argv[0])
        .args(&argv[1..])
        .envs(env.iter().map(|(key, value)| (key, value)))
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not run {:?}: {error}", argv[0]))?;

    let mut stderr = Vec::new();
    let mut pipe = child.stderr.take().expect("stderr is piped");
    let (status, _) = tokio::join!(child.wait(), pipe.read_to_end(&mut stderr));
    let status = status.map_err(|error| format!("waiting for {:?}: {error}", argv[0]))?;
    if status.success() {
        return Ok(());
    }
    let tail = tail(&stderr);
    tracing::warn!(gate, %status, stderr = %tail, "Validation check failed");
    Err(format!("{status}: {tail}"))
}

/// POST the gate to its check runner. A `200` is a pass and any other
/// response a fail, with the end of the body as the reason. A runner that
/// can't be reached yet, such as a sidecar still starting, is retried until
/// the deadline.
async fn run_url(gate: &str, url: &reqwest::Url, env: &[(&str, String)]) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| error.to_string())?;
    let body: serde_json::Map<_, _> = env
        .iter()
        .map(|(key, value)| (key.to_string(), value.clone().into()))
        .collect();
    loop {
        match client.post(url.clone()).json(&body).send().await {
            Ok(response) if response.status() == reqwest::StatusCode::OK => return Ok(()),
            Ok(response) => {
                let status = response.status();
                let body = response.bytes().await.unwrap_or_default();
                let reason = tail(&body);
                tracing::warn!(gate, %status, %reason, "Validation check failed");
                return Err(format!("check runner returned {status}: {reason}"));
            }
            Err(error) => {
                tracing::debug!(gate, %error, "Check runner not reachable yet; retrying");
                tokio::time::sleep(URL_RETRY_INTERVAL).await;
            }
        }
    }
}

fn tail(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[bytes.len().saturating_sub(REASON_TAIL_BYTES)..])
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn url_gate_waits_for_its_runner_and_takes_its_status() {
        use axum::{Router, http::StatusCode, routing::post};

        // Reserve a port, then start the runner only after the gate is trying.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let gate = |path: &str| GateSpec {
            name: "g".into(),
            check: GateCheck::Url(format!("http://{addr}{path}").parse().unwrap()),
            timeout: Duration::from_secs(10),
        };
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            let app = Router::new()
                .route("/pass", post(|| async { StatusCode::OK }))
                .route(
                    "/fail",
                    post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "no tool call") }),
                );
            let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
            axum::serve(listener, app).await.unwrap();
        });

        assert_eq!(run_gate(&gate("/pass"), &[]).await, Ok(()));
        let reason = run_gate(&gate("/fail"), &[]).await.unwrap_err();
        assert!(
            reason.contains("500") && reason.contains("no tool call"),
            "{reason}"
        );
    }

    #[tokio::test]
    async fn a_hung_check_fails_at_its_deadline() {
        let started = Instant::now();
        let gate = GateSpec {
            name: "g".into(),
            check: GateCheck::Exec(vec!["sleep".into(), "30".into()]),
            timeout: Duration::from_secs(1),
        };
        let reason = run_gate(&gate, &[]).await.unwrap_err();
        assert!(reason.contains("no verdict within"), "{reason}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}

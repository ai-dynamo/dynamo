// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Calls one check URL to a verdict. `Err` carries the reason shown on
//! `/health`.

use std::time::Duration;

/// How much of a failed check's response body is kept as its reason.
const REASON_BYTES: usize = 1024;

/// Backoff between attempts while a check runner can't be reached.
const RETRY_INITIAL: Duration = Duration::from_secs(1);
const RETRY_MAX: Duration = Duration::from_secs(10);

/// POST `body` to the check runner. A `200` is a pass and any other response
/// a fail, with the start of the body as the reason. A runner that can't be
/// reached, or drops the connection before answering, is retried with
/// backoff, so a check may run more than once. There is no deadline: the
/// runner bounds its own checks, and one that never answers keeps the worker
/// held.
pub(super) async fn run_gate(url: &reqwest::Url, body: &serde_json::Value) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| error.to_string())?;
    let mut backoff = RETRY_INITIAL;
    loop {
        match client.post(url.clone()).json(body).send().await {
            Ok(response) if response.status() == reqwest::StatusCode::OK => return Ok(()),
            Ok(response) => {
                let status = response.status();
                let body = response.bytes().await.unwrap_or_default();
                let reason = String::from_utf8_lossy(&body[..body.len().min(REASON_BYTES)])
                    .trim()
                    .to_string();
                tracing::warn!(%url, %status, %reason, "Validation check failed");
                return Err(format!("check runner returned {status}: {reason}"));
            }
            Err(error) => {
                tracing::debug!(%url, %error, "Check runner not reachable; retrying");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(RETRY_MAX);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn waits_for_its_runner_and_takes_its_status() {
        use axum::{Router, http::StatusCode, routing::post};

        // Reserve a port, then start the runner only after the gate is trying.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let url = |path: &str| format!("http://{addr}{path}").parse().unwrap();
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

        let body = serde_json::json!({});
        assert_eq!(run_gate(&url("/pass"), &body).await, Ok(()));
        let reason = run_gate(&url("/fail"), &body).await.unwrap_err();
        assert!(
            reason.contains("500") && reason.contains("no tool call"),
            "{reason}"
        );
    }
}

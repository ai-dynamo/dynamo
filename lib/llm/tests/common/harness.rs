// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//! Shared helpers for the Anthropic and Responses replay tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_llm::{http::service::service_v2::HttpService, model_card::ModelDeploymentCard};
use dynamo_runtime::CancellationToken;

use super::replay_engine::ReplayEngine;

pub const MODEL: &str = "harness-model";

/// Absolute path to a harness fixture under `lib/llm/tests/data/replays/harness/`.
pub fn harness_fixture(name: &str) -> PathBuf {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .join("tests")
        .join("data")
        .join("replays")
        .join("harness")
        .join(name)
}

/// A running HTTP service wired to a `ReplayEngine`.
///
/// The service is started on an ephemeral port; `port` is the bound port,
/// `base_url` is `http://127.0.0.1:<port>`.
pub struct HarnessService {
    pub port: u16,
    pub base_url: String,
    pub cancel: CancellationToken,
    pub engine: Arc<ReplayEngine>,
    join: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl HarnessService {
    /// Spin up a service with the given replay fixture, chat + anthropic +
    /// responses endpoints enabled, and wait until `/health` answers.
    pub async fn start(fixture: &str, interval: Duration) -> Self {
        let engine = Arc::new(
            ReplayEngine::from_jsonl(harness_fixture(fixture), interval)
                .expect("failed to load harness fixture"),
        );
        Self::start_with_engine(engine).await
    }

    pub async fn start_with_engine(engine: Arc<ReplayEngine>) -> Self {
        let port = get_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .host(String::from("127.0.0.1"))
            .enable_chat_endpoints(true)
            .enable_cmpl_endpoints(false)
            .enable_anthropic_endpoints(true)
            .build()
            .expect("failed to build HttpService");

        let state = service.state_clone();
        let manager = state.manager();
        let card = ModelDeploymentCard::with_name_only(MODEL);
        manager
            .add_chat_completions_model(MODEL, card.mdcsum(), engine.clone())
            .expect("failed to register harness model");

        let cancel = CancellationToken::new();
        let cancel_task = cancel.clone();
        let join = tokio::spawn(async move { service.run(cancel_task).await });

        wait_for_health(port).await;

        Self {
            port,
            base_url: format!("http://127.0.0.1:{}", port),
            cancel,
            engine,
            join,
        }
    }

    pub async fn shutdown(self) {
        self.cancel.cancel();
        // Best-effort: shutdown is graceful, may take up to ~5s per the
        // server config. We don't block tests on it.
        let _ = tokio::time::timeout(Duration::from_secs(8), self.join).await;
    }
}

async fn get_random_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

async fn wait_for_health(port: u16) {
    let start = Instant::now();
    let timeout = Duration::from_secs(5);
    let url = format!("http://127.0.0.1:{}/health", port);
    loop {
        if reqwest::get(&url).await.is_ok() {
            return;
        }
        if start.elapsed() >= timeout {
            panic!("harness service on port {} failed to become ready", port);
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
}

/// A parsed Server-Sent Event — `event:` name + JSON-decoded `data:` payload.
#[derive(Debug, serde::Serialize)]
pub struct ParsedEvent {
    pub event: String,
    pub data: serde_json::Value,
}

/// Parse a raw SSE stream body into `ParsedEvent`s. Keeps `[DONE]` sentinels
/// (OpenAI-style terminator) as data=null events so they remain visible in
/// snapshots.
pub fn parse_sse(body: &str) -> Vec<ParsedEvent> {
    let mut events = Vec::new();
    for block in body.split("\n\n") {
        let mut event: Option<String> = None;
        let mut data: Option<String> = None;
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                let trimmed = rest.trim().to_string();
                data = Some(match data.take() {
                    Some(existing) => format!("{}\n{}", existing, trimmed),
                    None => trimmed,
                });
            }
        }
        let Some(data) = data else { continue };
        let parsed = if data == "[DONE]" {
            serde_json::Value::String("[DONE]".into())
        } else {
            serde_json::from_str(&data).unwrap_or(serde_json::Value::String(data.clone()))
        };
        events.push(ParsedEvent {
            event: event.unwrap_or_default(),
            data: parsed,
        });
    }
    events
}

/// Recursively normalise volatile fields (IDs, timestamps, signatures) into
/// fixed tokens so `insta` snapshots stay stable across runs.
pub fn redact(mut value: serde_json::Value) -> serde_json::Value {
    redact_in_place(&mut value);
    value
}

fn redact_in_place(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                match k.as_str() {
                    "id" | "request_id" | "message_id" | "response_id" | "msg_id" | "item_id" => {
                        if v.is_string() || v.is_number() {
                            *v = serde_json::Value::String("[ID]".into());
                        }
                    }
                    "created" | "created_at" | "start_time" | "end_time" | "completed_at" => {
                        if v.is_number() {
                            *v = serde_json::Value::from(1_000_000_000u64);
                        }
                    }
                    "signature" => {
                        if v.is_string() {
                            *v = serde_json::Value::String("[SIG]".into());
                        }
                    }
                    "system_fingerprint" => {
                        if !v.is_null() {
                            *v = serde_json::Value::String("[FP]".into());
                        }
                    }
                    _ => redact_in_place(v),
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr.iter_mut() {
                redact_in_place(v);
            }
        }
        _ => {}
    }
}

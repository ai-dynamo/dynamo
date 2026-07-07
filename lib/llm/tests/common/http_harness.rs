// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared HTTP service and SSE helpers for agent-facing protocol tests.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::codec::create_message_stream;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_runtime::CancellationToken;
use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;

use super::ports::bind_random_port;
use super::scripted_chat_engine::{Script, ScriptGate, ScriptedChatEngine};

pub const MODEL: &str = "harness-model";

pub async fn load_agent_fixture(name: &str) -> Result<Script> {
    load_sse_fixture(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/replays/agent-harness")
            .join(name),
    )
    .await
}

pub struct HarnessService {
    pub base_url: String,
    pub client: reqwest::Client,
    pub engine: Arc<ScriptedChatEngine>,
    cancel: CancellationToken,
    join: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl HarnessService {
    pub async fn start(scripts: impl IntoIterator<Item = Script>) -> Self {
        let engine = Arc::new(ScriptedChatEngine::new(scripts));
        Self::start_with_engine(engine).await
    }

    pub async fn start_with_gated_tail(script: Script, split_at: usize) -> (Self, ScriptGate) {
        let (engine, gate) = ScriptedChatEngine::with_gated_tail(script, split_at);
        (Self::start_with_engine(Arc::new(engine)).await, gate)
    }

    async fn start_with_engine(engine: Arc<ScriptedChatEngine>) -> Self {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("failed to build harness HTTP client");
        let (listener, port) = bind_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .host("127.0.0.1")
            .enable_chat_endpoints(true)
            .enable_cmpl_endpoints(false)
            .enable_responses_endpoints(true)
            .enable_anthropic_endpoints(true)
            .build()
            .expect("failed to build harness HTTP service");

        let card = ModelDeploymentCard::with_name_only(MODEL);
        service
            .model_manager()
            .add_chat_completions_model(MODEL, card.mdcsum(), engine.clone())
            .expect("failed to register scripted harness model");

        let cancel = CancellationToken::new();
        let join = service.spawn_with_listener(cancel.clone(), listener).await;
        let base_url = format!("http://127.0.0.1:{port}");
        wait_for_health(&client, &base_url).await;

        Self {
            base_url,
            client,
            engine,
            cancel,
            join: Some(join),
        }
    }

    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        let join = self.join.take().expect("harness join handle is missing");
        tokio::time::timeout(Duration::from_secs(2), join)
            .await
            .expect("harness HTTP service did not stop within two seconds")
            .expect("harness HTTP service task panicked")
            .expect("harness HTTP service returned an error");
    }
}

impl Drop for HarnessService {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

async fn wait_for_health(client: &reqwest::Client, base_url: &str) {
    let url = format!("{base_url}/health");
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if client
                .get(&url)
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("harness HTTP service did not become healthy");
}

pub async fn load_sse_fixture(path: impl AsRef<Path>) -> Result<Script> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read replay fixture {}", path.display()))?;
    validate_terminal_data(&text, &format!("fixture {}", path.display()))?;
    let mut messages = create_message_stream(&text);
    let mut chunks = Vec::new();

    while let Some(message) = messages.next().await {
        let message = message.with_context(|| {
            format!("failed to decode SSE framing in fixture {}", path.display())
        })?;
        match message.data.as_deref() {
            Some("[DONE]") => break,
            Some(_) => chunks.push(message.decode_data::<NvCreateChatCompletionStreamResponse>()?),
            None => {
                return Err(anyhow!(
                    "fixture {} contains an SSE event without data",
                    path.display()
                ));
            }
        }
    }

    if chunks.is_empty() {
        return Err(anyhow!(
            "fixture {} contains no response chunks",
            path.display()
        ));
    }
    Ok(chunks)
}

fn validate_terminal_data(text: &str, description: &str) -> Result<()> {
    let mut seen_done = false;

    for data in text.lines().filter_map(|line| line.strip_prefix("data:")) {
        let data = data.trim();
        if seen_done {
            return Err(anyhow!("{description} contains SSE data after [DONE]"));
        }
        if data == "[DONE]" {
            seen_done = true;
        }
    }

    if !seen_done {
        return Err(anyhow!("{description} does not end with [DONE]"));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct JsonSseEvent {
    pub event: String,
    pub data: Value,
}

#[derive(Default)]
pub struct IncrementalSseParser {
    raw: Vec<u8>,
    parsed_through: usize,
}

impl IncrementalSseParser {
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        self.raw.extend_from_slice(bytes);
        let mut events = Vec::new();

        while let Some((frame_len, separator_len)) =
            find_sse_frame_separator(&self.raw[self.parsed_through..])
        {
            let frame_start = self.parsed_through;
            let frame_end = frame_start + frame_len;
            let frame = std::str::from_utf8(&self.raw[frame_start..frame_end])
                .context("response SSE frame was not UTF-8")?;
            events.push(
                frame
                    .lines()
                    .find_map(|line| line.strip_prefix("event:").map(str::trim))
                    .unwrap_or_default()
                    .to_string(),
            );
            self.parsed_through = frame_end + separator_len;
        }

        Ok(events)
    }

    pub fn into_body(self) -> Result<String> {
        String::from_utf8(self.raw).context("response SSE stream was not UTF-8")
    }
}

fn find_sse_frame_separator(bytes: &[u8]) -> Option<(usize, usize)> {
    for index in 0..bytes.len() {
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") {
            return Some((index, 2));
        }
    }
    None
}

pub async fn parse_json_sse(body: &str) -> Result<Vec<JsonSseEvent>> {
    validate_terminal_data(body, "response SSE stream")?;
    let mut messages = create_message_stream(body);
    let mut events = Vec::new();

    while let Some(message) = messages.next().await {
        let message = message.context("failed to decode response SSE framing")?;
        let data = message
            .data
            .ok_or_else(|| anyhow!("response SSE event is missing data"))?;
        let data = if data == "[DONE]" {
            Value::String(data)
        } else {
            serde_json::from_str(&data).context("response SSE data is not valid JSON")?
        };
        events.push(JsonSseEvent {
            event: message.event.unwrap_or_default(),
            data,
        });
    }
    Ok(events)
}

/// Canonicalize volatile values while preserving equality and distinctness of IDs.
pub fn canonicalize(mut value: Value) -> Value {
    let mut ids = HashMap::new();
    let mut next_id = 1;
    canonicalize_in_place(&mut value, &mut ids, &mut next_id);
    value
}

fn canonicalize_in_place(
    value: &mut Value,
    ids: &mut HashMap<String, String>,
    next_id: &mut usize,
) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                match key.as_str() {
                    "id" if value.as_str().is_some_and(is_service_generated_object_id) => {
                        canonicalize_id(value, ids, next_id);
                    }
                    "item_id" | "message_id" | "request_id" | "response_id" => {
                        canonicalize_id(value, ids, next_id);
                    }
                    "created" | "created_at" | "completed_at" | "start_time" | "end_time"
                        if value.is_number() =>
                    {
                        *value = Value::from(1_000_000_000u64);
                    }
                    _ => canonicalize_in_place(value, ids, next_id),
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                canonicalize_in_place(value, ids, next_id);
            }
        }
        _ => {}
    }
}

fn is_service_generated_object_id(id: &str) -> bool {
    ["msg_", "resp_", "fc_", "req_"]
        .iter()
        .any(|prefix| id.starts_with(prefix))
}

fn canonicalize_id(value: &mut Value, ids: &mut HashMap<String, String>, next_id: &mut usize) {
    if let Some(id) = value.as_str() {
        let canonical = ids.entry(id.to_string()).or_insert_with(|| {
            let canonical = format!("[ID-{}]", *next_id);
            *next_id += 1;
            canonical
        });
        *value = Value::String(canonical.clone());
    }
}

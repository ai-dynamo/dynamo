// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional read-only inspection of the EPP's own routing state.

use std::{sync::Arc, time::Duration};

use axum::{
    Json, Router as HttpRouter,
    body::to_bytes,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest;
use serde::Serialize;
use serde_json::Value;
use tokio::{net::TcpListener, sync::Semaphore, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::epp::Router;

const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

pub(crate) struct Config {
    port: u16,
    timeout: Duration,
    max_inflight: usize,
}

impl Config {
    pub(crate) fn from_env(standalone: bool) -> anyhow::Result<Option<Self>> {
        fn read<T: std::str::FromStr>(key: &str, default: T) -> anyhow::Result<T> {
            match std::env::var(key) {
                Ok(value) => value.parse().map_err(|_| anyhow::anyhow!("invalid {key}")),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(error) => Err(error.into()),
            }
        }
        let port = read("DYN_EPP_PROBE_PORT", 0u16)?;
        if port == 0 {
            return Ok(None);
        }
        anyhow::ensure!(
            !standalone,
            "probe API currently requires DYN_EPP_MODE=dynamo"
        );
        let timeout_ms = read("DYN_EPP_PROBE_TIMEOUT_MS", 1000u64)?;
        let max_inflight = read("DYN_EPP_PROBE_MAX_INFLIGHT", 4usize)?;
        anyhow::ensure!(
            timeout_ms > 0 && (1..=64).contains(&max_inflight),
            "probe timeout must be positive and max inflight must be between 1 and 64"
        );
        Ok(Some(Self {
            port,
            timeout: Duration::from_millis(timeout_ms),
            max_inflight,
        }))
    }
}

pub(crate) struct Error(pub StatusCode, pub &'static str);

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({"error": {"code": self.1}}))).into_response()
    }
}

pub(crate) fn unsupported() -> Error {
    Error(
        StatusCode::UNPROCESSABLE_ENTITY,
        "unsupported_probe_request",
    )
}

pub(crate) fn parse_request(
    body: &[u8],
    model: &str,
) -> Result<NvCreateChatCompletionRequest, Error> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|_| Error(StatusCode::BAD_REQUEST, "invalid_request"))?;
    if value.get("model").and_then(Value::as_str) != Some(model) {
        return Err(Error(StatusCode::BAD_REQUEST, "model_mismatch"));
    }
    let messages = value
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .ok_or(Error(StatusCode::BAD_REQUEST, "invalid_request"))?;
    if value
        .get("n")
        .is_some_and(|n| !n.is_null() && n.as_u64() != Some(1))
        || value.get("prompt").is_some()
        || messages.iter().any(|message| match message.get("content") {
            None | Some(Value::Null) | Some(Value::String(_)) => false,
            Some(Value::Array(parts)) => parts
                .iter()
                .any(|part| part.get("type").and_then(Value::as_str) != Some("text")),
            _ => true,
        })
    {
        return Err(unsupported());
    }
    // The native pick path does not preserve these inputs yet. Do not claim parity.
    if value.get("nvext").is_some_and(|nvext| {
        [
            "token_data",
            "worker_id",
            "prefill_worker_id",
            "decode_worker_id",
            "backend_instance_id",
            "dp_rank",
            "prefill_dp_rank",
            "lora_name",
            "agent_hints",
        ]
        .iter()
        .any(|field| nvext.get(field).is_some_and(|v| !v.is_null()))
    }) {
        return Err(unsupported());
    }
    serde_json::from_value(value).map_err(|_| Error(StatusCode::BAD_REQUEST, "invalid_request"))
}

#[derive(Serialize)]
pub(crate) struct ProbeResponse {
    pub model: String,
    pub epp_instance: String,
    pub sampled_at_unix_ms: u64,
    pub prompt_tokens: usize,
    pub block_size: u32,
    pub candidate: Candidate,
    pub pool: Pool,
}

#[derive(Serialize)]
pub(crate) struct Candidate {
    pub worker_id: String,
    pub dp_rank: u32,
    pub cache: Cache,
    pub load: Load,
}

#[derive(Serialize)]
pub(crate) struct Cache {
    pub estimate_source: &'static str,
    pub gpu_prefix_tokens: Option<u32>,
    pub cpu_prefix_tokens: Option<u32>,
    pub disk_prefix_tokens: Option<u32>,
    pub predicted_gpu_hit_rate: Option<f64>,
    pub predicted_cpu_inclusive_hit_rate: Option<f64>,
    pub predicted_disk_inclusive_hit_rate: Option<f64>,
    pub effective_prefill_tokens: usize,
}

#[derive(Serialize)]
pub(crate) struct Load {
    pub active_prefill_tokens: Option<usize>,
    pub prefill_token_capacity: Option<usize>,
    pub potential_decode_blocks: Option<u64>,
    pub total_kv_blocks: Option<u64>,
}

#[derive(Serialize)]
pub(crate) struct Pool {
    pub pending_requests: usize,
    pub pending_input_tokens: usize,
}

pub(crate) fn hit_rate(cached: Option<u64>, prompt: Option<u64>) -> Option<f64> {
    let (cached, prompt) = (cached?, prompt?);
    (prompt > 0 && cached <= prompt).then(|| cached as f64 / prompt as f64)
}

struct AppState {
    router: Arc<Router>,
    permits: Arc<Semaphore>,
    timeout: Duration,
}

async fn probe(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let Ok(permit) = state.permits.clone().try_acquire_owned() else {
        return Error(StatusCode::TOO_MANY_REQUESTS, "probe_overloaded").into_response();
    };
    let work = async {
        let (parts, body) = request.into_parts();
        if parts.headers.contains_key("content-encoding") {
            return Err(Error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_encoding",
            ));
        }
        if !parts
            .headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"))
            })
        {
            return Err(Error(StatusCode::UNSUPPORTED_MEDIA_TYPE, "expected_json"));
        }
        let body = to_bytes(body, MAX_BODY_BYTES)
            .await
            .map_err(|_| Error(StatusCode::PAYLOAD_TOO_LARGE, "request_too_large"))?;
        state
            .router
            .probe(body, parts.headers, Arc::new(permit))
            .await
    };
    match tokio::time::timeout(state.timeout, work).await {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(_) => Error(StatusCode::GATEWAY_TIMEOUT, "probe_timeout").into_response(),
    }
}

pub(crate) async fn start(
    config: Option<Config>,
    router: Arc<Router>,
    shutdown: CancellationToken,
) -> anyhow::Result<Option<JoinHandle<()>>> {
    let Some(config) = config else {
        return Ok(None);
    };
    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    let state = Arc::new(AppState {
        router,
        permits: Arc::new(Semaphore::new(config.max_inflight)),
        timeout: config.timeout,
    });
    let app = HttpRouter::new()
        .route("/v1/routing/probe", post(probe))
        .with_state(state);
    tracing::info!(port = config.port, "Serving internal routing probe API");
    Ok(Some(tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.clone().cancelled_owned())
            .await
        {
            tracing::error!(%error, "Probe listener failed");
            shutdown.cancel();
        }
    })))
}

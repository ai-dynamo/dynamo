// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo tokenizer HTTP sidecar.
//!
//! Runs the SAME `OpenAIPreprocessor` Dynamo uses for its own workers (chat
//! template + tokenizer from the model's `ModelDeploymentCard`), exposed as a
//! small HTTP `POST /tokenize` endpoint. The external EPP points
//! `DYN_EPP_TOKENIZE_URL` at this (loopback) sidecar to get exact Dynamo-native
//! token IDs for KV-prefix routing — no GPU, no DistributedRuntime, no
//! discovery: the model card (tokenizer + chat template) is built offline from
//! the model name via the HuggingFace hub.
//!
//! Enabled by `DYN_TOKENIZER_ONLY=true`; `DYN_MODEL_NAME` selects the model,
//! `DYN_TOKENIZER_PORT` (default 8788) / `DYN_TOKENIZER_HOST` (default
//! 127.0.0.1) the bind address.

use std::sync::Arc;

use anyhow::Result;
use axum::{
    Json, Router as AxumRouter,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde_json::{Value, json};

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest;

pub async fn run() -> Result<()> {
    let model_name =
        std::env::var("DYN_MODEL_NAME").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".to_string());
    let host = std::env::var("DYN_TOKENIZER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("DYN_TOKENIZER_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8788);

    tracing::info!(
        %model_name, %host, port,
        "Dynamo tokenizer server: building OpenAIPreprocessor offline from model name"
    );

    // Offline model-card bootstrap (no discovery / NATS / GPU): download the
    // tokenizer + config from HF, load the card from disk, resolve its
    // metadata, then build Dynamo's own preprocessor.
    let model_path = dynamo_llm::hub::from_hf(&model_name, /* ignore_weights = */ true).await?;
    let mut card = ModelDeploymentCard::load_from_disk(&model_path, None)?;
    card.download_config(None).await?;
    let preprocessor = OpenAIPreprocessor::new(card)?;
    tracing::info!("Dynamo tokenizer server: preprocessor ready");

    let app = AxumRouter::new()
        .route("/tokenize", post(tokenize_handler))
        .route("/health", get(|| async { "ok" }))
        .with_state(preprocessor);

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "Dynamo tokenizer server listening (POST /tokenize)");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn tokenize_handler(
    State(preprocessor): State<Arc<OpenAIPreprocessor>>,
    body: String,
) -> impl IntoResponse {
    match tokenize_request(&preprocessor, &body) {
        Ok(tokens) => (
            StatusCode::OK,
            Json(json!({ "count": tokens.len(), "tokens": tokens })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// Tokenize an OpenAI request: chat `messages` go through the chat template
/// then the tokenizer (matching what a Dynamo worker would cache); a raw
/// `prompt` string is tokenized directly.
fn tokenize_request(preprocessor: &OpenAIPreprocessor, body: &str) -> Result<Vec<u32>> {
    let v: Value = serde_json::from_str(body)?;
    if v.get("messages").map(|m| !m.is_null()).unwrap_or(false) {
        let request: NvCreateChatCompletionRequest = serde_json::from_str(body)?;
        let prompt = preprocessor.apply_template(&request)?.unwrap_or_default();
        Ok(preprocessor.tokenize(&prompt)?.token_ids().to_vec())
    } else if let Some(prompt) = v.get("prompt").and_then(|p| p.as_str()) {
        Ok(preprocessor.tokenize(prompt)?.token_ids().to_vec())
    } else {
        anyhow::bail!("request has neither 'messages' nor a string 'prompt'")
    }
}

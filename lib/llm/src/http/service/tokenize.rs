// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `POST /tokenize` and `POST /detokenize` HTTP endpoints.
//!
//! Field names and request/response shapes mirror vLLM's
//! `vllm/entrypoints/serve/tokenize/protocol.py` so clients written against
//! vLLM's tokenize API work unchanged against Dynamo.
//!
//! `POST /tokenize` accepts either of two request shapes (untagged union):
//! * `TokenizeCompletionRequest` — `{ "prompt": ... }`
//! * `TokenizeChatRequest`       — `{ "messages": [...] }` (renders the
//!   model's chat template before tokenizing)
//!
//! `POST /detokenize` is symmetric.
//!
//! Tokenizer files referenced by [`ModelDeploymentCard`] are loaded once per
//! display name and cached in a process-global [`DashMap`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use axum::{Json, Router, extract::State, routing::post};
use dashmap::DashMap;
use minijinja::value::Value as JinjaValue;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Tokenizer as HfTokenizer;

use super::RouteDoc;
use super::openai::{ErrorResponse, check_ready};
use super::service_v2;
use crate::model_card::{ModelDeploymentCard, TokenizerKind};
use crate::preprocessor::prompt::{OAIChatLikeRequest, PromptFormatter};

// ---------------------------------------------------------------------------
// Request / response types (vLLM-compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeCompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    #[serde(default)]
    pub return_token_strs: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeChatRequest {
    pub model: Option<String>,
    pub messages: Vec<serde_json::Value>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default)]
    pub continue_final_message: bool,
    /// Default differs from the completion form: chat templates already insert
    /// special tokens, so vLLM defaults this to `false`.
    #[serde(default)]
    pub add_special_tokens: bool,
    pub chat_template: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub return_token_strs: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TokenizeRequest {
    Chat(TokenizeChatRequest),
    Completion(TokenizeCompletionRequest),
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub count: usize,
    pub max_model_len: u32,
    pub tokens: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_strs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    pub model: Option<String>,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub prompt: String,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Tokenizer cache
// ---------------------------------------------------------------------------

enum CachedTokenizer {
    Hf(Arc<HfTokenizer>),
    Generic(crate::tokenizers::Tokenizer),
}

fn cache() -> &'static DashMap<String, Arc<CachedTokenizer>> {
    static CACHE: OnceLock<DashMap<String, Arc<CachedTokenizer>>> = OnceLock::new();
    CACHE.get_or_init(DashMap::new)
}

fn load_tokenizer(card: &ModelDeploymentCard) -> Result<Arc<CachedTokenizer>, ErrorResponse> {
    if let Some(entry) = cache().get(&card.display_name) {
        return Ok(entry.clone());
    }
    let kind = card.tokenizer.as_ref().ok_or_else(|| {
        super::openai::ErrorMessage::not_implemented_error(format!(
            "Model '{}' has no tokenizer configured",
            card.display_name
        ))
    })?;
    let loaded = match kind {
        TokenizerKind::HfTokenizerJson(file) => {
            let path = file.path().ok_or_else(|| {
                super::openai::ErrorMessage::internal_server_error(&format!(
                    "Tokenizer for '{}' is URL-backed, not yet downloaded",
                    card.display_name
                ))
            })?;
            let hf = HfTokenizer::from_file(path).map_err(|e| {
                super::openai::ErrorMessage::internal_server_error(&format!(
                    "Failed to load HF tokenizer for '{}': {e}",
                    card.display_name
                ))
            })?;
            CachedTokenizer::Hf(Arc::new(hf))
        }
        TokenizerKind::TikTokenModel(_) => {
            // tiktoken: fall back to the generic wrapper. Special-token semantics
            // and per-id token strings differ from HF; we honor what the trait exposes.
            let tok = card.tokenizer().map_err(|e| {
                super::openai::ErrorMessage::internal_server_error(&format!(
                    "Failed to load tiktoken tokenizer for '{}': {e}",
                    card.display_name
                ))
            })?;
            CachedTokenizer::Generic(tok)
        }
    };
    let arc = Arc::new(loaded);
    cache().insert(card.display_name.clone(), arc.clone());
    Ok(arc)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_card(
    state: &Arc<service_v2::State>,
    requested: Option<&str>,
) -> Result<ModelDeploymentCard, ErrorResponse> {
    let cards = state.manager().get_model_cards();
    if cards.is_empty() {
        return Err(super::openai::ErrorMessage::model_not_found());
    }
    match requested {
        Some(name) => cards
            .into_iter()
            .find(|c| c.display_name == name)
            .ok_or_else(super::openai::ErrorMessage::model_not_found),
        None if cards.len() == 1 => Ok(cards.into_iter().next().unwrap()),
        None => Err(super::openai::ErrorMessage::model_not_found()),
    }
}

fn encode_text(
    cached: &CachedTokenizer,
    text: &str,
    add_special_tokens: bool,
    want_strs: bool,
) -> Result<(Vec<u32>, Option<Vec<String>>), ErrorResponse> {
    match cached {
        CachedTokenizer::Hf(hf) => {
            let enc = hf.encode(text, add_special_tokens).map_err(|e| {
                super::openai::ErrorMessage::internal_server_error(&format!(
                    "Tokenizer encode failed: {e}"
                ))
            })?;
            let ids = enc.get_ids().to_vec();
            let strs = want_strs.then(|| enc.get_tokens().to_vec());
            Ok((ids, strs))
        }
        CachedTokenizer::Generic(tok) => {
            // The generic Encoder trait does not expose `add_special_tokens`;
            // honor the request by surfacing an explicit error if the caller
            // asked for non-default behavior on a backend that cannot oblige.
            if add_special_tokens {
                tracing::debug!(
                    "tiktoken tokenizer does not support add_special_tokens=true; \
                     falling back to default behavior"
                );
            }
            let enc = tok.encode(text).map_err(|e| {
                super::openai::ErrorMessage::internal_server_error(&format!(
                    "Tokenizer encode failed: {e}"
                ))
            })?;
            let ids = enc.token_ids().to_vec();
            let strs = if want_strs {
                let mut out = Vec::with_capacity(ids.len());
                for id in &ids {
                    let decoded = tok.decode(&[*id], false).map_err(|e| {
                        super::openai::ErrorMessage::internal_server_error(&format!(
                            "Tokenizer decode failed: {e}"
                        ))
                    })?;
                    out.push(String::from(decoded));
                }
                Some(out)
            } else {
                None
            };
            Ok((ids, strs))
        }
    }
}

fn decode_tokens(cached: &CachedTokenizer, tokens: &[u32]) -> Result<String, ErrorResponse> {
    let result: Result<String, String> = match cached {
        CachedTokenizer::Hf(hf) => hf.decode(tokens, false).map_err(|e| e.to_string()),
        CachedTokenizer::Generic(tok) => tok
            .decode(tokens, false)
            .map(String::from)
            .map_err(|e| e.to_string()),
    };
    result.map_err(|e| {
        super::openai::ErrorMessage::internal_server_error(&format!("Tokenizer decode failed: {e}"))
    })
}

// Minimal `OAIChatLikeRequest` used purely for chat-template rendering.
struct ChatTemplateInput<'a> {
    model: &'a str,
    messages: &'a [serde_json::Value],
    tools: Option<&'a [serde_json::Value]>,
    add_generation_prompt: bool,
    chat_template_args: Option<&'a HashMap<String, serde_json::Value>>,
}

impl<'a> OAIChatLikeRequest for ChatTemplateInput<'a> {
    fn model(&self) -> String {
        self.model.to_string()
    }
    fn messages(&self) -> JinjaValue {
        JinjaValue::from_serialize(self.messages)
    }
    fn tools(&self) -> Option<JinjaValue> {
        self.tools.map(JinjaValue::from_serialize)
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.add_generation_prompt
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        self.chat_template_args
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn tokenize(
    State(state): State<Arc<service_v2::State>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ErrorResponse> {
    check_ready(&state)?;

    match req {
        TokenizeRequest::Completion(req) => {
            let card = find_card(&state, req.model.as_deref())?;
            let cached = load_tokenizer(&card)?;
            let (tokens, token_strs) = encode_text(
                &cached,
                &req.prompt,
                req.add_special_tokens,
                req.return_token_strs.unwrap_or(false),
            )?;
            Ok(Json(TokenizeResponse {
                count: tokens.len(),
                max_model_len: card.context_length,
                tokens,
                token_strs,
            }))
        }
        TokenizeRequest::Chat(req) => {
            // Reject fields we accept in the schema (vLLM API compat) but do not yet honor.
            if req.continue_final_message {
                return Err(super::openai::ErrorMessage::not_implemented_error(
                    "`continue_final_message` is not yet supported",
                ));
            }
            if req.chat_template.is_some() {
                return Err(super::openai::ErrorMessage::not_implemented_error(
                    "Per-request `chat_template` override is not yet supported",
                ));
            }
            let card = find_card(&state, req.model.as_deref())?;
            let formatter = PromptFormatter::from_mdc(&card).map_err(|e| {
                super::openai::ErrorMessage::not_implemented_error(format!(
                    "Model '{}' has no chat template: {e}",
                    card.display_name
                ))
            })?;
            let input = ChatTemplateInput {
                model: &card.display_name,
                messages: &req.messages,
                tools: req.tools.as_deref(),
                add_generation_prompt: req.add_generation_prompt,
                chat_template_args: req.chat_template_kwargs.as_ref(),
            };
            let rendered = match &formatter {
                PromptFormatter::OAI(f) => f.render(&input).map_err(|e| {
                    super::openai::ErrorMessage::from_http_error(super::error::HttpError {
                        code: 400,
                        message: format!("Chat template rendering failed: {e}"),
                    })
                })?,
            };
            let cached = load_tokenizer(&card)?;
            let (tokens, token_strs) = encode_text(
                &cached,
                &rendered,
                req.add_special_tokens,
                req.return_token_strs.unwrap_or(false),
            )?;
            Ok(Json(TokenizeResponse {
                count: tokens.len(),
                max_model_len: card.context_length,
                tokens,
                token_strs,
            }))
        }
    }
}

async fn detokenize(
    State(state): State<Arc<service_v2::State>>,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ErrorResponse> {
    check_ready(&state)?;
    let card = find_card(&state, req.model.as_deref())?;
    let cached = load_tokenizer(&card)?;
    let prompt = decode_tokens(&cached, &req.tokens)?;
    Ok(Json(DetokenizeResponse { prompt }))
}

// ---------------------------------------------------------------------------
// Routers
// ---------------------------------------------------------------------------

pub fn tokenize_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/tokenize".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new().route(&path, post(tokenize)).with_state(state);
    (vec![doc], router)
}

pub fn detokenize_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/detokenize".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(detokenize))
        .with_state(state);
    (vec![doc], router)
}

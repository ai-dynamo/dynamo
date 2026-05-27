// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful `/v1/responses` proxy.
//!
//! Dynamo's in-process Responses endpoint is intentionally stateless today:
//! it rejects `previous_response_id` and expects callers to send the complete
//! Responses input chain. This module provides a small external Axum service
//! that accepts the stateful shape, expands it into that stateless request, and
//! forwards it to a regular Dynamo `/v1/responses` endpoint.

use std::collections::{HashMap, HashSet};
use std::env;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "stateful-responses-redb")]
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use axum::body::{Body, Bytes};
use axum::extract::{Path as AxumPath, State};
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE, HOST};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

const DEFAULT_BIND_HOST: &str = "0.0.0.0";
const DEFAULT_BIND_PORT: u16 = 8081;
const DEFAULT_ROUTE_PATH: &str = "/v1/responses";
const DEFAULT_UPSTREAM_RESPONSES_URL: &str = "http://127.0.0.1:8000/v1/responses";
const DEFAULT_TTL_SECS: u64 = 30 * 24 * 60 * 60;
const DEFAULT_TTL_SWEEP_SECS: u64 = 60 * 60;
#[cfg(feature = "stateful-responses-tikv")]
const DEFAULT_TIKV_PREFIX: &str = "dynamo/stateful-responses/";
#[cfg(feature = "stateful-responses-tikv")]
const TIKV_PURGE_SCAN_LIMIT: u32 = 1024;

const BIND_HOST_ENV: &str = "DYN_STATEFUL_RESPONSES_HOST";
const BIND_PORT_ENV: &str = "DYN_STATEFUL_RESPONSES_PORT";
const ROUTE_PATH_ENV: &str = "DYN_STATEFUL_RESPONSES_PATH";
const UPSTREAM_RESPONSES_URL_ENV: &str = "DYN_STATEFUL_RESPONSES_UPSTREAM_URL";
const REQUEST_TIMEOUT_SECS_ENV: &str = "DYN_STATEFUL_RESPONSES_REQUEST_TIMEOUT_SECS";
const COLLECT_TOKEN_IDS_ENV: &str = "DYN_STATEFUL_RESPONSES_COLLECT_TOKEN_IDS";
const DEFAULT_TTL_SECS_ENV: &str = "DYN_STATEFUL_RESPONSES_DEFAULT_TTL_SECS";
const TTL_SWEEP_SECS_ENV: &str = "DYN_STATEFUL_RESPONSES_TTL_SWEEP_SECS";
const STORE_URL_ENV: &str = "DYN_STATEFUL_RESPONSES_STORE_URL";

/// Proxy runtime configuration.
#[derive(Clone, Debug)]
pub struct StatefulResponsesConfig {
    pub listen_addr: SocketAddr,
    pub route_path: String,
    pub upstream_responses_url: String,
    pub request_timeout: Duration,
    pub default_ttl: Option<Duration>,
    pub ttl_sweep_interval: Option<Duration>,
    /// When true, the proxy asks Dynamo to return `nvext.completion_token_ids`.
    pub collect_token_ids: bool,
}

impl StatefulResponsesConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let host = env::var(BIND_HOST_ENV).unwrap_or_else(|_| DEFAULT_BIND_HOST.to_string());
        let port = env::var(BIND_PORT_ENV)
            .ok()
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(DEFAULT_BIND_PORT);
        let listen_addr = format!("{host}:{port}").parse()?;

        let route_path = normalize_route_path(
            &env::var(ROUTE_PATH_ENV).unwrap_or_else(|_| DEFAULT_ROUTE_PATH.into()),
        );
        let upstream_responses_url = env::var(UPSTREAM_RESPONSES_URL_ENV)
            .unwrap_or_else(|_| DEFAULT_UPSTREAM_RESPONSES_URL.to_string());
        let request_timeout = Duration::from_secs(
            env::var(REQUEST_TIMEOUT_SECS_ENV)
                .ok()
                .map(|value| value.parse())
                .transpose()?
                .unwrap_or(300),
        );
        let collect_token_ids = env::var(COLLECT_TOKEN_IDS_ENV)
            .map(|value| !is_falsey(&value))
            .unwrap_or(true);
        let default_ttl = duration_secs_from_env(
            DEFAULT_TTL_SECS_ENV,
            Some(Duration::from_secs(DEFAULT_TTL_SECS)),
        )?;
        let ttl_sweep_interval = duration_secs_from_env(
            TTL_SWEEP_SECS_ENV,
            Some(Duration::from_secs(DEFAULT_TTL_SWEEP_SECS)),
        )?;

        Ok(Self {
            listen_addr,
            route_path,
            upstream_responses_url,
            request_timeout,
            default_ttl,
            ttl_sweep_interval,
            collect_token_ids,
        })
    }
}

fn normalize_route_path(path: &str) -> String {
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

fn is_falsey(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn duration_secs_from_env(
    name: &str,
    default: Option<Duration>,
) -> anyhow::Result<Option<Duration>> {
    let Ok(value) = env::var(name) else {
        return Ok(default);
    };

    parse_optional_duration_secs(name, &value)
}

fn parse_optional_duration_secs(name: &str, value: &str) -> anyhow::Result<Option<Duration>> {
    let value = value.trim();
    if value.is_empty()
        || is_falsey(value)
        || matches!(value.to_ascii_lowercase().as_str(), "none" | "never")
    {
        return Ok(None);
    }

    let secs = value
        .parse::<u64>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))?;
    Ok(Some(Duration::from_secs(secs)))
}

#[derive(Clone)]
struct AppState {
    client: Client,
    config: StatefulResponsesConfig,
    store: Arc<dyn ResponseContextStore>,
}

/// Create a router that hosts the stateful Responses endpoint.
pub fn router(
    store: Arc<dyn ResponseContextStore>,
    config: StatefulResponsesConfig,
) -> anyhow::Result<Router> {
    let client = Client::builder()
        .no_proxy()
        .timeout(config.request_timeout)
        .build()?;
    let route_path = config.route_path.clone();
    let delete_path = format!("{}/{{response_id}}", route_path.trim_end_matches('/'));
    let state = Arc::new(AppState {
        client,
        config,
        store,
    });

    spawn_ttl_sweeper(&state);

    Ok(Router::new()
        .route(&route_path, post(handle_responses))
        .route(&delete_path, delete(delete_response))
        .route("/health", get(health))
        .with_state(state))
}

fn spawn_ttl_sweeper(state: &Arc<AppState>) {
    let Some(_) = state.config.default_ttl else {
        return;
    };
    let Some(interval) = state.config.ttl_sweep_interval else {
        return;
    };

    let store = state.store.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tick.tick().await;
            if let Err(err) = store.purge_expired(unix_timestamp()).await {
                tracing::warn!(%err, "failed to purge expired stateful Responses contexts");
            }
        }
    });
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

async fn delete_response(
    State(state): State<Arc<AppState>>,
    AxumPath(response_id): AxumPath<String>,
) -> Response {
    match state.store.delete(&response_id).await {
        Ok(deleted) => (
            StatusCode::OK,
            Json(json!({
                "id": response_id,
                "object": "response.deleted",
                "deleted": deleted
            })),
        )
            .into_response(),
        Err(err) => ProxyError::store(err).into_response(),
    }
}

async fn handle_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_responses_inner(state, headers, body).await {
        Ok(response) => response,
        Err(err) => err.into_response(),
    }
}

async fn handle_responses_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    let original_body: Value = serde_json::from_slice(&body)
        .map_err(|err| ProxyError::bad_request(format!("invalid Responses request JSON: {err}")))?;

    let previous_response_id_key = previous_response_id(&original_body);
    let previous_context = match previous_response_id_key.as_deref() {
        Some(key) => state.store.get(key).await.map_err(ProxyError::store)?,
        None => None,
    };

    let expanded = expand_request(
        original_body,
        previous_context.as_ref(),
        state.config.collect_token_ids,
    )
    .map_err(ProxyError::bad_request)?;

    let request_wants_store = request_wants_store(&expanded.body);
    let upstream = forward_request(&state, &headers, serde_json::to_vec(&expanded.body)?).await?;
    let status = upstream.status();
    let response_headers = upstream.headers().clone();
    let is_sse = is_event_stream(&response_headers) || request_streams(&expanded.body);
    let default_ttl = state.config.default_ttl;

    if is_sse {
        Ok(stream_response(
            state,
            status,
            response_headers,
            upstream,
            expanded,
            request_wants_store,
            default_ttl,
        ))
    } else {
        let response_body = upstream.bytes().await.map_err(ProxyError::upstream)?;
        if status.is_success() && request_wants_store {
            if let Err(err) = store_non_streaming_response(
                &state.store,
                &expanded.input_items,
                &response_body,
                expires_at_from_ttl(default_ttl),
            )
            .await
            {
                tracing::warn!(%err, "failed to persist stateful Responses context");
            }
        }

        Ok(bytes_response(status, response_headers, response_body))
    }
}

async fn forward_request(
    state: &AppState,
    headers: &HeaderMap,
    body: Vec<u8>,
) -> Result<reqwest::Response, ProxyError> {
    let mut request = state
        .client
        .request(Method::POST, &state.config.upstream_responses_url)
        .body(body);

    for (name, value) in headers {
        if should_forward_request_header(name) {
            request = request.header(name, value);
        }
    }
    if !headers.contains_key(CONTENT_TYPE) {
        request = request.header(CONTENT_TYPE, "application/json");
    }

    request.send().await.map_err(ProxyError::upstream)
}

fn stream_response(
    state: Arc<AppState>,
    status: StatusCode,
    headers: HeaderMap,
    upstream: reqwest::Response,
    expanded: ExpandedRequest,
    request_wants_store: bool,
    default_ttl: Option<Duration>,
) -> Response {
    let mut stream = upstream.bytes_stream();
    let store = state.store.clone();
    let mut collected = Vec::new();

    let body_stream = async_stream::stream! {
        while let Some(next) = stream.next().await {
            match next {
                Ok(chunk) => {
                    collected.extend_from_slice(&chunk);
                    yield Ok::<Bytes, axum::Error>(chunk);
                }
                Err(err) => {
                    yield Err(axum::Error::new(err));
                    return;
                }
            }
        }

        if status.is_success() && request_wants_store {
            if let Err(err) = store_streaming_response(
                &store,
                &expanded.input_items,
                &collected,
                expires_at_from_ttl(default_ttl),
            )
            .await
            {
                tracing::warn!(%err, "failed to persist streamed stateful Responses context");
            }
        }
    };

    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = status;
    copy_response_headers(&headers, response.headers_mut(), false);
    response
}

fn bytes_response(status: StatusCode, headers: HeaderMap, body: Bytes) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    copy_response_headers(&headers, response.headers_mut(), true);
    response
}

fn should_forward_request_header(name: &HeaderName) -> bool {
    !matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    ) && name != HOST
        && name != CONTENT_LENGTH
}

fn copy_response_headers(
    source: &HeaderMap,
    destination: &mut HeaderMap,
    include_content_length: bool,
) {
    for (name, value) in source {
        if should_forward_response_header(name, include_content_length) {
            destination.append(name, value.clone());
        }
    }
}

fn should_forward_response_header(name: &HeaderName, include_content_length: bool) -> bool {
    if !include_content_length && name == CONTENT_LENGTH {
        return false;
    }

    !matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
}

fn request_streams(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

#[derive(Debug)]
struct ExpandedRequest {
    body: Value,
    input_items: Vec<Value>,
}

fn expand_request(
    mut body: Value,
    previous_context: Option<&StoredResponseContext>,
    collect_token_ids: bool,
) -> Result<ExpandedRequest, String> {
    let obj = body
        .as_object_mut()
        .ok_or_else(|| "Responses request must be a JSON object".to_string())?;
    obj.remove("previous_response_id");

    let current_items = input_as_items(obj.get("input"))?;
    let mut input_items = Vec::new();
    if let Some(context) = previous_context {
        input_items.extend(context.input_items.iter().cloned());
    }
    input_items.extend(current_items);

    obj.insert("input".to_string(), Value::Array(input_items.clone()));

    remove_nvext_field(obj, "token_data");
    if collect_token_ids {
        ensure_nvext_extra_fields(obj, ["completion_token_ids"]);
    }

    Ok(ExpandedRequest { body, input_items })
}

fn input_as_items(input: Option<&Value>) -> Result<Vec<Value>, String> {
    match input {
        Some(Value::Array(items)) => Ok(items.clone()),
        Some(Value::String(text)) => Ok(vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": text }]
        })]),
        Some(_) => Err("Responses request `input` must be a string or array".to_string()),
        None => Ok(Vec::new()),
    }
}

fn remove_nvext_field(obj: &mut serde_json::Map<String, Value>, field: &str) {
    if let Some(nvext) = obj.get_mut("nvext")
        && let Some(nvext_obj) = nvext.as_object_mut()
    {
        nvext_obj.remove(field);
    }
}

fn ensure_nvext_extra_fields<const N: usize>(
    obj: &mut serde_json::Map<String, Value>,
    fields: [&str; N],
) {
    let nvext = obj.entry("nvext").or_insert_with(|| json!({}));
    if !nvext.is_object() {
        *nvext = json!({});
    }
    let Some(nvext_obj) = nvext.as_object_mut() else {
        return;
    };

    let existing = nvext_obj
        .entry("extra_fields")
        .or_insert_with(|| Value::Array(Vec::new()));
    if !existing.is_array() {
        *existing = Value::Array(Vec::new());
    }

    let Some(extra_fields) = existing.as_array_mut() else {
        return;
    };
    let mut seen = extra_fields
        .iter()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    for field in fields {
        if seen.insert(field.to_string()) {
            extra_fields.push(Value::String(field.to_string()));
        }
    }
}

fn previous_response_id(body: &Value) -> Option<String> {
    body.get("previous_response_id")
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn request_wants_store(body: &Value) -> bool {
    body.get("store").and_then(Value::as_bool).unwrap_or(true)
}

fn expires_at_from_ttl(ttl: Option<Duration>) -> Option<i64> {
    let ttl = ttl?;
    let ttl_secs = match i64::try_from(ttl.as_secs()) {
        Ok(ttl_secs) => ttl_secs,
        Err(_) => i64::MAX,
    };
    Some(unix_timestamp().saturating_add(ttl_secs))
}

fn unix_timestamp() -> i64 {
    let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return 0;
    };

    match i64::try_from(duration.as_secs()) {
        Ok(secs) => secs,
        Err(_) => i64::MAX,
    }
}

fn is_expired(expires_at: Option<i64>, now: i64) -> bool {
    expires_at.is_some_and(|expires_at| expires_at <= now)
}

async fn store_non_streaming_response(
    store: &Arc<dyn ResponseContextStore>,
    expanded_input_items: &[Value],
    response_body: &[u8],
    expires_at: Option<i64>,
) -> Result<(), StoreResponseError> {
    let response: Value = serde_json::from_slice(response_body)?;
    persist_context_for_response(store, expanded_input_items, &response, expires_at).await
}

async fn store_streaming_response(
    store: &Arc<dyn ResponseContextStore>,
    expanded_input_items: &[Value],
    response_body: &[u8],
    expires_at: Option<i64>,
) -> Result<(), StoreResponseError> {
    let Some(response) = completed_response_from_sse(response_body)? else {
        return Ok(());
    };
    persist_context_for_response(store, expanded_input_items, &response, expires_at).await
}

async fn persist_context_for_response(
    store: &Arc<dyn ResponseContextStore>,
    expanded_input_items: &[Value],
    response: &Value,
    expires_at: Option<i64>,
) -> Result<(), StoreResponseError> {
    let Some(response_id) = response.get("id").and_then(Value::as_str) else {
        return Ok(());
    };

    let mut input_items = expanded_input_items.to_vec();
    if let Some(output) = response.get("output").and_then(Value::as_array) {
        input_items.extend(output.iter().cloned());
    }

    let context = StoredResponseContext { input_items };

    store.put(response_id, &context, expires_at).await?;
    Ok(())
}

fn completed_response_from_sse(response_body: &[u8]) -> Result<Option<Value>, serde_json::Error> {
    let Ok(body) = std::str::from_utf8(response_body) else {
        return Ok(None);
    };
    let normalized = body.replace("\r\n", "\n");

    let mut completed = None;
    for event in normalized.split("\n\n") {
        let data = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        let value: Value = serde_json::from_str(&data)?;
        if value.get("type").and_then(Value::as_str) == Some("response.completed")
            && let Some(response) = value.get("response")
        {
            completed = Some(response.clone());
        }
    }

    Ok(completed)
}

/// Stored state for a Responses chain. `input_items` is the OpenResponses item
/// chain that can be sent directly as a stateless `input` array.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StoredResponseContext {
    pub input_items: Vec<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredResponseRecord {
    context: StoredResponseContext,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

#[cfg(any(
    feature = "stateful-responses-redb",
    feature = "stateful-responses-tikv"
))]
fn decode_record(bytes: &[u8]) -> Result<StoredResponseRecord, StoreError> {
    match serde_json::from_slice(bytes) {
        Ok(record) => Ok(record),
        Err(_) => Ok(StoredResponseRecord {
            context: serde_json::from_slice(bytes)?,
            expires_at: None,
        }),
    }
}

#[async_trait]
pub trait ResponseContextStore: Send + Sync + 'static {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError>;
    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError>;
    async fn delete(&self, key: &str) -> Result<bool, StoreError>;
    async fn purge_expired(&self, now: i64) -> Result<usize, StoreError>;
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("failed to serialize Responses context: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("state store lock is poisoned")]
    LockPoisoned,
    #[cfg(feature = "stateful-responses-redb")]
    #[error("redb state store error: {0}")]
    Redb(#[from] redb::Error),
    #[cfg(feature = "stateful-responses-tikv")]
    #[error("TiKV state store error: {0}")]
    Tikv(#[from] tikv_client::Error),
}

#[derive(Default)]
pub struct MemoryResponseContextStore {
    contexts: RwLock<HashMap<String, StoredResponseRecord>>,
}

impl MemoryResponseContextStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ResponseContextStore for MemoryResponseContextStore {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError> {
        let record = self
            .contexts
            .read()
            .map_err(|_| StoreError::LockPoisoned)?
            .get(key)
            .cloned();
        let Some(record) = record else {
            return Ok(None);
        };
        if is_expired(record.expires_at, unix_timestamp()) {
            self.delete(key).await?;
            return Ok(None);
        }
        Ok(Some(record.context))
    }

    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError> {
        self.contexts
            .write()
            .map_err(|_| StoreError::LockPoisoned)?
            .insert(
                key.to_string(),
                StoredResponseRecord {
                    context: context.clone(),
                    expires_at,
                },
            );
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        Ok(self
            .contexts
            .write()
            .map_err(|_| StoreError::LockPoisoned)?
            .remove(key)
            .is_some())
    }

    async fn purge_expired(&self, now: i64) -> Result<usize, StoreError> {
        let mut contexts = self
            .contexts
            .write()
            .map_err(|_| StoreError::LockPoisoned)?;
        let before = contexts.len();
        contexts.retain(|_, record| !is_expired(record.expires_at, now));
        Ok(before - contexts.len())
    }
}

#[cfg(feature = "stateful-responses-redb")]
const REDB_CONTEXTS: redb::TableDefinition<&str, &[u8]> =
    redb::TableDefinition::new("response_contexts");

#[cfg(feature = "stateful-responses-redb")]
pub struct RedbResponseContextStore {
    db: redb::Database,
}

#[cfg(feature = "stateful-responses-redb")]
impl RedbResponseContextStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let db = if path.as_ref().exists() {
            redb::Database::open(path).map_err(redb::Error::from)?
        } else {
            redb::Database::create(path).map_err(redb::Error::from)?
        };

        let write_txn = db.begin_write().map_err(redb::Error::from)?;
        {
            write_txn
                .open_table(REDB_CONTEXTS)
                .map_err(redb::Error::from)?;
        }
        write_txn.commit().map_err(redb::Error::from)?;

        Ok(Self { db })
    }
}

#[cfg(feature = "stateful-responses-redb")]
#[async_trait]
impl ResponseContextStore for RedbResponseContextStore {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError> {
        use redb::ReadableDatabase;

        let read_txn = self.db.begin_read().map_err(redb::Error::from)?;
        let table = read_txn
            .open_table(REDB_CONTEXTS)
            .map_err(redb::Error::from)?;
        let record = table
            .get(key)
            .map_err(redb::Error::from)?
            .map(|bytes| decode_record(bytes.value()))
            .transpose()?;
        drop(table);
        drop(read_txn);

        let Some(record) = record else {
            return Ok(None);
        };
        if is_expired(record.expires_at, unix_timestamp()) {
            self.delete(key).await?;
            return Ok(None);
        }
        Ok(Some(record.context))
    }

    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError> {
        let value = serde_json::to_vec(&StoredResponseRecord {
            context: context.clone(),
            expires_at,
        })?;
        let write_txn = self.db.begin_write().map_err(redb::Error::from)?;
        {
            let mut table = write_txn
                .open_table(REDB_CONTEXTS)
                .map_err(redb::Error::from)?;
            table
                .insert(key, value.as_slice())
                .map_err(redb::Error::from)?;
        }
        write_txn.commit().map_err(redb::Error::from)?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        let write_txn = self.db.begin_write().map_err(redb::Error::from)?;
        let removed = {
            let mut table = write_txn
                .open_table(REDB_CONTEXTS)
                .map_err(redb::Error::from)?;
            table.remove(key).map_err(redb::Error::from)?.is_some()
        };
        write_txn.commit().map_err(redb::Error::from)?;
        Ok(removed)
    }

    async fn purge_expired(&self, now: i64) -> Result<usize, StoreError> {
        use redb::{ReadableDatabase, ReadableTable};

        let read_txn = self.db.begin_read().map_err(redb::Error::from)?;
        let table = read_txn
            .open_table(REDB_CONTEXTS)
            .map_err(redb::Error::from)?;
        let mut expired = Vec::new();
        for entry in table.iter().map_err(redb::Error::from)? {
            let (key, value) = entry.map_err(redb::Error::from)?;
            let record = decode_record(value.value())?;
            if is_expired(record.expires_at, now) {
                expired.push(key.value().to_string());
            }
        }
        drop(table);
        drop(read_txn);

        if expired.is_empty() {
            return Ok(0);
        }

        let write_txn = self.db.begin_write().map_err(redb::Error::from)?;
        {
            let mut table = write_txn
                .open_table(REDB_CONTEXTS)
                .map_err(redb::Error::from)?;
            for key in &expired {
                table.remove(key.as_str()).map_err(redb::Error::from)?;
            }
        }
        write_txn.commit().map_err(redb::Error::from)?;
        Ok(expired.len())
    }
}

#[cfg(feature = "stateful-responses-tikv")]
pub struct TikvResponseContextStore {
    client: tikv_client::RawClient,
    prefix: Vec<u8>,
}

#[cfg(feature = "stateful-responses-tikv")]
impl TikvResponseContextStore {
    pub async fn connect(endpoints: Vec<String>, prefix: String) -> Result<Self, StoreError> {
        let client = tikv_client::RawClient::new(endpoints).await?;
        let prefix = normalize_tikv_prefix(&prefix)?;
        Ok(Self { client, prefix })
    }

    fn key(&self, key: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.prefix.len() + key.len());
        out.extend_from_slice(&self.prefix);
        out.extend_from_slice(key.as_bytes());
        out
    }
}

#[cfg(feature = "stateful-responses-tikv")]
#[async_trait]
impl ResponseContextStore for TikvResponseContextStore {
    async fn get(&self, key: &str) -> Result<Option<StoredResponseContext>, StoreError> {
        let Some(value) = self.client.get(self.key(key)).await? else {
            return Ok(None);
        };
        let record = decode_record(&value)?;
        if is_expired(record.expires_at, unix_timestamp()) {
            self.delete(key).await?;
            return Ok(None);
        }
        Ok(Some(record.context))
    }

    async fn put(
        &self,
        key: &str,
        context: &StoredResponseContext,
        expires_at: Option<i64>,
    ) -> Result<(), StoreError> {
        let value = serde_json::to_vec(&StoredResponseRecord {
            context: context.clone(),
            expires_at,
        })?;
        let key = self.key(key);
        match ttl_secs_for_put(expires_at) {
            Some(0) => self.client.delete(key).await?,
            Some(ttl) => self.client.put_with_ttl(key, value, ttl).await?,
            None => self.client.put(key, value).await?,
        }
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        let key = self.key(key);
        let existed = self.client.get(key.clone()).await?.is_some();
        self.client.delete(key).await?;
        Ok(existed)
    }

    async fn purge_expired(&self, now: i64) -> Result<usize, StoreError> {
        let Some(end) = prefix_range_end(&self.prefix) else {
            return Ok(0);
        };
        let pairs = self
            .client
            .scan(self.prefix.clone()..end, TIKV_PURGE_SCAN_LIMIT)
            .await?;
        let mut expired = Vec::new();
        for pair in pairs {
            let record = decode_record(pair.value())?;
            if is_expired(record.expires_at, now) {
                expired.push(pair.into_key());
            }
        }
        let deleted = expired.len();
        if deleted > 0 {
            self.client.batch_delete(expired).await?;
        }
        Ok(deleted)
    }
}

#[cfg(feature = "stateful-responses-tikv")]
fn normalize_tikv_prefix(prefix: &str) -> Result<Vec<u8>, StoreError> {
    let mut prefix = prefix.trim_matches('/').as_bytes().to_vec();
    if prefix.is_empty() {
        prefix.extend_from_slice(DEFAULT_TIKV_PREFIX.as_bytes());
    } else {
        prefix.push(b'/');
    }
    Ok(prefix)
}

#[cfg(feature = "stateful-responses-tikv")]
fn ttl_secs_for_put(expires_at: Option<i64>) -> Option<u64> {
    let expires_at = expires_at?;
    let ttl = expires_at.saturating_sub(unix_timestamp());
    Some(u64::try_from(ttl).unwrap_or(0))
}

#[cfg(feature = "stateful-responses-tikv")]
fn prefix_range_end(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    for index in (0..end.len()).rev() {
        if end[index] != u8::MAX {
            end[index] += 1;
            end.truncate(index + 1);
            return Some(end);
        }
    }
    None
}

#[derive(Clone, Debug, PartialEq)]
pub enum StoreConfig {
    Memory,
    #[cfg(feature = "stateful-responses-redb")]
    Redb(PathBuf),
    #[cfg(feature = "stateful-responses-tikv")]
    Tikv {
        endpoints: Vec<String>,
        prefix: String,
    },
}

impl StoreConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(&env::var(STORE_URL_ENV).unwrap_or_else(|_| "memory".to_string()))
    }

    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        let raw = raw.trim();
        if raw.eq_ignore_ascii_case("memory") {
            return Ok(Self::Memory);
        }
        #[cfg(feature = "stateful-responses-redb")]
        if let Some(path) = raw
            .strip_prefix("redb://")
            .or_else(|| raw.strip_prefix("redb:"))
        {
            return Ok(Self::Redb(PathBuf::from(path)));
        }
        #[cfg(feature = "stateful-responses-tikv")]
        if let Some(config) = raw
            .strip_prefix("tikv://")
            .or_else(|| raw.strip_prefix("tikv:"))
        {
            return parse_tikv_store_config(config);
        }
        anyhow::bail!(
            "unsupported store URL `{raw}`; enabled stores: {}",
            enabled_store_url_schemes()
        );
    }

    pub async fn open(&self) -> anyhow::Result<Arc<dyn ResponseContextStore>> {
        match self {
            Self::Memory => Ok(Arc::new(MemoryResponseContextStore::new())),
            #[cfg(feature = "stateful-responses-redb")]
            Self::Redb(path) => open_redb_store(path),
            #[cfg(feature = "stateful-responses-tikv")]
            Self::Tikv { endpoints, prefix } => open_tikv_store(endpoints, prefix).await,
        }
    }
}

fn enabled_store_url_schemes() -> String {
    let schemes = [
        "memory",
        #[cfg(feature = "stateful-responses-redb")]
        "redb:/path",
        #[cfg(feature = "stateful-responses-tikv")]
        "tikv://pd1:2379,pd2:2379/prefix",
    ];
    schemes.join(", ")
}

#[cfg(feature = "stateful-responses-tikv")]
fn parse_tikv_store_config(config: &str) -> anyhow::Result<StoreConfig> {
    let (endpoints, prefix) = config
        .split_once('/')
        .map(|(endpoints, prefix)| (endpoints, prefix))
        .unwrap_or((config, DEFAULT_TIKV_PREFIX));
    let endpoints = endpoints
        .split(',')
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if endpoints.is_empty() {
        anyhow::bail!("tikv store URL must include at least one PD endpoint");
    }
    Ok(StoreConfig::Tikv {
        endpoints,
        prefix: prefix.to_string(),
    })
}

#[cfg(feature = "stateful-responses-redb")]
fn open_redb_store(path: &Path) -> anyhow::Result<Arc<dyn ResponseContextStore>> {
    Ok(Arc::new(RedbResponseContextStore::open(path)?))
}

#[cfg(feature = "stateful-responses-tikv")]
async fn open_tikv_store(
    endpoints: &[String],
    prefix: &str,
) -> anyhow::Result<Arc<dyn ResponseContextStore>> {
    Ok(Arc::new(
        TikvResponseContextStore::connect(endpoints.to_vec(), prefix.to_string()).await?,
    ))
}

#[derive(Debug, Error)]
enum StoreResponseError {
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Store(#[from] StoreError),
}

#[derive(Debug, Error)]
enum ProxyError {
    #[error("{message}")]
    Client { status: StatusCode, message: String },
    #[error("upstream Responses request failed: {0}")]
    Upstream(#[source] reqwest::Error),
    #[error("state store failed: {0}")]
    Store(#[source] StoreError),
    #[error("failed to encode forwarded Responses request: {0}")]
    Serde(#[from] serde_json::Error),
}

impl ProxyError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self::Client {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn upstream(err: reqwest::Error) -> Self {
        Self::Upstream(err)
    }

    fn store(err: StoreError) -> Self {
        Self::Store(err)
    }

    fn into_response(self) -> Response {
        let status = match &self {
            Self::Client { status, .. } => *status,
            Self::Upstream(_) | Self::Store(_) | Self::Serde(_) => StatusCode::BAD_GATEWAY,
        };
        let message = self.to_string();
        (
            status,
            Json(json!({
                "error": {
                    "message": message,
                    "type": "stateful_responses_proxy_error",
                    "code": status.as_u16()
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_previous_context_and_string_input() {
        let previous = StoredResponseContext {
            input_items: vec![json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "old" }]
            })],
        };
        let body = json!({
            "model": "m",
            "previous_response_id": "resp_1",
            "input": "new"
        });

        let expanded = expand_request(body, Some(&previous), false).unwrap();

        assert!(expanded.body.get("previous_response_id").is_none());
        let input = expanded.body.get("input").unwrap().as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[1]["role"], "user");
        assert_eq!(input[1]["content"][0]["text"], "new");
    }

    #[test]
    fn requests_completion_token_ids_without_forwarding_token_data() {
        let previous = StoredResponseContext {
            input_items: vec![],
        };
        let body = json!({
            "model": "m",
            "input": [],
            "nvext": {
                "token_data": [3],
                "extra_fields": ["worker_id"]
            }
        });

        let expanded = expand_request(body, Some(&previous), true).unwrap();

        assert!(expanded.body["nvext"].get("token_data").is_none());
        assert_eq!(
            expanded.body["nvext"]["extra_fields"],
            json!(["worker_id", "completion_token_ids"])
        );
    }

    #[test]
    fn parses_completed_response_from_sse() {
        let sse = b"event: response.created\ndata: {\"type\":\"response.created\"}\n\n\
                    event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"output\":[]}}\n\n";

        let response = completed_response_from_sse(sse).unwrap().unwrap();
        assert_eq!(response["id"], "resp_1");
    }

    #[test]
    fn parses_completed_response_from_crlf_sse() {
        let sse = b"event: response.completed\r\n\
                    data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"output\":[]}}\r\n\r\n";

        let response = completed_response_from_sse(sse).unwrap().unwrap();
        assert_eq!(response["id"], "resp_1");
    }

    #[tokio::test]
    async fn memory_store_round_trips_context() {
        let store = MemoryResponseContextStore::new();
        let context = StoredResponseContext {
            input_items: vec![json!({"type": "message"})],
        };

        store.put("resp_1", &context, None).await.unwrap();

        assert_eq!(store.get("resp_1").await.unwrap(), Some(context));
    }

    #[tokio::test]
    async fn memory_store_expires_and_deletes_context() {
        let store = MemoryResponseContextStore::new();
        let context = StoredResponseContext {
            input_items: vec![json!({"type": "message"})],
        };

        store
            .put("resp_1", &context, Some(unix_timestamp() - 1))
            .await
            .unwrap();

        assert_eq!(store.get("resp_1").await.unwrap(), None);

        store.put("resp_2", &context, None).await.unwrap();
        assert!(store.delete("resp_2").await.unwrap());
        assert_eq!(store.get("resp_2").await.unwrap(), None);
    }

    #[cfg(feature = "stateful-responses-redb")]
    #[tokio::test]
    async fn redb_store_round_trips_context() {
        let dir = tempfile::tempdir().unwrap();
        let store = RedbResponseContextStore::open(dir.path().join("state.redb")).unwrap();
        let context = StoredResponseContext {
            input_items: vec![json!({"type": "message"})],
        };

        store.put("resp_1", &context, None).await.unwrap();

        assert_eq!(store.get("resp_1").await.unwrap(), Some(context));
    }

    #[test]
    fn parses_memory_store_url() {
        assert_eq!(StoreConfig::parse("memory").unwrap(), StoreConfig::Memory);
    }

    #[cfg(feature = "stateful-responses-redb")]
    #[test]
    fn parses_redb_store_url_when_feature_enabled() {
        assert_eq!(
            StoreConfig::parse("redb:/tmp/state.redb").unwrap(),
            StoreConfig::Redb(PathBuf::from("/tmp/state.redb"))
        );
    }

    #[cfg(feature = "stateful-responses-tikv")]
    #[test]
    fn parses_tikv_store_url_when_feature_enabled() {
        assert_eq!(
            StoreConfig::parse("tikv://127.0.0.1:2379,127.0.0.2:2379/prod/responses").unwrap(),
            StoreConfig::Tikv {
                endpoints: vec!["127.0.0.1:2379".to_string(), "127.0.0.2:2379".to_string()],
                prefix: "prod/responses".to_string(),
            }
        );
    }

    #[test]
    fn reads_previous_response_id() {
        let body = json!({ "previous_response_id": "resp_a", "input": [] });

        assert_eq!(previous_response_id(&body), Some("resp_a".to_string()));
    }

    #[test]
    fn parses_optional_ttl() {
        assert_eq!(
            parse_optional_duration_secs("ttl", "60").unwrap(),
            Some(Duration::from_secs(60))
        );
        assert_eq!(parse_optional_duration_secs("ttl", "0").unwrap(), None);
        assert_eq!(parse_optional_duration_secs("ttl", "never").unwrap(), None);
    }

    #[cfg(feature = "stateful-responses-tikv")]
    #[test]
    fn tikv_prefix_range_end_bounds_prefix() {
        assert_eq!(
            normalize_tikv_prefix("prod/responses").unwrap(),
            b"prod/responses/".to_vec()
        );
        assert_eq!(
            prefix_range_end(b"prod/responses/").unwrap(),
            b"prod/responses0".to_vec()
        );
    }
}

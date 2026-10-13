// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin sidecar client for SGLang's native `sglang.runtime.v1.SglangService`.

use std::collections::HashSet;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use dynamo_backend_common::{BackendError, DisaggregationMode, DynamoError, ErrorType};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcEndpoint, GrpcTransportConfig, format_error_chain,
};
use serde::Deserialize;
use serde_json::Value;
use tokio::time::{Instant, timeout_at};
use tonic::transport::{Channel, Endpoint};

use crate::proto as pb;
use crate::proto::sglang_service_client::SglangServiceClient;

pub type Client = SglangServiceClient<Channel>;

pub(crate) const WORKER_GROUP_KEY: &str = "sglang_worker_group_id";
pub(crate) const KV_CONFIG_KEY: &str = "sglang_sidecar_kv_events";

/// Node-local KV publishers reported by the engine's GetServerInfo RPC.
/// Other server fields (and unused source fields such as replay_endpoint) are
/// ignored so the metadata-only and full servers share the same wire contract.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct NodeMetadata {
    pub node_rank: u32,
    pub nnodes: u32,
    pub dp_size: u32,
    pub dist_init_addr: Option<String>,
    pub kv_event_sources: Vec<LocalKvEventSource>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct LocalKvEventSource {
    pub dp_rank: u32,
    pub endpoint: String,
    pub topic: String,
    pub block_size: u32,
}

impl NodeMetadata {
    /// An absent source list permits legacy leader discovery; an explicit empty
    /// list is authoritative and must never fall back to all global DP ranks.
    pub(crate) fn from_server_info(server_info: &Value) -> anyhow::Result<Option<Self>> {
        ensure!(
            server_info.is_object(),
            "GetServerInfo must contain a JSON object"
        );
        if server_info.get("kv_event_sources").is_none() {
            return Ok(None);
        }
        let metadata: Self = serde_json::from_value(server_info.clone())
            .context("invalid GetServerInfo node-local KV metadata")?;
        metadata.validate()?;
        Ok(Some(metadata))
    }

    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.nnodes > 0 && self.node_rank < self.nnodes,
            "invalid GetServerInfo node topology"
        );
        ensure!(self.dp_size > 0, "GetServerInfo dp_size must be positive");
        ensure!(
            self.nnodes == 1
                || self
                    .dist_init_addr
                    .as_ref()
                    .is_some_and(|addr| !addr.trim().is_empty()),
            "multinode sidecars require dist_init_addr for leader matching"
        );
        let mut ranks = HashSet::new();
        let mut endpoints = HashSet::new();
        for source in &self.kv_event_sources {
            ensure!(
                source.dp_rank < self.dp_size,
                "KV source rank {} is outside dp_size {}",
                source.dp_rank,
                self.dp_size
            );
            ensure!(
                ranks.insert(source.dp_rank),
                "duplicate local KV source rank {}",
                source.dp_rank
            );
            ensure!(
                endpoints.insert(&source.endpoint),
                "duplicate local KV source endpoint {}",
                source.endpoint
            );
            ensure!(
                source.block_size > 0,
                "KV source block_size must be positive"
            );
            validate_endpoint(&source.endpoint)?;
        }
        Ok(())
    }

    pub(crate) fn validate_registration(
        &self,
        dp_size: u32,
        block_size: Option<u32>,
    ) -> anyhow::Result<()> {
        ensure!(
            self.dp_size == dp_size,
            "GetServerInfo dp_size does not match leader registration"
        );
        for source in &self.kv_event_sources {
            ensure!(
                Some(source.block_size) == block_size,
                "KV source block_size does not match leader registration"
            );
        }
        Ok(())
    }

    pub(crate) async fn worker_group_id(
        &self,
        deadline: Instant,
    ) -> anyhow::Result<Option<String>> {
        if self.nnodes == 1 {
            return Ok(None);
        }
        let raw = self
            .dist_init_addr
            .as_deref()
            .context("missing dist_init_addr")?
            .trim();
        let url = if raw.contains("://") {
            raw.to_owned()
        } else {
            format!("tcp://{raw}")
        };
        let address = url::Url::parse(&url).context("invalid dist_init_addr")?;
        ensure!(
            address.scheme() == "tcp",
            "dist_init_addr must be a TCP address"
        );
        validate_endpoint(&url)?;
        // Match the in-process group key using the shared rendezvous address,
        // not the local source or gRPC address.
        resolve_rendezvous(move || address.socket_addrs(|| None), deadline)
            .await
            .and_then(worker_group_id_from_addresses)
            .map(Some)
    }
}

// getaddrinfo cannot be cancelled. Keep it off Tokio's workers and blocking
// pool so dropping startup or shutting down the runtime never waits for DNS.
// Each startup performs one lookup; an abandoned thread exits when DNS returns.
async fn resolve_rendezvous(
    resolve: impl FnOnce() -> std::io::Result<Vec<SocketAddr>> + Send + 'static,
    deadline: Instant,
) -> anyhow::Result<Vec<SocketAddr>> {
    anyhow::ensure!(
        Instant::now() < deadline,
        "rendezvous DNS startup deadline elapsed"
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("sglang-rendezvous-dns".into())
        .spawn(move || {
            let _ = tx.send(resolve());
        })
        .context("failed to start rendezvous DNS resolver")?;
    timeout_at(deadline, rx)
        .await
        .context("timed out resolving SGLang rendezvous address")?
        .context("rendezvous DNS resolver stopped")?
        .context("failed to resolve SGLang rendezvous address")
}

fn worker_group_id_from_addresses(mut resolved: Vec<SocketAddr>) -> anyhow::Result<String> {
    // Nodes may receive the same DNS answers in different orders.
    resolved.sort_unstable();
    let address = resolved
        .first()
        .context("dist_init_addr resolved to no addresses")?;
    Ok(format!("dist_init:tcp://{address}"))
}

fn validate_endpoint(endpoint: &str) -> anyhow::Result<()> {
    if let Some(path) = endpoint.strip_prefix("ipc://") {
        ensure!(
            (path.starts_with('/') || path.starts_with('@'))
                && path.len() > 1
                && !path.contains('\0'),
            "IPC source must have an absolute or abstract socket path"
        );
        return Ok(());
    }
    let url = url::Url::parse(endpoint).context("invalid KV source endpoint")?;
    let host = url
        .host_str()
        .context("KV source endpoint requires a host")?;
    ensure!(
        url.scheme() == "tcp" && url.port().is_some_and(|port| port > 0),
        "KV source must use tcp://HOST:PORT or ipc://PATH"
    );
    ensure!(
        !matches!(host, "*" | "0.0.0.0" | "[::]" | "::"),
        "KV source endpoint must be dialable, not a wildcard bind address"
    );
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        bail!("KV source TCP endpoint must contain only host and port");
    }
    Ok(())
}
const RETRY_LOG_INTERVAL: Duration = Duration::from_secs(30);

/// Metadata exposed by SGLang's model/server discovery RPCs.
#[derive(Clone, Debug)]
pub struct Discovery {
    pub model_path: String,
    pub tokenizer_path: String,
    pub served_model_name: Option<String>,
    pub max_model_len: Option<u32>,
    pub model_info: Value,
    pub server_info: Value,
}

#[derive(Debug)]
pub(crate) enum StartupDiscovery {
    Leader(Box<Discovery>),
    Follower,
}

pub(crate) async fn bootstrap_discover(
    endpoint: &GrpcEndpoint,
    transport: &GrpcTransportConfig,
    bootstrap: bool,
) -> Result<StartupDiscovery, DynamoError> {
    let deadline = Instant::now() + transport.startup_deadline;
    let mut client = connect(endpoint, transport, deadline, bootstrap).await?;
    let server_info = get_server_info(&mut client, deadline).await?;
    if json_u32(&server_info, "node_rank").is_some_and(|rank| rank > 0) {
        // Followers expose metadata only. Their local KV sources are
        // validated by the headless startup path before relaying.
        Ok(StartupDiscovery::Follower)
    } else {
        discover_with_server_info(&mut client, server_info, deadline)
            .await
            .map(|d| StartupDiscovery::Leader(Box::new(d)))
    }
}

/// `bootstrap`: true for synchronous constructors before logging setup;
/// false for deferred launcher discovery and `LLMEngine::start`.
pub async fn connect(
    uri: &GrpcEndpoint,
    cfg: &GrpcTransportConfig,
    deadline: Instant,
    bootstrap: bool,
) -> Result<Client, DynamoError> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|err| invalid_arg(format!("invalid SGLang gRPC endpoint `{uri}`: {err}")))?;
    let started = Instant::now();
    let mut attempt = 0_u64;
    let mut last_err;
    let mut last_logged_at: Option<Instant> = None;
    loop {
        attempt += 1;
        match try_connect_once(&endpoint, cfg, deadline).await {
            Ok(client) => return Ok(client),
            Err(err) => {
                last_err = err;
                if Instant::now() >= deadline {
                    return Err(cannot_connect(format!(
                        "could not reach SGLang gRPC at {uri} within {:?}: {last_err}",
                        cfg.startup_deadline
                    )));
                }
                let now = Instant::now();
                if last_logged_at.is_none_or(|last| now.duration_since(last) >= RETRY_LOG_INTERVAL)
                {
                    // Synchronous constructors may precede logging setup;
                    // deferred launcher discovery already has a subscriber.
                    if bootstrap {
                        eprintln!(
                            "SGLang gRPC connection attempt failed; retrying (endpoint={uri}, attempt={attempt}, elapsed={:?}, retry_interval={:?}, error={last_err})",
                            started.elapsed(),
                            cfg.retry_interval,
                        );
                    } else {
                        tracing::warn!(
                            endpoint = %uri,
                            attempt,
                            elapsed = ?started.elapsed(),
                            retry_interval = ?cfg.retry_interval,
                            error = %last_err,
                            "SGLang gRPC connection attempt failed; retrying"
                        );
                    }
                    last_logged_at = Some(now);
                }
                tokio::time::sleep_until((now + cfg.retry_interval).min(deadline)).await;
            }
        }
    }
}

async fn try_connect_once(
    endpoint: &Endpoint,
    cfg: &GrpcTransportConfig,
    deadline: Instant,
) -> Result<Client, String> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err("startup deadline elapsed".to_string());
    }
    let endpoint = endpoint
        .clone()
        .connect_timeout(cfg.connect_attempt_timeout.min(remaining));
    let channel = timeout_at(deadline, endpoint.connect())
        .await
        .map_err(|_| "startup deadline elapsed while connecting".to_string())?
        .map_err(|e| format_error_chain(&e))?;
    Ok(client_from_channel(channel))
}

fn client_from_channel(channel: Channel) -> Client {
    SglangServiceClient::new(channel)
        .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
}

/// Fixed-size pool of independent HTTP/2 connections. Generation calls are
/// round-robined so high concurrency does not funnel through one codec task.
pub struct Pool {
    clients: Vec<Client>,
    next: AtomicUsize,
}

impl Pool {
    // bootstrap=false: Pool::connect's only call site is LLMEngine::start
    // (lib/sidecar/sglang/src/engine.rs), after the tracing subscriber is
    // installed. See connect()'s own doc comment.
    pub async fn connect(
        uri: &GrpcEndpoint,
        cfg: &GrpcTransportConfig,
        deadline: Instant,
    ) -> Result<Self, DynamoError> {
        let size = cfg.connections.get();
        let mut clients = Vec::with_capacity(size);
        for _ in 0..size {
            clients.push(connect(uri, cfg, deadline, false).await?);
        }
        Ok(Self {
            clients,
            next: AtomicUsize::new(0),
        })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    pub fn stream_client(&self) -> Client {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        self.clients[index].clone()
    }

    pub fn control_client(&self) -> Client {
        self.clients[0].clone()
    }
}

pub async fn discover(client: &mut Client, deadline: Instant) -> Result<Discovery, DynamoError> {
    let server_info = get_server_info(client, deadline).await?;
    discover_with_server_info(client, server_info, deadline).await
}

async fn discover_with_server_info(
    client: &mut Client,
    server_info: Value,
    deadline: Instant,
) -> Result<Discovery, DynamoError> {
    if json_u32(&server_info, "node_rank").is_some_and(|rank| rank > 0) {
        return Err(invalid_arg(
            "inference discovery requires node_rank=0; followers expose only GetServerInfo",
        ));
    }
    let model = rpc_with_deadline(
        "GetModelInfo",
        deadline,
        client.get_model_info(pb::GetModelInfoRequest {}),
    )
    .await?
    .into_inner();
    let models = rpc_with_deadline(
        "ListModels",
        deadline,
        client.list_models(pb::ListModelsRequest {}),
    )
    .await?
    .into_inner()
    .models;

    parse_discovery(model, server_info, models)
}

/// Follower engines implement only this RPC, not model discovery or health
/// checks. Use it for both startup discovery and metadata-only liveness checks.
pub(crate) async fn get_server_info(
    client: &mut Client,
    deadline: Instant,
) -> Result<Value, DynamoError> {
    let server = rpc_with_deadline(
        "GetServerInfo",
        deadline,
        client.get_server_info(pb::GetServerInfoRequest {}),
    )
    .await?
    .into_inner();
    parse_json_object("GetServerInfo.json_info", &server.json_info)
}

pub async fn health_check(client: &mut Client, deadline: Instant) -> Result<bool, DynamoError> {
    rpc_with_deadline(
        "HealthCheck",
        deadline,
        client.health_check(pb::HealthCheckRequest {}),
    )
    .await
    .map(|response| response.into_inner().healthy)
}

pub async fn abort(
    client: &mut Client,
    request: pb::AbortRequest,
    timeout: Duration,
) -> Result<(), DynamoError> {
    rpc_with_deadline("Abort", Instant::now() + timeout, client.abort(request))
        .await
        .map(|_| ())
}

async fn rpc_with_deadline<T, F>(rpc: &str, deadline: Instant, future: F) -> Result<T, DynamoError>
where
    F: Future<Output = Result<T, tonic::Status>>,
{
    match timeout_at(deadline, future).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(status)) => Err(status_to_dynamo(rpc, status)),
        Err(_) => Err(connection_timeout(format!(
            "{rpc} exceeded the configured deadline"
        ))),
    }
}

fn parse_discovery(
    model: pb::GetModelInfoResponse,
    server_info: Value,
    models: Vec<pb::ModelCard>,
) -> Result<Discovery, DynamoError> {
    let model_info = parse_json_object("GetModelInfo.json_info", &model.json_info)?;
    // Generate responses are forwarded as token deltas. Accepting cumulative
    // output here would duplicate tokens and inflate completion usage.
    if server_info
        .get("incremental_streaming_output")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(invalid_arg(
            "SGLang sidecar requires incremental streaming output; restart the SGLang server \
             with --incremental-streaming-output to prevent duplicated tokens and inflated \
             completion-token counts",
        ));
    }
    let model_path = if model.model_path.trim().is_empty() {
        model_info
            .get("model_path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    } else {
        model.model_path
    };
    if model_path.trim().is_empty() {
        return Err(protocol_error(
            "SGLang GetModelInfo returned an empty model_path",
        ));
    }
    let tokenizer_path = model_info
        .get("tokenizer_path")
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .unwrap_or(&model_path)
        .to_string();

    let primary = models
        .iter()
        .find(|candidate| candidate.root == model_path || candidate.id == model_path)
        .or_else(|| models.first());
    let served_model_name = server_info
        .get("served_model_name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| {
            primary
                .map(|card| card.id.as_str())
                .filter(|name| !name.is_empty() && *name != model_path)
                .map(str::to_string)
        });
    let max_model_len = primary
        .and_then(|card| card.max_model_len)
        .and_then(|value| u32::try_from(value).ok())
        .or_else(|| json_u32(&server_info, "context_length"))
        .or_else(|| json_u32(&server_info, "max_req_input_len"));

    Ok(Discovery {
        model_path,
        tokenizer_path,
        served_model_name,
        max_model_len,
        model_info,
        server_info,
    })
}

pub(crate) fn discovery_mode(server_info: &Value) -> Result<DisaggregationMode, DynamoError> {
    match server_info
        .get("disaggregation_mode")
        .and_then(Value::as_str)
        .unwrap_or("null")
    {
        "null" | "agg" | "aggregated" => Ok(DisaggregationMode::Aggregated),
        "prefill" => Ok(DisaggregationMode::Prefill),
        "decode" => Ok(DisaggregationMode::Decode),
        mode => Err(protocol_error(format!(
            "unsupported SGLang disaggregation_mode `{mode}`"
        ))),
    }
}

fn parse_json_object(label: &str, raw: &str) -> Result<Value, DynamoError> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|err| protocol_error(format!("invalid {label}: {err}")))?;
    if !value.is_object() {
        return Err(protocol_error(format!("{label} must be a JSON object")));
    }
    Ok(value)
}

pub(crate) fn json_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|entry| {
        entry
            .as_u64()
            .or_else(|| entry.as_i64().and_then(|number| u64::try_from(number).ok()))
            .or_else(|| entry.as_str().and_then(|number| number.parse().ok()))
    })
}

pub(crate) fn json_u32(value: &Value, key: &str) -> Option<u32> {
    json_u64(value, key).and_then(|number| u32::try_from(number).ok())
}

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub fn invalid_arg(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

/// The frontend returns `message` to the client, so it takes only fixed request-validation text.
pub(crate) fn invalid_request(message: &'static str) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(message)
        .public_message(message)
        .build()
}

pub fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

pub fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

pub(crate) fn connection_timeout(message: impl Into<String>) -> DynamoError {
    backend(BackendError::ConnectionTimeout, message)
}

pub(crate) fn cancelled(message: impl Into<String>) -> DynamoError {
    backend(BackendError::Cancelled, message)
}

pub fn protocol_error(message: impl Into<String>) -> DynamoError {
    backend(BackendError::Unknown, message)
}

pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument | tonic::Code::NotFound | tonic::Code::OutOfRange => {
            BackendError::InvalidArgument
        }
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        _ => BackendError::Unknown,
    };
    backend(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_backend_common::{BackendError, ErrorType};
    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::time::Instant;
    use tonic::transport::Endpoint;

    use super::{
        NodeMetadata, client_from_channel, discover, discovery_mode, json_u32, json_u64,
        parse_discovery, rpc_with_deadline, status_to_dynamo, worker_group_id_from_addresses,
    };
    use crate::proto as pb;

    #[test]
    fn numeric_discovery_fields_accept_numbers_and_strings() {
        let value = json!({"a": 16, "b": "32", "c": -1});
        assert_eq!(json_u64(&value, "a"), Some(16));
        assert_eq!(json_u32(&value, "b"), Some(32));
        assert_eq!(json_u64(&value, "c"), None);
        for value in [json!(u64::MAX), json!(u64::MAX.to_string())] {
            let info = json!({"limit": value});
            assert_eq!(json_u64(&info, "limit"), Some(u64::MAX));
            assert_eq!(json_u32(&info, "limit"), None);
        }
        for value in [json!(u32::MAX), json!(u32::MAX.to_string())] {
            assert_eq!(json_u32(&json!({"limit": value}), "limit"), Some(u32::MAX));
        }
        for value in [
            json!(null),
            json!(true),
            json!(1.5),
            json!("-1"),
            json!("1.5"),
            json!("18446744073709551616"),
        ] {
            assert_eq!(json_u64(&json!({"limit": value}), "limit"), None);
        }
        assert_eq!(json_u32(&json!({}), "limit"), None);
    }

    #[test]
    fn discovery_preserves_distinct_tokenizer_path() {
        let discovery = parse_discovery(
            pb::GetModelInfoResponse {
                model_path: "model-repo".to_string(),
                json_info: json!({"tokenizer_path": "tokenizer-repo"}).to_string(),
            },
            json!({"incremental_streaming_output": true}),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(discovery.model_path, "model-repo");
        assert_eq!(discovery.tokenizer_path, "tokenizer-repo");
    }

    fn node_metadata_json() -> serde_json::Value {
        json!({
            "node_rank": 1, "nnodes": 2, "dp_size": 8,
            "dist_init_addr": "127.0.0.1:2345",
            "kv_event_sources": [{
                "dp_rank": 4, "endpoint": "tcp://127.0.0.1:5561",
                "topic": "", "block_size": 64
            }]
        })
    }

    #[tokio::test]
    async fn parses_node_local_sources_and_ignores_unrelated_server_fields() {
        let mut raw = node_metadata_json();
        raw["model_path"] = json!("model-repo");
        // Live-only relaying does not interpret the engine's optional replay field.
        raw["kv_event_sources"][0]["replay_endpoint"] = json!("unused");
        let metadata = NodeMetadata::from_server_info(&raw).unwrap().unwrap();
        assert_eq!(metadata.kv_event_sources[0].dp_rank, 4);
        assert_eq!(
            metadata
                .worker_group_id(Instant::now() + std::time::Duration::from_secs(1))
                .await
                .unwrap()
                .as_deref(),
            Some("dist_init:tcp://127.0.0.1:2345")
        );
        metadata.validate_registration(8, Some(64)).unwrap();
        assert!(metadata.validate_registration(4, Some(64)).is_err());
        assert!(metadata.validate_registration(8, Some(32)).is_err());
    }

    #[test]
    fn local_sources_distinguish_absent_empty_and_invalid_metadata() {
        assert!(
            NodeMetadata::from_server_info(&json!({}))
                .unwrap()
                .is_none()
        );
        let mut raw = node_metadata_json();
        raw["kv_event_sources"] = json!([]);
        assert!(
            NodeMetadata::from_server_info(&raw)
                .unwrap()
                .unwrap()
                .kv_event_sources
                .is_empty()
        );
        raw["kv_event_sources"] = serde_json::Value::Null;
        assert!(NodeMetadata::from_server_info(&raw).is_err());
    }

    #[test]
    fn rejects_duplicate_rank_or_endpoint() {
        let mut raw = node_metadata_json();
        let mut source = raw["kv_event_sources"][0].clone();
        source["endpoint"] = json!("tcp://127.0.0.1:5562");
        raw["kv_event_sources"].as_array_mut().unwrap().push(source);
        assert!(NodeMetadata::from_server_info(&raw).is_err());
        raw["kv_event_sources"][1]["dp_rank"] = json!(5);
        raw["kv_event_sources"][1]["endpoint"] = json!("tcp://127.0.0.1:5561");
        assert!(NodeMetadata::from_server_info(&raw).is_err());
    }

    #[test]
    fn source_rank_and_block_size_must_be_valid() {
        let mut raw = node_metadata_json();
        raw["kv_event_sources"][0]["dp_rank"] = json!(8);
        assert!(NodeMetadata::from_server_info(&raw).is_err());
        raw["kv_event_sources"][0]["dp_rank"] = json!(4);
        raw["kv_event_sources"][0]["block_size"] = json!(0);
        assert!(NodeMetadata::from_server_info(&raw).is_err());
    }

    #[tokio::test]
    async fn normalizes_ipv6_group_id_and_accepts_bound_ipc_sources() {
        let mut raw = node_metadata_json();
        raw["dist_init_addr"] = json!("tcp://[::1]:2345");
        raw["kv_event_sources"][0]["endpoint"] = json!("ipc:///engine/kv-events");
        let metadata = NodeMetadata::from_server_info(&raw).unwrap().unwrap();
        assert_eq!(
            metadata
                .worker_group_id(Instant::now() + std::time::Duration::from_secs(1))
                .await
                .unwrap()
                .as_deref(),
            Some("dist_init:tcp://[::1]:2345")
        );
    }

    #[tokio::test]
    async fn rendezvous_dns_deadline_bounds_a_blocked_resolver() {
        let (release, blocked) = std::sync::mpsc::channel::<()>();
        let error = super::resolve_rendezvous(
            move || {
                let _ = blocked.recv_timeout(std::time::Duration::from_secs(3));
                Ok(vec![])
            },
            Instant::now() + std::time::Duration::from_millis(25),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("timed out resolving"));
        drop(release);
    }

    #[test]
    fn cancelling_dns_does_not_hold_up_runtime_shutdown() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel::<()>();
        runtime.block_on(async {
            let lookup = super::resolve_rendezvous(
                move || {
                    let _ = started_tx.send(());
                    // Bound the injected delay even if runtime shutdown regresses.
                    let _ = blocked.recv_timeout(std::time::Duration::from_secs(3));
                    Ok(vec![])
                },
                Instant::now() + std::time::Duration::from_secs(10),
            );
            tokio::select! {
                result = lookup => panic!("resolver returned before cancellation: {result:?}"),
                _ = started_rx => {},
            }
        });
        let shutdown_started = std::time::Instant::now();
        drop(runtime);
        let elapsed = shutdown_started.elapsed();
        drop(release);
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "shutdown waited for DNS: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn rendezvous_dns_propagates_resolution_errors() {
        let error = super::resolve_rendezvous(
            || Err(std::io::Error::other("injected resolver failure")),
            Instant::now() + std::time::Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("injected resolver failure"));
    }

    #[test]
    fn worker_group_id_is_independent_of_dns_answer_order() {
        let addresses = vec![
            "[::1]:2345".parse().unwrap(),
            "127.0.0.2:2345".parse().unwrap(),
            "127.0.0.1:2345".parse().unwrap(),
        ];
        let reversed = addresses.iter().copied().rev().collect();
        let expected = "dist_init:tcp://127.0.0.1:2345";
        assert_eq!(worker_group_id_from_addresses(addresses).unwrap(), expected);
        assert_eq!(worker_group_id_from_addresses(reversed).unwrap(), expected);
        assert_eq!(
            worker_group_id_from_addresses(Vec::new())
                .unwrap_err()
                .to_string(),
            "dist_init_addr resolved to no addresses"
        );
    }

    #[test]
    fn discovery_roles_accept_native_aliases_and_reject_unknown_strings() {
        use dynamo_backend_common::DisaggregationMode;

        for (value, expected) in [
            (json!(null), DisaggregationMode::Aggregated),
            (json!("null"), DisaggregationMode::Aggregated),
            (json!("agg"), DisaggregationMode::Aggregated),
            (json!("aggregated"), DisaggregationMode::Aggregated),
            (json!("prefill"), DisaggregationMode::Prefill),
            (json!("decode"), DisaggregationMode::Decode),
        ] {
            assert_eq!(
                discovery_mode(&json!({"disaggregation_mode": value})).unwrap(),
                expected
            );
        }
        assert_eq!(
            discovery_mode(&json!({})).unwrap(),
            DisaggregationMode::Aggregated
        );
        for mode in ["encode", "unknown", ""] {
            let error = discovery_mode(&json!({"disaggregation_mode": mode})).unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::Unknown)
            );
            assert!(
                error
                    .to_string()
                    .contains("unsupported SGLang disaggregation_mode")
            );
        }
    }

    #[test]
    fn discovery_requires_incremental_streaming() {
        for info in [
            json!({}),
            json!({"incremental_streaming_output": false}),
            json!({"incremental_streaming_output": "true"}),
            json!({"incremental_streaming_output": null}),
        ] {
            let error = parse_discovery(
                pb::GetModelInfoResponse {
                    model_path: "model-repo".to_string(),
                    json_info: "{}".to_string(),
                },
                info,
                Vec::new(),
            )
            .unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
            assert!(
                error.to_string().contains("--incremental-streaming-output"),
                "{error}"
            );
        }
    }

    #[test]
    fn discovery_rejects_malformed_json_and_non_object_metadata() {
        for label in ["GetModelInfo.json_info", "GetServerInfo.json_info"] {
            for raw in ["{", "null", "[]", "1", "\"metadata\""] {
                let error = super::parse_json_object(label, raw).unwrap_err();
                assert_eq!(
                    error.error_type(),
                    ErrorType::Backend(BackendError::Unknown)
                );
                assert!(error.to_string().contains(label), "{raw}: {error}");
            }
        }
    }

    #[test]
    fn discovery_resolves_model_path_and_tokenizer_fallbacks() {
        for (native_path, json_path, expected) in [
            ("native-model", "json-model", Some("native-model")),
            (" ", "json-model", Some("json-model")),
            ("", " ", None),
        ] {
            for tokenizer in [json!(null), json!(""), json!(" "), json!(7)] {
                let result = parse_discovery(
                    pb::GetModelInfoResponse {
                        model_path: native_path.into(),
                        json_info: json!({"model_path": json_path, "tokenizer_path": tokenizer})
                            .to_string(),
                    },
                    json!({"incremental_streaming_output": true}),
                    vec![],
                );
                if let Some(expected) = expected {
                    let info = result.unwrap();
                    assert_eq!(info.model_path, expected);
                    assert_eq!(info.tokenizer_path, expected);
                    assert_eq!(info.served_model_name, None);
                    assert_eq!(info.max_model_len, None);
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(
                        error.error_type(),
                        ErrorType::Backend(BackendError::Unknown)
                    );
                    assert!(error.to_string().contains("empty model_path"));
                }
            }
        }
    }

    #[test]
    fn discovery_prefers_matching_model_and_server_alias() {
        for (id, root, fallback_name) in [
            ("card-alias", "model", Some("card-alias")),
            ("model", "different-root", None),
        ] {
            for server_name in [None, Some("server-alias")] {
                let model_info = json!({"tokenizer_path": "tokenizer", "custom": [1, 2]});
                let server_info = json!({
                    "incremental_streaming_output": true,
                    "served_model_name": server_name,
                    "context_length": 1024,
                });
                let info = parse_discovery(
                    pb::GetModelInfoResponse {
                        model_path: "model".into(),
                        json_info: model_info.to_string(),
                    },
                    server_info.clone(),
                    vec![
                        pb::ModelCard {
                            id: "unrelated".into(),
                            max_model_len: Some(512),
                            ..Default::default()
                        },
                        pb::ModelCard {
                            id: id.into(),
                            root: root.into(),
                            max_model_len: Some(4096),
                            ..Default::default()
                        },
                    ],
                )
                .unwrap();
                assert_eq!(
                    info.served_model_name.as_deref(),
                    server_name.or(fallback_name)
                );
                assert_eq!(info.max_model_len, Some(4096));
                assert_eq!(info.model_info, model_info);
                assert_eq!(info.server_info, server_info);
            }
        }
    }

    #[test]
    fn discovery_falls_back_to_first_card_and_valid_server_limits() {
        for (card_limit, context, input_limit, expected) in [
            (Some(2048), json!(4096), json!(8192), Some(2048)),
            (Some(-1), json!("4096"), json!(8192), Some(4096)),
            (
                None,
                json!(u64::from(u32::MAX) + 1),
                json!("8192"),
                Some(8192),
            ),
            (None, json!(null), json!(-1), None),
        ] {
            let info = parse_discovery(
                pb::GetModelInfoResponse {
                    model_path: "model".into(),
                    json_info: "{}".into(),
                },
                json!({
                    "incremental_streaming_output": true,
                    "served_model_name": "",
                    "context_length": context,
                    "max_req_input_len": input_limit,
                }),
                vec![pb::ModelCard {
                    id: "first-alias".into(),
                    max_model_len: card_limit,
                    ..Default::default()
                }],
            )
            .unwrap();
            assert_eq!(info.served_model_name.as_deref(), Some("first-alias"));
            assert_eq!(info.max_model_len, expected);
        }
    }

    #[test]
    fn rpc_status_mapping_preserves_error_kind_and_context() {
        for (code, kind) in [
            (tonic::Code::InvalidArgument, BackendError::InvalidArgument),
            (tonic::Code::NotFound, BackendError::InvalidArgument),
            (tonic::Code::OutOfRange, BackendError::InvalidArgument),
            (tonic::Code::Unavailable, BackendError::CannotConnect),
            (tonic::Code::Cancelled, BackendError::Cancelled),
            (
                tonic::Code::DeadlineExceeded,
                BackendError::ConnectionTimeout,
            ),
            (tonic::Code::Internal, BackendError::Unknown),
        ] {
            let error =
                status_to_dynamo("GetModelInfo", tonic::Status::new(code, "native failure"));
            assert_eq!(error.error_type(), ErrorType::Backend(kind));
            assert!(error.to_string().contains("GetModelInfo: native failure"));
            assert!(error.to_string().contains(&format!("{code:?}")));
        }
    }

    #[tokio::test]
    async fn rpc_deadline_preserves_success_status_and_timeout() {
        let deadline = Instant::now() + Duration::from_secs(60);
        assert_eq!(
            rpc_with_deadline("HealthCheck", deadline, async { Ok(17) })
                .await
                .unwrap(),
            17
        );
        let error = rpc_with_deadline::<(), _>("HealthCheck", deadline, async {
            Err(tonic::Status::unavailable("offline"))
        })
        .await
        .unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::CannotConnect)
        );

        let error =
            rpc_with_deadline::<(), _>("HealthCheck", Instant::now(), std::future::pending())
                .await
                .unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::ConnectionTimeout)
        );
        assert!(
            error
                .to_string()
                .contains("HealthCheck exceeded the configured deadline")
        );
    }

    #[tokio::test]
    async fn discovery_deadline_bounds_a_half_open_peer() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let peer = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        let channel = Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect_lazy();
        let mut client = client_from_channel(channel);
        let started = Instant::now();
        let result = discover(&mut client, started + Duration::from_millis(100)).await;
        peer.abort();

        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}

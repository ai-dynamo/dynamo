// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin sidecar client for SGLang's native `sglang.runtime.v1.SglangService`.

use std::collections::HashSet;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use dynamo_backend_common::{BackendError, DisaggregationMode, DynamoError, ErrorType};
use dynamo_sidecar_common::{DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcEndpoint, GrpcTransportConfig};
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

    pub(crate) fn worker_group_id(&self) -> anyhow::Result<Option<String>> {
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
        let resolved = address.socket_addrs(|| None)?;
        let address = resolved
            .first()
            .context("dist_init_addr resolved to no addresses")?;
        Ok(Some(format!("dist_init:tcp://{address}")))
    }
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

pub async fn connect(
    uri: &GrpcEndpoint,
    cfg: &GrpcTransportConfig,
    deadline: Instant,
) -> Result<Client, DynamoError> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|err| invalid_arg(format!("invalid SGLang gRPC endpoint `{uri}`: {err}")))?;
    let mut last_err;
    loop {
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
                tokio::time::sleep_until((Instant::now() + cfg.retry_interval).min(deadline)).await;
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
        .map_err(|e| e.to_string())?;
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
    pub async fn connect(
        uri: &GrpcEndpoint,
        cfg: &GrpcTransportConfig,
        deadline: Instant,
    ) -> Result<Self, DynamoError> {
        let size = cfg.connections.get();
        let mut clients = Vec::with_capacity(size);
        for _ in 0..size {
            clients.push(connect(uri, cfg, deadline).await?);
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
    if json_u32(&server_info, "node_rank").is_some_and(|rank| rank > 0) {
        return Err(invalid_arg(
            "full sidecar requires node_rank=0; use --telemetry-only for a follower with local KV sources",
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

    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::time::Instant;
    use tonic::transport::Endpoint;

    use super::{
        NodeMetadata, client_from_channel, discover, discovery_mode, json_u32, json_u64,
        parse_discovery,
    };
    use crate::proto as pb;

    #[test]
    fn numeric_discovery_fields_accept_numbers_and_strings() {
        let value = json!({"a": 16, "b": "32", "c": -1});
        assert_eq!(json_u64(&value, "a"), Some(16));
        assert_eq!(json_u32(&value, "b"), Some(32));
        assert_eq!(json_u64(&value, "c"), None);
    }

    #[test]
    fn discovery_preserves_distinct_tokenizer_path() {
        let discovery = parse_discovery(
            pb::GetModelInfoResponse {
                model_path: "model-repo".to_string(),
                json_info: json!({"tokenizer_path": "tokenizer-repo"}).to_string(),
            },
            json!({}),
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

    #[test]
    fn parses_node_local_sources_and_ignores_unrelated_server_fields() {
        let mut raw = node_metadata_json();
        raw["model_path"] = json!("model-repo");
        // Live-only relaying does not interpret the engine's optional replay field.
        raw["kv_event_sources"][0]["replay_endpoint"] = json!("unused");
        let metadata = NodeMetadata::from_server_info(&raw).unwrap().unwrap();
        assert_eq!(metadata.kv_event_sources[0].dp_rank, 4);
        assert_eq!(
            metadata.worker_group_id().unwrap().as_deref(),
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

    #[test]
    fn normalizes_ipv6_group_id_and_accepts_bound_ipc_sources() {
        let mut raw = node_metadata_json();
        raw["dist_init_addr"] = json!("tcp://[::1]:2345");
        raw["kv_event_sources"][0]["endpoint"] = json!("ipc:///engine/kv-events");
        let metadata = NodeMetadata::from_server_info(&raw).unwrap().unwrap();
        assert_eq!(
            metadata.worker_group_id().unwrap().as_deref(),
            Some("dist_init:tcp://[::1]:2345")
        );
    }

    #[test]
    fn discovery_mode_reads_only_server_metadata() {
        use dynamo_backend_common::DisaggregationMode;

        assert_eq!(
            discovery_mode(&json!({})).unwrap(),
            DisaggregationMode::Aggregated
        );
        assert_eq!(
            discovery_mode(&json!({"disaggregation_mode": "prefill"})).unwrap(),
            DisaggregationMode::Prefill
        );
        assert_eq!(
            discovery_mode(&json!({"disaggregation_mode": "decode"})).unwrap(),
            DisaggregationMode::Decode
        );
        assert!(discovery_mode(&json!({"disaggregation_mode": "unknown"})).is_err());
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

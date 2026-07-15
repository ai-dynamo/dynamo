// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for SGLang's native `sglang.runtime.v1.SglangService`.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use serde_json::Value;
use tokio::time::{Instant, timeout_at};
use tonic::transport::{Channel, Endpoint};

use crate::args::TransportConfig;
use crate::proto as pb;
use crate::proto::sglang_service_client::SglangServiceClient;

const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

pub type Client = SglangServiceClient<Channel>;

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
    uri: &str,
    cfg: &TransportConfig,
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
                        cfg.deadline
                    )));
                }
                tokio::time::sleep_until((Instant::now() + cfg.poll_interval).min(deadline)).await;
            }
        }
    }
}

async fn try_connect_once(
    endpoint: &Endpoint,
    cfg: &TransportConfig,
    deadline: Instant,
) -> Result<Client, String> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err("startup deadline elapsed".to_string());
    }
    let endpoint = endpoint
        .clone()
        .connect_timeout(cfg.connect_timeout.min(remaining));
    let channel = timeout_at(deadline, endpoint.connect())
        .await
        .map_err(|_| "startup deadline elapsed while connecting".to_string())?
        .map_err(|e| e.to_string())?;
    Ok(client_from_channel(channel))
}

fn client_from_channel(channel: Channel) -> Client {
    SglangServiceClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE)
}

/// Fixed-size pool of independent HTTP/2 connections. Generation calls are
/// round-robined so high concurrency does not funnel through one codec task.
pub struct Pool {
    clients: Vec<Client>,
    next: AtomicUsize,
}

impl Pool {
    pub async fn connect(
        uri: &str,
        cfg: &TransportConfig,
        size: usize,
        deadline: Instant,
    ) -> Result<Self, DynamoError> {
        let size = size.max(1);
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
    let model = rpc_with_deadline(
        "GetModelInfo",
        deadline,
        client.get_model_info(pb::GetModelInfoRequest {}),
    )
    .await?
    .into_inner();
    let server = rpc_with_deadline(
        "GetServerInfo",
        deadline,
        client.get_server_info(pb::GetServerInfoRequest {}),
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

    parse_discovery(model, server, models)
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
    server: pb::GetServerInfoResponse,
    models: Vec<pb::ModelCard>,
) -> Result<Discovery, DynamoError> {
    let model_info = parse_json_object("GetModelInfo.json_info", &model.json_info)?;
    let server_info = parse_json_object("GetServerInfo.json_info", &server.json_info)?;
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

fn connection_timeout(message: impl Into<String>) -> DynamoError {
    backend(BackendError::ConnectionTimeout, message)
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
    use tokio::time::{Instant, timeout};
    use tonic::transport::Endpoint;

    use super::{client_from_channel, connect, discover, json_u32, json_u64, parse_discovery};
    use crate::args::TransportConfig;
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
            pb::GetServerInfoResponse {
                json_info: json!({}).to_string(),
            },
            Vec::new(),
        )
        .unwrap();
        assert_eq!(discovery.model_path, "model-repo");
        assert_eq!(discovery.tokenizer_path, "tokenizer-repo");
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

    #[tokio::test]
    async fn malformed_endpoint_fails_before_retrying() {
        let transport = TransportConfig {
            poll_interval: Duration::from_secs(5),
            deadline: Duration::from_secs(30),
            ..TransportConfig::default()
        };
        let result = timeout(
            Duration::from_secs(1),
            connect("http://", &transport, Instant::now() + transport.deadline),
        )
        .await
        .expect("invalid endpoint should not enter the retry loop");
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("invalid endpoint unexpectedly connected"),
        };

        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }
}

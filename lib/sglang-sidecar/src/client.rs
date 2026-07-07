// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for SGLang's native `sglang.runtime.v1.SglangService`.

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use serde_json::Value;
use tokio::time::Instant;
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
    pub served_model_name: Option<String>,
    pub max_model_len: Option<u32>,
    pub model_info: Value,
    pub server_info: Value,
}

pub async fn connect(uri: &str, cfg: &TransportConfig) -> Result<Client, DynamoError> {
    let deadline = Instant::now() + cfg.deadline;
    let mut last_err;
    loop {
        match try_connect_once(uri, cfg).await {
            Ok(client) => return Ok(client),
            Err(err) => {
                last_err = err;
                if Instant::now() >= deadline {
                    return Err(cannot_connect(format!(
                        "could not reach SGLang gRPC at {uri} within {:?}: {last_err}",
                        cfg.deadline
                    )));
                }
                tokio::time::sleep(cfg.poll_interval).await;
            }
        }
    }
}

async fn try_connect_once(uri: &str, cfg: &TransportConfig) -> Result<Client, String> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|e| format!("invalid endpoint `{uri}`: {e}"))?
        .connect_timeout(cfg.connect_timeout);
    let channel = endpoint.connect().await.map_err(|e| e.to_string())?;
    Ok(SglangServiceClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE))
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
    ) -> Result<Self, DynamoError> {
        let size = size.max(1);
        let mut clients = Vec::with_capacity(size);
        for _ in 0..size {
            clients.push(connect(uri, cfg).await?);
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

pub async fn discover(client: &mut Client) -> Result<Discovery, DynamoError> {
    let model = client
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .map_err(|status| status_to_dynamo("GetModelInfo", status))?
        .into_inner();
    let server = client
        .get_server_info(pb::GetServerInfoRequest {})
        .await
        .map_err(|status| status_to_dynamo("GetServerInfo", status))?
        .into_inner();
    let models = client
        .list_models(pb::ListModelsRequest {})
        .await
        .map_err(|status| status_to_dynamo("ListModels", status))?
        .into_inner()
        .models;

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
    use serde_json::json;

    use super::{json_u32, json_u64};

    #[test]
    fn numeric_discovery_fields_accept_numbers_and_strings() {
        let value = json!({"a": 16, "b": "32", "c": -1});
        assert_eq!(json_u64(&value, "a"), Some(16));
        assert_eq!(json_u32(&value, "b"), Some(32));
        assert_eq!(json_u64(&value, "c"), None);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin sidecar client for SGLang's native `sglang.runtime.v1.SglangService`.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcEndpoint, GrpcTransportConfig, format_error_chain,
};
use serde_json::Value;
use tokio::time::{Instant, timeout_at};
use tonic::transport::{Channel, Endpoint};

use crate::proto as pb;
use crate::proto::sglang_service_client::SglangServiceClient;

pub type Client = SglangServiceClient<Channel>;

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
        client_from_channel, discover, json_u32, json_u64, parse_discovery, rpc_with_deadline,
        status_to_dynamo,
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
            pb::GetServerInfoResponse {
                json_info: json!({"incremental_streaming_output": true}).to_string(),
            },
            Vec::new(),
        )
        .unwrap();
        assert_eq!(discovery.model_path, "model-repo");
        assert_eq!(discovery.tokenizer_path, "tokenizer-repo");
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
                pb::GetServerInfoResponse {
                    json_info: info.to_string(),
                },
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
                let error = parse_discovery(
                    pb::GetModelInfoResponse {
                        model_path: "model".into(),
                        json_info: if label.starts_with("GetModelInfo") {
                            raw
                        } else {
                            "{}"
                        }
                        .into(),
                    },
                    pb::GetServerInfoResponse {
                        json_info: if label.starts_with("GetServerInfo") {
                            raw.into()
                        } else {
                            json!({"incremental_streaming_output": true}).to_string()
                        },
                    },
                    vec![],
                )
                .unwrap_err();
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
                    pb::GetServerInfoResponse {
                        json_info: json!({"incremental_streaming_output": true}).to_string(),
                    },
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
                    pb::GetServerInfoResponse {
                        json_info: server_info.to_string(),
                    },
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
                pb::GetServerInfoResponse {
                    json_info: json!({
                        "incremental_streaming_output": true,
                        "served_model_name": "",
                        "context_length": context,
                        "max_req_input_len": input_limit,
                    })
                    .to_string(),
                },
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

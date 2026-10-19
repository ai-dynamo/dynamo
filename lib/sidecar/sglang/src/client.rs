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
use tonic::metadata::{Ascii, MetadataValue};
use tonic::service::{Interceptor, interceptor::InterceptedService};
use tonic::transport::{Channel, Endpoint};

use crate::proto as pb;
use crate::proto::sglang_service_client::SglangServiceClient;

pub type Client = SglangServiceClient<InterceptedService<Channel, ApiKeyAuth>>;

#[derive(Clone, Default)]
pub struct ApiKeyAuth(Option<MetadataValue<Ascii>>);

impl ApiKeyAuth {
    pub fn new(api_key: Option<&str>) -> Result<Self, DynamoError> {
        let authorization = api_key
            .map(|key| {
                let mut value = format!("Bearer {key}")
                    .parse::<MetadataValue<Ascii>>()
                    .map_err(|_| invalid_arg("SGLang API key is not valid gRPC metadata"))?;
                value.set_sensitive(true);
                Ok::<_, DynamoError>(value)
            })
            .transpose()?;
        Ok(Self(authorization))
    }
}

impl Interceptor for ApiKeyAuth {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        if let Some(value) = &self.0 {
            request
                .metadata_mut()
                .insert("authorization", value.clone());
        }
        Ok(request)
    }
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

#[derive(Clone, Debug)]
pub struct EngineState {
    pub instance_id: u64,
    pub revision: u64,
    pub healthy: bool,
    pub is_pause: bool,
    pub discovery: Discovery,
}

/// `bootstrap`: true when called before `dynamo_backend_common::run` installs
/// the global tracing subscriber (the `bootstrap_discover` path during
/// `from_args()`), false once running inside `LLMEngine::start` (via
/// `Pool::connect`) where the subscriber is live.
pub async fn connect(
    uri: &GrpcEndpoint,
    cfg: &GrpcTransportConfig,
    deadline: Instant,
    bootstrap: bool,
    auth: &ApiKeyAuth,
) -> Result<Client, DynamoError> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|err| invalid_arg(format!("invalid SGLang gRPC endpoint `{uri}`: {err}")))?;
    let started = Instant::now();
    let mut attempt = 0_u64;
    let mut last_err;
    let mut last_logged_at: Option<Instant> = None;
    loop {
        attempt += 1;
        match try_connect_once(&endpoint, cfg, deadline, auth).await {
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
                    // eprintln! on the bootstrap path: tracing events emitted before
                    // dynamo_backend_common::run() installs the global subscriber are
                    // silently dropped, which would make this warning exactly as
                    // invisible as the debug! it replaced. On the post-init path
                    // (Pool::connect, called from LLMEngine::start), route through
                    // tracing like everything else so the line gets levels,
                    // timestamps, and filtering.
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
    auth: &ApiKeyAuth,
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
    Ok(client_from_channel(channel, auth.clone()))
}

fn client_from_channel(channel: Channel, auth: ApiKeyAuth) -> Client {
    SglangServiceClient::with_interceptor(channel, auth)
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
        auth: &ApiKeyAuth,
    ) -> Result<Self, DynamoError> {
        let size = cfg.connections.get();
        let mut clients = Vec::with_capacity(size);
        for _ in 0..size {
            clients.push(connect(uri, cfg, deadline, false, auth).await?);
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
    let models = list_models(client, deadline).await?;

    parse_discovery(model, server, models)
}

async fn list_models(
    client: &mut Client,
    deadline: Instant,
) -> Result<Vec<pb::ModelCard>, DynamoError> {
    Ok(rpc_with_deadline(
        "ListModels",
        deadline,
        client.list_models(pb::ListModelsRequest {}),
    )
    .await?
    .into_inner()
    .models)
}

/// Complete startup metadata with the model context window, not the scheduler's
/// input-token limit. Lifecycle observations do not need to fetch this model list.
pub async fn discover_engine_state(
    client: &mut Client,
    snapshot: pb::EngineStateSnapshot,
    deadline: Instant,
) -> Result<EngineState, DynamoError> {
    parse_engine_state_with_models(snapshot, list_models(client, deadline).await?)
}

pub async fn watch_engine_state(
    client: &mut Client,
    deadline: Instant,
) -> Result<tonic::Streaming<pb::EngineStateSnapshot>, DynamoError> {
    rpc_with_deadline(
        "WatchEngineState",
        deadline,
        client.watch_engine_state(pb::WatchEngineStateRequest {}),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// Point-in-time check, not a tensor checksum or a per-request version fence.
/// Bracket a fresh GetModelInfo with instance snapshots so a restarted engine
/// cannot inherit approval just because it reports the same weight version.
pub async fn verify_admission(
    client: &mut Client,
    expected: &dynamo_backend_common::admission::AdmissionIdentity,
    deadline: Instant,
) -> Result<(), DynamoError> {
    async fn check_instance(
        client: &mut Client,
        expected: u64,
        deadline: Instant,
    ) -> Result<(), DynamoError> {
        let mut stream = watch_engine_state(client, deadline).await?;
        let snapshot = rpc_with_deadline("WatchEngineState snapshot", deadline, stream.message())
            .await?
            .ok_or_else(|| {
                protocol_error("engine-state stream closed before admission snapshot")
            })?;
        let state = parse_engine_state(snapshot)?;
        if state.instance_id != expected || !state.healthy || state.is_pause {
            return Err(invalid_arg(
                "admission requires the expected healthy, unpaused engine instance",
            ));
        }
        Ok(())
    }

    check_instance(client, expected.engine_instance_id, deadline).await?;
    let model = rpc_with_deadline(
        "GetModelInfo admission",
        deadline,
        client.get_model_info(pb::GetModelInfoRequest {}),
    )
    .await?
    .into_inner();
    let info = parse_json_object("GetModelInfo.json_info", &model.json_info)?;
    if expected.weight_version.trim().is_empty()
        || info.get("weight_version").and_then(Value::as_str)
            != Some(expected.weight_version.as_str())
    {
        return Err(invalid_arg(
            "live engine weight_version does not match controller admission",
        ));
    }
    check_instance(client, expected.engine_instance_id, deadline).await
}

pub fn parse_engine_state(snapshot: pb::EngineStateSnapshot) -> Result<EngineState, DynamoError> {
    parse_engine_state_with_models(snapshot, Vec::new())
}

pub(crate) fn parse_engine_state_with_models(
    snapshot: pb::EngineStateSnapshot,
    models: Vec<pb::ModelCard>,
) -> Result<EngineState, DynamoError> {
    if snapshot.instance_id == 0 {
        return Err(protocol_error(
            "SGLang WatchEngineState returned a zero instance_id",
        ));
    }
    if snapshot.revision == 0 {
        return Err(protocol_error(
            "SGLang WatchEngineState returned a zero revision",
        ));
    }
    let model = snapshot
        .model_info
        .ok_or_else(|| protocol_error("SGLang engine state omitted model_info"))?;
    let server = snapshot
        .server_info
        .ok_or_else(|| protocol_error("SGLang engine state omitted server_info"))?;
    Ok(EngineState {
        instance_id: snapshot.instance_id,
        revision: snapshot.revision,
        healthy: snapshot.healthy,
        is_pause: snapshot.is_pause,
        discovery: parse_discovery(model, server, models)?,
    })
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

    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::time::Instant;
    use tonic::transport::Endpoint;

    use super::{
        client_from_channel, discover, discover_engine_state, json_u32, json_u64, parse_discovery,
        parse_engine_state,
    };
    use crate::proto as pb;

    #[test]
    fn credentials_are_sensitive_and_invalid_metadata_fails_without_echoing_it() {
        use tonic::service::Interceptor;
        let mut auth = super::ApiKeyAuth::new(Some("local-secret")).unwrap();
        let request = auth.call(tonic::Request::new(())).unwrap();
        let value = request.metadata().get("authorization").unwrap();
        assert_eq!(value, "Bearer local-secret");
        assert!(value.is_sensitive());
        assert!(
            super::ApiKeyAuth::default()
                .call(tonic::Request::new(()))
                .unwrap()
                .metadata()
                .is_empty()
        );
        let error = match super::ApiKeyAuth::new(Some("secret\ninvalid")) {
            Ok(_) => panic!("invalid credential accepted"),
            Err(error) => error.to_string(),
        };
        assert!(!error.contains("secret"));
    }

    #[derive(Clone)]
    struct AuthProbe;

    impl tonic::server::NamedService for AuthProbe {
        const NAME: &'static str = "sglang.runtime.v1.SglangService";
    }

    impl tonic::codegen::Service<tonic::codegen::http::Request<tonic::body::Body>> for AuthProbe {
        type Response = tonic::codegen::http::Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(
            &mut self,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(
            &mut self,
            request: tonic::codegen::http::Request<tonic::body::Body>,
        ) -> Self::Future {
            let status = if request
                .headers()
                .get("authorization")
                .is_some_and(|value| value == "Bearer wire-key")
            {
                tonic::Status::unimplemented("authenticated request reached the service")
            } else {
                tonic::Status::unauthenticated("Unauthorized")
            };
            std::future::ready(Ok(status.into_http()))
        }
    }

    #[tokio::test]
    async fn bootstrap_and_pool_clients_authenticate_all_rpc_shapes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = dynamo_sidecar_common::GrpcEndpoint::parse(
            &format!("http://{}", listener.local_addr().unwrap()),
            "test endpoint",
        )
        .unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            let result = listener.accept().await.map(|(socket, _)| socket);
            Some((result, listener))
        });
        let cancel = tokio_util::sync::CancellationToken::new();
        let server_cancel = cancel.clone();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(AuthProbe)
                .serve_with_incoming_shutdown(incoming, server_cancel.cancelled())
                .await
                .unwrap();
        });
        let transport = dynamo_sidecar_common::GrpcTransportConfig::default();
        for key in [None, Some("wrong"), Some("wire-key")] {
            let auth = super::ApiKeyAuth::new(key).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            let bootstrap = super::connect(&endpoint, &transport, deadline, true, &auth)
                .await
                .unwrap();
            let pool = super::Pool::connect(&endpoint, &transport, deadline, &auth)
                .await
                .unwrap();
            let expected = if key == Some("wire-key") {
                tonic::Code::Unimplemented
            } else {
                tonic::Code::Unauthenticated
            };
            for mut client in [bootstrap, pool.control_client(), pool.stream_client()] {
                assert_eq!(
                    client
                        .get_server_info(pb::GetServerInfoRequest {})
                        .await
                        .unwrap_err()
                        .code(),
                    expected
                );
                assert_eq!(
                    client
                        .get_model_info(pb::GetModelInfoRequest {})
                        .await
                        .unwrap_err()
                        .code(),
                    expected
                );
                assert_eq!(
                    client
                        .generate(pb::GenerateRequest::default())
                        .await
                        .unwrap_err()
                        .code(),
                    expected
                );
                assert_eq!(
                    client
                        .watch_engine_state(pb::WatchEngineStateRequest {})
                        .await
                        .unwrap_err()
                        .code(),
                    expected
                );
                assert_eq!(
                    client
                        .abort(pb::AbortRequest::default())
                        .await
                        .unwrap_err()
                        .code(),
                    expected
                );
            }
        }
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap();
    }

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
        let error = parse_discovery(
            pb::GetModelInfoResponse {
                model_path: "model-repo".to_string(),
                json_info: "{}".to_string(),
            },
            pb::GetServerInfoResponse {
                json_info: json!({"incremental_streaming_output": false}).to_string(),
            },
            Vec::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--incremental-streaming-output"), "{error}");
    }

    #[test]
    fn engine_state_requires_an_instance_and_full_discovery_snapshot() {
        let snapshot = pb::EngineStateSnapshot {
            instance_id: 42,
            revision: 7,
            healthy: true,
            is_pause: true,
            model_info: Some(pb::GetModelInfoResponse {
                model_path: "model-repo".to_string(),
                json_info: json!({"tokenizer_path": "tokenizer-repo"}).to_string(),
            }),
            server_info: Some(pb::GetServerInfoResponse {
                json_info: json!({"incremental_streaming_output": true}).to_string(),
            }),
        };
        let state = parse_engine_state(snapshot).unwrap();
        assert_eq!(state.instance_id, 42);
        assert_eq!(state.revision, 7);
        assert!(state.healthy);
        assert!(state.is_pause);
        assert_eq!(state.discovery.tokenizer_path, "tokenizer-repo");

        assert!(parse_engine_state(pb::EngineStateSnapshot::default()).is_err());
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
        let mut client = client_from_channel(channel, super::ApiKeyAuth::default());
        let started = Instant::now();
        let result = discover(&mut client, started + Duration::from_millis(100)).await;
        let startup_error = discover_engine_state(
            &mut client,
            pb::EngineStateSnapshot::default(),
            Instant::now() + Duration::from_millis(100),
        )
        .await
        .unwrap_err();
        peer.abort();

        assert!(result.is_err());
        assert!(startup_error.to_string().contains("ListModels"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}

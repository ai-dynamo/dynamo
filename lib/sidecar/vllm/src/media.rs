// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{num::NonZeroUsize, time::Duration};

use dynamo_backend_common::{
    BidirectionalEngine, CommonArgs, DisaggregationMode, DynamoError, EngineConfig,
    GenerateContext, HEALTH_CHECK_KEY, ModelInput, RuntimeConfig, WorkerConfig,
};
use dynamo_realtime::{Client, Limits, Protocol};
use dynamo_sidecar_common::{EngineBootstrapResult, HttpEndpoint, invalid_argument};
use futures::{StreamExt, stream::BoxStream};
use serde_json::{Value, json};

#[derive(clap::Subcommand)]
pub(crate) enum Mode {
    /// Serve /v1/realtime through an upstream vLLM or vLLM-Omni WebSocket.
    Realtime(RealtimeArgs),
    /// Serve /v1/audio/speech through an upstream vLLM-Omni HTTP service.
    Speech(crate::speech::SpeechArgs),
}

fn http_endpoint(raw: &str) -> Result<HttpEndpoint, String> {
    HttpEndpoint::parse(raw, "--http-endpoint").map_err(|error| error.to_string())
}

/// Runtime/bootstrap options shared by ASR, Omni, and TTS. No model-name
/// heuristics select protocols or engine sampling/preprocessing settings.
#[derive(clap::Args)]
pub(crate) struct MediaArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    /// Upstream HTTP base URL, including any reverse-proxy prefix.
    #[arg(long, env = "DYN_SIDECAR_HTTP_ENDPOINT", value_parser = http_endpoint)]
    pub http_endpoint: HttpEndpoint,
    /// Exact served model ID advertised by upstream /v1/models.
    #[arg(long)]
    pub model: String,
    /// API key for the upstream service.
    #[arg(long, env = "DYN_SIDECAR_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Maximum wait for engine startup (seconds).
    #[arg(long, default_value_t = 1800, value_parser = clap::value_parser!(u64).range(1..))]
    pub startup_deadline_secs: u64,
    /// Delay between engine startup health probes (seconds).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
    pub startup_retry_secs: u64,
    /// Maximum connection/handshake time (seconds).
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
    pub connect_timeout_secs: u64,
    /// Maximum post-commit turn duration or TTS idle interval (seconds).
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u64).range(1..))]
    pub response_timeout_secs: u64,
}

impl MediaArgs {
    pub(crate) fn validate(&self, endpoint: &str) -> Result<(), DynamoError> {
        if self.model.trim().is_empty()
            || self.common.endpoint_types != endpoint
            || self.common.disaggregation_mode != DisaggregationMode::Aggregated
            || self.common.route_to_encoder
            || self.common.enable_rl
            || self.common.custom_jinja_template.is_some()
            || self.common.dyn_tool_call_parser.is_some()
            || self.common.dyn_reasoning_parser.is_some()
        {
            return Err(invalid_argument(format!(
                "media sidecars require a nonempty model, only the {endpoint} endpoint, and aggregated mode without token-pipeline options"
            )));
        }
        let now = std::time::Instant::now();
        for seconds in [self.startup_deadline_secs, self.startup_retry_secs] {
            if now.checked_add(Duration::from_secs(seconds)).is_none() {
                return Err(invalid_argument("startup timeout is too large"));
            }
        }
        Ok(())
    }

    pub(crate) fn client(&self, mut limits: Limits) -> Result<Client, DynamoError> {
        limits.connect_timeout = Duration::from_secs(self.connect_timeout_secs);
        limits.response_timeout = Duration::from_secs(self.response_timeout_secs);
        Client::new(
            self.http_endpoint
                .as_str()
                .parse()
                .map_err(|_| invalid_argument("invalid HTTP endpoint"))?,
            self.model.clone(),
            limits,
            self.api_key.as_deref(),
        )
        .map_err(to_dynamo)
    }

    pub(crate) fn worker_config(&self) -> WorkerConfig {
        WorkerConfig {
            runtime: self.common.runtime.clone(),
            namespace: self.common.namespace.clone(),
            component: self.common.component.clone(),
            endpoint: self.common.endpoint.clone(),
            endpoint_types: self.common.endpoint_types.clone(),
            model_name: self.model.clone(),
            served_model_name: Some(self.model.clone()),
            model_input: ModelInput::Text,
            enable_local_indexer: false,
            enable_kv_routing: false,
            ..WorkerConfig::default()
        }
    }

    pub(crate) async fn wait_ready(
        &self,
        client: &Client,
        protocol: Option<Protocol>,
    ) -> Result<(), DynamoError> {
        let deadline = tokio::time::Instant::now()
            .checked_add(Duration::from_secs(self.startup_deadline_secs))
            .ok_or_else(|| invalid_argument("startup deadline is too large"))?;
        let probe = async {
            loop {
                let result = match protocol {
                    Some(protocol) => client.realtime_health(protocol).await,
                    None => client.health().await,
                };
                match result {
                    Ok(()) => return Ok(()),
                    Err(
                        error @ (dynamo_realtime::Error::Protocol(_)
                        | dynamo_realtime::Error::Invalid(_)),
                    ) => return Err(to_dynamo(error)),
                    Err(error) => tracing::debug!(%error, "waiting for upstream media service"),
                }
                tokio::time::sleep(Duration::from_secs(self.startup_retry_secs)).await;
            }
        };
        tokio::time::timeout_at(deadline, probe)
            .await
            .map_err(|_| {
                dynamo_sidecar_common::connection_timeout(
                    "upstream media service did not become ready",
                )
            })?
    }
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum RealtimeProtocol {
    Asr,
    Omni,
    OmniDuplex,
}

impl From<RealtimeProtocol> for Protocol {
    fn from(value: RealtimeProtocol) -> Self {
        match value {
            RealtimeProtocol::Asr => Self::VllmTranscription,
            RealtimeProtocol::Omni => Self::Omni,
            RealtimeProtocol::OmniDuplex => Self::OmniDuplex,
        }
    }
}

#[derive(clap::Args)]
pub(crate) struct RealtimeArgs {
    #[command(flatten)]
    pub media: MediaArgs,
    /// Explicit upstream wire contract: vLLM transcription or Omni OpenAI turns.
    #[arg(long, value_enum)]
    pub protocol: RealtimeProtocol,
    /// Concurrent upstream turns per session (default: ASR 1, Omni 8).
    #[arg(long)]
    pub concurrent_turns: Option<NonZeroUsize>,
    /// Additional queued turns per session (default: ASR 1, Omni 0).
    #[arg(long)]
    pub queued_turns: Option<usize>,
    /// Maximum events buffered per turn.
    #[arg(long)]
    pub event_capacity: Option<NonZeroUsize>,
    /// Maximum decoded PCM16 bytes in one append.
    #[arg(long)]
    pub max_audio_chunk_bytes: Option<NonZeroUsize>,
    /// Maximum decoded PCM16 input bytes per turn.
    #[arg(long)]
    pub max_turn_audio_bytes: Option<NonZeroUsize>,
    /// Maximum buffered response bytes per turn.
    #[arg(long)]
    pub max_buffered_output_bytes: Option<NonZeroUsize>,
}

pub(crate) struct RealtimeEngine {
    client: Client,
    model: String,
    protocol: Protocol,
}

impl RealtimeEngine {
    pub(crate) fn bootstrap(
        args: RealtimeArgs,
    ) -> Result<
        (
            RuntimeConfig,
            impl Future<Output = EngineBootstrapResult<Self>>,
        ),
        DynamoError,
    > {
        args.media.validate("realtime")?;
        let protocol = args.protocol.into();
        if protocol == Protocol::OmniDuplex
            && (args.concurrent_turns.is_some()
                || args.queued_turns.is_some()
                || args.max_audio_chunk_bytes.is_some()
                || args.max_turn_audio_bytes.is_some())
        {
            return Err(invalid_argument(
                "native duplex owns its audio and turn limits; use upstream configuration",
            ));
        }
        let mut limits = Limits::for_protocol(protocol);
        if let Some(value) = args.concurrent_turns {
            limits.concurrent_turns = value;
        }
        if let Some(value) = args.queued_turns {
            limits.queued_turns = value;
        }
        if let Some(value) = args.event_capacity {
            limits.event_capacity = value;
        }
        if let Some(value) = args.max_audio_chunk_bytes {
            limits.max_audio_chunk_bytes = value;
        }
        if let Some(value) = args.max_turn_audio_bytes {
            limits.max_turn_audio_bytes = value;
        }
        if let Some(value) = args.max_buffered_output_bytes {
            limits.max_buffered_output_bytes = value;
        }
        let client = args.media.client(limits)?;
        let config = args.media.worker_config();
        Ok((config.runtime.clone(), async move {
            args.media.wait_ready(&client, Some(protocol)).await?;
            Ok((
                Self {
                    client,
                    model: args.media.model,
                    protocol,
                },
                config,
            ))
        }))
    }
}

#[async_trait::async_trait]
impl BidirectionalEngine for RealtimeEngine {
    async fn start(&self, _: u64) -> Result<EngineConfig, DynamoError> {
        let mut config = registration(&self.model);
        if self.protocol == Protocol::OmniDuplex {
            config
                .runtime_data
                .insert("realtime_protocol".into(), json!("omni-duplex"));
        }
        Ok(config)
    }

    async fn generate(
        &self,
        mut requests: BoxStream<'static, Value>,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<Value, DynamoError>>, DynamoError> {
        let client = self.client.clone();
        let protocol = self.protocol;
        Ok(Box::pin(async_stream::try_stream! {
            let first = tokio::select! {
                _ = ctx.stopped() => None,
                _ = ctx.killed() => None,
                first = requests.next() => first,
            };
            if let Some(first) = first {
                if first.get(HEALTH_CHECK_KEY).and_then(Value::as_bool) == Some(true) {
                    let health = tokio::select! {
                        _ = ctx.stopped() => None,
                        _ = ctx.killed() => None,
                        result = async {
                            if protocol == Protocol::OmniDuplex {
                                // Startup already checked capabilities. Active probes must
                                // not consume one of the engine's live session slots.
                                client.health().await
                            } else {
                                client.realtime_health(protocol).await
                            }
                        } => Some(result),
                    };
                    if let Some(health) = health {
                        health.map_err(to_dynamo)?;
                        yield json!({"healthy": true});
                    }
                } else {
                    let requests = futures::stream::once(async { first }).chain(requests).boxed();
                    let cancel_ctx = ctx.inner_arc();
                    let mut session = client.session_with_cancellation(protocol, requests, async move {
                        tokio::select! {
                            _ = cancel_ctx.stopped() => {},
                            _ = cancel_ctx.killed() => {},
                        }
                    });
                    loop {
                        let event = tokio::select! {
                            _ = ctx.stopped() => None,
                            _ = ctx.killed() => None,
                            event = session.next() => event,
                        };
                        match event {
                            Some(event) => yield event,
                            None => break,
                        }
                    }
                }
            }
        }))
    }

    async fn health_check_payload(&self) -> Result<Option<Value>, DynamoError> {
        Ok(Some(json!({HEALTH_CHECK_KEY: true})))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.client.shutdown().await;
        Ok(())
    }
}

pub(crate) fn registration(model: &str) -> EngineConfig {
    EngineConfig {
        model: model.into(),
        served_model_name: Some(model.into()),
        ..EngineConfig::default()
    }
}

pub(crate) fn to_dynamo(error: dynamo_realtime::Error) -> DynamoError {
    match error {
        dynamo_realtime::Error::Invalid(message) => invalid_argument(message),
        dynamo_realtime::Error::Http(error) => {
            use reqwest::StatusCode;
            match error.status() {
                Some(StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY) => {
                    invalid_argument("upstream rejected media request parameters")
                }
                Some(StatusCode::TOO_MANY_REQUESTS) => DynamoError::builder()
                    .error_type(dynamo_backend_common::ErrorType::WorkerOverloaded)
                    .message("upstream media service is overloaded")
                    .build(),
                Some(StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => {
                    dynamo_sidecar_common::cannot_connect(
                        "upstream media credentials were rejected",
                    )
                }
                _ if error.is_timeout() => {
                    dynamo_sidecar_common::connection_timeout("upstream media request timed out")
                }
                _ if error.is_connect() => dynamo_sidecar_common::cannot_connect(
                    "could not connect to upstream media service",
                ),
                _ => dynamo_sidecar_common::engine_shutdown(error.to_string()),
            }
        }
        dynamo_realtime::Error::Closed => dynamo_sidecar_common::cancelled("media session closed"),
        dynamo_realtime::Error::Timeout => {
            dynamo_sidecar_common::connection_timeout("upstream media request timed out")
        }
        error => dynamo_sidecar_common::engine_shutdown(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::FromArgMatches;

    #[test]
    fn media_modes_parse_without_grpc_and_register_only_their_protocol() {
        for mode in ["realtime", "speech"] {
            let mut argv = vec![
                "sidecar",
                mode,
                "--http-endpoint",
                "http://localhost:8001",
                "--model",
                "test-model",
            ];
            if mode == "realtime" {
                argv.extend(["--protocol", "asr"]);
            }
            let matches = crate::command().try_get_matches_from(argv).unwrap();
            match Mode::from_arg_matches(&matches).unwrap() {
                Mode::Realtime(args) => {
                    assert_eq!(args.media.common.endpoint_types, "realtime");
                    let _bootstrap = RealtimeEngine::bootstrap(args).unwrap();
                }
                Mode::Speech(args) => {
                    let _bootstrap = crate::speech::SpeechEngine::bootstrap(args).unwrap();
                }
            }
        }
        let grpc = crate::command()
            .try_get_matches_from(["sidecar", "--grpc-endpoint", "localhost:50051"])
            .unwrap();
        assert!(grpc.subcommand_name().is_none());
        crate::args::Args::from_arg_matches(&grpc).unwrap();
    }

    #[test]
    fn realtime_rejects_token_pipeline_options_before_runtime_startup() {
        let matches = crate::command()
            .try_get_matches_from([
                "sidecar",
                "realtime",
                "--http-endpoint",
                "http://localhost:8001",
                "--model",
                "test-model",
                "--protocol",
                "omni",
                "--disaggregation-mode",
                "prefill",
            ])
            .unwrap();
        let Mode::Realtime(args) = Mode::from_arg_matches(&matches).unwrap() else {
            unreachable!()
        };
        assert!(RealtimeEngine::bootstrap(args).is_err());
        assert!(
            crate::command()
                .try_get_matches_from([
                    "sidecar",
                    "realtime",
                    "--http-endpoint",
                    "http://localhost:8001",
                    "--model",
                    "test-model",
                ])
                .is_err()
        );
    }

    #[tokio::test]
    async fn realtime_probe_checks_engine_health_model_and_upstream_protocol() {
        use axum::{Json, Router, extract::WebSocketUpgrade, http::StatusCode, routing::get};
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let healthy = Arc::new(AtomicBool::new(true));
        let state = healthy.clone();
        let app = Router::new()
            .route(
                "/health",
                get(move || {
                    let state = state.clone();
                    async move {
                        if state.load(Ordering::SeqCst) {
                            StatusCode::OK
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        }
                    }
                }),
            )
            .route(
                "/v1/models",
                get(|| async { Json(json!({"data": [{"id": "test-model"}]})) }),
            )
            .route(
                "/v1/realtime",
                get(|ws: WebSocketUpgrade| async {
                    ws.on_upgrade(|mut socket| async move {
                        let _ = socket
                            .send(axum::extract::ws::Message::Text(
                                json!({
                                    "type": "session.created", "id": "probe"
                                })
                                .to_string()
                                .into(),
                            ))
                            .await;
                        let _ = socket.recv().await;
                    })
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = Client::new(
            url,
            "test-model".into(),
            Limits::for_protocol(Protocol::VllmTranscription),
            None,
        )
        .unwrap();
        client
            .realtime_health(Protocol::VllmTranscription)
            .await
            .unwrap();
        assert!(client.realtime_health(Protocol::Omni).await.is_err());
        healthy.store(false, Ordering::SeqCst);
        assert!(
            client
                .realtime_health(Protocol::VllmTranscription)
                .await
                .is_err()
        );
        server.abort();
    }

    #[tokio::test]
    async fn omni_realtime_probe_accepts_the_upstream_conversation_greeting() {
        use axum::{Json, Router, extract::WebSocketUpgrade, http::StatusCode, routing::get};

        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/v1/models",
                get(|| async { Json(json!({"data": [{"id": "test-model"}]})) }),
            )
            .route(
                "/v1/realtime",
                get(|ws: WebSocketUpgrade| async {
                    ws.on_upgrade(|mut socket| async move {
                        for event in [
                            json!({"type": "session.created", "session": {}}),
                            json!({"type": "conversation.created", "conversation": {"id": "probe"}}),
                        ] {
                            socket
                                .send(axum::extract::ws::Message::Text(event.to_string().into()))
                                .await
                                .unwrap();
                        }
                        let update = socket.recv().await.unwrap().unwrap();
                        let update: Value = serde_json::from_str(update.to_text().unwrap()).unwrap();
                        assert_eq!(
                            update,
                            json!({"type": "session.update", "session": {
                                "type": "realtime", "model": "test-model"
                            }})
                        );
                        socket
                            .send(axum::extract::ws::Message::Text(
                                json!({"type": "session.updated"}).to_string().into(),
                            ))
                            .await
                            .unwrap();
                        let _ = socket.recv().await;
                    })
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = Client::new(
            url,
            "test-model".into(),
            Limits::for_protocol(Protocol::Omni),
            None,
        )
        .unwrap();
        client.realtime_health(Protocol::Omni).await.unwrap();
        server.abort();
    }
}

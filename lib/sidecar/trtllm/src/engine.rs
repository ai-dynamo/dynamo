// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for TensorRT-LLM's OpenEngine (`openengine.v1`) gRPC server.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, WorkerConfig, usage,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, TrtllmClient};
use crate::convert::{ResponseState, build_generate_request};
use crate::model::ConfiguredModel;

const ALREADY_STARTED: &str = "TensorRT-LLM sidecar has already started";

/// Terminal output emitted when a request is cancelled, carrying the usage
/// accumulated so far.
fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(state.prompt_tokens(), state.completion_tokens()))
}

pub struct TrtllmSidecarEngine {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    model: ConfiguredModel,
    /// Disaggregation role this worker plays. Selects the `context_only` /
    /// `kv.session` divergence in `convert`.
    mode: DisaggregationMode,
    client: OnceCell<TrtllmClient>,
    /// Model context length reported by `Control.GetModelInfo`, cached at
    /// `start` so `generate` can derive a default `max_tokens` for requests
    /// that omit one.
    context_length: OnceCell<u32>,
    cancel: CancellationToken,
}

impl TrtllmSidecarEngine {
    pub(crate) fn new(
        endpoint: GrpcEndpoint,
        transport: GrpcTransportConfig,
        model: ConfiguredModel,
        mode: DisaggregationMode,
    ) -> Self {
        Self {
            endpoint,
            transport,
            model,
            mode,
            client: OnceCell::new(),
            context_length: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::from_parsed(<Args as clap::Parser>::parse())
    }

    pub fn from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)
            .map_err(|err| client::invalid_argument(err.to_string()))?;
        Self::from_parsed(args)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        if args.model_path.trim().is_empty() {
            return Err(client::invalid_argument("model-path must not be empty"));
        }
        let mode = args.sidecar.common.disaggregation_mode;
        if mode.is_encode() {
            return Err(client::invalid_argument(
                "encode mode is not supported by the TensorRT-LLM sidecar",
            ));
        }
        if args.sidecar.common.route_to_encoder {
            return Err(client::invalid_argument(
                "route-to-encoder is not supported by the TensorRT-LLM sidecar",
            ));
        }

        let endpoint = GrpcEndpoint::parse(&args.trtllm_endpoint, "--trtllm-endpoint")?;
        let transport = args.sidecar.grpc.config();
        let model = ConfiguredModel {
            source: args.model_path,
            // Resolved from the server's `Control.GetModelInfo` at `start`.
            context_length: None,
        };
        let engine = Self::new(endpoint, transport, model.clone(), mode);
        let config = WorkerConfig {
            namespace: args.sidecar.common.namespace,
            // Prefill workers register under their own component so the
            // frontend's prefill router can target them separately.
            component: if mode == DisaggregationMode::Aggregated {
                args.sidecar.common.component
            } else {
                mode.discovery_component().to_string()
            },
            endpoint: args.sidecar.common.endpoint,
            endpoint_types: args.sidecar.common.endpoint_types,
            custom_jinja_template: args.sidecar.common.custom_jinja_template,
            model_name: model.source.clone(),
            served_model_name: None,
            tool_call_parser: args.sidecar.common.dyn_tool_call_parser,
            reasoning_parser: args.sidecar.common.dyn_reasoning_parser,
            exclude_tools_when_tool_choice_none: args
                .sidecar
                .common
                .exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: mode,
            route_to_encoder: false,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for TrtllmSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(client::engine_shutdown(ALREADY_STARTED));
        }
        tracing::info!(
            endpoint = %self.endpoint,
            connections = self.transport.connections.get(),
            "connecting to TensorRT-LLM gRPC"
        );
        let client = TrtllmClient::connect(&self.endpoint, self.transport).await?;
        let connection_count = client.connection_count();

        // The server owns the context length: it backs both the registered
        // context window and the default-`max_tokens` path in
        // `convert::max_tokens`. Fail startup rather than serve with an unknown
        // window, which would reject every request that omits `max_tokens`.
        let mut model = self.model.clone();
        let context_length = client
            .model_info(&self.model.source)
            .await
            .map_err(|error| {
                client::engine_shutdown(format!(
                    "Control.GetModelInfo failed, so the model context length is unknown: \
                     {error}. The TensorRT-LLM server must implement the OpenEngine Control \
                     service."
                ))
            })?
            .ok_or_else(|| {
                client::engine_shutdown(
                    "Control.GetModelInfo reported no max_context_length for this model",
                )
            })?;
        model.context_length = Some(context_length);
        let _ = self.context_length.set(context_length);

        self.client
            .set(client)
            .map_err(|_| client::engine_shutdown(ALREADY_STARTED))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections = connection_count,
            model = %model.source,
            context_length,
            "TensorRT-LLM gRPC is ready"
        );
        Ok(model.engine_config())
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("TensorRT-LLM sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        let proto_request = build_generate_request(
            &request,
            &request_id,
            &self.model.source,
            self.context_length.get().copied(),
            self.mode,
        )?;
        let mut state = ResponseState::new(&request, self.mode);
        let cancel = self.cancel.clone();

        let stream = tokio::select! {
            biased;
            _ = ctx.stopped() => None,
            _ = cancel.cancelled() => None,
            result = client.generate(proto_request) => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = cancelled(&state);
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    _ = cancel.cancelled() => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    message = stream.message() => {
                        match message {
                            Ok(Some(response)) => match state.convert(response) {
                                Ok(Some(output)) => {
                                    let terminal = output.finish_reason.is_some();
                                    yield Ok(output);
                                    if terminal {
                                        break;
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            },
                            Ok(None) => {
                                yield Err(client::protocol_error(
                                    "Generate ended before a terminal response",
                                ));
                                break;
                            }
                            Err(status) => {
                                yield Err(client::status_to_dynamo("Generate", status));
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(client) = self.client.get() else {
            return;
        };
        if let Err(error) = client.abort(ctx.id().to_string()).await {
            tracing::warn!(request_id = ctx.id(), %error, "TensorRT-LLM Control.Abort failed");
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("TensorRT-LLM sidecar shutdown complete");
        Ok(())
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, WorkerConfig, usage,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, VllmClient};
use crate::convert::{ResponseState, build_generate_request};
use crate::model::ConfiguredModel;

pub struct VllmSidecarEngine {
    endpoint: GrpcEndpoint,
    model: ConfiguredModel,
    mode: DisaggregationMode,
    transport: GrpcTransportConfig,
    client: OnceCell<VllmClient>,
    cancel: CancellationToken,
    /// When true, `start` surfaces this engine's gRPC endpoint + backend name in
    /// `EngineConfig.runtime_data` so the `--direct` registrar advertises a
    /// `TransportType::Grpc` instance for direct frontend→gRPC dispatch.
    is_direct: bool,
    /// Address the frontend dials, when it differs from the local `--vllm-endpoint`
    /// the sidecar connects to (multi-node). `None` advertises the local endpoint.
    advertise_grpc_endpoint: Option<String>,
}

fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(
        state.prompt_tokens(),
        state.reported_completion_tokens(),
    ))
}

impl VllmSidecarEngine {
    pub(crate) fn new(
        endpoint: GrpcEndpoint,
        model: ConfiguredModel,
        mode: DisaggregationMode,
        transport: GrpcTransportConfig,
    ) -> Self {
        Self {
            endpoint,
            model,
            mode,
            transport,
            client: OnceCell::new(),
            cancel: CancellationToken::new(),
            is_direct: false,
            advertise_grpc_endpoint: None,
        }
    }

    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let parsing_process_args = argv.is_none();
        let parsed = match argv {
            Some(argv) => <Args as clap::Parser>::try_parse_from(argv),
            None => <Args as clap::Parser>::try_parse(),
        };
        let args = match parsed {
            Ok(args) => args,
            Err(error)
                if parsing_process_args
                    && matches!(
                        error.kind(),
                        clap::error::ErrorKind::DisplayHelp
                            | clap::error::ErrorKind::DisplayVersion
                    ) =>
            {
                error.exit()
            }
            Err(error) => return Err(client::invalid_argument(error.to_string())),
        };
        Self::from_parsed(args)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        if args.model_path.trim().is_empty() {
            return Err(client::invalid_argument("model-path must not be empty"));
        }
        if args.sidecar.common.disaggregation_mode.is_encode() {
            return Err(client::invalid_argument(
                "encode mode is not supported by the vLLM sidecar",
            ));
        }
        if args.sidecar.common.route_to_encoder {
            return Err(client::invalid_argument(
                "route-to-encoder is not supported by the vLLM sidecar",
            ));
        }
        if args.sidecar.common.is_direct
            && args.sidecar.common.disaggregation_mode != DisaggregationMode::Aggregated
        {
            return Err(client::invalid_argument(
                "--direct supports aggregated serving only for the vLLM sidecar",
            ));
        }

        let endpoint = GrpcEndpoint::parse(&args.vllm_endpoint, "--vllm-endpoint")?;
        let transport = args.sidecar.grpc.config();
        let model = ConfiguredModel {
            source: args.model_path,
        };
        let mode = args.sidecar.common.disaggregation_mode;
        let mut engine = Self::new(endpoint, model.clone(), mode, transport);
        let (tool_call_parser, reasoning_parser) = if mode.is_prefill() {
            (None, None)
        } else {
            (
                args.sidecar.common.dyn_tool_call_parser,
                args.sidecar.common.dyn_reasoning_parser,
            )
        };
        let config = WorkerConfig {
            namespace: args.sidecar.common.namespace,
            // Prefill/decode must register under fixed role components so the
            // frontend can route the disaggregated handoff; aggregated keeps the
            // operator-configured component (`--component` / `DYN_COMPONENT`).
            component: match mode {
                DisaggregationMode::Aggregated => args.sidecar.common.component,
                _ => mode.discovery_component().to_string(),
            },
            endpoint: args.sidecar.common.endpoint,
            endpoint_types: args.sidecar.common.endpoint_types,
            custom_jinja_template: args.sidecar.common.custom_jinja_template,
            model_name: model.source.clone(),
            served_model_name: None,
            tool_call_parser,
            reasoning_parser,
            exclude_tools_when_tool_choice_none: args
                .sidecar
                .common
                .exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: mode,
            route_to_encoder: false,
            is_direct: args.sidecar.common.is_direct,
            ..Default::default()
        };
        // Mirror the single config flag so `start()` can surface the direct-gRPC
        // facts without threading WorkerConfig into the engine.
        engine.is_direct = config.is_direct;
        engine.advertise_grpc_endpoint = args.sidecar.common.advertise_grpc_endpoint;
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for VllmSidecarEngine {
    async fn start(
        &self,
        _worker_id: u64,
    ) -> Result<dynamo_backend_common::EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(client::engine_shutdown("vLLM sidecar has already started"));
        }
        tracing::info!(
            endpoint = %self.endpoint,
            connections = self.transport.connections,
            mode = %self.mode,
            "connecting to vLLM gRPC"
        );
        let client = VllmClient::connect(&self.endpoint, self.transport).await?;
        let connection_count = client.connection_count();
        self.client
            .set(client)
            .map_err(|_| client::engine_shutdown("vLLM sidecar has already started"))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections = connection_count,
            configured_model_source = %self.model.source,
            mode = %self.mode,
            "vLLM gRPC transport connected"
        );
        let mut engine_config = self.model.engine_config();
        if self.is_direct {
            // Advertise the backend name (frontend selects the vLLM gRPC
            // dispatcher) and the address the frontend dials — the
            // --advertise-grpc-endpoint override for multi-node, else the local
            // endpoint the sidecar connects to.
            let advertised = self
                .advertise_grpc_endpoint
                .clone()
                .unwrap_or_else(|| self.endpoint.as_str().to_string());
            engine_config.runtime_data.insert(
                dynamo_llm::discovery::DIRECT_BACKEND_KEY.to_string(),
                serde_json::Value::String(crate::direct::VLLM_BACKEND.to_string()),
            );
            engine_config.runtime_data.insert(
                dynamo_backend_common::DIRECT_GRPC_ENDPOINT_KEY.to_string(),
                serde_json::Value::String(advertised),
            );
        }
        Ok(engine_config)
    }

    async fn generate(
        &self,
        request: dynamo_backend_common::PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        let mut state = ResponseState::new(&request, self.mode);
        let proto_request = build_generate_request(request, request_id, self.mode)?;
        let stopped_ctx = ctx.inner_arc();
        let shutdown = self.cancel.clone();
        let mut cancellation = Box::pin(async move {
            tokio::select! {
                _ = stopped_ctx.stopped() => {}
                _ = shutdown.cancelled() => {}
            }
        });
        let stream = tokio::select! {
            biased;
            _ = cancellation.as_mut() => None,
            result = client.generate_stream(proto_request) => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = cancelled(&state);
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = cancellation.as_mut() => {
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
                                    "GenerateStream ended before a terminal response",
                                ));
                                break;
                            }
                            Err(status) => {
                                yield Err(client::status_to_dynamo("GenerateStream", status));
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    async fn health_check(&self) -> Result<(), DynamoError> {
        // Direct-mode liveness for the `--direct` orchestrator's health loop.
        // vLLM's gRPC exposes no health or model-info RPC (both HealthCheck and
        // grpc.health return UNIMPLEMENTED), so probe with a cheap fresh channel
        // connect: it succeeds while the engine is listening and fails fast once
        // it dies, letting the orchestrator unregister and re-register on
        // recovery. Not started yet is a failure, same as the other engines.
        if !self.client.initialized() {
            return Err(client::engine_shutdown("vLLM sidecar is not started"));
        }
        client::probe_liveness(&self.endpoint, self.transport).await
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        Ok(())
    }
}

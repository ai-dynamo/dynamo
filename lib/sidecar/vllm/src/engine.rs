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

        let endpoint = GrpcEndpoint::parse(&args.vllm_endpoint, "--vllm-endpoint")?;
        let transport = args.sidecar.grpc.config();
        let model = ConfiguredModel {
            source: args.model_path,
        };
        let mode = args.sidecar.common.disaggregation_mode;
        let engine = Self::new(endpoint, model.clone(), mode, transport);
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
            component: args.sidecar.common.component,
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
            ..Default::default()
        };
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
        Ok(self.model.engine_config())
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

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        Ok(())
    }
}

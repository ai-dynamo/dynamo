// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, normalize_endpoint};
use crate::client::{self, VllmClient};
use crate::convert::{ResponseState, build_generate_request};
use crate::model::ConfiguredModel;

const CONNECTIONS: usize = 8;

pub struct VllmRemoteEngine {
    endpoint: String,
    connections: usize,
    model: ConfiguredModel,
    mode: DisaggregationMode,
    client: OnceCell<VllmClient>,
    cancel: CancellationToken,
}

impl VllmRemoteEngine {
    pub(crate) fn new(
        endpoint: String,
        connections: usize,
        model: ConfiguredModel,
        mode: DisaggregationMode,
    ) -> Self {
        Self {
            endpoint,
            connections,
            model,
            mode,
            client: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::from_parsed(<Args as clap::Parser>::parse())
    }

    pub fn from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)
            .map_err(|error| client::invalid_argument(error.to_string()))?;
        Self::from_parsed(args)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        if args.model_path.trim().is_empty() {
            return Err(client::invalid_argument("model-path must not be empty"));
        }
        if args.common.disaggregation_mode.is_encode() {
            return Err(client::invalid_argument(
                "encode mode is not supported by the vLLM remote backend",
            ));
        }
        if args.common.route_to_encoder {
            return Err(client::invalid_argument(
                "route-to-encoder is not supported by the vLLM remote backend",
            ));
        }

        let endpoint = normalize_endpoint(&args.vllm_endpoint).map_err(client::invalid_argument)?;
        let model = ConfiguredModel {
            source: args.model_path,
        };
        let mode = args.common.disaggregation_mode;
        let engine = Self::new(endpoint, CONNECTIONS, model.clone(), mode);
        let (tool_call_parser, reasoning_parser) = if mode.is_prefill() {
            (None, None)
        } else {
            (
                args.common.dyn_tool_call_parser,
                args.common.dyn_reasoning_parser,
            )
        };
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            model_name: model.source.clone(),
            served_model_name: None,
            tool_call_parser,
            reasoning_parser,
            exclude_tools_when_tool_choice_none: args.common.exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: mode,
            route_to_encoder: false,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for VllmRemoteEngine {
    async fn start(
        &self,
        _worker_id: u64,
    ) -> Result<dynamo_backend_common::EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(client::engine_shutdown(
                "vLLM remote backend has already started",
            ));
        }
        tracing::info!(
            endpoint = %self.endpoint,
            connections = self.connections,
            "connecting to vLLM gRPC"
        );
        let client = VllmClient::connect(&self.endpoint, self.connections).await?;
        let connection_count = client.connection_count();
        self.client
            .set(client)
            .map_err(|_| client::engine_shutdown("vLLM remote backend has already started"))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections = connection_count,
            model = %self.model.source,
            "vLLM gRPC is ready"
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
            .ok_or_else(|| client::engine_shutdown("vLLM remote backend is not started"))?;
        let request_id = ctx.id().to_string();
        let proto_request = build_generate_request(&request, &request_id, self.mode)?;
        let mut state = ResponseState::new(&request, self.mode);
        let cancel = self.cancel.clone();
        let stream = tokio::select! {
            biased;
            _ = ctx.stopped() => None,
            _ = cancel.cancelled() => None,
            result = client.generate_stream(proto_request) => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = LLMEngineOutput::cancelled().with_usage(usage(
                state.prompt_tokens(),
                state.reported_completion_tokens(),
            ));
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(usage(
                            state.prompt_tokens(),
                            state.reported_completion_tokens(),
                        )));
                        break;
                    }
                    _ = cancel.cancelled() => {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(usage(
                            state.prompt_tokens(),
                            state.reported_completion_tokens(),
                        )));
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

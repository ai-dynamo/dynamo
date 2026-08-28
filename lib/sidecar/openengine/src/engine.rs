// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZeroUsize;

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, EngineConfig, GenerateContext, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, LlmRegistration, PreprocessedRequest, WorkerConfig, usage,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig, SidecarStartupError};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::OPENENGINE_SCHEMA_RELEASE;
use crate::args::Args;
use crate::client::{self, OpenEngineClient};
use crate::convert::{self, ResponseState};
use crate::proto as pb;

const SCHEMA_REVISION: u32 = 1;

#[derive(Clone, Debug)]
struct Discovery {
    info: pb::ServerInfo,
    role: pb::EngineRole,
    mode: DisaggregationMode,
    model_source: String,
    remote_model: String,
}

impl Discovery {
    fn from_server_info(
        info: pb::ServerInfo,
        model_path: Option<String>,
    ) -> Result<Self, DynamoError> {
        if info.engine_name.trim().is_empty() {
            return Err(client::protocol_error(
                "GetServerInfo returned an empty engine_name",
            ));
        }
        if info.schema_revision < SCHEMA_REVISION || info.minimum_client_revision > SCHEMA_REVISION
        {
            return Err(client::protocol_error(format!(
                "incompatible schema revisions: server={}, minimum_client={}, client={SCHEMA_REVISION}",
                info.schema_revision, info.minimum_client_revision
            )));
        }
        if info.schema_release != OPENENGINE_SCHEMA_RELEASE {
            return Err(client::protocol_error(format!(
                "server schema_release `{}` does not match `{OPENENGINE_SCHEMA_RELEASE}`",
                info.schema_release
            )));
        }
        let role = pb::EngineRole::try_from(info.engine_role).map_err(|_| {
            client::protocol_error(format!("unknown engine role {}", info.engine_role))
        })?;
        let mode = match role {
            pb::EngineRole::Aggregated => DisaggregationMode::Aggregated,
            pb::EngineRole::Prefill => DisaggregationMode::Prefill,
            pb::EngineRole::Decode => DisaggregationMode::Decode,
            pb::EngineRole::Unspecified => {
                return Err(client::protocol_error(
                    "GetServerInfo returned an unspecified engine role",
                ));
            }
        };
        let [remote_model] = info.supported_models.as_slice() else {
            return Err(client::protocol_error(format!(
                "this initial sidecar requires exactly one supported model; server returned {}",
                info.supported_models.len()
            )));
        };
        if remote_model.trim().is_empty() {
            return Err(client::protocol_error(
                "GetServerInfo returned an empty supported model",
            ));
        }
        if mode != DisaggregationMode::Aggregated {
            let connector = info.kv_connector.as_ref().ok_or_else(|| {
                client::protocol_error("disaggregated server omitted kv_connector")
            })?;
            if connector.enabled != Some(true) || connector.transfer_backend.trim().is_empty() {
                return Err(client::protocol_error(
                    "disaggregated server did not advertise an enabled KV transfer backend",
                ));
            }
        }
        let model_source = model_path.unwrap_or_else(|| remote_model.clone());
        if model_source.trim().is_empty() {
            return Err(client::invalid_argument("model-path must not be empty"));
        }
        Ok(Self {
            remote_model: remote_model.clone(),
            model_source,
            info,
            role,
            mode,
        })
    }

    fn ensure_same_server(&self, observed: &Self) -> Result<(), DynamoError> {
        if self.info.engine_name != observed.info.engine_name
            || self.info.schema_release != observed.info.schema_release
            || self.role != observed.role
            || self.remote_model != observed.remote_model
        {
            return Err(client::protocol_error(
                "OpenEngine identity, role, schema, or model changed during startup",
            ));
        }
        Ok(())
    }

    fn engine_config(&self) -> EngineConfig {
        let mut runtime_data = HashMap::new();
        runtime_data.insert(
            "grpc_service".to_string(),
            serde_json::Value::String("openengine.v1.Inference".to_string()),
        );
        runtime_data.insert(
            "openengine_schema_release".to_string(),
            serde_json::Value::String(self.info.schema_release.clone()),
        );
        runtime_data.insert(
            "engine_name".to_string(),
            serde_json::Value::String(self.info.engine_name.clone()),
        );

        let capacity = self.info.capacity.as_ref();
        let parallelism = self.info.parallelism.as_ref();
        EngineConfig {
            model: self.model_source.clone(),
            served_model_name: Some(self.remote_model.clone()),
            model_aliases: Vec::new(),
            runtime_data,
            llm: Some(LlmRegistration {
                context_length: None,
                kv_cache_block_size: capacity.and_then(|value| value.kv_block_size),
                total_kv_blocks: capacity.and_then(|value| value.total_kv_blocks),
                max_num_seqs: capacity.and_then(|value| value.max_running_requests),
                max_num_batched_tokens: capacity.and_then(|value| value.max_batched_tokens),
                data_parallel_size: parallelism
                    .and_then(|value| value.data_parallel_size)
                    .filter(|value| *value > 0),
                data_parallel_start_rank: parallelism
                    .and_then(|value| value.data_parallel_start_rank.or(value.data_parallel_rank)),
                bootstrap_host: None,
                bootstrap_port: None,
            }),
        }
    }
}

pub struct OpenEngineSidecarEngine {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    discovery: Discovery,
    client: OnceCell<OpenEngineClient>,
    cancel: CancellationToken,
}

impl OpenEngineSidecarEngine {
    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::from_parsed(<Args as clap::Parser>::parse())
    }

    pub fn from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::try_from_args(argv).map_err(SidecarStartupError::into_dynamo)
    }

    pub fn try_from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), SidecarStartupError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)?;
        Self::from_parsed(args).map_err(Into::into)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        if args.sidecar.common.route_to_encoder {
            return Err(client::invalid_argument(
                "route-to-encoder is not supported by the OpenEngine sidecar",
            ));
        }
        if args.sidecar.common.enable_rl {
            return Err(client::invalid_argument(
                "RL control is not implemented by the TensorRT-LLM OpenEngine server",
            ));
        }
        if args
            .model_path
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(client::invalid_argument("model-path must not be empty"));
        }

        let endpoint = args.sidecar.grpc_endpoint;
        let transport = args.sidecar.grpc.config();
        eprintln!(
            "Discovering OpenEngine metadata from {endpoint}; startup deadline: {:?}",
            transport.startup_deadline
        );
        let discovery = bootstrap_discover(&endpoint, transport, args.model_path)?;
        let mode = discovery.mode;
        let engine = Self {
            endpoint,
            transport,
            discovery: discovery.clone(),
            client: OnceCell::new(),
            cancel: CancellationToken::new(),
        };
        let config = WorkerConfig {
            namespace: args.sidecar.common.namespace,
            component: if mode == DisaggregationMode::Aggregated {
                args.sidecar.common.component
            } else {
                mode.discovery_component().to_string()
            },
            endpoint: args.sidecar.common.endpoint,
            endpoint_types: args.sidecar.common.endpoint_types,
            custom_jinja_template: args.sidecar.common.custom_jinja_template,
            model_name: discovery.model_source.clone(),
            served_model_name: Some(discovery.remote_model.clone()),
            tool_call_parser: args.sidecar.common.dyn_tool_call_parser,
            reasoning_parser: args.sidecar.common.dyn_reasoning_parser,
            exclude_tools_when_tool_choice_none: args
                .sidecar
                .common
                .exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: mode,
            route_to_encoder: false,
            enable_rl: false,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for OpenEngineSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(client::engine_shutdown(
                "OpenEngine sidecar has already started",
            ));
        }
        tracing::info!(
            endpoint = %self.endpoint,
            connections = self.transport.connections,
            mode = %self.discovery.mode,
            "connecting to OpenEngine gRPC"
        );
        let client = OpenEngineClient::connect(&self.endpoint, self.transport).await?;
        let info = client
            .server_info(self.transport.connect_attempt_timeout)
            .await?;
        let observed =
            Discovery::from_server_info(info, Some(self.discovery.model_source.clone()))?;
        self.discovery.ensure_same_server(&observed)?;
        let connections = client.connection_count();
        self.client
            .set(client)
            .map_err(|_| client::engine_shutdown("OpenEngine sidecar has already started"))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections,
            engine = %self.discovery.info.engine_name,
            model = %self.discovery.remote_model,
            mode = %self.discovery.mode,
            "OpenEngine gRPC is ready"
        );
        Ok(observed.engine_config())
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("OpenEngine sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        let proto_request = convert::build_generate_request(
            &request,
            &request_id,
            &self.discovery.remote_model,
            self.discovery.mode,
        )?;
        let metadata = convert::generate_metadata(&request, ctx.metadata(), self.discovery.mode)?;
        let mut grpc_request = tonic::Request::new(proto_request);
        *grpc_request.metadata_mut() = metadata;
        let mut state = ResponseState::new(self.discovery.mode, request.token_ids.len() as u32);
        let cancel = self.cancel.clone();

        let stream = tokio::select! {
            biased;
            _ = ctx.stopped() => None,
            _ = cancel.cancelled() => None,
            result = client.generate(grpc_request) => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = LLMEngineOutput::cancelled()
                .with_usage(usage(state.prompt_tokens(), state.completion_tokens()));
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(LLMEngineOutput::cancelled()
                            .with_usage(usage(state.prompt_tokens(), state.completion_tokens())));
                        break;
                    }
                    _ = cancel.cancelled() => {
                        yield Ok(LLMEngineOutput::cancelled()
                            .with_usage(usage(state.prompt_tokens(), state.completion_tokens())));
                        break;
                    }
                    message = stream.message() => {
                        match message {
                            Ok(Some(response)) => match state.convert(response, &request_id) {
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
                                yield Err(client::status_to_dynamo("Generate stream", status));
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
        tracing::info!("OpenEngine sidecar shutdown complete");
        Ok(())
    }
}

fn bootstrap_discover(
    endpoint: &GrpcEndpoint,
    transport: GrpcTransportConfig,
    model_path: Option<String>,
) -> Result<Discovery, DynamoError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| client::engine_shutdown(format!("bootstrap runtime: {error}")))?;
    runtime.block_on(async move {
        let bootstrap_transport = GrpcTransportConfig {
            connections: NonZeroUsize::new(1).expect("one is non-zero"),
            ..transport
        };
        let client = OpenEngineClient::connect(endpoint, bootstrap_transport).await?;
        let info = client
            .server_info(transport.connect_attempt_timeout)
            .await?;
        Discovery::from_server_info(info, model_path)
    })
}

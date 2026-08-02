// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, RlWorkerMetadata, WorkerConfig, rl_enabled, usage,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::stream::BoxStream;
use serde_json::json;
use tokio::sync::{Mutex, OnceCell};
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, VllmClient};
use crate::convert::{ResponseState, build_generate_request};
use crate::discovery::{
    BootstrapIdentity, bootstrap_discover, build_engine_config, inference_world_size, nonempty,
    validate_discovery,
};
use crate::lora::{next_lora_id, parse_load_lora, publish_lora_model};
use crate::model::ConfiguredModel;

pub struct VllmSidecarEngine {
    endpoint: GrpcEndpoint,
    model: ConfiguredModel,
    mode: DisaggregationMode,
    transport: GrpcTransportConfig,
    bootstrap_identity: Option<BootstrapIdentity>,
    client: OnceCell<VllmClient>,
    runtime_endpoint: OnceCell<dynamo_runtime::component::Endpoint>,
    lora_update_lock: Mutex<()>,
    supports_lora: bool,
    max_loras: u32,
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
            bootstrap_identity: None,
            client: OnceCell::new(),
            runtime_endpoint: OnceCell::new(),
            lora_update_lock: Mutex::new(()),
            supports_lora: false,
            max_loras: 0,
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
        let discovery = bootstrap_discover(&endpoint, transport)?;
        validate_discovery(&discovery)?;
        let rl_metadata = if rl_enabled() {
            let admin_endpoint = args.admin_endpoint.as_deref().ok_or_else(|| {
                client::invalid_argument(
                    "DYN_ENABLE_RL requires --admin-endpoint or VLLM_HTTP_ENDPOINT",
                )
            })?;
            let admin_base_url =
                GrpcEndpoint::parse(admin_endpoint, "--admin-endpoint")?.to_string();
            let parallelism = discovery.server.parallelism.as_ref().ok_or_else(|| {
                client::invalid_argument(
                    "vLLM GetServerInfo did not return parallelism for RL discovery",
                )
            })?;
            Some(rl_worker_metadata(
                admin_base_url,
                inference_world_size(parallelism)?,
                &discovery.model,
                args.rl_discovery_model_name.as_deref(),
            )?)
        } else {
            None
        };
        let engine = Self::new(endpoint, model.clone(), mode, transport)
            .with_bootstrap_identity(BootstrapIdentity::from_discovery(&discovery))
            .with_lora_support(discovery.model.supports_lora, discovery.server.max_loras);
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
            served_model_name: nonempty(&discovery.model.served_model_name),
            tool_call_parser,
            reasoning_parser,
            exclude_tools_when_tool_choice_none: args
                .sidecar
                .common
                .exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: mode,
            route_to_encoder: false,
            rl_metadata,
            ..Default::default()
        };
        Ok((engine, config))
    }

    fn with_bootstrap_identity(mut self, identity: BootstrapIdentity) -> Self {
        self.bootstrap_identity = Some(identity);
        self
    }

    fn with_lora_support(mut self, supports_lora: bool, max_loras: u32) -> Self {
        self.supports_lora = supports_lora && max_loras > 0;
        self.max_loras = max_loras;
        self
    }
}

pub(crate) fn rl_worker_metadata(
    admin_base_url: String,
    world_size: u32,
    model: &crate::proto::ModelInfo,
    configured_name: Option<&str>,
) -> Result<RlWorkerMetadata, DynamoError> {
    let model_name = match configured_name.map(str::trim) {
        None => model.model_id.as_str(),
        Some("") => {
            return Err(client::invalid_argument(
                "DYN_RL_DISCOVERY_MODEL_NAME must not be empty",
            ));
        }
        Some(name)
            if name == model.model_id
                || name == model.served_model_name
                || model.served_model_aliases.iter().any(|alias| alias == name) =>
        {
            name
        }
        Some(name) => {
            return Err(client::invalid_argument(format!(
                "RL model name `{name}` is not advertised by vLLM"
            )));
        }
    };
    Ok(RlWorkerMetadata {
        admin_base_url,
        world_size,
        model: model_name.to_string(),
    })
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
        let (client, discovery) =
            VllmClient::connect_and_discover(&self.endpoint, self.transport).await?;
        validate_discovery(&discovery)?;
        if let Some(identity) = &self.bootstrap_identity {
            identity.validate(&discovery)?;
        }
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
        Ok(build_engine_config(
            &self.model.source,
            &discovery,
            self.mode,
        ))
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

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        Ok(if self.supports_lora {
            vec!["load_lora".to_string()]
        } else {
            Vec::new()
        })
    }

    async fn engine_update(
        &self,
        update: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        if update != "load_lora" || !self.supports_lora {
            return Ok(json!({
                "status": "error",
                "message": format!("unsupported engine update: {update}"),
            }));
        }
        let request = parse_load_lora(&body)?;
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar is not started"))?;
        let endpoint = self
            .runtime_endpoint
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar runtime endpoint is not ready"))?;
        let _guard = self.lora_update_lock.lock().await;
        let adapters = client.list_loras().await?;
        let lora_id = next_lora_id(&adapters, &request.name)?;
        let response = client
            .load_lora(
                crate::proto::LoraAdapter {
                    lora_id,
                    lora_name: request.name,
                    source_path: request.path.to_string_lossy().into_owned(),
                },
                request.load_inplace,
            )
            .await?;
        let adapter = response.adapter.ok_or_else(|| {
            client::protocol_error("LoadLora response did not contain the loaded adapter")
        })?;
        publish_lora_model(endpoint, &adapter, self.max_loras).await?;
        Ok(json!({
            "status": "success",
            "lora_name": adapter.lora_name,
            "lora_id": adapter.lora_id,
            "already_loaded": response.already_loaded,
        }))
    }

    async fn on_endpoint_ready(
        &self,
        endpoint: dynamo_runtime::component::Endpoint,
    ) -> Result<(), DynamoError> {
        self.runtime_endpoint.set(endpoint).map_err(|_| {
            client::engine_shutdown("vLLM sidecar runtime endpoint was already initialized")
        })
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        Ok(())
    }
}

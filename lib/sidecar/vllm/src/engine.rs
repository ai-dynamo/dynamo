// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, WorkerConfig, usage,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, CONTROL_SERVICE, INFERENCE_SERVICE, VllmClient};
use crate::convert::{ResponseState, build_generate_request, data_parallel_rank};
use crate::model::DiscoveredModel;

pub struct VllmSidecarEngine {
    endpoint: GrpcEndpoint,
    model: DiscoveredModel,
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
        model: DiscoveredModel,
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

    /// Parse arguments and synchronously discover the vLLM model.
    ///
    /// Call this before `dynamo_backend_common::run`. Async callers must use
    /// `spawn_blocking` or a dedicated thread because discovery uses
    /// `Runtime::block_on`.
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
        if args.sidecar.common.dyn_tool_call_parser.is_some()
            || args.sidecar.common.dyn_reasoning_parser.is_some()
        {
            return Err(client::invalid_argument(
                "vLLM gRPC does not preserve the request options required by Dynamo tool-call and reasoning parsers",
            ));
        }

        let endpoint = GrpcEndpoint::parse(&args.vllm_endpoint, "--vllm-endpoint")?;
        let transport = args.sidecar.grpc.config();
        let bootstrap_deadline = client::startup_deadline(transport.startup_deadline)?;
        eprintln!(
            "Discovering vLLM model metadata from {endpoint}; startup deadline: {:?}",
            transport.startup_deadline
        );
        let model = bootstrap_discover(&endpoint, transport, bootstrap_deadline)?;
        let mode = args.sidecar.common.disaggregation_mode;
        let engine = Self::new(endpoint, model.clone(), mode, transport);
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
            served_model_name: Some(model.served_name.clone()),
            // gRPC cannot yet preserve the parser request semantics.
            tool_call_parser: None,
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: args
                .sidecar
                .common
                .exclude_tools_when_tool_choice_none,
            enable_kv_routing: true,
            disaggregation_mode: mode,
            route_to_encoder: false,
            enable_rl: args.sidecar.common.enable_rl,
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
        let startup_deadline = client::startup_deadline(self.transport.startup_deadline)?;
        let client = VllmClient::connect(&self.endpoint, self.transport, startup_deadline).await?;
        client
            .wait_for_services(
                &[CONTROL_SERVICE, INFERENCE_SERVICE],
                startup_deadline,
                self.transport.retry_interval,
            )
            .await?;
        let (model, server) = client.discover(startup_deadline).await?;
        let observed = DiscoveredModel::from_proto(model, server)?;
        self.model.ensure_startup_compatible(&observed)?;
        let connection_count = client.connection_count();
        self.client
            .set(client)
            .map_err(|_| client::engine_shutdown("vLLM sidecar has already started"))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections = connection_count,
            model = %observed.source,
            served_model_name = %observed.served_name,
            mode = %self.mode,
            "vLLM gRPC services are ready"
        );
        Ok(observed.engine_config())
    }

    async fn generate(
        &self,
        request: dynamo_backend_common::PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        if request
            .multi_modal_data
            .as_ref()
            .is_some_and(|media| media.values().any(|items| !items.is_empty()))
            && !self.model.supports_multimodal
        {
            return Err(client::invalid_argument(format!(
                "model `{}` does not advertise multimodal support",
                self.model.served_name
            )));
        }
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        let mut state = ResponseState::new(&request, self.mode);
        let data_parallel_rank = data_parallel_rank(&request, self.mode);
        let mut proto_request = build_generate_request(request, request_id, self.mode)?;
        proto_request.model.clone_from(&self.model.served_name);
        let defer_request_cancellation = self.mode.is_decode();
        let stopped_ctx = ctx.inner_arc();
        let shutdown = self.cancel.clone();
        let mut request_cancellation = Box::pin(async move { stopped_ctx.stopped().await });
        let mut shutdown_cancellation = Box::pin(async move { shutdown.cancelled().await });
        let stream = if defer_request_cancellation {
            // Decode must reach vLLM so NIXL can release transferred KV.
            tokio::select! {
                biased;
                _ = shutdown_cancellation.as_mut() => None,
                result = client.generate_stream(proto_request, data_parallel_rank) => Some(result?),
            }
        } else {
            tokio::select! {
                biased;
                _ = shutdown_cancellation.as_mut() => None,
                _ = request_cancellation.as_mut() => None,
                result = client.generate_stream(proto_request, data_parallel_rank) => Some(result?),
            }
        };
        let Some(mut stream) = stream else {
            let output = cancelled(&state);
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            let mut request_cancelled = false;
            let mut first_token_observed = false;
            loop {
                let message = if request_cancelled {
                    tokio::select! {
                        biased;
                        _ = shutdown_cancellation.as_mut() => None,
                        message = stream.message() => Some(message),
                    }
                } else {
                    tokio::select! {
                        biased;
                        _ = shutdown_cancellation.as_mut() => None,
                        _ = request_cancellation.as_mut() => {
                            if defer_request_cancellation && !first_token_observed {
                                request_cancelled = true;
                                continue;
                            }
                            None
                        }
                        message = stream.message() => Some(message),
                    }
                };

                let Some(message) = message else {
                    yield Ok(cancelled(&state));
                    break;
                };
                match message {
                    Ok(Some(response)) => {
                        let response_has_token = response
                            .outputs
                            .as_ref()
                            .is_some_and(|output| output.num_tokens > 0);
                        let transfer_completed = response.outputs.as_ref().is_some_and(|output| {
                            output.num_tokens > 0 || output.finish_info.is_some()
                        });
                        match state.convert(response) {
                            Ok(Some(output)) => {
                                first_token_observed |= response_has_token;
                                if request_cancelled && transfer_completed {
                                    // Dropping this stream aborts only this request.
                                    if first_token_observed {
                                        ctx.notify_first_token();
                                    }
                                    yield Ok(cancelled(&state));
                                    break;
                                }
                                let terminal = output.finish_reason.is_some();
                                yield Ok(output);
                                if terminal {
                                    break;
                                }
                            }
                            Ok(None) => {}
                            Err(error) if request_cancelled => {
                                tracing::warn!(
                                    %error,
                                    "vLLM response conversion failed after request cancellation"
                                );
                                yield Ok(cancelled(&state));
                                break;
                            }
                            Err(error) => {
                                yield Err(error);
                                break;
                            }
                        }
                    }
                    Ok(None) if request_cancelled => {
                        tracing::warn!(
                            "vLLM GenerateStream ended before transfer completion after request cancellation"
                        );
                        yield Ok(cancelled(&state));
                        break;
                    }
                    Ok(None) => {
                        yield Err(client::protocol_error(
                            "GenerateStream ended before a terminal response",
                        ));
                        break;
                    }
                    Err(status) if request_cancelled => {
                        tracing::warn!(
                            %status,
                            "vLLM GenerateStream failed before transfer completion after request cancellation"
                        );
                        yield Ok(cancelled(&state));
                        break;
                    }
                    Err(status) => {
                        yield Err(client::status_to_dynamo("GenerateStream", status));
                        break;
                    }
                }
            }
        }))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        Ok(())
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar is not started"))?;
        let expected_dp_size = self.model.data_parallel_size();
        let mut ranks = HashSet::new();
        let mut sources = Vec::new();
        let reported_sources = client.kv_event_sources().await?;
        if reported_sources.is_empty() {
            return Ok(Vec::new());
        }
        for source in reported_sources {
            if source.transport != "zmq" {
                tracing::warn!(
                    transport = %source.transport,
                    endpoint = %source.endpoint,
                    "Skipping unsupported vLLM KV-event transport"
                );
                continue;
            }
            let dp_rank = source.data_parallel_rank.ok_or_else(|| {
                client::protocol_error(
                    "GetKvEventSources returned a ZMQ source without data_parallel_rank",
                )
            })?;
            if dp_rank >= expected_dp_size {
                return Err(client::protocol_error(format!(
                    "GetKvEventSources returned rank {dp_rank}, outside the expected range 0..{expected_dp_size}",
                )));
            }
            if !ranks.insert(dp_rank) {
                return Err(client::protocol_error(format!(
                    "GetKvEventSources returned duplicate rank {dp_rank}",
                )));
            }
            if source.endpoint.trim().is_empty() {
                return Err(client::protocol_error(
                    "GetKvEventSources returned a ZMQ source without an endpoint",
                ));
            }
            sources.push(KvEventSource::Zmq {
                endpoint: zmq_connect_endpoint(&source.endpoint, &self.endpoint),
                topic: source.topic,
                dp_rank,
            });
        }
        if ranks.len() != expected_dp_size as usize {
            return Err(client::protocol_error(format!(
                "GetKvEventSources returned ZMQ sources for {} of {expected_dp_size} data-parallel ranks; KV routing requires one source for every rank",
                ranks.len()
            )));
        }
        Ok(sources)
    }
}

fn zmq_connect_endpoint(endpoint: &str, grpc_endpoint: &GrpcEndpoint) -> String {
    let port = endpoint
        .strip_prefix("tcp://*:")
        .or_else(|| endpoint.strip_prefix("tcp://0.0.0.0:"))
        .or_else(|| endpoint.strip_prefix("tcp://[::]:"));
    let Some(port) = port else {
        return endpoint.to_string();
    };

    format!("tcp://{}:{port}", grpc_endpoint.authority_host())
}

fn bootstrap_discover(
    endpoint: &GrpcEndpoint,
    transport: GrpcTransportConfig,
    startup_deadline: Instant,
) -> Result<DiscoveredModel, DynamoError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| client::engine_shutdown(format!("bootstrap runtime: {error}")))?;
    runtime.block_on(async {
        let bootstrap_transport = GrpcTransportConfig {
            connections: std::num::NonZeroUsize::MIN,
            ..transport
        };
        let client = VllmClient::connect(endpoint, bootstrap_transport, startup_deadline).await?;
        client
            .wait_for_services(
                &[CONTROL_SERVICE],
                startup_deadline,
                transport.retry_interval,
            )
            .await?;
        let (model, server) = client.discover(startup_deadline).await?;
        DiscoveredModel::from_proto(model, server)
    })
}

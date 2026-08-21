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
use serde_json::{Map, Value, json};
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

    fn started_client(&self) -> Result<&VllmClient, DynamoError> {
        self.client
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar is not started"))
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

    async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
        let Some(capabilities) = self.model.rl_capabilities() else {
            return Ok(Vec::new());
        };
        let mut controls = vec![
            "pause_generation".to_string(),
            "resume_generation".to_string(),
            "is_paused".to_string(),
            "is_sleeping".to_string(),
            "get_weight_version".to_string(),
        ];
        if capabilities.sleep_mode_enabled {
            controls.extend(["sleep".to_string(), "wake_up".to_string()]);
        }
        Ok(controls)
    }

    fn validate_engine_control(&self, control: &str, body: &Value) -> Result<(), DynamoError> {
        let body = request_object(body)?;
        match control {
            "pause_generation" => {
                pause_mode(body, "pause_generation")?;
                optional_bool(body, "clear_cache")?;
            }
            "sleep" => {
                sleep_level(body)?;
                pause_mode(body, "sleep")?;
            }
            "wake_up" => {
                wake_tags(body)?;
            }
            _ => {}
        }
        Ok(())
    }

    async fn engine_control(&self, control: String, body: Value) -> Result<Value, DynamoError> {
        if !self.supported_controls().await?.contains(&control) {
            return Ok(unsupported("control", &control));
        }
        let body = request_object(&body)?;
        let mut grpc = self.started_client()?.control_client();
        match control.as_str() {
            "pause_generation" => {
                let mode = pause_mode(body, "pause_generation")?;
                let clear_cache = optional_bool(body, "clear_cache")?;
                grpc.pause_generation(crate::proto::PauseGenerationRequest {
                    mode: mode as i32,
                    clear_cache,
                })
                .await
                .map_err(|status| client::status_to_dynamo("PauseGeneration", status))?;
                Ok(json!({"status": "paused"}))
            }
            "resume_generation" => {
                grpc.resume_generation(crate::proto::ResumeGenerationRequest {})
                    .await
                    .map_err(|status| client::status_to_dynamo("ResumeGeneration", status))?;
                let sleeping = if self
                    .model
                    .rl_capabilities()
                    .is_some_and(|capabilities| capabilities.sleep_mode_enabled)
                {
                    grpc_is_sleeping(&mut grpc).await?
                } else {
                    false
                };
                if sleeping {
                    Ok(json!({"status": "resumed", "is_sleeping": true}))
                } else {
                    Ok(json!({"status": "resumed"}))
                }
            }
            "is_paused" => {
                let response = grpc
                    .is_paused(crate::proto::IsPausedRequest {})
                    .await
                    .map_err(|status| client::status_to_dynamo("IsPaused", status))?
                    .into_inner();
                Ok(json!({"is_paused": response.paused}))
            }
            "sleep" => {
                let level = sleep_level(body)?;
                let mode = pause_mode(body, "sleep")?;
                grpc.sleep(crate::proto::SleepRequest {
                    level,
                    mode: mode as i32,
                })
                .await
                .map_err(|status| client::status_to_dynamo("Sleep", status))?;
                Ok(json!({"status": "sleeping"}))
            }
            "wake_up" => {
                let tags = wake_tags(body)?;
                grpc.wake_up(crate::proto::WakeUpRequest { tags })
                    .await
                    .map_err(|status| client::status_to_dynamo("WakeUp", status))?;
                if grpc_is_sleeping(&mut grpc).await? {
                    Ok(json!({"status": "partially_awake", "is_sleeping": true}))
                } else {
                    Ok(json!({"status": "awake"}))
                }
            }
            "is_sleeping" => Ok(json!({"is_sleeping": grpc_is_sleeping(&mut grpc).await?})),
            "get_weight_version" => {
                let response = grpc
                    .get_weight_version(crate::proto::GetWeightVersionRequest {})
                    .await
                    .map_err(|status| client::status_to_dynamo("GetWeightVersion", status))?
                    .into_inner();
                Ok(json!({"weight_version": response.weight_version}))
            }
            _ => Ok(unsupported("control", &control)),
        }
    }

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        let Some(capabilities) = self.model.rl_capabilities() else {
            return Ok(Vec::new());
        };
        let mut updates = vec!["update_weight_version".to_string()];
        if capabilities.weight_transfer_enabled {
            updates.extend([
                "init_weight_transfer_engine".to_string(),
                "start_weight_update".to_string(),
                "update_weights".to_string(),
                "finish_weight_update".to_string(),
            ]);
            if capabilities.draft_weight_updates_enabled {
                updates.push("start_draft_weight_update".to_string());
            }
        }
        Ok(updates)
    }

    async fn engine_update(&self, update: String, body: Value) -> Result<Value, DynamoError> {
        if !self.supported_updates().await?.contains(&update) {
            return Ok(unsupported("update", &update));
        }
        let body = request_object(&body)?;
        let mut grpc = self.started_client()?.control_client();
        match update.as_str() {
            "init_weight_transfer_engine" => {
                let init_info_json = required_object_json(body, "init_info")?;
                grpc.init_weight_transfer_engine(crate::proto::InitWeightTransferEngineRequest {
                    init_info_json,
                })
                .await
                .map_err(|status| client::status_to_dynamo("InitWeightTransferEngine", status))?;
                Ok(json!({"message": "Weight transfer initialized"}))
            }
            "start_weight_update" => {
                grpc.start_weight_update(crate::proto::StartWeightUpdateRequest {})
                    .await
                    .map_err(|status| client::status_to_dynamo("StartWeightUpdate", status))?;
                Ok(json!({"message": "Weight update started"}))
            }
            "start_draft_weight_update" => {
                grpc.start_draft_weight_update(crate::proto::StartDraftWeightUpdateRequest {})
                    .await
                    .map_err(|status| client::status_to_dynamo("StartDraftWeightUpdate", status))?;
                Ok(json!({"message": "Draft weight update started"}))
            }
            "update_weights" => {
                let update_info_json = required_object_json(body, "update_info")?;
                grpc.update_weights(crate::proto::UpdateWeightsRequest { update_info_json })
                    .await
                    .map_err(|status| client::status_to_dynamo("UpdateWeights", status))?;
                Ok(json!({"message": "Weights updated"}))
            }
            "finish_weight_update" => {
                let weight_version = optional_string(body, "weight_version")?;
                grpc.finish_weight_update(crate::proto::FinishWeightUpdateRequest {
                    weight_version,
                })
                .await
                .map_err(|status| client::status_to_dynamo("FinishWeightUpdate", status))?;
                Ok(json!({"message": "Weight update finished"}))
            }
            "update_weight_version" => {
                let weight_version = required_string(body, "new_version")?;
                grpc.update_weight_version(crate::proto::UpdateWeightVersionRequest {
                    weight_version: weight_version.clone(),
                })
                .await
                .map_err(|status| client::status_to_dynamo("UpdateWeightVersion", status))?;
                Ok(json!({"success": true, "new_version": weight_version}))
            }
            _ => Ok(unsupported("update", &update)),
        }
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

fn unsupported(kind: &str, name: &str) -> Value {
    json!({
        "status": "error",
        "message": format!("unsupported engine {kind}: {name}"),
    })
}

async fn grpc_is_sleeping(
    grpc: &mut crate::proto::control_client::ControlClient<tonic::transport::Channel>,
) -> Result<bool, DynamoError> {
    grpc.is_sleeping(crate::proto::IsSleepingRequest {})
        .await
        .map_err(|status| client::status_to_dynamo("IsSleeping", status))
        .map(|response| response.into_inner().sleeping)
}

fn request_object(body: &Value) -> Result<&Map<String, Value>, DynamoError> {
    body.as_object()
        .ok_or_else(|| client::invalid_argument("engine request body must be a JSON object"))
}

fn optional_bool(body: &Map<String, Value>, field: &str) -> Result<Option<bool>, DynamoError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(client::invalid_argument(format!(
            "`{field}` must be a boolean"
        ))),
    }
}

fn optional_u32(body: &Map<String, Value>, field: &str) -> Result<Option<u32>, DynamoError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| client::invalid_argument(format!("`{field}` must be a uint32"))),
    }
}

fn sleep_level(body: &Map<String, Value>) -> Result<Option<u32>, DynamoError> {
    let level = optional_u32(body, "level")?;
    if level.is_some_and(|level| level > 2) {
        return Err(client::invalid_argument(
            "`level` must be one of 0, 1, or 2",
        ));
    }
    Ok(level)
}

fn optional_string(body: &Map<String, Value>, field: &str) -> Result<Option<String>, DynamoError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(client::invalid_argument(format!(
            "`{field}` must be a string"
        ))),
    }
}

fn required_string(body: &Map<String, Value>, field: &str) -> Result<String, DynamoError> {
    optional_string(body, field)?
        .filter(|value| !value.is_empty())
        .ok_or_else(|| client::invalid_argument(format!("missing non-empty `{field}` string")))
}

fn pause_mode(
    body: &Map<String, Value>,
    operation: &str,
) -> Result<crate::proto::PauseMode, DynamoError> {
    match optional_string(body, "mode")?.as_deref().unwrap_or("abort") {
        "abort" => Ok(crate::proto::PauseMode::Abort),
        "wait" => Ok(crate::proto::PauseMode::Wait),
        "keep" => Ok(crate::proto::PauseMode::Keep),
        value => Err(client::invalid_argument(format!(
            "{operation} mode must be abort, wait, or keep; got `{value}`"
        ))),
    }
}

fn optional_strings(
    body: &Map<String, Value>,
    field: &str,
) -> Result<Option<Vec<String>>, DynamoError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(ToString::to_string).ok_or_else(|| {
                    client::invalid_argument(format!("`{field}` must contain only strings"))
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(client::invalid_argument(format!(
            "`{field}` must be an array of strings"
        ))),
    }
}

fn wake_tags(body: &Map<String, Value>) -> Result<Vec<String>, DynamoError> {
    let tags = optional_strings(body, "tags")?.unwrap_or_default();
    if let Some(tag) = tags
        .iter()
        .find(|tag| !matches!(tag.as_str(), "weights" | "kv_cache" | "scheduling"))
    {
        return Err(client::invalid_argument(format!(
            "wake_up tag must be weights, kv_cache, or scheduling; got `{tag}`"
        )));
    }
    Ok(tags)
}

fn required_object_json(body: &Map<String, Value>, field: &str) -> Result<Vec<u8>, DynamoError> {
    let value = body
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| client::invalid_argument(format!("missing `{field}` JSON object")))?;
    serde_json::to_vec(value)
        .map_err(|error| client::invalid_argument(format!("invalid `{field}`: {error}")))
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

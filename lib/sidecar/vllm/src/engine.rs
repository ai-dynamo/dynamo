// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, WorkerConfig, usage,
};
use dynamo_llm::lora::LoRADownloader;
use dynamo_runtime::component::Endpoint;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::stream::BoxStream;
use serde_json::{Value, json};
use tokio::sync::{Mutex, OnceCell};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, CONTROL_SERVICE, INFERENCE_SERVICE, VllmClient};
use crate::convert::{ResponseState, build_generate_request, data_parallel_rank};
use crate::lora::{
    build_downloader, parse_load_lora, parse_lora_name, publish_lora_model, resolve_source_path,
    unpublish_lora_model,
};
use crate::model::DiscoveredModel;

pub struct VllmSidecarEngine {
    endpoint: GrpcEndpoint,
    model: DiscoveredModel,
    mode: DisaggregationMode,
    transport: GrpcTransportConfig,
    client: OnceCell<VllmClient>,
    runtime_endpoint: OnceCell<Endpoint>,
    lora_downloader: OnceCell<LoRADownloader>,
    lora_reconciled: OnceCell<()>,
    lora_update_lock: Mutex<()>,
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
            runtime_endpoint: OnceCell::new(),
            lora_downloader: OnceCell::new(),
            lora_reconciled: OnceCell::new(),
            lora_update_lock: Mutex::new(()),
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
            ..Default::default()
        };
        Ok((engine, config))
    }

    fn ready_client(&self) -> Result<&VllmClient, DynamoError> {
        self.client
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar is not started"))
    }

    fn ready_endpoint(&self) -> Result<&Endpoint, DynamoError> {
        self.runtime_endpoint
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar runtime endpoint is not ready"))
    }

    async fn rollback_loaded_adapter(
        client: &VllmClient,
        adapter: &crate::proto::LoraAdapter,
    ) -> Result<(), DynamoError> {
        let response = client.unload_lora(adapter.lora_name.clone()).await?;
        match response.adapter {
            Some(removed) if removed == *adapter => Ok(()),
            observed => Err(client::protocol_error(format!(
                "native load rollback returned a different adapter identity: expected {adapter:?}, observed {observed:?}"
            ))),
        }
    }

    async fn restore_unloaded_adapter(
        &self,
        client: &VllmClient,
        endpoint: &Endpoint,
        adapter: &crate::proto::LoraAdapter,
    ) -> Result<(), DynamoError> {
        let response = client.load_lora(adapter.clone()).await?;
        match response.adapter {
            Some(restored) if restored == *adapter => {
                publish_lora_model(endpoint, adapter, self.model.max_loras()).await
            }
            observed => Err(client::protocol_error(format!(
                "native unload rollback returned a different adapter identity: expected {adapter:?}, observed {observed:?}"
            ))),
        }
    }

    fn validate_adapter_inventory(
        adapters: &[crate::proto::LoraAdapter],
    ) -> Result<(), DynamoError> {
        for (index, adapter) in adapters.iter().enumerate() {
            if adapter.lora_name.trim().is_empty() || adapter.lora_id <= 0 {
                return Err(client::protocol_error(format!(
                    "ListLoras returned an invalid adapter identity: {adapter:?}"
                )));
            }
            if let Some(conflict) = adapters[index + 1..].iter().find(|candidate| {
                candidate.lora_name == adapter.lora_name || candidate.lora_id == adapter.lora_id
            }) {
                return Err(client::protocol_error(format!(
                    "ListLoras returned conflicting adapter identities: {adapter:?} and {conflict:?}"
                )));
            }
        }
        Ok(())
    }

    async fn reconcile_loaded_loras(&self) -> Result<(), DynamoError> {
        self.lora_reconciled
            .get_or_try_init(|| async {
                let _guard = self.lora_update_lock.lock().await;
                let endpoint = self.ready_endpoint()?;
                let adapters = self.ready_client()?.list_loras().await?;
                Self::validate_adapter_inventory(&adapters)?;
                let mut published: Vec<String> = Vec::new();
                for adapter in &adapters {
                    if let Err(publish_error) =
                        publish_lora_model(endpoint, adapter, self.model.max_loras()).await
                    {
                        let mut rollback_errors = Vec::new();
                        for name in published.iter().rev() {
                            if let Err(error) = unpublish_lora_model(endpoint, name).await {
                                rollback_errors.push(format!("{name}: {error}"));
                            }
                        }
                        return if rollback_errors.is_empty() {
                            Err(publish_error)
                        } else {
                            Err(client::protocol_error(format!(
                                "failed to reconcile loaded LoRA adapters ({publish_error}); discovery rollback also failed: {}",
                                rollback_errors.join(", ")
                            )))
                        };
                    }
                    published.push(adapter.lora_name.clone());
                }
                tracing::info!(
                    adapter_count = adapters.len(),
                    "reconciled vLLM LoRA inventory into discovery"
                );
                Ok(())
            })
            .await
            .copied()
    }

    async fn load_lora(&self, body: &Value) -> Result<Value, DynamoError> {
        let request = parse_load_lora(body)?;
        if self.model.is_base_model_name(&request.name) {
            return Err(client::invalid_argument(format!(
                "LoRA adapter name `{}` conflicts with a served base model",
                request.name
            )));
        }
        let client = self.ready_client()?;
        let endpoint = self.ready_endpoint()?;
        let _guard = self.lora_update_lock.lock().await;
        let downloader = self
            .lora_downloader
            .get_or_try_init(|| async { build_downloader() })
            .await?;
        let source_path = resolve_source_path(downloader, &request.uri).await?;
        let source_path = source_path
            .to_str()
            .ok_or_else(|| client::invalid_argument("the resolved LoRA path is not valid UTF-8"))?;
        let lora_id = i64::from(dynamo_llm::utils::lora_name_to_id(&request.name));
        let requested = crate::proto::LoraAdapter {
            lora_id,
            lora_name: request.name.clone(),
            source_path: source_path.to_string(),
        };

        let loaded = client.list_loras().await?;
        if let Some(existing) = loaded
            .iter()
            .find(|adapter| adapter.lora_name == request.name)
        {
            if existing != &requested {
                return Err(client::invalid_argument(format!(
                    "LoRA adapter `{}` is already loaded with different identity",
                    request.name
                )));
            }
            publish_lora_model(endpoint, existing, self.model.max_loras()).await?;
            return Ok(json!({
                "status": "success",
                "message": format!("LoRA adapter '{}' is already loaded", request.name),
                "lora_name": existing.lora_name,
                "lora_id": existing.lora_id,
                "already_loaded": true,
            }));
        }
        if let Some(existing) = loaded
            .iter()
            .find(|adapter| adapter.lora_id == requested.lora_id)
        {
            return Err(client::invalid_argument(format!(
                "LoRA adapter ID {} for `{}` conflicts with loaded adapter `{}`",
                requested.lora_id, requested.lora_name, existing.lora_name
            )));
        }

        let response = match client.load_lora(requested.clone()).await {
            Ok(response) => response,
            Err(load_error) => match client.list_loras().await {
                Ok(adapters) => {
                    Self::validate_adapter_inventory(&adapters)?;
                    if adapters.iter().any(|adapter| adapter == &requested) {
                        tracing::warn!(
                            lora_name = %requested.lora_name,
                            %load_error,
                            "LoadLora failed after the adapter committed; reconciled with ListLoras"
                        );
                        crate::proto::LoadLoraResponse {
                            adapter: Some(requested.clone()),
                            already_loaded: false,
                        }
                    } else if let Some(conflict) = adapters.iter().find(|adapter| {
                        adapter.lora_name == requested.lora_name
                            || adapter.lora_id == requested.lora_id
                    }) {
                        return Err(client::protocol_error(format!(
                            "LoadLora failed ({load_error}) and reconciliation found a conflicting adapter: requested {requested:?}, observed {conflict:?}"
                        )));
                    } else {
                        return Err(load_error);
                    }
                }
                Err(list_error) => {
                    return Err(client::protocol_error(format!(
                        "LoadLora outcome is unknown: the request failed ({load_error}) and ListLoras reconciliation also failed ({list_error})"
                    )));
                }
            },
        };
        let adapter = match response.adapter {
            Some(adapter) if adapter == requested => adapter,
            observed => {
                let identity_error = client::protocol_error(format!(
                    "LoadLora returned a different adapter identity: expected {requested:?}, observed {observed:?}"
                ));
                return match Self::rollback_loaded_adapter(client, &requested).await {
                    Ok(()) => Err(identity_error),
                    Err(rollback_error) => Err(client::protocol_error(format!(
                        "{identity_error}; native rollback also failed: {rollback_error}"
                    ))),
                };
            }
        };
        if let Err(publish_error) =
            publish_lora_model(endpoint, &adapter, self.model.max_loras()).await
        {
            return match Self::rollback_loaded_adapter(client, &adapter).await {
                Ok(()) => Err(publish_error),
                Err(rollback_error) => Err(client::protocol_error(format!(
                    "failed to publish LoRA `{}` ({publish_error}); native rollback also failed: {rollback_error}",
                    adapter.lora_name
                ))),
            };
        }
        Ok(json!({
            "status": "success",
            "message": format!("LoRA adapter '{}' loaded successfully", adapter.lora_name),
            "lora_name": adapter.lora_name,
            "lora_id": adapter.lora_id,
            "already_loaded": response.already_loaded,
        }))
    }

    async fn unload_lora(&self, body: &Value) -> Result<Value, DynamoError> {
        let lora_name = parse_lora_name(body)?;
        let client = self.ready_client()?;
        let endpoint = self.ready_endpoint()?;
        let _guard = self.lora_update_lock.lock().await;
        let loaded = client.list_loras().await?;
        let adapter = loaded
            .into_iter()
            .find(|adapter| adapter.lora_name == lora_name)
            .ok_or_else(|| {
                client::invalid_argument(format!("LoRA adapter `{lora_name}` is not loaded"))
            })?;
        let unpublished = unpublish_lora_model(endpoint, &lora_name).await?;
        let response = match client.unload_lora(lora_name.clone()).await {
            Ok(response) => response,
            Err(unload_error) => match client.list_loras().await {
                Ok(adapters) => {
                    Self::validate_adapter_inventory(&adapters)?;
                    if let Some(observed) = adapters
                        .iter()
                        .find(|candidate| candidate.lora_name == lora_name)
                    {
                        if unpublished {
                            publish_lora_model(endpoint, observed, self.model.max_loras())
                                .await
                                .map_err(|restore_error| {
                                    client::protocol_error(format!(
                                        "UnloadLora failed ({unload_error}); the adapter remains loaded, but discovery restore also failed: {restore_error}"
                                    ))
                                })?;
                        }
                        if observed == &adapter {
                            return Err(unload_error);
                        }
                        return Err(client::protocol_error(format!(
                            "UnloadLora failed ({unload_error}) and reconciliation found a different adapter identity: expected {adapter:?}, observed {observed:?}"
                        )));
                    }
                    tracing::warn!(
                        %lora_name,
                        %unload_error,
                        "UnloadLora failed after the adapter committed; reconciled with ListLoras"
                    );
                    crate::proto::UnloadLoraResponse {
                        adapter: Some(adapter.clone()),
                    }
                }
                Err(list_error) => {
                    return Err(client::protocol_error(format!(
                        "UnloadLora outcome is unknown: the request failed ({unload_error}) and ListLoras reconciliation also failed ({list_error})"
                    )));
                }
            },
        };
        let removed = match response.adapter {
            Some(removed) if removed == adapter => removed,
            observed => {
                let identity_error = client::protocol_error(format!(
                    "UnloadLora returned a different adapter identity: expected {adapter:?}, observed {observed:?}"
                ));
                return match self
                    .restore_unloaded_adapter(client, endpoint, &adapter)
                    .await
                {
                    Ok(()) => Err(identity_error),
                    Err(restore_error) => Err(client::protocol_error(format!(
                        "{identity_error}; native and discovery restore also failed: {restore_error}"
                    ))),
                };
            }
        };
        Ok(json!({
            "status": "success",
            "message": format!("LoRA adapter '{lora_name}' unloaded successfully"),
            "lora_name": lora_name,
            "lora_id": removed.lora_id,
        }))
    }

    async fn list_loras(&self) -> Result<Value, DynamoError> {
        let client = self.ready_client()?;
        let _guard = self.lora_update_lock.lock().await;
        let mut adapters = client.list_loras().await?;
        adapters.sort_by(|left, right| left.lora_name.cmp(&right.lora_name));
        let loras: serde_json::Map<String, Value> = adapters
            .into_iter()
            .map(|adapter| (adapter.lora_name, json!(adapter.lora_id)))
            .collect();
        Ok(json!({
            "status": "success",
            "count": loras.len(),
            "loras": loras,
        }))
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

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        Ok(if self.model.supports_lora() {
            // Worker calls this after attaching the base model but before
            // exposing update routes, which makes this the first lifecycle
            // point where restart reconciliation can safely publish siblings.
            self.reconcile_loaded_loras().await?;
            vec![
                "load_lora".to_string(),
                "unload_lora".to_string(),
                "list_loras".to_string(),
            ]
        } else {
            Vec::new()
        })
    }

    async fn engine_update(&self, update: String, body: Value) -> Result<Value, DynamoError> {
        let lora_name = body
            .get("lora_name")
            .and_then(Value::as_str)
            .map(str::to_string);
        let result = if !self.model.supports_lora() {
            Err(client::invalid_argument(
                "vLLM did not advertise native LoRA lifecycle support",
            ))
        } else {
            match update.as_str() {
                "load_lora" => self.load_lora(&body).await,
                "unload_lora" => self.unload_lora(&body).await,
                "list_loras" => self.list_loras().await,
                _ => Err(client::invalid_argument(format!(
                    "unsupported engine update: {update}"
                ))),
            }
        };
        Ok(match result {
            Ok(response) => response,
            Err(error) => json!({
                "status": "error",
                "message": error.to_string(),
                "lora_name": lora_name,
            }),
        })
    }

    async fn on_endpoint_ready(&self, endpoint: Endpoint) -> Result<(), DynamoError> {
        self.runtime_endpoint.set(endpoint).map_err(|_| {
            client::engine_shutdown("vLLM sidecar runtime endpoint was already initialized")
        })
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

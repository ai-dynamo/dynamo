// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::PathBuf;

use async_trait::async_trait;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, WorkerConfig, usage,
};
use dynamo_llm::lora::{LoRADownloader, lora_serving_enabled};
use dynamo_runtime::component::Endpoint;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig, SidecarStartupError};
use futures::stream::BoxStream;
use serde_json::{Map, Value, json};
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, CONTROL_SERVICE, INFERENCE_SERVICE, VllmClient};
use crate::convert::{ResponseState, build_generate_request, data_parallel_rank};
use crate::lora::{self, build_downloader, parse_load_lora, parse_lora_name, resolve_source_path};
use crate::model::DiscoveredModel;
use crate::proto as pb;

pub struct VllmSidecarEngine {
    endpoint: GrpcEndpoint,
    model: DiscoveredModel,
    mode: DisaggregationMode,
    transport: GrpcTransportConfig,
    client: OnceCell<VllmClient>,
    runtime_endpoint: OnceCell<Endpoint>,
    lora_downloader: OnceCell<LoRADownloader>,
    lora_reconciled: OnceCell<()>,
    /// Resolved once at construction: the operator opted in via `DYN_LORA_ENABLED`
    /// and vLLM advertises adapter support with capacity for at least one.
    lora_enabled: bool,
    /// Resolved once at construction from the legacy `DYN_LORA_HOTSWAP_ENABLED` flag.
    hot_swap_requested: bool,
    lifecycle: lora::LoraLifecycle,
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
            lora_enabled: lora_serving_enabled() && model.supports_lora(),
            hot_swap_requested: hot_swap_requested(),
            endpoint,
            model,
            mode,
            transport,
            client: OnceCell::new(),
            runtime_endpoint: OnceCell::new(),
            lora_downloader: OnceCell::new(),
            lora_reconciled: OnceCell::new(),
            lifecycle: lora::LoraLifecycle::default(),
            cancel: CancellationToken::new(),
        }
    }

    /// Parse arguments and synchronously discover the vLLM model.
    ///
    /// Call this before `dynamo_backend_common::run`. Async callers must use
    /// `spawn_blocking` or a dedicated thread because discovery uses
    /// `Runtime::block_on`.
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        match argv {
            Some(argv) => Self::try_from_args(argv).map_err(SidecarStartupError::into_dynamo),
            None => Self::from_parsed(<Args as clap::Parser>::parse()),
        }
    }

    /// Parse injected arguments while retaining Clap's structured exit error.
    ///
    /// Embedded callers use this to distinguish help and version output from
    /// Dynamo startup failures without changing `from_args`'s error contract.
    pub fn try_from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), SidecarStartupError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)?;
        Self::from_parsed(args).map_err(Into::into)
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

        let endpoint = args.sidecar.grpc_endpoint;
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

    fn ready_endpoint(&self) -> Result<&Endpoint, DynamoError> {
        self.runtime_endpoint
            .get()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar runtime endpoint is not ready"))
    }

    /// LoRA management is exposed only when the operator enabled it, vLLM
    /// advertises adapter support, and the server reports capacity for at least
    /// one adapter. `DiscoveredModel::supports_lora` already folds in
    /// `max_loras > 0`.
    fn lora_enabled(&self) -> bool {
        self.lora_enabled
    }

    /// Force the enablement gate, so tests do not depend on a process-global env var.
    #[cfg(test)]
    pub(crate) fn with_lora_enabled(mut self, enabled: bool) -> Self {
        self.lora_enabled = enabled && self.model.supports_lora();
        self
    }

    /// Force the hot-swap request flag, so tests do not depend on a process-global env var.
    #[cfg(test)]
    pub(crate) fn with_hot_swap_requested(mut self, requested: bool) -> Self {
        self.hot_swap_requested = requested;
        self
    }

    /// Read vLLM's authoritative inventory, validated and sorted by name.
    async fn native_inventory(&self) -> Result<Vec<pb::LoraAdapter>, DynamoError> {
        let adapters = self
            .started_client()?
            .list_loras()
            .await
            .map_err(client::LoraRpcError::into_dynamo)?;
        crate::lora::validate_inventory(adapters)
    }

    /// The update names the RL weight-transfer surface contributes.
    ///
    /// Kept separate from [`Self::lora_updates`] so dispatch can check membership
    /// without triggering LoRA restart reconciliation.
    fn rl_updates(&self) -> Vec<String> {
        let Some(capabilities) = self.model.rl_capabilities() else {
            return Vec::new();
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
        updates
    }

    /// The update names the LoRA surface contributes, reconciling restart state first.
    ///
    /// Reconciliation is best effort: the worker calls this while bringing up its
    /// update routes, and a transient `ListLoras` or discovery failure must not
    /// stop the base model from serving.
    async fn lora_updates(&self) -> Vec<String> {
        if !self.lora_enabled() {
            return Vec::new();
        }
        if let Err(error) = self.reconcile_loaded_loras().await {
            tracing::warn!(
                %error,
                "LoRA restart reconciliation failed; serving the base model without \
                 republished adapters. Reconciliation is retried on the next lifecycle call."
            );
        }
        vec![
            lora::LOAD_LORA.to_string(),
            lora::UNLOAD_LORA.to_string(),
            lora::LIST_LORAS.to_string(),
        ]
    }

    /// Dispatch one LoRA lifecycle update, returning the legacy JSON envelope.
    async fn lora_engine_update(&self, update: &str, body: Value) -> Result<Value, DynamoError> {
        let lora_name = body
            .get("lora_name")
            .and_then(Value::as_str)
            .map(str::to_string);
        let result = if !self.lora_enabled() {
            Err(client::invalid_argument(
                "LoRA lifecycle is not available: it requires DYN_LORA_ENABLED and a vLLM \
                 server advertising LoRA support with max_loras > 0",
            ))
        } else {
            // Reconciliation is retried here when startup could not complete it, so a
            // transient failure at boot does not leave discovery permanently stale.
            if let Err(error) = self.reconcile_loaded_loras().await {
                tracing::warn!(%error, "LoRA reconciliation is still failing; continuing with the requested operation");
            }
            match update {
                lora::LOAD_LORA => self.load_lora(&body).await,
                lora::UNLOAD_LORA => self.unload_lora(&body).await,
                lora::LIST_LORAS => self.list_loras().await,
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

    /// Republish discovery records for adapters vLLM already holds, and drop
    /// Dynamo-only records for adapters it does not.
    ///
    /// Runs at most once per process; later lifecycle calls reconcile naturally.
    async fn reconcile_loaded_loras(&self) -> Result<(), DynamoError> {
        self.lora_reconciled
            .get_or_try_init(|| async {
                let endpoint = self.ready_endpoint()?;
                let adapters = self.native_inventory().await?;
                let mut records = Vec::with_capacity(adapters.len());
                for adapter in &adapters {
                    lora::publish_lora_model(endpoint, adapter, self.model.max_loras(), true)
                        .await?;
                    records.push(lora::LoraRecord {
                        name: adapter.lora_name.clone(),
                        id: adapter.lora_id,
                        source_uri: adapter.source_path.clone(),
                        path: PathBuf::from(&adapter.source_path),
                        published: true,
                    });
                }
                let stale = self.lifecycle.replace_published(records).await;
                for name in stale {
                    // Present in Dynamo but absent from vLLM: it must not stay routable.
                    if let Err(error) = lora::unpublish_lora_model(endpoint, &name).await {
                        tracing::warn!(%error, lora_name = %name, "failed to drop stale LoRA discovery record");
                    }
                }
                if !adapters.is_empty() {
                    tracing::info!(
                        count = adapters.len(),
                        "republished LoRA adapters loaded before sidecar start"
                    );
                }
                Ok::<(), DynamoError>(())
            })
            .await
            .copied()
    }

    /// Load one adapter, matching the legacy Python worker wherever the gRPC API allows.
    async fn load_lora(&self, body: &Value) -> Result<Value, DynamoError> {
        let request = parse_load_lora(body)?;
        let client = self.started_client()?;
        let endpoint = self.ready_endpoint()?;
        let _guard = self.lifecycle.lock(&request.name).await;

        // vLLM is authoritative for what is loaded.
        let loaded = self.native_inventory().await?;
        // Validated against the live inventory so a name that collapses onto another
        // adapter's discovery suffix is rejected before anything is mutated.
        lora::validate_adapter_name(
            &request.name,
            |name| self.model.is_base_model_name(name),
            &loaded,
        )?;
        let existing = loaded
            .iter()
            .find(|adapter| adapter.lora_name == request.name);

        if let Some(existing) = existing {
            if self.hot_swap_requested {
                return Err(client::invalid_argument(format!(
                    "LoRA adapter `{}` is already loaded and hot swap is not supported by the \
                     gRPC backend: it offers no atomic replace and no prefix-cache reset, so an \
                     unload/load pair would open a routing outage with no safe rollback. Unload \
                     the adapter explicitly, or load the new weights under a different name.",
                    request.name
                )));
            }
            // Idempotent even when the caller supplied a different URI, matching the
            // Python worker with hot swap disabled.
            self.ensure_published(endpoint, existing, &request.uri)
                .await?;
            tracing::info!(
                lora_name = %existing.lora_name,
                lora_id = existing.lora_id,
                "LoRA adapter already loaded"
            );
            return Ok(json!({
                "status": "success",
                "message": format!("LoRA adapter '{}' already loaded", existing.lora_name),
                "lora_name": existing.lora_name,
                "lora_id": existing.lora_id,
                "hot_swap": false,
            }));
        }

        // Held until this load finishes, so concurrent loads of other adapters
        // cannot collectively overshoot `max_loras`.
        let _slot = self
            .lifecycle
            .reserve(&request.name, loaded.len(), self.model.max_loras())
            .await?;

        let downloader = self
            .lora_downloader
            .get_or_try_init(|| async { build_downloader() })
            .await?;
        tracing::info!(lora_name = %request.name, uri = %request.uri, "resolving LoRA source");
        let source_path = resolve_source_path(downloader, &request.uri).await?;
        let source_path_arg = source_path.to_string_lossy().to_string();

        let adapter = match client
            .load_lora(request.name.clone(), source_path_arg.clone())
            .await
        {
            Ok(response) => self.expect_adapter(response.adapter, &request.name)?,
            Err(error) if error.is_definitive() => {
                if error.code == tonic::Code::AlreadyExists {
                    // Lost a race with another loader for this name; reconcile and
                    // report the identity vLLM settled on.
                    let observed = self.find_loaded(&request.name).await?.ok_or_else(|| {
                        client::protocol_error(format!(
                            "LoadLora reported `{}` as already loaded but ListLoras does not \
                             report it",
                            request.name
                        ))
                    })?;
                    self.ensure_published(endpoint, &observed, &request.uri)
                        .await?;
                    return Ok(json!({
                        "status": "success",
                        "message": format!("LoRA adapter '{}' already loaded", observed.lora_name),
                        "lora_name": observed.lora_name,
                        "lora_id": observed.lora_id,
                        "hot_swap": false,
                    }));
                }
                return Err(error.into_dynamo());
            }
            Err(error) => {
                // Timed out, or vLLM returned an internal/unknown status: the load may
                // still have committed, so let the inventory decide.
                tracing::warn!(%error, lora_name = %request.name, "LoadLora outcome is ambiguous; reconciling");
                match self.find_loaded(&request.name).await? {
                    Some(observed) if lora::paths_agree(&observed.source_path, &source_path) => {
                        observed
                    }
                    Some(observed) => {
                        return Err(client::protocol_error(format!(
                            "LoRA adapter `{}` is loaded from `{}` but this load requested `{}`; \
                             vLLM and Dynamo disagree about adapter state",
                            request.name, observed.source_path, source_path_arg
                        )));
                    }
                    None => return Err(error.into_dynamo()),
                }
            }
        };

        if let Err(error) =
            lora::publish_lora_model(endpoint, &adapter, self.model.max_loras(), false).await
        {
            // An adapter loaded into the GPU that no router can reach is worse than a
            // failed load: undo it rather than leaking capacity.
            tracing::error!(%error, lora_name = %adapter.lora_name, "failed to publish LoRA discovery record; rolling back the native load");
            self.rollback_loaded_adapter(client, &adapter).await;
            self.lifecycle.forget(&adapter.lora_name).await;
            return Err(error);
        }
        self.lifecycle
            .mark_published(lora::LoraRecord {
                name: adapter.lora_name.clone(),
                id: adapter.lora_id,
                source_uri: request.uri.clone(),
                path: source_path,
                published: true,
            })
            .await;

        tracing::info!(lora_name = %adapter.lora_name, lora_id = adapter.lora_id, "loaded LoRA adapter");
        Ok(json!({
            "status": "success",
            "message": format!("LoRA adapter '{}' loaded successfully", adapter.lora_name),
            "lora_name": adapter.lora_name,
            "lora_id": adapter.lora_id,
            "hot_swap": false,
        }))
    }

    /// Unload one adapter, stopping new routed traffic before mutating vLLM.
    async fn unload_lora(&self, body: &Value) -> Result<Value, DynamoError> {
        let lora_name = parse_lora_name(body)?;
        let client = self.started_client()?;
        let endpoint = self.ready_endpoint()?;
        let _guard = self.lifecycle.lock(&lora_name).await;

        let loaded = self.native_inventory().await?;
        let Some(existing) = loaded
            .iter()
            .find(|adapter| adapter.lora_name == lora_name)
            .cloned()
        else {
            let available: Vec<&str> = loaded
                .iter()
                .map(|adapter| adapter.lora_name.as_str())
                .collect();
            return Err(client::invalid_argument(format!(
                "LoRA adapter '{lora_name}' not found. Available LoRAs: {available:?}"
            )));
        };

        // Stop advertising before touching vLLM, so no request can be routed to an
        // adapter that is about to disappear.
        lora::unpublish_lora_model(endpoint, &lora_name).await?;
        let previous = self.lifecycle.forget(&lora_name).await;

        let removed = match client.unload_lora(lora_name.clone()).await {
            Ok(response) => self.expect_adapter(response.adapter, &lora_name)?,
            Err(error) if error.is_definitive() && error.code == tonic::Code::NotFound => {
                // Already gone; the unload is what the caller wanted.
                existing.clone()
            }
            Err(error) => {
                let definitive = error.is_definitive();
                let still_loaded = if definitive {
                    Some(existing.clone())
                } else {
                    tracing::warn!(%error, %lora_name, "UnloadLora outcome is ambiguous; reconciling");
                    self.find_loaded(&lora_name).await?
                };
                match still_loaded {
                    Some(observed) => {
                        // vLLM kept it, so it must stay routable.
                        self.restore_unloaded_adapter(endpoint, &observed, previous)
                            .await;
                        return Err(error.into_dynamo());
                    }
                    // It disappeared despite the error: treat the unload as committed.
                    None => existing.clone(),
                }
            }
        };

        tracing::info!(%lora_name, lora_id = removed.lora_id, "unloaded LoRA adapter");
        Ok(json!({
            "status": "success",
            "message": format!("LoRA adapter '{lora_name}' unloaded successfully"),
            "lora_name": lora_name,
            "lora_id": removed.lora_id,
        }))
    }

    /// Report vLLM's inventory as a deterministic name-to-ID map.
    async fn list_loras(&self) -> Result<Value, DynamoError> {
        let adapters = self.native_inventory().await?;
        let loras: Map<String, Value> = adapters
            .into_iter()
            .map(|adapter| (adapter.lora_name, json!(adapter.lora_id)))
            .collect();
        Ok(json!({
            "status": "success",
            "count": loras.len(),
            "loras": loras,
        }))
    }

    /// Hold the adapter's admission lock until the generation stream is established.
    ///
    /// The upstream server resolves the adapter before it starts generating, so once
    /// the streaming RPC is accepted an unload can no longer strand this request. The
    /// guard is returned to the caller so it drops with that borrow, not earlier.
    async fn admit_lora_request(&self, lora_name: &str) -> Result<lora::LoraGuard, DynamoError> {
        if !self.model.supports_lora() {
            return Err(client::invalid_argument(format!(
                "request selected LoRA adapter `{lora_name}` but this vLLM server did not \
                 advertise LoRA support"
            )));
        }
        let guard = self.lifecycle.lock(lora_name).await;
        // Checked against Dynamo's own published set rather than a `ListLoras` call:
        // discovery is what routers act on, and admission is on the request path, so a
        // control-plane round trip per request would be pure added latency. vLLM
        // independently rejects names it has not loaded, so a stale record cannot turn
        // into a silent base-model generation.
        if !self.lifecycle.is_published(lora_name).await {
            return Err(client::invalid_argument(format!(
                "unknown model or LoRA adapter: '{lora_name}'"
            )));
        }
        Ok(guard)
    }

    async fn find_loaded(&self, lora_name: &str) -> Result<Option<pb::LoraAdapter>, DynamoError> {
        Ok(self
            .native_inventory()
            .await?
            .into_iter()
            .find(|adapter| adapter.lora_name == lora_name))
    }

    fn expect_adapter(
        &self,
        adapter: Option<pb::LoraAdapter>,
        lora_name: &str,
    ) -> Result<pb::LoraAdapter, DynamoError> {
        let adapter = adapter.ok_or_else(|| {
            client::protocol_error(format!(
                "vLLM returned no adapter identity for LoRA `{lora_name}`"
            ))
        })?;
        if adapter.lora_name != lora_name {
            return Err(client::protocol_error(format!(
                "vLLM returned adapter `{}` for LoRA `{lora_name}`",
                adapter.lora_name
            )));
        }
        if adapter.lora_id <= 0 {
            return Err(client::protocol_error(format!(
                "vLLM returned a non-positive id {} for LoRA `{lora_name}`",
                adapter.lora_id
            )));
        }
        Ok(adapter)
    }

    /// Republish an adapter vLLM already holds, so an idempotent load still leaves
    /// discovery consistent.
    async fn ensure_published(
        &self,
        endpoint: &Endpoint,
        adapter: &pb::LoraAdapter,
        source_uri: &str,
    ) -> Result<(), DynamoError> {
        lora::publish_lora_model(endpoint, adapter, self.model.max_loras(), true).await?;
        self.lifecycle
            .mark_published(lora::LoraRecord {
                name: adapter.lora_name.clone(),
                id: adapter.lora_id,
                source_uri: source_uri.to_string(),
                path: PathBuf::from(&adapter.source_path),
                published: true,
            })
            .await;
        Ok(())
    }

    /// Best-effort removal of an adapter whose discovery publication failed.
    async fn rollback_loaded_adapter(&self, client: &VllmClient, adapter: &pb::LoraAdapter) {
        match client.unload_lora(adapter.lora_name.clone()).await {
            Ok(_) => tracing::info!(
                lora_name = %adapter.lora_name,
                "rolled back the native LoRA load"
            ),
            Err(error) => tracing::error!(
                %error,
                lora_name = %adapter.lora_name,
                "failed to roll back the native LoRA load; the adapter still occupies GPU capacity"
            ),
        }
    }

    /// Best-effort restoration of discovery after an unload that did not commit.
    async fn restore_unloaded_adapter(
        &self,
        endpoint: &Endpoint,
        adapter: &pb::LoraAdapter,
        previous: Option<lora::LoraRecord>,
    ) {
        match lora::publish_lora_model(endpoint, adapter, self.model.max_loras(), true).await {
            Ok(()) => {
                self.lifecycle
                    .mark_published(previous.unwrap_or_else(|| lora::LoraRecord {
                        name: adapter.lora_name.clone(),
                        id: adapter.lora_id,
                        source_uri: adapter.source_path.clone(),
                        path: PathBuf::from(&adapter.source_path),
                        published: true,
                    }))
                    .await;
                tracing::info!(
                    lora_name = %adapter.lora_name,
                    "restored the LoRA discovery record after a failed unload"
                );
            }
            Err(error) => tracing::error!(
                %error,
                lora_name = %adapter.lora_name,
                "failed to restore the LoRA discovery record; the adapter is loaded but unroutable"
            ),
        }
    }

    /// Drop every sibling record this sidecar published.
    async fn unpublish_all_loras(&self) {
        let Ok(endpoint) = self.ready_endpoint() else {
            return;
        };
        for name in self.lifecycle.published_names().await {
            match lora::unpublish_lora_model(endpoint, &name).await {
                Ok(_) => {
                    self.lifecycle.forget(&name).await;
                }
                Err(error) => {
                    tracing::warn!(%error, lora_name = %name, "failed to unpublish LoRA discovery record during shutdown");
                }
            }
        }
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
        if self.model.is_base_model_name(&proto_request.lora_name) {
            // Routers may address the base model by name through the adapter field.
            proto_request.lora_name.clear();
        }
        // Held until the streaming RPC is accepted. vLLM resolves the adapter before
        // it starts generating, so past that point an unload can no longer strand
        // this request, and the guard is released while the stream continues.
        let _admission = if proto_request.lora_name.is_empty() {
            None
        } else {
            Some(self.admit_lora_request(&proto_request.lora_name).await?)
        };
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
        let mut updates = self.lora_updates().await;
        updates.extend(self.rl_updates());
        Ok(updates)
    }

    async fn engine_update(&self, update: String, body: Value) -> Result<Value, DynamoError> {
        if crate::lora::is_lora_update(&update) {
            return self.lora_engine_update(&update, body).await;
        }
        if !self.rl_updates().contains(&update) {
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

    async fn on_endpoint_ready(&self, endpoint: Endpoint) -> Result<(), DynamoError> {
        self.runtime_endpoint.set(endpoint).map_err(|_| {
            client::engine_shutdown("vLLM sidecar runtime endpoint was already initialized")
        })
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        // The sidecar registers LoRA siblings itself, so the Worker's unregister of
        // the base card leaves them behind unless we drop them here.
        self.unpublish_all_loras().await;
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

/// True when the deployment asked for hot swap through the legacy Python env var.
///
/// The gRPC control surface cannot honor it, so this only exists to produce a clear
/// error instead of silently loading with different semantics.
fn hot_swap_requested() -> bool {
    dynamo_runtime::config::env_is_truthy("DYN_LORA_HOTSWAP_ENABLED")
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wraps Dynamo's KV-aware router for use from the ext_proc server.
//!
//! This is the native-Rust equivalent of the CGO bridge in
//! `lib/bindings/c/src/lib.rs`. Instead of crossing a C FFI boundary, the
//! ext_proc server calls these types directly as async Rust.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::config::{RouterConfigOverride, kv_router_config_from_dynamo_env};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};
use dynamo_llm::discovery::{ModelManager, WORKER_TYPE_DECODE};
use dynamo_llm::kv_router::prefill_router::PrefillQueryOutcome;
use dynamo_llm::kv_router::{KvRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::extensions::{
    HEADER_TENANT_ID, NvExt, request_cache_salt, routing_constraints_to_kv,
};
use dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::types::openai::completions::NvCreateCompletionRequest;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, hash_pod_name};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime};

use crate::envoy_helpers::find_header;
use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo};

const BOOKKEEPING_TIMEOUT: Duration = Duration::from_secs(5);
const DYN_KUBE_DISCOVERY_MODE: &str = "DYN_KUBE_DISCOVERY_MODE";

fn validate_kube_discovery_mode() -> Result<()> {
    match std::env::var(DYN_KUBE_DISCOVERY_MODE) {
        Ok(mode) => validate_kube_discovery_mode_value(Some(&mode)),
        Err(std::env::VarError::NotPresent) => validate_kube_discovery_mode_value(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{DYN_KUBE_DISCOVERY_MODE} must be valid Unicode")
        }
    }
}

fn validate_kube_discovery_mode_value(mode: Option<&str>) -> Result<()> {
    match mode {
        None | Some("pod") => Ok(()),
        Some("container") => {
            // TODO(epp-container-discovery): Resolve container-level discovery IDs to pod
            // endpoints, including non-main worker containers, then remove this restriction.
            anyhow::bail!(
                "Rust EPP does not support {DYN_KUBE_DISCOVERY_MODE}=container because it resolves \
                 worker endpoints by pod identity; use {DYN_KUBE_DISCOVERY_MODE}=pod"
            )
        }
        Some(mode) => anyhow::bail!(
            "Invalid {DYN_KUBE_DISCOVERY_MODE} value {mode:?}; valid values are 'pod' and 'container'"
        ),
    }
}

fn decode_router_config_override(is_disaggregated: bool) -> Option<RouterConfigOverride> {
    is_disaggregated.then_some(RouterConfigOverride {
        overlap_score_credit: Some(0.0),
        assume_kv_reuse: Some(false),
        track_prefill_tokens: Some(false),
        ..Default::default()
    })
}

fn cache_namespace_with_header_override(
    headers: &[(String, String)],
    body_cache_namespace: Option<String>,
) -> Option<String> {
    find_header(headers, HEADER_TENANT_ID)
        .filter(|tenant_id| !tenant_id.is_empty())
        .map(str::to_owned)
        .or(body_cache_namespace)
}

/// Name of the inference-serving HTTP port on a Dynamo worker pod.
///
/// Mirrors `commonconsts.DynamoContainerPortName` in
/// `deploy/operator/internal/consts/consts.go`. Worker pods may have multiple
/// containers (e.g. a `main` worker exposing metrics ports plus a
/// `sidecar-frontend` exposing the HTTP inference port); we route to frontend sidecar
/// container which exposes a port named `http`.
const DYNAMO_CONTAINER_PORT_NAME: &str = "http";

/// Holds all router state needed for request routing.
///
/// This is the async-native equivalent of `RouterHandles` from the C bindings,
/// without the `block_on` / unsafe FFI overhead.
pub struct Router {
    prefill_router: Arc<PrefillRouter>,
    decode_router: Arc<KvRouter>,
    preprocessor: Arc<OpenAIPreprocessor>,
    runtime: Runtime,
    pod_store: kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    pod_store_ready: Arc<AtomicBool>,
}

impl Router {
    /// Initialize the router from discovery.
    ///
    /// This waits for at least one decode worker to appear, fetches the model
    /// card, initializes the preprocessor, and creates both routers.
    pub async fn from_discovery(namespace: &str, component: &str) -> Result<Self> {
        validate_kube_discovery_mode()?;

        let runtime = Runtime::from_settings()?;
        let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Wait for workers
        wait_for_discovery_sync(&drt).await;

        let bootstrap = init_preprocessor(&drt, namespace).await?;
        let block_size = bootstrap.card.kv_cache_block_size;
        let model_name = bootstrap.card.display_name.clone();
        let enable_eagle = bootstrap.card.runtime_config.enable_eagle;
        let actual_namespace = &bootstrap.actual_namespace;

        // TODO(epp-rolling-namespace): Rebind both routers when the active
        // generation-suffixed worker namespace changes during a rolling update.
        let mut kv_router_config = kv_router_config_from_dynamo_env();
        // TODO(epp-multi-replica): Provide authoritative admission across EPP
        // replicas; replica-sync alone does not close the selection-to-booking race.
        kv_router_config.skip_initial_worker_wait = true;

        let component_handle = drt.namespace(actual_namespace)?.component(component)?;
        let endpoint = component_handle.endpoint("generate");

        let model_manager = Arc::new(ModelManager::new());

        let decode_router = model_manager
            .kv_chooser_for(
                &endpoint,
                block_size,
                Some(kv_router_config.clone()),
                None,
                WORKER_TYPE_DECODE,
                Some(model_name.clone()),
                enable_eagle,
            )
            .await?;

        // Wait for runtime config watch to populate
        {
            let mut config_watch = model_manager
                .get_or_create_runtime_config_watcher(&endpoint)
                .await?;
            tracing::info!("Waiting for decode workers to register ModelRuntimeConfig...");
            config_watch
                .wait_for(|m| !m.is_empty())
                .await
                .map(|_| ())
                .map_err(|_| {
                    anyhow::anyhow!("Runtime config watch closed before any workers appeared")
                })?;
            tracing::info!(
                worker_count = config_watch.borrow().len(),
                "Runtime config watch populated"
            );
        }

        let mut prefill_config = kv_router_config;
        prefill_config.router_track_active_blocks = false;

        let (prefill_tx, prefill_rx) = tokio::sync::oneshot::channel();
        let prefill_router = PrefillRouter::new(
            prefill_rx,
            model_manager.clone(),
            RouterMode::KV,
            block_size,
            Some(prefill_config),
            None,
            None,
            model_name.clone(),
            actual_namespace.to_string(),
            enable_eagle,
            // ext-proc constructs no KvWorkerMonitor; overload publishing is
            // unused on this path (matches the prior namespace-lookup miss).
            None,
        );

        spawn_prefill_discovery_watcher(drt.clone(), actual_namespace.to_string(), prefill_tx);

        // Use the BASE namespace (without rolling-update suffix) for the pod
        // selector. Workers register in discovery under the suffixed namespace
        // (e.g. "atchernych-qwen-9f792849"), but the K8s pod label
        // `nvidia.com/dynamo-namespace` is always set to the base
        // ("atchernych-qwen") by the operator. Using the suffixed name here
        // would silently match zero pods during/after a DGD rolling update.
        let (pod_store, pod_store_ready) = spawn_pod_reflector(namespace).await?;

        // `model_manager` and `drt` are intentionally not stored on the
        // Router. The KV chooser, prefill router, prefill discovery watcher,
        // and pod reflector all clone whatever they need from these
        // constructor-locals before this scope ends, so dropping them here
        // does not tear down any background work.
        Ok(Self {
            prefill_router,
            decode_router,
            preprocessor: bootstrap.preprocessor,
            runtime,
            pod_store,
            pod_store_ready,
        })
    }

    /// Tokenize a JSON request body and extract router queue priorities.
    ///
    /// Priorities default to zero when absent. Mirrors the standalone Dynamo
    /// preprocessor lift in `lib/llm/src/preprocessor.rs`.
    fn tokenize(&self, request_json: &str) -> Result<TokenizeResult> {
        // TODO(epp-request-routing): Reuse shared preprocessing so expected output
        // length, LoRA, pins, sessions, topology constraints, additional protocols,
        // and multimodal routing hashes are preserved.
        let mut request_json: serde_json::Value = serde_json::from_str(request_json)?;

        if request_json.get("prompt").is_some() || request_json.get("prompt_embeds").is_some() {
            if let Some(result) = tokenize_completion_prompt_embeds(&request_json)? {
                return Ok(result);
            }

            let token_data_policy = prepare_completion_prompt_for_routing(&mut request_json);
            let request: NvCreateCompletionRequest = serde_json::from_value(request_json)?;
            let nvext = request.nvext.as_ref();
            let priority_jump = extract_priority_jump(nvext);
            let strict_priority = extract_strict_priority(nvext);
            let routing_constraints = extract_routing_constraints(nvext);
            let cache_namespace = request_cache_salt(&request).map(str::to_owned);

            let (token_ids, _) = self.preprocessor.gather_tokens(&request, None, None)?;

            return Ok(TokenizeResult {
                token_ids,
                cache_namespace,
                priority_jump,
                strict_priority,
                routing_constraints,
                token_data_policy,
            });
        }

        let request: NvCreateChatCompletionRequest = serde_json::from_value(request_json)?;
        let nvext = request.nvext.as_ref();
        let priority_jump = extract_priority_jump(nvext);
        let strict_priority = extract_strict_priority(nvext);
        let routing_constraints = extract_routing_constraints(nvext);
        let cache_namespace = request_cache_salt(&request).map(str::to_owned);

        let formatted_prompt = self
            .preprocessor
            .apply_template(&request)?
            .unwrap_or_default();

        let encoding = self.preprocessor.tokenize(&formatted_prompt)?;
        Ok(TokenizeResult {
            token_ids: encoding.token_ids().to_vec(),
            cache_namespace,
            priority_jump,
            strict_priority,
            routing_constraints,
            token_data_policy: TokenDataPolicy::Inject,
        })
    }

    /// Resolve a worker_id to a pod endpoint address (ip:port).
    /// Lock-free read from the in-memory reflector store — no K8s API calls.
    /// Port is read from the pod's Dynamo HTTP container port.
    pub fn resolve_worker_endpoint(&self, worker_id: u64) -> Option<String> {
        for pod in self.pod_store.state() {
            let Some(pod_name) = pod.metadata.name.as_deref() else {
                continue;
            };
            if hash_pod_name(pod_name) == worker_id {
                return pod_endpoint_address(&pod);
            }
        }
        None
    }

    /// Resolve any available worker to its endpoint address (ip:port).
    /// Used for body-less requests (GET /v1/models) where we just need any backend.
    pub fn resolve_any_worker_endpoint(&self) -> Option<String> {
        self.pod_store
            .state()
            .iter()
            .find_map(|pod| pod_endpoint_address(pod))
    }

    /// Resolve any reflected worker whose worker_id is in `allowed`.
    /// Used for body-less requests that still carry an Envoy subset hint, so
    /// we never resolve a backend outside the requested subset.
    fn resolve_any_worker_endpoint_in_subset(&self, allowed: &HashSet<u64>) -> Option<String> {
        for pod in self.pod_store.state() {
            let Some(pod_name) = pod.metadata.name.as_deref() else {
                continue;
            };
            if allowed.contains(&hash_pod_name(pod_name))
                && let Some(addr) = pod_endpoint_address(&pod)
            {
                return Some(addr);
            }
        }
        None
    }

    /// Map an Envoy `candidate_subset` (endpoint addresses, "ip:port" or bare
    /// "ip") onto the worker IDs of the reflected pods that match it.
    ///
    /// This is how the InferencePool subset hint is honored on the hot path:
    /// the ext_proc server always calls `pick()` with an empty external
    /// endpoint list, so the subset must be intersected against the in-memory
    /// pod reflector rather than a caller-supplied slice. An empty result for
    /// a non-empty subset means no reflected pod matched the hint.
    fn subset_to_worker_ids(&self, candidate_subset: &[String]) -> HashSet<u64> {
        let candidates: HashSet<&str> = candidate_subset.iter().map(|s| s.as_str()).collect();
        let mut ids = HashSet::new();
        for pod in self.pod_store.state() {
            let Some(pod_name) = pod.metadata.name.as_deref() else {
                continue;
            };
            let Some(addr_port) = pod_endpoint_address(&pod) else {
                continue;
            };
            let ip = addr_port.split(':').next().unwrap_or("");
            if candidates.contains(addr_port.as_str()) || candidates.contains(ip) {
                ids.insert(hash_pod_name(pod_name));
            }
        }
        ids
    }

    /// Route a prefill request. Returns (worker_id, dp_rank).
    ///
    /// Queue priorities are forwarded to the prefill scheduler. `priority_jump`
    /// adjusts the policy score, while `strict_priority` selects the primary tier.
    pub async fn route_prefill(
        &self,
        tokens: &[u32],
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<u64>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<(u64, Option<u32>)> {
        if let Some(ref ids) = allowed_worker_ids {
            self.prefill_router.register_workers(ids);
        }

        // TODO(epp-prefill-booking): Atomically reserve the selected prefill worker
        // and release it on first output, cancellation, or routing failure.
        let outcome = self
            .prefill_router
            .query_prefill_worker(
                tokens,
                None,
                None,
                cache_namespace,
                priority_jump,
                strict_priority,
                allowed_worker_ids,
                routing_constraints,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Prefill query failed: {:?}", e))?;

        match outcome {
            // Advisory only: the gateway owns dispatch and lifecycle state.
            PrefillQueryOutcome::Routed { worker_id, dp_rank } => Ok((worker_id, dp_rank)),
            PrefillQueryOutcome::QueueRejected { rejection } => Err(anyhow::anyhow!(
                "Prefill router policy-class queue rejection: policy_class={}, limit_kind={}, current={}, limit={}",
                rejection.policy_class,
                rejection.limit_kind,
                rejection.current,
                rejection.limit
            )),
        }
    }

    /// Route a decode request. Returns (WorkerWithDpRank, overlap_blocks).
    ///
    /// Queue priorities are forwarded to the decode scheduler. `priority_jump`
    /// adjusts the policy score, while `strict_priority` selects the primary tier.
    pub async fn route_decode(
        &self,
        tokens: &[u32],
        is_disaggregated: bool,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<u64>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<(WorkerWithDpRank, u32)> {
        if let Some(ref ids) = allowed_worker_ids {
            self.decode_router.register_workers(ids);
        }

        let config_override = decode_router_config_override(is_disaggregated);

        self.decode_router
            .find_best_match(
                None,
                tokens,
                None,
                config_override.as_ref(),
                false,
                None,
                cache_namespace,
                priority_jump,
                strict_priority,
                None,
                allowed_worker_ids,
                routing_constraints,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Decode query failed: {:?}", e))
    }

    /// Register a request with the decode router for bookkeeping.
    pub async fn add_request(
        &self,
        request_id: &str,
        tokens: &[u32],
        worker_id: u64,
        dp_rank: u32,
        is_disaggregated: bool,
        cache_namespace: Option<String>,
    ) -> Result<()> {
        let decode_router = self.decode_router.clone();
        let request_id = request_id.to_owned();
        let tokens = tokens.to_vec();

        tokio::time::timeout(BOOKKEEPING_TIMEOUT, async {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);
            let router_config_override = decode_router_config_override(is_disaggregated);

            let overlap_blocks = decode_router
                .get_overlap_blocks(&tokens, None, worker, None, cache_namespace.as_deref())
                .await
                .map_err(|e| anyhow::anyhow!("get_overlap_blocks failed: {e:?}"))?;

            let cached_tokens = overlap_blocks as usize * decode_router.block_size() as usize;

            decode_router
                .add_request(
                    request_id,
                    &tokens,
                    None,
                    cached_tokens,
                    None,
                    worker,
                    None,
                    cache_namespace,
                    router_config_override.as_ref(),
                )
                .await;

            Ok(())
        })
        .await
        .map_err(|_| anyhow::anyhow!("add_request timed out"))?
    }

    /// Mark prefill as completed for a request.
    pub async fn mark_prefill_complete(&self, request_id: &str) -> Result<()> {
        let decode_router = self.decode_router.clone();
        let request_id = request_id.to_owned();

        tokio::time::timeout(BOOKKEEPING_TIMEOUT, async {
            decode_router
                .mark_prefill_completed(&request_id)
                .await
                .map_err(|e| anyhow::anyhow!("mark_prefill_completed failed: {e}"))
        })
        .await
        .map_err(|_| anyhow::anyhow!("mark_prefill_complete timed out"))?
    }

    /// Free a request from the router's bookkeeping.
    pub async fn free_request(&self, request_id: &str) -> Result<()> {
        let decode_router = self.decode_router.clone();
        let request_id = request_id.to_owned();

        tokio::time::timeout(BOOKKEEPING_TIMEOUT, async {
            decode_router
                .free(&request_id)
                .await
                .map_err(|e| anyhow::anyhow!("free failed: {e}"))
        })
        .await
        .map_err(|_| anyhow::anyhow!("free_request timed out"))?
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Shared handle to the pod reflector readiness flag.
    ///
    /// `from_discovery` returns as soon as worker discovery and the model card
    /// are ready, but the K8s pod reflector's initial LIST may still be in
    /// flight if it exceeded the startup timeout (see `spawn_pod_reflector`).
    /// `pick()` returns 503 until this flag flips to `true`, so callers (e.g.
    /// the gRPC health reporter) can gate their SERVING status on it to avoid
    /// advertising readiness while routing would still 503.
    pub fn pod_store_ready(&self) -> Arc<AtomicBool> {
        self.pod_store_ready.clone()
    }
}

struct TokenizeResult {
    token_ids: Vec<u32>,
    cache_namespace: Option<String>,
    priority_jump: f64,
    strict_priority: u32,
    routing_constraints: RoutingConstraints,
    token_data_policy: TokenDataPolicy,
}

#[derive(serde::Deserialize)]
struct CompletionPromptEmbedsRoutingFields {
    #[serde(rename = "model")]
    _model: String,
    #[serde(rename = "prompt_embeds")]
    _prompt_embeds: String,
    #[serde(default)]
    cache_salt: Option<String>,
    #[serde(default)]
    nvext: Option<NvExt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenDataPolicy {
    Inject,
    Strip,
}

impl TokenDataPolicy {
    fn injects(self) -> bool {
        matches!(self, Self::Inject)
    }

    fn strips(self) -> bool {
        matches!(self, Self::Strip)
    }
}

fn tokenize_completion_prompt_embeds(
    request_json: &serde_json::Value,
) -> Result<Option<TokenizeResult>> {
    let Some(prompt_embeds) = request_json.get("prompt_embeds") else {
        return Ok(None);
    };
    if prompt_embeds.is_null() {
        return Ok(None);
    }

    let fields: CompletionPromptEmbedsRoutingFields =
        serde_json::from_value(request_json.clone())?;
    let nvext = fields.nvext.as_ref();
    let priority_jump = extract_priority_jump(nvext);
    let strict_priority = extract_strict_priority(nvext);
    let routing_constraints = extract_routing_constraints(nvext);
    let cache_namespace = nvext
        .and_then(|nvext| nvext.cache_salt.as_deref())
        .filter(|salt| !salt.is_empty())
        .or_else(|| fields.cache_salt.as_deref().filter(|salt| !salt.is_empty()))
        .map(str::to_owned);

    Ok(Some(TokenizeResult {
        token_ids: Vec::new(),
        cache_namespace,
        priority_jump,
        strict_priority,
        routing_constraints,
        token_data_policy: TokenDataPolicy::Strip,
    }))
}

/// Pick a single completion prompt shape the router can score today.
///
/// Returns how the forwarded request's `nvext.token_data` should be handled.
/// Only scalar text completion prompts consume injected token data correctly.
/// Other completion shapes either already carry tokens, are batch-shaped, or
/// use embeddings, so stale flat token data must be stripped.
fn prepare_completion_prompt_for_routing(request_json: &mut serde_json::Value) -> TokenDataPolicy {
    if request_json.get("prompt_embeds").is_some() {
        return TokenDataPolicy::Strip;
    }

    let Some(prompt) = request_json.get_mut("prompt") else {
        return TokenDataPolicy::Strip;
    };
    if prompt.is_string() {
        return TokenDataPolicy::Inject;
    }
    let Some(items) = prompt.as_array() else {
        return TokenDataPolicy::Strip;
    };
    if items.is_empty() {
        return TokenDataPolicy::Strip;
    }

    if items.iter().all(|item| item.is_string()) {
        if items.len() > 1 {
            let first = items
                .first()
                .and_then(|item| item.as_str())
                .map(str::to_string);
            if let Some(first) = first {
                *prompt = serde_json::Value::String(first);
            }
        }
        return TokenDataPolicy::Strip;
    } else if items.iter().all(|item| item.is_array()) {
        if items.len() > 1 {
            let first = items.first().cloned();
            if let Some(first) = first {
                *prompt = first;
            }
        }
        return TokenDataPolicy::Strip;
    }

    TokenDataPolicy::Strip
}

/// Extract the router queue `priority_jump` from an OpenAI request's
/// `nvext.agent_hints.priority`.
///
/// Negative priorities are clamped to `0.0` so a low-priority hint never
/// pushes a request behind FCFS arrivals (matches the standalone preprocessor
/// in `lib/llm/src/preprocessor.rs`). Falls back to the deprecated
/// `latency_sensitivity` alias for callers still on the old field name.
/// Returns `0.0` when `nvext` is absent.
fn extract_priority_jump(nvext: Option<&NvExt>) -> f64 {
    nvext
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| h.priority.map(|p| p as f64).or(h.latency_sensitivity))
        .map(|p| p.max(0.0))
        .unwrap_or(0.0)
}

fn extract_strict_priority(nvext: Option<&NvExt>) -> u32 {
    nvext
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| h.strict_priority)
        .unwrap_or(0)
}

fn extract_routing_constraints(nvext: Option<&NvExt>) -> RoutingConstraints {
    nvext
        .and_then(|n| n.routing_constraints.clone())
        .map(routing_constraints_to_kv)
        .unwrap_or_default()
}

struct DiscoveredModelBootstrap {
    preprocessor: Arc<OpenAIPreprocessor>,
    card: ModelDeploymentCard,
    actual_namespace: String,
}

async fn wait_for_discovery_sync(drt: &DistributedRuntime) {
    tracing::info!("Waiting for discovery to sync (controlled by K8s StartupProbe)...");
    let discovery = drt.discovery();

    loop {
        match discovery.list(DiscoveryQuery::AllModels).await {
            Ok(instances) if !instances.is_empty() => {
                tracing::info!(count = instances.len(), "Discovery sync complete");
                return;
            }
            Ok(_) => {
                tracing::debug!("No instances yet, waiting...");
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(e) => {
                tracing::warn!("Discovery list error: {}, retrying...", e);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
}

async fn init_preprocessor(
    drt: &DistributedRuntime,
    target_namespace: &str,
) -> Result<DiscoveredModelBootstrap> {
    loop {
        match fetch_preprocessor_from_discovery(drt, target_namespace).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    target_namespace,
                    "Model card not available yet, retrying in 5s..."
                );
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}

async fn fetch_preprocessor_from_discovery(
    drt: &DistributedRuntime,
    target_namespace: &str,
) -> Result<DiscoveredModelBootstrap> {
    let discovery = drt.discovery();
    let instances = discovery.list(DiscoveryQuery::AllModels).await?;

    let mut model_card: Option<(ModelDeploymentCard, String)> = None;

    let discovered_namespaces: Vec<String> = instances
        .iter()
        .filter_map(|i| {
            if let DiscoveryInstance::Model { namespace, .. } = i {
                Some(namespace.clone())
            } else {
                None
            }
        })
        .collect();

    tracing::debug!(
        ?discovered_namespaces,
        target_namespace,
        "Discovery returned {} model instances",
        discovered_namespaces.len()
    );

    for instance in instances {
        if let DiscoveryInstance::Model { namespace, .. } = &instance {
            if !namespace.starts_with(target_namespace) {
                continue;
            }

            let actual_namespace = namespace.clone();
            match instance.deserialize_model::<ModelDeploymentCard>() {
                Ok(card) => {
                    if card.model_type.supports_prefill()
                        && !card.model_type.supports_chat()
                        && !card.model_type.supports_completions()
                    {
                        continue;
                    }
                    model_card = Some((card, actual_namespace));
                    break;
                }
                Err(e) => {
                    tracing::debug!(error = %e, "Failed to deserialize model card, skipping");
                    continue;
                }
            }
        }
    }

    let (mut card, actual_namespace) = model_card.ok_or_else(|| {
        anyhow::anyhow!(
            "No model found in namespace '{}' via discovery. \
             Found {} instances in namespaces: {:?}. \
             Set DYN_NAMESPACE_PREFIX (or DYN_NAMESPACE) to match your workers' registration namespace.",
            target_namespace,
            discovered_namespaces.len(),
            discovered_namespaces,
        )
    })?;

    tracing::info!(
        model_name = %card.display_name,
        kv_cache_block_size = card.kv_cache_block_size,
        actual_namespace = %actual_namespace,
        "Found model card via discovery"
    );

    card.download_config(None).await?;
    let preprocessor = OpenAIPreprocessor::new(card.clone())?;

    Ok(DiscoveredModelBootstrap {
        preprocessor, // already Arc<OpenAIPreprocessor>
        card,
        actual_namespace,
    })
}

/// Extract "ip:port" from a pod by reading its IP from status and the
/// container port named `http` (the Dynamo HTTP inference port) from the
/// container spec.
///
/// Worker pods commonly have multiple containers exposing multiple HTTP
/// ports: a `main` worker container exposing `system=9090` (probes +
/// Prometheus metrics) and `nixl=19090` (NIXL telemetry), plus a
/// `sidecar-frontend` container exposing `http=8000` (the OpenAI-compatible
/// inference API — the port the InferencePool's `targetPort` resolves to).
/// All three speak HTTP, but only the inference port is *named* `http` in
/// the pod spec. Picking `containers.first().ports.first()` would land on
/// `system=9090` and route inference traffic to the metrics endpoint; we
/// instead scan all containers for the port named `http`, mirroring how
/// Kubernetes resolves a string `targetPort`.
///
/// Returns `None` if the pod has no IP or no container exposes a port named
/// `http` — we never silently route to a guessed port.
fn pod_endpoint_address(pod: &k8s_openapi::api::core::v1::Pod) -> Option<String> {
    let ip = pod.status.as_ref()?.pod_ip.as_ref()?;
    let port = pod
        .spec
        .as_ref()?
        .containers
        .iter()
        .filter_map(|c| c.ports.as_ref())
        .flatten()
        .find(|p| p.name.as_deref() == Some(DYNAMO_CONTAINER_PORT_NAME))
        .map(|p| p.container_port)?;
    Some(format!("{ip}:{port}"))
}

/// Start a background pod reflector that watches worker pods matching the
/// InferencePool selector. The returned `Store` provides lock-free reads
/// of the current pod state — no K8s API calls on the hot path.
async fn spawn_pod_reflector(
    dynamo_namespace: &str,
) -> Result<(
    kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    Arc<AtomicBool>,
)> {
    use futures::StreamExt;
    use k8s_openapi::api::core::v1::Pod;
    use kube::{Api, Client, runtime::reflector, runtime::watcher};

    let client = Client::try_default().await?;

    let k8s_namespace = std::env::var("POD_NAMESPACE").map_err(|_| {
        anyhow::anyhow!(
            "POD_NAMESPACE environment variable is not set. \
             The operator injects this via the downward API — \
             ensure the EPP pod spec includes fieldRef metadata.namespace."
        )
    })?;

    let pods: Api<Pod> = Api::namespaced(client, &k8s_namespace);

    let selector = format!(
        "nvidia.com/dynamo-namespace={},nvidia.com/dynamo-component-class=worker",
        dynamo_namespace
    );

    let writer = reflector::store::Writer::default();
    let store = writer.as_reader();
    let ready = Arc::new(AtomicBool::new(false));
    let watcher_config = watcher::Config::default().labels(&selector);
    let reflect = reflector::reflector(writer, watcher(pods, watcher_config));

    tracing::info!(
        namespace = k8s_namespace,
        selector = selector,
        "Starting pod reflector for worker endpoint resolution"
    );

    let store_for_wait = store.clone();
    tokio::spawn(async move {
        tokio::pin!(reflect);
        while reflect.next().await.is_some() {}
        tracing::warn!("Pod reflector stream ended unexpectedly");
    });

    // Wait for the initial LIST to populate the store so the first inference
    // request after startup doesn't race against an empty cache. Bounded so
    // we don't block startup forever if the API server is slow.
    match tokio::time::timeout(Duration::from_secs(30), store_for_wait.wait_until_ready()).await {
        Ok(Ok(())) => {
            ready.store(true, Ordering::Release);
            tracing::info!("Pod reflector initial LIST sync complete");
        }
        Ok(Err(e)) => {
            tracing::warn!(
                error = %e,
                "Pod reflector writer was dropped before initial LIST completed; \
                 returning 503 until ready"
            );
        }
        Err(_) => {
            tracing::warn!(
                "Pod reflector initial LIST sync timed out after 30s; returning 503 until ready"
            );
            let store_for_background_wait = store.clone();
            let ready_for_background_wait = ready.clone();
            tokio::spawn(async move {
                match store_for_background_wait.wait_until_ready().await {
                    Ok(()) => {
                        ready_for_background_wait.store(true, Ordering::Release);
                        tracing::info!("Pod reflector became ready after startup timeout");
                    }
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            "Pod reflector writer dropped while waiting in background; \
                             store will remain not-ready"
                        );
                    }
                }
            });
        }
    }

    Ok((store, ready))
}

fn spawn_prefill_discovery_watcher(
    drt: DistributedRuntime,
    target_namespace: String,
    tx: tokio::sync::oneshot::Sender<dynamo_runtime::component::Endpoint>,
) {
    tokio::spawn(async move {
        let discovery = drt.discovery();
        tracing::info!(
            namespace = target_namespace,
            "Watching for prefill workers..."
        );

        loop {
            if let Ok(instances) = discovery.list(DiscoveryQuery::AllModels).await {
                for instance in instances {
                    if let DiscoveryInstance::Model {
                        namespace,
                        component,
                        endpoint,
                        ..
                    } = &instance
                    {
                        if namespace != &target_namespace {
                            continue;
                        }

                        let card = match instance.deserialize_model::<ModelDeploymentCard>() {
                            Ok(card) => card,
                            Err(_) => continue,
                        };

                        if !card.model_type.supports_prefill()
                            || card.model_type.supports_chat()
                            || card.model_type.supports_completions()
                        {
                            continue;
                        }

                        tracing::info!(
                            model_name = card.name(),
                            namespace = namespace.as_str(),
                            "Prefill worker discovered, activating PrefillRouter"
                        );

                        if let Ok(ns) = drt.namespace(namespace)
                            && let Ok(comp) = ns.component(component)
                        {
                            let ep = comp.endpoint(endpoint);
                            if tx.send(ep).is_err() {
                                tracing::debug!("PrefillRouter activation channel already closed");
                            }
                            return;
                        }
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });
}

// ---------------------------------------------------------------------------
// EndpointPicker trait implementation (mirrors Go LW-EPP from GAIE #2834)
// ---------------------------------------------------------------------------

/// Narrow `endpoints` down to only those whose address (or address:port)
/// appears in the `candidate_subset` sent via `envoy.lb.subset_hint`.
/// If `candidate_subset` is empty, returns the full list unchanged.
fn apply_subset_filter<'a>(
    endpoints: &'a [Endpoint],
    candidate_subset: &[String],
) -> Vec<&'a Endpoint> {
    if candidate_subset.is_empty() {
        return endpoints.iter().collect();
    }

    let candidates: HashSet<&str> = candidate_subset.iter().map(|s| s.as_str()).collect();
    endpoints
        .iter()
        .filter(|ep| {
            candidates.contains(ep.address_port().as_str())
                || candidates.contains(ep.address.as_str())
        })
        .collect()
}

#[tonic::async_trait]
impl EndpointPicker for Router {
    async fn pick(
        &self,
        req: &RequestInfo,
        endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        if !self.pod_store_ready.load(Ordering::Acquire) {
            return Err(PickError::RoutingFailed(
                "Pod reflector is not ready yet; endpoint cache is still syncing".to_string(),
            ));
        }

        // Constrain which workers the router may select.
        //
        // The ext_proc server always calls `pick()` with an empty external
        // endpoint list, so the Envoy InferencePool subset hint
        // (`req.candidate_subset`) must be intersected against the in-memory
        // pod reflector. When an external endpoint list is provided (e.g. a
        // future K8s-datastore caller), the subset is intersected against it
        // instead. In both cases a non-empty subset that matches nothing is a
        // hard NoEndpoints error — we never route outside the requested
        // subset.
        let (allowed_worker_ids, worker_map) = if endpoints.is_empty() {
            if req.candidate_subset.is_empty() {
                (None, Vec::new())
            } else {
                let ids = self.subset_to_worker_ids(&req.candidate_subset);
                if ids.is_empty() {
                    tracing::warn!(
                        subset = ?req.candidate_subset,
                        "No reflected pod matches the subset hint; refusing to route outside the subset"
                    );
                    return Err(PickError::NoEndpoints);
                }
                (Some(ids), Vec::new())
            }
        } else {
            let subset_filtered = apply_subset_filter(endpoints, &req.candidate_subset);
            if subset_filtered.is_empty() && !req.candidate_subset.is_empty() {
                tracing::warn!(
                    subset = ?req.candidate_subset,
                    total_endpoints = endpoints.len(),
                    "No endpoints match the subset hint; refusing to route outside the subset"
                );
                return Err(PickError::NoEndpoints);
            }

            if req.body.is_empty() {
                return Ok(PickResult {
                    endpoint: subset_filtered[0].address_port(),
                    ..Default::default()
                });
            }

            let wm: Vec<(u64, &Endpoint)> = subset_filtered
                .iter()
                .map(|ep| (hash_pod_name(&ep.pod_name), *ep))
                .collect();
            let ids: HashSet<u64> = wm.iter().map(|(id, _)| *id).collect();
            (Some(ids), wm)
        };

        if req.body.is_empty() {
            // No body (GET request) and no external endpoint list — resolve any
            // worker via discovery. If a subset hint is present, stay within it.
            let endpoint = match &allowed_worker_ids {
                Some(ids) => self.resolve_any_worker_endpoint_in_subset(ids),
                None => self.resolve_any_worker_endpoint(),
            }
            .ok_or(PickError::NoEndpoints)?;
            return Ok(PickResult {
                endpoint,
                ..Default::default()
            });
        }

        let body_str = std::str::from_utf8(&req.body)
            .map_err(|e| PickError::TokenizationFailed(format!("Invalid UTF-8: {e}")))?;

        let TokenizeResult {
            token_ids: tokens,
            cache_namespace: body_cache_namespace,
            priority_jump,
            strict_priority,
            routing_constraints,
            token_data_policy,
        } = self
            .tokenize(body_str)
            .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;
        let cache_namespace =
            cache_namespace_with_header_override(&req.headers, body_cache_namespace);

        // Try prefill routing first (disaggregated mode).
        //
        // If the prefill router is not activated (no prefill workers discovered yet, or the inner
        // router has been deactivated), fall back to aggregated routing.
        let prefill_result = self
            .route_prefill(
                &tokens,
                cache_namespace.clone(),
                priority_jump,
                strict_priority,
                allowed_worker_ids.clone(),
                routing_constraints.clone(),
            )
            .await;

        let is_disaggregated = match &prefill_result {
            Ok(_) => true,
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "Prefill routing failed; falling back to aggregated mode"
                );
                false
            }
        };

        // TODO(epp-atomic-admission): Replace query-only selection plus add_request
        // with one tracked operation. Propagate booking failures, use an internal
        // booking ID independent of x-request-id, handle cancellation races, roll
        // back endpoint-resolution failures, and never forward to an unbooked fallback.
        let (decode_worker, _overlap) = self
            .route_decode(
                &tokens,
                is_disaggregated,
                cache_namespace.clone(),
                priority_jump,
                strict_priority,
                allowed_worker_ids,
                routing_constraints,
            )
            .await
            .map_err(|e| PickError::RoutingFailed(e.to_string()))?;

        // TODO(epp-endpoint-reconciliation): Reconcile Dynamo discovery with the
        // pod reflector and retry selection when the chosen worker has no endpoint.
        let endpoint = if worker_map.is_empty() {
            self.resolve_worker_endpoint(decode_worker.worker_id)
                .ok_or_else(|| {
                    tracing::warn!(
                        worker_id = decode_worker.worker_id,
                        "Selected worker has no resolved endpoint"
                    );
                    PickError::NoEndpoints
                })?
        } else {
            worker_map
                .iter()
                .find(|(wid, _)| *wid == decode_worker.worker_id)
                .map(|(_, ep)| ep.address_port())
                .unwrap_or_else(|| {
                    tracing::warn!(
                        worker_id = decode_worker.worker_id,
                        "Selected worker not in endpoint list, using first available"
                    );
                    endpoints[0].address_port()
                })
        };

        // Register the request with the router for bookkeeping (load tracking).
        // Mirrors Go EPP's PreRequest() → CallAddRequest(requestID, tokenData, workerID, dpRank).
        if !req.request_id.is_empty()
            && let Err(e) = self
                .add_request(
                    &req.request_id,
                    &tokens,
                    decode_worker.worker_id,
                    decode_worker.dp_rank,
                    is_disaggregated,
                    cache_namespace,
                )
                .await
        {
            tracing::warn!(
                request_id = %req.request_id,
                error = %e,
                "Failed to register request with router bookkeeping"
            );
        }

        // Build routing headers matching the Go EPP's disagg plugin:
        // x-dynamo-worker-instance-id, x-dynamo-dp-rank,
        // x-dynamo-prefill-instance-id, x-dynamo-prefill-dp-rank, x-dynamo-routing-mode
        let mut headers = vec![
            (
                "x-dynamo-worker-instance-id".to_string(),
                format!("{}", decode_worker.worker_id),
            ),
            (
                "x-dynamo-dp-rank".to_string(),
                decode_worker.dp_rank.to_string(),
            ),
        ];

        if let Ok((prefill_worker_id, prefill_dp_rank)) = &prefill_result {
            headers.push((
                "x-dynamo-routing-mode".to_string(),
                "disaggregated".to_string(),
            ));
            headers.push((
                "x-dynamo-prefill-instance-id".to_string(),
                format!("{}", prefill_worker_id),
            ));
            if let Some(rank) = prefill_dp_rank {
                headers.push(("x-dynamo-prefill-dp-rank".to_string(), rank.to_string()));
            }
        } else {
            headers.push((
                "x-dynamo-routing-mode".to_string(),
                "aggregated".to_string(),
            ));
        }

        tracing::info!(
            worker_id = decode_worker.worker_id,
            worker_id_hex = format!("{:x}", decode_worker.worker_id),
            dp_rank = decode_worker.dp_rank,
            is_disaggregated,
            endpoint = %endpoint,
            token_count = tokens.len(),
            ?token_data_policy,
            priority_jump,
            model = %req.model,
            header_count = headers.len(),
            "Picked endpoint"
        );
        for (k, v) in &headers {
            tracing::debug!(key = %k, value = %v, "Routing header set in PickResult");
        }

        Ok(PickResult {
            endpoint,
            fallbacks: vec![],
            headers,
            strip_token_data: token_data_policy.strips(),
            token_ids: token_data_policy.injects().then_some(tokens),
        })
    }

    async fn on_prefill_complete(&self, request_id: &str) {
        if request_id.is_empty() {
            return;
        }
        if let Err(e) = self.mark_prefill_complete(request_id).await {
            tracing::debug!(
                request_id,
                error = %e,
                "Failed to mark prefill complete in router bookkeeping"
            );
        }
    }

    async fn on_request_complete(&self, request_id: &str) {
        if request_id.is_empty() {
            return;
        }
        if let Err(e) = self.free_request(request_id).await {
            tracing::debug!(
                request_id,
                error = %e,
                "Failed to free request from router bookkeeping"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_header_overrides_body_cache_namespace() {
        let headers = vec![("X-Tenant-ID".to_string(), "tenant-header".to_string())];

        assert_eq!(
            cache_namespace_with_header_override(&headers, Some("tenant-body".to_string()))
                .as_deref(),
            Some("tenant-header")
        );
    }

    #[test]
    fn empty_tenant_header_falls_back_to_body_cache_namespace() {
        let headers = vec![(HEADER_TENANT_ID.to_string(), String::new())];

        assert_eq!(
            cache_namespace_with_header_override(&headers, Some("tenant-body".to_string()))
                .as_deref(),
            Some("tenant-body")
        );
    }

    #[test]
    fn absent_cache_namespace_stays_absent() {
        assert_eq!(cache_namespace_with_header_override(&[], None), None);
    }

    /// Proves the core feature: `nvext.agent_hints.priority` lifts into a
    /// non-zero `priority_jump`, and absence collapses to `0.0`. If this
    /// regresses, the GAIE ext-proc path is back to ignoring priority.
    #[test]
    fn priority_jump_lifted_from_agent_hints_priority() {
        let with_priority: NvCreateChatCompletionRequest = serde_json::from_str(
            r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "nvext": {"agent_hints": {"priority": 5}}
                }"#,
        )
        .unwrap();
        assert_eq!(extract_priority_jump(with_priority.nvext.as_ref()), 5.0);

        let with_negative_alias: NvCreateChatCompletionRequest = serde_json::from_str(
            r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "nvext": {"agent_hints": {"latency_sensitivity": -3.5}}
                }"#,
        )
        .unwrap();
        assert_eq!(
            extract_priority_jump(with_negative_alias.nvext.as_ref()),
            0.0
        );

        let without_nvext: NvCreateChatCompletionRequest = serde_json::from_str(
            r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}]
                }"#,
        )
        .unwrap();
        assert_eq!(extract_priority_jump(without_nvext.nvext.as_ref()), 0.0);
    }

    #[test]
    fn strict_priority_lifted_from_agent_hints() {
        let with_priority: NvCreateChatCompletionRequest = serde_json::from_str(
            r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "nvext": {"agent_hints": {"strict_priority": 9}}
                }"#,
        )
        .unwrap();
        assert_eq!(extract_strict_priority(with_priority.nvext.as_ref()), 9);

        let without_nvext: NvCreateChatCompletionRequest = serde_json::from_str(
            r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}]
                }"#,
        )
        .unwrap();
        assert_eq!(extract_strict_priority(without_nvext.nvext.as_ref()), 0);
    }

    #[test]
    fn completion_agent_hints_are_lifted() {
        let request: NvCreateCompletionRequest = serde_json::from_str(
            r#"{
                "model": "test",
                "prompt": [101, 102, 103],
                "nvext": {"agent_hints": {"priority": 5, "strict_priority": 9}}
            }"#,
        )
        .unwrap();

        assert_eq!(extract_priority_jump(request.nvext.as_ref()), 5.0);
        assert_eq!(extract_strict_priority(request.nvext.as_ref()), 9);
    }

    #[test]
    fn completion_string_array_prompt_routes_first_prompt_without_token_injection() {
        let mut request = serde_json::json!({
            "model": "test",
            "prompt": ["hello", "world"]
        });

        let token_data_policy = prepare_completion_prompt_for_routing(&mut request);

        assert_eq!(token_data_policy, TokenDataPolicy::Strip);
        assert_eq!(request["prompt"], "hello");
    }

    #[test]
    fn completion_text_prompt_keeps_token_injection() {
        let mut request = serde_json::json!({
            "model": "test",
            "prompt": "hello"
        });

        let token_data_policy = prepare_completion_prompt_for_routing(&mut request);

        assert_eq!(token_data_policy, TokenDataPolicy::Inject);
        assert_eq!(request["prompt"], "hello");
    }

    #[test]
    fn completion_single_string_array_prompt_strips_token_data() {
        let mut request = serde_json::json!({
            "model": "test",
            "prompt": ["hello"]
        });

        let token_data_policy = prepare_completion_prompt_for_routing(&mut request);

        assert_eq!(token_data_policy, TokenDataPolicy::Strip);
        assert_eq!(request["prompt"], serde_json::json!(["hello"]));
    }

    #[test]
    fn completion_token_array_prompt_strips_token_data() {
        let mut request = serde_json::json!({
            "model": "test",
            "prompt": [101, 102, 103]
        });

        let token_data_policy = prepare_completion_prompt_for_routing(&mut request);

        assert_eq!(token_data_policy, TokenDataPolicy::Strip);
        assert_eq!(request["prompt"], serde_json::json!([101, 102, 103]));
    }

    #[test]
    fn completion_token_batch_prompt_routes_first_prompt_without_token_injection() {
        let mut request = serde_json::json!({
            "model": "test",
            "prompt": [[101, 102], [201, 202]]
        });

        let token_data_policy = prepare_completion_prompt_for_routing(&mut request);

        assert_eq!(token_data_policy, TokenDataPolicy::Strip);
        assert_eq!(request["prompt"], serde_json::json!([101, 102]));
    }

    #[test]
    fn completion_prompt_embeds_disables_token_injection() {
        let mut request = serde_json::json!({
            "model": "test",
            "prompt": "placeholder",
            "prompt_embeds": "encoded"
        });

        let token_data_policy = prepare_completion_prompt_for_routing(&mut request);

        assert_eq!(token_data_policy, TokenDataPolicy::Strip);
        assert_eq!(request["prompt"], "placeholder");
    }

    #[test]
    fn completion_prompt_embeds_without_prompt_routes_without_tokens() {
        let request = serde_json::json!({
            "model": "test",
            "prompt_embeds": "encoded",
            "nvext": {
                "agent_hints": {"priority": 5, "strict_priority": 9},
                "routing_constraints": {"required_taints": ["gpu=A100"]}
            }
        });

        let result = tokenize_completion_prompt_embeds(&request)
            .expect("prompt embeds should parse")
            .expect("prompt embeds should short-circuit");

        assert!(result.token_ids.is_empty());
        assert_eq!(result.priority_jump, 5.0);
        assert_eq!(result.strict_priority, 9);
        assert!(
            result
                .routing_constraints
                .required_taints
                .contains("gpu=A100")
        );
        assert_eq!(result.token_data_policy, TokenDataPolicy::Strip);
    }

    #[test]
    fn completion_prompt_embeds_helper_ignores_absent_embeds() {
        let request = serde_json::json!({
            "model": "test",
            "prompt": "hello"
        });

        assert!(
            tokenize_completion_prompt_embeds(&request)
                .expect("request should parse")
                .is_none()
        );
    }

    #[test]
    fn completion_routing_constraints_are_lifted() {
        let request: NvCreateCompletionRequest = serde_json::from_str(
            r#"{
                "model": "test",
                "prompt": [101, 102, 103],
                "nvext": {
                    "routing_constraints": {
                        "required_taints": ["gpu=A100"],
                        "preferred_taints": {"zone=west": 0.75}
                    }
                }
            }"#,
        )
        .unwrap();

        let constraints = extract_routing_constraints(request.nvext.as_ref());
        assert!(constraints.required_taints.contains("gpu=A100"));
        assert_eq!(constraints.preferred_taints.get("zone=west"), Some(&0.75));
    }
}

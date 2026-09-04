// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wraps Dynamo's KV-aware router for use from the ext_proc server.
//!
//! This is the native-Rust equivalent of the CGO bridge in
//! `lib/bindings/c/src/lib.rs`. Instead of crossing a C FFI boundary, the
//! ext_proc server calls these types directly as async Rust.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use dynamo_kv_router::config::{RouterConfigOverride, try_kv_router_config_from_dynamo_env};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};
use dynamo_llm::discovery::{ModelManager, WORKER_TYPE_DECODE};
use dynamo_llm::kv_router::prefill_router::PrefillReservation;
use dynamo_llm::kv_router::{ManagedKvRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::extensions::{
    HEADER_TENANT_ID, NvExt, last_non_empty_trimmed_value, request_cache_salt,
    routing_constraints_to_kv,
};
use dynamo_llm::types::openai::completions::NvCreateCompletionRequest;
use dynamo_protocols::types::Prompt;
use dynamo_runtime::discovery::{
    DiscoveryInstance, DiscoveryQuery, hash_container_name, hash_pod_name,
};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime};
use uuid::Uuid;

use crate::epp_router::endpoint_in_subset;
use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo, ResponseUsage};

const BOOKKEEPING_TIMEOUT: Duration = Duration::from_secs(5);
const DYN_KUBE_DISCOVERY_MODE: &str = "DYN_KUBE_DISCOVERY_MODE";

/// `(token_ids, cache_namespace, priority_jump, strict_priority,
/// routing_constraints, tokens_safe_to_inject)`, as returned by
/// [`Router::tokenize`] and its chat/completion helpers.
///
/// `tokens_safe_to_inject` is `false` when `token_ids` were computed from
/// only one prompt of a multi-prompt text batch (routing-only, matching
/// [`OpenAIPreprocessor::gather_tokens`]'s own refusal to trust `token_data`
/// for a `TextInput::Batch` of more than one prompt) — injecting them as
/// `nvext.token_data` would apply prompt 1's tokens to every split of the
/// batch. Chat and single/pre-tokenized completion requests are always safe.
type TokenizeResult = (Vec<u32>, Option<String>, f64, u32, RoutingConstraints, bool);

/// Validate `DYN_KUBE_DISCOVERY_MODE` and report whether *container* discovery
/// is in effect. Read once at startup and threaded down to the pod reflector
/// rather than re-read per pod event, since it cannot change while we run.
fn validate_kube_discovery_mode() -> Result<bool> {
    match std::env::var(DYN_KUBE_DISCOVERY_MODE) {
        Ok(mode) => validate_kube_discovery_mode_value(Some(&mode)),
        Err(std::env::VarError::NotPresent) => validate_kube_discovery_mode_value(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{DYN_KUBE_DISCOVERY_MODE} must be valid Unicode")
        }
    }
}

/// `Ok(true)` for container discovery, `Ok(false)` for pod discovery (the
/// default when unset).
fn validate_kube_discovery_mode_value(mode: Option<&str>) -> Result<bool> {
    match mode {
        None | Some("pod") => Ok(false),
        Some("container") => Ok(true),
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
    last_non_empty_trimmed_value(
        headers
            .iter()
            .filter(|(key, _)| key.eq_ignore_ascii_case(HEADER_TENANT_ID))
            .map(|(_, value)| value.as_str()),
    )
    .map(str::to_owned)
    .or(body_cache_namespace)
}

/// Name of the inference-serving HTTP port on a Dynamo worker pod.
const DYNAMO_CONTAINER_PORT_NAME: &str = "http";

/// Holds all router state needed for request routing.
///
/// This is the async-native equivalent of `RouterHandles` from the C bindings,
/// without the `block_on` / unsafe FFI overhead.
pub struct Router {
    prefill_router: Arc<PrefillRouter>,
    prefill_bookings: DashMap<String, PrefillReservation>,
    decode_router: ManagedKvRouter,
    preprocessor: Arc<OpenAIPreprocessor>,
    runtime: Runtime,
    worker_index: Arc<RwLock<WorkerEndpointIndex>>,
    pod_store_ready: Arc<AtomicBool>,
    served_model: String,
}

/// Remove and release a booking once. Both response lifecycle callbacks use
/// this helper so terminal completion before first output and duplicate signals
/// have identical behavior.
async fn release_prefill_booking(
    prefill_bookings: &DashMap<String, PrefillReservation>,
    booking_id: &str,
) {
    if let Some((_, reservation)) = prefill_bookings.remove(booking_id)
        && let Err(error) = reservation.release().await
    {
        tracing::debug!(
            reservation_id = booking_id,
            %error,
            "Failed to release native EPP prefill reservation"
        );
    }
}
impl Router {
    /// Initialize the router from discovery.
    ///
    /// This waits for at least one decode worker to appear, fetches the model
    /// card, initializes the preprocessor, and creates both routers.
    pub async fn from_discovery(namespace: &str, component: &str) -> Result<Self> {
        let container_discovery = validate_kube_discovery_mode()?;

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
        let mut kv_router_config =
            try_kv_router_config_from_dynamo_env().map_err(anyhow::Error::msg)?;
        // TODO(epp-multi-replica): Provide authoritative admission across EPP
        // replicas; replica-sync alone does not close the selection-to-booking race.
        kv_router_config.skip_initial_worker_wait = true;

        let component_handle = drt.namespace(actual_namespace)?.component(component)?;
        let endpoint = component_handle.endpoint("generate");

        let model_manager = Arc::new(ModelManager::new());

        let decode_router = model_manager
            .managed_kv_router_for_with_worker_role(
                &endpoint,
                block_size,
                Some(kv_router_config.clone()),
                None,
                bootstrap.card.worker_type,
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
            dynamo_llm::session_affinity::SessionAffinityMode::Hard,
            model_name.clone(),
            actual_namespace.to_string(),
            decode_router.load_context().load_thresholds(),
            drt.child_token(),
        );

        spawn_prefill_discovery_watcher(drt.clone(), actual_namespace.to_string(), prefill_tx);

        // Use the BASE namespace (without rolling-update suffix) for the pod
        // selector. Workers register in discovery under the suffixed namespace
        // (e.g. "atchernych-qwen-9f792849"), but the K8s pod label
        // `nvidia.com/dynamo-namespace` is always set to the base
        // ("atchernych-qwen") by the operator. Using the suffixed name here
        // would silently match zero pods during/after a DGD rolling update.
        let (worker_index, pod_store_ready) =
            spawn_pod_reflector(namespace, container_discovery).await?;

        // `model_manager` and `drt` are intentionally not stored on the
        // Router. The KV chooser, prefill router, prefill discovery watcher,
        // and pod reflector all clone whatever they need from these
        // constructor-locals before this scope ends, so dropping them here
        // does not tear down any background work.
        Ok(Self {
            prefill_router,
            prefill_bookings: DashMap::new(),
            decode_router,
            preprocessor: bootstrap.preprocessor,
            runtime,
            worker_index,
            pod_store_ready,
            served_model: model_name,
        })
    }

    /// The model this pool serves, from the discovered model card.
    ///
    /// Authoritative, unlike the `model` field of a request body, which the
    /// router accepts without checking.
    pub fn served_model(&self) -> &str {
        &self.served_model
    }

    /// Tokenize a JSON request body and extract router queue priorities and
    /// routing constraints.
    ///
    /// Returns `(token_ids, cache_namespace, priority_jump, strict_priority,
    /// routing_constraints)`. Priorities default to zero and constraints
    /// default to empty when absent. Supports both `/v1/chat/completions` and
    /// `/v1/completions` bodies; the request kind is discriminated by a
    /// non-empty `messages` array (chat) versus a `prompt` (completions).
    pub async fn tokenize(&self, request_json: &str) -> Result<TokenizeResult> {
        // Discriminating on a borrowed `Value` costs one scan plus the tree it
        // allocates; `from_value` then consumes that tree rather than re-reading
        // the body.
        //
        // A probe struct of `IgnoredAny` fields was tried here and reverted. It
        // allocates nothing, but `IgnoredAny` does not skip -- serde_json still
        // lexes every byte to find each value's end -- so the body ends up
        // tokenized end to end twice, and on large bodies the scan dominates the
        // allocation it saved. Measured, release build, 200 iterations, chat body
        // with one large user message:
        //
        //     467 B    Value 1.9us   IgnoredAny 1.1us   (probe wins)
        //      40 KB   Value 6.7us   IgnoredAny 8.5us
        //     160 KB   Value 19.7us  IgnoredAny 28.4us  (probe ~44% worse)
        //
        // Inference bodies skew large, so the crossover lands on the wrong side.
        // The real fix is not to read the body at all: this is
        // `/v1/chat/completions` vs `/v1/completions`, and the `:path`
        // pseudo-header would settle it in O(1) -- `req.headers` is already in
        // scope at the call site. That needs confirmation that Envoy's ext_proc
        // delivers pseudo-headers through to `ctx.request_headers` before being
        // relied on, which is why it is not done here.
        let value: serde_json::Value = serde_json::from_str(request_json)?;
        let has_messages = value
            .get("messages")
            .and_then(|m| m.as_array())
            .is_some_and(|messages| !messages.is_empty());
        if !has_messages && value.get("prompt").is_some() {
            let request: NvCreateCompletionRequest = serde_json::from_value(value)?;
            return self.tokenize_completion(request).await;
        }
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_value(value)?;
        self.tokenize_chat(&request)
    }

    /// Tokenize a `/v1/chat/completions` body via the chat template.
    fn tokenize_chat(
        &self,
        request: &dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest,
    ) -> Result<TokenizeResult> {
        // TODO(epp-request-routing): Reuse shared preprocessing so expected output
        // length, LoRA, pins, sessions, topology constraints, additional protocols,
        // and multimodal routing hashes are preserved.
        let priority_jump = extract_priority_jump(request.nvext.as_ref());
        let strict_priority = extract_strict_priority(request.nvext.as_ref());
        let routing_constraints = extract_routing_constraints(request.nvext.as_ref());
        let cache_namespace = request_cache_salt(request).map(str::to_owned);

        let encoding = match self.preprocessor.apply_template(request)? {
            Some(prompt) => self.preprocessor.tokenize_rendered_prompt(&prompt)?,
            None => self.preprocessor.tokenize("")?,
        };
        Ok((
            encoding.token_ids().to_vec(),
            cache_namespace,
            priority_jump,
            strict_priority,
            routing_constraints,
            true,
        ))
    }

    /// Tokenize a `/v1/completions` body.
    ///
    /// Pre-tokenized (integer) prompts route directly on their token IDs, while text prompts
    /// are tokenized as a raw completion prompt (no chat template) via the
    /// same [`OpenAIPreprocessor::gather_tokens`] path the backend uses for a
    /// live `/v1/completions` request, so the tokens computed here for
    /// routing/injection are identical to what the backend would compute on
    /// its own. Batched prompts route on the first entry, since KV prefix
    /// locality is computed per prompt — but for a multi-prompt text batch
    /// those tokens cover only prompt 1, so they are not safe to inject as
    /// `nvext.token_data` (see [`TokenizeResult`]); this matches
    /// `gather_tokens`'s own refusal to trust `token_data` for a
    /// `TextInput::Batch` of more than one prompt.
    async fn tokenize_completion(
        &self,
        request: NvCreateCompletionRequest,
    ) -> Result<TokenizeResult> {
        let priority_jump = extract_priority_jump(request.nvext.as_ref());
        let strict_priority = extract_strict_priority(request.nvext.as_ref());
        let routing_constraints = extract_routing_constraints(request.nvext.as_ref());
        let cache_namespace = request_cache_salt(&request).map(str::to_owned);

        let pre_tokenized = completion_prompt_token_ids(&request.inner.prompt);
        let (tokens, tokens_safe_to_inject) = match pre_tokenized {
            Some(ids) => (ids, true),
            None => {
                // Read the injection verdict before the request is consumed.
                let safe = completion_text_tokens_safe_to_inject(&request.inner.prompt);
                (self.tokenize_completion_text(request).await?, safe)
            }
        };

        Ok((
            tokens,
            cache_namespace,
            priority_jump,
            strict_priority,
            routing_constraints,
            tokens_safe_to_inject,
        ))
    }

    /// Tokenize `text` as a raw `/v1/completions` prompt — no chat template —
    /// via [`OpenAIPreprocessor::gather_tokens`], the same tokenization path
    /// the backend runs for a live completions request. Keeps the
    /// `nvext.token_data` injected downstream identical to what the backend
    /// would have tokenized itself, so preempting backend tokenization does
    /// not change the generated output.
    ///
    /// Consumes `request` (already parsed from the client body) with `prompt`
    /// replaced by the routing text, instead of serializing a synthetic body
    /// and re-parsing it. Takes ownership so the prompt is moved through rather
    /// than copied: cloning the request here duplicated every field including
    /// the full prompt -- for a `StringArray` batch, every prompt in the batch
    /// -- only for the next line to overwrite the one field that was expensive.
    async fn tokenize_completion_text(
        &self,
        mut request: NvCreateCompletionRequest,
    ) -> Result<Vec<u32>> {
        let prompt = std::mem::replace(&mut request.inner.prompt, Prompt::String(String::new()));
        request.inner.prompt = Prompt::String(completion_prompt_routing_text(prompt));
        let (tokens, _annotations) = self
            .preprocessor
            .gather_tokens(&request, None, None)
            .await?;
        Ok(tokens)
    }

    /// Resolve a worker_id to a pod endpoint address (ip:port).
    /// Lock-free-on-the-writer-side read from the incrementally maintained
    /// worker index — O(1), no K8s API calls, no pod scan, no worker-ID
    /// rehashing (see [`WorkerEndpointIndex`]).
    pub fn resolve_worker_endpoint(&self, worker_id: u64) -> Option<String> {
        read_index(&self.worker_index)
            .endpoints
            .get(&worker_id)
            .cloned()
    }

    /// Resolve any available worker to its endpoint address (ip:port).
    /// Used for body-less requests (GET /v1/models) where we just need any backend.
    pub fn resolve_any_worker_endpoint(&self) -> Option<String> {
        read_index(&self.worker_index)
            .endpoints
            .values()
            .next()
            .cloned()
    }

    /// Resolve any reflected worker whose worker_id is in `allowed`.
    /// Used for body-less requests that still carry an Envoy subset hint, so
    /// we never resolve a backend outside the requested subset.
    fn resolve_any_worker_endpoint_in_subset(&self, allowed: &HashSet<u64>) -> Option<String> {
        let index = read_index(&self.worker_index);
        allowed
            .iter()
            .find_map(|id| index.endpoints.get(id).cloned())
    }

    /// Map an Envoy `candidate_subset` (endpoint addresses, "ip:port" or bare
    /// "ip") onto the worker IDs of the reflected pods that match it.
    ///
    /// This is how the InferencePool subset hint is honored on the hot path:
    /// the ext_proc server always calls `pick()` with an empty external
    /// endpoint list, so the subset must be intersected against the in-memory
    /// worker index rather than a caller-supplied slice. An empty result for
    /// a non-empty subset means no reflected pod matched the hint.
    ///
    /// Matches a bare-IP candidate via [`endpoint_in_subset`] (`IpAddr`), not
    /// `addr_port.split(':')`: a bracketed IPv6 endpoint (`[fd00::2]:8000`)
    /// splits into garbage on `:`, silently never matching a bare `fd00::2`
    /// candidate.
    fn subset_to_worker_ids(&self, candidate_subset: &[String]) -> HashSet<u64> {
        let candidates: HashSet<&str> = candidate_subset.iter().map(|s| s.as_str()).collect();
        let candidate_ips: HashSet<IpAddr> = candidate_subset
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        let index = read_index(&self.worker_index);
        index
            .endpoints
            .iter()
            .filter(|(_, addr_port)| endpoint_in_subset(addr_port, &candidates, &candidate_ips))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Atomically select and reserve a prefill worker.
    ///
    /// Queue priorities are forwarded to the prefill scheduler. `priority_jump`
    /// adjusts the policy score, while `strict_priority` selects the primary
    /// tier. `routing_constraints` carries the request's required/preferred
    /// taints (lifted from `nvext.routing_constraints`); a hard `required_taints`
    /// mismatch excludes a worker from selection.
    #[expect(clippy::too_many_arguments)]
    pub async fn route_prefill(
        &self,
        reservation_id: &str,
        tokens: &[u32],
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<u64>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillReservation> {
        if let Some(ref ids) = allowed_worker_ids {
            self.prefill_router.register_workers(ids);
        }

        self.prefill_router
            .reserve_prefill_worker(
                reservation_id,
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
            .map_err(|e| anyhow::anyhow!("Prefill reservation failed: {e}"))
    }

    /// Route a decode request. Returns (WorkerWithDpRank, overlap_blocks).
    ///
    /// Queue priorities are forwarded to the decode scheduler. `priority_jump`
    /// adjusts the policy score, while `strict_priority` selects the primary
    /// tier. `routing_constraints` carries the request's required/preferred
    /// taints (lifted from `nvext.routing_constraints`); a hard `required_taints`
    /// mismatch excludes a worker from selection.
    #[allow(clippy::too_many_arguments)]
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

/// Extract the router queue `priority_jump` from a request's
/// `nvext.agent_hints.priority`.
///
/// Negative priorities are clamped to `0.0` so a low-priority hint never
/// pushes a request behind FCFS arrivals (matches the standalone preprocessor
/// in `lib/llm/src/preprocessor.rs`). Falls back to the deprecated
/// `latency_sensitivity` alias for callers still on the old field name.
/// Returns `0.0` when `nvext` is absent. Shared by the chat and completion
/// paths since both carry the same `nvext` block.
fn extract_priority_jump(nvext: Option<&NvExt>) -> f64 {
    nvext
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| {
            h.priority
                .map(|p| p.max(0) as f64)
                .or(h.latency_sensitivity)
        })
        .unwrap_or(0.0)
}

fn extract_strict_priority(nvext: Option<&NvExt>) -> u32 {
    nvext
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| h.strict_priority)
        .unwrap_or(0)
}

/// Extract the router's `RoutingConstraints` from a request's
/// `nvext.routing_constraints`.
///
/// A request carrying `required_taints` must reach the same hard
/// placement check the replaced shared/FFI preprocessing applied
/// (`lib/llm/src/preprocessor.rs`); dropping this here would let a request
/// with a hard constraint land on a worker that does not satisfy it.
/// Returns an empty (no-op) `RoutingConstraints` when absent. Shared by the
/// chat and completion paths since both carry the same `nvext` block.
fn extract_routing_constraints(nvext: Option<&NvExt>) -> RoutingConstraints {
    nvext
        .and_then(|n| n.routing_constraints.clone())
        .map(routing_constraints_to_kv)
        .unwrap_or_default()
}

/// Token IDs for a pre-tokenized completion prompt.
///
/// Integer prompts route directly on
/// their token IDs. Batched token prompts route on the first non-empty entry,
/// since KV prefix locality is computed per prompt. Returns `None` for text
/// prompts, which must go through the tokenizer instead.
fn completion_prompt_token_ids(prompt: &Prompt) -> Option<Vec<u32>> {
    match prompt {
        Prompt::IntegerArray(ids) => Some(ids.clone()),
        Prompt::ArrayOfIntegerArray(batches) => Some(
            batches
                .iter()
                .find(|ids| !ids.is_empty())
                .cloned()
                .unwrap_or_default(),
        ),
        Prompt::String(_) | Prompt::StringArray(_) => None,
    }
}

/// Routing text for a text completion prompt (the first prompt in a batch).
/// Returns an empty string for pre-tokenized prompts, which never reach this
/// path.
/// Takes the prompt by value so the routing text is moved out rather than
/// copied; this runs per request, and a batch's first prompt can be large.
fn completion_prompt_routing_text(prompt: Prompt) -> String {
    match prompt {
        Prompt::String(text) => text,
        Prompt::StringArray(texts) => texts.into_iter().next().unwrap_or_default(),
        Prompt::IntegerArray(_) | Prompt::ArrayOfIntegerArray(_) => String::new(),
    }
}

/// Whether tokens computed via [`completion_prompt_routing_text`] are safe to
/// inject as `nvext.token_data`.
///
/// They cover only prompt 1, so injecting them is only safe when the request
/// is not a multi-prompt text batch — otherwise the backend applies prompt
/// 1's tokens to every split of the batch (see [`TokenizeResult`]).
fn completion_text_tokens_safe_to_inject(prompt: &Prompt) -> bool {
    !matches!(prompt, Prompt::StringArray(texts) if texts.len() > 1)
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
/// Returns `None` if the pod has no IP, its IP doesn't parse, no container
/// exposes a port named `http`, or that port is out of `u16` range — we
/// never silently route to a guessed or malformed address.
///
/// Formats via `SocketAddr`, not `format!("{ip}:{port}")`: `pod_ip` is a bare
/// (unbracketed) address, so an IPv6 pod would otherwise produce an
/// ambiguous, unparseable string like `fd00::2:8000`. `SocketAddr`'s
/// `Display` brackets IPv6 for us.
fn pod_endpoint_address(pod: &k8s_openapi::api::core::v1::Pod) -> Option<String> {
    let ip: IpAddr = pod.status.as_ref()?.pod_ip.as_ref()?.parse().ok()?;
    let port: u16 = pod
        .spec
        .as_ref()?
        .containers
        .iter()
        .filter_map(|c| c.ports.as_ref())
        .flatten()
        .find(|p| p.name.as_deref() == Some(DYNAMO_CONTAINER_PORT_NAME))
        .map(|p| p.container_port)?
        .try_into()
        .ok()?;
    Some(SocketAddr::new(ip, port).to_string())
}

/// An externally supplied [`Endpoint`] rendered the way [`WorkerEndpointIndex`]
/// stores addresses, so the two can be compared.
///
/// [`Endpoint::address_port`] builds its string with `format!("{ip}:{port}")`,
/// which leaves an IPv6 literal unbracketed (`fd00::2:8000`), while the index
/// stores `SocketAddr`-rendered addresses (`[fd00::2]:8000`). Comparing the two
/// forms directly matches on IPv4 and silently never matches on IPv6, so both
/// sides go through `SocketAddr` here. Returns `None` for an address or port
/// that does not parse, which is not a routable endpoint either way.
fn indexed_endpoint_address(endpoint: &Endpoint) -> Option<String> {
    let ip: IpAddr = endpoint.address.parse().ok()?;
    let port: u16 = endpoint.port.parse().ok()?;
    Some(SocketAddr::new(ip, port).to_string())
}

/// The worker instance IDs `pod` is currently known under, per discovery mode:
/// its pod-level identity under pod discovery, or each `Ready` container's
/// identity under container discovery (`DYN_KUBE_DISCOVERY_MODE=container`,
/// e.g. intra-pod GMS failover). The HTTP endpoint stays pod-level either way
/// (see [`pod_endpoint_address`]).
///
/// The mode is exclusive, and so are the identities. A worker process picks one
/// `KubeDiscoveryTarget` from its own mode, so under container discovery
/// nothing registers under the bare pod identity — emitting it there would
/// invent a worker that `register_workers` upserts at zero load and zero KV
/// overlap, making it the most attractive candidate the scheduler sees.
/// `"main"` hashes to the pod identity (`hash_container_name`), so a pod whose
/// main container is Ready still contributes that id through the container
/// path; one whose main container is *not* Ready correctly contributes nothing
/// for it.
///
/// Symmetrically, under pod discovery per-container ids would be the phantoms,
/// which is why the container arm is gated at all.
///
/// Not fixed here: under container discovery this still trusts every `Ready`
/// container to be a Dynamo worker, matching `extract_ready_containers` in
/// `lib/runtime`. A pod carrying a Ready non-worker sidecar would contribute an
/// id for it. Nothing in the Pod distinguishes the two — the port name this
/// file keys on is the generic `"http"` — so filtering here would risk dropping
/// real workers; the authoritative fix is to intersect against the router's
/// registered worker set.
///
/// An unnamed pod yields nothing at all, container ids included:
/// `hash_container_name("", …)` is the same value for every unnamed pod, so
/// they would alias each other's endpoints in [`WorkerEndpointIndex`].
fn pod_worker_ids(
    pod: &k8s_openapi::api::core::v1::Pod,
    container_discovery: bool,
) -> impl Iterator<Item = u64> + '_ {
    let pod_name = pod.metadata.name.as_deref().unwrap_or_default();
    let named = !pod_name.is_empty();
    let pod_id = (named && !container_discovery).then(|| hash_pod_name(pod_name));
    let container_ids = (container_discovery && named)
        .then_some(pod.status.as_ref())
        .flatten()
        .and_then(|s| s.container_statuses.as_ref())
        .into_iter()
        .flatten()
        .filter(|cs| cs.ready)
        .map(move |cs| hash_container_name(pod_name, &cs.name));
    pod_id.into_iter().chain(container_ids)
}

/// Derived index mapping each worker ID (pod-level and, under container
/// discovery mode, per-ready-container — see [`pod_worker_ids`]) to its pod's
/// `ip:port` HTTP endpoint.
///
/// Incrementally maintained from the pod reflector's per-object
/// `Apply`/`Delete` events (see [`spawn_pod_reflector`]) so request-path
/// lookups (`resolve_worker_endpoint` and friends) are O(1) map reads that
/// never rescan the pod set or re-derive worker IDs — `pod_worker_ids`
/// allocates per hashed container name, so recomputing it per request scaled
/// with pod/container count and allocated on every miss.
#[derive(Default)]
struct WorkerEndpointIndex {
    /// Whether `DYN_KUBE_DISCOVERY_MODE=container` is in effect, resolved once
    /// at startup. Held here rather than read per event so the index cannot
    /// disagree with itself between two pods. Defaults to `false` (pod
    /// discovery), which is the mode that must not invent per-container ids.
    container_discovery: bool,
    /// worker_id -> endpoint.
    endpoints: HashMap<u64, String>,
    /// pod name -> worker_ids currently registered for that pod, so a later
    /// `Apply`/`Delete` for the same pod retracts exactly the ids it
    /// previously added instead of requiring a full rebuild.
    by_pod: HashMap<String, Vec<u64>>,
}

impl WorkerEndpointIndex {
    fn new(container_discovery: bool) -> Self {
        Self {
            container_discovery,
            ..Default::default()
        }
    }

    /// Apply one pod's current state: drop any ids it previously registered,
    /// then add its current worker IDs mapped to its endpoint. A pod with no
    /// worker IDs (unnamed, matching [`pod_worker_ids`]) or no resolvable
    /// HTTP endpoint contributes no entries — matching the pre-index
    /// `resolve_worker_endpoint`'s behavior of returning `None` for a
    /// worker_id whose pod lacks one.
    fn upsert(&mut self, pod: &k8s_openapi::api::core::v1::Pod) {
        let pod_name = pod.metadata.name.clone().unwrap_or_default();
        self.retract(&pod_name);
        let Some(endpoint) = pod_endpoint_address(pod) else {
            return;
        };
        let ids: Vec<u64> = pod_worker_ids(pod, self.container_discovery).collect();
        if ids.is_empty() {
            return;
        }
        for &id in &ids {
            self.endpoints.insert(id, endpoint.clone());
        }
        self.by_pod.insert(pod_name, ids);
    }

    /// Drop a pod's worker IDs entirely (deletion).
    fn remove(&mut self, pod: &k8s_openapi::api::core::v1::Pod) {
        let pod_name = pod.metadata.name.as_deref().unwrap_or_default();
        self.retract(pod_name);
    }

    fn retract(&mut self, pod_name: &str) {
        if let Some(ids) = self.by_pod.remove(pod_name) {
            for id in ids {
                self.endpoints.remove(&id);
            }
        }
    }

    /// Drop every entry — used when the reflector stream ends, since a frozen
    /// snapshot must not keep answering lookups as if it were still live.
    fn clear(&mut self) {
        // Deliberately leaves `container_discovery` alone: it is startup
        // configuration, not reflector state.
        self.endpoints.clear();
        self.by_pod.clear();
    }

    /// Full recompute from the reflector's current pod set. Only called once,
    /// at `InitDone` (see guardrail on incremental vs. recompute) — steady
    /// state applies each pod's delta via `upsert`/`remove` instead.
    fn rebuild(
        &mut self,
        store: &kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    ) {
        self.endpoints.clear();
        self.by_pod.clear();
        for pod in store.state() {
            self.upsert(&pod);
        }
    }
}

/// Read the worker index, recovering from a poisoned lock.
///
/// The index is derived state the reflector rebuilds, so a writer that panicked
/// mid-update leaves it stale at worst. Propagating the poison instead would
/// panic every subsequent `pick()` inside the tonic handler while the health
/// server on its own port keeps answering SERVING — a pod that fails 100% of
/// ext_proc calls and is never restarted. Stale routing degrades; poison does
/// not.
fn read_index(index: &RwLock<WorkerEndpointIndex>) -> RwLockReadGuard<'_, WorkerEndpointIndex> {
    index.read().unwrap_or_else(PoisonError::into_inner)
}

/// Write the worker index, recovering from a poisoned lock — see [`read_index`].
/// Without this a single writer panic would also kill every later update.
fn write_index(index: &RwLock<WorkerEndpointIndex>) -> RwLockWriteGuard<'_, WorkerEndpointIndex> {
    index.write().unwrap_or_else(PoisonError::into_inner)
}

/// Clears readiness when the reflector stops for *any* reason.
///
/// The stream-end path below lowers readiness explicitly, before clearing the
/// index, so the pod stops advertising itself while it can still answer. This
/// guard is the backstop for the path that cannot do that: a panic unwinding
/// out of the loop, which would otherwise leave a dead reflector task behind a
/// pod still reporting Ready.
struct ReflectorReadinessGuard(Arc<AtomicBool>);

impl Drop for ReflectorReadinessGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Drive the pod reflector stream, maintaining `index` from its per-object
/// events until the stream ends.
///
/// Split out of [`spawn_pod_reflector`] so the stream-end path can be tested
/// without a Kubernetes API server.
async fn run_pod_reflector(
    reflect: impl futures::Stream<
        Item = Result<
            kube::runtime::watcher::Event<k8s_openapi::api::core::v1::Pod>,
            kube::runtime::watcher::Error,
        >,
    >,
    store: kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    index: Arc<RwLock<WorkerEndpointIndex>>,
    ready: Arc<AtomicBool>,
) {
    use futures::StreamExt;
    use kube::runtime::watcher;

    let _readiness_guard = ReflectorReadinessGuard(ready.clone());

    tokio::pin!(reflect);
    loop {
        match reflect.next().await {
            None => {
                // Stop advertising readiness BEFORE dropping the index. The
                // runner mirrors this flag onto the gRPC health status
                // (`runner.rs::serve`), which documents it as a live signal
                // that flips both ways; leaving it set would keep the pod 1/1
                // Ready and SERVING with an empty index, no reflector task,
                // and every lookup returning None -- a state Kubernetes never
                // restarts out of. Mirrors pod_discovery.rs's stream-end path.
                tracing::warn!("Pod reflector stream ended unexpectedly; marking not ready");
                ready.store(false, Ordering::Release);
                write_index(&index).clear();
                break;
            }
            // During a relist the reflector emits Init + one InitApply per
            // pod + InitDone; rebuild once from the completed store
            // instead of applying per-object deltas mid-relist.
            Some(Ok(watcher::Event::Init | watcher::Event::InitApply(_))) => continue,
            Some(Ok(watcher::Event::InitDone)) => {
                write_index(&index).rebuild(&store);
                // Raise readiness here, and only here: this task owns the
                // reflector's whole lifecycle, so it is the only writer that
                // cannot be reordered against the stream-end/panic lowering
                // below. Raising it from the startup waiter instead let a
                // reflector that died right after the initial LIST be
                // overwritten back to ready -- exactly the ready-but-empty pod
                // the lowering exists to prevent, since `wait_until_ready()`
                // resolves when the store applies `InitDone`, before this arm
                // runs. Ordering it after `rebuild` also closes the lesser
                // window where a request passed the gate and saw an empty index.
                ready.store(true, Ordering::Release);
            }
            Some(Ok(watcher::Event::Apply(pod))) => {
                write_index(&index).upsert(&pod);
            }
            Some(Ok(watcher::Event::Delete(pod))) => {
                write_index(&index).remove(&pod);
            }
            Some(Err(e)) => {
                tracing::warn!(error = %e, "Pod reflector watch error; retrying");
            }
        }
    }
}

/// Start a background pod reflector that watches worker pods matching the
/// InferencePool selector and incrementally maintains a [`WorkerEndpointIndex`]
/// from its per-object events — O(1) request-path lookups, no K8s API calls
/// and no pod rescans on the hot path.
async fn spawn_pod_reflector(
    dynamo_namespace: &str,
    container_discovery: bool,
) -> Result<(Arc<RwLock<WorkerEndpointIndex>>, Arc<AtomicBool>)> {
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
    let index: Arc<RwLock<WorkerEndpointIndex>> =
        Arc::new(RwLock::new(WorkerEndpointIndex::new(container_discovery)));

    tracing::info!(
        namespace = k8s_namespace,
        selector = selector,
        "Starting pod reflector for worker endpoint resolution"
    );

    let store_for_wait = store.clone();
    tokio::spawn(run_pod_reflector(
        reflect,
        store,
        index.clone(),
        ready.clone(),
    ));

    // Wait for the initial LIST to populate the store so the first inference
    // request after startup doesn't race against an empty cache. Bounded so we
    // don't block startup forever if the API server is slow.
    //
    // This only *observes* the sync; readiness is raised by the reflector task
    // at `InitDone` (see `run_pod_reflector`). Storing it here as well would
    // make two independent tasks write one flag, and the loser of that race
    // decides -- so a reflector that died immediately after the initial LIST
    // could be marked ready by this task afterwards. Nothing is needed on the
    // timeout path either: whenever `InitDone` does arrive, the reflector
    // raises readiness itself, so the pod recovers without a second waiter.
    match tokio::time::timeout(Duration::from_secs(30), store_for_wait.wait_until_ready()).await {
        Ok(Ok(())) => tracing::info!("Pod reflector initial LIST sync complete"),
        Ok(Err(e)) => tracing::warn!(
            error = %e,
            "Pod reflector writer was dropped before initial LIST completed; \
             returning 503 until ready"
        ),
        Err(_) => tracing::warn!(
            "Pod reflector initial LIST sync timed out after 30s; returning 503 until ready"
        ),
    }

    Ok((index, ready))
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
// EndpointPicker trait implementation
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

            // Resolve each supplied endpoint to the worker IDs the reflector
            // actually holds at that address, rather than re-deriving
            // `hash_pod_name(&ep.pod_name)`.
            //
            // Only pod discovery registers a worker under its pod identity.
            // Under container discovery each engine container registers under
            // its own (`KubeDiscoveryTarget::Container`), so a hand-built pod
            // hash names a worker present in no registry: `register_workers`
            // would upsert it at zero load and zero KV overlap, making it the
            // scheduler's most attractive candidate, and the reverse lookup
            // below would then fail to match and silently forward to
            // `endpoints[0]`. The index is the one place that knows which
            // identity scheme is in effect (see `pod_worker_ids`), so ask it.
            let wm: Vec<(u64, &Endpoint)> = {
                let index = read_index(&self.worker_index);
                let mut wm = Vec::new();
                for ep in &subset_filtered {
                    let Some(addr) = indexed_endpoint_address(ep) else {
                        continue;
                    };
                    wm.extend(
                        index
                            .endpoints
                            .iter()
                            .filter(|(_, indexed)| **indexed == addr)
                            .map(|(id, _)| (*id, *ep)),
                    );
                }
                wm
            };
            if wm.is_empty() {
                tracing::warn!(
                    supplied_endpoints = subset_filtered.len(),
                    "No supplied endpoint resolves to a discovered worker; \
                     refusing to route to an unidentifiable backend"
                );
                return Err(PickError::NoEndpoints);
            }
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

        let (
            tokens,
            body_cache_namespace,
            priority_jump,
            strict_priority,
            routing_constraints,
            tokens_safe_to_inject,
        ) = self
            .tokenize(body_str)
            .await
            .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;
        let cache_namespace =
            cache_namespace_with_header_override(&req.headers, body_cache_namespace);
        let reservation_id = Uuid::new_v4().to_string();

        // Try prefill routing first (disaggregated mode).
        //
        // If the prefill router is not activated (no prefill workers discovered yet, or the inner
        // router has been deactivated), fall back to aggregated routing.
        let prefill_booking = self
            .route_prefill(
                &format!("epp-prefill/{reservation_id}"),
                &tokens,
                cache_namespace.clone(),
                priority_jump,
                strict_priority,
                allowed_worker_ids.clone(),
                routing_constraints.clone(),
            )
            .await;

        let is_disaggregated = match &prefill_booking {
            Ok(_) => true,
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "Prefill routing failed; falling back to aggregated mode"
                );
                false
            }
        };

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
        if let Err(e) = self
            .add_request(
                &reservation_id,
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

        let prefill_worker = prefill_booking
            .as_ref()
            .ok()
            .map(|booking| (booking.worker_id(), booking.dp_rank()));
        if let Ok(booking) = prefill_booking {
            self.prefill_bookings
                .insert(reservation_id.clone(), booking);
        }

        // Build routing headers: x-dynamo-worker-instance-id, x-dynamo-dp-rank,
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

        if let Some((prefill_worker_id, prefill_dp_rank)) = prefill_worker {
            headers.push((
                "x-dynamo-routing-mode".to_string(),
                "disaggregated".to_string(),
            ));
            headers.push((
                "x-dynamo-prefill-instance-id".to_string(),
                format!("{}", prefill_worker_id),
            ));
            if let Some(prefill_dp_rank) = prefill_dp_rank {
                headers.push((
                    "x-dynamo-prefill-dp-rank".to_string(),
                    prefill_dp_rank.to_string(),
                ));
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
            priority_jump,
            model = %req.model,
            header_count = headers.len(),
            "Picked endpoint"
        );
        for (k, v) in &headers {
            tracing::debug!(key = %k, value = %v, "Routing header set in PickResult");
        }

        // Only inject token_data when it covers the whole request; a
        // multi-prompt text batch's tokens cover prompt 1 alone and the
        // backend applies nvext.token_data to every split (see
        // `TokenizeResult`), so omit it and let the backend tokenize each
        // prompt itself.
        let token_ids = tokens_safe_to_inject.then_some(tokens);

        Ok(PickResult {
            endpoint,
            fallbacks: vec![],
            headers,
            token_ids,
            reservation_id: Some(reservation_id),
        })
    }

    async fn on_prefill_complete(&self, booking_id: &str) {
        if booking_id.is_empty() {
            return;
        }
        release_prefill_booking(&self.prefill_bookings, booking_id).await;
        if let Err(e) = self.mark_prefill_complete(booking_id).await {
            tracing::debug!(
                reservation_id = booking_id,
                error = %e,
                "Failed to mark prefill complete in router bookkeeping"
            );
        }
    }

    async fn on_request_complete_with_usage(&self, booking_id: &str, usage: Option<ResponseUsage>) {
        if booking_id.is_empty() {
            return;
        }
        if let Some(usage) = usage {
            tracing::debug!(
                reservation_id = booking_id,
                prompt_tokens = ?usage.prompt_tokens,
                completion_tokens = ?usage.completion_tokens,
                total_tokens = ?usage.total_tokens,
                cached_tokens = ?usage.cached_tokens,
                "Request complete with usage"
            );
        }
        release_prefill_booking(&self.prefill_bookings, booking_id).await;
        if let Err(e) = self.free_request(booking_id).await {
            tracing::debug!(
                reservation_id = booking_id,
                error = %e,
                "Failed to free request from router bookkeeping"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::Pod;

    use std::sync::{Arc, atomic::Ordering};

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
        let headers = vec![
            (HEADER_TENANT_ID.to_string(), String::new()),
            ("X-Tenant-ID".to_string(), "   ".to_string()),
        ];

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

    #[test]
    fn last_non_empty_trimmed_tenant_header_wins() {
        let headers = vec![
            (HEADER_TENANT_ID.to_string(), "tenant-client".to_string()),
            ("X-Tenant-ID".to_string(), "   ".to_string()),
            (HEADER_TENANT_ID.to_string(), " tenant-gateway ".to_string()),
        ];

        assert_eq!(
            cache_namespace_with_header_override(&headers, Some("tenant-body".to_string()))
                .as_deref(),
            Some("tenant-gateway")
        );
    }

    /// Proves the core feature: `nvext.agent_hints.priority` lifts into a
    /// non-zero `priority_jump`, and absence collapses to `0.0`. If this
    /// regresses, the GAIE ext-proc path is back to ignoring priority.
    #[test]
    fn priority_jump_lifted_from_agent_hints_priority() {
        let with_priority: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
                r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "nvext": {"agent_hints": {"priority": 5}}
                }"#,
            )
            .unwrap();
        assert_eq!(extract_priority_jump(with_priority.nvext.as_ref()), 5.0);

        let without_nvext: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
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
        let with_priority: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
                r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "nvext": {"agent_hints": {"strict_priority": 9}}
                }"#,
            )
            .unwrap();
        assert_eq!(extract_strict_priority(with_priority.nvext.as_ref()), 9);

        let without_nvext: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
                r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}]
                }"#,
            )
            .unwrap();
        assert_eq!(extract_strict_priority(without_nvext.nvext.as_ref()), 0);
    }

    /// A `/v1/completions` request carries the same `nvext` block as chat, so
    /// the shared priority extractors must lift `agent_hints` from it too.
    #[test]
    fn priority_lifted_from_completion_nvext() {
        let request: NvCreateCompletionRequest = serde_json::from_str(
            r#"{
                "model": "test",
                "prompt": "hello world",
                "nvext": {"agent_hints": {"priority": 4, "strict_priority": 7}}
            }"#,
        )
        .unwrap();
        assert_eq!(extract_priority_jump(request.nvext.as_ref()), 4.0);
        assert_eq!(extract_strict_priority(request.nvext.as_ref()), 7);
    }

    /// Proves the hard-constraint feature: `nvext.routing_constraints.
    /// required_taints` lifts into a non-empty `RoutingConstraints`, and
    /// absence collapses to the empty default. If this regresses, a request
    /// with `required_taints` can land on a worker that does not satisfy its
    /// hard placement requirement.
    #[test]
    fn routing_constraints_lifted_from_nvext() {
        let with_constraints: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
                r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "nvext": {"routing_constraints": {"required_taints": ["gpu=h100"]}}
                }"#,
            )
            .unwrap();
        let constraints = extract_routing_constraints(with_constraints.nvext.as_ref());
        assert!(constraints.has_hard_constraints());
        assert!(constraints.required_taints.contains("gpu=h100"));

        let without_nvext: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
                r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}]
                }"#,
            )
            .unwrap();
        assert!(extract_routing_constraints(without_nvext.nvext.as_ref()).is_empty());
    }

    /// A `/v1/completions` request carries the same `nvext` block as chat, so
    /// `required_taints` must be lifted from it too, not silently dropped to
    /// `RoutingConstraints::default()`.
    #[test]
    fn routing_constraints_lifted_from_completion_nvext() {
        let request: NvCreateCompletionRequest = serde_json::from_str(
            r#"{
                "model": "test",
                "prompt": "hello world",
                "nvext": {"routing_constraints": {"required_taints": ["zone=us-east-1a"]}}
            }"#,
        )
        .unwrap();
        let constraints = extract_routing_constraints(request.nvext.as_ref());
        assert!(constraints.required_taints.contains("zone=us-east-1a"));
    }

    /// Regression test for a text `/v1/completions` prompt: the tokens
    /// `Router::tokenize_completion_text` computes for routing/injection must
    /// be byte-identical to what `OpenAIPreprocessor::gather_tokens` produces
    /// for a real client-shaped completions request with the same prompt —
    /// i.e. ext-proc must reuse the backend's raw completion tokenization,
    /// not substitute a different one. It must also differ from the old
    /// (buggy) chat-template tokenization of the same text: injecting
    /// chat-shaped tokens as `nvext.token_data` for a completions request
    /// changes the literal prompt the model generates from, not just routing.
    #[tokio::test]
    async fn text_completion_tokens_match_backend_raw_completion_tokenization() {
        let mdc = ModelDeploymentCard::load_from_disk(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../lib/llm/tests/data/sample-models/TinyLlama_v1.1"
            ),
            None,
        )
        .expect("load fixture model card");
        let preprocessor =
            OpenAIPreprocessor::new(mdc).expect("build preprocessor from fixture card");

        let text = "The capital of France is";

        // Mirrors what `Router::tokenize_completion_text` sends to
        // `gather_tokens`: the client's own request with `prompt` overwritten.
        let ext_proc_request: NvCreateCompletionRequest = serde_json::from_str(
            &serde_json::json!({"model": "default", "prompt": text}).to_string(),
        )
        .unwrap();
        let (ext_proc_tokens, _) = preprocessor
            .gather_tokens(&ext_proc_request, None, None)
            .await
            .expect("gather_tokens on ext-proc's minimal completion request");

        // A real client-shaped `/v1/completions` request for the same prompt,
        // tokenized via the same path the backend runs on a live request.
        let backend_request: NvCreateCompletionRequest = serde_json::from_str(
            &serde_json::json!({
                "model": "test-model",
                "prompt": text,
                "max_tokens": 16,
            })
            .to_string(),
        )
        .unwrap();
        let (backend_tokens, _) = preprocessor
            .gather_tokens(&backend_request, None, None)
            .await
            .expect("gather_tokens on a real completions request");

        assert!(!ext_proc_tokens.is_empty());
        assert_eq!(
            ext_proc_tokens, backend_tokens,
            "ext-proc's routed tokens must match the backend's own raw completion tokenization"
        );

        // The old, buggy path wrapped the prompt as a chat user message and ran
        // it through the chat template — assert that no longer happens.
        let chat_wrapped: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
                &serde_json::json!({
                    "model": "default",
                    "messages": [{"role": "user", "content": text}],
                })
                .to_string(),
            )
            .unwrap();
        let chat_tokens = match preprocessor
            .apply_template(&chat_wrapped)
            .expect("apply_template on chat-wrapped request")
        {
            Some(rendered) => preprocessor
                .tokenize_rendered_prompt(&rendered)
                .expect("tokenize rendered chat prompt")
                .token_ids()
                .to_vec(),
            None => preprocessor
                .tokenize("")
                .expect("tokenize empty prompt")
                .token_ids()
                .to_vec(),
        };
        assert_ne!(
            ext_proc_tokens, chat_tokens,
            "raw completion tokenization must differ from chat-template tokenization"
        );
    }

    /// A multi-prompt text batch routes on prompt 1's tokens, but those
    /// tokens must never be injected as `nvext.token_data`: the backend
    /// applies injected `token_data` to every split of the batch, so
    /// injecting prompt 1's tokens would silently run prompt 2 (and beyond)
    /// on prompt 1's tokens instead of its own. This exercises the same
    /// `completion_prompt_routing_text` -> `gather_tokens` ->
    /// `completion_text_tokens_safe_to_inject` -> `token_ids` gating chain
    /// that `Router::tokenize_completion` and `Router::pick` run, end to end.
    #[tokio::test]
    async fn multi_prompt_text_batch_tokens_are_not_injected_as_token_data() {
        let mdc = ModelDeploymentCard::load_from_disk(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../lib/llm/tests/data/sample-models/TinyLlama_v1.1"
            ),
            None,
        )
        .expect("load fixture model card");
        let preprocessor =
            OpenAIPreprocessor::new(mdc).expect("build preprocessor from fixture card");

        let prompt1 = "The capital of France is";
        let prompt2 = "The capital of Japan is";
        let batch_request: NvCreateCompletionRequest = serde_json::from_str(
            &serde_json::json!({"model": "default", "prompt": [prompt1, prompt2]}).to_string(),
        )
        .unwrap();

        // Mirrors `Router::tokenize_completion`: route on prompt 1's tokens.
        let routing_text = completion_prompt_routing_text(batch_request.inner.prompt.clone());
        assert_eq!(routing_text, prompt1);
        let routing_request: NvCreateCompletionRequest = serde_json::from_str(
            &serde_json::json!({"model": "default", "prompt": routing_text}).to_string(),
        )
        .unwrap();
        let (routed_tokens, _) = preprocessor
            .gather_tokens(&routing_request, None, None)
            .await
            .expect("gather_tokens on the routing prompt");

        // The multi-entry batch is not safe to inject.
        assert!(!completion_text_tokens_safe_to_inject(
            &batch_request.inner.prompt
        ));
        let token_ids: Option<Vec<u32>> =
            completion_text_tokens_safe_to_inject(&batch_request.inner.prompt)
                .then_some(routed_tokens.clone());
        assert_eq!(
            token_ids, None,
            "token_data must be omitted for a multi-prompt text batch, not prompt 1's tokens"
        );

        // Prove why: prompt 2 tokenizes to something different from what was
        // routed on, so injecting `routed_tokens` for the whole batch would
        // have run prompt 2 on prompt 1's tokens.
        let prompt2_request: NvCreateCompletionRequest = serde_json::from_str(
            &serde_json::json!({"model": "default", "prompt": prompt2}).to_string(),
        )
        .unwrap();
        let (prompt2_tokens, _) = preprocessor
            .gather_tokens(&prompt2_request, None, None)
            .await
            .expect("gather_tokens on prompt 2");
        assert_ne!(
            routed_tokens, prompt2_tokens,
            "prompt 2 must tokenize differently from the routed prompt 1 tokens"
        );

        // A single-entry batch has nothing to split against, so it remains
        // safe to inject.
        let single_entry_request: NvCreateCompletionRequest = serde_json::from_str(
            &serde_json::json!({"model": "default", "prompt": [prompt1]}).to_string(),
        )
        .unwrap();
        assert!(completion_text_tokens_safe_to_inject(
            &single_entry_request.inner.prompt
        ));
    }

    /// Pre-tokenized `/v1/completions` prompts route directly on their token IDs.
    #[test]
    fn completion_token_prompt_uses_token_ids_directly() {
        let single: NvCreateCompletionRequest =
            serde_json::from_str(r#"{"model": "test", "prompt": [1, 2, 3, 4]}"#).unwrap();
        assert_eq!(
            completion_prompt_token_ids(&single.inner.prompt),
            Some(vec![1, 2, 3, 4])
        );

        // Batched token prompts route on the first non-empty entry.
        let batched: NvCreateCompletionRequest =
            serde_json::from_str(r#"{"model": "test", "prompt": [[10, 20], [30, 40, 50]]}"#)
                .unwrap();
        assert_eq!(
            completion_prompt_token_ids(&batched.inner.prompt),
            Some(vec![10, 20])
        );
    }

    /// Text `/v1/completions` prompts are not pre-tokenized, so they fall
    /// through to the tokenizer path with the first prompt as routing text.
    #[test]
    fn completion_string_prompt_routes_on_text() {
        let single: NvCreateCompletionRequest =
            serde_json::from_str(r#"{"model": "test", "prompt": "hello world"}"#).unwrap();
        assert_eq!(completion_prompt_token_ids(&single.inner.prompt), None);
        assert_eq!(
            completion_prompt_routing_text(single.inner.prompt.clone()),
            "hello world"
        );

        let batched: NvCreateCompletionRequest =
            serde_json::from_str(r#"{"model": "test", "prompt": ["first", "second"]}"#).unwrap();
        assert_eq!(completion_prompt_token_ids(&batched.inner.prompt), None);
        assert_eq!(
            completion_prompt_routing_text(batched.inner.prompt.clone()),
            "first"
        );
    }

    #[test]
    fn discovery_mode_accepts_pod_and_container_rejects_unknown() {
        // The bool is what gates per-container worker ids, so assert the
        // resolved mode rather than just that validation succeeded.
        assert!(
            !validate_kube_discovery_mode_value(None).unwrap(),
            "unset must default to pod discovery"
        );
        assert!(!validate_kube_discovery_mode_value(Some("pod")).unwrap());
        assert!(
            validate_kube_discovery_mode_value(Some("container")).unwrap(),
            "container mode (e.g. intra-pod GMS failover) must be accepted, not rejected at startup"
        );
        assert!(validate_kube_discovery_mode_value(Some("bogus")).is_err());
    }

    /// Builds a pod shaped like an intra-pod GMS failover worker: two engine
    /// containers with per-container readiness plus an optional
    /// sidecar-frontend exposing the pod's stable OpenAI-compatible port.
    fn failover_pod(engine_ready: &[(&str, bool)], with_sidecar_frontend: bool) -> Pod {
        use k8s_openapi::api::core::v1::{
            Container, ContainerPort, ContainerStatus, PodSpec, PodStatus,
        };
        use kube::api::ObjectMeta;

        let mut containers: Vec<Container> = engine_ready
            .iter()
            .enumerate()
            .map(|(i, (name, _))| Container {
                name: name.to_string(),
                ports: Some(vec![ContainerPort {
                    name: Some(format!("system-{i}")),
                    container_port: 9090 + i as i32,
                    ..Default::default()
                }]),
                ..Default::default()
            })
            .collect();
        if with_sidecar_frontend {
            containers.push(Container {
                name: "sidecar-frontend".to_string(),
                ports: Some(vec![ContainerPort {
                    name: Some(DYNAMO_CONTAINER_PORT_NAME.to_string()),
                    container_port: 8000,
                    ..Default::default()
                }]),
                ..Default::default()
            });
        }

        Pod {
            metadata: ObjectMeta {
                name: Some("worker-0".to_string()),
                ..Default::default()
            },
            spec: Some(PodSpec {
                containers,
                ..Default::default()
            }),
            status: Some(PodStatus {
                pod_ip: Some("10.0.0.1".to_string()),
                container_statuses: Some(
                    engine_ready
                        .iter()
                        .map(|(name, ready)| ContainerStatus {
                            name: name.to_string(),
                            ready: *ready,
                            ..Default::default()
                        })
                        .collect(),
                ),
                ..Default::default()
            }),
        }
    }

    /// Only currently-`Ready` engine containers contribute a live worker_id; a
    /// demoted/crashed standby must never be matched. The pod-level identity is
    /// *not* among them: a container-mode process registers as
    /// `KubeDiscoveryTarget::Container` (`CONTAINER_NAME` is required), so
    /// nothing in a failover pod ever registers the bare pod identity, and
    /// emitting it would put a zero-load phantom into `allowed_worker_ids`.
    #[test]
    fn pod_worker_ids_includes_ready_containers_only() {
        let pod = failover_pod(&[("engine-0", true), ("engine-1", false)], true);
        let ids: HashSet<u64> = pod_worker_ids(&pod, true).collect();

        assert!(
            !ids.contains(&hash_pod_name("worker-0")),
            "no container-mode worker registers under the bare pod identity"
        );
        assert!(ids.contains(&hash_container_name("worker-0", "engine-0")));
        assert!(
            !ids.contains(&hash_container_name("worker-0", "engine-1")),
            "the not-ready standby engine must not be a live worker_id"
        );
        assert_eq!(ids.len(), 1);
    }

    /// Builds an ordinary pod-discovery worker pod: a `main` engine container
    /// plus the sidecars a real Dynamo worker runs, all `Ready`. `simple_pod`
    /// sets no `container_statuses` at all, so it cannot reproduce what a live
    /// pod-mode pod looks like to `pod_worker_ids` — this fixture can.
    fn pod_mode_worker_pod() -> Pod {
        use k8s_openapi::api::core::v1::{
            Container, ContainerPort, ContainerStatus, PodSpec, PodStatus,
        };
        use kube::api::ObjectMeta;

        let names = ["main", "sidecar-frontend", "metrics"];
        Pod {
            metadata: ObjectMeta {
                name: Some("worker-0".to_string()),
                ..Default::default()
            },
            spec: Some(PodSpec {
                containers: names
                    .iter()
                    .map(|name| Container {
                        name: name.to_string(),
                        ports: (*name == "sidecar-frontend").then(|| {
                            vec![ContainerPort {
                                name: Some(DYNAMO_CONTAINER_PORT_NAME.to_string()),
                                container_port: 8000,
                                ..Default::default()
                            }]
                        }),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
            status: Some(PodStatus {
                pod_ip: Some("10.0.0.1".to_string()),
                container_statuses: Some(
                    names
                        .iter()
                        .map(|name| ContainerStatus {
                            name: name.to_string(),
                            ready: true,
                            ..Default::default()
                        })
                        .collect(),
                ),
                ..Default::default()
            }),
        }
    }

    /// Under pod discovery a worker registers under its pod identity alone, so
    /// a pod's ready sidecars must contribute no worker ids. Emitting them
    /// would invent workers no backend registered under: they miss
    /// `register_workers`' discovery lookup, default to `(0, 1)` with no load
    /// and no KV overlap, and so look maximally attractive to the scheduler.
    #[test]
    fn pod_worker_ids_ignores_containers_under_pod_discovery() {
        let pod = pod_mode_worker_pod();
        let ids: Vec<u64> = pod_worker_ids(&pod, false).collect();

        assert_eq!(
            ids,
            vec![hash_pod_name("worker-0")],
            "pod discovery must yield exactly the pod-level identity"
        );
        for sidecar in ["sidecar-frontend", "metrics"] {
            assert!(
                !ids.contains(&hash_container_name("worker-0", sidecar)),
                "{sidecar} must not become a phantom worker under pod discovery"
            );
        }
    }

    /// The same pod under container discovery: `main` collapses into the
    /// pod-level identity, the other ready containers each add one id.
    #[test]
    fn pod_worker_ids_includes_containers_under_container_discovery() {
        let ids: HashSet<u64> = pod_worker_ids(&pod_mode_worker_pod(), true).collect();

        assert!(ids.contains(&hash_pod_name("worker-0")));
        assert_eq!(
            hash_container_name("worker-0", "main"),
            hash_pod_name("worker-0"),
            "the main container collapses to the pod identity and must not double-count"
        );
        for sidecar in ["sidecar-frontend", "metrics"] {
            assert!(ids.contains(&hash_container_name("worker-0", sidecar)));
        }
        assert_eq!(ids.len(), 3);
    }

    /// Under container discovery a worker registers under its *container*
    /// identity, so the bare pod identity is not a worker. It is only covered
    /// because `"main"` hashes to it -- when main is not Ready, nothing may
    /// contribute it, or the scheduler gains a zero-load phantom for a pod
    /// whose main container is down.
    #[test]
    fn pod_worker_ids_omits_the_pod_identity_when_main_is_not_ready() {
        let mut pod = pod_mode_worker_pod();
        for cs in pod
            .status
            .as_mut()
            .and_then(|s| s.container_statuses.as_mut())
            .expect("fixture has container statuses")
        {
            if cs.name == "main" {
                cs.ready = false;
            }
        }

        let ids: HashSet<u64> = pod_worker_ids(&pod, true).collect();

        assert!(
            !ids.contains(&hash_pod_name("worker-0")),
            "the pod identity must not be advertised when no Ready container claims it"
        );
        assert_eq!(ids.len(), 2);
    }

    /// End of the chain that made this matter: the index feeds
    /// `subset_to_worker_ids` -> `allowed_worker_ids` -> `register_workers`,
    /// so one backend pod must contribute exactly one worker id under pod
    /// discovery rather than one per ready container.
    #[test]
    fn worker_index_registers_one_id_per_pod_under_pod_discovery() {
        let mut index = WorkerEndpointIndex::default();
        index.upsert(&pod_mode_worker_pod());

        assert_eq!(
            index.endpoints.len(),
            1,
            "one backend pod must not register as several scheduler workers"
        );
        assert_eq!(
            index.endpoints.get(&hash_pod_name("worker-0")),
            Some(&"10.0.0.1:8000".to_string())
        );
    }

    /// `clear()` drops reflector state; the discovery mode is startup config
    /// and must survive, or a stream restart would silently change behaviour.
    #[test]
    fn worker_index_clear_preserves_the_discovery_mode() {
        let mut index = WorkerEndpointIndex::new(true);
        index.upsert(&pod_mode_worker_pod());
        index.clear();
        index.upsert(&pod_mode_worker_pod());

        assert_eq!(
            index.endpoints.len(),
            3,
            "container discovery must still be in effect after a clear"
        );
    }

    /// A pod carrying ready containers but no `metadata.name`. Only the API
    /// server's own guarantees keep this out of the reflector store, so the
    /// index defends against it rather than relying on that.
    fn unnamed_pod(ip: &str) -> Pod {
        let mut pod = pod_mode_worker_pod();
        pod.metadata.name = None;
        pod.status.as_mut().expect("status").pod_ip = Some(ip.to_string());
        pod
    }

    /// `hash_container_name("", …)` is a constant per container name, so an
    /// unnamed pod must contribute no ids in either mode — otherwise every
    /// unnamed pod claims the same worker ids.
    #[test]
    fn pod_worker_ids_are_empty_for_an_unnamed_pod() {
        for container_discovery in [false, true] {
            let ids: Vec<u64> =
                pod_worker_ids(&unnamed_pod("10.0.0.1"), container_discovery).collect();
            assert!(
                ids.is_empty(),
                "unnamed pod yielded {ids:?} with container_discovery={container_discovery}"
            );
        }
    }

    /// Two unnamed pods used to alias: both keyed under `by_pod[""]`, so the
    /// second upsert retracted the first's entries and reinstalled the same ids
    /// against its own endpoint, silently making the first unreachable.
    #[test]
    fn worker_index_unnamed_pods_do_not_alias_each_other() {
        let mut index = WorkerEndpointIndex::new(true);
        index.upsert(&unnamed_pod("10.0.0.1"));
        index.upsert(&unnamed_pod("10.0.0.2"));

        assert!(
            index.endpoints.is_empty(),
            "unnamed pods must not be indexed: {:?}",
            index.endpoints
        );
        assert!(
            index.by_pod.is_empty(),
            "an unnamed pod must not occupy the \"\" key"
        );
    }

    /// The guard must not cost a named pod its container ids.
    #[test]
    fn worker_index_named_pod_still_indexed_alongside_unnamed_ones() {
        let mut index = WorkerEndpointIndex::new(true);
        index.upsert(&unnamed_pod("10.0.0.1"));
        index.upsert(&pod_mode_worker_pod());
        index.upsert(&unnamed_pod("10.0.0.2"));

        assert_eq!(
            index.endpoints.get(&hash_pod_name("worker-0")),
            Some(&"10.0.0.1:8000".to_string()),
            "an unnamed pod's upsert must not retract a named pod's entries"
        );
        assert_eq!(index.endpoints.len(), 3);
    }

    /// Drive [`run_pod_reflector`] over a canned event stream and return the
    /// readiness flag and index it leaves behind once the stream ends.
    async fn drive_reflector(
        events: Vec<Result<kube::runtime::watcher::Event<Pod>, kube::runtime::watcher::Error>>,
    ) -> (Arc<AtomicBool>, Arc<RwLock<WorkerEndpointIndex>>) {
        let writer = kube::runtime::reflector::store::Writer::<Pod>::default();
        let store = writer.as_reader();
        let index = Arc::new(RwLock::new(WorkerEndpointIndex::new(false)));
        // Starts ready, as it would after the initial LIST sync succeeded.
        let ready = Arc::new(AtomicBool::new(true));

        run_pod_reflector(
            futures::stream::iter(events),
            store,
            index.clone(),
            ready.clone(),
        )
        .await;

        (ready, index)
    }

    /// A terminated reflector must stop advertising readiness, not just drop
    /// its endpoints. `runner.rs::serve` mirrors this flag onto the gRPC health
    /// status, so leaving it set strands the pod 1/1 Ready and SERVING with an
    /// empty index and no reflector — every request fails and Kubernetes never
    /// restarts it.
    #[tokio::test]
    async fn reflector_stream_end_drops_readiness_and_endpoints() {
        let (ready, index) = drive_reflector(vec![Ok(kube::runtime::watcher::Event::Apply(
            pod_mode_worker_pod(),
        ))])
        .await;

        assert!(
            !ready.load(Ordering::Acquire),
            "readiness must drop when the reflector stream ends"
        );
        assert!(
            index.read().unwrap().endpoints.is_empty(),
            "a terminated reflector must not keep answering lookups"
        );
    }

    /// A watch error is transient — the reflector retries — so it must not be
    /// mistaken for stream end and must leave readiness alone.
    #[tokio::test]
    async fn reflector_watch_error_alone_keeps_readiness() {
        let index = Arc::new(RwLock::new(WorkerEndpointIndex::new(false)));
        let ready = Arc::new(AtomicBool::new(true));
        let writer = kube::runtime::reflector::store::Writer::<Pod>::default();
        let store = writer.as_reader();

        // Never-ending stream: one error, then pending forever, so the loop
        // stays in its retry path rather than reaching the stream-end arm.
        use futures::StreamExt as _;
        let events =
            futures::stream::iter(vec![Err(kube::runtime::watcher::Error::NoResourceVersion)])
                .chain(futures::stream::pending());

        let ready_probe = ready.clone();
        let task = tokio::spawn(run_pod_reflector(events, store, index, ready));
        tokio::task::yield_now().await;

        assert!(
            ready_probe.load(Ordering::Acquire),
            "a retryable watch error must not clear readiness"
        );
        task.abort();
    }

    /// A writer panic poisons the lock. The request path must still answer
    /// from the (stale) index rather than panicking inside the tonic handler
    /// on every subsequent `pick()`.
    #[test]
    fn read_index_recovers_from_a_poisoned_lock() {
        let index = Arc::new(RwLock::new(WorkerEndpointIndex::new(false)));
        write_index(&index).upsert(&pod_mode_worker_pod());

        let poisoner = index.clone();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = write_index(&poisoner);
            panic!("writer panicked while holding the index");
        }));
        assert!(panicked.is_err());
        assert!(
            index.is_poisoned(),
            "the panic must actually poison the lock"
        );

        assert_eq!(
            read_index(&index).endpoints.get(&hash_pod_name("worker-0")),
            Some(&"10.0.0.1:8000".to_string()),
            "reads must survive a poisoned lock; the index is rebuildable"
        );
        // Writers too, or one panic freezes the index forever.
        write_index(&index).clear();
        assert!(read_index(&index).endpoints.is_empty());
    }

    /// A panic unwinding out of the reflector loop kills the task, so the
    /// stream-end path never runs. Readiness must still drop, or the pod stays
    /// Ready with no reflector behind it.
    #[tokio::test]
    async fn reflector_panic_still_drops_readiness() {
        let ready = Arc::new(AtomicBool::new(true));
        let index = Arc::new(RwLock::new(WorkerEndpointIndex::new(false)));
        let writer = kube::runtime::reflector::store::Writer::<Pod>::default();
        let store = writer.as_reader();

        let ready_probe = ready.clone();
        let task = tokio::spawn(async move {
            let _guard = ReflectorReadinessGuard(ready);
            panic!("reflector task panicked");
        });
        assert!(task.await.is_err(), "the task must have panicked");
        assert!(
            !ready_probe.load(Ordering::Acquire),
            "a panicking reflector must stop advertising readiness"
        );

        // Keep the canned-stream plumbing honest: the guard is armed inside
        // run_pod_reflector itself, not only in this synthetic task.
        let ready2 = Arc::new(AtomicBool::new(true));
        run_pod_reflector(
            futures::stream::iter(Vec::<
                Result<kube::runtime::watcher::Event<Pod>, kube::runtime::watcher::Error>,
            >::new()),
            store,
            index,
            ready2.clone(),
        )
        .await;
        assert!(!ready2.load(Ordering::Acquire));
    }

    /// An externally supplied endpoint must resolve to the identity the
    /// reflector actually holds. Under container discovery that is the engine
    /// container's id, never `hash_pod_name` -- deriving the latter names a
    /// worker no registry contains, which `register_workers` then upserts at
    /// zero load as the scheduler's most attractive candidate.
    #[test]
    fn external_endpoint_resolves_to_the_indexed_container_identity() {
        let mut index = WorkerEndpointIndex::new(true);
        index.upsert(&failover_pod(&[("engine-0", true)], true));

        let endpoint = Endpoint {
            pod_name: "worker-0".to_string(),
            address: "10.0.0.1".to_string(),
            port: "8000".to_string(),
            labels: HashMap::new(),
        };
        let addr = indexed_endpoint_address(&endpoint).expect("endpoint parses");

        let resolved: Vec<u64> = index
            .endpoints
            .iter()
            .filter(|(_, indexed)| **indexed == addr)
            .map(|(id, _)| *id)
            .collect();

        assert_eq!(resolved, vec![hash_container_name("worker-0", "engine-0")]);
        assert!(
            !resolved.contains(&hash_pod_name("worker-0")),
            "the pod identity is not a registered worker under container discovery"
        );
    }

    /// `Endpoint::address_port` does not bracket IPv6, while the index stores
    /// `SocketAddr`-rendered addresses. Comparing the raw forms matches on
    /// IPv4 and silently never matches on IPv6, so the normalization has to
    /// agree with what the index stores.
    #[test]
    fn indexed_endpoint_address_brackets_ipv6_to_match_the_index() {
        let endpoint = Endpoint {
            pod_name: "worker-0".to_string(),
            address: "fd00::2".to_string(),
            port: "8000".to_string(),
            labels: HashMap::new(),
        };

        assert_eq!(
            indexed_endpoint_address(&endpoint).as_deref(),
            Some("[fd00::2]:8000")
        );
        assert_ne!(
            endpoint.address_port(),
            "[fd00::2]:8000",
            "guards the reason this helper exists: the raw form is unbracketed"
        );

        let mut index = WorkerEndpointIndex::default();
        index.upsert(&simple_pod("worker-0", "fd00::2"));
        assert_eq!(
            index.endpoints.get(&hash_pod_name("worker-0")).cloned(),
            indexed_endpoint_address(&endpoint),
            "normalized endpoint must equal what the index stored"
        );
    }

    /// A minimal pod exposing an `http`-named port, for
    /// [`WorkerEndpointIndex`] tests that don't need failover's
    /// multi-container shape.
    fn simple_pod(name: &str, ip: &str) -> Pod {
        use k8s_openapi::api::core::v1::{Container, ContainerPort, PodSpec, PodStatus};
        use kube::api::ObjectMeta;

        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                ..Default::default()
            },
            spec: Some(PodSpec {
                containers: vec![Container {
                    name: "main".to_string(),
                    ports: Some(vec![ContainerPort {
                        name: Some(DYNAMO_CONTAINER_PORT_NAME.to_string()),
                        container_port: 8000,
                        ..Default::default()
                    }]),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                ..Default::default()
            }),
        }
    }

    /// Upserting a pod maps every id `pod_worker_ids` reports for it to its
    /// resolved endpoint -- container identities under container discovery, and
    /// not the pod identity, which no container-mode worker registers under.
    #[test]
    fn worker_index_upsert_maps_container_ids_to_the_pod_endpoint() {
        let mut index = WorkerEndpointIndex::new(true);
        let pod = failover_pod(&[("engine-0", true), ("engine-1", false)], true);
        index.upsert(&pod);

        assert!(
            !index.endpoints.contains_key(&hash_pod_name("worker-0")),
            "the pod identity is not a worker under container discovery"
        );
        assert_eq!(
            index
                .endpoints
                .get(&hash_container_name("worker-0", "engine-0")),
            Some(&"10.0.0.1:8000".to_string())
        );
        assert!(
            !index
                .endpoints
                .contains_key(&hash_container_name("worker-0", "engine-1")),
            "the not-ready standby engine must not be indexed"
        );
    }

    /// Re-upserting a pod whose ready containers changed must drop ids no
    /// longer live — a stale index entry would resolve a worker_id to an
    /// endpoint for a container that stopped being ready.
    #[test]
    fn worker_index_upsert_retracts_ids_that_are_no_longer_ready() {
        let mut index = WorkerEndpointIndex::new(true);
        index.upsert(&failover_pod(
            &[("engine-0", true), ("engine-1", true)],
            true,
        ));
        assert!(
            index
                .endpoints
                .contains_key(&hash_container_name("worker-0", "engine-1"))
        );

        // engine-1 is demoted between two Apply events for the same pod.
        index.upsert(&failover_pod(
            &[("engine-0", true), ("engine-1", false)],
            true,
        ));
        assert!(
            !index
                .endpoints
                .contains_key(&hash_container_name("worker-0", "engine-1")),
            "demoting engine-1 must retract its stale index entry"
        );
        assert!(
            index
                .endpoints
                .contains_key(&hash_container_name("worker-0", "engine-0")),
            "the still-Ready engine keeps its entry"
        );
    }

    /// Deleting a pod drops every id it had registered, and only those ids —
    /// an unrelated pod's entries must survive.
    #[test]
    fn worker_index_remove_drops_only_the_deleted_pods_ids() {
        let mut index = WorkerEndpointIndex::default();
        index.upsert(&simple_pod("worker-a", "10.0.0.1"));
        index.upsert(&simple_pod("worker-b", "10.0.0.2"));
        assert_eq!(index.endpoints.len(), 2);

        index.remove(&simple_pod("worker-a", "10.0.0.1"));

        assert!(!index.endpoints.contains_key(&hash_pod_name("worker-a")));
        assert_eq!(
            index.endpoints.get(&hash_pod_name("worker-b")),
            Some(&"10.0.0.2:8000".to_string())
        );
    }

    /// A pod without a resolvable HTTP endpoint contributes no entries,
    /// matching the pre-index `resolve_worker_endpoint`'s behavior of
    /// returning `None` for a worker_id whose pod lacks one.
    #[test]
    fn worker_index_upsert_skips_pods_without_a_resolvable_endpoint() {
        let mut index = WorkerEndpointIndex::default();
        // No sidecar-frontend, so no port is named `http`.
        index.upsert(&failover_pod(&[("engine-0", true)], false));
        assert!(index.endpoints.is_empty());
        assert!(index.by_pod.is_empty());
    }

    /// The pod's HTTP inference endpoint is resolved independently of which
    /// engine container is currently active: it stays pinned to the
    /// sidecar-frontend's `http`-named port, which failover's container
    /// cloning never touches.
    #[test]
    fn pod_endpoint_address_resolves_sidecar_regardless_of_engine_containers() {
        let pod = failover_pod(&[("engine-0", true), ("engine-1", false)], true);
        assert_eq!(
            pod_endpoint_address(&pod),
            Some("10.0.0.1:8000".to_string())
        );
    }

    /// Without a sidecar-frontend, the engine containers only expose
    /// internal system ports (no port named `http`), so resolution correctly
    /// returns `None` rather than guessing a port — this is a pre-existing,
    /// orthogonal limitation of aggregated (no-sidecar) failover workers, not
    /// something container-mode discovery introduces.
    #[test]
    fn pod_endpoint_address_none_without_an_http_named_port() {
        let pod = failover_pod(&[("engine-0", true), ("engine-1", false)], false);
        assert_eq!(pod_endpoint_address(&pod), None);
    }

    /// `pod_ip` is a bare (unbracketed) address, so an IPv6 pod's endpoint
    /// must be formatted via `SocketAddr`, not `format!("{ip}:{port}")` —
    /// the latter produces an ambiguous, unparseable `fd00::2:8000`.
    #[test]
    fn pod_endpoint_address_brackets_ipv6() {
        let pod = simple_pod("worker-0", "fd00::2");
        assert_eq!(
            pod_endpoint_address(&pod),
            Some("[fd00::2]:8000".to_string())
        );
    }

    /// A malformed `pod_ip` must not silently produce a bogus endpoint
    /// string — it must be rejected up front, same as a missing IP.
    #[test]
    fn pod_endpoint_address_none_for_unparseable_ip() {
        let pod = simple_pod("worker-0", "not-an-ip");
        assert_eq!(pod_endpoint_address(&pod), None);
    }

    /// The worker index's subset matcher (`endpoint_in_subset`, shared with
    /// `epp_router`) must match a bracketed IPv6 endpoint against a bare-IP
    /// candidate via `IpAddr`, not by splitting on `:` — `[fd00::2]:8000`
    /// split on `:` never equals the bare candidate `fd00::2`.
    #[test]
    fn subset_to_worker_ids_matches_bracketed_ipv6_against_bare_candidate() {
        let mut index = WorkerEndpointIndex::default();
        index.upsert(&simple_pod("worker-0", "fd00::2"));
        let endpoint = index
            .endpoints
            .get(&hash_pod_name("worker-0"))
            .cloned()
            .expect("worker-0 indexed");
        assert_eq!(endpoint, "[fd00::2]:8000");

        let candidates: HashSet<&str> = ["fd00::2"].into_iter().collect();
        let candidate_ips: HashSet<IpAddr> = ["fd00::2".parse().unwrap()].into_iter().collect();
        assert!(
            endpoint_in_subset(&endpoint, &candidates, &candidate_ips),
            "bracketed IPv6 endpoint must match its bare-IP candidate"
        );
    }
}

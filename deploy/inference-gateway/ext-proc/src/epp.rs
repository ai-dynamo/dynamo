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
use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};
use dynamo_llm::discovery::{ModelManager, WORKER_TYPE_DECODE};
use dynamo_llm::kv_router::prefill_router::PrefillQueryOutcome;
use dynamo_llm::kv_router::{KvRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, hash_pod_name};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime};

use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo};

const BOOKKEEPING_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// `None` in external (vanilla-vLLM) mode: the worker tokenizes and the EPP
    /// routes load-aware without replicating the worker's tokenizer config.
    preprocessor: Option<Arc<OpenAIPreprocessor>>,
    /// Shared HTTP client (bounded timeout) for the tokenizer sidecar, so each
    /// pick reuses connections and a stalled sidecar can't pile up stuck
    /// routing tasks.
    http_client: reqwest::Client,
    runtime: Runtime,
    /// Port to route to on each pod, from the InferencePool's `targetPorts`
    /// (or `DYN_EPP_TARGET_PORT`). `None` falls back to the container port named
    /// `http` (the Dynamo worker convention).
    target_port: Option<i32>,
    /// When set (external "precise" mode), tokenize each query by POSTing the
    /// request to this URL — a co-located tokenizer **sidecar** over loopback
    /// (`DYN_EPP_TOKENIZE_URL`). This keeps tokenization a single local hop
    /// rather than a second round-trip to a worker, and yields exact token IDs
    /// for token-level KV-prefix routing. `None` = load-aware (no tokenizer).
    tokenize_url: Option<String>,
    pod_store: kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    pod_store_ready: Arc<AtomicBool>,
}

impl Router {
    /// Initialize the router from discovery.
    ///
    /// This waits for at least one decode worker to appear, fetches the model
    /// card, initializes the preprocessor, and creates both routers.
    pub async fn from_discovery(
        namespace: &str,
        component: &str,
        enforce_disagg: bool,
    ) -> Result<Self> {
        let external_mode = std::env::var("DYN_EPP_EXTERNAL")
            .ok()
            .map(|v| {
                matches!(
                    v.trim().to_lowercase().as_str(),
                    "true" | "1" | "yes" | "on"
                )
            })
            .unwrap_or(false);

        let runtime = Runtime::from_settings()?;
        let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Bootstrap model identity + tokenizer. In the default (dynamo-worker)
        // mode this comes from the worker-published ModelDeploymentCard via
        // discovery, which guarantees the EPP tokenizer matches the worker. In
        // external mode (vanilla `vllm serve`) no Dynamo worker registers a
        // card, so model name + block size come from env and the EPP runs
        // WITHOUT a tokenizer: the worker tokenizes and routing is load-aware,
        // sidestepping the tokenizer-consistency problem.
        let (block_size, model_name, enable_eagle, actual_namespace, preprocessor): (
            u32,
            String,
            bool,
            String,
            Option<Arc<OpenAIPreprocessor>>,
        ) = if external_mode {
            let block_size = std::env::var("DYN_KV_CACHE_BLOCK_SIZE")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .unwrap_or(16);
            let model_name = std::env::var("DYN_MODEL_NAME").unwrap_or_else(|_| "vllm".to_string());
            tracing::info!(
                block_size,
                model_name = %model_name,
                "External mode: skipping Dynamo discovery / model card; worker tokenizes, routing is load-aware"
            );
            (block_size, model_name, false, namespace.to_string(), None)
        } else {
            // Wait for workers
            wait_for_discovery_sync(&drt).await;

            let bootstrap = init_preprocessor(&drt, namespace).await?;
            let block_size = bootstrap.card.kv_cache_block_size;
            let model_name = bootstrap.card.display_name.clone();
            let enable_eagle = bootstrap.card.runtime_config.enable_eagle;
            (
                block_size,
                model_name,
                enable_eagle,
                bootstrap.actual_namespace,
                Some(bootstrap.preprocessor),
            )
        };

        let mut kv_router_config = kv_router_config_from_env();
        kv_router_config.skip_initial_worker_wait = true;

        let component_handle = drt.namespace(&actual_namespace)?.component(component)?;
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

        // Wait for runtime config watch to populate. Skipped in external mode:
        // vanilla vLLM pods never register a Dynamo ModelRuntimeConfig, so this
        // would otherwise block forever.
        if !external_mode {
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
            enforce_disagg,
            model_name.clone(),
            actual_namespace.clone(),
            enable_eagle,
        );

        spawn_prefill_discovery_watcher(drt.clone(), actual_namespace.clone(), prefill_tx);

        // Endpoint discovery. External (GIE) mode scans the referenced
        // InferencePool for its selector + targetPort; dynamo-worker mode uses
        // the worker label convention. (Pods are labeled with the BASE
        // namespace `nvidia.com/dynamo-namespace`, not the rolling-update
        // -suffixed registration namespace, so the worker selector uses it.)
        let (selector, target_port) = resolve_pod_discovery(namespace).await?;
        let (pod_store, pod_store_ready) = spawn_pod_reflector(selector).await?;

        // Precise KV-aware routing: when enabled, subscribe directly to each
        // worker pod's native vLLM KV-cache event stream (ZMQ) and feed it into
        // the decode router's indexer, keyed by hash_pod_name(pod). This adds
        // ground-truth prefix overlap (what each worker has actually cached) on
        // top of predict-on-route bookkeeping, giving precise prefix-cache
        // routing. Requires the vLLM pods to run with `--kv-events-config` (PUB
        // bound on DYN_EPP_KV_EVENT_PORT).
        if external_mode
            && std::env::var("DYN_EPP_KV_EVENTS")
                .map(|v| v.trim().eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        {
            let kv_port: i32 = std::env::var("DYN_EPP_KV_EVENT_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5557);
            let kv_topic = std::env::var("DYN_EPP_KV_EVENT_TOPIC").unwrap_or_default();
            tracing::info!(
                kv_port,
                kv_topic = %kv_topic,
                "Precise KV-event consumption enabled: spawning per-pod ZMQ listeners"
            );
            spawn_kv_event_reconciler(decode_router.clone(), pod_store.clone(), kv_port, kv_topic);
        }

        // `model_manager` and `drt` are intentionally not stored on the
        // Router. The KV chooser, prefill router, prefill discovery watcher,
        // and pod reflector all clone whatever they need from these
        // constructor-locals before this scope ends, so dropping them here
        // does not tear down any background work.
        // Precise external mode tokenizes each query via a sidecar endpoint
        // (DYN_EPP_TOKENIZE_URL, default a loopback tokenizer sidecar) so
        // token-level KV-prefix matching works without an in-EPP tokenizer.
        let tokenize_url = if external_mode
            && std::env::var("DYN_EPP_PREFIX_MODE")
                .map(|m| m.trim().eq_ignore_ascii_case("precise"))
                .unwrap_or(false)
        {
            let url = std::env::var("DYN_EPP_TOKENIZE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8788/tokenize".to_string());
            tracing::info!(%url, "Precise prefix mode: tokenizing via sidecar endpoint");
            Some(url)
        } else {
            None
        };

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("failed to build tokenizer HTTP client");

        Ok(Self {
            prefill_router,
            decode_router,
            preprocessor,
            http_client,
            runtime,
            target_port,
            tokenize_url,
            pod_store,
            pod_store_ready,
        })
    }

    /// Tokenize a JSON request body and extract the router-relevant
    /// `priority_jump` from `nvext.agent_hints.priority`.
    ///
    /// Returns `(token_ids, priority_jump)`. `priority_jump` is `0.0` when no
    /// hint is present. Mirrors the standalone Dynamo preprocessor lift in
    /// `lib/llm/src/preprocessor.rs` so this gateway path produces the same
    /// queue ordering as a non-GAIE deployment.
    pub fn tokenize(&self, request_json: &str) -> Result<(Vec<u32>, f64)> {
        // External (no-tokenizer) mode. Two strategies:
        //   * "precise" (DYN_EPP_PREFIX_MODE=precise): exact query tokens come
        //     from a tokenizer sidecar (DYN_EPP_TOKENIZE_URL) — handled
        //     asynchronously in `pick`, so this sync path is not reached.
        //   * "load" (default): no tokenizer; return placeholder tokens sized
        //     to the request (the scheduler asserts isl > 0). Pair with
        //     DYN_OVERLAP_SCORE_WEIGHT=0 for pure load-aware routing.
        let Some(preprocessor) = self.preprocessor.as_ref() else {
            const AVG_CHARS_PER_TOKEN: usize = 4;
            // No tokenizer, but still honor nvext.agent_hints.priority so a
            // vanilla-vLLM request keeps its scheduler priority.
            let priority_jump = serde_json::from_str::<serde_json::Value>(request_json)
                .map(|v| extract_priority_jump_from_json(&v))
                .unwrap_or(0.0);
            return Ok((
                vec![0u32; (request_json.len() / AVG_CHARS_PER_TOKEN).max(1)],
                priority_jump,
            ));
        };

        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(request_json)?;

        let priority_jump = extract_priority_jump(&request);

        let formatted_prompt = preprocessor.apply_template(&request)?.unwrap_or_default();

        let encoding = preprocessor.tokenize(&formatted_prompt)?;
        Ok((encoding.token_ids().to_vec(), priority_jump))
    }

    /// Resolve a worker_id to a pod endpoint address (ip:port).
    /// Lock-free read from the in-memory reflector store — no K8s API calls.
    /// Port is read from the pod's Dynamo HTTP container port.
    pub fn resolve_worker_endpoint(&self, worker_id: u64) -> Option<String> {
        for pod in self.pod_store.state() {
            let Some(pod_name) = pod.metadata.name.as_deref() else {
                continue;
            };
            if !pod_is_ready(&pod) {
                continue;
            }
            if hash_pod_name(pod_name) == worker_id {
                return pod_endpoint_address(&pod, self.target_port);
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
            .filter(|pod| pod_is_ready(pod))
            .find_map(|pod| pod_endpoint_address(pod, self.target_port))
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
                && let Some(addr) = pod_endpoint_address(&pod, self.target_port)
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
            let Some(addr_port) = pod_endpoint_address(&pod, self.target_port) else {
                continue;
            };
            let ip = addr_port.split(':').next().unwrap_or("");
            if candidates.contains(addr_port.as_str()) || candidates.contains(ip) {
                ids.insert(hash_pod_name(pod_name));
            }
        }
        ids
    }

    /// All ready reflected pods as worker IDs.
    ///
    /// Used when the ext_proc caller provides no endpoints *and* no Envoy
    /// subset hint (e.g. the shared agentgateway data plane, which does not
    /// emit `candidate_subset`). Without this fallback `pick()` would leave the
    /// scheduler with an empty candidate set and every request would 503. This
    /// makes the InferencePool reflector the authoritative worker source for
    /// vanilla workers that never register with Dynamo discovery.
    fn all_ready_worker_ids(&self) -> HashSet<u64> {
        let mut ids = HashSet::new();
        for pod in self.pod_store.state() {
            let Some(pod_name) = pod.metadata.name.as_deref() else {
                continue;
            };
            if !pod_is_ready(&pod) {
                continue;
            }
            if pod_endpoint_address(&pod, self.target_port).is_some() {
                ids.insert(hash_pod_name(pod_name));
            }
        }
        ids
    }

    /// Route a prefill request. Returns (worker_id, dp_rank).
    ///
    /// `priority_jump` is forwarded to the prefill scheduler queue so that
    /// requests carrying `nvext.agent_hints.priority` jump ahead of normal
    /// arrivals when the router queue is active.
    pub async fn route_prefill(
        &self,
        tokens: &[u32],
        priority_jump: f64,
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> Result<(u64, Option<u32>)> {
        if let Some(ref ids) = allowed_worker_ids {
            self.prefill_router.register_workers(ids);
        }

        let outcome = self
            .prefill_router
            .query_prefill_worker(
                tokens,
                None,
                false,
                None,
                priority_jump,
                allowed_worker_ids,
                RoutingConstraints::default(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Prefill query failed: {:?}", e))?;

        match outcome {
            PrefillQueryOutcome::Routed { worker_id, dp_rank } => Ok((worker_id, dp_rank)),
            // Surface backpressure as an error so the caller's
            // enforce_disagg / aggregated-fallback logic in `pick()` can
            // decide whether to fail the request or fall back to decode-only.
            PrefillQueryOutcome::Backpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            } => Err(anyhow::anyhow!(
                "Prefill router backpressure: {:?} (queued_isl_tokens={}, max={:?})",
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens
            )),
        }
    }

    /// Route a decode request. Returns (WorkerWithDpRank, overlap_blocks).
    ///
    /// `priority_jump` is forwarded to the decode scheduler queue so that
    /// requests carrying `nvext.agent_hints.priority` jump ahead of normal
    /// arrivals when the router queue is active.
    pub async fn route_decode(
        &self,
        tokens: &[u32],
        is_disaggregated: bool,
        priority_jump: f64,
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> Result<(WorkerWithDpRank, u32)> {
        if let Some(ref ids) = allowed_worker_ids {
            self.decode_router.register_workers(ids);
        }

        let config_override = if is_disaggregated {
            Some(RouterConfigOverride {
                overlap_score_credit: Some(0.0),
                assume_kv_reuse: Some(false),
                track_prefill_tokens: Some(false),
                ..Default::default()
            })
        } else {
            None
        };

        self.decode_router
            .find_best_match(
                None,
                tokens,
                None,
                config_override.as_ref(),
                false,
                None,
                priority_jump,
                None,
                allowed_worker_ids,
                RoutingConstraints::default(),
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
    ) -> Result<()> {
        let decode_router = self.decode_router.clone();
        let request_id = request_id.to_owned();
        let tokens = tokens.to_vec();
        // Predict-on-route (approximate) crediting of the chosen worker. Can be
        // turned off (DYN_EPP_PREDICT_ON_ROUTE=false) to isolate ground-truth
        // KV-event overlap — proving precise routing works without the
        // approximate signal.
        let predict_on_route = std::env::var("DYN_EPP_PREDICT_ON_ROUTE")
            .map(|v| !v.trim().eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let precise = self.tokenize_url.is_some() && predict_on_route;

        tokio::time::timeout(BOOKKEEPING_TIMEOUT, async {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);
            // Precise mode (real tokens from the sidecar): predict-on-route
            // affinity — credit the chosen worker for the prompt it just
            // received so subsequent same-prefix requests see overlap and stick
            // to it. Load-only mode (placeholder tokens) suppresses predicted
            // caching, since those tokens carry no real prefix information.
            let router_config_override = if precise {
                RouterConfigOverride {
                    assume_kv_reuse: Some(true),
                    track_prefill_tokens: Some(true),
                    ..Default::default()
                }
            } else {
                RouterConfigOverride {
                    overlap_score_credit: Some(0.0),
                    assume_kv_reuse: Some(false),
                    track_prefill_tokens: Some(false),
                    ..Default::default()
                }
            };

            let overlap_blocks = decode_router
                .get_overlap_blocks(&tokens, None, worker, None)
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
                    Some(&router_config_override),
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

/// Extract the router queue `priority_jump` from a chat completion request's
/// `nvext.agent_hints.priority`.
///
/// Negative priorities are clamped to `0.0` so a low-priority hint never
/// pushes a request behind FCFS arrivals (matches the standalone preprocessor
/// in `lib/llm/src/preprocessor.rs`). Falls back to the deprecated
/// `latency_sensitivity` alias for callers still on the old field name.
/// Returns `0.0` when `nvext` is absent.
/// Extract `priority_jump` directly from a raw request body `Value`, for the
/// external (no-`OpenAIPreprocessor`) paths that never deserialize into
/// `NvCreateChatCompletionRequest`. Mirrors [`extract_priority_jump`].
fn extract_priority_jump_from_json(body: &serde_json::Value) -> f64 {
    body.get("nvext")
        .and_then(|n| n.get("agent_hints"))
        .and_then(|h| {
            h.get("priority")
                .and_then(|p| p.as_i64())
                .map(|p| p.max(0) as f64)
                .or_else(|| h.get("latency_sensitivity").and_then(|l| l.as_f64()))
        })
        .unwrap_or(0.0)
}

fn extract_priority_jump(
    request: &dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest,
) -> f64 {
    request
        .nvext
        .as_ref()
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| {
            h.priority
                .map(|p| p.max(0) as f64)
                .or(h.latency_sensitivity)
        })
        .unwrap_or(0.0)
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
fn pod_endpoint_address(
    pod: &k8s_openapi::api::core::v1::Pod,
    target_port: Option<i32>,
) -> Option<String> {
    let ip = pod.status.as_ref()?.pod_ip.as_ref()?;

    // A numeric target port (from the InferencePool's `targetPorts`, or
    // `DYN_EPP_TARGET_PORT`) routes straight to `ip:port` — stock `vllm serve`
    // pods typically expose a single, unnamed OpenAI port (e.g. 8000). With no
    // numeric port we fall back to a named container port
    // (`DYN_EPP_TARGET_PORT_NAME`, default `http` — the Dynamo worker
    // convention), so existing dynamo-worker deployments are unaffected.
    if let Some(n) = target_port {
        return Some(format!("{ip}:{n}"));
    }
    let port_name = std::env::var("DYN_EPP_TARGET_PORT_NAME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DYNAMO_CONTAINER_PORT_NAME.to_string());
    let port = pod
        .spec
        .as_ref()?
        .containers
        .iter()
        .filter_map(|c| c.ports.as_ref())
        .flatten()
        .find(|p| p.name.as_deref() == Some(port_name.as_str()))
        .map(|p| p.container_port)?;
    Some(format!("{ip}:{port}"))
}

/// True only if the pod is actively serving: has a `Ready` condition `True` and
/// is not Terminating. Routing and discovery must skip pods that are starting
/// up or shutting down — otherwise the EPP hands the gateway a dead endpoint
/// (e.g. a pod stuck Terminating still in the reflector store).
/// Whether to emit GAIE primary+fallback endpoints (comma-joined
/// `x-gateway-destination-endpoint`). Off by default: the multi-endpoint value
/// is correct per the GAIE endpoint-picker protocol and lets a GAIE-compliant
/// proxy (Envoy/kgateway) retry the next ready pod on worker loss, but some
/// data planes (e.g. agentgateway) reject it. Opt in with
/// `DYN_EPP_EMIT_FALLBACKS=1`.
fn emit_fallbacks() -> bool {
    static EMIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EMIT.get_or_init(|| {
        std::env::var("DYN_EPP_EMIT_FALLBACKS")
            .ok()
            .map(|v| matches!(v.as_str(), "true" | "1" | "yes" | "on"))
            .unwrap_or(false)
    })
}

fn pod_is_ready(pod: &k8s_openapi::api::core::v1::Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return false;
    }
    pod.status
        .as_ref()
        .and_then(|s| s.conditions.as_ref())
        .map(|cs| cs.iter().any(|c| c.type_ == "Ready" && c.status == "True"))
        .unwrap_or(false)
}

/// Tokenize a request by POSTing it to a vLLM-style `/tokenize` endpoint (the
/// "precise" prefix strategy). `url` points at a co-located tokenizer
/// **sidecar** over loopback (`DYN_EPP_TOKENIZE_URL`), so this is one local hop,
/// not a second round-trip to a worker. The returned exact token IDs drive
/// token-level KV-prefix matching in the router (vs. load-aware placeholders).
///
/// The sidecar applies the model's chat template + tokenizer, matching what the
/// worker caches; the request forwards `model` plus `messages` or `prompt`. The
/// response is expected to contain a `tokens` array of integer IDs (vLLM's
/// `/tokenize` shape: `{"count": N, "tokens": [...]}`).
async fn remote_tokenize(
    client: &reqwest::Client,
    url: &str,
    request_json: &str,
) -> Result<Vec<u32>> {
    let v: serde_json::Value = serde_json::from_str(request_json)?;
    if v["messages"].is_null() && v["prompt"].is_null() {
        return Err(anyhow::anyhow!(
            "tokenize request has neither 'messages' nor 'prompt'"
        ));
    }

    // Forward the original body so template-affecting fields (tools,
    // tool_choice, chat_template_args, …) reach the sidecar and the returned
    // tokens match what the worker actually caches.
    let resp = client
        .post(url)
        .json(&v)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("tokenize request to {url} failed: {e}"))?;
    let status = resp.status();
    let val: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("tokenize response from {url} parse failed: {e}"))?;
    if !status.is_success() {
        return Err(anyhow::anyhow!(
            "tokenize endpoint {url} returned {status}: {val}"
        ));
    }

    let tokens: Vec<u32> = val["tokens"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("tokenize response missing 'tokens' array: {val}"))?
        .iter()
        .filter_map(|t| t.as_u64().map(|n| n as u32))
        .collect();
    if tokens.is_empty() {
        return Err(anyhow::anyhow!(
            "tokenize endpoint {url} returned no tokens"
        ));
    }
    Ok(tokens)
}

/// Resolve the pod label selector + target port for endpoint discovery.
///
/// External (GIE) mode reads them from the referenced `InferencePool`
/// (`DYN_EPP_INFERENCE_POOL`): the pool's `spec.selector` and
/// `spec.targetPorts`. Falls back to an explicit `DYN_EPP_POD_SELECTOR` /
/// `DYN_EPP_TARGET_PORT`, or the Dynamo worker convention.
async fn resolve_pod_discovery(dynamo_namespace: &str) -> Result<(String, Option<i32>)> {
    if let Ok(pool_name) = std::env::var("DYN_EPP_INFERENCE_POOL") {
        let pool_name = pool_name.trim();
        if !pool_name.is_empty() {
            let ns = std::env::var("POD_NAMESPACE").map_err(|_| {
                anyhow::anyhow!("POD_NAMESPACE not set (required to read the InferencePool)")
            })?;
            return discover_from_inference_pool(&ns, pool_name).await;
        }
    }
    let selector = std::env::var("DYN_EPP_POD_SELECTOR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            format!(
                "nvidia.com/dynamo-namespace={dynamo_namespace},nvidia.com/dynamo-component-class=worker"
            )
        });
    let target_port = std::env::var("DYN_EPP_TARGET_PORT")
        .ok()
        .and_then(|p| p.trim().parse::<i32>().ok());
    Ok((selector, target_port))
}

/// Read endpoint discovery from a GIE `InferencePool`: its `spec.selector`
/// becomes the pod label selector and `spec.targetPorts[0].number` the port to
/// route to. This is the standard GAIE mechanism — the pool is the source of
/// truth for pool membership, rather than a hand-passed label.
async fn discover_from_inference_pool(
    namespace: &str,
    pool_name: &str,
) -> Result<(String, Option<i32>)> {
    use kube::core::{ApiResource, DynamicObject, GroupVersionKind};
    use kube::{Api, Client};

    let client = Client::try_default().await?;
    let gvk = GroupVersionKind::gvk("inference.networking.k8s.io", "v1", "InferencePool");
    let ar = ApiResource::from_gvk(&gvk);
    let api: Api<DynamicObject> = Api::namespaced_with(client, namespace, &ar);
    let pool = api.get(pool_name).await.map_err(|e| {
        anyhow::anyhow!("failed to read InferencePool {namespace}/{pool_name}: {e}")
    })?;

    let match_labels = pool.data["spec"]["selector"]["matchLabels"]
        .as_object()
        .ok_or_else(|| {
            anyhow::anyhow!("InferencePool {pool_name} has no spec.selector.matchLabels")
        })?;
    let selector = match_labels
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|val| format!("{k}={val}")))
        .collect::<Vec<_>>()
        .join(",");
    let target_port = pool.data["spec"]["targetPorts"]
        .as_array()
        .and_then(|ports| ports.first())
        .and_then(|p| p["number"].as_i64())
        .map(|n| n as i32);

    tracing::info!(
        pool = pool_name,
        namespace,
        selector = %selector,
        ?target_port,
        "Endpoint discovery resolved from InferencePool"
    );
    Ok((selector, target_port))
}

/// Start a background pod reflector that watches the worker pods matching the
/// InferencePool selector. The returned `Store` provides lock-free reads of the
/// current pod state — no K8s API calls on the hot path.
async fn spawn_pod_reflector(
    selector: String,
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

/// Background reconciler that keeps exactly one native vLLM KV-event ZMQ
/// listener alive per worker pod. Each pod's events feed the decode router's
/// indexer stamped with `hash_pod_name(pod)` — the same worker_id `pick()` uses
/// — so the scheduler scores precise prefix overlap against each worker's real
/// KV cache. Reconciles every 5s: registers listeners for new Ready pods and
/// cancels them when pods disappear.
fn spawn_kv_event_reconciler(
    decode_router: Arc<KvRouter>,
    pod_store: kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
    kv_event_port: i32,
    kv_event_topic: String,
) {
    tokio::spawn(async move {
        let mut registered: std::collections::HashMap<u64, tokio_util::sync::CancellationToken> =
            std::collections::HashMap::new();
        loop {
            let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
            for pod in pod_store.state() {
                let Some(name) = pod.metadata.name.as_deref() else {
                    continue;
                };
                let Some(ip) = pod.status.as_ref().and_then(|s| s.pod_ip.as_deref()) else {
                    continue;
                };
                if !pod_is_ready(&pod) {
                    continue;
                }
                let worker_id = hash_pod_name(name);
                seen.insert(worker_id);
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    registered.entry(worker_id)
                {
                    let endpoint = format!("tcp://{ip}:{kv_event_port}");
                    let token = decode_router.register_worker_kv_events(
                        worker_id,
                        endpoint.clone(),
                        kv_event_topic.clone(),
                    );
                    slot.insert(token);
                    tracing::info!(%endpoint, worker_id, pod = %name, "Registered worker KV-event listener");
                }
            }
            registered.retain(|worker_id, token| {
                if seen.contains(worker_id) {
                    true
                } else {
                    tracing::info!(
                        worker_id = *worker_id,
                        "Worker pod gone; cancelling KV-event listener"
                    );
                    token.cancel();
                    false
                }
            });
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });
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

fn kv_router_config_from_env() -> KvRouterConfig {
    let mut cfg = KvRouterConfig::default();

    fn env_f64(key: &str) -> Option<f64> {
        std::env::var(key).ok().and_then(|v| v.parse().ok())
    }
    fn env_bool(key: &str) -> Option<bool> {
        std::env::var(key)
            .ok()
            .and_then(|v| match v.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                _ => None,
            })
    }

    if let Some(v) = env_f64("DYN_OVERLAP_SCORE_WEIGHT") {
        cfg.overlap_score_credit = v;
    }
    if let Some(v) = env_f64("DYN_ROUTER_TEMPERATURE") {
        cfg.router_temperature = v;
    }
    if let Some(v) = env_bool("DYN_USE_KV_EVENTS") {
        cfg.use_kv_events = v;
    }
    if let Some(v) = env_bool("DYN_ROUTER_REPLICA_SYNC") {
        cfg.router_replica_sync = v;
    }
    if let Some(v) = env_bool("DYN_ROUTER_TRACK_ACTIVE_BLOCKS") {
        cfg.router_track_active_blocks = v;
    }
    if let Some(v) = env_bool("DYN_ROUTER_TRACK_OUTPUT_BLOCKS") {
        cfg.router_track_output_blocks = v;
    }
    if let Some(v) = env_bool("DYN_ROUTER_TRACK_PREFILL_TOKENS") {
        cfg.router_track_prefill_tokens = v;
    }
    if let Some(v) = env_f64("DYN_ROUTER_QUEUE_THRESHOLD") {
        cfg.router_queue_threshold = Some(v);
    }

    tracing::info!(
        overlap_score_weight = cfg.overlap_score_credit,
        router_temperature = cfg.router_temperature,
        use_kv_events = cfg.use_kv_events,
        "KvRouterConfig initialized"
    );

    cfg
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
                // No Envoy subset hint (e.g. the shared agentgateway data
                // plane): fall back to all ready pods from the InferencePool
                // reflector so the scheduler has a candidate set. Registering
                // these worker_ids downstream is what makes vanilla vLLM pods
                // (which never join Dynamo discovery) schedulable.
                let ids = self.all_ready_worker_ids();
                if ids.is_empty() {
                    return Err(PickError::NoEndpoints);
                }
                (Some(ids), Vec::new())
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

        // Precise external mode: tokenize via the sidecar (one local hop, not a
        // second round-trip to a worker). Otherwise tokenize locally (the
        // dynamo-worker preprocessor) or use the load-aware placeholder.
        let (tokens, priority_jump) = if let Some(url) = self.tokenize_url.as_deref() {
            let toks = remote_tokenize(&self.http_client, url, body_str)
                .await
                .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;
            // The sidecar returns only tokens; recover the priority hint here so
            // precise-mode requests keep their scheduler priority.
            let priority_jump = serde_json::from_str::<serde_json::Value>(body_str)
                .map(|v| extract_priority_jump_from_json(&v))
                .unwrap_or(0.0);
            (toks, priority_jump)
        } else {
            self.tokenize(body_str)
                .map_err(|e| PickError::TokenizationFailed(e.to_string()))?
        };

        // Try prefill routing first (disaggregated mode).
        //
        // If the prefill router is not activated (no prefill workers
        // discovered yet, or the inner router has been deactivated), this
        // returns an error. Behavior on that error depends on
        // `DYN_ENFORCE_DISAGG`:
        //
        // * `enforce_disagg = false` (default): fall back to aggregated
        //   (decode-only) routing — matches `PrefillRouter::generate`.
        // * `enforce_disagg = true`: surface the error to Envoy and let the
        //   request fail. Silently downgrading to aggregated would defeat
        //   the operator's explicit "strict disagg" policy.
        let prefill_result = self
            .route_prefill(&tokens, priority_jump, allowed_worker_ids.clone())
            .await;

        let is_disaggregated = match &prefill_result {
            Ok(_) => true,
            Err(e) => {
                if self.prefill_router.enforce_disagg() {
                    tracing::warn!(
                        error = %e,
                        request_id = %req.request_id,
                        "Prefill routing failed under DYN_ENFORCE_DISAGG=true; failing request"
                    );
                    return Err(PickError::RoutingFailed(format!(
                        "prefill routing failed under enforce_disagg: {e}"
                    )));
                }
                tracing::debug!(
                    error = %e,
                    "Prefill routing failed; falling back to aggregated mode"
                );
                false
            }
        };

        let (decode_worker, _overlap) = self
            .route_decode(&tokens, is_disaggregated, priority_jump, allowed_worker_ids)
            .await
            .map_err(|e| PickError::RoutingFailed(e.to_string()))?;

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
        // x-worker-instance-id, x-dp-rank, x-prefill-instance-id,
        // x-prefill-dp-rank, x-dynamo-routing-mode
        let mut headers = vec![
            (
                "x-worker-instance-id".to_string(),
                format!("{}", decode_worker.worker_id),
            ),
            ("x-dp-rank".to_string(), decode_worker.dp_rank.to_string()),
        ];

        if let Ok((prefill_worker_id, prefill_dp_rank)) = &prefill_result {
            headers.push((
                "x-dynamo-routing-mode".to_string(),
                "disaggregated".to_string(),
            ));
            headers.push((
                "x-prefill-instance-id".to_string(),
                format!("{}", prefill_worker_id),
            ));
            if let Some(rank) = prefill_dp_rank {
                headers.push(("x-prefill-dp-rank".to_string(), rank.to_string()));
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

        // Fallback endpoints for the data plane to retry on (GAIE endpoint-
        // picker protocol: a primary plus comma-separated fallbacks,
        // "ip:port,ip:port,..."). On worker loss a GAIE-compliant proxy
        // (Envoy/kgateway) retries the next ready pod instead of failing the
        // client; the primary stays the KV/load-aware pick. Gated by
        // DYN_EPP_EMIT_FALLBACKS (default off): the multi-endpoint value is
        // protocol-correct but some data planes (e.g. agentgateway) reject it.
        let fallbacks: Vec<String> = if emit_fallbacks() {
            self.all_ready_worker_ids()
                .into_iter()
                .filter(|wid| *wid != decode_worker.worker_id)
                .filter_map(|wid| self.resolve_worker_endpoint(wid))
                .filter(|ep| *ep != endpoint)
                .collect()
        } else {
            Vec::new()
        };

        Ok(PickResult {
            endpoint,
            fallbacks,
            headers,
            // External mode (vanilla vLLM) tokenizes itself; don't inject
            // nvext.token_data. Tokens above are used only for routing/booking.
            token_ids: if self.preprocessor.is_some() {
                Some(tokens)
            } else {
                None
            },
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
        assert_eq!(extract_priority_jump(&with_priority), 5.0);

        let without_nvext: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(
                r#"{
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}]
                }"#,
            )
            .unwrap();
        assert_eq!(extract_priority_jump(&without_nvext), 0.0);
    }
}

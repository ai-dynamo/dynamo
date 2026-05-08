// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wraps Dynamo's KV-aware router for use from the ext_proc server.
//!
//! This is the native-Rust equivalent of the CGO bridge in
//! `lib/bindings/c/src/lib.rs`. Instead of crossing a C FFI boundary, the
//! ext_proc server calls these types directly as async Rust.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_llm::discovery::{ModelManager, WORKER_TYPE_DECODE};
use dynamo_llm::kv_router::{KvRouter, PrefillRouter};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, hash_pod_name};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime};

use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo};

const BOOKKEEPING_TIMEOUT: Duration = Duration::from_secs(5);

/// Holds all router state needed for request routing.
///
/// This is the async-native equivalent of `RouterHandles` from the C bindings,
/// without the `block_on` / unsafe FFI overhead.
pub struct Router {
    prefill_router: Arc<PrefillRouter>,
    decode_router: Arc<KvRouter>,
    #[allow(dead_code)]
    model_manager: Arc<ModelManager>,
    preprocessor: Arc<OpenAIPreprocessor>,
    runtime: Runtime,
    #[allow(dead_code)]
    drt: DistributedRuntime,
    pod_store: kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>,
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
        let runtime = Runtime::from_settings()?;
        let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Wait for workers
        wait_for_discovery_sync(&drt).await;

        let bootstrap = init_preprocessor(&drt, namespace).await?;
        let block_size = bootstrap.card.kv_cache_block_size;
        let model_name = bootstrap.card.display_name.clone();
        let enable_eagle = bootstrap.card.runtime_config.enable_eagle;
        let actual_namespace = &bootstrap.actual_namespace;

        let mut kv_router_config = kv_router_config_from_env();
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
            enforce_disagg,
            model_name.clone(),
            actual_namespace.to_string(),
            enable_eagle,
        );

        spawn_prefill_discovery_watcher(drt.clone(), actual_namespace.to_string(), prefill_tx);

        // Use the BASE namespace (without rolling-update suffix) for the pod
        // selector. Workers register in discovery under the suffixed namespace
        // (e.g. "atchernych-qwen-9f792849"), but the K8s pod label
        // `nvidia.com/dynamo-namespace` is always set to the base
        // ("atchernych-qwen") by the operator. Using the suffixed name here
        // would silently match zero pods during/after a DGD rolling update.
        let pod_store = spawn_pod_reflector(namespace).await?;

        Ok(Self {
            prefill_router,
            decode_router,
            model_manager,
            preprocessor: bootstrap.preprocessor,
            runtime,
            drt,
            pod_store,
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
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(request_json)?;

        let priority_jump = extract_priority_jump(&request);

        let formatted_prompt = self
            .preprocessor
            .apply_template(&request)?
            .unwrap_or_default();

        let encoding = self.preprocessor.tokenize(&formatted_prompt)?;
        Ok((encoding.token_ids().to_vec(), priority_jump))
    }

    /// Resolve a worker_id to a pod endpoint address (ip:port).
    /// Lock-free read from the in-memory reflector store — no K8s API calls.
    /// Port is read from the pod's first container port spec.
    pub fn resolve_worker_endpoint(&self, worker_id: u64) -> Option<String> {
        for pod in self.pod_store.state() {
            let pod_name = pod.metadata.name.as_deref()?;
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

        self.prefill_router
            .query_prefill_worker(tokens, None, false, None, priority_jump, allowed_worker_ids)
            .await
            .map_err(|e| anyhow::anyhow!("Prefill query failed: {:?}", e))
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
                overlap_score_weight: Some(0.0),
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

        tokio::time::timeout(BOOKKEEPING_TIMEOUT, async {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);
            let router_config_override = RouterConfigOverride {
                overlap_score_weight: Some(0.0),
                assume_kv_reuse: Some(false),
                track_prefill_tokens: Some(false),
                ..Default::default()
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
}

/// Extract the router queue `priority_jump` from a chat completion request's
/// `nvext.agent_hints.priority`.
///
/// Negative priorities are clamped to `0.0` so a low-priority hint never
/// pushes a request behind FCFS arrivals (matches the standalone preprocessor
/// in `lib/llm/src/preprocessor.rs`). Falls back to the deprecated
/// `latency_sensitivity` alias for callers still on the old field name.
/// Returns `0.0` when `nvext` is absent.
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

    card.download_config().await?;
    let preprocessor = OpenAIPreprocessor::new(card.clone())?;

    Ok(DiscoveredModelBootstrap {
        preprocessor, // already Arc<OpenAIPreprocessor>
        card,
        actual_namespace,
    })
}

/// Extract "ip:port" from a pod by reading its IP from status and its serving
/// port from the container spec. Returns `None` if the pod has no IP or no
/// container port defined — we never silently route to a guessed port.
fn pod_endpoint_address(pod: &k8s_openapi::api::core::v1::Pod) -> Option<String> {
    let ip = pod.status.as_ref()?.pod_ip.as_ref()?;
    let port = pod
        .spec
        .as_ref()
        .and_then(|spec| spec.containers.first())
        .and_then(|c| c.ports.as_ref())
        .and_then(|ports| ports.first())
        .map(|p| p.container_port)?;
    Some(format!("{ip}:{port}"))
}

/// Start a background pod reflector that watches worker pods matching the
/// InferencePool selector. The returned `Store` provides lock-free reads
/// of the current pod state — no K8s API calls on the hot path.
async fn spawn_pod_reflector(
    dynamo_namespace: &str,
) -> Result<kube::runtime::reflector::Store<k8s_openapi::api::core::v1::Pod>> {
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
        "nvidia.com/dynamo-namespace={},nvidia.com/dynamo-component-type=worker",
        dynamo_namespace
    );

    let writer = reflector::store::Writer::default();
    let store = writer.as_reader();
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
    let _ = tokio::time::timeout(Duration::from_secs(30), store_for_wait.wait_until_ready()).await;

    Ok(store)
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
        cfg.overlap_score_weight = v;
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
        overlap_score_weight = cfg.overlap_score_weight,
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
        // When the endpoint list is populated (e.g. from a K8s datastore),
        // use it to constrain which workers the router may select. When empty,
        // fall back to the router's own discovery-based worker set.
        let (allowed_worker_ids, worker_map) = if endpoints.is_empty() {
            (None, Vec::new())
        } else {
            let subset_filtered = apply_subset_filter(endpoints, &req.candidate_subset);
            let effective = if subset_filtered.is_empty() && !req.candidate_subset.is_empty() {
                tracing::warn!(
                    subset = ?req.candidate_subset,
                    total_endpoints = endpoints.len(),
                    "No endpoints match subset hint, falling back to full list"
                );
                endpoints.iter().collect::<Vec<_>>()
            } else {
                subset_filtered
            };

            if req.body.is_empty() {
                return Ok(PickResult {
                    endpoint: effective[0].address_port(),
                    ..Default::default()
                });
            }

            let wm: Vec<(u64, &Endpoint)> = effective
                .iter()
                .map(|ep| (hash_pod_name(&ep.pod_name), *ep))
                .collect();
            let ids: HashSet<u64> = wm.iter().map(|(id, _)| *id).collect();
            (Some(ids), wm)
        };

        if req.body.is_empty() {
            // No body (GET request) and no external endpoint list —
            // resolve any worker via discovery and forward to it.
            let endpoint = self
                .resolve_any_worker_endpoint()
                .ok_or(PickError::NoEndpoints)?;
            return Ok(PickResult {
                endpoint,
                ..Default::default()
            });
        }

        let body_str = std::str::from_utf8(&req.body)
            .map_err(|e| PickError::TokenizationFailed(format!("Invalid UTF-8: {e}")))?;

        let (tokens, priority_jump) = self
            .tokenize(body_str)
            .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;

        // Try prefill routing first (disaggregated mode).
        // If the prefill router is not activated, this returns an error
        // and we fall back to aggregated (decode-only) routing.
        let prefill_result = self
            .route_prefill(&tokens, priority_jump, allowed_worker_ids.clone())
            .await;

        let is_disaggregated = prefill_result.is_ok();

        let (decode_worker, _overlap) = self
            .route_decode(&tokens, is_disaggregated, priority_jump, allowed_worker_ids)
            .await
            .map_err(|e| PickError::RoutingFailed(e.to_string()))?;

        let endpoint = if worker_map.is_empty() {
            self.resolve_worker_endpoint(decode_worker.worker_id)
                .unwrap_or_else(|| format!("worker-{}", decode_worker.worker_id))
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

        Ok(PickResult {
            endpoint,
            fallbacks: vec![],
            headers,
            token_ids: Some(tokens),
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

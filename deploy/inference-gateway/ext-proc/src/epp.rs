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

/// Result of routing a request: which worker(s) to send it to.
pub struct RoutingResult {
    pub is_disaggregated: bool,
    pub prefill_worker_id: Option<u64>,
    pub prefill_dp_rank: Option<u32>,
    pub decode_worker_id: u64,
    pub decode_dp_rank: u32,
    pub token_ids: Vec<u32>,
}

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
                    anyhow::anyhow!(
                        "Runtime config watch closed before any workers appeared"
                    )
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

        Ok(Self {
            prefill_router,
            decode_router,
            model_manager,
            preprocessor: bootstrap.preprocessor,
            runtime,
        })
    }

    /// Tokenize a JSON request body and return token IDs.
    pub fn tokenize(&self, request_json: &str) -> Result<Vec<u32>> {
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(request_json)?;

        let formatted_prompt = self
            .preprocessor
            .apply_template(&request)?
            .unwrap_or_default();

        let encoding = self.preprocessor.tokenize(&formatted_prompt)?;
        Ok(encoding.token_ids().to_vec())
    }

    /// Route a prefill request. Returns (worker_id, dp_rank).
    pub async fn route_prefill(
        &self,
        tokens: &[u32],
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> Result<(u64, Option<u32>)> {
        if let Some(ref ids) = allowed_worker_ids {
            self.prefill_router.register_workers(ids);
        }

        self.prefill_router
            .query_prefill_worker(tokens, None, false, None, 0.0, allowed_worker_ids)
            .await
            .map_err(|e| anyhow::anyhow!("Prefill query failed: {:?}", e))
    }

    /// Route a decode request. Returns (WorkerWithDpRank, overlap_blocks).
    pub async fn route_decode(
        &self,
        tokens: &[u32],
        is_disaggregated: bool,
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
                0.0,
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

            let cached_tokens =
                overlap_blocks as usize * decode_router.block_size() as usize;

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

// ---------------------------------------------------------------------------
// Discovery helpers (ported from lib/bindings/c/src/lib.rs)
// ---------------------------------------------------------------------------

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
            "No model found in namespace '{}' via discovery",
            target_namespace
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
                                tracing::debug!(
                                    "PrefillRouter activation channel already closed"
                                );
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

/// Build a worker_id → Endpoint mapping and an allowed-ID set from the
/// endpoint list provided by the ext_proc server. Uses `hash_pod_name` to
/// convert pod names to Dynamo worker IDs.
fn build_worker_map(endpoints: &[Endpoint]) -> (HashSet<u64>, Vec<(u64, &Endpoint)>) {
    let mut ids = HashSet::new();
    let mut mapping = Vec::new();

    for ep in endpoints {
        let worker_id = hash_pod_name(&ep.pod_name);
        ids.insert(worker_id);
        mapping.push((worker_id, ep));
    }

    (ids, mapping)
}

#[tonic::async_trait]
impl EndpointPicker for Router {
    async fn pick(
        &self,
        req: &RequestInfo,
        endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        if endpoints.is_empty() {
            return Err(PickError::NoEndpoints);
        }

        // Build allowed worker ID set from the provided endpoints
        let (allowed_worker_ids, worker_to_endpoint) = build_worker_map(endpoints);

        // If no body, fall back to first endpoint (GET-style request)
        if req.body.is_empty() {
            return Ok(PickResult {
                endpoint: endpoints[0].address_port(),
                fallbacks: vec![],
            });
        }

        let body_str = std::str::from_utf8(&req.body)
            .map_err(|e| PickError::TokenizationFailed(format!("Invalid UTF-8: {e}")))?;

        let tokens = self
            .tokenize(body_str)
            .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;

        let filter = if allowed_worker_ids.is_empty() {
            None
        } else {
            Some(allowed_worker_ids)
        };

        let (decode_worker, _overlap) = self
            .route_decode(&tokens, false, filter)
            .await
            .map_err(|e| PickError::RoutingFailed(e.to_string()))?;

        // Map selected worker_id back to an endpoint address
        let endpoint = worker_to_endpoint
            .iter()
            .find(|(wid, _)| *wid == decode_worker.worker_id)
            .map(|(_, ep)| ep.address_port())
            .unwrap_or_else(|| {
                tracing::warn!(
                    worker_id = decode_worker.worker_id,
                    "Selected worker not in provided endpoints, using first"
                );
                endpoints[0].address_port()
            });

        tracing::info!(
            worker_id = decode_worker.worker_id,
            dp_rank = decode_worker.dp_rank,
            endpoint = %endpoint,
            token_count = tokens.len(),
            model = %req.model,
            "Picked endpoint"
        );

        Ok(PickResult {
            endpoint,
            fallbacks: vec![],
        })
    }
}

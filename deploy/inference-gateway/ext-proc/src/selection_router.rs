// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free raw-vLLM aggregate router backed by the selection service core.

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_kv_router::services::selection::SelectAndReserveRequest;
use dynamo_kv_router::services::selection::SelectionCore;
use dynamo_kv_router::services::selection::SelectionError;
use dynamo_kv_router::services::selection::WorkerRequest;
use dynamo_llm::local_model::LocalModel;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_runtime::discovery::hash_pod_name;
use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::Api;
use kube::Client;
use kube::runtime::reflector;
use kube::runtime::watcher;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::picker::Endpoint;
use crate::picker::EndpointPicker;
use crate::picker::PickError;
use crate::picker::PickResult;
use crate::picker::RequestInfo;

const DEFAULT_TENANT_ID: &str = "default";
const DEFAULT_TARGET_PORT: u16 = 8000;
const DEFAULT_KV_EVENT_PORT: u16 = 5557;
const DEFAULT_RECONCILE_INTERVAL_MS: u64 = 1000;

#[derive(Clone)]
pub struct SelectionRouterConfig {
    pub k8s_namespace: String,
    pub pod_selector: String,
    pub model_name: String,
    pub model_path: String,
    pub tenant_id: String,
    pub target_port: u16,
    pub kv_event_port: Option<u16>,
    pub block_size: u32,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub is_eagle: bool,
    pub indexer_threads: usize,
    pub reconcile_interval: Duration,
    pub kv_router_config: KvRouterConfig,
}

impl SelectionRouterConfig {
    pub fn from_env() -> Result<Self> {
        let k8s_namespace = required_env("POD_NAMESPACE")?;
        let pod_selector = required_env("DYN_EPP_POD_SELECTOR")?;
        let model_name = required_env("DYN_MODEL_NAME")?;
        let model_path = optional_env("DYN_MODEL_PATH").unwrap_or_else(|| model_name.clone());
        let tenant_id = DEFAULT_TENANT_ID.to_string();

        let block_size = parse_required_env("DYN_KV_CACHE_BLOCK_SIZE")?;
        let target_port = parse_optional_env("DYN_EPP_TARGET_PORT")?.unwrap_or(DEFAULT_TARGET_PORT);
        let max_num_batched_tokens = parse_optional_env("DYN_EPP_MAX_NUM_BATCHED_TOKENS")?;
        let total_kv_blocks = parse_optional_env("DYN_EPP_TOTAL_KV_BLOCKS")?;
        let is_eagle = parse_optional_bool_env("DYN_EPP_IS_EAGLE")?.unwrap_or(false);
        let reconcile_interval_ms = parse_optional_env("DYN_EPP_RECONCILE_INTERVAL_MS")?
            .unwrap_or(DEFAULT_RECONCILE_INTERVAL_MS);

        let mut kv_router_config = kv_router_config_from_dynamo_env();
        apply_kv_events_alias(&mut kv_router_config)?;
        kv_router_config.skip_initial_worker_wait = true;
        kv_router_config.router_replica_sync = false;

        if max_num_batched_tokens.is_none() {
            if optional_env("DYN_ROUTER_QUEUE_THRESHOLD").is_some()
                || optional_env("DYN_ROUTER_POLICY_CONFIG").is_some()
            {
                anyhow::bail!(
                    "DYN_EPP_MAX_NUM_BATCHED_TOKENS is required when selection queueing or policy config is enabled"
                );
            }
            tracing::info!(
                "DYN_EPP_MAX_NUM_BATCHED_TOKENS is unset; disabling the selection queue for router-only mode"
            );
            kv_router_config.router_queue_threshold = None;
        }

        let indexer_threads = parse_optional_env("DYN_EPP_SELECTION_INDEXER_THREADS")?
            .unwrap_or(kv_router_config.router_event_threads.max(1) as usize);
        let kv_event_port = if kv_router_config.use_kv_events {
            Some(
                parse_optional_env("DYN_EPP_KV_EVENT_PORT")?
                    .or(parse_optional_env("DYN_VLLM_KV_EVENT_PORT")?)
                    .unwrap_or(DEFAULT_KV_EVENT_PORT),
            )
        } else {
            None
        };

        Ok(Self {
            k8s_namespace,
            pod_selector,
            model_name,
            model_path,
            tenant_id,
            target_port,
            kv_event_port,
            block_size,
            max_num_batched_tokens,
            total_kv_blocks,
            is_eagle,
            indexer_threads,
            reconcile_interval: Duration::from_millis(reconcile_interval_ms),
            custom_template_path,
            kv_router_config,
        })
    }
}

pub struct SelectionRouter {
    core: Arc<SelectionCore>,
    preprocessor: Arc<OpenAIPreprocessor>,
    config: Arc<SelectionRouterConfig>,
    cancel_token: CancellationToken,
    pod_store: reflector::Store<Pod>,
    pod_store_ready: Arc<AtomicBool>,
}

impl SelectionRouter {
    pub async fn new(config: SelectionRouterConfig) -> Result<Self> {
        let preprocessor = init_preprocessor(&config).await?;
        let cancel_token = CancellationToken::new();
        let core = Arc::new(SelectionCore::new(
            config.kv_router_config.clone(),
            config.indexer_threads,
            cancel_token.clone(),
        ));
        let config = Arc::new(config);
        let (pod_store, pod_store_ready) =
            spawn_worker_reflector(core.clone(), config.clone(), cancel_token.clone()).await?;

        Ok(Self {
            core,
            preprocessor,
            config,
            cancel_token,
            pod_store,
            pod_store_ready,
        })
    }

    pub fn pod_store_ready(&self) -> Arc<AtomicBool> {
        self.pod_store_ready.clone()
    }

    fn tokenize(&self, request_json: &str) -> Result<(Vec<u32>, f64, u32)> {
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(request_json)?;

        let priority_jump = extract_priority_jump(&request);
        let strict_priority = extract_strict_priority(&request);
        let formatted_prompt = self
            .preprocessor
            .apply_template(&request)?
            .unwrap_or_default();
        let encoding = self.preprocessor.tokenize(&formatted_prompt)?;
        Ok((
            encoding.token_ids().to_vec(),
            priority_jump,
            strict_priority,
        ))
    }

    fn resolve_any_ready_endpoint(&self) -> Option<String> {
        self.pod_store
            .state()
            .iter()
            .find_map(|pod| ready_pod_endpoint(pod, self.config.target_port))
    }

    fn resolve_any_ready_endpoint_in_subset(&self, allowed: &HashSet<WorkerId>) -> Option<String> {
        for pod in self.pod_store.state() {
            let Some(pod_name) = pod.metadata.name.as_deref() else {
                continue;
            };
            if allowed.contains(&hash_pod_name(pod_name))
                && let Some(endpoint) = ready_pod_endpoint(&pod, self.config.target_port)
            {
                return Some(endpoint);
            }
        }
        None
    }

    fn subset_to_worker_ids(&self, candidate_subset: &[String]) -> HashSet<WorkerId> {
        let candidates: HashSet<&str> = candidate_subset.iter().map(|s| s.as_str()).collect();
        let mut ids = HashSet::new();
        for pod in self.pod_store.state() {
            let Some(pod_name) = pod.metadata.name.as_deref() else {
                continue;
            };
            let Some(endpoint) = ready_pod_endpoint(&pod, self.config.target_port) else {
                continue;
            };
            let ip = endpoint.split(':').next().unwrap_or("");
            if candidates.contains(endpoint.as_str()) || candidates.contains(ip) {
                ids.insert(hash_pod_name(pod_name));
            }
        }
        ids
    }
}

impl Drop for SelectionRouter {
    fn drop(&mut self) {
        self.core.shutdown();
        self.cancel_token.cancel();
    }
}

#[tonic::async_trait]
impl EndpointPicker for SelectionRouter {
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

        let allowed_worker_ids = if endpoints.is_empty() {
            if req.candidate_subset.is_empty() {
                None
            } else {
                let ids = self.subset_to_worker_ids(&req.candidate_subset);
                if ids.is_empty() {
                    tracing::warn!(
                        subset = ?req.candidate_subset,
                        "No reflected raw-vLLM pod matches the subset hint"
                    );
                    return Err(PickError::NoEndpoints);
                }
                Some(ids)
            }
        } else {
            let subset_filtered = apply_subset_filter(endpoints, &req.candidate_subset);
            if subset_filtered.is_empty() && !req.candidate_subset.is_empty() {
                tracing::warn!(
                    subset = ?req.candidate_subset,
                    total_endpoints = endpoints.len(),
                    "No endpoints match the subset hint"
                );
                return Err(PickError::NoEndpoints);
            }
            let ids = subset_filtered
                .iter()
                .map(|ep| hash_pod_name(&ep.pod_name))
                .collect();
            Some(ids)
        };

        if req.body.is_empty() {
            let endpoint = match &allowed_worker_ids {
                Some(ids) => self.resolve_any_ready_endpoint_in_subset(ids),
                None => self.resolve_any_ready_endpoint(),
            }
            .ok_or(PickError::NoEndpoints)?;
            return Ok(PickResult {
                endpoint,
                ..Default::default()
            });
        }

        let body_str = std::str::from_utf8(&req.body)
            .map_err(|e| PickError::TokenizationFailed(format!("Invalid UTF-8: {e}")))?;
        let (tokens, priority_jump, strict_priority) = self
            .tokenize(body_str)
            .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;

        let selection_req = build_select_and_reserve_request(
            &self.config,
            &req.request_id,
            &tokens,
            priority_jump,
            strict_priority,
            allowed_worker_ids.as_ref(),
        )
        .map_err(|e| PickError::RoutingFailed(e.to_string()))?;

        let response = self
            .core
            .select_and_reserve(selection_req)
            .await
            .map_err(selection_error_to_pick_error)?;

        let headers = vec![
            (
                "x-dynamo-worker-instance-id".to_string(),
                response.worker_id.to_string(),
            ),
            ("x-dynamo-dp-rank".to_string(), response.dp_rank.to_string()),
            (
                "x-dynamo-routing-mode".to_string(),
                "aggregated".to_string(),
            ),
        ];

        tracing::info!(
            worker_id = response.worker_id,
            worker_id_hex = format!("{:x}", response.worker_id),
            dp_rank = response.dp_rank,
            endpoint = %response.endpoint,
            token_count = tokens.len(),
            priority_jump,
            model = %self.config.model_name,
            overlap_tokens = response.overlap.longest_matched,
            "Picked raw-vLLM aggregate endpoint with embedded selection core"
        );

        Ok(PickResult {
            endpoint: response.endpoint,
            fallbacks: vec![],
            headers,
            token_ids: Some(tokens),
        })
    }

    async fn on_prefill_complete(&self, request_id: &str) {
        if request_id.is_empty() {
            return;
        }
        if let Err(error) = self.core.prefill_complete(request_id).await {
            log_lifecycle_error("prefill_complete", request_id, error);
        }
    }

    async fn on_request_complete(&self, request_id: &str) {
        if request_id.is_empty() {
            return;
        }
        if let Err(error) = self.core.free_reservation(request_id).await {
            log_lifecycle_error("free_reservation", request_id, error);
        }
    }
}

#[derive(Serialize)]
struct SelectAndReservePayload<'a> {
    model_name: &'a str,
    tenant_id: &'a str,
    reservation_id: &'a str,
    token_ids: &'a [u32],
    #[serde(skip_serializing_if = "Option::is_none")]
    priority_jump: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict_priority: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
}

fn build_select_and_reserve_request(
    config: &SelectionRouterConfig,
    reservation_id: &str,
    token_ids: &[u32],
    priority_jump: f64,
    strict_priority: u32,
    allowed_worker_ids: Option<&HashSet<WorkerId>>,
) -> Result<SelectAndReserveRequest> {
    let payload = SelectAndReservePayload {
        model_name: &config.model_name,
        tenant_id: &config.tenant_id,
        reservation_id,
        token_ids,
        priority_jump: (priority_jump != 0.0).then_some(priority_jump),
        strict_priority: (strict_priority != 0).then_some(strict_priority),
        allowed_worker_ids,
    };

    Ok(serde_json::from_value(serde_json::to_value(payload)?)?)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkerFingerprint {
    endpoint: String,
    kv_events_endpoint: Option<String>,
    block_size: u32,
    max_num_batched_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
    is_eagle: bool,
}

async fn spawn_worker_reflector(
    core: Arc<SelectionCore>,
    config: Arc<SelectionRouterConfig>,
    cancel_token: CancellationToken,
) -> Result<(reflector::Store<Pod>, Arc<AtomicBool>)> {
    let client = Client::try_default().await?;
    let pods: Api<Pod> = Api::namespaced(client, &config.k8s_namespace);
    let writer = reflector::store::Writer::default();
    let store = writer.as_reader();
    let ready = Arc::new(AtomicBool::new(false));
    let watcher_config = watcher::Config::default().labels(&config.pod_selector);
    let reflect = reflector::reflector(writer, watcher(pods, watcher_config));

    tracing::info!(
        namespace = %config.k8s_namespace,
        selector = %config.pod_selector,
        model = %config.model_name,
        target_port = config.target_port,
        kv_event_port = ?config.kv_event_port,
        "Starting raw-vLLM pod reflector for embedded selection"
    );

    let store_for_reflector = store.clone();
    tokio::spawn(async move {
        tokio::pin!(reflect);
        while reflect.next().await.is_some() {}
        tracing::warn!("Raw-vLLM pod reflector stream ended unexpectedly");
        drop(store_for_reflector);
    });

    let store_for_reconcile = store.clone();
    let ready_for_reconcile = ready.clone();
    tokio::spawn(async move {
        run_worker_reconciler(
            core,
            config,
            store_for_reconcile,
            ready_for_reconcile,
            cancel_token,
        )
        .await;
    });

    Ok((store, ready))
}

async fn run_worker_reconciler(
    core: Arc<SelectionCore>,
    config: Arc<SelectionRouterConfig>,
    store: reflector::Store<Pod>,
    ready: Arc<AtomicBool>,
    cancel_token: CancellationToken,
) {
    let mut known = HashMap::<WorkerId, WorkerFingerprint>::new();

    tokio::select! {
        _ = cancel_token.cancelled() => return,
        result = store.wait_until_ready() => {
            if let Err(error) = result {
                tracing::error!(error = %error, "Raw-vLLM pod reflector failed before initial LIST completed");
                return;
            }
        }
    }

    loop {
        reconcile_selection_workers(&core, &config, &store, &mut known).await;
        ready.store(true, Ordering::Release);

        tokio::select! {
            _ = cancel_token.cancelled() => return,
            _ = tokio::time::sleep(config.reconcile_interval) => {}
        }
    }
}

async fn reconcile_selection_workers(
    core: &SelectionCore,
    config: &SelectionRouterConfig,
    store: &reflector::Store<Pod>,
    known: &mut HashMap<WorkerId, WorkerFingerprint>,
) {
    let mut desired = HashSet::new();

    for pod in store.state() {
        let Some((request, fingerprint)) = worker_request_for_pod(&pod, config) else {
            continue;
        };
        desired.insert(request.worker_id);

        if known.get(&request.worker_id) == Some(&fingerprint) {
            continue;
        }

        let worker_id = request.worker_id;
        match core.upsert_worker(request).await {
            Ok(record) => {
                tracing::info!(
                    worker_id,
                    lifecycle = ?record.lifecycle,
                    endpoint = ?record.endpoint,
                    not_schedulable_reasons = ?record.not_schedulable_reasons,
                    "Reconciled raw-vLLM worker into selection core"
                );
                known.insert(worker_id, fingerprint);
            }
            Err(error) => {
                tracing::warn!(
                    worker_id,
                    error = %error,
                    "Failed to upsert raw-vLLM worker into selection core"
                );
            }
        }
    }

    let stale: Vec<_> = known
        .keys()
        .copied()
        .filter(|worker_id| !desired.contains(worker_id))
        .collect();
    for worker_id in stale {
        match core.delete_worker(worker_id).await {
            Ok(_) => {
                tracing::info!(worker_id, "Removed raw-vLLM worker from selection core");
            }
            Err(error) if error.kind() == "not_found" => {
                tracing::debug!(
                    worker_id,
                    "Raw-vLLM worker was already absent from selection core"
                );
            }
            Err(error) => {
                tracing::warn!(
                    worker_id,
                    error = %error,
                    "Failed to remove raw-vLLM worker from selection core"
                );
                continue;
            }
        }
        known.remove(&worker_id);
    }
}

fn worker_request_for_pod(
    pod: &Pod,
    config: &SelectionRouterConfig,
) -> Option<(WorkerRequest, WorkerFingerprint)> {
    if !is_pod_ready(pod) {
        return None;
    }

    let pod_name = pod.metadata.name.as_deref()?;
    let pod_ip = pod.status.as_ref()?.pod_ip.as_deref()?;
    let worker_id = hash_pod_name(pod_name);
    let endpoint = format!("{pod_ip}:{}", config.target_port);
    let kv_events_endpoint = config
        .kv_event_port
        .map(|port| format!("tcp://{pod_ip}:{port}"));

    let fingerprint = WorkerFingerprint {
        endpoint: endpoint.clone(),
        kv_events_endpoint: kv_events_endpoint.clone(),
        block_size: config.block_size,
        max_num_batched_tokens: config.max_num_batched_tokens,
        total_kv_blocks: config.total_kv_blocks,
        is_eagle: config.is_eagle,
    };

    let request = WorkerRequest {
        worker_id,
        model_name: config.model_name.clone(),
        tenant_id: config.tenant_id.clone(),
        endpoint: Some(endpoint),
        kv_events_endpoint,
        kv_events_endpoints: HashMap::new(),
        replay_endpoint: None,
        block_size: Some(config.block_size),
        data_parallel_start_rank: Some(0),
        data_parallel_size: Some(1),
        max_num_batched_tokens: config.max_num_batched_tokens,
        total_kv_blocks: config.total_kv_blocks,
        stable_routing_id: Some(pod_name.to_string()),
        is_eagle: Some(config.is_eagle),
        taints: HashSet::new(),
        topology_domains: HashMap::new(),
        kv_transfer_domain: None,
        kv_transfer_enforcement: None,
        kv_transfer_preferred_weight: None,
    };

    Some((request, fingerprint))
}

fn ready_pod_endpoint(pod: &Pod, target_port: u16) -> Option<String> {
    if !is_pod_ready(pod) {
        return None;
    }
    let ip = pod.status.as_ref()?.pod_ip.as_deref()?;
    Some(format!("{ip}:{target_port}"))
}

fn is_pod_ready(pod: &Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return false;
    }

    let Some(status) = pod.status.as_ref() else {
        return false;
    };
    if status.pod_ip.as_deref().is_none_or(str::is_empty) {
        return false;
    }

    status.conditions.as_ref().is_some_and(|conditions| {
        conditions
            .iter()
            .any(|condition| condition.type_ == "Ready" && condition.status == "True")
    })
}

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

async fn init_preprocessor(config: &SelectionRouterConfig) -> Result<Arc<OpenAIPreprocessor>> {
    let model_source = Path::new(&config.model_path);
    let model_path = if model_source.exists() {
        model_source.to_path_buf()
    } else {
        LocalModel::fetch(&config.model_path, true).await?
    };

    let mut card = ModelDeploymentCard::load_from_disk(&model_path).with_context(|| format!("loading model card from {}", model_path.display()))?;
    card.set_name(&config.model_name);
    card.kv_cache_block_size = config.block_size;

    tracing::info!(
        model = %config.model_name,
        source = %config.model_path,
        local_path = %model_path.display(),
        block_size = config.block_size,
        "Initialized router-only OpenAI preprocessor"
    );

    OpenAIPreprocessor::new(card).context("creating OpenAI preprocessor")
}

fn selection_error_to_pick_error(error: SelectionError) -> PickError {
    match error.kind() {
        "not_ready" => PickError::NoEndpoints,
        _ => PickError::RoutingFailed(error.to_string()),
    }
}

fn log_lifecycle_error(operation: &'static str, request_id: &str, error: SelectionError) {
    if error.kind() == "not_found" {
        tracing::debug!(
            request_id,
            operation,
            error = %error,
            "Selection lifecycle request was not found"
        );
    } else {
        tracing::warn!(
            request_id,
            operation,
            error = %error,
            "Selection lifecycle update failed"
        );
    }
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

fn extract_strict_priority(
    request: &dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest,
) -> u32 {
    request
        .nvext
        .as_ref()
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| h.strict_priority)
        .unwrap_or(0)
}

fn required_env(key: &str) -> Result<String> {
    optional_env(key).ok_or_else(|| anyhow::anyhow!("{key} environment variable is required"))
}

fn optional_env(key: &str) -> Option<String> {
    std::env::var(key).ok().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn parse_required_env<T>(key: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: Display,
{
    let value = required_env(key)?;
    value
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid {key}={value:?}: {error}"))
}

fn parse_optional_env<T>(key: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: Display,
{
    let Some(value) = optional_env(key) else {
        return Ok(None);
    };
    value
        .parse()
        .map(Some)
        .map_err(|error| anyhow::anyhow!("invalid {key}={value:?}: {error}"))
}

fn parse_optional_bool_env(key: &str) -> Result<Option<bool>> {
    optional_env(key)
        .map(|value| parse_bool_value(key, &value))
        .transpose()
}

fn parse_bool_value(key: &str, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => anyhow::bail!("invalid {key}={value:?}: expected true/false"),
    }
}

fn apply_kv_events_alias(config: &mut KvRouterConfig) -> Result<()> {
    if optional_env("DYN_USE_KV_EVENTS").is_some() {
        return Ok(());
    }
    if let Some(value) = optional_env("DYN_ROUTER_USE_KV_EVENTS") {
        config.use_kv_events = parse_bool_value("DYN_ROUTER_USE_KV_EVENTS", &value)?;
    }
    Ok(())
}

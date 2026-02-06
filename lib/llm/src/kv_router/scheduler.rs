// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::RuntimeConfigs;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::KV_HIT_RATE_SUBJECT;
use super::KvRouterConfig;
use super::RouterConfigOverride;
use super::WorkerSelector;
use super::protocols::{DpRank, OverlapScores, WorkerId, WorkerSelectionResult, WorkerWithDpRank};
use super::sequence::{ActiveSequencesMultiWorker, SequenceError};

use dynamo_tokens::SequenceHash;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVHitRateEvent {
    pub worker_id: WorkerId,
    #[serde(default)]
    pub dp_rank: DpRank,
    pub isl_blocks: usize,
    pub overlap_blocks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLoad {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub potential_prefill_tokens: usize,
    pub potential_decode_blocks: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints available to route work")]
    NoEndpoints,

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,

    #[error("failed to initialize event publisher: {0}")]
    InitFailed(String),
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker: WorkerWithDpRank,
    pub overlap_blocks: u32,
}

pub struct SchedulingRequest {
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub overlaps: OverlapScores,
    pub decode_blocks: HashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: HashMap<WorkerWithDpRank, usize>,
    // Router config overrides for this specific request
    pub router_config_override: Option<RouterConfigOverride>,
    // Whether to update scheduler states (false for query_instance_id requests)
    pub update_states: bool,
    // LORA adapter name extracted from request.model field
    pub lora_name: Option<String>,
    // Option to take it out to send the response without moving the struct
    resp_tx: Option<tokio::sync::oneshot::Sender<SchedulingResponse>>,
}

impl SchedulingRequest {
    pub fn respond(&mut self, response: SchedulingResponse) {
        // Changed to &mut self
        if let Some(tx) = self.resp_tx.take() {
            // Use take() to extract the sender
            if tx.send(response).is_err() {
                tracing::error!("failed to send response to requestor");
            }
        } else {
            tracing::error!("respond called multiple times on same request");
        }
    }
}

pub struct KvScheduler {
    request_tx: tokio::sync::mpsc::Sender<SchedulingRequest>,
    slots: Arc<ActiveSequencesMultiWorker>,
}

impl KvScheduler {
    pub async fn start(
        component: Component,
        block_size: u32,
        workers_with_configs: Arc<RuntimeConfigs>,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        replica_sync: bool,
        router_id: u64,
        worker_type: &'static str,
    ) -> Result<Self, KvSchedulerError> {
        let selector = selector.unwrap_or(Box::new(DefaultWorkerSelector::new(None)));

        // Get initial workers from DashMap for slot initialization.
        // Caller must ensure at least one worker is present (via wait_for_some).
        let initial_workers: HashMap<WorkerId, Option<ModelRuntimeConfig>> = workers_with_configs
            .configs
            .iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect();

        let slots = Arc::new(
            ActiveSequencesMultiWorker::new(
                component.clone(),
                block_size as usize,
                initial_workers,
                replica_sync,
                router_id,
                worker_type,
            )
            .await
            .map_err(|e| KvSchedulerError::InitFailed(e.to_string()))?,
        );

        // Spawn background task to sync slots with DashMap when notified of changes.
        // ModelManager's watcher updates the DashMap and notifies; we wait on watch receiver here.
        let slots_monitor = slots.clone();
        let subscriber = workers_with_configs.subscribe();
        let configs_monitor = subscriber.configs;
        let mut change_rx = subscriber.change_rx;
        let monitor_cancel_token = component.drt().child_token();
        tokio::spawn(async move {
            tracing::trace!("KvScheduler workers monitoring task started");
            let mut last_workers: HashSet<WorkerId> = HashSet::new();

            loop {
                // Wait for notification or cancellation
                tokio::select! {
                    _ = monitor_cancel_token.cancelled() => {
                        tracing::trace!("KvScheduler workers monitoring task shutting down");
                        break;
                    }
                    result = change_rx.changed() => {
                        if result.is_err() {
                            tracing::warn!("KvScheduler: config watch sender dropped, shutting down");
                            break;
                        }
                    }
                }

                // Get current workers from DashMap
                let current_workers: HashMap<WorkerId, Option<ModelRuntimeConfig>> =
                    configs_monitor
                        .iter()
                        .map(|r| (*r.key(), r.value().clone()))
                        .collect();
                let current_worker_ids: HashSet<WorkerId> =
                    current_workers.keys().copied().collect();

                // Only update slots if workers have changed
                if current_worker_ids != last_workers {
                    slots_monitor.update_workers(current_workers);
                    last_workers = current_worker_ids;
                }
            }
        });

        let slots_clone = slots.clone();
        let workers_scheduler = workers_with_configs.clone();
        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<SchedulingRequest>(1024);
        let scheduler_cancel_token = component.drt().primary_token();
        let hit_rate_publisher =
            EventPublisher::for_namespace(component.namespace(), KV_HIT_RATE_SUBJECT)
                .await
                .map_err(|e| KvSchedulerError::InitFailed(e.to_string()))?;

        // Background task to handle scheduling requests
        tokio::spawn(async move {
            let mut request_rx = request_rx;
            tracing::trace!("scheduler background task started");

            loop {
                // Check for cancellation at beginning of loop
                if scheduler_cancel_token.is_cancelled() {
                    tracing::trace!("scheduler background task shutting down");
                    break;
                }

                // Wait for a new request
                let Some(mut request) = request_rx.recv().await else {
                    tracing::warn!("scheduler shutdown");
                    break;
                };
                tracing::trace!("received request to be scheduled");

                let (decode_blocks, prefill_tokens) = slots_clone
                    .potential_blocks_and_tokens(
                        request.token_seq.clone(),
                        request.isl_tokens,
                        request.overlaps.clone(),
                    )
                    .await;
                request.decode_blocks = decode_blocks;
                request.prefill_tokens = prefill_tokens;

                // Read the current workers configuration from DashMap
                let workers: HashMap<WorkerId, Option<ModelRuntimeConfig>> = workers_scheduler
                    .configs
                    .iter()
                    .map(|r| (*r.key(), r.value().clone()))
                    .collect();

                match selector.select_worker(&workers, &request, block_size) {
                    Ok(selection) => {
                        let event = KVHitRateEvent {
                            worker_id: selection.worker.worker_id,
                            dp_rank: selection.worker.dp_rank,
                            isl_blocks: selection.required_blocks as usize,
                            overlap_blocks: selection.overlap_blocks,
                        };
                        if let Err(e) = hit_rate_publisher.publish(&event).await {
                            tracing::warn!("Failed to publish KV hit rate event: {:?}", e);
                        }

                        let response = SchedulingResponse {
                            best_worker: selection.worker,
                            overlap_blocks: selection.overlap_blocks,
                        };
                        request.respond(response);

                        // Skip state update if not requested
                        if !request.update_states {
                            continue;
                        }

                        let Some(request_id) = request.maybe_request_id else {
                            tracing::error!(
                                "No request_id provided to add_request to the slot tracker"
                            );
                            continue;
                        };

                        if let Err(e) = slots_clone
                            .add_request(
                                request_id.clone(),
                                request.token_seq,
                                request.isl_tokens,
                                selection.overlap_blocks,
                                None, // expected_output_tokens not available in scheduler loop
                                selection.worker,
                                request.lora_name.clone(),
                            )
                            .await
                        {
                            tracing::warn!("Failed to add request {request_id}: {e}");
                        }
                    }
                    Err(KvSchedulerError::NoEndpoints) => {
                        tracing::trace!("no endpoints available; waiting for endpoints update");
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        continue;
                    }
                    Err(e) => {
                        tracing::error!("error scheduling request: {:?}", e);
                        break;
                    }
                }
            }

            tracing::trace!("background endpoint subscriber shutting down");
        });

        Ok(KvScheduler { request_tx, slots })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn schedule(
        &self,
        maybe_request_id: Option<String>,
        isl_tokens: usize,
        token_seq: Option<Vec<SequenceHash>>,
        overlaps: OverlapScores,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
    ) -> Result<WorkerWithDpRank, KvSchedulerError> {
        #[cfg(feature = "bench")]
        let start = Instant::now();

        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            maybe_request_id,
            token_seq,
            isl_tokens,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: router_config_override.cloned(),
            update_states,
            lora_name,
            resp_tx: Some(resp_tx), // Wrap in Some()
        };

        self.request_tx
            .send(request)
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;

        #[cfg(feature = "bench")]
        let send_elapsed = start.elapsed();

        let response = resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;

        #[cfg(feature = "bench")]
        let total_elapsed = start.elapsed();
        #[cfg(feature = "bench")]
        tracing::info!(
            isl_tokens,
            send_us = send_elapsed.as_micros() as u64,
            total_us = total_elapsed.as_micros() as u64,
            "scheduler.schedule completed"
        );

        Ok(response.best_worker)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn add_request(
        &self,
        request_id: String,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        expected_output_tokens: Option<u32>,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) -> Result<(), SequenceError> {
        self.slots
            .add_request(
                request_id,
                token_sequence,
                isl,
                overlap,
                expected_output_tokens,
                worker,
                lora_name,
            )
            .await
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.slots
            .mark_prefill_completed(&request_id.to_string())
            .await
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.slots.free(&request_id.to_string()).await
    }

    /// Get the worker type for this scheduler ("prefill" or "decode").
    /// Used for Prometheus metric labeling.
    pub fn worker_type(&self) -> &'static str {
        self.slots.worker_type()
    }

    pub async fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.slots
            .add_output_block(&request_id.to_string(), decay_fraction)
            .await
    }

    pub async fn get_potential_loads(
        &self,
        token_seq: Option<Vec<SequenceHash>>,
        isl_tokens: usize,
        overlaps: OverlapScores,
    ) -> Vec<PotentialLoad> {
        let (decode_blocks, prefill_tokens) = self
            .slots
            .potential_blocks_and_tokens(token_seq, isl_tokens, overlaps)
            .await;

        // Get all unique WorkerWithDpRank from both hashmaps
        let mut workers: HashSet<WorkerWithDpRank> = HashSet::new();
        workers.extend(decode_blocks.keys().copied());
        workers.extend(prefill_tokens.keys().copied());

        // Create PotentialLoad for each worker
        let mut loads = Vec::new();
        for worker in workers {
            loads.push(PotentialLoad {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                potential_prefill_tokens: prefill_tokens
                    .get(&worker)
                    .copied()
                    .unwrap_or(isl_tokens),
                potential_decode_blocks: decode_blocks.get(&worker).copied().unwrap_or(0),
            });
        }

        loads
    }

    /// Get active request counts grouped by LORA name
    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        self.slots.get_active_lora_counts()
    }
}

// Helper function for softmax sampling
// Returns a vec of workers: multiple if tied, single if sampled
fn softmax_sample(
    logits: &HashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> Vec<WorkerWithDpRank> {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Guard: if temperature is 0, return all keys with the smallest logit value (ties)
    if temperature == 0.0 {
        // Find the minimum logit value
        let min_logit = logits.values().fold(f64::INFINITY, |a, &b| a.min(b));

        // Use epsilon for floating-point comparison to avoid non-deterministic
        // selection due to tiny arithmetic differences (e.g., 1e-15)
        const EPSILON: f64 = 1e-5;
        let min_keys: Vec<_> = logits
            .iter()
            .filter(|&(_, &v)| (v - min_logit).abs() < EPSILON)
            .map(|(k, _)| *k)
            .collect();

        return min_keys;
    }

    let keys: Vec<_> = logits.keys().copied().collect();
    let values: Vec<_> = logits.values().copied().collect();

    // Find min and max for normalization
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        // All values are the same, uniform probability
        vec![1.0 / keys.len() as f64; keys.len()]
    } else {
        // Normalize values
        let normalized: Vec<_> = values
            .iter()
            .map(|&v| {
                // Lower is better, so negate
                // Note we don't need to do actual min-max norm here, just off by an offset
                let norm = v / (max_val - min_val);
                -norm
            })
            .collect();

        // Apply temperature and softmax
        let scaled: Vec<_> = normalized.iter().map(|&v| v / temperature).collect();

        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<_> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();

        let sum_exp: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&v| v / sum_exp).collect()
    };

    // Sample from the probability distribution
    let mut rng = rand::rng();
    let sample: f64 = rng.random();

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return vec![keys[i]];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    vec![keys[keys.len() - 1]]
}

/// Cached routing decision for a token sequence prefix.
/// Tracks which worker was selected for sequences starting with specific token patterns,
/// enabling sticky routing for related requests (same prefix) even when KV events are delayed.
#[derive(Debug, Clone)]
struct PrefixRoute {
    worker: WorkerWithDpRank,
    timestamp: Instant,
    prefix_len: usize,
}

/// Prefix-based routing cache with bounded size and TTL.
/// Maps from token prefix hash to recent routing decisions.
struct PrefixCache {
    // Token prefix hash → cached route
    cache: HashMap<u64, PrefixRoute>,
    max_entries: usize,
    ttl: Duration,
}

impl PrefixCache {
    fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_entries),
            max_entries,
            ttl: Duration::from_secs(300), // Fixed 5 minutes TTL
        }
    }

    /// Hash token prefix (first N tokens) to create a cache key.
    /// Uses first min(prefix_len, tokens.len()) tokens.
    fn hash_prefix(tokens: &[dynamo_tokens::SequenceHash], prefix_len: usize) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let actual_len = prefix_len.min(tokens.len());
        tokens[..actual_len].hash(&mut hasher);
        hasher.finish()
    }

    /// Find cached route for token sequence.
    /// Returns worker if a matching prefix is found and the route is still fresh.
    fn get(&self, tokens: &[dynamo_tokens::SequenceHash], min_prefix_len: usize) -> Option<WorkerWithDpRank> {
        // Try decreasing prefix lengths (longest match first)
        for prefix_len in (min_prefix_len..=tokens.len()).rev() {
            let hash = Self::hash_prefix(tokens, prefix_len);

            if let Some(route) = self.cache.get(&hash) {
                // Check if route is still fresh (within TTL)
                if route.timestamp.elapsed() < self.ttl {
                    tracing::debug!(
                        "Prefix cache hit: prefix_len={} worker_id={} dp_rank={} age={}ms",
                        route.prefix_len,
                        route.worker.worker_id,
                        route.worker.dp_rank,
                        route.timestamp.elapsed().as_millis()
                    );
                    return Some(route.worker);
                }
            }
        }

        None
    }

    /// Store a routing decision for future prefix matches.
    fn insert(&mut self, tokens: &[dynamo_tokens::SequenceHash], worker: WorkerWithDpRank, prefix_len: usize) {
        // Evict oldest entries if cache is full
        if self.cache.len() >= self.max_entries {
            // Find oldest entry
            if let Some(oldest_key) = self.cache.iter()
                .min_by_key(|(_, route)| route.timestamp)
                .map(|(k, _)| *k)
            {
                self.cache.remove(&oldest_key);
            }
        }

        let hash = Self::hash_prefix(tokens, prefix_len);
        self.cache.insert(hash, PrefixRoute {
            worker,
            timestamp: Instant::now(),
            prefix_len,
        });

        tracing::debug!(
            "Prefix cache insert: prefix_len={} worker_id={} dp_rank={}",
            prefix_len,
            worker.worker_id,
            worker.dp_rank
        );
    }

    /// Remove stale entries (older than TTL).
    fn evict_stale(&mut self) {
        self.cache.retain(|_, route| route.timestamp.elapsed() < self.ttl);
    }
}

// Default implementation matching the Python _cost_function
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    prefix_cache: Arc<Mutex<PrefixCache>>,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>) -> Self {
        let config = kv_router_config.unwrap_or_default();

        eprintln!(
            "[RUST] DefaultWorkerSelector: prefix_cache={} size={}",
            config.router_enable_prefix_cache,
            config.router_prefix_cache_size
        );

        Self {
            kv_router_config: config.clone(),
            prefix_cache: Arc::new(Mutex::new(PrefixCache::new(
                config.router_prefix_cache_size,
            ))),
        }
    }
}

impl WorkerSelector for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, Option<ModelRuntimeConfig>>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        if workers.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let isl = request.isl_tokens;
        let request_blocks = isl.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;

        let decode_blocks = &request.decode_blocks;
        let prefill_tokens = &request.prefill_tokens;

        // Check prefix cache if enabled and we have token sequence
        // Only use prefix cache when:
        // 1. Feature is enabled via config
        // 2. We have token_seq (not a bare prompt)
        // 3. Temperature is 0 (deterministic routing)
        // 4. Sequence is long enough (min 3 tokens for meaningful prefix)
        let use_prefix_cache = self.kv_router_config.router_enable_prefix_cache
            && request.token_seq.is_some();

        if use_prefix_cache {
            if let Some(ref token_seq) = request.token_seq {
                if token_seq.len() >= 3 {
                    // Try to find a cached route for this prefix
                    let mut cache = self.prefix_cache.lock().unwrap();

                    // Evict stale entries before lookup
                    cache.evict_stale();

                    if let Some(cached_worker) = cache.get(token_seq, 3) {
                        // Verify cached worker is still available
                        let data_parallel_size = workers
                            .get(&cached_worker.worker_id)
                            .and_then(|cfg| cfg.as_ref())
                            .map(|c| c.data_parallel_size)
                            .unwrap_or(1);

                        if cached_worker.dp_rank < data_parallel_size {
                            let best_overlap = *overlaps.get(&cached_worker).unwrap_or(&0);

                            tracing::info!(
                                "Using prefix cached route: worker_id={} dp_rank={} cached_blocks={}",
                                cached_worker.worker_id,
                                cached_worker.dp_rank,
                                best_overlap
                            );

                            return Ok(WorkerSelectionResult {
                                worker: cached_worker,
                                required_blocks: request_blocks as u64,
                                overlap_blocks: best_overlap,
                            });
                        } else {
                            tracing::debug!(
                                "Cached worker no longer available: worker_id={} dp_rank={} (max dp_rank={})",
                                cached_worker.worker_id,
                                cached_worker.dp_rank,
                                data_parallel_size - 1
                            );
                        }
                    }
                }
            }
        }

        let mut worker_logits = HashMap::new();

        // Use override if provided, otherwise use default config
        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.overlap_score_weight)
            .unwrap_or(self.kv_router_config.overlap_score_weight);

        // Calculate logits for each worker with dp_rank
        // Outer loop: iterate over all workers from runtime config
        // Inner loop: iterate over all dp_ranks for each worker
        for (worker_id, config) in workers.iter() {
            // Get data_parallel_size from runtime config
            // data_parallel_size defaults to 1 in ModelRuntimeConfig
            let data_parallel_size = config.as_ref().map(|c| c.data_parallel_size).unwrap_or(1); // Fallback if config is None

            // Iterate over all dp_ranks for this worker
            for dp_rank in 0..data_parallel_size {
                let worker = WorkerWithDpRank::new(*worker_id, dp_rank);

                // Get overlap for this worker (defaults to 0 if not in overlaps)
                let overlap = *overlaps.get(&worker).unwrap_or(&0);

                // this is the number of prefill tokens the worker would have if the request were scheduled there
                let prefill_token = *prefill_tokens.get(&worker).unwrap_or(&isl);
                let potential_prefill_block = (prefill_token as f64) / (block_size as f64);

                // this is the number of decode blocks the worker would have if the request were scheduled there
                let decode_block = *decode_blocks
                    .get(&worker)
                    .unwrap_or(&(potential_prefill_block.floor() as usize))
                    as f64;

                // Calculate logit (lower is better)
                let logit = overlap_weight * potential_prefill_block + decode_block;

                worker_logits.insert(worker, logit);

                tracing::info!(
                    "Formula for worker_id={} dp_rank={:?} with {overlap} cached blocks: {logit:.3} \
                     = {overlap_weight:.1} * prefill_blocks + decode_blocks \
                     = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3}",
                    worker.worker_id,
                    worker.dp_rank
                );
            }
        }

        // Use softmax sampling to select worker(s)
        // Use override if provided, otherwise use default config
        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);
        let candidates = softmax_sample(&worker_logits, temperature);

        // If multiple candidates (tied), use tree size as tie-breaker
        // If tree sizes are also equal, min_by_key uses HashMap iteration order (pseudo-random)
        let best_worker = if candidates.len() > 1 {
            tracing::info!("Multiple workers tied with same logit, using tree size as tie-breaker");
            *candidates
                .iter()
                .min_by_key(|worker| {
                    request
                        .overlaps
                        .tree_sizes
                        .get(worker)
                        .copied()
                        .unwrap_or(0)
                })
                .expect("candidates should not be empty")
        } else {
            candidates[0]
        };

        let best_logit = worker_logits[&best_worker];

        let best_overlap = *overlaps.get(&best_worker).unwrap_or(&0);

        // this is a runtime config set on a per worker basis, not per dp-rank
        let total_blocks_info = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.as_ref())
            .and_then(|cfg| cfg.total_kv_blocks)
            .map(|blocks| format!(", total blocks: {}", blocks))
            .unwrap_or_default();

        let tree_size = request
            .overlaps
            .tree_sizes
            .get(&best_worker)
            .copied()
            .unwrap_or(0);

        tracing::info!(
            "Selected worker: worker_id={} dp_rank={:?}, logit: {:.3}, cached blocks: {}, tree size: {}{}",
            best_worker.worker_id,
            best_worker.dp_rank,
            best_logit,
            best_overlap,
            tree_size,
            total_blocks_info
        );

        // Store routing decision in prefix cache for future requests
        if use_prefix_cache {
            if let Some(ref token_seq) = request.token_seq {
                if token_seq.len() >= 3 {
                    let mut cache = self.prefix_cache.lock().unwrap();
                    // Cache the route using the full sequence length as prefix
                    cache.insert(token_seq, best_worker, token_seq.len());
                }
            }
        }

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks as u64,
            overlap_blocks: overlaps.get(&best_worker).copied().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sample_single_key() {
        // Test that with a single key, softmax_sample always returns that key
        let mut logits = HashMap::new();
        let worker = WorkerWithDpRank::from_worker_id(42);
        logits.insert(worker, 0.5); // The value doesn't matter

        // Test with different temperatures
        for temperature in &[0.1, 1.0, 10.0] {
            let result = softmax_sample(&logits, *temperature);
            assert_eq!(result.len(), 1, "Should return exactly one worker");
            assert_eq!(result[0], worker, "Should return the only available worker");
        }

        // Test with different logit values
        logits.clear();
        logits.insert(worker, -100.0); // Very negative value
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);

        logits.clear();
        logits.insert(worker, 100.0); // Very positive value
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);

        logits.clear();
        logits.insert(worker, 0.0); // Zero value
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        // Test that with temperature 0, softmax_sample returns all keys with smallest logit
        let mut logits = HashMap::new();
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);
        let worker4 = WorkerWithDpRank::from_worker_id(4);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0); // This has the smallest logit
        logits.insert(worker3, 7.0);
        logits.insert(worker4, 3.5);

        // With temperature 0, should always return only worker2 (smallest logit)
        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.len(),
            1,
            "Should return one worker when there's no tie"
        );
        assert_eq!(
            result[0], worker2,
            "Should return worker with smallest logit when temperature is 0"
        );

        // Test with tied minimum logits
        logits.clear();
        let worker5 = WorkerWithDpRank::from_worker_id(5);
        let worker6 = WorkerWithDpRank::from_worker_id(6);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0); // Tied for smallest
        logits.insert(worker5, 3.0); // Tied for smallest
        logits.insert(worker6, 7.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.len(),
            2,
            "Should return all workers with smallest logit when tied"
        );
        assert!(
            result.contains(&worker2) && result.contains(&worker5),
            "Should contain both tied workers"
        );

        // Test with negative values
        logits.clear();
        let worker10 = WorkerWithDpRank::from_worker_id(10);
        let worker20 = WorkerWithDpRank::from_worker_id(20);
        let worker30 = WorkerWithDpRank::from_worker_id(30);
        logits.insert(worker10, -1.0);
        logits.insert(worker20, -5.0); // This has the smallest logit
        logits.insert(worker30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0], worker20,
            "Should handle negative logits correctly"
        );
    }
}

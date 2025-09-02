// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::component::{Component, Instance};
use dynamo_runtime::traits::events::EventPublisher;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

use super::KV_HIT_RATE_SUBJECT;
use super::KvRouterConfig;
use super::RouterConfigOverride;
use super::WorkerSelector;
use super::indexer::OverlapScores;
use super::protocols::WorkerSelectionResult;
use super::sequence::ActiveSequencesMultiWorker;

use crate::tokens::SequenceHash;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVHitRateEvent {
    pub worker_id: i64,
    pub isl_blocks: usize,
    pub overlap_blocks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLoad {
    pub worker_id: i64,
    pub potential_prefill_tokens: usize,
    pub potential_decode_blocks: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints aviailable to route work")]
    NoEndpoints,

    #[error("all workers busy")]
    AllWorkersBusy,

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker_id: i64,
    pub overlap_blocks: u32,
}

pub struct SchedulingRequest {
    pub request_id: String,
    pub token_seq: Vec<SequenceHash>,
    pub isl_tokens: usize,
    pub overlaps: OverlapScores,
    pub decode_blocks: HashMap<i64, usize>,
    pub prefill_tokens: HashMap<i64, usize>,
    // Router config overrides for this specific request
    pub router_config_override: Option<RouterConfigOverride>,
    // Whether to update scheduler states (false for query_instance_id requests)
    pub update_states: bool,
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
        mut instances_rx: watch::Receiver<Vec<Instance>>,
        mut runtime_configs_rx: watch::Receiver<HashMap<i64, ModelRuntimeConfig>>,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        replica_sync: bool,
    ) -> Result<Self, KvSchedulerError> {
        let selector = selector.unwrap_or(Box::new(DefaultWorkerSelector::default()));
        let mut instances: Vec<Instance> = instances_rx.borrow_and_update().clone();
        let mut runtime_configs: HashMap<i64, ModelRuntimeConfig> =
            runtime_configs_rx.borrow_and_update().clone();

        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<KVHitRateEvent>();
        let ns_clone = component.namespace().clone();
        tokio::spawn(async move {
            let mut event_rx = event_rx;
            while let Some(event) = event_rx.recv().await {
                if let Err(e) = ns_clone.publish(KV_HIT_RATE_SUBJECT, &event).await {
                    tracing::warn!("Failed to publish KV hit rate event: {:?}", e);
                }
            }
        });

        let worker_ids: Vec<i64> = instances
            .iter()
            .map(|instance| instance.instance_id)
            .collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            component,
            block_size as usize,
            worker_ids,
            replica_sync,
        ));

        let slots_clone = slots.clone();
        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<SchedulingRequest>(1024);
        // Background task to handle scheduling requests
        tokio::spawn(async move {
            let mut request_rx = request_rx;
            tracing::trace!("scheduler background task started");
            let mut workers_with_configs: HashMap<i64, Option<ModelRuntimeConfig>> = HashMap::new();
            let mut needs_rebuild = true;

            loop {
                // Check for instance updates (non-blocking)
                let instances_changed = instances_rx.has_changed();
                let configs_changed = runtime_configs_rx.has_changed();

                match instances_changed {
                    Ok(true) => {
                        instances = instances_rx.borrow_and_update().clone();
                        let worker_ids: Vec<i64> = instances
                            .iter()
                            .map(|instance| instance.instance_id)
                            .collect();
                        slots_clone.update_workers(worker_ids);
                        needs_rebuild = true;
                    }
                    Ok(false) => {}
                    Err(_) => {
                        tracing::warn!("endpoint watch sender shutdown");
                        break;
                    }
                }

                // Check for runtime config updates
                match configs_changed {
                    Ok(true) => {
                        runtime_configs = runtime_configs_rx.borrow_and_update().clone();
                        needs_rebuild = true;
                    }
                    Ok(false) => {}
                    Err(_) => {
                        tracing::warn!("runtime configs watch sender shutdown");
                    }
                }

                // Rebuild workers hashmap only when needed
                if needs_rebuild {
                    workers_with_configs.clear();
                    for instance in &instances {
                        let worker_id = instance.instance_id;
                        let config = runtime_configs.get(&worker_id).cloned();
                        if config.is_none() {
                            tracing::warn!("Runtime config not found for worker_id: {}", worker_id);
                        }
                        workers_with_configs.insert(worker_id, config);
                    }
                    needs_rebuild = false;
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

                match selector.select_worker(&workers_with_configs, &request, block_size) {
                    Ok(selection) => {
                        if let Err(e) = event_tx.send(KVHitRateEvent {
                            worker_id: selection.worker_id,
                            isl_blocks: selection.required_blocks as usize,
                            overlap_blocks: selection.overlap_blocks,
                        }) {
                            tracing::warn!("Failed to send KV hit rate event: {:?}", e);
                        }

                        let response = SchedulingResponse {
                            best_worker_id: selection.worker_id,
                            overlap_blocks: selection.overlap_blocks,
                        };
                        request.respond(response);

                        // Only update the state if update_states is true
                        if request.update_states {
                            let _ = slots_clone
                                .add_request(
                                    request.request_id,
                                    request.token_seq,
                                    request.isl_tokens,
                                    selection.overlap_blocks,
                                    selection.worker_id,
                                )
                                .await;
                        }

                        continue;
                    }
                    Err(KvSchedulerError::NoEndpoints) => {
                        tracing::trace!("no endpoints available; waiting for endpoints update");
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        continue;
                    }
                    // TODO: this is not actually hooked up
                    Err(KvSchedulerError::AllWorkersBusy) => {
                        tracing::trace!("all workers busy; waiting for more capacity");
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

    pub async fn schedule(
        &self,
        request_id: String,
        isl_tokens: usize,
        token_seq: Vec<SequenceHash>,
        overlaps: OverlapScores,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
    ) -> Result<i64, KvSchedulerError> {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            request_id,
            token_seq,
            isl_tokens,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: router_config_override.cloned(),
            update_states,
            resp_tx: Some(resp_tx), // Wrap in Some()
        };

        self.request_tx
            .send(request)
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;
        let response = resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;

        let best_worker_id = response.best_worker_id;
        Ok(best_worker_id)
    }

    pub async fn add_request(
        &self,
        request_id: String,
        token_sequence: Vec<SequenceHash>,
        isl: usize,
        overlap: u32,
        worker_id: i64,
    ) {
        let _ = self
            .slots
            .add_request(request_id, token_sequence, isl, overlap, worker_id)
            .await;
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) {
        let _ = self
            .slots
            .mark_prefill_completed(&request_id.to_string())
            .await;
    }

    pub async fn free(&self, request_id: &str) {
        let _ = self.slots.free(&request_id.to_string()).await;
    }

    pub async fn get_potential_loads(
        &self,
        token_seq: Vec<SequenceHash>,
        isl_tokens: usize,
        overlaps: OverlapScores,
    ) -> Vec<PotentialLoad> {
        let (decode_blocks, prefill_tokens) = self
            .slots
            .potential_blocks_and_tokens(token_seq, isl_tokens, overlaps)
            .await;

        // Get all unique worker IDs from both hashmaps
        let mut worker_ids: HashSet<i64> = HashSet::new();
        worker_ids.extend(decode_blocks.keys().copied());
        worker_ids.extend(prefill_tokens.keys().copied());

        // Create PotentialLoad for each worker
        let mut loads = Vec::new();
        for worker_id in worker_ids {
            loads.push(PotentialLoad {
                worker_id,
                potential_prefill_tokens: prefill_tokens
                    .get(&worker_id)
                    .copied()
                    .unwrap_or(isl_tokens),
                potential_decode_blocks: decode_blocks.get(&worker_id).copied().unwrap_or(0),
            });
        }

        loads
    }
}

// Helper function for softmax sampling
fn softmax_sample(logits: &HashMap<i64, f64>, temperature: f64) -> i64 {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Guard: if temperature is 0, return the key with the smallest logit value
    if temperature == 0.0 {
        // Find the minimum logit value
        let min_logit = logits.values().fold(f64::INFINITY, |a, &b| a.min(b));

        // Collect all keys with the minimum logit value (to handle ties)
        let min_keys: Vec<_> = logits
            .iter()
            .filter(|&(_, &v)| v == min_logit)
            .map(|(k, _)| *k)
            .collect();

        // Randomly select from the minimum keys (handles single key case naturally)
        let mut rng = rand::rng();
        let index = rng.random_range(0..min_keys.len());
        return min_keys[index];
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
            return keys[i];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    keys[keys.len() - 1]
}

// Default implementation matching the Python _cost_function
#[derive(Debug, Clone, Default)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
        }
    }
}

impl WorkerSelector for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<i64, Option<ModelRuntimeConfig>>,
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

        let mut worker_logits = HashMap::new();
        let mut max_logit = f64::NEG_INFINITY;

        // Calculate logits for each worker
        for worker_id in workers.keys() {
            let overlap = *overlaps.get(worker_id).unwrap_or(&0);

            // this is the number of prefill tokens the worker would have if the request were scheduled there
            let prefill_token = *prefill_tokens.get(worker_id).unwrap_or(&isl);
            let potential_prefill_block = (prefill_token as f64) / (block_size as f64);

            // this is the number of decode blocks the worker would have if the request were scheduled there
            let decode_block = *decode_blocks
                .get(worker_id)
                .unwrap_or(&(potential_prefill_block.floor() as usize))
                as f64;

            // Use override if provided, otherwise use default config
            let overlap_weight = request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.overlap_score_weight)
                .unwrap_or(self.kv_router_config.overlap_score_weight);

            // Calculate logit (lower is better)
            let logit = overlap_weight * potential_prefill_block + decode_block;
            max_logit = max_logit.max(logit);

            worker_logits.insert(*worker_id, logit);

            tracing::info!(
                "Formula for {worker_id} with {overlap} cached blocks: {logit:.3} \
                 = {overlap_weight:.1} * prefill_blocks + decode_blocks \
                 = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3}"
            );
        }

        // Use softmax sampling to select worker
        // Use override if provided, otherwise use default config
        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);
        let best_worker_id = softmax_sample(&worker_logits, temperature);
        let best_logit = worker_logits[&best_worker_id];

        let best_overlap = *overlaps.get(&best_worker_id).unwrap_or(&0);
        let total_blocks_info = workers
            .get(&best_worker_id)
            .and_then(|cfg| cfg.as_ref())
            .and_then(|cfg| cfg.total_kv_blocks)
            .map(|blocks| format!(", total blocks: {}", blocks))
            .unwrap_or_default();

        tracing::info!(
            "Selected worker: {}, logit: {:.3}, cached blocks: {}{}",
            best_worker_id,
            best_logit,
            best_overlap,
            total_blocks_info
        );

        Ok(WorkerSelectionResult {
            worker_id: best_worker_id,
            required_blocks: request_blocks as u64,
            overlap_blocks: overlaps.get(&best_worker_id).copied().unwrap_or(0),
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
        let worker_id = 42;
        logits.insert(worker_id, 0.5); // The value doesn't matter

        // Test with different temperatures
        for temperature in &[0.1, 1.0, 10.0] {
            let result = softmax_sample(&logits, *temperature);
            assert_eq!(result, worker_id, "Should return the only available worker");
        }

        // Test with different logit values
        logits.clear();
        logits.insert(worker_id, -100.0); // Very negative value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);

        logits.clear();
        logits.insert(worker_id, 100.0); // Very positive value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);

        logits.clear();
        logits.insert(worker_id, 0.0); // Zero value
        assert_eq!(softmax_sample(&logits, 1.0), worker_id);
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        // Test that with temperature 0, softmax_sample returns the key with smallest logit
        let mut logits = HashMap::new();
        logits.insert(1, 5.0);
        logits.insert(2, 3.0); // This has the smallest logit
        logits.insert(3, 7.0);
        logits.insert(4, 3.5);

        // With temperature 0, should always return worker 2 (smallest logit)
        for _ in 0..10 {
            let result = softmax_sample(&logits, 0.0);
            assert_eq!(
                result, 2,
                "Should return worker with smallest logit when temperature is 0"
            );
        }

        // Test with negative values
        logits.clear();
        logits.insert(10, -1.0);
        logits.insert(20, -5.0); // This has the smallest logit
        logits.insert(30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(result, 20, "Should handle negative logits correctly");
    }
}

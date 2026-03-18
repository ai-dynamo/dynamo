// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

use std::{
    iter,
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::Duration,
};

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use super::{
    DumpRequest, GetWorkersRequest, KvIndexerInterface, KvIndexerMetrics, KvRouterError, RadixTree,
    RoutingDecisionRequest, ShardedMatchRequest,
};
use crate::indexer::pruning::{BlockEntry, PruneConfig, PruneManager};
use crate::protocols::*;
use dynamo_tokens::SequenceHash;

/// A sharded KV Indexer that partitions the RadixTree across multiple independent shards.
///
/// ## Sharding Strategy
/// - Each worker is **permanently assigned** to a single shard on first event
/// - All KV blocks from a worker exist only in that worker's assigned shard
/// - New workers are assigned to the shard with the fewest workers (load balancing)
///
/// ## Operation
/// - **Events**: Routed directly to the worker's assigned shard
/// - **Match requests**: Broadcast to all shards (scatter-gather pattern)
/// - **Threading**: Each shard runs in its own thread with a single-threaded runtime
///
/// This design ensures no cross-shard synchronization for writes while enabling
/// parallel processing and better scalability.
pub struct KvIndexerSharded {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
    worker_assignments: DashMap<WorkerId, usize, FxBuildHasher>,
    worker_counts: Arc<Mutex<Vec<usize>>>,

    event_tx: Vec<mpsc::Sender<RouterEvent>>,
    request_broadcast_tx: broadcast::Sender<ShardedMatchRequest>,
    remove_worker_tx: Vec<mpsc::Sender<WorkerId>>,
    remove_worker_dp_rank_tx: Vec<mpsc::Sender<(WorkerId, DpRank)>>,
    dump_tx: Vec<mpsc::Sender<DumpRequest>>,
    routing_tx: Vec<mpsc::Sender<RoutingDecisionRequest>>,
    tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl KvIndexerSharded {
    /// Create a new `KvIndexerSharded`.
    ///
    /// ### Arguments
    ///
    /// * `token` - A `CancellationToken` for managing shutdown.
    /// * `shards` - A list of kvindexer shards.
    /// * `expiration_duration` - The amount of time that block usage should be buffered.
    /// * `ttl` - The time-to-live for blocks before they expire.
    /// * `prune_config` - Configuration for tree-size based pruning.
    ///
    /// ### Returns
    ///
    /// A new `KvIndexer`.
    pub fn new_with_frequency(
        token: CancellationToken,
        num_shards: usize,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        let worker_assignments = DashMap::with_hasher(FxBuildHasher);
        let worker_counts = Arc::new(Mutex::new(vec![0; num_shards]));

        let mut event_tx = Vec::new();
        let mut remove_worker_tx = Vec::new();
        let mut remove_worker_dp_rank_tx = Vec::new();
        let mut get_workers_tx = Vec::new();
        let mut dump_tx = Vec::new();
        let mut routing_tx = Vec::new();
        let tasks = Arc::new(Mutex::new(Vec::new()));

        let (request_broadcast_tx, _) = broadcast::channel::<ShardedMatchRequest>(1048576);

        for _ in 0..num_shards {
            let (shard_event_tx, mut shard_event_rx) = mpsc::channel::<RouterEvent>(2048);
            let (shard_remove_worker_tx, mut shard_remove_worker_rx) =
                mpsc::channel::<WorkerId>(16);
            let (shard_remove_worker_dp_rank_tx, mut shard_remove_worker_dp_rank_rx) =
                mpsc::channel::<(WorkerId, DpRank)>(16);
            let (shard_get_workers_tx, mut shard_get_workers_rx) =
                mpsc::channel::<GetWorkersRequest>(16);
            let (shard_dump_tx, mut shard_dump_rx) = mpsc::channel::<DumpRequest>(16);
            let (shard_routing_tx, mut shard_routing_rx) =
                mpsc::channel::<RoutingDecisionRequest>(2048);
            let (shard_prune_tx, mut shard_prune_rx) = mpsc::channel::<()>(1);
            let mut shard_broadcast_rx = request_broadcast_tx.subscribe();
            let cancel = token.clone();
            let metrics = metrics.clone();
            let prune_config_clone = prune_config.clone();

            event_tx.push(shard_event_tx);
            remove_worker_tx.push(shard_remove_worker_tx);
            remove_worker_dp_rank_tx.push(shard_remove_worker_dp_rank_tx);
            get_workers_tx.push(shard_get_workers_tx);
            dump_tx.push(shard_dump_tx);
            routing_tx.push(shard_routing_tx);

            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            tasks.lock().unwrap().push(std::thread::spawn(move || {
                runtime.block_on(async move {
                    let mut trie = RadixTree::new_with_frequency(expiration_duration);

                    // Create PruneManager if prune_config is specified
                    let mut prune_manager = prune_config_clone.map(|config| {
                        PruneManager::<BlockEntry>::new(50, config)
                    });
                    let mut event_id_counter = 0u64;

                    loop {
                        // Create a future that sleeps until the next expiration time
                        let expiry_fut = if let Some(ref pm) = prune_manager
                            && let Some(next_expiry) = pm.peek_next_expiry() {
                            tokio::time::sleep_until(next_expiry)
                        } else {
                            tokio::time::sleep(Duration::MAX)
                        };

                        tokio::select! {
                            biased;

                            _ = cancel.cancelled() => {
                                tracing::trace!("KvCacheIndexer progress loop shutting down");
                                return;
                            }

                            Some(worker) = shard_remove_worker_rx.recv() => {
                                trie.remove_worker(worker);
                            }

                            Some((worker_id, dp_rank)) = shard_remove_worker_dp_rank_rx.recv() => {
                                trie.remove_worker_dp_rank(worker_id, dp_rank);
                            }

                            Some(get_workers_req) = shard_get_workers_rx.recv() => {
                                let workers = trie.get_workers();
                                let _ = get_workers_req.resp.send(workers);
                            }

                            Some(_) = shard_prune_rx.recv() => {
                                // Tree size-based pruning triggered
                                let Some(ref mut pm) = prune_manager else { continue };
                                let Ok(pruned) = pm.prune(trie.current_size()) else { continue };

                                for p in pruned {
                                    event_id_counter += 1;
                                    let event = RouterEvent::new(
                                        p.worker.worker_id,
                                        KvCacheEvent {
                                            event_id: event_id_counter,
                                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                                block_hashes: vec![p.key],
                                            }),
                                            dp_rank: p.worker.dp_rank,
                                        }
                                    );
                                    let _ = trie.apply_event(event);
                                }
                            }

                            Some(event) = shard_event_rx.recv() => {
                                let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
                                // Only clone if we need the event for prune_manager afterward
                                let event_for_prune = prune_manager.is_some().then(|| event.clone());
                                let result = trie.apply_event(event);
                                let result_is_ok = result.is_ok();
                                metrics.increment_event_applied(event_type, result);

                                // Track blocks in PruneManager if TTL is enabled and event was stored successfully
                                let Some(ref mut pm) = prune_manager else { continue };
                                if !result_is_ok { continue };
                                let Some(ref event) = event_for_prune else { continue };
                                let KvCacheEventData::Stored(ref store_data) = event.event.data else { continue };

                                let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                                let block_entries: Vec<BlockEntry> = store_data.blocks.iter().enumerate().map(|(idx, block)| {
                                    BlockEntry {
                                        key: block.block_hash,
                                        worker,
                                        seq_position: idx,
                                    }
                                }).collect();
                                pm.insert(block_entries);

                                // Check if we need to prune due to tree size
                                let Some(ref pc) = pm.prune_config else { continue };
                                let current_size = trie.current_size();
                                if current_size > pc.max_tree_size {
                                    tracing::info!(
                                        "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                        current_size,
                                        pc.max_tree_size
                                    );
                                    let _ = shard_prune_tx.try_send(());
                                }
                            }

                            Some(routing_req) = shard_routing_rx.recv() => {
                                // Process routing decisions when TTL/pruning is enabled
                                let Some(ref mut pm) = prune_manager else { continue };

                                event_id_counter += 1;

                                let hashes = routing_req.local_hashes.iter().zip(routing_req.sequence_hashes.iter());
                                let stored_event = KvCacheEventData::Stored(KvCacheStoreData {
                                    parent_hash: None,
                                    blocks: hashes.map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                                        tokens_hash: *local_hash,
                                        block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                mm_extra_info: None,
                                    }).collect(),
                                });

                                let event = RouterEvent::new(
                                    routing_req.worker.worker_id,
                                    KvCacheEvent {
                                        event_id: event_id_counter,
                                        data: stored_event,
                                        dp_rank: routing_req.worker.dp_rank,
                                    }
                                );

                                if trie.apply_event(event).is_err() {
                                    continue;
                                }

                                let block_entries: Vec<BlockEntry> = routing_req.sequence_hashes.iter().enumerate().map(|(idx, h)| {
                                    BlockEntry {
                                        key: ExternalSequenceBlockHash(*h),
                                        worker: routing_req.worker,
                                        seq_position: idx,
                                    }
                                }).collect();
                                pm.insert(block_entries);

                                // Check if we need to prune due to tree size
                                let Some(ref pc) = pm.prune_config else { continue };
                                let current_size = trie.current_size();
                                if current_size > pc.max_tree_size {
                                    tracing::info!(
                                        "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                        current_size,
                                        pc.max_tree_size
                                    );
                                    let _ = shard_prune_tx.try_send(());
                                }
                            }

                            Some(dump_req) = shard_dump_rx.recv() => {
                                let events = trie.dump_tree_as_events();
                                let _ = dump_req.resp.send(events);
                            }

                            Ok(req) = shard_broadcast_rx.recv() => {
                                #[cfg(feature = "bench")]
                                let queue_wait = req.created_at.elapsed();
                                #[cfg(feature = "bench")]
                                let seq_len = req.sequence.len();

                                #[cfg(feature = "bench")]
                                let process_start = Instant::now();
                                let matches = trie.find_matches(req.sequence, req.early_exit);
                                #[cfg(feature = "bench")]
                                let process_time = process_start.elapsed();

                                #[cfg(feature = "bench")]
                                tracing::info!(
                                    seq_len,
                                    queue_wait_us = queue_wait.as_micros() as u64,
                                    process_us = process_time.as_micros() as u64,
                                    "sharded indexer: processed find_matches"
                                );
                                if let Err(e) = req.resp.send(matches).await {
                                    tracing::trace!("Failed to send match response: {:?}", e);
                                }
                            }

                            _ = expiry_fut => {
                                // TTL-based expiry triggered
                                let Some(ref mut pm) = prune_manager else { continue };

                                let expired = pm.pop_expired();
                                for e in expired {
                                    event_id_counter += 1;
                                    let event = RouterEvent::new(
                                        e.worker.worker_id,
                                        KvCacheEvent {
                                            event_id: event_id_counter,
                                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                                block_hashes: vec![e.key],
                                            }),
                                            dp_rank: e.worker.dp_rank,
                                        }
                                    );
                                    let _ = trie.apply_event(event);
                                }
                            }
                        }
                    }
                });

                tracing::debug!("KvCacheIndexer task completed");
            }));
        }

        Self {
            cancel: token,
            kv_block_size,
            worker_assignments,
            worker_counts,
            event_tx,
            request_broadcast_tx,
            remove_worker_tx,
            remove_worker_dp_rank_tx,
            dump_tx,
            routing_tx,
            tasks,
        }
    }

    pub fn block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn new(
        token: CancellationToken,
        num_shards: usize,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        Self::new_with_frequency(token, num_shards, None, kv_block_size, metrics, None)
    }
}

#[async_trait]
impl KvIndexerInterface for KvIndexerSharded {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        #[cfg(feature = "bench")]
        let start = Instant::now();
        #[cfg(feature = "bench")]
        let seq_len = sequence.len();
        #[cfg(feature = "bench")]
        let num_shards = self.event_tx.len();

        'match_loop: loop {
            let (match_tx, mut match_rx) = mpsc::channel(self.event_tx.len());
            let sharded_req = ShardedMatchRequest::new(sequence.clone(), false, match_tx);
            self.request_broadcast_tx
                .send(sharded_req)
                .map_err(|_| KvRouterError::IndexerOffline)?;

            let mut scores = OverlapScores::new();

            for response_num in 0..self.event_tx.len() {
                match match_rx.recv().await {
                    Some(response) => {
                        scores.scores.extend(response.scores);
                        scores.tree_sizes.extend(response.tree_sizes);

                        if response_num == 0 {
                            scores.frequencies = response.frequencies;
                        } else {
                            let diff = (response.frequencies.len() as i64)
                                - (scores.frequencies.len() as i64);

                            if diff > 0 {
                                scores.frequencies.extend(iter::repeat_n(0, diff as usize));
                            }

                            for i in 0..response.frequencies.len() {
                                scores.frequencies[i] += response.frequencies[i];
                            }
                        }
                    }
                    None => {
                        // This can only happen if the broadcast channel overflows.
                        // In this case, we don't want to recursively call find_matches again. Otherwise, we could overflow the stack.
                        continue 'match_loop;
                    }
                }
            }

            #[cfg(feature = "bench")]
            {
                let elapsed = start.elapsed();
                tracing::info!(
                    seq_len,
                    num_shards,
                    elapsed_us = elapsed.as_micros() as u64,
                    "find_matches (sharded) completed"
                );
            }
            return Ok(scores);
        }
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None, lora_name);
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        let shard = self
            .worker_assignments
            .entry(event.worker_id)
            .or_insert_with(|| {
                // Get the shard with the smallest amount of workers.
                let worker_counts = self.worker_counts.lock().unwrap();
                let selected_shard = worker_counts
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, value)| value)
                    .unwrap()
                    .0;
                drop(worker_counts);

                // Increment the count for this shard
                self.worker_counts.lock().unwrap()[selected_shard] += 1;
                selected_shard
            });

        self.event_tx[*shard].send(event).await.unwrap();
    }

    async fn remove_worker(&self, worker: WorkerId) {
        if let Some((_, shard)) = self.worker_assignments.remove(&worker) {
            self.worker_counts.lock().unwrap()[shard] -= 1;
            self.remove_worker_tx[shard].send(worker).await.unwrap();
        }
    }

    async fn remove_worker_dp_rank(&self, worker: WorkerId, dp_rank: DpRank) {
        // Worker is assigned to a single shard, so route there directly.
        // Don't remove from worker_assignments since other dp_ranks may still exist.
        if let Some(shard) = self.worker_assignments.get(&worker) {
            self.remove_worker_dp_rank_tx[*shard]
                .send((worker, dp_rank))
                .await
                .unwrap();
        }
    }

    /// Shutdown the KV Indexer.
    fn shutdown(&self) {
        self.cancel.cancel();
        let mut tasks = self.tasks.lock().unwrap();
        while !tasks.is_empty() {
            tasks.pop().unwrap().join().unwrap();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all_events = Vec::new();

        // Create channels for each shard
        let mut receivers = Vec::new();

        for shard_dump_tx in &self.dump_tx {
            let (resp_tx, resp_rx) = oneshot::channel();
            let dump_req = DumpRequest { resp: resp_tx };

            if let Err(e) = shard_dump_tx.send(dump_req).await {
                tracing::error!("Failed to send dump request to shard: {:?}", e);
                return Err(KvRouterError::IndexerOffline);
            }

            receivers.push(resp_rx);
        }

        // Collect results from all shards
        for resp_rx in receivers {
            match resp_rx.await {
                Ok(events) => all_events.extend(events),
                Err(_) => return Err(KvRouterError::IndexerDroppedRequest),
            }
        }

        Ok(all_events)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();

        self.process_routing_decision_internal(worker, local_hashes, sequence_hashes)
            .await
    }

    async fn flush(&self) -> usize {
        let curr_size = self
            .event_tx
            .iter()
            .map(|tx| tx.max_capacity() - tx.capacity())
            .sum();
        loop {
            if self
                .event_tx
                .iter()
                .all(|tx| tx.capacity() == tx.max_capacity())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        curr_size
    }
}

impl KvIndexerSharded {
    /// Internal method to process a routing decision with pre-computed hashes.
    async fn process_routing_decision_internal(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        // Route to the appropriate shard based on worker assignment
        let shard_idx = self
            .worker_assignments
            .get(&worker.worker_id)
            .map(|shard_idx| *shard_idx)
            .unwrap_or_default();

        self.routing_tx[shard_idx]
            .send(RoutingDecisionRequest {
                worker,
                local_hashes,
                sequence_hashes,
            })
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        Ok(())
    }
}

impl Drop for KvIndexerSharded {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use super::{
    DumpRequest, GetWorkersRequest, KvIndexerInterface, KvIndexerMetrics, KvRouterError,
    MatchRequest, RadixTree, RoutingDecisionRequest,
};
use crate::indexer::pruning::{BlockEntry, PruneConfig, PruneManager};
use crate::protocols::*;
use dynamo_tokens::SequenceHash;

fn stored_block_entries(event: &RouterEvent) -> Option<Vec<BlockEntry>> {
    let KvCacheEventData::Stored(ref store_data) = event.event.data else {
        return None;
    };

    let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
    Some(
        store_data
            .blocks
            .iter()
            .enumerate()
            .map(|(idx, block)| BlockEntry {
                key: block.block_hash,
                worker,
                seq_position: idx,
            })
            .collect(),
    )
}

fn apply_event_with_prune_tracking(
    trie: &mut RadixTree,
    event: RouterEvent,
    metrics: &KvIndexerMetrics,
    prune_manager: &mut Option<PruneManager<BlockEntry>>,
    prune_tx: &mpsc::Sender<()>,
) {
    let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
    let event_id = event.event.event_id;
    let worker_id = event.worker_id;
    let event_for_prune = prune_manager.is_some().then(|| event.clone());
    let result = trie.apply_event(event);
    let result_is_ok = result.is_ok();
    let tree_size = trie.current_size();
    tracing::trace!(
        "Applied KV event to global radix tree: event_type={event_type}, event_id={event_id}, worker_id={worker_id}, success={result_is_ok}, global_radix_tree_size={tree_size}"
    );
    metrics.increment_event_applied(event_type, result);

    let Some(pm) = prune_manager.as_mut() else {
        return;
    };
    if !result_is_ok {
        return;
    }
    let Some(ref event) = event_for_prune else {
        return;
    };
    let Some(block_entries) = stored_block_entries(event) else {
        return;
    };

    pm.insert(block_entries);

    let Some(ref pc) = pm.prune_config else {
        return;
    };
    let current_size = trie.current_size();
    if current_size > pc.max_tree_size {
        tracing::info!(
            "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
            current_size,
            pc.max_tree_size
        );
        let _ = prune_tx.try_send(());
    }
}

/// The KV Indexer, managing the KV store and handling events and match requests.
#[derive(Clone)]
pub struct KvIndexer {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// A sender for `RouterEvent`s.
    event_tx: mpsc::Sender<RouterEvent>,
    /// A sender for `MatchRequest`s.
    match_tx: mpsc::Sender<MatchRequest>,
    /// A sender for remove worker requests.
    remove_worker_tx: mpsc::Sender<WorkerId>,
    /// A sender for remove worker dp_rank requests.
    remove_worker_dp_rank_tx: mpsc::Sender<(WorkerId, DpRank)>,
    /// A sender for get workers requests.
    get_workers_tx: mpsc::Sender<GetWorkersRequest>,
    /// A sender for dump requests.
    dump_tx: mpsc::Sender<DumpRequest>,
    /// A sender for routing decision requests.
    routing_tx: mpsc::Sender<RoutingDecisionRequest>,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
    /// Reference counter for Clone-aware Drop.
    /// Only the last clone should cancel the token on drop.
    _ref_count: Arc<()>,
}

impl KvIndexer {
    /// Create a new `KvIndexer`.
    ///
    /// ### Arguments
    ///
    /// * `token` - A `CancellationToken` for managing shutdown.
    /// * `expiration_duration` - The amount of time that block usage should be buffered.
    /// * `ttl` - The time-to-live for blocks before they expire.
    /// * `prune_config` - Configuration for tree-size based pruning.
    ///
    /// ### Returns
    ///
    /// A new `KvIndexer`.
    pub fn new_with_frequency(
        token: CancellationToken,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel::<RouterEvent>(16384);
        let (match_tx, match_rx) = mpsc::channel::<MatchRequest>(128);
        let (remove_worker_tx, remove_worker_rx) = mpsc::channel::<WorkerId>(16);
        let (remove_worker_dp_rank_tx, remove_worker_dp_rank_rx) =
            mpsc::channel::<(WorkerId, DpRank)>(16);
        let (get_workers_tx, get_workers_rx) = mpsc::channel::<GetWorkersRequest>(16);
        let (dump_tx, dump_rx) = mpsc::channel::<DumpRequest>(16);
        let (routing_tx, mut routing_rx) = mpsc::channel::<RoutingDecisionRequest>(2048);
        let (prune_tx, mut prune_rx) = mpsc::channel::<()>(1);

        let cancel_clone = token.clone();

        std::thread::spawn(move || {
            // Create a single-threaded tokio runtime
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            runtime.block_on(async move {
                let cancel = cancel_clone;
                let mut match_rx = match_rx;
                let mut event_rx = event_rx;
                let mut remove_worker_rx = remove_worker_rx;
                let mut remove_worker_dp_rank_rx = remove_worker_dp_rank_rx;
                let mut get_workers_rx = get_workers_rx;
                let mut dump_rx = dump_rx;
                let mut trie = RadixTree::new_with_frequency(expiration_duration);

                // Create PruneManager if prune_config is specified
                let mut prune_manager = prune_config.map(|config| {
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
                            tracing::debug!("KvCacheIndexer progress loop shutting down");
                            return;
                        }

                        Some(worker) = remove_worker_rx.recv() => {
                            trie.remove_worker(worker);
                        }

                        Some((worker_id, dp_rank)) = remove_worker_dp_rank_rx.recv() => {
                            trie.remove_worker_dp_rank(worker_id, dp_rank);
                        }

                        Some(get_workers_req) = get_workers_rx.recv() => {
                            let workers = trie.get_workers();
                            let _ = get_workers_req.resp.send(workers);
                        }

                        Some(_) = prune_rx.recv() => {
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

                        Some(event) = event_rx.recv() => {
                            apply_event_with_prune_tracking(
                                &mut trie,
                                event,
                                &metrics,
                                &mut prune_manager,
                                &prune_tx,
                            );
                        }

                        Some(dump_req) = dump_rx.recv() => {
                            // Flush pending events so tree is consistent with buffer
                            while let Ok(event) = event_rx.try_recv() {
                                apply_event_with_prune_tracking(
                                    &mut trie,
                                    event,
                                    &metrics,
                                    &mut prune_manager,
                                    &prune_tx,
                                );
                            }
                            let events = trie.dump_tree_as_events();
                            let _ = dump_req.resp.send(events);
                        }

                        Some(routing_req) = routing_rx.recv() => {
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
                                let _ = prune_tx.try_send(());
                            }
                        }

                        Some(req) = match_rx.recv() => {
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
                                "indexer: processed find_matches"
                            );
                            let _ = req.resp.send(matches);
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
        });

        Self {
            cancel: token,
            event_tx,
            match_tx,
            remove_worker_tx,
            remove_worker_dp_rank_tx,
            get_workers_tx,
            dump_tx,
            routing_tx,
            kv_block_size,
            _ref_count: Arc::new(()),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn new(
        token: CancellationToken,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        Self::new_with_frequency(token, None, kv_block_size, metrics, None)
    }

    /// Get a sender for `RouterEvent`s.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.event_tx.clone()
    }

    #[cfg(test)]
    pub fn snapshot_event_sender(&self) -> mpsc::Sender<DumpRequest> {
        self.dump_tx.clone()
    }

    /// Get a sender for worker removal requests.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `WorkerId`s.
    pub fn remove_worker_sender(&self) -> mpsc::Sender<WorkerId> {
        self.remove_worker_tx.clone()
    }

    /// Get a sender for get workers requests.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `GetWorkersRequest`s.
    pub fn get_workers_sender(&self) -> mpsc::Sender<GetWorkersRequest> {
        self.get_workers_tx.clone()
    }
}

#[async_trait]
impl KvIndexerInterface for KvIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        #[cfg(feature = "bench")]
        let start = Instant::now();
        let seq_len = sequence.len();
        let (resp_tx, resp_rx) = oneshot::channel();
        let req = MatchRequest::new(sequence, false, resp_tx);

        if let Err(e) = self.match_tx.send(req).await {
            tracing::error!(
                "Failed to send match request: {:?}; the indexer maybe offline",
                e
            );
            return Err(KvRouterError::IndexerOffline);
        }

        let result = resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest);

        #[cfg(feature = "bench")]
        {
            let elapsed = start.elapsed();
            tracing::info!(
                seq_len,
                elapsed_us = elapsed.as_micros() as u64,
                "find_matches completed"
            );
        }
        #[cfg(not(feature = "bench"))]
        let _ = seq_len;

        result
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
        is_eagle: Option<bool>,
    ) -> Result<OverlapScores, KvRouterError> {
        tracing::debug!(
            "Finding matches for request tokens: {:?} / len: {}",
            tokens,
            tokens.len()
        );
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.kv_block_size,
            BlockHashOptions {
                lora_name,
                is_eagle,
                ..Default::default()
            },
        );
        tracing::debug!("Computed sequence: {:?}", sequence);
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        self.event_tx.send(event).await.unwrap();
    }

    async fn remove_worker(&self, worker: WorkerId) {
        self.remove_worker_tx.send(worker).await.unwrap();
    }

    async fn remove_worker_dp_rank(&self, worker: WorkerId, dp_rank: DpRank) {
        self.remove_worker_dp_rank_tx
            .send((worker, dp_rank))
            .await
            .unwrap();
    }

    fn shutdown(&self) {
        self.cancel.cancel();
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let dump_req = DumpRequest { resp: resp_tx };

        if let Err(e) = self.dump_tx.send(dump_req).await {
            tracing::error!("Failed to send dump request: {:?}", e);
            return Err(KvRouterError::IndexerOffline);
        }

        resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();

        self.process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
            .await
    }
    async fn flush(&self) -> usize {
        let curr_size = self.event_tx.max_capacity() - self.event_tx.capacity();
        loop {
            if self.event_tx.capacity() == self.event_tx.max_capacity() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        curr_size
    }
}

impl KvIndexer {
    /// Process a routing decision with pre-computed hashes.
    pub async fn process_routing_decision_with_hashes(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        self.routing_tx
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

impl Drop for KvIndexer {
    fn drop(&mut self) {
        // Only cancel the token if we're the last reference.
        // This allows clones to be dropped without killing the background task.
        if Arc::strong_count(&self._ref_count) == 1 {
            self.shutdown();
        }
    }
}

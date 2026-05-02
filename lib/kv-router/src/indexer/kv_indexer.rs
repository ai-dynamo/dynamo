// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use super::{
    DumpRequest, EventKind, FlushRequest, GetWorkersRequest, KvIndexerInterface, KvIndexerMetrics,
    KvRouterError, MatchDetails, MatchDetailsRequest, MatchRequest, PreBoundEventCounters,
    RadixTree, RoutingDecisionRequest,
};
use crate::indexer::pruning::{BlockEntry, PruneConfig, WorkerPruneManager};
use crate::protocols::*;
use dynamo_tokens::SequenceHash;

fn apply_event_with_counters(
    trie: &mut RadixTree,
    event: RouterEvent,
    counters: &PreBoundEventCounters,
) {
    let kind = EventKind::of(&event.event.data);
    let event_id = event.event.event_id;
    let worker_id = event.worker_id;
    let result = trie.apply_event_with_counters(event, Some(counters));
    let result_is_ok = result.is_ok();
    let tree_size = trie.current_size();
    tracing::trace!(
        "Applied KV event to global radix tree: event_type={kind}, event_id={event_id}, worker_id={worker_id}, success={result_is_ok}, global_radix_tree_size={tree_size}"
    );
    counters.inc(kind, result);
}

fn apply_routing_decision_with_prune_tracking(
    trie: &mut RadixTree,
    routing_req: RoutingDecisionRequest,
    prune_manager: &Option<WorkerPruneManager>,
    event_id_counter: &mut u64,
) {
    let Some(pm) = prune_manager.as_ref() else {
        return;
    };

    *event_id_counter += 1;

    let hashes = routing_req
        .local_hashes
        .iter()
        .zip(routing_req.sequence_hashes.iter());
    let stored_event = KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash: None,
        start_position: None,
        blocks: hashes
            .map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                tokens_hash: *local_hash,
                block_hash: ExternalSequenceBlockHash(*sequence_hash),
                mm_extra_info: None,
            })
            .collect(),
    });

    let event = RouterEvent::new(
        routing_req.worker.worker_id,
        KvCacheEvent {
            event_id: *event_id_counter,
            data: stored_event,
            dp_rank: routing_req.worker.dp_rank,
        },
    );

    if trie.apply_event(event).is_err() {
        return;
    }

    let block_entries = routing_req
        .sequence_hashes
        .iter()
        .enumerate()
        .map(|(idx, h)| BlockEntry {
            key: ExternalSequenceBlockHash(*h),
            worker: routing_req.worker,
            seq_position: idx,
        })
        .collect();
    pm.insert_block_entries(block_entries);
}

fn apply_prune_removes(trie: &mut RadixTree, entries: Vec<BlockEntry>, event_id_counter: &mut u64) {
    for entry in entries {
        *event_id_counter += 1;
        let event = RouterEvent::new(
            entry.worker.worker_id,
            KvCacheEvent {
                event_id: *event_id_counter,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![entry.key],
                }),
                dp_rank: entry.worker.dp_rank,
            },
        );
        let _ = trie.apply_event(event);
    }
}

struct PendingMutationReceivers<'a> {
    event_rx: &'a mut mpsc::Receiver<RouterEvent>,
    remove_worker_rx: &'a mut mpsc::Receiver<WorkerId>,
    remove_worker_dp_rank_rx: &'a mut mpsc::Receiver<(WorkerId, DpRank)>,
    routing_rx: &'a mut mpsc::Receiver<RoutingDecisionRequest>,
}

fn drain_pending_mutations(
    trie: &mut RadixTree,
    receivers: PendingMutationReceivers<'_>,
    counters: &PreBoundEventCounters,
    prune_manager: &Option<WorkerPruneManager>,
    event_id_counter: &mut u64,
) {
    while let Ok(worker) = receivers.remove_worker_rx.try_recv() {
        trie.remove_worker(worker);
        if let Some(pm) = prune_manager {
            pm.remove_worker(worker);
        }
    }

    while let Ok((worker_id, dp_rank)) = receivers.remove_worker_dp_rank_rx.try_recv() {
        trie.remove_worker_dp_rank(worker_id, dp_rank);
        if let Some(pm) = prune_manager {
            pm.remove_worker_dp_rank(WorkerWithDpRank::new(worker_id, dp_rank));
        }
    }

    while let Ok(event) = receivers.event_rx.try_recv() {
        apply_event_with_counters(trie, event, counters);
    }

    while let Ok(routing_req) = receivers.routing_rx.try_recv() {
        apply_routing_decision_with_prune_tracking(
            trie,
            routing_req,
            prune_manager,
            event_id_counter,
        );
    }

    if let Some(pm) = prune_manager {
        let entries = pm.drain_due_and_pending(tokio::time::Instant::now());
        apply_prune_removes(trie, entries, event_id_counter);
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
    /// A sender for `MatchDetailsRequest`s.
    match_details_tx: mpsc::Sender<MatchDetailsRequest>,
    /// A sender for remove worker requests.
    remove_worker_tx: mpsc::Sender<WorkerId>,
    /// A sender for remove worker dp_rank requests.
    remove_worker_dp_rank_tx: mpsc::Sender<(WorkerId, DpRank)>,
    /// A sender for get workers requests.
    get_workers_tx: mpsc::Sender<GetWorkersRequest>,
    /// A sender for dump requests.
    dump_tx: mpsc::Sender<DumpRequest>,
    /// A sender for flush requests.
    flush_tx: mpsc::Sender<FlushRequest>,
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
    /// * `prune_config` - Optional TTL configuration for approximate-mode routing decisions.
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
        super::warn_on_unit_block_size("single", kv_block_size);

        let (event_tx, event_rx) = mpsc::channel::<RouterEvent>(16384);
        let (match_tx, match_rx) = mpsc::channel::<MatchRequest>(128);
        let (match_details_tx, match_details_rx) = mpsc::channel::<MatchDetailsRequest>(128);
        let (remove_worker_tx, remove_worker_rx) = mpsc::channel::<WorkerId>(16);
        let (remove_worker_dp_rank_tx, remove_worker_dp_rank_rx) =
            mpsc::channel::<(WorkerId, DpRank)>(16);
        let (get_workers_tx, get_workers_rx) = mpsc::channel::<GetWorkersRequest>(16);
        let (dump_tx, dump_rx) = mpsc::channel::<DumpRequest>(16);
        let (flush_tx, flush_rx) = mpsc::channel::<FlushRequest>(16);
        let (routing_tx, mut routing_rx) = mpsc::channel::<RoutingDecisionRequest>(2048);

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
                let mut match_details_rx = match_details_rx;
                let mut event_rx = event_rx;
                let mut remove_worker_rx = remove_worker_rx;
                let mut remove_worker_dp_rank_rx = remove_worker_dp_rank_rx;
                let mut get_workers_rx = get_workers_rx;
                let mut dump_rx = dump_rx;
                let mut flush_rx = flush_rx;
                let mut trie = RadixTree::new_with_frequency(expiration_duration);

                let prune_manager = prune_config.map(WorkerPruneManager::new);
                let mut prune_ready_rx = prune_manager.as_ref().map(|pm| pm.subscribe_ready());
                let mut event_id_counter = 0u64;
                let counters = metrics.prebind();

                loop {
                    tokio::select! {
                        biased;

                        _ = cancel.cancelled() => {
                            tracing::debug!("KvCacheIndexer progress loop shutting down");
                            return;
                        }

                        Some(worker) = remove_worker_rx.recv() => {
                            trie.remove_worker(worker);
                            if let Some(pm) = &prune_manager {
                                pm.remove_worker(worker);
                            }
                        }

                        Some((worker_id, dp_rank)) = remove_worker_dp_rank_rx.recv() => {
                            trie.remove_worker_dp_rank(worker_id, dp_rank);
                            if let Some(pm) = &prune_manager {
                                pm.remove_worker_dp_rank(WorkerWithDpRank::new(worker_id, dp_rank));
                            }
                        }

                        Some(event) = event_rx.recv() => {
                            apply_event_with_counters(&mut trie, event, &counters);
                        }

                        Some(get_workers_req) = get_workers_rx.recv() => {
                            let workers = trie.get_workers();
                            let _ = get_workers_req.resp.send(workers);
                        }

                        Some(dump_req) = dump_rx.recv() => {
                            drain_pending_mutations(
                                &mut trie,
                                PendingMutationReceivers {
                                    event_rx: &mut event_rx,
                                    remove_worker_rx: &mut remove_worker_rx,
                                    remove_worker_dp_rank_rx: &mut remove_worker_dp_rank_rx,
                                    routing_rx: &mut routing_rx,
                                },
                                &counters,
                                &prune_manager,
                                &mut event_id_counter,
                            );
                            let events = trie.dump_tree_as_events();
                            let _ = dump_req.resp.send(events);
                        }

                        Some(flush_req) = flush_rx.recv() => {
                            drain_pending_mutations(
                                &mut trie,
                                PendingMutationReceivers {
                                    event_rx: &mut event_rx,
                                    remove_worker_rx: &mut remove_worker_rx,
                                    remove_worker_dp_rank_rx: &mut remove_worker_dp_rank_rx,
                                    routing_rx: &mut routing_rx,
                                },
                                &counters,
                                &prune_manager,
                                &mut event_id_counter,
                            );
                            let _ = flush_req.resp.send(());
                        }

                        Some(routing_req) = routing_rx.recv() => {
                            apply_routing_decision_with_prune_tracking(
                                &mut trie,
                                routing_req,
                                &prune_manager,
                                &mut event_id_counter,
                            );
                        }

                        _ = async {
                            if let Some(rx) = prune_ready_rx.as_mut() {
                                let _ = rx.changed().await;
                            } else {
                                std::future::pending::<()>().await;
                            }
                        } => {
                            if let Some(pm) = &prune_manager {
                                loop {
                                    let entries = pm.drain_pending_removes();
                                    if entries.is_empty() {
                                        break;
                                    }
                                    apply_prune_removes(
                                        &mut trie,
                                        entries,
                                        &mut event_id_counter,
                                    );
                                }
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

                        Some(req) = match_details_rx.recv() => {
                            let matches = trie.find_match_details(req.sequence, req.early_exit);
                            let _ = req.resp.send(matches);
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
            match_details_tx,
            remove_worker_tx,
            remove_worker_dp_rank_tx,
            get_workers_tx,
            dump_tx,
            flush_tx,
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

    pub async fn find_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.match_details_tx
            .send(MatchDetailsRequest::new(sequence, false, resp_tx))
            .await
            .map_err(|_| KvRouterError::IndexerOffline)?;

        resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)
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
        let (resp_tx, resp_rx) = oneshot::channel();
        let flush_req = FlushRequest { resp: resp_tx };

        if let Err(error) = self.flush_tx.send(flush_req).await {
            tracing::error!("Failed to send flush request: {:?}", error);
            return curr_size;
        }

        let _ = resp_rx.await;
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

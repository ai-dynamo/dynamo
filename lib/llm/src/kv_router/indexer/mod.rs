// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError},
    protocols::{
        DpRank, LocalBlockHash, OverlapScores, RouterEvent, TokensWithHashes, WorkerId,
        WorkerWithDpRank,
    },
};
use dynamo_runtime::{component::Component, traits::DistributedRuntimeProvider};
use dynamo_tokens::SequenceHash;
use tokio::sync::oneshot;

mod jetstream;
pub mod remote;
mod subscriber;
mod worker_query;

use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub(crate) use subscriber::start_subscriber;
pub(crate) use worker_query::start_worker_kv_query_endpoint;

/// Default TTL for entries in the predict-on-route side indexer when
/// `router_predicted_ttl_secs` is unset. Deliberately short: entries the
/// engine never confirms (cancelled requests, prefill failures) age out on
/// their own rather than lingering.
const DEFAULT_PREDICTED_TTL_SECS: f64 = 5.0;

#[derive(Clone)]
pub struct Indexer {
    inner: InnerIndexer,
    /// Optional side approximate indexer used by `--router-predict-on-route`.
    /// Populated by routing decisions with a short TTL; the engine's KV
    /// events go to `inner` only. `find_matches` queries both and returns
    /// the per-worker max overlap.
    ///
    /// Keeping this as a separate indexer avoids the sequence-hash mismatch
    /// problem: the router cannot reproduce the engine's sequence hashes
    /// (vLLM/SGLang salt theirs and use cryptographic digests), so
    /// speculatively inserting into the primary indexer would key the same
    /// block under router-computed and engine-computed hashes — polluting
    /// the tree and double-counting overlap. A side indexer is self-keyed
    /// and expires quickly, so there is nothing to promote.
    approx: Option<KvIndexer>,
}

#[derive(Clone)]
enum InnerIndexer {
    KvIndexer(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
    Remote(Arc<RemoteIndexer>),
    None,
}

impl Indexer {
    pub async fn new(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
        model_name: Option<&str>,
    ) -> Result<Self> {
        let inner = Self::build_inner(component, kv_router_config, block_size, model_name).await?;

        let approx = Self::build_approx(component, kv_router_config, block_size, &inner);

        Ok(Self { inner, approx })
    }

    async fn build_inner(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
        model_name: Option<&str>,
    ) -> Result<InnerIndexer> {
        if kv_router_config.overlap_score_weight == 0.0 {
            return Ok(InnerIndexer::None);
        }

        if kv_router_config.use_remote_indexer {
            let model_name = model_name
                .ok_or_else(|| {
                    anyhow::anyhow!("model_name is required when use_remote_indexer is configured")
                })?
                .to_string();
            let indexer_component_name = component.name();
            tracing::info!(
                indexer_component = %indexer_component_name,
                model_name,
                "Using remote KV indexer"
            );
            let remote =
                RemoteIndexer::new(component, model_name, kv_router_config.use_kv_events).await?;
            return Ok(InnerIndexer::Remote(Arc::new(remote)));
        }

        if !kv_router_config.use_kv_events {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            let cancellation_token = component.drt().primary_token();
            let prune_config = Some(PruneConfig {
                ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
                max_tree_size: kv_router_config.router_max_tree_size,
                prune_target_ratio: kv_router_config.router_prune_target_ratio,
            });
            return Ok(InnerIndexer::KvIndexer(KvIndexer::new_with_frequency(
                cancellation_token,
                None,
                block_size,
                kv_indexer_metrics,
                prune_config,
            )));
        }

        if kv_router_config.router_event_threads > 1 {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            return Ok(InnerIndexer::Concurrent(Arc::new(
                ThreadPoolIndexer::new_with_metrics(
                    ConcurrentRadixTreeCompressed::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics),
                ),
            )));
        }

        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let cancellation_token = component.drt().primary_token();

        Ok(InnerIndexer::KvIndexer(KvIndexer::new_with_frequency(
            cancellation_token,
            None,
            block_size,
            kv_indexer_metrics,
            None,
        )))
    }

    fn build_approx(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
        inner: &InnerIndexer,
    ) -> Option<KvIndexer> {
        if !kv_router_config.router_predict_on_route {
            return None;
        }

        // The side approximate indexer only makes sense when the primary is
        // event-driven — in pure-approximate mode the primary already inserts
        // on routing decisions, and a second indexer would double-count.
        if !kv_router_config.use_kv_events {
            return None;
        }

        match inner {
            InnerIndexer::None => None,
            InnerIndexer::Remote(_) => {
                tracing::warn!(
                    "--router-predict-on-route is not yet supported with --use-remote-indexer; ignoring"
                );
                None
            }
            InnerIndexer::KvIndexer(_) | InnerIndexer::Concurrent(_) => {
                let ttl_secs = kv_router_config
                    .router_predicted_ttl_secs
                    .unwrap_or(DEFAULT_PREDICTED_TTL_SECS);
                let prune_config = Some(PruneConfig {
                    ttl: Duration::from_secs_f64(ttl_secs),
                    max_tree_size: kv_router_config.router_max_tree_size,
                    prune_target_ratio: kv_router_config.router_prune_target_ratio,
                });
                let metrics = KvIndexerMetrics::from_component(component);
                let cancellation_token = component.drt().primary_token();
                tracing::info!(
                    ttl_secs,
                    "Starting predict-on-route side indexer (short-TTL approximate)"
                );
                Some(KvIndexer::new_with_frequency(
                    cancellation_token,
                    None,
                    block_size,
                    metrics,
                    prune_config,
                ))
            }
        }
    }

    pub(crate) async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let primary = self.inner.find_matches(sequence.clone()).await?;

        let Some(approx) = &self.approx else {
            return Ok(primary);
        };

        match approx.find_matches(sequence).await {
            Ok(side) => Ok(merge_overlap_scores(primary, side)),
            Err(error) => {
                tracing::warn!(error = %error, "predict-on-route side indexer query failed; using primary only");
                Ok(primary)
            }
        }
    }

    pub(crate) async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        // predict-on-route: route the write to the side approximate indexer so
        // we don't contaminate the event-driven primary with router-computed
        // sequence hashes.
        if let Some(approx) = &self.approx {
            return approx
                .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                .await;
        }
        self.inner
            .record_hashed_routing_decision(worker, local_hashes, sequence_hashes)
            .await
    }

    pub(crate) async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.inner.dump_events().await
    }

    pub(crate) async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        if self.approx.is_some() {
            let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
            let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
            return self
                .record_hashed_routing_decision(worker, local_hashes, sequence_hashes)
                .await;
        }
        self.inner
            .process_routing_decision_for_request(tokens_with_hashes, worker)
            .await
    }

    pub(crate) async fn apply_event(&self, event: RouterEvent) {
        self.inner.apply_event(event).await;
    }

    pub(crate) async fn remove_worker(&self, worker_id: WorkerId) {
        self.inner.remove_worker(worker_id).await;
        if let Some(approx) = &self.approx
            && let Err(e) = approx.remove_worker_sender().send(worker_id).await
        {
            tracing::warn!(
                "Failed to send worker removal for {worker_id} to predict-on-route side indexer: {e}"
            );
        }
    }

    pub(crate) async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.inner.remove_worker_dp_rank(worker_id, dp_rank).await;
        if let Some(approx) = &self.approx {
            KvIndexerInterface::remove_worker_dp_rank(approx, worker_id, dp_rank).await;
        }
    }

    pub(crate) async fn get_workers(&self) -> Vec<WorkerId> {
        self.inner.get_workers().await
    }
}

impl InnerIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => indexer.find_matches(sequence).await,
            Self::Concurrent(tpi) => tpi.find_matches(sequence).await,
            Self::Remote(remote) => match remote.find_matches(sequence).await {
                Ok(scores) => Ok(scores),
                Err(error) => {
                    tracing::warn!(error = %error, "Remote indexer query failed");
                    Ok(OverlapScores::new())
                }
            },
            Self::None => Ok(OverlapScores::new()),
        }
    }

    async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::Concurrent(_) => {
                tracing::warn!(
                    "Hashed routing-decision recording is unsupported for concurrent indexers"
                );
                Err(KvRouterError::IndexerDroppedRequest)
            }
            Self::Remote(remote) => remote
                .record_hashed_routing_decision(worker, local_hashes, sequence_hashes)
                .await
                .map_err(|error| {
                    tracing::warn!(error = %error, "Remote indexer write failed");
                    KvRouterError::IndexerDroppedRequest
                }),
            Self::None => Ok(()),
        }
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(_) | Self::Remote(_) => {
                let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
                let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
                self.record_hashed_routing_decision(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::Concurrent(tpi) => {
                tpi.process_routing_decision_for_request(tokens_with_hashes, worker)
                    .await
            }
            Self::None => Ok(()),
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => indexer.dump_events().await,
            Self::Concurrent(tpi) => tpi.dump_events().await,
            Self::Remote(_) => Ok(Vec::new()),
            Self::None => {
                panic!(
                    "Cannot dump events: indexer does not exist (is overlap_score_weight set to 0?)"
                );
            }
        }
    }

    async fn apply_event(&self, event: RouterEvent) {
        match self {
            Self::KvIndexer(indexer) => {
                if let Err(e) = indexer.event_sender().send(event).await {
                    tracing::warn!("Failed to send event to indexer: {e}");
                }
            }
            Self::Concurrent(tpi) => tpi.apply_event(event).await,
            Self::Remote(_) | Self::None => {}
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer(indexer) => {
                if let Err(e) = indexer.remove_worker_sender().send(worker_id).await {
                    tracing::warn!("Failed to send worker removal for {worker_id}: {e}");
                }
            }
            Self::Concurrent(tpi) => {
                KvIndexerInterface::remove_worker(tpi.as_ref(), worker_id).await;
            }
            Self::Remote(_) | Self::None => {}
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        match self {
            Self::KvIndexer(indexer) => {
                KvIndexerInterface::remove_worker_dp_rank(indexer, worker_id, dp_rank).await;
            }
            Self::Concurrent(tpi) => {
                KvIndexerInterface::remove_worker_dp_rank(tpi.as_ref(), worker_id, dp_rank).await;
            }
            Self::Remote(_) | Self::None => {}
        }
    }

    async fn get_workers(&self) -> Vec<WorkerId> {
        match self {
            Self::KvIndexer(indexer) => {
                let (resp_tx, resp_rx) = oneshot::channel();
                let req = dynamo_kv_router::indexer::GetWorkersRequest { resp: resp_tx };
                if let Err(e) = indexer.get_workers_sender().send(req).await {
                    tracing::warn!("Failed to send get_workers request: {e}");
                    return Vec::new();
                }
                resp_rx.await.unwrap_or_default()
            }
            Self::Concurrent(tpi) => tpi.backend().get_workers(),
            Self::Remote(_) | Self::None => Vec::new(),
        }
    }
}

/// Merge two `OverlapScores` by taking the per-worker max. Used when the
/// predict-on-route side indexer is enabled: the primary indexer holds the
/// authoritative event-driven view, and the side indexer covers the window
/// where no engine event has arrived yet. A worker's overlap is whichever
/// indexer saw the longer prefix.
///
/// `tree_sizes` and `frequencies` are taken from the primary (event-driven)
/// indexer — they feed the cost model's load and cache-hit signals, for which
/// the side indexer's short-TTL view isn't meaningful.
fn merge_overlap_scores(mut primary: OverlapScores, side: OverlapScores) -> OverlapScores {
    for (worker, side_score) in side.scores {
        primary
            .scores
            .entry(worker)
            .and_modify(|s| {
                if side_score > *s {
                    *s = side_score;
                }
            })
            .or_insert(side_score);
    }
    primary
}

#[cfg(test)]
mod tests {
    use super::*;

    use dynamo_kv_router::protocols::WorkerWithDpRank;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    #[test]
    fn merge_keeps_primary_when_side_is_empty() {
        let mut primary = OverlapScores::new();
        primary.scores.insert(worker(1), 3);
        primary.scores.insert(worker(2), 1);
        let side = OverlapScores::new();
        let merged = merge_overlap_scores(primary, side);
        assert_eq!(merged.scores.get(&worker(1)).copied(), Some(3));
        assert_eq!(merged.scores.get(&worker(2)).copied(), Some(1));
    }

    #[test]
    fn merge_fills_in_missing_workers_from_side() {
        let mut primary = OverlapScores::new();
        primary.scores.insert(worker(1), 2);
        let mut side = OverlapScores::new();
        side.scores.insert(worker(2), 4);
        let merged = merge_overlap_scores(primary, side);
        assert_eq!(merged.scores.get(&worker(1)).copied(), Some(2));
        assert_eq!(merged.scores.get(&worker(2)).copied(), Some(4));
    }

    #[test]
    fn merge_takes_max_when_both_present() {
        let mut primary = OverlapScores::new();
        primary.scores.insert(worker(1), 2);
        primary.scores.insert(worker(2), 5);
        let mut side = OverlapScores::new();
        side.scores.insert(worker(1), 7);
        side.scores.insert(worker(2), 3);
        let merged = merge_overlap_scores(primary, side);
        assert_eq!(merged.scores.get(&worker(1)).copied(), Some(7));
        assert_eq!(merged.scores.get(&worker(2)).copied(), Some(5));
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::ConcurrentRadixTree;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, ThreadPoolIndexer,
};
use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, RouterEvent, RoutingConstraints, StorageTier, WorkerId,
};
use dynamo_kv_router::scheduling::TierOverlapBlocks;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, KvCacheEventSink, KvEventPublishers, MockEngineArgs,
};
use crate::replay::router_shared::{
    ReplayScheduler, replay_router_config, replay_selector, replay_slots,
    replay_workers_with_configs,
};
use crate::replay::{ReplayPrefillLoadEstimator, ReplayRouterMode};

#[derive(Clone)]
enum ReplayIndexer {
    Single(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTree>>),
}

impl ReplayIndexer {
    async fn apply_event(&self, event: RouterEvent) {
        // TODO: support lower tier events in replay indexer
        if !event.storage_tier.is_gpu() {
            return;
        }
        match self {
            Self::Single(indexer) => indexer.apply_event(event).await,
            Self::Concurrent(indexer) => indexer.apply_event(event).await,
        }
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores> {
        match self {
            Self::Single(indexer) => indexer
                .find_matches_for_request(tokens, lora_name, None)
                .await
                .map_err(Into::into),
            Self::Concurrent(indexer) => indexer
                .find_matches_for_request(tokens, lora_name, None)
                .await
                .map_err(Into::into),
        }
    }

    async fn flush(&self) -> usize {
        match self {
            Self::Single(indexer) => indexer.flush().await,
            Self::Concurrent(indexer) => KvIndexerInterface::flush(indexer.as_ref()).await,
        }
    }
}

fn create_replay_indexer(block_size: u32, num_threads: usize) -> ReplayIndexer {
    if num_threads > 1 {
        return ReplayIndexer::Concurrent(Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTree::new(),
            num_threads,
            block_size,
        )));
    }

    ReplayIndexer::Single(KvIndexer::new_with_pruning(
        CancellationToken::new(),
        block_size,
        Arc::new(KvIndexerMetrics::new_unregistered()),
        None,
    ))
}

#[derive(Clone)]
struct ReplayKvEventSink {
    worker_id: WorkerId,
    event_tx: mpsc::UnboundedSender<RouterEvent>,
}

impl KvCacheEventSink for ReplayKvEventSink {
    fn publish(&self, event: dynamo_kv_router::protocols::KvCacheEvent) -> anyhow::Result<()> {
        self.event_tx
            .send(RouterEvent::new(self.worker_id, event))
            .map_err(|_| anyhow!("replay router event channel closed"))
    }

    fn publish_with_storage_tier(
        &self,
        event: dynamo_kv_router::protocols::KvCacheEvent,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.event_tx
            .send(RouterEvent::with_storage_tier(
                self.worker_id,
                event,
                storage_tier,
            ))
            .map_err(|_| anyhow!("replay router event channel closed"))
    }
}

#[derive(Default)]
pub(crate) struct RoundRobinRouter {
    next_worker_idx: AtomicUsize,
}

impl RoundRobinRouter {
    fn select_worker(&self, num_workers: usize) -> usize {
        self.next_worker_idx.fetch_add(1, Ordering::AcqRel) % num_workers
    }
}

pub(crate) struct KvReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    scheduler: Arc<ReplayScheduler>,
    scheduler_cancel: CancellationToken,
    event_tx: Mutex<Option<mpsc::UnboundedSender<RouterEvent>>>,
    event_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    indexer: ReplayIndexer,
}

impl KvReplayRouter {
    fn new(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        let config = replay_router_config(args, router_config);
        let indexer =
            create_replay_indexer(args.block_size as u32, config.router_event_threads as usize);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let (_worker_config_tx, worker_config_rx) =
            tokio::sync::watch::channel(workers_with_configs);
        let selector = replay_selector(&config);
        let profile = config
            .configured_policy_profile()
            .map_err(anyhow::Error::from)?;
        let scheduler_cancel = CancellationToken::new();
        let scheduler = Arc::new(dynamo_kv_router::LocalScheduler::new_with_policy_profile(
            slots,
            worker_config_rx,
            profile,
            args.block_size as u32,
            selector,
            prefill_load_estimator,
            None,
            None,
            config.router_queue_recheck_interval(),
            config.router_track_prefill_tokens,
            scheduler_cancel.clone(),
            "replay",
            false,
        ));
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let indexer_clone = indexer.clone();
        let event_task = tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                indexer_clone.apply_event(event).await;
            }
            let _ = indexer_clone.flush().await;
        });

        Ok(Self {
            config,
            block_size: args.block_size as u32,
            scheduler,
            scheduler_cancel,
            event_tx: Mutex::new(Some(event_tx)),
            event_task: Mutex::new(Some(event_task)),
            indexer,
        })
    }

    fn sink(&self, worker_id: WorkerId) -> Arc<dyn KvCacheEventSink> {
        let event_tx = self
            .event_tx
            .lock()
            .unwrap()
            .as_ref()
            .expect("router event channel should exist while runtime is active")
            .clone();
        Arc::new(ReplayKvEventSink {
            worker_id,
            event_tx,
        })
    }

    async fn select_worker(&self, request: &DirectRequest) -> Result<usize> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("online replay requires requests to have stable UUIDs"))?;
        let overlaps = self
            .indexer
            .find_matches_for_request(&request.tokens, None)
            .await?;
        let effective_overlap_blocks = overlaps
            .scores
            .iter()
            .map(|(worker, overlap)| (*worker, *overlap as f64))
            .collect();
        let effective_cached_tokens = overlaps
            .scores
            .iter()
            .map(|(worker, overlap)| {
                (
                    *worker,
                    (*overlap as usize) * usize::try_from(self.block_size).unwrap_or(0),
                )
            })
            .collect();
        let token_seq = self.config.compute_seq_hashes_for_tracking(
            &request.tokens,
            self.block_size,
            None,
            BlockHashOptions::default(),
            None,
        );
        let (priority_jump, strict_priority) = request.router_priorities();
        let response = self
            .scheduler
            .schedule_with_policy_class_and_block_hashes(
                Some(uuid.to_string()),
                request.tokens.len(),
                token_seq,
                None,
                TierOverlapBlocks::default(),
                effective_overlap_blocks,
                effective_cached_tokens,
                None,
                true,
                None,
                priority_jump,
                strict_priority,
                request.policy_class.clone(),
                Some(
                    u32::try_from(request.max_output_tokens)
                        .context("max_output_tokens does not fit into u32")?,
                ),
                None,
                None,
                RoutingConstraints::default(),
                None,
            )
            .await?;
        usize::try_from(response.best_worker.worker_id)
            .map_err(|_| anyhow!("selected worker id does not fit into usize"))
    }

    async fn mark_prefill_completed(&self, uuid: Uuid) -> Result<()> {
        self.scheduler
            .mark_prefill_completed(&uuid.to_string())
            .await
            .map_err(anyhow::Error::from)
    }

    async fn free(&self, uuid: Uuid) -> Result<()> {
        self.scheduler
            .free(&uuid.to_string())
            .await
            .map_err(anyhow::Error::from)
    }

    async fn shutdown(&self) -> Result<()> {
        self.scheduler_cancel.cancel();
        self.event_tx.lock().unwrap().take();
        let Some(event_task) = self.event_task.lock().unwrap().take() else {
            return Ok(());
        };
        event_task
            .await
            .map_err(|e| anyhow!("replay router event task failed: {e}"))?;
        Ok(())
    }

    #[cfg(test)]
    fn debug_potential_loads(
        &self,
        isl_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Vec<dynamo_kv_router::PotentialLoad> {
        self.scheduler.get_potential_loads(
            None,
            isl_tokens,
            std::collections::HashMap::new(),
            track_prefill_tokens,
        )
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "ReplayRouter is long-lived and the KV router variant is intentional"
)]
pub(crate) enum ReplayRouter {
    RoundRobin(RoundRobinRouter),
    Kv(KvReplayRouter),
}

impl ReplayRouter {
    pub(crate) fn new(
        mode: ReplayRouterMode,
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        Ok(match mode {
            ReplayRouterMode::RoundRobin => Self::RoundRobin(RoundRobinRouter::default()),
            ReplayRouterMode::KvRouter => Self::Kv(KvReplayRouter::new(
                args,
                router_config,
                prefill_load_estimator,
                num_workers,
            )?),
        })
    }

    pub(crate) fn sink(&self, worker_id: WorkerId) -> KvEventPublishers {
        match self {
            Self::RoundRobin(_) => KvEventPublishers::default(),
            Self::Kv(router) => KvEventPublishers::new(Some(router.sink(worker_id)), None),
        }
    }

    pub(crate) async fn select_worker(
        &self,
        request: &DirectRequest,
        num_workers: usize,
    ) -> Result<usize> {
        match self {
            Self::RoundRobin(router) => Ok(router.select_worker(num_workers)),
            Self::Kv(router) => router.select_worker(request).await,
        }
    }

    pub(crate) async fn on_first_token(&self, uuid: Uuid) -> Result<bool> {
        match self {
            Self::RoundRobin(_) => Ok(false),
            Self::Kv(router) => {
                router.mark_prefill_completed(uuid).await?;
                Ok(true)
            }
        }
    }

    pub(crate) async fn on_complete(&self, uuid: Uuid) -> Result<bool> {
        match self {
            Self::RoundRobin(_) => Ok(false),
            Self::Kv(router) => {
                router.free(uuid).await?;
                Ok(true)
            }
        }
    }

    pub(crate) async fn shutdown(&self) -> Result<()> {
        match self {
            Self::RoundRobin(_) => Ok(()),
            Self::Kv(router) => router.shutdown().await,
        }
    }

    #[cfg(test)]
    pub(crate) fn debug_potential_loads(
        &self,
        isl_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Vec<dynamo_kv_router::PotentialLoad> {
        match self {
            Self::RoundRobin(_) => Vec::new(),
            Self::Kv(router) => router.debug_potential_loads(isl_tokens, track_prefill_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_kv_router::config::RouterQueuePolicy;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, StorageTier, WorkerWithDpRank,
        compute_block_hash_for_seq,
    };

    use super::*;

    fn priority_request(uuid: u128, priority: i32, strict_priority: u32) -> DirectRequest {
        DirectRequest {
            tokens: vec![uuid as u32; 64],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(0.0),
            priority,
            strict_priority,
            policy_class: None,
        }
    }

    #[tokio::test]
    async fn online_replay_forwards_priorities_to_scheduler_queue() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(64))
            .build()
            .unwrap();
        let config = KvRouterConfig {
            router_queue_threshold: Some(0.5),
            router_queue_policy: RouterQueuePolicy::Fcfs,
            ..KvRouterConfig::default()
        };
        let router = Arc::new(
            ReplayRouter::new(ReplayRouterMode::KvRouter, &args, Some(config), None, 1).unwrap(),
        );

        let active = priority_request(1, 0, 0);
        router.select_worker(&active, 1).await.unwrap();

        let (completed_tx, mut completed_rx) = mpsc::unbounded_channel();
        let low_task = {
            let router = Arc::clone(&router);
            let completed_tx = completed_tx.clone();
            tokio::spawn(async move {
                let request = priority_request(2, 1_000, 0);
                router.select_worker(&request, 1).await.unwrap();
                completed_tx.send(2).unwrap();
            })
        };
        tokio::task::yield_now().await;
        let high_task = {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let request = priority_request(3, 0, 1);
                router.select_worker(&request, 1).await.unwrap();
                completed_tx.send(3).unwrap();
            })
        };
        tokio::task::yield_now().await;

        router.on_complete(Uuid::from_u128(1)).await.unwrap();
        let first = tokio::time::timeout(Duration::from_secs(1), completed_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first, 3);

        router.on_complete(Uuid::from_u128(3)).await.unwrap();
        let second = tokio::time::timeout(Duration::from_secs(1), completed_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second, 2);
        router.on_complete(Uuid::from_u128(2)).await.unwrap();

        low_task.await.unwrap();
        high_task.await.unwrap();
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn online_replay_forwards_policy_class_and_returns_config_errors() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(64))
            .build()
            .unwrap();
        let missing = KvRouterConfig {
            router_policy_config: Some("/definitely/missing/router-policy.yaml".to_string()),
            ..KvRouterConfig::default()
        };
        assert!(
            ReplayRouter::new(ReplayRouterMode::KvRouter, &args, Some(missing), None, 1,).is_err()
        );

        let path = std::env::temp_dir().join(format!(
            "dynamo-online-replay-policy-{}.yaml",
            Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency
    policy_family: latency
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
  - name: batch
    policy_family: batch
    cache_bucket: all
    quantum: 4
    prefill_busy_threshold: 1024
"#,
        )
        .unwrap();
        let router = Arc::new(
            ReplayRouter::new(
                ReplayRouterMode::KvRouter,
                &args,
                Some(KvRouterConfig {
                    router_policy_config: Some(path.display().to_string()),
                    ..KvRouterConfig::default()
                }),
                None,
                1,
            )
            .unwrap(),
        );
        std::fs::remove_file(path).unwrap();

        let mut active = priority_request(10, 0, 0);
        active.policy_class = Some("latency".to_string());
        router.select_worker(&active, 1).await.unwrap();

        let queued_task = {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let mut queued = priority_request(11, 0, 0);
                queued.policy_class = Some("latency".to_string());
                router.select_worker(&queued, 1).await.unwrap()
            })
        };
        tokio::task::yield_now().await;

        let mut batch = priority_request(12, 0, 0);
        batch.policy_class = Some("batch".to_string());
        assert_eq!(router.select_worker(&batch, 1).await.unwrap(), 0);
        assert!(
            !queued_task.is_finished(),
            "latency request should remain queued while its class is busy"
        );

        router.on_complete(Uuid::from_u128(10)).await.unwrap();
        router.on_complete(Uuid::from_u128(12)).await.unwrap();
        assert_eq!(queued_task.await.unwrap(), 0);
        router.on_complete(Uuid::from_u128(11)).await.unwrap();
        router.shutdown().await.unwrap();
    }

    fn store_event(
        worker_id: WorkerId,
        event_id: u64,
        tokens_hash: LocalBlockHash,
        storage_tier: StorageTier,
    ) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(event_id),
                        tokens_hash,
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
            storage_tier,
        )
    }

    #[tokio::test]
    async fn replay_indexer_ignores_lower_tier_events_for_primary_overlap() {
        let worker = WorkerWithDpRank::new(7, 0);
        let tokens = vec![1, 2, 3, 4];
        let tokens_hash = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())[0];
        let indexer = create_replay_indexer(4, 1);

        indexer
            .apply_event(store_event(7, 1, tokens_hash, StorageTier::HostPinned))
            .await;
        indexer.flush().await;
        let matches = indexer
            .find_matches_for_request(&tokens, None)
            .await
            .unwrap();
        assert_eq!(matches.scores.get(&worker), None);

        indexer
            .apply_event(store_event(7, 2, tokens_hash, StorageTier::Device))
            .await;
        indexer.flush().await;
        let matches = indexer
            .find_matches_for_request(&tokens, None)
            .await
            .unwrap();
        assert_eq!(matches.scores.get(&worker), Some(&1));
    }
}

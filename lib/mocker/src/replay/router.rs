// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, ThreadPoolIndexer,
};
use dynamo_kv_router::protocols::{
    ActiveLoad, ActiveSequenceEvent, OverlapScores, RouterEvent, WorkerConfigLike, WorkerId,
    WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, ConcurrentRadixTree, DefaultWorkerSelector, LocalScheduler,
    RouterSchedulingPolicy, SequencePublisher,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{DirectRequest, KvCacheEventSink, MockEngineArgs};
use crate::replay::ReplayRouterMode;

#[derive(Clone, Copy, Debug, Default)]
struct ReplayNoopPublisher;

impl SequencePublisher for ReplayNoopPublisher {
    fn publish_event(
        &self,
        _event: &ActiveSequenceEvent,
    ) -> impl future::Future<Output = anyhow::Result<()>> + Send {
        future::ready(Ok(()))
    }

    fn publish_load(&self, _load: ActiveLoad) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReplayWorkerConfig {
    max_num_batched_tokens: u64,
    total_kv_blocks: u64,
}

impl WorkerConfigLike for ReplayWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        1
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(self.max_num_batched_tokens)
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        Some(self.total_kv_blocks)
    }
}

#[derive(Clone)]
enum ReplayIndexer {
    Single(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTree>>),
}

impl ReplayIndexer {
    async fn apply_event(&self, event: RouterEvent) {
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
                .find_matches_for_request(tokens, lora_name)
                .await
                .map_err(Into::into),
            Self::Concurrent(indexer) => indexer
                .find_matches_for_request(tokens, lora_name)
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

    ReplayIndexer::Single(KvIndexer::new_with_frequency(
        CancellationToken::new(),
        None,
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
    fn publish(
        &self,
        event: dynamo_kv_router::protocols::KvCacheEvent,
        _block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        self.event_tx
            .send(RouterEvent::new(self.worker_id, event))
            .map_err(|_| anyhow!("replay router event channel closed"))
    }
}

#[derive(Default)]
pub(super) struct RoundRobinRouter {
    next_worker_idx: AtomicUsize,
}

impl RoundRobinRouter {
    fn select_worker(&self, num_workers: usize) -> usize {
        self.next_worker_idx.fetch_add(1, Ordering::AcqRel) % num_workers
    }
}

pub(super) struct KvReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    scheduler: Arc<
        LocalScheduler<
            ReplayNoopPublisher,
            ReplayWorkerConfig,
            RouterSchedulingPolicy,
            DefaultWorkerSelector,
        >,
    >,
    event_tx: Mutex<Option<mpsc::UnboundedSender<RouterEvent>>>,
    event_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    indexer: ReplayIndexer,
}

impl KvReplayRouter {
    fn new(args: &MockEngineArgs, num_workers: usize) -> Self {
        let config = KvRouterConfig::default();
        let indexer =
            create_replay_indexer(args.block_size as u32, config.router_event_threads as usize);
        let worker_config = ReplayWorkerConfig {
            max_num_batched_tokens: args
                .max_num_batched_tokens
                .map(|tokens| tokens as u64)
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS),
            total_kv_blocks: args.num_gpu_blocks as u64,
        };
        let workers_with_configs: HashMap<WorkerId, ReplayWorkerConfig> = (0..num_workers)
            .map(|worker_idx| (worker_idx as WorkerId, worker_config.clone()))
            .collect();
        let dp_range = workers_with_configs
            .keys()
            .copied()
            .map(|worker_id| (worker_id, (0, 1)))
            .collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            ReplayNoopPublisher,
            args.block_size,
            dp_range,
            false,
            0,
            "replay",
        ));
        let (_worker_config_tx, worker_config_rx) =
            tokio::sync::watch::channel(workers_with_configs);
        let selector = DefaultWorkerSelector::new(Some(config.clone()), "replay");
        let policy = RouterSchedulingPolicy::new(config.router_queue_policy, args.block_size);
        let scheduler = Arc::new(LocalScheduler::new(
            slots,
            worker_config_rx,
            config.router_queue_threshold,
            args.block_size as u32,
            selector,
            policy,
            CancellationToken::new(),
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

        Self {
            config,
            block_size: args.block_size as u32,
            scheduler,
            event_tx: Mutex::new(Some(event_tx)),
            event_task: Mutex::new(Some(event_task)),
            indexer,
        }
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
        let token_seq = self.config.compute_seq_hashes_for_tracking(
            &request.tokens,
            self.block_size,
            None,
            None,
        );
        let response = self
            .scheduler
            .schedule(
                Some(uuid.to_string()),
                request.tokens.len(),
                token_seq,
                overlaps,
                None,
                true,
                None,
                0.0,
                Some(
                    u32::try_from(request.max_output_tokens)
                        .context("max_output_tokens does not fit into u32")?,
                ),
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
        self.event_tx.lock().unwrap().take();
        let Some(event_task) = self.event_task.lock().unwrap().take() else {
            return Ok(());
        };
        event_task
            .await
            .map_err(|e| anyhow!("replay router event task failed: {e}"))?;
        Ok(())
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "ReplayRouter is long-lived and the KV router variant is intentional"
)]
pub(super) enum ReplayRouter {
    RoundRobin(RoundRobinRouter),
    Kv(KvReplayRouter),
}

impl ReplayRouter {
    pub(super) fn new(mode: ReplayRouterMode, args: &MockEngineArgs, num_workers: usize) -> Self {
        match mode {
            ReplayRouterMode::RoundRobin => Self::RoundRobin(RoundRobinRouter::default()),
            ReplayRouterMode::KvRouter => Self::Kv(KvReplayRouter::new(args, num_workers)),
        }
    }

    pub(super) fn sink(&self, worker_id: WorkerId) -> Option<Arc<dyn KvCacheEventSink>> {
        match self {
            Self::RoundRobin(_) => None,
            Self::Kv(router) => Some(router.sink(worker_id)),
        }
    }

    pub(super) async fn select_worker(
        &self,
        request: &DirectRequest,
        num_workers: usize,
    ) -> Result<usize> {
        match self {
            Self::RoundRobin(router) => Ok(router.select_worker(num_workers)),
            Self::Kv(router) => router.select_worker(request).await,
        }
    }

    pub(super) async fn on_first_token(&self, uuid: Uuid) -> Result<bool> {
        match self {
            Self::RoundRobin(_) => Ok(false),
            Self::Kv(router) => {
                router.mark_prefill_completed(uuid).await?;
                Ok(true)
            }
        }
    }

    pub(super) async fn on_complete(&self, uuid: Uuid) -> Result<bool> {
        match self {
            Self::RoundRobin(_) => Ok(false),
            Self::Kv(router) => {
                router.free(uuid).await?;
                Ok(true)
            }
        }
    }

    pub(super) async fn shutdown(&self) -> Result<()> {
        match self {
            Self::RoundRobin(_) => Ok(()),
            Self::Kv(router) => router.shutdown().await,
        }
    }
}

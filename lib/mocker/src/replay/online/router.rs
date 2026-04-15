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
use dynamo_kv_router::protocols::{BlockHashOptions, OverlapScores, RouterEvent, WorkerId};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, KvCacheEventSink, KvEventPublishers, MockEngineArgs,
};
use crate::replay::router_shared::{
    ReplayScheduler, replay_policy, replay_router_config, replay_selector, replay_slots,
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
    fn publish(&self, event: dynamo_kv_router::protocols::KvCacheEvent) -> anyhow::Result<()> {
        self.event_tx
            .send(RouterEvent::new(self.worker_id, event))
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
    ) -> Self {
        let config = replay_router_config(args, router_config);
        let indexer =
            create_replay_indexer(args.block_size as u32, config.router_event_threads as usize);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let (_worker_config_tx, worker_config_rx) =
            tokio::sync::watch::channel(workers_with_configs);
        let selector = replay_selector(&config);
        let policy = replay_policy(&config, args);
        let scheduler = Arc::new(dynamo_kv_router::LocalScheduler::new(
            slots,
            worker_config_rx,
            config.router_queue_threshold,
            args.block_size as u32,
            selector,
            policy,
            prefill_load_estimator,
            config.router_queue_recheck_interval(),
            config.router_track_prefill_tokens,
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
            BlockHashOptions::default(),
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
                None,
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

    #[cfg(test)]
    fn debug_potential_loads(
        &self,
        isl_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Vec<dynamo_kv_router::PotentialLoad> {
        self.scheduler.get_potential_loads(
            None,
            isl_tokens,
            OverlapScores::default(),
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
    ) -> Self {
        match mode {
            ReplayRouterMode::RoundRobin => Self::RoundRobin(RoundRobinRouter::default()),
            ReplayRouterMode::KvRouter => Self::Kv(KvReplayRouter::new(
                args,
                router_config,
                prefill_load_estimator,
                num_workers,
            )),
        }
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

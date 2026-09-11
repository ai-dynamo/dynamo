// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::indexer::{
    KvIndexer, KvIndexerInterface, KvIndexerMetrics, ThreadPoolIndexer,
};
use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, RouterEvent, RoutingConstraints, StorageTier, WorkerId,
};
use dynamo_kv_router::scheduling::TierOverlapBlocks;
use dynamo_kv_router::{
    ConcurrentRadixTree, RoutingPartitionRef, TrackingHashContext, TrackingHashScope,
};
use tokio::sync::{mpsc, oneshot};
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
                .find_matches_for_request(tokens, lora_name, None, None)
                .await
                .map_err(Into::into),
            Self::Concurrent(indexer) => indexer
                .find_matches_for_request(tokens, lora_name, None, None)
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

/// Work item on the replay router's KV event lane.
///
/// `Barrier` rides the same channel as the events it orders behind. Because the
/// channel is FIFO, a barrier queued after a batch of events is only acknowledged
/// once every one of them has been applied -- which is what
/// [`KvReplayRouter::await_indexed`] needs and what a bare `flush()` cannot give,
/// since `flush()` only settles the indexer's own internal queue and knows nothing
/// about events still sitting in this channel.
enum ReplayIndexerMessage {
    Event(Box<RouterEvent>),
    Barrier(oneshot::Sender<()>),
}

#[derive(Clone)]
struct ReplayKvEventSink {
    worker_id: WorkerId,
    event_tx: mpsc::UnboundedSender<ReplayIndexerMessage>,
}

impl ReplayKvEventSink {
    fn send(&self, event: RouterEvent) -> anyhow::Result<()> {
        self.event_tx
            .send(ReplayIndexerMessage::Event(Box::new(event)))
            .map_err(|_| anyhow!("replay router event channel closed"))
    }
}

impl KvCacheEventSink for ReplayKvEventSink {
    fn publish(&self, event: dynamo_kv_router::protocols::KvCacheEvent) -> anyhow::Result<()> {
        self.send(RouterEvent::new(self.worker_id, event))
    }

    fn publish_with_storage_tier(
        &self,
        event: dynamo_kv_router::protocols::KvCacheEvent,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.send(RouterEvent::with_storage_tier(
            self.worker_id,
            event,
            storage_tier,
        ))
    }
}

pub(crate) struct RoundRobinRouter {
    next_worker_idx: AtomicUsize,
    next_rank_by_worker: Box<[AtomicUsize]>,
}

impl RoundRobinRouter {
    fn new(num_workers: usize) -> Self {
        Self {
            next_worker_idx: AtomicUsize::new(0),
            next_rank_by_worker: (0..num_workers).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    fn select_worker(
        &self,
        num_workers: usize,
        dp_size: usize,
        preferred_dp_rank: Option<u32>,
    ) -> Result<ReplayPlacement> {
        anyhow::ensure!(
            self.next_rank_by_worker.len() == num_workers,
            "online replay worker topology changed after router construction"
        );
        let worker_idx = self.next_worker_idx.fetch_add(1, Ordering::AcqRel) % num_workers;
        let dp_rank = match preferred_dp_rank {
            Some(dp_rank) => {
                let dp_rank = usize::try_from(dp_rank)
                    .map_err(|_| anyhow!("preferred attention-DP rank does not fit usize"))?;
                anyhow::ensure!(
                    dp_rank < dp_size,
                    "preferred attention-DP rank {dp_rank} is out of range for dp_size {dp_size}"
                );
                dp_rank
            }
            None => self.next_rank_by_worker[worker_idx].fetch_add(1, Ordering::AcqRel) % dp_size,
        };
        Ok(ReplayPlacement {
            worker_idx,
            dp_rank,
        })
    }
}

/// One router decision in the logical-worker/attention-DP topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplayPlacement {
    pub(crate) worker_idx: usize,
    pub(crate) dp_rank: usize,
}

pub(crate) struct KvReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    scheduler: Arc<ReplayScheduler>,
    scheduler_cancel: CancellationToken,
    event_tx: Mutex<Option<mpsc::UnboundedSender<ReplayIndexerMessage>>>,
    // Abort-on-drop, not a bare `JoinHandle`, for the same reason the admission
    // forwarder is (round-13 H8): `LiveRunSession::finish` reaches
    // `router.shutdown()` only on the happy path -- every `bail!` above it, and
    // every cancellation, drops the router instead, and a dropped `JoinHandle`
    // detaches its task rather than stopping it. This task owns the indexer and
    // parks on `recv()`, so it stops on its own only once every engine has
    // dropped its publisher; until then it outlives the run on a caller-owned
    // runtime. The happy path is unchanged -- `AbortOnDropHandle` is a `Future`,
    // so `shutdown` still awaits ordered settlement.
    event_task: Mutex<Option<tokio_util::task::AbortOnDropHandle<()>>>,
    indexer: ReplayIndexer,
    tracking_hash: TrackingHashContext,
    #[cfg(test)]
    fail_mark_prefill: AtomicBool,
    #[cfg(test)]
    fail_free: AtomicBool,
}

impl KvReplayRouter {
    fn new(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        let config = replay_router_config(args, router_config);
        let tracking_hash = TrackingHashContext::from_config(&config)?;
        let indexer =
            create_replay_indexer(args.block_size as u32, config.router_event_threads as usize);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let (_worker_config_tx, worker_config_rx) =
            tokio::sync::watch::channel(workers_with_configs);
        let selector = replay_selector(&config)?;
        let profile = config
            .configured_policy_profile()
            .map_err(anyhow::Error::from)?;
        let scheduler_cancel = CancellationToken::new();
        let scheduler = Arc::new(dynamo_kv_router::LocalScheduler::new(
            slots,
            worker_config_rx,
            profile,
            args.block_size as u32,
            selector,
            prefill_load_estimator,
            None,
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
            while let Some(message) = event_rx.recv().await {
                match message {
                    ReplayIndexerMessage::Event(event) => {
                        indexer_clone.apply_event(*event).await;
                    }
                    ReplayIndexerMessage::Barrier(ack) => {
                        // Flush before acking: the events above are handed to the
                        // indexer asynchronously, so draining this channel alone does
                        // not mean they are queryable yet.
                        let _ = indexer_clone.flush().await;
                        let _ = ack.send(());
                    }
                }
            }
            let _ = indexer_clone.flush().await;
        });

        Ok(Self {
            config,
            block_size: args.block_size as u32,
            scheduler,
            scheduler_cancel,
            event_tx: Mutex::new(Some(event_tx)),
            event_task: Mutex::new(Some(tokio_util::task::AbortOnDropHandle::new(event_task))),
            indexer,
            tracking_hash,
            #[cfg(test)]
            fail_mark_prefill: AtomicBool::new(false),
            #[cfg(test)]
            fail_free: AtomicBool::new(false),
        })
    }

    /// Fallible, not `expect`-ing: the "runtime is active" invariant the old
    /// message asserted is not enforced anywhere -- `shutdown` takes the sender,
    /// and `await_indexed` already treats both the poisoned lock and the taken
    /// sender as ordinary error/`None` cases. Panicking here for conditions its
    /// sibling handles was the inconsistency, and a poisoned lock is reachable
    /// from any panic in the event task regardless of the invariant.
    fn sink(&self, worker_id: WorkerId) -> Result<Arc<dyn KvCacheEventSink>> {
        let event_tx = self
            .event_tx
            .lock()
            .map_err(|_| anyhow!("replay router event channel lock poisoned"))?
            .as_ref()
            .ok_or_else(|| {
                anyhow!("replay router event channel is closed; the router was already shut down")
            })?
            .clone();
        Ok(Arc::new(ReplayKvEventSink {
            worker_id,
            event_tx,
        }))
    }

    /// Wait until every KV event published before this call is visible to the indexer.
    ///
    /// Engines publish store/remove events into an unbounded channel drained by a
    /// detached task. Without this barrier, whether a routing decision sees a prefix
    /// that a just-completed request published is decided by when that task happened
    /// to get scheduled -- so the same trace could route to a warm or a cold worker
    /// run to run. `ReplayIndexer::flush` alone is not enough: it settles the
    /// indexer's internal queue but not events still in this channel.
    ///
    /// This makes visibility a function of publish order rather than of task
    /// scheduling. It does not make online KV replay reproducible on its own --
    /// whether a completing request publishes before the next routing decision is
    /// still wall-clock dependent, which is inherent to a live replay.
    async fn await_indexed(&self) -> Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        // Scoped so the std mutex guard is dropped before the await below.
        let queued = {
            let event_tx = self
                .event_tx
                .lock()
                .map_err(|_| anyhow!("replay router event channel lock poisoned"))?;
            match event_tx.as_ref() {
                Some(event_tx) => event_tx.send(ReplayIndexerMessage::Barrier(ack_tx)).is_ok(),
                // `shutdown` already took the sender; the forwarder has drained and
                // flushed, so there is nothing left to wait for.
                None => false,
            }
        };
        if !queued {
            return Ok(());
        }
        ack_rx
            .await
            .map_err(|_| anyhow!("replay router event task dropped an indexer barrier"))
    }

    async fn select_worker(&self, request: &DirectRequest) -> Result<ReplayPlacement> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("online replay requires requests to have stable UUIDs"))?;
        self.await_indexed().await?;
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
                // Plain widening cast, not `try_from(..).unwrap_or(0)`: the
                // conversion is infallible for `u32` on every supported target,
                // and a silent `0` would not degrade gracefully -- it would zero
                // every worker's cached-token figure and make the scheduler route
                // as if no prefix cache existed anywhere. The offline path already
                // spells the same conversion `self.block_size as usize`.
                (*worker, (*overlap as usize) * self.block_size as usize)
            })
            .collect();
        let token_seq = self.config.compute_seq_hashes_for_tracking_with_context(
            &self.tracking_hash,
            TrackingHashScope {
                partition: RoutingPartitionRef::new("replay", "default"),
                block_size: self.block_size,
            },
            &request.tokens,
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
        Ok(ReplayPlacement {
            worker_idx: usize::try_from(response.best_worker.worker_id)
                .map_err(|_| anyhow!("selected worker id does not fit into usize"))?,
            dp_rank: usize::try_from(response.best_worker.dp_rank)
                .map_err(|_| anyhow!("selected DP rank does not fit into usize"))?,
        })
    }

    async fn mark_prefill_completed(&self, uuid: Uuid) -> Result<()> {
        #[cfg(test)]
        if self.fail_mark_prefill.load(Ordering::Acquire) {
            return Err(anyhow!("injected mark-prefill failure"));
        }
        self.scheduler
            .mark_prefill_completed(&uuid.to_string())
            .await
            .map_err(anyhow::Error::from)
    }

    async fn free(&self, uuid: Uuid) -> Result<()> {
        #[cfg(test)]
        if self.fail_free.load(Ordering::Acquire) {
            return Err(anyhow!("injected free failure"));
        }
        self.scheduler
            .free(&uuid.to_string())
            .await
            .map_err(anyhow::Error::from)
    }

    async fn shutdown(&self) -> Result<()> {
        self.scheduler_cancel.cancel();
        self.event_tx
            .lock()
            .map_err(|_| anyhow!("replay router event channel lock poisoned"))?
            .take();
        // Scoped so the guard is dropped before the `await` below.
        let event_task = {
            self.event_task
                .lock()
                .map_err(|_| anyhow!("replay router event task lock poisoned"))?
                .take()
        };
        let Some(event_task) = event_task else {
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

    #[cfg(test)]
    fn debug_pending_count(&self) -> usize {
        self.scheduler.pending_count()
    }

    #[cfg(test)]
    fn fail_mark_prefill(&self) {
        self.fail_mark_prefill.store(true, Ordering::Release);
    }

    #[cfg(test)]
    fn fail_free(&self) {
        self.fail_free.store(true, Ordering::Release);
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
            ReplayRouterMode::RoundRobin => Self::RoundRobin(RoundRobinRouter::new(num_workers)),
            ReplayRouterMode::KvRouter => Self::Kv(KvReplayRouter::new(
                args,
                router_config,
                prefill_load_estimator,
                num_workers,
            )?),
        })
    }

    pub(crate) fn sink(&self, worker_id: WorkerId) -> Result<KvEventPublishers> {
        Ok(match self {
            Self::RoundRobin(_) => KvEventPublishers::default(),
            Self::Kv(router) => KvEventPublishers::new(Some(router.sink(worker_id)?), None),
        })
    }

    pub(crate) async fn select_worker(
        &self,
        request: &DirectRequest,
        num_workers: usize,
        dp_size: usize,
    ) -> Result<ReplayPlacement> {
        anyhow::ensure!(
            num_workers > 0,
            "online replay requires at least one worker"
        );
        anyhow::ensure!(dp_size > 0, "online replay requires at least one DP rank");
        match self {
            Self::RoundRobin(router) => {
                router.select_worker(num_workers, dp_size, request.preferred_dp_rank())
            }
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

    /// Number of requests the KV scheduler currently has queued.
    ///
    /// Tests that need one request to be *in the queue* before submitting the next
    /// must poll this rather than counting `yield_now()` calls: `select_worker`
    /// awaits an indexer barrier before it ever reaches the scheduler, so the number
    /// of yields it takes to get queued is not a stable property.
    #[cfg(test)]
    pub(crate) fn debug_pending_count(&self) -> usize {
        match self {
            Self::RoundRobin(_) => 0,
            Self::Kv(router) => router.debug_pending_count(),
        }
    }

    #[cfg(test)]
    pub(crate) fn fail_mark_prefill(&self) {
        if let Self::Kv(router) = self {
            router.fail_mark_prefill();
        }
    }

    #[cfg(test)]
    pub(crate) fn fail_free(&self) {
        if let Self::Kv(router) = self {
            router.fail_free();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Duration;

    use tempfile::NamedTempFile;

    use dynamo_kv_router::config::RouterQueuePolicy;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, StorageTier, WorkerWithDpRank,
        compute_block_hash_for_seq,
    };

    use super::*;

    #[test]
    fn round_robin_cycles_logical_workers_before_attention_dp_ranks() {
        let router = RoundRobinRouter::new(2);
        let placements = (0..6)
            .map(|_| router.select_worker(2, 2, None).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            placements,
            vec![
                ReplayPlacement {
                    worker_idx: 0,
                    dp_rank: 0,
                },
                ReplayPlacement {
                    worker_idx: 1,
                    dp_rank: 0,
                },
                ReplayPlacement {
                    worker_idx: 0,
                    dp_rank: 1,
                },
                ReplayPlacement {
                    worker_idx: 1,
                    dp_rank: 1,
                },
                ReplayPlacement {
                    worker_idx: 0,
                    dp_rank: 0,
                },
                ReplayPlacement {
                    worker_idx: 1,
                    dp_rank: 0,
                },
            ]
        );
    }

    #[test]
    fn round_robin_honors_preferred_attention_dp_rank() {
        let router = RoundRobinRouter::new(2);

        assert_eq!(
            router.select_worker(2, 2, Some(1)).unwrap(),
            ReplayPlacement {
                worker_idx: 0,
                dp_rank: 1,
            }
        );
        assert_eq!(
            router.select_worker(2, 2, Some(1)).unwrap(),
            ReplayPlacement {
                worker_idx: 1,
                dp_rank: 1,
            }
        );
        assert!(
            router
                .select_worker(2, 2, Some(2))
                .unwrap_err()
                .to_string()
                .contains("out of range")
        );
    }

    fn priority_request(uuid: u128, priority: i32, strict_priority: u32) -> DirectRequest {
        DirectRequest {
            tokens: vec![uuid as u32; 64],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            preferred_dp_rank: None,
            preferred_prefill_dp_rank: None,
            arrival_timestamp_ms: Some(0.0),
            priority,
            strict_priority,
            policy_class: None,
            replay_context: None,
        }
    }

    #[tokio::test]
    async fn kv_router_preserves_selected_attention_dp_rank() {
        let mut args = MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(64))
            .build()
            .unwrap();
        args.dp_size = 2;
        let router = ReplayRouter::new(ReplayRouterMode::KvRouter, &args, None, None, 1).unwrap();
        let request = priority_request(100, 0, 0);
        let tokens_hash =
            compute_block_hash_for_seq(&request.tokens, 64, BlockHashOptions::default())[0];
        let ReplayRouter::Kv(kv_router) = &router else {
            unreachable!("test constructed a KV replay router")
        };
        kv_router
            .indexer
            .apply_event(store_event(0, 1, 1, tokens_hash, StorageTier::Device))
            .await;
        kv_router.indexer.flush().await;

        assert_eq!(
            router.select_worker(&request, 1, 2).await.unwrap(),
            ReplayPlacement {
                worker_idx: 0,
                dp_rank: 1,
            }
        );

        router.on_complete(request.uuid.unwrap()).await.unwrap();
        router.shutdown().await.unwrap();
    }

    /// Wait until the KV scheduler has exactly `expected` requests queued.
    ///
    /// Mirrors kv-router's own `wait_for_pending_count` helper. Polling a real
    /// observable is required here: `select_worker` awaits an indexer barrier before
    /// reaching the scheduler, so "one `yield_now` gets a spawned task queued" is not
    /// a property the runtime guarantees.
    async fn wait_for_pending_count(router: &ReplayRouter, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while router.debug_pending_count() != expected {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "scheduler never reached {expected} pending request(s); saw {}",
                router.debug_pending_count()
            )
        });
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
        router.select_worker(&active, 1, 1).await.unwrap();

        let (completed_tx, mut completed_rx) = mpsc::unbounded_channel();
        let low_task = {
            let router = Arc::clone(&router);
            let completed_tx = completed_tx.clone();
            tokio::spawn(async move {
                let request = priority_request(2, 1_000, 0);
                router.select_worker(&request, 1, 1).await.unwrap();
                completed_tx.send(2).unwrap();
            })
        };
        wait_for_pending_count(&router, 1).await;
        let high_task = {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let request = priority_request(3, 0, 1);
                router.select_worker(&request, 1, 1).await.unwrap();
                completed_tx.send(3).unwrap();
            })
        };
        wait_for_pending_count(&router, 2).await;

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
    async fn free_clears_prefill_load_without_first_token() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(64))
            .build()
            .unwrap();
        let router = ReplayRouter::new(
            ReplayRouterMode::KvRouter,
            &args,
            Some(KvRouterConfig {
                router_track_prefill_tokens: true,
                ..KvRouterConfig::default()
            }),
            None,
            1,
        )
        .unwrap();
        let uuid = Uuid::from_u128(4);

        router
            .select_worker(&priority_request(4, 0, 0), 1, 1)
            .await
            .unwrap();
        assert_eq!(
            router.debug_potential_loads(0, true)[0].potential_prefill_tokens,
            64
        );

        router.on_complete(uuid).await.unwrap();

        assert_eq!(
            router.debug_potential_loads(0, true)[0].potential_prefill_tokens,
            0
        );
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

        // NamedTempFile, not a hand-rolled temp_dir path plus a `remove_file`
        // call: every `unwrap` between the write and the removal is a panic
        // that would leak the file. The guard is held to the end of the test
        // so removal is tied to scope exit, not to reaching a statement.
        let mut policy_file = NamedTempFile::new().unwrap();
        policy_file
            .write_all(
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
"#
                .as_bytes(),
            )
            .unwrap();
        let router = Arc::new(
            ReplayRouter::new(
                ReplayRouterMode::KvRouter,
                &args,
                Some(KvRouterConfig {
                    router_policy_config: Some(policy_file.path().display().to_string()),
                    ..KvRouterConfig::default()
                }),
                None,
                1,
            )
            .unwrap(),
        );

        let mut active = priority_request(10, 0, 0);
        active.policy_class = Some("latency".to_string());
        router.select_worker(&active, 1, 1).await.unwrap();

        let queued_task = {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let mut queued = priority_request(11, 0, 0);
                queued.policy_class = Some("latency".to_string());
                router.select_worker(&queued, 1, 1).await.unwrap()
            })
        };
        // Poll the real observable rather than counting `yield_now()`s -- the
        // same reason `wait_for_pending_count` exists and the same warning
        // `debug_pending_count`'s doc carries: `select_worker` awaits an indexer
        // barrier before it ever reaches the scheduler, so "one yield gets the
        // spawned task queued" is not a property the runtime guarantees. A
        // single `yield_now` that lost that race would make the assertion below
        // pass for the wrong reason -- the task would be unfinished because it
        // had not yet been queued, not because its class is busy.
        wait_for_pending_count(&router, 1).await;

        let mut batch = priority_request(12, 0, 0);
        batch.policy_class = Some("batch".to_string());
        assert_eq!(
            router.select_worker(&batch, 1, 1).await.unwrap(),
            ReplayPlacement {
                worker_idx: 0,
                dp_rank: 0,
            }
        );
        assert!(
            !queued_task.is_finished(),
            "latency request should remain queued while its class is busy"
        );

        router.on_complete(Uuid::from_u128(10)).await.unwrap();
        router.on_complete(Uuid::from_u128(12)).await.unwrap();
        assert_eq!(
            queued_task.await.unwrap(),
            ReplayPlacement {
                worker_idx: 0,
                dp_rank: 0,
            }
        );
        router.on_complete(Uuid::from_u128(11)).await.unwrap();
        router.shutdown().await.unwrap();
    }

    /// The mirror of `dropping_a_session_terminates_the_admission_forwarder`
    /// (round-13 H8) for the router's own KV event task, which that fix did not
    /// cover. `shutdown()` is only reached on `LiveRunSession::finish`'s happy
    /// path; every `bail!` above it drops the router instead, and a bare
    /// `JoinHandle` detaches rather than stops.
    #[tokio::test]
    async fn dropping_the_router_terminates_its_kv_event_task() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(64))
            .build()
            .unwrap();
        let router = ReplayRouter::new(ReplayRouterMode::KvRouter, &args, None, None, 1).unwrap();
        // Held for the duration so the event task never observes a closed channel
        // and can only stop by being aborted -- exactly the case where a detached
        // task would outlive the run.
        let _publisher = router.sink(0).unwrap();

        let ReplayRouter::Kv(kv_router) = &router else {
            unreachable!("test constructed a KV replay router")
        };
        let event_task = kv_router
            .event_task
            .lock()
            .unwrap()
            .as_ref()
            .expect("a fresh router owns its event task")
            .abort_handle();

        tokio::task::yield_now().await;
        assert!(
            !event_task.is_finished(),
            "the event task must still be parked on recv() before the router drops"
        );

        drop(router);
        // Abort is asynchronous: the runtime has to reach the task to cancel it.
        for _ in 0..100 {
            if event_task.is_finished() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("dropping the router left its KV event task running");
    }

    /// `sink` used to `expect` an invariant nothing enforces ("the channel
    /// exists while the runtime is active"), so asking a shut-down router for a
    /// publisher aborted the process. It is a `pub(crate)` accessor with no
    /// state machine gating it -- it must report, not panic.
    #[tokio::test]
    async fn sink_after_shutdown_reports_an_error_instead_of_panicking() {
        let args = MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(64))
            .build()
            .unwrap();
        let router = ReplayRouter::new(ReplayRouterMode::KvRouter, &args, None, None, 1).unwrap();
        assert!(router.sink(0).is_ok(), "a live router hands out publishers");

        router.shutdown().await.unwrap();

        let error = match router.sink(0) {
            Ok(_) => panic!("a shut-down router must not hand out a publisher"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("already shut down"), "{error}");
    }

    fn store_event(
        worker_id: WorkerId,
        event_id: u64,
        dp_rank: u32,
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
                dp_rank,
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
            .apply_event(store_event(7, 1, 0, tokens_hash, StorageTier::HostPinned))
            .await;
        indexer.flush().await;
        let matches = indexer
            .find_matches_for_request(&tokens, None)
            .await
            .unwrap();
        assert_eq!(matches.scores.get(&worker), None);

        indexer
            .apply_event(store_event(7, 2, 0, tokens_hash, StorageTier::Device))
            .await;
        indexer.flush().await;
        let matches = indexer
            .find_matches_for_request(&tokens, None)
            .await
            .unwrap();
        assert_eq!(matches.scores.get(&worker), Some(&1));
    }
}

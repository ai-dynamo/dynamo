// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow};
use dynamo_kv_events::event::RouterEvent;
use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_hashing::LocalBlockHash;
use dynamo_kv_hashing::routing::{BlockHashOptions, compute_block_hash_for_seq};
use dynamo_kv_indexer::host::{
    ConcurrentRadixTreeCompressed, KvIndexer, KvIndexerInterface, KvIndexerMetrics,
    LowerTierIndexers, LowerTierQueryOptions, MatchDetails, ThreadPoolIndexer, TieredMatchDetails,
    query_lower_tiers_with_options,
};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::tracking_hash::tracking_hash_context;
use dynamo_router_protocol::identity::RoutingPartitionRef;
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerId;
use dynamo_scheduler::host::{
    CacheHitWeights, OverlapAnalysis, RoutingConstraints, ScheduleMode, ScheduleRequest,
    TrackingHashContext, TrackingHashScope,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, KvCacheEventSink, KvEventPublishers, MockerConfig, RawKvEvent,
};
use crate::loadgen::{ReplayRequestHashes, local_block_hashes};
use crate::replay::router_shared::{
    ReplayScheduler, replay_router_config, replay_selector, replay_slots,
    replay_workers_with_configs,
};
use crate::replay::{ReplayPrefillLoadEstimator, ReplayRouterMode};

#[derive(Clone)]
enum PrimaryIndexer {
    Single(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
}

/// Device (G1) index plus lower-tier (for example G2 `HostPinned`) residency.
#[derive(Clone)]
struct ReplayIndexer {
    primary: PrimaryIndexer,
    block_size: u32,
    lower_tier: LowerTierIndexers,
}

impl ReplayIndexer {
    async fn apply_event(&self, event: RouterEvent) {
        if !event.storage_tier.is_gpu() {
            let indexer = self.lower_tier.get_or_create(event.storage_tier);
            if let Err(error) = indexer.apply_event_and_wait(event).await {
                tracing::warn!(%error, "failed to apply replay lower-tier KV event");
            }
            return;
        }
        match &self.primary {
            PrimaryIndexer::Single(indexer) => indexer.apply_event(event).await,
            PrimaryIndexer::Concurrent(indexer) => indexer.apply_event(event).await,
        }
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<TieredMatchDetails> {
        // Until lower-tier residency arrives, the cheaper score-only lookup
        // gives the same device overlap.
        if self.lower_tier.all().is_empty() {
            let overlap_scores = match &self.primary {
                PrimaryIndexer::Single(indexer) => {
                    indexer
                        .find_matches_for_request(tokens, lora_name, None, None)
                        .await?
                }
                PrimaryIndexer::Concurrent(indexer) => {
                    indexer
                        .find_matches_for_request(tokens, lora_name, None, None)
                        .await?
                }
            };
            return Ok(TieredMatchDetails {
                device: MatchDetails {
                    overlap_scores,
                    ..MatchDetails::default()
                },
                lower_tier: Default::default(),
            });
        }
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                lora_name,
                ..Default::default()
            },
        );
        self.find_tiered_matches(sequence).await
    }

    /// Look up block hashes the caller already derived, for a request whose
    /// identities are not Dynamo's own token hashes.
    async fn find_matches_for_hashes(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails> {
        if self.lower_tier.all().is_empty() {
            let overlap_scores = match &self.primary {
                PrimaryIndexer::Single(indexer) => indexer.find_matches(sequence).await?,
                PrimaryIndexer::Concurrent(indexer) => indexer.find_matches(sequence).await?,
            };
            return Ok(TieredMatchDetails {
                device: MatchDetails {
                    overlap_scores,
                    ..MatchDetails::default()
                },
                lower_tier: Default::default(),
            });
        }
        self.find_tiered_matches(sequence).await
    }

    async fn find_tiered_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails> {
        let device = match &self.primary {
            PrimaryIndexer::Single(indexer) => indexer.find_match_details(sequence.clone()).await?,
            PrimaryIndexer::Concurrent(indexer) => indexer
                .backend()
                .find_match_details_impl_with_options(&sequence, false, false),
        };
        let lower_tier = query_lower_tiers_with_options(
            &self.lower_tier,
            &sequence,
            &device,
            LowerTierQueryOptions::default(),
        );
        Ok(TieredMatchDetails { device, lower_tier })
    }

    async fn flush(&self) -> usize {
        match &self.primary {
            PrimaryIndexer::Single(indexer) => indexer.flush().await,
            PrimaryIndexer::Concurrent(indexer) => {
                KvIndexerInterface::flush(indexer.as_ref()).await
            }
        }
    }
}

fn create_replay_indexer(block_size: u32, num_threads: usize) -> ReplayIndexer {
    let primary = if num_threads > 1 {
        PrimaryIndexer::Concurrent(Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            num_threads,
            block_size,
        )))
    } else {
        PrimaryIndexer::Single(KvIndexer::new_with_pruning(
            CancellationToken::new(),
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            None,
        ))
    };
    ReplayIndexer {
        primary,
        block_size,
        lower_tier: LowerTierIndexers::new(1, block_size),
    }
}

#[derive(Clone)]
struct ReplayKvEventSink {
    worker_id: WorkerId,
    event_tx: mpsc::UnboundedSender<RouterEvent>,
}

impl KvCacheEventSink for ReplayKvEventSink {
    fn publish_batch_with_metadata(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        for event in events {
            let mut routed =
                RouterEvent::with_storage_tier(self.worker_id, event.event, event.storage_tier);
            routed.session_id = event.session_id;
            self.event_tx
                .send(routed)
                .map_err(|_| anyhow!("replay router event channel closed"))?;
        }
        Ok(())
    }

    fn publish(&self, event: dynamo_kv_events::event::KvCacheEvent) -> anyhow::Result<()> {
        self.event_tx
            .send(RouterEvent::new(self.worker_id, event))
            .map_err(|_| anyhow!("replay router event channel closed"))
    }

    fn publish_with_storage_tier(
        &self,
        event: dynamo_kv_events::event::KvCacheEvent,
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
    event_tx: Mutex<Option<mpsc::UnboundedSender<RouterEvent>>>,
    event_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    indexer: ReplayIndexer,
    tracking_hash: TrackingHashContext,
    #[cfg(test)]
    fail_mark_prefill: AtomicBool,
    #[cfg(test)]
    fail_free: AtomicBool,
}

impl KvReplayRouter {
    fn new(
        args: &MockerConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        let config = replay_router_config(args, router_config);
        let tracking_hash = tracking_hash_context(&config)?;
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
        let scheduler = Arc::new(dynamo_scheduler::host::LocalScheduler::new(
            slots,
            worker_config_rx,
            profile,
            args.block_size as u32,
            selector,
            prefill_load_estimator,
            None,
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
            tracking_hash,
            #[cfg(test)]
            fail_mark_prefill: AtomicBool::new(false),
            #[cfg(test)]
            fail_free: AtomicBool::new(false),
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

    /// The request's cached-prefix matches and its active-block tracking
    /// identities, both in the request's cache namespace.
    async fn request_identities(
        &self,
        request: &DirectRequest,
    ) -> Result<(TieredMatchDetails, Option<Vec<u64>>)> {
        // The engine hashes a salted prompt in AISim's cache namespace and its
        // KV events carry those hashes, so derive the request's by the same
        // contract: the block seed behind `BlockHashOptions::cache_namespace`
        // is a different one.
        let cache_salt = request
            .cache_salt
            .as_deref()
            .filter(|salt| !salt.is_empty());
        let salted_hashes = cache_salt.map(|cache_salt| {
            local_block_hashes(
                ReplayRequestHashes::from_tokens_with_cache_salt(
                    &request.tokens,
                    self.block_size,
                    Some(cache_salt),
                )
                .local_block_hashes,
            )
        });
        // Public tracking chains the salted block hashes, as the engine does.
        // Keyed identities never meet engine hashes; their namespace only
        // keeps one salt's active blocks apart from another's.
        let token_seq = self.config.compute_seq_hashes_for_tracking_with_context(
            &self.tracking_hash,
            TrackingHashScope {
                partition: RoutingPartitionRef::new("replay", "default"),
                block_size: self.block_size,
            },
            &request.tokens,
            None,
            BlockHashOptions {
                cache_namespace: cache_salt,
                ..Default::default()
            },
            salted_hashes.as_deref(),
        );
        let tiered = match salted_hashes {
            Some(hashes) => self.indexer.find_matches_for_hashes(hashes).await?,
            None => {
                self.indexer
                    .find_matches_for_request(&request.tokens, None)
                    .await?
            }
        };
        Ok((tiered, token_seq))
    }

    async fn select_worker(&self, request: &DirectRequest) -> Result<ReplayPlacement> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("online replay requires requests to have stable UUIDs"))?;
        let (tiered, token_seq) = self.request_identities(request).await?;
        let overlap = OverlapAnalysis::new(
            &CacheHitWeights::from(&self.config),
            self.block_size,
            &tiered,
        )
        .signals();
        let (priority_jump, strict_priority) = request.router_priorities();
        let response = self
            .scheduler
            .schedule_request(ScheduleRequest {
                accounting_attribution: Default::default(),
                mode: ScheduleMode::Tracked {
                    request_id: uuid.to_string(),
                },
                token_seq,
                block_hashes: None,
                isl_tokens: request.tokens.len(),
                overlap,
                kv_transfer_candidates: None,
                retain_kv_transfer_chain: false,
                routing_constraints: RoutingConstraints::default(),
                router_config_override: None,
                lora_name: None,
                priority_jump,
                strict_priority,
                policy_class: request.policy_class.clone(),
                session_context: request
                    .replay_context
                    .as_ref()
                    .and_then(|context| context.session_id())
                    .map(|id| SessionContext::new(id.to_owned(), None, None, None)),
                expected_output_tokens: Some(
                    u32::try_from(request.max_output_tokens)
                        .context("max_output_tokens does not fit into u32")?,
                ),
                affinity_target: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                shared_cache_hits: None,
            })
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
    ) -> Vec<dynamo_scheduler::host::PotentialLoad> {
        self.scheduler.get_potential_loads(
            None,
            isl_tokens,
            Default::default(),
            track_prefill_tokens,
        )
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
        args: &MockerConfig,
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
        dp_size: usize,
    ) -> Result<ReplayPlacement> {
        anyhow::ensure!(
            request.kv_hints.is_none(),
            "Dynamo online replay does not support authored KV fetch hints"
        );
        anyhow::ensure!(
            request.images.is_empty(),
            "Dynamo online replay does not support image requests"
        );
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
    ) -> Vec<dynamo_scheduler::host::PotentialLoad> {
        match self {
            Self::RoundRobin(_) => Vec::new(),
            Self::Kv(router) => router.debug_potential_loads(isl_tokens, track_prefill_tokens),
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
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_admission::host::RouterQueuePolicy;
    use dynamo_kv_events::event::{
        KvCacheEvent, KvCacheEventData, KvCacheStoreData, KvCacheStoredBlockData,
    };
    use dynamo_kv_events::residency::StorageTier;
    use dynamo_kv_hashing::routing::compute_block_hash_for_seq;
    use dynamo_kv_hashing::{ExternalSequenceBlockHash, LocalBlockHash};
    use dynamo_router_protocol::worker::WorkerWithDpRank;
    use dynamo_scheduler::host::TrackingHashAlgorithm;
    use std::io::Write;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::common::protocols::EngineType;
    use crate::live::LiveEngine;
    use crate::replay::online::live_runtime::LiveRuntime;
    use crate::replay::online::state::LiveReplayMode;
    use crate::replay::online::{OnlineReplayConfig, OnlineReplayOptions};

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
            kv_hints: None,
            arrival_timestamp_ms: Some(0.0),
            priority,
            strict_priority,
            policy_class: None,
            replay_context: None,
            images: Vec::new(),
            cache_salt: None,
        }
    }

    #[tokio::test]
    async fn online_replay_rejects_unhandled_request_fields_before_placement() {
        for mode in [ReplayRouterMode::RoundRobin, ReplayRouterMode::KvRouter] {
            let router = ReplayRouter::new(mode, &MockerConfig::default(), None, None, 1).unwrap();
            let mut request = priority_request(1, 0, 0);
            request.kv_hints = Some(aisim_core::replay::AuthoredKvHints {
                source_worker: 0,
                source_dp_rank: 0,
            });
            let error = router.select_worker(&request, 1, 1).await.unwrap_err();
            assert!(error.to_string().contains("authored KV fetch hints"));
            request.kv_hints = None;
            request.images.push(aisim_core::engine::ImageSpec {
                identity: 1,
                token_start: 0,
                token_end: 1,
                encoder: aisim_core::engine::EncoderShape {
                    sequences: 1,
                    patch_tokens: 1,
                    transformer_tokens: 1,
                    output_tokens: 1,
                },
                embedding_bytes: 1,
            });
            let error = router.select_worker(&request, 1, 1).await.unwrap_err();
            assert!(error.to_string().contains("image requests"));
            request.images.clear();
            assert_eq!(
                router.select_worker(&request, 1, 1).await.unwrap(),
                ReplayPlacement {
                    worker_idx: 0,
                    dp_rank: 0
                }
            );
            router.on_complete(request.uuid.unwrap()).await.unwrap();
            router.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn kv_router_preserves_selected_attention_dp_rank() {
        let mut args = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64,
                "max_num_batched_tokens": 64
            }
        }))
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

    #[tokio::test]
    async fn online_replay_forwards_priorities_to_scheduler_queue() {
        let args = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64,
                "max_num_batched_tokens": 64
            }
        }))
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
        tokio::task::yield_now().await;
        let high_task = {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let request = priority_request(3, 0, 1);
                router.select_worker(&request, 1, 1).await.unwrap();
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
    async fn free_clears_prefill_load_without_first_token() {
        let args = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64,
                "max_num_batched_tokens": 64
            }
        }))
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
        let args = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64,
                "max_num_batched_tokens": 64
            }
        }))
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
default_policy_class: latency
policy_classes:
  - name: latency
    quantum: 1
    prefill_busy_threshold: 0
  - name: batch
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
        router.select_worker(&active, 1, 1).await.unwrap();

        let queued_task = {
            let router = Arc::clone(&router);
            tokio::spawn(async move {
                let mut queued = priority_request(11, 0, 0);
                queued.policy_class = Some("latency".to_string());
                router.select_worker(&queued, 1, 1).await.unwrap()
            })
        };
        tokio::task::yield_now().await;

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
    async fn online_kv_router_continues_a_device_match_into_host_pinned() {
        let args = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64
            }
        }))
        .unwrap();
        let router = ReplayRouter::new(ReplayRouterMode::KvRouter, &args, None, None, 2).unwrap();
        let mut request = priority_request(100, 0, 0);
        request.tokens = vec![100; 256];
        let hashes = compute_block_hash_for_seq(&request.tokens, 64, BlockHashOptions::default());
        let ReplayRouter::Kv(kv_router) = &router else {
            unreachable!("test constructed a KV replay router")
        };
        // Worker 1 holds block 0 in G1 and blocks 1..4 in G2, chained under it.
        let store = |event_id, first: usize, tier| {
            let block_hash = |index: usize| ExternalSequenceBlockHash(1_000 + index as u64);
            RouterEvent::with_storage_tier(
                1,
                KvCacheEvent {
                    event_id,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: first.checked_sub(1).map(block_hash),
                        start_position: None,
                        blocks: (first..if first == 0 { 1 } else { 4 })
                            .map(|index| KvCacheStoredBlockData {
                                block_hash: block_hash(index),
                                tokens_hash: hashes[index],
                                mm_extra_info: None,
                            })
                            .collect(),
                    }),
                    dp_rank: 0,
                },
                tier,
            )
        };
        kv_router
            .indexer
            .apply_event(store(1, 0, StorageTier::Device))
            .await;
        kv_router
            .indexer
            .apply_event(store(2, 1, StorageTier::HostPinned))
            .await;
        kv_router.indexer.flush().await;

        let tiered = kv_router
            .indexer
            .find_matches_for_request(&request.tokens, None)
            .await
            .unwrap();
        let worker = WorkerWithDpRank::new(1, 0);
        assert_eq!(tiered.device.overlap_scores.scores.get(&worker), Some(&1));
        assert_eq!(
            tiered.lower_tier[&StorageTier::HostPinned]
                .hits
                .get(&worker),
            Some(&3)
        );
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn online_kv_router_routes_to_a_host_pinned_prefix() {
        let args = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64
            }
        }))
        .unwrap();
        let router = ReplayRouter::new(ReplayRouterMode::KvRouter, &args, None, None, 2).unwrap();
        let request = priority_request(100, 0, 0);
        let tokens_hash =
            compute_block_hash_for_seq(&request.tokens, 64, BlockHashOptions::default())[0];
        let ReplayRouter::Kv(kv_router) = &router else {
            unreachable!("test constructed a KV replay router")
        };
        kv_router
            .indexer
            .apply_event(store_event(1, 1, 0, tokens_hash, StorageTier::HostPinned))
            .await;
        kv_router.indexer.flush().await;
        let tiered = kv_router
            .indexer
            .find_matches_for_request(&request.tokens, None)
            .await
            .unwrap();
        assert!(tiered.device.overlap_scores.scores.is_empty());
        assert_eq!(
            tiered.lower_tier[&StorageTier::HostPinned]
                .hits
                .get(&WorkerWithDpRank::new(1, 0)),
            Some(&1)
        );

        assert_eq!(
            router.select_worker(&request, 2, 1).await.unwrap(),
            ReplayPlacement {
                worker_idx: 1,
                dp_rank: 0,
            }
        );
        router.on_complete(request.uuid.unwrap()).await.unwrap();
        router.shutdown().await.unwrap();
    }

    const CACHE_SALT_BLOCK_SIZE: u32 = 4;

    fn cache_salt_args() -> MockerConfig {
        MockerConfig::from_value(serde_json::json!({
            "engine": {
                "backend": EngineType::Sglang,
                "block_size": CACHE_SALT_BLOCK_SIZE,
                "num_gpu_blocks": 512,
                "speedup_ratio": 1000.0
            }
        }))
        .unwrap()
    }

    /// One 10-token prompt (two complete 4-token blocks) in a cache namespace.
    fn cache_salt_request(uuid: u128, cache_salt: Option<&str>) -> DirectRequest {
        DirectRequest {
            tokens: (1..=10).collect(),
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(uuid)),
            arrival_timestamp_ms: Some(0.0),
            cache_salt: cache_salt.map(str::to_owned),
            ..Default::default()
        }
    }

    /// Run `request` on a grouped live engine until its completion effects are
    /// published. Returns the prompt tokens the engine reused.
    async fn run_on_engine(engine: &LiveEngine, request: DirectRequest) -> usize {
        let mut live = engine.submit(request).await.unwrap();
        let mut cached_tokens = None;
        loop {
            let output = live.recv().await.expect("terminal output");
            assert!(!output.rejected);
            cached_tokens = cached_tokens.or(output.cached_tokens);
            if output.completed {
                break;
            }
        }
        engine.drain_completion_boundary().await.unwrap();
        cached_tokens.expect("the engine reports its cache usage")
    }

    /// Wait until the engine's KV events for the prompt in one cache namespace
    /// reach the replay indexer. The probe names the published hashes itself.
    async fn wait_until_indexed(
        router: &KvReplayRouter,
        cache_salt: Option<&str>,
        worker_id: WorkerId,
    ) {
        let hashes = local_block_hashes(
            ReplayRequestHashes::from_tokens_with_cache_salt(
                &cache_salt_request(0, cache_salt).tokens,
                CACHE_SALT_BLOCK_SIZE,
                cache_salt,
            )
            .local_block_hashes,
        );
        let worker = WorkerWithDpRank::new(worker_id, 0);
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let tiered = router
                    .indexer
                    .find_matches_for_hashes(hashes.clone())
                    .await
                    .unwrap();
                if tiered.device.overlap_scores.scores.get(&worker) == Some(&2) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the engine's KV events reach the replay indexer");
    }

    /// Device overlap in blocks, per worker, that routing sees for the prompt
    /// in one cache namespace.
    async fn routed_overlap(router: &KvReplayRouter, cache_salt: Option<&str>) -> Vec<(u64, u32)> {
        let (tiered, _) = router
            .request_identities(&cache_salt_request(0, cache_salt))
            .await
            .unwrap();
        let mut overlap = tiered
            .device
            .overlap_scores
            .scores
            .iter()
            .map(|(worker, blocks)| (worker.worker_id, *blocks))
            .collect::<Vec<_>>();
        overlap.sort_unstable();
        overlap
    }

    /// The SGLang engine hashes a salted prompt in its own cache namespace and
    /// publishes those hashes; routing has to look the request up in the same
    /// one, whichever device indexer the event thread count selects.
    #[rstest::rstest]
    #[case::single_indexer(1)]
    #[case::concurrent_indexer(4)]
    #[tokio::test]
    async fn online_kv_router_sees_engine_cache_salt_namespaces(#[case] router_event_threads: u32) {
        let runtime = LiveRuntime::new(
            OnlineReplayConfig::new(
                cache_salt_args(),
                Some(KvRouterConfig {
                    router_event_threads,
                    ..KvRouterConfig::default()
                }),
                None,
                2,
                ReplayRouterMode::KvRouter,
                OnlineReplayOptions::default(),
            ),
            Default::default(),
            LiveReplayMode::Trace,
            CancellationToken::new(),
        )
        .unwrap();
        let engines = runtime.engines();
        let router = runtime.router();
        let ReplayRouter::Kv(kv_router) = router.as_ref() else {
            unreachable!("test constructed a KV replay router")
        };
        assert_eq!(
            matches!(kv_router.indexer.primary, PrimaryIndexer::Single(_)),
            router_event_threads == 1
        );

        // Warm worker 1 under salt "a" and worker 0 without a salt, on the
        // engines whose KV events feed this router.
        assert_eq!(
            run_on_engine(&engines[1], cache_salt_request(1, Some("a"))).await,
            0
        );
        assert_eq!(
            run_on_engine(&engines[0], cache_salt_request(2, None)).await,
            0
        );
        wait_until_indexed(kv_router, Some("a"), 1).await;
        wait_until_indexed(kv_router, None, 0).await;

        assert_eq!(routed_overlap(kv_router, Some("a")).await, [(1, 2)]);
        assert_eq!(routed_overlap(kv_router, Some("b")).await, []);
        assert_eq!(routed_overlap(kv_router, None).await, [(0, 2)]);
        // An empty salt is the unsalted namespace, as in the engine.
        assert_eq!(routed_overlap(kv_router, Some("")).await, [(0, 2)]);

        // Placement follows the warm worker of each namespace.
        for (uuid, cache_salt, worker_idx) in [(3, Some("a"), 1), (4, None, 0)] {
            let request = cache_salt_request(uuid, cache_salt);
            assert_eq!(
                router.select_worker(&request, 2, 1).await.unwrap(),
                ReplayPlacement {
                    worker_idx,
                    dp_rank: 0,
                },
                "{cache_salt:?}"
            );
            router.on_complete(request.uuid.unwrap()).await.unwrap();
        }

        // The engines agree with what routing saw: only the same salt reuses.
        assert_eq!(
            run_on_engine(&engines[1], cache_salt_request(5, Some("a"))).await,
            8
        );
        assert_eq!(
            run_on_engine(&engines[1], cache_salt_request(6, Some("b"))).await,
            0
        );
        assert_eq!(
            run_on_engine(&engines[0], cache_salt_request(7, None)).await,
            8
        );

        for engine in engines.iter() {
            engine.shutdown().await.unwrap();
        }
        router.shutdown().await.unwrap();
    }

    /// A salted prefix in a lower tier takes the detailed lookup; it matches
    /// by the same published hashes.
    #[tokio::test]
    async fn online_kv_router_finds_a_cache_salt_prefix_in_host_pinned() {
        let router = ReplayRouter::new(
            ReplayRouterMode::KvRouter,
            &cache_salt_args(),
            None,
            None,
            2,
        )
        .unwrap();
        let ReplayRouter::Kv(kv_router) = &router else {
            unreachable!("test constructed a KV replay router")
        };
        let published = local_block_hashes(
            ReplayRequestHashes::from_tokens_with_cache_salt(
                &cache_salt_request(0, Some("a")).tokens,
                CACHE_SALT_BLOCK_SIZE,
                Some("a"),
            )
            .local_block_hashes,
        );
        kv_router
            .indexer
            .apply_event(store_event(1, 1, 0, published[0], StorageTier::HostPinned))
            .await;
        kv_router.indexer.flush().await;

        let worker = WorkerWithDpRank::new(1, 0);
        for (cache_salt, host_blocks) in [(Some("a"), Some(1)), (Some("b"), None), (None, None)] {
            let (tiered, _) = kv_router
                .request_identities(&cache_salt_request(0, cache_salt))
                .await
                .unwrap();
            assert!(tiered.device.overlap_scores.scores.is_empty());
            assert_eq!(
                tiered
                    .lower_tier
                    .get(&StorageTier::HostPinned)
                    .and_then(|matches| matches.hits.get(&worker))
                    .copied(),
                host_blocks,
                "{cache_salt:?}"
            );
        }
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn online_kv_router_tracks_active_blocks_per_cache_salt() {
        let router = ReplayRouter::new(
            ReplayRouterMode::KvRouter,
            &cache_salt_args(),
            None,
            None,
            1,
        )
        .unwrap();
        let ReplayRouter::Kv(kv_router) = &router else {
            unreachable!("test constructed a KV replay router")
        };

        // Active copies of one two-block prompt share blocks only inside a
        // namespace; an empty salt is the unsalted one.
        let namespaces = [None, None, Some("a"), Some("a"), Some("b"), Some("")];
        let mut active_blocks = Vec::new();
        for (index, cache_salt) in namespaces.into_iter().enumerate() {
            router
                .select_worker(&cache_salt_request(index as u128 + 1, cache_salt), 1, 1)
                .await
                .unwrap();
            active_blocks.push(router.debug_potential_loads(0, false)[0].potential_decode_blocks);
        }
        assert_eq!(active_blocks, [2, 2, 4, 4, 6, 6]);

        // Public tracking identities are the engine's sequence hashes.
        for cache_salt in [None, Some(""), Some("a"), Some("b")] {
            let request = cache_salt_request(0, cache_salt);
            let (_, token_seq) = kv_router.request_identities(&request).await.unwrap();
            assert_eq!(
                token_seq,
                Some(
                    ReplayRequestHashes::from_tokens_with_cache_salt(
                        &request.tokens,
                        CACHE_SALT_BLOCK_SIZE,
                        cache_salt,
                    )
                    .sequence_hashes
                ),
                "{cache_salt:?}"
            );
        }

        for index in 0..namespaces.len() {
            router
                .on_complete(Uuid::from_u128(index as u128 + 1))
                .await
                .unwrap();
        }
        router.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn online_keyed_tracking_separates_cache_salts() {
        let mut key_file = NamedTempFile::new().unwrap();
        key_file.write_all(&[0x39; 32]).unwrap();
        let config = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            router_tracking_key_file: Some(key_file.path().to_path_buf()),
            router_tracking_key_id: Some("2026-01".to_string()),
            ..KvRouterConfig::default()
        };
        let router = ReplayRouter::new(
            ReplayRouterMode::KvRouter,
            &cache_salt_args(),
            Some(config),
            None,
            1,
        )
        .unwrap();
        let ReplayRouter::Kv(kv_router) = &router else {
            unreachable!("test constructed a KV replay router")
        };
        let mut identities = Vec::new();
        for cache_salt in [None, Some(""), Some("a"), Some("a"), Some("b")] {
            let (_, token_seq) = kv_router
                .request_identities(&cache_salt_request(0, cache_salt))
                .await
                .unwrap();
            identities.push(token_seq.expect("active blocks are tracked"));
        }
        let [unsalted, empty, a, a_again, b] = identities.as_slice() else {
            unreachable!("five namespaces were hashed")
        };
        assert_eq!(unsalted.len(), 2);
        assert_eq!(unsalted, empty);
        assert_eq!(a, a_again);
        assert_ne!(a, unsalted);
        assert_ne!(a, b);
        assert_ne!(b, unsalted);
        router.shutdown().await.unwrap();
    }
}

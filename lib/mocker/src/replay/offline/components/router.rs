// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, PrefillLoadHint, RouterEvent, WorkerConfigLike, WorkerId,
    WorkerWithDpRank, compute_block_hash_for_seq,
};
use dynamo_kv_router::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, DefaultWorkerSelector, RadixTree, RouterSchedulingPolicy,
    SchedulingPolicy, SchedulingRequest, SequenceRequest, WorkerSelector,
};
use dynamo_tokens::SequenceHash;
use tokio::time::Instant;
use uuid::Uuid;

use super::{RouterEffects, WorkerAdmission};
use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::ReplayRequestHashes;
use crate::replay::ReplayPrefillLoadEstimator;
use crate::replay::router_shared::{
    ReplayNoopPublisher, ReplayWorkerConfig, replay_policy, replay_router_config, replay_selector,
    replay_slots, replay_workers_with_configs,
};

type ReplayQueueKey = <RouterSchedulingPolicy as SchedulingPolicy>::Key;

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflinePendingRequestSnapshot {
    pub(crate) uuid: Uuid,
    pub(crate) overlap_blocks_by_worker: Vec<(usize, u32)>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineIndexerSnapshot {
    pub(crate) total_cached_blocks: usize,
    pub(crate) cached_blocks_by_worker: Vec<(usize, usize)>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineRouterSnapshot {
    pub(crate) pending: Vec<OfflinePendingRequestSnapshot>,
    pub(crate) active_blocks_by_worker: Vec<(usize, usize)>,
    pub(crate) active_tokens_by_worker: Vec<(usize, usize)>,
    pub(crate) indexer: OfflineIndexerSnapshot,
}

struct SyncReplayIndexer {
    block_size: u32,
    tree: RadixTree,
}

impl SyncReplayIndexer {
    fn new(block_size: u32) -> Self {
        Self {
            block_size,
            tree: RadixTree::new(),
        }
    }

    fn find_matches_for_request(&self, tokens: &[u32], lora_name: Option<&str>) -> OverlapScores {
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                lora_name,
                ..Default::default()
            },
        );
        self.tree.find_matches(sequence, false)
    }

    fn find_matches_for_hashes(&self, local_block_hashes: Vec<LocalBlockHash>) -> OverlapScores {
        self.tree.find_matches(local_block_hashes, false)
    }

    fn apply_event(&mut self, event: RouterEvent) -> Result<()> {
        self.tree.apply_event(event).map_err(Into::into)
    }

    #[cfg(test)]
    fn debug_snapshot(&self) -> OfflineIndexerSnapshot {
        let mut blocks_by_worker = HashMap::<usize, usize>::new();
        for event in self.tree.dump_tree_as_events() {
            *blocks_by_worker
                .entry(event.worker_id as usize)
                .or_default() += 1;
        }
        let mut cached_blocks_by_worker = blocks_by_worker.into_iter().collect::<Vec<_>>();
        cached_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        OfflineIndexerSnapshot {
            total_cached_blocks: self.tree.current_size(),
            cached_blocks_by_worker,
        }
    }
}

struct PendingRequest {
    uuid: Uuid,
    token_seq: Option<Vec<SequenceHash>>,
    isl_tokens: usize,
    overlaps: OverlapScores,
    track_prefill_tokens: bool,
    expected_output_tokens: Option<u32>,
}

impl PendingRequest {
    fn request_id(&self) -> String {
        self.uuid.to_string()
    }

    fn scheduling_request(
        &self,
        decode_blocks: HashMap<WorkerWithDpRank, usize>,
        prefill_tokens: HashMap<WorkerWithDpRank, usize>,
    ) -> SchedulingRequest {
        SchedulingRequest {
            maybe_request_id: Some(self.request_id()),
            token_seq: self.token_seq.clone(),
            isl_tokens: self.isl_tokens,
            overlaps: self.overlaps.clone(),
            decode_blocks,
            prefill_tokens,
            track_prefill_tokens: self.track_prefill_tokens,
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: self.expected_output_tokens,
            pinned_worker: None,
            allowed_worker_ids: None,
            resp_tx: None,
        }
    }
}

struct QueueEntry {
    key: ReplayQueueKey,
    _enqueue_time_ms: f64,
    enqueue_seq: u64,
    request: PendingRequest,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.enqueue_seq == other.enqueue_seq
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| other.enqueue_seq.cmp(&self.enqueue_seq))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) struct OfflineReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    queue_threshold: Option<f64>,
    workers_with_configs: HashMap<WorkerId, ReplayWorkerConfig>,
    slots: Arc<ActiveSequencesMultiWorker<ReplayNoopPublisher>>,
    selector: DefaultWorkerSelector,
    policy: RouterSchedulingPolicy,
    pending: BinaryHeap<QueueEntry>,
    next_enqueue_seq: u64,
    indexer: SyncReplayIndexer,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    decay_time_epoch: Instant,
}

impl OfflineReplayRouter {
    pub(crate) fn new(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        let config = replay_router_config(args, router_config);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let selector = replay_selector(&config);
        let policy = replay_policy(&config, args);
        let queue_threshold = if num_workers > 1 {
            config.router_queue_threshold
        } else {
            None
        };

        Ok(Self {
            config,
            block_size: args.block_size as u32,
            queue_threshold,
            workers_with_configs,
            slots,
            selector,
            policy,
            pending: BinaryHeap::new(),
            next_enqueue_seq: 0,
            indexer: SyncReplayIndexer::new(args.block_size as u32),
            prefill_load_estimator,
            // This is only a base Instant for converting replay `now_ms` values into
            // synthetic `Instant`s. All subsequent decay/accounting uses virtual replay
            // time derived from this epoch, not wall-clock progression.
            decay_time_epoch: Instant::now(),
        })
    }

    pub(crate) fn on_request_arrival(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        let pending = self.build_pending_request(request, replay_hashes)?;
        let decay_now = self.decay_now(now_ms);
        let should_queue = self
            .queue_threshold
            .is_some_and(|threshold| self.all_workers_busy(threshold, decay_now));

        if should_queue {
            let key = self.enqueue_key(now_ms, &pending);
            self.pending.push(QueueEntry {
                key,
                _enqueue_time_ms: now_ms,
                enqueue_seq: self.next_enqueue_seq,
                request: pending,
            });
            self.next_enqueue_seq += 1;
            return Ok(RouterEffects::default());
        }

        Ok(RouterEffects {
            admissions: vec![WorkerAdmission {
                uuid: request
                    .uuid
                    .expect("offline replay requests must have UUIDs before router submission"),
                worker_idx: self.admit_request(pending, decay_now)?,
            }],
        })
    }

    pub(crate) fn on_kv_events(&mut self, events: Vec<RouterEvent>) -> Result<RouterEffects> {
        for event in events {
            self.indexer.apply_event(event)?;
        }
        Ok(RouterEffects::default())
    }

    pub(crate) fn on_prefill_completed(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        let decay_now = self.decay_now(now_ms);
        self.slots
            .mark_prefill_completed(&uuid.to_string(), decay_now)
            .map_err(anyhow::Error::from)?;
        Ok(RouterEffects {
            admissions: self
                .drain_pending(decay_now)?
                .into_iter()
                .map(|(uuid, worker_idx)| WorkerAdmission { uuid, worker_idx })
                .collect(),
        })
    }

    pub(crate) fn on_request_completed(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        let decay_now = self.decay_now(now_ms);
        self.slots
            .free(&uuid.to_string(), decay_now)
            .map_err(anyhow::Error::from)?;
        Ok(RouterEffects {
            admissions: self
                .drain_pending(decay_now)?
                .into_iter()
                .map(|(uuid, worker_idx)| WorkerAdmission { uuid, worker_idx })
                .collect(),
        })
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Register a new worker with the router, cloning the config from existing workers.
    pub(crate) fn add_worker(
        &mut self,
        worker_id: usize,
    ) -> Result<()> {
        let config = self
            .workers_with_configs
            .values()
            .next()
            .ok_or_else(|| anyhow!("cannot add worker to router with no existing workers"))?
            .clone();
        let wid = worker_id as WorkerId;
        self.workers_with_configs.insert(wid, config);

        // Rebuild the slots with the full worker set
        let dp_range: HashMap<u64, (u32, u32)> = self
            .workers_with_configs
            .keys()
            .map(|&id| (id, (0u32, 1u32)))
            .collect();
        self.slots.update_workers(&dp_range);

        // Enable queueing if we now have more than one worker
        if self.workers_with_configs.len() > 1 && self.queue_threshold.is_none() {
            self.queue_threshold = self.config.router_queue_threshold;
        }

        Ok(())
    }

    /// Remove a worker from the router.
    pub(crate) fn remove_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.workers_with_configs.remove(&wid);

        // Remove from the radix tree indexer
        self.indexer.tree.remove_worker(wid);

        // Rebuild slots with updated worker set
        let dp_range: HashMap<u64, (u32, u32)> = self
            .workers_with_configs
            .keys()
            .map(|&id| (id, (0u32, 1u32)))
            .collect();
        self.slots.update_workers(&dp_range);

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self, now_ms: f64) -> OfflineRouterSnapshot {
        let decay_now = self.decay_now(now_ms);
        let mut pending = self
            .pending
            .iter()
            .map(|entry| {
                let mut overlap_blocks_by_worker = entry
                    .request
                    .overlaps
                    .scores
                    .iter()
                    .map(|(worker, overlap)| (worker.worker_id as usize, *overlap))
                    .collect::<Vec<_>>();
                overlap_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

                (
                    entry,
                    OfflinePendingRequestSnapshot {
                        uuid: entry.request.uuid,
                        overlap_blocks_by_worker,
                    },
                )
            })
            .collect::<Vec<_>>();
        pending.sort_unstable_by(|(left_entry, _), (right_entry, _)| {
            left_entry.cmp(right_entry).reverse()
        });

        let mut active_blocks_by_worker = self
            .slots
            .active_blocks()
            .into_iter()
            .map(|(worker, blocks)| (worker.worker_id as usize, blocks))
            .collect::<Vec<_>>();
        active_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        let mut active_tokens_by_worker = self
            .slots
            .active_tokens(decay_now)
            .into_iter()
            .map(|(worker, tokens)| (worker.worker_id as usize, tokens))
            .collect::<Vec<_>>();
        active_tokens_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        OfflineRouterSnapshot {
            pending: pending.into_iter().map(|(_, snapshot)| snapshot).collect(),
            active_blocks_by_worker,
            active_tokens_by_worker,
            indexer: self.indexer.debug_snapshot(),
        }
    }

    fn enqueue_key(&self, now_ms: f64, request: &PendingRequest) -> ReplayQueueKey {
        let arrival_offset = Duration::from_secs_f64((now_ms.max(0.0)) / 1000.0);
        self.policy.enqueue_key(
            arrival_offset,
            &request.scheduling_request(HashMap::new(), HashMap::new()),
        )
    }

    fn decay_now(&self, now_ms: f64) -> Instant {
        self.decay_time_epoch + Duration::from_secs_f64(now_ms.max(0.0) / 1000.0)
    }

    fn build_pending_request(
        &self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
    ) -> Result<PendingRequest> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("offline replay requires requests to have stable UUIDs"))?;
        let (overlaps, token_seq) = match replay_hashes {
            Some(replay_hashes) => {
                let overlaps = self
                    .indexer
                    .find_matches_for_hashes(replay_hashes.local_block_hashes);
                let token_seq = if !self.config.router_track_active_blocks {
                    None
                } else if self.config.router_assume_kv_reuse {
                    Some(replay_hashes.sequence_hashes)
                } else {
                    self.config.compute_seq_hashes_for_tracking(
                        &request.tokens,
                        self.block_size,
                        None,
                        BlockHashOptions::default(),
                        None,
                    )
                };
                (overlaps, token_seq)
            }
            None => {
                let overlaps = self.indexer.find_matches_for_request(&request.tokens, None);
                let token_seq = self.config.compute_seq_hashes_for_tracking(
                    &request.tokens,
                    self.block_size,
                    None,
                    BlockHashOptions::default(),
                    None,
                );
                (overlaps, token_seq)
            }
        };

        Ok(PendingRequest {
            uuid,
            token_seq,
            isl_tokens: request.tokens.len(),
            overlaps,
            track_prefill_tokens: self.config.router_track_prefill_tokens,
            expected_output_tokens: Some(
                u32::try_from(request.max_output_tokens)
                    .context("max_output_tokens does not fit into u32")?,
            ),
        })
    }

    fn admit_request(&mut self, request: PendingRequest, decay_now: Instant) -> Result<usize> {
        let (decode_blocks, prefill_tokens) = self
            .slots
            .potential_blocks_and_tokens_with_prefill_tracking(
                request.token_seq.as_deref(),
                request.isl_tokens,
                request.overlaps.clone(),
                request.track_prefill_tokens,
                decay_now,
            );
        let scheduling_request = request.scheduling_request(decode_blocks, prefill_tokens);
        let selection = self.selector.select_worker(
            &self.workers_with_configs,
            &scheduling_request,
            self.block_size,
        )?;
        let worker_idx = usize::try_from(selection.worker.worker_id)
            .map_err(|_| anyhow!("selected worker id does not fit into usize"))?;
        let request_id = request.request_id();
        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selection.overlap_blocks,
            request.track_prefill_tokens,
        );

        self.slots
            .add_request(
                SequenceRequest {
                    request_id,
                    token_sequence: request.token_seq,
                    isl: request.isl_tokens,
                    overlap: selection.overlap_blocks,
                    track_prefill_tokens: request.track_prefill_tokens,
                    expected_output_tokens: request.expected_output_tokens,
                    prefill_load_hint,
                    worker: selection.worker,
                    lora_name: None,
                },
                decay_now,
            )
            .map_err(anyhow::Error::from)?;

        Ok(worker_idx)
    }

    fn drain_pending(&mut self, decay_now: Instant) -> Result<Vec<(Uuid, usize)>> {
        let Some(threshold) = self.queue_threshold else {
            return Ok(Vec::new());
        };

        let mut admissions = Vec::new();
        while !self.all_workers_busy(threshold, decay_now) {
            let Some(QueueEntry { request, .. }) = self.pending.pop() else {
                break;
            };
            let uuid = request.uuid;
            let worker_idx = self.admit_request(request, decay_now)?;
            admissions.push((uuid, worker_idx));
        }

        Ok(admissions)
    }

    fn all_workers_busy(&self, threshold: f64, decay_now: Instant) -> bool {
        let mut checked_any = false;
        let any_worker_not_busy =
            self.slots
                .any_worker_matches_active_tokens(decay_now, |worker, tokens| {
                    let Some(config) = self.workers_with_configs.get(&worker.worker_id) else {
                        return false;
                    };
                    checked_any = true;
                    let max_batched = config
                        .max_num_batched_tokens()
                        .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
                    (tokens as f64) <= threshold * (max_batched as f64)
                });

        checked_any && !any_worker_not_busy
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        overlap_blocks: u32,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let prefix = (overlap_blocks as usize) * (self.block_size as usize);
        let effective_isl = isl_tokens.saturating_sub(prefix);
        if effective_isl == 0 {
            return None;
        }

        let Some(estimator) = &self.prefill_load_estimator else {
            return None;
        };

        match estimator.predict_prefill_duration(1, effective_isl, prefix) {
            Ok(expected_prefill_duration) => Some(PrefillLoadHint {
                initial_effective_prefill_tokens: effective_isl,
                expected_prefill_duration: Some(expected_prefill_duration),
            }),
            Err(error) => {
                tracing::warn!(
                    effective_isl,
                    prefix,
                    "failed to predict replay prefill duration for active load tracking: {error}"
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_kv_router::PrefillLoadEstimator;
    use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};
    use uuid::Uuid;

    use super::OfflineReplayRouter;
    use crate::common::protocols::{DirectRequest, MockEngineArgs};
    use crate::replay::ReplayPrefillLoadEstimator;

    struct FixedPrefillLoadEstimator {
        duration: Duration,
    }

    impl PrefillLoadEstimator for FixedPrefillLoadEstimator {
        fn predict_prefill_duration(
            &self,
            _batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> anyhow::Result<Duration> {
            Ok(self.duration)
        }
    }

    fn replay_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(256))
            .build()
            .unwrap()
    }

    fn router_config() -> KvRouterConfig {
        KvRouterConfig {
            router_track_prefill_tokens: true,
            router_prefill_load_model: RouterPrefillLoadModel::Aic,
            ..KvRouterConfig::default()
        }
    }

    fn estimator(duration: Duration) -> ReplayPrefillLoadEstimator {
        Arc::new(FixedPrefillLoadEstimator { duration })
    }

    fn request(uuid: u128, token: u32) -> DirectRequest {
        DirectRequest {
            tokens: vec![token; 64],
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(0.0),
        }
    }

    #[test]
    fn test_prefill_load_estimator_decays_offline_router_active_tokens() {
        let mut router = OfflineReplayRouter::new(
            &replay_args(),
            Some(router_config()),
            Some(estimator(Duration::from_secs(10))),
            1,
        )
        .unwrap();

        let effects = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        assert_eq!(effects.admissions.len(), 1);
        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64)]
        );
        assert_eq!(
            router.debug_snapshot(5_000.0).active_tokens_by_worker,
            vec![(0, 32)]
        );
        assert_eq!(
            router.debug_snapshot(10_000.0).active_tokens_by_worker,
            vec![(0, 0)]
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, PrefillLoadHint, RouterEvent, RoutingConstraints,
    WorkerConfigLike, WorkerId, WorkerWithDpRank, compute_block_hash_for_seq,
};
use dynamo_kv_router::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::scheduling::{
    OverlapSignals, PolicyClassConfig, PolicyProfile, PolicyQueue, QueueSnapshot, ScheduleMode,
    SessionEnqueueError,
};
use dynamo_kv_router::sequences::topology::WorkerDpRange;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, DefaultWorkerSelector, RadixTree, SchedulingRequest,
    SequenceRequest, WorkerLoadProjection, WorkerSelector, scheduling::TierOverlapBlocks,
};
use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use tokio::time::Instant;
use uuid::Uuid;

use super::{RouterEffects, WorkerAdmission};
use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::ReplayRequestHashes;
use crate::replay::ReplayPrefillLoadEstimator;
use crate::replay::router_shared::{
    ReplayNoopPublisher, ReplayWorkerConfig, replay_router_config, replay_selector, replay_slots,
    replay_worker_config, replay_workers_with_configs,
};

/// Internal result of a successful ``admit_request`` call: the chosen
/// worker plus the router's view of prefix-cache overlap, so callers can
/// forward the overlap stats to the traffic accumulator.
#[derive(Debug, Clone, Copy)]
struct AdmitOutcome {
    worker_idx: usize,
    overlap_blocks: u32,
    isl_blocks: u32,
}

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
        // TODO: support lower tier events in replay indexer
        if !event.storage_tier.is_gpu() {
            return Ok(());
        }
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
    priority_jump: f64,
    strict_priority: u32,
    policy_class: Option<String>,
    session_id: Option<String>,
}

impl PendingRequest {
    fn request_id(&self) -> String {
        self.uuid.to_string()
    }

    fn scheduling_request(
        &self,
        block_size: usize,
        worker_loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> SchedulingRequest {
        let effective_overlap_blocks = self
            .overlaps
            .scores
            .iter()
            .map(|(worker, overlap)| (*worker, *overlap as f64))
            .collect();
        let effective_cached_tokens = self
            .overlaps
            .scores
            .iter()
            .map(|(worker, overlap)| (*worker, *overlap as usize * block_size))
            .collect();
        SchedulingRequest {
            mode: ScheduleMode::Tracked {
                request_id: self.request_id(),
            },
            token_seq: self.token_seq.clone(),
            isl_tokens: self.isl_tokens,
            overlap: OverlapSignals {
                tier_overlap_blocks: TierOverlapBlocks::default(),
                effective_overlap_blocks,
                effective_cached_tokens,
            },
            worker_loads,
            track_prefill_tokens: self.track_prefill_tokens,
            router_config_override: None,
            lora_name: None,
            priority_jump: self.priority_jump,
            strict_priority: self.strict_priority,
            policy_class: self.policy_class.clone(),
            session_id: self.session_id.clone(),
            expected_output_tokens: self.expected_output_tokens,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }
}

pub(crate) struct OfflineReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    profile: PolicyProfile,
    worker_config_template: ReplayWorkerConfig,
    workers_with_configs: HashMap<WorkerId, ReplayWorkerConfig>,
    slots: Arc<ActiveSequencesMultiWorker<ReplayNoopPublisher>>,
    selector: DefaultWorkerSelector,
    pending: PolicyQueue<PendingRequest>,
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
        let worker_config_template = replay_worker_config(args);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let selector = replay_selector(&config);
        let profile = config
            .configured_policy_profile()
            .map_err(anyhow::Error::from)?;

        Ok(Self {
            config,
            block_size: args.block_size as u32,
            profile: profile.clone(),
            worker_config_template,
            workers_with_configs,
            slots,
            selector,
            pending: PolicyQueue::new(profile),
            indexer: SyncReplayIndexer::new(args.block_size as u32),
            prefill_load_estimator,
            // This is only a base Instant for converting replay `now_ms` values into
            // synthetic `Instant`s. All subsequent decay/accounting uses virtual replay
            // time derived from this epoch, not wall-clock progression.
            decay_time_epoch: Instant::now(),
        })
    }

    #[cfg(test)]
    pub(crate) fn on_request_arrival(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.on_request_arrival_for_session(request, replay_hashes, None, now_ms)
    }

    pub(crate) fn on_request_arrival_for_session(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        let pending = self.build_pending_request(request, replay_hashes, session_id)?;
        let decay_now = self.decay_now(now_ms);
        let (class_index, snapshot) = match self
            .profile
            .direct_class_index(pending.policy_class.as_deref())
        {
            Some(class_index) => (class_index, None),
            None => {
                let snapshot = self.snapshot_for(&pending);
                (
                    self.profile.resolve_class_index(
                        pending.policy_class.as_deref(),
                        snapshot.uncached_tokens,
                    ),
                    Some(snapshot),
                )
            }
        };
        let class = self.profile.class(class_index);
        let should_queue = class.queueing_enabled()
            && (self.pending.has_backlog(class_index) || self.all_workers_busy(class, decay_now));

        if should_queue {
            let snapshot = snapshot.unwrap_or_else(|| self.snapshot_for(&pending));
            let priority_jump = pending.priority_jump;
            let strict_priority = pending.strict_priority;
            let session_id = pending.session_id.clone();
            self.pending
                .enqueue_for_session(
                    class_index,
                    self.workers_with_configs.len(),
                    snapshot,
                    now_ms.max(0.0) / 1000.0,
                    priority_jump,
                    strict_priority,
                    session_id,
                    pending,
                )
                .map_err(|(error, _)| match error {
                    SessionEnqueueError::QueueRejected(rejection) => anyhow::Error::new(rejection),
                    duplicate @ SessionEnqueueError::DuplicatePending { .. } => {
                        anyhow::Error::new(duplicate)
                    }
                })?;
            return Ok(RouterEffects::default());
        }

        let uuid = request
            .uuid
            .expect("offline replay requests must have UUIDs before router submission");
        let outcome = self.admit_request(pending, decay_now)?;
        Ok(RouterEffects {
            admissions: vec![WorkerAdmission {
                uuid,
                worker_idx: outcome.worker_idx,
                overlap_blocks: outcome.overlap_blocks,
                isl_blocks: outcome.isl_blocks,
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
            admissions: self.drain_pending(decay_now)?,
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
            admissions: self.drain_pending(decay_now)?,
        })
    }

    /// Cancel a request that has not yet been assigned to a worker.
    pub(crate) fn cancel_pending(&mut self, uuid: Uuid) -> bool {
        let before = self.pending.pending_count();
        self.pending.retain(|request| request.uuid != uuid);
        self.pending.pending_count() != before
    }

    /// Drain queued requests that can now be admitted (e.g. after a new worker
    /// becomes available).
    pub(crate) fn try_drain_pending(&mut self, now_ms: f64) -> Result<RouterEffects> {
        let decay_now = self.decay_now(now_ms);
        Ok(RouterEffects {
            admissions: self.drain_pending(decay_now)?,
        })
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.pending.pending_count()
    }

    /// Register a new worker with the router without disturbing existing slot state.
    pub(crate) fn add_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        if self
            .workers_with_configs
            .insert(wid, self.worker_config_template.clone())
            .is_some()
        {
            return Err(anyhow!("router worker {worker_id} already exists"));
        }
        if let Err(error) = self.slots.upsert_worker(WorkerDpRange::new(wid, 0, 1)) {
            self.workers_with_configs.remove(&wid);
            return Err(error.into());
        }

        Ok(())
    }

    /// Remove a worker from routing eligibility.
    ///
    /// Only removes the worker from the config map so the selector won't
    /// pick it for new requests.  The radix tree and active-sequence slots
    /// are left intact so that in-flight requests on this worker can still
    /// complete (free / mark_prefill_completed) and KV events can still
    /// reference existing blocks without "parent block not found" errors.
    /// Stale slot and indexer state is harmless — the selector and
    /// `all_workers_busy` both skip workers absent from `workers_with_configs`.
    pub(crate) fn remove_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.workers_with_configs.remove(&wid);
        Ok(())
    }

    /// Drop the retained topology/cache state after the engine confirms that
    /// the draining worker no longer owns any request lifecycle state.
    pub(crate) fn finalize_worker_removal(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.slots
            .unregister_worker(wid)
            .map_err(anyhow::Error::from)?;
        self.indexer.tree.remove_worker(wid);
        Ok(())
    }

    pub(crate) fn on_topology_changed(&mut self, now_ms: f64) -> Result<RouterEffects> {
        if self.workers_with_configs.is_empty() {
            return Ok(RouterEffects::default());
        }
        let decay_now = self.decay_now(now_ms);
        Ok(RouterEffects {
            admissions: self.drain_pending(decay_now)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self, now_ms: f64) -> OfflineRouterSnapshot {
        let decay_now = self.decay_now(now_ms);
        let mut pending = self
            .pending
            .entries()
            .map(|entry| {
                let mut overlap_blocks_by_worker = entry
                    .payload()
                    .overlaps
                    .scores
                    .iter()
                    .map(|(worker, overlap)| (worker.worker_id as usize, *overlap))
                    .collect::<Vec<_>>();
                overlap_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

                (
                    entry,
                    OfflinePendingRequestSnapshot {
                        uuid: entry.payload().uuid,
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

    fn decay_now(&self, now_ms: f64) -> Instant {
        self.decay_time_epoch + Duration::from_secs_f64(now_ms.max(0.0) / 1000.0)
    }

    fn build_pending_request(
        &self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
    ) -> Result<PendingRequest> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("offline replay requires requests to have stable UUIDs"))?;
        let (priority_jump, strict_priority) = request.router_priorities();
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
            priority_jump,
            strict_priority,
            policy_class: request.policy_class.clone(),
            session_id,
        })
    }

    fn admit_request(
        &mut self,
        request: PendingRequest,
        decay_now: Instant,
    ) -> Result<AdmitOutcome> {
        let worker_loads = self
            .slots
            .project_worker_loads(request.token_seq.as_deref(), decay_now);
        let scheduling_request = request.scheduling_request(self.block_size as usize, worker_loads);
        let eligibility = scheduling_request.eligibility();
        let selection = self.selector.select_worker(
            &self.workers_with_configs,
            &scheduling_request,
            eligibility,
            self.block_size,
        )?;
        let worker_idx = usize::try_from(selection.worker.worker_id)
            .map_err(|_| anyhow!("selected worker id does not fit into usize"))?;
        let request_id = request.request_id();
        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selection.cached_tokens,
            request.track_prefill_tokens,
        );

        let isl_blocks = u32::try_from(request.isl_tokens.div_ceil(self.block_size as usize))
            .unwrap_or(u32::MAX);
        let overlap_blocks = selection.effective_overlap_blocks.floor() as u32;

        self.slots
            .add_request(
                SequenceRequest {
                    request_id,
                    token_sequence: request.token_seq,
                    track_prefill_tokens: request.track_prefill_tokens,
                    expected_output_tokens: request.expected_output_tokens,
                    prefill_load_hint,
                    worker: selection.worker,
                    lora_name: None,
                },
                decay_now,
            )
            .map_err(anyhow::Error::from)?;

        Ok(AdmitOutcome {
            worker_idx,
            overlap_blocks,
            isl_blocks,
        })
    }

    fn drain_pending(&mut self, decay_now: Instant) -> Result<Vec<WorkerAdmission>> {
        let mut admissions = Vec::new();
        loop {
            let active_tokens = self.slots.active_tokens(decay_now);
            let workers = &self.workers_with_configs;
            let Some(popped) = self.pending.pop_next(|_, class, _| {
                !Self::all_workers_busy_with(&active_tokens, workers, class)
            }) else {
                break;
            };
            let request = popped.into_payload();
            let uuid = request.uuid;
            let outcome = self.admit_request(request, decay_now)?;
            admissions.push(WorkerAdmission {
                uuid,
                worker_idx: outcome.worker_idx,
                overlap_blocks: outcome.overlap_blocks,
                isl_blocks: outcome.isl_blocks,
            });
        }

        Ok(admissions)
    }

    fn all_workers_busy(&self, class: &PolicyClassConfig, decay_now: Instant) -> bool {
        let active_tokens = self.slots.active_tokens(decay_now);
        Self::all_workers_busy_with(&active_tokens, &self.workers_with_configs, class)
    }

    fn all_workers_busy_with(
        active_tokens: &HashMap<WorkerWithDpRank, usize>,
        workers_with_configs: &HashMap<WorkerId, ReplayWorkerConfig>,
        class: &PolicyClassConfig,
    ) -> bool {
        workers_with_configs.iter().all(|(&worker_id, config)| {
            let worker = WorkerWithDpRank::new(worker_id, config.data_parallel_start_rank());
            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
            let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
            class.worker_is_busy(tokens, max_batched)
        })
    }

    fn snapshot_for(&self, request: &PendingRequest) -> QueueSnapshot {
        let cached_tokens = request
            .overlaps
            .scores
            .iter()
            .filter(|(worker, _)| self.workers_with_configs.contains_key(&worker.worker_id))
            .map(|(_, overlap)| *overlap)
            .max()
            .unwrap_or(0) as usize
            * self.block_size as usize;
        QueueSnapshot::new(request.isl_tokens, cached_tokens)
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        cached_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let prefix = cached_tokens.min(isl_tokens);
        let effective_isl = isl_tokens.saturating_sub(prefix);
        if effective_isl == 0 {
            return None;
        }

        let expected_prefill_duration = match &self.prefill_load_estimator {
            Some(estimator) => match estimator.predict_prefill_duration(1, effective_isl, prefix) {
                Ok(expected_prefill_duration) => Some(expected_prefill_duration),
                Err(error) => {
                    tracing::warn!(
                        effective_isl,
                        prefix,
                        "failed to predict replay prefill duration for active load tracking: {error}"
                    );
                    None
                }
            },
            None => None,
        };

        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: effective_isl,
            expected_prefill_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_kv_router::PrefillLoadEstimator;
    use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel, RouterQueuePolicy};
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier, WorkerId,
    };
    use uuid::Uuid;

    use super::{OfflineReplayRouter, ReplayRequestHashes, SyncReplayIndexer, WorkerAdmission};
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

    fn queueing_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(64))
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

    fn queueing_router_config() -> KvRouterConfig {
        queueing_router_config_with_policy(RouterQueuePolicy::Fcfs)
    }

    fn queueing_router_config_with_policy(policy: RouterQueuePolicy) -> KvRouterConfig {
        KvRouterConfig {
            router_queue_threshold: Some(0.5),
            router_queue_policy: policy,
            ..KvRouterConfig::default()
        }
    }

    fn estimator(duration: Duration) -> ReplayPrefillLoadEstimator {
        Arc::new(FixedPrefillLoadEstimator { duration })
    }

    fn request(uuid: u128, token: u32) -> DirectRequest {
        request_with_priorities(uuid, token, 64, 0, 0)
    }

    fn request_with_priorities(
        uuid: u128,
        token: u32,
        input_tokens: usize,
        priority: i32,
        strict_priority: u32,
    ) -> DirectRequest {
        DirectRequest {
            tokens: vec![token; input_tokens],
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

    fn store_event(
        worker_id: WorkerId,
        event_id: u64,
        tokens_hash: u64,
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
                        tokens_hash: LocalBlockHash(tokens_hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
            storage_tier,
        )
    }

    #[test]
    fn lower_tier_events_do_not_enter_offline_primary_index() {
        let mut indexer = SyncReplayIndexer::new(64);

        indexer
            .apply_event(store_event(7, 1, 101, StorageTier::HostPinned))
            .unwrap();
        assert_eq!(indexer.debug_snapshot().total_cached_blocks, 0);

        indexer
            .apply_event(store_event(7, 2, 101, StorageTier::Device))
            .unwrap();
        assert_eq!(indexer.debug_snapshot().total_cached_blocks, 1);
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

    #[test]
    fn test_single_worker_router_honors_queue_threshold() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        let first = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        assert_eq!(first.admissions.len(), 1);

        let second = router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert!(second.admissions.is_empty());
        assert_eq!(router.pending_count(), 1);
        assert_eq!(
            router
                .debug_snapshot(0.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>(),
            vec![Uuid::from_u128(2)]
        );
    }

    #[test]
    fn canceled_pending_request_is_not_admitted_after_capacity_frees() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        assert_eq!(
            router
                .on_request_arrival(&request(1, 7), None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1
        );
        assert!(
            router
                .on_request_arrival(&request(2, 8), None, 0.0)
                .unwrap()
                .admissions
                .is_empty()
        );
        assert!(router.cancel_pending(Uuid::from_u128(2)));
        assert!(!router.cancel_pending(Uuid::from_u128(2)));
        assert_eq!(router.pending_count(), 0);

        let effects = router
            .on_request_completed(Uuid::from_u128(1), 1.0)
            .unwrap();
        assert!(effects.admissions.is_empty());
    }

    #[test]
    fn policy_mapping_and_model_selection_use_shared_replay_queue_logic() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-policy-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            r#"
default_policy_family: root
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: root
    policy_family: root
    cache_bucket: all
    quantum: 1
models:
  replay-model:
    default_policy_family: latency
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: cached
      - min_tokens: 32
        bucket: uncached
    policy_classes:
      - name: latency_cached
        policy_family: latency
        cache_bucket: cached
        quantum: 1
        prefill_busy_threshold: 0
        request_queue_limit_per_worker: 1
      - name: latency_uncached
        policy_family: latency
        cache_bucket: uncached
        quantum: 1
        prefill_busy_threshold: 0
        request_queue_limit_per_worker: 0
      - name: batch_cached
        policy_family: batch
        cache_bucket: cached
        quantum: 4
        prefill_busy_threshold: 1024
      - name: batch_uncached
        policy_family: batch
        cache_bucket: uncached
        quantum: 4
        prefill_busy_threshold: 1024
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        }
        .with_policy_model_name(Some("replay-model".to_string()));
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();

        let mut active = request(1, 1);
        active.policy_class = Some("latency".to_string());
        assert_eq!(
            router
                .on_request_arrival(&active, None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1
        );

        let mut mapped_batch = request(2, 2);
        mapped_batch.policy_class = Some("batch".to_string());
        assert_eq!(
            router
                .on_request_arrival(&mapped_batch, None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1,
            "the batch family should select batch_uncached and use its higher busy threshold"
        );

        let mut cached_latency = request(3, 3);
        cached_latency.policy_class = Some("latency".to_string());
        let cached_hashes =
            ReplayRequestHashes::from_tokens(&cached_latency.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                0,
                1,
                cached_hashes.local_block_hashes[0].0,
                StorageTier::Device,
            )])
            .unwrap();
        assert!(
            router
                .on_request_arrival(&cached_latency, Some(cached_hashes.clone()), 0.0)
                .unwrap()
                .admissions
                .is_empty(),
            "the latency family should select latency_cached from observed cache state"
        );

        let mut ordinary_class_name = request(4, 4);
        ordinary_class_name.policy_class = Some("latency_cached".to_string());
        let error = router
            .on_request_arrival(&ordinary_class_name, None, 0.0)
            .unwrap_err();
        let rejection = error
            .downcast_ref::<dynamo_kv_router::scheduling::QueueRejection>()
            .expect("replay should preserve the typed queue rejection");
        assert_eq!(rejection.policy_class, "latency_uncached");
        assert_eq!(rejection.current, 0);
        assert_eq!(rejection.limit, 0);

        router.add_worker(1).unwrap();
        let mut scaled_latency = request(5, 3);
        scaled_latency.policy_class = Some("latency".to_string());
        assert!(
            router
                .on_request_arrival(&scaled_latency, Some(cached_hashes.clone()), 0.0)
                .unwrap()
                .admissions
                .is_empty(),
            "a second discovered worker should raise the effective queue limit"
        );
        assert_eq!(router.pending_count(), 2);

        router.remove_worker(1).unwrap();
        let mut rejected_after_shrink = request(6, 3);
        rejected_after_shrink.policy_class = Some("latency".to_string());
        let error = router
            .on_request_arrival(&rejected_after_shrink, Some(cached_hashes), 0.0)
            .unwrap_err();
        let rejection = error
            .downcast_ref::<dynamo_kv_router::scheduling::QueueRejection>()
            .expect("worker removal should retain the typed queue rejection");
        assert_eq!(rejection.current, 2);
        assert_eq!(rejection.limit, 1);
        assert_eq!(router.pending_count(), 2);
    }

    #[test]
    fn replay_cache_bucket_ignores_removed_worker_entries() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-policy-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 0
  - name: uncached
    policy_family: standard
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 1024
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 2).unwrap();
        std::fs::remove_file(path).unwrap();

        let target = request(2, 2);
        let target_hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                1,
                1,
                target_hashes.local_block_hashes[0].0,
                StorageTier::Device,
            )])
            .unwrap();
        router.remove_worker(1).unwrap();

        router
            .on_request_arrival(&request(1, 1), None, 0.0)
            .unwrap();
        let effects = router
            .on_request_arrival(&target, Some(target_hashes), 0.0)
            .unwrap();
        assert_eq!(
            effects.admissions.len(),
            1,
            "cache state from a removed worker must not select the cached queue"
        );
        assert_eq!(router.pending_count(), 0);
    }

    #[test]
    fn strict_priority_precedes_each_offline_queue_policy() {
        for policy in [
            RouterQueuePolicy::Fcfs,
            RouterQueuePolicy::Lcfs,
            RouterQueuePolicy::Wspt,
        ] {
            let mut router = OfflineReplayRouter::new(
                &queueing_args(),
                Some(queueing_router_config_with_policy(policy)),
                None,
                1,
            )
            .unwrap();
            router
                .on_request_arrival(&request(1, 1), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(2, 2, 1, 1_000_000, 0), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(3, 3, 64, 0, 1), None, 1_000.0)
                .unwrap();

            let pending = router
                .debug_snapshot(1_000.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>();
            assert_eq!(pending, vec![Uuid::from_u128(3), Uuid::from_u128(2)]);
        }
    }

    #[test]
    fn equal_strict_tiers_retain_offline_queue_policy_ordering() {
        let cases = [
            (RouterQueuePolicy::Fcfs, vec![2, 3]),
            (RouterQueuePolicy::Lcfs, vec![3, 2]),
            (RouterQueuePolicy::Wspt, vec![3, 2]),
        ];

        for (policy, expected) in cases {
            let mut router = OfflineReplayRouter::new(
                &queueing_args(),
                Some(queueing_router_config_with_policy(policy)),
                None,
                1,
            )
            .unwrap();
            router
                .on_request_arrival(&request(1, 1), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(2, 2, 64, 0, 4), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(3, 3, 1, 0, 4), None, 100.0)
                .unwrap();

            let pending = router
                .debug_snapshot(100.0)
                .pending
                .into_iter()
                .map(|request| request.uuid.as_u128())
                .collect::<Vec<_>>();
            assert_eq!(pending, expected);
        }
    }

    #[test]
    fn soft_priority_orders_within_an_offline_strict_tier() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        router
            .on_request_arrival(&request(1, 1), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request_with_priorities(2, 2, 64, 0, 2), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request_with_priorities(3, 3, 64, 1_000, 2), None, 100.0)
            .unwrap();

        let pending = router
            .debug_snapshot(100.0)
            .pending
            .into_iter()
            .map(|request| request.uuid)
            .collect::<Vec<_>>();
        assert_eq!(pending, vec![Uuid::from_u128(3), Uuid::from_u128(2)]);
    }

    #[test]
    fn test_scaled_down_router_matches_fresh_single_worker_queueing() {
        let mut fresh =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        fresh.on_request_arrival(&request(1, 7), None, 0.0).unwrap();
        fresh.on_request_arrival(&request(2, 8), None, 0.0).unwrap();

        let mut scaled =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 2)
                .unwrap();
        scaled.remove_worker(1).unwrap();
        scaled
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        scaled
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();

        assert_eq!(scaled.pending_count(), fresh.pending_count());
        assert_eq!(
            scaled
                .debug_snapshot(0.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>(),
            fresh
                .debug_snapshot(0.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_router_can_scale_from_zero_workers() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router.remove_worker(0).unwrap();
        router.add_worker(3).unwrap();

        let effects = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 3,
                overlap_blocks: 0,
                isl_blocks: 1,
            }]
        );
    }

    #[test]
    fn finalized_worker_removal_drops_retained_topology_and_cache_state() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        let target = request(1, 7);
        let hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                0,
                1,
                hashes.local_block_hashes[0].0,
                StorageTier::Device,
            )])
            .unwrap();
        assert_eq!(router.debug_snapshot(0.0).indexer.total_cached_blocks, 1);

        router.remove_worker(0).unwrap();
        router.finalize_worker_removal(0).unwrap();
        let snapshot = router.debug_snapshot(0.0);
        assert!(snapshot.active_blocks_by_worker.is_empty());
        assert!(snapshot.indexer.cached_blocks_by_worker.is_empty());
        router.add_worker(0).unwrap();
    }

    #[test]
    fn test_add_worker_preserves_draining_worker_state() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 2)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64), (1, 64)]
        );

        router.remove_worker(1).unwrap();
        router.add_worker(2).unwrap();

        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64), (1, 64), (2, 0)]
        );
    }

    #[test]
    fn test_readding_removed_worker_reuses_slot_state() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router.remove_worker(0).unwrap();
        router.add_worker(0).unwrap();

        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64)]
        );
    }

    #[test]
    fn test_topology_change_drains_pending_after_scale_up() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert_eq!(router.pending_count(), 1);

        router.add_worker(1).unwrap();
        let effects = router.on_topology_changed(0.0).unwrap();

        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(2),
                worker_idx: 1,
                overlap_blocks: 0,
                isl_blocks: 1,
            }]
        );
        assert_eq!(router.pending_count(), 0);
        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64), (1, 64)]
        );
    }

    #[test]
    fn no_workers_preserves_pending_requests_during_completion_drain() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert_eq!(router.pending_count(), 1);

        router.remove_worker(0).unwrap();
        let effects = router
            .on_request_completed(Uuid::from_u128(1), 0.0)
            .unwrap();

        assert!(effects.admissions.is_empty());
        assert_eq!(router.pending_count(), 1);
        assert_eq!(
            router.debug_snapshot(0.0).pending[0].uuid,
            Uuid::from_u128(2)
        );
    }
}

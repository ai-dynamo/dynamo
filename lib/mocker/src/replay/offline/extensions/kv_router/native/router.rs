// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for AISimulate: native Router integration with AISim-owned types.
// Derived from https://github.com/ai-dynamo/dynamo/blob/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/lib/mocker/src/replay/offline/extensions/kv_router/mod.rs

use dynamo_custom_policy_builtin::DefaultWorkerSelector;
use std::borrow::Cow;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::indexer::{MatchDetails, TieredMatchDetails};
use dynamo_kv_router::protocols::WorkerAffinityTarget;
use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, RouterEvent, RoutingConstraints, WorkerId, WorkerWithDpRank,
    compute_block_hash_for_seq,
};
use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;
use dynamo_kv_router::scheduling::{
    AdmissionAttempt, AdmittedSchedulingResponse, LocalScheduler, OverlapAnalysis, OverlapSignals,
    ScheduleMode, ScheduleRequest,
};
use dynamo_kv_router::sequences::topology::WorkerDpRange;
use dynamo_kv_router::services::selection::affinity::Hold;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, NoopSequencePublisher, RadixTree, RoutingPartitionRef,
    SessionContext, TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope,
};
use dynamo_tokens::SequenceHash;
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::runtime::{ReplayRuntime, WakeFlag};
use uuid::Uuid;

use super::ReplayPrefillLoadEstimator;
use super::helpers::NativeRouterEngineArgs;
use super::helpers::{
    ReplayWorkerConfig, replay_router_config, replay_selector_with_seed, replay_slots,
    replay_worker_config, replay_workers_with_configs,
};
use super::observation::RouterEventBatch;
use aisimulate_core::replay::DirectRequest;
use aisimulate_core::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use aisimulate_core::replay::{
    Placement, PlacementCacheSample, PlacementDecision, PlacementEffects, PlacementPolicy,
    ReplayAdmissionMetadata, WorkerTopology,
};

#[cfg(test)]
use super::affinity::ReplayAffinityMode;
use super::affinity::{ReplayAffinity, ReplayAffinityConfig, RoutingEvidence};

/// KV identities carried by AISimulate into the native Router placement policy.
#[derive(Debug, Default)]
pub struct KvReplayMetadata {
    hashes: Option<ReplayRequestHashes>,
    max_output_tokens_override: Option<usize>,
}

impl ReplayAdmissionMetadata for KvReplayMetadata {
    fn from_hashes(hashes: Option<ReplayRequestHashes>) -> Self {
        Self {
            hashes,
            max_output_tokens_override: None,
        }
    }

    fn for_prefill(mut self) -> Self {
        self.max_output_tokens_override = Some(1);
        self
    }

    fn max_output_tokens_override(&self) -> Option<usize> {
        self.max_output_tokens_override
    }

    fn into_hashes(self) -> Option<ReplayRequestHashes> {
        self.hashes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerAdmission {
    uuid: Uuid,
    worker_idx: usize,
    overlap_blocks: u32,
    best_available_overlap_blocks: u32,
    isl_blocks: u32,
}

#[derive(Debug, Default)]
pub(crate) struct RouterEffects {
    admissions: Vec<WorkerAdmission>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflinePendingRequestSnapshot {
    pub(crate) uuid: Uuid,
    pub(crate) expected_output_tokens: Option<u32>,
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
    /// Disaggregated decode placement never observes KV events, so its tree
    /// stays empty and hashing a prompt to query it would be wasted work.
    has_indexed_events: bool,
}

impl SyncReplayIndexer {
    fn new(block_size: u32) -> Self {
        Self {
            block_size,
            tree: RadixTree::new(),
            has_indexed_events: false,
        }
    }

    fn find_matches_for_request(&self, tokens: &[u32], lora_name: Option<&str>) -> OverlapScores {
        if !self.has_indexed_events {
            return OverlapScores::default();
        }
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
        self.has_indexed_events = true;
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

pub(super) struct PendingRequest {
    pub(super) uuid: Uuid,
    pub(super) token_seq: Option<Vec<SequenceHash>>,
    pub(super) isl_tokens: usize,
    pub(super) overlaps: OverlapSignals,
    pub(super) expected_output_tokens: Option<u32>,
    pub(super) priority_jump: f64,
    pub(super) strict_priority: u32,
    pub(super) policy_class: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) group_key: Option<String>,
}

impl PendingRequest {
    fn request_id(&self) -> String {
        self.uuid.to_string()
    }

    fn scheduling_request(mut self, target: Option<WorkerAffinityTarget>) -> ScheduleRequest {
        ScheduleRequest {
            mode: ScheduleMode::TrackedWithLifecycle {
                request_id: self.request_id(),
            },
            token_seq: self.token_seq.take(),
            block_hashes: None,
            isl_tokens: self.isl_tokens,
            overlap: self.overlaps,
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            router_config_override: None,
            lora_name: None,
            priority_jump: self.priority_jump,
            strict_priority: self.strict_priority,
            policy_class: self.policy_class,
            session_context: self
                .session_id
                .map(|id| SessionContext::new(id, None, None, None)),
            expected_output_tokens: self.expected_output_tokens,
            affinity_target: target,
            pinned_worker: target.and_then(|target| {
                target
                    .dp_rank
                    .map(|rank| WorkerWithDpRank::new(target.worker_id, rank))
            }),
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            shared_cache_hits: None,
        }
    }
}

type NativeScheduler =
    LocalScheduler<NoopSequencePublisher, ReplayWorkerConfig, DefaultWorkerSelector>;

struct PendingAdmission {
    uuid: Uuid,
    future:
        Pin<Box<dyn Future<Output = Result<(AdmittedSchedulingResponse, Option<Hold>)>> + Send>>,
    group_key: Option<String>,
    isl_blocks: u32,
    #[cfg(test)]
    snapshot: OfflinePendingRequestSnapshot,
}

struct OfflineReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    dp_size: u32,
    worker_config_template: ReplayWorkerConfig,
    workers_with_configs: HashMap<WorkerId, ReplayWorkerConfig>,
    slots: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    scheduler: Arc<NativeScheduler>,
    workers_tx: watch::Sender<HashMap<WorkerId, ReplayWorkerConfig>>,
    pending: Vec<PendingAdmission>,
    bookings: HashMap<Uuid, SchedulerBookingDescriptor>,
    runtime: ReplayRuntime,
    indexer: SyncReplayIndexer,
    decay_time_epoch: Instant,
    tracking_hash: TrackingHashContext,
    affinity: Option<ReplayAffinity>,
}

pub struct KvRouterPlacement {
    router: OfflineReplayRouter,
}

impl KvRouterPlacement {
    pub(super) fn new_with_selector_seed(
        args: &NativeRouterEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        selector_seed: Option<u64>,
    ) -> Result<Self> {
        let router = match selector_seed {
            Some(seed) => OfflineReplayRouter::new_with_selector_seed(
                args,
                router_config,
                prefill_load_estimator,
                num_workers,
                Some(seed),
            )?,
            None => {
                OfflineReplayRouter::new(args, router_config, prefill_load_estimator, num_workers)?
            }
        };
        Ok(Self { router })
    }

    pub(super) fn with_affinity(
        mut self,
        config: Option<ReplayAffinityConfig>,
        role: &'static str,
        capture: bool,
        evidence: RoutingEvidence,
    ) -> Result<Self> {
        if let Some(config) = config {
            self.router.affinity = Some(ReplayAffinity::new(
                config,
                self.router.decay_time_epoch,
                role,
                capture,
                evidence,
            )?);
        }
        Ok(self)
    }

    fn placement(&self, admission: WorkerAdmission) -> Placement {
        Placement {
            request_id: admission.uuid,
            scheduler_id: admission.worker_idx,
            placement_replica_id: None,
            reported_overlap_tokens: admission.overlap_blocks as usize
                * self.router.block_size as usize,
            cache_sample: Some(PlacementCacheSample {
                overlap_blocks: admission.overlap_blocks,
                best_available_overlap_blocks: admission.best_available_overlap_blocks,
                isl_blocks: admission.isl_blocks,
            }),
        }
    }

    fn placements(&self, admissions: Vec<WorkerAdmission>) -> Vec<Placement> {
        admissions
            .into_iter()
            .map(|admission| self.placement(admission))
            .collect()
    }
}

trait PlacementRequestView {
    fn metadata(&self) -> &DirectRequest;
    fn input_length(&self) -> usize;
    fn prompt_tokens_for_placement(&self) -> Result<Cow<'_, [u32]>>;
}

impl PlacementRequestView for DirectRequest {
    fn metadata(&self) -> &DirectRequest {
        self
    }

    fn input_length(&self) -> usize {
        self.tokens.len()
    }

    fn prompt_tokens_for_placement(&self) -> Result<Cow<'_, [u32]>> {
        if !self.prompt_tokens_are_placement_safe() {
            return Err(anyhow!(
                "Dynamo KV Router placement requires authored prompt token IDs or replay hashes; length-only execution tokens are not valid KV identities"
            ));
        }
        Ok(Cow::Borrowed(&self.tokens))
    }
}

impl PlacementRequestView for ReplayRequestPayload {
    fn metadata(&self) -> &DirectRequest {
        self.metadata()
    }

    fn input_length(&self) -> usize {
        self.input_length()
    }

    fn prompt_tokens_for_placement(&self) -> Result<Cow<'_, [u32]>> {
        if !self.metadata().prompt_tokens_are_placement_safe() {
            return Err(anyhow!(
                "Dynamo KV Router placement requires authored prompt token IDs or replay hashes; length-only execution tokens are not valid KV identities"
            ));
        }
        match self.materialized_tokens() {
            Some(tokens) => Ok(Cow::Borrowed(tokens)),
            None => Ok(Cow::Owned(ReplayRequestPayload::prompt_tokens(self))),
        }
    }
}

impl<Request: PlacementRequestView> PlacementPolicy<Request> for KvRouterPlacement {
    type Metadata = KvReplayMetadata;
    type Observation = RouterEventBatch;

    fn place(
        &mut self,
        request: &Request,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        let request_metadata = request.metadata();
        let effective_max_output_tokens = request_metadata.effective_max_output_tokens();
        let max_output_tokens = metadata
            .max_output_tokens_override()
            .map_or(effective_max_output_tokens, |override_tokens| {
                effective_max_output_tokens.min(override_tokens)
            });
        let request_id = request_metadata
            .uuid
            .ok_or_else(|| anyhow!("KV placement requires a request UUID"))?;
        let admissions = self
            .router
            .on_compact_request_arrival_for_session(
                request,
                max_output_tokens,
                metadata.into_hashes(),
                session_id,
                now_ms,
            )?
            .admissions;
        let mut decision = PlacementDecision::Queued;
        let mut released = Vec::with_capacity(admissions.len().saturating_sub(1));
        for admission in admissions {
            let placement = self.placement(admission);
            if placement.request_id == request_id {
                decision = PlacementDecision::Immediate(placement);
            } else {
                released.push(placement);
            }
        }
        Ok(PlacementEffects { decision, released })
    }

    fn observe(&mut self, observation: RouterEventBatch, now_ms: f64) -> Result<Vec<Placement>> {
        self.router.advance_time(now_ms)?;
        let effects = self.router.on_kv_events(observation.0)?;
        Ok(self.placements(effects.admissions))
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.router.cancel_pending(request_id)
    }

    fn dispatch_committed(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        if let Some(affinity) = &mut self.router.affinity {
            affinity.advance(now_ms)?;
            affinity.commit(request_id)?;
        }
        Ok(())
    }

    fn dispatch_aborted(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        self.router.advance_time(now_ms)?;
        if let Some(affinity) = &mut self.router.affinity {
            affinity.release(request_id, true)?;
        }
        self.router.free_booking(request_id)?;
        Ok(())
    }

    fn advance_clock(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        self.router.advance_time(now_ms)?;
        let admissions = self.router.collect_admissions()?;
        Ok(self.placements(admissions))
    }

    fn next_wakeup_ms(&self) -> Option<f64> {
        self.router
            .affinity
            .as_ref()
            .filter(|affinity| affinity.ready && self.router.pending_count() > 0)
            .map(|affinity| affinity.now_ms)
    }

    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        let effects = self.router.on_request_completed(request_id, now_ms)?;
        Ok(self.placements(effects.admissions))
    }

    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        let effects = self.router.on_prefill_completed(request_id, now_ms)?;
        Ok(self.placements(effects.admissions))
    }

    fn pending_count(&self) -> usize {
        self.router.pending_count()
    }

    fn worker_ready(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.router.add_worker(worker.worker_id)?;
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.router.remove_worker(worker.worker_id)?;
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.router.finalize_worker_removal(worker.worker_id)?;
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        let effects = self.router.on_topology_changed(now_ms)?;
        Ok(self.placements(effects.admissions))
    }
}

impl OfflineReplayRouter {
    fn new(
        args: &NativeRouterEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        Self::new_with_selector_seed(
            args,
            router_config,
            prefill_load_estimator,
            num_workers,
            None,
        )
    }

    fn new_with_selector_seed(
        args: &NativeRouterEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        selector_seed: Option<u64>,
    ) -> Result<Self> {
        let runtime = ReplayRuntime::new()?;
        let _entered = runtime.enter();
        let config = replay_router_config(router_config);
        let tracking_hash = TrackingHashContext::from_config(&config)?;
        let worker_config_template = replay_worker_config(args);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let selector = replay_selector_with_seed(&config, selector_seed)?;
        let profile = config
            .configured_policy_profile()
            .map_err(anyhow::Error::from)?;

        let (workers_tx, workers_rx) = watch::channel(workers_with_configs.clone());
        let scheduler = Arc::new(LocalScheduler::new(
            Arc::clone(&slots),
            workers_rx,
            profile,
            args.block_size,
            selector,
            prefill_load_estimator,
            None,
            None,
            None,
            Duration::from_secs(60),
            config.router_track_prefill_tokens,
            CancellationToken::new(),
            "replay",
            false,
        ));
        let decay_time_epoch = runtime.epoch();
        drop(_entered);
        Ok(Self {
            config,
            block_size: args.block_size,
            dp_size: args.dp_size.max(1),
            worker_config_template,
            workers_with_configs,
            workers_tx,
            slots,
            scheduler,
            pending: Vec::new(),
            bookings: HashMap::new(),
            indexer: SyncReplayIndexer::new(args.block_size),
            decay_time_epoch,
            runtime,
            tracking_hash,
            affinity: None,
        })
    }

    #[cfg(test)]
    fn on_request_arrival(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.on_request_arrival_for_session(request, replay_hashes, None, now_ms)
    }

    #[cfg(test)]
    fn on_request_arrival_for_session(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.on_compact_request_arrival_for_session(
            request,
            request.effective_max_output_tokens(),
            replay_hashes,
            session_id,
            now_ms,
        )
    }

    fn on_compact_request_arrival_for_session<Request: PlacementRequestView>(
        &mut self,
        request: &Request,
        max_output_tokens: usize,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        let pending =
            self.build_pending_request(request, max_output_tokens, replay_hashes, session_id)?;
        let uuid = pending.uuid;
        let group_key = pending.group_key.clone();
        let table = self
            .affinity
            .as_ref()
            .map(|affinity| affinity.table.clone());
        let scheduler = Arc::clone(&self.scheduler);
        let block_size = self.block_size;
        let isl_blocks =
            u32::try_from(pending.isl_tokens.div_ceil(block_size as usize)).unwrap_or(u32::MAX);
        #[cfg(test)]
        let snapshot = OfflinePendingRequestSnapshot {
            uuid,
            expected_output_tokens: pending.expected_output_tokens,
            overlap_blocks_by_worker: pending
                .overlaps
                .tier_overlap_blocks
                .device
                .iter()
                .map(|(worker, blocks)| (worker.worker_id as usize, *blocks as u32))
                .collect(),
        };
        let key = group_key.clone();
        let future = Box::pin(async move {
            let hold = match (table, key) {
                (Some(table), Some(key)) => Some(table.acquire(&key, None).await?),
                _ => None,
            };
            let target = hold.as_ref().and_then(Hold::target);
            let admitted = scheduler
                .schedule_request_admitted(pending.scheduling_request(target))
                .await?;
            Ok((admitted, hold))
        });
        self.pending.push(PendingAdmission {
            uuid,
            future,
            group_key,
            isl_blocks,
            #[cfg(test)]
            snapshot,
        });
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    /// Poll native admission futures to quiescence, with the scheduler's own
    /// update acknowledgement as the actor barrier. No simulator queue policy.
    fn collect_admissions(&mut self) -> Result<Vec<WorkerAdmission>> {
        let wake = Arc::new(WakeFlag::default());
        let waker = wake.waker();
        let mut context = TaskContext::from_waker(&waker);
        let mut admissions = Vec::new();
        loop {
            wake.take();
            let _entered = self.runtime.enter();
            let mut index = 0;
            while index < self.pending.len() {
                let Poll::Ready(result) = self.pending[index].future.as_mut().poll(&mut context)
                else {
                    index += 1;
                    continue;
                };
                let pending = self.pending.remove(index);
                let (admitted, hold) = result?;
                let selection = admitted.response;
                let AdmissionAttempt::Tracked(attempt_id) = admitted.attempt else {
                    anyhow::bail!("Dynamo returned an untracked simulator admission");
                };
                self.bookings.insert(
                    pending.uuid,
                    SchedulerBookingDescriptor {
                        request_id: pending.uuid.to_string(),
                        worker: selection.best_worker,
                        attempt_id,
                    },
                );
                let overlap_blocks = selection.effective_overlap_blocks.floor() as u32;
                let best_available_overlap_blocks = (selection.max_raw_cached_tokens.unwrap_or(0)
                    / self.block_size as usize)
                    as u32;
                if let Some(affinity) = &mut self.affinity {
                    affinity.stage(
                        pending.uuid,
                        hold.context("native affinity admission has no hold")?,
                        WorkerAffinityTarget {
                            worker_id: selection.best_worker.worker_id,
                            dp_rank: Some(selection.best_worker.dp_rank),
                        },
                        pending
                            .group_key
                            .as_deref()
                            .context("native affinity admission has no group key")?,
                        overlap_blocks,
                        best_available_overlap_blocks,
                    )?;
                }
                let worker_idx = usize::try_from(selection.best_worker.worker_id)?
                    .checked_mul(self.dp_size as usize)
                    .and_then(|base| base.checked_add(selection.best_worker.dp_rank as usize))
                    .context("selected worker/rank index overflow")?;
                admissions.push(WorkerAdmission {
                    uuid: pending.uuid,
                    worker_idx,
                    overlap_blocks,
                    best_available_overlap_blocks,
                    isl_blocks: pending.isl_blocks,
                });
            }
            drop(_entered);
            self.runtime
                .block_on(self.scheduler.admission_barrier())??;
            if !wake.take() {
                break;
            }
        }
        if let Some(affinity) = &mut self.affinity {
            affinity.ready = false;
        }
        Ok(admissions)
    }

    fn advance_time(&mut self, now_ms: f64) -> Result<()> {
        self.runtime.advance_to_ms(now_ms)?;
        if let Some(affinity) = &mut self.affinity {
            affinity.advance(now_ms)?;
        }
        Ok(())
    }

    fn free_booking(&mut self, uuid: Uuid) -> Result<()> {
        if let Some(booking) = self.bookings.remove(&uuid) {
            self.runtime
                .block_on(self.scheduler.free_if_booking(&booking))??;
        }
        Ok(())
    }

    fn on_kv_events(&mut self, events: Vec<RouterEvent>) -> Result<RouterEffects> {
        if let Some(affinity) = &self.affinity {
            anyhow::ensure!(
                events.iter().all(|event| event.storage_tier.is_gpu()),
                "conversation affinity supports device KV events only"
            );
            affinity.record_events(events.len())?;
        }
        for event in events {
            let worker_id = event.worker_id;
            let event_id = event.event.event_id;
            let dp_rank = event.event.dp_rank;
            self.indexer.apply_event(event).with_context(|| {
                format!(
                    "failed to apply replay KV event worker={worker_id} dp_rank={dp_rank} event_id={event_id}"
                )
            })?;
        }
        Ok(RouterEffects::default())
    }

    fn on_prefill_completed(&mut self, uuid: Uuid, now_ms: f64) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        if let Some(booking) = self.bookings.get(&uuid) {
            self.runtime
                .block_on(self.scheduler.mark_prefill_completed_if_booking(booking))??;
        }
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    fn on_request_completed(&mut self, uuid: Uuid, now_ms: f64) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        if let Some(affinity) = &mut self.affinity {
            affinity.release(uuid, false)?;
        }
        self.free_booking(uuid)?;
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    fn cancel_pending(&mut self, uuid: Uuid) -> bool {
        let before = self.pending.len();
        // Dropping the native lifecycle future cancels its admission lease.
        self.pending.retain(|pending| pending.uuid != uuid);
        let removed = self.pending.len() != before;
        if let Some(affinity) = &mut self.affinity {
            affinity.ready |= removed;
        }
        removed
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Register a new worker with the router without disturbing existing slot state.
    fn add_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        if self
            .workers_with_configs
            .insert(wid, self.worker_config_template.clone())
            .is_some()
        {
            return Err(anyhow!("router worker {worker_id} already exists"));
        }
        if let Err(error) = self
            .slots
            .upsert_worker(WorkerDpRange::new(wid, 0, self.dp_size))
        {
            self.workers_with_configs.remove(&wid);
            return Err(error.into());
        }

        self.workers_tx
            .send_replace(self.workers_with_configs.clone());
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
    fn remove_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.workers_with_configs.remove(&wid);
        self.workers_tx
            .send_replace(self.workers_with_configs.clone());
        Ok(())
    }

    /// Drop the retained topology/cache state after the engine confirms that
    /// the draining worker no longer owns any request lifecycle state.
    fn finalize_worker_removal(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.slots
            .unregister_worker(wid)
            .map_err(anyhow::Error::from)?;
        self.indexer.tree.remove_worker(wid);
        Ok(())
    }

    fn on_topology_changed(&mut self, now_ms: f64) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    #[cfg(test)]
    fn debug_snapshot(&self, now_ms: f64) -> OfflineRouterSnapshot {
        let decay_now = self.decay_now(now_ms);
        let pending = self
            .pending
            .iter()
            .map(|pending| pending.snapshot.clone())
            .collect();
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
            pending,
            active_blocks_by_worker,
            active_tokens_by_worker,
            indexer: self.indexer.debug_snapshot(),
        }
    }

    #[cfg(test)]
    fn decay_now(&self, now_ms: f64) -> Instant {
        self.decay_time_epoch + Duration::from_secs_f64(now_ms.max(0.0) / 1000.0)
    }

    fn build_pending_request<Request: PlacementRequestView>(
        &self,
        request_view: &Request,
        max_output_tokens: usize,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
    ) -> Result<PendingRequest> {
        let request = request_view.metadata();
        if self.affinity.is_some() {
            anyhow::ensure!(
                request.preferred_dp_rank.is_none() && request.preferred_prefill_dp_rank.is_none(),
                "conversation affinity does not support authored DP-rank pins"
            );
            anyhow::ensure!(
                request.policy_class.is_none(),
                "conversation affinity does not support authored policy classes"
            );
        }
        let input_length = request_view.input_length();
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("offline replay requires requests to have stable UUIDs"))?;
        let (priority_jump, strict_priority) = request.router_priorities();
        let (overlaps, token_seq) = match replay_hashes {
            Some(replay_hashes) => {
                let overlaps = self.indexer.find_matches_for_hashes(
                    replay_hashes
                        .local_block_hashes
                        .into_iter()
                        .map(LocalBlockHash)
                        .collect(),
                );
                let token_seq = if !self.config.router_track_active_blocks {
                    None
                } else if self.config.router_assume_kv_reuse
                    && self.tracking_hash.algorithm() == TrackingHashAlgorithm::PublicXxh3V1
                {
                    Some(replay_hashes.sequence_hashes)
                } else if !self.config.router_assume_kv_reuse {
                    self.config
                        .random_seq_hashes_for_tracking(input_length / self.block_size as usize)
                } else {
                    let tokens = request_view.prompt_tokens_for_placement()?;
                    self.config.compute_seq_hashes_for_tracking_with_context(
                        &self.tracking_hash,
                        self.tracking_hash_scope(),
                        &tokens,
                        None,
                        BlockHashOptions::default(),
                        None,
                    )
                };
                (overlaps, token_seq)
            }
            None => {
                let tokens = request_view.prompt_tokens_for_placement()?;
                let overlaps = self.indexer.find_matches_for_request(&tokens, None);
                let token_seq = self.config.compute_seq_hashes_for_tracking_with_context(
                    &self.tracking_hash,
                    self.tracking_hash_scope(),
                    &tokens,
                    None,
                    BlockHashOptions::default(),
                    None,
                );
                (overlaps, token_seq)
            }
        };

        let tiered = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: overlaps,
                ..Default::default()
            },
            ..Default::default()
        };
        let overlaps = OverlapAnalysis::new(&self.config, self.block_size, &tiered).signals();
        Ok(PendingRequest {
            uuid,
            token_seq,
            isl_tokens: input_length,
            overlaps,
            expected_output_tokens: Some(
                u32::try_from(max_output_tokens)
                    .context("max_output_tokens does not fit into u32")?,
            ),
            priority_jump,
            strict_priority,
            policy_class: request.policy_class.clone(),
            group_key: self
                .affinity
                .as_ref()
                .map(|affinity| affinity.config.group_key(request, session_id.as_deref()))
                .transpose()?,
            session_id,
        })
    }

    fn tracking_hash_scope(&self) -> TrackingHashScope<'_> {
        TrackingHashScope {
            partition: RoutingPartitionRef::new("replay", "default"),
            block_size: self.block_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel, RouterQueuePolicy};
    use dynamo_kv_router::protocols::{
        BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        WorkerId, compute_block_hash_for_seq,
    };
    use dynamo_kv_router::{PrefillLoadEstimator, TrackingHashAlgorithm};
    use serde_json::Value;
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    use super::super::ReplayPrefillLoadEstimator;
    use super::super::helpers::NativeRouterEngineArgs;
    use super::{OfflineReplayRouter, ReplayRequestHashes, SyncReplayIndexer, WorkerAdmission};
    use aisimulate_core::replay::DirectRequest;
    use aisimulate_core::replay::{ReplayPromptTokenSource, ReplayRequestContext};

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

    fn replay_args() -> NativeRouterEngineArgs {
        NativeRouterEngineArgs {
            dp_size: 1,
            block_size: 64,
            num_gpu_blocks: 16_384,
            max_num_batched_tokens: Some(256),
        }
    }

    fn queueing_args() -> NativeRouterEngineArgs {
        NativeRouterEngineArgs {
            max_num_batched_tokens: Some(64),
            ..replay_args()
        }
    }

    fn router_config() -> KvRouterConfig {
        KvRouterConfig {
            router_track_prefill_tokens: true,
            router_prefill_load_model: RouterPrefillLoadModel::Ais,
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
            preferred_dp_rank: None,
            preferred_prefill_dp_rank: None,
            arrival_timestamp_ms: Some(0.0),
            priority,
            strict_priority,
            policy_class: None,
            replay_context: None,
        }
    }

    #[test]
    fn length_only_execution_tokens_are_rejected_without_replay_hashes() {
        let router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        let mut request = request(1, 7);
        request.replay_context = Some(ReplayRequestContext {
            agentic: None,
            authored_id: "length-only".into(),
            session_id: None,
            turn_index: None,
            metadata: Value::Null,
            prompt_token_source: ReplayPromptTokenSource::LengthOnlySynthetic,
        });

        let error =
            match router.build_pending_request(&request, request.max_output_tokens, None, None) {
                Ok(_) => panic!("length-only request unexpectedly reached KV placement"),
                Err(error) => error,
            };
        assert!(
            error
                .to_string()
                .contains("requires authored prompt token IDs or replay hashes"),
            "{error}"
        );
    }

    fn store_event(
        worker_id: WorkerId,
        event_id: u64,
        tokens_hash: u64,
        storage_tier: StorageTier,
    ) -> RouterEvent {
        store_event_for_rank(worker_id, 0, event_id, tokens_hash, storage_tier)
    }

    fn store_event_for_rank(
        worker_id: WorkerId,
        dp_rank: u32,
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
                dp_rank,
            },
            storage_tier,
        )
    }

    #[test]
    fn session_identity_reaches_scheduling_request() {
        let router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        let request = request(1, 7);
        let pending = router
            .build_pending_request(
                &request,
                request.max_output_tokens,
                None,
                Some("session-a".to_string()),
            )
            .unwrap();
        let scheduling_request = pending.scheduling_request(None);

        assert_eq!(
            scheduling_request
                .session_context
                .as_ref()
                .map(|context| context.session_id()),
            Some("session-a")
        );
    }

    #[test]
    fn keyed_replay_hashes_derive_tracking_identities_from_tokens() {
        let mut key_file = NamedTempFile::new().unwrap();
        key_file.write_all(&[0x39; 32]).unwrap();
        let config = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            router_tracking_key_file: Some(key_file.path().to_path_buf()),
            router_tracking_key_id: Some("2026-01".to_string()),
            ..Default::default()
        };
        let router = OfflineReplayRouter::new(&replay_args(), Some(config), None, 1).unwrap();
        let request = request(1, 7);
        let replay_hashes = ReplayRequestHashes::from_tokens(&request.tokens, router.block_size);
        let expected = router.config.compute_seq_hashes_for_tracking_with_context(
            &router.tracking_hash,
            router.tracking_hash_scope(),
            &request.tokens,
            None,
            BlockHashOptions::default(),
            None,
        );

        let pending = router
            .build_pending_request(
                &request,
                request.max_output_tokens,
                Some(replay_hashes.clone()),
                None,
            )
            .unwrap();

        assert_eq!(pending.token_seq, expected);
        assert_ne!(pending.token_seq, Some(replay_hashes.sequence_hashes));
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
    fn attention_dp_routes_one_engine_worker_across_rank_targets() {
        let mut args = queueing_args();
        args.dp_size = 2;
        let mut router =
            OfflineReplayRouter::new(&args, Some(queueing_router_config()), None, 1).unwrap();

        let first = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        let second = router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        let mut targets = first
            .admissions
            .into_iter()
            .chain(second.admissions)
            .map(|admission| admission.worker_idx)
            .collect::<Vec<_>>();
        targets.sort_unstable();

        assert_eq!(targets, vec![0, 1]);
        assert_eq!(router.pending_count(), 0);
    }

    #[test]
    fn attention_dp_kv_router_routes_cached_prefix_to_matching_rank() {
        let mut args = replay_args();
        args.dp_size = 2;
        let mut router = OfflineReplayRouter::new(&args, Some(router_config()), None, 1).unwrap();
        let target = request(1, 7);
        let hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);

        router
            .on_kv_events(vec![store_event_for_rank(
                0,
                1,
                1,
                hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();

        let effects = router
            .on_request_arrival(&target, Some(hashes), 0.0)
            .unwrap();
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 1,
                overlap_blocks: 1,
                best_available_overlap_blocks: 1,
                isl_blocks: 1,
            }]
        );
    }

    #[test]
    fn prompt_token_lookup_scores_indexed_prefix() {
        let mut router = OfflineReplayRouter::new(&replay_args(), Some(router_config()), None, 2)
            .expect("router construction");
        let target = request(1, 7);
        let local_hashes = compute_block_hash_for_seq(
            &target.tokens,
            router.block_size,
            BlockHashOptions::default(),
        );
        router
            .on_kv_events(vec![store_event(
                1,
                1,
                local_hashes[0].0,
                StorageTier::Device,
            )])
            .unwrap();

        let effects = router.on_request_arrival(&target, None, 0.0).unwrap();
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 1,
                overlap_blocks: 1,
                best_available_overlap_blocks: 1,
                isl_blocks: 1,
            }]
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
                cached_hashes.local_block_hashes[0],
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
        let dynamo_kv_router::scheduling::KvSchedulerError::QueueRejected(rejection) = error
            .downcast_ref::<dynamo_kv_router::scheduling::KvSchedulerError>()
            .expect("replay should preserve the native scheduler error")
        else {
            panic!("{error}")
        };
        assert_eq!(rejection.policy_class, "latency_uncached");
        assert_eq!(rejection.current, 0);
        assert_eq!(rejection.limit, 0);

        // The native actor rechecks old backlog when a worker becomes available.
        router.add_worker(1).unwrap();
        let released = router.on_topology_changed(0.0).unwrap();
        assert_eq!(released.admissions.len(), 1);
        assert_eq!(released.admissions[0].uuid, Uuid::from_u128(3));
        assert_eq!(router.pending_count(), 0);
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
                target_hashes.local_block_hashes[0],
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
    fn priorities_and_queue_order_reach_the_native_scheduler() {
        for (policy, priority, expected) in [
            (RouterQueuePolicy::Fcfs, 0, 2),
            (RouterQueuePolicy::Lcfs, 0, 3),
            (RouterQueuePolicy::Wspt, 0, 3),
            (RouterQueuePolicy::Fcfs, 1, 3),
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
                .on_request_arrival(&request_with_priorities(2, 2, 128, 0, 0), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(3, 3, 64, 0, priority), None, 1.0)
                .unwrap();
            let admitted = router
                .on_request_completed(Uuid::from_u128(1), 2.0)
                .unwrap();
            assert_eq!(admitted.admissions[0].uuid, Uuid::from_u128(expected));
        }
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
                best_available_overlap_blocks: 0,
                isl_blocks: 1,
            }]
        );
    }

    #[test]
    fn cache_telemetry_excludes_removed_workers() {
        let mut router =
            OfflineReplayRouter::new(&replay_args(), Some(router_config()), None, 2).unwrap();
        let target = request(1, 7);
        let hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                1,
                1,
                hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();
        router.remove_worker(1).unwrap();

        let effects = router
            .on_request_arrival(&target, Some(hashes), 0.0)
            .unwrap();
        assert_eq!(effects.admissions.len(), 1);
        assert_eq!(effects.admissions[0].worker_idx, 0);
        assert_eq!(effects.admissions[0].overlap_blocks, 0);
        assert_eq!(effects.admissions[0].best_available_overlap_blocks, 0);
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
                hashes.local_block_hashes[0],
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
                best_available_overlap_blocks: 0,
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
    fn removing_all_workers_preserves_native_no_endpoints_error() {
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
        let error = router
            .on_request_completed(Uuid::from_u128(1), 0.0)
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<dynamo_kv_router::scheduling::KvSchedulerError>(),
            Some(dynamo_kv_router::scheduling::KvSchedulerError::NoEndpoints)
        ));
        assert_eq!(router.pending_count(), 0);
    }

    fn affinity_placement(dp_size: u32, queued: bool, capture: bool) -> super::KvRouterPlacement {
        let mut args = queueing_args();
        args.dp_size = dp_size;
        let config = if queued {
            queueing_router_config()
        } else {
            KvRouterConfig {
                router_queue_threshold: None,
                ..Default::default()
            }
        };
        super::KvRouterPlacement::new_with_selector_seed(&args, Some(config), None, 1, None)
            .unwrap()
            .with_affinity(
                Some(super::ReplayAffinityConfig {
                    mode: super::ReplayAffinityMode::Session,
                    ttl_seconds: 1.0,
                }),
                "aggregated",
                capture,
                super::RoutingEvidence::default(),
            )
            .unwrap()
    }

    fn affinity_place(
        placement: &mut super::KvRouterPlacement,
        id: u128,
        session: &str,
        now: f64,
    ) -> aisimulate_core::replay::PlacementEffects {
        use aisimulate_core::replay::PlacementPolicy;
        placement
            .place(
                &request(id, 7),
                super::KvReplayMetadata::default(),
                Some(session.into()),
                now,
            )
            .unwrap()
    }

    fn immediate_target(effects: aisimulate_core::replay::PlacementEffects) -> usize {
        match effects.decision {
            aisimulate_core::replay::PlacementDecision::Immediate(placement) => {
                placement.scheduler_id
            }
            other => panic!("expected immediate admission, got {other:?}"),
        }
    }

    fn affinity_commit(placement: &mut super::KvRouterPlacement, id: u128, now: f64) {
        <super::KvRouterPlacement as aisimulate_core::replay::PlacementPolicy<DirectRequest>>::dispatch_committed(placement, Uuid::from_u128(id), now).unwrap();
    }

    fn affinity_tick(
        placement: &mut super::KvRouterPlacement,
        now: f64,
    ) -> Vec<aisimulate_core::replay::Placement> {
        <super::KvRouterPlacement as aisimulate_core::replay::PlacementPolicy<DirectRequest>>::advance_clock(
            placement, now,
        )
        .unwrap()
    }

    #[test]
    fn affinity_dispatch_abort_releases_initializer_and_same_time_waiter() {
        use aisimulate_core::replay::{PlacementDecision, PlacementPolicy};
        let mut placement = affinity_placement(2, false, true);
        immediate_target(affinity_place(&mut placement, 1, "conversation", 0.0));
        assert!(matches!(
            affinity_place(&mut placement, 2, "conversation", 0.0).decision,
            PlacementDecision::Queued
        ));
        <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::dispatch_aborted(
            &mut placement,
            Uuid::from_u128(1),
            0.0,
        )
        .unwrap();
        assert_eq!(
            <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::next_wakeup_ms(
                &placement
            ),
            Some(0.0)
        );
        let released = affinity_tick(&mut placement, 0.0);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(2));
        affinity_commit(&mut placement, 2, 0.0);
        let bound = immediate_target(affinity_place(&mut placement, 3, "conversation", 0.0));
        assert_eq!(bound, released[0].scheduler_id);
        affinity_commit(&mut placement, 3, 0.0);
        let evidence = placement
            .router
            .affinity
            .as_ref()
            .unwrap()
            .evidence
            .snapshot()
            .unwrap();
        assert_eq!(evidence["roles"]["aggregated"]["dispatch_aborts"], 1);
        assert_eq!(
            evidence["roles"]["aggregated"]["decisions"][2]["binding_reused"],
            true
        );
        assert_eq!(placement.router.pending_count(), 0);
    }

    #[test]
    fn affinity_exact_dp_queue_does_not_block_other_rank_and_rechecks_on_release() {
        use aisimulate_core::replay::PlacementDecision;
        let mut placement = affinity_placement(2, true, true);
        let target = immediate_target(affinity_place(&mut placement, 1, "a", 0.0));
        affinity_commit(&mut placement, 1, 0.0);
        assert!(matches!(
            affinity_place(&mut placement, 2, "a", 0.0).decision,
            PlacementDecision::Queued
        ));
        let other = immediate_target(affinity_place(&mut placement, 3, "b", 0.0));
        assert_ne!(target, other, "native queue must skip the busy Exact lane");
        affinity_commit(&mut placement, 3, 0.0);
        let released = placement
            .router
            .on_prefill_completed(Uuid::from_u128(1), 1.0)
            .unwrap();
        assert_eq!(released.admissions.len(), 1);
        assert_eq!(released.admissions[0].uuid, Uuid::from_u128(2));
        assert_eq!(released.admissions[0].worker_idx, target);
        affinity_commit(&mut placement, 2, 1.0);
        assert_eq!(placement.router.pending_count(), 0);
    }

    #[test]
    fn affinity_cancel_queued_initializer_allows_follower_to_initialize() {
        use aisimulate_core::replay::PlacementDecision;
        let mut placement = affinity_placement(1, true, true);
        immediate_target(affinity_place(&mut placement, 1, "busy", 0.0));
        affinity_commit(&mut placement, 1, 0.0);
        for id in [2, 3] {
            assert!(matches!(
                affinity_place(&mut placement, id, "waiting", 0.0).decision,
                PlacementDecision::Queued
            ));
        }
        assert!(placement.router.cancel_pending(Uuid::from_u128(2)));
        assert!(affinity_tick(&mut placement, 0.0).is_empty());
        let released = placement
            .router
            .on_request_completed(Uuid::from_u128(1), 1.0)
            .unwrap();
        assert_eq!(released.admissions.len(), 1);
        assert_eq!(released.admissions[0].uuid, Uuid::from_u128(3));
        affinity_commit(&mut placement, 3, 1.0);
        assert_eq!(placement.router.pending_count(), 0);
    }

    #[test]
    fn affinity_queued_bound_lease_outlives_ttl_then_idle_binding_expires() {
        use aisimulate_core::replay::PlacementDecision;
        let mut placement = affinity_placement(1, true, true);
        immediate_target(affinity_place(&mut placement, 1, "a", 0.0));
        affinity_commit(&mut placement, 1, 0.0);
        assert!(matches!(
            affinity_place(&mut placement, 2, "a", 0.0).decision,
            PlacementDecision::Queued
        ));
        let released = placement
            .router
            .on_request_completed(Uuid::from_u128(1), 2000.0)
            .unwrap();
        assert_eq!(released.admissions.len(), 1);
        affinity_commit(&mut placement, 2, 2000.0);
        placement
            .router
            .on_request_completed(Uuid::from_u128(2), 2001.0)
            .unwrap();
        immediate_target(affinity_place(&mut placement, 3, "a", 3002.0));
        let evidence = placement
            .router
            .affinity
            .as_ref()
            .unwrap()
            .evidence
            .snapshot()
            .unwrap();
        let decisions = evidence["roles"]["aggregated"]["decisions"]
            .as_array()
            .unwrap();
        assert_eq!(decisions[1]["binding_reused"], true);
        assert_eq!(decisions[2]["binding_reused"], false);
    }

    #[test]
    fn affinity_clock_errors_preserve_time_and_uncaptured_evidence_keeps_counters() {
        let mut placement = affinity_placement(1, false, false);
        immediate_target(affinity_place(&mut placement, 1, "a", 1.0));
        affinity_commit(&mut placement, 1, 1.0);
        let affinity = placement.router.affinity.as_mut().unwrap();
        for invalid in [f64::MAX, f64::NAN, 0.0] {
            assert!(affinity.advance(invalid).is_err());
            assert_eq!(affinity.now_ms, 1.0);
        }
        let evidence = affinity.evidence.snapshot().unwrap();
        assert_eq!(evidence["roles"]["aggregated"]["decision_count"], 1);
        assert_eq!(evidence["roles"]["aggregated"]["post_dispatch_checks"], 1);
        assert_eq!(
            evidence["roles"]["aggregated"]["decisions"],
            serde_json::json!([])
        );
    }
}

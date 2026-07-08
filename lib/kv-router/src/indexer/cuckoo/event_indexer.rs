// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use rustc_hash::FxHashMap;

#[cfg(feature = "metrics")]
use crate::indexer::PreBoundCkfSearchCounters;
#[cfg(feature = "bench")]
use crate::indexer::WorkerObservationState;
use crate::indexer::{
    EventKind, EventWarningKind, KvIndexerMetrics, PreBoundEventCounters, SyncIndexer,
    WorkerLookupStats, WorkerTask,
};
use crate::protocols::{
    KvCacheEventData, KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, WorkerId,
    WorkerWithDpRank, compute_seq_hash_for_block,
};

use super::addressing::CkfAddressing;
use super::bucket::{CuckooBucketStore, PackedBucket, TransposedCkfTable};
use super::mutator::{CuckooMutator, DcWriterState, DirtyBucket, lane_rng_seed};
#[cfg(not(feature = "metrics"))]
use super::search::find_prefix_depths;
#[cfg(feature = "metrics")]
use super::search::find_prefix_depths_with_stats;
#[cfg(any(test, feature = "bench"))]
use super::search::linear_prefix_depths;
use super::{CkfBuildError, CkfConfig, DC_COUNT, MAX_KICKS, MAX_VERIFICATION_WINDOW};

/// Fixed-D=16 event-driven transposed Cuckoo-filter indexer.
#[derive(Debug)]
pub struct EventTransposedCkfIndexer {
    pub(super) table: TransposedCkfTable<DC_COUNT>,
    pub(super) addressing: CkfAddressing,
    workers: [WorkerWithDpRank; DC_COUNT],
    worker_to_lane: FxHashMap<WorkerWithDpRank, usize>,
    worker_lane_masks: FxHashMap<WorkerId, u16>,
    pub(super) config: CkfConfig,
    #[cfg(feature = "metrics")]
    search_counters: Option<PreBoundCkfSearchCounters>,
}

impl EventTransposedCkfIndexer {
    /// Construct an indexer with one immutable lane assignment per worker/rank.
    pub fn new(
        workers: [WorkerWithDpRank; DC_COUNT],
        config: CkfConfig,
    ) -> Result<Self, CkfBuildError> {
        validate_config(config)?;
        let bucket_count = bucket_count(config.expected_blocks_per_dc)?;

        let mut worker_to_lane = FxHashMap::default();
        worker_to_lane
            .try_reserve(DC_COUNT)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        let mut worker_lane_masks = FxHashMap::default();
        worker_lane_masks
            .try_reserve(DC_COUNT)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        for (lane, worker) in workers.iter().copied().enumerate() {
            if worker_to_lane.insert(worker, lane).is_some() {
                return Err(CkfBuildError::DuplicateWorker { worker });
            }
            *worker_lane_masks.entry(worker.worker_id).or_insert(0) |= 1u16 << lane;
        }

        Ok(Self {
            table: TransposedCkfTable::new(bucket_count)?,
            addressing: CkfAddressing::new(bucket_count, config.seed),
            workers,
            worker_to_lane,
            worker_lane_masks,
            config,
            #[cfg(feature = "metrics")]
            search_counters: None,
        })
    }

    pub(super) fn apply_event(
        &self,
        states: &mut [Option<DcWriterState>; DC_COUNT],
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        self.apply_event_with_dirty(states, event, counters, |_, _| {})
    }

    pub(super) fn apply_event_with_dirty(
        &self,
        states: &mut [Option<DcWriterState>; DC_COUNT],
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
        mut on_dirty: impl FnMut(usize, DirtyBucket),
    ) -> Result<(), KvCacheEventError> {
        let worker_id = event.worker_id;
        let event_id = event.event.event_id;
        let worker = WorkerWithDpRank::new(worker_id, event.event.dp_rank);
        match event.event.data {
            KvCacheEventData::Stored(store) => {
                let lane = self.lane_for(worker, event_id)?;
                let mut first_error = None;
                let mut entirely_duplicate = !store.blocks.is_empty();
                for block in store.blocks {
                    let state = match self.ensure_state(states, lane) {
                        Ok(state) => state,
                        Err(error) => {
                            entirely_duplicate = false;
                            retain_first_error(&mut first_error, error);
                            continue;
                        }
                    };
                    if !state.resident.insert(block.block_hash) {
                        continue;
                    }
                    entirely_duplicate = false;
                    let view = self.table.lane(lane);
                    let mutator =
                        CuckooMutator::new(&view, &self.addressing, self.config.max_kicks);
                    let result = mutator.insert(
                        block.block_hash,
                        &mut state.rng,
                        &mut state.scratch,
                        |dirty| on_dirty(lane, dirty),
                    );
                    if let Err(error) = result {
                        let removed = state.resident.remove(&block.block_hash);
                        debug_assert!(removed);
                        retain_first_error(&mut first_error, error);
                    }
                }
                if entirely_duplicate && let Some(counters) = counters {
                    counters.inc_warning(EventWarningKind::DuplicateStore);
                }
                finish_event(first_error)
            }
            KvCacheEventData::Removed(remove) => {
                let lane = self.lane_for(worker, event_id)?;
                let mut first_error = None;
                for hash in remove.block_hashes {
                    let Some(state) = states[lane].as_mut() else {
                        retain_first_error(&mut first_error, KvCacheEventError::BlockNotFound);
                        continue;
                    };
                    if !state.resident.remove(&hash) {
                        retain_first_error(&mut first_error, KvCacheEventError::BlockNotFound);
                        continue;
                    }
                    let view = self.table.lane(lane);
                    let mutator =
                        CuckooMutator::new(&view, &self.addressing, self.config.max_kicks);
                    match mutator.remove(hash, |dirty| on_dirty(lane, dirty)) {
                        Ok(()) => {}
                        Err(error) => {
                            let inserted = state.resident.insert(hash);
                            debug_assert!(inserted);
                            retain_first_error(&mut first_error, error);
                        }
                    }
                }
                finish_event(first_error)
            }
            KvCacheEventData::Cleared => {
                let Some(mask) = self.worker_lane_masks.get(&worker_id).copied() else {
                    tracing::warn!(
                        worker_id,
                        event_id,
                        "CKF event references an unknown worker"
                    );
                    return Err(KvCacheEventError::InvalidBlockSequence);
                };
                for (lane, state) in states.iter_mut().enumerate() {
                    if mask & (1u16 << lane) == 0 {
                        continue;
                    }
                    self.clear_lane(lane, |dirty| on_dirty(lane, dirty));
                    *state = None;
                }
                Ok(())
            }
        }
    }

    fn lane_for(
        &self,
        worker: WorkerWithDpRank,
        event_id: u64,
    ) -> Result<usize, KvCacheEventError> {
        let Some(lane) = self.worker_to_lane.get(&worker).copied() else {
            tracing::warn!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                event_id,
                "CKF event references an unknown worker/rank"
            );
            return Err(KvCacheEventError::InvalidBlockSequence);
        };
        Ok(lane)
    }

    fn ensure_state<'a>(
        &self,
        states: &'a mut [Option<DcWriterState>; DC_COUNT],
        lane: usize,
    ) -> Result<&'a mut DcWriterState, KvCacheEventError> {
        if states[lane].is_none() {
            states[lane] = Some(DcWriterState::new(
                self.config.expected_blocks_per_dc,
                self.config.max_kicks,
                lane_rng_seed(self.config.seed, lane),
            )?);
        }
        states[lane]
            .as_mut()
            .ok_or(KvCacheEventError::IndexerInvariantViolation)
    }

    fn clear_lane(&self, lane: usize, mut on_dirty: impl FnMut(DirtyBucket)) {
        let view = self.table.lane(lane);
        for bucket in 0..view.bucket_count() {
            let before = view.load_bucket(bucket);
            if before == PackedBucket::default() {
                continue;
            }
            view.store_bucket(bucket, PackedBucket::default());
            on_dirty(DirtyBucket {
                bucket,
                value: PackedBucket::default(),
            });
        }
    }

    fn clear_worker(&self, states: &mut [Option<DcWriterState>; DC_COUNT], worker_id: WorkerId) {
        let Some(mask) = self.worker_lane_masks.get(&worker_id).copied() else {
            return;
        };
        for (lane, state) in states.iter_mut().enumerate() {
            if mask & (1u16 << lane) != 0 {
                self.clear_lane(lane, |_| {});
                *state = None;
            }
        }
    }

    fn clear_worker_rank(
        &self,
        states: &mut [Option<DcWriterState>; DC_COUNT],
        worker: WorkerWithDpRank,
    ) {
        let Some(lane) = self.worker_to_lane.get(&worker).copied() else {
            return;
        };
        self.clear_lane(lane, |_| {});
        states[lane] = None;
    }

    fn worker_stats(&self, states: &[Option<DcWriterState>; DC_COUNT]) -> WorkerLookupStats {
        WorkerLookupStats::from_worker_block_counts(states.iter().enumerate().filter_map(
            |(lane, state)| {
                state
                    .as_ref()
                    .map(|state| (self.workers[lane], state.resident.len()))
            },
        ))
    }

    pub(super) fn prepared_probes(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Vec<super::addressing::CkfProbe> {
        compute_seq_hash_for_block(sequence)
            .into_iter()
            .map(|hash| self.addressing.prepare(hash))
            .collect()
    }

    #[cfg(any(test, feature = "bench"))]
    #[allow(dead_code)]
    pub(super) fn linear_depths(&self, probes: &[super::addressing::CkfProbe]) -> [u32; DC_COUNT] {
        linear_prefix_depths(probes.len(), u16::MAX, |position| {
            self.table.probe(probes[position])
        })
    }
}

impl SyncIndexer for EventTransposedCkfIndexer {
    fn configure_metrics(&mut self, metrics: Option<&KvIndexerMetrics>) {
        #[cfg(feature = "metrics")]
        {
            self.search_counters = metrics.map(KvIndexerMetrics::prebind_ckf_search);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = metrics;
    }

    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let mut states: [Option<DcWriterState>; DC_COUNT] = std::array::from_fn(|_| None);
        let counters = metrics.as_ref().map(|metrics| metrics.prebind());
        #[cfg(feature = "bench")]
        let mut observation = WorkerObservationState::default();

        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(event) => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut states, event, counters.as_ref());
                    record_worker_event_result(counters.as_ref(), kind, result);
                }
                WorkerTask::EventWithAck { event, resp } => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut states, event, counters.as_ref());
                    let applied = record_worker_event_result(counters.as_ref(), kind, result);
                    let _ = resp.send(applied);
                }
                #[cfg(feature = "bench")]
                WorkerTask::InstallObservation { writer, resp } => {
                    observation.install(writer, resp);
                }
                #[cfg(feature = "bench")]
                WorkerTask::ObservedEvent {
                    event,
                    correlation_id,
                } => {
                    let kind = EventKind::of(&event.event.data);
                    let result = self.apply_event(&mut states, event, counters.as_ref());
                    let applied = record_worker_event_result(counters.as_ref(), kind, result);
                    observation.record(correlation_id, applied);
                }
                #[cfg(feature = "bench")]
                WorkerTask::SealObservation(resp) => observation.seal(resp),
                #[cfg(feature = "bench")]
                WorkerTask::HarvestObservation(resp) => observation.harvest(resp),
                WorkerTask::Anchor { worker, anchor } => {
                    if let Err(error) = self.apply_anchor(worker, anchor) {
                        tracing::warn!(?error, "CKF does not support structural anchors");
                    }
                }
                WorkerTask::RemoveWorker { worker_id, .. } => {
                    self.clear_worker(&mut states, worker_id);
                }
                WorkerTask::RemoveWorkerDpRank {
                    worker_id, dp_rank, ..
                } => {
                    self.clear_worker_rank(&mut states, WorkerWithDpRank::new(worker_id, dp_rank));
                }
                WorkerTask::CleanupStaleChildren => {}
                WorkerTask::DumpEvents(sender) => {
                    let _ = sender.send(Err(anyhow::anyhow!(
                        "CKF cannot reconstruct stored router events"
                    )));
                }
                WorkerTask::Stats(sender) => {
                    let _ = sender.send(self.worker_stats(&states));
                }
                WorkerTask::Flush(sender) => {
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => break,
            }
        }

        tracing::debug!("EventTransposedCkfIndexer worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
        if sequence.is_empty() {
            return OverlapScores::new();
        }

        let probes = self.prepared_probes(sequence);
        #[cfg(not(feature = "metrics"))]
        let depths = find_prefix_depths::<DC_COUNT>(
            probes.len(),
            u16::MAX,
            self.config.search.verification_window,
            |position| self.table.prefetch_probe(probes[position]),
            |position| self.table.probe(probes[position]),
        );
        #[cfg(feature = "metrics")]
        let depths = {
            let result = find_prefix_depths_with_stats::<DC_COUNT>(
                probes.len(),
                u16::MAX,
                self.config.search.verification_window,
                |position| self.table.prefetch_probe(probes[position]),
                |position| self.table.probe(probes[position]),
            );
            if let Some(counters) = &self.search_counters {
                counters.record(
                    result.fallback.left_edge_lanes,
                    result.fallback.activated_lanes,
                    result.fallback.probe_calls,
                    result.fallback.lane_probes,
                    result.fallback.provenance_skips,
                );
            }
            result.depths
        };

        let mut scores = OverlapScores::new();
        scores
            .scores
            .reserve(depths.iter().filter(|&&depth| depth > 0).count());
        for (lane, depth) in depths.into_iter().enumerate() {
            if depth > 0 {
                scores.scores.insert(self.workers[lane], depth);
            }
        }
        scores
    }

    fn supports_event_dump(&self) -> bool {
        false
    }

    fn supports_routing_decision_pruning(&self) -> bool {
        false
    }
}

fn validate_config(config: CkfConfig) -> Result<(), CkfBuildError> {
    if config.expected_blocks_per_dc == 0 {
        return Err(CkfBuildError::ExpectedCapacityZero);
    }
    if !(1..=MAX_KICKS).contains(&config.max_kicks) {
        return Err(CkfBuildError::InvalidMaxKicks {
            value: config.max_kicks,
            maximum: MAX_KICKS,
        });
    }
    if !(1..=MAX_VERIFICATION_WINDOW).contains(&config.search.verification_window) {
        return Err(CkfBuildError::InvalidVerificationWindow {
            value: config.search.verification_window,
        });
    }
    Ok(())
}

pub(super) fn bucket_count(expected_blocks_per_dc: usize) -> Result<usize, CkfBuildError> {
    let numerator = expected_blocks_per_dc
        .checked_mul(5)
        .and_then(|value| value.checked_add(15))
        .ok_or(CkfBuildError::CapacityOverflow)?;
    let required = (numerator / 16).max(2);
    required
        .checked_next_power_of_two()
        .ok_or(CkfBuildError::CapacityOverflow)
}

fn retain_first_error(slot: &mut Option<KvCacheEventError>, error: KvCacheEventError) {
    if slot.is_none() {
        *slot = Some(error);
    }
}

fn finish_event(error: Option<KvCacheEventError>) -> Result<(), KvCacheEventError> {
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn record_worker_event_result(
    counters: Option<&PreBoundEventCounters>,
    kind: EventKind,
    result: Result<(), KvCacheEventError>,
) -> bool {
    let applied = result.is_ok();
    if let Err(error) = result.as_ref() {
        tracing::warn!(?error, "Failed to apply CKF event");
    }
    if let Some(counters) = counters {
        counters.inc(kind, result);
    }
    applied
}

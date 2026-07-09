// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use rustc_hash::FxHashSet;

#[cfg(feature = "metrics")]
use crate::indexer::PreBoundCkfSearchCounters;
#[cfg(feature = "bench")]
use crate::indexer::WorkerObservationState;
use crate::indexer::{
    CkfMutationKind, EventKind, EventWarningKind, KvIndexerMetrics, PreBoundEventCounters,
    SyncIndexer, WorkerTask,
};
#[cfg(test)]
use crate::protocols::RouterEvent;
use crate::protocols::{KvCacheEventError, LocalBlockHash, OverlapScores, WorkerWithDpRank};

#[cfg(test)]
use super::CkfDeltaBatch;
#[cfg(feature = "bench")]
use super::relay::CkfBenchLocalTelemetry;
#[cfg(not(feature = "metrics"))]
use super::search::find_prefix_depths;
#[cfg(feature = "metrics")]
use super::search::find_prefix_depths_with_stats;
#[cfg(any(test, feature = "bench"))]
use super::search::linear_prefix_depths;
use super::{
    CkfBuildError, CkfConfig, CkfEventOutcome, DC_COUNT, RelayManifest, RouterLocalCkfPipeline,
};

#[cfg(test)]
pub(super) use super::bucket_count;

/// Fixed-D=16 event-driven transposed Cuckoo-filter indexer.
#[derive(Debug)]
pub struct EventTransposedCkfIndexer {
    pub(super) pipeline: RouterLocalCkfPipeline,
    workers: [WorkerWithDpRank; DC_COUNT],
    #[cfg(feature = "metrics")]
    search_counters: Option<PreBoundCkfSearchCounters>,
}

impl EventTransposedCkfIndexer {
    /// Construct the normal one-worker/rank-per-lane specialization.
    pub fn new(
        workers: [WorkerWithDpRank; DC_COUNT],
        config: CkfConfig,
    ) -> Result<Self, CkfBuildError> {
        let manifest = RelayManifest::one_worker_per_lane(workers, config.expected_blocks_per_dc)?;
        Ok(Self {
            pipeline: RouterLocalCkfPipeline::new(manifest, config)?,
            workers,
            #[cfg(feature = "metrics")]
            search_counters: None,
        })
    }

    #[cfg(test)]
    pub(super) fn apply_event(
        &self,
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        let kind = EventKind::of(&event.event.data);
        let mut batch = self.pipeline.new_batch();
        let outcome = self.pipeline.apply_event(event, &mut batch);
        record_worker_event_result(counters, kind, outcome)
    }

    #[cfg(test)]
    pub(super) fn apply_event_with_batch(
        &self,
        event: RouterEvent,
        batch: &mut CkfDeltaBatch,
    ) -> CkfEventOutcome {
        self.pipeline.apply_event(event, batch)
    }

    #[cfg(test)]
    pub(super) fn prepared_probes(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Vec<super::addressing::CkfProbe> {
        self.pipeline.replica().prepared_probes(sequence)
    }

    #[cfg(any(test, feature = "bench"))]
    #[allow(dead_code)]
    pub(super) fn linear_depths(&self, probes: &[super::addressing::CkfProbe]) -> [u32; DC_COUNT] {
        linear_prefix_depths(probes.len(), u16::MAX, |position| {
            self.pipeline.replica().table.probe(probes[position])
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
        let counters = metrics.as_ref().map(|metrics| metrics.prebind());
        let mut batch = self.pipeline.new_batch();
        let mut seen_worker_ids = FxHashSet::default();
        #[cfg(feature = "bench")]
        let mut observation = WorkerObservationState::default();
        #[cfg(feature = "bench")]
        let mut bench_telemetry = CkfBenchLocalTelemetry::default();

        while let Ok(task) = event_receiver.recv() {
            match task {
                WorkerTask::Event(event) => {
                    seen_worker_ids.insert(event.worker_id);
                    let kind = EventKind::of(&event.event.data);
                    let outcome = self.pipeline.apply_event(event, &mut batch);
                    #[cfg(feature = "bench")]
                    bench_telemetry.absorb_outcome(&outcome);
                    let _ = record_worker_event_result(counters.as_ref(), kind, outcome);
                }
                WorkerTask::EventWithAck { event, resp } => {
                    seen_worker_ids.insert(event.worker_id);
                    let kind = EventKind::of(&event.event.data);
                    let outcome = self.pipeline.apply_event(event, &mut batch);
                    #[cfg(feature = "bench")]
                    bench_telemetry.absorb_outcome(&outcome);
                    let applied =
                        record_worker_event_result(counters.as_ref(), kind, outcome).is_ok();
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
                    seen_worker_ids.insert(event.worker_id);
                    let kind = EventKind::of(&event.event.data);
                    let outcome = self.pipeline.apply_event(event, &mut batch);
                    bench_telemetry.absorb_outcome(&outcome);
                    let applied =
                        record_worker_event_result(counters.as_ref(), kind, outcome).is_ok();
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
                    seen_worker_ids.insert(worker_id);
                    let outcome = self.pipeline.remove_worker(worker_id, &mut batch);
                    #[cfg(feature = "bench")]
                    bench_telemetry.absorb_outcome(&outcome);
                    record_control_result(counters.as_ref(), outcome);
                }
                WorkerTask::RemoveWorkerDpRank {
                    worker_id, dp_rank, ..
                } => {
                    seen_worker_ids.insert(worker_id);
                    let outcome = self
                        .pipeline
                        .remove_worker_rank(WorkerWithDpRank::new(worker_id, dp_rank), &mut batch);
                    #[cfg(feature = "bench")]
                    bench_telemetry.absorb_outcome(&outcome);
                    record_control_result(counters.as_ref(), outcome);
                }
                WorkerTask::CleanupStaleChildren => {}
                WorkerTask::DumpEvents(sender) => {
                    let _ = sender.send(Err(anyhow::anyhow!(
                        "CKF cannot reconstruct stored router events"
                    )));
                }
                WorkerTask::Stats(sender) => {
                    let _ = sender.send(self.pipeline.worker_stats(&seen_worker_ids));
                }
                WorkerTask::Flush(sender) => {
                    #[cfg(feature = "bench")]
                    let result = self
                        .pipeline
                        .flush_pending_with_telemetry(&mut batch, &mut bench_telemetry);
                    #[cfg(not(feature = "bench"))]
                    let result = self.pipeline.flush_pending(&mut batch);
                    if let Err(error) = result {
                        tracing::warn!(?error, "Failed to flush pending CKF publication");
                    }
                    #[cfg(feature = "bench")]
                    self.pipeline.merge_bench_telemetry(&mut bench_telemetry);
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => {
                    #[cfg(feature = "bench")]
                    let result = self
                        .pipeline
                        .flush_pending_with_telemetry(&mut batch, &mut bench_telemetry);
                    #[cfg(not(feature = "bench"))]
                    let result = self.pipeline.flush_pending(&mut batch);
                    if let Err(error) = result {
                        tracing::warn!(?error, "Failed to flush CKF publication at termination");
                    }
                    #[cfg(feature = "bench")]
                    self.pipeline.merge_bench_telemetry(&mut bench_telemetry);
                    break;
                }
            }
        }

        #[cfg(feature = "bench")]
        let result = self
            .pipeline
            .flush_pending_with_telemetry(&mut batch, &mut bench_telemetry);
        #[cfg(not(feature = "bench"))]
        let result = self.pipeline.flush_pending(&mut batch);
        if let Err(error) = result {
            tracing::warn!(
                ?error,
                "Failed to flush CKF publication at receiver closure"
            );
        }
        #[cfg(feature = "bench")]
        self.pipeline.merge_bench_telemetry(&mut bench_telemetry);

        tracing::debug!("EventTransposedCkfIndexer worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
        if sequence.is_empty() {
            return OverlapScores::new();
        }

        let replica = self.pipeline.replica();
        let probes = replica.prepared_probes(sequence);
        #[cfg(not(feature = "metrics"))]
        let depths = find_prefix_depths::<DC_COUNT>(
            probes.len(),
            u16::MAX,
            replica.config.search.verification_window,
            |position| replica.table.prefetch_probe(probes[position]),
            |position| replica.table.probe(probes[position]),
        );
        #[cfg(feature = "metrics")]
        let depths = {
            let result = find_prefix_depths_with_stats::<DC_COUNT>(
                probes.len(),
                u16::MAX,
                replica.config.search.verification_window,
                |position| replica.table.prefetch_probe(probes[position]),
                |position| replica.table.probe(probes[position]),
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

    fn timing_report(&self) -> String {
        #[cfg(feature = "bench")]
        {
            self.pipeline.timing_report()
        }
        #[cfg(not(feature = "bench"))]
        {
            String::new()
        }
    }
}

fn record_worker_event_result(
    counters: Option<&PreBoundEventCounters>,
    kind: EventKind,
    outcome: CkfEventOutcome,
) -> Result<(), KvCacheEventError> {
    if let Some(counters) = counters {
        counters.inc_ckf_mutation(CkfMutationKind::UnknownRemove, outcome.unknown_removals());
        counters.inc_ckf_mutation(
            CkfMutationKind::CapacityExhausted,
            outcome.capacity_failures(),
        );
        if outcome.duplicate_warning() {
            counters.inc_warning(EventWarningKind::DuplicateStore);
        }
    }
    let result = outcome.into_result();
    if let Err(error) = result.as_ref() {
        tracing::warn!(?error, "Failed to apply CKF event");
    }
    if let Some(counters) = counters {
        counters.inc(kind, result);
    }
    result
}

fn record_control_result(counters: Option<&PreBoundEventCounters>, outcome: CkfEventOutcome) {
    if let Some(counters) = counters {
        counters.inc_ckf_mutation(
            CkfMutationKind::CapacityExhausted,
            outcome.capacity_failures(),
        );
    }
    if let Err(error) = outcome.into_result() {
        tracing::warn!(?error, "Failed to apply CKF lifecycle operation");
    }
}

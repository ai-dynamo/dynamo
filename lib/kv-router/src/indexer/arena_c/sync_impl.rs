// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The lane loop: CRTC's, minus the graveyard, plus the free batch and pending unlinks.

use super::types::WorkerRemovalTarget;
use super::*;
#[cfg(feature = "bench")]
use crate::indexer::WorkerObservationState;
use crate::indexer::{
    EventKind, KvIndexerMetrics, KvRouterError, PreBoundEventCounters, SyncIndexer,
    WorkerLookupStats, WorkerTask,
};

impl ArenaIndexC {
    pub(super) fn apply_event(
        &self,
        lane: &mut CLane,
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let id = event.event.event_id;
        let result = match event.event.data {
            KvCacheEventData::Stored(op) => self.apply_stored(lane, rank, op, id, counters),
            KvCacheEventData::Removed(op) => self.apply_removed(lane, rank, op, id),
            KvCacheEventData::Cleared => {
                self.apply_cleared(lane, rank);
                Ok(())
            }
        };
        lane.tally.maybe_flush(&self.reclaim);
        result
    }

    fn record_event(
        &self,
        lane: &mut CLane,
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
    ) -> bool {
        let kind = EventKind::of(&event.event.data);
        let result = self.apply_event(lane, event, counters);
        if let Err(error) = &result {
            tracing::warn!("Failed to apply event: {error:?}");
        }
        let applied = result.is_ok();
        if let Some(counters) = counters {
            counters.inc(kind, result);
        }
        applied
    }

    /// Idle work: hand frees to the epoch, publish the tally, then retry pending unlinks
    /// in chunks while `idle` holds.
    fn on_idle(&self, lane: &mut CLane, idle: impl Fn() -> bool) {
        self.flush_frees(lane);
        lane.tally.flush(&self.reclaim);
        while idle() && self.retry_pending(lane) {}
        self.flush_frees(lane);
    }
}

impl SyncIndexer for ArenaIndexC {
    #[cfg_attr(feature = "profile", inline(never))]
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        let mut lane = self.new_lane();
        let counters = metrics.as_ref().map(|m| m.prebind());
        #[cfg(feature = "bench")]
        let mut observation = WorkerObservationState::default();

        loop {
            if event_receiver.is_empty() {
                self.on_idle(&mut lane, || event_receiver.is_empty());
            }
            let Ok(task) = event_receiver.recv() else {
                break;
            };
            match task {
                WorkerTask::Event(event) => {
                    self.record_event(&mut lane, event, counters.as_ref());
                }
                WorkerTask::EventWithAck { event, resp } => {
                    let applied = self.record_event(&mut lane, event, counters.as_ref());
                    let _ = resp.send(applied);
                }
                WorkerTask::ApproximateLru(task) => {
                    if let Some(response) = task.response {
                        let _ = response.send(Err(KvRouterError::Unsupported(
                            "arena-c does not support approximate LRU".to_string(),
                        )));
                    }
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
                    let applied = self.record_event(&mut lane, event, counters.as_ref());
                    observation.record(correlation_id, applied);
                }
                #[cfg(feature = "bench")]
                WorkerTask::SealObservation(resp) => observation.seal(resp),
                #[cfg(feature = "bench")]
                WorkerTask::HarvestObservation(resp) => observation.harvest(resp),
                WorkerTask::Anchor { .. } => {
                    tracing::warn!("arena-c does not support branch anchors; ignoring");
                }
                WorkerTask::RemoveWorker {
                    worker_id,
                    sweep_tree,
                    resp,
                } => {
                    self.remove_ranks(
                        &mut lane,
                        WorkerRemovalTarget::WorkerId(worker_id),
                        sweep_tree,
                    );
                    let _ = resp.send(());
                }
                WorkerTask::RemoveWorkerDpRank {
                    worker_id,
                    dp_rank,
                    sweep_tree,
                } => {
                    self.remove_ranks(
                        &mut lane,
                        WorkerRemovalTarget::DpRank(WorkerWithDpRank::new(worker_id, dp_rank)),
                        sweep_tree,
                    );
                }
                WorkerTask::CleanupStaleChildren => self.run_cleanup_on(&mut lane),
                WorkerTask::DumpEvents(sender) => {
                    let _ = sender.send(Ok(Vec::new()));
                }
                WorkerTask::Stats(sender) => {
                    let stats = WorkerLookupStats::from_worker_block_counts(lane.block_counts());
                    let _ = sender.send(stats);
                }
                WorkerTask::ContainsWorkerBlock {
                    worker,
                    block_hash,
                    resp,
                } => {
                    let resident = lane
                        .ranks
                        .get(&worker)
                        .is_some_and(|lookup| lookup.map.contains_key(block_hash));
                    let _ = resp.send(resident);
                }
                WorkerTask::Flush(sender) => {
                    self.flush_frees(&mut lane);
                    lane.tally.flush(&self.reclaim);
                    crossbeam_epoch::pin().flush();
                    let _ = sender.send(());
                }
                WorkerTask::Terminate => break,
            }
        }

        self.flush_frees(&mut lane);
        lane.tally.flush(&self.reclaim);
        tracing::debug!("arena-c worker thread shutting down");
        Ok(())
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.find_matches_impl(sequence, early_exit)
    }

    fn supports_routing_decision_pruning(&self) -> bool {
        false
    }

    fn try_schedule_cleanup(&self) -> bool {
        self.schedule_cleanup()
    }

    fn cancel_scheduled_cleanup(&self) {
        self.cancel_cleanup();
    }

    fn run_cleanup_task(&self) {
        let mut lane = self.new_lane();
        self.run_cleanup_on(&mut lane);
        // Unlinks this sweep could not take are left for the next sweep.
        self.forget_pending(&mut lane);
        lane.tally.flush(&self.reclaim);
        self.flush_frees(&mut lane);
    }

    fn dump_events(&self) -> Option<Vec<RouterEvent>> {
        Some(self.dump_tree_as_events())
    }

    fn timing_report(&self) -> String {
        #[cfg(feature = "bench")]
        {
            self.shape_report().to_string()
        }
        #[cfg(not(feature = "bench"))]
        {
            String::new()
        }
    }

    fn node_count(&self) -> usize {
        self.store.slab.allocated() as usize
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        self.run_lengths()
    }
}

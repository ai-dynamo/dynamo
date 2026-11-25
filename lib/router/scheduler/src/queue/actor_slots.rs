// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use rustc_hash::FxHashMap;
use tokio::time::Instant;

use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_slot_tracker::host::DecodeProgress;
use dynamo_slot_tracker::{
    ActiveSequencesMultiWorker, FullRankCounts, LifecycleMutationOutcome, RequestId, SequenceError,
    SequencePublisher, SequenceRequest, WorkerLoadProjection, WorkerLoadReader,
};
use dynamo_tokens::SequenceHash;

use super::ActorWriteLog;
use crate::types::AttemptId;

/// The actor's handle on the slot tracker. Its bookings and releases go through here and
/// are counted in the write log, so a load a caller projected can be brought up to date
/// with them; the reads forward unchanged. A release a caller applies on its own task
/// (`DirectBookingRelease`) is not counted: the actor does not re-read that rank.
///
/// The tracker is private to this module, so the actor cannot write to it any other way.
pub(super) struct ActorSlots<P: SequencePublisher> {
    tracker: Arc<ActiveSequencesMultiWorker<P>>,
    write_log: ActorWriteLog,
}

impl<P: SequencePublisher + 'static> ActorSlots<P> {
    pub(super) fn new(
        tracker: Arc<ActiveSequencesMultiWorker<P>>,
        actor_writes: Arc<AtomicU64>,
    ) -> Self {
        Self {
            tracker,
            write_log: ActorWriteLog::new(actor_writes),
        }
    }

    /// Book `request` on its rank.
    pub(super) fn add_request_admitted(
        &self,
        request: SequenceRequest,
        decay_now: Instant,
    ) -> Result<AttemptId, SequenceError> {
        let worker = request.worker;
        let attempt_id = self.tracker.add_request_admitted(request, decay_now)?;
        self.write_log.record(worker);
        Ok(attempt_id)
    }

    /// Free the booking the attempt holds, if it still does.
    pub(super) fn free_if_booking(
        &self,
        request_id: &RequestId,
        worker: WorkerWithDpRank,
        attempt_id: AttemptId,
        decay_now: Instant,
    ) -> Result<LifecycleMutationOutcome, SequenceError> {
        let outcome = self
            .tracker
            .free_if_booking(request_id, worker, attempt_id, decay_now)?;
        if outcome.is_applied() {
            self.write_log.record(worker);
        }
        Ok(outcome)
    }

    /// Expire the booking the attempt holds, if it still does.
    pub(super) fn expire_if_booking(
        &self,
        request_id: &RequestId,
        worker: WorkerWithDpRank,
        attempt_id: AttemptId,
        decay_now: Instant,
    ) -> Result<LifecycleMutationOutcome, SequenceError> {
        let outcome = self
            .tracker
            .expire_if_booking(request_id, worker, attempt_id, decay_now)?;
        if outcome.is_applied() {
            self.write_log.record(worker);
        }
        Ok(outcome)
    }

    /// The rank of each write after the first `since`; see [`ActorWriteLog::written_since`].
    pub(super) fn written_since(
        &self,
        since: u64,
    ) -> Option<impl ExactSizeIterator<Item = WorkerWithDpRank> + '_> {
        self.write_log.written_since(since)
    }

    #[inline(always)]
    pub(super) fn project_worker_loads_into(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        decay_now: Instant,
        projections: &mut FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        full_ranks: Option<&mut FullRankCounts>,
    ) -> u64 {
        self.tracker
            .project_worker_loads_into(token_sequence, decay_now, projections, full_ranks)
    }

    #[inline(always)]
    pub(super) fn reproject_worker_loads(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        decay_now: Instant,
        workers: &[WorkerWithDpRank],
        version: u64,
        projections: &mut FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        full_ranks: Option<&mut FullRankCounts>,
    ) -> bool {
        self.tracker.reproject_worker_loads(
            token_sequence,
            decay_now,
            workers,
            version,
            projections,
            full_ranks,
        )
    }

    #[inline(always)]
    pub(super) fn record_full_worker_ranks_at(
        &self,
        decay_now: Instant,
        full_ranks: &mut FullRankCounts,
    ) {
        self.tracker
            .record_full_worker_ranks_at(decay_now, full_ranks);
    }

    #[inline(always)]
    pub(super) fn worker_load_reader(&self, decay_now: Instant) -> WorkerLoadReader<'_> {
        self.tracker.worker_load_reader(decay_now)
    }

    #[inline(always)]
    pub(super) fn decode_progress_for(&self, worker: WorkerWithDpRank) -> Option<DecodeProgress> {
        self.tracker.decode_progress_for(worker)
    }

    #[inline(always)]
    pub(super) fn additional_active_blocks(
        &self,
        token_sequence: &[SequenceHash],
        worker: WorkerWithDpRank,
    ) -> usize {
        self.tracker
            .additional_active_blocks(token_sequence, worker)
    }
}

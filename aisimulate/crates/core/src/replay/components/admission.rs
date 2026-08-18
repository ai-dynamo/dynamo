// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::marker::PhantomData;

use anyhow::Result;
use uuid::Uuid;

use super::ReplayMode;
use crate::replay::core::{AdmissionSource as CoreAdmissionSource, ReadyArrival};
use crate::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload, WorkloadDriver};
use crate::replay::protocol::DirectRequest;

#[doc(hidden)]
pub trait ReplayAdmissionMetadata: Sized {
    fn from_hashes(hashes: Option<ReplayRequestHashes>) -> Self;
    fn for_prefill(self) -> Self;
    fn max_output_tokens_override(&self) -> Option<usize>;
    fn into_hashes(self) -> Option<ReplayRequestHashes>;
}

pub type NoReplayMetadata = ();

impl ReplayAdmissionMetadata for () {
    #[inline]
    fn from_hashes(_hashes: Option<ReplayRequestHashes>) -> Self {}

    #[inline]
    fn for_prefill(self) -> Self {}

    #[inline]
    fn max_output_tokens_override(&self) -> Option<usize> {
        None
    }

    #[inline]
    fn into_hashes(self) -> Option<ReplayRequestHashes> {
        None
    }
}

/// Replay's richer admission record. The placement-facing [`ReadyArrival`]
/// intentionally carries only policy metadata; this sidecar also retains the
/// workload-authored ready time and hashes needed by optional replay artifacts.
pub(crate) struct ReplayReadyArrival<Metadata> {
    pub(crate) request: ReplayRequestPayload,
    pub(crate) arrival_time_ms: f64,
    pub(crate) scheduled_ready_at_ms: f64,
    pub(crate) metadata: Metadata,
    pub(crate) replay_hashes: Option<ReplayRequestHashes>,
    pub(crate) session_id: Option<String>,
    pub(crate) turn_index: Option<usize>,
}

impl<Metadata> ReplayReadyArrival<Metadata> {
    fn into_core(self) -> ReadyArrival<ReplayRequestPayload, Metadata> {
        ReadyArrival {
            request: self.request,
            arrival_time_ms: self.arrival_time_ms,
            metadata: self.metadata,
            session_id: self.session_id,
            turn_index: self.turn_index,
        }
    }
}

enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

pub(crate) struct AdmissionQueue<Metadata = NoReplayMetadata> {
    source: AdmissionSource,
    mode: ReplayMode,
    metadata: PhantomData<Metadata>,
}

impl<Metadata: ReplayAdmissionMetadata> AdmissionQueue<Metadata> {
    pub(crate) fn new_requests(source: VecDeque<DirectRequest>, mode: ReplayMode) -> Self {
        Self {
            source: AdmissionSource::Requests(source),
            mode,
            metadata: PhantomData,
        }
    }

    pub(crate) fn new_workload(driver: WorkloadDriver, mode: ReplayMode) -> Self {
        Self {
            source: AdmissionSource::Workload(driver),
            mode,
            metadata: PhantomData,
        }
    }

    pub(crate) fn mode(&self) -> ReplayMode {
        self.mode
    }

    pub(crate) fn next_ready_time_ms(&mut self) -> Option<f64> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => driver.next_ready_time_ms(),
            // Concurrency: the driver owns the session cap and gates admission, so defer to
            // it directly (no in-flight clamp needed here).
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                driver.next_ready_time_ms()
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Requests(_)) => None,
        }
    }

    /// Offline replay keeps full-prompt workload arrivals compact while they
    /// wait in an aggregated or prefill router queue. Legacy request queues
    /// and cumulative-delta workloads remain materialized because they do not
    /// have an independent compact prompt representation.
    pub(crate) fn drain_ready_compact(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
        retain_artifact_hashes: bool,
    ) -> Result<Vec<ReplayReadyArrival<Metadata>>> {
        self.drain_ready_compact_with(now_ms, cluster_in_flight, retain_artifact_hashes, |ready| {
            ready
        })
    }

    fn drain_ready_compact_with<T>(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
        retain_artifact_hashes: bool,
        mut map: impl FnMut(ReplayReadyArrival<Metadata>) -> T,
    ) -> Result<Vec<T>> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                loop {
                    let arrival_ms = pending
                        .front()
                        .and_then(|request| request.arrival_timestamp_ms)
                        .filter(|arrival_ms| *arrival_ms <= now_ms);
                    let Some(arrival_time_ms) = arrival_ms else {
                        break;
                    };
                    let request = pending
                        .pop_front()
                        .expect("front request must exist when arrival is ready");
                    let (session_id, turn_index) = request
                        .replay_context
                        .as_ref()
                        .map(|context| (context.session_id.clone(), context.turn_index))
                        .unwrap_or_default();
                    ready.push(map(ReplayReadyArrival {
                        request: ReplayRequestPayload::materialized(request),
                        arrival_time_ms,
                        scheduled_ready_at_ms: arrival_time_ms,
                        metadata: Metadata::from_hashes(None),
                        replay_hashes: None,
                        session_id,
                        turn_index,
                    }));
                }
                Ok(ready)
            }
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => Ok(driver
                .pop_ready_compact(now_ms, usize::MAX)
                .into_iter()
                .map(|ready| {
                    let session_id = ready.emit_session_metadata.then_some(ready.session_id);
                    let turn_index = ready.emit_session_metadata.then_some(ready.turn_index);
                    let replay_hashes = ready.replay_hashes;
                    let (metadata_hashes, replay_hashes) = if retain_artifact_hashes {
                        (replay_hashes.clone(), replay_hashes)
                    } else {
                        (replay_hashes, None)
                    };
                    map(ReplayReadyArrival {
                        request: ready.request,
                        arrival_time_ms: ready.scheduled_ready_at_ms,
                        scheduled_ready_at_ms: ready.scheduled_ready_at_ms,
                        metadata: Metadata::from_hashes(metadata_hashes),
                        replay_hashes,
                        session_id,
                        turn_index,
                    })
                })
                .collect()),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                let mut simulated_in_flight = cluster_in_flight;
                while simulated_in_flight < *max_in_flight {
                    let Some(mut request) = pending.pop_front() else {
                        break;
                    };
                    request.arrival_timestamp_ms = Some(now_ms);
                    let (session_id, turn_index) = request
                        .replay_context
                        .as_ref()
                        .map(|context| (context.session_id.clone(), context.turn_index))
                        .unwrap_or_default();
                    ready.push(map(ReplayReadyArrival {
                        request: ReplayRequestPayload::materialized(request),
                        arrival_time_ms: now_ms,
                        scheduled_ready_at_ms: now_ms,
                        metadata: Metadata::from_hashes(None),
                        replay_hashes: None,
                        session_id,
                        turn_index,
                    }));
                    simulated_in_flight += 1;
                }
                Ok(ready)
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                // The driver owns the session cap and only ever holds active sessions'
                // turns in its heap, so drain everything ready in heap (i.e. limit=usize MAX).
                Ok(driver
                    .pop_ready_compact(now_ms, usize::MAX)
                    .into_iter()
                    .map(|ready| {
                        let session_id = ready.emit_session_metadata.then_some(ready.session_id);
                        let turn_index = ready.emit_session_metadata.then_some(ready.turn_index);
                        let replay_hashes = ready.replay_hashes;
                        let (metadata_hashes, replay_hashes) = if retain_artifact_hashes {
                            (replay_hashes.clone(), replay_hashes)
                        } else {
                            (replay_hashes, None)
                        };
                        map(ReplayReadyArrival {
                            request: ready.request,
                            arrival_time_ms: now_ms,
                            scheduled_ready_at_ms: ready.scheduled_ready_at_ms,
                            metadata: Metadata::from_hashes(metadata_hashes),
                            replay_hashes,
                            session_id,
                            turn_index,
                        })
                    })
                    .collect())
            }
        }
    }

    pub(crate) fn on_request_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        rejected: bool,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_terminal(uuid, now_ms, rejected)
    }

    pub(crate) fn on_output_token(&mut self, uuid: Uuid, token_id: u32) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_output_token(uuid, token_id)
    }

    pub(crate) fn is_drained(&self) -> bool {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.is_empty(),
            AdmissionSource::Workload(driver) => driver.is_drained(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_workload(&self) -> bool {
        matches!(self.source, AdmissionSource::Workload(_))
    }

    pub(crate) fn total_requests(&self) -> usize {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::Workload(driver) => driver.total_turns(),
        }
    }
}

impl<Metadata: ReplayAdmissionMetadata> CoreAdmissionSource for AdmissionQueue<Metadata> {
    type Request = ReplayRequestPayload;
    type Metadata = Metadata;

    fn next_ready_time_ms(&mut self) -> Option<f64> {
        AdmissionQueue::next_ready_time_ms(self)
    }

    fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>> {
        self.drain_ready_compact_with(
            now_ms,
            cluster_in_flight,
            false,
            ReplayReadyArrival::into_core,
        )
    }

    fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()> {
        AdmissionQueue::on_output_token(self, request_id, token_id)
    }

    fn on_terminal(&mut self, request_id: Uuid, now_ms: f64, rejected: bool) -> Result<()> {
        AdmissionQueue::on_request_terminal(self, request_id, now_ms, rejected)
    }

    fn is_drained(&self) -> bool {
        AdmissionQueue::is_drained(self)
    }

    fn total_requests(&self) -> usize {
        AdmissionQueue::total_requests(self)
    }
}

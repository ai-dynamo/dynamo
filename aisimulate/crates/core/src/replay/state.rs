// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail};
use uuid::Uuid;

use crate::replay::handoff::{
    HandoffCoordinatorCore, HandoffId, HandoffOrder, IssuedHandoffAction,
};
use crate::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use crate::replay::protocol::DirectRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggRequestPhase {
    QueuedAtRouter,
    Running,
}

pub(crate) struct AggRequestState {
    request: Option<ReplayRequestPayload>,
    pub(crate) phase: AggRequestPhase,
    pub(crate) prefill_completed: bool,
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
}

impl AggRequestState {
    pub(crate) fn new_queued(request: ReplayRequestPayload) -> Self {
        let input_tokens = request.input_length();
        let output_tokens = request.metadata().effective_max_output_tokens();
        Self {
            request: Some(request),
            phase: AggRequestPhase::QueuedAtRouter,
            prefill_completed: false,
            input_tokens,
            output_tokens,
        }
    }

    pub(crate) fn new_running(input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            request: None,
            phase: AggRequestPhase::Running,
            prefill_completed: false,
            input_tokens,
            output_tokens,
        }
    }

    pub(crate) fn take_queued_request(&mut self, uuid: Uuid) -> Result<DirectRequest> {
        if self.phase != AggRequestPhase::QueuedAtRouter {
            bail!("offline replay expected queued request state for {uuid}");
        }
        let request = self
            .request
            .take()
            .ok_or_else(|| anyhow!("offline replay missing queued request payload for {uuid}"))?;
        self.phase = AggRequestPhase::Running;
        Ok(request.into_direct_request())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisaggPhase {
    AwaitingDestination,
    QueuedPrefill,
    RunningPrefill,
    TransferPending,
    ReadyDecode,
    RunningDecode,
    CleanupPending,
    Done,
}

pub(crate) struct DisaggRequestState {
    original: Option<ReplayRequestPayload>,
    session_id: Option<String>,
    #[cfg(test)]
    arrival_ms: f64,
    pub(crate) phase: DisaggPhase,
    pub(crate) handoff_id: HandoffId,
    pub(crate) coordinator: HandoffCoordinatorCore,
    pub(crate) counted_in_flight: bool,
    replay_hashes: Option<ReplayRequestHashes>,
    prefill_worker_idx: Option<usize>,
    decode_worker_idx: Option<usize>,
    pub(crate) prefill_routed: bool,
    pub(crate) destination_routed: bool,
    pub(crate) pending_prefill_action: Option<IssuedHandoffAction>,
    pub(crate) pending_destination_action: Option<IssuedHandoffAction>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DisaggRequestSnapshot {
    pub(crate) arrival_ms: f64,
    pub(crate) phase: DisaggPhase,
    pub(crate) prefill_worker_idx: Option<usize>,
    pub(crate) decode_worker_idx: Option<usize>,
}

impl DisaggRequestState {
    pub(crate) fn new(
        request: ReplayRequestPayload,
        arrival_ms: f64,
        handoff_id: HandoffId,
        order: HandoffOrder,
        handoff_latency_ms: f64,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
    ) -> Self {
        #[cfg(not(test))]
        let _ = arrival_ms;
        Self {
            original: Some(request),
            session_id,
            #[cfg(test)]
            arrival_ms,
            phase: match order {
                HandoffOrder::SourceFirst => DisaggPhase::QueuedPrefill,
                HandoffOrder::DestinationFirst => DisaggPhase::AwaitingDestination,
            },
            handoff_id,
            coordinator: HandoffCoordinatorCore::new_with_fallback(
                handoff_id,
                order,
                handoff_latency_ms,
            ),
            counted_in_flight: true,
            replay_hashes,
            prefill_worker_idx: None,
            decode_worker_idx: None,
            prefill_routed: false,
            destination_routed: false,
            pending_prefill_action: None,
            pending_destination_action: None,
        }
    }

    pub(crate) fn original_request(&self) -> Result<&DirectRequest> {
        self.original
            .as_ref()
            .ok_or_else(|| anyhow!("offline disagg replay request payload was already released"))
            .and_then(|request| {
                request.materialized_request().ok_or_else(|| {
                    anyhow!("offline disagg replay request payload is not materialized")
                })
            })
    }

    pub(crate) fn request_payload(&self) -> Result<&ReplayRequestPayload> {
        self.original
            .as_ref()
            .ok_or_else(|| anyhow!("offline disagg replay request payload was already released"))
    }

    pub(crate) fn input_length(&self) -> Result<usize> {
        self.original
            .as_ref()
            .map(ReplayRequestPayload::input_length)
            .ok_or_else(|| anyhow!("offline disagg replay request payload was already released"))
    }

    #[cfg(test)]
    pub(crate) fn materialized_tokens(&self) -> Result<Option<&[u32]>> {
        self.original
            .as_ref()
            .map(ReplayRequestPayload::materialized_tokens)
            .ok_or_else(|| anyhow!("offline disagg replay request payload was already released"))
    }

    pub(crate) fn materialize_original_request(&mut self) -> Result<&DirectRequest> {
        self.original
            .as_mut()
            .ok_or_else(|| anyhow!("offline disagg replay request payload was already released"))?
            .materialize()
            .ok_or_else(|| anyhow!("offline disagg replay request payload failed to materialize"))
    }

    pub(crate) fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub(crate) fn build_prefill_request(&mut self) -> Result<DirectRequest> {
        Ok(self
            .materialize_original_request()?
            .clone_with_output_limit(1))
    }

    pub(crate) fn take_replay_hashes(&mut self) -> Option<ReplayRequestHashes> {
        self.replay_hashes.take()
    }

    pub(crate) fn start_prefill(&mut self, worker_idx: usize) {
        self.phase = DisaggPhase::RunningPrefill;
        self.prefill_worker_idx = Some(worker_idx);
    }

    pub(crate) fn prefill_worker_idx(&self) -> Option<usize> {
        self.prefill_worker_idx
    }

    pub(crate) fn await_destination(&mut self) {
        self.phase = DisaggPhase::AwaitingDestination;
    }

    pub(crate) fn assign_decode(&mut self, worker_idx: usize) {
        self.decode_worker_idx = Some(worker_idx);
    }

    pub(crate) fn decode_worker_idx(&self) -> Option<usize> {
        self.decode_worker_idx
    }

    pub(crate) fn transfer_pending(&mut self) {
        self.phase = DisaggPhase::TransferPending;
    }

    pub(crate) fn ready_decode(&mut self) {
        self.phase = DisaggPhase::ReadyDecode;
    }

    pub(crate) fn start_decode(&mut self) {
        self.phase = DisaggPhase::RunningDecode;
    }

    pub(crate) fn complete_decode(&mut self) {
        self.phase = DisaggPhase::CleanupPending;
        self.original = None;
        self.replay_hashes = None;
        self.pending_prefill_action = None;
        self.pending_destination_action = None;
    }

    pub(crate) fn mark_done(&mut self) {
        self.phase = DisaggPhase::Done;
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self) -> DisaggRequestSnapshot {
        DisaggRequestSnapshot {
            arrival_ms: self.arrival_ms,
            phase: self.phase,
            prefill_worker_idx: self.prefill_worker_idx,
            decode_worker_idx: self.decode_worker_idx,
        }
    }
}

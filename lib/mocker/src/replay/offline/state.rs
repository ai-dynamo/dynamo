// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail};

use crate::common::handoff::{
    HandoffCoordinatorCore, HandoffId, HandoffOrder, IssuedHandoffAction,
};
use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::ReplayRequestHashes;
use crate::replay::TraceCollector;
use crate::scheduler::{
    EngineCore, EnginePassResult, SchedulerCommand, SchedulerCommandEffects,
    SchedulerCommandResult, SchedulerLifecycleEvent,
};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggRequestPhase {
    QueuedAtRouter,
    Running,
}

pub(crate) struct AggRequestState {
    request: Option<DirectRequest>,
    pub(in crate::replay::offline) phase: AggRequestPhase,
    pub(in crate::replay::offline) prefill_completed: bool,
    pub(in crate::replay::offline) input_tokens: usize,
    pub(in crate::replay::offline) output_tokens: usize,
}

impl AggRequestState {
    pub(crate) fn new_queued(request: DirectRequest) -> Self {
        let input_tokens = request.tokens.len();
        let output_tokens = request.max_output_tokens;
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
        Ok(request)
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
    original: Option<DirectRequest>,
    session_id: Option<String>,
    #[cfg(test)]
    arrival_ms: f64,
    pub(in crate::replay::offline) phase: DisaggPhase,
    pub(in crate::replay::offline) handoff_id: HandoffId,
    pub(in crate::replay::offline) coordinator: HandoffCoordinatorCore,
    pub(in crate::replay::offline) counted_in_flight: bool,
    replay_hashes: Option<ReplayRequestHashes>,
    prefill_worker_idx: Option<usize>,
    decode_worker_idx: Option<usize>,
    pub(in crate::replay::offline) prefill_routed: bool,
    pub(in crate::replay::offline) destination_routed: bool,
    pub(in crate::replay::offline) pending_prefill_action: Option<IssuedHandoffAction>,
    pub(in crate::replay::offline) pending_destination_action: Option<IssuedHandoffAction>,
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
        request: DirectRequest,
        arrival_ms: f64,
        handoff_id: HandoffId,
        order: HandoffOrder,
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
            coordinator: HandoffCoordinatorCore::new(handoff_id, order),
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
    }

    pub(crate) fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub(crate) fn build_prefill_request(&self) -> Result<DirectRequest> {
        let mut request = self.original_request()?.clone();
        request.max_output_tokens = request.max_output_tokens.min(1);
        Ok(request)
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

pub(crate) struct OfflineWorkerState {
    core: EngineCore,
    worker_id: u64,
    dp_rank: u32,
    busy: bool,
    in_flight: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineWorkerSnapshot {
    pub(crate) busy: bool,
    pub(crate) in_flight: usize,
    pub(crate) ready: bool,
    pub(crate) drained: bool,
}

impl OfflineWorkerState {
    pub(crate) fn new(worker_idx: usize, args: MockEngineArgs, capture_kv_events: bool) -> Self {
        Self::new_with_rank(worker_idx, worker_idx as u64, 0, args, capture_kv_events)
    }

    pub(crate) fn new_with_rank(
        worker_idx: usize,
        worker_id: u64,
        dp_rank: u32,
        args: MockEngineArgs,
        capture_kv_events: bool,
    ) -> Self {
        let core = match args.engine_type {
            crate::common::protocols::EngineType::Vllm
            | crate::common::protocols::EngineType::Trtllm => {
                #[cfg_attr(not(feature = "kvbm-offload"), allow(unused_mut))]
                let mut core = crate::scheduler::VllmCore::new_with_worker_rank(
                    args,
                    worker_id,
                    dp_rank,
                    worker_idx as u64,
                    capture_kv_events,
                );
                #[cfg(feature = "kvbm-offload")]
                if let Err(e) = core.init_offload_offline() {
                    tracing::error!(
                        "kvbm-offload offline init failed for worker {worker_idx}: {e}"
                    );
                }
                EngineCore::Vllm(core)
            }
            crate::common::protocols::EngineType::Sglang => {
                EngineCore::Sglang(crate::scheduler::SglangCore::new_with_worker_rank(
                    args,
                    worker_id,
                    dp_rank,
                    worker_idx as u64,
                    capture_kv_events,
                ))
            }
        };

        Self {
            core,
            worker_id,
            dp_rank,
            busy: false,
            in_flight: 0,
        }
    }

    pub(crate) fn rank_identity(&self) -> (u64, u32) {
        (self.worker_id, self.dp_rank)
    }

    pub(crate) fn in_flight(&self) -> usize {
        debug_assert!(self.in_flight >= self.core.num_requests());
        self.in_flight
    }

    pub(crate) fn receive_request(&mut self, mut request: DirectRequest) {
        self.in_flight = self
            .in_flight
            .checked_add(1)
            .expect("offline worker in-flight request count overflow");
        request.dp_rank = self.dp_rank;
        self.core.receive(request);
    }

    pub(crate) fn apply_command(
        &mut self,
        command: SchedulerCommand,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        enum Accounting {
            Submit,
            ReserveDestination,
            CancelRequest,
            CancelSource,
            CancelDestination,
            None,
        }

        let accounting = match &command {
            SchedulerCommand::Submit(_) | SchedulerCommand::SubmitHandoffPrefill { .. } => {
                Accounting::Submit
            }
            SchedulerCommand::ReserveDestination { .. } => Accounting::ReserveDestination,
            SchedulerCommand::CancelRequest { .. } => Accounting::CancelRequest,
            SchedulerCommand::CancelSource { .. } => Accounting::CancelSource,
            SchedulerCommand::CancelDestination { .. } => Accounting::CancelDestination,
            SchedulerCommand::ReleaseSource { .. }
            | SchedulerCommand::ActivateDestination { .. } => Accounting::None,
        };
        let requests_before = self.core.num_requests();
        let mut effects = self.core.apply_command_effects(command, !self.busy)?;
        effects.kv_events = self.core.drain_kv_events();
        match (accounting, effects.result) {
            (Accounting::Submit, SchedulerCommandResult::Submitted(_))
            | (
                Accounting::ReserveDestination,
                SchedulerCommandResult::DestinationAccepted { .. },
            ) => self.increment_in_flight(),
            (Accounting::CancelDestination, SchedulerCommandResult::Applied) => {
                self.decrement_in_flight(1)
            }
            (Accounting::CancelRequest, SchedulerCommandResult::Applied) => {
                self.decrement_in_flight(1)
            }
            (Accounting::CancelSource, SchedulerCommandResult::Applied) => {
                let removed = requests_before
                    .checked_sub(self.core.num_requests())
                    .expect("source cancellation increased scheduler request ownership");
                if removed > 0 {
                    self.decrement_in_flight(removed);
                }
            }
            _ => {}
        }
        Ok(effects)
    }

    pub(crate) fn mark_completed(&mut self, completed_requests: usize) {
        self.decrement_in_flight(completed_requests);
    }

    pub(crate) fn mark_busy(&mut self) {
        self.busy = true;
    }

    pub(crate) fn mark_idle(&mut self) {
        self.busy = false;
    }

    pub(crate) fn is_ready(&self) -> bool {
        !self.busy && !self.core.is_empty()
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.busy
    }

    pub(crate) fn is_drained(&self) -> bool {
        self.in_flight == 0 && !self.busy && self.core.is_drained()
    }

    pub(crate) fn retry_pending_destinations(&mut self) -> Vec<SchedulerLifecycleEvent> {
        self.core.retry_pending_destinations()
    }

    pub(crate) fn drain_kv_events(&self) -> Vec<dynamo_kv_router::protocols::RouterEvent> {
        self.core.drain_kv_events()
    }

    fn increment_in_flight(&mut self) {
        self.in_flight = self
            .in_flight
            .checked_add(1)
            .expect("offline worker in-flight request count overflow");
    }

    fn decrement_in_flight(&mut self, count: usize) {
        self.in_flight = self
            .in_flight
            .checked_sub(count)
            .expect("offline worker completed more requests than it owned");
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.core.execute_pass(collector, now_ms)
    }

    pub(crate) fn execute_hidden_pass(&mut self, now_ms: f64) -> EnginePassResult {
        self.core.execute_hidden_pass(now_ms)
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn tick_offload_only(
        &mut self,
        now_ms: f64,
    ) -> crate::scheduler::OffloadTickEffects {
        self.core.tick_offload_only(now_ms)
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn tick_offload_transport_only(
        &mut self,
        now_ms: f64,
    ) -> crate::scheduler::OffloadTickEffects {
        self.core.tick_offload_transport_only(now_ms)
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn earliest_offload_deadline(&self) -> Option<f64> {
        self.core.earliest_offload_deadline()
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self) -> OfflineWorkerSnapshot {
        OfflineWorkerSnapshot {
            busy: self.busy,
            in_flight: self.in_flight,
            ready: self.is_ready(),
            drained: self.is_drained(),
        }
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::{DisaggRequestState, OfflineWorkerState};
    use crate::common::handoff::{HandoffId, HandoffOrder};
    use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, WorkerType};
    use crate::scheduler::{SchedulerCommand, SchedulerCommandResult};
    use dynamo_kv_router::protocols::KvCacheEventData;

    fn worker(
        engine_type: EngineType,
        worker_type: WorkerType,
        blocks: usize,
    ) -> OfflineWorkerState {
        worker_with_capture(engine_type, worker_type, blocks, false)
    }

    fn worker_with_capture(
        engine_type: EngineType,
        worker_type: WorkerType,
        blocks: usize,
        capture_kv_events: bool,
    ) -> OfflineWorkerState {
        let mut builder = MockEngineArgs::builder()
            .engine_type(engine_type)
            .worker_type(worker_type)
            .block_size(4)
            .num_gpu_blocks(blocks)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .speedup_ratio(0.0);
        if engine_type == EngineType::Sglang {
            builder = builder.sglang(Some(Default::default()));
        }
        OfflineWorkerState::new(0, builder.build().unwrap(), capture_kv_events)
    }

    fn request(uuid: u128, tokens: usize) -> DirectRequest {
        DirectRequest {
            uuid: Some(Uuid::from_u128(uuid)),
            tokens: (0..tokens as u32).collect(),
            max_output_tokens: 2,
            ..Default::default()
        }
    }

    #[test]
    fn request_cancellation_releases_offline_in_flight_slot() {
        let mut worker = worker(EngineType::Vllm, WorkerType::Aggregated, 8);
        let request_id = Uuid::from_u128(850);
        worker.receive_request(request(request_id.as_u128(), 8));
        assert_eq!(worker.in_flight(), 1);

        assert_eq!(
            worker
                .apply_command(SchedulerCommand::CancelRequest { request_id })
                .unwrap()
                .result,
            SchedulerCommandResult::Applied
        );
        assert_eq!(worker.in_flight(), 0);
        assert!(worker.is_drained());

        assert_eq!(
            worker
                .apply_command(SchedulerCommand::CancelRequest { request_id })
                .unwrap()
                .result,
            SchedulerCommandResult::Noop
        );
        assert_eq!(worker.in_flight(), 0);
    }

    #[test]
    fn ranked_worker_preserves_router_worker_and_dp_rank_identity() {
        for engine_type in [EngineType::Vllm, EngineType::Sglang] {
            let mut builder = MockEngineArgs::builder()
                .engine_type(engine_type)
                .block_size(4)
                .num_gpu_blocks(64)
                .max_num_batched_tokens(Some(64))
                .max_num_seqs(Some(4))
                .dp_size(4)
                .enable_prefix_caching(true)
                .speedup_ratio(1000.0);
            if engine_type == EngineType::Sglang {
                builder = builder.sglang(Some(Default::default()));
            }
            let mut worker =
                OfflineWorkerState::new_with_rank(3, 7, 3, builder.build().unwrap(), true);
            worker.receive_request(request(900 + engine_type as u128, 8));

            let mut now_ms = 0.0;
            let mut events = Vec::new();
            while !worker.core.is_empty() {
                let pass = worker.execute_hidden_pass(now_ms);
                now_ms = pass.end_ms;
                worker.mark_completed(pass.completed_requests);
                events.extend(pass.kv_events);
            }
            events.extend(worker.drain_kv_events());

            assert!(!events.is_empty());
            assert!(events.iter().all(|event| event.worker_id == 7));
            assert!(events.iter().all(|event| event.event.dp_rank == 3));
        }
    }

    #[test]
    fn disagg_prefill_request_preserves_router_priorities() {
        let state = DisaggRequestState::new(
            DirectRequest {
                tokens: vec![1; 8],
                max_output_tokens: 12,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                priority: -3,
                strict_priority: 9,
                policy_class: None,
            },
            0.0,
            HandoffId::from(Uuid::from_u128(2)),
            HandoffOrder::SourceFirst,
            None,
            None,
        );

        let request = state.build_prefill_request().unwrap();
        assert_eq!(request.max_output_tokens, 1);
        assert_eq!(request.priority, -3);
        assert_eq!(request.strict_priority, 9);
    }

    #[test]
    fn handoff_worker_accounting_tracks_role_ownership_exactly_once() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut prefill = worker(engine_type, WorkerType::Prefill, 8);
            let source_handoff = HandoffId::from(Uuid::from_u128(10_000 + case as u128));
            assert!(matches!(
                prefill
                    .apply_command(SchedulerCommand::SubmitHandoffPrefill {
                        handoff_id: source_handoff,
                        request: request(10_100 + case as u128, 8),
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Submitted(_)
            ));
            assert_eq!(prefill.in_flight(), 1);
            let mut now_ms = 0.0;
            while !prefill.core.is_empty() {
                let pass = prefill.execute_hidden_pass(now_ms);
                now_ms = pass.end_ms;
                prefill.mark_completed(pass.completed_requests);
            }
            assert_eq!(prefill.in_flight(), 0);
            assert!(!prefill.is_drained());
            assert_eq!(
                prefill
                    .apply_command(SchedulerCommand::ReleaseSource {
                        handoff_id: source_handoff,
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Applied
            );
            assert_eq!(prefill.in_flight(), 0);
            assert!(prefill.is_drained());

            let mut canceled_prefill = worker(engine_type, WorkerType::Prefill, 8);
            let canceled_handoff = HandoffId::from(Uuid::from_u128(10_150 + case as u128));
            assert!(matches!(
                canceled_prefill
                    .apply_command(SchedulerCommand::SubmitHandoffPrefill {
                        handoff_id: canceled_handoff,
                        request: request(10_175 + case as u128, 8),
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Submitted(_)
            ));
            assert_eq!(canceled_prefill.in_flight(), 1);
            assert_eq!(
                canceled_prefill
                    .apply_command(SchedulerCommand::CancelSource {
                        handoff_id: canceled_handoff,
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Applied
            );
            assert_eq!(canceled_prefill.in_flight(), 0);
            assert!(canceled_prefill.is_drained());

            let mut decode = worker(engine_type, WorkerType::Decode, 2);
            let first_handoff = HandoffId::from(Uuid::from_u128(10_200 + case as u128));
            let pending_handoff = HandoffId::from(Uuid::from_u128(10_300 + case as u128));
            assert!(matches!(
                decode
                    .apply_command(SchedulerCommand::ReserveDestination {
                        handoff_id: first_handoff,
                        request: request(10_400 + case as u128, 8),
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::DestinationAccepted { .. }
            ));
            assert!(matches!(
                decode
                    .apply_command(SchedulerCommand::ReserveDestination {
                        handoff_id: pending_handoff,
                        request: request(10_500 + case as u128, 4),
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::DestinationAccepted { .. }
            ));
            assert_eq!(decode.in_flight(), 2);
            assert_eq!(
                decode
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: pending_handoff,
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Applied
            );
            assert_eq!(decode.in_flight(), 1);
            assert_eq!(
                decode
                    .apply_command(SchedulerCommand::ActivateDestination {
                        handoff_id: first_handoff,
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Applied
            );
            assert_eq!(decode.in_flight(), 1);
            assert_eq!(
                decode
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: first_handoff,
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Applied
            );
            assert_eq!(decode.in_flight(), 0);
            assert_eq!(
                decode
                    .apply_command(SchedulerCommand::CancelDestination {
                        handoff_id: first_handoff,
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Noop
            );
            assert_eq!(decode.in_flight(), 0);
            assert!(decode.is_drained());

            let mut completed_decode = worker(engine_type, WorkerType::Decode, 8);
            let completed_handoff = HandoffId::from(Uuid::from_u128(10_600 + case as u128));
            assert!(matches!(
                completed_decode
                    .apply_command(SchedulerCommand::ReserveDestination {
                        handoff_id: completed_handoff,
                        request: request(10_700 + case as u128, 8),
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::DestinationAccepted { .. }
            ));
            assert_eq!(
                completed_decode
                    .apply_command(SchedulerCommand::ActivateDestination {
                        handoff_id: completed_handoff,
                    })
                    .unwrap()
                    .result,
                SchedulerCommandResult::Applied
            );
            assert_eq!(completed_decode.in_flight(), 1);
            let mut now_ms = 0.0;
            while !completed_decode.core.is_empty() {
                let pass = completed_decode.execute_hidden_pass(now_ms);
                now_ms = pass.end_ms.max(now_ms + 1.0);
                completed_decode.mark_completed(pass.completed_requests);
            }
            assert_eq!(completed_decode.in_flight(), 0);
            assert!(completed_decode.is_drained());
        }
    }

    #[test]
    fn offline_destination_activation_carries_stored_events_once() {
        for (case, engine_type) in [EngineType::Vllm, EngineType::Sglang]
            .into_iter()
            .enumerate()
        {
            let mut decode = worker_with_capture(engine_type, WorkerType::Decode, 8, true);
            let handoff_id = HandoffId::from(Uuid::from_u128(11_000 + case as u128));
            let request_id = 11_100 + case as u128;
            let reserve = decode
                .apply_command(SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: request(request_id, 8),
                })
                .unwrap();
            assert!(
                reserve
                    .kv_events
                    .iter()
                    .all(|event| !matches!(event.event.data, KvCacheEventData::Stored(_)))
            );

            let activation = decode
                .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap();
            assert!(
                activation
                    .kv_events
                    .iter()
                    .any(|event| matches!(event.event.data, KvCacheEventData::Stored(_)))
            );

            let pass = decode.execute_hidden_pass(0.0);
            assert!(
                pass.kv_events
                    .iter()
                    .all(|event| !matches!(event.event.data, KvCacheEventData::Stored(_)))
            );
        }
    }
}

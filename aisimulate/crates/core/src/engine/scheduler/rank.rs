// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin neutral contract adapter over the mechanically moved scheduler cores.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use uuid::Uuid;

use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{
    DirectRequest, EngineType, KvTransferTimingMode, MockEngineArgs,
    PreemptionMode as CorePreemptionMode, SglangArgs, WorkerType as CoreWorkerType,
};
use crate::engine::generalized::{CommandContext, RankEngine, RankIdentity, RankPass};
use crate::engine::{
    Admission, Backend, Command, CommandEffects, CommandResult, EngineConfig, ForwardPassMetrics,
    HandoffId, LifecycleEvent, Metrics, Output, PassCompletionEffects, PassStartEffects,
    PendingPass, PreemptionMode, Request, TimingModel, TransferTimingMode, WorkerType,
};

use super::{
    EngineCore, EnginePassResult, KvEventVisibility, MockerMetrics,
    SchedulerCommand as CoreCommand, SchedulerCommandEffects as CoreCommandEffects,
    SchedulerCommandResult as CoreCommandResult, SchedulerLifecycleEvent as CoreLifecycle,
    SglangCore, VllmCore,
};

pub fn engine_seed_offset(identity: RankIdentity) -> Result<u64> {
    identity
        .worker_id
        .checked_mul(u64::from(identity.dp_size.get()))
        .and_then(|base| base.checked_add(u64::from(identity.dp_rank)))
        .ok_or_else(|| anyhow!("native mock-engine seed offset overflow"))
}

/// One preserved vLLM/SGLang scheduler rank behind the neutral contract.
pub struct SchedulerRank {
    core: EngineCore,
    handoff_requests: HashMap<HandoffId, Uuid>,
}

impl SchedulerRank {
    pub fn new_with_timing_model(
        identity: RankIdentity,
        config: &EngineConfig,
        timing: Arc<dyn TimingModel>,
        seed_offset: u64,
    ) -> Result<Self> {
        config.validate()?;
        let args = core_args(config, timing);
        let capture_kv_events = config.emit_kv_events;
        let core = match config.backend {
            Backend::Vllm | Backend::Trtllm => EngineCore::Vllm(VllmCore::new_with_worker_rank(
                args,
                identity.worker_id,
                identity.dp_rank,
                seed_offset,
                capture_kv_events,
            )),
            Backend::Sglang => EngineCore::Sglang(SglangCore::new_with_worker_rank(
                args,
                identity.worker_id,
                identity.dp_rank,
                seed_offset,
                capture_kv_events,
            )),
        };
        Ok(Self {
            core,
            handoff_requests: HashMap::new(),
        })
    }

    fn core_command(command: Command) -> CoreCommand {
        match command {
            Command::Submit(request) => CoreCommand::Submit(core_request(request)),
            Command::CancelRequest { request_id, .. } => CoreCommand::CancelRequest { request_id },
            Command::SubmitHandoffPrefill {
                handoff_id,
                request,
            } => CoreCommand::SubmitHandoffPrefill {
                handoff_id,
                request: core_request(request),
            },
            Command::ReserveDestination {
                handoff_id,
                request,
            } => CoreCommand::ReserveDestination {
                handoff_id,
                request: core_request(request),
            },
            Command::ActivateDestination { handoff_id } => {
                CoreCommand::ActivateDestination { handoff_id }
            }
            Command::ReleaseSource { handoff_id } => CoreCommand::ReleaseSource { handoff_id },
            Command::CancelSource { handoff_id } => CoreCommand::CancelSource { handoff_id },
            Command::CancelDestination { handoff_id } => {
                CoreCommand::CancelDestination { handoff_id }
            }
        }
    }

    fn metrics(&self) -> Metrics {
        let metrics = match &self.core {
            EngineCore::Vllm(core) => core.mocker_metrics(),
            EngineCore::Sglang(core) => core.mocker_metrics(),
        };
        map_metrics(metrics)
    }
}

impl RankEngine for SchedulerRank {
    type Config = EngineConfig;
    type Command = Command;
    type CommandEffects = CommandEffects;
    type PassStartEffects = PassStartEffects;
    type PendingPass = PendingPass;
    type PassCompletionEffects = PassCompletionEffects;
    type InternalEffects = ();

    fn new(identity: RankIdentity, config: &Self::Config) -> Result<Self> {
        let timing = config.built_in_timing_model()?;
        let seed_offset = engine_seed_offset(identity)?;
        Self::new_with_timing_model(identity, config, timing, seed_offset)
    }

    fn apply_command_effects(
        &mut self,
        command: Self::Command,
        context: CommandContext,
        pending_pass: Option<&mut Self::PendingPass>,
    ) -> Result<Self::CommandEffects> {
        let pending_suppression = pending_output_suppression(&command, &self.handoff_requests);
        let handoff_update = handoff_tracking_update(&command);
        let core_command = Self::core_command(command);
        let mut effects = self
            .core
            .apply_command_effects(core_command, context.allow_immediate_admission())?;
        // Preserve the scheduler's command boundary: native G1 publishes KV
        // mutations into the rank-local capture sink, while command effects
        // are the only observation returned to the driver at this point. If
        // these events remain buffered, a later pass may drain and discard an
        // incomplete prefix of the stream before the Router observes it.
        effects.kv_events.extend(self.core.drain_kv_events());
        let suppressed_pending_output = if let (Some((request_id, discard_on_noop)), Some(pending)) =
            (pending_suppression, pending_pass)
            && (effects.result != CoreCommandResult::Noop || discard_on_noop)
        {
            let before = pending.effects.outputs.len();
            pending
                .effects
                .outputs
                .retain(|output| output.request_id != request_id);
            pending
                .effects
                .lifecycle_events
                .retain(|event| match *event {
                    LifecycleEvent::SourceHeld { request_id: id, .. }
                    | LifecycleEvent::DestinationReserved { request_id: id, .. } => {
                        id != request_id
                    }
                });
            before != pending.effects.outputs.len()
        } else {
            false
        };
        if effects.result != CoreCommandResult::Noop || suppressed_pending_output {
            self.apply_handoff_tracking_update(handoff_update);
        }
        for request_id in &effects.retired_requests {
            self.handoff_requests
                .retain(|_, tracked_request| tracked_request != request_id);
        }
        map_command_effects(effects, self.metrics(), suppressed_pending_output)
    }

    fn is_ready(&self) -> bool {
        !self.core.is_drained()
    }

    fn waiting_for_external_command(&self) -> bool {
        self.core.waiting_for_external_command()
    }

    fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<RankPass<Self::PassStartEffects, Self::PendingPass>> {
        let pass = self.core.try_execute_hidden_pass(now_ms)?;
        let end_ms = pass.end_ms;
        let (same_timestamp_retry, start_effects, completion_effects) = split_pass(pass)?;
        Ok(RankPass {
            end_ms,
            same_timestamp_retry,
            start_effects,
            pending: PendingPass {
                started_at_ms: now_ms,
                effects: completion_effects,
            },
        })
    }

    fn complete_pass(
        &mut self,
        mut pending: Self::PendingPass,
        end_ms: f64,
    ) -> Result<Self::PassCompletionEffects> {
        // The preserved scheduler retries deferred destination reservations
        // when a forward pass releases capacity. Keep that wakeup at the
        // pass-completion boundary: command-time retry is suppressed while a
        // pass is in flight, and waiting until an unrelated later pass can
        // leave disaggregated replay permanently asleep.
        pending.effects.lifecycle_events.extend(
            self.core
                .retry_pending_destinations()
                .into_iter()
                .map(map_lifecycle),
        );
        let completion_kv_events = self.core.drain_kv_events();
        pending.effects.kv_events.extend(completion_kv_events);
        pending.effects.metrics = self.metrics();
        pending.effects.forward_pass_metrics.duration_ms =
            (end_ms - pending.started_at_ms).max(0.0);
        for output in &pending.effects.outputs {
            if output.completed {
                self.handoff_requests
                    .retain(|_, request_id| *request_id != output.request_id);
            }
        }
        Ok(pending.effects)
    }

    fn complete_idle_group_pass(
        &mut self,
        started_at_ms: f64,
        end_ms: f64,
    ) -> Result<Option<Self::PassCompletionEffects>> {
        let lifecycle_events = self
            .core
            .retry_pending_destinations()
            .into_iter()
            .map(map_lifecycle)
            .collect::<Vec<_>>();
        let kv_events = self.core.drain_kv_events();
        Ok(Some(PassCompletionEffects {
            lifecycle_events,
            kv_events,
            metrics: self.metrics(),
            forward_pass_metrics: ForwardPassMetrics {
                duration_ms: (end_ms - started_at_ms).max(0.0),
                ..Default::default()
            },
            ..PassCompletionEffects::default()
        }))
    }

    fn next_internal_deadline_ms(&self) -> Option<f64> {
        None
    }

    fn process_internal_work(
        &mut self,
        _now_ms: f64,
        _pass_in_flight: bool,
    ) -> Result<Self::InternalEffects> {
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.core.is_drained()
    }
}

#[derive(Clone, Copy)]
enum HandoffTrackingUpdate {
    None,
    Insert(HandoffId, Uuid),
    RemoveHandoff(HandoffId),
    RemoveRequest(Uuid),
}

impl SchedulerRank {
    fn apply_handoff_tracking_update(&mut self, update: HandoffTrackingUpdate) {
        match update {
            HandoffTrackingUpdate::None => {}
            HandoffTrackingUpdate::Insert(handoff_id, request_id) => {
                self.handoff_requests.insert(handoff_id, request_id);
            }
            HandoffTrackingUpdate::RemoveHandoff(handoff_id) => {
                self.handoff_requests.remove(&handoff_id);
            }
            HandoffTrackingUpdate::RemoveRequest(request_id) => self
                .handoff_requests
                .retain(|_, tracked_request| *tracked_request != request_id),
        }
    }
}

fn handoff_tracking_update(command: &Command) -> HandoffTrackingUpdate {
    match command {
        Command::SubmitHandoffPrefill {
            handoff_id,
            request,
        }
        | Command::ReserveDestination {
            handoff_id,
            request,
        } => HandoffTrackingUpdate::Insert(*handoff_id, request.request_id),
        Command::ReleaseSource { handoff_id }
        | Command::CancelSource { handoff_id }
        | Command::CancelDestination { handoff_id } => {
            HandoffTrackingUpdate::RemoveHandoff(*handoff_id)
        }
        Command::CancelRequest { request_id, .. } => {
            HandoffTrackingUpdate::RemoveRequest(*request_id)
        }
        Command::Submit(_) | Command::ActivateDestination { .. } => HandoffTrackingUpdate::None,
    }
}

fn core_args(config: &EngineConfig, timing: Arc<dyn TimingModel>) -> MockEngineArgs {
    MockEngineArgs {
        engine_type: match config.backend {
            Backend::Vllm => EngineType::Vllm,
            Backend::Sglang => EngineType::Sglang,
            Backend::Trtllm => EngineType::Trtllm,
        },
        num_gpu_blocks: config.num_gpu_blocks,
        block_size: config.block_size,
        max_model_len: config.max_model_len,
        max_num_seqs: Some(config.max_num_seqs),
        max_num_batched_tokens: Some(config.max_num_batched_tokens),
        enable_prefix_caching: config.enable_prefix_caching,
        enable_chunked_prefill: config.enable_chunked_prefill,
        speedup_ratio: config.speedup_ratio,
        decode_speedup_ratio: config.decode_speedup_ratio,
        worker_type: match config.worker_type {
            WorkerType::Aggregated => CoreWorkerType::Aggregated,
            WorkerType::Prefill => CoreWorkerType::Prefill,
            WorkerType::Decode => CoreWorkerType::Decode,
        },
        perf_model: Arc::new(PerfModel::External { timing }),
        aic_nextn: config.aic_nextn,
        aic_nextn_accept_rates: config.aic_nextn_accept_rates.clone(),
        aic_mtp_seed: config.aic_mtp_seed,
        kv_bytes_per_token: config.kv_bytes_per_token,
        kv_transfer_bandwidth: config.kv_transfer_bandwidth,
        kv_transfer_timing_mode: match config.kv_transfer_timing_mode {
            TransferTimingMode::FullPrompt => KvTransferTimingMode::FullPrompt,
            TransferTimingMode::DestinationMissing => KvTransferTimingMode::DestinationMissing,
        },
        preemption_mode: match config.preemption_mode {
            PreemptionMode::Lifo => CorePreemptionMode::Lifo,
            PreemptionMode::Fifo => CorePreemptionMode::Fifo,
        },
        sglang: Some(SglangArgs {
            schedule_policy: Some(
                match config.sglang.schedule_policy {
                    crate::engine::SglangSchedulePolicy::Fifo => "fifo",
                    crate::engine::SglangSchedulePolicy::Lpm => "lpm",
                }
                .to_string(),
            ),
            page_size: Some(config.block_size),
            max_prefill_tokens: Some(config.sglang.max_prefill_tokens),
            chunked_prefill_size: Some(config.sglang.chunked_prefill_size),
            clip_max_new_tokens: Some(config.sglang.clip_max_new_tokens),
            schedule_conservativeness: Some(config.sglang.schedule_conservativeness),
        }),
        emit_kv_events: config.emit_kv_events,
        emit_kv_token_ids: config.emit_kv_token_ids,
    }
}

fn core_request(request: Request) -> DirectRequest {
    DirectRequest {
        tokens: request.tokens,
        max_output_tokens: request.max_output_tokens,
        output_token_ids: request.output_token_ids,
        uuid: Some(request.request_id),
        arrival_timestamp_ms: None,
    }
}

fn pending_output_suppression(
    command: &Command,
    handoffs: &HashMap<HandoffId, Uuid>,
) -> Option<(Uuid, bool)> {
    match *command {
        Command::CancelRequest {
            request_id,
            discard_pending_output,
        } => Some((request_id, discard_pending_output)),
        Command::CancelSource { handoff_id } | Command::CancelDestination { handoff_id } => {
            handoffs
                .get(&handoff_id)
                .copied()
                .map(|request_id| (request_id, false))
        }
        _ => None,
    }
}

fn map_command_effects(
    effects: CoreCommandEffects,
    metrics: Metrics,
    suppressed_pending_output: bool,
) -> Result<CommandEffects> {
    let result = match effects.result {
        CoreCommandResult::Submitted(id) => CommandResult::Submitted(id),
        CoreCommandResult::DestinationAccepted { request_id } => {
            CommandResult::DestinationAccepted { request_id }
        }
        CoreCommandResult::Applied => CommandResult::Applied,
        CoreCommandResult::Noop => CommandResult::Noop,
    };
    Ok(CommandEffects {
        result,
        lifecycle_events: effects
            .lifecycle_events
            .into_iter()
            .map(map_lifecycle)
            .collect(),
        kv_events: effects.kv_events,
        retired_requests: effects.retired_requests,
        metrics,
        suppressed_pending_output,
    })
}

fn map_lifecycle(event: CoreLifecycle) -> LifecycleEvent {
    match event {
        CoreLifecycle::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing,
        } => LifecycleEvent::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing,
        },
        CoreLifecycle::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        } => LifecycleEvent::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        },
    }
}

fn map_metrics(metrics: MockerMetrics) -> Metrics {
    Metrics {
        dp_rank: metrics.dp_rank,
        active_blocks: metrics.active_decode_blocks,
        total_blocks: metrics.total_blocks,
        cache_usage: metrics.gpu_cache_usage_perc,
        running_requests: metrics.running_requests,
        waiting_requests: metrics.waiting_requests,
        preemptions_total: metrics.vllm_preemptions_total,
        sglang_cache_hit_tokens: metrics.sglang_cache_hit_tokens,
        sglang_cache_total_tokens: metrics.sglang_cache_total_tokens,
    }
}

fn split_pass(
    pass: EnginePassResult,
) -> Result<(
    crate::engine::generalized::SameTimestampRetry,
    PassStartEffects,
    PassCompletionEffects,
)> {
    let EnginePassResult {
        same_timestamp_retry,
        output_signals,
        admissions,
        pressure_events,
        lifecycle_events,
        mocker_metrics,
        kv_event_visibility,
        kv_events,
        fpm,
        ..
    } = pass;
    let (start_kv, completion_kv) = match kv_event_visibility {
        KvEventVisibility::PassEnd => (Vec::new(), kv_events),
    };
    let start = PassStartEffects {
        admissions: admissions
            .into_iter()
            .map(|admission| Admission {
                request_id: admission.uuid,
                reused_input_tokens: admission.reused_input_tokens,
            })
            .collect(),
        pressure_events,
        kv_events: start_kv,
    };
    let completion = PassCompletionEffects {
        outputs: output_signals
            .into_iter()
            .map(|output| Output {
                request_id: output.uuid,
                token_id: output.token_id,
                completed: output.completed,
                rejected: output.rejected,
            })
            .collect(),
        lifecycle_events: lifecycle_events.into_iter().map(map_lifecycle).collect(),
        kv_events: completion_kv,
        metrics: map_metrics(mocker_metrics),
        forward_pass_metrics: fpm.map(map_fpm).unwrap_or_default(),
    };
    Ok((same_timestamp_retry, start, completion))
}

fn map_fpm(fpm: crate::engine::common::protocols::ForwardPassSnapshot) -> ForwardPassMetrics {
    ForwardPassMetrics {
        num_prefill_requests: fpm.num_prefill_requests,
        sum_prefill_tokens: fpm.sum_prefill_tokens,
        var_prefill_length: fpm.var_prefill_length,
        sum_prefill_kv_tokens: fpm.sum_prefill_kv_tokens,
        num_decode_requests: fpm.num_decode_requests,
        sum_decode_kv_tokens: fpm.sum_decode_kv_tokens,
        var_decode_kv_tokens: fpm.var_decode_kv_tokens,
        num_queued_prefill: fpm.num_queued_prefill,
        sum_queued_prefill_tokens: fpm.sum_queued_prefill_tokens,
        var_queued_prefill_length: fpm.var_queued_prefill_length,
        num_queued_decode: fpm.num_queued_decode,
        sum_queued_decode_kv_tokens: fpm.sum_queued_decode_kv_tokens,
        var_queued_decode_kv_tokens: fpm.var_queued_decode_kv_tokens,
        duration_ms: fpm.wall_time_secs * 1_000.0,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::engine::{PressureKind, TimingModelConfig};

    fn rank() -> SchedulerRank {
        rank_for_worker(WorkerType::Aggregated)
    }

    fn rank_for_worker(worker_type: WorkerType) -> SchedulerRank {
        let config = EngineConfig {
            worker_type,
            num_gpu_blocks: 8,
            block_size: 4,
            max_num_seqs: 2,
            max_num_batched_tokens: 16,
            speedup_ratio: 0.0,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 10.0,
            },
            ..EngineConfig::default()
        };
        SchedulerRank::new(
            RankIdentity {
                worker_id: 1,
                dp_rank: 0,
                dp_size: NonZeroU32::MIN,
            },
            &config,
        )
        .unwrap()
    }

    fn start_request_pass(
        rank: &mut SchedulerRank,
        request_id: Uuid,
        output_token_ids: Vec<u32>,
    ) -> PendingPass {
        let effects = rank
            .apply_command_effects(
                Command::Submit(Request {
                    request_id,
                    tokens: vec![1, 2, 3, 4],
                    max_output_tokens: output_token_ids.len(),
                    output_token_ids: Some(output_token_ids),
                }),
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(effects.result, CommandResult::Submitted(request_id));
        let pass = rank.execute_pass(0.0).unwrap();
        assert!(
            pass.pending
                .effects
                .outputs
                .iter()
                .any(|output| output.request_id == request_id)
        );
        pass.pending
    }

    #[test]
    fn ordinary_cancel_suppresses_pending_output_when_scheduler_state_is_removed() {
        let request_id = Uuid::from_u128(90_001);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5, 6]);

        let effects = rank
            .apply_command_effects(
                Command::CancelRequest {
                    request_id,
                    discard_pending_output: false,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, CommandResult::Applied);
        assert!(effects.suppressed_pending_output);
        assert!(pending.effects.outputs.is_empty());
    }

    #[test]
    fn ordinary_noop_cancel_preserves_pending_output() {
        let request_id = Uuid::from_u128(90_002);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5]);

        let effects = rank
            .apply_command_effects(
                Command::CancelRequest {
                    request_id,
                    discard_pending_output: false,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, CommandResult::Noop);
        assert!(!effects.suppressed_pending_output);
        assert_eq!(pending.effects.outputs.len(), 1);
    }

    #[test]
    fn explicit_discard_suppresses_pending_output_after_noop_cancellation() {
        let request_id = Uuid::from_u128(90_003);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5]);

        let effects = rank
            .apply_command_effects(
                Command::CancelRequest {
                    request_id,
                    discard_pending_output: true,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, CommandResult::Noop);
        assert!(effects.suppressed_pending_output);
        assert!(pending.effects.outputs.is_empty());
    }

    #[test]
    fn handoff_tracking_is_inserted_on_success_and_cleared_by_cancel() {
        let mut rank = rank_for_worker(WorkerType::Decode);
        let handoff_id = HandoffId::from(Uuid::from_u128(91_001));
        let request_id = Uuid::from_u128(91_002);
        let reservation = rank
            .apply_command_effects(
                Command::ReserveDestination {
                    handoff_id,
                    request: Request {
                        request_id,
                        tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 1,
                        output_token_ids: Some(vec![5]),
                    },
                },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert!(matches!(
            reservation.result,
            CommandResult::DestinationAccepted { .. }
        ));
        assert_eq!(rank.handoff_requests.get(&handoff_id), Some(&request_id));

        let cancellation = rank
            .apply_command_effects(
                Command::CancelDestination { handoff_id },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(cancellation.result, CommandResult::Applied);
        assert!(!rank.handoff_requests.contains_key(&handoff_id));
    }

    #[test]
    fn pass_start_exposes_vllm_preemption_pressure_event() {
        let config = EngineConfig {
            num_gpu_blocks: 6,
            block_size: 4,
            max_num_seqs: 2,
            max_num_batched_tokens: 16,
            enable_prefix_caching: false,
            enable_chunked_prefill: true,
            speedup_ratio: 0.0,
            preemption_mode: PreemptionMode::Lifo,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 10.0,
            },
            ..EngineConfig::default()
        };
        let mut rank = SchedulerRank::new(
            RankIdentity {
                worker_id: 7,
                dp_rank: 0,
                dp_size: NonZeroU32::MIN,
            },
            &config,
        )
        .unwrap();
        let first = Uuid::from_u128(92_001);
        let second = Uuid::from_u128(92_002);
        for (request_id, tokens) in [
            (first, (0..8).collect::<Vec<_>>()),
            (second, (100..108).collect::<Vec<_>>()),
        ] {
            let effects = rank
                .apply_command_effects(
                    Command::Submit(Request {
                        request_id,
                        tokens,
                        max_output_tokens: 8,
                        output_token_ids: None,
                    }),
                    CommandContext {
                        now_ms: 0.0,
                        pass_in_flight: false,
                    },
                    None,
                )
                .unwrap();
            assert_eq!(effects.result, CommandResult::Submitted(request_id));
        }

        let mut now_ms = 0.0;
        let mut observed = None;
        for _ in 0..16 {
            let pass = rank.execute_pass(now_ms).unwrap();
            if let Some(event) = pass.start_effects.pressure_events.first() {
                assert_eq!(pass.start_effects.pressure_events.len(), 1);
                observed = Some(event.clone());
            }
            let end_ms = pass.end_ms;
            rank.complete_pass(pass.pending, end_ms).unwrap();
            if observed.is_some() {
                break;
            }
            now_ms = end_ms.max(now_ms + 1.0);
        }

        let event = observed.expect("tight native G1 capacity should preempt one vLLM request");
        assert_eq!(event.at_ms, now_ms);
        assert_eq!(event.kind, PressureKind::VllmPreemption);
        assert_eq!(event.request_id, second);
        assert_eq!(event.state_before.running_requests, 2);
        assert_eq!(event.state_before.waiting_requests, Some(0));
        assert_eq!(event.state_after.running_requests, 1);
        assert_eq!(event.state_after.waiting_requests, Some(1));
        assert!(event.request_active_blocks_before > 0);
        assert!(event.state_after.active_blocks < event.state_before.active_blocks);
        assert_eq!(event.logical_available_blocks_before, None);
        assert_eq!(event.required_blocks_before, None);
    }

    #[test]
    fn terminal_handoff_output_clears_tracking() {
        let mut rank = rank_for_worker(WorkerType::Prefill);
        let handoff_id = HandoffId::from(Uuid::from_u128(92_001));
        let request_id = Uuid::from_u128(92_002);
        let submission = rank
            .apply_command_effects(
                Command::SubmitHandoffPrefill {
                    handoff_id,
                    request: Request {
                        request_id,
                        tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 1,
                        output_token_ids: Some(vec![5]),
                    },
                },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(submission.result, CommandResult::Submitted(request_id));
        assert_eq!(rank.handoff_requests.get(&handoff_id), Some(&request_id));

        let pass = rank.execute_pass(0.0).unwrap();
        let completion = rank.complete_pass(pass.pending, pass.end_ms).unwrap();
        assert!(
            completion
                .outputs
                .iter()
                .any(|output| output.request_id == request_id && output.completed)
        );
        assert!(!rank.handoff_requests.contains_key(&handoff_id));
    }
}

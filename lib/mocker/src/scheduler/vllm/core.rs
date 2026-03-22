// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use dynamo_tokens::blocks::UniqueBlock;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, KvCacheEventSink, MockEngineArgs, MoveBlock, OutputSignal, PreemptionMode,
    WorkerType,
};
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::KvManager;
use crate::replay::TraceCollector;
use crate::scheduler::{AdmissionEvent, EnginePassResult, KvEventBuffer, capture_kv_event_sink};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestStatus {
    Waiting,
    Running,
    Preempted,
}

pub(crate) struct VllmRequestState {
    pub(crate) sequence: ActiveSequence,
    pub(crate) status: RequestStatus,
    pub(crate) num_computed_tokens: usize,
    pub(crate) num_preemptions: usize,
}

#[derive(Default)]
pub(crate) struct SchedulerState {
    pub(crate) waiting: VecDeque<Uuid>,
    pub(crate) running: Vec<Uuid>,
    pub(crate) requests: HashMap<Uuid, VllmRequestState>,
}

struct PreemptedRequest {
    uuid: Uuid,
    signals: Vec<MoveBlock>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ScheduledWork {
    total_tokens: usize,
    prompt_tokens: usize,
    prefix_tokens: usize,
}

enum ScheduleOutcome {
    Scheduled {
        tokens_used: usize,
        admission: Option<AdmissionEvent>,
    },
    Blocked,
    CurrentPreempted,
}

impl SchedulerState {
    pub(crate) fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn push_waiting(&mut self, uuid: Uuid) {
        if self.waiting.contains(&uuid) {
            return;
        }
        self.waiting.push_back(uuid);
    }

    fn prepend_waiting(&mut self, uuid: Uuid) {
        if self.waiting.contains(&uuid) {
            self.waiting.retain(|queued| *queued != uuid);
        }
        self.waiting.push_front(uuid);
    }

    fn pop_front_waiting(&mut self, uuid: Uuid) {
        let Some(front) = self.waiting.front() else {
            return;
        };
        if *front == uuid {
            self.waiting.pop_front();
            return;
        }
        self.waiting.retain(|queued| *queued != uuid);
    }

    fn transition_to_running(&mut self, uuid: Uuid) {
        self.pop_front_waiting(uuid);
        if !self.running.contains(&uuid) {
            self.running.push(uuid);
        }
        if let Some(request) = self.requests.get_mut(&uuid) {
            request.status = RequestStatus::Running;
        }
    }

    pub(crate) fn complete(&mut self, uuid: &Uuid) {
        self.waiting.retain(|queued| queued != uuid);
        self.running.retain(|running| running != uuid);
        self.requests.remove(uuid);
    }

    pub(crate) fn running_sequence_mut(&mut self, uuid: Uuid) -> Option<&mut ActiveSequence> {
        if !self.running.contains(&uuid) {
            return None;
        }
        self.requests
            .get_mut(&uuid)
            .map(|request| &mut request.sequence)
    }

    fn preempt(&mut self, mode: PreemptionMode) -> Option<PreemptedRequest> {
        let idx = match mode {
            PreemptionMode::Lifo => self.running.len().checked_sub(1)?,
            PreemptionMode::Fifo => 0,
        };
        let uuid = self.running.remove(idx);
        let request = self.requests.get_mut(&uuid)?;
        request.status = RequestStatus::Preempted;
        request.num_computed_tokens = 0;
        request.num_preemptions += 1;
        let signals = request.sequence.reset_with_signal();
        self.prepend_waiting(uuid);
        Some(PreemptedRequest { uuid, signals })
    }
}

pub(crate) struct VllmCore {
    args: MockEngineArgs,
    pub(super) state: SchedulerState,
    pub(super) kv_manager: KvManager,
    kv_event_buffer: Option<KvEventBuffer>,
}

impl VllmCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        Self::new_internal(args, 0, None, None)
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        let (buffer, sink) = capture_kv_event_sink(worker_id);
        Self::new_internal(args, 0, Some(buffer), Some(sink))
    }

    pub(super) fn new_with_sink(
        args: MockEngineArgs,
        dp_rank: u32,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
    ) -> Self {
        Self::new_internal(args, dp_rank, None, kv_event_sink)
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        kv_event_buffer: Option<KvEventBuffer>,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
    ) -> Self {
        let args = args.normalized().expect("invalid MockEngineArgs");
        Self {
            kv_manager: KvManager::new_with_event_sink(
                args.num_gpu_blocks,
                args.block_size,
                kv_event_sink,
                dp_rank,
            ),
            args,
            state: SchedulerState::default(),
            kv_event_buffer,
        }
    }

    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        let sequence = ActiveSequence::new(
            request.tokens,
            request.max_output_tokens,
            Some(self.args.block_size),
            self.args.enable_prefix_caching,
            self.args.zmq_kv_events_port.is_some(),
        );
        self.state.requests.insert(
            uuid,
            VllmRequestState {
                sequence,
                status: RequestStatus::Waiting,
                num_computed_tokens: 0,
                num_preemptions: 0,
            },
        );
        self.state.push_waiting(uuid);
        uuid
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.state.requests.len()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.execute_pass_internal(Some(collector), now_ms, None)
    }

    pub(super) fn execute_pass_internal(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        now_ms: f64,
        admission_tx: Option<&mpsc::UnboundedSender<AdmissionEvent>>,
    ) -> EnginePassResult {
        let requests_before = self.state.requests.len();
        let mut token_budget = self.args.max_num_batched_tokens.unwrap_or(usize::MAX);
        let mut scheduled = HashMap::new();
        let mut batch_count = 0usize;
        let mut batch_total_isl = 0usize;
        let mut batch_total_prefix = 0usize;
        let mut admissions = Vec::new();
        let mut preempted_any = false;

        let mut req_index = 0usize;
        while req_index < self.state.running.len() && token_budget > 0 {
            let uuid = self.state.running[req_index];
            match self.schedule_request(
                uuid,
                false,
                &mut token_budget,
                &mut scheduled,
                &mut batch_count,
                &mut batch_total_isl,
                &mut batch_total_prefix,
                &mut preempted_any,
            ) {
                ScheduleOutcome::Scheduled { admission, .. } => {
                    if let Some(admission) = admission {
                        if let Some(collector) = collector.as_deref_mut() {
                            collector.on_admit(
                                admission.uuid,
                                now_ms,
                                admission.reused_input_tokens,
                            );
                        }
                        if let Some(admission_tx) = admission_tx {
                            let _ = admission_tx.send(admission.clone());
                        }
                        admissions.push(admission);
                    }
                    req_index += 1;
                }
                ScheduleOutcome::Blocked => break,
                ScheduleOutcome::CurrentPreempted => {}
            }
        }

        let max_num_running = self.args.max_num_seqs.unwrap_or(usize::MAX);
        while !preempted_any && self.state.running.len() < max_num_running {
            let Some(&uuid) = self.state.waiting.front() else {
                break;
            };
            match self.schedule_request(
                uuid,
                true,
                &mut token_budget,
                &mut scheduled,
                &mut batch_count,
                &mut batch_total_isl,
                &mut batch_total_prefix,
                &mut preempted_any,
            ) {
                ScheduleOutcome::Scheduled {
                    admission,
                    tokens_used,
                } => {
                    if let Some(admission) = admission {
                        if let Some(collector) = collector.as_deref_mut() {
                            collector.on_admit(
                                admission.uuid,
                                now_ms,
                                admission.reused_input_tokens,
                            );
                        }
                        if let Some(admission_tx) = admission_tx {
                            let _ = admission_tx.send(admission.clone());
                        }
                        admissions.push(admission);
                    }
                    if tokens_used == 0 && token_budget == 0 {
                        break;
                    }
                }
                ScheduleOutcome::Blocked | ScheduleOutcome::CurrentPreempted => break,
            }
        }

        let prefill_time =
            predict_prefill_duration(batch_count, batch_total_isl, batch_total_prefix, &self.args);
        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let (decode_time, output_signals) = self.emit_ready_tokens(collector, decode_start_ms);
        let end_ms = decode_start_ms + decode_time.as_secs_f64() * 1000.0;

        EnginePassResult {
            end_ms,
            completed_requests: requests_before.saturating_sub(self.state.requests.len()),
            output_signals,
            admissions,
            active_decode_blocks: self.kv_manager.num_active_blocks() as u64,
            kv_events: self
                .kv_event_buffer
                .as_ref()
                .map(KvEventBuffer::drain)
                .unwrap_or_default(),
        }
    }

    pub(super) fn drop_request(&mut self, uuid: Uuid) {
        let Some(request) = self.state.requests.get(&uuid) else {
            return;
        };
        for signal in request.sequence.free_signal() {
            self.kv_manager.process(&signal);
        }
        self.state.complete(&uuid);
    }

    #[allow(clippy::too_many_arguments)]
    fn schedule_request(
        &mut self,
        uuid: Uuid,
        from_waiting: bool,
        token_budget: &mut usize,
        scheduled: &mut HashMap<Uuid, ScheduledWork>,
        batch_count: &mut usize,
        batch_total_isl: &mut usize,
        batch_total_prefix: &mut usize,
        preempted_any: &mut bool,
    ) -> ScheduleOutcome {
        let Some(request) = self.state.requests.get(&uuid) else {
            return ScheduleOutcome::Blocked;
        };
        let prefill_cost = self.kv_manager.get_prefill_cost(&request.sequence);
        let cached_prefix_tokens = if request.num_computed_tokens == 0 {
            prefill_cost.cached_tokens
        } else {
            0
        };
        let effective_computed_before = request.num_computed_tokens + cached_prefix_tokens;
        let prompt_len = request.sequence.num_input_tokens();
        let prompt_before = effective_computed_before.min(prompt_len);
        let remaining_known_tokens = request
            .sequence
            .len()
            .saturating_sub(effective_computed_before);
        let prompt_remaining = prompt_len.saturating_sub(prompt_before);
        if prompt_remaining > 0
            && !self.args.enable_chunked_prefill
            && prompt_remaining > *token_budget
        {
            return ScheduleOutcome::Blocked;
        }

        let desired_tokens = remaining_known_tokens.min(*token_budget);
        if desired_tokens == 0 && remaining_known_tokens > 0 {
            return ScheduleOutcome::Blocked;
        }

        let desired_computed_after = effective_computed_before + desired_tokens;
        let mut actual_computed_after = desired_computed_after;

        loop {
            let allocation = {
                let Some(request) = self.state.requests.get_mut(&uuid) else {
                    return ScheduleOutcome::Blocked;
                };
                let allocation_target = if request.status == RequestStatus::Running {
                    desired_computed_after.min(request.sequence.num_input_tokens())
                } else {
                    desired_computed_after
                };
                let prev_allocated_tokens = request.sequence.num_allocated_tokens();
                if allocation_target <= prev_allocated_tokens {
                    request.num_computed_tokens = actual_computed_after;
                    None
                } else {
                    let maybe_signal = request.sequence.prepare_allocation(allocation_target);
                    Some((allocation_target, prev_allocated_tokens, maybe_signal))
                }
            };
            let Some((allocation_target, prev_allocated_tokens, maybe_signal)) = allocation else {
                break;
            };
            let Some(signal) = maybe_signal else {
                let Some(request) = self.state.requests.get_mut(&uuid) else {
                    return ScheduleOutcome::Blocked;
                };
                request.sequence.commit_allocation(allocation_target);
                request.num_computed_tokens = actual_computed_after;
                break;
            };

            let expected = match &signal {
                MoveBlock::Use(blocks, ..) => blocks.len(),
                _ => unreachable!(),
            };
            let allocated = self.kv_manager.process(&signal);
            let (_committed_tokens, current_computed_tokens) = {
                let Some(request) = self.state.requests.get_mut(&uuid) else {
                    return ScheduleOutcome::Blocked;
                };
                let committed_tokens = if allocated == expected {
                    allocation_target
                } else {
                    let prev_blocks = prev_allocated_tokens
                        .div_ceil(request.sequence.block_size())
                        .min(request.sequence.unique_blocks().len());
                    (prev_blocks + allocated) * request.sequence.block_size()
                };
                request
                    .sequence
                    .commit_allocation(committed_tokens.min(allocation_target));
                request.num_computed_tokens = actual_computed_after.min(committed_tokens);
                (committed_tokens, request.num_computed_tokens)
            };
            if allocated == expected {
                break;
            }

            let Some(preempted) = self.state.preempt(self.args.preemption_mode) else {
                actual_computed_after = current_computed_tokens;
                break;
            };
            for signal in preempted.signals {
                self.kv_manager.process(&signal);
            }
            *preempted_any = true;
            if let Some(undone) = scheduled.remove(&preempted.uuid) {
                *token_budget += undone.total_tokens;
                if undone.prompt_tokens > 0 && self.args.worker_type != WorkerType::Decode {
                    *batch_count = batch_count.saturating_sub(1);
                    *batch_total_isl =
                        batch_total_isl.saturating_sub(undone.prefix_tokens + undone.prompt_tokens);
                    *batch_total_prefix = batch_total_prefix.saturating_sub(undone.prefix_tokens);
                }
            }
            if preempted.uuid == uuid {
                return ScheduleOutcome::CurrentPreempted;
            }
        }

        let tokens_used = actual_computed_after.saturating_sub(effective_computed_before);
        if tokens_used == 0
            && actual_computed_after < request_sequence_len(&self.state.requests, uuid)
        {
            return ScheduleOutcome::Blocked;
        }

        let prompt_after = actual_computed_after.min(prompt_len);
        let prompt_tokens = prompt_after.saturating_sub(prompt_before);
        scheduled.insert(
            uuid,
            ScheduledWork {
                total_tokens: tokens_used,
                prompt_tokens,
                prefix_tokens: prompt_before,
            },
        );
        if prompt_tokens > 0 && self.args.worker_type != WorkerType::Decode {
            *batch_count += 1;
            *batch_total_isl += prompt_before + prompt_tokens;
            *batch_total_prefix += prompt_before;
        }

        if from_waiting {
            self.state.transition_to_running(uuid);
        }
        *token_budget = token_budget.saturating_sub(tokens_used);

        let admission = if from_waiting {
            Some(AdmissionEvent {
                uuid,
                reused_input_tokens: cached_prefix_tokens,
            })
        } else {
            None
        };
        ScheduleOutcome::Scheduled {
            tokens_used,
            admission,
        }
    }

    fn emit_ready_tokens(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        decode_start_ms: f64,
    ) -> (Duration, Vec<OutputSignal>) {
        let ready = self
            .state
            .running
            .iter()
            .copied()
            .filter(|uuid| {
                let Some(request) = self.state.requests.get(uuid) else {
                    return false;
                };
                request.num_computed_tokens >= request.sequence.len()
                    && request.sequence.generated_tokens() < request.sequence.max_output_tokens()
            })
            .collect::<Vec<_>>();
        if ready.is_empty() {
            return (Duration::ZERO, Vec::new());
        }

        let active_kv_tokens = self.kv_manager.num_active_blocks() * self.args.block_size;
        let total_length = ready
            .iter()
            .filter_map(|uuid| self.state.requests.get(uuid))
            .map(|request| request.sequence.len())
            .sum::<usize>();
        let context_length = total_length / ready.len();
        let decode_ms =
            self.args
                .perf_model
                .predict_decode_time(ready.len(), active_kv_tokens, context_length);
        let decode_time = scale_decode_time(decode_ms, &self.args);
        let decode_end_ms = decode_start_ms + decode_time.as_secs_f64() * 1000.0;

        let mut output_signals = Vec::with_capacity(ready.len());
        for uuid in ready {
            let mut emitted = false;
            loop {
                let Some(sequence) = self.state.running_sequence_mut(uuid) else {
                    break;
                };
                let signals = sequence.generate();
                if process_signals(&mut self.kv_manager, &signals) {
                    emitted = true;
                    break;
                }
                sequence.pop();

                let Some(preempted) = self.state.preempt(self.args.preemption_mode) else {
                    break;
                };
                for signal in preempted.signals {
                    self.kv_manager.process(&signal);
                }
                if preempted.uuid == uuid {
                    break;
                }
            }
            if !emitted {
                continue;
            }

            let Some(request) = self.state.requests.get(&uuid) else {
                continue;
            };
            if let Some(collector) = collector.as_deref_mut() {
                collector.on_token(uuid, decode_end_ms);
            }
            let completed =
                request.sequence.generated_tokens() >= request.sequence.max_output_tokens();
            output_signals.push(OutputSignal { uuid, completed });
            if completed {
                self.state.complete(&uuid);
            }
        }

        if output_signals.is_empty() {
            return (Duration::ZERO, output_signals);
        }

        (decode_time, output_signals)
    }
}

fn request_sequence_len(requests: &HashMap<Uuid, VllmRequestState>, uuid: Uuid) -> usize {
    requests
        .get(&uuid)
        .map(|request| request.sequence.len())
        .unwrap_or_default()
}

fn predict_prefill_duration(
    batch_count: usize,
    batch_total_isl: usize,
    batch_total_prefix: usize,
    args: &MockEngineArgs,
) -> Duration {
    if batch_count == 0 || args.worker_type == WorkerType::Decode {
        return Duration::ZERO;
    }

    let mean_isl = batch_total_isl / batch_count;
    let mean_prefix = batch_total_prefix / batch_count;
    let prefill_ms = args
        .perf_model
        .predict_prefill_time(batch_count, mean_isl, mean_prefix);
    let total_time = Duration::from_secs_f64(prefill_ms / 1000.0);
    if args.speedup_ratio <= 0.0 || total_time <= Duration::ZERO {
        return total_time;
    }
    Duration::from_secs_f64(total_time.as_secs_f64() / args.speedup_ratio)
}

fn scale_decode_time(decode_ms: f64, args: &MockEngineArgs) -> Duration {
    let unscaled = Duration::from_secs_f64(decode_ms / 1000.0);
    let effective_ratio = args.speedup_ratio * args.decode_speedup_ratio;
    if effective_ratio <= 0.0 || unscaled <= Duration::ZERO {
        return unscaled;
    }
    Duration::from_secs_f64(unscaled.as_secs_f64() / effective_ratio)
}

fn process_signals(kv_manager: &mut KvManager, signals: &[MoveBlock]) -> bool {
    for signal in signals {
        if kv_manager.process(signal) > 0 {
            continue;
        }

        let MoveBlock::Use(blocks, ..) = signal else {
            panic!("Failed signal is invalid. Expected decode allocation failure, got {signal:?}");
        };
        if blocks.len() != 1 {
            panic!(
                "Failed signal is invalid. Tried to allocate {} blocks during decode.",
                blocks.len()
            );
        }
        if !matches!(blocks[0], UniqueBlock::PartialBlock(_)) {
            panic!("Failed signal is invalid. Decode allocation must use a partial block.");
        }
        return false;
    }
    true
}

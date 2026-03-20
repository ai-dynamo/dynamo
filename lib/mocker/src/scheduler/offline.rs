// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::vllm::{Request, SchedulerState, simulate_decode_step, simulate_prefill_step};
use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::common::running_mean::RunningMean;
use crate::kv_manager::KvManager;
use crate::simulation::{TraceCollector, TraceSimulationReport};
use anyhow::bail;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use tokio::sync::mpsc;
use uuid::Uuid;
use validator::Validate;

#[derive(Debug, Clone, Copy)]
enum ReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[derive(Debug, Clone)]
struct ClusterSnapshot {
    now_ms: f64,
    cluster_in_flight: usize,
    per_worker_in_flight: Vec<usize>,
}

struct OfflineWorkerState {
    worker_idx: usize,
    scheduler: SchedulerState,
    kv_manager: KvManager,
    hit_rates: RunningMean<f32>,
    busy: bool,
}

impl OfflineWorkerState {
    fn new(worker_idx: usize, args: &MockEngineArgs) -> Self {
        Self {
            worker_idx,
            scheduler: SchedulerState::default(),
            kv_manager: KvManager::new(args.num_gpu_blocks, args.block_size),
            hit_rates: RunningMean::new(1000),
            busy: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WorkerCompletion {
    at_ms: f64,
    worker_idx: usize,
    completed_requests: usize,
}

impl PartialEq for WorkerCompletion {
    fn eq(&self, other: &Self) -> bool {
        self.at_ms.to_bits() == other.at_ms.to_bits()
            && self.worker_idx == other.worker_idx
            && self.completed_requests == other.completed_requests
    }
}

impl Eq for WorkerCompletion {}

impl PartialOrd for WorkerCompletion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkerCompletion {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at_ms
            .partial_cmp(&self.at_ms)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.worker_idx.cmp(&self.worker_idx))
            .then_with(|| other.completed_requests.cmp(&self.completed_requests))
    }
}

struct OfflineRuntime {
    snapshot: ClusterSnapshot,
    next_worker_idx: usize,
    pending: VecDeque<DirectRequest>,
    workers: Vec<OfflineWorkerState>,
    collector: TraceCollector,
    completions: BinaryHeap<WorkerCompletion>,
    mode: ReplayMode,
}

#[derive(Debug)]
struct ExecutedPass {
    end_ms: f64,
    completed_requests: usize,
    made_progress: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkerProgressSnapshot {
    waiting_len: usize,
    prefill_len: usize,
    decode_len: usize,
    request_count: usize,
    total_generated_tokens: usize,
    total_allocated_tokens: usize,
    active_blocks: usize,
}

fn snapshot_worker_progress(worker: &OfflineWorkerState) -> WorkerProgressSnapshot {
    let mut total_generated_tokens = 0;
    let mut total_allocated_tokens = 0;

    for request in worker.scheduler.requests.values() {
        if let Request::Active(sequence) = request {
            total_generated_tokens += sequence.generated_tokens();
            total_allocated_tokens += sequence.num_allocated_tokens();
        }
    }

    WorkerProgressSnapshot {
        waiting_len: worker.scheduler.waiting.len(),
        prefill_len: worker.scheduler.prefill.len(),
        decode_len: worker.scheduler.decode.len(),
        request_count: worker.scheduler.requests.len(),
        total_generated_tokens,
        total_allocated_tokens,
        active_blocks: worker.kv_manager.num_active_blocks(),
    }
}

fn normalize_trace_requests(
    mut requests: Vec<DirectRequest>,
) -> anyhow::Result<VecDeque<DirectRequest>> {
    requests.sort_by(|left, right| {
        let left_ts = left
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        let right_ts = right
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        left_ts.total_cmp(&right_ts)
    });

    let first_arrival_ms = requests
        .first()
        .and_then(|request| request.arrival_timestamp_ms)
        .ok_or_else(|| anyhow::anyhow!("trace replay requires at least one timestamped request"))?;

    Ok(VecDeque::from(
        requests
            .into_iter()
            .map(|mut request| {
                let arrival_timestamp_ms = request
                    .arrival_timestamp_ms
                    .expect("trace replay requests must have an arrival timestamp")
                    - first_arrival_ms;
                request.arrival_timestamp_ms = Some(arrival_timestamp_ms);
                request
            })
            .collect::<Vec<_>>(),
    ))
}

fn assign_request(
    runtime: &mut OfflineRuntime,
    mut request: DirectRequest,
    arrival_time_ms: f64,
) -> Uuid {
    let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
    request.uuid = Some(uuid);
    if matches!(runtime.mode, ReplayMode::Concurrency { .. }) {
        request.arrival_timestamp_ms = Some(arrival_time_ms);
    }

    runtime.collector.on_arrival(
        uuid,
        arrival_time_ms,
        request.tokens.len(),
        request.max_output_tokens,
    );

    let worker_idx = runtime.next_worker_idx;
    runtime.next_worker_idx = (runtime.next_worker_idx + 1) % runtime.workers.len();

    runtime.snapshot.cluster_in_flight += 1;
    runtime.snapshot.per_worker_in_flight[worker_idx] += 1;

    // TODO: If future cluster scheduling needs worker-local metrics beyond the
    // in-flight counts in ClusterSnapshot, promote those metrics into the
    // snapshot or move to a finer event model.
    runtime.workers[worker_idx].scheduler.receive(request);

    uuid
}

fn execute_worker_pass(
    worker: &mut OfflineWorkerState,
    args: &MockEngineArgs,
    collector: &mut TraceCollector,
    now_ms: f64,
) -> ExecutedPass {
    let before = snapshot_worker_progress(worker);
    let requests_before = worker.scheduler.requests.len();
    let output_tx: Option<mpsc::UnboundedSender<OutputSignal>> = None;

    let prefill_time = simulate_prefill_step(
        &mut worker.scheduler,
        &mut worker.kv_manager,
        &mut worker.hit_rates,
        args,
        Some(collector),
        now_ms,
        true,
    );
    let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
    let decode_time = simulate_decode_step(
        &mut worker.scheduler,
        &mut worker.kv_manager,
        &output_tx,
        args,
        Some(collector),
        decode_start_ms,
        true,
    );
    let end_ms = decode_start_ms + decode_time.as_secs_f64() * 1000.0;

    let after = snapshot_worker_progress(worker);
    let requests_after = worker.scheduler.requests.len();

    ExecutedPass {
        end_ms,
        completed_requests: requests_before.saturating_sub(requests_after),
        made_progress: end_ms > now_ms || before != after,
    }
}

fn simulation_done(runtime: &OfflineRuntime) -> bool {
    runtime.pending.is_empty()
        && runtime.completions.is_empty()
        && runtime.snapshot.cluster_in_flight == 0
        && runtime
            .snapshot
            .per_worker_in_flight
            .iter()
            .all(|count| *count == 0)
        && runtime
            .workers
            .iter()
            .all(|worker| !worker.busy && worker.scheduler.is_empty())
}

fn next_runtime_timestamp(runtime: &OfflineRuntime) -> Option<f64> {
    let next_completion_ms = runtime
        .completions
        .peek()
        .map(|completion| completion.at_ms);
    let next_arrival_ms = match runtime.mode {
        ReplayMode::Trace => runtime
            .pending
            .front()
            .and_then(|request| request.arrival_timestamp_ms),
        ReplayMode::Concurrency { .. } => None,
    };

    match (next_arrival_ms, next_completion_ms) {
        (Some(arrival_ms), Some(completion_ms)) => Some(arrival_ms.min(completion_ms)),
        (Some(arrival_ms), None) => Some(arrival_ms),
        (None, Some(completion_ms)) => Some(completion_ms),
        (None, None) => None,
    }
}

fn apply_worker_completions(runtime: &mut OfflineRuntime) {
    while runtime
        .completions
        .peek()
        .is_some_and(|completion| completion.at_ms == runtime.snapshot.now_ms)
    {
        let completion = runtime
            .completions
            .pop()
            .expect("completion must exist after peek");
        let worker = &mut runtime.workers[completion.worker_idx];
        worker.busy = false;
        runtime.snapshot.cluster_in_flight = runtime
            .snapshot
            .cluster_in_flight
            .saturating_sub(completion.completed_requests);
        runtime.snapshot.per_worker_in_flight[completion.worker_idx] =
            runtime.snapshot.per_worker_in_flight[completion.worker_idx]
                .saturating_sub(completion.completed_requests);
    }
}

fn release_trace_arrivals(runtime: &mut OfflineRuntime) {
    while runtime
        .pending
        .front()
        .and_then(|request| request.arrival_timestamp_ms)
        .is_some_and(|arrival_ms| arrival_ms <= runtime.snapshot.now_ms)
    {
        let request = runtime
            .pending
            .pop_front()
            .expect("front request must exist when arrival is ready");
        let arrival_ms = request
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        assign_request(runtime, request, arrival_ms);
    }
}

fn top_off_concurrency(runtime: &mut OfflineRuntime, max_in_flight: usize) {
    while runtime.snapshot.cluster_in_flight < max_in_flight {
        let Some(request) = runtime.pending.pop_front() else {
            break;
        };
        assign_request(runtime, request, runtime.snapshot.now_ms);
    }
}

fn drive_ready_workers(runtime: &mut OfflineRuntime, args: &MockEngineArgs) -> anyhow::Result<()> {
    for worker_idx in 0..runtime.workers.len() {
        loop {
            if runtime.workers[worker_idx].busy {
                break;
            }
            if runtime.workers[worker_idx].scheduler.is_empty() {
                break;
            }

            let executed = {
                let (workers, collector) = (&mut runtime.workers, &mut runtime.collector);
                execute_worker_pass(
                    &mut workers[worker_idx],
                    args,
                    collector,
                    runtime.snapshot.now_ms,
                )
            };

            if !executed.made_progress {
                bail!(
                    "offline replay worker {} made no progress at {} ms",
                    runtime.workers[worker_idx].worker_idx,
                    runtime.snapshot.now_ms
                );
            }

            if executed.end_ms == runtime.snapshot.now_ms {
                runtime.snapshot.cluster_in_flight = runtime
                    .snapshot
                    .cluster_in_flight
                    .saturating_sub(executed.completed_requests);
                runtime.snapshot.per_worker_in_flight[worker_idx] =
                    runtime.snapshot.per_worker_in_flight[worker_idx]
                        .saturating_sub(executed.completed_requests);
                continue;
            }

            runtime.workers[worker_idx].busy = true;
            runtime.completions.push(WorkerCompletion {
                at_ms: executed.end_ms,
                worker_idx,
                completed_requests: executed.completed_requests,
            });
            break;
        }
    }

    Ok(())
}

fn run_offline_simulation(
    args: MockEngineArgs,
    pending: VecDeque<DirectRequest>,
    num_workers: usize,
    mode: ReplayMode,
) -> anyhow::Result<TraceCollector> {
    let mut runtime = OfflineRuntime {
        snapshot: ClusterSnapshot {
            now_ms: 0.0,
            cluster_in_flight: 0,
            per_worker_in_flight: vec![0; num_workers],
        },
        next_worker_idx: 0,
        pending,
        workers: (0..num_workers)
            .map(|worker_idx| OfflineWorkerState::new(worker_idx, &args))
            .collect(),
        collector: TraceCollector::default(),
        completions: BinaryHeap::new(),
        mode,
    };

    match runtime.mode {
        ReplayMode::Trace => release_trace_arrivals(&mut runtime),
        ReplayMode::Concurrency { max_in_flight } => {
            top_off_concurrency(&mut runtime, max_in_flight);
        }
    }
    drive_ready_workers(&mut runtime, &args)?;

    while !simulation_done(&runtime) {
        let Some(next_timestamp_ms) = next_runtime_timestamp(&runtime) else {
            bail!(
                "offline replay reached a dead end with {} in-flight requests remaining",
                runtime.snapshot.cluster_in_flight
            );
        };

        runtime.snapshot.now_ms = next_timestamp_ms;
        apply_worker_completions(&mut runtime);

        match runtime.mode {
            ReplayMode::Trace => release_trace_arrivals(&mut runtime),
            ReplayMode::Concurrency { max_in_flight } => {
                top_off_concurrency(&mut runtime, max_in_flight);
            }
        }
        drive_ready_workers(&mut runtime, &args)?;
    }

    Ok(runtime.collector)
}

pub fn simulate_trace_multi(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
) -> anyhow::Result<TraceSimulationReport> {
    args.validate()?;
    let pending = normalize_trace_requests(requests)?;
    let collector = run_offline_simulation(args, pending, num_workers, ReplayMode::Trace)?;
    Ok(collector.finish())
}

pub fn simulate_concurrency_multi(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> anyhow::Result<TraceSimulationReport> {
    args.validate()?;
    let pending = VecDeque::from(requests);
    let collector = run_offline_simulation(
        args,
        pending,
        num_workers,
        ReplayMode::Concurrency { max_in_flight },
    )?;
    Ok(collector.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay_args(enable_prefix_caching: bool, enable_chunked_prefill: bool) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(enable_chunked_prefill)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn run_trace_multi_collect(
        args: &MockEngineArgs,
        requests: Vec<DirectRequest>,
        num_workers: usize,
    ) -> TraceCollector {
        let pending = normalize_trace_requests(requests).unwrap();
        run_offline_simulation(args.clone(), pending, num_workers, ReplayMode::Trace).unwrap()
    }

    fn run_concurrency_multi_collect(
        args: &MockEngineArgs,
        requests: Vec<DirectRequest>,
        max_in_flight: usize,
        num_workers: usize,
    ) -> TraceCollector {
        run_offline_simulation(
            args.clone(),
            VecDeque::from(requests),
            num_workers,
            ReplayMode::Concurrency { max_in_flight },
        )
        .unwrap()
    }

    #[test]
    fn test_multi_worker_trace_round_robin_assigns_same_timestamp_requests_deterministically() {
        let args = replay_args(false, true);
        let collector = run_trace_multi_collect(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                    max_output_tokens: 4,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
                },
                DirectRequest {
                    tokens: vec![7, 7, 7, 7, 8, 8, 8, 8],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(44)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
                },
            ],
            2,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(11)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(22)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(33)).unwrap();
        let request_4 = collector.snapshot(Uuid::from_u128(44)).unwrap();
        let report = collector.finish();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, 1.0);
        assert_eq!(request_4.arrival_time_ms, 1.0);

        assert!(request_3.first_admit_ms.unwrap() >= request_1.first_token_ms.unwrap());
        assert!(request_4.first_admit_ms.unwrap() >= request_2.first_token_ms.unwrap());
        assert!(request_3.first_admit_ms.unwrap() < request_4.first_admit_ms.unwrap());

        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(report.request_counts.total_input_tokens, 40);
        assert_eq!(report.request_counts.total_output_tokens, 10);
    }

    #[test]
    fn test_multi_worker_concurrency_uses_cluster_snapshot_for_cap_checks() {
        let args = replay_args(false, false);
        let collector = run_concurrency_multi_collect(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(900.0),
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4],
                    max_output_tokens: 4,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(1000.0),
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
            ],
            2,
            2,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(11)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(22)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(33)).unwrap();
        let report = collector.finish();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, request_1.last_token_ms.unwrap());
        assert!(request_3.arrival_time_ms < request_2.last_token_ms.unwrap());
        assert_eq!(request_3.first_admit_ms.unwrap(), request_3.arrival_time_ms);

        assert_eq!(report.request_counts.completed_requests, 3);
        assert_eq!(report.request_counts.total_input_tokens, 24);
        assert_eq!(report.request_counts.total_output_tokens, 8);
    }
}

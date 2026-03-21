// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashSet, VecDeque};

use anyhow::{Result, anyhow, bail};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{TraceCollector, TraceSimulationReport, normalize_trace_requests};
use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::scheduler::{AdmissionEvent, Scheduler, SchedulerHandle};

#[derive(Clone, Copy, Debug)]
enum LiveReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[derive(Debug, Default, PartialEq, Eq)]
struct LiveRuntimeStats {
    dispatch_history: Vec<usize>,
    max_in_flight_seen: usize,
}

struct LiveRuntime {
    collector: TraceCollector,
    pending: VecDeque<DirectRequest>,
    senders: Vec<mpsc::UnboundedSender<DirectRequest>>,
    schedulers: Vec<Scheduler>,
    output_rx: mpsc::UnboundedReceiver<OutputSignal>,
    admission_rx: mpsc::UnboundedReceiver<AdmissionEvent>,
    cancel_token: CancellationToken,
    start: Instant,
    mode: LiveReplayMode,
    next_worker_idx: usize,
    total_requests: usize,
    in_flight: usize,
    completed: HashSet<Uuid>,
    stats: LiveRuntimeStats,
}

impl LiveRuntime {
    fn new(
        args: MockEngineArgs,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: LiveReplayMode,
    ) -> Result<Self> {
        let total_requests = pending.len();
        if total_requests == 0 {
            bail!("online replay requires at least one request");
        }

        let cancel_token = CancellationToken::new();
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let (admission_tx, admission_rx) = mpsc::unbounded_channel();
        let mut schedulers = Vec::with_capacity(num_workers);
        let mut senders = Vec::with_capacity(num_workers);

        for worker_idx in 0..num_workers {
            let scheduler = Scheduler::new_with_admission(
                args.clone(),
                worker_idx as u32,
                Some(output_tx.clone()),
                None,
                Some(cancel_token.clone()),
                Some(admission_tx.clone()),
            );
            senders.push(scheduler.request_sender());
            schedulers.push(scheduler);
        }

        drop(output_tx);
        drop(admission_tx);

        Ok(Self {
            collector: TraceCollector::default(),
            pending,
            senders,
            schedulers,
            output_rx,
            admission_rx,
            cancel_token,
            start: Instant::now(),
            mode,
            next_worker_idx: 0,
            total_requests,
            in_flight: 0,
            completed: HashSet::with_capacity(total_requests),
            stats: LiveRuntimeStats::default(),
        })
    }

    fn now_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    fn dispatch_request(&mut self, request: DirectRequest) -> Result<()> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("online replay requires requests to have stable UUIDs"))?;
        let worker_idx = self.next_worker_idx % self.senders.len();
        self.next_worker_idx = (self.next_worker_idx + 1) % self.senders.len();
        self.collector.on_arrival(
            uuid,
            self.now_ms(),
            request.tokens.len(),
            request.max_output_tokens,
        );
        self.senders[worker_idx].send(request).map_err(|_| {
            anyhow!("online replay failed to dispatch request to worker {worker_idx}")
        })?;
        self.in_flight += 1;
        self.stats.dispatch_history.push(worker_idx);
        self.stats.max_in_flight_seen = self.stats.max_in_flight_seen.max(self.in_flight);
        Ok(())
    }

    fn release_trace_ready_requests(&mut self) -> Result<()> {
        loop {
            let should_release = self
                .pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms)
                .is_some_and(|arrival_ms| arrival_ms <= self.now_ms());
            if !should_release {
                return Ok(());
            }

            let request = self
                .pending
                .pop_front()
                .expect("pending trace request should exist");
            self.dispatch_request(request)?;
        }
    }

    fn top_off_concurrency(&mut self, max_in_flight: usize) -> Result<()> {
        while self.in_flight < max_in_flight {
            let Some(request) = self.pending.pop_front() else {
                break;
            };
            self.dispatch_request(request)?;
        }
        Ok(())
    }

    fn handle_admission(&mut self, event: AdmissionEvent) {
        self.collector
            .on_admit(event.uuid, self.now_ms(), event.reused_input_tokens);
    }

    fn handle_output(&mut self, signal: OutputSignal) {
        self.collector.on_token(signal.uuid, self.now_ms());
        if signal.completed && self.completed.insert(signal.uuid) {
            self.in_flight = self.in_flight.saturating_sub(1);
        }
    }

    fn next_trace_deadline(&self) -> Option<Instant> {
        let arrival_ms = self.pending.front()?.arrival_timestamp_ms?;
        Some(self.start + tokio::time::Duration::from_secs_f64(arrival_ms / 1000.0))
    }

    async fn run(mut self) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
        match self.mode {
            LiveReplayMode::Trace => self.release_trace_ready_requests()?,
            LiveReplayMode::Concurrency { max_in_flight } => {
                self.top_off_concurrency(max_in_flight)?
            }
        }

        while self.completed.len() < self.total_requests {
            match self.mode {
                LiveReplayMode::Trace => self.release_trace_ready_requests()?,
                LiveReplayMode::Concurrency { max_in_flight } => {
                    self.top_off_concurrency(max_in_flight)?
                }
            }

            if let Some(deadline) = self.next_trace_deadline() {
                tokio::select! {
                    output = self.output_rx.recv() => {
                        let Some(output) = output else {
                            bail!("online replay output channel closed before all requests completed");
                        };
                        self.handle_output(output);
                    }
                    admission = self.admission_rx.recv() => {
                        if let Some(admission) = admission {
                            self.handle_admission(admission);
                        }
                    }
                    _ = tokio::time::sleep_until(deadline) => {}
                }
                continue;
            }

            tokio::select! {
                output = self.output_rx.recv() => {
                    let Some(output) = output else {
                        bail!("online replay output channel closed before all requests completed");
                    };
                    self.handle_output(output);
                }
                admission = self.admission_rx.recv() => {
                    if let Some(admission) = admission {
                        self.handle_admission(admission);
                    }
                }
            }
        }

        self.cancel_token.cancel();
        self.schedulers.clear();

        let wall_time_ms = self.now_ms();
        let report = self.collector.finish().with_wall_time_ms(wall_time_ms);
        Ok((report, self.stats))
    }
}

fn run_live_runtime(
    args: MockEngineArgs,
    pending: VecDeque<DirectRequest>,
    num_workers: usize,
    mode: LiveReplayMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to create online replay runtime: {e}"))?;

    runtime.block_on(async move {
        LiveRuntime::new(args, pending, num_workers, mode)?
            .run()
            .await
    })
}

pub(crate) fn simulate_trace_live_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    let pending = normalize_trace_requests(requests)?;
    let (report, _) = run_live_runtime(args, pending, num_workers, LiveReplayMode::Trace)?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_live_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    if requests.is_empty() {
        bail!("online concurrency replay requires at least one request");
    }

    let pending = VecDeque::from(requests);
    let (report, _) = run_live_runtime(
        args,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
    )?;
    Ok(report)
}

#[cfg(test)]
fn simulate_trace_live_requests_with_stats(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let pending = normalize_trace_requests(requests)?;
    run_live_runtime(args, pending, num_workers, LiveReplayMode::Trace)
}

#[cfg(test)]
fn simulate_concurrency_live_requests_with_stats(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let pending = VecDeque::from(requests);
    run_live_runtime(
        args,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::DirectRequest;

    fn request(uuid: u128, token: u32, arrival_timestamp_ms: Option<f64>) -> DirectRequest {
        DirectRequest {
            tokens: vec![token; 64],
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms,
        }
    }

    #[test]
    fn test_online_trace_replay_single_worker_completes() {
        let args = MockEngineArgs::builder()
            .speedup_ratio(1000.0)
            .block_size(64)
            .build()
            .unwrap();
        let requests = vec![request(1, 11, Some(0.0)), request(2, 22, Some(1.0))];

        let report = simulate_trace_live_requests(args, requests, 1).unwrap();

        assert_eq!(report.request_counts.num_requests, 2);
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_output_tokens, 4);
        assert!(report.throughput.wall_time_ms >= 0.0);
    }

    #[test]
    fn test_online_trace_replay_uses_round_robin_dispatch() {
        let args = MockEngineArgs::builder()
            .speedup_ratio(1000.0)
            .block_size(64)
            .build()
            .unwrap();
        let requests = vec![
            request(1, 1, Some(0.0)),
            request(2, 2, Some(0.0)),
            request(3, 3, Some(0.0)),
            request(4, 4, Some(0.0)),
            request(5, 5, Some(0.0)),
        ];

        let (_, stats) = simulate_trace_live_requests_with_stats(args, requests, 3).unwrap();

        assert_eq!(stats.dispatch_history, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn test_online_concurrency_replay_respects_max_in_flight() {
        let args = MockEngineArgs::builder()
            .speedup_ratio(1000.0)
            .block_size(64)
            .build()
            .unwrap();
        let requests = vec![
            request(1, 10, None),
            request(2, 20, None),
            request(3, 30, None),
            request(4, 40, None),
        ];

        let (report, stats) =
            simulate_concurrency_live_requests_with_stats(args, requests, 2, 2).unwrap();

        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(stats.max_in_flight_seen, 2);
    }

    #[test]
    fn test_online_trace_replay_populates_admit_reuse_stats() {
        let args = MockEngineArgs::builder()
            .speedup_ratio(1000.0)
            .block_size(64)
            .build()
            .unwrap();
        let requests = vec![request(1, 77, Some(0.0)), request(2, 77, Some(5.0))];

        let report = simulate_trace_live_requests(args, requests, 1).unwrap();

        assert_eq!(report.request_counts.completed_requests, 2);
        assert!(report.prefix_cache_reused_ratio > 0.0);
    }
}

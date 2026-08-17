// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public controller boundary for offline replay.
//!
//! A controller owns request release and its own logical timers. Dynamo owns
//! placement, continuous batching, engine timing, and request lifecycle events.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::time::Duration;

use anyhow::{Result, ensure};
use dynamo_kv_router::config::KvRouterConfig;
use uuid::Uuid;

use super::agg::AggRuntimeImpl;
use super::components::{KvReplayMetadata, ReplayAdmissionMetadata};
use super::core::round_robin::AggregatedRoundRobinPlacement;
use super::core::{AdmissionSource, NoEngineEvents, ReadyArrival};
use super::extensions::kv_events::RouterEventObservation;
use super::extensions::kv_router::KvRouterPlacement;
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use crate::replay::{ReplayTerminalStatus, TraceSimulationReport};

/// Runtime state visible to a work source when Dynamo asks for ready requests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplayWorkSourceContext {
    pub now_ms: f64,
    pub cluster_in_flight: usize,
}

/// One inference request released by an external replay controller.
#[derive(Debug)]
pub struct ReplayWorkSubmission {
    /// Source-owned identity used to correlate output and terminal callbacks.
    pub request_id: Uuid,
    pub arrival_time_ms: f64,
    pub request: ReplayRequestPayload,
    pub replay_hashes: Option<ReplayRequestHashes>,
    pub session_id: Option<String>,
    pub turn_index: Option<usize>,
}

/// Event-driven request source for Dynamo offline replay.
///
/// `next_internal_event_ms` reports only timers already known by the source.
/// It must return `None` while progress depends solely on an in-flight request;
/// Dynamo will call `on_terminal` when that request finishes.
pub trait ReplayWorkSource {
    fn next_internal_event_ms(&mut self) -> Option<f64>;

    fn drain_ready(
        &mut self,
        context: ReplayWorkSourceContext,
    ) -> Result<Vec<ReplayWorkSubmission>>;

    fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()>;

    fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()>;

    fn is_drained(&self) -> bool;

    fn total_requests(&self) -> usize;
}

/// Optional detail capture for an externally controlled replay.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ControlledReplayOptions {
    /// Retain one [`crate::replay::PerRequestRecord`] for every terminal request.
    pub capture_per_request: bool,
    /// Keep later requests from the same session on the initially selected
    /// worker/rank until this logical-time idle TTL expires.
    pub session_affinity_ttl: Option<Duration>,
    /// Predict route-time KV placement for this logical-time TTL while the
    /// engine's authoritative KV event is still in flight.
    pub router_predicted_ttl: Option<Duration>,
    /// Block-equivalent decode cost charged for every active request on a
    /// candidate worker.
    pub decode_active_request_weight: Option<f64>,
}

fn controlled_router_config(
    predicted_ttl: Option<Duration>,
    decode_active_request_weight: Option<f64>,
) -> Result<Option<KvRouterConfig>> {
    if predicted_ttl.is_none() && decode_active_request_weight.is_none() {
        return Ok(None);
    }
    let mut config = KvRouterConfig::default();
    if let Some(ttl) = predicted_ttl {
        config.use_kv_events = true;
        config.router_predicted_ttl_secs = Some(ttl.as_secs_f64());
    }
    if let Some(weight) = decode_active_request_weight {
        ensure!(
            weight.is_finite() && weight >= 0.0,
            "decode active-request weight must be finite and nonnegative"
        );
        config.decode_active_request_weight = weight;
    }
    Ok(Some(config))
}

struct ControlledAdmission<'source, Source, Metadata> {
    source: &'source mut Source,
    metadata: PhantomData<Metadata>,
}

impl<'source, Source, Metadata> ControlledAdmission<'source, Source, Metadata> {
    fn new(source: &'source mut Source) -> Self {
        Self {
            source,
            metadata: PhantomData,
        }
    }
}

impl<Source, Metadata> AdmissionSource for ControlledAdmission<'_, Source, Metadata>
where
    Source: ReplayWorkSource,
    Metadata: ReplayAdmissionMetadata,
{
    type Request = ReplayRequestPayload;
    type Metadata = Metadata;

    fn next_internal_event_ms(&mut self) -> Option<f64> {
        self.source.next_internal_event_ms()
    }

    fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival<Self::Request, Self::Metadata>>> {
        self.source
            .drain_ready(ReplayWorkSourceContext {
                now_ms,
                cluster_in_flight,
            })?
            .into_iter()
            .map(|mut submission| {
                ensure!(
                    submission.arrival_time_ms.is_finite()
                        && submission.arrival_time_ms >= 0.0
                        && submission.arrival_time_ms <= now_ms,
                    "controlled replay source emitted request {} with invalid arrival time {} at now={now_ms}",
                    submission.request_id,
                    submission.arrival_time_ms,
                );
                if let Some(payload_id) = submission.request.metadata().uuid {
                    ensure!(
                        payload_id == submission.request_id,
                        "controlled replay submission ID {} does not match payload ID {payload_id}",
                        submission.request_id,
                    );
                }
                submission.request.metadata_mut().uuid = Some(submission.request_id);
                Ok(ReadyArrival {
                    request: submission.request,
                    arrival_time_ms: submission.arrival_time_ms,
                    metadata: Metadata::from_hashes(submission.replay_hashes),
                    session_id: submission.session_id,
                    turn_index: submission.turn_index,
                })
            })
            .collect()
    }

    fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()> {
        self.source.on_output_token(request_id, token_id)
    }

    fn on_terminal(&mut self, request_id: Uuid, now_ms: f64, rejected: bool) -> Result<()> {
        let status = if rejected {
            ReplayTerminalStatus::Rejected
        } else {
            ReplayTerminalStatus::Completed
        };
        self.source.on_terminal(request_id, now_ms, status)
    }

    fn is_drained(&self) -> bool {
        self.source.is_drained()
    }

    fn total_requests(&self) -> usize {
        self.source.total_requests()
    }
}

/// Run an externally controlled workload through Dynamo's aggregated,
/// round-robin offline replay engine.
pub fn simulate_controlled_aggregated<Source>(
    args: &MockEngineArgs,
    num_workers: usize,
    source: &mut Source,
) -> Result<TraceSimulationReport>
where
    Source: ReplayWorkSource,
{
    simulate_controlled_aggregated_with_options(
        args,
        num_workers,
        source,
        ControlledReplayOptions::default(),
    )
}

/// Run an externally controlled workload and optionally retain request detail.
pub fn simulate_controlled_aggregated_with_options<Source>(
    args: &MockEngineArgs,
    num_workers: usize,
    source: &mut Source,
    options: ControlledReplayOptions,
) -> Result<TraceSimulationReport>
where
    Source: ReplayWorkSource,
{
    ensure!(
        options.session_affinity_ttl.is_none()
            && options.router_predicted_ttl.is_none()
            && options.decode_active_request_weight.is_none(),
        "controlled round-robin replay does not support KV-router TTL options"
    );
    ensure!(
        num_workers > 0,
        "controlled replay requires at least one worker"
    );
    let admission = ControlledAdmission::<Source, ()>::new(source);
    let runtime = AggRuntimeImpl::<
        AggregatedRoundRobinPlacement<()>,
        NoEngineEvents,
        (),
        ControlledAdmission<'_, Source, ()>,
    >::new_composed(args, admission, num_workers, |args, topology| {
        Ok(AggregatedRoundRobinPlacement::new(args.dp_size, topology))
    })?
    .with_per_request_records(options.capture_per_request);
    let (collector, _) = runtime.run()?;
    Ok(collector.finish())
}

/// Run an externally controlled workload through Dynamo's aggregated KV
/// router. Request constraints and `MockEngineArgs::worker_taints` determine
/// the eligible workers.
pub fn simulate_controlled_aggregated_kv_router_with_options<Source>(
    args: &MockEngineArgs,
    num_workers: usize,
    source: &mut Source,
    options: ControlledReplayOptions,
) -> Result<TraceSimulationReport>
where
    Source: ReplayWorkSource,
{
    ensure!(
        num_workers > 0,
        "controlled replay requires at least one worker"
    );
    let admission = ControlledAdmission::<Source, KvReplayMetadata>::new(source);
    let runtime = AggRuntimeImpl::<
        KvRouterPlacement,
        RouterEventObservation,
        KvReplayMetadata,
        ControlledAdmission<'_, Source, KvReplayMetadata>,
    >::new_composed(args, admission, num_workers, |args, _topology| {
        KvRouterPlacement::new_with_session_affinity(
            args,
            controlled_router_config(
                options.router_predicted_ttl,
                options.decode_active_request_weight,
            )?,
            None,
            num_workers,
            options.session_affinity_ttl,
        )
    })?
    .with_per_request_records(options.capture_per_request);
    let (collector, _) = runtime.run()?;
    Ok(collector.finish())
}

/// Run an externally controlled workload across heterogeneous aggregated
/// workers. Each worker owns its own normalized engine topology and
/// performance model; required taints select between those worker pools.
pub fn simulate_controlled_heterogeneous_aggregated_kv_router_with_options<Source>(
    worker_args: &[MockEngineArgs],
    worker_taints: &[HashSet<String>],
    source: &mut Source,
    options: ControlledReplayOptions,
) -> Result<TraceSimulationReport>
where
    Source: ReplayWorkSource,
{
    ensure!(
        !worker_args.is_empty(),
        "controlled heterogeneous replay requires at least one worker"
    );
    let admission = ControlledAdmission::<Source, KvReplayMetadata>::new(source);
    let runtime = AggRuntimeImpl::<
        KvRouterPlacement,
        RouterEventObservation,
        KvReplayMetadata,
        ControlledAdmission<'_, Source, KvReplayMetadata>,
    >::new_composed_heterogeneous(
        worker_args,
        worker_taints,
        admission,
        |worker_args, worker_taints, _topology| {
            KvRouterPlacement::new_heterogeneous_with_session_affinity(
                worker_args,
                worker_taints,
                controlled_router_config(
                    options.router_predicted_ttl,
                    options.decode_active_request_weight,
                )?,
                None,
                options.session_affinity_ttl,
            )
        },
    )?
    .with_per_request_records(options.capture_per_request);
    let (collector, _) = runtime.run()?;
    Ok(collector.finish())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::loadgen::{SessionTrace, Trace, TurnTrace, WorkloadDriver};

    struct DriverSource {
        driver: WorkloadDriver,
        turns: HashMap<Uuid, (String, usize)>,
        submissions: Vec<(String, usize, f64)>,
        terminals: Vec<(String, usize, f64)>,
    }

    impl DriverSource {
        fn new(trace: Trace) -> Self {
            let engine_block_size = trace.block_size;
            Self {
                driver: WorkloadDriver::new_concurrency(trace, engine_block_size, usize::MAX)
                    .unwrap()
                    .with_deterministic_request_ids(7),
                turns: HashMap::new(),
                submissions: Vec::new(),
                terminals: Vec::new(),
            }
        }
    }

    impl ReplayWorkSource for DriverSource {
        fn next_internal_event_ms(&mut self) -> Option<f64> {
            self.driver.next_ready_time_ms()
        }

        fn drain_ready(
            &mut self,
            context: ReplayWorkSourceContext,
        ) -> Result<Vec<ReplayWorkSubmission>> {
            Ok(self
                .driver
                .pop_ready_replay(context.now_ms, usize::MAX)
                .into_iter()
                .map(|ready| {
                    self.turns.insert(
                        ready.request_uuid,
                        (ready.session_id.clone(), ready.turn_index),
                    );
                    self.submissions.push((
                        ready.session_id.clone(),
                        ready.turn_index,
                        context.now_ms,
                    ));
                    ReplayWorkSubmission {
                        request_id: ready.request_uuid,
                        arrival_time_ms: context.now_ms,
                        request: ready.request,
                        replay_hashes: ready.replay_hashes,
                        session_id: Some(ready.session_id),
                        turn_index: Some(ready.turn_index),
                    }
                })
                .collect())
        }

        fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()> {
            self.driver.on_output_token(request_id, token_id)
        }

        fn on_terminal(
            &mut self,
            request_id: Uuid,
            now_ms: f64,
            status: ReplayTerminalStatus,
        ) -> Result<()> {
            let (session_id, turn_index) = self.turns.remove(&request_id).unwrap();
            self.terminals.push((session_id, turn_index, now_ms));
            self.driver
                .on_terminal(request_id, now_ms, status == ReplayTerminalStatus::Rejected)
        }

        fn is_drained(&self) -> bool {
            self.driver.is_drained()
        }

        fn total_requests(&self) -> usize {
            self.driver.total_turns()
        }
    }

    fn args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8192))
            .max_num_seqs(Some(8))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(1000.0)
            .build()
            .unwrap()
    }

    fn turn(hash_id: u32, required_taint: Option<&str>) -> TurnTrace {
        let mut turn = TurnTrace {
            input_length: 64,
            max_output_tokens: 8,
            hash_ids: vec![hash_id],
            ..Default::default()
        };
        if let Some(taint) = required_taint {
            turn.routing_constraints
                .required_taints
                .insert(taint.to_string());
        }
        turn
    }

    #[test]
    fn controlled_router_config_applies_decode_active_request_weight() {
        let config = controlled_router_config(None, Some(128.0))
            .unwrap()
            .unwrap();
        assert_eq!(config.decode_active_request_weight, 128.0);
        assert!(controlled_router_config(None, Some(f64::NAN)).is_err());
    }

    #[test]
    fn public_controlled_replay_arms_delay_after_terminal() {
        let trace = Trace {
            block_size: 64,
            sessions: vec![SessionTrace {
                session_id: "two-turn".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    turn(1, None),
                    TurnTrace {
                        delay_after_previous_ms: 10.0,
                        ..turn(2, None)
                    },
                ],
            }],
        };
        let mut source = DriverSource::new(trace);

        let report = simulate_controlled_aggregated(&args(), 1, &mut source).unwrap();

        assert_eq!(report.request_counts.completed_requests, 2);
        let first_terminal = source
            .terminals
            .iter()
            .find(|(_, turn, _)| *turn == 0)
            .unwrap()
            .2;
        let second_submission = source
            .submissions
            .iter()
            .find(|(_, turn, _)| *turn == 1)
            .unwrap()
            .2;
        assert_eq!(second_submission, first_terminal + 10.0);
    }

    #[test]
    fn controlled_kv_router_honors_required_worker_taints() {
        let trace = Trace {
            block_size: 64,
            sessions: vec![
                SessionTrace {
                    session_id: "high".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![turn(1, Some("tier=high"))],
                },
                SessionTrace {
                    session_id: "low".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![turn(2, Some("tier=low"))],
                },
            ],
        };
        let mut source = DriverSource::new(trace);
        let mut args = args();
        args.worker_taints = vec![
            HashSet::from(["tier=high".to_string()]),
            HashSet::from(["tier=low".to_string()]),
        ];

        let report = simulate_controlled_aggregated_kv_router_with_options(
            &args,
            2,
            &mut source,
            ControlledReplayOptions {
                capture_per_request: true,
                ..ControlledReplayOptions::default()
            },
        )
        .unwrap();

        let workers = report
            .per_request
            .iter()
            .map(|request| {
                (
                    request.session_id.as_deref().unwrap(),
                    request.decode_worker_idx.unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(workers["high"], 0);
        assert_eq!(workers["low"], 1);
    }

    #[cfg(feature = "replay-bench")]
    #[test]
    fn controlled_kv_router_replay_is_reproducible() {
        fn run() -> (f64, Vec<(String, usize, f64)>) {
            let trace = Trace {
                block_size: 64,
                sessions: (0..32)
                    .map(|index| SessionTrace {
                        session_id: format!("session-{index}"),
                        first_arrival_timestamp_ms: Some(0.0),
                        turns: vec![turn(index + 1, None)],
                    })
                    .collect(),
            };
            let mut source = DriverSource::new(trace);
            let report = simulate_controlled_aggregated_kv_router_with_options(
                &args(),
                4,
                &mut source,
                ControlledReplayOptions {
                    capture_per_request: true,
                    ..ControlledReplayOptions::default()
                },
            )
            .unwrap();
            let mut requests = report
                .per_request
                .into_iter()
                .map(|request| {
                    (
                        request.session_id.unwrap(),
                        request.decode_worker_idx.unwrap(),
                        request.terminal_time_ms,
                    )
                })
                .collect::<Vec<_>>();
            requests.sort_by(|left, right| left.0.cmp(&right.0));
            (report.throughput.duration_ms, requests)
        }

        assert_eq!(run(), run());
    }

    #[test]
    fn controlled_kv_router_supports_heterogeneous_worker_topologies() {
        let trace = Trace {
            block_size: 64,
            sessions: vec![
                SessionTrace {
                    session_id: "throughput".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![turn(1, Some("tier=throughput"))],
                },
                SessionTrace {
                    session_id: "tail".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![turn(2, Some("tier=tail"))],
                },
            ],
        };
        let mut source = DriverSource::new(trace);
        let mut throughput_args = args();
        throughput_args.dp_size = 2;
        let mut tail_args = args();
        tail_args.aic_tp_size = Some(2);
        tail_args.max_num_seqs = Some(1);
        let worker_args = vec![throughput_args, tail_args];
        let worker_taints = vec![
            HashSet::from(["tier=throughput".to_string()]),
            HashSet::from(["tier=tail".to_string()]),
        ];

        let report = simulate_controlled_heterogeneous_aggregated_kv_router_with_options(
            &worker_args,
            &worker_taints,
            &mut source,
            ControlledReplayOptions {
                capture_per_request: true,
                ..ControlledReplayOptions::default()
            },
        )
        .unwrap();

        let workers = report
            .per_request
            .iter()
            .map(|request| {
                (
                    request.session_id.as_deref().unwrap(),
                    request.decode_worker_idx.unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        assert!(workers["throughput"] < 2);
        assert_eq!(workers["tail"], 2);
        assert_eq!(report.throughput.decode_gpus_per_worker, 2);
    }

    #[test]
    fn controlled_replay_applies_per_worker_sequence_limits() {
        let trace = Trace {
            block_size: 64,
            sessions: (0..4)
                .map(|index| SessionTrace {
                    session_id: format!("session-{index}"),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![turn(index + 1, None)],
                })
                .collect(),
        };
        let mut source = DriverSource::new(trace);
        let mut args = args();
        args.worker_max_num_seqs = vec![1, 2];

        let report = simulate_controlled_aggregated_with_options(
            &args,
            2,
            &mut source,
            ControlledReplayOptions {
                capture_per_request: true,
                ..ControlledReplayOptions::default()
            },
        )
        .unwrap();

        let mut by_worker: HashMap<usize, Vec<_>> = HashMap::new();
        for request in &report.per_request {
            by_worker
                .entry(request.decode_worker_idx.unwrap())
                .or_default()
                .push(request);
        }
        for requests in by_worker.values_mut() {
            requests.sort_by(|left, right| {
                left.first_admit_ms
                    .unwrap()
                    .total_cmp(&right.first_admit_ms.unwrap())
            });
        }
        assert_eq!(by_worker[&0].len(), 2);
        assert_eq!(by_worker[&1].len(), 2);
        assert!(by_worker[&0][1].first_admit_ms.unwrap() >= by_worker[&0][0].terminal_time_ms);
        assert_eq!(
            by_worker[&1][0].first_admit_ms,
            by_worker[&1][1].first_admit_ms
        );
    }
}

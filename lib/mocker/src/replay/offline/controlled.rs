// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public controller boundary for offline replay.
//!
//! A controller owns request release and its own logical timers. Dynamo owns
//! placement, continuous batching, engine timing, and request lifecycle events.

use std::marker::PhantomData;

use anyhow::{Result, ensure};
use uuid::Uuid;

use super::agg::AggRuntimeImpl;
use super::components::{KvReplayMetadata, ReplayAdmissionMetadata};
use super::core::round_robin::AggregatedRoundRobinPlacement;
use super::core::{AdmissionSource, NoEngineEvents, ReadyArrival};
use super::extensions::kv_events::RouterEventObservation;
use super::extensions::kv_router::KvRouterPlacement;
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::{
    CascadedWorkloadTerminal, ReplayRequestHashes, ReplayRequestPayload, WorkloadTerminalStatus,
};
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ControlledReplayOptions {
    /// Retain one [`crate::replay::PerRequestRecord`] for every terminal request.
    pub capture_per_request: bool,
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
    type TerminalStatus = WorkloadTerminalStatus;
    type CascadedTerminal = CascadedWorkloadTerminal;

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
                    logical_request_id: None,
                    authored_turn_index: None,
                })
            })
            .collect()
    }

    fn on_output_token(&mut self, request_id: Uuid, token_id: u32) -> Result<()> {
        self.source.on_output_token(request_id, token_id)
    }

    fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        status: WorkloadTerminalStatus,
    ) -> Result<Vec<CascadedWorkloadTerminal>> {
        self.source.on_terminal(request_id, now_ms, status.into())?;
        Ok(Vec::new())
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
        Ok(AggregatedRoundRobinPlacement::with_taints(
            args.dp_size,
            topology,
            &args.worker_taints,
        ))
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
        KvRouterPlacement::new(args, None, None, num_workers)
    })?
    .with_per_request_records(options.capture_per_request);
    let (collector, _) = runtime.run()?;
    Ok(collector.finish())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::loadgen::{
        AgenticTrace, AgenticTurnTrace, SessionTrace, Trace, TurnTrace, WorkloadDriver,
    };
    use crate::replay::offline::{
        OfflineReplaySession, ReplayAgenticRequest, ReplayAgenticWorkflow, ReplayRequestSpec,
        ReplayRoutingConstraints, ReplaySessionRouter, ReplayStepStatus,
        simulate_agentic_trace_workload,
    };
    use crate::replay::{ReplayRouterMode, SlaThresholds};

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

        fn new_agentic(trace: AgenticTrace, engine_block_size: usize) -> Self {
            Self {
                driver: WorkloadDriver::new_agentic_trace_without_replay_hashes(
                    trace,
                    engine_block_size,
                )
                .unwrap(),
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
            let cascaded = self.driver.on_terminal(request_id, now_ms, status.into())?;
            ensure!(
                cascaded.is_empty(),
                "non-agentic controlled test source unexpectedly cascaded terminals"
            );
            Ok(())
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

    fn adapter_conformance_trace() -> AgenticTrace {
        AgenticTrace {
            block_size: 64,
            turns: vec![
                AgenticTurnTrace {
                    request_id: "adapter-a".to_string(),
                    session_id: "adapter-session-a".to_string(),
                    authored_turn_index: 0,
                    internal_uuid: Some(Uuid::from_u128(0xa1)),
                    input_length: 128,
                    max_output_tokens: 4,
                    output_token_ids: Some(vec![101, 102, 103, 104]),
                    hash_ids: vec![11, 12],
                    first_ready_timestamp_ms: Some(0.0),
                    prefix_reset: false,
                    ..Default::default()
                },
                AgenticTurnTrace {
                    request_id: "adapter-b".to_string(),
                    session_id: "adapter-session-b".to_string(),
                    authored_turn_index: 0,
                    internal_uuid: Some(Uuid::from_u128(0xb2)),
                    input_length: 64,
                    max_output_tokens: 3,
                    output_token_ids: Some(vec![201, 202, 203]),
                    hash_ids: vec![21],
                    first_ready_timestamp_ms: Some(0.0),
                    prefix_reset: false,
                    ..Default::default()
                },
            ],
        }
    }

    fn run_interactive_conformance(trace: &AgenticTrace) -> TraceSimulationReport {
        let mut replay = OfflineReplaySession::new(
            &args(),
            2,
            trace.block_size,
            ReplaySessionRouter::RoundRobin,
        )
        .unwrap();
        replay
            .append_agentic_workflow(
                ReplayAgenticWorkflow {
                    trace_block_size: trace.block_size,
                    requests: trace
                        .turns
                        .iter()
                        .map(|turn| ReplayAgenticRequest {
                            request: ReplayRequestSpec {
                                logical_request_id: turn.request_id.clone(),
                                attempt_id: "0".to_string(),
                                group_id: turn.session_id.clone(),
                                internal_uuid: turn.internal_uuid,
                                session_id: turn.session_id.clone(),
                                authored_turn_index: turn.authored_turn_index,
                                ready_time_ms: turn.first_ready_timestamp_ms.unwrap_or_default(),
                                input_length: turn.input_length,
                                hash_ids: turn.hash_ids.clone(),
                                trace_block_size: trace.block_size,
                                output_length: turn.max_output_tokens,
                                output_token_ids: turn.output_token_ids.clone(),
                                priority: turn.priority,
                                strict_priority: turn.strict_priority,
                                policy_class: turn.policy_class.clone(),
                                routing_constraints: ReplayRoutingConstraints::default(),
                                target: None,
                            },
                            wait_for: turn.wait_for.clone(),
                            dependency_delay_ms: turn.delay_after_dependencies_ms,
                            prefix_reset: turn.prefix_reset,
                        })
                        .collect(),
                },
                0.0,
            )
            .unwrap();
        replay.close_admission().unwrap();
        for _ in 0..10_000 {
            replay.settle_current_time().unwrap();
            replay.drain_events().unwrap();
            if replay.is_drained().unwrap() {
                return replay.finalize().unwrap();
            }
            assert!(!matches!(
                replay.advance_next().unwrap(),
                ReplayStepStatus::Quiescent { .. }
            ));
            replay.drain_events().unwrap();
        }
        panic!("interactive conformance fixture did not drain")
    }

    #[derive(Debug, PartialEq)]
    struct AdapterPhysicsRecord {
        uuid: String,
        arrival_time_ms: f64,
        first_admit_ms: Option<f64>,
        terminal_time_ms: f64,
        first_token_ms: Option<f64>,
        last_token_ms: Option<f64>,
        input_length: usize,
        requested_output_length: usize,
        output_length: usize,
        reused_input_tokens: usize,
        decode_worker_idx: Option<usize>,
        terminal_status: ReplayTerminalStatus,
    }

    fn physics_records(report: &TraceSimulationReport) -> Vec<AdapterPhysicsRecord> {
        let mut records = report
            .per_request
            .iter()
            .map(|record| AdapterPhysicsRecord {
                uuid: record.uuid.clone(),
                arrival_time_ms: record.arrival_time_ms,
                first_admit_ms: record.first_admit_ms,
                terminal_time_ms: record.terminal_time_ms,
                first_token_ms: record.first_token_ms,
                last_token_ms: record.last_token_ms,
                input_length: record.input_length,
                requested_output_length: record.requested_output_length,
                output_length: record.output_length,
                reused_input_tokens: record.reused_input_tokens,
                decode_worker_idx: record.decode_worker_idx,
                terminal_status: record.terminal_status,
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| left.uuid.cmp(&right.uuid));
        records
    }

    #[test]
    fn one_shot_work_source_and_interactive_share_fixed_placement_physics() {
        let trace = adapter_conformance_trace();
        let one_shot = simulate_agentic_trace_workload(
            args(),
            None,
            None,
            trace.clone(),
            2,
            ReplayRouterMode::RoundRobin,
            true,
            SlaThresholds::default(),
        )
        .unwrap();
        let mut source = DriverSource::new_agentic(trace.clone(), 64);
        let controlled = simulate_controlled_aggregated_with_options(
            &args(),
            2,
            &mut source,
            ControlledReplayOptions {
                capture_per_request: true,
            },
        )
        .unwrap();
        let interactive = run_interactive_conformance(&trace);

        assert_eq!(physics_records(&one_shot), physics_records(&controlled));
        assert_eq!(physics_records(&one_shot), physics_records(&interactive));
        assert_eq!(one_shot.request_counts.num_requests, 2);
        assert_eq!(controlled.request_counts.completed_requests, 2);
        assert_eq!(interactive.request_counts.completed_requests, 2);
        assert_eq!(
            one_shot.throughput.decode_worker_seconds,
            controlled.throughput.decode_worker_seconds
        );
        assert_eq!(
            one_shot.throughput.decode_worker_seconds,
            interactive.throughput.decode_worker_seconds
        );
        assert_eq!(
            physics_records(&one_shot)
                .iter()
                .map(|record| record.decode_worker_idx)
                .collect::<Vec<_>>(),
            [Some(0), Some(1)]
        );
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

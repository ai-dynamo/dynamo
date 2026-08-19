// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public replay facade over the mechanically moved topology runtimes.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::Result as AnyResult;
use uuid::Uuid;

use crate::replay::OfflineDisaggReplayConfig;
use crate::replay::agg::AggRuntimeImpl;
use crate::replay::artifact::{
    ReplayArtifactKvEventVisibility, ReplayArtifactSink, ReplayArtifacts,
};
use crate::replay::components::{
    AdmissionQueue, NoReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation, ReplayMode,
};
use crate::replay::core::round_robin::{AggregatedRoundRobinPlacement, PoolRoundRobinPlacement};
use crate::replay::core::{NoEngineEvents, PlacementPolicy, WorkerTopology};
use crate::replay::disagg::DisaggRuntimeImpl;
use crate::replay::engine::{ReplayEngineConfig, ReplayEngineFactory};
use crate::replay::error::{placement_boundary, runtime_error, scaling_boundary};
use crate::replay::loadgen::ReplayRequestPayload;
use crate::replay::loadgen::WorkloadDriver;
use crate::replay::protocol::{DirectRequest, ReplayPromptTokenSource, ReplayRequestContext};
use crate::replay::scaling::ReplayScalingPolicy;
use crate::replay::{
    ReplayCaptureOptions, ReplayDeterminism, ReplayError, ReplayReport, ReplayResult, ReplaySpec,
    ReplayTopology, SlaThresholds, WorkerStage,
};

/// Runtime composition supplied by the built-in engine stack or a Dynamo
/// adapter. The adapter owns concrete Router/Planner construction; Replay only
/// sees the already-neutral placement and scaling contracts.
pub trait ReplayComposition {
    type Metadata: ReplayAdmissionMetadata;
    type Observation: ReplayEngineObservation;
    type AggregatedPlacement: PlacementPolicy<
            ReplayRequestPayload,
            Metadata = Self::Metadata,
            Observation = <Self::Observation as ReplayEngineObservation>::Batch,
        >;
    type DisaggregatedPlacement: PlacementPolicy<
            ReplayRequestPayload,
            Metadata = Self::Metadata,
            Observation = <Self::Observation as ReplayEngineObservation>::Batch,
        >;

    fn validate_spec(&self, _spec: &ReplaySpec) -> ReplayResult<()> {
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> AnyResult<Self::AggregatedPlacement>;

    fn create_disaggregated_placements(
        &mut self,
        prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> AnyResult<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)>;

    /// Return the run-owned scaling policy, if this composition has one.
    fn take_scaling_policy(&mut self) -> AnyResult<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(None)
    }

    /// Inform policy construction about explicitly requested deterministic
    /// selection. Implementations should use
    /// [`ReplayDeterminism::selector_seed`] and leave normal runs unseeded.
    fn set_determinism(&mut self, _determinism: ReplayDeterminism) -> ReplayResult<()> {
        Ok(())
    }
}

/// Classifies every fallible callback from a placement policy at the boundary
/// where the policy enters the otherwise policy-neutral runtime.
struct PlacementPolicyBoundary<P>(P);

impl<Request, P> PlacementPolicy<Request> for PlacementPolicyBoundary<P>
where
    P: PlacementPolicy<Request>,
{
    type Metadata = P::Metadata;
    type Observation = P::Observation;

    fn place(
        &mut self,
        request: &Request,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> AnyResult<crate::replay::core::PlacementEffects> {
        self.0
            .place(request, metadata, session_id, now_ms)
            .map_err(placement_boundary)
    }

    fn observe(
        &mut self,
        observation: Self::Observation,
        now_ms: f64,
    ) -> AnyResult<Vec<crate::replay::core::Placement>> {
        self.0
            .observe(observation, now_ms)
            .map_err(placement_boundary)
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.0.cancel_pending(request_id)
    }

    fn request_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> AnyResult<Vec<crate::replay::core::Placement>> {
        self.0
            .request_terminal(request_id, now_ms)
            .map_err(placement_boundary)
    }

    fn prefill_completed(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> AnyResult<Vec<crate::replay::core::Placement>> {
        self.0
            .prefill_completed(request_id, now_ms)
            .map_err(placement_boundary)
    }

    fn pending_count(&self) -> usize {
        self.0.pending_count()
    }

    fn worker_ready(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> AnyResult<Vec<crate::replay::core::Placement>> {
        self.0
            .worker_ready(worker, now_ms)
            .map_err(placement_boundary)
    }

    fn worker_draining(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> AnyResult<Vec<crate::replay::core::Placement>> {
        self.0
            .worker_draining(worker, now_ms)
            .map_err(placement_boundary)
    }

    fn worker_removed(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> AnyResult<Vec<crate::replay::core::Placement>> {
        self.0
            .worker_removed(worker, now_ms)
            .map_err(placement_boundary)
    }

    fn topology_settled(&mut self, now_ms: f64) -> AnyResult<Vec<crate::replay::core::Placement>> {
        self.0.topology_settled(now_ms).map_err(placement_boundary)
    }
}

/// Classifies Planner/scaling callbacks without exposing policy-specific types
/// to the aggregated or disaggregated runtime.
struct ScalingPolicyBoundary(Box<dyn ReplayScalingPolicy>);

impl ReplayScalingPolicy for ScalingPolicyBoundary {
    fn capture_lifecycle_evidence(&self) -> bool {
        self.0.capture_lifecycle_evidence()
    }

    fn initial_tick_ms(&mut self) -> AnyResult<f64> {
        self.0.initial_tick_ms().map_err(scaling_boundary)
    }

    fn on_tick(
        &mut self,
        snapshot: crate::replay::scaling::ReplayScalingSnapshot,
    ) -> AnyResult<crate::replay::scaling::ReplayScalingDecision> {
        self.0.on_tick(snapshot).map_err(scaling_boundary)
    }
}

/// Replay-owned runtime input used by compatibility runners that already
/// lowered a trace into the shared workload driver.
///
/// Serializable callers should keep using [`ReplaySpec::requests`]. Dynamo's
/// legacy entrypoints use this seam to preserve multi-turn, concurrency, and
/// agentic scheduling without recompiling Replay sources in the Dynamo crate.
#[doc(hidden)]
pub enum ReplayRuntimeInput {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

/// Built-in engine-only composition: Round-robin placement and fixed capacity.
#[derive(Debug, Default, Clone, Copy)]
pub struct RoundRobinComposition;

impl ReplayComposition for RoundRobinComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn validate_spec(&self, spec: &ReplaySpec) -> ReplayResult<()> {
        if spec.adapters.placement.provider != "round_robin" {
            return Err(ReplayError::InvalidSpec(format!(
                "engine composition requires round_robin placement, got {:?}",
                spec.adapters.placement.provider
            )));
        }
        if spec.adapters.scaling.provider != "none" {
            return Err(ReplayError::InvalidSpec(format!(
                "engine composition does not provide scaling, got {:?}",
                spec.adapters.scaling.provider
            )));
        }
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> AnyResult<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> AnyResult<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }
}

/// Owns one replay execution: canonical spec, engine construction, and
/// the selected placement/scaling composition.
pub struct Replayer<C = RoundRobinComposition> {
    spec: ReplaySpec,
    factory: ReplayEngineFactory,
    composition: C,
    runtime_input: Option<ReplayRuntimeInput>,
    capture: ReplayCaptureOptions,
}

impl Replayer<RoundRobinComposition> {
    pub fn new(spec: ReplaySpec, factory: ReplayEngineFactory) -> ReplayResult<Self> {
        Self::with_composition(spec, factory, RoundRobinComposition)
    }

    /// Run a fixed, aggregated single-worker replay and retain detailed
    /// request/output/native-KV observations from the same Replayer-owned
    /// aggregated runtime that produces the normal report.
    ///
    /// This contract intentionally targets one worker artifact. Multi-worker,
    /// scaling, and disaggregated runs should consume the normal report and
    /// placement/scaling observation contracts instead of creating a second
    /// scheduler loop solely for artifact generation.
    pub fn run_with_artifacts(
        self,
        visibility: ReplayArtifactKvEventVisibility,
    ) -> ReplayResult<(ReplayReport, ReplayArtifacts)> {
        let sink = ReplayArtifactSink::new(visibility);
        let report = self.run_inner(Some(sink.clone()))?;
        Ok((report, sink.take()?))
    }
}

impl<C: ReplayComposition> Replayer<C> {
    pub fn with_composition(
        spec: ReplaySpec,
        factory: ReplayEngineFactory,
        composition: C,
    ) -> ReplayResult<Self> {
        spec.validate()?;
        composition.validate_spec(&spec)?;
        Ok(Self {
            spec,
            factory,
            composition,
            runtime_input: None,
            capture: ReplayCaptureOptions::default(),
        })
    }

    /// Override the serializable request list with an already-lowered,
    /// Replay-owned runtime input.
    #[doc(hidden)]
    pub fn with_runtime_input(mut self, input: ReplayRuntimeInput) -> Self {
        self.runtime_input = Some(input);
        self
    }

    /// Configure detailed capture and canonical determinism for this run.
    pub fn with_capture_options(mut self, options: ReplayCaptureOptions) -> Self {
        self.capture = options;
        self
    }

    pub fn run(self) -> ReplayResult<ReplayReport> {
        self.run_inner(None)
    }

    fn run_inner(
        mut self,
        artifact_sink: Option<ReplayArtifactSink>,
    ) -> ReplayResult<ReplayReport> {
        let wall_start = Instant::now();
        self.composition.set_determinism(self.capture.determinism)?;
        let engine_config = ReplayEngineConfig::parse(&self.spec.engine)?;
        engine_config.validate_topology(&self.spec.topology)?;
        let runtime_input = match self.runtime_input.take() {
            Some(mut input) => {
                apply_runtime_determinism(&mut input, self.capture.determinism);
                input
            }
            None => {
                ReplayRuntimeInput::Requests(lower_requests(&self.spec, self.capture.determinism)?)
            }
        };
        let mode = self
            .spec
            .max_in_flight
            .map_or(ReplayMode::Trace, |max_in_flight| ReplayMode::Concurrency {
                max_in_flight,
            });
        let scaling = self
            .composition
            .take_scaling_policy()
            .map_err(|error| ReplayError::Scaling(format!("{error:#}")))?;

        let collector = match &self.spec.topology {
            ReplayTopology::Aggregated { workers } => {
                let role_factory = self.factory.role_factory(
                    &engine_config,
                    WorkerStage::Aggregated,
                    C::Observation::capture_engine_kv_events(WorkerStage::Aggregated)
                        || artifact_sink.is_some(),
                )?;
                let startup_time_ms = positive_delay(workers.startup_delay_ms);
                if artifact_sink.is_some()
                    && (workers.initial_workers != 1
                        || role_factory.dp_size() != 1
                        || scaling.is_some())
                {
                    return Err(ReplayError::InvalidSpec(
                        "detailed replay artifacts require fixed aggregated topology with one logical DP1 worker"
                            .to_string(),
                    ));
                }

                let mut runtime = AggRuntimeImpl::<
                    PlacementPolicyBoundary<C::AggregatedPlacement>,
                    C::Observation,
                    C::Metadata,
                >::new_composed(
                    role_factory,
                    admission_queue(runtime_input, mode),
                    workers.initial_workers,
                    startup_time_ms,
                    |dp_size, topology| {
                        self.composition
                            .create_aggregated_placement(dp_size, topology)
                            .map(PlacementPolicyBoundary)
                            .map_err(placement_boundary)
                    },
                )
                .map_err(runtime_error)?
                .with_capture_options(self.capture)
                .with_per_request_records(
                    self.spec.record_per_request || self.capture.effective_per_request(),
                )
                .with_max_sim_time_ms(self.spec.max_sim_time_ms);
                if let Some(sink) = artifact_sink {
                    runtime = runtime.with_artifact_sink(sink);
                }
                if let Some(policy) = scaling {
                    runtime = runtime.with_scaling_policy(Box::new(ScalingPolicyBoundary(policy)));
                }
                runtime.run().map_err(runtime_error)?.0
            }
            ReplayTopology::Disaggregated {
                prefill,
                decode,
                handoff_latency_ms,
            } => {
                if artifact_sink.is_some() {
                    return Err(ReplayError::InvalidSpec(
                        "detailed replay artifacts require aggregated topology".to_string(),
                    ));
                }
                let prefill_factory = self.factory.role_factory(
                    &engine_config,
                    WorkerStage::Prefill,
                    C::Observation::capture_engine_kv_events(WorkerStage::Prefill),
                )?;
                let decode_factory = self.factory.role_factory(
                    &engine_config,
                    WorkerStage::Decode,
                    C::Observation::capture_engine_kv_events(WorkerStage::Decode),
                )?;
                let config = OfflineDisaggReplayConfig {
                    prefill_factory,
                    decode_factory,
                    prefill_startup_time_ms: positive_delay(prefill.startup_delay_ms),
                    decode_startup_time_ms: positive_delay(decode.startup_delay_ms),
                    num_prefill_workers: prefill.initial_workers,
                    num_decode_workers: decode.initial_workers,
                    handoff_latency_ms: *handoff_latency_ms,
                };
                let mut runtime = DisaggRuntimeImpl::<
                    PlacementPolicyBoundary<C::DisaggregatedPlacement>,
                    C::Observation,
                    C::Metadata,
                >::new_composed(
                    &config,
                    admission_queue(runtime_input, mode),
                    false,
                    |prefill_dp, prefill_topology, decode_dp, decode_topology| {
                        self.composition
                            .create_disaggregated_placements(
                                prefill_dp,
                                prefill_topology,
                                decode_dp,
                                decode_topology,
                            )
                            .map(|(prefill, decode)| {
                                (
                                    PlacementPolicyBoundary(prefill),
                                    PlacementPolicyBoundary(decode),
                                )
                            })
                            .map_err(placement_boundary)
                    },
                )
                .map_err(runtime_error)?
                .with_capture_options(self.capture)
                .with_per_request_records(
                    self.spec.record_per_request || self.capture.effective_per_request(),
                )
                .with_max_sim_time_ms(self.spec.max_sim_time_ms);
                if let Some(policy) = scaling {
                    runtime = runtime.with_scaling_policy(Box::new(ScalingPolicyBoundary(policy)));
                }
                runtime.run().map_err(runtime_error)?.0
            }
        };

        Ok(finish_report(collector, self.spec.sla)
            .with_wall_time_ms(wall_start.elapsed().as_secs_f64() * 1_000.0))
    }
}

fn admission_queue<Metadata: ReplayAdmissionMetadata>(
    input: ReplayRuntimeInput,
    mode: ReplayMode,
) -> AdmissionQueue<Metadata> {
    match input {
        ReplayRuntimeInput::Requests(requests) => AdmissionQueue::new_requests(requests, mode),
        ReplayRuntimeInput::Workload(driver) => AdmissionQueue::new_workload(driver, mode),
    }
}

fn positive_delay(delay_ms: f64) -> Option<f64> {
    (delay_ms > 0.0).then_some(delay_ms)
}

fn lower_requests(
    spec: &ReplaySpec,
    determinism: ReplayDeterminism,
) -> ReplayResult<VecDeque<DirectRequest>> {
    let mut pending = spec
        .requests
        .iter()
        .enumerate()
        .map(|(index, request)| -> ReplayResult<_> {
            let request_id = match determinism {
                ReplayDeterminism::Random => Uuid::new_v4(),
                ReplayDeterminism::CanonicalV1 => Uuid::from_u128(
                    u128::try_from(index)
                        .expect("usize always fits u128")
                        .checked_add(1)
                        .expect("replay request index overflow"),
                ),
            };
            let (tokens, prompt_token_source) = match &request.input_token_ids {
                Some(tokens) => (tokens.clone(), ReplayPromptTokenSource::Materialized),
                None => {
                    let seed = u32::try_from(index)
                        .unwrap_or(u32::MAX)
                        .wrapping_mul(1_000_003);
                    (
                        (0..request.input_tokens)
                            .map(|offset| {
                                seed.wrapping_add(u32::try_from(offset).unwrap_or(u32::MAX))
                            })
                            .collect(),
                        ReplayPromptTokenSource::LengthOnlySynthetic,
                    )
                }
            };
            let routing = request.routing_metadata()?;
            Ok(DirectRequest {
                tokens,
                max_output_tokens: request.output_tokens,
                output_token_ids: request.output_token_ids.clone(),
                uuid: Some(request_id),
                dp_rank: 0,
                preferred_dp_rank: request.dp_rank,
                arrival_timestamp_ms: Some(request.arrival_time_ms),
                priority: routing.priority,
                strict_priority: routing.strict_priority,
                policy_class: routing.policy_class,
                replay_context: Some(ReplayRequestContext {
                    authored_id: request.id.clone(),
                    session_id: request.session_id.clone(),
                    turn_index: request.turn_index,
                    metadata: request.metadata.clone(),
                    prompt_token_source,
                }),
            })
        })
        .collect::<ReplayResult<Vec<_>>>()?;
    pending.sort_by(|left, right| {
        left.arrival_timestamp_ms
            .expect("ReplaySpec request always has an arrival")
            .total_cmp(
                &right
                    .arrival_timestamp_ms
                    .expect("ReplaySpec request always has an arrival"),
            )
    });
    Ok(pending.into())
}

fn apply_runtime_determinism(input: &mut ReplayRuntimeInput, determinism: ReplayDeterminism) {
    if determinism != ReplayDeterminism::CanonicalV1 {
        return;
    }
    match input {
        ReplayRuntimeInput::Requests(requests) => {
            for (index, request) in requests.iter_mut().enumerate() {
                request.uuid = Some(Uuid::from_u128(
                    u128::try_from(index)
                        .expect("usize always fits u128")
                        .checked_add(1)
                        .expect("replay request index overflow"),
                ));
            }
        }
        ReplayRuntimeInput::Workload(driver) => {
            driver.set_deterministic_request_ids(1);
        }
    }
}

fn finish_report(mut collector: crate::replay::TraceCollector, sla: SlaThresholds) -> ReplayReport {
    collector.set_sla_thresholds(sla);
    collector.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::{
        ProviderSpec, ReplayAdapters, ReplayRequest, ReplayTopology, WorkerPoolSpec,
    };

    #[test]
    fn replay_spec_lowering_preserves_correlation_routing_and_prompt_provenance() {
        let spec = ReplaySpec {
            version: 1,
            topology: ReplayTopology::Aggregated {
                workers: WorkerPoolSpec::default(),
            },
            engine: serde_json::Value::Null,
            adapters: ReplayAdapters {
                placement: ProviderSpec::round_robin(),
                scaling: ProviderSpec::no_scaling(),
            },
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: true,
            sla: Default::default(),
            requests: vec![
                ReplayRequest {
                    id: "length-only".into(),
                    arrival_time_ms: 0.0,
                    input_tokens: 3,
                    input_token_ids: None,
                    output_tokens: 2,
                    output_token_ids: None,
                    dp_rank: Some(2),
                    session_id: Some("session-a".into()),
                    turn_index: Some(4),
                    metadata: serde_json::json!({
                        "priority": -7,
                        "strict_priority": 9,
                        "policy_class": "latency",
                        "caller_tag": "preserved"
                    }),
                },
                ReplayRequest {
                    id: "materialized".into(),
                    arrival_time_ms: 1.0,
                    input_tokens: 2,
                    input_token_ids: Some(vec![41, 42]),
                    output_tokens: 1,
                    output_token_ids: None,
                    dp_rank: None,
                    session_id: None,
                    turn_index: None,
                    metadata: serde_json::Value::Null,
                },
            ],
        };

        let lowered = lower_requests(&spec, ReplayDeterminism::CanonicalV1)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        let first = &lowered[0];
        assert_eq!(first.uuid, Some(Uuid::from_u128(1)));
        assert_eq!(first.priority, -7);
        assert_eq!(first.strict_priority, 9);
        assert_eq!(first.policy_class.as_deref(), Some("latency"));
        assert_eq!(first.preferred_dp_rank, Some(2));
        assert!(!first.prompt_tokens_are_placement_safe());
        let context = first.replay_context.as_ref().unwrap();
        assert_eq!(context.authored_id, "length-only");
        assert_eq!(context.session_id.as_deref(), Some("session-a"));
        assert_eq!(context.turn_index, Some(4));
        assert_eq!(context.metadata["caller_tag"], "preserved");

        assert_eq!(lowered[1].tokens, vec![41, 42]);
        assert!(lowered[1].prompt_tokens_are_placement_safe());
    }

    #[test]
    fn random_lowering_does_not_use_ordinal_request_uuids() {
        let spec = ReplaySpec {
            version: 1,
            topology: ReplayTopology::aggregated(1),
            engine: serde_json::Value::Null,
            adapters: ReplayAdapters::default(),
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: false,
            sla: Default::default(),
            requests: vec![ReplayRequest {
                id: "random".into(),
                arrival_time_ms: 0.0,
                input_tokens: 1,
                input_token_ids: Some(vec![1]),
                output_tokens: 1,
                output_token_ids: None,
                dp_rank: None,
                session_id: None,
                turn_index: None,
                metadata: serde_json::Value::Null,
            }],
        };

        let first = lower_requests(&spec, ReplayDeterminism::Random)
            .unwrap()
            .pop_front()
            .unwrap();
        assert_ne!(first.uuid, Some(Uuid::from_u128(1)));
    }
}

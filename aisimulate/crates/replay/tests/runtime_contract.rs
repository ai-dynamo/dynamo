// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use aisimulate_engine::{Backend, EngineConfig, TimingModel, TimingModelConfig};
use aisimulate_replay::loadgen::ReplayRequestPayload;
use aisimulate_replay::{
    AggregatedRoundRobinPlacement, NoEngineEvents, NoReplayMetadata, Placement, PlacementEffects,
    PlacementPolicy, PoolRoundRobinPlacement, ProviderSpec, ReplayAdapters, ReplayCaptureOptions,
    ReplayComposition, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayError,
    ReplayRequest, ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec,
    ReplayTopology, Replayer, WorkerPoolSpec, WorkerTopology,
};
use uuid::Uuid;

fn request(
    id: &str,
    arrival_time_ms: f64,
    input_tokens: usize,
    output_tokens: usize,
) -> ReplayRequest {
    ReplayRequest {
        id: id.to_string(),
        arrival_time_ms,
        input_tokens,
        input_token_ids: None,
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        session_id: None,
        turn_index: None,
        metadata: serde_json::Value::Null,
    }
}

fn aggregated_spec(
    backend: Backend,
    workers: usize,
    startup_delay_ms: f64,
    requests: Vec<ReplayRequest>,
) -> ReplaySpec {
    let rank = EngineConfig {
        num_gpu_blocks: 64,
        block_size: 4,
        max_num_seqs: 4,
        max_num_batched_tokens: 64,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 10.0,
            decode_ms: 2.0,
        },
        ..EngineConfig::for_backend(backend)
    };
    ReplaySpec {
        version: 1,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                startup_delay_ms,
            },
        },
        engine: serde_json::to_value(ReplayEngineConfig {
            rank,
            ..ReplayEngineConfig::default()
        })
        .unwrap(),
        adapters: ReplayAdapters {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests,
    }
}

struct GeneralRoundRobinComposition {
    scaling: Option<Box<dyn ReplayScalingPolicy>>,
    scaling_construction_error: Option<&'static str>,
}

impl GeneralRoundRobinComposition {
    fn fixed_capacity() -> Self {
        Self {
            scaling: None,
            scaling_construction_error: None,
        }
    }

    fn with_scaling(scaling: impl ReplayScalingPolicy + 'static) -> Self {
        Self {
            scaling: Some(Box::new(scaling)),
            scaling_construction_error: None,
        }
    }

    fn failing_scaling_construction(message: &'static str) -> Self {
        Self {
            scaling: None,
            scaling_construction_error: Some(message),
        }
    }
}

impl ReplayComposition for GeneralRoundRobinComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }

    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn ReplayScalingPolicy>>> {
        if let Some(message) = self.scaling_construction_error {
            anyhow::bail!(message);
        }
        Ok(self.scaling.take())
    }
}

struct FailingPlacement;

impl PlacementPolicy<ReplayRequestPayload> for FailingPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        _request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<PlacementEffects> {
        anyhow::bail!("router placement callback failed")
    }

    fn observe(
        &mut self,
        _observation: Self::Observation,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
        false
    }

    fn request_terminal(
        &mut self,
        _request_id: Uuid,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(
        &mut self,
        _request_id: Uuid,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

struct FailingPlacementComposition {
    fail_construction: bool,
}

impl ReplayComposition for FailingPlacementComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = FailingPlacement;
    type DisaggregatedPlacement = FailingPlacement;

    fn create_aggregated_placement(
        &mut self,
        _dp_size: u32,
        _topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        if self.fail_construction {
            anyhow::bail!("router placement construction failed");
        }
        Ok(FailingPlacement)
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        _prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        _decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        if self.fail_construction {
            anyhow::bail!("router placement construction failed");
        }
        Ok((FailingPlacement, FailingPlacement))
    }
}

struct FailingScalingPolicy;

impl ReplayScalingPolicy for FailingScalingPolicy {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn on_tick(
        &mut self,
        _snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        anyhow::bail!("planner scaling callback failed")
    }
}

struct FailingTimingModel;

impl TimingModel for FailingTimingModel {
    fn predict_prefill_ms(
        &self,
        _batch_size: usize,
        _mean_isl: usize,
        _mean_prefix: usize,
    ) -> anyhow::Result<f64> {
        anyhow::bail!("engine timing callback failed")
    }

    fn predict_decode_ms(
        &self,
        _batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> anyhow::Result<f64> {
        anyhow::bail!("engine timing callback failed")
    }
}

fn canonical_options() -> ReplayCaptureOptions {
    ReplayCaptureOptions {
        determinism: ReplayDeterminism::CanonicalV1,
        ..ReplayCaptureOptions::default()
    }
}

#[test]
fn placement_construction_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("placement-construction", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        FailingPlacementComposition {
            fail_construction: true,
        },
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Placement("router placement construction failed".to_string())
    );
}

#[test]
fn placement_callback_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("placement-callback", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        FailingPlacementComposition {
            fail_construction: false,
        },
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Placement("router placement callback failed".to_string())
    );
}

#[test]
fn scaling_construction_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("scaling-construction", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::failing_scaling_construction(
            "planner scaling construction failed",
        ),
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Scaling("planner scaling construction failed".to_string())
    );
}

#[test]
fn scaling_callback_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("scaling-callback", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::with_scaling(FailingScalingPolicy),
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Scaling("planner scaling callback failed".to_string())
    );
}

#[test]
fn engine_callback_failure_retains_its_error_provenance() {
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("engine-callback", 0.0, 8, 2)],
    );
    let mut config: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
    config.rank.timing_model = TimingModelConfig::External {
        provider: "failing_test_timing".to_string(),
        config: serde_json::Value::Null,
    };
    spec.engine = serde_json::to_value(config).unwrap();

    let error = Replayer::new(
        spec,
        ReplayEngineFactory::with_timing_model(Arc::new(FailingTimingModel)),
    )
    .unwrap()
    .run()
    .unwrap_err();

    let ReplayError::Engine(message) = error else {
        panic!("expected an engine error, got {error:?}");
    };
    assert!(message.contains("engine timing callback failed"));
}

#[test]
fn authored_rank_is_validated_by_the_default_aggregated_runtime() {
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("invalid-rank", 0.0, 8, 3)],
    );
    spec.requests[0].dp_rank = Some(1);

    let default_error = Replayer::new(spec.clone(), ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(canonical_options())
        .run()
        .unwrap_err()
        .to_string();
    let general_error = Replayer::with_composition(
        spec,
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::fixed_capacity(),
    )
    .unwrap()
    .with_capture_options(canonical_options())
    .run()
    .unwrap_err()
    .to_string();

    assert_eq!(default_error, general_error);
    assert!(default_error.contains("preferred attention-DP rank 1 is out of range"));
}

#[test]
fn max_sim_time_is_a_soft_cap_for_the_aggregated_runtime() {
    let requests = (0..5)
        .map(|index| {
            request(
                &format!("request-{index}"),
                f64::from(index) * 1_000.0,
                4,
                2,
            )
        })
        .collect();
    let mut capped = aggregated_spec(Backend::Vllm, 1, 0.0, requests);
    capped.max_sim_time_ms = Some(2_500.0);

    let report = Replayer::new(capped, ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(canonical_options())
        .run()
        .unwrap();

    assert_eq!(report.request_counts.num_requests, 3);
    assert_eq!(report.request_counts.completed_requests, 3);
    assert!(report.throughput.duration_ms <= 2_500.0);
}

#[derive(Debug, Clone, PartialEq)]
struct ScalingObservation {
    now_ms: f64,
    active: Vec<usize>,
    starting: Vec<usize>,
    draining: Vec<usize>,
}

struct ScaleUpThenDown {
    step: usize,
    observations: Rc<RefCell<Vec<ScalingObservation>>>,
}

impl ReplayScalingPolicy for ScaleUpThenDown {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(100.0)
    }

    fn on_tick(
        &mut self,
        snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        self.observations.borrow_mut().push(ScalingObservation {
            now_ms: snapshot.now_ms,
            active: snapshot.active_decode_ids,
            starting: snapshot.starting_decode_ids,
            draining: snapshot.draining_decode_ids,
        });
        let decision = match self.step {
            0 => ReplayScalingDecision {
                target_decode: Some(2),
                next_tick_ms: Some(200.0),
                ..ReplayScalingDecision::default()
            },
            1 => ReplayScalingDecision {
                next_tick_ms: Some(300.0),
                ..ReplayScalingDecision::default()
            },
            2 => ReplayScalingDecision {
                target_decode: Some(1),
                next_tick_ms: Some(400.0),
                ..ReplayScalingDecision::default()
            },
            3 => ReplayScalingDecision::default(),
            _ => panic!("unexpected scaling tick {}", self.step),
        };
        self.step += 1;
        Ok(decision)
    }
}

#[test]
fn scaling_honors_startup_delay_then_scales_the_ready_worker_back_down() {
    let observations = Rc::new(RefCell::new(Vec::new()));
    let policy = ScaleUpThenDown {
        step: 0,
        observations: Rc::clone(&observations),
    };
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        200.0,
        vec![request("long-running", 0.0, 4, 2)],
    );
    let mut engine: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
    engine.rank.timing_model = TimingModelConfig::Fixed {
        prefill_ms: 1_000.0,
        decode_ms: 1_000.0,
    };
    spec.engine = serde_json::to_value(engine).unwrap();
    spec.adapters.scaling = ProviderSpec {
        provider: "scripted_test_policy".to_string(),
        config: serde_json::Value::Null,
    };

    let report = Replayer::with_composition(
        spec,
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::with_scaling(policy),
    )
    .unwrap()
    .run()
    .unwrap();

    assert_eq!(
        *observations.borrow(),
        vec![
            ScalingObservation {
                now_ms: 100.0,
                active: vec![0],
                starting: vec![],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 200.0,
                active: vec![0],
                starting: vec![1],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 300.0,
                active: vec![0, 1],
                starting: vec![],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 400.0,
                active: vec![0],
                starting: vec![],
                draining: vec![],
            },
        ]
    );
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(report.throughput.duration_ms, 3_000.0);
    assert!((report.throughput.decode_worker_seconds - 3.2).abs() < 1e-9);
}

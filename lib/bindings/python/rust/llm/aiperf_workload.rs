// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native workload composition selected by Dynamo's existing Python runner entrypoint.

use anyhow::Context;
use std::rc::Rc;

use aisimulate_core::replay::{
    ReplayComposition, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory,
    loadgen::{SteppableAgg, SteppableDisagg, SteppableReplay},
};
use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DeploymentTopology {
    Aggregated,
    #[default]
    Disaggregated,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DeploymentPlacement {
    RoundRobin,
    #[default]
    KvRouter,
}

fn one_worker() -> usize { 1 }

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeploymentConfig {
    #[serde(default)]
    topology: DeploymentTopology,
    #[serde(default)]
    placement: DeploymentPlacement,
    #[serde(default = "one_worker")]
    workers: usize,
    engine: ReplayEngineConfig,
    prefill_workers: usize,
    decode_workers: usize,
    admission_capacity: usize,
    router_config: Option<dynamo_mocker::placement::KvRouterConfig>,
}

/// Execute a complete strict AIPerf request inside the selected Dynamo provider.
/// Python performs discovery only; workload and per-token feedback stay in Rust.
#[pyfunction]
pub fn run_mocker_aiperf_workload(
    py: Python<'_>,
    run_request_json: String,
    deployment_config_json: String,
    identity_seed_hex: String,
) -> PyResult<String> {
    const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
    if run_request_json.len() > MAX_INPUT_BYTES || deployment_config_json.len() > MAX_INPUT_BYTES {
        return Err(PyValueError::new_err("workload input exceeds 64 MiB"));
    }
    if identity_seed_hex.len() != 64
        || !identity_seed_hex
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(PyValueError::new_err(
            "identity seed requires 64 hexadecimal digits",
        ));
    }
    let mut seed = [0; 32];
    for (index, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&identity_seed_hex[index * 2..index * 2 + 2], 16)
            .map_err(|_| PyValueError::new_err("invalid identity seed"))?;
    }
    py.allow_threads(move || -> anyhow::Result<String> {
        let request = aiperf_protocol::runner_v2::decode_run_request_v2(run_request_json.as_bytes())
            .context("invalid native run request")?;
        let deployment: DeploymentConfig = serde_json::from_str(&deployment_config_json)
            .context("invalid deployment configuration")?;
        anyhow::ensure!(deployment.prefill_workers > 0 && deployment.decode_workers > 0,
            "prefill and decode worker counts must be positive");
        let role_config = |role: Option<&aisimulate_core::replay::ReplayRoleConfig>| -> anyhow::Result<MockerConfig> {
            let rank = role.map_or(&deployment.engine.rank, |role| &role.rank);
            let tp = role.map_or(deployment.engine.tensor_parallel_size, |role| role.tensor_parallel_size);
            let cp = role.map_or(deployment.engine.prefill_context_parallel_size, |role| role.prefill_context_parallel_size);
            anyhow::ensure!(cp == 1, "prefill context parallelism is not represented by this provider launch contract");
            let launch = aisimulate_core::engine::EngineLaunchConfig {
                engine: rank.clone(),
                dp_size: role.map_or(deployment.engine.dp_size, |role| role.dp_size),
                tensor_parallel_size: usize::try_from(tp)?,
                num_gpu_blocks_is_explicit: role.and_then(|role| role.num_gpu_blocks_is_explicit)
                    .or(deployment.engine.num_gpu_blocks_is_explicit).unwrap_or(false),
                ..Default::default()
            };
            MockerConfig::from_value(serde_json::to_value(launch)?)
        };
        let prefill = role_config(deployment.engine.prefill.as_ref()).context("unsupported prefill role configuration")?;
        let decode = role_config(deployment.engine.decode.as_ref()).context("unsupported decode role configuration")?;
        let model: Box<dyn SteppableReplay> = match (deployment.topology, deployment.placement) {
            (DeploymentTopology::Aggregated, DeploymentPlacement::RoundRobin) => {
                anyhow::ensure!(deployment.workers > 0, "aggregate worker count must be positive");
                anyhow::ensure!(deployment.router_config.is_none(), "router configuration requires KV placement");
                Box::new(SteppableAgg::new(deployment.engine, &ReplayEngineFactory::new(), deployment.workers)?)
            }
            (DeploymentTopology::Disaggregated, DeploymentPlacement::RoundRobin) => {
                anyhow::ensure!(deployment.router_config.is_none(), "router configuration requires KV placement");
                Box::new(SteppableDisagg::new(deployment.engine, &ReplayEngineFactory::new(), deployment.prefill_workers, deployment.decode_workers)?)
            }
            (DeploymentTopology::Disaggregated, DeploymentPlacement::KvRouter) => {
                let mut composition = KvReplayComposition::disaggregated(
                    prefill, decode, deployment.prefill_workers, deployment.decode_workers,
                    deployment.router_config, None, None,
                );
                composition.set_determinism(ReplayDeterminism::CanonicalV1)?;
                Box::new(SteppableDisagg::<_, dynamo_mocker::placement::RouterEventObservation,
                    dynamo_mocker::placement::KvReplayMetadata>::with_placements(
                    deployment.engine, &ReplayEngineFactory::new(),
                    deployment.prefill_workers, deployment.decode_workers,
                    |pdp, pt, ddp, dt| composition.create_disaggregated_placements(pdp, pt, ddp, dt),
                )?)
            }
            (DeploymentTopology::Aggregated, DeploymentPlacement::KvRouter) => {
                anyhow::bail!("aggregate KV placement binding is not implemented")
            }
        };
        let report = aisim_aiperf_workload_runner::execute_with_deployment(
            &request.run,
            aiperf_clock_runtime::cancellation::RunCancellation::new(),
            Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
            model, deployment.admission_capacity, seed,
        )?;
        Ok(serde_json::to_string(&report)?)
    }).map_err(|error| {
        // Only static outer-stage labels cross this boundary; nested source
        // diagnostics may contain private prompts, paths or endpoint credentials.
        let stage = error.to_string();
        let safe = match stage.as_str() {
            "invalid native run request" | "invalid deployment configuration"
            | "unsupported prefill role configuration" | "unsupported decode role configuration"
            | "prefill and decode worker counts must be positive" => stage,
            _ => "native workload configuration or execution failed".into(),
        };
        PyValueError::new_err(safe)
    })
}

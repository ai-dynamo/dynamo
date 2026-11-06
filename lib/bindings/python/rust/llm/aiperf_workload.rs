// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native workload composition selected by Dynamo's existing Python runner entrypoint.

use std::rc::Rc;

use aisimulate_core::replay::{
    ReplayComposition, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory,
    loadgen::SteppableDisagg,
};
use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeploymentConfig {
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
    if identity_seed_hex.len() != 64 || !identity_seed_hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PyValueError::new_err("identity seed requires 64 hexadecimal digits"));
    }
    let mut seed = [0; 32];
    for (index, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&identity_seed_hex[index * 2..index * 2 + 2], 16)
            .map_err(|_| PyValueError::new_err("invalid identity seed"))?;
    }
    py.allow_threads(move || -> anyhow::Result<String> {
        let request = aiperf_protocol::runner_v2::decode_run_request_v2(run_request_json.as_bytes())?;
        let deployment: DeploymentConfig = serde_json::from_str(&deployment_config_json)?;
        anyhow::ensure!(deployment.prefill_workers > 0 && deployment.decode_workers > 0,
            "prefill and decode worker counts must be positive");
        let mut prefill = MockerConfig::default();
        prefill.engine.engine = deployment.engine.prefill.as_ref()
            .map_or_else(|| deployment.engine.rank.clone(), |role| role.rank.clone());
        prefill.engine.dp_size = deployment.engine.prefill.as_ref()
            .map_or(deployment.engine.dp_size, |role| role.dp_size);
        let mut decode = MockerConfig::default();
        decode.engine.engine = deployment.engine.decode.as_ref()
            .map_or_else(|| deployment.engine.rank.clone(), |role| role.rank.clone());
        decode.engine.dp_size = deployment.engine.decode.as_ref()
            .map_or(deployment.engine.dp_size, |role| role.dp_size);
        let mut composition = KvReplayComposition::disaggregated(
            prefill, decode, deployment.prefill_workers, deployment.decode_workers,
            deployment.router_config, None, None,
        );
        composition.set_determinism(ReplayDeterminism::CanonicalV1)?;
        let model = SteppableDisagg::<_, dynamo_mocker::placement::RouterEventObservation,
            dynamo_mocker::placement::KvReplayMetadata>::with_placements(
            deployment.engine, &ReplayEngineFactory::new(),
            deployment.prefill_workers, deployment.decode_workers,
            |pdp, pt, ddp, dt| composition.create_disaggregated_placements(pdp, pt, ddp, dt),
        )?;
        let report = aisim_aiperf_workload_runner::execute_with_deployment(
            &request.run,
            aiperf_clock_runtime::cancellation::RunCancellation::new(),
            Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
            Box::new(model), deployment.admission_capacity, seed,
        )?;
        Ok(serde_json::to_string(&report)?)
    }).map_err(|error| PyValueError::new_err(format!("native workload execution failed: {error:#}")))
}

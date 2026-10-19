// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Source and modification provenance: adjacent README.md.

use aisimulate_core::replay::{ReplayCaptureOptions, ReplayTelemetryObserver};
use anyhow::{Result, ensure};

/// Execute native routing inside AISimulate, retaining optional consumer observers
/// and scaling callbacks without transferring ownership of the simulation loop.
#[cfg(feature = "python-replay")]
pub fn execute_native_router_replay(
    payload: &str,
    capture: ReplayCaptureOptions,
    scaling: Option<Box<dyn aisimulate_core::replay::ReplayScalingPolicy>>,
    telemetry: Option<(f64, Box<dyn ReplayTelemetryObserver>)>,
    prefill_load_estimator: Option<super::ReplayPrefillLoadEstimator>,
) -> Result<aisimulate_core::ReplayExecutionResult> {
    use super::{NATIVE_DYNAMO_PROVIDER, NativeDynamoReplayComposition, NativeDynamoRouterConfig};
    let envelope: serde_json::Value = serde_json::from_str(payload)?;
    let spec = envelope.get("spec").unwrap_or(&envelope);
    ensure!(
        spec.pointer("/adapters/placement/provider")
            .and_then(|v| v.as_str())
            == Some(NATIVE_DYNAMO_PROVIDER),
        "native routing requires the dynamo_kv_router placement provider"
    );
    let config = NativeDynamoRouterConfig::from_value(
        spec.pointer("/adapters/placement/config")
            .unwrap_or(&serde_json::Value::Null),
    )?;
    let capture_decisions = capture.effective_per_request()
        || spec
            .get("record_per_request")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
    let mut composition = NativeDynamoReplayComposition::from_config(config, scaling)
        .with_capture_decisions(capture_decisions);
    if let Some(estimator) = prefill_load_estimator {
        composition = composition.with_prefill_load_estimator(estimator);
    }
    let evidence = composition.routing_evidence();
    let mut result =
        aisimulate_core::execute_replay_with_composition(payload, composition, capture, telemetry)?;
    result
        .report_fields
        .insert("routing_policy".into(), evidence.snapshot()?);
    Ok(result)
}

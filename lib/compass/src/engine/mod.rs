// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod critical_path;
pub mod floor;
pub mod saturation;
pub mod verdict;

use crate::config::CompassConfig;
use crate::types::{
    AttributionReport, ComponentContribution, DeploymentInfo, EndToEndMetrics, FloorCheck,
    SaturationEntry, SubComponentBreakdown, TimeWindow, WorkloadSummary,
};

pub use critical_path::{CriticalPathResult, SpanNode};
pub use floor::{FloorInput, FloorResult};
pub use saturation::{QueueSnapshot, SaturationResult};
pub use verdict::ScoredComponent;

pub struct EngineInput {
    pub window: TimeWindow,
    pub deployment: DeploymentInfo,
    pub workload: WorkloadSummary,
    pub end_to_end: EndToEndMetrics,
    pub traces: Vec<Vec<SpanNode>>,
    pub queue_snapshots: Vec<QueueSnapshot>,
    pub floor_inputs: Vec<FloorInput>,
    pub sub_component_breakdowns: Vec<SubComponentBreakdown>,
}

pub fn run_attribution(input: EngineInput, config: &CompassConfig) -> AttributionReport {
    let cp_result = critical_path::analyze_critical_path(&input.traces);
    let sat_results = saturation::detect_saturation(&input.queue_snapshots);
    let floor_results = floor::compute_floors(&input.floor_inputs, &config.floor_params);

    let verdict = verdict::compute_verdict(
        &cp_result,
        &sat_results,
        &floor_results,
        &config.weights,
        &config.confidence_thresholds,
    );

    let per_component: Vec<ComponentContribution> = cp_result.components;

    let saturation_entries: Vec<SaturationEntry> = sat_results
        .iter()
        .map(|s| SaturationEntry {
            component: s.component.clone(),
            utilization: s.utilization,
            queue_trend: s.queue_trend,
            warning: s.saturation_score > 0.8,
        })
        .collect();

    let floor_checks: Vec<FloorCheck> = floor_results
        .iter()
        .map(|f| FloorCheck {
            component: f.component.clone(),
            observed_ms: f.observed_ms,
            floor_ms: f.theoretical_floor_ms,
            ratio: f.floor_ratio,
            is_optimization_candidate: f.is_optimization_candidate,
        })
        .collect();

    AttributionReport {
        compass_version: crate::config::COMPASS_VERSION.to_string(),
        window: input.window,
        deployment: input.deployment,
        workload: input.workload,
        end_to_end: input.end_to_end,
        verdict,
        per_component,
        saturation: saturation_entries,
        sub_component_breakdown: input.sub_component_breakdowns,
        floor_checks,
    }
}

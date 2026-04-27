// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::types::SweepResult;

#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    pub target_component: String,
    pub current_ttft_p99_ms: f64,
    pub slo_ttft_p99_ms: f64,
    pub required_reduction_pct: f64,
    pub predicted_ttft_after_ms: f64,
    pub is_achievable: bool,
}

pub fn find_minimum_improvement(
    target_component: &str,
    current_ttft_p99_ms: f64,
    slo_ttft_p99_ms: f64,
    component_contribution_pct: f64,
) -> CounterfactualResult {
    if current_ttft_p99_ms <= slo_ttft_p99_ms {
        return CounterfactualResult {
            target_component: target_component.to_string(),
            current_ttft_p99_ms,
            slo_ttft_p99_ms,
            required_reduction_pct: 0.0,
            predicted_ttft_after_ms: current_ttft_p99_ms,
            is_achievable: true,
        };
    }

    let excess = current_ttft_p99_ms - slo_ttft_p99_ms;
    let component_contribution_fraction = component_contribution_pct / 100.0;
    let component_latency = current_ttft_p99_ms * component_contribution_fraction;

    let required_reduction_ms = excess;
    let required_reduction_pct = if component_latency > 0.0 {
        (required_reduction_ms / component_latency) * 100.0
    } else {
        f64::INFINITY
    };

    let is_achievable = required_reduction_pct <= 100.0;
    let actual_reduction = if is_achievable {
        required_reduction_ms
    } else {
        component_latency
    };

    CounterfactualResult {
        target_component: target_component.to_string(),
        current_ttft_p99_ms,
        slo_ttft_p99_ms,
        required_reduction_pct: required_reduction_pct.min(100.0),
        predicted_ttft_after_ms: current_ttft_p99_ms - actual_reduction,
        is_achievable,
    }
}

pub fn format_counterfactual(result: &CounterfactualResult) -> String {
    if result.is_achievable {
        format!(
            "A {:.0}% reduction in {} latency would bring p99 TTFT from {:.0}ms to ~{:.0}ms (SLO: {:.0}ms).",
            result.required_reduction_pct,
            result.target_component,
            result.current_ttft_p99_ms,
            result.predicted_ttft_after_ms,
            result.slo_ttft_p99_ms
        )
    } else {
        format!(
            "Eliminating {} entirely would reduce p99 TTFT to ~{:.0}ms, which is still above SLO ({:.0}ms). Multiple components need improvement.",
            result.target_component,
            result.predicted_ttft_after_ms,
            result.slo_ttft_p99_ms
        )
    }
}

impl std::fmt::Display for CounterfactualResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format_counterfactual(self))
    }
}

impl From<CounterfactualResult> for SweepResult {
    fn from(cf: CounterfactualResult) -> Self {
        SweepResult {
            perturbation: cf.target_component,
            multiplier: 1.0 - cf.required_reduction_pct / 100.0,
            concurrency: 0,
            predicted_ttft_p99_ms: cf.predicted_ttft_after_ms,
            predicted_throughput_rps: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_already_within_slo() {
        let result = find_minimum_improvement("kvbm.allocate", 300.0, 400.0, 47.0);
        assert!(result.is_achievable);
        assert_eq!(result.required_reduction_pct, 0.0);
    }

    #[test]
    fn test_achievable_improvement() {
        let result = find_minimum_improvement("kvbm.allocate", 620.0, 400.0, 47.0);
        assert!(result.is_achievable);
        assert!(result.required_reduction_pct > 0.0);
        assert!(result.required_reduction_pct < 100.0);
    }

    #[test]
    fn test_not_achievable() {
        let result = find_minimum_improvement("router.dispatch", 620.0, 400.0, 5.0);
        assert!(!result.is_achievable);
    }

    #[test]
    fn test_format() {
        let result = find_minimum_improvement("kvbm.allocate", 620.0, 400.0, 47.0);
        let formatted = format_counterfactual(&result);
        assert!(formatted.contains("kvbm.allocate"));
        assert!(formatted.contains("reduction"));
    }
}

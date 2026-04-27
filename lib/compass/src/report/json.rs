// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::types::{AttributionReport, CalibrationResult, DiffReport, SensitivityMatrix};

pub fn format_report(report: &AttributionReport) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

pub fn format_sensitivity_matrix(matrix: &SensitivityMatrix) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(matrix)?)
}

pub fn format_diff(diff: &DiffReport) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(diff)?)
}

pub fn format_calibration(result: &CalibrationResult) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(result)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    #[test]
    fn test_format_report() {
        let report = AttributionReport {
            compass_version: "1.0.0".to_string(),
            window: TimeWindow {
                start: Utc::now(),
                end: Utc::now(),
            },
            deployment: DeploymentInfo {
                name: "test".to_string(),
                engine: "vllm".to_string(),
                topology: "disaggregated".to_string(),
            },
            workload: WorkloadSummary {
                qps: 12.3,
                isl_p50: 2048,
                osl_p50: 256,
            },
            end_to_end: EndToEndMetrics {
                ttft_ms: PercentileSet {
                    p50: 142.0,
                    p95: 380.0,
                    p99: 620.0,
                },
                itl_ms: PercentileSet {
                    p50: 18.0,
                    p95: 31.0,
                    p99: 47.0,
                },
            },
            verdict: Verdict {
                primary_bottleneck: "kvbm.allocate".to_string(),
                attribution_pct: 47.2,
                confidence: Confidence::High,
                evidence: vec!["trace://example".to_string()],
                recommended_action: "investigate".to_string(),
            },
            per_component: vec![],
            saturation: vec![],
            sub_component_breakdown: vec![],
            floor_checks: vec![],
        };

        let json = format_report(&report).unwrap();
        assert!(json.contains("kvbm.allocate"));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["verdict"]["confidence"], "high");
    }
}

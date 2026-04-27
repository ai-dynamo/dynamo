// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileSet {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl PercentileSet {
    pub fn from_samples(samples: &mut [f64]) -> Self {
        if samples.is_empty() {
            return Self {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = samples.len();
        let idx = |pct: usize| ((len - 1) * pct / 100).min(len - 1);
        Self {
            p50: samples[idx(50)],
            p95: samples[idx(95)],
            p99: samples[idx(99)],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    pub name: String,
    pub engine: String,
    pub topology: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSummary {
    pub qps: f64,
    pub isl_p50: u64,
    pub osl_p50: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::High => write!(f, "high"),
            Self::Medium => write!(f, "medium"),
            Self::Low => write!(f, "low"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub primary_bottleneck: String,
    pub attribution_pct: f64,
    pub confidence: Confidence,
    pub evidence: Vec<String>,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentContribution {
    pub component: String,
    pub sub_component: Option<String>,
    pub contribution_pct: f64,
    pub latency_ms: PercentileSet,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueTrend {
    Growing,
    Stable,
    Draining,
}

impl std::fmt::Display for QueueTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Growing => write!(f, "growing"),
            Self::Stable => write!(f, "stable"),
            Self::Draining => write!(f, "draining"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationEntry {
    pub component: String,
    pub utilization: f64,
    pub queue_trend: QueueTrend,
    pub warning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubComponentBreakdown {
    pub component: String,
    pub phases: Vec<PhaseMetrics>,
    pub theoretical_floor_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    pub name: String,
    pub p99_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_cpu_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lock_wait_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorCheck {
    pub component: String,
    pub observed_ms: f64,
    pub floor_ms: f64,
    pub ratio: f64,
    pub is_optimization_candidate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionReport {
    pub compass_version: String,
    pub window: TimeWindow,
    pub deployment: DeploymentInfo,
    pub workload: WorkloadSummary,
    pub end_to_end: EndToEndMetrics,
    pub verdict: Verdict,
    pub per_component: Vec<ComponentContribution>,
    pub saturation: Vec<SaturationEntry>,
    pub sub_component_breakdown: Vec<SubComponentBreakdown>,
    pub floor_checks: Vec<FloorCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndToEndMetrics {
    pub ttft_ms: PercentileSet,
    pub itl_ms: PercentileSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub perturbation: String,
    pub multiplier: f64,
    pub concurrency: usize,
    pub predicted_ttft_p99_ms: f64,
    pub predicted_throughput_rps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityMatrix {
    pub sweep_config: SweepConfig,
    pub results: Vec<SweepResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    pub trace_source: String,
    pub perturbations: Vec<PerturbationSpec>,
    pub concurrency_levels: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationSpec {
    pub component: String,
    pub multipliers: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffReport {
    pub report_a: String,
    pub report_b: String,
    pub verdict_changed: bool,
    pub component_deltas: Vec<ComponentDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDelta {
    pub component: String,
    pub attribution_pct_a: f64,
    pub attribution_pct_b: f64,
    pub delta_pct: f64,
    pub latency_p99_a: f64,
    pub latency_p99_b: f64,
    pub latency_delta_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub trace_source: String,
    pub mocker_ttft_p99_ms: f64,
    pub real_ttft_p99_ms: f64,
    pub residual_pct: f64,
    pub is_calibrated: bool,
    pub threshold_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_set_from_samples() {
        let mut samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let ps = PercentileSet::from_samples(&mut samples);
        assert!((ps.p50 - 50.0).abs() < 1.0);
        assert!((ps.p95 - 95.0).abs() < 1.0);
        assert!((ps.p99 - 99.0).abs() < 1.0);
    }

    #[test]
    fn test_percentile_set_empty() {
        let ps = PercentileSet::from_samples(&mut []);
        assert_eq!(ps.p50, 0.0);
    }

    #[test]
    fn test_attribution_report_serialization() {
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
                recommended_action: "investigate radix tree lock contention".to_string(),
            },
            per_component: vec![],
            saturation: vec![],
            sub_component_breakdown: vec![],
            floor_checks: vec![],
        };
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("kvbm.allocate"));
        assert!(json.contains("47.2"));

        let deserialized: AttributionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.verdict.primary_bottleneck, "kvbm.allocate");
    }
}

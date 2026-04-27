// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::{ConfidenceThresholds, WeightProfile};
use crate::engine::critical_path::CriticalPathResult;
use crate::engine::floor::FloorResult;
use crate::engine::saturation::SaturationResult;
use crate::types::{Confidence, Verdict};

#[derive(Debug, Clone)]
pub struct ScoredComponent {
    pub component: String,
    pub critical_path_pct: f64,
    pub saturation_score: f64,
    pub floor_ratio: f64,
    pub composite_score: f64,
}

pub fn compute_verdict(
    critical_path: &CriticalPathResult,
    saturation: &[SaturationResult],
    floors: &[FloorResult],
    weights: &WeightProfile,
    thresholds: &ConfidenceThresholds,
) -> Verdict {
    let mut scored: Vec<ScoredComponent> = critical_path
        .components
        .iter()
        .map(|cp| {
            let component_key = match &cp.sub_component {
                Some(sub) => format!("{}.{}", cp.component, sub),
                None => cp.component.clone(),
            };

            let sat_score = saturation
                .iter()
                .find(|s| s.component == component_key || s.component.starts_with(&cp.component))
                .map(|s| s.saturation_score)
                .unwrap_or(0.0);

            let fr = floors
                .iter()
                .find(|f| f.component == component_key || f.component.starts_with(&cp.component))
                .map(|f| f.floor_ratio)
                .unwrap_or(1.0);

            let cp_normalized = cp.contribution_pct / 100.0;
            let floor_term = if fr > 1.0 { 1.0 - 1.0 / fr } else { 0.0 };

            let composite = weights.critical_path_weight * cp_normalized
                + weights.saturation_weight * sat_score
                + weights.floor_ratio_weight * floor_term;

            ScoredComponent {
                component: component_key,
                critical_path_pct: cp.contribution_pct,
                saturation_score: sat_score,
                floor_ratio: fr,
                composite_score: composite,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.composite_score
            .partial_cmp(&a.composite_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if scored.is_empty() {
        return Verdict {
            primary_bottleneck: "unknown".to_string(),
            attribution_pct: 0.0,
            confidence: Confidence::Low,
            evidence: vec![],
            recommended_action: "insufficient data for attribution".to_string(),
        };
    }

    let top = &scored[0];
    let gap = if scored.len() > 1 {
        top.composite_score - scored[1].composite_score
    } else {
        top.composite_score
    };

    let confidence = if gap > thresholds.high_gap {
        Confidence::High
    } else if gap > thresholds.medium_gap {
        Confidence::Medium
    } else {
        Confidence::Low
    };

    let recommended_action = generate_recommendation(top);

    Verdict {
        primary_bottleneck: top.component.clone(),
        attribution_pct: top.critical_path_pct,
        confidence,
        evidence: generate_evidence(top),
        recommended_action,
    }
}

fn generate_recommendation(top: &ScoredComponent) -> String {
    if top.component.contains("kvbm") && top.saturation_score > 0.8 {
        format!(
            "Investigate radix tree lock contention in {}. Floor ratio {:.1}x suggests significant optimization potential.",
            top.component, top.floor_ratio
        )
    } else if top.component.contains("nixl") {
        format!(
            "Check NIXL transfer bandwidth and tier placement. {} shows {:.1}% critical path contribution.",
            top.component, top.critical_path_pct
        )
    } else if top.component.contains("prefill") && top.floor_ratio < 1.1 {
        format!(
            "{} is near theoretical floor ({:.2}x). Consider model parallelism or hardware upgrade.",
            top.component, top.floor_ratio
        )
    } else if top.component.contains("router") {
        format!(
            "Router overhead at {:.1}% of critical path. Review KV overlap scoring and worker selection logic.",
            top.critical_path_pct
        )
    } else {
        format!(
            "{} contributes {:.1}% of p99 critical path (floor ratio: {:.1}x, saturation: {:.0}%). Investigate for optimization.",
            top.component,
            top.critical_path_pct,
            top.floor_ratio,
            top.saturation_score * 100.0
        )
    }
}

fn generate_evidence(top: &ScoredComponent) -> Vec<String> {
    let mut evidence = vec![format!(
        "critical_path_contribution: {:.1}%",
        top.critical_path_pct
    )];
    if top.saturation_score > 0.0 {
        evidence.push(format!(
            "saturation_score: {:.2}",
            top.saturation_score
        ));
    }
    if top.floor_ratio > 1.0 {
        evidence.push(format!("floor_ratio: {:.2}x", top.floor_ratio));
    }
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::critical_path::CriticalPathResult;
    use crate::types::{ComponentContribution, PercentileSet};

    fn make_cp(component: &str, sub: Option<&str>, pct: f64) -> ComponentContribution {
        ComponentContribution {
            component: component.to_string(),
            sub_component: sub.map(String::from),
            contribution_pct: pct,
            latency_ms: PercentileSet {
                p50: 1.0,
                p95: 2.0,
                p99: 3.0,
            },
            score: 0.0,
        }
    }

    #[test]
    fn test_clear_bottleneck_high_confidence() {
        let cp = CriticalPathResult {
            components: vec![
                make_cp("kvbm", Some("allocate"), 47.2),
                make_cp("nixl", Some("transfer"), 25.0),
                make_cp("prefill", Some("compute"), 20.0),
            ],
            total_latency_ms: PercentileSet {
                p50: 142.0,
                p95: 380.0,
                p99: 620.0,
            },
        };
        let sat = vec![SaturationResult {
            component: "kvbm.radix_tree_lock".to_string(),
            utilization: 0.91,
            queue_trend: crate::types::QueueTrend::Growing,
            saturation_score: 0.95,
        }];
        let floors = vec![FloorResult {
            component: "kvbm.allocate".to_string(),
            observed_ms: 3.51,
            theoretical_floor_ms: 0.80,
            floor_ratio: 4.39,
            is_optimization_candidate: true,
        }];

        let verdict = compute_verdict(
            &cp,
            &sat,
            &floors,
            &WeightProfile::default(),
            &ConfidenceThresholds::default(),
        );

        assert_eq!(verdict.primary_bottleneck, "kvbm.allocate");
        assert!((verdict.attribution_pct - 47.2).abs() < 0.01);
        assert_eq!(verdict.confidence, Confidence::High);
    }

    #[test]
    fn test_ambiguous_low_confidence() {
        let cp = CriticalPathResult {
            components: vec![
                make_cp("a", None, 35.0),
                make_cp("b", None, 33.0),
                make_cp("c", None, 32.0),
            ],
            total_latency_ms: PercentileSet {
                p50: 100.0,
                p95: 150.0,
                p99: 200.0,
            },
        };
        let verdict = compute_verdict(
            &cp,
            &[],
            &[],
            &WeightProfile::default(),
            &ConfidenceThresholds::default(),
        );
        assert!(matches!(
            verdict.confidence,
            Confidence::Low | Confidence::Medium
        ));
    }

    #[test]
    fn test_empty_data() {
        let cp = CriticalPathResult {
            components: vec![],
            total_latency_ms: PercentileSet {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            },
        };
        let verdict = compute_verdict(
            &cp,
            &[],
            &[],
            &WeightProfile::default(),
            &ConfidenceThresholds::default(),
        );
        assert_eq!(verdict.primary_bottleneck, "unknown");
        assert_eq!(verdict.confidence, Confidence::Low);
    }

    #[test]
    fn test_prefill_at_floor_not_bottleneck() {
        let cp = CriticalPathResult {
            components: vec![
                make_cp("kvbm", Some("allocate"), 30.0),
                make_cp("prefill", Some("compute"), 50.0),
            ],
            total_latency_ms: PercentileSet {
                p50: 100.0,
                p95: 150.0,
                p99: 200.0,
            },
        };
        let sat = vec![SaturationResult {
            component: "kvbm.allocate".to_string(),
            utilization: 0.9,
            queue_trend: crate::types::QueueTrend::Growing,
            saturation_score: 0.9,
        }];
        let floors = vec![
            FloorResult {
                component: "kvbm.allocate".to_string(),
                observed_ms: 3.5,
                theoretical_floor_ms: 0.8,
                floor_ratio: 4.4,
                is_optimization_candidate: true,
            },
            FloorResult {
                component: "prefill.compute".to_string(),
                observed_ms: 12.1,
                theoretical_floor_ms: 11.9,
                floor_ratio: 1.02,
                is_optimization_candidate: false,
            },
        ];

        let verdict = compute_verdict(
            &cp,
            &sat,
            &floors,
            &WeightProfile::default(),
            &ConfidenceThresholds::default(),
        );

        assert_eq!(verdict.primary_bottleneck, "kvbm.allocate");
    }
}

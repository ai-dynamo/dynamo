// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::types::{
    AttributionReport, CalibrationResult, Confidence, DiffReport, SensitivityMatrix,
};

pub fn format_report(report: &AttributionReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "\n  COMPASS ATTRIBUTION REPORT  (v{})\n",
        report.compass_version
    ));
    out.push_str(&"=".repeat(60));
    out.push('\n');

    out.push_str(&format!(
        "\nDeployment: {}  |  Engine: {}  |  Topology: {}\n",
        report.deployment.name, report.deployment.engine, report.deployment.topology
    ));
    out.push_str(&format!(
        "Window: {} -> {}\n",
        report.window.start.format("%H:%M:%S"),
        report.window.end.format("%H:%M:%S")
    ));
    out.push_str(&format!(
        "Workload: {:.1} qps  |  ISL p50: {}  |  OSL p50: {}\n",
        report.workload.qps, report.workload.isl_p50, report.workload.osl_p50
    ));

    out.push_str(&format!(
        "\nEnd-to-End:\n  TTFT  p50: {:.0}ms  p95: {:.0}ms  p99: {:.0}ms\n  ITL   p50: {:.0}ms  p95: {:.0}ms  p99: {:.0}ms\n",
        report.end_to_end.ttft_ms.p50,
        report.end_to_end.ttft_ms.p95,
        report.end_to_end.ttft_ms.p99,
        report.end_to_end.itl_ms.p50,
        report.end_to_end.itl_ms.p95,
        report.end_to_end.itl_ms.p99
    ));

    out.push('\n');
    out.push_str(&"-".repeat(60));
    let confidence_marker = match report.verdict.confidence {
        Confidence::High => "[HIGH]",
        Confidence::Medium => "[MED]",
        Confidence::Low => "[LOW]",
    };
    out.push_str(&format!(
        "\nVERDICT: {} is the primary bottleneck ({:.1}% of p99 TTFT excess)\n",
        report.verdict.primary_bottleneck, report.verdict.attribution_pct
    ));
    out.push_str(&format!(
        "         confidence: {} (top-2 gap)\n",
        confidence_marker
    ));
    out.push_str(&"-".repeat(60));
    out.push('\n');

    if !report.per_component.is_empty() {
        out.push_str("\nPER-COMPONENT (p99 contribution to critical path):\n");
        for comp in &report.per_component {
            let label = match &comp.sub_component {
                Some(sub) => format!("{}.{}", comp.component, sub),
                None => comp.component.clone(),
            };
            let marker = if label == report.verdict.primary_bottleneck {
                " <-- bottleneck"
            } else {
                ""
            };
            out.push_str(&format!(
                "  {:30} {:>7.2} ms  ({:>5.1}%){}\n",
                label, comp.latency_ms.p99, comp.contribution_pct, marker
            ));
        }
    }

    if !report.sub_component_breakdown.is_empty() {
        out.push_str("\nSUB-COMPONENT BREAKDOWN:\n");
        for breakdown in &report.sub_component_breakdown {
            out.push_str(&format!("  {}:\n", breakdown.component));
            for phase in &breakdown.phases {
                let mut detail = format!("{:.2} ms", phase.p99_ms);
                if let (Some(cpu), Some(lock)) = (phase.on_cpu_ms, phase.lock_wait_ms) {
                    detail.push_str(&format!("  (on_cpu: {:.2}ms, lock_wait: {:.2}ms)", cpu, lock));
                }
                out.push_str(&format!("    {:20} {}\n", phase.name, detail));
            }
            if let Some(floor) = breakdown.theoretical_floor_ms {
                out.push_str(&format!("    {:20} {:.2} ms\n", "theoretical_floor", floor));
            }
        }
    }

    if !report.saturation.is_empty() {
        out.push_str("\nSATURATION (Little's Law):\n");
        for sat in &report.saturation {
            let warning = if sat.warning { " !!" } else { "" };
            out.push_str(&format!(
                "  {:30} utilization {:.2}, {}{}\n",
                sat.component, sat.utilization, sat.queue_trend, warning
            ));
        }
    }

    if !report.floor_checks.is_empty() {
        out.push_str("\nTHEORETICAL FLOOR CHECK:\n");
        for floor in &report.floor_checks {
            let marker = if floor.is_optimization_candidate {
                " !!"
            } else {
                " (at floor)"
            };
            out.push_str(&format!(
                "  {:30} observed {:.2} / floor {:.2} = {:.2}x{}\n",
                floor.component, floor.observed_ms, floor.floor_ms, floor.ratio, marker
            ));
        }
    }

    out.push_str("\nRECOMMENDED ACTION:\n");
    out.push_str(&format!("  {}\n", report.verdict.recommended_action));

    out
}

pub fn format_sensitivity_matrix(matrix: &SensitivityMatrix) -> String {
    let mut out = String::new();
    out.push_str("\n  SENSITIVITY MATRIX\n");
    out.push_str(&"=".repeat(80));
    out.push('\n');

    out.push_str(&format!(
        "\n{:>20} {:>12} {:>12} {:>16} {:>16}\n",
        "Component", "Multiplier", "Concurrency", "TTFT p99 (ms)", "Throughput (rps)"
    ));
    out.push_str(&"-".repeat(80));
    out.push('\n');

    for row in &matrix.results {
        out.push_str(&format!(
            "{:>20} {:>12.2}x {:>12} {:>16.1} {:>16.1}\n",
            row.perturbation,
            row.multiplier,
            row.concurrency,
            row.predicted_ttft_p99_ms,
            row.predicted_throughput_rps,
        ));
    }
    out
}

pub fn format_diff(diff: &DiffReport) -> String {
    let mut out = String::new();
    out.push_str("\n  COMPASS DIFF REPORT\n");
    out.push_str(&"=".repeat(70));
    out.push_str(&format!(
        "\n  A: {}  vs  B: {}\n",
        diff.report_a, diff.report_b
    ));
    let verdict_status = if diff.verdict_changed {
        "CHANGED"
    } else {
        "unchanged"
    };
    out.push_str(&format!("  Verdict: {}\n\n", verdict_status));

    out.push_str(&format!(
        "{:>25} {:>12} {:>12} {:>10} {:>12} {:>12} {:>10}\n",
        "Component", "Attr% A", "Attr% B", "Delta%", "p99 A(ms)", "p99 B(ms)", "Delta(ms)"
    ));
    out.push_str(&"-".repeat(95));
    out.push('\n');

    for delta in &diff.component_deltas {
        let attr_arrow = if delta.delta_pct > 1.0 {
            "+"
        } else if delta.delta_pct < -1.0 {
            ""
        } else {
            " "
        };
        out.push_str(&format!(
            "{:>25} {:>12.1} {:>12.1} {:>9}{:.1} {:>12.2} {:>12.2} {:>+10.2}\n",
            delta.component,
            delta.attribution_pct_a,
            delta.attribution_pct_b,
            attr_arrow,
            delta.delta_pct,
            delta.latency_p99_a,
            delta.latency_p99_b,
            delta.latency_delta_ms,
        ));
    }
    out
}

pub fn format_calibration(result: &CalibrationResult) -> String {
    let mut out = String::new();
    out.push_str("\n  CALIBRATION RESULT\n");
    out.push_str(&"=".repeat(50));
    out.push_str(&format!("\n  Trace: {}\n", result.trace_source));
    out.push_str(&format!(
        "  Mocker TTFT p99:  {:.1} ms\n",
        result.mocker_ttft_p99_ms
    ));
    out.push_str(&format!(
        "  Real TTFT p99:    {:.1} ms\n",
        result.real_ttft_p99_ms
    ));
    out.push_str(&format!(
        "  Residual:         {:.1}%\n",
        result.residual_pct
    ));
    out.push_str(&format!(
        "  Threshold:        {:.1}%\n",
        result.threshold_pct
    ));
    let status = if result.is_calibrated {
        "CALIBRATED - mocker predictions are trustworthy"
    } else {
        "NOT CALIBRATED - mocker predictions may be inaccurate"
    };
    out.push_str(&format!("  Status:           {}\n", status));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;

    #[test]
    fn test_human_report_contains_verdict() {
        let report = AttributionReport {
            compass_version: "1.0.0".to_string(),
            window: TimeWindow {
                start: Utc::now(),
                end: Utc::now(),
            },
            deployment: DeploymentInfo {
                name: "test-deploy".to_string(),
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
                evidence: vec![],
                recommended_action: "investigate radix tree lock contention".to_string(),
            },
            per_component: vec![ComponentContribution {
                component: "kvbm".to_string(),
                sub_component: Some("allocate".to_string()),
                contribution_pct: 47.2,
                latency_ms: PercentileSet {
                    p50: 2.0,
                    p95: 3.0,
                    p99: 3.51,
                },
                score: 0.0,
            }],
            saturation: vec![SaturationEntry {
                component: "kvbm.radix_tree_lock".to_string(),
                utilization: 0.91,
                queue_trend: QueueTrend::Growing,
                warning: true,
            }],
            sub_component_breakdown: vec![],
            floor_checks: vec![FloorCheck {
                component: "kvbm.allocate".to_string(),
                observed_ms: 3.51,
                floor_ms: 0.80,
                ratio: 4.39,
                is_optimization_candidate: true,
            }],
        };

        let output = format_report(&report);
        assert!(output.contains("VERDICT"));
        assert!(output.contains("kvbm.allocate"));
        assert!(output.contains("47.2%"));
        assert!(output.contains("[HIGH]"));
        assert!(output.contains("radix tree lock contention"));
    }
}

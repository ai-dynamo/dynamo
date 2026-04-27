// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::types::{ComponentContribution, PercentileSet};

#[derive(Debug, Clone)]
pub struct CriticalPathResult {
    pub components: Vec<ComponentContribution>,
    pub total_latency_ms: PercentileSet,
}

#[derive(Debug, Clone)]
pub struct SpanNode {
    pub component: String,
    pub sub_component: Option<String>,
    pub duration_ms: f64,
    pub children: Vec<SpanNode>,
}

pub fn analyze_critical_path(traces: &[Vec<SpanNode>]) -> CriticalPathResult {
    if traces.is_empty() {
        return CriticalPathResult {
            components: vec![],
            total_latency_ms: PercentileSet {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            },
        };
    }

    let mut component_latencies: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    let mut total_latencies: Vec<f64> = Vec::with_capacity(traces.len());

    for trace in traces {
        let mut request_total = 0.0;
        for root_span in trace {
            let path = find_critical_path(root_span);
            for (component_key, duration) in &path {
                component_latencies
                    .entry(component_key.clone())
                    .or_default()
                    .push(*duration);
                request_total += duration;
            }
        }
        total_latencies.push(request_total);
    }

    let total_latency_ms = PercentileSet::from_samples(&mut total_latencies);

    let p99_total = total_latency_ms.p99;
    let mut components: Vec<ComponentContribution> = component_latencies
        .into_iter()
        .map(|(key, mut latencies)| {
            let ps = PercentileSet::from_samples(&mut latencies);
            let contribution_pct = if p99_total > 0.0 {
                (ps.p99 / p99_total) * 100.0
            } else {
                0.0
            };
            let parts: Vec<&str> = key.splitn(2, '.').collect();
            ComponentContribution {
                component: parts[0].to_string(),
                sub_component: parts.get(1).map(|s| s.to_string()),
                contribution_pct,
                latency_ms: ps,
                score: 0.0,
            }
        })
        .collect();

    components.sort_by(|a, b| {
        b.contribution_pct
            .partial_cmp(&a.contribution_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    CriticalPathResult {
        components,
        total_latency_ms,
    }
}

fn find_critical_path(node: &SpanNode) -> Vec<(String, f64)> {
    let key = match &node.sub_component {
        Some(sub) => format!("{}.{}", node.component, sub),
        None => node.component.clone(),
    };

    if node.children.is_empty() {
        return vec![(key, node.duration_ms)];
    }

    let mut longest_path: Vec<(String, f64)> = vec![];
    let mut longest_duration = 0.0_f64;

    for child in &node.children {
        let child_path = find_critical_path(child);
        let child_total: f64 = child_path.iter().map(|(_, d)| d).sum();
        if child_total > longest_duration {
            longest_duration = child_total;
            longest_path = child_path;
        }
    }

    let mut result = vec![(key, node.duration_ms)];
    result.extend(longest_path);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span(component: &str, sub: Option<&str>, dur: f64, children: Vec<SpanNode>) -> SpanNode {
        SpanNode {
            component: component.to_string(),
            sub_component: sub.map(String::from),
            duration_ms: dur,
            children,
        }
    }

    #[test]
    fn test_single_span_critical_path() {
        let traces = vec![vec![make_span("prefill", Some("compute"), 12.0, vec![])]];
        let result = analyze_critical_path(&traces);
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.components[0].component, "prefill");
        assert!((result.components[0].contribution_pct - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_component_critical_path() {
        let traces = vec![vec![make_span(
            "request",
            None,
            1.0,
            vec![
                make_span("router", Some("dispatch"), 0.5, vec![]),
                make_span("kvbm", Some("allocate"), 3.5, vec![]),
                make_span("prefill", Some("compute"), 12.0, vec![]),
            ],
        )]];
        let result = analyze_critical_path(&traces);
        assert!(!result.components.is_empty());
        let prefill = result
            .components
            .iter()
            .find(|c| c.component == "prefill")
            .unwrap();
        assert!(prefill.contribution_pct > 50.0);
    }

    #[test]
    fn test_empty_traces() {
        let result = analyze_critical_path(&[]);
        assert!(result.components.is_empty());
        assert_eq!(result.total_latency_ms.p50, 0.0);
    }
}

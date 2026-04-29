// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Merger: reads per-component `.pftrace.gz` files, builds per-request
//! happens-before DAGs, computes critical paths, and produces merged output.
//!
//! The merger is the core of `dynamo-sysprofile-merge`. It runs either as a
//! K8s Job (`--in-cluster`) or locally (`dynamo sysprofile merge ./run-dir`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::reader::{self, Slice};

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-request critical-path result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestCriticalPath {
    pub traceparent: String,
    pub total_duration_ns: u64,
    pub total_duration_ms: f64,
    pub stages: Vec<CriticalPathStage>,
}

/// One stage's contribution to a request's critical path.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CriticalPathStage {
    pub stage: String,
    pub component: String,
    pub duration_ns: u64,
    pub duration_ms: f64,
    pub fraction: f64,
}

/// Attribution of a stage to the p99 critical path.
#[derive(Debug, Clone, serde::Serialize)]
pub struct P99Attribution {
    pub stage: String,
    pub duration_ms: f64,
    pub fraction: f64,
}

/// Edge in the causality DAG with weight = critical-path frequency.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CausalityEdge {
    pub from_stage: String,
    pub to_stage: String,
    pub weight: f64,
    pub count: u32,
}

/// Complete merge result containing everything the report needs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MergeResult {
    pub run_id: String,
    pub capture_window_start_ns: u64,
    pub capture_window_end_ns: u64,
    pub capture_duration_ms: f64,
    pub total_requests: usize,
    pub total_slices: usize,
    pub components: Vec<ComponentSummary>,
    pub p99_critical_path: Vec<P99Attribution>,
    pub p99_total_ms: f64,
    pub p50_total_ms: f64,
    pub top_slow_requests: Vec<RequestCriticalPath>,
    pub per_request_paths: Vec<RequestCriticalPath>,
    pub causality_edges: Vec<CausalityEdge>,
    pub component_utilization: Vec<ComponentUtilization>,
    pub clock_alignment: ClockAlignment,
}

/// Summary info for one component.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComponentSummary {
    pub name: String,
    pub host: String,
    pub slice_count: usize,
    pub total_busy_ns: u64,
}

/// Utilization data for View A (heat-strip).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComponentUtilization {
    pub component: String,
    pub host: String,
    pub bins: Vec<UtilizationBin>,
    pub overall_utilization: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UtilizationBin {
    pub start_ms: f64,
    pub end_ms: f64,
    pub utilization: f64,
}

/// Clock alignment metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClockAlignment {
    pub method: String,
    pub max_residual_ns: u64,
    pub per_host: Vec<HostClockInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HostClockInfo {
    pub host: String,
    pub offset_ns: i64,
    pub residual_ns: u64,
}

// ── Merger implementation ─────────────────────────────────────────────────────

/// Read all trace files in a run directory and produce a complete merge result.
pub fn merge_run(run_dir: &Path) -> anyhow::Result<MergeResult> {
    let trace_files = discover_trace_files(run_dir)?;
    if trace_files.is_empty() {
        anyhow::bail!("no .pftrace.gz files found in {}", run_dir.display());
    }

    let mut all_slices: Vec<Slice> = Vec::new();
    let mut all_instants = Vec::new();
    let mut components: Vec<ComponentSummary> = Vec::new();

    for path in &trace_files {
        let parsed = reader::read_trace(path)?;
        let comp_name = component_from_filename(&parsed.source_file);
        let host = host_from_process_name(parsed.slices.first());

        let total_busy: u64 = parsed.slices.iter().map(|s| s.duration_ns).sum();
        components.push(ComponentSummary {
            name: comp_name,
            host,
            slice_count: parsed.slices.len(),
            total_busy_ns: total_busy,
        });

        all_slices.extend(parsed.slices);
        all_instants.extend(parsed.instants);
    }

    let total_slices = all_slices.len();

    // Determine capture window
    let window_start = all_slices.iter().map(|s| s.start_ns).min().unwrap_or(0);
    let window_end = all_slices.iter().map(|s| s.end_ns).max().unwrap_or(0);
    let capture_duration_ns = window_end.saturating_sub(window_start);

    // Group slices by traceparent
    let mut by_request: HashMap<String, Vec<&Slice>> = HashMap::new();
    for slice in &all_slices {
        if !slice.traceparent.is_empty() {
            by_request
                .entry(slice.traceparent.clone())
                .or_default()
                .push(slice);
        }
    }

    let total_requests = by_request.len();

    // Build critical path per request
    let mut per_request_paths: Vec<RequestCriticalPath> = Vec::new();
    let mut edge_counts: HashMap<(String, String), u32> = HashMap::new();

    for (_tp, slices) in &by_request {
        if let Some(path) = compute_critical_path(slices) {
            // Count edge frequencies for causality DAG
            for w in path.stages.windows(2) {
                let key = (w[0].stage.clone(), w[1].stage.clone());
                *edge_counts.entry(key).or_insert(0) += 1;
            }
            per_request_paths.push(path);
        }
    }

    // Sort ascending for percentile computation, then reverse for top-K
    per_request_paths.sort_by_key(|p| p.total_duration_ns);

    // Compute p99 and p50 (ascending order: index 0 = fastest)
    let n = per_request_paths.len();
    let p99_idx = ((n as f64) * 0.99).ceil() as usize;
    let p99_idx = p99_idx.min(n).saturating_sub(1);
    let p50_idx = n / 2;

    let p99_path = per_request_paths.get(p99_idx);
    let p50_total_ms = per_request_paths
        .get(p50_idx)
        .map(|p| p.total_duration_ms)
        .unwrap_or(0.0);

    let p99_critical_path: Vec<P99Attribution> = p99_path
        .map(|p| {
            p.stages
                .iter()
                .map(|s| P99Attribution {
                    stage: s.stage.clone(),
                    duration_ms: s.duration_ms,
                    fraction: s.fraction,
                })
                .collect()
        })
        .unwrap_or_default();

    let p99_total_ms = p99_path.map(|p| p.total_duration_ms).unwrap_or(0.0);

    // Top-10 slow requests (take from the end since sorted ascending)
    let top_slow_requests: Vec<RequestCriticalPath> =
        per_request_paths.iter().rev().take(10).cloned().collect();

    // Build causality edges
    let total_paths = per_request_paths.len().max(1) as f64;
    let mut causality_edges: Vec<CausalityEdge> = edge_counts
        .into_iter()
        .map(|((from, to), count)| CausalityEdge {
            from_stage: from,
            to_stage: to,
            weight: count as f64 / total_paths,
            count,
        })
        .collect();
    causality_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));

    // Compute component utilization (View A)
    let component_utilization =
        compute_utilization(&all_slices, window_start, window_end, capture_duration_ns);

    // Extract run_id
    let run_id = all_slices
        .first()
        .map(|s| s.run_id.clone())
        .unwrap_or_else(|| {
            run_dir
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default()
        });

    Ok(MergeResult {
        run_id,
        capture_window_start_ns: window_start,
        capture_window_end_ns: window_end,
        capture_duration_ms: capture_duration_ns as f64 / 1_000_000.0,
        total_requests,
        total_slices,
        components,
        p99_critical_path,
        p99_total_ms,
        p50_total_ms,
        top_slow_requests,
        per_request_paths,
        causality_edges,
        component_utilization,
        clock_alignment: ClockAlignment {
            method: "single-host".into(),
            max_residual_ns: 0,
            per_host: vec![],
        },
    })
}

// ── Critical-path computation ─────────────────────────────────────────────────

/// Build a happens-before DAG for one request's slices and find the longest path.
fn compute_critical_path(slices: &[&Slice]) -> Option<RequestCriticalPath> {
    if slices.is_empty() {
        return None;
    }

    // Sort by start time
    let mut sorted: Vec<&Slice> = slices.to_vec();
    sorted.sort_by_key(|s| s.start_ns);

    let traceparent = sorted[0].traceparent.clone();

    // Build DAG: each slice is a node, edges from earlier to later slices
    // that represent causal dependencies.
    let mut graph = DiGraph::<usize, u64>::new();
    let mut nodes: Vec<NodeIndex> = Vec::new();

    for (i, _) in sorted.iter().enumerate() {
        nodes.push(graph.add_node(i));
    }

    // Add edges based on causal ordering:
    // 1. Nesting: if B is inside A (same start, earlier end), A → B
    // 2. Sequential same-component: B starts after A ends within 50ms
    // 3. Cross-component handoff: B starts after A ends on a different component
    //    (closest predecessor per component to avoid redundant edges)
    for i in 0..sorted.len() {
        for j in (i + 1)..sorted.len() {
            let a = sorted[i];
            let b = sorted[j];

            // Nesting: B starts after A starts and B ends before A ends
            if b.start_ns >= a.start_ns && b.end_ns <= a.end_ns && a.stage != b.stage {
                graph.add_edge(nodes[i], nodes[j], b.start_ns - a.start_ns);
                continue;
            }

            // B starts after A ends (causal ordering)
            if b.start_ns >= a.end_ns {
                let gap = b.start_ns - a.end_ns;

                if same_component(&a.stage, &b.stage) {
                    // Sequential on same component within 50ms
                    if gap < 50_000_000 {
                        graph.add_edge(nodes[i], nodes[j], a.duration_ns);
                    }
                } else if gap < 10_000_000 {
                    // Cross-component handoff within 10ms
                    graph.add_edge(nodes[i], nodes[j], a.duration_ns);
                }
            }
        }
    }

    // Find longest path using topological order (DAG guaranteed)
    let topo = match petgraph::algo::toposort(&graph, None) {
        Ok(t) => t,
        Err(_) => return None, // cycle — shouldn't happen
    };

    let mut dist: HashMap<NodeIndex, u64> = HashMap::new();
    let mut pred: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for &node in &topo {
        let slice_dur = sorted[*graph.node_weight(node).unwrap()].duration_ns;
        let mut best_dist = slice_dur;
        let mut best_pred = None;

        for edge in graph.edges_directed(node, petgraph::Direction::Incoming) {
            let src_dist = dist.get(&edge.source()).copied().unwrap_or(0);
            let candidate = src_dist + slice_dur;
            if candidate > best_dist {
                best_dist = candidate;
                best_pred = Some(edge.source());
            }
        }

        dist.insert(node, best_dist);
        if let Some(p) = best_pred {
            pred.insert(node, p);
        }
    }

    // Find the node with maximum distance (end of critical path)
    let &end_node = dist.iter().max_by_key(|&(_, &d)| d).map(|(n, _)| n)?;
    let total_duration_ns = dist[&end_node];

    // Trace back to build the path
    let mut path_nodes = vec![end_node];
    let mut current = end_node;
    while let Some(&p) = pred.get(&current) {
        path_nodes.push(p);
        current = p;
    }
    path_nodes.reverse();

    // Aggregate by stage name
    let mut stage_durations: Vec<(String, String, u64)> = Vec::new();
    for &node in &path_nodes {
        let idx = *graph.node_weight(node).unwrap();
        let slice = sorted[idx];
        let component = component_from_stage(&slice.stage);
        stage_durations.push((slice.stage.clone(), component, slice.duration_ns));
    }

    // Merge adjacent same-stage entries
    let mut merged_stages: Vec<CriticalPathStage> = Vec::new();
    for (stage, component, dur) in &stage_durations {
        if let Some(last) = merged_stages.last_mut() {
            if last.stage == *stage {
                last.duration_ns += dur;
                last.duration_ms = last.duration_ns as f64 / 1_000_000.0;
                continue;
            }
        }
        merged_stages.push(CriticalPathStage {
            stage: stage.clone(),
            component: component.clone(),
            duration_ns: *dur,
            duration_ms: *dur as f64 / 1_000_000.0,
            fraction: 0.0,
        });
    }

    // Compute fractions
    let total = total_duration_ns.max(1) as f64;
    for stage in &mut merged_stages {
        stage.fraction = stage.duration_ns as f64 / total;
    }

    Some(RequestCriticalPath {
        traceparent,
        total_duration_ns,
        total_duration_ms: total_duration_ns as f64 / 1_000_000.0,
        stages: merged_stages,
    })
}

// ── Utilization computation (View A) ──────────────────────────────────────────

fn compute_utilization(
    slices: &[Slice],
    window_start: u64,
    _window_end: u64,
    window_duration: u64,
) -> Vec<ComponentUtilization> {
    if window_duration == 0 {
        return vec![];
    }

    // Group by (component, host)
    let mut by_component: HashMap<(String, String), Vec<&Slice>> = HashMap::new();
    for slice in slices {
        let comp = component_from_stage(&slice.stage);
        let host = host_from_process(&slice.process_name);
        by_component
            .entry((comp, host))
            .or_default()
            .push(slice);
    }

    let num_bins = 50;
    let bin_width_ns = window_duration / num_bins as u64;

    let mut result = Vec::new();
    for ((component, host), comp_slices) in &by_component {
        let mut bins = Vec::new();
        let mut total_busy: u64 = 0;

        for bin_idx in 0..num_bins {
            let bin_start = window_start + bin_idx as u64 * bin_width_ns;
            let bin_end = bin_start + bin_width_ns;

            let mut busy_ns: u64 = 0;
            for slice in comp_slices {
                let overlap_start = slice.start_ns.max(bin_start);
                let overlap_end = slice.end_ns.min(bin_end);
                if overlap_start < overlap_end {
                    busy_ns += overlap_end - overlap_start;
                }
            }

            let utilization = busy_ns as f64 / bin_width_ns as f64;
            total_busy += busy_ns;

            bins.push(UtilizationBin {
                start_ms: (bin_start - window_start) as f64 / 1_000_000.0,
                end_ms: (bin_end - window_start) as f64 / 1_000_000.0,
                utilization: utilization.min(1.0),
            });
        }

        result.push(ComponentUtilization {
            component: component.clone(),
            host: host.clone(),
            bins,
            overall_utilization: total_busy as f64 / window_duration as f64,
        });
    }

    result.sort_by(|a, b| a.component.cmp(&b.component));
    result
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn discover_trace_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gz")
                && path.to_string_lossy().contains("pftrace")
            {
                files.push(path);
            } else if path.is_dir() {
                // Recurse one level for host subdirectories
                files.extend(discover_trace_files(&path)?);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn component_from_filename(filename: &str) -> String {
    filename
        .trim_end_matches(".pftrace.gz")
        .trim_end_matches(".pftrace")
        .to_string()
}

fn component_from_stage(stage: &str) -> String {
    // "dynamo.frontend.recv" -> "frontend"
    let parts: Vec<&str> = stage.split('.').collect();
    if parts.len() >= 2 {
        parts[1].to_string()
    } else {
        stage.to_string()
    }
}

fn host_from_process_name(slice: Option<&Slice>) -> String {
    slice
        .map(|s| host_from_process(&s.process_name))
        .unwrap_or_else(|| "unknown".into())
}

fn host_from_process(process_name: &str) -> String {
    // "engine-prefill-0@node-0" -> "node-0"
    process_name
        .rsplit('@')
        .next()
        .unwrap_or("unknown")
        .to_string()
}

fn same_component(a: &str, b: &str) -> bool {
    component_from_stage(a) == component_from_stage(b)
}

#[allow(dead_code)]
fn is_transport_pair(a: &str, b: &str) -> bool {
    (a.contains("transport.send") && b.contains("transport.recv"))
        || (a.contains(".send") && b.contains(".recv") && !same_component(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slice(stage: &str, start_ms: u64, dur_ms: u64, tp: &str, process: &str) -> Slice {
        let start_ns = start_ms * 1_000_000;
        let dur_ns = dur_ms * 1_000_000;
        Slice {
            stage: stage.into(),
            traceparent: tp.into(),
            run_id: "test-run".into(),
            start_ns,
            end_ns: start_ns + dur_ns,
            duration_ns: dur_ns,
            track_uuid: 1,
            process_name: process.into(),
            thread_name: "main".into(),
            pid: 1,
            tid: 1,
        }
    }

    #[test]
    fn critical_path_simple() {
        let tp = "00-abc-def-01";
        let slices = vec![
            make_slice("dynamo.frontend.recv", 0, 50, tp, "frontend@node-0"),
            make_slice("dynamo.frontend.preprocess", 5, 10, tp, "frontend@node-0"),
            make_slice("dynamo.router.schedule", 55, 20, tp, "router@node-0"),
            make_slice("dynamo.prefill.compute", 80, 100, tp, "prefill@node-0"),
        ];

        let refs: Vec<&Slice> = slices.iter().collect();
        let path = compute_critical_path(&refs).unwrap();

        assert!(!path.stages.is_empty());
        assert!(path.total_duration_ms > 0.0);

        // The longest stage should be prefill.compute at 100ms
        let longest = path.stages.iter().max_by_key(|s| s.duration_ns).unwrap();
        assert_eq!(longest.stage, "dynamo.prefill.compute");
    }

    #[test]
    fn component_extraction() {
        assert_eq!(component_from_stage("dynamo.frontend.recv"), "frontend");
        assert_eq!(component_from_stage("dynamo.router.schedule"), "router");
        assert_eq!(component_from_stage("dynamo.prefill.compute"), "prefill");
        assert_eq!(component_from_stage("dynamo.decode.first_token"), "decode");
    }
}

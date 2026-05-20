// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lower a [`LoadedAgentTrace`] to agentic Mooncake rows.
//!
//! Beyond the per-request row data carried by [`super::mooncake`], this module
//! infers the workflow DAG (sequential same-trajectory waits, subagent
//! spawn/join edges, prefix-reset markers) and attributes harness tool spans to
//! the LLM request that consumed them. It also produces the convert-time
//! [`ToolSummary`] that the binary prints after writing the JSONL.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, anyhow, bail};
use dynamo_data_gen::{AgenticMooncakeRow, AgenticToolEvent, RollingHashIdMapper};

use super::load::{LoadedAgentTrace, RequestEntry, ToolEntry};

/// Convert an agent trace into agentic Mooncake rows, inferring trajectory
/// dependencies and attributing tool spans.
pub fn build_agentic_mooncake_rows(
    mut loaded: LoadedAgentTrace,
) -> Result<(usize, Vec<AgenticMooncakeRow>)> {
    let global_start_ms = loaded
        .requests
        .iter()
        .map(|request| request.start_ms)
        .min()
        .ok_or_else(|| anyhow!("no request records to convert"))?;
    let trace_block_size = loaded.requests[0].replay.trace_block_size;
    for request in &loaded.requests {
        if request.replay.trace_block_size != trace_block_size {
            bail!(
                "mixed replay trace_block_size values are not supported: {} and {}",
                trace_block_size,
                request.replay.trace_block_size
            );
        }
    }

    loaded.requests.sort_by(|left, right| {
        (left.start_ms, left.end_ms, &left.request.request_id).cmp(&(
            right.start_ms,
            right.end_ms,
            &right.request.request_id,
        ))
    });

    let mut id_to_index = HashMap::new();
    for (idx, request) in loaded.requests.iter().enumerate() {
        if id_to_index
            .insert(request.request.request_id.clone(), idx)
            .is_some()
        {
            bail!("duplicate request_id {}", request.request.request_id);
        }
    }

    let mut trajectory_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
    let mut parent_by_trajectory: HashMap<String, String> = HashMap::new();
    for (idx, request) in loaded.requests.iter().enumerate() {
        let trajectory_id = trajectory_id_for(request);
        trajectory_to_indices
            .entry(trajectory_id.clone())
            .or_default()
            .push(idx);
        if let Some(parent) = request
            .agent_context
            .as_ref()
            .and_then(|context| context.parent_trajectory_id.clone())
        {
            match parent_by_trajectory.get(&trajectory_id) {
                Some(existing) if existing != &parent => {
                    bail!(
                        "trajectory {} has conflicting parent_trajectory_id values: {} and {}",
                        trajectory_id,
                        existing,
                        parent
                    );
                }
                Some(_) => {}
                None => {
                    parent_by_trajectory.insert(trajectory_id, parent);
                }
            }
        }
    }
    for indices in trajectory_to_indices.values_mut() {
        indices.sort_by_key(|idx| {
            let request = &loaded.requests[*idx];
            (
                request.start_ms,
                request.end_ms,
                request.request.request_id.clone(),
            )
        });
    }

    let mut wait_for: Vec<Vec<String>> = vec![Vec::new(); loaded.requests.len()];
    let mut branches: Vec<Vec<String>> = vec![Vec::new(); loaded.requests.len()];
    let mut prefix_reset = vec![false; loaded.requests.len()];

    for indices in trajectory_to_indices.values() {
        for (pos, idx) in indices.iter().copied().enumerate() {
            prefix_reset[idx] = pos == 0;
            if pos > 0 {
                let previous = &loaded.requests[indices[pos - 1]].request.request_id;
                push_unique(&mut wait_for[idx], previous.clone());
            }
        }
    }

    for (trajectory_id, parent_id) in &parent_by_trajectory {
        let Some(child_indices) = trajectory_to_indices.get(trajectory_id) else {
            continue;
        };
        let Some(parent_indices) = trajectory_to_indices.get(parent_id) else {
            continue;
        };
        let first_child_idx = child_indices[0];
        let last_finishing_child_idx = *child_indices
            .iter()
            .max_by(|left, right| {
                let left_request = &loaded.requests[**left];
                let right_request = &loaded.requests[**right];
                (
                    left_request.end_ms,
                    left_request.start_ms,
                    &left_request.request.request_id,
                )
                    .cmp(&(
                        right_request.end_ms,
                        right_request.start_ms,
                        &right_request.request.request_id,
                    ))
            })
            .expect("child trajectory is non-empty");
        let child_start_ms = loaded.requests[first_child_idx].start_ms;
        let child_end_ms = loaded.requests[last_finishing_child_idx].end_ms;

        if let Some(parent_spawn_idx) =
            latest_request_ending_before(&loaded.requests, parent_indices, child_start_ms)
        {
            let parent_request_id = loaded.requests[parent_spawn_idx].request.request_id.clone();
            push_unique(&mut wait_for[first_child_idx], parent_request_id);
            let child_request_id = loaded.requests[first_child_idx].request.request_id.clone();
            push_unique(&mut branches[parent_spawn_idx], child_request_id);
        }

        if let Some(parent_join_idx) =
            first_request_starting_after(&loaded.requests, parent_indices, child_end_ms)
        {
            let child_request_id = loaded.requests[last_finishing_child_idx]
                .request
                .request_id
                .clone();
            push_unique(&mut wait_for[parent_join_idx], child_request_id);
        }
    }

    let mut tools_by_trajectory: HashMap<String, Vec<ToolEntry>> = HashMap::new();
    for tool in loaded.tools {
        tools_by_trajectory
            .entry(tool.trajectory_id.clone())
            .or_default()
            .push(tool);
    }
    for tools in tools_by_trajectory.values_mut() {
        tools.sort_by_key(|tool| (tool.start_ms, tool.end_ms));
    }

    let mut mapper = RollingHashIdMapper::new(trace_block_size);
    let mut rows = Vec::with_capacity(loaded.requests.len());
    for (idx, request) in loaded.requests.iter().enumerate() {
        let hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        let trajectory_id = trajectory_id_for(request);
        let dep_end_ms = wait_for[idx]
            .iter()
            .filter_map(|dependency| id_to_index.get(dependency))
            .map(|dep_idx| loaded.requests[*dep_idx].end_ms)
            .max();
        let (delay, tool_wait_ms, tool_events) = if let Some(dep_end_ms) = dep_end_ms {
            let observed_gap_ms = request.start_ms.saturating_sub(dep_end_ms).max(0) as f64;
            let (raw_tool_wait_ms, contributing) = tools_by_trajectory
                .get(&trajectory_id)
                .map(|tools| collect_tools_in_window(tools, dep_end_ms, request.start_ms))
                .unwrap_or_else(|| (0.0, Vec::new()));
            let tool_wait_ms = raw_tool_wait_ms.min(observed_gap_ms);
            let non_tool_wait_ms = (observed_gap_ms - tool_wait_ms).max(0.0);
            let events = contributing
                .into_iter()
                .map(tool_entry_to_event)
                .collect::<Vec<_>>();
            (
                Some(non_tool_wait_ms),
                (tool_wait_ms > 0.0).then_some(tool_wait_ms),
                events,
            )
        } else {
            (None, None, Vec::new())
        };

        rows.push(AgenticMooncakeRow {
            request_id: request.request.request_id.clone(),
            session_id: Some(trajectory_id),
            input_length: Some(request.replay.input_length),
            output_length: Some(
                usize::try_from(output_length).context("output length does not fit in usize")?,
            ),
            hash_ids: Some(hash_ids),
            timestamp: Some((request.start_ms - global_start_ms) as f64),
            delay,
            wait_for: wait_for[idx].clone(),
            branches: branches[idx].clone(),
            prefix_reset: Some(prefix_reset[idx]),
            tool_wait_ms,
            tool_events,
        });
    }

    Ok((trace_block_size, rows))
}

fn trajectory_id_for(request: &RequestEntry) -> String {
    request
        .agent_context
        .as_ref()
        .map(|context| context.trajectory_id.clone())
        .unwrap_or_else(|| request.request.request_id.clone())
}

fn latest_request_ending_before(
    requests: &[RequestEntry],
    indices: &[usize],
    timestamp_ms: i64,
) -> Option<usize> {
    indices
        .iter()
        .copied()
        .filter(|idx| requests[*idx].end_ms <= timestamp_ms)
        .max_by_key(|idx| requests[*idx].end_ms)
}

fn first_request_starting_after(
    requests: &[RequestEntry],
    indices: &[usize],
    timestamp_ms: i64,
) -> Option<usize> {
    indices
        .iter()
        .copied()
        .filter(|idx| requests[*idx].start_ms >= timestamp_ms)
        .min_by_key(|idx| requests[*idx].start_ms)
}

/// Return the parallel-aware tool wall-time over `[start_ms, end_ms]` along
/// with the originating tool entries whose intervals contribute to it.
///
/// The wall-time is computed by clipping each tool span to the window and
/// sweep-merging overlapping intervals -- two tools that ran concurrently
/// contribute the union of their durations, not the sum. The returned entries
/// are the full, unclipped originals so callers can preserve per-tool metadata
/// without losing the original event times.
fn collect_tools_in_window(
    tools: &[ToolEntry],
    start_ms: i64,
    end_ms: i64,
) -> (f64, Vec<&ToolEntry>) {
    if end_ms <= start_ms {
        return (0.0, Vec::new());
    }

    let mut contributing: Vec<&ToolEntry> = Vec::new();
    let mut intervals = Vec::new();
    for tool in tools {
        let clipped_start = tool.start_ms.max(start_ms);
        let clipped_end = tool.end_ms.min(end_ms);
        if clipped_end > clipped_start {
            contributing.push(tool);
            intervals.push((clipped_start, clipped_end));
        }
    }
    intervals.sort_unstable();

    let mut total = 0_i64;
    let mut current: Option<(i64, i64)> = None;
    for (start, end) in intervals {
        match current {
            None => current = Some((start, end)),
            Some((current_start, current_end)) if start <= current_end => {
                current = Some((current_start, current_end.max(end)));
            }
            Some((current_start, current_end)) => {
                total += current_end - current_start;
                current = Some((start, end));
            }
        }
    }
    if let Some((current_start, current_end)) = current {
        total += current_end - current_start;
    }
    (total as f64, contributing)
}

fn tool_entry_to_event(entry: &ToolEntry) -> AgenticToolEvent {
    AgenticToolEvent {
        tool_call_id: entry.tool_call_id.clone(),
        tool_class: entry.tool_class.clone(),
        started_at_unix_ms: entry.start_ms.max(0) as u64,
        ended_at_unix_ms: entry.end_ms.max(0) as u64,
        duration_ms: entry.duration_ms,
        status: entry.status.clone(),
        output_bytes: entry.output_bytes,
        output_tokens: entry.output_tokens,
        error_type: entry.error_type.clone(),
    }
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

/// Aggregate tool-event stats printed by the converter after writing the JSONL.
#[derive(Debug, Default)]
pub struct ToolSummary {
    pub total_spans: usize,
    pub trajectories: usize,
    pub by_status: BTreeMap<String, usize>,
    pub by_class: BTreeMap<String, usize>,
    pub total_wall_ms: f64,
}

impl ToolSummary {
    /// Render the summary as the multi-line block printed after conversion.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Tool events: {} spans across {} trajectories\n",
            self.total_spans, self.trajectories
        ));
        out.push_str(&format!(
            "  Status:    {}\n",
            format_counts(&self.by_status)
        ));
        out.push_str(&format!("  Class:     {}\n", format_counts(&self.by_class)));
        out.push_str(&format!(
            "  Wall-time: {:.2}s\n",
            self.total_wall_ms / 1000.0,
        ));
        out
    }
}

fn format_counts(counts: &BTreeMap<String, usize>) -> String {
    if counts.is_empty() {
        return "(none)".to_string();
    }
    let mut pairs: Vec<(&String, &usize)> = counts.iter().collect();
    pairs.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));
    pairs
        .iter()
        .map(|(name, count)| format!("{name}={count}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Aggregate counts and total tool wall-time across all loaded tool entries.
pub fn summarize_tools(tools: &[ToolEntry]) -> ToolSummary {
    let mut summary = ToolSummary::default();
    if tools.is_empty() {
        return summary;
    }
    summary.total_spans = tools.len();

    let mut trajectories: HashMap<&str, ()> = HashMap::new();
    for tool in tools {
        let status_key = if tool.status.is_empty() {
            tool.terminal_event.clone()
        } else {
            tool.status.clone()
        };
        *summary.by_status.entry(status_key).or_insert(0) += 1;
        let class_key = if tool.tool_class.is_empty() {
            "(unlabeled)".to_string()
        } else {
            tool.tool_class.clone()
        };
        *summary.by_class.entry(class_key).or_insert(0) += 1;
        summary.total_wall_ms += tool.duration_ms;
        trajectories.insert(tool.trajectory_id.as_str(), ());
    }
    summary.trajectories = trajectories.len();
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_trace::load::{
        AgentContextFields, AgentReplayMetrics, AgentRequestMetrics, RequestEntry, ToolEntry,
    };

    fn request(
        request_id: &str,
        start_ms: i64,
        end_ms: i64,
        sequence_hashes: Vec<u64>,
    ) -> RequestEntry {
        RequestEntry {
            start_ms,
            end_ms,
            agent_context: None,
            request: AgentRequestMetrics {
                request_id: request_id.to_string(),
                output_tokens: Some(5),
                request_received_ms: Some(start_ms as u64),
                total_time_ms: Some((end_ms - start_ms) as f64),
                replay: None,
            },
            replay: AgentReplayMetrics {
                trace_block_size: 2,
                input_length: sequence_hashes.len() * 2,
                input_sequence_hashes: sequence_hashes,
            },
        }
    }

    fn contextual_request(
        request_id: &str,
        trajectory_id: &str,
        parent_trajectory_id: Option<&str>,
        start_ms: i64,
        end_ms: i64,
        sequence_hashes: Vec<u64>,
    ) -> RequestEntry {
        let mut entry = request(request_id, start_ms, end_ms, sequence_hashes);
        entry.agent_context = Some(AgentContextFields {
            _session_id: "session".to_string(),
            trajectory_id: trajectory_id.to_string(),
            parent_trajectory_id: parent_trajectory_id.map(str::to_string),
        });
        entry
    }

    fn tool(
        trajectory_id: &str,
        tool_call_id: &str,
        tool_class: &str,
        start_ms: i64,
        end_ms: i64,
    ) -> ToolEntry {
        ToolEntry {
            trajectory_id: trajectory_id.to_string(),
            start_ms,
            end_ms,
            tool_call_id: tool_call_id.to_string(),
            tool_class: tool_class.to_string(),
            status: "succeeded".to_string(),
            duration_ms: (end_ms - start_ms).max(0) as f64,
            output_bytes: None,
            output_tokens: None,
            error_type: None,
            terminal_event: "tool_end".to_string(),
        }
    }

    #[test]
    fn agentic_converter_builds_sequential_waits_and_tool_wait_components() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("r1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("r2", "root", None, 1_300, 1_400, vec![11, 22]),
            ],
            tools: vec![tool("root", "call-1", "ls", 1_150, 1_250)],
        };

        let (_, rows) = build_agentic_mooncake_rows(loaded).unwrap();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].wait_for.is_empty());
        assert_eq!(rows[0].prefix_reset, Some(true));
        assert!(rows[0].tool_events.is_empty());
        assert_eq!(rows[1].wait_for, vec!["r1"]);
        assert_eq!(rows[1].delay, Some(100.0));
        assert_eq!(rows[1].tool_wait_ms, Some(100.0));
        assert_eq!(rows[1].dependency_delay_ms(), 200.0);
        assert_eq!(rows[1].tool_events.len(), 1);
        assert_eq!(rows[1].tool_events[0].tool_class, "ls");
        assert_eq!(rows[1].tool_events[0].tool_call_id, "call-1");
        assert_eq!(rows[1].tool_events[0].started_at_unix_ms, 1_150);
        assert_eq!(rows[1].tool_events[0].ended_at_unix_ms, 1_250);
    }

    #[test]
    fn agentic_converter_attaches_parallel_tool_events_with_union_wait() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("r1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("r2", "root", None, 1_400, 1_500, vec![11, 22]),
            ],
            // Two tools that overlap heavily: union is 200ms (1_100..1_300),
            // naive sum would be 350ms.
            tools: vec![
                tool("root", "call-1", "read", 1_100, 1_300),
                tool("root", "call-2", "read", 1_150, 1_250),
                tool("root", "call-3", "find", 1_200, 1_250),
            ],
        };

        let (_, rows) = build_agentic_mooncake_rows(loaded).unwrap();

        assert_eq!(rows[1].tool_wait_ms, Some(200.0));
        assert_eq!(rows[1].tool_events.len(), 3);
        let classes: Vec<_> = rows[1]
            .tool_events
            .iter()
            .map(|event| event.tool_class.as_str())
            .collect();
        assert!(classes.contains(&"read"));
        assert!(classes.contains(&"find"));
    }

    #[test]
    fn agentic_converter_adds_subagent_launch_and_join_dependencies() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("parent-1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("child-1", "child", Some("root"), 1_200, 1_300, vec![33]),
                contextual_request("parent-2", "root", None, 1_500, 1_600, vec![11, 22]),
            ],
            tools: Vec::new(),
        };

        let (_, rows) = build_agentic_mooncake_rows(loaded).unwrap();
        let by_id = rows
            .iter()
            .map(|row| (row.request_id.as_str(), row))
            .collect::<HashMap<_, _>>();

        assert_eq!(by_id["child-1"].wait_for, vec!["parent-1"]);
        assert_eq!(by_id["parent-1"].branches, vec!["child-1"]);
        assert_eq!(by_id["parent-2"].wait_for, vec!["parent-1", "child-1"]);
        assert_eq!(by_id["parent-2"].delay, Some(200.0));
    }

    #[test]
    fn agentic_converter_rejects_conflicting_trajectory_parents() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("child-1", "child", Some("root-a"), 1_000, 1_100, vec![11]),
                contextual_request("child-2", "child", Some("root-b"), 1_200, 1_300, vec![22]),
            ],
            tools: Vec::new(),
        };

        let err = build_agentic_mooncake_rows(loaded).unwrap_err();
        assert!(err.to_string().contains("conflicting parent_trajectory_id"));
    }

    #[test]
    fn agentic_converter_joins_on_last_finishing_child_request() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("parent-1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("child-slow", "child", Some("root"), 1_200, 1_900, vec![33]),
                contextual_request("child-fast", "child", Some("root"), 1_300, 1_400, vec![44]),
                contextual_request("parent-2", "root", None, 1_500, 1_600, vec![11, 22]),
                contextual_request("parent-3", "root", None, 2_000, 2_100, vec![11, 22, 33]),
            ],
            tools: Vec::new(),
        };

        let (_, rows) = build_agentic_mooncake_rows(loaded).unwrap();
        let by_id = rows
            .iter()
            .map(|row| (row.request_id.as_str(), row))
            .collect::<HashMap<_, _>>();

        assert!(!by_id["parent-2"].wait_for.contains(&"child-fast".into()));
        assert!(by_id["parent-3"].wait_for.contains(&"child-slow".into()));
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agentic lowering: infer the workflow DAG and attribute tool spans to the LLM
//! row that consumed them.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{AgenticMooncakeRow, AgenticToolEvent, RollingHashIdMapper};
use anyhow::{Context, Result, anyhow, bail};

use super::load::{LoadedAgentTrace, RequestEntry, ToolEntry};

/// Streams agentic Mooncake-compatible rows into the replay builder.
///
/// This is an in-memory compatibility layer; it does not write a Mooncake trace.
pub fn lower_agentic_mooncake_rows<F>(mut loaded: LoadedAgentTrace, mut emit: F) -> Result<usize>
where
    F: FnMut(usize, AgenticMooncakeRow) -> Result<()>,
{
    loaded.ensure_agentic_compatible()?;
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

    let mut session_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
    let mut parent_by_session: HashMap<String, String> = HashMap::new();
    for (idx, request) in loaded.requests.iter().enumerate() {
        let session_id = session_id_for(request);
        session_to_indices
            .entry(session_id.clone())
            .or_default()
            .push(idx);
        if let Some(parent) = request
            .agent_context
            .as_ref()
            .and_then(|context| context.parent_session_id.clone())
        {
            match parent_by_session.get(&session_id) {
                Some(existing) if existing != &parent => {
                    bail!(
                        "session {} has conflicting parent_session_id values: {} and {}",
                        session_id,
                        existing,
                        parent
                    );
                }
                Some(_) => {}
                None => {
                    parent_by_session.insert(session_id, parent);
                }
            }
        }
    }
    for indices in session_to_indices.values_mut() {
        indices.sort_by_key(|idx| {
            let request = &loaded.requests[*idx];
            (
                request.start_ms,
                request.end_ms,
                request.request.request_id.clone(),
            )
        });
    }

    let mut explicit_tool_by_child = HashMap::new();
    let mut background_sessions = HashSet::new();
    for tool in &loaded.tools {
        let Some(claude) = tool.claude.as_ref() else {
            continue;
        };
        if !matches!(claude.execution_mode.as_str(), "blocking" | "background") {
            bail!(
                "tool {} has unsupported execution_mode {}",
                tool.tool_call_id,
                claude.execution_mode
            );
        }
        for request_id in [
            Some(claude.source_request_id.as_str()),
            claude.consumer_request_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            let Some(request_idx) = id_to_index.get(request_id) else {
                bail!(
                    "tool {} references unknown request_id {}",
                    tool.tool_call_id,
                    request_id
                );
            };
            if session_id_for(&loaded.requests[*request_idx]) != tool.session_id {
                bail!(
                    "tool {} request {} belongs to a different session",
                    tool.tool_call_id,
                    request_id
                );
            }
        }
        let Some(child_session_id) = claude.child_session_id.as_deref() else {
            continue;
        };
        if !session_to_indices.contains_key(child_session_id) {
            continue;
        }
        if explicit_tool_by_child
            .insert(child_session_id.to_string(), tool)
            .is_some()
        {
            bail!("multiple tool events reference child session {child_session_id}");
        }
        if claude.execution_mode == "background" {
            background_sessions.insert(child_session_id.to_string());
        }
    }

    let mut wait_for: Vec<Vec<String>> = vec![Vec::new(); loaded.requests.len()];
    let mut branches: Vec<Vec<String>> = vec![Vec::new(); loaded.requests.len()];
    let mut prefix_reset = vec![false; loaded.requests.len()];
    let mut previous_request_start_ms = vec![None; loaded.requests.len()];

    for indices in session_to_indices.values() {
        for (pos, idx) in indices.iter().copied().enumerate() {
            prefix_reset[idx] = pos == 0;
            if pos > 0 {
                let previous_request = &loaded.requests[indices[pos - 1]];
                let previous = &previous_request.request.request_id;
                push_unique(&mut wait_for[idx], previous.clone());
                previous_request_start_ms[idx] = Some(previous_request.start_ms);
            }
        }
    }

    for (session_id, parent_id) in &parent_by_session {
        let Some(child_indices) = session_to_indices.get(session_id) else {
            continue;
        };
        let Some(parent_indices) = session_to_indices.get(parent_id) else {
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
            .expect("child session is non-empty");
        if let Some(tool) = explicit_tool_by_child.get(session_id) {
            let claude = tool
                .claude
                .as_ref()
                .expect("explicit child tool has Claude metadata");
            let source_request_id = claude.source_request_id.as_str();
            let parent_spawn_idx = id_to_index[source_request_id];
            if !parent_indices.contains(&parent_spawn_idx) {
                bail!(
                    "tool {} source request {} is not in parent session {}",
                    tool.tool_call_id,
                    source_request_id,
                    parent_id
                );
            }
            let parent_request_id = loaded.requests[parent_spawn_idx].request.request_id.clone();
            push_unique(&mut wait_for[first_child_idx], parent_request_id);
            let child_request_id = loaded.requests[first_child_idx].request.request_id.clone();
            push_unique(&mut branches[parent_spawn_idx], child_request_id);
            if let Some(consumer_request_id) = claude.consumer_request_id.as_deref() {
                let parent_join_idx = id_to_index[consumer_request_id];
                if !parent_indices.contains(&parent_join_idx) {
                    bail!(
                        "tool {} consumer request {} is not in parent session {}",
                        tool.tool_call_id,
                        consumer_request_id,
                        parent_id
                    );
                }
                let child_request_id = loaded.requests[last_finishing_child_idx]
                    .request
                    .request_id
                    .clone();
                push_unique(&mut wait_for[parent_join_idx], child_request_id);
            }
            continue;
        }

        let child_start_ms = loaded.requests[first_child_idx].start_ms;
        let child_end_ms = loaded.requests[last_finishing_child_idx].end_ms;
        if let Some(parent_spawn_idx) =
            latest_request_starting_before(&loaded.requests, parent_indices, child_start_ms)
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
    validate_dependency_dag(&loaded.requests, &wait_for, &id_to_index)?;

    let mut tools_by_session: HashMap<String, Vec<ToolEntry>> = HashMap::new();
    for tool in loaded.tools {
        tools_by_session
            .entry(tool.session_id.clone())
            .or_default()
            .push(tool);
    }
    for tools in tools_by_session.values_mut() {
        tools.sort_by_key(|tool| (tool.start_ms, tool.end_ms));
    }

    let mut mapper = RollingHashIdMapper::new(trace_block_size);
    for (idx, request) in loaded.requests.iter().enumerate() {
        let hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        let session_id = session_id_for(request);
        let dep_end_ms = wait_for[idx]
            .iter()
            .filter_map(|dependency| id_to_index.get(dependency))
            .map(|dep_idx| loaded.requests[*dep_idx].end_ms)
            .max();
        let (delay, tool_wait_ms, tool_events) = if let Some(dep_end_ms) = dep_end_ms {
            let observed_gap_ms = request.start_ms.saturating_sub(dep_end_ms).max(0) as f64;
            let tool_event_start_ms = previous_request_start_ms[idx].unwrap_or(dep_end_ms);
            let (raw_tool_wait_ms, contributing) = tools_by_session
                .get(&session_id)
                .map(|tools| {
                    collect_tools_in_window(
                        tools,
                        &request.request.request_id,
                        tool_event_start_ms,
                        dep_end_ms,
                        request.start_ms,
                    )
                })
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

        emit(
            trace_block_size,
            AgenticMooncakeRow {
                request_id: request.request.request_id.clone(),
                session_id: Some(session_id.clone()),
                input_length: Some(request.replay.input_length),
                output_length: Some(
                    usize::try_from(output_length)
                        .context("output length does not fit in usize")?,
                ),
                hash_ids: Some(hash_ids),
                request_kind: Some(
                    if background_sessions.contains(&session_id) {
                        "background_agent"
                    } else if parent_by_session.contains_key(&session_id) {
                        "agent"
                    } else {
                        "foreground"
                    }
                    .to_string(),
                ),
                timestamp: Some((request.start_ms - global_start_ms) as f64),
                delay,
                wait_for: std::mem::take(&mut wait_for[idx]),
                branches: std::mem::take(&mut branches[idx]),
                prefix_reset: Some(prefix_reset[idx]),
                tool_wait_ms,
                tool_events,
                ..Default::default()
            },
        )?;
    }

    Ok(trace_block_size)
}

fn session_id_for(request: &RequestEntry) -> String {
    request
        .agent_context
        .as_ref()
        .map(|context| context.session_id.clone())
        .unwrap_or_else(|| request.request.request_id.clone())
}

fn latest_request_starting_before(
    requests: &[RequestEntry],
    indices: &[usize],
    timestamp_ms: i64,
) -> Option<usize> {
    indices
        .iter()
        .copied()
        .filter(|idx| requests[*idx].start_ms <= timestamp_ms)
        .max_by_key(|idx| requests[*idx].start_ms)
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

/// Return tools completed since the previous request started, while computing
/// wait time only from their overlap with `[wait_start_ms, end_ms]`.
fn collect_tools_in_window<'a>(
    tools: &'a [ToolEntry],
    request_id: &str,
    event_start_ms: i64,
    wait_start_ms: i64,
    end_ms: i64,
) -> (f64, Vec<&'a ToolEntry>) {
    let mut contributing: Vec<&ToolEntry> = Vec::new();
    let mut intervals = Vec::new();
    for tool in tools {
        let claude = tool.claude.as_ref();
        if let Some(consumer_request_id) =
            claude.and_then(|metadata| metadata.consumer_request_id.as_deref())
        {
            if consumer_request_id != request_id {
                continue;
            }
        } else if claude.is_some_and(|metadata| metadata.execution_mode == "background")
            || tool.end_ms <= event_start_ms
            || tool.end_ms > end_ms
        {
            continue;
        }
        contributing.push(tool);
        let clipped_start = tool.start_ms.max(wait_start_ms);
        let clipped_end = tool.end_ms.min(end_ms);
        if clipped_end > clipped_start {
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

fn validate_dependency_dag(
    requests: &[RequestEntry],
    wait_for: &[Vec<String>],
    id_to_index: &HashMap<String, usize>,
) -> Result<()> {
    let mut indegree = wait_for.iter().map(Vec::len).collect::<Vec<_>>();
    let mut dependents = vec![Vec::new(); requests.len()];
    for (request_idx, dependencies) in wait_for.iter().enumerate() {
        for dependency in dependencies {
            let dependency_idx = id_to_index.get(dependency).ok_or_else(|| {
                anyhow!(
                    "request {} depends on unknown request {}",
                    requests[request_idx].request.request_id,
                    dependency
                )
            })?;
            dependents[*dependency_idx].push(request_idx);
        }
    }

    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(idx, count)| (*count == 0).then_some(idx))
        .collect::<VecDeque<_>>();
    let mut visited = 0;
    while let Some(idx) = ready.pop_front() {
        visited += 1;
        for dependent in &dependents[idx] {
            indegree[*dependent] -= 1;
            if indegree[*dependent] == 0 {
                ready.push_back(*dependent);
            }
        }
    }
    if visited != requests.len() {
        bail!("agentic request dependencies contain a cycle");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_trace::load::{
        AgentContextFields, ClaudeToolReplayMetrics, RequestEntry, RequestTraceReplayMetrics,
        RequestTraceRequestMetrics, ToolEntry,
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
            request: RequestTraceRequestMetrics {
                request_id: request_id.to_string(),
                output_tokens: Some(5),
                request_received_ms: Some(start_ms as u64),
                total_time_ms: Some((end_ms - start_ms) as f64),
                replay: None,
            },
            replay: RequestTraceReplayMetrics {
                trace_block_size: 2,
                input_length: sequence_hashes.len() * 2,
                input_sequence_hashes: sequence_hashes,
            },
        }
    }

    fn contextual_request(
        request_id: &str,
        session_id: &str,
        parent_session_id: Option<&str>,
        start_ms: i64,
        end_ms: i64,
        sequence_hashes: Vec<u64>,
    ) -> RequestEntry {
        let mut entry = request(request_id, start_ms, end_ms, sequence_hashes);
        entry.agent_context = Some(AgentContextFields {
            session_id: session_id.to_string(),
            parent_session_id: parent_session_id.map(str::to_string),
        });
        entry
    }

    fn tool(
        session_id: &str,
        tool_call_id: &str,
        tool_class: &str,
        start_ms: i64,
        end_ms: i64,
    ) -> ToolEntry {
        ToolEntry {
            session_id: session_id.to_string(),
            start_ms,
            end_ms,
            tool_call_id: tool_call_id.to_string(),
            tool_class: tool_class.to_string(),
            claude: None,
            status: "succeeded".to_string(),
            duration_ms: (end_ms - start_ms).max(0) as f64,
            output_bytes: None,
            output_tokens: None,
            error_type: None,
        }
    }

    fn lower_rows(loaded: LoadedAgentTrace) -> Result<Vec<AgenticMooncakeRow>> {
        let mut rows = Vec::with_capacity(loaded.requests.len());
        lower_agentic_mooncake_rows(loaded, |_, row| {
            rows.push(row);
            Ok(())
        })?;
        Ok(rows)
    }

    #[test]
    fn agentic_lowering_builds_sequential_waits_and_tool_wait_components() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("r1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("r2", "root", None, 1_300, 1_400, vec![11, 22]),
            ],
            tools: vec![tool("root", "call-1", "ls", 1_150, 1_250)],
        };

        let rows = lower_rows(loaded).unwrap();

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
    fn agentic_lowering_attaches_parallel_tool_events_with_union_wait() {
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

        let rows = lower_rows(loaded).unwrap();

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
    fn agentic_lowering_adds_subagent_launch_and_join_dependencies() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("parent-1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("child-1", "child", Some("root"), 1_200, 1_300, vec![33]),
                contextual_request("parent-2", "root", None, 1_500, 1_600, vec![11, 22]),
            ],
            tools: Vec::new(),
        };

        let rows = lower_rows(loaded).unwrap();
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
    fn explicit_background_agent_causality_allows_parent_work_until_join() {
        let mut agent_tool = tool("root", "agent-call", "Agent", 1_100, 1_800);
        agent_tool.claude = Some(ClaudeToolReplayMetrics {
            source_request_id: "parent-1".to_string(),
            consumer_request_id: Some("parent-3".to_string()),
            child_session_id: Some("child".to_string()),
            execution_mode: "background".to_string(),
        });
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("parent-1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("child-1", "child", Some("root"), 1_200, 1_700, vec![33]),
                contextual_request("parent-2", "root", None, 1_300, 1_400, vec![11, 22]),
                contextual_request("parent-3", "root", None, 1_850, 1_950, vec![11, 22, 44]),
            ],
            tools: vec![agent_tool],
        };

        let rows = lower_rows(loaded).unwrap();
        let by_id = rows
            .iter()
            .map(|row| (row.request_id.as_str(), row))
            .collect::<HashMap<_, _>>();

        assert_eq!(by_id["parent-1"].branches, vec!["child-1"]);
        assert_eq!(by_id["child-1"].wait_for, vec!["parent-1"]);
        assert_eq!(
            by_id["child-1"].request_kind.as_deref(),
            Some("background_agent")
        );
        assert_eq!(by_id["parent-2"].wait_for, vec!["parent-1"]);
        assert_eq!(by_id["parent-3"].wait_for, vec!["parent-2", "child-1"]);
        assert_eq!(by_id["parent-3"].tool_wait_ms, Some(100.0));
        assert_eq!(by_id["parent-3"].delay, Some(50.0));
        assert_eq!(by_id["parent-3"].tool_events.len(), 1);
    }

    #[test]
    fn explicit_causality_rejects_cycles() {
        let mut agent_tool = tool("root", "agent-call", "Agent", 1_100, 1_200);
        agent_tool.claude = Some(ClaudeToolReplayMetrics {
            source_request_id: "parent-2".to_string(),
            consumer_request_id: Some("parent-1".to_string()),
            child_session_id: Some("child".to_string()),
            execution_mode: "background".to_string(),
        });
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("parent-1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("child-1", "child", Some("root"), 1_200, 1_300, vec![33]),
                contextual_request("parent-2", "root", None, 1_400, 1_500, vec![11, 22]),
            ],
            tools: vec![agent_tool],
        };

        let err = lower_rows(loaded).unwrap_err();
        assert!(err.to_string().contains("dependencies contain a cycle"));
    }

    #[test]
    fn missing_child_trace_replays_as_external_background_tool() {
        let mut agent_tool = tool("root", "agent-call", "Agent", 1_100, 1_250);
        agent_tool.claude = Some(ClaudeToolReplayMetrics {
            source_request_id: "parent-1".to_string(),
            consumer_request_id: Some("parent-2".to_string()),
            child_session_id: Some("missing-child".to_string()),
            execution_mode: "background".to_string(),
        });
        let rows = lower_rows(LoadedAgentTrace {
            requests: vec![
                contextual_request("parent-1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("parent-2", "root", None, 1_300, 1_400, vec![11, 22]),
            ],
            tools: vec![agent_tool],
        })
        .unwrap();

        assert!(rows[0].branches.is_empty());
        assert_eq!(rows[1].wait_for, vec!["parent-1"]);
        assert_eq!(rows[1].tool_wait_ms, Some(150.0));
        assert_eq!(rows[1].delay, Some(50.0));
        assert_eq!(rows[1].tool_events.len(), 1);
    }

    #[test]
    fn agentic_lowering_rejects_conflicting_session_parents() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("child-1", "child", Some("root-a"), 1_000, 1_100, vec![11]),
                contextual_request("child-2", "child", Some("root-b"), 1_200, 1_300, vec![22]),
            ],
            tools: Vec::new(),
        };

        let err = lower_rows(loaded).unwrap_err();
        assert!(err.to_string().contains("conflicting parent_session_id"));
    }

    #[test]
    fn agentic_lowering_joins_on_last_finishing_child_request() {
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

        let rows = lower_rows(loaded).unwrap();
        let by_id = rows
            .iter()
            .map(|row| (row.request_id.as_str(), row))
            .collect::<HashMap<_, _>>();

        assert!(!by_id["parent-2"].wait_for.contains(&"child-fast".into()));
        assert!(by_id["parent-3"].wait_for.contains(&"child-slow".into()));
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agentic lowering from Dynamo request traces into the typed causal schema.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    AgenticDependency, AgenticDependencyRelation, AgenticDependencyTrigger, AgenticMooncakeRow,
    RollingHashIdMapper,
};
use anyhow::{Context, Result, anyhow, bail};

use super::load::{LoadedAgentTrace, RequestEntry};

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
            .and_then(|context| context.parent_session_id.as_deref())
            .map(str::to_string)
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

    let mut dependencies: Vec<Vec<AgenticDependency>> = vec![Vec::new(); loaded.requests.len()];

    for indices in session_to_indices.values() {
        for (pos, idx) in indices.iter().copied().enumerate() {
            if pos > 0 {
                let previous_request = &loaded.requests[indices[pos - 1]];
                push_dependency(
                    &mut dependencies[idx],
                    AgenticDependency {
                        request_id: previous_request.request.request_id.clone(),
                        trigger: AgenticDependencyTrigger::Completion,
                        delay_ms: loaded.requests[idx]
                            .start_ms
                            .saturating_sub(previous_request.end_ms)
                            .max(0) as f64,
                        relation: AgenticDependencyRelation::Sequence,
                    },
                );
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
            let parent_request = &loaded.requests[parent_spawn_idx];
            let child_request = &loaded.requests[first_child_idx];
            let (trigger, delay_ms) = if child_request.start_ms < parent_request.end_ms {
                (
                    AgenticDependencyTrigger::Dispatch,
                    child_request
                        .start_ms
                        .saturating_sub(parent_request.start_ms)
                        .max(0) as f64,
                )
            } else {
                (
                    AgenticDependencyTrigger::Completion,
                    child_request
                        .start_ms
                        .saturating_sub(parent_request.end_ms)
                        .max(0) as f64,
                )
            };
            push_dependency(
                &mut dependencies[first_child_idx],
                AgenticDependency {
                    request_id: parent_request.request.request_id.clone(),
                    trigger,
                    delay_ms,
                    relation: AgenticDependencyRelation::Spawn,
                },
            );
            if claude.execution_mode == "blocking"
                && let Some(consumer_request_id) = claude.consumer_request_id.as_deref()
            {
                let parent_join_idx = id_to_index[consumer_request_id];
                if !parent_indices.contains(&parent_join_idx) {
                    bail!(
                        "tool {} consumer request {} is not in parent session {}",
                        tool.tool_call_id,
                        consumer_request_id,
                        parent_id
                    );
                }
                let child_request = &loaded.requests[last_finishing_child_idx];
                push_dependency(
                    &mut dependencies[parent_join_idx],
                    AgenticDependency {
                        request_id: child_request.request.request_id.clone(),
                        trigger: AgenticDependencyTrigger::Completion,
                        delay_ms: loaded.requests[parent_join_idx]
                            .start_ms
                            .saturating_sub(child_request.end_ms)
                            .max(0) as f64,
                        relation: AgenticDependencyRelation::Join,
                    },
                );
            }
            continue;
        }

        let child_start_ms = loaded.requests[first_child_idx].start_ms;
        let child_end_ms = loaded.requests[last_finishing_child_idx].end_ms;
        if let Some(parent_spawn_idx) =
            latest_request_starting_before(&loaded.requests, parent_indices, child_start_ms)
        {
            let parent_request = &loaded.requests[parent_spawn_idx];
            let child_request = &loaded.requests[first_child_idx];
            let (trigger, delay_ms) = if child_request.start_ms < parent_request.end_ms {
                (
                    AgenticDependencyTrigger::Dispatch,
                    child_request
                        .start_ms
                        .saturating_sub(parent_request.start_ms)
                        .max(0) as f64,
                )
            } else {
                (
                    AgenticDependencyTrigger::Completion,
                    child_request
                        .start_ms
                        .saturating_sub(parent_request.end_ms)
                        .max(0) as f64,
                )
            };
            push_dependency(
                &mut dependencies[first_child_idx],
                AgenticDependency {
                    request_id: parent_request.request.request_id.clone(),
                    trigger,
                    delay_ms,
                    relation: AgenticDependencyRelation::Spawn,
                },
            );
        }
        if !background_sessions.contains(session_id)
            && let Some(parent_join_idx) =
                first_request_starting_after(&loaded.requests, parent_indices, child_end_ms)
        {
            let child_request = &loaded.requests[last_finishing_child_idx];
            push_dependency(
                &mut dependencies[parent_join_idx],
                AgenticDependency {
                    request_id: child_request.request.request_id.clone(),
                    trigger: AgenticDependencyTrigger::Completion,
                    delay_ms: loaded.requests[parent_join_idx]
                        .start_ms
                        .saturating_sub(child_request.end_ms)
                        .max(0) as f64,
                    relation: AgenticDependencyRelation::Join,
                },
            );
        }
    }
    validate_dependency_dag(&loaded.requests, &dependencies, &id_to_index)?;

    let play_by_session = resolve_play_ids(&session_to_indices, &parent_by_session)?;

    let mut mapper = RollingHashIdMapper::new(trace_block_size);
    for (idx, request) in loaded.requests.iter().enumerate() {
        let mut hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
        let full_blocks = request.replay.input_length / trace_block_size;
        hash_ids.truncate(full_blocks);
        while hash_ids.len() < full_blocks {
            hash_ids.push(private_request_hash(
                &request.request.request_id,
                hash_ids.len(),
                request.replay.input_length,
            ));
        }
        if request.replay.input_length % trace_block_size != 0 {
            hash_ids.push(private_request_hash(
                &request.request.request_id,
                full_blocks,
                request.replay.input_length,
            ));
        }
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        let session_id = session_id_for(request);

        emit(
            trace_block_size,
            AgenticMooncakeRow {
                request_id: request.request.request_id.clone(),
                play_id: play_by_session[&session_id].clone(),
                session_id,
                model: request.request.model.clone().ok_or_else(|| {
                    anyhow!("request {} is missing model", request.request.request_id)
                })?,
                input_length: Some(request.replay.input_length),
                output_length: Some(
                    usize::try_from(output_length)
                        .context("output length does not fit in usize")?,
                ),
                hash_ids: Some(hash_ids),
                not_before_ms: (request.start_ms - global_start_ms) as f64,
                dependencies: std::mem::take(&mut dependencies[idx]),
                ..Default::default()
            },
        )?;
    }

    Ok(trace_block_size)
}

fn private_request_hash(request_id: &str, block_index: usize, input_length: usize) -> u64 {
    let digest = blake3::hash(
        format!("dynamo-request-trace-private-block\0{request_id}\0{block_index}\0{input_length}")
            .as_bytes(),
    );
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap())
}

fn resolve_play_ids(
    session_to_indices: &HashMap<String, Vec<usize>>,
    parent_by_session: &HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    let mut result = HashMap::new();
    for session_id in session_to_indices.keys() {
        let mut cursor = session_id.as_str();
        let mut seen = HashSet::new();
        while let Some(parent) = parent_by_session.get(cursor) {
            if !seen.insert(cursor.to_string()) {
                bail!("agent session parent cycle includes {cursor}");
            }
            if !session_to_indices.contains_key(parent) {
                bail!("agent session {cursor} references unknown parent session {parent}");
            }
            cursor = parent;
        }
        result.insert(session_id.clone(), cursor.to_string());
    }
    Ok(result)
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

fn push_dependency(values: &mut Vec<AgenticDependency>, dependency: AgenticDependency) {
    if !values.iter().any(|existing| {
        existing.request_id == dependency.request_id
            && existing.trigger == dependency.trigger
            && existing.relation == dependency.relation
    }) {
        values.push(dependency);
    }
}

fn validate_dependency_dag(
    requests: &[RequestEntry],
    dependencies: &[Vec<AgenticDependency>],
    id_to_index: &HashMap<String, usize>,
) -> Result<()> {
    let mut indegree = dependencies.iter().map(Vec::len).collect::<Vec<_>>();
    let mut dependents = vec![Vec::new(); requests.len()];
    for (request_idx, dependencies) in dependencies.iter().enumerate() {
        for dependency in dependencies {
            let dependency_idx = id_to_index.get(&dependency.request_id).ok_or_else(|| {
                anyhow!(
                    "request {} depends on unknown request {}",
                    requests[request_idx].request.request_id,
                    dependency.request_id
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
                ..Default::default()
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
    fn agentic_lowering_builds_completion_sequences() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("r1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("r2", "root", None, 1_300, 1_400, vec![11, 22]),
            ],
            tools: vec![tool("root", "call-1", "ls", 1_150, 1_250)],
        };

        let rows = lower_rows(loaded).unwrap();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].dependencies.is_empty());
        assert_eq!(rows[0].play_id, "root");
        assert_eq!(rows[1].dependencies.len(), 1);
        let edge = &rows[1].dependencies[0];
        assert_eq!(edge.request_id, "r1");
        assert_eq!(edge.trigger, AgenticDependencyTrigger::Completion);
        assert_eq!(edge.relation, AgenticDependencyRelation::Sequence);
        assert_eq!(edge.delay_ms, 200.0);
    }

    #[test]
    fn non_agent_tool_time_is_preserved_by_the_sequence_gap() {
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

        assert_eq!(rows[1].dependencies.len(), 1);
        assert_eq!(rows[1].dependencies[0].delay_ms, 300.0);
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

        assert!(by_id["child-1"].dependencies.iter().any(|edge| {
            edge.request_id == "parent-1"
                && edge.trigger == AgenticDependencyTrigger::Completion
                && edge.relation == AgenticDependencyRelation::Spawn
                && edge.delay_ms == 100.0
        }));
        assert!(by_id["parent-2"].dependencies.iter().any(|edge| {
            edge.request_id == "child-1" && edge.relation == AgenticDependencyRelation::Join
        }));
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

        assert!(by_id["child-1"].dependencies.iter().any(|edge| {
            edge.request_id == "parent-1" && edge.relation == AgenticDependencyRelation::Spawn
        }));
        assert!(!by_id["parent-3"].dependencies.iter().any(|edge| {
            edge.request_id == "child-1" && edge.relation == AgenticDependencyRelation::Join
        }));
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

        assert_eq!(rows[1].dependencies.len(), 1);
        assert_eq!(rows[1].dependencies[0].request_id, "parent-1");
        assert_eq!(rows[1].dependencies[0].delay_ms, 200.0);
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

        assert!(!by_id["parent-2"].dependencies.iter().any(|edge| {
            edge.request_id == "child-fast" && edge.relation == AgenticDependencyRelation::Join
        }));
        assert!(by_id["parent-3"].dependencies.iter().any(|edge| {
            edge.request_id == "child-slow" && edge.relation == AgenticDependencyRelation::Join
        }));
    }
}

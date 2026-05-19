// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use dynamo_bench::coding::common::expand_user_path;
use dynamo_data_gen::{AgenticMooncakeRow, MooncakeJsonlWriter, MooncakeRow, RollingHashIdMapper};
use flate2::read::MultiGzDecoder;
use serde::Deserialize;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(name = "agent_trace_to_mooncake")]
#[command(about = "Convert Dynamo agent trace JSONL/JSONL.GZ records to Mooncake replay JSONL")]
struct Args {
    #[arg(long, action = clap::ArgAction::Append, required = true, num_args = 1..)]
    input_path: Vec<String>,

    #[arg(long)]
    output_file: String,

    #[arg(long)]
    agentic: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentTraceRecord {
    event_type: String,
    event_time_unix_ms: u64,
    #[serde(default)]
    agent_context: Option<AgentContextFields>,
    #[serde(default)]
    request: Option<AgentRequestMetrics>,
    #[serde(default)]
    tool: Option<AgentToolEventMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentContextFields {
    #[serde(rename = "session_id")]
    _session_id: String,
    trajectory_id: String,
    #[serde(default)]
    parent_trajectory_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentRequestMetrics {
    request_id: String,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    request_received_ms: Option<u64>,
    #[serde(default)]
    total_time_ms: Option<f64>,
    #[serde(default)]
    replay: Option<AgentReplayMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentReplayMetrics {
    trace_block_size: usize,
    input_length: usize,
    input_sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentToolEventMetrics {
    #[serde(default)]
    started_at_unix_ms: Option<u64>,
    #[serde(default)]
    ended_at_unix_ms: Option<u64>,
    #[serde(default)]
    duration_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct RequestEntry {
    start_ms: i64,
    end_ms: i64,
    agent_context: Option<AgentContextFields>,
    request: AgentRequestMetrics,
    replay: AgentReplayMetrics,
}

#[derive(Debug, Clone)]
struct ToolEntry {
    trajectory_id: String,
    start_ms: i64,
    end_ms: i64,
}

#[derive(Debug, Default)]
struct LoadedAgentTrace {
    requests: Vec<RequestEntry>,
    tools: Vec<ToolEntry>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_paths = args
        .input_path
        .iter()
        .map(|path| expand_user_path(path))
        .collect::<Vec<_>>();
    let output_path = expand_user_path(&args.output_file);

    let loaded = load_agent_trace_records(&input_paths)?;
    let mut writer = MooncakeJsonlWriter::create(&output_path, None)?;

    let trace_block_size = if args.agentic {
        let (trace_block_size, rows) = build_agentic_mooncake_rows(loaded)?;
        for row in &rows {
            writer.write_agentic_row(row)?;
        }
        trace_block_size
    } else {
        let (trace_block_size, rows) = build_mooncake_rows(loaded.requests)?;
        for row in &rows {
            writer.write_row(row)?;
        }
        trace_block_size
    };

    let stats = writer.finish()?;
    if args.agentic {
        println!(
            "Wrote {} Agentic Mooncake rows to {}",
            stats.row_count,
            output_path.display()
        );
    } else {
        println!(
            "Wrote {} Mooncake rows to {}",
            stats.row_count,
            output_path.display()
        );
    }
    println!("Trace block size: {trace_block_size}");
    Ok(())
}

fn load_agent_trace_records(paths: &[PathBuf]) -> Result<LoadedAgentTrace> {
    let mut loaded = LoadedAgentTrace::default();

    for path in paths {
        let reader = open_trace_reader(path)?;
        for (line_index, line) in reader.lines().enumerate() {
            let line = line
                .with_context(|| format!("failed to read {}:{}", path.display(), line_index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let Some(record) = parse_trace_record(&line).with_context(|| {
                format!("failed to parse {}:{}", path.display(), line_index + 1)
            })?
            else {
                continue;
            };
            if record.event_type == "request_end" {
                loaded.requests.push(request_entry(record).with_context(|| {
                    format!(
                        "invalid request_end at {}:{}",
                        path.display(),
                        line_index + 1
                    )
                })?);
            } else if matches!(record.event_type.as_str(), "tool_end" | "tool_error")
                && let Some(tool) = tool_entry(record)
            {
                loaded.tools.push(tool);
            }
        }
    }

    if loaded.requests.is_empty() {
        bail!("no request_end records with replay fields found");
    }

    Ok(loaded)
}

fn open_trace_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

fn parse_trace_record(line: &str) -> Result<Option<AgentTraceRecord>> {
    let value: Value = serde_json::from_str(line)?;
    let event = value.get("event").unwrap_or(&value);
    if !event.is_object() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_value(event.clone())?))
}

fn request_entry(record: AgentTraceRecord) -> Result<RequestEntry> {
    let request = record
        .request
        .ok_or_else(|| anyhow!("request_end record is missing request payload"))?;
    let replay = request
        .replay
        .clone()
        .ok_or_else(|| anyhow!("request payload is missing replay metrics"))?;
    if replay.trace_block_size == 0 {
        bail!("request replay trace_block_size must be greater than 0");
    }
    if replay.input_sequence_hashes.len() * replay.trace_block_size < replay.input_length {
        bail!(
            "input_length {} exceeds replay hash capacity {}",
            replay.input_length,
            replay.input_sequence_hashes.len() * replay.trace_block_size
        );
    }

    let (start_ms, end_ms) = request_times(record.event_time_unix_ms, &request);
    Ok(RequestEntry {
        start_ms,
        end_ms,
        agent_context: record.agent_context,
        request,
        replay,
    })
}

fn tool_entry(record: AgentTraceRecord) -> Option<ToolEntry> {
    let context = record.agent_context?;
    let tool = record.tool?;
    let end_ms = tool
        .ended_at_unix_ms
        .map(saturating_i64)
        .unwrap_or_else(|| saturating_i64(record.event_time_unix_ms));
    let start_ms = tool
        .started_at_unix_ms
        .map(saturating_i64)
        .or_else(|| {
            tool.duration_ms
                .map(|duration_ms| end_ms.saturating_sub(duration_ms.max(0.0).round() as i64))
        })
        .unwrap_or(end_ms);
    if end_ms < start_ms {
        return None;
    }
    Some(ToolEntry {
        trajectory_id: context.trajectory_id,
        start_ms,
        end_ms,
    })
}

fn request_times(event_time_unix_ms: u64, request: &AgentRequestMetrics) -> (i64, i64) {
    let total_ms = request
        .total_time_ms
        .map(|value| value.max(0.0).round() as u64)
        .unwrap_or_else(|| {
            event_time_unix_ms
                .saturating_sub(request.request_received_ms.unwrap_or(event_time_unix_ms))
        });
    let end_ms = request
        .request_received_ms
        .map(|start| start.saturating_add(total_ms))
        .unwrap_or(event_time_unix_ms);
    let start_ms = request
        .request_received_ms
        .unwrap_or_else(|| event_time_unix_ms.saturating_sub(total_ms));
    (saturating_i64(start_ms), saturating_i64(end_ms))
}

fn build_mooncake_rows(mut requests: Vec<RequestEntry>) -> Result<(usize, Vec<MooncakeRow>)> {
    let global_start_ms = requests
        .iter()
        .map(|request| request.start_ms)
        .min()
        .ok_or_else(|| anyhow!("no request records to convert"))?;
    let trace_block_size = requests[0].replay.trace_block_size;
    for request in &requests {
        if request.replay.trace_block_size != trace_block_size {
            bail!(
                "mixed replay trace_block_size values are not supported: {} and {}",
                trace_block_size,
                request.replay.trace_block_size
            );
        }
    }

    requests.sort_by(|left, right| {
        (left.start_ms, left.end_ms, &left.request.request_id).cmp(&(
            right.start_ms,
            right.end_ms,
            &right.request.request_id,
        ))
    });

    let mut mapper = RollingHashIdMapper::new(trace_block_size);
    let mut rows = Vec::new();
    for request in requests {
        let hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        rows.push(MooncakeRow {
            session_id: None,
            input_length: Some(request.replay.input_length),
            output_length: Some(
                usize::try_from(output_length).context("output length does not fit in usize")?,
            ),
            hash_ids: Some(hash_ids),
            timestamp: Some((request.start_ms - global_start_ms) as f64),
            delay: None,
        });
    }

    Ok((trace_block_size, rows))
}

fn build_agentic_mooncake_rows(
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
        let (delay, tool_wait_ms) = if let Some(dep_end_ms) = dep_end_ms {
            let observed_gap_ms = request.start_ms.saturating_sub(dep_end_ms).max(0) as f64;
            let tool_wait_ms = tools_by_trajectory
                .get(&trajectory_id)
                .map(|tools| tool_wait_between(tools, dep_end_ms, request.start_ms))
                .unwrap_or(0.0)
                .min(observed_gap_ms);
            let non_tool_wait_ms = (observed_gap_ms - tool_wait_ms).max(0.0);
            (
                Some(non_tool_wait_ms),
                (tool_wait_ms > 0.0).then_some(tool_wait_ms),
            )
        } else {
            (None, None)
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

fn tool_wait_between(tools: &[ToolEntry], start_ms: i64, end_ms: i64) -> f64 {
    if end_ms <= start_ms {
        return 0.0;
    }

    let mut intervals = tools
        .iter()
        .filter_map(|tool| {
            let start = tool.start_ms.max(start_ms);
            let end = tool.end_ms.min(end_ms);
            (end > start).then_some((start, end))
        })
        .collect::<Vec<_>>();
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
    total as f64
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn saturating_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn converter_preserves_absolute_timestamps_per_request() {
        let requests = vec![
            request("req-a", 1_000, 1_100, vec![11, 22]),
            request("req-b", 1_500, 1_600, vec![11, 33]),
        ];

        let (_, entries) = build_mooncake_rows(requests).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp, Some(0.0));
        assert_eq!(entries[0].delay, None);
        assert_eq!(entries[1].timestamp, Some(500.0));
        assert_eq!(entries[1].delay, None);
        assert_eq!(entries[0].session_id, None);
        assert_eq!(entries[1].session_id, None);
        assert_eq!(
            entries[0].hash_ids.as_ref().unwrap()[0],
            entries[1].hash_ids.as_ref().unwrap()[0]
        );
    }

    #[test]
    fn converter_preserves_parallel_start_times_as_independent_rows() {
        let requests = vec![
            request("req-a", 1_000, 1_500, vec![11]),
            request("req-b", 1_000, 1_700, vec![22]),
        ];

        let (_, entries) = build_mooncake_rows(requests).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp, Some(0.0));
        assert_eq!(entries[1].timestamp, Some(0.0));
        assert_eq!(entries[0].delay, None);
        assert_eq!(entries[1].delay, None);
        assert_eq!(entries[0].session_id, None);
        assert_eq!(entries[1].session_id, None);
    }

    #[test]
    fn request_times_uses_event_time_when_total_duration_is_missing() {
        let request = AgentRequestMetrics {
            request_id: "req".to_string(),
            output_tokens: Some(1),
            request_received_ms: Some(1_000),
            total_time_ms: None,
            replay: None,
        };

        assert_eq!(request_times(1_250, &request), (1_000, 1_250));
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

    #[test]
    fn agentic_converter_builds_sequential_waits_and_tool_wait_components() {
        let loaded = LoadedAgentTrace {
            requests: vec![
                contextual_request("r1", "root", None, 1_000, 1_100, vec![11]),
                contextual_request("r2", "root", None, 1_300, 1_400, vec![11, 22]),
            ],
            tools: vec![ToolEntry {
                trajectory_id: "root".to_string(),
                start_ms: 1_150,
                end_ms: 1_250,
            }],
        };

        let (_, rows) = build_agentic_mooncake_rows(loaded).unwrap();

        assert_eq!(rows.len(), 2);
        assert!(rows[0].wait_for.is_empty());
        assert_eq!(rows[0].prefix_reset, Some(true));
        assert_eq!(rows[1].wait_for, vec!["r1"]);
        assert_eq!(rows[1].delay, Some(100.0));
        assert_eq!(rows[1].tool_wait_ms, Some(100.0));
        assert_eq!(rows[1].dependency_delay_ms(), 200.0);
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

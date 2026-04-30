// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use dynamo_bench::coding::common::{expand_user_path, sidecar_path_for};
use dynamo_bench::coding::mooncake::{MooncakeJsonlWriter, MooncakeRow, RollingHashIdMapper};
use flate2::read::MultiGzDecoder;
use serde::{Deserialize, Serialize};
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
    sidecar_file: Option<String>,

    #[arg(long)]
    no_sidecar: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentTraceRecord {
    event_type: String,
    event_time_unix_ms: u64,
    agent_context: AgentContext,
    #[serde(default)]
    request: Option<AgentRequestMetrics>,
    #[serde(default)]
    tool: Option<AgentToolEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentContext {
    workflow_type_id: String,
    workflow_id: String,
    program_id: String,
    #[serde(default)]
    parent_program_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentRequestMetrics {
    request_id: String,
    #[serde(default)]
    x_request_id: Option<String>,
    model: String,
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
    #[serde(default)]
    output_length: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentToolEvent {
    tool_class: String,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Debug, Clone)]
struct RequestEntry {
    session_id: String,
    start_ms: i64,
    end_ms: i64,
    agent_context: AgentContext,
    request: AgentRequestMetrics,
    replay: AgentReplayMetrics,
}

#[derive(Debug, Clone)]
struct ToolEntry {
    session_id: String,
    event_time_ms: i64,
    event_type: String,
    tool_class: String,
    status: Option<String>,
}

#[derive(Debug, Clone)]
struct OutputEntry {
    sort_ms: i64,
    row: MooncakeRow,
    sidecar: AgentTraceSidecar,
}

#[derive(Debug, Clone, Serialize)]
struct AgentTraceSidecar {
    session_id: String,
    workflow_type_id: String,
    workflow_id: String,
    program_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_program_id: Option<String>,
    request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    x_request_id: Option<String>,
    model: String,
    request_start_ms: i64,
    request_end_ms: i64,
    tool_gap: ToolGapSummary,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ToolGapSummary {
    event_count: usize,
    error_count: usize,
    by_class: BTreeMap<String, usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_paths = args
        .input_path
        .iter()
        .map(|path| expand_user_path(path))
        .collect::<Vec<_>>();
    let output_path = expand_user_path(&args.output_file);
    let sidecar_path = if args.no_sidecar {
        None
    } else {
        Some(
            args.sidecar_file
                .as_deref()
                .map(expand_user_path)
                .unwrap_or_else(|| sidecar_path_for(&output_path)),
        )
    };

    let (requests, tools) = load_agent_trace(&input_paths)?;
    let (trace_block_size, output_entries) = build_output_entries(requests, tools)?;
    let mut writer = MooncakeJsonlWriter::create(&output_path, sidecar_path.as_deref())?;

    for entry in &output_entries {
        writer.write_row(&entry.row)?;
        if writer.has_sidecar() {
            writer.write_sidecar(&entry.sidecar)?;
        }
    }

    let stats = writer.finish()?;
    println!(
        "Wrote {} Mooncake rows to {}",
        stats.row_count,
        output_path.display()
    );
    if let Some(sidecar_path) = sidecar_path {
        println!(
            "Wrote {} sidecar rows to {}",
            stats.sidecar_count,
            sidecar_path.display()
        );
    }
    println!("Trace block size: {trace_block_size}");
    Ok(())
}

fn load_agent_trace(paths: &[PathBuf]) -> Result<(Vec<RequestEntry>, Vec<ToolEntry>)> {
    let mut requests = Vec::new();
    let mut tools = Vec::new();

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
                requests.push(request_entry(record).with_context(|| {
                    format!(
                        "invalid request_end at {}:{}",
                        path.display(),
                        line_index + 1
                    )
                })?);
            } else if record.event_type.starts_with("tool_")
                && let Some(tool) = record.tool
            {
                let session_id = session_id(&record.agent_context);
                tools.push(ToolEntry {
                    session_id,
                    event_time_ms: saturating_i64(record.event_time_unix_ms),
                    event_type: record.event_type,
                    tool_class: tool.tool_class,
                    status: tool.status,
                });
            }
        }
    }

    if requests.is_empty() {
        bail!("no request_end records with replay fields found");
    }

    Ok((requests, tools))
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
        session_id: session_id(&record.agent_context),
        start_ms,
        end_ms,
        agent_context: record.agent_context,
        request,
        replay,
    })
}

fn request_times(event_time_unix_ms: u64, request: &AgentRequestMetrics) -> (i64, i64) {
    let total_ms = request
        .total_time_ms
        .map(|value| value.max(0.0).round() as u64)
        .unwrap_or(0);
    let end_ms = request
        .request_received_ms
        .map(|start| start.saturating_add(total_ms))
        .unwrap_or(event_time_unix_ms);
    let start_ms = request
        .request_received_ms
        .unwrap_or_else(|| event_time_unix_ms.saturating_sub(total_ms));
    (saturating_i64(start_ms), saturating_i64(end_ms))
}

fn session_id(agent_context: &AgentContext) -> String {
    if agent_context.program_id == agent_context.workflow_id
        || agent_context
            .program_id
            .starts_with(&format!("{}:", agent_context.workflow_id))
    {
        return agent_context.program_id.clone();
    }

    format!("{}:{}", agent_context.workflow_id, agent_context.program_id)
}

fn build_output_entries(
    requests: Vec<RequestEntry>,
    tools: Vec<ToolEntry>,
) -> Result<(usize, Vec<OutputEntry>)> {
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

    let mut by_session: HashMap<String, Vec<RequestEntry>> = HashMap::new();
    for request in requests {
        by_session
            .entry(request.session_id.clone())
            .or_default()
            .push(request);
    }

    let mut tools_by_session: HashMap<String, Vec<ToolEntry>> = HashMap::new();
    for tool in tools {
        tools_by_session
            .entry(tool.session_id.clone())
            .or_default()
            .push(tool);
    }
    for tools in tools_by_session.values_mut() {
        tools.sort_by_key(|tool| tool.event_time_ms);
    }

    let mut mapper = RollingHashIdMapper::new(trace_block_size);
    let mut output_entries = Vec::new();
    for (session_id, mut session_requests) in by_session {
        session_requests.sort_by_key(|request| (request.start_ms, request.end_ms));
        let session_tools = tools_by_session.get(&session_id).map(Vec::as_slice);
        let mut previous_end_ms: Option<i64> = None;

        for request in session_requests {
            let hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
            let output_length = request
                .replay
                .output_length
                .or(request.request.output_tokens)
                .ok_or_else(|| {
                    anyhow!(
                        "request {} is missing output length",
                        request.request.request_id
                    )
                })?;
            let row = MooncakeRow {
                session_id: session_id.clone(),
                input_length: request.replay.input_length,
                output_length: usize::try_from(output_length)
                    .context("output length does not fit in usize")?,
                hash_ids,
                timestamp: previous_end_ms
                    .is_none()
                    .then_some(request.start_ms - global_start_ms),
                delay: previous_end_ms.map(|previous_end| (request.start_ms - previous_end).max(0)),
            };
            let sidecar = AgentTraceSidecar {
                session_id: session_id.clone(),
                workflow_type_id: request.agent_context.workflow_type_id.clone(),
                workflow_id: request.agent_context.workflow_id.clone(),
                program_id: request.agent_context.program_id.clone(),
                parent_program_id: request.agent_context.parent_program_id.clone(),
                request_id: request.request.request_id.clone(),
                x_request_id: request.request.x_request_id.clone(),
                model: request.request.model.clone(),
                request_start_ms: request.start_ms,
                request_end_ms: request.end_ms,
                tool_gap: summarize_tool_gap(session_tools, previous_end_ms, request.start_ms),
            };
            previous_end_ms = Some(request.end_ms);
            output_entries.push(OutputEntry {
                sort_ms: request.start_ms,
                row,
                sidecar,
            });
        }
    }

    output_entries.sort_by_key(|entry| (entry.sort_ms, entry.row.session_id.clone()));
    Ok((trace_block_size, output_entries))
}

fn summarize_tool_gap(
    tools: Option<&[ToolEntry]>,
    previous_end_ms: Option<i64>,
    request_start_ms: i64,
) -> ToolGapSummary {
    let Some(previous_end_ms) = previous_end_ms else {
        return ToolGapSummary::default();
    };
    let Some(tools) = tools else {
        return ToolGapSummary::default();
    };

    let mut summary = ToolGapSummary::default();
    for tool in tools {
        if tool.event_time_ms < previous_end_ms || tool.event_time_ms > request_start_ms {
            continue;
        }
        summary.event_count += 1;
        *summary.by_class.entry(tool.tool_class.clone()).or_default() += 1;
        if tool.event_type == "tool_error" || tool.status.as_deref() == Some("error") {
            summary.error_count += 1;
        }
    }
    summary
}

fn saturating_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(
        session: &str,
        start_ms: i64,
        end_ms: i64,
        sequence_hashes: Vec<u64>,
    ) -> RequestEntry {
        let workflow_id = "workflow-a".to_string();
        let program_id = session.to_string();
        RequestEntry {
            session_id: format!("{workflow_id}:{program_id}"),
            start_ms,
            end_ms,
            agent_context: AgentContext {
                workflow_type_id: "ms_agent".to_string(),
                workflow_id,
                program_id,
                parent_program_id: None,
            },
            request: AgentRequestMetrics {
                request_id: format!("req-{start_ms}"),
                x_request_id: None,
                model: "test-model".to_string(),
                output_tokens: Some(5),
                request_received_ms: Some(start_ms as u64),
                total_time_ms: Some((end_ms - start_ms) as f64),
                replay: None,
            },
            replay: AgentReplayMetrics {
                trace_block_size: 2,
                input_length: sequence_hashes.len() * 2,
                input_sequence_hashes: sequence_hashes,
                output_length: None,
            },
        }
    }

    #[test]
    fn converter_uses_timestamp_then_delay_per_session() {
        let requests = vec![
            request("agent", 1_000, 1_100, vec![11, 22]),
            request("agent", 1_500, 1_600, vec![11, 33]),
        ];

        let (_, entries) = build_output_entries(requests, vec![]).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].row.timestamp, Some(0));
        assert_eq!(entries[0].row.delay, None);
        assert_eq!(entries[1].row.timestamp, None);
        assert_eq!(entries[1].row.delay, Some(400));
        assert_eq!(entries[0].row.hash_ids[0], entries[1].row.hash_ids[0]);
    }

    #[test]
    fn converter_records_tool_gap_summary() {
        let requests = vec![
            request("agent", 1_000, 1_100, vec![11]),
            request("agent", 1_500, 1_600, vec![11, 22]),
        ];
        let tools = vec![
            ToolEntry {
                session_id: "workflow-a:agent".to_string(),
                event_time_ms: 1_200,
                event_type: "tool_start".to_string(),
                tool_class: "web_search".to_string(),
                status: None,
            },
            ToolEntry {
                session_id: "workflow-a:agent".to_string(),
                event_time_ms: 1_300,
                event_type: "tool_error".to_string(),
                tool_class: "web_search".to_string(),
                status: Some("error".to_string()),
            },
        ];

        let (_, entries) = build_output_entries(requests, tools).unwrap();

        assert_eq!(entries[1].sidecar.tool_gap.event_count, 2);
        assert_eq!(entries[1].sidecar.tool_gap.error_count, 1);
        assert_eq!(entries[1].sidecar.tool_gap.by_class["web_search"], 2);
    }
}

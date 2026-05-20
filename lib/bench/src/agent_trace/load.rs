// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-format records and JSONL/gz loader.
//!
//! Local subset of `dynamo.agent.trace.v1`; the canonical producer-side schema
//! lives in `lib/llm/src/agents/trace/types.rs`.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::MultiGzDecoder;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentTraceRecord {
    pub(crate) event_type: String,
    pub(crate) event_time_unix_ms: u64,
    #[serde(default)]
    pub(crate) agent_context: Option<AgentContextFields>,
    #[serde(default)]
    pub(crate) request: Option<AgentRequestMetrics>,
    #[serde(default)]
    pub(crate) tool: Option<AgentToolEventMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentContextFields {
    pub(crate) trajectory_id: String,
    #[serde(default)]
    pub(crate) parent_trajectory_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentRequestMetrics {
    pub(crate) request_id: String,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) request_received_ms: Option<u64>,
    #[serde(default)]
    pub(crate) total_time_ms: Option<f64>,
    #[serde(default)]
    pub(crate) replay: Option<AgentReplayMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentReplayMetrics {
    pub(crate) trace_block_size: usize,
    pub(crate) input_length: usize,
    pub(crate) input_sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentToolEventMetrics {
    #[serde(default)]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default)]
    pub(crate) tool_class: Option<String>,
    #[serde(default)]
    pub(crate) started_at_unix_ms: Option<u64>,
    #[serde(default)]
    pub(crate) ended_at_unix_ms: Option<u64>,
    #[serde(default)]
    pub(crate) duration_ms: Option<f64>,
    #[serde(default)]
    pub(crate) status: Option<String>,
    #[serde(default)]
    pub(crate) output_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) error_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RequestEntry {
    pub(crate) start_ms: i64,
    pub(crate) end_ms: i64,
    pub(crate) agent_context: Option<AgentContextFields>,
    pub(crate) request: AgentRequestMetrics,
    pub(crate) replay: AgentReplayMetrics,
}

#[derive(Debug, Clone)]
pub struct ToolEntry {
    pub(crate) trajectory_id: String,
    pub(crate) start_ms: i64,
    pub(crate) end_ms: i64,
    pub(crate) tool_call_id: String,
    pub(crate) tool_class: String,
    pub(crate) status: String,
    pub(crate) duration_ms: f64,
    pub(crate) output_bytes: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) error_type: Option<String>,
    pub(crate) terminal_event: String,
}

#[derive(Debug, Default)]
pub struct LoadedAgentTrace {
    pub requests: Vec<RequestEntry>,
    pub tools: Vec<ToolEntry>,
}

/// Records other than `request_end` / `tool_end` / `tool_error` are skipped.
/// Errors if no `request_end` rows were found.
pub fn load_agent_trace_records(paths: &[PathBuf]) -> Result<LoadedAgentTrace> {
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
            } else if matches!(record.event_type.as_str(), "tool_end" | "tool_error") {
                let terminal_event = record.event_type.clone();
                if let Some(tool) = tool_entry(record, terminal_event) {
                    loaded.tools.push(tool);
                }
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

fn tool_entry(record: AgentTraceRecord, terminal_event: String) -> Option<ToolEntry> {
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
    let duration_ms = tool
        .duration_ms
        .unwrap_or_else(|| (end_ms - start_ms).max(0) as f64);
    let status = tool.status.unwrap_or_else(|| {
        if terminal_event == "tool_error" {
            "error".to_string()
        } else {
            "succeeded".to_string()
        }
    });
    Some(ToolEntry {
        trajectory_id: context.trajectory_id,
        start_ms,
        end_ms,
        tool_call_id: tool.tool_call_id.unwrap_or_default(),
        tool_class: tool.tool_class.unwrap_or_default(),
        status,
        duration_ms,
        output_bytes: tool.output_bytes,
        output_tokens: tool.output_tokens,
        error_type: tool.error_type,
        terminal_event,
    })
}

pub(crate) fn request_times(event_time_unix_ms: u64, request: &AgentRequestMetrics) -> (i64, i64) {
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

pub(crate) fn saturating_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

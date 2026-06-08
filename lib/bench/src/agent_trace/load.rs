// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-format records and JSONL/gz loader.
//!
//! Local subset of the supported Dynamo request trace schemas. Canonical
//! producer-side schemas live in `lib/llm/src/agents/trace/types.rs` and
//! `lib/llm/src/request_trace/types.rs`.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::MultiGzDecoder;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentTraceRecord {
    pub(crate) schema: TraceSchema,
    pub(crate) event_type: String,
    pub(crate) event_time_unix_ms: u64,
    #[serde(default)]
    pub(crate) agent_context: Option<AgentContextFields>,
    #[serde(default)]
    pub(crate) request: Option<AgentRequestMetrics>,
    #[serde(default)]
    pub(crate) tool: Option<AgentToolEventMetrics>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub(crate) enum TraceSchema {
    #[serde(rename = "dynamo.agent.trace.v1")]
    AgentV1,
    #[serde(rename = "dynamo.request.trace.v1")]
    RequestV1,
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
    pub contains_request_trace: bool,
}

impl LoadedAgentTrace {
    pub fn ensure_agentic_compatible(&self) -> Result<()> {
        if self.contains_request_trace {
            bail!(
                "dynamo.request.trace.v1 records do not contain agent context and cannot be converted with --agentic"
            );
        }
        Ok(())
    }
}

/// Records other than `request_end` / `tool_end` / `tool_error` are skipped.
/// Errors if no `request_end` rows were found.
pub fn load_agent_trace_records(paths: &[PathBuf]) -> Result<LoadedAgentTrace> {
    let mut loaded = LoadedAgentTrace::default();
    let mut request_ids = HashSet::new();

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
            if record.schema == TraceSchema::RequestV1 {
                loaded.contains_request_trace = true;
                if record.event_type != "request_end" {
                    bail!(
                        "request trace schema only supports request_end, got {} at {}:{}",
                        record.event_type,
                        path.display(),
                        line_index + 1
                    );
                }
            }
            if record.event_type == "request_end" {
                let entry = request_entry(record).with_context(|| {
                    format!(
                        "invalid request_end at {}:{}",
                        path.display(),
                        line_index + 1
                    )
                })?;
                if !request_ids.insert(entry.request.request_id.clone()) {
                    bail!(
                        "duplicate request_id {} at {}:{}",
                        entry.request.request_id,
                        path.display(),
                        line_index + 1
                    );
                }
                loaded.requests.push(entry);
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
    let schema = record.schema;
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
    let expected_hashes = replay.input_length.div_ceil(replay.trace_block_size);
    if replay.input_sequence_hashes.len() != expected_hashes {
        bail!(
            "input_length {} with trace_block_size {} requires exactly {} replay hashes, got {}",
            replay.input_length,
            replay.trace_block_size,
            expected_hashes,
            replay.input_sequence_hashes.len()
        );
    }
    if schema == TraceSchema::RequestV1 {
        if request.request_received_ms.is_none() {
            bail!("request trace is missing request_received_ms");
        }
        if request.output_tokens.is_none() {
            bail!("request trace is missing output_tokens");
        }
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
    if record.schema != TraceSchema::AgentV1 {
        return None;
    }
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
    use std::io::Write;

    use tempfile::NamedTempFile;

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

    #[test]
    fn loads_context_free_request_trace() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":3,"input_sequence_hashes":[11,22]}}}}}}"#
        )
        .unwrap();

        let loaded = load_agent_trace_records(&[file.path().to_path_buf()]).unwrap();
        assert!(loaded.contains_request_trace);
        assert_eq!(loaded.requests.len(), 1);
        assert!(loaded.requests[0].agent_context.is_none());
        assert_eq!(loaded.requests[0].start_ms, 1_000);
        assert_eq!(loaded.requests[0].end_ms, 1_100);
    }

    #[test]
    fn rejects_unknown_trace_schema() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v2","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[11]}}}}}}"#
        )
        .unwrap();

        let error = load_agent_trace_records(&[file.path().to_path_buf()]).unwrap_err();
        assert!(error.to_string().contains("failed to parse"));
        assert!(format!("{error:#}").contains("unknown variant `dynamo.request.trace.v2`"));
    }

    #[test]
    fn request_trace_requires_arrival_time_and_output_length() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","replay":{{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[11]}}}}}}"#
        )
        .unwrap();

        let error = load_agent_trace_records(&[file.path().to_path_buf()]).unwrap_err();
        assert!(format!("{error:#}").contains("request trace is missing request_received_ms"));
    }

    #[test]
    fn rejects_duplicate_request_ids() {
        let mut file = NamedTempFile::new().unwrap();
        for _ in 0..2 {
            writeln!(
                file,
                r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[11]}}}}}}"#
            )
            .unwrap();
        }

        let error = load_agent_trace_records(&[file.path().to_path_buf()]).unwrap_err();
        assert!(error.to_string().contains("duplicate request_id req-1"));
    }

    #[test]
    fn rejects_extra_replay_hashes() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[11,22]}}}}}}"#
        )
        .unwrap();

        let error = load_agent_trace_records(&[file.path().to_path_buf()]).unwrap_err();
        assert!(format!("{error:#}").contains("requires exactly 1 replay hashes, got 2"));
    }

    #[test]
    fn request_trace_is_rejected_for_agentic_conversion() {
        let loaded = LoadedAgentTrace {
            requests: Vec::new(),
            tools: Vec::new(),
            contains_request_trace: true,
        };

        let error = loaded.ensure_agentic_compatible().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot be converted with --agentic")
        );
    }
}

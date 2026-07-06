// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-format records and JSONL/gz loader.
//!
//! Local subset of Dynamo request trace rows consumed by replay.

use std::collections::HashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use flate2::read::MultiGzDecoder;
use serde::Deserialize;
use serde::de::IgnoredAny;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RequestTraceRecord {
    pub(crate) schema: TraceSchema,
    pub(crate) event_type: String,
    pub(crate) event_time_unix_ms: u64,
    #[serde(default)]
    pub(crate) agent_context: Option<AgentContextFields>,
    #[serde(default)]
    pub(crate) request: Option<RequestTraceRequestMetrics>,
    #[serde(default)]
    pub(crate) tool: Option<RequestTraceToolEventMetrics>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub(crate) enum TraceSchema {
    #[serde(rename = "dynamo.request.trace.v1")]
    RequestV1,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AgentContextFields {
    pub(crate) session_id: String,
    #[serde(default)]
    pub(crate) parent_session_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RequestTraceRequestMetrics {
    pub(crate) request_id: String,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) request_received_ms: Option<u64>,
    #[serde(default)]
    pub(crate) total_time_ms: Option<f64>,
    #[serde(default)]
    pub(crate) replay: Option<RequestTraceReplayMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RequestTraceReplayMetrics {
    pub(crate) trace_block_size: usize,
    pub(crate) input_length: usize,
    pub(crate) input_sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RequestTraceToolEventMetrics {
    pub(crate) tool_call_id: String,
    pub(crate) tool_class: String,
    #[serde(default)]
    pub(crate) claude: Option<ClaudeToolReplayMetrics>,
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

/// Claude-only evidence used to disambiguate an offline replay DAG.
///
/// A completed session reveals the future request that consumed a tool result;
/// live ZMQ tool producers do not have that information. Missing metadata is
/// expected and leaves agentic lowering on its timestamp-based fallback.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ClaudeToolReplayMetrics {
    /// Request that emitted the tool call.
    pub(crate) source_request_id: String,
    /// Later request that consumed the terminal result.
    #[serde(default)]
    pub(crate) consumer_request_id: Option<String>,
    /// Child agent session launched by the tool, when present in the export.
    #[serde(default)]
    pub(crate) child_session_id: Option<String>,
    /// Whether parent execution blocked or continued in the background.
    pub(crate) execution_mode: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum JsonLineEnvelope {
    Object(TraceRecordEnvelope),
    Other(IgnoredAny),
}

#[derive(Debug, Deserialize)]
struct TraceRecordEnvelope {
    #[serde(default)]
    event_type: Option<String>,
    #[serde(default)]
    event: Option<TraceEventEnvelope>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TraceEventEnvelope {
    Object(TraceEventFields),
    Other(IgnoredAny),
}

#[derive(Debug, Deserialize)]
struct TraceEventFields {
    #[serde(default)]
    event_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WrappedTraceRecord {
    event: RequestTraceRecord,
}

#[derive(Debug, Clone)]
pub struct RequestEntry {
    pub(crate) start_ms: i64,
    pub(crate) end_ms: i64,
    pub(crate) agent_context: Option<AgentContextFields>,
    pub(crate) request: RequestTraceRequestMetrics,
    pub(crate) replay: RequestTraceReplayMetrics,
}

#[derive(Debug, Clone)]
pub struct ToolEntry {
    pub(crate) session_id: String,
    pub(crate) start_ms: i64,
    pub(crate) end_ms: i64,
    pub(crate) tool_call_id: String,
    pub(crate) tool_class: String,
    pub(crate) claude: Option<ClaudeToolReplayMetrics>,
    pub(crate) status: String,
    pub(crate) duration_ms: f64,
    pub(crate) output_bytes: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) error_type: Option<String>,
}

#[derive(Debug, Default)]
pub struct LoadedAgentTrace {
    pub requests: Vec<RequestEntry>,
    pub tools: Vec<ToolEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestTraceMode {
    Standard,
    Agentic,
}

impl LoadedAgentTrace {
    pub fn mode(&self) -> Result<RequestTraceMode> {
        if self.requests.is_empty() {
            bail!("Dynamo request trace contains no requests");
        }
        let contextual_requests = self
            .requests
            .iter()
            .filter(|request| request.agent_context.is_some())
            .count();
        match contextual_requests {
            0 => Ok(RequestTraceMode::Standard),
            count if count == self.requests.len() => Ok(RequestTraceMode::Agentic),
            _ => bail!("Dynamo request trace cannot mix requests with and without agent_context"),
        }
    }

    pub fn ensure_agentic_compatible(&self) -> Result<()> {
        if self
            .requests
            .iter()
            .any(|request| request.agent_context.is_none())
        {
            bail!("agentic lowering requires agent_context on every request");
        }
        Ok(())
    }
}

/// `request_payload` records are skipped; replay consumes `request_end` and
/// terminal tool events only. Errors if no `request_end` rows were found.
pub fn load_request_trace_records(paths: &[PathBuf]) -> Result<LoadedAgentTrace> {
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
            let _schema = record.schema;
            if record.event_type == "request_payload" {
                continue;
            }
            if !matches!(
                record.event_type.as_str(),
                "request_end" | "tool_start" | "tool_end" | "tool_error"
            ) {
                bail!(
                    "request trace schema only supports request_end/tool_* and request_payload events, got {} at {}:{}",
                    record.event_type,
                    path.display(),
                    line_index + 1
                );
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

fn parse_trace_record(line: &str) -> Result<Option<RequestTraceRecord>> {
    let envelope = match serde_json::from_str::<JsonLineEnvelope>(line)? {
        JsonLineEnvelope::Object(envelope) => envelope,
        JsonLineEnvelope::Other(_) => return Ok(None),
    };

    match envelope.event {
        Some(TraceEventEnvelope::Object(event)) => {
            if event.event_type.as_deref() == Some("request_payload") {
                return Ok(None);
            }
            Ok(Some(
                serde_json::from_str::<WrappedTraceRecord>(line)?.event,
            ))
        }
        Some(TraceEventEnvelope::Other(_)) => Ok(None),
        None => {
            if envelope.event_type.as_deref() == Some("request_payload") {
                return Ok(None);
            }
            Ok(Some(serde_json::from_str(line)?))
        }
    }
}

fn request_entry(record: RequestTraceRecord) -> Result<RequestEntry> {
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
    if request.request_received_ms.is_none() {
        bail!("request trace is missing request_received_ms");
    }
    if request.output_tokens.is_none() {
        bail!("request trace is missing output_tokens");
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

fn tool_entry(record: RequestTraceRecord, terminal_event: String) -> Option<ToolEntry> {
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
        session_id: context.session_id,
        start_ms,
        end_ms,
        tool_call_id: tool.tool_call_id,
        tool_class: tool.tool_class,
        claude: tool.claude,
        status,
        duration_ms,
        output_bytes: tool.output_bytes,
        output_tokens: tool.output_tokens,
        error_type: tool.error_type,
    })
}

pub(crate) fn request_times(
    event_time_unix_ms: u64,
    request: &RequestTraceRequestMetrics,
) -> (i64, i64) {
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
        let request = RequestTraceRequestMetrics {
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

        let loaded = load_request_trace_records(&[file.path().to_path_buf()]).unwrap();
        assert_eq!(loaded.requests.len(), 1);
        assert!(loaded.requests[0].agent_context.is_none());
        assert_eq!(loaded.requests[0].start_ms, 1_000);
        assert_eq!(loaded.requests[0].end_ms, 1_100);
    }

    #[test]
    fn loads_wrapped_request_trace_record() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"timestamp":1,"event":{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":3,"input_sequence_hashes":[11,22]}}}}}}}}"#
        )
        .unwrap();

        let loaded = load_request_trace_records(&[file.path().to_path_buf()]).unwrap();
        assert_eq!(loaded.requests.len(), 1);
        assert_eq!(loaded.requests[0].request.request_id, "req-1");
    }

    #[test]
    fn skips_request_payload_records() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_payload","event_time_unix_ms":1050,"payload":{{"request_id":"req-1","endpoint":"openai.chat_completion","requested_streaming":false,"model":"test","request":{{"model":"test","messages":[{{"role":"user","content":"hi"}}]}},"payload_complete":true}}}}"#
        )
        .unwrap();
        let large_payload = "x".repeat(4096);
        writeln!(
            file,
            "{}",
            serde_json::json!({
                "timestamp": 1051,
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_payload",
                    "event_time_unix_ms": 1051,
                    "payload": {
                        "request_id": "req-1",
                        "endpoint": "openai.chat_completion",
                        "requested_streaming": false,
                        "model": "test",
                        "request": {
                            "model": "test",
                            "messages": [{
                                "role": "user",
                                "content": large_payload.clone(),
                            }],
                        },
                        "response": {
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": large_payload,
                                },
                            }],
                        },
                        "payload_complete": true,
                    },
                },
            })
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":3,"input_sequence_hashes":[11,22]}}}}}}"#
        )
        .unwrap();

        let loaded = load_request_trace_records(&[file.path().to_path_buf()]).unwrap();
        assert_eq!(loaded.requests.len(), 1);
        assert_eq!(loaded.requests[0].request.request_id, "req-1");
    }

    #[test]
    fn rejects_unknown_trace_schema() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v2","event_type":"request_end","event_time_unix_ms":1100,"request":{{"request_id":"req-1","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":2,"input_sequence_hashes":[11]}}}}}}"#
        )
        .unwrap();

        let error = load_request_trace_records(&[file.path().to_path_buf()]).unwrap_err();
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

        let error = load_request_trace_records(&[file.path().to_path_buf()]).unwrap_err();
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

        let error = load_request_trace_records(&[file.path().to_path_buf()]).unwrap_err();
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

        let error = load_request_trace_records(&[file.path().to_path_buf()]).unwrap_err();
        assert!(format!("{error:#}").contains("requires exactly 1 replay hashes, got 2"));
    }

    #[test]
    fn loads_agentic_request_trace_rows_and_tool_events() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"event_source":"dynamo","agent_context":{{"session_id":"root"}},"request":{{"request_id":"req-1","model":"test","request_received_ms":1000,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":3,"input_sequence_hashes":[11,22]}}}}}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"tool_end","event_time_unix_ms":1200,"event_source":"harness","agent_context":{{"session_id":"root"}},"tool":{{"tool_call_id":"tool-1","tool_class":"search","claude":{{"source_request_id":"req-1","consumer_request_id":"req-2","child_session_id":"child","execution_mode":"background"}},"started_at_unix_ms":1110,"ended_at_unix_ms":1200,"status":"succeeded","duration_ms":90}}}}"#
        )
        .unwrap();

        let loaded = load_request_trace_records(&[file.path().to_path_buf()]).unwrap();
        loaded.ensure_agentic_compatible().unwrap();
        assert_eq!(loaded.requests.len(), 1);
        assert_eq!(
            loaded.requests[0]
                .agent_context
                .as_ref()
                .expect("agent context")
                .session_id,
            "root"
        );
        assert_eq!(loaded.tools.len(), 1);
        assert_eq!(loaded.tools[0].tool_call_id, "tool-1");
        assert_eq!(loaded.tools[0].tool_class, "search");
        let claude = loaded.tools[0].claude.as_ref().expect("Claude metadata");
        assert_eq!(claude.source_request_id, "req-1");
        assert_eq!(claude.consumer_request_id.as_deref(), Some("req-2"));
        assert_eq!(claude.child_session_id.as_deref(), Some("child"));
        assert_eq!(claude.execution_mode, "background");
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trace replay utilities for stateful responses
//!
//! Parses Braintrust-style JSONL trace files and replays them through
//! the storage system to verify conversation handling.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use super::{ResponseStorage, StorageError};

/// A parsed span from a Braintrust trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub id: String,
    pub span_id: String,
    pub root_span_id: String,
    #[serde(default)]
    pub span_parents: Vec<String>,
    pub created: String,
    #[serde(default)]
    pub input: Value,
    #[serde(default)]
    pub output: Option<Value>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    #[serde(default)]
    pub span_attributes: HashMap<String, Value>,
    #[serde(default)]
    pub metrics: HashMap<String, Value>,
}

/// A conversation turn extracted from trace spans
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub turn_id: String,
    pub turn_number: usize,
    pub user_input: String,
    pub assistant_output: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub timestamp: String,
}

/// A tool call within a turn
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub tool_name: String,
    pub input: Value,
    pub output: Option<Value>,
}

/// Parsed trace with session info and turns
#[derive(Debug)]
pub struct ParsedTrace {
    pub session_id: String,
    pub root_span_id: String,
    pub turns: Vec<ConversationTurn>,
    pub raw_spans: Vec<TraceSpan>,
}

/// Parse a Braintrust JSONL trace file
pub fn parse_trace_file(path: &Path) -> Result<ParsedTrace, TraceParseError> {
    let content =
        std::fs::read_to_string(path).map_err(|e| TraceParseError::IoError(e.to_string()))?;
    parse_trace_content(&content)
}

/// Parse trace content from a string
pub fn parse_trace_content(content: &str) -> Result<ParsedTrace, TraceParseError> {
    let mut spans: Vec<TraceSpan> = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let span: TraceSpan = serde_json::from_str(line)
            .map_err(|e| TraceParseError::JsonError(line_num + 1, e.to_string()))?;
        spans.push(span);
    }

    if spans.is_empty() {
        return Err(TraceParseError::EmptyTrace);
    }

    // Find root span (the one with no parents or self-referencing)
    let root_span = spans
        .iter()
        .find(|s| s.span_parents.is_empty() || s.span_parents.contains(&s.span_id))
        .ok_or(TraceParseError::NoRootSpan)?;

    let root_span_id = root_span.span_id.clone();

    // Extract session_id from root span metadata
    let session_id = root_span
        .metadata
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| root_span_id.clone());

    // Find turn spans (direct children of root with type "task" and name starting with "Turn")
    let mut turns: Vec<ConversationTurn> = Vec::new();
    let mut turn_number = 0;

    for span in &spans {
        // Check if this is a turn span
        let is_turn = span.span_parents.contains(&root_span_id)
            && span.span_attributes.get("type").and_then(|v| v.as_str()) == Some("task")
            && span
                .span_attributes
                .get("name")
                .and_then(|v| v.as_str())
                .map(|n| n.starts_with("Turn"))
                .unwrap_or(false);

        if is_turn {
            turn_number += 1;

            // Extract user input
            let user_input = match &span.input {
                Value::String(s) => s.clone(),
                Value::Array(arr) => {
                    // Find user message in array
                    arr.iter()
                        .filter_map(|item| {
                            if item.get("role")?.as_str()? == "user" {
                                item.get("content")?.as_str().map(|s| s.to_string())
                            } else {
                                None
                            }
                        })
                        .next()
                        .unwrap_or_default()
                }
                _ => String::new(),
            };

            // Find LLM spans that are children of this turn to get output
            let assistant_output = spans
                .iter()
                .filter(|s| s.span_parents.contains(&span.span_id))
                .filter(|s| s.span_attributes.get("type").and_then(|v| v.as_str()) == Some("llm"))
                .filter_map(|s| {
                    s.output.as_ref().and_then(|o| {
                        o.get("content")
                            .and_then(|c| c.as_str())
                            .map(|s| s.to_string())
                    })
                })
                .last();

            // Find tool calls
            let tool_calls: Vec<ToolCall> = spans
                .iter()
                .filter(|s| s.span_parents.contains(&span.span_id))
                .filter(|s| s.span_attributes.get("type").and_then(|v| v.as_str()) == Some("tool"))
                .map(|s| {
                    let tool_name = s
                        .metadata
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    ToolCall {
                        tool_name,
                        input: s.input.clone(),
                        output: s.output.clone(),
                    }
                })
                .collect();

            turns.push(ConversationTurn {
                turn_id: span.span_id.clone(),
                turn_number,
                user_input,
                assistant_output,
                tool_calls,
                timestamp: span.created.clone(),
            });
        }
    }

    // Sort turns by number
    turns.sort_by_key(|t| t.turn_number);

    Ok(ParsedTrace {
        session_id,
        root_span_id,
        turns,
        raw_spans: spans,
    })
}

/// Replay a parsed trace through the storage system
///
/// This simulates what would happen if the same conversation
/// went through our Responses API with stateful storage.
pub async fn replay_trace<S: ResponseStorage>(
    storage: &S,
    trace: &ParsedTrace,
    tenant_id: &str,
) -> Result<ReplayResult, StorageError> {
    let mut response_ids: Vec<String> = Vec::new();
    let mut previous_response_id: Option<String> = None;

    for turn in &trace.turns {
        // Build a response object similar to what our API would store
        let response_data = serde_json::json!({
            "id": format!("resp_{}", turn.turn_id),
            "object": "response",
            "created_at": turn.timestamp,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": turn.assistant_output.clone().unwrap_or_default()
                }
            ],
            "metadata": {
                "turn_number": turn.turn_number,
                "user_input": turn.user_input,
                "tool_calls_count": turn.tool_calls.len(),
                "previous_response_id": previous_response_id.clone()
            }
        });

        // Store with the turn_id as response_id for deterministic replay
        let response_id = storage
            .store_response(
                tenant_id,
                &trace.session_id,
                Some(&format!("resp_{}", turn.turn_id)),
                response_data,
                None,
            )
            .await?;

        response_ids.push(response_id.clone());
        previous_response_id = Some(response_id);
    }

    Ok(ReplayResult {
        session_id: trace.session_id.clone(),
        tenant_id: tenant_id.to_string(),
        turns_replayed: trace.turns.len(),
        response_ids,
    })
}

/// Result of replaying a trace
#[derive(Debug)]
pub struct ReplayResult {
    pub session_id: String,
    pub tenant_id: String,
    pub turns_replayed: usize,
    pub response_ids: Vec<String>,
}

/// Errors that can occur during trace parsing
#[derive(Debug, thiserror::Error)]
pub enum TraceParseError {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("JSON parse error on line {0}: {1}")]
    JsonError(usize, String),

    #[error("Empty trace file")]
    EmptyTrace,

    #[error("No root span found")]
    NoRootSpan,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TRACE: &str = r#"{"id": "root-123", "span_id": "root-123", "root_span_id": "root-123", "created": "2026-02-03T02:52:03.000Z", "input": "Session: test", "metadata": {"session_id": "root-123"}, "span_attributes": {"name": "Test Session", "type": "task"}}
{"id": "turn-1", "span_id": "turn-1", "root_span_id": "root-123", "span_parents": ["root-123"], "created": "2026-02-03T02:52:04.000Z", "input": "Hello, how are you?", "span_attributes": {"name": "Turn 1", "type": "task"}}
{"id": "llm-1", "span_id": "llm-1", "root_span_id": "root-123", "span_parents": ["turn-1"], "created": "2026-02-03T02:52:05.000Z", "input": [{"role": "user", "content": "Hello"}], "output": {"role": "assistant", "content": "I'm doing well!"}, "span_attributes": {"name": "llm", "type": "llm"}}"#;

    #[test]
    fn test_parse_trace_content() {
        let result = parse_trace_content(SAMPLE_TRACE);
        assert!(result.is_ok());

        let trace = result.unwrap();
        assert_eq!(trace.session_id, "root-123");
        assert_eq!(trace.turns.len(), 1);
        assert_eq!(trace.turns[0].user_input, "Hello, how are you?");
        assert_eq!(
            trace.turns[0].assistant_output,
            Some("I'm doing well!".to_string())
        );
    }

    #[test]
    fn test_parse_empty_trace() {
        let result = parse_trace_content("");
        assert!(matches!(result, Err(TraceParseError::EmptyTrace)));
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = parse_trace_content("not valid json");
        assert!(matches!(result, Err(TraceParseError::JsonError(1, _))));
    }
}

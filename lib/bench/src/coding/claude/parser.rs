// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::coding::common::{
    anonymized_session_id, canonical_json_string, content_blocks, flatten_block_content_text,
    object_field, parse_utc_timestamp_ms,
};
use crate::coding::tokenizer::TokenizerWorker;
use anyhow::Result;
use rustc_hash::FxHashMap;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct TraceRecord {
    pub session_id: String,
    pub row_type: String,
    pub timestamp_ms: i64,
    pub source_order: u64,
    pub raw: Value,
}

#[derive(Clone, Debug)]
struct ConversationEntry {
    kind: String,
    rendered: String,
}

#[derive(Clone, Debug)]
struct ToolCallSummary {
    name: String,
    normalized_id: Option<String>,
    arg_size_chars: usize,
}

#[derive(Clone, Debug)]
struct CachedProgressMetrics {
    progress_event_count: usize,
    agent_ids: BTreeSet<String>,
    assistant_text_blocks: usize,
    tool_counts: BTreeMap<String, usize>,
    tool_result_count: usize,
    tool_error_count: usize,
    tool_intervals: Vec<(i64, i64)>,
    first_ts: i64,
    last_ts: i64,
}

#[derive(Debug)]
struct AssistantGroupSummary {
    entries: Vec<ConversationEntry>,
    output_length: usize,
    assistant_text_blocks: usize,
    top_level_tool_calls: Vec<ToolCallSummary>,
    raw_task_tool_ids: Vec<String>,
    start_ms: i64,
    end_ms: i64,
}

#[derive(Clone, Debug)]
pub struct TurnDraft {
    pub session_id: String,
    pub export_session_id: String,
    pub turn_index: usize,
    pub input_text: String,
    pub output_length: usize,
    pub assistant_start_ms: i64,
    pub assistant_end_ms: i64,
    pub delay_ms: Option<i64>,
    pub sidecar: Value,
}

#[derive(Debug, Default)]
struct ToolIdNormalizer {
    raw_to_normalized: FxHashMap<String, String>,
}

impl ToolIdNormalizer {
    fn normalize(&mut self, raw_id: Option<&str>) -> Option<String> {
        let raw_id = raw_id?;
        if let Some(existing) = self.raw_to_normalized.get(raw_id) {
            return Some(existing.clone());
        }
        let next_id = self.raw_to_normalized.len() + 1;
        let normalized = format!("tool_{next_id:04}");
        self.raw_to_normalized
            .insert(raw_id.to_string(), normalized.clone());
        Some(normalized)
    }
}

#[derive(Debug)]
pub struct SessionTurnBuilder {
    session_id: String,
    export_session_id: String,
    records: Vec<TraceRecord>,
    top_level_indices: Vec<usize>,
    progress_metrics_index: FxHashMap<String, CachedProgressMetrics>,
    top_level_cursor: usize,
    normalizer: ToolIdNormalizer,
    conversation_entries: Vec<ConversationEntry>,
    prompt_text: String,
    previous_assistant_end_ms: Option<i64>,
    turn_index: usize,
    pending_compact_reset: bool,
}

impl SessionTurnBuilder {
    pub fn new(session_id: String, records: Vec<TraceRecord>, preserve_session_ids: bool) -> Self {
        let export_session_id = if preserve_session_ids {
            session_id.clone()
        } else {
            anonymized_session_id(&session_id)
        };

        let progress_index = build_progress_index(&records);
        let progress_metrics_index = build_progress_metrics_index(&progress_index, &records);
        let top_level_indices = records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                let is_top_level =
                    matches!(record.row_type.as_str(), "user" | "assistant" | "system")
                        && !record
                            .raw
                            .get("isSidechain")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                is_top_level.then_some(index)
            })
            .collect();

        Self {
            session_id,
            export_session_id,
            records,
            top_level_indices,
            progress_metrics_index,
            top_level_cursor: 0,
            normalizer: ToolIdNormalizer::default(),
            conversation_entries: Vec::new(),
            prompt_text: String::new(),
            previous_assistant_end_ms: None,
            turn_index: 0,
            pending_compact_reset: false,
        }
    }

    pub fn next_turn(&mut self, tokenizer: &mut impl TokenizerWorker) -> Result<Option<TurnDraft>> {
        while self.top_level_cursor < self.top_level_indices.len() {
            let record_index = self.top_level_indices[self.top_level_cursor];
            let record = &self.records[record_index];

            if record.row_type == "system" {
                self.pending_compact_reset = is_compact_boundary(record);
                self.top_level_cursor += 1;
                continue;
            }

            if record.row_type == "user" {
                if should_skip_user_record(record)? {
                    self.pending_compact_reset = false;
                    self.top_level_cursor += 1;
                    continue;
                }

                let message = object_field(&record.raw, "message");
                let rendered_entries = render_user_entries(message, &mut self.normalizer)?;
                if self.pending_compact_reset && is_compact_summary(record) {
                    self.replace_conversation_entries(rendered_entries);
                } else {
                    self.extend_conversation_entries(rendered_entries);
                }
                self.pending_compact_reset = false;
                self.top_level_cursor += 1;
                continue;
            }

            self.pending_compact_reset = false;
            let group_key = assistant_group_key(record);
            let mut group_indices = vec![record_index];
            self.top_level_cursor += 1;
            while self.top_level_cursor < self.top_level_indices.len() {
                let next_index = self.top_level_indices[self.top_level_cursor];
                let next_record = &self.records[next_index];
                if next_record.row_type != "assistant" {
                    break;
                }
                if assistant_group_key(next_record) != group_key {
                    break;
                }
                group_indices.push(next_index);
                self.top_level_cursor += 1;
            }

            let group_summary = summarize_assistant_group(
                &self.records,
                &group_indices,
                &mut self.normalizer,
                tokenizer,
            )?;
            let input_text = self.prompt_text.clone();

            let top_level_tool_names = group_summary
                .top_level_tool_calls
                .iter()
                .map(|tool_call| tool_call.name.clone())
                .collect::<Vec<_>>();
            let used_task_tool = top_level_tool_names.iter().any(|name| name == "Task");
            let top_level_tool_calls = group_summary
                .top_level_tool_calls
                .iter()
                .map(|tool_call| {
                    json!({
                        "name": tool_call.name,
                        "tool_id": tool_call.normalized_id,
                        "arg_size_chars": tool_call.arg_size_chars,
                    })
                })
                .collect::<Vec<_>>();

            let mut sidecar = Map::new();
            sidecar.insert(
                "session_id".to_string(),
                Value::String(self.export_session_id.clone()),
            );
            sidecar.insert("turn_index".to_string(), json!(self.turn_index));
            sidecar.insert(
                "num_messages_in_context".to_string(),
                json!(self.conversation_entries.len()),
            );
            sidecar.insert(
                "context_shape".to_string(),
                Value::Array(
                    self.conversation_entries
                        .iter()
                        .map(|entry| Value::String(entry.kind.clone()))
                        .collect(),
                ),
            );
            sidecar.insert(
                "tool_rounds_before_answer".to_string(),
                json!(count_trailing_tool_results(&self.conversation_entries)),
            );
            sidecar.insert("used_task_tool".to_string(), Value::Bool(used_task_tool));
            sidecar.insert(
                "assistant_text_blocks".to_string(),
                json!(group_summary.assistant_text_blocks),
            );
            sidecar.insert(
                "top_level_tool_call_count".to_string(),
                json!(group_summary.top_level_tool_calls.len()),
            );
            sidecar.insert(
                "top_level_tool_names".to_string(),
                Value::Array(
                    top_level_tool_names
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
            sidecar.insert(
                "top_level_tool_calls".to_string(),
                Value::Array(top_level_tool_calls),
            );

            let progress_metrics = aggregate_progress_metrics(
                &group_summary.raw_task_tool_ids,
                &self.progress_metrics_index,
                &mut self.normalizer,
            );
            if let Some(progress_map) = progress_metrics.as_object() {
                for (key, value) in progress_map {
                    sidecar.insert(key.clone(), value.clone());
                }
            }

            let turn = TurnDraft {
                session_id: self.session_id.clone(),
                export_session_id: self.export_session_id.clone(),
                turn_index: self.turn_index,
                input_text,
                output_length: group_summary.output_length,
                assistant_start_ms: group_summary.start_ms,
                assistant_end_ms: group_summary.end_ms,
                delay_ms: self
                    .previous_assistant_end_ms
                    .map(|previous_end| (group_summary.start_ms - previous_end).max(0)),
                sidecar: Value::Object(sidecar),
            };

            self.extend_conversation_entries(group_summary.entries);
            self.previous_assistant_end_ms = Some(group_summary.end_ms);
            self.turn_index += 1;
            return Ok(Some(turn));
        }

        Ok(None)
    }

    fn replace_conversation_entries(&mut self, entries: Vec<ConversationEntry>) {
        self.prompt_text = render_entry_buffer(&entries);
        self.conversation_entries = entries;
    }

    fn extend_conversation_entries(&mut self, entries: Vec<ConversationEntry>) {
        append_rendered_entries(&mut self.prompt_text, &entries);
        self.conversation_entries.extend(entries);
    }
}

pub fn load_trace_records(trace_files: &[PathBuf]) -> Result<FxHashMap<String, Vec<TraceRecord>>> {
    let mut sessions: FxHashMap<String, Vec<TraceRecord>> = FxHashMap::default();
    let mut source_order = 0_u64;

    for trace_file in trace_files {
        let file = File::open(trace_file)?;
        let reader = BufReader::new(file);
        for (line_number, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let payload: Value = serde_json::from_str(&line).map_err(|error| {
                anyhow::anyhow!(
                    "invalid JSON in {}:{}: {}",
                    trace_file.display(),
                    line_number + 1,
                    error
                )
            })?;

            let session_id = payload
                .get("sessionId")
                .and_then(Value::as_str)
                .map(str::to_string);
            let row_type = payload
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string);
            let timestamp_raw = payload.get("timestamp").and_then(Value::as_str);

            let (Some(session_id), Some(row_type), Some(timestamp_raw)) =
                (session_id, row_type, timestamp_raw)
            else {
                source_order += 1;
                continue;
            };

            let Ok(timestamp_ms) = parse_utc_timestamp_ms(timestamp_raw) else {
                source_order += 1;
                continue;
            };

            sessions
                .entry(session_id.clone())
                .or_default()
                .push(TraceRecord {
                    session_id,
                    row_type,
                    timestamp_ms,
                    source_order,
                    raw: payload,
                });
            source_order += 1;
        }
    }

    for records in sessions.values_mut() {
        records.sort_by_key(|record| (record.timestamp_ms, record.source_order));
    }

    Ok(sessions)
}

fn assistant_group_key(record: &TraceRecord) -> String {
    if let Some(request_id) = record.raw.get("requestId").and_then(Value::as_str) {
        return request_id.to_string();
    }
    if let Some(message_id) = object_field(&record.raw, "message")
        .and_then(|message| message.get("id"))
        .and_then(Value::as_str)
    {
        return message_id.to_string();
    }
    if let Some(uuid) = record.raw.get("uuid").and_then(Value::as_str) {
        return uuid.to_string();
    }
    format!("row-{}", record.source_order)
}

fn is_compact_boundary(record: &TraceRecord) -> bool {
    record.row_type == "system"
        && record
            .raw
            .get("subtype")
            .and_then(Value::as_str)
            .map(|subtype| subtype == "compact_boundary")
            .unwrap_or(false)
}

fn is_compact_summary(record: &TraceRecord) -> bool {
    record.row_type == "user"
        && record
            .raw
            .get("isCompactSummary")
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

fn is_local_command_wrapper_text(text: &str) -> bool {
    let stripped = text.trim();
    [
        "<command-name>",
        "<command-message>",
        "<command-args>",
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
    ]
    .iter()
    .any(|prefix| stripped.starts_with(prefix))
}

fn should_skip_user_record(record: &TraceRecord) -> Result<bool> {
    if record.row_type != "user" {
        return Ok(false);
    }
    if record
        .raw
        .get("isMeta")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Ok(true);
    }

    let Some(message) = object_field(&record.raw, "message") else {
        return Ok(false);
    };
    let blocks = content_blocks(message.get("content"));
    if blocks.is_empty() {
        return Ok(false);
    }
    if blocks
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) != Some("text"))
    {
        return Ok(false);
    }

    let texts = blocks
        .iter()
        .map(|block| {
            block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect::<Vec<_>>();
    Ok(!texts.is_empty() && texts.iter().all(|text| is_local_command_wrapper_text(text)))
}

fn sanitize_structure(value: &Value, normalizer: &mut ToolIdNormalizer) -> Value {
    match value {
        Value::Object(map) => {
            let mut sanitized = Map::new();
            for (key, item) in map {
                if matches!(
                    key.as_str(),
                    "tool_use_id" | "toolUseID" | "parentToolUseID"
                ) && let Some(raw_id) = item.as_str()
                    && let Some(normalized) = normalizer.normalize(Some(raw_id))
                {
                    sanitized.insert(key.clone(), Value::String(normalized));
                    continue;
                }
                sanitized.insert(key.clone(), sanitize_structure(item, normalizer));
            }
            Value::Object(sanitized)
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| sanitize_structure(item, normalizer))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn count_trailing_tool_results(entries: &[ConversationEntry]) -> usize {
    entries
        .iter()
        .rev()
        .take_while(|entry| entry.kind == "user_tool_result")
        .count()
}

fn append_rendered_entries(buffer: &mut String, entries: &[ConversationEntry]) {
    for entry in entries {
        if !buffer.is_empty() {
            buffer.push('\n');
        }
        buffer.push_str(&entry.rendered);
    }
}

fn render_entry_buffer(entries: &[ConversationEntry]) -> String {
    let mut buffer = String::new();
    append_rendered_entries(&mut buffer, entries);
    buffer
}

fn render_user_entries(
    message: Option<&Map<String, Value>>,
    normalizer: &mut ToolIdNormalizer,
) -> Result<Vec<ConversationEntry>> {
    let mut rendered_entries = Vec::new();
    let Some(message) = message else {
        return Ok(rendered_entries);
    };

    for block in content_blocks(message.get("content")) {
        let block_type = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if matches!(block_type, "thinking" | "redacted_thinking") {
            continue;
        }
        if block_type == "text" {
            let text = block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !text.is_empty() {
                rendered_entries.push(ConversationEntry {
                    kind: "user_text".to_string(),
                    rendered: format!("[user] {text}"),
                });
            }
            continue;
        }
        if block_type == "tool_result" {
            let normalized_id =
                normalizer.normalize(block.get("tool_use_id").and_then(Value::as_str));
            let content_text =
                flatten_block_content_text(block.get("content").unwrap_or(&Value::Null))?;
            let is_error = block
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let header = format!(
                "[user_tool_result id={} error={}]",
                normalized_id.unwrap_or_else(|| "tool_unknown".to_string()),
                if is_error { "true" } else { "false" }
            );
            let rendered = if content_text.is_empty() {
                header
            } else {
                format!("{header} {content_text}")
            };
            rendered_entries.push(ConversationEntry {
                kind: "user_tool_result".to_string(),
                rendered,
            });
            continue;
        }

        let sanitized = sanitize_structure(&block, normalizer);
        rendered_entries.push(ConversationEntry {
            kind: "user_block".to_string(),
            rendered: format!(
                "[user_block type={block_type}] {}",
                canonical_json_string(&sanitized)?
            ),
        });
    }

    Ok(rendered_entries)
}

fn summarize_assistant_group(
    records: &[TraceRecord],
    group_indices: &[usize],
    normalizer: &mut ToolIdNormalizer,
    tokenizer: &mut impl TokenizerWorker,
) -> Result<AssistantGroupSummary> {
    let mut entries = Vec::new();
    let mut tool_calls = Vec::new();
    let mut raw_task_tool_ids = Vec::new();
    let mut assistant_text_blocks = 0;
    let mut output_lengths = Vec::new();

    for index in group_indices {
        let record = &records[*index];
        let Some(message) = object_field(&record.raw, "message") else {
            continue;
        };
        if let Some(output_tokens) = object_field(&Value::Object(message.clone()), "usage")
            .and_then(|usage| usage.get("output_tokens"))
            .and_then(Value::as_u64)
        {
            output_lengths.push(output_tokens as usize);
        }

        for block in content_blocks(message.get("content")) {
            let block_type = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            if matches!(block_type, "thinking" | "redacted_thinking") {
                continue;
            }
            if block_type == "text" {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !text.is_empty() {
                    assistant_text_blocks += 1;
                    entries.push(ConversationEntry {
                        kind: "assistant_text".to_string(),
                        rendered: format!("[assistant] {text}"),
                    });
                }
                continue;
            }
            if block_type == "tool_use" {
                let raw_id = block.get("id").and_then(Value::as_str);
                let normalized_id = normalizer.normalize(raw_id);
                let tool_name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let args_json = canonical_json_string(&sanitize_structure(
                    block.get("input").unwrap_or(&Value::Null),
                    normalizer,
                ))?;
                entries.push(ConversationEntry {
                    kind: "assistant_tool_use".to_string(),
                    rendered: format!(
                        "[assistant_tool_use id={} name={} args={}]",
                        normalized_id
                            .clone()
                            .unwrap_or_else(|| "tool_unknown".to_string()),
                        tool_name,
                        args_json
                    ),
                });
                tool_calls.push(ToolCallSummary {
                    name: tool_name.clone(),
                    normalized_id,
                    arg_size_chars: args_json.len(),
                });
                if tool_name == "Task"
                    && let Some(raw_id) = raw_id
                {
                    raw_task_tool_ids.push(raw_id.to_string());
                }
                continue;
            }

            let sanitized = sanitize_structure(&block, normalizer);
            entries.push(ConversationEntry {
                kind: "assistant_block".to_string(),
                rendered: format!(
                    "[assistant_block type={block_type}] {}",
                    canonical_json_string(&sanitized)?
                ),
            });
        }
    }

    let output_length = if let Some(max_length) = output_lengths.into_iter().max() {
        max_length
    } else {
        let rendered_text = render_entry_buffer(&entries);
        tokenizer.encode(&rendered_text)?.len()
    };

    let start_ms = group_indices
        .first()
        .map(|index| records[*index].timestamp_ms)
        .unwrap_or_default();
    let end_ms = group_indices
        .last()
        .map(|index| records[*index].timestamp_ms)
        .unwrap_or(start_ms);

    Ok(AssistantGroupSummary {
        entries,
        output_length,
        assistant_text_blocks,
        top_level_tool_calls: tool_calls,
        raw_task_tool_ids,
        start_ms,
        end_ms,
    })
}

fn progress_timestamp_ms(record: &TraceRecord) -> i64 {
    record
        .raw
        .get("data")
        .and_then(Value::as_object)
        .and_then(|data| data.get("message"))
        .and_then(Value::as_object)
        .and_then(|message| message.get("timestamp"))
        .and_then(Value::as_str)
        .and_then(|timestamp| parse_utc_timestamp_ms(timestamp).ok())
        .unwrap_or(record.timestamp_ms)
}

fn build_progress_index(records: &[TraceRecord]) -> FxHashMap<String, Vec<usize>> {
    let mut progress_index: FxHashMap<String, Vec<usize>> = FxHashMap::default();
    for (index, record) in records.iter().enumerate() {
        if !record.row_type.contains("progress") {
            continue;
        }
        let Some(parent_tool_use_id) = record.raw.get("parentToolUseID").and_then(Value::as_str)
        else {
            continue;
        };
        progress_index
            .entry(parent_tool_use_id.to_string())
            .or_default()
            .push(index);
    }

    for indices in progress_index.values_mut() {
        indices.sort_by_key(|index| {
            let record = &records[*index];
            (progress_timestamp_ms(record), record.source_order)
        });
    }
    progress_index
}

fn build_progress_metrics_index(
    progress_index: &FxHashMap<String, Vec<usize>>,
    records: &[TraceRecord],
) -> FxHashMap<String, CachedProgressMetrics> {
    progress_index
        .iter()
        .map(|(tool_id, indices)| {
            (
                tool_id.clone(),
                summarize_progress_indices(indices, records),
            )
        })
        .collect()
}

fn aggregate_progress_metrics(
    task_tool_ids: &[String],
    progress_metrics_index: &FxHashMap<String, CachedProgressMetrics>,
    normalizer: &mut ToolIdNormalizer,
) -> Value {
    let mut selected_metrics = Vec::new();
    let mut seen_task_ids = BTreeSet::new();
    for task_tool_id in task_tool_ids {
        if !seen_task_ids.insert(task_tool_id.as_str()) {
            continue;
        }
        if let Some(metrics) = progress_metrics_index.get(task_tool_id) {
            selected_metrics.push(metrics);
        }
    }

    if selected_metrics.is_empty() {
        return json!({
            "task_parent_tool_ids": task_tool_ids
                .iter()
                .filter_map(|tool_id| normalizer.normalize(Some(tool_id)))
                .collect::<Vec<_>>(),
            "nested_progress_event_count": 0,
            "nested_agent_count": 0,
            "nested_tool_call_count": 0,
            "nested_tool_result_count": 0,
            "nested_tool_error_count": 0,
            "nested_tool_counts": BTreeMap::<String, usize>::new(),
            "nested_tool_names": Vec::<String>::new(),
            "nested_tool_total_latency_ms": 0,
            "nested_tool_max_latency_ms": 0,
            "nested_tool_avg_latency_ms": 0,
            "nested_tool_max_parallelism": 0,
            "nested_assistant_text_blocks": 0,
            "task_duration_ms": 0,
        });
    }

    let mut agent_ids = BTreeSet::new();
    let mut assistant_text_blocks = 0_usize;
    let mut tool_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut tool_intervals = Vec::new();
    let mut tool_result_count = 0_usize;
    let mut tool_error_count = 0_usize;
    let mut first_ts = i64::MAX;
    let mut last_ts = i64::MIN;
    let mut progress_event_count = 0_usize;

    for metrics in selected_metrics {
        progress_event_count += metrics.progress_event_count;
        assistant_text_blocks += metrics.assistant_text_blocks;
        tool_result_count += metrics.tool_result_count;
        tool_error_count += metrics.tool_error_count;
        first_ts = first_ts.min(metrics.first_ts);
        last_ts = last_ts.max(metrics.last_ts);
        agent_ids.extend(metrics.agent_ids.iter().cloned());
        tool_intervals.extend(metrics.tool_intervals.iter().copied());
        for (tool_name, count) in &metrics.tool_counts {
            *tool_counts.entry(tool_name.clone()).or_insert(0) += count;
        }
    }

    let total_latency: i64 = tool_intervals
        .iter()
        .map(|(start, end)| (end - start).max(0))
        .sum();
    let max_latency = tool_intervals
        .iter()
        .map(|(start, end)| (end - start).max(0))
        .max()
        .unwrap_or(0);
    let avg_latency = if tool_intervals.is_empty() {
        0
    } else {
        total_latency / tool_intervals.len() as i64
    };

    let mut parallel_events = Vec::new();
    for (start_ms, end_ms) in &tool_intervals {
        parallel_events.push((*start_ms, 1_i32));
        parallel_events.push((*end_ms, -1_i32));
    }
    parallel_events.sort_by_key(|(timestamp, delta)| (*timestamp, -delta));
    let mut current_parallelism = 0_i32;
    let mut max_parallelism = 0_i32;
    for (_, delta) in parallel_events {
        current_parallelism += delta;
        max_parallelism = max_parallelism.max(current_parallelism);
    }

    json!({
        "task_parent_tool_ids": task_tool_ids
            .iter()
            .filter_map(|tool_id| normalizer.normalize(Some(tool_id)))
            .collect::<Vec<_>>(),
        "nested_progress_event_count": progress_event_count,
        "nested_agent_count": agent_ids.len(),
        "nested_tool_call_count": tool_counts.values().sum::<usize>(),
        "nested_tool_result_count": tool_result_count,
        "nested_tool_error_count": tool_error_count,
        "nested_tool_counts": tool_counts,
        "nested_tool_names": tool_counts.keys().cloned().collect::<Vec<_>>(),
        "nested_tool_total_latency_ms": total_latency,
        "nested_tool_max_latency_ms": max_latency,
        "nested_tool_avg_latency_ms": avg_latency,
        "nested_tool_max_parallelism": max_parallelism,
        "nested_assistant_text_blocks": assistant_text_blocks,
        "task_duration_ms": (last_ts - first_ts).max(0),
    })
}

fn summarize_progress_indices(indices: &[usize], records: &[TraceRecord]) -> CachedProgressMetrics {
    let mut agent_ids = BTreeSet::new();
    let mut assistant_text_blocks = 0_usize;
    let mut tool_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut tool_start_times: FxHashMap<String, i64> = FxHashMap::default();
    let mut tool_intervals = Vec::new();
    let mut tool_result_count = 0_usize;
    let mut tool_error_count = 0_usize;
    let first_ts = progress_timestamp_ms(&records[indices[0]]);
    let last_ts = progress_timestamp_ms(&records[*indices.last().unwrap()]);

    for index in indices {
        let record = &records[*index];
        let timestamp_ms = progress_timestamp_ms(record);
        let Some(data) = record.raw.get("data").and_then(Value::as_object) else {
            continue;
        };
        if let Some(agent_id) = data.get("agentId").and_then(Value::as_str) {
            agent_ids.insert(agent_id.to_string());
        }

        let Some(nested_message) = data.get("message").and_then(Value::as_object) else {
            continue;
        };
        let nested_type = nested_message
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let nested_payload = nested_message.get("message").and_then(Value::as_object);

        if nested_type == "assistant" {
            let Some(nested_payload) = nested_payload else {
                continue;
            };
            for block in content_blocks(nested_payload.get("content")) {
                let block_type = block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if block_type == "text" {
                    assistant_text_blocks += 1;
                    continue;
                }
                if block_type != "tool_use" {
                    continue;
                }
                let Some(raw_id) = block.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let tool_name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                *tool_counts.entry(tool_name).or_insert(0) += 1;
                tool_start_times.insert(raw_id.to_string(), timestamp_ms);
            }
            continue;
        }

        if nested_type != "user" {
            continue;
        }
        let Some(nested_payload) = nested_payload else {
            continue;
        };
        for block in content_blocks(nested_payload.get("content")) {
            if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            let Some(raw_tool_id) = block.get("tool_use_id").and_then(Value::as_str) else {
                continue;
            };
            if let Some(start_ms) = tool_start_times.remove(raw_tool_id) {
                tool_intervals.push((start_ms, timestamp_ms));
            }
            tool_result_count += 1;
            if block
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                tool_error_count += 1;
            }
        }
    }

    CachedProgressMetrics {
        progress_event_count: indices.len(),
        agent_ids,
        assistant_text_blocks,
        tool_counts,
        tool_result_count,
        tool_error_count,
        tool_intervals,
        first_ts,
        last_ts,
    }
}

#[cfg(test)]
mod tests {
    use super::{SessionTurnBuilder, TraceRecord};
    use crate::coding::tokenizer::TokenizerWorker;
    use anyhow::Result;
    use serde_json::{Value, json};

    struct StubTokenizer;

    impl TokenizerWorker for StubTokenizer {
        fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
            Ok(vec![text.len() as u32])
        }
    }

    fn make_record(
        row_type: &str,
        timestamp_ms: i64,
        source_order: u64,
        raw: Value,
    ) -> TraceRecord {
        TraceRecord {
            session_id: "session-1".to_string(),
            row_type: row_type.to_string(),
            timestamp_ms,
            source_order,
            raw,
        }
    }

    #[test]
    fn compact_boundary_restarts_transcript_from_summary() {
        let records = vec![
            make_record(
                "user",
                1_000,
                0,
                json!({"type":"user","message":{"role":"user","content":"before compact"}}),
            ),
            make_record(
                "assistant",
                2_000,
                1,
                json!({"type":"assistant","message":{"id":"assistant-1","content":[{"type":"text","text":"first answer"}],"usage":{"output_tokens":3}}}),
            ),
            make_record(
                "system",
                3_000,
                2,
                json!({"type":"system","subtype":"compact_boundary"}),
            ),
            make_record(
                "user",
                3_001,
                3,
                json!({"type":"user","isCompactSummary":true,"message":{"role":"user","content":"compacted summary"}}),
            ),
            make_record(
                "assistant",
                4_000,
                4,
                json!({"type":"assistant","message":{"id":"assistant-2","content":[{"type":"text","text":"second answer"}],"usage":{"output_tokens":5}}}),
            ),
        ];

        let mut builder = SessionTurnBuilder::new("session-1".to_string(), records, true);
        let mut tokenizer = StubTokenizer;
        let mut turns = Vec::new();
        while let Some(turn) = builder.next_turn(&mut tokenizer).unwrap() {
            turns.push(turn);
        }

        assert_eq!(
            turns
                .iter()
                .map(|turn| turn.input_text.as_str())
                .collect::<Vec<_>>(),
            vec!["[user] before compact", "[user] compacted summary"]
        );
    }

    #[test]
    fn compact_boundary_skips_local_command_noise_after_summary() {
        let records = vec![
            make_record(
                "user",
                1_000,
                0,
                json!({"type":"user","message":{"role":"user","content":"before compact"}}),
            ),
            make_record(
                "assistant",
                2_000,
                1,
                json!({"type":"assistant","message":{"id":"assistant-1","content":[{"type":"text","text":"first answer"}],"usage":{"output_tokens":3}}}),
            ),
            make_record(
                "system",
                3_000,
                2,
                json!({"type":"system","subtype":"compact_boundary"}),
            ),
            make_record(
                "user",
                3_001,
                3,
                json!({"type":"user","isCompactSummary":true,"message":{"role":"user","content":"compacted summary"}}),
            ),
            make_record(
                "user",
                3_002,
                4,
                json!({"type":"user","isMeta":true,"message":{"role":"user","content":"<local-command-caveat>ignore me</local-command-caveat>"}}),
            ),
            make_record(
                "user",
                3_003,
                5,
                json!({"type":"user","message":{"role":"user","content":"<command-name>/compact</command-name>\n<command-message>compact</command-message>"}}),
            ),
            make_record(
                "user",
                3_004,
                6,
                json!({"type":"user","message":{"role":"user","content":"<local-command-stdout>Compacted</local-command-stdout>"}}),
            ),
            make_record(
                "assistant",
                4_000,
                7,
                json!({"type":"assistant","message":{"id":"assistant-2","content":[{"type":"text","text":"second answer"}],"usage":{"output_tokens":5}}}),
            ),
        ];

        let mut builder = SessionTurnBuilder::new("session-1".to_string(), records, true);
        let mut tokenizer = StubTokenizer;
        let mut turns = Vec::new();
        while let Some(turn) = builder.next_turn(&mut tokenizer).unwrap() {
            turns.push(turn);
        }

        assert_eq!(
            turns
                .iter()
                .map(|turn| turn.input_text.as_str())
                .collect::<Vec<_>>(),
            vec!["[user] before compact", "[user] compacted summary"]
        );
    }
}

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
    pub parent_session_id: Option<String>,
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
    raw_id: Option<String>,
    name: String,
    normalized_id: Option<String>,
    arg_size_chars: usize,
    started_at_ms: i64,
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
    entries_by_record: BTreeMap<usize, Vec<ConversationEntry>>,
    model: String,
    output_length: usize,
    assistant_text_blocks: usize,
    top_level_tool_calls: Vec<ToolCallSummary>,
    raw_task_tool_ids: Vec<String>,
    input_length: Option<usize>,
    cache_read_input_tokens: Option<usize>,
    cache_creation_input_tokens: Option<usize>,
    start_ms: i64,
    end_ms: i64,
}

#[derive(Clone, Debug)]
pub struct ToolDraft {
    pub tool_call_id: String,
    pub tool_class: String,
    pub started_at_ms: i64,
    pub ended_at_ms: i64,
    pub is_error: bool,
    pub output_bytes: usize,
    pub child_session_id: Option<String>,
    pub consumer_turn_index: Option<usize>,
    pub execution_mode: String,
}

#[derive(Clone, Debug)]
pub struct TurnDraft {
    pub session_id: String,
    pub source_request_id: String,
    pub export_session_id: String,
    pub export_parent_session_id: Option<String>,
    pub turn_index: usize,
    pub model: String,
    pub input_text: String,
    pub output_length: usize,
    pub observed_input_length: Option<usize>,
    pub cache_read_input_tokens: Option<usize>,
    pub cache_creation_input_tokens: Option<usize>,
    pub request_start_ms: i64,
    pub assistant_start_ms: i64,
    pub assistant_end_ms: i64,
    pub delay_ms: Option<i64>,
    pub tools: Vec<ToolDraft>,
    pub sidecar: Value,
    pub compaction: Option<CompactionMetadata>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionMetadata {
    pub sequence: usize,
    pub trigger: String,
    pub pre_tokens: usize,
    pub post_tokens: usize,
    pub duration_ms: i64,
    pub ended_at_ms: i64,
}

#[derive(Clone, Debug, Default)]
pub struct SourceRequestExpectation {
    pub input_length: Option<usize>,
    pub cache_read_input_tokens: Option<usize>,
    pub cache_creation_input_tokens: Option<usize>,
    pub output_length: Option<usize>,
    pub request_start_ms: i64,
    pub assistant_end_ms: i64,
}

#[derive(Clone, Debug, Default)]
pub struct SourceFidelityOracle {
    pub requests: BTreeMap<(String, String), SourceRequestExpectation>,
    pub compactions: BTreeMap<(String, String), CompactionMetadata>,
    pub tools_by_class: BTreeMap<String, usize>,
    pub paired_tools: usize,
    pub tool_errors: usize,
    pub child_links: usize,
    pub background_tools: usize,
    pub background_agents: usize,
    pub background_completions_missing: usize,
    pub background_titles: usize,
    pub unmatched_tool_calls: usize,
    pub unmatched_tool_results: usize,
}

#[derive(Debug)]
struct PendingCompaction {
    metadata: CompactionMetadata,
    prompt_text: String,
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
    export_parent_session_id: Option<String>,
    records: Vec<TraceRecord>,
    top_level_indices: Vec<usize>,
    progress_metrics_index: FxHashMap<String, CachedProgressMetrics>,
    request_index_by_group_key: FxHashMap<String, usize>,
    request_start_ms_by_group_key: FxHashMap<String, i64>,
    top_level_cursor: usize,
    normalizer: ToolIdNormalizer,
    conversation_entries: Vec<ConversationEntry>,
    prompt_text: String,
    pending_request_start_ms: Option<i64>,
    previous_assistant_end_ms: Option<i64>,
    turn_index: usize,
    pending_compaction: Option<PendingCompaction>,
    previous_model: Option<String>,
    compaction_sequence: usize,
    preserve_session_ids: bool,
}

impl SessionTurnBuilder {
    pub fn new(trace_id: String, records: Vec<TraceRecord>, preserve_session_ids: bool) -> Self {
        let root_session_id = records
            .first()
            .map(|record| record.session_id.clone())
            .unwrap_or_else(|| trace_id.clone());
        let is_subagent = trace_id != root_session_id;
        let parent_session_id = records
            .iter()
            .find_map(|record| record.parent_session_id.clone())
            .unwrap_or_else(|| root_session_id.clone());
        let export_session_id = if preserve_session_ids {
            trace_id.clone()
        } else {
            anonymized_session_id(&trace_id)
        };
        let export_parent_session_id = is_subagent.then(|| {
            if preserve_session_ids {
                parent_session_id
            } else {
                anonymized_session_id(&parent_session_id)
            }
        });

        let progress_index = build_progress_index(&records);
        let progress_metrics_index = build_progress_metrics_index(&progress_index, &records);
        let top_level_indices: Vec<usize> = records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                let is_top_level =
                    matches!(record.row_type.as_str(), "user" | "assistant" | "system")
                        && (is_subagent
                            || !record
                                .raw
                                .get("isSidechain")
                                .and_then(Value::as_bool)
                                .unwrap_or(false));
                is_top_level.then_some(index)
            })
            .collect();
        let mut request_index_by_group_key = FxHashMap::default();
        let mut request_start_ms_by_group_key = FxHashMap::default();
        for index in &top_level_indices {
            let record = &records[*index];
            if record.row_type != "assistant" {
                continue;
            }
            let group_key = assistant_group_key(record);
            let next_index = request_index_by_group_key.len();
            request_index_by_group_key
                .entry(group_key.clone())
                .or_insert(next_index);
            request_start_ms_by_group_key
                .entry(group_key)
                .and_modify(|start: &mut i64| *start = (*start).min(record.timestamp_ms))
                .or_insert(record.timestamp_ms);
        }

        Self {
            session_id: trace_id,
            export_session_id,
            export_parent_session_id,
            records,
            top_level_indices,
            progress_metrics_index,
            request_index_by_group_key,
            request_start_ms_by_group_key,
            top_level_cursor: 0,
            normalizer: ToolIdNormalizer::default(),
            conversation_entries: Vec::new(),
            prompt_text: String::new(),
            pending_request_start_ms: None,
            previous_assistant_end_ms: None,
            turn_index: 0,
            pending_compaction: None,
            previous_model: None,
            compaction_sequence: 0,
            preserve_session_ids,
        }
    }

    pub fn next_turn(&mut self, tokenizer: &mut impl TokenizerWorker) -> Result<Option<TurnDraft>> {
        while self.top_level_cursor < self.top_level_indices.len() {
            let record_index = self.top_level_indices[self.top_level_cursor];
            let record = &self.records[record_index];

            if record.row_type == "system" {
                if is_compact_boundary(record) {
                    let metadata = compaction_metadata(record, self.compaction_sequence)?;
                    self.compaction_sequence += 1;
                    self.pending_compaction = Some(PendingCompaction {
                        metadata,
                        prompt_text: self.prompt_text.clone(),
                    });
                }
                self.top_level_cursor += 1;
                continue;
            }

            if record.row_type == "user" {
                if should_skip_user_record(record)? {
                    self.top_level_cursor += 1;
                    continue;
                }

                let request_start_ms = record.timestamp_ms;
                let message = object_field(&record.raw, "message");
                let rendered_entries = render_user_entries(message, &mut self.normalizer)?;
                if is_compact_summary(record) {
                    let summary_text = flatten_block_content_text(
                        message
                            .and_then(|message| message.get("content"))
                            .unwrap_or(&Value::Null),
                    )?;
                    self.replace_conversation_entries(rendered_entries);
                    self.pending_request_start_ms = Some(request_start_ms);
                    self.top_level_cursor += 1;

                    let Some(pending) = self.pending_compaction.take() else {
                        continue;
                    };

                    let output_length = tokenizer.encode(&summary_text)?.len();
                    let input_text = if pending.prompt_text.is_empty() {
                        "[system] Compact the conversation.".to_string()
                    } else {
                        format!(
                            "{}\n[system] Compact the conversation.",
                            pending.prompt_text
                        )
                    };
                    let source_request_id = format!("compact:{}", pending.metadata.sequence);
                    let mut sidecar = Map::new();
                    sidecar.insert(
                        "session_id".to_string(),
                        Value::String(self.export_session_id.clone()),
                    );
                    if let Some(parent_session_id) = &self.export_parent_session_id {
                        sidecar.insert(
                            "parent_session_id".to_string(),
                            Value::String(parent_session_id.clone()),
                        );
                    }
                    sidecar.insert("turn_index".to_string(), json!(self.turn_index));
                    sidecar.insert(
                        "source_request_id".to_string(),
                        Value::String(source_request_id.clone()),
                    );
                    sidecar.insert("request_kind".to_string(), json!("compaction"));
                    sidecar.insert(
                        "input_fidelity".to_string(),
                        json!("claude_cache_safe_fork"),
                    );
                    sidecar.insert(
                        "replay_hash_fidelity".to_string(),
                        json!("synthetic_usage_shaped"),
                    );
                    sidecar.insert(
                        "compaction".to_string(),
                        compaction_json(&pending.metadata, output_length),
                    );
                    let request_start_ms = pending
                        .metadata
                        .ended_at_ms
                        .saturating_sub(pending.metadata.duration_ms);
                    self.previous_assistant_end_ms = Some(pending.metadata.ended_at_ms);
                    return Ok(Some(TurnDraft {
                        session_id: self.session_id.clone(),
                        source_request_id,
                        export_session_id: self.export_session_id.clone(),
                        export_parent_session_id: self.export_parent_session_id.clone(),
                        turn_index: self.turn_index,
                        model: self
                            .previous_model
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        input_text,
                        output_length,
                        observed_input_length: Some(pending.metadata.pre_tokens),
                        cache_read_input_tokens: None,
                        cache_creation_input_tokens: None,
                        request_start_ms,
                        assistant_start_ms: pending.metadata.ended_at_ms,
                        assistant_end_ms: pending.metadata.ended_at_ms,
                        delay_ms: None,
                        tools: Vec::new(),
                        sidecar: Value::Object(sidecar),
                        compaction: Some(pending.metadata),
                    }));
                } else {
                    self.pending_compaction = None;
                    self.extend_conversation_entries(rendered_entries);
                }
                self.pending_request_start_ms = Some(request_start_ms);
                self.top_level_cursor += 1;
                continue;
            }

            self.pending_compaction = None;
            let group_key = assistant_group_key(record);
            let mut group_indices = vec![record_index];
            let mut interleaved_user_indices = Vec::new();
            self.top_level_cursor += 1;
            while self.top_level_cursor < self.top_level_indices.len() {
                let next_index = self.top_level_indices[self.top_level_cursor];
                let next_record = &self.records[next_index];
                if next_record.row_type == "system" && !is_compact_boundary(next_record) {
                    self.top_level_cursor += 1;
                    continue;
                }
                if next_record.row_type == "user" && is_tool_result_user_record(next_record) {
                    interleaved_user_indices.push(next_index);
                    self.top_level_cursor += 1;
                    continue;
                }
                if next_record.row_type != "assistant"
                    || assistant_group_key(next_record) != group_key
                {
                    break;
                }
                group_indices.push(next_index);
                self.top_level_cursor += 1;
            }

            let mut group_summary = summarize_assistant_group(
                &self.records,
                &group_indices,
                &mut self.normalizer,
                tokenizer,
            )?;
            let input_text = self.prompt_text.clone();
            let request_start_ms = self
                .pending_request_start_ms
                .take()
                .unwrap_or(group_summary.start_ms);
            let tools = pair_tool_results(
                &group_summary.top_level_tool_calls,
                &interleaved_user_indices,
                &self.records,
                &self.request_index_by_group_key,
                &self.request_start_ms_by_group_key,
                &group_key,
                self.preserve_session_ids,
            )?;

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
            if let Some(parent_session_id) = &self.export_parent_session_id {
                sidecar.insert(
                    "parent_session_id".to_string(),
                    Value::String(parent_session_id.clone()),
                );
            }
            sidecar.insert("turn_index".to_string(), json!(self.turn_index));
            sidecar.insert(
                "source_request_id".to_string(),
                Value::String(group_key.clone()),
            );
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
            sidecar.insert(
                "input_fidelity".to_string(),
                Value::String(
                    if group_summary.input_length.is_some() {
                        "claude_usage_cache_prefix"
                    } else {
                        "rendered_transcript"
                    }
                    .to_string(),
                ),
            );
            sidecar.insert(
                "replay_hash_fidelity".to_string(),
                Value::String(
                    if group_summary.input_length.is_some() {
                        "synthetic_usage_shaped"
                    } else {
                        "rendered_transcript"
                    }
                    .to_string(),
                ),
            );
            if let Some(input_length) = group_summary.input_length {
                sidecar.insert("observed_input_tokens".to_string(), json!(input_length));
            }
            if let Some(cache_read) = group_summary.cache_read_input_tokens {
                sidecar.insert(
                    "observed_cache_read_input_tokens".to_string(),
                    json!(cache_read),
                );
            }
            if let Some(cache_creation) = group_summary.cache_creation_input_tokens {
                sidecar.insert(
                    "observed_cache_creation_input_tokens".to_string(),
                    json!(cache_creation),
                );
            }

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

            let model = group_summary.model;
            self.previous_model = Some(model.clone());
            let turn = TurnDraft {
                session_id: self.session_id.clone(),
                source_request_id: group_key,
                export_session_id: self.export_session_id.clone(),
                export_parent_session_id: self.export_parent_session_id.clone(),
                turn_index: self.turn_index,
                model,
                input_text,
                output_length: group_summary.output_length,
                observed_input_length: group_summary.input_length,
                cache_read_input_tokens: group_summary.cache_read_input_tokens,
                cache_creation_input_tokens: group_summary.cache_creation_input_tokens,
                request_start_ms,
                assistant_start_ms: group_summary.start_ms,
                assistant_end_ms: group_summary.end_ms,
                delay_ms: self
                    .previous_assistant_end_ms
                    .map(|previous_end| (request_start_ms - previous_end).max(0)),
                tools,
                sidecar: Value::Object(sidecar),
                compaction: None,
            };

            let mut ordered_indices = group_indices;
            ordered_indices.extend(interleaved_user_indices.iter().copied());
            ordered_indices.sort_unstable();
            let mut ordered_entries = Vec::new();
            for index in ordered_indices {
                if let Some(entries) = group_summary.entries_by_record.remove(&index) {
                    ordered_entries.extend(entries);
                } else {
                    ordered_entries.extend(render_user_entries(
                        object_field(&self.records[index].raw, "message"),
                        &mut self.normalizer,
                    )?);
                }
            }
            self.extend_conversation_entries(ordered_entries);
            self.pending_request_start_ms = interleaved_user_indices
                .last()
                .map(|index| self.records[*index].timestamp_ms);
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
            let agent_id = payload
                .get("agentId")
                .and_then(Value::as_str)
                .map(str::to_string);
            let row_type = payload
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string);
            let (Some(session_id), Some(row_type)) = (session_id, row_type) else {
                source_order += 1;
                continue;
            };
            let timestamp_ms = match payload.get("timestamp").and_then(Value::as_str) {
                Some(timestamp) => match parse_utc_timestamp_ms(timestamp) {
                    Ok(timestamp_ms) => timestamp_ms,
                    Err(_) => {
                        source_order += 1;
                        continue;
                    }
                },
                None if row_type == "ai-title" => 0,
                None => {
                    source_order += 1;
                    continue;
                }
            };

            let trace_id = agent_id.clone().unwrap_or_else(|| session_id.clone());
            sessions.entry(trace_id).or_default().push(TraceRecord {
                session_id,
                parent_session_id: None,
                row_type,
                timestamp_ms,
                source_order,
                raw: payload,
            });
            source_order += 1;
        }
    }

    let mut parent_by_session = FxHashMap::default();
    for (parent_session_id, records) in &sessions {
        for record in records {
            if let Some(child_session_id) = record
                .raw
                .get("toolUseResult")
                .and_then(Value::as_object)
                .and_then(|result| result.get("agentId"))
                .and_then(Value::as_str)
            {
                parent_by_session
                    .entry(child_session_id.to_string())
                    .or_insert_with(|| parent_session_id.clone());
            }
        }
    }

    for (session_id, records) in &mut sessions {
        let parent_session_id = parent_by_session.get(session_id).cloned();
        for record in records.iter_mut() {
            record.parent_session_id.clone_from(&parent_session_id);
        }
        records.sort_by_key(|record| record.source_order);
    }

    Ok(sessions)
}

pub fn build_source_fidelity_oracle(
    sessions: &FxHashMap<String, Vec<TraceRecord>>,
) -> Result<SourceFidelityOracle> {
    let mut oracle = SourceFidelityOracle::default();
    let mut tool_calls: FxHashMap<(String, String), String> = FxHashMap::default();
    let mut background_titles = BTreeSet::new();
    let mut background_tool_ids = BTreeSet::new();
    let background_completions = sessions
        .iter()
        .flat_map(|(trace_id, records)| {
            records.iter().filter_map(|record| {
                let content = (record.row_type == "queue-operation"
                    && record.raw.get("operation").and_then(Value::as_str) == Some("enqueue"))
                .then(|| record.raw.get("content").and_then(Value::as_str))
                .flatten()?;
                Some((
                    (trace_id.clone(), queued_tool_id(content)?.to_string()),
                    !content.contains("<status>completed</status>"),
                ))
            })
        })
        .collect::<FxHashMap<_, _>>();

    for (trace_id, records) in sessions {
        let root_session_id = records
            .first()
            .map(|record| record.session_id.as_str())
            .unwrap_or(trace_id);
        let is_subagent = trace_id != root_session_id;
        let mut pending_request_start_ms = None;
        let mut compaction_sequence = 0;

        for record in records {
            if record.row_type == "ai-title" {
                let title = record
                    .raw
                    .get("aiTitle")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                background_titles.insert((record.session_id.clone(), title.to_string()));
                continue;
            }
            let is_top_level = matches!(record.row_type.as_str(), "user" | "assistant" | "system")
                && (is_subagent
                    || !record
                        .raw
                        .get("isSidechain")
                        .and_then(Value::as_bool)
                        .unwrap_or(false));
            if !is_top_level {
                continue;
            }
            if is_compact_boundary(record) {
                let metadata = compaction_metadata(record, compaction_sequence)?;
                let source_request_id = format!("compact:{}", metadata.sequence);
                oracle
                    .compactions
                    .insert((trace_id.clone(), source_request_id), metadata);
                compaction_sequence += 1;
                continue;
            }
            if record.row_type == "system" {
                continue;
            }

            if record.row_type == "user" {
                if should_skip_user_record(record)? {
                    continue;
                }
                pending_request_start_ms = Some(record.timestamp_ms);
                let message = object_field(&record.raw, "message");
                for block in content_blocks(message.and_then(|message| message.get("content"))) {
                    if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                        continue;
                    }
                    let Some(raw_id) = block.get("tool_use_id").and_then(Value::as_str) else {
                        continue;
                    };
                    let key = (trace_id.clone(), raw_id.to_string());
                    let Some(tool_class) = tool_calls.remove(&key) else {
                        oracle.unmatched_tool_results += 1;
                        continue;
                    };
                    oracle.paired_tools += 1;
                    *oracle.tools_by_class.entry(tool_class).or_insert(0) += 1;
                    let launch_error = block
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let mut is_async = false;
                    if let Some(result) = record.raw.get("toolUseResult").and_then(Value::as_object)
                    {
                        is_async = result
                            .get("isAsync")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                            || result.get("backgroundTaskId").is_some();
                        if result.get("agentId").and_then(Value::as_str).is_some() {
                            oracle.child_links += 1;
                            oracle.background_agents += usize::from(is_async);
                        }
                        if is_async {
                            oracle.background_tools += 1;
                            background_tool_ids.insert(key);
                        }
                    }
                    oracle.tool_errors += usize::from(if is_async {
                        background_completions
                            .get(&(trace_id.clone(), raw_id.to_string()))
                            .copied()
                            .unwrap_or(launch_error)
                    } else {
                        launch_error
                    });
                }
                continue;
            }

            let group_key = assistant_group_key(record);
            let request_key = (trace_id.clone(), group_key);
            if !oracle.requests.contains_key(&request_key) {
                oracle.requests.insert(
                    request_key.clone(),
                    SourceRequestExpectation {
                        request_start_ms: pending_request_start_ms
                            .take()
                            .unwrap_or(record.timestamp_ms),
                        assistant_end_ms: record.timestamp_ms,
                        ..Default::default()
                    },
                );
            }
            let expectation = oracle
                .requests
                .get_mut(&request_key)
                .expect("request expectation was inserted");
            expectation.assistant_end_ms = expectation.assistant_end_ms.max(record.timestamp_ms);
            let Some(message) = object_field(&record.raw, "message") else {
                continue;
            };
            if let Some(usage) = object_field(&Value::Object(message.clone()), "usage") {
                let input_length = [
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ]
                .into_iter()
                .filter_map(|key| usage.get(key).and_then(Value::as_u64))
                .fold(0_usize, |total, value| total.saturating_add(value as usize));
                expectation.input_length = Some(
                    expectation
                        .input_length
                        .unwrap_or_default()
                        .max(input_length),
                );
                let cache_read = usage
                    .get("cache_read_input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                expectation.cache_read_input_tokens = Some(
                    expectation
                        .cache_read_input_tokens
                        .unwrap_or_default()
                        .max(cache_read),
                );
                let cache_creation = usage
                    .get("cache_creation_input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                expectation.cache_creation_input_tokens = Some(
                    expectation
                        .cache_creation_input_tokens
                        .unwrap_or_default()
                        .max(cache_creation),
                );
                if let Some(output_length) = usage.get("output_tokens").and_then(Value::as_u64) {
                    expectation.output_length = Some(
                        expectation
                            .output_length
                            .unwrap_or_default()
                            .max(output_length as usize),
                    );
                }
            }
            for block in content_blocks(message.get("content")) {
                if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                    continue;
                }
                let Some(raw_id) = block.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let tool_class = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                tool_calls.insert((trace_id.clone(), raw_id.to_string()), tool_class);
            }
        }
    }

    oracle.background_titles = background_titles.len();
    oracle.background_completions_missing = background_tool_ids
        .iter()
        .filter(|tool_id| !background_completions.contains_key(*tool_id))
        .count();
    oracle.unmatched_tool_calls = tool_calls.len();
    Ok(oracle)
}

fn queued_tool_id(content: &str) -> Option<&str> {
    let start = content.find("<tool-use-id>")? + "<tool-use-id>".len();
    let end = content[start..].find("</tool-use-id>")? + start;
    Some(&content[start..end])
}

pub(crate) fn assistant_group_key(record: &TraceRecord) -> String {
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

fn compaction_metadata(record: &TraceRecord, sequence: usize) -> Result<CompactionMetadata> {
    let metadata = record
        .raw
        .get("compactMetadata")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("compact_boundary is missing compactMetadata"))?;
    let trigger = metadata
        .get("trigger")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("compactMetadata is missing trigger"))?
        .to_string();
    let pre_tokens = metadata
        .get("preTokens")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow::anyhow!("compactMetadata has invalid preTokens"))?;
    if pre_tokens == 0 {
        anyhow::bail!("compactMetadata preTokens leaves no recoverable prefix");
    }
    let post_tokens = metadata
        .get("postTokens")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow::anyhow!("compactMetadata has invalid postTokens"))?;
    let duration_ms = metadata
        .get("durationMs")
        .and_then(Value::as_u64)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or_else(|| anyhow::anyhow!("compactMetadata has invalid durationMs"))?;
    Ok(CompactionMetadata {
        sequence,
        trigger,
        pre_tokens,
        post_tokens,
        duration_ms,
        ended_at_ms: record.timestamp_ms,
    })
}

fn compaction_json(metadata: &CompactionMetadata, summary_output_tokens: usize) -> Value {
    json!({
        "trigger": metadata.trigger,
        "pre_tokens": metadata.pre_tokens,
        "post_tokens": metadata.post_tokens,
        "duration_ms": metadata.duration_ms,
        "summary_output_tokens": summary_output_tokens,
        "cache_fidelity": "recoverable_cache_safe_prefix",
        "output_fidelity": "tokenized_compact_summary",
    })
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

fn is_tool_result_user_record(record: &TraceRecord) -> bool {
    if record.row_type != "user" {
        return false;
    }
    let Some(message) = object_field(&record.raw, "message") else {
        return false;
    };
    let blocks = content_blocks(message.get("content"));
    !blocks.is_empty()
        && blocks
            .iter()
            .all(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
}

fn pair_tool_results(
    calls: &[ToolCallSummary],
    user_indices: &[usize],
    records: &[TraceRecord],
    request_index_by_group_key: &FxHashMap<String, usize>,
    request_start_ms_by_group_key: &FxHashMap<String, i64>,
    current_group_key: &str,
    preserve_session_ids: bool,
) -> Result<Vec<ToolDraft>> {
    let calls_by_id = calls
        .iter()
        .filter_map(|call| call.raw_id.as_deref().map(|id| (id, call)))
        .collect::<FxHashMap<_, _>>();
    let mut tools = Vec::new();

    for index in user_indices {
        let record = &records[*index];
        let Some(message) = object_field(&record.raw, "message") else {
            continue;
        };
        for block in content_blocks(message.get("content")) {
            if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            let Some(raw_id) = block.get("tool_use_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(call) = calls_by_id.get(raw_id) else {
                continue;
            };
            let content = flatten_block_content_text(block.get("content").unwrap_or(&Value::Null))?;
            let tool_result = record.raw.get("toolUseResult").and_then(Value::as_object);
            let is_async = tool_result
                .and_then(|result| result.get("isAsync"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
                || tool_result.is_some_and(|result| result.get("backgroundTaskId").is_some());
            let child_session_id = tool_result
                .and_then(|result| result.get("agentId"))
                .and_then(Value::as_str)
                .map(|id| {
                    if preserve_session_ids {
                        id.to_string()
                    } else {
                        anonymized_session_id(id)
                    }
                });
            let launch_error = block
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let (ended_at_ms, output_bytes, is_error, consumer_turn_index) = if is_async {
                async_tool_completion(
                    records,
                    raw_id,
                    record.timestamp_ms,
                    request_index_by_group_key,
                    request_start_ms_by_group_key,
                )
                .unwrap_or((
                    record.timestamp_ms,
                    content.len(),
                    launch_error,
                    None,
                ))
            } else {
                (
                    record.timestamp_ms,
                    content.len(),
                    launch_error,
                    next_consumer_turn(
                        records,
                        record.source_order,
                        current_group_key,
                        request_index_by_group_key,
                    ),
                )
            };
            tools.push(ToolDraft {
                tool_call_id: call
                    .normalized_id
                    .clone()
                    .unwrap_or_else(|| "tool_unknown".to_string()),
                tool_class: call.name.clone(),
                started_at_ms: call.started_at_ms,
                ended_at_ms,
                is_error,
                output_bytes,
                child_session_id,
                consumer_turn_index,
                execution_mode: if is_async {
                    "background".to_string()
                } else {
                    "blocking".to_string()
                },
            });
        }
    }

    Ok(tools)
}

fn next_consumer_turn(
    records: &[TraceRecord],
    after_source_order: u64,
    current_group_key: &str,
    request_index_by_group_key: &FxHashMap<String, usize>,
) -> Option<usize> {
    records
        .iter()
        .filter(|record| record.source_order > after_source_order && record.row_type == "assistant")
        .find_map(|record| {
            let group_key = assistant_group_key(record);
            (group_key != current_group_key)
                .then(|| request_index_by_group_key.get(&group_key).copied())
                .flatten()
        })
}

fn async_tool_completion(
    records: &[TraceRecord],
    raw_tool_id: &str,
    after_timestamp_ms: i64,
    request_index_by_group_key: &FxHashMap<String, usize>,
    request_start_ms_by_group_key: &FxHashMap<String, i64>,
) -> Option<(i64, usize, bool, Option<usize>)> {
    let tool_marker = format!("<tool-use-id>{raw_tool_id}</tool-use-id>");
    let completion = records
        .iter()
        .filter(|record| {
            record.timestamp_ms >= after_timestamp_ms
                && record.row_type == "queue-operation"
                && record.raw.get("operation").and_then(Value::as_str) == Some("enqueue")
                && record
                    .raw
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| content.contains(&tool_marker))
        })
        .min_by_key(|record| (record.timestamp_ms, record.source_order))?;
    let content = completion
        .raw
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let consumer_turn_index = request_index_by_group_key
        .iter()
        .filter_map(|(group_key, turn_index)| {
            let start_ms = *request_start_ms_by_group_key.get(group_key)?;
            (start_ms > completion.timestamp_ms).then_some((start_ms, *turn_index))
        })
        .min()
        .map(|(_, turn_index)| turn_index);
    Some((
        completion.timestamp_ms,
        content.len(),
        !content.contains("<status>completed</status>"),
        consumer_turn_index,
    ))
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
    let mut entries_by_record = BTreeMap::new();
    let mut tool_calls = Vec::new();
    let mut raw_task_tool_ids = Vec::new();
    let mut assistant_text_blocks = 0;
    let mut output_lengths = Vec::new();
    let mut input_lengths = Vec::new();
    let mut cache_read_lengths = Vec::new();
    let mut cache_creation_lengths = Vec::new();
    let mut model = None;

    for index in group_indices {
        let record = &records[*index];
        let mut record_entries = Vec::new();
        let Some(message) = object_field(&record.raw, "message") else {
            continue;
        };
        if model.is_none() {
            model = message
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        if let Some(usage) = object_field(&Value::Object(message.clone()), "usage") {
            let input_tokens = usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let cache_read = usage
                .get("cache_read_input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let cache_creation = usage
                .get("cache_creation_input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            input_lengths.push(
                input_tokens
                    .saturating_add(cache_read)
                    .saturating_add(cache_creation),
            );
            cache_read_lengths.push(cache_read);
            cache_creation_lengths.push(cache_creation);
            if let Some(output_tokens) = usage.get("output_tokens").and_then(Value::as_u64) {
                output_lengths.push(output_tokens as usize);
            }
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
                    record_entries.push(ConversationEntry {
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
                record_entries.push(ConversationEntry {
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
                    raw_id: raw_id.map(str::to_string),
                    name: tool_name.clone(),
                    normalized_id,
                    arg_size_chars: args_json.len(),
                    started_at_ms: record.timestamp_ms,
                });
                if matches!(tool_name.as_str(), "Agent" | "Task")
                    && let Some(raw_id) = raw_id
                {
                    raw_task_tool_ids.push(raw_id.to_string());
                }
                continue;
            }

            let sanitized = sanitize_structure(&block, normalizer);
            record_entries.push(ConversationEntry {
                kind: "assistant_block".to_string(),
                rendered: format!(
                    "[assistant_block type={block_type}] {}",
                    canonical_json_string(&sanitized)?
                ),
            });
        }
        entries.extend(record_entries.iter().cloned());
        entries_by_record.insert(*index, record_entries);
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
        entries_by_record,
        model: model.unwrap_or_else(|| "unknown".to_string()),
        output_length,
        assistant_text_blocks,
        top_level_tool_calls: tool_calls,
        raw_task_tool_ids,
        input_length: input_lengths.into_iter().max(),
        cache_read_input_tokens: cache_read_lengths.into_iter().max(),
        cache_creation_input_tokens: cache_creation_lengths.into_iter().max(),
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
    use super::{
        SessionTurnBuilder, TraceRecord, build_source_fidelity_oracle, load_trace_records,
    };
    use crate::coding::common::anonymized_session_id;
    use crate::coding::tokenizer::TokenizerWorker;
    use anyhow::Result;
    use rustc_hash::FxHashMap;
    use serde_json::{Value, json};
    use tempfile::TempDir;

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
            parent_session_id: None,
            row_type: row_type.to_string(),
            timestamp_ms,
            source_order,
            raw,
        }
    }

    #[test]
    fn groups_interleaved_fragments_and_pairs_tool_results() {
        let records = vec![
            make_record(
                "user",
                1_000,
                0,
                json!({"type":"user","message":{"role":"user","content":"start"}}),
            ),
            make_record(
                "assistant",
                1_100,
                1,
                json!({"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":[{"type":"text","text":"working"}],"usage":{"input_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":5,"output_tokens":7}}}),
            ),
            make_record(
                "assistant",
                1_200,
                2,
                json!({"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":[{"type":"tool_use","id":"raw-1","name":"Read","input":{}}],"usage":{"input_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":5,"output_tokens":7}}}),
            ),
            make_record(
                "system",
                1_250,
                3,
                json!({"type":"system","subtype":"turn_duration"}),
            ),
            make_record(
                "user",
                1_300,
                4,
                json!({"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"raw-1","content":"ok"}]}}),
            ),
            make_record(
                "assistant",
                1_400,
                5,
                json!({"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":[{"type":"tool_use","id":"raw-2","name":"Bash","input":{}}],"usage":{"input_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":5,"output_tokens":7}}}),
            ),
            make_record(
                "user",
                1_500,
                6,
                json!({"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"raw-2","content":"failed","is_error":true}]}}),
            ),
            make_record(
                "assistant",
                1_600,
                7,
                json!({"type":"assistant","requestId":"req-2","message":{"id":"msg-2","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":10,"output_tokens":2}}}),
            ),
        ];

        let mut builder = SessionTurnBuilder::new("session-1".to_string(), records, true);
        let first = builder.next_turn(&mut StubTokenizer).unwrap().unwrap();
        let second = builder.next_turn(&mut StubTokenizer).unwrap().unwrap();

        assert_eq!(first.request_start_ms, 1_000);
        assert_eq!(first.assistant_end_ms, 1_400);
        assert_eq!(first.observed_input_length, Some(10));
        assert_eq!(first.output_length, 7);
        assert_eq!(first.tools.len(), 2);
        assert_eq!(first.tools[0].tool_class, "Read");
        assert_eq!(first.tools[0].started_at_ms, 1_200);
        assert_eq!(first.tools[0].ended_at_ms, 1_300);
        assert!(first.tools[1].is_error);
        assert_eq!(second.request_start_ms, 1_500);
        assert!(builder.next_turn(&mut StubTokenizer).unwrap().is_none());
    }

    #[test]
    fn loader_preserves_source_order_for_compaction_markers() {
        let temp = TempDir::new().unwrap();
        let trace = temp.path().join("session.jsonl");
        std::fs::write(
            &trace,
            concat!(
                "{\"type\":\"system\",\"subtype\":\"compact_boundary\",\"sessionId\":\"session-1\",\"timestamp\":\"2026-01-01T00:00:00.002Z\",\"compactMetadata\":{\"trigger\":\"manual\",\"preTokens\":10,\"postTokens\":3,\"durationMs\":1}}\n",
                "{\"type\":\"user\",\"isCompactSummary\":true,\"sessionId\":\"session-1\",\"timestamp\":\"2026-01-01T00:00:00.001Z\",\"message\":{\"content\":\"summary\"}}\n",
                "{\"type\":\"assistant\",\"sessionId\":\"session-1\",\"timestamp\":\"2026-01-01T00:00:00.003Z\",\"message\":{\"id\":\"msg-1\",\"content\":[{\"type\":\"text\",\"text\":\"done\"}],\"usage\":{\"output_tokens\":1}}}\n"
            ),
        )
        .unwrap();

        let sessions = load_trace_records(&[trace]).unwrap();
        let records = sessions.get("session-1").unwrap();
        assert_eq!(records[0].row_type, "system");
        assert_eq!(records[1].row_type, "user");

        let mut builder = SessionTurnBuilder::new("session-1".to_string(), records.clone(), true);
        let compaction = builder.next_turn(&mut StubTokenizer).unwrap().unwrap();
        assert!(compaction.compaction.is_some());
        let turn = builder.next_turn(&mut StubTokenizer).unwrap().unwrap();
        assert_eq!(turn.input_text, "[user] summary");
    }

    #[test]
    fn loader_infers_immediate_parent_from_agent_result() {
        let temp = TempDir::new().unwrap();
        let parent = temp.path().join("parent.jsonl");
        let child = temp.path().join("child.jsonl");
        std::fs::write(
            &parent,
            "{\"type\":\"user\",\"sessionId\":\"root\",\"agentId\":\"parent\",\"timestamp\":\"2026-01-01T00:00:00.001Z\",\"toolUseResult\":{\"agentId\":\"child\"},\"message\":{\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"agent-call\",\"content\":\"done\"}]}}\n",
        )
        .unwrap();
        std::fs::write(
            &child,
            "{\"type\":\"assistant\",\"sessionId\":\"root\",\"agentId\":\"child\",\"timestamp\":\"2026-01-01T00:00:00.002Z\",\"message\":{\"id\":\"msg-1\",\"content\":[{\"type\":\"text\",\"text\":\"done\"}],\"usage\":{\"output_tokens\":1}}}\n",
        )
        .unwrap();

        let sessions = load_trace_records(&[parent, child]).unwrap();
        assert_eq!(
            sessions["child"][0].parent_session_id.as_deref(),
            Some("parent")
        );
    }

    #[test]
    fn fidelity_oracle_ignores_root_sidechain_compaction() {
        let mut sessions = FxHashMap::default();
        sessions.insert(
            "session-1".to_string(),
            vec![make_record(
                "system",
                1_000,
                0,
                json!({"type":"system","subtype":"compact_boundary","isSidechain":true,"compactMetadata":{"trigger":"manual","preTokens":10,"postTokens":3,"durationMs":500}}),
            )],
        );

        assert!(
            build_source_fidelity_oracle(&sessions)
                .unwrap()
                .compactions
                .is_empty()
        );
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
                json!({"type":"system","subtype":"compact_boundary","compactMetadata":{"trigger":"manual","preTokens":10,"postTokens":3,"durationMs":500}}),
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
                .filter(|turn| turn.compaction.is_none())
                .map(|turn| turn.input_text.as_str())
                .collect::<Vec<_>>(),
            vec!["[user] before compact", "[user] compacted summary"]
        );
        assert_eq!(
            turns
                .iter()
                .filter(|turn| turn.compaction.is_some())
                .count(),
            1
        );
    }

    #[test]
    fn compact_boundary_survives_ignored_rows_around_summary() {
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
                json!({"type":"system","subtype":"compact_boundary","compactMetadata":{"trigger":"manual","preTokens":10,"postTokens":3,"durationMs":500}}),
            ),
            make_record(
                "system",
                3_001,
                3,
                json!({"type":"system","subtype":"turn_duration"}),
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
                json!({"type":"user","isCompactSummary":true,"message":{"role":"user","content":"compacted summary"}}),
            ),
            make_record(
                "user",
                3_004,
                6,
                json!({"type":"user","isMeta":true,"message":{"role":"user","content":"<local-command-caveat>ignore me</local-command-caveat>"}}),
            ),
            make_record(
                "user",
                3_005,
                7,
                json!({"type":"user","message":{"role":"user","content":"<command-name>/compact</command-name>\n<command-message>compact</command-message>"}}),
            ),
            make_record(
                "user",
                3_006,
                8,
                json!({"type":"user","message":{"role":"user","content":"<local-command-stdout>Compacted</local-command-stdout>"}}),
            ),
            make_record(
                "assistant",
                4_000,
                9,
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
                .filter(|turn| turn.compaction.is_none())
                .map(|turn| turn.input_text.as_str())
                .collect::<Vec<_>>(),
            vec!["[user] before compact", "[user] compacted summary"]
        );
    }

    #[test]
    fn orphan_compact_summary_still_replaces_transcript() {
        let records = vec![
            make_record(
                "user",
                1_000,
                0,
                json!({"type":"user","message":{"role":"user","content":"old prompt"}}),
            ),
            make_record(
                "assistant",
                2_000,
                1,
                json!({"type":"assistant","message":{"id":"assistant-1","content":[{"type":"text","text":"old answer"}],"usage":{"output_tokens":2}}}),
            ),
            make_record(
                "user",
                3_000,
                2,
                json!({"type":"user","isCompactSummary":true,"message":{"role":"user","content":"summary only"}}),
            ),
            make_record(
                "assistant",
                4_000,
                3,
                json!({"type":"assistant","message":{"id":"assistant-2","content":[{"type":"text","text":"new answer"}],"usage":{"output_tokens":2}}}),
            ),
        ];

        let mut builder = SessionTurnBuilder::new("session-1".to_string(), records, true);
        let mut tokenizer = StubTokenizer;
        assert_eq!(
            builder
                .next_turn(&mut tokenizer)
                .unwrap()
                .unwrap()
                .input_text,
            "[user] old prompt"
        );
        assert_eq!(
            builder
                .next_turn(&mut tokenizer)
                .unwrap()
                .unwrap()
                .input_text,
            "[user] summary only"
        );
    }

    #[test]
    fn background_agent_joins_only_after_completion_notification() {
        let records = vec![
            make_record(
                "user",
                1_000,
                0,
                json!({"type":"user","message":{"role":"user","content":"start agent"}}),
            ),
            make_record(
                "assistant",
                1_100,
                1,
                json!({"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":[{"type":"tool_use","id":"agent-call","name":"Agent","input":{"run_in_background":true}}],"usage":{"output_tokens":2}}}),
            ),
            make_record(
                "user",
                1_150,
                2,
                json!({"type":"user","toolUseResult":{"isAsync":true,"agentId":"child-agent","status":"async_launched"},"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"agent-call","content":"launched"}]}}),
            ),
            make_record(
                "user",
                1_300,
                3,
                json!({"type":"user","message":{"role":"user","content":"keep working"}}),
            ),
            make_record(
                "assistant",
                1_400,
                4,
                json!({"type":"assistant","requestId":"req-2","message":{"id":"msg-2","content":[{"type":"text","text":"still working"}],"usage":{"output_tokens":2}}}),
            ),
            make_record(
                "queue-operation",
                1_800,
                5,
                json!({"type":"queue-operation","operation":"enqueue","content":"<tool-use-id>agent-call</tool-use-id><status>completed</status>done"}),
            ),
            make_record(
                "assistant",
                1_810,
                6,
                json!({"type":"assistant","requestId":"req-2","message":{"id":"msg-2","content":[{"type":"text","text":"late fragment"}],"usage":{"output_tokens":2}}}),
            ),
            make_record(
                "user",
                1_850,
                7,
                json!({"type":"user","message":{"role":"user","content":"use result"}}),
            ),
            make_record(
                "assistant",
                1_900,
                8,
                json!({"type":"assistant","requestId":"req-3","message":{"id":"msg-3","content":[{"type":"text","text":"finished"}],"usage":{"output_tokens":1}}}),
            ),
        ];

        let mut builder = SessionTurnBuilder::new("session-1".to_string(), records, true);
        let turns = std::iter::from_fn(|| builder.next_turn(&mut StubTokenizer).transpose())
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].tools.len(), 1);
        let tool = &turns[0].tools[0];
        assert_eq!(tool.ended_at_ms, 1_800);
        assert_eq!(tool.consumer_turn_index, Some(2));
        assert_eq!(tool.child_session_id.as_deref(), Some("child-agent"));
        assert_eq!(tool.execution_mode, "background");
    }

    #[test]
    fn background_bash_uses_terminal_notification_status() {
        let records = vec![
            make_record(
                "user",
                1_000,
                0,
                json!({"type":"user","message":{"role":"user","content":"start command"}}),
            ),
            make_record(
                "assistant",
                1_100,
                1,
                json!({"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":[{"type":"tool_use","id":"bash-call","name":"Bash","input":{"run_in_background":true}}],"usage":{"output_tokens":2}}}),
            ),
            make_record(
                "user",
                1_150,
                2,
                json!({"type":"user","toolUseResult":{"backgroundTaskId":"task-1"},"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"bash-call","content":"launched","is_error":false}]}}),
            ),
            make_record(
                "assistant",
                1_300,
                3,
                json!({"type":"assistant","requestId":"req-2","message":{"id":"msg-2","content":[{"type":"text","text":"other work"}],"usage":{"output_tokens":1}}}),
            ),
            make_record(
                "queue-operation",
                1_500,
                4,
                json!({"type":"queue-operation","operation":"enqueue","content":"<task-id>task-1</task-id><tool-use-id>bash-call</tool-use-id><status>failed</status>"}),
            ),
            make_record(
                "user",
                1_510,
                5,
                json!({"type":"user","message":{"role":"user","content":"<tool-use-id>bash-call</tool-use-id><status>failed</status>"}}),
            ),
            make_record(
                "assistant",
                1_600,
                6,
                json!({"type":"assistant","requestId":"req-3","message":{"id":"msg-3","content":[{"type":"text","text":"handled"}],"usage":{"output_tokens":1}}}),
            ),
        ];
        let mut sessions = FxHashMap::default();
        sessions.insert("session-1".to_string(), records.clone());
        let oracle = build_source_fidelity_oracle(&sessions).unwrap();
        assert_eq!(oracle.background_tools, 1);
        assert_eq!(oracle.background_agents, 0);
        assert_eq!(oracle.background_completions_missing, 0);
        assert_eq!(oracle.tool_errors, 1);

        let mut builder = SessionTurnBuilder::new("session-1".to_string(), records, true);
        let first = builder.next_turn(&mut StubTokenizer).unwrap().unwrap();
        let tool = &first.tools[0];
        assert_eq!(tool.ended_at_ms, 1_500);
        assert_eq!(tool.consumer_turn_index, Some(2));
        assert!(tool.is_error);
        assert!(tool.child_session_id.is_none());
        assert_eq!(tool.execution_mode, "background");
    }

    #[test]
    fn child_identity_anonymizes_child_and_parent_consistently() {
        let records = vec![
            make_record(
                "user",
                1_000,
                0,
                json!({"type":"user","isSidechain":true,"message":{"role":"user","content":"task"}}),
            ),
            make_record(
                "assistant",
                2_000,
                1,
                json!({"type":"assistant","isSidechain":true,"message":{"id":"child-1","content":[{"type":"text","text":"done"}],"usage":{"output_tokens":1}}}),
            ),
        ];
        let mut builder = SessionTurnBuilder::new("child-agent".to_string(), records, false);

        let turn = builder.next_turn(&mut StubTokenizer).unwrap().unwrap();

        assert_eq!(turn.export_session_id, anonymized_session_id("child-agent"));
        assert_eq!(
            turn.export_parent_session_id.as_deref(),
            Some(anonymized_session_id("session-1").as_str())
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text-free reduction of one Codex rollout file into the events of the thread that owns it.
//!
//! Every rollout line carries a contiguous `ordinal`. A forked child copies part of its parent's
//! history into ordinals `1..subagent_history_start_ordinal`; those lines describe the parent and
//! are skipped. Each completed model response writes one `token_usage_record`, so requests are
//! reconstructed from those records and the response items between them.

use crate::coding::common::parse_utc_timestamp_ms;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Usage of one completed response, as recorded in `token_usage_record`.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub cached_input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

/// One completed model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexRequest {
    pub response_id: String,
    pub model: String,
    /// The latest input line before the response's first output, or the compaction start.
    pub start_ms: i64,
    pub end_ms: i64,
    pub usage: Usage,
    pub compaction: bool,
}

/// One tool call emitted by a response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexTool {
    pub call_id: String,
    pub name: String,
    /// Index into [`RolloutThread::requests`] of the response that emitted the call.
    pub source_request: Option<usize>,
    pub started_at_ms: i64,
    pub ended_at_ms: Option<i64>,
}

/// A spawn, follow-up, or message this thread queued for another thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSend {
    pub call_id: String,
    pub recipient_thread_id: String,
    pub queued_at_ms: i64,
}

/// A child's completion, recorded in the parent's rollout when the child's turn finishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildCompletion {
    pub child_thread_id: String,
    pub queued_at_ms: i64,
}

/// Responses whose usage Codex never recorded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnrecordedResponses {
    /// Cut off by a queued inter-agent message.
    pub preempted: usize,
    /// Ended by an interrupt, an error, or the end of the turn.
    pub aborted: usize,
    /// Followed by input without a usage record, as in turns written before the rollout
    /// recorded usage.
    pub untracked: usize,
}

/// Text-free events of one Codex thread, from its own rollout lines only.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RolloutThread {
    pub path: PathBuf,
    pub thread_id: String,
    pub parent_thread_id: Option<String>,
    pub forked_from_id: Option<String>,
    pub cwd: String,
    pub requests: Vec<CodexRequest>,
    pub tools: Vec<CodexTool>,
    pub sends: Vec<AgentSend>,
    pub child_completions: Vec<ChildCompletion>,
    pub unrecorded: UnrecordedResponses,
    pub unmatched_compactions: usize,
}

/// Why a rollout file contributes no thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// The file predates per-line ordinals, so inherited fork history cannot be separated.
    NoOrdinals,
    /// The file has no usage records, so no request can be replayed.
    NoUsageRecords,
}

#[derive(Debug, Deserialize)]
struct Line {
    #[serde(default)]
    ordinal: Option<u64>,
    timestamp: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    payload: Payload,
}

/// Union of the payload fields the reduction reads. Serde skips every other field without
/// allocating, so message and tool text is never materialized.
#[derive(Debug, Default, Deserialize)]
struct Payload {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    parent_thread_id: Option<String>,
    #[serde(default)]
    forked_from_id: Option<String>,
    #[serde(default)]
    subagent_history_start_ordinal: Option<u64>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    compaction_response_id: Option<String>,
    #[serde(default)]
    started_at_ms: Option<i64>,
    #[serde(default)]
    item: Option<Item>,
}

#[derive(Debug, Default, Deserialize)]
struct Item {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "kind", default)]
    activity: Option<String>,
    #[serde(default)]
    agent_thread_id: Option<String>,
}

/// Reduces one rollout file to the events of the thread that owns it.
pub fn parse_rollout_file(path: &Path) -> Result<std::result::Result<RolloutThread, SkipReason>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reducer = Reducer {
        thread: RolloutThread {
            path: path.to_path_buf(),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut line_number = 0;
    loop {
        line.clear();
        if reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read {}", path.display()))?
            == 0
        {
            break;
        }
        line_number += 1;
        let complete = line.ends_with(b"\n");
        if line.trim_ascii().is_empty() {
            continue;
        }
        let parsed: Line = match serde_json::from_slice(&line) {
            Ok(parsed) => parsed,
            // A thread that is still running can end in a partially written line, which may
            // stop inside a multi-byte character.
            Err(error) if error.is_eof() && !complete => break,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("invalid rollout line {}:{line_number}", path.display())
                });
            }
        };
        let Some(ordinal) = parsed.ordinal else {
            return Ok(Err(SkipReason::NoOrdinals));
        };
        reducer.apply(ordinal, parsed)?;
    }
    if reducer.thread.requests.is_empty() {
        return Ok(Err(SkipReason::NoUsageRecords));
    }
    Ok(Ok(reducer.thread))
}

#[derive(Debug, Default)]
struct Reducer {
    thread: RolloutThread,
    own_history_start: u64,
    model: String,
    last_input_ms: Option<i64>,
    /// Start of the response whose outputs have been written but whose usage has not.
    response_start_ms: Option<i64>,
    pending_tools: Vec<usize>,
    previous_line_was_usage: bool,
}

impl Reducer {
    fn apply(&mut self, ordinal: u64, line: Line) -> Result<()> {
        let payload = line.payload;
        if ordinal == 0 {
            if line.kind == "session_meta" {
                self.thread.thread_id = payload.id.unwrap_or_default();
                self.thread.parent_thread_id = payload.parent_thread_id;
                self.thread.forked_from_id = payload.forked_from_id;
                self.thread.cwd = payload.cwd.unwrap_or_default();
                self.own_history_start = payload.subagent_history_start_ordinal.unwrap_or(0);
            }
            return Ok(());
        }
        if ordinal < self.own_history_start {
            return Ok(());
        }
        let at_ms = parse_utc_timestamp_ms(&line.timestamp)?;
        let previous_line_was_usage = std::mem::take(&mut self.previous_line_was_usage);
        match (line.kind.as_str(), payload.kind.as_deref()) {
            ("token_usage_record", _) => {
                self.previous_line_was_usage = true;
                let start_ms = self
                    .response_start_ms
                    .take()
                    .or(self.last_input_ms)
                    .unwrap_or(at_ms)
                    .min(at_ms);
                let index = self.thread.requests.len();
                for tool in self.pending_tools.drain(..) {
                    self.thread.tools[tool].source_request = Some(index);
                }
                self.thread.requests.push(CodexRequest {
                    response_id: payload.response_id.unwrap_or_default(),
                    model: self.model.clone(),
                    start_ms,
                    end_ms: at_ms,
                    usage: payload.usage.unwrap_or_default(),
                    compaction: false,
                });
            }
            ("compacted", _) => {
                // The compaction's own usage record is the line just before its checkpoint.
                match self.thread.requests.last_mut() {
                    Some(request)
                        if previous_line_was_usage
                            && payload.compaction_response_id.as_deref()
                                == Some(request.response_id.as_str()) =>
                    {
                        request.compaction = true;
                    }
                    _ => self.thread.unmatched_compactions += 1,
                }
            }
            ("turn_context", _) => {
                if let Some(model) = payload.model {
                    self.model = model;
                }
                self.last_input_ms = Some(at_ms);
            }
            ("world_state", _) => self.last_input_ms = Some(at_ms),
            // Codex also counts tokens when a response is cut off, before the line saying why.
            ("event_msg", Some("token_count")) => self.last_input_ms = Some(at_ms),
            ("event_msg", Some("turn_aborted" | "task_complete")) => {
                self.close_unrecorded_response(|unrecorded| &mut unrecorded.aborted);
            }
            ("event_msg", Some("item_completed")) => self.apply_item(payload, at_ms),
            ("response_item", Some("agent_message")) => {
                // A queued message cuts off the in-flight response without a usage record.
                self.close_unrecorded_response(|unrecorded| &mut unrecorded.preempted);
                self.last_input_ms = Some(at_ms);
            }
            ("response_item", Some("message")) if payload.role.as_deref() != Some("assistant") => {
                self.close_unrecorded_response(|unrecorded| &mut unrecorded.untracked);
                self.last_input_ms = Some(at_ms);
            }
            ("response_item", Some("function_call_output" | "custom_tool_call_output")) => {
                self.close_unrecorded_response(|unrecorded| &mut unrecorded.untracked);
                self.last_input_ms = Some(at_ms);
                if let Some(call_id) = payload.call_id
                    && let Some(tool) = self
                        .thread
                        .tools
                        .iter_mut()
                        .rev()
                        .find(|tool| tool.call_id == call_id && tool.ended_at_ms.is_none())
                {
                    tool.ended_at_ms = Some(at_ms);
                }
            }
            (
                "response_item",
                Some("reasoning" | "message" | "function_call" | "custom_tool_call"),
            ) => {
                self.response_start_ms
                    .get_or_insert(self.last_input_ms.unwrap_or(at_ms));
                if let Some(call_id) = payload.call_id {
                    self.pending_tools.push(self.thread.tools.len());
                    self.thread.tools.push(CodexTool {
                        call_id,
                        name: payload.name.unwrap_or_default(),
                        source_request: None,
                        started_at_ms: at_ms,
                        ended_at_ms: None,
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Ends a response whose outputs were written but whose usage never was.
    ///
    /// Tool outputs follow the usage record, so input written while a response is open means the
    /// response ended without one. Its tool calls stay in the thread without a source request.
    fn close_unrecorded_response(
        &mut self,
        counter: impl FnOnce(&mut UnrecordedResponses) -> &mut usize,
    ) {
        if self.response_start_ms.take().is_some() {
            *counter(&mut self.thread.unrecorded) += 1;
            self.pending_tools.clear();
        }
    }

    fn apply_item(&mut self, payload: Payload, at_ms: i64) {
        let Some(item) = payload.item else {
            return;
        };
        match item.kind.as_deref() {
            Some("ContextCompaction") => {
                if let Some(request) = self.thread.requests.last_mut()
                    && request.compaction
                    && let Some(started_at_ms) = payload.started_at_ms
                    && started_at_ms <= request.end_ms
                {
                    request.start_ms = started_at_ms;
                }
            }
            Some("SubAgentActivity") => {
                let (Some(id), Some(agent_thread_id)) = (item.id, item.agent_thread_id) else {
                    return;
                };
                match item.activity.as_deref() {
                    Some("started" | "interacted") => self.thread.sends.push(AgentSend {
                        call_id: id,
                        recipient_thread_id: agent_thread_id,
                        queued_at_ms: at_ms,
                    }),
                    Some("completed") => self.thread.child_completions.push(ChildCompletion {
                        child_thread_id: agent_thread_id,
                        queued_at_ms: at_ms,
                    }),
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{SkipReason, parse_rollout_file};
    use crate::coding::common::parse_utc_timestamp_ms;
    use serde_json::{Value, json};
    use std::io::Write;
    use tempfile::NamedTempFile;

    pub(crate) fn rollout(lines: &[Value]) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        for (ordinal, line) in lines.iter().enumerate() {
            let mut line = line.clone();
            line["ordinal"] = json!(ordinal);
            writeln!(file, "{line}").unwrap();
        }
        file
    }

    pub(crate) fn line(second: u32, kind: &str, payload: Value) -> Value {
        json!({"timestamp": format!("2026-09-30T10:{:02}:{:02}.000Z", second / 60, second % 60), "type": kind, "payload": payload})
    }

    fn usage(second: u32, response_id: &str, input: u64, cached: u64, output: u64) -> Value {
        line(
            second,
            "token_usage_record",
            json!({"response_id": response_id, "usage": {"input_tokens": input, "cached_input_tokens": cached, "output_tokens": output}}),
        )
    }

    #[test]
    fn forked_child_skips_inherited_history_and_counts_preemption() {
        let file = rollout(&[
            line(
                0,
                "session_meta",
                json!({"id": "child", "parent_thread_id": "parent", "forked_from_id": "parent", "subagent_history_start_ordinal": 4, "cwd": "/repo"}),
            ),
            line(0, "session_meta", json!({"id": "parent"})),
            line(0, "token_usage_record", json!({"response_id": "inherited"})),
            line(
                0,
                "compacted",
                json!({"compaction_response_id": "inherited"}),
            ),
            line(1, "turn_context", json!({"model": "gpt"})),
            line(2, "response_item", json!({"type": "reasoning"})),
            line(
                3,
                "response_item",
                json!({"type": "agent_message", "author": "/root"}),
            ),
            line(
                5,
                "response_item",
                json!({"type": "function_call", "call_id": "call-1", "name": "exec"}),
            ),
            usage(6, "resp-1", 100, 90, 10),
            line(
                7,
                "response_item",
                json!({"type": "function_call_output", "call_id": "call-1"}),
            ),
        ]);

        let thread = parse_rollout_file(file.path()).unwrap().unwrap();
        assert_eq!(thread.thread_id, "child");
        assert_eq!(thread.forked_from_id.as_deref(), Some("parent"));
        assert_eq!(thread.unrecorded.preempted, 1);
        assert_eq!(thread.unmatched_compactions, 0);
        assert_eq!(thread.requests.len(), 1);
        let request = &thread.requests[0];
        assert_eq!(
            (request.response_id.as_str(), request.model.as_str()),
            ("resp-1", "gpt")
        );
        // The cut-off response's start does not leak into the next one.
        assert_eq!(request.end_ms - request.start_ms, 3_000);
        assert_eq!(thread.tools[0].source_request, Some(0));
        assert_eq!(thread.tools[0].ended_at_ms, Some(request.end_ms + 1_000));
    }

    #[test]
    fn compaction_is_the_usage_record_before_its_checkpoint() {
        let started_at_ms = parse_utc_timestamp_ms("2026-09-30T10:00:02.000Z").unwrap();
        let file = rollout(&[
            line(0, "session_meta", json!({"id": "root"})),
            line(
                1,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            usage(5, "resp-compact", 200, 190, 20),
            line(
                5,
                "compacted",
                json!({"compaction_response_id": "resp-compact"}),
            ),
            line(
                5,
                "event_msg",
                json!({"type": "item_completed", "started_at_ms": started_at_ms, "item": {"type": "ContextCompaction", "id": "c"}}),
            ),
            line(
                6,
                "event_msg",
                json!({"type": "item_completed", "item": {"type": "SubAgentActivity", "id": "call-2", "kind": "interacted", "agent_thread_id": "child"}}),
            ),
        ]);

        let thread = parse_rollout_file(file.path()).unwrap().unwrap();
        assert!(thread.requests[0].compaction);
        assert_eq!(thread.requests[0].start_ms, started_at_ms);
        assert_eq!(thread.sends[0].recipient_thread_id, "child");
    }

    #[test]
    fn partially_written_final_line_is_ignored() {
        let mut file = rollout(&[
            line(0, "session_meta", json!({"id": "root"})),
            line(
                1,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            line(2, "response_item", json!({"type": "reasoning"})),
            usage(3, "resp-1", 100, 0, 10),
        ]);
        // Cut a write off inside the three-byte encoding of an em dash.
        let partial = format!(
            r#"{{"ordinal":4,"timestamp":"2026-09-30T10:00:04.000Z","type":"response_item","payload":{{"type":"message","role":"user","content":"a{}"#,
            '\u{2014}'
        );
        file.write_all(&partial.as_bytes()[..partial.len() - 1])
            .unwrap();
        assert_eq!(
            parse_rollout_file(file.path())
                .unwrap()
                .unwrap()
                .requests
                .len(),
            1
        );

        let mut corrupt = rollout(&[line(0, "session_meta", json!({"id": "root"}))]);
        writeln!(corrupt, "{{\"ordinal\":1,").unwrap();
        writeln!(corrupt, "{}", usage(3, "resp-1", 100, 0, 10)).unwrap();
        assert!(parse_rollout_file(corrupt.path()).is_err());
    }

    #[test]
    fn outputs_without_usage_do_not_stretch_the_next_request() {
        let file = rollout(&[
            line(0, "session_meta", json!({"id": "root"})),
            line(
                1,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            line(
                2,
                "response_item",
                json!({"type": "function_call", "call_id": "legacy", "name": "exec"}),
            ),
            line(
                3,
                "response_item",
                json!({"type": "function_call_output", "call_id": "legacy"}),
            ),
            line(
                10,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            line(11, "response_item", json!({"type": "reasoning"})),
            usage(12, "resp-1", 100, 0, 10),
        ]);

        let thread = parse_rollout_file(file.path()).unwrap().unwrap();
        assert_eq!(thread.unrecorded.untracked, 1);
        assert_eq!(
            thread.requests[0].end_ms - thread.requests[0].start_ms,
            2_000
        );
        assert_eq!(thread.tools[0].source_request, None);
    }

    #[test]
    fn files_without_usage_or_ordinals_are_skipped() {
        let empty = rollout(&[line(0, "session_meta", json!({"id": "root"}))]);
        assert_eq!(
            parse_rollout_file(empty.path()).unwrap(),
            Err(SkipReason::NoUsageRecords)
        );

        let mut legacy = NamedTempFile::with_suffix(".jsonl").unwrap();
        writeln!(legacy, "{}", line(0, "session_meta", json!({"id": "root"}))).unwrap();
        assert_eq!(
            parse_rollout_file(legacy.path()).unwrap(),
            Err(SkipReason::NoOrdinals)
        );
    }
}

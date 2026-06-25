// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude-specific request-trace export orchestration.
//!
//! Handles session scheduling, parallel tokenization with text-overlap reuse,
//! and global ordering across sessions.

use crate::coding::claude::parser::{SessionTurnBuilder, TraceRecord, TurnDraft};
use crate::coding::tokenizer::{TokenizerFactory, TokenizerWorker, last_word_overlap_start};
use anyhow::{Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use dynamo_data_gen::{sequence_hashes_for_tokens, write_empty_files};
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone, Copy)]
pub struct ExportConfig {
    pub block_size: usize,
    pub delta_overlap_words: usize,
    pub tokenizer_workers: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExportStats {
    pub row_count: usize,
    pub tool_row_count: usize,
    pub sidecar_count: usize,
    pub max_heap_len: usize,
}

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
struct HeapEntry {
    request_start_ms: i64,
    turn_index: usize,
    export_session_id: String,
    session_id: String,
}

#[derive(Debug)]
struct OverlapBase {
    previous_text: String,
    previous_tokens: Vec<u32>,
}

#[derive(Debug)]
struct ReadyTurn {
    current_text: String,
    tokens: Vec<u32>,
}

#[derive(Debug)]
struct HeadTurn {
    turn: TurnDraft,
    turn_key: u64,
    scheduled: bool,
    ready: Option<ReadyTurn>,
}

#[derive(Debug)]
struct SessionState {
    builder: SessionTurnBuilder,
    head: Option<HeadTurn>,
    overlap_base: Option<OverlapBase>,
    replay_base: Option<Vec<u32>>,
    next_turn_key: u64,
}

#[derive(Debug)]
struct TokenizeJob {
    session_id: String,
    turn_key: u64,
    current_text: String,
    overlap_start: Option<usize>,
    previous_overlap_text: Option<String>,
    previous_tokens: Option<Vec<u32>>,
    overlap_words: usize,
}

#[derive(Debug)]
struct TokenizeResponse {
    session_id: String,
    turn_key: u64,
    outcome: Result<ReadyTurn, String>,
}

pub fn write_streamed_request_trace_rows<F>(
    output_path: &Path,
    sidecar_path: &Path,
    sessions: FxHashMap<String, Vec<TraceRecord>>,
    preserve_session_ids: bool,
    tokenizer_factory: F,
    config: ExportConfig,
) -> Result<ExportStats>
where
    F: TokenizerFactory,
{
    if config.block_size == 0 {
        bail!("block_size must be greater than 0");
    }
    if config.tokenizer_workers == 0 {
        bail!("tokenizer_workers must be greater than 0");
    }

    let mut parser_tokenizer = tokenizer_factory.create_worker()?;
    let mut states = FxHashMap::default();
    let mut heap = BinaryHeap::new();
    let mut unscheduled_sessions = VecDeque::new();
    let mut stats = ExportStats::default();

    for (session_id, records) in sessions {
        let mut builder =
            SessionTurnBuilder::new(session_id.clone(), records, preserve_session_ids);
        let Some(first_turn) = builder.next_turn(&mut parser_tokenizer)? else {
            continue;
        };

        let head = HeadTurn {
            turn: first_turn,
            turn_key: 0,
            scheduled: false,
            ready: None,
        };
        states.insert(
            session_id.clone(),
            SessionState {
                builder,
                head: Some(head),
                overlap_base: None,
                replay_base: None,
                next_turn_key: 1,
            },
        );
        push_heap_entry(&mut heap, &session_id, states.get(&session_id).unwrap());
        unscheduled_sessions.push_back(session_id);
    }

    if states.is_empty() {
        write_empty_files(output_path, Some(sidecar_path))?;
        return Ok(stats);
    }

    stats.max_heap_len = heap.len();
    let trace_start_ms = states
        .values()
        .filter_map(|state| state.head.as_ref())
        .map(|head| head.turn.request_start_ms)
        .min()
        .unwrap_or_default();
    let mut output = create_writer(output_path)?;
    let mut sidecar = create_writer(sidecar_path)?;

    let (job_tx, job_rx) = bounded::<TokenizeJob>(config.tokenizer_workers);
    let (result_tx, result_rx) = unbounded::<TokenizeResponse>();
    let workers = spawn_tokenizer_workers(
        tokenizer_factory,
        config.tokenizer_workers,
        job_rx,
        result_tx,
    );

    let mut inflight_jobs = 0_usize;
    while !heap.is_empty() {
        schedule_pending_jobs(
            &mut states,
            &mut unscheduled_sessions,
            &job_tx,
            &mut inflight_jobs,
            config.delta_overlap_words,
            config.tokenizer_workers,
        )?;

        let Some(Reverse(entry)) = heap.peek() else {
            break;
        };
        let head_ready = states
            .get(&entry.session_id)
            .and_then(|state| state.head.as_ref())
            .and_then(|head| head.ready.as_ref())
            .is_some();
        if !head_ready {
            let response = result_rx
                .recv()
                .map_err(|_| anyhow!("tokenizer worker channel closed unexpectedly"))?;
            inflight_jobs = inflight_jobs.saturating_sub(1);
            apply_tokenize_response(&mut states, response)?;
            continue;
        }

        let Reverse(entry) = heap.pop().unwrap();
        let session_id = entry.session_id.clone();
        let (turn, ready_turn) = {
            let state = states
                .get_mut(&session_id)
                .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
            let mut head = state
                .head
                .take()
                .ok_or_else(|| anyhow!("missing head for session {}", session_id))?;
            let ready_turn = head
                .ready
                .take()
                .ok_or_else(|| anyhow!("missing tokenized result for session {}", session_id))?;
            (head.turn, ready_turn)
        };

        let next_turn = {
            let state = states
                .get_mut(&session_id)
                .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
            state.builder.next_turn(&mut parser_tokenizer)?
        };
        let replay_tokens = {
            let state = states
                .get(&session_id)
                .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
            materialize_replay_tokens(&turn, &ready_turn.tokens, state.replay_base.as_deref())
        };
        let input_sequence_hashes = sequence_hashes_for_tokens(&replay_tokens, config.block_size)?;
        let mut agent_context = Map::from_iter([(
            "session_id".to_string(),
            Value::String(turn.export_session_id.clone()),
        )]);
        if let Some(parent_session_id) = &turn.export_parent_session_id {
            agent_context.insert(
                "parent_session_id".to_string(),
                Value::String(parent_session_id.clone()),
            );
        }
        let event = json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": nonnegative_ms(turn.assistant_end_ms),
            "event_source": "harness",
            "agent_context": agent_context,
            "request": {
                "request_id": format!("claude:{}:{}", turn.export_session_id, turn.turn_index),
                "model": turn.model,
                "input_tokens": replay_tokens.len(),
                "output_tokens": turn.output_length,
                "cached_tokens": turn.cache_read_input_tokens,
                "request_received_ms": nonnegative_ms(turn.request_start_ms),
                "total_time_ms": (turn.assistant_end_ms - turn.request_start_ms).max(0) as f64,
                "replay": {
                    "trace_block_size": config.block_size,
                    "input_length": replay_tokens.len(),
                    "input_sequence_hashes": input_sequence_hashes,
                }
            }
        });
        let row = json!({
            "timestamp": nonnegative_ms(turn.assistant_end_ms - trace_start_ms),
            "event": event,
        });

        write_json_line(&mut output, &row)?;
        for tool in &turn.tools {
            let event_type = if tool.is_error {
                "tool_error"
            } else {
                "tool_end"
            };
            let tool_row = json!({
                "timestamp": nonnegative_ms(tool.ended_at_ms - trace_start_ms),
                "event": {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": event_type,
                    "event_time_unix_ms": nonnegative_ms(tool.ended_at_ms),
                    "event_source": "harness",
                    "agent_context": agent_context,
                    "tool": {
                        "tool_call_id": tool.tool_call_id,
                        "tool_class": tool.tool_class,
                        "started_at_unix_ms": nonnegative_ms(tool.started_at_ms),
                        "ended_at_unix_ms": nonnegative_ms(tool.ended_at_ms),
                        "duration_ms": (tool.ended_at_ms - tool.started_at_ms).max(0) as f64,
                        "status": if tool.is_error { "error" } else { "succeeded" },
                        "output_bytes": tool.output_bytes,
                        "error_type": if tool.is_error { Some("claude_tool_error") } else { None },
                    }
                }
            });
            write_json_line(&mut output, &tool_row)?;
            stats.tool_row_count += 1;
        }
        write_json_line(&mut sidecar, &turn.sidecar)?;
        stats.row_count += 1;
        stats.sidecar_count += 1;

        let state = states
            .get_mut(&session_id)
            .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
        state.overlap_base = Some(OverlapBase {
            previous_text: ready_turn.current_text,
            previous_tokens: ready_turn.tokens,
        });
        state.replay_base = Some(replay_tokens);

        if let Some(next_turn) = next_turn {
            let turn_key = state.next_turn_key;
            state.next_turn_key += 1;
            state.head = Some(HeadTurn {
                turn: next_turn,
                turn_key,
                scheduled: false,
                ready: None,
            });
            push_heap_entry(&mut heap, &session_id, state);
            unscheduled_sessions.push_back(session_id);
            stats.max_heap_len = stats.max_heap_len.max(heap.len());
            continue;
        }

        states.remove(&session_id);
    }

    drop(job_tx);
    for worker in workers {
        worker
            .join()
            .map_err(|_| anyhow!("tokenizer worker panicked"))?;
    }
    output.flush()?;
    sidecar.flush()?;
    Ok(stats)
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn write_json_line(writer: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn nonnegative_ms(value: i64) -> u64 {
    value.max(0) as u64
}

fn materialize_replay_tokens(
    turn: &TurnDraft,
    rendered_tokens: &[u32],
    previous_tokens: Option<&[u32]>,
) -> Vec<u32> {
    let Some(input_length) = turn.observed_input_length else {
        return rendered_tokens.to_vec();
    };

    let cached_length = turn.cache_read_input_tokens.unwrap_or(0).min(input_length);
    let mut tokens = Vec::with_capacity(input_length);
    if let Some(previous_tokens) = previous_tokens {
        tokens.extend_from_slice(&previous_tokens[..cached_length.min(previous_tokens.len())]);
    }
    while tokens.len() < cached_length {
        tokens.push(synthetic_token(
            &turn.export_session_id,
            turn.turn_index.saturating_sub(1),
            tokens.len(),
            rendered_tokens,
        ));
    }
    while tokens.len() < input_length {
        tokens.push(synthetic_token(
            &turn.export_session_id,
            turn.turn_index,
            tokens.len(),
            rendered_tokens,
        ));
    }
    tokens
}

fn synthetic_token(
    session_id: &str,
    turn_index: usize,
    position: usize,
    rendered_tokens: &[u32],
) -> u32 {
    let mut hash = 0x811c_9dc5_u32;
    for byte in session_id.bytes() {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    hash = (hash ^ turn_index as u32).wrapping_mul(0x0100_0193);
    hash = (hash ^ position as u32).wrapping_mul(0x0100_0193);
    if rendered_tokens.is_empty() {
        hash
    } else {
        hash ^ rendered_tokens[position % rendered_tokens.len()]
    }
}

fn push_heap_entry(
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    session_id: &str,
    state: &SessionState,
) {
    if let Some(head) = state.head.as_ref() {
        heap.push(Reverse(HeapEntry {
            request_start_ms: head.turn.request_start_ms,
            turn_index: head.turn.turn_index,
            export_session_id: head.turn.export_session_id.clone(),
            session_id: session_id.to_string(),
        }));
    }
}

fn schedule_pending_jobs(
    states: &mut FxHashMap<String, SessionState>,
    unscheduled_sessions: &mut VecDeque<String>,
    job_tx: &Sender<TokenizeJob>,
    inflight_jobs: &mut usize,
    overlap_words: usize,
    worker_limit: usize,
) -> Result<()> {
    while *inflight_jobs < worker_limit {
        let Some(session_id) = unscheduled_sessions.pop_front() else {
            return Ok(());
        };
        let Some(state) = states.get_mut(&session_id) else {
            continue;
        };
        let Some(head) = state.head.as_mut() else {
            continue;
        };
        if head.scheduled || head.ready.is_some() {
            continue;
        }

        let overlap_base = state.overlap_base.take();
        let current_text = std::mem::take(&mut head.turn.input_text);
        let (overlap_start, previous_overlap_text, previous_tokens) =
            prepare_overlap_inputs(overlap_base, &current_text, overlap_words);
        let job = TokenizeJob {
            session_id: session_id.clone(),
            turn_key: head.turn_key,
            current_text,
            overlap_start,
            previous_overlap_text,
            previous_tokens,
            overlap_words,
        };
        job_tx
            .send(job)
            .map_err(|_| anyhow!("failed to schedule tokenization job"))?;
        head.scheduled = true;
        *inflight_jobs += 1;
    }
    Ok(())
}

fn apply_tokenize_response(
    states: &mut FxHashMap<String, SessionState>,
    response: TokenizeResponse,
) -> Result<()> {
    let Some(state) = states.get_mut(&response.session_id) else {
        return Ok(());
    };
    let Some(head) = state.head.as_mut() else {
        return Ok(());
    };
    if head.turn_key != response.turn_key {
        return Ok(());
    }
    head.scheduled = false;
    match response.outcome {
        Ok(ready) => {
            head.ready = Some(ready);
            Ok(())
        }
        Err(message) => bail!("{message}"),
    }
}

fn prepare_overlap_inputs(
    overlap_base: Option<OverlapBase>,
    current_text: &str,
    overlap_words: usize,
) -> (Option<usize>, Option<String>, Option<Vec<u32>>) {
    if overlap_words == 0 {
        return (None, None, None);
    }
    let Some(overlap_base) = overlap_base else {
        return (None, None, None);
    };
    if !current_text.starts_with(&overlap_base.previous_text) {
        return (None, None, None);
    }

    let overlap_start = last_word_overlap_start(&overlap_base.previous_text, overlap_words);
    (
        Some(overlap_start),
        Some(overlap_base.previous_text[overlap_start..].to_string()),
        Some(overlap_base.previous_tokens),
    )
}

fn spawn_tokenizer_workers<F>(
    factory: F,
    worker_count: usize,
    job_rx: Receiver<TokenizeJob>,
    result_tx: Sender<TokenizeResponse>,
) -> Vec<JoinHandle<()>>
where
    F: TokenizerFactory,
{
    (0..worker_count)
        .map(|_| {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let factory = factory.clone();
            thread::spawn(move || {
                let mut tokenizer = match factory.create_worker() {
                    Ok(tokenizer) => tokenizer,
                    Err(error) => {
                        let _ = result_tx.send(TokenizeResponse {
                            session_id: "__worker_init__".to_string(),
                            turn_key: 0,
                            outcome: Err(format!(
                                "failed to initialize tokenizer worker: {error:#}"
                            )),
                        });
                        return;
                    }
                };
                while let Ok(job) = job_rx.recv() {
                    let outcome = tokenize_job(&mut tokenizer, &job)
                        .map(|tokens| ReadyTurn {
                            current_text: job.current_text,
                            tokens,
                        })
                        .map_err(|error| {
                            format!("failed to tokenize session {}: {error:#}", job.session_id)
                        });
                    let _ = result_tx.send(TokenizeResponse {
                        session_id: job.session_id,
                        turn_key: job.turn_key,
                        outcome,
                    });
                }
            })
        })
        .collect()
}

fn tokenize_job(tokenizer: &mut impl TokenizerWorker, job: &TokenizeJob) -> Result<Vec<u32>> {
    let Some(overlap_start) = job.overlap_start else {
        return tokenizer.encode(&job.current_text);
    };
    let Some(previous_overlap_text) = job.previous_overlap_text.as_deref() else {
        return tokenizer.encode(&job.current_text);
    };
    let Some(previous_tokens) = job.previous_tokens.as_deref() else {
        return tokenizer.encode(&job.current_text);
    };
    if job.overlap_words == 0 || !job.current_text.is_char_boundary(overlap_start) {
        return tokenizer.encode(&job.current_text);
    }

    let previous_overlap_tokens = tokenizer.encode(previous_overlap_text)?;
    let prefix_token_count = previous_tokens
        .len()
        .saturating_sub(previous_overlap_tokens.len());
    let suffix_tokens = tokenizer.encode(&job.current_text[overlap_start..])?;
    let mut merged = Vec::with_capacity(prefix_token_count + suffix_tokens.len());
    merged.extend_from_slice(&previous_tokens[..prefix_token_count]);
    merged.extend(suffix_tokens);
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::{
        ExportConfig, HeadTurn, ReadyTurn, SessionState, TurnDraft, apply_tokenize_response,
        write_streamed_request_trace_rows,
    };
    use crate::coding::claude::parser::{SessionTurnBuilder, TraceRecord};
    use crate::coding::tokenizer::{TokenizerFactory, TokenizerWorker};
    use anyhow::Result;
    use rustc_hash::FxHashMap;
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;
    use tempfile::TempDir;

    #[derive(Clone, Default)]
    struct StubFactory {
        calls: Arc<Mutex<Vec<String>>>,
    }

    struct StubWorker {
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl TokenizerFactory for StubFactory {
        type Worker = StubWorker;

        fn create_worker(&self) -> Result<Self::Worker> {
            Ok(StubWorker {
                calls: self.calls.clone(),
            })
        }
    }

    impl TokenizerWorker for StubWorker {
        fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
            if text.contains("slow") {
                thread::sleep(Duration::from_millis(20));
            }
            self.calls.lock().unwrap().push(text.to_string());
            Ok(text
                .split_whitespace()
                .map(|word| word.len() as u32)
                .collect())
        }
    }

    fn make_record(
        session_id: &str,
        row_type: &str,
        timestamp_ms: i64,
        source_order: u64,
        raw: Value,
    ) -> TraceRecord {
        TraceRecord {
            session_id: session_id.to_string(),
            parent_session_id: None,
            row_type: row_type.to_string(),
            timestamp_ms,
            source_order,
            raw,
        }
    }

    #[test]
    fn stale_result_is_dropped_by_turn_key() {
        let mut states = FxHashMap::default();
        states.insert(
            "session-a".to_string(),
            SessionState {
                builder: SessionTurnBuilder::new("session-a".to_string(), Vec::new(), true),
                head: Some(HeadTurn {
                    turn: TurnDraft {
                        session_id: "session-a".to_string(),
                        export_session_id: "session-a".to_string(),
                        export_parent_session_id: None,
                        turn_index: 1,
                        model: "test-model".to_string(),
                        input_text: String::new(),
                        output_length: 1,
                        observed_input_length: None,
                        cache_read_input_tokens: None,
                        request_start_ms: 1,
                        assistant_start_ms: 1,
                        assistant_end_ms: 2,
                        delay_ms: None,
                        tools: Vec::new(),
                        sidecar: json!({}),
                    },
                    turn_key: 9,
                    scheduled: true,
                    ready: None,
                }),
                overlap_base: None,
                replay_base: None,
                next_turn_key: 10,
            },
        );

        apply_tokenize_response(
            &mut states,
            super::TokenizeResponse {
                session_id: "session-a".to_string(),
                turn_key: 7,
                outcome: Ok(ReadyTurn {
                    current_text: "stale".to_string(),
                    tokens: vec![1],
                }),
            },
        )
        .unwrap();

        assert!(
            states
                .get("session-a")
                .unwrap()
                .head
                .as_ref()
                .unwrap()
                .ready
                .is_none()
        );
    }

    #[test]
    fn streamed_writer_preserves_global_order_with_parallel_tokenization() {
        let temp = TempDir::new().unwrap();
        let output_path = temp.path().join("trace.jsonl");
        let sidecar_path = temp.path().join("trace.sidecar.jsonl");
        let mut sessions = FxHashMap::default();
        sessions.insert(
            "session-a".to_string(),
            vec![
                make_record(
                    "session-a",
                    "user",
                    1_000,
                    0,
                    json!({"type":"user","message":{"role":"user","content":"slow first a"}}),
                ),
                make_record(
                    "session-a",
                    "assistant",
                    2_000,
                    1,
                    json!({"type":"assistant","message":{"id":"a-1","content":[{"type":"text","text":"done a"}],"usage":{"input_tokens":4,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":3}}}),
                ),
                make_record(
                    "session-a",
                    "user",
                    2_100,
                    2,
                    json!({"type":"user","message":{"role":"user","content":"follow a"}}),
                ),
                make_record(
                    "session-a",
                    "assistant",
                    2_200,
                    3,
                    json!({"type":"assistant","message":{"id":"a-2","content":[{"type":"text","text":"done a 2"}],"usage":{"input_tokens":2,"cache_read_input_tokens":4,"cache_creation_input_tokens":0,"output_tokens":4}}}),
                ),
            ],
        );
        sessions.insert(
            "session-b".to_string(),
            vec![
                make_record(
                    "session-b",
                    "user",
                    900,
                    4,
                    json!({"type":"user","message":{"role":"user","content":"first b"}}),
                ),
                make_record(
                    "session-b",
                    "assistant",
                    1_100,
                    5,
                    json!({"type":"assistant","message":{"id":"b-1","content":[{"type":"text","text":"done b"}],"usage":{"output_tokens":2}}}),
                ),
            ],
        );

        let stats = write_streamed_request_trace_rows(
            &output_path,
            &sidecar_path,
            sessions,
            true,
            StubFactory::default(),
            ExportConfig {
                block_size: 2,
                delta_overlap_words: 50,
                tokenizer_workers: 2,
            },
        )
        .unwrap();

        let rows = std::fs::read_to_string(&output_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let sidecar_rows = std::fs::read_to_string(&sidecar_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(stats.row_count, 3);
        assert_eq!(stats.sidecar_count, 3);
        assert!(stats.max_heap_len <= 2);
        assert_eq!(rows.len(), 3);
        assert_eq!(sidecar_rows.len(), 3);
        assert_eq!(rows[0]["event"]["agent_context"]["session_id"], "session-b");
        assert_eq!(rows[1]["event"]["agent_context"]["session_id"], "session-a");
        assert!(
            rows[1]["event"]["agent_context"]
                .get("session_final")
                .is_none()
        );
        assert_eq!(rows[2]["event"]["request"]["request_received_ms"], 2_100);
        assert_eq!(rows[1]["event"]["request"]["replay"]["input_length"], 4);
        assert_eq!(rows[2]["event"]["request"]["replay"]["input_length"], 6);
        let first_hashes = rows[1]["event"]["request"]["replay"]["input_sequence_hashes"]
            .as_array()
            .unwrap();
        let second_hashes = rows[2]["event"]["request"]["replay"]["input_sequence_hashes"]
            .as_array()
            .unwrap();
        assert_eq!(first_hashes.as_slice(), &second_hashes[..2]);
    }

    #[test]
    fn streamed_writer_emits_canonical_tool_terminal_events() {
        use dynamo_data_gen::request_trace::load::load_request_trace_records;

        let temp = TempDir::new().unwrap();
        let output_path = temp.path().join("trace.jsonl");
        let sidecar_path = temp.path().join("trace.sidecar.jsonl");
        let mut sessions = FxHashMap::default();
        sessions.insert(
            "session-a".to_string(),
            vec![
                make_record(
                    "session-a",
                    "user",
                    1_000,
                    0,
                    json!({"type":"user","message":{"role":"user","content":"run"}}),
                ),
                make_record(
                    "session-a",
                    "assistant",
                    1_100,
                    1,
                    json!({"type":"assistant","requestId":"req-1","message":{"id":"a-1","content":[{"type":"tool_use","id":"raw-1","name":"Bash","input":{}}],"usage":{"input_tokens":2,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":3}}}),
                ),
                make_record(
                    "session-a",
                    "user",
                    1_200,
                    2,
                    json!({"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"raw-1","content":"bad","is_error":true}]}}),
                ),
            ],
        );

        let stats = write_streamed_request_trace_rows(
            &output_path,
            &sidecar_path,
            sessions,
            true,
            StubFactory::default(),
            ExportConfig {
                block_size: 2,
                delta_overlap_words: 50,
                tokenizer_workers: 1,
            },
        )
        .unwrap();

        assert_eq!(stats.row_count, 1);
        assert_eq!(stats.tool_row_count, 1);
        let rows = std::fs::read_to_string(&output_path).unwrap();
        assert!(rows.lines().any(|line| {
            let row: Value = serde_json::from_str(line).unwrap();
            row["event"]["event_type"] == "tool_error"
                && row["event"]["tool"]["tool_class"] == "Bash"
        }));
        let loaded = load_request_trace_records(&[output_path]).unwrap();
        assert_eq!(loaded.tools.len(), 1);
    }

    #[test]
    fn request_trace_preserves_claude_child_identity_and_converts_to_agentic() {
        use dynamo_data_gen::request_trace::{
            agentic::lower_agentic_mooncake_rows, load::load_request_trace_records,
        };

        let temp = TempDir::new().unwrap();
        let output_path = temp.path().join("trace.jsonl");
        let sidecar_path = temp.path().join("trace.sidecar.jsonl");
        let mut sessions = FxHashMap::default();
        sessions.insert(
            "root-session".to_string(),
            vec![
                make_record(
                    "root-session",
                    "user",
                    1_000,
                    0,
                    json!({"type":"user","message":{"role":"user","content":"spawn child"}}),
                ),
                make_record(
                    "root-session",
                    "assistant",
                    1_100,
                    1,
                    json!({"type":"assistant","message":{"id":"root-1","content":[{"type":"text","text":"spawning"}],"usage":{"output_tokens":2}}}),
                ),
                make_record(
                    "root-session",
                    "user",
                    2_000,
                    4,
                    json!({"type":"user","message":{"role":"user","content":"child done"}}),
                ),
                make_record(
                    "root-session",
                    "assistant",
                    2_100,
                    5,
                    json!({"type":"assistant","message":{"id":"root-2","content":[{"type":"text","text":"finished"}],"usage":{"output_tokens":1}}}),
                ),
            ],
        );
        sessions.insert(
            "child-agent".to_string(),
            vec![
                make_record(
                    "root-session",
                    "user",
                    1_200,
                    2,
                    json!({"type":"user","isSidechain":true,"agentId":"child-agent","message":{"role":"user","content":"investigate"}}),
                ),
                make_record(
                    "root-session",
                    "assistant",
                    1_300,
                    3,
                    json!({"type":"assistant","isSidechain":true,"agentId":"child-agent","message":{"id":"child-1","content":[{"type":"text","text":"result"}],"usage":{"output_tokens":1}}}),
                ),
            ],
        );

        write_streamed_request_trace_rows(
            &output_path,
            &sidecar_path,
            sessions,
            true,
            StubFactory::default(),
            ExportConfig {
                block_size: 2,
                delta_overlap_words: 50,
                tokenizer_workers: 2,
            },
        )
        .unwrap();

        let rows = std::fs::read_to_string(&output_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let child = rows
            .iter()
            .find(|row| row["event"]["agent_context"]["session_id"] == "child-agent")
            .unwrap();
        assert_eq!(
            child["event"]["agent_context"]["parent_session_id"],
            "root-session"
        );
        assert!(
            child["event"]["agent_context"]
                .get("session_final")
                .is_none()
        );

        let loaded = load_request_trace_records(&[output_path]).unwrap();
        let mut agentic_rows = Vec::new();
        lower_agentic_mooncake_rows(loaded, |_, row| {
            agentic_rows.push(row);
            Ok(())
        })
        .unwrap();
        assert_eq!(agentic_rows.len(), 3);
        assert!(agentic_rows.iter().any(|row| !row.branches.is_empty()));
    }
}

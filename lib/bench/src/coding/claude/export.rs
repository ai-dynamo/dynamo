// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude-specific orchestration on top of the generic Mooncake primitives.
//!
//! Handles session scheduling, parallel tokenization with text-overlap reuse,
//! and global ordering across sessions. The actual row schema, hash-id
//! mapping, and JSONL writer all live in [`crate::coding::mooncake`] so that
//! other producers can reuse them without depending on Claude parser types.

use crate::coding::claude::parser::{SessionTurnBuilder, TraceRecord, TurnDraft};
use crate::coding::mooncake::{
    MooncakeJsonlWriter, MooncakeRow, RollingHashIdMapper, write_empty_files,
};
use crate::coding::tokenizer::{TokenizerFactory, TokenizerWorker, last_word_overlap_start};
use anyhow::{Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
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
    pub sidecar_count: usize,
    pub max_heap_len: usize,
}

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
struct HeapEntry {
    assistant_start_ms: i64,
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

pub fn write_streamed_mooncake_rows<F>(
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
    let mut global_trace_start_ms: Option<i64> = None;
    let mut stats = ExportStats::default();

    for (session_id, records) in sessions {
        let mut builder =
            SessionTurnBuilder::new(session_id.clone(), records, preserve_session_ids);
        let Some(first_turn) = builder.next_turn(&mut parser_tokenizer)? else {
            continue;
        };

        global_trace_start_ms = Some(
            global_trace_start_ms
                .map(|current| current.min(first_turn.assistant_start_ms))
                .unwrap_or(first_turn.assistant_start_ms),
        );

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
    let global_trace_start_ms =
        global_trace_start_ms.ok_or_else(|| anyhow!("no assistant turns were reconstructed"))?;

    let mut writer = MooncakeJsonlWriter::create(output_path, Some(sidecar_path))?;
    let mut hasher = RollingHashIdMapper::new(config.block_size);

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

        let hash_ids = hasher.hash_token_blocks(&ready_turn.tokens);
        let row = MooncakeRow {
            session_id: turn.export_session_id.clone(),
            input_length: ready_turn.tokens.len(),
            output_length: turn.output_length,
            hash_ids,
            timestamp: turn
                .delay_ms
                .is_none()
                .then_some(turn.assistant_start_ms - global_trace_start_ms),
            delay: turn.delay_ms,
        };

        writer.write_row(&row)?;
        writer.write_sidecar(&turn.sidecar)?;

        let next_turn = {
            let state = states
                .get_mut(&session_id)
                .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
            state.overlap_base = Some(OverlapBase {
                previous_text: ready_turn.current_text,
                previous_tokens: ready_turn.tokens,
            });
            state.builder.next_turn(&mut parser_tokenizer)?
        };

        if let Some(next_turn) = next_turn {
            let state = states
                .get_mut(&session_id)
                .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
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
    let writer_stats = writer.finish()?;
    stats.row_count = writer_stats.row_count;
    stats.sidecar_count = writer_stats.sidecar_count;
    Ok(stats)
}

fn push_heap_entry(
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    session_id: &str,
    state: &SessionState,
) {
    if let Some(head) = state.head.as_ref() {
        heap.push(Reverse(HeapEntry {
            assistant_start_ms: head.turn.assistant_start_ms,
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
        write_streamed_mooncake_rows,
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
                        turn_index: 1,
                        input_text: String::new(),
                        output_length: 1,
                        assistant_start_ms: 1,
                        assistant_end_ms: 2,
                        delay_ms: None,
                        sidecar: json!({}),
                    },
                    turn_key: 9,
                    scheduled: true,
                    ready: None,
                }),
                overlap_base: None,
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
                    json!({"type":"assistant","message":{"id":"a-1","content":[{"type":"text","text":"done a"}],"usage":{"output_tokens":3}}}),
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
                    json!({"type":"assistant","message":{"id":"a-2","content":[{"type":"text","text":"done a 2"}],"usage":{"output_tokens":4}}}),
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

        let stats = write_streamed_mooncake_rows(
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
        assert_eq!(rows[0]["session_id"], json!("session-b"));
        assert_eq!(rows[0]["timestamp"], json!(0));
        assert_eq!(rows[1]["session_id"], json!("session-a"));
        assert_eq!(rows[2]["delay"], json!(200));
    }
}

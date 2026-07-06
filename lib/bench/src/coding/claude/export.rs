// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude-specific request-trace export orchestration.
//!
//! Handles session scheduling, parallel tokenization with text-overlap reuse,
//! and global ordering across sessions.

use crate::coding::claude::parser::{
    SessionTurnBuilder, SourceFidelityOracle, TraceRecord, TurnDraft, build_source_fidelity_oracle,
};
use crate::coding::tokenizer::{TokenizerFactory, TokenizerWorker, last_word_overlap_start};
use anyhow::{Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use dynamo_data_gen::{sequence_hashes_for_tokens, write_empty_files};
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
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

#[derive(Debug, Clone, Default)]
pub struct ExportStats {
    pub row_count: usize,
    pub tool_row_count: usize,
    pub sidecar_count: usize,
    pub max_heap_len: usize,
    pub fidelity: FidelityReport,
}

#[derive(Debug, Clone, Default)]
pub struct FidelityReport {
    pub requests_verified: usize,
    pub compactions_verified: usize,
    pub usage_requests_verified: usize,
    pub tools_verified: usize,
    pub child_links_verified: usize,
    pub background_tools: usize,
    pub background_agents: usize,
    pub background_completions_missing: usize,
    pub background_titles_unreplayable: usize,
    pub cache_prefix_blocks_verified: usize,
    pub compaction_prefix_blocks_verified: usize,
    pub post_compaction_prefix_blocks_verified: usize,
    pub unmatched_tool_calls: usize,
    pub unmatched_tool_results: usize,
    pub unresolved_child_sessions: usize,
}

/// Claude-only evidence used to reconstruct tool scheduling after export.
///
/// Live tool events cannot know their future consumer request. Claude's saved
/// session can, so the exporter stores that post-hoc evidence under
/// `tool.claude` without extending the live request-trace tool API.
#[derive(Serialize)]
struct ClaudeToolReplayMetadata {
    source_request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    consumer_request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    child_session_id: Option<String>,
    execution_mode: String,
}

impl FidelityReport {
    pub fn render(&self) -> String {
        let ordinary_requests = self
            .requests_verified
            .saturating_sub(self.compactions_verified);
        format!(
            "Fidelity: requests={0}/{0} compactions={1}/{1} usage={2}/{15} tools={3}/{3} child_links={4}/{4} cache_prefix_blocks={5} compaction_prefix_blocks={6} post_compaction_prefix_blocks={7}\nBackground: tools={8} agents={9} missing_completions={10} title_requests_unreplayable={11}\nLimitations: synthetic_kv_hashes={0} unmatched_tool_calls={12} unmatched_tool_results={13} unresolved_child_sessions={14}",
            self.requests_verified,
            self.compactions_verified,
            self.usage_requests_verified,
            self.tools_verified,
            self.child_links_verified,
            self.cache_prefix_blocks_verified,
            self.compaction_prefix_blocks_verified,
            self.post_compaction_prefix_blocks_verified,
            self.background_tools,
            self.background_agents,
            self.background_completions_missing,
            self.background_titles_unreplayable,
            self.unmatched_tool_calls,
            self.unmatched_tool_results,
            self.unresolved_child_sessions,
            ordinary_requests,
        )
    }
}

struct FidelityVerifier {
    oracle: SourceFidelityOracle,
    seen_requests: BTreeSet<(String, String)>,
    seen_compactions: BTreeSet<(String, String)>,
    tools_by_class: BTreeMap<String, usize>,
    tool_count: usize,
    tool_errors: usize,
    child_links: usize,
    background_tools: usize,
    background_agents: usize,
    usage_requests: usize,
    cache_prefix_blocks_verified: usize,
    compaction_prefix_blocks_verified: usize,
    post_compaction_prefix_blocks_verified: usize,
    next_turn_by_session: FxHashMap<String, usize>,
    previous_hashes_by_session: FxHashMap<String, Vec<u64>>,
    previous_input_length_by_session: FxHashMap<String, usize>,
    previous_was_compaction_by_session: FxHashMap<String, bool>,
    expected_next_cache_read_by_session: FxHashMap<String, usize>,
    export_sessions: BTreeSet<String>,
    causal_references: Vec<(String, usize, String)>,
    child_session_references: Vec<String>,
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

impl FidelityVerifier {
    fn new(oracle: SourceFidelityOracle) -> Self {
        Self {
            oracle,
            seen_requests: BTreeSet::new(),
            seen_compactions: BTreeSet::new(),
            tools_by_class: BTreeMap::new(),
            tool_count: 0,
            tool_errors: 0,
            child_links: 0,
            background_tools: 0,
            background_agents: 0,
            usage_requests: 0,
            cache_prefix_blocks_verified: 0,
            compaction_prefix_blocks_verified: 0,
            post_compaction_prefix_blocks_verified: 0,
            next_turn_by_session: FxHashMap::default(),
            previous_hashes_by_session: FxHashMap::default(),
            previous_input_length_by_session: FxHashMap::default(),
            previous_was_compaction_by_session: FxHashMap::default(),
            expected_next_cache_read_by_session: FxHashMap::default(),
            export_sessions: BTreeSet::new(),
            causal_references: Vec::new(),
            child_session_references: Vec::new(),
        }
    }

    fn observe(
        &mut self,
        turn: &TurnDraft,
        replay_tokens: &[u32],
        input_sequence_hashes: &[u64],
        block_size: usize,
    ) -> Result<()> {
        let key = (turn.session_id.clone(), turn.source_request_id.clone());
        if let Some(compaction) = &turn.compaction {
            let expected = self.oracle.compactions.get(&key).ok_or_else(|| {
                anyhow!(
                    "fidelity verification found unexpected compaction {} in session {}",
                    turn.source_request_id,
                    turn.session_id
                )
            })?;
            if compaction != expected || !self.seen_compactions.insert(key) {
                bail!(
                    "fidelity verification compaction mismatch for session {} sequence {}",
                    turn.session_id,
                    compaction.sequence
                );
            }
            let expected_turn = self
                .next_turn_by_session
                .get(&turn.session_id)
                .copied()
                .unwrap_or_default();
            let expected_start = compaction
                .ended_at_ms
                .saturating_sub(compaction.duration_ms);
            if turn.turn_index != expected_turn
                || turn.request_start_ms != expected_start
                || turn.assistant_end_ms != compaction.ended_at_ms
                || turn.observed_input_length != Some(compaction.pre_tokens)
                || turn.cache_read_input_tokens.is_some()
                || replay_tokens.len() != compaction.pre_tokens
            {
                bail!(
                    "fidelity verification compaction timing/cache mismatch for session {} sequence {}",
                    turn.session_id,
                    compaction.sequence
                );
            }
        } else {
            let expected = self.oracle.requests.get(&key).ok_or_else(|| {
                anyhow!(
                    "fidelity verification found unexpected request {} in session {}",
                    turn.source_request_id,
                    turn.session_id
                )
            })?;
            if !self.seen_requests.insert(key) {
                bail!(
                    "fidelity verification found duplicate request {} in session {}",
                    turn.source_request_id,
                    turn.session_id
                );
            }
            let expected_turn = self
                .next_turn_by_session
                .entry(turn.session_id.clone())
                .or_default();
            if turn.turn_index != *expected_turn {
                bail!(
                    "fidelity verification expected turn {} for session {}, got {}",
                    *expected_turn,
                    turn.session_id,
                    turn.turn_index
                );
            }
            *expected_turn += 1;
            if turn.request_start_ms != expected.request_start_ms
                || turn.assistant_end_ms != expected.assistant_end_ms
            {
                bail!(
                    "fidelity verification timing mismatch for session {} turn {}: expected {}..{}, got {}..{}",
                    turn.session_id,
                    turn.turn_index,
                    expected.request_start_ms,
                    expected.assistant_end_ms,
                    turn.request_start_ms,
                    turn.assistant_end_ms
                );
            }
            if let Some(output_length) = expected.output_length
                && turn.output_length != output_length
            {
                bail!(
                    "fidelity verification output mismatch for session {} turn {}: expected {}, got {}",
                    turn.session_id,
                    turn.turn_index,
                    output_length,
                    turn.output_length
                );
            }
            if let Some(input_length) = expected.input_length {
                self.usage_requests += 1;
                if replay_tokens.len() != input_length
                    || turn.cache_read_input_tokens != expected.cache_read_input_tokens
                    || turn.cache_creation_input_tokens != expected.cache_creation_input_tokens
                {
                    bail!(
                        "fidelity verification input/cache mismatch for session {} turn {}",
                        turn.session_id,
                        turn.turn_index
                    );
                }
            }
        }
        self.export_sessions.insert(turn.export_session_id.clone());
        if turn.request_start_ms > turn.assistant_start_ms
            || turn.assistant_start_ms > turn.assistant_end_ms
        {
            bail!(
                "invalid request timing for session {} turn {}",
                turn.session_id,
                turn.turn_index
            );
        }
        let expected_hashes = replay_tokens.len().div_ceil(block_size);
        if input_sequence_hashes.len() != expected_hashes {
            bail!(
                "fidelity verification expected {} hashes for session {} turn {}, got {}",
                expected_hashes,
                turn.session_id,
                turn.turn_index,
                input_sequence_hashes.len()
            );
        }
        let previous_was_compaction = self
            .previous_was_compaction_by_session
            .get(&turn.session_id)
            .copied()
            .unwrap_or(false);
        let previous_input_length = self
            .previous_input_length_by_session
            .get(&turn.session_id)
            .copied();
        let previous_hashes = self.previous_hashes_by_session.get(&turn.session_id);
        if turn.compaction.is_some() && previous_hashes.is_none() {
            bail!(
                "fidelity verification cannot recover compaction prefix for session {}",
                turn.session_id
            );
        }
        if let (Some(previous_hashes), Some(previous_input_length)) =
            (previous_hashes, previous_input_length)
        {
            let verifiable_blocks = if let Some(compaction) = &turn.compaction {
                previous_input_length.min(compaction.pre_tokens.saturating_sub(1)) / block_size
            } else {
                let cached_blocks = turn.cache_read_input_tokens.unwrap_or(0) / block_size;
                cached_blocks
                    .min(previous_input_length / block_size)
                    .min(previous_hashes.len())
                    .min(input_sequence_hashes.len())
            };
            if previous_hashes[..verifiable_blocks] != input_sequence_hashes[..verifiable_blocks] {
                bail!(
                    "fidelity verification cached prefix mismatch for session {} turn {}",
                    turn.session_id,
                    turn.turn_index
                );
            }
            self.cache_prefix_blocks_verified += verifiable_blocks;
            if turn.compaction.is_some() {
                if verifiable_blocks == 0 {
                    bail!(
                        "fidelity verification found no recoverable compaction prefix blocks for session {}",
                        turn.session_id
                    );
                }
                self.compaction_prefix_blocks_verified += verifiable_blocks;
            } else if previous_was_compaction {
                let cached_tokens = turn.cache_read_input_tokens.unwrap_or(0);
                let cached_blocks = cached_tokens / block_size;
                let cache_creation_tokens = turn.cache_creation_input_tokens.unwrap_or(0);
                if cached_blocks == 0
                    || cache_creation_tokens == 0
                    || cached_tokens > previous_input_length
                    || verifiable_blocks != cached_blocks
                {
                    bail!(
                        "fidelity verification found post-compaction cache miss for session {}",
                        turn.session_id
                    );
                }
                self.post_compaction_prefix_blocks_verified += verifiable_blocks;
                self.expected_next_cache_read_by_session.insert(
                    turn.session_id.clone(),
                    cached_tokens.saturating_add(cache_creation_tokens),
                );
            }
        }
        if turn.compaction.is_none()
            && !previous_was_compaction
            && let Some(expected_cache_read) = self
                .expected_next_cache_read_by_session
                .remove(&turn.session_id)
            && turn.cache_read_input_tokens != Some(expected_cache_read)
        {
            bail!(
                "fidelity verification expected {} post-compaction cache-read tokens for session {}, got {:?}",
                expected_cache_read,
                turn.session_id,
                turn.cache_read_input_tokens
            );
        }
        self.previous_hashes_by_session
            .insert(turn.session_id.clone(), input_sequence_hashes.to_vec());
        self.previous_input_length_by_session
            .insert(turn.session_id.clone(), replay_tokens.len());
        self.previous_was_compaction_by_session
            .insert(turn.session_id.clone(), turn.compaction.is_some());

        for tool in &turn.tools {
            if tool.started_at_ms > tool.ended_at_ms {
                bail!(
                    "invalid tool timing for {} in session {}",
                    tool.tool_call_id,
                    turn.session_id
                );
            }
            self.tool_count += 1;
            *self
                .tools_by_class
                .entry(tool.tool_class.clone())
                .or_insert(0) += 1;
            self.tool_errors += usize::from(tool.is_error);
            self.child_links += usize::from(tool.child_session_id.is_some());
            if let Some(child_session_id) = &tool.child_session_id {
                self.child_session_references.push(child_session_id.clone());
            }
            self.background_tools += usize::from(tool.execution_mode == "background");
            self.background_agents +=
                usize::from(tool.execution_mode == "background" && tool.child_session_id.is_some());
            if !matches!(tool.execution_mode.as_str(), "blocking" | "background") {
                bail!(
                    "fidelity verification found invalid execution mode {} for {}",
                    tool.execution_mode,
                    tool.tool_call_id
                );
            }
            if tool.child_session_id.as_deref() == Some(turn.export_session_id.as_str()) {
                bail!(
                    "fidelity verification found self-referential child session for {}",
                    tool.tool_call_id
                );
            }
            if let Some(consumer_turn_index) = tool.consumer_turn_index {
                if consumer_turn_index <= turn.turn_index {
                    bail!(
                        "fidelity verification found non-forward consumer for {}",
                        tool.tool_call_id
                    );
                }
                self.causal_references.push((
                    turn.session_id.clone(),
                    consumer_turn_index,
                    tool.tool_call_id.clone(),
                ));
            }
        }
        Ok(())
    }

    fn finish(
        self,
        request_rows: usize,
        tool_rows: usize,
        sidecar_rows: usize,
    ) -> Result<FidelityReport> {
        for (session_id, consumer_turn_index, tool_call_id) in &self.causal_references {
            let turn_count = self
                .next_turn_by_session
                .get(session_id)
                .copied()
                .unwrap_or(0);
            if *consumer_turn_index >= turn_count {
                bail!(
                    "fidelity verification found missing consumer turn {} for {}",
                    consumer_turn_index,
                    tool_call_id
                );
            }
        }
        let unresolved_child_sessions = self
            .child_session_references
            .iter()
            .filter(|session_id| !self.export_sessions.contains(*session_id))
            .count();
        let source_request_rows = self.oracle.requests.len() + self.oracle.compactions.len();
        let seen_request_rows = self.seen_requests.len() + self.seen_compactions.len();
        if request_rows != source_request_rows
            || request_rows != seen_request_rows
            || self.seen_compactions.len() != self.oracle.compactions.len()
            || sidecar_rows != request_rows
        {
            bail!(
                "fidelity verification request mismatch: source={} ({} compactions), emitted={}, sidecar={}",
                source_request_rows,
                self.oracle.compactions.len(),
                request_rows,
                sidecar_rows
            );
        }
        if tool_rows != self.oracle.paired_tools
            || self.tool_count != self.oracle.paired_tools
            || self.tool_errors != self.oracle.tool_errors
            || self.tools_by_class != self.oracle.tools_by_class
        {
            bail!(
                "fidelity verification tool mismatch: count={}/{}, errors={}/{}, classes_equal={}",
                self.oracle.paired_tools,
                tool_rows,
                self.oracle.tool_errors,
                self.tool_errors,
                self.tools_by_class == self.oracle.tools_by_class
            );
        }
        if self.child_links != self.oracle.child_links
            || self.background_tools != self.oracle.background_tools
            || self.background_agents != self.oracle.background_agents
        {
            bail!(
                "fidelity verification agent mismatch: child_links={}/{}, background_tools={}/{}, background_agents={}/{}",
                self.oracle.child_links,
                self.child_links,
                self.oracle.background_tools,
                self.background_tools,
                self.oracle.background_agents,
                self.background_agents
            );
        }
        Ok(FidelityReport {
            requests_verified: request_rows,
            compactions_verified: self.seen_compactions.len(),
            usage_requests_verified: self.usage_requests,
            tools_verified: tool_rows,
            child_links_verified: self.child_links,
            background_tools: self.background_tools,
            background_agents: self.background_agents,
            background_completions_missing: self.oracle.background_completions_missing,
            background_titles_unreplayable: self.oracle.background_titles,
            cache_prefix_blocks_verified: self.cache_prefix_blocks_verified,
            compaction_prefix_blocks_verified: self.compaction_prefix_blocks_verified,
            post_compaction_prefix_blocks_verified: self.post_compaction_prefix_blocks_verified,
            unmatched_tool_calls: self.oracle.unmatched_tool_calls,
            unmatched_tool_results: self.oracle.unmatched_tool_results,
            unresolved_child_sessions,
        })
    }
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

    let mut verifier = FidelityVerifier::new(build_source_fidelity_oracle(&sessions)?);
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
        stats.fidelity = verifier.finish(0, 0, 0)?;
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
        verifier.observe(
            &turn,
            &replay_tokens,
            &input_sequence_hashes,
            config.block_size,
        )?;
        let request_id = turn.compaction.as_ref().map_or_else(
            || canonical_request_id(&turn.export_session_id, turn.turn_index),
            |compaction| {
                canonical_compaction_request_id(&turn.export_session_id, compaction.sequence)
            },
        );
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
        let mut request = Map::from_iter([
            ("request_id".to_string(), json!(request_id)),
            ("model".to_string(), json!(turn.model)),
            ("input_tokens".to_string(), json!(replay_tokens.len())),
            ("output_tokens".to_string(), json!(turn.output_length)),
            (
                "request_received_ms".to_string(),
                json!(nonnegative_ms(turn.request_start_ms)),
            ),
            (
                "total_time_ms".to_string(),
                json!((turn.assistant_end_ms - turn.request_start_ms).max(0) as f64),
            ),
            (
                "replay".to_string(),
                json!({
                    "trace_block_size": config.block_size,
                    "input_length": replay_tokens.len(),
                    "input_sequence_hashes": input_sequence_hashes,
                }),
            ),
        ]);
        if let Some(cached_tokens) = turn.cache_read_input_tokens {
            request.insert("cached_tokens".to_string(), json!(cached_tokens));
        }
        if let Some(compaction) = &turn.compaction {
            request.insert(
                "claude".to_string(),
                json!({
                    "compaction": {
                        "trigger": compaction.trigger,
                        "pre_tokens": compaction.pre_tokens,
                        "post_tokens": compaction.post_tokens,
                        "duration_ms": compaction.duration_ms,
                        "cache_fidelity": "recoverable_cache_safe_prefix",
                        "output_fidelity": "tokenized_compact_summary",
                    }
                }),
            );
        }
        let event = json!({
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": nonnegative_ms(turn.assistant_end_ms),
            "event_source": "harness",
            "agent_context": agent_context,
            "request": request,
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
            let claude = ClaudeToolReplayMetadata {
                source_request_id: request_id.clone(),
                consumer_request_id: tool
                    .consumer_turn_index
                    .map(|turn_index| canonical_request_id(&turn.export_session_id, turn_index)),
                child_session_id: tool.child_session_id.clone(),
                execution_mode: tool.execution_mode.clone(),
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
                        "claude": claude,
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
    stats.fidelity = verifier.finish(stats.row_count, stats.tool_row_count, stats.sidecar_count)?;
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

fn canonical_request_id(session_id: &str, turn_index: usize) -> String {
    format!("claude:{session_id}:{turn_index}")
}

fn canonical_compaction_request_id(session_id: &str, sequence: usize) -> String {
    format!("claude:{session_id}:compact:{sequence}")
}

fn materialize_replay_tokens(
    turn: &TurnDraft,
    rendered_tokens: &[u32],
    previous_tokens: Option<&[u32]>,
) -> Vec<u32> {
    let Some(input_length) = turn.observed_input_length else {
        return rendered_tokens.to_vec();
    };

    if turn.compaction.is_some() {
        let shared_length = previous_tokens
            .map(|tokens| tokens.len())
            .unwrap_or_default()
            .min(input_length.saturating_sub(1));
        let mut tokens = Vec::with_capacity(input_length);
        if let Some(previous_tokens) = previous_tokens {
            tokens.extend_from_slice(&previous_tokens[..shared_length]);
        }
        while tokens.len() < input_length {
            tokens.push(synthetic_token(
                &turn.export_session_id,
                turn.turn_index,
                tokens.len(),
                rendered_tokens,
            ));
        }
        return tokens;
    }

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
                        source_request_id: "req-1".to_string(),
                        export_session_id: "session-a".to_string(),
                        export_parent_session_id: None,
                        turn_index: 1,
                        model: "test-model".to_string(),
                        input_text: String::new(),
                        output_length: 1,
                        observed_input_length: None,
                        cache_read_input_tokens: None,
                        cache_creation_input_tokens: None,
                        request_start_ms: 1,
                        assistant_start_ms: 1,
                        assistant_end_ms: 2,
                        delay_ms: None,
                        tools: Vec::new(),
                        sidecar: json!({}),
                        compaction: None,
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
    fn streamed_writer_replays_cache_safe_compaction() {
        use dynamo_data_gen::request_trace::{
            agentic::lower_agentic_mooncake_rows, load::load_request_trace_records,
        };

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
                    json!({"type":"user","message":{"role":"user","content":"first prompt"}}),
                ),
                make_record(
                    "session-a",
                    "assistant",
                    1_100,
                    1,
                    json!({"type":"assistant","requestId":"req-0","message":{"id":"a-0","model":"test-model","content":[{"type":"text","text":"first answer"}],"usage":{"input_tokens":8,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":2}}}),
                ),
                make_record(
                    "session-a",
                    "system",
                    2_000,
                    2,
                    json!({"type":"system","subtype":"compact_boundary","compactMetadata":{"trigger":"manual","preTokens":10,"postTokens":3,"durationMs":500}}),
                ),
                make_record(
                    "session-a",
                    "user",
                    2_000,
                    3,
                    json!({"type":"user","isCompactSummary":true,"message":{"role":"user","content":"compact summary"}}),
                ),
                make_record(
                    "session-a",
                    "assistant",
                    2_100,
                    4,
                    json!({"type":"assistant","requestId":"req-1","message":{"id":"a-1","model":"test-model","content":[{"type":"text","text":"after compact"}],"usage":{"input_tokens":2,"cache_read_input_tokens":4,"cache_creation_input_tokens":6,"output_tokens":2}}}),
                ),
                make_record(
                    "session-a",
                    "user",
                    2_200,
                    5,
                    json!({"type":"user","message":{"role":"user","content":"next prompt"}}),
                ),
                make_record(
                    "session-a",
                    "assistant",
                    2_300,
                    6,
                    json!({"type":"assistant","requestId":"req-2","message":{"id":"a-2","model":"test-model","content":[{"type":"text","text":"next answer"}],"usage":{"input_tokens":2,"cache_read_input_tokens":10,"cache_creation_input_tokens":2,"output_tokens":2}}}),
                ),
            ],
        );

        let no_prefix_error = write_streamed_request_trace_rows(
            &temp.path().join("no-prefix.jsonl"),
            &temp.path().join("no-prefix.sidecar.jsonl"),
            sessions.clone(),
            true,
            StubFactory::default(),
            ExportConfig {
                block_size: 16,
                delta_overlap_words: 50,
                tokenizer_workers: 1,
            },
        )
        .unwrap_err();
        assert!(
            no_prefix_error
                .to_string()
                .contains("no recoverable compaction prefix")
        );

        let mut no_summary_write = sessions.clone();
        let first_post = no_summary_write
            .get_mut("session-a")
            .unwrap()
            .iter_mut()
            .find(|record| record.raw["requestId"] == "req-1")
            .unwrap();
        first_post.raw["message"]["usage"]["cache_creation_input_tokens"] = json!(0);
        let no_summary_write_error = write_streamed_request_trace_rows(
            &temp.path().join("no-summary-write.jsonl"),
            &temp.path().join("no-summary-write.sidecar.jsonl"),
            no_summary_write,
            true,
            StubFactory::default(),
            ExportConfig {
                block_size: 2,
                delta_overlap_words: 50,
                tokenizer_workers: 1,
            },
        )
        .unwrap_err();
        assert!(
            no_summary_write_error
                .to_string()
                .contains("post-compaction cache miss")
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

        let rows = std::fs::read_to_string(&output_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let sidecars = std::fs::read_to_string(&sidecar_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(stats.row_count, 4);
        assert_eq!(stats.sidecar_count, 4);
        assert_eq!(stats.fidelity.compactions_verified, 1);
        assert_eq!(stats.fidelity.compaction_prefix_blocks_verified, 4);
        assert_eq!(stats.fidelity.post_compaction_prefix_blocks_verified, 2);
        assert_eq!(rows.len(), 4);
        assert_eq!(sidecars.len(), 4);
        assert_eq!(
            rows[1]["event"]["request"]["request_id"],
            "claude:session-a:compact:0"
        );
        assert_eq!(rows[1]["event"]["request"]["request_received_ms"], 1_500);
        assert_eq!(rows[1]["event"]["event_time_unix_ms"], 2_000);
        assert_eq!(rows[1]["event"]["request"]["total_time_ms"], 500.0);
        assert!(rows[1]["event"]["request"].get("cached_tokens").is_none());
        assert_eq!(rows[1]["event"]["request"]["replay"]["input_length"], 10);
        assert_eq!(
            rows[1]["event"]["request"]["claude"]["compaction"]["pre_tokens"],
            10
        );
        assert_eq!(
            rows[1]["event"]["request"]["claude"]["compaction"]["post_tokens"],
            3
        );

        let hashes = rows
            .iter()
            .map(|row| {
                row["event"]["request"]["replay"]["input_sequence_hashes"]
                    .as_array()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(hashes[0], &hashes[1][..4]);
        assert_eq!(&hashes[1][..2], &hashes[2][..2]);
        assert_ne!(hashes[1][2], hashes[2][2]);
        assert_eq!(&hashes[2][..5], &hashes[3][..5]);
        assert_ne!(hashes[2][5], hashes[3][5]);

        let loaded = load_request_trace_records(&[output_path]).unwrap();
        assert_eq!(loaded.requests.len(), 4);
        let mut agentic_rows = Vec::new();
        lower_agentic_mooncake_rows(loaded, |_, row| {
            agentic_rows.push(row);
            Ok(())
        })
        .unwrap();
        assert_eq!(agentic_rows.len(), 4);
        assert_eq!(agentic_rows[1].request_id, "claude:session-a:compact:0");
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
                make_record(
                    "session-a",
                    "ai-title",
                    0,
                    3,
                    json!({"type":"ai-title","aiTitle":"Background title"}),
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
        assert_eq!(stats.fidelity.requests_verified, 1);
        assert_eq!(stats.fidelity.tools_verified, 1);
        assert_eq!(stats.fidelity.background_titles_unreplayable, 1);
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
    fn request_trace_preserves_child_identity_and_anonymized_causality() {
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
                    json!({"type":"assistant","requestId":"root-1","message":{"id":"root-1","content":[{"type":"tool_use","id":"agent-call","name":"Agent","input":{"run_in_background":true}}],"usage":{"output_tokens":2}}}),
                ),
                make_record(
                    "root-session",
                    "user",
                    1_150,
                    2,
                    json!({"type":"user","toolUseResult":{"isAsync":true,"agentId":"child-agent","status":"async_launched"},"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"agent-call","content":"launched"}]}}),
                ),
                make_record(
                    "root-session",
                    "user",
                    1_300,
                    4,
                    json!({"type":"user","message":{"role":"user","content":"continue parent work"}}),
                ),
                make_record(
                    "root-session",
                    "assistant",
                    1_400,
                    5,
                    json!({"type":"assistant","requestId":"root-2","message":{"id":"root-2","content":[{"type":"text","text":"working"}],"usage":{"output_tokens":1}}}),
                ),
                make_record(
                    "root-session",
                    "queue-operation",
                    1_800,
                    6,
                    json!({"type":"queue-operation","operation":"enqueue","content":"<tool-use-id>agent-call</tool-use-id><status>completed</status>done"}),
                ),
                make_record(
                    "root-session",
                    "user",
                    1_850,
                    7,
                    json!({"type":"user","message":{"role":"user","content":"child done"}}),
                ),
                make_record(
                    "root-session",
                    "assistant",
                    1_950,
                    8,
                    json!({"type":"assistant","requestId":"root-3","message":{"id":"root-3","content":[{"type":"text","text":"finished"}],"usage":{"output_tokens":1}}}),
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
                    3,
                    json!({"type":"user","isSidechain":true,"agentId":"child-agent","message":{"role":"user","content":"investigate"}}),
                ),
                make_record(
                    "root-session",
                    "assistant",
                    1_700,
                    9,
                    json!({"type":"assistant","isSidechain":true,"agentId":"child-agent","message":{"id":"child-1","content":[{"type":"text","text":"result"}],"usage":{"output_tokens":1}}}),
                ),
            ],
        );

        let config = ExportConfig {
            block_size: 2,
            delta_overlap_words: 50,
            tokenizer_workers: 2,
        };
        let stats = write_streamed_request_trace_rows(
            &output_path,
            &sidecar_path,
            sessions.clone(),
            true,
            StubFactory::default(),
            config,
        )
        .unwrap();

        let anonymous_stats = write_streamed_request_trace_rows(
            &temp.path().join("anonymous.jsonl"),
            &temp.path().join("anonymous.sidecar.jsonl"),
            sessions,
            false,
            StubFactory::default(),
            config,
        )
        .unwrap();
        assert_eq!(anonymous_stats.fidelity.requests_verified, 4);

        assert_eq!(stats.fidelity.requests_verified, 4);
        assert_eq!(stats.fidelity.tools_verified, 1);
        assert_eq!(stats.fidelity.child_links_verified, 1);
        assert_eq!(stats.fidelity.background_tools, 1);
        assert_eq!(stats.fidelity.background_agents, 1);

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
        assert_eq!(agentic_rows.len(), 4);
        let by_id = agentic_rows
            .iter()
            .map(|row| (row.request_id.as_str(), row))
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(
            by_id["claude:root-session:0"].branches,
            vec!["claude:child-agent:0"]
        );
        assert_eq!(
            by_id["claude:child-agent:0"].request_kind.as_deref(),
            Some("background_agent")
        );
        assert_eq!(
            by_id["claude:root-session:1"].wait_for,
            vec!["claude:root-session:0"]
        );
        assert!(
            by_id["claude:root-session:2"]
                .wait_for
                .contains(&"claude:child-agent:0".to_string())
        );
        assert_eq!(by_id["claude:root-session:2"].tool_wait_ms, Some(100.0));
    }
}

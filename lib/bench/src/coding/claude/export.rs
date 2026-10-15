// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude-specific request-trace export orchestration.
//!
//! Handles session scheduling, parallel tokenization with text-overlap reuse,
//! and global ordering across sessions.

use crate::coding::claude::parser::{
    SessionTurnBuilder, SourceFidelityOracle, TraceIndex, TraceRecord, TurnDraft,
    request_start_bound_ms,
};
use crate::coding::replay::{
    AgentContext, HARNESS_EVENT_SOURCE, PrefixPool, REQUEST_TRACE_SCHEMA, ReplayBase, ReplayFields,
    RequestEndEvent, RequestFields, ToolEvent, ToolFields, TraceLine, synthetic_stream_seed,
    synthetic_token, usage_shaped_tokens,
};
use crate::coding::tokenizer::{
    LazyTokenizer, TokenizerFactory, TokenizerWorker, last_word_overlap_start,
};
use anyhow::{Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::json;
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
    pub pooled_prefix_blocks: usize,
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
            "Fidelity: requests={0}/{0} compactions={1}/{1} usage={2}/{15} tools={3}/{3} child_links={4}/{4} cache_prefix_blocks={5} compaction_prefix_blocks={6} post_compaction_prefix_blocks={7} pooled_prefix_blocks={16}\nBackground: tools={8} agents={9} missing_completions={10} title_requests_unreplayable={11}\nLimitations: synthetic_kv_hashes={0} unmatched_tool_calls={12} unmatched_tool_results={13} unresolved_child_sessions={14}",
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
            self.pooled_prefix_blocks,
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
    sidecar_line: String,
    turn_key: u64,
    scheduled: bool,
    ready: Option<ReadyTurn>,
}

/// A turn waiting for the merge, with its sidecar row already serialized.
#[derive(Debug)]
struct QueuedTurn {
    turn: TurnDraft,
    sidecar_line: String,
}

impl QueuedTurn {
    fn new(mut turn: TurnDraft) -> Result<Self> {
        let sidecar_line = serde_json::to_string(&turn.sidecar)?;
        turn.sidecar = serde_json::Value::Null;
        // Usage-shaped turns replay synthetic hashes, so their transcript is never tokenized.
        if turn.observed_input_length.is_some() {
            turn.input_text = String::new();
        }
        Ok(Self { turn, sidecar_line })
    }
}

/// Where a session's remaining turns come from.
#[derive(Debug)]
enum SessionTurns {
    /// Built when the session opened, so its rows are already freed.
    Built(VecDeque<QueuedTurn>),
    /// Built one at a time, because each transcript-shaped turn carries the transcript so far.
    Lazy(Box<SessionTurnBuilder>),
}

impl SessionTurns {
    /// Builds every turn now unless some turn needs its transcript tokenized.
    fn open(mut builder: SessionTurnBuilder, tokenizer: &mut impl TokenizerWorker) -> Result<Self> {
        if !builder.requests_are_usage_shaped() {
            return Ok(Self::Lazy(Box::new(builder)));
        }
        let mut turns = VecDeque::new();
        while let Some(turn) = builder.next_turn(tokenizer)? {
            turns.push_back(QueuedTurn::new(turn)?);
        }
        Ok(Self::Built(turns))
    }

    fn next(&mut self, tokenizer: &mut impl TokenizerWorker) -> Result<Option<QueuedTurn>> {
        match self {
            Self::Built(turns) => Ok(turns.pop_front()),
            Self::Lazy(builder) => builder
                .next_turn(tokenizer)?
                .map(QueuedTurn::new)
                .transpose(),
        }
    }
}

#[derive(Debug)]
struct SessionState {
    turns: SessionTurns,
    head: Option<HeadTurn>,
    overlap_base: Option<OverlapBase>,
    replay_base: Option<ReplayBase>,
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
            background_titles_unreplayable: self.oracle.background_titles.len(),
            cache_prefix_blocks_verified: self.cache_prefix_blocks_verified,
            compaction_prefix_blocks_verified: self.compaction_prefix_blocks_verified,
            post_compaction_prefix_blocks_verified: self.post_compaction_prefix_blocks_verified,
            pooled_prefix_blocks: 0,
            unmatched_tool_calls: self.oracle.unmatched_tool_calls,
            unmatched_tool_results: self.oracle.unmatched_tool_results,
            unresolved_child_sessions,
        })
    }
}

/// Claude sessions the export opens one at a time.
pub trait SessionSource {
    /// Every session, with a lower bound on when its requests can start when one is known.
    fn session_bounds(&self) -> Vec<(String, Option<i64>)>;

    /// The session's rows in source order.
    fn take_session(&mut self, trace_id: &str) -> Result<Vec<TraceRecord>>;
}

impl SessionSource for TraceIndex {
    fn session_bounds(&self) -> Vec<(String, Option<i64>)> {
        self.sessions()
            .map(|(trace_id, bound)| (trace_id.to_string(), bound))
            .collect()
    }

    fn take_session(&mut self, trace_id: &str) -> Result<Vec<TraceRecord>> {
        self.load_session(trace_id)
    }
}

impl SessionSource for FxHashMap<String, Vec<TraceRecord>> {
    fn session_bounds(&self) -> Vec<(String, Option<i64>)> {
        self.iter()
            .map(|(trace_id, records)| {
                let bound = records.iter().filter_map(request_start_bound_ms).min();
                (trace_id.clone(), bound)
            })
            .collect()
    }

    fn take_session(&mut self, trace_id: &str) -> Result<Vec<TraceRecord>> {
        self.remove(trace_id)
            .ok_or_else(|| anyhow!("unknown Claude session {trace_id}"))
    }
}

/// Writes request-trace and sidecar rows in global request-start order.
///
/// A session is parsed only once the merge could reach its earliest request and is dropped after
/// its last turn, so memory follows the sessions active at one time rather than the corpus.
pub fn write_streamed_request_trace_rows<F, S>(
    output_path: &Path,
    sidecar_path: &Path,
    mut sessions: S,
    preserve_session_ids: bool,
    tokenizer_factory: F,
    config: ExportConfig,
) -> Result<ExportStats>
where
    F: TokenizerFactory,
    S: SessionSource,
{
    if config.block_size == 0 {
        bail!("block_size must be greater than 0");
    }
    if config.tokenizer_workers == 0 {
        bail!("tokenizer_workers must be greater than 0");
    }

    let mut verifier = FidelityVerifier::new(SourceFidelityOracle::default());
    let mut parser_tokenizer = LazyTokenizer::new(tokenizer_factory.clone());
    let mut states = FxHashMap::default();
    let mut heap = BinaryHeap::new();
    let mut unscheduled_sessions = VecDeque::new();
    let mut stats = ExportStats::default();
    let mut unopened = sessions.session_bounds();
    unopened.sort_by(|left, right| {
        (left.1.unwrap_or(i64::MIN), &left.0).cmp(&(right.1.unwrap_or(i64::MIN), &right.0))
    });
    let mut unopened = unopened.into_iter().peekable();

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

    let mut trace_start_ms = None;
    let mut prefix_pool = PrefixPool::default();
    let mut pooled_prefix_blocks = 0_usize;
    let mut inflight_jobs = 0_usize;
    loop {
        // Open every session whose requests could precede the earliest open head. Sessions open
        // in bound order, so each remaining session starts after that head.
        while let Some((_, bound)) = unopened.peek() {
            let earliest_head_ms = heap
                .peek()
                .map(|Reverse(entry): &Reverse<HeapEntry>| entry.request_start_ms);
            if let (Some(bound), Some(earliest_head_ms)) = (bound, earliest_head_ms)
                && *bound > earliest_head_ms
            {
                break;
            }
            let (session_id, _) = unopened.next().expect("peeked session");
            let records = sessions.take_session(&session_id)?;
            verifier.oracle.add_session(&session_id, &records)?;
            let builder =
                SessionTurnBuilder::new(session_id.clone(), records, preserve_session_ids);
            let mut turns = SessionTurns::open(builder, &mut parser_tokenizer)?;
            let Some(first_turn) = turns.next(&mut parser_tokenizer)? else {
                continue;
            };
            let state = SessionState {
                turns,
                head: Some(head_turn(first_turn, 0)),
                overlap_base: None,
                replay_base: None,
                next_turn_key: 1,
            };
            push_heap_entry(&mut heap, &session_id, &state);
            states.insert(session_id.clone(), state);
            unscheduled_sessions.push_back(session_id);
            stats.max_heap_len = stats.max_heap_len.max(heap.len());
        }
        if heap.is_empty() {
            break;
        }
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
        let (turn, sidecar_line, ready_turn) = {
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
            (head.turn, head.sidecar_line, ready_turn)
        };

        let trace_start_ms = *trace_start_ms.get_or_insert(turn.request_start_ms);
        let next_turn = {
            let state = states
                .get_mut(&session_id)
                .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
            state.turns.next(&mut parser_tokenizer)?
        };
        let replay = {
            let state = states
                .get(&session_id)
                .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
            let cached_tokens = turn.cache_read_input_tokens.unwrap_or(0);
            let source = match state.replay_base.as_ref() {
                Some(base) => Some(base),
                None if turn.compaction.is_none()
                    && turn.observed_input_length.is_some()
                    && cached_tokens > 0 =>
                {
                    Some(prefix_pool.prefix(
                        &turn.prefix_pool_key,
                        cached_tokens,
                        config.block_size,
                    )?)
                }
                None => None,
            };
            let (tokens, shared_tokens) = materialize_replay_tokens(
                &turn,
                &ready_turn.tokens,
                source.map(|base| base.tokens.as_slice()),
            );
            if state.replay_base.is_none() {
                pooled_prefix_blocks += shared_tokens / config.block_size;
            }
            ReplayBase::derive(source, shared_tokens, tokens, config.block_size)?
        };
        verifier.observe(&turn, &replay.tokens, &replay.hashes, config.block_size)?;
        let request_id = turn.compaction.as_ref().map_or_else(
            || canonical_request_id(&turn.export_session_id, turn.turn_index),
            |compaction| {
                canonical_compaction_request_id(&turn.export_session_id, compaction.sequence)
            },
        );
        let agent_context = AgentContext {
            session_id: &turn.export_session_id,
            parent_session_id: turn.export_parent_session_id.as_deref(),
        };
        let claude = turn.compaction.as_ref().map(|compaction| {
            json!({
                "compaction": {
                    "trigger": compaction.trigger,
                    "pre_tokens": compaction.pre_tokens,
                    "post_tokens": compaction.post_tokens,
                    "duration_ms": compaction.duration_ms,
                    "cache_fidelity": "recoverable_cache_safe_prefix",
                    "output_fidelity": "tokenized_compact_summary",
                }
            })
        });
        let row = TraceLine {
            timestamp: nonnegative_ms(turn.assistant_end_ms - trace_start_ms),
            event: RequestEndEvent {
                schema: REQUEST_TRACE_SCHEMA,
                event_type: "request_end",
                event_time_unix_ms: nonnegative_ms(turn.assistant_end_ms),
                event_source: HARNESS_EVENT_SOURCE,
                agent_context: &agent_context,
                request: RequestFields {
                    request_id: &request_id,
                    model: &turn.model,
                    input_tokens: replay.tokens.len(),
                    output_tokens: turn.output_length,
                    request_received_ms: nonnegative_ms(turn.request_start_ms),
                    total_time_ms: (turn.assistant_end_ms - turn.request_start_ms).max(0) as f64,
                    replay: ReplayFields {
                        trace_block_size: config.block_size,
                        input_length: replay.tokens.len(),
                        input_sequence_hashes: &replay.hashes,
                        dependencies: &[],
                    },
                    cached_tokens: turn.cache_read_input_tokens,
                    claude,
                },
            },
        };
        write_json_line(&mut output, &row)?;
        for tool in &turn.tools {
            let claude = ClaudeToolReplayMetadata {
                source_request_id: request_id.clone(),
                consumer_request_id: tool
                    .consumer_turn_index
                    .map(|turn_index| canonical_request_id(&turn.export_session_id, turn_index)),
                child_session_id: tool.child_session_id.clone(),
                execution_mode: tool.execution_mode.clone(),
            };
            let tool_row = TraceLine {
                timestamp: nonnegative_ms(tool.ended_at_ms - trace_start_ms),
                event: ToolEvent {
                    schema: REQUEST_TRACE_SCHEMA,
                    event_type: if tool.is_error {
                        "tool_error"
                    } else {
                        "tool_end"
                    },
                    event_time_unix_ms: nonnegative_ms(tool.ended_at_ms),
                    event_source: HARNESS_EVENT_SOURCE,
                    agent_context: &agent_context,
                    tool: ToolFields {
                        tool_call_id: &tool.tool_call_id,
                        tool_class: &tool.tool_class,
                        claude: Some(claude),
                        started_at_unix_ms: nonnegative_ms(tool.started_at_ms),
                        ended_at_unix_ms: nonnegative_ms(tool.ended_at_ms),
                        duration_ms: (tool.ended_at_ms - tool.started_at_ms).max(0) as f64,
                        status: if tool.is_error { "error" } else { "succeeded" },
                        output_bytes: Some(tool.output_bytes),
                        error_type: tool.is_error.then_some("claude_tool_error"),
                    },
                },
            };
            write_json_line(&mut output, &tool_row)?;
            stats.tool_row_count += 1;
        }
        sidecar.write_all(sidecar_line.as_bytes())?;
        sidecar.write_all(b"\n")?;
        stats.row_count += 1;
        stats.sidecar_count += 1;

        let state = states
            .get_mut(&session_id)
            .ok_or_else(|| anyhow!("missing session state for {}", session_id))?;
        state.overlap_base = turn.observed_input_length.is_none().then_some(OverlapBase {
            previous_text: ready_turn.current_text,
            previous_tokens: ready_turn.tokens,
        });
        state.replay_base = Some(replay);

        if let Some(next_turn) = next_turn {
            let turn_key = state.next_turn_key;
            state.next_turn_key += 1;
            state.head = Some(head_turn(next_turn, turn_key));
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
    stats.fidelity.pooled_prefix_blocks = pooled_prefix_blocks;
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

/// Returns replay tokens and how many of them were copied from `prefix_source`.
///
/// Usage-shaped turns copy their cached prefix from the session's previous request, or for a
/// session's first request from the shared prefix pool, and fill the rest with synthetic tokens.
fn materialize_replay_tokens(
    turn: &TurnDraft,
    rendered_tokens: &[u32],
    prefix_source: Option<&[u32]>,
) -> (Vec<u32>, usize) {
    let Some(input_length) = turn.observed_input_length else {
        return (rendered_tokens.to_vec(), 0);
    };
    let seed = synthetic_stream_seed(&turn.export_session_id);

    if turn.compaction.is_some() {
        let shared_length = prefix_source
            .map(<[u32]>::len)
            .unwrap_or_default()
            .min(input_length.saturating_sub(1));
        let mut tokens = Vec::with_capacity(input_length);
        if let Some(prefix_source) = prefix_source {
            tokens.extend_from_slice(&prefix_source[..shared_length]);
        }
        let start = tokens.len();
        tokens.extend(
            (start..input_length).map(|position| synthetic_token(seed, turn.turn_index, position)),
        );
        return (tokens, shared_length);
    }

    usage_shaped_tokens(
        seed,
        turn.turn_index,
        input_length,
        turn.cache_read_input_tokens.unwrap_or(0),
        prefix_source,
    )
}

/// Usage-shaped turns replay synthetic hashes, so only transcript-shaped turns are tokenized.
fn head_turn(queued: QueuedTurn, turn_key: u64) -> HeadTurn {
    let ready = queued
        .turn
        .observed_input_length
        .is_some()
        .then(|| ReadyTurn {
            current_text: String::new(),
            tokens: Vec::new(),
        });
    HeadTurn {
        turn: queued.turn,
        sidecar_line: queued.sidecar_line,
        turn_key,
        scheduled: false,
        ready,
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
                let mut tokenizer = LazyTokenizer::new(factory);
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
        ExportConfig, HeadTurn, ReadyTurn, SessionState, SessionTurns, TurnDraft,
        apply_tokenize_response, write_streamed_request_trace_rows,
    };
    use crate::coding::claude::parser::TraceRecord;
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
                turns: SessionTurns::Built(std::collections::VecDeque::new()),
                head: Some(HeadTurn {
                    sidecar_line: String::new(),
                    turn: TurnDraft {
                        session_id: "session-a".to_string(),
                        source_request_id: "req-1".to_string(),
                        export_session_id: "session-a".to_string(),
                        export_parent_session_id: None,
                        turn_index: 1,
                        model: "test-model".to_string(),
                        input_text: String::new(),
                        prefix_pool_key: String::new(),
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
    fn transcript_shaped_sessions_tokenize_during_the_merge() {
        let temp = TempDir::new().unwrap();
        let output_path = temp.path().join("trace.jsonl");
        let sidecar_path = temp.path().join("trace.sidecar.jsonl");
        let mut sessions = FxHashMap::default();
        sessions.insert(
            "session-u".to_string(),
            vec![
                make_record(
                    "session-u",
                    "user",
                    1_000,
                    0,
                    json!({"type":"user","message":{"role":"user","content":"usage prompt"}}),
                ),
                make_record(
                    "session-u",
                    "assistant",
                    1_100,
                    1,
                    json!({"type":"assistant","message":{"id":"u-1","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":4,"output_tokens":1}}}),
                ),
            ],
        );
        sessions.insert(
            "session-t".to_string(),
            vec![
                make_record(
                    "session-t",
                    "user",
                    1_050,
                    2,
                    json!({"type":"user","message":{"role":"user","content":"first t prompt"}}),
                ),
                make_record(
                    "session-t",
                    "assistant",
                    1_200,
                    3,
                    json!({"type":"assistant","message":{"id":"t-1","content":[{"type":"text","text":"answer"}]}}),
                ),
                make_record(
                    "session-t",
                    "user",
                    1_300,
                    4,
                    json!({"type":"user","message":{"role":"user","content":"second t"}}),
                ),
                make_record(
                    "session-t",
                    "assistant",
                    1_400,
                    5,
                    json!({"type":"assistant","message":{"id":"t-2","content":[{"type":"text","text":"again"}]}}),
                ),
            ],
        );
        let factory = StubFactory::default();
        let calls = factory.calls.clone();

        let stats = write_streamed_request_trace_rows(
            &output_path,
            &sidecar_path,
            sessions,
            true,
            factory,
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
        assert_eq!(stats.row_count, 3);
        let order = rows
            .iter()
            .map(|row| row["event"]["request"]["request_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            [
                "claude:session-u:0",
                "claude:session-t:0",
                "claude:session-t:1"
            ]
        );
        // The transcript-shaped turn replays its tokenized transcript: "[user] first t prompt".
        assert_eq!(rows[1]["event"]["request"]["replay"]["input_length"], 4);
        assert!(
            rows[2]["event"]["request"]["replay"]["input_length"]
                .as_u64()
                .unwrap()
                > 4
        );
        // The usage-shaped session never reaches the tokenizer.
        assert!(
            calls
                .lock()
                .unwrap()
                .iter()
                .all(|text| !text.contains("usage prompt"))
        );
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
        assert!(
            by_id["claude:child-agent:0"]
                .dependencies
                .iter()
                .any(|edge| {
                    edge.request_id == "claude:root-session:0"
                        && edge.relation == dynamo_data_gen::AgenticDependencyRelation::Spawn
                })
        );
        assert!(
            by_id["claude:root-session:1"]
                .dependencies
                .iter()
                .any(|edge| {
                    edge.request_id == "claude:root-session:0"
                        && edge.relation == dynamo_data_gen::AgenticDependencyRelation::Sequence
                })
        );
        assert!(
            !by_id["claude:root-session:2"]
                .dependencies
                .iter()
                .any(|edge| {
                    edge.request_id == "claude:child-agent:0"
                        && edge.relation == dynamo_data_gen::AgenticDependencyRelation::Join
                })
        );
    }
}

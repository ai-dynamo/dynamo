// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Usage-shaped replay tokens and request-trace rows shared by coding-agent exporters.

use anyhow::{Result, bail};
use dynamo_data_gen::{
    AgenticDependencyRelation, AgenticDependencyTrigger, extend_sequence_hashes,
};
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::Value;

pub const REQUEST_TRACE_SCHEMA: &str = "dynamo.request.trace.v1";
pub const HARNESS_EVENT_SOURCE: &str = "harness";

/// FNV-1a seed of one synthetic token stream.
pub fn synthetic_stream_seed(stream_id: &str) -> u32 {
    let mut hash = 0x811c_9dc5_u32;
    for byte in stream_id.bytes() {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    hash
}

/// Deterministic stand-in for a token whose identity the source does not record.
pub fn synthetic_token(stream_seed: u32, turn_index: usize, position: usize) -> u32 {
    let hash = (stream_seed ^ turn_index as u32).wrapping_mul(0x0100_0193);
    (hash ^ position as u32).wrapping_mul(0x0100_0193)
}

/// Builds `input_length` replay tokens whose first `cached_length` tokens are a cached prefix.
///
/// The prefix is copied from `prefix_source` as far as it reaches; the rest of the cached prefix
/// and the uncached suffix are synthetic tokens of `turn_index - 1` and `turn_index`. Returns the
/// tokens and how many were copied.
pub fn usage_shaped_tokens(
    stream_seed: u32,
    turn_index: usize,
    input_length: usize,
    cached_length: usize,
    prefix_source: Option<&[u32]>,
) -> (Vec<u32>, usize) {
    let cached_length = cached_length.min(input_length);
    let mut tokens = Vec::with_capacity(input_length);
    if let Some(prefix_source) = prefix_source {
        tokens.extend_from_slice(&prefix_source[..cached_length.min(prefix_source.len())]);
    }
    let shared_length = tokens.len();
    tokens.extend(
        (shared_length..cached_length)
            .map(|position| synthetic_token(stream_seed, turn_index.saturating_sub(1), position)),
    );
    let start = tokens.len();
    tokens.extend(
        (start..input_length).map(|position| synthetic_token(stream_seed, turn_index, position)),
    );
    (tokens, shared_length)
}

/// Key of the shared prefix stream for sessions of one harness, model, and working directory.
pub fn prefix_pool_key(harness: &str, model: &str, cwd: &str) -> String {
    format!("{harness}\u{1f}{model}\u{1f}{cwd}")
}

/// Replay tokens of one request and their sequence hashes, including a trailing partial block.
#[derive(Debug, Default, Clone)]
pub struct ReplayBase {
    pub tokens: Vec<u32>,
    pub hashes: Vec<u64>,
}

impl ReplayBase {
    /// Hashes `tokens`, whose first `shared_tokens` tokens were copied from `source`.
    ///
    /// Sequence hashes chain from the first block, so every full block inside the copied prefix
    /// keeps the source's hash and only the remaining blocks are hashed.
    pub fn derive(
        source: Option<&ReplayBase>,
        shared_tokens: usize,
        tokens: Vec<u32>,
        block_size: usize,
    ) -> Result<Self> {
        let reused = source.map_or(&[][..], |source| {
            let blocks = (shared_tokens / block_size).min(source.tokens.len() / block_size);
            &source.hashes[..blocks]
        });
        // Reused hashes are trusted rather than recomputed, so check the copied boundary block.
        let boundary = reused.len().saturating_sub(1) * block_size..reused.len() * block_size;
        if let Some(source) = source
            && source.tokens.get(boundary.clone()) != tokens.get(boundary)
        {
            bail!(
                "replay prefix diverges from the {} reused hash blocks",
                reused.len()
            );
        }
        debug_assert_eq!(
            source.map(|source| &source.tokens[..reused.len() * block_size]),
            source.map(|_| &tokens[..reused.len() * block_size])
        );
        let hashes = extend_sequence_hashes(reused, &tokens, block_size)?;
        Ok(Self { tokens, hashes })
    }
}

/// Synthetic prefixes shared by sessions of one harness, model, and working directory.
///
/// Usage reports how much of a session's first request was already cached, but not which earlier
/// request wrote it. The pool attributes that prefix to context such sessions share, such as the
/// system prompt and tool definitions, so replay can reuse it across sessions.
#[derive(Debug, Default)]
pub struct PrefixPool {
    streams: FxHashMap<String, ReplayBase>,
}

impl PrefixPool {
    /// Returns the shared stream for `key`, extended to at least `length` tokens.
    pub fn prefix(&mut self, key: &str, length: usize, block_size: usize) -> Result<&ReplayBase> {
        if !self.streams.contains_key(key) {
            self.streams.insert(key.to_string(), ReplayBase::default());
        }
        let stream = self.streams.get_mut(key).expect("stream was inserted");
        if stream.tokens.len() < length {
            let seed = synthetic_stream_seed(key);
            let start = stream.tokens.len();
            stream
                .tokens
                .extend((start..length).map(|position| synthetic_token(seed, 0, position)));
            stream.hashes.truncate(start / block_size);
            stream.hashes = extend_sequence_hashes(&stream.hashes, &stream.tokens, block_size)?;
        }
        Ok(stream)
    }
}

/// Exporter-only edge from a request to an earlier request it could not start before.
///
/// Live request traces cannot know these edges; an exporter that reads complete sessions can.
/// Lowering adds them to the per-session sequence edges it always derives.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayDependency {
    pub request_id: String,
    pub relation: AgenticDependencyRelation,
    pub trigger: AgenticDependencyTrigger,
}

#[derive(Serialize)]
pub struct TraceLine<E> {
    pub timestamp: u64,
    pub event: E,
}

#[derive(Serialize)]
pub struct AgentContext<'a> {
    pub session_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<&'a str>,
}

#[derive(Serialize)]
pub struct RequestEndEvent<'a> {
    pub schema: &'static str,
    pub event_type: &'static str,
    pub event_time_unix_ms: u64,
    pub event_source: &'static str,
    pub agent_context: &'a AgentContext<'a>,
    pub request: RequestFields<'a>,
}

#[derive(Serialize)]
pub struct RequestFields<'a> {
    pub request_id: &'a str,
    pub model: &'a str,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub request_received_ms: u64,
    pub total_time_ms: f64,
    pub replay: ReplayFields<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claude: Option<Value>,
}

#[derive(Serialize)]
pub struct ReplayFields<'a> {
    pub trace_block_size: usize,
    pub input_length: usize,
    pub input_sequence_hashes: &'a [u64],
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub dependencies: &'a [ReplayDependency],
}

#[derive(Serialize)]
pub struct ToolEvent<'a, M> {
    pub schema: &'static str,
    pub event_type: &'static str,
    pub event_time_unix_ms: u64,
    pub event_source: &'static str,
    pub agent_context: &'a AgentContext<'a>,
    pub tool: ToolFields<'a, M>,
}

#[derive(Serialize)]
pub struct ToolFields<'a, M> {
    pub tool_call_id: &'a str,
    pub tool_class: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claude: Option<M>,
    pub started_at_unix_ms: u64,
    pub ended_at_unix_ms: u64,
    pub duration_ms: f64,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_bytes: Option<usize>,
    pub error_type: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use super::{PrefixPool, ReplayBase};
    use dynamo_data_gen::sequence_hashes_for_tokens;

    #[test]
    fn pooled_prefixes_are_stable_as_the_stream_grows() {
        let mut pool = PrefixPool::default();
        let short = pool.prefix("claude|model|/repo", 100, 16).unwrap().tokens[..100].to_vec();
        let long = pool.prefix("claude|model|/repo", 300, 16).unwrap();
        assert_eq!(long.tokens[..100], short);
        assert_eq!(
            long.hashes,
            sequence_hashes_for_tokens(&long.tokens, 16).unwrap()
        );
        let other = pool.prefix("claude|model|/other", 100, 16).unwrap();
        assert_ne!(other.tokens[..100], short);
    }

    #[test]
    fn derived_hashes_match_full_hashing() {
        let source = ReplayBase {
            tokens: (0..70).collect(),
            hashes: sequence_hashes_for_tokens(&(0..70).collect::<Vec<_>>(), 16).unwrap(),
        };
        let mut tokens = source.tokens[..50].to_vec();
        tokens.extend(1_000..1_040);
        let derived = ReplayBase::derive(Some(&source), 50, tokens.clone(), 16).unwrap();
        assert_eq!(
            derived.hashes,
            sequence_hashes_for_tokens(&tokens, 16).unwrap()
        );
        // Claiming more shared tokens than were copied must not reuse a mismatched block hash.
        assert!(ReplayBase::derive(Some(&source), 64, tokens, 16).is_err());
    }
}

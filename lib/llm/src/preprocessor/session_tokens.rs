// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-session incremental prompt tokenization (`DYN_TOKENIZER_SESSION_CACHE`, default off).
//!
//! Agent clients resend the whole conversation on every turn, so a long session pays a full
//! encode of a prompt that grew by a few thousand tokens. The process-wide L1 prefix cache is
//! shared by every session, so once many long sessions interleave it rarely still holds a
//! session's previous turn.
//!
//! Per session id this keeps the previous prompt, its token ids, and checkpoints
//! `(byte offset, token count)` right after every boundary special token. The next prompt of
//! the session reuses the ids up to the deepest checkpoint inside the byte-identical common
//! prefix and encodes only the rest, one boundary-delimited segment at a time.
//!
//! Splitting right after a boundary token is the L1 cache's invariant:
//! `encode(a) ++ encode(b) == encode(a ++ b)` when `a` ends with an atomic special token.
//! Construction refuses tokenizers that do not allow prefix caching, boundary tokens that can
//! overlap any added token, and tokenizers that fail a split check on short probe strings.
//! `verify` mode also runs the normal path on every request and returns its ids.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use aho_corasick::AhoCorasick;
use parking_lot::Mutex;

use crate::tokenizers::traits::Tokenizer;

/// `off` (default), `on`, or `verify`.
pub const ENV_TOKENIZER_SESSION_CACHE: &str = "DYN_TOKENIZER_SESSION_CACHE";
/// Sessions kept, least recently used evicted first. Default 4096.
pub const ENV_TOKENIZER_SESSION_CACHE_MAX_SESSIONS: &str =
    "DYN_TOKENIZER_SESSION_CACHE_MAX_SESSIONS";
/// Byte budget for all kept sessions (prompt text, token ids, checkpoints). Default 2 GiB.
pub const ENV_TOKENIZER_SESSION_CACHE_BYTES: &str = "DYN_TOKENIZER_SESSION_CACHE_BYTES";

const DEFAULT_MAX_SESSIONS: usize = 4096;
const DEFAULT_MAX_BYTES: usize = 2 << 30;
const STATS_EVERY: u64 = 1000;

/// Text encoded after each boundary token at construction to check split invariance.
const SPLIT_PROBES: [&str; 4] = ["hello world", " lead", "\nline", "x1 \u{e9}"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionTokenizeMode {
    Off,
    On,
    Verify,
}

impl SessionTokenizeMode {
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("on" | "1" | "true") => Self::On,
            Some("verify") => Self::Verify,
            _ => Self::Off,
        }
    }

    pub fn from_env() -> Self {
        Self::parse(std::env::var(ENV_TOKENIZER_SESSION_CACHE).ok().as_deref())
    }
}

tokio::task_local! {
    /// Session key of the chat request being preprocessed.
    pub static SESSION_KEY: Arc<str>;
}

/// The session key of the current task, if any. Read it before `spawn_blocking`.
pub fn current_session_key() -> Option<Arc<str>> {
    SESSION_KEY.try_with(|key| key.clone()).ok()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionEncodeStats {
    pub reused_tokens: usize,
    pub new_tokens: usize,
}

struct SessionState {
    /// The last prompt; only `text[..checkpoints.last().0]` is reused.
    text: String,
    /// Token ids of `text[..checkpoints.last().0]`.
    tokens: Vec<u32>,
    /// Ascending `(byte offset right after a boundary token, ids of text[..offset])`.
    checkpoints: Vec<(usize, usize)>,
}

impl SessionState {
    fn bytes(&self) -> usize {
        self.text.capacity()
            + self.tokens.capacity() * std::mem::size_of::<u32>()
            + self.checkpoints.capacity() * std::mem::size_of::<(usize, usize)>()
    }
}

#[derive(Default)]
struct SessionMap {
    sessions: HashMap<Arc<str>, (Arc<SessionState>, u64)>,
    bytes: usize,
    clock: u64,
}

#[derive(Default)]
struct Counters {
    requests: AtomicU64,
    reused_tokens: AtomicU64,
    new_tokens: AtomicU64,
    mismatches: AtomicU64,
    evictions: AtomicU64,
}

pub struct SessionTokenCache {
    tokenizer: Arc<dyn Tokenizer>,
    matcher: AhoCorasick,
    mode: SessionTokenizeMode,
    max_sessions: usize,
    max_bytes: usize,
    map: Mutex<SessionMap>,
    counters: Counters,
}

/// Whether occurrences of `a` and `b` in one text can share bytes.
fn can_overlap(a: &str, b: &str) -> bool {
    if a.contains(b) || b.contains(a) {
        return true;
    }
    let (a, b) = (a.as_bytes(), b.as_bytes());
    (1..a.len().min(b.len())).any(|n| a[a.len() - n..] == b[..n] || b[b.len() - n..] == a[..n])
}

fn can_overlap_itself(token: &str) -> bool {
    let b = token.as_bytes();
    (1..b.len()).any(|n| b[b.len() - n..] == b[..n])
}

/// First pair of tokens whose occurrences could make a boundary ambiguous. `others` are the
/// remaining added tokens, which the tokenizer could match in place of a boundary token.
fn first_ambiguous_pair<'a>(
    boundaries: &'a [String],
    others: &'a [String],
) -> Option<(&'a str, &'a str)> {
    for (i, token) in boundaries.iter().enumerate() {
        if can_overlap_itself(token) {
            return Some((token, token));
        }
        for other in boundaries[i + 1..].iter().chain(others) {
            if can_overlap(token, other) {
                return Some((token, other));
            }
        }
    }
    None
}

fn check_split_invariance(tokenizer: &dyn Tokenizer, boundaries: &[String]) -> anyhow::Result<()> {
    for token in boundaries {
        let head = format!("a{token}");
        let head_ids = tokenizer.encode(&head)?.token_ids().to_vec();
        for probe in SPLIT_PROBES {
            let mut split = head_ids.clone();
            split.extend_from_slice(tokenizer.encode(probe)?.token_ids());
            let whole = tokenizer.encode(&format!("{head}{probe}"))?;
            anyhow::ensure!(
                whole.token_ids() == split.as_slice(),
                "encoding is not split-invariant after special token {token:?}"
            );
        }
    }
    Ok(())
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    const CHUNK: usize = 4096;
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + CHUNK <= n && a[i..i + CHUNK] == b[i..i + CHUNK] {
        i += CHUNK;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

impl SessionTokenCache {
    /// `tokenizer` must produce the same ids as the normal path without its L1 cache.
    /// `boundary_tokens` are the split points; `added_tokens` every token the tokenizer splits
    /// out of raw text.
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        mut boundary_tokens: Vec<String>,
        added_tokens: Vec<String>,
        mode: SessionTokenizeMode,
        max_sessions: usize,
        max_bytes: usize,
    ) -> anyhow::Result<Self> {
        tokenizer.validate_prefix_cache()?;
        boundary_tokens.retain(|t| !t.is_empty());
        boundary_tokens.sort();
        boundary_tokens.dedup();
        anyhow::ensure!(!boundary_tokens.is_empty(), "no special tokens to split at");
        let others: Vec<String> = added_tokens
            .into_iter()
            .filter(|t| !t.is_empty() && boundary_tokens.binary_search(t).is_err())
            .collect();
        if let Some((a, b)) = first_ambiguous_pair(&boundary_tokens, &others) {
            anyhow::bail!("special tokens {a:?} and {b:?} can overlap");
        }
        check_split_invariance(tokenizer.as_ref(), &boundary_tokens)?;
        let matcher = AhoCorasick::new(&boundary_tokens)?;
        Ok(Self {
            tokenizer,
            matcher,
            mode,
            max_sessions: max_sessions.max(1),
            max_bytes,
            map: Mutex::new(SessionMap::default()),
            counters: Counters::default(),
        })
    }

    /// `(max sessions, max bytes)` from the environment.
    pub fn limits_from_env() -> (usize, usize) {
        let get = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(default)
        };
        (
            get(
                ENV_TOKENIZER_SESSION_CACHE_MAX_SESSIONS,
                DEFAULT_MAX_SESSIONS,
            ),
            get(ENV_TOKENIZER_SESSION_CACHE_BYTES, DEFAULT_MAX_BYTES),
        )
    }

    fn encode_incremental(
        &self,
        prev: Option<&SessionState>,
        input: String,
    ) -> anyhow::Result<(Vec<u32>, SessionEncodeStats, SessionState)> {
        let (reuse_bytes, reuse_tokens, kept) = prev
            .and_then(|prev| {
                let last = prev.checkpoints.last()?.0;
                let lcp = common_prefix_len(&prev.text.as_bytes()[..last], input.as_bytes());
                let kept = prev.checkpoints.partition_point(|cp| cp.0 <= lcp);
                let (bytes, tokens) = *prev.checkpoints.get(kept.checked_sub(1)?)?;
                Some((bytes, tokens, kept))
            })
            .unwrap_or((0, 0, 0));

        let mut tokens = Vec::with_capacity(reuse_tokens + (input.len() - reuse_bytes) / 3 + 16);
        let mut checkpoints = Vec::new();
        if let Some(prev) = prev {
            tokens.extend_from_slice(&prev.tokens[..reuse_tokens]);
            checkpoints.extend_from_slice(&prev.checkpoints[..kept]);
        }
        let mut pos = reuse_bytes;
        // A boundary at the very end has nothing after it, so it is not a checkpoint.
        for found in self.matcher.find_iter(&input[reuse_bytes..]) {
            let end = reuse_bytes + found.end();
            if end == input.len() {
                break;
            }
            tokens.extend_from_slice(self.tokenizer.encode(&input[pos..end])?.token_ids());
            checkpoints.push((end, tokens.len()));
            pos = end;
        }
        let state_tokens = tokens.len();
        if pos < input.len() {
            tokens.extend_from_slice(self.tokenizer.encode(&input[pos..])?.token_ids());
        }
        let stats = SessionEncodeStats {
            reused_tokens: reuse_tokens,
            new_tokens: tokens.len() - reuse_tokens,
        };
        let state = SessionState {
            text: input,
            tokens: tokens[..state_tokens].to_vec(),
            checkpoints,
        };
        Ok((tokens, stats, state))
    }

    /// Encode `input` for `session`. `full` is the normal path, run only in `verify` mode.
    pub fn encode(
        &self,
        session: &str,
        input: String,
        full: &dyn Tokenizer,
    ) -> anyhow::Result<(Vec<u32>, SessionEncodeStats)> {
        let prev = self
            .map
            .lock()
            .sessions
            .get(session)
            .map(|(state, _)| state.clone());
        let verify_input = (self.mode == SessionTokenizeMode::Verify).then(|| input.clone());
        let started = Instant::now();
        let (mut ids, stats, state) = self.encode_incremental(prev.as_deref(), input)?;
        let incremental_us = started.elapsed().as_micros() as u64;

        let mut keep = true;
        if let Some(verify_input) = verify_input {
            let full_ids = full.encode(&verify_input)?.token_ids().to_vec();
            let matched = full_ids == ids;
            tracing::debug!(
                tokens = full_ids.len(),
                reused_tokens = stats.reused_tokens,
                new_tokens = stats.new_tokens,
                incremental_us,
                matched,
                "tokenizer session cache verify"
            );
            if !matched {
                self.counters.mismatches.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    incremental_tokens = ids.len(),
                    full_tokens = full_ids.len(),
                    first_diff = ?ids.iter().zip(&full_ids).position(|(a, b)| a != b),
                    "tokenizer session cache mismatch; using the full encode and dropping the session"
                );
                ids = full_ids;
                keep = false;
            }
        }

        let c = &self.counters;
        let requests = c.requests.fetch_add(1, Ordering::Relaxed) + 1;
        c.reused_tokens
            .fetch_add(stats.reused_tokens as u64, Ordering::Relaxed);
        c.new_tokens
            .fetch_add(stats.new_tokens as u64, Ordering::Relaxed);
        if keep {
            self.insert(session, state);
        } else {
            self.forget(session);
        }
        if requests.is_multiple_of(STATS_EVERY) {
            self.log_stats();
        }
        Ok((ids, stats))
    }

    fn insert(&self, session: &str, state: SessionState) {
        let bytes = state.bytes();
        let mut map = self.map.lock();
        map.clock += 1;
        let clock = map.clock;
        if let Some((old, _)) = map
            .sessions
            .insert(Arc::from(session), (Arc::new(state), clock))
        {
            map.bytes = map.bytes.saturating_sub(old.bytes());
        }
        map.bytes += bytes;
        while map.sessions.len() > self.max_sessions
            || (map.bytes > self.max_bytes && map.sessions.len() > 1)
        {
            let Some(oldest) = map
                .sessions
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some((old, _)) = map.sessions.remove(&oldest) {
                map.bytes = map.bytes.saturating_sub(old.bytes());
                self.counters.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn forget(&self, session: &str) {
        let mut map = self.map.lock();
        if let Some((old, _)) = map.sessions.remove(session) {
            map.bytes = map.bytes.saturating_sub(old.bytes());
        }
    }

    fn log_stats(&self) {
        let (sessions, bytes) = {
            let map = self.map.lock();
            (map.sessions.len(), map.bytes)
        };
        let c = &self.counters;
        let reused = c.reused_tokens.load(Ordering::Relaxed);
        let new = c.new_tokens.load(Ordering::Relaxed);
        tracing::info!(
            mode = ?self.mode,
            requests = c.requests.load(Ordering::Relaxed),
            reused_tokens = reused,
            new_tokens = new,
            reuse_ratio = reused as f64 / (reused + new).max(1) as f64,
            mismatches = c.mismatches.load(Ordering::Relaxed),
            evictions = c.evictions.load(Ordering::Relaxed),
            sessions,
            bytes,
            "tokenizer session cache stats"
        );
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.lock().sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizers::HuggingFaceTokenizer;
    use crate::tokenizers::traits::{Decoder, Encoder};
    use crate::tokenizers::{DecodeResult, Encoding};

    const TINYLLAMA: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
    );

    fn hf() -> Arc<dyn Tokenizer> {
        static TOKENIZER: std::sync::OnceLock<Arc<dyn Tokenizer>> = std::sync::OnceLock::new();
        TOKENIZER
            .get_or_init(|| Arc::new(HuggingFaceTokenizer::from_file(TINYLLAMA).unwrap()))
            .clone()
    }

    fn new_cache(
        tokenizer: Arc<dyn Tokenizer>,
        boundaries: &[&str],
        added: &[&str],
        mode: SessionTokenizeMode,
        max_sessions: usize,
    ) -> anyhow::Result<SessionTokenCache> {
        SessionTokenCache::new(
            tokenizer,
            boundaries.iter().map(|t| t.to_string()).collect(),
            added.iter().map(|t| t.to_string()).collect(),
            mode,
            max_sessions,
            1 << 30,
        )
    }

    fn cache(mode: SessionTokenizeMode) -> SessionTokenCache {
        new_cache(hf(), &["<s>", "</s>"], &["<unk>"], mode, 16).unwrap()
    }

    fn full(text: &str) -> Vec<u32> {
        hf().encode(text).unwrap().token_ids().to_vec()
    }

    /// A chat prompt in the shape a chat template renders, ending with the generation prompt.
    fn chat(messages: &[&str]) -> String {
        let mut prompt = String::new();
        for (i, message) in messages.iter().enumerate() {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            prompt.push_str(&format!("<s>{role}\n{message}</s>\n"));
        }
        prompt.push_str("<s>assistant\n");
        prompt
    }

    fn encode(
        c: &SessionTokenCache,
        session: &str,
        prompt: &str,
    ) -> (Vec<u32>, SessionEncodeStats) {
        let (ids, stats) = c
            .encode(session, prompt.to_string(), hf().as_ref())
            .unwrap();
        assert_eq!(ids, full(prompt), "session {session}: {prompt:?}");
        assert_eq!(stats.reused_tokens + stats.new_tokens, ids.len());
        (ids, stats)
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(SessionTokenizeMode::parse(None), SessionTokenizeMode::Off);
        assert_eq!(
            SessionTokenizeMode::parse(Some("off")),
            SessionTokenizeMode::Off
        );
        assert_eq!(
            SessionTokenizeMode::parse(Some("0")),
            SessionTokenizeMode::Off
        );
        assert_eq!(
            SessionTokenizeMode::parse(Some("on")),
            SessionTokenizeMode::On
        );
        assert_eq!(
            SessionTokenizeMode::parse(Some(" Verify ")),
            SessionTokenizeMode::Verify
        );
    }

    #[test]
    fn growing_conversation_matches_full_encode_and_reuses_the_history() {
        let c = cache(SessionTokenizeMode::On);
        let long = "x".repeat(500);
        let messages = [
            "Hello there, how are you?",
            "I'm fine.   Thanks!\n\n  indented",
            "unicode: h\u{e9}llo w\u{f6}rld \u{4f60}\u{597d} \u{1f642}?",
            "tabs\tand\r\nwindows lines   ",
            long.as_str(),
            "",
            "end",
        ];
        for n in 1..=messages.len() {
            let (_, stats) = encode(&c, "s", &chat(&messages[..n]));
            if n > 1 {
                assert!(stats.reused_tokens > 0, "turn {n} reused nothing");
            }
        }
    }

    #[test]
    fn edited_or_truncated_history_falls_back_to_the_deepest_shared_checkpoint() {
        let c = cache(SessionTokenizeMode::On);
        let first = chat(&["one", "two", "three", "four"]);
        encode(&c, "s", &first);
        // An edit inside the third message keeps the first two messages.
        let edited = chat(&["one", "two", "threX", "four", "five"]);
        let (_, stats) = encode(&c, "s", &edited);
        let shared = full("<s>user\none</s>\n<s>assistant\ntwo</s>\n<s>");
        assert_eq!(stats.reused_tokens, shared.len());
        // The client drops the oldest turns: nothing before the first message is shared.
        let truncated = chat(&["three", "four", "five"]);
        let (_, stats) = encode(&c, "s", &truncated);
        assert!(stats.reused_tokens <= full("<s>").len());
        // The new prompt is a prefix of the old one: reuse stops at its last boundary.
        let (_, stats) = encode(&c, "s", &chat(&["three"]));
        assert!(stats.reused_tokens > 0);
        // A prompt without special tokens is encoded whole.
        let (_, stats) = encode(&c, "s", "no special tokens at all");
        assert_eq!(stats.reused_tokens, 0);
    }

    #[test]
    fn text_after_the_previous_prompt_end_is_encoded_with_it() {
        // The text after the last boundary is never reused, so a continuation of the
        // previous generation prompt merges as in a full encode.
        let c = cache(SessionTokenizeMode::On);
        encode(&c, "s", "<s>user\nhi</s>\n<s>assistant\n");
        encode(
            &c,
            "s",
            "<s>user\nhi</s>\n<s>assistant\nnewline merges   here</s>\n<s>user\nok",
        );
    }

    #[test]
    fn sessions_are_isolated_and_least_recently_used_is_evicted() {
        let c = new_cache(hf(), &["<s>", "</s>"], &[], SessionTokenizeMode::On, 2).unwrap();
        for s in ["a", "b", "c"] {
            encode(&c, s, &chat(&[s, "reply"]));
        }
        assert_eq!(c.len(), 2);
        let (_, stats) = encode(&c, "a", &chat(&["a", "reply", "more"]));
        assert_eq!(stats.reused_tokens, 0, "session a was evicted");
        // Reusing another session's id with a different prompt is still exact.
        encode(&c, "c", &chat(&["zzz"]));
    }

    #[test]
    fn verify_mode_finds_no_mismatch_on_exact_encodes() {
        let c = cache(SessionTokenizeMode::Verify);
        for n in 1..6 {
            let messages: Vec<String> = (0..n)
                .map(|i| format!("message {i} with  spaces"))
                .collect();
            let refs: Vec<&str> = messages.iter().map(String::as_str).collect();
            encode(&c, "v", &chat(&refs));
        }
        assert_eq!(c.counters.mismatches.load(Ordering::Relaxed), 0);
    }

    /// Byte ids, plus a trailing id 0 for inputs longer than 32 bytes: split-invariant on the
    /// short construction probes, not on longer text.
    struct LongInputMarker;
    impl Encoder for LongInputMarker {
        fn encode(&self, input: &str) -> crate::tokenizers::Result<Encoding> {
            let mut ids: Vec<u32> = input.bytes().map(u32::from).collect();
            if input.len() > 32 {
                ids.push(0);
            }
            Ok(Encoding::Sp(ids))
        }
        fn encode_batch(&self, inputs: &[&str]) -> crate::tokenizers::Result<Vec<Encoding>> {
            inputs.iter().map(|i| self.encode(i)).collect()
        }
    }
    impl Decoder for LongInputMarker {
        fn decode(&self, _: &[u32], _: bool) -> crate::tokenizers::Result<DecodeResult> {
            Ok(DecodeResult::Complete(String::new()))
        }
    }
    impl Tokenizer for LongInputMarker {
        fn validate_prefix_cache(&self) -> crate::tokenizers::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn verify_mode_returns_the_full_encode_on_a_mismatch_and_drops_the_session() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(LongInputMarker);
        let c = new_cache(
            tokenizer.clone(),
            &["<s>"],
            &[],
            SessionTokenizeMode::Verify,
            4,
        )
        .unwrap();
        let prompt = format!("<s>{}<s>{}", "a".repeat(40), "b".repeat(40));
        let (ids, _) = c.encode("x", prompt.clone(), tokenizer.as_ref()).unwrap();
        assert_eq!(ids, tokenizer.encode(&prompt).unwrap().token_ids());
        assert_eq!(c.counters.mismatches.load(Ordering::Relaxed), 1);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn ambiguous_special_tokens_are_refused() {
        let refused = |boundaries: &[&str], added: &[&str]| {
            new_cache(hf(), boundaries, added, SessionTokenizeMode::On, 4).is_err()
        };
        assert!(refused(&["<a>", "a>b"], &[]), "boundaries that overlap");
        assert!(
            refused(&["<s>"], &["<s>x"]),
            "an added token containing a boundary"
        );
        assert!(refused(&["aba"], &[]), "a boundary that overlaps itself");
        assert!(refused(&[], &[]), "no boundary");
    }

    #[test]
    fn tokenizers_that_are_not_split_invariant_are_refused() {
        // Metaspace with `prepend_scheme: first` marks only the start of the whole input, so
        // the text after a split would be encoded as a new start.
        let json = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[{"id":1,"content":"<s>","single_word":false,"lstrip":false,
                "rstrip":false,"normalized":false,"special":true}],
            "normalizer":null,
            "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"<unk>":0,"<s>":1,"▁a":2,"hello":3,"▁hello":4},
                "unk_token":"<unk>"}}"#;
        let hf: tokenizers::Tokenizer = json.parse().unwrap();
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf));
        let error = new_cache(tokenizer, &["<s>"], &[], SessionTokenizeMode::On, 4)
            .err()
            .expect("refused");
        assert!(error.to_string().contains("split-invariant"), "{error}");
    }

    #[test]
    fn common_prefix_len_works_across_chunks() {
        let a = vec![7u8; 10_000];
        let mut b = a.clone();
        assert_eq!(common_prefix_len(&a, &b), 10_000);
        b[9_000] = 1;
        assert_eq!(common_prefix_len(&a, &b), 9_000);
        assert_eq!(common_prefix_len(&a[..5], &b), 5);
        assert_eq!(common_prefix_len(&[], &b), 0);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reasoning parser for Google Gemma 4 thinking models.
//!
//! Gemma 4 emits chain-of-thought reasoning between dedicated channel
//! delimiters:
//!
//! ```text
//! <|channel>thought
//! ...chain of thought reasoning...<channel|>
//! Final answer text.
//! ```
//!
//! The `thought\n` role label inside the channel is a structural artefact
//! analogous to the `user\n` label in `<|turn>user\n...`. Downstream consumers
//! expect the reasoning content without that label, so this parser strips it.
//!
//! Both the start and end markers are special tokens. They are visible in the
//! decoded text only when the inference engine is configured with
//! `skip_special_tokens=False`. If the markers are stripped before the parser
//! sees them, the parser falls back to passing the text through as
//! `normal_text` (no reasoning extracted).

use crate::ParserResult;
use crate::ReasoningParser;

const START_TOKEN: &str = "<|channel>";
const END_TOKEN: &str = "<channel|>";
const THOUGHT_PREFIX: &str = "thought\n";

/// Returns the length of the longest suffix of `s` that is also a prefix of
/// `delim`. Used to detect partial multi-byte markers split across streaming
/// chunk boundaries (e.g. `"...<|chan"` is a partial prefix of `<|channel>`).
fn overlap(s: &str, delim: &str) -> usize {
    let max = delim.len().min(s.len());
    for i in (1..=max).rev() {
        if !delim.is_char_boundary(i) || !s.is_char_boundary(s.len() - i) {
            continue;
        }
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}

#[derive(Debug, Clone)]
pub struct Gemma4ReasoningParser {
    /// Streaming buffer for accumulated text yet to be classified.
    buffer: String,
    /// True once we've observed `<|channel>` and are inside a reasoning span.
    in_reasoning: bool,
    /// True once we've stripped the `thought\n` prefix (or determined it is
    /// not present) for the current reasoning span.
    prefix_resolved: bool,
    /// Reasoning text accumulated so far for the current span. Used to decide
    /// whether the accumulated bytes are still a strict prefix of
    /// `thought\n` (case 2) or have diverged (case 3).
    reasoning_accum: String,
}

impl Gemma4ReasoningParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            in_reasoning: false,
            prefix_resolved: false,
            reasoning_accum: String::new(),
        }
    }

    fn reset_span(&mut self) {
        self.in_reasoning = false;
        self.prefix_resolved = false;
        self.reasoning_accum.clear();
    }
}

impl Default for Gemma4ReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip the `thought\n` role label from the start of `text` if present.
fn strip_thought_prefix(text: &str) -> &str {
    text.strip_prefix(THOUGHT_PREFIX).unwrap_or(text)
}

/// Decide what slice of `raw_reasoning` to emit given the current
/// prefix-stripping state. Returns `(emit, new_prefix_resolved)`.
///
/// **Precondition:** `raw_reasoning` MUST be a suffix of `accum` — i.e. the
/// caller has already pushed the new delta into the accumulator before
/// calling this. We rely on that to compute `prev_len = accum.len() -
/// raw_reasoning.len()` (the length of the accumulator *before* this delta).
/// Violating the precondition would underflow `prev_len` and corrupt the
/// emit-slice. The streaming driver in `parse_reasoning_streaming_incremental`
/// upholds this invariant by always pushing `raw` into `self.reasoning_accum`
/// immediately before the call.
///
/// Case 1: accumulated reasoning starts with `thought\n` — strip it from the
///   delta (or suppress entirely if the delta lies inside the prefix).
/// Case 2: accumulated reasoning is a strict prefix of `thought\n` — suppress
///   so we can decide once more bytes arrive.
/// Case 3: accumulated reasoning diverged from the prefix — emit the
///   buffered reasoning verbatim (data preservation).
fn resolve_prefix<'a>(accum: &'a str, raw_reasoning: &'a str) -> (&'a str, bool) {
    debug_assert!(
        accum.ends_with(raw_reasoning),
        "resolve_prefix precondition violated: raw_reasoning ({:?}) must be a suffix of accum ({:?})",
        raw_reasoning,
        accum,
    );
    if accum.starts_with(THOUGHT_PREFIX) {
        let prev_len = accum.len() - raw_reasoning.len();
        if prev_len >= THOUGHT_PREFIX.len() {
            // Prefix was already consumed by earlier deltas — pass through.
            return (raw_reasoning, true);
        }
        let chars_of_prefix_in_delta = THOUGHT_PREFIX.len() - prev_len;
        let stripped = &raw_reasoning[chars_of_prefix_in_delta.min(raw_reasoning.len())..];
        if !stripped.is_empty() || accum.len() >= THOUGHT_PREFIX.len() {
            return (stripped, true);
        }
        return ("", false);
    }
    if THOUGHT_PREFIX.starts_with(accum) {
        // Strict prefix of "thought\n" — suppress until we know.
        return ("", false);
    }
    // Diverged: emit full buffered reasoning verbatim.
    (accum, true)
}

impl ReasoningParser for Gemma4ReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        // Non-streaming path: we have the complete text, so we can use plain
        // string operations.
        let start_idx = text.find(START_TOKEN);
        let end_idx = text.find(END_TOKEN);
        match (start_idx, end_idx) {
            (None, None) => {
                // No reasoning markers visible at all (either model didn't
                // emit them, or skip_special_tokens stripped them).
                ParserResult {
                    normal_text: text.to_string(),
                    reasoning_text: String::new(),
                }
            }
            (Some(s), end_opt) => {
                let pre = &text[..s];
                let rest = &text[s + START_TOKEN.len()..];
                let (reasoning_raw, post) = match end_opt
                    .filter(|e| *e > s + START_TOKEN.len())
                    .map(|e| e - (s + START_TOKEN.len()))
                {
                    Some(end_rel) => (&rest[..end_rel], &rest[end_rel + END_TOKEN.len()..]),
                    None => (rest, ""),
                };
                let reasoning = strip_thought_prefix(reasoning_raw).to_string();
                let mut normal = String::with_capacity(pre.len() + post.len());
                normal.push_str(pre);
                normal.push_str(post);
                ParserResult {
                    normal_text: normal,
                    reasoning_text: reasoning,
                }
            }
            (None, Some(e)) => {
                // Dangling end marker without start marker — upstream's
                // offline parser still treats text-before as reasoning. Mirror
                // that so model emissions where the start tag was stripped
                // (e.g. tokenizer with skip_special_tokens) don't lose the
                // reasoning content entirely.
                let reasoning_raw = &text[..e];
                let post = &text[e + END_TOKEN.len()..];
                let reasoning = strip_thought_prefix(reasoning_raw).to_string();
                ParserResult {
                    normal_text: post.to_string(),
                    reasoning_text: reasoning,
                }
            }
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        // Aggregate this delta with the carry buffer for prefix detection.
        let mut work = std::mem::take(&mut self.buffer);
        work.push_str(text);

        let mut normal = String::new();
        let mut reasoning_emit = String::new();

        loop {
            if !self.in_reasoning {
                // Look for either the full start marker or a partial-prefix at
                // the buffer's end (which we must hold back).
                if let Some(idx) = work.find(START_TOKEN) {
                    normal.push_str(&work[..idx]);
                    work = work[idx + START_TOKEN.len()..].to_string();
                    self.in_reasoning = true;
                    self.prefix_resolved = false;
                    self.reasoning_accum.clear();
                    continue;
                }
                let lap = overlap(&work, START_TOKEN);
                if lap > 0 {
                    let split = work.len() - lap;
                    normal.push_str(&work[..split]);
                    self.buffer = work[split..].to_string();
                } else {
                    normal.push_str(&work);
                    self.buffer.clear();
                }
                break;
            }

            // self.in_reasoning == true
            if let Some(idx) = work.find(END_TOKEN) {
                let raw = &work[..idx];
                self.reasoning_accum.push_str(raw);
                if !self.prefix_resolved {
                    let (emit, resolved) = resolve_prefix(&self.reasoning_accum, raw);
                    if resolved {
                        reasoning_emit.push_str(emit);
                        self.prefix_resolved = true;
                    }
                    // If still unresolved at end-of-span, the accumulated text
                    // was a strict prefix of `thought\n` — by definition there
                    // is nothing useful to emit; drop it.
                } else {
                    reasoning_emit.push_str(raw);
                }
                work = work[idx + END_TOKEN.len()..].to_string();
                self.reset_span();
                continue;
            }

            // No end marker yet. Hold back any partial-end-marker suffix.
            let lap = overlap(&work, END_TOKEN);
            let split = work.len() - lap;
            let raw = work[..split].to_string();
            self.buffer = work[split..].to_string();

            if !raw.is_empty() {
                self.reasoning_accum.push_str(&raw);
                if !self.prefix_resolved {
                    let (emit, resolved) = resolve_prefix(&self.reasoning_accum, &raw);
                    if resolved {
                        reasoning_emit.push_str(emit);
                        self.prefix_resolved = true;
                    }
                } else {
                    reasoning_emit.push_str(&raw);
                }
            }
            break;
        }

        ParserResult {
            normal_text: normal,
            reasoning_text: reasoning_emit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] // CASE.10 — non-streaming basic case
    fn detect_basic_thinking() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning(
            "<|channel>thought\nstep one\nstep two<channel|>The answer is 42.",
            &[],
        );
        assert_eq!(r.reasoning_text, "step one\nstep two");
        assert_eq!(r.normal_text, "The answer is 42.");
    }

    #[test] // CASE.3 — no reasoning markers, pass through
    fn detect_no_markers_passes_through() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("just a plain answer", &[]);
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "just a plain answer");
    }

    #[test] // CASE.5 — reasoning open without close (truncation): everything after
    // start marker is reasoning content.
    fn detect_truncated_reasoning_open_only() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("intro <|channel>thought\npartial", &[]);
        assert_eq!(r.reasoning_text, "partial");
        assert_eq!(r.normal_text, "intro ");
    }

    #[test] // CASE.13 — text before AND after the reasoning span preserved
    fn detect_text_before_and_after() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning(
            "Hello. <|channel>thought\nrumination<channel|> Goodbye.",
            &[],
        );
        assert_eq!(r.reasoning_text, "rumination");
        assert_eq!(r.normal_text, "Hello.  Goodbye.");
    }

    #[test] // CASE.5, CASE.13 — dangling end marker, missing start (upstream INVALID_SIMPLE)
    fn detect_dangling_end_marker_extracts_prefix_as_reasoning() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("some thinking<channel|>final answer", &[]);
        assert_eq!(r.reasoning_text, "some thinking");
        assert_eq!(r.normal_text, "final answer");
    }

    #[test] // CASE.5, CASE.13 — dangling end + thought prefix on the head (upstream INVALID_COMPLETE)
    fn detect_dangling_end_marker_strips_thought_prefix() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning("thought\nrumination<channel|>final answer", &[]);
        assert_eq!(r.reasoning_text, "rumination");
        assert_eq!(r.normal_text, "final answer");
    }

    #[test] // CASE.20 — `thought\n` prefix absent (some tokens drop it): pass through unchanged
    fn detect_no_thought_prefix() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.detect_and_parse_reasoning(
            "<|channel>raw reasoning without prefix<channel|>answer",
            &[],
        );
        assert_eq!(r.reasoning_text, "raw reasoning without prefix");
        assert_eq!(r.normal_text, "answer");
    }

    #[test] // CASE.8 — streaming arrival, single chunk
    fn streaming_single_chunk() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.parse_reasoning_streaming_incremental(
            "<|channel>thought\nrumination<channel|>final",
            &[],
        );
        assert_eq!(r.reasoning_text, "rumination");
        assert_eq!(r.normal_text, "final");
    }

    #[test] // CASE.8 — streaming with `thought\n` split across deltas
    fn streaming_thought_prefix_split_across_deltas() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "<|channel>",
            "thou",
            "ght\n",
            "real reasoning here",
            "<channel|>",
            "the answer.",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "real reasoning here");
        assert_eq!(normal, "the answer.");
    }

    #[test] // CASE.8 — start marker split across deltas
    fn streaming_start_marker_split() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "intro ",
            "<|chan", // partial start marker
            "nel>thought\n",
            "rumination",
            "<channel|>",
            "outro",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "rumination");
        assert_eq!(normal, "intro outro");
    }

    #[test] // CASE.8 — end marker split across deltas
    fn streaming_end_marker_split() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "<|channel>thought\n",
            "thinking",
            "<chan", // partial end marker
            "nel|>",
            "answer",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "thinking");
        assert_eq!(normal, "answer");
    }

    #[test] // CASE.8 — diverged accumulated text (no `thought\n` prefix at all)
    fn streaming_no_thought_prefix_streaming() {
        let mut p = Gemma4ReasoningParser::new();
        let chunks = [
            "<|channel>",
            "raw stream of consciousness",
            "<channel|>",
            "answer",
        ];
        let mut reasoning = String::new();
        let mut normal = String::new();
        for c in chunks {
            let r = p.parse_reasoning_streaming_incremental(c, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "raw stream of consciousness");
        assert_eq!(normal, "answer");
    }

    #[test] // CASE.3 — streaming with no markers at all
    fn streaming_no_markers() {
        let mut p = Gemma4ReasoningParser::new();
        let r = p.parse_reasoning_streaming_incremental("plain text only", &[]);
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "plain text only");
    }

    #[test] // CASE.2 — multiple reasoning spans back-to-back
    fn streaming_multiple_reasoning_spans() {
        let mut p = Gemma4ReasoningParser::new();
        let input =
            "<|channel>thought\nfirst<channel|>answer1<|channel>thought\nsecond<channel|>answer2";
        let r = p.parse_reasoning_streaming_incremental(input, &[]);
        // Both spans concatenated into the cumulative deltas.
        assert!(r.reasoning_text.contains("first"));
        assert!(r.reasoning_text.contains("second"));
        assert!(r.normal_text.contains("answer1"));
        assert!(r.normal_text.contains("answer2"));
    }

    #[test] // CASE.9 — paired reasoning + tool call. The reasoning parser
    // must extract the channel content as `reasoning_text` and leave the
    // following `<|tool_call>...<tool_call|>` markers intact in
    // `normal_text` for the tool-call parser to consume downstream.
    fn paired_reasoning_then_tool_call_non_streaming() {
        let mut p = Gemma4ReasoningParser::new();
        let input = concat!(
            "<|channel>thought\nthinking about the request<channel|>",
            "<|tool_call>call:get_weather{location:<|\"|>Tokyo<|\"|>}<tool_call|>",
        );
        let r = p.detect_and_parse_reasoning(input, &[]);
        assert_eq!(r.reasoning_text, "thinking about the request");
        assert_eq!(
            r.normal_text, r#"<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>"#,
            "tool-call markers must survive reasoning extraction",
        );
    }

    // ----- Explicit N/A coverage notes (per lib/parsers/TEST_CASES.md) -----
    //
    // CASE.1, CASE.4, CASE.6, CASE.7  — Tool-call-only categories. N/A for
    //          a reasoning parser.
    // CASE.11, CASE.12 — `tool_choice` and `finish_reason`: tool-call concerns,
    //          N/A for reasoning. (Universal cross-parser gap regardless;
    //          see notes in `tool_calling/gemma4/parser.rs`.)
    // CASE.14 — Empty / null content: empty input is covered implicitly
    //          via the no-markers passthrough cases (`detect_no_markers_*`,
    //          `streaming_no_markers`).
    // CASE.15 — Duplicate calls: tool-call concept; N/A. Multi-span
    //          reasoning is the analog and is covered by
    //          `streaming_multiple_reasoning_spans`.
    // CASE.xml1 / CASE.xml2 — XML-family only. N/A.
    // CASE.harmony1 — Harmony only. N/A.
}

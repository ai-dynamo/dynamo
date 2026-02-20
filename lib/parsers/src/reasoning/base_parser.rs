// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Reasoning and Tool Call Interplay
//!
//! Models like GLM-4.5/4.7 and Qwen3 interleave reasoning blocks with tool calls:
//!
//! ```text
//! <think>reasoning about what tool to call</think>
//! <tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>
//! <think>reasoning about the result</think>
//! <tool_call>summarize<arg_key>text</arg_key><arg_value>...</arg_value></tool_call>
//! ```
//!
//! The reasoning parser and the tool call parser are **independent, sequential** stages:
//!
//! 1. **Reasoning parser** (`BasicReasoningParser`) splits the stream into:
//!    - `reasoning_content`: everything inside `<think>...</think>` blocks
//!    - `normal_text`: everything outside (including tool call tags)
//! 2. **Tool call parser** (`glm47` / others) then processes `normal_text` to extract
//!    `<tool_call>...</tool_call>` blocks.
//!
//! This means tool calls **must** appear outside `<think>` blocks to be detected.
//! If a model erroneously emits a tool call inside a `<think>` block (observed in
//! GLM-4.7 under very long contexts), the tool call parser will not see it.
//!
//! ## `force_reasoning` and tokenizer behavior
//!
//! Some models (e.g. GLM-5-FP8 served via ZAI) consume `<think>` as a special
//! tokenizer token and never emit it as literal text. In that case use
//! `force_reasoning=true` (`deepseek_r1` parser), which treats all output as
//! reasoning until `</think>` is seen. Models that do emit `<think>` as text
//! (standard serving, Qwen3, GLM-4.5) should use `force_reasoning=false`
//! (`glm45`, `nemotron_deci`, `qwen3` parsers).

use crate::{ParserResult, ReasoningParser};

/// Returns the length of the longest suffix of `s` that is also a prefix of `delim`.
///
/// Ported from ollama's `thinking/parser.go::overlap()`. Used to detect partial
/// tags split across streaming chunk boundaries (e.g., `"Hello world <th"` where
/// `<th` is a prefix of `<think>`).
fn overlap(s: &str, delim: &str) -> usize {
    let max = delim.len().min(s.len());
    for i in (1..=max).rev() {
        if !delim.is_char_boundary(i) {
            continue; // Skip mid-codepoint positions (e.g., multi-byte `◁` in Kimi tags)
        }
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}

#[derive(Default, Debug, Clone)]
pub struct BasicReasoningParser {
    think_start_token: String,
    think_end_token: String,
    _in_reasoning: bool,
    stream_reasoning: bool,
    _buffer: String,
    stripped_think_start: bool,
}

impl BasicReasoningParser {
    pub fn new(
        think_start_token: String,
        think_end_token: String,
        force_reasoning: bool,
        stream_reasoning: bool,
    ) -> Self {
        Self {
            think_start_token,
            think_end_token,
            _in_reasoning: force_reasoning,
            stream_reasoning,
            _buffer: String::new(),
            stripped_think_start: false,
        }
    }
}

impl ReasoningParser for BasicReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        let has_think_tag = text.contains(&self.think_start_token);
        let in_reasoning = self._in_reasoning || has_think_tag;
        if !in_reasoning {
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: String::new(),
            };
        }

        // If force_reasoning and no start tag, treat entire text as reasoning
        if self._in_reasoning && !has_think_tag && !text.contains(&self.think_end_token) {
            return ParserResult {
                normal_text: String::new(),
                reasoning_text: text.to_string(),
            };
        }

        // Extract all <think>...</think> pairs using cursor-based iteration
        let mut reasoning_parts = Vec::new();
        let mut normal_parts = Vec::new();
        let mut cursor = 0;
        let mut currently_reasoning = self._in_reasoning;

        while cursor < text.len() {
            if currently_reasoning {
                // We're inside a reasoning block — look for end token
                if let Some(end_offset) = text[cursor..].find(&self.think_end_token) {
                    reasoning_parts.push(&text[cursor..cursor + end_offset]);
                    cursor += end_offset + self.think_end_token.len();
                    currently_reasoning = false;
                } else {
                    // No end token — rest is reasoning (truncated)
                    reasoning_parts.push(&text[cursor..]);
                    cursor = text.len();
                }
            } else {
                // We're in normal text — look for start token
                if let Some(start_offset) = text[cursor..].find(&self.think_start_token) {
                    normal_parts.push(&text[cursor..cursor + start_offset]);
                    cursor += start_offset + self.think_start_token.len();
                    currently_reasoning = true;
                } else {
                    // No more think blocks — rest is normal text
                    normal_parts.push(&text[cursor..]);
                    cursor = text.len();
                }
            }
        }

        let reasoning_text = reasoning_parts.join("").trim().to_string();
        let normal_text = normal_parts.join("").trim().to_string();

        // Note: self._in_reasoning is intentionally NOT updated here. This method is
        // documented to "reset or ignore internal streaming state" (see trait doc). Callers
        // should not mix detect_and_parse_reasoning with parse_reasoning_streaming_incremental
        // on the same parser instance.

        ParserResult {
            normal_text,
            reasoning_text,
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        self._buffer.push_str(text);

        let mut accumulated_normal = String::new();
        let mut accumulated_reasoning = String::new();

        // Loop to exhaust all state transitions within a single chunk. Without this,
        // a chunk containing two complete <think>...</think> blocks would process only
        // the first transition and buffer the rest, risking content loss at end-of-stream.
        loop {
            let current_text = self._buffer.clone();

            // Strip leading <think> tag if not yet stripped. Handles two cases:
            // 1. force_reasoning=true where the model also emits <think> as text
            // 2. First call where <think> arrives at buffer position 0
            // Mid-text <think> (position > 0) falls through to the find() branch below.
            if !self.stripped_think_start
                && current_text.starts_with(self.think_start_token.as_str())
            {
                self._buffer = current_text[self.think_start_token.len()..].to_string();
                self.stripped_think_start = true;
                self._in_reasoning = true;
                continue;
            }

            if self._in_reasoning {
                if let Some(end_idx) = current_text.find(self.think_end_token.as_str()) {
                    // End of reasoning block: accumulate content and transition out.
                    accumulated_reasoning.push_str(&current_text[..end_idx]);
                    let after_end = end_idx + self.think_end_token.len();
                    self._buffer = current_text[after_end..].to_string();
                    self._in_reasoning = false;
                    self.stripped_think_start = false; // Allow detecting next <think> block
                    continue; // Process remainder — may contain further blocks
                } else {
                    // No complete end token — check for partial at end of buffer
                    // (e.g., "reasoning content</th" where "</th" is a prefix of "</think>").
                    if self.stream_reasoning {
                        let ol = overlap(&current_text, &self.think_end_token);
                        if ol >= 2 {
                            let safe_end = current_text.len() - ol;
                            if safe_end > 0 {
                                accumulated_reasoning.push_str(&current_text[..safe_end]);
                            }
                            self._buffer = current_text[safe_end..].to_string();
                        } else {
                            accumulated_reasoning.push_str(&current_text);
                            self._buffer.clear();
                        }
                    }
                    // When stream_reasoning=false, buffer retains all content until
                    // </think> arrives — no overlap check needed.
                    break;
                }
            } else {
                // Not in reasoning — look for the next <think> block.
                if let Some(think_pos) = current_text.find(self.think_start_token.as_str()) {
                    accumulated_normal.push_str(&current_text[..think_pos]);
                    let after_start = think_pos + self.think_start_token.len();
                    self._buffer = current_text[after_start..].to_string();
                    self._in_reasoning = true;
                    self.stripped_think_start = true;
                    continue; // Process reasoning content
                } else {
                    // No complete start token — check for partial at end of buffer
                    // (e.g., "Hello world <th" where "<th" is a prefix of "<think>").
                    // Require overlap >= 2 so a lone `<` passes through for tool call
                    // XML tags like `<invoke>` or `<minimax:tool_call>`.
                    let ol = overlap(&current_text, &self.think_start_token);
                    if ol >= 2 {
                        let safe_end = current_text.len() - ol;
                        if safe_end > 0 {
                            accumulated_normal.push_str(&current_text[..safe_end]);
                        }
                        self._buffer = current_text[safe_end..].to_string();
                    } else {
                        accumulated_normal.push_str(&current_text);
                        self._buffer.clear();
                    }
                    break;
                }
            }
        }

        ParserResult {
            normal_text: accumulated_normal,
            reasoning_text: accumulated_reasoning,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_and_parse_reasoning_reasoning() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result =
            parser.detect_and_parse_reasoning("<think>with reasoning</think> and more text.", &[]);
        assert_eq!(result.normal_text, "and more text.");
        assert_eq!(result.reasoning_text, "with reasoning");
    }
    #[test]
    fn test_detect_and_parse_reasoning_reasoning_no_reasoning() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("This is a test without reasoning.", &[]);
        assert_eq!(result.normal_text, "This is a test without reasoning.");
        assert_eq!(result.reasoning_text, "");
    }
    #[test]
    fn test_detect_and_parse_reasoning_reasoning_truncated_reasoning() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("<think>with truncated reasoning", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "with truncated reasoning");
    }

    #[test]
    fn test_parse_reasoning_streaming_incremental() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.parse_reasoning_streaming_incremental("<thi", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_parse_reasoning_streaming_incremental_complete() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.parse_reasoning_streaming_incremental(
            "<think>with reasoning</think> and more text.",
            &[],
        );
        assert_eq!(result.normal_text, " and more text.");
        assert_eq!(result.reasoning_text, "with reasoning");
    }

    #[test]
    fn test_parse_reasoning_streaming_incremental_no_end_token() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), true, true);
        let result = parser.parse_reasoning_streaming_incremental("<think>with reasoning", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "with reasoning");
    }

    #[test]
    fn test_detect_and_parse_reasoning_multiple_reasoning_blocks() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning(
            "<think>first reasoning</think> middle <think>second reasoning</think> end",
            &[],
        );
        assert_eq!(result.normal_text, "middle  end");
        assert_eq!(result.reasoning_text, "first reasoningsecond reasoning");
    }

    #[test]
    fn test_streaming_multiple_reasoning_blocks() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, false);
        let result1 = parser
            .parse_reasoning_streaming_incremental("<think>first reasoning</think> middle", &[]);
        assert_eq!(result1.normal_text, " middle");
        assert_eq!(result1.reasoning_text, "first reasoning");

        // Second reasoning block: space before <think> is normal prefix, reasoning extracted
        let result2 = parser
            .parse_reasoning_streaming_incremental(" <think>second reasoning</think> end", &[]);
        assert_eq!(result2.reasoning_text, "second reasoning");
        assert_eq!(result2.normal_text, "  end"); // " " prefix + " end" suffix
    }

    #[test]
    fn test_partial_token_matching_opening_tag() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        // Feed partial opening tag
        let result1 = parser.parse_reasoning_streaming_incremental("<th", &[]);
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "");

        // Complete the opening tag and add content
        let result2 = parser.parse_reasoning_streaming_incremental(
            "ink>reasoning content</think> normal text",
            &[],
        );
        assert_eq!(result2.normal_text, " normal text");
        assert_eq!(result2.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_partial_token_matching_closing_tag() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, false);

        // Start with complete opening and partial content
        let result1 =
            parser.parse_reasoning_streaming_incremental("<think>reasoning content</th", &[]);
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "");

        // Complete the closing tag
        let result2 = parser.parse_reasoning_streaming_incremental("ink> normal text", &[]);
        assert_eq!(result2.normal_text, " normal text");
        assert_eq!(result2.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_buffer_state_persistence_across_calls() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, false);

        // First call - partial opening tag
        let result1 = parser.parse_reasoning_streaming_incremental("<th", &[]);
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "");

        // Second call - complete opening tag, start reasoning
        let result2 = parser.parse_reasoning_streaming_incremental("ink>part1 ", &[]);
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "");

        // Third call - more reasoning content
        let result3 = parser.parse_reasoning_streaming_incremental("part2 ", &[]);
        assert_eq!(result3.normal_text, "");
        assert_eq!(result3.reasoning_text, "");

        // Fourth call - end reasoning and normal text
        let result4 = parser.parse_reasoning_streaming_incremental("part3</think> normal", &[]);
        assert_eq!(result4.normal_text, " normal");
        assert_eq!(result4.reasoning_text, "part1 part2 part3");
    }

    #[test]
    fn test_streaming_with_stream_reasoning_enabled() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        // Start reasoning block
        let result1 = parser.parse_reasoning_streaming_incremental("<think>reasoning ", &[]);
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "reasoning ");

        // Continue streaming reasoning
        let result2 = parser.parse_reasoning_streaming_incremental("content ", &[]);
        assert_eq!(result2.normal_text, "");
        assert_eq!(result2.reasoning_text, "content ");

        // End reasoning block
        let result3 = parser.parse_reasoning_streaming_incremental("more</think> normal", &[]);
        assert_eq!(result3.normal_text, " normal");
        assert_eq!(result3.reasoning_text, "more");
    }

    #[test]
    fn test_nested_reasoning_blocks() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning(
            "<think>outer <think>inner</think> reasoning</think> normal",
            &[],
        );
        // Cursor-based parsing: first <think> starts reasoning, first </think> ends it.
        // "outer <think>inner" is reasoning (inner <think> is just text within reasoning).
        // " reasoning</think> normal" is normal text (stray </think> passes through).
        assert_eq!(result.reasoning_text, "outer <think>inner");
        assert_eq!(result.normal_text, "reasoning</think> normal");
    }

    #[test]
    fn test_malformed_missing_closing_tag() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("<think>reasoning without closing tag", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "reasoning without closing tag");
    }

    #[test]
    fn test_malformed_stray_closing_tag() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("normal text</think> more normal", &[]);
        assert_eq!(result.normal_text, "normal text</think> more normal");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_malformed_multiple_opening_tags() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser
            .detect_and_parse_reasoning("<think>first <think>second reasoning</think> normal", &[]);
        // Cursor-based: first <think> opens reasoning, finds first </think>.
        // Inner <think> is just text within the reasoning block.
        assert_eq!(result.reasoning_text, "first <think>second reasoning");
        assert_eq!(result.normal_text, "normal");
    }

    #[test]
    fn test_empty_reasoning_block() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("<think></think> normal text", &[]);
        assert_eq!(result.normal_text, "normal text");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_whitespace_only_reasoning_block() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("<think>   \n\t  </think> normal text", &[]);
        assert_eq!(result.normal_text, "normal text");
        assert_eq!(result.reasoning_text, ""); // Should be empty after trim
    }

    #[test]
    fn test_force_reasoning_mode() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), true, true);
        let result = parser.detect_and_parse_reasoning("no think tags here", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "no think tags here");
    }

    #[test]
    fn test_streaming_reset_state_after_complete_block() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        // Process complete reasoning block
        let result1 =
            parser.parse_reasoning_streaming_incremental("<think>reasoning</think> normal", &[]);
        assert_eq!(result1.normal_text, " normal");
        assert_eq!(result1.reasoning_text, "reasoning");

        // Process normal text - should not be affected by previous state
        let result2 = parser.parse_reasoning_streaming_incremental(" more normal text", &[]);
        assert_eq!(result2.normal_text, " more normal text");
        assert_eq!(result2.reasoning_text, "");

        // Subsequent reasoning blocks should now be parsed (interleaved thinking)
        // The leading " " before <think> is normal-text prefix; " final" is suffix.
        let result3 = parser
            .parse_reasoning_streaming_incremental(" <think>new reasoning</think> final", &[]);
        assert_eq!(result3.reasoning_text, "new reasoning");
        assert_eq!(result3.normal_text, "  final"); // " " prefix + " final" suffix

        // Same test with separate chunks for clarity
        let mut parser2 =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser2.parse_reasoning_streaming_incremental("<think>first</think> normal", &[]);
        assert_eq!(r1.reasoning_text, "first");
        assert_eq!(r1.normal_text, " normal");

        let r2 = parser2.parse_reasoning_streaming_incremental(" between", &[]);
        assert_eq!(r2.normal_text, " between");
        assert_eq!(r2.reasoning_text, "");

        let r3 = parser2.parse_reasoning_streaming_incremental("<think>second</think> final", &[]);
        assert_eq!(r3.reasoning_text, "second");
        assert_eq!(r3.normal_text, " final");
    }

    #[test]
    fn test_post_reasoning_angle_bracket_not_buffered() {
        // After reasoning ends, a standalone `<` should pass through immediately
        // as normal text. It must NOT be buffered as a potential prefix of <think>
        // or </think>, because that would cause the downstream tool call jail to
        // miss the `<` (e.g., `<invoke` becomes `invoke`).
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        // Process a complete reasoning block
        let r1 =
            parser.parse_reasoning_streaming_incremental("<think>reasoning content</think>", &[]);
        assert_eq!(r1.reasoning_text, "reasoning content");
        assert_eq!(r1.normal_text, "");

        // After reasoning ends, a lone `<` must pass through as normal text
        let r2 = parser.parse_reasoning_streaming_incremental("<", &[]);
        assert_eq!(r2.normal_text, "<");
        assert_eq!(r2.reasoning_text, "");

        // The next token should arrive independently (not merged with buffered `<`)
        let r3 = parser.parse_reasoning_streaming_incremental("invoke name=\"get_weather\">", &[]);
        assert_eq!(r3.normal_text, "invoke name=\"get_weather\">");
        assert_eq!(r3.reasoning_text, "");
    }

    #[test]
    fn test_post_reasoning_tool_call_xml_preserved() {
        // Simulates the MiniMax tool call scenario: reasoning followed by XML tool call.
        // The `<` in `<invoke` must not be consumed by the reasoning parser.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>let me check", &[]);
        assert_eq!(r1.reasoning_text, "let me check");

        let r2 = parser.parse_reasoning_streaming_incremental("</think>", &[]);
        assert_eq!(r2.normal_text, "");
        assert_eq!(r2.reasoning_text, "");

        // Tool call markers should pass through completely
        let r3 = parser.parse_reasoning_streaming_incremental("<minimax:tool_call>", &[]);
        assert_eq!(r3.normal_text, "<minimax:tool_call>");

        let r4 = parser.parse_reasoning_streaming_incremental("\n", &[]);
        assert_eq!(r4.normal_text, "\n");

        // `<` arriving as a separate token after reasoning must NOT be buffered
        let r5 = parser.parse_reasoning_streaming_incremental("<", &[]);
        assert_eq!(r5.normal_text, "<");

        let r6 = parser.parse_reasoning_streaming_incremental("invoke name=\"get_weather\">", &[]);
        assert_eq!(r6.normal_text, "invoke name=\"get_weather\">");
    }

    #[test]
    fn test_interleaved_streaming_across_chunks() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>thought 1</think>", &[]);
        assert_eq!(r1.reasoning_text, "thought 1");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental(" answer 1 ", &[]);
        assert_eq!(r2.normal_text, " answer 1 ");
        assert_eq!(r2.reasoning_text, "");

        let r3 = parser.parse_reasoning_streaming_incremental("<think>thought 2</think>", &[]);
        assert_eq!(r3.reasoning_text, "thought 2");
        assert_eq!(r3.normal_text, "");

        let r4 = parser.parse_reasoning_streaming_incremental(" answer 2", &[]);
        assert_eq!(r4.normal_text, " answer 2");
        assert_eq!(r4.reasoning_text, "");

        let r5 = parser.parse_reasoning_streaming_incremental("<think>thought 3</think>", &[]);
        assert_eq!(r5.reasoning_text, "thought 3");
        assert_eq!(r5.normal_text, "");

        let r6 = parser.parse_reasoning_streaming_incremental(" final answer", &[]);
        assert_eq!(r6.normal_text, " final answer");
        assert_eq!(r6.reasoning_text, "");
    }

    #[test]
    fn test_three_reasoning_blocks_non_streaming() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning(
            "<think>A</think> one <think>B</think> two <think>C</think> three",
            &[],
        );
        assert_eq!(result.reasoning_text, "ABC");
        assert_eq!(result.normal_text, "one  two  three");
    }

    #[test]
    fn test_streaming_transition_chunk() {
        // </think> and <think> arrive in the same chunk.
        // With loop-based processing, the second block's opening content is emitted
        // immediately (stream_reasoning=true) rather than buffered until the next call.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>first", &[]);
        assert_eq!(r1.reasoning_text, "first");

        // Mid-chunk transition: </think> then normal text then <think> with more content.
        // The loop transitions out of reasoning, emits " middle " as normal text, enters
        // the next reasoning block, and streams "second" immediately.
        let r2 = parser.parse_reasoning_streaming_incremental("</think> middle <think>second", &[]);
        assert_eq!(r2.reasoning_text, "second");
        assert_eq!(r2.normal_text, " middle ");

        // Continuation of second reasoning block
        let r3 = parser.parse_reasoning_streaming_incremental(" more</think> end", &[]);
        assert_eq!(r3.reasoning_text, " more");
        assert_eq!(r3.normal_text, " end");
    }

    #[test]
    fn test_interleaved_with_force_reasoning() {
        // deepseek_r1 mode: force_reasoning=true, first tokens are reasoning without <think>
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), true, true);

        // No <think> tag — treated as reasoning because force_reasoning=true
        let r1 = parser.parse_reasoning_streaming_incremental("initial reasoning", &[]);
        assert_eq!(r1.reasoning_text, "initial reasoning");
        assert_eq!(r1.normal_text, "");

        // End of forced reasoning block
        let r2 = parser.parse_reasoning_streaming_incremental("</think> answer", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, " answer");

        // Second reasoning block with explicit <think>
        let r3 =
            parser.parse_reasoning_streaming_incremental("<think>second thought</think> done", &[]);
        assert_eq!(r3.reasoning_text, "second thought");
        assert_eq!(r3.normal_text, " done");
    }

    #[test]
    fn test_interleaved_partial_think_tag_between_blocks() {
        // After first reasoning block, partial <think> tag arrives across chunks
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>first</think> normal", &[]);
        assert_eq!(r1.reasoning_text, "first");
        assert_eq!(r1.normal_text, " normal");

        // Partial <think> prefix: "<th" (2 chars, meets threshold)
        let r2 = parser.parse_reasoning_streaming_incremental("<th", &[]);
        assert_eq!(r2.normal_text, "");
        assert_eq!(r2.reasoning_text, "");

        // Complete the tag
        let r3 = parser.parse_reasoning_streaming_incremental("ink>second</think> end", &[]);
        assert_eq!(r3.reasoning_text, "second");
        assert_eq!(r3.normal_text, " end");
    }

    #[test]
    fn test_lone_angle_bracket_between_reasoning_blocks() {
        // A lone `<` between reasoning blocks should pass through (not buffer)
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>thought</think>", &[]);
        assert_eq!(r1.reasoning_text, "thought");

        // Lone `<` must not be buffered — could be a tool call
        let r2 = parser.parse_reasoning_streaming_incremental("<", &[]);
        assert_eq!(r2.normal_text, "<");
        assert_eq!(r2.reasoning_text, "");

        let r3 = parser.parse_reasoning_streaming_incremental("tool_call>", &[]);
        assert_eq!(r3.normal_text, "tool_call>");
        assert_eq!(r3.reasoning_text, "");

        // But a real <think> should still work after
        let r4 =
            parser.parse_reasoning_streaming_incremental("<think>more thought</think> done", &[]);
        assert_eq!(r4.reasoning_text, "more thought");
        assert_eq!(r4.normal_text, " done");
    }

    #[test]
    fn test_force_reasoning_stream_false_buffers_until_end_token() {
        // force_reasoning=true, stream_reasoning=false: content is buffered until </think>
        // arrives, then returned as a single chunk. This is the expected behavior.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), true, false);

        // No <think> — forced into reasoning, stream_reasoning=false means buffer silently
        let r1 = parser.parse_reasoning_streaming_incremental("chunk one", &[]);
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental(" chunk two", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");

        // </think> arrives — entire buffered reasoning is flushed
        let r3 = parser.parse_reasoning_streaming_incremental("</think> answer", &[]);
        assert_eq!(r3.reasoning_text, "chunk one chunk two");
        assert_eq!(r3.normal_text, " answer");
    }

    #[test]
    fn test_multiple_full_blocks_in_single_streaming_chunk() {
        // Two complete <think>...</think> blocks arrive in one chunk.
        // The loop exhausts all transitions in a single call — both blocks are fully
        // processed and no follow-up call is needed to flush buffered content.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental(
            "<think>A</think> mid <think>B</think> end",
            &[],
        );
        assert_eq!(r1.reasoning_text, "AB");
        assert_eq!(r1.normal_text, " mid  end");

        // Buffer is fully drained; empty follow-up returns nothing
        let r2 = parser.parse_reasoning_streaming_incremental("", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");
    }

    #[test]
    fn test_partial_end_token_stream_reasoning_true() {
        // Partial </think> split across chunks with stream_reasoning=true.
        // The partial-end-token buffer check only fires when the parser is ALREADY in
        // reasoning mode from a prior call. If <think> and </th arrive in the same chunk,
        // stream_reasoning=true emits the reasoning content immediately (including </th).
        // So <think> must arrive as its own chunk first.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>reasoning", &[]);
        assert_eq!(r1.reasoning_text, "reasoning");
        assert_eq!(r1.normal_text, "");

        // Partial end token while already in reasoning — buffered, nothing emitted
        let r2 = parser.parse_reasoning_streaming_incremental("</th", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");

        // Complete the end token
        let r3 = parser.parse_reasoning_streaming_incremental("ink> normal", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, " normal");
    }

    #[test]
    fn test_empty_string_input_various_states() {
        // Empty string input should always return empty results without changing state
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        // State: idle
        let r1 = parser.parse_reasoning_streaming_incremental("", &[]);
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");

        // Enter reasoning
        parser.parse_reasoning_streaming_incremental("<think>content", &[]);

        // State: in reasoning
        let r2 = parser.parse_reasoning_streaming_incremental("", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");

        // Complete and exit reasoning
        parser.parse_reasoning_streaming_incremental("</think>", &[]);

        // State: post-reasoning (normal text)
        let r3 = parser.parse_reasoning_streaming_incremental("", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "");
    }

    #[test]
    fn test_force_reasoning_stream_false_multiple_blocks() {
        // force_reasoning=true (deepseek_r1 mode), stream_reasoning=false.
        // First block uses forced-reasoning (no explicit <think>); subsequent blocks
        // use explicit tags.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), true, false);

        // Forced reasoning without open tag, flushed on </think>
        let r1 =
            parser.parse_reasoning_streaming_incremental("initial reasoning</think> normal1 ", &[]);
        assert_eq!(r1.reasoning_text, "initial reasoning");
        assert_eq!(r1.normal_text, " normal1 ");

        // Subsequent explicit <think> block works correctly
        let r2 = parser
            .parse_reasoning_streaming_incremental("<think>second block</think> normal2", &[]);
        assert_eq!(r2.reasoning_text, "second block");
        assert_eq!(r2.normal_text, " normal2");
    }

    #[test]
    fn test_glm5_pattern_a_burst_single_chunk() {
        // GLM-5 Pattern A: the entire completion arrives in one SSE event.
        // Format: <think>T1</think><tool_call>A</tool_call><think>T2</think><tool_call>B</tool_call>
        //
        // Both reasoning blocks must be extracted into reasoning_text; both tool calls
        // must land in normal_text for the downstream tool call parser. No follow-up
        // call should be needed — the loop fully drains the buffer in a single call.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental(
            "<think>T1</think><tool_call>A</tool_call><think>T2</think><tool_call>B</tool_call>",
            &[],
        );
        assert_eq!(r1.reasoning_text, "T1T2");
        assert_eq!(
            r1.normal_text,
            "<tool_call>A</tool_call><tool_call>B</tool_call>"
        );

        // Buffer is fully drained; stream can end here with no content loss
        let r2 = parser.parse_reasoning_streaming_incremental("", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");
    }

    #[test]
    fn test_tool_call_xml_between_reasoning_blocks_streaming() {
        // GLM-5 Pattern A chunk-by-chunk: verifies that tool call XML between reasoning
        // blocks lands in normal_text, not reasoning_text, across separate SSE events.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>T1</think>", &[]);
        assert_eq!(r1.reasoning_text, "T1");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("<tool_call>A</tool_call>", &[]);
        assert_eq!(r2.normal_text, "<tool_call>A</tool_call>");
        assert_eq!(r2.reasoning_text, "");

        let r3 = parser.parse_reasoning_streaming_incremental("<think>T2</think>", &[]);
        assert_eq!(r3.reasoning_text, "T2");
        assert_eq!(r3.normal_text, "");

        let r4 = parser.parse_reasoning_streaming_incremental("<tool_call>B</tool_call>", &[]);
        assert_eq!(r4.normal_text, "<tool_call>B</tool_call>");
        assert_eq!(r4.reasoning_text, "");
    }

    // =========================================================================
    // Mid-string partial tag tests (overlap-based buffering)
    //
    // These test scenarios where a <think> or </think> tag is split mid-string
    // (not at the start of the buffer). Backends that batch multiple forward-pass
    // tokens into a single chunked response can produce these patterns.
    //
    // Ported from PR #6448 (ryanolson) with additional fakeout tests.
    // =========================================================================

    #[test]
    fn test_mid_string_partial_opening_tag_batched() {
        // Backend batches tokens: "Hello world <th" arrives as one chunk
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("Hello world <th", &[]);
        // "Hello world " emitted as normal, "<th" held in buffer
        assert_eq!(r1.normal_text, "Hello world ");
        assert_eq!(r1.reasoning_text, "");

        let r2 = parser
            .parse_reasoning_streaming_incremental("ink>reasoning content</think> answer", &[]);
        assert_eq!(r2.reasoning_text, "reasoning content");
        assert_eq!(r2.normal_text, " answer");
    }

    #[test]
    fn test_batched_tag_boundary_split() {
        // Aggressive batching: <think> tag split with normal text prefix
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("The answer is <thi", &[]);
        assert_eq!(r1.normal_text, "The answer is ");
        assert_eq!(r1.reasoning_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("nk>let me think</think>42", &[]);
        assert_eq!(r2.reasoning_text, "let me think");
        assert_eq!(r2.normal_text, "42");
    }

    #[test]
    fn test_mid_string_partial_closing_tag_stream_reasoning_false() {
        // With stream_reasoning=false, content stays buffered until </think>.
        // Partial </think> split mid-string while in reasoning mode.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, false);

        let r1 =
            parser.parse_reasoning_streaming_incremental("<think>reasoning content and </th", &[]);
        assert_eq!(r1.normal_text, "");
        assert_eq!(r1.reasoning_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("ink> normal text", &[]);
        assert_eq!(r2.reasoning_text, "reasoning content and ");
        assert_eq!(r2.normal_text, " normal text");
    }

    #[test]
    fn test_mid_string_partial_closing_tag_stream_reasoning_true() {
        // With stream_reasoning=true, reasoning content is emitted incrementally.
        // The partial "</th" at the end must NOT be emitted as reasoning text.
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 =
            parser.parse_reasoning_streaming_incremental("<think>reasoning content and </th", &[]);
        // "reasoning content and " emitted as reasoning, "</th" held
        assert_eq!(r1.reasoning_text, "reasoning content and ");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("ink> normal text", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, " normal text");
    }

    #[test]
    fn test_batched_interleaved_with_mid_string_partial() {
        // First block complete in chunk 1, second block's <think> split at boundary
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 =
            parser.parse_reasoning_streaming_incremental("<think>thought1</think>answer1<thi", &[]);
        assert_eq!(r1.reasoning_text, "thought1");
        assert_eq!(r1.normal_text, "answer1");

        let r2 = parser.parse_reasoning_streaming_incremental("nk>thought2</think>answer2", &[]);
        assert_eq!(r2.reasoning_text, "thought2");
        assert_eq!(r2.normal_text, "answer2");
    }

    #[test]
    fn test_partial_tag_false_positive() {
        // "<th" looks like partial <think> but "thesis" is not <think>
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("value <thesis on", &[]);
        // No suffix of "value <thesis on" is a prefix of "<think>" — all emitted
        let r2 = parser.parse_reasoning_streaming_incremental(" AI> is great", &[]);

        let combined_normal = format!("{}{}", r1.normal_text, r2.normal_text);
        assert_eq!(combined_normal, "value <thesis on AI> is great");
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r2.reasoning_text, "");
    }

    #[test]
    fn test_partial_closing_tag_fakeout() {
        // Ollama-style fakeout: "</th" buffered, but "ing>" completes "</thing>" not "</think>"
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);

        let r1 = parser.parse_reasoning_streaming_incremental("<think>abc</th", &[]);
        assert_eq!(r1.reasoning_text, "abc");
        assert_eq!(r1.normal_text, "");

        // "ing>def" completes the partial as "</thing>def" — not a closing tag
        let r2 = parser.parse_reasoning_streaming_incremental("ing>def", &[]);
        assert_eq!(r2.reasoning_text, "</thing>def");
        assert_eq!(r2.normal_text, "");

        // Real closing tag arrives
        let r3 = parser.parse_reasoning_streaming_incremental("</think>done", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "done");
    }

    #[test]
    fn test_overlap_helper_function() {
        // Direct tests for the overlap utility
        assert_eq!(overlap("abc</th", "</think>"), 4);
        assert_eq!(overlap("abc</thing>def", "</think>"), 0);
        assert_eq!(overlap("<", "<think>"), 1);
        assert_eq!(overlap("<th", "<think>"), 3);
        assert_eq!(overlap("<think>", "<think>"), 7); // full match
        assert_eq!(overlap("no match", "<think>"), 0);
        assert_eq!(overlap("", "<think>"), 0);
        assert_eq!(overlap("Hello world <thi", "<think>"), 4);
        // Multi-byte delimiters (Kimi parser uses ◁think▷ / ◁/think▷)
        assert_eq!(overlap("text◁", "◁think▷"), 3); // ◁ is 3 bytes
        assert_eq!(overlap("text◁th", "◁think▷"), 5);
        assert_eq!(overlap("text◁/thi", "◁/think▷"), 7);
        assert_eq!(overlap("no match", "◁think▷"), 0);
    }
}

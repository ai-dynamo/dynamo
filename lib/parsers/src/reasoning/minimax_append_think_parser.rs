// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ParserResult, ReasoningParser};

/// MiniMax Append-Think Reasoning Parser.
///
/// The MiniMax model starts generating reasoning content immediately WITHOUT
/// emitting a `<think>` opener in its output. SGLang's `MiniMaxAppendThinkDetector`
/// and vLLM's `MiniMaxM2AppendThinkReasoningParser` both handle this by simply
/// prepending `<think>` to the emitted text and classifying the whole stream
/// as `normal_text`/content — neither extracts reasoning based on a `</think>`
/// marker. The tag is left inline for downstream consumers that want to render
/// or post-process it.
///
/// This parser matches those upstream implementations verbatim: a pass-through
/// with a one-time `<think>` prefix on the first streamed chunk. Reasoning
/// content is never populated.
///
/// References:
/// - SGLang MiniMaxAppendThinkDetector:
///   <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/reasoning_parser.py>
/// - vLLM MiniMaxM2AppendThinkReasoningParser:
///   <https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/minimax_m2_reasoning_parser.py>
#[derive(Debug, Default)]
pub struct MiniMaxAppendThinkParser {
    /// Flips to true after the first streamed chunk has received the `<think>`
    /// prefix so subsequent chunks pass through unchanged.
    prefix_emitted: bool,
}

impl MiniMaxAppendThinkParser {
    pub fn new() -> Self {
        Self::default()
    }
}

const THINK_START_TOKEN: &str = "<think>";

impl ReasoningParser for MiniMaxAppendThinkParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        // Non-streaming: return the full text with a single `<think>` prefix,
        // all as normal_text.  Reasoning extraction is intentionally a no-op.
        ParserResult {
            normal_text: format!("{THINK_START_TOKEN}{text}"),
            reasoning_text: String::new(),
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        let normal_text = if !self.prefix_emitted {
            self.prefix_emitted = true;
            format!("{THINK_START_TOKEN}{text}")
        } else {
            text.to_string()
        };
        ParserResult {
            normal_text,
            reasoning_text: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] // CASE.10 — minimax inline-reasoning
    fn test_detect_and_parse_prepends_think_all_as_normal_text() {
        let mut parser = MiniMaxAppendThinkParser::new();
        let result = parser.detect_and_parse_reasoning("reasoning content here", &[]);
        // Matches SGLang: everything is normal_text with a `<think>` prefix.
        assert_eq!(result.normal_text, "<think>reasoning content here");
        assert_eq!(result.reasoning_text, "");
    }

    #[test] // CASE.10 — minimax inline-reasoning
    fn test_detect_and_parse_with_end_token_is_still_normal_text() {
        let mut parser = MiniMaxAppendThinkParser::new();
        let result =
            parser.detect_and_parse_reasoning("reasoning content</think>normal response", &[]);
        // SGLang does not split on `</think>` — the whole string (with the
        // prepended `<think>`) flows through as normal_text.
        assert_eq!(
            result.normal_text,
            "<think>reasoning content</think>normal response"
        );
        assert_eq!(result.reasoning_text, "");
    }

    #[test] // CASE.8, CASE.10
    fn test_streaming_first_chunk_gets_prefix_rest_pass_through() {
        let mut parser = MiniMaxAppendThinkParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("I need to ", &[]);
        assert_eq!(r1.normal_text, "<think>I need to ");
        assert_eq!(r1.reasoning_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("check the weather", &[]);
        assert_eq!(r2.normal_text, "check the weather");
        assert_eq!(r2.reasoning_text, "");

        let r3 = parser.parse_reasoning_streaming_incremental("</think>The weather is sunny.", &[]);
        // No split — `</think>` passes through verbatim in normal_text.
        assert_eq!(r3.normal_text, "</think>The weather is sunny.");
        assert_eq!(r3.reasoning_text, "");
    }

    #[test] // CASE.13 — minimax leaves tool-call shape inline
    fn test_streaming_bare_json_tool_call_is_normal_text() {
        // Regression: under SGLang guided decoding the model emits a bare
        // JSON array with no `</think>`. The parser must not capture it as
        // reasoning — it must pass through so the tool-call jail can extract
        // it into structured tool_calls.
        let mut parser = MiniMaxAppendThinkParser::new();
        let r = parser.parse_reasoning_streaming_incremental(
            r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#,
            &[],
        );
        assert_eq!(
            r.normal_text,
            r#"<think>[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#
        );
        assert_eq!(r.reasoning_text, "");
    }

    #[test] // CASE.9, CASE.13 — minimax inline-reasoning
    fn test_streaming_tool_call_after_reasoning_is_all_normal_text() {
        let mut parser = MiniMaxAppendThinkParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("let me call a tool", &[]);
        assert_eq!(r1.normal_text, "<think>let me call a tool");

        let r2 = parser.parse_reasoning_streaming_incremental(
            "</think><minimax:tool_call><invoke name=\"get_weather\">",
            &[],
        );
        // Entire chunk is normal_text — `</think>` is not consumed.
        assert_eq!(
            r2.normal_text,
            "</think><minimax:tool_call><invoke name=\"get_weather\">"
        );
        assert_eq!(r2.reasoning_text, "");
    }
}

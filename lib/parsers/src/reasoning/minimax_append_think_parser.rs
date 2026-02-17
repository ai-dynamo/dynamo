// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ParserResult, ReasoningParser};

use super::base_parser::BasicReasoningParser;

/// MiniMax Append-Think Reasoning Parser.
///
/// The MiniMax model starts generating reasoning content immediately WITHOUT
/// a `<think>` prefix. The model output looks like:
///   `reasoning content here...</think>actual response`
///
/// This parser prepends `<think>` to the first chunk, transforming the stream into:
///   `<think>reasoning content here...</think>actual response`
///
/// It then delegates to `BasicReasoningParser` for standard `<think>...</think>`
/// extraction, splitting output into `reasoning_text` and `normal_text`.
///
/// Reference: SGLang MiniMaxAppendThinkDetector
/// https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/reasoning_parser.py
#[derive(Debug)]
pub struct MiniMaxAppendThinkParser {
    inner: BasicReasoningParser,
    is_first_chunk: bool,
}

impl Default for MiniMaxAppendThinkParser {
    fn default() -> Self {
        Self {
            inner: BasicReasoningParser::new(
                "<think>".into(),
                "</think>".into(),
                false, // force_reasoning=false; we synthesize <think> ourselves
                true,  // stream_reasoning=true
            ),
            is_first_chunk: true,
        }
    }
}

impl MiniMaxAppendThinkParser {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ReasoningParser for MiniMaxAppendThinkParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        // Prepend <think> and delegate to the inner parser
        let augmented = format!("<think>{}", text);
        self.inner.detect_and_parse_reasoning(&augmented, token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        if self.is_first_chunk {
            self.is_first_chunk = false;
            let augmented = format!("<think>{}", text);
            self.inner
                .parse_reasoning_streaming_incremental(&augmented, token_ids)
        } else {
            self.inner
                .parse_reasoning_streaming_incremental(text, token_ids)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_and_parse_no_end_token() {
        let mut parser = MiniMaxAppendThinkParser::new();
        let result = parser.detect_and_parse_reasoning("reasoning content here", &[]);
        assert_eq!(result.reasoning_text, "reasoning content here");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_detect_and_parse_with_end_token() {
        let mut parser = MiniMaxAppendThinkParser::new();
        let result =
            parser.detect_and_parse_reasoning("reasoning content</think>normal response", &[]);
        assert_eq!(result.reasoning_text, "reasoning content");
        assert_eq!(result.normal_text, "normal response");
    }

    #[test]
    fn test_streaming_basic_flow() {
        let mut parser = MiniMaxAppendThinkParser::new();

        // First chunk: model starts reasoning without <think>
        let r1 = parser.parse_reasoning_streaming_incremental("I need to ", &[]);
        assert_eq!(r1.reasoning_text, "I need to ");
        assert_eq!(r1.normal_text, "");

        // Middle chunk: still reasoning
        let r2 = parser.parse_reasoning_streaming_incremental("check the weather", &[]);
        assert_eq!(r2.reasoning_text, "check the weather");
        assert_eq!(r2.normal_text, "");

        // End of reasoning
        let r3 = parser.parse_reasoning_streaming_incremental("</think>The weather is sunny.", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "The weather is sunny.");
    }

    #[test]
    fn test_streaming_end_token_split_across_chunks() {
        let mut parser = MiniMaxAppendThinkParser::new();

        // With stream_reasoning=true, reasoning is emitted immediately
        let r1 = parser.parse_reasoning_streaming_incremental("reasoning", &[]);
        assert_eq!(r1.reasoning_text, "reasoning");
        assert_eq!(r1.normal_text, "");

        // </think> split across chunks - partial match should buffer
        let r2 = parser.parse_reasoning_streaming_incremental("</thi", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");

        // Complete the end token - reasoning already streamed in r1,
        // so r3 only contains the normal text after </think>
        let r3 = parser.parse_reasoning_streaming_incremental("nk>normal text", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "normal text");
    }

    #[test]
    fn test_streaming_only_reasoning_no_end() {
        let mut parser = MiniMaxAppendThinkParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("still thinking", &[]);
        assert_eq!(r1.reasoning_text, "still thinking");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental(" more thought", &[]);
        assert_eq!(r2.reasoning_text, " more thought");
        assert_eq!(r2.normal_text, "");
    }

    #[test]
    fn test_streaming_with_tool_call_after_reasoning() {
        let mut parser = MiniMaxAppendThinkParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("let me call a tool", &[]);
        assert_eq!(r1.reasoning_text, "let me call a tool");

        let r2 = parser.parse_reasoning_streaming_incremental(
            "</think><minimax:tool_call><invoke name=\"get_weather\">",
            &[],
        );
        assert_eq!(r2.reasoning_text, "");
        assert!(
            r2.normal_text
                .contains("<minimax:tool_call><invoke name=\"get_weather\">")
        );
    }

    #[test]
    fn test_streaming_tool_call_angle_bracket_split_tokens() {
        // Reproduces the bug where `<` before `<invoke` is consumed by the
        // reasoning parser's prefix matching after reasoning ends.
        let mut parser = MiniMaxAppendThinkParser::new();

        // Reasoning phase
        let r1 = parser.parse_reasoning_streaming_incremental("let me check the weather", &[]);
        assert_eq!(r1.reasoning_text, "let me check the weather");

        // End reasoning
        let r2 = parser.parse_reasoning_streaming_incremental("</think>", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");

        // Tool call start marker
        let r3 = parser.parse_reasoning_streaming_incremental("<minimax:tool_call>", &[]);
        assert_eq!(r3.normal_text, "<minimax:tool_call>");

        // Newline
        let r4 = parser.parse_reasoning_streaming_incremental("\n", &[]);
        assert_eq!(r4.normal_text, "\n");

        // `<` as a separate token must NOT be buffered after reasoning ends
        let r5 = parser.parse_reasoning_streaming_incremental("<", &[]);
        assert_eq!(r5.normal_text, "<");

        // Rest of the invoke tag
        let r6 = parser.parse_reasoning_streaming_incremental("invoke name=\"get_weather\">", &[]);
        assert_eq!(r6.normal_text, "invoke name=\"get_weather\">");
    }
}

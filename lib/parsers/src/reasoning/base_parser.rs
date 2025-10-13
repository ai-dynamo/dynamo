// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ParserResult, ReasoningParser};

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
        let in_reasoning = self._in_reasoning || text.contains(&self.think_start_token);
        if !in_reasoning {
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: String::new(),
            };
        }

        // The text is considered to be in a reasoning block.
        let processed_text = text.replace(&self.think_start_token, "").trim().to_string();

        if !processed_text.contains(&self.think_end_token) {
            // Assume reasoning was truncated before `think_end_token`
            return ParserResult {
                normal_text: String::new(),
                reasoning_text: processed_text,
            };
        }

        // Extract reasoning content
        let splits: Vec<&str> = processed_text.splitn(2, &self.think_end_token).collect();
        let reasoning_text = splits.first().unwrap_or(&"").to_string();
        let normal_text = splits
            .get(1)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

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
        // Incrementally parse the streaming text
        self._buffer.push_str(text);
        let mut current_text = self._buffer.to_string();
        // If the current text is a prefix of the think token, keep buffering

        if self.think_start_token.starts_with(&current_text)
            && self.think_start_token.as_str() != current_text.as_str()
        {
            return ParserResult {
                normal_text: String::new(),
                reasoning_text: String::new(),
            };
        }
        if self.think_end_token.starts_with(&current_text)
            && self.think_end_token.as_str() != current_text.as_str()
        {
            return ParserResult {
                normal_text: String::new(),
                reasoning_text: String::new(),
            };
        }

        // Strip `<think>` token if present
        if !self.stripped_think_start && current_text.contains(&self.think_start_token) {
            current_text = current_text.replace(&self.think_start_token, "");
            self._buffer = current_text.to_string();
            self.stripped_think_start = true;
            self._in_reasoning = true;
        }
        // Handle end of reasoning block
        let mut think_end_idx = current_text.len();
        if self._in_reasoning {
            think_end_idx = current_text
                .find(&self.think_end_token)
                .unwrap_or(current_text.len());
        }
        if self._in_reasoning && think_end_idx < current_text.len() {
            let reasoning_text = &current_text[..think_end_idx];
            self._buffer.clear();
            self._in_reasoning = false;
            let start_idx = think_end_idx + self.think_end_token.len();
            let normal_text = if start_idx < current_text.len() {
                &current_text[start_idx..]
            } else {
                ""
            };
            return ParserResult {
                normal_text: normal_text.to_string(),
                reasoning_text: reasoning_text.to_string(),
            };
        }
        // Continue with reasoning content
        if self._in_reasoning && self.stream_reasoning {
            // Stream the content immediately
            let reasoning_text = current_text;
            self._buffer.clear();
            ParserResult {
                normal_text: String::new(),
                reasoning_text,
            }
        } else if !self._in_reasoning {
            // If we're not in a reasoning block return as normal text
            let normal_text = current_text;
            self._buffer.clear();
            ParserResult {
                normal_text,
                reasoning_text: String::new(),
            }
        } else {
            // If we are in a reasoning block but no end token is found, return the current buffer
            ParserResult {
                normal_text: String::new(),
                reasoning_text: String::new(),
            }
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
        // The current implementation only handles the first occurrence properly
        assert_eq!(result.normal_text, "middle second reasoning</think> end");
        assert_eq!(result.reasoning_text, "first reasoning");
    }

    #[test]
    fn test_streaming_multiple_reasoning_blocks() {
        let mut parser =
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, false);
        let result1 = parser
            .parse_reasoning_streaming_incremental("<think>first reasoning</think> middle", &[]);
        assert_eq!(result1.normal_text, " middle");
        assert_eq!(result1.reasoning_text, "first reasoning");

        // Basic parser assumes only one reasoning block at a time
        let result2 = parser
            .parse_reasoning_streaming_incremental(" <think>second reasoning</think> end", &[]);
        assert_eq!(result2.normal_text, " <think>second reasoning</think> end");
        assert_eq!(result2.reasoning_text, "");
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
        // Current implementation should handle this by finding the first closing tag
        assert_eq!(result.normal_text, "reasoning</think> normal");
        // All <think> tags are stripped, so <think>inner is not included
        assert_eq!(result.reasoning_text, "outer inner");
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
        // Should handle by replacing all opening tags and using first closing tag
        assert_eq!(result.normal_text, "normal");
        assert_eq!(result.reasoning_text, "first second reasoning");
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
            BasicReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, false);

        // Process complete reasoning block
        let result1 =
            parser.parse_reasoning_streaming_incremental("<think>reasoning</think> normal", &[]);
        assert_eq!(result1.normal_text, " normal");
        assert_eq!(result1.reasoning_text, "reasoning");

        // Process normal text - should not be affected by previous state
        let result2 = parser.parse_reasoning_streaming_incremental(" more normal text", &[]);
        assert_eq!(result2.normal_text, " more normal text");
        assert_eq!(result2.reasoning_text, "");

        // Basic parser does not expect more than one reasoning block at a time
        // So this should not affect the state
        let result3 = parser
            .parse_reasoning_streaming_incremental(" <think>new reasoning</think> final", &[]);
        assert_eq!(result3.normal_text, " <think>new reasoning</think> final");
        assert_eq!(result3.reasoning_text, "");
    }
}

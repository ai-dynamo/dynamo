// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::ParserResult;
use crate::ReasoningParser;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraniteReasoningParser {
    think_start_tokens: Vec<String>,
    think_end_tokens: Vec<String>,
    buffer: String,
    stripped_think_start: bool,
    in_reasoning: bool,
}

impl GraniteReasoningParser {
    pub fn new() -> Self {
        Self {
            think_start_tokens: ["Here's my thought process:", "Here is my thought process:"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            think_end_tokens: ["Here's my response:", "Here is my response:"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            buffer: String::new(),
            stripped_think_start: false,
            in_reasoning: false,
        }
    }
}

impl Default for GraniteReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for GraniteReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _: &[u32]) -> ParserResult {
        let think_start_token = self
            .think_start_tokens
            .iter()
            .find(|&token| text.contains(token))
            .unwrap_or_else(|| self.think_start_tokens.first().unwrap());

        let think_end_token = self
            .think_end_tokens
            .iter()
            .find(|&token| text.contains(token))
            .unwrap_or_else(|| self.think_end_tokens.first().unwrap());
        // Implement parsing logic specific to Granite format
        let in_reasoning = self.in_reasoning
            || self
                .think_start_tokens
                .iter()
                .any(|token| text.contains(token));
        if !in_reasoning {
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: String::new(),
            };
        }

        // The text is considered to be in a reasoning block.
        let processed_text = text.replacen(think_start_token, "", 1).trim().to_string();

        if !processed_text.contains(think_end_token) {
            // Assume reasoning was truncated before `think_end_token`
            return ParserResult {
                normal_text: String::new(),
                reasoning_text: processed_text,
            };
        }

        // Extract reasoning content
        let splits: Vec<&str> = processed_text.splitn(2, think_end_token).collect();
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

    fn parse_reasoning_streaming_incremental(&mut self, text: &str, _: &[u32]) -> ParserResult {
        // Implement streaming parsing logic specific to Granite format

        // Incrementally parse the streaming text
        self.buffer.push_str(text);
        let mut current_text = self.buffer.to_string();
        // If the current text is a prefix of the think token, keep buffering

        for think_start_token in &self.think_start_tokens {
            if think_start_token.starts_with(&current_text)
                && think_start_token.as_str() != current_text.as_str()
            {
                return ParserResult {
                    normal_text: String::new(),
                    reasoning_text: String::new(),
                };
            }
        }
        for think_end_token in &self.think_end_tokens {
            if think_end_token.starts_with(&current_text)
                && think_end_token.as_str() != current_text.as_str()
            {
                return ParserResult {
                    normal_text: String::new(),
                    reasoning_text: String::new(),
                };
            }
        }

        let think_start_token = self
            .think_start_tokens
            .iter()
            .find(|&token| current_text.contains(token))
            .unwrap_or_else(|| self.think_start_tokens.first().unwrap());

        let think_end_token = self
            .think_end_tokens
            .iter()
            .find(|&token| current_text.contains(token))
            .unwrap_or_else(|| self.think_end_tokens.first().unwrap());

        if !self.stripped_think_start && current_text.contains(think_start_token) {
            current_text = current_text.replacen(think_start_token, "", 1);
            self.buffer = current_text.to_string();
            self.stripped_think_start = true;
            self.in_reasoning = true;
        }
        // Handle end of reasoning block
        let mut think_end_idx = current_text.len();
        if self.in_reasoning {
            think_end_idx = current_text
                .find(think_end_token)
                .unwrap_or(current_text.len());
        }
        if self.in_reasoning && think_end_idx < current_text.len() {
            let reasoning_text = &current_text[..think_end_idx];
            self.buffer.clear();
            self.in_reasoning = false;
            let start_idx = think_end_idx + think_end_token.len();
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
        if self.in_reasoning {
            // Stream the content immediately
            let reasoning_text = current_text;
            self.buffer.clear();
            ParserResult {
                normal_text: String::new(),
                reasoning_text,
            }
        } else {
            // If we're not in a reasoning block return as normal text
            let normal_text = current_text;
            self.buffer.clear();
            ParserResult {
                normal_text,
                reasoning_text: String::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_reasoning_detection() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: I need to think about this. Here's my response: The answer is 42.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, " I need to think about this. ");
        assert_eq!(result.normal_text, " The answer is 42.");
    }

    #[test]
    fn test_alternative_start_token() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here is my thought process: Different thinking here. Here is my response: Final answer.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, " Different thinking here. ");
        assert_eq!(result.normal_text, " Final answer.");
    }

    #[test]
    fn test_streaming_partial_tokens() {
        let mut parser = GraniteReasoningParser::new();

        // Test partial start token
        let result1 = parser.parse_reasoning_streaming_incremental("Here's", &[]);
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "");

        // Complete the start token and add reasoning
        let result2 = parser
            .parse_reasoning_streaming_incremental(" my thought process: This is reasoning", &[]);
        assert_eq!(result2.reasoning_text, " This is reasoning");
        assert_eq!(result2.normal_text, "");
    }

    #[test]
    fn test_streaming_partial_end_tokens() {
        let mut parser = GraniteReasoningParser::new();

        // Start reasoning
        parser
            .parse_reasoning_streaming_incremental("Here's my thought process: Thinking... ", &[]);

        parser.parse_reasoning_streaming_incremental("Here", &[]);

        // Partial end token should buffer
        let result = parser.parse_reasoning_streaming_incremental("'s my", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "");

        // Complete end token
        let result2 = parser.parse_reasoning_streaming_incremental(" response: Done!", &[]);
        assert_eq!(result2.reasoning_text, "");
        assert_eq!(result2.normal_text, " Done!");
    }

    #[test]
    fn test_no_reasoning_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "This is just normal text without any special tokens.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_only_start_token_no_end() {
        let mut parser = GraniteReasoningParser::new();

        let result1 = parser.parse_reasoning_streaming_incremental(
            "Here's my thought process: This is reasoning content",
            &[],
        );
        assert_eq!(result1.reasoning_text, " This is reasoning content");
        assert_eq!(result1.normal_text, "");

        // More reasoning content without end token
        let result2 = parser.parse_reasoning_streaming_incremental(" and more thinking", &[]);
        assert_eq!(result2.reasoning_text, " and more thinking");
        assert_eq!(result2.normal_text, "");
    }

    #[test]
    fn test_empty_reasoning_block() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:Here's my response: Direct answer.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, " Direct answer.");
    }

    #[test]
    fn test_reasoning_with_whitespace() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:   \n  Indented reasoning  \n  Here's my response:   Final result  ";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, "   \n  Indented reasoning  \n  ");
        assert_eq!(result.normal_text, "   Final result  ");
    }

    #[test]
    fn test_case_sensitive_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "here's my thought process: lowercase. here's my response: answer.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        // Should not detect lowercase tokens
        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_nested_or_repeated_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: I think Here's my thought process: is confusing. Here's my response: Done.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(
            result.reasoning_text,
            " I think Here's my thought process: is confusing. "
        );
        assert_eq!(result.normal_text, " Done.");
    }

    #[test]
    fn test_detect_and_parse_reasoning_basic() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: I need to analyze this problem. Here's my response: The solution is clear.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "I need to analyze this problem. ");
        assert_eq!(result.normal_text, "The solution is clear.");
    }

    #[test]
    fn test_detect_and_parse_reasoning_alternative_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here is my thought process: Different reasoning approach. Here is my response: Final conclusion.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "Different reasoning approach. ");
        assert_eq!(result.normal_text, "Final conclusion.");
    }

    #[test]
    fn test_detect_and_parse_reasoning_no_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "This is just normal text without special markers.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_detect_and_parse_reasoning_only_start_token() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: This reasoning has no end marker.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "This reasoning has no end marker.");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_detect_and_parse_reasoning_empty_sections() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:Here's my response:";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_detect_and_parse_reasoning_whitespace_handling() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:   \n\tSpaced reasoning\n   Here's my response:  \n  Spaced response\n";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "Spaced reasoning\n   ");
        assert_eq!(result.normal_text, "Spaced response");
    }

    #[test]
    fn test_detect_and_parse_reasoning_multiple_end_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: Thinking about Here's my response: in the middle. Here's my response: Real end.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "Thinking about ");
        assert_eq!(
            result.normal_text,
            "in the middle. Here's my response: Real end."
        );
    }

    #[test]
    fn test_detect_and_parse_reasoning_case_sensitivity() {
        let mut parser = GraniteReasoningParser::new();
        let text =
            "here's my thought process: lowercase tokens. here's my response: should not work.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_detect_and_parse_reasoning_mixed_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: First reasoning. Here is my response: Mixed token response.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "First reasoning. ");
        assert_eq!(result.normal_text, "Mixed token response.");
    }

    #[test]
    fn test_detect_and_parse_reasoning_long_content() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: This is a very long reasoning section that spans multiple sentences. I need to consider various factors. The analysis requires careful thought. Here's my response: After all that thinking, here is the comprehensive answer with multiple parts and detailed explanation.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(
            result.reasoning_text,
            "This is a very long reasoning section that spans multiple sentences. I need to consider various factors. The analysis requires careful thought. "
        );
        assert_eq!(
            result.normal_text,
            "After all that thinking, here is the comprehensive answer with multiple parts and detailed explanation."
        );
    }
}

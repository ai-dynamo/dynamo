// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::postprocessor::reasoning_parser::{ParserResult, ReasoningParser};

#[derive(Default)]
pub struct BaseReasoningParser {
    think_start_token: String,
    think_end_token: String,
    _in_reasoning: bool,
    stream_reasoning: bool,
    _buffer: String,
    stripped_think_start: bool,
}

impl BaseReasoningParser {
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

impl ReasoningParser for BaseReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> ParserResult {
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

    fn parse_reasoning_streaming_incremental(&mut self, text: &str) -> ParserResult {
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
            self.stripped_think_start = true;
            self._in_reasoning = true;
        }
        // Handle end of reasoning block
        if self._in_reasoning && current_text.contains(&self.think_end_token) {
            let end_idx = current_text
                .find(&self.think_end_token)
                .unwrap_or(current_text.len());
            let reasoning_text = &current_text[..end_idx];
            self._buffer.clear();
            self._in_reasoning = false;
            let start_idx = end_idx + self.think_end_token.len();
            let normal_text = if start_idx < current_text.len() {
                &current_text[start_idx..]
            } else {
                ""
            };
            return ParserResult {
                normal_text: normal_text.to_string(),
                reasoning_text: reasoning_text.trim().to_string(),
            };
        }
        // Continue with reasoning content
        if self._in_reasoning {
            if self.stream_reasoning {
                // Stream the content immediately
                let reasoning_text = self._buffer.clone();
                self._buffer.clear();
                return ParserResult {
                    normal_text: String::new(),
                    reasoning_text,
                };
            } else {
                return ParserResult {
                    normal_text: String::new(),
                    reasoning_text: String::new(),
                };
            }
        }
        // If we're not in a reasoning block return as normal text
        if !self._in_reasoning {
            let normal_text = self._buffer.clone();
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
    fn test_detect_and_parse_reasoning() {
        let mut parser =
            BaseReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result =
            parser.detect_and_parse_reasoning("<think>with reasoning</think> and more text.");
        assert_eq!(result.normal_text, "and more text.");
        assert_eq!(result.reasoning_text, "with reasoning");
    }
    #[test]
    fn test_detect_and_parse_reasoning_no_reasoning() {
        let mut parser =
            BaseReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("This is a test without reasoning.");
        assert_eq!(result.normal_text, "This is a test without reasoning.");
        assert_eq!(result.reasoning_text, "");
    }
    #[test]
    fn test_detect_and_parse_reasoning_truncated_reasoning() {
        let mut parser =
            BaseReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.detect_and_parse_reasoning("<think>with truncated reasoning");
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "with truncated reasoning");
    }

    #[test]
    fn test_parse_reasoning_streaming_incremental() {
        let mut parser =
            BaseReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser.parse_reasoning_streaming_incremental("<thi");
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_parse_reasoning_streaming_incremental_complete() {
        let mut parser =
            BaseReasoningParser::new("<think>".to_string(), "</think>".to_string(), false, true);
        let result = parser
            .parse_reasoning_streaming_incremental("<think>with reasoning</think> and more text.");
        assert_eq!(result.normal_text, " and more text.");
        assert_eq!(result.reasoning_text, "with reasoning");
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod base_parser;
mod deepseek_r1_parser;

// Re-export main types and functions for convenience
pub use base_parser::BasicReasoningParser;
pub use deepseek_r1_parser::DeepseekR1ReasoningParser;

pub struct ParserResult {
    /// The normal text outside of reasoning blocks.
    pub normal_text: String,

    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_text: String,
}

pub trait ReasoningParser {
    /// Detects and parses reasoning from the input text.
    fn detect_and_parse_reasoning(&mut self, text: &str) -> ParserResult;

    /// Parses reasoning incrementally from streaming input.
    fn parse_reasoning_streaming_incremental(&mut self, text: &str) -> ParserResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningParserType {
    DeepseekR1,
    Basic,
}

impl ReasoningParserType {
    pub fn get_reasoning_parser(self) -> Box<dyn ReasoningParser> {
        match self {
            ReasoningParserType::DeepseekR1 => Box::new(DeepseekR1ReasoningParser::new()),
            ReasoningParserType::Basic => Box::new(BasicReasoningParser::new(
                "<think>".to_string(),
                "</think>".to_string(),
                false,
                true,
            )),
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod base_parser;
mod deepseek_r1_parser;

// Re-export main types and functions for convenience
pub use base_parser::BasicReasoningParser;
pub use deepseek_r1_parser::DeepseekR1ReasoningParser;

#[derive(Debug, Clone, Default)]
pub struct ParserResult {
    /// The normal text outside of reasoning blocks.
    pub normal_text: String,

    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_text: String,
}

pub trait ReasoningParser {
    /// Parses a standalone, non-streaming input chunk. Implementations may reset or ignore
    /// internal streaming state and should return the split of normal vs reasoning text for
    /// this complete input. Marker tokens must not be included in either output.
    fn detect_and_parse_reasoning(&mut self, text: &str) -> ParserResult;

    /// Parses a streaming chunk and updates internal state. The return value should be the
    /// delta: only the newly discovered normal and reasoning text attributable to this chunk
    /// (not the cumulative totals). Marker tokens must not be included in either output.
    fn parse_reasoning_streaming_incremental(&mut self, text: &str) -> ParserResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningParserType {
    DeepseekR1,
    Basic,
}

impl ReasoningParserType {
    pub fn get_reasoning_parser(self) -> Box<dyn ReasoningParser + Send> {
        match self {
            ReasoningParserType::DeepseekR1 => Box::new(DeepseekR1ReasoningParser::new()),
            ReasoningParserType::Basic => Box::new(BasicReasoningParser::new(
                "<think>".into(),
                "</think>".into(),
                false,
                true,
            )),
        }
    }
}

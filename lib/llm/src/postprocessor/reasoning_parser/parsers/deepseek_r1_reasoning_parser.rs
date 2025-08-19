// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::postprocessor::reasoning_parser::parsers::base_reasoning_parser::BaseReasoningParser;

use crate::postprocessor::reasoning_parser::{ParserResult, ReasoningParser};

#[derive(Default)]
pub struct DeepseekR1ReasoningParser {
    base: BaseReasoningParser,
}

impl DeepseekR1ReasoningParser {
    pub fn new() -> Self {
        Self {
            base: BaseReasoningParser::new(
                "<think>".to_string(),
                "</think>".to_string(),
                true,
                true,
            ),
        }
    }
}

impl ReasoningParser for DeepseekR1ReasoningParser {
    fn parse_reasoning_streaming_incremental(&mut self, text: &str) -> ParserResult {
        self.base.parse_reasoning_streaming_incremental(text)
    }

    fn detect_and_parse_reasoning(&mut self, text: &str) -> ParserResult {
        self.base.detect_and_parse_reasoning(text)
    }
}

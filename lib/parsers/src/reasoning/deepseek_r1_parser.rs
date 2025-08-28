// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::base_parser::BasicReasoningParser;
use crate::ParserResult;
use crate::ReasoningParser;

#[derive(Default, Debug, Clone)]
pub struct DeepseekR1ReasoningParser {
    base: BasicReasoningParser,
}

impl DeepseekR1ReasoningParser {
    pub fn new() -> Self {
        Self {
            base: BasicReasoningParser::new(
                "<think>".to_string(),
                "</think>".to_string(),
                true,
                true,
            ),
        }
    }
}

impl ReasoningParser for DeepseekR1ReasoningParser {
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        self.base
            .parse_reasoning_streaming_incremental(text, token_ids)
    }

    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        self.base.detect_and_parse_reasoning(text, token_ids)
    }
}

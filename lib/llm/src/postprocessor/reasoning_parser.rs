// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod parsers;
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

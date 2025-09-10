// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::collections::HashMap;
use std::sync::OnceLock;

mod base_parser;
mod gpt_oss_parser;

// Re-export main types and functions for convenience
pub use base_parser::BasicReasoningParser;
pub use gpt_oss_parser::GptOssReasoningParser;

static REASONING_PARSER_MAP: OnceLock<HashMap<&'static str, ReasoningParserType>> = OnceLock::new();

/// Initialize the global reasoning parser map
fn get_reasoning_parser_map() -> &'static HashMap<&'static str, ReasoningParserType> {
    REASONING_PARSER_MAP.get_or_init(|| {
        let mut map = HashMap::new();
        map.insert("deepseek_r1", ReasoningParserType::DeepseekR1);
        map.insert("basic", ReasoningParserType::Basic);
        map.insert("gpt_oss", ReasoningParserType::GptOss);
        map.insert("qwen3", ReasoningParserType::Qwen);
        map.insert("nemotron_deci", ReasoningParserType::NemotronDeci);
        map.insert("kimi", ReasoningParserType::Kimi);
        map.insert("step3", ReasoningParserType::Step3);
        map.insert("mistral", ReasoningParserType::Mistral);
        map
    })
}

/// Get all available reasoning parser names
pub fn get_available_reasoning_parsers() -> Vec<&'static str> {
    get_reasoning_parser_map().keys().copied().collect()
}

#[derive(Debug, Clone, Default)]
pub struct ParserResult {
    /// The normal text outside of reasoning blocks.
    pub normal_text: String,

    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_text: String,
}

impl ParserResult {
    pub fn get_some_reasoning(&self) -> Option<String> {
        if self.reasoning_text.is_empty() {
            None
        } else {
            Some(self.reasoning_text.clone())
        }
    }

    pub fn get_some_normal_text(&self) -> Option<String> {
        if self.normal_text.is_empty() {
            None
        } else {
            Some(self.normal_text.clone())
        }
    }
}

pub trait ReasoningParser: Send + std::fmt::Debug {
    /// Parses a standalone, non-streaming input chunk. Implementations may reset or ignore
    /// internal streaming state and should return the split of normal vs reasoning text for
    /// this complete input. Marker tokens must not be included in either output.
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult;

    /// Parses a streaming chunk and updates internal state. The return value should be the
    /// delta: only the newly discovered normal and reasoning text attributable to this chunk
    /// (not the cumulative totals). Marker tokens must not be included in either output.
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningParserType {
    DeepseekR1,
    Step3,
    Basic,
    GptOss,
    Qwen,
    NemotronDeci,
    Kimi,
    Mistral,
}

#[derive(std::fmt::Debug)]
pub struct ReasoningParserWrapper {
    parser: Box<dyn ReasoningParser>,
}

impl ReasoningParser for ReasoningParserWrapper {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        self.parser.detect_and_parse_reasoning(text, token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        self.parser
            .parse_reasoning_streaming_incremental(text, token_ids)
    }
}

impl ReasoningParserType {
    pub fn get_reasoning_parser(self) -> ReasoningParserWrapper {
        let basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), false, true);
        let force_reasoning_basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true);
        match self {
            ReasoningParserType::DeepseekR1 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Step3 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Basic => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Qwen => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::NemotronDeci => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Kimi => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "◁think▷".into(),
                    "◁/think▷".into(),
                    false,
                    true,
                )),
            },
            ReasoningParserType::Mistral => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "[THINK]".into(),
                    "[/THINK]".into(),
                    true,
                    true,
                )),
            },
            ReasoningParserType::GptOss => match GptOssReasoningParser::new() {
                Ok(parser) => ReasoningParserWrapper {
                    parser: Box::new(parser),
                },
                Err(e) => {
                    tracing::warn!(
                        "GptOssReasoningParser could not be initialized, falling back to Basic Reasoning Parser: {e}"
                    );
                    ReasoningParserWrapper {
                        parser: Box::new(BasicReasoningParser::new(
                            "<think>".into(),
                            "</think>".into(),
                            false,
                            true,
                        )),
                    }
                }
            },
        }
    }

    pub fn get_reasoning_parser_from_name(name: &str) -> ReasoningParserWrapper {
        tracing::debug!("Selected reasoning parser: {}", name);

        let parser_map = get_reasoning_parser_map();
        let normalized_name = name.to_lowercase();

        match parser_map.get(normalized_name.as_str()) {
            Some(parser_type) => parser_type.get_reasoning_parser(),
            None => {
                tracing::warn!(
                    parser_name = name,
                    "Unknown reasoning parser type, falling back to Basic Reasoning Parser",
                );
                Self::Basic.get_reasoning_parser()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_available_reasoning_parsers() {
        let parsers = get_available_reasoning_parsers();
        assert!(!parsers.is_empty());
        // Update this list when adding a new parser
        let available_parsers = [
            "deepseek_r1",
            "basic",
            "gpt_oss",
            "qwen3",
            "nemotron_deci",
            "kimi",
            "step3",
            "mistral",
        ];
        for parser in available_parsers {
            assert!(parsers.contains(&parser));
        }
    }
}

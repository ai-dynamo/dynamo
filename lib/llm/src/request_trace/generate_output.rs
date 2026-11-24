// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-end parsing of native `/generate` output for request traces.
//!
//! `/generate` streams raw tokens, so the frontend never runs the model's
//! reasoning or tool-call parsers on the response path. When tracing is on,
//! the trace task decodes the finished output once and runs those parsers so
//! trace rows carry the same reasoning length and tool calls as chat rows.

use dynamo_parsers::tool_calling::parsers::{get_tool_parser_map, try_tool_call_parse};
use dynamo_parsers::{ParserResult, ReasoningParser, ReasoningParserType};

use crate::protocols::TokenIdType;
use crate::tokenizers::Tokenizer;

/// The tokenizer and parsers a model's `/generate` output is traced with.
#[derive(Clone)]
pub(crate) struct GenerateOutputParsers {
    tokenizer: Tokenizer,
    reasoning_parser: Option<String>,
    tool_call_parser: Option<String>,
}

impl GenerateOutputParsers {
    /// `None` when the model has no tokenizer or no parser to run.
    pub(crate) fn new(
        tokenizer: Option<Tokenizer>,
        reasoning_parser: Option<String>,
        tool_call_parser: Option<String>,
    ) -> Option<Self> {
        let reasoning_parser = reasoning_parser.filter(|parser| !parser.is_empty());
        let tool_call_parser = tool_call_parser.filter(|parser| !parser.is_empty());
        if reasoning_parser.is_none() && tool_call_parser.is_none() {
            return None;
        }
        Some(Self {
            tokenizer: tokenizer?,
            reasoning_parser,
            tool_call_parser,
        })
    }
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct GenerateOutputSummary {
    /// `None` when the model has no reasoning parser.
    pub(crate) reasoning_tokens: Option<u64>,
    pub(crate) tool_names: Vec<String>,
}

/// Counts reasoning tokens with the chat path's rule: a token is reasoning if
/// it adds reasoning text, or adds no visible text while reasoning is open.
#[derive(Default)]
struct ReasoningCounter {
    tokens: u64,
    active: bool,
}

impl ReasoningCounter {
    fn observe(&mut self, result: &ParserResult, tokens: u64) {
        let has_reasoning = !result.reasoning_text.is_empty();
        let has_visible = !result.normal_text.is_empty();
        if has_reasoning || (self.active && !has_visible) {
            self.tokens = self.tokens.saturating_add(tokens);
        }
        if has_reasoning {
            self.active = true;
        } else if has_visible {
            self.active = false;
        }
    }
}

pub(crate) async fn summarize(
    parsers: &GenerateOutputParsers,
    token_ids: &[TokenIdType],
) -> anyhow::Result<GenerateOutputSummary> {
    let mut decoder = parsers.tokenizer.decode_stream(&[], true);
    let mut reasoning = parsers
        .reasoning_parser
        .as_deref()
        .map(ReasoningParserType::get_reasoning_parser_from_name);
    let mut counter = ReasoningCounter::default();
    let mut normal_text = String::new();

    let mut feed = |text: &str, ids: &[TokenIdType], tokens: u64| match reasoning.as_mut() {
        Some(parser) => {
            let result = parser.parse_reasoning_streaming_incremental(text, ids);
            counter.observe(&result, tokens);
            normal_text.push_str(&result.normal_text);
        }
        None => normal_text.push_str(text),
    };
    for &id in token_ids {
        let piece = decoder.step(id)?.unwrap_or_default();
        feed(&piece, &[id], 1);
    }
    if let Some(tail) = decoder.finish()? {
        feed(&tail, &[], 0);
    }
    let reasoning_tokens = match reasoning.as_mut() {
        Some(parser) => {
            let tail = parser.finish_reasoning_stream();
            normal_text.push_str(&tail.normal_text);
            Some(counter.tokens)
        }
        None => None,
    };

    let mut tool_names = Vec::new();
    if let Some(config) = parsers
        .tool_call_parser
        .as_deref()
        .and_then(|name| get_tool_parser_map().get(name))
    {
        let (calls, _) = try_tool_call_parse(&normal_text, config, None).await?;
        tool_names = calls.into_iter().map(|call| call.function.name).collect();
    }
    Ok(GenerateOutputSummary {
        reasoning_tokens,
        tool_names,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::protocols::common::FinishReason;
    use crate::request_trace::{
        SharedFinishReasonMetadata, record_backend_finish_reason_metadata,
        record_generate_tool_calls, snapshot_finish_reason_metadata,
    };

    const TOOL_CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    fn tokenizer() -> Tokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
        );
        let tokenizer: Arc<dyn crate::tokenizers::traits::Tokenizer> =
            Arc::new(crate::tokenizers::HuggingFaceTokenizer::from_file(path).unwrap());
        Tokenizer::from(tokenizer)
    }

    fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<TokenIdType> {
        tokenizer.encode(text).unwrap().token_ids().to_vec()
    }

    fn parsers(reasoning: Option<&str>, tool_call: Option<&str>) -> GenerateOutputParsers {
        GenerateOutputParsers::new(
            Some(tokenizer()),
            reasoning.map(str::to_string),
            tool_call.map(str::to_string),
        )
        .unwrap()
    }

    #[test]
    fn parsers_need_a_tokenizer_and_a_parser() {
        assert!(GenerateOutputParsers::new(Some(tokenizer()), None, Some(String::new())).is_none());
        assert!(GenerateOutputParsers::new(None, Some("qwen3".into()), None).is_none());
    }

    #[tokio::test]
    async fn counts_reasoning_tokens_and_parses_tool_calls() {
        let tokenizer = tokenizer();
        let thought = ids(&tokenizer, "Look up the weather in Paris first.");
        let reasoning = ids(
            &tokenizer,
            "<think>Look up the weather in Paris first.</think>",
        );
        let output = [reasoning.clone(), ids(&tokenizer, TOOL_CALL)].concat();

        let summary = summarize(&parsers(Some("deepseek_r1"), Some("hermes")), &output)
            .await
            .unwrap();

        let reasoning_tokens = summary.reasoning_tokens.unwrap() as usize;
        assert!(
            (thought.len() - 1..=reasoning.len()).contains(&reasoning_tokens),
            "{reasoning_tokens} not in {}..={} ({reasoning:?})",
            thought.len() - 1,
            reasoning.len()
        );
        assert_eq!(summary.tool_names, ["get_weather"]);
    }

    #[tokio::test]
    async fn plain_or_malformed_output_has_no_tool_calls() {
        let tokenizer = tokenizer();
        for text in ["The weather is sunny.", "<tool_call>\n{\"name\": \"get_wea"] {
            let summary = summarize(
                &parsers(Some("qwen3"), Some("hermes")),
                &ids(&tokenizer, text),
            )
            .await
            .unwrap();
            assert_eq!(summary.reasoning_tokens, Some(0));
            assert!(summary.tool_names.is_empty());
        }
        let summary = summarize(&parsers(None, Some("hermes")), &ids(&tokenizer, TOOL_CALL))
            .await
            .unwrap();
        assert_eq!(summary.reasoning_tokens, None);
        assert_eq!(summary.tool_names, ["get_weather"]);
    }

    #[test]
    fn tool_calls_set_finish_reason_only_on_a_natural_stop() {
        for (backend, expected) in [
            (
                FinishReason::Stop,
                Some(dynamo_protocols::types::FinishReason::ToolCalls),
            ),
            (FinishReason::Length, None),
        ] {
            let metadata = SharedFinishReasonMetadata::default();
            record_backend_finish_reason_metadata(Some(&metadata), Some(0), Some(&backend), None);
            record_generate_tool_calls(&metadata, &["get_weather".to_string()]);
            let snapshot = snapshot_finish_reason_metadata(&metadata).unwrap();
            assert_eq!(snapshot.finish_reason, expected);
            assert_eq!(snapshot.tool_calls.len(), 1);
            assert_eq!(snapshot.tool_calls[0].name.as_deref(), Some("get_weather"));
            assert_eq!(snapshot.tool_calls[0].id, None);
        }
    }
}

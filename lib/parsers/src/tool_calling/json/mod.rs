// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod base_json_parser;
pub mod deepseek_v3_1_parser;
pub mod deepseek_v3_parser;

pub use super::{config, response};
pub use base_json_parser::{detect_tool_call_start_basic_json, try_tool_call_parse_basic_json};
pub use deepseek_v3_1_parser::{
    detect_tool_call_start_deepseek_v3_1, parse_tool_calls_deepseek_v3_1,
};
pub use deepseek_v3_parser::{detect_tool_call_start_deepseek_v3, parse_tool_calls_deepseek_v3};

pub use super::config::JsonParserConfig;
pub use super::response::ToolCallResponse;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub enum JsonParserType {
    // Basic is generic json parser which can handle most of the cases
    #[default]
    Basic,
    // Model Specific JSON Parsers
    DeepseekV3,
    DeepseekV31,
}

pub fn try_tool_call_parse_json(
    message: &str,
    config: &JsonParserConfig,
    tools: Option<&[super::ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    match config.parser_type {
        JsonParserType::Basic => try_tool_call_parse_basic_json(message, config, tools),
        JsonParserType::DeepseekV3 => parse_tool_calls_deepseek_v3(message, config, tools),
        JsonParserType::DeepseekV31 => parse_tool_calls_deepseek_v3_1(message, config, tools),
    }
}

pub fn detect_tool_call_start_json(chunk: &str, config: &JsonParserConfig) -> bool {
    match config.parser_type {
        JsonParserType::Basic => detect_tool_call_start_basic_json(chunk, config),
        JsonParserType::DeepseekV3 => detect_tool_call_start_deepseek_v3(chunk, config),
        JsonParserType::DeepseekV31 => detect_tool_call_start_deepseek_v3_1(chunk, config),
    }
}

pub fn find_tool_call_end_position_json(
    chunk: &str,
    parser: &str,
    config: &JsonParserConfig,
) -> usize {
    match parser {
        "hermes" | "nemotron_deci" => {
            let start_token = config.tool_call_start_tokens.first().map(|s| s.as_str());
            if let Some(end_token) = config.tool_call_end_tokens.first() {
                let Some(first_end) = chunk.find(end_token.as_str()) else {
                    return chunk.len();
                };
                let mut cursor = first_end + end_token.len();

                // Advance past any additional consecutive start→end blocks
                // so that parallel tool calls are captured as one jailed region.
                if let Some(start_tok) = start_token {
                    loop {
                        let rest = &chunk[cursor..];
                        let trimmed = rest.trim_start();
                        if !trimmed.starts_with(start_tok) {
                            break;
                        }
                        let trim_offset = rest.len() - trimmed.len();
                        let search_from = cursor + trim_offset + start_tok.len();
                        if let Some(end_pos) = chunk[search_from..].find(end_token.as_str()) {
                            cursor = search_from + end_pos + end_token.len();
                        } else {
                            break;
                        }
                    }
                }
                cursor
            } else {
                chunk.len()
            }
        }
        "mistral" | "phi4" => {
            if let Some(pos) = chunk.rfind(']') {
                pos + 1
            } else {
                chunk.len()
            }
        }
        _ => chunk.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for issue #6822: parallel tool calls in a single chunk must
    /// all be captured by find_tool_call_end_position_json so that the jail passes the
    /// entire group to the parser rather than emitting the second (and later) calls
    /// as raw trailing text.
    #[test] // PARSER.batch.2, helper
    fn test_find_tool_call_end_position_parallel_calls() {
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<tool_call>".to_string()],
            tool_call_end_tokens: vec!["</tool_call>".to_string()],
            ..Default::default()
        };

        // Two parallel calls with no whitespace between them.
        let two_calls = concat!(
            "<tool_call>{\"name\": \"foo\", \"arguments\": {\"x\": 1}}</tool_call>",
            "<tool_call>{\"name\": \"bar\", \"arguments\": {\"y\": 2}}</tool_call>",
            "trailing"
        );
        let pos = find_tool_call_end_position_json(two_calls, "hermes", &config);
        assert!(
            two_calls[..pos].ends_with("</tool_call>"),
            "should end at last </tool_call>, got: {:?}",
            &two_calls[..pos]
        );
        assert_eq!(&two_calls[pos..], "trailing");

        // Three parallel calls separated by whitespace / newlines.
        let three_calls = concat!(
            "<tool_call>{\"name\": \"a\"}</tool_call>\n",
            "<tool_call>{\"name\": \"b\"}</tool_call>\n",
            "<tool_call>{\"name\": \"c\"}</tool_call> done"
        );
        let pos3 = find_tool_call_end_position_json(three_calls, "hermes", &config);
        assert!(
            three_calls[..pos3].ends_with("</tool_call>"),
            "should end at last </tool_call>, got: {:?}",
            &three_calls[..pos3]
        );
        assert_eq!(three_calls[pos3..].trim(), "done");

        // Incomplete second call — should stop after the first complete one.
        let incomplete = concat!(
            "<tool_call>{\"name\": \"a\"}</tool_call>",
            "<tool_call>{\"name\": \"b\""
        );
        let pos_inc = find_tool_call_end_position_json(incomplete, "hermes", &config);
        let first_end = "<tool_call>{\"name\": \"a\"}</tool_call>".len();
        assert_eq!(
            pos_inc, first_end,
            "should stop at end of first complete call when second is incomplete"
        );
    }

    // Recovery for missing outer </TOOLCALL> (max_tokens / EOS truncation):
    // when the inner JSON array is well-formed, treat EOF as the end token
    // and extract the call rather than silently dropping it.
    #[test] // PARSER.batch.5 — nemotron_deci
    fn test_parse_nemotron_deci_no_outer_close_recovers() {
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
            allow_eof_recovery: true,
            ..Default::default()
        };
        // JSON array fully complete; only outer </TOOLCALL> missing.
        let input = r#"<TOOLCALL>[{"name":"get_weather","arguments":{"city":"NYC"}}]"#;

        let (calls, _normal_text) = try_tool_call_parse_json(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    // Verifies multi-call works correctly for nemotron_deci. The shared
    // dispatcher tests this at the integration layer (see
    // `parsers.rs::test_detect_and_parse_tool_call_default_parser_nemotron_deci_multiple`),
    // but no parser-level test pinned it. This test makes the contract
    // visible at the per-parser surface so a JSON-family refactor can't
    // silently break parallel-call extraction without a per-parser failure.
    #[test] // PARSER.batch.2 — nemotron_deci
    fn test_parse_nemotron_deci_multiple_calls() {
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
            ..Default::default()
        };
        // Two calls in a single <TOOLCALL>...</TOOLCALL> block, JSON array form.
        let input = r#"<TOOLCALL>[{"name":"get_weather","arguments":{"city":"NYC"}},{"name":"get_time","arguments":{"tz":"EST"}}]</TOOLCALL>"#;

        let (calls, normal_text) = try_tool_call_parse_json(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert_eq!(normal_text, Some("".to_string()));
    }

    // Recovery for truncated JSON args (max_tokens fires inside
    // `"city":"NYC` with no closing quote, brace, or array bracket). The
    // base parser balances unclosed strings/braces and retries the parse,
    // surfacing the call rather than silently dropping it.
    #[test] // PARSER.batch.4 — nemotron_deci
    fn test_parse_nemotron_deci_truncated_json_recovers() {
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
            allow_eof_recovery: true,
            ..Default::default()
        };
        let input = r#"<TOOLCALL>[{"name":"get_weather","arguments":{"city":"NYC</TOOLCALL>"#;

        let (calls, _) = try_tool_call_parse_json(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    fn nemotron_deci_config() -> JsonParserConfig {
        JsonParserConfig {
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
            ..Default::default()
        }
    }

    /// Parser-level invariant: the json-family parser is byte-stable — it
    /// doesn't see `finish_reason` and produces the same output regardless
    /// of the upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop /
    /// tool_calls / length mapping) lives in
    /// `lib/llm/tests/test_streaming_tool_parsers.rs` and belongs in the
    /// cross-parser finish_reason mapping work-item (tracked separately).
    #[test]
    fn test_nemotron_deci_parser_output_independent_of_upstream_finish() {
        let config = nemotron_deci_config();
        let input = r#"<TOOLCALL>[{"name":"get_weather","arguments":{"city":"NYC"}}]</TOOLCALL>"#;
        let (calls, _) = try_tool_call_parse_json(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
    }

    /// PARSER.batch.6 — empty args. A no-arg call (`{}`) must still be returned
    /// with the function name intact.
    #[test] // PARSER.batch.6 — nemotron_deci
    fn test_parse_nemotron_deci_empty_args() {
        let config = nemotron_deci_config();
        let input = r#"<TOOLCALL>[{"name":"current_time","arguments":{}}]</TOOLCALL>"#;
        let (calls, _) = try_tool_call_parse_json(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "current_time");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({}));
    }

    /// PARSER.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls; normal_text
    /// collapses to the empty string.
    #[test] // PARSER.batch.9 — nemotron_deci
    fn test_parse_nemotron_deci_empty_and_whitespace_inputs() {
        let config = nemotron_deci_config();
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (calls, normal) = try_tool_call_parse_json(input, &config, None).unwrap();
            assert!(
                calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(""),
                "Empty/whitespace input collapses to empty normal_text (input={:?})",
                input
            );
        }
    }

    /// PARSER.batch.10 — duplicate calls (same function name twice in one section).
    /// JSON-array form pin parser-level behavior — both calls returned with
    /// distinct ids.
    #[test] // PARSER.batch.10 — nemotron_deci
    fn test_parse_nemotron_deci_duplicate_calls_same_name() {
        let config = nemotron_deci_config();
        let input = r#"<TOOLCALL>[{"name":"get_weather","arguments":{"city":"NYC"}},{"name":"get_weather","arguments":{"city":"LA"}}]</TOOLCALL>"#;
        let (calls, _) = try_tool_call_parse_json(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2, "Both duplicate-name calls must be returned");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");
        assert_ne!(
            calls[0].id, calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let args0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["city"], "NYC");
        assert_eq!(args1["city"], "LA");
    }
}

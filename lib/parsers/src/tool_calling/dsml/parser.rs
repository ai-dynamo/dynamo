// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementations:
// V3.2: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding/encoding_dsv32.py
// V4:   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/encoding/encoding_dsv4.py
//
// V4 reuses this same DSML engine; only the outer block name changes from
// `function_calls` to `tool_calls` (configured via DsmlParserConfig).

use regex::Regex;
use uuid::Uuid;

use super::super::config::DsmlParserConfig;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// DeepSeek V3.2 / V4 use DSML (DeepSeek Markup Language) format for tool calls.
/// V3.2 wraps calls in `<｜DSML｜function_calls>`; V4 wraps them in
/// `<｜DSML｜tool_calls>`. The inner invoke / parameter grammar is identical:
///
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="function_name">
/// <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
/// ...
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// Check if a chunk contains the start of a DSML tool call
pub fn detect_tool_call_start_dsml(chunk: &str, config: &DsmlParserConfig) -> bool {
    let start_token = &config.block_start;

    // Check for complete start token
    if chunk.contains(start_token.as_str()) {
        return true;
    }

    // Check for partial match at the end (streaming scenario)
    let start_chars: Vec<char> = start_token.chars().collect();
    for i in 1..start_chars.len() {
        let partial: String = start_chars[..i].iter().collect();
        if chunk.ends_with(&partial) {
            return true;
        }
    }

    false
}

/// Find the end position of a DSML tool call block
pub fn find_tool_call_end_position_dsml(chunk: &str, config: &DsmlParserConfig) -> usize {
    let end_token = &config.block_end;

    if let Some(pos) = chunk.find(end_token.as_str()) {
        pos + end_token.len()
    } else {
        chunk.len()
    }
}

/// Build the regex that matches a complete DSML tool_calls / function_calls block.
/// Shared by `extract_tool_calls_with_regex` and `try_tool_call_parse_dsml` so
/// the two stay in lockstep on how a block is recognised.
fn build_block_regex(config: &DsmlParserConfig) -> anyhow::Result<Regex> {
    // Matches: <｜DSML｜function_calls> ... </｜DSML｜function_calls>
    // Pattern: (?s) = dot matches newlines
    //          \s*(.*?)\s* = capture content between start/end tags (non-greedy)
    let block_pattern = format!(
        r"(?s){}\s*(.*?)\s*{}",
        regex::escape(&config.block_start),
        regex::escape(&config.block_end)
    );
    Ok(Regex::new(&block_pattern)?)
}

/// Parse DSML formatted tool calls from a message
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_dsml(
    message: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let trimmed = message.trim();

    // Early exit if no content
    if trimmed.is_empty() {
        return Ok((vec![], Some(String::new())));
    }

    // Check if tool call block exists
    let start_idx = trimmed.find(&config.block_start);
    if start_idx.is_none() {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Extract tool calls blocks
    let block_regex = build_block_regex(config)?;
    let tool_calls = extract_tool_calls_with_regex(trimmed, &block_regex, config)?;

    if tool_calls.is_empty() {
        // A block-start was detected but no valid invokes parsed. Do NOT leak
        // the DSML markup back to the client; return only the pre-block text
        // and emit a diagnostic with a prefix of the failed block.
        //
        // Note: an unterminated block-start here means `block_regex` finds no
        // match at all, so any valid block *after* the unterminated one is
        // lost. This matches the pre-existing conservative P1-3 contract.
        if let Some(idx) = start_idx {
            let failed = &trimmed[idx..];
            let prefix: String = failed.chars().take(120).collect();
            tracing::warn!(
                "DSML tool_calls block parsed no invokes; suppressing markup. prefix={:?}",
                prefix
            );
        }
        let pre_block_text = start_idx
            .map(|idx| trimmed[..idx].to_string())
            .unwrap_or_default();
        return Ok((vec![], Some(pre_block_text)));
    }

    // Preserve inter-block and trailing text: strip every complete block span
    // from the trimmed input rather than slicing up to the first start token.
    // Without this we silently lose text between and after multiple blocks.
    let normal_text = block_regex.replace_all(trimmed, "").to_string();

    Ok((tool_calls, Some(normal_text)))
}

/// Extract all tool calls matched by `block_regex` from the DSML formatted text.
fn extract_tool_calls_with_regex(
    text: &str,
    block_regex: &Regex,
    config: &DsmlParserConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut tool_calls = Vec::new();

    for block_match in block_regex.captures_iter(text) {
        if let Some(block_content) = block_match.get(1) {
            let block = block_content.as_str();

            // Extract individual invokes from this block
            let invokes = extract_invokes(block, config)?;
            tool_calls.extend(invokes);
        }
    }

    Ok(tool_calls)
}

/// Extract individual invoke blocks from function_calls content
fn extract_invokes(
    block: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut invokes = Vec::new();

    // Regex to match: <｜DSML｜invoke name="function_name">..content..</｜DSML｜invoke>
    // Note: invoke_start_prefix is "<｜DSML｜invoke name=" (no quotes, we add them in pattern)
    let invoke_pattern = format!(
        r#"(?s){}\"([^"]+)\"\s*>(.*?){}"#,
        regex::escape(&config.invoke_start_prefix),
        regex::escape(&config.invoke_end)
    );
    let invoke_regex = Regex::new(&invoke_pattern)?;

    for invoke_match in invoke_regex.captures_iter(block) {
        if let (Some(name_match), Some(content_match)) = (invoke_match.get(1), invoke_match.get(2))
        {
            let function_name = name_match.as_str().trim().to_string();
            let invoke_content = content_match.as_str();

            // Parse parameters from invoke content
            let parameters = parse_parameters(invoke_content, config)?;

            // Create tool call response
            let arguments_json = serde_json::to_string(&parameters)?;

            // OpenAI-style id: "call_" + 24 lowercase hex chars.
            // Take the simple (32-hex, no hyphens) form of a v4 UUID and truncate.
            let uuid_simple = Uuid::new_v4().simple().to_string();
            let id = format!("call_{}", &uuid_simple[..24]);

            invokes.push(ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: function_name,
                    arguments: arguments_json,
                },
            });
        }
    }

    Ok(invokes)
}

/// Parse parameters from invoke content
fn parse_parameters(
    content: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut parameters = serde_json::Map::new();

    // Build pattern with proper escaping
    // Match: <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
    // Note: parameter_prefix is "<｜DSML｜parameter name=" (no quotes, we add them in pattern)
    let prefix_escaped = regex::escape(&config.parameter_prefix);
    let end_escaped = regex::escape(&config.parameter_end);

    // The `string="true|false"` attribute is optional: some model outputs omit it.
    // When absent we best-effort parse the value (JSON → String fallback).
    let param_pattern = format!(
        r#"(?s){}\"([^"]+)\"(?:\s+string=\"(true|false)\")?\s*>(.*?){}"#,
        prefix_escaped, end_escaped
    );

    let param_regex = Regex::new(&param_pattern)?;

    for param_match in param_regex.captures_iter(content) {
        if let (Some(name_match), Some(value_match)) = (param_match.get(1), param_match.get(3)) {
            let param_name = name_match.as_str().trim();
            let param_value = value_match.as_str().trim();

            // Parse value based on string attribute (if present).
            // `string="true"` forces the String branch; every other case
            // (`string="false"` or attribute omitted) tries JSON first and
            // falls back to String.
            let string_attr = param_match.get(2).map(|m| m.as_str());
            let value = if string_attr == Some("true") {
                serde_json::Value::String(param_value.to_string())
            } else {
                serde_json::from_str(param_value)
                    .unwrap_or_else(|_| serde_json::Value::String(param_value.to_string()))
            };

            parameters.insert(param_name.to_string(), value);
        }
    }

    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    fn get_test_config() -> DsmlParserConfig {
        DsmlParserConfig::default()
    }

    fn get_v4_test_config() -> DsmlParserConfig {
        DsmlParserConfig {
            block_start: "<｜DSML｜tool_calls>".to_string(),
            block_end: "</｜DSML｜tool_calls>".to_string(),
            ..Default::default()
        }
    }

    #[test] // CASE.20
    fn test_detect_tool_call_start() {
        let config = get_test_config();
        assert!(detect_tool_call_start_dsml(
            "<｜DSML｜function_calls>",
            &config
        ));
        assert!(detect_tool_call_start_dsml(
            "text <｜DSML｜function_calls>",
            &config
        ));
        assert!(detect_tool_call_start_dsml("<｜DSML｜function_c", &config)); // Partial
        assert!(!detect_tool_call_start_dsml("no tool call here", &config));
    }

    // -------------------------------------------------------------------
    // DeepSeek V4 coverage (see lib/parsers/TEST_CASES.md for CASE.* taxonomy).
    //
    // Covered by the V4 tests below (or by a shared DSML generic test):
    //   - CASE.1   single-call            (parsers.rs :: test_deepseek_v4_single_tool_call)
    //   - CASE.2   multi-calls            (test_parse_deepseek_v4_multiple_tool_calls)
    //   - CASE.3   no-call                (shared: test_parse_no_tool_calls)
    //   - CASE.4   malformed-args         (test_parse_deepseek_v4_malformed_json_value_falls_back_to_string,
    //                                      test_parse_deepseek_v4_missing_invoke_close_drops_call)
    //   - CASE.5   missing-end-token      (test_parse_deepseek_v4_missing_end_token{,_multiple_calls})
    //                                      — PINNED AS BROKEN: parser drops the call. See TODO below.
    //   - CASE.6   empty-args             (test_parse_deepseek_v4_no_parameters)
    //   - CASE.7   complex-args           (shared: test_parse_mixed_types_realistic, test_parse_nested_object_parameter,
    //                                      lib/llm/tests/test_streaming_tool_parsers :: ..._mixed_param_types_vllm,
    //                                      ..._special_chars_vllm)
    //   - CASE.8   streaming              (test_detect_tool_call_start_v4, test_find_tool_call_end_position_v4,
    //                                      lib/llm/tests/test_streaming_tool_parsers :: ..._fragmented_tokens_vllm)
    //   - CASE.9   reasoning-plus-tool    (lib/llm/tests/test_streaming_tool_parsers :: ..._with_tools_vllm
    //                                      — fixtures include <think>...</think> alongside DSML)
    //   - CASE.10  reasoning-only         (reasoning/mod.rs :: test_deepseek_v4_detect_and_parse etc.)
    //   - CASE.12  finish-reason          (lib/llm/tests/test_streaming_tool_parsers :: ..._with_tools_vllm →
    //                                      FinishReason::ToolCalls; ..._with_no_tools_vllm → FinishReason::Stop
    //                                      — Length variant NOT covered, see TODO)
    //   - CASE.13  interleaved-text       (test_parse_deepseek_v4_multiple_tool_calls prefix text;
    //                                      lib/llm/tests/test_streaming_tool_parsers :: ..._content_before_tool_vllm)
    //
    //   - CASE.xml.*  N/A — DSML carries per-parameter string="true|false" type hints,
    //                  so XML entity decoding (CASE.xml.entities) and schema-aware
    //                  coercion (CASE.xml.schema-coercion) don't apply.
    //   - CASE.harmony.* N/A — Harmony-only.
    //
    // TODO — not yet covered for V4:
    //   - CASE.5  Fix mid-stream truncation: parser currently drops all calls when
    //             </｜DSML｜tool_calls> is absent (max_tokens / EOS before close).
    //             Same class as Kimi K2 pre-PR #8208. Recovery pattern: scan for
    //             complete <｜DSML｜invoke>...</｜DSML｜invoke> pairs even without
    //             the outer close fence (see kimi_k2_parser.rs for precedent).
    //             Pinning tests below capture the current silent-drop behavior;
    //             flip them when recovery lands.
    //   - CASE.4  Variants not pinned: missing </｜DSML｜parameter> close tag,
    //             middle-invoke truncation corrupting subsequent invokes (non-greedy
    //             regex bleed-through). Same structural class as CASE.5.
    //   - CASE.11 tool_choice auto/required/named/none — cross-parser suites at
    //             lib/llm/tests/tool_choice.rs run hermes only; V4 not exercised.
    //   - CASE.12 FinishReason::Length — current E2E fixtures only cover Stop and
    //             ToolCalls finish reasons. No truncation-forcing fixture.
    //   - CASE.14 empty-content / null response at the e2e layer.
    //   - CASE.15 duplicate-calls (same name twice) — universal gap across all parsers.
    //   - CASE.16 regression — V4 is hours old (2026-04-24); no customer bugs filed yet.
    // -------------------------------------------------------------------

    /// `CASE.8` — streaming start-token detection (V4 variant).
    #[test] // CASE.20, CASE.23 — V4 token variant
    fn test_detect_tool_call_start_v4() {
        let config = get_v4_test_config();
        assert!(detect_tool_call_start_dsml("<｜DSML｜tool_calls>", &config));
        assert!(detect_tool_call_start_dsml(
            "text <｜DSML｜tool_calls>",
            &config
        ));
        assert!(detect_tool_call_start_dsml("<｜DSML｜tool_c", &config));
        assert!(!detect_tool_call_start_dsml(
            "<｜DSML｜function_calls>",
            &config
        ));
        assert!(!detect_tool_call_start_dsml("no tool call here", &config));
    }

    #[test] // CASE.20
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let text = "<｜DSML｜function_calls><｜DSML｜invoke name=\"test\"></｜DSML｜invoke></｜DSML｜function_calls>more";
        let pos = find_tool_call_end_position_dsml(text, &config);
        assert_eq!(&text[pos..], "more");
    }

    /// `CASE.8` — streaming end-position lookup (V4 variant).
    #[test] // CASE.20, CASE.23 — V4 token variant
    fn test_find_tool_call_end_position_v4() {
        let config = get_v4_test_config();
        let text = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"test\"></｜DSML｜invoke></｜DSML｜tool_calls>more";
        let pos = find_tool_call_end_position_dsml(text, &config);
        assert_eq!(&text[pos..], "more");
    }

    #[test] // CASE.1
    fn test_parse_single_tool_call_string_param() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">San Francisco</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let result = try_tool_call_parse_dsml(input, &config);
        if let Err(e) = &result {
            eprintln!("Parse error: {:?}", e);
        }
        let (calls, normal) = result.unwrap();

        if calls.is_empty() {
            eprintln!("Input: {}", input);
            eprintln!("No calls parsed!");
        }

        assert_eq!(calls.len(), 1, "Expected 1 tool call, got {}", calls.len());
        assert_eq!(normal, Some("".to_string()));

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco");
    }

    #[test] // CASE.1, CASE.7
    fn test_parse_single_tool_call_mixed_params() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">test query</｜DSML｜parameter>
<｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "search");
        assert_eq!(args["query"], "test query");
        assert_eq!(args["topn"], 10);
    }

    #[test] // CASE.2
    fn test_parse_multiple_tool_calls() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Hangzhou</｜DSML｜parameter>
<｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 2);

        let (name1, args1) = extract_name_and_args(calls[0].clone());
        assert_eq!(name1, "get_weather");
        assert_eq!(args1["location"], "Beijing");

        let (name2, args2) = extract_name_and_args(calls[1].clone());
        assert_eq!(name2, "get_weather");
        assert_eq!(args2["location"], "Hangzhou");
    }

    /// `CASE.2` multi-calls + `CASE.13` interleaved-text (prefix text before the block).
    #[test] // CASE.2, CASE.23 — V4 variant
    fn test_parse_deepseek_v4_multiple_tool_calls() {
        let input = r#"Let's check this. <｜DSML｜tool_calls>
<｜DSML｜invoke name="get_favorite_tourist_spot">
<｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">search agent benchmark 2024</｜DSML｜parameter>
<｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
<｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;

        let config = get_v4_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 2);
        // Tolerant match: preamble must carry the prose; whitespace is
        // implementation-defined.
        let normal = normal.unwrap();
        assert_eq!(normal.trim(), "Let's check this.");

        let (name1, args1) = extract_name_and_args(calls[0].clone());
        assert_eq!(name1, "get_favorite_tourist_spot");
        assert_eq!(args1["city"], "Beijing");

        let (name2, args2) = extract_name_and_args(calls[1].clone());
        assert_eq!(name2, "search");
        assert_eq!(args2["query"], "search agent benchmark 2024");
        assert_eq!(args2["topn"], 10);
        assert_eq!(args2["source"], "web");
    }

    /// `CASE.6` — empty args (no-parameter invoke).
    #[test] // CASE.6, CASE.23 — V4 variant
    fn test_parse_deepseek_v4_no_parameters() {
        let input = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_current_time">
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;

        let config = get_v4_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(normal, Some("".to_string()));

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "get_current_time");
        assert_eq!(args, serde_json::json!({}));
    }

    #[test] // CASE.13
    fn test_parse_with_normal_text() {
        let input = r#"Here's the result: <｜DSML｜function_calls>
<｜DSML｜invoke name="test">
<｜DSML｜parameter name="value" string="true">test</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
        // Tolerant whitespace match.
        let normal = normal.unwrap();
        assert_eq!(normal.trim(), "Here's the result:");
    }

    #[test]
    fn test_parse_preserves_whitespace_before_dsml_block() {
        // vLLM preserves whitespace verbatim before the DSML block; the parser
        // must as well so clients see identical prompts across servers.
        let input = "Let me check the forecast.\n\n<｜DSML｜tool_calls>
<｜DSML｜invoke name=\"get_weather\">
<｜DSML｜parameter name=\"city\" string=\"true\">SF</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>";

        let config = get_v4_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
        let normal = normal.unwrap();
        assert!(
            normal.ends_with("\n\n"),
            "Expected trailing \\n\\n preserved, got {:?}",
            normal
        );
        assert_eq!(normal, "Let me check the forecast.\n\n");
    }

    #[test] // CASE.3
    fn test_parse_no_tool_calls() {
        let input = "This is just normal text without any tool calls.";
        let config = get_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some(input.to_string()));
    }

    #[test] // CASE.7
    fn test_parse_json_parameter_value() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="process">
<｜DSML｜parameter name="config" string="false">{"key": "value", "count": 42}</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["config"].is_object());
        assert_eq!(args["config"]["key"], "value");
        assert_eq!(args["config"]["count"], 42);
    }

    #[test] // CASE.7
    fn test_parse_array_parameter_value() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="process">
<｜DSML｜parameter name="items" string="false">[1, 2, 3]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["items"].is_array());
        assert_eq!(args["items"][0], 1);
        assert_eq!(args["items"][2], 3);
    }

    #[test] // CASE.7
    fn test_parse_boolean_parameters() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="config">
<｜DSML｜parameter name="enabled" string="false">true</｜DSML｜parameter>
<｜DSML｜parameter name="disabled" string="false">false</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(args["enabled"], true);
        assert_eq!(args["disabled"], false);
    }

    #[test] // CASE.7
    fn test_parse_number_parameters() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="calculate">
<｜DSML｜parameter name="integer" string="false">42</｜DSML｜parameter>
<｜DSML｜parameter name="float" string="false">2.7</｜DSML｜parameter>
<｜DSML｜parameter name="negative" string="false">-100</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(args["integer"], 42);
        assert_eq!(args["float"], 2.7);
        assert_eq!(args["negative"], -100);
    }

    #[test] // CASE.7
    fn test_parse_mixed_types_realistic() {
        // Realistic example based on test data
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">search agent benchmark 2024</｜DSML｜parameter>
<｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
<｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "search");
        assert_eq!(args["query"], "search agent benchmark 2024");
        assert_eq!(args["topn"], 10); // Should be number, not string
        assert_eq!(args["source"], "web");
    }

    #[test] // CASE.7
    fn test_parse_nested_object_parameter() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="configure">
<｜DSML｜parameter name="settings" string="false">{"timeout": 30, "retry": true, "endpoints": ["a", "b"]}</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["settings"].is_object());
        assert_eq!(args["settings"]["timeout"], 30);
        assert_eq!(args["settings"]["retry"], true);
        assert!(args["settings"]["endpoints"].is_array());
        assert_eq!(args["settings"]["endpoints"][0], "a");
    }

    #[test] // CASE.7
    fn test_parse_empty_string_parameter() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="test">
<｜DSML｜parameter name="empty" string="true"></｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(args["empty"], "");
    }

    #[test]
    fn test_empty_invokes_does_not_leak_dsml_markup() {
        // Valid block-start + mangled content (invoke tag but no closing/params)
        // followed by block-end. extract_tool_calls returns empty; we must not
        // leak DSML markup into normal_content.
        let input = "Let me check. <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"broken\">\n</｜DSML｜tool_calls>";

        let config = get_v4_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();

        assert!(
            calls.is_empty(),
            "Expected no tool calls, got {}",
            calls.len()
        );
        let normal = normal.unwrap();
        assert!(
            normal.contains("Let me check."),
            "Expected preamble in normal_content, got {:?}",
            normal
        );
        assert!(
            !normal.contains("<｜DSML｜"),
            "normal_content leaked DSML markup: {:?}",
            normal
        );
    }

    #[test]
    fn test_parse_parameter_missing_string_attribute() {
        // Model emits a parameter without the `string="..."` attribute.
        // The parser should best-effort parse: JSON first, then fall back to string.
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="greet">
<｜DSML｜parameter name="name">Alice</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "greet");
        assert_eq!(args["name"], "Alice");
    }

    #[test]
    fn test_parse_string_false_with_bare_word_value() {
        // `string="false"` with a non-JSON bare word should still appear
        // in the arguments (as the string fallback).
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="run">
<｜DSML｜parameter name="mode" string="false">quickly</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(args["mode"], "quickly");
    }

    #[test]
    fn test_tool_call_id_format_openai_style() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">San Francisco</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        // Shape-only assertion: OpenAI-style `call_` prefix + at least 20
        // lowercase alphanumeric characters. We intentionally do NOT pin the
        // exact length / alphabet so the id generator can evolve without
        // churning this test.
        let id = &calls[0].id;
        assert!(
            id.starts_with("call_"),
            "id should start with call_: {}",
            id
        );
        let suffix = &id["call_".len()..];
        assert!(
            suffix.len() >= 20,
            "suffix must be at least 20 chars: {}",
            suffix
        );
        assert!(
            suffix
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()),
            "suffix must match [a-z0-9]+: {}",
            suffix
        );
    }

    #[test]
    fn test_multi_block_preserves_inter_and_trailing_text() {
        // Two complete DSML blocks with text before, between, and after.
        // Both blocks must be parsed AND the inter-block / trailing text must
        // survive in normal_content.
        let input = "pre <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"a\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> middle <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"b\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> tail";

        let config = get_v4_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 2, "expected both blocks parsed");
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");

        let normal = normal.unwrap();
        assert!(
            normal.contains(" middle "),
            "inter-block text lost: {:?}",
            normal
        );
        assert!(normal.contains(" tail"), "trailing text lost: {:?}", normal);
        assert!(
            !normal.contains("<｜DSML｜"),
            "normal_content leaked DSML markup: {:?}",
            normal
        );
    }

    #[test]
    fn test_unterminated_block_followed_by_valid_block() {
        // An unterminated DSML start appears before a complete block. The
        // non-greedy block regex spans from the FIRST block_start to the
        // sole block_end, swallowing the nested second block_start as part
        // of the block content.
        //
        // Within that captured span the non-greedy invoke regex pairs the
        // FIRST `<invoke name="broken">` with the FIRST `</invoke>` — so
        // one tool call is recovered under the name "broken". This is the
        // observed contract today; the test locks it in so any future
        // behavior change is explicit rather than silent.
        let input = "pre <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"broken\">\n mid <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"ok\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> tail";

        let config = get_v4_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();

        assert_eq!(calls.len(), 1, "exactly one invoke recovered");
        assert_eq!(
            calls[0].function.name, "broken",
            "outer invoke name is matched first (non-greedy)"
        );

        let normal = normal.unwrap();
        assert!(
            normal.starts_with("pre"),
            "pre-block text must survive: {:?}",
            normal
        );
        assert!(
            normal.contains(" tail"),
            "trailing text must survive: {:?}",
            normal
        );
        assert!(
            !normal.contains("<｜DSML｜tool_calls>"),
            "normal_content leaked block_start: {:?}",
            normal
        );
        assert!(
            !normal.contains("</｜DSML｜tool_calls>"),
            "normal_content leaked block_end: {:?}",
            normal
        );
    }

    #[test] // CASE.7, CASE.14
    fn test_parse_null_parameter() {
        let input = r#"<｜DSML｜function_calls>
<｜DSML｜invoke name="test">
<｜DSML｜parameter name="value" string="false">null</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (_, args) = extract_name_and_args(calls[0].clone());
        assert!(args["value"].is_null());
    }

    // Corner-case pinning tests. See the V4 coverage manifest above for the
    // full mapping from CASE.* → test. Each test's doc-comment names the
    // specific CASE it pins.

    /// `CASE.5` — missing end-token recovery.
    /// **Pinned as broken** — parser drops the call; see the TODO block above.
    ///
    /// If a DeepSeek V4 stream is truncated before `</｜DSML｜tool_calls>`
    /// arrives (max_tokens cut-off, EOS mid-generation, connection drop),
    /// the block regex requires both fences and matches zero times. The
    /// entire DSML-looking payload falls through as raw `normal_text`; no
    /// tool calls are recovered.
    ///
    /// This is the same structural failure mode Kimi K2 had before its
    /// parser gained end-token recovery; see
    /// `kimi_k2_parser.rs::test_parse_malformed_no_section_end` for the
    /// post-fix recovery pattern.
    ///
    /// Note: post-hardening, the parser no longer leaks raw DSML markup
    /// into `normal_text` when block-start appears but no invokes parse —
    /// it returns the pre-block text only (empty here, since the input
    /// starts with the block-start fence). The call is still dropped.
    #[test] // CASE.5, CASE.23 — V4 variant
    fn test_parse_deepseek_v4_missing_end_token() {
        // Start fence + complete invoke, but no </｜DSML｜tool_calls>.
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>";

        let config = get_v4_test_config();
        let (calls, normal_text) = try_tool_call_parse_dsml(input, &config).unwrap();

        assert!(
            calls.is_empty(),
            "V4 DSML parser currently drops tool calls when \
             </｜DSML｜tool_calls> is missing. \
             If recovery is added, flip this assertion."
        );
        assert_eq!(
            normal_text.as_deref(),
            Some(""),
            "Pre-block text is empty here; raw DSML markup must not leak \
             into normal_text (post-hardening behavior)."
        );
    }

    /// `CASE.5` — multiple complete invokes, missing end fence.
    ///
    /// Even with multiple fully-formed invokes inside the start fence, the
    /// absence of the closing fence prevents the block regex from matching.
    /// All calls are dropped. If the parser ever gains partial-block
    /// recovery, this test will fail and force an intentional update.
    #[test] // CASE.2, CASE.5, CASE.23
    fn test_parse_deepseek_v4_missing_end_token_multiple_calls() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"a\">\n\
<｜DSML｜parameter name=\"x\" string=\"true\">1</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"b\">\n\
<｜DSML｜parameter name=\"y\" string=\"true\">2</｜DSML｜parameter>\n\
</｜DSML｜invoke>";

        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();

        assert!(
            calls.is_empty(),
            "Even two fully-formed invokes are dropped when the outer \
             </｜DSML｜tool_calls> is missing."
        );
    }

    /// `CASE.4` — malformed JSON in a `string="false"` parameter value falls back
    /// to a string. `parse_parameters` explicitly swallows the serde error
    /// (unwrap_or_else → Value::String). Pin the fallback so removing it
    /// (which would cause the whole call to 500 on ragged-edge JSON) is a
    /// deliberate change.
    #[test] // CASE.4, CASE.23
    fn test_parse_deepseek_v4_malformed_json_value_falls_back_to_string() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"test\">\n\
<｜DSML｜parameter name=\"payload\" string=\"false\">{this is not valid json</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";

        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);

        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "test");
        assert_eq!(
            args["payload"], "{this is not valid json",
            "Malformed JSON should fall back to the raw string, not drop \
             the parameter or the call."
        );
    }

    /// `CASE.4` — malformed invoke (missing `</｜DSML｜invoke>` but block fences
    /// intact). The invoke regex requires its own close tag, so the call is
    /// silently dropped. Pin the behavior.
    #[test] // CASE.4, CASE.23
    fn test_parse_deepseek_v4_missing_invoke_close_drops_call() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"test\">\n\
<｜DSML｜parameter name=\"x\" string=\"true\">value</｜DSML｜parameter>\n\
</｜DSML｜tool_calls>";

        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert!(
            calls.is_empty(),
            "Malformed invoke (missing </｜DSML｜invoke>) is dropped today. \
             If recovery is added, flip this assertion."
        );
    }
}

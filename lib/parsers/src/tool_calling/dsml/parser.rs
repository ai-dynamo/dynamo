// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding/encoding_dsv32.py

use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use uuid::Uuid;

use super::super::config::DsmlParserConfig;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Compiled regex trio for a given `DsmlParserConfig`. Compiled once and
/// reused across every subsequent parse/stream chunk.
struct DsmlRegexes {
    block: Regex,
    invoke: Regex,
    parameter: Regex,
}

/// Cache key = the six config strings that drive the three regex patterns.
/// V3.2 and V4 are the only variants in use, so the cache has at most two
/// entries for the lifetime of the process.
type DsmlRegexKey = (String, String, String, String, String, String);

fn regex_cache() -> &'static RwLock<HashMap<DsmlRegexKey, Arc<DsmlRegexes>>> {
    static CACHE: OnceLock<RwLock<HashMap<DsmlRegexKey, Arc<DsmlRegexes>>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Return the compiled regex trio for `config`, compiling on first use.
///
/// Each parse call previously recompiled three regexes from `format!`'d
/// patterns that embed the config strings — expensive on streaming hot paths.
/// The cache is keyed on the raw config strings (not the escaped patterns)
/// so distinct configs that happen to escape identically still get distinct
/// entries.
fn get_dsml_regexes(config: &DsmlParserConfig) -> anyhow::Result<Arc<DsmlRegexes>> {
    let key: DsmlRegexKey = (
        config.function_calls_start.clone(),
        config.function_calls_end.clone(),
        config.invoke_start_prefix.clone(),
        config.invoke_end.clone(),
        config.parameter_prefix.clone(),
        config.parameter_end.clone(),
    );
    // Fast path: shared read lock, common after the first parse of each config.
    if let Some(regexes) = regex_cache()
        .read()
        .expect("DSML regex cache read lock poisoned")
        .get(&key)
    {
        return Ok(Arc::clone(regexes));
    }
    // Slow path: compile and install. Use `entry` so a concurrent compiler of
    // the same key only inserts once (we still compile speculatively, then
    // drop the duplicate — cheap on a map of <= 2 keys).
    let block = Regex::new(&format!(
        r"(?s){}\s*(.*?)\s*{}",
        regex::escape(&config.function_calls_start),
        regex::escape(&config.function_calls_end),
    ))?;
    let invoke = Regex::new(&format!(
        r#"(?s){}\"([^"]+)\"\s*>(.*?){}"#,
        regex::escape(&config.invoke_start_prefix),
        regex::escape(&config.invoke_end),
    ))?;
    let parameter = Regex::new(&format!(
        r#"(?s){}\"([^"]+)\"\s+string=\"(true|false)\"\s*>(.*?){}"#,
        regex::escape(&config.parameter_prefix),
        regex::escape(&config.parameter_end),
    ))?;
    let regexes = Arc::new(DsmlRegexes {
        block,
        invoke,
        parameter,
    });
    let mut cache = regex_cache()
        .write()
        .expect("DSML regex cache write lock poisoned");
    Ok(Arc::clone(cache.entry(key).or_insert(regexes)))
}

/// DeepSeek V3.2 uses DSML (DeepSeek Markup Language) format for tool calls:
///
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="function_name">
/// <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
/// ...
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// Check if a chunk contains the start of a DSML tool call
pub fn detect_tool_call_start_dsml(chunk: &str, config: &DsmlParserConfig) -> bool {
    let start_token = &config.function_calls_start;

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
    let end_token = &config.function_calls_end;

    if let Some(pos) = chunk.find(end_token.as_str()) {
        pos + end_token.len()
    } else {
        chunk.len()
    }
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
    if !trimmed.contains(&config.function_calls_start) {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    // Extract normal text before tool calls
    let normal_text = if let Some(start_idx) = trimmed.find(&config.function_calls_start) {
        let text = trimmed[..start_idx].trim();
        if text.is_empty() {
            String::new()
        } else {
            text.to_string()
        }
    } else {
        String::new()
    };

    // Extract tool calls blocks
    let tool_calls = extract_tool_calls(trimmed, config)?;

    if tool_calls.is_empty() {
        // No valid tool calls found
        return Ok((vec![], Some(trimmed.to_string())));
    }

    Ok((tool_calls, Some(normal_text)))
}

/// Extract all tool calls from the DSML formatted text
fn extract_tool_calls(
    text: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut tool_calls = Vec::new();
    let regexes = get_dsml_regexes(config)?;

    // Find all function_calls blocks — the block regex captures the content
    // between start/end tags (non-greedy, dot-matches-newline).
    for block_match in regexes.block.captures_iter(text) {
        if let Some(block_content) = block_match.get(1) {
            let block = block_content.as_str();

            // Extract individual invokes from this block
            let invokes = extract_invokes(block, &regexes)?;
            tool_calls.extend(invokes);
        }
    }

    Ok(tool_calls)
}

/// Extract individual invoke blocks from function_calls content
fn extract_invokes(block: &str, regexes: &DsmlRegexes) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut invokes = Vec::new();

    // Matches: <｜DSML｜invoke name="function_name">..content..</｜DSML｜invoke>
    for invoke_match in regexes.invoke.captures_iter(block) {
        if let (Some(name_match), Some(content_match)) = (invoke_match.get(1), invoke_match.get(2))
        {
            let function_name = name_match.as_str().trim().to_string();
            let invoke_content = content_match.as_str();

            // Parse parameters from invoke content
            let parameters = parse_parameters(invoke_content, regexes)?;

            // Create tool call response
            let arguments_json = serde_json::to_string(&parameters)?;

            invokes.push(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
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
    regexes: &DsmlRegexes,
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut parameters = serde_json::Map::new();

    // Matches: <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
    for param_match in regexes.parameter.captures_iter(content) {
        if let (Some(name_match), Some(string_match), Some(value_match)) =
            (param_match.get(1), param_match.get(2), param_match.get(3))
        {
            let param_name = name_match.as_str().trim();
            let is_string = string_match.as_str() == "true";
            let param_value = value_match.as_str().trim();

            // Parse value based on string attribute
            let value = if is_string {
                // String type - use as-is
                serde_json::Value::String(param_value.to_string())
            } else {
                // Non-string type - parse as JSON
                serde_json::from_str(param_value).unwrap_or_else(|_| {
                    // Fallback to string if JSON parsing fails
                    serde_json::Value::String(param_value.to_string())
                })
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
            function_calls_start: "<｜DSML｜tool_calls>".to_string(),
            function_calls_end: "</｜DSML｜tool_calls>".to_string(),
            ..Default::default()
        }
    }

    #[test]
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
    // DeepSeek V4 coverage (see lib/parsers/TESTING.md for CASE.* taxonomy).
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
    //             Same class as Kimi K2 pre-DIS-1765. Recovery pattern: scan for
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
    #[test]
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

    #[test]
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let text = "<｜DSML｜function_calls><｜DSML｜invoke name=\"test\"></｜DSML｜invoke></｜DSML｜function_calls>more";
        let pos = find_tool_call_end_position_dsml(text, &config);
        assert_eq!(&text[pos..], "more");
    }

    /// `CASE.8` — streaming end-position lookup (V4 variant).
    #[test]
    fn test_find_tool_call_end_position_v4() {
        let config = get_v4_test_config();
        let text = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"test\"></｜DSML｜invoke></｜DSML｜tool_calls>more";
        let pos = find_tool_call_end_position_dsml(text, &config);
        assert_eq!(&text[pos..], "more");
    }

    #[test]
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

    #[test]
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

    #[test]
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
    #[test]
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
        assert_eq!(normal, Some("Let's check this.".to_string()));

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
    #[test]
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

    #[test]
    fn test_parse_with_normal_text() {
        let input = r#"Here's the result: <｜DSML｜function_calls>
<｜DSML｜invoke name="test">
<｜DSML｜parameter name="value" string="true">test</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let config = get_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(normal, Some("Here's the result:".to_string()));
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let input = "This is just normal text without any tool calls.";
        let config = get_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some(input.to_string()));
    }

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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
    #[test]
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
            Some(input),
            "Unrecovered payload should fall through to normal_text verbatim."
        );
    }

    /// `CASE.5` — multiple complete invokes, missing end fence.
    ///
    /// Even with multiple fully-formed invokes inside the start fence, the
    /// absence of the closing fence prevents the block regex from matching.
    /// All calls are dropped. If the parser ever gains partial-block
    /// recovery, this test will fail and force an intentional update.
    #[test]
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
    #[test]
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
    #[test]
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

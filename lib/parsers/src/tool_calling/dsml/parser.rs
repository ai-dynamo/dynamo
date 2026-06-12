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

    // Check for complete outer block start, or a bare invoke when the outer
    // wrapper opener is missing.
    if chunk.contains(start_token.as_str()) || chunk.contains(config.invoke_start_prefix.as_str()) {
        return true;
    }

    // Check for partial match at the end (streaming scenario).
    for token in [start_token, &config.invoke_start_prefix] {
        let chars: Vec<char> = token.chars().collect();
        for i in 1..chars.len() {
            let partial: String = chars[..i].iter().collect();
            if chunk.ends_with(&partial) {
                return true;
            }
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

/// Parse DSML formatted tool calls from a message.
///
/// Returns `(parsed_tool_calls, normal_text_content)`. `normal_text` is the
/// text BEFORE the first `<｜DSML｜tool_calls>` / `<｜DSML｜function_calls>`
/// start marker. Text between blocks, after the last block, and any
/// back-to-back-block content are all dropped — matching upstream vLLM
/// (`vllm/tool_parsers/deepseek_v4_tool_parser.py` and the V3.2 sibling),
/// which compute `content = model_output[:content_end]` where
/// `content_end = model_output.find(self.tool_call_start_token)`.
///
/// Per `tests/parity/README.md`: vLLM and SGLang both drop trailing text
/// after the wrapper across XML-style families; this aligns Dynamo to that
/// behavior. Cases: TOOLCALLING.batch.{2.b, 2.c, 8.b, 8.c, 8.d}.
pub fn try_tool_call_parse_dsml(
    message: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let trimmed = message.trim();

    // Early exit if no content
    if trimmed.is_empty() {
        return Ok((vec![], Some(String::new())));
    }

    let Some(start_idx) = trimmed.find(&config.block_start) else {
        if let Some(marker_idx) = first_orphan_dsml_marker_index(trimmed, config) {
            let marker_tail = &trimmed[marker_idx..];
            if marker_tail.starts_with(config.invoke_start_prefix.as_str())
                && (marker_tail.contains(config.block_end.as_str()) || config.allow_eof_recovery)
            {
                let tool_calls = extract_invokes(marker_tail, config)?;
                if !tool_calls.is_empty() {
                    tracing::warn!(
                        why = "bare_invoke_recovery",
                        recovered_calls = tool_calls.len(),
                        recovered_bytes = marker_tail.len(),
                        kept_prefix_bytes = marker_idx,
                        "DSML recovery: recovered complete bare invoke(s) without outer block_start"
                    );
                    return Ok((
                        tool_calls,
                        Some(trimmed[..marker_idx].trim_end().to_string()),
                    ));
                }
            }
            let stripped = &trimmed[marker_idx..];
            tracing::warn!(
                why = "DSML tool-call marker found without the outer block_start; dropping orphan marker tail so wire tags do not leak into normal_text",
                stripped_bytes = stripped.len(),
                "DSML strip (orphan markers)"
            );
            return Ok((vec![], Some(trimmed[..marker_idx].trim_end().to_string())));
        }
        return Ok((vec![], Some(trimmed.to_string())));
    };

    // Extract tool calls blocks. Finalize paths can opt into EOF recovery so
    // a missing outer block end still yields any complete inner invokes.
    let tool_calls = extract_tool_calls(trimmed, config)?;

    // Whether or not invokes parsed, normal_text is the prefix before the
    // first block_start — mirrors vLLM's success path. On no-invokes the
    // markup-leak warning still fires for the diagnostic trail.
    let pre_block_span = &trimmed[..start_idx];
    let pre_block_text = first_orphan_dsml_marker_index(pre_block_span, config)
        .filter(|idx| pre_block_span[*idx..].starts_with(config.invoke_start_prefix.as_str()))
        .map(|idx| pre_block_span[..idx].trim_end().to_string())
        .unwrap_or_else(|| pre_block_span.to_string());

    if tool_calls.is_empty() {
        // A block-start was detected but no valid invokes parsed. Do NOT leak
        // the DSML markup back to the client; emit a diagnostic with a prefix
        // of the failed block.
        //
        // Note: an unterminated block-start here means `block_regex` finds no
        // match at all, so any valid block *after* the unterminated one is
        // lost. This matches the pre-existing conservative P1-3 contract.
        let failed = &trimmed[start_idx..];
        let prefix: String = failed.chars().take(120).collect();
        tracing::warn!(
            why = "no_invokes_parsed",
            stripped_bytes = failed.len(),
            "DSML strip (recovery): block_start detected but extract_tool_calls returned 0 invokes; suppressing all bytes from block_start onward so tool-call markup never bleeds into normal_text. preview={:?}",
            prefix
        );
        return Ok((vec![], Some(pre_block_text)));
    }

    // Success path: prefix-only contract — everything from the first block_start
    // onward (the block(s) themselves plus any inter-block / trailing narration)
    // is stripped from normal_text. Mirrors vLLM's
    // `content = model_output[:content_end]`.
    let stripped = &trimmed[start_idx..];
    if !stripped.is_empty() {
        let preview: String = stripped.chars().take(120).collect();
        tracing::debug!(
            why = "prefix_only_contract",
            n_calls = tool_calls.len(),
            kept_prefix_bytes = pre_block_text.len(),
            stripped_bytes = stripped.len(),
            "DSML strip (success): kept prefix before first block_start; dropped parsed-block(s) + any inter-block / trailing narration. preview={:?}",
            preview
        );
    }

    Ok((tool_calls, Some(pre_block_text)))
}

fn first_orphan_dsml_marker_index(text: &str, config: &DsmlParserConfig) -> Option<usize> {
    [
        config.block_end.as_str(),
        config.invoke_start_prefix.as_str(),
        config.invoke_end.as_str(),
        config.parameter_prefix.as_str(),
        config.parameter_end.as_str(),
    ]
    .into_iter()
    .filter_map(|marker| text.find(marker))
    .min()
}

/// Extract all tool calls from DSML formatted text.
fn extract_tool_calls(
    text: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let mut tool_calls = Vec::new();
    let mut cursor = 0;

    while cursor < text.len() {
        let Some(rel_start) = text[cursor..].find(config.block_start.as_str()) else {
            if let Some((_, mut recovered)) =
                recover_orphan_invokes_in_span(&text[cursor..], config)?
            {
                tool_calls.append(&mut recovered);
            }
            break;
        };
        let abs_start = cursor + rel_start;
        if let Some((_, mut recovered)) =
            recover_orphan_invokes_in_span(&text[cursor..abs_start], config)?
        {
            tool_calls.append(&mut recovered);
        }

        let block_content_start = abs_start + config.block_start.len();
        let after_start = &text[block_content_start..];

        let (block, next_cursor) =
            if let Some(rel_end) = after_start.find(config.block_end.as_str()) {
                (
                    &after_start[..rel_end],
                    block_content_start + rel_end + config.block_end.len(),
                )
            } else if config.allow_eof_recovery {
                (&text[block_content_start..], text.len())
            } else {
                break;
            };

        let invokes = extract_invokes(block, config)?;
        tool_calls.extend(invokes);

        cursor = next_cursor;
    }

    Ok(tool_calls)
}

fn recover_orphan_invokes_in_span(
    span: &str,
    config: &DsmlParserConfig,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    let Some(marker_idx) = first_orphan_dsml_marker_index(span, config) else {
        return Ok(None);
    };
    let marker_tail = &span[marker_idx..];
    if !marker_tail.starts_with(config.invoke_start_prefix.as_str()) {
        return Ok(None);
    }
    if !marker_tail.contains(config.block_end.as_str()) && !config.allow_eof_recovery {
        return Ok(None);
    }

    let tool_calls = extract_invokes(marker_tail, config)?;
    if tool_calls.is_empty() {
        return Ok(None);
    }

    tracing::warn!(
        why = "bare_invoke_gap_recovery",
        recovered_calls = tool_calls.len(),
        recovered_bytes = marker_tail.len(),
        kept_prefix_bytes = marker_idx,
        "DSML recovery: recovered complete bare invoke(s) before a later outer block"
    );
    Ok(Some((
        span[..marker_idx].trim_end().to_string(),
        tool_calls,
    )))
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

    fn get_v4_recovery_test_config() -> DsmlParserConfig {
        DsmlParserConfig {
            allow_eof_recovery: true,
            ..get_v4_test_config()
        }
    }

    #[test] // helper
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
    // DEPRECATED(parser-fixture-duplicate): Legacy V4 coverage manifest.
    // The blackbox case mapping now lives in the YAML parser fixtures under
    // tests/parity/toolcalling/fixtures/deepseek_v4/ and the taxonomy in
    // lib/parsers/TOOLCALLING_CASES.md; keep this temporarily as a pointer while
    // the duplicate Rust tests are being retired.
    //
    // DeepSeek V4 coverage (see lib/parsers/TOOLCALLING_CASES.md for TOOLCALLING.* taxonomy).
    //
    // Covered by the V4 tests below (or by a shared DSML generic test):
    //   - TOOLCALLING.batch.1   single-call            (parsers.rs :: test_deepseek_v4_single_tool_call)
    //   - TOOLCALLING.batch.2   multi-calls            (test_parse_deepseek_v4_multiple_tool_calls)
    //   - TOOLCALLING.batch.3   no-call                (shared: test_parse_no_tool_calls)
    //   - TOOLCALLING.batch.4   malformed-args         (test_parse_deepseek_v4_malformed_json_value_falls_back_to_string,
    //                                      test_parse_deepseek_v4_missing_invoke_close_drops_call)
    //   - TOOLCALLING.batch.5   missing-end-token      (test_parse_deepseek_v4_missing_end_token{,_multiple_calls})
    //                                      — PINNED AS BROKEN: parser drops the call. See TODO below.
    //   - TOOLCALLING.batch.6   empty-args             (test_parse_deepseek_v4_no_parameters)
    //   - TOOLCALLING.batch.7   complex-args           (shared: test_parse_mixed_types_realistic, test_parse_nested_object_parameter,
    //                                      lib/llm/tests/test_streaming_tool_parsers :: ..._mixed_param_types_vllm,
    //                                      ..._special_chars_vllm)
    //   - TOOLCALLING.stream.3   streaming              (test_detect_tool_call_start_v4, test_find_tool_call_end_position_v4,
    //                                      test_streaming_chunk_boundary_split_v4,
    //                                      lib/llm/tests/test_streaming_tool_parsers :: ..._fragmented_tokens_vllm)
    //   - TOOLCALLING.batch.8   reasoning-plus-tool    (test_parse_reasoning_plus_tool_v4;
    //                                      lib/llm/tests/test_streaming_tool_parsers :: ..._with_tools_vllm)
    //   - TOOLCALLING.batch.3  reasoning-only         (test_parse_reasoning_only_no_tool_v4;
    //                                      reasoning/mod.rs :: test_deepseek_v4_detect_and_parse etc.)
    //   - FRONTEND.tool_choice  tool_choice            (lib/llm/tests/tool_choice.rs ::
    //                                      test_deepseek_v4_tool_choice_{auto,required_pins_current_behavior,
    //                                      named_correct_tool_passes,named_wrong_tool_filtered};
    //                                      parser-level invariant in
    //                                      test_parser_does_not_filter_by_tool_choice_v4;
    //                                      cross-parser tool_choice parametrisation work-item (tracked separately) covers full cross-parser parametrisation)
    //   - PIPELINE.finish_reason  finish-reason          (parser-level invariant in
    //                                      test_parser_output_independent_of_upstream_finish_v4;
    //                                      cross-parser stop/tool_calls/length mapping is
    //                                      cross-parser finish_reason mapping work-item (tracked separately); lib/llm/tests/test_streaming_tool_parsers
    //                                      covers ToolCalls / Stop on E2E fixtures)
    //   - TOOLCALLING.batch.8  interleaved-text       (test_parse_deepseek_v4_multiple_tool_calls prefix text;
    //                                      lib/llm/tests/test_streaming_tool_parsers :: ..._content_before_tool_vllm)
    //   - TOOLCALLING.batch.9  empty/null             (test_parse_empty_and_whitespace_inputs_v4)
    //   - TOOLCALLING.batch.10  duplicate-calls        (test_parse_duplicate_invokes_same_name_v4)
    //
    //   - TOOLCALLING.xml.1 / TOOLCALLING.xml.2  N/A — DSML carries per-parameter
    //                  string="true|false" type hints, so XML entity decoding
    //                  and schema-aware coercion don't apply.
    //   - TOOLCALLING.harmony.1 / TOOLCALLING.harmony.2 N/A — Harmony-only.
    //
    // EOF recovery:
    //   - TOOLCALLING.stream.4.a / TOOLCALLING.batch.5  Both the stream-finalize
    //             and batch/non-streaming finalize paths set
    //             `allow_eof_recovery=true` (via
    //             `detect_and_parse_tool_call_with_recovery`) and recover every
    //             complete <｜DSML｜invoke>...</｜DSML｜invoke> pair even when the
    //             outer close fence is absent at EOS, keeping the pre-block prose
    //             as normal_text and dropping any trailing invoke that was never
    //             closed. Only streaming early-exit keeps recovery disabled so it
    //             does not claim DSML calls before `</｜DSML｜tool_calls>` actually
    //             arrives (see test_parse_deepseek_v4_missing_end_token_without_recovery).
    //
    // TODO — bugs pinned, parser still needs to be fixed:
    //   - (TOOLCALLING.batch.5  FIXED: batch/non-streaming finalize now recovers
    //     complete invokes when </｜DSML｜tool_calls> is absent, matching the
    //     stream-finalize path. Same class as Kimi K2 pre-PR #8208.)
    //   - (TOOLCALLING.batch.4 missing-parameter-close & middle-invoke-truncation now
    //     pinned: see test_parse_deepseek_v4_missing_parameter_close_loses_param,
    //     test_parse_deepseek_v4_middle_invoke_truncation_corrupts_next.)
    // No customer-incident regression tests yet — V4 is hours old
    // (2026-04-24) and no bugs have been filed against it.
    // -------------------------------------------------------------------

    /// `TOOLCALLING.stream.3` — streaming start-token detection (V4 variant).
    #[test] // helper, TOOLCALLING.fmt.3 — V4 token variant
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

    #[test] // helper
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let text = "<｜DSML｜function_calls><｜DSML｜invoke name=\"test\"></｜DSML｜invoke></｜DSML｜function_calls>more";
        let pos = find_tool_call_end_position_dsml(text, &config);
        assert_eq!(&text[pos..], "more");
    }

    /// `TOOLCALLING.stream.3` — streaming end-position lookup (V4 variant).
    #[test] // helper, TOOLCALLING.fmt.3 — V4 token variant
    fn test_find_tool_call_end_position_v4() {
        let config = get_v4_test_config();
        let text = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"test\"></｜DSML｜invoke></｜DSML｜tool_calls>more";
        let pos = find_tool_call_end_position_dsml(text, &config);
        assert_eq!(&text[pos..], "more");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1 in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1, TOOLCALLING.batch.7.a in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml, tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1, TOOLCALLING.batch.7
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.a in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.2.yaml.
    #[test] // TOOLCALLING.batch.2
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

    /// `TOOLCALLING.batch.2` multi-calls + `TOOLCALLING.batch.8` interleaved-text (prefix text before the block).
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.c, TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.2, TOOLCALLING.fmt.3 — V4 variant
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

    /// `TOOLCALLING.batch.6` — empty args (no-parameter invoke).
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.6.a in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.6.yaml.
    #[test] // TOOLCALLING.batch.6, TOOLCALLING.fmt.3 — V4 variant
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.8
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

    #[test] // TOOLCALLING.batch.3
    fn test_parse_no_tool_calls() {
        let input = "This is just normal text without any tool calls.";
        let config = get_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some(input.to_string()));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.a in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.a in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.a in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.a in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
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

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.b in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
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
    fn test_multi_block_drops_inter_and_trailing_text() {
        // Two complete DSML blocks with text before, between, and after.
        // Both blocks must be parsed; only the pre-block prefix survives in
        // normal_content — matches vLLM (drops inter / trailing).
        let input = "pre <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"a\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> middle <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"b\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> tail";

        let config = get_v4_test_config();
        let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 2, "expected both blocks parsed");
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");

        let normal = normal.unwrap();
        assert_eq!(normal, "pre ", "only pre-block text survives: {:?}", normal);
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

        // After alignment to vLLM, normal_text is the prefix BEFORE the first
        // block_start only — trailing text after the block is dropped.
        let normal = normal.unwrap();
        assert_eq!(normal, "pre ", "only pre-block text survives: {:?}", normal);
        assert!(
            !normal.contains("<｜DSML｜tool_calls>"),
            "normal_content leaked block_start: {:?}",
            normal
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.a, TOOLCALLING.batch.9 in tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.7.yaml, tests/parity/toolcalling/fixtures/deepseek_v3_2/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.7, TOOLCALLING.batch.9
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
    // full mapping from TOOLCALLING.* → test. Each test's doc-comment names the
    // specific CASE it pins.

    /// `TOOLCALLING.batch.5` — missing end-token, streaming-safe path.
    ///
    /// Without EOF recovery, the parser must not claim a complete tool call
    /// before `</｜DSML｜tool_calls>` arrives. Streaming early-exit uses this
    /// path and keeps buffering until stream finalization.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.5, TOOLCALLING.fmt.3 — V4 variant
    fn test_parse_deepseek_v4_missing_end_token_without_recovery() {
        // Start fence + complete invoke, but no </｜DSML｜tool_calls>.
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>";

        let config = get_v4_test_config();
        let (calls, normal_text) = try_tool_call_parse_dsml(input, &config).unwrap();

        assert!(
            calls.is_empty(),
            "Streaming-safe DSML parser must wait for </｜DSML｜tool_calls> \
             instead of recovering early."
        );
        assert_eq!(
            normal_text.as_deref(),
            Some(""),
            "Pre-block text is empty here; raw DSML markup must not leak \
             into normal_text (post-hardening behavior)."
        );
    }

    /// `TOOLCALLING.batch.5` — multiple complete invokes, missing end fence.
    ///
    /// Even with multiple fully-formed invokes inside the start fence, the
    /// absence of the closing fence prevents the block regex from matching.
    /// All calls are dropped. If the parser ever gains partial-block
    /// recovery, this test will fail and force an intentional update.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.a, TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.2, TOOLCALLING.batch.5, TOOLCALLING.fmt.3
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

    /// `TOOLCALLING.stream.4.a` — missing end-token recovery at stream finalize.
    ///
    /// Stream finalization flips `allow_eof_recovery=true`, treating an
    /// unterminated outer block as extending to EOF and recovering complete
    /// inner invokes. Batch/non-streaming aggregate parsing remains strict.
    #[test] // TOOLCALLING.stream.4.a — V4 variant
    fn test_parse_deepseek_v4_missing_end_token_with_recovery() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>";

        let config = get_v4_recovery_test_config();
        let (calls, normal_text) = try_tool_call_parse_dsml(input, &config).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(normal_text.as_deref(), Some(""));
        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], "NYC");
    }

    /// Stream-finalize recovery with multiple complete invokes and no outer end fence.
    #[test]
    fn test_parse_deepseek_v4_missing_end_token_multiple_calls_with_recovery() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"a\">\n\
<｜DSML｜parameter name=\"x\" string=\"true\">1</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"b\">\n\
<｜DSML｜parameter name=\"y\" string=\"true\">2</｜DSML｜parameter>\n\
</｜DSML｜invoke>";

        let config = get_v4_recovery_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();

        assert_eq!(calls.len(), 2);
        let (name1, args1) = extract_name_and_args(calls[0].clone());
        let (name2, args2) = extract_name_and_args(calls[1].clone());
        assert_eq!(name1, "a");
        assert_eq!(args1["x"], "1");
        assert_eq!(name2, "b");
        assert_eq!(args2["y"], "2");
    }

    /// `TOOLCALLING.batch.4` — malformed JSON in a `string="false"` parameter value falls back
    /// to a string. `parse_parameters` explicitly swallows the serde error
    /// (unwrap_or_else → Value::String). Pin the fallback so removing it
    /// (which would cause the whole call to 500 on ragged-edge JSON) is a
    /// deliberate change.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.b in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.4.yaml.
    #[test] // TOOLCALLING.batch.4, TOOLCALLING.fmt.3
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

    /// `TOOLCALLING.batch.4` — malformed invoke (missing `</｜DSML｜invoke>` but block fences
    /// intact). The invoke regex requires its own close tag, so the call is
    /// silently dropped. Pin the behavior.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.d in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.4.yaml.
    #[test] // TOOLCALLING.batch.4, TOOLCALLING.fmt.3
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

    /// `TOOLCALLING.batch.4` — malformed invoke (missing `</｜DSML｜parameter>` close tag).
    /// The parameter regex requires its closing tag; if a parameter never
    /// closes before `</｜DSML｜invoke>`, the parameter is silently lost
    /// while the call itself still parses. Pin the partial behavior.
    ///
    /// TODO(TOOLCALLING.batch.4) — BUG, NEEDS FIX: parser silently loses the parameter
    /// and ships an under-specified call to the user. The fix should keep
    /// the raw value up to `</｜DSML｜invoke>`. Flip this test once fixed.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.d in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.4.yaml.
    #[test] // TOOLCALLING.batch.4, TOOLCALLING.fmt.3
    fn test_parse_deepseek_v4_missing_parameter_close_loses_param() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"test\">\n\
<｜DSML｜parameter name=\"x\" string=\"true\">value\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";

        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        // PIN_ME: replace with observed behavior after first run.
        assert_eq!(calls.len(), 1);
        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "test");
        assert!(
            args.get("x").is_none(),
            "Expected 'x' to be dropped because </｜DSML｜parameter> is missing; \
             got args={args}"
        );
    }

    /// `TOOLCALLING.batch.4` — middle-invoke truncation. If invoke A is missing its
    /// `</｜DSML｜invoke>` and invoke B follows inside the same outer block,
    /// A's body bleeds through into B (regex non-greedy match consumes B's
    /// markup). Pin the corruption.
    ///
    /// TODO(TOOLCALLING.batch.4) — BUG, NEEDS FIX: A swallows B's parameters and B is
    /// silently dropped — caller receives wrong args for A and never sees
    /// B at all. Fix: anchor on `<｜DSML｜invoke name=` to re-sync between
    /// invokes. Flip this test once fixed.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.d in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.4.yaml.
    #[test] // TOOLCALLING.batch.4, TOOLCALLING.fmt.3
    fn test_parse_deepseek_v4_middle_invoke_truncation_corrupts_next() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"a\">\n\
<｜DSML｜parameter name=\"x\" string=\"true\">1</｜DSML｜parameter>\n\
<｜DSML｜invoke name=\"b\">\n\
<｜DSML｜parameter name=\"y\" string=\"true\">2</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";

        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        // Today: invoke A absorbs invoke B's parameter (regex bleed) and B is
        // dropped entirely. Wrong-but-stable; pin so a fix is intentional.
        assert_eq!(calls.len(), 1, "B is dropped; A is the lone survivor");
        let (name, args) = extract_name_and_args(calls[0].clone());
        assert_eq!(name, "a");
        assert_eq!(args["x"], "1", "A's own parameter still parses correctly");
        assert_eq!(
            args["y"], "2",
            "BUG: B's parameter bleeds into A because A's body match runs \
             past the missing </｜DSML｜invoke> until B's close tag"
        );
    }

    /// `TOOLCALLING.stream.3` — streaming chunk-boundary split. Token-by-token assembly:
    /// the start-token detector and end-position lookup must each tolerate
    /// the block boundary landing in the middle of a multi-byte fence.
    #[test] // TOOLCALLING.stream.3
    fn test_streaming_chunk_boundary_split_v4() {
        let config = get_v4_test_config();
        // Detector should fire on a partial start fence (one char short).
        assert!(detect_tool_call_start_dsml("<｜DSML｜tool_call", &config));
        // And on an empty buffer that ends with the very first char of the fence.
        assert!(detect_tool_call_start_dsml("<", &config));
        // End-position lookup must return chunk.len() when the end fence
        // hasn't arrived yet — caller is expected to keep buffering.
        let partial = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"a\">\n";
        assert_eq!(
            find_tool_call_end_position_dsml(partial, &config),
            partial.len(),
            "Partial chunk without close fence must report end=len so caller buffers more"
        );
    }

    /// `TOOLCALLING.batch.8` — paired reasoning + tool in same response. DSv4 emits
    /// `<think>...</think>` before the DSML block; the tool parser is
    /// concerned only with the DSML, but normal text must round-trip
    /// the reasoning markup verbatim for the reasoning parser to pick up.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.8
    fn test_parse_reasoning_plus_tool_v4() {
        let input = "<think>Let me check the weather.</think>\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";
        let config = get_v4_test_config();
        let (calls, normal_text) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let normal = normal_text.unwrap_or_default();
        assert!(
            normal.contains("<think>") && normal.contains("</think>"),
            "Reasoning markup must be preserved in normal_text for the \
             downstream reasoning parser; got {:?}",
            normal
        );
    }

    /// `TOOLCALLING.batch.3` — reasoning only (think tags, no tool call). Parser must
    /// return zero calls and pass the entire input through as normal text.
    #[test] // TOOLCALLING.batch.3
    fn test_parse_reasoning_only_no_tool_v4() {
        let input = "<think>Just thinking out loud, no tools needed.</think>";
        let config = get_v4_test_config();
        let (calls, normal_text) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal_text.as_deref(), Some(input));
    }

    /// Parser-level invariant: the dsml parser does NOT filter by
    /// `tool_choice` — it returns every well-formed invoke, and the jail /
    /// response builder above this layer is responsible for filtering per
    /// `tool_choice=named`/`required`/`none`. Real FRONTEND.tool_choice coverage lives
    /// at the integration layer (`lib/llm/tests/tool_choice.rs`).
    #[test]
    fn test_parser_does_not_filter_by_tool_choice_v4() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"get_time\">\n\
<｜DSML｜parameter name=\"tz\" string=\"true\">EST</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";
        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 2);
    }

    /// Parser-level invariant: the dsml parser is byte-stable — it doesn't
    /// see `finish_reason` and produces the same output for any upstream
    /// stream-end reason. Real PIPELINE.finish_reason coverage (stop / tool_calls / length
    /// mapping) lives in `lib/llm/tests/test_streaming_tool_parsers.rs` and
    /// belongs in cross-parser finish_reason mapping work-item (tracked separately).
    #[test]
    fn test_parser_output_independent_of_upstream_finish_v4() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";
        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
    }

    /// `TOOLCALLING.batch.9` — empty / null content variants. Pin behavior on truly
    /// empty bytes and whitespace-only inputs.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.9 in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.9
    fn test_parse_empty_and_whitespace_inputs_v4() {
        let config = get_v4_test_config();
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (calls, normal) = try_tool_call_parse_dsml(input, &config).unwrap();
            assert!(
                calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            // Empty input fast-path returns Some(""); other whitespace is
            // trimmed before the search and the no-block branch returns the
            // trimmed (also empty) string.
            assert_eq!(
                normal.as_deref(),
                Some(""),
                "Empty/whitespace input collapses to empty normal_text"
            );
        }
    }

    /// `TOOLCALLING.batch.10` — duplicate calls (same invoke name twice in one block).
    /// Universal gap noted in the test taxonomy; first DSML coverage.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.10 in tests/parity/toolcalling/fixtures/deepseek_v4/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.10
    fn test_parse_duplicate_invokes_same_name_v4() {
        let input = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">LA</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";
        let config = get_v4_test_config();
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(
            calls.len(),
            2,
            "Both duplicate-name invokes must be returned"
        );
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");
        assert_ne!(
            calls[0].id, calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let (_, args0) = extract_name_and_args(calls[0].clone());
        let (_, args1) = extract_name_and_args(calls[1].clone());
        assert_eq!(args0["city"], "NYC");
        assert_eq!(args1["city"], "LA");
    }

    #[test] // helper — serialized arguments must keep the model-emitted parameter order
    fn test_arguments_preserve_model_emitted_param_order() {
        let config = get_test_config();
        // path, old_str, command is deliberately NOT alphabetical.
        let input = "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"file_editor\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/a</｜DSML｜parameter>\n<｜DSML｜parameter name=\"old_str\" string=\"true\">x</｜DSML｜parameter>\n<｜DSML｜parameter name=\"command\" string=\"true\">replace</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>";
        let (calls, _) = try_tool_call_parse_dsml(input, &config).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].function.arguments, r#"{"path":"/a","old_str":"x","command":"replace"}"#,
            "serde_json::Map must preserve insertion order (needs preserve_order)"
        );
    }
}

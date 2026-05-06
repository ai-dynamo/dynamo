// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::ToolDefinition;
use super::super::json::base_json_parser::try_repair_truncated_json;
use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use openai_harmony::chat::{Content::Text, Role};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

static COMMENTARY_BLOCK_REGEX: OnceLock<Regex> = OnceLock::new();

/// Regex fallback used only when `openai_harmony`'s tokenizer rejects the
/// input — alternative on this path is silent-drop. Worst case is missing
/// a call, never fabricating one (full structural signature required).
fn commentary_block_regex() -> &'static Regex {
    COMMENTARY_BLOCK_REGEX.get_or_init(|| {
        // Name is `[\w.\-]+` (alphanumeric / dot / hyphen / underscore).
        // Between name and `<|message|>` we tolerate optional
        // `<|constrain|>json` and whitespace by using non-greedy `.*?`.
        // Args end at either `<|call|>` (normal close) or end-of-string
        // (`\z`, the bare-envelope PARSER.batch.5 variant where the model never
        // emitted `<|call|>` before EOS / max_tokens).
        Regex::new(
            r"(?s)<\|channel\|>commentary to=functions\.(?P<name>[\w.\-]+).*?<\|message\|>(?P<args>.*?)(?:<\|call\|>|\z)",
        )
        .expect("commentary block regex")
    })
}

/// Extract calls via regex when harmony's strict tokenizer rejects the input
/// (truncated JSON, multiple back-to-back commentary blocks, etc.).
/// Returns (calls, residual_text) where residual_text is everything not
/// consumed by a matched commentary block — preserved so analysis prose and
/// non-tool suffixes aren't dropped.
fn extract_calls_via_regex(text: &str) -> (Vec<ToolCallResponse>, String) {
    let mut out = Vec::new();
    let mut residual = String::new();
    let mut cursor = 0;
    for (i, cap) in commentary_block_regex().captures_iter(text).enumerate() {
        let m = cap.get(0).expect("regex match has full span");
        residual.push_str(&text[cursor..m.start()]);
        cursor = m.end();

        let name = cap.name("name").map(|x| x.as_str()).unwrap_or("");
        let raw_args = cap.name("args").map(|x| x.as_str().trim()).unwrap_or("{}");
        if name.is_empty() {
            continue;
        }
        let args_json = match serde_json::from_str::<Value>(raw_args) {
            Ok(v) => serde_json::to_string(&v).unwrap_or_else(|_| raw_args.to_string()),
            Err(_) => match try_repair_truncated_json(raw_args)
                .and_then(|r| serde_json::from_str::<Value>(&r).ok())
            {
                Some(v) => serde_json::to_string(&v).unwrap_or_else(|_| raw_args.to_string()),
                None => raw_args.to_string(),
            },
        };
        out.push(ToolCallResponse {
            id: format!("call-{}", i + 1),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: args_json,
            },
        });
    }
    residual.push_str(&text[cursor..]);
    (out, residual.trim().to_string())
}

static GLOBAL_HARMONY_GPTOSS_ENCODING: tokio::sync::OnceCell<
    Result<HarmonyEncoding, anyhow::Error>,
> = tokio::sync::OnceCell::const_new();

pub async fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING
        .get_or_init(|| async {
            tokio::task::spawn_blocking(|| {
                load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            })
            .await
            .map_err(anyhow::Error::msg)
            .flatten()
        })
        .await
}

/// Parse tool calls from a complete Harmony Format text chunk using direct token parsing.
///
/// This function is optimized for parsing complete text chunks where the entire content
/// is available at once. It uses `parse_messages_from_completion_tokens` to directly
/// parse all tokens into Harmony Format messages, then extracts tool calls from messages
/// with the "commentary" channel and "functions.*" recipients.
///
/// This function doesn't perform start token detection
/// or token-by-token streaming, making it more efficient for complete chunks.
///
/// # Arguments
/// * `text` - The full Harmony-format string to be parsed, excluding any trailing stop tokens.
///   Example:
///   `<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}`
/// * `_config` - Parser configuration (currently unused but kept for API consistency)
///
/// # Returns
/// * `Ok((tool_calls, normal_text))` - Tuple containing extracted tool calls and any normal text
/// * `Err(e)` - If parsing fails due to encoding or tokenization errors
pub async fn parse_tool_calls_harmony_complete(
    text: &str,
    config: &JsonParserConfig,
    _tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let enc = match get_harmony_encoding().await.as_ref() {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!("Failed to load harmony encoding: {e}. Tool calls will not be parsed.");
            return Ok((vec![], Some(text.to_string())));
        }
    };

    // // Encode the text into tokens using harmony encoding
    let tokens: Vec<u32> = enc.tokenizer().encode_with_special_tokens(text);
    let messages = match enc.parse_messages_from_completion_tokens(tokens, Some(Role::Assistant)) {
        Ok(messages) => messages,
        Err(e) => {
            tracing::debug!(
                "Failed to parse messages from completion tokens: {e}. Falling back to regex extraction."
            );
            // Recovery: harmony rejects parallel commentary blocks and
            // truncated JSON. Gated on `allow_eof_recovery` so streaming
            // jails (where the tokenizer often rejects mid-chunk before all
            // tokens have arrived) don't extract a partial call.
            if config.allow_eof_recovery {
                let (calls, residual) = extract_calls_via_regex(text);
                if !calls.is_empty() {
                    return Ok((calls, Some(residual)));
                }
            }
            return Ok((vec![], Some(text.to_string())));
        }
    };

    let mut normal_text = String::new();

    let mut res = Vec::with_capacity(messages.len());
    let mut call_idx = 0; // Index of the tool call

    for message in messages.iter() {
        if message.author.role != Role::Assistant {
            continue;
        }

        let channel = message.channel.as_deref();
        let recipient = message.recipient.as_deref().unwrap_or_default();

        // Handle commentary channel
        if channel == Some("commentary") && recipient.starts_with("functions.") {
            let Some(fname) = message
                .recipient
                .as_ref()
                .and_then(|r| r.split('.').nth(1))
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
            else {
                continue;
            };

            let args = match message.content.first() {
                Some(Text(text)) => {
                    let trimmed = text.text.trim();
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(value) => value,
                        Err(_) if config.allow_eof_recovery => {
                            // Truncation recovery: balance unclosed strings /
                            // braces (max_tokens / EOS pattern) and retry.
                            // Gated so streaming early-exit doesn't extract
                            // a partial call with synthesized closers.
                            try_repair_truncated_json(trimmed)
                                .and_then(|r| serde_json::from_str::<Value>(&r).ok())
                                .unwrap_or(Value::Null)
                        }
                        Err(_) => Value::Null,
                    }
                }
                _ => {
                    Value::Null // Set args to null if it's not a text content
                }
            };
            // Add tool call to result if args is valid JSON
            if !args.is_null() {
                call_idx += 1;
                res.push(ToolCallResponse {
                    id: format!("call-{}", call_idx),
                    tp: ToolCallType::Function,
                    function: CalledFunction {
                        name: fname.to_string(),
                        // Safety: `Value::Object` is always valid JSON, so serialization cannot fail
                        arguments: serde_json::to_string(&args).unwrap(),
                    },
                });
            }
        // Handle reasoning(analysis) channel
        } else if channel == Some("analysis") {
            normal_text.push_str(match &message.content[0] {
                Text(t) => &t.text,
                _ => "",
            });
        }
    }
    Ok((res, Some(normal_text.to_string())))
}

pub fn detect_tool_call_start_harmony(
    chunk: &str,
    config: &JsonParserConfig,
    strict: bool,
) -> bool {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return false;
    }

    if strict {
        // Check for complete start tokens first
        let has_complete_token = config
            .tool_call_start_tokens
            .iter()
            .any(|token| !token.is_empty() && trimmed.contains(token));

        if has_complete_token {
            return true;
        }

        // Check for partial start tokens (streaming scenario)
        // This handles cases where start tokens are split across multiple chunks
        config.tool_call_start_tokens.iter().any(|token| {
            if token.is_empty() {
                return false;
            }
            // Check if the chunk could be a prefix of this start token
            // Handle Unicode character boundaries properly
            for i in 1..=token.chars().count() {
                if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                    let prefix_str = &prefix[..prefix.len()];
                    if trimmed == prefix_str || trimmed.ends_with(prefix_str) {
                        return true;
                    }
                }
            }
            false
        })
    } else {
        // Non-strict mode: check complete tokens and some heuristics
        let has_complete_token = config
            .tool_call_start_tokens
            .iter()
            .any(|token| !token.is_empty() && trimmed.contains(token));

        if has_complete_token {
            return true;
        }

        // Check for partial start tokens or known patterns
        let has_partial_token = config.tool_call_start_tokens.iter().any(|token| {
            if token.is_empty() {
                return false;
            }
            // Check if the chunk could be a prefix of this start token
            // Handle Unicode character boundaries properly
            for i in 1..=token.chars().count() {
                if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                    let prefix_str = &prefix[..prefix.len()];
                    if trimmed == prefix_str || trimmed.ends_with(prefix_str) {
                        return true;
                    }
                }
            }
            false
        });

        has_partial_token || trimmed.contains("<|channel|>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[tokio::test] // PARSER.batch.1, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_complete_basic() {
        let text = r#"<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"format":"celsius","location":"San Francisco"}"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["format"], "celsius");
    }

    #[tokio::test] // PARSER.batch.4, PARSER.harmony.2
    async fn test_parse_tools_harmony_without_start_token() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some(text.trim().to_string()));
        assert_eq!(tool_calls.len(), 0);
    }

    #[tokio::test] // PARSER.batch.7, PARSER.batch.8, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_with_multi_args() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco", "unit":"fahrenheit"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(
            normal_content,
            Some("Need to use function get_current_weather.".to_string())
        );
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test] // PARSER.batch.8, PARSER.batch.8, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_with_normal_text() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(
            normal_content,
            Some("Need to use function get_current_weather.".to_string())
        );
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
    }

    // Harmony's `<|call|>` plays the role of an outer end-token. When
    // max_tokens fires before it lands, the existing `analysis ... <|end|>`
    // envelope still gives the parser enough context to recover the call —
    // this test pins that recovery behavior. The bare-envelope variant
    // (no preceding analysis block) currently does NOT recover, but adding
    // a test for that is a parser-change discussion.
    // Pin current behavior on two back-to-back commentary blocks. The
    // harmony parser today does NOT extract both calls — the second
    // `<|start|>assistant<|channel|>commentary` block is left in normal
    // content. Same failure class as PARSER.batch.5: parser drops in-flight work,
    // customer sees HTTP 200 with fewer tool_calls than the model emitted.
    // Promoting this to recovery is a parser change.
    #[tokio::test] // PARSER.batch.2 — gpt-oss
    async fn test_parse_harmony_multiple_calls_recovers() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x":1}<|call|><|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json<|message|>{"y":2}<|call|>"#;
        let (tool_calls, _normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 2);
        let (n0, a0) = extract_name_and_args(tool_calls[0].clone());
        let (n1, a1) = extract_name_and_args(tool_calls[1].clone());
        assert_eq!(n0, "a");
        assert_eq!(a0["x"], 1);
        assert_eq!(n1, "b");
        assert_eq!(a1["y"], 2);
    }

    // Pin current behavior on truncated JSON args. harmony today drops the
    // call entirely rather than falling back to a string-form arguments or
    // surfacing an explicit error.
    #[tokio::test] // PARSER.batch.4 — gpt-oss
    async fn test_parse_harmony_truncated_json_recovers() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC<|call|>"#;
        let (tool_calls, _normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "NYC");
    }

    // Bare-envelope PARSER.batch.5: no preceding `analysis` block, no `<|call|>`
    // at the end. harmony's tokenizer rejects this; the regex fallback
    // accepts EOS as a synthetic close.
    #[tokio::test] // PARSER.batch.5 — gpt-oss
    async fn test_parse_harmony_bare_envelope_no_call_token_recovers() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}"#;
        let (tool_calls, _normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "NYC");
    }

    // The regex fallback must preserve any non-tool spans (prose before
    // the call, suffix after `<|call|>`) as `normal_text`, not zero them.
    #[tokio::test]
    async fn test_parse_harmony_regex_fallback_preserves_residual_text() {
        let text = r#"PREFIX <|start|>assistant<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x":1}<|call|> SUFFIX"#;
        let (tool_calls, normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let normal = normal.unwrap_or_default();
        assert!(
            normal.contains("PREFIX"),
            "normal must keep prefix: {normal:?}"
        );
        assert!(
            normal.contains("SUFFIX"),
            "normal must keep suffix: {normal:?}"
        );
    }

    #[tokio::test] // PARSER.batch.4, PARSER.batch.5, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_without_call_token() {
        let text = r#"<|channel|>analysis<|message|>We need to call get_weather function. The user asks "What's the weather like in San Francisco in Celsius?" So location: "San Francisco, CA" unit: "celsius". Let's call function.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco, CA","unit":"celsius"}"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("We need to call get_weather function. The user asks \"What's the weather like in San Francisco in Celsius?\" So location: \"San Francisco, CA\" unit: \"celsius\". Let's call function.".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "celsius");
    }

    /// Parser-level invariant: the harmony parser is byte-stable — it
    /// doesn't see `finish_reason` and produces the same output regardless
    /// of the upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop /
    /// tool_calls / length mapping) lives in
    /// `lib/llm/tests/test_streaming_tool_parsers.rs` and belongs in the
    /// cross-parser finish_reason mapping work-item (tracked separately).
    #[tokio::test]
    async fn test_harmony_parser_output_independent_of_upstream_finish() {
        let text = r#"<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"NYC"}"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
    }

    /// PARSER.batch.6 — empty args. A no-arg harmony call (`{}`) must still surface
    /// the function name.
    #[tokio::test] // PARSER.batch.6 — gpt-oss
    async fn test_parse_harmony_empty_args() {
        let text =
            r#"<|channel|>commentary to=functions.current_time <|constrain|>json<|message|>{}"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "current_time");
        assert_eq!(args, serde_json::json!({}));
    }

    /// PARSER.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls. Unlike the
    /// XML/JSON parsers (which trim whitespace down to `Some("")`), the
    /// harmony parser passes the input verbatim through to normal_text —
    /// pin that distinction here.
    #[tokio::test] // PARSER.batch.9 — gpt-oss
    async fn test_parse_harmony_empty_and_whitespace_inputs() {
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (tool_calls, normal) =
                parse_tool_calls_harmony_complete(input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(
                tool_calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(*input),
                "harmony passes empty/whitespace input verbatim to normal_text (input={:?})",
                input
            );
        }
    }

    /// PARSER.batch.10 — duplicate calls (same function name twice). Two
    /// back-to-back commentary blocks for the same function. Pin
    /// parser-level behavior — both calls returned with distinct ids
    /// and distinct args.
    #[tokio::test] // PARSER.batch.10 — gpt-oss
    async fn test_parse_harmony_duplicate_calls_same_name() {
        let text = r#"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"LA"}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(
            tool_calls.len(),
            2,
            "Both duplicate-name calls must be returned"
        );
        assert_ne!(
            tool_calls[0].id, tool_calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let (name0, args0) = extract_name_and_args(tool_calls[0].clone());
        let (name1, args1) = extract_name_and_args(tool_calls[1].clone());
        assert_eq!(name0, "get_weather");
        assert_eq!(name1, "get_weather");
        assert_eq!(args0["city"], "NYC");
        assert_eq!(args1["city"], "LA");
    }
}

#[cfg(test)]
mod detect_parser_tests {
    use super::*;

    #[test] // helper
    fn test_detect_tool_call_start_harmony_chunk_with_tool_call_start_token() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_harmony(text, &config, false);
        assert!(result);
    }

    #[test] // helper
    fn test_detect_tool_call_start_harmony_chunk_without_tool_call_start_token() {
        // This is a warkaround for now. Right now everything is treated as tool call start token.
        // We need to improve this in the future.
        let text = r#"<|channel|>commentary to=functions.get_current_weather"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_harmony(text, &config, false);
        assert!(result);
    }

    #[test] // helper, PARSER.stream.3
    fn test_detect_tool_call_start_harmony_partial_tokens() {
        // Test partial token detection for streaming scenarios
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };

        // Test various partial prefixes in strict mode
        assert!(
            detect_tool_call_start_harmony("<", &config, true),
            "'<' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|", &config, true),
            "'<|' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|start|>", &config, true),
            "'<|start|>' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|start|>assistant", &config, true),
            "'<|start|>assistant' should be detected as potential start"
        );

        // Test that unrelated text is not detected in strict mode
        assert!(
            !detect_tool_call_start_harmony("hello world", &config, true),
            "'hello world' should not be detected in strict mode"
        );
        assert!(
            !detect_tool_call_start_harmony("xyz", &config, true),
            "'xyz' should not be detected in strict mode"
        );
    }
}

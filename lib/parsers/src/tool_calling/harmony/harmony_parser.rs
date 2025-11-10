// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use openai_harmony::chat::{Content::Text, Role};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use serde_json::Value;

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
    _config: &JsonParserConfig,
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
                "Failed to parse messages from completion tokens: {e}. Tool calls will not be parsed."
            );
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
                Some(Text(text)) => match serde_json::from_str::<Value>(text.text.trim()) {
                    Ok(value) => value,
                    Err(_) => {
                        Value::Null // Set args to null if it's not valid JSON
                    }
                },
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

    #[tokio::test]
    async fn test_parse_tool_calls_harmony_complete_basic() {
        let text = r#"<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"format":"celsius","location":"San Francisco"}"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default())
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["format"], "celsius");
    }

    #[tokio::test]
    async fn test_parse_tools_harmony_without_start_token() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default())
                .await
                .unwrap();
        assert_eq!(normal_content, Some(text.trim().to_string()));
        assert_eq!(tool_calls.len(), 0);
    }

    #[tokio::test]
    async fn test_parse_tool_calls_harmony_with_multi_args() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco", "unit":"fahrenheit"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default())
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

    #[tokio::test]
    async fn test_parse_tool_calls_harmony_with_normal_text() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default())
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

    #[tokio::test]
    async fn test_parse_tool_calls_harmony_without_call_token() {
        let text = r#"<|channel|>analysis<|message|>We need to call get_weather function. The user asks "What's the weather like in San Francisco in Celsius?" So location: "San Francisco, CA" unit: "celsius". Let's call function.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco, CA","unit":"celsius"}"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default())
                .await
                .unwrap();
        assert_eq!(normal_content, Some("We need to call get_weather function. The user asks \"What's the weather like in San Francisco in Celsius?\" So location: \"San Francisco, CA\" unit: \"celsius\". Let's call function.".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "celsius");
    }
}

#[cfg(test)]
mod detect_parser_tests {
    use super::*;

    #[test]
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

    #[test]
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

    #[test]
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

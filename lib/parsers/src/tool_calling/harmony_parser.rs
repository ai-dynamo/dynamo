// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use openai_harmony::StreamableParser;
use openai_harmony::chat::{Content::Text, Role};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use serde_json::Value;
use std::sync::OnceLock;

static GLOBAL_HARMONY_GPTOSS_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> =
    OnceLock::new();

pub fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING
        .get_or_init(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
}

/// Parse tool calls from Harmony Format text
/// <|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
pub fn parse_tool_calls_harmony(
    text: &str,
    config: &JsonParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let trimmed = text.trim();

    // Check if tool call start tokens are present, if not return everything as normal text
    // Start Token: "<|start|>assistant<|channel|>commentary" should be present in the text if tool calls are present
    // End Token: "<|call|>"
    if !config
        .tool_call_start_tokens
        .iter()
        .any(|token| trimmed.contains(token))
    {
        return Ok((vec![], Some(trimmed.to_string())));
    }

    let enc = match get_harmony_encoding().as_ref() {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!("Failed to load harmony encoding: {e}. Tool calls will not be parsed.");
            return Ok((vec![], Some(text.to_string())));
        }
    };

    // Encode the text into tokens using harmony encoding
    let tokens = enc.tokenizer().encode_with_special_tokens(text);

    // Create StreamableParser to process each token and create Harmony Format messages
    // Set Role to Assistant because we are parsing tool calls from an assistant message
    let mut parser = match StreamableParser::new(enc.clone(), Some(Role::Assistant)) {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(
                "Failed to create harmony streamable parser: {e}. Tool calls will not be parsed."
            );
            return Ok((vec![], Some(text.to_string())));
        }
    };

    // Process each token to create Harmony Format messages
    for token in tokens {
        if parser.process(token).is_err() {
            // Skip the token if it causes an error. Some special tokens are not supported by the parser.
            continue;
        }
    }

    // Get the Harmony Format messages
    let messages = parser.messages();

    let mut normal_text = String::new();

    let mut res = Vec::with_capacity(messages.len());

    // Iteratate through messages and extract tool calls if there
    // For tool call, role should be Assistant, channel should be commentary and recipient should start with functions.
    //     Message {
    //    author: Author {
    //        role: Assistant,
    //        name: None
    //    },
    //    recipient: Some("functions.get_current_weather"),
    //    content: [
    //        Text(
    //            TextContent {
    //                text: "{\"location\":\"San Francisco\"}"
    //            }
    //        )
    //    ],
    //    channel: Some("commentary"),
    //    content_type: Some("<|constrain|>json")
    for (idx, message) in messages.iter().enumerate() {
        if message.author.role == Role::Assistant
            && message.channel.as_deref() == Some("commentary")
            && message
                .recipient
                .as_deref()
                .unwrap_or_default()
                .starts_with("functions.")
        {
            let fname = message
                .recipient
                .as_ref()
                .and_then(|r| r.split('.').nth(1))
                .unwrap_or("")
                .to_string();

            let args = match &message.content[0] {
                Text(text) => match serde_json::from_str::<Value>(text.text.trim()) {
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
                res.push(ToolCallResponse {
                    id: format!("call-{}", idx + 1),
                    tp: ToolCallType::Function,
                    function: CalledFunction {
                        name: fname.to_string(),
                        // Safety: `Value::Object` is always valid JSON, so serialization cannot fail
                        arguments: serde_json::to_string(&args).unwrap(),
                    },
                });
            }
        }
        if message.author.role == Role::Assistant && message.channel.as_deref() == Some("analysis")
        {
            normal_text.push_str(match &message.content[0] {
                Text(t) => &t.text,
                _ => "",
            });
        }
    }
    Ok((res, Some(normal_text.to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[test]
    fn test_parse_tool_calls_harmony_basic() {
        let text = r#"
<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json
<|message|>{"location":"San Francisco"}<|call|>
"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
    }

    #[test]
    fn test_parse_tools_harmony_without_start_token() {
        let text = r#"
<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
<|message|>{"location":"San Francisco"}<|call|>
"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).unwrap();
        assert_eq!(normal_content, Some(text.trim().to_string()));
        assert_eq!(tool_calls.len(), 0);
    }

    #[test]
    fn test_parse_tool_calls_harmony_with_multi_args() {
        let text = r#"
        <|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
        <|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json
        <|message|>{"location":"San Francisco", "unit":"fahrenheit"}<|call|>
        "#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_parse_tool_calls_harmony_with_normal_text() {
        let text = r#"
        <|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
        <|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json
        <|message|>{"location":"San Francisco"}<|call|>
        "#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).unwrap();
        assert_eq!(
            normal_content,
            Some("Need to use function get_current_weather.".to_string())
        );
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use openai_harmony::StreamableParser;
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

/// Parse tool calls from Harmony Format text
/// <|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
pub async fn parse_tool_calls_harmony(
    text: &str,
    config: &JsonParserConfig,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let mut trimmed = text.trim().to_string();
    let original_text = trimmed.clone();

    // Check if tool call start tokens are present, if not return everything as normal text
    // Start Token: "<|start|>assistant<|channel|>commentary" should be present in the text if tool calls are present
    // End Token: "<|call|>"
    if !detect_tool_call_start_harmony(text, config, true) {
        return Ok((vec![], Some(trimmed)));
    }

    // Workaround to add <|call|> token to the end of the text if it is not present. Otherwise, StreamableParser will not be able to parse the text.
    let end_token = config
        .tool_call_end_tokens
        .first()
        .map(String::as_str)
        .unwrap_or("<|call|>");
    if !trimmed.ends_with(end_token) {
        trimmed.push_str(end_token);
    }

    let enc = match get_harmony_encoding().await.as_ref() {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!("Failed to load harmony encoding: {e}. Tool calls will not be parsed.");
            return Ok((vec![], Some(original_text)));
        }
    };

    // Encode the text into tokens using harmony encoding
    let tokens = enc.tokenizer().encode_with_special_tokens(&trimmed);

    // Create StreamableParser to process each token and create Harmony Format messages
    // Set Role to Assistant because we are parsing tool calls from an assistant message
    let mut parser = match StreamableParser::new(enc.clone(), Some(Role::Assistant)) {
        Ok(p) => p,
        Err(e) => {
            tracing::debug!(
                "Failed to create harmony streamable parser: {e}. Tool calls will not be parsed."
            );
            return Ok((vec![], Some(original_text)));
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
    let mut call_idx = 0usize; // Index of the tool call

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
    for message in messages.iter() {
        if message.author.role == Role::Assistant
            && message.channel.as_deref() == Some("commentary")
            && message
                .recipient
                .as_deref()
                .unwrap_or_default()
                .starts_with("functions.")
        {
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
        config
            .tool_call_start_tokens
            .iter()
            .any(|token| trimmed.contains(token))
    } else {
        config
            .tool_call_start_tokens
            .iter()
            .any(|token| trimmed.contains(token))
            || trimmed.contains("<|channel|>")
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
    async fn test_parse_tool_calls_harmony_basic() {
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
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).await.unwrap();
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
    async fn test_parse_tools_harmony_without_start_token() {
        let text = r#"
<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
<|message|>{"location":"San Francisco"}<|call|>
"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).await.unwrap();
        assert_eq!(normal_content, Some(text.trim().to_string()));
        assert_eq!(tool_calls.len(), 0);
    }

    #[tokio::test]
    async fn test_parse_tool_calls_harmony_with_multi_args() {
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
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).await.unwrap();
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
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).await.unwrap();
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
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let (tool_calls, normal_content) = parse_tool_calls_harmony(text, &config).await.unwrap();
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
}

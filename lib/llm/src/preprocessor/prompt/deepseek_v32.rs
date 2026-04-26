// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V3.2 prompt formatting.
//!
//! Production rendering goes through `jinja_chat::render_v32` and the inline
//! template at `templates/deepseek_v32_inline.jinja`. This module only keeps
//! public types (`ThinkingMode`, `DeepSeekV32Formatter`) and constants
//! (`tokens::*`), the request-side glue (`normalize_message_contents`, the
//! `OAIPromptFormatter` impl that injects tools / response_format into the
//! system message), and a thin shim entry point (`encode_messages`) that
//! routes through `jinja_chat::render_v32`.
//!
//! Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;

/// Special tokens for DeepSeek V3.2
pub mod tokens {
    pub const BOS: &str = "<｜begin▁of▁sentence｜>";
    pub const EOS: &str = "<｜end▁of▁sentence｜>";
    pub const THINKING_START: &str = "<think>";
    pub const THINKING_END: &str = "</think>";
    pub const DSML_TOKEN: &str = "｜DSML｜";
    pub const USER_START: &str = "<｜User｜>";
    pub const ASSISTANT_START: &str = "<｜Assistant｜>";
}

/// Thinking mode for the model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkingMode::Chat => "chat",
            ThinkingMode::Thinking => "thinking",
        }
    }
}

/// Extract visible text from OpenAI-style message content.
///
/// - string content: returned as-is
/// - array content: concatenates `type=text` parts and raw string items
/// - other JSON types: empty string (the template handles `null` etc.)
fn extract_visible_text(content: &JsonValue) -> String {
    match content {
        JsonValue::String(text) => text.clone(),
        JsonValue::Array(items) => items
            .iter()
            .filter_map(|item| {
                if let Some(text) = item.as_str() {
                    return Some(text.to_string());
                }

                let item_type = item.get("type").and_then(|v| v.as_str());
                if item_type == Some("text") {
                    return item
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|text| text.to_string());
                }

                tracing::warn!(
                    chunk_type = item_type.unwrap_or("unknown"),
                    "DeepSeek V3.2 formatter dropped non-text content chunk while normalizing message content",
                );

                None
            })
            .collect::<String>(),
        _ => String::new(),
    }
}

/// Normalize message `content` fields for text-only DeepSeek V3.2 rendering.
fn normalize_message_contents(messages: &mut [JsonValue]) {
    for msg in messages {
        let Some(content) = msg.get("content") else {
            continue;
        };
        let normalized = extract_visible_text(content);
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("content".to_string(), JsonValue::String(normalized));
        }
    }
}

/// Encode messages to prompt string.
///
/// Thin shim over `jinja_chat::render_v32` — production rendering lives in
/// the inline Jinja template at `templates/deepseek_v32_inline.jinja`.
pub fn encode_messages(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
) -> Result<String> {
    use super::jinja_chat;
    let jinja_mode = match thinking_mode {
        ThinkingMode::Chat => jinja_chat::ThinkingMode::Chat,
        ThinkingMode::Thinking => jinja_chat::ThinkingMode::Thinking,
    };
    // V3.2 has no caller-facing drop_thinking knob; the upstream Python ref
    // applies the equivalent of drop_thinking=true unconditionally.
    jinja_chat::render_v32(messages, jinja_mode, add_bos_token, true)
}

/// DeepSeek V3.2 Prompt Formatter
///
/// Implements OAIPromptFormatter for DeepSeek V3.2 models using native Rust implementation
#[derive(Debug)]
pub struct DeepSeekV32Formatter {
    thinking_mode: ThinkingMode,
}

impl DeepSeekV32Formatter {
    pub fn new(thinking_mode: ThinkingMode) -> Self {
        Self { thinking_mode }
    }

    /// Create formatter with thinking mode enabled (default for DSV3.2)
    pub fn new_thinking() -> Self {
        Self::new(ThinkingMode::Thinking)
    }

    /// Create formatter with chat mode
    pub fn new_chat() -> Self {
        Self::new(ThinkingMode::Chat)
    }

    /// Resolve thinking mode from per-request `chat_template_args`, falling back to the
    /// formatter's default. Two conventions are supported:
    ///   - `{"thinking": bool}` — common across models (e.g. Kimi K25)
    ///   - `{"thinking_mode": "chat"|"thinking"}` — matches the DSV3.2 Jinja template parameter
    fn resolve_thinking_mode(
        &self,
        args: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> ThinkingMode {
        if let Some(args) = args {
            if let Some(thinking) = args.get("thinking").and_then(|v| v.as_bool()) {
                return if thinking {
                    ThinkingMode::Thinking
                } else {
                    ThinkingMode::Chat
                };
            }
            if let Some(mode) = args.get("thinking_mode").and_then(|v| v.as_str()) {
                match mode {
                    "chat" => return ThinkingMode::Chat,
                    "thinking" => return ThinkingMode::Thinking,
                    _ => {}
                }
            }
        }
        self.thinking_mode
    }
}

impl super::OAIPromptFormatter for DeepSeekV32Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn super::OAIChatLikeRequest) -> Result<String> {
        let thinking_mode = self.resolve_thinking_mode(req.chat_template_args());

        // Get messages from request
        let messages_value = req.messages();

        // Convert minijinja Value to serde_json Value
        let messages_json =
            serde_json::to_value(&messages_value).context("Failed to convert messages to JSON")?;

        let mut messages_array = messages_json
            .as_array()
            .context("Messages is not an array")?
            .clone();

        // DeepSeek V3.2 native formatter expects text content in each message.
        // Normalize OpenAI content arrays (e.g. [{type: "text", text: "..."}]) to strings.
        normalize_message_contents(&mut messages_array);

        // Inject tools and response_format from request into the first system message
        // DeepSeek V3.2 expects these to be part of the system message for prompt rendering
        let tools_json = req
            .tools()
            .map(|t| serde_json::to_value(&t))
            .transpose()
            .context("Failed to convert tools to JSON")?;

        let response_format_json = req
            .response_format()
            .map(|rf| serde_json::to_value(&rf))
            .transpose()
            .context("Failed to convert response_format to JSON")?;

        if tools_json.is_some() || response_format_json.is_some() {
            // Find or create system message
            let system_idx = messages_array
                .iter()
                .position(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("system"));

            if let Some(idx) = system_idx {
                // Add to existing system message
                if let Some(msg) = messages_array.get_mut(idx)
                    && let Some(obj) = msg.as_object_mut()
                {
                    if let Some(tools) = tools_json {
                        obj.insert("tools".to_string(), tools);
                    }
                    if let Some(rf) = response_format_json {
                        obj.insert("response_format".to_string(), rf);
                    }
                }
            } else {
                // Create a system message if none exists
                let mut system_msg = serde_json::json!({
                    "role": "system",
                    "content": ""
                });
                if let Some(obj) = system_msg.as_object_mut() {
                    if let Some(tools) = tools_json {
                        obj.insert("tools".to_string(), tools);
                    }
                    if let Some(rf) = response_format_json {
                        obj.insert("response_format".to_string(), rf);
                    }
                }
                messages_array.insert(0, system_msg);
            }
        }

        // Encode with native implementation
        encode_messages(
            &messages_array,
            thinking_mode,
            true, // always add BOS token
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_conversation() {
        let messages = json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]);

        let result =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();

        assert!(result.starts_with(tokens::BOS));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains(tokens::USER_START));
        assert!(result.contains("Hello!"));
        assert!(result.contains(tokens::ASSISTANT_START));
        assert!(result.contains(tokens::THINKING_START));
    }

    #[test]
    fn test_extract_visible_text_from_content_array() {
        let content = json!([
            {"type": "text", "text": "who "},
            {"type": "text", "text": "are "},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "you?"}
        ]);

        let result = extract_visible_text(&content);
        assert_eq!(result, "who are you?");
    }

    #[test]
    fn test_formatter_handles_user_content_array() {
        use super::super::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "who are you?"}
            ]}
        ]));

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(result.contains("who are you?"));
        assert!(result.contains(tokens::USER_START));
        assert!(result.contains(tokens::ASSISTANT_START));
    }

    #[test]
    fn test_tools_rendering() {
        let messages = json!([
            {
                "role": "system",
                "content": "You are helpful.",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }]
            },
            {"role": "user", "content": "What's the weather?"}
        ]);

        let result =
            encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true).unwrap();

        assert!(result.contains("## Tools"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("<functions>"));
    }

    // Mock request for testing OAIPromptFormatter implementation
    struct MockRequest {
        messages: JsonValue,
        tools: Option<JsonValue>,
        response_format: Option<JsonValue>,
        chat_template_args: Option<std::collections::HashMap<String, JsonValue>>,
    }

    impl MockRequest {
        fn new(messages: JsonValue) -> Self {
            Self {
                messages,
                tools: None,
                response_format: None,
                chat_template_args: None,
            }
        }

        fn with_tools(mut self, tools: JsonValue) -> Self {
            self.tools = Some(tools);
            self
        }

        fn with_response_format(mut self, response_format: JsonValue) -> Self {
            self.response_format = Some(response_format);
            self
        }

        fn with_chat_template_args(
            mut self,
            args: std::collections::HashMap<String, JsonValue>,
        ) -> Self {
            self.chat_template_args = Some(args);
            self
        }
    }

    impl super::super::OAIChatLikeRequest for MockRequest {
        fn model(&self) -> String {
            "deepseek-v3.2".to_string()
        }

        fn messages(&self) -> minijinja::value::Value {
            minijinja::value::Value::from_serialize(&self.messages)
        }

        fn tools(&self) -> Option<minijinja::value::Value> {
            self.tools
                .as_ref()
                .map(minijinja::value::Value::from_serialize)
        }

        fn response_format(&self) -> Option<minijinja::value::Value> {
            self.response_format
                .as_ref()
                .map(minijinja::value::Value::from_serialize)
        }

        fn should_add_generation_prompt(&self) -> bool {
            true
        }

        fn chat_template_args(
            &self,
        ) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
            self.chat_template_args.as_ref()
        }
    }

    #[test]
    fn test_formatter_injects_tools_into_existing_system_message() {
        use super::super::OAIPromptFormatter;

        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Moscow?"}
        ]))
        .with_tools(tools);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify tools were injected into the prompt
        assert!(
            result.contains("## Tools"),
            "Should contain Tools section header"
        );
        assert!(
            result.contains("get_weather"),
            "Should contain function name"
        );
        assert!(
            result.contains("<functions>"),
            "Should contain functions block"
        );
        assert!(
            result.contains("</functions>"),
            "Should contain closing functions tag"
        );
        assert!(
            result.contains("You are a helpful assistant."),
            "Should preserve original system content"
        );
        assert!(
            result.contains(&format!("<{}function_calls>", tokens::DSML_TOKEN)),
            "Should contain DSML format instructions"
        );
    }

    #[test]
    fn test_formatter_creates_system_message_for_tools_when_missing() {
        use super::super::OAIPromptFormatter;

        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current time in a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"}
                    },
                    "required": ["timezone"]
                }
            }
        }]);

        // Request without system message
        let request = MockRequest::new(json!([
            {"role": "user", "content": "What time is it in Tokyo?"}
        ]))
        .with_tools(tools);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify tools were injected via auto-created system message
        assert!(
            result.contains("## Tools"),
            "Should contain Tools section even without explicit system message"
        );
        assert!(
            result.contains("get_current_time"),
            "Should contain function name"
        );
        assert!(
            result.contains("<functions>"),
            "Should contain functions block"
        );
    }

    #[test]
    fn test_formatter_without_tools_does_not_add_tools_section() {
        use super::super::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]));

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify no tools section was added
        assert!(
            !result.contains("## Tools"),
            "Should not contain Tools section when no tools provided"
        );
        assert!(
            !result.contains("<functions>"),
            "Should not contain functions block when no tools provided"
        );
        assert!(
            result.contains("You are a helpful assistant."),
            "Should preserve system content"
        );
    }

    #[test]
    fn test_formatter_with_multiple_tools() {
        use super::super::OAIPromptFormatter;

        let tools = json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string"}
                        }
                    }
                }
            }
        ]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather and time in Moscow?"}
        ]))
        .with_tools(tools);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify both tools are present
        assert!(
            result.contains("get_weather"),
            "Should contain first function"
        );
        assert!(
            result.contains("get_current_time"),
            "Should contain second function"
        );
    }

    // ==================== Structured Output Tests ====================

    #[test]
    fn test_formatter_injects_response_format_into_existing_system_message() {
        use super::super::OAIPromptFormatter;

        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "city_info",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                        "population": {"type": "number"}
                    },
                    "required": ["city", "country", "population"]
                }
            }
        });

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Moscow."}
        ]))
        .with_response_format(response_format);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify response format was injected into the prompt
        assert!(
            result.contains("## Response Format:"),
            "Should contain Response Format section header"
        );
        assert!(
            result.contains("json_schema"),
            "Should contain json_schema type"
        );
        assert!(result.contains("city_info"), "Should contain schema name");
        assert!(
            result.contains("You are a helpful assistant."),
            "Should preserve original system content"
        );
    }

    #[test]
    fn test_formatter_creates_system_message_for_response_format_when_missing() {
        use super::super::OAIPromptFormatter;

        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "weather_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "conditions": {"type": "string"}
                    }
                }
            }
        });

        // Request without system message
        let request = MockRequest::new(json!([
            {"role": "user", "content": "What's the weather?"}
        ]))
        .with_response_format(response_format);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify response format was injected via auto-created system message
        assert!(
            result.contains("## Response Format:"),
            "Should contain Response Format section even without explicit system message"
        );
        assert!(
            result.contains("weather_response"),
            "Should contain schema name"
        );
    }

    #[test]
    fn test_formatter_with_both_tools_and_response_format() {
        use super::super::OAIPromptFormatter;

        let tools = json!([{
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        }]);

        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "search_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "total_count": {"type": "number"}
                    }
                }
            }
        });

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a search assistant."},
            {"role": "user", "content": "Find documents about Rust."}
        ]))
        .with_tools(tools)
        .with_response_format(response_format);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify both tools and response format are present
        assert!(result.contains("## Tools"), "Should contain Tools section");
        assert!(
            result.contains("search_database"),
            "Should contain function name"
        );
        assert!(
            result.contains("## Response Format:"),
            "Should contain Response Format section"
        );
        assert!(
            result.contains("search_result"),
            "Should contain schema name"
        );
        assert!(
            result.contains("You are a search assistant."),
            "Should preserve original system content"
        );
    }

    #[test]
    fn test_formatter_without_response_format_does_not_add_response_format_section() {
        use super::super::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]));

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // Verify no response format section was added
        assert!(
            !result.contains("## Response Format:"),
            "Should not contain Response Format section when not provided"
        );
    }

    // ==================== Thinking Mode Override Tests ====================

    #[test]
    fn test_chat_mode_via_thinking_false() {
        use super::super::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(false))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // In chat mode, the last user message should be followed by </think> (closing tag)
        // rather than <think> (opening tag)
        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Chat mode should end with </think> after Assistant token, got: ...{}",
            &result[result.len().saturating_sub(80)..],
        );
        assert!(
            !result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Chat mode should NOT end with <think>",
        );
    }

    #[test]
    fn test_explicit_thinking_true_via_args() {
        use super::super::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(true))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Thinking mode should end with <think> after Assistant token",
        );
    }

    #[test]
    fn test_chat_mode_via_thinking_mode_string() {
        use super::super::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking_mode".to_string(), json!("chat"))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "thinking_mode='chat' should produce chat mode (ends with </think>)",
        );
    }

    #[test]
    fn test_thinking_mode_string_thinking() {
        use super::super::OAIPromptFormatter;

        let args =
            std::collections::HashMap::from([("thinking_mode".to_string(), json!("thinking"))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "thinking_mode='thinking' should produce thinking mode (ends with <think>)",
        );
    }

    #[test]
    fn test_default_thinking_mode_without_args() {
        use super::super::OAIPromptFormatter;

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]));

        // No chat_template_args — should default to formatter's thinking mode
        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Default (new_thinking) should produce thinking mode",
        );

        // Verify new_chat() default also works
        let formatter_chat = DeepSeekV32Formatter::new_chat();
        let result_chat = formatter_chat.render(&request).unwrap();

        assert!(
            result_chat.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Default (new_chat) should produce chat mode",
        );
    }

    #[test]
    fn test_thinking_false_overrides_default_thinking() {
        use super::super::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(false))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        // Formatter defaults to thinking, but request overrides to chat
        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Per-request thinking=false should override new_thinking() default",
        );
    }

    #[test]
    fn test_thinking_true_overrides_default_chat() {
        use super::super::OAIPromptFormatter;

        let args = std::collections::HashMap::from([("thinking".to_string(), json!(true))]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        // Formatter defaults to chat, but request overrides to thinking
        let formatter = DeepSeekV32Formatter::new_chat();
        let result = formatter.render(&request).unwrap();

        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_START
            )),
            "Per-request thinking=true should override new_chat() default",
        );
    }

    #[test]
    fn test_thinking_bool_takes_precedence_over_thinking_mode_string() {
        use super::super::OAIPromptFormatter;

        let args = std::collections::HashMap::from([
            ("thinking".to_string(), json!(false)),
            ("thinking_mode".to_string(), json!("thinking")),
        ]);

        let request = MockRequest::new(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]))
        .with_chat_template_args(args);

        let formatter = DeepSeekV32Formatter::new_thinking();
        let result = formatter.render(&request).unwrap();

        // "thinking": false should win over "thinking_mode": "thinking"
        assert!(
            result.ends_with(&format!(
                "{}{}",
                tokens::ASSISTANT_START,
                tokens::THINKING_END
            )),
            "Boolean 'thinking' key should take precedence over 'thinking_mode' string",
        );
    }
}

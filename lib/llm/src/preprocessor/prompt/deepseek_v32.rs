// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V3.2 native prompt formatting
//!
//! This module provides native Rust implementation of DeepSeek V3.2's chat template,
//! based on their official Python code: encoding_dsv32.py
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

/// System message template for tools
const TOOLS_SYSTEM_TEMPLATE: &str = r#"## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml_token}function_calls>" block like the following as part of your reply to the user:
<{dsml_token}function_calls>
<{dsml_token}invoke name="$FUNCTION_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$FUNCTION_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml_token}function_calls>
...
</{dsml_token}function_calls>

<function_results>
...
</function_results>

{thinking_start_token}...thinking about results{thinking_end_token}

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"#;

const RESPONSE_FORMAT_TEMPLATE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}";

const TOOL_CALL_TEMPLATE: &str =
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>";

#[allow(dead_code)]
const TOOL_CALLS_TEMPLATE: &str =
    "<{dsml_token}function_calls>\n{tool_calls}\n</{dsml_token}function_calls>";

const TOOL_OUTPUT_TEMPLATE: &str = "\n<result>{content}</result>";

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

/// Convert value to JSON string matching Python's json.dumps() format with spaces
fn to_json(value: &JsonValue) -> String {
    // Python's json.dumps() adds spaces after colons and commas
    // {"name": "value", "key": "value2"}
    // Rust's serde_json::to_string() produces:
    // {"name":"value","key":"value2"}
    // We need to match Python's format for test compatibility

    let compact = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());

    // Add spaces after colons and commas (but not inside strings)
    let mut result = String::with_capacity(compact.len() + compact.len() / 4);
    let mut in_string = false;
    let mut prev_char = '\0';

    for ch in compact.chars() {
        if ch == '"' && prev_char != '\\' {
            in_string = !in_string;
        }

        result.push(ch);

        // Add space after ':' or ',' if not inside a string
        if !in_string && (ch == ':' || ch == ',') {
            result.push(' ');
        }

        prev_char = ch;
    }

    result
}

/// Extract tools from OpenAI format
fn tools_from_openai_format(tools: &[JsonValue]) -> Vec<JsonValue> {
    tools
        .iter()
        .filter_map(|tool| tool.get("function").cloned())
        .collect()
}

/// Render tools section for system prompt
fn render_tools(tools: &[JsonValue]) -> String {
    let tools_json: Vec<String> = tools_from_openai_format(tools)
        .iter()
        .map(to_json)
        .collect();

    TOOLS_SYSTEM_TEMPLATE
        .replace("{tool_schemas}", &tools_json.join("\n"))
        .replace("{dsml_token}", tokens::DSML_TOKEN)
        .replace("{thinking_start_token}", tokens::THINKING_START)
        .replace("{thinking_end_token}", tokens::THINKING_END)
}

/// Find the last user or developer message index
fn find_last_user_index(messages: &[JsonValue]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, msg)| {
            msg.get("role")
                .and_then(|r| r.as_str())
                .map(|r| r == "user" || r == "developer")
                .unwrap_or(false)
        })
        .map(|(idx, _)| idx)
}

/// Encode arguments to DSML parameter format
fn encode_arguments_to_dsml(tool_call: &JsonValue) -> Result<String> {
    let arguments_str = tool_call
        .get("arguments")
        .and_then(|a| a.as_str())
        .context("Missing or invalid 'arguments' field")?;

    let arguments: JsonValue =
        serde_json::from_str(arguments_str).context("Failed to parse arguments JSON")?;

    let arguments_obj = arguments
        .as_object()
        .context("Arguments must be an object")?;

    let mut params = Vec::new();
    for (key, value) in arguments_obj {
        let is_string = value.is_string();
        let value_str = if is_string {
            value.as_str().unwrap().to_string()
        } else {
            to_json(value)
        };

        let param = format!(
            "<{}parameter name=\"{}\" string=\"{}\">{}</{}parameter>",
            tokens::DSML_TOKEN,
            key,
            if is_string { "true" } else { "false" },
            value_str,
            tokens::DSML_TOKEN
        );
        params.push(param);
    }

    Ok(params.join("\n"))
}

/// Render a single message
fn render_message(
    index: usize,
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    last_user_idx: Option<usize>,
) -> Result<String> {
    let msg = &messages[index];
    let role = msg
        .get("role")
        .and_then(|r| r.as_str())
        .context("Missing 'role' field")?;

    let mut prompt = String::new();

    match role {
        "system" => {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(content);

            if let Some(tools) = msg.get("tools").and_then(|t| t.as_array()) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools));
            }

            if let Some(response_format) = msg.get("response_format") {
                prompt.push_str("\n\n");
                prompt.push_str(
                    &RESPONSE_FORMAT_TEMPLATE.replace("{schema}", &to_json(response_format)),
                );
            }
        }

        "user" => {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(tokens::USER_START);
            prompt.push_str(content);
            prompt.push_str(tokens::ASSISTANT_START);

            if Some(index) == last_user_idx && thinking_mode == ThinkingMode::Thinking {
                prompt.push_str(tokens::THINKING_START);
            } else {
                prompt.push_str(tokens::THINKING_END);
            }
        }

        "developer" => {
            let content = msg
                .get("content")
                .and_then(|c| c.as_str())
                .context("Developer role requires content")?;

            let mut content_developer = String::new();

            if let Some(tools) = msg.get("tools").and_then(|t| t.as_array()) {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(tools));
            }

            if let Some(response_format) = msg.get("response_format") {
                content_developer.push_str("\n\n");
                content_developer.push_str(
                    &RESPONSE_FORMAT_TEMPLATE.replace("{schema}", &to_json(response_format)),
                );
            }

            content_developer.push_str(&format!("\n\n# The user's message is: {}", content));

            prompt.push_str(tokens::USER_START);
            prompt.push_str(&content_developer);
            prompt.push_str(tokens::ASSISTANT_START);

            if Some(index) == last_user_idx && thinking_mode == ThinkingMode::Thinking {
                prompt.push_str(tokens::THINKING_START);
            } else {
                prompt.push_str(tokens::THINKING_END);
            }
        }

        "assistant" => {
            // Handle reasoning content
            // NOTE: If this assistant comes after last user message, the opening <think>
            // was already added in the user message. We only need to add content and closing tag.
            //
            // Handle reasoning_content which may be a plain string or an array of segments.
            // DeepSeek V3.2 always places its <think> block before all tool calls, so
            // joining segments produces the correct flat form here.
            if thinking_mode == ThinkingMode::Thinking
                && last_user_idx.is_some_and(|idx| index > idx)
            {
                let reasoning = msg.get("reasoning_content").and_then(|v| match v {
                    serde_json::Value::String(s) => {
                        if s.is_empty() {
                            None
                        } else {
                            Some(s.clone())
                        }
                    }
                    serde_json::Value::Array(arr) => {
                        let joined = arr
                            .iter()
                            .filter_map(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n");
                        if joined.is_empty() {
                            None
                        } else {
                            Some(joined)
                        }
                    }
                    _ => None,
                });

                if let Some(reasoning) = reasoning {
                    // DON'T add THINKING_START - it was already added in user message
                    prompt.push_str(&reasoning);
                    prompt.push_str(tokens::THINKING_END);
                }
            }

            // Handle content
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                prompt.push_str(content);
            }

            // Handle tool calls
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array())
                && !tool_calls.is_empty()
            {
                prompt.push_str("\n\n");
                prompt.push_str(&format!("<{}function_calls>\n", tokens::DSML_TOKEN));

                for tool_call in tool_calls {
                    let name = tool_call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .context("Missing tool call name")?;

                    let arguments = encode_arguments_to_dsml(
                        tool_call.get("function").context("Missing function")?,
                    )?;

                    let invoke = TOOL_CALL_TEMPLATE
                        .replace("{dsml_token}", tokens::DSML_TOKEN)
                        .replace("{name}", name)
                        .replace("{arguments}", &arguments);

                    prompt.push_str(&invoke);
                    prompt.push('\n');
                }

                prompt.push_str(&format!("</{}function_calls>", tokens::DSML_TOKEN));
            }

            prompt.push_str(tokens::EOS);
        }

        "tool" => {
            // Find the previous assistant message
            let mut prev_assistant_idx = None;
            let mut tool_count = 0;

            for i in (0..index).rev() {
                let prev_role = messages[i].get("role").and_then(|r| r.as_str());
                if prev_role == Some("tool") {
                    tool_count += 1;
                } else if prev_role == Some("assistant") {
                    prev_assistant_idx = Some(i);
                    break;
                }
            }

            let tool_call_order = tool_count + 1;

            // Add opening tag for first tool result
            if tool_call_order == 1 {
                prompt.push_str("\n\n<function_results>");
            }

            // Add result
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(&TOOL_OUTPUT_TEMPLATE.replace("{content}", content));

            // Check if this is the last tool result
            if let Some(prev_idx) = prev_assistant_idx {
                let tool_calls_count = messages[prev_idx]
                    .get("tool_calls")
                    .and_then(|t| t.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                if tool_call_order == tool_calls_count {
                    prompt.push_str("\n</function_results>");

                    if last_user_idx.is_some_and(|idx| index >= idx)
                        && thinking_mode == ThinkingMode::Thinking
                    {
                        prompt.push_str("\n\n");
                        prompt.push_str(tokens::THINKING_START);
                    } else {
                        prompt.push_str("\n\n");
                        prompt.push_str(tokens::THINKING_END);
                    }
                }
            }
        }

        _ => anyhow::bail!("Unknown role: {}", role),
    }

    Ok(prompt)
}

/// Encode messages to prompt string
///
/// # Arguments
/// * `messages` - Array of messages in OpenAI format
/// * `thinking_mode` - Whether to use thinking mode
/// * `add_bos_token` - Whether to add BOS token at start
///
/// # Returns
/// Formatted prompt string ready for tokenization
pub fn encode_messages(
    messages: &[JsonValue],
    thinking_mode: ThinkingMode,
    add_bos_token: bool,
) -> Result<String> {
    let mut prompt = String::new();

    if add_bos_token {
        prompt.push_str(tokens::BOS);
    }

    let last_user_idx = find_last_user_index(messages);

    for (index, _) in messages.iter().enumerate() {
        let msg_prompt = render_message(index, messages, thinking_mode, last_user_idx)?;
        prompt.push_str(&msg_prompt);
    }

    Ok(prompt)
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
}

impl super::OAIPromptFormatter for DeepSeekV32Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn super::OAIChatLikeRequest) -> Result<String> {
        // Get messages from request
        let messages_value = req.messages();

        // Convert minijinja Value to serde_json Value
        let messages_json =
            serde_json::to_value(&messages_value).context("Failed to convert messages to JSON")?;

        let mut messages_array = messages_json
            .as_array()
            .context("Messages is not an array")?
            .clone();

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
            self.thinking_mode,
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
    }

    impl MockRequest {
        fn new(messages: JsonValue) -> Self {
            Self {
                messages,
                tools: None,
                response_format: None,
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
}

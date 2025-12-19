// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use minijinja::{context, value::Value};
use std::result::Result::Ok;

use crate::preprocessor::media::MediaDecoder;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};
use tracing;

use crate::preprocessor::prompt::{PromptInput, TextInput, TokenInput};

fn may_be_fix_tool_schema(tools: serde_json::Value) -> Option<Value> {
    // No need to validate or enforce other schema checks as the basic Named function schema is already validated while creating the request.
    // Empty parameters is allowed by OpenAI at request level. Need to enforce it at template level.
    // Whenever parameters is empty, insert "type": "object" and "properties": {}
    let mut updated_tools = Vec::new();
    if let Some(arr) = tools.as_array() {
        for tool in arr {
            let mut tool = tool.clone();
            if let Some(function) = tool.get_mut("function")
                && let Some(parameters) = function.get_mut("parameters")
            {
                // Only operate if parameters is an object
                if parameters.is_object() {
                    let mut needs_type = false;
                    let mut needs_properties = false;
                    let is_empty = parameters
                        .as_object()
                        .map(|o| o.is_empty())
                        .unwrap_or(false);

                    // If empty, we need to insert both
                    if is_empty {
                        needs_type = true;
                        needs_properties = true;
                    } else {
                        // If not empty, check if type/properties are missing
                        if let Some(obj) = parameters.as_object() {
                            if !obj.contains_key("type") {
                                needs_type = true;
                            }
                            if !obj.contains_key("properties") {
                                needs_properties = true;
                            }
                        }
                    }

                    if (needs_type || needs_properties)
                        && let Some(obj) = parameters.as_object_mut()
                    {
                        if needs_type {
                            obj.insert(
                                "type".to_string(),
                                serde_json::Value::String("object".to_string()),
                            );
                        }
                        if needs_properties {
                            obj.insert(
                                "properties".to_string(),
                                serde_json::Value::Object(Default::default()),
                            );
                        }
                    }
                }
            }
            updated_tools.push(tool);
        }
    }
    Some(Value::from_serialize(&updated_tools))
}

fn may_be_fix_msg_content(messages: serde_json::Value, preserve_arrays: bool) -> Value {
    // preserve_arrays=true: strings → arrays (multimodal)
    // preserve_arrays=false: text-only arrays → strings (standard)

    let Some(arr) = messages.as_array() else {
        return Value::from_serialize(&messages);
    };

    let updated_messages: Vec<_> = arr
        .iter()
        .map(|msg| {
            match msg.get("content") {
                // Case 1: String to Array (for multimodal templates)
                Some(serde_json::Value::String(text)) if preserve_arrays => {
                    let mut modified_msg = msg.clone();
                    if let Some(msg_object) = modified_msg.as_object_mut() {
                        let content_array = serde_json::json!([{
                            "type": "text",
                            "text": text
                        }]);
                        msg_object.insert("content".to_string(), content_array);
                    }
                    modified_msg
                }
                // Case 2: Array to String (for standard templates)
                Some(serde_json::Value::Array(content_array)) if !preserve_arrays => {
                    let is_text_only_array = !content_array.is_empty()
                        && content_array.iter().all(|part| {
                            part.get("type")
                                .and_then(|type_field| type_field.as_str())
                                .map(|type_str| type_str == "text")
                                .unwrap_or(false)
                        });

                    if is_text_only_array {
                        let mut modified_msg = msg.clone();
                        if let Some(msg_object) = modified_msg.as_object_mut() {
                            let text_parts: Vec<&str> = content_array
                                .iter()
                                .filter_map(|part| part.get("text")?.as_str())
                                .collect();
                            let concatenated_text = text_parts.join("\n");

                            msg_object.insert(
                                "content".to_string(),
                                serde_json::Value::String(concatenated_text),
                            );
                        }
                        modified_msg // Concatenated string content
                    } else {
                        msg.clone() // Mixed content or non-text only
                    }
                }
                _ => msg.clone(), // No conversion needed
            }
        })
        .collect();

    Value::from_serialize(&updated_messages)
}

fn normalize_tool_arguments_in_messages(messages: &mut serde_json::Value) {
    // Deserialize tool call arguments from JSON strings to objects/arrays before template rendering
    // avoids double encoding and enables iteration
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };

    for msg in msgs.iter_mut() {
        if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
            for tc in tool_calls {
                if let Some(function) = tc.get_mut("function").and_then(|v| v.as_object_mut())
                    && let Some(args) = function.get_mut("arguments")
                    && let Some(s) = args.as_str()
                    && let Ok(parsed) = serde_json::from_str(s)
                {
                    *args = parsed;
                }
            }
        }

        if let Some(function_call) = msg.get_mut("function_call").and_then(|v| v.as_object_mut())
            && let Some(args) = function_call.get_mut("arguments")
            && let Some(s) = args.as_str()
            && let Ok(parsed) = serde_json::from_str(s)
        {
            *args = parsed;
        }
    }
}

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        let messages_json = serde_json::to_value(&self.inner.messages).unwrap();
        Value::from_serialize(&messages_json)
    }

    fn tools(&self) -> Option<Value> {
        if self.inner.tools.is_none() {
            None
        } else {
            // Try to fix the tool schema if it is missing type and properties
            Some(may_be_fix_tool_schema(
                serde_json::to_value(&self.inner.tools).unwrap(),
            )?)
        }
    }

    fn tool_choice(&self) -> Option<Value> {
        if self.inner.tool_choice.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.inner.tool_choice))
        }
    }

    fn should_add_generation_prompt(&self) -> bool {
        // Only add generation prompt if the last message was not assistant (default to true when no last message)
        self.inner
            .messages
            .last()
            .map(|last| {
                !matches!(
                    last,
                    dynamo_async_openai::types::ChatCompletionRequestMessage::Assistant(_)
                )
            })
            .unwrap_or(true)
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }

    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        self.media_io_kwargs.as_ref()
    }
}

impl OAIChatLikeRequest for NvCreateCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }
    fn messages(&self) -> minijinja::value::Value {
        let message = dynamo_async_openai::types::ChatCompletionRequestMessage::User(
            dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    crate::protocols::openai::completions::prompt_to_string(&self.inner.prompt),
                ),
                name: None,
            },
        );

        minijinja::value::Value::from_serialize(vec![message])
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }

    fn prompt_input_type(&self) -> PromptInput {
        match &self.inner.prompt {
            dynamo_async_openai::types::Prompt::IntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Single(vec![]))
            }
            dynamo_async_openai::types::Prompt::ArrayOfIntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Batch(vec![]))
            }
            dynamo_async_openai::types::Prompt::String(_) => {
                PromptInput::Text(TextInput::Single(String::new()))
            }
            dynamo_async_openai::types::Prompt::StringArray(_) => {
                PromptInput::Text(TextInput::Batch(vec![]))
            }
        }
    }

    fn extract_tokens(&self) -> Option<TokenInput> {
        match &self.inner.prompt {
            dynamo_async_openai::types::Prompt::IntegerArray(tokens) => {
                Some(TokenInput::Single(tokens.clone()))
            }
            dynamo_async_openai::types::Prompt::ArrayOfIntegerArray(arrays) => {
                Some(TokenInput::Batch(arrays.clone()))
            }
            _ => None,
        }
    }

    fn extract_text(&self) -> Option<TextInput> {
        match &self.inner.prompt {
            dynamo_async_openai::types::Prompt::String(text) => {
                Some(TextInput::Single(text.to_string()))
            }
            dynamo_async_openai::types::Prompt::StringArray(texts) => {
                Some(TextInput::Batch(texts.to_vec()))
            }
            _ => None,
        }
    }
}

impl OAIPromptFormatter for HfTokenizerConfigJsonFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        self.supports_add_generation_prompt
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        let mixins = Value::from_dyn_object(self.mixins.clone());

        let tools = req.tools();
        // has_tools should be true if tools is a non-empty array
        let has_tools = tools.as_ref().and_then(|v| v.len()).is_some_and(|l| l > 0);
        let add_generation_prompt = req.should_add_generation_prompt();

        tracing::trace!(
            "Rendering prompt with tools: {:?}, add_generation_prompt: {}",
            has_tools,
            add_generation_prompt
        );

        let messages_canonical = req.messages();
        let mut messages_for_template: serde_json::Value =
            serde_json::to_value(&messages_canonical).unwrap();

        messages_for_template = serde_json::to_value(may_be_fix_msg_content(
            messages_for_template,
            self.requires_content_arrays,
        ))
        .unwrap();

        normalize_tool_arguments_in_messages(&mut messages_for_template);

        let ctx = context! {
            messages => messages_for_template,
            tools => tools,
            bos_token => self.config.bos_tok(),
            eos_token => self.config.eos_tok(),
            unk_token => self.config.unk_tok(),
            add_generation_prompt => add_generation_prompt,
            ..mixins
        };

        // Merge any additional args into the context last so they take precedence
        let ctx = if let Some(args) = req.chat_template_args() {
            let extra = Value::from_serialize(args);
            context! { ..ctx, ..extra }
        } else {
            ctx
        };

        let tmpl: minijinja::Template<'_, '_> = if has_tools {
            self.env.get_template("tool_use")?
        } else {
            self.env.get_template("default")?
        };
        Ok(tmpl.render(&ctx)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_async_openai::types::ChatCompletionRequestMessage as Msg;
    use minijinja::{Environment, context};

    #[test]
    fn test_may_be_fix_tool_schema_missing_type_and_properties() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {},
                        "strict": null
                    }
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let tools = serde_json::to_value(request.tools()).unwrap();

        assert!(tools[0]["function"]["parameters"]["type"] == "object");
        assert!(
            tools[0]["function"]["parameters"]["properties"]
                == serde_json::Value::Object(Default::default())
        );
    }

    #[test]
    fn test_may_be_fix_tool_schema_missing_type() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'"
                                }
                            }
                        },
                        "strict": null
                    }
                }
            ]
        }"#;
        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

        let tools = serde_json::to_value(request.tools()).unwrap();

        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");

        let mut expected_properties = serde_json::Map::new();
        let mut location = serde_json::Map::new();
        location.insert(
            "type".to_string(),
            serde_json::Value::String("string".to_string()),
        );
        location.insert(
            "description".to_string(),
            serde_json::Value::String("City and state, e.g., 'San Francisco, CA'".to_string()),
        );
        expected_properties.insert("location".to_string(), serde_json::Value::Object(location));

        assert_eq!(
            tools[0]["function"]["parameters"]["properties"],
            serde_json::Value::Object(expected_properties)
        );
    }

    #[test]
    fn test_may_be_fix_tool_schema_missing_properties() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {"type": "object"},
                        "strict": null
                    }
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let tools = serde_json::to_value(request.tools()).unwrap();

        assert_eq!(
            tools[0]["function"]["parameters"]["properties"],
            serde_json::Value::Object(Default::default())
        );
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    /// Tests that content arrays (containing only text parts) are correctly concatenated.
    #[test]
    fn test_may_be_fix_msg_content_user_multipart() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part 1"},
                        {"type": "text", "text": "part 2"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test array → string normalization (preserve_arrays=false for standard templates)
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Verify: text-only array is concatenated into a single string
        assert_eq!(
            messages[0]["content"],
            serde_json::Value::String("part 1\npart 2".to_string())
        );
    }

    /// Tests that the function correctly handles a conversation
    /// with multiple roles and mixed message types:
    #[test]
    fn test_may_be_fix_msg_content_mixed_messages() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Another"},
                        {"type": "text", "text": "multi-part"},
                        {"type": "text", "text": "message"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test array → string normalization (preserve_arrays=false for standard templates)
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Verify: System message with string content remains unchanged
        assert_eq!(
            messages[0]["content"],
            serde_json::Value::String("You are a helpful assistant".to_string())
        );

        // Verify: User message with text-only array is concatenated
        assert_eq!(
            messages[1]["content"],
            serde_json::Value::String("Hello\nWorld".to_string())
        );

        // Verify: Assistant message with string content remains unchanged
        assert_eq!(
            messages[2]["content"],
            serde_json::Value::String("Hi there!".to_string())
        );

        // Verify: Second user message with text-only array is concatenated
        assert_eq!(
            messages[3]["content"],
            serde_json::Value::String("Another\nmulti-part\nmessage".to_string())
        );
    }

    /// Tests that empty content arrays remain unchanged.
    #[test]
    fn test_may_be_fix_msg_content_empty_array() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": []
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Empty arrays should be preserved regardless of preserve_arrays setting
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Verify: Empty arrays are preserved as-is
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 0);
    }

    /// Tests that messages with simple string content remain unchanged.
    #[test]
    fn test_may_be_fix_msg_content_single_text() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "Simple text message"
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test with preserve_arrays=false (standard templates)
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Verify: String content is not modified
        assert_eq!(
            messages[0]["content"],
            serde_json::Value::String("Simple text message".to_string())
        );
    }

    /// Tests that content arrays with mixed types (text + non-text) remain as arrays.
    #[test]
    fn test_may_be_fix_msg_content_mixed_types() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this image:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                        {"type": "text", "text": "What do you see?"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Mixed content should be preserved regardless of preserve_arrays setting
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Verify: Mixed content types are preserved as array for template handling
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 3);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1]["type"], "image_url");
        assert_eq!(content_array[2]["type"], "text");
    }

    /// Tests that content arrays containing only non-text types remain as arrays.
    #[test]
    fn test_may_be_fix_msg_content_non_text_only() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Non-text arrays should be preserved regardless of preserve_arrays setting
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Verify: Non-text content arrays are preserved for template handling
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "image_url");
        assert_eq!(content_array[1]["type"], "image_url");
    }

    #[test]
    fn test_none_tools_safe_for_all_templates() {
        use super::tokcfg::ChatTemplate;
        use super::{ContextMixins, HfTokenizerConfigJsonFormatter};

        // Due to minijinja limitations the expressions in conditional statements may not be short-circuited
        // This checks that our custom length filter works to avoid errors in this scenario
        // length should return 0 if tools is None and 'if tools is iterable and tools | length > 0' should evaluate to false
        let length_template = r#"
{%- if tools is iterable and tools | length > 0 %}
Tools available: {{ tools | length }}
{%- else %}
No tools
{%- endif %}
"#;

        // Because we return None for tools when there are no tools this scenario should also be evaluate to false
        // This is similar to the default jinja template behavior seen with llama models which check if tools is not none to activate tool mode
        let no_tool_template = r#"
{%- if tools is not none %}
TOOL MODE
{%- else %}
NORMAL MODE
{%- endif %}
"#;

        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": [
                {"safe_length": length_template},
                {"no_tool": no_tool_template}
            ]
        }))
        .unwrap();

        let formatter =
            HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap();

        let ctx = context! { tools => Option::<Value>::None };

        let result1 = formatter
            .env
            .get_template("safe_length")
            .unwrap()
            .render(&ctx);
        println!("Safe length template with no tools => None: {:?}", result1);
        assert!(
            result1.is_ok(),
            "Jinja template with and conditional and length filter should handle None: {:?}",
            result1
        );
        assert!(
            result1.unwrap().contains("No tools"),
            "Should show 'No tools'"
        );

        let result2 = formatter.env.get_template("no_tool").unwrap().render(&ctx);
        println!("Default template with no tools => None: {:?}", result2);
        assert!(
            result2.is_ok(),
            "Jinja template with if tools is not none conditional should handle None: {:?}",
            result2
        );
        assert!(result2.unwrap().contains("NORMAL MODE"));
    }

    /// Tests mixed content type scenarios.
    #[test]
    fn test_may_be_fix_msg_content_multiple_content_types() {
        // Scenario 1: Multiple different content types (text + image + audio)
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Listen to this:"},
                        {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}},
                        {"type": "text", "text": "And look at:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                        {"type": "text", "text": "What do you think?"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Mixed types should preserve array structure
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 5);

        // Scenario 2: Unknown/future content types mixed with text
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this:"},
                        {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                        {"type": "text", "text": "Interesting?"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        // Unknown types mixed with text should preserve array
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_normalize_tool_arguments_tojson() {
        let tmpl = r#"{{ messages[0].tool_calls[0].function.arguments | tojson }}"#;

        // Message with tool_calls containing JSON string arguments
        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": "{\"format\":\"celsius\",\"location\":\"San Francisco, CA\"}"
                }
            }]
        })]);

        normalize_tool_arguments_in_messages(&mut messages);

        let mut env = Environment::new();
        env.add_filter("tojson", super::super::tokcfg::tojson);
        env.add_template("t", tmpl).unwrap();
        let out = env
            .get_template("t")
            .unwrap()
            .render(context! { messages => messages.as_array().unwrap() })
            .unwrap();

        // Should produce clean JSON without double-encoding
        assert_eq!(
            out,
            r#"{"format":"celsius","location":"San Francisco, CA"}"#
        );
    }

    #[test]
    fn test_normalize_tool_arguments_items_loop() {
        let tmpl = r#"{% for k, v in messages[0].tool_calls[0].function.arguments|items %}{{k}}={{v}};{% endfor %}"#;

        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "f",
                    "arguments": "{\"a\":1,\"b\":\"x\"}"
                }
            }]
        })]);

        normalize_tool_arguments_in_messages(&mut messages);

        let mut env = Environment::new();
        env.add_template("t", tmpl).unwrap();
        let out = env
            .get_template("t")
            .unwrap()
            .render(context! { messages => messages.as_array().unwrap() })
            .unwrap();

        assert!(out == "a=1;b=x;" || out == "b=x;a=1;");
    }

    #[test]
    fn test_normalize_tool_arguments_legacy_function_call() {
        // Test deprecated function_call format (OpenAI compat)
        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "function_call": {
                "name": "get_weather",
                "arguments": "{\"location\":\"NYC\"}"
            }
        })]);

        normalize_tool_arguments_in_messages(&mut messages);

        assert_eq!(
            messages[0]["function_call"]["arguments"],
            serde_json::json!({"location": "NYC"})
        );
    }

    #[test]
    fn test_normalize_tool_arguments_malformed_json_passthrough() {
        // Malformed JSON should be left as a string
        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "f",
                    "arguments": "not valid json at all"
                }
            }]
        })]);

        normalize_tool_arguments_in_messages(&mut messages);

        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            serde_json::Value::String("not valid json at all".to_string())
        );
    }

    #[test]
    fn test_normalize_tool_arguments_with_multimodal_content() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this:"},
                        {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                        {"type": "text", "text": "Interesting?"}
                    ]
                },
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "analyze_video",
                            "arguments": "{\"url\":\"https://example.com/vid.mp4\",\"format\":\"mp4\"}"
                        }
                    }]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Apply content normalization with preserve_arrays=false (standard templates)
        let mut messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false)).unwrap();

        normalize_tool_arguments_in_messages(&mut messages);

        // Multimodal content preserved as array (mixed types not flattened)
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 3);

        // Tool arguments deserialized to object
        assert!(messages[1]["tool_calls"][0]["function"]["arguments"].is_object());
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"]["url"],
            "https://example.com/vid.mp4"
        );
    }

    /// Tests string → array normalization for multimodal templates
    #[test]
    fn test_may_be_fix_msg_content_string_to_array() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test with preserve_arrays=true (multimodal templates)
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, true)).unwrap();

        // Verify: String is converted to array format
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 1);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[0]["text"], "Hello, how are you?");
    }

    /// Tests that arrays are preserved when preserve_arrays=true
    #[test]
    fn test_may_be_fix_msg_content_array_preserved_with_multimodal() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part 1"},
                        {"type": "text", "text": "part 2"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test with preserve_arrays=true (multimodal templates)
        let messages = serde_json::to_value(may_be_fix_msg_content(messages_raw, true)).unwrap();

        // Verify: Array is preserved as-is
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["text"], "part 1");
        assert_eq!(content_array[1]["text"], "part 2");
    }

    fn user() -> Msg {
        Msg::User(Default::default())
    }
    fn asst() -> Msg {
        Msg::Assistant(Default::default())
    }
    fn tool() -> Msg {
        Msg::Tool(Default::default())
    }

    fn dummy_state(messages: Vec<Msg>) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": "test-model",
            "messages": messages
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn add_after_user() {
        let s = dummy_state(vec![user()]);
        assert!(s.should_add_generation_prompt());
    }

    #[test]
    fn add_after_tool() {
        let s = dummy_state(vec![tool()]);
        assert!(s.should_add_generation_prompt());
    }

    #[test]
    fn no_after_assistant() {
        let s = dummy_state(vec![asst()]);
        assert!(!s.should_add_generation_prompt());
    }

    #[test]
    fn add_when_empty() {
        let s = dummy_state(vec![]);
        assert!(s.should_add_generation_prompt());
    }
}

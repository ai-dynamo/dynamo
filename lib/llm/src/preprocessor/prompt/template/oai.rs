// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use minijinja::{context, value::Value};

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

fn may_be_fix_msg_content(messages: serde_json::Value) -> Value {
    // If messages[content] is provided as a list containing ONLY text parts,
    // concatenate them into a string to match chat template expectations.
    // Mixed content types are left for chat templates to handle.

    let Some(arr) = messages.as_array() else {
        return Value::from_serialize(&messages);
    };

    let updated_messages: Vec<_> = arr
        .iter()
        .map(|msg| {
            match msg.get("content") {
                Some(serde_json::Value::Array(content_array)) => {
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
                _ => msg.clone(), // String content or missing content - return unchanged
            }
        })
        .collect();

    Value::from_serialize(&updated_messages)
}

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        let messages_json = serde_json::to_value(&self.inner.messages).unwrap();

        let needs_fixing = if let Some(arr) = messages_json.as_array() {
            arr.iter()
                .any(|msg| msg.get("content").and_then(|c| c.as_array()).is_some())
        } else {
            false
        };

        if needs_fixing {
            may_be_fix_msg_content(messages_json)
        } else {
            Value::from_serialize(&messages_json)
        }
    }

    fn tools(&self) -> Option<Value> {
        if self.inner.tools.is_none() {
            // ISSUE: {%- if tools is iterable and tools | length > 0 %}
            // For cases like above, minijinja will not error out in calculating the length of tools
            // as it evaluates both the sides an don't do short circuiting.
            // Safe to return an empty array here. This will work even if tools are not present as length = 0
            Some(Value::from_serialize(Vec::<serde_json::Value>::new()))
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
        if let Some(last) = self.inner.messages.last() {
            matches!(
                last,
                dynamo_async_openai::types::ChatCompletionRequestMessage::User(_)
            )
        } else {
            true
        }
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
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

        let ctx = context! {
            messages => req.messages(),
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
        let messages = serde_json::to_value(request.messages()).unwrap();

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
        let messages = serde_json::to_value(request.messages()).unwrap();

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
        let messages = serde_json::to_value(request.messages()).unwrap();

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
        let messages = serde_json::to_value(request.messages()).unwrap();

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
        let messages = serde_json::to_value(request.messages()).unwrap();

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
        let messages = serde_json::to_value(request.messages()).unwrap();

        // Verify: Non-text content arrays are preserved for template handling
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "image_url");
        assert_eq!(content_array[1]["type"], "image_url");
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
        let messages = serde_json::to_value(request.messages()).unwrap();

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
        let messages = serde_json::to_value(request.messages()).unwrap();

        // Unknown types mixed with text should preserve array
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 3);
    }
}

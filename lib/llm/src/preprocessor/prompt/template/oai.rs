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

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        Value::from_serialize(&self.inner.messages)
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
        let has_tools = tools.is_some();
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

        let ctx = context! { ..ctx, ..context! {

        }};

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
}

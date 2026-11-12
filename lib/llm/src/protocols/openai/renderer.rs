// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`OAIChatLikeRequest`] for Dynamo's `Nv*` request wrappers, the view the
//! `dynamo_renderer` prompt formatters render a request through.

use minijinja::value::Value;

use dynamo_renderer::{
    OAIChatLikeRequest, PromptInput, TextInput, TokenInput, may_be_fix_tool_schema,
};

use super::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        let messages_json = self.template_messages_json();
        Value::from_serialize(&messages_json)
    }

    fn template_messages_json_override(&self) -> Option<serde_json::Value> {
        self.responses_tools
            .as_ref()
            .filter(|context| !context.history_phases.is_empty())
            .map(|_| self.template_messages_json())
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        Some(self.inner.messages.as_slice())
    }

    fn tools(&self) -> Option<Value> {
        if let Some(tools) = self.declared_template_tools() {
            return may_be_fix_tool_schema(tools);
        }
        let tools = self
            .chat_template_tools
            .as_deref()
            .or(self.inner.tools.as_deref())?;
        // Custom tools use the function schema before template-specific repair.
        may_be_fix_tool_schema(
            serde_json::to_value(
                tools
                    .iter()
                    .map(|tool| tool.function_template_tool())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
    }

    fn template_tools(&self) -> Option<minijinja::value::Value> {
        self.native_template_tools()
    }

    fn raw_tools(&self) -> Option<Value> {
        self.inner.tools.as_ref().map(Value::from_serialize)
    }

    fn tool_choice(&self) -> Option<Value> {
        self.inner
            .tool_choice
            .as_ref()
            .map(|choice| Value::from_serialize(choice.function_template_choice()))
    }

    fn response_format(&self) -> Option<Value> {
        self.inner
            .response_format
            .as_ref()
            .map(Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        // vLLM / HF: continue_final_message leaves the last turn open, which is
        // incompatible with appending a new generation prompt. Validation already
        // rejects the omitted/true add_generation_prompt combination; keep this
        // guard so internal callers that skip ValidateRequest still match HF.
        if self.common.continue_final_message == Some(true) {
            return false;
        }
        self.common.add_generation_prompt.unwrap_or(true)
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.inner.mm_processor_kwargs.as_ref()
    }
}

impl OAIChatLikeRequest for NvCreateCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }
    fn messages(&self) -> minijinja::value::Value {
        let message = dynamo_protocols::types::ChatCompletionRequestMessage::User(
            dynamo_protocols::types::ChatCompletionRequestUserMessage {
                content: dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
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
            dynamo_protocols::types::Prompt::IntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Single(vec![]))
            }
            dynamo_protocols::types::Prompt::ArrayOfIntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Batch(vec![]))
            }
            dynamo_protocols::types::Prompt::String(_) => {
                PromptInput::Text(TextInput::Single(String::new()))
            }
            dynamo_protocols::types::Prompt::StringArray(_) => {
                PromptInput::Text(TextInput::Batch(vec![]))
            }
        }
    }

    fn extract_tokens(&self) -> Option<TokenInput> {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::IntegerArray(tokens) => {
                Some(TokenInput::Single(tokens.clone()))
            }
            dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arrays) => {
                Some(TokenInput::Batch(arrays.clone()))
            }
            _ => None,
        }
    }

    fn extract_text(&self) -> Option<TextInput> {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::String(text) => {
                Some(TextInput::Single(text.to_string()))
            }
            dynamo_protocols::types::Prompt::StringArray(texts) => {
                Some(TextInput::Batch(texts.to_vec()))
            }
            _ => None,
        }
    }
}

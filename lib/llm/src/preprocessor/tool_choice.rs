// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::get_json_schema_from_tools;

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, ResponseFormat};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
}

impl OpenAIPreprocessor {
    /// Whether this request permits model output to be interpreted as tool calls.
    ///
    /// A configured parser describes the model's wire format; it does not grant
    /// every request permission to return tool calls. Assistant-output constraints
    /// take precedence for auto choice, matching native vLLM's request handling.
    pub(crate) fn tool_call_parsing_enabled(request: &NvCreateChatCompletionRequest) -> bool {
        if request.inner.tools.as_ref().is_none_or(Vec::is_empty) {
            return false;
        }

        match request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto)
        {
            ChatCompletionToolChoiceOption::None => false,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_) => {
                true
            }
            ChatCompletionToolChoiceOption::Auto => {
                !has_explicit_guided_decoding(request) && !has_response_format_constraint(request)
            }
        }
    }

    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. Named K3 requests always use their native XTML structural
    /// tag because generic JSON cannot represent K3 tool calls. Other forced
    /// tool-choice requests fall back to the legacy JSON-schema constraint when
    /// structural tags are not applied.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<bool, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        let tools = request.inner.tools.as_deref().unwrap_or(&[]);
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        let has_explicit_guided_decoding = has_explicit_guided_decoding(request);
        let has_response_format_constraint = has_response_format_constraint(request);

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(concat!(
                "guided decoding cannot be used in the same request as ",
                "tool_choice=\"required\" or a named tool_choice.",
            )));
        }

        // For non-forced tool choice, explicit guided decoding and response_format
        // constrain assistant content, so tool-choice guided decoding stays inactive.
        let has_assistant_constraint =
            has_explicit_guided_decoding || has_response_format_constraint;
        if !is_forced_tool_choice && has_assistant_constraint {
            return Ok(false);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        if self.apply_tool_choice_structural_tag(
            &convert_tool_choice(tool_choice),
            &convert_tools(tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            return Ok(true);
        }

        let uses_kimi_k3_parser = self
            .tool_call_parser
            .as_deref()
            .is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"))
            || self
                .runtime_config
                .reasoning_parser
                .as_deref()
                .is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"));
        if is_forced_tool_choice && uses_kimi_k3_parser {
            if matches!(tool_choice, ChatCompletionToolChoiceOption::Named(_)) {
                return Err(invalid_argument(
                    "named tool choice for Kimi K3 requires --dyn-tool-call-parser kimi_k3 \
                     with XTML structural-tag support",
                ));
            }

            // K3's prompt-level required instruction produces an XTML `tools`
            // channel. Generic JSON guided decoding would constrain the wrong
            // wire format and prevent the Rust K3 parser from seeing it.
            return Ok(false);
        }

        match get_json_schema_from_tools(Some(tool_choice), Some(tools)) {
            Ok(Some(schema)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                gd.json = Some(schema);
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string()));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(false)
    }
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.structured_outputs.is_some()
        || request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

fn has_response_format_constraint(request: &NvCreateChatCompletionRequest) -> bool {
    request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn request(extra: Value) -> NvCreateChatCompletionRequest {
        let mut value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}]
        });
        value
            .as_object_mut()
            .expect("base request is an object")
            .extend(extra.as_object().expect("extra is an object").clone());
        serde_json::from_value(value).expect("request must deserialize")
    }

    fn tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }])
    }

    #[test]
    fn tool_call_parsing_requires_tools() {
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({})
        )));
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({"tool_choice": "required"})
        )));
    }

    #[test]
    fn tool_call_parsing_honors_each_tool_choice() {
        for (tool_choice, expected) in [
            (json!(null), true),
            (json!("none"), false),
            (json!("auto"), true),
            (json!("required"), true),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                true,
            ),
        ] {
            let mut extra = json!({"tools": tools()});
            if !tool_choice.is_null() {
                extra["tool_choice"] = tool_choice;
            }
            assert_eq!(
                OpenAIPreprocessor::tool_call_parsing_enabled(&request(extra)),
                expected,
            );
        }
    }

    #[test]
    fn assistant_constraints_disable_only_auto_tool_parsing() {
        let constraints = [
            json!({"response_format": {"type": "json_object"}}),
            json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "event",
                        "schema": {"type": "object"}
                    }
                }
            }),
            json!({"guided_json": {"type": "object"}}),
            json!({"guided_regex": "[a-z]+"}),
            json!({"guided_choice": ["a", "b"]}),
            json!({"guided_grammar": "root ::= 'a'"}),
            json!({"structured_outputs": {"regex": "[a-z]+"}}),
        ];

        for constraint in constraints {
            let mut auto = json!({"tools": tools(), "tool_choice": "auto"});
            auto.as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                auto
            )));

            let mut required = json!({"tools": tools(), "tool_choice": "required"});
            required
                .as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                required
            )));
        }
    }

    #[test]
    fn text_response_format_does_not_disable_auto_tool_parsing() {
        assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({
                "tools": tools(),
                "tool_choice": "auto",
                "response_format": {"type": "text"}
            })
        )));
    }
}

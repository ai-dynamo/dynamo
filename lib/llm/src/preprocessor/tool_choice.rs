// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::GuidedToolConstraint;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::{
    ToolChoiceGuidance, get_tool_choice_guidance_from_tools, validate_openai_tool_choice,
};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
    CreateChatCompletionRequest, FunctionObject, ResponseFormat,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use serde_json::Value;

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
    /// every request permission to return tool calls. Permission depends only on
    /// whether the request supplies effective tools and whether `tool_choice`
    /// forbids them. Effective tools include both the OpenAI top-level list and
    /// Kimi-style dynamic declarations carried by system messages.
    /// Assistant-output constraints apply to assistant content and do not revoke
    /// an `auto` request's ability to choose a tool call.
    pub(crate) fn tool_call_parsing_enabled(request: &NvCreateChatCompletionRequest) -> bool {
        if !request.inner.has_effective_tools() {
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
            ChatCompletionToolChoiceOption::Auto => true,
        }
    }

    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. Supported K2 forced requests and named K3 requests
    /// intrinsically use their native structural tags because generic JSON cannot
    /// represent their tool calls. Other forced choices fall back to the legacy
    /// JSON-schema constraint when structural tags are not applied, except K3
    /// required requests, which stay on the prompt-level XTML path.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<GuidedToolConstraint, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        let tools = effective_tools(&request.inner)?;
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
            return Ok(GuidedToolConstraint::None);
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
            &convert_tools(&tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            return Ok(GuidedToolConstraint::StructuralTag);
        }

        let uses_kimi_k3_parser = uses_kimi_k3_parser(
            self.tool_call_parser.as_deref(),
            self.runtime_config.reasoning_parser.as_deref(),
        );
        if is_forced_tool_choice && uses_kimi_k3_parser {
            if matches!(tool_choice, ChatCompletionToolChoiceOption::Named(_)) {
                return Err(invalid_argument(
                    "named tool choice for Kimi K3 requires --dyn-tool-call-parser kimi_k3 \
                     with XTML structural-tag support",
                ));
            }

            // K3's prompt-level required instruction produces an XTML `tools`
            // channel. Generic JSON guided decoding would constrain the wrong
            // wire format and prevent the Rust K3 parser from seeing it. No JSON
            // schema is installed, so this is NOT a guided-JSON request.
            return Ok(GuidedToolConstraint::None);
        }

        match get_tool_choice_guidance_from_tools(
            Some(tool_choice),
            Some(&tools),
            request.inner.parallel_tool_calls,
        ) {
            Ok(Some(guidance)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                match guidance {
                    ToolChoiceGuidance::Json(schema) => gd.json = Some(schema),
                    ToolChoiceGuidance::Regex(regex) => gd.regex = Some(regex),
                }

                // Report the parser constraint implied by the installed grammar,
                // not merely the tool_choice that requested it. Both JSON and
                // regex guidance still use the guided-JSON tool-output parser.
                return Ok(installed_json_constraint(tool_choice));
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string()));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(GuidedToolConstraint::None)
    }
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
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

pub(crate) fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

pub(crate) fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

/// Normalize all tools visible to the model without rewriting the request.
///
/// Top-level OpenAI tools remain first. Kimi-style tools declared on system
/// messages follow in message order and may use either the wrapped OpenAI form
/// or Kimi's bare function-schema form. This view is used only by validation,
/// parser, and guided-decoding policy; dynamic declarations stay at their
/// original message positions for prompt rendering and KV-cache correctness.
pub(crate) fn effective_tools(
    request: &CreateChatCompletionRequest,
) -> Result<Vec<ChatCompletionTool>, DynamoError> {
    let dynamic_tools = request
        .dynamic_system_tools()
        .enumerate()
        .map(|(index, tool)| normalize_dynamic_system_tool(tool, index))
        .collect::<Result<Vec<_>, _>>()?;
    let mut tools = request.tools.clone().unwrap_or_default();
    tools.extend(dynamic_tools);
    Ok(tools)
}

/// Parser-facing form of [`effective_tools`].
pub(crate) fn effective_tool_definitions(
    request: &CreateChatCompletionRequest,
) -> Result<Vec<ToolDefinition>, DynamoError> {
    effective_tools(request).map(|tools| convert_tools(&tools))
}

fn normalize_dynamic_system_tool(
    tool: &Value,
    index: usize,
) -> Result<ChatCompletionTool, DynamoError> {
    let object = tool.as_object().ok_or_else(|| {
        invalid_argument(format!(
            "dynamic system tool at index {index} must be a JSON object"
        ))
    })?;
    let function = match (object.get("type"), object.get("function")) {
        (Some(Value::String(kind)), Some(Value::Object(function))) if kind == "function" => {
            function
        }
        (Some(kind), _) if kind.as_str() != Some("function") => {
            return Err(invalid_argument(format!(
                "dynamic system tool at index {index} must have type=\"function\""
            )));
        }
        (Some(_), _) => {
            return Err(invalid_argument(format!(
                "dynamic system tool at index {index} with type=\"function\" needs a function object"
            )));
        }
        (None, Some(_)) => {
            return Err(invalid_argument(format!(
                "dynamic system tool at index {index} with a function field needs type=\"function\""
            )));
        }
        (None, None) => object,
    };

    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            invalid_argument(format!(
                "dynamic system tool at index {index} needs a non-empty string name"
            ))
        })?;
    let description = optional_string(function.get("description"), "description", index)?;
    let parameters = match function.get("parameters") {
        None | Some(Value::Null) => None,
        Some(parameters @ Value::Object(_)) => Some(parameters.clone()),
        Some(_) => {
            return Err(invalid_argument(format!(
                "dynamic system tool at index {index} parameters must be a JSON Schema object"
            )));
        }
    };
    let strict = match function.get("strict") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(strict)) => Some(*strict),
        Some(_) => {
            return Err(invalid_argument(format!(
                "dynamic system tool at index {index} strict must be a boolean"
            )));
        }
    };

    Ok(ChatCompletionTool {
        r#type: ChatCompletionToolType::Function,
        function: FunctionObject {
            name: name.to_string(),
            description,
            parameters,
            strict,
        },
    })
}

fn optional_string(
    value: Option<&Value>,
    field: &str,
    index: usize,
) -> Result<Option<String>, DynamoError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid_argument(format!(
            "dynamic system tool at index {index} {field} must be a string"
        ))),
    }
}

/// The guided-tool constraint a request implies, given what the structural-tag stage
/// already decided.
///
/// Direct postprocessor callers and remote frontends use this compatibility helper
/// when they did not run request preprocessing. The worker streaming path carries the
/// enum returned by `apply_tool_choice_guided_decoding`; topology-B aggregation may
/// reconstruct a structural-tag request as guided JSON, so its complete-output parser
/// also checks the generated wire shape.
pub(crate) fn guided_tool_constraint(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
) -> Result<GuidedToolConstraint, DynamoError> {
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    let tool_choice = request
        .inner
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    let tools = effective_tools(&request.inner)?;
    validate_openai_tool_choice(Some(tool_choice), Some(&tools))
        .map_err(|error| invalid_argument(error.to_string()))?;
    let is_forced_tool_choice = matches!(
        tool_choice,
        ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
    );
    // Only a forced choice installs a tool-level JSON schema. Auto/None leave any
    // JSON constraint to `response_format`, which governs assistant CONTENT.
    if !is_forced_tool_choice {
        return Ok(GuidedToolConstraint::None);
    }
    // Forced + explicit guided decoding is rejected at request time, so reaching here
    // with both means the request never ran.
    if has_explicit_guided_decoding(request) {
        return Ok(GuidedToolConstraint::None);
    }
    // K3 forced requests are served by a prompt-level XTML instruction; no JSON schema.
    if uses_kimi_k3_parser(tool_call_parser, reasoning_parser) {
        return Ok(GuidedToolConstraint::None);
    }
    // Validate the forced choice against the actual tools the same way
    // `apply_tool_choice_guided_decoding` does, instead of blindly installing a
    // constraint for a `tool_choice` that names a tool absent from `tools` (or an
    // empty `tools` list under `tool_choice: "required"`).
    match get_tool_choice_guidance_from_tools(
        Some(tool_choice),
        Some(&tools),
        request.inner.parallel_tool_calls,
    ) {
        Ok(Some(_)) => Ok(installed_json_constraint(tool_choice)),
        Ok(None) => Ok(GuidedToolConstraint::None),
        Err(e) => Err(invalid_argument(e.to_string())),
    }
}

/// True when either configured parser is Kimi K3.
///
/// K3 forced requests are served by a prompt-level XTML instruction, so they must
/// NOT be reported as guided JSON even though their `tool_choice` is forced.
fn uses_kimi_k3_parser(tool_call_parser: Option<&str>, reasoning_parser: Option<&str>) -> bool {
    let is_k3 = |parser: &str| matches!(parser, "kimi_k3" | "kimi-k3");
    tool_call_parser.is_some_and(is_k3) || reasoning_parser.is_some_and(is_k3)
}

/// Map a forced `tool_choice` onto the guided-JSON parser constraint for its grammar.
///
/// Only reachable once `get_tool_choice_guidance_from_tools` produced either JSON or
/// regex guidance. It returns `None` for `Auto`/`None`, so those arms are unreachable
/// in practice and report [`GuidedToolConstraint::None`] rather than guessing.
fn installed_json_constraint(tool_choice: &ChatCompletionToolChoiceOption) -> GuidedToolConstraint {
    match tool_choice {
        ChatCompletionToolChoiceOption::Named(named) => GuidedToolConstraint::GuidedJsonNamed {
            tool_name: named.function.name.clone(),
        },
        ChatCompletionToolChoiceOption::Required => GuidedToolConstraint::GuidedJsonRequired,
        ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::None => {
            GuidedToolConstraint::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{ChatCompletionNamedToolChoice, FunctionName};
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
    fn dynamic_system_tools_enable_parsing_and_honor_tool_choice_none() {
        for (tool_choice, expected) in [
            (None, true),
            (Some(json!("none")), false),
            (Some(json!("auto")), true),
            (Some(json!("required")), true),
            (
                Some(json!({
                    "type": "function",
                    "function": {"name": "lookup"}
                })),
                true,
            ),
        ] {
            let mut extra = json!({
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                        "tools": [{"name": "lookup", "parameters": {"type": "object"}}]
                    },
                    {"role": "user", "content": "look it up"}
                ]
            });
            if let Some(tool_choice) = tool_choice {
                extra["tool_choice"] = tool_choice;
            }
            assert_eq!(
                OpenAIPreprocessor::tool_call_parsing_enabled(&request(extra)),
                expected,
            );
        }
    }

    #[test]
    fn effective_tools_preserve_order_and_normalize_wrapped_and_bare_shapes() {
        let request = request(json!({
            "tools": [{
                "type": "function",
                "function": {"name": "static_tool", "parameters": {"type": "object"}}
            }],
            "messages": [
                {"role": "user", "content": "start"},
                {
                    "role": "system",
                    "content": "",
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "wrapped_dynamic",
                            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
                            "strict": true
                        }
                    }]
                },
                {
                    "role": "system",
                    "content": "",
                    "tools": [{
                        "name": "bare_dynamic",
                        "description": "bare form",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": false,
                        "vendor_hint": "kept only in the original message"
                    }]
                },
                {"role": "user", "content": "continue"}
            ]
        }));

        let normalized = effective_tools(&request.inner).expect("dynamic tools must normalize");
        assert_eq!(
            normalized
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<Vec<_>>(),
            ["static_tool", "wrapped_dynamic", "bare_dynamic"]
        );
        assert_eq!(normalized[1].function.strict, Some(true));
        assert_eq!(normalized[2].function.strict, Some(false));
        assert_eq!(
            normalized[2].function.description.as_deref(),
            Some("bare form")
        );
        let original = serde_json::to_value(&request.inner.messages).unwrap();
        assert_eq!(
            original[2]["tools"][0]["vendor_hint"], "kept only in the original message",
            "normalization must not rewrite dynamic message declarations"
        );
    }

    #[test]
    fn effective_tools_reject_invalid_dynamic_schema_fields() {
        for (field, value) in [
            ("strict", json!("yes")),
            ("parameters", json!([])),
            ("description", json!(7)),
        ] {
            let mut tool = json!({"name": "lookup"});
            tool[field] = value;
            let request = request(json!({
                "messages": [
                    {"role": "system", "content": "", "tools": [tool]},
                    {"role": "user", "content": "go"}
                ]
            }));
            let error = effective_tools(&request.inner).expect_err("invalid field must fail");
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn assistant_constraints_do_not_revoke_auto_tool_permission() {
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
        ];

        for constraint in constraints {
            let mut auto = json!({"tools": tools(), "tool_choice": "auto"});
            auto.as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
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

    fn named(name: &str) -> ChatCompletionToolChoiceOption {
        ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
            r#type: dynamo_protocols::types::ChatCompletionToolType::Function,
            function: FunctionName {
                name: name.to_string(),
            },
        })
    }

    #[test]
    fn only_structural_tag_reports_a_structural_tag() {
        assert!(GuidedToolConstraint::StructuralTag.uses_structural_tag());
        assert!(!GuidedToolConstraint::None.uses_structural_tag());
        assert!(!GuidedToolConstraint::GuidedJsonRequired.uses_structural_tag());
        assert!(
            !GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
            .uses_structural_tag()
        );
    }

    #[test]
    fn required_reports_the_array_shape() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Required),
            GuidedToolConstraint::GuidedJsonRequired
        );
    }

    #[test]
    fn named_carries_the_tool_name_from_the_request() {
        assert_eq!(
            installed_json_constraint(&named("get_weather")),
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
        );
    }

    #[test]
    fn named_closed_zero_arg_tool_keeps_the_named_parser_constraint() {
        let request = request(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_server_time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_server_time"}
            }
        }));

        assert_eq!(
            guided_tool_constraint(&request, None, None, false).expect("constraint is valid"),
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_server_time".to_string(),
            }
        );
    }

    #[test]
    fn unforced_choices_install_no_tool_constraint() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Auto),
            GuidedToolConstraint::None
        );
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::None),
            GuidedToolConstraint::None
        );
    }

    #[test]
    fn kimi_k3_is_detected_from_either_parser_slot() {
        assert!(uses_kimi_k3_parser(Some("kimi_k3"), None));
        assert!(uses_kimi_k3_parser(Some("kimi-k3"), None));
        assert!(uses_kimi_k3_parser(None, Some("kimi_k3")));
        assert!(uses_kimi_k3_parser(None, Some("kimi-k3")));
    }

    #[test]
    fn non_k3_parsers_are_not_mistaken_for_k3() {
        assert!(!uses_kimi_k3_parser(None, None));
        assert!(!uses_kimi_k3_parser(Some("kimi_k2"), Some("qwen3")));
        assert!(!uses_kimi_k3_parser(Some("qwen3_coder"), None));
    }

    #[test]
    fn kimi_k3_required_without_tools_is_rejected_before_the_xtml_return() {
        for extra in [
            json!({"tool_choice": "required"}),
            json!({"tool_choice": "required", "tools": []}),
        ] {
            let request = request(extra);
            let result = guided_tool_constraint(&request, Some("kimi_k3"), None, false);
            assert!(
                result.is_err(),
                "Kimi K3 required must reject both missing and empty tools"
            );
        }
    }

    #[test]
    fn kimi_k3_forced_choices_accept_dynamic_system_tools() {
        for tool_choice in [
            json!("required"),
            json!({"type": "function", "function": {"name": "lookup"}}),
        ] {
            let request = request(json!({
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                        "tools": [{"name": "lookup", "parameters": {"type": "object"}}]
                    },
                    {"role": "user", "content": "go"}
                ],
                "tool_choice": tool_choice
            }));
            assert_eq!(
                guided_tool_constraint(&request, Some("kimi_k3"), None, false)
                    .expect("dynamic forced choice must validate"),
                GuidedToolConstraint::None,
                "K3 forced choices use native XTML rather than guided JSON"
            );
        }
    }
}

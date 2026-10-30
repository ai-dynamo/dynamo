// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::preprocessor::mistral_grammar::CallMode;
use crate::preprocessor::structural_tag::requires_native_tool_call_format;
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::chat_completions::{NvCreateChatCompletionRequest, unified_parser};
use crate::protocols::openai::tools::{
    ToolChoiceGuidance, get_tool_choice_guidance_from_tools, validate_openai_tool_choice,
};
use crate::protocols::openai::{GuidedToolConstraint, validate};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, ResponseFormat,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use serde_json::Value;

fn invalid_argument(message: impl Into<String>, public_message: Option<&str>) -> DynamoError {
    let mut error = DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message);
    if let Some(public_message) = public_message {
        error = error.public_message(public_message);
    }
    error.build()
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
    /// tool-call parser. When disabled or unavailable, ordinary forced choices
    /// retain the legacy JSON-schema fallback. Kimi's marker-delimited forced
    /// formats cannot be represented by that fallback and remain on their
    /// prompt/parser best-effort path instead. The Unified `mistral` family gets
    /// mistral-common's native-format grammar for forced choices, for `auto`
    /// with tools, and for a JSON schema on the reply, as upstream vLLM installs
    /// it.
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
        let parser_tool_choice = convert_tool_choice(tool_choice);

        self.validate_structured_output_composition(request, tool_choice, tools.as_ref())?;

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(
                concat!(
                    "guided decoding cannot be used in the same request as ",
                    "tool_choice=\"required\" or a named tool_choice.",
                ),
                None,
            ));
        }

        // Present only when the Unified mistral family serves this model. As
        // upstream vLLM does, a JSON schema for the reply joins the Mistral
        // grammar, so the backend sees lark for every request the grammar
        // constrains and resolves them all to one structured-output backend.
        // Also as upstream, a forced choice may then answer with JSON instead of
        // a call. Regex, choice, grammar and whitespace-pattern constraints keep
        // their own path.
        if let Some(grammar) = &self.mistral_tool_call_grammar
            && !has_non_json_guided_decoding(request)
            && let Some(reply_schema) = reply_json_schema(request)?
        {
            validate_openai_tool_choice(Some(tool_choice), Some(tools.as_ref()))
                .map_err(|error| invalid_argument(error.to_string(), None))?;
            let rendered = match mistral_call_mode(&parser_tool_choice, !tools.is_empty()) {
                Some(mode) => grammar.render(
                    mode,
                    convert_tools(tools.as_ref()),
                    request.inner.parallel_tool_calls.unwrap_or(true),
                    Some(&reply_schema),
                ),
                None => grammar.render_json(&reply_schema),
            };
            return install_mistral_grammar(common_request, rendered);
        }

        // Explicit guided decoding constrains assistant content and cannot be
        // composed with structural-tag tool calls.
        if !is_forced_tool_choice && has_explicit_guided_decoding {
            return Ok(GuidedToolConstraint::None);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        let tool_definitions = convert_tools(tools.as_ref());
        if self.apply_tool_choice_structural_tag(
            &parser_tool_choice,
            &tool_definitions,
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            request.inner.response_format.as_ref(),
            common_request,
        )? {
            return Ok(GuidedToolConstraint::StructuralTag);
        }

        // If the structural-tag layer did not compose this response format with
        // auto tool calls, it remains the sole assistant-output constraint.
        if !is_forced_tool_choice && has_response_format_constraint {
            return Ok(GuidedToolConstraint::None);
        }

        // Present only when the Unified mistral family serves this model.
        if let Some(grammar) = &self.mistral_tool_call_grammar
            && let Some(mode) = mistral_call_mode(&parser_tool_choice, !tool_definitions.is_empty())
        {
            let rendered = grammar.render(
                mode,
                tool_definitions,
                request.inner.parallel_tool_calls.unwrap_or(true),
                None,
            );
            return install_mistral_grammar(common_request, rendered);
        }

        if is_forced_tool_choice
            && forced_choice_keeps_native_format(
                self.tool_call_parser.as_deref(),
                self.runtime_config.reasoning_parser.as_deref(),
                self.unified_family(),
                &parser_tool_choice,
            )
        {
            tracing::debug!(
                tool_choice = ?parser_tool_choice,
                "Structural tag was not applied and the parser uses a native tool-call format; using prompt/parser compatibility fallback"
            );
            return Ok(GuidedToolConstraint::None);
        }

        match get_tool_choice_guidance_from_tools(
            Some(tool_choice),
            Some(tools.as_ref()),
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
                return Err(invalid_argument(err.to_string(), None));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(GuidedToolConstraint::None)
    }

    fn validate_structured_output_composition(
        &self,
        request: &NvCreateChatCompletionRequest,
        tool_choice: &ChatCompletionToolChoiceOption,
        tools: &[ChatCompletionTool],
    ) -> Result<(), DynamoError> {
        let allow_tool_calls_with_structured_output = self
            .runtime_config
            .structural_tag
            .as_ref()
            .is_some_and(|config| config.allow_tool_calls_with_structured_output);

        if matches!(tool_choice, ChatCompletionToolChoiceOption::Auto)
            && !tools.is_empty()
            && allow_tool_calls_with_structured_output
        {
            validate::validate_response_format_conflicts(
                &request.inner.response_format,
                &request.common,
            )
            .map_err(|error| {
                let message = error.to_string();
                invalid_argument(&message, Some(&message))
            })?;
        }

        Ok(())
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

/// Constraints the Mistral grammar does not carry, as upstream vLLM's Mistral
/// parser leaves them out of it: regex, choice, a grammar and a whitespace
/// pattern.
fn has_non_json_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    let common = &request.common;
    common.guided_regex.is_some()
        || common.guided_choice.as_ref().is_some_and(|v| !v.is_empty())
        || common.guided_grammar.is_some()
        || common.guided_whitespace_pattern.is_some()
}

/// What upstream vLLM's Mistral parser constrains a `json_object` reply to.
static ANY_OBJECT_OR_ARRAY: LazyLock<Value> =
    LazyLock::new(|| serde_json::json!({"anyOf": [{"type": "object"}, {"type": "array"}]}));

/// The JSON schema a request sets for the reply, read as upstream vLLM's Mistral
/// parser reads it: `guided_json` first, parsed when sent as text, then
/// `response_format`, where `json_object` allows any object or array.
fn reply_json_schema(
    request: &NvCreateChatCompletionRequest,
) -> Result<Option<Cow<'_, Value>>, DynamoError> {
    match &request.common.guided_json {
        Some(Value::String(text)) => serde_json::from_str(text)
            .map(|schema| Some(Cow::Owned(schema)))
            .map_err(|error| invalid_argument(format!("invalid guided_json: {error}"), None)),
        Some(schema) => Ok(Some(Cow::Borrowed(schema))),
        None => Ok(match request.inner.response_format.as_ref() {
            None | Some(ResponseFormat::Text) => None,
            Some(ResponseFormat::JsonObject) => Some(Cow::Borrowed(&*ANY_OBJECT_OR_ARRAY)),
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                Some(Cow::Borrowed(&json_schema.schema))
            }
        }),
    }
}

/// The Mistral grammar's mode for a `tool_choice`, or `None` for `none` and for
/// `auto` without tools.
fn mistral_call_mode(tool_choice: &ToolChoice, has_tools: bool) -> Option<CallMode<'_>> {
    match tool_choice {
        ToolChoice::Named(name) => Some(CallMode::Named(name)),
        ToolChoice::Required => Some(CallMode::Required),
        ToolChoice::Auto if has_tools => Some(CallMode::Auto),
        ToolChoice::Auto | ToolChoice::None => None,
    }
}

/// Install a Mistral grammar as the request's only structured-output constraint.
fn install_mistral_grammar(
    common_request: &mut PreprocessedRequest,
    grammar: String,
) -> Result<GuidedToolConstraint, DynamoError> {
    let guided = common_request
        .sampling_options
        .guided_decoding
        .get_or_insert_default();
    // The grammar carries any reply schema, and the backend takes one
    // constraint kind per request.
    guided.json = None;
    guided.grammar = Some(grammar);
    // The grammar admits the optional thought itself, so a backend reasoning
    // parser must apply it from the first token.
    let Value::Object(extra_args) = common_request
        .extra_args
        .get_or_insert_with(|| Value::Object(Default::default()))
    else {
        return Err(DynamoError::builder()
            .error_type(ErrorType::Internal)
            .message("preprocessed extra_args must be an object")
            .build());
    };
    extra_args.insert("reasoning_ended".to_string(), Value::Bool(true));
    Ok(GuidedToolConstraint::None)
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

/// Normalize and validate all tools visible to the model without rewriting the request.
///
/// Top-level OpenAI tools remain first. Kimi-style tools declared on system
/// messages follow in message order and may use either the wrapped OpenAI form
/// or Kimi's bare function-schema form. This view is used only by validation,
/// parser, and guided-decoding policy; dynamic declarations stay at their
/// original message positions for prompt rendering and KV-cache correctness.
pub(crate) fn effective_tools(
    request: &CreateChatCompletionRequest,
) -> Result<Cow<'_, [ChatCompletionTool]>, DynamoError> {
    validate::validated_effective_tools(request)
        .map_err(|error| invalid_argument(error.to_string(), None))
}

pub(crate) fn effective_tool_definitions(
    request: &CreateChatCompletionRequest,
) -> Result<Vec<ToolDefinition>, DynamoError> {
    effective_tools(request).map(|tools| convert_tools(tools.as_ref()))
}

/// [`guided_tool_constraint_for_model`] with the Unified family selected from the
/// parser names alone.
#[cfg(test)]
pub(crate) fn guided_tool_constraint(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
) -> Result<GuidedToolConstraint, DynamoError> {
    guided_tool_constraint_for_model(
        request,
        tool_call_parser,
        reasoning_parser,
        unified_parser::selected_family(tool_call_parser, reasoning_parser),
        uses_structural_tag,
    )
}

/// The guided-tool constraint a request implies, given what the structural-tag stage
/// already decided and the Unified family that serves the model, which setup
/// resolves with its tokenizer.
///
/// Direct postprocessor callers and remote frontends use this compatibility helper
/// when they did not run request preprocessing. The worker streaming path carries the
/// enum returned by `apply_tool_choice_guided_decoding`; topology-B aggregation may
/// reconstruct a structural-tag request as guided JSON, so its complete-output parser
/// also checks the generated wire shape.
pub(crate) fn guided_tool_constraint_for_model(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    unified_family: Option<&str>,
    uses_structural_tag: bool,
) -> Result<GuidedToolConstraint, DynamoError> {
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    let tools = effective_tools(&request.inner)?;
    effective_tools_constraint(
        request,
        tool_call_parser,
        reasoning_parser,
        unified_family,
        tools.as_ref(),
    )
}

/// Derive the guided-tool constraint from an effective tool set the caller has
/// already normalized and validated through [`effective_tools`].
pub(crate) fn guided_tool_constraint_with_effective_tools(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
    tools: &[ChatCompletionTool],
) -> Result<GuidedToolConstraint, DynamoError> {
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    effective_tools_constraint(
        request,
        tool_call_parser,
        reasoning_parser,
        unified_parser::selected_family(tool_call_parser, reasoning_parser),
        tools,
    )
}

fn effective_tools_constraint(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    unified_family: Option<&str>,
    tools: &[ChatCompletionTool],
) -> Result<GuidedToolConstraint, DynamoError> {
    let tool_choice = request
        .inner
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    validate_openai_tool_choice(Some(tool_choice), Some(tools))
        .map_err(|error| invalid_argument(error.to_string(), None))?;
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
    let parser_tool_choice = convert_tool_choice(tool_choice);
    // Kimi and Unified Mistral forced requests use native formats that generic
    // JSON guidance cannot represent. Keep this compatibility path aligned with
    // request preprocessing.
    if forced_choice_keeps_native_format(
        tool_call_parser,
        reasoning_parser,
        unified_family,
        &parser_tool_choice,
    ) {
        return Ok(GuidedToolConstraint::None);
    }
    // Validate the forced choice against the actual tools the same way
    // `apply_tool_choice_guided_decoding` does, instead of blindly installing a
    // constraint for a `tool_choice` that names a tool absent from `tools` (or an
    // empty `tools` list under `tool_choice: "required"`).
    match get_tool_choice_guidance_from_tools(
        Some(tool_choice),
        Some(tools),
        request.inner.parallel_tool_calls,
    ) {
        Ok(Some(_)) => Ok(installed_json_constraint(tool_choice)),
        Ok(None) => Ok(GuidedToolConstraint::None),
        Err(e) => Err(invalid_argument(e.to_string(), None)),
    }
}

/// Whether a forced choice keeps the model's native tool-call format instead of
/// the generic JSON-schema fallback. Kimi is matched on either parser slot. The
/// v1 Mistral parsers share the `mistral` name, so Mistral is matched on the
/// model's Unified family.
fn forced_choice_keeps_native_format(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    unified_family: Option<&str>,
    tool_choice: &ToolChoice,
) -> bool {
    requires_native_tool_call_format(tool_call_parser, tool_choice)
        || requires_native_tool_call_format(reasoning_parser, tool_choice)
        || (matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_))
            && unified_family == Some(unified_parser::MISTRAL_UNIFIED_FAMILY))
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
    use std::{path::PathBuf, sync::Arc};

    use super::*;
    use crate::local_model::runtime_config::{
        LLGUIDANCE_LARK_GRAMMAR_RUNTIME_KEY, StructuralTagMode,
    };
    use crate::model_card::ModelDeploymentCard;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_protocols::types::{ChatCompletionNamedToolChoice, FunctionName};
    use dynamo_runtime::config::environment_names::llm as env_llm;
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

    fn preprocessor(parser: &str, mode: StructuralTagMode) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = (mode == StructuralTagMode::On)
            .then(crate::local_model::runtime_config::StructuralTagConfig::default);
        mdc.runtime_config.tool_call_parser = Some(parser.to_string());
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn preprocessed_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(Vec::new())
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
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
    fn effective_tools_borrows_top_level_tools_without_dynamic_declarations() {
        let request = request(json!({"tools": tools()}));
        let top_level = request.inner.tools.as_deref().unwrap();

        let effective = effective_tools(&request.inner).expect("top-level tools must validate");

        assert!(matches!(&effective, Cow::Borrowed(_)));
        assert!(std::ptr::eq(effective.as_ptr(), top_level.as_ptr()));
    }

    #[test]
    fn dynamic_system_tools_enable_parsing_and_honor_tool_choice_none() {
        for (tool_choice, expected) in [(None, true), (Some(json!("none")), false)] {
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
                            "parameters": {
                                "type": "object",
                                "properties": {"x": {"type": "integer"}},
                                "required": ["x"],
                                "additionalProperties": false
                            },
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
        assert!(matches!(&normalized, Cow::Owned(_)));
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
    fn effective_tools_reject_invalid_dynamic_metadata() {
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

        let request = request(json!({
            "messages": [
                {"role": "system", "content": "", "tools": [{"name": "bad name"}]},
                {"role": "user", "content": "go"}
            ]
        }));
        let error = effective_tools(&request.inner).expect_err("invalid name must fail");
        assert_eq!(error.error_type(), ErrorType::InvalidArgument);
        assert!(error.to_string().contains("has an invalid name"));
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
    fn native_format_forced_choices_install_no_compatibility_json_constraint() {
        let requests = [
            request(json!({
                "tools": tools(),
                "tool_choice": "required"
            })),
            request(json!({
                "tools": tools(),
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"}
                }
            })),
        ];

        for request in &requests {
            for (tool_call_parser, reasoning_parser) in [
                (Some("kimi_k2"), None),
                (None, Some("kimi_k2")),
                (Some("kimi_k3"), None),
                (None, Some("kimi-k3")),
            ] {
                assert_eq!(
                    guided_tool_constraint(request, tool_call_parser, reasoning_parser, false,)
                        .expect("native-format choice is valid"),
                    GuidedToolConstraint::None,
                );
            }
        }
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
    fn structural_tag_off_preserves_kimi_k2_required_native_tag() {
        let request = request(json!({
            "tools": tools(),
            "tool_choice": "required"
        }));
        let preprocessor = preprocessor("kimi_k2", StructuralTagMode::Off);
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();

        assert_eq!(applied, GuidedToolConstraint::StructuralTag);
        assert!(
            common_request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.structural_tag.as_ref())
                .is_some()
        );
    }

    #[test]
    fn structural_tag_off_preserves_kimi_k3_named_native_tag() {
        let request = request(json!({
            "tools": tools(),
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            }
        }));
        let preprocessor = preprocessor("kimi_k3", StructuralTagMode::Off);
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();

        assert_eq!(applied, GuidedToolConstraint::StructuralTag);
        assert!(
            common_request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.structural_tag.as_ref())
                .is_some()
        );
    }

    #[test]
    fn structural_tag_off_preserves_generic_forced_tool_fallback() {
        let request = request(json!({
            "tools": tools(),
            "tool_choice": "required"
        }));
        let preprocessor = preprocessor("hermes", StructuralTagMode::Off);
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();

        assert_eq!(applied, GuidedToolConstraint::GuidedJsonRequired);
        assert!(
            common_request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.json.as_ref())
                .is_some()
        );
    }

    #[test]
    fn kimi_k3_builder_error_uses_native_fallback() {
        let request = request(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "$id": "urn:example:scoped",
                                "$defs": {
                                    "allowed": {"type": "string", "enum": ["safe"]}
                                },
                                "allOf": [{"$ref": "#/$defs/allowed"}]
                            }
                        },
                        "required": ["value"]
                    }
                }
            }],
            "tool_choice": "auto"
        }));
        let preprocessor = preprocessor("kimi_k3", StructuralTagMode::On);
        let tools = convert_tools(request.inner.tools.as_deref().unwrap());
        let ctx = dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
            tool_choice: &ToolChoice::Auto,
            tools: &tools,
            parallel_tool_calls: request.inner.parallel_tool_calls,
            schema_mode: preprocessor.runtime_config.structural_tag_schema,
            starts_in_reasoning: false,
        };
        // A scoped string reference cannot be safely rendered as raw XTML.
        // Verify an actual builder error, rather than a policy skip or Ok(None).
        let error = dynamo_parsers::tool_calling::StructuralTagBuilder::KimiK3
            .build_tool_call_format(&ctx)
            .expect_err("scoped string reference must reach the builder-error fallback");
        let message = format!("{error:#}");
        assert!(
            message.contains("value") && message.contains("string"),
            "{message}"
        );

        let mut common_request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .expect("builder errors must allow the request to continue without a structural tag");

        assert_eq!(applied, GuidedToolConstraint::None);
        assert!(common_request.sampling_options.guided_decoding.is_none());
    }

    fn mistral_preprocessor(llguidance_lark_grammar: bool) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-mistral-tekken");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.model_type = crate::model_type::ModelType::Chat;
        mdc.runtime_config.tool_call_parser = Some("mistral".to_string());
        mdc.runtime_config.reasoning_parser = Some("mistral".to_string());
        if llguidance_lark_grammar {
            mdc.runtime_config
                .set_engine_specific(LLGUIDANCE_LARK_GRAMMAR_RUNTIME_KEY, true)
                .unwrap();
        }
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    #[test]
    fn unified_mistral_forced_choices_install_the_native_grammar() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::unified_mistral_forced_choices_install_the_native_grammar"
            ),
            &[],
        ) {
            return;
        }
        let preprocessor = mistral_preprocessor(true);
        for (tool_choice, name) in [
            (json!("required"), "/.+/"),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                "\"get_weather\"",
            ),
        ] {
            let request = request(json!({"tools": tools(), "tool_choice": tool_choice}));
            let mut common_request = preprocessed_request();

            let applied = preprocessor
                .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
                .unwrap();

            assert_eq!(applied, GuidedToolConstraint::None);
            let guided = common_request.sampling_options.guided_decoding.unwrap();
            assert!(guided.json.is_none());
            let grammar = guided.grammar.unwrap();
            assert!(
                grammar.starts_with("start: body\nbody: think? fcalls\n")
                    && grammar.contains(&format!("(<[9]> SAFE_WS? {name} <[32]>")),
                "{grammar}"
            );
            assert_eq!(common_request.extra_args.unwrap()["reasoning_ended"], true);
            assert_eq!(
                guided_tool_constraint(&request, Some("mistral"), Some("mistral"), false).unwrap(),
                GuidedToolConstraint::None
            );
        }

        // A backend without llguidance lark grammars gets no grammar and keeps
        // native parsing instead of the JSON schema the parser cannot read.
        let request = request(json!({"tools": tools(), "tool_choice": "required"}));
        let mut common_request = preprocessed_request();
        let applied = mistral_preprocessor(false)
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();
        assert_eq!(applied, GuidedToolConstraint::None);
        assert!(common_request.sampling_options.guided_decoding.is_none());
    }

    /// Upstream vLLM constrains `auto` with the same grammar, so call arguments
    /// stay JSON objects.
    #[test]
    fn unified_mistral_auto_with_tools_installs_the_native_grammar() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::unified_mistral_auto_with_tools_installs_the_native_grammar"
            ),
            &[],
        ) {
            return;
        }
        let preprocessor = mistral_preprocessor(true);
        for extra in [
            json!({"tools": tools(), "tool_choice": "auto"}),
            json!({"tools": tools()}),
        ] {
            let request = request(extra);
            let mut common_request = preprocessed_request();

            let applied = preprocessor
                .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
                .unwrap();

            assert_eq!(applied, GuidedToolConstraint::None);
            let guided = common_request.sampling_options.guided_decoding.unwrap();
            assert!(guided.json.is_none());
            let expected = preprocessor.mistral_tool_call_grammar.unwrap().render(
                CallMode::Auto,
                convert_tools(request.inner.tools.as_deref().unwrap()),
                true,
                None,
            );
            assert_eq!(guided.grammar.unwrap(), expected);
            assert_eq!(common_request.extra_args.unwrap()["reasoning_ended"], true);
        }
    }

    /// Requests without tools or a reply schema, `none` without one, and
    /// constraints the grammar does not carry get no grammar.
    #[test]
    fn unified_mistral_leaves_unconstrained_requests_alone() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::unified_mistral_leaves_unconstrained_requests_alone"
            ),
            &[],
        ) {
            return;
        }
        let preprocessor = mistral_preprocessor(true);
        for extra in [
            json!({}),
            json!({"tools": tools(), "tool_choice": "none"}),
            json!({"tools": tools(), "guided_regex": "[a-z]+"}),
            json!({
                "tools": tools(),
                "guided_json": {"type": "object"},
                "guided_whitespace_pattern": " "
            }),
        ] {
            let request = request(extra.clone());
            let mut common_request = preprocessed_request();
            preprocessor
                .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
                .unwrap();
            assert!(
                common_request
                    .sampling_options
                    .guided_decoding
                    .and_then(|guided| guided.grammar)
                    .is_none(),
                "{extra}"
            );
            assert!(common_request.extra_args.is_none(), "{extra}");
        }

        let request = request(json!({"tools": tools(), "tool_choice": "auto"}));
        let mut common_request = preprocessed_request();
        mistral_preprocessor(false)
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();
        assert!(common_request.sampling_options.guided_decoding.is_none());
    }

    /// `guided_json` as preprocessing sets it from the request's reply schema.
    fn guided_json(schema: &Value) -> Option<crate::protocols::common::GuidedDecodingOptions> {
        crate::protocols::common::GuidedDecodingOptions::from_optional(
            Some(schema.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    /// A JSON schema for the reply joins the Mistral grammar, as upstream vLLM
    /// folds it, instead of reaching the backend as a separate `guided_json`.
    #[test]
    fn unified_mistral_folds_reply_schemas_into_the_grammar() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::unified_mistral_folds_reply_schemas_into_the_grammar"
            ),
            &[],
        ) {
            return;
        }
        let preprocessor = mistral_preprocessor(true);
        let grammar = preprocessor.mistral_tool_call_grammar.unwrap();
        let event = json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let schema_format = json!({
            "type": "json_schema",
            "json_schema": {"name": "event", "schema": event}
        });
        let tool_definitions = convert_tools(
            request(json!({"tools": tools()}))
                .inner
                .tools
                .as_deref()
                .unwrap(),
        );
        for (extra, expected) in [
            (
                json!({"response_format": schema_format}),
                grammar.render_json(&event),
            ),
            (
                json!({"response_format": {"type": "json_object"}}),
                grammar.render_json(&ANY_OBJECT_OR_ARRAY),
            ),
            (
                json!({"tools": tools(), "guided_json": event.to_string()}),
                grammar.render(CallMode::Auto, tool_definitions.clone(), true, Some(&event)),
            ),
            (
                json!({"tools": tools(), "tool_choice": "required", "response_format": schema_format}),
                grammar.render(
                    CallMode::Required,
                    tool_definitions.clone(),
                    true,
                    Some(&event),
                ),
            ),
            (
                json!({
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "anything", "schema": {}}
                    }
                }),
                grammar.render_json(&json!({})),
            ),
        ] {
            let request = request(extra.clone());
            let mut common_request = preprocessed_request();
            common_request.sampling_options.guided_decoding = guided_json(&event);

            let applied = preprocessor
                .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
                .unwrap();

            assert_eq!(applied, GuidedToolConstraint::None, "{extra}");
            let guided = common_request.sampling_options.guided_decoding.unwrap();
            assert!(guided.json.is_none(), "{extra}");
            assert_eq!(
                guided.grammar.as_deref(),
                Some(expected.as_str()),
                "{extra}"
            );
            assert_eq!(
                common_request.extra_args.unwrap()["reasoning_ended"],
                true,
                "{extra}"
            );
        }

        // A forced choice still rejects explicit guided decoding.
        let forced = request(json!({
            "tools": tools(),
            "tool_choice": "required",
            "guided_json": event
        }));
        let error = preprocessor
            .apply_tool_choice_guided_decoding(&forced, &mut preprocessed_request(), false)
            .unwrap_err();
        assert_eq!(error.error_type(), ErrorType::InvalidArgument);

        // A whitespace pattern keeps the forced call's grammar without a reply
        // schema.
        let forced = request(json!({
            "tools": tools(),
            "tool_choice": "required",
            "guided_whitespace_pattern": " "
        }));
        let mut common_request = preprocessed_request();
        preprocessor
            .apply_tool_choice_guided_decoding(&forced, &mut common_request, false)
            .unwrap();
        assert_eq!(
            common_request
                .sampling_options
                .guided_decoding
                .unwrap()
                .grammar,
            Some(grammar.render(CallMode::Required, tool_definitions, true, None))
        );

        // A backend without llguidance lark grammars keeps the reply schema as
        // `guided_json`.
        let auto = request(json!({"tools": tools(), "response_format": schema_format}));
        let mut common_request = preprocessed_request();
        common_request.sampling_options.guided_decoding = guided_json(&event);
        mistral_preprocessor(false)
            .apply_tool_choice_guided_decoding(&auto, &mut common_request, false)
            .unwrap();
        let guided = common_request.sampling_options.guided_decoding.unwrap();
        assert_eq!(guided.json, Some(event));
        assert!(guided.grammar.is_none());
    }

    /// `mistral` also names the v1 parsers, which keep the JSON-schema fallback.
    #[test]
    fn v1_mistral_forced_choices_keep_the_json_schema_fallback() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::v1_mistral_forced_choices_keep_the_json_schema_fallback"
            ),
            &[(env_llm::DYN_PARSER_VERSION, "1")],
        ) {
            return;
        }
        let preprocessor = mistral_preprocessor(true);
        let request = request(json!({"tools": tools(), "tool_choice": "required"}));
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();

        assert_eq!(applied, GuidedToolConstraint::GuidedJsonRequired);
        let guided = common_request.sampling_options.guided_decoding.unwrap();
        assert!(guided.json.is_some() && guided.grammar.is_none());
        assert!(common_request.extra_args.is_none());
        assert_eq!(
            guided_tool_constraint(&request, Some("mistral"), Some("mistral"), false).unwrap(),
            GuidedToolConstraint::GuidedJsonRequired
        );

        let auto = self::request(json!({"tools": tools(), "tool_choice": "auto"}));
        let mut common_request = preprocessed_request();
        preprocessor
            .apply_tool_choice_guided_decoding(&auto, &mut common_request, false)
            .unwrap();
        assert!(common_request.sampling_options.guided_decoding.is_none());
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

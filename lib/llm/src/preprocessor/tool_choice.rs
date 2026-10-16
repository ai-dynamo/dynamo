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
    /// every request permission to return tool calls. Permission depends only on
    /// whether the request supplies tools and whether `tool_choice` forbids them.
    /// Assistant-output constraints apply to assistant content and do not revoke
    /// an `auto` request's ability to choose a tool call.
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
    /// required requests, which stay on the prompt-level XTML path, and Muse
    /// Spark forced requests, which generate native ATEM tool calls.
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
            &convert_tools(tools),
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

        if is_forced_tool_choice
            && uses_muse_spark_parser(
                self.tool_call_parser.as_deref(),
                self.runtime_config.reasoning_parser.as_deref(),
            )
        {
            validate_openai_tool_choice(Some(tool_choice), request.inner.tools.as_deref())
                .map_err(|error| invalid_argument(error.to_string()))?;
            // Native Muse serving skips the generic JSON/regex tool grammar.
            // Keep required/named intent, but parse generated calls as ATEM.
            return Ok(GuidedToolConstraint::None);
        }

        match get_tool_choice_guidance_from_tools(
            Some(tool_choice),
            Some(tools),
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
    validate_openai_tool_choice(Some(tool_choice), request.inner.tools.as_deref())
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
    if uses_muse_spark_parser(tool_call_parser, reasoning_parser) {
        return Ok(GuidedToolConstraint::None);
    }
    // Validate the forced choice against the actual tools the same way
    // `apply_tool_choice_guided_decoding` does, instead of blindly installing a
    // constraint for a `tool_choice` that names a tool absent from `tools` (or an
    // empty `tools` list under `tool_choice: "required"`).
    let tools = request.inner.tools.as_deref().unwrap_or(&[]);
    match get_tool_choice_guidance_from_tools(
        Some(tool_choice),
        Some(tools),
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

/// Follow the same family precedence as the native ATEM response route.
pub(super) fn uses_muse_spark_parser(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> bool {
    crate::protocols::openai::chat_completions::tool_parser_v2::unified_family(
        tool_call_parser,
        reasoning_parser,
    )
    .as_deref()
        == Some("muse_spark")
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

    fn preprocessor(
        tool_parser: Option<&str>,
        reasoning_parser: Option<&str>,
    ) -> std::sync::Arc<OpenAIPreprocessor> {
        let mut mdc = crate::model_card::ModelDeploymentCard::load_from_disk(
            "tests/data/sample-models/mock-llama-3.1-8b-instruct",
            None,
        )
        .unwrap();
        mdc.runtime_config.tool_call_parser = tool_parser.map(str::to_owned);
        mdc.runtime_config.reasoning_parser = reasoning_parser.map(str::to_owned);
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    #[tokio::test]
    async fn muse_spark_open_forced_requests_keep_native_atem() {
        let preprocessor = preprocessor(Some("muse_spark_open"), Some("muse_spark_open"));
        for choice in [
            json!("required"),
            serde_json::to_value(named("get_weather")).unwrap(),
        ] {
            let request = request(json!({"tools": tools(), "tool_choice": choice}));
            let (mut common, _, reasoning) = preprocessor
                .preprocess_request(&request, None)
                .await
                .unwrap();
            assert_eq!(
                preprocessor
                    .apply_tool_choice_guided_decoding(&request, &mut common, reasoning)
                    .unwrap(),
                GuidedToolConstraint::None,
                "the open parser must not reinstall the incompatible JSON tool grammar"
            );
            assert!(common.sampling_options.guided_decoding.is_none());
        }
    }

    #[test]
    fn muse_spark_required_constraint_is_native() {
        let request = request(json!({"tools": tools(), "tool_choice": "required"}));
        for (tool_parser, reasoning_parser) in [
            (Some("muse_spark"), None),
            (None, Some("muse_spark")),
            (Some("muse_spark"), Some("muse_spark")),
        ] {
            assert_eq!(
                guided_tool_constraint(&request, tool_parser, reasoning_parser, false).unwrap(),
                GuidedToolConstraint::None,
                "Muse Spark required output must use the native ATEM parser",
            );
        }
    }

    #[test]
    fn muse_spark_named_constraint_is_native_including_zero_arg_tools() {
        for parameters in [
            json!({"type": "object", "properties": {"location": {"type": "string"}}}),
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        ] {
            let request = request(json!({
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "parameters": parameters
                }}],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
            }));
            assert_eq!(
                guided_tool_constraint(&request, Some("muse_spark"), None, false).unwrap(),
                GuidedToolConstraint::None,
            );
        }
    }

    #[tokio::test]
    async fn muse_spark_forced_choices_do_not_install_tool_grammars() {
        let preprocessor = preprocessor(Some("muse_spark"), Some("muse_spark"));
        for choice in [
            json!("required"),
            serde_json::to_value(named("get_weather")).unwrap(),
        ] {
            for parallel in [Value::Null, json!(true), json!(false)] {
                let request = request(json!({
                    "tools": tools(), "tool_choice": choice, "parallel_tool_calls": parallel
                }));
                let (mut common, _, reasoning) = preprocessor
                    .preprocess_request(&request, None)
                    .await
                    .unwrap();
                let constraint = preprocessor
                    .apply_tool_choice_guided_decoding(&request, &mut common, reasoning)
                    .unwrap();
                assert!(
                    common
                        .sampling_options
                        .guided_decoding
                        .as_ref()
                        .is_none_or(|gd| gd.json.is_none()
                            && gd.regex.is_none()
                            && gd.structural_tag.is_none()),
                    "Muse Spark must generate native ATEM without a synthetic tool grammar: {:?}",
                    common.sampling_options.guided_decoding,
                );
                assert_eq!(constraint, GuidedToolConstraint::None);
                assert_eq!(
                    request.inner.tool_choice.as_ref(),
                    Some(&serde_json::from_value(choice.clone()).unwrap())
                );
            }
        }
    }

    #[tokio::test]
    async fn muse_spark_bypass_preserves_invalid_tool_choice_errors() {
        let preprocessor = preprocessor(Some("muse_spark"), None);
        for extra in [
            json!({"tool_choice": "required"}),
            json!({"tools": [], "tool_choice": "required"}),
            json!({"tools": tools(), "tool_choice": {"type": "function", "function": {"name": "missing"}}}),
        ] {
            let request = request(extra);
            assert!(guided_tool_constraint(&request, Some("muse_spark"), None, false).is_err());
            let (mut common, _, reasoning) = preprocessor
                .preprocess_request(&request, None)
                .await
                .unwrap();
            assert!(
                preprocessor
                    .apply_tool_choice_guided_decoding(&request, &mut common, reasoning)
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn muse_spark_bypass_preserves_assistant_json_schema() {
        let preprocessor = preprocessor(Some("muse_spark"), None);
        let schema = json!({"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]});
        for choice in [json!("auto"), json!("none")] {
            let request = request(json!({
                "tools": tools(), "tool_choice": choice,
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
            }));
            let (mut common, _, reasoning) = preprocessor
                .preprocess_request(&request, None)
                .await
                .unwrap();
            assert_eq!(
                preprocessor
                    .apply_tool_choice_guided_decoding(&request, &mut common, reasoning)
                    .unwrap(),
                GuidedToolConstraint::None
            );
            assert_eq!(
                common
                    .sampling_options
                    .guided_decoding
                    .as_ref()
                    .and_then(|gd| gd.json.as_ref()),
                Some(&schema)
            );
        }
    }

    #[tokio::test]
    async fn other_parsers_keep_forced_tool_guidance() {
        for parser in [None, Some("hermes"), Some("muse_glimmer")] {
            let preprocessor = preprocessor(parser, None);
            for choice in [
                json!("required"),
                serde_json::to_value(named("get_weather")).unwrap(),
            ] {
                let request = request(json!({"tools": tools(), "tool_choice": choice}));
                let (mut common, _, reasoning) = preprocessor
                    .preprocess_request(&request, None)
                    .await
                    .unwrap();
                let expected =
                    installed_json_constraint(request.inner.tool_choice.as_ref().unwrap());
                assert_eq!(
                    preprocessor
                        .apply_tool_choice_guided_decoding(&request, &mut common, reasoning)
                        .unwrap(),
                    expected
                );
                assert!(
                    common
                        .sampling_options
                        .guided_decoding
                        .as_ref()
                        .and_then(|gd| gd.json.as_ref())
                        .is_some()
                );
                assert_eq!(
                    guided_tool_constraint(&request, parser, None, false).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn muse_spark_named_prompt_exposes_only_selected_tool() {
        let template =
            serde_json::from_value(json!({"chat_template": "{{ tools | tojson }}"})).unwrap();
        let dynamo_renderer::PromptFormatter::OAI(formatter) =
            dynamo_renderer::PromptFormatter::from_parts(template, Default::default(), false)
                .unwrap();
        let mut request = request(json!({
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
                {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        for parser in ["muse_spark", "hermes"] {
            let mut mdc = crate::model_card::ModelDeploymentCard::load_from_disk(
                "tests/data/sample-models/mock-llama-3.1-8b-instruct",
                None,
            )
            .unwrap();
            mdc.runtime_config.tool_call_parser = Some(parser.to_owned());
            let tokenizer = mdc.tokenizer().unwrap();
            let preprocessor =
                OpenAIPreprocessor::new_with_parts(mdc, formatter.clone(), tokenizer).unwrap();
            let prompt = preprocessor.apply_template(&request).unwrap().unwrap();
            let tools: Value = serde_json::from_str(prompt.as_str()).unwrap();
            assert_eq!(
                tools.as_array().unwrap().len(),
                if parser == "muse_spark" { 1 } else { 2 }
            );
            assert_eq!(tools[0]["function"]["name"], "get_weather");
            assert_eq!(request.inner.tool_choice, Some(named("get_weather")));
            request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);
            let prompt = preprocessor.apply_template(&request).unwrap().unwrap();
            assert!(
                prompt.as_str().contains("get_time"),
                "required must retain the whole registry"
            );
            request.inner.tool_choice = Some(named("get_weather"));
        }
        assert_eq!(
            request.inner.tools.as_ref().unwrap().len(),
            2,
            "prompt filtering must not mutate the request registry"
        );
    }

    #[test]
    fn muse_spark_named_responses_prompt_keeps_qualified_name_and_full_schema() {
        let response: crate::protocols::openai::responses::NvCreateResponse = serde_json::from_value(json!({
            "model": "test-model", "input": "Get the weather",
            "tools": [{"type": "namespace", "name": "weather", "defer_loading": true, "tools": [
                {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
                {"type": "function", "name": "get_time", "parameters": {"type": "object", "properties": {}}}
            ]}],
            "tool_choice": {"type": "function", "name": "weather.get_weather"}
        })).unwrap();
        let mut request: NvCreateChatCompletionRequest = response.try_into().unwrap();
        request.normalize_responses_tools(true).unwrap();
        let template = serde_json::from_value(json!({"chat_template": "{% set deferred_render_mode = deferred_render_mode | default('tool_name_and_description') %}{{ tools | tojson }}"})).unwrap();
        let dynamo_renderer::PromptFormatter::OAI(formatter) =
            dynamo_renderer::PromptFormatter::from_parts(template, Default::default(), false)
                .unwrap();
        let mut mdc = crate::model_card::ModelDeploymentCard::load_from_disk(
            "tests/data/sample-models/mock-llama-3.1-8b-instruct",
            None,
        )
        .unwrap();
        mdc.runtime_config.tool_call_parser = Some("muse_spark".to_owned());
        let tokenizer = mdc.tokenizer().unwrap();
        let preprocessor = OpenAIPreprocessor::new_with_parts(mdc, formatter, tokenizer).unwrap();
        let prompt = preprocessor.apply_template(&request).unwrap().unwrap();
        let tools: Value = serde_json::from_str(prompt.as_str()).unwrap();
        assert_eq!(tools.as_array().unwrap().len(), 1);
        assert_eq!(tools[0]["function"]["name"], "weather.get_weather");
        assert_eq!(
            tools[0]["function"]["parameters"]["required"],
            json!(["location"])
        );
        assert_eq!(request.inner.tools.as_ref().unwrap().len(), 2);
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
}

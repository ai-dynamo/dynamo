// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

mod v1;
mod v2;

use crate::local_model::runtime_config::{
    ModelRuntimeConfig, StructuralTagConfig, StructuralTagReasoningBoundary, StructuralTagScope,
    TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::tools::{ToolChoiceValidation, validate_tool_choice_against_names};

use dynamo_parsers::tool_calling::{StructuralTagSchemaMode, ToolChoice, ToolDefinition};
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::error::{DynamoError, ErrorType};

struct StructuralTagBuildRequest<'a> {
    tool_choice: &'a ToolChoice,
    tools: &'a [ToolDefinition],
    parallel_tool_calls: Option<bool>,
    schema_mode: StructuralTagSchemaMode,
    exclude_special_tokens: Option<bool>,
    reasoning_boundary: StructuralTagReasoningBoundary,
    tool_arguments_any_order: bool,
    starts_in_reasoning: bool,
    structured_output_schema: Option<&'a serde_json::Value>,
}

#[derive(Clone, Copy)]
pub(crate) enum SelectedStructuralTagBuilder {
    V1(v1::StructuralTagBuilder),
    V2(v2::StructuralTagBuilder),
}

impl SelectedStructuralTagBuilder {
    fn for_parser(parser_name: &str) -> Option<Self> {
        if v2::enabled()
            && v2::supports_family(parser_name)
            && let Some(builder) = v2::StructuralTagBuilder::for_parser(parser_name)
        {
            return Some(Self::V2(builder));
        }

        v1::StructuralTagBuilder::for_parser(parser_name).map(Self::V1)
    }

    fn build(
        self,
        request: &StructuralTagBuildRequest<'_>,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        match self {
            Self::V1(builder) => builder.build(request),
            Self::V2(builder) => builder.build(request),
        }
    }
}

/// Validate a forced `tool_choice` against the request's actual `tools` list.
///
/// `Required` with no tools, and a `Named` choice for a tool absent from `tools`,
/// have nothing valid to constrain against. The rule itself lives in
/// `validate_tool_choice_against_names`, which is also used by the OpenAI-wire
/// schema builder; this adapter only maps the parser-facing types and error.
fn validate_forced_tool_choice(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
) -> Result<(), DynamoError> {
    let tool_choice = match tool_choice {
        ToolChoice::Required => ToolChoiceValidation::Required,
        ToolChoice::Named(name) => ToolChoiceValidation::Named(name),
        ToolChoice::None | ToolChoice::Auto => ToolChoiceValidation::Unforced,
    };
    validate_tool_choice_against_names(tool_choice, tools.iter().map(|tool| tool.name.as_str()))
        .map_err(|error| {
            DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .message(error.to_string())
                .build()
        })
}

fn is_kimi_k3_parser(parser_name: Option<&str>) -> bool {
    parser_name.is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"))
}

// Unlike K3, the parser registry exposes only the canonical `kimi_k2` spelling.
fn is_kimi_k2_parser(parser_name: Option<&str>) -> bool {
    parser_name == Some("kimi_k2")
}

fn requires_intrinsic_structural_tag(parser_name: Option<&str>, tool_choice: &ToolChoice) -> bool {
    // K2 forced calls and K3 named calls cannot use Dynamo's generic JSON-schema
    // fallback because both families emit native, marker-delimited formats.
    // Treat their structural tags as part of implementing these standard OpenAI
    // request shapes, not as an operator opt-in. K3 required remains on its
    // intentional prompt-level XTML path.
    (is_kimi_k2_parser(parser_name)
        && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)))
        || (is_kimi_k3_parser(parser_name) && matches!(tool_choice, ToolChoice::Named(_)))
}

fn should_skip_tool_call_ban(exclude_tools_when_none: bool, tool_choice: &ToolChoice) -> bool {
    exclude_tools_when_none && matches!(tool_choice, ToolChoice::None)
}

/// Whether a request is entitled to use a structural tag, bundled with proof the
/// parser registry can actually build one.
///
/// This is the single owner of "is a structural tag applicable and available" —
/// covering model-family/tool-choice intrinsic eligibility, the configured
/// structural-tag policy, tools-with-structured-output composition, the
/// `tool_choice=None` ban-tag exclusion, and real parser-registry builder
/// availability. A caller can only reach `Required` with a registered builder
/// selected for the active parser generation — it cannot ask for the eligibility
/// half of this decision without also getting the registry-availability proof in
/// the same value. Both the real
/// preprocessing path
/// (`apply_tool_choice_structural_tag`) and the HTTP-layer reconstruction
/// (`http::service::apply_request_tool_call_parsing_options`) must consult this
/// function; neither may re-derive eligibility, mode, scope, or registry
/// availability independently.
pub(crate) enum StructuralTagDecision {
    Required(SelectedStructuralTagBuilder),
    NotApplicable,
}

impl StructuralTagDecision {
    pub(crate) fn is_required(&self) -> bool {
        matches!(self, Self::Required(_))
    }
}

/// Decide whether a structural tag applies to this request. See
/// [`StructuralTagDecision`] for why this is the single shared owner.
///
/// Returns `Err` when `tool_choice` is a forced choice (`Required` or `Named`)
/// that is not valid against the request's `tools` — e.g. `required` with no
/// tools, or a named tool absent from `tools`. Both the real preprocessing path
/// (`apply_tool_choice_structural_tag`) and the HTTP-layer reconstruction
/// (`http::service::apply_request_tool_call_parsing_options`) must propagate
/// this error rather than falling back to `NotApplicable`, or an invalid forced
/// choice would silently install a structural tag anyway.
pub(crate) fn structural_tag_decision(
    parser_name: Option<&str>,
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    config: Option<&StructuralTagConfig>,
    has_structured_output: bool,
    exclude_tools_when_tool_choice_none: bool,
) -> Result<StructuralTagDecision, DynamoError> {
    // Validate before any mode or family gate. Kimi K3 required requests use a
    // prompt-level XTML path, but they still need an actual tool to require.
    validate_forced_tool_choice(tool_choice, tools)?;

    if config.is_none() && !requires_intrinsic_structural_tag(parser_name, tool_choice) {
        return Ok(StructuralTagDecision::NotApplicable);
    }

    let requires_tool_calls_with_structured_output = if has_structured_output {
        match tool_choice {
            ToolChoice::Auto => {
                if config.is_some_and(|config| config.allow_tool_calls_with_structured_output)
                    && !tools.is_empty()
                {
                    true
                } else {
                    return Ok(StructuralTagDecision::NotApplicable);
                }
            }
            ToolChoice::None => return Ok(StructuralTagDecision::NotApplicable),
            ToolChoice::Required | ToolChoice::Named(_) => false,
        }
    } else {
        false
    };

    if should_skip_tool_call_ban(exclude_tools_when_tool_choice_none, tool_choice) {
        // The prompt formatter already omits tools for this request. Avoid
        // sending a redundant AnyTokens structural tag: vLLM cannot
        // validate token-string exclusions without tokenizer metadata.
        return Ok(StructuralTagDecision::NotApplicable);
    }

    let Some(parser_name) = parser_name else {
        tracing::warn!(
            "Structural tag is enabled but --dyn-tool-call-parser is not set; \
             structural tags will not be applied"
        );
        return Ok(StructuralTagDecision::NotApplicable);
    };

    let Some(builder) = SelectedStructuralTagBuilder::for_parser(parser_name) else {
        return Ok(StructuralTagDecision::NotApplicable);
    };

    if matches!(tool_choice, ToolChoice::None) {
        if tools.is_empty() {
            return Ok(StructuralTagDecision::NotApplicable);
        }
        return Ok(StructuralTagDecision::Required(builder));
    }

    if requires_tool_calls_with_structured_output {
        return Ok(StructuralTagDecision::Required(builder));
    }

    let scope = config.map_or_else(Default::default, |config| config.scope);

    if !OpenAIPreprocessor::should_apply_tool_call_format(
        scope,
        tool_choice,
        tools,
        parallel_tool_calls,
    ) {
        return Ok(StructuralTagDecision::NotApplicable);
    }

    Ok(StructuralTagDecision::Required(builder))
}

impl OpenAIPreprocessor {
    /// Apply structural tag guided decoding when enabled for this request.
    pub(super) fn apply_tool_choice_structural_tag(
        &self,
        tool_choice: &ToolChoice,
        tools: &[ToolDefinition],
        parallel_tool_calls: Option<bool>,
        prompt_injected_reasoning: bool,
        structured_output_schema: Option<&serde_json::Value>,
        preprocessed_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        let parser_name = self.tool_call_parser.as_deref();
        let explicit_config = self.runtime_config.structural_tag.as_ref();

        let mut config = explicit_config.cloned().unwrap_or_default();
        if explicit_config.is_none() && self.backend_excludes_reasoning_from_structural_tag() {
            config.reasoning_boundary = StructuralTagReasoningBoundary::Backend;
        }
        let StructuralTagDecision::Required(builder) = structural_tag_decision(
            parser_name,
            tool_choice,
            tools,
            parallel_tool_calls,
            explicit_config,
            structured_output_schema.is_some(),
            self.runtime_config.exclude_tools_when_tool_choice_none,
        )?
        else {
            return Ok(false);
        };
        let parser_name = parser_name.expect("Required decision implies a parser name");

        let structured_output_schema = if matches!(tool_choice, ToolChoice::Auto)
            && config.allow_tool_calls_with_structured_output
        {
            structured_output_schema
        } else {
            None
        };

        let request = StructuralTagBuildRequest {
            tool_choice,
            tools,
            parallel_tool_calls,
            schema_mode: config.schema,
            exclude_special_tokens: config.exclude_special_tokens,
            reasoning_boundary: config.reasoning_boundary,
            tool_arguments_any_order: config.tool_arguments_any_order,
            starts_in_reasoning: prompt_injected_reasoning,
            structured_output_schema,
        };

        let applied =
            apply_structural_tag(parser_name, builder.build(&request), preprocessed_request)?;
        if applied && structured_output_schema.is_some() {
            // Request preprocessing materializes `response_format` as guided JSON.
            // The composite structural tag now owns that schema.
            preprocessed_request
                .sampling_options
                .guided_decoding
                .get_or_insert_default()
                .json = None;
        }
        if applied && prompt_injected_reasoning {
            preprocessed_request.require_reasoning =
                config.reasoning_boundary == StructuralTagReasoningBoundary::Backend;
        }
        Ok(applied)
    }

    fn backend_excludes_reasoning_from_structural_tag(&self) -> bool {
        match self
            .runtime_config
            .get_engine_specific::<bool>(TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY)
        {
            Ok(Some(excludes_reasoning)) => excludes_reasoning,
            Ok(None) => false,
            Err(error) => {
                tracing::warn!(
                    %error,
                    key = TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
                    "Ignoring invalid structural-tag reasoning metadata"
                );
                false
            }
        }
    }

    /// Decide whether this request should use a tool-call format tag.
    fn should_apply_tool_call_format(
        scope: StructuralTagScope,
        tool_choice: &ToolChoice,
        tools: &[ToolDefinition],
        parallel_tool_calls: Option<bool>,
    ) -> bool {
        match tool_choice {
            ToolChoice::None => false,
            ToolChoice::Required | ToolChoice::Named(_) => true,
            ToolChoice::Auto => match scope {
                StructuralTagScope::Always => true,
                StructuralTagScope::Auto => {
                    let explicit_single_call = parallel_tool_calls == Some(false);
                    tools.iter().any(|t| t.strict.unwrap_or(false)) || explicit_single_call
                }
            },
        }
    }
}

fn apply_structural_tag(
    parser_name: &str,
    structural_tag: anyhow::Result<Option<serde_json::Value>>,
    request: &mut PreprocessedRequest,
) -> Result<bool, DynamoError> {
    let structural_tag = match structural_tag {
        Ok(Some(tag)) => tag,
        Ok(None) => return Ok(false),
        Err(error) => {
            return Err(DynamoError::builder()
                .error_type(ErrorType::Unknown)
                .message(format!(
                    "failed to build structural_tag for parser '{parser_name}': {error}"
                ))
                .build());
        }
    };

    let guided_decoding = request
        .sampling_options
        .guided_decoding
        .get_or_insert_default();
    guided_decoding.structural_tag = Some(structural_tag);
    Ok(true)
}

pub(super) fn validate_runtime_config(runtime_config: &ModelRuntimeConfig) -> anyhow::Result<()> {
    let Some(config) = runtime_config.structural_tag.as_ref() else {
        return Ok(());
    };

    let v2_only_option = config
        .allow_tool_calls_with_structured_output
        .then_some("allow_tool_calls_with_structured_output")
        .or(config
            .exclude_special_tokens
            .is_some()
            .then_some("exclude_special_tokens"))
        .or(
            (config.reasoning_boundary == StructuralTagReasoningBoundary::Backend)
                .then_some("reasoning_boundary"),
        )
        .or(config
            .tool_arguments_any_order
            .then_some("tool_arguments_any_order"));
    let Some(v2_only_option) = v2_only_option else {
        return Ok(());
    };

    anyhow::ensure!(
        v2::enabled(),
        "structural_tag.{v2_only_option} requires {}=1",
        env_llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2,
    );
    let parser_name = runtime_config.tool_call_parser.as_deref().ok_or_else(|| {
        anyhow::anyhow!("structural_tag.{v2_only_option} requires a tool-call parser")
    })?;
    anyhow::ensure!(
        v2::supports_family(parser_name),
        "structural_tag.{v2_only_option} is not supported by parser '{parser_name}'"
    );
    anyhow::ensure!(
        v2::StructuralTagBuilder::for_parser(parser_name).is_some(),
        "parser '{parser_name}' does not provide a parsers-v2 structural-tag builder"
    );

    if config.reasoning_boundary == StructuralTagReasoningBoundary::Backend {
        anyhow::ensure!(
            runtime_config.reasoning_parser.is_some(),
            "structural_tag.reasoning_boundary=backend requires a reasoning parser"
        );
        let capability = runtime_config
            .get_engine_specific::<bool>(TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY)?
            .unwrap_or(false);
        anyhow::ensure!(
            capability,
            "structural_tag.reasoning_boundary=backend is not supported by this backend"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use crate::{
        local_model::runtime_config::StructuralTagConfig,
        model_card::ModelDeploymentCard,
        protocols::common::{OutputOptions, SamplingOptions, StopConditions},
    };

    use super::*;

    fn structural_tag_preprocessor(exclude_tools_when_none: bool) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = Some(StructuralTagConfig::default());
        mdc.runtime_config.tool_call_parser = Some("qwen3_coder".to_string());
        mdc.runtime_config.exclude_tools_when_tool_choice_none = exclude_tools_when_none;

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

    fn kimi_k2_preprocessor(excludes_reasoning: Option<bool>) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = Some(StructuralTagConfig::default());
        mdc.runtime_config.tool_call_parser = Some("kimi_k2".to_string());
        if let Some(excludes_reasoning) = excludes_reasoning {
            mdc.runtime_config
                .set_engine_specific(
                    TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
                    excludes_reasoning,
                )
                .unwrap();
        }

        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn kimi_k3_preprocessor() -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.tool_call_parser = Some("kimi_k3".to_string());
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn kimi_k2_required_format(excludes_reasoning: Option<bool>) -> serde_json::Value {
        let preprocessor = kimi_k2_preprocessor(excludes_reasoning);
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        assert!(
            preprocessor
                .apply_tool_choice_structural_tag(
                    &ToolChoice::Required,
                    &tools,
                    None,
                    true,
                    None,
                    &mut request,
                )
                .unwrap()
        );

        request
            .sampling_options
            .guided_decoding
            .unwrap()
            .structural_tag
            .unwrap()["format"]
            .clone()
    }

    #[test]
    fn reasoning_metadata_controls_whether_forced_tool_tag_models_reasoning() {
        for policy in [None, Some(false)] {
            let format = kimi_k2_required_format(policy);
            assert_eq!(format["type"], "sequence");
            assert_eq!(format["elements"][0]["type"], "tag");
            assert_eq!(format["elements"][0]["end"], "</think>");
        }

        let format = kimi_k2_required_format(Some(true));
        assert_eq!(format["type"], "sequence");
        assert_eq!(format["elements"][0]["type"], "const_string");
        assert_eq!(
            format["elements"][0]["value"],
            "<|tool_calls_section_begin|>"
        );
    }

    #[test]
    fn named_kimi_k3_is_intrinsic_even_when_global_mode_is_off() {
        let named = ToolChoice::Named("get_weather".to_string());
        assert!(requires_intrinsic_structural_tag(Some("kimi_k3"), &named));
        assert!(requires_intrinsic_structural_tag(Some("kimi-k3"), &named));
    }

    #[test]
    fn forced_kimi_k2_is_intrinsic_even_when_global_mode_is_off() {
        assert!(requires_intrinsic_structural_tag(
            Some("kimi_k2"),
            &ToolChoice::Required
        ));
        assert!(requires_intrinsic_structural_tag(
            Some("kimi_k2"),
            &ToolChoice::Named("get_weather".to_string())
        ));
    }

    // Durable registry-parity property: every parser/tool_choice combination the
    // intrinsic predicate recognizes must have a real, registered
    // `StructuralTagBuilder` in the parser registry. If a future edit registers a
    // new intrinsic-family alias, or de-registers an existing one, without keeping
    // both in lockstep, `structural_tag_decision` would otherwise report `Required`
    // with no way to actually build the tag — this is the exact gap the shared
    // decision owner exists to close. No global-state mutation, no scratch worktree,
    // runs against the real production registry.
    #[test]
    fn every_intrinsic_parser_choice_has_a_registered_structural_tag_builder() {
        let cases: &[(&str, ToolChoice)] = &[
            ("kimi_k2", ToolChoice::Required),
            ("kimi_k2", ToolChoice::Named("get_weather".to_string())),
            ("kimi_k3", ToolChoice::Named("get_weather".to_string())),
            ("kimi-k3", ToolChoice::Named("get_weather".to_string())),
        ];
        for (parser, choice) in cases {
            assert!(
                requires_intrinsic_structural_tag(Some(parser), choice),
                "test fixture drifted: '{parser}' + {choice:?} is no longer intrinsic"
            );
            assert!(
                SelectedStructuralTagBuilder::for_parser(parser).is_some(),
                "'{parser}' is intrinsic per the predicate but the parser registry has \
                 no structural-tag builder for it — the predicate and the registry have \
                 drifted apart"
            );
        }
    }

    #[test]
    fn other_choices_and_parsers_still_follow_the_global_mode() {
        assert!(!requires_intrinsic_structural_tag(
            Some("kimi_k3"),
            &ToolChoice::Required
        ));
        assert!(!requires_intrinsic_structural_tag(
            Some("hermes"),
            &ToolChoice::Named("get_weather".to_string())
        ));
        assert!(!requires_intrinsic_structural_tag(
            Some("kimi_k2"),
            &ToolChoice::Auto
        ));
    }

    #[test]
    fn tools_with_structured_output_require_a_tag_independently_of_scope() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let config = StructuralTagConfig {
            scope: StructuralTagScope::Auto,
            allow_tool_calls_with_structured_output: true,
            ..Default::default()
        };

        let decision = structural_tag_decision(
            Some("qwen3_coder"),
            &ToolChoice::Auto,
            &tools,
            None,
            Some(&config),
            true,
            true,
        )
        .unwrap();

        assert!(decision.is_required());
    }

    #[test]
    fn structured_output_requires_explicit_tool_call_composition_opt_in() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        for scope in [StructuralTagScope::Auto, StructuralTagScope::Always] {
            let config = StructuralTagConfig {
                scope,
                allow_tool_calls_with_structured_output: false,
                ..Default::default()
            };
            let decision = structural_tag_decision(
                Some("qwen3_coder"),
                &ToolChoice::Auto,
                &tools,
                None,
                Some(&config),
                true,
                true,
            )
            .unwrap();

            assert!(!decision.is_required());
        }

        let config = StructuralTagConfig {
            scope: StructuralTagScope::Auto,
            allow_tool_calls_with_structured_output: true,
            ..Default::default()
        };
        let decision = structural_tag_decision(
            Some("qwen3_coder"),
            &ToolChoice::Auto,
            &tools,
            None,
            Some(&config),
            false,
            true,
        )
        .unwrap();

        assert!(!decision.is_required());
    }

    #[test]
    fn structured_output_takes_precedence_over_a_tool_call_ban() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let config = StructuralTagConfig::default();

        let decision = structural_tag_decision(
            Some("qwen3_coder"),
            &ToolChoice::None,
            &tools,
            None,
            Some(&config),
            true,
            false,
        )
        .unwrap();

        assert!(!decision.is_required());
    }

    #[test]
    fn kimi_k2_required_installs_native_tag_when_global_mode_is_off() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.tool_call_parser = Some("kimi_k2".to_string());
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            })),
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(applied);
        let format = &request
            .sampling_options
            .guided_decoding
            .as_ref()
            .unwrap()
            .structural_tag
            .as_ref()
            .unwrap()["format"];
        assert_eq!(format["type"], "sequence");
        assert_eq!(
            format["elements"][0]["value"],
            "<|tool_calls_section_begin|>"
        );
        assert_eq!(format["elements"][1]["type"], "tags_with_separator");
        assert_eq!(format["elements"][1]["at_least_one"], true);
        assert_eq!(
            format["elements"][1]["tags"][0]["begin"],
            "<|tool_call_begin|>functions.get_weather:"
        );
        assert_eq!(format["elements"][2]["value"], "<|tool_calls_section_end|>");
    }

    #[test]
    fn kimi_k3_required_stays_non_structural_when_global_mode_is_off() {
        let preprocessor = kimi_k3_preprocessor();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    // Regression for the CodeRabbit finding on PR #12576 (structural_tag.rs is the
    // real root cause; the finding was filed against http/service.rs lines 66-81):
    // `should_apply_tool_call_format` returned `true` for `Required`/`Named`
    // unconditionally, so `structural_tag_decision` built a structural tag for a
    // forced tool_choice that `get_json_schema_from_tools` would have rejected on
    // the non-structural-tag path. This must be rejected on the REAL preprocessing
    // path (`apply_tool_choice_structural_tag`), not just the HTTP compatibility
    // helper.
    #[test]
    fn kimi_k2_required_with_empty_tools_is_rejected() {
        let preprocessor = kimi_k2_preprocessor(None);
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "kimi_k2 + required with no tools must be rejected, not silently \
                 resolved to a structural tag",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn kimi_k3_required_with_empty_tools_is_rejected_before_the_mode_gate() {
        let preprocessor = kimi_k3_preprocessor();
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .expect_err("required with no tools must be rejected before Kimi K3's XTML path");
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn kimi_k2_named_tool_absent_from_tools_is_rejected() {
        let preprocessor = kimi_k2_preprocessor(None);
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Named("does_not_exist".to_string()),
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "a named tool_choice for a tool absent from `tools` must be rejected, \
                 not silently resolved to a structural tag",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn operator_enabled_qwen3_coder_required_with_empty_tools_is_rejected() {
        let preprocessor = structural_tag_preprocessor(false);
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "operator-configured structural tags must not let required-with-no-tools \
                 through for a non-Kimi registry-supported parser either",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn operator_enabled_qwen3_coder_named_tool_absent_from_tools_is_rejected() {
        let preprocessor = structural_tag_preprocessor(false);
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Named("does_not_exist".to_string()),
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "operator-enabled structural tags must reject a named tool_choice \
                 that is absent from `tools`",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    // `None`/`Auto` are not forced choices; the new validation must not affect them
    // even when `tools` is empty.
    #[test]
    fn none_and_auto_are_unaffected_by_forced_choice_validation_with_empty_tools() {
        let preprocessor = kimi_k2_preprocessor(None);

        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::None,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();
        assert!(!applied);

        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Auto,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();
        assert!(!applied);
    }

    #[test]
    fn tool_choice_none_skips_ban_only_when_prompt_excludes_tools() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        let preprocessor = structural_tag_preprocessor(true);
        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::None,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());

        let preprocessor = structural_tag_preprocessor(false);
        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::None,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(applied);
        let structural_tag = request
            .sampling_options
            .guided_decoding
            .as_ref()
            .and_then(|guided| guided.structural_tag.as_ref())
            .expect("tool-call ban should be installed");
        assert_eq!(
            structural_tag["format"]["content"]["exclude_tokens"],
            serde_json::json!(["<tool_call>"])
        );
    }
}

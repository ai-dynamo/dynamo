// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{
    StructuralTagMode, StructuralTagScope, TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::tools::{ToolChoiceValidation, validate_tool_choice_against_names};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_runtime::error::{DynamoError, ErrorType};

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

pub(super) fn requires_native_tool_call_format(
    parser_name: Option<&str>,
    tool_choice: &ToolChoice,
) -> bool {
    // The generic forced-tool JSON grammar describes a different wire format
    // from Kimi's native marker-delimited calls. This is fallback metadata, not
    // an activation bypass: the deployment-level off switch remains absolute.
    (is_kimi_k2_parser(parser_name)
        && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)))
        || (is_kimi_k3_parser(parser_name)
            && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)))
}

fn should_skip_tool_call_ban(exclude_tools_when_none: bool, tool_choice: &ToolChoice) -> bool {
    exclude_tools_when_none && matches!(tool_choice, ToolChoice::None)
}

/// Whether a request is entitled to use a structural tag, bundled with proof the
/// parser registry can actually build one.
///
/// This is the single owner of "is a structural tag applicable and available" —
/// covering the operator's global `structural_tag_mode`/`structural_tag_scope`,
/// backend transport support, the `tool_choice=None` ban-tag exclusion, and real
/// parser-registry builder availability. A caller can only
/// reach `Required` by way of a real, registered
/// [`dynamo_parsers::tool_calling::StructuralTagBuilder`] — it cannot ask for the
/// eligibility half of this decision without also getting the registry-availability
/// proof in the same value. Both the real preprocessing path
/// (`apply_tool_choice_structural_tag`) and the HTTP-layer reconstruction
/// (`http::service::apply_request_tool_call_parsing_options`) must consult this
/// function; neither may re-derive eligibility, mode, scope, or registry
/// availability independently.
pub(crate) enum StructuralTagDecision {
    Required(&'static dynamo_parsers::tool_calling::StructuralTagBuilder),
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn structural_tag_decision(
    parser_name: Option<&str>,
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    mode: StructuralTagMode,
    scope: StructuralTagScope,
    exclude_tools_when_tool_choice_none: bool,
    backend_supports_structural_tags: bool,
) -> Result<StructuralTagDecision, DynamoError> {
    // Validate before any policy or capability gate. Kimi K3 required requests use a
    // prompt-level XTML path, but they still need an actual tool to require.
    validate_forced_tool_choice(tool_choice, tools)?;

    if mode == StructuralTagMode::Off || !backend_supports_structural_tags {
        return Ok(StructuralTagDecision::NotApplicable);
    }

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

    let Some(builder) = OpenAIPreprocessor::structural_tag_builder_for_parser(parser_name) else {
        return Ok(StructuralTagDecision::NotApplicable);
    };

    if matches!(tool_choice, ToolChoice::None) {
        if tools.is_empty() {
            return Ok(StructuralTagDecision::NotApplicable);
        }
        return Ok(StructuralTagDecision::Required(builder));
    }

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
        preprocessed_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        let parser_name = self.tool_call_parser.as_deref();
        let StructuralTagDecision::Required(builder) = structural_tag_decision(
            parser_name,
            tool_choice,
            tools,
            parallel_tool_calls,
            self.runtime_config.structural_tag_mode,
            self.runtime_config.structural_tag_scope,
            self.runtime_config.exclude_tools_when_tool_choice_none,
            self.runtime_config.tool_call_structural_tag_supported(),
        )?
        else {
            return Ok(false);
        };
        // `structural_tag_decision` only returns `Required` once `parser_name` is
        // confirmed `Some` (it is the source of the registry lookup that produced
        // `builder`), so this is a real value, not a placeholder.
        let parser_name = parser_name.expect("Required decision implies a parser name");

        if matches!(tool_choice, ToolChoice::None) {
            return Self::apply_tool_call_ban(builder, preprocessed_request);
        }

        // `structural_tag_decision` already confirmed `should_apply_tool_call_format`
        // for this non-None tool_choice before returning `Required`.
        let ctx = dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
            tool_choice,
            tools,
            parallel_tool_calls,
            schema_mode: self.runtime_config.structural_tag_schema,
            starts_in_reasoning: prompt_injected_reasoning
                && !self.tool_call_structural_tag_excludes_reasoning(),
        };

        Self::apply_tool_call_format(parser_name, builder, &ctx, preprocessed_request)
    }

    fn tool_call_structural_tag_excludes_reasoning(&self) -> bool {
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
                    "Ignoring invalid structural-tag reasoning metadata; using the compatibility behavior"
                );
                false
            }
        }
    }

    /// Find the structural tag builder for a parser, if supported.
    fn structural_tag_builder_for_parser(
        parser_name: &str,
    ) -> Option<&'static dynamo_parsers::tool_calling::StructuralTagBuilder> {
        let parser_map = dynamo_parsers::tool_calling::parsers::get_tool_parser_map();
        let builder = parser_map
            .get(parser_name)
            .and_then(|tc| tc.structural_tag_builder.as_ref());

        if builder.is_none() {
            tracing::debug!(
                parser = parser_name,
                "Structural tag enabled but parser does not support it; \
                 falling back to default behaviour"
            );
        }

        builder
    }

    /// Apply the `tool_choice=none` ban tag, if configured.
    fn apply_tool_call_ban(
        builder: &dynamo_parsers::tool_calling::StructuralTagBuilder,
        common_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        if let Some(ban_tag) = builder.build_tool_call_ban().map_err(|e| {
            DynamoError::builder()
                .error_type(ErrorType::Unknown)
                .message(format!("failed to build tool-call ban structural tag: {e}"))
                .build()
        })? {
            let gd = common_request
                .sampling_options
                .guided_decoding
                .get_or_insert_default();
            gd.structural_tag = Some(ban_tag);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Build and inject the tool-call format tag, if one is needed.
    fn apply_tool_call_format(
        parser_name: &str,
        builder: &dynamo_parsers::tool_calling::StructuralTagBuilder,
        ctx: &dynamo_parsers::tool_calling::ToolCallFormatBuildContext<'_>,
        common_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        let structural_tag = match builder.build_tool_call_format(ctx) {
            Ok(Some(tag)) => tag,
            Ok(None) => {
                tracing::debug!(
                    parser = parser_name,
                    "Builder returned None for structural_tag (tool_choice={:?})",
                    ctx.tool_choice,
                );
                return Ok(false);
            }
            Err(e) => {
                tracing::warn!(
                    parser = parser_name,
                    error = %e,
                    "Failed to build structural tag; using compatibility fallback"
                );
                return Ok(false);
            }
        };

        let gd = common_request
            .sampling_options
            .guided_decoding
            .get_or_insert_default();
        gd.structural_tag = Some(structural_tag);
        Ok(true)
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

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use crate::{
        local_model::runtime_config::TOOL_CALL_STRUCTURAL_TAG_SUPPORTED_RUNTIME_KEY,
        model_card::ModelDeploymentCard,
        protocols::common::{OutputOptions, SamplingOptions, StopConditions},
    };

    use super::*;

    fn structural_tag_preprocessor(exclude_tools_when_none: bool) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::On;
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
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::On;
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
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::Off;
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
    fn kimi_forced_choices_are_classified_as_native_format_fallbacks() {
        let named = ToolChoice::Named("get_weather".to_string());
        assert!(requires_native_tool_call_format(
            Some("kimi_k2"),
            &ToolChoice::Required
        ));
        assert!(requires_native_tool_call_format(Some("kimi_k2"), &named));
        assert!(requires_native_tool_call_format(
            Some("kimi_k3"),
            &ToolChoice::Required
        ));
        assert!(requires_native_tool_call_format(Some("kimi_k3"), &named));
        assert!(requires_native_tool_call_format(Some("kimi-k3"), &named));
        assert!(!requires_native_tool_call_format(
            Some("kimi_k3"),
            &ToolChoice::Auto
        ));
        assert!(!requires_native_tool_call_format(
            Some("hermes"),
            &ToolChoice::Named("get_weather".to_string())
        ));
        assert!(!requires_native_tool_call_format(
            Some("kimi_k2"),
            &ToolChoice::Auto
        ));
    }

    #[test]
    fn global_mode_off_is_authoritative_for_kimi_k2_required() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::Off;
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
                &mut request,
            )
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());
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
            .apply_tool_choice_structural_tag(&ToolChoice::Required, &[], None, false, &mut request)
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
            .apply_tool_choice_structural_tag(&ToolChoice::Required, &[], None, false, &mut request)
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
            .apply_tool_choice_structural_tag(&ToolChoice::Required, &[], None, false, &mut request)
            .expect_err(
                "operator-enabled structural_tag_mode must not let required-with-no-tools \
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
            .apply_tool_choice_structural_tag(&ToolChoice::None, &[], None, false, &mut request)
            .unwrap();
        assert!(!applied);

        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(&ToolChoice::Auto, &[], None, false, &mut request)
            .unwrap();
        assert!(!applied);
    }

    #[test]
    fn global_mode_off_is_authoritative_for_kimi_k3_auto() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::Off;
        mdc.runtime_config.structural_tag_scope = StructuralTagScope::Auto;
        mdc.runtime_config.tool_call_parser = Some("kimi_k3".to_string());
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            })),
            // K3 auto must be constrained even when the caller does not opt in
            // to OpenAI strict schema enforcement.
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(&ToolChoice::Auto, &tools, None, false, &mut request)
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn always_scope_activates_auto_when_strict_is_omitted() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        assert!(OpenAIPreprocessor::should_apply_tool_call_format(
            StructuralTagScope::Always,
            &ToolChoice::Auto,
            &tools,
            None,
        ));
        assert!(!OpenAIPreprocessor::should_apply_tool_call_format(
            StructuralTagScope::Auto,
            &ToolChoice::Auto,
            &tools,
            None,
        ));
    }

    #[test]
    fn backend_capability_false_skips_structural_tag() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::On;
        mdc.runtime_config.structural_tag_scope = StructuralTagScope::Always;
        mdc.runtime_config.tool_call_parser = Some("qwen3_coder".to_string());
        mdc.runtime_config
            .set_engine_specific(TOOL_CALL_STRUCTURAL_TAG_SUPPORTED_RUNTIME_KEY, false)
            .unwrap();
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(&ToolChoice::Auto, &tools, None, false, &mut request)
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn malformed_backend_capability_skips_structural_tag() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::On;
        mdc.runtime_config.structural_tag_scope = StructuralTagScope::Always;
        mdc.runtime_config.tool_call_parser = Some("qwen3_coder".to_string());
        mdc.runtime_config.runtime_data.insert(
            TOOL_CALL_STRUCTURAL_TAG_SUPPORTED_RUNTIME_KEY.to_string(),
            serde_json::json!("not-a-boolean"),
        );
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(&ToolChoice::Auto, &tools, None, false, &mut request)
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());
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
            .apply_tool_choice_structural_tag(&ToolChoice::None, &tools, None, false, &mut request)
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());

        let preprocessor = structural_tag_preprocessor(false);
        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(&ToolChoice::None, &tools, None, false, &mut request)
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

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{
    StructuralTagMode, StructuralTagScope, TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_runtime::error::{DynamoError, ErrorType};

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
/// covering model-family/tool-choice intrinsic eligibility, the operator's global
/// `structural_tag_mode`/`structural_tag_scope`, the `tool_choice=None` ban-tag
/// exclusion, and real parser-registry builder availability. A caller can only
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn structural_tag_decision(
    parser_name: Option<&str>,
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    mode: StructuralTagMode,
    scope: StructuralTagScope,
    exclude_tools_when_tool_choice_none: bool,
) -> StructuralTagDecision {
    if mode == StructuralTagMode::Off
        && !requires_intrinsic_structural_tag(parser_name, tool_choice)
    {
        return StructuralTagDecision::NotApplicable;
    }

    if should_skip_tool_call_ban(exclude_tools_when_tool_choice_none, tool_choice) {
        // The prompt formatter already omits tools for this request. Avoid
        // sending a redundant AnyTokens structural tag: vLLM cannot
        // validate token-string exclusions without tokenizer metadata.
        return StructuralTagDecision::NotApplicable;
    }

    let Some(parser_name) = parser_name else {
        tracing::warn!(
            "Structural tag is enabled but --dyn-tool-call-parser is not set; \
             structural tags will not be applied"
        );
        return StructuralTagDecision::NotApplicable;
    };

    let Some(builder) = OpenAIPreprocessor::structural_tag_builder_for_parser(parser_name) else {
        return StructuralTagDecision::NotApplicable;
    };

    if matches!(tool_choice, ToolChoice::None) {
        if tools.is_empty() {
            return StructuralTagDecision::NotApplicable;
        }
        return StructuralTagDecision::Required(builder);
    }

    if !OpenAIPreprocessor::should_apply_tool_call_format(
        scope,
        tool_choice,
        tools,
        parallel_tool_calls,
    ) {
        return StructuralTagDecision::NotApplicable;
    }

    StructuralTagDecision::Required(builder)
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
        ) else {
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
            tracing::warn!(
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
                return Err(DynamoError::builder()
                    .error_type(ErrorType::Unknown)
                    .message(format!(
                        "failed to build structural_tag for parser '{parser_name}': {e}"
                    ))
                    .build());
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
                OpenAIPreprocessor::structural_tag_builder_for_parser(parser).is_some(),
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
    fn kimi_k2_required_installs_native_tag_when_global_mode_is_off() {
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
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag_mode = StructuralTagMode::Off;
        mdc.runtime_config.tool_call_parser = Some("kimi_k3".to_string());
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
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

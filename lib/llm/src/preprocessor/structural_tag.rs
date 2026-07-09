// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{
    StructuralTagMode, StructuralTagSchemaMode, StructuralTagScope,
};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_runtime::error::{DynamoError, ErrorType};

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
        let mode = self.runtime_config.structural_tag_mode;
        let reasoning_required = Self::should_use_reasoning_structural_tag(
            mode,
            tool_choice,
            preprocessed_request.require_reasoning,
        );

        if mode == StructuralTagMode::Off
            || (mode == StructuralTagMode::ReasoningRequired && !reasoning_required)
        {
            return Ok(false);
        }

        let Some(parser_name) = self.tool_call_parser.as_deref() else {
            if reasoning_required {
                return Err(Self::reasoning_structural_tag_error(
                    "the deployment does not configure --dyn-tool-call-parser",
                ));
            }
            tracing::warn!(
                "Structural tag is enabled but --dyn-tool-call-parser is not set; \
                 structural tags will not be applied"
            );
            return Ok(false);
        };

        let Some(builder) =
            Self::structural_tag_builder_for_parser(parser_name, reasoning_required)?
        else {
            return Ok(false);
        };

        if matches!(tool_choice, ToolChoice::None) {
            if tools.is_empty() {
                return Ok(false);
            }
            return Self::apply_tool_call_ban(builder, preprocessed_request);
        }

        if !Self::should_apply_tool_call_format(
            self.runtime_config.structural_tag_scope,
            tool_choice,
            tools,
            parallel_tool_calls,
        ) {
            return Ok(false);
        }

        let ctx = dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
            tool_choice,
            tools,
            parallel_tool_calls,
            schema_mode: if reasoning_required {
                StructuralTagSchemaMode::Strict
            } else {
                self.runtime_config.structural_tag_schema
            },
            starts_in_reasoning: reasoning_required || prompt_injected_reasoning,
        };

        Self::apply_tool_call_format(
            parser_name,
            builder,
            &ctx,
            reasoning_required,
            preprocessed_request,
        )
    }

    fn should_use_reasoning_structural_tag(
        mode: StructuralTagMode,
        tool_choice: &ToolChoice,
        require_reasoning: bool,
    ) -> bool {
        mode != StructuralTagMode::Off
            && require_reasoning
            && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_))
    }

    /// Find the structural tag builder for a parser, if supported.
    fn structural_tag_builder_for_parser(
        parser_name: &str,
        reasoning_required: bool,
    ) -> Result<Option<&'static dynamo_parsers::tool_calling::StructuralTagBuilder>, DynamoError>
    {
        let parser_map = dynamo_parsers::tool_calling::parsers::get_tool_parser_map();
        let builder = parser_map
            .get(parser_name)
            .and_then(|tc| tc.structural_tag_builder.as_ref());

        if builder.is_none() {
            if reasoning_required {
                return Err(Self::reasoning_structural_tag_error(format!(
                    "tool-call parser '{parser_name}' does not provide a structural-tag builder"
                )));
            }
            tracing::warn!(
                parser = parser_name,
                "Structural tag enabled but parser does not support it; \
                 falling back to default behaviour"
            );
        }

        Ok(builder)
    }

    fn reasoning_structural_tag_error(reason: impl std::fmt::Display) -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message(format!(
                "cannot preserve reasoning before the guided tool call because {reason}"
            ))
            .build()
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
        reasoning_required: bool,
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
                if reasoning_required {
                    return Err(Self::reasoning_structural_tag_error(format!(
                        "tool-call parser '{parser_name}' cannot build a reasoning-aware structural tag: {e}"
                    )));
                }
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
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_parsers::tool_calling::{StructuralTagBuilder, TriggeredTagsConfig};
    use serde_json::json;

    fn request(require_reasoning: bool) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .require_reasoning(require_reasoning)
            .build()
            .unwrap()
    }

    fn tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            })),
            strict: Some(false),
        }]
    }

    #[test]
    fn reasoning_required_mode_only_activates_for_forced_reasoning_tool_choices() {
        let named = ToolChoice::Named("get_weather".to_string());

        assert!(OpenAIPreprocessor::should_use_reasoning_structural_tag(
            StructuralTagMode::ReasoningRequired,
            &ToolChoice::Required,
            true,
        ));
        assert!(OpenAIPreprocessor::should_use_reasoning_structural_tag(
            StructuralTagMode::ReasoningRequired,
            &named,
            true,
        ));
        assert!(!OpenAIPreprocessor::should_use_reasoning_structural_tag(
            StructuralTagMode::ReasoningRequired,
            &ToolChoice::Required,
            false,
        ));
        assert!(!OpenAIPreprocessor::should_use_reasoning_structural_tag(
            StructuralTagMode::ReasoningRequired,
            &ToolChoice::Auto,
            true,
        ));
        assert!(!OpenAIPreprocessor::should_use_reasoning_structural_tag(
            StructuralTagMode::ReasoningRequired,
            &ToolChoice::None,
            true,
        ));
        assert!(!OpenAIPreprocessor::should_use_reasoning_structural_tag(
            StructuralTagMode::Off,
            &ToolChoice::Required,
            true,
        ));
        assert!(OpenAIPreprocessor::should_use_reasoning_structural_tag(
            StructuralTagMode::On,
            &ToolChoice::Required,
            true,
        ));
    }

    #[test]
    fn forced_reasoning_structural_tags_use_strict_schema_and_reasoning_prefix() {
        let tools = tools();
        let builder = OpenAIPreprocessor::structural_tag_builder_for_parser("hermes", true)
            .unwrap()
            .unwrap();
        for tool_choice in [
            ToolChoice::Required,
            ToolChoice::Named("get_weather".to_string()),
        ] {
            let ctx = dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
                tool_choice: &tool_choice,
                tools: &tools,
                parallel_tool_calls: Some(false),
                schema_mode: StructuralTagSchemaMode::Strict,
                starts_in_reasoning: true,
            };
            let mut request = request(true);

            assert!(
                OpenAIPreprocessor::apply_tool_call_format(
                    "hermes",
                    builder,
                    &ctx,
                    true,
                    &mut request,
                )
                .unwrap()
            );

            let tag = request
                .sampling_options
                .guided_decoding
                .unwrap()
                .structural_tag
                .unwrap();
            assert_eq!(tag["format"]["type"], "sequence");
            assert_eq!(tag["format"]["elements"][0]["end"], "</think>");
            assert_eq!(
                tag["format"]["elements"][1]["tags"][0]["content"]["json_schema"]["required"][0],
                "location"
            );
        }
    }

    #[test]
    fn reasoning_structural_tag_rejects_parser_without_builder() {
        let err = OpenAIPreprocessor::structural_tag_builder_for_parser("llama3_json", true)
            .expect_err("llama3_json has no structural-tag builder");

        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        let message = err.to_string();
        assert!(message.contains("llama3_json"), "{message}");
        assert!(message.contains("structural-tag builder"), "{message}");
    }

    #[test]
    fn reasoning_structural_tag_rejects_builder_without_reasoning_terminator() {
        let builder = StructuralTagBuilder::TriggeredTags(TriggeredTagsConfig {
            begin_template: "<tool_call>{}".to_string(),
            end_template: "</tool_call>".to_string(),
            triggers: vec!["<tool_call>".to_string()],
            content_style: Default::default(),
            tool_call_ban_tokens: vec![],
            reasoning_end: None,
        });
        let tools = tools();
        let ctx = dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
            tool_choice: &ToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Strict,
            starts_in_reasoning: true,
        };
        let mut request = request(true);

        let err = OpenAIPreprocessor::apply_tool_call_format(
            "test_parser",
            &builder,
            &ctx,
            true,
            &mut request,
        )
        .expect_err("reasoning-aware tags require a terminator");

        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        let message = err.to_string();
        assert!(message.contains("test_parser"), "{message}");
        assert!(message.contains("reasoning end tag"), "{message}");
    }
}

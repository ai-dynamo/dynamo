// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{StructuralTagMode, StructuralTagScope};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn is_kimi_k3_parser(parser_name: Option<&str>) -> bool {
    parser_name.is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"))
}

fn requires_intrinsic_structural_tag(parser_name: Option<&str>, tool_choice: &ToolChoice) -> bool {
    // Named K3 calls cannot use Dynamo's generic JSON-schema fallback because
    // K3 emits an XTML tools channel. Treat the K3 structural tag as part of
    // implementing this standard OpenAI request shape, not as an operator
    // opt-in. The global flag still controls optional structural enforcement
    // for auto/required and every other model family.
    is_kimi_k3_parser(parser_name) && matches!(tool_choice, ToolChoice::Named(_))
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
        if self.runtime_config.structural_tag_mode == StructuralTagMode::Off
            && !requires_intrinsic_structural_tag(parser_name, tool_choice)
        {
            return Ok(false);
        }

        let Some(parser_name) = parser_name else {
            tracing::warn!(
                "Structural tag is enabled but --dyn-tool-call-parser is not set; \
                 structural tags will not be applied"
            );
            return Ok(false);
        };

        let Some(builder) = Self::structural_tag_builder_for_parser(parser_name) else {
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
            schema_mode: self.runtime_config.structural_tag_schema,
            starts_in_reasoning: prompt_injected_reasoning,
        };

        Self::apply_tool_call_format(parser_name, builder, &ctx, preprocessed_request)
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
    use super::*;

    #[test]
    fn named_kimi_k3_is_intrinsic_even_when_global_mode_is_off() {
        let named = ToolChoice::Named("get_weather".to_string());
        assert!(requires_intrinsic_structural_tag(Some("kimi_k3"), &named));
        assert!(requires_intrinsic_structural_tag(Some("kimi-k3"), &named));
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
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{StructuralTagMode, StructuralTagScope};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::common::GuidedDecodingOptions;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption};
use dynamo_runtime::error::{DynamoError, ErrorType};

pub(super) enum StructuralTagApplyResult {
    None,
    /// Ban-only tag for `tool_choice=none`; no tool-call jail is needed.
    ToolCallBan,
    /// Tool-call format tag; the jail should parse the constrained output.
    ToolCallFormat,
}

impl OpenAIPreprocessor {
    /// Apply structural tag guided decoding when enabled for this request.
    pub(super) fn try_apply_structural_tag(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<StructuralTagApplyResult, DynamoError> {
        if self.runtime_config.structural_tag_mode == StructuralTagMode::Off {
            return Ok(StructuralTagApplyResult::None);
        }

        let Some(parser_name) = self.tool_call_parser.as_deref() else {
            tracing::warn!(
                "Structural tag is enabled but --dyn-tool-call-parser is not set; \
                 structural tags will not be applied"
            );
            return Ok(StructuralTagApplyResult::None);
        };

        let Some(builder) = Self::structural_tag_builder_for_parser(parser_name) else {
            return Ok(StructuralTagApplyResult::None);
        };

        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);

        let tools = request.inner.tools.as_deref().unwrap_or(&[]);

        Self::validate_structural_tag_tool_request(tool_choice, tools)?;

        // `tool_choice=none` uses a ban tag instead of a tool-call format tag.
        if *tool_choice == ChatCompletionToolChoiceOption::None {
            let applied_ban = if tools.is_empty() {
                false
            } else {
                Self::apply_tool_call_ban_structural_tag(builder, common_request)?
            };

            return Ok(if applied_ban {
                StructuralTagApplyResult::ToolCallBan
            } else {
                StructuralTagApplyResult::None
            });
        }

        // `auto` only activates under the configured scope policy.
        if !Self::should_activate_structural_tag(
            self.runtime_config.structural_tag_scope,
            tool_choice,
            tools,
            request.inner.parallel_tool_calls,
        ) {
            return Ok(StructuralTagApplyResult::None);
        }

        let ctx = dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
            tool_choice,
            tools,
            parallel_tool_calls: request.inner.parallel_tool_calls,
            schema_mode: self.runtime_config.structural_tag_schema,
            starts_in_reasoning: prompt_injected_reasoning,
        };

        if Self::apply_tool_call_format_structural_tag(parser_name, builder, &ctx, common_request)?
        {
            Ok(StructuralTagApplyResult::ToolCallFormat)
        } else {
            Ok(StructuralTagApplyResult::None)
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
    fn apply_tool_call_ban_structural_tag(
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
            Self::set_structural_tag_guidance(gd, ban_tag);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Build and inject the tool-call format tag, if one is needed.
    fn apply_tool_call_format_structural_tag(
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
        Self::set_structural_tag_guidance(gd, structural_tag);
        Ok(true)
    }

    fn set_structural_tag_guidance(
        gd: &mut GuidedDecodingOptions,
        structural_tag: serde_json::Value,
    ) {
        gd.json = None;
        gd.regex = None;
        gd.choice = None;
        gd.grammar = None;
        gd.whitespace_pattern = None;
        gd.structural_tag = Some(structural_tag);
    }

    /// Validate only structural-tag requests; other parser paths keep existing behavior.
    fn validate_structural_tag_tool_request(
        tool_choice: &ChatCompletionToolChoiceOption,
        tools: &[ChatCompletionTool],
    ) -> Result<(), DynamoError> {
        match tool_choice {
            ChatCompletionToolChoiceOption::Required if tools.is_empty() => {
                Err(DynamoError::builder()
                    .error_type(ErrorType::InvalidArgument)
                    .message("tool_choice is \"required\" but tools is empty")
                    .build())
            }
            ChatCompletionToolChoiceOption::Named(named) => {
                if tools.iter().any(|t| t.function.name == named.function.name) {
                    Ok(())
                } else {
                    Err(DynamoError::builder()
                        .error_type(ErrorType::InvalidArgument)
                        .message(format!(
                            "tool named \"{}\" in tool_choice is not present in tools",
                            named.function.name
                        ))
                        .build())
                }
            }
            _ => Ok(()),
        }
    }

    /// Decide whether this request should use a tool-call format tag.
    fn should_activate_structural_tag(
        scope: StructuralTagScope,
        tool_choice: &ChatCompletionToolChoiceOption,
        tools: &[ChatCompletionTool],
        parallel_tool_calls: Option<bool>,
    ) -> bool {
        match tool_choice {
            ChatCompletionToolChoiceOption::None => false,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_) => {
                true
            }
            ChatCompletionToolChoiceOption::Auto => match scope {
                StructuralTagScope::Always => true,
                StructuralTagScope::Auto => {
                    let explicit_single_call = parallel_tool_calls == Some(false);
                    tools.iter().any(|t| t.function.strict.unwrap_or(false)) || explicit_single_call
                }
            },
        }
    }
}

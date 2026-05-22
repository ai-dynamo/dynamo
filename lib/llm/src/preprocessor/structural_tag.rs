// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{StructuralTagMode, StructuralTagScope};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
}

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

        // Whether the request already carries an explicit guided-decoding constraint
        let has_user_constraint = common_request
            .sampling_options
            .guided_decoding
            .as_ref()
            .is_some_and(|gd| gd.has_user_constraint());

        match tool_choice {
            ChatCompletionToolChoiceOption::None => {
                // response_format already prevents tool-call generation, ban not needed.
                if has_user_constraint || tools.is_empty() {
                    return Ok(StructuralTagApplyResult::None);
                }

                let applied = Self::apply_tool_call_ban(builder, common_request)?;
                return Ok(if applied {
                    StructuralTagApplyResult::ToolCallBan
                } else {
                    StructuralTagApplyResult::None
                });
            }
            ChatCompletionToolChoiceOption::Auto => {
                // User-provided response_format takes precedence over optional
                // tool-call guidance; tool calls are not forced so no conflict.
                if has_user_constraint {
                    return Ok(StructuralTagApplyResult::None);
                }
            }
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_) => {
                Self::validate_tool_choice(tool_choice, tools, has_user_constraint)?;
            }
        }

        if !Self::should_apply_tool_call_format(
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

        if Self::apply_tool_call_format(parser_name, builder, &ctx, common_request)? {
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

    /// Validate forced tool-choice requests.
    fn validate_tool_choice(
        tool_choice: &ChatCompletionToolChoiceOption,
        tools: &[ChatCompletionTool],
        has_user_constraint: bool,
    ) -> Result<(), DynamoError> {
        const RESPONSE_FORMAT_CONFLICT: &str = concat!(
            "response_format cannot be used in the same request as ",
            "tool_choice=\"required\" or a named tool_choice.",
        );

        match tool_choice {
            ChatCompletionToolChoiceOption::Required if tools.is_empty() => Err(invalid_argument(
                "tool_choice is \"required\" but tools is empty",
            )),
            ChatCompletionToolChoiceOption::Required if has_user_constraint => {
                Err(invalid_argument(RESPONSE_FORMAT_CONFLICT))
            }
            ChatCompletionToolChoiceOption::Named(named) => {
                if !tools.iter().any(|t| t.function.name == named.function.name) {
                    return Err(invalid_argument(format!(
                        "tool named \"{}\" in tool_choice is not present in tools",
                        named.function.name
                    )));
                }
                if has_user_constraint {
                    return Err(invalid_argument(RESPONSE_FORMAT_CONFLICT));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Decide whether this request should use a tool-call format tag.
    fn should_apply_tool_call_format(
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

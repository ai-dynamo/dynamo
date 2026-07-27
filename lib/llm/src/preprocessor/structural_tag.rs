// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

use crate::local_model::runtime_config::{StructuralTagMode, StructuralTagScope};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_parsers_v2::{
    KimiK3StructuralTagBuilder, Tool as UnifiedTool, UnifiedStructuralTagBuilder,
    UnifiedToolCallFormatContext, UnifiedToolChoice,
};
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
        if self.runtime_config.structural_tag_mode == StructuralTagMode::Off {
            return Ok(false);
        }

        let Some(parser_name) = self.tool_call_parser.as_deref() else {
            tracing::warn!(
                "Structural tag is enabled but --dyn-tool-call-parser is not set; \
                 structural tags will not be applied"
            );
            return Ok(false);
        };

        if crate::protocols::openai::chat_completions::unified_parser::selected_family(
            Some(parser_name),
            self.runtime_config.reasoning_parser.as_deref(),
        ) == Some(dynamo_parsers_v2::KIMI_K3_FAMILY)
        {
            return self.apply_kimi_k3_structural_tag(
                tool_choice,
                tools,
                parallel_tool_calls,
                prompt_injected_reasoning,
                preprocessed_request,
            );
        }

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

    fn apply_kimi_k3_structural_tag(
        &self,
        tool_choice: &ToolChoice,
        tools: &[ToolDefinition],
        parallel_tool_calls: Option<bool>,
        prompt_injected_reasoning: bool,
        preprocessed_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        if matches!(tool_choice, ToolChoice::None) {
            return Ok(false);
        }
        if !Self::should_apply_tool_call_format(
            self.runtime_config.structural_tag_scope,
            tool_choice,
            tools,
            parallel_tool_calls,
        ) {
            return Ok(false);
        }

        let tools = tools
            .iter()
            .map(|tool| UnifiedTool {
                name: tool.name.clone(),
                description: None,
                parameters: tool.parameters.clone().unwrap_or(serde_json::Value::Null),
                strict: tool.strict,
            })
            .collect::<Vec<_>>();
        let tool_choice = match tool_choice {
            ToolChoice::None => UnifiedToolChoice::None,
            ToolChoice::Auto => UnifiedToolChoice::Auto,
            ToolChoice::Required => UnifiedToolChoice::Required,
            ToolChoice::Named(name) => UnifiedToolChoice::Named(name),
        };
        let ctx = UnifiedToolCallFormatContext {
            tool_choice,
            tools: &tools,
            parallel_tool_calls,
            strict_schema: self.runtime_config.structural_tag_schema
                == crate::local_model::runtime_config::StructuralTagSchemaMode::Strict,
            starts_in_reasoning: prompt_injected_reasoning,
        };
        let structural_tag = KimiK3StructuralTagBuilder
            .build_tool_call_format(&ctx)
            .map_err(|error| {
                DynamoError::builder()
                    .error_type(ErrorType::Unknown)
                    .message(format!(
                        "failed to build structural_tag for parser 'kimi_k3': {error}"
                    ))
                    .build()
            })?
            .ok_or_else(|| {
                DynamoError::builder()
                    .error_type(ErrorType::Unknown)
                    .message("Kimi K3 structural-tag builder returned no format")
                    .build()
            })?;

        preprocessed_request
            .sampling_options
            .guided_decoding
            .get_or_insert_default()
            .structural_tag = Some(structural_tag);
        Ok(true)
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

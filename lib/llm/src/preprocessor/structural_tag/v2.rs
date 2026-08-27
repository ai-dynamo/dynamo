// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_parsers_v2::Tool;
use dynamo_parsers_v2::structural_tag::{
    ReasoningBoundary as ReasoningBoundaryV2, StructuralTagBuilder as InnerBuilder,
    StructuralTagContext, StructuralTagOptions,
    StructuralTagSchemaMode as StructuralTagSchemaModeV2,
    StructuralTagToolChoice as StructuralTagToolChoiceV2,
};

use crate::protocols::openai::chat_completions::tool_parser_v2;

use super::StructuralTagBuildRequest;

#[derive(Clone, Copy)]
pub(crate) struct StructuralTagBuilder(&'static InnerBuilder);

impl StructuralTagBuilder {
    pub(super) fn for_parser(parser_name: &str) -> Option<Self> {
        dynamo_parsers_v2::structural_tag_builder_for_family(tool_parser_v2::canonical_family(
            parser_name,
        ))
        .map(Self)
    }

    pub(super) fn build(
        self,
        request: &StructuralTagBuildRequest<'_>,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        let tools = request.tools.iter().map(to_v2_tool).collect::<Vec<_>>();
        let tool_choice = match request.tool_choice {
            ToolChoice::None => StructuralTagToolChoiceV2::None,
            ToolChoice::Auto => StructuralTagToolChoiceV2::Auto,
            ToolChoice::Required => StructuralTagToolChoiceV2::Required,
            ToolChoice::Named(name) => StructuralTagToolChoiceV2::Named(name),
        };
        let schema_mode = match request.schema_mode {
            dynamo_parsers::tool_calling::StructuralTagSchemaMode::Auto => {
                StructuralTagSchemaModeV2::Auto
            }
            dynamo_parsers::tool_calling::StructuralTagSchemaMode::Strict => {
                StructuralTagSchemaModeV2::Strict
            }
        };
        let reasoning_boundary = match request.reasoning_boundary {
            crate::local_model::runtime_config::StructuralTagReasoningBoundary::StructuralTag => {
                ReasoningBoundaryV2::StructuralTag
            }
            crate::local_model::runtime_config::StructuralTagReasoningBoundary::Backend => {
                ReasoningBoundaryV2::External
            }
        };

        self.0.build_with_options(
            &StructuralTagContext {
                tool_choice,
                tools: &tools,
                parallel_tool_calls: request.parallel_tool_calls,
                schema_mode,
                structured_output_schema: request.structured_output_schema,
                starts_in_reasoning: request.starts_in_reasoning,
            },
            &StructuralTagOptions {
                exclude_special_tokens: request.exclude_special_tokens,
                reasoning_boundary,
                tool_arguments_any_order: request.tool_arguments_any_order,
            },
        )
    }
}

fn to_v2_tool(tool: &ToolDefinition) -> Tool {
    Tool {
        name: tool.name.clone(),
        description: None,
        parameters: tool.parameters.clone().unwrap_or(serde_json::Value::Null),
        strict: tool.strict,
    }
}

pub(super) fn enabled() -> bool {
    tool_parser_v2::enabled()
}

pub(super) fn supports_family(parser_name: &str) -> bool {
    StructuralTagBuilder::for_parser(parser_name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_v4_engine_aliases_select_the_structural_tag_builder() {
        for alias in ["deepseek_v4", "deepseek-v4", "deepseekv4"] {
            assert!(supports_family(alias), "unsupported alias: {alias}");
            assert!(StructuralTagBuilder::for_parser(alias).is_some());
        }
    }
}

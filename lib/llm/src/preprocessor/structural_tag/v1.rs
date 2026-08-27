// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::{
    StructuralTagBuilder as InnerBuilder, ToolCallFormatBuildContext,
};

use super::{StructuralTagBuildRequest, StructuralTagReasoningBoundary};

#[derive(Clone, Copy)]
pub(crate) struct StructuralTagBuilder(&'static InnerBuilder);

impl StructuralTagBuilder {
    pub(super) fn for_parser(parser_name: &str) -> Option<Self> {
        dynamo_parsers::tool_calling::parsers::get_tool_parser_map()
            .get(parser_name)
            .and_then(|config| config.structural_tag_builder.as_ref())
            .map(Self)
    }

    pub(super) fn build(
        self,
        request: &StructuralTagBuildRequest<'_>,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        anyhow::ensure!(
            request.structured_output_schema.is_none(),
            "tool calls with structured output require parsers v2"
        );

        if matches!(
            request.tool_choice,
            dynamo_parsers::tool_calling::ToolChoice::None
        ) {
            return self.0.build_tool_call_ban();
        }

        self.0.build_tool_call_format(&ToolCallFormatBuildContext {
            tool_choice: request.tool_choice,
            tools: request.tools,
            parallel_tool_calls: request.parallel_tool_calls,
            schema_mode: request.schema_mode,
            starts_in_reasoning: request.starts_in_reasoning
                && request.reasoning_boundary == StructuralTagReasoningBoundary::StructuralTag,
        })
    }
}

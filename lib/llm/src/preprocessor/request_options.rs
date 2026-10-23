// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    preprocessor::OpenAIPreprocessor,
    protocols::openai::{ParsingOptions, chat_completions::NvCreateChatCompletionRequest},
};

use dynamo_protocols::types::ChatCompletionToolChoiceOption;
use dynamo_runtime::error::DynamoError;

/// Apply the request-level tool-call gates shared by the HTTP protocol handlers.
pub(crate) fn apply_request_tool_call_parsing_options(
    mut parsing_options: ParsingOptions,
    request: &NvCreateChatCompletionRequest,
) -> Result<ParsingOptions, DynamoError> {
    let unified_family =
        crate::protocols::openai::chat_completions::unified_parser::selected_batch_family(
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
        );
    parsing_options.reasoning_disabled = OpenAIPreprocessor::request_disables_reasoning_with_family(
        request,
        parsing_options.tool_call_parser.as_deref(),
        parsing_options.reasoning_parser.as_deref(),
        parsing_options.default_thinking_mode.as_deref(),
        unified_family,
    );
    parsing_options.structured_response =
        OpenAIPreprocessor::has_structured_response_format(request);
    parsing_options.tool_choice = request.inner.tool_choice.clone();
    let tool_call_parsing_enabled = OpenAIPreprocessor::tool_call_parsing_enabled(request);
    let tool_choice = request
        .inner
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    let converted_tool_choice = crate::preprocessor::tool_choice::convert_tool_choice(tool_choice);
    let effective_tools = crate::preprocessor::tool_choice::effective_tools(&request.inner)?;
    let converted_tools = crate::preprocessor::tool_choice::convert_tools(effective_tools.as_ref());
    let structural_tag_config = parsing_options.structural_tag.as_ref();
    let has_structured_output = request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, dynamo_protocols::types::ResponseFormat::Text));
    let uses_structural_tag = crate::preprocessor::structural_tag::structural_tag_decision(
        (
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
        ),
        &converted_tool_choice,
        &converted_tools,
        request.inner.parallel_tool_calls,
        structural_tag_config,
        has_structured_output,
        parsing_options.exclude_tools_when_tool_choice_none,
    )?
    .is_required();
    let guided_tool_constraint =
        crate::preprocessor::tool_choice::guided_tool_constraint_with_effective_tools(
            request,
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
            uses_structural_tag,
            effective_tools.as_ref(),
        )?;
    Ok(parsing_options
        .with_guided_tool_constraint(guided_tool_constraint)
        .with_tool_call_parsing_enabled(tool_call_parsing_enabled)
        .with_tools(converted_tools))
}

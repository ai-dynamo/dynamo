// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet, hash_map::Entry};

use anyhow::{Context, ensure};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCallChunk, ChatCompletionToolChoiceOption, ChatCompletionToolType,
    FunctionType,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use serde::Deserialize;

use super::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    NvCreateChatCompletionStreamResponse,
};

/// The HTTP adapter consumes the new choice object before the function-oriented
/// request type sees it. Existing requests use their usual typed deserializer.
#[derive(Deserialize)]
pub(crate) struct AllowedToolsRequest {
    tool_choice: Choice,
    #[serde(flatten)]
    request: NvCreateChatCompletionRequest,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Choice {
    AllowedTools { allowed_tools: Selection },
}

#[derive(Deserialize)]
struct Selection {
    mode: Mode,
    tools: Vec<Tool>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Auto,
    Required,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Tool {
    Function {
        function: dynamo_protocols::types::FunctionName,
    },
}

pub(crate) struct ToolAllowlist(HashSet<String>);

enum CallState {
    Unnamed(ChatCompletionMessageToolCallChunk),
    Named(String),
}

impl AllowedToolsRequest {
    pub(crate) fn normalize(
        self,
    ) -> anyhow::Result<(NvCreateChatCompletionRequest, ToolAllowlist)> {
        let Self {
            tool_choice: Choice::AllowedTools { allowed_tools },
            mut request,
        } = self;
        let Selection { mode, tools } = allowed_tools;
        ensure!(
            !tools.is_empty(),
            "allowed_tools must contain at least one function"
        );
        let catalog = request
            .inner
            .tools
            .take()
            .context("allowed_tools requires a declared tools catalog")?;
        ensure!(
            catalog
                .iter()
                .all(|tool| tool.r#type == ChatCompletionToolType::Function)
                && request.inner.dynamic_system_tools().next().is_none(),
            "allowed_tools supports only top-level function tools"
        );
        super::super::validate::validate_tools(&Some(catalog.as_slice()))?;
        ensure!(
            tools.len() <= catalog.len(),
            "allowed_tools must be a subset of the declared catalog"
        );
        let declared: HashSet<_> = catalog
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        ensure!(
            declared.len() == catalog.len(),
            "allowed_tools requires unique names in the declared catalog"
        );
        let mut names = HashSet::with_capacity(tools.len());
        for Tool::Function { function } in tools {
            ensure!(
                declared.contains(function.name.as_str()),
                "allowed_tools references unknown function '{}'",
                function.name
            );
            ensure!(
                names.insert(function.name),
                "allowed_tools contains duplicate functions"
            );
        }
        request.inner.tools = Some(
            catalog
                .iter()
                .filter(|tool| names.contains(&tool.function.name))
                .cloned()
                .collect(),
        );
        request.chat_template_tools = Some(catalog.into());
        request.inner.tool_choice = Some(match mode {
            Mode::Auto => ChatCompletionToolChoiceOption::Auto,
            Mode::Required => ChatCompletionToolChoiceOption::Required,
        });
        Ok((request, ToolAllowlist(names)))
    }
}

impl ToolAllowlist {
    #[allow(deprecated)]
    pub(crate) fn validate_response(
        &self,
        response: &NvCreateChatCompletionResponse,
    ) -> anyhow::Result<()> {
        for choice in &response.inner.choices {
            ensure!(
                choice.message.function_call.is_none(),
                "Backend returned a legacy function_call for allowed_tools"
            );
            for call in choice.message.tool_calls.iter().flatten() {
                self.check_name(
                    &call.function.name,
                    call.r#type == FunctionType::Function && call.custom.is_none(),
                )?;
            }
        }
        Ok(())
    }

    fn check_name(&self, name: &str, is_function: bool) -> anyhow::Result<()> {
        ensure!(
            is_function && self.0.contains(name),
            "Backend returned tool '{name}' outside allowed_tools"
        );
        Ok(())
    }

    pub(crate) fn stream(
        self,
        source: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send,
        single_tool: bool,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send {
        async_stream::stream! {
            let mut calls = HashMap::new();
            futures::pin_mut!(source);
            while let Some(mut response) = source.next().await {
                if response.is_error() {
                    yield response;
                    return;
                }
                if let Some(data) = &mut response.data
                    && let Err(error) = self.validate_chunk(data, &mut calls, single_tool)
                {
                    yield Annotated::from_error(error.to_string());
                    return;
                }
                yield response;
            }
            if calls.values().any(|call| matches!(call, CallState::Unnamed(_))) {
                yield Annotated::from_error("Backend ended an allowed_tools stream with an unnamed tool call");
            }
        }
    }

    fn validate_chunk(
        &self,
        response: &mut NvCreateChatCompletionStreamResponse,
        calls: &mut HashMap<(u32, u32), CallState>,
        single_tool: bool,
    ) -> anyhow::Result<()> {
        for choice in &mut response.inner.choices {
            ensure!(
                choice.delta.function_call.is_none(),
                "Backend returned a legacy function_call for allowed_tools"
            );
            if let Some(tool_calls) = choice.delta.tool_calls.as_mut() {
                let mut result = Ok(());
                tool_calls.retain_mut(|call| {
                    if result.is_err() || (single_tool && call.index > 0) {
                        return false;
                    }
                    match self.validate_fragment(call, calls.entry((choice.index, call.index))) {
                        Ok(ready) => ready,
                        Err(error) => {
                            result = Err(error);
                            false
                        }
                    }
                });
                result?;
                if tool_calls.is_empty() {
                    choice.delta.tool_calls = None;
                }
            }
            if choice.finish_reason.is_some() {
                ensure!(
                    !calls.iter().any(|((index, _), call)| {
                        *index == choice.index && matches!(call, CallState::Unnamed(_))
                    }),
                    "Backend finished an allowed_tools choice with an unnamed tool call"
                );
            }
        }
        Ok(())
    }

    fn validate_fragment(
        &self,
        call: &mut ChatCompletionMessageToolCallChunk,
        state: Entry<'_, (u32, u32), CallState>,
    ) -> anyhow::Result<bool> {
        ensure!(
            call.custom.is_none() && call.r#type != Some(FunctionType::Custom),
            "Backend returned a custom tool outside allowed_tools"
        );
        let name = call
            .function
            .as_mut()
            .and_then(|function| function.name.as_mut())
            .filter(|name| !name.is_empty());
        match state {
            Entry::Vacant(entry) => {
                if let Some(name) = name {
                    self.check_name(name, true)?;
                    entry.insert(CallState::Named(name.clone()));
                } else {
                    entry.insert(CallState::Unnamed(std::mem::take(call)));
                    return Ok(false);
                }
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                CallState::Named(approved) => {
                    if let Some(name) = name {
                        ensure!(
                            name == approved,
                            "Backend changed the name of an allowed_tools call"
                        );
                        if let Some(function) = &mut call.function {
                            function.name = None;
                        }
                    }
                }
                CallState::Unnamed(pending) => {
                    if let Some(name) = name {
                        self.check_name(name, true)?;
                        let approved = name.clone();
                        merge_fragment(pending, std::mem::take(call));
                        *call = std::mem::take(pending);
                        entry.insert(CallState::Named(approved));
                    } else {
                        merge_fragment(pending, std::mem::take(call));
                        return Ok(false);
                    }
                }
            },
        }
        Ok(true)
    }
}

// A parser may deliver arguments before identity. Once the name is validated,
// emit one first chunk with the original call index, identity and arguments.
fn merge_fragment(
    pending: &mut ChatCompletionMessageToolCallChunk,
    fragment: ChatCompletionMessageToolCallChunk,
) {
    if fragment.id.as_ref().is_some_and(|id| !id.is_empty()) {
        pending.id = fragment.id;
    }
    if fragment.r#type.is_some() {
        pending.r#type = fragment.r#type;
    }
    if let Some(function) = fragment.function {
        let target = pending.function.get_or_insert_with(Default::default);
        if function.name.as_ref().is_some_and(|name| !name.is_empty()) {
            target.name = function.name;
        }
        if let Some(arguments) = function.arguments {
            target
                .arguments
                .get_or_insert_with(String::new)
                .push_str(&arguments);
        }
    }
}

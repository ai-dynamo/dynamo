// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped filtering after GLM parsing, shared by streaming and aggregation.

use std::collections::{BTreeMap, HashSet};
use std::pin::Pin;

use crate::protocols::{Annotated, openai::chat_completions::NvCreateChatCompletionStreamResponse};
use dynamo_protocols::types::{ChatCompletionMessageToolCallChunk, FinishReason};
use futures::{Stream, StreamExt};

#[derive(Default)]
enum Decision {
    #[default]
    Pending,
    Rejected,
    Accepted(u32),
}

#[derive(Default)]
struct Call {
    pending: ChatCompletionMessageToolCallChunk,
    decision: Decision,
}

impl Call {
    fn decide(&mut self, names: &HashSet<String>, next_index: &mut u32) {
        let name = self
            .pending
            .function
            .as_ref()
            .and_then(|function| function.name.as_ref());
        self.decision = if name.is_some_and(|name| names.contains(name)) {
            let index = *next_index;
            *next_index += 1;
            Decision::Accepted(index)
        } else {
            self.pending = ChatCompletionMessageToolCallChunk::default();
            Decision::Rejected
        };
    }

    fn release(&mut self) -> Option<ChatCompletionMessageToolCallChunk> {
        let Decision::Accepted(index) = self.decision else {
            return None;
        };
        let mut chunk = std::mem::take(&mut self.pending);
        chunk.index = index;
        Some(chunk)
    }

    fn push(
        &mut self,
        mut chunk: ChatCompletionMessageToolCallChunk,
        names: &HashSet<String>,
        next_index: &mut u32,
    ) -> Option<ChatCompletionMessageToolCallChunk> {
        match self.decision {
            Decision::Accepted(index) => {
                chunk.index = index;
                return Some(chunk);
            }
            Decision::Rejected => return None,
            Decision::Pending => {}
        }
        if chunk.id.is_some() {
            self.pending.id = chunk.id;
        }
        if chunk.r#type.is_some() {
            self.pending.r#type = chunk.r#type;
        }
        if let Some(function) = chunk.function {
            let pending = self.pending.function.get_or_insert_with(Default::default);
            if let Some(name) = function.name {
                pending.name.get_or_insert_with(String::new).push_str(&name);
            }
            if let Some(arguments) = function.arguments {
                pending
                    .arguments
                    .get_or_insert_with(String::new)
                    .push_str(&arguments);
            }
        }
        // Names may arrive in fragments. No call metadata escapes before its
        // arguments begin (or its terminal arrives), when the name is complete.
        if self
            .pending
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_ref())
            .is_some_and(|arguments| !arguments.is_empty())
        {
            self.decide(names, next_index);
            return self.release();
        }
        None
    }
}

pub(super) fn apply_stream<S>(
    stream: S,
    names: Option<HashSet<String>>,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let Some(names) = names else {
        return Box::pin(stream);
    };
    Box::pin(async_stream::stream! {
        let mut stream = Box::pin(stream);
        let mut calls: BTreeMap<(u32, u32), Call> = BTreeMap::new();
        let mut next_indices: BTreeMap<u32, u32> = BTreeMap::new();
        while let Some(mut response) = stream.next().await {
            if let Some(data) = response.data.as_mut() {
                for choice in &mut data.inner.choices {
                    let next_index = next_indices.entry(choice.index).or_default();
                    let mut output = Vec::new();
                    for chunk in choice.delta.tool_calls.take().unwrap_or_default() {
                        let call = calls.entry((choice.index, chunk.index)).or_default();
                        if let Some(chunk) = call.push(chunk, &names, next_index) {
                            output.push(chunk);
                        }
                    }
                    if choice.finish_reason.is_some() {
                        for (_, call) in calls.range_mut((choice.index, 0)..=(choice.index, u32::MAX)) {
                            if matches!(call.decision, Decision::Pending) {
                                call.decide(&names, next_index);
                                if let Some(chunk) = call.release() {
                                    output.push(chunk);
                                }
                            }
                        }
                        if *next_index == 0 && choice.finish_reason == Some(FinishReason::ToolCalls) {
                            choice.finish_reason = Some(FinishReason::Stop);
                        }
                    }
                    if !output.is_empty() {
                        choice.delta.tool_calls = Some(output);
                    }
                }
            }
            yield response;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use serde_json::{Value, json};

    fn response(calls: Value, finish: Value) -> Annotated<NvCreateChatCompletionStreamResponse> {
        Annotated::from_data(serde_json::from_value(json!({
            "id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
            "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": calls}, "finish_reason": finish}]
        })).unwrap())
    }

    #[tokio::test]
    async fn allow_undefined_tools_handles_parameterless_calls_and_empty_turns() {
        for name in ["known", "unknown"] {
            let chunks = vec![
                response(
                    json!([{"index": 0, "id": "call", "type": "function", "function": {"name": name}}]),
                    json!(null),
                ),
                response(json!(null), json!("tool_calls")),
            ];
            let output: Vec<_> = apply_stream(
                stream::iter(chunks.clone()),
                Some(HashSet::from(["known".into()])),
            )
            .collect()
            .await;
            let terminal = &output.last().unwrap().data.as_ref().unwrap().inner.choices[0];
            assert_eq!(terminal.delta.tool_calls.is_some(), name == "known");
            assert_eq!(
                terminal.finish_reason,
                Some(if name == "known" {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                })
            );
            let unfiltered: Vec<_> = apply_stream(stream::iter(chunks), None).collect().await;
            assert!(
                unfiltered[0].data.as_ref().unwrap().inner.choices[0]
                    .delta
                    .tool_calls
                    .is_some()
            );
        }
    }
    #[tokio::test]
    async fn allow_undefined_tools_batch_filters_after_schema_coercion() {
        use crate::protocols::openai::{ParsingOptions, chat_completions::DeltaAggregator};
        let markup = "<tool_call>unknown<arg_key>x</arg_key><arg_value>1</arg_value></tool_call><tool_call>known<arg_key>x</arg_key><arg_value>2</arg_value></tool_call>";
        for enabled in [false, true] {
            let chunk = Annotated::from_data(serde_json::from_value::<NvCreateChatCompletionStreamResponse>(json!({
                "id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": markup}, "finish_reason": "stop"}]
            })).unwrap());
            let options = ParsingOptions {
                tool_call_parser: Some("glm47".into()),
                allow_undefined_tools: enabled,
                tools: vec![dynamo_parsers::tool_calling::ToolDefinition {
                    name: "known".into(),
                    strict: None,
                    parameters: Some(
                        json!({"type": "object", "properties": {"x": {"type": "integer"}}}),
                    ),
                }],
                ..Default::default()
            };
            let result = DeltaAggregator::apply(stream::iter([chunk]), options)
                .await
                .unwrap();
            let calls = result.inner.choices[0].message.tool_calls.as_ref().unwrap();
            assert_eq!(calls.len(), if enabled { 2 } else { 1 });
            let known = calls
                .iter()
                .find(|call| call.function.name == "known")
                .unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&known.function.arguments).unwrap(),
                json!({"x": 2})
            );
            if enabled {
                assert_eq!(calls[0].function.name, "unknown");
                assert_eq!(
                    serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
                    json!({"x": "1"})
                );
            }
        }
    }
    async fn check_parsed_stream_policy() {
        use crate::model_card::ModelDeploymentCard;
        use crate::preprocessor::OpenAIPreprocessor;
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
        let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut card = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        card.runtime_config.tool_call_parser = Some("glm47".into());
        card.runtime_config.reasoning_parser = None;
        card.runtime_config.structural_tag = None;
        let processor = OpenAIPreprocessor::new(card).unwrap();
        let markup = "<tool_call>unknown<arg_key>x</arg_key><arg_value>1</arg_value></tool_call><tool_call>known<arg_key>x</arg_key><arg_value>2</arg_value></tool_call>";
        for enabled in [false, true] {
            let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "test", "messages": [{"role": "user", "content": "test"}],
                "tools": [{"type": "function", "function": {"name": "known", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}],
                "chat_template_args": {"allow_undefined_tools": enabled}
            })).unwrap();
            let mut chunks: Vec<_> = markup.chars().map(|character| Annotated::from_data(serde_json::from_value::<NvCreateChatCompletionStreamResponse>(json!({
                "id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": character.to_string()}, "finish_reason": null}]
            })).unwrap())).collect();
            chunks.push(response(json!(null), json!("stop")));
            let parsed = processor
                .postprocessor_parsing_stream(stream::iter(chunks), &request, false, false)
                .unwrap();
            let output: Vec<_> = parsed.collect().await;
            let mut calls: BTreeMap<u32, (String, String)> = BTreeMap::new();
            for response in &output {
                for choice in &response.data.as_ref().unwrap().inner.choices {
                    for call in choice.delta.tool_calls.iter().flatten() {
                        let entry = calls.entry(call.index).or_default();
                        if let Some(function) = &call.function {
                            if let Some(name) = &function.name {
                                entry.0.push_str(name);
                            }
                            if let Some(arguments) = &function.arguments {
                                entry.1.push_str(arguments);
                            }
                        }
                    }
                }
            }
            assert_eq!(calls.len(), if enabled { 2 } else { 1 });
            assert_eq!(
                calls.keys().copied().collect::<Vec<_>>(),
                if enabled { vec![0, 1] } else { vec![0] }
            );
            let known = calls.values().find(|(name, _)| name == "known").unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&known.1).unwrap(),
                json!({"x": 2})
            );
            if enabled {
                assert_eq!(calls[&0].0, "unknown");
            }
        }
    }

    #[tokio::test]
    async fn allow_undefined_tools_parsed_stream_v1() {
        if crate::test_utils::run_isolated(
            concat!(module_path!(), "::allow_undefined_tools_parsed_stream_v1"),
            &[(
                dynamo_runtime::config::environment_names::llm::DYN_PARSER_VERSION,
                "1",
            )],
        ) {
            return;
        }
        check_parsed_stream_policy().await;
    }

    #[tokio::test]
    async fn allow_undefined_tools_parsed_stream_v2() {
        if crate::test_utils::run_isolated(
            concat!(module_path!(), "::allow_undefined_tools_parsed_stream_v2"),
            &[(
                dynamo_runtime::config::environment_names::llm::DYN_PARSER_VERSION,
                "2",
            )],
        ) {
            return;
        }
        check_parsed_stream_policy().await;
    }
}

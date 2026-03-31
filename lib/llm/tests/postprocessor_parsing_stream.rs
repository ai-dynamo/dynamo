// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dynamo_async_openai::types::{
    ChatCompletionMessageContent, ChatCompletionToolChoiceOption, FinishReason,
};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use serde_json::Value;

const REQUEST_JSON: &str = r#"{"messages":[{"role":"user","content":"What is the capital of Tuvalu?"}],"model":"Qwen/Qwen3-0.6B","max_completion_tokens":3000,"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":false},"temperature":1.0,"top_p":1.0}"#;

fn build_preprocessor(
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
) -> Arc<OpenAIPreprocessor> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    mdc.runtime_config.reasoning_parser = reasoning_parser.map(ToString::to_string);
    mdc.runtime_config.tool_call_parser = tool_call_parser.map(ToString::to_string);
    OpenAIPreprocessor::new(mdc).unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/replays")
        .join(name)
}

fn parse_fixture(
    jsonl_path: &Path,
) -> (
    NvCreateChatCompletionRequest,
    Vec<Value>,
    Vec<NvCreateChatCompletionStreamResponse>,
) {
    let content = fs::read_to_string(jsonl_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", jsonl_path.display()));

    let mut expected_stream_json = Vec::new();
    let mut input_chunks = Vec::new();

    for line in content.lines().filter(|l| !l.is_empty()) {
        let value: Value = serde_json::from_str(line).unwrap();
        let chunk: NvCreateChatCompletionStreamResponse =
            serde_json::from_value(value.clone()).unwrap();
        expected_stream_json.push(value);
        input_chunks.push(chunk);
    }

    let request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    assert!(
        !input_chunks.is_empty(),
        "missing stream chunks in fixture {}",
        jsonl_path.display()
    );

    (request, expected_stream_json, input_chunks)
}

fn get_text(content: &ChatCompletionMessageContent) -> &str {
    match content {
        ChatCompletionMessageContent::Text(text) => text.as_str(),
        ChatCompletionMessageContent::Parts(_) => "",
    }
}

/// Accumulates streamed tool call deltas into complete tool calls for assertion.
#[derive(Default, Clone)]
struct MergedToolCall {
    id: Option<String>,
    r#type: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl MergedToolCall {
    fn merge_from(
        &mut self,
        tool_call: &dynamo_async_openai::types::ChatCompletionMessageToolCallChunk,
    ) {
        if self.id.is_none() {
            self.id = tool_call.id.clone();
        }
        if self.r#type.is_none() {
            self.r#type = tool_call.r#type.as_ref().map(|t| {
                serde_json::to_string(t)
                    .unwrap()
                    .trim_matches('"')
                    .to_string()
            });
        }
        if let Some(function) = &tool_call.function {
            if self.name.is_none() {
                self.name = function.name.clone();
            }
            if let Some(arguments) = &function.arguments {
                self.arguments.push_str(arguments);
            }
        }
    }
}

#[tokio::test]
async fn postprocessor_parsing_stream_replays_unit_test_fixture() {
    let preprocessor = build_preprocessor(None, None);
    let (request, expected_stream_json, input_chunks) =
        parse_fixture(&fixture_path("stream_interval_1.jsonl"));

    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    assert_eq!(output_chunks.len(), expected_stream_json.len());

    for (idx, (output, expected)) in output_chunks
        .iter()
        .zip(expected_stream_json.iter())
        .enumerate()
    {
        let output_data = output
            .data
            .as_ref()
            .expect("output stream chunk should include data");
        let output_json = serde_json::to_value(output_data).unwrap();
        assert_eq!(output_json, *expected, "chunk {idx} did not match fixture");
    }
}

#[tokio::test]
async fn postprocessor_parsing_stream_replays_interval_20_fixture() {
    let preprocessor = build_preprocessor(Some("qwen"), Some("hermes"));
    let (mut request, _expected_stream_json, input_chunks) =
        parse_fixture(&fixture_path("stream_interval_20.jsonl"));

    // Mirror tests/frontend/test_prepost.py::request_for_sampling
    let tools: Vec<dynamo_async_openai::types::ChatCompletionTool> =
        serde_json::from_value(serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "search_gutenberg_books",
                    "description": "Search for books in the Project Gutenberg library",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search terms to find books"
                            }
                        },
                        "required": ["search_terms"]
                    }
                }
            }
        ]))
        .unwrap();
    request.inner.tools = Some(tools);
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Auto);

    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut all_content = String::new();
    let mut finish_reasons = Vec::new();
    let mut merged_tool_calls: BTreeMap<u32, MergedToolCall> = BTreeMap::new();

    for output in &output_chunks {
        let Some(output_data) = output.data.as_ref() else {
            continue;
        };

        for choice in &output_data.inner.choices {
            if let Some(reasoning_content) = &choice.delta.reasoning_content {
                reasoning.push_str(reasoning_content);
            }

            if let Some(content) = &choice.delta.content {
                all_content.push_str(get_text(content));
            }

            if let Some(reason) = choice.finish_reason {
                finish_reasons.push(reason);
            }

            if let Some(tool_calls) = &choice.delta.tool_calls {
                for tool_call in tool_calls {
                    merged_tool_calls
                        .entry(tool_call.index)
                        .or_default()
                        .merge_from(tool_call);
                }
            }
        }
    }

    let tool_calls: Vec<MergedToolCall> = merged_tool_calls.values().cloned().collect();

    // Port of tests/frontend/test_prepost.py::test_stream_interval_20
    assert!(
        reasoning.contains("the user is asking for the titles of some James Joyce books"),
        "reasoning did not contain expected phrase: {reasoning}"
    );
    assert!(
        reasoning.contains("the user's request.\n"),
        "reasoning did not contain expected ending: {reasoning}"
    );

    assert_eq!(
        tool_calls.len(),
        1,
        "Expected 1 tool call but got {}. Tool-call markup was likely emitted as plain content instead.",
        tool_calls.len()
    );
    let tc = &tool_calls[0];
    assert_eq!(tc.name.as_deref(), Some("search_gutenberg_books"));
    let arguments_json: Value = serde_json::from_str(&tc.arguments).unwrap();
    assert_eq!(
        arguments_json,
        serde_json::json!({
            "search_terms": ["James Joyce", "Project Gutenberg"]
        })
    );
    assert!(
        tc.id
            .as_ref()
            .is_some_and(|id| id.starts_with("call-") || id.starts_with("chatcmpl-tool-")),
        "tool call id did not match expected prefix: {:?}",
        tc.id
    );
    assert_eq!(tc.r#type.as_deref(), Some("function"));

    assert!(
        !all_content.contains("<tool_call>"),
        "Raw <tool_call> markup leaked into content: {all_content:?}"
    );
    assert!(!all_content.contains("</tool_call>"));

    if !finish_reasons.is_empty() {
        assert!(
            finish_reasons.contains(&FinishReason::Stop)
                || finish_reasons.contains(&FinishReason::ToolCalls),
            "expected terminal finish reason (stop/tool_calls), got: {:?}",
            finish_reasons
        );
    }
}

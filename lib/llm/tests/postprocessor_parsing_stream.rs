// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionNamedToolChoice, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionToolType, FinishReason, FunctionName,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use serde_json::Value;

const REQUEST_JSON: &str = r#"{"messages":[{"role":"user","content":"What is the capital of Tuvalu?"}],"model":"Qwen/Qwen3-0.6B","max_completion_tokens":3000,"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":false},"temperature":1.0,"top_p":1.0}"#;

const FORCE_REASONING_PARSERS: &[&str] = &[
    "deepseek_r1",
    "step3",
    "kimi_k25",
    "mistral",
    "minimax_append_think",
    "nemotron_nano",
    "nemotron3",
    "nemotron_v3",
];

const REASONING_BEFORE_GUIDED_JSON_PARSERS: &[(&str, &str)] = &[
    ("deepseek_r1", "</think>"),
    ("step3", "</think>"),
    ("kimi_k25", "</think>"),
    ("mistral", "[/THINK]"),
    ("nemotron_nano", "</think>"),
    ("nemotron3", "</think>"),
    ("nemotron_v3", "</think>"),
];

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
        // Round-trip through the typed struct so expected JSON matches current serialization
        // (upstream async-openai skips None fields that the old fork serialized as null).
        let normalized = serde_json::to_value(&chunk).unwrap();
        expected_stream_json.push(normalized);
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
#[derive(Default, Clone, Debug)]
struct MergedToolCall {
    id: Option<String>,
    r#type: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl MergedToolCall {
    fn merge_from(
        &mut self,
        tool_call: &dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
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
        .postprocessor_parsing_stream(input_stream, &request, false, false)
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
    let tools: Vec<dynamo_protocols::types::ChatCompletionTool> =
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
        .postprocessor_parsing_stream(input_stream, &request, false, false)
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

/// Construct a minimal stream chunk carrying `content` as a text delta.
fn mock_content_chunk(content: &str) -> NvCreateChatCompletionStreamResponse {
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
        Role,
    };
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    }
}

/// Construct a stream chunk carrying one text delta per choice.
fn mock_multi_choice_content_chunk(
    choices: &[(u32, &str)],
) -> NvCreateChatCompletionStreamResponse {
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
        Role,
    };

    #[allow(deprecated)]
    let choices = choices
        .iter()
        .map(|(index, content)| ChatChoiceStream {
            index: *index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(ChatCompletionMessageContent::Text((*content).to_string())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: None,
            logprobs: None,
        })
        .collect();

    NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices,
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    }
}

/// Construct a chunk that carries only `reasoning_content` (no text delta).
/// Mirrors what upstream `parse_reasoning_content_from_stream` emits while the
/// model is still inside `<think>...</think>`; exercises the jail's
/// `Immediate` mode initialization when the first chunk for a choice has
/// `delta.content=None`.
fn mock_reasoning_only_chunk(reasoning: &str) -> NvCreateChatCompletionStreamResponse {
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
        Role,
    };
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: Some(reasoning.to_string()),
        },
        finish_reason: None,
        logprobs: None,
    };
    NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    }
}

/// Construct a terminal `finish_reason=Stop` chunk with no content.
fn mock_final_chunk() -> NvCreateChatCompletionStreamResponse {
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
    };
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: None,
            content: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: Some(FinishReason::Stop),
        logprobs: None,
    };
    NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    }
}

/// Regression for DeepSeek V4 tool-continuation turns.
///
/// The V4 formatter seeds `<think>` into the prompt after a merged tool result,
/// so the completion starts inside a reasoning block and does not emit an
/// opening `<think>`. `postprocessor_parsing_stream` must preserve the
/// prompt-injected reasoning signal even when the original request's last
/// message is `role=tool`.
#[tokio::test]
async fn postprocessor_parsing_stream_deepseek_v4_tool_continuation_keeps_injected_reasoning() {
    let preprocessor = build_preprocessor(Some("deepseek_v4"), None);
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "messages": [
            {"role": "user", "content": "Create and run a hello-world script."},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_python",
                        "arguments": "{\"path\":\"/tmp/hello.py\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Hello, world!"
            }
        ],
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "stream": true
    }))
    .unwrap();

    let input_chunks = vec![
        mock_content_chunk("The script ran successfully."),
        mock_content_chunk("</think>"),
        mock_content_chunk("Done. Output: `Hello, world!`"),
        mock_final_chunk(),
    ];

    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(reasoning, "The script ran successfully.");
    assert_eq!(content, "Done. Output: `Hello, world!`");
    assert!(
        !content.contains("</think>"),
        "literal closing tag leaked into content: {content:?}"
    );
}

/// Regression for Kimi K2.5 tool-continuation turns.
///
/// Kimi K2.5 direct answers after tool results should remain normal content.
/// The `last_is_tool` guard from PR #8442 must still suppress forced
/// prompt-injected reasoning for Kimi, even though DeepSeek V4 preserves it.
#[tokio::test]
async fn postprocessor_parsing_stream_kimi_k25_tool_continuation_suppresses_injected_reasoning() {
    let preprocessor = build_preprocessor(Some("kimi_k25"), None);
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "messages": [
            {"role": "user", "content": "Create and run a hello-world script."},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_python",
                        "arguments": "{\"path\":\"/tmp/hello.py\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Hello, world!"
            }
        ],
        "model": "moonshotai/Kimi-K2.5-Instruct",
        "stream": true
    }))
    .unwrap();

    let input_chunks = vec![
        mock_content_chunk("Done. Output: `Hello, world!`"),
        mock_final_chunk(),
    ];

    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(
        reasoning, "",
        "direct post-tool Kimi answer must not be mislabeled as reasoning_content",
    );
    assert_eq!(content, "Done. Output: `Hello, world!`");
}

/// vLLM parity: `chat_template_kwargs={"enable_thinking": false}` disables
/// Nemotron v3 reasoning extraction. Plain backend text should remain normal
/// content and must not be reclassified as `reasoning_content`.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_enable_thinking_false_returns_content() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "enable_thinking": false
        }))
        .unwrap(),
    );

    let input_chunks = vec![mock_content_chunk("This is plain content")];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(reasoning, "");
    assert_eq!(content, "This is plain content");
}

/// vLLM parity: `chat_template_kwargs={"force_nonempty_content": true}` turns
/// a leading `<think>...` response into normal content instead of reasoning.
/// Dynamo checks this in the postprocessor because request flags are applied
/// before stream parsing, not inside the raw reasoning parser.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_strips_start_token() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "force_nonempty_content": true
        }))
        .unwrap(),
    );

    let input_chunks = vec![
        mock_content_chunk("<thi"),
        mock_content_chunk("nk>This is plain content"),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(reasoning, "");
    assert_eq!(content, "This is plain content");
}

/// Regression: if the stream ends after a partial `<think>` prefix, those bytes
/// are valid content and must be flushed before the terminal chunk is emitted.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_flushes_partial_prefix_on_finish()
{
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "force_nonempty_content": true
        }))
        .unwrap(),
    );

    let input_chunks = vec![mock_content_chunk("<thi"), mock_final_chunk()];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut finish_reasons = Vec::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
            if let Some(fr) = choice.finish_reason {
                finish_reasons.push(fr);
            }
        }
    }

    assert_eq!(reasoning, "");
    assert_eq!(content, "<thi");
    assert!(finish_reasons.contains(&FinishReason::Stop));
}

/// Regression: the EOF path has no terminal delta to carry the buffered bytes,
/// so the postprocessor must emit one final content chunk itself.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_flushes_partial_prefix_on_eof() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "force_nonempty_content": true
        }))
        .unwrap(),
    );

    let input_chunks = vec![mock_content_chunk("<thi")];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(reasoning, "");
    assert_eq!(content, "<thi");
}

/// Dynamo already represents streamed responses as `choices: Vec<_>`, so this
/// test is not adding new `n > 1` behavior. It verifies that the Nemotron v3
/// `force_nonempty_content=true` path does not use one shared strip buffer for
/// all choices. Both choices receive a split `<think>` prefix (`"<thi"` then
/// `"nk>..."`). If the helper keeps only one global buffer/decided flag, choice
/// 0 can consume the prefix state and choice 1 can leak `<think>` or lose text.
/// The expected behavior is that each `choice.index` strips its own leading
/// prefix independently and returns only normal content.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_tracks_prefix_per_choice() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "force_nonempty_content": true
        }))
        .unwrap(),
    );

    let input_chunks = vec![
        mock_multi_choice_content_chunk(&[(0, "<thi"), (1, "<thi")]),
        mock_multi_choice_content_chunk(&[(0, "nk>First"), (1, "nk>Second")]),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut content_by_choice = BTreeMap::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(c) = &choice.delta.content {
                content_by_choice
                    .entry(choice.index)
                    .or_insert_with(String::new)
                    .push_str(get_text(c));
            }
            assert!(
                choice.delta.reasoning_content.is_none(),
                "reasoning_content must stay empty when force_nonempty_content=true"
            );
        }
    }

    assert_eq!(content_by_choice.get(&0).map(String::as_str), Some("First"));
    assert_eq!(
        content_by_choice.get(&1).map(String::as_str),
        Some("Second")
    );
}

/// Regression: MiniMax + tool_choice=required + SGLang guided decoding.
///
/// The reasoning parser (minimax_append_think) synthesizes a `<think>` opener
/// on the first chunk, so without guardrails the constrained JSON tool-call
/// payload would be classified entirely as `reasoning_content` because the
/// constrained output never emits `</think>`. tool_choice=required/named
/// must therefore bypass the reasoning parser, letting the jail extract the
/// bare JSON array into structured tool_calls.
#[tokio::test]
async fn postprocessor_parsing_stream_minimax_required_bypasses_reasoning() {
    let preprocessor = build_preprocessor(Some("minimax_append_think"), Some("minimax_m2"));

    // Baseline request with tools, then force tool_choice=required.
    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    let tools: Vec<dynamo_protocols::types::ChatCompletionTool> =
        serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }]))
        .unwrap();
    request.inner.tools = Some(tools);
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    // Simulate SGLang guided-decoding output: bare JSON array, no markers.
    let bare_json = r#"[{"name": "get_weather", "parameters": {"location": "San Francisco"}}]"#;
    let input_chunks = vec![mock_content_chunk(bare_json), mock_final_chunk()];

    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut merged_tool_calls: BTreeMap<u32, MergedToolCall> = BTreeMap::new();
    let mut finish_reasons = Vec::new();

    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
            if let Some(tcs) = &choice.delta.tool_calls {
                for tc in tcs {
                    merged_tool_calls
                        .entry(tc.index)
                        .or_default()
                        .merge_from(tc);
                }
            }
            if let Some(fr) = choice.finish_reason {
                finish_reasons.push(fr);
            }
        }
    }

    // The bare-JSON tool call must end up in tool_calls — not in reasoning_content.
    assert!(
        reasoning.is_empty(),
        "reasoning_content must be empty when tool_choice=required forces bare JSON, got: {reasoning:?}"
    );
    assert!(
        !content.contains("get_weather"),
        "tool call JSON must not leak into content, got: {content:?}"
    );

    let tool_calls: Vec<MergedToolCall> = merged_tool_calls.values().cloned().collect();
    assert_eq!(tool_calls.len(), 1, "expected one tool call");
    assert_eq!(tool_calls[0].name.as_deref(), Some("get_weather"));
    let args: Value = serde_json::from_str(&tool_calls[0].arguments).unwrap();
    assert_eq!(args, serde_json::json!({"location": "San Francisco"}));

    // tool_choice=required: finish_reason must be rewritten to ToolCalls.
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// Regression: Nemotron Nano/Super + the smoke-test required tool-call case.
///
/// Mirrors dynamo-deploy smoke_test.py::case_completions_tool_call_required:
/// "What is the weather in San Francisco?" with `tool_choice="required"` and
/// the `get_weather` tool. The backend emits a bare guided-decoding JSON
/// payload; the JSON must be consumed by the tool jail, not surfaced as
/// content or `reasoning_content`. Two parser families:
///   * `nemotron_nano` is force-reasoning, so the preprocessor skips reasoning
///     parsing entirely under `tool_choice=required`. `prompt_injected_reasoning`
///     is moot.
///   * `nemotron_deci` is non-force-reasoning (alias for the basic_parser shape
///     also used by `glm45`).
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_required_smoke_case() {
    for (case, parser, prompt_injected_reasoning) in [
        ("nano", "nemotron_nano", true),
        ("super/deci", "nemotron_deci", false),
    ] {
        let preprocessor = build_preprocessor(Some(parser), Some(parser));

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "nvidia/nvidia/nemotron-3-super-120b-long-ctx",
                "messages": [
                    {"role": "user", "content": "What is the weather in San Francisco?"}
                ],
                "stream": true,
                "temperature": 0.0
            }))
            .unwrap();
        let tools: Vec<dynamo_protocols::types::ChatCompletionTool> =
            serde_json::from_value(serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }]))
            .unwrap();
        request.inner.tools = Some(tools);
        request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

        let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
        let input_chunks = vec![mock_content_chunk(bare_json), mock_final_chunk()];

        let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
        let output_stream = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, prompt_injected_reasoning, false)
            .expect("postprocessor_parsing_stream should build");

        let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
            output_stream.collect().await;

        let mut reasoning = String::new();
        let mut content = String::new();
        let mut merged_tool_calls: BTreeMap<u32, MergedToolCall> = BTreeMap::new();
        let mut finish_reasons = Vec::new();

        for output in &output_chunks {
            let Some(data) = output.data.as_ref() else {
                continue;
            };
            for choice in &data.inner.choices {
                if let Some(r) = &choice.delta.reasoning_content {
                    reasoning.push_str(r);
                }
                if let Some(c) = &choice.delta.content {
                    content.push_str(get_text(c));
                }
                if let Some(tcs) = &choice.delta.tool_calls {
                    for tc in tcs {
                        merged_tool_calls
                            .entry(tc.index)
                            .or_default()
                            .merge_from(tc);
                    }
                }
                if let Some(fr) = choice.finish_reason {
                    finish_reasons.push(fr);
                }
            }
        }

        assert!(
            reasoning.is_empty(),
            "{case}: reasoning_content must be empty when tool_choice=required forces bare JSON, got: {reasoning:?}"
        );
        assert!(
            !content.contains("get_weather"),
            "{case}: tool-call JSON must not leak into content, got: {content:?}"
        );
        assert!(
            !content.contains("<tool_call>"),
            "{case}: raw <tool_call> XML must not leak into content, got: {content:?}"
        );

        let tool_calls: Vec<MergedToolCall> = merged_tool_calls.values().cloned().collect();
        assert_eq!(tool_calls.len(), 1, "{case}: expected one tool call");
        assert_eq!(
            tool_calls[0].name.as_deref(),
            Some("get_weather"),
            "{case}: wrong tool name"
        );
        let args: Value = serde_json::from_str(&tool_calls[0].arguments).unwrap();
        assert_eq!(
            args,
            serde_json::json!({"location": "San Francisco"}),
            "{case}: wrong arguments"
        );
        assert!(
            finish_reasons.contains(&FinishReason::ToolCalls),
            "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
        );
    }
}

/// Regression: MiniMax + tool_choice=named + SGLang guided decoding.
/// Same constraint as the required variant, but OpenAI spec says named
/// keeps finish_reason=Stop.
#[tokio::test]
async fn postprocessor_parsing_stream_minimax_named_bypasses_reasoning() {
    let preprocessor = build_preprocessor(Some("minimax_append_think"), Some("minimax_m2"));

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    let tools: Vec<dynamo_protocols::types::ChatCompletionTool> =
        serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }]))
        .unwrap();
    request.inner.tools = Some(tools);
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        "get_weather".to_string().into(),
    ));

    let bare_json = r#"[{"name": "get_weather", "parameters": {"location": "Tokyo"}}]"#;
    let input_chunks = vec![mock_content_chunk(bare_json), mock_final_chunk()];

    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut merged_tool_calls: BTreeMap<u32, MergedToolCall> = BTreeMap::new();
    let mut finish_reasons = Vec::new();

    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(tcs) = &choice.delta.tool_calls {
                for tc in tcs {
                    merged_tool_calls
                        .entry(tc.index)
                        .or_default()
                        .merge_from(tc);
                }
            }
            if let Some(fr) = choice.finish_reason {
                finish_reasons.push(fr);
            }
        }
    }

    assert!(
        reasoning.is_empty(),
        "reasoning_content must be empty for tool_choice=named, got: {reasoning:?}"
    );

    let tool_calls: Vec<MergedToolCall> = merged_tool_calls.values().cloned().collect();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name.as_deref(), Some("get_weather"));

    // OpenAI spec: emitting tool_calls always rewrites finish_reason to ToolCalls,
    // regardless of whether tool_choice was auto, required, or named.
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "named tool_choice with emitted tool_calls should finish as ToolCalls, got: {finish_reasons:?}"
    );
}

/// Regression: MiniMax + tool_choice=named + the SingleObject guided-decoding
/// schema (bare parameters, no `{name, parameters}` wrapper). Exercises the
/// `parse_tool_choice_json` fallback — if the reasoning parser weren't gated
/// off, the `<think>` prefix it unconditionally prepends would make the bare
/// JSON unparseable by that fallback, and the tool call would leak as content.
#[tokio::test]
async fn postprocessor_parsing_stream_minimax_named_bare_parameters() {
    let preprocessor = build_preprocessor(Some("minimax_append_think"), Some("minimax_m2"));

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    let tools: Vec<dynamo_protocols::types::ChatCompletionTool> =
        serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }]))
        .unwrap();
    request.inner.tools = Some(tools);
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        "get_weather".to_string().into(),
    ));

    // SingleObject schema: just the parameters, no wrapper.
    let bare_params = r#"{"location": "Paris", "unit": "celsius"}"#;
    let input_chunks = vec![mock_content_chunk(bare_params), mock_final_chunk()];

    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut merged_tool_calls: BTreeMap<u32, MergedToolCall> = BTreeMap::new();

    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
            if let Some(tcs) = &choice.delta.tool_calls {
                for tc in tcs {
                    merged_tool_calls
                        .entry(tc.index)
                        .or_default()
                        .merge_from(tc);
                }
            }
        }
    }

    assert!(
        reasoning.is_empty(),
        "reasoning_content must be empty (parser must be gated off), got: {reasoning:?}"
    );
    assert!(
        !content.contains("<think>"),
        "no <think> prefix should reach the client, got: {content:?}"
    );

    let tool_calls: Vec<MergedToolCall> = merged_tool_calls.values().cloned().collect();
    assert_eq!(tool_calls.len(), 1, "expected one tool call");
    assert_eq!(tool_calls[0].name.as_deref(), Some("get_weather"));
    let args: Value = serde_json::from_str(&tool_calls[0].arguments).unwrap();
    assert_eq!(
        args,
        serde_json::json!({"location": "Paris", "unit": "celsius"})
    );
}

// Guided tool-choice × reasoning-parser family × prompt injection × backend
// output shape. Each row asserts: tool_calls extracted correctly, no JSON
// or <think> leakage into content, reasoning_content holds only reasoning.

/// Single `get_weather(location)` tool shared by every matrix row.
fn single_weather_tool() -> Vec<ChatCompletionTool> {
    serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }]))
    .unwrap()
}

/// Streaming chat completion request preconfigured with the matrix tool.
fn streaming_tool_request(
    tool_choice: ChatCompletionToolChoiceOption,
) -> NvCreateChatCompletionRequest {
    let mut request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
        "stream": true,
        "temperature": 0.0
    }))
    .unwrap();
    request.inner.tools = Some(single_weather_tool());
    request.inner.tool_choice = Some(tool_choice);
    request
}

struct DrainOutput {
    reasoning: String,
    content: String,
    tool_calls: Vec<MergedToolCall>,
    finish_reasons: Vec<FinishReason>,
}

async fn drain_stream(
    output_stream: impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
) -> DrainOutput {
    let output_chunks: Vec<_> = Box::pin(output_stream).collect().await;
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut merged: BTreeMap<u32, MergedToolCall> = BTreeMap::new();
    let mut finish_reasons = Vec::new();

    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(r) = &choice.delta.reasoning_content {
                reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
            if let Some(tcs) = &choice.delta.tool_calls {
                for tc in tcs {
                    merged.entry(tc.index).or_default().merge_from(tc);
                }
            }
            if let Some(fr) = choice.finish_reason {
                finish_reasons.push(fr);
            }
        }
    }
    DrainOutput {
        reasoning,
        content,
        tool_calls: merged.values().cloned().collect(),
        finish_reasons,
    }
}

/// Assert the standard "tool call extracted, nothing leaks" success shape
/// shared by every matrix row that expects a successful extraction.
fn assert_clean_tool_call(
    case: &str,
    content: &str,
    tool_calls: &[MergedToolCall],
    expected_location: &str,
) {
    assert!(
        !content.contains("get_weather"),
        "{case}: tool-call JSON must not leak into content, got: {content:?}"
    );
    assert!(
        !content.contains("<think>") && !content.contains("</think>"),
        "{case}: think markers must not leak into content, got: {content:?}"
    );
    assert_eq!(tool_calls.len(), 1, "{case}: expected one tool call");
    assert_eq!(
        tool_calls[0].name.as_deref(),
        Some("get_weather"),
        "{case}: wrong tool name"
    );
    let args: Value = serde_json::from_str(&tool_calls[0].arguments)
        .unwrap_or_else(|e| panic!("{case}: arguments not valid JSON: {e}"));
    assert_eq!(
        args,
        serde_json::json!({"location": expected_location}),
        "{case}: wrong arguments"
    );
}

/// Force-reasoning parser + required + bare JSON, both prompt_injected values.
/// The reasoning stage must bypass bare JSON so the jail can extract the tool call.
#[tokio::test]
async fn tool_choice_matrix_force_reasoning_required_bare_json() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;

    for &reasoning_parser in FORCE_REASONING_PARSERS {
        for (case, prompt_injected) in [
            (
                "1a: force-reasoning + required + prompt_injected=false",
                false,
            ),
            (
                "1b: force-reasoning + required + prompt_injected=true",
                true,
            ),
        ] {
            let case = format!("{case} + {reasoning_parser}");
            let preprocessor = build_preprocessor(Some(reasoning_parser), Some("nemotron_nano"));
            let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
            let input_stream = stream::iter(
                vec![
                    mock_content_chunk(" \n"),
                    mock_content_chunk("["),
                    mock_content_chunk(&bare_json[1..]),
                    mock_final_chunk(),
                ]
                .into_iter()
                .map(Annotated::from_data),
            );
            let output_stream = preprocessor
                .postprocessor_parsing_stream(input_stream, &request, prompt_injected, false)
                .expect("postprocessor_parsing_stream should build");
            let DrainOutput {
                reasoning,
                content,
                tool_calls,
                finish_reasons,
            } = drain_stream(output_stream).await;

            assert!(
                reasoning.is_empty(),
                "{case}: guided JSON must not become reasoning_content, got: {reasoning:?}"
            );
            assert_clean_tool_call(&case, &content, &tool_calls, "San Francisco");
            assert!(
                finish_reasons.contains(&FinishReason::ToolCalls),
                "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
            );
        }
    }
}

/// Force-reasoning parser + named + bare parameters; same bare-JSON contract
/// as Required, but the jail supplies the selected function name.
#[tokio::test]
async fn tool_choice_matrix_force_reasoning_named_bare_json() {
    let bare_params = r#"{"location":"San Francisco"}"#;

    for &reasoning_parser in FORCE_REASONING_PARSERS {
        let preprocessor = build_preprocessor(Some(reasoning_parser), Some("nemotron_nano"));
        let request = streaming_tool_request(ChatCompletionToolChoiceOption::Named(
            ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: FunctionName {
                    name: "get_weather".to_string(),
                },
            },
        ));

        let input_stream = stream::iter(
            vec![mock_content_chunk(bare_params), mock_final_chunk()]
                .into_iter()
                .map(Annotated::from_data),
        );
        let output_stream = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, true, false)
            .expect("postprocessor_parsing_stream should build");
        let DrainOutput {
            reasoning,
            content,
            tool_calls,
            ..
        } = drain_stream(output_stream).await;

        let case = format!("2: force-reasoning + named + bare params + {reasoning_parser}");
        assert!(
            reasoning.is_empty(),
            "{case}: reasoning_content must be empty, got: {reasoning:?}"
        );
        assert_clean_tool_call(&case, &content, &tool_calls, "San Francisco");
    }
}

/// Per-request thinking disablement must retain the old required-tool behavior
/// after Nemotron v3 opts into guided-output shape detection.
#[tokio::test]
async fn tool_choice_nemotron_v3_required_thinking_disabled_keeps_bare_json() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("nemotron_v3"), Some("nemotron_nano"));
    let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    request.chat_template_args =
        Some(serde_json::from_value(serde_json::json!({"enable_thinking": false})).unwrap());
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        ..
    } = drain_stream(output_stream).await;

    let case = "Nemotron v3 required + thinking disabled + bare JSON";
    assert!(
        reasoning.is_empty(),
        "{case}: reasoning_content must be empty, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
}

/// Required guided JSON must follow each force parser's close marker,
/// including markers split across stream chunks.
#[tokio::test]
async fn tool_choice_force_reasoning_required_keeps_reasoning_before_guided_json() {
    for &(reasoning_parser, close_marker) in REASONING_BEFORE_GUIDED_JSON_PARSERS {
        for prompt_injected in [false, true] {
            let preprocessor = build_preprocessor(Some(reasoning_parser), Some("nemotron_nano"));
            let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
            let split_at = close_marker.len() - 2;
            let (close_prefix, close_suffix) = close_marker.split_at(split_at);
            let close_and_json = format!(
                r#"{close_suffix}[{{"name":"get_weather","parameters":{{"location":"San Francisco"}}}}]"#
            );
            let input_stream = stream::iter(
                vec![
                    mock_content_chunk("Let me "),
                    mock_content_chunk("check."),
                    mock_content_chunk(close_prefix),
                    mock_content_chunk(&close_and_json),
                    mock_final_chunk(),
                ]
                .into_iter()
                .map(Annotated::from_data),
            );
            let output_stream = preprocessor
                .postprocessor_parsing_stream(input_stream, &request, prompt_injected, false)
                .expect("postprocessor_parsing_stream should build");
            let DrainOutput {
                reasoning,
                content,
                tool_calls,
                finish_reasons,
            } = drain_stream(output_stream).await;

            let case = format!(
                "{reasoning_parser} required + reasoning boundary + prompt_injected={prompt_injected}"
            );
            assert_eq!(
                reasoning, "Let me check.",
                "{case}: reasoning_content must preserve the pre-boundary text"
            );
            assert_clean_tool_call(&case, &content, &tool_calls, "San Francisco");
            assert!(
                finish_reasons.contains(&FinishReason::ToolCalls),
                "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
            );
        }
    }
}

/// Named guided decoding emits only the selected function's parameter object.
/// Reasoning before that object must be separated without preventing the named
/// immediate jail from constructing the tool call.
#[tokio::test]
async fn tool_choice_force_reasoning_named_keeps_reasoning_before_guided_params() {
    for &(reasoning_parser, close_marker) in REASONING_BEFORE_GUIDED_JSON_PARSERS {
        let preprocessor = build_preprocessor(Some(reasoning_parser), Some("nemotron_nano"));
        let request = streaming_tool_request(ChatCompletionToolChoiceOption::Named(
            "get_weather".to_string().into(),
        ));
        let input_stream = stream::iter(
            vec![
                mock_content_chunk("Let me check."),
                mock_content_chunk(close_marker),
                mock_content_chunk(r#"{"location":"San Francisco"}"#),
                mock_final_chunk(),
            ]
            .into_iter()
            .map(Annotated::from_data),
        );
        let output_stream = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, true, false)
            .expect("postprocessor_parsing_stream should build");
        let DrainOutput {
            reasoning,
            content,
            tool_calls,
            finish_reasons,
        } = drain_stream(output_stream).await;

        let case = format!("{reasoning_parser} named + reasoning boundary + guided params");
        assert_eq!(
            reasoning, "Let me check.",
            "{case}: reasoning_content must preserve the pre-boundary text"
        );
        assert_clean_tool_call(&case, &content, &tool_calls, "San Francisco");
        assert!(
            finish_reasons.contains(&FinishReason::ToolCalls),
            "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
        );
    }
}

/// Non-force parser + required + no prompt injection + bare JSON: parser
/// runs in non-reasoning mode and passes JSON through.
#[tokio::test]
async fn tool_choice_matrix_non_force_required_no_injection_bare_json() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("qwen3"), Some("hermes"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        ..
    } = drain_stream(output_stream).await;

    let case = "3: non-force + required + prompt_injected=false + bare JSON";
    assert!(
        reasoning.is_empty(),
        "{case}: parser must not produce reasoning when no <think> seen, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
}

/// Non-force parser + required + reasoning</think>JSON (the Qwen3.x production
/// shape; verified end-to-end against Qwen3.6-35B-A3B-FP8). Parser strips
/// reasoning, jail gets JSON.
#[tokio::test]
async fn tool_choice_matrix_non_force_required_prompt_injected_with_close_marker() {
    let stream_text = r#"Let me check.</think>[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("qwen3"), Some("hermes"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let input_stream = stream::iter(
        vec![mock_content_chunk(stream_text), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        ..
    } = drain_stream(output_stream).await;

    let case = "4: non-force + required + prompt_injected=true + reasoning</think>JSON";
    assert_eq!(
        reasoning.trim(),
        "Let me check.",
        "{case}: reasoning_content should hold only the pre-</think> text, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
}

/// CASE 5 — non-force parser + required + `prompt_injected_reasoning=true`
/// + bare JSON (no `</think>`). Documents the **backend contract** rather
/// than asserting recovery: when `--dyn-reasoning-parser X` is set, vLLM's
/// auto-forward in `components/src/dynamo/vllm/main.py:506-507` instantiates
/// a reasoner whose `should_fill_bitmask` gate (vLLM
/// `v1/structured_output/__init__.py:301`) keeps the xgrammar bitmask off
/// until `</think>` appears in the output. Consequently any "bare guided
/// JSON" emitted before `</think>` was never grammar-constrained — it's a
/// backend-bug shape, not a normal production output.
///
/// This test pins the current behavior so future regressions are loud: if
/// we later add an EOF fallback to `BasicReasoningParser` to flush
/// accumulated reasoning as content, this assertion needs to flip.
#[tokio::test]
async fn tool_choice_matrix_non_force_required_prompt_injected_bare_json_contract() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("qwen3"), Some("hermes"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        ..
    } = drain_stream(output_stream).await;

    let case = "5 (contract): non-force + required + prompt_injected=true + bare JSON";
    assert!(
        tool_calls.is_empty(),
        "{case}: contract case currently extracts no tool_calls (backend bug shape), got: {tool_calls:?}"
    );
    assert!(
        content.is_empty(),
        "{case}: content must remain empty (no leak), got: {content:?}"
    );
    assert!(
        reasoning.contains("get_weather"),
        "{case}: parser pins the JSON in reasoning_content under the broken contract, got: {reasoning:?}"
    );
}

/// DeepSeek V4 + required + `prompt_injected_reasoning=true` + bare JSON.
///
/// This is the production failure shape from DeepSeek V4 Pro: the V4 formatter
/// seeds `<think>`, but vLLM guided decoding emits the constrained JSON payload
/// without a closing `</think>`. The postprocessor must let the immediate jail
/// parse that JSON instead of classifying it as reasoning_content.
#[tokio::test]
async fn tool_choice_deepseek_v4_required_prompt_injected_bare_json_recovers() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("deepseek_v4"), Some("deepseek_v4"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "DeepSeek V4 required + prompt_injected=true + bare JSON";
    assert!(
        reasoning.is_empty(),
        "{case}: guided JSON must not be classified as reasoning_content, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// MiniMax M3 + required + `prompt_injected_reasoning=true` + bare JSON.
///
/// MiniMax M3 chat templates seed `<mm:think>` rather than `<think>`. When
/// guided decoding emits the constrained tool-call JSON from token 0, that JSON
/// must bypass reasoning parsing so the immediate jail can extract tool_calls.
#[tokio::test]
async fn tool_choice_minimax_m3_required_prompt_injected_bare_json_recovers() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("minimax_m3"), Some("minimax_m3"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "MiniMax M3 required + prompt_injected=true + bare JSON";
    assert!(
        reasoning.is_empty(),
        "{case}: guided JSON must not be classified as reasoning_content, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// Exercises the experimental parsers-v2 gate end-to-end. `tool_choice=Auto` + a v2
/// family (`qwen3_coder`) is the only combination the gate routes to
/// `tool_parser_v2::apply_stream`; `required`/`named` (above) always stay on the v1
/// jail. The flag is read once at process startup, so a single test covers BOTH
/// paths via the startup switch: run with `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` unset
/// and this goes through the v1 jail; set it and the identical stream goes through
/// the v2 parser. A complete tool call must extract cleanly with no raw markup
/// leaking into content on either path, so the same assertion validates both.
#[tokio::test]
async fn tool_calls_qwen3_coder_auto_routes_through_experimental_gate() {
    let xml = "<tool_call>\n<function=get_weather>\n<parameter=location>\nSan Francisco\n</parameter>\n</function>\n</tool_call>";
    let preprocessor = build_preprocessor(None, Some("qwen3_coder"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Auto);
    let input_stream = stream::iter(
        vec![mock_content_chunk(xml), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        content,
        tool_calls,
        finish_reasons,
        ..
    } = drain_stream(output_stream).await;

    let path = if std::env::var("DYN_ENABLE_EXPERIMENTAL_PARSERS_V2")
        .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
    {
        "qwen3_coder auto -> dynamo-parsers-v2 (DYN_ENABLE_EXPERIMENTAL_PARSERS_V2 on)"
    } else {
        "qwen3_coder auto -> v1 jail (flag off)"
    };
    assert_clean_tool_call(path, &content, &tool_calls, "San Francisco");
    // Both paths must honor the OpenAI contract: a tool-call stream terminates with
    // finish_reason=ToolCalls — v1 via the jail's fix_finish_reason, v2 via apply_stream.
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{path}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// DeepSeek V4/GLM + required + `prompt_injected_reasoning=true` +
/// reasoning-close-marker JSON. This is not bare JSON; the reasoning parser
/// must strip the pre-`</think>` prefix before the immediate jail sees JSON.
#[tokio::test]
async fn tool_choice_prompt_injected_close_marker_json_keeps_reasoning_parser_for_dsv4_glm() {
    let stream_text = r#"Let me check.</think>[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;

    for (case, reasoning_parser, tool_call_parser) in [
        ("DeepSeek V4", "deepseek_v4", "deepseek_v4"),
        ("GLM45", "glm45", "glm47"),
    ] {
        let preprocessor = build_preprocessor(Some(reasoning_parser), Some(tool_call_parser));
        let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
        let input_stream = stream::iter(
            vec![mock_content_chunk(stream_text), mock_final_chunk()]
                .into_iter()
                .map(Annotated::from_data),
        );
        let output_stream = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, true, false)
            .expect("postprocessor_parsing_stream should build");
        let DrainOutput {
            reasoning,
            content,
            tool_calls,
            finish_reasons,
        } = drain_stream(output_stream).await;

        assert_eq!(
            reasoning.trim(),
            "Let me check.",
            "{case}: reasoning_content should hold only the pre-</think> text, got: {reasoning:?}"
        );
        assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
        assert!(
            finish_reasons.contains(&FinishReason::ToolCalls),
            "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
        );
    }
}

/// DeepSeek V4 + named tool_choice + `prompt_injected_reasoning=true` + bare
/// parameters object. Same bug as the required case, but exercises the named
/// SingleObject immediate-jail path.
#[tokio::test]
async fn tool_choice_deepseek_v4_named_prompt_injected_bare_params_recovers() {
    let bare_params = r#"{"location":"San Francisco"}"#;
    let preprocessor = build_preprocessor(Some("deepseek_v4"), Some("deepseek_v4"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Named(
        "get_weather".to_string().into(),
    ));
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_params), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "DeepSeek V4 named + prompt_injected=true + bare params";
    assert!(
        reasoning.is_empty(),
        "{case}: guided JSON must not be classified as reasoning_content, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// MiniMax M3 + named tool_choice + `prompt_injected_reasoning=true` + bare
/// parameters object. Exercises the named SingleObject immediate-jail path.
#[tokio::test]
async fn tool_choice_minimax_m3_named_prompt_injected_bare_params_recovers() {
    let bare_params = r#"{"location":"San Francisco"}"#;
    let preprocessor = build_preprocessor(Some("minimax-m3"), Some("minimax-m3"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Named(
        "get_weather".to_string().into(),
    ));
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_params), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "MiniMax M3 named + prompt_injected=true + bare params";
    assert!(
        reasoning.is_empty(),
        "{case}: guided JSON must not be classified as reasoning_content, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// GLM + required + `prompt_injected_reasoning=true` + bare JSON.
///
/// Mirrors the DeepSeek V4 guided-decoding failure shape for the `glm45`
/// reasoning parser paired with the `glm47` tool-call parser used by GLM-5.1.
#[tokio::test]
async fn tool_choice_glm45_required_prompt_injected_bare_json_recovers() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("glm45"), Some("glm47"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "GLM45 required + prompt_injected=true + bare JSON";
    assert!(
        reasoning.is_empty(),
        "{case}: guided JSON must not be classified as reasoning_content, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// GLM + named tool_choice + `prompt_injected_reasoning=true` + bare
/// parameters object. Exercises the named SingleObject immediate-jail path.
#[tokio::test]
async fn tool_choice_glm45_named_prompt_injected_bare_params_recovers() {
    let bare_params = r#"{"location":"San Francisco"}"#;
    let preprocessor = build_preprocessor(Some("glm45"), Some("glm47"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Named(
        "get_weather".to_string().into(),
    ));
    let input_stream = stream::iter(
        vec![mock_content_chunk(bare_params), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "GLM45 named + prompt_injected=true + bare params";
    assert!(
        reasoning.is_empty(),
        "{case}: guided JSON must not be classified as reasoning_content, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// Structural tags must preserve both prompt-injected and force reasoners.
#[tokio::test]
async fn tool_choice_structural_tag_keeps_reasoning_parser() {
    for (case, reasoning_parser, tool_call_parser, structural_tool_call, prompt_injected) in [
        (
            "DeepSeek V4 DSML",
            "deepseek_v4",
            "deepseek_v4",
            "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"location\" string=\"true\">San Francisco</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>",
            true,
        ),
        (
            "GLM XML",
            "glm45",
            "glm47",
            "<tool_call>get_weather\
<arg_key>location</arg_key><arg_value>San Francisco</arg_value>\
</tool_call>",
            true,
        ),
        (
            "Nemotron v3 Qwen3-Coder XML",
            "nemotron_v3",
            "qwen3_coder",
            "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
San Francisco\n\
</parameter>\n\
</function>\n\
</tool_call>",
            false,
        ),
    ] {
        let preprocessor = build_preprocessor(Some(reasoning_parser), Some(tool_call_parser));
        let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
        let stream_text = format!("Let me check.</think>{structural_tool_call}");
        let input_stream = stream::iter(
            vec![mock_content_chunk(&stream_text), mock_final_chunk()]
                .into_iter()
                .map(Annotated::from_data),
        );
        let output_stream = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, prompt_injected, true)
            .expect("postprocessor_parsing_stream should build");
        let DrainOutput {
            reasoning,
            content,
            tool_calls,
            finish_reasons,
        } = drain_stream(output_stream).await;

        assert_eq!(
            reasoning.trim(),
            "Let me check.",
            "{case}: reasoning_content should hold only the pre-</think> text, got: {reasoning:?}"
        );
        assert!(
            content.is_empty(),
            "{case}: reasoning prefix or structural tags leaked into content: {content:?}"
        );
        assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
        assert!(
            finish_reasons.contains(&FinishReason::ToolCalls),
            "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
        );
    }
}

/// CASE 6 — Immediate jail mode + first chunk has only `reasoning_content`
/// (no text delta) + JSON arrives in a later chunk. Regression for the
/// `jail.rs:678` fix: before the fix, the else branch hardcoded
/// `starts_jailed=false`, silently disabling Immediate mode whenever the
/// first chunk for a choice initialized through the no-content path. After
/// the fix, the state respects `JailMode::Immediate` and the JSON in the
/// later chunk is captured by the jail.
#[tokio::test]
async fn tool_choice_matrix_immediate_jail_reasoning_only_first_chunk() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("qwen3"), Some("hermes"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let input_stream = stream::iter(
        vec![
            mock_reasoning_only_chunk("thinking briefly"),
            mock_content_chunk(bare_json),
            mock_final_chunk(),
        ]
        .into_iter()
        .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning,
        content,
        tool_calls,
        ..
    } = drain_stream(output_stream).await;

    let case = "6: Immediate jail + reasoning-only first chunk + JSON later";
    assert!(
        reasoning.contains("thinking briefly"),
        "{case}: reasoning_content from the first chunk must reach the client, got: {reasoning:?}"
    );
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
}

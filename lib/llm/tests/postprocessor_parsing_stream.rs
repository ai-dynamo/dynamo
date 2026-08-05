// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::ParsingOptions;
use dynamo_llm::protocols::openai::chat_completions::aggregator::ChatCompletionAggregator;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    NvCreateChatCompletionStreamResponse,
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
    "deepseek_v3",
    "deepseek_v3_1",
    "deepseek_v3_2",
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
    ("deepseek_v3", "</think>"),
    ("deepseek_v3_1", "</think>"),
    ("deepseek_v3_2", "</think>"),
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

/// Trailing usage-only chunk: `choices: []` plus a usage payload, matching what
/// `transform_postprocessor_stream` appends when usage reporting is on (always
/// for non-streaming). It carries no delta slot, so it must never be picked as
/// the EOF-flush envelope.
fn mock_usage_only_chunk() -> NvCreateChatCompletionStreamResponse {
    use dynamo_protocols::types::{CompletionUsage, CreateChatCompletionStreamResponse};
    NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: Some(CompletionUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    }
}

/// Terminal `finish_reason=Stop` chunk carrying one finish per listed choice
/// index — the multi-choice analog of `mock_final_chunk`, so an `n > 1` stream
/// flushes every choice's jail state.
fn mock_multi_choice_final_chunk(indices: &[u32]) -> NvCreateChatCompletionStreamResponse {
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
    };
    #[allow(deprecated)]
    let choices = indices
        .iter()
        .map(|index| ChatChoiceStream {
            index: *index,
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

fn kimi_tool_continuation_request(
    model: &str,
    thinking: Option<bool>,
) -> NvCreateChatCompletionRequest {
    let mut request = serde_json::json!({
        "messages": [
            {"role": "user", "content": "What is the weather in London?"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"London\"}"
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"temperature\":15,\"unit\":\"celsius\",\"condition\":\"cloudy\"}"
            }
        ],
        "model": model,
        "stream": true
    });
    if let Some(thinking) = thinking {
        request["chat_template_kwargs"] = serde_json::json!({"thinking": thinking});
    }
    serde_json::from_value(request).unwrap()
}

async fn run_kimi_tool_continuation(
    request: NvCreateChatCompletionRequest,
    input_chunks: Vec<NvCreateChatCompletionStreamResponse>,
) -> DrainOutput {
    let preprocessor = build_preprocessor(Some("kimi_k25"), None);
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    drain_stream(output_stream).await
}

/// K2.6 explicit thinking can emit implicit reasoning and a split close marker.
#[tokio::test]
async fn postprocessor_parsing_stream_kimi_k25_tool_continuation_with_thinking_parses_reasoning() {
    let request = kimi_tool_continuation_request("moonshotai/Kimi-K2.6", Some(true));
    let output = run_kimi_tool_continuation(
        request,
        vec![
            mock_content_chunk("The tool returned 15°C and cloudy."),
            mock_content_chunk("</thi"),
            mock_content_chunk("nk>The current weather in London is 15°C and cloudy."),
            mock_final_chunk(),
        ],
    )
    .await;

    assert_eq!(output.reasoning, "The tool returned 15°C and cloudy.");
    assert_eq!(
        output.content,
        "The current weather in London is 15°C and cloudy."
    );
    assert!(
        !output.content.contains("</think>"),
        "literal closing tag leaked into content: {:?}",
        output.content
    );
}

/// Kimi K2.6 enables thinking by default. An omitted `thinking` argument must
/// therefore parse post-tool reasoning exactly like an explicit `true`.
#[tokio::test]
async fn postprocessor_parsing_stream_kimi_k26_omitted_thinking_parses_reasoning() {
    let request = kimi_tool_continuation_request("moonshotai/Kimi-K2.6", None);

    let output = run_kimi_tool_continuation(
        request,
        vec![
            mock_content_chunk("The tool returned 15°C and cloudy."),
            mock_content_chunk("</thi"),
            mock_content_chunk("nk>The current weather in London is 15°C and cloudy."),
            mock_final_chunk(),
        ],
    )
    .await;

    assert_eq!(
        (output.reasoning.as_str(), output.content.as_str()),
        (
            "The tool returned 15°C and cloudy.",
            "The current weather in London is 15°C and cloudy."
        )
    );
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

/// Streaming Nemotron `force_nonempty_content=true`, reasoning-only turn: the
/// parser stays ON (it is no longer disabled for streaming), so an unterminated
/// `<think>...` with no answer parses entirely as reasoning. Under
/// `force_nonempty_content` those deltas are held back rather than streamed as
/// `reasoning_content`, because until the parser leaves the reasoning block it
/// is not yet known whether an answer follows. Here none does, so at
/// end-of-stream the held text is emitted as `content` — the template's
/// non-empty-content promise is honored on the streaming path too.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_reasoning_only_becomes_content() {
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

    assert_eq!(
        content, "This is plain content",
        "with no answer to follow, the held reasoning is emitted as content"
    );
    assert_eq!(
        reasoning, "",
        "nothing is reported as reasoning_content when it had to become content"
    );
}

/// Streaming Nemotron `force_nonempty_content=true`, reasoning+answer turn: the
/// headline fix (Case 1). With the parser on, `<think>reason</think>answer`
/// streams `reason` as `reasoning_content` and `answer` as `content` — no
/// reasoning text or raw `</think>` leaks into content. On `main` (and before
/// this change) the parser was disabled for streaming and only the leading
/// `<think>` was stripped, so `content` was `"reason</think>answer"`.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_stream_reasoning_and_answer_split()
{
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    let input_chunks = vec![
        mock_content_chunk("<think>Let me greet them.</think>Hello!"),
        mock_final_chunk(),
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

    assert_eq!(
        reasoning, "Let me greet them.",
        "reasoning must stream as reasoning_content, not leak into content"
    );
    assert_eq!(
        content, "Hello!",
        "answer must be the content, with no reasoning or </think> leaked in"
    );
}

/// Streaming Nemotron `force_nonempty_content=true` with plain output and no
/// `<think>` at all. Nemotron aliases are force-reasoning parsers, so keeping
/// the parser on raises the question of whether leading text with no start
/// token gets swallowed as `reasoning_content`. It does not: the answer streams
/// through as `content` and `reasoning_content` stays empty. This pins the
/// dynamo-parsers behavior the fix depends on, so a parser-side change that
/// started treating bare leading text as reasoning fails here instead of
/// silently emptying `content` for every plain Nemotron turn.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_stream_plain_content_stays_content()
 {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    let input_chunks = vec![
        mock_content_chunk("Hello"),
        mock_content_chunk("!"),
        mock_final_chunk(),
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

    assert_eq!(
        content, "Hello!",
        "plain output with no <think> must stay content"
    );
    assert_eq!(
        reasoning, "",
        "no <think> means nothing may be reported as reasoning_content"
    );
}

/// Non-streaming parity for the Nemotron `force_nonempty_content` flag,
/// reasoning-only case. Reasoning parsing stays ON for non-streaming, so a
/// `<think>` with no answer parses entirely into `reasoning_content`, leaving
/// `content` empty; the aggregator then surfaces reasoning as `content` when
/// content is empty, gated by `ParsingOptions::move_reasoning_to_content_when_empty`.
/// The chat handler sets that flag for `force_nonempty_content` via
/// `OpenAIPreprocessor::wants_reasoning_as_content_when_empty`; this test mirrors
/// it and asserts non-empty, `<think>`-stripped `content`.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_aggregated_strips_start_token() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "force_nonempty_content": true
        }))
        .unwrap(),
    );

    // Leading `<think>` arrives split across chunks (same input as the streaming
    // test), then a terminal stop chunk closes the choice.
    let input_chunks = vec![
        mock_content_chunk("<thi"),
        mock_content_chunk("nk>This is plain content"),
        mock_final_chunk(),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));

    // Step 1: the shared postprocessor stream (reasoning gate + `<think>` strip).
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    // Step 2: the non-streaming fold, identical to the `stream=false` HTTP path.
    // Options mirror what the chat handler sets for `force_nonempty_content`.
    let response = NvCreateChatCompletionResponse::from_annotated_stream(
        output_stream,
        ParsingOptions::default().with_move_reasoning_to_content_when_empty(true),
    )
    .await
    .expect("aggregation should succeed");

    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.content.as_ref().map(get_text),
        Some("This is plain content"),
        "reasoning-only turn must surface reasoning as non-empty content"
    );
    assert_eq!(
        choice.message.reasoning_content, None,
        "reasoning_content must stay empty when force_nonempty_content=true"
    );
}

/// Non-streaming parity, EOF-flush case: when the stream ends after only a
/// partial `<think>` prefix (`<thi`), those bytes must not be dropped. Reasoning
/// parsing stays on, so `parse_reasoning_content_from_stream` flushes the
/// unterminated buffer at EOF and the aggregator move (mirroring the chat handler
/// for `force_nonempty_content`) surfaces it as non-empty `content`.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_aggregated_flushes_partial_prefix()
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

    // Options mirror what the chat handler sets for `force_nonempty_content`.
    let response = NvCreateChatCompletionResponse::from_annotated_stream(
        output_stream,
        ParsingOptions::default().with_move_reasoning_to_content_when_empty(true),
    )
    .await
    .expect("aggregation should succeed");

    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.content.as_ref().map(get_text),
        Some("<thi"),
        "a partial <think> prefix is valid content and must survive aggregation"
    );
    assert_eq!(choice.message.reasoning_content, None);
}

/// Regression for the leak this PR fixes: a non-streaming Nemotron turn that
/// emits real reasoning AND a real answer must split them, not dump reasoning
/// (and a raw `</think>`) into `content`. On `main` this returned
/// `content = "Let me greet them.</think>Hello!"`, `reasoning_content = None`.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_reasoning_and_answer_split() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);
    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );
    let input_chunks = vec![
        mock_content_chunk("<think>Let me greet them.</think>Hello!"),
        mock_final_chunk(),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let response = NvCreateChatCompletionResponse::from_annotated_stream(
        output_stream,
        ParsingOptions::default().with_move_reasoning_to_content_when_empty(true),
    )
    .await
    .expect("aggregation should succeed");
    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.content.as_ref().map(get_text),
        Some("Hello!"),
        "answer must be the content, with no leaked reasoning or </think>"
    );
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Let me greet them."),
        "reasoning must be preserved in reasoning_content, not moved into content"
    );
}

/// Same reasoning+answer input WITHOUT `force_nonempty_content`: non-streaming
/// reasoning parsing is unchanged, so the split is identical. Confirms the flag
/// is a no-op when content is already present.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_reasoning_and_answer_split_no_flag() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);
    let request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    let input_chunks = vec![
        mock_content_chunk("<think>Let me greet them.</think>Hello!"),
        mock_final_chunk(),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let response = NvCreateChatCompletionResponse::from_annotated_stream(
        output_stream,
        ParsingOptions::default(),
    )
    .await
    .expect("aggregation should succeed");
    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.content.as_ref().map(get_text),
        Some("Hello!")
    );
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Let me greet them.")
    );
}

/// Regression: if the stream ends after a partial `<think>` prefix, those bytes
/// must be flushed (not dropped) before the terminal chunk is emitted. With the
/// parser on for streaming, the parser reports the unterminated buffer as
/// reasoning at `finish_reasoning_stream`, so `<thi` surfaces as
/// `reasoning_content` and `content` stays empty (streaming does no move — the
/// non-streaming aggregated path moves the same bytes into content).
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

    assert_eq!(content, "<thi", "partial prefix must survive as content");
    assert_eq!(reasoning, "");
    assert!(finish_reasons.contains(&FinishReason::Stop));
}

/// Regression: the EOF path has no terminal delta to carry the buffered bytes,
/// so the postprocessor must emit one final chunk itself. With the parser on,
/// the unterminated `<thi` flushes as `reasoning_content` (streaming does no
/// move); the point of the test is that the bytes are not silently dropped.
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

    assert_eq!(content, "<thi", "partial prefix must survive as content");
    assert_eq!(reasoning, "");
}

/// Regression: the EOF flush must survive the trailing usage-only chunk.
///
/// In production `transform_postprocessor_stream` appends a final usage chunk
/// with `choices: []` (always on for non-streaming, opt-in for streaming). If
/// the flush reuses that chunk as its envelope, the per-choice loop iterates an
/// empty vec and the buffered bytes are dropped — exactly the truncated-`<think>`
/// loss the flush exists to prevent. Only content-bearing chunks are retained as
/// the envelope, so `<thi` still surfaces here.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_flushes_partial_prefix_after_usage_chunk()
 {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "force_nonempty_content": true
        }))
        .unwrap(),
    );

    let input_chunks = vec![mock_content_chunk("<thi"), mock_usage_only_chunk()];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut content = String::new();
    let mut usage_chunks = 0;
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        if data.inner.usage.is_some() {
            usage_chunks += 1;
        }
        for choice in &data.inner.choices {
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(
        content, "<thi",
        "partial prefix must survive the trailing usage-only chunk"
    );
    assert_eq!(
        usage_chunks, 1,
        "the flush chunk must not duplicate the usage payload"
    );
}

/// Each choice's parser flushes its own buffered bytes at EOF. With `n > 1` the
/// last content-bearing chunk need not carry every choice, so the flush rebuilds
/// the choice list from the indices that actually have buffered text rather than
/// reusing whichever choices happened to be in that envelope.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_flushes_partial_prefix_per_choice()
{
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "force_nonempty_content": true
        }))
        .unwrap(),
    );

    let input_chunks = vec![
        mock_multi_choice_content_chunk(&[(0, "<th"), (1, "<thi")]),
        // Only choice 0 is present in the last content-bearing chunk; choice 1's
        // buffered "<thi" must still be flushed.
        mock_multi_choice_content_chunk(&[(0, "i")]),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");

    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let mut content_by_index: HashMap<u32, String> = HashMap::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(c) = &choice.delta.content {
                content_by_index
                    .entry(choice.index)
                    .or_default()
                    .push_str(get_text(c));
            }
            assert!(
                choice.delta.reasoning_content.is_none(),
                "no answer followed, so nothing may be reported as reasoning_content"
            );
        }
    }

    assert_eq!(content_by_index.get(&0).map(String::as_str), Some("<thi"));
    assert_eq!(content_by_index.get(&1).map(String::as_str), Some("<thi"));
}

/// Dynamo already represents streamed responses as `choices: Vec<_>`, so this
/// test is not adding new `n > 1` behavior. It verifies that the disabled-reasoning
/// strip path (`enable_thinking=false`, which is where the leading-`<think>` strip
/// still runs) does not use one shared strip buffer for all choices. Both choices
/// receive a split `<think>` prefix (`"<thi"` then `"nk>..."`). If the helper keeps
/// only one global buffer/decided flag, choice 0 can consume the prefix state and
/// choice 1 can leak `<think>` or lose text. The expected behavior is that each
/// `choice.index` strips its own leading prefix independently and returns only
/// normal content. (`force_nonempty_content` no longer takes this path — it keeps
/// the reasoning parser on — so this exercises the strip path via `enable_thinking`.)
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_disabled_reasoning_tracks_prefix_per_choice() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({
            "enable_thinking": false
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
                "reasoning_content must stay empty when reasoning is disabled"
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

/// Streaming chat completion request with OpenAI structured output.
fn streaming_json_schema_request(enable_thinking: bool) -> NvCreateChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Return a country and its capital."}],
        "stream": true,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "capital",
                "schema": {
                    "type": "object",
                    "properties": {
                        "country": {"type": "string"},
                        "capital": {"type": "string"}
                    },
                    "required": ["country", "capital"],
                    "additionalProperties": false
                }
            }
        }
    }))
    .unwrap()
}

/// Streaming chat completion request with OpenAI JSON object response format.
fn streaming_json_object_request(enable_thinking: bool) -> NvCreateChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Return a JSON object."}],
        "stream": true,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "response_format": {
            "type": "json_object"
        }
    }))
    .unwrap()
}

fn enable_opt_in_reasoning(request: &mut NvCreateChatCompletionRequest, reasoning_parser: &str) {
    if matches!(reasoning_parser, "deepseek_v3" | "deepseek_v3_1") {
        request.chat_template_args =
            Some(serde_json::from_value(serde_json::json!({"thinking": true})).unwrap());
    } else if reasoning_parser == "mistral" {
        request.chat_template_args =
            Some(serde_json::from_value(serde_json::json!({"reasoning_effort": "high"})).unwrap());
    }
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

/// One choice's accumulated stream output, keyed by `choice.index` in
/// `demux_by_choice`. Tool calls stay keyed by tool index so fragments merge.
#[derive(Default)]
struct PerChoice {
    reasoning: String,
    content: String,
    tool_calls: BTreeMap<u32, MergedToolCall>,
}

/// Demux collected output chunks by `choice.index`, merging each choice's
/// reasoning/content/tool-call deltas in arrival order.
fn demux_by_choice(
    output_chunks: &[Annotated<NvCreateChatCompletionStreamResponse>],
) -> BTreeMap<u32, PerChoice> {
    let mut by_choice: BTreeMap<u32, PerChoice> = BTreeMap::new();
    for output in output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            let entry = by_choice.entry(choice.index).or_default();
            if let Some(r) = &choice.delta.reasoning_content {
                entry.reasoning.push_str(r);
            }
            if let Some(c) = &choice.delta.content {
                entry.content.push_str(get_text(c));
            }
            if let Some(tcs) = &choice.delta.tool_calls {
                for tc in tcs {
                    entry.tool_calls.entry(tc.index).or_default().merge_from(tc);
                }
            }
        }
    }
    by_choice
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
            let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
            enable_opt_in_reasoning(&mut request, reasoning_parser);
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

/// Regression (GH-11997): with `n > 1`, each choice must decide bare-JSON vs.
/// reasoning-first INDEPENDENTLY. Choice 0 streams bare guided JSON (bypass →
/// tool call), choice 1 streams `reasoning</think>{json}` in the SAME chunks.
/// The pre-fix stream made ONE global bypass decision from whichever choice
/// emitted content first (choice 0's `[`) and applied it to every choice, so
/// choice 1's reasoning + `</think>` leaked into `content` and its
/// `reasoning_content` was lost. Per-choice state keeps the two isolated, and
/// the different locations (SF vs. Boston) confirm no cross-contamination.
#[tokio::test]
async fn postprocessor_parsing_stream_multi_choice_isolates_guided_bypass_decision() {
    let preprocessor = build_preprocessor(Some("deepseek_r1"), Some("nemotron_nano"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);

    // choice 0: bare guided JSON. choice 1: reasoning, then `</think>`, then JSON.
    let json0 = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let json1 = r#"[{"name":"get_weather","parameters":{"location":"Boston"}}]"#;

    let input_chunks = vec![
        // First content chunk: choice 0 leads with `[` (decides bypass), choice 1
        // leads with reasoning text (must decide NOT to bypass).
        mock_multi_choice_content_chunk(&[(0, "["), (1, "Let me check.")]),
        mock_multi_choice_content_chunk(&[(0, &json0[1..]), (1, "</think>")]),
        mock_multi_choice_content_chunk(&[(1, json1)]),
        mock_multi_choice_final_chunk(&[0, 1]),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    let by_choice = demux_by_choice(&output_chunks);

    let c0 = by_choice.get(&0).expect("choice 0 produced output");
    assert!(
        c0.reasoning.is_empty(),
        "choice 0 (bare JSON) must not produce reasoning_content, got: {:?}",
        c0.reasoning
    );
    let c0_calls: Vec<MergedToolCall> = c0.tool_calls.values().cloned().collect();
    assert_clean_tool_call(
        "choice 0 bare JSON",
        &c0.content,
        &c0_calls,
        "San Francisco",
    );

    let c1 = by_choice.get(&1).expect("choice 1 produced output");
    assert_eq!(
        c1.reasoning, "Let me check.",
        "choice 1 reasoning must be separated, not bypassed by choice 0's decision"
    );
    let c1_calls: Vec<MergedToolCall> = c1.tool_calls.values().cloned().collect();
    assert_clean_tool_call("choice 1 reasoning-first", &c1.content, &c1_calls, "Boston");
}

// ---------------------------------------------------------------------------
// GH-11997 step 3: n>1 synthetic-interleave lane (per-choice state isolation).
//
// Invariant: `demux(parse(interleave(A@0, B@1, ...)))[i] == parse(shape_i)`.
// Each choice's demuxed output must equal what that same single-choice shape
// produces on its own (n=1). No n>1 fixture is authored — the golden is each
// shape's own solo run, computed at test time. Single-choice fixtures are
// mathematically blind to this bug class: with one choice, per-response and
// per-choice state are identical, so only interleaving two divergent shapes in
// one stream reveals a stage that keyed state per response instead of per
// `choice.index`.
//
// Sibling lanes over the frontend-crates v1 jail and v2 stream corpus live in
// ai-dynamo/frontend-crates (`parsers/v1/tests/jail_interleave.rs`,
// `conformance/tests/parity_toolcalling_stream_interleave.rs`); the schedule
// core below is the dynamo-side copy of that lane's origin module.
mod interleave {
    //! Deterministic multi-choice interleave schedules. No RNG: every failure
    //! reproduces byte-exactly. A "shape" is one choice's ordered delta texts.
    //! `interleave` merges k shapes into one ordered list of chunks, each chunk
    //! a `(choice.index, delta)` list, ready for `mock_multi_choice_content_chunk`.

    #[derive(Clone, Copy, Debug)]
    pub enum Schedule {
        /// One delta from each choice per round, in index order: 0,1,0,1,...
        RoundRobin,
        /// Choice `i` starts `i * offset` rounds late (choice 0 streams alone
        /// first), then the rest interleave by global round.
        FirstByteOffset(usize),
        /// Split choice 0's delta in half around the other choices' deltas of
        /// the same round — stresses a marker split across a choice boundary.
        BoundarySplit,
    }

    /// Every schedule the lane exercises — shared by the lossless roundtrip
    /// test and the isolation assertion so the two cannot drift apart.
    pub const ALL_SCHEDULES: [Schedule; 4] = [
        Schedule::RoundRobin,
        Schedule::FirstByteOffset(1),
        Schedule::FirstByteOffset(2),
        Schedule::BoundarySplit,
    ];

    /// Split `s` near the middle on a UTF-8 char boundary. Returns `("", s)`
    /// when the string is too short to split (single char / empty).
    fn split_mid(s: &str) -> (&str, &str) {
        if s.len() < 2 {
            return ("", s);
        }
        let mut mid = s.len() / 2;
        while mid < s.len() && !s.is_char_boundary(mid) {
            mid += 1;
        }
        if mid == s.len() {
            return ("", s);
        }
        s.split_at(mid)
    }

    pub fn interleave(shapes: &[Vec<&str>], schedule: Schedule) -> Vec<Vec<(u32, String)>> {
        let max_len = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut chunks: Vec<Vec<(u32, String)>> = Vec::new();
        match schedule {
            // RoundRobin is FirstByteOffset(0): every choice starts at round 0.
            Schedule::RoundRobin | Schedule::FirstByteOffset(_) => {
                let offset = match schedule {
                    Schedule::FirstByteOffset(o) => o,
                    _ => 0,
                };
                // Emit (round, index, delta) events, then order by (round, index)
                // so each choice's stream starts `index * offset` rounds late.
                let mut events: Vec<(usize, u32, String)> = Vec::new();
                for (i, shape) in shapes.iter().enumerate() {
                    for (j, delta) in shape.iter().enumerate() {
                        events.push((i * offset + j, i as u32, (*delta).to_string()));
                    }
                }
                events.sort_by_key(|(round, idx, _)| (*round, *idx));
                chunks = events
                    .into_iter()
                    .map(|(_, idx, d)| vec![(idx, d)])
                    .collect();
            }
            Schedule::BoundarySplit => {
                for t in 0..max_len {
                    // Choice 0's delta halves bracket the other choices' deltas
                    // of the same round (no halves when choice 0 is exhausted).
                    let halves = shapes.first().and_then(|s| s.get(t)).map(|d| split_mid(d));
                    if let Some((h1, _)) = halves
                        && !h1.is_empty()
                    {
                        chunks.push(vec![(0, h1.to_string())]);
                    }
                    for (i, shape) in shapes.iter().enumerate().skip(1) {
                        if let Some(delta) = shape.get(t) {
                            chunks.push(vec![(i as u32, (*delta).to_string())]);
                        }
                    }
                    if let Some((_, h2)) = halves {
                        chunks.push(vec![(0, h2.to_string())]);
                    }
                }
            }
        }
        chunks
    }

    /// Rewrite the generator's positional `0..k` indices onto arbitrary
    /// `indices`, so a lane can present non-contiguous and unsorted
    /// `choice.index` values. State keyed by vector position, or code assuming
    /// sorted contiguous indices, survives the plain lanes but not this one.
    pub fn remap(chunks: &[Vec<(u32, String)>], indices: &[u32]) -> Vec<Vec<(u32, String)>> {
        chunks
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|(i, d)| (indices[*i as usize], d.clone()))
                    .collect()
            })
            .collect()
    }

    /// Merge each run of adjacent chunks carrying disjoint indices into one
    /// chunk, so a single response carries several choices' deltas — the packed
    /// shape real engines emit. Per-choice delta order is preserved because a
    /// chunk is never merged with one that repeats an index it already holds.
    pub fn pack(chunks: &[Vec<(u32, String)>]) -> Vec<Vec<(u32, String)>> {
        let mut out: Vec<Vec<(u32, String)>> = Vec::new();
        for chunk in chunks {
            let disjoint = out
                .last()
                .is_some_and(|last| chunk.iter().all(|(i, _)| last.iter().all(|(j, _)| j != i)));
            if disjoint {
                out.last_mut()
                    .expect("checked non-empty")
                    .extend(chunk.iter().cloned());
            } else {
                out.push(chunk.clone());
            }
        }
        out
    }

    /// De-interleave: concatenate every delta per `choice.index` in arrival
    /// order. The lossless invariant is on concatenated content per choice.
    pub fn deinterleave(chunks: &[Vec<(u32, String)>]) -> std::collections::BTreeMap<u32, String> {
        let mut map: std::collections::BTreeMap<u32, String> = std::collections::BTreeMap::new();
        for chunk in chunks {
            for (idx, delta) in chunk {
                map.entry(*idx).or_default().push_str(delta);
            }
        }
        map
    }
}

/// Every schedule must be lossless: de-interleaving its output by
/// `choice.index` recovers each shape's concatenated content byte-exactly.
#[test]
fn interleave_schedules_are_lossless() {
    use interleave::{ALL_SCHEDULES, deinterleave, interleave, pack, remap};
    let shapes = vec![
        vec!["Let me ch", "eck.", "</think>", "answer0"],
        vec!["Short", " reasoning</think>", "answer1"],
        vec!["one-shot choice2"],
    ];
    let expected: BTreeMap<u32, String> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u32, s.concat()))
        .collect();
    for schedule in ALL_SCHEDULES {
        let recovered = deinterleave(&interleave(&shapes, schedule));
        assert_eq!(
            recovered, expected,
            "schedule {schedule:?} lost or reordered per-choice content"
        );
    }

    // remap + pack must be lossless too. A `pack` that swallowed a delta would
    // otherwise weaken the isolation lanes built on top of it.
    let indices = [u32::MAX, 7, 2];
    let expected_mapped: BTreeMap<u32, String> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| (indices[i], s.concat()))
        .collect();
    for schedule in ALL_SCHEDULES {
        let chunks = pack(&remap(&interleave(&shapes, schedule), &indices));
        assert_eq!(
            deinterleave(&chunks),
            expected_mapped,
            "schedule {schedule:?} lost content under remap+pack"
        );
        assert!(
            chunks.iter().any(|c| c.len() > 1),
            "schedule {schedule:?}: pack produced no multi-choice chunk, so the \
             packed lane would not differ from the unpacked one"
        );
    }
}

/// Comparable projection of one choice's demuxed output: reasoning, content,
/// and assembled tool calls (name + arguments per tool index). This lane
/// intentionally does not assert finish reasons.
#[derive(Default, PartialEq, Eq, Debug)]
struct ChoiceOutput {
    reasoning: String,
    content: String,
    tool_calls: Vec<(Option<String>, String)>,
}

/// Drive `postprocessor_parsing_stream` over pre-built content chunks (one
/// `(index, delta)` list per chunk) plus a terminal finish for every index,
/// then demux the output by `choice.index` into a `ChoiceOutput` per choice.
async fn run_interleaved(
    preprocessor: &Arc<OpenAIPreprocessor>,
    request: &NvCreateChatCompletionRequest,
    chunks: &[Vec<(u32, String)>],
) -> BTreeMap<u32, ChoiceOutput> {
    let mut indices: Vec<u32> = chunks
        .iter()
        .flat_map(|c| c.iter().map(|(i, _)| *i))
        .collect();
    indices.sort_unstable();
    indices.dedup();

    let mut input: Vec<NvCreateChatCompletionStreamResponse> = chunks
        .iter()
        .map(|chunk| {
            let refs: Vec<(u32, &str)> = chunk.iter().map(|(i, s)| (*i, s.as_str())).collect();
            mock_multi_choice_content_chunk(&refs)
        })
        .collect();
    input.push(mock_multi_choice_final_chunk(&indices));

    let input_stream = stream::iter(input.into_iter().map(Annotated::from_data));
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> =
        output_stream.collect().await;

    demux_by_choice(&output_chunks)
        .into_iter()
        .map(|(idx, acc)| {
            (
                idx,
                ChoiceOutput {
                    reasoning: acc.reasoning,
                    content: acc.content,
                    tool_calls: acc
                        .tool_calls
                        .into_values()
                        .map(|tc| (tc.name, tc.arguments))
                        .collect(),
                },
            )
        })
        .collect()
}

/// Run one shape solo (single choice at index 0) and return its `ChoiceOutput`.
async fn solo_output(
    preprocessor: &Arc<OpenAIPreprocessor>,
    request: &NvCreateChatCompletionRequest,
    shape: &[&str],
) -> ChoiceOutput {
    let chunks = interleave::interleave(&[shape.to_vec()], interleave::Schedule::RoundRobin);
    let mut out = run_interleaved(preprocessor, request, &chunks).await;
    out.remove(&0).unwrap_or_default()
}

/// Core invariant assertion: for `shapes` interleaved under `schedule`, each
/// choice's demuxed output equals that shape's solo (n=1) output.
async fn assert_interleave_isolated(
    label: &str,
    preprocessor: &Arc<OpenAIPreprocessor>,
    request: &NvCreateChatCompletionRequest,
    shapes: &[Vec<&str>],
) {
    let indices: Vec<u32> = (0..shapes.len() as u32).collect();
    assert_interleave_isolated_mapped(label, preprocessor, request, shapes, &indices, false).await;
}

/// Same invariant with an explicit `choice.index` per shape and optional
/// chunk packing. `assert_interleave_isolated` is the contiguous, one-choice-
/// per-chunk case; both route through here so the two cannot drift apart.
async fn assert_interleave_isolated_mapped(
    label: &str,
    preprocessor: &Arc<OpenAIPreprocessor>,
    request: &NvCreateChatCompletionRequest,
    shapes: &[Vec<&str>],
    indices: &[u32],
    pack_chunks: bool,
) {
    let mut goldens: Vec<ChoiceOutput> = Vec::new();
    for shape in shapes {
        goldens.push(solo_output(preprocessor, request, shape).await);
    }
    // Sanity: the shapes must actually diverge, or the lane proves nothing.
    if shapes.len() >= 2 {
        assert!(
            goldens.iter().any(|g| *g != goldens[0]),
            "{label}: shapes do not diverge; interleave lane would prove nothing"
        );
    }
    for schedule in interleave::ALL_SCHEDULES {
        let mut chunks = interleave::remap(&interleave::interleave(shapes, schedule), indices);
        if pack_chunks {
            chunks = interleave::pack(&chunks);
        }
        let demuxed = run_interleaved(preprocessor, request, &chunks).await;
        for (i, golden) in goldens.iter().enumerate() {
            let idx = indices[i];
            let got = demuxed.get(&idx).unwrap_or_else(|| {
                panic!("{label} [{schedule:?}]: choice index {idx} produced no output")
            });
            assert_eq!(
                got, golden,
                "{label} [{schedule:?}]: choice index {idx} demuxed output != its solo (n=1) output \
                 — cross-choice state leak"
            );
        }
    }
}

/// Guided-JSON bypass + tool-jail stage: a choice that leads with bare guided
/// JSON must not freeze the bypass decision for a reasoning-first choice.
/// Generalizes `..._isolates_guided_bypass_decision` across schedules/pairs.
#[tokio::test]
async fn postprocessor_parsing_stream_interleave_isolates_tool_bypass_stage() {
    let preprocessor = build_preprocessor(Some("deepseek_r1"), Some("nemotron_nano"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);

    let json_sf = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let json_boston = r#"[{"name":"get_weather","parameters":{"location":"Boston"}}]"#;
    let json_paris = r#"[{"name":"get_weather","parameters":{"location":"Paris"}}]"#;
    let json_denver = r#"[{"name":"get_weather","parameters":{"location":"Denver"}}]"#;

    let bare_json = vec!["[", &json_sf[1..]];
    let reasoning_first = vec!["Let me check.", "</think>", json_boston];
    let reasoning_first2 = vec!["Thinking hard.", "</think>", json_paris];
    // reasoning text and the `</think>` close marker split across deltas.
    let tag_split = vec!["Let me ch", "eck.", "</th", "ink>", json_denver];
    // Whitespace-only leading deltas leave this choice's bypass decision on the
    // undecided (`None`) arm until its first non-whitespace delta. The pre-fix
    // code read the decision from whichever choice spoke first anywhere in the
    // chunk, so an undecided choice was exactly where a foreign verdict landed.
    let ws_then_bare_json = vec!["  ", "[", &json_paris[1..]];
    let ws_then_reasoning = vec![" ", "\t", "Let me check.", "</think>", json_denver];

    // k=2 pairs: the known killer, two reasoning choices, and a split-marker pair.
    assert_interleave_isolated(
        "bare-JSON x reasoning-first",
        &preprocessor,
        &request,
        &[bare_json.clone(), reasoning_first.clone()],
    )
    .await;
    assert_interleave_isolated(
        "reasoning x reasoning (distinct)",
        &preprocessor,
        &request,
        &[reasoning_first.clone(), reasoning_first2.clone()],
    )
    .await;
    assert_interleave_isolated(
        "bare-JSON x split-marker reasoning",
        &preprocessor,
        &request,
        &[bare_json.clone(), tag_split.clone()],
    )
    .await;
    // Whitespace-lead shapes exercise the undecided (`None`) bypass arm, which
    // the schedules above never reach because every shape's first delta already
    // decides.
    assert_interleave_isolated(
        "whitespace-lead bare-JSON x reasoning-first",
        &preprocessor,
        &request,
        &[ws_then_bare_json.clone(), reasoning_first2.clone()],
    )
    .await;
    assert_interleave_isolated(
        "whitespace-lead reasoning x whitespace-lead bare-JSON",
        &preprocessor,
        &request,
        &[ws_then_reasoning, ws_then_bare_json],
    )
    .await;
    // k=3: prove the per-choice state map generalizes past two choices.
    assert_interleave_isolated(
        "k=3 bare-JSON x reasoning x reasoning",
        &preprocessor,
        &request,
        &[bare_json, reasoning_first, reasoning_first2],
    )
    .await;
}

/// Non-contiguous, unsorted `choice.index` values packed several-per-chunk.
/// Every other lane emits contiguous `0..k`, one choice per chunk, so state
/// keyed by vector position — or code assuming sorted, contiguous, one-per-
/// chunk indices — passes those and fails only here.
#[tokio::test]
async fn postprocessor_parsing_stream_interleave_isolates_noncontiguous_packed_indices() {
    let preprocessor = build_preprocessor(Some("deepseek_r1"), Some("nemotron_nano"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);

    let json_sf = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let json_boston = r#"[{"name":"get_weather","parameters":{"location":"Boston"}}]"#;

    let ws_lead = vec!["  ", "Delayed.", "</think>", json_boston];
    let bare_json = vec!["[", &json_sf[1..]];
    let reasoning_first = vec!["Let me check.", "</think>", json_boston];

    // `u32::MAX` exercises the key's full range; running both descending and
    // ascending proves nothing depends on the indices' relative order.
    for indices in [vec![u32::MAX, 7, 2], vec![2, 7, u32::MAX]] {
        assert_interleave_isolated_mapped(
            &format!("non-contiguous {indices:?}, packed chunks"),
            &preprocessor,
            &request,
            &[ws_lead.clone(), bare_json.clone(), reasoning_first.clone()],
            &indices,
            true,
        )
        .await;
    }
}

/// Pure reasoning-split / `<think>`-strip stage (no tools): each choice's
/// `reasoning_content` vs `content` split must be decided per `choice.index`.
#[tokio::test]
async fn postprocessor_parsing_stream_interleave_isolates_reasoning_split_stage() {
    let preprocessor = build_preprocessor(Some("deepseek_r1"), None);
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Explain your answer."}],
        "stream": true,
        "temperature": 0.0
    }))
    .unwrap();

    // deepseek_r1 is force-reasoning: output starts inside reasoning, `</think>`
    // closes it. Divergent shapes stress per-choice reasoning state.
    let reason_then_answer_a = vec!["I should compute.", "</think>", "The result is 42."];
    let reason_then_answer_b = vec!["A different path.", "</think>", "The result is 7."];
    let reason_only = vec!["Still thinking, never closes."];
    // `</think>` split across deltas stresses per-choice strip buffering.
    let tag_split = vec!["Split rea", "soning.", "</th", "ink>", "Final answer."];
    // Whitespace-only lead: nothing for the parser to act on until delta 2.
    let ws_then_reason = vec!["  ", "Delayed start.", "</think>", "Answer C."];
    // Malformed: a second `</think>` after reasoning already closed. It must be
    // treated as ordinary content, and only for the choice that emitted it.
    let double_close = vec!["Reason.", "</think>", "Answer.", "</think>", " tail."];

    assert_interleave_isolated(
        "reason+answer A x B",
        &preprocessor,
        &request,
        &[reason_then_answer_a.clone(), reason_then_answer_b.clone()],
    )
    .await;
    assert_interleave_isolated(
        "reason+answer x reason-only",
        &preprocessor,
        &request,
        &[reason_then_answer_a.clone(), reason_only.clone()],
    )
    .await;
    assert_interleave_isolated(
        "reason+answer x split-marker",
        &preprocessor,
        &request,
        &[reason_then_answer_a.clone(), tag_split],
    )
    .await;
    // Whitespace-lead and malformed-marker shapes: the reasoning state machine
    // must stay per-choice on the paths the schedules above do not enumerate.
    assert_interleave_isolated(
        "whitespace-lead reasoning x reason+answer",
        &preprocessor,
        &request,
        &[ws_then_reason, reason_then_answer_b.clone()],
    )
    .await;
    assert_interleave_isolated(
        "double-close marker x reason+answer",
        &preprocessor,
        &request,
        &[double_close, reason_then_answer_a.clone()],
    )
    .await;
    // k=3 across the reasoning stage.
    assert_interleave_isolated(
        "k=3 reasoning split",
        &preprocessor,
        &request,
        &[reason_then_answer_a, reason_then_answer_b, reason_only],
    )
    .await;
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

/// DeepSeek V3 aliases use a force parser, but disabling thinking must leave
/// required-tool JSON available to the tool jail.
#[tokio::test]
async fn tool_choice_deepseek_v3_required_thinking_disabled_keeps_bare_json() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;

    for reasoning_parser in ["deepseek_v3", "deepseek_v3_1", "deepseek_v3_2"] {
        let preprocessor = build_preprocessor(Some(reasoning_parser), Some("nemotron_nano"));
        let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
        request.chat_template_args =
            Some(serde_json::from_value(serde_json::json!({"thinking": false})).unwrap());
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
            finish_reasons,
        } = drain_stream(output_stream).await;

        let case = format!("{reasoning_parser} required + thinking disabled + bare JSON");
        assert!(
            reasoning.is_empty(),
            "{case}: reasoning_content must be empty, got: {reasoning:?}"
        );
        assert_clean_tool_call(&case, &content, &tool_calls, "San Francisco");
        assert!(
            finish_reasons.contains(&FinishReason::ToolCalls),
            "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
        );
    }
}

/// Required guided JSON must follow each force parser's close marker,
/// including markers split across stream chunks.
#[tokio::test]
async fn tool_choice_force_reasoning_required_keeps_reasoning_before_guided_json() {
    for &(reasoning_parser, close_marker) in REASONING_BEFORE_GUIDED_JSON_PARSERS {
        for prompt_injected in [false, true] {
            let preprocessor = build_preprocessor(Some(reasoning_parser), Some("nemotron_nano"));
            let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
            enable_opt_in_reasoning(&mut request, reasoning_parser);
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

/// Mistral's bracket-prefixed opener must not be mistaken for a bare JSON array,
/// even when the opener is split across stream chunks.
#[tokio::test]
async fn tool_choice_mistral_required_recognizes_split_reasoning_start() {
    let preprocessor = build_preprocessor(Some("mistral"), Some("nemotron_nano"));
    let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    enable_opt_in_reasoning(&mut request, "mistral");
    let input_stream = stream::iter(
        vec![
            mock_content_chunk(" \n"),
            mock_content_chunk("["),
            mock_content_chunk("TH"),
            mock_content_chunk("INK]Let me check.[/TH"),
            mock_content_chunk(
                r#"INK][{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#,
            ),
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
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "Mistral required + whitespace + split [THINK] opener + guided JSON";
    assert_eq!(reasoning, "Let me check.", "{case}: wrong reasoning");
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(
        finish_reasons.contains(&FinishReason::ToolCalls),
        "{case}: expected ToolCalls finish_reason, got: {finish_reasons:?}"
    );
}

/// Named guided decoding emits only the selected function's parameter object.
/// Reasoning before that object must be separated without preventing the named
/// immediate jail from constructing the tool call.
#[tokio::test]
async fn tool_choice_force_reasoning_named_keeps_reasoning_before_guided_params() {
    for &(reasoning_parser, close_marker) in REASONING_BEFORE_GUIDED_JSON_PARSERS {
        let preprocessor = build_preprocessor(Some(reasoning_parser), Some("nemotron_nano"));
        let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Named(
            "get_weather".to_string().into(),
        ));
        enable_opt_in_reasoning(&mut request, reasoning_parser);
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

/// `enable_thinking=true` still preserves genuine reasoning followed by a
/// response_format JSON payload in their respective OpenAI fields.
#[tokio::test]
async fn response_format_qwen3_prompt_injected_reasoning_then_json_preserves_channels() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let stream_text = format!("France is a country in Europe.</think>{json}");
    let preprocessor = build_preprocessor(Some("qwen3"), None);
    let request = streaming_json_schema_request(true);
    let input_stream = stream::iter(
        vec![mock_content_chunk(&stream_text), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert_eq!(reasoning.trim(), "France is a country in Europe.");
    assert_eq!(content, json);
}

/// If SGLang emits response_format JSON immediately after a prompt-injected
/// Qwen `<think>`, Dynamo must recover the structured answer as assistant
/// content instead of classifying the JSON as reasoning.
#[tokio::test]
async fn response_format_qwen3_prompt_injected_bare_json_stays_content() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let preprocessor = build_preprocessor(Some("qwen3"), None);
    let request = streaming_json_schema_request(true);
    let input_stream = stream::iter(
        vec![mock_content_chunk(json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert!(
        reasoning.is_empty(),
        "response_format JSON must not be reasoning_content, got: {reasoning:?}"
    );
    assert_eq!(content, json);
}

/// With thinking disabled, response_format JSON is ordinary assistant content.
#[tokio::test]
async fn response_format_qwen3_no_thinking_json_stays_content() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let preprocessor = build_preprocessor(Some("qwen3"), None);
    let request = streaming_json_schema_request(false);
    let input_stream = stream::iter(
        vec![mock_content_chunk(json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert!(reasoning.is_empty());
    assert_eq!(content, json);
}

/// Gemma4 structured output without visible reasoning markers should remain
/// assistant content even when the reasoning parser is configured.
#[tokio::test]
async fn response_format_gemma4_bare_json_stays_content() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let preprocessor = build_preprocessor(Some("gemma4"), None);
    let request = streaming_json_schema_request(true);
    let input_stream = stream::iter(
        vec![mock_content_chunk(json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert!(
        reasoning.is_empty(),
        "response_format JSON must not be reasoning_content, got: {reasoning:?}"
    );
    assert_eq!(content, json);
}

/// MiniMax append-think is a force-reasoning parser that is not yet proven to
/// preserve native reasoning before guided JSON. Structured bare JSON should
/// therefore use the legacy guided-output bypass and remain assistant content.
#[tokio::test]
async fn response_format_minimax_append_think_bare_json_stays_content() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let preprocessor = build_preprocessor(Some("minimax_append_think"), Some("minimax_m2"));
    let request = streaming_json_schema_request(true);
    let input_stream = stream::iter(
        vec![mock_content_chunk(json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert!(
        reasoning.is_empty(),
        "response_format JSON must not be reasoning_content, got: {reasoning:?}"
    );
    assert_eq!(content, json);
}

/// GPT-OSS/Harmony guided structured output may be emitted as bare JSON from
/// token 0. The reasoning parser should preserve that JSON as assistant
/// content instead of dropping it while waiting for Harmony channel markers.
#[tokio::test]
async fn response_format_gpt_oss_bare_json_stays_content() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let preprocessor = build_preprocessor(Some("gpt_oss"), None);
    let request = streaming_json_schema_request(true);
    let input_stream = stream::iter(
        vec![mock_content_chunk(json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert!(
        reasoning.is_empty(),
        "response_format JSON must not be reasoning_content, got: {reasoning:?}"
    );
    assert_eq!(content, json);
}

/// `json_object` response_format is also translated to guided JSON. It should
/// follow the same GPT-OSS structured-output bypass as `json_schema`.
#[tokio::test]
async fn response_format_gpt_oss_json_object_bare_json_stays_content() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let preprocessor = build_preprocessor(Some("gpt_oss"), None);
    let request = streaming_json_object_request(true);
    let input_stream = stream::iter(
        vec![mock_content_chunk(json), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert!(
        reasoning.is_empty(),
        "response_format JSON must not be reasoning_content, got: {reasoning:?}"
    );
    assert_eq!(content, json);
}

/// If GPT-OSS emits native Harmony reasoning before the structured payload,
/// shape detection must fall back to the parser path and preserve channels.
#[tokio::test]
async fn response_format_gpt_oss_reasoning_then_json_preserves_channels() {
    let json = r#"{"country":"France","capital":"Paris"}"#;
    let stream_text = format!(
        "<|channel|>analysis<|message|>Need answer as JSON.<|end|><|start|>assistant<|channel|>final<|message|>{json}"
    );
    let preprocessor = build_preprocessor(Some("gpt_oss"), None);
    let request = streaming_json_schema_request(true);
    let input_stream = stream::iter(
        vec![mock_content_chunk(&stream_text), mock_final_chunk()]
            .into_iter()
            .map(Annotated::from_data),
    );
    let output_stream = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build");
    let DrainOutput {
        reasoning, content, ..
    } = drain_stream(output_stream).await;

    assert_eq!(reasoning, "Need answer as JSON.");
    assert_eq!(content, json);
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

#[tokio::test]
async fn tool_choice_minimax_m2_required_keeps_reasoning_before_tool_xml() {
    let preprocessor = build_preprocessor(Some("minimax_m2"), Some("minimax_m2"));
    let request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    let tool_call = "<minimax:tool_call>\
<invoke name=\"get_weather\"><parameter name=\"location\">San Francisco</parameter></invoke>\
</minimax:tool_call>";
    let input_stream = stream::iter(
        vec![
            mock_content_chunk("I should call weather."),
            mock_content_chunk("</think>"),
            mock_content_chunk(tool_call),
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

    let case = "MiniMax M2 required + reasoning boundary + XML tool call";
    assert_eq!(reasoning, "I should call weather.");
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(finish_reasons.contains(&FinishReason::ToolCalls));
}

#[tokio::test]
async fn tool_choice_minimax_m2_required_bare_json_bypasses_reasoning() {
    let bare_json = r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#;
    let preprocessor = build_preprocessor(Some("minimax_m2"), Some("minimax_m2"));
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

    let case = "MiniMax M2 required + bare guided JSON";
    assert!(reasoning.is_empty());
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(finish_reasons.contains(&FinishReason::ToolCalls));
}

#[tokio::test]
async fn tool_choice_minimax_m2_required_thinking_disabled_keeps_tool_xml() {
    let preprocessor = build_preprocessor(Some("minimax_m2"), Some("minimax_m2"));
    let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Required);
    request.chat_template_args =
        Some(serde_json::from_value(serde_json::json!({"thinking": false})).unwrap());
    let tool_call = "<minimax:tool_call>\
<invoke name=\"get_weather\"><parameter name=\"location\">San Francisco</parameter></invoke>\
</minimax:tool_call>";
    let input_stream = stream::iter(
        vec![mock_content_chunk(tool_call), mock_final_chunk()]
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
        finish_reasons,
    } = drain_stream(output_stream).await;

    let case = "MiniMax M2 required + thinking=false + XML tool call";
    assert!(reasoning.is_empty());
    assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    assert!(finish_reasons.contains(&FinishReason::ToolCalls));
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

    let path = if dynamo_runtime::config::env_is_truthy("DYN_ENABLE_EXPERIMENTAL_PARSERS_V2") {
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

/// Recovered text must ship on the terminal chunk, not after it.
///
/// The held reasoning is only known to be the answer once the stream finishes,
/// but emitting it as an extra chunk would place `content` after
/// `finish_reason=stop` and after the trailing usage chunk. A streaming client
/// that stops reading at the terminal chunk would then see empty content — the
/// exact failure `force_nonempty_content` exists to prevent. So the drain
/// happens on the chunk that carries `finish_reason`, and usage stays last.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_recovers_on_terminal_chunk() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    let input_chunks = vec![
        mock_content_chunk("<think>answer"),
        mock_final_chunk(),
        mock_usage_only_chunk(),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build")
        .collect()
        .await;

    let mut finish_index = None;
    let mut content_indices = Vec::new();
    let mut usage_indices = Vec::new();
    let mut content = String::new();
    for (i, output) in output_chunks.iter().enumerate() {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        if data.inner.usage.is_some() {
            usage_indices.push(i);
        }
        for choice in &data.inner.choices {
            if choice.finish_reason.is_some() {
                finish_index = Some(i);
            }
            if let Some(c) = &choice.delta.content {
                content_indices.push(i);
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(content, "answer", "the held text must still be recovered");
    let finish_index = finish_index.expect("terminal chunk must survive");
    assert!(
        content_indices.iter().all(|i| *i <= finish_index),
        "content must not be emitted after finish_reason: content at {content_indices:?}, finish at {finish_index}"
    );
    assert!(
        usage_indices.iter().all(|i| *i > finish_index),
        "the usage chunk must stay last"
    );
}

/// Once an answer has streamed as content, the non-empty-content promise is met,
/// so a later truncated reasoning block must stay in `reasoning_content` rather
/// than being appended to the answer. Input `<think>r1</think>A<think>r2</th`
/// ends mid-`</think>` of a second reasoning block: `content` is exactly the
/// answer and the dangling `</th` is reported as reasoning.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_second_block_stays_reasoning() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    let input_chunks = vec![mock_content_chunk("<think>r1</think>A<think>r2</th")];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build")
        .collect()
        .await;

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
        content, "A",
        "a truncated later reasoning block must not leak into the answer"
    );
    assert_eq!(
        reasoning, "r1r2</th",
        "the dangling reasoning bytes stay in reasoning_content"
    );
}

/// Pins the streaming granularity cost of the deferred path, which the
/// concatenating assertions elsewhere cannot see.
///
/// With no `</think>` in the turn, the held text is the answer, so it can only
/// be released once the stream is known to be over — it arrives as one delta on
/// the terminal chunk instead of token by token. That is a real regression in
/// granularity versus `main` for this case, accepted because these parsers emit
/// reasoning with no opening `<think>`: until `</think>` arrives the bytes are
/// equally consistent with reasoning and with a plain answer, and emitting
/// eagerly is what leaked reasoning into `content`. If a future change restores
/// incremental delivery here, this test should fail and force that tradeoff to
/// be re-decided rather than drifting silently.
///
/// The contrast case is deliberate: once `</think>` has arrived the ambiguity is
/// gone and the answer streams one delta per chunk again.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_delta_shape() {
    async fn content_delta_count(chunks: Vec<NvCreateChatCompletionStreamResponse>) -> usize {
        let preprocessor = build_preprocessor(Some("nemotron_v3"), None);
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_str(REQUEST_JSON).unwrap();
        request.chat_template_args = Some(
            serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
        );
        let input_stream = stream::iter(chunks.into_iter().map(Annotated::from_data));
        let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, false, false)
            .expect("postprocessor_parsing_stream should build")
            .collect()
            .await;
        output_chunks
            .iter()
            .filter_map(|o| o.data.as_ref())
            .flat_map(|d| d.inner.choices.iter())
            .filter(|c| c.delta.content.is_some())
            .count()
    }

    // No `</think>`: three source chunks of answer text collapse into one delta.
    let no_close = vec![
        mock_content_chunk("Hel"),
        mock_content_chunk("lo"),
        mock_content_chunk("!"),
        mock_final_chunk(),
    ];
    assert_eq!(
        content_delta_count(no_close).await,
        1,
        "without </think> the whole answer must arrive as a single delta"
    );

    // Same answer after a closing marker: streams one delta per source chunk.
    let with_close = vec![
        mock_content_chunk("r</think>Hel"),
        mock_content_chunk("lo"),
        mock_content_chunk("!"),
        mock_final_chunk(),
    ];
    assert_eq!(
        content_delta_count(with_close).await,
        3,
        "after </think> the answer must stream incrementally again"
    );
}

/// The end-of-stream fallback clones the last content-bearing chunk as its
/// envelope, and that chunk's tokens were already counted. `metrics.rs` sums
/// `chunk_tokens` across chunks carrying `llm_metrics` and samples one ITL point
/// per such chunk, so carrying the annotation over would double-count the tokens
/// and add a latency sample for a chunk that generated nothing. The recovery
/// chunk re-channels bytes the parser was already holding, so it must report no
/// metrics of its own.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_flush_drops_llm_metrics() {
    use dynamo_llm::protocols::common::metrics::LLMMetricAnnotation;

    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    // No terminal chunk, so the stream takes the EOF fallback that clones the
    // envelope. The source chunk carries metrics, as it would in production.
    let mut source = mock_content_chunk("<thi");
    source.llm_metrics = Some(LLMMetricAnnotation {
        input_tokens: 7,
        output_tokens: 3,
        chunk_tokens: 3,
        ..Default::default()
    });

    let input_stream = stream::iter(vec![Annotated::from_data(source)]);
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build")
        .collect()
        .await;

    let metric_bearing: Vec<usize> = output_chunks
        .iter()
        .enumerate()
        .filter(|(_, o)| {
            o.data
                .as_ref()
                .is_some_and(|d| d.llm_metrics.as_ref().is_some_and(|m| m.chunk_tokens > 0))
        })
        .map(|(i, _)| i)
        .collect();

    assert_eq!(
        metric_bearing.len(),
        1,
        "exactly one chunk may report chunk_tokens; the recovery chunk must not \
         re-report the envelope's, got chunks {metric_bearing:?}"
    );

    let recovered: String = output_chunks
        .iter()
        .filter_map(|o| o.data.as_ref())
        .flat_map(|d| d.inner.choices.iter())
        .filter_map(|c| c.delta.content.as_ref().map(get_text))
        .collect();
    assert_eq!(recovered, "<thi", "the recovery itself must still happen");
}

/// A backend that keeps streaming after a choice's `finish_reason` is out of
/// protocol, but those bytes still reach the parser, so they still need a
/// destination. The terminal drain marks the choice drained; a later content
/// chunk reopens it, so `r2` is emitted rather than stranded.
///
/// What is NOT recovered is whatever the parser is still buffering internally at
/// that point — here the dangling `</th`. Recovering it would mean calling
/// `finish_reasoning_stream` a second time on the same parser, and the trait does
/// not promise that is idempotent, so a parser that re-emits on a second
/// finalize would duplicate text. Losing it is the better trade twice over: the
/// stream is already out of protocol, and the bytes in question are a truncated
/// marker fragment, which should never reach `content` anyway.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_content_after_finish_survives() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    let input_chunks = vec![
        mock_content_chunk("<think>r1</think>A"),
        mock_final_chunk(),
        mock_content_chunk("<think>r2</th"),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build")
        .collect()
        .await;

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

    assert_eq!(content, "A", "the answer is unchanged by the stray chunk");
    assert_eq!(
        reasoning, "r1r2",
        "post-finish content is emitted, but the parser's dangling marker fragment is not \
         re-finalized into the stream"
    );
}

/// Streaming and non-streaming must agree on a turn whose only answer text is
/// whitespace. The aggregator counts whitespace-only content as empty (matching
/// vLLM's `not final_content.strip()`) and replaces it with the reasoning, so the
/// streaming path holds whitespace back instead of treating it as the answer.
/// Releasing it would settle the turn early and leave streaming with
/// content="\n" and reasoning kept, disagreeing with non-streaming on the same
/// input.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_whitespace_answer_parity() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    let input_chunks = vec![
        mock_content_chunk("<think>r1</think>\n"),
        mock_final_chunk(),
    ];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, false, false)
        .expect("postprocessor_parsing_stream should build")
        .collect()
        .await;

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
        content, "r1",
        "a whitespace-only answer is empty, so the reasoning becomes the content"
    );
    assert_eq!(
        reasoning, "",
        "the moved text must not also be reported as reasoning"
    );
}

/// The deferred drain attaches a choice's whole buffered payload to the chunk
/// carrying `finish_reason`, which is a shape the downstream tool-call jail never
/// saw before: every other test feeds content and the terminal chunk separately.
/// If the jail finalized on `finish_reason` before consuming that chunk's own
/// content, tool markup delivered this way would be dropped or leaked as raw text.
///
/// It does not. Both orderings extract the call cleanly with nothing leaking into
/// `content`. The no-`</think>` case is strictly better than without the flag:
/// there the force-reasoning parser swallows the markup as `reasoning_content` and
/// no tool call is produced at all, whereas the drain routes it to the jail.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_tool_call_survives_terminal_drain()
{
    let xml = "<tool_call>\n<function=get_weather>\n<parameter=location>\nSan Francisco\n</parameter>\n</function>\n</tool_call>";

    for (case, payload) in [
        // Reasoning closes first, so the jail sees the call mid-stream as usual.
        ("reasoning then tool call", format!("thinking</think>{xml}")),
        // No `</think>`: the call is held and drained onto the terminal chunk.
        ("tool call with no </think>", xml.to_string()),
    ] {
        let preprocessor = build_preprocessor(Some("nemotron_v3"), Some("qwen3_coder"));
        let mut request = streaming_tool_request(ChatCompletionToolChoiceOption::Auto);
        request.chat_template_args = Some(
            serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
        );

        let input_stream = stream::iter(
            vec![mock_content_chunk(&payload), mock_final_chunk()]
                .into_iter()
                .map(Annotated::from_data),
        );
        let output_stream = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, true, false)
            .expect("postprocessor_parsing_stream should build");
        let DrainOutput {
            content,
            tool_calls,
            ..
        } = drain_stream(output_stream).await;

        assert_clean_tool_call(case, &content, &tool_calls, "San Francisco");
    }
}

/// The streaming and non-streaming paths must produce the same `content` and
/// `reasoning_content` for the same model output. They reach it by different
/// means — the aggregator moves reasoning into empty content, while the stream
/// holds reasoning until it knows whether an answer follows — so nothing but a
/// direct comparison proves they agree.
///
/// This is also why the streaming side buffers rather than emitting reasoning
/// live and re-emitting it as content at the end: that alternative would leave a
/// reasoning-only turn with the same text in BOTH fields on the streaming path
/// and in only `content` on the non-streaming path, which is a different observable
/// result for identical model output.
#[tokio::test]
async fn postprocessor_parsing_stream_nemotron_v3_force_nonempty_stream_matches_aggregated() {
    for (case, payload) in [
        ("reasoning-only", "<think>Let me greet them."),
        (
            "reasoning+answer",
            "<think>Let me greet them.</think>Hello!",
        ),
        ("plain, no <think>", "Hello!"),
    ] {
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_str(REQUEST_JSON).unwrap();
        request.chat_template_args = Some(
            serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
        );

        // Streaming: accumulate the deltas a client would receive.
        let preprocessor = build_preprocessor(Some("nemotron_v3"), None);
        let input_stream = stream::iter(
            vec![mock_content_chunk(payload), mock_final_chunk()]
                .into_iter()
                .map(Annotated::from_data),
        );
        let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, false, false)
            .expect("postprocessor_parsing_stream should build")
            .collect()
            .await;
        let mut streamed_content = String::new();
        let mut streamed_reasoning = String::new();
        for output in &output_chunks {
            let Some(data) = output.data.as_ref() else {
                continue;
            };
            for choice in &data.inner.choices {
                if let Some(c) = &choice.delta.content {
                    streamed_content.push_str(get_text(c));
                }
                if let Some(r) = &choice.delta.reasoning_content {
                    streamed_reasoning.push_str(r);
                }
            }
        }

        // Non-streaming: the same stream aggregated the way the chat handler does.
        let preprocessor = build_preprocessor(Some("nemotron_v3"), None);
        let input_stream = stream::iter(
            vec![mock_content_chunk(payload), mock_final_chunk()]
                .into_iter()
                .map(Annotated::from_data),
        );
        let response = NvCreateChatCompletionResponse::from_annotated_stream(
            preprocessor
                .postprocessor_parsing_stream(input_stream, &request, false, false)
                .expect("postprocessor_parsing_stream should build"),
            ParsingOptions::default().with_move_reasoning_to_content_when_empty(true),
        )
        .await
        .expect("aggregate");
        let message = &response.inner.choices[0].message;
        let aggregated_content = message
            .content
            .as_ref()
            .map(get_text)
            .unwrap_or_default()
            .to_string();
        let aggregated_reasoning = message.reasoning_content.clone().unwrap_or_default();

        assert_eq!(
            streamed_content, aggregated_content,
            "{case}: content must match across paths"
        );
        assert_eq!(
            streamed_reasoning, aggregated_reasoning,
            "{case}: reasoning_content must match across paths"
        );
    }
}

/// `force_nonempty_content` is a request-level contract, not a model feature, so
/// it is honored after parsing for ANY model rather than being keyed on a
/// parser allow-list. `qwen3` is neither a Nemotron parser nor a force-reasoning
/// one — before this was made generic the flag was ignored for it entirely, and
/// a reasoning-only turn returned empty `content`.
///
/// The flag must also not over-reach: when an answer WAS generated, reasoning
/// stays in `reasoning_content`.
#[tokio::test]
async fn postprocessor_parsing_stream_force_nonempty_applies_to_any_model() {
    async fn run(force: bool, payload: &str) -> (String, String) {
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_str(REQUEST_JSON).unwrap();
        request.chat_template_args = Some(
            serde_json::from_value(serde_json::json!({ "force_nonempty_content": force })).unwrap(),
        );
        let preprocessor = build_preprocessor(Some("qwen3"), None);
        let input_stream = stream::iter(
            vec![mock_content_chunk(payload), mock_final_chunk()]
                .into_iter()
                .map(Annotated::from_data),
        );
        let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
            .postprocessor_parsing_stream(input_stream, &request, false, false)
            .expect("postprocessor_parsing_stream should build")
            .collect()
            .await;
        let (mut content, mut reasoning) = (String::new(), String::new());
        for output in &output_chunks {
            let Some(data) = output.data.as_ref() else {
                continue;
            };
            for choice in &data.inner.choices {
                if let Some(c) = &choice.delta.content {
                    content.push_str(get_text(c));
                }
                if let Some(r) = &choice.delta.reasoning_content {
                    reasoning.push_str(r);
                }
            }
        }
        (content, reasoning)
    }

    // Reasoning-only turn: with the flag the reasoning becomes the answer.
    assert_eq!(
        run(true, "<think>Hm...</think>").await,
        ("Hm...".to_string(), String::new()),
        "a non-Nemotron model must honor force_nonempty_content"
    );
    // Without the flag the same turn is untouched.
    assert_eq!(
        run(false, "<think>Hm...</think>").await,
        (String::new(), "Hm...".to_string()),
        "requests without the flag must be unchanged"
    );
    // An answer was generated, so reasoning stays where it belongs.
    assert_eq!(
        run(true, "<think>Hm...</think>Hi!").await,
        ("Hi!".to_string(), "Hm...".to_string()),
        "the flag must not move reasoning when content exists"
    );
}

/// A turn whose entire output is whitespace still has to come back with that
/// whitespace under `force_nonempty_content=true`. Held whitespace is normally
/// dropped in favour of the reasoning text (matching the aggregator), but when
/// whitespace is genuinely all the model produced there is nothing to prefer it
/// to — dropping it would hand back a completely empty `content` to the one
/// request that asked for the opposite.
#[tokio::test]
async fn postprocessor_parsing_stream_force_nonempty_whitespace_only_turn_survives() {
    let preprocessor = build_preprocessor(Some("nemotron_v3"), None);

    let mut request: NvCreateChatCompletionRequest = serde_json::from_str(REQUEST_JSON).unwrap();
    request.chat_template_args = Some(
        serde_json::from_value(serde_json::json!({ "force_nonempty_content": true })).unwrap(),
    );

    // `</think>` leaves the reasoning block, so "\n  " is normal text — and it is
    // whitespace-only, so it is held rather than settling the turn. Nothing else
    // follows, so the held whitespace is the whole response.
    let input_chunks = vec![mock_content_chunk("</think>\n  "), mock_final_chunk()];
    let input_stream = stream::iter(input_chunks.into_iter().map(Annotated::from_data));
    let output_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
        .postprocessor_parsing_stream(input_stream, &request, true, false)
        .expect("postprocessor_parsing_stream should build")
        .collect()
        .await;

    let mut content = String::new();
    for output in &output_chunks {
        let Some(data) = output.data.as_ref() else {
            continue;
        };
        for choice in &data.inner.choices {
            if let Some(c) = &choice.delta.content {
                content.push_str(get_text(c));
            }
        }
    }

    assert_eq!(
        content, "\n  ",
        "whitespace-only output must not vanish for a force_nonempty_content request"
    );
}

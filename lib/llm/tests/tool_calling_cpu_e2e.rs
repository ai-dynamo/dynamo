// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-only end-to-end coverage for reasoning-aware tool calling.
//!
//! The mock boundary is the model worker, not the OpenAI protocol. Every case
//! traverses the real HTTP service, request preprocessing, guided-decoding or
//! structural-tag selection, raw backend stream transformation, reasoning
//! parser, tool parser/jail, finish normalization, and streaming/unary response
//! serialization.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dynamo_llm::local_model::runtime_config::StructuralTagMode;
use dynamo_llm::model_card::{ModelDeploymentCard, PromptFormatterArtifact};
use dynamo_llm::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::codec::create_message_stream;
use dynamo_llm::protocols::common::FinishReason as BackendFinishReason;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::request_trace::{self, RequestTraceEventType};
use dynamo_protocols::types::FinishReason as OpenAIFinishReason;
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS;
use dynamo_runtime::config::environment_names::llm::request_trace as request_trace_env;
use futures::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio_util::sync::CancellationToken;

#[path = "common/ports.rs"]
mod ports;
#[path = "common/raw_chat_harness.rs"]
mod raw_chat_harness;

use raw_chat_harness::{MODEL, RawChatHarness, WorkerScript};

const ENV: [(&str, Option<&str>); 3] = [
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
    (request_trace_env::DYN_REQUEST_TRACE, Some("1")),
    (request_trace_env::DYN_REQUEST_TRACE_SINKS, Some("")),
];
const REASONING_AWARE_GUIDED_DECODING: &str = "reasoning_aware_guided_decoding";
const STRUCTURAL_MARKERS: &[&str] = &[
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<TOOLCALL>",
    "</TOOLCALL>",
    "<|channel|>",
    "<|start|>",
    "<|constrain|>",
    "<|message|>",
    "<|end|>",
    "<|call|>",
    "<｜DSML｜tool_calls>",
    "</｜DSML｜tool_calls>",
    "<｜DSML｜invoke",
    "</｜DSML｜invoke>",
    "<｜DSML｜parameter",
    "</｜DSML｜parameter>",
];

#[derive(Clone, Debug)]
struct ExpectedCall {
    name: &'static str,
    arguments: Value,
}

#[derive(Clone, Debug)]
enum ContentExpectation {
    Empty,
    Exact(&'static str),
}

#[derive(Clone, Debug)]
struct ExpectedResponse {
    reasoning: Option<String>,
    content: ContentExpectation,
    calls: Vec<ExpectedCall>,
    finish_reason: &'static str,
}

#[derive(Default, Debug)]
struct ObservedCall {
    id: String,
    call_type: String,
    name: String,
    arguments: String,
}

#[derive(Default, Debug)]
struct ObservedResponse {
    reasoning: String,
    unexpected_reasoning_alias: String,
    content: String,
    calls: BTreeMap<u64, ObservedCall>,
    finish_reason: Option<String>,
    streaming: bool,
    next_position: usize,
    last_reasoning_position: Option<usize>,
    first_tool_position: Option<usize>,
    finish_position: Option<usize>,
    last_output_position: Option<usize>,
    usage_position: Option<usize>,
    usage_count: usize,
    finish_count: usize,
}

impl ObservedResponse {
    fn ingest_choice(&mut self, choice: &Value, payload_key: &str) {
        let position = self.next_position;
        self.next_position += 1;
        let payload = &choice[payload_key];
        if let Some(reasoning) = payload.get("reasoning_content").and_then(Value::as_str) {
            self.reasoning.push_str(reasoning);
            if !reasoning.is_empty() {
                self.last_reasoning_position = Some(position);
                self.last_output_position = Some(position);
            }
        }
        if let Some(reasoning) = payload.get("reasoning").and_then(Value::as_str) {
            self.unexpected_reasoning_alias.push_str(reasoning);
        }
        if let Some(content) = payload.get("content").and_then(Value::as_str) {
            self.content.push_str(content);
            if !content.is_empty() {
                self.last_output_position = Some(position);
            }
        }

        if let Some(calls) = payload.get("tool_calls").and_then(Value::as_array) {
            if !calls.is_empty() {
                self.first_tool_position.get_or_insert(position);
                self.last_output_position = Some(position);
            }
            for (ordinal, call) in calls.iter().enumerate() {
                let index = if self.streaming {
                    call.get("index")
                        .and_then(Value::as_u64)
                        .expect("streaming tool-call delta is missing its index")
                } else {
                    ordinal as u64
                };
                let observed = self.calls.entry(index).or_default();
                if let Some(id) = call.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    if observed.id.is_empty() {
                        observed.id.push_str(id);
                    } else {
                        assert_eq!(observed.id, id, "tool call ID changed across deltas");
                    }
                }
                if let Some(call_type) = call.get("type").and_then(Value::as_str)
                    && !call_type.is_empty()
                {
                    if observed.call_type.is_empty() {
                        observed.call_type.push_str(call_type);
                    } else {
                        assert_eq!(observed.call_type, call_type);
                    }
                }
                let function = &call["function"];
                if let Some(name) = function.get("name").and_then(Value::as_str)
                    && !name.is_empty()
                {
                    if observed.name.is_empty() {
                        observed.name.push_str(name);
                    } else if observed.name != name {
                        observed.name.push_str(name);
                    }
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    observed.arguments.push_str(arguments);
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
            self.finish_position = Some(position);
            self.finish_count += 1;
        }
    }

    fn ingest_usage(&mut self) {
        self.usage_position = Some(self.next_position);
        self.usage_count += 1;
        self.next_position += 1;
    }
}

static REQUEST_TRACE_INIT: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();

async fn ensure_request_trace() {
    REQUEST_TRACE_INIT
        .get_or_init(|| async {
            request_trace::init_from_env_with_shutdown(CancellationToken::new())
                .await
                .expect("failed to initialize request tracing");
        })
        .await;
}

fn model_card(
    reasoning_parser: &str,
    tool_call_parser: &str,
    structural_tags: bool,
    reasoning_aware_guided_decoding: bool,
) -> ModelDeploymentCard {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let mut card = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    card.runtime_config.reasoning_parser = Some(reasoning_parser.to_string());
    card.runtime_config.tool_call_parser = Some(tool_call_parser.to_string());
    card.runtime_config.structural_tag_mode = if structural_tags {
        StructuralTagMode::On
    } else {
        StructuralTagMode::Off
    };
    card.runtime_config
        .set_engine_specific(
            REASONING_AWARE_GUIDED_DECODING,
            reasoning_aware_guided_decoding,
        )
        .unwrap();
    card.kv_cache_block_size = 16;
    card
}

fn model_card_with_prompt_injected_reasoning(
    reasoning_parser: &str,
    tool_call_parser: &str,
    structural_tags: bool,
) -> ModelDeploymentCard {
    let mut card = model_card(reasoning_parser, tool_call_parser, structural_tags, true);
    let template_dir = data_path("replays/tool-calling-cpu-e2e/prompt-injected");
    card.chat_template_file = PromptFormatterArtifact::chat_template_from_disk(&template_dir)
        .expect("failed to load prompt-injected chat template");
    assert!(card.chat_template_file.is_some());
    card
}

fn worker_output(
    text: Option<String>,
    finish_reason: Option<BackendFinishReason>,
) -> LLMEngineOutput {
    LLMEngineOutput {
        text,
        finish_reason,
        index: Some(0),
        ..Default::default()
    }
}

fn scripted_text(parts: &[&str]) -> WorkerScript {
    scripted_text_with_finish(parts, BackendFinishReason::Stop)
}

fn scripted_text_with_finish(parts: &[&str], finish_reason: BackendFinishReason) -> WorkerScript {
    parts
        .iter()
        .map(|part| worker_output(Some((*part).to_string()), None))
        .chain(std::iter::once(worker_output(None, Some(finish_reason))))
        .collect()
}

fn data_path(relative: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(relative)
}

fn load_data_json(relative: &str) -> Result<Value> {
    let path = data_path(relative);
    serde_json::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

/// Convert an existing parser trace's raw content chunks into worker outputs.
fn load_recorded_parser_trace(relative: &str) -> Result<WorkerScript> {
    let path = data_path(relative);
    let root = load_data_json(relative)?;
    let input_stream = root["input_stream"]
        .as_array()
        .with_context(|| format!("{} has no input_stream", path.display()))?;
    let mut script = Vec::new();

    for annotated in input_stream {
        let Some(choices) = annotated["data"]["choices"].as_array() else {
            continue;
        };
        for choice in choices {
            let text = choice["delta"]["content"].as_str().map(ToString::to_string);
            let finish_reason = match choice["finish_reason"].as_str() {
                Some("length") => Some(BackendFinishReason::Length),
                Some("content_filter") => Some(BackendFinishReason::ContentFilter),
                Some(_) => Some(BackendFinishReason::Stop),
                None => None,
            };
            if text.is_some() || finish_reason.is_some() {
                script.push(worker_output(text, finish_reason));
            }
        }
    }

    anyhow::ensure!(
        !script.is_empty(),
        "{} yielded no backend chunks",
        path.display()
    );
    anyhow::ensure!(
        script.iter().any(|chunk| chunk.finish_reason.is_some()),
        "{} has no terminal backend chunk",
        path.display()
    );
    Ok(script)
}

fn load_recorded_expected_text(relative: &str, field: &str) -> Result<String> {
    load_data_json(relative)?["expected_output"][field]
        .as_str()
        .map(ToString::to_string)
        .with_context(|| format!("{relative} has no expected_output.{field}"))
}

fn load_nemotron_required_trace() -> Result<WorkerScript> {
    let root = load_data_json("replays/tool-calling-cpu-e2e/nemotron-nano-required-guided.json")?;
    root["raw_backend_stream"]
        .as_array()
        .context("Nemotron trace has no raw_backend_stream")?
        .iter()
        .map(|chunk| {
            let text = chunk["text"].as_str().map(ToString::to_string);
            let finish_reason = match chunk["finish_reason"].as_str() {
                Some("length") => Some(BackendFinishReason::Length),
                Some(_) => Some(BackendFinishReason::Stop),
                None => None,
            };
            Ok(worker_output(text, finish_reason))
        })
        .collect()
}

fn load_nemotron_expected_reasoning() -> Result<String> {
    load_data_json("replays/tool-calling-cpu-e2e/nemotron-nano-required-guided.json")?
        ["raw_backend_stream"][0]["text"]
        .as_str()
        .map(ToString::to_string)
        .context("Nemotron trace has no initial reasoning text")
}

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"}
                },
                "required": ["location"],
                "additionalProperties": false
            }
        }
    })
}

fn time_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the time in a timezone.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
                "additionalProperties": false
            }
        }
    })
}

fn deepseek_weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "format": {"type": "string"}
                },
                "required": ["location", "format"],
                "additionalProperties": false
            }
        }
    })
}

fn strict_tool(mut tool: Value) -> Value {
    tool["function"]["strict"] = Value::Bool(true);
    tool
}

fn chat_body(
    tool_choice: &str,
    tools: Vec<Value>,
    parallel_tool_calls: bool,
    thinking_key: &str,
    reasoning_enabled: bool,
) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Use the available tools when appropriate."}],
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "temperature": 0,
        "max_tokens": 1024,
        "stream_options": {"include_usage": true},
        "chat_template_kwargs": {thinking_key: reasoning_enabled}
    })
}

async fn post_chat(svc: &RawChatHarness, body: &Value, streaming: bool) -> ObservedResponse {
    let mut body = body.clone();
    body["stream"] = Value::Bool(streaming);
    if !streaming {
        body.as_object_mut().unwrap().remove("stream_options");
    }
    let response = svc
        .client
        .post(format!("{}/v1/chat/completions", svc.base_url))
        .header("x-dynamo-session-id", "cpu-tool-e2e-session")
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");
    if response.status() != reqwest::StatusCode::OK {
        let status = response.status();
        let error = response.text().await.unwrap_or_default();
        panic!("chat request failed with {status}: {error}");
    }

    let mut observed = ObservedResponse {
        streaming,
        ..Default::default()
    };
    if streaming {
        let raw = response.text().await.unwrap();
        let data_lines: Vec<_> = raw
            .lines()
            .filter(|line| line.starts_with("data: "))
            .collect();
        assert_eq!(
            data_lines
                .iter()
                .filter(|line| **line == "data: [DONE]")
                .count(),
            1,
            "{raw}"
        );
        assert_eq!(
            data_lines.last().copied(),
            Some("data: [DONE]"),
            "SSE data appeared after [DONE]: {raw}"
        );
        let mut messages = create_message_stream(&raw);
        while let Some(message) = messages.next().await {
            let message = message.expect("invalid SSE response frame");
            let Some(data) = message.data else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let chunk: Value = serde_json::from_str(&data).expect("invalid SSE response JSON");
            if chunk.get("usage").is_some_and(|usage| !usage.is_null()) {
                assert_eq!(
                    chunk["choices"].as_array().map(Vec::len),
                    Some(0),
                    "streaming usage must be emitted in a choices-free terminal chunk"
                );
                observed.ingest_usage();
            }
            for choice in chunk["choices"].as_array().into_iter().flatten() {
                observed.ingest_choice(choice, "delta");
            }
        }
    } else {
        let response: Value = response.json().await.unwrap();
        for choice in response["choices"].as_array().into_iter().flatten() {
            observed.ingest_choice(choice, "message");
        }
    }
    observed
}

fn assert_response(observed: &ObservedResponse, expected: &ExpectedResponse) {
    assert!(
        observed.unexpected_reasoning_alias.is_empty(),
        "response used nonstandard `reasoning` instead of `reasoning_content`: {:?}",
        observed.unexpected_reasoning_alias
    );
    match expected.reasoning.as_deref() {
        Some(reasoning) => assert_eq!(observed.reasoning, reasoning),
        None => assert!(
            observed.reasoning.is_empty(),
            "reasoning should be disabled: {:?}",
            observed.reasoning
        ),
    }
    match expected.content {
        ContentExpectation::Empty => assert_eq!(
            observed.content, "",
            "tool-only response leaked content or separators"
        ),
        ContentExpectation::Exact(expected) => assert_eq!(observed.content, expected),
    }
    assert_eq!(observed.calls.len(), expected.calls.len(), "{observed:#?}");
    assert_eq!(
        observed.calls.keys().copied().collect::<Vec<_>>(),
        (0..expected.calls.len() as u64).collect::<Vec<_>>(),
        "tool call indices must be dense and ordered"
    );
    let mut ids = HashSet::new();
    for (observed, expected) in observed.calls.values().zip(&expected.calls) {
        assert!(!observed.id.is_empty(), "tool call ID is empty");
        assert!(
            ids.insert(&observed.id),
            "duplicate tool call ID: {}",
            observed.id
        );
        assert_eq!(observed.call_type, "function");
        assert_eq!(observed.name, expected.name);
        assert_eq!(
            serde_json::from_str::<Value>(&observed.arguments).unwrap(),
            expected.arguments
        );
    }
    assert_eq!(
        observed.finish_reason.as_deref(),
        Some(expected.finish_reason)
    );
    assert_eq!(observed.finish_count, 1, "{observed:#?}");
    if observed.streaming && observed.last_output_position.is_some() {
        assert!(
            observed.last_output_position <= observed.finish_position,
            "semantic output arrived after the terminal finish: {observed:#?}"
        );
    }
    if observed.streaming && !expected.calls.is_empty() && expected.reasoning.is_some() {
        assert!(
            observed.last_reasoning_position < observed.first_tool_position,
            "tool delta arrived before reasoning completed: {observed:#?}"
        );
    }
    if observed.streaming {
        assert_eq!(observed.usage_count, 1, "{observed:#?}");
        assert!(
            observed.finish_position < observed.usage_position,
            "usage arrived before the terminal choice: {observed:#?}"
        );
    }

    let visible = format!(
        "{}{}{}",
        observed.reasoning,
        observed.content,
        observed
            .calls
            .values()
            .map(|call| call.arguments.as_str())
            .collect::<String>()
    );
    for marker in STRUCTURAL_MARKERS {
        assert!(
            !visible.contains(marker),
            "marker {marker:?} leaked: {visible:?}"
        );
    }
}

async fn run_streaming_and_unary(
    card: ModelDeploymentCard,
    script: WorkerScript,
    body: Value,
    expected: ExpectedResponse,
) -> Vec<PreprocessedRequest> {
    ensure_request_trace().await;
    let svc = RawChatHarness::start(card, [script.clone(), script]).await;
    for streaming in [false, true] {
        let observed = post_chat(&svc, &body, streaming).await;
        assert_response(&observed, &expected);
    }
    assert_eq!(svc.engine.remaining_scripts().await, 0);
    let requests = svc.engine.take_requests().await;
    assert_eq!(requests.len(), 2);
    svc.shutdown().await;
    requests
}

fn assert_json_guidance(requests: &[PreprocessedRequest], tools: &[Value]) {
    for request in requests {
        let guided = request
            .sampling_options
            .guided_decoding
            .as_ref()
            .expect("required tool choice did not create guided decoding");
        let schema = guided
            .json
            .as_ref()
            .expect("required tool choice has no JSON schema");
        assert!(guided.structural_tag.is_none());
        assert!(guided.regex.is_none());
        assert!(guided.choice.is_none());
        assert!(guided.grammar.is_none());
        assert!(guided.backend.is_none());
        assert!(guided.whitespace_pattern.is_none());
        assert_eq!(schema["type"], "array");
        assert_eq!(schema["minItems"], 1);
        let alternatives = schema["items"]["anyOf"]
            .as_array()
            .expect("required tool schema has no alternatives");
        assert_eq!(alternatives.len(), tools.len());
        for (alternative, tool) in alternatives.iter().zip(tools) {
            assert_eq!(
                alternative["properties"]["name"]["enum"],
                json!([tool["function"]["name"]])
            );
            assert_eq!(
                alternative["properties"]["parameters"],
                tool["function"]["parameters"]
            );
            assert_eq!(alternative["required"], json!(["name", "parameters"]));
        }
    }
}

fn assert_no_guidance(requests: &[PreprocessedRequest]) {
    for request in requests {
        if let Some(guided) = request.sampling_options.guided_decoding.as_ref() {
            assert!(
                guided.json.is_none(),
                "unexpected JSON guidance: {guided:#?}"
            );
            assert!(
                guided.regex.is_none(),
                "unexpected regex guidance: {guided:#?}"
            );
            assert!(
                guided.choice.is_none(),
                "unexpected choice guidance: {guided:#?}"
            );
            assert!(
                guided.grammar.is_none(),
                "unexpected grammar guidance: {guided:#?}"
            );
            assert!(
                guided.structural_tag.is_none(),
                "unexpected structural guidance: {guided:#?}"
            );
            assert!(guided.backend.is_none(), "unexpected backend: {guided:#?}");
            assert!(
                guided.whitespace_pattern.is_none(),
                "unexpected whitespace policy: {guided:#?}"
            );
        }
    }
}

fn collect_tool_tags<'a>(value: &'a Value, tags: &mut Vec<(&'a str, &'a Value)>) {
    match value {
        Value::Object(object) => {
            if let (Some(begin), Some(content)) = (
                object.get("begin").and_then(Value::as_str),
                object.get("content"),
            ) && content["type"] == "json_schema"
            {
                tags.push((begin, &content["json_schema"]));
            }
            for child in object.values() {
                collect_tool_tags(child, tags);
            }
        }
        Value::Array(array) => {
            for child in array {
                collect_tool_tags(child, tags);
            }
        }
        _ => {}
    }
}

fn assert_structural_guidance(
    requests: &[PreprocessedRequest],
    tools: &[Value],
    required: bool,
    parallel: bool,
    starts_in_reasoning: bool,
) {
    for request in requests {
        let guided = request
            .sampling_options
            .guided_decoding
            .as_ref()
            .expect("structural-tag policy did not create guided decoding");
        let structural_tag = guided
            .structural_tag
            .as_ref()
            .expect("structural-tag policy did not create a tag");
        assert!(guided.json.is_none());
        assert!(guided.regex.is_none());
        assert!(guided.choice.is_none());
        assert!(guided.grammar.is_none());
        assert!(guided.backend.is_none());
        assert!(guided.whitespace_pattern.is_none());
        assert_eq!(structural_tag["type"], "structural_tag");

        let mut format = &structural_tag["format"];
        if starts_in_reasoning {
            assert_eq!(format["type"], "sequence");
            assert_eq!(format["elements"][0]["end"], "</think>");
            format = &format["elements"][1];
        } else {
            assert_ne!(format["type"], "sequence");
        }
        assert_eq!(format["at_least_one"], required);
        assert_eq!(format["stop_after_first"], !parallel);

        let mut tool_tags = Vec::new();
        collect_tool_tags(format, &mut tool_tags);
        assert_eq!(tool_tags.len(), tools.len(), "{structural_tag:#?}");
        for ((begin, schema), tool) in tool_tags.into_iter().zip(tools) {
            let name = tool["function"]["name"].as_str().unwrap();
            assert!(
                begin.contains(name),
                "tool tag {begin:?} does not select {name:?}"
            );
            assert_eq!(schema, &tool["function"]["parameters"]);
        }
    }
}

fn assert_reasoning_metadata(
    requests: &[PreprocessedRequest],
    ended: bool,
    expected_chat_template_kwargs: &Value,
) {
    for request in requests {
        let extra = request
            .extra_args
            .as_ref()
            .and_then(Value::as_object)
            .expect("reasoning-aware backend metadata is missing");
        assert_eq!(
            extra
                .get("reasoning_parser_kwargs")
                .and_then(|value| value.get("chat_template_kwargs")),
            Some(expected_chat_template_kwargs),
            "unexpected native reasoning parser kwargs: {extra:#?}"
        );
        assert_eq!(
            extra.get("reasoning_ended").and_then(Value::as_bool),
            ended.then_some(true),
            "unexpected initial reasoning state: {extra:#?}"
        );
    }
}

#[path = "tool_calling_cpu_e2e/cases.rs"]
mod cases;

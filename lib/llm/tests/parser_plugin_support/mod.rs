// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drive statically registered parser families through the streaming postprocessor.

use std::path::PathBuf;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FinishReason, Role,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};

/// Model output split at awkward points, including inside every marker.
pub const CHUNKS: &[&str] = &[
    "<dummy:th",
    "ink>user wants wea",
    "ther</dummy:think>Checking. <dum",
    "my:call>{\"name\":\"get_weather\",",
    "\"arguments\":{\"location\":\"SF\"}}</dummy:ca",
    "ll>",
];

#[derive(Debug, Default, PartialEq)]
pub struct Seen {
    pub reasoning: String,
    pub content: String,
    pub calls: Vec<(String, String)>,
    pub finish: Vec<FinishReason>,
}

fn chunk(
    content: Option<&str>,
    finish: Option<FinishReason>,
) -> NvCreateChatCompletionStreamResponse {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: content.map(|_| Role::Assistant),
            content: content.map(|text| ChatCompletionMessageContent::Text(text.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: finish,
        logprobs: None,
    };
    NvCreateChatCompletionStreamResponse {
        message_boundary: None,
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
        prompt_logprobs: None,
        llm_metrics: None,
        tool_call_completion: Vec::new(),
        model_eos: Vec::new(),
    }
}

pub async fn run_dummy_stream(parser: &str) -> Seen {
    run_stream(parser, serde_json::json!({}), CHUNKS).await
}

/// `extra` is merged into the request body.
pub async fn run_stream(parser: &str, extra: serde_json::Value, chunks: &[&str]) -> Seen {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    mdc.runtime_config.reasoning_parser = Some(parser.to_string());
    mdc.runtime_config.tool_call_parser = Some(parser.to_string());
    let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();

    let mut body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Weather in SF?"}],
        "stream": true,
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }}],
    });
    for (key, value) in extra.as_object().unwrap() {
        body[key] = value.clone();
    }
    let request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();

    let mut input: Vec<_> = chunks.iter().map(|text| chunk(Some(text), None)).collect();
    input.push(chunk(None, Some(FinishReason::Stop)));
    let output: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = preprocessor
        .postprocessor_parsing_stream(
            stream::iter(input.into_iter().map(Annotated::from_data)),
            &request,
            false,
            false,
        )
        .expect("stream should build")
        .collect()
        .await;

    let mut seen = Seen::default();
    for choice in output
        .iter()
        .filter_map(|c| c.data.as_ref())
        .flat_map(|d| &d.inner.choices)
    {
        if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
            seen.content.push_str(text);
        }
        if let Some(reasoning) = &choice.delta.reasoning_content {
            seen.reasoning.push_str(&reasoning.to_string());
        }
        for call in choice.delta.tool_calls.iter().flatten() {
            let function = call.function.as_ref();
            let name = function.and_then(|f| f.name.clone());
            let arguments = function
                .and_then(|f| f.arguments.clone())
                .unwrap_or_default();
            match (name, seen.calls.last_mut()) {
                (Some(name), _) => seen.calls.push((name, arguments)),
                (None, Some(last)) => last.1.push_str(&arguments),
                (None, None) => panic!("argument delta before a named call"),
            }
        }
        seen.finish.extend(choice.finish_reason);
    }
    seen
}

pub fn expected() -> Seen {
    Seen {
        reasoning: "user wants weather".to_string(),
        content: "Checking. ".to_string(),
        calls: vec![(
            "get_weather".to_string(),
            r#"{"location":"SF"}"#.to_string(),
        )],
        finish: vec![FinishReason::ToolCalls],
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::time::Duration;

use dynamo_llm::backend::Backend;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    NvCreateChatCompletionStreamResponse, aggregator::ChatCompletionAggregator,
};
use dynamo_llm::protocols::openai::{GuidedToolConstraint, ParsingOptions};
use dynamo_llm::tokenizers::Tokenizer;
use dynamo_runtime::pipeline::{
    AsyncEngine, Context, ManyOut, Operator, SegmentSource, ServiceBackend, SingleIn, Source,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{Controller, Event, RequestHandle, RequestPlan};
use futures::{StreamExt, stream};
use serde_json::{Value, json};

use super::process::Environment;
use super::support::{FixtureConfig, ProcessFixture, sglang, vllm};

trait ToolFixture: ProcessFixture {
    fn respond_with_tokens(&self, id: &str, tokens: &[u32], tokenizer: &Tokenizer);
    fn prompt_and_schema(handle: &RequestHandle<Self::Protocol>) -> (Vec<u32>, Option<Value>);
}

impl ToolFixture for vllm::Fixture {
    fn respond_with_tokens(&self, id: &str, tokens: &[u32], tokenizer: &Tokenizer) {
        use dynamo_vllm_sidecar::proto as pb;
        let mut responses: Vec<_> = tokens
            .chunks(7)
            .map(|chunk| pb::GenerateResponse {
                outputs: Some(pb::SequenceOutput {
                    token_ids: chunk.to_vec(),
                    text: tokenizer.decode(chunk, false).unwrap().into(),
                    num_tokens: chunk.len() as u32,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect();
        responses.push(pb::GenerateResponse {
            outputs: Some(pb::SequenceOutput {
                finish_info: Some(pb::FinishInfo {
                    num_output_tokens: tokens.len() as u32,
                    finish_reason: pb::finish_info::FinishReason::Stop as i32,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        });
        self.respond(id, responses);
    }

    fn prompt_and_schema(handle: &RequestHandle<Self::Protocol>) -> (Vec<u32>, Option<Value>) {
        use dynamo_vllm_sidecar::proto as pb;
        let native = handle.native_request().unwrap();
        let Some(pb::generate_request::Prompt::TokenIds(prompt)) = native.prompt else {
            panic!("expected a tokenized native request");
        };
        let schema = native
            .decoding
            .unwrap()
            .structured_output
            .map(|constraint| {
                let pb::decoding_parameters::StructuredOutput::Json(schema) = constraint else {
                    panic!("expected native JSON guidance");
                };
                serde_json::from_str(&schema).unwrap()
            });
        (prompt.ids, schema)
    }
}

impl ToolFixture for sglang::Fixture {
    fn respond_with_tokens(&self, id: &str, tokens: &[u32], _tokenizer: &Tokenizer) {
        use dynamo_sglang_sidecar::proto as pb;
        let mut responses: Vec<_> = tokens
            .chunks(7)
            .map(|chunk| pb::GenerateResponse {
                output_ids: chunk.iter().map(|&token| token as i32).collect(),
                ..Default::default()
            })
            .collect();
        responses.push(pb::GenerateResponse {
            finished: true,
            meta_info: [("finish_reason".into(), json!({"type": "stop"}).to_string())].into(),
            ..Default::default()
        });
        self.respond(id, responses);
    }

    fn prompt_and_schema(handle: &RequestHandle<Self::Protocol>) -> (Vec<u32>, Option<Value>) {
        use dynamo_sglang_sidecar::proto as pb;
        let native = handle.native_request().unwrap();
        let schema = native
            .sampling_params
            .unwrap()
            .guided_decoding
            .map(|guided| {
                let Some(pb::guided_decoding::Constraint::JsonSchema(schema)) = guided.constraint
                else {
                    panic!("expected native JSON guidance");
                };
                serde_json::from_str(&schema).unwrap()
            });
        (
            native
                .input_ids
                .into_iter()
                .map(|token| u32::try_from(token).unwrap())
                .collect(),
            schema,
        )
    }
}

fn configure_chat_model(model: &str) {
    let model = Path::new(model);
    let mut vocab = serde_json::Map::from_iter([
        ("[UNK]".into(), json!(0)),
        ("<s>".into(), json!(1)),
        ("</s>".into(), json!(2)),
    ]);
    for byte in b' '..=b'~' {
        vocab.insert(char::from(byte).to_string(), json!(u32::from(byte) + 3));
    }
    vocab.insert("\n".into(), json!(13));
    std::fs::write(
        model.join("tokenizer.json"),
        json!({
            "version": "1.0", "added_tokens": [],
            "decoder": {"type": "Fuse"},
            "model": {"type": "BPE", "vocab": vocab, "merges": [], "unk_token": "[UNK]"}
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        model.join("tokenizer_config.json"),
        json!({
            "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096,
            "bos_token": "<s>", "eos_token": "</s>",
            "chat_template": "{{ (tools or []) | tojson }}\n{{ messages | tojson }}"
        })
        .to_string(),
    )
    .unwrap();
}

fn weather_schema() -> Value {
    json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
    })
}

fn chat_request(model: &str, tool_choice: Value, messages: Value) -> NvCreateChatCompletionRequest {
    serde_json::from_value(json!({
        "model": model,
        "stream": true,
        "max_tokens": 256,
        "messages": messages,
        "tools": [{"type": "function", "function": {
            "name": "weather", "parameters": weather_schema()
        }}],
        "tool_choice": tool_choice,
        "parallel_tool_calls": false
    }))
    .unwrap()
}

async fn collect_chat(
    stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    options: ParsingOptions,
    finish_reason: &str,
) -> Value {
    let chunks: Vec<_> = bounded("chat response completion", stream.collect()).await;
    let mut terminals = Vec::new();
    for chunk in &chunks {
        if let Some(data) = chunk.clone().into_data().unwrap() {
            let value = serde_json::to_value(data).unwrap();
            for choice in value["choices"].as_array().unwrap() {
                if !choice["finish_reason"].is_null() {
                    terminals.push(choice["finish_reason"].as_str().unwrap().to_owned());
                }
                if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
                    for call in calls {
                        assert_eq!(call["index"], 0);
                        if let Some(id) = call["id"].as_str() {
                            assert!(!id.is_empty());
                        }
                    }
                }
                if finish_reason == "tool_calls" {
                    assert!(
                        choice["delta"]["content"]
                            .as_str()
                            .unwrap_or_default()
                            .is_empty(),
                        "{value}"
                    );
                }
            }
        }
    }
    assert_eq!(terminals, [finish_reason]);
    let response =
        NvCreateChatCompletionResponse::from_annotated_stream(stream::iter(chunks), options)
            .await
            .unwrap();
    serde_json::to_value(response).unwrap()
}

async fn tool_call_round_trip<F: ToolFixture>() {
    let env = Environment::new().await;
    configure_chat_model(&env.model);
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    let mut child =
        env.spawn_with_args::<F>(&peer.endpoint(), &["--dyn-tool-call-parser", "hermes"]);
    let router = env.ready("backend").await;
    let mut cards = env.cards().await;
    assert_eq!(cards.len(), 1);
    let card = &mut cards[0];
    bounded("sidecar model metadata", card.download_config(None))
        .await
        .unwrap();
    assert_eq!(
        card.runtime_config.tool_call_parser.as_deref(),
        Some("hermes")
    );
    let tokenizer = card.tokenizer().unwrap();
    let frontend = SegmentSource::<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >::new();
    let preprocessor = OpenAIPreprocessor::new(card.clone())
        .unwrap()
        .into_operator();
    let decoder = Backend::from_tokenizer(tokenizer.clone()).into_operator();
    let pipeline = frontend
        .link(preprocessor.forward_edge())
        .unwrap()
        .link(decoder.forward_edge())
        .unwrap()
        .link(ServiceBackend::from_engine(router.clone()))
        .unwrap()
        .link(decoder.backward_edge())
        .unwrap()
        .link(preprocessor.backward_edge())
        .unwrap()
        .link_terminal(frontend)
        .unwrap();

    for (name, is_forced) in [("auto", false), ("named", true)] {
        let id = format!("tool-{name}");
        let handle = control.request(&id, RequestPlan::default());
        let text = if is_forced {
            r#"{"city":"Paris"}"#
        } else {
            r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#
        };
        let tokens = tokenizer.encode(text).unwrap().token_ids().to_vec();
        peer.respond_with_tokens(&id, &tokens, &tokenizer);
        let user_messages = json!([{"role": "user", "content": "Weather in Paris?"}]);
        let tool_choice = if is_forced {
            json!({"type": "function", "function": {"name": "weather"}})
        } else {
            json!("auto")
        };
        let request = chat_request(&env.model, tool_choice, user_messages.clone());
        let mut options = ParsingOptions::new(card.runtime_config.tool_call_parser.clone(), None);
        options.tool_choice = request.inner.tool_choice.clone();
        if is_forced {
            options.guided_tool_constraint = GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "weather".into(),
            };
        }
        let stream = bounded(
            "tool request ingress",
            pipeline.generate(Context::with_id_and_metadata(
                request,
                id,
                Default::default(),
            )),
        )
        .await
        .unwrap();
        let response = collect_chat(stream, options, "tool_calls").await;
        assert_eq!(response["choices"].as_array().unwrap().len(), 1);
        let choice = &response["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        let calls = choice["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert!(!calls[0]["id"].as_str().unwrap().is_empty());
        assert_eq!(calls[0]["function"]["name"], "weather");
        assert_eq!(
            serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            json!({"city":"Paris"})
        );
        assert_eq!(handle.tokens(), tokens);
        let (prompt, schema) = F::prompt_and_schema(&handle);
        let prompt = tokenizer.decode(&prompt, false).unwrap();
        let (tools, messages) = prompt.as_str().split_once('\n').unwrap();
        let tools: Value = serde_json::from_str(tools).unwrap();
        assert_eq!(tools[0]["function"]["parameters"], weather_schema());
        assert_eq!(
            serde_json::from_str::<Value>(messages).unwrap(),
            user_messages
        );
        assert_eq!(schema, is_forced.then(weather_schema));
        bounded("tool native release", handle.wait(Event::Dropped)).await;

        if is_forced {
            continue;
        }
        let id = format!("result-{name}");
        let handle = control.request(&id, RequestPlan::default());
        peer.respond_with_tokens(
            &id,
            tokenizer.encode("Paris is 21C.").unwrap().token_ids(),
            &tokenizer,
        );
        let mut messages = user_messages.as_array().unwrap().clone();
        messages.push(choice["message"].clone());
        messages.push(json!({"role": "tool", "tool_call_id": calls[0]["id"], "content": "21C"}));
        let request = chat_request(&env.model, json!("none"), json!(messages));
        let stream = bounded(
            "tool result ingress",
            pipeline.generate(Context::with_id_and_metadata(
                request,
                id,
                Default::default(),
            )),
        )
        .await
        .unwrap();
        let response = collect_chat(stream, ParsingOptions::default(), "stop").await;
        assert_eq!(
            response["choices"][0]["message"]["content"],
            "Paris is 21C."
        );
        assert!(response["choices"][0]["message"]["tool_calls"].is_null());
        let (prompt, schema) = F::prompt_and_schema(&handle);
        let prompt = tokenizer.decode(&prompt, false).unwrap();
        let (tools, history) = prompt.as_str().split_once('\n').unwrap();
        assert_eq!(serde_json::from_str::<Value>(tools).unwrap(), json!([]));
        let history: Value = serde_json::from_str(history).unwrap();
        assert_eq!(history[1]["tool_calls"][0]["id"], calls[0]["id"]);
        assert_eq!(history[1]["tool_calls"][0]["function"]["name"], "weather");
        assert_eq!(
            history[1]["tool_calls"][0]["function"]["arguments"],
            json!({"city": "Paris"})
        );
        assert_eq!(history[2]["role"], "tool");
        assert_eq!(history[2]["content"], "21C");
        assert_eq!(history[2]["tool_call_id"], calls[0]["id"]);
        assert!(schema.is_none());
        bounded("tool result native release", handle.wait(Event::Dropped)).await;
    }
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

#[tokio::test]
async fn vllm_tool_calls_cross_frontend_and_native_transport() {
    tokio::time::timeout(
        Duration::from_secs(60),
        tool_call_round_trip::<vllm::Fixture>(),
    )
    .await
    .expect("tool scenario exceeded its overall deadline");
}

#[tokio::test]
async fn sglang_tool_calls_cross_frontend_and_native_transport() {
    tokio::time::timeout(
        Duration::from_secs(60),
        tool_call_round_trip::<sglang::Fixture>(),
    )
    .await
    .expect("tool scenario exceeded its overall deadline");
}

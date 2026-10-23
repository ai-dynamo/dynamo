// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::super::*;
use dynamo_renderer::{ChatTemplate, ContextMixins, OAIPromptFormatter, PromptFormatter};
use serde_json::{Value, json};

const TEMPLATE: &str =
    include_str!("../../../../../tests/data/muse_spark_open_deferred_tools.jinja");

fn formatter(template: &str) -> std::sync::Arc<dyn OAIPromptFormatter> {
    let config: ChatTemplate = serde_json::from_value(json!({"chat_template":template})).unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), true).unwrap();
    formatter
}

fn func(name: &str) -> Value {
    json!({"type":"function","name":name,"description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}})
}

fn http_params(request: &NvCreateResponse) -> ResponseParams {
    let loaded: Vec<_> = effective_response_tools(request)
        .unwrap()
        .into_iter()
        .map(|t| t.inner)
        .collect();
    ResponseParams {
        tool_names: Some(ToolNameMap::new(&loaded, Some(&request.inner.input))),
        tools: request
            .inner
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(|tool| tool.inner.clone()).collect()),
        template_tools: request.inner.tools.clone(),
        loaded_tools: loaded,
        tool_choice: request.inner.tool_choice.clone(),
        ..Default::default()
    }
}

fn unary(params: &ResponseParams, name: &str, args: &str) -> anyhow::Result<NvResponse> {
    let response: NvCreateChatCompletionResponse = serde_json::from_value(json!({
        "id":"chat-1", "object":"chat.completion", "created":0, "model":"/model",
        "choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
            {"id":"call-1","type":"function","function":{"name":name,"arguments":args}}
        ]},"finish_reason":"tool_calls","logprobs":null}]
    }))?;
    chat_completion_to_response(response, params, None)
}

/// Only allowed declarations reach a legacy template.
#[test]
fn allowed_tools_filter_legacy_prompt() {
    let req: NvCreateResponse = serde_json::from_value(json!({"model":"m","input":"hi",
        "tools":[func("alpha"), func("beta")],
        "tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"function","name":"alpha"}]}}))
    .unwrap();
    let mut chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
    let f = formatter("{{ tools | tojson }}");
    chat.normalize_responses_tools(f.supports_native_tools(&chat))
        .unwrap();
    let inner: Vec<_> = chat
        .inner
        .tools
        .as_ref()
        .unwrap()
        .iter()
        .map(|t| t.function.name.clone())
        .collect();
    let rendered = f.render(&chat).unwrap();
    assert_eq!(inner, vec!["alpha".to_string()]);
    assert!(
        !rendered.contains("beta"),
        "unselected tool rendered: {rendered}"
    );
}

/// Native top-level names remain allowed across a namespace collision; the
/// unselected namespace member stays forbidden.
#[test]
fn native_collision_accepts_allowed_top_level_call() {
    let req: NvCreateResponse = serde_json::from_value(json!({"model":"m","input":"hi",
        "tools":[func("search"), {"type":"namespace","name":"web","description":"Web","tools":[func("search")]}],
        "tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"function","name":"search"}]}}))
    .unwrap();
    let params = http_params(&req);
    let mut chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
    let f = formatter(TEMPLATE);
    let native = f.supports_native_tools(&chat);
    chat.normalize_responses_tools(native).unwrap();
    let inner: Vec<_> = chat
        .inner
        .tools
        .as_ref()
        .unwrap()
        .iter()
        .map(|t| t.function.name.clone())
        .collect();
    let result = unary(&params, "search", r#"{"q":"x"}"#);
    assert!(native);
    assert_eq!(inner, vec!["search".to_string()]);
    let value = serde_json::to_value(result.unwrap()).unwrap();
    assert_eq!(value["output"][0]["name"], "search");
    assert!(value["output"][0]["namespace"].is_null());
    assert!(
        unary(&params, "web.search", r#"{"q":"x"}"#)
            .unwrap_err()
            .to_string()
            .contains("outside allowed_tools")
    );
}

/// Legacy templates retain main's coalesced assistant turns even if the
/// caller replays Responses phase metadata.
#[test]
fn legacy_history_coalescing_ignores_phases() {
    let make = |phase: Option<&str>| {
        let mut msg = json!({"type":"message","role":"assistant","id":"msg_1","status":"completed",
            "content":[{"type":"output_text","text":"C","annotations":[]}]});
        if let Some(phase) = phase {
            msg["phase"] = json!(phase);
        }
        let req: NvCreateResponse = serde_json::from_value(json!({"model":"m","tools":[func("alpha")],"input":[
            {"type":"message","role":"user","content":"hi"},
            {"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"R"}]},
            msg,
            {"type":"function_call","call_id":"c1","name":"alpha","arguments":"{}"},
            {"type":"function_call_output","call_id":"c1","output":"ok"}
        ]})).unwrap();
        let mut chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        chat.normalize_responses_tools(false).unwrap();
        serde_json::to_value(&chat.inner.messages).unwrap()
    };
    let without = make(None);
    let with = make(Some("commentary"));
    let count = |v: &Value| {
        v.as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "assistant")
            .count()
    };
    assert_eq!(count(&without), 1);
    assert_eq!(with, without);
    assert_eq!(count(&with), 1);
}

/// Freeform input remains byte-exact even when it happens to spell JSON.
#[tokio::test]
async fn custom_input_is_byte_exact_for_json_looking_payloads() {
    use dynamo_runtime::protocols::annotated::Annotated;
    let req: NvCreateResponse = serde_json::from_value(json!({"model":"m","input":"hi",
        "tools":[{"type":"custom","name":"run","description":"d"}]}))
    .unwrap();
    let params = http_params(&req);
    for payload in [
        r#"{"a":1}"#,
        "42",
        r#""quoted""#,
        "plain text",
        "true",
        "null",
    ] {
        let turn = format!(
            "<|start|>assistant to=run<|message|><atem:function_calls><atem:invoke name=\"run\"><atem:parameter name=\"input\">{payload}</atem:parameter></atem:invoke></atem:function_calls><|eom|>"
        );
        let raw: crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse = serde_json::from_value(
            json!({"id":"chat-1","object":"chat.completion.chunk","created":0,"model":"/model",
            "choices":[{"index":0,"delta":{"content":turn},"finish_reason":"stop","logprobs":null}]}),
        ).unwrap();
        let chat = crate::protocols::openai::chat_completions::DeltaAggregator::apply(
            futures::stream::iter([Annotated::from_data(raw)]),
            crate::protocols::openai::ParsingOptions::new(Some("muse_spark".into()), None)
                .with_tools(vec![dynamo_parsers::tool_calling::ToolDefinition {
                    name: "run".into(),
                    parameters: Some(custom_tool_parameters()),
                    strict: None,
                }]),
        )
        .await
        .unwrap();
        let args = chat.inner.choices[0]
            .message
            .tool_calls
            .as_ref()
            .map(|c| c[0].function.arguments.clone());
        let out = chat_completion_to_response(chat, &params, None)
            .map(|r| serde_json::to_value(r).unwrap()["output"].to_string())
            .map_err(|e| e.to_string());
        let out: Value = serde_json::from_str(&out.unwrap()).unwrap();
        assert_eq!(out[0]["input"], payload, "args={args:?}");
    }
}

/// Unary and streaming expose the same interleaved reasoning/message order.
#[tokio::test]
async fn unary_and_streaming_preserve_interleaved_reasoning_order() {
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    const TURN: &str = concat!(
        "<|start|>assistant to=self<|message|>R1<|eom|>",
        "<|start|>assistant to=commentary<|message|>C<|eom|>",
        "<|start|>assistant to=self<|message|>R2<|eom|>",
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    );
    let raw: crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse =
        serde_json::from_value(
            json!({"id":"chat-1","object":"chat.completion.chunk","created":0,"model":"/model",
        "choices":[{"index":0,"delta":{"content":TURN},"finish_reason":null,"logprobs":null}]}),
        )
        .unwrap();
    let fin: crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse =
        serde_json::from_value(
            json!({"id":"chat-1","object":"chat.completion.chunk","created":0,"model":"/model",
        "choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop","logprobs":null}]}),
        )
        .unwrap();
    let chunks: Vec<_> =
        crate::protocols::openai::chat_completions::unified_parser::apply_stream_with_policy(
            futures::stream::iter([Annotated::from_data(raw), Annotated::from_data(fin)]),
            None,
            crate::protocols::openai::GuidedToolConstraint::None,
            dynamo_parsers_v2::unified::UnifiedParserStartingState::None,
            "muse_spark",
            false,
            Default::default(),
        )
        .filter_map(|chunk| async move { chunk.data })
        .collect()
        .await;
    let params = ResponseParams {
        reasoning: Some(serde_json::from_value(json!({"summary":"auto"})).unwrap()),
        ..Default::default()
    };
    let mut converter =
        stream_converter::ResponseStreamConverter::new("/model".into(), params.clone());
    let mut events = converter.emit_start_events();
    for chunk in &chunks {
        events.extend(converter.process_chunk(chunk));
    }
    events.extend(converter.emit_end_events());
    use axum::response::IntoResponse;
    let response = axum::response::Sse::new(futures::stream::iter(events)).into_response();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: Vec<Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let completed = &parsed.last().unwrap()["response"]["output"];
    let stream_items: Vec<String> = completed
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            format!(
                "{}:{}",
                i["type"],
                i.get("phase").cloned().unwrap_or(Value::Null)
            )
        })
        .collect();
    let chat = crate::protocols::openai::chat_completions::DeltaAggregator::apply(
        futures::stream::iter(chunks.into_iter().map(Annotated::from_data)),
        Default::default(),
    )
    .await
    .unwrap();
    let unary =
        serde_json::to_value(chat_completion_to_response(chat, &params, None).unwrap()).unwrap();
    let unary_shape: Vec<_> = unary["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            format!(
                "{}:{}",
                i["type"],
                i.get("phase").cloned().unwrap_or(Value::Null)
            )
        })
        .collect();
    assert_eq!(unary_shape, stream_items);
    assert_eq!(unary["output"][0]["content"][0]["text"], "R1");
    assert_eq!(unary["output"][2]["content"][0]["text"], "R2");
    assert_eq!(unary["output"][3]["content"][0]["text"], "Done.");
}

/// Phase metadata must not hide typed images from the media collector.
#[test]
fn phased_history_keeps_typed_images_for_media_collection() {
    use dynamo_renderer::OAIChatLikeRequest;
    let req: NvCreateResponse = serde_json::from_value(json!({"model":"m","input":[
        {"type":"message","role":"assistant","id":"msg_1","status":"completed","phase":"final_answer",
         "content":[{"type":"output_text","text":"Earlier.","annotations":[]}]},
        {"type":"message","role":"user","content":[
            {"type":"input_text","text":"What is in this image?"},
            {"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo=","detail":"auto"}]}
    ]})).unwrap();
    let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
    let has_image = serde_json::to_string(&chat.inner.messages)
        .unwrap()
        .contains("image_url");
    assert!(has_image);
    assert!(chat.typed_messages().is_some());
}

#[test]
fn allowed_tools_filter_native_namespace_members() {
    let req: NvCreateResponse = serde_json::from_value(json!({"model":"m","input":"hi",
        "tools":[{"type":"namespace","name":"web","description":"Web","defer_loading":true,"tools":[func("alpha"),func("beta")]}],
        "tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"function","name":"alpha","namespace":"web"}]}})).unwrap();
    for template in [TEMPLATE, "{{ tools | tojson }}"] {
        let f = formatter(template);
        let mut chat: NvCreateChatCompletionRequest = req.clone().try_into().unwrap();
        chat.normalize_responses_tools(f.supports_native_tools(&chat))
            .unwrap();
        // The native namespace is deferred, so use name-only mode to make the
        // member set visible in the rendered prompt as well as its metadata.
        chat.chat_template_args
            .get_or_insert_default()
            .insert("deferred_render_mode".into(), json!("tool_name_only"));
        let rendered = f.render(&chat).unwrap();
        assert!(rendered.contains("alpha"));
        assert!(!rendered.contains("beta"), "{rendered}");
        assert_eq!(chat.inner.tools.as_ref().unwrap().len(), 1);
    }
}

#[test]
fn custom_allowed_tools_are_supported_but_unenforced_grammars_are_rejected() {
    let request = |format: Value| -> NvCreateResponse {
        serde_json::from_value(json!({
        "model":"m","input":"hi","tools":[{"type":"custom","name":"run","format":format}],
        "tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"custom","name":"run"}]}
    })).unwrap()
    };
    let req = request(json!({"type":"text"}));
    let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
    assert_eq!(chat.inner.tools.as_ref().unwrap()[0].function.name, "run");
    let req = request(json!({"type":"grammar","syntax":"regex","definition":"[a-z]+"}));
    let error = NvCreateChatCompletionRequest::try_from(req).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("grammar formats are not supported")
    );
}

#[test]
fn native_tool_support_alone_does_not_enable_phase_replay_for_other_models() {
    let req: NvCreateResponse = serde_json::from_value(json!({"model":"m","input":[
        {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"R"}]},
        {"role":"assistant","phase":"commentary","content":"C"}
    ]}))
    .unwrap();
    let mut chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
    chat.normalize_responses_history_phases(false);
    chat.normalize_responses_tools(true).unwrap();
    assert_eq!(chat.inner.messages.len(), 1);
    assert!(
        chat.responses_tools
            .as_ref()
            .unwrap()
            .history_phases
            .is_empty()
    );
    chat.normalize_responses_history_phases(true);
    chat.normalize_responses_tools(true).unwrap();
    assert_eq!(chat.inner.messages.len(), 2);
    assert_eq!(
        chat.responses_tools.as_ref().unwrap().history_phases.len(),
        1
    );
}

#[test]
fn python_history_defaults_to_legacy_coalescing_before_model_selection() {
    for extra_system in [false, true] {
        let request = |phase: bool| -> NvCreateResponse {
            let mut message = json!({"role":"assistant","content":"Checking."});
            if phase {
                message["phase"] = json!("commentary");
            }
            let mut input = vec![];
            if extra_system {
                input.push(json!({"role":"developer","content":"Developer rules"}));
            }
            input.extend([
                json!({"role":"user","content":[{"type":"input_text","text":"Look"},
                    {"type":"input_image","image_url":"https://example.com/a.png","detail":"auto"}]}),
                json!({"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"R"}]}),
                message,
                json!({"type":"function_call","call_id":"call-1","name":"alpha","arguments":"{}"}),
                json!({"type":"function_call_output","call_id":"call-1","output":"ok"}),
                json!({"role":"user","content":"Continue"}),
            ]);
            let mut body = json!({"model":"m","tools":[func("alpha")],"input":input});
            if extra_system {
                body["instructions"] = json!("System rules");
            }
            serde_json::from_value(body).unwrap()
        };
        let phased: NvCreateChatCompletionRequest = request(true).try_into().unwrap();
        let plain: NvCreateChatCompletionRequest = request(false).try_into().unwrap();
        assert_eq!(
            serde_json::to_value(&phased.inner.messages).unwrap(),
            serde_json::to_value(&plain.inner.messages).unwrap(),
            "model-independent conversion must retain legacy history"
        );
        let python = phased
            .python_chat_processor_request_json(&Default::default())
            .unwrap();
        assert_eq!(
            python["messages"],
            serde_json::to_value(&plain.inner.messages).unwrap()
        );
        assert!(python.get("responses_history_phases").is_none());
        for (tool, reasoning) in [(Some("muse_spark_open"), None), (None, Some("muse_spark"))] {
            let config = crate::local_model::runtime_config::ModelRuntimeConfig {
                tool_call_parser: tool.map(str::to_owned),
                reasoning_parser: reasoning.map(str::to_owned),
                ..Default::default()
            };
            let spark = phased.python_chat_processor_request_json(&config).unwrap();
            assert_eq!(
                spark["responses_history_phases"],
                json!([null, "commentary", null])
            );
            assert_eq!(
                spark["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|m| m["role"] == "assistant")
                    .count(),
                3
            );
            assert_eq!(
                spark["messages"][0]["role"],
                if extra_system { "system" } else { "user" }
            );
        }
        // Even a previously selected phase view cannot leak to another model.
        let mut reused = phased;
        reused.normalize_responses_history_phases(true);
        let other = crate::local_model::runtime_config::ModelRuntimeConfig {
            reasoning_parser: Some("qwen3".into()),
            ..Default::default()
        };
        assert_eq!(
            reused.python_chat_processor_request_json(&other).unwrap()["messages"],
            python["messages"]
        );
    }
}

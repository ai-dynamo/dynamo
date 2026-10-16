// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_renderer::{ChatTemplate, ContextMixins, OAIPromptFormatter, PromptFormatter};
use serde_json::{Value, json};

const TEMPLATE: &str = include_str!("../../../../tests/data/muse_spark_open_deferred_tools.jinja");

fn formatter() -> std::sync::Arc<dyn OAIPromptFormatter> {
    let config: ChatTemplate = serde_json::from_value(json!({"chat_template":TEMPLATE})).unwrap();
    let PromptFormatter::OAI(f) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), true).unwrap();
    f
}

fn client_tools() -> Value {
    json!([{"type":"tool_search", "execution":"client", "description":"Discover tools.",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}])
}

fn request(tools: Value, input: Value) -> NvCreateResponse {
    serde_json::from_value(json!({"model":"/model", "tools":tools, "input":input,
        "chat_template_kwargs":{"current_date":"2026-10-06"}}))
    .unwrap()
}

#[test]
fn client_search_definition_is_rendered_as_a_callable_function() {
    let response = request(client_tools(), json!("Find an order tool."));
    let mut chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
    let f = formatter();
    chat.normalize_responses_tools(f.supports_native_tools(&chat))
        .unwrap();
    assert!(
        chat.inner
            .tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|tool| tool.function.name == "tool_search"))
    );
    let prompt = f.render(&chat).unwrap();
    assert!(prompt.contains("\"name\": \"tool_search\""));
    assert!(prompt.contains("\"query\""));
}

#[test]
fn client_search_output_loads_tools_at_the_history_position() {
    let input = json!([
        {"role":"user","content":"Find an order tool."},
        {"type":"tool_search_call","execution":"client","call_id":"search-1","status":"completed","arguments":{"query":"order lookup"}},
        {"type":"tool_search_output","execution":"client","call_id":"search-1","status":"completed","tools":[
            {"type":"namespace","name":"orders","description":"Orders","tools":[
                {"type":"function","name":"status","defer_loading":true,"parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}}
            ]}
        ]}
    ]);
    let response = request(client_tools(), input);
    let mut chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
    let f = formatter();
    chat.normalize_responses_tools(f.supports_native_tools(&chat))
        .unwrap();
    assert!(
        chat.inner
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .any(|tool| tool.function.name == "orders.status")
    );
    let prompt = f.render(&chat).unwrap();
    assert!(prompt.contains("<|start|>assistant to=tool_search<|message|>"));
    assert!(prompt.contains("<|start|>tool tool_search<|message|>"));
    assert!(prompt.contains("order_id"));
    let first = request(client_tools(), json!("Find an order tool."));
    let mut first_chat: NvCreateChatCompletionRequest = first.try_into().unwrap();
    first_chat
        .normalize_responses_tools(f.supports_native_tools(&first_chat))
        .unwrap();
    let first_prompt = f.render(&first_chat).unwrap();
    let prefix = first_prompt.strip_suffix("<|start|>assistant").unwrap();
    assert!(
        prompt.starts_with(prefix),
        "loading schemas must preserve the earlier prompt prefix"
    );
}

#[test]
fn client_search_unary_output_has_the_dedicated_api_type() {
    let request = request(client_tools(), json!("Find a tool."));
    let params = ResponseParams {
        tools: Some(
            request
                .inner
                .tools
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| t.inner.clone())
                .collect(),
        ),
        template_tools: request.inner.tools.clone(),
        ..Default::default()
    };
    let chat: NvCreateChatCompletionResponse = serde_json::from_value(json!({
        "id":"chat-1","object":"chat.completion","created":0,"model":"/model",
        "choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
            {"id":"search-1","type":"function","function":{"name":"tool_search","arguments":"{\"query\":\"order lookup\"}"}}
        ]},"finish_reason":"tool_calls","logprobs":null}]
    })).unwrap();
    let result =
        serde_json::to_value(chat_completion_to_response(chat, &params, None).unwrap()).unwrap();
    assert_eq!(result["output"][0]["type"], "tool_search_call");
    assert_eq!(result["output"][0]["execution"], "client");
    assert_eq!(result["output"][0]["call_id"], "search-1");
    assert_eq!(
        result["output"][0]["arguments"],
        json!({"query":"order lookup"})
    );
}

#[test]
fn interrupted_client_search_keeps_its_incomplete_status() {
    let request = request(client_tools(), json!("Find a tool."));
    let chat: NvCreateChatCompletionResponse = serde_json::from_value(json!({
        "id":"chat-1","object":"chat.completion","created":0,"model":"/model",
        "choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
            {"id":"search-1","type":"function","function":{"name":"tool_search","arguments":"{\"query\":\"order lookup\"}"}}
        ]},"finish_reason":"length","logprobs":null}]
    })).unwrap();
    let result =
        serde_json::to_value(chat_completion_to_response(chat, &params(&request), None).unwrap())
            .unwrap();
    assert_eq!(result["status"], "incomplete");
    assert_eq!(result["output"][0]["type"], "tool_search_call");
    assert_eq!(result["output"][0]["status"], "incomplete");
}

#[test]
fn commentary_replay_preserves_both_phase_boundaries() {
    let input = json!([
        {"role":"assistant","phase":"commentary","content":"Checking."},
        {"role":"assistant","phase":"final_answer","content":"Done."},
        {"role":"user","content":"Continue."}
    ]);
    let chat: NvCreateChatCompletionRequest = request(json!([]), input).try_into().unwrap();
    let prompt = formatter().render(&chat).unwrap();
    assert!(
        prompt.contains("assistant to=commentary<|message|>Checking.<|eom|>"),
        "{prompt}"
    );
    assert!(
        prompt.contains("assistant to=user<|message|>Done.<|eot|>"),
        "{prompt}"
    );
}

fn raw_chunk(
    text: &str,
    finish: Option<&str>,
) -> super::super::chat_completions::NvCreateChatCompletionStreamResponse {
    serde_json::from_value(
        json!({"id":"chat-1","object":"chat.completion.chunk","created":0,"model":"/model",
        "choices":[{"index":0,"delta":{"content":text},"finish_reason":finish,"logprobs":null}]}),
    )
    .unwrap()
}

async fn stream_json(events: Vec<Result<axum::response::sse::Event, anyhow::Error>>) -> Vec<Value> {
    use axum::response::IntoResponse;
    let response = axum::response::Sse::new(futures::stream::iter(events)).into_response();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

async fn phased_chunks(
    text: &str,
    char_chunks: bool,
) -> Vec<super::super::chat_completions::NvCreateChatCompletionStreamResponse> {
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    let texts = if char_chunks {
        text.chars().map(|c| c.to_string()).collect()
    } else {
        vec![text.to_owned()]
    };
    let mut chunks: Vec<_> = texts
        .iter()
        .map(|text| Annotated::from_data(raw_chunk(text, None)))
        .collect();
    chunks.push(Annotated::from_data(raw_chunk("", Some("stop"))));
    super::super::chat_completions::tool_parser_v2::apply_unified_stream_with_phases(
        futures::stream::iter(chunks),
        None,
        "muse_spark".into(),
        super::super::GuidedToolConstraint::None,
        dynamo_parsers_v2::unified::UnifiedParserStartingState::None,
        false,
        true,
        true,
    )
    .filter_map(|chunk| async move { chunk.data })
    .collect()
    .await
}

#[tokio::test]
async fn commentary_stream_lifecycle_and_unary_order_are_preserved() {
    use super::super::chat_completions::DeltaAggregator;
    use dynamo_runtime::protocols::annotated::Annotated;
    const TURN: &str = concat!(
        "<|start|>assistant to=self<|message|>Private.<|eom|>",
        "<|start|>assistant to=commentary<|message|>Checking.<|eom|>",
        "<|start|>assistant to=weather<|message|><atem:function_calls>",
        "<atem:invoke name=\"weather\"><atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke></atem:function_calls><|eom|>",
        "<|start|>assistant to=commentary<|message|>Found it.<|eom|>",
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    );
    for char_chunks in [false, true] {
        let chunks = phased_chunks(TURN, char_chunks).await;
        let mut converter =
            stream_converter::ResponseStreamConverter::new("/model".into(), Default::default());
        let mut events = converter.emit_start_events();
        for chunk in &chunks {
            events.extend(converter.process_chunk(chunk));
        }
        events.extend(converter.emit_end_events());
        let events = stream_json(events).await;
        let completed = &events.last().unwrap()["response"];
        let output = &completed["output"];
        assert_eq!(output.as_array().unwrap().len(), 4, "{output}");
        assert_eq!(output[0]["phase"], "commentary");
        assert_eq!(output[0]["content"][0]["text"], "Checking.");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[2]["phase"], "commentary");
        assert_eq!(output[3]["phase"], "final_answer");
        for (seq, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], seq as u64);
        }
        let commentary_done = events
            .iter()
            .position(|e| {
                e["type"] == "response.output_item.done" && e["item"]["phase"] == "commentary"
            })
            .unwrap();
        let tool_added = events
            .iter()
            .position(|e| {
                e["type"] == "response.output_item.added" && e["item"]["type"] == "function_call"
            })
            .unwrap();
        assert!(commentary_done < tool_added);
        for index in [0, 2, 3] {
            let id = &output[index]["id"];
            let lifecycle: Vec<_> = events
                .iter()
                .filter(|event| &event["item_id"] == id || &event["item"]["id"] == id)
                .map(|event| event["type"].as_str().unwrap())
                .collect();
            assert_eq!(
                &lifecycle[..2],
                &["response.output_item.added", "response.content_part.added"]
            );
            assert!(
                lifecycle[2..lifecycle.len() - 3]
                    .iter()
                    .all(|event| *event == "response.output_text.delta")
            );
            assert_eq!(
                &lifecycle[lifecycle.len() - 3..],
                &[
                    "response.output_text.done",
                    "response.content_part.done",
                    "response.output_item.done"
                ]
            );
        }

        assert!(!completed.to_string().contains("Private."));
        let unary = DeltaAggregator::apply(
            futures::stream::iter(chunks.into_iter().map(Annotated::from_data)),
            Default::default(),
        )
        .await
        .unwrap();
        let unary = serde_json::to_value(
            chat_completion_to_response(unary, &Default::default(), None).unwrap(),
        )
        .unwrap();
        let unary = unary["output"].as_array().unwrap();
        for index in [0, 2, 3] {
            assert_eq!(unary[index]["phase"], output[index]["phase"]);
            assert_eq!(unary[index]["content"], output[index]["content"]);
        }
        assert_eq!(unary[1]["type"], "function_call");
    }
}

fn params(request: &NvCreateResponse) -> ResponseParams {
    ResponseParams {
        tools: request
            .inner
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(|tool| tool.inner.clone()).collect()),
        template_tools: request.inner.tools.clone(),
        loaded_tools: effective_response_tools(request)
            .unwrap()
            .into_iter()
            .map(|tool| tool.inner)
            .collect(),
        tool_choice: request.inner.tool_choice.clone(),
        ..Default::default()
    }
}

fn tool_chunk(
    name: Option<&str>,
    id: Option<&str>,
    args: &str,
    finish: Option<&str>,
) -> super::super::chat_completions::NvCreateChatCompletionStreamResponse {
    serde_json::from_value(json!({"id":"chat-1","object":"chat.completion.chunk","created":0,"model":"/model",
        "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":id,"type":"function","function":{"name":name,"arguments":args}}]},"finish_reason":finish,"logprobs":null}]})).unwrap()
}

#[tokio::test]
async fn client_search_stream_buffers_object_arguments_and_replays_the_output() {
    let request = request(client_tools(), json!("Find a tool."));
    let mut converter =
        stream_converter::ResponseStreamConverter::new("/model".into(), params(&request));
    let mut events = converter.emit_start_events();
    for chunk in [
        tool_chunk(Some("tool_search"), Some("search-1"), "{\"query\":", None),
        tool_chunk(None, None, "\"order lookup\"}", Some("tool_calls")),
    ] {
        events.extend(converter.process_chunk(&chunk));
    }
    events.extend(converter.emit_end_events());
    let events = stream_json(events).await;
    assert!(!events.iter().any(|event| {
        event["type"]
            .as_str()
            .unwrap()
            .starts_with("response.function_call_arguments")
    }));
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added")
        .unwrap();
    assert_eq!(added["item"]["type"], "tool_search_call");
    let done = events
        .iter()
        .find(|event| event["type"] == "response.output_item.done")
        .unwrap();
    assert_eq!(done["item"]["arguments"], json!({"query":"order lookup"}));
    let result = events.last().unwrap()["response"]["output"][0].clone();
    let replay = super::phase_search_tests::request(
        client_tools(),
        json!([
            {"role":"user","content":"Find a tool."},result,
            {"type":"tool_search_output","call_id":"search-1","tools":[{"type":"namespace","name":"orders","defer_loading":true,"tools":[
                {"type":"function","name":"status","parameters":{"type":"object","properties":{"order_id":{"type":"string"}}}},
                {"type":"custom","name":"patch"}]}]}
        ]),
    );
    let output = params(&replay)
        .tool_output(
            "fc-2".into(),
            "call-2".into(),
            "orders.patch",
            r#"{"input":"*** Begin Patch\n*** End Patch"}"#,
            OutputStatus::Completed,
        )
        .unwrap();
    let output = serde_json::to_value(output).unwrap();
    assert_eq!(output["type"], "custom_tool_call");
    assert_eq!(output["namespace"], "orders");
    let mut chat: NvCreateChatCompletionRequest = replay.try_into().unwrap();
    let f = formatter();
    chat.normalize_responses_tools(f.supports_native_tools(&chat))
        .unwrap();
    let prompt = f.render(&chat).unwrap();
    assert!(prompt.contains("order_id"));
    assert!(
        chat.inner
            .tools
            .unwrap()
            .iter()
            .any(|tool| tool.function.name == "orders.patch")
    );
}

#[tokio::test]
async fn malformed_client_search_arguments_fail_the_response() {
    let request = request(client_tools(), json!("Find a tool."));
    let mut converter =
        stream_converter::ResponseStreamConverter::new("/model".into(), params(&request));
    let mut events = converter.process_chunk(&tool_chunk(
        Some("tool_search"),
        Some("s1"),
        "{\"query\":",
        None,
    ));
    events.extend(converter.emit_end_events());
    let events = stream_json(events).await;
    assert_eq!(events.last().unwrap()["type"], "response.failed");
    assert_eq!(events.last().unwrap()["response"]["output"], json!([]));
}

#[test]
fn ordinary_function_named_tool_search_keeps_function_call_identity() {
    let request = request(
        json!([{"type":"function","name":"tool_search","parameters":{"type":"object","properties":{}}}]),
        json!("Search."),
    );
    let output = params(&request)
        .tool_output(
            "fc-1".into(),
            "call-1".into(),
            "tool_search",
            "{}",
            OutputStatus::Completed,
        )
        .unwrap();
    assert!(matches!(output, OutputItem::FunctionCall(_)));
    let mut tools = client_tools().as_array().unwrap().clone();
    tools.push(json!({"type":"function","name":"tool_search"}));
    assert!(
        effective_response_tools(&super::phase_search_tests::request(
            json!(tools),
            json!("Search.")
        ))
        .is_err()
    );
}

#[test]
fn client_search_replay_requires_matching_unique_call_ids() {
    for input in [
        json!([{"type":"tool_search_output","call_id":"unknown","tools":[]}]),
        json!([{"type":"tool_search_call","execution":"client","arguments":{}}]),
        json!([{"type":"tool_search_call","execution":"client","call_id":"s1","arguments":[]}]),
        json!([{"type":"tool_search_call","execution":"client","call_id":"s1","arguments":{}},
               {"type":"tool_search_output","call_id":"s1","tools":[]},
               {"type":"tool_search_output","call_id":"s1","tools":[]}]),
    ] {
        let error = effective_response_tools(&request(client_tools(), input)).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<ResponsesConversionError>(),
            Some(ResponsesConversionError::InvalidArgument(_))
        ));
    }
    for execution in [Value::Null, json!("server")] {
        let error = effective_response_tools(&request(
            json!([{"type":"tool_search","execution":execution}]),
            json!("Search."),
        ))
        .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<ResponsesConversionError>(),
            Some(ResponsesConversionError::UnsupportedContent(_))
        ));
    }
}

#[tokio::test]
async fn raw_unary_parser_preserves_phase_and_tool_order() {
    use dynamo_runtime::protocols::annotated::Annotated;
    let turn = concat!(
        "<|start|>assistant to=commentary<|message|>Checking.<|eom|>",
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    );
    let chat = super::super::chat_completions::DeltaAggregator::apply(
        futures::stream::iter([Annotated::from_data(raw_chunk(turn, Some("stop")))]),
        super::super::ParsingOptions::new(Some("muse_spark".into()), None),
    )
    .await
    .unwrap();
    let output =
        serde_json::to_value(chat_completion_to_response(chat, &Default::default(), None).unwrap())
            .unwrap();
    assert_eq!(output["output"][0]["phase"], "commentary");
    assert_eq!(output["output"][1]["phase"], "final_answer");
}

#[tokio::test]
async fn interrupted_final_answer_leaves_completed_commentary_intact() {
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    let turn = concat!(
        "<|start|>assistant to=commentary<|message|>Checking.<|eom|>",
        "<|start|>assistant to=user<|message|>Unfinished"
    );
    let chunks: Vec<_> =
        super::super::chat_completions::tool_parser_v2::apply_unified_stream_with_phases(
            futures::stream::iter([Annotated::from_data(raw_chunk(turn, Some("length")))]),
            None,
            "muse_spark".into(),
            super::super::GuidedToolConstraint::None,
            dynamo_parsers_v2::unified::UnifiedParserStartingState::None,
            false,
            true,
            true,
        )
        .filter_map(|chunk| async move { chunk.data })
        .collect()
        .await;
    let mut converter =
        stream_converter::ResponseStreamConverter::new("/model".into(), Default::default());
    let mut events = Vec::new();
    for chunk in &chunks {
        events.extend(converter.process_chunk(chunk));
    }
    events.extend(converter.emit_end_events());
    let events = stream_json(events).await;
    let output = &events.last().unwrap()["response"]["output"];
    assert_eq!(output[0]["status"], "completed");
    assert_eq!(output[1]["status"], "incomplete");
    let chat = super::super::chat_completions::DeltaAggregator::apply(
        futures::stream::iter(chunks.into_iter().map(Annotated::from_data)),
        Default::default(),
    )
    .await
    .unwrap();
    let unary =
        serde_json::to_value(chat_completion_to_response(chat, &Default::default(), None).unwrap())
            .unwrap();
    assert_eq!(unary["output"][0]["status"], "completed");
    assert_eq!(unary["output"][1]["status"], "incomplete");
}

#[test]
fn python_chat_processor_preserves_history_phases_without_changing_worker_wire() {
    let input = json!([
        {"role":"developer","content":"Be helpful."},
        {"role":"assistant","phase":"commentary","content":"Checking."},
        {"type":"function_call","call_id":"call-1","name":"lookup","arguments":"{}"},
        {"type":"function_call_output","call_id":"call-1","output":"Found."},
        {"role":"assistant","phase":"final_answer","content":"Done."},
        {"role":"user","content":"Continue."}
    ]);
    let mut response = request(json!([]), input);
    response.inner.instructions = Some("Follow instructions.".into());
    let chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
    let python = chat.python_chat_processor_request_json().unwrap();
    assert_eq!(
        python["responses_history_phases"],
        json!(["commentary", null, "final_answer"])
    );
    let wire = serde_json::to_value(&chat).unwrap();
    assert!(wire.get("responses_history_phases").is_none());
    assert!(wire.get("responses_tools").is_none());
    assert_eq!(python["messages"], wire["messages"]);
    let mut spoofed = wire;
    spoofed["responses_history_phases"] = json!(["commentary"]);
    let decoded: NvCreateChatCompletionRequest = serde_json::from_value(spoofed).unwrap();
    assert!(decoded.responses_tools.is_none());
    assert!(
        decoded
            .python_chat_processor_request_json()
            .unwrap()
            .get("responses_history_phases")
            .is_none()
    );
}

#[test]
fn phases_are_frontend_only_and_cannot_be_spoofed_on_the_worker_wire() {
    use super::super::chat_completions::MessageBoundary;
    let mut chunk = raw_chunk("Done.", None);
    chunk.message_boundary = Some(MessageBoundary::Start(MessagePhase::FinalAnswer));
    let value = serde_json::to_value(&chunk).unwrap();
    assert!(value.get("message_boundary").is_none());
    let roundtrip: super::super::chat_completions::NvCreateChatCompletionStreamResponse =
        serde_json::from_value(value).unwrap();
    assert!(roundtrip.message_boundary.is_none());
}

#[tokio::test]
async fn forced_search_selection_is_rendered_and_echoed() {
    let mut request = request(client_tools(), json!("Search."));
    request.inner.tool_choice = Some(ToolChoiceParam::ToolSearch);
    let mut chat: NvCreateChatCompletionRequest = request.clone().try_into().unwrap();
    let f = formatter();
    chat.normalize_responses_tools(f.supports_native_tools(&chat))
        .unwrap();
    assert!(
        matches!(chat.inner.tool_choice,Some(ChatCompletionToolChoiceOption::Named(choice)) if choice.function.name=="tool_search")
    );
    let params = params(&request);
    let chat: NvCreateChatCompletionResponse = serde_json::from_value(json!({
        "id":"chat-1","object":"chat.completion","created":0,"model":"/model",
        "choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
            {"id":"s1","type":"function","function":{"name":"tool_search","arguments":"{}"}}]},"finish_reason":"tool_calls","logprobs":null}]
    })).unwrap();
    let unary =
        serde_json::to_value(chat_completion_to_response(chat, &params, None).unwrap()).unwrap();
    assert_eq!(unary["tool_choice"], json!({"type":"tool_search"}));
    let roundtrip: NvResponse = serde_json::from_value(unary.clone()).unwrap();
    assert_eq!(
        serde_json::to_value(roundtrip).unwrap()["tool_choice"],
        unary["tool_choice"]
    );
    let mut converter = stream_converter::ResponseStreamConverter::new("/model".into(), params);
    let mut events = converter.emit_start_events();
    events.extend(converter.process_chunk(&tool_chunk(
        Some("tool_search"),
        Some("s1"),
        "{}",
        Some("tool_calls"),
    )));
    events.extend(converter.emit_end_events());
    for event in stream_json(events).await {
        if !event["response"].is_null() {
            assert_eq!(
                event["response"]["tool_choice"],
                json!({"type":"tool_search"})
            );
        }
    }
}

#[tokio::test]
async fn repeated_and_empty_commentary_messages_have_unique_ids() {
    let turn = concat!(
        "<|start|>assistant to=commentary<|message|><|eom|>",
        "<|start|>assistant to=commentary<|message|>Working.<|eom|>",
        "<|start|>assistant to=commentary<|message|>Still working.<|eom|>",
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    );
    let mut converter =
        stream_converter::ResponseStreamConverter::new("/model".into(), Default::default());
    let mut events = Vec::new();
    for chunk in phased_chunks(turn, true).await {
        events.extend(converter.process_chunk(&chunk));
    }
    events.extend(converter.emit_end_events());
    let events = stream_json(events).await;
    let output = events.last().unwrap()["response"]["output"]
        .as_array()
        .unwrap();
    assert_eq!(output.len(), 4);
    let ids: std::collections::HashSet<_> = output
        .iter()
        .map(|item| item["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 4);
    assert_eq!(output[0]["content"][0]["text"], "");
    assert_eq!(output[1]["content"][0]["text"], "Working.");
    for item in output {
        let done_count = events
            .iter()
            .filter(|event| {
                event["type"] == "response.output_item.done" && event["item"]["id"] == item["id"]
            })
            .count();
        assert_eq!(done_count, 1);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_renderer::{ChatTemplate, ContextMixins, OAIPromptFormatter, PromptFormatter};
use serde_json::{Value, json};

const TEMPLATE: &str = include_str!("../../../../tests/data/muse_spark_open_deferred_tools.jinja");

fn formatter(template: &str) -> std::sync::Arc<dyn OAIPromptFormatter> {
    let config: ChatTemplate = serde_json::from_value(json!({"chat_template":template})).unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), true).unwrap();
    formatter
}

fn request(tools: Value, input: Value) -> NvCreateResponse {
    serde_json::from_value(json!({"model":"/model", "input":input, "tools":tools,
        "chat_template_args":{"current_date":"2026-10-06"}}))
    .unwrap()
}

fn namespace(defer: bool) -> Value {
    json!([{"type":"namespace", "name":"fs", "description":"Files", "defer_loading":defer,
        "tools":[{"type":"function", "name":"read", "description":"Read files", "parameters":{"type":"object", "properties":{"path":{"type":"string"}},"required":["path"]}}]}])
}

fn response_params(request: &NvCreateResponse) -> ResponseParams {
    ResponseParams {
        tools: request
            .inner
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(|tool| tool.inner.clone()).collect()),
        template_tools: request.inner.tools.clone(),
        ..Default::default()
    }
}

fn unary_tool_response(
    params: &ResponseParams,
    name: &str,
    args: Value,
) -> anyhow::Result<NvResponse> {
    let response: NvCreateChatCompletionResponse = serde_json::from_value(json!({
        "id":"chat-1", "object":"chat.completion", "created":0, "model":"/model",
        "choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
            {"id":"call-1","type":"function","function":{"name":name,"arguments":args.to_string()}}
        ]},"finish_reason":"tool_calls","logprobs":null}]
    }))?;
    chat_completion_to_response(response, params, None)
}

#[test]
fn typed_responses_request_preserves_namespace_metadata_in_actual_template() {
    let response = request(namespace(true), json!("Read the file."));
    assert_eq!(
        serde_json::to_value(&response).unwrap()["tools"][0]["defer_loading"],
        true
    );
    let mut chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
    let f = formatter(TEMPLATE);
    chat.normalize_responses_tools(f.supports_native_tools(&chat))
        .unwrap();
    assert_eq!(
        chat.inner.tools.as_ref().unwrap()[0].function.name,
        "fs.read"
    );
    let rendered = f.render(&chat).unwrap();
    assert!(rendered.contains("\"has_deferred_functions\": true"));
    assert!(rendered.contains("\"fs.*\""));
    assert!(!rendered.contains("\"parameters\""));
    chat.chat_template_args
        .as_mut()
        .unwrap()
        .insert("deferred_render_mode".into(), json!("tool_name_only"));
    let rendered = f.render(&chat).unwrap();
    assert!(rendered.contains("\"name\": \"fs.read\""));
    assert!(rendered.contains("\"defer_loading\": true"));
    assert!(!rendered.contains("\"parameters\""));
}

#[test]
fn legacy_templates_retain_eager_bare_names_and_flat_structure() {
    let response = request(namespace(true), json!("Read the file."));
    let mut chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
    let f = formatter("{{ tools | tojson }}");
    chat.normalize_responses_tools(f.supports_native_tools(&chat))
        .unwrap();
    let rendered: Value = serde_json::from_str(&f.render(&chat).unwrap()).unwrap();
    assert_eq!(rendered[0]["function"]["name"], "read");
    assert!(rendered[0]["function"]["parameters"].is_object());
    assert!(rendered[0].get("defer_loading").is_none());
}

#[test]
fn native_names_distinguish_same_leaf_in_different_namespaces() {
    let mut tools = namespace(false);
    let mut second = tools[0].clone();
    second["name"] = json!("store");
    tools.as_array_mut().unwrap().push(second);
    let mut chat: NvCreateChatCompletionRequest =
        request(tools, json!("Read.")).try_into().unwrap();
    chat.normalize_responses_tools(true).unwrap();
    let tools = chat.inner.tools.as_ref().unwrap();
    assert_eq!(tools[0].function.name, "fs.read");
    assert_eq!(tools[1].function.name, "store.read");
    assert!(chat.normalize_responses_tools(false).is_err());
}

#[test]
fn namespaced_history_and_named_choice_follow_selected_template_idempotently() {
    let input = json!([
        {"type":"function_call","call_id":"old-call","name":"read","namespace":"fs","arguments":"{\"path\":\"a.txt\"}"},
        {"type":"function_call_output","call_id":"old-call","output":"Contents"},
        {"role":"user","content":"Read it again."}
    ]);
    let mut response = request(namespace(false), input);
    response.inner.tool_choice =
        Some(serde_json::from_value(json!({"type":"function","name":"fs.read"})).unwrap());
    let mut chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
    for (native, expected) in [(true, "fs.read"), (false, "read"), (true, "fs.read")] {
        chat.normalize_responses_tools(native).unwrap();
        let ChatCompletionRequestMessage::Assistant(message) = &chat.inner.messages[0] else {
            panic!("assistant history missing");
        };
        assert_eq!(
            message.tool_calls.as_ref().unwrap()[0].function.name,
            expected
        );
        let Some(ChatCompletionToolChoiceOption::Named(choice)) = &chat.inner.tool_choice else {
            panic!("named choice missing");
        };
        assert_eq!(choice.function.name, expected);
        let rendered = formatter(if native {
            TEMPLATE
        } else {
            "{{ messages | tojson }}"
        })
        .render(&chat)
        .unwrap();
        if native {
            assert!(rendered.contains("assistant to=fs.read<|message|>"));
            assert!(rendered.contains("<|start|>tool fs.read<|message|>"));
        }
    }
}

#[test]
fn unique_bare_history_is_qualified_without_rewriting_top_level_tools() {
    let input =
        json!([{"type":"function_call","call_id":"old-call","name":"read","arguments":"{}"}]);
    let mut chat: NvCreateChatCompletionRequest =
        request(namespace(false), input.clone()).try_into().unwrap();
    chat.normalize_responses_tools(true).unwrap();
    let ChatCompletionRequestMessage::Assistant(message) = &chat.inner.messages[0] else {
        panic!();
    };
    assert_eq!(
        message.tool_calls.as_ref().unwrap()[0].function.name,
        "fs.read"
    );
    let mut tools = namespace(false);
    tools
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"function","name":"read"}));
    let mut chat: NvCreateChatCompletionRequest = request(tools, input).try_into().unwrap();
    chat.normalize_responses_tools(true).unwrap();
    let ChatCompletionRequestMessage::Assistant(message) = &chat.inner.messages[0] else {
        panic!();
    };
    assert_eq!(
        message.tool_calls.as_ref().unwrap()[0].function.name,
        "read"
    );
}

#[test]
fn namespaced_unary_output_uses_leaf_name_and_namespace_and_echoes_deferral() {
    let response = request(namespace(true), json!("Read."));
    let output =
        unary_tool_response(&response_params(&response), "fs.read", json!({"path":"a"})).unwrap();
    let value = serde_json::to_value(output).unwrap();
    assert_eq!(value["output"][0]["type"], "function_call");
    assert_eq!(value["output"][0]["name"], "read");
    assert_eq!(value["output"][0]["namespace"], "fs");
    assert_eq!(value["tools"][0]["defer_loading"], true);
}

#[test]
fn custom_tool_unary_output_and_history_preserve_verbatim_freeform_payload() {
    let input = "*** Begin Patch\nUnicode: café 日本語\n\"quoted\" \\ backslash\n*** End Patch";
    let tools =
        json!([{"type":"namespace","name":"fs","tools":[{"type":"custom","name":"apply_patch"}]}]);
    let response = request(tools, json!("Apply the patch."));
    let output = unary_tool_response(
        &response_params(&response),
        "fs.apply_patch",
        json!({"input":input}),
    )
    .unwrap();
    let value = serde_json::to_value(output).unwrap();
    assert_eq!(value["output"][0]["type"], "custom_tool_call");
    assert_eq!(value["output"][0]["name"], "apply_patch");
    assert_eq!(value["output"][0]["namespace"], "fs");
    assert_eq!(value["output"][0]["input"], input);
    let mut replay = request(
        value["tools"].clone(),
        json!([
            value["output"][0].clone(),
            {"type":"custom_tool_call_output","call_id":"call-1","output":"Patch applied"},
            {"role":"user","content":"Continue."}
        ]),
    );
    replay.inner.tool_choice =
        Some(serde_json::from_value(json!({"type":"custom","name":"fs.apply_patch"})).unwrap());
    let mut chat: NvCreateChatCompletionRequest = replay.try_into().unwrap();
    chat.normalize_responses_tools(true).unwrap();
    let ChatCompletionRequestMessage::Assistant(message) = &chat.inner.messages[0] else {
        panic!();
    };
    assert_eq!(
        message.tool_calls.as_ref().unwrap()[0].function.name,
        "fs.apply_patch"
    );
    assert_eq!(
        custom_input(&message.tool_calls.as_ref().unwrap()[0].function.arguments).unwrap(),
        input
    );
    let rendered = formatter(TEMPLATE).render(&chat).unwrap();
    assert!(rendered.contains("<atem:parameter name=\"input\">*** Begin Patch"));
    assert!(rendered.contains("<|start|>tool fs.apply_patch<|message|>"));
    assert!(
        unary_tool_response(
            &response_params(&response),
            "fs.apply_patch",
            json!({"input":42})
        )
        .is_err()
    );
}

#[test]
fn plain_functions_retain_the_existing_flat_prompt_view() {
    let response = request(
        json!([{"type":"function","name":"read","parameters":{"type":"object","properties":{}},"strict":true}]),
        json!("Read."),
    );
    let chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
    assert!(chat.native_template_tools().is_none());
    let rendered = formatter("{{ tools | tojson }}").render(&chat).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&rendered).unwrap()[0]["function"]["strict"],
        true
    );
    assert!(
        serde_json::to_value(&chat)
            .unwrap()
            .get("responses_tools")
            .is_none()
    );
}

#[test]
fn responses_validation_accepts_internal_qualified_names_but_chat_keeps_its_name_rules() {
    let chat: NvCreateChatCompletionRequest = request(namespace(false), json!("Read."))
        .try_into()
        .unwrap();
    crate::engines::ValidateRequest::validate(&chat).unwrap();
    let wire_chat: NvCreateChatCompletionRequest = serde_json::from_value(json!({"model":"/model",
        "messages":[{"role":"user","content":"Read."}],
        "tools":[{"type":"function","function":{"name":"fs.read"}}]
    }))
    .unwrap();
    assert!(crate::engines::ValidateRequest::validate(&wire_chat).is_err());
    let mut invalid = namespace(false);
    invalid[0]["name"] = json!("bad.namespace");
    let chat: NvCreateChatCompletionRequest = request(invalid, json!("Read.")).try_into().unwrap();
    assert!(crate::engines::ValidateRequest::validate(&chat).is_err());
}

#[test]
fn native_and_legacy_names_reach_muse_parser_with_verbatim_custom_input() {
    let input = "  *** Begin Patch\n+ café 日本語\n*** End Patch  ";
    let response = request(
        json!([{"type":"namespace","name":"fs","tools":[
            {"type":"custom","name":"apply_patch"}
        ]}]),
        json!("Apply the patch."),
    );
    for (native, name) in [(true, "fs.apply_patch"), (false, "apply_patch")] {
        let mut chat: NvCreateChatCompletionRequest = response.clone().try_into().unwrap();
        chat.normalize_responses_tools(native).unwrap();
        let tools =
            crate::preprocessor::tool_choice::convert_tools(chat.inner.tools.as_deref().unwrap());
        let raw = format!(
            "<|start|>assistant to={name}<|message|><atem:function_calls>\n<atem:invoke name=\"{name}\">\n<atem:parameter name=\"input\">{input}</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>"
        );
        let (calls, _, _) =
            crate::protocols::openai::chat_completions::tool_parser_v2::parse_complete_unified(
                &raw,
                Some(&tools),
                "muse_spark",
            )
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, name);
        assert_eq!(custom_input(&calls[0].function.arguments).unwrap(), input);
        let params = response_params(&response);
        let output = params
            .tool_output(
                "item-1".into(),
                "call-1".into(),
                name,
                &calls[0].function.arguments,
                OutputStatus::Completed,
            )
            .unwrap();
        assert_eq!(serde_json::to_value(output).unwrap()["input"], input);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep Chat custom-tool API identity outside the function-oriented engine path.

use super::super::tools::{custom_input, custom_input_or_raw};
use super::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    NvCreateChatCompletionStreamResponse,
};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCallChunk, CustomToolCall, CustomToolCallStream, FinishReason,
    FunctionType,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use std::collections::{BTreeMap, HashSet};

#[derive(Default)]
pub(crate) struct ChatCustomToolAdapter {
    names: HashSet<String>,
    single_tool_call: bool,
    // Choices may interleave, and tool indices restart independently per choice.
    pending: BTreeMap<(u32, u32), String>,
}

impl ChatCustomToolAdapter {
    pub(crate) fn prepare(request: &mut NvCreateChatCompletionRequest) -> anyhow::Result<Self> {
        Ok(Self {
            names: request
                .inner
                .normalize_custom_tools()?
                .into_iter()
                .collect(),
            single_tool_call: request.inner.parallel_tool_calls == Some(false),
            pending: BTreeMap::new(),
        })
    }

    pub(crate) fn project_response(
        &self,
        response: &mut NvCreateChatCompletionResponse,
    ) -> anyhow::Result<()> {
        if self.names.is_empty() {
            return Ok(());
        }
        for choice in &mut response.inner.choices {
            for call in choice.message.tool_calls.iter_mut().flatten() {
                if self.names.contains(&call.function.name) {
                    let input = custom_input_or_raw(&call.function.arguments);
                    call.custom = Some(CustomToolCall {
                        name: call.function.name.clone(),
                        input,
                    });
                    call.r#type = FunctionType::Custom;
                }
            }
        }
        Ok(())
    }

    fn project_chunk(
        &mut self,
        response: &mut NvCreateChatCompletionStreamResponse,
    ) -> anyhow::Result<()> {
        if self.names.is_empty() {
            return Ok(());
        }
        for choice in &mut response.inner.choices {
            let mut output = Vec::new();
            for mut call in choice.delta.tool_calls.take().into_iter().flatten() {
                // Apply the HTTP single-call cap before validating discarded calls.
                if self.single_tool_call && call.index > 0 {
                    continue;
                }
                let key = (choice.index, call.index);
                let custom_name = call
                    .function
                    .as_ref()
                    .and_then(|f| f.name.as_ref())
                    .filter(|name| self.names.contains(*name))
                    .cloned();
                if custom_name.is_some() {
                    self.pending.entry(key).or_default();
                }
                if let Some(arguments) = self.pending.get_mut(&key) {
                    if let Some(fragment) =
                        call.function.as_ref().and_then(|f| f.arguments.as_ref())
                    {
                        arguments.push_str(fragment);
                    }
                    // The bridge's JSON escapes can cross chunks. Buffer custom input
                    // until complete, then expose text without leaking the wrapper.
                    call.function = None;
                    if custom_name.is_some() || call.id.is_some() {
                        call.r#type = Some(FunctionType::Custom);
                        call.custom = Some(CustomToolCallStream {
                            name: custom_name,
                            input: Some(String::new()),
                        });
                        output.push(call);
                    }
                } else {
                    output.push(call);
                }
            }
            if choice.finish_reason.is_some() {
                let keys: Vec<_> = self
                    .pending
                    .keys()
                    .filter(|(index, _)| *index == choice.index)
                    .copied()
                    .collect();
                for key in keys {
                    let arguments = self
                        .pending
                        .remove(&key)
                        .expect("key belongs to pending call");
                    let input = match custom_input(&arguments) {
                        Ok(input) => input,
                        // Match unary truncation handling before applying the raw-output
                        // fallback: a token limit must not turn a partial call into input.
                        Err(_)
                            if matches!(choice.finish_reason, Some(FinishReason::Length))
                                && serde_json::from_str::<serde_json::Value>(&arguments)
                                    .is_err() =>
                        {
                            output.retain(|call| call.index != key.1);
                            continue;
                        }
                        Err(_) => custom_input_or_raw(&arguments),
                    };
                    // One entry per index: older SDKs read repeats as separate calls.
                    if let Some(opened) = output.iter_mut().find(|call| call.index == key.1) {
                        opened.custom.get_or_insert_with(Default::default).input = Some(input);
                        continue;
                    }
                    output.push(ChatCompletionMessageToolCallChunk {
                        index: key.1,
                        id: None,
                        r#type: None,
                        function: None,
                        custom: Some(CustomToolCallStream {
                            name: None,
                            input: Some(input),
                        }),
                    });
                }
            }
            choice.delta.tool_calls = (!output.is_empty()).then_some(output);
        }
        Ok(())
    }

    pub(crate) fn stream<S>(
        mut self,
        source: S,
    ) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
    where
        S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send,
    {
        async_stream::stream! {
            futures::pin_mut!(source);
            while let Some(mut response) = source.next().await {
                if response.is_error() { yield response; return; }
                if let Some(data) = response.data.as_mut()
                    && let Err(error) = self.project_chunk(data) {
                        yield Annotated::from_error(format!("Invalid custom tool output: {error}"));
                        return;
                    }
                yield response;
            }
            if !self.pending.is_empty() {
                yield Annotated::from_error("Custom tool stream ended before its completion boundary");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ValidateRequest;
    use serde_json::{Value, json};

    fn request() -> NvCreateChatCompletionRequest {
        serde_json::from_value(json!({"model":"test","messages":[{"role":"user","content":"Calculate"}],
            "tools":[{"type":"custom","custom":{"name":"run_python","format":{"type":"text"}}},
                     {"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
            "tool_choice":{"type":"custom","custom":{"name":"run_python"}}})).unwrap()
    }

    fn chunk(choice: u32, calls: Value, finish: Value) -> NvCreateChatCompletionStreamResponse {
        serde_json::from_value(json!({"id":"chat-1","object":"chat.completion.chunk","created":1,"model":"test",
            "choices":[{"index":choice,"delta":{"tool_calls":calls},"finish_reason":finish,"logprobs":null}]})).unwrap()
    }

    #[test]
    fn chat_custom_normalization_retains_schema_choice_and_history() {
        let mut request = request();
        request.inner.messages.extend(serde_json::from_value::<Vec<_>>(json!([
            {"role":"assistant","tool_calls":[{"id":"call1","type":"custom","custom":{"name":"run_python","input":"print(\"日本語\")\n"}}]},
            {"role":"tool","tool_call_id":"call1","content":"日本語"}])).unwrap());
        ValidateRequest::validate(&request).unwrap();
        let adapter = ChatCustomToolAdapter::prepare(&mut request).unwrap();
        assert!(adapter.names.contains("run_python"));
        ValidateRequest::validate(&request).unwrap();
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(
            value["tools"][0]["function"]["parameters"]["properties"]["input"]["type"],
            "string"
        );
        assert_eq!(
            value["tool_choice"],
            json!({"type":"function","function":{"name":"run_python"}})
        );
        assert_eq!(value["messages"][1]["tool_calls"][0]["type"], "function");
        let arguments: Value = serde_json::from_str(
            value["messages"][1]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(arguments["input"], "print(\"日本語\")\n");
    }

    #[test]
    fn chat_custom_unary_response_preserves_text_and_regular_functions() {
        let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
        let input = "print(\"café 日本語\")\n# \\backslash";
        let mut response: NvCreateChatCompletionResponse = serde_json::from_value(json!({
            "id":"chat-1","object":"chat.completion","created":1,"model":"test",
            "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"run_python","arguments":json!({"input":input}).to_string()}},
                {"id":"c2","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls","logprobs":null}]})).unwrap();
        adapter.project_response(&mut response).unwrap();
        let value = serde_json::to_value(response).unwrap();
        assert_eq!(
            value["choices"][0]["message"]["tool_calls"][0],
            json!({"id":"c1","type":"custom","custom":{"name":"run_python","input":input}})
        );
        assert_eq!(
            value["choices"][0]["message"]["tool_calls"][1]["type"],
            "function"
        );
    }

    #[tokio::test]
    async fn chat_custom_stream_decodes_every_split_and_isolates_choices() {
        let input = "print(\"café 日本語\")\n# \\backslash";
        let arguments = json!({"input":input}).to_string();
        for split in arguments
            .char_indices()
            .map(|(i, _)| i)
            .chain([arguments.len()])
        {
            let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
            let source = vec![
                chunk(
                    0,
                    json!([{"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":&arguments[..split]}}]),
                    json!(null),
                ),
                chunk(
                    1,
                    json!([{"index":0,"id":"c2","type":"function","function":{"name":"lookup","arguments":"{}"}}]),
                    json!("tool_calls"),
                ),
                chunk(
                    0,
                    json!([{"index":0,"function":{"arguments":&arguments[split..]}}]),
                    json!(null),
                ),
                chunk(0, json!(null), json!("tool_calls")),
            ];
            let out: Vec<_> = adapter
                .stream(futures::stream::iter(
                    source.into_iter().map(Annotated::from_data),
                ))
                .collect()
                .await;
            assert!(out.iter().all(|item| !item.is_error()));
            let values: Vec<_> = out
                .into_iter()
                .map(|item| serde_json::to_value(item.data.unwrap()).unwrap())
                .collect();
            assert_eq!(
                values[0]["choices"][0]["delta"]["tool_calls"][0]["type"],
                "custom"
            );
            assert_eq!(
                values[1]["choices"][0]["delta"]["tool_calls"][0]["type"],
                "function"
            );
            assert_eq!(
                values[3]["choices"][0]["delta"]["tool_calls"][0]["custom"]["input"],
                input
            );
            for (index, value) in values.iter().enumerate().filter(|(i, _)| *i != 1) {
                for call in value["choices"][0]["delta"]["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                {
                    assert!(call.get("function").is_none(), "chunk {index}");
                }
            }
        }
    }

    #[test]
    fn chat_custom_unary_malformed_output_preserves_siblings_text_and_usage() {
        let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
        for arguments in [
            r#"{"code":"print(1)"}"#,
            r#"{"input":42}"#,
            "print(1)\n",
            "",
        ] {
            let mut response: NvCreateChatCompletionResponse = serde_json::from_value(json!({
                "id":"chat-1","object":"chat.completion","created":1,"model":"test",
                "choices":[{"index":0,"message":{"role":"assistant","content":"Checking.","tool_calls":[
                    {"id":"c1","type":"function","function":{"name":"run_python","arguments":arguments}},
                    {"id":"c2","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls","logprobs":null}],
                "usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}})).unwrap();
            adapter.project_response(&mut response).unwrap();
            let value = serde_json::to_value(response).unwrap();
            assert_eq!(value["choices"][0]["message"]["content"], "Checking.");
            assert_eq!(
                value["choices"][0]["message"]["tool_calls"][0],
                json!({"id":"c1","type":"custom","custom":{"name":"run_python","input":arguments}})
            );
            assert_eq!(
                value["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"],
                "{}"
            );
            assert_eq!(value["usage"]["total_tokens"], 7);
        }
    }

    #[tokio::test]
    async fn chat_custom_stream_malformed_output_preserves_siblings_text_and_usage() {
        for arguments in [
            r#"{"code":"print(1)"}"#,
            r#"{"input":42}"#,
            "print(1)\n",
            "",
        ] {
            let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
            let mut terminal = chunk(0, json!(null), json!("tool_calls"));
            terminal.inner.choices[0].delta.content = Some(
                dynamo_protocols::types::ChatCompletionMessageContent::Text("Checking.".to_owned()),
            );
            terminal.inner.usage = Some(
                serde_json::from_value(
                    json!({"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}),
                )
                .unwrap(),
            );
            let out: Vec<_> = adapter.stream(futures::stream::iter([
                Annotated::from_data(chunk(0, json!([
                    {"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":arguments}},
                    {"index":1,"id":"c2","type":"function","function":{"name":"lookup","arguments":"{}"}}
                ]), json!(null))),
                Annotated::from_data(terminal),
            ])).collect().await;
            assert!(out.iter().all(|item| !item.is_error()), "{arguments:?}");
            let value = serde_json::to_value(out[0].data.as_ref().unwrap()).unwrap();
            assert_eq!(
                value["choices"][0]["delta"]["tool_calls"][1]["function"]["arguments"],
                "{}"
            );
            let value = serde_json::to_value(out.last().unwrap().data.as_ref().unwrap()).unwrap();
            assert_eq!(
                value["choices"][0]["delta"]["tool_calls"][0]["custom"]["input"],
                arguments
            );
            assert_eq!(value["choices"][0]["delta"]["content"], "Checking.");
            assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(value["usage"]["total_tokens"], 7);
        }
    }

    #[tokio::test]
    async fn chat_custom_stream_rejects_missing_terminal() {
        let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
        let out: Vec<_> = adapter.stream(futures::stream::iter([Annotated::from_data(chunk(0,
            json!([{"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":"{\"input\":\"x\"}"}}]),
            json!(null)))])).collect().await;
        assert!(out.last().unwrap().is_error());
    }

    #[tokio::test]
    async fn chat_custom_stream_aggregates_multiple_freeform_calls() {
        let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
        let source = vec![
            chunk(
                0,
                json!([
                    {"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":"{\"input\":\"print(1)\"}"}},
                    {"index":1,"id":"c2","type":"function","function":{"name":"run_python","arguments":"{\"input\":\"print(2)\"}"}}
                ]),
                json!(null),
            ),
            chunk(0, json!(null), json!("tool_calls")),
        ];
        let response = super::super::DeltaAggregator::apply(
            adapter.stream(futures::stream::iter(
                source.into_iter().map(Annotated::from_data),
            )),
            crate::protocols::openai::ParsingOptions::default(),
        )
        .await
        .unwrap();
        let value = serde_json::to_value(response).unwrap();
        let calls = value["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["custom"]["input"], "print(1)");
        assert_eq!(calls[1]["custom"]["input"], "print(2)");
    }

    #[tokio::test]
    async fn chat_custom_single_call_cap_discards_malformed_extra_call() {
        let mut request = request();
        request.inner.parallel_tool_calls = Some(false);
        let adapter = ChatCustomToolAdapter::prepare(&mut request).unwrap();
        let out: Vec<_> = adapter.stream(futures::stream::iter([Annotated::from_data(chunk(0,json!([
            {"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":"{\"input\":\"print(1)\"}"}},
            {"index":1,"id":"c2","type":"function","function":{"name":"run_python","arguments":"invalid"}}
        ]),json!("tool_calls")))])).collect().await;
        assert!(out.iter().all(|item| !item.is_error()));
        let value = serde_json::to_value(out[0].data.as_ref().unwrap()).unwrap();
        assert!(
            value["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .unwrap()
                .iter()
                .all(|call| call["index"] == 0)
        );
    }

    #[tokio::test]
    async fn chat_custom_stream_length_preserves_complete_input_and_drops_truncated_tail() {
        for (arguments, expected) in [
            (r#"{"input":"print(1)"}"#, Some("print(1)")),
            (r#"{"code":"print(1)"}"#, Some(r#"{"code":"print(1)"}"#)),
            (r#"{"input":"print("#, None),
        ] {
            let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
            let out: Vec<_> = adapter.stream(futures::stream::iter([Annotated::from_data(chunk(
                0, json!([{"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":arguments}}]),
                json!("length"),
            ))])).collect().await;
            assert!(out.iter().all(|item| !item.is_error()));
            let value = serde_json::to_value(out[0].data.as_ref().unwrap()).unwrap();
            assert_eq!(value["choices"][0]["finish_reason"], "length");
            if let Some(input) = expected {
                assert_eq!(
                    value["choices"][0]["delta"]["tool_calls"]
                        .as_array()
                        .unwrap()
                        .len(),
                    1
                );
                assert_eq!(
                    value["choices"][0]["delta"]["tool_calls"][0]["custom"]["input"],
                    input
                );
            } else {
                assert!(value["choices"][0]["delta"]["tool_calls"].is_null());
            }
        }
    }

    #[tokio::test]
    async fn chat_custom_stream_length_stop_drops_truncated_input() {
        let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
        let out: Vec<_> = adapter
            .stream(futures::stream::iter([
                Annotated::from_data(chunk(
                    0,
                    json!([{"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":"{\"input\":\"print("}}]),
                    json!(null),
                )),
                Annotated::from_data(chunk(0, json!(null), json!("length"))),
            ]))
            .collect()
            .await;
        // The unary aggregator drops a truncated call; streaming keeps the length stop.
        assert!(out.iter().all(|item| !item.is_error()));
        let last = serde_json::to_value(out.last().unwrap().data.as_ref().unwrap()).unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert!(last["choices"][0]["delta"]["tool_calls"].is_null());
    }

    #[tokio::test]
    async fn chat_custom_stream_call_completed_in_terminal_chunk_uses_one_entry() {
        let adapter = ChatCustomToolAdapter::prepare(&mut request()).unwrap();
        let out: Vec<_> = adapter
            .stream(futures::stream::iter([Annotated::from_data(chunk(
                0,
                json!([{"index":0,"id":"c1","type":"function","function":{"name":"run_python","arguments":"{\"input\":\"print(1)\"}"}}]),
                json!("tool_calls"),
            ))]))
            .collect()
            .await;
        let value = serde_json::to_value(out[0].data.as_ref().unwrap()).unwrap();
        // Older OpenAI SDK accumulators treat repeated indexes in one delta as separate calls.
        assert_eq!(
            value["choices"][0]["delta"]["tool_calls"],
            json!([{"index":0,"id":"c1","type":"custom","custom":{"name":"run_python","input":"print(1)"}}])
        );
    }

    #[test]
    fn chat_custom_rejects_grammar_and_wrong_choice_type_before_generation() {
        for change in [
            json!({"format":{"type":"grammar","grammar":{"syntax":"regex","definition":".*"}}}),
            json!({}),
        ] {
            let mut value = serde_json::to_value(request()).unwrap();
            if let Some(format) = change.get("format") {
                value["tools"][0]["custom"]["format"] = format.clone();
            } else {
                value["tool_choice"] = json!({"type":"function","function":{"name":"run_python"}});
            }
            let mut request = serde_json::from_value(value).unwrap();
            assert!(ChatCustomToolAdapter::prepare(&mut request).is_err());
        }
    }
}

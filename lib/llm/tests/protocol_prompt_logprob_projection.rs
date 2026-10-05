// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::{
    Annotated,
    common::{FinishReason, llm_backend::BackendOutput},
    openai::{
        DeltaGeneratorExt, ParsingOptions,
        chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
            NvCreateChatCompletionStreamResponse, aggregator::ChatCompletionAggregator,
        },
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use serde_json::{Value, json};

fn payload() -> Value {
    // Prompt projection must preserve a finite value below the generated-token
    // floor, as well as the worker's already-normalized negative-infinity value.
    json!([null, {
        "0": {"logprob": -0.25, "rank": 1, "decoded_token": "!"},
        "1": {"logprob": -10000.0, "rank": 3, "decoded_token": "finite"},
        "2": {"logprob": -9999.0, "rank": 2, "decoded_token": "sentinel"}
    }])
}

fn backend(payload: Value, finished: bool) -> BackendOutput {
    let mut output: BackendOutput = serde_json::from_value(json!({
        "token_ids": [0], "tokens": ["!"], "text": "!",
        "engine_data": {"prompt_logprobs": payload}
    }))
    .unwrap();
    output.finish_reason = finished.then_some(FinishReason::Length);
    output.index = Some(0);
    output
}

#[tokio::test]
async fn prompt_and_generated_probabilities_keep_distinct_normalization() {
    let mut output = backend(payload(), true);
    output.log_probs = Some(vec![-10000.0]);
    let chat: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "projection-test", "messages": [], "prompt_logprobs": 1,
        "logprobs": true, "top_logprobs": 1,
        "nvext": {"extra_fields": ["prompt_logprobs"]}
    }))
    .unwrap();
    let delta = chat
        .response_generator("floor".into())
        .choice_from_postprocessor(output.clone())
        .unwrap();
    let response = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter([Annotated::from_data(delta)]),
        ParsingOptions::default(),
    )
    .await
    .unwrap();
    let public = serde_json::to_value(response).unwrap();
    assert_eq!(
        public["choices"][0]["logprobs"]["content"][0]["logprob"],
        -9999.0
    );
    assert_eq!(public["prompt_logprobs"], payload());
    assert_eq!(public["nvext"]["prompt_logprobs"], payload());

    let completion: NvCreateCompletionRequest = serde_json::from_value(json!({
        "model": "projection-test", "prompt": "test", "prompt_logprobs": 1,
        "logprobs": 1, "nvext": {"extra_fields": ["prompt_logprobs"]}
    }))
    .unwrap();
    let delta = completion
        .response_generator("floor".into())
        .choice_from_postprocessor(output)
        .unwrap();
    let response = NvCreateCompletionResponse::from_annotated_stream(
        futures::stream::iter([Annotated::from_data(delta)]),
        ParsingOptions::default(),
    )
    .await
    .unwrap();
    let public = serde_json::to_value(response).unwrap();
    assert_eq!(
        public["choices"][0]["logprobs"]["token_logprobs"][0],
        -9999.0
    );
    assert_eq!(public["choices"][0]["prompt_logprobs"], payload());
    assert_eq!(public["nvext"]["prompt_logprobs"], payload());
}

#[tokio::test]
async fn chat_prompt_payload_is_unary_only_and_independent_of_nvext() {
    for count in [Value::Null, json!(0), json!(1)] {
        for nvext in [false, true] {
            for stream in [false, true] {
                let mut body = json!({
                    "model": "projection-test", "messages": [],
                    "prompt_logprobs": count, "stream": stream
                });
                if nvext {
                    body["nvext"] = json!({"extra_fields": ["prompt_logprobs"]});
                }
                let request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();
                let mut generator = request.response_generator("test".into());
                let first = generator
                    .choice_from_postprocessor(backend(payload(), false))
                    .unwrap();
                let last = generator
                    .choice_from_postprocessor(backend(payload(), true))
                    .unwrap();
                for delta in [&first, &last] {
                    let public = serde_json::to_value(delta).unwrap();
                    assert!(public.get("internal_prompt_logprobs").is_none());
                    assert!(public.get("prompt_logprobs").is_none());
                    assert_eq!(delta.internal_prompt_logprobs.is_some(), !count.is_null());
                }
                assert!(
                    first
                        .nvext
                        .as_ref()
                        .and_then(|v| v.get("prompt_logprobs"))
                        .is_none()
                );
                assert_eq!(
                    last.nvext
                        .as_ref()
                        .and_then(|v| v.get("prompt_logprobs"))
                        .is_some(),
                    nvext
                );
                let response = NvCreateChatCompletionResponse::from_annotated_stream(
                    futures::stream::iter([
                        Annotated::from_data(first),
                        Annotated::from_data(last),
                    ]),
                    ParsingOptions::default(),
                )
                .await
                .unwrap();
                let public = serde_json::to_value(response).unwrap();
                assert_eq!(
                    public.get("prompt_logprobs"),
                    (!count.is_null()).then(payload).as_ref()
                );
                assert_eq!(
                    public.get("nvext").and_then(|v| v.get("prompt_logprobs")),
                    nvext.then(payload).as_ref()
                );
            }
        }
    }
}

#[test]
fn requested_malformed_prompt_data_is_an_error_not_an_omission() {
    for count in [Value::Null, json!(0), json!(1)] {
        for nvext in [false, true] {
            let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "projection-test", "messages": [], "prompt_logprobs": count,
                "nvext": {"extra_fields": if nvext {vec!["prompt_logprobs"]} else {vec![]}}
            }))
            .unwrap();
            let mut generator = request.response_generator("malformed".into());
            let result =
                generator.choice_from_postprocessor(backend(json!("private-invalid-data"), true));
            assert_eq!(result.is_err(), nvext || !count.is_null());
            if let Err(error) = result {
                assert_eq!(
                    error.to_string(),
                    "Malformed backend prompt_logprobs payload"
                );
            }
            let request: NvCreateCompletionRequest = serde_json::from_value(json!({
                "model": "projection-test", "prompt": "test", "prompt_logprobs": count,
                "nvext": {"extra_fields": if nvext {vec!["prompt_logprobs"]} else {vec![]}}
            }))
            .unwrap();
            let mut generator = request.response_generator("malformed-completion".into());
            let result =
                generator.choice_from_postprocessor(backend(json!("private-invalid-data"), true));
            assert_eq!(result.is_err(), nvext || !count.is_null());
            if let Err(error) = result {
                assert_eq!(
                    error.to_string(),
                    "Malformed backend prompt_logprobs payload"
                );
            }
        }
    }
}

#[tokio::test]
async fn completion_prompt_payload_is_choice_scoped_and_unary_only() {
    for count in [Value::Null, json!(0), json!(1)] {
        for nvext in [false, true] {
            for stream in [false, true] {
                let request: NvCreateCompletionRequest = serde_json::from_value(json!({
                    "model": "projection-test", "prompt": "test",
                    "prompt_logprobs": count, "stream": stream,
                    "nvext": {"extra_fields": if nvext {vec!["prompt_logprobs"]} else {vec![]}}
                }))
                .unwrap();
                let mut generator = request.response_generator("completion".into());
                let first = generator
                    .choice_from_postprocessor(backend(payload(), false))
                    .unwrap();
                // The worker may emit prompt data once, before the terminal chunk.
                let mut last_output = backend(Value::Null, true);
                last_output.engine_data = None;
                let last = generator.choice_from_postprocessor(last_output).unwrap();
                for delta in [&first, &last] {
                    let public = serde_json::to_value(delta).unwrap();
                    assert!(public.get("prompt_logprobs").is_none());
                    assert!(public["choices"][0].get("prompt_logprobs").is_none());
                    assert!(
                        public["choices"][0]
                            .get("internal_prompt_logprobs")
                            .is_none()
                    );
                }
                assert_eq!(
                    first.inner.choices[0].internal_prompt_logprobs.is_some(),
                    !count.is_null()
                );
                assert!(last.inner.choices[0].internal_prompt_logprobs.is_none());
                let response = NvCreateCompletionResponse::from_annotated_stream(
                    futures::stream::iter([
                        Annotated::from_data(first),
                        Annotated::from_data(last),
                    ]),
                    ParsingOptions::default(),
                )
                .await
                .unwrap();
                let public = serde_json::to_value(response).unwrap();
                assert_eq!(
                    public["choices"][0].get("prompt_logprobs"),
                    (!count.is_null()).then(payload).as_ref()
                );
                assert!(public.get("prompt_logprobs").is_none());
                assert!(
                    public["choices"][0]
                        .get("internal_prompt_logprobs")
                        .is_none()
                );
            }
        }
    }
}

#[tokio::test]
async fn completion_prompt_payloads_stay_with_remapped_choices() {
    let request: NvCreateCompletionRequest = serde_json::from_value(json!({
        "model": "projection-test", "prompt": "test", "prompt_logprobs": 0
    }))
    .unwrap();
    let mut chunks = Vec::new();
    for index in [2, 0, 1] {
        let mut generator = request.response_generator("batch".into());
        let mut response = generator.choice_from_postprocessor(backend(
            json!([null, {"0": {"logprob": -0.25, "rank": 1, "decoded_token": index.to_string()}}]),
            true,
        )).unwrap();
        // Same choice-index remapping used by the HTTP batched-prompt handler.
        response.inner.choices[0].index = index;
        chunks.push(Annotated::from_data(response));
    }
    let response = NvCreateCompletionResponse::from_annotated_stream(
        futures::stream::iter(chunks),
        ParsingOptions::default(),
    )
    .await
    .unwrap();
    let public = serde_json::to_value(response).unwrap();
    for index in 0..3 {
        assert_eq!(public["choices"][index]["index"], index);
        assert_eq!(
            public["choices"][index]["prompt_logprobs"][1]["0"]["decoded_token"],
            index.to_string()
        );
    }
}

#[test]
fn plain_completion_wire_shape_is_unchanged_and_public_prompt_payload_roundtrips() {
    let original: dynamo_protocols::types::CreateCompletionResponse =
        serde_json::from_value(json!({
            "id": "plain", "model": "test", "created": 0, "object": "text_completion",
            "choices": [{"index": 0, "text": "hello"}]
        }))
        .unwrap();
    let expected = serde_json::to_value(&original).unwrap();
    let converted = NvCreateCompletionResponse {
        inner: original.into(),
        nvext: None,
    };
    assert_eq!(serde_json::to_value(converted).unwrap(), expected);
    let mut with_prompt = expected;
    with_prompt["choices"][0]["prompt_logprobs"] = payload();
    let restored: NvCreateCompletionResponse = serde_json::from_value(with_prompt.clone()).unwrap();
    assert_eq!(serde_json::to_value(restored).unwrap(), with_prompt);
    let restored: NvCreateCompletionResponse =
        serde_json::from_str(&with_prompt.to_string()).unwrap();
    assert_eq!(serde_json::to_value(restored).unwrap(), with_prompt);
    for key in ["-1", "4294967296", "1.5", "true", "token"] {
        let mut invalid = with_prompt.clone();
        invalid["choices"][0]["prompt_logprobs"] = json!([null, {key: {"logprob": -0.25}}]);
        assert!(serde_json::from_value::<NvCreateCompletionResponse>(invalid).is_err());
    }
    let mut invalid = with_prompt;
    invalid["choices"][0]["prompt_logprobs"] =
        json!([null, {"0": {"logprob": -0.25}, "00": {"logprob": -0.5}}]);
    assert!(serde_json::from_value::<NvCreateCompletionResponse>(invalid).is_err());
}

#[tokio::test]
async fn completion_legacy_nvext_can_coexist_with_native_unary_prompt_payload() {
    let request: NvCreateCompletionRequest = serde_json::from_value(json!({
        "model": "projection-test", "prompt": "test", "prompt_logprobs": 0,
        "nvext": {"extra_fields": ["prompt_logprobs"]}
    }))
    .unwrap();
    let mut generator = request.response_generator("legacy".into());
    let response = generator
        .choice_from_postprocessor(backend(payload(), true))
        .unwrap();
    let stream = serde_json::to_value(&response).unwrap();
    assert_eq!(stream["nvext"]["prompt_logprobs"], payload());
    assert!(stream["choices"][0].get("prompt_logprobs").is_none());
    let aggregated = NvCreateCompletionResponse::from_annotated_stream(
        futures::stream::iter([Annotated::from_data(response)]),
        ParsingOptions::default(),
    )
    .await
    .unwrap();
    let unary = serde_json::to_value(aggregated).unwrap();
    assert_eq!(unary["nvext"]["prompt_logprobs"], payload());
    assert_eq!(unary["choices"][0]["prompt_logprobs"], payload());
}

#[tokio::test]
async fn python_bridge_internal_field_reaches_unary_but_not_public_stream() {
    let delta: NvCreateChatCompletionStreamResponse = serde_json::from_value(json!({
        "id": "python-test", "model": "projection-test", "created": 0,
        "object": "chat.completion.chunk", "choices": [],
        "internal_prompt_logprobs": payload()
    }))
    .unwrap();
    let public_stream = serde_json::to_value(&delta).unwrap();
    assert!(public_stream.get("internal_prompt_logprobs").is_none());
    assert!(public_stream.get("prompt_logprobs").is_none());
    let response = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter([Annotated::from_data(delta)]),
        ParsingOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        serde_json::to_value(response).unwrap()["prompt_logprobs"],
        payload()
    );
}

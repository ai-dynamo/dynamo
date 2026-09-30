// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP regressions for frontend-crates#299 compatibility.
use dynamo_llm::{
    http::service::service_v2::HttpService,
    model_card::ModelDeploymentCard,
    protocols::{
        Annotated,
        openai::completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
};
use dynamo_runtime::{
    CancellationToken,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
    },
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;
use http_harness::{MODEL, load_agent_fixture};

#[derive(Default)]
struct CompletionEngine(Mutex<Vec<Value>>);
#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        anyhow::Error,
    > for CompletionEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, anyhow::Error> {
        let (request, context) = request.transfer(());
        self.0.lock().unwrap().push(serde_json::to_value(request)?);
        let response = serde_json::from_value(json!({
            "id": "cmpl-test", "object": "text_completion", "created": 1000000000, "model": MODEL,
            "choices": [{"index": 0, "text": "Pong.", "finish_reason": "stop", "logprobs": null}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }))?;
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(response)])),
            context.context(),
        ))
    }
}

#[tokio::test]
async fn issue_299_http_compatibility() {
    temp_env::async_with_vars([
        ("DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS", Some("0")),
        ("DYN_ENABLE_FORCE_INCLUDE_USAGE", Some("false")),
    ], async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let chat = Arc::new(scripted_chat_engine::ScriptedChatEngine::new((0..32).map(|_| Ok(script.clone()))));
        let completion = Arc::new(CompletionEngine::default());
        let (listener, port) = ports::bind_random_port().await;
        let service = HttpService::builder().port(port).host("127.0.0.1")
            .enable_chat_endpoints(true).enable_cmpl_endpoints(true).enable_responses_endpoints(true).build().unwrap();
        let card = ModelDeploymentCard::with_name_only(MODEL);
        service.model_manager().add_chat_completions_model(MODEL, card.mdcsum(), chat.clone()).unwrap();
        service.model_manager().add_completions_model(MODEL, card.mdcsum(), completion.clone()).unwrap();
        let cancel = CancellationToken::new();
        let task = service.spawn_with_listener(cancel.clone(), listener).await;
        let client = reqwest::Client::builder().no_proxy().timeout(std::time::Duration::from_secs(10)).build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        for (text, expected_format) in [
            (json!({"verbosity": "low"}), Value::Null),
            (json!({"verbosity": "high", "format": {"type": "json_object"}}), json!({"type": "json_object"})),
            (json!({"verbosity": "high", "format": {"type": "json_schema", "name": "answer", "strict": true, "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": false}}}), json!({"type": "json_schema"})),
        ] {
            let response = client.post(format!("{base}/v1/responses")).json(&json!({"model": MODEL, "input": "Say hello", "text": text})).send().await.unwrap();
            assert_eq!(response.status(), 200, "{text}");
            let output: Value = response.json().await.unwrap();
            assert_eq!(output["usage"]["total_tokens"], 7);
            let requests = chat.take_requests().await;
            assert_eq!(requests.len(), 1);
            let wire = serde_json::to_value(&requests[0]).unwrap();
            assert!(wire.get("text").is_none());
            assert!(wire.get("verbosity").is_none());
            assert_eq!(wire["response_format"]["type"], expected_format["type"]);
            if text["format"]["type"] == "json_schema" {
                assert_eq!(wire["response_format"]["json_schema"]["schema"], text["format"]["schema"]);
            }
            println!("Responses {text}: HTTP 200, format mapped, text/verbosity absent from dispatched request, usage=7");
        }
        for endpoint in ["chat/completions", "completions"] {
            let mut body = json!({"model": MODEL});
            if endpoint == "chat/completions" { body["messages"] = json!([{"role": "user", "content": "Say hello"}]); }
            else { body["prompt"] = json!("Say hello"); }
            for stream in [None, Some(json!(null)), Some(json!(false)), Some(json!(true))] {
                body.as_object_mut().unwrap().remove("stream");
                if let Some(flag) = &stream { body["stream"] = flag.clone(); }
                for include_usage in [false, true] {
                    let options = json!({"include_usage": include_usage, "continuous_usage_stats": true});
                    body["stream_options"] = options.clone();
                    let response = client.post(format!("{base}/v1/{endpoint}")).json(&body).send().await.unwrap();
                    let streaming = stream == Some(json!(true));
                    assert_eq!(response.status().as_u16(), 200, "{body}");
                    if streaming { assert!(response.text().await.unwrap().contains("[DONE]")); }
                    else {
                        let output: Value = response.json().await.unwrap();
                        assert_eq!(output["usage"]["total_tokens"], 7);
                    }
                    let captured: Vec<Value> = if endpoint == "chat/completions" {
                        chat.take_requests().await.iter().map(|r| serde_json::to_value(r).unwrap()).collect()
                    } else { std::mem::take(&mut *completion.0.lock().unwrap()) };
                    assert_eq!(captured.len(), 1);
                    if streaming { assert_eq!(captured[0]["stream_options"], options); }
                    else { assert!(captured[0].get("stream_options").is_none()); }
                    println!("{endpoint} stream={stream:?}, include_usage={include_usage}: HTTP {}, dispatches={}", 200, captured.len());
                }
                if stream != Some(json!(true)) {
                    body.as_object_mut().unwrap().remove("stream_options");
                    let response = client.post(format!("{base}/v1/{endpoint}")).json(&body).send().await.unwrap();
                    assert_eq!(response.status(), 200);
                    let output: Value = response.json().await.unwrap();
                    assert_eq!(output["usage"]["total_tokens"], 7);
                    if endpoint == "chat/completions" { assert_eq!(chat.take_requests().await.len(), 1); }
                    else { assert_eq!(std::mem::take(&mut *completion.0.lock().unwrap()).len(), 1); }
                    println!("{endpoint} stream={stream:?}, options removed control: HTTP 200, usage=7");
                }
            }
        }
        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(3), task).await.unwrap().unwrap().unwrap();
    }).await;
}

#[test]
fn issue_299_internal_usage_after_ingress_options_are_cleared() {
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    let mut chat: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL, "messages": [{"role": "user", "content": "Say hello"}]
    }))
    .unwrap();
    let mut completion: NvCreateCompletionRequest = serde_json::from_value(json!({
        "model": MODEL, "prompt": "Say hello"
    }))
    .unwrap();
    // These are the same helpers called by each production preprocessor.
    chat.enable_usage_for_nonstreaming(false);
    completion.enable_usage_for_nonstreaming(false);
    for options in [chat.inner.stream_options, completion.inner.stream_options] {
        let options = options.unwrap();
        assert!(options.include_usage);
        assert!(!options.continuous_usage_stats);
    }
}

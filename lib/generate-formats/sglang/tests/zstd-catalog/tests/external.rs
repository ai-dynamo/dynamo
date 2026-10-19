// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_llm::generate_format::{Registry, SharedFormat};
use dynamo_llm::protocols::{
    Annotated,
    common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};
use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Error, ManyOut, SingleIn};
use dynamo_sglang_generate::testkit::laws::Harness;
use serde_json::{Value, json};

fn custom() -> SharedFormat {
    let mut registry = Registry::default();
    custom_generate_format_catalog::register(&mut registry).unwrap();
    registry.get("sglang-zstd").unwrap()
}

#[test]
fn backend_property_laws_apply_to_the_external_codec() {
    Harness::native(custom()).check_all();
}

#[test]
fn catalog_installs_builtins_and_selects_external_format() {
    // A premature lookup must not poison installation with an empty registry.
    assert!(dynamo_llm::generate_format::named("sglang-zstd").is_err());
    let mut registry = Registry::default();
    custom_generate_format_catalog::register(&mut registry).unwrap();
    assert_eq!(registry.get("sglang").unwrap().name(), "sglang");
    assert_eq!(registry.get("vllm").unwrap().name(), "vllm");
    assert!(custom_generate_format_catalog::register(&mut registry).is_err());
    assert!(dynamo_llm::generate_format::install(registry.clone()));
    assert!(!dynamo_llm::generate_format::install(registry));
    let selected = std::env::var("DYN_GENERATE_FORMAT").unwrap_or_else(|_| "sglang".into());
    assert_eq!(
        dynamo_llm::generate_format::selected().unwrap().name(),
        selected
    );
    assert_eq!(
        dynamo_llm::generate_format::named("sglang-zstd")
            .unwrap()
            .name(),
        "sglang-zstd"
    );
}

struct Worker(Value);
#[async_trait::async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for Worker
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        assert_eq!(request.content().token_ids.as_slice(), &[1, 2]);
        assert_eq!(
            request.content().extra_args.as_ref().unwrap()["sglang_tito"]["return_routed_experts"],
            true
        );
        let output = dynamo_sglang_generate::inspect(
            json!({
                "output_ids": [3], "meta_info": {
                    "id": "worker-private", "finish_reason": {"type":"length"},
                    "completion_tokens": 1, "routed_experts": self.0,
                }
            }),
            false,
        )
        .unwrap()
        .output;
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(output)])),
            request.context(),
        ))
    }
}

async fn serve(value: Value) -> (String, tokio::task::JoinHandle<()>) {
    let format = custom();
    let state = dynamo_llm::http::service::service_v2::HttpService::builder()
        .build()
        .unwrap()
        .state_clone();
    let mut card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("test-model");
    card.runtime_config
        .runtime_data
        .insert(format.capability().into(), json!(true));
    let workers = dynamo_llm::discovery::WorkerSet::new("external".into(), "test".into(), card)
        .with_generate_engine(Arc::new(Worker(value)));
    assert!(
        state
            .manager()
            .add_worker_set("test-model", "external", workers)
    );
    let (_, router) = dynamo_llm::http::service::format_generate::router(state, None, format);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{address}/generate"), task)
}

#[tokio::test]
async fn real_http_unary_and_sse_encode_exact_bytes() {
    // Empty, partial Base64 groups, high bytes, and compressible expert data.
    for bytes in [
        vec![],
        vec![0],
        vec![0, 1],
        vec![0, 1, 2],
        vec![0, 1, 2, 3],
        vec![255; 4],
        b"hello".to_vec(),
        vec![42; 4096],
    ] {
        let (url, task) = serve(json!(STANDARD.encode(&bytes))).await;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        for stream in [false, true] {
            let response = client
                .post(&url)
                .json(&json!({
                    "input_ids": [1,2], "rid": "client-id", "stream": stream,
                    "return_routed_experts": true, "sampling_params": {"max_new_tokens":1},
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let text = response.text().await.unwrap();
            let body: Value = if stream {
                assert!(text.contains("data: [DONE]"), "{text}");
                let data = text
                    .lines()
                    .find_map(|line| line.strip_prefix("data: {").map(|s| format!("{{{s}")))
                    .unwrap();
                serde_json::from_str(&data).unwrap()
            } else {
                serde_json::from_str(&text).unwrap()
            };
            let compressed = STANDARD
                .decode(body["meta_info"]["routed_experts"].as_str().unwrap())
                .unwrap();
            let decoded = zstd::stream::decode_all(compressed.as_slice()).unwrap();
            assert_eq!(decoded, bytes, "stream={stream}");
            if bytes.len() == 4096 {
                assert!(compressed.len() < bytes.len());
            }
            assert_eq!(body["output_ids"], json!([3]));
            assert_eq!(body["meta_info"]["id"], "client-id");
        }
        task.abort();
    }
}

#[tokio::test]
async fn malformed_binary_data_uses_http_and_sse_error_handling() {
    for malformed in [json!("not base64!"), json!([1, 2, 3])] {
        let (url, task) = serve(malformed).await;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        for stream in [false, true] {
            let response = client
                .post(&url)
                .json(&json!({
                    "input_ids": [1,2], "stream": stream, "return_routed_experts": true,
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), if stream { 200 } else { 500 });
            let body = response.text().await.unwrap();
            assert!(body.contains("Internal server error"), "{body}");
            assert!(
                !body.contains("routed_experts"),
                "backend details must stay private: {body}"
            );
        }
        task.abort();
    }
}

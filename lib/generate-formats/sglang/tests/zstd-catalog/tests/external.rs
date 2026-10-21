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
use proptest::prelude::*;
use serde_json::{Value, json};

fn custom() -> SharedFormat {
    let mut registry = Registry::default();
    custom_generate_format_catalog::register(&mut registry).unwrap();
    registry.get("sglang-zstd").unwrap()
}

struct CompressedWorker;
impl dynamo_sglang_generate::testkit::sim::Worker for CompressedWorker {
    fn stream(&self, request: dynamo_sglang_generate::testkit::sim::Request<'_>) -> Vec<Value> {
        let mut chunks = dynamo_sglang_generate::testkit::sim::NativeWorker.stream(request);
        for response in &mut chunks {
            if let Some(value) = response.pointer_mut("/meta_info/routed_experts")
                && let Some(encoded) = value.as_str()
            {
                *value = compressed(&STANDARD.decode(encoded).unwrap()).into();
            }
        }
        chunks
    }
}

#[test]
fn backend_property_laws_apply_to_the_external_codec() {
    let mut harness = Harness::native(custom());
    harness.worker = Arc::new(CompressedWorker);
    harness.check_all();
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
async fn real_http_unary_and_sse_preserve_worker_encoding() {
    let cases = [vec![], vec![0, 255, 1], vec![42; 4096]]
        .into_iter()
        .map(|bytes| (compressed(&bytes), Some(bytes)))
        // Neither transport may even validate the encoding without a P/D join.
        .chain([("opaque-worker-format".into(), None)]);
    for (encoded, bytes) in cases {
        let (url, task) = serve(json!(encoded)).await;
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
            assert_eq!(body["meta_info"]["routed_experts"], encoded);
            if let Some(bytes) = &bytes {
                let frame = STANDARD.decode(&encoded).unwrap();
                assert_eq!(zstd::stream::decode_all(frame.as_slice()).unwrap(), *bytes);
            }
            assert_eq!(body["output_ids"], json!([3]));
            assert_eq!(body["meta_info"]["id"], "client-id");
        }
        task.abort();
    }
}

fn compressed(bytes: &[u8]) -> String {
    STANDARD.encode(zstd::stream::encode_all(bytes, 3).unwrap())
}

fn joined(prefill: Value, decode: Value, stream: bool) -> Value {
    use dynamo_llm::generate_format::ClientRequest;
    let format = custom();
    let body = json!({"input_ids": [1,2], "stream": stream, "return_routed_experts": true});
    let opened = format
        .open(ClientRequest {
            body: body.to_string().as_bytes(),
            headers: &axum::http::HeaderMap::from_iter([(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            )]),
            default_request_id: "id",
        })
        .unwrap();
    let mut exchange = (opened.bind)(&dynamo_sglang_generate::testkit::worker(&Value::Null))
        .unwrap()
        .exchange;
    let mut chunks = Vec::new();
    for (value, is_prefill) in [(prefill, true), (decode, false)] {
        let output = dynamo_sglang_generate::inspect(value, is_prefill)
            .unwrap()
            .output;
        exchange
            .chunk(output, is_prefill, &mut |v| chunks.push(v.clone()))
            .unwrap();
    }
    exchange.finish().unwrap().or_else(|| chunks.pop()).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn compressed_pd_join_matches_uncompressed_prefix_overlay(
        prefix in prop::collection::vec(any::<u8>(), 0..128),
        tail in prop::collection::vec(any::<u8>(), 0..128),
        stream in any::<bool>(),
    ) {
        let body = |bytes: &[u8]| json!({"meta_info": {
            "routed_experts": compressed(bytes),
            "indexer_topk": STANDARD.encode(bytes),
        }});
        let response = joined(body(&prefix), body(&tail), stream);
        let frame = STANDARD.decode(response["meta_info"]["routed_experts"].as_str().unwrap()).unwrap();
        let actual = zstd::stream::decode_all(frame.as_slice()).unwrap();
        // Independent byte oracle; the other sequence field keeps native Base64.
        let expected: Vec<u8> = prefix.iter().chain(tail.iter().skip(prefix.len())).copied().collect();
        prop_assert_eq!(&actual, &expected);
        prop_assert_eq!(response["meta_info"]["indexer_topk"].as_str().unwrap(), STANDARD.encode(&expected));
    }
}

#[test]
fn missing_or_corrupt_prefill_drops_stale_decode_experts() {
    let valid = json!(compressed(b"prompt+output"));
    for prefix in [
        Value::Null,
        json!("bad Base64!"),
        json!(STANDARD.encode(b"not zstd")),
        json!([1]),
    ] {
        for stream in [false, true] {
            for (prefill, decode) in [(&prefix, &valid), (&valid, &prefix)] {
                let response = joined(
                    json!({"meta_info": {"routed_experts": prefill}}),
                    json!({"meta_info": {"routed_experts": decode}}),
                    stream,
                );
                assert!(response["meta_info"].get("routed_experts").is_none());
            }
        }
    }
    let response = joined(
        json!({}),
        json!({"meta_info": {"routed_experts": valid}}),
        false,
    );
    assert!(response["meta_info"].get("routed_experts").is_none());
}

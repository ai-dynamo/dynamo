#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context as TaskContext, Poll},
    };

    use super::service_v2::{HttpService, VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV};
    use super::*;
    use crate::http::service::metrics::{Endpoint, RequestType, Status};
    use crate::protocols::{Annotated, common::llm_backend::LLMEngineOutput};
    use dynamo_runtime::{
        engine::{AsyncEngine, ResponseStream},
        pipeline::{Error, ManyOut, SingleIn},
    };
    use futures::Stream;
    use tokio::sync::Notify;
    use tokio_util::sync::CancellationToken;
    use tracing::field::{Field, Visit};
    use tracing::{Subscriber, span};
    use tracing_subscriber::Layer;
    use tracing_subscriber::prelude::*;

    #[test]
    fn multimodal_lora_is_rejected_before_backend_dispatch() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1],
            "sampling_params": {},
            "features": {}
        }))
        .unwrap();

        let error = preprocessed_from_generate(
            request,
            "adapter-a",
            None,
            "req-mm-lora",
            16,
            Some("adapter-a".to_string()),
        )
        .unwrap_err();

        assert!(error.to_string().contains("tower-LoRA"));
    }

    #[derive(Clone, Copy)]
    enum PendingPhase {
        Generate,
        Stream,
    }

    struct PendingOperation {
        started: Arc<Notify>,
        dropped: Arc<AtomicBool>,
        polled: bool,
    }

    impl PendingOperation {
        fn new(started: Arc<Notify>, dropped: Arc<AtomicBool>) -> Self {
            Self {
                started,
                dropped,
                polled: false,
            }
        }

        fn mark_started(&mut self) {
            if !self.polled {
                self.polled = true;
                self.started.notify_one();
            }
        }
    }

    impl Future for PendingOperation {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
            self.get_mut().mark_started();
            Poll::Pending
        }
    }

    impl Stream for PendingOperation {
        type Item = Annotated<LLMEngineOutput>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            self.get_mut().mark_started();
            Poll::Pending
        }
    }

    impl Drop for PendingOperation {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    struct PendingEngine {
        phase: PendingPhase,
        started: Arc<Notify>,
        dropped: Arc<AtomicBool>,
    }

    struct TerminalEngine(crate::protocols::common::FinishReason);

    struct CancelledEngine;

    struct MetricEngine;

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CancelledEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            Err(dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::Cancelled)
                .message("backend cancelled before opening a stream")
                .build()
                .into())
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for TerminalEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
                index: Some(0),
                finish_reason: Some(self.0.clone()),
                ..Default::default()
            })]);
            Ok(ResponseStream::new(Box::pin(stream), request.context()))
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for MetricEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let stream = futures::stream::iter([
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![10],
                    index: Some(0),
                    ..Default::default()
                }),
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![11],
                    index: Some(0),
                    finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                    completion_usage: Some(dynamo_protocols::types::CompletionUsage {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        total_tokens: 5,
                        prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                            audio_tokens: None,
                            cached_tokens: Some(2),
                        }),
                        completion_tokens_details: None,
                    }),
                    ..Default::default()
                }),
            ]);
            Ok(ResponseStream::new(Box::pin(stream), request.context()))
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for PendingEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let operation = PendingOperation::new(self.started.clone(), self.dropped.clone());
            match self.phase {
                PendingPhase::Generate => {
                    operation.await;
                    unreachable!("pending generate operation completed")
                }
                PendingPhase::Stream => {
                    Ok(ResponseStream::new(Box::pin(operation), request.context()))
                }
            }
        }
    }

    #[derive(Clone)]
    struct RequestIdCaptureLayer(Arc<Mutex<Option<String>>>);

    impl<S: Subscriber> Layer<S> for RequestIdCaptureLayer {
        fn on_new_span(
            &self,
            attrs: &span::Attributes<'_>,
            _id: &span::Id,
            _context: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = RequestIdVisitor::default();
            attrs.record(&mut visitor);
            if visitor.request_id.is_some() {
                *self.0.lock().unwrap() = visitor.request_id;
            }
        }
    }

    #[derive(Default)]
    struct RequestIdVisitor {
        request_id: Option<String>,
    }

    impl Visit for RequestIdVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "request_id" {
                self.request_id = Some(value.to_string());
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "request_id" {
                self.request_id = Some(format!("{value:?}"));
            }
        }
    }

    /// Spin up an `HttpService` bound to an ephemeral port and return the port
    /// plus the run handle. Mirrors the reqwest-based router tests in
    /// `service_v2`.
    async fn serve(enable_generate: Option<bool>) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let builder = HttpService::builder().port(port);
        let builder = match enable_generate {
            Some(enabled) => builder.enable_engine_apis(enabled),
            None => builder,
        };
        let service = builder.build().unwrap();
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async move {
            service.run_with_listener(cancel_token, listener).await.ok();
        });
        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (port, handle)
    }

    #[tokio::test]
    async fn generate_route_no_model_returns_structured_404() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{"top_k":-1}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_found");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_streaming_returns_501() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{},"stream":true}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_IMPLEMENTED.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_implemented");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_rejects_empty_token_ids() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");

        assert_eq!(resp.status().as_u16(), StatusCode::BAD_REQUEST.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"].as_str().is_some_and(
                |message| message.contains("token_ids must contain at least one token")
            )
        );
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_enforces_vllm_rust_request_rules() {
        let (port, handle) = serve(Some(true)).await;
        let client = reqwest::Client::new();
        let invalid = [
            r#"{"token_ids":[1],"sampling_params":{},"stream_options":{"include_usage":true}}"#,
            r#"{"token_ids":[1],"sampling_params":{"max_tokens":0}}"#,
            r#"{"token_ids":[1],"sampling_params":{"prompt_logprobs":-2}}"#,
            r#"{"token_ids":[1],"sampling_params":{"min_tokens":3,"max_tokens":2}}"#,
        ];

        for body in invalid {
            let resp = client
                .post(format!("http://localhost:{port}/inference/v1/generate"))
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .expect("generate request failed");
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body: serde_json::Value = resp.json().await.expect("json body");
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }

        handle.abort();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_404_by_default() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, None::<&str>)],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                handle.abort();
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_is_registered_when_enabled_by_env() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, Some("1"))],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                let body: serde_json::Value = resp.json().await.expect("json body");
                assert_eq!(body["error"]["type"], "not_found");
                handle.abort();
            },
        )
        .await;
    }

    #[test]
    fn engine_fields_reach_envelope_with_resolved_id_and_cache_namespace() {
        let raw = serde_json::json!({
            "request_id": "req-forward",
            "token_ids": [1, 2],
            "sampling_params": {
                "max_tokens": 8,
                "future_sampling_field": {"nested": true}
            },
            "model": "test-model",
            "stream": true,
            "stream_options": {"include_usage": true},
            "cache_salt": "tenant-a",
            "features": {"future_feature": [1, 2, 3]},
            "priority": 7,
            "kv_transfer_params": {"remote": "worker-a"},
            "future_top_level_field": {"anything": "works"}
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 16, None)
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            Some(8)
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority),
            Some(-7),
            "vLLM lower-is-higher priority must be inverted for Dynamo routing"
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority_jump),
            Some(0.0)
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.cache_namespace.as_deref()),
            Some("tenant-a")
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");

        let mut expected_envelope = raw;
        expected_envelope["request_id"] = serde_json::json!("resolved-request");
        let expected_token_ids = expected_envelope
            .as_object_mut()
            .and_then(|object| object.remove("token_ids"))
            .expect("token_ids in client request");
        assert_eq!(preprocessed.token_ids, vec![1, 2]);
        assert_eq!(
            preprocessed
                .tracker
                .as_ref()
                .and_then(|tracker| tracker.isl_tokens()),
            Some(2)
        );
        assert_eq!(expected_token_ids, serde_json::json!([1, 2]));
        assert_eq!(envelope, &expected_envelope);
        assert!(envelope.get("token_ids").is_none());
    }

    #[test]
    fn multimodal_features_build_exact_routing_tokens_without_changing_execution_payload() {
        use base64::Engine as _;

        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let kwargs_a = vec![0x80];
        let kwargs_b = vec![0x81];
        let encoded_a = base64::engine::general_purpose::STANDARD.encode(&kwargs_a);
        let encoded_b = base64::engine::general_purpose::STANDARD.encode(&kwargs_b);
        let raw = serde_json::json!({
            "token_ids": [10, 11, 12, 12, 12, 15, 16, 17, 17, 19],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [hash_a, hash_b]},
                "mm_placeholders": {"image": [
                    {"offset": 2, "length": 3},
                    {"offset": 7, "length": 2}
                ]},
                "kwargs_data": {"image": [encoded_a, encoded_b]}
            }
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 4, None)
                .expect("build request");

        let canonical_a =
            mm_routing::preprocessed_mm_cache_identifier("image", &kwargs_a);
        let canonical_b =
            mm_routing::preprocessed_mm_cache_identifier("image", &kwargs_b);
        let pad_a = dynamo_kv_router::protocols::pad_value_for_mm_hash(
            dynamo_kv_router::zmq_wire::hash_mm_identifier(&canonical_a).unwrap(),
        );
        let pad_b = dynamo_kv_router::protocols::pad_value_for_mm_hash(
            dynamo_kv_router::zmq_wire::hash_mm_identifier(&canonical_b).unwrap(),
        );
        let mm = preprocessed
            .mm_routing_info
            .as_ref()
            .expect("multimodal routing projection");
        assert_eq!(
            mm.routing_token_ids,
            vec![10, 11, pad_a, pad_a, pad_a, 15, 16, pad_b, pad_b, 19, 0, 0]
        );
        assert!(mm.block_mm_infos.is_empty());
        assert_eq!(mm.expanded_prompt_len, 10);

        assert_eq!(
            preprocessed.token_ids,
            vec![10, 11, 12, 12, 12, 15, 16, 17, 17, 19]
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");
        assert_eq!(envelope["features"], raw["features"]);
    }

    #[test]
    fn multimodal_projection_handles_many_blocks_and_sparse_placeholders() {
        use base64::Engine as _;

        let token_count = 1024;
        let mut token_ids = vec![7_u32; token_count];
        let mut hashes = Vec::new();
        let mut placeholders = Vec::new();
        let mut kwargs_data = Vec::new();
        let mut expected_pads = Vec::new();

        for index in 0..64 {
            let offset = index * 16 + 7;
            let identifier = format!("image-{index}");
            let kwargs = vec![index as u8];
            let canonical =
                mm_routing::preprocessed_mm_cache_identifier("image", &kwargs);
            let hash = dynamo_kv_router::zmq_wire::hash_mm_identifier(&canonical)
                .expect("non-empty identifier");
            hashes.push(identifier);
            placeholders.push(serde_json::json!({"offset": offset, "length": 1}));
            kwargs_data.push(base64::engine::general_purpose::STANDARD.encode(kwargs));
            expected_pads.push((
                offset,
                dynamo_kv_router::protocols::pad_value_for_mm_hash(hash),
            ));
        }

        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": token_ids,
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": hashes},
                "mm_placeholders": {"image": placeholders},
                "kwargs_data": {"image": kwargs_data}
            }
        }))
        .expect("deserialize request");
        let routing = generate_mm_routing_info(&request, 8)
            .expect("valid sparse MM routing metadata")
            .expect("MM routing projection");

        assert_eq!(routing.routing_token_ids.len(), token_count);
        for (offset, pad) in expected_pads {
            assert_eq!(routing.routing_token_ids[offset], pad);
            token_ids[offset] = pad;
        }
        assert_eq!(routing.routing_token_ids, token_ids);
    }

    #[test]
    fn generate_mm_routing_hash_matches_normalized_vllm_worker_event() {
        use base64::Engine as _;

        let mm_identifier = "1234567890abcdef".repeat(2);
        let kwargs = vec![0x80];
        let canonical_identifier =
            mm_routing::preprocessed_mm_cache_identifier("image", &kwargs);
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [mm_identifier.clone()]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": [
                    base64::engine::general_purpose::STANDARD.encode(kwargs)
                ]}
            }
        }))
        .expect("deserialize request");
        let routing = generate_mm_routing_info(&request, 4)
            .expect("valid MM routing metadata")
            .expect("MM routing projection");
        let request_hashes = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            &routing.routing_token_ids,
            4,
            dynamo_kv_router::protocols::BlockHashOptions::default(),
        );

        let event_block = dynamo_kv_router::zmq_wire::create_stored_block_from_parts(
            4,
            7,
            &[10, 99, 99, 20],
            dynamo_kv_router::zmq_wire::StoredBlockOptions {
                mm_extra_info: Some(dynamo_kv_router::protocols::BlockExtraInfo {
                    mm_objects: vec![dynamo_kv_router::protocols::BlockMmObjectInfo {
                        mm_hash: dynamo_kv_router::zmq_wire::hash_mm_identifier(
                            &canonical_identifier,
                        )
                        .expect("non-empty identifier"),
                        offsets: vec![(1, 3)],
                    }],
                }),
                image_token_id: Some(99),
                ..Default::default()
            },
        );

        assert_eq!(request_hashes[0], event_block.tokens_hash);
    }

    #[test]
    fn mixed_placeholder_span_without_embed_mask_disables_exact_routing() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 42, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
                "kwargs_data": {"image": ["gA=="]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 5)
                .expect_err("an omitted sparse mask must not over-substitute tokens"),
            "mixed multimodal placeholder spans require is_embed"
        );
    }

    #[test]
    fn sparse_mm_embed_mask_matches_normalized_vllm_worker_event() {
        let mm_identifier = "opaque-renderer-image-0";
        let kwargs = [0x80];
        let canonical_identifier =
            mm_routing::preprocessed_mm_cache_identifier("image", &kwargs);
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 42, 99, 20],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [mm_identifier]},
                "mm_placeholders": {"image": [{
                    "offset": 1,
                    "length": 3,
                    "is_embed": [true, false, true]
                }]},
                "kwargs_data": {"image": ["gA=="]}
            }
        }))
        .expect("deserialize request");
        let routing = generate_mm_routing_info(&request, 5)
            .expect("valid sparse MM routing metadata")
            .expect("MM routing projection");
        let mm_hash = dynamo_kv_router::zmq_wire::hash_mm_identifier(&canonical_identifier)
            .expect("non-empty identifier");
        let pad = dynamo_kv_router::protocols::pad_value_for_mm_hash(mm_hash);
        assert_eq!(routing.routing_token_ids, vec![10, pad, 42, pad, 20]);

        let request_hash = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            &routing.routing_token_ids,
            5,
            dynamo_kv_router::protocols::BlockHashOptions::default(),
        )[0];
        let event_block = dynamo_kv_router::zmq_wire::create_stored_block_from_parts(
            5,
            7,
            &[10, 99, 42, 99, 20],
            dynamo_kv_router::zmq_wire::StoredBlockOptions {
                mm_extra_info: Some(dynamo_kv_router::protocols::BlockExtraInfo {
                    mm_objects: vec![dynamo_kv_router::protocols::BlockMmObjectInfo {
                        mm_hash,
                        offsets: vec![],
                    }],
                }),
                image_token_id: Some(99),
                ..Default::default()
            },
        );

        assert_eq!(request_hash, event_block.tokens_hash);
    }

    #[test]
    fn sparse_multi_object_block_disables_inexact_projection() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [10, 99, 42, 99, 20, 99, 30],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-a", "image-b"]},
                "mm_placeholders": {"image": [
                    {
                        "offset": 1,
                        "length": 3,
                        "is_embed": [true, false, true]
                    },
                    {"offset": 5, "length": 1}
                ]},
                "kwargs_data": {"image": ["gA==", "gQ=="]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 7)
                .expect_err("worker run-order mapping would assign image B to image A"),
            "sparse multimodal layout cannot be normalized exactly by worker events"
        );
    }

    #[test]
    fn invalid_multimodal_routing_metadata_falls_back_without_dropping_features() {
        let raw = serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": [""]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": ["opaque"]}
            }
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 4, None)
                .expect("malformed routing metadata must not reject execution");

        assert!(preprocessed.mm_routing_info.is_none());
        assert_eq!(preprocessed.token_ids, vec![1, 2, 3, 4]);
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");
        assert_eq!(envelope["features"], raw["features"]);
    }

    #[test]
    fn multimodal_execution_contract_requires_bounded_inline_data() {
        use base64::Engine as _;

        let encoded = base64::engine::general_purpose::STANDARD.encode([0x80]);
        let valid: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-a"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
                "kwargs_data": {"image": [encoded.clone()]}
            }
        }))
        .unwrap();
        validate_generate_mm_features(&valid).expect("valid execution features");

        for features in [
            serde_json::json!({
                "mm_hashes": {"image": ["image-a"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
                "kwargs_data": null
            }),
            serde_json::json!({
                "mm_hashes": {"image": ["image-a"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
                "kwargs_data": {"image": ["not-base64!"]}
            }),
            serde_json::json!({
                "mm_hashes": {"image": ["image-a"]},
                "mm_placeholders": {"image": [{"offset": 3, "length": 1}]},
                "kwargs_data": {"image": [encoded.clone()]}
            }),
        ] {
            let request: GenerateRequest = serde_json::from_value(serde_json::json!({
                "token_ids": [1, 2, 3],
                "sampling_params": {},
                "features": features
            }))
            .unwrap();
            assert!(validate_generate_mm_features(&request).is_err());
        }
    }

    #[test]
    fn canonical_preprocessed_mm_identity_matches_native_grpc_vector() {
        assert_eq!(
            mm_routing::preprocessed_mm_cache_identifier("image", &[0x80]),
            "grpc-mm:835d2213e413e78e540c88905d36dfa708aa0eb02e9d546026a832c6f5ac5825"
        );
    }

    #[test]
    fn multimodal_execution_contract_compares_modality_keys_as_sets() {
        use base64::Engine as _;

        let encoded = base64::engine::general_purpose::STANDARD.encode([0x80]);
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-a"], "video": ["video-a"]},
                "mm_placeholders": {
                    "video": [{"offset": 2, "length": 1}],
                    "image": [{"offset": 1, "length": 1}]
                },
                "kwargs_data": {
                    "image": [encoded.clone()],
                    "video": [encoded]
                }
            }
        }))
        .unwrap();

        validate_generate_mm_features(&request).expect("modality key order is irrelevant");
    }

    #[test]
    fn overlapping_multimodal_placeholders_disable_mm_routing() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [9, 9, 9, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["a".repeat(64), "b".repeat(64)]},
                "mm_placeholders": {"image": [
                    {"offset": 0, "length": 2},
                    {"offset": 1, "length": 2}
                ]},
                "kwargs_data": {"image": ["gA==", "gQ=="]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 4).expect_err("overlap must disable MM routing"),
            "multimodal placeholder ranges must not overlap"
        );
    }

    #[test]
    fn non_image_modality_disables_exact_mm_projection() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"audio": ["audio-0"]},
                "mm_placeholders": {"audio": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"audio": ["gA=="]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 4)
                .expect_err("non-image modality must disable exact projection"),
            "exact /generate MM routing currently supports image placeholders only"
        );
    }

    #[test]
    fn image_hashes_and_placeholders_must_be_paired() {
        for features in [
            serde_json::json!({
                "mm_hashes": {"image": ["image-0"]},
                "mm_placeholders": {},
                "kwargs_data": {"image": ["gA=="]}
            }),
            serde_json::json!({
                "mm_hashes": {},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": ["gA=="]}
            }),
        ] {
            let request: GenerateRequest = serde_json::from_value(serde_json::json!({
                "token_ids": [1, 2, 3, 4],
                "sampling_params": {},
                "features": features
            }))
            .expect("deserialize request");

            assert_eq!(
                generate_mm_routing_info(&request, 4)
                    .expect_err("one-sided image metadata must disable exact routing"),
                "image hashes, placeholders, and kwargs_data must all be present"
            );
        }
    }

    #[test]
    fn image_hashes_and_placeholders_must_have_equal_lengths() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-0", "image-1"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 2}]},
                "kwargs_data": {"image": ["gA==", "gQ=="]}
            }
        }))
        .expect("deserialize request");

        assert_eq!(
            generate_mm_routing_info(&request, 4)
                .expect_err("item count mismatch must disable exact routing"),
            "image hashes, placeholders, and kwargs_data must have equal lengths"
        );
    }

    #[test]
    fn omitted_max_tokens_stays_omitted_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 16, None)
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, None);
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            None
        );
    }

    #[test]
    fn explicit_zero_min_tokens_stays_explicit_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {"min_tokens": 0},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(request, "test-model", None, "resolved-request", 16, None)
                .expect("build request");
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(0));
    }

    #[test]
    fn generate_request_context_matches_vllm_header_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(X_REQUEST_ID_HEADER, "header-request".parse().unwrap());
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "3".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "header-request");
        assert_eq!(context.data_parallel_rank, Some(3));
    }

    #[test]
    fn generate_request_context_falls_back_and_ignores_invalid_dp_rank() {
        let mut headers = HeaderMap::new();
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "invalid".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "body-request");
        assert_eq!(context.data_parallel_rank, None);
    }

    #[test]
    fn generate_dispatch_span_uses_resolved_request_id() {
        let captured_request_id = Arc::new(Mutex::new(None));
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(RequestIdCaptureLayer(captured_request_id.clone())),
        );

        let _dispatch_span = generate_dispatch_span("header-request");

        assert_eq!(
            captured_request_id.lock().unwrap().as_deref(),
            Some("header-request")
        );
    }

    fn dispatch_test_context() -> Context<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test-model".to_string())
                .token_ids(vec![1])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .expect("build dispatch test request"),
        )
    }

    fn metric_value<'a>(
        families: &'a [prometheus::proto::MetricFamily],
        name: &str,
        labels: &[(&str, &str)],
    ) -> &'a prometheus::proto::Metric {
        let family = families
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing metric family {name}"));
        family
            .get_metric()
            .iter()
            .find(|metric| {
                labels.iter().all(|(expected_name, expected_value)| {
                    metric.get_label().iter().any(|label| {
                        label.name() == *expected_name && label.value() == *expected_value
                    })
                })
            })
            .unwrap_or_else(|| panic!("missing {name} series with labels {labels:?}"))
    }

    fn assert_cancelled_dispatch_metrics(state: &service_v2::State) {
        let metric_model = state.manager().metric_model_for("test-model");
        let metrics = state.metrics_clone();
        assert_eq!(metrics.get_inflight_count(metric_model), 0);
        assert_eq!(
            metrics.get_request_counter(
                metric_model,
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            1
        );
    }

    async fn await_cancelled_dispatch(
        task: tokio::task::JoinHandle<Response>,
        dropped: &AtomicBool,
        state: &service_v2::State,
    ) {
        let response = tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("dispatch did not stop promptly after request kill")
            .expect("dispatch task panicked");
        assert_eq!(response.status().as_u16(), 499);
        assert!(dropped.load(Ordering::SeqCst));
        assert_cancelled_dispatch_metrics(state);
    }

    async fn assert_request_kill_interrupts_pending(phase: PendingPhase) {
        let started = Arc::new(Notify::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(PendingEngine {
                phase,
                started: started.clone(),
                dropped: dropped.clone(),
            });
        let context = dispatch_test_context();
        let request_context = context.context();
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let task = tokio::spawn(generate_dispatch(
            engine,
            context,
            "req-pending-dispatch".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        ));

        started.notified().await;
        assert_eq!(
            state
                .metrics_clone()
                .get_inflight_count(state.manager().metric_model_for("test-model")),
            1
        );
        request_context.kill();

        await_cancelled_dispatch(task, dropped.as_ref(), state.as_ref()).await;
    }

    async fn dispatch_terminal_finish_reason(
        finish_reason: crate::protocols::common::FinishReason,
    ) -> (Response, Arc<service_v2::State>) {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(TerminalEngine(finish_reason));
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let response = generate_dispatch(
            engine,
            dispatch_test_context(),
            "req-terminal-dispatch".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;
        (response, state)
    }

    #[tokio::test]
    async fn request_kill_interrupts_pending_engine_generate() {
        assert_request_kill_interrupts_pending(PendingPhase::Generate).await;
    }

    #[tokio::test]
    async fn request_kill_interrupts_pending_response_stream() {
        assert_request_kill_interrupts_pending(PendingPhase::Stream).await;
    }

    #[tokio::test]
    async fn backend_error_finish_returns_sanitized_500() {
        let backend_detail = "sensitive backend failure";
        let (response, _state) = dispatch_terminal_finish_reason(
            crate::protocols::common::FinishReason::Error(backend_detail.to_string()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read error response");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("parse error response");
        assert_eq!(body["error"]["message"], "internal server error");
        assert!(!body.to_string().contains(backend_detail));
    }

    #[tokio::test]
    async fn backend_cancelled_finish_returns_499() {
        let (response, state) =
            dispatch_terminal_finish_reason(crate::protocols::common::FinishReason::Cancelled)
                .await;

        assert_eq!(response.status().as_u16(), 499);
        assert_cancelled_dispatch_metrics(state.as_ref());
    }

    #[tokio::test]
    async fn immediate_engine_cancellation_returns_499() {
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(CancelledEngine);
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();

        let response = generate_dispatch(
            engine,
            dispatch_test_context(),
            "req-immediate-cancel".to_string(),
            "test-model".to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;

        assert_eq!(response.status().as_u16(), 499);
        assert_cancelled_dispatch_metrics(state.as_ref());
    }

    #[tokio::test]
    async fn successful_generate_populates_frontend_metrics() {
        const MODEL: &str = "generate-metric-test-model";
        const WORKER_ID: &str = "987654321";
        const DP_RANK: &str = "3";

        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(3, None);
        tracker.record_worker(
            WORKER_ID.parse().unwrap(),
            Some(DP_RANK.parse().unwrap()),
            crate::discovery::WORKER_TYPE_DECODE,
        );
        let context = Context::new(
            PreprocessedRequest::builder()
                .model(MODEL.to_string())
                .token_ids(vec![1, 2, 3])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .tracker(Some(tracker))
                .build()
                .expect("build metric test request"),
        );
        let engine: crate::types::openai::generate::GenerateStreamingEngine =
            Arc::new(MetricEngine);
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let metric_model = state.manager().metric_model_for(MODEL).to_string();
        let registry = prometheus::Registry::new();
        state.metrics_clone().register(&registry).unwrap();

        let response = generate_dispatch(
            engine,
            context,
            "req-generate-metrics".to_string(),
            MODEL.to_string(),
            state.clone(),
            GenerateResponseOptions::default(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.metrics_clone().get_inflight_count(&metric_model), 0);
        assert_eq!(
            state.metrics_clone().get_request_counter(
                &metric_model,
                &Endpoint::Generate,
                &RequestType::Unary,
                &Status::Success,
                &ErrorType::None,
            ),
            1
        );

        let families = registry.gather();
        let model_labels = [("model", metric_model.as_str())];
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_requests_started_total",
                &[("model", metric_model.as_str()), ("endpoint", "generate")],
            )
            .get_counter()
            .value(),
            1.0
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_active_requests", &model_labels)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_queued_requests", &model_labels)
                .get_gauge()
                .value(),
            0.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_request_duration_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_input_sequence_tokens",
                &model_labels,
            )
            .get_histogram()
            .get_sample_sum(),
            3.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_output_sequence_tokens",
                &model_labels,
            )
            .get_histogram()
            .get_sample_sum(),
            2.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_output_tokens_total",
                &model_labels,
            )
            .get_counter()
            .value(),
            2.0
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_time_to_first_token_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(
                &families,
                "dynamo_frontend_inter_token_latency_seconds",
                &model_labels,
            )
            .get_histogram()
            .get_sample_count(),
            1
        );
        assert_eq!(
            metric_value(&families, "dynamo_frontend_cached_tokens", &model_labels,)
                .get_histogram()
                .get_sample_sum(),
            2.0
        );

        let worker_labels = [WORKER_ID, DP_RANK, crate::discovery::WORKER_TYPE_DECODE];
        assert_eq!(
            crate::http::service::metrics::WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE
                .with_label_values(&worker_labels)
                .get(),
            3
        );
        assert!(
            crate::http::service::metrics::WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE
                .with_label_values(&worker_labels)
                .get()
                > 0.0
        );
        assert!(
            crate::http::service::metrics::WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE
                .with_label_values(&worker_labels)
                .get()
                > 0.0
        );
    }

    #[test]
    fn generate_control_shadow_carries_dp_rank_and_inverted_priority() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "priority": -7
        }))
        .expect("deserialize request");

        let preprocessed = preprocessed_from_generate(
            request,
            "test-model",
            Some(3),
            "resolved-request",
            16,
            None,
        )
        .expect("build request");
        let routing = preprocessed.routing.as_ref().expect("routing hints");

        assert_eq!(routing.dp_rank, Some(3));
        assert_eq!(routing.priority, Some(7));
        assert_eq!(routing.priority_jump, Some(7.0));
    }

    #[test]
    fn priority_inversion_saturates_at_i32_min() {
        assert_eq!(dynamo_routing_priority(i32::MIN), i32::MAX);
    }
}

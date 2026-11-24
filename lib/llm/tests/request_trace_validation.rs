// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Error;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::request_trace::{self, RequestTraceEventType};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{AsyncEngine, Context, ManyOut, Operator, SingleIn, async_trait};
use dynamo_runtime::protocols::annotated::Annotated;

struct UnreachableBackend;

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
    for UnreachableBackend
{
    async fn generate(
        &self,
        _request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
        panic!("validation must reject the request before backend dispatch")
    }
}

#[tokio::test]
async fn uuid_only_rejection_keeps_request_payload() {
    temp_env::async_with_vars(
        [
            ("DYN_REQUEST_TRACE", Some("1")),
            ("DYN_REQUEST_TRACE_RECORDS", Some("request_payload")),
            ("DYN_REQUEST_TRACE_SINKS", Some("stderr")),
            ("DYN_MM_ALLOW_CLIENT_CACHE_UUIDS", None),
        ],
        async {
            let shutdown = tokio_util::sync::CancellationToken::new();
            request_trace::init_from_env_with_shutdown(shutdown.clone())
                .await
                .unwrap();
            let mut records = request_trace::subscribe();
            let card = ModelDeploymentCard::load_from_disk(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
                None,
            )
            .unwrap();
            let preprocessor = OpenAIPreprocessor::new(card).unwrap();
            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": "test-model", "stream": false,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": null, "uuid": "cached-id"}
                    ]}]
                }))
                .unwrap();
            let context = Context::new(request);
            let request_id = context.id().to_string();
            let next: Arc<
                dyn AsyncEngine<
                        SingleIn<PreprocessedRequest>,
                        ManyOut<Annotated<BackendOutput>>,
                        Error,
                    >,
            > = Arc::new(UnreachableBackend);
            let Err(error) = Operator::generate(preprocessor.as_ref(), context, next).await else {
                panic!("a UUID-only image must be rejected by default")
            };
            assert_eq!(
                error.downcast_ref::<DynamoError>().unwrap().error_type(),
                ErrorType::InvalidArgument
            );
            let reason = error.to_string();
            let public_message = error
                .downcast_ref::<DynamoError>()
                .unwrap()
                .public_message()
                .unwrap();
            assert!(public_message.contains("DYN_MM_ALLOW_CLIENT_CACHE_UUIDS"));
            let record = tokio::time::timeout(Duration::from_secs(5), records.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(record.event_type, RequestTraceEventType::RequestPayload);
            let payload = record.payload.unwrap();
            assert_eq!(payload.request_id, request_id);
            assert!(!payload.payload_complete);
            assert!(payload.response.is_none());
            assert_eq!(
                payload.payload_drop_reason.as_deref(),
                Some(reason.as_str())
            );
            let captured = payload.request.unwrap();
            assert_eq!(captured.inner.stream, Some(false));
            let captured = serde_json::to_value(captured.as_ref()).unwrap();
            assert_eq!(captured["messages"][0]["content"][0]["uuid"], "cached-id");
            assert!(records.try_recv().is_err());
            shutdown.cancel();
            request_trace::shutdown_workers().await;
        },
    )
    .await;
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(feature = "kserve")]
pub mod kserve_test {
    // For using gRPC client for test
    pub mod inference {
        tonic::include_proto!("inference");
    }
    use chrono::format;
    use dynamo_llm::entrypoint::input::grpc;
    use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
    use inference::{ModelInferRequest, ModelInferResponse, InferParameter, ModelStreamInferResponse, ModelMetadataRequest, ModelMetadataResponse, ModelConfigRequest, ModelConfigResponse};

    use anyhow::Error;
    use async_openai::config::OpenAIConfig;
    use async_stream::stream;
    use dynamo_llm::http::{
        service::{
            error::HttpError,
            metrics::{Endpoint, RequestType, Status, FRONTEND_METRIC_PREFIX},
            service_v2::HttpService,
            Metrics,
        },
    };
    use dynamo_llm::grpc::service::kserve::GrpcService;
    use dynamo_llm::protocols::{
        codec::SseLineCodec,
        convert_sse_stream,
        openai::{
            chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
        },
        Annotated,
    };
    use dynamo_runtime::{
        engine::AsyncEngineContext,
        pipeline::{
            async_trait, AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
        },
        CancellationToken,
    };
    use futures::StreamExt;
    use prometheus::{proto::MetricType, Registry};
    use reqwest::StatusCode;
    use rstest::*;
    use tonic::{transport::Channel, Request, Response};
    use std::time::{self, Duration};
    use std::{io::Cursor, sync::Arc};
    use tokio::time::{sleep, timeout};
    use tokio_util::codec::FramedRead;

    struct CounterEngine {}

    // Add a new long-running test engine
    struct LongRunningEngine {
        delay_ms: u64,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
    }

    impl LongRunningEngine {
        fn new(delay_ms: u64) -> Self {
            Self {
                delay_ms,
                cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }
        }

        fn was_cancelled(&self) -> bool {
            self.cancelled.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateCompletionRequest>,
            ManyOut<Annotated<NvCreateCompletionResponse>>,
            Error,
        > for CounterEngine
    {
        async fn generate(
            &self,
            request: SingleIn<NvCreateCompletionRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
            let (request, context) = request.transfer(());
            let ctx = context.context();

            // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
            #[allow(deprecated)]
            let max_tokens = request.inner.max_tokens.unwrap_or(0) as u64;

            // let generator = NvCreateChatCompletionStreamResponse::generator(request.model.clone());
            let generator = request.response_generator();

            let stream = stream! {
                tokio::time::sleep(std::time::Duration::from_millis(max_tokens)).await;
                for i in 0..10 {
                    yield Annotated::from_data(generator.create_choice(i,Some(format!("choice {i}")), None));
                }
            };

            Ok(ResponseStream::new(Box::pin(stream), ctx))
        }
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateChatCompletionRequest>,
            ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            Error,
        > for LongRunningEngine
    {
        async fn generate(
            &self,
            request: SingleIn<NvCreateChatCompletionRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
            let (_request, context) = request.transfer(());
            let ctx = context.context();

            tracing::info!(
                "LongRunningEngine: Starting generation with {}ms delay",
                self.delay_ms
            );

            let cancelled_flag = self.cancelled.clone();
            let delay_ms = self.delay_ms;

            let ctx_clone = ctx.clone();
            let stream = async_stream::stream! {

                // the stream can be dropped or it can be cancelled
                // either way we consider this a cancellation
                cancelled_flag.store(true, std::sync::atomic::Ordering::SeqCst);

                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {
                        // the stream went to completion
                        cancelled_flag.store(false, std::sync::atomic::Ordering::SeqCst);

                    }
                    _ = ctx_clone.stopped() => {
                        cancelled_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }

                yield Annotated::<NvCreateChatCompletionStreamResponse>::from_annotation("event.dynamo.test.sentinel", &"DONE".to_string()).expect("Failed to create annotated response");
            };

            Ok(ResponseStream::new(Box::pin(stream), ctx))
        }
    }

    struct AlwaysFailEngine {}

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateChatCompletionRequest>,
            ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            Error,
        > for AlwaysFailEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<NvCreateChatCompletionRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
            Err(HttpError {
                code: 403,
                message: "Always fail".to_string(),
            })?
        }
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateCompletionRequest>,
            ManyOut<Annotated<NvCreateCompletionResponse>>,
            Error,
        > for AlwaysFailEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<NvCreateCompletionRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
            Err(HttpError {
                code: 401,
                message: "Always fail".to_string(),
            })?
        }
    }

    fn compare_counter(
        metrics: &Metrics,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
        expected: u64,
    ) {
        assert_eq!(
            metrics.get_request_counter(model, endpoint, request_type, status),
            expected,
            "model: {}, endpoint: {:?}, request_type: {:?}, status: {:?}",
            model,
            endpoint.as_str(),
            request_type.as_str(),
            status.as_str()
        );
    }

    fn compute_index(endpoint: &Endpoint, request_type: &RequestType, status: &Status) -> usize {
        let endpoint = match endpoint {
            Endpoint::Completions => 0,
            Endpoint::ChatCompletions => 1,
            Endpoint::Embeddings => todo!(),
            Endpoint::Responses => todo!(),
        };

        let request_type = match request_type {
            RequestType::Unary => 0,
            RequestType::Stream => 1,
        };

        let status = match status {
            Status::Success => 0,
            Status::Error => 1,
        };

        endpoint * 4 + request_type * 2 + status
    }

    fn compare_counters(metrics: &Metrics, model: &str, expected: &[u64; 8]) {
        for endpoint in &[Endpoint::Completions, Endpoint::ChatCompletions] {
            for request_type in &[RequestType::Unary, RequestType::Stream] {
                for status in &[Status::Success, Status::Error] {
                    let index = compute_index(endpoint, request_type, status);
                    compare_counter(
                        metrics,
                        model,
                        endpoint,
                        request_type,
                        status,
                        expected[index],
                    );
                }
            }
        }
    }

    fn inc_counter(
        endpoint: Endpoint,
        request_type: RequestType,
        status: Status,
        expected: &mut [u64; 8],
    ) {
        let index = compute_index(&endpoint, &request_type, &status);
        expected[index] += 1;
    }

    /// Wait for the HTTP service to be ready by checking its health endpoint
    async fn get_ready_client(port: u16, timeout_secs: u64) -> GrpcInferenceServiceClient<Channel> {
        let start = tokio::time::Instant::now();
        let timeout = tokio::time::Duration::from_secs(timeout_secs);
        loop {
            let address = format!("http://0.0.0.0:{}", port);
            match GrpcInferenceServiceClient::connect(address).await {
                Ok(client) => return client,
                Err(_) if start.elapsed() < timeout => {
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
                Err(e) => panic!("Service failed to start within timeout: {}", e),
            }
        }
    }

    #[fixture]
    fn text_input(
        #[default("dummy input")] text: &str,
    ) -> inference::model_infer_request::InferInputTensor {
        inference::model_infer_request::InferInputTensor{
                name: "text_input".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                contents: Some(inference::InferTensorContents {
                    bytes_contents: vec![text.into()],
                    ..Default::default()
                }),
                ..Default::default()
            }
    }

    #[fixture]
    fn service_with_engines(
        #[default(8990)] port: u16,
    ) -> (GrpcService, Arc<CounterEngine>, Arc<AlwaysFailEngine>) {
        let service = GrpcService::builder().port(8989).build().unwrap();
        let manager = service.model_manager();

        let counter = Arc::new(CounterEngine {});
        let failure = Arc::new(AlwaysFailEngine {});

        manager
            .add_completions_model("counter", counter.clone())
            .unwrap();
        manager
            .add_chat_completions_model("failure", failure.clone())
            .unwrap();
        manager
            .add_completions_model("failure", failure.clone())
            .unwrap();

        (service, counter, failure)
    }

    #[rstest]
    #[tokio::test]
    async fn test_infer_failure(
        service_with_engines: (GrpcService, Arc<CounterEngine>, Arc<AlwaysFailEngine>),
        text_input: inference::model_infer_request::InferInputTensor) {
        // start server
        let service = service_with_engines.0;

        let token = CancellationToken::new();
        let task = tokio::spawn(async move { service.run(token.clone()).await });

        // create client and send request to unregistered model
        let mut client = get_ready_client(8989, 5).await;

        // unknown_model
        let request = tonic::Request::new(ModelInferRequest {
            model_name: "Tonic".into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![text_input.clone()],
            ..Default::default()
        });

        let response = client.model_infer(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::NotFound,
            "Expected NotFound error for unregistered model, get {}", err
        );

        // missing input
        let request = tonic::Request::new(ModelInferRequest {
            model_name: "counter".into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![],
            ..Default::default()
        });

        let response = client.model_infer(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "Expected InvalidArgument error for missing input, get {}", err
        );

        // request streaming
        let request = tonic::Request::new(ModelInferRequest {
            model_name: "counter".into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![text_input.clone(),
            inference::model_infer_request::InferInputTensor{
                name: "streaming".into(),
                datatype: "BOOL".into(),
                shape: vec![1],
                contents: Some(inference::InferTensorContents {
                    bool_contents: vec![true],
                    ..Default::default()
                }),
                ..Default::default()
            }
            ],
            ..Default::default()
        });

        let response = client.model_infer(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "Expected InvalidArgument error for streaming, get {}", err
        );
        // assert "stream" in error message
        assert!(
            err.message().contains("Streaming is not supported"),
            "Expected error message to contain 'Streaming is not supported', got: {}",
            err.message()
        );

        // AlwaysFailEngine
        let request = tonic::Request::new(ModelInferRequest {
            model_name: "failure".into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![text_input.clone()],
            ..Default::default()
        });

        let response = client.model_infer(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::Internal,
            "Expected Internal error for streaming, get {}", err
        );
        assert!(
            err.message().contains("Failed to generate completions:"),
            "Expected error message to contain 'Failed to generate completions:', got: {}",
            err.message()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_infer_sucess(text_input: inference::model_infer_request::InferInputTensor) {
        // start server
        let service = GrpcService::builder().port(8989).build().unwrap();
        let state = service.state_clone();
        let manager = state.manager();

        let token = CancellationToken::new();
        let cancel_token = token.clone();
        let task = tokio::spawn(async move { service.run(token.clone()).await });

        // register model
        let counter = Arc::new(CounterEngine {});
        let result = manager.add_completions_model("foo", counter);
        assert!(result.is_ok());

        // create client and send request to unregistered model
        let mut client = get_ready_client(8989, 5).await;

        let model_name = "foo";
        let request = tonic::Request::new(ModelInferRequest {
            model_name: model_name.into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![text_input],
            ..Default::default()
        });

        let response = client.model_infer(request).await.unwrap();
        assert_eq!(
            response.get_ref().model_name,
            model_name,
            "Expected response of the same model name",
        );
        for output in &response.get_ref().outputs {
            match output.name.as_str() {
                "text_output" => {
                    assert_eq!(
                        output.datatype, "BYTES",
                        "Expected 'text_output' to have datatype 'BYTES'"
                    );
                    assert_eq!(
                        output.shape, vec![10],
                        "Expected 'text_output' to have shape [10]"
                    );
                    let expected_output : Vec<Vec<u8>> = (0..10).map(|i| format!("choice {i}").into()).collect();
                    assert_eq!(
                        output.contents.as_ref().unwrap().bytes_contents,
                        expected_output,
                        "Expected 'text_output' to contain 'dummy output'"
                    );
                }
                "finish_reason" => {
                    assert_eq!(
                        output.datatype, "BYTES",
                        "Expected 'finish_reason' to have datatype 'BYTES'"
                    );
                    assert_eq!(
                        output.shape, vec![0],
                        "Expected 'finish_reason' to have shape [0]"
                    );
                }
                _ => panic!("Unexpected output name: {}", output.name),
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_stream_infer_sucess(service_with_engines: (GrpcService, Arc<CounterEngine>, Arc<AlwaysFailEngine>),
        text_input: inference::model_infer_request::InferInputTensor) {
        // start server
        let service = service_with_engines.0;

        let token = CancellationToken::new();
        let cancel_token = token.clone();
        let task = tokio::spawn(async move { service.run(token.clone()).await });

        // create client and send request to unregistered model
        let mut client = get_ready_client(8989, 5).await;

        let model_name = "counter";

        let outbound = async_stream::stream! {
            let request_count = 1;
            for _ in 0..request_count {
                let request = ModelInferRequest {
                    model_name: model_name.into(),
                    model_version: "1".into(),
                    id: "1234".into(),
                    inputs: vec![text_input.clone()],
                    ..Default::default()
                };

                yield request;
            }
        };

        let response = client.model_stream_infer(Request::new(outbound)).await.unwrap();
        let mut inbound = response.into_inner();

        let mut response_idx = 0;
        while let Some(response) = inbound.message().await.unwrap() {
            assert!(response.error_message.is_empty(), "Expected successful inference");
            assert!(response.infer_response.is_some(), "Expected successful inference");
            
            if let Some(response) = &response.infer_response {
                assert_eq!(
                    response.model_name,
                    model_name,
                    "Expected response of the same model name",
                );
                for output in &response.outputs {
                    match output.name.as_str() {
                        "text_output" => {
                            assert_eq!(
                                output.datatype, "BYTES",
                                "Expected 'text_output' to have datatype 'BYTES'"
                            );
                            assert_eq!(
                                output.shape, vec![1],
                                "Expected 'text_output' to have shape [1]"
                            );
                            let expected_output : Vec<u8> = format!("choice {response_idx}").into();
                            assert_eq!(
                                output.contents.as_ref().unwrap().bytes_contents,
                                vec![expected_output],
                                "Expected 'text_output' to contain 'dummy output'"
                            );
                        }
                        "finish_reason" => {
                            assert_eq!(
                                output.datatype, "BYTES",
                                "Expected 'finish_reason' to have datatype 'BYTES'"
                            );
                            assert_eq!(
                                output.shape, vec![0],
                                "Expected 'finish_reason' to have shape [0]"
                            );
                        }
                        _ => panic!("Unexpected output name: {}", output.name),
                    }
                }
            }
            response_idx += 1;
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_stream_infer_failure(service_with_engines: (GrpcService, Arc<CounterEngine>, Arc<AlwaysFailEngine>),
        text_input: inference::model_infer_request::InferInputTensor) {
        // start server
        let service = service_with_engines.0;

        let token = CancellationToken::new();
        let cancel_token = token.clone();
        let task = tokio::spawn(async move { service.run(token.clone()).await });

        // create client and send request to unregistered model
        let mut client = get_ready_client(8989, 5).await;

        let model_name = "failure";

        let outbound = async_stream::stream! {
            let request_count = 1;
            for _ in 0..request_count {
                let request = ModelInferRequest {
                    model_name: model_name.into(),
                    model_version: "1".into(),
                    id: "1234".into(),
                    inputs: vec![text_input.clone()],
                    ..Default::default()
                };

                yield request;
            }
        };

        let response = client.model_stream_infer(Request::new(outbound)).await.unwrap();
        let mut inbound = response.into_inner();

        loop {
            match inbound.message().await {
                Ok(Some(_)) => {
                    panic!("Expecting failure in the stream");
                },
                Err(err) => {
                    assert_eq!(
                        err.code(),
                        tonic::Code::Internal,
                        "Expected Internal error for streaming, get {}", err
                    );
                    assert!(
                        err.message().contains("Failed to generate completions:"),
                        "Expected error message to contain 'Failed to generate completions:', got: {}",
                        err.message()
                    );
                }
                Ok(None) => {
                    // End of stream
                    break;
                }
            }
        }
    }
}
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "common/ports.rs"]
mod ports;

pub mod kserve_test {
    // For using gRPC client for test
    pub mod inference {
        tonic::include_proto!("inference");
    }
    use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
    use dynamo_llm::model_card::ModelDeploymentCard;
    use dynamo_llm::model_type::{ModelInput, ModelType};
    use dynamo_llm::protocols::tensor;
    use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
    use inference::{
        DataType, ModelConfigRequest, ModelInferRequest, ModelInferResponse, ModelMetadataRequest,
    };

    use anyhow::Error;
    use async_stream::stream;
    use dynamo_llm::grpc::service::kserve::KserveService;
    use dynamo_llm::grpc::service::kserve::inference as kserve_inference;
    use dynamo_llm::protocols::{
        Annotated,
        openai::{
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            },
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
        },
        tensor::{NvCreateTensorRequest, NvCreateTensorResponse},
    };
    use dynamo_runtime::{
        CancellationToken,
        pipeline::{
            AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
        },
    };
    use rstest::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::timeout;
    use tonic::{Request, Response, transport::Channel};

    use crate::ports::get_random_port;
    use dynamo_async_openai::types::Prompt;
    use prost::Message;

    struct SplitEngine {}

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

        // Wait for the duration of generation delay to ensure the generate stream
        // has been terminated early (`was_cancelled` remains true).
        async fn wait_for_delay(&self) {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        }
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateCompletionRequest>,
            ManyOut<Annotated<NvCreateCompletionResponse>>,
            Error,
        > for SplitEngine
    {
        async fn generate(
            &self,
            request: SingleIn<NvCreateCompletionRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
            let (request, context) = request.transfer(());
            let ctx = context.context();

            // let generator = NvCreateChatCompletionStreamResponse::generator(request.model.clone());
            let generator = request.response_generator(ctx.id().to_string());

            let word_list: Vec<String> = match request.inner.prompt {
                Prompt::String(str) => str.split(' ').map(|s| s.to_string()).collect(),
                _ => {
                    return Err(Error::msg("SplitEngine only support prompt type String"))?;
                }
            };
            let stream = stream! {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                for word in word_list {
                    yield Annotated::from_data(generator.create_choice(0, Some(word.to_string()), None, None));
                }
            };

            Ok(ResponseStream::new(Box::pin(stream), ctx))
        }
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateCompletionRequest>,
            ManyOut<Annotated<NvCreateCompletionResponse>>,
            Error,
        > for LongRunningEngine
    {
        async fn generate(
            &self,
            request: SingleIn<NvCreateCompletionRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
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

                yield Annotated::<NvCreateCompletionResponse>::from_annotation("event.dynamo.test.sentinel", &"DONE".to_string()).expect("Failed to create annotated response");
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
            Err(Error::msg("Always fail"))?
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
            Err(Error::msg("Always fail"))?
        }
    }

    struct TensorEngine {}

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<NvCreateTensorRequest>,
            ManyOut<Annotated<NvCreateTensorResponse>>,
            Error,
        > for TensorEngine
    {
        async fn generate(
            &self,
            request: SingleIn<NvCreateTensorRequest>,
        ) -> Result<ManyOut<Annotated<NvCreateTensorResponse>>, Error> {
            // Echo input tensor in response, additionally check if there is input tensor
            // name "repeat", if so, send the same response as many time as the value of the tensor
            let (request, context) = request.transfer(());
            let ctx = context.context();

            let repeat_count = request
                .tensors
                .iter()
                .find_map(|t| {
                    if t.metadata.name == "repeat"
                        && let tensor::FlattenTensor::Int32(data) = &t.data
                        && !data.is_empty()
                    {
                        return Some(data[0]);
                    }
                    None
                })
                .unwrap_or(1);
            let stream = async_stream::stream! {
                for _ in 0..repeat_count {
                    yield Annotated::from_data(NvCreateTensorResponse {
                        id: request.id.clone(),
                        model: request.model.clone(),
                        tensors: request.tensors.clone(),
                        parameters: Default::default(),
                    });
                }
            };

            Ok(ResponseStream::new(Box::pin(stream), ctx))
        }
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
        inference::model_infer_request::InferInputTensor {
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
    fn int_input(
        #[default(vec![42,43,44])] input: Vec<u32>,
    ) -> inference::model_infer_request::InferInputTensor {
        inference::model_infer_request::InferInputTensor {
            name: "int_input".into(),
            datatype: "UINT32".into(),
            shape: vec![1],
            contents: Some(inference::InferTensorContents {
                uint_contents: input,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[fixture]
    fn service_with_engines(
        #[default(8990)] port: u16,
    ) -> (
        KserveService,
        Arc<SplitEngine>,
        Arc<AlwaysFailEngine>,
        Arc<LongRunningEngine>,
    ) {
        let service = KserveService::builder().port(port).build().unwrap();
        let manager = service.model_manager();

        let split = Arc::new(SplitEngine {});
        let failure = Arc::new(AlwaysFailEngine {});
        let long_running = Arc::new(LongRunningEngine::new(1_000));

        let mut card = ModelDeploymentCard::with_name_only("split");
        card.model_type = ModelType::Completions;
        card.model_input = ModelInput::Text;
        manager
            .add_completions_model("split", card.mdcsum(), split.clone())
            .unwrap();
        let _ = manager.save_model_card("split", card.clone());

        let mut card = ModelDeploymentCard::with_name_only("failure");
        card.model_type = ModelType::Completions | ModelType::Chat;
        card.model_input = ModelInput::Text;
        manager
            .add_chat_completions_model("failure", card.mdcsum(), failure.clone())
            .unwrap();
        manager
            .add_completions_model("failure", card.mdcsum(), failure.clone())
            .unwrap();
        let _ = manager.save_model_card("failure", card);

        let mut card = ModelDeploymentCard::with_name_only("long_running");
        card.model_type = ModelType::Completions;
        card.model_input = ModelInput::Text;
        manager
            .add_completions_model("long_running", card.mdcsum(), long_running.clone())
            .unwrap();
        let _ = manager.save_model_card("long_running", card);

        (service, split, failure, long_running)
    }

    struct RunningService {
        token: CancellationToken,
    }

    impl RunningService {
        fn spawn(service: KserveService) -> Self {
            let token = CancellationToken::new();
            tokio::spawn({
                let t = token.clone();
                async move { service.run(t).await }
            });
            Self { token }
        }
    }

    impl Drop for RunningService {
        fn drop(&mut self) {
            self.token.cancel();
        }
    }

    // Tests may run in parallel, use this enum to keep track of port used for different
    // test cases
    enum TestPort {
        InferFailure = 8988,
        InferSuccess = 8989,
        StreamInferFailure = 8990,
        StreamInferSuccess = 8991,
        InferCancellation = 8992,
        StreamInferCancellation = 8993,
        ModelInfo = 8994,
        TensorModel = 8995,
        TensorModelTypes = 8996,
        TritonModelConfig = 8997,
    }

    #[rstest]
    #[tokio::test]
    async fn test_infer_failure(
        #[with(TestPort::InferFailure as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        text_input: inference::model_infer_request::InferInputTensor,
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0);

        // create client and send request to unregistered model
        let mut client = get_ready_client(TestPort::InferFailure as u16, 5).await;

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
            "Expected NotFound error for unregistered model, get {}",
            err
        );

        // missing input
        let request = tonic::Request::new(ModelInferRequest {
            model_name: "split".into(),
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
            "Expected InvalidArgument error for missing input, get {}",
            err
        );

        // request streaming
        let request = tonic::Request::new(ModelInferRequest {
            model_name: "split".into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![
                text_input.clone(),
                inference::model_infer_request::InferInputTensor {
                    name: "streaming".into(),
                    datatype: "BOOL".into(),
                    shape: vec![1],
                    contents: Some(inference::InferTensorContents {
                        bool_contents: vec![true],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });

        let response = client.model_infer(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "Expected InvalidArgument error for streaming, get {}",
            err
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
            "Expected Internal error for streaming, get {}",
            err
        );
        assert!(
            err.message().contains("Failed to generate completions:"),
            "Expected error message to contain 'Failed to generate completions:', got: {}",
            err.message()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_infer_success(
        #[with(TestPort::InferSuccess as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        text_input: inference::model_infer_request::InferInputTensor,
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0);

        let mut client = get_ready_client(TestPort::InferSuccess as u16, 5).await;

        let model_name = "split";
        let request = tonic::Request::new(ModelInferRequest {
            model_name: model_name.into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![text_input.clone()],
            ..Default::default()
        });

        let response = client.model_infer(request).await.unwrap();
        validate_infer_response(response, model_name);

        // Input data in raw_input_content
        let mut text_input = text_input.clone();
        text_input.contents = None; // Clear contents to use raw_input_contents
        let text_input_str = "dummy input";
        let input_len = text_input_str.len() as u32;
        let mut serialized_input = input_len.to_le_bytes().to_vec();
        serialized_input.extend_from_slice(text_input_str.as_bytes());
        let request = tonic::Request::new(ModelInferRequest {
            model_name: model_name.into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![text_input],
            raw_input_contents: vec![serialized_input],
            ..Default::default()
        });
        let response = client.model_infer(request).await.unwrap();
        validate_infer_response(response, model_name);
    }

    fn validate_infer_response(response: Response<ModelInferResponse>, model_name: &str) {
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
                        output.shape,
                        vec![1],
                        "Expected 'text_output' to have shape [1]"
                    );
                    let expected_output: Vec<Vec<u8>> = vec!["dummyinput".into()];
                    assert_eq!(
                        output.contents.as_ref().unwrap().bytes_contents,
                        expected_output,
                        "Expected 'text_output' to contain 'dummy input'"
                    );
                }
                "finish_reason" => {
                    assert_eq!(
                        output.datatype, "BYTES",
                        "Expected 'finish_reason' to have datatype 'BYTES'"
                    );
                    assert_eq!(
                        output.shape,
                        vec![1],
                        "Expected 'finish_reason' to have shape [1]"
                    );
                }
                _ => panic!("Unexpected output name: {}", output.name),
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_infer_cancellation(
        #[with(TestPort::InferCancellation as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        text_input: inference::model_infer_request::InferInputTensor,
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0);
        let long_running = service_with_engines.3;

        // create client and send request to unregistered model
        let mut client = get_ready_client(TestPort::InferCancellation as u16, 5).await;

        let model_name = "long_running";
        let request = tonic::Request::new(ModelInferRequest {
            model_name: model_name.into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: vec![text_input],
            ..Default::default()
        });

        assert!(
            !long_running.was_cancelled(),
            "Expected long running engine is not cancelled"
        );

        // Cancelling the request by dropping the request future after 1 second
        let response = match timeout(Duration::from_millis(500), client.model_infer(request)).await
        {
            Ok(_) => Err("Expect request timed out"),
            Err(_) => {
                println!("Cancelled request after 500ms");
                Ok("timed out")
            }
        };
        assert!(response.is_ok(), "Expected client timed out",);
        long_running.wait_for_delay().await;
        assert!(
            long_running.was_cancelled(),
            "Expected long running engine to be cancelled"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_stream_infer_success(
        #[with(TestPort::StreamInferSuccess as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        text_input: inference::model_infer_request::InferInputTensor,
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0);

        // create client and send request to unregistered model
        let mut client = get_ready_client(TestPort::StreamInferSuccess as u16, 5).await;

        let model_name = "split";
        let request_id = "1234";

        // Response streaming true
        {
            let text_input = text_input.clone();
            let outbound = async_stream::stream! {
                let request_count = 1;
                for _ in 0..request_count {
                    let request = ModelInferRequest {
                        model_name: model_name.into(),
                        model_version: "1".into(),
                        id: request_id.into(),
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
                        }],
                        ..Default::default()
                    };

                    yield request;
                }
            };

            let response = client
                .model_stream_infer(Request::new(outbound))
                .await
                .unwrap();
            let mut inbound = response.into_inner();

            let mut response_idx = 0;
            while let Some(response) = inbound.message().await.unwrap() {
                assert!(
                    response.error_message.is_empty(),
                    "Expected successful inference"
                );
                assert!(
                    response.infer_response.is_some(),
                    "Expected successful inference"
                );

                if let Some(response) = &response.infer_response {
                    assert_eq!(
                        response.model_name, model_name,
                        "Expected response of the same model name",
                    );
                    assert_eq!(
                        response.id, request_id,
                        "Expected response ID to match request ID"
                    );
                    let expected_output: Vec<Vec<u8>> = vec!["dummy".into(), "input".into()];
                    for output in &response.outputs {
                        match output.name.as_str() {
                            "text_output" => {
                                assert_eq!(
                                    output.datatype, "BYTES",
                                    "Expected 'text_output' to have datatype 'BYTES'"
                                );
                                assert_eq!(
                                    output.shape,
                                    vec![1],
                                    "Expected 'text_output' to have shape [1]"
                                );
                                assert_eq!(
                                    output.contents.as_ref().unwrap().bytes_contents,
                                    vec![expected_output[response_idx].clone()],
                                    "Expected 'text_output' to contain 'dummy input'"
                                );
                            }
                            "finish_reason" => {
                                assert_eq!(
                                    output.datatype, "BYTES",
                                    "Expected 'finish_reason' to have datatype 'BYTES'"
                                );
                                assert_eq!(
                                    output.shape,
                                    vec![1],
                                    "Expected 'finish_reason' to have shape [1]"
                                );
                            }
                            _ => panic!("Unexpected output name: {}", output.name),
                        }
                    }
                }
                response_idx += 1;
            }
            assert_eq!(response_idx, 2, "Expected 2 responses")
        }

        // Response streaming false
        {
            let text_input = text_input.clone();
            let outbound = async_stream::stream! {
                let request_count = 2;
                for idx in 0..request_count {
                    let request = ModelInferRequest {
                        model_name: model_name.into(),
                        model_version: "1".into(),
                        id: format!("{idx}"),
                        inputs: vec![text_input.clone()],
                        ..Default::default()
                    };

                    yield request;
                }
            };

            let response = client
                .model_stream_infer(Request::new(outbound))
                .await
                .unwrap();
            let mut inbound = response.into_inner();

            let mut response_idx = 0;
            while let Some(response) = inbound.message().await.unwrap() {
                assert!(
                    response.error_message.is_empty(),
                    "Expected successful inference"
                );
                assert!(
                    response.infer_response.is_some(),
                    "Expected successful inference"
                );

                // Each response is the complete inference
                if let Some(response) = &response.infer_response {
                    assert_eq!(
                        response.model_name, model_name,
                        "Expected response of the same model name",
                    );
                    // [gluo NOTE] Here we assume the responses across requests are
                    // processed in the order of receiving requests, which is not true
                    // if we improve stream handling in gRPC frontend. Consider:
                    //   time 0: request 0 -> long running -> response 0 (time 5)
                    //   time 1: request 1 -> short running -> response 1 (time 2)
                    // We expect response 1 to be received before response 0 as their
                    // requests are independent from each other.
                    assert_eq!(
                        response.id,
                        format!("{response_idx}"),
                        "Expected response ID to match request ID"
                    );
                    for output in &response.outputs {
                        match output.name.as_str() {
                            "text_output" => {
                                assert_eq!(
                                    output.datatype, "BYTES",
                                    "Expected 'text_output' to have datatype 'BYTES'"
                                );
                                assert_eq!(
                                    output.shape,
                                    vec![1],
                                    "Expected 'text_output' to have shape [1]"
                                );
                                let expected_output: Vec<Vec<u8>> = vec!["dummyinput".into()];
                                assert_eq!(
                                    output.contents.as_ref().unwrap().bytes_contents,
                                    expected_output,
                                    "Expected 'text_output' to contain 'dummyinput'"
                                );
                            }
                            "finish_reason" => {
                                assert_eq!(
                                    output.datatype, "BYTES",
                                    "Expected 'finish_reason' to have datatype 'BYTES'"
                                );
                                assert_eq!(
                                    output.shape,
                                    vec![1],
                                    "Expected 'finish_reason' to have shape [1]"
                                );
                            }
                            _ => panic!("Unexpected output name: {}", output.name),
                        }
                    }
                }
                response_idx += 1;
            }
            assert_eq!(
                response_idx, 2,
                "Expected 2 responses, each for one of the two requests"
            )
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_stream_infer_failure(
        #[with(TestPort::StreamInferFailure as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        text_input: inference::model_infer_request::InferInputTensor,
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0);

        // create client and send request to unregistered model
        let mut client = get_ready_client(TestPort::StreamInferFailure as u16, 5).await;

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

        let response = client
            .model_stream_infer(Request::new(outbound))
            .await
            .unwrap();
        let mut inbound = response.into_inner();

        loop {
            match inbound.message().await {
                Ok(Some(_)) => {
                    panic!("Expecting failure in the stream");
                }
                Err(err) => {
                    assert_eq!(
                        err.code(),
                        tonic::Code::Internal,
                        "Expected Internal error for streaming, get {}",
                        err
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

    #[rstest]
    #[tokio::test]
    async fn test_stream_infer_cancellation(
        #[with(TestPort::StreamInferCancellation as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        text_input: inference::model_infer_request::InferInputTensor,
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0);
        let long_running = service_with_engines.3;

        // create client and send request to unregistered model
        let mut client = get_ready_client(TestPort::StreamInferCancellation as u16, 5).await;

        let model_name = "long_running";
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

        assert!(
            !long_running.was_cancelled(),
            "Expected long running engine is still running"
        );

        // Cancelling the request by dropping the request future after 1 second
        let response = match timeout(
            Duration::from_millis(500),
            client.model_stream_infer(Request::new(outbound)),
        )
        .await
        {
            Ok(response) => response.unwrap(),
            Err(_) => {
                panic!("Expected response stream is returned immediately");
            }
        };
        std::mem::drop(response); // Drop the response to cancel the stream

        long_running.wait_for_delay().await;
        assert!(
            long_running.was_cancelled(),
            "Expected long running engine to be cancelled"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_model_info(
        #[with(TestPort::ModelInfo as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0);

        // create client and send request to unregistered model
        let mut client = get_ready_client(TestPort::ModelInfo as u16, 5).await;

        // Failure unknown_model
        let request = tonic::Request::new(ModelMetadataRequest {
            name: "Tonic".into(),
            version: "".into(),
        });

        let response = client.model_metadata(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::NotFound,
            "Expected NotFound error for unregistered model, get {}",
            err
        );

        let request = tonic::Request::new(ModelConfigRequest {
            name: "Tonic".into(),
            version: "".into(),
        });

        let response = client.model_config(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::NotFound,
            "Expected NotFound error for unregistered model, get {}",
            err
        );

        // Success metadata
        let model_name = "split";
        let request = tonic::Request::new(ModelMetadataRequest {
            name: model_name.into(),
            version: "1".into(),
        });

        let response = client.model_metadata(request).await.unwrap();
        assert_eq!(
            response.get_ref().name,
            model_name,
            "Expected response of the same model name",
        );
        // input
        for io in &response.get_ref().inputs {
            match io.name.as_str() {
                "text_input" => {
                    assert_eq!(
                        io.datatype, "BYTES",
                        "Expected 'text_input' to have datatype 'BYTES'"
                    );
                    assert_eq!(
                        io.shape,
                        vec![1],
                        "Expected 'text_output' to have shape [1]"
                    );
                }
                "streaming" => {
                    assert_eq!(
                        io.datatype, "BOOL",
                        "Expected 'streaming' to have datatype 'BOOL'"
                    );
                    assert_eq!(io.shape, vec![1], "Expected 'streaming' to have shape [1]");
                }
                _ => panic!("Unexpected output name: {}", io.name),
            }
        }
        // output
        for io in &response.get_ref().outputs {
            match io.name.as_str() {
                "text_output" => {
                    assert_eq!(
                        io.datatype, "BYTES",
                        "Expected 'text_output' to have datatype 'BYTES'"
                    );
                    assert_eq!(
                        io.shape,
                        vec![-1],
                        "Expected 'text_output' to have shape [-1]"
                    );
                }
                "finish_reason" => {
                    assert_eq!(
                        io.datatype, "BYTES",
                        "Expected 'finish_reason' to have datatype 'BYTES'"
                    );
                    assert_eq!(
                        io.shape,
                        vec![-1],
                        "Expected 'finish_reason' to have shape [-1]"
                    );
                }
                _ => panic!("Unexpected output name: {}", io.name),
            }
        }

        // success config
        let request = tonic::Request::new(ModelConfigRequest {
            name: model_name.into(),
            version: "1".into(),
        });

        let response = client
            .model_config(request)
            .await
            .unwrap()
            .into_inner()
            .config;
        let Some(config) = response else {
            panic!("Expected Some(config), got None");
        };
        assert_eq!(
            config.name, model_name,
            "Expected response of the same model name",
        );
        // input
        for io in &config.input {
            match io.name.as_str() {
                "text_input" => {
                    assert_eq!(
                        io.data_type,
                        DataType::TypeString as i32,
                        "Expected 'text_input' to have datatype 'TYPE_STRING'"
                    );
                    assert_eq!(io.dims, vec![1], "Expected 'text_output' to have shape [1]");
                }
                "streaming" => {
                    assert_eq!(
                        io.data_type,
                        DataType::TypeBool as i32,
                        "Expected 'streaming' to have datatype 'TYPE_BOOL'"
                    );
                    assert_eq!(io.dims, vec![1], "Expected 'streaming' to have shape [1]");
                }
                _ => panic!("Unexpected output name: {}", io.name),
            }
        }
        // output
        for io in &config.output {
            match io.name.as_str() {
                "text_output" => {
                    assert_eq!(
                        io.data_type,
                        DataType::TypeString as i32,
                        "Expected 'text_output' to have datatype 'TYPE_STRING'"
                    );
                    assert_eq!(
                        io.dims,
                        vec![-1],
                        "Expected 'text_output' to have shape [-1]"
                    );
                }
                "finish_reason" => {
                    assert_eq!(
                        io.data_type,
                        DataType::TypeString as i32,
                        "Expected 'finish_reason' to have datatype 'TYPE_STRING'"
                    );
                    assert_eq!(
                        io.dims,
                        vec![-1],
                        "Expected 'finish_reason' to have shape [-1]"
                    );
                }
                _ => panic!("Unexpected output name: {}", io.name),
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_tensor_infer_dtypes(
        #[with(TestPort::TensorModelTypes as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        int_input: inference::model_infer_request::InferInputTensor,
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0.clone());

        let mut client = get_ready_client(TestPort::TensorModelTypes as u16, 5).await;

        // Register a tensor model
        let mut card = ModelDeploymentCard::with_name_only("tensor");
        card.model_type = ModelType::TensorBased;
        card.model_input = ModelInput::Tensor;
        card.runtime_config = ModelRuntimeConfig {
            tensor_model_config: Some(tensor::TensorModelConfig {
                name: "tensor".to_string(),
                inputs: vec![tensor::TensorMetadata {
                    name: "input".to_string(),
                    data_type: tensor::DataType::Int32,
                    shape: vec![1],
                    parameters: Default::default(),
                }],
                outputs: vec![tensor::TensorMetadata {
                    name: "output".to_string(),
                    data_type: tensor::DataType::Bool,
                    shape: vec![-1],
                    parameters: Default::default(),
                }],
                triton_model_config: None,
            }),
            ..Default::default()
        };
        let tensor = Arc::new(TensorEngine {});
        service_with_engines
            .0
            .model_manager()
            .add_tensor_model("tensor", card.mdcsum(), tensor.clone())
            .unwrap();
        let _ = service_with_engines
            .0
            .model_manager()
            .save_model_card("key", card);

        let model_name = "tensor";
        let inputs = vec![int_input.clone()];
        let request = tonic::Request::new(ModelInferRequest {
            model_name: model_name.into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: inputs.clone(),
            ..Default::default()
        });

        let response = client.model_infer(request).await.unwrap();
        validate_tensor_response(
            response,
            model_name,
            inputs,
            std::collections::HashMap::new(),
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_triton_model_config(
        #[with(TestPort::TritonModelConfig as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
    ) {
        // start server
        let _running = RunningService::spawn(service_with_engines.0.clone());

        let mut client = get_ready_client(TestPort::TritonModelConfig as u16, 5).await;

        let model_name = "tensor";
        let expected_model_config = inference::ModelConfig {
            name: model_name.to_string(),
            platform: "custom".to_string(),
            backend: "custom".to_string(),
            input: vec![
                inference::ModelInput {
                    name: "input".to_string(),
                    data_type: DataType::TypeInt32 as i32,
                    dims: vec![1],
                    optional: false,
                    ..Default::default()
                },
                inference::ModelInput {
                    name: "optional_input".to_string(),
                    data_type: DataType::TypeInt32 as i32,
                    dims: vec![1],
                    optional: true,
                    ..Default::default()
                },
            ],
            output: vec![inference::ModelOutput {
                name: "output".to_string(),
                data_type: DataType::TypeBool as i32,
                dims: vec![-1],
                ..Default::default()
            }],
            model_transaction_policy: Some(inference::ModelTransactionPolicy { decoupled: true }),
            ..Default::default()
        };

        let mut buf = vec![];
        expected_model_config.encode(&mut buf).unwrap();

        // Register a tensor model
        let mut card = ModelDeploymentCard::with_name_only(model_name);
        card.model_type = ModelType::TensorBased;
        card.model_input = ModelInput::Tensor;
        card.runtime_config = ModelRuntimeConfig {
            tensor_model_config: Some(tensor::TensorModelConfig {
                triton_model_config: Some(buf.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tensor = Arc::new(TensorEngine {});
        service_with_engines
            .0
            .model_manager()
            .add_tensor_model("tensor", card.mdcsum(), tensor.clone())
            .unwrap();
        let _ = service_with_engines
            .0
            .model_manager()
            .save_model_card("key", card);

        // success config
        let request = tonic::Request::new(ModelConfigRequest {
            name: model_name.into(),
            version: "".into(),
        });

        let response = client
            .model_config(request)
            .await
            .unwrap()
            .into_inner()
            .config;
        let Some(config) = response else {
            panic!("Expected Some(config), got None");
        };
        assert_eq!(
            config, expected_model_config,
            "Expected same model config to be returned",
        );

        // Pass config with both TensorModelConfig and triton_model_config,
        // check if the Triton model config is used.
        let _ = service_with_engines
            .0
            .model_manager()
            .remove_model_card("key");
        let mut card = ModelDeploymentCard::with_name_only(model_name);
        card.model_type = ModelType::TensorBased;
        card.model_input = ModelInput::Tensor;
        let mut card = ModelDeploymentCard::with_name_only("tensor");
        card.model_type = ModelType::TensorBased;
        card.model_input = ModelInput::Tensor;
        card.runtime_config = ModelRuntimeConfig {
            tensor_model_config: Some(tensor::TensorModelConfig {
                name: "tensor".to_string(),
                inputs: vec![tensor::TensorMetadata {
                    name: "input".to_string(),
                    data_type: tensor::DataType::Int32,
                    shape: vec![1],
                    parameters: Default::default(),
                }],
                outputs: vec![tensor::TensorMetadata {
                    name: "output".to_string(),
                    data_type: tensor::DataType::Bool,
                    shape: vec![-1],
                    parameters: Default::default(),
                }],
                triton_model_config: Some(buf.clone()),
            }),
            ..Default::default()
        };
        let _ = service_with_engines
            .0
            .model_manager()
            .save_model_card("key", card);
        let request = tonic::Request::new(ModelConfigRequest {
            name: model_name.into(),
            version: "".into(),
        });

        let response = client
            .model_config(request)
            .await
            .unwrap()
            .into_inner()
            .config;
        let Some(config) = response else {
            panic!("Expected Some(config), got None");
        };
        assert_eq!(
            config, expected_model_config,
            "Expected same model config to be returned",
        );

        // Test invalid triton model config
        let _ = service_with_engines
            .0
            .model_manager()
            .remove_model_card("key");
        let mut card = ModelDeploymentCard::with_name_only(model_name);
        card.model_type = ModelType::TensorBased;
        card.model_input = ModelInput::Tensor;
        card.runtime_config = ModelRuntimeConfig {
            tensor_model_config: Some(tensor::TensorModelConfig {
                triton_model_config: Some(vec![1, 2, 3, 4, 5]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let _ = service_with_engines
            .0
            .model_manager()
            .save_model_card("key", card);

        // success config
        let request = tonic::Request::new(ModelConfigRequest {
            name: model_name.into(),
            version: "".into(),
        });

        let response = client.model_config(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "Expected InvalidArgument error, get {}",
            err
        );
        assert!(
            err.message().contains("failed to decode Protobuf message"),
            "Expected error message to contain 'failed to decode Protobuf message', got: {}",
            err.message()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_tensor_infer(
        #[with(TestPort::TensorModel as u16)] service_with_engines: (
            KserveService,
            Arc<SplitEngine>,
            Arc<AlwaysFailEngine>,
            Arc<LongRunningEngine>,
        ),
        text_input: inference::model_infer_request::InferInputTensor,
    ) {
        // add tensor model

        // Failure, model registered as Tensor but does not provide model config (in runtime config)
        let mut card = ModelDeploymentCard::with_name_only("tensor");
        card.model_type = ModelType::TensorBased;
        card.model_input = ModelInput::Tensor;
        let tensor = Arc::new(TensorEngine {});
        service_with_engines
            .0
            .model_manager()
            .add_tensor_model("tensor", card.mdcsum(), tensor.clone())
            .unwrap();

        // start server
        let _running = RunningService::spawn(service_with_engines.0.clone());

        let mut client = get_ready_client(TestPort::TensorModel as u16, 5).await;

        let request = tonic::Request::new(ModelMetadataRequest {
            name: "tensor".into(),
            version: "".into(),
        });

        let _ = service_with_engines
            .0
            .model_manager()
            .save_model_card("key", card);

        let response = client.model_metadata(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "Expected InvalidArgument error for unregistered model, get {}",
            err
        );
        assert!(
            err.message().contains("no model config is provided"),
            "Expected error message to contain 'no model config is provided', got: {}",
            err.message()
        );

        let request = tonic::Request::new(ModelConfigRequest {
            name: "tensor".into(),
            version: "".into(),
        });

        let response = client.model_config(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "Expected InvalidArgument error for unregistered model, get {}",
            err
        );
        assert!(
            err.message().contains("no model config is provided"),
            "Expected error message to contain 'no model config is provided', got: {}",
            err.message()
        );

        // Change model entry to have model config
        service_with_engines
            .0
            .model_manager()
            .remove_model_card("key");
        let mut card = ModelDeploymentCard::with_name_only("tensor");
        card.model_type = ModelType::TensorBased;
        card.model_input = ModelInput::Tensor;
        card.runtime_config = ModelRuntimeConfig {
            tensor_model_config: Some(tensor::TensorModelConfig {
                name: "tensor".to_string(),
                inputs: vec![tensor::TensorMetadata {
                    name: "input".to_string(),
                    data_type: tensor::DataType::Bytes,
                    shape: vec![1],
                    parameters: Default::default(),
                }],
                outputs: vec![tensor::TensorMetadata {
                    name: "output".to_string(),
                    data_type: tensor::DataType::Bool,
                    shape: vec![-1],
                    parameters: Default::default(),
                }],
                triton_model_config: None,
            }),
            ..Default::default()
        };
        let _ = service_with_engines
            .0
            .model_manager()
            .save_model_card("key", card);

        // Success
        let request = tonic::Request::new(ModelMetadataRequest {
            name: "tensor".into(),
            version: "".into(),
        });
        let response = client.model_metadata(request).await.unwrap();
        assert_eq!(
            response.get_ref().name,
            "tensor",
            "Expected response of the same model name",
        );
        // input
        for io in &response.get_ref().inputs {
            match io.name.as_str() {
                "input" => {
                    assert_eq!(
                        io.datatype, "BYTES",
                        "Expected 'input' to have datatype 'BYTES'"
                    );
                    assert_eq!(io.shape, vec![1], "Expected 'input' to have shape [1]");
                }
                _ => panic!("Unexpected output name: {}", io.name),
            }
        }
        // output
        for io in &response.get_ref().outputs {
            match io.name.as_str() {
                "output" => {
                    assert_eq!(
                        io.datatype, "BOOL",
                        "Expected 'output' to have datatype 'BOOL'"
                    );
                    assert_eq!(io.shape, vec![-1], "Expected 'output' to have shape [-1]");
                }
                _ => panic!("Unexpected output name: {}", io.name),
            }
        }

        let model_name = "tensor";
        let inputs = vec![text_input.clone()];
        let request = tonic::Request::new(ModelInferRequest {
            model_name: model_name.into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: inputs.clone(),
            ..Default::default()
        });

        let response = client.model_infer(request).await.unwrap();
        validate_tensor_response(
            response,
            model_name,
            inputs,
            std::collections::HashMap::new(),
        );

        // streaming response in model_infer(), expect failure
        let repeat = inference::model_infer_request::InferInputTensor {
            name: "repeat".into(),
            datatype: "INT32".into(),
            shape: vec![1],
            contents: Some(inference::InferTensorContents {
                int_contents: vec![2],
                ..Default::default()
            }),
            ..Default::default()
        };
        let inputs = vec![text_input.clone(), repeat.clone()];
        let request = tonic::Request::new(ModelInferRequest {
            model_name: model_name.into(),
            model_version: "1".into(),
            id: "1234".into(),
            inputs: inputs.clone(),
            ..Default::default()
        });

        let response = client.model_infer(request).await;
        assert!(response.is_err());
        let err = response.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::Internal,
            "Expected Internal error for trying to stream response in ModelInfer, get {}",
            err
        );
        // assert "stream" in error message
        assert!(
            err.message()
                .contains("Multiple responses in non-streaming mode"),
            "Expected error message to contain 'Multiple responses in non-streaming mode', got: {}",
            err.message()
        );

        // model_stream_infer() and raw_input_contents
        {
            let inputs = vec![text_input.clone(), repeat.clone()];
            let outbound = async_stream::stream! {
                let request_count = 1;
                for _ in 0..request_count {
                    let mut text_input = text_input.clone();
                    text_input.contents = None; // Clear contents to use raw_input_contents
                    let text_input_str = "dummy input";
                    let input_len = text_input_str.len() as u32;
                    let mut serialized_text_input = input_len.to_le_bytes().to_vec();
                    serialized_text_input.extend_from_slice(text_input_str.as_bytes());

                    let mut repeat = repeat.clone();
                    repeat.contents = None; // Clear contents to use raw_input_contents
                    let serialized_repeat = 2i32.to_le_bytes().to_vec();

                    let request = ModelInferRequest {
                        model_name: model_name.into(),
                        model_version: "1".into(),
                        id: "1234".into(),
                        inputs: vec![text_input.clone(), repeat.clone()],
                        raw_input_contents: vec![serialized_text_input, serialized_repeat],
                        ..Default::default()
                    };

                    yield request;
                }
            };

            let response = client
                .model_stream_infer(Request::new(outbound))
                .await
                .unwrap();
            let mut inbound = response.into_inner();

            let mut response_idx = 0;
            while let Some(response) = inbound.message().await.unwrap() {
                assert!(
                    response.error_message.is_empty(),
                    "Expected successful inference"
                );
                assert!(
                    response.infer_response.is_some(),
                    "Expected successful inference"
                );

                let text_input_str = "dummy input";
                let input_len = text_input_str.len() as u32;
                let mut serialized_text_input = input_len.to_le_bytes().to_vec();
                serialized_text_input.extend_from_slice(text_input_str.as_bytes());
                let serialized_repeat = 2i32.to_le_bytes().to_vec();
                if let Some(response) = &response.infer_response {
                    validate_tensor_response(
                        Response::new(response.clone()),
                        model_name,
                        inputs.clone(),
                        std::collections::HashMap::from([
                            ("text_input".into(), serialized_text_input.clone()),
                            ("repeat".into(), serialized_repeat.clone()),
                        ]),
                    );
                }
                response_idx += 1;
            }
            assert_eq!(response_idx, 2, "Expected 2 responses")
        }
    }

    fn validate_tensor_response(
        response: Response<ModelInferResponse>,
        model_name: &str,
        inputs: Vec<inference::model_infer_request::InferInputTensor>,
        expected_raw_outputs: std::collections::HashMap<String, Vec<u8>>,
    ) {
        assert_eq!(
            response.get_ref().model_name,
            model_name,
            "Expected response of the same model name",
        );
        assert_eq!(
            response.get_ref().model_version,
            "1",
            "Expected response of the same model version"
        );
        assert_eq!(
            response.get_ref().id,
            "1234",
            "Expected response of the same request ID"
        );
        assert_eq!(
            response.get_ref().outputs.len(),
            inputs.len(),
            "Expected the same number of outputs as inputs",
        );
        assert_eq!(
            response.get_ref().raw_output_contents.len(),
            expected_raw_outputs.len(),
            "Expected the same number of raw_output_contents as expected_raw_outputs",
        );
        for (idx, output) in response.get_ref().outputs.iter().enumerate() {
            let mut found = false;
            for input in &inputs {
                if input.name != output.name {
                    continue;
                }
                assert_eq!(
                    output.name, input.name,
                    "Expected output name to be '{}', got '{}'",
                    input.name, output.name
                );
                assert_eq!(
                    output.datatype, input.datatype,
                    "Expected output datatype to be '{}', got '{}'",
                    input.datatype, output.datatype
                );
                assert_eq!(
                    output.shape, input.shape,
                    "Expected output shape to be '{:?}', got '{:?}'",
                    input.shape, output.shape
                );
                if expected_raw_outputs.contains_key(&output.name) {
                    assert_eq!(
                        &response.get_ref().raw_output_contents[idx],
                        expected_raw_outputs.get(&output.name).unwrap(),
                        "Expected output contents to match raw_input_contents",
                    );
                } else {
                    assert_eq!(
                        output.contents, input.contents,
                        "Expected output contents to match input contents",
                    );
                }
                found = true;
                break;
            }
            if !found {
                panic!("Unexpected output name: {}", output.name);
            }
        }
    }

    #[test]
    fn test_parameter_conversion_round_trip() {
        use kserve_inference::infer_parameter::ParameterChoice;

        // Test all 5 parameter types for round-trip conversion
        let test_cases = vec![
            ("bool_param", ParameterChoice::BoolParam(true)),
            ("int64_param", ParameterChoice::Int64Param(42)),
            (
                "string_param",
                ParameterChoice::StringParam("test_value".to_string()),
            ),
            ("double_param", ParameterChoice::DoubleParam(2.5)),
            ("uint64_param", ParameterChoice::Uint64Param(9999)),
        ];

        for (name, choice) in test_cases {
            let kserve_param = kserve_inference::InferParameter {
                parameter_choice: Some(choice.clone()),
            };

            // Convert KServe -> Dynamo -> KServe
            let dynamo_param =
                dynamo_llm::grpc::service::tensor::kserve_param_to_dynamo(name, &kserve_param)
                    .expect("Conversion to Dynamo should succeed");

            let back_to_kserve =
                dynamo_llm::grpc::service::tensor::dynamo_param_to_kserve(&dynamo_param);

            // Verify round-trip preserves the value
            assert_eq!(
                kserve_param.parameter_choice, back_to_kserve.parameter_choice,
                "Parameter '{}' failed round-trip conversion",
                name
            );
        }
    }

    #[test]
    fn test_parameter_conversion_error_cases() {
        // Test conversion of parameter with no value
        let empty_param = kserve_inference::InferParameter {
            parameter_choice: None,
        };

        let result =
            dynamo_llm::grpc::service::tensor::kserve_param_to_dynamo("empty_param", &empty_param);

        assert!(
            result.is_err(),
            "Expected error for parameter with no value"
        );
        assert!(
            result.unwrap_err().message().contains("has no value"),
            "Expected error message about missing value"
        );
    }

    async fn wait_for_http_ready(port: u16, timeout_secs: u64) {
        let client = reqwest::Client::new();
        let start = tokio::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);
        let url = format!("http://localhost:{}/metrics", port);

        loop {
            match client.get(&url).send().await {
                Ok(_) => return,
                Err(_) if start.elapsed() < timeout => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(e) => panic!("HTTP service failed to start within timeout: {}", e),
            }
        }
    }

    fn assert_metric_value(metrics: &str, model_name: &str, endpoint: &str, expected_count: u32) {
        // Find the metric line that contains both the model and endpoint labels
        let metric_line = metrics
            .lines()
            .find(|line| {
                line.starts_with("dynamo_frontend_requests_total{")
                    && line.contains(&format!("model=\"{}\"", model_name))
                    && line.contains(&format!("endpoint=\"{}\"", endpoint))
            })
            .unwrap_or_else(|| {
                panic!(
                    "Could not find metric for model='{}' endpoint='{}' in metrics:\n{}",
                    model_name, endpoint, metrics
                )
            });

        let value_str = metric_line.split_whitespace().last().unwrap_or("0");
        let actual_count: u32 = value_str.parse().unwrap_or(0);

        assert_eq!(
            actual_count, expected_count,
            "Expected {} requests for model='{}' endpoint='{}', but found {}",
            expected_count, model_name, endpoint, actual_count
        );
    }

    #[tokio::test]
    async fn test_kserve_grpc_metrics_endpoint() {
        let grpc_port = get_random_port().await;
        let http_metrics_port = get_random_port().await;

        let service = KserveService::builder()
            .port(grpc_port)
            .http_metrics_port(http_metrics_port)
            .build()
            .unwrap();

        let state = service.state_clone();
        let manager = state.manager();

        // Register completion model
        let mut card = ModelDeploymentCard::with_name_only("test_model");
        card.model_type = ModelType::Completions;
        card.model_input = ModelInput::Text;
        manager
            .add_completions_model("test_model", card.mdcsum(), Arc::new(SplitEngine {}))
            .unwrap();
        manager.save_model_card("test_model", card).unwrap();

        // Register tensor model
        let mut tensor_card = ModelDeploymentCard::with_name_only("test_tensor_model");
        tensor_card.model_type = ModelType::TensorBased;
        tensor_card.model_input = ModelInput::Tensor;
        tensor_card.runtime_config = ModelRuntimeConfig {
            tensor_model_config: Some(tensor::TensorModelConfig {
                name: "test_tensor_model".to_string(),
                inputs: vec![tensor::TensorMetadata {
                    name: "input".to_string(),
                    data_type: tensor::DataType::Int32,
                    shape: vec![1],
                    parameters: Default::default(),
                }],
                outputs: vec![tensor::TensorMetadata {
                    name: "output".to_string(),
                    data_type: tensor::DataType::Int32,
                    shape: vec![1],
                    parameters: Default::default(),
                }],
                triton_model_config: None,
            }),
            ..Default::default()
        };
        manager
            .add_tensor_model(
                "test_tensor_model",
                tensor_card.mdcsum(),
                Arc::new(TensorEngine {}),
            )
            .unwrap();
        manager
            .save_model_card("test_tensor_model", tensor_card)
            .unwrap();

        // Start services
        let cancel_token = CancellationToken::new();
        let grpc_task = service.spawn(cancel_token.clone()).await;
        let http_task = service.http_service().spawn(cancel_token.clone()).await;

        // Wait for services to be ready
        let mut grpc_client = get_ready_client(grpc_port, 10).await;
        wait_for_http_ready(http_metrics_port, 10).await;

        // Test completion model
        grpc_client
            .model_infer(Request::new(ModelInferRequest {
                model_name: "test_model".into(),
                model_version: "1".into(),
                id: "test-metrics".into(),
                inputs: vec![inference::model_infer_request::InferInputTensor {
                    name: "text_input".into(),
                    datatype: "BYTES".into(),
                    shape: vec![1],
                    contents: Some(inference::InferTensorContents {
                        bytes_contents: vec!["test input".into()],
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }))
            .await
            .expect("Inference request failed");

        // Test tensor model
        grpc_client
            .model_infer(Request::new(ModelInferRequest {
                model_name: "test_tensor_model".into(),
                model_version: "1".into(),
                id: "test-tensor-metrics".into(),
                inputs: vec![inference::model_infer_request::InferInputTensor {
                    name: "input".into(),
                    datatype: "INT32".into(),
                    shape: vec![1],
                    contents: Some(inference::InferTensorContents {
                        int_contents: vec![42],
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }))
            .await
            .expect("Tensor inference request failed");

        // Verify metrics are exposed via HTTP endpoint
        let metrics_url = format!("http://localhost:{}/metrics", http_metrics_port);
        let metrics_body = reqwest::get(&metrics_url)
            .await
            .expect("Failed to fetch metrics")
            .text()
            .await
            .unwrap();

        // Verify metrics are present and have correct values
        assert!(
            metrics_body.contains("dynamo_frontend_inflight_requests"),
            "Metrics should contain inflight gauge"
        );
        assert_metric_value(&metrics_body, "test_model", "completions", 1);
        assert_metric_value(&metrics_body, "test_tensor_model", "tensor", 1);

        // Clean up
        cancel_token.cancel();
        let _ = tokio::join!(grpc_task, http_task);
    }
}

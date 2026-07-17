// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Error;
use base64::Engine as _;
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::protocols::Annotated;
use dynamo_llm::protocols::openai::transcriptions::{
    NvAudioTranscriptionResponse, NvCreateAudioTranscriptionRequest,
};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};
use reqwest::StatusCode;

const MODEL: &str = "test-whisper";

struct TestTranscriptionEngine {
    fail: bool,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateAudioTranscriptionRequest>,
        ManyOut<Annotated<NvAudioTranscriptionResponse>>,
        Error,
    > for TestTranscriptionEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateAudioTranscriptionRequest>,
    ) -> Result<ManyOut<Annotated<NvAudioTranscriptionResponse>>, Error> {
        let (request, context) = request.transfer(());
        let response = if self.fail {
            let error = DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .message("unsupported transcription language")
                .build();
            Annotated {
                data: None,
                id: None,
                event: Some("error".to_string()),
                comment: None,
                error: Some(error),
            }
        } else {
            let audio = base64::engine::general_purpose::STANDARD
                .decode(&request.audio_b64)
                .expect("frontend should produce valid base64");
            Annotated::from_data(NvAudioTranscriptionResponse {
                text: format!("{}:{}", request.filename, audio.len()),
                usage: None,
                language: request.language,
                duration: None,
                segments: None,
                words: None,
            })
        };
        let stream = futures::stream::iter([response]);

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

struct TestService {
    base_url: String,
    client: reqwest::Client,
    cancel: CancellationToken,
    join: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl TestService {
    async fn start(fail: bool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind test listener");
        let port = listener
            .local_addr()
            .expect("test listener has no local address")
            .port();
        let service = HttpService::builder()
            .host("127.0.0.1")
            .port(port)
            .build()
            .expect("failed to build HTTP service");
        service.enable_model_endpoint(EndpointType::Transcriptions, true);
        service
            .model_manager()
            .add_transcriptions_model(MODEL, "0", Arc::new(TestTranscriptionEngine { fail }))
            .expect("failed to register transcription model");

        let cancel = CancellationToken::new();
        let join = service.spawn_with_listener(cancel.clone(), listener).await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("failed to build HTTP client");
        let base_url = format!("http://127.0.0.1:{port}");

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if client
                    .get(format!("{base_url}/health"))
                    .send()
                    .await
                    .is_ok_and(|response| response.status().is_success())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("HTTP service did not become healthy");

        Self {
            base_url,
            client,
            cancel,
            join,
        }
    }

    async fn shutdown(self) {
        self.cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), self.join)
            .await
            .expect("HTTP service did not stop")
            .expect("HTTP service task panicked")
            .expect("HTTP service returned an error");
    }

    async fn transcribe(&self) -> reqwest::Response {
        let form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(b"audio".to_vec())
                    .file_name("sample.wav")
                    .mime_str("audio/wav")
                    .unwrap(),
            )
            .text("language", "en")
            .text("response_format", "json");

        self.client
            .post(format!("{}/v1/audio/transcriptions", self.base_url))
            .multipart(form)
            .send()
            .await
            .expect("transcription request failed")
    }
}

#[tokio::test]
async fn multipart_request_uses_default_model_and_returns_json() {
    let service = TestService::start(false).await;

    let response = service.transcribe().await;

    assert_eq!(response.status(), StatusCode::OK);
    let response: NvAudioTranscriptionResponse = response.json().await.unwrap();
    assert_eq!(response.text, "sample.wav:5");
    assert_eq!(response.language.as_deref(), Some("en"));
    service.shutdown().await;
}

#[tokio::test]
async fn annotated_invalid_argument_returns_bad_request() {
    let service = TestService::start(true).await;

    let response = service.transcribe().await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    service.shutdown().await;
}

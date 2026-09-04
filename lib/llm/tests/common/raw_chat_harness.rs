// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full chat-completions pipeline backed by deterministic raw backend output.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Error, Result, anyhow};
use dynamo_llm::backend::Backend;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use dynamo_llm::protocols::{
    Annotated,
    common::llm_backend::LLMEngineOutput,
    openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, Operator, ResponseStream, ServiceBackend,
    ServiceFrontend, SingleIn, Source, async_trait,
};
use tokio::sync::Mutex;

use super::ports::bind_random_port;

pub const MODEL: &str = "harness-model";
pub type WorkerScript = Vec<LLMEngineOutput>;

/// Captures fully preprocessed requests and emits one raw backend script per request.
pub struct ScriptedBackendEngine {
    scripts: Mutex<VecDeque<WorkerScript>>,
    requests: Mutex<Vec<PreprocessedRequest>>,
}

impl ScriptedBackendEngine {
    pub fn new(scripts: impl IntoIterator<Item = WorkerScript>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub async fn take_requests(&self) -> Vec<PreprocessedRequest> {
        std::mem::take(&mut *self.requests.lock().await)
    }

    pub async fn remaining_scripts(&self) -> usize {
        self.scripts.lock().await.len()
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for ScriptedBackendEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        self.requests.lock().await.push(request);

        let script = self
            .scripts
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| anyhow!("ScriptedBackendEngine received an unexpected request"))?;
        let output = futures::stream::iter(script.into_iter().map(Annotated::from_data));
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

/// Real HTTP and preprocessing/postprocessing pipeline with only generation mocked.
pub struct RawChatHarness {
    pub base_url: String,
    pub client: reqwest::Client,
    pub engine: Arc<ScriptedBackendEngine>,
    cancel: CancellationToken,
    join: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl RawChatHarness {
    pub async fn start(
        card: ModelDeploymentCard,
        scripts: impl IntoIterator<Item = WorkerScript>,
    ) -> Self {
        let engine = Arc::new(ScriptedBackendEngine::new(scripts));
        let frontend = ServiceFrontend::<
            SingleIn<NvCreateChatCompletionRequest>,
            ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        >::new();
        let preprocessor = OpenAIPreprocessor::new(card.clone())
            .expect("failed to build test preprocessor")
            .into_operator();
        let token_backend = Backend::from_mdc(&card).into_operator();
        let worker = ServiceBackend::from_engine(engine.clone());
        let pipeline = frontend
            .link(preprocessor.forward_edge())
            .and_then(|node| node.link(token_backend.forward_edge()))
            .and_then(|node| node.link(worker))
            .and_then(|node| node.link(token_backend.backward_edge()))
            .and_then(|node| node.link(preprocessor.backward_edge()))
            .and_then(|node| node.link(frontend))
            .expect("failed to link raw chat test pipeline");

        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("failed to build harness HTTP client");
        let (listener, port) = bind_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .host("127.0.0.1")
            .enable_chat_endpoints(true)
            .enable_cmpl_endpoints(false)
            .build()
            .expect("failed to build harness HTTP service");
        service
            .model_manager()
            .add_chat_completions_model(MODEL, card.mdcsum(), pipeline)
            .expect("failed to register raw chat harness model");

        let cancel = CancellationToken::new();
        let join = service.spawn_with_listener(cancel.clone(), listener).await;
        let base_url = format!("http://127.0.0.1:{port}");
        wait_for_health(&client, &base_url).await;

        Self {
            base_url,
            client,
            engine,
            cancel,
            join: Some(join),
        }
    }

    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        let join = self.join.take().expect("harness join handle is missing");
        tokio::time::timeout(Duration::from_secs(2), join)
            .await
            .expect("harness HTTP service did not stop within two seconds")
            .expect("harness HTTP service task panicked")
            .expect("harness HTTP service returned an error");
    }
}

impl Drop for RawChatHarness {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

async fn wait_for_health(client: &reqwest::Client, base_url: &str) {
    let url = format!("{base_url}/health");
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if client
                .get(&url)
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("harness HTTP service did not become healthy");
}

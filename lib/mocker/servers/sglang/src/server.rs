// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang-compatible Mocker gRPC service.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use clap::ValueEnum;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, OutputSignal, WorkerType};
use dynamo_mocker::live::{LiveEngine, LiveRequest, stable_request_uuid};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_sglang_sidecar::proto as pb;
use futures::Stream;
use serde_json::json;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::{Request, Response, Status};

#[path = "server_request.rs"]
mod request;

use request::PreparedRequest;

const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 256;
pub(super) const DP_RANK: u32 = 0;
type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;
pub(super) type BoxedStatusResult<T> = Result<T, Box<Status>>;

/// Wire-level SGLang role exposed by one mock server process.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ServerMode {
    Aggregated,
    Prefill,
    Decode,
}

impl ServerMode {
    fn discovery_value(self) -> &'static str {
        match self {
            Self::Aggregated => "null",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

impl fmt::Display for ServerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Aggregated => "aggregated",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        })
    }
}

/// Discovery metadata and deterministic generation settings for the service.
#[derive(Clone, Debug)]
pub struct MockerServerConfig {
    pub model: String,
    pub mode: ServerMode,
    pub seed: u64,
    pub context_length: u32,
    pub max_concurrent_requests: usize,
    pub bootstrap_host: String,
    pub bootstrap_port: u16,
}

impl Default for MockerServerConfig {
    fn default() -> Self {
        Self {
            model: "mocker-model".to_string(),
            mode: ServerMode::Aggregated,
            seed: 42,
            context_length: 32_768,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            bootstrap_host: "127.0.0.1".to_string(),
            bootstrap_port: 8_998,
        }
    }
}

#[derive(Clone, Debug)]
struct DiscoveryMetadata {
    page_size: usize,
    max_total_num_tokens: usize,
    max_running_requests: usize,
    max_prefill_tokens: usize,
}

/// SGLang-compatible service driven by one shared Mocker scheduler.
#[derive(Clone)]
pub struct SglangMockerService {
    config: Arc<MockerServerConfig>,
    discovery: Arc<DiscoveryMetadata>,
    engine: LiveEngine,
    request_permits: Arc<Semaphore>,
}

impl SglangMockerService {
    pub fn new(config: MockerServerConfig, engine_args: MockEngineArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(!config.model.trim().is_empty(), "model must not be empty");
        anyhow::ensure!(
            config.context_length > 0,
            "context_length must be greater than 0"
        );
        anyhow::ensure!(
            config.context_length <= i32::MAX as u32,
            "context_length must fit SGLang's int32 ModelCard field"
        );
        anyhow::ensure!(
            config.max_concurrent_requests > 0,
            "max_concurrent_requests must be greater than 0"
        );
        if config.mode == ServerMode::Prefill {
            anyhow::ensure!(
                !config.bootstrap_host.trim().is_empty(),
                "prefill bootstrap_host must not be empty"
            );
            anyhow::ensure!(
                config.bootstrap_port != 0,
                "prefill bootstrap_port must not be zero"
            );
        }

        let engine_args = engine_args.normalized()?;
        anyhow::ensure!(
            engine_args.engine_type == EngineType::Sglang,
            "Mocker engine_type must be sglang"
        );
        anyhow::ensure!(engine_args.dp_size == 1, "Mocker dp_size must be 1");
        anyhow::ensure!(
            engine_args.worker_type == WorkerType::Aggregated,
            "Mocker worker_type must be aggregated; use the server mode for the emulated wire role"
        );

        let max_total_num_tokens = engine_args
            .num_gpu_blocks
            .checked_mul(engine_args.block_size)
            .ok_or_else(|| anyhow::anyhow!("num_gpu_blocks * block_size overflows usize"))?;
        let discovery = DiscoveryMetadata {
            page_size: engine_args.block_size,
            max_total_num_tokens,
            max_running_requests: engine_args
                .max_num_seqs
                .unwrap_or(engine_args.num_gpu_blocks),
            max_prefill_tokens: engine_args.max_num_batched_tokens.unwrap_or(8_192),
        };
        let engine = LiveEngine::start(engine_args, DP_RANK)?;
        let max_concurrent_requests = config.max_concurrent_requests;
        Ok(Self {
            config: Arc::new(config),
            discovery: Arc::new(discovery),
            engine,
            request_permits: Arc::new(Semaphore::new(max_concurrent_requests)),
        })
    }

    pub fn config(&self) -> &MockerServerConfig {
        &self.config
    }

    pub fn active_request_count(&self) -> usize {
        self.engine.active_request_count()
    }

    pub fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.engine.metrics_receiver()
    }

    async fn start_generation(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<(PreparedRequest, LiveRequest, OwnedSemaphorePermit), Status> {
        let permit = self
            .request_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("Mocker concurrent request limit reached"))?;
        let prepared = PreparedRequest::new(request, &self.config).map_err(|status| *status)?;
        let live = self
            .engine
            .submit(prepared.direct_request())
            .await
            .map_err(|error| {
                Status::internal(format!("Mocker request submission failed: {error}"))
            })?;
        Ok((prepared, live, permit))
    }

    fn model_info(&self) -> pb::GetModelInfoResponse {
        pb::GetModelInfoResponse {
            model_path: self.config.model.clone(),
            json_info: json!({
                "model_path": self.config.model,
                "tokenizer_path": self.config.model,
            })
            .to_string(),
        }
    }

    fn server_info(&self) -> pb::GetServerInfoResponse {
        pb::GetServerInfoResponse {
            json_info: json!({
                "disaggregation_mode": self.config.mode.discovery_value(),
                "page_size": self.discovery.page_size,
                "max_total_num_tokens": self.discovery.max_total_num_tokens,
                "max_running_requests": self.discovery.max_running_requests,
                "max_prefill_tokens": self.discovery.max_prefill_tokens,
                "dp_size": 1,
                "context_length": self.config.context_length,
                "served_model_name": self.config.model,
                "disaggregation_bootstrap_port": self.config.bootstrap_port,
                "dist_init_addr": format!(
                    "{}:{}",
                    self.config.bootstrap_host, self.config.bootstrap_port
                ),
            })
            .to_string(),
        }
    }
}

#[tonic::async_trait]
impl pb::sglang_service_server::SglangService for SglangMockerService {
    type TextGenerateStream = BoxStream<pb::TextGenerateResponse>;
    type GenerateStream = BoxStream<pb::GenerateResponse>;
    type ChatCompleteStream = BoxStream<pb::OpenAiStreamChunk>;
    type CompleteStream = BoxStream<pb::OpenAiStreamChunk>;

    async fn text_generate(
        &self,
        _request: Request<pb::TextGenerateRequest>,
    ) -> Result<Response<Self::TextGenerateStream>, Status> {
        unsupported("TextGenerate")
    }

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (prepared, mut live, permit) = self.start_generation(request.into_inner()).await?;
        // Keep LiveEngine's fixed delivery queue independent of client and
        // transport pacing. This request-owned buffer cannot exceed the
        // validated output-token budget, and dropping the client stream closes
        // the receiver so the pump drops `live` and cancels unfinished work.
        let (signal_tx, mut signal_rx) = tokio::sync::mpsc::channel(prepared.max_output_tokens);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = signal_tx.closed() => break,
                    signal = live.recv() => {
                        let Some(signal) = signal else { break };
                        let completed = signal.completed;
                        if signal_tx.send(signal).await.is_err() || completed {
                            break;
                        }
                    }
                }
            }
        });
        let stream = async_stream::try_stream! {
            let _permit = permit;
            // The sidecar contract enables SGLang's incremental streaming output,
            // so each response contains only this chunk's token and metadata.
            while let Some(signal) = signal_rx.recv().await {
                let token_id = checked_token(&signal).map_err(|status| *status)?;
                let output_id = i32::try_from(token_id)
                    .map_err(|_| Status::internal("synthetic token ID does not fit i32"))?;
                yield pb::GenerateResponse {
                    output_ids: vec![output_id],
                    meta_info: prepared.meta_info(&[token_id], signal.completed),
                    finished: signal.completed,
                };
                if signal.completed {
                    return;
                }
            }
            Err(Status::internal(
                "Mocker output channel closed before a terminal response",
            ))?;
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn text_embed(
        &self,
        _request: Request<pb::TextEmbedRequest>,
    ) -> Result<Response<pb::TextEmbedResponse>, Status> {
        unsupported("TextEmbed")
    }

    async fn embed(
        &self,
        _request: Request<pb::EmbedRequest>,
    ) -> Result<Response<pb::EmbedResponse>, Status> {
        unsupported("Embed")
    }

    async fn classify(
        &self,
        _request: Request<pb::ClassifyRequest>,
    ) -> Result<Response<pb::ClassifyResponse>, Status> {
        unsupported("Classify")
    }

    async fn tokenize(
        &self,
        _request: Request<pb::TokenizeRequest>,
    ) -> Result<Response<pb::TokenizeResponse>, Status> {
        unsupported("Tokenize")
    }

    async fn detokenize(
        &self,
        _request: Request<pb::DetokenizeRequest>,
    ) -> Result<Response<pb::DetokenizeResponse>, Status> {
        unsupported("Detokenize")
    }

    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        Ok(Response::new(pb::HealthCheckResponse { healthy: true }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::GetModelInfoResponse>, Status> {
        Ok(Response::new(self.model_info()))
    }

    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::GetServerInfoResponse>, Status> {
        Ok(Response::new(self.server_info()))
    }

    async fn list_models(
        &self,
        _request: Request<pb::ListModelsRequest>,
    ) -> Result<Response<pb::ListModelsResponse>, Status> {
        Ok(Response::new(pb::ListModelsResponse {
            models: vec![pb::ModelCard {
                id: self.config.model.clone(),
                root: self.config.model.clone(),
                parent: None,
                max_model_len: Some(self.config.context_length as i32),
            }],
        }))
    }

    async fn get_load(
        &self,
        _request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::GetLoadResponse>, Status> {
        unsupported("GetLoad")
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request = request.into_inner();
        if request.abort_all {
            return Err(Status::unimplemented(
                "Abort with abort_all=true is not supported by the Mocker server",
            ));
        }
        if request.rid.trim().is_empty() {
            return Err(Status::invalid_argument("Abort.rid must not be empty"));
        }
        let request_id = stable_request_uuid(self.config.seed, &request.rid);
        self.engine.cancel(request_id).await.map_err(|error| {
            Status::internal(format!("Mocker request cancellation failed: {error}"))
        })?;
        Ok(Response::new(pb::AbortResponse { success: true }))
    }

    async fn flush_cache(
        &self,
        _request: Request<pb::FlushCacheRequest>,
    ) -> Result<Response<pb::FlushCacheResponse>, Status> {
        unsupported("FlushCache")
    }

    async fn pause_generation(
        &self,
        _request: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::PauseGenerationResponse>, Status> {
        unsupported("PauseGeneration")
    }

    async fn continue_generation(
        &self,
        _request: Request<pb::ContinueGenerationRequest>,
    ) -> Result<Response<pb::ContinueGenerationResponse>, Status> {
        unsupported("ContinueGeneration")
    }

    async fn chat_complete(
        &self,
        _request: Request<pb::OpenAiRequest>,
    ) -> Result<Response<Self::ChatCompleteStream>, Status> {
        unsupported("ChatComplete")
    }

    async fn complete(
        &self,
        _request: Request<pb::OpenAiRequest>,
    ) -> Result<Response<Self::CompleteStream>, Status> {
        unsupported("Complete")
    }

    async fn open_ai_embed(
        &self,
        _request: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        unsupported("OpenAIEmbed")
    }

    async fn open_ai_classify(
        &self,
        _request: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        unsupported("OpenAIClassify")
    }

    async fn score(
        &self,
        _request: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        unsupported("Score")
    }

    async fn rerank(
        &self,
        _request: Request<pb::OpenAiRequest>,
    ) -> Result<Response<pb::OpenAiResponse>, Status> {
        unsupported("Rerank")
    }

    async fn start_profile(
        &self,
        _request: Request<pb::StartProfileRequest>,
    ) -> Result<Response<pb::StartProfileResponse>, Status> {
        unsupported("StartProfile")
    }

    async fn stop_profile(
        &self,
        _request: Request<pb::StopProfileRequest>,
    ) -> Result<Response<pb::StopProfileResponse>, Status> {
        unsupported("StopProfile")
    }

    async fn update_weights_from_disk(
        &self,
        _request: Request<pb::UpdateWeightsRequest>,
    ) -> Result<Response<pb::UpdateWeightsResponse>, Status> {
        unsupported("UpdateWeightsFromDisk")
    }
}

// Tonic requires `Status` at the RPC boundary, so boxing here would only be
// undone immediately by every unsupported trait method.
#[allow(clippy::result_large_err)]
fn unsupported<T>(rpc: &str) -> Result<Response<T>, Status> {
    Err(Status::unimplemented(format!(
        "{rpc} is outside the SGLang sidecar test contract"
    )))
}

fn checked_token(signal: &OutputSignal) -> BoxedStatusResult<u32> {
    if signal.rejected {
        return Err(
            Status::resource_exhausted("request exceeds the simulated KV-cache capacity").into(),
        );
    }
    signal
        .token_id
        .ok_or_else(|| Status::internal("Mocker output signal is missing a token ID"))
        .map_err(Into::into)
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod tests;

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use clap::ValueEnum;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, OutputSignal, WorkerType};
use dynamo_mocker::live::{LiveEngine, LiveRequest, stable_request_uuid};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_vllm_sidecar::proto as pb;
use futures::Stream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::{Request, Response, Status};

use request::{PreparedRequest, SequenceOutputExt};

#[path = "server_request.rs"]
mod request;

const DP_RANK: u32 = 0;
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 256;
type BoxedStatusResult<T> = Result<T, Box<Status>>;

/// Wire-level role exposed by one mock server process.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ServerMode {
    Aggregated,
    Prefill,
    Decode,
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

#[derive(Clone, Debug)]
pub struct MockerServerConfig {
    pub model: String,
    pub mode: ServerMode,
    pub seed: u64,
    pub max_concurrent_requests: usize,
}

impl Default for MockerServerConfig {
    fn default() -> Self {
        Self {
            model: "mocker-model".to_string(),
            mode: ServerMode::Aggregated,
            seed: 42,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }
}

/// Mocker-backed vLLM services.
#[derive(Clone)]
pub struct VllmMockerService {
    config: Arc<MockerServerConfig>,
    model_info: Arc<pb::ModelInfo>,
    server_info: Arc<pb::ServerInfo>,
    engine: LiveEngine,
    request_permits: Arc<Semaphore>,
}

impl VllmMockerService {
    pub fn new(config: MockerServerConfig, engine_args: MockEngineArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(
            engine_args.engine_type == EngineType::Vllm,
            "Mocker engine_type must be vllm"
        );
        anyhow::ensure!(engine_args.dp_size == 1, "Mocker dp_size must be 1");
        anyhow::ensure!(
            config.max_concurrent_requests > 0,
            "max_concurrent_requests must be greater than 0"
        );
        anyhow::ensure!(
            engine_args.worker_type == WorkerType::Aggregated,
            "Mocker worker_type must be aggregated; use the server mode for the emulated wire role"
        );
        let max_concurrent_requests = config.max_concurrent_requests;
        let model_info = pb::ModelInfo {
            model_id: config.model.clone(),
            served_model_name: config.model.clone(),
            served_model_aliases: Vec::new(),
            supports_text_input: false,
            supports_token_ids_input: true,
            supports_multimodal: false,
            reasoning_parser: String::new(),
            tool_call_parser: String::new(),
        };
        let server_info = pb::ServerInfo {
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            api_version: "vllm".to_string(),
            instance_id: format!("dynamo-vllm-mocker-{}", config.mode),
            parallelism: Some(pb::ParallelismInfo {
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                data_parallel_size: engine_args.dp_size,
                data_parallel_rank: DP_RANK,
                decode_context_parallel_size: 1,
            }),
            max_model_len: engine_args
                .max_model_len
                .map(u32::try_from)
                .transpose()
                .map_err(|_| anyhow::anyhow!("max_model_len exceeds the Control API range"))?
                .unwrap_or_default(),
            kv_block_size: u32::try_from(engine_args.block_size)
                .map_err(|_| anyhow::anyhow!("block_size exceeds the Control API range"))?,
            total_kv_blocks: u64::try_from(engine_args.num_gpu_blocks)
                .map_err(|_| anyhow::anyhow!("num_gpu_blocks exceeds the Control API range"))?,
            max_running_requests: engine_args
                .max_num_seqs
                .map(u64::try_from)
                .transpose()
                .map_err(|_| anyhow::anyhow!("max_num_seqs exceeds the Control API range"))?
                .unwrap_or_default(),
            max_batched_tokens: engine_args
                .max_num_batched_tokens
                .map(u64::try_from)
                .transpose()
                .map_err(|_| {
                    anyhow::anyhow!("max_num_batched_tokens exceeds the Control API range")
                })?
                .unwrap_or_default(),
            rl_capabilities: None,
        };
        Ok(Self {
            config: Arc::new(config),
            model_info: Arc::new(model_info),
            server_info: Arc::new(server_info),
            engine: LiveEngine::start(engine_args, DP_RANK)?,
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
        request: Request<pb::GenerateRequest>,
    ) -> Result<(PreparedRequest, LiveRequest, OwnedSemaphorePermit), Status> {
        let data_parallel_rank = request
            .metadata()
            .get("x-data-parallel-rank")
            .map(|value| {
                value
                    .to_str()
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .ok_or_else(|| {
                        Box::new(Status::invalid_argument(
                            "x-data-parallel-rank metadata must be an unsigned 32-bit integer",
                        ))
                    })
            })
            .transpose()
            .map_err(|status| *status)?;
        if let Some(rank) = data_parallel_rank
            && rank != DP_RANK
        {
            return Err(Status::invalid_argument(format!(
                "data_parallel_rank {rank} is not served; expected {DP_RANK}"
            )));
        }
        let permit = self
            .request_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("Mocker concurrent request limit reached"))?;
        let prepared =
            PreparedRequest::new(request.into_inner(), &self.config).map_err(|status| *status)?;
        let live = self
            .engine
            .submit(prepared.direct_request())
            .await
            .map_err(|error| {
                Status::internal(format!("Mocker request submission failed: {error}"))
            })?;
        Ok((prepared, live, permit))
    }
}

#[tonic::async_trait]
impl pb::inference_server::Inference for VllmMockerService {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send + 'static>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let (prepared, mut live, _permit) = self.start_generation(request).await?;
        let mut output_ids = Vec::with_capacity(prepared.max_output_tokens);
        while let Some(signal) = live.recv().await {
            let token_id = checked_token(&signal).map_err(|status| *status)?;
            output_ids.push(token_id);
            if signal.completed {
                return Ok(Response::new(pb::GenerateResponse {
                    prompt_info: Some(prepared.prompt_info()),
                    outputs: Some(prepared.sequence_output(&output_ids, true)),
                }));
            }
        }
        Err(Status::internal(
            "Mocker output channel closed before a terminal response",
        ))
    }

    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let (prepared, mut live, permit) = self.start_generation(request).await?;
        // Decouple LiveEngine's small fixed per-request buffer from client and
        // transport pacing. A pump drains the engine promptly into a buffer
        // bounded by this request's own token budget, so a bursty producer
        // racing ahead of a slow gRPC consumer no longer trips LiveEngine's
        // slow-consumer shedding and surfaces as a spurious INTERNAL. The buffer
        // cannot grow past the request's declared output length, and dropping
        // the client stream still cancels unfinished scheduler work.
        let (signal_tx, mut signal_rx) =
            tokio::sync::mpsc::channel(prepared.max_output_tokens.saturating_add(1));
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    // The client dropped the stream: stop and let `live` drop,
                    // which cancels any unfinished scheduler work promptly.
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
            yield pb::GenerateResponse {
                prompt_info: Some(prepared.prompt_info()),
                outputs: None,
            };

            let mut generated = 0usize;
            while let Some(signal) = signal_rx.recv().await {
                let token_id = checked_token(&signal).map_err(|status| *status)?;
                generated += 1;
                yield pb::GenerateResponse {
                    prompt_info: None,
                    outputs: Some(prepared.sequence_output(&[token_id], signal.completed)
                        .with_total_output_tokens(generated)),
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
}

#[tonic::async_trait]
impl pb::control_server::Control for VllmMockerService {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Ok(Response::new((*self.server_info).clone()))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        Ok(Response::new((*self.model_info).clone()))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        for request_id in request.into_inner().request_ids {
            self.engine
                .cancel(stable_request_uuid(self.config.seed, &request_id))
                .await
                .map_err(|error| Status::internal(format!("Mocker abort failed: {error}")))?;
        }
        Ok(Response::new(pb::AbortResponse {}))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        Ok(Response::new(pb::GetKvEventSourcesResponse {
            sources: Vec::new(),
        }))
    }

    async fn pause_generation(
        &self,
        _request: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::PauseGenerationResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn resume_generation(
        &self,
        _request: Request<pb::ResumeGenerationRequest>,
    ) -> Result<Response<pb::ResumeGenerationResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn is_paused(
        &self,
        _request: Request<pb::IsPausedRequest>,
    ) -> Result<Response<pb::IsPausedResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn sleep(
        &self,
        _request: Request<pb::SleepRequest>,
    ) -> Result<Response<pb::SleepResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn wake_up(
        &self,
        _request: Request<pb::WakeUpRequest>,
    ) -> Result<Response<pb::WakeUpResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn is_sleeping(
        &self,
        _request: Request<pb::IsSleepingRequest>,
    ) -> Result<Response<pb::IsSleepingResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn init_weight_transfer_engine(
        &self,
        _request: Request<pb::InitWeightTransferEngineRequest>,
    ) -> Result<Response<pb::InitWeightTransferEngineResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn start_weight_update(
        &self,
        _request: Request<pb::StartWeightUpdateRequest>,
    ) -> Result<Response<pb::StartWeightUpdateResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn start_draft_weight_update(
        &self,
        _request: Request<pb::StartDraftWeightUpdateRequest>,
    ) -> Result<Response<pb::StartDraftWeightUpdateResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn update_weights(
        &self,
        _request: Request<pb::UpdateWeightsRequest>,
    ) -> Result<Response<pb::UpdateWeightsResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn finish_weight_update(
        &self,
        _request: Request<pb::FinishWeightUpdateRequest>,
    ) -> Result<Response<pb::FinishWeightUpdateResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn update_weight_version(
        &self,
        _request: Request<pb::UpdateWeightVersionRequest>,
    ) -> Result<Response<pb::UpdateWeightVersionResponse>, Status> {
        Err(rl_control_unavailable())
    }

    async fn get_weight_version(
        &self,
        _request: Request<pb::GetWeightVersionRequest>,
    ) -> Result<Response<pb::GetWeightVersionResponse>, Status> {
        Err(rl_control_unavailable())
    }
}

fn rl_control_unavailable() -> Status {
    Status::unimplemented("the vLLM mocker does not implement RL control RPCs")
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

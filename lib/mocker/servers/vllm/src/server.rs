// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use clap::ValueEnum;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, OutputSignal, WorkerType};
use dynamo_mocker::live::{LiveEngine, LiveRequest};
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

/// vLLM-compatible Generate service driven by one shared Mocker scheduler.
#[derive(Clone)]
pub struct VllmMockerService {
    config: Arc<MockerServerConfig>,
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
        Ok(Self {
            config: Arc::new(config),
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
}

#[tonic::async_trait]
impl pb::generate_server::Generate for VllmMockerService {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send + 'static>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let (prepared, mut live, _permit) = self.start_generation(request.into_inner()).await?;
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
        let (prepared, mut live, permit) = self.start_generation(request.into_inner()).await?;
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

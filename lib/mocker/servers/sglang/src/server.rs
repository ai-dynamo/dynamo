// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang-compatible Mocker gRPC service: `sglang.api.v1.SglangService`,
//! shaped like SGLang's Rust server, with typed discovery, `GenerateStreamItem`
//! frames, and in-stream error items carrying HTTP status codes.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use clap::ValueEnum;
use dynamo_mocker::common::protocols::{
    EngineType, KvEventPublishers, MockerConfig, OutputSignal, WorkerType,
};
use dynamo_mocker::live::{LiveEngine, LiveEngineConfig, stable_request_uuid};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use dynamo_sglang_sidecar::proto as pb;
use futures::Stream;
use tokio::sync::Semaphore;
use tonic::{Request, Response, Status};

#[path = "server_request.rs"]
mod request;

use request::PreparedRequest;

const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 256;
/// The Mocker's one scheduler rank, whatever DP rank discovery advertises.
pub(super) const DP_RANK: u32 = 0;
/// SGLang's code for a request cancelled by Abort or a disconnect.
const CLIENT_CLOSED_REQUEST: u32 = 499;
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
    /// This listener's DP rank, as discovery and its KV events report it.
    pub dp_rank: u32,
    /// Every DP rank of the deployment, counting attention-DP ranks.
    pub dp_size: u32,
    /// Attention DP: report the launch-wide running-request limit.
    pub enable_dp_attention: bool,
    /// This listener's offset among its node's listeners.
    pub port_offset: u32,
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
            dp_rank: 0,
            dp_size: 1,
            enable_dp_attention: false,
            port_offset: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct DiscoveryMetadata {
    page_size: u32,
    /// The bound KV-event publisher port.
    kv_event_port: Option<u16>,
    max_total_num_tokens: usize,
    max_running_requests: usize,
    max_prefill_tokens: usize,
}

/// SGLang-compatible service driven by one Mocker scheduler: one Rust-server
/// listener.
#[derive(Clone)]
pub struct SglangMockerService {
    config: Arc<MockerServerConfig>,
    discovery: Arc<DiscoveryMetadata>,
    engine: LiveEngine,
    request_permits: Arc<Semaphore>,
}

impl SglangMockerService {
    pub fn new(config: MockerServerConfig, engine_args: MockerConfig) -> anyhow::Result<Self> {
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
        anyhow::ensure!(config.dp_size > 0, "dp_size must be greater than 0");
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
            engine_args.backend == EngineType::Sglang,
            "Mocker engine_type must be sglang"
        );
        anyhow::ensure!(engine_args.dp_size == 1, "Mocker dp_size must be 1");
        anyhow::ensure!(
            engine_args.worker_type == WorkerType::Aggregated,
            "Mocker worker_type must be aggregated; use the server mode for the emulated wire role"
        );

        let page_size = u32::try_from(engine_args.block_size).map_err(|_| {
            anyhow::anyhow!("block_size exceeds the SGLang page_size discovery range")
        })?;
        let max_total_num_tokens = engine_args
            .num_gpu_blocks
            .checked_mul(engine_args.block_size)
            .ok_or_else(|| anyhow::anyhow!("num_gpu_blocks * block_size overflows usize"))?;
        let sink = if engine_args.needs_kv_publisher() && config.mode != ServerMode::Decode {
            match ZmqKvEventSink::bind(
                engine_args.runtime.zmq_kv_events_port,
                engine_args.runtime.zmq_replay_port,
                config.dp_rank,
                page_size,
            ) {
                Ok(sink) => Some(sink),
                Err(error) => {
                    tracing::error!(dp_rank = config.dp_rank, %error, "Failed to create ZMQ KV event sink");
                    None
                }
            }
        } else {
            None
        };
        let kv_event_port = sink.as_ref().map(|sink| {
            sink.endpoint()
                .strip_prefix("tcp://")
                .expect("ZMQ sink uses TCP")
                .parse::<std::net::SocketAddr>()
                .expect("ZMQ sink reports a bound socket address")
                .port()
        });
        let discovery = DiscoveryMetadata {
            page_size,
            kv_event_port,
            max_total_num_tokens,
            max_running_requests: engine_args.effective_handoff_capacity(),
            max_prefill_tokens: if engine_args.max_num_batched_tokens == usize::MAX {
                engine_args.sglang.max_prefill_tokens
            } else {
                engine_args.max_num_batched_tokens
            },
        };
        let engine = LiveEngine::start_with_config(
            engine_args,
            DP_RANK,
            LiveEngineConfig {
                kv_event_publishers: KvEventPublishers::new(
                    None,
                    sink.map(|sink| Arc::new(sink) as _),
                ),
                ..Default::default()
            },
        )?;
        let max_concurrent_requests = config.max_concurrent_requests;
        Ok(Self {
            config: Arc::new(config),
            discovery: Arc::new(discovery),
            engine,
            request_permits: Arc::new(Semaphore::new(max_concurrent_requests)),
        })
    }

    /// The `listeners` listeners of one node: listener `k` serves DP rank
    /// `config.dp_rank + k`, bootstrap port `config.bootstrap_port + k` and
    /// the configured ZMQ ports + `k`, with its own scheduler.
    pub fn node(
        config: MockerServerConfig,
        engine_args: MockerConfig,
        listeners: u32,
    ) -> anyhow::Result<Vec<Self>> {
        (0..listeners)
            .map(|offset| {
                let port = |port: u16| {
                    u16::try_from(u32::from(port) + offset)
                        .map_err(|_| anyhow::anyhow!("listener {offset}'s port exceeds 65535"))
                };
                let mut engine_args = engine_args.clone();
                for zmq_port in [
                    &mut engine_args.runtime.zmq_kv_events_port,
                    &mut engine_args.runtime.zmq_replay_port,
                ] {
                    *zmq_port = zmq_port.map(port).transpose()?;
                }
                Self::new(
                    MockerServerConfig {
                        dp_rank: config.dp_rank + offset,
                        bootstrap_port: port(config.bootstrap_port)?,
                        port_offset: offset,
                        ..config.clone()
                    },
                    engine_args,
                )
            })
            .collect()
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

    fn model_info(&self) -> pb::GetModelInfoResponse {
        let model = &self.config.model;
        pb::GetModelInfoResponse {
            model_path: model.clone(),
            served_model_name: model.clone(),
            tokenizer_path: model.clone(),
            is_generation: true,
            ..Default::default()
        }
    }

    fn server_info(&self) -> pb::GetServerInfoResponse {
        let config = &self.config;
        let discovery = &self.discovery;
        let to_u64 = |value: usize| u64::try_from(value).unwrap_or(u64::MAX);
        // Like SGLang, the publisher binds every interface and is reported at
        // a dialable node-local address; the sidecar runs on the same node.
        let kv_event_sources = discovery
            .kv_event_port
            .map(|port| pb::KvEventSource {
                dp_rank: config.dp_rank,
                endpoint: format!("tcp://127.0.0.1:{port}"),
                topic: String::new(),
                block_size: discovery.page_size,
            })
            .into_iter()
            .collect();
        pb::GetServerInfoResponse {
            model_path: config.model.clone(),
            served_model_name: config.model.clone(),
            tokenizer_path: config.model.clone(),
            max_context_length: config.context_length,
            max_total_num_tokens: Some(to_u64(discovery.max_total_num_tokens)),
            num_reserved_tokens: 0,
            version: env!("CARGO_PKG_VERSION").to_string(),
            internal_states: Vec::new(),
            host: config.bootstrap_host.clone(),
            // The Mocker serves no native HTTP API.
            port: 0,
            incremental_streaming_output: true,
            disaggregation_mode: config.mode.discovery_value().to_string(),
            allow_auto_truncate: false,
            skip_tokenizer_init: false,
            deployment_info: Some(pb::DeploymentInfo {
                disaggregation_bootstrap_port: Some(u32::from(config.bootstrap_port)),
                node_rank: 0,
                nnodes: 1,
                dp_size: config.dp_size,
                dp_rank: Some(config.dp_rank),
                // The node's rendezvous: the first listener's bootstrap address.
                dist_init_addr: Some(format!(
                    "{}:{}",
                    config.bootstrap_host,
                    u32::from(config.bootstrap_port) - config.port_offset
                )),
                page_size: Some(discovery.page_size),
                dcp_size: 1,
                // SGLang reports the launch-wide limit and splits it across
                // attention-DP ranks.
                max_running_requests: Some(to_u64(discovery.max_running_requests).saturating_mul(
                    if config.enable_dp_attention {
                        u64::from(config.dp_size)
                    } else {
                        1
                    },
                )),
                max_prefill_tokens: to_u64(discovery.max_prefill_tokens),
                enable_dp_attention: config.enable_dp_attention,
                kv_event_sources,
                ..Default::default()
            }),
        }
    }
}

#[tonic::async_trait]
impl pb::sglang_service_server::SglangService for SglangMockerService {
    type GenerateStream = BoxStream<pb::GenerateStreamItem>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let permit = self
            .request_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("Mocker concurrent request limit reached"))?;
        let prepared =
            PreparedRequest::new(request.into_inner(), &self.config).map_err(|status| *status)?;
        if let Some(message) = prepared.context_error.clone() {
            // SGLang checks the context after admission, inside the stream.
            let item = error_item(400, message);
            return Ok(Response::new(Box::pin(futures::stream::once(async move {
                let _permit = permit;
                Ok(item)
            }))));
        }
        let direct = prepared.direct_request();
        let mut live = if prepared.has_decode_handoff {
            self.engine.submit_decode(direct).await
        } else {
            self.engine.submit(direct).await
        }
        .map_err(|error| Status::internal(format!("Mocker request submission failed: {error}")))?;
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
                        let Some(signal) = signal else {
                            if live.is_aborted() {
                                let _ = signal_tx.send(Err(error_item(
                                    CLIENT_CLOSED_REQUEST,
                                    "client disconnected".to_string(),
                                ))).await;
                            }
                            break;
                        };
                        let completed = signal.completed;
                        if signal_tx.send(Ok(signal)).await.is_err() || completed {
                            break;
                        }
                    }
                }
            }
        });
        let stream = async_stream::stream! {
            let _permit = permit;
            let mut completion_tokens = 0_usize;
            // The sidecar contract enables SGLang's incremental streaming output,
            // so each frame contains only this chunk's token and metadata.
            while let Some(signal) = signal_rx.recv().await {
                let signal = match signal {
                    Ok(signal) => signal,
                    Err(item) => {
                        yield Ok(item);
                        return;
                    }
                };
                let token_id = match checked_token(&signal) {
                    Ok(token_id) => token_id,
                    Err((code, message)) => {
                        yield Ok(error_item(code, message.to_string()));
                        return;
                    }
                };
                completion_tokens = completion_tokens.saturating_add(1);
                yield Ok(pb::GenerateStreamItem {
                    item: Some(pb::generate_stream_item::Item::Frame(prepared.frame(
                        &[token_id],
                        completion_tokens,
                        completion_tokens == 1,
                        signal.completed,
                    ))),
                });
                if signal.completed {
                    return;
                }
            }
            yield Ok(error_item(
                500,
                "Mocker output channel closed before a terminal response".to_string(),
            ));
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        // The Mocker is ready once constructed. It does not simulate SGLang's
        // one-token probe, which would perturb scheduler-state assertions.
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
                max_model_len: u64::from(self.config.context_length),
            }],
        }))
    }

    async fn detokenize(
        &self,
        _request: Request<pb::DetokenizeRequest>,
    ) -> Result<Response<pb::DetokenizeResponse>, Status> {
        // SGLang answers the same way when it runs without a tokenizer.
        Err(Status::unimplemented("detokenization requires a tokenizer"))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request = request.into_inner();
        if request.abort_all != request.rid.is_empty() {
            return Err(Status::invalid_argument(
                "provide either a nonempty rid or abort_all=true, not both",
            ));
        }
        if request.abort_all {
            return Err(Status::unimplemented(
                "Abort with abort_all=true is not supported by the Mocker server",
            ));
        }
        let request_id = stable_request_uuid(self.config.seed, &request.rid);
        let success = self.engine.cancel(request_id).await.map_err(|error| {
            Status::internal(format!("Mocker request cancellation failed: {error}"))
        })?;
        Ok(Response::new(pb::AbortResponse { success }))
    }

    // The Mocker has no hierarchical cache storage backend to manage.
    async fn get_hi_cache_storage(
        &self,
        _request: Request<pb::GetHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStatus>, Status> {
        Err(hicache_unimplemented())
    }

    async fn attach_hi_cache_storage(
        &self,
        _request: Request<pb::AttachHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStorageResponse>, Status> {
        Err(hicache_unimplemented())
    }

    async fn detach_hi_cache_storage(
        &self,
        _request: Request<pb::DetachHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStorageResponse>, Status> {
        Err(hicache_unimplemented())
    }

    async fn clear_hi_cache_storage(
        &self,
        _request: Request<pb::ClearHiCacheStorageRequest>,
    ) -> Result<Response<pb::HiCacheStorageResponse>, Status> {
        Err(hicache_unimplemented())
    }
}

fn hicache_unimplemented() -> Status {
    Status::unimplemented("HiCache storage management is not supported by the Mocker server")
}

/// One request's failure inside an open Generate stream; `code` is the HTTP
/// status SGLang's native error body carries.
fn error_item(code: u32, message: String) -> pb::GenerateStreamItem {
    pb::GenerateStreamItem {
        item: Some(pb::generate_stream_item::Item::Error(
            pb::GenerateStreamError {
                error: Some(pb::ErrorBody { message, code }),
                index: None,
            },
        )),
    }
}

/// The signal's token, or the code and message of the error item that ends
/// the stream instead.
fn checked_token(signal: &OutputSignal) -> Result<u32, (u32, &'static str)> {
    if signal.rejected {
        // SGLang reports scheduler overload as 503.
        return Err((503, "request exceeds the simulated KV-cache capacity"));
    }
    signal
        .token_id
        .ok_or((500, "Mocker output signal is missing a token ID"))
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod tests;

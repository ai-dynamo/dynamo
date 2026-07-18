// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang-compatible Mocker gRPC service.

use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use clap::ValueEnum;
use dynamo_mocker::common::protocols::{
    DirectRequest, EngineType, MockEngineArgs, OutputSignal, WorkerType,
};
use dynamo_mocker::live::{
    LiveEngine, LiveRequest, deterministic_output_tokens, stable_request_uuid,
};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_sglang_grpc as pb;
use futures::Stream;
use serde_json::{Value, json};
use tonic::{Request, Response, Status};
use uuid::Uuid;

const DP_RANK: u32 = 0;
const DEFAULT_MAX_NEW_TOKENS: i32 = 20;
const MAX_NEW_TOKENS: i32 = 1_000_000;
const MAX_TOP_LOGPROBS: usize = 20;

type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

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
    last_disaggregated_params: Arc<Mutex<Option<pb::DisaggregatedParams>>>,
}

impl SglangMockerService {
    pub fn new(config: MockerServerConfig, engine_args: MockEngineArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(!config.model.trim().is_empty(), "model must not be empty");
        anyhow::ensure!(
            config.context_length <= i32::MAX as u32,
            "context_length must fit SGLang's int32 ModelCard field"
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
        Ok(Self {
            config: Arc::new(config),
            discovery: Arc::new(discovery),
            engine,
            last_disaggregated_params: Arc::new(Mutex::new(None)),
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

    /// Most recent rendezvous metadata accepted by `Generate`.
    pub fn last_disaggregated_params(&self) -> Option<pb::DisaggregatedParams> {
        self.last_disaggregated_params
            .lock()
            .expect("last disaggregated parameters mutex poisoned")
            .clone()
    }

    async fn start_generation(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<(PreparedRequest, LiveRequest), Status> {
        let disaggregated_params = request.disaggregated_params.clone();
        let prepared = PreparedRequest::new(request, &self.config)?;
        let live = self
            .engine
            .submit(prepared.direct_request())
            .await
            .map_err(|error| {
                if error.to_string().contains("already active") {
                    Status::already_exists(error.to_string())
                } else {
                    Status::internal(format!("Mocker request submission failed: {error}"))
                }
            })?;
        *self
            .last_disaggregated_params
            .lock()
            .expect("last disaggregated parameters mutex poisoned") = disaggregated_params;
        Ok((prepared, live))
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
        let (prepared, mut live) = self.start_generation(request.into_inner()).await?;
        let stream = async_stream::try_stream! {
            let mut output_tokens = Vec::with_capacity(prepared.max_output_tokens);
            while let Some(signal) = live.recv().await {
                let token_id = checked_token(&signal)?;
                output_tokens.push(token_id);
                let output_id = i32::try_from(token_id)
                    .map_err(|_| Status::internal("synthetic token ID does not fit i32"))?;
                yield pb::GenerateResponse {
                    output_ids: vec![output_id],
                    meta_info: prepared.meta_info(&output_tokens, signal.completed),
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

fn unsupported<T>(rpc: &str) -> Result<Response<T>, Status> {
    Err(Status::unimplemented(format!(
        "{rpc} is outside the SGLang sidecar test contract"
    )))
}

fn checked_token(signal: &OutputSignal) -> Result<u32, Status> {
    if signal.rejected {
        return Err(Status::resource_exhausted(
            "request exceeds the simulated KV-cache capacity",
        ));
    }
    signal
        .token_id
        .ok_or_else(|| Status::internal("Mocker output signal is missing a token ID"))
}

#[derive(Debug)]
struct PreparedRequest {
    uuid: Uuid,
    request_id: String,
    prompt_tokens: Vec<u32>,
    max_output_tokens: usize,
    output_token_ids: Vec<u32>,
    return_logprob: bool,
    top_logprobs_num: usize,
    logprob_start_len: i32,
}

impl PreparedRequest {
    fn new(request: pb::GenerateRequest, config: &MockerServerConfig) -> Result<Self, Status> {
        if request.input_ids.is_empty() {
            return Err(Status::invalid_argument("input_ids must not be empty"));
        }
        let prompt_tokens = request
            .input_ids
            .iter()
            .map(|token| {
                u32::try_from(*token).map_err(|_| {
                    Status::invalid_argument(format!(
                        "input_ids contains a negative token ID: {token}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(n) = request.sampling_params.as_ref().and_then(|params| params.n)
            && n != 1
        {
            return Err(Status::invalid_argument("sampling_params.n must be 1"));
        }

        let requested_max = request
            .sampling_params
            .as_ref()
            .and_then(|params| params.max_new_tokens)
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
        if requested_max <= 0 || requested_max > MAX_NEW_TOKENS {
            return Err(Status::invalid_argument(format!(
                "max_new_tokens must be between 1 and {MAX_NEW_TOKENS}"
            )));
        }
        let max_output_tokens = if config.mode == ServerMode::Prefill {
            1
        } else {
            requested_max as usize
        };

        validate_role(config, request.disaggregated_params.as_ref())?;

        let top_logprobs_num = request.top_logprobs_num.unwrap_or(0);
        if top_logprobs_num < 0 {
            return Err(Status::invalid_argument(
                "top_logprobs_num must not be negative",
            ));
        }
        let logprob_start_len = request.logprob_start_len.unwrap_or(-1);
        if logprob_start_len < -1 {
            return Err(Status::invalid_argument(
                "logprob_start_len must be -1 or greater",
            ));
        }

        let request_id = request
            .rid
            .filter(|request_id| !request_id.trim().is_empty())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let uuid = stable_request_uuid(config.seed, &request_id);
        let output_token_ids =
            deterministic_output_tokens(config.seed, &request_id, max_output_tokens);
        Ok(Self {
            uuid,
            request_id,
            prompt_tokens,
            max_output_tokens,
            output_token_ids,
            return_logprob: request.return_logprob.unwrap_or(false),
            top_logprobs_num: (top_logprobs_num as usize).min(MAX_TOP_LOGPROBS),
            logprob_start_len,
        })
    }

    fn direct_request(&self) -> DirectRequest {
        DirectRequest {
            tokens: self.prompt_tokens.clone(),
            max_output_tokens: self.max_output_tokens,
            output_token_ids: Some(self.output_token_ids.clone()),
            uuid: Some(self.uuid),
            dp_rank: DP_RANK,
            ..Default::default()
        }
    }

    fn meta_info(&self, output_tokens: &[u32], terminal: bool) -> HashMap<String, String> {
        let mut meta = HashMap::from([
            (
                "prompt_tokens".to_string(),
                Value::from(self.prompt_tokens.len()).to_string(),
            ),
            (
                "mocker_request_id".to_string(),
                Value::String(self.request_id.clone()).to_string(),
            ),
        ]);
        if self.return_logprob {
            insert_json(
                &mut meta,
                "output_token_logprobs",
                Value::Array(
                    output_tokens
                        .iter()
                        .map(|token| logprob_entry(*token))
                        .collect(),
                ),
            );
            insert_json(
                &mut meta,
                "output_top_logprobs",
                Value::Array(
                    output_tokens
                        .iter()
                        .map(|token| top_logprob_entries(*token, self.top_logprobs_num))
                        .collect(),
                ),
            );
        }
        if terminal {
            insert_json(&mut meta, "finish_reason", json!({"type": "length"}));
            if self.return_logprob && self.logprob_start_len >= 0 {
                let start = (self.logprob_start_len as usize)
                    .saturating_add(1)
                    .min(self.prompt_tokens.len());
                insert_json(
                    &mut meta,
                    "input_token_logprobs",
                    Value::Array(
                        self.prompt_tokens[start..]
                            .iter()
                            .map(|token| logprob_entry(*token))
                            .collect(),
                    ),
                );
                insert_json(
                    &mut meta,
                    "input_top_logprobs",
                    Value::Array(
                        self.prompt_tokens[start..]
                            .iter()
                            .map(|token| top_logprob_entries(*token, self.top_logprobs_num))
                            .collect(),
                    ),
                );
            }
        }
        meta
    }
}

fn validate_role(
    config: &MockerServerConfig,
    params: Option<&pb::DisaggregatedParams>,
) -> Result<(), Status> {
    match (config.mode, params) {
        (ServerMode::Aggregated, None) => Ok(()),
        (ServerMode::Aggregated, Some(_)) => Err(Status::failed_precondition(
            "aggregated mock server received disaggregated parameters",
        )),
        (ServerMode::Prefill | ServerMode::Decode, None) => Err(Status::failed_precondition(
            "disaggregated mock server requires bootstrap_host, bootstrap_port, and bootstrap_room",
        )),
        (ServerMode::Prefill | ServerMode::Decode, Some(params)) => {
            if params.bootstrap_host.trim().is_empty()
                || params.bootstrap_port <= 0
                || params.bootstrap_room < 0
            {
                return Err(Status::invalid_argument(
                    "disaggregated parameters must contain a host, positive port, and non-negative room",
                ));
            }
            if config.mode == ServerMode::Prefill
                && i32::from(config.bootstrap_port) != params.bootstrap_port
            {
                return Err(Status::failed_precondition(format!(
                    "prefill bootstrap_port {} does not match discovered port {}",
                    params.bootstrap_port, config.bootstrap_port
                )));
            }
            Ok(())
        }
    }
}

fn selected_logprob(token_id: u32) -> f64 {
    -0.1 * f64::from((token_id % 10) + 1)
}

fn logprob_entry(token_id: u32) -> Value {
    json!([
        selected_logprob(token_id),
        token_id,
        format!("<token:{token_id}>")
    ])
}

fn top_logprob_entries(token_id: u32, count: usize) -> Value {
    Value::Array(
        (0..count)
            .map(|offset| {
                let candidate = token_id.saturating_add(offset as u32);
                json!([
                    selected_logprob(candidate) - (offset as f64 * 0.01),
                    candidate,
                    format!("<token:{candidate}>")
                ])
            })
            .collect(),
    )
}

fn insert_json(meta: &mut HashMap<String, String>, key: &str, value: Value) {
    meta.insert(key.to_string(), value.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_sglang_grpc::sglang_service_server::SglangService;

    fn engine_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(128)
            .max_num_seqs(Some(8))
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn request(request_id: &str) -> pb::GenerateRequest {
        pb::GenerateRequest {
            input_ids: vec![1, 2, 3],
            sampling_params: Some(pb::SamplingParams {
                max_new_tokens: Some(2),
                n: Some(1),
                ..Default::default()
            }),
            stream: Some(true),
            return_logprob: Some(true),
            top_logprobs_num: Some(2),
            logprob_start_len: Some(0),
            rid: Some(request_id.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn request_plan_is_stable_and_meta_is_sglang_shaped() {
        let config = MockerServerConfig::default();
        let first = PreparedRequest::new(request("stable"), &config).unwrap();
        let second = PreparedRequest::new(request("stable"), &config).unwrap();
        assert_eq!(first.uuid, second.uuid);
        assert_eq!(first.output_token_ids, second.output_token_ids);

        let meta = first.meta_info(&first.output_token_ids, true);
        assert_eq!(meta["prompt_tokens"], "3");
        assert_eq!(
            serde_json::from_str::<Value>(&meta["finish_reason"]).unwrap(),
            json!({"type": "length"})
        );
        assert_eq!(
            serde_json::from_str::<Value>(&meta["output_token_logprobs"])
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            serde_json::from_str::<Value>(&meta["input_token_logprobs"])
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn request_validation_rejects_bad_ids_limits_and_roles() {
        let config = MockerServerConfig::default();
        let mut negative = request("negative");
        negative.input_ids = vec![-1];
        assert_eq!(
            PreparedRequest::new(negative, &config).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );

        let mut bad_n = request("bad-n");
        bad_n.sampling_params.as_mut().unwrap().n = Some(2);
        assert_eq!(
            PreparedRequest::new(bad_n, &config).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );

        let prefill_config = MockerServerConfig {
            mode: ServerMode::Prefill,
            ..Default::default()
        };
        assert_eq!(
            PreparedRequest::new(request("missing-handoff"), &prefill_config)
                .unwrap_err()
                .code(),
            tonic::Code::FailedPrecondition
        );
    }

    #[tokio::test]
    async fn discovery_matches_engine_capacity_and_mode() {
        let config = MockerServerConfig {
            mode: ServerMode::Prefill,
            ..Default::default()
        };
        let service = SglangMockerService::new(config, engine_args()).unwrap();
        let server = service
            .get_server_info(Request::new(pb::GetServerInfoRequest {}))
            .await
            .unwrap()
            .into_inner();
        let info: Value = serde_json::from_str(&server.json_info).unwrap();
        assert_eq!(info["disaggregation_mode"], "prefill");
        assert_eq!(info["page_size"], 4);
        assert_eq!(info["max_total_num_tokens"], 512);
        assert_eq!(info["max_running_requests"], 8);
        assert_eq!(info["max_prefill_tokens"], 64);
    }

    #[tokio::test]
    async fn abort_is_targeted_idempotent_and_abort_all_is_unsupported() {
        let service =
            SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
        for _ in 0..2 {
            let response = service
                .abort(Request::new(pb::AbortRequest {
                    rid: "missing-or-finished".to_string(),
                    abort_all: false,
                }))
                .await
                .unwrap()
                .into_inner();
            assert!(response.success);
        }
        let error = service
            .abort(Request::new(pb::AbortRequest {
                rid: String::new(),
                abort_all: true,
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Unimplemented);
    }

    #[tokio::test]
    async fn unrelated_rpc_is_unimplemented() {
        let service =
            SglangMockerService::new(MockerServerConfig::default(), engine_args()).unwrap();
        let error = service
            .tokenize(Request::new(pb::TokenizeRequest::default()))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Unimplemented);
    }
}

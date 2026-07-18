// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use clap::ValueEnum;
use dynamo_mocker::common::protocols::{
    DirectRequest, EngineType, MockEngineArgs, OutputSignal, WorkerType,
};
use dynamo_mocker::live::{LiveEngine, LiveRequest};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_vllm_grpc as pb;
use futures::Stream;
use prost_types::{ListValue, Struct, Value, value::Kind};
use tonic::{Request, Response, Status};
use uuid::Uuid;

const DP_RANK: u32 = 0;
const DEFAULT_MAX_NEW_TOKENS: u32 = 20;
const MAX_NEW_TOKENS: u32 = 1_000_000;
const MAX_CANDIDATES: usize = 20;

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
}

impl Default for MockerServerConfig {
    fn default() -> Self {
        Self {
            model: "mocker-model".to_string(),
            mode: ServerMode::Aggregated,
            seed: 42,
        }
    }
}

/// vLLM-compatible Generate service driven by one shared Mocker scheduler.
#[derive(Clone)]
pub struct VllmMockerService {
    config: Arc<MockerServerConfig>,
    engine: LiveEngine,
}

impl VllmMockerService {
    pub fn new(config: MockerServerConfig, engine_args: MockEngineArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(
            engine_args.engine_type == EngineType::Vllm,
            "Mocker engine_type must be vllm"
        );
        anyhow::ensure!(engine_args.dp_size == 1, "Mocker dp_size must be 1");
        anyhow::ensure!(
            engine_args.worker_type == WorkerType::Aggregated,
            "Mocker worker_type must be aggregated; use the server mode for the emulated wire role"
        );
        Ok(Self {
            config: Arc::new(config),
            engine: LiveEngine::start(engine_args, DP_RANK)?,
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
    ) -> Result<(PreparedRequest, LiveRequest), Status> {
        let prepared = PreparedRequest::new(request, &self.config)?;
        let live = self
            .engine
            .submit(prepared.direct_request())
            .await
            .map_err(|error| {
                Status::internal(format!("Mocker request submission failed: {error}"))
            })?;
        Ok((prepared, live))
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
        let (prepared, mut live) = self.start_generation(request.into_inner()).await?;
        let mut output_ids = Vec::with_capacity(prepared.max_output_tokens);
        while let Some(signal) = live.recv().await {
            let token_id = checked_token(&signal)?;
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
        let (prepared, mut live) = self.start_generation(request.into_inner()).await?;
        let stream = async_stream::try_stream! {
            yield pb::GenerateResponse {
                prompt_info: Some(prepared.prompt_info()),
                outputs: None,
            };

            let mut generated = 0usize;
            while let Some(signal) = live.recv().await {
                let token_id = checked_token(&signal)?;
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

trait SequenceOutputExt {
    fn with_total_output_tokens(self, count: usize) -> Self;
}

impl SequenceOutputExt for pb::SequenceOutput {
    fn with_total_output_tokens(mut self, count: usize) -> Self {
        if let Some(finish) = self.finish_info.as_mut() {
            finish.num_output_tokens = count as u32;
        }
        self
    }
}

#[derive(Debug)]
struct PreparedRequest {
    uuid: Uuid,
    request_id: String,
    prompt_tokens: Vec<u32>,
    max_output_tokens: usize,
    priority: i32,
    response: pb::ResponseOptions,
    output_token_ids: Vec<u32>,
    mode: ServerMode,
}

impl PreparedRequest {
    fn new(mut request: pb::GenerateRequest, config: &MockerServerConfig) -> Result<Self, Status> {
        if !request.model.is_empty() && request.model != config.model {
            return Err(Status::not_found(format!(
                "model '{}' is not served; expected '{}'",
                request.model, config.model
            )));
        }
        let mut prompt_tokens = match request.prompt.take() {
            Some(pb::generate_request::Prompt::TokenIds(tokens)) => tokens.ids,
            Some(pb::generate_request::Prompt::Text(_)) => {
                return Err(Status::unimplemented(
                    "text tokenization is not available in the CPU-only Mocker server; send token_ids",
                ));
            }
            None => return Err(Status::invalid_argument("prompt is required")),
        };
        if prompt_tokens.is_empty() {
            return Err(Status::invalid_argument("token_ids must not be empty"));
        }
        if request.truncate_prompt_tokens > 0 {
            let keep = request.truncate_prompt_tokens as usize;
            if prompt_tokens.len() > keep {
                prompt_tokens.drain(..prompt_tokens.len() - keep);
            }
        }
        if request
            .sampling
            .as_ref()
            .is_some_and(|sampling| sampling.num_sequences > 1)
        {
            return Err(Status::invalid_argument("num_sequences must be 0 or 1"));
        }

        let kv = request
            .kv
            .as_ref()
            .and_then(|kv| kv.kv_transfer_params.as_ref());
        let is_prefill = kv.is_some_and(|kv| struct_bool(kv, "do_remote_decode") == Some(true))
            && kv.is_none_or(|kv| !kv.fields.contains_key("remote_engine_id"));
        let is_decode = kv.is_some_and(|kv| {
            struct_bool(kv, "do_remote_prefill") == Some(true)
                || kv.fields.contains_key("remote_engine_id")
        });
        match config.mode {
            ServerMode::Aggregated if is_prefill || is_decode => {
                return Err(Status::failed_precondition(
                    "aggregated mock server received disaggregated KV transfer parameters",
                ));
            }
            ServerMode::Prefill if !is_prefill => {
                return Err(Status::failed_precondition(
                    "prefill mock server requires do_remote_decode=true",
                ));
            }
            ServerMode::Decode if !is_decode => {
                return Err(Status::failed_precondition(
                    "decode mock server requires a prefill KV transfer payload",
                ));
            }
            _ => {}
        }

        let max_new_tokens = request
            .stopping
            .as_ref()
            .map(|stopping| stopping.max_new_tokens)
            .unwrap_or_default();
        let max_new_tokens = if max_new_tokens == 0 {
            DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens
        };
        if max_new_tokens > MAX_NEW_TOKENS {
            return Err(Status::invalid_argument(format!(
                "max_new_tokens must not exceed {MAX_NEW_TOKENS}"
            )));
        }

        let request_id = if request.request_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            request.request_id
        };
        let uuid = stable_uuid(config.seed, &request_id);
        let max_output_tokens = max_new_tokens as usize;
        let output_token_ids = (0..max_output_tokens)
            .map(|position| synthetic_token(config.seed, &request_id, position))
            .collect();
        Ok(Self {
            uuid,
            request_id,
            prompt_tokens,
            max_output_tokens,
            priority: request.priority,
            response: request.response.unwrap_or_default(),
            output_token_ids,
            mode: config.mode,
        })
    }

    fn direct_request(&self) -> DirectRequest {
        DirectRequest {
            tokens: self.prompt_tokens.clone(),
            max_output_tokens: self.max_output_tokens,
            output_token_ids: Some(self.output_token_ids.clone()),
            uuid: Some(self.uuid),
            dp_rank: DP_RANK,
            priority: self.priority,
            ..Default::default()
        }
    }

    fn prompt_info(&self) -> pb::PromptInfo {
        let wants_logprobs = self.response.prompt_logprobs;
        let wants_tokens = self.response.prompt_token_ids || wants_logprobs;
        let token_ids = wants_tokens
            .then(|| self.prompt_tokens.clone())
            .unwrap_or_default();
        let (logprobs, ranks, candidate_tokens) = if wants_logprobs {
            let rows = self
                .prompt_tokens
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    if index == 0 {
                        (0.0, 0, pb::CandidateTokenInfo::default())
                    } else {
                        (
                            selected_logprob(*token),
                            1,
                            candidate_info(*token, self.response.prompt_candidates.as_ref()),
                        )
                    }
                })
                .collect::<Vec<_>>();
            unzip_rows(rows)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        pb::PromptInfo {
            num_prompt_tokens: self.prompt_tokens.len() as u32,
            token_ids,
            logprobs,
            ranks,
            candidate_tokens,
        }
    }

    fn sequence_output(&self, token_ids: &[u32], terminal: bool) -> pb::SequenceOutput {
        let wants_logprobs = self.response.output_logprobs;
        let output_ids = self
            .response
            .output_token_ids
            .then(|| token_ids.to_vec())
            .unwrap_or_default();
        let (logprobs, ranks, candidate_tokens) = if wants_logprobs {
            unzip_rows(
                token_ids
                    .iter()
                    .map(|token| {
                        (
                            selected_logprob(*token),
                            1,
                            candidate_info(*token, self.response.output_candidates.as_ref()),
                        )
                    })
                    .collect(),
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let text = self
            .response
            .output_text
            .unwrap_or(true)
            .then(|| {
                token_ids
                    .iter()
                    .map(|token| format!("<token:{token}>"))
                    .collect::<String>()
            })
            .unwrap_or_default();
        pb::SequenceOutput {
            index: 0,
            text,
            num_tokens: token_ids.len() as u32,
            token_ids: output_ids,
            logprobs,
            ranks,
            candidate_tokens,
            finish_info: terminal.then(|| pb::FinishInfo {
                num_output_tokens: token_ids.len() as u32,
                finish_reason: pb::finish_info::FinishReason::Length as i32,
                stop_reason: None,
                kv_transfer_params: (self.mode == ServerMode::Prefill).then(|| self.handoff()),
            }),
        }
    }

    fn handoff(&self) -> Struct {
        let remote_block_ids = self
            .prompt_tokens
            .chunks(64)
            .enumerate()
            .map(|(index, _)| Value {
                kind: Some(Kind::NumberValue(index as f64)),
            })
            .collect();
        Struct {
            fields: BTreeMap::from([
                ("do_remote_decode".to_string(), bool_value(false)),
                ("do_remote_prefill".to_string(), bool_value(true)),
                (
                    "remote_engine_id".to_string(),
                    string_value(format!("mocker-prefill-{}", self.uuid)),
                ),
                ("remote_host".to_string(), string_value("127.0.0.1")),
                ("remote_port".to_string(), number_value(0.0)),
                (
                    "remote_block_ids".to_string(),
                    Value {
                        kind: Some(Kind::ListValue(ListValue {
                            values: remote_block_ids,
                        })),
                    },
                ),
                (
                    "mocker_request_id".to_string(),
                    string_value(self.request_id.clone()),
                ),
            ]),
        }
    }
}

fn stable_uuid(seed: u64, request_id: &str) -> Uuid {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(request_id.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    // Mark the stable digest as an RFC 4122 variant/version-4 UUID. It remains
    // deterministic; these bits only make diagnostics parse cleanly.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

fn synthetic_token(seed: u64, request_id: &str, position: usize) -> u32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(request_id.as_bytes());
    hasher.update(&(position as u64).to_le_bytes());
    let bytes = hasher.finalize();
    1_000 + (u32::from_le_bytes(bytes.as_bytes()[..4].try_into().unwrap()) % 31_000)
}

fn selected_logprob(token_id: u32) -> f32 {
    -0.1 * ((token_id % 10) + 1) as f32
}

fn candidate_info(selected: u32, request: Option<&pb::CandidateTokens>) -> pb::CandidateTokenInfo {
    let ids: Vec<u32> = match request.and_then(|request| request.select.as_ref()) {
        None => Vec::new(),
        Some(pb::candidate_tokens::Select::TopN(count)) => (1..=(*count as usize)
            .min(MAX_CANDIDATES))
            .map(|offset| selected.wrapping_add(offset as u32))
            .collect(),
        Some(pb::candidate_tokens::Select::TokenIds(ids)) => {
            ids.ids.iter().copied().take(MAX_CANDIDATES).collect()
        }
        Some(pb::candidate_tokens::Select::All(true)) => (1..=MAX_CANDIDATES)
            .map(|offset| selected.wrapping_add(offset as u32))
            .collect(),
        Some(pb::candidate_tokens::Select::All(false)) => Vec::new(),
    };
    pb::CandidateTokenInfo {
        tokens: ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| pb::candidate_token_info::TokenInfo {
                id,
                logprob: selected_logprob(selected) - 0.1 * (index + 1) as f32,
                rank: index as u32 + 2,
            })
            .collect(),
    }
}

fn unzip_rows(
    rows: Vec<(f32, u32, pb::CandidateTokenInfo)>,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    let mut logprobs = Vec::with_capacity(rows.len());
    let mut ranks = Vec::with_capacity(rows.len());
    let mut candidates = Vec::with_capacity(rows.len());
    for (logprob, rank, candidate) in rows {
        logprobs.push(logprob);
        ranks.push(rank);
        candidates.push(candidate);
    }
    (logprobs, ranks, candidates)
}

fn struct_bool(value: &Struct, key: &str) -> Option<bool> {
    match value.fields.get(key).and_then(|value| value.kind.as_ref()) {
        Some(Kind::BoolValue(value)) => Some(*value),
        _ => None,
    }
}

fn bool_value(value: bool) -> Value {
    Value {
        kind: Some(Kind::BoolValue(value)),
    }
}

fn string_value(value: impl Into<String>) -> Value {
    Value {
        kind: Some(Kind::StringValue(value.into())),
    }
}

fn number_value(value: f64) -> Value {
    Value {
        kind: Some(Kind::NumberValue(value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: &str) -> pb::GenerateRequest {
        pb::GenerateRequest {
            request_id: id.to_string(),
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 2,
                ..Default::default()
            }),
            response: Some(pb::ResponseOptions {
                prompt_logprobs: true,
                output_text: Some(true),
                output_token_ids: true,
                output_logprobs: true,
                output_candidates: Some(pb::CandidateTokens {
                    select: Some(pb::candidate_tokens::Select::TopN(2)),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn preparation_is_deterministic_and_shapes_logprobs() {
        let config = MockerServerConfig::default();
        let first = PreparedRequest::new(request("stable"), &config).unwrap();
        let second = PreparedRequest::new(request("stable"), &config).unwrap();
        assert_eq!(first.uuid, second.uuid);
        assert_eq!(first.output_token_ids, second.output_token_ids);

        let prompt = first.prompt_info();
        assert_eq!(prompt.token_ids, [1, 2, 3]);
        assert_eq!(prompt.logprobs.len(), 3);
        assert_eq!(prompt.candidate_tokens.len(), 3);

        let output = first.sequence_output(&first.output_token_ids, true);
        assert_eq!(output.num_tokens, 2);
        assert_eq!(output.logprobs.len(), 2);
        assert_eq!(output.candidate_tokens[0].tokens.len(), 2);
        assert_eq!(
            output.finish_info.unwrap().finish_reason,
            pb::finish_info::FinishReason::Length as i32
        );
    }

    #[test]
    fn role_validation_requires_the_expected_handoff_shape() {
        let mut prefill = request("prefill");
        prefill.kv = Some(pb::KvCacheParameters {
            kv_transfer_params: Some(Struct {
                fields: BTreeMap::from([("do_remote_decode".to_string(), bool_value(true))]),
            }),
            ..Default::default()
        });
        let config = MockerServerConfig {
            mode: ServerMode::Prefill,
            ..Default::default()
        };
        let prepared = PreparedRequest::new(prefill, &config).unwrap();
        let finish = prepared.sequence_output(&[42], true).finish_info.unwrap();
        assert!(finish.kv_transfer_params.is_some());

        let error = PreparedRequest::new(request("wrong-role"), &config).unwrap_err();
        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn text_prompts_fail_with_an_actionable_status() {
        let mut request = request("text");
        request.prompt = Some(pb::generate_request::Prompt::Text("hello".to_string()));
        let error = PreparedRequest::new(request, &MockerServerConfig::default()).unwrap_err();
        assert_eq!(error.code(), tonic::Code::Unimplemented);
        assert!(error.message().contains("token_ids"));
    }

    #[test]
    fn service_rejects_non_vllm_or_multi_rank_engines() {
        let sglang = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .build()
            .unwrap();
        assert!(
            VllmMockerService::new(MockerServerConfig::default(), sglang)
                .err()
                .unwrap()
                .to_string()
                .contains("engine_type")
        );

        let multi_rank = MockEngineArgs::builder().dp_size(2).build().unwrap();
        assert!(
            VllmMockerService::new(MockerServerConfig::default(), multi_rank)
                .err()
                .unwrap()
                .to_string()
                .contains("dp_size")
        );
    }

    #[tokio::test]
    async fn unary_generate_aggregates_the_full_response() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(128)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let service = VllmMockerService::new(MockerServerConfig::default(), args).unwrap();
        let response =
            pb::generate_server::Generate::generate(&service, Request::new(request("unary")))
                .await
                .unwrap()
                .into_inner();
        let prompt = response.prompt_info.unwrap();
        assert_eq!(prompt.num_prompt_tokens, 3);
        let output = response.outputs.unwrap();
        assert_eq!(output.num_tokens, 2);
        assert_eq!(output.token_ids.len(), 2);
        assert_eq!(output.finish_info.unwrap().num_output_tokens, 2);
    }
}

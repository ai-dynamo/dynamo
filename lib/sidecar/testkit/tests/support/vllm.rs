// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic_health_v14 as tonic_health;
use tonic_v14 as tonic;

use std::collections::HashMap;
use std::process::Command;
use std::sync::{Arc, Mutex};

use dynamo_backend_common::{BackendError, DisaggregationMode, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sidecar_testkit::control::{Controller, Protocol, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;
use dynamo_sidecar_testkit::server::TestServer;
use dynamo_vllm_mocker::{MockerServerConfig, ServerMode, VllmMockerService};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use dynamo_vllm_sidecar::proto::{
    self as pb,
    control_server::ControlServer,
    inference_server::{Inference, InferenceServer},
};
use futures::stream::BoxStream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::{
    FixtureConfig, GenerateOpening, ProcessFixture, SidecarFixture, WireFixture, fast_engine_args,
    sidecar_command, wait_scheduler_idle,
};

pub struct Fixture {
    config: FixtureConfig,
    pub service: VllmMockerService,
    pub server: TestServer,
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateResponse>>>>,
}

impl Fixture {
    pub fn respond(&self, request_id: &str, responses: Vec<pb::GenerateResponse>) {
        assert!(
            self.scripted
                .lock()
                .unwrap()
                .insert(request_id.to_owned(), responses)
                .is_none()
        );
    }
}

impl SidecarFixture for Fixture {
    type Engine = VllmSidecarEngine;
    type Protocol = Adapter;
    const GENERATE_OPENING: GenerateOpening = GenerateOpening::WaitsForHeaders;

    async fn start(control: Controller<Adapter>, config: FixtureConfig) -> Self {
        let mut args = fast_engine_args(EngineType::Vllm);
        args.speedup_ratio = config.speedup_ratio;
        let service = VllmMockerService::new(
            MockerServerConfig {
                model: config.model.clone(),
                mode: match config.disaggregation_mode {
                    DisaggregationMode::Aggregated => ServerMode::Aggregated,
                    DisaggregationMode::Prefill => ServerMode::Prefill,
                    DisaggregationMode::Decode => ServerMode::Decode,
                    DisaggregationMode::Encode => panic!("Mocker does not support encode mode"),
                },
                ..Default::default()
            },
            args,
        )
        .unwrap();
        let scripted = Arc::new(Mutex::new(HashMap::new()));
        let controlled = ControlledService {
            inner: service.clone(),
            control,
            scripted: scripted.clone(),
        };
        let control_service = service.clone();
        let (health, health_service) = tonic_health::server::health_reporter();
        health
            .set_serving::<ControlServer<VllmMockerService>>()
            .await;
        health
            .set_serving::<InferenceServer<ControlledService>>()
            .await;
        let server = TestServer::start(move |listener, shutdown| async move {
            tonic::transport::Server::builder()
                .add_service(InferenceServer::new(controlled))
                .add_service(ControlServer::new(control_service))
                .add_service(health_service)
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown.await;
                })
                .await?;
            Ok(())
        })
        .await
        .unwrap();
        Self {
            config,
            service,
            server,
            scripted,
        }
    }

    async fn engine(&self) -> Self::Engine {
        let argv = vec![
            "dynamo-vllm-sidecar".into(),
            "--grpc-endpoint".into(),
            self.server.endpoint(),
            "--disaggregation-mode".into(),
            self.config.disaggregation_mode.to_string(),
            "--grpc-connections".into(),
            self.config.connections.to_string(),
            "--grpc-startup-deadline-secs".into(),
            "5".into(),
            "--grpc-connect-attempt-timeout-secs".into(),
            "1".into(),
        ];
        tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)).unwrap().0)
            .await
            .unwrap()
    }

    fn eof_error() -> BackendError {
        BackendError::Unknown
    }

    fn native_model(request: &pb::GenerateRequest) -> Option<&str> {
        Some(&request.model)
    }

    fn active_request_count(&self) -> usize {
        self.service.active_request_count()
    }

    async fn scheduler_idle(&self) {
        wait_scheduler_idle(self.service.metrics_receiver(), || {
            self.active_request_count()
        })
        .await;
    }

    async fn shutdown(&mut self) {
        self.server.shutdown().await.unwrap();
    }
}

impl WireFixture for Fixture {
    fn assert_stream(
        handle: &RequestHandle<Adapter>,
        request: &PreprocessedRequest,
        outputs: &Outputs,
    ) {
        let native = handle.native_request().unwrap();
        assert_eq!(native.model, request.model);
        assert_eq!(
            native.prompt,
            Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: request.token_ids.as_ref().clone()
            }))
        );
        assert_eq!(
            native.stopping.as_ref().unwrap().max_new_tokens,
            request.stop_conditions.max_tokens.unwrap()
        );
        assert_eq!(native.temperature, request.sampling_options.temperature);
        assert_eq!(
            native.decoding.as_ref().unwrap().presence_penalty,
            request
                .sampling_options
                .presence_penalty
                .unwrap_or_default()
        );
        assert_eq!(
            native.decoding.as_ref().unwrap().frequency_penalty,
            request
                .sampling_options
                .frequency_penalty
                .unwrap_or_default()
        );
        assert!(native.response.as_ref().unwrap().output_logprobs);
        assert!(native.response.as_ref().unwrap().prompt_logprobs);
        let native_outputs: Vec<_> = handle
            .native_responses()
            .into_iter()
            .filter_map(|response| response.outputs)
            .collect();
        let outputs: Vec<_> = outputs
            .iter()
            .map(|output| output.as_ref().unwrap())
            .collect();
        let native_tokens: Vec<_> = native_outputs
            .iter()
            .flat_map(|output| &output.token_ids)
            .copied()
            .collect();
        assert_eq!(
            native_tokens.len(),
            request.stop_conditions.max_tokens.unwrap() as usize
        );
        assert_eq!(
            outputs
                .iter()
                .flat_map(|output| &output.token_ids)
                .copied()
                .collect::<Vec<_>>(),
            native_tokens
        );
        assert_eq!(
            outputs
                .iter()
                .filter_map(|output| output.text.as_deref())
                .collect::<String>(),
            native_outputs
                .iter()
                .map(|output| output.text.as_str())
                .collect::<String>()
        );
        assert_eq!(
            outputs
                .iter()
                .filter_map(|output| output.log_probs.as_ref())
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            native_outputs
                .iter()
                .flat_map(|output| &output.logprobs)
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>()
        );
        for output in &outputs {
            assert_eq!(
                output.log_probs.as_ref().map_or(0, Vec::len),
                output.token_ids.len()
            );
            assert_eq!(
                output.top_logprobs.as_ref().map_or(0, Vec::len),
                output.token_ids.len()
            );
        }
        let alternatives: Vec<_> = outputs
            .iter()
            .filter_map(|output| output.top_logprobs.as_ref())
            .flatten()
            .collect();
        let native_positions: Vec<_> = native_outputs
            .iter()
            .flat_map(|output| (0..output.token_ids.len()).map(move |index| (output, index)))
            .collect();
        assert_eq!(alternatives.len(), native_positions.len());
        for (candidates, (native, index)) in alternatives.iter().zip(native_positions) {
            assert_eq!(
                candidates.len(),
                request.output_options.logprobs.unwrap() as usize + 1
            );
            let expected = std::iter::once((
                native.token_ids[index],
                native.ranks[index],
                f64::from(native.logprobs[index]),
            ))
            .chain(
                native.candidate_tokens[index]
                    .tokens
                    .iter()
                    .map(|token| (token.id, token.rank, f64::from(token.logprob))),
            )
            .collect::<Vec<_>>();
            assert_eq!(
                candidates
                    .iter()
                    .map(|token| (token.token_id, token.rank, token.logprob))
                    .collect::<Vec<_>>(),
                expected
            );
        }
        assert!(
            outputs[..outputs.len() - 1]
                .iter()
                .all(|output| output.engine_data.is_none())
        );
        let prompts = &outputs.last().unwrap().engine_data.as_ref().unwrap()["prompt_logprobs"];
        let prompt_info = handle
            .native_responses()
            .into_iter()
            .find_map(|response| response.prompt_info)
            .unwrap();
        assert_eq!(prompts.as_array().unwrap().len(), request.token_ids.len());
        assert!(prompts[0].is_null());
        for index in 1..request.token_ids.len() {
            let selected = &prompts[index][request.token_ids[index].to_string()];
            assert_eq!(
                selected["logprob"].as_f64().unwrap(),
                f64::from(prompt_info.logprobs[index])
            );
            assert_eq!(selected["rank"], prompt_info.ranks[index]);
        }
    }

    async fn scheduler_active(&self) {
        let mut metrics = self.service.metrics_receiver();
        dynamo_sidecar_testkit::bounded("Mocker active scheduler work", async {
            loop {
                let snapshot = metrics.borrow_and_update().clone();
                if snapshot.running_requests + snapshot.waiting_requests > 0 {
                    assert!(self.service.active_request_count() > 0);
                    return;
                }
                metrics.changed().await.unwrap();
            }
        })
        .await;
    }
}

#[derive(Clone)]
struct ControlledService {
    inner: VllmMockerService,
    control: Controller<Adapter>,
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateResponse>>>>,
}

#[tonic::async_trait]
impl Inference for ControlledService {
    type GenerateStreamStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        self.inner.generate(request).await
    }

    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let opened = self.control.open(request.get_ref()).await?;
        let scripted = self
            .scripted
            .lock()
            .unwrap()
            .remove(&request.get_ref().request_id);
        if let Some(responses) = scripted {
            return Ok(Response::new(opened.wrap(Box::pin(futures::stream::iter(
                responses.into_iter().map(Ok),
            )))));
        }
        let response = self.inner.generate_stream(request).await?;
        Ok(Response::new(opened.wrap(response.into_inner())))
    }
}

#[derive(Clone, Copy)]
pub struct Adapter;

impl Protocol for Adapter {
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateResponse;
    type Error = Status;

    fn request_id(request: &Self::Request) -> &str {
        &request.request_id
    }

    fn record_tokens(response: &Self::Response, tokens: &mut Vec<u32>) -> bool {
        let Some(output) = &response.outputs else {
            return false;
        };
        tokens.extend_from_slice(&output.token_ids);
        !output.token_ids.is_empty()
    }

    fn is_terminal(response: &Self::Response) -> bool {
        response
            .outputs
            .as_ref()
            .is_some_and(|output| output.finish_info.is_some())
    }

    fn injected_error(message: &'static str) -> Self::Error {
        Status::unavailable(message)
    }
}

impl ProcessFixture for Fixture {
    fn endpoint(&self) -> String {
        self.server.endpoint()
    }

    fn command() -> Command {
        let mut command = sidecar_command("dynamo-vllm-sidecar", "DYNAMO_VLLM_SIDECAR");
        command.env_remove("VLLM_HTTP_ENDPOINT");
        command
    }

    fn configure_request(request: &mut PreprocessedRequest) {
        request.sampling_options.temperature = Some(0.125);
        request.sampling_options.presence_penalty = Some(0.25);
        request.sampling_options.frequency_penalty = Some(0.75);
        request.output_options.logprobs = Some(2);
        request.output_options.prompt_logprobs = Some(1);
    }

    fn assert_registration(card: &ModelDeploymentCard) {
        assert_eq!(card.kv_cache_block_size, 4);
        assert_eq!(card.runtime_config.total_kv_blocks, Some(4096));
        assert_eq!(card.runtime_config.max_num_seqs, Some(64));
        assert_eq!(card.runtime_config.max_num_batched_tokens, Some(1024));
        assert_eq!(card.runtime_config.data_parallel_start_rank, 0);
        assert_eq!(card.runtime_config.data_parallel_size, 1);
        assert!(card.runtime_config.tool_call_parser.is_none());
        assert!(card.runtime_config.reasoning_parser.is_none());
        assert_eq!(card.effective_context_length(), 4096);
    }
}

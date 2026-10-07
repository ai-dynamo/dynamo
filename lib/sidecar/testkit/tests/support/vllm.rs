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
    control_server::{Control, ControlServer},
    inference_server::{Inference, InferenceServer},
};
use futures::stream::BoxStream;
use tokio::sync::watch;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::server::NamedService;
use tonic::{Request, Response, Status};
use tonic_health::pb::{
    HealthCheckRequest, HealthCheckResponse,
    health_server::{Health, HealthServer},
};

use super::{
    FixtureConfig, GenerateOpening, HandoffFixture, ProcessFixture, SidecarFixture, WireFixture,
    fast_engine_args, sidecar_command, wait_scheduler_idle,
};

pub struct Fixture {
    config: FixtureConfig,
    pub service: VllmMockerService,
    pub server: TestServer,
    scripted: Arc<Mutex<HashMap<String, Vec<pb::GenerateResponse>>>>,
    served_model_name: Arc<Mutex<Option<String>>>,
    readiness: watch::Sender<Option<bool>>,
    health_received: watch::Receiver<bool>,
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
        let served_model_name = Arc::new(Mutex::new(None));
        let controlled = ControlledService {
            inner: service.clone(),
            control,
            scripted: scripted.clone(),
            served_model_name: served_model_name.clone(),
        };
        let control_service = controlled.clone();
        let health = tonic_health::server::HealthReporter::new();
        health
            .set_serving::<ControlServer<ControlledService>>()
            .await;
        health
            .set_serving::<InferenceServer<ControlledService>>()
            .await;
        let (readiness, is_healthy) = watch::channel(Some(true));
        let (health_requested, health_received) = watch::channel(false);
        let health_service = HealthServer::new(ControlledHealth {
            inner: tonic_health::server::HealthService::from_health_reporter(health),
            is_healthy,
            health_requested,
        });
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
            served_model_name,
            readiness,
            health_received,
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
    served_model_name: Arc<Mutex<Option<String>>>,
}

impl ControlledService {
    fn normalize_model_alias(&self, request: &mut pb::GenerateRequest) {
        if self.served_model_name.lock().unwrap().as_deref() == Some(request.model.as_str()) {
            request.model.clone_from(&self.inner.config().model);
        }
    }
}

#[tonic::async_trait]
impl Inference for ControlledService {
    type GenerateStreamStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;

    async fn generate(
        &self,
        mut request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        self.normalize_model_alias(request.get_mut());
        self.inner.generate(request).await
    }

    async fn generate_stream(
        &self,
        mut request: Request<pb::GenerateRequest>,
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
        self.normalize_model_alias(request.get_mut());
        let response = self.inner.generate_stream(request).await?;
        Ok(Response::new(opened.wrap(response.into_inner())))
    }
}

macro_rules! delegate_control {
    ($($method:ident($request:ty) -> $response:ty;)*) => {
        #[tonic::async_trait]
        impl Control for ControlledService {
            async fn get_model_info(
                &self,
                request: Request<pb::GetModelInfoRequest>,
            ) -> Result<Response<pb::ModelInfo>, Status> {
                let mut response = self.inner.get_model_info(request).await?;
                if let Some(name) = self.served_model_name.lock().unwrap().as_ref() {
                    response.get_mut().served_model_name.clone_from(name);
                }
                Ok(response)
            }

            $(
                async fn $method(
                    &self,
                    request: Request<$request>,
                ) -> Result<Response<$response>, Status> {
                    self.inner.$method(request).await
                }
            )*
        }
    };
}

delegate_control! {
    get_server_info(pb::GetServerInfoRequest) -> pb::ServerInfo;
    abort(pb::AbortRequest) -> pb::AbortResponse;
    load_lora(pb::LoadLoraRequest) -> pb::LoadLoraResponse;
    unload_lora(pb::UnloadLoraRequest) -> pb::UnloadLoraResponse;
    list_loras(pb::ListLorasRequest) -> pb::ListLorasResponse;
    get_kv_event_sources(pb::GetKvEventSourcesRequest) -> pb::GetKvEventSourcesResponse;
    pause_generation(pb::PauseGenerationRequest) -> pb::PauseGenerationResponse;
    resume_generation(pb::ResumeGenerationRequest) -> pb::ResumeGenerationResponse;
    is_paused(pb::IsPausedRequest) -> pb::IsPausedResponse;
    sleep(pb::SleepRequest) -> pb::SleepResponse;
    wake_up(pb::WakeUpRequest) -> pb::WakeUpResponse;
    is_sleeping(pb::IsSleepingRequest) -> pb::IsSleepingResponse;
    init_weight_transfer_engine(pb::InitWeightTransferEngineRequest) -> pb::InitWeightTransferEngineResponse;
    start_weight_update(pb::StartWeightUpdateRequest) -> pb::StartWeightUpdateResponse;
    start_draft_weight_update(pb::StartDraftWeightUpdateRequest) -> pb::StartDraftWeightUpdateResponse;
    update_weights(pb::UpdateWeightsRequest) -> pb::UpdateWeightsResponse;
    finish_weight_update(pb::FinishWeightUpdateRequest) -> pb::FinishWeightUpdateResponse;
    update_weight_version(pb::UpdateWeightVersionRequest) -> pb::UpdateWeightVersionResponse;
    get_weight_version(pb::GetWeightVersionRequest) -> pb::GetWeightVersionResponse;
}

struct ControlledHealth {
    inner: tonic_health::server::HealthService,
    is_healthy: watch::Receiver<Option<bool>>,
    health_requested: watch::Sender<bool>,
}

#[tonic::async_trait]
impl Health for ControlledHealth {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        if request.get_ref().service != InferenceServer::<ControlledService>::NAME {
            return self.inner.check(request).await;
        }
        self.health_requested.send_replace(true);
        let mut readiness = self.is_healthy.clone();
        let is_healthy = readiness
            .wait_for(|is_healthy| is_healthy.is_some())
            .await
            .map_err(|_| Status::unavailable("health fixture stopped"))?
            .unwrap();
        let status = if is_healthy {
            tonic_health::pb::health_check_response::ServingStatus::Serving
        } else {
            tonic_health::pb::health_check_response::ServingStatus::NotServing
        };
        Ok(Response::new(HealthCheckResponse {
            status: status as i32,
        }))
    }

    type WatchStream = <tonic_health::server::HealthService as Health>::WatchStream;

    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        self.inner.watch(request).await
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
    fn set_served_model_name(&self, name: &str) {
        *self.served_model_name.lock().unwrap() = Some(name.to_owned());
    }

    fn set_health(&self, is_healthy: Option<bool>) {
        self.readiness.send_replace(is_healthy);
    }

    async fn health_check_received(&self) {
        let mut received = self.health_received.clone();
        dynamo_sidecar_testkit::bounded(
            "vLLM Inference health check received",
            received.wait_for(|has_received| *has_received),
        )
        .await
        .unwrap();
    }

    fn assert_unhealthy_startup(logs: &str) {
        assert!(logs.contains("did not become SERVING"), "{logs}");
    }

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

impl HandoffFixture for Fixture {
    const HAS_BOOTSTRAP: bool = false;

    fn assert_handoff(
        prefill: &RequestHandle<Self::Protocol>,
        decode: &RequestHandle<Self::Protocol>,
        id: &str,
    ) {
        let prefill_wire = prefill.native_request().unwrap();
        let decode_wire = decode.native_request().unwrap();
        assert_eq!(prefill_wire.request_id, id);
        assert_eq!(decode_wire.request_id, id);
        assert_eq!(prefill_wire.stopping.unwrap().max_new_tokens, 1);
        assert_eq!(decode_wire.stopping.unwrap().max_new_tokens, 3);
        let mut native_handoff = prefill
            .native_responses()
            .into_iter()
            .find_map(|response| response.outputs?.finish_info?.kv_transfer_params)
            .expect("native prefill handoff");
        let mut forwarded = decode_wire.kv.unwrap().kv_transfer_params.unwrap();
        assert_eq!(
            forwarded.fields["mocker_request_id"].kind,
            Some(prost_types_v14::value::Kind::StringValue(id.to_string()))
        );
        assert_eq!(
            native_handoff.fields.remove("remote_port").unwrap().kind,
            Some(prost_types_v14::value::Kind::NumberValue(0.0))
        );
        assert_eq!(
            forwarded.fields.remove("remote_port").unwrap().kind,
            Some(prost_types_v14::value::Kind::StringValue("0".to_string()))
        );
        assert_eq!(forwarded, native_handoff);
    }
}

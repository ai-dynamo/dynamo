// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use dynamo_backend_common::{BackendError, DisaggregationMode, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_mocker::common::protocols::EngineType;
use dynamo_sidecar_testkit::control::{Controller, Protocol, RequestHandle};
use dynamo_sidecar_testkit::fixtures::Outputs;
use dynamo_sidecar_testkit::server::TestServer;
use dynamo_trtllm_mocker::{MockerServerConfig, ServerMode, TrtllmMockerService};
use dynamo_trtllm_sidecar::TrtllmSidecarEngine;
use dynamo_trtllm_sidecar::proto::{
    self as pb,
    control_server::ControlServer,
    inference_server::{Inference, InferenceServer},
};
use futures::stream::BoxStream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use super::{
    FixtureConfig, GenerateOpening, HandoffFixture, ProcessFixture, SidecarFixture, WireFixture,
    fast_engine_args, sidecar_command, wait_scheduler_idle,
};

pub struct Fixture {
    config: FixtureConfig,
    service: TrtllmMockerService,
    server: TestServer,
}

impl SidecarFixture for Fixture {
    type Engine = TrtllmSidecarEngine;
    type Protocol = Adapter;
    const GENERATE_OPENING: GenerateOpening = GenerateOpening::WaitsForHeaders;

    async fn start(control: Controller<Adapter>, config: FixtureConfig) -> Self {
        let mut args = fast_engine_args(EngineType::Trtllm);
        args.speedup_ratio = config.speedup_ratio;
        let service = TrtllmMockerService::new(
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
        let controlled = ControlledInference {
            inner: service.clone(),
            control,
        };
        let control_service = service.clone();
        let server = TestServer::start(move |listener, shutdown| async move {
            tonic::transport::Server::builder()
                .add_service(InferenceServer::new(controlled))
                .add_service(ControlServer::new(control_service))
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
        }
    }

    async fn engine(&self) -> Self::Engine {
        TrtllmSidecarEngine::from_args(vec![
            "dynamo-trtllm-sidecar".into(),
            "--grpc-endpoint".into(),
            self.server.endpoint(),
            "--model-path".into(),
            self.config.model.clone(),
            "--disaggregation-mode".into(),
            self.config.disaggregation_mode.to_string(),
            "--grpc-connections".into(),
            self.config.connections.to_string(),
            "--grpc-startup-deadline-secs".into(),
            "5".into(),
            "--grpc-connect-attempt-timeout-secs".into(),
            "1".into(),
        ])
        .unwrap()
        .0
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
            native.input,
            Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: request.token_ids.as_ref().clone(),
            }))
        );
        assert_eq!(
            native.stopping.as_ref().unwrap().max_tokens,
            request.stop_conditions.max_tokens
        );
        assert_eq!(
            native.sampling.as_ref().unwrap().temperature,
            request.sampling_options.temperature.map(f64::from)
        );
        let response = native.response.as_ref().unwrap();
        assert_eq!(response.return_output_logprobs, Some(true));
        assert_eq!(response.return_prompt_logprobs, None);
        assert_eq!(
            response.output_candidates.as_ref().unwrap().selection,
            Some(pb::candidate_token_selection::Selection::TopN(2))
        );

        let native_responses = handle.native_responses();
        let tokens: Vec<_> = native_responses
            .iter()
            .filter_map(|response| match &response.event {
                Some(pb::generate_response::Event::Token(output)) => Some(output),
                _ => None,
            })
            .collect();
        let outputs: Vec<_> = outputs
            .iter()
            .map(|output| output.as_ref().unwrap())
            .collect();
        assert_eq!(outputs.len(), tokens.len() + 1);
        assert_eq!(
            tokens.len(),
            request.stop_conditions.max_tokens.unwrap() as usize
        );
        for (output, native) in outputs.iter().zip(tokens) {
            assert_eq!(
                output.token_ids,
                native
                    .tokens
                    .iter()
                    .map(|token| token.token_id)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                output.log_probs.as_ref().unwrap(),
                &native
                    .tokens
                    .iter()
                    .map(|token| token.logprob.unwrap())
                    .collect::<Vec<_>>()
            );
            let alternatives = output.top_logprobs.as_ref().unwrap();
            assert_eq!(alternatives.len(), native.tokens.len());
            for (actual, token) in alternatives.iter().zip(&native.tokens) {
                assert_eq!(actual.len(), 2);
                assert_eq!(actual.len(), token.candidates.len());
                for (actual, expected) in actual.iter().zip(&token.candidates) {
                    assert_eq!(actual.token_id, expected.token_id);
                    assert_eq!(actual.logprob, expected.logprob);
                    assert_eq!(actual.rank, expected.rank.unwrap());
                }
            }
        }
        assert!(outputs.last().unwrap().token_ids.is_empty());
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
struct ControlledInference {
    inner: TrtllmMockerService,
    control: Controller<Adapter>,
}

#[tonic::async_trait]
impl Inference for ControlledInference {
    type GenerateStream = BoxStream<'static, Result<pb::GenerateResponse, Status>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let opened = self.control.open(request.get_ref()).await?;
        let response = self.inner.generate(request).await?;
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
        if let Some(pb::generate_response::Event::Token(output)) = &response.event {
            tokens.extend(output.tokens.iter().map(|token| token.token_id));
            !output.tokens.is_empty()
        } else {
            false
        }
    }

    fn is_terminal(response: &Self::Response) -> bool {
        matches!(
            response.event,
            Some(
                pb::generate_response::Event::Finished(_)
                    | pb::generate_response::Event::PrefillReady(_)
                    | pb::generate_response::Event::Error(_)
            )
        )
    }

    fn injected_error(message: &'static str) -> Self::Error {
        Status::unavailable(message)
    }
}

impl ProcessFixture for Fixture {
    fn endpoint(&self) -> String {
        self.server.endpoint()
    }

    fn command(model: &str) -> Command {
        let mut command = sidecar_command("dynamo-trtllm-sidecar", "DYNAMO_TRTLLM_SIDECAR");
        command.args(["--model-path", model]);
        command
    }

    fn configure_request(request: &mut PreprocessedRequest) {
        request.sampling_options.temperature = Some(0.125);
        request.output_options.logprobs = Some(2);
    }

    fn assert_registration(card: &ModelDeploymentCard) {
        // OpenEngine currently supplies the context window, not KV-routing capacity.
        assert_eq!(card.effective_context_length(), 32_768);
        assert_eq!(card.kv_cache_block_size, 16);
        assert!(card.runtime_config.total_kv_blocks.is_none());
        assert!(card.runtime_config.tool_call_parser.is_none());
        assert!(card.runtime_config.reasoning_parser.is_none());
    }
}

impl HandoffFixture for Fixture {
    const HAS_BOOTSTRAP: bool = false;
    const KV_BLOCK_SIZE: u32 = 16;

    fn assert_handoff(prefill: &RequestHandle<Adapter>, decode: &RequestHandle<Adapter>, id: &str) {
        let prefill_wire = prefill.native_request().unwrap();
        let decode_wire = decode.native_request().unwrap();
        assert_eq!(prefill_wire.request_id, id);
        assert_eq!(decode_wire.request_id, id);
        assert_eq!(prefill_wire.stopping.unwrap().max_tokens, Some(1));
        assert_eq!(decode_wire.stopping.unwrap().max_tokens, Some(3));
        let session = prefill
            .native_responses()
            .into_iter()
            .find_map(|response| match response.event {
                Some(pb::generate_response::Event::PrefillReady(ready)) => ready.kv_session,
                _ => None,
            })
            .expect("prefill must return a KV session");
        assert!(!session.session_id.is_empty());
        assert_eq!(session.transfer_backend, "MOCKER");
        assert_eq!(decode_wire.kv.unwrap().session.unwrap(), session);
    }
}

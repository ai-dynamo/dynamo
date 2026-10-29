// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An in-process fake of an LPX engine's OpenEngine `Inference` and
//! `Control` services on a loopback port.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::DisaggregationMode;
use futures::Stream;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::pb;

/// The token every scripted stream starts with.
const FIRST_TOKEN: u32 = 100;
/// The EOS token a completed stream reports as its stop match.
const EOS_TOKEN: u32 = 2;

/// What `Generate` does with a request it has recorded.
#[derive(Clone, Copy)]
pub(super) enum Behavior {
    /// Streams one token, then finishes on EOS.
    Complete,
    /// Streams one token, then fails the request in-band as retryable.
    ErrorAfterFirstToken,
    /// Never answers, so the response stream never opens.
    NeverOpens,
    /// Opens the stream and never sends a message.
    HangBeforeFirstToken,
}

pub(super) struct State {
    server_info: pb::ServerInfo,
    /// GetServerInfo never answers this many more times.
    pub(super) server_info_stalls: Mutex<usize>,
    /// The next GetModelInfo fails with this status, when set.
    pub(super) model_info_status: Mutex<Option<tonic::Code>>,
    /// GetModelInfo never answers this many more times.
    pub(super) model_info_stalls: Mutex<usize>,
    pub(super) model_info_calls: AtomicUsize,
    pub(super) behavior: Mutex<Behavior>,

    pub(super) requests: Mutex<Vec<pb::GenerateRequest>>,
    /// The trace metadata each Generate call carried.
    pub(super) metadata: Mutex<Vec<BTreeMap<String, String>>>,
    pub(super) aborts: Mutex<Vec<String>>,
    /// Requests whose response stream the client dropped before it finished.
    pub(super) dropped: Mutex<Vec<String>>,
}

#[derive(Clone)]
pub(super) struct FakeEngine(pub(super) Arc<State>);

impl FakeEngine {
    pub(super) fn new(mode: DisaggregationMode) -> Self {
        Self(Arc::new(State {
            server_info: server_info(mode),
            server_info_stalls: Mutex::new(0),
            model_info_status: Mutex::new(None),
            model_info_stalls: Mutex::new(0),
            model_info_calls: AtomicUsize::new(0),
            behavior: Mutex::new(Behavior::Complete),
            requests: Mutex::new(Vec::new()),
            metadata: Mutex::new(Vec::new()),
            aborts: Mutex::new(Vec::new()),
            dropped: Mutex::new(Vec::new()),
        }))
    }

    pub(super) fn with_behavior(self, behavior: Behavior) -> Self {
        *self.0.behavior.lock().unwrap() = behavior;
        self
    }

    pub(super) fn requests(&self) -> Vec<pb::GenerateRequest> {
        self.0.requests.lock().unwrap().clone()
    }

    pub(super) fn dropped(&self) -> Vec<String> {
        self.0.dropped.lock().unwrap().clone()
    }

    /// Waits until the client has dropped `request_id`'s stream.
    pub(super) async fn wait_for_drop(&self, request_id: &str) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !self.dropped().iter().any(|id| id == request_id) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the client never dropped the stream of {request_id}"));
    }
}

/// The fields a healthy engine in `mode` reports that the sidecar reads.
pub(super) fn server_info(mode: DisaggregationMode) -> pb::ServerInfo {
    pb::ServerInfo {
        engine_name: "lpx-test".to_string(),
        engine_version: "1.13.1-test".to_string(),
        engine_role: if mode.is_decode() {
            pb::EngineRole::Decode
        } else {
            pb::EngineRole::Aggregated
        } as i32,
        instance_id: "lpx-test-instance-7".to_string(),
        minimum_client_revision: 1,
        capacity: Some(pb::DeploymentCapacity {
            kv_block_size: Some(32),
            total_kv_blocks: Some(256),
            max_running_requests: Some(4),
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub(super) fn model_info() -> pb::ModelInfo {
    pb::ModelInfo {
        served_model_name: "llama32-1b".to_string(),
        max_context_length: Some(8192),
        ..Default::default()
    }
}

/// Records the stream as dropped unless it ran to its end.
struct DropGuard {
    request_id: String,
    state: Arc<State>,
    finished: bool,
}

impl DropGuard {
    /// Called as a method so that the stream captures the whole guard: a
    /// stream that only wrote `finished` would capture just that field.
    fn finish(&mut self) {
        self.finished = true;
    }
}

impl Drop for DropGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.state
                .dropped
                .lock()
                .unwrap()
                .push(self.request_id.clone());
        }
    }
}

fn response(request_id: &str, event: pb::generate_response::Event) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(event),
        usage: None,
    }
}

fn first_token(request_id: &str) -> pb::GenerateResponse {
    response(
        request_id,
        pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: Some(0),
            tokens: vec![pb::TokenInfo {
                token_id: FIRST_TOKEN,
                ..Default::default()
            }],
            text: String::new(),
        }),
    )
}

#[tonic::async_trait]
impl pb::inference_server::Inference for FakeEngine {
    type GenerateStream = Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let trace = ["traceparent", "tracestate"]
            .into_iter()
            .filter_map(|key| {
                let value = request.metadata().get(key)?.to_str().ok()?;
                Some((key.to_string(), value.to_string()))
            })
            .collect();
        self.0.metadata.lock().unwrap().push(trace);
        let request = request.into_inner();
        self.0.requests.lock().unwrap().push(request.clone());
        let behavior = *self.0.behavior.lock().unwrap();
        if let Behavior::NeverOpens = behavior {
            std::future::pending::<()>().await;
        }

        let request_id = request.request_id.clone();
        let mut guard = DropGuard {
            request_id: request_id.clone(),
            state: Arc::clone(&self.0),
            finished: false,
        };

        let stream = async_stream::try_stream! {
            match behavior {
                Behavior::Complete => {
                    yield first_token(&request_id);
                    yield response(
                        &request_id,
                        pb::generate_response::Event::Finished(pb::GenerationFinished {
                            output_index: Some(0),
                            reason: pb::FinishReason::Stop as i32,
                            message: String::new(),
                            stop_match: Some(pb::StopMatch {
                                r#match: Some(pb::stop_match::Match::EosTokenId(EOS_TOKEN)),
                            }),
                        }),
                    );
                }
                Behavior::ErrorAfterFirstToken => {
                    yield first_token(&request_id);
                    yield response(
                        &request_id,
                        pb::generate_response::Event::Error(pb::EngineError {
                            code: pb::ErrorCode::Internal as i32,
                            message: "scripted failure".to_string(),
                            retryable: true,
                        }),
                    );
                }
                Behavior::NeverOpens => unreachable!("Generate never answered"),
                Behavior::HangBeforeFirstToken => {
                    std::future::pending::<()>().await;
                }
            }
            guard.finish();
        };
        Ok(Response::new(Box::pin(stream)))
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for FakeEngine {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        let stall = {
            let mut left = self.0.server_info_stalls.lock().unwrap();
            let stall = *left > 0;
            if stall {
                *left -= 1;
            }
            stall
        };
        if stall {
            std::future::pending::<()>().await;
        }
        Ok(Response::new(self.0.server_info.clone()))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        self.0.model_info_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(code) = self.0.model_info_status.lock().unwrap().take() {
            return Err(Status::new(code, "scripted GetModelInfo failure"));
        }
        let stall = {
            let mut left = self.0.model_info_stalls.lock().unwrap();
            let stall = *left > 0;
            if stall {
                *left -= 1;
            }
            stall
        };
        if stall {
            std::future::pending::<()>().await;
        }
        Ok(Response::new(model_info()))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request_id = match request.into_inner().target {
            Some(pb::abort_request::Target::RequestId(id)) => id,
            other => {
                return Err(Status::invalid_argument(format!(
                    "unexpected abort target {other:?}"
                )));
            }
        };
        self.0.aborts.lock().unwrap().push(request_id);
        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: String::new(),
        }))
    }

    async fn get_load(
        &self,
        _request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        Err(Status::unimplemented("GetLoad is not used"))
    }

    async fn health(
        &self,
        _request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        Err(Status::unimplemented("Health is not used"))
    }

    async fn load_lora(
        &self,
        _request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        Err(Status::unimplemented(
            "LoadLora is not implemented by the fake",
        ))
    }

    async fn unload_lora(
        &self,
        _request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        Err(Status::unimplemented(
            "UnloadLora is not implemented by the fake",
        ))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        Err(Status::unimplemented(
            "ListLoras is not implemented by the fake",
        ))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        Err(Status::unimplemented(
            "GetKvEventSources is not implemented by the fake",
        ))
    }

    type SubscribeKvEventsStream =
        Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented(
            "SubscribeKvEvents is not implemented by the fake",
        ))
    }
}

/// A FakeEngine served on a loopback port until dropped.
pub(super) struct FakeServer {
    pub(super) endpoint: String,
    pub(super) fake: FakeEngine,
    /// Dropping it starts the graceful shutdown: no new
    /// connections, and each open one closes once its streams end.
    _shutdown: oneshot::Sender<()>,
}

impl FakeServer {
    pub(super) async fn start(fake: FakeEngine) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let (shutdown, shutdown_rx) = oneshot::channel();
        let router = tonic::transport::Server::builder()
            .add_service(pb::inference_server::InferenceServer::new(fake.clone()))
            .add_service(pb::control_server::ControlServer::new(fake.clone()));
        tokio::spawn(async move {
            router
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("serve the fake engine");
        });
        Self {
            endpoint: format!("http://{address}"),
            fake,
            _shutdown: shutdown,
        }
    }
}

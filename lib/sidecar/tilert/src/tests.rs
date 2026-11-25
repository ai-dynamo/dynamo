// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic_health_v14 as tonic_health;
use tonic_v14 as tonic;

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use dynamo_backend_common::testing::{mock_context, run_conformance};
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, DynamoError, FinishReason, GenerateContext,
    GuidedDecodingOptions, LLMEngine, LLMEngineOutput, PreprocessedRequest, SamplingOptions,
    StopConditions,
};
use dynamo_sidecar_common::HttpEndpoint;
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::time::Instant;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};
use tonic_health::ServingStatus;
use vllm_proto as pb;

use crate::args::Args;
use crate::decode::{DecodeCapabilities, DecodeClient};
use crate::engine::{RequestPlan, TilertSidecarEngine};
use crate::error::is_kind;

const FIRST_TOKEN: u32 = 42;
const EOS: u32 = 1;
/// Bounds every wait, so a deadlock fails the test instead of hanging it.
const STEP: Duration = Duration::from_secs(10);

/// How the fake decode server answers a streaming `/pd/decode`.
#[derive(Clone, Copy, Debug, Default)]
enum Decode {
    #[default]
    Complete,
    /// One token batch, then nothing until the client cancels or goes away.
    Stall,
    ErrorAfterTokens,
    EndsWithoutDone,
    WrongCount,
    Refuse,
}

/// What the two fake engines saw, and the knobs that change their answers.
#[derive(Default)]
struct Shared {
    /// Request whose KV has landed and waits for its decode. Like the real
    /// server, the fake holds one and refuses every other request's KV.
    landed: Option<String>,
    /// Requests a decode gave up waiting for; their KV is discarded on arrival.
    dropped: HashSet<String>,
    /// Decode in progress, the only thing `/pd/cancel` matches.
    decoding: Option<String>,
    prefill_requests: Vec<pb::GenerateRequest>,
    aborted: Vec<String>,
    decode_bodies: Vec<Value>,
    cancelled: Vec<String>,

    eos_first: bool,
    /// The prefill pushes no KV, as vLLM does without TileRT's connector.
    no_kv: bool,
    /// The prefill pushes its KV and then fails.
    prefill_fails: bool,
    /// The prefill pushes its KV and then never answers.
    prefill_hangs: bool,
    decode: Decode,
    penalties: bool,
    /// The decode server's `/health` fails.
    unhealthy: bool,
    /// `/pd/decode` waits at `decode_gate` before it looks for the KV.
    hold_decode: bool,
    decode_entered: Arc<Notify>,
    decode_gate: Arc<Notify>,
}

impl Shared {
    fn push_kv(&mut self, rid: &str) {
        if self.dropped.remove(rid) {
            return;
        }
        if self.landed.is_none() && self.decoding.is_none() {
            self.landed = Some(rid.to_string());
        }
    }
}

type SharedState = Arc<Mutex<Shared>>;

#[derive(Clone)]
struct FakeVllm {
    shared: SharedState,
    prefill_entered: Arc<Notify>,
}

#[tonic::async_trait]
impl pb::inference_server::Inference for FakeVllm {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    async fn generate(
        &self,
        _request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        Err(Status::unimplemented("unary generation is not used"))
    }

    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let request = request.into_inner();
        let prompt_tokens = match &request.prompt {
            Some(pb::generate_request::Prompt::TokenIds(ids)) => ids.ids.len() as u32,
            _ => 0,
        };
        let (eos_first, fails, hangs) = {
            let mut shared = self.shared.lock().unwrap();
            if !shared.no_kv {
                shared.push_kv(&request.request_id);
            }
            shared.prefill_requests.push(request);
            (shared.eos_first, shared.prefill_fails, shared.prefill_hangs)
        };
        self.prefill_entered.notify_one();
        if hangs {
            std::future::pending::<()>().await;
        }
        if fails {
            return Err(Status::internal("the prefill failed"));
        }
        let finish = if eos_first {
            pb::FinishInfo {
                finish_reason: pb::finish_info::FinishReason::Stop as i32,
                stop_reason: Some(pb::finish_info::StopReason::EosTokenId(EOS)),
                num_cached_tokens: Some(2),
                ..Default::default()
            }
        } else {
            pb::FinishInfo {
                num_output_tokens: 1,
                finish_reason: pb::finish_info::FinishReason::Length as i32,
                num_cached_tokens: Some(2),
                ..Default::default()
            }
        };
        let response = pb::GenerateResponse {
            prompt_info: Some(pb::PromptInfo {
                num_prompt_tokens: prompt_tokens,
                ..Default::default()
            }),
            outputs: Some(pb::SequenceOutput {
                token_ids: if eos_first { vec![] } else { vec![FIRST_TOKEN] },
                finish_info: Some(finish),
                ..Default::default()
            }),
        };
        Ok(Response::new(Box::pin(futures::stream::iter([Ok(
            response,
        )]))))
    }
}

/// `vllm.Control` with a working `Abort`; the sidecar calls nothing else.
macro_rules! fake_control {
    ($($rpc:ident($request:ident) -> $response:ident),* $(,)?) => {
        #[tonic::async_trait]
        impl pb::control_server::Control for FakeVllm {
            async fn abort(
                &self,
                request: Request<pb::AbortRequest>,
            ) -> Result<Response<pb::AbortResponse>, Status> {
                let mut shared = self.shared.lock().unwrap();
                shared.aborted.extend(request.into_inner().request_ids);
                Ok(Response::new(pb::AbortResponse {}))
            }

            $(async fn $rpc(
                &self,
                _: Request<pb::$request>,
            ) -> Result<Response<pb::$response>, Status> {
                Err(Status::unimplemented(stringify!($rpc)))
            })*
        }
    };
}

fake_control! {
    get_server_info(GetServerInfoRequest) -> ServerInfo,
    get_model_info(GetModelInfoRequest) -> ModelInfo,
    shutdown(ShutdownRequest) -> ShutdownResponse,
    load_lora(LoadLoraRequest) -> LoadLoraResponse,
    unload_lora(UnloadLoraRequest) -> UnloadLoraResponse,
    list_loras(ListLorasRequest) -> ListLorasResponse,
    get_kv_event_sources(GetKvEventSourcesRequest) -> GetKvEventSourcesResponse,
    pause_generation(PauseGenerationRequest) -> PauseGenerationResponse,
    resume_generation(ResumeGenerationRequest) -> ResumeGenerationResponse,
    is_paused(IsPausedRequest) -> IsPausedResponse,
    sleep(SleepRequest) -> SleepResponse,
    wake_up(WakeUpRequest) -> WakeUpResponse,
    is_sleeping(IsSleepingRequest) -> IsSleepingResponse,
    init_weight_transfer_engine(InitWeightTransferEngineRequest) -> InitWeightTransferEngineResponse,
    start_weight_update(StartWeightUpdateRequest) -> StartWeightUpdateResponse,
    start_draft_weight_update(StartDraftWeightUpdateRequest) -> StartDraftWeightUpdateResponse,
    update_weights(UpdateWeightsRequest) -> UpdateWeightsResponse,
    finish_weight_update(FinishWeightUpdateRequest) -> FinishWeightUpdateResponse,
    update_weight_version(UpdateWeightVersionRequest) -> UpdateWeightVersionResponse,
    get_weight_version(GetWeightVersionRequest) -> GetWeightVersionResponse,
}

async fn decode_health(State(shared): State<SharedState>) -> StatusCode {
    if shared.lock().unwrap().unhealthy {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}

async fn capabilities(State(shared): State<SharedState>) -> Json<Value> {
    let penalties = shared.lock().unwrap().penalties;
    Json(json!({"capabilities": {"penalties": penalties, "ignore_eos": true}}))
}

async fn pd_cancel(State(shared): State<SharedState>, Json(body): Json<Value>) -> StatusCode {
    let rid = body["rid"].as_str().unwrap_or_default().to_string();
    let mut shared = shared.lock().unwrap();
    let in_flight = shared.decoding.as_deref() == Some(rid.as_str());
    shared.cancelled.push(rid);
    if in_flight {
        shared.decoding = None;
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

fn ndjson<const N: usize>(lines: [Value; N]) -> axum::response::Response {
    lines
        .iter()
        .map(|line| format!("{line}\n"))
        .collect::<String>()
        .into_response()
}

async fn pd_decode(
    State(shared): State<SharedState>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let rid = body["rid"].as_str().unwrap().to_string();
    let held = {
        let shared = shared.lock().unwrap();
        shared
            .hold_decode
            .then(|| (shared.decode_entered.clone(), shared.decode_gate.clone()))
    };
    if let Some((entered, gate)) = held {
        entered.notify_one();
        gate.notified().await;
    }
    let script = {
        let mut shared = shared.lock().unwrap();
        shared.decode_bodies.push(body.clone());
        if shared.landed.as_deref() != Some(rid.as_str()) {
            // The real server first waits `timeout_s` for the KV.
            shared.dropped.insert(rid.clone());
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(json!({"error": "kv_transfer_timeout", "rid": rid})),
            )
                .into_response();
        }
        shared.landed = None;
        shared.decode
    };
    let first = body["first_token_id"].as_u64().unwrap();
    let budget = body["max_tokens"].as_u64().unwrap() as usize;
    let tokens: Vec<u64> = [first, 101, 102, 103].into_iter().take(budget).collect();
    if body["stream"] == json!(false) {
        return Json(json!({"rid": rid, "token_ids": tokens, "phase_ms": {}})).into_response();
    }
    // The real server emits nothing for an end-of-sequence first token.
    if first == u64::from(EOS) {
        return ndjson([json!({"done": true, "n": 0, "finish_reason": "stop"})]);
    }
    let batch = json!({"t": tokens});
    let done = |n: usize| json!({"done": true, "n": n, "finish_reason": "length", "phase_ms": {"decode": 1.0}});
    match script {
        Decode::Complete => ndjson([
            json!({"t": [tokens[0]]}),
            json!({"t": tokens[1..]}),
            done(tokens.len()),
        ]),
        Decode::WrongCount => ndjson([batch, done(tokens.len() + 1)]),
        Decode::ErrorAfterTokens => ndjson([batch, json!({"error": "decode stalled"})]),
        Decode::EndsWithoutDone => ndjson([batch]),
        Decode::Refuse => (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({"error": "not supported", "error_type": "capability_unavailable"})),
        )
            .into_response(),
        Decode::Stall => {
            shared.lock().unwrap().decoding = Some(rid);
            let first_batch =
                futures::stream::iter([Ok::<_, std::io::Error>(format!("{batch}\n"))]);
            Body::from_stream(first_batch.chain(futures::stream::pending())).into_response()
        }
    }
}

struct Fakes {
    shared: SharedState,
    health: tonic_health::server::HealthReporter,
    prefill_entered: Arc<Notify>,
    decode_entered: Arc<Notify>,
    decode_gate: Arc<Notify>,
    grpc: String,
    decode: String,
}

async fn start_fakes() -> Fakes {
    let shared = SharedState::default();
    let prefill_entered = Arc::new(Notify::new());

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let grpc = format!("http://{}", listener.local_addr().unwrap());
    let (health, health_service) = tonic_health::server::health_reporter();
    health
        .set_service_status("vllm.Inference", ServingStatus::Serving)
        .await;
    let vllm = FakeVllm {
        shared: shared.clone(),
        prefill_entered: prefill_entered.clone(),
    };
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(pb::inference_server::InferenceServer::new(vllm.clone()))
            .add_service(pb::control_server::ControlServer::new(vllm))
            .add_service(health_service)
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let decode = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/health", get(decode_health))
        .route("/capabilities", get(capabilities))
        .route("/pd/decode", post(pd_decode))
        .route("/pd/cancel", post(pd_cancel))
        .with_state(shared.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let (decode_entered, decode_gate) = {
        let shared = shared.lock().unwrap();
        (shared.decode_entered.clone(), shared.decode_gate.clone())
    };
    Fakes {
        shared,
        health,
        prefill_entered,
        decode_entered,
        decode_gate,
        grpc,
        decode,
    }
}

async fn bounded<T>(step: impl Future<Output = T>) -> T {
    tokio::time::timeout(STEP, step)
        .await
        .expect("the step finished in time")
}

fn args(fakes: &Fakes, extra: &[&str]) -> Args {
    args_for(&fakes.grpc, &fakes.decode, extra)
}

fn args_for(grpc: &str, decode: &str, extra: &[&str]) -> Args {
    let mut argv = vec![
        "dynamo-tilert-sidecar",
        "--grpc-endpoint",
        grpc,
        "--grpc-connections",
        "1",
        "--grpc-startup-deadline-secs",
        "10",
        "--decode-http-endpoint",
        decode,
        "--decode-kv-host",
        "decode-node",
        "--decode-ctrl-port",
        "6001",
        "--model-path",
        "test-model",
        "--context-length",
        "1000",
        "--kv-transfer-timeout-secs",
        "2",
    ];
    argv.extend_from_slice(extra);
    Args::try_parse_from(argv).unwrap()
}

async fn started_engine(fakes: &Fakes, extra: &[&str]) -> TilertSidecarEngine {
    let (engine, _config) = bounded(TilertSidecarEngine::bootstrap(args(fakes, extra)))
        .await
        .expect("bootstrap");
    bounded(engine.start(0)).await.expect("start");
    engine
}

fn request(max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test-model".to_string())
        .token_ids(vec![10, 11, 12, 13, 14])
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(1.0),
            seed: Some(7),
            ..Default::default()
        })
        .output_options(Default::default())
        .build()
        .unwrap()
}

async fn run(
    engine: &TilertSidecarEngine,
    request: PreprocessedRequest,
    context: Arc<dyn AsyncEngineContext>,
) -> Vec<Result<LLMEngineOutput, DynamoError>> {
    let stream = engine
        .generate(request, GenerateContext::new(context, None))
        .await
        .expect("generate");
    bounded(stream.collect()).await
}

async fn collect(
    engine: &TilertSidecarEngine,
    request: PreprocessedRequest,
) -> Vec<LLMEngineOutput> {
    run(engine, request, mock_context())
        .await
        .into_iter()
        .map(|item| item.expect("stream item"))
        .collect()
}

fn tokens(outputs: &[LLMEngineOutput]) -> Vec<u32> {
    outputs.iter().flat_map(|o| o.token_ids.clone()).collect()
}

#[tokio::test]
async fn request_runs_prefill_then_streams_decode() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;

    let outputs = collect(&engine, request(3)).await;
    assert_eq!(tokens(&outputs), vec![FIRST_TOKEN, 101, 102]);
    let last = outputs.last().unwrap();
    assert_eq!(last.finish_reason, Some(FinishReason::Length));
    let usage = last.completion_usage.as_ref().unwrap();
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (5, 3));
    assert_eq!(
        usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
        Some(2)
    );

    let shared = fakes.shared.lock().unwrap();
    let prefill = &shared.prefill_requests[0];
    assert_eq!(prefill.stopping.as_ref().unwrap().max_new_tokens, 1);
    // The decode stage samples without top-k or top-p, so the prefill must too.
    let sampling = prefill.sampling.as_ref().unwrap();
    assert_eq!((sampling.top_k, sampling.top_p), (Some(0), Some(1.0)));
    let kv = dynamo_sidecar_common::struct_to_json_v14(
        prefill.kv.clone().unwrap().kv_transfer_params.unwrap(),
        "test",
        "kv_transfer_params",
    )
    .unwrap();
    assert_eq!(kv["tilert_host"], "decode-node");
    // A protobuf Struct carries numbers as doubles; the connector casts with int().
    assert_eq!(kv["tilert_ctrl_port"].as_f64(), Some(6001.0));

    let body = &shared.decode_bodies[0];
    // The decode claims the KV under the id the prefill request carried.
    assert_eq!(body["rid"], json!(prefill.request_id));
    assert!(!prefill.request_id.contains('-'));
    assert_eq!(body["first_token_id"], json!(FIRST_TOKEN));
    assert_eq!(body["max_tokens"], json!(3));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["sampling"]["seed"], json!(7));
    assert!(shared.aborted.is_empty() && shared.cancelled.is_empty());
}

#[tokio::test]
async fn requests_are_served_one_at_a_time() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;

    // The decode server would refuse the KV of a request that overlaps
    // another, and that request's decode would time out.
    let (first, second) = tokio::join!(collect(&engine, request(2)), collect(&engine, request(2)));
    for outputs in [first, second] {
        assert_eq!(tokens(&outputs), vec![FIRST_TOKEN, 101]);
    }
}

#[tokio::test]
async fn eos_as_first_token_hands_the_decode_server_the_stop_token() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;
    fakes.shared.lock().unwrap().eos_first = true;

    let outputs = collect(&engine, request(8)).await;
    assert!(tokens(&outputs).is_empty());
    assert_eq!(
        outputs.last().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
    let shared = fakes.shared.lock().unwrap();
    assert_eq!(shared.decode_bodies[0]["first_token_id"], json!(EOS));
}

#[tokio::test]
async fn cancel_during_prefill_aborts_vllm_and_releases_the_kv() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;
    fakes.shared.lock().unwrap().prefill_hangs = true;

    let context = mock_context();
    let cancel = async {
        bounded(fakes.prefill_entered.notified()).await;
        context.stop_generating();
    };
    let (outputs, ()) = tokio::join!(run(&engine, request(8), context.clone()), cancel);
    let last = outputs.last().unwrap().as_ref().unwrap();
    assert_eq!(last.finish_reason, Some(FinishReason::Cancelled));
    fakes.shared.lock().unwrap().prefill_hangs = false;

    // The next request gets the slot only once both engines are clean.
    assert_eq!(
        tokens(&collect(&engine, request(2)).await),
        vec![FIRST_TOKEN, 101]
    );
    let shared = fakes.shared.lock().unwrap();
    let rid = shared.prefill_requests[0].request_id.clone();
    assert_eq!(shared.aborted, vec![rid.clone()]);
    // The KV had already landed; a one-token decode claimed it.
    let release = &shared.decode_bodies[0];
    assert_eq!(release["rid"], json!(rid));
    assert_eq!(release["max_tokens"], json!(1));
}

/// The framework drops a stream without cancelling it when, for example, the
/// client's connection resets.
#[tokio::test]
async fn stream_dropped_during_prefill_still_cleans_up_both_engines() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;
    fakes.shared.lock().unwrap().prefill_hangs = true;

    let mut stream = engine
        .generate(request(8), GenerateContext::new(mock_context(), None))
        .await
        .expect("generate");
    tokio::select! {
        item = stream.next() => panic!("the prefill never answers, got {item:?}"),
        _ = bounded(fakes.prefill_entered.notified()) => {}
    }
    drop(stream);
    fakes.shared.lock().unwrap().prefill_hangs = false;

    assert_eq!(
        tokens(&collect(&engine, request(2)).await),
        vec![FIRST_TOKEN, 101]
    );
    let shared = fakes.shared.lock().unwrap();
    assert_eq!(
        shared.aborted,
        vec![shared.prefill_requests[0].request_id.clone()]
    );
    assert_eq!(shared.decode_bodies[0]["max_tokens"], json!(1));
}

#[tokio::test]
async fn failed_prefill_releases_the_kv_it_pushed() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;
    fakes.shared.lock().unwrap().prefill_fails = true;

    let outputs = run(&engine, request(8), mock_context()).await;
    assert!(outputs.last().unwrap().is_err());
    fakes.shared.lock().unwrap().prefill_fails = false;

    assert_eq!(
        tokens(&collect(&engine, request(2)).await),
        vec![FIRST_TOKEN, 101]
    );
    let shared = fakes.shared.lock().unwrap();
    assert_eq!(shared.decode_bodies[0]["max_tokens"], json!(1));
    assert!(shared.aborted.is_empty());
}

#[tokio::test]
async fn cancel_while_the_decode_is_starting_stops_it_by_id() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;
    {
        let mut shared = fakes.shared.lock().unwrap();
        shared.hold_decode = true;
        shared.decode = Decode::Stall;
    }

    let context = mock_context();
    let cancel = async {
        bounded(fakes.decode_entered.notified()).await;
        context.stop_generating();
        // The decode server claims the KV a moment after the cancel.
        fakes.decode_gate.notify_one();
    };
    let (outputs, ()) = tokio::join!(run(&engine, request(8), context.clone()), cancel);
    let last = outputs.last().unwrap().as_ref().unwrap();
    assert_eq!(last.finish_reason, Some(FinishReason::Cancelled));

    let shared = fakes.shared.lock().unwrap();
    assert_eq!(
        shared.cancelled,
        vec![shared.prefill_requests[0].request_id.clone()]
    );
    assert!(shared.decoding.is_none());
}

#[tokio::test]
async fn cancel_mid_decode_stops_the_decode_server() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;
    fakes.shared.lock().unwrap().decode = Decode::Stall;

    let context = mock_context();
    let mut stream = engine
        .generate(request(8), GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");
    let first = bounded(stream.next()).await.unwrap().unwrap();
    assert_eq!(first.token_ids, vec![FIRST_TOKEN, 101, 102, 103]);
    context.stop_generating();
    let last = bounded(stream.next()).await.unwrap().unwrap();
    assert_eq!(last.finish_reason, Some(FinishReason::Cancelled));
    assert_eq!(last.completion_usage.unwrap().completion_tokens, 4);
    assert!(bounded(stream.next()).await.is_none());

    let shared = fakes.shared.lock().unwrap();
    assert_eq!(
        shared.cancelled,
        vec![shared.prefill_requests[0].request_id.clone()]
    );
    assert!(shared.decoding.is_none());
}

#[tokio::test]
async fn decode_failures_end_the_stream_and_free_the_slot() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;

    for (script, kind) in [
        (Decode::ErrorAfterTokens, BackendError::Unknown),
        (Decode::EndsWithoutDone, BackendError::StreamIncomplete),
        // Every token was delivered, so this must not look retryable.
        (Decode::WrongCount, BackendError::Unknown),
        (Decode::Refuse, BackendError::InvalidArgument),
    ] {
        fakes.shared.lock().unwrap().decode = script;
        let outputs = run(&engine, request(8), mock_context()).await;
        let error = outputs.last().unwrap().as_ref().unwrap_err();
        assert!(is_kind(error, kind), "{script:?}: {error}");

        fakes.shared.lock().unwrap().decode = Decode::Complete;
        assert_eq!(
            tokens(&collect(&engine, request(2)).await),
            vec![FIRST_TOKEN, 101],
            "{script:?}"
        );
    }
}

#[tokio::test]
async fn startup_probe_checks_the_kv_handoff() {
    let fakes = start_fakes().await;
    let probe = ["--probe-token-ids", "1,2,3"];
    started_engine(&fakes, &probe).await;
    {
        let shared = fakes.shared.lock().unwrap();
        let body = &shared.decode_bodies[0];
        assert_eq!(body["rid"], json!(shared.prefill_requests[0].request_id));
        assert_eq!(body["stream"], json!(false));
    }

    fakes.shared.lock().unwrap().no_kv = true;
    let error = bounded(TilertSidecarEngine::bootstrap(args(&fakes, &probe)))
        .await
        .err()
        .expect("a prefill that pushes no KV fails startup");
    assert!(is_kind(&error, BackendError::ConnectionTimeout), "{error}");
    assert!(error.to_string().contains("KV connector"), "{error}");
}

#[tokio::test]
async fn unsupported_requests_are_refused_before_any_engine_call() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;

    let mut with_logprobs = request(4);
    with_logprobs.output_options.logprobs = Some(1);
    let error = engine
        .generate(with_logprobs, GenerateContext::new(mock_context(), None))
        .await
        .err()
        .expect("logprobs are refused");
    assert!(is_kind(&error, BackendError::InvalidArgument), "{error}");
    assert!(fakes.shared.lock().unwrap().prefill_requests.is_empty());
}

#[tokio::test]
async fn penalties_reach_both_engines_when_the_decode_supports_them() {
    let fakes = start_fakes().await;
    fakes.shared.lock().unwrap().penalties = true;
    let engine = started_engine(&fakes, &[]).await;

    let mut with_penalty = request(2);
    with_penalty.sampling_options.repetition_penalty = Some(1.5);
    collect(&engine, with_penalty).await;

    let shared = fakes.shared.lock().unwrap();
    let prefill = shared.prefill_requests[0].decoding.as_ref().unwrap();
    assert_eq!(prefill.repetition_penalty, 1.5);
    assert_eq!(
        shared.decode_bodies[0]["sampling"]["repetition_penalty"],
        json!(1.5)
    );
}

#[tokio::test]
async fn engine_passes_the_shared_conformance_suite() {
    let fakes = start_fakes().await;
    let mut engines = Vec::new();
    for _ in 0..2 {
        let (engine, _config) = bounded(TilertSidecarEngine::bootstrap(args(&fakes, &[])))
            .await
            .expect("bootstrap");
        engines.push(engine);
    }
    bounded(run_conformance(|| {
        engines.pop().expect("the suite builds two engines")
    }))
    .await
    .expect("conformance");
}

#[tokio::test]
async fn engine_lost_follows_the_vllm_health_service() {
    let fakes = start_fakes().await;
    let engine = started_engine(&fakes, &[]).await;

    fakes
        .health
        .set_service_status("", ServingStatus::NotServing)
        .await;
    let lost = bounded(engine.engine_lost()).await;
    assert!(lost.to_string().contains("vLLM prefill"), "{lost}");
}

#[tokio::test]
async fn decode_health_watch_reports_a_server_that_stops_answering() {
    let fakes = start_fakes().await;
    let decode = DecodeClient::new(HttpEndpoint::parse(&fakes.decode, "test").unwrap()).unwrap();
    let watch = || decode.watch_health(Duration::from_millis(5), 2);

    let healthy = tokio::time::timeout(Duration::from_millis(50), watch()).await;
    assert!(healthy.is_err(), "a healthy server is not reported");

    fakes.shared.lock().unwrap().unhealthy = true;
    let why = bounded(watch()).await;
    assert!(why.contains("503"), "{why}");
}

#[tokio::test]
async fn startup_fails_when_an_engine_cannot_be_checked() {
    let fakes = start_fakes().await;

    // vLLM without the gRPC health service: retrying until the startup
    // deadline would only hide the misconfiguration.
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let bare_vllm = format!("http://{}", listener.local_addr().unwrap());
    let vllm = FakeVllm {
        shared: fakes.shared.clone(),
        prefill_entered: fakes.prefill_entered.clone(),
    };
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(pb::inference_server::InferenceServer::new(vllm))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let error = bounded(TilertSidecarEngine::bootstrap(args_for(
        &bare_vllm,
        &fakes.decode,
        &[],
    )))
    .await
    .err()
    .expect("a prefill without the health service fails startup");
    assert!(
        error
            .to_string()
            .contains("unavailable through the standard gRPC health API"),
        "{error}"
    );

    let closed = TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap()
        .local_addr()
        .unwrap();
    let decode =
        DecodeClient::new(HttpEndpoint::parse(&format!("http://{closed}"), "test").unwrap())
            .unwrap();
    let deadline = Instant::now() + Duration::from_millis(100);
    let error = bounded(decode.wait_healthy(deadline, Duration::from_millis(20)))
        .await
        .unwrap_err();
    assert!(is_kind(&error, BackendError::CannotConnect), "{error}");
}

#[tokio::test]
async fn calls_outside_the_engine_lifecycle_are_refused() {
    let fakes = start_fakes().await;
    let (engine, _config) = bounded(TilertSidecarEngine::bootstrap(args(&fakes, &[])))
        .await
        .expect("bootstrap");

    let early = engine
        .generate(request(2), GenerateContext::new(mock_context(), None))
        .await
        .err()
        .expect("generate before start is refused");
    assert!(is_kind(&early, BackendError::EngineShutdown), "{early}");

    bounded(engine.start(0)).await.expect("start");
    let again = bounded(engine.start(0))
        .await
        .expect_err("a second start is refused");
    assert!(is_kind(&again, BackendError::EngineShutdown), "{again}");
}

#[test]
fn plan_refuses_settings_the_decode_stage_cannot_honour() {
    let plain = DecodeCapabilities {
        penalties: false,
        ignore_eos: false,
    };
    type Edit = fn(&mut PreprocessedRequest);
    // One row for each way a request is refused: a sampling option, structured
    // output, and an option the decode server's capabilities rule out.
    let edits: [(&str, Edit); 3] = [
        ("top_p", |r| r.sampling_options.top_p = Some(0.9)),
        ("structured output", |r| {
            r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                regex: Some("a+".to_string()),
                ..Default::default()
            })
        }),
        ("penalties", |r| {
            r.sampling_options.repetition_penalty = Some(1.5)
        }),
    ];
    for (what, edit) in edits {
        let mut refused = request(8);
        edit(&mut refused);
        let error = RequestPlan::new(&refused, 1000, plain).expect_err(what);
        assert!(
            is_kind(&error, BackendError::InvalidArgument),
            "{what}: {error}"
        );
        assert!(error.to_string().contains(what), "{what}: {error}");
    }

    // Clients often spell out the neutral values.
    let mut neutral = request(8);
    neutral.sampling_options.top_p = Some(1.0);
    neutral.sampling_options.top_k = Some(-1);
    neutral.sampling_options.repetition_penalty = Some(1.0);
    neutral.sampling_options.presence_penalty = Some(0.0);
    neutral.sampling_options.frequency_penalty = Some(0.0);
    RequestPlan::new(&neutral, 1000, plain).expect("neutral values are accepted");
}

#[test]
fn plan_caps_max_tokens_at_the_context_window() {
    let capabilities = DecodeCapabilities::default();
    let plan = |request: &PreprocessedRequest, context_length| {
        RequestPlan::new(request, context_length, capabilities)
    };
    assert_eq!(plan(&request(900), 100).unwrap().max_tokens, 95);
    let mut no_budget = request(900);
    no_budget.stop_conditions.max_tokens = None;
    assert_eq!(plan(&no_budget, 100).unwrap().max_tokens, 95);
    assert!(plan(&request(1), 5).is_err());
}

#[test]
fn decode_endpoint_supplies_the_kv_host_and_takes_no_path() {
    let parse = |endpoint: &str, extra: &[&str]| {
        let mut argv = vec![
            "dynamo-tilert-sidecar",
            "--grpc-endpoint",
            "http://127.0.0.1:1",
            "--model-path",
            "test-model",
            "--context-length",
            "8",
            "--decode-http-endpoint",
            endpoint,
        ];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv)
    };
    let args = parse("http://[::1]:5557", &[]).unwrap();
    assert_eq!(args.decode_kv_host(), "::1");
    let args = parse("http://127.0.0.1:5557", &["--decode-kv-host", "10.0.0.2"]).unwrap();
    assert_eq!(args.decode_kv_host(), "10.0.0.2");
    assert!(parse("http://127.0.0.1:5557/prefix", &[]).is_err());
}

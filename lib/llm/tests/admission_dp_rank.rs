// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend admission applies only at a backend's typed request boundary, over
//! the full Rust stack and TCP request plane.
//!
//! A native backend serves its engine through `AdmissionEngine`, which admits
//! each request under the DP rank its decoded body names, read through the
//! request type's `AdmissionRequest` implementation; context metadata never
//! names one. Ordinary endpoints — unary or request-stream, whose request and
//! response types implement no admission trait — are routed and answered by
//! the generic transport alone: they never touch the gate, even while it is
//! saturated and refusing backend work, and a request stream is still answered
//! before it sends a frame.
//!
//! The backend's worker owns its gate, configured through the environment and
//! given its served ranks below; its own `tests/` binary keeps that environment
//! this test's alone.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Error;
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use dynamo_llm::admission_gate::{
    AdmissionEngine, AdmissionRequest, AdmissionResponse, ResponseOutput, WorkerAdmissionGate,
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Endpoint,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, DataStream},
    error::{DynamoError, ErrorType},
    metrics::MetricsHierarchy,
    pipeline::{
        ManyIn, ManyOut, RequestStream, ResponseStream, SingleIn,
        context::Context,
        network::{Ingress, RequestPlanePayloadCodec},
    },
    protocols::maybe_error::MaybeError,
};

use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};

const FAMILY: &str = "dynamo_backend_admission_";
const RECEIVED: &str = "dynamo_backend_admission_request_receive_total";
const ENGINE_REQUESTS: &str = "dynamo_backend_admission_engine_request_count";
const WAITING: &str = "dynamo_backend_admission_engine_request_waiting_count";
const REJECTED: &str = "dynamo_backend_admission_rejection_total";
const UNSPECIFIED: &str = "unspecified";
const TIMEOUT: Duration = Duration::from_secs(10);

/// A backend request, which names a rank as `PreprocessedRequest` names
/// `routing.dp_rank`. A held request's stream stays open, without a response,
/// until its caller goes away.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct Ranked {
    dp_rank: Option<u32>,
    #[serde(default)]
    hold: bool,
}

impl AdmissionRequest for Ranked {
    fn dp_rank(&self) -> Option<u32> {
        self.dp_rank
    }
}

/// A backend response carrying a generated token, and the rank of the request
/// it answers.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct Token {
    dp_rank: Option<u32>,
    #[serde(default)]
    error: Option<DynamoError>,
}

impl AdmissionResponse for Token {
    fn admission_output(&self) -> ResponseOutput {
        ResponseOutput {
            generated_token: true,
            completion: true,
        }
    }
}

/// An ordinary RPC: its body even holds rank-shaped fields, which nothing
/// reads, and neither it nor its reply implements an admission trait.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct Plain {
    value: u32,
    dp_rank: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct Reply {
    value: Option<u32>,
    #[serde(default)]
    error: Option<DynamoError>,
}

macro_rules! maybe_error {
    ($ty:ident) => {
        impl MaybeError for $ty {
            fn from_err(err: impl std::error::Error + 'static) -> Self {
                Self {
                    error: Some(DynamoError::from(
                        Box::new(err) as Box<dyn std::error::Error + 'static>
                    )),
                    ..Default::default()
                }
            }

            fn err(&self) -> Option<DynamoError> {
                self.error.clone()
            }
        }
    };
}

maybe_error!(Token);
maybe_error!(Reply);

fn one<T: dynamo_runtime::engine::Data>(
    item: T,
    context: Arc<dyn AsyncEngineContext>,
) -> ManyOut<T> {
    let stream: DataStream<T> = Box::pin(futures::stream::once(async move { item }));
    ResponseStream::new(stream, context)
}

/// The native backend's engine, counting the requests that reach it.
struct Backend {
    reached: Arc<AtomicUsize>,
}

#[async_trait]
impl AsyncEngine<SingleIn<Ranked>, ManyOut<Token>, Error> for Backend {
    async fn generate(&self, request: SingleIn<Ranked>) -> Result<ManyOut<Token>, Error> {
        self.reached.fetch_add(1, Ordering::SeqCst);
        let context = request.context();
        if request.hold {
            let stopped = Arc::clone(&context);
            let open: DataStream<Token> = Box::pin(
                futures::stream::once(async move { stopped.stopped().await })
                    .filter_map(|()| async { None }),
            );
            return Ok(ResponseStream::new(open, context));
        }
        let dp_rank = request.dp_rank;
        Ok(one(
            Token {
                dp_rank,
                error: None,
            },
            context,
        ))
    }
}

/// The engine behind every ordinary endpoint.
struct Answer;

#[async_trait]
impl AsyncEngine<SingleIn<Plain>, ManyOut<Reply>, Error> for Answer {
    async fn generate(&self, request: SingleIn<Plain>) -> Result<ManyOut<Reply>, Error> {
        let context = request.context();
        let value = Some(request.value);
        Ok(one(Reply { value, error: None }, context))
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<serde_json::Value>, ManyOut<Reply>, Error> for Answer {
    async fn generate(
        &self,
        request: SingleIn<serde_json::Value>,
    ) -> Result<ManyOut<Reply>, Error> {
        let context = request.context();
        let value = request["value"].as_u64().map(|value| value as u32);
        Ok(one(Reply { value, error: None }, context))
    }
}

/// Answers once on arrival, before reading any frame, then answers each frame
/// with its value.
#[async_trait]
impl AsyncEngine<ManyIn<Plain>, ManyOut<Reply>, Error> for Answer {
    async fn generate(&self, input: ManyIn<Plain>) -> Result<ManyOut<Reply>, Error> {
        let context = input.context();
        let (frames, _) = input.into_parts();
        let frames = frames.take().expect("the request stream is taken once");
        let replies =
            futures::stream::once(async { Reply::default() }).chain(frames.map(|frame| Reply {
                value: Some(frame.value),
                error: None,
            }));
        Ok(ResponseStream::new(Box::pin(replies), context))
    }
}

/// Every sample of the gate's family, keyed by metric name and labels.
fn family(drt: &DistributedRuntime) -> BTreeMap<String, f64> {
    drt.metrics()
        .prometheus_expfmt()
        .unwrap()
        .lines()
        .filter(|line| line.starts_with(FAMILY))
        .map(|line| {
            let (series, value) = line.rsplit_once(' ').unwrap();
            (series.to_string(), value.parse().unwrap())
        })
        .collect()
}

/// One metric's samples, by `dp_rank`.
fn by_rank(drt: &DistributedRuntime, metric: &str) -> BTreeMap<String, f64> {
    let mut samples = BTreeMap::new();
    for (series, value) in family(drt) {
        let Some(labels) = series
            .strip_prefix(metric)
            .and_then(|rest| rest.strip_prefix('{'))
        else {
            continue;
        };
        let rank = labels
            .split("dp_rank=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("every series carries dp_rank");
        *samples.entry(rank.to_string()).or_default() += value;
    }
    samples
}

fn total(drt: &DistributedRuntime, metric: &str) -> f64 {
    by_rank(drt, metric).values().sum()
}

/// The one `dp_rank` the gate counted a request under while `send` ran.
async fn admitted_under<F: Future<Output = ()>>(drt: &DistributedRuntime, send: F) -> String {
    let before = by_rank(drt, RECEIVED);
    tokio::time::timeout(TIMEOUT, send)
        .await
        .expect("the request must complete");
    let after = by_rank(drt, RECEIVED);
    let counted: Vec<(&String, f64)> = after
        .iter()
        .map(|(rank, count)| (rank, count - before.get(rank).copied().unwrap_or(0.0)))
        .filter(|(_, delta)| *delta != 0.0)
        .collect();
    assert_eq!(counted.len(), 1, "one series moved: {counted:?}");
    assert_eq!(counted[0].1, 1.0, "by one request: {counted:?}");
    counted[0].0.clone()
}

/// Whether a worker's overload refusal appears anywhere in an error chain.
/// Across the request plane it keeps its semantic class and the stable
/// worker-overload reason.
fn is_worker_overloaded(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<DynamoError>()
            && error.class() == ErrorType::CapacityExhausted
            && error.reason().as_str() == "capacity.worker_overloaded"
        {
            return true;
        }
        current = error.source();
    }
    false
}

/// Wait until the gate's occupancy totals read `(engine requests, waiting)`.
async fn occupancy_reaches(drt: &DistributedRuntime, expected: (f64, f64)) {
    tokio::time::timeout(TIMEOUT, async {
        while (total(drt, ENGINE_REQUESTS), total(drt, WAITING)) != expected {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "occupancy stayed at {:?}, never {expected:?}",
            (total(drt, ENGINE_REQUESTS), total(drt, WAITING))
        )
    });
}

fn endpoint(drt: &DistributedRuntime, name: &str) -> Endpoint {
    drt.namespace("test_admission_dp_rank".to_string())
        .unwrap()
        .component("backend".to_string())
        .unwrap()
        .endpoint(name.to_string())
}

async fn serve<H>(endpoint: Endpoint, handler: Arc<H>) -> dynamo_runtime::component::Client
where
    H: dynamo_runtime::pipeline::network::PushWorkHandler + 'static,
{
    let server = endpoint.clone();
    tokio::spawn(async move {
        let _ = server.endpoint_builder().handler(handler).start().await;
    });
    let client = endpoint.client().await.unwrap();
    client.wait_for_instances().await.unwrap();
    client
}

#[tokio::test]
async fn admission_applies_only_at_the_typed_backend_boundary() {
    // A standalone process: clear any inherited codec override before the
    // request-plane codec cache is initialized, and configure the gate before
    // it is created. Per rank: one engine request, one waiting, no queue.
    unsafe {
        std::env::remove_var("DYN_REQUEST_PLANE_CODEC");
        std::env::set_var("DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT", "1");
        std::env::set_var("DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT", "1");
        std::env::set_var("DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT", "0");
        for alias in [
            "DYN_ENGINE_REQUEST_LIMIT",
            "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT",
            "DYN_DYNAMO_REQUEST_QUEUE_LIMIT",
        ] {
            std::env::remove_var(alias);
        }
    }
    assert_eq!(
        RequestPlanePayloadCodec::configured(),
        RequestPlanePayloadCodec::Msgpack
    );

    let rt = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
        .await
        .unwrap();

    // The backend opts in as a worker serving it does: its gate is created for
    // the component it serves, which exposes the gate's family, and its typed
    // wrapper admits through that gate.
    let reached = Arc::new(AtomicUsize::new(0));
    let backend_endpoint = endpoint(&drt, "generate");
    let gate = WorkerAdmissionGate::default().get_or_create(backend_endpoint.component());
    // This worker serves global ranks 4 and 5: two engine requests, and two
    // waiting, worker-wide.
    gate.record_engine_capacity(None, 4, 2);
    let backend = serve(
        backend_endpoint,
        Ingress::for_engine(AdmissionEngine::<Ranked, Token>::new(
            gate,
            Arc::new(Backend {
                reached: Arc::clone(&reached),
            }),
        ))
        .unwrap(),
    )
    .await;
    let backend = PushRouter::<Ranked, Token>::from_client(backend, RouterMode::RoundRobin)
        .await
        .unwrap();

    let plain = serve(
        endpoint(&drt, "plain"),
        Ingress::<SingleIn<Plain>, ManyOut<Reply>>::for_engine(Arc::new(Answer)).unwrap(),
    )
    .await;
    let plain = PushRouter::<Plain, Reply>::from_client(plain, RouterMode::RoundRobin)
        .await
        .unwrap();
    let generic = serve(
        endpoint(&drt, "generic"),
        Ingress::<SingleIn<serde_json::Value>, ManyOut<Reply>>::for_engine(Arc::new(Answer))
            .unwrap(),
    )
    .await;
    let generic =
        PushRouter::<serde_json::Value, Reply>::from_client(generic, RouterMode::RoundRobin)
            .await
            .unwrap();
    let stream = serve(
        endpoint(&drt, "stream"),
        Ingress::<ManyIn<Plain>, ManyOut<Reply>>::for_engine(Arc::new(Answer)).unwrap(),
    )
    .await;
    let stream = PushRouter::<Plain, Reply>::from_client(stream, RouterMode::RoundRobin)
        .await
        .unwrap();

    // Ranked admission through the typed wrapper: under the rank the body
    // names, when the worker serves it.
    let unary = |request: Context<Ranked>| {
        let backend = &backend;
        async move {
            let expected = request.dp_rank;
            let tokens: Vec<Token> = backend.generate(request).await.unwrap().collect().await;
            assert_eq!(tokens.len(), 1);
            assert_eq!(tokens[0].dp_rank, expected, "the engine gets the same body");
        }
    };
    let ranked = |dp_rank| {
        Context::new(Ranked {
            dp_rank,
            hold: false,
        })
    };
    assert_eq!(admitted_under(&drt, unary(ranked(Some(5)))).await, "5");
    assert_eq!(admitted_under(&drt, unary(ranked(Some(4)))).await, "4");
    // No rank, or one the worker does not serve.
    assert_eq!(admitted_under(&drt, unary(ranked(None))).await, UNSPECIFIED);
    assert_eq!(
        admitted_under(&drt, unary(ranked(Some(6)))).await,
        UNSPECIFIED
    );
    // Context metadata never names the rank: only the body does.
    let mut request = ranked(None);
    request.insert_metadata("dp_rank", "5");
    request.insert_metadata("dynamo.dp_rank", "5");
    assert_eq!(admitted_under(&drt, unary(request)).await, UNSPECIFIED);
    let mut request = ranked(Some(4));
    request.insert_metadata("dp_rank", "5");
    assert_eq!(admitted_under(&drt, unary(request)).await, "4");
    assert_eq!(reached.load(Ordering::SeqCst), 6);
    occupancy_reaches(&drt, (0.0, 0.0)).await;

    // Saturate the backend: one held request on each served rank fills the
    // worker's two engine requests.
    let hold = |dp_rank| {
        Context::new(Ranked {
            dp_rank: Some(dp_rank),
            hold: true,
        })
    };
    let held = tokio::time::timeout(TIMEOUT, async {
        [
            backend.generate(hold(4)).await.unwrap(),
            backend.generate(hold(5)).await.unwrap(),
        ]
    })
    .await
    .expect("both held requests are admitted");
    occupancy_reaches(&drt, (2.0, 2.0)).await;

    // Further backend work is refused before its engine, as an overload.
    let rejected = total(&drt, REJECTED);
    let refused = tokio::time::timeout(TIMEOUT, backend.generate(ranked(None)))
        .await
        .expect("a saturated backend refuses promptly");
    let error = match refused {
        Ok(_) => panic!("a saturated backend must refuse the request"),
        Err(error) => error,
    };
    assert!(is_worker_overloaded(error.as_ref()), "{error:?}");
    assert_eq!(total(&drt, REJECTED), rejected + 1.0);
    assert_eq!(
        reached.load(Ordering::SeqCst),
        8,
        "the refused request never reached the engine"
    );

    // Ordinary RPCs carry on untouched by the saturated gate: none needs an
    // admission trait, and not one sample of the gate's family moves.
    let before = family(&drt);
    let replies: Vec<Reply> = tokio::time::timeout(TIMEOUT, async {
        plain
            .generate(Context::new(Plain {
                value: 7,
                dp_rank: Some(5),
            }))
            .await
            .unwrap()
            .collect()
            .await
    })
    .await
    .unwrap();
    assert_eq!(
        replies.iter().map(|reply| reply.value).collect::<Vec<_>>(),
        [Some(7)]
    );
    let body = serde_json::json!({"value": 8, "dp_rank": 5, "routing": {"dp_rank": 5}});
    let replies: Vec<Reply> = tokio::time::timeout(TIMEOUT, async {
        generic
            .generate(Context::new(body))
            .await
            .unwrap()
            .collect()
            .await
    })
    .await
    .unwrap();
    assert_eq!(
        replies.iter().map(|reply| reply.value).collect::<Vec<_>>(),
        [Some(8)]
    );
    // A request stream is answered before it sends a frame, and only then is
    // its first frame sent.
    tokio::time::timeout(TIMEOUT, async {
        let (frames, frame_rx) = tokio::sync::mpsc::channel::<Plain>(1);
        let input: ManyIn<Plain> = Context::new(RequestStream::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(frame_rx),
        )));
        let mut replies = stream.generate(input).await.unwrap();
        let first = replies.next().await.expect("answered before any frame");
        assert_eq!(first.value, None);
        assert!(first.error.is_none(), "{first:?}");
        frames
            .send(Plain {
                value: 9,
                dp_rank: Some(4),
            })
            .await
            .unwrap();
        drop(frames);
        let rest: Vec<Reply> = replies.collect().await;
        assert_eq!(
            rest.iter().map(|reply| reply.value).collect::<Vec<_>>(),
            [Some(9)],
            "the frame still reaches the engine: {rest:?}"
        );
    })
    .await
    .unwrap();
    assert_eq!(family(&drt), before, "ordinary RPCs never touch the gate");

    // The held requests' callers leave, and the gate drains.
    for stream in held {
        stream.context().stop_generating();
        drop(stream);
    }
    occupancy_reaches(&drt, (0.0, 0.0)).await;

    rt.shutdown();
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A DP rank's refusal stays that rank's, over the native token path a mocker
//! worker serves (`entrypoint::input::endpoint::run`), the TCP request plane
//! and one `PushRouter` and its client throughout.
//!
//! One worker serves global ranks 4 and 5, with one engine request and one
//! queued request per rank. With rank 4 full and rank 5 idle, the worker
//! refuses another rank 4 request for rank 4 alone, and the router that carried
//! the refusal sends a rank 5 request straight on: its client never marks the
//! worker overloaded, so the request reaches the engine without waiting out an
//! overload lease, a retry or a new client. Once nothing more can enter or
//! queue anywhere in the worker, a refusal is the whole worker's, and the same
//! client does mark it: the lease this harness would otherwise have caught.
//!
//! The TCP request-plane server is process-global and runs on the runtime that
//! first starts it, so the scenario runs under one test, and its own `tests/`
//! binary keeps the gate settings below its own.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Error;
use async_trait::async_trait;
use futures::StreamExt;
use tokio::sync::mpsc;

use dynamo_llm::admission_gate::WorkerAdmissionGate;
use dynamo_llm::entrypoint::EngineConfig;
use dynamo_llm::entrypoint::input::endpoint;
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Client,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider, DataStream},
    error::{DynamoError, ErrorType},
    metrics::MetricsHierarchy,
    pipeline::{
        ManyOut, ResponseStream, SingleIn,
        context::Context,
        network::egress::push_router::{PushRouter, RouterMode},
    },
    protocols::annotated::Annotated,
};

const NAMESPACE: &str = "test_admission_rank_refusal";
const COMPONENT: &str = "backend";
const RECEIVED: &str = "dynamo_backend_admission_request_receive_total";
const ADMITTED: &str = "dynamo_backend_admission_request_admit_total";
const REJECTED: &str = "dynamo_backend_admission_rejection_total";
const ENGINE_REQUESTS: &str = "dynamo_backend_admission_engine_request_count";
const QUEUED: &str = "dynamo_backend_admission_request_queue_count";
const TIMEOUT: Duration = Duration::from_secs(10);

type Output = Annotated<LLMEngineOutput>;
type Router = PushRouter<PreprocessedRequest, Output>;

/// A token engine that reports the rank of each request reaching it, answers
/// with one generated token, which ends the request's wait, then holds the
/// stream open until its caller goes away.
struct Holding {
    reached: mpsc::UnboundedSender<Option<u32>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Output>, Error> for Holding {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Output>, Error> {
        let context = request.context();
        let dp_rank = request.routing.as_ref().and_then(|routing| routing.dp_rank);
        let _ = self.reached.send(dp_rank);
        let token = Annotated::from_data(LLMEngineOutput {
            token_ids: vec![1],
            ..Default::default()
        });
        let stopped = Arc::clone(&context);
        let stream: DataStream<Output> = Box::pin(
            futures::stream::once(async move { token }).chain(
                futures::stream::once(async move { stopped.stopped().await })
                    .filter_map(|()| async { None }),
            ),
        );
        Ok(ResponseStream::new(stream, context))
    }
}

/// A request routed to `dp_rank`, as routing names it in `routing.dp_rank`.
fn request(dp_rank: Option<u32>) -> Context<PreprocessedRequest> {
    Context::new(
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "token_ids": [1],
            "routing": {"dp_rank": dp_rank},
        }))
        .unwrap(),
    )
}

/// Send a request to `dp_rank` that the worker must admit; return its stream
/// once the engine has it and its token has arrived.
async fn hold(
    router: &Router,
    reached: &mut mpsc::UnboundedReceiver<Option<u32>>,
    dp_rank: u32,
) -> ManyOut<Output> {
    let mut stream = tokio::time::timeout(TIMEOUT, router.generate(request(Some(dp_rank))))
        .await
        .expect("an admitted request answers")
        .expect("the request is admitted");
    let arrival = tokio::time::timeout(TIMEOUT, reached.recv())
        .await
        .expect("the request reaches the engine");
    assert_eq!(arrival, Some(Some(dp_rank)), "on its own rank");
    tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("the token arrives")
        .expect("the stream yields its token");
    stream
}

/// Send a request to the worker in the background, where it waits in a gate
/// queue until capacity frees.
fn queue(
    router: &Router,
    dp_rank: u32,
) -> tokio::task::JoinHandle<anyhow::Result<ManyOut<Output>>> {
    let router = router.clone();
    tokio::spawn(async move { router.generate(request(Some(dp_rank))).await })
}

/// Send a request the worker's gate must refuse, and return its error.
async fn refused(router: &Router, dp_rank: Option<u32>) -> Error {
    match tokio::time::timeout(TIMEOUT, router.generate(request(dp_rank)))
        .await
        .expect("a full gate refuses promptly")
    {
        Ok(_) => panic!("the gate must refuse the request"),
        Err(error) => error,
    }
}

/// The DP rank of the worker overload in an error chain: `Some(None)` for an
/// overload of the whole worker, and `None` when the chain holds no worker
/// overload.
fn overloaded_rank(error: &Error) -> Option<Option<u32>> {
    error
        .chain()
        .filter_map(|error| error.downcast_ref::<DynamoError>())
        .find(|error| error.reason().as_str() == "capacity.worker_overloaded")
        .map(|overload| match overload.error_type() {
            ErrorType::WorkerRankOverloaded(dp_rank) => Some(dp_rank),
            _ => None,
        })
}

/// Whether `client` has the worker out of routing for overload.
fn overloaded(client: &Client, instance_id: u64) -> bool {
    client
        .overloaded_instance_ids()
        .is_some_and(|overloaded| overloaded.contains(&instance_id))
}

/// One metric of the gate's family, summed over its series matching `labels`.
fn sample(drt: &DistributedRuntime, metric: &str, labels: &[(&str, &str)]) -> f64 {
    drt.metrics()
        .prometheus_expfmt()
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix(metric)?.strip_prefix('{'))
        .filter(|rest| {
            labels
                .iter()
                .all(|(key, value)| rest.contains(&format!("{key}=\"{value}\"")))
        })
        .map(|rest| rest.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
        .sum()
}

/// Wait until `dp_rank`'s gate queue holds `expected` requests.
async fn queued_at(drt: &DistributedRuntime, dp_rank: &str, expected: f64) {
    tokio::time::timeout(TIMEOUT, async {
        while sample(drt, QUEUED, &[("dp_rank", dp_rank)]) != expected {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("rank {dp_rank}'s queue never held {expected}"));
}

/// Let a held request's caller go away.
fn release(stream: ManyOut<Output>) {
    stream.context().stop_generating();
    drop(stream);
}

#[tokio::test]
async fn a_full_rank_refuses_for_itself_and_the_same_router_reaches_its_sibling_at_once() {
    // Per rank: one engine request, one waiting, one queued.
    unsafe {
        std::env::set_var("DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT", "1");
        std::env::set_var("DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT", "1");
        std::env::set_var("DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT", "1");
        for alias in [
            "DYN_ENGINE_REQUEST_LIMIT",
            "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT",
            "DYN_DYNAMO_REQUEST_QUEUE_LIMIT",
        ] {
            std::env::remove_var(alias);
        }
    }
    let rt = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
        .await
        .unwrap();

    // The worker's card reports global ranks 4 and 5, before the endpoint is
    // discoverable.
    let runtime_config = ModelRuntimeConfig {
        max_num_seqs: Some(1),
        data_parallel_start_rank: 4,
        data_parallel_size: 2,
        ..Default::default()
    };
    let model = LocalModelBuilder::default()
        .model_name(Some("rank-refusal".to_string()))
        .runtime_config(runtime_config)
        .build()
        .await
        .unwrap();
    let (reached_tx, mut reached) = mpsc::unbounded_channel();
    let config = EngineConfig::InProcessTokens {
        engine: Arc::new(Holding {
            reached: reached_tx,
        }),
        model: Box::new(model),
        is_prefill: false,
        is_decode: false,
        admission_gate: WorkerAdmissionGate::default(),
    };
    tokio::spawn(endpoint::run(
        drt.clone(),
        format!("dyn://{NAMESPACE}.{COMPONENT}.generate"),
        config,
    ));
    let client = drt
        .namespace(NAMESPACE.to_string())
        .unwrap()
        .component(COMPONENT.to_string())
        .unwrap()
        .endpoint("generate".to_string())
        .client()
        .await
        .unwrap();
    let instance_id = tokio::time::timeout(TIMEOUT, client.wait_for_instances())
        .await
        .expect("the endpoint is discovered")
        .unwrap()[0]
        .id();
    let router = Router::from_client(client.clone(), RouterMode::RoundRobin)
        .await
        .unwrap();

    // Rank 4 fills its engine request and its queue place, while rank 5 is
    // idle and the worker has room for it.
    let held_four = hold(&router, &mut reached, 4).await;
    let queued_four = queue(&router, 4);
    queued_at(&drt, "4", 1.0).await;

    // Rank 4 refuses another request, before its engine, for rank 4 alone...
    let error = refused(&router, Some(4)).await;
    // ...and the client that carried the refusal marks nothing. Nothing is
    // awaited between the refusal's return and these checks, so a lease it
    // set could not have lapsed in between.
    assert!(
        !overloaded(&client, instance_id),
        "the worker is not marked"
    );
    assert!(
        client.instance_ids_avail().contains(&instance_id),
        "nor down"
    );
    assert_eq!(overloaded_rank(&error), Some(Some(4)), "{error:?}");
    assert!(reached.try_recv().is_err(), "it never reached the engine");
    assert_eq!(sample(&drt, REJECTED, &[("dp_rank", "4")]), 1.0);

    // The same router sends rank 5's request straight on: it reaches the
    // engine and is admitted while rank 4 stays full.
    let held_five = hold(&router, &mut reached, 5).await;
    let admitted = [("dp_rank", "5"), ("source", "direct")];
    assert_eq!(sample(&drt, ADMITTED, &admitted), 1.0);
    assert_eq!(sample(&drt, ENGINE_REQUESTS, &[("dp_rank", "4")]), 1.0);
    assert_eq!(sample(&drt, QUEUED, &[("dp_rank", "4")]), 1.0);

    // With rank 5's queue place taken too, nothing more can enter or queue
    // anywhere in the worker: a refusal now is the whole worker's, and the
    // same client marks it, so its next request never reaches the worker.
    let queued_five = queue(&router, 5);
    queued_at(&drt, "5", 1.0).await;
    let error = refused(&router, None).await;
    assert!(overloaded(&client, instance_id), "the worker is marked");
    assert_eq!(overloaded_rank(&error), Some(None), "{error:?}");
    let received = sample(&drt, RECEIVED, &[]);
    refused(&router, Some(5)).await;
    assert_eq!(
        sample(&drt, RECEIVED, &[]),
        received,
        "held back at the router"
    );
    assert!(
        client.instance_ids_avail().contains(&instance_id),
        "never down"
    );

    // The held requests' callers leave, the queued ones are admitted in turn,
    // and the gate drains.
    release(held_four);
    release(held_five);
    for queued in [queued_four, queued_five] {
        let stream = tokio::time::timeout(TIMEOUT, queued)
            .await
            .expect("a queued request is admitted once capacity frees")
            .unwrap()
            .expect("the queued request is admitted");
        release(stream);
    }
    tokio::time::timeout(TIMEOUT, async {
        while sample(&drt, ENGINE_REQUESTS, &[]) != 0.0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the gate drains");

    rt.shutdown();
}

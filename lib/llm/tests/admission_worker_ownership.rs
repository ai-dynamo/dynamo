// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend admission gates belong to workers, over the native token path a
//! mocker worker serves: `entrypoint::input::endpoint::run` with an
//! `EngineConfig::InProcessTokens`, the TCP request plane and `PushRouter`.
//!
//! Two workers in one process, each its own `DistributedRuntime` and config,
//! own independent gates: each is sized by its own model card, fills and
//! refuses on its own, and exposes its family on its own component's registry,
//! labelled with its own `worker_id`. A config cloned to serve a sibling
//! endpoint is still one worker, so both endpoints admit through its one gate.
//! A later capacity or rank report reaches only its own worker's gate, and a
//! LoRA adapter's card reports none.
//!
//! The TCP request-plane server is process-global and runs on the runtime that
//! first starts it, so every scenario runs under this one test, and its own
//! `tests/` binary keeps the gate settings below its own.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Error;
use async_trait::async_trait;
use futures::StreamExt;
use tokio::sync::mpsc;

use dynamo_llm::admission_gate::WorkerAdmissionGate;
use dynamo_llm::entrypoint::EngineConfig;
use dynamo_llm::entrypoint::input::endpoint;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::model_card::{LoraInfo, ModelDeploymentCard};
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Component,
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

const NAMESPACE: &str = "test_admission_worker_ownership";
const COMPONENT: &str = "backend";
const FAMILY: &str = "dynamo_backend_admission_";
const RECEIVED: &str = "dynamo_backend_admission_request_receive_total";
const ADMITTED: &str = "dynamo_backend_admission_request_admit_total";
const REJECTED: &str = "dynamo_backend_admission_rejection_total";
const ENGINE_REQUESTS: &str = "dynamo_backend_admission_engine_request_count";
const WAITING: &str = "dynamo_backend_admission_engine_request_waiting_count";
const TIMEOUT: Duration = Duration::from_secs(10);

type Output = Annotated<LLMEngineOutput>;
type Router = PushRouter<PreprocessedRequest, Output>;

/// A token engine that tells the test each request reached it, answers with
/// one generated token, which ends the request's wait, then holds the stream
/// open until its caller goes away.
struct Holding {
    reached: mpsc::UnboundedSender<()>,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Output>, Error> for Holding {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Output>, Error> {
        let context = request.context();
        let _ = self.reached.send(());
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

/// One worker: its runtime, its config's admission gate handle, and the
/// arrivals its engine reports.
struct Worker {
    drt: DistributedRuntime,
    admission_gate: WorkerAdmissionGate,
    config: EngineConfig,
    reached: mpsc::UnboundedReceiver<()>,
}

/// A model whose card reports `max_num_seqs` per rank, for the DP ranks
/// `start_rank..start_rank + rank_count`.
async fn model(name: &str, max_num_seqs: u64, start_rank: u32, rank_count: u32) -> Box<LocalModel> {
    let runtime_config = ModelRuntimeConfig {
        max_num_seqs: Some(max_num_seqs),
        data_parallel_start_rank: start_rank,
        data_parallel_size: rank_count,
        ..Default::default()
    };
    Box::new(
        LocalModelBuilder::default()
            .model_name(Some(name.to_string()))
            .runtime_config(runtime_config)
            .build()
            .await
            .unwrap(),
    )
}

impl Worker {
    /// A worker serving DP rank 0, whose model card reports `max_num_seqs`, on
    /// a runtime of its own.
    async fn new(rt: &Runtime, name: &str, max_num_seqs: u64) -> Self {
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let (reached_tx, reached) = mpsc::unbounded_channel();
        let admission_gate = WorkerAdmissionGate::default();
        let config = EngineConfig::InProcessTokens {
            engine: Arc::new(Holding {
                reached: reached_tx,
            }),
            model: model(name, max_num_seqs, 0, 1).await,
            is_prefill: false,
            is_decode: false,
            admission_gate: admission_gate.clone(),
        };
        Self {
            drt,
            admission_gate,
            config,
            reached,
        }
    }

    /// This worker's engine and gate under a newly registered model card: the
    /// same worker, reporting new capacity and ranks.
    fn reregistered(&self, model: Box<LocalModel>) -> EngineConfig {
        let EngineConfig::InProcessTokens { engine, .. } = &self.config else {
            unreachable!("a worker serves a token engine")
        };
        EngineConfig::InProcessTokens {
            engine: Arc::clone(engine),
            model,
            is_prefill: false,
            is_decode: false,
            admission_gate: self.admission_gate.clone(),
        }
    }

    fn component(&self) -> Component {
        self.drt
            .namespace(NAMESPACE.to_string())
            .unwrap()
            .component(COMPONENT.to_string())
            .unwrap()
    }

    /// Serve a clone of this worker's config on endpoint `name`, as the mocker
    /// does, and return a router that has found it.
    async fn serve(&self, name: &str) -> Router {
        self.serve_config(name, self.config.clone()).await
    }

    /// Serve `config` on endpoint `name` of this worker's runtime. Its model
    /// card is registered, and reported to its gate, before the endpoint is
    /// discoverable, so the returned router finds it sized.
    async fn serve_config(&self, name: &str, config: EngineConfig) -> Router {
        let path = format!("dyn://{NAMESPACE}.{COMPONENT}.{name}");
        tokio::spawn(endpoint::run(self.drt.clone(), path, config));
        self.router(name).await
    }

    /// The `dp_rank` of every series of this worker's family.
    fn ranks(&self) -> BTreeSet<String> {
        self.family()
            .keys()
            .map(|(_, labels)| labels["dp_rank"].clone())
            .collect()
    }

    /// A fresh router to endpoint `name`, holding no overload mark from any
    /// earlier refusal.
    async fn router(&self, name: &str) -> Router {
        let client = self
            .component()
            .endpoint(name.to_string())
            .client()
            .await
            .unwrap();
        tokio::time::timeout(TIMEOUT, client.wait_for_instances())
            .await
            .expect("the endpoint is discovered")
            .unwrap();
        Router::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap()
    }

    /// Wait for the engine to report one more arrival.
    async fn reached(&mut self) {
        tokio::time::timeout(TIMEOUT, self.reached.recv())
            .await
            .expect("the request reaches the engine")
            .unwrap();
    }

    /// No request reached the engine beyond those already awaited.
    fn reached_nothing_more(&mut self) {
        assert!(
            self.reached.try_recv().is_err(),
            "a refused request must never reach the engine"
        );
    }

    /// Every sample of the gate's family this worker's runtime exposes, keyed
    /// by metric name and labels.
    fn family(&self) -> BTreeMap<(String, BTreeMap<String, String>), f64> {
        self.drt
            .metrics()
            .prometheus_expfmt()
            .unwrap()
            .lines()
            .filter(|line| line.starts_with(FAMILY))
            .map(|line| {
                let (series, value) = line.rsplit_once(' ').unwrap();
                let (name, labels) = series.split_once('{').unwrap();
                let labels = labels
                    .trim_end_matches('}')
                    .split(',')
                    .map(|pair| {
                        let (key, value) = pair.split_once('=').unwrap();
                        (key.to_string(), value.trim_matches('"').to_string())
                    })
                    .collect();
                ((name.to_string(), labels), value.parse().unwrap())
            })
            .collect()
    }

    fn total(&self, metric: &str) -> f64 {
        self.family()
            .iter()
            .filter(|((name, _), _)| name == metric)
            .map(|(_, value)| value)
            .sum()
    }

    /// Wait until this worker's gate reads `(engine requests, waiting)`.
    async fn occupancy_reaches(&self, expected: (f64, f64)) {
        let occupancy = || (self.total(ENGINE_REQUESTS), self.total(WAITING));
        tokio::time::timeout(TIMEOUT, async {
            while occupancy() != expected {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("occupancy stayed at {:?}, never {expected:?}", occupancy()));
    }
}

fn request() -> Context<PreprocessedRequest> {
    Context::new(
        serde_json::from_value(serde_json::json!({"model": "m", "token_ids": [1]})).unwrap(),
    )
}

/// Send a request the worker's gate must admit; return its stream once the
/// engine has it and its token, which ends its wait, has arrived.
async fn hold(router: &Router, worker: &mut Worker) -> ManyOut<Output> {
    let mut stream = tokio::time::timeout(TIMEOUT, router.generate(request()))
        .await
        .expect("an admitted request answers")
        .expect("the request is admitted");
    worker.reached().await;
    let token = tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("the token arrives")
        .expect("the stream yields its token");
    assert_eq!(token.data.map(|output| output.token_ids), Some(vec![1]));
    stream
}

/// Send a request the worker's gate must refuse, before its engine, as an
/// overload.
async fn refused(router: &Router, worker: &mut Worker) {
    let refused = tokio::time::timeout(TIMEOUT, router.generate(request()))
        .await
        .expect("a full gate refuses promptly");
    let error = match refused {
        Ok(_) => panic!("a full gate must refuse the request"),
        Err(error) => error,
    };
    assert!(is_worker_overloaded(error.as_ref()), "{error:?}");
    worker.reached_nothing_more();
}

/// The gate families `component`'s own registry holds, children aside.
fn admission_families(component: &Component) -> usize {
    component
        .get_metrics_registry()
        .get_prometheus_registry()
        .gather()
        .iter()
        .filter(|family| family.name().starts_with(FAMILY))
        .count()
}

/// Let a held request's caller go away.
fn release(stream: ManyOut<Output>) {
    stream.context().stop_generating();
    drop(stream);
}

/// Whether a worker's overload refusal appears anywhere in an error chain.
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

#[tokio::test]
async fn each_worker_admits_through_its_own_gate_and_its_endpoints_share_it() {
    // No queue, so a full gate refuses at once; the engine limits are left to
    // each worker's own card.
    unsafe {
        std::env::set_var("DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT", "0");
        for name in [
            "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT",
            "DYN_ENGINE_REQUEST_LIMIT",
            "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT",
            "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT",
            "DYN_DYNAMO_REQUEST_QUEUE_LIMIT",
        ] {
            std::env::remove_var(name);
        }
    }
    let rt = Runtime::from_current().unwrap();
    // Worker A reports one sequence, worker B two.
    let mut a = Worker::new(&rt, "worker-a", 1).await;
    let mut b = Worker::new(&rt, "worker-b", 2).await;
    let to_a = a.serve("generate").await;
    let to_b = b.serve("generate").await;

    // Each worker's gate exists once it serves: its own, never the other's,
    // and the same one through every clone of its handle.
    let (component_a, component_b) = (a.component(), b.component());
    let gate_a = a.admission_gate.get_or_create(&component_a);
    assert!(Arc::ptr_eq(
        &gate_a,
        &a.admission_gate.clone().get_or_create(&component_a)
    ));
    assert!(!Arc::ptr_eq(
        &gate_a,
        &b.admission_gate.get_or_create(&component_b)
    ));

    // A gate registers its family once, on the registry of the component it
    // is created for: reached again through another component object of the
    // worker, as each `endpoint::run` makes its own, it registers nothing more.
    // On a runtime of its own, so A's and B's families stay theirs alone.
    let c = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let component = || {
        c.namespace(NAMESPACE.to_string())
            .unwrap()
            .component(COMPONENT.to_string())
            .unwrap()
    };
    let (first, second) = (component(), component());
    let handle = WorkerAdmissionGate::default();
    let gate_c = handle.get_or_create(&first);
    assert!(Arc::ptr_eq(&gate_c, &handle.get_or_create(&second)));
    assert_eq!(admission_families(&first), 7);
    assert_eq!(admission_families(&second), 0);
    // Another gate created for that same component object finds the family
    // already registered there: it is still created, with its collectors kept
    // unregistered, and adds no family.
    let duplicate_gate = WorkerAdmissionGate::default().get_or_create(&first);
    assert!(!Arc::ptr_eq(&gate_c, &duplicate_gate));
    assert_eq!(admission_families(&first), 7);

    // Each runtime exposes only its own worker's family, labelled with the
    // component the worker serves and its own `worker_id`.
    for worker in [&a, &b] {
        let owner = BTreeMap::from([
            ("dynamo_component".to_string(), COMPONENT.to_string()),
            ("dynamo_namespace".to_string(), NAMESPACE.to_string()),
            (
                "worker_id".to_string(),
                format!("{:x}", worker.drt.connection_id()),
            ),
        ]);
        for (name, labels) in worker.family().keys() {
            let identity: BTreeMap<String, String> = labels
                .iter()
                .filter(|(key, _)| owner.contains_key(*key))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            assert_eq!(identity, owner, "{name} {labels:?}");
        }
    }
    assert_ne!(a.drt.connection_id(), b.drt.connection_id());

    // Each card sized its own worker's gate: A admits one request and B two,
    // each refusing the next on its own, whatever the other holds.
    let held_a = hold(&to_a, &mut a).await;
    a.occupancy_reaches((1.0, 0.0)).await;
    let held_b = [hold(&to_b, &mut b).await, hold(&to_b, &mut b).await];
    b.occupancy_reaches((2.0, 0.0)).await;
    refused(&to_a, &mut a).await;
    refused(&to_b, &mut b).await;
    assert_eq!(
        [a.total(RECEIVED), a.total(ADMITTED), a.total(REJECTED)],
        [2.0, 1.0, 1.0]
    );
    assert_eq!(
        [b.total(RECEIVED), b.total(ADMITTED), b.total(REJECTED)],
        [3.0, 2.0, 1.0]
    );

    // A sibling endpoint serving a clone of A's config is the same worker:
    // A's held request leaves it no room, until that request goes away.
    let sibling = a.serve("sibling").await;
    refused(&sibling, &mut a).await;
    assert_eq!(a.total(REJECTED), 2.0);
    release(held_a);
    a.occupancy_reaches((0.0, 0.0)).await;
    let sibling = a.router("sibling").await;
    let held_sibling = hold(&sibling, &mut a).await;
    a.occupancy_reaches((1.0, 0.0)).await;
    assert_eq!(
        b.total(ENGINE_REQUESTS),
        2.0,
        "B's own requests are untouched"
    );
    assert_eq!(
        [a.total(RECEIVED), a.total(ADMITTED), a.total(REJECTED)],
        [4.0, 2.0, 2.0]
    );

    // Capacity and rank reports stay with their worker. A LoRA adapter's card
    // reports nothing, so B, full at two, stays capped.
    let mut adapter = ModelDeploymentCard::with_name_only("adapter");
    adapter.lora = Some(LoraInfo {
        name: "adapter".to_string(),
        max_gpu_lora_count: None,
    });
    adapter.runtime_config.max_num_seqs = Some(8);
    b.admission_gate
        .get_or_create(&component_b)
        .record_model_card(&adapter);
    refused(&b.router("generate").await, &mut b).await;
    assert_eq!(
        b.ranks(),
        BTreeSet::from(["0".into(), "unspecified".into()])
    );

    // A's worker registers a new card, three sequences on rank 4, through the
    // same path: A, holding one request, now admits two more and refuses the
    // next, while B stays capped at two and never sees A's rank.
    let resized = a.reregistered(model("worker-a", 3, 4, 1).await);
    a.serve_config("resized", resized).await;
    let to_a = a.router("generate").await;
    let held_a = [hold(&to_a, &mut a).await, hold(&to_a, &mut a).await];
    a.occupancy_reaches((3.0, 0.0)).await;
    refused(&to_a, &mut a).await;
    refused(&b.router("generate").await, &mut b).await;
    assert_eq!(
        a.ranks(),
        BTreeSet::from(["0".into(), "4".into(), "unspecified".into()])
    );
    assert_eq!(
        b.ranks(),
        BTreeSet::from(["0".into(), "unspecified".into()])
    );
    assert_eq!(
        [a.total(RECEIVED), a.total(ADMITTED), a.total(REJECTED)],
        [7.0, 4.0, 3.0]
    );
    assert_eq!(
        [b.total(RECEIVED), b.total(ADMITTED), b.total(REJECTED)],
        [5.0, 2.0, 3.0]
    );

    release(held_sibling);
    for stream in held_a.into_iter().chain(held_b) {
        release(stream);
    }
    a.occupancy_reaches((0.0, 0.0)).await;
    b.occupancy_reaches((0.0, 0.0)).await;

    rt.shutdown();
}

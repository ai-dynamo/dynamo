// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A frontend `RoutingHost`, built by the frontend's routing builder, routing
//! over [`GatewayDispatch`] across two discovered workers. Each test drives
//! the host the way the Gateway EPP does: generate, take the route, feed the
//! progress Envoy relays, then complete or abort.

use std::{
    collections::{BTreeMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use dynamo_kv_router::{
    config::KvRouterConfig,
    scheduling::{ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier},
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Client,
    discovery::EventTransportKind,
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    pipeline::{Context, RouterMode, ServiceEngine},
    storage::kv::Selector,
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use tokio::{sync::watch, task::JoinHandle};

use super::*;
use crate::{
    discovery::{LoadThresholdConfig, LoadThresholdHandle, ModelManager},
    entrypoint::build_preprocessed_routing_with_dispatch,
    http::service::client_error_response,
    kv_router::{KvRouter, RouterLoadSource, RoutingLoadContext, SelectionPolicySource},
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
};

type Engine = ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>;

struct Harness {
    engine: Engine,
    dispatch: Arc<GatewayDispatch>,
    chooser: Arc<KvRouter>,
    client: Client,
    workers: HashSet<u64>,
    runtime: Runtime,
    _store: tempfile::TempDir,
    _drts: Vec<DistributedRuntime>,
}

#[derive(Default)]
struct Options {
    session_affinity_ttl_secs: Option<u64>,
    classifier: Option<RecordingClassifier>,
    track_output_blocks: bool,
    prefill_tokens_threshold: Option<u64>,
}

async fn harness(namespace: &str, options: Options) -> Harness {
    async fn shared_drt(runtime: Runtime, store: &std::path::Path) -> DistributedRuntime {
        DistributedRuntime::new(
            runtime,
            DistributedConfig {
                discovery_backend: DiscoveryBackend::KvStore(Selector::File(store.to_path_buf())),
                nats_config: None,
                request_plane: RequestPlaneMode::Tcp,
                response_plane: None,
                event_transport_kind: EventTransportKind::Zmq,
            },
        )
        .await
        .unwrap()
    }

    let runtime = Runtime::from_current().unwrap();
    let store = tempfile::tempdir().unwrap();
    // The router's runtime doubles as the first worker: each runtime holds a
    // discovery watch, and hosts with many sessions run short of watches.
    let router_drt = shared_drt(runtime.clone(), store.path()).await;
    let worker_drt = shared_drt(runtime.clone(), store.path()).await;
    let endpoint_for = |drt: &DistributedRuntime| {
        drt.namespace(namespace.to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate")
    };
    for drt in [&router_drt, &worker_drt] {
        endpoint_for(drt)
            .register_endpoint_instance()
            .await
            .unwrap();
    }

    let endpoint = endpoint_for(&router_drt);
    let client = endpoint.client().await.unwrap();
    let load_context = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::Aggregated,
        LoadThresholdHandle::new(LoadThresholdConfig {
            active_prefill_tokens_threshold: options.prefill_tokens_threshold,
            ..Default::default()
        }),
        &client.endpoint.drt().child_token(),
        None,
    )
    .await
    .unwrap();
    let workers = tokio::time::timeout(Duration::from_secs(5), async {
        let mut source = client.instance_source.as_ref().clone();
        loop {
            let instances = source.borrow_and_update().clone();
            if instances.len() == 2 {
                return instances
                    .into_iter()
                    .map(|i| i.id())
                    .collect::<HashSet<_>>();
            }
            source.changed().await.unwrap();
        }
    })
    .await
    .expect("both workers must be discovered");

    let configs = workers
        .iter()
        .map(|id| (*id, ModelRuntimeConfig::default()))
        .collect();
    let (_configs_tx, configs) = watch::channel(configs);
    let config = KvRouterConfig {
        skip_initial_worker_wait: true,
        use_kv_events: false,
        router_track_active_blocks: true,
        router_track_output_blocks: options.track_output_blocks,
        ..Default::default()
    };
    let mut chooser = KvRouter::new_with_worker_role_and_scheduler_load(
        endpoint,
        client.clone(),
        configs,
        None,
        16,
        SelectionPolicySource::Registry,
        Some(config),
        None,
        None,
        "decode",
        None,
        false,
        None,
        None,
        load_context.scheduler_load_sender(),
        load_context.cancellation_token(),
    )
    .await
    .unwrap();
    if let Some(classifier) = options.classifier {
        chooser = chooser.with_request_classifier(classifier).unwrap();
    }
    let chooser = Arc::new(chooser);

    let dispatch = Arc::new(GatewayDispatch::new());
    let routing = build_preprocessed_routing_with_dispatch(
        &client,
        Arc::new(ModelManager::new()),
        RouterMode::KV,
        load_context,
        Some(chooser.clone()),
        options.session_affinity_ttl_secs,
        dispatch.clone(),
    )
    .await
    .unwrap();

    Harness {
        engine: routing.backend_engine().clone(),
        dispatch,
        chooser,
        client,
        workers,
        runtime,
        _store: store,
        _drts: vec![router_drt, worker_drt],
    }
}

fn prompt(first_token: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids((first_token..first_token + 32).collect::<Vec<_>>())
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .build()
        .unwrap()
}

/// What the EPP's `pick` does: route, take the worker, keep the stream polled.
async fn route(
    harness: &Harness,
    request_id: &str,
    request: PreprocessedRequest,
    session: Option<&str>,
) -> anyhow::Result<(GatewayRoute, JoinHandle<Vec<Annotated<LLMEngineOutput>>>)> {
    let mut context =
        Context::with_id_and_metadata(request, request_id.to_string(), BTreeMap::new());
    if let Some(session) = session {
        context.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new(session.to_string()),
        );
    }
    let stream = harness.engine.generate(context).await?;
    let route = harness
        .dispatch
        .take_route(request_id)
        .expect("a returned stream means the host dispatched");
    Ok((route, tokio::spawn(stream.collect())))
}

async fn active_requests(harness: &Harness) -> usize {
    harness
        .chooser
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap()
        .iter()
        .map(|load| load.active_requests)
        .sum()
}

fn finish() -> GatewayProgress {
    GatewayProgress::Finish {
        reason: Some(FinishReason::Stop),
        usage: None,
    }
}

#[tokio::test]
#[serial_test::serial]
async fn dispatch_hands_out_the_worker_and_frees_the_booking_on_completion() {
    let harness = harness("gateway-dispatch-complete", Options::default()).await;

    let (route, drained) = route(&harness, "req-1", prompt(1), None).await.unwrap();
    assert!(harness.workers.contains(&route.worker_id));
    assert_eq!(route.token_ids, (1..33).collect::<Vec<_>>());
    assert_eq!(active_requests(&harness).await, 1);

    assert!(
        harness
            .dispatch
            .progress("req-1", GatewayProgress::Tokens(3))
    );
    assert!(harness.dispatch.progress("req-1", finish()));
    harness.dispatch.complete("req-1");
    let outputs = drained.await.unwrap();

    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|output| output.err().is_none()));
    assert_eq!(active_requests(&harness).await, 0);
    assert_eq!(harness.dispatch.open_requests(), 0);
    harness.runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn teardown_frees_the_booking_without_reporting_the_worker_down() {
    let harness = harness("gateway-dispatch-teardown", Options::default()).await;

    // Mid-stream: the gateway stream closed after some output.
    let (route, drained) = route(&harness, "req-1", prompt(1), None).await.unwrap();
    harness
        .dispatch
        .progress("req-1", GatewayProgress::Tokens(2));
    harness.dispatch.abort("req-1", "client disconnected");
    drained.await.unwrap();
    assert_eq!(active_requests(&harness).await, 0);
    assert!(
        harness
            .client
            .instance_ids_avail()
            .contains(&route.worker_id),
        "a gateway-side cancellation is not a worker fault"
    );

    // Before any output: the EPP dropped its pick right after dispatch.
    let context = Context::with_id_and_metadata(prompt(100), "req-2".into(), BTreeMap::new());
    let stream = harness.engine.generate(context).await.unwrap();
    drop(stream);
    harness.dispatch.abort("req-2", "pick cancelled");
    tokio::time::timeout(Duration::from_secs(5), async {
        while active_requests(&harness).await != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("dropping the stream before polling it must release the booking");
    assert_eq!(harness.client.instance_ids_avail().len(), 2);
    harness.runtime.shutdown();
}

/// #16577: the pick books its worker, so a second concurrent pick sees that
/// load and goes elsewhere instead of racing onto the same idle worker.
#[tokio::test]
#[serial_test::serial]
async fn concurrent_picks_see_each_others_booking() {
    let harness = harness("gateway-dispatch-concurrent", Options::default()).await;

    let (first, first_drained) = route(&harness, "req-1", prompt(1), None).await.unwrap();
    let (second, second_drained) = route(&harness, "req-2", prompt(100), None).await.unwrap();
    assert_ne!(first.worker_id, second.worker_id);
    assert_eq!(active_requests(&harness).await, 2);

    for id in ["req-1", "req-2"] {
        harness.dispatch.complete(id);
    }
    first_drained.await.unwrap();
    second_drained.await.unwrap();
    assert_eq!(active_requests(&harness).await, 0);
    harness.runtime.shutdown();
}

/// #16582: with session affinity, a session's second request returns to the
/// worker it was bound to even though the other worker is idle. The control
/// request without a session shows that load alone would have moved it.
#[tokio::test]
#[serial_test::serial]
async fn session_affinity_pins_a_session_to_its_worker() {
    let harness = harness(
        "gateway-dispatch-affinity",
        Options {
            session_affinity_ttl_secs: Some(60),
            ..Default::default()
        },
    )
    .await;

    let (bound, bound_drained) = route(&harness, "req-1", prompt(1), Some("session-a"))
        .await
        .unwrap();
    let (control, control_drained) = route(&harness, "req-2", prompt(100), None).await.unwrap();
    assert_ne!(
        control.worker_id, bound.worker_id,
        "load alone moves a second request"
    );
    harness.dispatch.complete("req-2");
    control_drained.await.unwrap();

    let (repeat, repeat_drained) = route(&harness, "req-3", prompt(200), Some("session-a"))
        .await
        .unwrap();
    assert_eq!(repeat.worker_id, bound.worker_id);

    for id in ["req-1", "req-3"] {
        harness.dispatch.complete(id);
    }
    bound_drained.await.unwrap();
    repeat_drained.await.unwrap();
    harness.runtime.shutdown();
}

#[derive(Clone, Default)]
struct RecordingClassifier {
    events: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait::async_trait]
impl RequestClassifier for RecordingClassifier {
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        self.events.lock().unwrap().push("classify");
        Box::pin(async move { Ok(request) })
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        let name = match event {
            ClassifyEvent::Sent { .. } => "sent",
            ClassifyEvent::Responding { .. } => "responding",
            ClassifyEvent::Completed { .. } => "completed",
            ClassifyEvent::Aborted { .. } => "aborted",
            _ => "other",
        };
        self.events.lock().unwrap().push(name);
    }
}

/// #16591: a classifier installed on the host follows the whole lifecycle,
/// which the EPP could not drive while it only booked.
#[tokio::test]
#[serial_test::serial]
async fn classifier_sees_the_request_lifecycle() {
    let classifier = RecordingClassifier::default();
    let events = classifier.events.clone();
    let harness = harness(
        "gateway-dispatch-classifier",
        Options {
            classifier: Some(classifier),
            ..Default::default()
        },
    )
    .await;

    let (_, drained) = route(&harness, "req-1", prompt(1), None).await.unwrap();
    harness
        .dispatch
        .progress("req-1", GatewayProgress::Tokens(1));
    harness.dispatch.progress("req-1", finish());
    harness.dispatch.complete("req-1");
    drained.await.unwrap();

    tokio::time::timeout(Duration::from_secs(5), async {
        while !events.lock().unwrap().contains(&"completed") {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the classifier must see completion");
    let events = events.lock().unwrap().clone();
    let position = |name| events.iter().position(|event| *event == name);
    assert!(position("classify") < position("sent"), "{events:?}");
    assert!(position("sent") < position("responding"), "{events:?}");
    assert!(position("responding") < position("completed"), "{events:?}");
    harness.runtime.shutdown();
}

/// #16587: streamed output grows the worker's decode load block by block.
#[tokio::test]
#[serial_test::serial]
async fn streamed_progress_grows_output_block_load() {
    let harness = harness(
        "gateway-dispatch-output-blocks",
        Options {
            track_output_blocks: true,
            ..Default::default()
        },
    )
    .await;
    let decode_blocks = |harness: &Harness, worker_id: u64| {
        let chooser = harness.chooser.clone();
        async move {
            chooser
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap()
                .into_iter()
                .find(|load| load.worker_id == worker_id)
                .unwrap()
                .potential_decode_blocks
        }
    };

    let (route, drained) = route(&harness, "req-1", prompt(1), None).await.unwrap();
    let before = decode_blocks(&harness, route.worker_id).await;
    // One token per SSE event, as the EPP reports it. The 32-token prompt
    // fills two blocks exactly, so tokens 1 and 17 each open a new block.
    for _ in 0..17 {
        harness
            .dispatch
            .progress("req-1", GatewayProgress::Tokens(1));
    }
    let grown = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let blocks = decode_blocks(&harness, route.worker_id).await;
            if blocks >= before + 2 {
                return blocks;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("output progress must reach the scheduler");
    assert_eq!(grown, before + 2);

    harness.dispatch.complete("req-1");
    drained.await.unwrap();
    assert_eq!(decode_blocks(&harness, route.worker_id).await, 0);
    harness.runtime.shutdown();
}

/// #14638: when every worker is over its load threshold the host rejects
/// with the frontend's overload error, so the EPP answers with the frontend's
/// status (529 by default) rather than a blanket 503.
#[tokio::test]
#[serial_test::serial]
async fn overload_rejection_carries_the_frontend_status() {
    let harness = harness(
        "gateway-dispatch-rejection",
        Options {
            prefill_tokens_threshold: Some(1),
            ..Default::default()
        },
    )
    .await;

    // Both workers now hold an unfinished prefill above the threshold.
    let (_, first_drained) = route(&harness, "req-1", prompt(1), None).await.unwrap();
    let (_, second_drained) = route(&harness, "req-2", prompt(100), None).await.unwrap();

    let error = match route(&harness, "req-3", prompt(200), None).await {
        Ok(_) => panic!("every worker is over its prefill threshold"),
        Err(error) => error,
    };
    let chain = format!("{error:#}");
    let (status, _) = client_error_response(error);
    assert_eq!(status, 529, "{chain}");
    assert_eq!(active_requests(&harness).await, 2);

    for id in ["req-1", "req-2"] {
        harness.dispatch.complete(id);
    }
    first_drained.await.unwrap();
    second_drained.await.unwrap();
    harness.runtime.shutdown();
}

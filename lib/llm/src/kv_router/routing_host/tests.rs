// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use dynamo_kv_router::{
    DefaultWorkerSelector, WorkerSelectionPolicy,
    config::KvRouterConfig,
    protocols::RoutingConstraints,
    scheduling::{
        ClassifyError, ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
    },
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::{Client, Instance},
    discovery::EventTransportKind,
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    error::{ErrorType, match_error_chain},
    pipeline::{
        AddressedRequest, AsyncEngineContext, Context, ManyIn, Operator, PushRouter, RouterMode,
        ServerStreamingEngine, StreamingDispatch, context::Controller,
    },
    storage::kv::Selector,
    traits::DistributedRuntimeProvider,
};
use tokio::sync::{mpsc, watch};

use super::*;
use crate::{
    http::service::metrics::Metrics,
    kv_router::RoutingLoadContext,
    local_model::runtime_config::ModelRuntimeConfig,
    lora::{LoraReplicaConfig, LoraRoutingTable, LoraStateTracker},
    migration::Migration,
    protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
    protocols::common::preprocessor::MmRoutingInfo,
};

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids(vec![1])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .build()
        .unwrap()
}

async fn test_load_context(client: &Client) -> Arc<RoutingLoadContext> {
    RoutingLoadContext::start(
        client.clone(),
        crate::kv_router::RouterLoadSource::Aggregated,
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        &client.endpoint.drt().child_token(),
        None,
    )
    .await
    .unwrap()
}

#[test]
fn response_item_failed_includes_typed_terminal_failures() {
    let mut output = LLMEngineOutput::default();
    assert!(!response_item_failed(&Annotated::from_data(output.clone())));

    output.finish_reason = Some(FinishReason::Error("decode failed".to_string()));
    assert!(response_item_failed(&Annotated::from_data(output.clone())));

    output.finish_reason = Some(FinishReason::Cancelled);
    assert!(response_item_failed(&Annotated::from_data(output.clone())));

    output.finish_reason = Some(FinishReason::Length);
    assert!(!response_item_failed(&Annotated::from_data(output)));
}

#[test]
fn selector_state_remains_owned_by_the_scheduler_actor() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<RoutingHost<WorkerSelectionPolicy>>();
}

#[test]
fn builtin_policies_declare_capabilities() {
    for mode in [RouterMode::RoundRobin, RouterMode::Random] {
        let selector = BuiltinWorkerSelector::new(mode).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::NONE);
    }
    for mode in [RouterMode::PowerOfTwoChoices, RouterMode::LeastLoaded] {
        let selector = BuiltinWorkerSelector::new(mode).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::OCCUPANCY);
    }
}

#[tokio::test]
async fn builtin_host_constructs_only_declared_capabilities() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-capabilities".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    let load_context = test_load_context(&client).await;
    let inner = PushRouter::from_client(client.clone(), RouterMode::RoundRobin)
        .await
        .unwrap();
    let host =
        RoutingHost::<DefaultWorkerSelector>::new_builtin(inner, load_context.clone()).unwrap();

    assert_eq!(host.required_worker_inputs(), WorkerInputs::NONE);
    assert!(host.hosted_occupancy.is_none());

    drop(host);

    client.override_discovered_instances(vec![1, 2]);
    client.override_instance_avail(vec![1, 2]);
    let inner = PushRouter::from_client(client, RouterMode::PowerOfTwoChoices)
        .await
        .unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin(inner, load_context).unwrap();
    let RoutingPolicy::Builtin(selector) = &host.policy else {
        unreachable!()
    };
    assert_eq!(host.required_worker_inputs(), WorkerInputs::OCCUPANCY);
    assert!(host.hosted_occupancy.is_some());
    let selection = host
        .hosted_occupancy
        .as_ref()
        .unwrap()
        .select_and_reserve(&host.inner, selector, Some(1))
        .unwrap();
    assert_eq!(selection.worker_id, 1);
    assert_eq!(selection.occupancy, 1);
    assert_eq!(host.inner.occupancy_for_test(1), 1);
    let mut guard: RequestGuard<DefaultWorkerSelector> = RequestGuard::new_builtin(
        Arc::clone(&host.request_metrics),
        selection.worker_id,
        Some(selection.reservation),
        None,
        &request(),
    );
    assert_eq!(guard.retarget_worker(1), Some(1));
    guard.abort().await;
    assert_eq!(host.inner.occupancy_for_test(1), 0);

    drop(host);
    runtime.shutdown();
}

#[tokio::test]
async fn builtin_occupancy_selection_uses_all_selectable_workers() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-occupancy-workers".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    let load_context = test_load_context(&client).await;
    let inner = PushRouter::from_client(client.clone(), RouterMode::LeastLoaded)
        .await
        .unwrap();
    client.override_discovered_instances(vec![1, 2]);
    client.override_instance_avail(vec![1, 2]);
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin(inner, load_context).unwrap();
    let RoutingPolicy::Builtin(selector) = &host.policy else {
        unreachable!()
    };
    let occupancy = host.hosted_occupancy.as_ref().unwrap();
    let first = occupancy
        .select_and_reserve(&host.inner, selector, Some(1))
        .unwrap();
    let second = occupancy
        .select_and_reserve(&host.inner, selector, None)
        .unwrap();

    assert_eq!(first.worker_id, 1);
    assert_eq!(second.worker_id, 2);
    assert_eq!(second.candidate_count, 2);
    assert_eq!(second.occupancy, 1);

    drop(second);
    drop(first);
    drop(host);
    runtime.shutdown();
}

#[tokio::test]
async fn builtin_direct_without_worker_is_invalid_argument() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-direct-invalid-argument".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    let load_context = test_load_context(&client).await;
    let inner = PushRouter::from_client(client, RouterMode::Direct)
        .await
        .unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin_with_coordinator(
        inner,
        load_context,
        None,
    )
    .unwrap();

    let error = host.generate(Context::new(request())).await.unwrap_err();
    assert!(match_error_chain(
        error.as_ref(),
        &[ErrorType::InvalidArgument],
        &[]
    ));

    drop(host);
    runtime.shutdown();
}

#[derive(Default)]
struct CompletedBuiltinDispatch {
    worker_ids: Mutex<Vec<u64>>,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>
    for CompletedBuiltinDispatch
{
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (_, _, instance) = addressed.into_parts();
        self.worker_ids
            .lock()
            .unwrap()
            .push(instance.expect("selected worker instance").id());
        let output = Annotated::from_data(LLMEngineOutput {
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        });
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { output })),
            context.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("the routing host dispatches unary requests")
    }
}

#[tokio::test]
#[serial_test::serial]
async fn builtin_direct_dispatch_ignores_local_inhibition() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-direct-local-inhibition".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    let load_context = test_load_context(&client).await;
    endpoint.register_endpoint_instance().await.unwrap();
    let worker_id = client.wait_for_instances().await.unwrap()[0].id();
    let dispatch = Arc::new(CompletedBuiltinDispatch::default());
    let inner = PushRouter::from_client_with_dispatch(
        client.clone(),
        RouterMode::Direct,
        Arc::clone(&dispatch) as Arc<dyn StreamingDispatch<_, _>>,
    )
    .await
    .unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin_with_coordinator(
        inner,
        load_context,
        None,
    )
    .unwrap();

    client.report_instance_down(worker_id);
    assert!(client.instance_ids().contains(&worker_id));
    assert!(!client.instance_ids_avail().contains(&worker_id));

    let mut direct_request = request();
    direct_request.routing_mut().backend_instance_id = Some(worker_id);
    let mut stream = host.generate(Context::new(direct_request)).await.unwrap();
    while stream.next().await.is_some() {}
    assert_eq!(dispatch.worker_ids.lock().unwrap().as_slice(), &[worker_id]);

    drop(host);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn builtin_lora_keeps_separate_selection_and_cleanup() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-lora-capability".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    let load_context = test_load_context(&client).await;
    endpoint.register_endpoint_instance().await.unwrap();
    let worker_id = client.wait_for_instances().await.unwrap()[0].id();
    let stale_worker = worker_id.wrapping_add(1);
    client.override_instance_avail(vec![stale_worker, worker_id]);
    let dispatch = Arc::new(CompletedBuiltinDispatch::default());
    let inner = PushRouter::from_client_with_dispatch(
        client,
        RouterMode::RoundRobin,
        Arc::clone(&dispatch) as Arc<dyn StreamingDispatch<_, _>>,
    )
    .await
    .unwrap();
    let routing_table = LoraRoutingTable::new();
    routing_table.update_allocation(
        "adapter".to_string(),
        LoraReplicaConfig {
            lora_name: "adapter".to_string(),
            replica_factor: 1,
            replica_set: vec![WorkerWithDpRank::new(worker_id, 0)],
            updated_at: Instant::now(),
            is_active: true,
        },
    );
    let filter = Arc::new(LoraFilter::new(routing_table, LoraStateTracker::new()));
    let estimator = Arc::new(LoadEstimator::new());
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin_with_capabilities(
        inner,
        load_context,
        None,
        Some((filter, Arc::clone(&estimator))),
    )
    .unwrap();

    let mut base_stream = host.generate(Context::new(request())).await.unwrap();
    while base_stream.next().await.is_some() {}

    let mut adapter_request = request();
    adapter_request.routing_mut().lora_name = Some("adapter".to_string());
    let mut adapter_stream = host.generate(Context::new(adapter_request)).await.unwrap();
    assert_eq!(estimator.get_inflight_counts().get("adapter"), Some(&1));
    while adapter_stream.next().await.is_some() {}

    assert_eq!(dispatch.worker_ids.lock().unwrap().len(), 2);
    assert_eq!(dispatch.worker_ids.lock().unwrap()[1], worker_id);
    assert!(!estimator.get_inflight_counts().contains_key("adapter"));

    drop(host);
    runtime.shutdown();
}

fn affinity_request(
    session_id: &str,
    explicit_worker: Option<u64>,
) -> SingleIn<PreprocessedRequest> {
    let mut content = request();
    content.routing_mut().backend_instance_id = explicit_worker;
    let mut request = Context::new(content);
    request.insert(
        SESSION_AFFINITY_CONTEXT_KEY,
        SessionAffinityId::new(session_id),
    );
    request
}

#[tokio::test]
#[serial_test::serial]
async fn builtin_affinity_uses_common_host_for_every_policy() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let component = distributed
        .namespace("builtin-affinity-lifecycle".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap();

    for (index, mode) in [
        RouterMode::RoundRobin,
        RouterMode::Random,
        RouterMode::PowerOfTwoChoices,
        RouterMode::LeastLoaded,
        RouterMode::DeviceAwareWeighted,
        RouterMode::Direct,
    ]
    .into_iter()
    .enumerate()
    {
        let endpoint = component.endpoint(format!("mode-{index}"));
        let client = endpoint.client().await.unwrap();
        let load_context = test_load_context(&client).await;
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let dispatch = Arc::new(CompletedBuiltinDispatch::default());
        let inner = PushRouter::from_client_with_dispatch(
            client,
            mode,
            Arc::clone(&dispatch) as Arc<dyn StreamingDispatch<_, _>>,
        )
        .await
        .unwrap();
        let affinity = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
        let host = RoutingHost::<DefaultWorkerSelector>::new_builtin_with_coordinator(
            inner,
            load_context,
            Some(affinity.clone()),
        )
        .unwrap();
        let session_id = format!("session-{index}");
        let affinity_id = SessionAffinityId::new(session_id.clone());
        let explicit_worker = (mode == RouterMode::Direct).then_some(worker_id);

        let mut first = host
            .generate(affinity_request(&session_id, explicit_worker))
            .await
            .unwrap();
        while first.next().await.is_some() {}
        assert_eq!(
            affinity.query_target(&affinity_id, None).unwrap(),
            Some(AffinityTarget::worker(worker_id))
        );

        let mut second = host
            .generate(affinity_request(&session_id, explicit_worker))
            .await
            .unwrap();
        while second.next().await.is_some() {}
        assert_eq!(
            dispatch.worker_ids.lock().unwrap().as_slice(),
            &[worker_id; 2]
        );
    }

    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn builtin_direct_fallback_stays_disabled_for_affinity() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-direct-worker-loss".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    let load_context = test_load_context(&client).await;
    endpoint.register_endpoint_instance().await.unwrap();
    let real_worker = client.wait_for_instances().await.unwrap()[0].id();
    let stale_worker = real_worker.wrapping_add(1);
    client.override_instance_avail(vec![stale_worker, real_worker]);
    let dispatch = Arc::new(CompletedBuiltinDispatch::default());
    let inner = PushRouter::from_client_with_dispatch(
        client,
        RouterMode::Direct,
        Arc::clone(&dispatch) as Arc<dyn StreamingDispatch<_, _>>,
    )
    .await
    .unwrap();
    let affinity = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin_with_coordinator(
        inner,
        load_context,
        Some(affinity.clone()),
    )
    .unwrap();

    let mut standalone = request();
    standalone.routing_mut().backend_instance_id = Some(stale_worker);
    let mut stream = host.generate(Context::new(standalone)).await.unwrap();
    while stream.next().await.is_some() {}
    assert_eq!(
        dispatch.worker_ids.lock().unwrap().as_slice(),
        &[real_worker]
    );

    let session_id = SessionAffinityId::new("direct-affinity");
    let AffinityAcquire::Initialize(initializer) =
        affinity.acquire(&session_id, None).await.unwrap()
    else {
        panic!("new affinity session must initialize");
    };
    drop(
        initializer
            .commit(AffinityTarget::worker(stale_worker))
            .unwrap(),
    );
    assert!(
        host.generate(affinity_request("direct-affinity", Some(stale_worker)))
            .await
            .is_err()
    );
    assert_eq!(
        dispatch.worker_ids.lock().unwrap().as_slice(),
        &[real_worker]
    );
    assert_eq!(affinity.query_target(&session_id, None).unwrap(), None);

    drop(host);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn terminal_item_does_not_skip_transport_eof() {
    let (router, runtime) = router(None).await;
    let inputs = router.required_worker_inputs();
    assert!(inputs.contains(WorkerInputs::CACHE));
    assert!(inputs.contains(WorkerInputs::LOAD));
    let context = Context::new(()).context();
    let drained = Arc::new(AtomicBool::new(false));
    let source_drained = Arc::clone(&drained);
    let source = ResponseStream::new(
        Box::pin(async_stream::stream! {
            yield Annotated::from_data(LLMEngineOutput {
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            });
            source_drained.store(true, Ordering::Release);
        }),
        Arc::clone(&context),
    );
    let guard = RequestGuard::new_kv(
        Arc::clone(router.kv_router()),
        Arc::clone(&router.request_metrics),
        "terminal-drain".to_string(),
        WorkerWithDpRank::from_worker_id(0),
        &request(),
        false,
        None,
    );
    let monitored = monitor_response_stream(source, context, guard);
    tokio::pin!(monitored);

    assert!(monitored.next().await.is_some());
    assert!(monitored.next().await.is_none());
    assert!(drained.load(Ordering::Acquire));

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn stream_failure_releases_booking_before_error_is_observable() {
    let (router, runtime) = router(None).await;
    let context_id = "stream-failure-cleanup".to_string();
    let failed_request =
        Context::with_id_and_metadata(request(), context_id.clone(), Default::default());
    let (mut failed_selection, _) = router
        .select_with_affinity(&failed_request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    let failed_worker = failed_selection.worker;
    let failed_guard = router
        .track_selection(&failed_request, &mut failed_selection, false)
        .await
        .unwrap();
    let failure = Annotated {
        data: None,
        id: None,
        event: Some("error".to_string()),
        comment: None,
        error: Some(
            DynamoError::builder()
                .error_type(ErrorType::WorkerOverloaded)
                .message("selected worker is overloaded")
                .build(),
        ),
    };
    let source = ResponseStream::new(
        Box::pin(stream::once(async move { failure })),
        failed_request.context().clone(),
    );
    let monitored = monitor_response_stream(source, failed_request.context().clone(), failed_guard);
    tokio::pin!(monitored);

    let item = monitored.next().await.expect("failed item must be yielded");
    assert!(item.error.is_some());

    // The monitored stream is still suspended at its yield point. Rebooking
    // the same id on the same worker proves cleanup completed before the
    // failure became visible, rather than relying on EOF or Drop cleanup.
    let retry_request =
        Context::with_id_and_metadata(request(), context_id.clone(), Default::default());
    let (mut retry_selection, _) = router
        .select_with_affinity(&retry_request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    assert_eq!(retry_selection.worker, failed_worker);
    let mut retry_guard = router
        .track_selection(&retry_request, &mut retry_selection, false)
        .await
        .expect("same-worker booking must be released before yielding the error");
    retry_guard.abort().await;

    drop(router);
    runtime.shutdown();
}

async fn router(session_affinity_ttl: Option<Duration>) -> (RoutingHost, Runtime) {
    router_with_workers(session_affinity_ttl, &[7]).await
}

async fn router_with_workers(
    session_affinity_ttl: Option<Duration>,
    worker_ids: &[u64],
) -> (RoutingHost, Runtime) {
    let workers = worker_ids
        .iter()
        .copied()
        .map(|worker_id| (worker_id, ModelRuntimeConfig::default()))
        .collect();
    router_with_worker_configs(session_affinity_ttl, workers).await
}

async fn router_with_worker_configs(
    session_affinity_ttl: Option<Duration>,
    workers: HashMap<u64, ModelRuntimeConfig>,
) -> (RoutingHost, Runtime) {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let component = distributed
        .namespace("affinity-selection-cancellation".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap();
    let endpoint = component.endpoint("generate");
    let client = endpoint.client().await.unwrap();
    let worker_ids = workers.keys().copied().collect::<Vec<_>>();
    let (_tx, workers) = watch::channel(workers);
    let config = KvRouterConfig {
        skip_initial_worker_wait: true,
        use_kv_events: false,
        router_track_active_blocks: false,
        ..Default::default()
    };
    let chooser = KvRouter::new(
        endpoint,
        client.clone(),
        workers,
        None,
        16,
        DefaultWorkerSelector::new(Some(config.clone()), "decode"),
        Some(config),
        None,
        "decode",
        None,
        false,
        None,
        None,
    )
    .await
    .unwrap();
    let inner = PushRouter::from_client(client, RouterMode::KV)
        .await
        .unwrap();
    let router = RoutingHost::new(inner, Arc::new(chooser), session_affinity_ttl).unwrap();
    router
        .inner
        .client
        .override_discovered_instances(worker_ids.clone());
    router.inner.client.override_instance_avail(worker_ids);
    (router, runtime)
}

async fn router_with_classifier(
    classifier: impl RequestClassifier,
    session_affinity_ttl: Option<Duration>,
) -> (RoutingHost, Runtime) {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("request-classifier-routing-host".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate");
    let client = endpoint.client().await.unwrap();
    let worker_id = 7;
    let (_tx, workers) =
        watch::channel(HashMap::from([(worker_id, ModelRuntimeConfig::default())]));
    let config = KvRouterConfig {
        skip_initial_worker_wait: true,
        use_kv_events: false,
        router_track_active_blocks: false,
        ..Default::default()
    };
    let chooser = KvRouter::new(
        endpoint,
        client.clone(),
        workers,
        None,
        16,
        DefaultWorkerSelector::new(Some(config.clone()), "decode"),
        Some(config),
        None,
        "decode",
        None,
        false,
        None,
        None,
    )
    .await
    .unwrap()
    .with_request_classifier(classifier)
    .unwrap();
    let inner = PushRouter::from_client(client, RouterMode::KV)
        .await
        .unwrap();
    let router = RoutingHost::new(inner, Arc::new(chooser), session_affinity_ttl).unwrap();
    router
        .inner
        .client
        .override_discovered_instances(vec![worker_id]);
    router.inner.client.override_instance_avail(vec![worker_id]);
    (router, runtime)
}

#[derive(Debug, PartialEq, Eq)]
enum ClassifierObservation {
    Completed(usize),
    Aborted,
}

struct RecordingClassifier {
    calls: Arc<AtomicUsize>,
    observations: mpsc::UnboundedSender<ClassifierObservation>,
}

impl RequestClassifier for RecordingClassifier {
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move { Ok(request) })
    }

    fn on_event(&mut self, event: ClassifyEvent<'_>) {
        let observation = match event {
            ClassifyEvent::Completed {
                context_tokens: Some(context_tokens),
                ..
            } => Some(ClassifierObservation::Completed(context_tokens)),
            ClassifyEvent::Aborted { .. } => Some(ClassifierObservation::Aborted),
            _ => None,
        };
        if let Some(observation) = observation {
            self.observations.send(observation).unwrap();
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("classifier rejected request")]
struct ClassifierRejected;

#[test]
fn classification_failures_are_not_affinity_retryable() {
    let failures = [
        KvSchedulerError::RequestClassifierPanicked("panic".to_string()),
        KvSchedulerError::RequestClassifierFailed(
            Box::new(ClassifierRejected) as Box<ClassifyError>
        ),
        KvSchedulerError::RequestClassifierReplacedRequest,
        KvSchedulerError::DuplicateClassificationRequestId("request".to_string()),
        KvSchedulerError::InvalidClassificationMetadata("metadata".to_string()),
    ];

    for failure in failures {
        let error = anyhow::Error::from(failure);
        assert!(classification_failure(&error).is_some());
    }
    assert!(classification_failure(&anyhow::Error::from(KvSchedulerError::NoEndpoints)).is_none());
}

struct RejectingClassifier {
    calls: Arc<AtomicUsize>,
    observations: mpsc::UnboundedSender<Option<(bool, String)>>,
}

impl RequestClassifier for RejectingClassifier {
    fn classify(&mut self, _request: ClassifyRequest) -> ClassifyFuture {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async { Err(Box::new(ClassifierRejected) as Box<ClassifyError>) })
    }

    fn on_event(&mut self, event: ClassifyEvent<'_>) {
        if let ClassifyEvent::Aborted { error, .. } = event {
            self.observations
                .send(error.map(|error| {
                    (
                        error.downcast_ref::<ClassifierRejected>().is_some(),
                        error.to_string(),
                    )
                }))
                .unwrap();
        }
    }
}

#[tokio::test]
async fn classifier_failure_aborts_once_with_original_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (observations_tx, mut observations_rx) = mpsc::unbounded_channel();
    let (router, runtime) = router_with_classifier(
        RejectingClassifier {
            calls: Arc::clone(&calls),
            observations: observations_tx,
        },
        Some(Duration::from_secs(10)),
    )
    .await;
    let session_id = SessionAffinityId::new("classifier-failure-stale-affinity");
    bind_affinity_target(&router, &session_id, AffinityTarget::new(7, Some(1))).await;
    let mut request = Context::new(request());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id);

    assert!(
        router
            .select_with_affinity(&request, RequestPhase::Aggregated, false)
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), observations_rx.recv())
            .await
            .expect("classifier abort event timed out"),
        Some(Some((true, "classifier rejected request".to_string())))
    );
    assert!(observations_rx.try_recv().is_err());

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn stale_affinity_rebind_classifies_once_without_intermediate_abort() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (observations_tx, mut observations_rx) = mpsc::unbounded_channel();
    let (router, runtime) = router_with_classifier(
        RecordingClassifier {
            calls: Arc::clone(&calls),
            observations: observations_tx,
        },
        Some(Duration::from_secs(10)),
    )
    .await;
    let session_id = SessionAffinityId::new("classifier-stale-affinity");
    bind_affinity_target(&router, &session_id, AffinityTarget::new(7, Some(1))).await;
    let mut request = Context::new(request());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id);

    let (mut selection, operation) = router
        .select_with_affinity(&request, RequestPhase::Aggregated, false)
        .await
        .unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(observations_rx.try_recv().is_err());
    let mut lifecycle = selection.lifecycle.take().unwrap();
    lifecycle.observe_context_tokens(1);
    lifecycle.complete();
    assert_eq!(
        observations_rx.recv().await,
        Some(ClassifierObservation::Completed(1))
    );
    router.kv_router().free(request.id()).await.unwrap();

    drop(operation);
    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn query_only_selection_bypasses_classifier_and_request_lifecycle() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (observations_tx, mut observations_rx) = mpsc::unbounded_channel();
    let (router, runtime) = router_with_classifier(
        RecordingClassifier {
            calls: Arc::clone(&calls),
            observations: observations_tx,
        },
        None,
    )
    .await;
    let request = Context::new(request());

    let (selection, operation) = router
        .select_with_affinity(&request, RequestPhase::Aggregated, true)
        .await
        .unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(selection.lifecycle.is_none());
    assert!(operation.is_none());
    drop(selection);
    assert!(observations_rx.try_recv().is_err());

    drop(router);
    runtime.shutdown();
}

struct TokenContractClassifier {
    classified: mpsc::UnboundedSender<usize>,
    completed: mpsc::UnboundedSender<usize>,
}

impl RequestClassifier for TokenContractClassifier {
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        self.classified.send(request.input_tokens()).unwrap();
        Box::pin(async move { Ok(request) })
    }

    fn on_event(&mut self, event: ClassifyEvent<'_>) {
        if let ClassifyEvent::Completed {
            context_tokens: Some(context_tokens),
            ..
        } = event
        {
            self.completed.send(context_tokens).unwrap();
        }
    }
}

#[tokio::test]
async fn classifier_and_completion_use_authoritative_multimodal_token_counts() {
    let (classified_tx, mut classified_rx) = mpsc::unbounded_channel();
    let (completed_tx, mut completed_rx) = mpsc::unbounded_channel();
    let (router, runtime) = router_with_classifier(
        TokenContractClassifier {
            classified: classified_tx,
            completed: completed_tx,
        },
        None,
    )
    .await;
    let mut content = request();
    content.mm_routing_info = Some(MmRoutingInfo {
        routing_token_ids: vec![1, 2, 3, 4, 5, 0, 0, 0],
        block_mm_infos: vec![None],
        expanded_prompt_len: 5,
    });
    let request = Context::new(content);
    let (mut selection, _) = router
        .select_with_affinity(&request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    assert_eq!(classified_rx.recv().await, Some(5));

    let mut guard = router
        .track_selection(&request, &mut selection, false)
        .await
        .unwrap();
    guard.mark_dispatched();
    let output = LLMEngineOutput {
        completion_usage: Some(dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 5,
            completion_tokens: 6,
            total_tokens: 11,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
        ..Default::default()
    };
    guard.on_item(&Annotated::from_data(output)).await;
    guard.finish().await;
    assert_eq!(completed_rx.recv().await, Some(11));

    drop(router);
    runtime.shutdown();
}

async fn track_request(
    router: &RoutingHost,
    is_query_only: bool,
) -> (SingleIn<PreprocessedRequest>, WorkerSelection, RequestGuard) {
    let request = Context::new(request());
    let (mut selection, _) = router
        .select_with_affinity(&request, RequestPhase::Aggregated, is_query_only)
        .await
        .unwrap();
    let guard = router
        .track_selection(&request, &mut selection, is_query_only)
        .await
        .unwrap();
    (request, selection, guard)
}

#[tokio::test]
async fn session_affinity_disabled_does_not_create_coordinator() {
    let (router, runtime) = router(None).await;
    assert!(router.affinity.is_none());

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn router_request_counters_follow_admission_and_completion_lifecycle() {
    let (router, runtime) = router(None).await;
    let metrics = router.request_metrics.clone();
    let started_before = metrics.requests_started_total().get();
    let completed_before = metrics.requests_total.get();

    let controller = Controller::new("pre-admission-cancellation".to_string());
    controller.stop();
    let cancelled_request = Context::with_controller(request(), controller);
    assert!(
        router
            .select_with_affinity(&cancelled_request, RequestPhase::Aggregated, false)
            .await
            .is_err()
    );
    assert_eq!(metrics.requests_started_total().get(), started_before);

    let (_, _, mut query_guard) = track_request(&router, true).await;
    query_guard.abort().await;
    drop(query_guard);
    assert_eq!(metrics.requests_started_total().get(), started_before);

    let (_, _, mut cancelled_guard) = track_request(&router, false).await;

    assert_eq!(metrics.requests_started_total().get(), started_before + 1);
    assert_eq!(metrics.requests_total.get(), completed_before);

    // Admission remains counted even when the request aborts before dispatch.
    cancelled_guard.abort().await;
    drop(cancelled_guard);
    assert_eq!(metrics.requests_started_total().get(), started_before + 1);
    assert_eq!(metrics.requests_total.get(), completed_before);

    let mut failed_input = request();
    failed_input.migration_state = Some(Default::default());
    let migration_state = failed_input.migration_state.clone().unwrap();
    let failed_request = Context::new(failed_input);
    let (mut failed_selection, _) = router
        .select_with_affinity(&failed_request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    let failed_worker = failed_selection.worker.worker_id;
    let failed_dispatch_guard = router
        .track_selection(&failed_request, &mut failed_selection, false)
        .await
        .unwrap();
    assert!(
        router
            .dispatch_selection(failed_request, failed_selection, failed_dispatch_guard,)
            .await
            .is_err()
    );
    assert_eq!(migration_state.excluded_worker_ids(), vec![failed_worker]);
    assert_eq!(metrics.requests_started_total().get(), started_before + 2);
    assert_eq!(metrics.requests_total.get(), completed_before);

    let (_, _, mut completed_guard) = track_request(&router, false).await;
    completed_guard.start_dispatch("aggregated");
    completed_guard.mark_dispatched();
    completed_guard.finish().await;
    drop(completed_guard);
    assert_eq!(metrics.requests_started_total().get(), started_before + 3);
    assert_eq!(metrics.requests_total.get(), completed_before + 1);

    let mut builtin_guard = RequestGuard::<DefaultWorkerSelector>::new_builtin(
        Arc::clone(&metrics),
        7,
        None,
        None,
        &request(),
    );
    assert_eq!(metrics.requests_started_total().get(), started_before + 4);
    builtin_guard.abort().await;
    drop(builtin_guard);
    assert_eq!(metrics.requests_total.get(), completed_before + 1);

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn session_affinity_post_selection_cancellation_preserves_binding() {
    let (router, runtime) = router(Some(Duration::from_secs(10))).await;
    let affinity = router.affinity.as_ref().unwrap();
    let session_id = SessionAffinityId::new("cancelled-after-selection");
    let original_target = AffinityTarget {
        worker_id: 7,
        dp_rank: Some(0),
    };
    let AffinityAcquire::Initialize(initializer) =
        affinity.acquire(&session_id, None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(original_target).unwrap());

    let mut operation = Some(affinity.acquire(&session_id, None).await.unwrap());
    let cancellation = cancellation::cancelled_error("cancelled-after-selection-request");
    invalidate_on_non_cancellation(&mut operation, &cancellation);
    assert!(operation.is_some());
    drop(operation);
    assert_eq!(
        affinity.query_target(&session_id, None).unwrap(),
        Some(original_target)
    );

    let mut operation = Some(affinity.acquire(&session_id, None).await.unwrap());
    let failure = anyhow::anyhow!("dispatch failed");
    invalidate_on_non_cancellation(&mut operation, &failure);
    assert!(operation.is_none());
    assert_eq!(affinity.query_target(&session_id, None).unwrap(), None);

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn session_affinity_existing_selection_cancellation_preserves_binding_without_retry() {
    let (router, runtime) = router(Some(Duration::from_secs(10))).await;
    let session_id = SessionAffinityId::new("cancelled-selection");
    let original_target = AffinityTarget {
        worker_id: 7,
        dp_rank: Some(0),
    };
    let AffinityAcquire::Initialize(initializer) = router
        .affinity
        .as_ref()
        .unwrap()
        .acquire(&session_id, None)
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(original_target).unwrap());

    let controller = Controller::new("cancelled-selection-request".to_string());
    controller.stop();
    let mut request = Context::with_controller(request(), controller);
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());

    let Err(error) = router
        .select_with_affinity(&request, RequestPhase::Aggregated, false)
        .await
    else {
        panic!("stopped request must return cancellation");
    };
    assert!(match_error_chain(
        error.as_ref(),
        &[ErrorType::Cancelled],
        &[]
    ));
    assert_eq!(
        router
            .affinity
            .as_ref()
            .unwrap()
            .query_target(&session_id, None)
            .unwrap(),
        Some(original_target)
    );

    let AffinityAcquire::Bound { target, lease } = router
        .affinity
        .as_ref()
        .unwrap()
        .acquire(&session_id, None)
        .await
        .unwrap()
    else {
        panic!("cancellation must preserve the existing binding");
    };
    assert_eq!(target, original_target);
    drop(lease);

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn query_affinity_target_returns_existing_binding_without_reserving() {
    let (router, runtime) = router(Some(Duration::from_secs(10))).await;
    let session_id = SessionAffinityId::new("query-existing-binding");
    let target = AffinityTarget {
        worker_id: 7,
        dp_rank: Some(0),
    };
    let AffinityAcquire::Initialize(initializer) = router
        .affinity
        .as_ref()
        .unwrap()
        .acquire(&session_id, None)
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target).unwrap());

    let mut request = Context::new(request());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());

    assert_eq!(
        router
            .query_affinity_target(&request, RequestPhase::Prefill)
            .unwrap(),
        Some(target)
    );
    assert_eq!(
        router
            .affinity
            .as_ref()
            .unwrap()
            .query_target(&session_id, None)
            .unwrap(),
        Some(target)
    );

    drop(router);
    runtime.shutdown();
}

async fn bind_affinity_target(
    router: &RoutingHost,
    session_id: &SessionAffinityId,
    target: AffinityTarget,
) {
    let AffinityAcquire::Initialize(initializer) = router
        .affinity
        .as_ref()
        .unwrap()
        .acquire(session_id, None)
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target).unwrap());
}

#[tokio::test]
async fn request_constraints_preserve_worker_only_affinity() {
    let mut request_worker = ModelRuntimeConfig::default();
    request_worker.taints.insert("request-pool".to_string());
    let workers = HashMap::from([(7, ModelRuntimeConfig::default()), (8, request_worker)]);
    let (router, runtime) =
        router_with_worker_configs(Some(Duration::from_secs(10)), workers).await;
    let target = AffinityTarget::worker(7);

    let allowlist_session = SessionAffinityId::new("affinity-allowlist-conflict");
    bind_affinity_target(&router, &allowlist_session, target).await;
    let mut allowlist_input = request();
    allowlist_input.routing_mut().allowed_worker_ids = Some(HashSet::from([8]));
    let mut allowlist_request = Context::new(allowlist_input);
    allowlist_request.insert(SESSION_AFFINITY_CONTEXT_KEY, allowlist_session.clone());
    assert!(
        router
            .select_with_affinity(&allowlist_request, RequestPhase::Aggregated, false)
            .await
            .is_err()
    );
    assert_eq!(
        router
            .affinity
            .as_ref()
            .unwrap()
            .query_target(&allowlist_session, None)
            .unwrap(),
        Some(target)
    );

    let taint_session = SessionAffinityId::new("affinity-taint-conflict");
    bind_affinity_target(&router, &taint_session, target).await;
    let mut taint_input = request();
    taint_input.routing_mut().routing_constraints = Some(RoutingConstraints {
        required_taints: HashSet::from(["request-pool".to_string()]),
        ..Default::default()
    });
    let mut taint_request = Context::new(taint_input);
    taint_request.insert(SESSION_AFFINITY_CONTEXT_KEY, taint_session.clone());
    assert!(
        router
            .select_with_affinity(&taint_request, RequestPhase::Aggregated, false)
            .await
            .is_err()
    );
    assert_eq!(
        router
            .affinity
            .as_ref()
            .unwrap()
            .query_target(&taint_session, None)
            .unwrap(),
        Some(target)
    );

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn stale_affinity_rank_rebinds_once() {
    let (router, runtime) = router(Some(Duration::from_secs(10))).await;
    let session_id = SessionAffinityId::new("stale-affinity-rank");
    bind_affinity_target(&router, &session_id, AffinityTarget::new(7, Some(1))).await;
    let mut request = Context::new(request());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id);

    let (selection, operation) = router
        .select_with_affinity(&request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    assert_eq!(selection.worker, WorkerWithDpRank::new(7, 0));
    assert!(matches!(operation, Some(AffinityAcquire::Initialize(_))));
    router.kv_router().free(request.id()).await.unwrap();

    drop(operation);
    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn migration_exclusion_rebinds_affinity_without_widening_or_escaping_hard_pins() {
    let mut constrained_worker = ModelRuntimeConfig::default();
    constrained_worker.taints.insert("retry-pool".to_string());
    let workers = HashMap::from([
        (7, constrained_worker),
        (8, ModelRuntimeConfig::default()),
        (9, ModelRuntimeConfig::default()),
    ]);
    let (router, runtime) =
        router_with_worker_configs(Some(Duration::from_secs(10)), workers).await;
    let session_id = SessionAffinityId::new("migration-exclusion");
    let original_target = AffinityTarget {
        worker_id: 7,
        dp_rank: Some(0),
    };
    let AffinityAcquire::Initialize(initializer) = router
        .affinity
        .as_ref()
        .unwrap()
        .acquire(&session_id, None)
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(original_target).unwrap());

    let mut retry_input = request();
    retry_input.routing_mut().allowed_worker_ids = Some(HashSet::from([7, 8]));
    retry_input.migration_state = Some(Default::default());
    retry_input
        .migration_state
        .as_ref()
        .unwrap()
        .record_failure(
            7,
            Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("worker 7 overloaded")
                    .build(),
            ),
        );
    let mut retry_request = Context::new(retry_input);
    retry_request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id);

    let (selection, operation) = router
        .select_with_affinity(&retry_request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    assert_eq!(selection.worker.worker_id, 8);
    router.kv_router().free(retry_request.id()).await.unwrap();
    drop(operation);

    let mut exhausted_input = request();
    exhausted_input.routing_mut().allowed_worker_ids = Some(HashSet::from([7, 10]));
    exhausted_input.migration_state = Some(Default::default());
    exhausted_input
        .migration_state
        .as_ref()
        .unwrap()
        .record_failure(
            7,
            Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("worker 7 overloaded")
                    .build(),
            ),
        );
    let exhausted_request = Context::new(exhausted_input);
    let Err(error) = router
        .select_with_affinity(&exhausted_request, RequestPhase::Aggregated, false)
        .await
    else {
        panic!("exhausting the constrained worker set must reject the retry");
    };
    assert!(match_error_chain(
        error.as_ref(),
        &[ErrorType::ResourceExhausted],
        &[]
    ));

    let mut constrained_input = request();
    constrained_input.routing_mut().routing_constraints = Some(RoutingConstraints {
        required_taints: HashSet::from(["retry-pool".to_string()]),
        ..Default::default()
    });
    constrained_input.migration_state = Some(Default::default());
    constrained_input
        .migration_state
        .as_ref()
        .unwrap()
        .record_failure(
            7,
            Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("worker 7 overloaded")
                    .build(),
            ),
        );
    let constrained_request = Context::new(constrained_input);
    let Err(error) = router
        .select_with_affinity(&constrained_request, RequestPhase::Aggregated, false)
        .await
    else {
        panic!("routing constraints must not be widened during retry");
    };
    assert!(match_error_chain(
        error.as_ref(),
        &[ErrorType::ResourceExhausted],
        &[]
    ));

    let mut pinned_input = request();
    let routing = pinned_input.routing_mut();
    routing.backend_instance_id = Some(7);
    routing.dp_rank = Some(0);
    pinned_input.migration_state = Some(Default::default());
    pinned_input
        .migration_state
        .as_ref()
        .unwrap()
        .record_failure(7, None);
    let pinned_request = Context::new(pinned_input);
    let (selection, _) = router
        .select_with_affinity(&pinned_request, RequestPhase::Aggregated, true)
        .await
        .unwrap();
    assert_eq!(selection.worker.worker_id, 7);

    drop(router);
    runtime.shutdown();
}

#[derive(Default)]
struct RejectFirstDispatch {
    attempts: Mutex<Vec<(u64, Vec<u64>)>>,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for RejectFirstDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (request, _, instance) = addressed.into_parts();
        let worker_id = instance.expect("selected worker instance").id();
        let excluded_worker_ids = request
            .migration_state
            .as_ref()
            .map(|state| state.excluded_worker_ids())
            .unwrap_or_default();
        let attempt = {
            let mut attempts = self.attempts.lock().unwrap();
            attempts.push((worker_id, excluded_worker_ids));
            attempts.len()
        };

        if attempt == 1 {
            let output = Annotated {
                data: None,
                id: None,
                event: Some("error".to_string()),
                comment: None,
                error: Some(
                    DynamoError::builder()
                        .error_type(ErrorType::WorkerOverloaded)
                        .message("selected worker is overloaded")
                        .build(),
                ),
            };
            return Ok(ResponseStream::new(
                Box::pin(stream::once(async move { output })),
                context.context(),
            ));
        }

        let output = Annotated::from_data(LLMEngineOutput {
            token_ids: vec![2],
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        });
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { output })),
            context.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("the KV router dispatches unary requests")
    }
}

#[tokio::test]
#[serial_test::serial]
async fn worker_overload_stream_migration_releases_and_reselects() {
    async fn shared_drt(runtime: Runtime, store_path: &std::path::Path) -> DistributedRuntime {
        DistributedRuntime::new(
            runtime,
            DistributedConfig {
                discovery_backend: DiscoveryBackend::KvStore(Selector::File(
                    store_path.to_path_buf(),
                )),
                nats_config: None,
                request_plane: RequestPlaneMode::Tcp,
                event_transport_kind: EventTransportKind::Zmq,
            },
        )
        .await
        .unwrap()
    }

    let runtime = Runtime::from_current().unwrap();
    let store = tempfile::tempdir().unwrap();
    let router_drt = shared_drt(runtime.clone(), store.path()).await;
    let first_worker_drt = shared_drt(runtime.clone(), store.path()).await;
    let second_worker_drt = shared_drt(runtime.clone(), store.path()).await;
    let namespace = "worker-overload-migration";
    let endpoint_for = |drt: &DistributedRuntime| {
        drt.namespace(namespace.to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate")
    };
    let first_worker_endpoint = endpoint_for(&first_worker_drt);
    let second_worker_endpoint = endpoint_for(&second_worker_drt);
    first_worker_endpoint
        .register_endpoint_instance()
        .await
        .unwrap();
    second_worker_endpoint
        .register_endpoint_instance()
        .await
        .unwrap();

    let endpoint = endpoint_for(&router_drt);
    let client = endpoint.client().await.unwrap();
    let load_context = test_load_context(&client).await;
    let instances = tokio::time::timeout(Duration::from_secs(5), async {
        let mut source = client.instance_source.as_ref().clone();
        loop {
            let instances = source.borrow_and_update().clone();
            if instances.len() == 2 {
                return instances;
            }
            source.changed().await.unwrap();
        }
    })
    .await
    .expect("both workers must be discovered");
    let registered_ids = instances
        .into_iter()
        .map(|instance| instance.id())
        .collect::<HashSet<_>>();
    assert_eq!(registered_ids.len(), 2);

    let workers = registered_ids
        .iter()
        .copied()
        .map(|worker_id| (worker_id, ModelRuntimeConfig::default()))
        .collect::<HashMap<_, _>>();
    let (_workers_tx, workers) = watch::channel(workers);
    let config = KvRouterConfig {
        skip_initial_worker_wait: true,
        use_kv_events: false,
        router_track_active_blocks: false,
        ..Default::default()
    };
    let classifier_calls = Arc::new(AtomicUsize::new(0));
    let (classifier_observations_tx, mut classifier_observations_rx) = mpsc::unbounded_channel();
    let chooser = KvRouter::new_with_worker_role_and_scheduler_load(
        endpoint,
        client.clone(),
        workers,
        None,
        16,
        DefaultWorkerSelector::new(Some(config.clone()), "decode"),
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
    .unwrap()
    .with_request_classifier(RecordingClassifier {
        calls: Arc::clone(&classifier_calls),
        observations: classifier_observations_tx,
    })
    .unwrap();
    let dispatch = Arc::new(RejectFirstDispatch::default());
    let push_router =
        PushRouter::from_client_with_dispatch(client.clone(), RouterMode::KV, dispatch.clone())
            .await
            .unwrap();
    let chooser = Arc::new(chooser);
    let kv_router = Arc::new(
        RoutingHost::new_with_load_context(push_router, chooser.clone(), load_context, None)
            .unwrap(),
    );
    let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> = kv_router;
    let migration = Migration::new(1, None, "test".to_string(), Arc::new(Metrics::new()));

    let responses: Vec<_> = migration
        .generate(Context::new(request()), next)
        .await
        .unwrap()
        .collect()
        .await;

    assert_eq!(responses.len(), 1);
    assert!(responses[0].error.is_none());
    assert_eq!(responses[0].data.as_ref().unwrap().token_ids, vec![2]);
    assert_eq!(classifier_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        classifier_observations_rx.recv().await,
        Some(ClassifierObservation::Completed(2))
    );
    assert!(classifier_observations_rx.try_recv().is_err());
    let attempts = {
        let attempts = dispatch.attempts.lock().unwrap();
        attempts.clone()
    };
    assert_eq!(attempts.len(), 2);
    let failed_worker = attempts[0].0;
    let retried_worker = attempts[1].0;
    assert_ne!(failed_worker, retried_worker);
    assert!(registered_ids.contains(&failed_worker));
    assert!(registered_ids.contains(&retried_worker));
    assert!(attempts[0].1.is_empty());
    assert_eq!(attempts[1].1, vec![failed_worker]);
    let loads = chooser
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(
        loads.iter().all(|load| load.active_requests == 0),
        "all scheduler bookings must be released after migration: {loads:?}"
    );
    runtime.shutdown();
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use dynamo_kv_router::{
    DefaultWorkerSelector, WorkerSelectionPolicy, config::KvRouterConfig,
    protocols::RoutingConstraints,
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::{Client, Instance},
    discovery::EventTransportKind,
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    error::{BackendError, ErrorType, match_error_chain},
    pipeline::{
        AddressedRequest, AsyncEngineContext, Context, ManyIn, Operator, PushRouter, RouterMode,
        ServerStreamingEngine, StreamingDispatch, context::Controller,
    },
    storage::kv::Selector,
    traits::DistributedRuntimeProvider,
};
use tokio::sync::watch;

use super::*;
use crate::{
    http::service::metrics::Metrics,
    kv_router::RoutingLoadContext,
    local_model::runtime_config::ModelRuntimeConfig,
    lora::{LoraReplicaConfig, LoraRoutingTable, LoraStateTracker},
    migration::Migration,
    protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
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
fn classify_response_item_separates_terminal_failures_from_healthy_frames() {
    let outcome =
        |output: &LLMEngineOutput| classify_response_item(&Annotated::from_data(output.clone()));

    let mut output = LLMEngineOutput::default();
    assert!(matches!(outcome(&output), ResponseItemOutcome::Healthy));

    // Carries only a bare message, so migration has nothing to act on: drain for a typed error.
    output.finish_reason = Some(FinishReason::Error("decode failed".to_string()));
    assert!(matches!(
        outcome(&output),
        ResponseItemOutcome::DrainableTerminal
    ));

    output.finish_reason = Some(FinishReason::Cancelled);
    assert!(matches!(
        outcome(&output),
        ResponseItemOutcome::DrainableTerminal
    ));

    output.finish_reason = Some(FinishReason::Length);
    assert!(matches!(outcome(&output), ResponseItemOutcome::Healthy));
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
    let affinity = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin_with_coordinator(
        inner,
        load_context,
        Some(affinity),
        crate::session_affinity::SessionAffinityMode::Hard,
    )
    .unwrap();

    let error = host
        .generate(affinity_request("direct-unbound", None))
        .await
        .unwrap_err();
    assert!(match_error_chain(
        error.as_ref(),
        &[ErrorType::InvalidArgument],
        &[]
    ));

    drop(host);
    runtime.shutdown();
}

#[tokio::test]
async fn builtin_direct_uses_bound_soft_affinity_as_exact_target() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-direct-soft-affinity".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker_id = client.wait_for_instances().await.unwrap()[0].id();
    let load_context = test_load_context(&client).await;
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
        crate::session_affinity::SessionAffinityMode::Soft,
    )
    .unwrap();
    let session_id = SessionAffinityId::new("direct-soft-bound");
    bind_affinity_target(&host, &session_id, AffinityTarget::worker(worker_id)).await;

    let mut stream = host
        .generate(affinity_request("direct-soft-bound", None))
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    assert_eq!(dispatch.worker_ids.lock().unwrap().as_slice(), &[worker_id]);
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
async fn builtin_hard_affinity_ignores_local_inhibition() {
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
    let affinity = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin_with_coordinator(
        inner,
        load_context,
        Some(affinity.clone()),
        crate::session_affinity::SessionAffinityMode::Hard,
    )
    .unwrap();

    let session_id = SessionAffinityId::new("local-inhibition");
    let AffinityAcquire::Initialize(initializer) =
        affinity.acquire(&session_id, None).await.unwrap()
    else {
        panic!("new affinity session must initialize");
    };
    drop(
        initializer
            .commit(AffinityTarget::worker(worker_id))
            .unwrap(),
    );

    client.report_instance_down(worker_id);
    assert!(client.instance_ids().contains(&worker_id));
    assert!(!client.instance_ids_avail().contains(&worker_id));

    let mut stream = host
        .generate(affinity_request("local-inhibition", None))
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    assert_eq!(dispatch.worker_ids.lock().unwrap().as_slice(), &[worker_id]);
    assert_eq!(
        affinity.query_target(&session_id, None).unwrap(),
        Some(AffinityTarget::worker(worker_id))
    );

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
        crate::session_affinity::SessionAffinityMode::Hard,
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
            crate::session_affinity::SessionAffinityMode::Hard,
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
async fn builtin_hard_affinity_ignores_overload_while_soft_affinity_falls_back() {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("builtin-affinity-modes".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker_id = client.wait_for_instances().await.unwrap()[0].id();
    let fallback_worker_id = worker_id + 1;
    client.override_instance_avail(vec![worker_id, fallback_worker_id]);
    client.set_overloaded_instances(&[worker_id]);
    let load_context = test_load_context(&client).await;
    let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin(inner, load_context).unwrap();
    let request = Context::new(request());

    let hard = host
        .select_hosted_worker(&request, Some(AffinityTarget::worker(worker_id)), None)
        .unwrap();
    let soft = host
        .select_hosted_worker(&request, None, Some(AffinityTarget::worker(worker_id)))
        .unwrap();

    assert_eq!(hard.initial_worker, worker_id);
    assert_eq!(
        hard.target_constraint,
        Some(AffinityTarget::worker(worker_id))
    );
    assert_eq!(soft.initial_worker, fallback_worker_id);
    assert_eq!(soft.target_constraint, None);

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
        crate::session_affinity::SessionAffinityMode::Hard,
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
        dynamo_kv_router::scheduling::AdmissionAttempt::Untracked,
        &request(),
    );
    let monitored = monitor_response_stream(source, context, guard);
    tokio::pin!(monitored);

    assert!(monitored.next().await.is_some());
    assert!(monitored.next().await.is_none());
    assert!(drained.load(Ordering::Acquire));

    drop(router);
    runtime.shutdown();
}

fn cancelled_frame() -> Annotated<LLMEngineOutput> {
    Annotated::from_data(LLMEngineOutput {
        finish_reason: Some(FinishReason::Cancelled),
        ..Default::default()
    })
}

fn engine_shutdown_frame() -> Annotated<LLMEngineOutput> {
    Annotated {
        data: None,
        id: None,
        event: Some("error".to_string()),
        comment: None,
        error: Some(
            DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::EngineShutdown))
                .message("engine is shutting down")
                .build(),
        ),
    }
}

fn is_engine_shutdown(item: &Annotated<LLMEngineOutput>) -> bool {
    item.error.as_ref().is_some_and(|error| {
        match_error_chain(
            error,
            &[ErrorType::Backend(BackendError::EngineShutdown)],
            &[],
        )
    })
}

/// A trailing `EngineShutdown` error must reach migration, and the `Cancelled` frame that
/// preceded it must not be forwarded once it does.
#[tokio::test]
#[serial_test::serial]
async fn shutdown_cancellation_drains_trailing_engine_shutdown_error() {
    let (router, runtime) = router(None).await;
    let context = Context::new(()).context();
    // Reaching this at all is what the old code could not do.
    let polled_past_cancel = Arc::new(AtomicBool::new(false));
    let source_polled = Arc::clone(&polled_past_cancel);
    let source = ResponseStream::new(
        Box::pin(async_stream::stream! {
            yield cancelled_frame();
            source_polled.store(true, Ordering::Release);
            yield engine_shutdown_frame();
        }),
        Arc::clone(&context),
    );
    let guard = RequestGuard::new_kv(
        Arc::clone(router.kv_router()),
        Arc::clone(&router.request_metrics),
        "shutdown-drain".to_string(),
        WorkerWithDpRank::from_worker_id(0),
        dynamo_kv_router::scheduling::AdmissionAttempt::Untracked,
        &request(),
    );
    let monitored = monitor_response_stream(source, context, guard);
    tokio::pin!(monitored);

    let shutdown = monitored
        .next()
        .await
        .expect("the trailing engine-shutdown error must not be swallowed");
    assert!(
        is_engine_shutdown(&shutdown),
        "migration keys off the typed error, got {:?}",
        shutdown.error
    );
    assert!(monitored.next().await.is_none());
    assert!(polled_past_cancel.load(Ordering::Acquire));

    drop(router);
    runtime.shutdown();
}

/// A client cancel arrives through the context, not as a frame, so it must still preempt.
#[tokio::test]
#[serial_test::serial]
async fn client_cancellation_still_ends_stream_without_draining() {
    let (router, runtime) = router(None).await;
    let controller = Controller::new("client-cancelled-drain".to_string());
    controller.stop();
    let cancelled_request = Context::with_controller((), controller);
    let context = cancelled_request.context();
    let source_polled = Arc::new(AtomicBool::new(false));
    let polled = Arc::clone(&source_polled);
    let source = ResponseStream::new(
        Box::pin(async_stream::stream! {
            polled.store(true, Ordering::Release);
            yield cancelled_frame();
            yield engine_shutdown_frame();
        }),
        Arc::clone(&context),
    );
    let guard = RequestGuard::new_kv(
        Arc::clone(router.kv_router()),
        Arc::clone(&router.request_metrics),
        "client-cancelled-drain".to_string(),
        WorkerWithDpRank::from_worker_id(0),
        dynamo_kv_router::scheduling::AdmissionAttempt::Untracked,
        &request(),
    );
    let monitored = monitor_response_stream(source, context, guard);
    tokio::pin!(monitored);

    assert!(monitored.next().await.is_none());
    assert!(
        !source_polled.load(Ordering::Acquire),
        "a client cancel must not read further worker frames"
    );

    drop(router);
    runtime.shutdown();
}

/// A worker that sends a terminal frame and then stops talking without closing the transport
/// must not hold the request open; the drain is bounded.
#[tokio::test]
#[serial_test::serial]
async fn drain_without_trailing_error_gives_up_at_the_deadline() {
    let (router, runtime) = router(None).await;
    // Auto-advances the drain deadline once the task idles, so this costs no wall clock.
    tokio::time::pause();
    let context = Context::new(()).context();
    let source = ResponseStream::new(
        Box::pin(async_stream::stream! {
            yield cancelled_frame();
            // Never ends and never sends the error: the transport is wedged open.
            std::future::pending::<()>().await;
        }),
        Arc::clone(&context),
    );
    let guard = RequestGuard::new_kv(
        Arc::clone(router.kv_router()),
        Arc::clone(&router.request_metrics),
        "shutdown-drain-deadline".to_string(),
        WorkerWithDpRank::from_worker_id(0),
        dynamo_kv_router::scheduling::AdmissionAttempt::Untracked,
        &request(),
    );
    let monitored = monitor_response_stream(source, context, guard);
    tokio::pin!(monitored);

    // Generous relative to DRAIN_TIMEOUT: the assertion is that the drain is bounded at all.
    let deadline = DRAIN_TIMEOUT * 4;
    let item = tokio::time::timeout(deadline, monitored.next())
        .await
        .expect("the drain must give up at DRAIN_TIMEOUT")
        .expect("the withheld frame must be released once the drain gives up");
    assert!(matches!(
        item.data
            .as_ref()
            .and_then(|data| data.finish_reason.as_ref()),
        Some(FinishReason::Cancelled)
    ));
    assert!(
        tokio::time::timeout(deadline, monitored.next())
            .await
            .expect("the stream must end after the drain gives up")
            .is_none()
    );

    drop(router);
    runtime.shutdown();
}

/// The drain window must actually be armed for DRAIN_TIMEOUT, not merely exist. A worker
/// that sends its typed error a scheduler tick after the `Cancelled` frame -- well inside
/// the window -- must still have that error reach migration, and the withheld frame must
/// not be published ahead of it. Removing the `drain_deadline` reset leaves the deadline at
/// its already-elapsed initial value, which ends the stream on the `Cancelled` frame and
/// fails this test.
#[tokio::test]
#[serial_test::serial]
async fn trailing_error_within_the_drain_window_still_reaches_migration() {
    let (router, runtime) = router(None).await;
    tokio::time::pause();
    let context = Context::new(()).context();
    let source = ResponseStream::new(
        Box::pin(async_stream::stream! {
            yield cancelled_frame();
            // Comfortably inside the window: the drain must still be open here.
            tokio::time::sleep(DRAIN_TIMEOUT / 2).await;
            yield engine_shutdown_frame();
        }),
        Arc::clone(&context),
    );
    let guard = RequestGuard::new_kv(
        Arc::clone(router.kv_router()),
        Arc::clone(&router.request_metrics),
        "drain-window-armed".to_string(),
        WorkerWithDpRank::from_worker_id(0),
        dynamo_kv_router::scheduling::AdmissionAttempt::Untracked,
        &request(),
    );
    let monitored = monitor_response_stream(source, context, guard);
    tokio::pin!(monitored);

    let first = monitored
        .next()
        .await
        .expect("the trailing engine-shutdown error must survive the drain window");
    assert!(
        is_engine_shutdown(&first),
        "migration keys off the typed error, and the withheld cancel frame must not \
         precede it; got {first:?}"
    );
    assert!(
        monitored.next().await.is_none(),
        "the shutdown error is the whole response"
    );

    drop(router);
    runtime.shutdown();
}

/// `biased` polls the stream arm before the drain deadline, so a worker that stays
/// continuously ready with terminal frames must not be able to postpone the deadline
/// forever. The drain is armed once and checked against the clock on every frame.
#[tokio::test]
#[serial_test::serial]
async fn always_ready_terminals_cannot_starve_the_drain_deadline() {
    let (router, runtime) = router(None).await;
    tokio::time::pause();
    let context = Context::new(()).context();
    let source = ResponseStream::new(
        Box::pin(async_stream::stream! {
            loop {
                yield cancelled_frame();
            }
        }),
        Arc::clone(&context),
    );
    let guard = RequestGuard::new_kv(
        Arc::clone(router.kv_router()),
        Arc::clone(&router.request_metrics),
        "starvation-guard".to_string(),
        WorkerWithDpRank::from_worker_id(0),
        dynamo_kv_router::scheduling::AdmissionAttempt::Untracked,
        &request(),
    );
    let monitored = monitor_response_stream(source, context, guard);
    tokio::pin!(monitored);

    // Paused time does not auto-advance while the source is always ready, so the consumer
    // moves the clock past the deadline itself.
    let mut frames = 0u32;
    while monitored.next().await.is_some() {
        frames += 1;
        if frames == 8 {
            tokio::time::advance(DRAIN_TIMEOUT + Duration::from_millis(1)).await;
        }
        assert!(
            frames < 1_000,
            "the drain deadline was starved by an always-ready terminal stream"
        );
    }

    drop(router);
    runtime.shutdown();
}

/// Transport EOF ends the drain and releases the booking.
#[tokio::test]
#[serial_test::serial]
async fn shutdown_cancellation_without_trailing_error_still_aborts() {
    let (router, runtime) = router(None).await;
    let context_id = "shutdown-cancel-without-error".to_string();
    let cancelled_request =
        Context::with_id_and_metadata(request(), context_id.clone(), Default::default());
    let (mut selection, _) = router
        .select_with_affinity(&cancelled_request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    let cancelled_worker = selection.worker;
    let guard = router
        .track_selection(&cancelled_request, &mut selection, false)
        .await
        .unwrap();
    let source = ResponseStream::new(
        Box::pin(stream::once(async { cancelled_frame() })),
        cancelled_request.context().clone(),
    );
    let monitored = monitor_response_stream(source, cancelled_request.context().clone(), guard);
    tokio::pin!(monitored);

    // Below DRAIN_TIMEOUT so only transport EOF can satisfy these assertions: at 10s the
    // test would also pass if the drain deadline, not EOF, had ended the stream.
    let eof_bound = DRAIN_TIMEOUT / 2;
    let item = tokio::time::timeout(eof_bound, monitored.next())
        .await
        .expect("the drain must end at transport EOF, not block on a trailing error")
        .expect("the cancelled frame must be yielded once EOF proves it was the last");
    assert!(matches!(
        item.data
            .as_ref()
            .and_then(|data| data.finish_reason.as_ref()),
        Some(FinishReason::Cancelled)
    ));
    assert!(
        tokio::time::timeout(eof_bound, monitored.next())
            .await
            .expect("the drain must end at transport EOF, not wait for a trailing error")
            .is_none()
    );

    let retry_request =
        Context::with_id_and_metadata(request(), context_id.clone(), Default::default());
    let (mut retry_selection, _) = router
        .select_with_affinity(&retry_request, RequestPhase::Aggregated, false)
        .await
        .unwrap();
    assert_eq!(retry_selection.worker, cancelled_worker);
    let mut retry_guard = router
        .track_selection(&retry_request, &mut retry_selection, false)
        .await
        .expect("the booking must be released when the drain reaches transport EOF");
    retry_guard.abort().await;

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
#[serial_test::serial]
async fn route_plan_from_preview_holds_and_releases_the_decode_reservation() {
    let (router, runtime) = router(None).await;
    let request = Context::new(request());
    let requests_started_before = router.request_metrics.requests_started_total().get();

    let preview = router
        .preview_kv_route(&request, RequestPhase::Decode)
        .await
        .expect("decode preview should select one request");
    let plan = router
        .plan_kv_route_from_preview(&request, preview)
        .await
        .expect("decode plan should admit one request");
    assert_eq!(plan.signals().worker.worker_id, 7);
    assert_eq!(
        router.request_metrics.requests_started_total().get(),
        requests_started_before,
        "a topology decision is not a started request"
    );
    let admitted_loads = router
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert_eq!(
        admitted_loads
            .iter()
            .find(|load| load.worker_id == 7 && load.dp_rank == 0)
            .expect("selected worker must be reported")
            .active_requests,
        1
    );

    plan.abort().await;
    let released_loads = router
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(
        released_loads.iter().all(|load| load.active_requests == 0),
        "abandoned plans must release their scheduler reservation: {released_loads:?}"
    );
    assert_eq!(
        router.request_metrics.requests_started_total().get(),
        requests_started_before,
        "an abandoned plan must not count as a started request"
    );

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn route_preview_does_not_admit_a_request() {
    let (router, runtime) = router(None).await;
    let request = Context::new(request());
    let requests_started_before = router.request_metrics.requests_started_total().get();

    let preview = router
        .preview_kv_route(&request, RequestPhase::Decode)
        .await
        .expect("decode preview should select one request");
    assert_eq!(preview.signals().worker.worker_id, 7);
    let loads = router
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(loads.iter().all(|load| load.active_requests == 0));
    assert_eq!(
        router.request_metrics.requests_started_total().get(),
        requests_started_before
    );

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn route_plan_from_preview_admits_the_previewed_worker() {
    let (router, runtime) = router_with_workers(None, &[7, 8]).await;
    let request = Context::new(request());
    let preview = router
        .preview_kv_route(&request, RequestPhase::Decode)
        .await
        .unwrap();
    let previewed_worker = preview.signals().worker;

    let plan = router
        .plan_kv_route_from_preview(&request, preview)
        .await
        .unwrap();
    assert_eq!(plan.signals().worker, previewed_worker);
    let loads = router
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert_eq!(
        loads
            .iter()
            .find(|load| {
                load.worker_id == previewed_worker.worker_id
                    && load.dp_rank == previewed_worker.dp_rank
            })
            .expect("previewed worker must be reported")
            .active_requests,
        1
    );
    assert_eq!(
        loads.iter().filter(|load| load.active_requests > 0).count(),
        1
    );

    plan.abort().await;
    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn route_preview_does_not_acquire_session_affinity() {
    let (router, runtime) = router(Some(Duration::from_secs(10))).await;
    let session_id = SessionAffinityId::new("preview-only");
    let mut request = Context::new(request());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());

    router
        .preview_kv_route(&request, RequestPhase::Decode)
        .await
        .unwrap();

    let affinity = router.affinity.as_ref().unwrap();
    let acquisition = tokio::time::timeout(
        Duration::from_millis(100),
        affinity.acquire(&session_id, None),
    )
    .await
    .expect("preview must not leave affinity initialization pending")
    .unwrap();
    assert!(matches!(acquisition, AffinityAcquire::Initialize(_)));
    drop(acquisition);

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn planned_dispatch_transfers_the_reservation_to_request_cleanup() {
    let (router, runtime) = router(None).await;
    let requests_started_before = router.request_metrics.requests_started_total().get();
    let request = Context::new(request());
    let preview = router
        .preview_kv_route(&request, RequestPhase::Decode)
        .await
        .unwrap();
    let plan = router
        .plan_kv_route_from_preview(&request, preview)
        .await
        .unwrap();

    assert!(router.dispatch_kv_plan(request, plan).await.is_err());
    assert_eq!(
        router.request_metrics.requests_started_total().get(),
        requests_started_before + 1
    );
    let loads = router
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(loads.iter().all(|load| load.active_requests == 0));

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
#[serial_test::serial]
async fn prefill_busy_probe_does_not_admit_a_request() {
    let (router, runtime) = router(None).await;
    let request = Context::new(request());
    let requests_started_before = router.request_metrics.requests_started_total().get();

    assert!(!router.prefill_worker_busy(&request, 0.5).await.unwrap());
    let loads = router
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(loads.iter().all(|load| load.active_requests == 0));
    assert_eq!(
        router.request_metrics.requests_started_total().get(),
        requests_started_before
    );

    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn aborted_route_plan_drops_pending_affinity_initialization() {
    let (router, runtime) = router(Some(Duration::from_secs(10))).await;
    let session_id = SessionAffinityId::new("abandoned-route-plan");
    let mut request = Context::new(request());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());

    let preview = router
        .preview_kv_route(&request, RequestPhase::Decode)
        .await
        .unwrap();
    router
        .plan_kv_route_from_preview(&request, preview)
        .await
        .unwrap()
        .abort()
        .await;

    let affinity = router.affinity.as_ref().unwrap();
    let acquisition = tokio::time::timeout(
        Duration::from_millis(100),
        affinity.acquire(&session_id, None),
    )
    .await
    .expect("abandoned plan must not leave affinity initialization pending")
    .unwrap();
    assert!(matches!(acquisition, AffinityAcquire::Initialize(_)));
    drop(acquisition);

    drop(router);
    runtime.shutdown();
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
async fn session_affinity_post_selection_failures_preserve_binding() {
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

    let operation = Some(affinity.acquire(&session_id, None).await.unwrap());
    drop(operation);
    assert_eq!(
        affinity.query_target(&session_id, None).unwrap(),
        Some(original_target)
    );

    let operation = Some(affinity.acquire(&session_id, None).await.unwrap());
    drop(operation);
    assert_eq!(
        affinity.query_target(&session_id, None).unwrap(),
        Some(original_target)
    );

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
async fn stale_affinity_rank_recovers_within_request() {
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
async fn config_watch_gap_preserves_hard_affinity() {
    let (router, runtime) = router(Some(Duration::from_secs(10))).await;
    let session_id = SessionAffinityId::new("config-watch-gap");
    let target = AffinityTarget::new(8, Some(0));
    router
        .inner
        .client
        .override_discovered_instances(vec![7, target.worker_id]);
    router
        .inner
        .client
        .override_instance_avail(vec![7, target.worker_id]);
    bind_affinity_target(&router, &session_id, target).await;
    assert!(router.affinity_target_is_valid(target));

    let mut request = Context::new(request());
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());
    assert!(
        router
            .select_with_affinity(&request, RequestPhase::Aggregated, false)
            .await
            .is_err()
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

#[tokio::test]
async fn migration_exclusion_preserves_hard_affinity_without_widening_or_escaping_hard_pins() {
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
    retry_request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());

    let Err(error) = router
        .select_with_affinity(&retry_request, RequestPhase::Aggregated, false)
        .await
    else {
        panic!("migration exclusions must not rebind hard affinity");
    };
    assert!(error.to_string().contains("worker 7"));
    assert_eq!(
        router
            .affinity
            .as_ref()
            .unwrap()
            .query_target(&session_id, None)
            .unwrap(),
        Some(original_target)
    );

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

/// A two-worker routing host wired to a caller-supplied dispatch, for driving migration
/// end to end by choosing what the first worker answers with.
struct MigrationHarness {
    runtime: Runtime,
    chooser: Arc<KvRouter>,
    engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    registered_ids: HashSet<u64>,
    // Kept alive: dropping these tears down discovery for the workers the router must see.
    _store: tempfile::TempDir,
    _drts: Vec<DistributedRuntime>,
}

async fn two_worker_migration_harness(
    namespace: &str,
    dispatch: Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>>,
) -> MigrationHarness {
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
    .unwrap();
    let push_router =
        PushRouter::from_client_with_dispatch(client.clone(), RouterMode::KV, dispatch)
            .await
            .unwrap();
    let chooser = Arc::new(chooser);
    let kv_router = Arc::new(
        RoutingHost::new_with_load_context(
            push_router,
            chooser.clone(),
            load_context,
            None,
            crate::session_affinity::SessionAffinityMode::Hard,
        )
        .unwrap(),
    );

    MigrationHarness {
        runtime,
        chooser,
        engine: kv_router,
        registered_ids,
        _store: store,
        _drts: vec![router_drt, first_worker_drt, second_worker_drt],
    }
}

#[tokio::test]
#[serial_test::serial]
async fn worker_overload_stream_migration_releases_and_reselects() {
    let dispatch = Arc::new(RejectFirstDispatch::default());
    let harness = two_worker_migration_harness("worker-overload-migration", dispatch.clone()).await;
    let MigrationHarness {
        runtime,
        chooser,
        engine: next,
        registered_ids,
        ..
    } = &harness;
    let migration = Migration::new(1, None, "test".to_string(), Arc::new(Metrics::new()));

    let responses: Vec<_> = migration
        .generate(Context::new(request()), next.clone())
        .await
        .unwrap()
        .collect()
        .await;

    assert_eq!(responses.len(), 1);
    assert!(responses[0].error.is_none());
    assert_eq!(responses[0].data.as_ref().unwrap().token_ids, vec![2]);
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

/// A gracefully shutting-down worker: `Cancelled` data frame, then the reason.
#[derive(Default)]
struct ShutdownAfterCancelDispatch {
    attempts: Mutex<Vec<(u64, Vec<u64>)>>,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>
    for ShutdownAfterCancelDispatch
{
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
            return Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![
                    cancelled_frame(),
                    engine_shutdown_frame(),
                ])),
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

/// The shutdown error must reach migration, which is the only layer that can move the request.
#[tokio::test]
#[serial_test::serial]
async fn engine_shutdown_after_cancel_frame_migrates_and_reselects() {
    let dispatch = Arc::new(ShutdownAfterCancelDispatch::default());
    let harness = two_worker_migration_harness("engine-shutdown-migration", dispatch.clone()).await;
    let migration = Migration::new(1, None, "test".to_string(), Arc::new(Metrics::new()));

    let responses: Vec<_> = migration
        .generate(Context::new(request()), harness.engine.clone())
        .await
        .unwrap()
        .collect()
        .await;

    let attempts = {
        let attempts = dispatch.attempts.lock().unwrap();
        attempts.clone()
    };
    assert_eq!(attempts.len(), 2, "the shutdown must trigger a migration");
    let failed_worker = attempts[0].0;
    let retried_worker = attempts[1].0;
    assert_ne!(failed_worker, retried_worker);
    assert!(harness.registered_ids.contains(&failed_worker));
    assert!(harness.registered_ids.contains(&retried_worker));
    assert!(attempts[0].1.is_empty());
    assert_eq!(attempts[1].1, vec![failed_worker]);
    assert_eq!(
        responses.len(),
        1,
        "a migrated request must read as one clean stream; the aborted attempt's \
         terminal frame must not surface ahead of the retry: {responses:?}"
    );
    let last = responses.last().expect("the retry must produce output");
    assert!(last.error.is_none());
    assert_eq!(last.data.as_ref().unwrap().token_ids, vec![2]);
    assert_eq!(
        last.data.as_ref().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
    let loads = harness
        .chooser
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(
        loads.iter().all(|load| load.active_requests == 0),
        "all scheduler bookings must be released after migration: {loads:?}"
    );
    harness.runtime.shutdown();
}

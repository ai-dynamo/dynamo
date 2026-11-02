// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `PrefillRouter::schedule` as a caller that never dispatches uses it (the
//! inference-gateway EPP), and the prefill binding's worker availability.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::local_model::runtime_config::ModelRuntimeConfig;
use async_trait::async_trait;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Instance,
    discovery::{EndpointInstanceId, EventTransportKind},
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    engine::{AsyncEngine, AsyncEngineContext},
    pipeline::{
        AddressedRequest, Context, Error, ManyIn, ManyOut, ResponseStream, RouterMode, SingleIn,
        StreamingDispatch, context::Controller,
    },
    storage::kv,
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use tokio::sync::watch;

use super::*;
use crate::{
    discovery::ModelManager,
    kv_router::{
        KvRouter, RouteBooking, RouterLoadSource, RoutingHost, RoutingLoadContext,
        route_request::TestRouteRequest, stage_router::StageRouter,
    },
    protocols::common::{
        FinishReason, llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest,
    },
    worker_type::WorkerType,
};

type LlmResponse = dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>;

struct RecordingDispatch {
    worker_ids: Mutex<Vec<u64>>,
    pending_responses: bool,
}

impl RecordingDispatch {
    fn completed() -> Self {
        Self {
            worker_ids: Mutex::new(Vec::new()),
            pending_responses: false,
        }
    }

    fn pending() -> Self {
        Self {
            worker_ids: Mutex::new(Vec::new()),
            pending_responses: true,
        }
    }

    fn response_stream(&self) -> ManyOut<LlmResponse> {
        let context: Arc<dyn AsyncEngineContext> = Arc::new(Controller::default());
        if self.pending_responses {
            return ResponseStream::new(Box::pin(futures::stream::pending()), context);
        }
        ResponseStream::new(
            Box::pin(tokio_stream::iter(vec![LlmResponse::from_data(
                LLMEngineOutput {
                    finish_reason: Some(FinishReason::EoS),
                    ..Default::default()
                },
            )])),
            context,
        )
    }
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, LlmResponse> for RecordingDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        let (addressed, _) = request.transfer(());
        let (_, _, instance) = addressed.into_parts();
        self.worker_ids
            .lock()
            .unwrap()
            .push(instance.expect("selected instance").id());
        Ok(self.response_stream())
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        anyhow::bail!("bidirectional dispatch is unused in this test")
    }

    async fn on_instance_removed(&self, _id: &EndpointInstanceId) {}
}

fn distributed_config(root: &std::path::Path) -> DistributedConfig {
    DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(root.to_path_buf())),
        nats_config: None,
        request_plane: RequestPlaneMode::Tcp,
        response_plane: None,
        event_transport_kind: EventTransportKind::Zmq,
    }
}

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .build()
        .unwrap()
}

/// Schedule a prefill worker and keep its booking without dispatching, as
/// the EPP does.
async fn book(
    router: &PrefillRouter,
    request: &dyn crate::kv_router::RouteRequest,
) -> (AffinityTarget, RouteBooking) {
    let route = router
        .schedule(request)
        .await
        .unwrap()
        .into_route()
        .expect("an active prefill router books a worker");
    (route.target(), route.into_booking())
}

async fn shared_router(
    runtime: &Runtime,
    discovery_root: &std::path::Path,
    namespace: &str,
    mode: RouterMode,
    dispatch: Arc<RecordingDispatch>,
) -> (
    Arc<RoutingHost>,
    Arc<PrefillRouter>,
    Vec<DistributedRuntime>,
    Vec<u64>,
) {
    let component = "workers";
    let endpoint_name = "generate";
    let mut worker_runtimes = Vec::new();
    for _ in 0..4 {
        let worker_runtime =
            DistributedRuntime::new(runtime.clone(), distributed_config(discovery_root))
                .await
                .unwrap();
        worker_runtime
            .namespace(namespace.to_string())
            .unwrap()
            .component(component.to_string())
            .unwrap()
            .endpoint(endpoint_name)
            .register_endpoint_instance()
            .await
            .unwrap();
        worker_runtimes.push(worker_runtime);
    }

    let router_runtime =
        DistributedRuntime::new(runtime.clone(), distributed_config(discovery_root))
            .await
            .unwrap();
    let client = router_runtime
        .namespace(namespace.to_string())
        .unwrap()
        .component(component.to_string())
        .unwrap()
        .endpoint(endpoint_name)
        .client()
        .await
        .unwrap();
    let instances = tokio::time::timeout(Duration::from_secs(5), async {
        let mut source = client.instance_source.as_ref().clone();
        loop {
            let instances = source.borrow_and_update().clone();
            if instances.len() == 4 {
                return instances;
            }
            source
                .changed()
                .await
                .expect("discovery source must remain open");
        }
    })
    .await
    .expect("all four workers must be discovered");
    let mut workers = instances.iter().map(Instance::id).collect::<Vec<_>>();
    workers.sort_unstable();
    let load_context = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::Prefill,
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        &client.endpoint.drt().child_token(),
        None,
    )
    .await
    .unwrap();
    let shared = Arc::new(
        RoutingHost::builder(client, mode)
            .load_context(load_context)
            .dispatch(dispatch as crate::kv_router::LlmDispatch)
            .build()
            .await
            .unwrap(),
    );
    let prefill = PrefillRouter::disabled(Arc::new(ModelManager::new()), mode, None);
    prefill.binding.store(Some(Arc::new(
        crate::kv_router::prefill_router::PrefillBinding {
            target_id: crate::discovery::WorkerSetTargetId::Legacy(
                dynamo_runtime::protocols::EndpointId {
                    namespace: namespace.to_string(),
                    component: component.to_string(),
                    name: endpoint_name.to_string(),
                },
            ),
            endpoint_id: dynamo_runtime::protocols::EndpointId {
                namespace: namespace.to_string(),
                component: component.to_string(),
                name: endpoint_name.to_string(),
            },
            router: shared.clone(),
            prefill_router_mode: mode,
        },
    )));
    prefill.lifecycle.store(
        PrefillLifecycleState::Active as u8,
        std::sync::atomic::Ordering::Release,
    );
    worker_runtimes.push(router_runtime);
    (shared, prefill, worker_runtimes, workers)
}

async fn tracked_binding(label: &str) -> (Arc<PrefillBinding>, Arc<KvRouter>) {
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime, DistributedConfig::process_local())
        .await
        .unwrap();
    let component = distributed
        .namespace(format!(
            "prefill-reservation-{label}-{}",
            uuid::Uuid::new_v4()
        ))
        .unwrap()
        .component("workers".to_string())
        .unwrap();
    let endpoint = component.endpoint("generate");
    let endpoint_id = endpoint.id();
    let client = endpoint.client().await.unwrap();
    let (_workers_tx, workers_rx) = watch::channel(HashMap::from([
        (7, ModelRuntimeConfig::default()),
        (8, ModelRuntimeConfig::default()),
    ]));
    let config = KvRouterConfig {
        overlap_score_credit: 0.0,
        router_temperature: 0.0,
        use_kv_events: false,
        router_track_active_blocks: false,
        router_track_prefill_tokens: true,
        skip_initial_worker_wait: true,
        ..Default::default()
    };
    let chooser = Arc::new(
        KvRouter::new_with_worker_role(
            endpoint,
            client.clone(),
            workers_rx,
            None,
            16,
            crate::kv_router::SelectionPolicySource::Registry,
            Some(config),
            None,
            Some(WorkerType::Prefill),
            "prefill",
            Some(format!("prefill-reservation-{label}")),
            false,
            None,
            None,
        )
        .await
        .unwrap(),
    );
    let router = RoutingHost::builder(client, RouterMode::KV)
        .kv_router(chooser.clone())
        .build()
        .await
        .unwrap();
    let binding = Arc::new(PrefillBinding {
        target_id: crate::discovery::WorkerSetTargetId::Legacy(endpoint_id.clone()),
        endpoint_id,
        router: Arc::new(router),
        prefill_router_mode: RouterMode::KV,
    });
    (binding, chooser)
}

async fn tracked_prefill_router() -> (Arc<PrefillRouter>, Arc<KvRouter>) {
    let (binding, chooser) = tracked_binding("primary").await;
    let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
    router.binding.store(Some(binding));
    router.lifecycle.store(
        PrefillLifecycleState::Active as u8,
        std::sync::atomic::Ordering::Release,
    );
    (router, chooser)
}

#[tokio::test]
async fn binding_availability_tracks_local_inhibition_and_cancellation() {
    use crate::kv_router::prefill_router::PrefillRouterLifecycle;

    let (router, chooser) = tracked_prefill_router().await;
    let endpoint = chooser.client().endpoint.id();
    chooser.client().override_discovered_instances(vec![7, 8]);
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint),
        Some(HashSet::from([7, 8]))
    );
    chooser.client().report_instance_down(7);
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint),
        Some(HashSet::from([8]))
    );
    let mut other = endpoint.clone();
    other.name = "other".into();
    assert!(router.available_worker_ids_for("legacy", &other).is_none());
    router.lifecycle.store(
        PrefillLifecycleState::Unavailable as u8,
        std::sync::atomic::Ordering::Release,
    );
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint),
        Some(HashSet::new())
    );
    router.lifecycle.store(
        PrefillLifecycleState::Active as u8,
        std::sync::atomic::Ordering::Release,
    );
    router.cancel_token.cancel();
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint),
        Some(HashSet::new())
    );
}

#[tokio::test]
async fn binding_availability_waits_for_rebind_publication() {
    use crate::kv_router::prefill_router::PrefillRouterLifecycle;
    use std::sync::mpsc;

    let (router, chooser) = tracked_prefill_router().await;
    let original = chooser.client().endpoint.id();
    chooser.client().override_discovered_instances(vec![7]);
    let (replacement, _) = tracked_binding("availability-replacement").await;
    let mut target = router.target.lock();
    *target = Some(crate::discovery::WorkerSetTargetId::Legacy(
        original.clone(),
    ));
    let (started_tx, started_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let reader_router = router.clone();
    let reader = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        result_tx
            .send(reader_router.available_worker_ids_for("legacy", &original))
            .unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_millis(100)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    *target = Some(replacement.target_id.clone());
    router.binding.store(Some(replacement));
    router.lifecycle.store(
        PrefillLifecycleState::Active as u8,
        std::sync::atomic::Ordering::Release,
    );
    drop(target);
    assert_eq!(
        result_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        None
    );
    reader.join().unwrap();
}

/// A kept prefill booking releases the exact scheduler booking it admitted,
/// even after the router rebinds to another prefill worker set.
#[tokio::test]
async fn kept_prefill_booking_releases_exactly_after_rebind() {
    let (router, original_chooser) = tracked_prefill_router().await;
    let (target, booking) = book(
        &router,
        &TestRouteRequest::new("epp-prefill/rebind", &[1u32; 64]),
    )
    .await;
    assert!(target.dp_rank.is_some());
    let (_, dropped_booking) = book(
        &router,
        &TestRouteRequest::new("epp-prefill/drop", &[1u32; 64]),
    )
    .await;

    let loads = original_chooser
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert!(
        loads.iter().any(|load| load.potential_prefill_tokens > 0),
        "booking did not add prefill load: {loads:?}"
    );

    let (replacement, _) = tracked_binding("replacement").await;
    router.binding.store(Some(replacement));
    booking.release().await;

    drop(dropped_booking);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let loads = original_chooser
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap();
            if loads.iter().all(|load| load.potential_prefill_tokens == 0) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("a dropped booking must release its admitted booking");
}

/// A round-robin prefill worker is chosen at `schedule`, as it is for a
/// dispatched request, so a kept booking advances the cursor the frontend's
/// sends share. A booking keeps nothing for a built-in mode.
#[tokio::test]
async fn rr_prefill_schedule_advances_the_dispatch_cursor() {
    let runtime = Runtime::from_current().unwrap();
    let discovery_root = tempfile::tempdir().unwrap();
    let namespace = "prefill-schedule-rr";
    let dispatch = Arc::new(RecordingDispatch::completed());
    let (shared_router, prefill_router, worker_runtimes, expected_workers) = shared_router(
        &runtime,
        discovery_root.path(),
        namespace,
        RouterMode::RoundRobin,
        dispatch.clone(),
    )
    .await;

    let (target, booking) = book(
        &prefill_router,
        &TestRouteRequest::new("non-kv-booking", &[1, 2, 3]),
    )
    .await;
    assert_eq!(target.worker_id, expected_workers[0]);
    assert_eq!(target.dp_rank, None);
    booking.release().await;

    for _ in &expected_workers {
        let mut stream = shared_router
            .generate(Context::new(request()))
            .await
            .unwrap();
        while stream.next().await.is_some() {}
    }

    let mut after_booking = expected_workers[1..].to_vec();
    after_booking.push(expected_workers[0]);
    assert_eq!(*dispatch.worker_ids.lock().unwrap(), after_booking);
    drop(worker_runtimes);
    runtime.shutdown();
}

/// A scheduled load-based prefill plan holds an occupancy reservation until
/// its booking is kept, which releases it: occupancy is not keyed by request.
#[tokio::test]
async fn kept_prefill_booking_releases_builtin_occupancy() {
    let runtime = Runtime::from_current().unwrap();
    let discovery_root = tempfile::tempdir().unwrap();

    for (index, mode) in [
        RouterMode::PowerOfTwoChoices,
        RouterMode::LeastLoaded,
        RouterMode::DeviceAwareWeighted,
    ]
    .into_iter()
    .enumerate()
    {
        let dispatch = Arc::new(RecordingDispatch::pending());
        let (shared, prefill, runtimes, workers) = shared_router(
            &runtime,
            discovery_root.path(),
            &format!("prefill-schedule-occupancy-{index}"),
            mode,
            dispatch,
        )
        .await;

        // Device-aware routing reads the frontend's request.
        let route = prefill
            .schedule(&SingleIn::new(request()))
            .await
            .unwrap()
            .into_route()
            .expect("an active prefill router selects a worker");
        assert_eq!(
            workers
                .iter()
                .map(|worker| shared.occupancy_for_test(*worker))
                .sum::<u64>(),
            1,
            "{mode:?} a scheduled plan holds its occupancy reservation"
        );
        route.into_booking().release().await;
        assert_eq!(
            workers
                .iter()
                .map(|worker| shared.occupancy_for_test(*worker))
                .sum::<u64>(),
            0,
            "{mode:?} keeping the booking releases built-in occupancy"
        );

        let stream = shared.generate(Context::new(request())).await.unwrap();
        assert_eq!(
            workers
                .iter()
                .map(|worker| shared.occupancy_for_test(*worker))
                .sum::<u64>(),
            1,
            "{mode:?} committed dispatch must retain exactly one lease"
        );
        drop(stream);
        assert_eq!(
            workers
                .iter()
                .map(|worker| shared.occupancy_for_test(*worker))
                .sum::<u64>(),
            0,
            "{mode:?} dropping the response stream must release the lease"
        );
        drop(runtimes);
    }

    runtime.shutdown();
}

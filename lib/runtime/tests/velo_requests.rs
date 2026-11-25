// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "common/selector.rs"]
mod selector;

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Error, Result};
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::{Client, Endpoint, StartedEndpoint, TransportType},
    discovery::{DiscoveryInstance, DiscoverySpec},
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    engine::{AsyncEngine, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType, match_error_chain},
    metrics::MetricsHierarchy,
    pipeline::{
        ManyOut, ResponseStream, SingleIn,
        network::{
            Ingress, ResponsePlaneMode,
            egress::push_router::{PushRouter, RouterMode},
        },
    },
    protocols::maybe_error::MaybeError,
    storage::kv,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Notify, Semaphore},
    time::timeout,
};

#[derive(Debug, Serialize, Deserialize)]
struct Item {
    value: usize,
    error: Option<DynamoError>,
}

impl MaybeError for Item {
    fn from_err(error: impl std::error::Error + 'static) -> Self {
        Self {
            value: 0,
            error: Some(DynamoError::from(
                Box::new(error) as Box<dyn std::error::Error>
            )),
        }
    }

    fn err(&self) -> Option<DynamoError> {
        self.error.clone()
    }
}

struct Worker {
    calls: AtomicUsize,
    entered: Notify,
    release: Semaphore,
}

impl Default for Worker {
    fn default() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            entered: Notify::new(),
            release: Semaphore::new(0),
        }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Item>, Error> for Worker {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Item>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (input, context) = input.into_parts();
        if input == "block" {
            self.entered.notify_one();
            self.release.acquire().await?.forget();
        }
        let len = input.len();
        let items = (0..3).map(move |index| Item {
            value: len + index,
            error: None,
        });
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(items)),
            context.context(),
        ))
    }
}

async fn runtime(
    owner: Runtime,
    root: &Path,
    request: RequestPlaneMode,
    response: ResponsePlaneMode,
) -> Result<DistributedRuntime> {
    DistributedRuntime::new(
        owner,
        DistributedConfig {
            discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(root.into())),
            request_plane: request,
            response_plane: Some(response),
            ..DistributedConfig::process_local()
        },
    )
    .await
}

fn endpoint(drt: &DistributedRuntime, name: &str) -> Endpoint {
    drt.namespace("velo_requests")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint(name)
}

async fn serve(endpoint: &Endpoint, worker: Arc<Worker>) -> Result<StartedEndpoint> {
    endpoint
        .clone()
        .endpoint_builder()
        .handler(Ingress::for_engine(worker)?)
        .graceful_shutdown(false)
        .start_with_registration()
        .await
}

async fn router(endpoint: &Endpoint) -> Result<(Client, Arc<PushRouter<String, Item>>)> {
    let client = endpoint
        .client_with_inhibited_duration(Duration::from_secs(2))
        .await?;
    let router = PushRouter::from_client_with_options(
        client.clone(),
        RouterMode::RoundRobin,
        selector::options(),
    )
    .await?;
    Ok((client, Arc::new(router)))
}

async fn wait_for_instance(client: &Client, id: u64, present: bool) {
    timeout(Duration::from_secs(5), async {
        let mut updates = client.instance_source.as_ref().clone();
        while updates.borrow_and_update().iter().any(|i| i.id() == id) != present {
            updates.changed().await.unwrap();
        }
    })
    .await
    .expect("discovery did not publish the expected endpoint set");
}

async fn assert_items(response: ManyOut<Item>, len: usize) {
    let items = timeout(Duration::from_secs(4), response.collect::<Vec<_>>())
        .await
        .expect("response stream did not end");
    assert!(items.iter().all(|item| item.err().is_none()), "{items:?}");
    assert_eq!(
        items.iter().map(|item| item.value).collect::<Vec<_>>(),
        [len, len + 1, len + 2]
    );
}

fn assert_error(result: Result<ManyOut<Item>>, kinds: &[ErrorType]) {
    let error = result.expect_err("request unexpectedly succeeded");
    assert!(match_error_chain(error.as_ref(), kinds, &[]), "{error:#}");
}

fn velo_request_handler_calls(drt: &DistributedRuntime) -> Result<u64> {
    let metrics = drt.get_metrics_registry().prometheus_expfmt_combined()?;
    metrics
        .lines()
        .filter(|line| {
            line.starts_with("velo_messenger_handler_requests_total{")
                && line.contains("handler=\"dynamo_request\"")
        })
        .map(|line| {
            Ok(line
                .rsplit_once(' ')
                .expect("counter value")
                .1
                .parse::<u64>()?)
        })
        .sum()
}

fn assert_local_routing_error(result: Result<ManyOut<Item>>) {
    let err = result.expect_err("mixed transport unexpectedly succeeded");
    let typed = err
        .downcast_ref::<DynamoError>()
        .expect("typed router error");
    assert_eq!(typed.error_type(), ErrorType::Internal, "{err:#}");
    assert_eq!(typed.reason().as_str(), "runtime.internal");
    assert_eq!(err.chain().count(), 1, "error must have no retryable cause");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn velo_request_plane_serves_and_refuses_requests() -> Result<()> {
    temp_env::async_with_vars(
        [
            ("DYN_TCP_REQUEST_TIMEOUT", Some("2")),
            ("DYN_TCP_WORKER_POOL_SIZE", Some("1")),
            ("DYN_TCP_WORK_QUEUE_SIZE", Some("1")),
            ("DYN_RESPONSE_STREAM_ESTABLISH_TIMEOUT_SECS", Some("1")),
            ("DYN_VELO_RESPONSE_TRANSPORT", Some("tcp")),
            ("DYN_VELO_RESPONSE_PORT", Some("0")),
            ("DYN_TCP_RPC_HOST", Some("127.0.0.1")),
        ],
        async {
            let plane: RequestPlaneMode = "velo".parse()?;
            let second_runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(2)
                .build()?;
            let result = exercise_request_plane(plane, second_runtime.handle().clone()).await;
            second_runtime.shutdown_background();
            result
        },
    )
    .await
}

async fn exercise_request_plane(
    plane: RequestPlaneMode,
    worker_handle: tokio::runtime::Handle,
) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let frontend_owner = Runtime::from_current()?;
    let worker_owner = Runtime::from_handle(worker_handle.clone())?;
    let worker_second_owner = Runtime::from_handle(worker_handle)?;
    let frontend = runtime(
        frontend_owner.clone(),
        dir.path(),
        plane,
        ResponsePlaneMode::Velo,
    )
    .await?;
    let worker_runtime = runtime(
        worker_owner.clone(),
        dir.path(),
        plane,
        ResponsePlaneMode::Velo,
    )
    .await?;
    let backup = runtime(
        worker_second_owner.clone(),
        dir.path(),
        plane,
        ResponsePlaneMode::Velo,
    )
    .await?;
    let tcp = runtime(
        Runtime::from_current()?,
        dir.path(),
        RequestPlaneMode::Tcp,
        ResponsePlaneMode::Tcp,
    )
    .await?;

    let worker = Arc::new(Worker::default());
    let serve_on = endpoint(&worker_runtime, "generate");
    let started = serve(&serve_on, worker.clone()).await?;
    let instance = started.instance().clone();
    let id = instance.id();
    let TransportType::Velo(address) = &instance.transport else {
        panic!("worker published {:?}", instance.transport);
    };
    let worker_server = worker_runtime.request_plane_server().await?;
    assert!(address.starts_with(&format!("{}/", worker_server.address())));
    let frontend_server = frontend.request_plane_server().await?;
    assert_ne!(
        worker_server.address().split('/').nth(2),
        frontend_server.address().split('/').nth(2)
    );

    let front_endpoint = endpoint(&frontend, "generate");
    let (client, frontend_router) = router(&front_endpoint).await?;
    wait_for_instance(&client, id, true).await;
    assert_items(
        frontend_router
            .direct(SingleIn::new("abc".into()), id)
            .await?,
        3,
    )
    .await;

    let local = serve(&endpoint(&frontend, "local"), Arc::new(Worker::default())).await?;
    let (local_client, local_router) = router(&endpoint(&frontend, "local")).await?;
    let local_id = local.instance().id();
    let before_local = velo_request_handler_calls(&frontend)?;
    assert!(
        before_local > 0,
        "remote round trip must count a Velo request"
    );
    wait_for_instance(&local_client, local_id, true).await;
    assert_items(
        local_router
            .direct(SingleIn::new("local".into()), local_id)
            .await?,
        5,
    )
    .await;
    assert_eq!(
        velo_request_handler_calls(&frontend)?,
        before_local,
        "self-target dispatch must bypass the Velo messenger and its ACK slot"
    );

    let tcp_worker = Arc::new(Worker::default());
    let tcp_instance = serve(&endpoint(&tcp, "generate"), tcp_worker.clone()).await?;
    let tcp_id = tcp_instance.instance().id();
    wait_for_instance(&client, tcp_id, true).await;
    let before = worker.calls.load(Ordering::Relaxed);
    assert_local_routing_error(
        timeout(
            Duration::from_secs(1),
            frontend_router.direct(SingleIn::new("".into()), tcp_id),
        )
        .await?,
    );
    assert!(client.instance_ids_avail().contains(&tcp_id));
    assert_eq!(tcp_worker.calls.load(Ordering::Relaxed), 0);
    assert_eq!(worker.calls.load(Ordering::Relaxed), before);
    let (tcp_client, tcp_router) = router(&endpoint(&tcp, "generate")).await?;
    wait_for_instance(&tcp_client, id, true).await;
    assert_local_routing_error(
        timeout(
            Duration::from_secs(1),
            tcp_router.direct(SingleIn::new("".into()), id),
        )
        .await?,
    );
    assert!(tcp_client.instance_ids_avail().contains(&id));
    assert_eq!(worker.calls.load(Ordering::Relaxed), before);
    assert_eq!(tcp_worker.calls.load(Ordering::Relaxed), 0);

    let active = tokio::spawn({
        let router = frontend_router.clone();
        async move { router.direct(SingleIn::new("block".into()), id).await }
    });
    timeout(Duration::from_secs(1), worker.entered.notified()).await?;
    // The second ACK proves the worker reserved the queue slot before the third send.
    let request_client = frontend.network_manager().create_client()?;
    assert!(
        request_client
            .send_request(
                address.clone(),
                Bytes::from_static(b"unused envelope"),
                Default::default()
            )
            .await?
            .is_empty()
    );
    assert_error(
        frontend_router.direct(SingleIn::new("".into()), id).await,
        &[ErrorType::WorkerOverloaded],
    );
    assert!(client.overloaded_instance_ids().unwrap().contains(&id));
    assert!(client.instance_ids_avail().contains(&id));
    worker.release.add_permits(1);
    assert_items(active.await??, 5).await;
    timeout(Duration::from_secs(4), async {
        while client
            .overloaded_instance_ids()
            .is_some_and(|ids| ids.contains(&id))
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;

    let invalid = SingleIn::with_id_and_metadata("".into(), "x".repeat(1025), Default::default());
    assert_error(
        frontend_router.direct(invalid, id).await,
        &[ErrorType::InvalidArgument],
    );
    assert!(client.instance_ids_avail().contains(&id));
    assert!(
        !client
            .overloaded_instance_ids()
            .is_some_and(|ids| ids.contains(&id))
    );

    let len = 300 * 1024;
    assert_items(
        frontend_router
            .direct(SingleIn::new("x".repeat(len)), id)
            .await?,
        len,
    )
    .await;

    assert_error(
        timeout(
            Duration::from_secs(3),
            frontend_router.direct(SingleIn::new("block".into()), id),
        )
        .await?,
        &[ErrorType::ResponseTimeout],
    );
    worker.release.add_permits(1);

    let unknown_id = frontend.discovery().instance_id();
    let advertised = frontend
        .discovery()
        .register(DiscoverySpec::Endpoint {
            namespace: instance.namespace.clone(),
            component: instance.component.clone(),
            endpoint: instance.endpoint.clone(),
            transport: TransportType::Velo(format!("{address}-unknown")),
            device_type: instance.device_type.clone(),
            request_plane_codec: instance.request_plane_codec,
        })
        .await?;
    wait_for_instance(&client, unknown_id, true).await;
    assert_error(
        frontend_router
            .direct(SingleIn::new("".into()), unknown_id)
            .await,
        &[ErrorType::WorkerUnavailable],
    );
    assert!(!client.instance_ids_avail().contains(&unknown_id));
    frontend.discovery().unregister(advertised).await?;

    let other = serve(&endpoint(&backup, "backup"), Arc::new(Worker::default())).await?;
    let (backup_client, backup_router) = router(&endpoint(&frontend, "backup")).await?;
    let backup_id = other.instance().id();
    wait_for_instance(&backup_client, backup_id, true).await;
    started.shutdown().await?;
    worker_owner.shutdown();
    wait_for_instance(&client, id, false).await;
    assert_items(
        backup_router
            .direct(SingleIn::new("backup".into()), backup_id)
            .await?,
        6,
    )
    .await;

    let backup_address = other.instance().transport.address().to_owned();
    backup
        .discovery()
        .unregister(DiscoveryInstance::Endpoint(other.instance().clone()))
        .await?;
    wait_for_instance(&backup_client, backup_id, false).await;
    // Leave the handler open so a generic worker refusal cannot mask missing retirement.
    let retired = timeout(Duration::from_secs(3), async {
        loop {
            let result = request_client
                .send_request(
                    backup_address.clone(),
                    Bytes::from_static(b"unused envelope"),
                    Default::default(),
                )
                .await;
            match result {
                Ok(ack) => assert!(ack.is_empty(), "unexpected handler refusal: {ack:?}"),
                // A send that races route removal can lose the Velo connection.
                Err(error)
                    if match_error_chain(
                        error.as_ref(),
                        &[ErrorType::CannotConnect, ErrorType::ConnectionTimeout],
                        &[],
                    ) => {}
                Err(error) => break error,
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    let typed = retired
        .downcast_ref::<DynamoError>()
        .expect("typed retirement error");
    assert_eq!(
        typed.error_type(),
        ErrorType::WorkerUnavailable,
        "{retired:#}"
    );
    assert_eq!(typed.message(), "Velo request peer has departed");
    other.shutdown().await?;
    worker_second_owner.shutdown();

    local.shutdown().await?;
    tcp_instance.shutdown().await?;
    frontend_owner.shutdown();
    Ok(())
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Result, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::{DashMap, Entry};
use parking_lot::{Mutex, RwLock};
use tokio::sync::Notify;
use velo::Handler;

use crate::{
    SystemHealth,
    pipeline::network::{
        ACK_UNAVAILABLE_PREFIX, PushWorkHandler, get_tcp_max_message_size,
        ingress::unified_server::RequestPlaneServer,
        velo_response::{RuntimeService, VeloResponseService},
    },
    protocols::EndpointId,
    transports::tcp::instance_path,
};

use super::{
    REQUEST_HANDLER,
    admission::{Admission, RequestWork, WorkReservation},
    encode_worker_address,
};

pub(super) const ENDPOINT_PATH_HEADER: &str = "x-endpoint-path";

#[derive(Default)]
pub(in crate::pipeline::network) struct RequestSlot {
    state: RwLock<SlotState>,
}

#[derive(Default)]
struct SlotState {
    server: Weak<SharedServer>,
    registered: bool,
    closed: bool,
}

impl RequestSlot {
    fn attach(
        self: &Arc<Self>,
        service: Arc<VeloResponseService>,
        pool: usize,
        queue: usize,
    ) -> Result<Arc<SharedServer>> {
        let mut state = self.state.write();
        ensure!(!state.closed, "Velo request server is shut down");
        if let Some(server) = state.server.upgrade() {
            return Ok(server);
        }
        let info = service.velo.peer_info();
        let address = format!(
            "velo://{}/{}",
            info.instance_id(),
            encode_worker_address(info.worker_address())?,
        );
        let admission = Admission::new(pool, queue, service.velo.runtime().clone())?;
        if !state.registered {
            let slot = self.clone();
            service.velo.register_handler(
                Handler::unary_handler(REQUEST_HANDLER, move |ctx| {
                    let server = slot.state.read().server.upgrade();
                    match server {
                        Some(server) => server
                            .handle_request(ctx.payload, ctx.headers.unwrap_or_default())
                            .map(Some),
                        None => Ok(Some(Bytes::from_static(
                            b"Server unavailable: worker pool closed",
                        ))),
                    }
                })
                .build(),
            )?;
            state.registered = true;
        }
        let server = Arc::new(SharedServer {
            address,
            service,
            admission,
            endpoints: EndpointTable::default(),
        });
        state.server = Arc::downgrade(&server);
        Ok(server)
    }

    pub(in crate::pipeline::network) async fn shutdown(&self, headroom: Duration) {
        let server = {
            let mut state = self.state.write();
            state.closed = true;
            state.server.upgrade()
        };
        if let Some(server) = server {
            server.admission.close();
            if tokio::time::timeout(headroom, server.admission.wait_idle())
                .await
                .is_err()
            {
                tracing::warn!("Velo request pool did not drain before shutdown");
            }
        }
    }
}

#[derive(Clone)]
pub(in crate::pipeline::network) struct VeloRequestServer {
    shared: Arc<SharedServer>,
    work: Arc<RequestWork>,
}

struct SharedServer {
    address: String,
    service: Arc<VeloResponseService>,
    admission: Admission,
    endpoints: EndpointTable,
}

impl SharedServer {
    fn handle_request(&self, payload: Bytes, headers: HashMap<String, String>) -> Result<Bytes> {
        let Some(path) = headers.get(ENDPOINT_PATH_HEADER) else {
            return Ok(Bytes::from_static(
                b"Server unavailable: missing endpoint path",
            ));
        };
        let guard = match self.endpoints.acquire(path) {
            Ok(guard) => guard,
            Err(ack) => return Ok(ack),
        };
        let limit = get_tcp_max_message_size();
        ensure!(
            payload.len() <= limit,
            "request payload {} bytes exceeds DYN_TCP_MAX_MESSAGE_SIZE {limit}",
            payload.len(),
        );
        Ok(self.admission.try_admit(guard, payload, headers))
    }
}

impl VeloRequestServer {
    pub(in crate::pipeline::network) fn new(
        service: Arc<VeloResponseService>,
        owner: &RuntimeService,
    ) -> Result<Self> {
        let size = |name, default| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(default)
        };
        Self::with_limits(
            service,
            owner.request_work().clone(),
            size("DYN_TCP_WORKER_POOL_SIZE", 10_000),
            size("DYN_TCP_WORK_QUEUE_SIZE", 40_000),
        )
    }

    fn with_limits(
        service: Arc<VeloResponseService>,
        work: Arc<RequestWork>,
        pool: usize,
        queue: usize,
    ) -> Result<Self> {
        ensure!(
            service.request_identity.is_some(),
            "Velo request plane is disabled"
        );
        ensure!(!work.is_closed(), "Velo request owner is shut down");
        let shared = service.request_slot.clone().attach(service, pool, queue)?;
        Ok(Self { shared, work })
    }

    pub(in crate::pipeline::network) fn handle_local(
        &self,
        payload: Bytes,
        headers: HashMap<String, String>,
    ) -> Result<Bytes> {
        self.shared.handle_request(payload, headers)
    }
}

#[derive(Default)]
struct EndpointTable {
    entries: DashMap<String, Arc<EndpointHandler>>,
}

pub(super) struct EndpointHandler {
    pub(super) id: EndpointId,
    pub(super) instance_id: u64,
    pub(super) handler: Arc<dyn PushWorkHandler>,
    pub(super) work: Arc<RequestWork>,
    health: Arc<Mutex<SystemHealth>>,
    inflight: AtomicUsize,
    idle: Notify,
    closing: AtomicBool,
}

pub(super) struct InflightGuard {
    pub(super) endpoint: Arc<EndpointHandler>,
    _work: WorkReservation,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if self.endpoint.inflight.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.endpoint.idle.notify_waiters();
        }
    }
}

impl EndpointTable {
    fn acquire(&self, path: &str) -> Result<InflightGuard, Bytes> {
        let endpoint = self.entries.get(path).ok_or_else(|| {
            Bytes::from(format!("{ACK_UNAVAILABLE_PREFIX} unknown endpoint {path}"))
        })?;
        if endpoint.closing.load(Ordering::Acquire) {
            return Err(Bytes::from_static(
                b"Server unavailable: endpoint shutting down",
            ));
        }
        // Keep the table guard until both the endpoint and its owner count this request.
        let work = endpoint.work.reserve().ok_or_else(|| {
            Bytes::from_static(b"Server unavailable: worker runtime shutting down")
        })?;
        endpoint.inflight.fetch_add(1, Ordering::AcqRel);
        Ok(InflightGuard {
            endpoint: Arc::clone(&endpoint),
            _work: work,
        })
    }

    async fn remove(&self, path: &str, work: &Arc<RequestWork>) {
        let Some((_, endpoint)) = self
            .entries
            .remove_if(path, |_, endpoint| Arc::ptr_eq(work, &endpoint.work))
        else {
            return;
        };
        endpoint
            .health
            .lock()
            .set_endpoint_health_status(&endpoint.id.name, crate::HealthStatus::NotReady);
        let idle = async {
            loop {
                let changed = endpoint.idle.notified();
                if endpoint.inflight.load(Ordering::Acquire) == 0 {
                    return;
                }
                changed.await;
            }
        };
        let timeout = crate::runtime::graceful_shutdown_timeout();
        if tokio::time::timeout(timeout, idle).await.is_err() {
            tracing::warn!(
                endpoint = %endpoint.id,
                remaining = endpoint.inflight.load(Ordering::Acquire),
                "Velo endpoint did not drain before its timeout",
            );
        }
    }
}

#[async_trait]
impl RequestPlaneServer for VeloRequestServer {
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        ensure!(self.is_healthy(), "Velo request server is shut down");
        let id = EndpointId {
            namespace,
            component: component_name,
            name: endpoint_name,
        };
        let path = instance_path(&id, instance_id);
        let Entry::Vacant(entry) = self.shared.endpoints.entries.entry(path) else {
            anyhow::bail!("Velo endpoint {id}/{instance_id:x} is already registered");
        };
        let name = id.name.clone();
        entry.insert(Arc::new(EndpointHandler {
            id,
            instance_id,
            handler: service_handler,
            work: self.work.clone(),
            health: system_health.clone(),
            inflight: AtomicUsize::new(0),
            idle: Notify::new(),
            closing: AtomicBool::new(false),
        }));
        system_health.lock().set_endpoint_registered(&name);
        Ok(())
    }

    async fn close_endpoint_instance(&self, id: &EndpointId, instance_id: u64) -> Result<()> {
        if let Some(endpoint) = self
            .shared
            .endpoints
            .entries
            .get_mut(&instance_path(id, instance_id))
            && Arc::ptr_eq(&self.work, &endpoint.work)
        {
            endpoint.closing.store(true, Ordering::Release);
        }
        Ok(())
    }

    async fn unregister_endpoint(&self, endpoint_name: &str, instance_id: u64) -> Result<()> {
        let path = {
            let mut matches = self.shared.endpoints.entries.iter().filter(|endpoint| {
                Arc::ptr_eq(&self.work, &endpoint.work)
                    && endpoint.id.name == endpoint_name
                    && endpoint.instance_id == instance_id
            });
            let path = matches.next().map(|entry| entry.key().clone());
            ensure!(
                matches.next().is_none(),
                "Ambiguous Velo endpoint {endpoint_name}/{instance_id:x}; use unregister_endpoint_instance"
            );
            path
        };
        if let Some(path) = path {
            self.shared.endpoints.remove(&path, &self.work).await;
        }
        Ok(())
    }

    async fn unregister_endpoint_instance(&self, id: &EndpointId, instance_id: u64) -> Result<()> {
        self.shared
            .endpoints
            .remove(&instance_path(id, instance_id), &self.work)
            .await;
        Ok(())
    }

    fn address(&self) -> String {
        self.shared.address.clone()
    }

    fn transport_name(&self) -> &'static str {
        "velo"
    }

    fn is_healthy(&self) -> bool {
        !self.shared.service.is_closing()
            && !self.shared.admission.is_closed()
            && !self.work.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::AsyncEngineContextProvider;
    use crate::metrics::work_handler_pool::{
        WORK_HANDLER_POOL_ACTIVE_TASKS, WORK_HANDLER_QUEUE_DEPTH,
    };
    use crate::pipeline::network::velo_request::peers::establish_route;
    use crate::pipeline::network::velo_response::{ResponseTransport, VeloServiceOptions};
    use crate::pipeline::{Context, PipelineError};
    use futures::StreamExt;
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::{Notify, Semaphore};
    use tokio::time::Instant;

    struct Handler<F>(F);

    #[async_trait::async_trait]
    impl<F, Fut> PushWorkHandler for Handler<F>
    where
        F: Fn(Bytes) -> Fut + Send + Sync,
        Fut: Future<Output = ()> + Send,
    {
        async fn handle_payload(
            &self,
            payload: Bytes,
            _: Option<String>,
        ) -> Result<(), PipelineError> {
            (self.0)(payload).await;
            Ok(())
        }

        fn add_metrics(
            &self,
            _: &crate::component::Endpoint,
            _: Option<&[(&str, &str)]>,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn endpoint(name: &str) -> EndpointId {
        EndpointId {
            namespace: "test".into(),
            component: "worker".into(),
            name: name.into(),
        }
    }

    fn request_options() -> VeloServiceOptions {
        VeloServiceOptions {
            request_plane_enabled: true,
            ..Default::default()
        }
    }

    fn enable_requests(owner: &RuntimeService) {
        owner.set_options(request_options()).unwrap();
    }

    fn health() -> Arc<Mutex<SystemHealth>> {
        Arc::new(Mutex::new(SystemHealth::new(
            crate::HealthStatus::Ready,
            vec![],
            false,
            "/health".into(),
            "/live".into(),
        )))
    }

    async fn bounded<T>(future: impl Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("operation stalled")
    }

    struct Fixture {
        frontend: Arc<VeloResponseService>,
        worker: Arc<VeloResponseService>,
        owner: Arc<RuntimeService>,
        server: VeloRequestServer,
    }

    impl Fixture {
        async fn new(pool: usize, queue: usize) -> Self {
            let owner = crate::Runtime::from_current()
                .unwrap()
                .velo_response_service();
            enable_requests(&owner);
            let worker = owner.service().await.unwrap();
            let server = VeloRequestServer::with_limits(
                worker.clone(),
                owner.request_work().clone(),
                pool,
                queue,
            )
            .unwrap();
            let frontend = VeloResponseService::new_with_options(
                ResponseTransport::Tcp,
                "127.0.0.1:0".parse().unwrap(),
                request_options(),
            )
            .await
            .unwrap();
            Self {
                frontend,
                worker,
                owner,
                server,
            }
        }

        async fn register(&self, name: &str, handler: Arc<dyn PushWorkHandler>) -> EndpointId {
            register(&self.server, name, handler).await
        }

        async fn send(&self, endpoint: &EndpointId, payload: &'static [u8]) -> Result<Bytes> {
            let target = self.worker.velo.peer_info();
            let address = encode_worker_address(target.worker_address())?;
            establish_route(
                &self.frontend,
                target.instance_id(),
                &address,
                Instant::now() + Duration::from_secs(2),
            )
            .await?;
            self.frontend
                .velo
                .unary(REQUEST_HANDLER)?
                .headers(headers(endpoint))
                .raw_payload(Bytes::from_static(payload))
                .timeout(Duration::from_secs(2))
                .send_to(target.instance_id())
                .await
        }
    }

    async fn register(
        server: &VeloRequestServer,
        name: &str,
        handler: Arc<dyn PushWorkHandler>,
    ) -> EndpointId {
        let endpoint = endpoint(name);
        server
            .register_endpoint(
                endpoint.name.clone(),
                handler,
                1,
                endpoint.namespace.clone(),
                endpoint.component.clone(),
                health(),
            )
            .await
            .unwrap();
        endpoint
    }

    fn headers(endpoint: &EndpointId) -> HashMap<String, String> {
        HashMap::from([(ENDPOINT_PATH_HEADER.into(), instance_path(endpoint, 1))])
    }

    fn inflight(server: &VeloRequestServer, endpoint: &EndpointId) -> Arc<EndpointHandler> {
        Arc::clone(
            &server
                .shared
                .endpoints
                .entries
                .get(&instance_path(endpoint, 1))
                .unwrap(),
        )
    }

    struct Blocked {
        entered: Semaphore,
        released: Semaphore,
        completed: AtomicUsize,
    }

    impl Blocked {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                entered: Semaphore::new(0),
                released: Semaphore::new(0),
                completed: AtomicUsize::new(0),
            })
        }

        fn handler(self: &Arc<Self>) -> Arc<dyn PushWorkHandler> {
            let blocked = self.clone();
            Arc::new(Handler(move |_| {
                let blocked = blocked.clone();
                async move {
                    blocked.entered.add_permits(1);
                    blocked.released.acquire().await.unwrap().forget();
                    blocked.completed.fetch_add(1, Ordering::Relaxed);
                }
            }))
        }

        async fn wait_for(&self, count: u32) {
            bounded(self.entered.acquire_many(count))
                .await
                .unwrap()
                .forget();
        }

        fn release(&self, count: usize) {
            self.released.add_permits(count);
        }
    }

    #[tokio::test]
    async fn shutdown_tracks_work_between_acquire_and_spawn() {
        let fixture = Fixture::new(1, 0).await;
        let endpoint = fixture
            .register("pre-spawn", Blocked::new().handler())
            .await;
        let path = instance_path(&endpoint, 1);
        let guard = fixture.server.shared.endpoints.acquire(&path).unwrap();
        let mut shutdown = std::pin::pin!(fixture.owner.shutdown());
        assert!(futures::poll!(&mut shutdown).is_pending());
        assert_eq!(fixture.owner.request_work().tasks.len(), 1);
        assert_eq!(
            fixture
                .server
                .shared
                .endpoints
                .acquire(&path)
                .err()
                .unwrap(),
            "Server unavailable: worker runtime shutting down"
        );
        drop(guard);
        bounded(shutdown).await;
        assert_eq!(fixture.owner.request_work().outstanding(), 0);
        assert!(fixture.owner.request_work().tasks.is_empty());
    }

    #[tokio::test]
    async fn ack_returns_before_handler_finishes() {
        let fixture = Fixture::new(1, 0).await;
        let blocked = Blocked::new();
        let endpoint = fixture.register("block", blocked.handler()).await;
        assert!(
            fixture
                .send(&endpoint, b"request")
                .await
                .unwrap()
                .is_empty()
        );
        blocked.wait_for(1).await;
        assert_eq!(blocked.completed.load(Ordering::Relaxed), 0);
        blocked.release(1);
        bounded(fixture.server.unregister_endpoint_instance(&endpoint, 1))
            .await
            .unwrap();
        assert_eq!(blocked.completed.load(Ordering::Relaxed), 1);
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn bound_is_exactly_pool_plus_queue() {
        let fixture = Fixture::new(2, 1).await;
        let blocked = Blocked::new();
        let endpoint = fixture.register("capacity", blocked.handler()).await;
        for _ in 0..3 {
            assert!(
                fixture
                    .send(&endpoint, b"request")
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
        blocked.wait_for(2).await;
        assert_eq!(
            fixture.send(&endpoint, b"request").await.unwrap(),
            "Server overloaded: worker at capacity"
        );
        assert_eq!(fixture.owner.request_work().outstanding(), 3);
        blocked.release(3);
        bounded(fixture.server.unregister_endpoint_instance(&endpoint, 1))
            .await
            .unwrap();
        assert_eq!(blocked.completed.load(Ordering::Relaxed), 3);
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn closed_admission_refuses() {
        let fixture = Fixture::new(1, 1).await;
        let endpoint = fixture
            .register(
                "closed",
                Arc::new(Handler(|_| async { panic!("closed pool ran work") })),
            )
            .await;
        let state = inflight(&fixture.server, &endpoint);
        fixture.server.shared.admission.close();
        assert_eq!(
            fixture.send(&endpoint, b"request").await.unwrap(),
            "Server unavailable: worker pool closed"
        );
        assert_eq!(state.inflight.load(Ordering::Acquire), 0);
        assert_eq!(fixture.owner.request_work().outstanding(), 0);
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn gauges_follow_admission() {
        if crate::test_utils::run_isolated(
            concat!(module_path!(), "::gauges_follow_admission"),
            &[],
        ) {
            return;
        }
        let fixture = Fixture::new(1, 1).await;
        let blocked = Blocked::new();
        let endpoint = fixture.register("gauges", blocked.handler()).await;
        let queue = WORK_HANDLER_QUEUE_DEPTH.get();
        let active = WORK_HANDLER_POOL_ACTIVE_TASKS.get();
        assert!(
            fixture
                .send(&endpoint, b"running")
                .await
                .unwrap()
                .is_empty()
        );
        blocked.wait_for(1).await;
        assert_eq!(WORK_HANDLER_QUEUE_DEPTH.get(), queue);
        assert_eq!(WORK_HANDLER_POOL_ACTIVE_TASKS.get(), active + 1);
        assert!(fixture.send(&endpoint, b"queued").await.unwrap().is_empty());
        assert_eq!(WORK_HANDLER_QUEUE_DEPTH.get(), queue + 1);
        assert_eq!(WORK_HANDLER_POOL_ACTIVE_TASKS.get(), active + 1);
        blocked.release(1);
        blocked.wait_for(1).await;
        assert_eq!(WORK_HANDLER_QUEUE_DEPTH.get(), queue);
        assert_eq!(WORK_HANDLER_POOL_ACTIVE_TASKS.get(), active + 1);
        fixture.owner.shutdown().await;
        assert_eq!(WORK_HANDLER_QUEUE_DEPTH.get(), queue);
        assert_eq!(WORK_HANDLER_POOL_ACTIVE_TASKS.get(), active);
    }

    #[tokio::test]
    async fn refused_request_then_unregister_returns_promptly() {
        let fixture = Fixture::new(1, 0).await;
        let blocked = Blocked::new();
        let endpoint = fixture.register("busy", blocked.handler()).await;
        assert!(
            fixture
                .send(&endpoint, b"request")
                .await
                .unwrap()
                .is_empty()
        );
        blocked.wait_for(1).await;
        let other = crate::Runtime::from_current()
            .unwrap()
            .velo_response_service();
        enable_requests(&other);
        let service = other.service().await.unwrap();
        let server =
            VeloRequestServer::with_limits(service, other.request_work().clone(), 1, 0).unwrap();
        let victim = register(
            &server,
            "rejected",
            Arc::new(Handler(|_| async { panic!("rejected work ran") })),
        )
        .await;
        let state = inflight(&server, &victim);
        assert_eq!(
            fixture.send(&victim, b"request").await.unwrap(),
            "Server overloaded: worker at capacity"
        );
        tokio::time::timeout(
            Duration::from_millis(100),
            server.unregister_endpoint_instance(&victim, 1),
        )
        .await
        .expect("refusal leaked an inflight entry")
        .unwrap();
        assert_eq!(state.inflight.load(Ordering::Acquire), 0);
        assert_eq!(other.request_work().outstanding(), 0);
        other.shutdown().await;
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn panicking_handler_releases_inflight() {
        let fixture = Fixture::new(1, 0).await;
        let endpoint = fixture
            .register(
                "panic",
                Arc::new(Handler(|_| async { panic!("test handler panic") })),
            )
            .await;
        let state = inflight(&fixture.server, &endpoint);
        assert!(
            fixture
                .send(&endpoint, b"request")
                .await
                .unwrap()
                .is_empty()
        );
        bounded(fixture.server.unregister_endpoint_instance(&endpoint, 1))
            .await
            .unwrap();
        assert_eq!(state.inflight.load(Ordering::Acquire), 0);
        assert_eq!(fixture.owner.request_work().outstanding(), 0);
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn cancelled_task_releases_inflight() {
        let fixture = Fixture::new(1, 1).await;
        let other = crate::Runtime::from_current()
            .unwrap()
            .velo_response_service();
        enable_requests(&other);
        let second = VeloRequestServer::with_limits(
            other.service().await.unwrap(),
            other.request_work().clone(),
            1,
            1,
        )
        .unwrap();
        let served = Arc::new(Notify::new());
        let witness = served.clone();
        let surviving = register(
            &second,
            "survives-cancel",
            Arc::new(Handler(move |_| {
                let witness = witness.clone();
                async move { witness.notify_one() }
            })),
        )
        .await;
        let blocked = Blocked::new();
        let endpoint = fixture.register("cancel", blocked.handler()).await;
        let state = inflight(&fixture.server, &endpoint);
        for _ in 0..2 {
            assert!(
                fixture
                    .send(&endpoint, b"request")
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
        blocked.wait_for(1).await;
        fixture
            .owner
            .drain(Duration::ZERO, Duration::from_millis(50))
            .await;
        assert_eq!(state.inflight.load(Ordering::Acquire), 2);
        bounded(fixture.owner.shutdown()).await;
        assert_eq!(state.inflight.load(Ordering::Acquire), 0);
        assert_eq!(fixture.owner.request_work().outstanding(), 0);
        assert_eq!(blocked.completed.load(Ordering::Relaxed), 0);
        assert!(fixture.owner.request_work().tasks.is_empty());
        assert!(
            fixture
                .send(&surviving, b"another owner")
                .await
                .unwrap()
                .is_empty()
        );
        bounded(served.notified()).await;
        other.shutdown().await;
    }

    #[tokio::test]
    async fn oversize_eager_payload_releases_inflight() {
        if crate::test_utils::run_isolated(
            concat!(module_path!(), "::oversize_eager_payload_releases_inflight"),
            &[("DYN_TCP_MAX_MESSAGE_SIZE", "4")],
        ) {
            return;
        }
        let fixture = Fixture::new(1, 0).await;
        let endpoint = fixture
            .register(
                "size",
                Arc::new(Handler(|_| async { panic!("oversize work ran") })),
            )
            .await;
        let state = inflight(&fixture.server, &endpoint);
        let error = fixture.send(&endpoint, b"large").await.unwrap_err();
        assert!(
            error.to_string().contains("DYN_TCP_MAX_MESSAGE_SIZE"),
            "{error:#}"
        );
        assert_eq!(state.inflight.load(Ordering::Acquire), 0);
        assert_eq!(fixture.owner.request_work().outstanding(), 0);
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn admitted_work_runs_on_the_velo_runtime() {
        let fixture = Fixture::new(1, 0).await;
        let local = tokio::runtime::Handle::current().id();
        let (ran, did_run) = tokio::sync::oneshot::channel();
        let ran = Arc::new(Mutex::new(Some(ran)));
        let release = Arc::new(Notify::new());
        let wait = release.clone();
        let endpoint = fixture
            .register(
                "runtime",
                Arc::new(Handler(move |_| {
                    let ran = ran.lock().take().unwrap();
                    let wait = wait.clone();
                    async move {
                        wait.notified().await;
                        ran.send(tokio::runtime::Handle::current().id()).unwrap();
                    }
                })),
            )
            .await;
        let server = fixture.server.clone();
        let ack = bounded(tokio::task::spawn_blocking(move || {
            let caller = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            caller.block_on(async { server.handle_local(Bytes::new(), headers(&endpoint)) })
        }))
        .await
        .unwrap()
        .unwrap();
        assert!(ack.is_empty());
        release.notify_one();
        assert_eq!(bounded(did_run).await.unwrap(), local);
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn dropping_the_last_server_breaks_the_cycle() {
        let fixture = Fixture::new(1, 0).await;
        let service = Arc::downgrade(&fixture.worker);
        let server = Arc::downgrade(&fixture.server.shared);
        let keep_alive = fixture.worker.clone();
        fixture
            .register(
                "cycle",
                Arc::new(Handler(move |_| {
                    let keep_alive = keep_alive.clone();
                    async move { drop(keep_alive) }
                })),
            )
            .await;
        drop(fixture.server);
        assert!(
            server.upgrade().is_none(),
            "the Velo handler retained the server"
        );
        fixture.owner.shutdown().await;
        drop(fixture.worker);
        assert!(
            service.upgrade().is_none(),
            "the endpoint table retained the service"
        );
    }

    #[tokio::test]
    async fn close_endpoint_refuses_new_and_drains_accepted() {
        let fixture = Fixture::new(1, 1).await;
        let blocked = Blocked::new();
        let endpoint = fixture.register("drain", blocked.handler()).await;
        for _ in 0..2 {
            assert!(
                fixture
                    .send(&endpoint, b"accepted")
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
        blocked.wait_for(1).await;
        fixture
            .server
            .close_endpoint_instance(&endpoint, 1)
            .await
            .unwrap();
        assert_eq!(
            fixture.send(&endpoint, b"late").await.unwrap(),
            "Server unavailable: endpoint shutting down"
        );
        let mut unregister =
            std::pin::pin!(fixture.server.unregister_endpoint_instance(&endpoint, 1));
        assert!(futures::poll!(&mut unregister).is_pending());
        blocked.release(2);
        bounded(unregister).await.unwrap();
        assert_eq!(blocked.completed.load(Ordering::Relaxed), 2);
        fixture.owner.shutdown().await;
    }

    #[tokio::test]
    async fn queued_request_during_shutdown_with_second_owner() {
        let fixture = Fixture::new(1, 1).await;
        let other = crate::Runtime::from_current()
            .unwrap()
            .velo_response_service();
        enable_requests(&other);
        let service = other.service().await.unwrap();
        assert!(Arc::ptr_eq(&service, &fixture.worker));
        let second =
            VeloRequestServer::with_limits(service, other.request_work().clone(), 1, 1).unwrap();
        let served = Arc::new(Notify::new());
        let did_serve = served.clone();
        let b = register(
            &second,
            "second",
            Arc::new(Handler(move |_| {
                let did_serve = did_serve.clone();
                async move { did_serve.notify_one() }
            })),
        )
        .await;
        let registered = fixture
            .frontend
            .register_response(Context::new(()).context(), false)
            .unwrap();
        let connection = registered.connection_info.clone();
        let a_started = Arc::new(Notify::new());
        let a_release = Arc::new(Notify::new());
        let queued_started = Arc::new(Notify::new());
        let queued_release = Arc::new(Notify::new());
        let owner = Arc::downgrade(&fixture.owner);
        let a = fixture
            .register(
                "first",
                Arc::new(Handler({
                    let a_started = a_started.clone();
                    let a_release = a_release.clone();
                    let queued_started = queued_started.clone();
                    let queued_release = queued_release.clone();
                    move |payload: Bytes| {
                        let (a_started, a_release) = (a_started.clone(), a_release.clone());
                        let (queued_started, queued_release) =
                            (queued_started.clone(), queued_release.clone());
                        let (owner, connection) = (owner.clone(), connection.clone());
                        async move {
                            if payload == b"running"[..] {
                                a_started.notify_one();
                                a_release.notified().await;
                            } else {
                                queued_started.notify_one();
                                queued_release.notified().await;
                                let service =
                                    owner.upgrade().unwrap().existing_service().await.unwrap();
                                let mut sender = service
                                    .sender(Context::new(()).context(), connection, None)
                                    .await
                                    .unwrap();
                                sender.send_prologue(None).await.unwrap();
                                sender.send(Bytes::from_static(b"drained")).await.unwrap();
                                sender.finish().await.unwrap();
                            }
                        }
                    }
                })),
            )
            .await;

        assert!(fixture.send(&a, b"running").await.unwrap().is_empty());
        bounded(a_started.notified()).await;
        assert!(fixture.send(&a, b"queued").await.unwrap().is_empty());
        assert_eq!(
            fixture.send(&b, b"full").await.unwrap(),
            "Server overloaded: worker at capacity"
        );
        fixture.owner.close();
        let a_owner = fixture.owner.clone();
        let drain = tokio::spawn(async move {
            a_owner.drain(Duration::ZERO, Duration::from_secs(2)).await;
            a_owner.shutdown().await;
        });
        a_release.notify_one();
        bounded(queued_started.notified()).await;
        assert!(
            fixture
                .send(&b, b"capacity restored")
                .await
                .unwrap()
                .is_empty()
        );
        queued_release.notify_one();
        let (_, provider) = registered.into_parts();
        let mut response = bounded(provider).await.unwrap().unwrap();
        assert_eq!(
            bounded(response.rx.next()).await.unwrap().unwrap(),
            "drained"
        );
        assert!(bounded(response.rx.next()).await.is_none());
        bounded(served.notified()).await;
        bounded(drain).await.unwrap();
        assert_eq!(fixture.owner.request_work().outstanding(), 0);
        assert!(fixture.owner.request_work().tasks.is_empty());
        assert!(
            fixture
                .send(&b, b"after shutdown")
                .await
                .unwrap()
                .is_empty()
        );
        bounded(served.notified()).await;
        other.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn owner_hard_deadline_retains_lease_until_task_exits() {
        struct SlowExit {
            exiting: Arc<Notify>,
            release: std::sync::mpsc::Receiver<()>,
        }

        impl Drop for SlowExit {
            fn drop(&mut self) {
                self.exiting.notify_one();
                let _ = self.release.recv_timeout(Duration::from_secs(8));
            }
        }

        let fixture = Fixture::new(1, 1).await;
        let (release, wait) = std::sync::mpsc::channel();
        let exiting = Arc::new(Notify::new());
        let started = Arc::new(Notify::new());
        let guard = Arc::new(Mutex::new(Some(SlowExit {
            exiting: exiting.clone(),
            release: wait,
        })));
        let a = fixture
            .register(
                "slow-exit",
                Arc::new(Handler({
                    let started = started.clone();
                    move |_| {
                        let guard = guard.lock().take().unwrap();
                        let started = started.clone();
                        async move {
                            let _guard = guard;
                            started.notify_one();
                            std::future::pending::<()>().await;
                        }
                    }
                })),
            )
            .await;
        let state = inflight(&fixture.server, &a);
        assert!(fixture.send(&a, b"request").await.unwrap().is_empty());
        bounded(started.notified()).await;
        fixture.owner.close();
        let started = Instant::now();
        fixture
            .owner
            .drain(Duration::ZERO, Duration::from_millis(50))
            .await;
        assert!(
            started.elapsed() >= Duration::from_millis(50),
            "drain ignored outstanding admission"
        );
        bounded(fixture.owner.shutdown()).await;
        bounded(exiting.notified()).await;
        assert!(
            !fixture.worker.is_closing(),
            "Velo shut down while an admitted task was exiting"
        );
        assert_eq!(fixture.owner.request_work().finalizers.len(), 1);
        assert_eq!(state.inflight.load(Ordering::Acquire), 1);
        release.send(()).unwrap();
        bounded(async {
            while !fixture.worker.is_closing()
                || !fixture.owner.request_work().finalizers.is_empty()
            {
                tokio::task::yield_now().await;
            }
        })
        .await;
        assert_eq!(state.inflight.load(Ordering::Acquire), 0);
        assert!(fixture.owner.request_work().tasks.is_empty());
    }
}

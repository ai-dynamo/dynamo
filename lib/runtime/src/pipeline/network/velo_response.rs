// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo owns response framing, batching, flow control, and stream lifecycle.

use anyhow::{Result, bail, ensure};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    pin::Pin,
    sync::{
        Arc, LazyLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::{OnceCell, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use velo::{
    PeerInfo, Velo,
    streaming::{
        MuxConfig, StreamAnchor, StreamAnchorHandle, StreamController, StreamFrame, StreamSender,
        control::StreamOpenTicket,
    },
};

use super::{
    ConnectionInfo, RegisteredStream, ResponseStreamPrologue, StreamPrologueError, StreamReceiver,
};
use crate::{
    discovery::EndpointInstanceId,
    engine::AsyncEngineContext,
    error::{DynamoError, ErrorType},
};

pub const TRANSPORT_NAME: &str = "velo_response";
const VERSION: u32 = 2;
const TOMBSTONE_TTL: Duration = Duration::from_secs(5);
static PROCESS_SERVICE: tokio::sync::Mutex<Weak<SharedService>> =
    tokio::sync::Mutex::const_new(Weak::new());
static PROCESS_METRICS: LazyLock<(crate::MetricsRegistry, Arc<velo::VeloMetrics>)> =
    LazyLock::new(|| {
        let registry = crate::MetricsRegistry::new();
        let metrics = velo::VeloMetrics::register(&registry.prometheus_registry.read().unwrap())
            .expect("register Velo response metrics");
        (registry, Arc::new(metrics))
    });

pub(crate) fn register_metrics(registry: &crate::MetricsRegistry) {
    registry.add_child_registry(&PROCESS_METRICS.0);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseTransport {
    Tcp,
    Ucx,
}

/// Keep 32 explicit here, matching Velo's current default. In the earlier
/// mocker campaign, a larger window filled the shared path from a
/// worker to the frontend, so a new stream's first token waited behind it.
/// On the mocker rig with a saturated 24-core frontend, a window of 256
/// put TTFT p50 at 135 to 355 ms and 32 put it at 88 to 102 ms. Throughput was
/// within noise, and ITL p99 was no higher.
const RESPONSE_CREDIT_WINDOW: u32 = 32;

/// TCP lanes to each frontend for response streams.
///
/// 4, not velo's default of 1. Each lane is its own TCP connection, read by
/// its own task on the frontend, and the mux spreads the response streams over
/// the lanes. One connection is limited by the task that reads it. On the
/// mocker rig (8 worker processes of 64, concurrency 8,192, ISL 1,024, OSL
/// 900), 4 lanes against 1 gave 2,406 against 1,359 requests/s, ITL p99 4.9
/// against 15.0 ms, end-to-end p99 4.5 against 13.5 s, and 20.0 against
/// 37.4 ms of frontend CPU per request.
const RESPONSE_TCP_LANES: u16 = 4;

fn response_mux_config() -> MuxConfig {
    MuxConfig {
        enabled: true,
        initial_credit: RESPONSE_CREDIT_WINDOW,
        ..Default::default()
    }
}

fn tcp_transport(address: SocketAddr) -> Result<velo::transports::tcp::TcpTransport> {
    velo::transports::tcp::TcpTransportBuilder::new()
        .bind_addr(address)
        .lanes(RESPONSE_TCP_LANES)
        .build()
}

impl ResponseTransport {
    fn configured() -> Result<Self> {
        match std::env::var(
            crate::config::environment_names::response_plane::DYN_VELO_RESPONSE_TRANSPORT,
        )
        .as_deref()
        {
            Err(std::env::VarError::NotPresent) | Ok("tcp") => Ok(Self::Tcp),
            Ok("ucx") => Ok(Self::Ucx),
            _ => bail!("DYN_VELO_RESPONSE_TRANSPORT must be tcp or ucx"),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ResponseAddress {
    version: u32,
    transport: ResponseTransport,
    peer: PeerInfo,
    anchor: StreamAnchorHandle,
    ticket: StreamOpenTicket,
}

#[derive(Serialize, Deserialize)]
enum ResponseFrame {
    Prologue(ResponseStreamPrologue),
    Data(Bytes),
}

struct Registration {
    controller: StreamController,
    done: CancellationToken,
    instance: Option<EndpointInstanceId>,
}

#[derive(Default)]
struct Registrations {
    requests: HashMap<Uuid, Registration>,
    tombstones: HashMap<EndpointInstanceId, Instant>,
}

/// One owner per Dynamo Runtime, shared by all of that runtime's clones.
pub(crate) struct RuntimeService {
    runtime: crate::runtime::RuntimeType,
    owner: tokio::sync::Mutex<Option<Arc<SharedService>>>,
    closed: AtomicBool,
}

impl std::fmt::Debug for RuntimeService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeService")
            .field("closed", &self.is_closed())
            .finish_non_exhaustive()
    }
}

impl RuntimeService {
    pub(crate) fn new(runtime: crate::runtime::RuntimeType) -> Self {
        Self {
            runtime,
            owner: tokio::sync::Mutex::new(None),
            closed: AtomicBool::new(false),
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub(crate) async fn service(&self) -> Result<Arc<VeloResponseService>> {
        let mut owner = self.owner.lock().await;
        ensure!(!self.is_closed(), "Velo response runtime is shut down");
        if let Some(owner) = owner.as_ref() {
            return Ok(owner.service.clone());
        }
        // Acquisition and final-owner shutdown use the same lock. A new
        // runtime cannot acquire a service while that service is closing.
        let mut shared = PROCESS_SERVICE.lock().await;
        let lease = match shared.upgrade() {
            Some(lease) => lease,
            None => {
                let runtime = self.runtime.clone();
                // The spawned result owns cleanup even if its caller stops
                // waiting before construction finishes.
                let lease = Arc::new(
                    runtime
                        .handle()
                        .spawn(async move {
                            Ok::<_, anyhow::Error>(SharedService {
                                service: VeloResponseService::from_env().await?,
                                runtime,
                            })
                        })
                        .await??,
                );
                *shared = Arc::downgrade(&lease);
                lease
            }
        };
        let service = lease.service.clone();
        *owner = Some(lease);
        Ok(service)
    }

    pub(crate) async fn shutdown(&self) {
        let mut owner = self.owner.lock().await;
        self.closed.store(true, Ordering::Release);
        let Some(lease) = owner.take() else {
            return;
        };
        let mut shared = PROCESS_SERVICE.lock().await;
        if let Some(lease) = Arc::into_inner(lease) {
            *shared = Weak::new();
            lease.service.shutdown().await;
        }
    }
}

/// Stream handles retain the service, but do not count as runtime owners.
struct SharedService {
    service: Arc<VeloResponseService>,
    // Retain an owned Tokio runtime without retaining the Dynamo Runtime and
    // its owner slot. External runtimes remain the caller's responsibility.
    runtime: crate::runtime::RuntimeType,
}

impl Drop for SharedService {
    fn drop(&mut self) {
        if self.service.closed.initialized() {
            return;
        }
        // Explicit Runtime shutdown awaits this work. Drop can only make a
        // best effort, using the runtime that created the service.
        let service = self.service.clone();
        let runtime = self.runtime.clone();
        runtime.handle().spawn(async move {
            service.shutdown().await;
            drop(runtime);
        });
    }
}

pub struct VeloResponseService {
    velo: Arc<Velo>,
    transport: ResponseTransport,
    registrations: Mutex<Registrations>,
    /// Share initial peer registration and the lifecycle handshake across
    /// concurrent requests. A restarted frontend has a new instance ID.
    prepared_peers: dashmap::DashMap<velo::InstanceId, Arc<OnceCell<()>>>,
    closing: AtomicBool,
    closed: OnceCell<()>,
}

impl VeloResponseService {
    async fn from_env() -> Result<Arc<Self>> {
        let transport = ResponseTransport::configured()?;
        let host = crate::utils::ip_resolver::host_override_from_env(
            crate::config::environment_names::tcp_response_stream::DYN_TCP_RESPONSE_STREAM_HOST,
        )?;
        let resolver = crate::utils::ip_resolver::DefaultIpResolver;
        let host = match host {
            Some(host) => crate::utils::ip_resolver::resolve_host_or_interface(&host, &resolver)?,
            None => crate::utils::ip_resolver::resolve_local_host(&resolver)?,
        };
        Self::new(transport, SocketAddr::new(host.advertise_ip(), 0)).await
    }

    async fn new(transport: ResponseTransport, address: SocketAddr) -> Result<Arc<Self>> {
        let mut builder = Velo::builder()
            .metrics(PROCESS_METRICS.1.clone())
            .mux_only()
            .messenger_mux(response_mux_config())?;
        match transport {
            ResponseTransport::Tcp => {
                builder = builder.add_transport(Arc::new(tcp_transport(address)?));
            }
            ResponseTransport::Ucx => {
                #[cfg(all(target_os = "linux", feature = "velo-ucx"))]
                {
                    builder = builder.add_transport(Arc::new(
                        velo::transports::ucx::UcxTransportBuilder::new().build()?,
                    ));
                }
                #[cfg(not(all(target_os = "linux", feature = "velo-ucx")))]
                bail!("Velo UCX responses require a Linux build with the velo-ucx feature");
            }
        }
        Ok(Arc::new(Self {
            velo: builder.build().await?,
            transport,
            registrations: Mutex::new(Registrations::default()),
            prepared_peers: dashmap::DashMap::new(),
            closing: AtomicBool::new(false),
            closed: OnceCell::new(),
        }))
    }

    async fn shutdown(&self) {
        self.closed
            .get_or_init(|| async {
                self.closing.store(true, Ordering::Release);
                let requests = std::mem::take(&mut self.registrations.lock().requests);
                for registration in requests.into_values() {
                    registration.done.cancel();
                    registration.controller.cancel();
                }
                // Bound the drain to five seconds, then await transport close.
                self.velo
                    .shutdown(velo::ShutdownPolicy::Timeout(Duration::from_secs(5)))
                    .await;
            })
            .await;
    }

    pub fn register_response(
        self: &Arc<Self>,
        context: Arc<dyn AsyncEngineContext>,
    ) -> Result<RegisteredStream<StreamReceiver>> {
        ensure!(
            !self.closing.load(Ordering::Acquire),
            "Velo response service is shut down"
        );
        let anchor = self.velo.create_anchor::<ResponseFrame>();
        let ticket = self
            .velo
            .prebind_anchor(anchor.handle())
            .ok_or_else(|| anyhow::anyhow!("Velo mux prebind unavailable"))?;
        let controller = anchor.controller();
        let info = ConnectionInfo {
            transport: TRANSPORT_NAME.into(),
            info: serde_json::to_string(&ResponseAddress {
                version: VERSION,
                transport: self.transport,
                peer: self.velo.peer_info(),
                anchor: anchor.handle(),
                ticket,
            })?,
        };
        let id = Uuid::new_v4();
        let done = CancellationToken::new();
        let mut registrations = self.registrations.lock();
        ensure!(
            !self.closing.load(Ordering::Acquire),
            "Velo response service is shut down"
        );
        registrations.requests.insert(
            id,
            Registration {
                controller: controller.clone(),
                done: done.clone(),
                instance: None,
            },
        );
        drop(registrations);
        tracing::debug!(context_id = %context.id(), registration_id = %id, "Registered Velo response stream");
        let lease = ResponseLease {
            service: self.clone(),
            id,
        };
        let (mut tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let mut anchor = anchor;
            let mut stopped = false;
            // This task handles setup and lifecycle only. The caller polls all
            // response data directly from the anchor after the prologue.
            let first = loop {
                tokio::select! {
                    biased;
                    _ = done.cancelled() => return,
                    _ = context.killed() => {
                        tracing::debug!(context_id = %context.id(), "Cancelling Velo response stream");
                        controller.cancel(); return;
                    }
                    _ = tx.closed() => return,
                    _ = context.stopped(), if !stopped => {
                        tracing::debug!(context_id = %context.id(), "Stopping Velo response generation");
                        controller.request_stop(); stopped = true;
                    }
                    first = anchor.next() => break first,
                }
            };
            match first {
                Some(Ok(StreamFrame::Item(ResponseFrame::Prologue(prologue)))) => {
                    if let Some(error) = prologue.error {
                        tracing::debug!(context_id = %context.id(), %error, "Velo response prologue returned an error");
                        let _ = tx.send(Err(StreamPrologueError {
                            message: error,
                            typed_error: prologue.typed_error,
                        }));
                        return;
                    }
                }
                other => {
                    tracing::warn!(context_id = %context.id(), reason = %first_kind(&other), "Velo response ended before its prologue");
                    let _ = tx.send(Err(StreamPrologueError::from_message(format!(
                        "Velo response ended before a valid prologue: {}",
                        first_kind(&other)
                    ))));
                    return;
                }
            }
            let receiver = VeloResponseReceiver {
                anchor,
                _lease: lease,
                context_id: context.id().to_string(),
                ended: false,
            };
            if tx
                .send(Ok(StreamReceiver {
                    rx: super::ByteReceiver::Velo(Box::new(receiver)),
                }))
                .is_err()
            {
                return;
            }
            loop {
                tokio::select! {
                    biased;
                    _ = done.cancelled() => return,
                    _ = context.killed() => {
                        tracing::debug!(context_id = %context.id(), "Cancelling Velo response stream");
                        controller.cancel(); return;
                    }
                    _ = context.stopped(), if !stopped => {
                        tracing::debug!(context_id = %context.id(), "Stopping Velo response generation");
                        controller.request_stop(); stopped = true;
                    }
                }
            }
        });
        let service = self.clone();
        Ok(RegisteredStream::new(info, rx)
            .with_registration_id(id)
            .with_cleanup(move || service.remove(id)))
    }

    fn remove(&self, id: Uuid) {
        let registration = self.registrations.lock().requests.remove(&id);
        if let Some(registration) = registration {
            registration.done.cancel();
            registration.controller.cancel();
        }
    }

    pub async fn associate_instance(&self, id: Uuid, instance: &EndpointInstanceId) -> bool {
        let mut state = self.registrations.lock();
        state
            .tombstones
            .retain(|_, when| when.elapsed() < TOMBSTONE_TTL);
        if state.tombstones.contains_key(instance) {
            return false;
        }
        if let Some(request) = state.requests.get_mut(&id) {
            request.instance = Some(instance.clone());
            true
        } else {
            false
        }
    }

    pub async fn cancel_response(&self, id: Uuid) {
        self.remove(id);
    }

    pub async fn cancel_instance_streams(&self, instance: &EndpointInstanceId) -> usize {
        let ids = {
            let mut state = self.registrations.lock();
            state
                .tombstones
                .retain(|_, when| when.elapsed() < TOMBSTONE_TTL);
            state.tombstones.insert(instance.clone(), Instant::now());
            state
                .requests
                .iter()
                .filter_map(|(id, r)| (r.instance.as_ref() == Some(instance)).then_some(*id))
                .collect::<Vec<_>>()
        };
        for id in &ids {
            self.remove(*id);
        }
        ids.len()
    }

    pub async fn clear_instance_tombstone(&self, instance: &EndpointInstanceId) {
        self.registrations.lock().tombstones.remove(instance);
    }

    async fn prepare_peer(&self, peer: PeerInfo) -> Result<()> {
        let peer_id = peer.instance_id();
        let ready = self
            .prepared_peers
            .entry(peer_id)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        // Release the map guard before awaiting. A failed or cancelled attempt
        // leaves the cell empty, so the next caller can retry.
        ready
            .get_or_try_init(|| async {
                self.velo.register_peer(peer)?;
                if peer_id != self.velo.instance_id() {
                    // The hello also installs the reverse UCX address.
                    tokio::time::timeout(
                        Duration::from_secs(10),
                        self.velo.wait_for_handler(peer_id, "_stream_stop"),
                    )
                    .await??;
                }
                Ok::<(), anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    pub async fn sender(
        self: &Arc<Self>,
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
        cancellation: Option<prometheus::IntCounter>,
    ) -> Result<VeloResponseSender> {
        ensure!(
            !self.closing.load(Ordering::Acquire),
            "Velo response service is shut down"
        );
        ensure!(
            info.transport == TRANSPORT_NAME,
            "invalid Velo response transport"
        );
        let address: ResponseAddress = serde_json::from_str(&info.info)?;
        ensure!(
            address.version == VERSION
                && address.ticket.streaming_transport_key.as_str()
                    == velo::streaming::MESSENGER_MUX_KEY,
            "unsupported Velo response lifecycle version"
        );
        ensure!(
            address.transport == self.transport,
            "Velo response transport mismatch"
        );
        ensure!(
            address.anchor.unpack().0 == address.peer.worker_id(),
            "Velo anchor and peer identity differ"
        );
        self.prepare_peer(address.peer).await.inspect_err(|error| {
            tracing::warn!(context_id = %context.id(), %error, "Failed to prepare Velo response peer");
        })?;
        let sender = if address.anchor.unpack().0 == self.velo.instance_id().worker_id() {
            self.velo
                .attach_anchor::<ResponseFrame>(address.anchor)
                .await?
        } else {
            self.velo
                .open_anchor_stream::<ResponseFrame>(address.anchor, address.ticket)
                .await?
        };
        tracing::debug!(context_id = %context.id(), "Opened Velo response sender");
        let stop = sender.stop_token();
        let cancel = sender.cancellation_token();
        let finished = CancellationToken::new();
        let monitor_done = finished.clone();
        tokio::spawn(async move {
            let mut stopped = false;
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        if let Some(counter) = &cancellation { counter.inc(); }
                        tracing::debug!(context_id = %context.id(), "Velo response peer cancelled the request");
                        context.kill(); return;
                    }
                    _ = monitor_done.cancelled() => return,
                    _ = stop.cancelled(), if !stopped => {
                        tracing::debug!(context_id = %context.id(), "Velo response peer requested a stop");
                        context.stop_generating(); stopped = true;
                    }
                }
            }
        });
        Ok(VeloResponseSender {
            sender: Some(sender),
            _service: self.clone(),
            finished,
            prologue_sent: false,
        })
    }
}

struct ResponseLease {
    service: Arc<VeloResponseService>,
    id: Uuid,
}
impl Drop for ResponseLease {
    fn drop(&mut self) {
        self.service.remove(self.id);
    }
}

fn first_kind(
    frame: &Option<Result<StreamFrame<ResponseFrame>, velo::streaming::StreamError>>,
) -> String {
    match frame {
        None => "closed".into(),
        Some(Err(error)) => error.to_string(),
        Some(Ok(_)) => "unexpected frame".into(),
    }
}

pub(super) struct VeloResponseReceiver {
    anchor: StreamAnchor<ResponseFrame>,
    _lease: ResponseLease,
    context_id: String,
    ended: bool,
}
impl Stream for VeloResponseReceiver {
    type Item = Result<Bytes, DynamoError>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        match std::task::ready!(Pin::new(&mut this.anchor).poll_next(cx)) {
            Some(Ok(StreamFrame::Item(ResponseFrame::Data(bytes)))) => Poll::Ready(Some(Ok(bytes))),
            Some(Ok(StreamFrame::Finalized)) => {
                this.ended = true;
                Poll::Ready(None)
            }
            terminal => {
                this.ended = true;
                tracing::warn!(context_id = %this.context_id, reason = %first_kind(&terminal), "Velo response ended without finalization");
                Poll::Ready(Some(Err(DynamoError::builder()
                    .error_type(ErrorType::Disconnected)
                    .message(format!(
                        "Velo response ended without finalization: {}",
                        first_kind(&terminal)
                    ))
                    .build())))
            }
        }
    }
}

pub struct VeloResponseSender {
    sender: Option<StreamSender<ResponseFrame>>,
    _service: Arc<VeloResponseService>,
    finished: CancellationToken,
    prologue_sent: bool,
}

impl VeloResponseSender {
    pub async fn send(&self, bytes: Bytes) -> Result<()> {
        ensure!(self.prologue_sent, "Velo response prologue missing");
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Velo response closed"))?
            .send(ResponseFrame::Data(bytes))
            .await?;
        Ok(())
    }
    pub async fn send_prologue(&mut self, error: Option<StreamPrologueError>) -> Result<()> {
        ensure!(!self.prologue_sent, "Velo response prologue already sent");
        let (error, typed_error) = error.map_or((None, None), |e| (Some(e.message), e.typed_error));
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Velo response closed"))?
            .send(ResponseFrame::Prologue(ResponseStreamPrologue {
                error,
                typed_error,
            }))
            .await?;
        self.prologue_sent = true;
        Ok(())
    }
    pub async fn finish(&mut self) -> Result<()> {
        self.finished.cancel();
        if let Some(sender) = self.sender.take() {
            sender.finalize()?;
        }
        Ok(())
    }
    pub async fn abort(&mut self) -> Result<()> {
        self.finished.cancel();
        self.sender.take();
        Ok(())
    }
}

impl Drop for VeloResponseSender {
    fn drop(&mut self) {
        self.finished.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::AsyncEngineContextProvider;
    use crate::pipeline::Context as EngineContext;

    #[tokio::test]
    async fn runtime_shutdown_closes_only_the_last_owner_after_endpoint_drain() {
        temp_env::async_with_vars(
            [
                ("DYN_TCP_RESPONSE_STREAM_HOST", Some("127.0.0.1")),
                ("DYN_VELO_RESPONSE_TRANSPORT", Some("tcp")),
            ],
            async {
                // Separate owned Tokio runtimes: the service must remain alive
                // even after the runtime that created it has been dropped.
                let first = crate::Runtime::single_threaded().unwrap();
                let first_clone = first.clone();
                let second = crate::Runtime::single_threaded().unwrap();
                let service = first.velo_response_service().service().await.unwrap();
                assert!(Arc::ptr_eq(
                    &service,
                    &first_clone.velo_response_service().service().await.unwrap()
                ));
                assert!(Arc::ptr_eq(
                    &service,
                    &second.velo_response_service().service().await.unwrap()
                ));
                let entry = service
                    .velo
                    .peer_info()
                    .worker_address()
                    .get_entry("tcp")
                    .unwrap()
                    .unwrap();
                let address = velo::transports::utils::interfaces::parse_endpoints(&entry).unwrap()
                    [0]
                .socket_addr()
                .unwrap();

                first.shutdown();
                tokio::time::timeout(Duration::from_secs(10), first.primary_token().cancelled())
                    .await
                    .unwrap();
                assert!(first_clone.velo_response_service().service().await.is_err());
                drop(first_clone);
                drop(first);
                assert!(!service.closed.initialized());
                tokio::net::TcpStream::connect(address).await.unwrap();

                // Retain both the service and a registration across shutdown.
                let registration = service
                    .register_response(EngineContext::new(()).context())
                    .unwrap();
                let endpoint = second.graceful_shutdown_tracker().register_task();
                let main = second.primary_token();
                second.shutdown();
                second.child_token().cancelled().await;
                assert!(!main.is_cancelled());
                assert!(!service.closed.initialized());
                tokio::net::TcpStream::connect(address).await.unwrap();
                drop(endpoint);
                tokio::time::timeout(Duration::from_secs(10), main.cancelled())
                    .await
                    .unwrap();
                assert!(service.closed.initialized());
                assert!(tokio::net::TcpStream::connect(address).await.is_err());
                assert!(service.registrations.lock().requests.is_empty());
                assert!(
                    service
                        .register_response(EngineContext::new(()).context())
                        .is_err()
                );
                assert!(second.velo_response_service().service().await.is_err());
                drop(registration);

                // A later runtime receives a new service, even while callers
                // still hold handles to the service that was shut down.
                let replacement = crate::Runtime::from_current().unwrap();
                let fresh = replacement.velo_response_service().service().await.unwrap();
                assert!(!Arc::ptr_eq(&service, &fresh));
                replacement.shutdown();
                tokio::time::timeout(
                    Duration::from_secs(10),
                    replacement.primary_token().cancelled(),
                )
                .await
                .unwrap();
            },
        )
        .await;
    }

    #[tokio::test]
    async fn concurrent_peer_preparation_shares_a_handshake_and_retries_failure() {
        use std::sync::atomic::AtomicUsize;
        use tracing_subscriber::prelude::*;

        // Count actual hello requests at the peer, rather than just checking
        // the cache size after concurrent calls have completed.
        struct HelloCount(Arc<AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for HelloCount {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if event.metadata().target() != "crate::messenger::system" {
                    return;
                }
                struct Message(bool);
                impl tracing::field::Visit for Message {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "message"
                            && format!("{value:?}") == "Received _hello handshake from peer"
                        {
                            self.0 = true;
                        }
                    }
                }
                let mut message = Message(false);
                event.record(&mut message);
                if message.0 {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let hellos = Arc::new(AtomicUsize::new(0));
        let _logging = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(HelloCount(hellos.clone())),
        );
        let (consumer, producer) = pair(ResponseTransport::Tcp).await;
        let peer = consumer.velo.peer_info();
        let invalid = PeerInfo::new(peer.instance_id(), velo::WorkerAddress::empty());
        assert!(producer.prepare_peer(invalid).await.is_err());
        assert!(
            !producer
                .prepared_peers
                .get(&peer.instance_id())
                .unwrap()
                .initialized()
        );
        let results =
            futures::future::join_all((0..8).map(|_| producer.prepare_peer(peer.clone()))).await;
        for result in results {
            result.unwrap();
        }
        assert!(
            producer
                .prepared_peers
                .get(&peer.instance_id())
                .unwrap()
                .initialized()
        );
        assert_eq!(hellos.load(Ordering::Relaxed), 1);
        producer.prepare_peer(peer).await.unwrap();
        assert_eq!(hellos.load(Ordering::Relaxed), 1);
        producer.shutdown().await;
        consumer.shutdown().await;
    }

    #[test]
    fn tcp_responses_use_four_lanes() {
        use velo::Transport;
        let transport = tcp_transport("127.0.0.1:0".parse().unwrap()).unwrap();
        assert_eq!(transport.lanes(velo::InstanceId::new_v4()).get(), 4);
    }

    async fn pair(
        transport: ResponseTransport,
    ) -> (Arc<VeloResponseService>, Arc<VeloResponseService>) {
        let address = "127.0.0.1:0".parse().unwrap();
        (
            VeloResponseService::new(transport, address).await.unwrap(),
            VeloResponseService::new(transport, address).await.unwrap(),
        )
    }

    async fn lifecycle_suite(transport: ResponseTransport) {
        stop_drains_output_and_cancel_reaches_an_idle_engine(transport).await;
        completion_and_failure_keep_distinct_results(transport).await;
        prologue_preserves_typed_errors_and_rejects_version_mismatch(transport).await;
        blocked_stream_isolation_and_peer_failure(transport).await;
    }

    #[tokio::test]
    async fn tcp_lifecycle() {
        lifecycle_suite(ResponseTransport::Tcp).await;
    }

    #[cfg(feature = "velo-ucx")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ucx_lifecycle() {
        lifecycle_suite(ResponseTransport::Ucx).await;
    }

    async fn stop_drains_output_and_cancel_reaches_an_idle_engine(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        let client = EngineContext::new(()).context();
        let engine = EngineContext::new(()).context();
        let registered = consumer.register_response(client.clone()).unwrap();
        client.stop_generating();
        let mut sender = producer
            .sender(engine.clone(), registered.connection_info.clone(), None)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), engine.stopped())
            .await
            .unwrap();
        assert!(!engine.is_killed());
        sender.send_prologue(None).await.unwrap();
        let (_, provider) = registered.into_parts();
        let mut receiver = provider.await.unwrap().unwrap();
        // Cover both an ordinary item and an item above Velo's rendezvous
        // threshold. Mux-only services must send the latter in chunks.
        for size in [128 * 1024, 1024 * 1024] {
            let payload = Bytes::from(vec![7; size]);
            sender.send(payload.clone()).await.unwrap();
            assert_eq!(receiver.rx.next().await.unwrap().unwrap(), payload);
        }
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(5), engine.killed())
            .await
            .unwrap();
    }

    async fn completion_and_failure_keep_distinct_results(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        for complete in [true, false] {
            let registered = consumer
                .register_response(EngineContext::new(()).context())
                .unwrap();
            let mut sender = producer
                .sender(
                    EngineContext::new(()).context(),
                    registered.connection_info.clone(),
                    None,
                )
                .await
                .unwrap();
            sender.send_prologue(None).await.unwrap();
            let (_, provider) = registered.into_parts();
            let mut receiver = provider.await.unwrap().unwrap();
            sender.send(Bytes::from_static(b"last")).await.unwrap();
            if complete {
                sender.finish().await.unwrap();
            } else {
                sender.abort().await.unwrap();
            }
            assert_eq!(
                receiver.rx.next().await.unwrap().unwrap(),
                Bytes::from_static(b"last")
            );
            let terminal = receiver.rx.next().await;
            if complete {
                assert!(terminal.is_none());
            } else {
                assert!(terminal.unwrap().is_err());
            }
            assert!(receiver.rx.next().await.is_none());
        }
    }

    async fn blocked_stream_isolation_and_peer_failure(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        let mut streams = Vec::new();
        for _ in 0..2 {
            let registered = consumer
                .register_response(EngineContext::new(()).context())
                .unwrap();
            let engine = EngineContext::new(()).context();
            let mut sender = producer
                .sender(engine.clone(), registered.connection_info.clone(), None)
                .await
                .unwrap();
            sender.send_prologue(None).await.unwrap();
            let (_, provider) = registered.into_parts();
            streams.push((sender, provider.await.unwrap().unwrap(), engine));
        }
        let (blocked_sender, blocked_receiver, blocked_engine) = streams.remove(0);
        let (mut sender, mut receiver, engine) = streams.remove(0);
        let flood = tokio::spawn(async move {
            loop {
                blocked_sender.send(Bytes::from(vec![1; 4096])).await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        });
        sender
            .send(Bytes::from_static(b"independent"))
            .await
            .unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), receiver.rx.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            Bytes::from_static(b"independent")
        );
        drop(blocked_receiver);
        tokio::time::timeout(Duration::from_secs(5), blocked_engine.killed())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), flood)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        // Releasing one worker's service owner cannot shut down another stream.
        drop(producer);
        sender
            .send(Bytes::from_static(b"still alive"))
            .await
            .unwrap();
        assert!(receiver.rx.next().await.unwrap().is_ok());
        consumer
            .velo
            .graceful_shutdown(velo::ShutdownPolicy::Timeout(Duration::from_secs(1)))
            .await;
        tokio::time::timeout(Duration::from_secs(30), engine.killed())
            .await
            .unwrap();
        assert!(sender.send(Bytes::new()).await.is_err());
        sender.abort().await.unwrap();
    }

    async fn prologue_preserves_typed_errors_and_rejects_version_mismatch(
        transport: ResponseTransport,
    ) {
        let (consumer, producer) = pair(transport).await;
        let registered = consumer
            .register_response(EngineContext::new(()).context())
            .unwrap();
        let error = DynamoError::builder()
            .error_type(ErrorType::Disconnected)
            .message("backend failure")
            .build();
        let mut sender = producer
            .sender(
                EngineContext::new(()).context(),
                registered.connection_info.clone(),
                None,
            )
            .await
            .unwrap();
        sender
            .send_prologue(Some(StreamPrologueError::new(
                "backend failure",
                error.clone(),
            )))
            .await
            .unwrap();
        let (_, provider) = registered.into_parts();
        match provider.await.unwrap() {
            Err(actual) => {
                let actual = actual.typed_error.expect("typed prologue error");
                assert_eq!(actual.class(), error.class());
                assert_eq!(actual.reason(), error.reason());
                assert_eq!(actual.to_string(), error.to_string());
            }
            Ok(_) => panic!("error prologue succeeded"),
        }
        let registered = consumer
            .register_response(EngineContext::new(()).context())
            .unwrap();
        let mut address: ResponseAddress =
            serde_json::from_str(&registered.connection_info.info).unwrap();
        address.version -= 1;
        let info = ConnectionInfo {
            transport: TRANSPORT_NAME.into(),
            info: serde_json::to_string(&address).unwrap(),
        };
        assert!(
            producer
                .sender(EngineContext::new(()).context(), info, None)
                .await
                .is_err()
        );
    }
}

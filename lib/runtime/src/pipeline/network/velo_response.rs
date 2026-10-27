// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo owns response framing, batching, flow control, and stream lifecycle.

use anyhow::{Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    pin::Pin,
    sync::{
        Arc, LazyLock, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::{OnceCell, oneshot};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use velo::{
    InstanceId, PeerInfo, Velo, WorkerAddress,
    streaming::{
        AnchorConfig, AutoFlush, FlushPolicy, MuxConfig, StreamAnchor, StreamAnchorHandle,
        StreamController, StreamFrame, StreamSender, control::StreamOpenTicket,
    },
};

use super::{
    ConnectionInfo, RegisteredResponsePlane, RegisteredStream, ResponseStreamPrologue,
    StreamPrologueError, StreamReceiver, tombstones::InstanceTombstones,
};
use crate::{
    discovery::EndpointInstanceId,
    engine::AsyncEngineContext,
    error::{DynamoError, ErrorType},
};

pub const TRANSPORT_NAME: &str = "velo_response";
/// A worker reports a version mismatch over the stream it opens, so later
/// versions must keep `ResponseAddress`'s `version`, `peer_id`,
/// `peer_address`, `anchor` and `ticket` decodable.
const VERSION: u32 = 4;
/// Process-wide services, keyed by [`service_key`].
static PROCESS_SERVICES: LazyLock<
    tokio::sync::Mutex<HashMap<Option<tokio::runtime::Id>, Weak<SharedService>>>,
> = LazyLock::new(Default::default);
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

/// Per-stream credit. A larger window lets busy streams delay a new stream's
/// first token on the shared connection.
const RESPONSE_CREDIT_WINDOW: u32 = 32;

/// TCP lanes to each frontend. The frontend reads each lane on its own task,
/// so a single lane limits a busy worker's throughput.
const RESPONSE_TCP_LANES: u16 = 4;

/// A restarted frontend returns with a new instance ID, so a worker forgets a
/// frontend that has had no open stream for this long.
const PEER_IDLE_TTL: Duration = Duration::from_secs(600);
const PEER_SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Per-stream heartbeat, in place of Velo's 5 s. Velo does not report a closed
/// connection to its streams, so a frontend sees a lost worker only after
/// three to four missed heartbeats.
const RESPONSE_HEARTBEAT_MS: u64 = 1_000;

/// Reads the setting `name`, or returns `default` when it is unset.
fn env_setting<T>(
    name: &str,
    default: T,
    expected: &str,
    parse: impl FnOnce(&str) -> Option<T>,
) -> Result<T> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(default),
        Ok(value) => parse(&value).ok_or_else(|| anyhow::anyhow!("{name} must be {expected}")),
        Err(error) => bail!("{name}: {error}"),
    }
}

fn response_heartbeat() -> Result<Duration> {
    env_setting(
        crate::config::environment_names::response_plane::DYN_VELO_RESPONSE_HEARTBEAT_MS,
        RESPONSE_HEARTBEAT_MS,
        "a positive integer in milliseconds",
        |value| value.parse().ok().filter(|ms| *ms > 0),
    )
    .map(Duration::from_millis)
}

fn response_mux_config() -> Result<MuxConfig> {
    let batch_ms: u64 = env_setting(
        crate::config::environment_names::response_plane::DYN_VELO_RESPONSE_BATCH_MS,
        5,
        "a non-negative integer in milliseconds",
        |value| value.parse().ok(),
    )?;
    Ok(MuxConfig {
        enabled: true,
        initial_credit: RESPONSE_CREDIT_WINDOW,
        flush_policy: FlushPolicy::Auto(AutoFlush {
            on_admission: batch_ms == 0,
            max_linger: (batch_ms != 0).then(|| Duration::from_millis(batch_ms)),
        }),
        ..Default::default()
    })
}

/// Velo has no TLS, so refuse to start rather than send responses in
/// plaintext while TLS is configured for the TCP planes.
pub(crate) fn reject_tls_settings() -> Result<()> {
    use crate::config::environment_names::tcp_response_stream::tls::*;

    let set: Vec<&str> = [
        DYN_TCP_TLS_CERT_PATH,
        DYN_TCP_TLS_KEY_PATH,
        DYN_TCP_TLS_CA_CERT_PATH,
        DYN_TCP_TLS_CLIENT_CERT_PATH,
        DYN_TCP_TLS_CLIENT_KEY_PATH,
        DYN_TCP_TLS_CLIENT_CA_CERT_PATH,
    ]
    .into_iter()
    .filter(|name| std::env::var_os(name).is_some())
    .chain(crate::config::env_is_truthy(DYN_TCP_TLS_INSECURE).then_some(DYN_TCP_TLS_INSECURE))
    .collect();
    ensure!(
        set.is_empty(),
        "the Velo response plane has no TLS, but {} is set; unset it or use DYN_RESPONSE_PLANE=tcp",
        set.join(", ")
    );
    Ok(())
}

fn response_port() -> Result<u16> {
    env_setting(
        crate::config::environment_names::response_plane::DYN_VELO_RESPONSE_PORT,
        0,
        "a port number from 0 to 65535",
        |value| value.parse().ok(),
    )
}

fn tcp_transport(address: SocketAddr) -> Result<velo::transports::tcp::TcpTransport> {
    velo::transports::tcp::TcpTransportBuilder::new()
        .bind_addr(address)
        .lanes(RESPONSE_TCP_LANES)
        .build()
}

impl ResponseTransport {
    fn configured() -> Result<Self> {
        env_setting(
            crate::config::environment_names::response_plane::DYN_VELO_RESPONSE_TRANSPORT,
            Self::Tcp,
            "tcp or ucx",
            |value| match value {
                "tcp" => Some(Self::Tcp),
                "ucx" => Some(Self::Ucx),
                _ => None,
            },
        )
    }
}

/// The frontend's response endpoint, carried in every request.
#[derive(Serialize, Deserialize)]
struct ResponseAddress<'a> {
    version: u32,
    transport: ResponseTransport,
    peer_id: InstanceId,
    /// Base64 of the frontend's `WorkerAddress`, decoded only for a frontend
    /// the worker has not prepared.
    peer_address: &'a str,
    anchor: StreamAnchorHandle,
    ticket: StreamOpenTicket,
}

struct Registration {
    controller: StreamController,
    done: CancellationToken,
    instance: Option<EndpointInstanceId>,
}

impl Registration {
    fn cancel(&self) {
        self.done.cancel();
        self.controller.cancel();
    }
}

#[derive(Default)]
struct Registrations {
    requests: HashMap<Uuid, Registration>,
    /// Requests still waiting for their prologue. As on TCP, a request leaves
    /// this index when its prologue arrives, so worker removal lets it drain.
    instance_requests: HashMap<EndpointInstanceId, HashSet<Uuid>>,
    tombstones: InstanceTombstones,
}

impl Registrations {
    /// Unlinks an established request from its instance. Returns false when a
    /// cancellation already removed it.
    fn accept(&mut self, id: Uuid) -> bool {
        let Some(registration) = self.requests.get_mut(&id) else {
            return false;
        };
        if let Some(instance) = registration.instance.take()
            && let Some(ids) = self.instance_requests.get_mut(&instance)
        {
            ids.remove(&id);
            if ids.is_empty() {
                self.instance_requests.remove(&instance);
            }
        }
        true
    }

    fn take(&mut self, id: Uuid) -> Option<Registration> {
        let registration = self.requests.remove(&id)?;
        if let Some(instance) = &registration.instance
            && let Some(ids) = self.instance_requests.get_mut(instance)
        {
            ids.remove(&id);
            if ids.is_empty() {
                self.instance_requests.remove(instance);
            }
        }
        Some(registration)
    }
}

/// A service on an owned runtime keeps that runtime alive, so all owned
/// runtimes share one. A service on an External runtime stops with it, so only
/// callers on that Tokio runtime share it.
fn service_key(runtime: &crate::runtime::RuntimeType) -> Option<tokio::runtime::Id> {
    match runtime {
        crate::runtime::RuntimeType::Shared(_) => None,
        crate::runtime::RuntimeType::External(handle) => Some(handle.id()),
    }
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

    /// Refuses new acquisitions. Callers that already hold the service can
    /// still register responses, so migration retries work during a drain.
    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    /// Keeps the service open for `window`, while a frontend may still admit
    /// requests that have yet to register, then waits up to `headroom` more
    /// for its response streams to finish.
    pub(crate) async fn drain(&self, window: Duration, headroom: Duration) {
        let Some(service) = self
            .owner
            .lock()
            .await
            .as_ref()
            .map(|lease| lease.service.clone())
        else {
            return;
        };
        tokio::time::sleep(window).await;
        let _ = tokio::time::timeout(headroom, async {
            while !service.registrations.lock().requests.is_empty() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await;
    }

    pub(crate) async fn service(&self) -> Result<Arc<VeloResponseService>> {
        let mut owner = self.owner.lock().await;
        ensure!(!self.is_closed(), "Velo response runtime is shut down");
        if let Some(owner) = owner.as_ref() {
            return Ok(owner.service.clone());
        }
        // Acquisition and final-owner shutdown use the same lock. A new
        // runtime cannot acquire a service while that service is closing.
        let key = service_key(&self.runtime);
        let mut shared = PROCESS_SERVICES.lock().await;
        let lease = match shared.get(&key).and_then(Weak::upgrade) {
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
                shared.retain(|_, service| service.strong_count() > 0);
                shared.insert(key, Arc::downgrade(&lease));
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
        let mut shared = PROCESS_SERVICES.lock().await;
        if let Some(lease) = Arc::into_inner(lease) {
            shared.remove(&service_key(&lease.runtime));
            lease.service.shutdown().await;
        }
    }
}

/// Stream handles retain the service, but do not count as runtime owners.
struct SharedService {
    service: Arc<VeloResponseService>,
    // Keeps an owned Tokio runtime alive without retaining the Dynamo Runtime.
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

/// A frontend this worker has registered with Velo.
#[derive(Default)]
struct PreparedPeer {
    /// Registration and the lifecycle handshake, shared by concurrent requests.
    ready: OnceCell<()>,
    /// Open senders to this frontend. Only a peer with none is evicted.
    active: AtomicUsize,
    /// When its last sender finished, in seconds since the service started.
    last_used: AtomicU64,
}

/// Keeps a prepared frontend from eviction while one sender uses it.
struct PeerLease {
    peer: Arc<PreparedPeer>,
    started: tokio::time::Instant,
}

impl PeerLease {
    fn new(peer: &Arc<PreparedPeer>, started: tokio::time::Instant) -> Self {
        peer.active.fetch_add(1, Ordering::AcqRel);
        Self {
            peer: peer.clone(),
            started,
        }
    }
}

impl Drop for PeerLease {
    fn drop(&mut self) {
        // Stamp before releasing, so a sweep that sees no senders also sees the time.
        self.peer
            .last_used
            .store(self.started.elapsed().as_secs(), Ordering::Relaxed);
        self.peer.active.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct VeloResponseService {
    velo: Arc<Velo>,
    transport: ResponseTransport,
    heartbeat: Duration,
    /// This service's `ResponseAddress::peer_address`, encoded once.
    peer_address: String,
    registrations: Mutex<Registrations>,
    /// Frontends this worker serves, by instance ID. Idle ones are evicted.
    prepared_peers: dashmap::DashMap<velo::InstanceId, Arc<PreparedPeer>>,
    started: tokio::time::Instant,
    stop_sweep: CancellationToken,
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
        Self::new(
            transport,
            SocketAddr::new(host.advertise_ip(), response_port()?),
        )
        .await
    }

    async fn new(transport: ResponseTransport, address: SocketAddr) -> Result<Arc<Self>> {
        let heartbeat = response_heartbeat()?;
        let mut builder = Velo::builder()
            .metrics(PROCESS_METRICS.1.clone())
            .mux_only()
            .messenger_mux(response_mux_config()?)?;
        if transport == ResponseTransport::Ucx {
            // Added first, so Velo prefers UCX for every peer that has it.
            #[cfg(all(target_os = "linux", feature = "velo-ucx"))]
            {
                builder = builder.add_transport(Arc::new(
                    velo::transports::ucx::UcxTransportBuilder::new().build()?,
                ));
            }
            #[cfg(not(all(target_os = "linux", feature = "velo-ucx")))]
            bail!("Velo UCX responses require a Linux build with the velo-ucx feature");
        }
        // Every service also has TCP, so a worker configured for another
        // transport can still open the stream and report the mismatch.
        builder = builder.add_transport(Arc::new(tcp_transport(address)?));
        let velo = builder.build().await?;
        let service = Arc::new(Self {
            peer_address: BASE64.encode(velo.peer_info().worker_address().as_bytes()),
            velo,
            transport,
            heartbeat,
            registrations: Mutex::new(Registrations::default()),
            prepared_peers: dashmap::DashMap::new(),
            started: tokio::time::Instant::now(),
            stop_sweep: CancellationToken::new(),
            closing: AtomicBool::new(false),
            closed: OnceCell::new(),
        });
        let weak = Arc::downgrade(&service);
        let stop = service.stop_sweep.clone();
        tokio::spawn(async move {
            let mut sweep = tokio::time::interval(PEER_SWEEP_INTERVAL);
            sweep.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            sweep.tick().await;
            loop {
                tokio::select! {
                    _ = stop.cancelled() => return,
                    _ = sweep.tick() => {}
                }
                let Some(service) = weak.upgrade() else {
                    return;
                };
                let evicted = service.evict_idle_peers(service.started.elapsed().as_secs());
                if evicted > 0 {
                    tracing::debug!(evicted, "Forgot idle Velo response peers");
                }
            }
        });
        Ok(service)
    }

    /// Forget frontends with no open sender that have been idle for
    /// [`PEER_IDLE_TTL`], or whose preparation failed. `now` is in seconds
    /// since the service started.
    fn evict_idle_peers(&self, now: u64) -> usize {
        let own = self.velo.instance_id();
        let mut evicted = 0;
        self.prepared_peers.retain(|peer_id, peer| {
            let idle = *peer_id != own
                && peer.active.load(Ordering::Acquire) == 0
                && (!peer.ready.initialized()
                    || now.saturating_sub(peer.last_used.load(Ordering::Relaxed))
                        >= PEER_IDLE_TTL.as_secs());
            if idle {
                // Under the shard lock, so no request can prepare this peer
                // again until Velo has forgotten it.
                self.velo.unregister_peer(*peer_id);
                evicted += 1;
            }
            !idle
        });
        evicted
    }

    async fn shutdown(&self) {
        self.closed
            .get_or_init(|| async {
                self.closing.store(true, Ordering::Release);
                self.stop_sweep.cancel();
                let requests = {
                    let mut state = self.registrations.lock();
                    state.instance_requests.clear();
                    std::mem::take(&mut state.requests)
                };
                for registration in requests.into_values() {
                    registration.cancel();
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
        defer_cancellation_until_prologue: bool,
    ) -> Result<RegisteredStream<StreamReceiver>> {
        ensure!(
            !self.closing.load(Ordering::Acquire),
            "Velo response service is shut down"
        );
        // Item 0 is the JSON prologue; later items are response data. The
        // ticket carries the heartbeat interval to the worker.
        let anchor = self
            .velo
            .anchor_manager()
            .create_anchor_with_config::<Bytes>(AnchorConfig {
                heartbeat_interval: Some(self.heartbeat),
                ..Default::default()
            });
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
                peer_id: self.velo.instance_id(),
                peer_address: &self.peer_address,
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
                    _ = context.killed(), if !defer_cancellation_until_prologue => {
                        tracing::debug!(context_id = %context.id(), "Cancelling Velo response stream");
                        controller.cancel(); return;
                    }
                    _ = tx.closed() => return,
                    _ = context.stopped(), if !stopped && !defer_cancellation_until_prologue => {
                        tracing::debug!(context_id = %context.id(), "Stopping Velo response generation");
                        controller.request_stop(); stopped = true;
                    }
                    first = anchor.next_bytes() => break first,
                }
            };
            match first {
                Some(Ok(StreamFrame::Item(prologue))) => {
                    let prologue = match serde_json::from_slice::<ResponseStreamPrologue>(&prologue)
                    {
                        Ok(prologue) => prologue,
                        Err(error) => {
                            let _ = tx.send(Err(StreamPrologueError::from_message(format!(
                                "Invalid Velo response prologue: {error}"
                            ))));
                            return;
                        }
                    };
                    if let Some(error) = prologue.error {
                        tracing::debug!(context_id = %context.id(), %error, "Velo response prologue returned an error");
                        let _ = tx.send(Err(StreamPrologueError {
                            message: error,
                            typed_error: prologue.typed_error,
                        }));
                        return;
                    }
                }
                Some(Err(velo::streaming::StreamError::SenderError(error))) => {
                    // The worker opened the stream only to report why it cannot serve it.
                    tracing::debug!(context_id = %context.id(), %error, "Velo response worker reported a setup error");
                    let _ = tx.send(Err(StreamPrologueError::from_message(error)));
                    return;
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
            if !lease.service.registrations.lock().accept(lease.id) {
                return;
            }
            let receiver = VeloResponseReceiver {
                anchor,
                lease,
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
        let registration = self.registrations.lock().take(id);
        if let Some(registration) = registration {
            registration.cancel();
        }
    }

    fn complete(&self, id: Uuid) {
        let registration = self.registrations.lock().take(id);
        if let Some(registration) = registration {
            registration.done.cancel();
        }
    }

    /// `address` is the peer's base64 `WorkerAddress`, decoded only if the peer
    /// is not prepared yet. The lease is taken under the map guard, so the
    /// sweep cannot evict the peer while it is in use.
    async fn prepare_peer(&self, peer_id: InstanceId, address: &str) -> Result<PeerLease> {
        // Steady state: the peer is already prepared, so skip the shard write lock.
        if let Some(peer) = self.prepared_peers.get(&peer_id)
            && peer.ready.initialized()
        {
            return Ok(PeerLease::new(&peer, self.started));
        }
        let lease = PeerLease::new(
            &self.prepared_peers.entry(peer_id).or_default(),
            self.started,
        );
        // Release the map guard before awaiting. A failed or cancelled attempt
        // leaves the cell empty, so the next caller can retry.
        lease
            .peer
            .ready
            .get_or_try_init(|| async {
                let address = WorkerAddress::from_encoded(BASE64.decode(address)?);
                self.velo.register_peer(PeerInfo::new(peer_id, address))?;
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
        Ok(lease)
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
            address.ticket.streaming_transport_key.as_str() == velo::streaming::MESSENGER_MUX_KEY,
            "unsupported Velo response stream transport"
        );
        ensure!(
            address.anchor.unpack().0 == address.peer_id.worker_id(),
            "Velo anchor and peer identity differ"
        );
        let peer = self
            .prepare_peer(address.peer_id, address.peer_address)
            .await
            .inspect_err(|error| {
            tracing::warn!(context_id = %context.id(), %error, "Failed to prepare Velo response peer");
        })?;
        let sender = if address.anchor.unpack().0 == self.velo.instance_id().worker_id() {
            self.velo.attach_anchor::<Bytes>(address.anchor).await?
        } else {
            self.velo
                .open_anchor_stream::<Bytes>(address.anchor, address.ticket)
                .await?
        };
        // Check compatibility only once the stream is open, so the frontend
        // gets the mismatch at once. `send_err` does not use this protocol's frames.
        let mismatch = if address.version != VERSION {
            Some(format!(
                "Velo response protocol mismatch: frontend version {}, worker version {VERSION}",
                address.version
            ))
        } else if address.transport != self.transport {
            Some(format!(
                "Velo response transport mismatch: frontend {:?}, worker {:?}",
                address.transport, self.transport
            ))
        } else {
            None
        };
        if let Some(message) = mismatch {
            let _ = sender.send_err(&message).await;
            let _ = sender.finalize();
            bail!(message);
        }
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
                        if !stopped && let Some(counter) = &cancellation { counter.inc(); }
                        tracing::info!(target: "request_span",
                            request_id = context.id(),
                            {"cancellation.signal"} = "kill",
                            {"cancellation.source"} = "upstream",
                            "request cancellation received");
                        tracing::debug!(context_id = %context.id(), "Velo response peer cancelled the request");
                        context.kill(); return;
                    }
                    _ = monitor_done.cancelled() => return,
                    _ = stop.cancelled(), if !stopped => {
                        if let Some(counter) = &cancellation { counter.inc(); }
                        tracing::info!(target: "request_span",
                            request_id = context.id(),
                            {"cancellation.signal"} = "stop",
                            {"cancellation.source"} = "upstream",
                            "request cancellation received");
                        tracing::debug!(context_id = %context.id(), "Velo response peer requested a stop");
                        context.stop_generating(); stopped = true;
                    }
                }
            }
        });
        Ok(VeloResponseSender {
            sender: Some(sender),
            _peer: peer,
            _service: self.clone(),
            finished,
            prologue_sent: false,
        })
    }
}

#[async_trait::async_trait]
impl RegisteredResponsePlane for VeloResponseService {
    fn register_stream(
        self: Arc<Self>,
        context: Arc<dyn AsyncEngineContext>,
        defer_cancellation_until_prologue: bool,
    ) -> Result<RegisteredStream<StreamReceiver>> {
        VeloResponseService::register_response(&self, context, defer_cancellation_until_prologue)
    }

    async fn associate_instance(&self, id: Uuid, instance: &EndpointInstanceId) -> bool {
        let mut state = self.registrations.lock();
        if state.tombstones.contains(instance) {
            let registration = state.take(id);
            drop(state);
            if let Some(registration) = registration {
                registration.cancel();
            }
            return false;
        }
        if let Some(request) = state.requests.get_mut(&id) {
            debug_assert!(request.instance.is_none(), "registration associated twice");
            request.instance = Some(instance.clone());
            state
                .instance_requests
                .entry(instance.clone())
                .or_default()
                .insert(id);
        }
        true
    }

    async fn cancel_response(&self, id: Uuid) {
        self.remove(id);
    }

    async fn cancel_instance_streams(&self, instance: &EndpointInstanceId) -> usize {
        let registrations = {
            let mut state = self.registrations.lock();
            state.tombstones.insert(instance);
            let ids = state.instance_requests.remove(instance).unwrap_or_default();
            ids.into_iter()
                .filter_map(|id| state.requests.remove(&id))
                .collect::<Vec<_>>()
        };
        for registration in &registrations {
            registration.cancel();
        }
        registrations.len()
    }

    async fn clear_instance_tombstone(&self, instance: &EndpointInstanceId) {
        self.registrations.lock().tombstones.remove(instance);
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

fn first_kind(frame: &Option<Result<StreamFrame<Bytes>, velo::streaming::StreamError>>) -> String {
    match frame {
        None => "closed".into(),
        Some(Err(error)) => error.to_string(),
        Some(Ok(_)) => "unexpected frame".into(),
    }
}

pub(super) struct VeloResponseReceiver {
    anchor: StreamAnchor<Bytes>,
    lease: ResponseLease,
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
        match std::task::ready!(this.anchor.poll_next_bytes(cx)) {
            Some(Ok(StreamFrame::Item(bytes))) => Poll::Ready(Some(Ok(bytes))),
            Some(Ok(StreamFrame::Finalized)) => {
                this.ended = true;
                this.lease.service.complete(this.lease.id);
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
    sender: Option<StreamSender<Bytes>>,
    _peer: PeerLease,
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
            .send_bytes(bytes)
            .await?;
        Ok(())
    }
    pub async fn send_prologue(&mut self, error: Option<StreamPrologueError>) -> Result<()> {
        ensure!(!self.prologue_sent, "Velo response prologue already sent");
        let (error, typed_error) = error.map_or((None, None), |e| (Some(e.message), e.typed_error));
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Velo response closed"))?
            .send_bytes(serde_json::to_vec(&ResponseStreamPrologue {
                error,
                typed_error,
            })?)
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

    #[test]
    fn response_batch_policy() {
        use crate::config::environment_names::response_plane::DYN_VELO_RESPONSE_BATCH_MS;

        if crate::test_utils::run_isolated(concat!(module_path!(), "::response_batch_policy"), &[])
        {
            return;
        }
        for (value, on_admission, max_linger) in [
            (None, false, Some(Duration::from_millis(5))),
            (Some("0"), true, None),
            (Some("12"), false, Some(Duration::from_millis(12))),
        ] {
            temp_env::with_var(DYN_VELO_RESPONSE_BATCH_MS, value, || {
                let config = response_mux_config().unwrap();
                assert_eq!(
                    config.flush_policy,
                    FlushPolicy::Auto(AutoFlush {
                        on_admission,
                        max_linger
                    })
                );
            });
        }
        for value in ["", "-1", "1.5", "invalid", "18446744073709551616"] {
            temp_env::with_var(DYN_VELO_RESPONSE_BATCH_MS, Some(value), || {
                assert!(
                    response_mux_config()
                        .unwrap_err()
                        .to_string()
                        .contains(DYN_VELO_RESPONSE_BATCH_MS)
                );
            });
        }
    }

    #[tokio::test]
    async fn runtime_shutdown_closes_only_the_last_owner_after_its_streams_drain() {
        temp_env::async_with_vars(
            [
                ("DYN_TCP_RESPONSE_STREAM_HOST", Some("127.0.0.1")),
                ("DYN_VELO_RESPONSE_TRANSPORT", Some("tcp")),
                ("DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS", Some("1")),
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
                let address = tcp_address(&service);

                first.shutdown();
                tokio::time::timeout(Duration::from_secs(10), first.primary_token().cancelled())
                    .await
                    .unwrap();
                assert!(first_clone.velo_response_service().service().await.is_err());
                drop(first_clone);
                drop(first);
                assert!(!service.closed.initialized());
                tokio::net::TcpStream::connect(address).await.unwrap();

                // Retain both the service and an in-flight registration across shutdown.
                let registration = service
                    .register_response(EngineContext::new(()).context(), false)
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
                // The main token starts the HTTP drain. The service refuses new
                // owners, but its holders can still register during the drain,
                // and it stays open past the drain while a response is in flight.
                assert!(second.velo_response_service().service().await.is_err());
                drop(
                    service
                        .register_response(EngineContext::new(()).context(), false)
                        .unwrap(),
                );
                tokio::time::sleep(Duration::from_millis(1_500)).await;
                assert!(!service.closed.initialized());
                tokio::net::TcpStream::connect(address).await.unwrap();
                drop(registration);
                tokio::time::timeout(Duration::from_secs(10), async {
                    while !service.closed.initialized() {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await
                .unwrap();
                assert!(tokio::net::TcpStream::connect(address).await.is_err());
                assert!(service.registrations.lock().requests.is_empty());
                assert!(
                    service
                        .register_response(EngineContext::new(()).context(), false)
                        .is_err()
                );

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

    fn tcp_address(service: &VeloResponseService) -> SocketAddr {
        let entry = service
            .velo
            .peer_info()
            .worker_address()
            .get_entry("tcp")
            .unwrap()
            .unwrap();
        velo::transports::utils::interfaces::parse_endpoints(&entry).unwrap()[0]
            .socket_addr()
            .unwrap()
    }

    #[tokio::test]
    async fn configured_port_binds_the_listener() {
        // Bound and released, so the port is very likely still free.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
            .to_string();
        if crate::test_utils::run_isolated(
            concat!(module_path!(), "::configured_port_binds_the_listener"),
            &[
                ("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1"),
                ("DYN_VELO_RESPONSE_PORT", &port),
            ],
        ) {
            return;
        }
        let service = VeloResponseService::from_env().await.unwrap();
        assert_eq!(
            tcp_address(&service).port().to_string(),
            std::env::var("DYN_VELO_RESPONSE_PORT").unwrap()
        );
        service.shutdown().await;
    }

    /// A service built on an External runtime holds only a `Handle`, so its
    /// listeners die with that Tokio runtime. A Runtime on another Tokio
    /// runtime must get its own service rather than share that one.
    #[test]
    fn external_runtimes_get_their_own_service() {
        temp_env::with_vars(
            [
                ("DYN_TCP_RESPONSE_STREAM_HOST", Some("127.0.0.1")),
                ("DYN_VELO_RESPONSE_TRANSPORT", Some("tcp")),
            ],
            || {
                let tokio_runtime = || {
                    tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(1)
                        .enable_all()
                        .build()
                        .unwrap()
                };
                let first_tokio = tokio_runtime();
                let second_tokio = tokio_runtime();
                let service_on = |tokio: &tokio::runtime::Runtime| {
                    tokio.block_on(async {
                        let runtime = crate::Runtime::from_current().unwrap();
                        let service = runtime.velo_response_service().service().await.unwrap();
                        (runtime, service)
                    })
                };
                let (first_owner, first) = service_on(&first_tokio);
                let (second_owner, second) = service_on(&second_tokio);
                assert!(!Arc::ptr_eq(&first, &second));

                drop((first_owner, first));
                drop(first_tokio);
                second_tokio.block_on(async {
                    tokio::net::TcpStream::connect(tcp_address(&second))
                        .await
                        .unwrap();
                    second_owner.shutdown();
                    tokio::time::timeout(
                        Duration::from_secs(10),
                        second_owner.primary_token().cancelled(),
                    )
                    .await
                    .unwrap();
                });
            },
        );
    }

    /// Runs alone: it counts `_hello` events with a thread-scoped subscriber,
    /// and other tests' handshakes race it through tracing's global callsite
    /// interest cache.
    #[tokio::test]
    async fn concurrent_peer_preparation_shares_a_handshake_and_retries_failure() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::concurrent_peer_preparation_shares_a_handshake_and_retries_failure"
            ),
            &[],
        ) {
            return;
        }
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
        let peer = consumer.velo.instance_id();
        let address = consumer.peer_address.as_str();
        assert!(producer.prepare_peer(peer, "").await.is_err());
        let results =
            futures::future::join_all((0..8).map(|_| producer.prepare_peer(peer, address))).await;
        for result in results {
            result.unwrap();
        }
        assert_eq!(hellos.load(Ordering::Relaxed), 1);
        // A prepared peer's address is not decoded again.
        producer.prepare_peer(peer, "not base64").await.unwrap();
        assert_eq!(hellos.load(Ordering::Relaxed), 1);
        producer.shutdown().await;
        consumer.shutdown().await;
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

    /// UCX services also listen on TCP. A frontend and worker set to different
    /// transports reach each other over TCP, and the worker reports the
    /// mismatch at once. Two UCX services still pick UCX.
    #[cfg(feature = "velo-ucx")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn mismatched_transports_report_over_shared_tcp() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::mismatched_transports_report_over_shared_tcp"
            ),
            &[],
        ) {
            return;
        }
        let address = "127.0.0.1:0".parse().unwrap();
        let ucx = VeloResponseService::new(ResponseTransport::Ucx, address)
            .await
            .unwrap();
        let ucx_peer = VeloResponseService::new(ResponseTransport::Ucx, address)
            .await
            .unwrap();
        let tcp = VeloResponseService::new(ResponseTransport::Tcp, address)
            .await
            .unwrap();
        for (frontend, worker) in [(&ucx, &tcp), (&tcp, &ucx)] {
            let registered = frontend
                .register_response(EngineContext::new(()).context(), false)
                .unwrap();
            let error = worker
                .sender(
                    EngineContext::new(()).context(),
                    registered.connection_info.clone(),
                    None,
                )
                .await
                .err()
                .expect("mismatched worker opened a sender");
            assert!(error.to_string().contains("transport mismatch"), "{error}");
            match tokio::time::timeout(Duration::from_secs(5), registered.wait())
                .await
                .expect("frontend waited for a mismatched worker")
                .unwrap()
            {
                Err(error) => assert!(error.message.contains("transport mismatch")),
                Ok(_) => panic!("mismatched worker's stream succeeded"),
            }
            assert_eq!(
                worker
                    .velo
                    .primary_transport_key(frontend.velo.instance_id())
                    .as_deref(),
                Some("tcp")
            );
        }
        let registered = ucx
            .register_response(EngineContext::new(()).context(), false)
            .unwrap();
        let mut sender = ucx_peer
            .sender(
                EngineContext::new(()).context(),
                registered.connection_info.clone(),
                None,
            )
            .await
            .unwrap();
        sender.send_prologue(None).await.unwrap();
        registered.wait().await.unwrap().unwrap();
        assert_eq!(
            ucx_peer
                .velo
                .primary_transport_key(ucx.velo.instance_id())
                .as_deref(),
            Some("ucx")
        );
        drop(sender);
        for service in [ucx, ucx_peer, tcp] {
            service.shutdown().await;
        }
    }

    /// A worker forgets a frontend that is idle or failed to prepare, in its
    /// own map and in Velo, and prepares it again on its next request. An open
    /// sender keeps its frontend.
    #[tokio::test]
    async fn idle_peers_are_forgotten_and_prepared_again() {
        let (consumer, producer) = pair(ResponseTransport::Tcp).await;
        let peer = consumer.velo.instance_id();
        assert!(producer.prepare_peer(peer, "").await.is_err());
        assert_eq!(producer.evict_idle_peers(0), 1);

        let lease = producer
            .prepare_peer(peer, &consumer.peer_address)
            .await
            .unwrap();
        assert_eq!(producer.evict_idle_peers(u64::MAX), 0);
        drop(lease);
        let used = producer
            .prepared_peers
            .get(&peer)
            .unwrap()
            .last_used
            .load(Ordering::Relaxed);
        let ttl = PEER_IDLE_TTL.as_secs();
        assert_eq!(producer.evict_idle_peers(used + ttl - 1), 0);
        assert_eq!(producer.evict_idle_peers(used + ttl), 1);
        assert!(producer.prepared_peers.is_empty());
        assert_eq!(producer.velo.primary_transport_key(peer), None);

        let registered = consumer
            .register_response(EngineContext::new(()).context(), false)
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
        registered.wait().await.unwrap().unwrap();
        assert_eq!(
            producer.velo.primary_transport_key(peer).as_deref(),
            Some("tcp")
        );
        sender.finish().await.unwrap();
        producer.shutdown().await;
        consumer.shutdown().await;
    }

    async fn lifecycle_suite(transport: ResponseTransport) {
        completion_and_failure_keep_distinct_results(transport).await;
        stop_drains_output_and_cancel_reaches_an_idle_engine(transport).await;
        prologue_preserves_typed_errors_and_rejects_version_mismatch(transport).await;
        blocked_stream_isolation_and_peer_failure(transport).await;
        deferred_cancellation_ends_at_prologue_or_registration_removal(transport).await;
    }

    #[tokio::test]
    async fn tcp_lifecycle() {
        if crate::test_utils::run_isolated(concat!(module_path!(), "::tcp_lifecycle"), &[]) {
            return;
        }
        lifecycle_suite(ResponseTransport::Tcp).await;
    }

    /// A worker that dies leaves its streams open on the frontend until they
    /// miss enough heartbeats. The short interval here bounds that wait.
    #[tokio::test]
    async fn lost_worker_ends_its_streams_after_missed_heartbeats() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::lost_worker_ends_its_streams_after_missed_heartbeats"
            ),
            &[("DYN_VELO_RESPONSE_HEARTBEAT_MS", "200")],
        ) {
            return;
        }
        let address: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let consumer = VeloResponseService::new(ResponseTransport::Tcp, address)
            .await
            .unwrap();
        let registered = consumer
            .register_response(EngineContext::new(()).context(), false)
            .unwrap();
        // The worker runs on its own Tokio runtime, so shutting that runtime
        // down stops its tasks and closes its sockets, as a crash would.
        let worker = tokio::runtime::Runtime::new().unwrap();
        let info = registered.connection_info.clone();
        let (opened_tx, opened_rx) = oneshot::channel();
        worker.spawn(async move {
            let producer = VeloResponseService::new(ResponseTransport::Tcp, address)
                .await
                .unwrap();
            let mut sender = producer
                .sender(EngineContext::new(()).context(), info, None)
                .await
                .unwrap();
            sender.send_prologue(None).await.unwrap();
            sender.send(Bytes::from_static(b"token")).await.unwrap();
            opened_tx.send(()).unwrap();
            std::future::pending::<()>().await;
            drop((producer, sender));
        });
        opened_rx.await.unwrap();
        let (_, provider) = registered.into_parts();
        let mut receiver = provider.await.unwrap().unwrap();
        assert_eq!(
            receiver.rx.next().await.unwrap().unwrap(),
            Bytes::from_static(b"token")
        );
        worker.shutdown_background();
        // At the 1 s default this takes 3 to 4 s, and at Velo's 5 s 15 to 20 s.
        let error = tokio::time::timeout(Duration::from_millis(2_500), receiver.rx.next())
            .await
            .expect("the frontend did not notice the lost worker")
            .unwrap()
            .unwrap_err();
        assert_eq!(error.error_type(), ErrorType::Disconnected, "{error}");
        consumer.shutdown().await;
    }

    #[cfg(feature = "velo-ucx")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ucx_lifecycle() {
        if crate::test_utils::run_isolated(concat!(module_path!(), "::ucx_lifecycle"), &[]) {
            return;
        }
        lifecycle_suite(ResponseTransport::Ucx).await;
    }

    async fn stop_drains_output_and_cancel_reaches_an_idle_engine(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        let client = EngineContext::new(()).context();
        let engine = EngineContext::new(()).context();
        let registered = consumer.register_response(client.clone(), false).unwrap();
        client.stop_generating();
        let cancellations = prometheus::IntCounter::new("cancellations", "test").unwrap();
        let mut sender = producer
            .sender(
                engine.clone(),
                registered.connection_info.clone(),
                Some(cancellations.clone()),
            )
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), engine.stopped())
            .await
            .unwrap();
        assert!(!engine.is_killed());
        assert_eq!(cancellations.get(), 1);
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
        assert_eq!(cancellations.get(), 1, "Stop then Kill counts once");
    }

    fn cancellation_operations() -> f64 {
        PROCESS_METRICS
            .0
            .prometheus_registry
            .read()
            .unwrap()
            .gather()
            .iter()
            .filter(|family| family.name() == "velo_streaming_anchor_operations_total")
            .flat_map(|family| family.get_metric())
            .filter(|metric| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == "operation" && label.value() == "cancel")
            })
            .map(|metric| metric.get_counter().value())
            .sum()
    }

    async fn completion_and_failure_keep_distinct_results(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        for complete in [true, false] {
            let registered = consumer
                .register_response(EngineContext::new(()).context(), false)
                .unwrap();
            let id = registered.registration_id().unwrap();
            let done = consumer.registrations.lock().requests[&id].done.clone();
            let cancelled_before = cancellation_operations();
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
            if complete {
                assert!(done.is_cancelled());
                drop(receiver);
                // The TCP suite runs in isolation so other tests cannot change this counter.
                if transport == ResponseTransport::Tcp {
                    assert_eq!(cancellation_operations(), cancelled_before);
                }
            }
        }
    }

    async fn blocked_stream_isolation_and_peer_failure(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        let mut streams = Vec::new();
        for _ in 0..2 {
            let registered = consumer
                .register_response(EngineContext::new(()).context(), false)
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
        let credit_exhaustions = || {
            PROCESS_METRICS
                .0
                .prometheus_registry
                .read()
                .unwrap()
                .gather()
                .iter()
                .filter(|family| family.name() == "velo_streaming_slot_credit_exhausted_total")
                .flat_map(|family| family.get_metric())
                .map(|metric| metric.get_counter().value())
                .sum::<f64>()
        };
        let exhausted_before = credit_exhaustions();
        let flood = tokio::spawn(async move {
            loop {
                blocked_sender.send(Bytes::from(vec![1; 4096])).await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        });
        // Both suites run in isolation; only the unpolled stream can exhaust credit here.
        tokio::time::timeout(Duration::from_secs(5), async {
            while credit_exhaustions() <= exhausted_before {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("unpolled stream did not exhaust its credit");
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

    async fn deferred_cancellation_ends_at_prologue_or_registration_removal(
        transport: ResponseTransport,
    ) {
        let (consumer, producer) = pair(transport).await;
        let instance = EndpointInstanceId {
            namespace: "test".into(),
            component: "worker".into(),
            endpoint: "generate".into(),
            instance_id: 1,
        };
        for remove in [false, true] {
            let client = EngineContext::new(()).context();
            client.kill();
            let registered = consumer.register_response(client, true).unwrap();
            let id = registered.registration_id().unwrap();
            let engine = EngineContext::new(()).context();
            let cancellations = prometheus::IntCounter::new("cancellations", "test").unwrap();
            let mut sender = producer
                .sender(
                    engine.clone(),
                    registered.connection_info.clone(),
                    Some(cancellations.clone()),
                )
                .await
                .unwrap();
            assert!(!engine.is_stopped());
            let (_, mut provider) = registered.into_parts();
            assert!(matches!(
                provider.try_recv(),
                Err(oneshot::error::TryRecvError::Empty)
            ));
            let _receiver = if remove {
                consumer.cancel_response(id).await;
                assert!(provider.await.is_err());
                assert!(consumer.associate_instance(id, &instance).await);
                consumer.cancel_instance_streams(&instance).await;
                assert!(!consumer.associate_instance(id, &instance).await);
                None
            } else {
                sender.send_prologue(None).await.unwrap();
                Some(provider.await.unwrap().unwrap())
            };
            tokio::time::timeout(Duration::from_secs(5), engine.killed())
                .await
                .unwrap();
            assert_eq!(cancellations.get(), 1, "Kill alone counts once");
        }
    }

    async fn prologue_preserves_typed_errors_and_rejects_version_mismatch(
        transport: ResponseTransport,
    ) {
        let (consumer, producer) = pair(transport).await;
        for diagnostic in [None, Some("private backend detail")] {
            let registered = consumer
                .register_response(EngineContext::new(()).context(), false)
                .unwrap();
            let mut builder = DynamoError::builder()
                .error_type(ErrorType::InvalidRequest)
                .public_details(crate::error::PublicDetails::SizeLimit {
                    limit: 10,
                    actual: Some(11),
                });
            if let Some(diagnostic) = diagnostic {
                builder = builder.diagnostic(diagnostic);
            }
            let error = builder.build();
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
            match registered.wait().await.unwrap() {
                Err(actual) => {
                    let actual = actual.typed_error.expect("typed prologue error");
                    assert_eq!(actual.class(), ErrorType::InvalidRequest);
                    assert_eq!(actual.reason(), error.reason());
                    assert_eq!(actual.public_details(), error.public_details());
                    assert_eq!(
                        actual.diagnostic().map(|d| d.as_str()).unwrap_or_default(),
                        diagnostic.unwrap_or_default()
                    );
                }
                Ok(_) => panic!("error prologue succeeded"),
            }
        }
        let registered = consumer
            .register_response(EngineContext::new(()).context(), false)
            .unwrap();
        let sender = producer
            .sender(
                EngineContext::new(()).context(),
                registered.connection_info.clone(),
                None,
            )
            .await
            .unwrap();
        sender
            .sender
            .as_ref()
            .unwrap()
            .send_bytes(b"{invalid")
            .await
            .unwrap();
        match registered.wait().await.unwrap() {
            Err(error) => assert!(error.message.contains("Invalid Velo response prologue")),
            Ok(_) => panic!("malformed prologue succeeded"),
        }
        // A version or transport mismatch reaches the frontend over the opened
        // stream at once, rather than as a stream that never starts.
        type Skew = fn(&mut ResponseAddress);
        let skews: [(Skew, &str); 2] = [
            (|address| address.version -= 1, "protocol mismatch"),
            (
                |address| {
                    address.transport = match address.transport {
                        ResponseTransport::Tcp => ResponseTransport::Ucx,
                        ResponseTransport::Ucx => ResponseTransport::Tcp,
                    }
                },
                "transport mismatch",
            ),
        ];
        for (skew, expected) in skews {
            let registered = consumer
                .register_response(EngineContext::new(()).context(), false)
                .unwrap();
            let info = registered.connection_info.info.clone();
            let mut address: ResponseAddress = serde_json::from_str(&info).unwrap();
            skew(&mut address);
            let info = ConnectionInfo {
                transport: TRANSPORT_NAME.into(),
                info: serde_json::to_string(&address).unwrap(),
            };
            let error = producer
                .sender(EngineContext::new(()).context(), info, None)
                .await
                .err()
                .expect("mismatched worker opened a sender");
            assert!(error.to_string().contains(expected), "{error}");
            match tokio::time::timeout(Duration::from_secs(5), registered.wait())
                .await
                .expect("frontend waited for a mismatched worker")
                .unwrap()
            {
                Err(error) => assert!(error.message.contains(expected), "{}", error.message),
                Ok(_) => panic!("mismatched worker's stream succeeded"),
            }
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared TCP Server with Endpoint Multiplexing
//!
//! Provides a shared TCP server that can handle multiple endpoints on a single port
//! by adding endpoint routing to the TCP wire protocol.

use crate::SystemHealth;
use crate::metrics::work_handler_pool::{
    WORK_HANDLER_ENQUEUE_REJECTED_TOTAL, WORK_HANDLER_PERMIT_WAIT_SECONDS,
    WORK_HANDLER_POOL_ACTIVE_TASKS, WORK_HANDLER_POOL_CAPACITY, WORK_HANDLER_QUEUE_CAPACITY,
    WORK_HANDLER_QUEUE_DEPTH,
};
use crate::pipeline::network::PushWorkHandler;
use crate::{protocols::EndpointId, transports::tcp::instance_path};
use anyhow::{Context, Result};
use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;

type BoxRead = Box<dyn AsyncRead + Unpin + Send>;
type BoxWrite = Box<dyn AsyncWrite + Unpin + Send>;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio_util::bytes::BytesMut;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

/// Default worker pool size for TCP request handling
const DEFAULT_WORKER_POOL_SIZE: usize = 10000;

/// Default work queue size for TCP request handling
/// this is 4X the worker pool size to handle burst traffic
const DEFAULT_WORK_QUEUE_SIZE: usize = 40000;

/// Get worker pool size from environment or use default
fn get_worker_pool_size() -> usize {
    std::env::var("DYN_TCP_WORKER_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WORKER_POOL_SIZE)
}

/// Get work queue size from environment or use default
fn get_work_queue_size() -> usize {
    std::env::var("DYN_TCP_WORK_QUEUE_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WORK_QUEUE_SIZE)
}

/// RAII guard for `WORK_HANDLER_POOL_ACTIVE_TASKS`. `new()` increments and
/// `Drop` decrements, so a single owner expresses the "task is active" interval.
/// Constructed in the dispatcher *before* `tokio::spawn` and moved into the
/// future, the gauge is incremented before any worker thread can poll the task,
/// and the decrement runs on every exit path — normal return, panic, or
/// cancellation.
struct ActiveTaskGuard;

impl ActiveTaskGuard {
    fn new() -> Self {
        WORK_HANDLER_POOL_ACTIVE_TASKS.inc();
        Self
    }
}

impl Drop for ActiveTaskGuard {
    fn drop(&mut self) {
        WORK_HANDLER_POOL_ACTIVE_TASKS.dec();
    }
}

/// Work item for the worker pool
struct WorkItem {
    service_handler: Arc<dyn PushWorkHandler>,
    payload: Bytes,
    headers: std::collections::HashMap<String, String>,
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
    instance_id: u64,
    namespace: String,
    component_name: String,
    endpoint_name: String,
}

/// Shared TCP server that handles multiple endpoints on a single port
pub struct SharedTcpServer {
    handlers: Arc<DashMap<String, Arc<EndpointHandler>>>,
    /// The address to bind to (may have port 0 for OS-assigned port)
    bind_addr: SocketAddr,
    /// The actual bound address (populated after bind_and_start, contains actual port)
    actual_addr: RwLock<Option<SocketAddr>>,
    cancellation_token: CancellationToken,
    /// Channel for sending work to the worker pool
    work_tx: tokio::sync::mpsc::Sender<WorkItem>,
    /// Worker-pool semaphore bounding concurrent TCP worker tasks. Shared with
    /// `read_loop` so it can front-acquire a permit and dispatch directly.
    /// Unrelated to the backend admission gate's concurrency limit.
    engine_sem: Arc<Semaphore>,
    /// Overflow-queue capacity; `read_loop` compares against it to tell whether
    /// the queue is empty for the FIFO direct-dispatch rule.
    queue_capacity: usize,
    /// Optional TLS acceptor for encrypting request plane connections.
    tls_acceptor: Option<TlsAcceptor>,
}

struct EndpointHandler {
    service_handler: Arc<dyn PushWorkHandler>,
    instance_id: u64,
    namespace: String,
    component_name: String,
    endpoint_name: String,
    system_health: Arc<Mutex<SystemHealth>>,
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
}

impl SharedTcpServer {
    pub fn new(
        bind_addr: SocketAddr,
        cancellation_token: CancellationToken,
    ) -> anyhow::Result<Arc<Self>> {
        // TCP request-plane sizing only. Backend admission (the engine
        // concurrency limit and its overflow queue) is owned by
        // `crate::admission_gate`, which every transport shares.
        let worker_pool_size = get_worker_pool_size();
        let work_queue_size = get_work_queue_size();

        tracing::info!(
            "Initializing TCP server with dispatcher (concurrency={}, queue={})",
            worker_pool_size,
            work_queue_size,
        );

        // Publish static capacities so dashboards can compute saturation ratios.
        // These gauges are process-global and harmless to re-set if multiple TCP
        // servers are spun up in the same process (tests).
        WORK_HANDLER_POOL_CAPACITY.set(crate::metrics::prometheus_names::clamp_u64_to_i64(
            worker_pool_size as u64,
        ));
        WORK_HANDLER_QUEUE_CAPACITY.set(crate::metrics::prometheus_names::clamp_u64_to_i64(
            work_queue_size as u64,
        ));

        // Create bounded channel for work items
        let (work_tx, work_rx) = tokio::sync::mpsc::channel(work_queue_size);

        // Shared with read_loop, which front-acquires permits for direct dispatch.
        let engine_sem = Arc::new(Semaphore::new(worker_pool_size));

        // Dispatcher drains the overflow queue.
        Self::start_worker_pool(engine_sem.clone(), work_rx, cancellation_token.clone());

        // Build TLS acceptor from the same env vars as the call-home transport.
        let tls_acceptor = {
            use crate::config::environment_names::tcp_response_stream::tls as env;
            let cert = std::env::var(env::DYN_TCP_TLS_CERT_PATH).ok();
            let key = std::env::var(env::DYN_TCP_TLS_KEY_PATH).ok();
            let client_ca = std::env::var(env::DYN_TCP_TLS_CLIENT_CA_CERT_PATH).ok();
            Self::request_plane_tls_acceptor(
                cert.as_deref().map(std::path::Path::new),
                key.as_deref().map(std::path::Path::new),
                client_ca.as_deref().map(std::path::Path::new),
            )?
        };

        Ok(Arc::new(Self {
            handlers: Arc::new(DashMap::new()),
            bind_addr,
            actual_addr: RwLock::new(None),
            cancellation_token,
            work_tx,
            engine_sem,
            queue_capacity: work_queue_size,
            tls_acceptor,
        }))
    }

    /// Build the request-plane TLS acceptor from cert/key/client-CA paths.
    /// Validation and diagnostics are shared with the response-stream server via
    /// [`crate::tls_utils::server_tls_acceptor_config`]; when a client CA is
    /// provided, mTLS is enforced (clients must present a trusted certificate).
    fn request_plane_tls_acceptor(
        cert: Option<&std::path::Path>,
        key: Option<&std::path::Path>,
        client_ca: Option<&std::path::Path>,
    ) -> Result<Option<TlsAcceptor>> {
        Ok(
            crate::tls_utils::server_tls_acceptor_config(
                "TCP request plane",
                cert,
                key,
                client_ca,
            )?
            .map(|config| TlsAcceptor::from(Arc::new(config))),
        )
    }

    /// Start the worker pool dispatcher that processes requests with bounded concurrency
    ///
    /// Uses a single receiver with a semaphore to bound concurrent execution,
    /// avoiding mutex contention that would serialize all workers.
    fn start_worker_pool(
        semaphore: Arc<Semaphore>,
        mut work_rx: tokio::sync::mpsc::Receiver<WorkItem>,
        cancellation_token: CancellationToken,
    ) {
        let pool_size = semaphore.available_permits();

        tokio::spawn(async move {
            tracing::trace!(
                "TCP worker dispatcher started with concurrency limit {}",
                pool_size
            );

            loop {
                tokio::select! {
                    biased;

                    _ = cancellation_token.cancelled() => {
                        tracing::trace!("TCP worker dispatcher shutting down: cancellation requested");
                        break;
                    }

                    msg = work_rx.recv() => {
                        let Some(work_item) = msg else {
                            tracing::trace!("TCP worker dispatcher shutting down: channel closed");
                            break;
                        };
                        // Item is out of the mpsc channel — drop queue_depth now so the
                        // gauge strictly reflects channel occupancy. Permit-acquire wait is
                        // tracked separately by WORK_HANDLER_PERMIT_WAIT_SECONDS.
                        WORK_HANDLER_QUEUE_DEPTH.dec();

                        // Acquire permit before spawning (bounds concurrency). Time the wait so
                        // pool starvation (permit exhaustion) shows up as rising p99 in
                        // `dynamo_work_handler_permit_wait_seconds`. Only the queued (dispatcher)
                        // path observes permit-wait; the direct path in read_loop already holds
                        // a permit before enqueuing is ever considered.
                        let permit_wait_start = Instant::now();
                        let permit = match semaphore.clone().acquire_owned().await {
                            Ok(p) => p,
                            Err(_) => {
                                tracing::trace!("TCP worker dispatcher: semaphore closed");
                                break;
                            }
                        };
                        WORK_HANDLER_PERMIT_WAIT_SECONDS
                            .observe(permit_wait_start.elapsed().as_secs_f64());

                        Self::spawn_handle(work_item, permit);
                    }
                }
            }

            tracing::trace!("TCP worker dispatcher exited");
        });

        tracing::info!(
            "Started TCP worker dispatcher with concurrency limit {}",
            pool_size
        );
    }

    /// Spawn the worker task for an item that already holds a permit (the
    /// direct path in `read_loop` and the queued path in the dispatcher). The
    /// `ActiveTaskGuard` is built synchronously so the gauge increments before
    /// the task is polled; the permit drops on completion.
    fn spawn_handle(work_item: WorkItem, permit: OwnedSemaphorePermit) {
        let active_guard = ActiveTaskGuard::new();
        tokio::spawn(async move {
            let _active_guard = active_guard;
            Self::handle_work_item(work_item).await;
            drop(permit);
        });
    }

    /// Handle a single work item
    async fn handle_work_item(work_item: WorkItem) {
        tracing::trace!(
            instance_id = work_item.instance_id,
            "TCP worker processing request"
        );

        // Compute network transit time from the transport header stamped right
        // before the TCP write on the frontend side.
        if let Some(t1_str) = work_item.headers.get("x-frontend-send-ts-ns")
            && let Ok(t1_ns) = t1_str.parse::<u64>()
        {
            let t2_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let transit_ns = t2_ns.saturating_sub(t1_ns);
            crate::metrics::work_handler_perf::WORK_HANDLER_NETWORK_TRANSIT_SECONDS
                .observe(transit_ns as f64 / 1_000_000_000.0);
        }

        // Create span with trace context from headers
        let span = crate::logging::make_handle_payload_span_from_tcp_headers(
            &work_item.headers,
            &work_item.component_name,
            &work_item.endpoint_name,
            &work_item.namespace,
            work_item.instance_id,
        );

        let request_id = work_item
            .headers
            .get("request-id")
            .or_else(|| work_item.headers.get("x-dynamo-request-id"))
            .cloned();

        let result = work_item
            .service_handler
            .handle_payload(work_item.payload, request_id)
            .instrument(span)
            .await;

        if let Err(e) = result {
            tracing::warn!(
                instance_id = work_item.instance_id,
                error = %e,
                "TCP worker failed to handle request"
            );
        }

        work_item.inflight.fetch_sub(1, Ordering::SeqCst);
        work_item.notify.notify_one();
    }

    /// Bind the server and start accepting connections.
    ///
    /// This method binds to the configured address first, then starts the accept loop.
    /// If the configured port is 0, the OS will assign a free port.
    /// The actual bound address is stored and can be retrieved via `actual_address()`.
    ///
    /// Returns the actual bound address (useful when port 0 was specified).
    pub async fn bind_and_start(self: Arc<Self>) -> Result<SocketAddr> {
        tracing::info!("Binding TCP server to {}", self.bind_addr);

        let listener = TcpListener::bind(&self.bind_addr).await?;
        let actual_addr = listener.local_addr()?;

        tracing::info!(
            requested = %self.bind_addr,
            actual = %actual_addr,
            "TCP server bound successfully"
        );

        // Store the actual bound address
        *self.actual_addr.write() = Some(actual_addr);

        // Start accepting connections in a background task
        let server = self.clone();
        tokio::spawn(async move {
            server.accept_loop(listener).await;
        });

        Ok(actual_addr)
    }

    /// Get the actual bound address (after bind_and_start has been called).
    ///
    /// Returns None if the server hasn't been started yet.
    pub fn actual_address(&self) -> Option<SocketAddr> {
        *self.actual_addr.read()
    }

    /// Internal accept loop - runs after binding
    async fn accept_loop(self: Arc<Self>, listener: TcpListener) {
        let cancellation_token = self.cancellation_token.clone();

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, peer_addr)) => {
                            tracing::trace!("Accepted TCP connection from {peer_addr}");

                            let handlers = self.handlers.clone();
                            let work_tx = self.work_tx.clone();
                            let engine_sem = self.engine_sem.clone();
                            let queue_capacity = self.queue_capacity;
                            let tls_acceptor = self.tls_acceptor.clone();
                            tokio::spawn(async move {
                                let (reader, writer): (BoxRead, BoxWrite) =
                                    if let Some(ref tls) = tls_acceptor {
                                        match tokio::time::timeout(
                                            crate::tls_utils::handshake_timeout(),
                                            tls.accept(stream),
                                        )
                                        .await
                                        {
                                            Ok(Ok(tls_stream)) => {
                                                let (r, w) = tokio::io::split(tls_stream);
                                                (Box::new(r), Box::new(w))
                                            }
                                            Ok(Err(e)) => {
                                                tracing::warn!(
                                                    peer_addr = %peer_addr,
                                                    error = %e,
                                                    "Request-plane TLS handshake failed"
                                                );
                                                return;
                                            }
                                            Err(_) => {
                                                tracing::warn!(
                                                    peer_addr = %peer_addr,
                                                    "Request-plane TLS handshake timed out"
                                                );
                                                return;
                                            }
                                        }
                                    } else {
                                        let (r, w) = tokio::io::split(stream);
                                        (Box::new(r), Box::new(w))
                                    };
                                if let Err(e) = Self::handle_connection(reader, writer, handlers, work_tx, engine_sem, queue_capacity).await {
                                    tracing::error!("TCP connection error: {e}");
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Failed to accept TCP connection: {e}");
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    tracing::info!("SharedTcpServer received cancellation signal, shutting down");
                    return;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn register_endpoint(
        &self,
        endpoint_path: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        endpoint_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let fqn_endpoint = format!("{namespace}.{component_name}.{endpoint_name}");

        let handler = Arc::new(EndpointHandler {
            service_handler,
            instance_id,
            namespace,
            component_name,
            endpoint_name: endpoint_name.clone(),
            system_health: system_health.clone(),
            inflight: Arc::new(AtomicU64::new(0)),
            notify: Arc::new(Notify::new()),
        });

        // Insert handler FIRST to ensure it's ready to receive requests
        self.handlers.insert(endpoint_path, handler);

        system_health.lock().set_endpoint_registered(&endpoint_name);

        tracing::info!(
            "Registered endpoint '{fqn_endpoint}' with shared TCP server on {}",
            self.actual_address().unwrap_or(self.bind_addr)
        );

        Ok(())
    }

    pub async fn remove_handler(&self, endpoint_path: &str, endpoint_name: &str) {
        if let Some((_, handler)) = self.handlers.remove(endpoint_path) {
            handler
                .system_health
                .lock()
                .set_endpoint_health_status(endpoint_name, crate::HealthStatus::NotReady);
            tracing::info!(
                endpoint_name = %endpoint_name,
                endpoint_path = %endpoint_path,
                "Unregistered TCP endpoint handler"
            );

            super::drain_inflight(
                handler.inflight.clone(),
                handler.notify.clone(),
                endpoint_name,
                crate::runtime::graceful_shutdown_timeout(),
            )
            .await;
        }
    }

    /// Start the server (legacy method - prefer bind_and_start for new code).
    ///
    /// This method is kept for backwards compatibility. It binds and starts
    /// the server but doesn't return the actual bound address.
    pub async fn start(self: Arc<Self>) -> Result<()> {
        let cancel_token = self.cancellation_token.clone();
        self.bind_and_start().await?;
        // Wait for cancellation (the accept loop runs in background)
        cancel_token.cancelled().await;
        Ok(())
    }

    async fn handle_connection(
        read_half: BoxRead,
        write_half: BoxWrite,
        handlers: Arc<DashMap<String, Arc<EndpointHandler>>>,
        work_tx: tokio::sync::mpsc::Sender<WorkItem>,
        engine_sem: Arc<Semaphore>,
        queue_capacity: usize,
    ) -> Result<()> {
        use crate::pipeline::network::codec::{TcpRequestMessage, TcpResponseMessage};

        // Channel for sending responses to the write task (zero-copy Bytes)
        let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();

        // Spawn write task
        let write_task = tokio::spawn(Self::write_loop(write_half, response_rx));

        // Run read task in current context
        let read_result = Self::read_loop(
            read_half,
            handlers,
            response_tx,
            work_tx,
            engine_sem,
            queue_capacity,
        )
        .await;

        // Write task will end when response_tx is dropped
        write_task.await??;

        read_result
    }

    #[allow(clippy::too_many_arguments)]
    async fn read_loop(
        mut read_half: BoxRead,
        handlers: Arc<DashMap<String, Arc<EndpointHandler>>>,
        response_tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
        work_tx: tokio::sync::mpsc::Sender<WorkItem>,
        engine_sem: Arc<Semaphore>,
        queue_capacity: usize,
    ) -> Result<()> {
        use crate::pipeline::network::codec::{TcpResponseMessage, ZeroCopyTcpDecoder};

        // Create zero-copy decoder with optimized buffer size
        let mut decoder = ZeroCopyTcpDecoder::new();

        // Encode and send a response frame; returns false if the write task is gone.
        let send_response = |msg: TcpResponseMessage| -> bool {
            match msg.encode() {
                Ok(encoded) => response_tx.send(encoded).is_ok(),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to encode TCP response");
                    true
                }
            }
        };

        loop {
            // Read one complete message with ZERO copies!
            let request_msg = match decoder.read_message(&mut read_half).await {
                Ok(msg) => msg,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    tracing::trace!("Connection closed by peer");
                    break;
                }
                Err(e) => {
                    tracing::warn!("Failed to read TCP request: {e}");
                    // Send error response
                    let error_response =
                        TcpResponseMessage::new(Bytes::from(format!("Read error: {}", e)));
                    if let Ok(encoded) = error_response.encode() {
                        let _ = response_tx.send(encoded);
                    }
                    return Err(e.into());
                }
            };

            // Get endpoint path (zero-copy string slice)
            let endpoint_path = match request_msg.endpoint_path() {
                Ok(path) => path,
                Err(e) => {
                    tracing::warn!("Invalid UTF-8 in endpoint path: {e}");
                    let error_response =
                        TcpResponseMessage::new(Bytes::from_static(b"Invalid endpoint path"));
                    if let Ok(encoded) = error_response.encode() {
                        let _ = response_tx.send(encoded);
                    }
                    continue;
                }
            };

            // Get headers (parsed from message)
            let headers = request_msg.headers();

            // Get payload (zero-copy Bytes - just Arc clone!)
            let payload = request_msg.payload();

            tracing::trace!(
                endpoint = endpoint_path,
                payload_len = payload.len(),
                total_size = request_msg.total_size(),
                "Received TCP request"
            );

            // Look up handler (lock-free read with DashMap)
            let handler = handlers.get(endpoint_path).map(|h| h.clone());

            let handler = match handler {
                Some(h) => h,
                None => {
                    tracing::warn!("No handler found for endpoint: {endpoint_path}");
                    // The client only treats this prefix as a rejection; any other reply is
                    // read as a success ACK and it waits for a response stream that never opens.
                    let error_response = TcpResponseMessage::new(Bytes::from(format!(
                        "{} unknown endpoint {endpoint_path}",
                        crate::pipeline::network::ACK_UNAVAILABLE_PREFIX
                    )));
                    if let Ok(encoded) = error_response.encode() {
                        let _ = response_tx.send(encoded);
                    }
                    continue;
                }
            };

            handler.inflight.fetch_add(1, Ordering::SeqCst);

            // Build work item
            // NOTE: payload is Bytes (Arc-counted), so cloning is extremely cheap
            let work_item = WorkItem {
                service_handler: handler.service_handler.clone(),
                payload,
                headers,
                inflight: handler.inflight.clone(),
                notify: handler.notify.clone(),
                instance_id: handler.instance_id,
                namespace: handler.namespace.clone(),
                component_name: handler.component_name.clone(),
                endpoint_name: handler.endpoint_name.clone(),
            };

            // Engine permit free and nothing queued ahead → dispatch directly.
            // Take the direct path only when the queue is empty so a new request
            // can't jump ahead of queued ones (FIFO).
            let queue_empty = work_tx.capacity() == queue_capacity;
            let direct_permit = if queue_empty {
                engine_sem.clone().try_acquire_owned().ok()
            } else {
                None
            };

            if let Some(permit) = direct_permit {
                // Bypass the queue — routing through work_tx would make it a throat.
                Self::spawn_handle(work_item, permit);
                if !send_response(TcpResponseMessage::empty()) {
                    break;
                }
                continue;
            }

            // All pool permits busy (or items already queued): try the work queue.
            match work_tx.try_reserve() {
                Ok(slot) => {
                    WORK_HANDLER_QUEUE_DEPTH.inc();
                    slot.send(work_item);
                    // Queued: the dispatcher owns inflight, so don't touch it here.
                    if !send_response(TcpResponseMessage::empty()) {
                        break;
                    }
                }
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    // TCP worker pool and work queue both full → shed; keep the
                    // connection open.
                    WORK_HANDLER_ENQUEUE_REJECTED_TOTAL.inc();
                    tracing::warn!(
                        endpoint = handler.endpoint_name.as_str(),
                        instance_id = handler.instance_id,
                        "TCP worker pool and work queue full, rejecting request"
                    );
                    send_response(TcpResponseMessage::new(Bytes::from_static(
                        b"Server overloaded: worker at capacity",
                    )));
                    handler.inflight.fetch_sub(1, Ordering::SeqCst);
                    handler.notify.notify_one();
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    WORK_HANDLER_ENQUEUE_REJECTED_TOTAL.inc();
                    send_response(TcpResponseMessage::new(Bytes::from(format!(
                        "{} worker pool channel closed",
                        crate::pipeline::network::ACK_UNAVAILABLE_PREFIX
                    ))));
                    handler.inflight.fetch_sub(1, Ordering::SeqCst);
                    handler.notify.notify_one();
                    tracing::error!("Worker pool channel closed, shutting down read loop");
                    break;
                }
            }
        }

        Ok(())
    }

    async fn write_loop(
        mut write_half: BoxWrite,
        mut response_rx: tokio::sync::mpsc::UnboundedReceiver<Bytes>,
    ) -> Result<()> {
        while let Some(response) = response_rx.recv().await {
            write_half.write_all(&response).await?;
            write_half.flush().await?;
        }
        Ok(())
    }
}

// Implement RequestPlaneServer trait for SharedTcpServer
#[async_trait::async_trait]
impl super::unified_server::RequestPlaneServer for SharedTcpServer {
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let endpoint_id = EndpointId {
            namespace: namespace.clone(),
            component: component_name.clone(),
            name: endpoint_name.clone(),
        };
        self.register_endpoint(
            instance_path(&endpoint_id, instance_id),
            service_handler,
            instance_id,
            namespace,
            component_name,
            endpoint_name,
            system_health,
        )
        .await
    }

    async fn unregister_endpoint(&self, endpoint_name: &str, instance_id: u64) -> Result<()> {
        let path = {
            let mut matches = self.handlers.iter().filter(|entry| {
                entry.value().endpoint_name == endpoint_name
                    && entry.value().instance_id == instance_id
            });
            let path = matches.next().map(|entry| entry.key().clone());
            anyhow::ensure!(
                matches.next().is_none(),
                "Ambiguous endpoint {endpoint_name}/{instance_id:x}; use unregister_endpoint_instance"
            );
            path
        };
        if let Some(path) = path {
            self.remove_handler(&path, endpoint_name).await;
        }
        Ok(())
    }

    async fn unregister_endpoint_instance(
        &self,
        endpoint_id: &EndpointId,
        instance_id: u64,
    ) -> Result<()> {
        self.remove_handler(&instance_path(endpoint_id, instance_id), &endpoint_id.name)
            .await;
        Ok(())
    }

    fn address(&self) -> String {
        // Return actual bound address if available (after bind_and_start),
        // otherwise fall back to configured bind address
        let addr = self.actual_address().unwrap_or(self.bind_addr);
        format!("tcp://{}:{}", addr.ip(), addr.port())
    }

    fn transport_name(&self) -> &'static str {
        "tcp"
    }

    fn is_healthy(&self) -> bool {
        // Server is healthy if it has been created
        // TODO: Add more sophisticated health checks (e.g., check if listener is active)
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::error::PipelineError;
    use crate::pipeline::network::egress::tcp_client::TcpRequestClient;
    use crate::pipeline::network::egress::unified_client::{Headers, RequestPlaneClient};
    use crate::pipeline::network::ingress::unified_server::RequestPlaneServer;
    use crate::tls_utils::test_certs::self_signed_pair;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use tokio::time::Instant;

    /// Mock handler that simulates slow request processing for testing
    struct SlowMockHandler {
        /// Tracks if a request is currently being processed
        request_in_flight: Arc<AtomicBool>,
        /// Notifies when request processing starts
        request_started: Arc<Notify>,
        /// Notifies when request processing completes
        request_completed: Arc<Notify>,
        /// Duration to simulate request processing
        processing_duration: Duration,
    }

    impl SlowMockHandler {
        fn new(processing_duration: Duration) -> Self {
            Self {
                request_in_flight: Arc::new(AtomicBool::new(false)),
                request_started: Arc::new(Notify::new()),
                request_completed: Arc::new(Notify::new()),
                processing_duration,
            }
        }
    }

    #[async_trait]
    impl PushWorkHandler for SlowMockHandler {
        async fn handle_payload(
            &self,
            _payload: Bytes,
            _request_id: Option<String>,
        ) -> Result<(), PipelineError> {
            self.request_in_flight.store(true, Ordering::SeqCst);
            self.request_started.notify_one();

            tracing::debug!(
                "SlowMockHandler: Request started, sleeping for {:?}",
                self.processing_duration
            );

            // Simulate slow request processing
            tokio::time::sleep(self.processing_duration).await;

            tracing::debug!("SlowMockHandler: Request completed");

            self.request_in_flight.store(false, Ordering::SeqCst);
            self.request_completed.notify_one();
            Ok(())
        }

        fn add_metrics(
            &self,
            _endpoint: &crate::component::Endpoint,
            _metrics_labels: Option<&[(&str, &str)]>,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn ready_system_health() -> Arc<Mutex<SystemHealth>> {
        Arc::new(Mutex::new(SystemHealth::new(
            crate::HealthStatus::Ready,
            vec![],
            false, // health_check_enabled
            "/health".to_string(),
            "/live".to_string(),
        )))
    }

    #[tokio::test]
    async fn test_graceful_shutdown_waits_for_inflight_tcp_requests() {
        // Initialize tracing for test debugging
        crate::logging::init();

        let cancellation_token = CancellationToken::new();
        let bind_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

        // Create SharedTcpServer
        let server = SharedTcpServer::new(bind_addr, cancellation_token.clone()).unwrap();

        // Create a handler that takes 1s to process requests
        let handler = Arc::new(SlowMockHandler::new(Duration::from_secs(1)));
        let request_started = handler.request_started.clone();
        let request_completed = handler.request_completed.clone();
        let request_in_flight = handler.request_in_flight.clone();

        // Register endpoint
        let endpoint_path = "test_endpoint".to_string();
        let system_health = ready_system_health();

        server
            .register_endpoint(
                endpoint_path.clone(),
                handler.clone() as Arc<dyn PushWorkHandler>,
                1,
                "test_namespace".to_string(),
                "test_component".to_string(),
                "test_endpoint".to_string(),
                system_health,
            )
            .await
            .expect("Failed to register endpoint");

        tracing::debug!("Endpoint registered");

        // Get the endpoint handler to simulate request processing
        let endpoint_handler = server
            .handlers
            .get(&endpoint_path)
            .expect("Handler should be registered")
            .clone();

        // Spawn a task that simulates an inflight request
        let request_task = tokio::spawn({
            let handler = handler.clone();
            async move {
                let payload = Bytes::from("test payload");
                handler.handle_payload(payload, None).await
            }
        });

        // Increment inflight counter manually to simulate the request being tracked
        endpoint_handler.inflight.fetch_add(1, Ordering::SeqCst);

        // Wait for request to start processing
        tokio::select! {
            _ = request_started.notified() => {
                tracing::debug!("Request processing started");
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                panic!("Timeout waiting for request to start");
            }
        }

        // Verify request is in flight
        assert!(
            request_in_flight.load(Ordering::SeqCst),
            "Request should be in flight"
        );

        // Now unregister the endpoint while request is inflight
        let unregister_start = Instant::now();
        tracing::debug!("Starting unregister_endpoint with inflight request");

        // Spawn unregister in a separate task so we can monitor its behavior
        let unregister_task = tokio::spawn({
            let server = server.clone();
            let endpoint_path = endpoint_path.clone();
            async move {
                server.remove_handler(&endpoint_path, "test_endpoint").await;
                Instant::now()
            }
        });

        // Give unregister a moment to remove handler and start waiting
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Verify that unregister_endpoint hasn't returned yet (it should be waiting)
        assert!(
            !unregister_task.is_finished(),
            "unregister_endpoint should still be waiting for inflight request"
        );

        tracing::debug!("Verified unregister is waiting, now waiting for request to complete");

        // Wait for the request to complete
        tokio::select! {
            _ = request_completed.notified() => {
                tracing::debug!("Request completed");
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                panic!("Timeout waiting for request to complete");
            }
        }

        // Decrement inflight counter and notify (simulating what the real code does)
        endpoint_handler.inflight.fetch_sub(1, Ordering::SeqCst);
        endpoint_handler.notify.notify_one();

        // Now wait for unregister to complete
        let unregister_end = tokio::time::timeout(Duration::from_secs(2), unregister_task)
            .await
            .expect("unregister_endpoint should complete after inflight request finishes")
            .expect("unregister task should not panic");

        let unregister_duration = unregister_end - unregister_start;

        tracing::debug!("unregister_endpoint completed in {:?}", unregister_duration);

        // Verify unregister_endpoint waited for the inflight request
        assert!(
            unregister_duration >= Duration::from_secs(1),
            "unregister_endpoint should have waited ~1s for inflight request, but only took {:?}",
            unregister_duration
        );

        // Verify request completed successfully
        assert!(
            !request_in_flight.load(Ordering::SeqCst),
            "Request should have completed"
        );

        // Wait for request task to finish
        request_task
            .await
            .expect("Request task should complete")
            .expect("Request should succeed");

        tracing::info!("Test passed: unregister_endpoint properly waited for inflight TCP request");
    }

    async fn send_ack(client: &TcpRequestClient, address: &str) -> Bytes {
        tokio::time::timeout(
            Duration::from_secs(5),
            client.send_request(
                address.to_string(),
                Bytes::from_static(b"payload"),
                Headers::new(),
            ),
        )
        .await
        .unwrap_or_else(|_| panic!("no ACK within 5s for {address}"))
        .expect("request-plane send should succeed")
    }

    #[tokio::test]
    async fn unregister_endpoint_removes_only_the_matching_endpoint_instance() {
        let endpoint = EndpointId {
            namespace: "test_namespace".into(),
            component: "test_component".into(),
            name: "generate".into(),
        };
        let survivors = [
            (endpoint.clone(), 0xb),
            (
                EndpointId {
                    namespace: "other_namespace".into(),
                    ..endpoint.clone()
                },
                0xa,
            ),
            (
                EndpointId {
                    component: "other_component".into(),
                    ..endpoint.clone()
                },
                0xa,
            ),
            (
                EndpointId {
                    name: "other_endpoint".into(),
                    ..endpoint.clone()
                },
                0xa,
            ),
        ];

        let id = |namespace: &str, component: &str, name: &str| EndpointId {
            namespace: namespace.into(),
            component: component.into(),
            name: name.into(),
        };
        let cases = survivors
            .into_iter()
            .map(|(survivor, instance_id)| (endpoint.clone(), survivor, instance_id))
            .chain([
                (id("a/b", "c", "d"), id("a", "b/c", "d"), 0xa),
                (id("a", "b/c", "d"), id("a", "b", "c/d"), 0xa),
                (id("a/b", "c", "d"), id("a%2Fb", "c", "d"), 0xa),
            ]);

        for (endpoint, survivor_endpoint, survivor_id) in cases {
            let cancel = CancellationToken::new();
            let server =
                SharedTcpServer::new("127.0.0.1:0".parse().unwrap(), cancel.clone()).unwrap();
            let addr = server.clone().bind_and_start().await.unwrap();
            let plane: &dyn RequestPlaneServer = server.as_ref();
            let removed = Arc::new(SlowMockHandler::new(Duration::ZERO));
            let survivor = Arc::new(SlowMockHandler::new(Duration::ZERO));
            for (id, instance_id, handler) in [
                (&endpoint, 0xa, removed.clone()),
                (&survivor_endpoint, survivor_id, survivor.clone()),
            ] {
                plane
                    .register_endpoint(
                        id.name.clone(),
                        handler,
                        instance_id,
                        id.namespace.clone(),
                        id.component.clone(),
                        ready_system_health(),
                    )
                    .await
                    .unwrap();
            }

            if survivor_id == 0xa && survivor_endpoint.name == endpoint.name {
                assert!(
                    plane
                        .unregister_endpoint(&endpoint.name, 0xa)
                        .await
                        .is_err()
                );
            }

            // The second registration must not redirect requests for the first.
            let client = TcpRequestClient::new().unwrap();
            let removed_address = format!("{addr}/{}", instance_path(&endpoint, 0xa));
            assert!(send_ack(&client, &removed_address).await.is_empty());
            let delivered_to_removed = tokio::time::timeout(Duration::from_secs(5), async {
                tokio::select! {
                    _ = removed.request_started.notified() => true,
                    _ = survivor.request_started.notified() => false,
                }
            })
            .await
            .expect("a registered handler should receive the request");
            assert!(
                delivered_to_removed,
                "request redirected to {survivor_endpoint:?}/{survivor_id:x}"
            );

            // Removing one registration must preserve every distinct endpoint instance.
            plane
                .unregister_endpoint_instance(&endpoint, 0xa)
                .await
                .unwrap();
            let address = format!("{addr}/{}", instance_path(&survivor_endpoint, survivor_id));
            let ack = send_ack(&client, &address).await;
            assert!(
                ack.is_empty(),
                "survivor {survivor_endpoint:?}/{survivor_id:x} rejected: {ack:?}"
            );
            tokio::time::timeout(Duration::from_secs(5), survivor.request_started.notified())
                .await
                .expect("surviving handler should receive the request");

            let ack = send_ack(&client, &removed_address).await;
            assert!(
                ack.starts_with(crate::pipeline::network::ACK_UNAVAILABLE_PREFIX.as_bytes()),
                "removed endpoint must reject new requests: {ack:?}"
            );
            // With one match left, the original API must still remove that registration.
            plane
                .unregister_endpoint(&survivor_endpoint.name, survivor_id)
                .await
                .unwrap();
            assert!(
                send_ack(&client, &address)
                    .await
                    .starts_with(crate::pipeline::network::ACK_UNAVAILABLE_PREFIX.as_bytes())
            );
            plane
                .unregister_endpoint(&survivor_endpoint.name, survivor_id)
                .await
                .unwrap();
            cancel.cancel();
        }
    }

    ///////////////////// TESTS FOR CONCURRENCY BOUNDING /////////////////////

    /// Mock handler that tracks concurrent execution count
    struct ConcurrencyTrackingHandler {
        /// Current number of concurrent requests being processed
        concurrent_count: Arc<AtomicU64>,
        /// Maximum concurrent count observed
        max_concurrent: Arc<AtomicU64>,
        /// Duration to simulate request processing
        processing_duration: Duration,
        /// Notifies when a request completes
        completed: Arc<Notify>,
    }

    impl ConcurrencyTrackingHandler {
        fn new(processing_duration: Duration) -> Self {
            Self {
                concurrent_count: Arc::new(AtomicU64::new(0)),
                max_concurrent: Arc::new(AtomicU64::new(0)),
                processing_duration,
                completed: Arc::new(Notify::new()),
            }
        }
    }

    #[async_trait]
    impl PushWorkHandler for ConcurrencyTrackingHandler {
        async fn handle_payload(
            &self,
            _payload: Bytes,
            _request_id: Option<String>,
        ) -> Result<(), PipelineError> {
            // Increment concurrent count
            let current = self.concurrent_count.fetch_add(1, Ordering::SeqCst) + 1;

            // Update max if this is higher
            self.max_concurrent.fetch_max(current, Ordering::SeqCst);

            // Simulate work
            tokio::time::sleep(self.processing_duration).await;

            // Decrement concurrent count
            self.concurrent_count.fetch_sub(1, Ordering::SeqCst);
            self.completed.notify_one();

            Ok(())
        }

        fn add_metrics(
            &self,
            _endpoint: &crate::component::Endpoint,
            _metrics_labels: Option<&[(&str, &str)]>,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_worker_pool_bounds_concurrency() {
        crate::logging::init();

        // Use a small pool size for testing
        let pool_size = 3;
        let total_requests = 10;

        // Create bounded channel and dispatcher directly
        let (work_tx, work_rx) = tokio::sync::mpsc::channel::<WorkItem>(total_requests);
        let cancellation_token = CancellationToken::new();

        // Start worker pool with small concurrency limit
        SharedTcpServer::start_worker_pool(
            Arc::new(Semaphore::new(pool_size)),
            work_rx,
            cancellation_token.clone(),
        );

        // Create tracking handler
        let handler = Arc::new(ConcurrencyTrackingHandler::new(Duration::from_millis(50)));

        // Create dummy inflight/notify for work items
        let inflight = Arc::new(AtomicU64::new(0));
        let notify = Arc::new(Notify::new());

        // Send more work items than pool size. Mirror the production read_loop's
        // queue-depth accounting so `handle_work_item`'s decrement has a matching
        // increment and the global gauge stays consistent for other tests.
        for i in 0..total_requests {
            inflight.fetch_add(1, Ordering::SeqCst);
            WORK_HANDLER_QUEUE_DEPTH.inc();
            let work_item = WorkItem {
                service_handler: handler.clone() as Arc<dyn PushWorkHandler>,
                payload: Bytes::from(format!("request {}", i)),
                headers: std::collections::HashMap::new(),
                inflight: inflight.clone(),
                notify: notify.clone(),
                instance_id: 1,
                namespace: "test".to_string(),
                component_name: "test".to_string(),
                endpoint_name: "test".to_string(),
            };
            work_tx.send(work_item).await.expect("send should succeed");
        }

        // Wait for all requests to complete
        let timeout = tokio::time::timeout(Duration::from_secs(5), async {
            while inflight.load(Ordering::SeqCst) > 0 {
                notify.notified().await;
            }
        })
        .await;

        assert!(
            timeout.is_ok(),
            "All requests should complete within timeout"
        );

        // Verify concurrency was bounded
        let max_observed = handler.max_concurrent.load(Ordering::SeqCst);
        assert!(
            max_observed <= pool_size as u64,
            "Max concurrent ({}) should not exceed pool size ({})",
            max_observed,
            pool_size
        );

        // Verify all requests completed
        assert_eq!(
            inflight.load(Ordering::SeqCst),
            0,
            "All requests should have completed"
        );

        tracing::info!(
            "Test passed: max concurrent {} <= pool size {}",
            max_observed,
            pool_size
        );

        // Cleanup
        cancellation_token.cancel();
    }

    #[tokio::test]
    async fn test_worker_pool_metrics_are_observed() {
        crate::logging::init();

        // Monotonic histogram counters: safe to assert even with parallel tests
        // moving the gauges.
        let permit_observations_before = WORK_HANDLER_PERMIT_WAIT_SECONDS.get_sample_count();

        let pool_size = 2;
        let total_requests = 4;
        let (work_tx, work_rx) = tokio::sync::mpsc::channel::<WorkItem>(total_requests);
        let cancellation_token = CancellationToken::new();
        SharedTcpServer::start_worker_pool(
            Arc::new(Semaphore::new(pool_size)),
            work_rx,
            cancellation_token.clone(),
        );

        let handler = Arc::new(ConcurrencyTrackingHandler::new(Duration::from_millis(25)));
        let inflight = Arc::new(AtomicU64::new(0));
        let notify = Arc::new(Notify::new());

        for i in 0..total_requests {
            inflight.fetch_add(1, Ordering::SeqCst);
            // Mirror the production read_loop's inc so handle_work_item's dec has a pair.
            WORK_HANDLER_QUEUE_DEPTH.inc();
            let work_item = WorkItem {
                service_handler: handler.clone() as Arc<dyn PushWorkHandler>,
                payload: Bytes::from(format!("request {}", i)),
                headers: std::collections::HashMap::new(),
                inflight: inflight.clone(),
                notify: notify.clone(),
                instance_id: 1,
                namespace: "test".to_string(),
                component_name: "test".to_string(),
                endpoint_name: "test".to_string(),
            };
            work_tx.send(work_item).await.expect("send should succeed");
        }

        // Wait for all work to drain
        tokio::time::timeout(Duration::from_secs(5), async {
            while inflight.load(Ordering::SeqCst) > 0 {
                notify.notified().await;
            }
        })
        .await
        .expect("all requests should complete");

        // permit_wait histogram is monotonic and records one sample per dispatched
        // work item — reliable across parallel test threads.
        assert!(
            WORK_HANDLER_PERMIT_WAIT_SECONDS.get_sample_count()
                >= permit_observations_before + total_requests as u64,
            "permit_wait histogram should record at least one sample per dispatched work item"
        );

        cancellation_token.cancel();
    }

    #[tokio::test]
    async fn test_capacities_published_on_server_init() {
        crate::logging::init();

        // SharedTcpServer::new publishes static capacities. Any test that instantiates
        // a SharedTcpServer will have populated the gauges; we just assert they're > 0.
        let cancellation_token = CancellationToken::new();
        let bind_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let _server = SharedTcpServer::new(bind_addr, cancellation_token.clone()).unwrap();

        assert!(
            WORK_HANDLER_POOL_CAPACITY.get() > 0,
            "pool_capacity should be set to DEFAULT_WORKER_POOL_SIZE"
        );
        assert!(
            WORK_HANDLER_QUEUE_CAPACITY.get() > 0,
            "queue_capacity should be set to DEFAULT_WORK_QUEUE_SIZE"
        );
        cancellation_token.cancel();
    }

    #[tokio::test]
    async fn new_no_tls_env_is_plaintext() {
        let token = CancellationToken::new();
        temp_env::with_vars_unset(["DYN_TCP_TLS_CERT_PATH", "DYN_TCP_TLS_KEY_PATH"], || {
            let server =
                SharedTcpServer::new("127.0.0.1:0".parse().unwrap(), token.clone()).unwrap();
            assert!(server.tls_acceptor.is_none());
        });
        token.cancel();
    }

    #[tokio::test]
    async fn new_partial_tls_config_errors() {
        let (cert, key) = self_signed_pair();
        let cert_str = cert.path().to_str().unwrap();
        let key_str = key.path().to_str().unwrap();
        let token = CancellationToken::new();
        // only cert set -> returns Err (must not panic)
        temp_env::with_vars(
            [
                ("DYN_TCP_TLS_CERT_PATH", Some(cert_str)),
                ("DYN_TCP_TLS_KEY_PATH", None),
            ],
            || {
                assert!(
                    SharedTcpServer::new("127.0.0.1:0".parse().unwrap(), token.clone()).is_err()
                );
            },
        );
        // only key set -> returns Err
        temp_env::with_vars(
            [
                ("DYN_TCP_TLS_CERT_PATH", None),
                ("DYN_TCP_TLS_KEY_PATH", Some(key_str)),
            ],
            || {
                assert!(
                    SharedTcpServer::new("127.0.0.1:0".parse().unwrap(), token.clone()).is_err()
                );
            },
        );
        token.cancel();
    }

    #[tokio::test]
    async fn new_both_paths_enables_tls() {
        let (cert, key) = self_signed_pair();
        let token = CancellationToken::new();
        temp_env::with_vars(
            [
                ("DYN_TCP_TLS_CERT_PATH", Some(cert.path().to_str().unwrap())),
                ("DYN_TCP_TLS_KEY_PATH", Some(key.path().to_str().unwrap())),
            ],
            || {
                let server =
                    SharedTcpServer::new("127.0.0.1:0".parse().unwrap(), token.clone()).unwrap();
                assert!(server.tls_acceptor.is_some());
            },
        );
        token.cancel();
    }

    #[test]
    fn request_plane_tls_acceptor_enables_mtls() {
        let (cert, key) = self_signed_pair();
        // cert + key + client CA -> mTLS acceptor built.
        assert!(
            SharedTcpServer::request_plane_tls_acceptor(
                Some(cert.path()),
                Some(key.path()),
                Some(cert.path()),
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn request_plane_tls_rejects_client_ca_without_server_identity() {
        let (client_ca, _) = self_signed_pair();
        let error = SharedTcpServer::request_plane_tls_acceptor(None, None, Some(client_ca.path()))
            .err()
            .expect("a client CA without a server certificate/key must fail");
        assert!(
            error
                .to_string()
                .contains("DYN_TCP_TLS_CLIENT_CA_CERT_PATH requires")
        );
    }

    #[test]
    fn request_plane_tls_reads_client_ca_path() {
        let (cert, key) = self_signed_pair();
        let error = SharedTcpServer::request_plane_tls_acceptor(
            Some(cert.path()),
            Some(key.path()),
            Some(std::path::Path::new(
                "/nonexistent/request-plane-client-ca.pem",
            )),
        )
        .err()
        .expect("an invalid client CA path must fail mTLS configuration");
        assert!(format!("{error:#}").contains("reading client CA cert"));
    }
}

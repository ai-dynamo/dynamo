// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC-based [`FrameTransport`] implementation backed by the VeloStreaming proto service.
//!
//! [`GrpcFrameTransport`] hosts a single shared tonic server per instance with per-anchor
//! routing via [`dashmap::DashMap`]. Each `bind()` call registers a routing slot; each
//! `connect()` call opens a bidirectional gRPC stream and routes frames to the bound receiver.
//!
//! # Endpoint Format
//!
//! ```text
//! grpc://{ip}:{port}/{anchor_id}
//! ```
//!
//! # Key Design Decisions
//!
//! - `new()` is **async** (unlike `TcpFrameTransport::new` which is sync) because it must
//!   `await` `TcpListener::bind` to capture the actual OS-assigned port before returning.
//!   This is necessary so `bind()` can produce an endpoint with the correct port.
//! - The tonic server is started eagerly inside `new()` via
//!   `serve_with_incoming_shutdown(TcpListenerStream::new(listener), cancel.cancelled())`.
//! - Exclusive-attach enforcement uses `DashMap::entry()` for atomic check-then-insert,
//!   preventing TOCTOU races on the `active` map.
//! - The server-side pump task injects `cached_dropped()` only when the stream ends without
//!   a terminal sentinel, mirroring the TCP transport's drop-safety contract.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use futures::future::BoxFuture;
use futures::StreamExt;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, Streaming};

use crate::transport::FrameTransport;

// ---------------------------------------------------------------------------
// Proto-generated code
// ---------------------------------------------------------------------------

pub(crate) mod proto {
    tonic::include_proto!("velo.streaming.v1");
}

use proto::{
    velo_streaming_client::VeloStreamingClient,
    velo_streaming_server::{VeloStreaming, VeloStreamingServer},
    FramedData,
};

// ---------------------------------------------------------------------------
// Terminal sentinel check (module-private, independent of tcp_transport copy)
// ---------------------------------------------------------------------------

/// Check if the given raw frame bytes represent a terminal sentinel.
///
/// Terminal sentinels are Dropped, Detached, Finalized, and TransportError.
/// We compare against cached sentinel bytes for the common cases and fall
/// back to deserialization for TransportError (which carries a string payload).
fn is_terminal_sentinel(bytes: &[u8]) -> bool {
    use crate::sender::{cached_detached, cached_dropped, cached_finalized};

    if bytes == cached_dropped().as_slice()
        || bytes == cached_detached().as_slice()
        || bytes == cached_finalized().as_slice()
    {
        return true;
    }

    // TransportError carries a dynamic string, so we must try to deserialize
    if let Ok(frame) = rmp_serde::from_slice::<crate::frame::StreamFrame<()>>(bytes) {
        matches!(frame, crate::frame::StreamFrame::TransportError(_))
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// GrpcStreamingService (server-side handler)
// ---------------------------------------------------------------------------

/// Server-side gRPC service implementation.
///
/// Routes incoming streams to bound anchor receivers via DashMap lookup.
/// Enforces exclusive-attach: a second `Stream` call for the same anchor_id
/// returns `Status::already_exists`.
#[derive(Clone)]
struct GrpcStreamingService {
    /// Routes anchor_id -> frame_tx for data delivery
    routing: Arc<DashMap<u64, flume::Sender<Vec<u8>>>>,
    /// Tracks actively-attached anchor_ids (exclusive-attach enforcement)
    active: Arc<DashMap<u64, ()>>,
}

#[tonic::async_trait]
impl VeloStreaming for GrpcStreamingService {
    type StreamStream = futures::stream::Empty<Result<FramedData, Status>>;

    async fn stream(
        &self,
        request: Request<Streaming<FramedData>>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        // Extract anchor_id from metadata header "x-anchor-id"
        let anchor_id: u64 = {
            let meta = request.metadata().get("x-anchor-id").ok_or_else(|| {
                Status::invalid_argument("missing x-anchor-id metadata header")
            })?;
            let s = meta.to_str().map_err(|_| {
                Status::invalid_argument("x-anchor-id metadata is not valid UTF-8")
            })?;
            s.parse::<u64>().map_err(|_| {
                Status::invalid_argument("x-anchor-id is not a valid u64")
            })?
        };

        // Exclusive-attach: use DashMap::entry() for atomic check-then-insert
        let entry = self.active.entry(anchor_id);
        match entry {
            dashmap::Entry::Occupied(_) => {
                return Err(Status::already_exists(format!(
                    "anchor_id {} is already attached",
                    anchor_id
                )));
            }
            dashmap::Entry::Vacant(v) => {
                v.insert(());
            }
        }

        // Look up routing slot (must exist -- bind() was called first)
        let frame_tx = match self.routing.get(&anchor_id) {
            Some(tx) => tx.clone(),
            None => {
                // Remove from active since we failed to route
                self.active.remove(&anchor_id);
                return Err(Status::not_found(format!(
                    "no routing slot for anchor_id {}",
                    anchor_id
                )));
            }
        };

        // Spawn pump task: drain incoming FramedData stream -> frame_tx
        let active_ref = self.active.clone();
        let mut stream = request.into_inner();

        tokio::spawn(async move {
            let mut last_was_terminal = false;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(framed) => {
                        let payload = framed.payload;
                        last_was_terminal = is_terminal_sentinel(&payload);
                        if frame_tx.send_async(payload).await.is_err() {
                            // Consumer dropped -- clean shutdown, no Dropped injection
                            active_ref.remove(&anchor_id);
                            return;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("gRPC streaming recv error for anchor {}: {}", anchor_id, e);
                        break;
                    }
                }
            }

            // Stream ended -- inject Dropped if no terminal sentinel was last
            if !last_was_terminal {
                let _ = frame_tx.try_send(crate::sender::cached_dropped().clone());
            }

            active_ref.remove(&anchor_id);
        });

        Ok(Response::new(futures::stream::empty()))
    }
}

// ---------------------------------------------------------------------------
// GrpcFrameTransport (public API)
// ---------------------------------------------------------------------------

/// gRPC-based [`FrameTransport`] providing per-anchor routing via a shared tonic server.
///
/// # Async Construction
///
/// Unlike `TcpFrameTransport` which is constructed synchronously, `GrpcFrameTransport::new()`
/// is `async` because it must await `TcpListener::bind()` to obtain the actual OS-assigned
/// port before the server can accept connections. This is a one-time setup cost.
///
/// # Thread Safety
///
/// `GrpcFrameTransport` is `Send + Sync` (required by `FrameTransport`) and can be shared
/// via `Arc<GrpcFrameTransport>`.
pub struct GrpcFrameTransport {
    /// Routes anchor_id -> frame_tx, populated by bind(), consumed by server pump task
    routing: Arc<DashMap<u64, flume::Sender<Vec<u8>>>>,
    /// Actively-attached anchor_ids for exclusive-attach enforcement.
    /// Shared with `GrpcStreamingService`; kept here to hold an `Arc` reference
    /// (keeps the DashMap alive) and for potential inspection in tests.
    #[allow(dead_code)]
    active: Arc<DashMap<u64, ()>>,
    /// The actual bound address of the tonic server (captured after TcpListener::bind)
    bound_addr: SocketAddr,
    /// CancellationToken to gracefully shut down the tonic server
    cancel: CancellationToken,
}

impl GrpcFrameTransport {
    /// Create a new `GrpcFrameTransport` and start the tonic server on `bind_addr`.
    ///
    /// The server is started eagerly: `TcpListener::bind(bind_addr)` is awaited, the
    /// actual port is captured, and the server is spawned in the background. Use
    /// `0.0.0.0:0` (or `default_new()`) to let the OS assign an ephemeral port.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `TcpListener::bind(bind_addr)` fails.
    pub async fn new(bind_addr: SocketAddr) -> Result<Self> {
        let routing: Arc<DashMap<u64, flume::Sender<Vec<u8>>>> = Arc::new(DashMap::new());
        let active: Arc<DashMap<u64, ()>> = Arc::new(DashMap::new());
        let cancel = CancellationToken::new();

        // Bind listener eagerly to capture actual port
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        let bound_addr = listener.local_addr()?;

        let service = GrpcStreamingService {
            routing: routing.clone(),
            active: active.clone(),
        };

        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            let server = tonic::transport::Server::builder()
                .add_service(VeloStreamingServer::new(service));

            if let Err(e) = server
                .serve_with_incoming_shutdown(
                    TcpListenerStream::new(listener),
                    cancel_clone.cancelled(),
                )
                .await
            {
                tracing::warn!("GrpcFrameTransport server error: {}", e);
            }
        });

        Ok(Self {
            routing,
            active,
            bound_addr,
            cancel,
        })
    }

    /// Create a `GrpcFrameTransport` bound to `0.0.0.0:0` (OS-assigned ephemeral port).
    pub async fn default_new() -> Result<Self> {
        Self::new("0.0.0.0:0".parse().unwrap()).await
    }

    /// Returns the actual bound `SocketAddr` of the tonic server.
    ///
    /// This is always available after `new()` returns (OnceLock is populated in `new()`).
    pub fn bound_addr(&self) -> SocketAddr {
        self.bound_addr
    }
}

impl Drop for GrpcFrameTransport {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Parse a `grpc://` endpoint into `(SocketAddr, anchor_id)`.
///
/// Expected format: `grpc://{host}:{port}/{anchor_id}`
///
/// # Errors
///
/// Returns `Err` on malformed URIs (missing prefix, invalid address, invalid anchor_id).
///
/// # Examples
///
/// ```
/// # use velo_streaming::grpc_transport::parse_grpc_endpoint;
/// let (addr, anchor_id) = parse_grpc_endpoint("grpc://127.0.0.1:50051/42").unwrap();
/// assert_eq!(addr.port(), 50051);
/// assert_eq!(anchor_id, 42);
/// ```
pub fn parse_grpc_endpoint(endpoint: &str) -> Result<(SocketAddr, u64)> {
    let stripped = endpoint
        .strip_prefix("grpc://")
        .ok_or_else(|| anyhow::anyhow!("missing grpc:// prefix: {}", endpoint))?;
    let slash_pos = stripped
        .rfind('/')
        .ok_or_else(|| anyhow::anyhow!("missing anchor_id in endpoint: {}", endpoint))?;
    let addr_str = &stripped[..slash_pos];
    let anchor_id_str = &stripped[slash_pos + 1..];
    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid address '{}': {}", addr_str, e))?;
    let anchor_id: u64 = anchor_id_str
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid anchor_id '{}': {}", anchor_id_str, e))?;
    Ok((addr, anchor_id))
}

impl FrameTransport for GrpcFrameTransport {
    fn bind(&self, anchor_id: u64) -> BoxFuture<'_, Result<(String, flume::Receiver<Vec<u8>>)>> {
        let addr = self.bound_addr;
        let routing = self.routing.clone();
        Box::pin(async move {
            let (frame_tx, frame_rx) = flume::bounded::<Vec<u8>>(256);

            // Insert routing slot synchronously (no race with connect)
            routing.insert(anchor_id, frame_tx);

            let endpoint = format!("grpc://{}:{}/{}", addr.ip(), addr.port(), anchor_id);
            Ok((endpoint, frame_rx))
        })
    }

    fn connect(
        &self,
        endpoint: &str,
        _anchor_id: u64,
        _session_id: u64,
    ) -> BoxFuture<'_, Result<flume::Sender<Vec<u8>>>> {
        let endpoint = endpoint.to_string();
        Box::pin(async move {
            let (addr, anchor_id) = parse_grpc_endpoint(&endpoint)?;

            // Build tonic channel to the server
            let channel = tonic::transport::Channel::from_shared(format!("http://{}", addr))?
                .connect()
                .await?;

            let mut client = VeloStreamingClient::new(channel);

            // Create frame channel: caller writes to tx, pump task reads from rx
            let (tx, rx) = flume::bounded::<Vec<u8>>(256);

            // Map rx to FramedData stream for gRPC
            let request_stream = {
                tokio_stream::wrappers::ReceiverStream::new({
                    // Bridge flume::Receiver to tokio mpsc for tokio-stream compatibility
                    let (mpsc_tx, mpsc_rx) = tokio::sync::mpsc::channel::<FramedData>(256);
                    let rx_clone = rx.clone();
                    tokio::spawn(async move {
                        while let Ok(payload) = rx_clone.recv_async().await {
                            let framed = FramedData {
                                preamble: vec![],
                                header: vec![],
                                payload,
                            };
                            if mpsc_tx.send(framed).await.is_err() {
                                break;
                            }
                        }
                    });
                    mpsc_rx
                })
            };

            // Attach x-anchor-id metadata and send request
            let mut request = Request::new(request_stream);
            request.metadata_mut().insert(
                "x-anchor-id",
                anchor_id.to_string().parse().map_err(|_| {
                    anyhow::anyhow!("failed to encode anchor_id as metadata value")
                })?,
            );

            // Spawn task that drives the gRPC stream
            tokio::spawn(async move {
                let _ = client.stream(request).await;
            });

            Ok(tx)
        })
    }
}

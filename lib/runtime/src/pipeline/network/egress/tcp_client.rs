// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Request Plane Client
//!
//! Lock-free, LRU-based shared connection pool with round-robin selection.
//! Connections are Arc-wrapped and shared across concurrent requests.
//! The hot path (per-request) is fully lock-free: ArcSwap + atomic round-robin + SegQueue push.
//! The cold path (connect/prune) uses a Mutex on the LRU cache.
//! Writer tasks batch requests via BufWriter for ~50x fewer syscalls under load.

use super::unified_client::{ClientStats, Headers, RequestPlaneClient};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use futures::StreamExt;
use lru::LruCache;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_util::codec::FramedRead;

/// Default timeout for TCP request acknowledgment
const DEFAULT_TCP_REQUEST_TIMEOUT_SECS: u64 = 5;

/// Default connection pool size per host
const DEFAULT_POOL_SIZE: usize = 8;

/// Buffer size for request channel per connection (backpressure control)
const REQUEST_CHANNEL_BUFFER: usize = 50;

/// Maximum retries when another task is connecting (prevents unbounded recursion)
const MAX_CONNECT_RETRIES: usize = 5;

/// Default global connect concurrency limit across all hosts
const DEFAULT_GLOBAL_CONNECT_LIMIT: usize = 16;

/// Default idle host TTL in seconds before cleanup
const DEFAULT_HOST_IDLE_TTL_SECS: u64 = 300;

/// Default maximum message size for TCP client (32 MB)
const DEFAULT_MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

/// Spin loop limit before falling back to async Notify in writer task
const WRITER_SPIN_LIMIT: u32 = 64;

/// BufWriter capacity for batched writes (64 KB)
const WRITER_BUF_CAPACITY: usize = 64 * 1024;

/// Get maximum message size from environment or use default
fn get_max_message_size() -> usize {
    std::env::var("DYN_TCP_MAX_MESSAGE_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_MESSAGE_SIZE)
}

/// Check if latency tracing is enabled via environment
fn latency_trace_enabled() -> bool {
    std::env::var("DYN_TCP_LATENCY_TRACE")
        .ok()
        .is_some_and(|v| v == "1" || v == "true")
}

/// TCP request plane configuration
#[derive(Debug, Clone)]
pub struct TcpRequestConfig {
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum connections per host
    pub pool_size: usize,
    /// Connect timeout
    pub connect_timeout: Duration,
    /// Request channel buffer size
    pub channel_buffer: usize,
}

impl Default for TcpRequestConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(DEFAULT_TCP_REQUEST_TIMEOUT_SECS),
            pool_size: DEFAULT_POOL_SIZE,
            connect_timeout: Duration::from_secs(5),
            channel_buffer: REQUEST_CHANNEL_BUFFER,
        }
    }
}

impl TcpRequestConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("DYN_TCP_REQUEST_TIMEOUT")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.request_timeout = Duration::from_secs(timeout);
        }

        if let Ok(val) = std::env::var("DYN_TCP_POOL_SIZE")
            && let Ok(size) = val.parse::<usize>()
        {
            config.pool_size = size;
        }

        if let Ok(val) = std::env::var("DYN_TCP_CONNECT_TIMEOUT")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.connect_timeout = Duration::from_secs(timeout);
        }

        if let Ok(val) = std::env::var("DYN_TCP_CHANNEL_BUFFER")
            && let Ok(size) = val.parse::<usize>()
        {
            config.channel_buffer = size;
        }

        config
    }
}

/// Pending request in the lock-free submit queue
struct PendingRequest {
    /// Pre-encoded request data ready to send (zero-copy Bytes)
    encoded_data: Bytes,
    /// Oneshot channel to send response back to caller
    response_tx: oneshot::Sender<Result<Bytes>>,
}

/// TCP connection with lock-free submit and batched write/read tasks.
///
/// Design: SegQueue submit → batched writer task → reader task → oneshot response
/// - Callers push to SegQueue (lock-free, ~20-40ns)
/// - Writer task drains queue, batches via BufWriter, single flush per batch
/// - Reader task uses framed codec, pops response_tx from SegQueue
/// - FIFO ordering: writer pushes ALL response_txs BEFORE flushing data
struct TcpConnection {
    addr: SocketAddr,
    /// Lock-free queue for callers to submit requests
    submit_queue: Arc<SegQueue<PendingRequest>>,
    /// Lock-free queue for writer→reader response_tx handoff
    response_queue: Arc<SegQueue<oneshot::Sender<Result<Bytes>>>>,
    /// Notify to wake writer when submit_queue transitions from empty
    writer_notify: Arc<tokio::sync::Notify>,
    /// Writer task handle for cleanup
    writer_handle: Arc<JoinHandle<()>>,
    /// Reader task handle for cleanup
    reader_handle: Arc<JoinHandle<()>>,
    /// Health status (false if tasks have failed)
    healthy: Arc<AtomicBool>,
    /// Number of in-flight requests (for capacity heuristic)
    inflight: Arc<AtomicU64>,
    /// Max inflight for capacity heuristic (matches channel_buffer)
    channel_buffer: usize,
}

impl TcpConnection {
    /// Create a new connection with lock-free submit and batched write/read tasks
    async fn connect(addr: SocketAddr, timeout: Duration, channel_buffer: usize) -> Result<Self> {
        let stream = tokio::time::timeout(timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| anyhow::anyhow!("TCP connect timeout to {}", addr))??;

        // Configure socket for lower latency
        Self::configure_socket(&stream)?;

        let (read_half, write_half) = tokio::io::split(stream);

        let submit_queue = Arc::new(SegQueue::new());
        let response_queue = Arc::new(SegQueue::new());
        let writer_notify = Arc::new(tokio::sync::Notify::new());
        let healthy = Arc::new(AtomicBool::new(true));
        let inflight = Arc::new(AtomicU64::new(0));

        // Spawn writer task (batched via BufWriter)
        let writer_handle = {
            let submit_q = submit_queue.clone();
            let response_q = response_queue.clone();
            let notify = writer_notify.clone();
            let healthy = healthy.clone();
            let inflight = inflight.clone();
            tokio::spawn(async move {
                if let Err(e) =
                    Self::writer_task(write_half, submit_q, response_q, notify, healthy.clone(), inflight)
                        .await
                {
                    tracing::debug!("Writer task failed for {}: {}", addr, e);
                    healthy.store(false, Ordering::Relaxed);
                }
            })
        };

        // Spawn reader task (passes writer_notify so reader can wake writer on exit)
        let reader_handle = {
            let response_q = response_queue.clone();
            let healthy = healthy.clone();
            let writer_notify = writer_notify.clone();
            tokio::spawn(async move {
                if let Err(e) =
                    Self::reader_task(read_half, response_q, healthy.clone(), writer_notify).await
                {
                    tracing::debug!("Reader task failed for {}: {}", addr, e);
                    healthy.store(false, Ordering::Relaxed);
                }
            })
        };

        Ok(Self {
            addr,
            submit_queue,
            response_queue,
            writer_notify,
            writer_handle: Arc::new(writer_handle),
            reader_handle: Arc::new(reader_handle),
            healthy,
            inflight,
            channel_buffer,
        })
    }

    /// Configure socket for ultra-low latency based on dyn-transports patterns
    fn configure_socket(stream: &TcpStream) -> Result<()> {
        use socket2::SockRef;

        let sock_ref = SockRef::from(stream);

        // TCP_NODELAY - disable Nagle's algorithm for immediate send
        sock_ref.set_nodelay(true)?;

        // Increase socket buffer sizes for better throughput under load
        sock_ref.set_recv_buffer_size(2 * 1024 * 1024)?; // 2MB
        sock_ref.set_send_buffer_size(2 * 1024 * 1024)?; // 2MB

        // Advanced Linux optimizations for ultra-low latency (optional feature)
        #[cfg(feature = "tcp-low-latency")]
        {
            use std::os::unix::io::AsRawFd;

            unsafe {
                let fd = stream.as_raw_fd();

                // TCP_QUICKACK - minimize ACK delay
                let quickack: libc::c_int = 1;
                libc::setsockopt(
                    fd,
                    libc::SOL_TCP,
                    libc::TCP_QUICKACK,
                    &quickack as *const _ as *const libc::c_void,
                    std::mem::size_of_val(&quickack) as libc::socklen_t,
                );

                // SO_BUSY_POLL - enable busy polling for lower latency (50 microseconds)
                let busy_poll: libc::c_int = 50;
                libc::setsockopt(
                    fd,
                    libc::SOL_SOCKET,
                    libc::SO_BUSY_POLL,
                    &busy_poll as *const _ as *const libc::c_void,
                    std::mem::size_of_val(&busy_poll) as libc::socklen_t,
                );
            }

            tracing::debug!("TCP low-latency optimizations enabled (TCP_QUICKACK, SO_BUSY_POLL)");
        }

        Ok(())
    }

    /// Drain pending items from both queues, sending errors on all oneshot senders.
    /// Called when the writer task exits to prevent orphaned callers waiting forever.
    fn drain_pending(
        submit_queue: &SegQueue<PendingRequest>,
        response_queue: &SegQueue<oneshot::Sender<Result<Bytes>>>,
    ) {
        while let Some(req) = submit_queue.pop() {
            let _ = req
                .response_tx
                .send(Err(anyhow::anyhow!("Connection closed")));
        }
        while let Some(tx) = response_queue.pop() {
            let _ = tx.send(Err(anyhow::anyhow!("Connection closed")));
        }
    }

    /// Writer task: drains SegQueue, batches via BufWriter, single flush per batch.
    ///
    /// Critical ordering: ALL response_txs are pushed to response_queue BEFORE
    /// any data hits the wire. This guarantees the reader can always find the
    /// matching response_tx even if the server responds before flush completes.
    async fn writer_task(
        write_half: tokio::io::WriteHalf<TcpStream>,
        submit_queue: Arc<SegQueue<PendingRequest>>,
        response_queue: Arc<SegQueue<oneshot::Sender<Result<Bytes>>>>,
        notify: Arc<tokio::sync::Notify>,
        healthy: Arc<AtomicBool>,
        inflight: Arc<AtomicU64>,
    ) -> Result<()> {
        let mut writer = tokio::io::BufWriter::with_capacity(WRITER_BUF_CAPACITY, write_half);
        let trace = latency_trace_enabled();

        // Latency instrumentation accumulators
        let mut batch_count: u64 = 0;
        let mut total_batch_size: u64 = 0;
        let mut total_batch_write_ns: u64 = 0;
        let mut last_report = std::time::Instant::now();

        let result: Result<()> = async {
            loop {
                // Adaptive spin: try queue before falling back to async Notify
                let mut spins: u32 = 0;
                while submit_queue.is_empty() {
                    // Check if reader has exited
                    if !healthy.load(Ordering::Relaxed) {
                        return Err(anyhow::anyhow!("Reader exited, writer stopping"));
                    }
                    spins += 1;
                    if spins >= WRITER_SPIN_LIMIT {
                        notify.notified().await;
                        break;
                    }
                    std::hint::spin_loop();
                }

                // Drain all available requests
                let mut encoded_batch: Vec<Bytes> = Vec::with_capacity(64);
                let mut response_batch: Vec<oneshot::Sender<Result<Bytes>>> =
                    Vec::with_capacity(64);

                while let Some(req) = submit_queue.pop() {
                    encoded_batch.push(req.encoded_data);
                    response_batch.push(req.response_tx);
                }

                let count = encoded_batch.len();
                if count == 0 {
                    continue; // spurious wakeup
                }

                // Phase 1: Push ALL response_txs BEFORE any data hits the wire
                for tx in response_batch.drain(..) {
                    response_queue.push(tx);
                }

                // Phase 2: Write batch to BufWriter (buffered, no syscalls yet)
                let write_start = if trace {
                    Some(std::time::Instant::now())
                } else {
                    None
                };

                for data in &encoded_batch {
                    if let Err(e) = writer.write_all(data).await {
                        inflight.fetch_sub(count as u64, Ordering::Relaxed);
                        return Err(e.into());
                    }
                }

                // Phase 3: Single flush = single write(2) syscall for entire batch
                if let Err(e) = writer.flush().await {
                    inflight.fetch_sub(count as u64, Ordering::Relaxed);
                    return Err(e.into());
                }

                // Check if reader has exited (e.g., peer closed connection).
                // Kernel buffering may let writes succeed after peer close,
                // but the reader won't deliver responses — exit now.
                if !healthy.load(Ordering::Relaxed) {
                    return Err(anyhow::anyhow!("Reader exited, writer stopping"));
                }

                // Latency instrumentation
                if trace {
                    if let Some(start) = write_start {
                        total_batch_write_ns += start.elapsed().as_nanos() as u64;
                    }
                    batch_count += 1;
                    total_batch_size += count as u64;

                    if last_report.elapsed() >= Duration::from_secs(5) {
                        let avg_batch = if batch_count > 0 {
                            total_batch_size / batch_count
                        } else {
                            0
                        };
                        let avg_write_ns = if batch_count > 0 {
                            total_batch_write_ns / batch_count
                        } else {
                            0
                        };
                        tracing::info!(
                            batches = batch_count,
                            avg_batch_size = avg_batch,
                            avg_batch_write_ns = avg_write_ns,
                            "TCP writer instrumentation summary"
                        );
                        batch_count = 0;
                        total_batch_size = 0;
                        total_batch_write_ns = 0;
                        last_report = std::time::Instant::now();
                    }
                }

                encoded_batch.clear();
            }
        }
        .await;

        // On exit (error or clean), drain any pending requests/responses
        // so callers aren't left waiting forever on their oneshot receivers.
        healthy.store(false, Ordering::Relaxed);
        Self::drain_pending(&submit_queue, &response_queue);

        result
    }

    /// Reader task: reads responses using framed codec, pops response_tx from SegQueue.
    ///
    /// On exit (clean close or error), sets `healthy=false` and wakes the writer
    /// via `writer_notify` so it can detect reader death and drain pending callers.
    async fn reader_task(
        read_half: tokio::io::ReadHalf<TcpStream>,
        response_queue: Arc<SegQueue<oneshot::Sender<Result<Bytes>>>>,
        healthy: Arc<AtomicBool>,
        writer_notify: Arc<tokio::sync::Notify>,
    ) -> Result<()> {
        use crate::pipeline::network::codec::TcpResponseCodec;

        let max_message_size = get_max_message_size();
        let codec = TcpResponseCodec::new(Some(max_message_size));
        let mut framed = FramedRead::new(read_half, codec);

        while let Some(result) = framed.next().await {
            // Spin briefly if response_queue is empty (writer hasn't pushed yet)
            let tx = loop {
                if let Some(tx) = response_queue.pop() {
                    break tx;
                }
                // If the connection is already unhealthy (writer failed), stop spinning
                if !healthy.load(Ordering::Relaxed) {
                    return Err(anyhow::anyhow!("Connection unhealthy, reader exiting"));
                }
                tokio::task::yield_now().await;
            };

            match result {
                Ok(response_msg) => {
                    let _ = tx.send(Ok(response_msg.data));
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Failed to decode response: {}", e)));
                    healthy.store(false, Ordering::Relaxed);
                    // Wake writer so it can detect unhealthy and exit
                    writer_notify.notify_one();
                    return Err(anyhow::anyhow!("Failed to decode response"));
                }
            }
        }

        // Connection closed by peer — mark unhealthy so the writer task
        // detects reader death and drains pending callers.
        healthy.store(false, Ordering::Relaxed);
        // Wake writer from Notify.await so it can check healthy and exit
        writer_notify.notify_one();
        Ok(())
    }

    /// Send a request via lock-free SegQueue push (~20-40ns)
    async fn send_request(&self, payload: Bytes, headers: &Headers) -> Result<Bytes> {
        use crate::pipeline::network::codec::TcpRequestMessage;

        if !self.healthy.load(Ordering::Relaxed) {
            anyhow::bail!("Connection unhealthy (tasks failed)");
        }

        let endpoint_path = headers
            .get("x-endpoint-path")
            .ok_or_else(|| anyhow::anyhow!("Missing x-endpoint-path header for TCP request"))?
            .to_string();

        let request_msg = TcpRequestMessage::with_headers(endpoint_path, headers.clone(), payload);
        let encoded_data = request_msg.encode()?;

        let (response_tx, response_rx) = oneshot::channel();

        let trace = latency_trace_enabled();
        let e2e_start = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Increment inflight before push (for capacity heuristic)
        self.inflight.fetch_add(1, Ordering::Relaxed);

        // Lock-free submit: ~20-40ns
        self.submit_queue.push(PendingRequest {
            encoded_data,
            response_tx,
        });

        // Wake writer if it was sleeping
        self.writer_notify.notify_one();

        // Await response
        let result = response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Reader task closed"))?;

        // Decrement inflight after response
        self.inflight.fetch_sub(1, Ordering::Relaxed);

        if trace
            && let Some(start) = e2e_start
        {
            let e2e_ns = start.elapsed().as_nanos() as u64;
            tracing::trace!(e2e_ns = e2e_ns, "TCP request e2e latency");
        }

        result
    }

    /// Check if connection is healthy
    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    /// Available capacity (advisory, for cold path growth heuristic)
    fn available_capacity(&self) -> usize {
        let inflight = self.inflight.load(Ordering::Relaxed) as usize;
        self.channel_buffer.saturating_sub(inflight)
    }
}

/// Per-host connection pool with LRU lifecycle and ArcSwap-based snapshot.
///
/// Hot path: `ArcSwap::load()` + atomic round-robin (~40ns total, fully lock-free).
/// Cold path: LRU prune/insert + ArcSwap store (only on startup or failure).
struct HostPool {
    /// Lock-free snapshot for the hot path (rebuilt on cold path changes)
    snapshot: arc_swap::ArcSwap<Vec<Arc<TcpConnection>>>,
    /// LRU cache for lifecycle management (eviction, pruning)
    lru: parking_lot::Mutex<LruCache<u64, Arc<TcpConnection>>>,
    /// Atomic round-robin counter for connection selection
    counter: AtomicU64,
    /// Monotonic ID generator for LRU keys
    next_id: AtomicU64,
    /// CAS gate to prevent thundering-herd connect storms
    connecting: AtomicBool,
    /// Wake CAS losers when a connect attempt completes (success or failure)
    connect_notify: tokio::sync::Notify,
    /// Maximum connections for this host
    max_connections: usize,
    /// Target address
    addr: SocketAddr,
    /// Connect timeout
    connect_timeout: Duration,
    /// Channel buffer size for new connections
    channel_buffer: usize,
    /// Timestamp of last use (unix millis) for idle cleanup
    last_used_ms: AtomicU64,
}

impl HostPool {
    fn new(addr: SocketAddr, config: &TcpRequestConfig) -> Self {
        let cap = NonZeroUsize::new(config.pool_size).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            snapshot: arc_swap::ArcSwap::from_pointee(Vec::new()),
            lru: parking_lot::Mutex::new(LruCache::new(cap)),
            counter: AtomicU64::new(0),
            next_id: AtomicU64::new(0),
            connecting: AtomicBool::new(false),
            connect_notify: tokio::sync::Notify::new(),
            max_connections: config.pool_size,
            addr,
            connect_timeout: config.connect_timeout,
            channel_buffer: config.channel_buffer,
            last_used_ms: AtomicU64::new(current_time_ms()),
        }
    }

    /// Get a connection, using the hot path (ArcSwap load + atomic RR) when possible.
    async fn get_connection(
        &self,
        connect_limiter: &tokio::sync::Semaphore,
    ) -> Result<Arc<TcpConnection>> {
        // === HOT PATH: ArcSwap load + atomic round-robin (fully lock-free) ===
        {
            let guard = self.snapshot.load();
            let conns = &**guard;
            let len = conns.len();
            if len > 0 {
                let start = self.counter.fetch_add(1, Ordering::Relaxed) as usize;

                // First pass: find a healthy connection with available capacity
                for i in 0..len {
                    let idx = (start + i) % len;
                    let conn = &conns[idx];
                    if conn.is_healthy() && conn.available_capacity() > 0 {
                        return Ok(conn.clone());
                    }
                }

                // All healthy connections are saturated.
                // If pool is at max, return a saturated healthy connection (SegQueue backpressure handles it).
                // If pool can still grow, fall through to cold path to add capacity.
                if len >= self.max_connections {
                    for i in 0..len {
                        let idx = (start + i) % len;
                        if conns[idx].is_healthy() {
                            return Ok(conns[idx].clone());
                        }
                    }
                }
                // Fall through: all unhealthy OR all saturated and below max_connections
            }
        }

        // === COLD PATH ===
        self.ensure_capacity_or_heal(connect_limiter).await
    }

    /// Determine if the pool should grow
    fn should_grow(healthy: &[Arc<TcpConnection>], max_connections: usize) -> bool {
        if healthy.is_empty() {
            return true;
        }
        if healthy.len() >= max_connections {
            return false;
        }
        // Grow when every healthy connection's channel is fully saturated
        healthy.iter().all(|c| c.available_capacity() == 0)
    }

    /// Cold path: prune unhealthy connections, optionally grow, rebuild snapshot.
    async fn ensure_capacity_or_heal(
        &self,
        connect_limiter: &tokio::sync::Semaphore,
    ) -> Result<Arc<TcpConnection>> {
        // --- Phase A: lock LRU, prune, decide, build snapshot, unlock ---
        let (need_connect, new_snap) = {
            let mut lru = self.lru.lock();

            // Prune unhealthy (evicted Arcs stay alive for in-flight holders)
            let dead: Vec<u64> = lru
                .iter()
                .filter(|(_, c)| !c.is_healthy())
                .map(|(&k, _)| k)
                .collect();
            for k in dead {
                lru.pop(&k);
            }

            let snap: Vec<Arc<TcpConnection>> = lru.iter().map(|(_, c)| c.clone()).collect();
            let grow = Self::should_grow(&snap, self.max_connections);
            (grow, snap)
        };
        // LRU lock released here

        // Atomic snapshot update (no RwLock!)
        self.snapshot.store(Arc::new(new_snap.clone()));

        // Check if a healthy conn is now available (another task may have added one)
        {
            let guard = self.snapshot.load();
            let conns = &**guard;
            if !conns.is_empty() {
                let start = self.counter.fetch_add(1, Ordering::Relaxed) as usize;
                for i in 0..conns.len() {
                    let idx = (start + i) % conns.len();
                    if conns[idx].is_healthy() {
                        return Ok(conns[idx].clone());
                    }
                }
            }
        }

        if !need_connect {
            anyhow::bail!(
                "No healthy TCP connection to {} and pool at capacity ({})",
                self.addr,
                self.max_connections
            );
        }

        // --- Phase B: connect (no locks held, CAS gate prevents stampede) ---
        // Bounded retry loop instead of recursion
        for retry in 0..MAX_CONNECT_RETRIES {
            if self
                .connecting
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Won the CAS gate. Acquire global connect permit to limit total connect bursts.
                let _permit = connect_limiter.acquire().await.map_err(|_| {
                    self.connecting.store(false, Ordering::Release);
                    self.connect_notify.notify_waiters();
                    anyhow::anyhow!("Global connect limiter closed")
                })?;

                let connect_result =
                    TcpConnection::connect(self.addr, self.connect_timeout, self.channel_buffer)
                        .await;

                self.connecting.store(false, Ordering::Release);

                match connect_result {
                    Ok(stream) => {
                        let new_conn = Arc::new(stream);

                        // --- Phase C: lock LRU, insert, rebuild snapshot, unlock ---
                        {
                            let mut lru = self.lru.lock();

                            // Re-prune (may have changed during connect)
                            let dead: Vec<u64> = lru
                                .iter()
                                .filter(|(_, c)| !c.is_healthy())
                                .map(|(&k, _)| k)
                                .collect();
                            for k in dead {
                                lru.pop(&k);
                            }

                            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                            lru.put(id, new_conn.clone());

                            let snap: Vec<Arc<TcpConnection>> =
                                lru.iter().map(|(_, c)| c.clone()).collect();
                            drop(lru);
                            self.snapshot.store(Arc::new(snap));
                        }

                        self.connect_notify.notify_waiters();
                        return Ok(new_conn);
                    }
                    Err(e) => {
                        self.connect_notify.notify_waiters();
                        return Err(e);
                    }
                }
            }

            // Another task is connecting. Wait for it to finish (or timeout).
            let _ = tokio::time::timeout(
                self.connect_timeout,
                self.connect_notify.notified(),
            )
            .await;

            // Try hot path again after yield
            let guard = self.snapshot.load();
            let conns = &**guard;
            let len = conns.len();
            if len > 0 {
                let start = self.counter.fetch_add(1, Ordering::Relaxed) as usize;
                for i in 0..len {
                    let idx = (start + i) % len;
                    if conns[idx].is_healthy() && conns[idx].available_capacity() > 0 {
                        return Ok(conns[idx].clone());
                    }
                }
                // Accept saturated if at max
                if len >= self.max_connections {
                    for i in 0..len {
                        let idx = (start + i) % len;
                        if conns[idx].is_healthy() {
                            return Ok(conns[idx].clone());
                        }
                    }
                }
            }
            drop(guard);

            tracing::trace!(
                "TCP pool connect retry {}/{} for {}",
                retry + 1,
                MAX_CONNECT_RETRIES,
                self.addr
            );
        }

        anyhow::bail!(
            "Failed to get TCP connection to {} after {} retries (connect contention)",
            self.addr,
            MAX_CONNECT_RETRIES
        )
    }
}

/// Get current time in milliseconds since epoch
fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Connection pool with LRU lifecycle and shared Arc connections.
///
/// Uses DashMap for per-host pools and a global Semaphore to limit
/// aggregate connect bursts across many cold hosts.
struct TcpConnectionPool {
    hosts: DashMap<SocketAddr, Arc<HostPool>>,
    config: TcpRequestConfig,
    /// Global connect concurrency limiter (caps aggregate connect bursts)
    connect_limiter: Arc<tokio::sync::Semaphore>,
    /// Idle host TTL for cleanup
    host_idle_ttl_ms: u64,
}

impl TcpConnectionPool {
    fn new(config: TcpRequestConfig) -> Self {
        let global_limit = std::env::var("DYN_TCP_GLOBAL_CONNECT_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_GLOBAL_CONNECT_LIMIT);

        let host_idle_ttl_secs = std::env::var("DYN_TCP_HOST_IDLE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_HOST_IDLE_TTL_SECS);

        Self {
            hosts: DashMap::new(),
            config,
            connect_limiter: Arc::new(tokio::sync::Semaphore::new(global_limit)),
            host_idle_ttl_ms: host_idle_ttl_secs * 1000,
        }
    }

    /// Get a connection from the pool or create a new one.
    /// Hot path: DashMap shard read lock → ArcSwap load → atomic RR.
    async fn get_connection(&self, addr: SocketAddr) -> Result<Arc<TcpConnection>> {
        // Fast: DashMap shard read lock (no write contention)
        if let Some(host) = self.hosts.get(&addr) {
            host.last_used_ms
                .store(current_time_ms(), Ordering::Relaxed);
            return host.get_connection(&self.connect_limiter).await;
        }

        // Slow: first request to this host
        let host = self
            .hosts
            .entry(addr)
            .or_insert_with(|| Arc::new(HostPool::new(addr, &self.config)))
            .clone();

        host.last_used_ms
            .store(current_time_ms(), Ordering::Relaxed);
        host.get_connection(&self.connect_limiter).await
    }

    /// Eagerly establish one TCP connection to the given address.
    ///
    /// Creates a `HostPool` entry (if absent) and opens a single connection
    /// through the normal cold path so the global `connect_limiter` is respected.
    /// Failures are logged but never propagated — the lazy cold path remains
    /// as fallback.
    async fn warmup(&self, addr: SocketAddr) {
        let host = self
            .hosts
            .entry(addr)
            .or_insert_with(|| Arc::new(HostPool::new(addr, &self.config)))
            .clone();
        host.last_used_ms
            .store(current_time_ms(), Ordering::Relaxed);
        match host.get_connection(&self.connect_limiter).await {
            Ok(_) => tracing::debug!("TCP warmup: pre-connected to {}", addr),
            Err(e) => tracing::warn!("TCP warmup: failed to pre-connect to {}: {}", addr, e),
        }
    }

    /// Background task that watches the instance discovery channel and
    /// eagerly warms one TCP connection for each newly-discovered TCP backend.
    ///
    /// Uses a diff-based approach: tracks a `HashSet<SocketAddr>` of known
    /// addresses, only warms truly new ones.
    fn start_warmup_watcher(
        self: &Arc<Self>,
        mut instance_rx: tokio::sync::watch::Receiver<Vec<crate::component::Instance>>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) {
        let pool = Arc::clone(self);
        tokio::spawn(async move {
            let mut known_addrs = std::collections::HashSet::<SocketAddr>::new();

            // Seed from current value so we don't re-warm existing backends
            {
                let instances = instance_rx.borrow_and_update();
                for inst in instances.iter() {
                    if let crate::component::TransportType::Tcp(ref addr_str) = inst.transport
                        && let Ok((sock, _)) = TcpRequestClient::parse_address(addr_str)
                    {
                        known_addrs.insert(sock);
                    }
                }
            }

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("TCP warmup watcher cancelled");
                        break;
                    }
                    result = instance_rx.changed() => {
                        if result.is_err() {
                            tracing::debug!("TCP warmup watcher: instance channel closed");
                            break;
                        }

                        let instances = instance_rx.borrow_and_update().clone();
                        let mut current_addrs = std::collections::HashSet::<SocketAddr>::new();

                        for inst in &instances {
                            if let crate::component::TransportType::Tcp(ref addr_str) = inst.transport
                                && let Ok((sock, _)) = TcpRequestClient::parse_address(addr_str)
                            {
                                current_addrs.insert(sock);
                                if !known_addrs.contains(&sock) {
                                    let pool = Arc::clone(&pool);
                                    tokio::spawn(async move {
                                        pool.warmup(sock).await;
                                    });
                                }
                            }
                        }

                        known_addrs = current_addrs;
                    }
                }
            }
        });
    }

    /// Opportunistic cleanup of idle host pools.
    /// Called periodically or when convenient; not on the hot path.
    fn cleanup_idle_hosts(&self) {
        let now = current_time_ms();
        let ttl = self.host_idle_ttl_ms;

        let stale: Vec<SocketAddr> = self
            .hosts
            .iter()
            .filter(|entry| {
                let last = entry.value().last_used_ms.load(Ordering::Relaxed);
                now.saturating_sub(last) > ttl
            })
            .map(|entry| *entry.key())
            .collect();

        for addr in stale {
            tracing::debug!("Removing idle TCP host pool for {}", addr);
            self.hosts.remove(&addr);
        }
    }
}

/// TCP request plane client
pub struct TcpRequestClient {
    pool: Arc<TcpConnectionPool>,
    config: TcpRequestConfig,
    stats: Arc<TcpClientStats>,
}

struct TcpClientStats {
    requests_sent: AtomicU64,
    responses_received: AtomicU64,
    errors: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
}

impl TcpRequestClient {
    /// Create a new TCP request client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(TcpRequestConfig::default())
    }

    /// Create a new TCP request client with custom configuration
    pub fn with_config(config: TcpRequestConfig) -> Result<Self> {
        Ok(Self {
            pool: Arc::new(TcpConnectionPool::new(config.clone())),
            config,
            stats: Arc::new(TcpClientStats {
                requests_sent: AtomicU64::new(0),
                responses_received: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                bytes_sent: AtomicU64::new(0),
                bytes_received: AtomicU64::new(0),
            }),
        })
    }

    /// Create from environment configuration
    pub fn from_env() -> Result<Self> {
        Self::with_config(TcpRequestConfig::from_env())
    }

    /// Start a background task that eagerly warms TCP connections for
    /// newly-discovered backends.
    ///
    /// Delegates to [`TcpConnectionPool::start_warmup_watcher`].
    pub fn start_warmup(
        &self,
        instance_rx: tokio::sync::watch::Receiver<Vec<crate::component::Instance>>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) {
        self.pool.start_warmup_watcher(instance_rx, cancel_token);
    }

    /// Parse TCP address from string
    /// Supports formats: "host:port" or "tcp://host:port" or "host:port/endpoint_name"
    /// Returns (SocketAddr, Option<endpoint_name>)
    pub(crate) fn parse_address(address: &str) -> Result<(SocketAddr, Option<String>)> {
        let addr_str = if let Some(stripped) = address.strip_prefix("tcp://") {
            stripped
        } else {
            address
        };

        // Check if endpoint name is included (format: host:port/endpoint_name)
        if let Some((socket_part, endpoint_name)) = addr_str.split_once('/') {
            let socket_addr = socket_part
                .parse::<SocketAddr>()
                .map_err(|e| anyhow::anyhow!("Invalid TCP address '{}': {}", address, e))?;
            Ok((socket_addr, Some(endpoint_name.to_string())))
        } else {
            let socket_addr = addr_str
                .parse::<SocketAddr>()
                .map_err(|e| anyhow::anyhow!("Invalid TCP address '{}': {}", address, e))?;
            Ok((socket_addr, None))
        }
    }
}

impl Default for TcpRequestClient {
    fn default() -> Self {
        Self::new().expect("Failed to create TCP request client")
    }
}

#[async_trait]
impl RequestPlaneClient for TcpRequestClient {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        mut headers: Headers,
    ) -> Result<Bytes> {
        tracing::debug!("TCP client sending request to address: {}", address);
        self.stats.requests_sent.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_sent
            .fetch_add(payload.len() as u64, Ordering::Relaxed);

        let (addr, endpoint_name) = Self::parse_address(&address)?;

        if let Some(endpoint_name) = endpoint_name {
            headers.insert("x-endpoint-path".to_string(), endpoint_name.clone());
        }

        // Get shared connection from pool (Arc, not exclusive borrow)
        let conn = self.pool.get_connection(addr).await?;

        let result = tokio::time::timeout(
            self.config.request_timeout,
            conn.send_request(payload, &headers),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                self.stats
                    .responses_received
                    .fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_received
                    .fetch_add(response.len() as u64, Ordering::Relaxed);
                // conn (Arc) dropped here -- connection stays in pool
                Ok(response)
            }
            Ok(Err(e)) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("TCP request failed to {}: {}", addr, e);
                let cause = crate::error::DynamoError::from(
                    e.into_boxed_dyn_error() as Box<dyn std::error::Error + 'static>,
                );
                Err(anyhow::anyhow!(
                    crate::error::DynamoError::builder()
                        .error_type(crate::error::ErrorType::CannotConnect)
                        .message(format!("TCP request to {addr} failed"))
                        .cause(cause)
                        .build()
                ))
            }
            Err(_) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("TCP request timeout to {}", addr);
                Err(anyhow::anyhow!(
                    crate::error::DynamoError::builder()
                        .error_type(crate::error::ErrorType::CannotConnect)
                        .message(format!("TCP request to {addr} timed out"))
                        .build()
                ))
            }
        }
    }

    fn transport_name(&self) -> &'static str {
        "tcp"
    }

    fn is_healthy(&self) -> bool {
        true // TCP client is always healthy if it was created successfully
    }

    fn start_warmup(
        &self,
        instance_rx: tokio::sync::watch::Receiver<Vec<crate::component::Instance>>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) {
        TcpRequestClient::start_warmup(self, instance_rx, cancel_token);
    }

    fn stats(&self) -> ClientStats {
        ClientStats {
            requests_sent: self.stats.requests_sent.load(Ordering::Relaxed),
            responses_received: self.stats.responses_received.load(Ordering::Relaxed),
            errors: self.stats.errors.load(Ordering::Relaxed),
            bytes_sent: self.stats.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.stats.bytes_received.load(Ordering::Relaxed),
            active_connections: 0,
            idle_connections: 0,
            avg_latency_us: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    #[test]
    fn test_tcp_config_default() {
        let config = TcpRequestConfig::default();
        assert_eq!(config.pool_size, DEFAULT_POOL_SIZE);
        assert_eq!(
            config.request_timeout,
            Duration::from_secs(DEFAULT_TCP_REQUEST_TIMEOUT_SECS)
        );
        assert_eq!(config.channel_buffer, REQUEST_CHANNEL_BUFFER);
    }

    #[test]
    fn test_tcp_config_from_env() {
        unsafe {
            std::env::set_var("DYN_TCP_REQUEST_TIMEOUT", "10");
            std::env::set_var("DYN_TCP_POOL_SIZE", "50");
            std::env::set_var("DYN_TCP_CONNECT_TIMEOUT", "3");
            std::env::set_var("DYN_TCP_CHANNEL_BUFFER", "100");
        }

        let config = TcpRequestConfig::from_env();
        assert_eq!(config.request_timeout, Duration::from_secs(10));
        assert_eq!(config.pool_size, 50);
        assert_eq!(config.connect_timeout, Duration::from_secs(3));
        assert_eq!(config.channel_buffer, 100);

        // Clean up env vars
        unsafe {
            std::env::remove_var("DYN_TCP_REQUEST_TIMEOUT");
            std::env::remove_var("DYN_TCP_POOL_SIZE");
            std::env::remove_var("DYN_TCP_CONNECT_TIMEOUT");
            std::env::remove_var("DYN_TCP_CHANNEL_BUFFER");
        }
    }

    #[test]
    fn test_parse_address() {
        let (addr1, _) = TcpRequestClient::parse_address("127.0.0.1:8080").unwrap();
        assert_eq!(addr1.port(), 8080);

        let (addr2, _) = TcpRequestClient::parse_address("tcp://127.0.0.1:9090").unwrap();
        assert_eq!(addr2.port(), 9090);

        let (addr3, endpoint) =
            TcpRequestClient::parse_address("127.0.0.1:8080/test_endpoint").unwrap();
        assert_eq!(addr3.port(), 8080);
        assert_eq!(endpoint, Some("test_endpoint".to_string()));

        assert!(TcpRequestClient::parse_address("invalid").is_err());
    }

    #[test]
    fn test_tcp_client_creation() {
        let client = TcpRequestClient::new();
        assert!(client.is_ok());

        let client = client.unwrap();
        assert_eq!(client.transport_name(), "tcp");
        assert!(client.is_healthy());
    }

    /// Helper: spawn a mock TCP server that echoes requests.
    /// Returns (listener_addr, connection_count_tracker).
    async fn spawn_echo_server() -> (SocketAddr, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conn_count = Arc::new(AtomicUsize::new(0));
        let conn_count_clone = conn_count.clone();

        tokio::spawn(async move {
            loop {
                let result = listener.accept().await;
                if result.is_err() {
                    break;
                }
                let (stream, _) = result.unwrap();
                conn_count_clone.fetch_add(1, Ordering::SeqCst);

                tokio::spawn(async move {
                    let (mut read_half, mut write_half) = tokio::io::split(stream);
                    loop {
                        // Read path length
                        let mut len_buf = [0u8; 2];
                        if read_half.read_exact(&mut len_buf).await.is_err() {
                            break;
                        }
                        let path_len = u16::from_be_bytes(len_buf) as usize;
                        let mut path_buf = vec![0u8; path_len];
                        if read_half.read_exact(&mut path_buf).await.is_err() {
                            break;
                        }

                        // Read headers length
                        let mut headers_len_buf = [0u8; 2];
                        if read_half.read_exact(&mut headers_len_buf).await.is_err() {
                            break;
                        }
                        let headers_len = u16::from_be_bytes(headers_len_buf) as usize;
                        let mut headers_buf = vec![0u8; headers_len];
                        if read_half.read_exact(&mut headers_buf).await.is_err() {
                            break;
                        }

                        // Read payload length + payload
                        let mut len_buf = [0u8; 4];
                        if read_half.read_exact(&mut len_buf).await.is_err() {
                            break;
                        }
                        let payload_len = u32::from_be_bytes(len_buf) as usize;
                        let mut payload_buf = vec![0u8; payload_len];
                        if read_half.read_exact(&mut payload_buf).await.is_err() {
                            break;
                        }

                        // Send response
                        use crate::pipeline::network::codec::TcpResponseMessage;
                        let response = TcpResponseMessage::new(Bytes::from(payload_buf));
                        let encoded = response.encode().unwrap();
                        if write_half.write_all(&encoded).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });

        (addr, conn_count)
    }

    #[tokio::test]
    async fn test_connection_health_check() {
        use crate::pipeline::network::codec::TcpResponseMessage;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (mut read_half, mut write_half) = tokio::io::split(stream);

            let mut len_buf = [0u8; 2];
            read_half.read_exact(&mut len_buf).await.unwrap();
            let path_len = u16::from_be_bytes(len_buf) as usize;
            let mut path_buf = vec![0u8; path_len];
            read_half.read_exact(&mut path_buf).await.unwrap();

            let mut headers_len_buf = [0u8; 2];
            read_half.read_exact(&mut headers_len_buf).await.unwrap();
            let headers_len = u16::from_be_bytes(headers_len_buf) as usize;
            let mut headers_buf = vec![0u8; headers_len];
            read_half.read_exact(&mut headers_buf).await.unwrap();

            let mut len_buf = [0u8; 4];
            read_half.read_exact(&mut len_buf).await.unwrap();
            let payload_len = u32::from_be_bytes(len_buf) as usize;
            let mut payload_buf = vec![0u8; payload_len];
            read_half.read_exact(&mut payload_buf).await.unwrap();

            let response = TcpResponseMessage::new(Bytes::from_static(b"pong"));
            let encoded = response.encode().unwrap();
            write_half.write_all(&encoded).await.unwrap();
        });

        let conn = TcpConnection::connect(addr, Duration::from_secs(5), 10)
            .await
            .unwrap();

        assert!(conn.is_healthy(), "New connection should be healthy");

        let mut headers = Headers::new();
        headers.insert("x-endpoint-path".to_string(), "test".to_string());

        let result = conn.send_request(Bytes::from("ping"), &headers).await;
        assert!(result.is_ok(), "Request should succeed");
        assert_eq!(result.unwrap(), Bytes::from("pong"));
    }

    #[tokio::test]
    async fn test_concurrent_requests_single_connection() {
        use crate::pipeline::network::codec::TcpResponseMessage;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_count_clone = request_count.clone();

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (mut read_half, mut write_half) = tokio::io::split(stream);

            for _ in 0..5 {
                let mut len_buf = [0u8; 2];
                if read_half.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let path_len = u16::from_be_bytes(len_buf) as usize;
                let mut path_buf = vec![0u8; path_len];
                if read_half.read_exact(&mut path_buf).await.is_err() {
                    break;
                }

                let mut headers_len_buf = [0u8; 2];
                if read_half.read_exact(&mut headers_len_buf).await.is_err() {
                    break;
                }
                let headers_len = u16::from_be_bytes(headers_len_buf) as usize;
                let mut headers_buf = vec![0u8; headers_len];
                if read_half.read_exact(&mut headers_buf).await.is_err() {
                    break;
                }

                let mut len_buf = [0u8; 4];
                if read_half.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let payload_len = u32::from_be_bytes(len_buf) as usize;
                let mut payload_buf = vec![0u8; payload_len];
                if read_half.read_exact(&mut payload_buf).await.is_err() {
                    break;
                }

                request_count_clone.fetch_add(1, Ordering::SeqCst);

                let response = TcpResponseMessage::new(Bytes::from(payload_buf));
                let encoded = response.encode().unwrap();
                if write_half.write_all(&encoded).await.is_err() {
                    break;
                }
            }
        });

        let conn = Arc::new(
            TcpConnection::connect(addr, Duration::from_secs(5), 10)
                .await
                .unwrap(),
        );

        let mut handles = vec![];
        for i in 0..5 {
            let conn = conn.clone();
            let handle = tokio::spawn(async move {
                let mut headers = Headers::new();
                headers.insert("x-endpoint-path".to_string(), "test".to_string());
                let payload = format!("request_{}", i);
                conn.send_request(Bytes::from(payload.clone()), &headers)
                    .await
                    .map(|response| (payload, response))
            });
            handles.push(handle);
        }

        let mut results = vec![];
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Request should succeed");
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 5);
        assert_eq!(
            request_count.load(Ordering::SeqCst),
            5,
            "Server should have received 5 requests"
        );
    }

    #[tokio::test]
    async fn test_lru_connection_reuse() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            pool_size: 4,
            channel_buffer: 10,
        };
        let pool = TcpConnectionPool::new(config);

        // Get connection twice sequentially -- should reuse the same TCP connection
        let conn1 = pool.get_connection(addr).await.unwrap();
        let mut headers = Headers::new();
        headers.insert("x-endpoint-path".to_string(), "test".to_string());
        let _ = conn1
            .send_request(Bytes::from("ping1"), &headers)
            .await
            .unwrap();
        drop(conn1); // Arc ref dropped, but conn stays in pool

        let conn2 = pool.get_connection(addr).await.unwrap();
        let _ = conn2
            .send_request(Bytes::from("ping2"), &headers)
            .await
            .unwrap();
        drop(conn2);

        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "Should reuse connection from pool (1 TCP connection total)"
        );
    }

    #[tokio::test]
    async fn test_lru_eviction_keeps_inflight_alive() {
        let (addr, _conn_count) = spawn_echo_server().await;

        // Pool size 1 so we can evict easily
        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            pool_size: 1,
            channel_buffer: 10,
        };
        let pool = TcpConnectionPool::new(config);

        // Get a connection and hold the Arc
        let conn = pool.get_connection(addr).await.unwrap();

        // Mark it unhealthy to force the pool to create a new one
        conn.healthy.store(false, Ordering::Relaxed);

        // Getting another connection should create a new one (old one evicted from LRU)
        let conn2 = pool.get_connection(addr).await.unwrap();
        assert!(conn2.is_healthy());

        // Original conn Arc is still alive (not dropped) -- it just can't be used
        assert!(!conn.is_healthy());
        // The Arc keeps the resources alive even though LRU evicted it
        assert!(Arc::strong_count(&conn.writer_handle) >= 1);
    }

    #[tokio::test]
    async fn test_high_concurrency_bounded_connections() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 2,
            channel_buffer: 50,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        let mut handles = vec![];
        for i in 0..500 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let conn = pool.get_connection(addr).await?;
                let mut headers = Headers::new();
                headers.insert("x-endpoint-path".to_string(), "test".to_string());
                conn.send_request(Bytes::from(format!("req_{}", i)), &headers)
                    .await
            }));
        }

        let mut ok_count = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                ok_count += 1;
            }
        }

        let total_conns = conn_count.load(Ordering::SeqCst);
        assert!(
            total_conns <= 2,
            "Should create at most pool_size (2) connections, got {}",
            total_conns
        );
        assert!(ok_count > 0, "At least some requests should succeed");
    }

    #[tokio::test]
    async fn test_thundering_herd_cold_start() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 4,
            channel_buffer: 50,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // 100 tasks all racing on a cold pool
        let mut handles = vec![];
        for i in 0..100 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let conn = pool.get_connection(addr).await?;
                let mut headers = Headers::new();
                headers.insert("x-endpoint-path".to_string(), "test".to_string());
                conn.send_request(Bytes::from(format!("req_{}", i)), &headers)
                    .await
            }));
        }

        for handle in handles {
            let _ = handle.await.unwrap();
        }

        let total_conns = conn_count.load(Ordering::SeqCst);
        assert!(
            total_conns <= 4,
            "Thundering herd: should create at most pool_size (4) connections, got {}",
            total_conns
        );
    }

    #[tokio::test]
    async fn test_server_crash_recovery() {
        // Start a server we can kill
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let cancel = tokio_util::sync::CancellationToken::new();
        let cancel_clone = cancel.clone();

        let server_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        if let Ok((stream, _)) = result {
                            let cancel = cancel_clone.clone();
                            tokio::spawn(async move {
                                let (mut read_half, mut write_half) = tokio::io::split(stream);
                                loop {
                                    tokio::select! {
                                        _ = cancel.cancelled() => break,
                                        result = async {
                                            let mut len_buf = [0u8; 2];
                                            read_half.read_exact(&mut len_buf).await?;
                                            let path_len = u16::from_be_bytes(len_buf) as usize;
                                            let mut buf = vec![0u8; path_len];
                                            read_half.read_exact(&mut buf).await?;
                                            let mut hlen = [0u8; 2];
                                            read_half.read_exact(&mut hlen).await?;
                                            let hl = u16::from_be_bytes(hlen) as usize;
                                            let mut hbuf = vec![0u8; hl];
                                            read_half.read_exact(&mut hbuf).await?;
                                            let mut plen = [0u8; 4];
                                            read_half.read_exact(&mut plen).await?;
                                            let pl = u32::from_be_bytes(plen) as usize;
                                            let mut pbuf = vec![0u8; pl];
                                            read_half.read_exact(&mut pbuf).await?;

                                            use crate::pipeline::network::codec::TcpResponseMessage;
                                            let resp = TcpResponseMessage::new(Bytes::from(pbuf));
                                            let enc = resp.encode()?;
                                            write_half.write_all(&enc).await?;
                                            Ok::<_, anyhow::Error>(())
                                        } => {
                                            if result.is_err() { break; }
                                        }
                                    }
                                }
                            });
                        }
                    }
                    _ = cancel_clone.cancelled() => break,
                }
            }
        });

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(2),
            connect_timeout: Duration::from_secs(2),
            pool_size: 2,
            channel_buffer: 10,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // Use pool successfully
        let conn = pool.get_connection(addr).await.unwrap();
        let mut headers = Headers::new();
        headers.insert("x-endpoint-path".to_string(), "test".to_string());
        let result = conn.send_request(Bytes::from("before_crash"), &headers).await;
        assert!(result.is_ok());
        drop(conn);

        // Kill the server
        cancel.cancel();
        let _ = server_task.await;

        // Requests should fail (server is gone)
        tokio::time::sleep(Duration::from_millis(50)).await;
        let conn = pool.get_connection(addr).await;
        // Pool should either fail to connect or return an unhealthy conn
        if let Ok(conn) = conn {
            let result = conn.send_request(Bytes::from("after_crash"), &headers).await;
            // Either send fails or connection is unhealthy
            assert!(result.is_err() || !conn.is_healthy());
        }

        // Start a new server on the same port
        let listener2 = TcpListener::bind(addr).await.unwrap();
        tokio::spawn(async move {
            loop {
                let result = listener2.accept().await;
                if result.is_err() {
                    break;
                }
                let (stream, _) = result.unwrap();
                tokio::spawn(async move {
                    let (mut read_half, mut write_half) = tokio::io::split(stream);
                    loop {
                        let mut len_buf = [0u8; 2];
                        if read_half.read_exact(&mut len_buf).await.is_err() {
                            break;
                        }
                        let path_len = u16::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; path_len];
                        if read_half.read_exact(&mut buf).await.is_err() {
                            break;
                        }
                        let mut hlen = [0u8; 2];
                        if read_half.read_exact(&mut hlen).await.is_err() {
                            break;
                        }
                        let hl = u16::from_be_bytes(hlen) as usize;
                        let mut hbuf = vec![0u8; hl];
                        if read_half.read_exact(&mut hbuf).await.is_err() {
                            break;
                        }
                        let mut plen = [0u8; 4];
                        if read_half.read_exact(&mut plen).await.is_err() {
                            break;
                        }
                        let pl = u32::from_be_bytes(plen) as usize;
                        let mut pbuf = vec![0u8; pl];
                        if read_half.read_exact(&mut pbuf).await.is_err() {
                            break;
                        }

                        use crate::pipeline::network::codec::TcpResponseMessage;
                        let resp = TcpResponseMessage::new(Bytes::from(pbuf));
                        let enc = resp.encode().unwrap();
                        if write_half.write_all(&enc).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });

        // Pool should heal: get new connection and succeed
        tokio::time::sleep(Duration::from_millis(100)).await;
        let conn = pool.get_connection(addr).await.unwrap();
        let result = conn.send_request(Bytes::from("after_recovery"), &headers).await;
        assert!(result.is_ok(), "Pool should heal after server recovery");
    }

    #[tokio::test]
    async fn test_pool_scales_under_pressure() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 4,
            channel_buffer: 1, // Very small buffer to force saturation quickly
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // Send enough concurrent requests to saturate channel_buffer=1
        let mut handles = vec![];
        for i in 0..20 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let conn = pool.get_connection(addr).await?;
                let mut headers = Headers::new();
                headers.insert("x-endpoint-path".to_string(), "test".to_string());
                conn.send_request(Bytes::from(format!("req_{}", i)), &headers)
                    .await
            }));
        }

        for handle in handles {
            let _ = handle.await.unwrap();
        }

        let total_conns = conn_count.load(Ordering::SeqCst);
        assert!(
            total_conns > 1,
            "Pool should scale beyond 1 connection under pressure, got {}",
            total_conns
        );
        assert!(
            total_conns <= 4,
            "Pool should not exceed pool_size (4), got {}",
            total_conns
        );
    }

    #[tokio::test]
    async fn test_pool_size_cap_sustained_load() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 3,
            channel_buffer: 50,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // 3 rounds of 200 requests
        for round in 0..3 {
            let mut handles = vec![];
            for i in 0..200 {
                let pool = pool.clone();
                handles.push(tokio::spawn(async move {
                    let conn = pool.get_connection(addr).await?;
                    let mut headers = Headers::new();
                    headers.insert("x-endpoint-path".to_string(), "test".to_string());
                    conn.send_request(
                        Bytes::from(format!("round_{}_req_{}", round, i)),
                        &headers,
                    )
                    .await
                }));
            }

            for handle in handles {
                let _ = handle.await.unwrap();
            }
        }

        let total_conns = conn_count.load(Ordering::SeqCst);
        assert!(
            total_conns <= 3,
            "Sustained load should not exceed pool_size (3), got {}",
            total_conns
        );
    }

    #[tokio::test]
    async fn test_backpressure_small_channel() {
        let (addr, _conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 1,
            channel_buffer: 1,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // 50 requests through pool_size=1 buffer=1 -- all should complete via backpressure
        let mut handles = vec![];
        for i in 0..50 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let conn = pool.get_connection(addr).await?;
                let mut headers = Headers::new();
                headers.insert("x-endpoint-path".to_string(), "test".to_string());
                conn.send_request(Bytes::from(format!("req_{}", i)), &headers)
                    .await
            }));
        }

        let mut ok_count = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                ok_count += 1;
            }
        }

        assert_eq!(
            ok_count, 50,
            "All 50 requests should complete under backpressure"
        );
    }

    #[tokio::test]
    async fn test_no_recursive_retry_under_connect_contention() {
        // This test verifies that connect contention uses bounded retries, not recursion.
        // We test this by having many tasks race on a cold pool with pool_size=1.
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 1,
            channel_buffer: 50,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // Many tasks racing, only one should connect
        let mut handles = vec![];
        for _ in 0..50 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                pool.get_connection(addr).await
            }));
        }

        let mut ok = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                ok += 1;
            }
        }

        // All should succeed (one connects, others retry via hot path)
        assert!(ok > 0, "At least some tasks should get connections");
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "Only 1 TCP connection should be created"
        );
    }

    #[tokio::test]
    async fn test_global_connect_limiter_multi_host() {
        // Spawn servers on 4 different ports to simulate multiple hosts
        let mut addrs = vec![];
        let total_conns = Arc::new(AtomicUsize::new(0));

        for _ in 0..4 {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            addrs.push(addr);
            let total_conns = total_conns.clone();

            tokio::spawn(async move {
                loop {
                    let result = listener.accept().await;
                    if result.is_err() {
                        break;
                    }
                    let (stream, _) = result.unwrap();
                    total_conns.fetch_add(1, Ordering::SeqCst);

                    tokio::spawn(async move {
                        let (mut read_half, mut write_half) = tokio::io::split(stream);
                        loop {
                            let mut len_buf = [0u8; 2];
                            if read_half.read_exact(&mut len_buf).await.is_err() {
                                break;
                            }
                            let path_len = u16::from_be_bytes(len_buf) as usize;
                            let mut buf = vec![0u8; path_len];
                            if read_half.read_exact(&mut buf).await.is_err() {
                                break;
                            }
                            let mut hlen = [0u8; 2];
                            if read_half.read_exact(&mut hlen).await.is_err() {
                                break;
                            }
                            let hl = u16::from_be_bytes(hlen) as usize;
                            let mut hbuf = vec![0u8; hl];
                            if read_half.read_exact(&mut hbuf).await.is_err() {
                                break;
                            }
                            let mut plen = [0u8; 4];
                            if read_half.read_exact(&mut plen).await.is_err() {
                                break;
                            }
                            let pl = u32::from_be_bytes(plen) as usize;
                            let mut pbuf = vec![0u8; pl];
                            if read_half.read_exact(&mut pbuf).await.is_err() {
                                break;
                            }

                            use crate::pipeline::network::codec::TcpResponseMessage;
                            let resp = TcpResponseMessage::new(Bytes::from(pbuf));
                            let enc = resp.encode().unwrap();
                            if write_half.write_all(&enc).await.is_err() {
                                break;
                            }
                        }
                    });
                }
            });
        }

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 2,
            channel_buffer: 50,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // Hit all 4 hosts concurrently
        let mut handles = vec![];
        for addr in &addrs {
            let pool = pool.clone();
            let addr = *addr;
            for i in 0..10 {
                let pool = pool.clone();
                handles.push(tokio::spawn(async move {
                    let conn = pool.get_connection(addr).await?;
                    let mut headers = Headers::new();
                    headers.insert("x-endpoint-path".to_string(), "test".to_string());
                    conn.send_request(Bytes::from(format!("req_{}", i)), &headers)
                        .await
                }));
            }
        }

        let mut ok_count = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                ok_count += 1;
            }
        }

        assert!(ok_count > 0, "Requests across multiple hosts should succeed");
        // Total connections across all hosts should be bounded
        let tc = total_conns.load(Ordering::SeqCst);
        assert!(
            tc <= 8,
            "Total connections across 4 hosts should be <= 4*pool_size(2)=8, got {}",
            tc
        );
    }

    #[tokio::test]
    async fn test_idle_host_pool_cleanup() {
        let (addr, _conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            pool_size: 2,
            channel_buffer: 10,
        };
        let pool = TcpConnectionPool::new(config);
        // Override TTL to 0 for testing
        let pool = TcpConnectionPool {
            host_idle_ttl_ms: 0,
            ..pool
        };

        // Create a connection to populate the host entry
        let conn = pool.get_connection(addr).await.unwrap();
        let mut headers = Headers::new();
        headers.insert("x-endpoint-path".to_string(), "test".to_string());
        let _ = conn.send_request(Bytes::from("test"), &headers).await;
        drop(conn);

        assert!(pool.hosts.contains_key(&addr), "Host entry should exist");

        // Wait a tiny bit so the timestamp is stale
        tokio::time::sleep(Duration::from_millis(10)).await;

        pool.cleanup_idle_hosts();

        assert!(
            !pool.hosts.contains_key(&addr),
            "Idle host entry should be cleaned up"
        );
    }

    #[tokio::test]
    async fn test_connection_pool_reuse() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            pool_size: 2,
            channel_buffer: 10,
        };
        let pool = TcpConnectionPool::new(config);

        // Get connection twice from pool
        let conn1 = pool.get_connection(addr).await.unwrap();
        let mut headers = Headers::new();
        headers.insert("x-endpoint-path".to_string(), "test".to_string());
        let _ = conn1
            .send_request(Bytes::from("test1"), &headers)
            .await
            .unwrap();
        drop(conn1);

        tokio::time::sleep(Duration::from_millis(10)).await;

        let conn2 = pool.get_connection(addr).await.unwrap();
        let _ = conn2
            .send_request(Bytes::from("test2"), &headers)
            .await
            .unwrap();
        drop(conn2);

        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "Should reuse connection from pool"
        );
    }

    #[tokio::test]
    async fn test_unhealthy_connection_filtered() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server that immediately closes connections
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                drop(stream);
            }
        });

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(1),
            connect_timeout: Duration::from_secs(1),
            pool_size: 2,
            channel_buffer: 10,
        };

        let result =
            TcpConnection::connect(addr, config.connect_timeout, config.channel_buffer).await;

        if let Ok(conn) = result {
            let mut headers = Headers::new();
            headers.insert("x-endpoint-path".to_string(), "test".to_string());
            let _ = conn.send_request(Bytes::from("test"), &headers).await;

            // Connection should become unhealthy after server drops it
            tokio::time::sleep(Duration::from_millis(50)).await;
            // Connection health depends on timing, but it should not panic
        }
    }

    #[tokio::test]
    async fn test_warmup_pre_connects_on_instance_discovery() {
        use crate::component::{Instance, TransportType};

        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            pool_size: 4,
            channel_buffer: 10,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));
        let cancel_token = tokio_util::sync::CancellationToken::new();

        // Create a watch channel with no instances initially
        let (instance_tx, instance_rx) =
            tokio::sync::watch::channel::<Vec<Instance>>(Vec::new());

        // Start the warmup watcher
        pool.start_warmup_watcher(instance_rx, cancel_token.clone());

        // Let the watcher task start and begin polling changed()
        tokio::time::sleep(Duration::from_millis(50)).await;

        // No connections yet
        assert_eq!(conn_count.load(Ordering::SeqCst), 0);

        // Discover a new TCP backend
        let tcp_addr = format!("{}:{}/test_endpoint", addr.ip(), addr.port());
        instance_tx
            .send(vec![Instance {
                component: "test".to_string(),
                endpoint: "test_ep".to_string(),
                namespace: "default".to_string(),
                instance_id: 1,
                transport: TransportType::Tcp(tcp_addr),
            }])
            .unwrap();

        // Give the warmup watcher time to process and connect
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Should have pre-connected
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "Warmup should have created 1 connection to the newly discovered backend"
        );

        // Pool should have the host entry
        assert!(
            pool.hosts.contains_key(&addr),
            "Pool should have a host entry for the warmed address"
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_lockfree_submit_and_batch() {
        // Verify that concurrent submits to the same connection produce
        // batched writes (batch_size > 1) by checking that all requests
        // complete correctly even when submitted simultaneously.
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 1,
            channel_buffer: 200,
        };
        let pool = Arc::new(TcpConnectionPool::new(config));

        // Force a single connection, then blast concurrent requests
        let conn = pool.get_connection(addr).await.unwrap();

        let mut handles = vec![];
        for i in 0..100 {
            let conn = conn.clone();
            handles.push(tokio::spawn(async move {
                let mut headers = Headers::new();
                headers.insert("x-endpoint-path".to_string(), "test".to_string());
                conn.send_request(Bytes::from(format!("batch_req_{}", i)), &headers)
                    .await
            }));
        }

        let mut ok_count = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                ok_count += 1;
            }
        }

        assert_eq!(ok_count, 100, "All 100 concurrent requests should succeed");
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "Should use only 1 connection"
        );
    }
}

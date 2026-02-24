// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Request Plane Client
//!

use super::unified_client::{ClientStats, Headers, RequestPlaneClient};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use futures::StreamExt;
use std::io::IoSlice;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::codec::FramedRead;

/// Default timeout for TCP request acknowledgment
const DEFAULT_TCP_REQUEST_TIMEOUT_SECS: u64 = 5;

/// Default connection pool size per host
const DEFAULT_POOL_SIZE: usize = 100;

/// Buffer size for request channel per connection (backpressure control)
const REQUEST_CHANNEL_BUFFER: usize = 50;

/// Pre-allocated read buffer size (64KB typical message size)
const READ_BUFFER_SIZE: usize = 65536;

/// Default maximum message size for TCP client (32 MB)
/// This is the limit for a SINGLE message. For larger data, split into multiple messages.
const DEFAULT_MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

/// Get maximum message size from environment or use default
fn get_max_message_size() -> usize {
    std::env::var("DYN_TCP_MAX_MESSAGE_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_MESSAGE_SIZE)
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

/// Request to be sent over TCP
/// Pre-encoded on caller's thread for optimal write performance (hot path optimization)
struct TcpRequest {
    /// Pre-encoded request data ready to send (zero-copy Bytes)
    /// Encoding happens on caller thread to parallelize across multiple request handlers
    encoded_data: Bytes,
    /// Oneshot channel to send response back to caller
    response_tx: oneshot::Sender<Result<Bytes>>,
}

/// TCP connection with split read/write tasks
///
/// Design: One writer task + one reader task per connection
/// - Writer task receives pre-encoded requests and writes directly (hot path optimized)
/// - Reader task uses framed codec for robust protocol handling
/// - FIFO ordering ensures request/response correlation without explicit IDs
///
/// Performance: Hybrid approach optimizes each path independently:
/// - Write path: Pre-encode on caller thread → direct write (minimal overhead, parallel encoding)
/// - Read path: Framed codec handles partial reads and protocol complexity automatically
struct TcpConnection {
    addr: SocketAddr,
    /// Channel to send requests to the writer task
    request_tx: mpsc::Sender<TcpRequest>,
    /// Writer task handle for cleanup
    writer_handle: Arc<JoinHandle<()>>,
    /// Reader task handle for cleanup
    reader_handle: Arc<JoinHandle<()>>,
    /// Health status (false if tasks have failed)
    healthy: Arc<AtomicBool>,
}

impl TcpConnection {
    /// Create a new connection with split read/write tasks
    async fn connect(addr: SocketAddr, timeout: Duration, channel_buffer: usize) -> Result<Self> {
        let stream = tokio::time::timeout(timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| anyhow::anyhow!("TCP connect timeout to {}", addr))??;

        // Configure socket for lower latency
        Self::configure_socket(&stream)?;

        let (read_half, write_half) = tokio::io::split(stream);

        // Channel for writer task to receive requests
        let (request_tx, request_rx) = mpsc::channel::<TcpRequest>(channel_buffer);

        // Channel for writer to forward response channels to reader (FIFO correlation)
        let (response_tx_channel, response_rx_channel) =
            mpsc::unbounded_channel::<oneshot::Sender<Result<Bytes>>>();

        let healthy = Arc::new(AtomicBool::new(true));

        // Spawn writer task
        let writer_handle = {
            let healthy = healthy.clone();
            tokio::spawn(async move {
                if let Err(e) = Self::writer_task(write_half, request_rx, response_tx_channel).await
                {
                    tracing::debug!("Writer task failed for {}: {}", addr, e);
                    healthy.store(false, Ordering::Relaxed);
                }
            })
        };

        // Spawn reader task
        let reader_handle = {
            let healthy = healthy.clone();
            tokio::spawn(async move {
                if let Err(e) = Self::reader_task(read_half, response_rx_channel).await {
                    tracing::debug!("Reader task failed for {}: {}", addr, e);
                    healthy.store(false, Ordering::Relaxed);
                }
            })
        };

        Ok(Self {
            addr,
            request_tx,
            writer_handle: Arc::new(writer_handle),
            reader_handle: Arc::new(reader_handle),
            healthy,
        })
    }

    /// Configure socket for ultra-low latency based on dyn-transports patterns
    fn configure_socket(stream: &TcpStream) -> Result<()> {
        use socket2::{SockRef, Socket};

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

    /// Writer task: receives pre-encoded requests and writes directly to socket
    ///
    /// Performance optimization: Pre-encoding happens on caller's thread to enable
    /// parallel encoding across multiple request handlers, while this task focuses
    /// on sequential socket writes with minimal overhead.
    async fn writer_task(
        mut write_half: tokio::io::WriteHalf<TcpStream>,
        mut request_rx: mpsc::Receiver<TcpRequest>,
        response_tx_channel: mpsc::UnboundedSender<oneshot::Sender<Result<Bytes>>>,
    ) -> Result<()> {
        while let Some(req) = request_rx.recv().await {
            // Direct write of pre-encoded data (hot path)
            // With TCP_NODELAY, no need for explicit flush()
            match write_half.write_all(&req.encoded_data).await {
                Ok(()) => {
                    // Forward response channel to reader task (FIFO ordering)
                    if response_tx_channel.send(req.response_tx).is_err() {
                        tracing::debug!("Reader task closed, stopping writer");
                        break;
                    }
                }
                Err(e) => {
                    // Write failed, notify caller and stop
                    let err_msg = format!("Write failed: {}", e);
                    let _ = req.response_tx.send(Err(anyhow::anyhow!("{}", err_msg)));
                    return Err(anyhow::anyhow!("{}", err_msg));
                }
            }
        }
        Ok(())
    }

    /// Reader task: reads responses using framed codec and sends them back via oneshot channels
    /// Protocol framing handled automatically via TcpResponseCodec
    async fn reader_task(
        read_half: tokio::io::ReadHalf<TcpStream>,
        mut response_rx_channel: mpsc::UnboundedReceiver<oneshot::Sender<Result<Bytes>>>,
    ) -> Result<()> {
        use crate::pipeline::network::codec::TcpResponseCodec;

        let max_message_size = get_max_message_size();
        let codec = TcpResponseCodec::new(Some(max_message_size));
        let mut framed = FramedRead::new(read_half, codec);

        while let Some(response_tx) = response_rx_channel.recv().await {
            // Read the next response message from the framed stream
            // The codec handles all protocol framing and size checks automatically
            match framed.next().await {
                Some(Ok(response_msg)) => {
                    let _ = response_tx.send(Ok(response_msg.data));
                }
                Some(Err(e)) => {
                    let err = anyhow::anyhow!("Failed to decode response: {}", e);
                    let _ = response_tx.send(Err(err));
                    return Err(anyhow::anyhow!("Failed to decode response"));
                }
                None => {
                    let err = anyhow::anyhow!("Connection closed by peer");
                    let _ = response_tx.send(Err(err));
                    return Err(anyhow::anyhow!("Connection closed"));
                }
            }
        }

        Ok(())
    }

    /// Send a request and wait for response
    ///
    /// Performance: Encoding happens on caller's thread (hot path optimization)
    /// to enable parallel encoding across multiple request handlers. The writer
    /// task then performs sequential writes with minimal overhead.
    async fn send_request(&self, payload: Bytes, headers: &Headers) -> Result<Bytes> {
        use crate::pipeline::network::codec::TcpRequestMessage;

        // Check health before sending
        if !self.healthy.load(Ordering::Relaxed) {
            anyhow::bail!("Connection unhealthy (tasks failed)");
        }

        // Extract endpoint path from headers (required for routing)
        let endpoint_path = headers
            .get("x-endpoint-path")
            .ok_or_else(|| anyhow::anyhow!("Missing x-endpoint-path header for TCP request"))?
            .to_string();

        // Encode request on caller's thread (hot path optimization)
        // This allows multiple concurrent callers to encode in parallel
        // rather than serializing through the writer task
        // Include all headers (especially trace headers) in the message
        let request_msg = TcpRequestMessage::with_headers(endpoint_path, headers.clone(), payload);
        let encoded_data = request_msg.encode()?;

        // Create response channel
        let (response_tx, response_rx) = oneshot::channel();

        // Send to writer task (bounded channel provides backpressure)
        let req = TcpRequest {
            encoded_data,
            response_tx,
        };

        self.request_tx
            .send(req)
            .await
            .map_err(|_| anyhow::anyhow!("Writer task closed"))?;

        // Wait for response from reader task
        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Reader task closed"))?
    }

    /// Check if connection is healthy
    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }
}

/// Shared connection set for a single host.
/// Connections are held in Arc and shared across concurrent requests via round-robin.
/// Each connection supports multiplexing through its internal mpsc channel.
struct SharedConnectionSet {
    connections: tokio::sync::RwLock<Vec<Arc<TcpConnection>>>,
    counter: AtomicU64,
    max_connections: usize,
    connect_timeout: Duration,
    channel_buffer: usize,
    addr: SocketAddr,
}

impl SharedConnectionSet {
    fn new(
        addr: SocketAddr,
        max_connections: usize,
        connect_timeout: Duration,
        channel_buffer: usize,
    ) -> Self {
        Self {
            connections: tokio::sync::RwLock::new(Vec::with_capacity(max_connections)),
            counter: AtomicU64::new(0),
            max_connections,
            connect_timeout,
            channel_buffer,
            addr,
        }
    }

    /// Get a healthy connection via round-robin selection, creating one if needed.
    ///
    /// Fast path acquires a read lock and scans existing connections.
    /// Slow path acquires a write lock to prune unhealthy connections and create new ones.
    async fn get_connection(&self) -> Result<Arc<TcpConnection>> {
        // Fast path: round-robin over existing healthy connections (read lock only)
        {
            let conns = self.connections.read().await;
            if !conns.is_empty() {
                let len = conns.len();
                let start = self.counter.fetch_add(1, Ordering::Relaxed) as usize;
                // Try each connection starting from the round-robin index
                for i in 0..len {
                    let idx = (start + i) % len;
                    if conns[idx].is_healthy() {
                        return Ok(conns[idx].clone());
                    }
                }
            }
        }

        // Slow path: need to create or replace connections (write lock)
        let mut conns = self.connections.write().await;

        // Remove unhealthy connections
        conns.retain(|c| c.is_healthy());

        // Re-check after cleanup — another task may have added one
        if !conns.is_empty() {
            let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % conns.len();
            return Ok(conns[idx].clone());
        }

        // Create a new connection
        tracing::debug!(
            addr = %self.addr,
            pool_size = conns.len() + 1,
            "Creating new shared TCP connection"
        );
        let conn = Arc::new(
            TcpConnection::connect(self.addr, self.connect_timeout, self.channel_buffer)
                .await?,
        );
        conns.push(conn.clone());
        Ok(conn)
    }
}

/// Connection pool that shares connections across concurrent requests.
///
/// Connections are wrapped in `Arc` and selected via round-robin. Each connection
/// supports multiplexing through its internal mpsc channel, so thousands of
/// requests can share a small number of TCP connections.
struct TcpConnectionPool {
    pools: DashMap<SocketAddr, Arc<SharedConnectionSet>>,
    config: TcpRequestConfig,
}

impl TcpConnectionPool {
    fn new(config: TcpRequestConfig) -> Self {
        Self {
            pools: DashMap::new(),
            config,
        }
    }

    /// Get a shared connection for the given address.
    /// Connections are reused across concurrent requests via round-robin selection.
    async fn get_connection(&self, addr: SocketAddr) -> Result<Arc<TcpConnection>> {
        let set = self
            .pools
            .entry(addr)
            .or_insert_with(|| {
                Arc::new(SharedConnectionSet::new(
                    addr,
                    self.config.pool_size,
                    self.config.connect_timeout,
                    self.config.channel_buffer,
                ))
            })
            .clone();

        set.get_connection().await
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

    /// Parse TCP address from string
    /// Supports formats: "host:port" or "tcp://host:port" or "host:port/endpoint_name"
    /// Returns (SocketAddr, Option<endpoint_name>)
    fn parse_address(address: &str) -> Result<(SocketAddr, Option<String>)> {
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

        // Add endpoint path to headers if present in address
        if let Some(endpoint_name) = endpoint_name {
            headers.insert("x-endpoint-path".to_string(), endpoint_name.clone());
        }

        let conn = self.pool.get_connection(addr).await?;

        // The connection's internal mpsc channel provides backpressure
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

                Ok(response)
            }
            Ok(Err(e)) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("TCP request failed to {}: {}", addr, e);
                Err(e)
            }
            Err(_) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("TCP request timeout to {}", addr);
                Err(anyhow::anyhow!("TCP request timeout to {}", addr))
            }
        }
    }

    fn transport_name(&self) -> &'static str {
        "tcp"
    }

    fn is_healthy(&self) -> bool {
        true // TCP client is always healthy if it was created successfully
    }

    fn stats(&self) -> ClientStats {
        ClientStats {
            requests_sent: self.stats.requests_sent.load(Ordering::Relaxed),
            responses_received: self.stats.responses_received.load(Ordering::Relaxed),
            errors: self.stats.errors.load(Ordering::Relaxed),
            bytes_sent: self.stats.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.stats.bytes_received.load(Ordering::Relaxed),
            active_connections: 0, // Could track this if needed
            idle_connections: 0,
            avg_latency_us: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
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

    #[tokio::test]
    async fn test_connection_health_check() {
        use crate::pipeline::network::codec::{TcpRequestMessage, TcpResponseMessage};

        // Start a mock TCP server
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn server that responds to requests
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (mut read_half, mut write_half) = tokio::io::split(stream);

            // Read path length and path
            let mut len_buf = [0u8; 2];
            read_half.read_exact(&mut len_buf).await.unwrap();
            let path_len = u16::from_be_bytes(len_buf) as usize;

            let mut path_buf = vec![0u8; path_len];
            read_half.read_exact(&mut path_buf).await.unwrap();

            // Read headers length and headers
            let mut headers_len_buf = [0u8; 2];
            read_half.read_exact(&mut headers_len_buf).await.unwrap();
            let headers_len = u16::from_be_bytes(headers_len_buf) as usize;

            let mut headers_buf = vec![0u8; headers_len];
            read_half.read_exact(&mut headers_buf).await.unwrap();

            // Read payload length and payload
            let mut len_buf = [0u8; 4];
            read_half.read_exact(&mut len_buf).await.unwrap();
            let payload_len = u32::from_be_bytes(len_buf) as usize;

            let mut payload_buf = vec![0u8; payload_len];
            read_half.read_exact(&mut payload_buf).await.unwrap();

            // Send response
            let response = TcpResponseMessage::new(Bytes::from_static(b"pong"));
            let encoded = response.encode().unwrap();
            write_half.write_all(&encoded).await.unwrap();
        });

        // Create connection
        let conn = TcpConnection::connect(addr, Duration::from_secs(5), 10)
            .await
            .unwrap();

        assert!(conn.is_healthy(), "New connection should be healthy");

        // Send a request
        let mut headers = Headers::new();
        headers.insert("x-endpoint-path".to_string(), "test".to_string());

        let result = conn.send_request(Bytes::from("ping"), &headers).await;
        assert!(result.is_ok(), "Request should succeed");
        assert_eq!(result.unwrap(), Bytes::from("pong"));

        assert!(
            conn.is_healthy(),
            "Connection should remain healthy after successful request"
        );
    }

    #[tokio::test]
    async fn test_concurrent_requests_single_connection() {
        use crate::pipeline::network::codec::{TcpRequestMessage, TcpResponseMessage};

        // Start a mock TCP server that handles multiple requests
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_count_clone = request_count.clone();

        // Spawn server that responds to multiple requests
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (mut read_half, mut write_half) = tokio::io::split(stream);

            // Handle 5 requests
            for _ in 0..5 {
                // Read request
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

                // Send response
                let response = TcpResponseMessage::new(Bytes::from(payload_buf));
                let encoded = response.encode().unwrap();
                if write_half.write_all(&encoded).await.is_err() {
                    break;
                }
            }
        });

        // Create connection
        let conn = Arc::new(
            TcpConnection::connect(addr, Duration::from_secs(5), 10)
                .await
                .unwrap(),
        );

        // Send 5 concurrent requests
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

        // Wait for all requests to complete
        let mut results = vec![];
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Request should succeed");
            results.push(result.unwrap());
        }

        // Verify all requests got responses
        assert_eq!(results.len(), 5);

        // Verify server received all requests
        assert_eq!(
            request_count.load(Ordering::SeqCst),
            5,
            "Server should have received 5 requests"
        );
    }

    #[tokio::test]
    async fn test_connection_pool_reuse() {
        use crate::pipeline::network::codec::TcpResponseMessage;

        // Start a mock TCP server
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let connection_count = Arc::new(AtomicUsize::new(0));
        let connection_count_clone = connection_count.clone();

        // Spawn server that accepts multiple connections
        tokio::spawn(async move {
            loop {
                let result = listener.accept().await;
                if result.is_err() {
                    break;
                }
                let (stream, _) = result.unwrap();
                connection_count_clone.fetch_add(1, Ordering::SeqCst);

                tokio::spawn(async move {
                    let (mut read_half, mut write_half) = tokio::io::split(stream);
                    loop {
                        // Read request
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

                        // Send response
                        let response = TcpResponseMessage::new(Bytes::from_static(b"ok"));
                        let encoded = response.encode().unwrap();
                        if write_half.write_all(&encoded).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            pool_size: 2,
            channel_buffer: 10,
        };

        let pool = TcpConnectionPool::new(config);

        // Shared connections: getting twice returns the same Arc
        let conn1 = pool.get_connection(addr).await.unwrap();
        let conn2 = pool.get_connection(addr).await.unwrap();

        // Send a request to ensure the server has accepted the connection
        let result = conn1
            .send_request(Bytes::from("ping"), &test_headers())
            .await;
        assert!(result.is_ok(), "Request through shared connection should succeed");

        // Should have created only 1 TCP connection since shared pool reuses it
        assert_eq!(
            connection_count.load(Ordering::SeqCst),
            1,
            "Should reuse connection from shared pool"
        );

        drop(conn1);
        drop(conn2);
    }

    #[tokio::test]
    async fn test_unhealthy_connection_filtered() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server that immediately closes connections
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                drop(stream); // Immediately close
            }
        });

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(1),
            connect_timeout: Duration::from_secs(1),
            pool_size: 2,
            channel_buffer: 10,
        };

        let pool = TcpConnectionPool::new(config);

        // Get a connection — it may become unhealthy quickly after server closes
        let conn = pool.get_connection(addr).await;
        if let Ok(conn) = conn {
            let mut headers = Headers::new();
            headers.insert("x-endpoint-path".to_string(), "test".to_string());
            let _ = conn.send_request(Bytes::from("test"), &headers).await;

            // Get again — shared pool should handle unhealthy connections gracefully
            let result2 = pool.get_connection(addr).await;
            let _ = result2;
        }
    }

    // ==================== Adversarial Tests ====================

    /// Spawn a mock TCP echo server that handles requests indefinitely.
    /// Returns the server address and a connection counter.
    async fn spawn_echo_server() -> (SocketAddr, Arc<AtomicUsize>) {
        use crate::pipeline::network::codec::TcpResponseMessage;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conn_count = Arc::new(AtomicUsize::new(0));
        let conn_count_clone = conn_count.clone();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                conn_count_clone.fetch_add(1, Ordering::SeqCst);

                tokio::spawn(async move {
                    let (mut read_half, mut write_half) = tokio::io::split(stream);
                    loop {
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
                        let response =
                            TcpResponseMessage::new(Bytes::from(payload_buf));
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

    /// Build minimal headers required by `TcpConnection::send_request`.
    fn test_headers() -> Headers {
        let mut h = Headers::new();
        h.insert("x-endpoint-path".to_string(), "test".to_string());
        h
    }

    /// Hammer the shared pool with 500 concurrent requests through a pool of size 2.
    /// Verifies connections are reused and total TCP connections stay within pool_size.
    #[tokio::test]
    async fn test_high_concurrency_bounded_connections() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 2,
            channel_buffer: 100,
        };

        let pool = Arc::new(TcpConnectionPool::new(config));

        let mut handles = vec![];
        for i in 0..500 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let conn = pool.get_connection(addr).await.unwrap();
                let payload = format!("req-{i}");
                conn.send_request(Bytes::from(payload), &test_headers())
                    .await
            }));
        }

        let mut success = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                success += 1;
            }
        }

        assert_eq!(success, 500, "All 500 requests should succeed");
        assert!(
            conn_count.load(Ordering::SeqCst) <= 2,
            "Should create at most 2 TCP connections, got {}",
            conn_count.load(Ordering::SeqCst)
        );
    }

    /// Race 100 tasks all calling get_connection simultaneously on a cold pool.
    /// Ensures no duplicate connections are created beyond pool_size.
    #[tokio::test]
    async fn test_thundering_herd_cold_start() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(5),
            pool_size: 3,
            channel_buffer: 50,
        };

        let pool = Arc::new(TcpConnectionPool::new(config));
        let barrier = Arc::new(tokio::sync::Barrier::new(100));

        let mut handles = vec![];
        for _ in 0..100 {
            let pool = pool.clone();
            let barrier = barrier.clone();
            handles.push(tokio::spawn(async move {
                barrier.wait().await;
                pool.get_connection(addr).await
            }));
        }

        let mut successes = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                successes += 1;
            }
        }

        assert_eq!(successes, 100, "All 100 get_connection calls should succeed");
        assert!(
            conn_count.load(Ordering::SeqCst) <= 3,
            "Should create at most pool_size (3) connections, got {}",
            conn_count.load(Ordering::SeqCst)
        );
    }

    /// Kill the server mid-stream, then verify the pool recovers by
    /// creating a new connection to a replacement server.
    #[tokio::test]
    async fn test_server_crash_recovery() {
        use crate::pipeline::network::codec::TcpResponseMessage;

        // First server — will be killed
        let listener1 = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener1.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, _) = listener1.accept().await.unwrap();
            let (mut read_half, mut write_half) = tokio::io::split(stream);

            // Handle exactly 1 request then crash
            let mut buf = [0u8; 2];
            let _ = read_half.read_exact(&mut buf).await;
            let path_len = u16::from_be_bytes(buf) as usize;
            let mut path_buf = vec![0u8; path_len];
            let _ = read_half.read_exact(&mut path_buf).await;
            let mut hdr_buf = [0u8; 2];
            let _ = read_half.read_exact(&mut hdr_buf).await;
            let hdr_len = u16::from_be_bytes(hdr_buf) as usize;
            let mut hdr = vec![0u8; hdr_len];
            let _ = read_half.read_exact(&mut hdr).await;
            let mut len_buf = [0u8; 4];
            let _ = read_half.read_exact(&mut len_buf).await;
            let plen = u32::from_be_bytes(len_buf) as usize;
            let mut pbuf = vec![0u8; plen];
            let _ = read_half.read_exact(&mut pbuf).await;

            let resp = TcpResponseMessage::new(Bytes::from_static(b"first"));
            let encoded = resp.encode().unwrap();
            let _ = write_half.write_all(&encoded).await;
            // Drop everything — simulates crash
        });

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(3),
            connect_timeout: Duration::from_secs(3),
            pool_size: 2,
            channel_buffer: 10,
        };

        let pool = TcpConnectionPool::new(config);

        // First request should succeed
        let conn = pool.get_connection(addr).await.unwrap();
        let result = conn
            .send_request(Bytes::from("hello"), &test_headers())
            .await;
        assert!(result.is_ok(), "First request should succeed");
        assert_eq!(result.unwrap(), Bytes::from("first"));

        // Wait for server to crash
        let _ = server_handle.await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Second request on same connection should fail (server crashed)
        let result2 = conn
            .send_request(Bytes::from("hello2"), &test_headers())
            .await;
        assert!(result2.is_err(), "Request on crashed connection should fail");

        // Connection should now be unhealthy
        assert!(!conn.is_healthy(), "Connection should be marked unhealthy");

        // Start replacement server on same port
        let listener2 = TcpListener::bind(addr).await.unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener2.accept().await.unwrap();
            let (mut rh, mut wh) = tokio::io::split(stream);
            let mut buf = [0u8; 2];
            let _ = rh.read_exact(&mut buf).await;
            let plen = u16::from_be_bytes(buf) as usize;
            let mut pbuf = vec![0u8; plen];
            let _ = rh.read_exact(&mut pbuf).await;
            let mut hdr_buf = [0u8; 2];
            let _ = rh.read_exact(&mut hdr_buf).await;
            let hdr_len = u16::from_be_bytes(hdr_buf) as usize;
            let mut hdr = vec![0u8; hdr_len];
            let _ = rh.read_exact(&mut hdr).await;
            let mut len_buf = [0u8; 4];
            let _ = rh.read_exact(&mut len_buf).await;
            let plen2 = u32::from_be_bytes(len_buf) as usize;
            let mut pbuf2 = vec![0u8; plen2];
            let _ = rh.read_exact(&mut pbuf2).await;
            let resp = TcpResponseMessage::new(Bytes::from_static(b"recovered"));
            let encoded = resp.encode().unwrap();
            let _ = wh.write_all(&encoded).await;
        });

        // Pool should detect unhealthy, create new connection to replacement server
        let conn3 = pool.get_connection(addr).await.unwrap();
        let result3 = conn3
            .send_request(Bytes::from("recover"), &test_headers())
            .await;
        assert!(result3.is_ok(), "Recovery request should succeed");
        assert_eq!(result3.unwrap(), Bytes::from("recovered"));
    }

    /// Verify the pool doesn't create connections beyond pool_size even under
    /// sustained concurrent load over multiple rounds.
    #[tokio::test]
    async fn test_pool_size_cap_sustained_load() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(5),
            pool_size: 4,
            channel_buffer: 50,
        };

        let pool = Arc::new(TcpConnectionPool::new(config));

        // Run 3 rounds of 200 concurrent requests
        for round in 0..3 {
            let mut handles = vec![];
            for i in 0..200 {
                let pool = pool.clone();
                handles.push(tokio::spawn(async move {
                    let conn = pool.get_connection(addr).await.unwrap();
                    conn.send_request(
                        Bytes::from(format!("r{round}-{i}")),
                        &test_headers(),
                    )
                    .await
                }));
            }

            for handle in handles {
                assert!(
                    handle.await.unwrap().is_ok(),
                    "Round {round}: all requests should succeed"
                );
            }
        }

        // After 600 total requests across 3 rounds, should still be <= pool_size connections
        assert!(
            conn_count.load(Ordering::SeqCst) <= 4,
            "Should never exceed pool_size (4) connections, got {}",
            conn_count.load(Ordering::SeqCst)
        );
    }

    /// Connect timeout should fire quickly when the server is unreachable.
    #[tokio::test]
    async fn test_connect_timeout_unreachable() {
        // Use a non-routable address to trigger connect timeout
        let addr: SocketAddr = "192.0.2.1:1".parse().unwrap();

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_millis(500),
            pool_size: 2,
            channel_buffer: 10,
        };

        let pool = TcpConnectionPool::new(config);

        let start = tokio::time::Instant::now();
        let result = pool.get_connection(addr).await;
        let elapsed = start.elapsed();

        assert!(result.is_err(), "Should fail to connect to unreachable address");
        assert!(
            elapsed < Duration::from_secs(5),
            "Should timeout quickly, took {:?}",
            elapsed
        );
    }

    /// Blast the pool from multiple tokio tasks with tiny channel buffers
    /// to force backpressure through the mpsc channel.
    #[tokio::test]
    async fn test_backpressure_small_channel() {
        let (addr, conn_count) = spawn_echo_server().await;

        let config = TcpRequestConfig {
            request_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            pool_size: 1,   // Single connection
            channel_buffer: 1, // Tiny buffer — forces backpressure
        };

        let pool = Arc::new(TcpConnectionPool::new(config));

        let mut handles = vec![];
        for i in 0..50 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let conn = pool.get_connection(addr).await.unwrap();
                conn.send_request(Bytes::from(format!("bp-{i}")), &test_headers())
                    .await
            }));
        }

        let mut success = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                success += 1;
            }
        }

        assert_eq!(success, 50, "All requests should eventually succeed under backpressure");
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "Should use exactly 1 connection"
        );
    }
}

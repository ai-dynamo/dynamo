// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Request Plane Client
//!
//! Implements a connection-pooled TCP client for sending requests over raw TCP.

use super::unified_client::{ClientStats, Headers, RequestPlaneClient};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

/// Default timeout for TCP request acknowledgment
const DEFAULT_TCP_REQUEST_TIMEOUT_SECS: u64 = 5;

/// Default connection pool size per host
const DEFAULT_POOL_SIZE: usize = 100;

/// TCP request plane configuration
#[derive(Debug, Clone)]
pub struct TcpRequestConfig {
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum connections per host
    pub pool_size: usize,
    /// Connect timeout
    pub connect_timeout: Duration,
}

impl Default for TcpRequestConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(DEFAULT_TCP_REQUEST_TIMEOUT_SECS),
            pool_size: DEFAULT_POOL_SIZE,
            connect_timeout: Duration::from_secs(5),
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

        config
    }
}

/// TCP connection wrapper
struct TcpConnection {
    stream: TcpStream,
    addr: SocketAddr,
}

impl TcpConnection {
    async fn connect(addr: SocketAddr, timeout: Duration) -> Result<Self> {
        let stream = tokio::time::timeout(timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| anyhow::anyhow!("TCP connect timeout to {}", addr))??;

        stream.set_nodelay(true)?;

        Ok(Self { stream, addr })
    }

    async fn send_request(&mut self, payload: Bytes, headers: &Headers) -> Result<Bytes> {
        // Extract endpoint path from headers (required for routing)
        let endpoint_path = headers
            .get("x-endpoint-path")
            .ok_or_else(|| anyhow::anyhow!("Missing x-endpoint-path header for TCP request"))?;

        let endpoint_bytes = endpoint_path.as_bytes();
        let endpoint_len = endpoint_bytes.len() as u16;

        // Write endpoint path length (2 bytes)
        self.stream.write_all(&endpoint_len.to_be_bytes()).await?;

        // Write endpoint path
        self.stream.write_all(endpoint_bytes).await?;

        // Write payload length (4 bytes)
        let len = payload.len() as u32;
        self.stream.write_all(&len.to_be_bytes()).await?;

        // Write payload
        self.stream.write_all(&payload).await?;
        self.stream.flush().await?;

        // Read acknowledgment length
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await?;
        let ack_len = u32::from_be_bytes(len_buf) as usize;

        // Sanity check
        if ack_len > 16 * 1024 * 1024 {
            anyhow::bail!("Acknowledgment too large: {} bytes", ack_len);
        }

        // Read acknowledgment
        let mut ack_buf = vec![0u8; ack_len];
        self.stream.read_exact(&mut ack_buf).await?;

        Ok(Bytes::from(ack_buf))
    }
}

/// Simple connection pool for TCP connections
struct TcpConnectionPool {
    pools: DashMap<SocketAddr, Arc<Mutex<Vec<TcpConnection>>>>,
    config: TcpRequestConfig,
}

impl TcpConnectionPool {
    fn new(config: TcpRequestConfig) -> Self {
        Self {
            pools: DashMap::new(),
            config,
        }
    }

    async fn get_connection(&self, addr: SocketAddr) -> Result<TcpConnection> {
        // Try to get from pool (lock-free read with DashMap)
        if let Some(pool) = self.pools.get(&addr) {
            let mut pool = pool.lock().await;
            if let Some(conn) = pool.pop() {
                return Ok(conn);
            }
        }

        // Create new connection
        TcpConnection::connect(addr, self.config.connect_timeout).await
    }

    async fn return_connection(&self, conn: TcpConnection) {
        let addr = conn.addr;

        // Get or create pool for this address (lock-free with DashMap)
        let pool = self
            .pools
            .entry(addr)
            .or_insert_with(|| Arc::new(Mutex::new(Vec::new())))
            .clone();

        let mut pool = pool.lock().await;
        if pool.len() < self.config.pool_size {
            pool.push(conn);
        }
        // Otherwise drop the connection
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
        tracing::info!("TCP client sending request to address: {}", address);
        self.stats.requests_sent.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_sent
            .fetch_add(payload.len() as u64, Ordering::Relaxed);

        let (addr, endpoint_name) = Self::parse_address(&address)?;
        tracing::info!(
            "Parsed address: socket={}, endpoint={:?}",
            addr,
            endpoint_name
        );

        // Add endpoint path to headers if present in address
        if let Some(endpoint_name) = endpoint_name {
            headers.insert("x-endpoint-path".to_string(), endpoint_name.clone());
            tracing::debug!("Added x-endpoint-path header: {}", endpoint_name);
        }

        // Get connection from pool
        let mut conn = self.pool.get_connection(addr).await?;

        // Send request with timeout
        let result = tokio::time::timeout(
            self.config.request_timeout,
            conn.send_request(payload, &headers),
        )
        .await;

        match result {
            Ok(Ok(ack)) => {
                self.stats
                    .responses_received
                    .fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_received
                    .fetch_add(ack.len() as u64, Ordering::Relaxed);

                // Return connection to pool
                self.pool.return_connection(conn).await;

                Ok(ack)
            }
            Ok(Err(e)) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
            Err(_) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
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

    #[test]
    fn test_tcp_config_default() {
        let config = TcpRequestConfig::default();
        assert_eq!(config.pool_size, DEFAULT_POOL_SIZE);
        assert_eq!(
            config.request_timeout,
            Duration::from_secs(DEFAULT_TCP_REQUEST_TIMEOUT_SECS)
        );
    }

    #[test]
    fn test_parse_address() {
        let (addr1, _) = TcpRequestClient::parse_address("127.0.0.1:8080").unwrap();
        assert_eq!(addr1.port(), 8080);

        let (addr2, _) = TcpRequestClient::parse_address("tcp://127.0.0.1:9090").unwrap();
        assert_eq!(addr2.port(), 9090);

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
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 client for request plane

use super::unified_client::{Headers, RequestPlaneClient};
use crate::Result;
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Duration;

/// Default timeout for HTTP requests (ack only, not full response)
const DEFAULT_HTTP_REQUEST_TIMEOUT_SECS: u64 = 5;

/// HTTP/2 Performance Configuration Constants
const DEFAULT_MAX_FRAME_SIZE: u32 = 1024 * 1024; // 1MB frame size for better throughput
const DEFAULT_MAX_CONCURRENT_STREAMS: u32 = 1000; // Allow more concurrent streams
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 100; // Increased connection pool
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 90; // Keep connections alive longer
const DEFAULT_HTTP2_KEEP_ALIVE_INTERVAL_SECS: u64 = 30; // Send pings every 30s
const DEFAULT_HTTP2_KEEP_ALIVE_TIMEOUT_SECS: u64 = 10; // Timeout for ping responses
const DEFAULT_HTTP2_ADAPTIVE_WINDOW: bool = true; // Enable adaptive flow control

/// HTTP/2 Performance Configuration
#[derive(Debug, Clone)]
pub struct Http2Config {
    pub max_frame_size: u32,
    pub max_concurrent_streams: u32,
    pub pool_max_idle_per_host: usize,
    pub pool_idle_timeout: Duration,
    pub keep_alive_interval: Duration,
    pub keep_alive_timeout: Duration,
    pub adaptive_window: bool,
    pub request_timeout: Duration,
}

impl Default for Http2Config {
    fn default() -> Self {
        Self {
            max_frame_size: DEFAULT_MAX_FRAME_SIZE,
            max_concurrent_streams: DEFAULT_MAX_CONCURRENT_STREAMS,
            pool_max_idle_per_host: DEFAULT_POOL_MAX_IDLE_PER_HOST,
            pool_idle_timeout: Duration::from_secs(DEFAULT_POOL_IDLE_TIMEOUT_SECS),
            keep_alive_interval: Duration::from_secs(DEFAULT_HTTP2_KEEP_ALIVE_INTERVAL_SECS),
            keep_alive_timeout: Duration::from_secs(DEFAULT_HTTP2_KEEP_ALIVE_TIMEOUT_SECS),
            adaptive_window: DEFAULT_HTTP2_ADAPTIVE_WINDOW,
            request_timeout: Duration::from_secs(DEFAULT_HTTP_REQUEST_TIMEOUT_SECS),
        }
    }
}

impl Http2Config {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("DYN_HTTP2_MAX_FRAME_SIZE")
            && let Ok(size) = val.parse::<u32>()
        {
            config.max_frame_size = size;
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_MAX_CONCURRENT_STREAMS")
            && let Ok(streams) = val.parse::<u32>()
        {
            config.max_concurrent_streams = streams;
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_POOL_MAX_IDLE_PER_HOST")
            && let Ok(pool_size) = val.parse::<usize>()
        {
            config.pool_max_idle_per_host = pool_size;
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_POOL_IDLE_TIMEOUT_SECS")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.pool_idle_timeout = Duration::from_secs(timeout);
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_KEEP_ALIVE_INTERVAL_SECS")
            && let Ok(interval) = val.parse::<u64>()
        {
            config.keep_alive_interval = Duration::from_secs(interval);
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_KEEP_ALIVE_TIMEOUT_SECS")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.keep_alive_timeout = Duration::from_secs(timeout);
        }

        if let Ok(val) = std::env::var("DYN_HTTP2_ADAPTIVE_WINDOW") {
            config.adaptive_window = val.parse().unwrap_or(DEFAULT_HTTP2_ADAPTIVE_WINDOW);
        }

        if let Ok(val) = std::env::var("DYN_HTTP_REQUEST_TIMEOUT")
            && let Ok(timeout) = val.parse::<u64>()
        {
            config.request_timeout = Duration::from_secs(timeout);
        }

        config
    }
}

/// HTTP/2 request plane client
pub struct HttpRequestClient {
    client: reqwest::Client,
    config: Http2Config,
}

impl HttpRequestClient {
    /// Create a new HTTP request client with HTTP/2 and default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(Http2Config::default())
    }

    /// Create a new HTTP request client with custom timeout (legacy method)
    /// Uses HTTP/2 with prior knowledge to avoid ALPN negotiation overhead
    pub fn with_timeout(timeout: Duration) -> Result<Self> {
        let config = Http2Config {
            request_timeout: timeout,
            ..Http2Config::default()
        };
        Self::with_config(config)
    }

    /// Create a new HTTP request client with basic configuration
    ///
    /// Note: Advanced HTTP/2 configuration methods may not be available in all versions of reqwest.
    /// This implementation uses only the stable, widely-supported configuration options.
    pub fn with_config(config: Http2Config) -> Result<Self> {
        let builder = reqwest::Client::builder()
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .pool_idle_timeout(config.pool_idle_timeout)
            .timeout(config.request_timeout);
        // HTTP/2 is automatically negotiated by reqwest when available

        let client = builder.build()?;

        Ok(Self { client, config })
    }

    /// Create from environment configuration
    pub fn from_env() -> Result<Self> {
        Self::with_config(Http2Config::from_env())
    }

    /// Get the current HTTP/2 configuration
    pub fn config(&self) -> &Http2Config {
        &self.config
    }
}

impl Default for HttpRequestClient {
    fn default() -> Self {
        Self::new().expect("Failed to create HTTP request client")
    }
}

#[async_trait]
impl RequestPlaneClient for HttpRequestClient {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes> {
        let mut req = self
            .client
            .post(&address)
            .header("Content-Type", "application/octet-stream")
            .body(payload);

        // Add custom headers
        for (key, value) in headers {
            req = req.header(key, value);
        }

        let response = req.send().await?;

        if !response.status().is_success() {
            anyhow::bail!(
                "HTTP request failed with status {}: {}",
                response.status(),
                response.text().await.unwrap_or_default()
            );
        }

        let body = response.bytes().await?;
        Ok(body)
    }

    fn transport_name(&self) -> &'static str {
        "http2"
    }

    fn is_healthy(&self) -> bool {
        // HTTP client is stateless and always healthy if created successfully
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Bytes as AxumBytes, extract::State as AxumState, routing::post};
    use std::sync::Arc;
    use tokio::sync::Mutex as TokioMutex;

    #[test]
    fn test_http_client_creation() {
        let client = HttpRequestClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_http_client_with_custom_timeout() {
        let client = HttpRequestClient::with_timeout(Duration::from_secs(10));
        assert!(client.is_ok());
        assert_eq!(
            client.unwrap().config.request_timeout,
            Duration::from_secs(10)
        );
    }

    #[test]
    fn test_http2_config_from_env() {
        // Set environment variables
        unsafe {
            std::env::set_var("DYN_HTTP2_MAX_FRAME_SIZE", "2097152"); // 2MB
            std::env::set_var("DYN_HTTP2_MAX_CONCURRENT_STREAMS", "2000");
            std::env::set_var("DYN_HTTP2_POOL_MAX_IDLE_PER_HOST", "200");
            std::env::set_var("DYN_HTTP2_KEEP_ALIVE_INTERVAL_SECS", "60");
            std::env::set_var("DYN_HTTP2_ADAPTIVE_WINDOW", "false");
        }

        let config = Http2Config::from_env();

        assert_eq!(config.max_frame_size, 2097152);
        assert_eq!(config.max_concurrent_streams, 2000);
        assert_eq!(config.pool_max_idle_per_host, 200);
        assert_eq!(config.keep_alive_interval, Duration::from_secs(60));
        assert!(!config.adaptive_window);

        // Clean up
        unsafe {
            std::env::remove_var("DYN_HTTP2_MAX_FRAME_SIZE");
            std::env::remove_var("DYN_HTTP2_MAX_CONCURRENT_STREAMS");
            std::env::remove_var("DYN_HTTP2_POOL_MAX_IDLE_PER_HOST");
            std::env::remove_var("DYN_HTTP2_KEEP_ALIVE_INTERVAL_SECS");
            std::env::remove_var("DYN_HTTP2_ADAPTIVE_WINDOW");
        }
    }

    #[test]
    fn test_http_client_with_custom_config() {
        let config = Http2Config {
            max_frame_size: 512 * 1024, // 512KB
            max_concurrent_streams: 500,
            pool_max_idle_per_host: 75,
            pool_idle_timeout: Duration::from_secs(60),
            keep_alive_interval: Duration::from_secs(45),
            keep_alive_timeout: Duration::from_secs(15),
            adaptive_window: false,
            request_timeout: Duration::from_secs(8),
        };

        let client = HttpRequestClient::with_config(config.clone());
        assert!(client.is_ok());

        let client = client.unwrap();
        assert_eq!(client.config.max_frame_size, 512 * 1024);
        assert_eq!(client.config.max_concurrent_streams, 500);
        assert_eq!(client.config.pool_max_idle_per_host, 75);
        assert_eq!(client.config.request_timeout, Duration::from_secs(8));
    }

    #[tokio::test]
    async fn test_http_client_send_request_invalid_url() {
        let client = HttpRequestClient::new().unwrap();
        let result = client
            .send_request(
                "http://invalid-host-that-does-not-exist:9999/test".to_string(),
                Bytes::from("test"),
                std::collections::HashMap::new(),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_http2_client_server_integration() {
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use hyper_util::server::conn::auto::Builder as ConnBuilder;
        use hyper_util::service::TowerToHyperService;

        // Create a test server that accepts HTTP/2
        #[derive(Clone)]
        struct TestState {
            received: Arc<TokioMutex<Vec<Bytes>>>,
            protocol_version: Arc<TokioMutex<Option<String>>>,
        }

        async fn test_handler(
            AxumState(state): AxumState<TestState>,
            body: AxumBytes,
        ) -> &'static str {
            state.received.lock().await.push(body);
            "OK"
        }

        let state = TestState {
            received: Arc::new(TokioMutex::new(Vec::new())),
            protocol_version: Arc::new(TokioMutex::new(None)),
        };

        let app = Router::new()
            .route("/test", post(test_handler))
            .with_state(state.clone());

        // Bind to a random port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Start HTTP/2 server
        let server_handle = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };

                let app = app.clone();
                tokio::spawn(async move {
                    let conn_builder = ConnBuilder::new(TokioExecutor::new());
                    let io = TokioIo::new(stream);
                    let tower_service = app.into_service();
                    let hyper_service = TowerToHyperService::new(tower_service);

                    let _ = conn_builder.serve_connection(io, hyper_service).await;
                });
            }
        });

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Create HTTP/2 client with prior knowledge
        let client = HttpRequestClient::new().unwrap();

        // Send request
        let test_data = Bytes::from("test_payload");
        let result = client
            .send_request(
                format!("http://{}/test", addr),
                test_data.clone(),
                std::collections::HashMap::new(),
            )
            .await;

        // Verify request succeeded
        assert!(result.is_ok(), "Request failed: {:?}", result.err());

        // Verify server received the data
        tokio::time::sleep(Duration::from_millis(100)).await;
        let received = state.received.lock().await;
        assert_eq!(received.len(), 1);
        assert_eq!(received[0], test_data);

        // Cleanup
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_http2_headers_propagation() {
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use hyper_util::server::conn::auto::Builder as ConnBuilder;
        use hyper_util::service::TowerToHyperService;

        // Create a test server that captures headers
        #[derive(Clone)]
        struct HeaderState {
            headers: Arc<TokioMutex<Vec<(String, String)>>>,
        }

        async fn header_handler(
            AxumState(state): AxumState<HeaderState>,
            headers: axum::http::HeaderMap,
        ) -> &'static str {
            let mut captured = state.headers.lock().await;
            for (name, value) in headers.iter() {
                if let Ok(val_str) = value.to_str() {
                    captured.push((name.to_string(), val_str.to_string()));
                }
            }
            "OK"
        }

        let state = HeaderState {
            headers: Arc::new(TokioMutex::new(Vec::new())),
        };

        let app = Router::new()
            .route("/test", post(header_handler))
            .with_state(state.clone());

        // Bind to a random port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Start HTTP/2 server
        let server_handle = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };

                let app = app.clone();
                tokio::spawn(async move {
                    let conn_builder = ConnBuilder::new(TokioExecutor::new());
                    let io = TokioIo::new(stream);
                    let tower_service = app.into_service();
                    let hyper_service = TowerToHyperService::new(tower_service);

                    let _ = conn_builder.serve_connection(io, hyper_service).await;
                });
            }
        });

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Create HTTP/2 client
        let client = HttpRequestClient::new().unwrap();

        // Send request with custom headers
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-test-header".to_string(), "test-value".to_string());
        headers.insert("x-request-id".to_string(), "req-123".to_string());

        let result = client
            .send_request(
                format!("http://{}/test", addr),
                Bytes::from("test"),
                headers,
            )
            .await;

        // Verify request succeeded
        assert!(result.is_ok());

        // Verify headers were received
        tokio::time::sleep(Duration::from_millis(100)).await;
        let received_headers = state.headers.lock().await;

        let header_map: std::collections::HashMap<_, _> = received_headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        assert!(header_map.contains_key("x-test-header"));
        assert_eq!(header_map.get("x-test-header"), Some(&"test-value"));
        assert!(header_map.contains_key("x-request-id"));
        assert_eq!(header_map.get("x-request-id"), Some(&"req-123"));

        // Cleanup
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_http2_concurrent_requests() {
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use hyper_util::server::conn::auto::Builder as ConnBuilder;
        use hyper_util::service::TowerToHyperService;
        use std::sync::atomic::{AtomicU64, Ordering};

        // Create a test server that counts requests
        #[derive(Clone)]
        struct CounterState {
            count: Arc<AtomicU64>,
        }

        async fn counter_handler(AxumState(state): AxumState<CounterState>) -> String {
            let count = state.count.fetch_add(1, Ordering::SeqCst);
            format!("{}", count)
        }

        let state = CounterState {
            count: Arc::new(AtomicU64::new(0)),
        };

        let app = Router::new()
            .route("/test", post(counter_handler))
            .with_state(state.clone());

        // Bind to a random port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Start HTTP/2 server
        let server_handle = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };

                let app = app.clone();
                tokio::spawn(async move {
                    let conn_builder = ConnBuilder::new(TokioExecutor::new());
                    let io = TokioIo::new(stream);
                    let tower_service = app.into_service();
                    let hyper_service = TowerToHyperService::new(tower_service);

                    let _ = conn_builder.serve_connection(io, hyper_service).await;
                });
            }
        });

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Create HTTP/2 client
        let client = Arc::new(HttpRequestClient::new().unwrap());

        // Send multiple concurrent requests (HTTP/2 multiplexing)
        let mut handles = vec![];
        for _ in 0..10 {
            let client = client.clone();
            let handle = tokio::spawn(async move {
                client
                    .send_request(
                        format!("http://{}/test", addr),
                        Bytes::from("test"),
                        std::collections::HashMap::new(),
                    )
                    .await
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        let mut success_count = 0;
        for handle in handles {
            if let Ok(Ok(_)) = handle.await {
                success_count += 1;
            }
        }

        // Verify all requests succeeded
        assert_eq!(success_count, 10);

        // Verify server received all requests
        assert_eq!(state.count.load(Ordering::SeqCst), 10);

        // Cleanup
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_http2_performance_benchmark() {
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use hyper_util::server::conn::auto::Builder as ConnBuilder;
        use hyper_util::service::TowerToHyperService;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Instant;

        // Create a test server that measures performance
        #[derive(Clone)]
        struct PerfState {
            request_count: Arc<AtomicU64>,
            total_bytes: Arc<AtomicU64>,
        }

        async fn perf_handler(
            AxumState(state): AxumState<PerfState>,
            body: AxumBytes,
        ) -> &'static str {
            state.request_count.fetch_add(1, Ordering::Relaxed);
            state
                .total_bytes
                .fetch_add(body.len() as u64, Ordering::Relaxed);
            "OK"
        }

        let state = PerfState {
            request_count: Arc::new(AtomicU64::new(0)),
            total_bytes: Arc::new(AtomicU64::new(0)),
        };

        let app = Router::new()
            .route("/perf", post(perf_handler))
            .with_state(state.clone());

        // Bind to a random port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Start HTTP/2 server
        let server_handle = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };

                let app = app.clone();
                tokio::spawn(async move {
                    let conn_builder = ConnBuilder::new(TokioExecutor::new());
                    let io = TokioIo::new(stream);
                    let tower_service = app.into_service();
                    let hyper_service = TowerToHyperService::new(tower_service);

                    let _ = conn_builder.serve_connection(io, hyper_service).await;
                });
            }
        });

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Create optimized HTTP/2 client
        let optimized_config = Http2Config {
            max_frame_size: 1024 * 1024, // 1MB frames
            max_concurrent_streams: 1000,
            pool_max_idle_per_host: 100,
            pool_idle_timeout: Duration::from_secs(90),
            keep_alive_interval: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(10),
            adaptive_window: true,
            request_timeout: Duration::from_secs(30),
        };

        let client = Arc::new(HttpRequestClient::with_config(optimized_config).unwrap());

        // Performance test: Send many concurrent requests
        let num_requests = 100;
        let payload_size = 64 * 1024; // 64KB payload
        let payload = Bytes::from(vec![0u8; payload_size]);

        let start_time = Instant::now();
        let mut handles = vec![];

        for _ in 0..num_requests {
            let client = client.clone();
            let payload = payload.clone();

            let handle = tokio::spawn(async move {
                let headers = std::collections::HashMap::new();
                client
                    .send_request(format!("http://{}/perf", addr), payload, headers)
                    .await
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        let mut successful_requests = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                successful_requests += 1;
            }
        }

        let elapsed = start_time.elapsed();
        let requests_per_sec = successful_requests as f64 / elapsed.as_secs_f64();
        let throughput_mbps =
            (successful_requests * payload_size) as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);

        println!("Performance Results:");
        println!(
            "  Successful requests: {}/{}",
            successful_requests, num_requests
        );
        println!("  Total time: {:?}", elapsed);
        println!("  Requests/sec: {:.2}", requests_per_sec);
        println!("  Throughput: {:.2} MB/s", throughput_mbps);

        // Verify server received all requests
        let server_count = state.request_count.load(Ordering::Relaxed);
        let server_bytes = state.total_bytes.load(Ordering::Relaxed);

        assert_eq!(server_count, successful_requests as u64);
        assert_eq!(server_bytes, (successful_requests * payload_size) as u64);

        // Performance assertions (adjust based on your requirements)
        assert!(successful_requests >= num_requests * 95 / 100); // At least 95% success rate
        assert!(requests_per_sec > 50.0); // At least 50 requests per second
        assert!(throughput_mbps > 10.0); // At least 10 MB/s throughput

        // Cleanup
        server_handle.abort();
    }
}

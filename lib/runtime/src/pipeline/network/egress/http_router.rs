// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 client for request plane

use super::*;
use crate::logging::{DistributedTraceContext, get_distributed_tracing_context};
use crate::pipeline::network::{
    ConnectionInfo, NetworkStreamWrapper, PendingConnections, ResponseStream, STREAM_ERR_MSG,
    StreamOptions,
    codec::{TwoPartCodec, TwoPartMessage},
    request_plane::{Headers, RequestPlaneClient},
};
use crate::pipeline::{AddressedRequest, AsyncEngine, Data, Error, ManyOut, SingleIn};
use crate::{Result, protocols::maybe_error::MaybeError};
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::StreamExt;
use tracing as log;
use tracing::Instrument;

/// Default timeout for HTTP requests (ack only, not full response)
const DEFAULT_HTTP_REQUEST_TIMEOUT_SECS: u64 = 5;

/// HTTP/2 request plane client
pub struct HttpRequestClient {
    client: reqwest::Client,
    request_timeout: Duration,
}

impl HttpRequestClient {
    /// Create a new HTTP request client with HTTP/2
    pub fn new() -> Result<Self> {
        Self::with_timeout(Duration::from_secs(DEFAULT_HTTP_REQUEST_TIMEOUT_SECS))
    }

    /// Create a new HTTP request client with custom timeout
    /// Uses HTTP/2 with prior knowledge (no protocol negotiation)
    pub fn with_timeout(timeout: Duration) -> Result<Self> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(50) // Connection pooling
            .timeout(timeout)
            // Note: HTTP/2 will be negotiated automatically by reqwest
            .build()?;

        Ok(Self {
            client,
            request_timeout: timeout,
        })
    }

    /// Create from environment configuration
    pub fn from_env() -> Result<Self> {
        let timeout_secs = std::env::var("DYN_HTTP_REQUEST_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_HTTP_REQUEST_TIMEOUT_SECS);

        Self::with_timeout(Duration::from_secs(timeout_secs))
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
}

/// HTTP-based AddressedPushRouter
///
/// This router sends requests via HTTP/2 instead of NATS.
/// Responses still stream back over TCP (unchanged).
pub struct HttpAddressedRouter {
    // HTTP client for request plane
    http_client: Arc<HttpRequestClient>,

    // TCP server for response plane (unchanged)
    resp_transport: Arc<tcp::server::TcpStreamServer>,
}

impl HttpAddressedRouter {
    pub fn new(
        http_client: Arc<HttpRequestClient>,
        resp_transport: Arc<tcp::server::TcpStreamServer>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            http_client,
            resp_transport,
        }))
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<AddressedRequest<T>>, ManyOut<U>, Error> for HttpAddressedRouter
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
        let request_id = request.context().id().to_string();
        let (addressed_request, context) = request.transfer(());
        let (request, address) = addressed_request.into_parts();
        let engine_ctx = context.context();
        let engine_ctx_ = engine_ctx.clone();

        // Registration options for the data plane in a single in / many out configuration
        let options = StreamOptions::builder()
            .context(engine_ctx.clone())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        // Register our needs with the data plane (TCP for responses)
        let pending_connections: PendingConnections = self.resp_transport.register(options).await;

        // Validate and unwrap the RegisteredStream object
        let pending_response_stream = match pending_connections.into_parts() {
            (None, Some(recv_stream)) => recv_stream,
            _ => {
                panic!("Invalid data plane registration for a SingleIn/ManyOut transport");
            }
        };

        // Separate out the connection info and the stream provider from the registered stream
        let (connection_info, response_stream_provider) = pending_response_stream.into_parts();

        // Package up the connection info as part of the "header" component of the two part message
        let control_message = RequestControlMessage {
            id: engine_ctx.id().to_string(),
            request_type: RequestType::SingleIn,
            response_type: ResponseType::ManyOut,
            connection_info,
        };

        // Build the two part message where we package the connection info and the request
        let ctrl = serde_json::to_vec(&control_message)?;
        let data = serde_json::to_vec(&request)?;

        log::trace!(
            request_id,
            "packaging two-part message; ctrl: {} bytes, data: {} bytes",
            ctrl.len(),
            data.len()
        );

        let msg = TwoPartMessage::from_parts(ctrl.into(), data.into());
        let codec = TwoPartCodec::default();
        let buffer = codec.encode_message(msg)?;

        log::trace!(request_id, "sending HTTP request to {}", address);

        // Insert Trace Context into Headers
        let mut headers = std::collections::HashMap::new();
        if let Some(trace_context) = get_distributed_tracing_context() {
            headers.insert(
                "traceparent".to_string(),
                trace_context.create_traceparent(),
            );
            if let Some(tracestate) = trace_context.tracestate {
                headers.insert("tracestate".to_string(), tracestate);
            }
            if let Some(x_request_id) = trace_context.x_request_id {
                headers.insert("x-request-id".to_string(), x_request_id);
            }
            if let Some(x_dynamo_request_id) = trace_context.x_dynamo_request_id {
                headers.insert("x-dynamo-request-id".to_string(), x_dynamo_request_id);
            }
        }

        // Send HTTP request (replaces NATS request)
        let _response = self
            .http_client
            .send_request(address, buffer, headers)
            .await?;

        log::trace!(request_id, "awaiting transport handshake");
        let response_stream = response_stream_provider
            .await
            .map_err(|_| PipelineError::DetachedStreamReceiver)?
            .map_err(PipelineError::ConnectionFailed)?;

        // TODO: Detect end-of-stream using Server-Sent Events (SSE)
        let mut is_complete_final = false;
        let stream = tokio_stream::StreamNotifyClose::new(
            tokio_stream::wrappers::ReceiverStream::new(response_stream.rx),
        )
        .filter_map(move |res| {
            if let Some(res_bytes) = res {
                if is_complete_final {
                    return Some(U::from_err(
                        Error::msg(
                            "Response received after generation ended - this should never happen",
                        )
                        .into(),
                    ));
                }
                match serde_json::from_slice::<NetworkStreamWrapper<U>>(&res_bytes) {
                    Ok(item) => {
                        is_complete_final = item.complete_final;
                        if let Some(data) = item.data {
                            Some(data)
                        } else if is_complete_final {
                            None
                        } else {
                            Some(U::from_err(
                                Error::msg("Empty response received - this should never happen")
                                    .into(),
                            ))
                        }
                    }
                    Err(err) => {
                        let json_str = String::from_utf8_lossy(&res_bytes);
                        log::warn!(%err, %json_str, "Failed deserializing JSON to response");
                        Some(U::from_err(Error::new(err).into()))
                    }
                }
            } else if is_complete_final {
                // end of stream
                None
            } else if engine_ctx_.is_stopped() {
                log::debug!("Request cancelled and then trying to read a response");
                None
            } else {
                // stream ended unexpectedly
                log::debug!("{STREAM_ERR_MSG}");
                Some(U::from_err(Error::msg(STREAM_ERR_MSG).into()))
            }
        });

        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::post, body::Bytes as AxumBytes, extract::State as AxumState};
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
        assert_eq!(client.unwrap().request_timeout, Duration::from_secs(10));
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
            let addr = addr;
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
}

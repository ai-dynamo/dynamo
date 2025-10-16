# Transport Registry - Usage Examples

This document provides practical code examples for using the Transport Registry system.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Priority 0: Enabling Connection Pooling](#priority-0-enabling-connection-pooling)
3. [Priority 1: HTTP Response Path](#priority-1-http-response-path)
4. [Priority 2: TCP Request Path](#priority-2-tcp-request-path)
5. [Custom Transport Implementation](#custom-transport-implementation)

---

## Basic Setup

### Creating and Initializing the Registry

```rust
use dynamo::pipeline::network::transport_registry::*;
use std::sync::Arc;

async fn initialize_transport_registry() -> Result<Arc<TransportRegistry>> {
    let registry = Arc::new(TransportRegistry::new());

    // Register available transports based on configuration
    register_default_transports(&registry).await?;

    Ok(registry)
}

async fn register_default_transports(registry: &TransportRegistry) -> Result<()> {
    // Register HTTP request transport
    if let Ok(http_client) = create_http_request_client() {
        let http_server = create_http_request_server()?;

        registry.register_request_transport(
            TransportRegistration {
                transport_id: TransportId::Http,
                plane_type: PlaneType::Request,
                capabilities: TransportCapabilities {
                    streaming: false,
                    persistent_connections: true,
                    bidirectional: false,
                    max_message_size: None,
                },
                priority: 10,
            },
            Arc::new(http_client),
            Arc::new(http_server),
        ).await?;
    }

    // Register TCP response transport
    if let Ok(tcp_server) = create_tcp_response_server().await {
        let tcp_client = create_tcp_response_client();

        registry.register_response_transport(
            TransportRegistration {
                transport_id: TransportId::Tcp,
                plane_type: PlaneType::Response,
                capabilities: TransportCapabilities {
                    streaming: true,
                    persistent_connections: true,
                    bidirectional: false,
                    max_message_size: None,
                },
                priority: 10,
            },
            Arc::new(tcp_client),
            Arc::new(tcp_server),
        ).await?;
    }

    Ok(())
}
```

### Using the Registry in an Endpoint

```rust
use dynamo::component::Endpoint;
use dynamo::pipeline::network::transport_registry::*;

struct MyEndpoint {
    transport_registry: Arc<TransportRegistry>,
    handler: Arc<dyn PushWorkHandler>,
}

impl MyEndpoint {
    pub async fn new(handler: Arc<dyn PushWorkHandler>) -> Result<Self> {
        let registry = initialize_transport_registry().await?;

        Ok(Self {
            transport_registry: registry,
            handler,
        })
    }

    pub async fn start(&self) -> Result<()> {
        // Get request server based on configuration
        let request_plane_mode = RequestPlaneMode::from_env();
        let transport_id = request_plane_mode.into();

        if let Some(server) = self.transport_registry
            .get_request_server(&transport_id)
            .await
        {
            let bind_address = get_bind_address_from_env();
            server.start(bind_address).await?;

            tracing::info!(
                transport = ?transport_id,
                address = server.public_address(),
                "Request plane server started"
            );
        }

        // Similarly for response server
        let response_plane_mode = ResponsePlaneMode::from_env();
        let response_transport_id = response_plane_mode.into();

        if let Some(server) = self.transport_registry
            .get_response_server(&response_transport_id)
            .await
        {
            let bind_address = get_response_bind_address_from_env();
            server.start(bind_address).await?;

            tracing::info!(
                transport = ?response_transport_id,
                address = server.public_address(),
                "Response plane server started"
            );
        }

        Ok(())
    }
}
```

---

## Priority 0: Enabling Connection Pooling

### Configuration

```bash
# Enable connection pooling
export DYN_TCP_POOL_ENABLED=true

# Optional: Tune parameters
export DYN_TCP_POOL_MAX_PER_TARGET=50
export DYN_TCP_POOL_MIN_IDLE=5
export DYN_TCP_POOL_IDLE_TIMEOUT_SECS=300
export DYN_TCP_POOL_MAX_LIFETIME_SECS=3600
export DYN_TCP_POOL_KEEPALIVE_SECS=60
```

### Worker Side: Using Pooled Connections

```rust
use dynamo::pipeline::network::tcp::connection_pool::*;
use std::sync::Arc;

pub struct WorkerResponseHandler {
    connection_pool: Arc<TcpResponseConnectionPool>,
}

impl WorkerResponseHandler {
    pub fn new() -> Self {
        let config = PoolConfig {
            max_connections_per_target: std::env::var("DYN_TCP_POOL_MAX_PER_TARGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
            min_idle_connections: std::env::var("DYN_TCP_POOL_MIN_IDLE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            idle_timeout: Duration::from_secs(
                std::env::var("DYN_TCP_POOL_IDLE_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(300)
            ),
            max_lifetime: Duration::from_secs(
                std::env::var("DYN_TCP_POOL_MAX_LIFETIME_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3600)
            ),
            tcp_keepalive: Some(Duration::from_secs(
                std::env::var("DYN_TCP_POOL_KEEPALIVE_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60)
            )),
        };

        Self {
            connection_pool: Arc::new(TcpResponseConnectionPool::new(config)),
        }
    }

    pub async fn send_response(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        connection_info: &ConnectionInfo,
        response: Bytes,
    ) -> Result<()> {
        // Parse TCP connection info
        let tcp_info = TcpStreamConnectionInfo::try_from(connection_info.clone())?;

        // Acquire pooled connection (reuses existing if available)
        let mut pooled_stream = self.connection_pool
            .acquire(&tcp_info.address, &tcp_info.context)
            .await?;

        tracing::debug!(
            target = %tcp_info.address,
            "Acquired pooled connection"
        );

        // Send handshake
        let handshake = CallHomeHandshake {
            subject: tcp_info.subject.clone(),
            stream_type: StreamType::Response,
        };

        let mut writer = FramedWrite::new(
            pooled_stream.stream(),
            TwoPartCodec::default()
        );

        writer.send(TwoPartMessage::from_header(
            serde_json::to_vec(&handshake)?.into()
        )).await?;

        // Send response
        writer.send(TwoPartMessage::from_data(response)).await?;

        // Connection automatically returned to pool on drop
        Ok(())
    }
}
```

### Monitoring Pool Usage

```rust
use prometheus::{Counter, Gauge, Histogram};

struct ConnectionPoolMetrics {
    connections_created: Counter,
    connections_reused: Counter,
    active_connections: Gauge,
    idle_connections: Gauge,
    acquisition_latency: Histogram,
}

impl ConnectionPoolMetrics {
    pub fn new(registry: &prometheus::Registry) -> Result<Self> {
        let connections_created = Counter::new(
            "tcp_pool_connections_created_total",
            "Total TCP connections created"
        )?;
        registry.register(Box::new(connections_created.clone()))?;

        let connections_reused = Counter::new(
            "tcp_pool_connections_reused_total",
            "Total TCP connections reused from pool"
        )?;
        registry.register(Box::new(connections_reused.clone()))?;

        let active_connections = Gauge::new(
            "tcp_pool_active_connections",
            "Current active TCP connections"
        )?;
        registry.register(Box::new(active_connections.clone()))?;

        let idle_connections = Gauge::new(
            "tcp_pool_idle_connections",
            "Current idle TCP connections in pool"
        )?;
        registry.register(Box::new(idle_connections.clone()))?;

        let acquisition_latency = Histogram::with_opts(
            HistogramOpts::new(
                "tcp_pool_acquisition_latency_seconds",
                "Connection acquisition latency"
            ).buckets(vec![0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
        )?;
        registry.register(Box::new(acquisition_latency.clone()))?;

        Ok(Self {
            connections_created,
            connections_reused,
            active_connections,
            idle_connections,
            acquisition_latency,
        })
    }
}
```

---

## Priority 1: HTTP Response Path

### Configuration

```bash
# Enable HTTP response path
export DYN_RESPONSE_PLANE=http

# Configure HTTP response server
export DYN_HTTP_RESPONSE_HOST=0.0.0.0
export DYN_HTTP_RESPONSE_PORT=8082
export DYN_HTTP_RESPONSE_ROOT_PATH=/v1/rpc/response
export DYN_HTTP_RESPONSE_STREAM_FORMAT=sse
```

### Server Side: HTTP Response Endpoint

```rust
use axum::{
    Router,
    routing::post,
    extract::{Path, State},
    response::{Response, sse::{Event, Sse}},
    http::StatusCode,
};
use futures::stream::Stream;
use tokio::sync::mpsc;
use std::convert::Infallible;

pub struct HttpResponseServer {
    app: Router,
    bind_address: String,
    public_address: String,
    handler: Arc<dyn PushWorkHandler>,
}

impl HttpResponseServer {
    pub fn new(handler: Arc<dyn PushWorkHandler>) -> Self {
        let app = Router::new()
            .route("/:subject", post(handle_streaming_request))
            .with_state(handler.clone());

        let bind_address = format!(
            "{}:{}",
            get_http_response_host_from_env(),
            get_http_response_port_from_env()
        );

        let public_address = format!(
            "http://{}:{}{}",
            get_public_http_response_host_from_env(),
            get_http_response_port_from_env(),
            get_http_response_root_path_from_env()
        );

        Self {
            app,
            bind_address,
            public_address,
            handler,
        }
    }
}

async fn handle_streaming_request(
    State(handler): State<Arc<dyn PushWorkHandler>>,
    Path(subject): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    // Extract tracing context
    let trace_ctx = extract_tracing_context(&headers);

    // Create response stream
    let (tx, rx) = mpsc::channel(64);

    // Spawn handler
    tokio::spawn(async move {
        let _guard = trace_ctx.attach();

        match handler.handle_payload(body).await {
            Ok(()) => {
                tracing::debug!("Request processed successfully");
            }
            Err(e) => {
                tracing::error!("Error processing request: {}", e);
                let _ = tx.send(Err(e)).await;
            }
        }
    });

    // Convert to SSE stream
    let stream = ReceiverStream::new(rx).map(|result| {
        match result {
            Ok(data) => {
                let encoded = base64::encode(&data);
                Ok(Event::default().data(encoded))
            }
            Err(e) => {
                Ok(Event::default()
                    .event("error")
                    .data(e.to_string()))
            }
        }
    });

    Ok(Sse::new(stream))
}

#[async_trait]
impl ResponsePlaneServer for HttpResponseServer {
    async fn start(&mut self, bind_address: String) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(&bind_address).await?;

        tracing::info!(
            address = %bind_address,
            "Starting HTTP response server"
        );

        axum::serve(listener, self.app.clone()).await?;

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        // Graceful shutdown handled by axum
        Ok(())
    }

    fn public_address(&self) -> String {
        self.public_address.clone()
    }

    fn capabilities(&self) -> &TransportCapabilities {
        &TransportCapabilities {
            streaming: true,
            persistent_connections: true,
            bidirectional: false,
            max_message_size: None,
        }
    }
}
```

### Client Side: HTTP Response Client

```rust
use reqwest::{Client, Response};
use futures::StreamExt;

pub struct HttpResponseClient {
    client: Client,
}

impl HttpResponseClient {
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .http2_prior_knowledge()
            .pool_max_idle_per_host(50)
            .timeout(Duration::from_secs(3600))
            .build()?;

        Ok(Self { client })
    }
}

#[async_trait]
impl ResponsePlaneClient for HttpResponseClient {
    async fn create_response_stream(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        connection_info: ResponseConnectionInfo,
    ) -> Result<Box<dyn ResponseStreamWriter>> {
        let url = match &connection_info.address {
            TransportAddress::Http { url } => url.clone(),
            _ => return Err(anyhow!("Expected HTTP address")),
        };

        // Create SSE stream reader
        let response = self.client
            .get(&url)
            .header("Accept", "text/event-stream")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "HTTP request failed: {}",
                response.status()
            ));
        }

        let stream = response.bytes_stream();
        let reader = parse_sse_stream(stream);

        Ok(Box::new(HttpResponseStreamReader {
            reader,
            context,
        }))
    }

    fn capabilities(&self) -> &TransportCapabilities {
        &TransportCapabilities {
            streaming: true,
            persistent_connections: true,
            bidirectional: false,
            max_message_size: None,
        }
    }

    async fn release_connection(
        &self,
        _connection_info: &ResponseConnectionInfo,
    ) -> Result<()> {
        // HTTP/2 connection pooling handled by reqwest
        Ok(())
    }

    async fn is_healthy(&self) -> bool {
        true
    }
}

struct HttpResponseStreamReader {
    reader: mpsc::Receiver<Result<Bytes>>,
    context: Arc<dyn AsyncEngineContext>,
}

#[async_trait]
impl ResponseStreamReader for HttpResponseStreamReader {
    async fn recv(&mut self) -> Option<Result<Bytes>> {
        self.reader.recv().await
    }

    fn transport_id(&self) -> TransportId {
        TransportId::Http
    }
}

fn parse_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
) -> mpsc::Receiver<Result<Bytes>> {
    let (tx, rx) = mpsc::channel(64);

    tokio::spawn(async move {
        let mut stream = stream;
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));

                    // Parse SSE events
                    while let Some(event) = parse_next_sse_event(&mut buffer) {
                        if event.event_type == "error" {
                            let _ = tx.send(Err(anyhow!(event.data))).await;
                        } else {
                            match base64::decode(&event.data) {
                                Ok(decoded) => {
                                    let _ = tx.send(Ok(decoded.into())).await;
                                }
                                Err(e) => {
                                    tracing::error!("Failed to decode SSE data: {}", e);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow!(e))).await;
                    break;
                }
            }
        }
    });

    rx
}
```

---

## Priority 2: TCP Request Path

### Configuration

```bash
# Enable TCP request path
export DYN_REQUEST_PLANE=tcp

# Configure TCP request server
export DYN_TCP_REQUEST_HOST=0.0.0.0
export DYN_TCP_REQUEST_PORT=9001
export DYN_TCP_REQUEST_MAX_CONNECTIONS=1000
export DYN_TCP_REQUEST_HEARTBEAT_INTERVAL_SECS=30
```

### Protocol Implementation

```rust
use uuid::Uuid;
use tokio_util::codec::{Decoder, Encoder};

/// TCP frame header for request/response multiplexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpFrameHeader {
    pub request_id: Uuid,
    pub frame_type: TcpFrameType,
    pub payload_length: u64,
    pub end_stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TcpFrameType {
    Request,
    Response,
    Control,
    Heartbeat,
}

pub struct TcpFrame {
    pub header: TcpFrameHeader,
    pub payload: Bytes,
}

pub struct TcpMultiplexCodec {
    state: CodecState,
}

enum CodecState {
    Header,
    Payload {
        header: TcpFrameHeader,
        remaining: usize,
    },
}

impl Decoder for TcpMultiplexCodec {
    type Item = TcpFrame;
    type Error = std::io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            match &mut self.state {
                CodecState::Header => {
                    // Need UUID(16) + type(1) + length(8) + flag(1) = 26 bytes minimum
                    if src.len() < 26 {
                        return Ok(None);
                    }

                    // Parse header
                    let request_id_bytes: [u8; 16] = src[0..16].try_into().unwrap();
                    let request_id = Uuid::from_bytes(request_id_bytes);

                    let frame_type = match src[16] {
                        0 => TcpFrameType::Request,
                        1 => TcpFrameType::Response,
                        2 => TcpFrameType::Control,
                        3 => TcpFrameType::Heartbeat,
                        _ => return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid frame type"
                        )),
                    };

                    let payload_length = u64::from_be_bytes(
                        src[17..25].try_into().unwrap()
                    );

                    let end_stream = src[25] != 0;

                    src.advance(26);

                    let header = TcpFrameHeader {
                        request_id,
                        frame_type,
                        payload_length,
                        end_stream,
                    };

                    self.state = CodecState::Payload {
                        header,
                        remaining: payload_length as usize,
                    };
                }
                CodecState::Payload { header, remaining } => {
                    if src.len() < *remaining {
                        return Ok(None);
                    }

                    let payload = src.split_to(*remaining).freeze();

                    let frame = TcpFrame {
                        header: header.clone(),
                        payload,
                    };

                    self.state = CodecState::Header;
                    return Ok(Some(frame));
                }
            }
        }
    }
}

impl Encoder<TcpFrame> for TcpMultiplexCodec {
    type Error = std::io::Error;

    fn encode(&mut self, frame: TcpFrame, dst: &mut BytesMut) -> Result<(), Self::Error> {
        // Write header
        dst.extend_from_slice(frame.header.request_id.as_bytes());

        let type_byte = match frame.header.frame_type {
            TcpFrameType::Request => 0,
            TcpFrameType::Response => 1,
            TcpFrameType::Control => 2,
            TcpFrameType::Heartbeat => 3,
        };
        dst.put_u8(type_byte);

        dst.put_u64(frame.header.payload_length);
        dst.put_u8(if frame.header.end_stream { 1 } else { 0 });

        // Write payload
        dst.extend_from_slice(&frame.payload);

        Ok(())
    }
}
```

### Server Implementation

```rust
pub struct TcpRequestServer {
    listener: Option<TcpListener>,
    bind_address: String,
    public_address: String,
    handler: Arc<dyn PushWorkHandler>,
    max_connections: usize,
}

impl TcpRequestServer {
    pub fn new(
        handler: Arc<dyn PushWorkHandler>,
        bind_address: String,
        public_address: String,
    ) -> Self {
        let max_connections = std::env::var("DYN_TCP_REQUEST_MAX_CONNECTIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);

        Self {
            listener: None,
            bind_address,
            public_address,
            handler,
            max_connections,
        }
    }

    async fn handle_connection(&self, socket: TcpStream) -> Result<()> {
        socket.set_nodelay(true)?;

        let (read_half, write_half) = tokio::io::split(socket);
        let mut reader = FramedRead::new(read_half, TcpMultiplexCodec::default());
        let writer = Arc::new(Mutex::new(
            FramedWrite::new(write_half, TcpMultiplexCodec::default())
        ));

        let handler = self.handler.clone();

        // Spawn heartbeat task
        let heartbeat_writer = writer.clone();
        let heartbeat_task = tokio::spawn(async move {
            let interval = std::env::var("DYN_TCP_REQUEST_HEARTBEAT_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30);

            let mut ticker = tokio::time::interval(Duration::from_secs(interval));

            loop {
                ticker.tick().await;

                let frame = TcpFrame {
                    header: TcpFrameHeader {
                        request_id: Uuid::nil(),
                        frame_type: TcpFrameType::Heartbeat,
                        payload_length: 0,
                        end_stream: false,
                    },
                    payload: Bytes::new(),
                };

                let mut w = heartbeat_writer.lock().await;
                if w.send(frame).await.is_err() {
                    break;
                }
            }
        });

        // Handle incoming frames
        while let Some(Ok(frame)) = reader.next().await {
            match frame.header.frame_type {
                TcpFrameType::Request => {
                    let request_id = frame.header.request_id;
                    let handler = handler.clone();
                    let writer = writer.clone();

                    tokio::spawn(async move {
                        match handler.handle_payload(frame.payload).await {
                            Ok(()) => {
                                // Send success response
                                let response_frame = TcpFrame {
                                    header: TcpFrameHeader {
                                        request_id,
                                        frame_type: TcpFrameType::Response,
                                        payload_length: 0,
                                        end_stream: true,
                                    },
                                    payload: Bytes::new(),
                                };

                                let mut w = writer.lock().await;
                                let _ = w.send(response_frame).await;
                            }
                            Err(e) => {
                                tracing::error!("Error handling request: {}", e);

                                // Send error response
                                let error_bytes = e.to_string().into_bytes();
                                let response_frame = TcpFrame {
                                    header: TcpFrameHeader {
                                        request_id,
                                        frame_type: TcpFrameType::Response,
                                        payload_length: error_bytes.len() as u64,
                                        end_stream: true,
                                    },
                                    payload: error_bytes.into(),
                                };

                                let mut w = writer.lock().await;
                                let _ = w.send(response_frame).await;
                            }
                        }
                    });
                }
                TcpFrameType::Heartbeat => {
                    // Respond to heartbeat
                    let frame = TcpFrame {
                        header: TcpFrameHeader {
                            request_id: Uuid::nil(),
                            frame_type: TcpFrameType::Heartbeat,
                            payload_length: 0,
                            end_stream: false,
                        },
                        payload: Bytes::new(),
                    };

                    let mut w = writer.lock().await;
                    let _ = w.send(frame).await;
                }
                _ => {}
            }
        }

        heartbeat_task.abort();

        Ok(())
    }
}

#[async_trait]
impl RequestPlaneServer for TcpRequestServer {
    async fn start(&mut self, bind_address: String) -> Result<()> {
        let listener = TcpListener::bind(&bind_address).await?;

        tracing::info!(
            address = %bind_address,
            "Starting TCP request server"
        );

        let handler = self.handler.clone();
        let max_connections = self.max_connections;

        // Accept connections
        loop {
            match listener.accept().await {
                Ok((socket, addr)) => {
                    tracing::debug!("Accepted connection from {}", addr);

                    let server = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_connection(socket).await {
                            tracing::error!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    tracing::error!("Accept error: {}", e);
                }
            }
        }
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(listener) = self.listener.take() {
            drop(listener);
        }
        Ok(())
    }

    fn public_address(&self) -> String {
        self.public_address.clone()
    }
}
```

---

## Custom Transport Implementation

### Implementing a Custom Request Transport

```rust
use dynamo::pipeline::network::transport_registry::*;

pub struct MyCustomRequestClient {
    // Your custom implementation
}

#[async_trait]
impl RequestPlaneClient for MyCustomRequestClient {
    async fn send_request(
        &self,
        address: &TransportAddress,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes> {
        // Your implementation here
        todo!()
    }

    fn capabilities(&self) -> &TransportCapabilities {
        &TransportCapabilities {
            streaming: false,
            persistent_connections: true,
            bidirectional: false,
            max_message_size: Some(10 * 1024 * 1024),
        }
    }

    async fn is_healthy(&self) -> bool {
        true
    }
}

pub struct MyCustomRequestServer {
    // Your custom implementation
}

#[async_trait]
impl RequestPlaneServer for MyCustomRequestServer {
    async fn start(&mut self, bind_address: String) -> Result<()> {
        // Your implementation here
        todo!()
    }

    async fn stop(&mut self) -> Result<()> {
        // Your implementation here
        todo!()
    }

    fn public_address(&self) -> String {
        // Your implementation here
        todo!()
    }
}

// Register your custom transport
async fn register_custom_transport(registry: &TransportRegistry) -> Result<()> {
    let client = Arc::new(MyCustomRequestClient::new()?);
    let server = Arc::new(MyCustomRequestServer::new()?);

    registry.register_request_transport(
        TransportRegistration {
            transport_id: TransportId::Custom("my_custom".to_string()),
            plane_type: PlaneType::Request,
            capabilities: client.capabilities().clone(),
            priority: 20, // Lower priority than default transports
        },
        client,
        server,
    ).await?;

    Ok(())
}
```

---

## Testing Examples

### Unit Test: Connection Pool

```rust
#[tokio::test]
async fn test_connection_pool_reuse() {
    let config = PoolConfig {
        max_connections_per_target: 10,
        min_idle_connections: 2,
        idle_timeout: Duration::from_secs(60),
        max_lifetime: Duration::from_secs(3600),
        tcp_keepalive: Some(Duration::from_secs(30)),
    };

    let pool = TcpResponseConnectionPool::new(config);

    // First acquisition creates new connection
    let conn1 = pool.acquire("localhost:8080", "ctx1").await.unwrap();
    let initial_created = pool.metrics.connections_created.get();
    drop(conn1);

    // Second acquisition should reuse
    let conn2 = pool.acquire("localhost:8080", "ctx1").await.unwrap();
    assert_eq!(
        pool.metrics.connections_created.get(),
        initial_created,
        "Should not create new connection"
    );
    assert_eq!(
        pool.metrics.connections_reused.get(),
        1,
        "Should reuse connection"
    );
}
```

### Integration Test: End-to-End Flow

```rust
#[tokio::test]
async fn test_http_request_tcp_response() {
    // Start servers
    let tcp_server = start_tcp_response_server().await;
    let http_server = start_http_request_server().await;

    // Register in service discovery
    register_test_endpoint(&tcp_server, &http_server).await;

    // Send request
    let client = HttpRequestClient::new().unwrap();
    let response = client.send_request(
        "http://localhost:8081/test-subject",
        b"test payload".to_vec().into(),
        HashMap::new(),
    ).await.unwrap();

    // Verify response received via TCP
    assert_eq!(response, b"test response");
}
```

---

## Debugging Tips

### Enable Debug Logging

```bash
export RUST_LOG=dynamo::pipeline::network=debug

# Or more specific
export RUST_LOG=dynamo::pipeline::network::tcp::connection_pool=trace
export RUST_LOG=dynamo::pipeline::network::transport_registry=debug
```

### Monitor Metrics

```bash
# Connection pool metrics
curl http://localhost:9090/metrics | grep tcp_pool

# Transport metrics
curl http://localhost:9090/metrics | grep transport_
```

### Test with curl

```bash
# Test HTTP response endpoint
curl -N -H "Accept: text/event-stream" \
  http://localhost:8082/v1/rpc/response/test-subject

# Test HTTP request endpoint
curl -X POST \
  -H "Content-Type: application/octet-stream" \
  --data-binary @payload.bin \
  http://localhost:8081/v1/rpc/test-subject
```

---

## See Also

- [TRANSPORT_REGISTRY_DESIGN.md](TRANSPORT_REGISTRY_DESIGN.md) - Full design document
- [TRANSPORT_REGISTRY_SUMMARY.md](TRANSPORT_REGISTRY_SUMMARY.md) - Quick reference
- [lib/runtime/src/pipeline/network/transport_registry.rs](lib/runtime/src/pipeline/network/transport_registry.rs) - Interface definitions


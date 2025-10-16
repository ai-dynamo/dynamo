# Transport Registry and Service Discovery Design

## Overview

This document describes the high-level interface and design for registering different transport mechanisms and advertising them through service discovery in Dynamo. The goal is to create a flexible, extensible architecture that supports multiple request and response transports with efficient connection management.

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Transport Abstraction Interface](#transport-abstraction-interface)
3. [Service Discovery Integration](#service-discovery-integration)
4. [Priority 0: TCP Connection Pooling](#priority-0-tcp-connection-pooling)
5. [Priority 1: HTTP Response Path](#priority-1-http-response-path)
6. [Priority 2: TCP Request Path](#priority-2-tcp-request-path)
7. [Migration Strategy](#migration-strategy)
8. [Configuration](#configuration)

---

## Current Architecture

### Supported Transport Combinations

Currently, Dynamo supports two hybrid transport modes:

| Request Plane | Response Plane | Status |
|--------------|----------------|---------|
| NATS         | TCP (call-home) | Legacy (default) |
| HTTP/2       | TCP (call-home) | Production-ready |

### Current Components

#### Request Plane (`lib/runtime/src/pipeline/network/request_plane.rs`)
- **Trait**: `RequestPlaneClient` - sends requests to workers
- **Trait**: `RequestPlaneServer` - receives requests from routers
- **Implementations**: `NatsClient`, `HttpRequestClient`

#### Response Plane (`lib/runtime/src/pipeline/network/tcp/`)
- **Call-home pattern**: Worker connects back to router with response stream
- **Components**: `TcpStreamServer` (router-side), `TcpClient` (worker-side)
- **Codec**: `TwoPartCodec` for framing messages

#### Service Discovery (`lib/runtime/src/component.rs`)
```rust
pub enum TransportType {
    NatsTcp(String),        // NATS request + TCP response
    HttpTcp {               // HTTP request + TCP response
        http_endpoint: String,
    },
}

pub struct Instance {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub instance_id: i64,
    pub transport: TransportType,
}
```

### Current Flow

**Request Flow (HTTP mode)**:
1. Router registers TCP response stream with `TcpStreamServer`
2. Router gets `ConnectionInfo` with TCP server address
3. Router sends HTTP POST with request + `ConnectionInfo`
4. Worker receives request, parses `ConnectionInfo`
5. Worker connects back to router's TCP server (call-home)
6. Worker streams responses over TCP connection

**Problem**: Each request creates a new TCP connection, wasting CPU cycles on 3-way handshakes.

---

## Transport Abstraction Interface

### Core Traits

#### 1. Transport Registration

```rust
/// Identifies a transport mechanism
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportId {
    Nats,
    Http,
    Tcp,
    Grpc,
    // Future: Quic, WebSocket, etc.
}

/// Identifies request vs response plane
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneType {
    Request,
    Response,
}

/// Transport capabilities and features
#[derive(Debug, Clone)]
pub struct TransportCapabilities {
    /// Supports streaming responses
    pub streaming: bool,
    /// Supports persistent connections
    pub persistent_connections: bool,
    /// Supports bidirectional communication
    pub bidirectional: bool,
    /// Maximum message size (None = unlimited)
    pub max_message_size: Option<usize>,
}

/// Transport registration information
pub struct TransportRegistration {
    pub transport_id: TransportId,
    pub plane_type: PlaneType,
    pub capabilities: TransportCapabilities,
    pub priority: u8,  // Lower = higher priority
}
```

#### 2. Request Plane Interface (existing, enhanced)

```rust
#[async_trait]
pub trait RequestPlaneClient: Send + Sync {
    /// Send a request with support for connection reuse
    async fn send_request(
        &self,
        address: &TransportAddress,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes>;

    /// Get transport capabilities
    fn capabilities(&self) -> &TransportCapabilities;

    /// Health check
    async fn is_healthy(&self) -> bool;
}

#[async_trait]
pub trait RequestPlaneServer: Send + Sync {
    async fn start(&mut self, bind_address: String) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    fn public_address(&self) -> String;
    fn capabilities(&self) -> &TransportCapabilities;
}
```

#### 3. Response Plane Interface (new)

```rust
/// Response plane client - sends responses back to requester
#[async_trait]
pub trait ResponsePlaneClient: Send + Sync {
    /// Create or reuse connection to send response stream
    async fn create_response_stream(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        connection_info: ResponseConnectionInfo,
    ) -> Result<ResponseStreamWriter>;

    /// Get transport capabilities
    fn capabilities(&self) -> &TransportCapabilities;

    /// Release/return connection to pool
    async fn release_connection(&self, connection_info: &ResponseConnectionInfo) -> Result<()>;
}

/// Response plane server - receives response streams
#[async_trait]
pub trait ResponsePlaneServer: Send + Sync {
    /// Register a new response stream endpoint
    async fn register_stream(
        &self,
        options: StreamOptions,
    ) -> Result<PendingResponseStream>;

    async fn start(&mut self, bind_address: String) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    fn public_address(&self) -> String;
    fn capabilities(&self) -> &TransportCapabilities;
}

/// Unified response stream writer interface
pub struct ResponseStreamWriter {
    inner: Box<dyn ResponseStreamWriterInner>,
    transport_id: TransportId,
}

#[async_trait]
trait ResponseStreamWriterInner: Send + Sync {
    async fn send(&mut self, data: Bytes) -> Result<()>;
    async fn send_control(&mut self, control: ControlMessage) -> Result<()>;
    async fn flush(&mut self) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
}
```

#### 4. Transport Registry

```rust
/// Central registry for all transport implementations
pub struct TransportRegistry {
    request_clients: HashMap<TransportId, Arc<dyn RequestPlaneClient>>,
    request_servers: HashMap<TransportId, Arc<dyn RequestPlaneServer>>,
    response_clients: HashMap<TransportId, Arc<dyn ResponsePlaneClient>>,
    response_servers: HashMap<TransportId, Arc<dyn ResponsePlaneServer>>,

    // Service discovery integration
    service_discovery: Arc<ServiceDiscovery>,
}

impl TransportRegistry {
    pub fn new(service_discovery: Arc<ServiceDiscovery>) -> Self;

    /// Register request plane transport
    pub fn register_request_transport(
        &mut self,
        id: TransportId,
        client: Arc<dyn RequestPlaneClient>,
        server: Arc<dyn RequestPlaneServer>,
    ) -> Result<()>;

    /// Register response plane transport
    pub fn register_response_transport(
        &mut self,
        id: TransportId,
        client: Arc<dyn ResponsePlaneClient>,
        server: Arc<dyn ResponsePlaneServer>,
    ) -> Result<()>;

    /// Get transport for endpoint based on priority and capabilities
    pub fn select_transport(
        &self,
        endpoint: &Endpoint,
        plane: PlaneType,
        required_capabilities: &TransportCapabilities,
    ) -> Result<TransportId>;

    /// Get client implementation
    pub fn get_request_client(&self, id: &TransportId) -> Option<Arc<dyn RequestPlaneClient>>;
    pub fn get_response_client(&self, id: &TransportId) -> Option<Arc<dyn ResponsePlaneClient>>;

    /// Get server implementation
    pub fn get_request_server(&self, id: &TransportId) -> Option<Arc<dyn RequestPlaneServer>>;
    pub fn get_response_server(&self, id: &TransportId) -> Option<Arc<dyn ResponsePlaneServer>>;
}
```

---

## Service Discovery Integration

### Enhanced Transport Type

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum TransportType {
    // Legacy formats (for backward compatibility)
    NatsTcp(String),
    HttpTcp { http_endpoint: String },

    // New unified format
    Unified {
        request_transport: TransportEndpoint,
        response_transport: TransportEndpoint,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct TransportEndpoint {
    pub transport_id: TransportId,
    pub address: String,
    pub capabilities: TransportCapabilities,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Instance {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub instance_id: i64,
    pub transport: TransportType,

    // New fields
    pub health_status: HealthStatus,
    pub last_heartbeat: Option<i64>,  // Unix timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
}
```

### Service Discovery Manager

```rust
pub struct ServiceDiscovery {
    etcd_client: Arc<EtcdClient>,
    registry: Arc<TransportRegistry>,
    instance_cache: Arc<RwLock<HashMap<EndpointId, Vec<Instance>>>>,
}

impl ServiceDiscovery {
    /// Register local endpoint with all configured transports
    pub async fn register_endpoint(
        &self,
        endpoint: &Endpoint,
        lease_id: i64,
    ) -> Result<()> {
        let transport_type = self.build_transport_type(endpoint).await?;

        let instance = Instance {
            component: endpoint.component.clone(),
            endpoint: endpoint.name.clone(),
            namespace: endpoint.namespace.clone(),
            instance_id: lease_id,
            transport: transport_type,
            health_status: HealthStatus::Healthy,
            last_heartbeat: Some(current_timestamp()),
        };

        let key = format!("{}/{:x}", INSTANCE_ROOT_PATH, lease_id);
        self.etcd_client.kv_create(&key, serde_json::to_vec(&instance)?, Some(lease_id)).await?;

        Ok(())
    }

    /// Discover instances for endpoint
    pub async fn discover_instances(
        &self,
        endpoint_id: &EndpointId,
    ) -> Result<Vec<Instance>> {
        // Check cache first
        if let Some(cached) = self.instance_cache.read().await.get(endpoint_id) {
            return Ok(cached.clone());
        }

        // Fetch from etcd
        let prefix = format!("{}/{}/{}/{}",
            INSTANCE_ROOT_PATH,
            endpoint_id.namespace,
            endpoint_id.component,
            endpoint_id.name
        );

        let instances = self.etcd_client.get_prefix(&prefix).await?
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_slice(&value).ok())
            .collect();

        self.instance_cache.write().await.insert(endpoint_id.clone(), instances.clone());

        Ok(instances)
    }

    /// Build transport type from registry and configuration
    async fn build_transport_type(&self, endpoint: &Endpoint) -> Result<TransportType> {
        let request_plane_mode = RequestPlaneMode::from_env();
        let response_plane_mode = ResponsePlaneMode::from_env();

        // Legacy compatibility mode
        if matches!(request_plane_mode, RequestPlaneMode::Http)
           && matches!(response_plane_mode, ResponsePlaneMode::Tcp) {
            let http_endpoint = self.build_http_endpoint(endpoint)?;
            return Ok(TransportType::HttpTcp { http_endpoint });
        }

        if matches!(request_plane_mode, RequestPlaneMode::Nats)
           && matches!(response_plane_mode, ResponsePlaneMode::Tcp) {
            return Ok(TransportType::NatsTcp(endpoint.subject.clone()));
        }

        // New unified format
        let request_transport = self.build_transport_endpoint(
            &request_plane_mode.into(),
            PlaneType::Request,
            endpoint
        ).await?;

        let response_transport = self.build_transport_endpoint(
            &response_plane_mode.into(),
            PlaneType::Response,
            endpoint
        ).await?;

        Ok(TransportType::Unified {
            request_transport,
            response_transport,
        })
    }

    async fn build_transport_endpoint(
        &self,
        transport_id: &TransportId,
        plane: PlaneType,
        endpoint: &Endpoint,
    ) -> Result<TransportEndpoint> {
        let (address, capabilities) = match (plane, transport_id) {
            (PlaneType::Request, TransportId::Http) => {
                let host = get_http_rpc_host_from_env();
                let port = get_http_rpc_port_from_env();
                let root = get_http_rpc_root_path_from_env();
                let addr = format!("http://{}:{}{}/{}", host, port, root, endpoint.subject);

                let caps = TransportCapabilities {
                    streaming: false,
                    persistent_connections: true,
                    bidirectional: false,
                    max_message_size: None,
                };
                (addr, caps)
            },
            (PlaneType::Response, TransportId::Tcp) => {
                let tcp_server = self.registry
                    .get_response_server(&TransportId::Tcp)
                    .ok_or_else(|| anyhow!("TCP response server not registered"))?;

                let addr = tcp_server.public_address();
                let caps = tcp_server.capabilities().clone();
                (addr, caps)
            },
            (PlaneType::Response, TransportId::Http) => {
                let host = get_http_response_host_from_env();
                let port = get_http_response_port_from_env();
                let root = get_http_response_root_path_from_env();
                let addr = format!("http://{}:{}{}/{}", host, port, root, endpoint.subject);

                let caps = TransportCapabilities {
                    streaming: true,
                    persistent_connections: true,
                    bidirectional: false,
                    max_message_size: None,
                };
                (addr, caps)
            },
            _ => return Err(anyhow!("Unsupported transport combination")),
        };

        Ok(TransportEndpoint {
            transport_id: transport_id.clone(),
            address,
            capabilities,
            metadata: HashMap::new(),
        })
    }
}
```

---

## Priority 0: TCP Connection Pooling

### Problem Statement

Currently, each request-response cycle creates a new TCP connection using the "call-home" pattern:
1. Router registers response stream endpoint
2. Worker receives request with `ConnectionInfo`
3. Worker creates **new TCP connection** to router
4. Worker sends responses, closes connection

This wastes CPU cycles on:
- 3-way TCP handshake (SYN, SYN-ACK, ACK)
- 4-way connection teardown (FIN, ACK, FIN, ACK)
- TCP slow start for each connection

### Solution: Persistent Connection Pool

#### Architecture

```rust
/// Connection pool for TCP response streams
pub struct TcpResponseConnectionPool {
    pools: HashMap<String, Arc<ConnectionSubPool>>,
    config: PoolConfig,
}

pub struct PoolConfig {
    /// Maximum connections per target
    pub max_connections_per_target: usize,
    /// Minimum idle connections to maintain
    pub min_idle_connections: usize,
    /// Connection idle timeout
    pub idle_timeout: Duration,
    /// Maximum connection lifetime
    pub max_lifetime: Duration,
    /// Enable TCP keepalive
    pub tcp_keepalive: Option<Duration>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_target: 50,
            min_idle_connections: 5,
            idle_timeout: Duration::from_secs(300),  // 5 minutes
            max_lifetime: Duration::from_secs(3600), // 1 hour
            tcp_keepalive: Some(Duration::from_secs(60)),
        }
    }
}

struct ConnectionSubPool {
    target: String,
    idle_connections: Arc<Mutex<VecDeque<PooledConnection>>>,
    active_connections: Arc<AtomicUsize>,
    config: PoolConfig,
}

struct PooledConnection {
    stream: TcpStream,
    created_at: Instant,
    last_used: Instant,
    use_count: usize,
}
```

#### Implementation

```rust
impl TcpResponseConnectionPool {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            pools: HashMap::new(),
            config,
        }
    }

    /// Get or create connection to target
    pub async fn acquire(
        &self,
        target: &str,
        context_id: &str,
    ) -> Result<PooledTcpStream> {
        let pool = self.get_or_create_pool(target);

        // Try to reuse idle connection
        if let Some(mut conn) = pool.pop_idle().await {
            if self.validate_connection(&mut conn).await {
                tracing::debug!(target, "Reusing pooled TCP connection");
                return Ok(PooledTcpStream::new(conn, pool.clone()));
            }
        }

        // Create new connection
        tracing::debug!(target, "Creating new TCP connection");
        let stream = self.create_connection(target, context_id).await?;
        let conn = PooledConnection {
            stream,
            created_at: Instant::now(),
            last_used: Instant::now(),
            use_count: 0,
        };

        pool.active_connections.fetch_add(1, Ordering::SeqCst);

        Ok(PooledTcpStream::new(conn, pool.clone()))
    }

    async fn create_connection(
        &self,
        target: &str,
        context_id: &str,
    ) -> Result<TcpStream> {
        let stream = TcpStream::connect(target).await?;
        stream.set_nodelay(true)?;

        if let Some(keepalive) = self.config.tcp_keepalive {
            let socket = socket2::Socket::from(stream.into_std()?);
            socket.set_keepalive(true)?;
            #[cfg(unix)]
            {
                socket.set_tcp_keepalive_idle(keepalive)?;
                socket.set_tcp_keepalive_interval(keepalive / 3)?;
                socket.set_tcp_keepalive_retries(3)?;
            }
            TcpStream::from_std(socket.into())
        } else {
            Ok(stream)
        }
    }

    async fn validate_connection(&self, conn: &mut PooledConnection) -> bool {
        let now = Instant::now();

        // Check age
        if now.duration_since(conn.created_at) > self.config.max_lifetime {
            tracing::debug!("Connection expired (max lifetime)");
            return false;
        }

        // Check idle time
        if now.duration_since(conn.last_used) > self.config.idle_timeout {
            tracing::debug!("Connection expired (idle timeout)");
            return false;
        }

        // TCP-level health check (try to peek)
        let mut buf = [0u8; 1];
        match conn.stream.try_read(&mut buf) {
            Ok(0) => {
                tracing::debug!("Connection closed by remote");
                false
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Connection is alive and has no data
                true
            }
            Err(_) => {
                tracing::debug!("Connection health check failed");
                false
            }
            Ok(_) => {
                // Unexpected data, connection may be in bad state
                tracing::warn!("Unexpected data on idle connection");
                false
            }
        }
    }

    fn get_or_create_pool(&self, target: &str) -> Arc<ConnectionSubPool> {
        self.pools
            .entry(target.to_string())
            .or_insert_with(|| {
                Arc::new(ConnectionSubPool {
                    target: target.to_string(),
                    idle_connections: Arc::new(Mutex::new(VecDeque::new())),
                    active_connections: Arc::new(AtomicUsize::new(0)),
                    config: self.config.clone(),
                })
            })
            .clone()
    }
}

/// RAII wrapper that returns connection to pool on drop
pub struct PooledTcpStream {
    connection: Option<PooledConnection>,
    pool: Arc<ConnectionSubPool>,
}

impl PooledTcpStream {
    fn new(connection: PooledConnection, pool: Arc<ConnectionSubPool>) -> Self {
        Self {
            connection: Some(connection),
            pool,
        }
    }

    pub fn stream(&mut self) -> &mut TcpStream {
        &mut self.connection.as_mut().unwrap().stream
    }
}

impl Drop for PooledTcpStream {
    fn drop(&mut self) {
        if let Some(mut conn) = self.connection.take() {
            conn.last_used = Instant::now();
            conn.use_count += 1;

            self.pool.active_connections.fetch_sub(1, Ordering::SeqCst);

            // Return to pool if still healthy
            if self.is_connection_reusable(&conn) {
                let mut idle = self.pool.idle_connections.blocking_lock();
                if idle.len() < self.pool.config.max_connections_per_target {
                    idle.push_back(conn);
                    tracing::debug!(target = %self.pool.target, "Returned connection to pool");
                }
            }
        }
    }

    fn is_connection_reusable(&self, conn: &PooledConnection) -> bool {
        Instant::now().duration_since(conn.created_at) < self.pool.config.max_lifetime
    }
}

impl ConnectionSubPool {
    async fn pop_idle(&self) -> Option<PooledConnection> {
        let mut idle = self.idle_connections.lock().await;
        idle.pop_front()
    }
}
```

#### Integration with Existing Code

**Worker Side** (`TcpClient`):
```rust
// Replace in lib/runtime/src/pipeline/network/tcp/client.rs

pub struct TcpClient {
    worker_id: String,
    connection_pool: Arc<TcpResponseConnectionPool>,
}

impl TcpClient {
    pub fn new(worker_id: String) -> Self {
        Self {
            worker_id,
            connection_pool: Arc::new(TcpResponseConnectionPool::new(PoolConfig::default())),
        }
    }

    pub async fn create_response_stream(
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
    ) -> Result<StreamSender> {
        let info = TcpStreamConnectionInfo::try_from(info)?;

        // Acquire pooled connection instead of creating new one
        let mut pooled_stream = CLIENT_POOL
            .acquire(&info.address, &info.context)
            .await?;

        // Perform handshake on pooled connection
        let handshake = CallHomeHandshake {
            subject: info.subject.clone(),
            stream_type: StreamType::Response,
        };

        // ... rest of existing logic using pooled_stream
    }
}

// Global pool instance
lazy_static! {
    static ref CLIENT_POOL: Arc<TcpResponseConnectionPool> = {
        let config = PoolConfig {
            max_connections_per_target: std::env::var("DYN_TCP_POOL_MAX_PER_TARGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
            min_idle_connections: 5,
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(3600),
            tcp_keepalive: Some(Duration::from_secs(60)),
        };
        Arc::new(TcpResponseConnectionPool::new(config))
    };
}
```

**Router Side** (`TcpStreamServer`):
```rust
// Update in lib/runtime/src/pipeline/network/tcp/server.rs

impl TcpStreamServer {
    // Server must handle persistent connections:
    // - Keep socket alive after first response
    // - Match incoming handshakes to waiting response streams
    // - Implement connection keepalive

    async fn handle_persistent_connection(
        socket: TcpStream,
        state: Arc<Mutex<State>>,
    ) -> Result<()> {
        let (read_half, write_half) = tokio::io::split(socket);
        let mut reader = FramedRead::new(read_half, TwoPartCodec::default());

        // Connection can be reused for multiple response streams
        loop {
            // Read handshake for next response stream
            match reader.next().await {
                Some(Ok(msg)) if msg.msg_type == TwoPartMessageType::Header => {
                    let handshake: CallHomeHandshake = serde_json::from_slice(&msg.data)?;

                    // Match to waiting response stream
                    if let Some(requested) = state.lock().await
                        .rx_subjects.remove(&handshake.subject)
                    {
                        // Create stream sender for this response cycle
                        let (tx, rx) = mpsc::channel(64);
                        let sender = StreamSender { tx, prologue: Some(ResponseStreamPrologue { error: None }) };

                        requested.connection.send(Ok(StreamReceiver { rx })).ok();

                        // Handle this response stream
                        self.handle_response_stream(&mut reader, write_half, sender).await?;

                        // Connection can be reused after this response completes
                        tracing::debug!("Response stream complete, connection ready for reuse");
                    } else {
                        tracing::warn!("No matching response stream for handshake");
                        break;
                    }
                }
                Some(Ok(_)) => {
                    tracing::warn!("Expected handshake header, got data");
                    break;
                }
                Some(Err(e)) => {
                    tracing::error!("Error reading from persistent connection: {}", e);
                    break;
                }
                None => {
                    tracing::debug!("Connection closed by peer");
                    break;
                }
            }
        }

        Ok(())
    }
}
```

#### Benefits

1. **Performance**:
   - Eliminates 3-way handshake overhead (~1-2ms per request)
   - Eliminates TCP slow start
   - Reduces kernel overhead from socket creation/destruction
   - Expected: **5-10% latency reduction** for p50, **10-20% for p99**

2. **Resource Efficiency**:
   - Fewer file descriptors
   - Less kernel memory for connection tracking
   - Reduced TIME_WAIT socket accumulation

3. **Scalability**:
   - Support higher request throughput
   - Better behavior under burst load

#### Configuration

Environment variables:
- `DYN_TCP_POOL_MAX_PER_TARGET`: Max connections per target (default: 50)
- `DYN_TCP_POOL_MIN_IDLE`: Min idle connections (default: 5)
- `DYN_TCP_POOL_IDLE_TIMEOUT_SECS`: Idle timeout in seconds (default: 300)
- `DYN_TCP_POOL_MAX_LIFETIME_SECS`: Max connection lifetime (default: 3600)
- `DYN_TCP_POOL_KEEPALIVE_SECS`: TCP keepalive interval (default: 60)

---

## Priority 1: HTTP Response Path

### Motivation

HTTP-based response streaming provides:
- Standard SSE (Server-Sent Events) or HTTP/2 streaming
- Better compatibility with proxies/load balancers
- No "call-home" pattern complexity
- Native HTTP/2 multiplexing

### Architecture

#### Response Flow

```
[Router]
  ├─ POST /v1/rpc/{subject} (request)
  │   Headers: Accept: text/event-stream (or application/x-ndjson)
  └─ Receive streaming response

[Worker]
  ├─ Receive POST request
  ├─ Process request
  └─ Stream responses as SSE or newline-delimited JSON
```

#### Implementation

**Response Server** (worker-side):
```rust
pub struct HttpResponseServer {
    router: axum::Router,
    bind_address: String,
    public_address: String,
    handler: Arc<dyn PushWorkHandler>,
}

impl HttpResponseServer {
    pub fn new(handler: Arc<dyn PushWorkHandler>) -> Self {
        let router = Router::new()
            .route("/{subject}", post(handle_streaming_request))
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(TimeoutLayer::new(Duration::from_secs(3600)))
            );

        Self {
            router,
            bind_address: "0.0.0.0:8081".to_string(),
            public_address: get_http_response_host_from_env(),
            handler,
        }
    }
}

async fn handle_streaming_request(
    Path(subject): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Check Accept header
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream");

    let format = if accept.contains("text/event-stream") {
        StreamFormat::ServerSentEvents
    } else if accept.contains("application/x-ndjson") {
        StreamFormat::NewlineDelimitedJson
    } else {
        StreamFormat::Binary
    };

    // Create response stream
    let (tx, rx) = mpsc::channel(64);

    // Spawn handler
    tokio::spawn(async move {
        // Decode and process request
        let handler = /* get handler */;
        if let Err(e) = handler.handle_payload(body).await {
            let _ = tx.send(Err(e)).await;
        }
    });

    // Convert rx to response stream
    let stream = ReceiverStream::new(rx);
    let body = StreamBody::new(stream.map(|item| {
        match format {
            StreamFormat::ServerSentEvents => {
                format_as_sse(item)
            }
            StreamFormat::NewlineDelimitedJson => {
                format_as_ndjson(item)
            }
            StreamFormat::Binary => {
                format_as_binary(item)
            }
        }
    }));

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, format.content_type())],
        body,
    ))
}

enum StreamFormat {
    ServerSentEvents,
    NewlineDelimitedJson,
    Binary,
}

impl StreamFormat {
    fn content_type(&self) -> &'static str {
        match self {
            Self::ServerSentEvents => "text/event-stream",
            Self::NewlineDelimitedJson => "application/x-ndjson",
            Self::Binary => "application/octet-stream",
        }
    }
}

fn format_as_sse(item: Result<Bytes, PipelineError>) -> Result<Bytes, std::io::Error> {
    match item {
        Ok(data) => {
            let encoded = base64::encode(&data);
            Ok(format!("data: {}\n\n", encoded).into())
        }
        Err(e) => {
            Ok(format!("event: error\ndata: {}\n\n", e).into())
        }
    }
}
```

**Response Client** (router-side):
```rust
pub struct HttpResponseClient {
    client: reqwest::Client,
    connection_pool: Arc<Http2ConnectionPool>,
}

#[async_trait]
impl ResponsePlaneClient for HttpResponseClient {
    async fn create_response_stream(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        connection_info: ResponseConnectionInfo,
    ) -> Result<ResponseStreamWriter> {
        let response = self.client
            .post(&connection_info.http_endpoint)
            .header(header::ACCEPT, "text/event-stream")
            .body(connection_info.request_payload)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("HTTP response failed: {}", response.status()));
        }

        // Parse SSE stream
        let stream = response.bytes_stream();
        let receiver = parse_sse_stream(stream);

        Ok(ResponseStreamWriter {
            inner: Box::new(HttpResponseStreamWriter { receiver }),
            transport_id: TransportId::Http,
        })
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

struct HttpResponseStreamWriter {
    receiver: mpsc::Receiver<Result<Bytes, PipelineError>>,
}

#[async_trait]
impl ResponseStreamWriterInner for HttpResponseStreamWriter {
    async fn send(&mut self, data: Bytes) -> Result<()> {
        // For HTTP responses, reading from receiver, not sending
        // This trait may need redesign for unidirectional transports
        unimplemented!("HTTP response is read-only from client perspective")
    }
}
```

#### Benefits

1. **Simplicity**: No call-home pattern complexity
2. **Standardization**: Use HTTP/2 standard streaming
3. **Compatibility**: Works with standard HTTP infrastructure
4. **Debugging**: Can use standard HTTP tools (curl, Postman, etc.)

#### Trade-offs

- **Latency**: Slightly higher than direct TCP (HTTP framing overhead)
- **Connection Management**: Must implement HTTP/2 connection pooling
- **Error Handling**: HTTP error model different from TCP

---

## Priority 2: TCP Request Path

### Motivation

TCP request path provides:
- Lower latency than HTTP (no HTTP framing)
- Persistent connections with multiplexing
- Bidirectional streaming support
- Better for high-throughput scenarios

### Architecture

#### Request Flow

```
[Router]
  ├─ Persistent TCP connection to worker
  ├─ Multiplex multiple requests over single connection
  └─ Receive response streams on same connection

[Worker]
  ├─ Accept persistent TCP connections
  ├─ Demultiplex requests by request_id
  └─ Stream responses on same connection
```

#### Implementation

**Protocol Design**:
```rust
/// Frame header for TCP request/response multiplexing
#[derive(Debug, Serialize, Deserialize)]
struct TcpFrameHeader {
    /// Unique request identifier
    request_id: Uuid,
    /// Frame type
    frame_type: TcpFrameType,
    /// Payload length
    payload_length: u64,
    /// Stream end flag
    end_stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
enum TcpFrameType {
    Request,
    Response,
    Control,
    Heartbeat,
}

/// Codec for framing TCP messages
pub struct TcpMultiplexCodec {
    state: CodecState,
}

enum CodecState {
    Header,
    Payload { header: TcpFrameHeader, remaining: usize },
}

impl Decoder for TcpMultiplexCodec {
    type Item = TcpFrame;
    type Error = std::io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            match &mut self.state {
                CodecState::Header => {
                    // Need at least header size
                    if src.len() < 48 {  // UUID(16) + type(1) + length(8) + flag(1) + padding
                        return Ok(None);
                    }

                    let header = self.decode_header(src)?;
                    self.state = CodecState::Payload {
                        header,
                        remaining: header.payload_length as usize,
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
```

**Request Client** (router-side):
```rust
pub struct TcpRequestClient {
    connections: Arc<DashMap<String, Arc<TcpMultiplexConnection>>>,
    pool_config: PoolConfig,
}

struct TcpMultiplexConnection {
    writer: Arc<Mutex<FramedWrite<WriteHalf<TcpStream>, TcpMultiplexCodec>>>,
    pending_responses: Arc<DashMap<Uuid, oneshot::Sender<mpsc::Receiver<TcpFrame>>>>,
    health: Arc<AtomicBool>,
}

impl TcpMultiplexConnection {
    async fn send_request(
        &self,
        request_id: Uuid,
        payload: Bytes,
    ) -> Result<mpsc::Receiver<TcpFrame>> {
        let frame = TcpFrame {
            header: TcpFrameHeader {
                request_id,
                frame_type: TcpFrameType::Request,
                payload_length: payload.len() as u64,
                end_stream: true,
            },
            payload,
        };

        // Register response handler
        let (tx, rx) = oneshot::channel();
        self.pending_responses.insert(request_id, tx);

        // Send request
        let mut writer = self.writer.lock().await;
        writer.send(frame).await?;

        // Wait for response stream
        Ok(rx.await?)
    }

    async fn handle_reader(
        reader: FramedRead<ReadHalf<TcpStream>, TcpMultiplexCodec>,
        pending: Arc<DashMap<Uuid, oneshot::Sender<mpsc::Receiver<TcpFrame>>>>,
    ) {
        let mut active_streams: HashMap<Uuid, mpsc::Sender<TcpFrame>> = HashMap::new();

        let mut reader = reader;
        while let Some(Ok(frame)) = reader.next().await {
            match frame.header.frame_type {
                TcpFrameType::Response => {
                    // Get or create response stream
                    if let Some(tx) = active_streams.get(&frame.header.request_id) {
                        let _ = tx.send(frame).await;

                        if frame.header.end_stream {
                            active_streams.remove(&frame.header.request_id);
                        }
                    } else if let Some((_, response_tx)) = pending.remove(&frame.header.request_id) {
                        let (tx, rx) = mpsc::channel(64);
                        let _ = response_tx.send(rx);
                        let _ = tx.send(frame).await;

                        if !frame.header.end_stream {
                            active_streams.insert(frame.header.request_id, tx);
                        }
                    }
                }
                TcpFrameType::Heartbeat => {
                    // Respond to heartbeat
                }
                _ => {}
            }
        }
    }
}

#[async_trait]
impl RequestPlaneClient for TcpRequestClient {
    async fn send_request(
        &self,
        address: &TransportAddress,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes> {
        let conn = self.get_or_create_connection(&address.tcp_address).await?;
        let request_id = Uuid::new_v4();

        let mut response_stream = conn.send_request(request_id, payload).await?;

        // For non-streaming, collect all responses
        let mut result = BytesMut::new();
        while let Some(frame) = response_stream.recv().await {
            result.extend_from_slice(&frame.payload);
            if frame.header.end_stream {
                break;
            }
        }

        Ok(result.freeze())
    }
}
```

**Request Server** (worker-side):
```rust
pub struct TcpRequestServer {
    listener: TcpListener,
    bind_address: String,
    public_address: String,
    handler: Arc<dyn PushWorkHandler>,
}

impl TcpRequestServer {
    async fn handle_connection(&self, socket: TcpStream) -> Result<()> {
        let (read_half, write_half) = tokio::io::split(socket);
        let mut reader = FramedRead::new(read_half, TcpMultiplexCodec::default());
        let writer = Arc::new(Mutex::new(FramedWrite::new(write_half, TcpMultiplexCodec::default())));

        while let Some(Ok(frame)) = reader.next().await {
            match frame.header.frame_type {
                TcpFrameType::Request => {
                    let request_id = frame.header.request_id;
                    let handler = self.handler.clone();
                    let writer = writer.clone();

                    tokio::spawn(async move {
                        // Process request
                        match handler.handle_payload(frame.payload).await {
                            Ok(response) => {
                                // Send response frame
                                let response_frame = TcpFrame {
                                    header: TcpFrameHeader {
                                        request_id,
                                        frame_type: TcpFrameType::Response,
                                        payload_length: response.len() as u64,
                                        end_stream: true,
                                    },
                                    payload: response,
                                };

                                let mut w = writer.lock().await;
                                let _ = w.send(response_frame).await;
                            }
                            Err(e) => {
                                tracing::error!("Error handling request: {}", e);
                            }
                        }
                    });
                }
                _ => {}
            }
        }

        Ok(())
    }
}
```

#### Benefits

1. **Performance**: Lowest latency option
2. **Efficiency**: Single persistent connection, multiplexing
3. **Bidirectional**: Support for streaming requests and responses
4. **Control**: Full control over protocol and framing

#### Trade-offs

- **Complexity**: Custom protocol implementation
- **Debugging**: Harder to debug than HTTP
- **Infrastructure**: May not work through HTTP-only proxies

---

## Migration Strategy

### Phase 0: TCP Connection Pooling (Immediate)

**Goal**: Improve performance of existing HTTP+TCP and NATS+TCP modes

**Steps**:
1. Implement `TcpResponseConnectionPool`
2. Update `TcpClient` to use connection pool
3. Update `TcpStreamServer` to support persistent connections
4. Add metrics for pool utilization
5. Deploy behind feature flag `DYN_TCP_POOL_ENABLED`
6. Gradual rollout with A/B testing

**Validation**:
- Monitor connection creation rate (should decrease)
- Measure latency improvements (p50, p95, p99)
- Check for connection leaks or pool exhaustion

### Phase 1: HTTP Response Path (3-6 months)

**Goal**: Provide pure HTTP/2 option (HTTP request + HTTP response)

**Steps**:
1. Implement `HttpResponseServer` and `HttpResponseClient`
2. Add `ResponsePlaneMode::Http` configuration
3. Update service discovery to support `HttpHttp` transport type
4. Implement SSE and NDJSON streaming formats
5. Deploy alongside existing TCP response path
6. Migrate low-traffic endpoints first
7. Monitor and optimize

**Validation**:
- Compare latency with TCP response path
- Validate streaming correctness
- Test with various HTTP clients
- Measure CPU and memory overhead

### Phase 2: TCP Request Path (6-12 months)

**Goal**: Provide low-latency TCP option (TCP request + TCP response)

**Steps**:
1. Design and implement `TcpMultiplexCodec`
2. Implement `TcpRequestClient` and `TcpRequestServer`
3. Add `RequestPlaneMode::Tcp` configuration
4. Implement connection multiplexing and pooling
5. Add heartbeat and health checking
6. Deploy for internal high-throughput endpoints
7. Gradual rollout to production

**Validation**:
- Compare latency with HTTP request path
- Test multiplexing correctness under load
- Validate connection recovery and failover
- Measure throughput improvements

### Backward Compatibility

All phases maintain backward compatibility:

```rust
// Service discovery supports all modes
match instance.transport {
    TransportType::NatsTcp(subject) => {
        // Legacy NATS + TCP (Phase 0 improvement)
    }
    TransportType::HttpTcp { http_endpoint } => {
        // Current HTTP + TCP (Phase 0 improvement)
    }
    TransportType::HttpHttp { request_endpoint, response_endpoint } => {
        // Phase 1: HTTP + HTTP
    }
    TransportType::TcpTcp { endpoint } => {
        // Phase 2: TCP + TCP
    }
    TransportType::Unified { request_transport, response_transport } => {
        // Future: Fully flexible transport selection
    }
}
```

---

## Configuration

### Environment Variables

#### Connection Pooling (Priority 0)
- `DYN_TCP_POOL_ENABLED`: Enable TCP connection pooling (default: `false`)
- `DYN_TCP_POOL_MAX_PER_TARGET`: Max connections per target (default: `50`)
- `DYN_TCP_POOL_MIN_IDLE`: Min idle connections (default: `5`)
- `DYN_TCP_POOL_IDLE_TIMEOUT_SECS`: Idle timeout (default: `300`)
- `DYN_TCP_POOL_MAX_LIFETIME_SECS`: Max connection lifetime (default: `3600`)
- `DYN_TCP_POOL_KEEPALIVE_SECS`: TCP keepalive interval (default: `60`)

#### HTTP Response Path (Priority 1)
- `DYN_RESPONSE_PLANE`: `tcp` (default) or `http`
- `DYN_HTTP_RESPONSE_HOST`: Public hostname for HTTP response server
- `DYN_HTTP_RESPONSE_PORT`: Port for HTTP response server (default: `8082`)
- `DYN_HTTP_RESPONSE_ROOT_PATH`: Root path for HTTP response API (default: `/v1/rpc/response`)
- `DYN_HTTP_RESPONSE_STREAM_FORMAT`: `sse`, `ndjson`, or `binary` (default: `sse`)
- `DYN_HTTP_RESPONSE_TIMEOUT_SECS`: Response stream timeout (default: `3600`)

#### TCP Request Path (Priority 2)
- `DYN_REQUEST_PLANE`: `nats`, `http` (current), or `tcp` (future)
- `DYN_TCP_REQUEST_HOST`: Bind address for TCP request server (default: `0.0.0.0`)
- `DYN_TCP_REQUEST_PORT`: Port for TCP request server (default: `9001`)
- `DYN_TCP_REQUEST_MAX_CONNECTIONS`: Max concurrent connections (default: `1000`)
- `DYN_TCP_REQUEST_HEARTBEAT_INTERVAL_SECS`: Heartbeat interval (default: `30`)
- `DYN_TCP_REQUEST_CONNECTION_TIMEOUT_SECS`: Connection timeout (default: `60`)

### Configuration File (Future)

```yaml
# dynamo-transport.yaml
transport:
  request_plane:
    mode: http  # nats, http, tcp
    http:
      host: 0.0.0.0
      port: 8081
      root_path: /v1/rpc
      timeout_secs: 5
    tcp:
      host: 0.0.0.0
      port: 9001
      max_connections: 1000
      heartbeat_interval_secs: 30

  response_plane:
    mode: tcp  # tcp, http
    tcp:
      connection_pool:
        enabled: true
        max_per_target: 50
        min_idle: 5
        idle_timeout_secs: 300
        max_lifetime_secs: 3600
        keepalive_secs: 60
    http:
      host: 0.0.0.0
      port: 8082
      root_path: /v1/rpc/response
      stream_format: sse  # sse, ndjson, binary
      timeout_secs: 3600

  service_discovery:
    etcd:
      endpoints:
        - http://localhost:2379
      lease_ttl_secs: 60
      watch_enabled: true
```

---

## Metrics and Observability

### Connection Pool Metrics

```rust
pub struct ConnectionPoolMetrics {
    /// Total connections created
    pub connections_created: Counter,
    /// Total connections reused from pool
    pub connections_reused: Counter,
    /// Current active connections
    pub active_connections: Gauge,
    /// Current idle connections
    pub idle_connections: Gauge,
    /// Connection acquisition time
    pub acquisition_latency: Histogram,
    /// Connection validation failures
    pub validation_failures: Counter,
}
```

### Transport Metrics

```rust
pub struct TransportMetrics {
    /// Requests sent by transport type
    pub requests_sent: CounterVec,  // labels: transport_type
    /// Request latency by transport type
    pub request_latency: HistogramVec,  // labels: transport_type
    /// Transport errors
    pub transport_errors: CounterVec,  // labels: transport_type, error_type
    /// Active request streams
    pub active_streams: GaugeVec,  // labels: transport_type
}
```

### Service Discovery Metrics

```rust
pub struct ServiceDiscoveryMetrics {
    /// Endpoint registrations
    pub registrations: Counter,
    /// Endpoint discoveries
    pub discoveries: Counter,
    /// Etcd operation latency
    pub etcd_latency: Histogram,
    /// Cache hits/misses
    pub cache_hits: Counter,
    pub cache_misses: Counter,
}
```

---

## Testing Strategy

### Unit Tests

1. **Connection Pool**:
   - Connection acquisition and release
   - Pool size limits
   - Connection validation
   - Idle timeout handling
   - Max lifetime enforcement

2. **Transport Implementations**:
   - Client send/receive
   - Server start/stop
   - Error handling
   - Capability reporting

3. **Service Discovery**:
   - Endpoint registration
   - Instance discovery
   - Cache management
   - Transport type serialization

### Integration Tests

1. **End-to-End Flow**:
   - Request → Response with different transport combinations
   - Connection reuse validation
   - Failover scenarios
   - Load distribution

2. **Migration Scenarios**:
   - Mixed transport types in same deployment
   - Gradual rollout simulation
   - Backward compatibility

### Performance Tests

1. **Connection Pool**:
   - Pool exhaustion under high load
   - Connection reuse rate
   - Latency improvements vs. baseline

2. **Transport Comparison**:
   - Latency: TCP < HTTP
   - Throughput: all transports
   - CPU/memory overhead

3. **Scale Tests**:
   - 1000+ concurrent connections
   - High request rate (10K+ RPS)
   - Connection pool efficiency

---

## Summary

This design provides a flexible, extensible transport architecture for Dynamo:

**Priority 0 (Immediate)**: TCP connection pooling delivers immediate performance improvements (5-20% latency reduction) with minimal changes to existing architecture.

**Priority 1 (3-6 months)**: HTTP response path simplifies deployment and improves compatibility at the cost of slight latency overhead.

**Priority 2 (6-12 months)**: TCP request path provides the ultimate performance option for high-throughput internal services.

All phases maintain backward compatibility and allow gradual migration. The `TransportRegistry` and service discovery integration enable flexible transport selection based on deployment requirements and performance characteristics.


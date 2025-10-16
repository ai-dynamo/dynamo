# Dynamo Network Layer Architecture

## Overview

The Dynamo network layer provides a comprehensive, high-performance networking infrastructure designed for distributed AI/ML workloads. It implements a sophisticated multi-transport architecture supporting both NATS and HTTP/2 for request routing, with TCP streaming for response data. The system is built with fault tolerance, observability, and scalability as core principles.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DYNAMO NETWORK LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   INGRESS   │    │ REQUEST PLANE│    │     EGRESS      │    │
│  │             │    │              │    │                 │    │
│  │ HTTP/NATS   │◄──►│ NATS / HTTP/2│◄──►│   ROUTERS       │    │
│  │ Endpoints   │    │              │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│         │                   │                      │           │
│         ▼                   ▼                      ▼           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │    CODEC    │    │     TCP      │    │     CODEC       │    │
│  │             │    │   TRANSPORT  │    │                 │    │
│  │ TwoPartCodec│    │              │    │  TwoPartCodec   │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Request Plane (`request_plane.rs`)

The request plane provides transport-agnostic abstractions for sending requests between services.

#### Key Traits

**`RequestPlaneClient`** - Sends requests to workers
```rust
pub trait RequestPlaneClient: Send + Sync {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes>;
}
```

**`RequestPlaneServer`** - Receives requests from routers
```rust
pub trait RequestPlaneServer: Send + Sync {
    async fn start(&mut self, bind_address: String) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    fn public_address(&self) -> String;
}
```

**Key Features:**
- Transport abstraction (NATS/HTTP/2)
- Header-based tracing and metadata
- Acknowledgment-based reliability
- Service discovery integration

### 2. Codec System (`codec/`)

#### TwoPartCodec (`codec/two_part.rs`)

A high-performance binary protocol for structured message encoding with integrity checking.

**Message Structure:**
```
┌─────────────┬─────────────┬──────────┬─────────────┬──────────────┐
│ header_len  │ body_len    │ checksum │   header    │     body     │
│   (8 bytes) │  (8 bytes)  │(8 bytes) │ (variable)  │  (variable)  │
└─────────────┴─────────────┴──────────┴─────────────┴──────────────┘
```

**Key Features:**
- **Integrity Checking**: XXHash-based checksums (debug mode)
- **Size Limits**: Configurable maximum message sizes
- **Streaming Support**: Compatible with Tokio codec framework
- **Zero-Copy**: Efficient `Bytes` usage for minimal allocations
- **Partial Read Handling**: Robust handling of incomplete network data

**Message Types:**
```rust
pub enum TwoPartMessageType {
    HeaderOnly(Bytes),     // Control messages
    DataOnly(Bytes),       // Pure data
    HeaderAndData(Bytes, Bytes), // Request/response with metadata
    Empty,                 // Heartbeat/keepalive
}
```

### 3. TCP Transport (`tcp/`)

#### TCP Server (`tcp/server.rs`)

A high-performance TCP server supporting call-home connections and stream multiplexing.

**Key Features:**
- **Call-Home Architecture**: Clients connect back to establish data streams
- **Stream Multiplexing**: Multiple logical streams over single connections
- **Graceful Shutdown**: Proper connection lifecycle management
- **IP Resolution**: Automatic network interface detection with fallbacks
- **Connection Pooling**: Efficient resource management

**Connection Flow:**
1. Client connects to server
2. Handshake message identifies stream purpose
3. Prologue indicates readiness/errors
4. Data streaming begins
5. Sentinel message signals completion

#### TCP Client (`tcp/client.rs`)

Establishes response streams back to servers with robust error handling.

**Features:**
- **Automatic Retry**: Linear backoff for connection failures
- **Context Integration**: Lifecycle tied to execution context
- **Flow Control**: Backpressure-aware streaming
- **Clean Shutdown**: Proper resource cleanup

### 4. Egress Layer (`egress/`)

The egress layer handles outbound requests with intelligent routing and fault tolerance.

#### AddressedPushRouter (`egress/addressed_router.rs`)

Core router supporting multiple transport modes with fault detection.

**Transport Modes:**
- **NATS**: Message queue-based routing
- **HTTP/2**: Direct HTTP communication with multiplexing

**Routing Strategies:**
```rust
pub enum RouterMode {
    RoundRobin,        // Distribute evenly across instances
    Random,            // Random selection for load balancing
    Direct(i64),       // Target specific instance
    KV,                // Key-value based routing (external)
}
```

**Fault Detection:**
- Instance health monitoring
- Automatic failover
- Circuit breaker patterns
- Distributed tracing integration

#### HTTP Router (`egress/http_router.rs`)

HTTP/2-specific implementation with advanced features.

**Features:**
- **HTTP/2 Prior Knowledge**: No protocol negotiation overhead
- **Connection Pooling**: Efficient connection reuse (50 connections/host)
- **Concurrent Requests**: Full HTTP/2 multiplexing support
- **Timeout Management**: Configurable request timeouts
- **Header Propagation**: Automatic tracing header forwarding

**Performance Optimizations:**
- Connection pooling with configurable limits
- HTTP/2 multiplexing for concurrent requests
- Efficient binary payload handling
- Minimal serialization overhead

#### PushRouter (`egress/push_router.rs`)

High-level router with service discovery and load balancing.

**Features:**
- **Service Discovery**: etcd-based instance management
- **Load Balancing**: Multiple routing algorithms
- **Health Monitoring**: Worker busy detection and overload protection
- **Metrics Integration**: Comprehensive observability

**Worker Monitoring:**
- Busy threshold detection
- Automatic instance filtering
- Service overload protection
- Real-time health updates

### 5. Ingress Layer (`ingress/`)

The ingress layer handles inbound requests with protocol-specific endpoints.

#### HTTP Endpoint (`ingress/http_endpoint.rs`)

HTTP/2 server implementation with shared port multiplexing.

**SharedHttpServer Features:**
- **Port Multiplexing**: Multiple endpoints on single port
- **HTTP/2 Support**: Full HTTP/2 with prior knowledge
- **Graceful Shutdown**: Proper inflight request handling
- **Health Integration**: Automatic health status management
- **Tracing Integration**: OpenTelemetry header extraction

**Request Processing:**
1. HTTP/2 connection establishment
2. Route resolution by endpoint path
3. Async request handling with metrics
4. Immediate acknowledgment (202 Accepted)
5. Background processing with tracing

#### NATS Endpoint (`ingress/push_endpoint.rs`)

NATS-based service endpoint with async processing.

**Features:**
- **Async Processing**: Non-blocking request handling
- **Graceful Shutdown**: Inflight request completion
- **Health Management**: Automatic status updates
- **Metrics Integration**: Request counting and timing

#### Push Handler (`ingress/push_handler.rs`)

Core request processing engine with comprehensive metrics.

**Processing Pipeline:**
1. **Payload Decoding**: TwoPartCodec message parsing
2. **Context Creation**: Request context with distributed tracing
3. **Stream Establishment**: TCP response stream setup
4. **Generation**: Business logic execution
5. **Response Streaming**: Real-time result streaming
6. **Completion Signaling**: End-of-stream handling

**Metrics Collection:**
- Request counters and timing
- Inflight request tracking
- Byte transfer metrics
- Error categorization and counting

## Message Flow Architecture

### Request Flow (Egress)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───►│ PushRouter  │───►│AddressRouter│───►│   Worker    │
│             │    │             │    │             │    │             │
│ SingleIn<T> │    │ Load Balance│    │ Transport   │    │ Processing  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ Service     │    │ Request     │
                   │ Discovery   │    │ Plane       │
                   │ (etcd)      │    │ (NATS/HTTP) │
                   └─────────────┘    └─────────────┘
```

### Response Flow (Ingress)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Worker    │───►│ PushHandler │───►│ TCP Client  │───►│   Router    │
│             │    │             │    │             │    │             │
│ ManyOut<U>  │    │ Stream      │    │ Response    │    │ Stream      │
│             │    │ Processing  │    │ Stream      │    │ Consumer    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │ TCP Server  │
                   │ (Response   │
                   │  Plane)     │
                   └─────────────┘
```

## Transport Protocols

### NATS Transport
- **Message Queuing**: Reliable message delivery
- **Subject-based Routing**: Hierarchical addressing
- **Header Support**: Metadata and tracing
- **Acknowledgments**: Delivery confirmation

### HTTP/2 Transport
- **Multiplexing**: Concurrent request handling
- **Binary Protocol**: Efficient wire format
- **Header Compression**: Reduced bandwidth usage
- **Connection Pooling**: Resource optimization

### TCP Streaming
- **Call-Home Pattern**: Client-initiated connections
- **Flow Control**: Backpressure handling
- **Binary Protocol**: TwoPartCodec framing
- **Graceful Shutdown**: Clean connection termination

## Configuration and Environment

### Request Plane Mode
```bash
# Environment variable controls transport selection
DYN_REQUEST_PLANE_MODE=nats    # Use NATS transport
DYN_REQUEST_PLANE_MODE=http    # Use HTTP/2 transport
```

### HTTP Configuration
```bash
DYN_HTTP_REQUEST_TIMEOUT=5     # HTTP request timeout (seconds)
DYN_HTTP_RPC_ROOT_PATH=/v1/dynamo  # HTTP endpoint root path
```

### TCP Configuration
- **Interface Selection**: Automatic or manual interface binding
- **Port Assignment**: Dynamic port allocation
- **Connection Limits**: Configurable connection pooling
- **Timeout Settings**: Customizable timeout values

## Error Handling and Resilience

### Fault Detection
- **Instance Health Monitoring**: Real-time worker status
- **Circuit Breaker**: Automatic failure isolation
- **Retry Logic**: Exponential backoff strategies
- **Graceful Degradation**: Service overload handling

### Error Categories
- **Transport Errors**: Network-level failures
- **Protocol Errors**: Message format issues
- **Application Errors**: Business logic failures
- **Resource Errors**: Memory/connection limits

### Recovery Mechanisms
- **Automatic Retry**: Configurable retry policies
- **Failover**: Instance-level fault tolerance
- **Load Shedding**: Overload protection
- **Circuit Breaking**: Cascade failure prevention

## Performance Characteristics

### Throughput Optimizations
- **Zero-Copy Operations**: Minimal memory allocations
- **Connection Pooling**: Reduced connection overhead
- **HTTP/2 Multiplexing**: Concurrent request processing
- **Binary Protocols**: Efficient serialization

### Latency Optimizations
- **TCP_NODELAY**: Immediate packet transmission
- **Connection Reuse**: Reduced handshake overhead
- **Async Processing**: Non-blocking operations
- **Direct Memory Access**: Efficient buffer management

### Scalability Features
- **Horizontal Scaling**: Multi-instance support
- **Load Balancing**: Multiple routing strategies
- **Resource Pooling**: Efficient resource utilization
- **Backpressure Handling**: Flow control mechanisms

## Observability and Monitoring

### Distributed Tracing
- **OpenTelemetry Integration**: Standard tracing format
- **Header Propagation**: Cross-service trace correlation
- **Span Management**: Request lifecycle tracking
- **Context Preservation**: Async operation tracing

### Metrics Collection
- **Request Metrics**: Count, duration, bytes transferred
- **Error Metrics**: Categorized error tracking
- **Resource Metrics**: Connection and memory usage
- **Performance Metrics**: Latency and throughput

### Health Monitoring
- **Endpoint Health**: Service availability tracking
- **Worker Status**: Instance health monitoring
- **System Health**: Overall system status
- **Canary Checks**: Proactive health validation

## Security Considerations

### Transport Security
- **TLS Support**: Encrypted communication channels
- **Certificate Management**: Automated certificate handling
- **Authentication**: Service-to-service authentication
- **Authorization**: Request-level access control

### Data Protection
- **Payload Encryption**: End-to-end data protection
- **Header Sanitization**: Sensitive data filtering
- **Audit Logging**: Security event tracking
- **Rate Limiting**: DoS protection mechanisms

## Usage Patterns

### Single Request/Multiple Response
```rust
// Router sends single request, receives stream of responses
let router = PushRouter::from_client(client, RouterMode::RoundRobin).await?;
let response_stream = router.generate(request).await?;
```

### HTTP/2 Request Routing
```rust
// HTTP-based request with automatic load balancing
let http_client = HttpRequestClient::new()?;
let router = AddressedPushRouter::new_http(http_client, tcp_server)?;
```

### Service Registration
```rust
// Register HTTP endpoint with shared server
shared_server.register_endpoint(
    subject,
    service_handler,
    instance_id,
    namespace,
    component_name,
    endpoint_name,
    system_health,
).await?;
```

## Best Practices

### Performance
1. **Connection Pooling**: Reuse connections when possible
2. **Batch Operations**: Group related requests
3. **Async Processing**: Use non-blocking operations
4. **Resource Cleanup**: Proper resource lifecycle management

### Reliability
1. **Error Handling**: Comprehensive error recovery
2. **Timeout Configuration**: Appropriate timeout values
3. **Health Monitoring**: Proactive health checks
4. **Graceful Shutdown**: Clean service termination

### Observability
1. **Distributed Tracing**: Enable request correlation
2. **Metrics Collection**: Monitor key performance indicators
3. **Structured Logging**: Use consistent log formats
4. **Alert Configuration**: Set up proactive monitoring

## Future Enhancements

### Planned Features
- **gRPC Support**: Native gRPC transport integration
- **WebSocket Streaming**: Real-time bidirectional communication
- **Advanced Load Balancing**: ML-based routing decisions
- **Enhanced Security**: Zero-trust networking features

### Performance Improvements
- **QUIC Protocol**: Next-generation transport protocol
- **Hardware Acceleration**: GPU-accelerated networking
- **Memory Optimization**: Further zero-copy improvements
- **Compression**: Adaptive payload compression

This network layer provides a robust, scalable, and observable foundation for distributed AI/ML workloads, with careful attention to performance, reliability, and operational excellence.

# Transport Registry Implementation Roadmap

This document provides a concrete, task-based roadmap for implementing the Transport Registry and related enhancements.

## Overview

| Priority | Feature | Timeline | Complexity | Impact |
|----------|---------|----------|------------|--------|
| 0 | TCP Connection Pooling | 1-2 weeks | Medium | High (5-20% latency reduction) |
| 1 | HTTP Response Path | 2-3 months | Medium-High | Medium (simplified deployment) |
| 2 | TCP Request Path | 4-6 months | High | High (lowest latency option) |

---

## Priority 0: TCP Connection Pooling

**Goal**: Eliminate TCP handshake overhead by reusing connections

**Timeline**: Week 1-2

**Team Size**: 1-2 engineers

### Tasks

#### Week 1: Implementation

- [ ] **Task 0.1**: Create connection pool module
  - **File**: `lib/runtime/src/pipeline/network/tcp/connection_pool.rs`
  - **Deliverable**: `TcpResponseConnectionPool` struct with basic pooling
  - **Effort**: 2 days
  - **Details**:
    - Implement `PoolConfig` with configurable parameters
    - Implement `ConnectionSubPool` per target
    - Implement `PooledConnection` with lifecycle tracking
    - Implement `PooledTcpStream` RAII wrapper

- [ ] **Task 0.2**: Add connection validation
  - **File**: `lib/runtime/src/pipeline/network/tcp/connection_pool.rs`
  - **Deliverable**: Connection health checking logic
  - **Effort**: 1 day
  - **Details**:
    - Implement age-based validation (max lifetime)
    - Implement idle timeout validation
    - Implement TCP-level health check (peek)
    - Add validation metrics

- [ ] **Task 0.3**: Integrate with TcpClient
  - **File**: `lib/runtime/src/pipeline/network/tcp/client.rs`
  - **Deliverable**: Updated `TcpClient` using connection pool
  - **Effort**: 1 day
  - **Details**:
    - Replace `TcpStream::connect()` with pool acquisition
    - Add global pool instance (lazy_static)
    - Preserve existing handshake logic
    - Add environment variable configuration

- [ ] **Task 0.4**: Update TcpStreamServer for persistence
  - **File**: `lib/runtime/src/pipeline/network/tcp/server.rs`
  - **Deliverable**: Server supports connection reuse
  - **Effort**: 2 days
  - **Details**:
    - Modify connection handler to loop for multiple requests
    - Match handshakes to waiting response streams
    - Add connection keepalive logic
    - Handle graceful connection closure

- [ ] **Task 0.5**: Add metrics and monitoring
  - **File**: `lib/runtime/src/pipeline/network/tcp/connection_pool.rs`
  - **Deliverable**: Prometheus metrics for pool usage
  - **Effort**: 1 day
  - **Details**:
    - Add `ConnectionPoolMetrics` struct
    - Instrument connection creation/reuse
    - Track active/idle connection counts
    - Add acquisition latency histogram

#### Week 2: Testing and Rollout

- [ ] **Task 0.6**: Unit tests
  - **File**: `lib/runtime/src/pipeline/network/tcp/connection_pool.rs`
  - **Deliverable**: Comprehensive test coverage
  - **Effort**: 2 days
  - **Tests**:
    - Connection reuse behavior
    - Pool size limits
    - Validation logic (age, idle, health)
    - Concurrent access
    - Connection leaks

- [ ] **Task 0.7**: Integration tests
  - **File**: `lib/runtime/tests/tcp_pool_integration_test.rs`
  - **Deliverable**: End-to-end pool testing
  - **Effort**: 1 day
  - **Tests**:
    - Multiple requests over same connection
    - Connection pool exhaustion recovery
    - Server-side connection reuse
    - Metrics validation

- [ ] **Task 0.8**: Performance benchmarks
  - **File**: `lib/runtime/benches/tcp_pool_benchmark.rs`
  - **Deliverable**: Before/after latency comparison
  - **Effort**: 1 day
  - **Metrics**:
    - p50, p95, p99 latency with/without pooling
    - Throughput comparison
    - Connection creation rate

- [ ] **Task 0.9**: Documentation
  - **Files**: Update design docs
  - **Deliverable**: Usage guide and configuration docs
  - **Effort**: 1 day
  - **Content**:
    - Configuration options
    - Tuning guidelines
    - Troubleshooting guide

- [ ] **Task 0.10**: Feature flag and rollout
  - **File**: `lib/runtime/src/config/mod.rs`
  - **Deliverable**: Gradual rollout with monitoring
  - **Effort**: 1 day
  - **Steps**:
    - Add `DYN_TCP_POOL_ENABLED` feature flag
    - Deploy to staging
    - Monitor metrics (latency, error rate)
    - Gradual production rollout (10% → 50% → 100%)

### Success Criteria

- [ ] 5-10% p50 latency reduction
- [ ] 10-20% p99 latency reduction
- [ ] Connection reuse rate > 80%
- [ ] No increase in error rate
- [ ] No connection leaks (stable idle pool size)

### Rollback Plan

- Set `DYN_TCP_POOL_ENABLED=false` to revert to original behavior
- No code changes required for rollback

---

## Priority 1: HTTP Response Path

**Goal**: Provide pure HTTP/2 mode (HTTP request + HTTP response)

**Timeline**: Month 1-3

**Team Size**: 2-3 engineers

### Month 1: Core Implementation

#### Week 1-2: Server Side

- [ ] **Task 1.1**: Create HTTP response server
  - **File**: `lib/runtime/src/pipeline/network/ingress/http_response_endpoint.rs`
  - **Deliverable**: Axum-based streaming response server
  - **Effort**: 3 days
  - **Details**:
    - Implement `HttpResponseServer` struct
    - Create POST handler for requests
    - Implement SSE streaming format
    - Add tracing context propagation

- [ ] **Task 1.2**: Implement streaming formats
  - **File**: `lib/runtime/src/pipeline/network/ingress/http_response_endpoint.rs`
  - **Deliverable**: Support SSE, NDJSON, binary formats
  - **Effort**: 2 days
  - **Details**:
    - Create `StreamFormat` enum
    - Implement SSE formatting (base64-encoded data)
    - Implement NDJSON formatting
    - Implement binary streaming
    - Add format negotiation via Accept header

- [ ] **Task 1.3**: Integrate with existing handlers
  - **File**: `lib/runtime/src/pipeline/network/ingress/http_response_endpoint.rs`
  - **Deliverable**: Bridge to `PushWorkHandler` trait
  - **Effort**: 2 days
  - **Details**:
    - Create adapter for existing handlers
    - Handle async response generation
    - Propagate errors as SSE error events
    - Add completion signaling

- [ ] **Task 1.4**: Add configuration
  - **File**: `lib/runtime/src/config/response_plane.rs`
  - **Deliverable**: Environment variable configuration
  - **Effort**: 1 day
  - **Details**:
    - Add `ResponsePlaneMode` enum
    - Add environment variable parsing
    - Add server configuration (host, port, paths)
    - Add format selection

#### Week 3-4: Client Side

- [ ] **Task 1.5**: Create HTTP response client
  - **File**: `lib/runtime/src/pipeline/network/egress/http_response_client.rs`
  - **Deliverable**: reqwest-based streaming client
  - **Effort**: 3 days
  - **Details**:
    - Implement `HttpResponseClient` struct
    - Create SSE stream parser
    - Implement backpressure handling
    - Add HTTP/2 connection pooling

- [ ] **Task 1.6**: Implement ResponsePlaneClient trait
  - **File**: `lib/runtime/src/pipeline/network/egress/http_response_client.rs`
  - **Deliverable**: Standard interface implementation
  - **Effort**: 2 days
  - **Details**:
    - Implement `create_response_stream()`
    - Implement `ResponseStreamWriter` wrapper
    - Handle stream lifecycle
    - Add capability reporting

- [ ] **Task 1.7**: Update router logic
  - **File**: `lib/runtime/src/pipeline/network/egress/http_router.rs`
  - **Deliverable**: Support HTTP response path selection
  - **Effort**: 2 days
  - **Details**:
    - Check response plane mode from config
    - Select appropriate response client
    - Update request flow to use HTTP responses
    - Remove TCP call-home when using HTTP responses

### Month 2: Integration and Testing

#### Week 5: Service Discovery Integration

- [ ] **Task 1.8**: Update TransportType enum
  - **File**: `lib/runtime/src/component.rs`
  - **Deliverable**: Add `HttpHttp` variant
  - **Effort**: 1 day
  - **Details**:
    - Add new transport type for HTTP+HTTP mode
    - Update serialization/deserialization
    - Maintain backward compatibility

- [ ] **Task 1.9**: Update endpoint registration
  - **File**: `lib/runtime/src/component/endpoint.rs`
  - **Deliverable**: Advertise HTTP response endpoints
  - **Effort**: 2 days
  - **Details**:
    - Build HTTP response endpoint URLs
    - Register in etcd with appropriate transport type
    - Add health check integration
    - Update endpoint discovery logic

- [ ] **Task 1.10**: Update client discovery
  - **File**: `lib/runtime/src/component/client.rs`
  - **Deliverable**: Discover and use HTTP response endpoints
  - **Effort**: 2 days
  - **Details**:
    - Parse HTTP response endpoints from etcd
    - Route requests to HTTP response URLs
    - Handle mixed transport deployments
    - Add fallback logic

#### Week 6-7: Testing

- [ ] **Task 1.11**: Unit tests
  - **Files**: Various test modules
  - **Deliverable**: Component-level test coverage
  - **Effort**: 3 days
  - **Tests**:
    - SSE parsing and formatting
    - NDJSON parsing and formatting
    - Error handling and propagation
    - Stream lifecycle management
    - Configuration parsing

- [ ] **Task 1.12**: Integration tests
  - **File**: `lib/runtime/tests/http_response_integration_test.rs`
  - **Deliverable**: End-to-end HTTP response flow
  - **Effort**: 2 days
  - **Tests**:
    - HTTP request + HTTP response flow
    - Multiple concurrent streams
    - Error scenarios
    - Service discovery integration
    - Graceful shutdown

- [ ] **Task 1.13**: Performance testing
  - **File**: `lib/runtime/benches/http_vs_tcp_response.rs`
  - **Deliverable**: Latency comparison
  - **Effort**: 2 days
  - **Metrics**:
    - Latency comparison (HTTP vs TCP response)
    - Throughput comparison
    - CPU and memory overhead
    - Connection overhead

#### Week 8: Documentation and Examples

- [ ] **Task 1.14**: Update documentation
  - **Files**: Various .md files
  - **Deliverable**: Complete usage guide
  - **Effort**: 2 days
  - **Content**:
    - Configuration guide
    - Migration guide (TCP → HTTP response)
    - Troubleshooting guide
    - API documentation

- [ ] **Task 1.15**: Create examples
  - **File**: `examples/http_response_demo.rs`
  - **Deliverable**: Working example application
  - **Effort**: 1 day
  - **Content**:
    - Simple echo service
    - Streaming response example
    - Error handling example

### Month 3: Pilot and Rollout

#### Week 9-10: Pilot Deployment

- [ ] **Task 1.16**: Deploy to test environment
  - **Deliverable**: HTTP response path in test cluster
  - **Effort**: 2 days
  - **Steps**:
    - Deploy to test cluster
    - Configure test endpoints for HTTP response
    - Run automated tests
    - Manual validation

- [ ] **Task 1.17**: Performance validation
  - **Deliverable**: Production-like load testing
  - **Effort**: 2 days
  - **Tests**:
    - Load test at 100%, 200%, 500% expected load
    - Sustained load over 24 hours
    - Burst traffic scenarios
    - Failover testing

- [ ] **Task 1.18**: Pilot in production
  - **Deliverable**: HTTP response on low-traffic endpoints
  - **Effort**: 3 days
  - **Steps**:
    - Select 1-2 low-traffic endpoints
    - Enable HTTP response mode
    - Monitor metrics for 7 days
    - Validate correctness and performance

#### Week 11-12: Gradual Rollout

- [ ] **Task 1.19**: Expand to more endpoints
  - **Deliverable**: Gradual traffic migration
  - **Effort**: Ongoing
  - **Steps**:
    - Week 11: 10% of endpoints
    - Week 12: 50% of endpoints
    - Monitor and adjust

- [ ] **Task 1.20**: Production readiness review
  - **Deliverable**: Go/no-go decision
  - **Effort**: 1 day
  - **Criteria**:
    - Latency within acceptable range (<10% increase vs TCP)
    - No increase in error rate
    - Positive feedback from operators
    - Troubleshooting tools in place

### Success Criteria

- [ ] HTTP response path successfully handles production traffic
- [ ] Latency within 10% of TCP response path
- [ ] Error rate <= TCP response path
- [ ] Easier debugging and troubleshooting reported by operators
- [ ] Compatible with standard HTTP infrastructure (proxies, load balancers)

---

## Priority 2: TCP Request Path

**Goal**: Full TCP mode (TCP request + TCP response) for lowest latency

**Timeline**: Month 4-9

**Team Size**: 2-3 engineers

### Month 4-5: Protocol Design and Implementation

#### Week 13-14: Protocol Design

- [ ] **Task 2.1**: Finalize protocol specification
  - **File**: `docs/tcp_multiplex_protocol.md`
  - **Deliverable**: Complete protocol specification
  - **Effort**: 3 days
  - **Details**:
    - Frame header format
    - Frame types (request, response, control, heartbeat)
    - Multiplexing semantics
    - Error handling
    - Flow control

- [ ] **Task 2.2**: Design codec implementation
  - **File**: Design document
  - **Deliverable**: Codec architecture
  - **Effort**: 2 days
  - **Details**:
    - State machine for decoding
    - Buffer management
    - Error recovery
    - Performance considerations

#### Week 15-16: Codec Implementation

- [ ] **Task 2.3**: Implement TcpMultiplexCodec
  - **File**: `lib/runtime/src/pipeline/network/tcp/multiplex_codec.rs`
  - **Deliverable**: Working codec for framing
  - **Effort**: 4 days
  - **Details**:
    - Implement `Decoder` trait
    - Implement `Encoder` trait
    - Add frame validation
    - Add error handling

- [ ] **Task 2.4**: Codec unit tests
  - **File**: `lib/runtime/src/pipeline/network/tcp/multiplex_codec.rs`
  - **Deliverable**: Comprehensive codec tests
  - **Effort**: 2 days
  - **Tests**:
    - Encoding/decoding round-trip
    - Partial frame handling
    - Invalid frame handling
    - Large frame handling

#### Week 17-18: Server Implementation

- [ ] **Task 2.5**: Implement TcpRequestServer
  - **File**: `lib/runtime/src/pipeline/network/ingress/tcp_request_endpoint.rs`
  - **Deliverable**: TCP server accepting multiplexed requests
  - **Effort**: 4 days
  - **Details**:
    - Accept persistent connections
    - Demultiplex requests by request_id
    - Spawn handlers for each request
    - Send multiplexed responses

- [ ] **Task 2.6**: Implement heartbeat mechanism
  - **File**: `lib/runtime/src/pipeline/network/ingress/tcp_request_endpoint.rs`
  - **Deliverable**: Connection keepalive
  - **Effort**: 2 days
  - **Details**:
    - Periodic heartbeat frames
    - Timeout detection
    - Connection health monitoring

#### Week 19-20: Client Implementation

- [ ] **Task 2.7**: Implement TcpRequestClient
  - **File**: `lib/runtime/src/pipeline/network/egress/tcp_request_client.rs`
  - **Deliverable**: TCP client with connection pooling
  - **Effort**: 4 days
  - **Details**:
    - Persistent connection management
    - Request multiplexing
    - Response demultiplexing
    - Connection recovery

- [ ] **Task 2.8**: Implement TcpMultiplexConnection
  - **File**: `lib/runtime/src/pipeline/network/egress/tcp_request_client.rs`
  - **Deliverable**: Connection abstraction with multiplexing
  - **Effort**: 3 days
  - **Details**:
    - Send requests with unique IDs
    - Route responses to waiting futures
    - Handle concurrent requests
    - Implement backpressure

### Month 6-7: Integration and Testing

#### Week 21-22: Integration

- [ ] **Task 2.9**: Update TransportType
  - **File**: `lib/runtime/src/component.rs`
  - **Deliverable**: Add `TcpTcp` transport type
  - **Effort**: 2 days

- [ ] **Task 2.10**: Update service discovery
  - **File**: `lib/runtime/src/component/endpoint.rs`
  - **Deliverable**: Register TCP request endpoints
  - **Effort**: 2 days

- [ ] **Task 2.11**: Update router logic
  - **File**: `lib/runtime/src/pipeline/network/egress/`
  - **Deliverable**: Support TCP request path
  - **Effort**: 3 days

#### Week 23-24: Testing

- [ ] **Task 2.12**: Unit tests
  - **Deliverable**: Component test coverage
  - **Effort**: 3 days

- [ ] **Task 2.13**: Integration tests
  - **File**: `lib/runtime/tests/tcp_request_integration_test.rs`
  - **Deliverable**: End-to-end TCP request/response
  - **Effort**: 3 days
  - **Tests**:
    - Single request/response
    - Concurrent requests over same connection
    - Connection recovery
    - Heartbeat timeout
    - Large payload handling

- [ ] **Task 2.14**: Stress tests
  - **Deliverable**: High-load testing
  - **Effort**: 2 days
  - **Tests**:
    - 10K+ requests per second
    - 1000+ concurrent connections
    - Sustained load over 24 hours
    - Connection churn scenarios

### Month 8-9: Rollout

#### Week 25-28: Pilot and Production

- [ ] **Task 2.15**: Deploy to test environment
  - **Effort**: 2 days

- [ ] **Task 2.16**: Performance validation
  - **Effort**: 3 days
  - **Validate**:
    - Lower latency than HTTP request path
    - High throughput
    - Efficient connection usage

- [ ] **Task 2.17**: Pilot on internal endpoints
  - **Effort**: 2 weeks
  - **Steps**:
    - Enable on internal high-throughput endpoints
    - Monitor for 2 weeks
    - Validate stability

- [ ] **Task 2.18**: Production rollout decision
  - **Effort**: 1 day

### Success Criteria

- [ ] Lowest latency among all transport options
- [ ] Successful multiplexing of 100+ concurrent requests per connection
- [ ] Stable under high load (10K+ RPS)
- [ ] Clean connection recovery on failures

---

## Infrastructure Tasks (Ongoing)

### Metrics and Monitoring

- [ ] **Task I.1**: Add transport selection metrics
  - Track which transports are being used
  - Monitor transport health
  - Alert on transport failures

- [ ] **Task I.2**: Add latency breakdown metrics
  - Measure time spent in each transport layer
  - Identify performance bottlenecks
  - Track p50/p95/p99 by transport type

- [ ] **Task I.3**: Create monitoring dashboards
  - Connection pool utilization
  - Transport performance comparison
  - Error rates by transport

### Configuration Management

- [ ] **Task I.4**: Create configuration validation
  - Validate environment variables
  - Detect conflicting configuration
  - Provide helpful error messages

- [ ] **Task I.5**: Add configuration reload
  - Support dynamic configuration changes
  - Graceful transport switching
  - Zero-downtime configuration updates

### Documentation

- [ ] **Task I.6**: Maintain design docs
  - Keep TRANSPORT_REGISTRY_DESIGN.md up to date
  - Document lessons learned
  - Add troubleshooting guides

- [ ] **Task I.7**: Create operator runbook
  - Common issues and solutions
  - Performance tuning guide
  - Capacity planning guide

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|-----------|
| Connection pool leaks | Medium | High | Extensive testing, metrics, automatic leak detection |
| HTTP response latency higher than expected | Medium | Medium | Performance testing, optimization, fallback to TCP |
| TCP multiplexing protocol bugs | Low | High | Formal protocol spec, extensive testing, fuzzing |
| Service discovery race conditions | Low | Medium | Careful state management, integration tests |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|-----------|
| Gradual rollout issues | Medium | High | Feature flags, gradual traffic shifting, rollback plan |
| Configuration errors | Medium | Medium | Validation, documentation, examples |
| Debugging difficulty | Medium | Medium | Good logging, metrics, debugging tools |

---

## Resource Requirements

### Engineering

- **Priority 0**: 1-2 engineers, 2 weeks
- **Priority 1**: 2-3 engineers, 3 months
- **Priority 2**: 2-3 engineers, 6 months

### Infrastructure

- Test clusters for validation
- Load testing infrastructure
- Monitoring and alerting setup

### Timeline Summary

```
Month 1  : Priority 0 (TCP Connection Pooling)
Month 2-4: Priority 1 (HTTP Response Path)
Month 5-9: Priority 2 (TCP Request Path)

┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  M1  │  M2  │  M3  │  M4  │  M5  │  M6  │  M7  │  M8  │  M9  │
├──────┼──────┴──────┴──────┼──────┴──────┴──────┴──────┴──────┤
│ P0   │      Priority 1     │         Priority 2               │
│ Pool │   HTTP Response     │       TCP Request Path           │
└──────┴─────────────────────┴──────────────────────────────────┘
```

---

## Success Metrics

### Overall

- [ ] All three priorities implemented and in production
- [ ] No regression in error rates
- [ ] Improved debugging experience reported by operators
- [ ] Flexible deployment options (HTTP-only, mixed, TCP-only)

### Priority 0 (Connection Pooling)

- [ ] 5-10% p50 latency reduction
- [ ] 10-20% p99 latency reduction
- [ ] Connection reuse rate > 80%

### Priority 1 (HTTP Response)

- [ ] Latency within 10% of TCP response path
- [ ] Deployed on > 50% of production endpoints
- [ ] Positive operator feedback

### Priority 2 (TCP Request)

- [ ] Lowest latency option available
- [ ] Successfully handling > 10K RPS per connection
- [ ] Deployed on high-throughput internal services

---

## Next Steps

1. **Review and approve** this roadmap
2. **Assign ownership** for each priority
3. **Set up tracking** (JIRA, GitHub Projects, etc.)
4. **Schedule kickoff** for Priority 0
5. **Establish metrics** and monitoring
6. **Plan regular reviews** (weekly for active work, monthly for upcoming work)

## References

- [TRANSPORT_REGISTRY_DESIGN.md](TRANSPORT_REGISTRY_DESIGN.md) - Detailed design
- [TRANSPORT_REGISTRY_SUMMARY.md](TRANSPORT_REGISTRY_SUMMARY.md) - Quick reference
- [TRANSPORT_REGISTRY_EXAMPLE.md](TRANSPORT_REGISTRY_EXAMPLE.md) - Code examples


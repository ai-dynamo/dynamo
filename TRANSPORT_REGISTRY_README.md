# Transport Registry and Service Discovery Enhancement

> **High-level interface and design for registering different transport mechanisms and advertising them through service discovery**

## Overview

This document collection describes a comprehensive enhancement to Dynamo's network transport layer, introducing a flexible, extensible architecture that supports multiple request and response transports with efficient connection management.

### Current State

Dynamo currently supports:
- **Request Plane**: NATS (legacy) or HTTP/2
- **Response Plane**: TCP with call-home pattern

### Enhancement Goals

1. **Priority 0** (Immediate): TCP connection pooling to eliminate handshake overhead
2. **Priority 1** (3-6 months): HTTP-based response path for simplified deployment
3. **Priority 2** (6-12 months): TCP-based request path for lowest latency

## Documentation Structure

### 📘 [TRANSPORT_REGISTRY_DESIGN.md](TRANSPORT_REGISTRY_DESIGN.md)
**Comprehensive architectural design document**

Complete technical specification covering:
- Current architecture analysis
- Transport abstraction interfaces
- Service discovery integration
- Detailed implementation for all three priorities
- Migration strategy and configuration

**Read this if you want to understand the full architecture and design decisions.**

### 📄 [TRANSPORT_REGISTRY_SUMMARY.md](TRANSPORT_REGISTRY_SUMMARY.md)
**Quick reference and usage guide**

Condensed reference including:
- Supported transport combinations
- Environment variables
- Metrics and monitoring
- Architecture diagrams
- Troubleshooting tips
- FAQ

**Read this for quick configuration and operational guidance.**

### 💻 [TRANSPORT_REGISTRY_EXAMPLE.md](TRANSPORT_REGISTRY_EXAMPLE.md)
**Code examples and practical usage**

Working code examples for:
- Basic setup and initialization
- Priority 0: Enabling connection pooling
- Priority 1: HTTP response path implementation
- Priority 2: TCP request path implementation
- Custom transport implementation

**Read this when implementing or integrating with the transport registry.**

### 🗺️ [TRANSPORT_REGISTRY_ROADMAP.md](TRANSPORT_REGISTRY_ROADMAP.md)
**Implementation roadmap and task breakdown**

Concrete implementation plan including:
- Week-by-week task breakdown
- Success criteria for each priority
- Resource requirements
- Risk mitigation
- Timeline and milestones

**Read this for project planning and task assignment.**

### 🔧 [lib/runtime/src/pipeline/network/transport_registry.rs](lib/runtime/src/pipeline/network/transport_registry.rs)
**Interface definitions and trait implementations**

Rust code defining:
- `TransportRegistry` - Central registry for all transports
- `RequestPlaneClient` / `RequestPlaneServer` - Request plane traits
- `ResponsePlaneClient` / `ResponsePlaneServer` - Response plane traits
- `TransportCapabilities` - Transport feature description
- Configuration enums and helpers

**Read this for API reference and type definitions.**

## Quick Start

### For Operators

To enable TCP connection pooling (Priority 0):

```bash
# Enable connection pooling
export DYN_TCP_POOL_ENABLED=true

# Restart your services
# Monitor metrics at http://localhost:9090/metrics
```

See [TRANSPORT_REGISTRY_SUMMARY.md](TRANSPORT_REGISTRY_SUMMARY.md) for more configuration options.

### For Developers

To integrate with the transport registry:

```rust
use dynamo::pipeline::network::transport_registry::*;

// Create registry
let registry = Arc::new(TransportRegistry::new());

// Register transports
registry.register_request_transport(
    registration,
    client,
    server,
).await?;

// Select transport based on capabilities
let transport_id = registry.select_transport(
    PlaneType::Response,
    &required_capabilities,
).await?;
```

See [TRANSPORT_REGISTRY_EXAMPLE.md](TRANSPORT_REGISTRY_EXAMPLE.md) for complete examples.

### For Project Managers

Implementation timeline:
- **Month 1**: Priority 0 (TCP Connection Pooling)
- **Month 2-4**: Priority 1 (HTTP Response Path)
- **Month 5-9**: Priority 2 (TCP Request Path)

See [TRANSPORT_REGISTRY_ROADMAP.md](TRANSPORT_REGISTRY_ROADMAP.md) for detailed planning.

## Key Features

### Priority 0: TCP Connection Pooling

**Benefits**:
- ✅ 5-10% p50 latency reduction
- ✅ 10-20% p99 latency reduction
- ✅ Lower CPU usage
- ✅ Fewer file descriptors
- ✅ Backward compatible

**Effort**: 1-2 weeks, 1-2 engineers

**Status**: Design complete, ready to implement

### Priority 1: HTTP Response Path

**Benefits**:
- ✅ Simplified deployment (no call-home pattern)
- ✅ Standard HTTP tooling support
- ✅ Better proxy/load balancer compatibility
- ✅ Easier debugging

**Trade-offs**:
- ⚠️ Slightly higher latency than TCP (HTTP framing overhead)

**Effort**: 2-3 months, 2-3 engineers

**Status**: Design complete, ready to implement

### Priority 2: TCP Request Path

**Benefits**:
- ✅ Lowest possible latency
- ✅ Single persistent connection
- ✅ Request/response multiplexing
- ✅ Bidirectional streaming support

**Trade-offs**:
- ⚠️ Custom protocol (harder to debug)
- ⚠️ Not compatible with HTTP-only proxies

**Effort**: 4-6 months, 2-3 engineers

**Status**: Design complete, protocol specification pending

## Architecture Overview

### Current: HTTP Request + TCP Response

```
┌─────────┐                                    ┌────────┐
│ Router  │ ──HTTP POST /subject────────────▶ │ Worker │
│         │                                    │        │
│ TCP Srv │ ◀─TCP connect (call-home)───────── │ TCP Cli│
│         │ ◀─Response stream (TCP)──────────── │        │
└─────────┘                                    └────────┘
```

### Priority 0: With Connection Pool

```
┌─────────┐                                    ┌────────┐
│ Router  │ ──HTTP POST /subject────────────▶ │ Worker │
│         │                                    │        │
│ TCP Srv │ ═══Persistent Connection═════════ │ TCP Pool│
│         │ ◀══Response stream (reused)═══════ │        │
└─────────┘     No handshake overhead!        └────────┘
```

### Priority 1: HTTP Request + HTTP Response

```
┌─────────┐                                    ┌────────┐
│ Router  │ ──HTTP POST /subject────────────▶ │ Worker │
│         │ ◀─200 OK + SSE stream──────────── │        │
│         │    data: {...}                    │        │
│         │    data: {...}                    │        │
└─────────┘    Simple, standard!              └────────┘
```

### Priority 2: TCP Request + TCP Response

```
┌─────────┐    ╔════════════════════════╗     ┌────────┐
│ Router  │ ═══║ Persistent TCP Conn    ║═════│ Worker │
│         │    ║ (multiplexed)          ║     │        │
│ TCP Cli │────║─Request ID: A──────────║────▶│ TCP Srv│
│         │◀───║─Response ID: A─────────║─────│        │
└─────────┘    ╚════════════════════════╝     └────────┘
               Lowest latency!
```

## Technology Stack

### Languages and Frameworks
- **Rust** - Core implementation
- **Tokio** - Async runtime
- **Axum** - HTTP server framework (Priority 1)
- **reqwest** - HTTP client (Priority 1)

### Protocols
- **HTTP/2** - Request and response streaming
- **TCP** - Low-level transport with custom framing
- **SSE** (Server-Sent Events) - HTTP response streaming format

### Infrastructure
- **etcd** - Service discovery
- **Prometheus** - Metrics and monitoring

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_REQUEST_PLANE` | `nats` | Request transport: `nats`, `http`, `tcp` |
| `DYN_RESPONSE_PLANE` | `tcp` | Response transport: `tcp`, `http` |
| `DYN_TCP_POOL_ENABLED` | `false` | Enable TCP connection pooling |
| `DYN_TCP_POOL_MAX_PER_TARGET` | `50` | Max pooled connections per target |
| `DYN_HTTP_RESPONSE_PORT` | `8082` | HTTP response server port |
| `DYN_TCP_REQUEST_PORT` | `9001` | TCP request server port |

See [TRANSPORT_REGISTRY_SUMMARY.md](TRANSPORT_REGISTRY_SUMMARY.md) for complete configuration reference.

## Metrics

### Connection Pool (Priority 0)
```
tcp_pool_connections_created_total
tcp_pool_connections_reused_total
tcp_pool_active_connections
tcp_pool_idle_connections
tcp_pool_acquisition_latency_seconds
```

### Transport Performance
```
transport_requests_total{transport="http|tcp|nats"}
transport_request_latency_seconds{transport="http|tcp|nats"}
transport_errors_total{transport="http|tcp|nats"}
transport_active_streams{transport="http|tcp|nats"}
```

## Testing Strategy

### Unit Tests
- Connection pool behavior
- Transport trait implementations
- Codec encoding/decoding
- Configuration parsing

### Integration Tests
- End-to-end request/response flows
- Service discovery integration
- Connection recovery scenarios
- Mixed transport deployments

### Performance Tests
- Latency comparison across transports
- Throughput benchmarks
- Load testing (10K+ RPS)
- Connection pool efficiency

## Success Criteria

### Priority 0 (Connection Pooling)
- [ ] 5-10% p50 latency reduction
- [ ] 10-20% p99 latency reduction
- [ ] Connection reuse rate > 80%
- [ ] No increase in error rate

### Priority 1 (HTTP Response)
- [ ] Latency within 10% of TCP
- [ ] Deployed on > 50% of endpoints
- [ ] Positive operator feedback
- [ ] Works with standard HTTP infrastructure

### Priority 2 (TCP Request)
- [ ] Lowest latency option available
- [ ] > 10K RPS per connection
- [ ] Stable under high load
- [ ] Clean failover and recovery

## Contributing

### Getting Started

1. Read [TRANSPORT_REGISTRY_DESIGN.md](TRANSPORT_REGISTRY_DESIGN.md) for architecture
2. Check [TRANSPORT_REGISTRY_ROADMAP.md](TRANSPORT_REGISTRY_ROADMAP.md) for open tasks
3. Review [TRANSPORT_REGISTRY_EXAMPLE.md](TRANSPORT_REGISTRY_EXAMPLE.md) for code patterns
4. Look at [transport_registry.rs](lib/runtime/src/pipeline/network/transport_registry.rs) for interfaces

### Development Workflow

1. Pick a task from the roadmap
2. Create a feature branch
3. Implement with tests
4. Run benchmarks if applicable
5. Update documentation
6. Submit PR with:
   - Code changes
   - Test coverage
   - Performance impact (if applicable)
   - Documentation updates

### Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Pass all lints (`cargo clippy`)
- Add comprehensive tests
- Document public APIs
- Include examples for new features

## FAQ

**Q: Should I use HTTP or TCP for response streaming?**

A: For most deployments, TCP with connection pooling (Priority 0) provides the best balance. Use HTTP (Priority 1) if you need better proxy compatibility. Use TCP request path (Priority 2) only for high-throughput internal services.

**Q: Can I mix transport types in the same deployment?**

A: Yes! Different endpoints can use different transport combinations, enabling gradual migration and A/B testing.

**Q: What's the performance impact of connection pooling?**

A: Expected 5-10% p50 latency reduction and 10-20% p99 reduction with minimal CPU/memory overhead.

**Q: How do I migrate from TCP to HTTP response path?**

A: Set `DYN_RESPONSE_PLANE=http` and restart services. Service discovery automatically updates. Both modes can coexist during migration.

**Q: What happens during failover?**

A: Connection pools automatically create new connections on failure. The transport registry allows fallback to alternative transports if configured.

## Support and Troubleshooting

### Debugging

```bash
# Enable debug logging
export RUST_LOG=dynamo::pipeline::network=debug

# Check metrics
curl http://localhost:9090/metrics | grep -E 'tcp_pool|transport_'

# Test endpoints
curl -v http://localhost:8081/v1/rpc/test
```

### Common Issues

**Connection pool not reusing connections**
- Check `DYN_TCP_POOL_ENABLED=true`
- Verify metrics show reuse: `tcp_pool_connections_reused_total`
- Check pool configuration (max connections, timeouts)

**HTTP response stream not working**
- Verify `Accept: text/event-stream` header
- Check server is running on configured port
- Try different format: `DYN_HTTP_RESPONSE_STREAM_FORMAT=ndjson`

See [TRANSPORT_REGISTRY_SUMMARY.md](TRANSPORT_REGISTRY_SUMMARY.md) for more troubleshooting tips.

## Roadmap

```
Q1 2025: Priority 0 (TCP Connection Pooling)
  └─ Design ✓
  └─ Implementation (pending)
  └─ Rollout (pending)

Q2 2025: Priority 1 (HTTP Response Path)
  └─ Design ✓
  └─ Implementation (pending)
  └─ Testing (pending)
  └─ Rollout (pending)

Q3-Q4 2025: Priority 2 (TCP Request Path)
  └─ Design ✓
  └─ Protocol specification (pending)
  └─ Implementation (pending)
  └─ Testing (pending)
  └─ Rollout (pending)
```

## License

SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0

## Related Documentation

- [HTTP_REQUEST_PLANE_IMPLEMENTATION.md](HTTP_REQUEST_PLANE_IMPLEMENTATION.md) - Existing HTTP request plane
- [transport_agnostic_dynamo-v2.md](transport_agnostic_dynamo-v2.md) - Transport-agnostic architecture
- [NETWORK_LAYER_DOCUMENTATION.md](NETWORK_LAYER_DOCUMENTATION.md) - Network layer overview

## Contact

For questions or discussions:
- GitHub Issues: Tag with `transport-registry`
- Slack: #dynamo-networking
- Email: dynamo-dev@nvidia.com

---

**Status**: Design Complete ✓ | Implementation Pending ⏳

**Last Updated**: October 16, 2025


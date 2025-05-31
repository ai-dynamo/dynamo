# Dynamo Test System Port Configuration

This document describes the port allocation strategy for the Dynamo test system.

## Fixed Service Ports

These ports are hardcoded for core infrastructure services to ensure consistency across test runs:

| Service | Port | Type | Description |
|---------|------|------|-------------|
| ETCD Client | 2379 | TCP | ETCD client communication |
| ETCD Peer | 2380 | TCP | ETCD peer communication |
| NATS Client | 4222 | TCP | NATS message broker |
| NATS HTTP Monitor | 8222 | HTTP | NATS monitoring interface |

## Dynamic Port Allocation

These services use dynamic port allocation to avoid conflicts:

| Service | Port Range | Type | Description |
|---------|------------|------|-------------|
| Dynamo Serve | 8000+ | HTTP | Dynamo serve API endpoints |
| Dynamo Run | Random | HTTP | Dynamo run HTTP interface |
| Test Services | Random | Various | Temporary test services |

## Port Management Strategy

1. **Core Infrastructure**: Uses fixed ports for reliability and easy debugging
2. **Application Services**: Uses dynamic allocation via `find_free_port()` to avoid conflicts
3. **Test Isolation**: Each test gets its own port allocation to prevent interference

## Configuration

Port configuration is centralized in `tests/conftest.py`:

```python
SERVICE_PORTS = {
    "etcd_client": 2379,
    "etcd_peer": 2380,
    "nats_client": 4222,
    "nats_http": 8222,
    "dynamo_serve_base": 8000,  # Base port, actual port will be dynamic
}
```

## Health Check Endpoints

| Service | Health Check URL | Description |
|---------|------------------|-------------|
| NATS | `http://localhost:8222/varz` | NATS server status |
| ETCD | `http://localhost:2379/health` | ETCD health status |
| Dynamo Serve | `http://localhost:{port}/health` | Service health |
| Dynamo Serve | `http://localhost:{port}/v1/models` | Model readiness |

## Troubleshooting

### Port Conflicts
- Check if services are already running: `netstat -tulpn | grep :{port}`
- Kill existing processes: `pkill -f "service_name"`
- Use different ports in development if needed

### Service Discovery
- All ports are logged during test execution
- Use `service_ports` fixture to get current port configuration
- Check test output for allocated ports

## Best Practices

1. Always use `find_free_port()` for new test services
2. Document any new fixed ports in this file
3. Use the `service_ports` fixture to access port configuration
4. Include port information in test logging for debugging 
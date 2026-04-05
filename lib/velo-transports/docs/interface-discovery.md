# Interface Discovery & Endpoint Selection

## Problem

When a transport binds to `0.0.0.0:PORT`, the OS listens on all interfaces. But the address advertised to peers must be a routable IP, not `0.0.0.0`. On multi-NIC machines (wifi, eth0, ConnectX), we need to:

1. Discover which interfaces are available and advertise all of them
2. Let peers pick the best interface based on subnet affinity and NUMA topology

## Architecture

```
Builder                          Register (peer side)
  │                                │
  ▼                                ▼
resolve_advertise_endpoints()    parse_endpoints()
  │                                │
  ▼                                ▼
discover_interfaces()            select_best_endpoint()
  │  (getifaddrs + sysfs)           │  (NUMA + subnet matching)
  ▼                                ▼
Vec<InterfaceEndpoint>           SocketAddr
  │
  ▼
rmp_serde::to_vec() → WorkerAddress
```

The work is purely **advertisement** (what addresses to tell peers) and **selection** (which peer address to connect to). No per-interface listeners are needed.

## InterfaceEndpoint

Each discovered interface produces an `InterfaceEndpoint`:

```rust
pub struct InterfaceEndpoint {
    pub name: String,           // "eth0", "mlx5_0"
    pub ip: String,             // "192.168.1.10"
    pub port: u16,              // from the bound listener
    pub prefix_len: u8,         // CIDR prefix: 24 for /24
    pub numa_node: Option<i32>, // from sysfs, None if unavailable
}
```

These are serialized as a msgpack array into the `WorkerAddress` entry for the transport key.

## Discovery

`discover_interfaces()` calls `nix::ifaddrs::getifaddrs()` and filters:

- **Included**: interfaces that are `IFF_UP` with IPv4 or IPv6 addresses
- **Excluded**: loopback (`IFF_LOOPBACK`, `127.0.0.1`, `::1`)
- **Excluded**: IPv6 link-local (`fe80::`) — scope_id routing complexity not worth it

For each interface, NUMA node is read from `/sys/class/net/<name>/device/numa_node`. Returns `None` if the file is missing (virtual interfaces, non-Linux) or contains `-1`.

Prefix length is computed from the netmask via `netmask_to_prefix_len()` — counting leading 1-bits.

## Interface Filtering

Users control which interfaces are advertised via `InterfaceFilter`:

```rust
pub enum InterfaceFilter {
    All,                             // all UP, non-loopback (default)
    ByName(String),                  // specific NIC, e.g. "mlx5_0"
    Explicit(Vec<InterfaceEndpoint>), // manual override (testing)
}
```

Set on the builder:

```rust
TcpTransportBuilder::new()
    .interface_filter(InterfaceFilter::ByName("mlx5_0".into()))
    .numa_hint(1)
    .build()?;
```

## Endpoint Selection

When a peer registers, `select_best_endpoint()` picks the best remote endpoint to connect to. Selection phases (first match wins):

| Phase | Criteria | Rationale |
|-------|----------|-----------|
| 1 | NUMA match + subnet match | Same NUMA node as GPU and on a reachable subnet |
| 2 | NUMA match only | Same NUMA node, even without subnet overlap |
| 3 | Subnet match | Any local interface shares a network with the remote |
| 4 | First non-loopback | Fallback to any routable address |
| 5 | First entry | Last resort |

The `numa_hint` is provided by the caller (typically from `dynamo_memory::numa::get_device_numa_node(gpu_id)`), keeping velo-transports CUDA-agnostic.

## Wire Format & Backward Compatibility

The `WorkerAddress` entry for a transport key contains either:

- **New format**: msgpack-encoded `Vec<InterfaceEndpoint>`
- **Legacy format**: UTF-8 string like `"tcp://127.0.0.1:5555"` or `"grpc://host:port"`

`parse_endpoints()` tries msgpack deserialization first, then falls back to legacy string parsing. This means new code can register peers advertising the old format and vice versa.

## Builder Pre-binding

The builder always pre-binds a `std::net::TcpListener` during `build()` to resolve port 0 to a real port before discovering interfaces. This ensures all advertised endpoints have the correct port. The pre-bound listener is passed through to the transport's `start()` method.

If `from_listener()` is used (common in tests), the provided listener's address is used directly.

## Loopback Fallback

If no non-loopback interfaces are discovered (e.g. in a container), `resolve_advertise_endpoints()` falls back to advertising loopback (`127.0.0.1` or `::1`) with a warning. Same-machine peers should prefer UDS transport anyway.

## Example: Multi-NIC Server

A machine with `eth0` (192.168.1.10/24, NUMA 0) and `mlx5_0` (10.0.0.1/16, NUMA 1):

```
Builder binds 0.0.0.0:0 → OS assigns port 45000
discover_interfaces() returns:
  - eth0:   192.168.1.10, prefix=24, numa=0
  - mlx5_0: 10.0.0.1,     prefix=16, numa=1

WorkerAddress["tcp"] = msgpack([
  {name: "eth0",   ip: "192.168.1.10", port: 45000, prefix_len: 24, numa_node: 0},
  {name: "mlx5_0", ip: "10.0.0.1",     port: 45000, prefix_len: 16, numa_node: 1},
])
```

A peer with GPU on NUMA 1 and local interface `10.0.0.50/16` would select `10.0.0.1:45000` (phase 1: NUMA + subnet match).

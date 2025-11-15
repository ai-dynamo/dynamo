# dynamo-discovery (lib/discovery)

A small, capability-driven discovery layer for the Dynamo runtime. The core idea is to separate “what the application needs to discover” from “how a particular backend provides it,” and to provide a thin manager that composes multiple backends with caching and a concise, stable API.

## Philosophy

- Discovery is application-specific. Each application defines discovery traits that describe the information it needs (e.g., peers, topics, shards, services) and the operations required to work with that information.
- Systems are concrete implementations. A system (e.g., etcd, libp2p, an HTTP microservice, S3, NATS) implements one or more of the discovery traits. Different systems have different capabilities; not every system can implement every trait or policy.
- Managers orchestrate and cache. A manager owns the logic to coordinate multiple systems that implement the same trait, deduplicate concurrent lookups, maintain a local cache, and expose a clean public API tailored for the runtime.

This division lets you grow capabilities without coupling the runtime to any one backend. Traits define the contract; systems provide the plumbing; managers keep the runtime simple and fast.

## Core Concepts

- Discovery traits
  - Define what the application wants to discover and the related operations.
  - Include both public-facing operations (what the runtime calls) and internal operations (used for registration, consistency checks, etc.).
- Systems
  - Backend-specific code that implements one or more discovery traits (and only the parts they can support).
  - Example systems: `etcd` (centralized + TTL), `libp2p` (DHT), an HTTP service client, S3, NATS, in-memory.
  - A system may expose just a subset of traits based on its capability.
- Managers
  - Constructed with one or more system implementations of a trait.
  - Provide a concise, stable public API, while handling caching, coalescing, retries, and capability differences behind the scenes.
  - Allow you to mix-and-match systems for resilience and performance (e.g., fast in-memory cache + remote etcd).

## Capability Model

- Traits describe behavior; systems opt into the parts they can implement.
- The `DiscoverySystem` abstraction can vend one or more trait implementations. If a system cannot support a trait, it simply does not provide it.
- Managers accept a set of trait implementations and will use whatever is provided, with graceful fallback rules (e.g., local cache first, then remote sources).

## Example: Peer Discovery

The peer discovery trait is used by the runtime to translate identifiers into addresses and to manage lifecycle around registration.

- Trait methods (conceptual):
  - `discover_by_worker_id(worker_id) -> PeerInfo`
  - `discover_by_instance_id(instance_id) -> PeerInfo`
  - `register_instance(instance_id, address) -> ()`
  - `unregister_instance(instance_id) -> ()`

- Manager API (public vs. internal):
  - Public: discovery queries
    - `discover_by_worker_id(worker_id)`
    - `discover_by_instance_id(instance_id)`
  - Internal: lifecycle
    - Registration and unregistration are handled by the manager when it is constructed (register the local peer) and during shutdown/cleanup. These are not exposed as public manager methods.

- Why hide registration on the manager?
  - Keeps the runtime call surface minimal and intentional.
  - Enforces consistent lifecycle semantics (checksums, collisions, TTLs) in one place.
  - Avoids leaking backend mechanics into the runtime path.

### How the Manager Works (at a glance)

- On construction, the manager registers the local peer in its local cache and in all configured remote systems that support the peer discovery trait.
- On lookup, it consults the local cache first, then queries remotes if needed. Concurrent lookups for the same key are coalesced into a shared query. Successful remote results are cached locally for future fast paths.

## Typical Wiring

- Choose your systems and build them (e.g., Etcd with TTL, Libp2p, HTTP client, or an in-memory source for tests).
- Extract the trait implementations the runtime needs (e.g., `PeerDiscovery`).
- Create a manager with the local peer and a list of trait impls:

```rust
use std::sync::Arc;
use dynamo_am_discovery::peer::{PeerInfo, WorkerAddress, InstanceId};
use dynamo_am_discovery::peer::manager::PeerDiscoveryManager;
use dynamo_am_discovery::systems::DiscoverySystem; // e.g., etcd system builds this

# async fn example(system: Arc<dyn DiscoverySystem>) -> anyhow::Result<()> {
    let local_instance = InstanceId::new_v4();
    let local_address = WorkerAddress::from_bytes(b"tcp://127.0.0.1:5555".as_slice());
    let local_peer = PeerInfo::new(local_instance, local_address);

    // Get one or more implementations of the peer discovery trait
    let mut sources = Vec::new();
    if let Some(peer_disc) = system.peer_discovery() {
        sources.push(peer_disc);
    }

    // Build the manager that orchestrates cache + remotes
    let manager = PeerDiscoveryManager::new(local_peer, sources).await?;

    // Look up a peer by worker_id or instance_id
    // (The manager will hit local cache first, then remotes as needed.)
    let _maybe = manager.discover_by_worker_id(local_instance.worker_id()).await;
    let _maybe2 = manager.discover_by_instance_id(local_instance).await;
#   Ok(())
# }
```

Note: The manager deliberately keeps registration/unregistration internal. If your application lifecycle requires explicit registration timing, do that by constructing the manager at the appropriate point in startup, and let it handle registration with all configured systems.

## Extending the Crate

- Add a new discovery trait when the application needs to discover a new kind of thing (e.g., shard ownership). Keep the trait small and precise.
- Implement the trait in one or more systems. It’s fine if only some systems can implement it.
- Add a manager for the trait if you need composition, caching, or a slimmer public API for the runtime.
- Keep trait-level semantics strict and documented. Managers can hide backend-specific details while enforcing common policies (e.g., collision detection, address checksums, TTLs).

## Notes on Consistency and Errors

- Systems may enforce additional policies (e.g., TTL expiry in etcd, collision detection, checksum validation). Managers use these and surface simple success/not-found/backend-error semantics to the runtime.
- Local caches accelerate the common path and are populated opportunistically from successful remote lookups.
- Concurrent lookups are deduplicated to reduce load on remote systems.

## Available Systems (examples)

- Etcd-backed system (centralized, TTL-based, keep-alive, transactional collision detection).
- Libp2p-backed system (decentralized DHT).
- HTTP service client.
- In-memory (useful for tests and single-node scenarios).

Not all systems will implement every trait; use the manager’s composition to mix what you need.

## Why This Design?

- Keeps the runtime portable: swap discovery backends without changing call sites.
- Embraces partial capability: wire up the systems that can do the job, skip the rest.
- Minimizes API surface for the runtime: managers expose only the operations the runtime actually needs, while handling lifecycle internally.
- Encourages small traits and pluggable systems, so the crate can evolve without lock-in.

If you’re adding a new trait or system, keep the trait narrowly scoped, stick to clear semantics, and lean on the manager to integrate, cache, and present a clean API to the rest of the runtime.


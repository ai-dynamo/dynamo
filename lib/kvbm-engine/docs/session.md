# Session Module

The session module manages distributed block transfer sessions between
instances. Sessions coordinate the search, staging, and RDMA transfer of
KV cache blocks between a requesting instance (Prefill) and a serving
instance (Decode).

## Protocol Overview

### Onboard Protocol (InitiatorSession ↔ ResponderSession)

Multi-peer search and staging using `OnboardMessage`:

```text
  Initiator (Prefill)              Responder (Decode)
        │                                │
        │──── CreateSession ────────────▶│
        │                                │  search local G2/G3
        │◀─── G2Results ────────────────│
        │◀─── G3Results ────────────────│  (if G3 blocks found)
        │◀─── SearchComplete ───────────│
        │                                │
        │──── HoldBlocks ───────────────▶│
        │◀─── Acknowledged ─────────────│
        │                                │
        │──── StageBlocks ──────────────▶│  G3→G2 staging (optional)
        │◀─── BlocksReady ──────────────│
        │                                │
        │     RDMA pull (remote G2→local G2)
        │                                │
        │──── CloseSession ─────────────▶│
```

When G4 (object storage) is configured, the initiator also runs a parallel
G4 search via internal `G4Results`/`G4LoadComplete` messages (not sent over
the network).

### Unified Session Protocol (SessionHandle ↔ ServerSession)

Point-to-point session using `SessionMessage`:

```text
  Controller (Prefill)             ServerSession (Decode)
        │                                │
        │──── Attach ───────────────────▶│
        │◀─── StateResponse ────────────│  (current state snapshot)
        │                                │
        │──── TriggerStaging ───────────▶│  (if G3 blocks pending)
        │◀─── BlocksStaged ────────────│  (newly staged blocks)
        │                                │
        │     RDMA pull (remote G2→local G2)
        │                                │
        │──── BlocksPulled ─────────────▶│  (release pulled blocks)
        │──── Detach ───────────────────▶│
```

Control can be transferred bidirectionally via `YieldControl`/`AcquireControl`.
For layerwise transfer, `BlocksStaged` includes an optional `layer_range`.

## Session Types

| Session | Role | Protocol | Description |
|---------|------|----------|-------------|
| **ServerSession** | Holds blocks, exposes for pull | SessionMessage | Merged from former EndpointSession + ControllableSession |
| **SessionHandle** | Client-side control | SessionMessage | Attach/detach, state queries, RDMA pulls |
| **InitiatorSession** | Multi-peer search orchestrator | OnboardMessage | Created by InstanceLeader for distributed search |
| **ResponderSession** | Responds to search requests | OnboardMessage | Searches local G2/G3, holds blocks, stages on request |

### ServerSession

Server-side session that holds blocks and exposes them for remote RDMA pull.
Supports two modes:

- **G2-only**: Blocks are already in G2 with pre-assigned layout handles
  (`BlockMetadataMap::Direct`). `TriggerStaging` is a no-op. Created via
  `ServerSession::new_g2_only()` or the `create_server_session()` factory.
- **Staging**: G3 blocks need to be staged to G2. Layout handles are assigned
  round-robin across workers (`BlockMetadataMap::RoundRobin`). Supports
  `auto_stage` option via `ServerSessionOptions`. Created via
  `ServerSession::new_with_staging()`.

Created from `InstanceLeader` via:
- `create_endpoint_session()` / `create_endpoint_session_for_blocks()` — G2-only
- `create_controllable_session()` / `create_controllable_session_with_options()` — with staging

**ServerSessionHandle** provides local control: `notify_layers_ready()` for
layerwise transfer notifications, and `close()` for graceful shutdown.

### InitiatorSession

The requesting side. Sends `CreateSession` to one or more remote instances,
collects results, applies first-responder-wins deduplication, and orchestrates
staging and RDMA pulls. Supports three staging modes:
- **Hold**: Find and hold blocks (G2+G3), no staging
- **Prepare**: Stage G3→G2 everywhere, keep session alive
- **Full**: Stage G3→G2 + RDMA pull remote G2→local G2, session completes

Created by `InstanceLeader::find_matches_with_options()` when
`search_remote == true`.

### ResponderSession

The serving side. Receives `CreateSession`, searches local block managers
(G2 then G3 for remaining), holds matched blocks via `BlockHolder`, and
responds with match results. Handles staging requests and keeps blocks
alive until the session ends.

## Core Building Blocks

### BlockHolder

RAII container for holding blocks during sessions. Tier-agnostic (`BlockHolder<G2>`,
`BlockHolder<G3>`). Blocks are automatically released when the holder is dropped,
preventing leaks even if session handling panics. Key operations: `retain()`,
`release()`, `extend()`, `take_all()`.

### SessionEndpoint

Point-to-point session primitive with a state machine. Encapsulates:
- Identity (`session_id`, `instance_id`)
- State machine (`ControlRole` + `AttachmentState` + `SessionPhase`)
- Message receive channel (`mpsc::Receiver<SessionMessage>`)
- State publication via watch channel
- Transport for sending messages to peer

Used internally by `ServerSession`. Does NOT handle block holding or staging
logic — those are the caller's responsibility.

### SessionHandle

Handle for controlling a remote session. Supports:
- State observation: `current_state()`, `wait_for_ready()`, `wait_for_complete()`
- Control commands: `trigger_staging()`, `mark_blocks_pulled()`, `detach()`
- Bidirectional control: `yield_control()`, `acquire_control()`
- RDMA transfers: `pull_blocks_rdma()`, `pull_blocks_rdma_with_options()`

Used by the controller side (Prefill) to drive a remote `ServerSession` (Decode).

### SessionHandleStateTx

Sender side of the state observation channel. Used by the session receiver
task to forward `StateResponse` and `BlocksStaged` messages into the
watch channel that `SessionHandle` observes.

### Staging

Shared G3→G2 staging logic extracted into `staging::stage_g3_to_g2()`.
Core kernel: allocate G2 destinations → execute local transfer (G3→G2) →
register new blocks with source sequence hashes. Used by `InitiatorSession`,
`ResponderSession`, and `ServerSession` to avoid code duplication.

## Transport Layer

`MessageTransport` is an enum with two variants:

- **`VeloTransport`**: Uses Velo active messages for distributed
  communication between instances.
- **`LocalTransport`**: Direct channel dispatch for in-process testing
  without network overhead.

Methods:
- `send()` — Send an `OnboardMessage` to a target instance
- `send_session()` — Send a `SessionMessage` to a target instance
- `request_metadata()` — RPC call to get remote worker layout metadata for RDMA

## Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| `OnboardMessage` | Initiator ↔ Responder | Block search, hold/drop, staging requests |
| `SessionMessage` | Controller ↔ ServerSession | Attach/detach, control transfer, block ops, state sync |

### OnboardMessage Variants

| Variant | Sender | Description |
|---------|--------|-------------|
| `CreateSession` | Initiator | Start new session with sequence hashes |
| `G2Results` | Responder | G2 search matches (sequence hashes + block IDs) |
| `G3Results` | Responder | G3 search matches (sequence hashes only) |
| `SearchComplete` | Responder | All local searching done |
| `HoldBlocks` | Initiator | Which blocks to hold vs drop |
| `Acknowledged` | Responder | Hold/drop processed |
| `StageBlocks` | Initiator | G3 hashes to stage to G2 |
| `BlocksReady` | Responder | Newly staged G2 blocks ready |
| `ReleaseBlocks` | Initiator | Release specific blocks |
| `CloseSession` | Initiator | Session complete, clean up |
| `G4Results` | Internal | Object storage search results (not sent over network) |
| `G4LoadComplete` | Internal | Object storage load results (not sent over network) |

### SessionMessage Variants

| Variant | Category | Description |
|---------|----------|-------------|
| `Attach` | Connection | Peer attaches with a control role |
| `Detach` | Connection | Peer detaches gracefully |
| `YieldControl` | Control | Sender yields controller role |
| `AcquireControl` | Control | Sender acquires controller role |
| `TriggerStaging` | Block ops | Request G3→G2 staging |
| `HoldBlocks` | Block ops | Request blocks be held |
| `ReleaseBlocks` | Block ops | Release specific blocks |
| `BlocksPulled` | Block ops | Notify blocks were pulled via RDMA |
| `StateResponse` | State sync | Full state snapshot (phase, role, blocks) |
| `BlocksStaged` | State sync | Newly staged blocks (with optional layer range) |
| `Close` | Lifecycle | Graceful session close |
| `Error` | Lifecycle | Report error |

## State Machine

### SessionPhase

Lifecycle of block operations. Staging is optional — blocks already in the
target tier (G2) skip it:

```text
Searching → Holding ──────────────────── Ready → Complete
                    └── Staging ────────┘
                    └── Complete  (direct pull, no staging needed)
                    └── Failed
```

### ControlRole

Dynamic role in session relationship:
- `Neutral` — Initial state, can transition either way
- `Controller` — Issues commands to peer
- `Controllee` — Executes commands from peer

Supports bidirectional transfer via `YieldControl`/`AcquireControl`.

### AttachmentState

Peer connection state: `Unattached` (waiting) or `Attached { peer }` (connected).

## Dispatch Functions

- **`dispatch_onboard_message`**: Routes `OnboardMessage` to per-session task
  channels by session ID. Used by the Velo onboard handler.
- **`dispatch_session_message`**: Routes `SessionMessage` to per-session task
  channels by session ID. Used by the Velo session handler.

## File Structure

```text
session/
├── mod.rs              # Module declarations, dispatch functions, re-exports
├── blocks.rs           # BlockHolder<T> — RAII block container
├── endpoint.rs         # SessionEndpoint — state machine primitive
├── handle.rs           # SessionHandle + SessionHandleStateTx
├── server_session.rs   # ServerSession + ServerSessionHandle + BlockMetadataMap
├── staging.rs          # Shared stage_g3_to_g2() function
├── state.rs            # SessionPhase, ControlRole, AttachmentState
├── messages.rs         # OnboardMessage, SessionMessage, BlockInfo, etc.
├── initiator.rs        # InitiatorSession (multi-peer orchestrator)
├── responder.rs        # ResponderSession (search + hold + stage)
└── transport.rs        # MessageTransport (Velo + Local)
```

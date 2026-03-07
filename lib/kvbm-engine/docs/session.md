# Session Module

The session module manages distributed block transfer sessions between
instances. Sessions coordinate the search, staging, and RDMA transfer of
KV cache blocks between a requesting instance (Prefill) and a serving
instance (Decode).

## Protocol Overview

```text
  Initiator (Prefill)              Responder (Decode)
        │                                │
        │──── FindMatches ──────────────▶│
        │                                │  search local blocks
        │◀─── MatchResult ──────────────│
        │                                │
        │──── TriggerStaging ───────────▶│
        │                                │  G3→G2 staging
        │◀─── BlocksStaged ────────────│
        │                                │
        │──── RDMA Pull ────────────────▶│
        │     (remote G2 → local G2)     │
        │◀─── Complete ─────────────────│
```

## Session Roles

### InitiatorSession

The requesting side. Sends `FindMatches` to one or more remote instances,
collects results, and orchestrates staging and RDMA pulls. Created by
`InstanceLeader` when `search_remote == true`.

### ResponderSession

The serving side. Receives `FindMatches`, searches local block managers,
holds matched blocks via `BlockHolder`, and responds with match results.
Handles staging requests and keeps blocks alive until the session ends.

### ControllableSession

A decode-side session that can be remotely controlled by a prefill instance.
Supports the "inverted control" pattern where the controller attaches,
queries state, triggers staging, and pulls blocks via RDMA.

## Core Building Blocks

### BlockHolder

RAII container for holding blocks during sessions. Tier-agnostic (`BlockHolder<G2>`,
`BlockHolder<G3>`). Blocks are automatically released when the holder is dropped,
preventing leaks even if session handling panics.

### SessionEndpoint

Point-to-point session primitive with a state machine. Encapsulates:
- Identity (session_id, instance_id)
- State machine (`ControlRole` + `AttachmentState` + `SessionPhase`)
- Message receive channel
- State publication via watch channel

### SessionHandle

Handle for controlling a remote session. Supports attach/detach, state
queries, staging triggers, and RDMA block pulls. Used by the controller side
(Prefill) to drive a remote session (Decode).

## Transport Layer

- **`VeloTransport`**: Uses Velo (RPC) active messages for distributed
  communication between instances.
- **`LocalTransport`**: Direct channel dispatch for in-process testing
  without network overhead.

Both implement the `MessageTransport` enum which provides `send`,
`send_remote_session`, and `send_session_message` methods.

## Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| `OnboardMessage` | Initiator → Responder | Block search and staging requests |
| `RemoteSessionMessage` | Bidirectional | Inverted control protocol messages |
| `SessionMessage` | Bidirectional | Unified protocol (replaces both above) |

## State Machine

### SessionPhase

Lifecycle of block operations: `Searching` → `Holding` → `Staging` → `Ready` → `Complete` (or `Failed`).

### ControlRole

Dynamic role in session relationship: `Neutral` (initial), `Controller` (issues commands), `Controllee` (executes commands). Supports bidirectional transfer via yield/acquire.

### AttachmentState

Peer connection state: `Unattached` (waiting) or `Attached { peer }` (connected).

## Dispatch Functions

- **`dispatch_onboard_message`**: Routes `OnboardMessage` to per-session task
  channels by session ID.
- **`dispatch_remote_session_message`**: Routes `RemoteSessionMessage` to either
  `controllable_sessions` (decode-bound: AttachSession, TriggerStaging,
  BlocksPulled, DetachSession) or `remote_sessions` (prefill-bound:
  SessionState, BlocksStaged, SessionError).
- **`dispatch_session_message`**: Routes unified `SessionMessage` by session ID.
  This is the replacement for both legacy dispatch functions.

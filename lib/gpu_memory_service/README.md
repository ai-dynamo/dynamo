# GPU Memory Service (GMS)

## Overview

The **GPU Memory Service (GMS)** is an out-of-process GPU memory manager that decouples ownership of GPU memory from the processes that use it. This enables:

- **Zero-copy sharing** of GPU memory across multiple processes
- **Data survival** across process crashes
- **Fast model loading** via memory import instead of disk I/O for subsequent workers

GMS provides PyTorch integration via `CUDAPluggableAllocator` and pre-built integrations for inference frameworks like **vLLM** and **SGLang**.

## Problem Statement

In traditional LLM inference deployments, each worker process:
1. Loads model weights from disk/network into GPU memory
2. Owns that GPU memory for the lifetime of the process
3. Cannot share weights with other workers on the same GPU

This leads to:
- **Slow worker startup** (weight loading is I/O bound)
- **Memory waste** (duplicate weights when running multiple workers)
- **No crash resilience** (GPU memory lost when process dies)

## Solution Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│  ┌────────────────────┐                  ┌─────────────────────────────────────────┐ │
│  │        GMS         │                  │    GMSClientMemoryManager (Writer)      │ │
│  │                    │                  │                                         │ │
│  │ ┌────────────────┐ │                  │  ┌─────────────────────────────────┐    │ │
│  │ │ Memory Manager │ │ ◄── Unix ───────►│  │         GMS Session             │    │ │
│  │ └────────────────┘ │    Socket        │  └─────────────────────────────────┘    │ │
│  │                    │       +          │                                         │ │
│  │ ┌────────────────┐ │      FD          │  Writer-only: create_mapping, commit    │ │
│  │ │ Session / FSM  │ │  (SCM_RIGHTS)    └─────────────────────────────────────────┘ │
│  │ └────────────────┘ │                                                              │
│  │                    │                  ┌─────────────────────────────────────────┐ │
│  │ ┌────────────────┐ │                  │    GMSClientMemoryManager (Reader)      │ │
│  │ │ Metadata Store │ │                  │                                         │ │
│  │ └────────────────┘ │ ◄── Unix ───────►│  ┌─────────────────────────────────┐    │ │
│  │                    │    Socket        │  │         GMS Session             │    │ │
│  └────────────────────┘       +          │  └─────────────────────────────────┘    │ │
│                              FD          │                                         │ │
│                          (SCM_RIGHTS)    │  Reader-only: create_mapping (import),   │ │
│                                          │               unmap_all_vas, remap      │ │
│                                          └─────────────────────────────────────────┘ │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

GMS follows a client-server architecture where the **server** owns GPU memory allocations and the **clients** map that memory into their own address spaces. The key insight is that the socket connection itself acts as a distributed lock.

### Server

The GMS server runs as an independent process that manages GPU memory without ever mapping it to its own address space. This design allows the server to:

- **Survive GPU driver failures** - no CUDA context means no vulnerability to driver resets
- **Outlive client processes** - memory persists across client crashes
- **Arbitrate access** - enforce single-writer, multiple-reader semantics

The server consists of three main components:

1. **Memory Manager** - Allocates physical GPU memory via CUDA VMM (`cuMemCreate`) and exports shareable file descriptors (`cuMemExportToShareableHandle`). Critically, it never calls `cuMemMap` - clients handle all virtual address mapping. Allocation requests retry on OOM until they succeed or the optional retry timeout is reached.

2. **State Machine (FSM)** - Manages global lock state, waiter coordination, and disconnect cleanup.

3. **Metadata Store / Epoch State** - Owns active vs committed epoch state. Metadata and the memory layout hash are both scoped to a specific epoch and move with that epoch when it is committed.

Each GMS server is responsible for managing memory of only 1 GPU, and does not interact with GMS servers corresponding to other GPUs.

### Client

Clients connect to the server to acquire locks and access GPU memory. The supported client API is:

1. **GMSClientMemoryManager** - High-level client that wraps an internal RPC transport layer and handles all CUDA VMM operations for memory import and mapping safely:
   - Imports file descriptors and converts them to CUDA memory handles
   - Reserves virtual address space and maps physical memory
   - Sets appropriate access permissions (RW for writers, RO for readers)
   - Supports **unmap/remap** for VA-stable memory release under memory pressure

> **Note**: Always use `GMSClientMemoryManager` to interact with GMS from client code. The low-level RPC client is an implementation detail and should not be used directly.

### Memory Allocation and Import Flow

The following diagram shows how `GMSClientMemoryManager` interacts with the server and GPU. **Writers** allocate new memory while **readers** import existing allocations - both flows share the same export/import/map sequence.

```mermaid
sequenceDiagram
    participant C as GMSClientMemoryManager
    participant S as GMS
    participant GPU as GPU Memory

    %% Connection
    C->>S: Connect (Unix Socket)
    C->>S: HandshakeRequest(lock_type)
    S-->>C: HandshakeResponse(granted_lock)

    %% Allocation (Writer only)
    rect rgb(255, 245, 230)
        Note over C,GPU: Writer only: Allocate new memory
        C->>S: AllocateRequest(size, tag)
        S->>GPU: cuMemCreate(size)
        GPU-->>S: handle
        S-->>C: AllocateResponse(allocation_id)
    end

    %% Export/Import (Both Writer and Reader)
    Note over C,GPU: Both Writer and Reader: Export and map
    C->>S: ExportAllocationRequest(allocation_id)
    S->>GPU: cuMemExportToShareableHandle(handle)
    GPU-->>S: fd
    S-->>C: Response + fd (via SCM_RIGHTS)

    C->>GPU: cuMemImportFromShareableHandle(fd)
    C->>GPU: cuMemAddressReserve(size)
    C->>GPU: cuMemMap(va, handle)
    C->>GPU: cuMemSetAccess(va, RW or RO)

    Note over C,GPU: Memory now accessible at VA
```

---

## State Machine

The server maintains a finite state machine (FSM) that governs lock acquisition and memory access. The state is **derived** from the current connections rather than stored explicitly.

### States and Transitions

```mermaid
stateDiagram-v2
    [*] --> EMPTY

    EMPTY --> RW : RW_CONNECT
    RW --> COMMITTED : RW_COMMIT
    RW --> EMPTY : RW_ABORT

    COMMITTED --> RW : RW_CONNECT
    COMMITTED --> RO : RO_CONNECT

    RO --> RO : RO_CONNECT
    RO --> RO : RO_DISCONNECT (not last)
    RO --> COMMITTED : RO_DISCONNECT (last)
```

### State Descriptions

| State | Description | Can Connect RW | Can Connect RO |
|-------|-------------|:--------------:|:--------------:|
| `EMPTY` | No connections, no committed epoch visible | ✓ | ✗ |
| `RW` | Writer connected (exclusive access) | ✗ | ✗ |
| `COMMITTED` | Committed epoch visible to readers, no active connections | ✓ | ✓ |
| `RO` | One or more readers connected (shared access) | ✗ | ✓ |

### Events

| Event | Trigger | Description |
|-------|---------|-------------|
| `RW_CONNECT` | Writer connects | Acquires exclusive write lock, invalidates previous committed visibility immediately, and opens a new active RW epoch |
| `RW_COMMIT` | Writer calls `commit()` | Publishes the current active RW epoch as the committed epoch and releases the lock |
| `RW_ABORT` | Writer disconnects without commit | Drops active RW epoch and returns to `EMPTY` |
| `RO_CONNECT` | Reader connects | Acquires shared read lock |
| `RO_DISCONNECT` | Reader disconnects | Releases shared lock; if last reader, returns to COMMITTED |

### Lock Semantics

The socket connection **is** the lock:

- **Crash resilience**: Connection close (including process crash) automatically releases the lock
- **No explicit unlock**: Eliminates forgotten locks and deadlocks
- **Atomic transitions**: State changes happen atomically with socket operations

### Epoch Model

Epoch creation and publication work like this:

```mermaid
flowchart LR
    A[EMPTY or COMMITTED] -->|RW_CONNECT| B[New epoch N]
    B -->|Allocate memory and write metadata in epoch N| C{Writer outcome}
    C -->|RW_COMMIT| D[Publish epoch N as committed]
    C -->|RW_ABORT| E[Discard epoch N]
    D -->|Next RW_CONNECT| F[New epoch N + 1]
    E -->|Next RW_CONNECT| F
```

- `RW_CONNECT` creates a new active epoch.
- `RW_COMMIT` publishes the current active epoch; it does not create another epoch.
- `RW_ABORT` discards the current active epoch and returns the system to `EMPTY`.
- Every allocation gets an immutable `epoch_id` at creation time.
- Metadata entries belong to exactly one epoch, and the memory layout hash is computed for one committed epoch at a time.
- RO requests are served only from the committed epoch, while RW requests mutate only the active epoch.
- Read RPCs (`export`, allocation lookup/listing, metadata lookup/listing) are scoped to the committed epoch for RO clients and to the active epoch for RW clients.
- `metadata_put` validates allocation ownership and offset bounds, `free` cascades metadata cleanup, and `commit` rejects dangling metadata references.

### Allocation Backpressure on OOM

When a writer requests a new allocation, GMS treats CUDA OOM as a transient condition:

- `cuMemCreate` OOM does **not** immediately fail the request.
- The server retries in a loop and only returns success after allocation is created.
- Server CLI flags:
  - `--alloc-retry-interval` (default `0.5`)
  - `--alloc-retry-timeout` (default unset = wait indefinitely)

This ensures the "new epoch gets new allocations" workflow can wait for memory reclamation instead of racing into immediate OOM failures.

### Guarantees

- GMS guarantees that its own RPCs do not mix committed and active generations, and that `GMSClientMemoryManager.commit()` performs a CUDA synchronize and unmaps the writer's local mappings before publish.
- After local unmap, `commit()` does not attempt in-process recovery. Non-CUDA failures raise, and CUDA VMM failures exit the process.
- The only non-fatal client connection failure is lock acquisition timeout. Other client-side GMS transport, protocol, and server error responses raise.
- Any non-OOM CUDA VMM failure on either client or server is fatal and exits the process.
- On the server, an untrusted client connection is isolated to that connection: transport loss and response-send failures unwind the connection state, and only server invariant violations or CUDA failures kill the server.
- GMS *does not* prove that a disconnected or already-submitted writer has no in-flight GPU work left on the device. The mitigation in this design is that new RW epochs use fresh allocations and may wait for memory reclamation before allocation succeeds.

---

### Server Trust Boundary

```mermaid
flowchart TD
    A[Client event on server connection] --> B{Can server read and decode it?}
    B -- no --> C[Drop connection]
    C --> D[Run disconnect cleanup]
    D --> E[RW_ABORT or RO_DISCONNECT]

    B -- yes --> F{Valid client request?}
    F -- no --> G[Send ErrorResponse]

    F -- yes --> H{Did request expose server invariant failure?}
    H -- yes --> I[Exit server process]

    H -- no --> J[Build response or apply commit]
    J --> K{Can server send response?}
    K -- no --> D
    K -- yes --> L[Continue session or close committed writer]
```

- `Drop connection` means the server stops trusting that socket and unwinds only that connection's lock state.
- After `RW_COMMIT`, disconnect cleanup only closes the committed writer socket; it does not roll the server back to `RW_ABORT`.
- `Valid client request?` covers mode/state violations, unknown requests, and request validation failures like bad metadata offsets.
- `Did request expose server invariant failure?` covers impossible epoch/FSM states and commit-time metadata integrity failures.

## Sequence Diagrams

### Writer Flow (Cold Start)

The first worker loads weights from disk and publishes them to GMS.

```mermaid
sequenceDiagram
    participant W as Writer Process
    participant C as GMSClientMemoryManager
    participant S as GMS

    W->>C: mgr = GMSClientMemoryManager(socket_path, device=0)
    W->>C: mgr.connect(RW)
    C->>S: HandshakeRequest(lock_type=RW)
    S->>S: Session FSM: EMPTY/COMMITTED -> RW
    S->>S: Invalidate prior committed epoch
    S->>S: Create new active epoch
    S-->>C: HandshakeResponse(success=true)

    loop For each tensor
        W->>C: mgr.create_mapping(size=size, tag=tag)
        Note over C,S: See Memory Allocation Flow above
        W->>C: mgr.metadata_put(key, allocation_id, offset, shape)
    end

    W->>C: mgr.commit()
    C->>GPU: synchronize()
    C->>GPU: cuMemUnmap(...) + cuMemRelease(...)
    C->>S: CommitRequest()
    S->>S: Publish active epoch as committed
    S->>S: FSM: RW → COMMITTED
    S-->>C: CommitResponse(success=true)
    W->>C: mgr.connect(RO)
    W->>C: mgr.remap_all_vas()
```

### Reader Flow (Warm Start)

Subsequent workers import weights from GMS instead of loading from disk.

```mermaid
sequenceDiagram
    participant R as Reader Process
    participant C as GMSClientMemoryManager
    participant S as GMS

    R->>C: mgr = GMSClientMemoryManager(socket_path, device=0)
    R->>C: mgr.connect(RO)
    C->>S: HandshakeRequest(lock_type=RO)
    S-->>C: HandshakeResponse(success=true, committed=true)

    R->>C: mgr.metadata_list()
    S-->>C: keys=[...]

    loop For each tensor key
        R->>C: mgr.metadata_get(key)
        S-->>C: allocation_id, offset, shape
        R->>C: mgr.create_mapping(allocation_id=allocation_id)
        Note over C,S: See Memory Import Flow above
    end

    Note over R,C: Keep connection open during inference
```

### Unmap/Remap Flow (Memory Pressure)

Readers can temporarily release GPU memory while preserving virtual address reservations. This enables "shadow engine" patterns where inactive workers release memory for active ones.

```mermaid
sequenceDiagram
    participant R as Reader Process
    participant C as GMSClientMemoryManager
    participant S as GMS
    participant GPU as GPU Memory

    Note over R,GPU: Need to temporarily release GPU memory

    R->>C: mgr.unmap_all_vas()
    C->>GPU: cudaDeviceSynchronize()

    loop For each mapping
        C->>GPU: cuMemUnmap(va)
        C->>GPU: cuMemRelease(handle)
        Note over C: Keep VA reservation!
    end

    R->>C: mgr.disconnect()
    C->>S: Close socket (release RO lock)
    S->>S: FSM: RO → COMMITTED (if last reader)

    Note over R,GPU: GPU memory released, VA preserved
    Note over R,GPU: Another writer could publish a new epoch here

    R->>C: mgr.connect(RO)
    R->>C: mgr.remap_all_vas()
    C->>S: GetStateHashRequest()
    S-->>C: GetStateHashResponse(hash)

    alt hash == saved_hash
        C->>S: Export preserved allocations from committed epoch
        S-->>C: Response + FDs
        C->>GPU: Import handles and remap at preserved VAs
        C-->>R: Remap succeeds and tensor pointers stay valid
    else hash != saved_hash
        C-->>R: StaleMemoryLayoutError
        C-->>R: Re-import from scratch
    end
```

### Auto-Mode (RW_OR_RO)

The `RW_OR_RO` mode automatically selects writer or reader based on server state, simplifying multi-worker deployments.

```mermaid
sequenceDiagram
    participant P as Process
    participant C as GMSClientMemoryManager
    participant S as GMS

    Note over P,S: Auto-mode: try RW only when no committed epoch exists

    P->>C: mgr = GMSClientMemoryManager(socket_path, device=0)
    P->>C: mgr.connect(RW_OR_RO)
    C->>S: HandshakeRequest(lock_type=RW_OR_RO)

    alt No committed weights AND no RW holder
        S->>S: Grant RW lock
        S->>S: FSM: EMPTY → RW
        S-->>C: HandshakeResponse(granted=RW, committed=false)
        Note over P: First process - load from disk
    else Weights already committed
        S->>S: Grant RO lock
        S->>S: FSM: COMMITTED → RO
        S-->>C: HandshakeResponse(granted=RO, committed=true)
        Note over P: Subsequent process - import from GMS
    else RW held by another
        S->>S: Wait until a committed epoch becomes available
        S->>S: Then grant RO from COMMITTED
        S-->>C: HandshakeResponse(granted=RO, committed=true)
        Note over P: Wait for writer to publish committed weights
    end
```

---

## Key Design Decisions

### 1. No VA Mapping on Server

The server never maps memory to virtual addresses (`cuMemMap`). This means:
- **No CUDA context** required on the server
- Server can survive GPU driver resets
- Memory management is fully delegated to clients

### 2. Socket-as-Lock

The socket connection **is** the lock:
- RW lock: Exclusive connection (only one RW at a time)
- RO lock: Shared connection (multiple RO allowed)
- Lock release = socket close (automatic on crash)

Benefits:
- **Crash resilience**: If a reader crashes, its lock is automatically released
- **No explicit unlock**: No forgotten locks or deadlocks

### 3. VA-Stable Unmap/Remap

During `unmap_all_vas()`:
- Physical memory is released (`cuMemUnmap` + `cuMemRelease`)
- VA reservations are **kept** (`cuMemAddressReserve` still valid)

During `remap_all_vas()`:
- Same VAs are reused for mapping
- **Tensor pointers remain valid** (no need to update PyTorch tensors)

### 4. Memory Layout Hash

On commit, the server computes a hash of:
- All allocation IDs, sizes, aligned sizes, tags, and epoch IDs
- All metadata entries

On `remap_all_vas()`, this hash is checked:
- If match: Safe to remap (layout unchanged)
- If mismatch: Raise `StaleMemoryLayoutError` (must re-import)

The hash is tied to the currently committed epoch and is cleared as soon as a writer acquires RW.

**Important**: This detects **structural** changes, not **content** changes.
Weight values can be modified in-place (e.g., RL training updates) as long as the structure is preserved.

---

## Wire Protocol

### Message Format

```
┌──────────────┬────────────────────────────────────────┐
│ Length (4B)  │  msgpack-encoded Message               │
│ big-endian   │                                        │
└──────────────┴────────────────────────────────────────┘
```

### FD Passing

File descriptors are passed out-of-band using Unix socket `SCM_RIGHTS`:

```python
# Server side (send FD)
socket.send_fds(sock, [message_bytes], [fd])

# Client side (receive FD)
data, fds, _, _ = socket.recv_fds(sock, bufsize, maxfds=1)
fd = fds[0] if fds else -1
```

---

## API Reference

### GMSClientMemoryManager

The API is organized in two tiers. **Tier 2 (convenience)** is what integrations normally use. **Tier 1 (atomic)** exposes individual operations for advanced callers.

```python
class GMSClientMemoryManager:
    def __init__(socket_path: str, *, device: int = 0): ...

    # Properties
    @property granted_lock_type: Optional[GrantedLockType]
    @property is_connected: bool
    @property is_unmapped: bool
    @property total_bytes: int

    # --- Tier 1: Connection ---
    def connect(lock_type: RequestedLockType, timeout_ms: Optional[int] = None) -> None
    def disconnect() -> None

    # --- Tier 1: Handle ops (server-side, RW only) ---
    def allocate_handle(size: int, tag: str = "default") -> str     # Returns allocation_id
    def export_handle(allocation_id: str) -> int                     # Returns FD
    def get_handle_info(allocation_id: str) -> GetAllocationResponse
    def free_handle(allocation_id: str) -> bool
    def commit() -> bool                                             # Sync + unmap local mappings + publish; raises on non-CUDA failure after unmap
    def get_memory_layout_hash() -> str
    def list_handles(tag: Optional[str] = None) -> List[GetAllocationResponse]

    # --- Tier 1: VA ops (local) ---
    def reserve_va(size: int) -> int                                 # Returns VA
    def map_va(fd, va, size, allocation_id, tag) -> int              # Returns handle
    def unmap_va(va: int) -> None                                    # Keeps VA reservation
    def free_va(va: int) -> None                                     # Releases VA reservation

    # --- Tier 1: Metadata ---
    def metadata_put(key: str, allocation_id: str, offset_bytes: int, value: bytes) -> bool
    def metadata_get(key: str) -> Optional[Tuple[str, int, bytes]]
    def metadata_list(prefix: str = "") -> List[str]
    def metadata_delete(key: str) -> bool

    # --- Tier 2: Convenience ---
    def create_mapping(allocation_id=None, size=0, tag="default") -> int  # Allocate or import
    def destroy_mapping(va: int) -> None
    def unmap_all_vas() -> None          # Sync + unmap all, preserve VA reservations
    def remap_all_vas() -> None          # Re-import at preserved VAs (checks layout hash)
    def reallocate_all_handles(tag="default") -> None  # Fresh server handles for preserved VAs
    def close() -> None
```

---

## Framework Integration (vLLM / SGLang)

GMS provides pre-built integrations for vLLM and SGLang. Enable GMS by passing `--load-format gms` when launching an engine.

### How It Works

When `--load-format gms` is set:

1. **A GMS server must already be running** for the target GPU device. The engine connects to it via a Unix socket derived from the GPU UUID.
2. The engine uses `RW_OR_RO` mode by default: if no committed epoch exists and no writer holds the lock, the first process gets RW and loads weights from disk. Otherwise clients wait for a committed epoch and then get RO to import published weights.
3. Weights are managed by GMS; KV cache is managed by the framework's own allocator (e.g., vLLM's `CuMemAllocator`).

#### vLLM

```bash
python -m dynamo.vllm \
  --model <model> \
  --load-format gms \
  --worker-cls gpu_memory_service.integrations.vllm.worker:GMSWorker \
  --enable-sleep-mode \
  --gpu-memory-utilization 0.9
```

The integration uses a custom worker class (`GMSWorker`) that:
- Establishes the GMS connection early in `init_device()` so vLLM's `MemorySnapshot` can account for committed weights
- Registers a custom model loader (`GMSModelLoader`) for the `gms` load format
- Patches `torch.cuda.empty_cache` to avoid releasing GMS-managed memory
- Routes weight allocation through a `CUDAPluggableAllocator` backed by GMS

#### SGLang

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --load-format gms \
  --enable-memory-saver \
  --mem-fraction-static 0.9
```

The integration patches `torch_memory_saver` to route weight operations through GMS:
- Weights (`"weights"` / `"model_weights"` tags) go through `GMSMemorySaverImpl`
- Other tags (e.g., `"kv_cache"`) are delegated to the default torch mempool implementation
- The `--enable-memory-saver` flag is required to activate the memory saver pathway

### Shadow Engine Failover (Sleep / Wake)

Both integrations support releasing and reclaiming GPU memory for shadow engine patterns. The API names differ by framework:

- **vLLM**: `sleep` / `wake_up` (via `/engine/sleep` and `/engine/wake_up` HTTP endpoints)
- **SGLang**: `release_memory_occupation` / `resume_memory_occupation` (via the corresponding HTTP endpoints)

Under the hood, sleeping calls `unmap_all_vas()` + `disconnect()` to release GPU memory while preserving VA reservations, and waking calls `connect(RO)` + `remap_all_vas()` to re-import weights at the same virtual addresses. Tensor pointers remain valid, so no model re-initialization is needed.

This enables a shadow engine to release its GPU memory, let a primary engine use the GPU, and then reclaim the memory after the primary is killed.

### Configuration via `model_loader_extra_config`

To force read-only mode (import only, never load from disk), pass `gms_read_only` via the framework's `--model-loader-extra-config` flag:

```bash
--model-loader-extra-config '{"gms_read_only": true}'
```

This forces `RO` lock mode instead of the default `RW_OR_RO` auto-detection. The engine will only import existing committed weights and fail if none are available.

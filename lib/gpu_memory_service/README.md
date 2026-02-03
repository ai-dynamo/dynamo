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
│  │    GMS Server      │                  │    GMSClientMemoryManager (Writer)      │ │
│  │                    │                  │                                         │ │
│  │ ┌────────────────┐ │                  │  ┌─────────────────────────────────┐    │ │
│  │ │ Memory Manager │ │ ◄── Unix ───────►│  │         GMSRPCClient            │    │ │
│  │ └────────────────┘ │    Socket        │  └─────────────────────────────────┘    │ │
│  │                    │       +          │                                         │ │
│  │ ┌────────────────┐ │      FD          │  Writer-only: allocate_and_map, commit  │ │
│  │ │ State Machine  │ │  (SCM_RIGHTS)    └─────────────────────────────────────────┘ │
│  │ └────────────────┘ │                                                              │
│  │                    │                  ┌─────────────────────────────────────────┐ │
│  │ ┌────────────────┐ │                  │    GMSClientMemoryManager (Reader)      │ │
│  │ │ Metadata Store │ │                  │                                         │ │
│  │ └────────────────┘ │ ◄── Unix ───────►│  ┌─────────────────────────────────┐    │ │
│  │                    │    Socket        │  │         GMSRPCClient            │    │ │
│  └────────────────────┘       +          │  └─────────────────────────────────┘    │ │
│                              FD          │                                         │ │
│                          (SCM_RIGHTS)    │  Reader-only: import_allocation,        │ │
│                                          │               unmap, remap              │ │
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

1. **Memory Manager** - Allocates physical GPU memory via CUDA VMM (`cuMemCreate`) and exports shareable file descriptors (`cuMemExportToShareableHandle`). Critically, it never calls `cuMemMap` - clients handle all virtual address mapping.

2. **State Machine (FSM)** - Manages the global lock state and enforces access rules that ensures consistency across multiple clients. See [State Machine](#state-machine) below for details.

3. **Metadata Store** - Key-value store for tensor metadata (shapes, dtypes, offsets), enabling clients to reconstruct model structure.

### Client

Clients connect to the server to acquire locks and access GPU memory. Two client classes are provided:

1. **GMSRPCClient** - Low-level RPC client for direct protocol access. Handles socket communication, msgpack serialization, and file descriptor passing via `SCM_RIGHTS`. The socket connection **is** the lock - connection lifetime equals lock lifetime, providing automatic crash resilience.

2. **GMSClientMemoryManager** - High-level client that wraps `GMSRPCClient` and handles all CUDA VMM operations for memory import and mapping safely:
   - Imports file descriptors and converts them to CUDA memory handles
   - Reserves virtual address space and maps physical memory
   - Sets appropriate access permissions (RW for writers, RO for readers)
   - Supports **unmap/remap** for VA-stable memory release under memory pressure

> **Note**: Always use `GMSClientMemoryManager` to interact with GMS from client code. The low-level `GMSRPCClient` is an implementation detail and should not be used directly.

### Memory Allocation and Import Flow

The following diagram shows how `GMSClientMemoryManager` interacts with the server and GPU. **Writers** allocate new memory while **readers** import existing allocations - both flows share the same export/import/map sequence.

```mermaid
sequenceDiagram
    participant C as GMSClientMemoryManager
    participant S as GMS Server
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
    C->>S: ExportRequest(allocation_id)
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
| `EMPTY` | No connections, no committed weights | ✓ | ✗ |
| `RW` | Writer connected (exclusive access) | ✗ | ✗ |
| `COMMITTED` | Weights published, no active connections | ✓ | ✓ |
| `RO` | One or more readers connected (shared access) | ✗ | ✓ |

### Events

| Event | Trigger | Description |
|-------|---------|-------------|
| `RW_CONNECT` | Writer connects | Acquires exclusive write lock |
| `RW_COMMIT` | Writer calls `commit()` | Publishes weights, releases lock |
| `RW_ABORT` | Writer disconnects without commit | Discards allocations, releases lock |
| `RO_CONNECT` | Reader connects | Acquires shared read lock |
| `RO_DISCONNECT` | Reader disconnects | Releases shared lock; if last reader, returns to COMMITTED |

### Lock Semantics

The socket connection **is** the lock:

- **Crash resilience**: Connection close (including process crash) automatically releases the lock
- **No explicit unlock**: Eliminates forgotten locks and deadlocks
- **Atomic transitions**: State changes happen atomically with socket operations

---

## Sequence Diagrams

### Writer Flow (Cold Start)

The first worker loads weights from disk and publishes them to GMS.

```mermaid
sequenceDiagram
    participant W as Writer Process
    participant C as GMSClientMemoryManager
    participant S as GMS Server

    W->>C: new GMSClientMemoryManager(mode=RW)
    C->>S: HandshakeRequest(lock_type=RW)
    S-->>C: HandshakeResponse(success=true)

    loop For each tensor
        W->>C: allocate_and_map(size, tag)
        Note over C,S: See Memory Allocation Flow above
        W->>C: metadata_put(key, allocation_id, offset, shape)
    end

    W->>C: commit()
    C->>S: CommitRequest()
    S->>S: FSM: RW → COMMITTED
    S-->>C: CommitResponse(success=true)
```

### Reader Flow (Warm Start)

Subsequent workers import weights from GMS instead of loading from disk.

```mermaid
sequenceDiagram
    participant R as Reader Process
    participant C as GMSClientMemoryManager
    participant S as GMS Server

    R->>C: new GMSClientMemoryManager(mode=RO)
    C->>S: HandshakeRequest(lock_type=RO)
    S-->>C: HandshakeResponse(success=true, committed=true)

    R->>C: metadata_list()
    S-->>C: keys=[...]

    loop For each tensor key
        R->>C: metadata_get(key)
        S-->>C: allocation_id, offset, shape
        R->>C: import_allocation(allocation_id)
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
    participant S as GMS Server
    participant GPU as GPU Memory

    Note over R,GPU: Need to temporarily release GPU memory

    R->>C: unmap()
    C->>GPU: cudaDeviceSynchronize()

    loop For each mapping
        C->>GPU: cuMemUnmap(va)
        C->>GPU: cuMemRelease(handle)
        Note over C: Keep VA reservation!
    end

    C->>C: Save memory_layout_hash
    C->>S: Close socket (release RO lock)
    S->>S: FSM: RO → COMMITTED (if last reader)

    Note over R,GPU: GPU memory released, VA preserved
    Note over R,GPU: Another writer could modify weights here

    R->>C: remap()
    C->>S: HandshakeRequest(lock_type=RO)
    S->>S: FSM: COMMITTED → RO
    S-->>C: HandshakeResponse(success=true)

    C->>S: GetStateHashRequest()
    S-->>C: GetStateHashResponse(hash)

    alt hash == saved_hash
        loop For each preserved VA
            C->>S: ExportRequest(allocation_id)
            S-->>C: Response + fd
            C->>GPU: cuMemImportFromShareableHandle(fd)
            C->>GPU: cuMemMap(same_va, handle)
            Note over C: Tensors valid at same addresses!
        end
    else hash != saved_hash
        C-->>R: StaleMemoryLayoutError
        Note over R: Must re-import from scratch
    end
```

### Auto-Mode (RW_OR_RO)

The `RW_OR_RO` mode automatically selects writer or reader based on server state, simplifying multi-worker deployments.

```mermaid
sequenceDiagram
    participant P as Process
    participant C as GMSClientMemoryManager
    participant S as GMS Server

    Note over P,S: Auto-mode: Writer if first, Reader if weights exist

    P->>C: new GMSClientMemoryManager(mode=RW_OR_RO)
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
        S->>S: Wait for RO availability
        S->>S: FSM: COMMITTED → RO
        S-->>C: HandshakeResponse(granted=RO, committed=true)
        Note over P: Wait for writer to finish
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

During `unmap()`:
- Physical memory is released (`cuMemUnmap` + `cuMemRelease`)
- VA reservations are **kept** (`cuMemAddressReserve` still valid)

During `remap()`:
- Same VAs are reused for mapping
- **Tensor pointers remain valid** (no need to update PyTorch tensors)

### 4. Memory Layout Hash

On commit, the server computes a hash of:
- All allocation IDs, sizes, and tags
- All metadata entries

On `remap()`, this hash is checked:
- If match: Safe to remap (layout unchanged)
- If mismatch: Raise `StaleMemoryLayoutError` (must re-import)

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

```python
class GMSClientMemoryManager:
    def __init__(
        socket_path: str,
        mode: RequestedLockType,  # RW, RO, or RW_OR_RO
        device: int = 0,
        timeout_ms: Optional[int] = None,
    ): ...

    # Properties
    @property mode: GrantedLockType  # Actual granted mode
    @property is_connected: bool
    @property is_unmapped: bool
    @property total_bytes: int

    # Allocation (RW only)
    def allocate_and_map(size: int, tag: str = "default") -> int  # Returns VA
    def free_mapping(va: int) -> None
    def clear_all() -> int  # Returns count cleared

    # Import (RO or RW)
    def import_allocation(allocation_id: str) -> int  # Returns VA

    # Metadata (RW: put/delete, RO: get/list)
    def metadata_put(key: str, allocation_id: str, offset: int, value: bytes) -> bool
    def metadata_get(key: str) -> Optional[Tuple[str, int, bytes]]
    def metadata_list(prefix: str = "") -> List[str]
    def metadata_delete(key: str) -> bool

    # Lifecycle
    def commit() -> bool  # Publish weights, release RW lock
    def switch_to_read(timeout_ms: Optional[int] = None) -> None
    def unmap() -> None   # Release RO lock, preserve VAs
    def remap(timeout_ms: Optional[int] = None) -> bool
    def close() -> None
```


## Limitations

1. **Single-GPU per server**: Each GMS server manages one GPU device
2. **CUDA VMM required**: Requires a GPU with Virtual Memory Management support. Check at runtime via `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` - there is no guaranteed minimum compute capability
3. **No content validation**: Remap doesn't detect in-place weight modifications

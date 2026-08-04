# Binary Injector Design

## Problem

When restoring a container, the agent needs to make binaries (`criu`, `nsrestore`,
`cuda-checkpoint`) available inside the placeholder container's mount namespace.
The placeholder image intentionally does not ship these binaries — they are injected
at runtime from the agent.

Currently `injectBinaries()` in `agent/internal/executor/inject.go` is a stub.
This document defines the packages, interfaces, and structs needed to implement it,
and how they integrate into the restore flow.

---

## Constraints

1. **Go is multithreaded.** Linux forbids `setns(CLONE_NEWNS)` in a multithreaded
   process. All namespace-switching mount operations must be delegated to a
   single-threaded C helper binary.
2. **Mount source is not assumed to be a PVC.** The source can be a host path,
   an emptyDir, a projected volume, a future image layer — the abstraction must
   not leak that assumption.
3. **Mounts are temporary.** Every mount injected at restore time must be cleaned
   up after nsrestore exits, regardless of success or failure.
4. **Implementations must be testable.** All side-effecting operations are behind
   interfaces so unit tests can swap in fakes.

---

## Package Map

```
agent/internal/
├── namespacemount/         # Low-level: mount into a target PID's namespace
│   ├── mounter.go          # Mounter interface, MountHandle, MountOptions
│   ├── nsbindmount.go      # NSBindMountHelper — delegates to C helper
│   ├── nsentertmpfs.go     # NSEnterTmpfsMounter — tmpfs via nsenter+mount
│   └── fake.go             # FakeMounter for tests
│
├── binaryinjector/         # High-level: inject agent binaries into a namespace
│   ├── injector.go         # Injector interface, InjectionHandle, Config
│   ├── nsmount_injector.go # NSMountInjector — uses namespacemount.Mounter
│   └── noop.go             # NoopInjector for unit tests
│
└── executor/
    ├── inject.go           # Wires Config → NSMountInjector (replaces stub)
    ├── restore.go          # Calls injector; restore flow unchanged otherwise
    └── executor.go         # (new) Executor struct holding pluggable strategies
```

---

## `namespacemount` — Namespace Mount Abstraction

Responsibility: mount a source into the mount namespace of a target process.
Has no knowledge of what is being mounted or why.

### `MountOptions`

```go
// MountOptions configures a single namespace-aware mount operation.
type MountOptions struct {
    ReadOnly bool   // mount with MS_RDONLY
    FSType   string // filesystem type: "bind", "tmpfs", "overlay", …
    Data     string // comma-separated mount options passed to the kernel
}
```

### `MountHandle`

```go
// MountHandle represents an active mount inside a foreign namespace.
// The owner must call Unmount when the mount is no longer needed.
type MountHandle interface {
    // Unmount detaches the mount from the target namespace.
    // Idempotent — safe to call multiple times.
    Unmount(ctx context.Context) error

    // TargetPath is the dst path as seen inside the target namespace.
    TargetPath() string
}
```

### `Mounter`

```go
// Mounter mounts src at dst inside the mount namespace identified by targetPID.
// The returned MountHandle must be Unmount-ed by the caller.
//
// Implementations are responsible for:
//   - entering the target namespace without violating Go's threading model
//   - ensuring the destination path exists before mounting
//   - propagating context cancellation to any subprocess they launch
type Mounter interface {
    Mount(ctx context.Context, targetPID int, src, dst string, opts MountOptions) (MountHandle, error)
}
```

### `NSBindMountHelper` (implementation)

Uses the single-threaded C binary `ns-bind-mount` (already present in the agent
image). The helper uses `open_tree(OPEN_TREE_CLONE)` to capture the source mount
subtree before `setns(CLONE_NEWNS)`, then `move_mount` to attach it in the target
namespace.

```go
// NSBindMountHelper implements Mounter via the ns-bind-mount C helper.
type NSBindMountHelper struct {
    // HelperPath is the absolute host path to the ns-bind-mount binary.
    // Defaults to DefaultNSBindMountHelperPath if empty.
    HelperPath string
}
```

Unmount delegates to `nsenter -t <pid> -m -- umount <dst>` so unmounting also
stays namespace-scoped.

### `NSEnterTmpfsMounter` (implementation)

For situations where the source is not a host path but a new ephemeral filesystem
(e.g., CRIU work dir). Uses `nsenter -t <pid> -m -- mount -t tmpfs tmpfs <dst>`.

```go
type NSEnterTmpfsMounter struct{}
```

Unmount is `nsenter -t <pid> -m -- umount <dst>`.

---

## `binaryinjector` — Binary Injection Orchestration

Responsibility: given a target container (identified by placeholder PID), make
the agent's restore binaries available inside its mount namespace at a known path.

### `Config`

```go
// Config is the static (per-agent-deployment) configuration for injection.
type Config struct {
    // SourceDir is the host-side directory where binaries live.
    // Default: types.AgentBinDir ("/usr/local/sbin").
    // Must not be empty.
    SourceDir string

    // DestinationDir is the path INSIDE the placeholder where binaries appear.
    // Default: types.SnapshotBinDir ("/tmp/snapshot-binaries").
    // Must not be empty.
    DestinationDir string

    // NSBindMountHelperPath is the absolute path to the C helper binary.
    // Default: DefaultNSBindMountHelperPath.
    NSBindMountHelperPath string
}

func (c *Config) Validate() error { … }
func (c *Config) WithDefaults() Config { … }
```

### `InjectionHandle`

```go
// InjectionHandle represents a live binary injection into a placeholder namespace.
// The caller must call Cleanup after nsrestore returns.
type InjectionHandle interface {
    // BinPath returns the in-namespace absolute path to the named binary.
    // Example: handle.BinPath("nsrestore") → "/tmp/snapshot-binaries/nsrestore"
    BinPath(name string) string

    // Cleanup unmounts the injected directory from the target namespace.
    // Idempotent — safe to call from a defer even if Inject partially failed.
    Cleanup(ctx context.Context) error
}
```

### `Injector`

```go
// Injector mounts the agent's restore binaries into a placeholder container's
// mount namespace and returns a handle for cleanup.
//
// The injection is scoped to a single restore attempt. Create one Injector per
// agent startup (it holds no per-restore state) and call Inject per restore.
type Injector interface {
    Inject(ctx context.Context, log logr.Logger, placeholderPID int) (InjectionHandle, error)
}
```

### `NSMountInjector` (implementation)

```go
// NSMountInjector implements Injector using a namespacemount.Mounter.
// Construction: binaryinjector.New(cfg, namespacemount.NewNSBindMountHelper(""))
type NSMountInjector struct {
    cfg    Config
    mounter namespacemount.Mounter
}

func New(cfg Config, mounter namespacemount.Mounter) (*NSMountInjector, error) { … }
```

`Inject`:
1. Validate `cfg.SourceDir` exists on the host (`os.Stat`).
2. Call `mounter.Mount(ctx, placeholderPID, cfg.SourceDir, cfg.DestinationDir, MountOptions{ReadOnly: true})`.
3. Return an `injectionHandle` wrapping the `MountHandle`.

`Cleanup` on the returned handle calls `MountHandle.Unmount`.

### `NoopInjector` (test helper)

```go
// NoopInjector implements Injector without performing any mount operations.
// BinPath returns paths constructed from the configured DestinationDir.
type NoopInjector struct {
    DestinationDir string
}
```

---

## `executor` Changes

### `Executor` struct (new file `executor.go`)

Rather than passing the injector through `RestoreRequest` (which is a transport
struct), introduce an `Executor` that owns pluggable strategies:

```go
// Executor holds strategy implementations for the restore flow.
// Construct once at agent startup; call Restore per restore request.
type Executor struct {
    BinaryInjector binaryinjector.Injector
}

func NewExecutor(injector binaryinjector.Injector) *Executor {
    return &Executor{BinaryInjector: injector}
}

// Restore is the existing top-level function, now a method on Executor.
func (e *Executor) Restore(ctx context.Context, rt snapshotruntime.Runtime, log logr.Logger, req RestoreRequest) (int, error) { … }
```

The package-level `Restore` function in `restore.go` can remain as a thin
adapter that constructs a default `Executor` — preserving the existing call
sites in `controller` with no changes needed until the injector is wired.

### `inject.go` (replaces stub)

```go
// DefaultInjector constructs the production NSMountInjector from agent config.
// Called once at startup from the controller or agent main.
func DefaultInjector(cfg binaryinjector.Config) (binaryinjector.Injector, error) {
    mounter := namespacemount.NewNSBindMountHelper(cfg.NSBindMountHelperPath)
    return binaryinjector.New(cfg, mounter)
}
```

### Integration point in `restore.go`

```go
// Phase 2: Inject restore binaries into the placeholder's mount namespace.
handle, err := e.BinaryInjector.Inject(ctx, log, snap.PlaceholderPID)
if err != nil {
    return 0, fmt.Errorf("binary injection failed: %w", err)
}
defer func() {
    if cleanupErr := handle.Cleanup(ctx); cleanupErr != nil {
        log.Error(cleanupErr, "Failed to cleanup binary injection mounts; mounts may leak")
    }
}()

// NSRestorePath is now derived from the injection handle, not hardcoded.
req.NSRestorePath = handle.BinPath(types.NSRestoreBin)
```

---

## Error Handling Strategy

| Failure point | Action | Rationale |
|---|---|---|
| `cfg.Validate()` at startup | Fatal — agent won't start | Config is static; no point continuing |
| `Inject` fails (helper not found, target PID gone) | Return error, abort restore | No partial state; placeholder is still healthy |
| `Inject` fails after partial mounts | `Cleanup` is called in the error path; any unmount errors are logged | Leak is preferable to masking the original error |
| `nsrestore` fails | `Cleanup` still runs via `defer` | Deferred regardless of restore outcome |
| `Cleanup` fails | Log error, do not propagate | Restore already succeeded; leaked mount is a node-level concern, not a restore failure |
| Context cancelled mid-inject | Helper subprocess receives SIGKILL via `exec.CommandContext`; `Cleanup` runs | Fast path: don't leave mounts if the restore is abandoned |

---

## Integration Roadmap

### Stage 1 — This PR (packages only, no wiring)
- Create `agent/internal/namespacemount/` with `Mounter`, `MountHandle`, `MountOptions`.
- Create `NSBindMountHelper` and `NSEnterTmpfsMounter` implementations.
- Create `agent/internal/binaryinjector/` with `Injector`, `InjectionHandle`, `Config`.
- Create `NSMountInjector` and `NoopInjector`.
- Create `executor.go` with the `Executor` struct.
- Replace the `injectBinaries()` stub with `DefaultInjector` wiring.
- All tests pass; `Restore` behavior is unchanged (injector is not yet called by default).

### Stage 2 — Wiring
- In `agent/internal/controller/`, construct `DefaultInjector` from `AgentConfig`.
- Replace call to package-level `executor.Restore` with `(*Executor).Restore`.
- Update `AgentConfig`/`RestoreSpec` with `BinaryInjectorConfig` sub-struct.
- Integration test: mock `Mounter`; assert `Inject` and `Cleanup` are called.

### Stage 3 — C helper integration
- Confirm `ns-bind-mount` binary is built and available at `DefaultNSBindMountHelperPath`.
- Add E2E test that verifies binaries appear at `types.SnapshotBinDir` inside a
  placeholder container after `Inject`.

### Stage 4 — Cleanup of existing CRIU work-dir tmpfs mount
- Move the `nsenter -m -- mount -t tmpfs` for the CRIU work dir into
  `NSEnterTmpfsMounter` so all namespace mounts go through the same abstraction.
- `Executor` can hold a second `namespacemount.Mounter` for the work dir or use
  a composite `InjectionHandle` that manages multiple mounts atomically.

---

## What Is Explicitly Out of Scope

- **Checkpoint-side mounts.** `namespacemount` is restore-only; checkpoint uses
  `runtime.BuildMountPolicy` which lives in `runtime/mounts.go` and is unrelated.
- **Storage backend abstraction.** How the checkpoint data reaches the node (PVC,
  object store, NFS) is a `types.StorageSpec` concern, not `binaryinjector`'s.
- **Image baking.** The design assumes binaries always come from the running agent
  container, not from an OCI image layer. A future `ImageLayerSource` could
  implement `namespacemount.Mounter` without touching `binaryinjector`.

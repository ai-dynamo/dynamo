# Code Review — feat/restore-ns-mount-stage

Reviewed files: `deploy/snapshot/internal/{injection,nsbindmount,executor,types,controller}/`,
`deploy/snapshot/cmd/`.

---

## Summary

This PR replaces the static `nsRestorePath` config field with a dynamic binary injection
mechanism: the agent now bind-mounts its own binary directory into the placeholder
container's mount namespace (via a new `ns-bind-mount` C helper) before calling nsrestore,
so nsrestore is always reachable under a fixed in-namespace path regardless of how the
container image was built.

The overall design is clean and well-layered. Issues below are ordered most-critical first.

---

## Findings

### 1. Duplicate error-message prefix causes confusing double-wrapping

**Severity: Minor bug**

`NSMountInjector.Inject` already wraps the mount error:

```go
// injection/injector.go:91
return nil, fmt.Errorf("mount agent bundle into placeholder: %w", err)
```

Then `Restore` wraps it again with the same string:

```go
// executor/restore.go:68
return 0, fmt.Errorf("mount agent bundle into placeholder: %w", err)
```

The resulting error chain reads:
`"mount agent bundle into placeholder: mount agent bundle into placeholder: ns-bind-mount …"`

**Fix:** Remove one of the two wrapping layers. The most natural choice is to drop the
wrapping in `Restore` and let the error surface directly from `Inject`:

```go
handle, err := injector.Inject(ctx, snap.PlaceholderPID)
if err != nil {
    return 0, err
}
```

---

### 2. `nsbindmount/nsbindmount.go` missing `//go:build linux` build tag

**Severity: Portability**

`nsbindmount_test.go` correctly gates itself with `//go:build linux` because it exercises
`/proc/<pid>/ns/mnt`. The implementation `nsbindmount.go` has no such tag. While the code
compiles on macOS/Windows (there are no OS-specific imports), attempting to run the
mount path on a non-Linux host produces a confusing runtime error rather than a build
failure.

The test guard being more restrictive than the implementation is a mismatch that will bite
anyone running tests on macOS CI. Add the build tag to the implementation:

```go
//go:build linux
```

or, if the package must compile cross-platform, add a stub `nsbindmount_stub.go` with
`//go:build !linux` that returns `errors.New("nsbindmount not supported on this OS")`.

---

### 3. No compile-time interface satisfaction check for `NSMountInjector`

**Severity: Nit**

`NSMountInjector` implicitly satisfies `Injector` but there is no blank-identifier check.
Add to `injector.go`:

```go
var _ Injector = (*NSMountInjector)(nil)
```

This catches signature drift at compile time without a test.

---

### 4. `paths.go` — `SnapshotBinDir` built via string concatenation

**Severity: Style**

```go
// paths.go
agentBinDir    = "/snapshot-binaries"
SnapshotBinDir = "/tmp" + agentBinDir
```

This relies on `agentBinDir` having a leading slash. If that invariant ever breaks (e.g.
the slash is dropped during a rename), the resulting path `/tmpsnapshot-binaries` silently
becomes wrong. Use an explicit literal or `filepath.Join`:

```go
SnapshotBinDir = "/tmp/snapshot-binaries"
```

---

### 5. `nsbindmount.go` — cleanup unmount on `os.Open` failure uses `exec.Command` (no context)

**Severity: Observation / intentional**

In `ExecMounter.Mount`, if `os.Open("/proc/<pid>/ns/mnt")` fails after the mount
succeeds, the code runs a best-effort cleanup:

```go
exec.Command(m.binaryPath, "umount", pidStr, dst).CombinedOutput()
```

This uses `exec.Command` (no deadline) rather than `exec.CommandContext`. If the helper
binary hangs, the `Mount` call blocks indefinitely. Adding a short timeout here would be
consistent with `Unmount`'s 10-second budget. Low risk in practice since this is an error
path, but worth aligning.

---

### 6. `restore.go` — defer captures a potentially-cancelled context

**Severity: Informational (no real bug)**

```go
defer func() {
    if cleanupErr := handle.Cleanup(ctx); cleanupErr != nil { … }
}()
```

`ctx` here is the restore context, which may be cancelled by the time the defer fires
(context deadline, caller cancel). The actual `Unmount` implementation in `mountHandle`
ignores the passed context and creates its own `context.Background()` + 10 s timeout, so
this is not a live bug. However it creates a misleading API surface — `Cleanup` accepts a
`context.Context` that is in practice discarded. Consider documenting this in the
`InjectionHandle.Cleanup` doc comment, or dropping the context from the `Cleanup`
signature entirely.

---

### 7. Missing test: cleanup is deferred on the error path of `execNSRestore`

**Severity: Test coverage gap**

There is no test verifying that `handle.Cleanup` is called (via the defer) when
`execNSRestore` returns an error. The `injection` package tests confirm `Cleanup` calls
`Unmount`, but the end-to-end defer in `Restore` is untested for the failure branch. A
unit test of `Restore` with a mock injector and a failing `execNSRestore` (e.g. bad
`nsrestore` path) would close this gap.

---

### 8. `TestExecMounter_Mount_NsFdOpenFailure` doesn't verify cleanup-unmount was attempted

**Severity: Minor coverage gap**

The test confirms that `Mount` returns an error when `/proc/<pid>/ns/mnt` can't be opened,
but it doesn't assert that the best-effort cleanup `exec.Command(… "umount" …)` was called.
Because the fake binary exits 0, the cleanup silently succeeds (or is never reached if pid
`math.MaxInt32` is truly dead). Capturing cleanup calls in the fake binary's log file
would make the test's intent explicit.

---

## Positive notes

- Layering is clean: `nsbindmount` knows nothing about binaries; `injection` knows nothing
  about restore orchestration; `executor` only depends on the `Injector` interface.
- `sync.Once` + stored error in `mountHandle.Unmount` is the right pattern for idempotent
  cleanup.
- Holding the `/proc/<pid>/ns/mnt` fd in the handle to survive PID recycling is the
  correct fix for a real race; the comment explaining it is clear.
- Removing `NSRestorePath` from config simplifies operator deployment — no more per-node
  binary path configuration.
- Test coverage for the happy path across all three packages is solid.

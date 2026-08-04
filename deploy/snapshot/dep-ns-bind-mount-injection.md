# DEP: Binaries and Checkpoints Mount for CRIU Restore

**Labels:** `dep:draft` · `enhancement` · `fault-tolerance` · `snapshot`
**Related PR:** #12542

---

## Summary

Introduce a C helper binary (`ns-bind-mount`) and a Go mount package that
bind-mount the agent's restore binaries and checkpoint data into a placeholder
container's mount namespace before CRIU restore runs. The Go runtime cannot call
`setns(CLONE_NEWNS)` directly (multithreaded restriction), so a single-threaded
C subprocess performs the namespace switch using `open_tree(2)` / `move_mount(2)`.

---

## Motivation

CRIU restore requires the `nsrestore` agent binary to be visible inside the
placeholder container's mount namespace at a known path. Volume mounts and init
containers were considered but add scheduling complexity and Kubernetes coupling.
A targeted bind-mount performed just before restore is simpler and self-contained.

---

## Proposal

1. **C helper** (`ns-bind-mount`) — three subcommands:
   - `mount <pid> <src> <dst> [ro]` — clones the source tree with `open_tree(OPEN_TREE_CLONE)` before entering the namespace, then attaches with `move_mount`.
   - `umount <pid> <dst>` — legacy PID-based unmount (kept for compatibility).
   - `umount-fd <ns_fd> <dst>` — preferred unmount; accepts an open `/proc/<pid>/ns/mnt` fd passed via `ExtraFiles` to avoid PID-reuse races.

2. **Go `nsbindmount` package** — `ExecMounter` wraps the C helper. After a successful mount it opens `/proc/<pid>/ns/mnt` and holds the fd in the `MountHandle` so `Unmount` is safe even after the target process exits.

3. **Go `nsbindmount` package** — `NSMounter` binds a configurable `SourceDir` into `DestinationDir` inside the placeholder namespace and returns a `MountHandle` with `BinPath(name)` for in-namespace binary paths and `Cleanup(ctx)` for teardown.

---

## Requirements

1. The C helper must be a single-threaded process; Go must not call `setns` directly.
2. `Mount` must be atomic from the caller's perspective: on any failure the namespace is left unchanged.
3. `Unmount` must be idempotent and bounded (10 s timeout on the subprocess).
4. `MountHandle` must hold a namespace fd, not a PID, to eliminate the PID-reuse window between mount and cleanup.
5. Requires Linux 5.2+ (`open_tree` / `move_mount` syscalls).

---

## References

- PR #12542 — implementation
- `deploy/snapshot/cmd/ns-bind-mount/main.c`
- `deploy/snapshot/internal/nsbindmount/`
- `deploy/snapshot/internal/injection/`

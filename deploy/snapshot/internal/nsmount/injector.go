// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nsmount bind-mounts a directory into a foreign process's mount
// namespace via the ns-bind-mount C helper (cmd/ns-bind-mount).
package nsmount

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"
)

const (
	// SnapshotBinSrc is the agent-side directory containing the binary bundle.
	SnapshotBinSrc = "/snapshot-binaries"
	// SnapshotBinDst is the mount destination inside the placeholder namespace.
	SnapshotBinDst = "/tmp/snapshot-binaries"
	// CheckpointDst is where the selected checkpoint artifact is exposed inside
	// the placeholder namespace.
	CheckpointDst = "/tmp/checkpoint"
)

// The two attribute sets this agent mounts with. They are named for what they
// hold rather than what uses them; the wrapper types below bind each one.
var (
	// The binary bundle is executed inside the target, so exec stays allowed.
	readOnly = MountOptions{ReadOnly: true, NoSuid: true, NoDev: true}
	// The checkpoint artifact is data brought in from shared storage. Letting
	// the target execute it would turn the mount into a code-injection channel.
	readOnlyNoExec = MountOptions{ReadOnly: true, NoSuid: true, NoDev: true, NoExec: true}
)

// MountPoint represents an active bind-mount of a directory inside a foreign
// namespace. The caller must call Unmount when done.
type MountPoint interface {
	// Path returns the in-namespace absolute path to the named binary.
	// name must be a single path element with no separators or dot-dot components.
	// Example: mp.Path("nsrestore") → "/tmp/snapshot-binaries/nsrestore", nil
	Path(name string) (string, error)

	// Unmount removes the bind-mount from the target namespace. strict selects
	// umount2(0) over MNT_DETACH, so a busy mount is reported rather than
	// deferred; the caller chooses per mount.
	// Idempotent — safe to call from a defer even if Mount partially failed.
	// A non-nil error means the mount may still be active; callers must treat
	// this as fatal and exit so Kubernetes restarts into a clean namespace.
	Unmount(ctx context.Context, strict bool) error

	// NsFd returns the pinned mount-namespace fd opened at Mount time.
	// Use /proc/self/fd/<Fd()> as the --mount= argument to nsenter so the
	// correct namespace is entered even if the original PID is recycled.
	// Valid until Unmount is called. Test mocks may return nil.
	NsFd() *os.File
}

// NSMounter mounts directories into a placeholder container's mount namespace
// and returns a MountPoint for later cleanup. It holds no paths of its own:
// each call names its source and destination.
type NSMounter struct {
	mounter mounter
	log     logr.Logger
}

// probeKernelMountAPI verifies that mount_setattr (Linux 5.12) is available.
// Called with a NULL attr, so a kernel that implements the syscall fails in
// copy_struct_from_user with EFAULT; ENOSYS means the kernel is too old.
func probeKernelMountAPI() error {
	err := unix.MountSetattr(-1, "", 0, nil)
	if errors.Is(err, syscall.ENOSYS) {
		return fmt.Errorf("ns-bind-mount requires Linux 5.12+ (mount_setattr): kernel does not support it")
	}
	return nil
}

// New returns an NSMounter backed by the ns-bind-mount binary at its default
// location. It errors if the helper binary is missing or if the kernel does not
// support mount_setattr (Linux 5.12+), so a misconfigured node fails at agent
// startup rather than at first restore.
func New(log logr.Logger) (*NSMounter, error) {
	if err := probeKernelMountAPI(); err != nil {
		return nil, err
	}
	m, err := newExecMounter(defaultBinaryPath, log)
	if err != nil {
		return nil, err
	}
	return newWithMounter(m, log), nil
}

// newWithMounter is the test seam: it takes an arbitrary mounter so tests can
// exercise Mount without the ns-bind-mount subprocess.
func newWithMounter(m mounter, log logr.Logger) *NSMounter {
	return &NSMounter{mounter: m, log: log}
}

// Mount bind-mounts src at dst inside pid's mount namespace with exactly the
// given attributes. It is the general operation; the wrapper types below bind
// the attribute sets this agent actually uses. How the mount is released is not
// decided here: the caller passes that to MountPoint.Unmount.
func (nsm *NSMounter) Mount(ctx context.Context, pid int, src, dst string, opts MountOptions) (MountPoint, error) {
	nsm.log.Info("mounting into placeholder namespace", "pid", pid, "src", src, "dst", dst, "attrs", opts.attrArgs())

	ref, err := nsm.mounter.Mount(ctx, pid, src, dst, opts)
	if err != nil {
		return nil, err
	}

	nsm.log.Info("mounted into placeholder namespace", "pid", pid, "dst", ref.TargetPath())
	return &mountPoint{mount: ref}, nil
}

// ReadOnlyMounter mounts read-only, nosuid and nodev, leaving execution
// allowed for callers that run binaries out of the mount.
type ReadOnlyMounter struct{ nsm *NSMounter }

func (w ReadOnlyMounter) Mount(ctx context.Context, pid int, src, dst string) (MountPoint, error) {
	return w.nsm.Mount(ctx, pid, src, dst, readOnly)
}

// ReadOnlyNoExecMounter additionally forbids execution, for data the target
// should be able to read but never run.
type ReadOnlyNoExecMounter struct{ nsm *NSMounter }

func (w ReadOnlyNoExecMounter) Mount(ctx context.Context, pid int, src, dst string) (MountPoint, error) {
	return w.nsm.Mount(ctx, pid, src, dst, readOnlyNoExec)
}

// ReadOnly and ReadOnlyNoExec return the wrapper for each attribute set.
func (nsm *NSMounter) ReadOnly() ReadOnlyMounter             { return ReadOnlyMounter{nsm: nsm} }
func (nsm *NSMounter) ReadOnlyNoExec() ReadOnlyNoExecMounter { return ReadOnlyNoExecMounter{nsm: nsm} }

// mountPoint wraps a mountRef to expose the MountPoint surface:
// Path resolves binary names relative to the mounted directory,
// and Unmount delegates to the underlying mount's Unmount.
type mountPoint struct {
	mount mountRef
}

// Path returns the in-namespace absolute path to the named binary.
// name must be a single path element: callers pass literals, and anything
// else could redirect a privileged exec outside the mounted bundle.
func (h *mountPoint) Path(name string) (string, error) {
	if name == "" || name == "." || name == ".." || strings.ContainsRune(name, os.PathSeparator) {
		return "", fmt.Errorf("nsmount: invalid binary name %q", name)
	}
	return filepath.Join(h.mount.TargetPath(), name), nil
}

func (h *mountPoint) Unmount(ctx context.Context, strict bool) error {
	return h.mount.Unmount(ctx, strict)
}

func (h *mountPoint) NsFd() *os.File {
	return h.mount.NsFd()
}

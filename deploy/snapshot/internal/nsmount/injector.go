// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nsmount bind-mounts a directory into a foreign process's mount
// namespace via the ns-bind-mount C helper (cmd/ns-bind-mount).
package nsmount

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"
)

// MountPoint represents an active bind-mount of a directory inside a foreign
// namespace. The caller must call Unmount when done.
type MountPoint interface {
	// Path returns the in-namespace absolute path to the named binary.
	// name must be a single path element with no separators or dot-dot components.
	// Example: mp.Path("nsrestore") → "/tmp/snapshot-binaries/nsrestore", nil
	Path(name string) (string, error)

	// Unmount removes the bind-mount from the target namespace.
	// Idempotent — safe to call from a defer even if Mount partially failed.
	// A non-nil error means the mount may still be active; callers must treat
	// this as fatal and exit so Kubernetes restarts into a clean namespace.
	Unmount(ctx context.Context) error
}

// NSMounter mounts a source directory into a placeholder container's mount
// namespace and returns a MountPoint for later cleanup.
type NSMounter struct {
	src     string
	dst     string
	mounter mounter
	log     logr.Logger
}

// New returns an NSMounter backed by the ns-bind-mount binary at its default
// location. It errors if the helper binary is missing, so a misconfigured
// node fails at startup rather than at the first mount.
// Requires Linux 5.12+ (mount_setattr; open_tree/move_mount need only 5.2).
func New(src, dst string, log logr.Logger) (*NSMounter, error) {
	m, err := newExecMounter(defaultBinaryPath, log)
	if err != nil {
		return nil, err
	}
	return newWithMounter(src, dst, m, log)
}

// newWithMounter is the test seam: it takes an arbitrary mounter so tests can
// exercise Mount without the ns-bind-mount subprocess.
func newWithMounter(src, dst string, m mounter, log logr.Logger) (*NSMounter, error) {
	return &NSMounter{src: src, dst: dst, mounter: m, log: log}, nil
}

// Mount bind-mounts src into dst inside the mount namespace of pid.
// The caller must call MountPoint.Unmount when done.
func (nsm *NSMounter) Mount(ctx context.Context, pid int) (MountPoint, error) {
	nsm.log.Info("mounting agent bundle into placeholder namespace", "pid", pid, "src", nsm.src, "dst", nsm.dst)

	ref, err := nsm.mounter.Mount(ctx, pid, nsm.src, nsm.dst, MountOptions{ReadOnly: true})
	if err != nil {
		return nil, err
	}

	nsm.log.Info("agent bundle mounted", "pid", pid, "dst", ref.TargetPath())
	return &mountPoint{mount: ref}, nil
}

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

func (h *mountPoint) Unmount(ctx context.Context) error {
	return h.mount.Unmount(ctx)
}

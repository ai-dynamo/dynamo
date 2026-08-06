// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nsmount stages the agent's restore binaries inside a placeholder
// container's mount namespace before CRIU restore can run.
//
// The agent's binary bundle is bind-mounted read-only from agentBinDir into
// SnapshotBinDir inside the target namespace, via the ns-bind-mount C helper
// (cmd/ns-bind-mount). The placeholder image ships none of these binaries.
package nsmount

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"
)

const (
	// agentBinDir is the host-side directory in the agent image holding the
	// binaries to inject.
	agentBinDir = "/snapshot-binaries"

	// SnapshotBinDir is where that bundle appears inside the placeholder
	// namespace. This is the bundle-layout contract: nsrestore resolves criu,
	// the criu plugin directory, and the cuda-checkpoint helpers underneath it,
	// and deploy/snapshot/Dockerfile builds the bundle to match.
	SnapshotBinDir = "/tmp" + agentBinDir
)

// hasAllowedPrefix reports whether path equals prefix or begins with prefix+"/".
func hasAllowedPrefix(path, prefix string) bool {
	rest, found := strings.CutPrefix(path, prefix)
	return found && (rest == "" || rest[0] == '/')
}

// Handle represents a live binary mount into a placeholder namespace.
// The caller must call Cleanup after nsrestore returns.
type Handle interface {
	// BinPath returns the in-namespace absolute path to the named binary.
	// name must be a single path element with no separators or dot-dot components.
	// Example: handle.BinPath("nsrestore") → "/tmp/snapshot-binaries/nsrestore", nil
	BinPath(name string) (string, error)

	// Cleanup unmounts the mounted directory from the target namespace.
	// Idempotent — safe to call from a defer even if Mount partially failed.
	Cleanup(ctx context.Context) error
}

// NSMounter mounts the agent's restore binaries into a placeholder
// container's mount namespace and returns a handle for cleanup.
type NSMounter struct {
	src     string
	dst     string
	mounter mounter
	log     logr.Logger
}

// New returns an NSMounter backed by the ns-bind-mount binary at its
// default location. src and dst must begin with the allowed prefixes enforced
// by the C helper. It errors if the paths are invalid or the helper binary is
// missing, so a misconfigured node fails at startup rather than at the first restore.
// Requires Linux 5.12+ (mount_setattr; open_tree/move_mount need only 5.2).
func New(src, dst string, log logr.Logger) (*NSMounter, error) {
	if !hasAllowedPrefix(src, agentBinDir) {
		return nil, fmt.Errorf("nsmount: src must start with %s: %s", agentBinDir, src)
	}
	if !hasAllowedPrefix(dst, SnapshotBinDir) {
		return nil, fmt.Errorf("nsmount: dst must start with %s: %s", SnapshotBinDir, dst)
	}
	m, err := newExecMounter(defaultBinaryPath, log)
	if err != nil {
		return nil, err
	}
	return newWithMounter(src, dst, m, log)
}

// newWithMounter is the test seam: it takes an arbitrary mounter so tests can
// exercise Mount without the ns-bind-mount subprocess. It skips path validation
// so tests can pass arbitrary paths.
func newWithMounter(src, dst string, m mounter, log logr.Logger) (*NSMounter, error) {
	return &NSMounter{src: src, dst: dst, mounter: m, log: log}, nil
}

// Mount bind-mounts src into dst inside the mount namespace of pid.
// The caller must call Handle.Cleanup when done.
func (nsm *NSMounter) Mount(ctx context.Context, pid int) (Handle, error) {
	nsm.log.Info("mounting agent bundle into placeholder namespace", "pid", pid, "src", nsm.src, "dst", nsm.dst)

	handle, err := nsm.mounter.Mount(ctx, pid, nsm.src, nsm.dst, MountOptions{ReadOnly: true})
	if err != nil {
		return nil, err
	}

	nsm.log.Info("agent bundle mounted", "pid", pid, "dst", handle.TargetPath())
	return &bundleHandle{mount: handle}, nil
}

// bundleHandle wraps a MountHandle to expose the bundle-oriented Handle
// surface: BinPath resolves binary names relative to the mounted directory,
// and Cleanup delegates to the underlying mount's Unmount.
type bundleHandle struct {
	mount MountHandle
}

// BinPath returns the in-namespace absolute path to the named binary.
// name must be a single path element: callers pass literals, and anything
// else could redirect a privileged exec outside the mounted bundle.
func (h *bundleHandle) BinPath(name string) (string, error) {
	if name == "" || name == "." || name == ".." || strings.ContainsRune(name, os.PathSeparator) {
		return "", fmt.Errorf("nsmount: invalid binary name %q", name)
	}
	return filepath.Join(h.mount.TargetPath(), name), nil
}

func (h *bundleHandle) Cleanup(ctx context.Context) error {
	return h.mount.Unmount(ctx)
}

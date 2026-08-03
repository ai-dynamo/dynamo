// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nsbindmount wraps the ns-bind-mount C helper binary, which performs
// cross-namespace bind mounts using open_tree(2)/move_mount(2) after entering
// the target process's mount namespace via setns(CLONE_NEWNS). The C helper is
// necessary because Go's multithreaded runtime cannot call setns(CLONE_NEWNS)
// directly.
package nsbindmount

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
)

const (
	binaryName        = "ns-bind-mount"
	defaultBinaryPath = "/usr/local/sbin/" + binaryName
)

// MountOptions configures a single namespace-aware mount operation.
type MountOptions struct {
	ReadOnly bool // mount with MS_RDONLY
}

// MountHandle represents an active mount inside a foreign namespace.
// The owner must call Unmount when the mount is no longer needed.
type MountHandle interface {
	// Unmount detaches the mount from the target namespace.
	// Idempotent — safe to call multiple times.
	Unmount(ctx context.Context) error

	// TargetPath returns the dst path as seen inside the target namespace.
	TargetPath() string
}

// Mounter mounts src at dst inside the mount namespace identified by pid.
// The returned MountHandle must be Unmount-ed by the caller.
type Mounter interface {
	Mount(ctx context.Context, pid int, src, dst string, opts MountOptions) (MountHandle, error)
}

// ExecMounter implements Mounter by invoking the ns-bind-mount C helper as a
// subprocess.
type ExecMounter struct {
	binaryPath string
	log        logr.Logger
}

// New returns an ExecMounter using the default ns-bind-mount binary path.
func New(log logr.Logger) *ExecMounter {
	return &ExecMounter{binaryPath: defaultBinaryPath, log: log}
}

// NewWithBinary returns an ExecMounter using a custom binary path (e.g. for tests).
func NewWithBinary(path string, log logr.Logger) *ExecMounter {
	return &ExecMounter{binaryPath: path, log: log}
}

// mountHandle is the concrete MountHandle returned by ExecMounter.Mount.
type mountHandle struct {
	binaryPath string
	pidStr     string
	dst        string
	log        logr.Logger
	once       sync.Once
	unmountErr error
}

func (h *mountHandle) TargetPath() string { return h.dst }

func (h *mountHandle) Unmount(_ context.Context) error {
	h.once.Do(func() {
		// Use a fresh context with a hard timeout so a hung umount does not block
		// indefinitely. Parent context is intentionally not forwarded: cleanup must
		// complete even if the caller's context is already cancelled.
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, h.binaryPath, "umount", h.pidStr, h.dst).CombinedOutput()
		if err != nil {
			h.log.Error(err, "failed to unmount from namespace", "dst", h.dst, "output", strings.TrimSpace(string(out)))
			h.unmountErr = fmt.Errorf("ns-bind-mount umount %s: %w\noutput: %s", h.dst, err, strings.TrimSpace(string(out)))
			return
		}
		h.log.Info("unmounted from namespace", "dst", h.dst)
	})
	return h.unmountErr
}

// Mount bind-mounts src (in the current namespace) to dst inside the mount
// namespace of pid. It uses open_tree(OPEN_TREE_CLONE) to capture the source
// before the namespace switch so the mount is independent of the source tree.
func (m *ExecMounter) Mount(ctx context.Context, pid int, src, dst string, opts MountOptions) (MountHandle, error) {
	pidStr := strconv.Itoa(pid)
	args := []string{pidStr, src, dst}
	if opts.ReadOnly {
		args = append(args, "ro")
	}
	out, err := exec.CommandContext(ctx, m.binaryPath, args...).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("ns-bind-mount %s -> %s: %w\noutput: %s", src, dst, err, strings.TrimSpace(string(out)))
	}
	m.log.Info("mounted into namespace", "src", src, "dst", dst, "readonly", opts.ReadOnly, "pid", pid)

	return &mountHandle{
		binaryPath: m.binaryPath,
		pidStr:     pidStr,
		dst:        dst,
		log:        m.log,
	}, nil
}

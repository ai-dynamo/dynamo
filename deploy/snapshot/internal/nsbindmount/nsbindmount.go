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

	"github.com/go-logr/logr"
)

const (
	binaryName        = "ns-bind-mount"
	defaultBinaryPath = "/usr/local/sbin/" + binaryName
)

// MountOptions controls bind mount behavior.
type MountOptions struct {
	ReadOnly bool
}

// Mounter performs a bind mount of src into the mount namespace of pid at dst.
// It returns an unmount func that reverts the mount; the func is safe to call
// after the target process has exited.
type Mounter interface {
	Mount(ctx context.Context, pid int, src, dst string, opts MountOptions) (unmount func() error, err error)
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

// Mount bind-mounts src (in the current namespace) to dst inside the mount
// namespace of pid. It uses open_tree(OPEN_TREE_CLONE) to capture the source
// before the namespace switch so the mount is independent of the source tree.
func (m *ExecMounter) Mount(ctx context.Context, pid int, src, dst string, opts MountOptions) (func() error, error) {
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

	return func() error {
		// umount tolerates ENOENT/EINVAL: the mount may already be gone if CRIU
		// cleaned up the namespace during restore.
		out, err := exec.Command(m.binaryPath, "umount", pidStr, dst).CombinedOutput()
		if err != nil {
			m.log.Error(err, "failed to unmount from namespace", "dst", dst, "output", strings.TrimSpace(string(out)))
			return err
		}
		m.log.Info("unmounted from namespace", "dst", dst)
		return nil
	}, nil
}

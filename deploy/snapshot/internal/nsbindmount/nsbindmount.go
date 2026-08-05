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
	"os"
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

	// nsFdChildNum is the fd number that the ns/mnt file descriptor will have
	// inside the child process. Go's exec package maps ExtraFiles[i] to fd
	// (3+i) — after stdin(0), stdout(1), stderr(2). nsFd is the only entry in
	// ExtraFiles, so it lands at fd 3. If ExtraFiles ever gains additional
	// entries before nsFd, this constant must be updated to match.
	nsFdChildNum = 3
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
// Returns an error if the binary is not found on the host so callers fail
// at startup rather than at the first mount operation.
func New(log logr.Logger) (*ExecMounter, error) {
	return NewWithBinary(defaultBinaryPath, log)
}

// NewWithBinary returns an ExecMounter using a custom binary path (e.g. for tests).
// Returns an error if the binary does not exist at path.
func NewWithBinary(path string, log logr.Logger) (*ExecMounter, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("%s binary not found at %s: %w", binaryName, path, err)
	}
	return &ExecMounter{binaryPath: path, log: log}, nil
}

// mountHandle is the concrete MountHandle returned by ExecMounter.Mount.
// It holds an open /proc/<pid>/ns/mnt fd captured at mount time so that
// Unmount can re-enter the correct namespace even after the target process
// has exited and its PID been recycled by the kernel.
type mountHandle struct {
	binaryPath string
	nsFd       *os.File // /proc/<pid>/ns/mnt opened at Mount time
	dst        string
	log        logr.Logger
	once       sync.Once
	unmountErr error
}

func (h *mountHandle) TargetPath() string { return h.dst }

func (h *mountHandle) Unmount(_ context.Context) error {
	h.once.Do(func() {
		defer h.nsFd.Close()
		// Use a fresh context with a hard timeout so a hung umount does not block
		// indefinitely. Parent context is intentionally not forwarded: cleanup must
		// complete even if the caller's context is already cancelled.
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		// Pass the ns fd via ExtraFiles; it lands at fd nsFdChildNum in the child.
		cmd := exec.CommandContext(ctx, h.binaryPath, "umount-fd", strconv.Itoa(nsFdChildNum), h.dst)
		cmd.ExtraFiles = []*os.File{h.nsFd}
		out, err := cmd.CombinedOutput()
		if err != nil {
			h.log.Error(err, "failed to unmount from namespace", "dst", h.dst, "output", strings.TrimSpace(string(out)))
			h.unmountErr = fmt.Errorf("ns-bind-mount umount-fd %s: %w\noutput: %s", h.dst, err, strings.TrimSpace(string(out)))
			return
		}
		h.log.Info("unmounted from namespace", "dst", h.dst)
	})
	return h.unmountErr
}

// Mount bind-mounts src (in the current namespace) to dst inside the mount
// namespace of pid. It uses open_tree(OPEN_TREE_CLONE) to capture the source
// before the namespace switch so the mount is independent of the source tree.
// After a successful mount it opens /proc/<pid>/ns/mnt and holds the fd in
// the returned handle so Unmount can re-enter the namespace without relying
// on the PID still being alive.
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

	nsFd, err := os.Open(fmt.Sprintf("/proc/%d/ns/mnt", pid))
	if err != nil {
		// Mount succeeded but we cannot hold the namespace reference — unmount
		// synchronously so the caller never receives an un-unmountable handle.
		// Use a short timeout; if cleanup hangs we still return the original error.
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if out, umErr := exec.CommandContext(cleanupCtx, m.binaryPath, "umount", pidStr, dst).CombinedOutput(); umErr != nil {
			m.log.Error(umErr, "cleanup unmount failed after ns fd open error", "dst", dst, "output", strings.TrimSpace(string(out)))
		}
		return nil, fmt.Errorf("open /proc/%d/ns/mnt: %w", pid, err)
	}

	return &mountHandle{
		binaryPath: m.binaryPath,
		nsFd:       nsFd,
		dst:        dst,
		log:        m.log,
	}, nil
}

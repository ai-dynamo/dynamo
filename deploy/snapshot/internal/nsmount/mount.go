// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nsmount

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
	nsMntNsPathFmt    = "/proc/%d/ns/mnt"

	// nsFdChildNum is the fd number that the ns/mnt file descriptor will have
	// inside the child process. Go's exec package maps ExtraFiles[i] to fd
	// (3+i) — after stdin(0), stdout(1), stderr(2). nsFd is the only entry in
	// ExtraFiles, so it lands at fd 3. If ExtraFiles ever gains additional
	// entries before nsFd, this constant must be updated to match.
	nsFdChildNum = 3

	// unmountTimeout bounds a single ns-bind-mount cleanup invocation so a hung
	// umount cannot block the caller indefinitely.
	unmountTimeout = 10 * time.Second
)

// MountOptions configures a single namespace-aware mount operation.
type MountOptions struct {
	ReadOnly bool // mount with MS_RDONLY
}

// mountRef represents an active mount inside a foreign namespace.
// The owner must call Unmount when the mount is no longer needed.
type mountRef interface {
	// Unmount detaches the mount from the target namespace.
	// Idempotent — safe to call multiple times.
	Unmount(ctx context.Context) error

	// TargetPath returns the dst path as seen inside the target namespace.
	TargetPath() string
}

// mounter mounts src at dst inside the mount namespace identified by pid.
// It exists so tests can substitute the ns-bind-mount subprocess; production
// callers get an execMounter via New and never name this type.
type mounter interface {
	Mount(ctx context.Context, pid int, src, dst string, opts MountOptions) (mountRef, error)
}

// execMounter implements mounter by invoking the ns-bind-mount C helper as a
// subprocess. The helper performs cross-namespace bind mounts using
// open_tree(2)/move_mount(2) after entering the target process's mount
// namespace via setns(CLONE_NEWNS); it is a separate binary because Go's
// multithreaded runtime cannot call setns(CLONE_NEWNS) directly.
type execMounter struct {
	binaryPath string
	log        logr.Logger
}

// newExecMounter returns an execMounter for the ns-bind-mount binary at path.
// It errors if the binary is absent so callers fail at startup rather than at
// the first mount operation.
func newExecMounter(path string, log logr.Logger) (*execMounter, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("%s binary not found at %s: %w", binaryName, path, err)
	}
	return &execMounter{binaryPath: path, log: log}, nil
}

// execMountRef is the concrete mountRef returned by execMounter.Mount.
type execMountRef struct {
	binaryPath string
	nsFd       *os.File // /proc/<pid>/ns/mnt pinned before the subprocess runs; held so Unmount re-enters the same namespace without relying on the PID
	dst        string
	createdDst bool // true if the mount helper created dst; controls rmdir on cleanup
	log        logr.Logger
	once       sync.Once
	unmountErr error
}

func (h *execMountRef) TargetPath() string { return h.dst }

func (h *execMountRef) Unmount(_ context.Context) error {
	h.once.Do(func() {
		defer h.nsFd.Close()
		// Fresh context with a hard timeout. The parent context is intentionally
		// not forwarded: cleanup must complete even if the caller's context is
		// already cancelled.
		ctx, cancel := context.WithTimeout(context.Background(), unmountTimeout)
		defer cancel()
		// Pass the ns fd via ExtraFiles; it lands at fd nsFdChildNum in the child.
		args := []string{"umount-fd", strconv.Itoa(nsFdChildNum), h.dst}
		if h.createdDst {
			args = append(args, "created")
		}
		cmd := exec.CommandContext(ctx, h.binaryPath, args...)
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
// namespace of pid. It opens /proc/<pid>/ns/mnt *before* launching the helper
// so the namespace is pinned against PID reuse, then passes the fd to the
// mount-fd subcommand via ExtraFiles. The fd is retained in the returned handle
// so Unmount can re-enter the namespace without relying on the PID.
func (m *execMounter) Mount(ctx context.Context, pid int, src, dst string, opts MountOptions) (mountRef, error) {
	// Pin the namespace fd before calling the helper so mount and cleanup
	// provably act on the same namespace regardless of PID reuse.
	nsFdPath := fmt.Sprintf(nsMntNsPathFmt, pid)
	nsFd, err := os.Open(nsFdPath)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", nsFdPath, err)
	}

	args := []string{"mount-fd", strconv.Itoa(nsFdChildNum), src, dst}
	if opts.ReadOnly {
		args = append(args, "ro")
	}

	cmd := exec.CommandContext(ctx, m.binaryPath, args...)
	cmd.ExtraFiles = []*os.File{nsFd}
	var stdout strings.Builder
	var stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		nsFd.Close()
		return nil, fmt.Errorf("ns-bind-mount mount-fd %s -> %s: %w\noutput: %s", src, dst, err, strings.TrimSpace(stderr.String()))
	}
	m.log.Info("mounted into namespace", "src", src, "dst", dst, "readonly", opts.ReadOnly, "pid", pid)

	createdDst := strings.Contains(stdout.String(), "created_dst=1")

	return &execMountRef{
		binaryPath: m.binaryPath,
		nsFd:       nsFd,
		dst:        dst,
		createdDst: createdDst,
		log:        m.log,
	}, nil
}

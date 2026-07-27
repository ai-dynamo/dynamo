// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"golang.org/x/sys/unix"
)

const maxGPUUUIDOrderBytes = 16 * 1024

// ReadGPUUUIDOrderFile reads the workload-provided GPU order through the
// agent's host view of the container root.
func ReadGPUUUIDOrderFile(hostPID int) ([]byte, error) {
	if hostPID <= 0 {
		return nil, fmt.Errorf("invalid host PID %d for GPU UUID order", hostPID)
	}
	path := filepath.Join(
		HostProcPath,
		strconv.Itoa(hostPID),
		"root",
		snapshotprotocol.SnapshotControlMountPath,
		snapshotprotocol.GPUUUIDsFile,
	)
	return readGPUUUIDOrderFile(path)
}

func readGPUUUIDOrderFile(path string) ([]byte, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK, 0)
	if err != nil {
		return nil, fmt.Errorf("open GPU UUID order file: %w", err)
	}
	file := os.NewFile(uintptr(fd), path)
	if file == nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("create GPU UUID order file handle")
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat GPU UUID order file: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("GPU UUID order file must be regular, got mode %s", info.Mode())
	}
	if info.Size() > maxGPUUUIDOrderBytes {
		return nil, fmt.Errorf(
			"GPU UUID order file is too large: %d bytes exceeds %d",
			info.Size(),
			maxGPUUUIDOrderBytes,
		)
	}

	data, err := io.ReadAll(io.LimitReader(file, maxGPUUUIDOrderBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read GPU UUID order file: %w", err)
	}
	if len(data) > maxGPUUUIDOrderBytes {
		return nil, fmt.Errorf(
			"GPU UUID order file grew beyond %d bytes while reading",
			maxGPUUUIDOrderBytes,
		)
	}
	return data, nil
}

// WriteControlSentinel writes a sentinel file into the workload container's
// snapshot-control volume at SnapshotControlMountPath/<name>, accessed through
// the agent's /host/proc/<pid>/root view of the container's mount namespace.
//
// hostPID must be a PID inside the container's mount namespace (the container
// task PID is the canonical choice). The sentinel is observed by the workload
// via inotify on the control directory; it replaces the SIGUSR1/SIGCONT
// agent-to-workload signals that previously required the workload to run as
// PID 1.
//
// The write uses create-then-rename so the workload never observes a partial
// file.
func WriteControlSentinel(hostPID int, name string) error {
	if hostPID <= 0 {
		return fmt.Errorf("invalid host PID %d for control sentinel %q", hostPID, name)
	}
	dir := filepath.Join(HostProcPath, strconv.Itoa(hostPID), "root", snapshotprotocol.SnapshotControlMountPath)
	return writeSentinelInDir(dir, name)
}

func writeSentinelInDir(dir, name string) error {
	tmpPath := filepath.Join(dir, "."+name+".tmp")
	finalPath := filepath.Join(dir, name)
	if err := os.WriteFile(tmpPath, []byte("done\n"), 0o644); err != nil {
		return fmt.Errorf("write temp sentinel %s: %w", tmpPath, err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename sentinel %s -> %s: %w", tmpPath, finalPath, err)
	}
	return nil
}

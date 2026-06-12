// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

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
	dir := controlDirForHostPID(hostPID)
	return writeSentinelInDir(dir, name)
}

// WriteRestoreControlSentinel writes the restore sentinel into both control
// views that need it after CRIU restore: the restored workload's root for the
// polling process and the placeholder root for kubelet startup probes.
func WriteRestoreControlSentinel(placeholderHostPID, restoredPID int, name string) error {
	dirs, err := restoreControlDirs(placeholderHostPID, restoredPID)
	if err != nil {
		return err
	}
	for _, dir := range dirs {
		if err := writeSentinelInDir(dir, name); err != nil {
			return fmt.Errorf("write restore control sentinel in %s: %w", dir, err)
		}
	}
	return nil
}

// WriteRestoredControlSentinel writes a sentinel through the CRIU-restored
// process root instead of the placeholder container root.
//
// restoredPID is namespace-relative: it is the PID returned by CRIU while
// nsrestore is running inside the placeholder container's PID namespace. The
// snapshot agent reaches that procfs view through the placeholder's root at
// /host/proc/<placeholderHostPID>/root/proc/<restoredPID>/root.
func WriteRestoredControlSentinel(placeholderHostPID, restoredPID int, name string) error {
	dir, err := restoredControlDir(placeholderHostPID, restoredPID)
	if err != nil {
		return err
	}
	return writeSentinelInDir(dir, name)
}

func controlDirForHostPID(hostPID int) string {
	return filepath.Join(
		HostProcPath,
		strconv.Itoa(hostPID),
		"root",
		snapshotprotocol.SnapshotControlMountPath,
	)
}

func restoreControlDirs(placeholderHostPID, restoredPID int) ([]string, error) {
	restoredDir, err := restoredControlDir(placeholderHostPID, restoredPID)
	if err != nil {
		return nil, err
	}
	return []string{
		restoredDir,
		controlDirForHostPID(placeholderHostPID),
	}, nil
}

func restoredControlDir(placeholderHostPID, restoredPID int) (string, error) {
	if placeholderHostPID <= 0 {
		return "", fmt.Errorf("invalid placeholder host PID %d for control sentinel", placeholderHostPID)
	}
	if restoredPID <= 0 {
		return "", fmt.Errorf("invalid restored PID %d for control sentinel", restoredPID)
	}
	return filepath.Join(
		HostProcPath,
		strconv.Itoa(placeholderHostPID),
		"root",
		"proc",
		strconv.Itoa(restoredPID),
		"root",
		snapshotprotocol.SnapshotControlMountPath,
	), nil
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

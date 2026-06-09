// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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
	dir := filepath.Join(HostProcPath, strconv.Itoa(hostPID), "root", snapshotprotocol.SnapshotControlMountPath)
	return writeSentinelInDir(dir, name)
}

type RendezvousConfig struct {
	RestoreID string          `json:"restore_id,omitempty"`
	Store     RendezvousStore `json:"store"`
}

type RendezvousStore struct {
	Host       string `json:"host"`
	Port       int    `json:"port"`
	MasterRank int    `json:"master_rank"`
}

func WriteRendezvousFile(hostPID int, config RendezvousConfig) error {
	if hostPID <= 0 {
		return fmt.Errorf("invalid host PID %d for rendezvous file", hostPID)
	}
	if config.Store.Host == "" {
		return fmt.Errorf("rendezvous store host is empty")
	}
	if config.Store.Port <= 0 {
		return fmt.Errorf("invalid rendezvous store port %d", config.Store.Port)
	}
	dir := filepath.Join(HostProcPath, strconv.Itoa(hostPID), "root", snapshotprotocol.SnapshotControlMountPath)
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal rendezvous file: %w", err)
	}
	return writeFileInDir(dir, snapshotprotocol.TorchC10dRendezvousFile, append(data, '\n'))
}

func WriteNCCLCheckpointKVSFile(hostPID int, endpoint string) error {
	if hostPID <= 0 {
		return fmt.Errorf("invalid host PID %d for NCCL checkpoint KVS file", hostPID)
	}
	root := filepath.Join(HostProcPath, strconv.Itoa(hostPID), "root")
	return writeNCCLCheckpointKVSFileInRoot(root, endpoint)
}

func writeNCCLCheckpointKVSFileInRoot(root string, endpoint string) error {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return fmt.Errorf("NCCL checkpoint KVS endpoint is empty")
	}
	if strings.ContainsAny(endpoint, "\r\n") {
		return fmt.Errorf("NCCL checkpoint KVS endpoint must fit on one line")
	}
	controlDir := filepath.Join(root, snapshotprotocol.SnapshotControlMountPath)
	return writeFileInDir(
		controlDir,
		snapshotprotocol.NCCLCheckpointKVSFile,
		[]byte(endpoint+"\n"),
	)
}

func writeSentinelInDir(dir, name string) error {
	return writeFileInDir(dir, name, []byte("done\n"))
}

func writeFileInDir(dir, name string, data []byte) error {
	tmpPath := filepath.Join(dir, "."+name+".tmp")
	finalPath := filepath.Join(dir, name)
	if err := os.WriteFile(tmpPath, data, 0o644); err != nil {
		return fmt.Errorf("write temp file %s: %w", tmpPath, err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename file %s -> %s: %w", tmpPath, finalPath, err)
	}
	return nil
}

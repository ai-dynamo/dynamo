// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func TestWriteSentinelInDir_CreatesFileAtomically(t *testing.T) {
	dir := t.TempDir()

	if err := writeSentinelInDir(dir, "snapshot-complete"); err != nil {
		t.Fatalf("writeSentinelInDir failed: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "snapshot-complete"))
	if err != nil {
		t.Fatalf("sentinel not found: %v", err)
	}
	if string(data) != "done\n" {
		t.Errorf("unexpected sentinel contents: %q", data)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("failed to read dir: %v", err)
	}
	for _, e := range entries {
		if e.Name() != "snapshot-complete" {
			t.Errorf("unexpected leftover file %q in control dir", e.Name())
		}
	}
}

func TestWriteSentinelInDir_Overwrites(t *testing.T) {
	dir := t.TempDir()
	if err := writeSentinelInDir(dir, "restore-complete"); err != nil {
		t.Fatalf("first write failed: %v", err)
	}
	if err := writeSentinelInDir(dir, "restore-complete"); err != nil {
		t.Fatalf("second write failed: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "restore-complete"))
	if err != nil {
		t.Fatalf("sentinel not found: %v", err)
	}
	if string(data) != "done\n" {
		t.Errorf("unexpected sentinel contents: %q", data)
	}
}

func TestWriteSentinelInDir_DirMissing(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	if err := writeSentinelInDir(missing, "snapshot-complete"); err == nil {
		t.Fatal("expected error writing into missing directory")
	}
}

func TestWriteControlSentinel_RejectsInvalidPID(t *testing.T) {
	if err := WriteControlSentinel(0, "snapshot-complete"); err == nil {
		t.Fatal("expected error for PID 0")
	}
	if err := WriteControlSentinel(-1, "snapshot-complete"); err == nil {
		t.Fatal("expected error for negative PID")
	}
}

func TestWriteNCCLCheckpointKVSFileInRoot(t *testing.T) {
	root := t.TempDir()
	controlDir := filepath.Join(root, "snapshot-control")
	if err := os.MkdirAll(controlDir, 0o755); err != nil {
		t.Fatalf("failed to create control dir: %v", err)
	}

	endpoint := "leader-pod.default.svc:46379/checkpoint-1_main"
	if err := writeNCCLCheckpointKVSFileInRoot(root, endpoint); err != nil {
		t.Fatalf("writeNCCLCheckpointKVSFileInRoot failed: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(controlDir, snapshotprotocol.NCCLCheckpointKVSFile))
	if err != nil {
		t.Fatalf("expected NCCL KVS file: %v", err)
	}
	if string(data) != endpoint+"\n" {
		t.Fatalf("unexpected NCCL KVS file contents: %q", data)
	}
}

func TestWriteNCCLCheckpointKVSFileInRootRejectsInvalidEndpoint(t *testing.T) {
	if err := writeNCCLCheckpointKVSFileInRoot(t.TempDir(), ""); err == nil {
		t.Fatal("expected empty endpoint to be rejected")
	}
	if err := writeNCCLCheckpointKVSFileInRoot(t.TempDir(), "leader:46379/prefix\nother"); err == nil {
		t.Fatal("expected multiline endpoint to be rejected")
	}
}

func TestWriteC10DRendezvousFileInRoot(t *testing.T) {
	root := t.TempDir()
	controlDir := filepath.Join(root, "snapshot-control")
	if err := os.MkdirAll(controlDir, 0o755); err != nil {
		t.Fatalf("failed to create control dir: %v", err)
	}

	if err := writeC10DRendezvousFileInRoot(root, "leader.default.svc", 29500, "restore-1"); err != nil {
		t.Fatalf("writeC10DRendezvousFileInRoot failed: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(controlDir, snapshotprotocol.C10DRendezvousFile))
	if err != nil {
		t.Fatalf("expected c10d rendezvous file: %v", err)
	}
	var parsed struct {
		RestoreID string `json:"restore_id"`
		Store     struct {
			Host       string `json:"host"`
			Port       int    `json:"port"`
			MasterRank int    `json:"master_rank"`
		} `json:"store"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("invalid c10d rendezvous JSON: %v", err)
	}
	if parsed.RestoreID != "restore-1" || parsed.Store.Host != "leader.default.svc" || parsed.Store.Port != 29500 || parsed.Store.MasterRank != 0 {
		t.Fatalf("unexpected c10d rendezvous payload: %#v", parsed)
	}
}

func TestWriteC10DRendezvousFileInRootRejectsInvalidEndpoint(t *testing.T) {
	if err := writeC10DRendezvousFileInRoot(t.TempDir(), "", 29500, "restore-1"); err == nil {
		t.Fatal("expected empty host to be rejected")
	}
	if err := writeC10DRendezvousFileInRoot(t.TempDir(), "leader", 0, "restore-1"); err == nil {
		t.Fatal("expected invalid port to be rejected")
	}
}

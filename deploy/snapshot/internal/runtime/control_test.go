// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

func TestReadGPUUUIDOrderFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu-uuids")
	want := []byte("GPU-aaaaaaaa-1111-2222-3333-444444444444\n")
	if err := os.WriteFile(path, want, 0o644); err != nil {
		t.Fatalf("write GPU UUID order: %v", err)
	}

	got, err := readGPUUUIDOrderFile(path)
	if err != nil {
		t.Fatalf("readGPUUUIDOrderFile: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestReadGPUUUIDOrderFileRejectsUnsafeEntries(t *testing.T) {
	t.Run("symlink", func(t *testing.T) {
		dir := t.TempDir()
		target := filepath.Join(dir, "target")
		if err := os.WriteFile(target, []byte("secret"), 0o644); err != nil {
			t.Fatalf("write symlink target: %v", err)
		}
		path := filepath.Join(dir, "gpu-uuids")
		if err := os.Symlink(target, path); err != nil {
			t.Fatalf("create symlink: %v", err)
		}
		if _, err := readGPUUUIDOrderFile(path); err == nil {
			t.Fatal("expected symlink to be rejected")
		}
	})

	t.Run("fifo", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "gpu-uuids")
		if err := unix.Mkfifo(path, 0o644); err != nil {
			t.Fatalf("create FIFO: %v", err)
		}
		if _, err := readGPUUUIDOrderFile(path); err == nil {
			t.Fatal("expected FIFO to be rejected")
		}
	})

	t.Run("oversized", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "gpu-uuids")
		data := bytes.Repeat([]byte("x"), maxGPUUUIDOrderBytes+1)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatalf("write oversized file: %v", err)
		}
		if _, err := readGPUUUIDOrderFile(path); err == nil {
			t.Fatal("expected oversized file to be rejected")
		}
	})
}

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
	if _, err := ReadGPUUUIDOrderFile(0); err == nil {
		t.Fatal("expected GPU UUID order read to reject PID 0")
	}
}

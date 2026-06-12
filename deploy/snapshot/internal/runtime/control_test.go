// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"os"
	"path/filepath"
	"testing"
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

func TestRestoredControlDirUsesRestoredProcessRoot(t *testing.T) {
	got, err := restoredControlDir(1234, 106)
	if err != nil {
		t.Fatalf("restoredControlDir: %v", err)
	}

	want := filepath.Join(
		HostProcPath,
		"1234",
		"root",
		"proc",
		"106",
		"root",
		"/snapshot-control",
	)
	if got != want {
		t.Fatalf("restoredControlDir() = %q, want %q", got, want)
	}
	if got == controlDirForHostPID(1234) {
		t.Fatalf("restoredControlDir() unexpectedly matched placeholder control dir %q", got)
	}
}

func TestRestoreControlDirsIncludesRestoredAndPlaceholderViews(t *testing.T) {
	dirs, err := restoreControlDirs(1234, 106)
	if err != nil {
		t.Fatalf("restoreControlDirs: %v", err)
	}
	if len(dirs) != 2 {
		t.Fatalf("len(restoreControlDirs()) = %d, want 2: %v", len(dirs), dirs)
	}

	restoredDir, err := restoredControlDir(1234, 106)
	if err != nil {
		t.Fatalf("restoredControlDir: %v", err)
	}
	placeholderDir := controlDirForHostPID(1234)
	if dirs[0] != restoredDir {
		t.Fatalf("first restore control dir = %q, want restored dir %q", dirs[0], restoredDir)
	}
	if dirs[1] != placeholderDir {
		t.Fatalf("second restore control dir = %q, want placeholder dir %q", dirs[1], placeholderDir)
	}
	if dirs[0] == dirs[1] {
		t.Fatalf("restore control dirs should be distinct, got %v", dirs)
	}
}

func TestWriteRestoredControlSentinel_RejectsInvalidPIDs(t *testing.T) {
	if err := WriteRestoredControlSentinel(0, 106, "restore-complete"); err == nil {
		t.Fatal("expected error for invalid placeholder PID")
	}
	if err := WriteRestoredControlSentinel(1234, 0, "restore-complete"); err == nil {
		t.Fatal("expected error for invalid restored PID")
	}
}

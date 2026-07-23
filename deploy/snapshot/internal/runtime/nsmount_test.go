// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package runtime

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

func TestCheckpointMountAttrs(t *testing.T) {
	want := uint64(unix.MOUNT_ATTR_RDONLY |
		unix.MOUNT_ATTR_NOSUID |
		unix.MOUNT_ATTR_NODEV |
		unix.MOUNT_ATTR_NOEXEC)
	if checkpointMountAttrs != want {
		t.Fatalf("checkpointMountAttrs = %#x, want %#x", checkpointMountAttrs, want)
	}
}

func TestOpenOrCreateMountTargetCreatesCleanDirectory(t *testing.T) {
	target := filepath.Join(t.TempDir(), "checkpoint", "id", "versions", "1")

	fd, err := openOrCreateMountTarget(target)
	if err != nil {
		t.Fatalf("openOrCreateMountTarget(%q): %v", target, err)
	}
	defer unix.Close(fd)

	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		t.Fatalf("fstat target fd: %v", err)
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFDIR {
		t.Fatalf("target fd mode = %#o, want directory", stat.Mode)
	}
	if info, err := os.Stat(target); err != nil {
		t.Fatalf("stat created target: %v", err)
	} else if !info.IsDir() {
		t.Fatalf("created target mode = %v, want directory", info.Mode())
	}
}

func TestOpenOrCreateMountTargetRejectsSymlinkComponent(t *testing.T) {
	root := t.TempDir()
	redirect := t.TempDir()
	link := filepath.Join(root, "checkpoint")
	if err := os.Symlink(redirect, link); err != nil {
		t.Fatalf("create target symlink: %v", err)
	}

	target := filepath.Join(link, "id", "versions", "1")
	fd, err := openOrCreateMountTarget(target)
	if fd >= 0 {
		unix.Close(fd)
		t.Fatalf("openOrCreateMountTarget(%q) unexpectedly returned fd %d", target, fd)
	}
	if err == nil {
		t.Fatalf("openOrCreateMountTarget(%q) unexpectedly succeeded", target)
	}
	if _, statErr := os.Stat(filepath.Join(redirect, "id")); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("symlink target was modified, stat error = %v", statErr)
	}
}

func TestOpenOrCreateMountTargetRejectsUnsafePaths(t *testing.T) {
	for _, target := range []string{
		"relative/path",
		"/",
		"/checkpoint/../escape",
		"/checkpoint//double",
	} {
		t.Run(target, func(t *testing.T) {
			fd, err := openOrCreateMountTarget(target)
			if fd >= 0 {
				unix.Close(fd)
				t.Fatalf("openOrCreateMountTarget(%q) unexpectedly returned fd %d", target, fd)
			}
			if err == nil {
				t.Fatalf("openOrCreateMountTarget(%q) unexpectedly succeeded", target)
			}
		})
	}
}

func TestOpenCheckpointTreeRejectsUnsafePathsBeforeSyscall(t *testing.T) {
	for _, path := range []string{
		"relative/path",
		"/tmp/../tmp/checkpoint",
		"/",
	} {
		t.Run(path, func(t *testing.T) {
			if _, err := OpenCheckpointTree(path); err == nil {
				t.Fatalf("OpenCheckpointTree(%q) succeeded, want error", path)
			}
		})
	}
}

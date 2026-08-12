// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package nsmount

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-logr/logr"
)

// writeFakeBinary writes a shell script at a temp path and returns the path.
// The script is passed to sh -c, so $@ refers to the arguments passed by execMounter.
func writeFakeBinary(t *testing.T, script string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "ns-bind-mount")
	content := "#!/bin/sh\n" + script + "\n"
	if err := os.WriteFile(p, []byte(content), 0755); err != nil {
		t.Fatal(err)
	}
	return p
}

func newMounterForTest(t *testing.T, bin string) *execMounter {
	t.Helper()
	m, err := newExecMounter(bin, logr.Discard())
	if err != nil {
		t.Fatalf("newExecMounter: %v", err)
	}
	return m
}

// readLines reads a file and splits it into non-empty lines.
func readLines(t *testing.T, path string) []string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("readLines: %v", err)
	}
	var lines []string
	for _, l := range strings.Split(string(data), "\n") {
		if l != "" {
			lines = append(lines, l)
		}
	}
	return lines
}

func TestExecMounter_Mount_Args(t *testing.T) {
	logFile := filepath.Join(t.TempDir(), "args.log")
	// Print each argument on its own line so we can parse them individually.
	// The Go caller now uses "mount-fd <nsFdChildNum> <src> <dst>" so $@ logs
	// those four tokens.
	bin := writeFakeBinary(t, `printf '%s\n' "$@" >> `+logFile)

	m := newMounterForTest(t, bin)
	pid := os.Getpid()
	_, err := m.Mount(context.Background(), pid, "/src", "/dst", MountOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := readLines(t, logFile)
	want := []string{"mount-fd", fmt.Sprintf("%d", nsFdChildNum), "/src", "/dst"}
	if len(got) != len(want) {
		t.Fatalf("args: got %v want %v", got, want)
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("arg[%d]: got %q want %q", i, got[i], w)
		}
	}
}

func TestExecMounter_Mount_ReadOnly(t *testing.T) {
	logFile := filepath.Join(t.TempDir(), "args.log")
	bin := writeFakeBinary(t, `printf '%s\n' "$@" >> `+logFile)

	m := newMounterForTest(t, bin)
	pid := os.Getpid()
	_, err := m.Mount(context.Background(), pid, "/src", "/dst", MountOptions{ReadOnly: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := readLines(t, logFile)
	want := []string{"mount-fd", fmt.Sprintf("%d", nsFdChildNum), "/src", "/dst", "ro"}
	if len(got) != len(want) {
		t.Fatalf("args: got %v want %v", got, want)
	}
	if got[4] != "ro" {
		t.Errorf("expected 'ro' flag at index 4, got %q", got[4])
	}
}

func TestExecMounter_Mount_ErrorWrapped(t *testing.T) {
	bin := writeFakeBinary(t, `echo "subprocess boom" >&2; exit 1`)
	m := newMounterForTest(t, bin)

	// Use current pid so the ns fd open succeeds; the helper itself then fails.
	_, err := m.Mount(context.Background(), os.Getpid(), "/src", "/dst", MountOptions{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	s := err.Error()
	for _, want := range []string{"/src", "/dst", "subprocess boom"} {
		if !strings.Contains(s, want) {
			t.Errorf("error missing %q: %s", want, s)
		}
	}
}

func TestExecMounter_Mount_NsFdOpenFailure(t *testing.T) {
	// The ns fd is now opened BEFORE the helper runs. A dead PID (MaxInt32)
	// causes os.Open("/proc/<pid>/ns/mnt") to fail before the binary is called.
	bin := writeFakeBinary(t, `exit 0`)
	m := newMounterForTest(t, bin)

	_, err := m.Mount(context.Background(), math.MaxInt32, "/src", "/dst", MountOptions{})
	if err == nil {
		t.Fatal("expected error when ns fd cannot be opened, got nil")
	}
	if !strings.Contains(err.Error(), "ns/mnt") {
		t.Errorf("error should mention ns/mnt path, got: %v", err)
	}
}

func TestExecMounter_Unmount_Error(t *testing.T) {
	bin := writeFakeBinary(t, `if [ "$1" = "umount-fd" ]; then echo "boom" >&2; exit 1; fi`)
	m := newMounterForTest(t, bin)
	handle, err := m.Mount(context.Background(), os.Getpid(), "/src", "/dst", MountOptions{})
	if err != nil {
		t.Fatalf("Mount: %v", err)
	}

	err = handle.Unmount(context.Background(), false)
	if err == nil {
		t.Fatal("expected error from umount-fd, got nil")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Errorf("error should contain subprocess output, got: %v", err)
	}

	// Second call must return the same stored error without invoking the binary again.
	err2 := handle.Unmount(context.Background(), false)
	if err2 != err {
		t.Errorf("second Unmount returned different error: %v", err2)
	}
}

func TestExecMounter_Unmount_Idempotent(t *testing.T) {
	callLog := filepath.Join(t.TempDir(), "calls.log")
	// Record the first argument of each invocation (subcommand name).
	bin := writeFakeBinary(t, `printf '%s\n' "$1" >> `+callLog)

	m := newMounterForTest(t, bin)
	handle, err := m.Mount(context.Background(), os.Getpid(), "/src", "/dst", MountOptions{})
	if err != nil {
		t.Fatalf("Mount: %v", err)
	}

	if err := handle.Unmount(context.Background(), false); err != nil {
		t.Fatalf("first Unmount: %v", err)
	}
	if err := handle.Unmount(context.Background(), false); err != nil {
		t.Fatalf("second Unmount: %v", err)
	}

	calls := readLines(t, callLog)
	umountFdCalls := 0
	for _, c := range calls {
		if c == "umount-fd" {
			umountFdCalls++
		}
	}
	if umountFdCalls != 1 {
		t.Errorf("expected exactly 1 umount-fd call, got %d (all calls: %v)", umountFdCalls, calls)
	}
}

// TestExecMounter_Mount_AttributeTokens covers the whole attribute set reaching
// the helper. Attributes are inputs now rather than something the subcommand
// implies, so the order and completeness of the tokens is the contract.
func TestExecMounter_Mount_AttributeTokens(t *testing.T) {
	tests := []struct {
		name string
		opts MountOptions
		want []string
	}{
		{
			name: "no attributes at all",
			opts: MountOptions{},
			want: nil,
		},
		{
			name: "bundle policy leaves execution alone",
			opts: MountOptions{ReadOnly: true, NoSuid: true, NoDev: true},
			want: []string{"ro", "nosuid", "nodev"},
		},
		{
			name: "artifact policy adds noexec",
			opts: MountOptions{ReadOnly: true, NoSuid: true, NoDev: true, NoExec: true},
			want: []string{"ro", "nosuid", "nodev", "noexec"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logFile := filepath.Join(t.TempDir(), "args.log")
			bin := writeFakeBinary(t, `printf '%s\n' "$@" >> `+logFile)
			m := newMounterForTest(t, bin)

			if _, err := m.Mount(context.Background(), os.Getpid(), "/src", "/dst", tc.opts); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			got := readLines(t, logFile)
			fixed := []string{"mount-fd", fmt.Sprintf("%d", nsFdChildNum), "/src", "/dst"}
			if len(got) < len(fixed) {
				t.Fatalf("args: got %v, want at least %v", got, fixed)
			}
			gotAttrs := got[len(fixed):]
			if len(gotAttrs) != len(tc.want) {
				t.Fatalf("attrs: got %v, want %v", gotAttrs, tc.want)
			}
			for i := range tc.want {
				if gotAttrs[i] != tc.want[i] {
					t.Errorf("attr[%d]: got %q, want %q", i, gotAttrs[i], tc.want[i])
				}
			}
		})
	}
}

// TestExecMounter_Unmount_StrictToken covers strictness reaching the helper on
// the unmount side, both through a held ref and through the refless path
// startup recovery uses.
func TestExecMounter_Unmount_StrictToken(t *testing.T) {
	t.Run("held ref carries strict", func(t *testing.T) {
		logFile := filepath.Join(t.TempDir(), "args.log")
		bin := writeFakeBinary(t, `printf '%s\n' "$@" >> `+logFile)
		m := newMounterForTest(t, bin)

		ref, err := m.Mount(context.Background(), os.Getpid(), "/src", "/dst", MountOptions{})
		if err != nil {
			t.Fatalf("Mount: %v", err)
		}
		if err := ref.Unmount(context.Background(), true); err != nil {
			t.Fatalf("Unmount: %v", err)
		}

		got := readLines(t, logFile)
		want := []string{"umount-fd", fmt.Sprintf("%d", nsFdChildNum), "/dst", "strict"}
		if len(got) < len(want) {
			t.Fatalf("args: got %v", got)
		}
		unmountArgs := got[len(got)-len(want):]
		for i := range want {
			if unmountArgs[i] != want[i] {
				t.Fatalf("unmount args: got %v, want %v", unmountArgs, want)
			}
		}
	})

}

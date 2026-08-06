// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nsmount

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr"
)

// fakeMountHandle implements MountHandle for tests.
type fakeMountHandle struct {
	dst        string
	unmountLog *[]string
}

func (h *fakeMountHandle) TargetPath() string { return h.dst }

func (h *fakeMountHandle) Unmount(_ context.Context) error {
	*h.unmountLog = append(*h.unmountLog, h.dst)
	return nil
}

// mountCall records a single Mount invocation.
type mountCall struct {
	pid      int
	src, dst string
	opts     MountOptions
}

// mockMounter lets tests control per-call Mount results and record call order.
type mockMounter struct {
	// results[i] is returned for the i-th Mount call (in order).
	results    []error
	calls      []mountCall
	unmountLog []string
}

func (m *mockMounter) Mount(_ context.Context, pid int, src, dst string, opts MountOptions) (MountHandle, error) {
	i := len(m.calls)
	m.calls = append(m.calls, mountCall{pid: pid, src: src, dst: dst, opts: opts})
	if i < len(m.results) && m.results[i] != nil {
		return nil, m.results[i]
	}
	return &fakeMountHandle{dst: dst, unmountLog: &m.unmountLog}, nil
}

const testPID = 42

func newMounter(t *testing.T, m *mockMounter) *NSMounter {
	t.Helper()
	nsm, err := newWithMounter(agentBinDir, SnapshotBinDir, m, logr.Discard())
	if err != nil {
		t.Fatalf("newWithMounter: %v", err)
	}
	return nsm
}

func TestMount_MountsAgentBundle(t *testing.T) {
	m := &mockMounter{}
	_, err := newMounter(t, m).Mount(context.Background(), testPID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []mountCall{
		{pid: testPID, src: agentBinDir, dst: SnapshotBinDir, opts: MountOptions{ReadOnly: true}},
	}
	if len(m.calls) != len(want) {
		t.Fatalf("got %d mount calls, want %d", len(m.calls), len(want))
	}
	if m.calls[0] != want[0] {
		t.Errorf("call[0]: got %+v, want %+v", m.calls[0], want[0])
	}
}

func TestMount_BinPath(t *testing.T) {
	m := &mockMounter{}
	handle, err := newMounter(t, m).Mount(context.Background(), testPID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got, err := handle.BinPath("nsrestore")
	if err != nil {
		t.Fatalf("BinPath: unexpected error: %v", err)
	}
	want := filepath.Join(SnapshotBinDir, "nsrestore")
	if got != want {
		t.Errorf("BinPath: got %q, want %q", got, want)
	}
}

func TestMount_CleanupUnmounts(t *testing.T) {
	m := &mockMounter{}
	handle, err := newMounter(t, m).Mount(context.Background(), testPID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := handle.Cleanup(context.Background()); err != nil {
		t.Fatalf("unexpected cleanup error: %v", err)
	}

	if len(m.unmountLog) != 1 || m.unmountLog[0] != SnapshotBinDir {
		t.Errorf("expected unmount of %q, got %v", SnapshotBinDir, m.unmountLog)
	}
}

func TestMount_Fails(t *testing.T) {
	mountErr := errors.New("mount failed")
	m := &mockMounter{results: []error{mountErr}}

	_, err := newMounter(t, m).Mount(context.Background(), testPID)
	if !errors.Is(err, mountErr) {
		t.Fatalf("got %v, want %v", err, mountErr)
	}
	if len(m.unmountLog) != 0 {
		t.Errorf("expected no unmounts, got %v", m.unmountLog)
	}
}

func TestBinPath_RejectsInvalidNames(t *testing.T) {
	m := &mockMounter{}
	handle, err := newMounter(t, m).Mount(context.Background(), testPID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	invalid := []string{"", ".", "..", "foo/bar", "../../etc/passwd"}
	for _, name := range invalid {
		_, err := handle.BinPath(name)
		if err == nil {
			t.Errorf("BinPath(%q): expected error, got nil", name)
		}
	}
}

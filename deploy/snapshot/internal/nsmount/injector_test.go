// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nsmount

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr"
)

const (
	testSrc = "/snapshot-binaries"
	testDst = "/tmp/snapshot-binaries"
)

// fakemountRef implements mountRef for tests.
type fakemountRef struct {
	dst           string
	unmountLog    *[]string
	strictUnmount bool
}

func (h *fakemountRef) TargetPath() string { return h.dst }
func (h *fakemountRef) NsFd() *os.File     { return nil }

func (h *fakemountRef) Unmount(_ context.Context, strict bool) error {
	*h.unmountLog = append(*h.unmountLog, h.dst)
	h.strictUnmount = strict
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

func (m *mockMounter) Mount(_ context.Context, pid int, src, dst string, opts MountOptions) (mountRef, error) {
	i := len(m.calls)
	m.calls = append(m.calls, mountCall{pid: pid, src: src, dst: dst, opts: opts})
	if i < len(m.results) && m.results[i] != nil {
		return nil, m.results[i]
	}
	return &fakemountRef{dst: dst, unmountLog: &m.unmountLog}, nil
}

const testPID = 42

func newMounter(t *testing.T, m *mockMounter) *NSMounter {
	t.Helper()
	return newWithMounter(m, logr.Discard())
}

func TestMount_MountsAgentBundle(t *testing.T) {
	m := &mockMounter{}
	_, err := newMounter(t, m).ReadOnly().Mount(context.Background(), testPID, testSrc, testDst)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// nosuid and nodev used to be implied by the helper's `ro` token; they are
	// requested explicitly now that attributes are inputs.
	want := []mountCall{
		{pid: testPID, src: testSrc, dst: testDst,
			opts: MountOptions{ReadOnly: true, NoSuid: true, NoDev: true}},
	}
	if len(m.calls) != len(want) {
		t.Fatalf("got %d mount calls, want %d", len(m.calls), len(want))
	}
	if m.calls[0] != want[0] {
		t.Errorf("call[0]: got %+v, want %+v", m.calls[0], want[0])
	}
}

func TestMount_Path(t *testing.T) {
	m := &mockMounter{}
	mp, err := newMounter(t, m).ReadOnly().Mount(context.Background(), testPID, testSrc, testDst)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got, err := mp.Path("nsrestore")
	if err != nil {
		t.Fatalf("Path: unexpected error: %v", err)
	}
	want := filepath.Join(testDst, "nsrestore")
	if got != want {
		t.Errorf("Path: got %q, want %q", got, want)
	}
}

func TestMount_Unmounts(t *testing.T) {
	m := &mockMounter{}
	mp, err := newMounter(t, m).ReadOnly().Mount(context.Background(), testPID, testSrc, testDst)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := mp.Unmount(context.Background(), false); err != nil {
		t.Fatalf("unexpected unmount error: %v", err)
	}

	if len(m.unmountLog) != 1 || m.unmountLog[0] != testDst {
		t.Errorf("expected unmount of %q, got %v", testDst, m.unmountLog)
	}
}

func TestMount_Fails(t *testing.T) {
	mountErr := errors.New("mount failed")
	m := &mockMounter{results: []error{mountErr}}

	_, err := newMounter(t, m).ReadOnly().Mount(context.Background(), testPID, testSrc, testDst)
	if !errors.Is(err, mountErr) {
		t.Fatalf("got %v, want %v", err, mountErr)
	}
	if len(m.unmountLog) != 0 {
		t.Errorf("expected no unmounts, got %v", m.unmountLog)
	}
}

func TestPath_RejectsInvalidNames(t *testing.T) {
	m := &mockMounter{}
	mp, err := newMounter(t, m).ReadOnly().Mount(context.Background(), testPID, testSrc, testDst)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	invalid := []string{"", ".", "..", "foo/bar", "../../etc/passwd"}
	for _, name := range invalid {
		_, err := mp.Path(name)
		if err == nil {
			t.Errorf("Path(%q): expected error, got nil", name)
		}
	}
}

// TestWrappers_CarryTheirPolicy pins what each wrapper binds. The wrappers are
// the only place mount attributes are chosen, so these two cases are the whole
// policy this package applies.
func TestWrappers_CarryTheirPolicy(t *testing.T) {
	tests := []struct {
		name     string
		mount    func(nsm *NSMounter) (MountPoint, error)
		wantSrc  string
		wantDst  string
		wantOpts MountOptions
	}{
		{
			// nsrestore is executed out of this mount, so noexec must not be
			// requested.
			name: "read-only leaves execution alone",
			mount: func(nsm *NSMounter) (MountPoint, error) {
				return nsm.ReadOnly().Mount(context.Background(), testPID, testSrc, testDst)
			},
			wantSrc:  testSrc,
			wantDst:  testDst,
			wantOpts: MountOptions{ReadOnly: true, NoSuid: true, NoDev: true},
		},
		{
			// The artifact is data from shared storage: executing it would turn
			// the mount into a code-injection channel.
			name: "read-only noexec forbids execution",
			mount: func(nsm *NSMounter) (MountPoint, error) {
				return nsm.ReadOnlyNoExec().Mount(context.Background(), testPID, "/checkpoints/abc/versions/1", CheckpointDst)
			},
			wantSrc:  "/checkpoints/abc/versions/1",
			wantDst:  CheckpointDst,
			wantOpts: MountOptions{ReadOnly: true, NoSuid: true, NoDev: true, NoExec: true},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &mockMounter{}
			mp, err := tc.mount(newMounter(t, m))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(m.calls) != 1 {
				t.Fatalf("got %d mount calls, want 1", len(m.calls))
			}
			want := mountCall{pid: testPID, src: tc.wantSrc, dst: tc.wantDst, opts: tc.wantOpts}
			if m.calls[0] != want {
				t.Errorf("got %+v, want %+v", m.calls[0], want)
			}
			if _, err := mp.Path("anything"); err != nil {
				t.Errorf("Path: %v", err)
			}
		})
	}
}

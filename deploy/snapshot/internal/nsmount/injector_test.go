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

const (
	testSrc = "/snapshot-binaries"
	testDst = "/tmp/snapshot-binaries"
)

// fakemountRef implements mountRef for tests.
type fakemountRef struct {
	dst        string
	unmountLog *[]string
}

func (h *fakemountRef) TargetPath() string { return h.dst }

func (h *fakemountRef) Unmount(_ context.Context) error {
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
	nsm, err := newWithMounter(testSrc, testDst, m, logr.Discard())
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
		{pid: testPID, src: testSrc, dst: testDst, opts: MountOptions{ReadOnly: true}},
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
	mp, err := newMounter(t, m).Mount(context.Background(), testPID)
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
	mp, err := newMounter(t, m).Mount(context.Background(), testPID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := mp.Unmount(context.Background()); err != nil {
		t.Fatalf("unexpected unmount error: %v", err)
	}

	if len(m.unmountLog) != 1 || m.unmountLog[0] != testDst {
		t.Errorf("expected unmount of %q, got %v", testDst, m.unmountLog)
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

func TestPath_RejectsInvalidNames(t *testing.T) {
	m := &mockMounter{}
	mp, err := newMounter(t, m).Mount(context.Background(), testPID)
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

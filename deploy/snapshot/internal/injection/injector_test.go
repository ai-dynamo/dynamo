// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package injection

import (
	"context"
	"errors"
	"testing"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/nsbindmount"
)

// mountCall records a single Mount invocation.
type mountCall struct {
	pid      int
	src, dst string
	opts     nsbindmount.MountOptions
}

// mockMounter lets tests control per-call Mount results and record call order.
type mockMounter struct {
	// results[i] is returned for the i-th Mount call (in order).
	results []error
	calls   []mountCall
	// unmountOrder records which dst was unmounted and in what order.
	unmountOrder []string
}

func (m *mockMounter) Mount(_ context.Context, pid int, src, dst string, opts nsbindmount.MountOptions) (func() error, error) {
	i := len(m.calls)
	m.calls = append(m.calls, mountCall{pid: pid, src: src, dst: dst, opts: opts})
	if i < len(m.results) && m.results[i] != nil {
		return nil, m.results[i]
	}
	return func() error {
		m.unmountOrder = append(m.unmountOrder, dst)
		return nil
	}, nil
}

const testPID = 42

func newInjector(m *mockMounter) *Injector {
	return New(m, logr.Discard())
}

func TestInject_MountsAgentBundle(t *testing.T) {
	m := &mockMounter{}
	_, err := newInjector(m).Inject(context.Background(), testPID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []mountCall{
		{pid: testPID, src: agentBinDir, dst: SnapshotBinDir, opts: nsbindmount.MountOptions{ReadOnly: true}},
	}
	if len(m.calls) != len(want) {
		t.Fatalf("got %d mount calls, want %d", len(m.calls), len(want))
	}
	if m.calls[0] != want[0] {
		t.Errorf("call[0]: got %+v, want %+v", m.calls[0], want[0])
	}
}

func TestInject_CleanupUnmounts(t *testing.T) {
	m := &mockMounter{}
	cleanup, err := newInjector(m).Inject(context.Background(), testPID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := cleanup(); err != nil {
		t.Fatalf("unexpected cleanup error: %v", err)
	}

	if len(m.unmountOrder) != 1 || m.unmountOrder[0] != SnapshotBinDir {
		t.Errorf("expected unmount of %q, got %v", SnapshotBinDir, m.unmountOrder)
	}
}

func TestInject_MountFails(t *testing.T) {
	mountErr := errors.New("mount failed")
	m := &mockMounter{results: []error{mountErr}}

	_, err := newInjector(m).Inject(context.Background(), testPID)
	if !errors.Is(err, mountErr) {
		t.Fatalf("got %v, want %v", err, mountErr)
	}
	if len(m.unmountOrder) != 0 {
		t.Errorf("expected no unmounts, got %v", m.unmountOrder)
	}
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clock

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var t0 = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

func TestReal(t *testing.T) {
	c := Real()
	assert.WithinDuration(t, time.Now(), c.Now(), time.Second)
	tk := c.NewTicker(time.Millisecond)
	<-tk.C()
	tk.Stop()
	tm := c.NewTimer(time.Millisecond)
	<-tm.C()
	assert.False(t, tm.Stop())
}

func TestFakeTicker(t *testing.T) {
	f := NewFake(t0)
	tk := f.NewTicker(time.Second)
	assert.Equal(t, 1, f.Waiters())

	f.Advance(500 * time.Millisecond)
	select {
	case <-tk.C():
		t.Fatal("ticker fired early")
	default:
	}
	f.Advance(500 * time.Millisecond)
	assert.Equal(t, t0.Add(time.Second), <-tk.C())

	// Missed ticks are dropped: one pending tick after a long advance.
	f.Advance(5 * time.Second)
	<-tk.C()
	select {
	case <-tk.C():
		t.Fatal("ticks were not dropped")
	default:
	}
	tk.Stop()
	assert.Equal(t, 0, f.Waiters())
}

func TestFakeTimer(t *testing.T) {
	f := NewFake(t0)
	tm := f.NewTimer(time.Second)
	f.Set(t0.Add(2 * time.Second))
	assert.Equal(t, t0.Add(2*time.Second), <-tm.C())
	assert.False(t, tm.Stop(), "a fired timer is inactive")
	assert.Equal(t, t0.Add(2*time.Second), f.Now())

	tm = f.NewTimer(time.Second)
	assert.True(t, tm.Stop())

	tm = f.NewTimer(0)
	<-tm.C()
	assert.Panics(t, func() { f.NewTicker(0) })
}

func TestFakeBlockUntil(t *testing.T) {
	f := NewFake(t0)
	done := make(chan struct{})
	go func() {
		f.BlockUntil(2)
		close(done)
	}()
	f.NewTicker(time.Second)
	f.NewTimer(time.Second)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		require.FailNow(t, "BlockUntil did not return")
	}
}

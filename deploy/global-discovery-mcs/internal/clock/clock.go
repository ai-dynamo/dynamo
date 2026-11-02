// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package clock abstracts time so that leases, refresh loops, bookmarks and
// compaction can be tested with a fake clock.
package clock

import (
	"sort"
	"sync"
	"time"
)

// Clock tells time and creates tickers and timers.
type Clock interface {
	Now() time.Time
	// NewTicker returns a ticker that fires every d. It drops ticks for a
	// slow receiver, like time.Ticker.
	NewTicker(d time.Duration) Ticker
	// NewTimer returns a timer that fires once after d.
	NewTimer(d time.Duration) Timer
}

// Ticker is a time.Ticker.
type Ticker interface {
	C() <-chan time.Time
	Stop()
}

// Timer is a time.Timer.
type Timer interface {
	C() <-chan time.Time
	Stop() bool
}

// Real returns the system clock.
func Real() Clock { return realClock{} }

type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

func (realClock) NewTicker(d time.Duration) Ticker { return realTicker{time.NewTicker(d)} }

func (realClock) NewTimer(d time.Duration) Timer { return realTimer{time.NewTimer(d)} }

type realTicker struct{ t *time.Ticker }

func (t realTicker) C() <-chan time.Time { return t.t.C }
func (t realTicker) Stop()               { t.t.Stop() }

type realTimer struct{ t *time.Timer }

func (t realTimer) C() <-chan time.Time { return t.t.C }
func (t realTimer) Stop() bool          { return t.t.Stop() }

// Fake is a manually advanced Clock for tests. Tickers and timers fire when
// Advance or Set moves the time past their deadline.
type Fake struct {
	mu      sync.Mutex
	now     time.Time
	waiters []*fakeWaiter
	changed chan struct{}
}

type fakeWaiter struct {
	clock  *Fake
	c      chan time.Time
	next   time.Time
	period time.Duration // 0 for a timer
	active bool
}

// NewFake returns a fake clock set to now.
func NewFake(now time.Time) *Fake {
	return &Fake{now: now, changed: make(chan struct{})}
}

// Now implements Clock.
func (f *Fake) Now() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.now
}

// NewTicker implements Clock.
func (f *Fake) NewTicker(d time.Duration) Ticker {
	if d <= 0 {
		panic("clock: non-positive ticker interval")
	}
	return fakeTicker{f.add(d, d)}
}

// NewTimer implements Clock.
func (f *Fake) NewTimer(d time.Duration) Timer {
	return f.add(d, 0)
}

func (f *Fake) add(d, period time.Duration) *fakeWaiter {
	f.mu.Lock()
	defer f.mu.Unlock()
	w := &fakeWaiter{clock: f, c: make(chan time.Time, 1), next: f.now.Add(d), period: period, active: true}
	f.waiters = append(f.waiters, w)
	f.notifyLocked()
	if period == 0 && d <= 0 {
		f.fireLocked()
	}
	return w
}

// Advance moves the clock forward by d and fires due tickers and timers.
func (f *Fake) Advance(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.now = f.now.Add(d)
	f.fireLocked()
}

// Set moves the clock to t, which may be in the past, and fires due tickers
// and timers.
func (f *Fake) Set(t time.Time) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.now = t
	f.fireLocked()
}

// Waiters returns the number of active tickers and timers.
func (f *Fake) Waiters() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.waiters)
}

// BlockUntil waits until at least n tickers and timers are active, so that a
// test can advance the clock after the code under test created them.
func (f *Fake) BlockUntil(n int) {
	for {
		f.mu.Lock()
		if len(f.waiters) >= n {
			f.mu.Unlock()
			return
		}
		ch := f.changed
		f.mu.Unlock()
		<-ch
	}
}

func (f *Fake) notifyLocked() {
	close(f.changed)
	f.changed = make(chan struct{})
}

func (f *Fake) fireLocked() {
	sort.SliceStable(f.waiters, func(i, j int) bool { return f.waiters[i].next.Before(f.waiters[j].next) })
	kept := f.waiters[:0]
	for _, w := range f.waiters {
		if !w.next.After(f.now) {
			select {
			case w.c <- f.now:
			default:
			}
			if w.period == 0 {
				w.active = false
				continue
			}
			for !w.next.After(f.now) {
				w.next = w.next.Add(w.period)
			}
		}
		kept = append(kept, w)
	}
	for i := len(kept); i < len(f.waiters); i++ {
		f.waiters[i] = nil
	}
	f.waiters = kept
	f.notifyLocked()
}

func (f *Fake) remove(w *fakeWaiter) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	wasActive := w.active
	w.active = false
	for i, x := range f.waiters {
		if x == w {
			f.waiters = append(f.waiters[:i], f.waiters[i+1:]...)
			f.notifyLocked()
			break
		}
	}
	return wasActive
}

func (w *fakeWaiter) C() <-chan time.Time { return w.c }

// Stop implements Timer.
func (w *fakeWaiter) Stop() bool { return w.clock.remove(w) }

type fakeTicker struct{ w *fakeWaiter }

func (t fakeTicker) C() <-chan time.Time { return t.w.c }
func (t fakeTicker) Stop()               { t.w.Stop() }

/*
Copyright 2026 NVIDIA Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package disagg

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
)

func newTestBookingExecutor(t *testing.T, mutate func(*bookingExecutorConfig)) *bookingExecutor {
	t.Helper()
	cfg := bookingExecutorConfig{
		markerWorkers:     1,
		cleanupWorkers:    1,
		markerQueueSize:   1,
		cleanupQueueSize:  1,
		reconcileInterval: 5 * time.Millisecond,
		cleanupRetryDelay: 10 * time.Millisecond,
		cleanupRetention:  time.Second,
		cleanupBackoff:    time.Millisecond,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	executor := newBookingExecutor(cfg)
	t.Cleanup(executor.stop)
	return executor
}

func registerTestLifecycle(t *testing.T, executor *bookingExecutor, freeBooking func(string) error) *bookingLifecycle {
	t.Helper()
	bookingID := uuid.NewString()
	lifecycle := registerBookingLifecycleWithExecutor(bookingID, freeBooking, executor)
	t.Cleanup(func() { bookingLifecycles.Delete(bookingID) })
	return lifecycle
}

func waitForLifecycleRemoval(t *testing.T, bookingID string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for findBookingLifecycle(bookingID) != nil {
		if time.Now().After(deadline) {
			t.Fatalf("booking lifecycle %q was not removed", bookingID)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestBookingCleanupRetentionFollowsRouterExpiryOverride(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want time.Duration
	}{
		{name: "unset", want: minimumBookingCleanupRetention},
		{name: "invalid", raw: "invalid", want: minimumBookingCleanupRetention},
		{name: "zero", raw: "0", want: minimumBookingCleanupRetention},
		{name: "shorter than minimum", raw: "60", want: minimumBookingCleanupRetention},
		{name: "long router expiry", raw: "3600", want: 62 * time.Minute},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := bookingCleanupRetentionFromLookup(func(string) string { return test.raw })
			if got != test.want {
				t.Fatalf("cleanup retention = %s, want %s", got, test.want)
			}
		})
	}
}

func TestBookingExecutorBoundsPrefillMarkerQueue(t *testing.T) {
	executor := newTestBookingExecutor(t, nil)
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	var calls atomic.Int32
	marker := func(string) error {
		call := calls.Add(1)
		if call == 1 {
			close(firstStarted)
			select {
			case <-releaseFirst:
			case <-executor.stopCh:
			}
		}
		return nil
	}

	first := registerTestLifecycle(t, executor, func(string) error { return nil })
	second := registerTestLifecycle(t, executor, func(string) error { return nil })
	overflow := registerTestLifecycle(t, executor, func(string) error { return nil })

	first.startPrefillMarker(marker, logr.Discard(), "first")
	select {
	case <-firstStarted:
	case <-time.After(time.Second):
		t.Fatal("first marker did not start")
	}
	second.startPrefillMarker(marker, logr.Discard(), "second")
	if got := len(executor.markerQueue); got != 1 {
		t.Fatalf("marker queue length = %d, want 1", got)
	}
	overflow.startPrefillMarker(marker, logr.Discard(), "overflow")
	select {
	case <-overflow.markerComplete():
	case <-time.After(50 * time.Millisecond):
		t.Fatal("overflow marker was not rejected immediately")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("active marker calls = %d, want 1", got)
	}

	close(releaseFirst)
	for name, lifecycle := range map[string]*bookingLifecycle{"first": first, "second": second} {
		select {
		case <-lifecycle.markerComplete():
		case <-time.After(time.Second):
			t.Fatalf("%s marker did not finish", name)
		}
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("marker calls = %d, want 2 accepted jobs", got)
	}
}

func TestBookingExecutorReconcilesCleanupQueueOverflow(t *testing.T) {
	executor := newTestBookingExecutor(t, nil)
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	first := registerTestLifecycle(t, executor, func(string) error {
		close(firstStarted)
		select {
		case <-releaseFirst:
		case <-executor.stopCh:
		}
		return nil
	})
	second := registerTestLifecycle(t, executor, func(string) error { return nil })
	overflow := registerTestLifecycle(t, executor, func(string) error { return nil })

	if !first.cleanup(context.Background(), "first") {
		t.Fatal("first cleanup did not start")
	}
	select {
	case <-firstStarted:
	case <-time.After(time.Second):
		t.Fatal("first cleanup did not enter the worker")
	}
	if !second.cleanup(context.Background(), "second") {
		t.Fatal("second cleanup did not start")
	}
	if got := len(executor.cleanupQueue); got != 1 {
		t.Fatalf("cleanup queue length = %d, want 1", got)
	}
	if !overflow.cleanup(context.Background(), "overflow") {
		t.Fatal("overflow cleanup did not retain ownership")
	}
	overflow.mu.Lock()
	overflowQueued := overflow.cleanupQueued
	overflowRunning := overflow.cleanupRunning
	overflow.mu.Unlock()
	if overflowQueued || overflowRunning {
		t.Fatal("overflow cleanup unexpectedly entered the full executor")
	}

	close(releaseFirst)
	for name, lifecycle := range map[string]*bookingLifecycle{
		"first":    first,
		"second":   second,
		"overflow": overflow,
	} {
		select {
		case <-lifecycle.cleanupComplete():
		case <-time.After(time.Second):
			t.Fatalf("%s cleanup did not finish", name)
		}
		if findBookingLifecycle(lifecycle.bookingID) != nil {
			t.Fatalf("%s cleanup retained its lifecycle", name)
		}
	}
}

func TestBookingExecutorRetriesExhaustedCleanup(t *testing.T) {
	executor := newTestBookingExecutor(t, func(cfg *bookingExecutorConfig) {
		cfg.cleanupRetryDelay = 100 * time.Millisecond
	})
	var calls atomic.Int32
	lifecycle := registerTestLifecycle(t, executor, func(string) error {
		if calls.Add(1) <= cleanupMaxAttempts {
			return errors.New("transient cleanup outage")
		}
		return nil
	})

	if !lifecycle.cleanup(context.Background(), "retry exhausted cleanup") {
		t.Fatal("cleanup did not start")
	}
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("initial cleanup attempts did not finish")
	}
	if got := calls.Load(); got != cleanupMaxAttempts {
		t.Fatalf("initial cleanup calls = %d, want %d", got, cleanupMaxAttempts)
	}
	if findBookingLifecycle(lifecycle.bookingID) == nil {
		t.Fatal("exhausted cleanup lost retry ownership")
	}

	waitForLifecycleRemoval(t, lifecycle.bookingID)
	if got := calls.Load(); got != cleanupMaxAttempts+1 {
		t.Fatalf("cleanup calls after recovery = %d, want %d", got, cleanupMaxAttempts+1)
	}
}

func TestBookingExecutorExpiresCleanupTombstone(t *testing.T) {
	executor := newTestBookingExecutor(t, func(cfg *bookingExecutorConfig) {
		cfg.cleanupRetryDelay = time.Hour
		cfg.cleanupRetention = 250 * time.Millisecond
	})
	var calls atomic.Int32
	lifecycle := registerTestLifecycle(t, executor, func(string) error {
		calls.Add(1)
		return errors.New("persistent cleanup outage")
	})

	if !lifecycle.cleanup(context.Background(), "expire cleanup ownership") {
		t.Fatal("cleanup did not start")
	}
	select {
	case <-lifecycle.cleanupComplete():
	case <-time.After(time.Second):
		t.Fatal("cleanup attempts did not finish")
	}
	if got := calls.Load(); got != cleanupMaxAttempts {
		t.Fatalf("cleanup calls = %d, want %d", got, cleanupMaxAttempts)
	}
	if findBookingLifecycle(lifecycle.bookingID) == nil {
		t.Fatal("cleanup tombstone expired before its retention deadline")
	}

	waitForLifecycleRemoval(t, lifecycle.bookingID)
	lifecycle.mu.Lock()
	expired := lifecycle.cleanupExpired
	lifecycle.mu.Unlock()
	if !expired {
		t.Fatal("cleanup lifecycle was removed without recording expiry")
	}
	if got := calls.Load(); got != cleanupMaxAttempts {
		t.Fatalf("cleanup calls after expiry = %d, want %d", got, cleanupMaxAttempts)
	}
}

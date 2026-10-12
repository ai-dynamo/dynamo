/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dgdrrunpublisher

import (
	"context"
	"errors"
	"log"
	"os"
	"reflect"
	"sync"
	"time"
)

// Input is the latest accepted desired state plus the most recent problem reading the
// input. Snapshot is never mutated after it is stored, so readers need no copy.
type Input struct {
	// Snapshot is the last valid snapshot, or nil if none has been accepted yet.
	Snapshot *Snapshot
	// Err is why the most recent read failed (invalid or regressing snapshot, or an I/O
	// error). It is cleared by the next valid read.
	Err error
}

// InputStore atomically holds the latest desired state for the reconciler.
type InputStore struct {
	mu    sync.RWMutex
	input Input
}

// Load returns the current input.
func (s *InputStore) Load() Input {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.input
}

// accept stores snap as the latest desired state if it differs semantically from the
// current one, and reports whether it did. A snapshot that moves progress backwards is
// rejected and recorded as the input error; the last valid snapshot stays in place.
func (s *InputStore) accept(snap *Snapshot) (changed bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := checkProgress(s.input.Snapshot, snap); err != nil {
		s.input.Err = err
		return false
	}
	s.input.Err = nil
	if s.input.Snapshot != nil && sameContent(s.input.Snapshot, snap) {
		return false
	}
	s.input.Snapshot = snap
	return true
}

// fail records why the input could not be read, keeping the last valid snapshot.
func (s *InputStore) fail(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.input.Err = err
}

// sameContent compares two snapshots ignoring the write timestamp: rewriting identical
// content with a new timestamp is not a change the cluster needs to hear about.
func sameContent(a, b *Snapshot) bool {
	left, right := *a, *b
	left.Timestamp, right.Timestamp = "", ""
	return reflect.DeepEqual(left, right)
}

// checkProgress rejects a snapshot that moves progress backwards relative to the
// previously accepted one: an older round, or fewer evaluated points within the same
// round. A terminal snapshot is exempt from the evaluated check because its count is the
// search's final total, not the live counter.
func checkProgress(previous, next *Snapshot) error {
	if previous == nil {
		return nil
	}
	if next.Progress.Round < previous.Progress.Round {
		return violation("snapshot round regressed from %d to %d", previous.Progress.Round, next.Progress.Round)
	}
	if !next.Run.Terminal && next.Progress.Round == previous.Progress.Round && next.Progress.Evaluated < previous.Progress.Evaluated {
		return violation("snapshot evaluated count regressed from %d to %d in round %d", previous.Progress.Evaluated, next.Progress.Evaluated, previous.Progress.Round)
	}
	return nil
}

// Observer is the input-observation loop: it checks the Sweeper's snapshot file and, on a
// semantic change, stores it and enqueues the run key. It never talks to the API server.
type Observer struct {
	Store *InputStore
	Path  string
	// Enqueue asks the controller to reconcile the run. It must not block.
	Enqueue func()
	// MinInterval is the check interval after a change; it doubles while the input is
	// unchanged, up to MaxInterval.
	MinInterval time.Duration
	MaxInterval time.Duration

	mu      sync.Mutex // serializes Check between the loop and the reconciler
	lastErr string
}

// Check reads the snapshot once and reports whether it was a semantic change (in which
// case it has been stored and the run key enqueued). A missing file is not an error. The
// reconciler also calls it directly once the Sweeper has exited, to be certain the
// Sweeper's final write has been seen.
func (o *Observer) Check() bool {
	o.mu.Lock()
	defer o.mu.Unlock()

	data, err := os.ReadFile(o.Path)
	if errors.Is(err, os.ErrNotExist) {
		return false
	}
	if err != nil {
		o.noteError(err)
		o.Store.fail(err)
		return false
	}
	snap, err := ParseSnapshot(data)
	if err != nil {
		o.noteError(err)
		o.Store.fail(err)
		return false
	}
	changed := o.Store.accept(snap)
	if err := o.Store.Load().Err; err != nil {
		o.noteError(err)
		return false
	}
	o.lastErr = ""
	if changed && o.Enqueue != nil {
		o.Enqueue()
	}
	return changed
}

// noteError logs a problem once per distinct message rather than on every check.
func (o *Observer) noteError(err error) {
	if err.Error() == o.lastErr {
		return
	}
	o.lastErr = err.Error()
	log.Printf("reading snapshot: %v", err)
}

// NextInterval is the delay before the next check: it resets to min after a change and
// otherwise doubles up to max.
func NextInterval(current, minimum, maximum time.Duration, changed bool) time.Duration {
	if changed || current < minimum {
		return minimum
	}
	next := current * 2
	if next > maximum {
		return maximum
	}
	return next
}

// Run checks the snapshot until ctx is done.
func (o *Observer) Run(ctx context.Context) error {
	interval := o.MinInterval
	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-timer.C:
		}
		interval = NextInterval(interval, o.MinInterval, o.MaxInterval, o.Check())
		timer.Reset(interval)
	}
}

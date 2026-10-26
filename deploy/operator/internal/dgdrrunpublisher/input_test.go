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
	"os"
	"path/filepath"
	"testing"
	"time"
)

func newObserver(t *testing.T) (*Observer, *InputStore, string, *int) {
	t.Helper()
	dir := t.TempDir()
	enqueued := 0
	store := &InputStore{}
	return &Observer{
		Store:       store,
		Path:        filepath.Join(dir, SnapshotFileName),
		Enqueue:     func() { enqueued++ },
		MinInterval: time.Millisecond,
		MaxInterval: 4 * time.Millisecond,
	}, store, dir, &enqueued
}

func TestMissingSnapshotIsNotAChange(t *testing.T) {
	o, store, _, enqueued := newObserver(t)
	if o.Check() || *enqueued != 0 || store.Load().Snapshot != nil || store.Load().Err != nil {
		t.Fatalf("missing file: input=%+v enqueued=%d", store.Load(), *enqueued)
	}
}

func TestChangedSnapshotIsStoredAndEnqueuedOnce(t *testing.T) {
	o, store, dir, enqueued := newObserver(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	if !o.Check() || *enqueued != 1 {
		t.Fatalf("first snapshot: enqueued=%d", *enqueued)
	}
	if store.Load().Snapshot == nil || store.Load().Snapshot.Candidates[0].ID != "a" {
		t.Fatalf("snapshot not stored: %+v", store.Load())
	}
	if o.Check() || *enqueued != 1 {
		t.Fatalf("unchanged input must not enqueue again: enqueued=%d", *enqueued)
	}
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}))
	if !o.Check() || *enqueued != 2 {
		t.Fatalf("changed input must enqueue: enqueued=%d", *enqueued)
	}
}

func TestRewriteWithOnlyANewTimestampIsNotAChange(t *testing.T) {
	for _, phase := range []string{PhaseRunning, PhaseSucceeded} {
		t.Run(phase, func(t *testing.T) {
			o, _, dir, enqueued := newObserver(t)

			t.Log("Accept the first snapshot")
			writeSnapshot(t, dir, withTimestamp(t, snapshotJSON(t, phase, cand{id: "a"}), "2026-10-05T21:35:12Z"))
			o.Check()

			t.Log("Rewrite the same snapshot with only a new timestamp")
			writeSnapshot(t, dir, withTimestamp(t, snapshotJSON(t, phase, cand{id: "a"}), "2026-10-05T21:35:14Z"))
			if o.Check() || *enqueued != 1 {
				t.Fatalf("a new timestamp alone is not a semantic change: enqueued=%d", *enqueued)
			}
		})
	}
}

func TestInvalidSnapshotKeepsTheLastValidOneAndIsRecorded(t *testing.T) {
	o, store, dir, enqueued := newObserver(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	o.Check()
	writeSnapshot(t, dir, []byte(`{"schemaVersion":2}`))
	if o.Check() || *enqueued != 1 {
		t.Fatalf("an invalid snapshot must not enqueue: enqueued=%d", *enqueued)
	}
	in := store.Load()
	if !errors.Is(in.Err, ErrProtocolViolation) || in.Snapshot == nil || in.Snapshot.Candidates[0].ID != "a" {
		t.Fatalf("input = %+v", in)
	}
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}))
	if !o.Check() || store.Load().Err != nil {
		t.Fatalf("a valid snapshot must clear the error: %+v", store.Load())
	}
}

func TestRegressingSnapshotIsRejected(t *testing.T) {
	for name, regressed := range map[string][2]int{"older round": {2, 99}, "fewer evaluated in the same round": {3, 4}} {
		t.Run(name, func(t *testing.T) {
			o, store, dir, enqueued := newObserver(t)
			writeProgress(t, dir, PhaseRunning, 3, 5)
			o.Check()
			writeProgress(t, dir, PhaseRunning, regressed[0], regressed[1])
			if o.Check() || *enqueued != 1 {
				t.Fatalf("a regressing snapshot must not be stored: enqueued=%d", *enqueued)
			}
			in := store.Load()
			if !errors.Is(in.Err, ErrProtocolViolation) || ExitCode(in.Err) != ExitProtocolViolation {
				t.Fatalf("err = %v", in.Err)
			}
			if in.Snapshot.Progress.Round != 3 || in.Snapshot.Progress.Evaluated != 5 {
				t.Fatalf("the last valid snapshot must stay: %+v", in.Snapshot.Progress)
			}
		})
	}
}

func TestTerminalSnapshotMayReportItsFinalEvaluatedTotal(t *testing.T) {
	o, store, dir, _ := newObserver(t)
	writeProgress(t, dir, PhaseRunning, 3, 9)
	o.Check()
	writeProgress(t, dir, PhaseSucceeded, 3, 7)
	if !o.Check() || store.Load().Err != nil || !store.Load().Snapshot.Run.Terminal {
		t.Fatalf("terminal snapshot rejected: %+v", store.Load())
	}
}

func TestTerminalSnapshotCannotBeReplaced(t *testing.T) {
	tests := []struct {
		name        string
		replacement []byte
	}{
		{name: "return to running", replacement: snapshotJSON(t, PhaseRunning, cand{id: "a"})},
		{name: "change final candidates", replacement: snapshotJSON(t, PhaseSucceeded, cand{id: "a"}, cand{id: "b"})},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			o, store, dir, enqueued := newObserver(t)

			t.Log("Accept the terminal snapshot")
			writeSnapshot(t, dir, snapshotJSON(t, PhaseSucceeded, cand{id: "a"}))
			if !o.Check() {
				t.Fatal("terminal snapshot was not accepted")
			}

			t.Log("Attempt to replace the latched terminal state")
			writeSnapshot(t, dir, test.replacement)
			if o.Check() || *enqueued != 1 {
				t.Fatalf("replacement must not be stored: enqueued=%d", *enqueued)
			}

			input := store.Load()
			if !errors.Is(input.Err, ErrProtocolViolation) {
				t.Fatalf("err = %v, want protocol violation", input.Err)
			}
			if input.Snapshot.Run.Phase != PhaseSucceeded || len(input.Snapshot.Candidates) != 1 {
				t.Fatalf("terminal snapshot was replaced: %+v", input.Snapshot)
			}
		})
	}
}

func TestUnreadableSnapshotIsRecordedAsANonViolation(t *testing.T) {
	o, store, dir, _ := newObserver(t)
	if err := os.Mkdir(filepath.Join(dir, SnapshotFileName), 0o700); err != nil { // reading a directory fails
		t.Fatal(err)
	}
	o.Check()
	if err := store.Load().Err; err == nil || errors.Is(err, ErrProtocolViolation) {
		t.Fatalf("err = %v, want a plain read error", err)
	}
}

func TestNextIntervalBacksOffWhileUnchangedAndResetsOnChange(t *testing.T) {
	minimum, maximum := time.Second, 5*time.Second
	got := []time.Duration{}
	interval := minimum
	for i := 0; i < 5; i++ {
		interval = NextInterval(interval, minimum, maximum, false)
		got = append(got, interval)
	}
	want := []time.Duration{2 * time.Second, 4 * time.Second, 5 * time.Second, 5 * time.Second, 5 * time.Second}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("intervals = %v, want %v", got, want)
		}
	}
	if NextInterval(5*time.Second, minimum, maximum, true) != minimum {
		t.Fatal("a change must reset the interval")
	}
}

func TestObserverRunEnqueuesAChangeAndStopsWithItsContext(t *testing.T) {
	o, _, dir, _ := newObserver(t)
	events := make(chan struct{}, 4)
	o.Enqueue = func() { events <- struct{}{} }
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- o.Run(ctx) }()
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	select {
	case <-events:
	case <-time.After(5 * time.Second):
		t.Fatal("the observer never reported the snapshot")
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run returned %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not stop")
	}
}

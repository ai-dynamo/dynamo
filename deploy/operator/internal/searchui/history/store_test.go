/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package history

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestLocalStoreRoundTrip(t *testing.T) {
	ctx := context.Background()
	store, err := NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 11, 12, 10, 0, 0, 0, time.UTC)

	t.Log("Store one run, two status samples, two candidates and a log")
	run := RunEnvelope{Namespace: "ns", Name: "run-1", UID: "uid-1", FirstSeen: base, LastSeen: base,
		Object: map[string]any{"metadata": map[string]any{"name": "run-1"}}}
	if err := store.PutRun(ctx, run); err != nil {
		t.Fatal(err)
	}
	for i, published := range []float64{1, 3} {
		sample := StatusSample{Time: base.Add(time.Duration(i) * time.Minute),
			Status: map[string]any{"progress": map[string]any{"candidates": map[string]any{"published": published}}}}
		if err := store.PutRunSample(ctx, "ns", "uid-1", sample); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"run-1-aaaa-point-1", "run-1-aaaa-point-2"} {
		env := CandidateEnvelope{Namespace: "ns", Name: name, RunUID: "uid-1", FirstSeen: base, LastSeen: base,
			Object: map[string]any{"metadata": map[string]any{"name": name}}}
		if err := store.PutCandidate(ctx, env); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.PutLog(ctx, "ns", "uid-1", "sweeper", []byte("round 1 started\n")); err != nil {
		t.Fatal(err)
	}

	t.Log("Read everything back")
	runs, err := store.ListRuns(ctx)
	if err != nil || len(runs) != 1 || runs[0].UID != "uid-1" {
		t.Fatalf("ListRuns = %+v, %v", runs, err)
	}
	samples, err := store.ListRunSamples(ctx, "ns", "uid-1")
	if err != nil || len(samples) != 2 || !samples[0].Time.Before(samples[1].Time) {
		t.Fatalf("ListRunSamples = %+v, %v", samples, err)
	}
	candidates, err := store.ListCandidates(ctx, "ns", "uid-1")
	if err != nil || len(candidates) != 2 {
		t.Fatalf("ListCandidates = %+v, %v", candidates, err)
	}
	logData, err := store.GetLog(ctx, "ns", "uid-1", "sweeper")
	if err != nil || string(logData) != "round 1 started\n" {
		t.Fatalf("GetLog = %q, %v", logData, err)
	}

	t.Log("Overwriting a run keeps a single run.json; a missing log is ErrNotFound")
	run.LastSeen = base.Add(time.Hour)
	if err := store.PutRun(ctx, run); err != nil {
		t.Fatal(err)
	}
	runs, _ = store.ListRuns(ctx)
	if len(runs) != 1 || !runs[0].LastSeen.Equal(base.Add(time.Hour)) {
		t.Fatalf("ListRuns after overwrite = %+v", runs)
	}
	if _, err := store.GetLog(ctx, "ns", "uid-1", "publisher"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetLog missing = %v, want ErrNotFound", err)
	}
}

func TestLocalStoreRejectsEscapingKeys(t *testing.T) {
	store, err := NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	err = store.PutLog(context.Background(), "ns", "uid", "../../etc/passwd", []byte("x"))
	if err == nil {
		t.Fatal("expected an error for a container name with path separators")
	}
}

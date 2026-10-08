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
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
)

// fakeCluster records every mutation so tests can assert exactly which writes happened.
type fakeCluster struct {
	mu          sync.Mutex
	candidates  map[string]string                         // name -> id
	created     map[string]dgdcreconcile.DesiredCandidate // name -> content at creation
	creates     []string
	deletes     []string
	patches     []RunStatus
	ops         []string // ordered log of every mutation
	incomplete  map[string]bool
	state       SweeperState
	patchErr    error
	createErr   error
	createCalls int
}

func newFakeCluster() *fakeCluster {
	return &fakeCluster{candidates: map[string]string{}, created: map[string]dgdcreconcile.DesiredCandidate{}, incomplete: map[string]bool{}}
}

func (f *fakeCluster) ListCandidates(context.Context) ([]dgdcreconcile.CurrentDGDC, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []dgdcreconcile.CurrentDGDC
	for name, id := range f.candidates {
		out = append(out, dgdcreconcile.CurrentDGDC{Name: name, ID: id, Incomplete: f.incomplete[name]})
	}
	return out, nil
}

func (f *fakeCluster) CreateCandidate(_ context.Context, name string, c dgdcreconcile.DesiredCandidate) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.createCalls++
	if f.createErr != nil {
		return f.createErr
	}
	delete(f.incomplete, name)
	f.ops = append(f.ops, "create:"+name)
	f.candidates[name] = c.ID
	f.created[name] = c
	f.creates = append(f.creates, name)
	return nil
}

func (f *fakeCluster) DeleteCandidate(_ context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.candidates, name)
	f.ops = append(f.ops, "delete:"+name)
	f.deletes = append(f.deletes, name)
	return nil
}

func (f *fakeCluster) PatchRunStatus(_ context.Context, s RunStatus) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.patchErr != nil {
		return f.patchErr
	}
	f.ops = append(f.ops, "patch")
	f.patches = append(f.patches, s)
	return nil
}

func (f *fakeCluster) RunUID(context.Context) (string, error) { return testRunUID, nil }

func (f *fakeCluster) SweeperState(context.Context) (SweeperState, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.state, nil
}

func (f *fakeCluster) setState(s SweeperState) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.state = s
}

const validManifest = "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\nspec: {}\n"

const testRunUID = "uid-1"

func cn(id string) string { return CandidateName("run", testRunUID, id) }

type cand struct {
	id      string
	failed  bool
	metrics map[string]any
}

// snapshotJSON renders a snapshot (JSON is valid YAML, which the real parser accepts).
func snapshotJSON(t *testing.T, phase string, cands ...cand) []byte {
	t.Helper()
	var list []map[string]any
	for _, c := range cands {
		entry := map[string]any{"id": c.id, "parameters": map[string]any{"tp": 2}, "metrics": c.metrics}
		if c.failed {
			entry["outcome"] = OutcomeMaterializationFailed
			entry["error"] = "boom"
		} else {
			entry["outcome"] = OutcomeMaterialized
			entry["manifest"] = validManifest
		}
		list = append(list, entry)
	}
	data, err := json.Marshal(map[string]any{
		"schemaVersion": 1,
		"run":           map[string]any{"phase": phase, "terminal": phase != PhaseRunning},
		"progress":      map[string]any{"round": 3, "evaluated": len(cands)},
		"candidates":    list,
	})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func writeSnapshot(t *testing.T, dir string, data []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, SnapshotFileName), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func newPublisher(t *testing.T) (*Publisher, *fakeCluster, string) {
	t.Helper()
	dir := t.TempDir()
	fc := newFakeCluster()
	return &Publisher{Cluster: fc, SnapshotDir: dir, RunName: "run", PollInterval: time.Millisecond}, fc, dir
}

func mustParse(t *testing.T, data []byte) *Snapshot {
	t.Helper()
	snap, err := ParseSnapshot(data)
	if err != nil {
		t.Fatal(err)
	}
	return snap
}

func TestSwapLeavesBothCandidatesUntouched(t *testing.T) {
	p, fc, _ := newPublisher(t)
	ctx := context.Background()
	if err := p.Reconcile(ctx, mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "a", metrics: map[string]any{"score": 2.0}}, cand{id: "b", metrics: map[string]any{"score": 1.0}}))); err != nil {
		t.Fatal(err)
	}
	before := map[string]dgdcreconcile.DesiredCandidate{}
	for name, content := range fc.created {
		before[name] = content
	}
	if err := p.Reconcile(ctx, mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "b", metrics: map[string]any{"score": 3.0}}, cand{id: "a", metrics: map[string]any{"score": 2.0}}))); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fc.creates, []string{cn("a"), cn("b")}) || len(fc.deletes) != 0 {
		t.Fatalf("swap touched DGDCs: creates=%v deletes=%v", fc.creates, fc.deletes)
	}
	if !reflect.DeepEqual(before, fc.created) {
		t.Fatalf("a published DGDC's content changed: before=%v after=%v", before, fc.created)
	}
	if got, want := fc.patches[len(fc.patches)-1].CandidateNames, []string{cn("b"), cn("a")}; !reflect.DeepEqual(got, want) {
		t.Fatalf("ordered references = %v, want %v", got, want)
	}
}

func TestEvaluatedPointReachesTheClusterComplete(t *testing.T) {
	p, fc, _ := newPublisher(t)
	if err := p.Reconcile(context.Background(), mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "a", metrics: map[string]any{"goodputPerGpu": 87.7}}))); err != nil {
		t.Fatal(err)
	}
	got := fc.created[cn("a")]
	if got.Spec != validManifest {
		t.Fatalf("manifest = %q", got.Spec)
	}
	if v, ok := got.Parameters["tp"]; !ok || v != float64(2) {
		t.Fatalf("resolved evaluation parameters were lost: %v", got.Parameters)
	}
	if v, ok := got.Metrics["goodputPerGpu"]; !ok || v != 87.7 {
		t.Fatalf("evaluation metrics were lost: %v", got.Metrics)
	}
}

func TestReturningCandidateIsRecreatedIdentically(t *testing.T) {
	p, fc, _ := newPublisher(t)
	ctx := context.Background()
	steps := [][]cand{
		{{id: "a"}, {id: "b", metrics: map[string]any{"score": 1.0}}},
		{{id: "a"}},
		{{id: "a"}, {id: "b", metrics: map[string]any{"score": 1.0}}},
	}
	var firstB dgdcreconcile.DesiredCandidate
	for i, step := range steps {
		if err := p.Reconcile(ctx, mustParse(t, snapshotJSON(t, PhaseRunning, step...))); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			firstB = fc.created[cn("b")]
		}
	}
	if !reflect.DeepEqual(fc.creates, []string{cn("a"), cn("b"), cn("b")}) || !reflect.DeepEqual(fc.deletes, []string{cn("b")}) {
		t.Fatalf("creates=%v deletes=%v", fc.creates, fc.deletes)
	}
	if !reflect.DeepEqual(firstB, fc.created[cn("b")]) {
		t.Fatalf("recreated candidate differs: %v vs %v", firstB, fc.created[cn("b")])
	}
}

func TestRepeatedSnapshotCausesNoWrites(t *testing.T) {
	p, fc, _ := newPublisher(t)
	ctx := context.Background()
	snap := mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	for i := 0; i < 3; i++ {
		if err := p.Reconcile(ctx, snap); err != nil {
			t.Fatal(err)
		}
	}
	if len(fc.creates) != 1 || len(fc.patches) != 1 || len(fc.deletes) != 0 {
		t.Fatalf("creates=%v patches=%d deletes=%v", fc.creates, len(fc.patches), fc.deletes)
	}
}

func TestDroppedCandidateIsDeletedAfterStatusPatch(t *testing.T) {
	p, fc, _ := newPublisher(t)
	ctx := context.Background()
	_ = p.Reconcile(ctx, mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"})))
	_ = p.Reconcile(ctx, mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "a"})))
	if !reflect.DeepEqual(fc.deletes, []string{cn("b")}) {
		t.Fatalf("deletes = %v", fc.deletes)
	}
	if got := fc.patches[len(fc.patches)-1].CandidateNames; !reflect.DeepEqual(got, []string{cn("a")}) {
		t.Fatalf("status refs = %v", got)
	}
	want := []string{"create:" + cn("a"), "create:" + cn("b"), "patch", "patch", "delete:" + cn("b")}
	if !reflect.DeepEqual(fc.ops, want) {
		t.Fatalf("ops = %v, want %v", fc.ops, want)
	}
}

func TestFailedMaterializationIsNotCreatedButCounted(t *testing.T) {
	p, fc, _ := newPublisher(t)
	if err := p.Reconcile(context.Background(), mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "x", failed: true}))); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fc.creates, []string{cn("a")}) {
		t.Fatalf("creates = %v", fc.creates)
	}
	if got := fc.patches[0].CandidateNames; !reflect.DeepEqual(got, []string{cn("a")}) {
		t.Fatalf("refs must not include the failed candidate: %v", got)
	}
	if msg := fc.patches[0].Message; !strings.Contains(msg, "1 candidate(s) failed materialization") {
		t.Fatalf("message = %q", msg)
	}
}

func withTimestamp(t *testing.T, data []byte, timestamp string) []byte {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	doc["timestamp"] = timestamp
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestLastProgressComesFromTheSnapshotTimestamp(t *testing.T) {
	p, fc, _ := newPublisher(t)
	data := withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), "2026-10-05T21:35:12Z")
	if err := p.Reconcile(context.Background(), mustParse(t, data)); err != nil {
		t.Fatal(err)
	}
	want := time.Date(2026, 10, 5, 21, 35, 12, 0, time.UTC)
	if got := fc.patches[0].LastProgress; !got.Equal(want) {
		t.Fatalf("LastProgress = %v, want %v", got, want)
	}
}

func TestProgressWithoutRankChangeStillPatchesTheProgressTime(t *testing.T) {
	p, fc, _ := newPublisher(t)
	first := withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), "2026-10-05T21:35:12Z")
	second := withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), "2026-10-05T21:35:14Z")
	for _, data := range [][]byte{first, second, second} {
		if err := p.Reconcile(context.Background(), mustParse(t, data)); err != nil {
			t.Fatal(err)
		}
	}
	if len(fc.patches) != 2 {
		t.Fatalf("want one patch per distinct progress time, got %d", len(fc.patches))
	}
	if !reflect.DeepEqual(fc.patches[0].CandidateNames, fc.patches[1].CandidateNames) {
		t.Fatalf("refs changed: %v", fc.patches)
	}
	if len(fc.creates) != 1 {
		t.Fatalf("an unchanged candidate must not be created again: %v", fc.creates)
	}
}

func TestMissingOrMalformedTimestampIsNotAnError(t *testing.T) {
	for _, timestamp := range []string{"", "yesterday"} {
		p, fc, _ := newPublisher(t)
		data := withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), timestamp)
		if err := p.Reconcile(context.Background(), mustParse(t, data)); err != nil {
			t.Fatalf("timestamp %q: %v", timestamp, err)
		}
		if got := fc.patches[0].LastProgress; !got.IsZero() {
			t.Fatalf("timestamp %q: LastProgress = %v, want zero", timestamp, got)
		}
	}
}

func TestCreateFailureDoesNotPatchStatus(t *testing.T) {
	p, fc, _ := newPublisher(t)
	fc.createErr = errors.New("apiserver down")
	if err := p.Reconcile(context.Background(), mustParse(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}))); err == nil {
		t.Fatal("want error")
	}
	if len(fc.patches) != 0 {
		t.Fatalf("status must not reference candidates that were not created: %v", fc.patches)
	}
}

func TestParseSnapshotValidation(t *testing.T) {
	bad := map[string]string{
		"schema":        `{"schemaVersion":2,"run":{"phase":"Running","terminal":false}}`,
		"phase":         `{"schemaVersion":1,"run":{"phase":"Weird","terminal":false}}`,
		"terminal flag": `{"schemaVersion":1,"run":{"phase":"Succeeded","terminal":false}}`,
		"duplicate":     `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"},{"id":"a","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
		"no manifest":   `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialized"}]}`,
		"no error":      `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialization_failed"}]}`,
		"unsafe id":     `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"Not_Valid","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
		"long id":       `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"%s","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
		"wrong kind":    `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"a","outcome":"materialized","manifest":"kind: ConfigMap\\nspec: {}\\n"}]}`,
		"empty id":      `{"schemaVersion":1,"run":{"phase":"Running","terminal":false},"candidates":[{"id":"","outcome":"materialized","manifest":"apiVersion: nvidia.com/v1beta1\\nkind: DynamoGraphDeployment\\nspec: {}\\n"}]}`,
	}
	bad["long id"] = strings.Replace(bad["long id"], "%s", strings.Repeat("a", 64), 1)
	for name, doc := range bad {
		if _, err := ParseSnapshot([]byte(doc)); !errors.Is(err, ErrProtocolViolation) {
			t.Errorf("%s: want protocol violation, got %v", name, err)
		}
	}
}

func runWithTimeout(t *testing.T, p *Publisher) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return p.Run(ctx)
}

func lastPatch(t *testing.T, fc *fakeCluster) RunStatus {
	t.Helper()
	if len(fc.patches) == 0 {
		t.Fatal("no run status patch was written")
	}
	return fc.patches[len(fc.patches)-1]
}

// Orderly completion: terminal snapshot is reconciled and the publisher exits 0 even
// though the Sweeper container is still shutting down.
func TestRunTerminalSnapshotAcknowledges(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseSucceeded, cand{id: "a"}))
	err := runWithTimeout(t, p)
	if err != nil || ExitCode(err) != ExitAcknowledged {
		t.Fatalf("err = %v", err)
	}
	if got := lastPatch(t, fc).CandidateNames; !reflect.DeepEqual(got, []string{cn("a")}) {
		t.Fatalf("refs = %v", got)
	}
}

func TestRunTerminalFailedSnapshotIsReconciledThenFails(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseFailed, cand{id: "a"}))
	err := runWithTimeout(t, p)
	if !errors.Is(err, ErrRunFailed) || ExitCode(err) != ExitRunFailed {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(fc.creates, []string{cn("a")}) {
		t.Fatalf("creates = %v", fc.creates)
	}
}

func TestRunReconcilesTerminalSnapshotWrittenAfterRunningOne(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	done := make(chan error, 1)
	go func() { done <- runWithTimeout(t, p) }()
	time.Sleep(20 * time.Millisecond)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseSucceeded, cand{id: "a"}, cand{id: "b"}))
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if got := lastPatch(t, fc).CandidateNames; len(got) != 2 {
		t.Fatalf("terminal projection lost: %v", got)
	}
}

// Abrupt non-zero exit without a terminal snapshot: the last snapshot is reconciled and
// the publisher exits 0.
func TestRunCrashWithoutTerminalSnapshotReconcilesLastSnapshot(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	fc.setState(SweeperState{Exited: true, ExitCode: 137})
	if err := runWithTimeout(t, p); err != nil {
		t.Fatal(err)
	}
	if got := lastPatch(t, fc).CandidateNames; !reflect.DeepEqual(got, []string{cn("a")}) {
		t.Fatalf("refs = %v", got)
	}
}

// Nothing was ever written and the Sweeper crashed: nothing to reconcile, exit 0.
func TestRunCrashBeforeAnySnapshot(t *testing.T) {
	p, fc, _ := newPublisher(t)
	fc.setState(SweeperState{Exited: true, ExitCode: 137})
	if err := runWithTimeout(t, p); err != nil {
		t.Fatal(err)
	}
	if len(fc.creates) != 0 || len(fc.patches) != 0 {
		t.Fatalf("unexpected writes: %v %v", fc.creates, fc.patches)
	}
}

// Exit 0 without a terminal snapshot is a protocol violation: the last snapshot is still
// reconciled but the publisher fails with the dedicated exit code.
func TestRunCleanExitWithoutTerminalSnapshotIsProtocolViolation(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	fc.setState(SweeperState{Exited: true, ExitCode: 0})
	err := runWithTimeout(t, p)
	if !errors.Is(err, ErrMissingTerminalSnapshot) || ExitCode(err) != ExitProtocolViolation {
		t.Fatalf("err = %v", err)
	}
	if !reflect.DeepEqual(fc.creates, []string{cn("a")}) {
		t.Fatalf("last snapshot must still be reconciled: %v", fc.creates)
	}
}

func TestRunCleanExitWithNoSnapshotAtAllIsProtocolViolation(t *testing.T) {
	p, fc, _ := newPublisher(t)
	fc.setState(SweeperState{Exited: true, ExitCode: 0})
	if err := runWithTimeout(t, p); !errors.Is(err, ErrMissingTerminalSnapshot) {
		t.Fatalf("err = %v", err)
	}
}

func TestRunFinalReconcileFailureFails(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	fc.createErr = errors.New("apiserver down")
	fc.setState(SweeperState{Exited: true, ExitCode: 1})
	err := runWithTimeout(t, p)
	if err == nil || ExitCode(err) != ExitReconcileFailed {
		t.Fatalf("err = %v", err)
	}
	if fc.createCalls < finalReconcileAttempts {
		t.Fatalf("expected %d attempts, got %d", finalReconcileAttempts, fc.createCalls)
	}
}

func TestRunTerminalSnapshotReconcileFailureIsRetriedUntilSuccess(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, snapshotJSON(t, PhaseSucceeded, cand{id: "a"}))
	fc.createErr = errors.New("flaky")
	done := make(chan error, 1)
	go func() { done <- runWithTimeout(t, p) }()
	time.Sleep(20 * time.Millisecond)
	fc.mu.Lock()
	fc.createErr = nil
	fc.mu.Unlock()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fc.creates, []string{cn("a")}) {
		t.Fatalf("creates = %v", fc.creates)
	}
}

func TestRunMalformedProjectionFailsImmediately(t *testing.T) {
	p, fc, dir := newPublisher(t)
	fc.candidates["x"] = "dup"
	fc.candidates["y"] = "dup"
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	err := runWithTimeout(t, p)
	var inputErr *dgdcreconcile.DiffInputError
	if !errors.As(err, &inputErr) {
		t.Fatalf("err = %v", err)
	}
}

func TestRunContextCancelledReturnsError(t *testing.T) {
	p, _, _ := newPublisher(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.Run(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}

func TestCandidateNameIsBoundedAndUniquePerRunIncarnation(t *testing.T) {
	id := "evaluated-point-0123456789ab"
	if cn("a") != CandidateName("run", testRunUID, "a") || cn("a") == CandidateName("run", "uid-2", "a") {
		t.Fatalf("a recreated run must get different names: %q", cn("a"))
	}
	long := strings.Repeat("r", 253)
	other := strings.Repeat("r", 252) + "s"
	name := CandidateName(long, testRunUID, id)
	if len(name) > 253 || !strings.HasSuffix(name, "-"+id) || name == CandidateName(other, testRunUID, id) || name != CandidateName(long, testRunUID, id) {
		t.Fatalf("name = %q (%d)", name, len(name))
	}
}

func TestUnchangedSnapshotRepairsDrift(t *testing.T) {
	p, fc, dir := newPublisher(t)
	ctx := context.Background()
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	if err := p.syncOnce(ctx); err != nil {
		t.Fatal(err)
	}
	delete(fc.candidates, cn("a")) // someone deletes the referenced candidate
	if err := p.syncOnce(ctx); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fc.creates, []string{cn("a"), cn("a")}) {
		t.Fatalf("creates = %v", fc.creates)
	}
}

func TestCandidateWithoutStatusIsCompletedBeforeItIsReferenced(t *testing.T) {
	p, fc, dir := newPublisher(t)
	ctx := context.Background()
	fc.candidates[cn("a")] = "a"
	fc.incomplete[cn("a")] = true // created, but the status write failed
	writeSnapshot(t, dir, snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	if err := p.syncOnce(ctx); err != nil {
		t.Fatal(err)
	}
	if want := []string{"create:" + cn("a"), "patch"}; !reflect.DeepEqual(fc.ops, want) {
		t.Fatalf("ops = %v, want %v", fc.ops, want)
	}
}

func writeProgress(t *testing.T, dir, phase string, round, evaluated int) {
	t.Helper()
	doc := map[string]any{}
	if err := json.Unmarshal(snapshotJSON(t, phase, cand{id: "a"}), &doc); err != nil {
		t.Fatal(err)
	}
	doc["progress"] = map[string]any{"round": round, "evaluated": evaluated}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	writeSnapshot(t, dir, data)
}

func TestRegressingSnapshotIsRejectedBeforeReconcile(t *testing.T) {
	for name, regressed := range map[string][2]int{"older round": {2, 99}, "fewer evaluated in the same round": {3, 4}} {
		t.Run(name, func(t *testing.T) {
			p, fc, dir := newPublisher(t)
			ctx := context.Background()
			writeProgress(t, dir, PhaseRunning, 3, 5)
			if err := p.syncOnce(ctx); err != nil {
				t.Fatal(err)
			}
			ops := len(fc.ops)
			writeProgress(t, dir, PhaseRunning, regressed[0], regressed[1])
			err := p.syncOnce(ctx)
			if !errors.Is(err, ErrProtocolViolation) || ExitCode(err) != ExitProtocolViolation {
				t.Fatalf("err = %v", err)
			}
			if len(fc.ops) != ops {
				t.Fatalf("a regressing snapshot must not be reconciled: %v", fc.ops[ops:])
			}
		})
	}
}

func TestTerminalSnapshotMayReportItsFinalEvaluatedTotal(t *testing.T) {
	p, _, dir := newPublisher(t)
	ctx := context.Background()
	writeProgress(t, dir, PhaseRunning, 3, 9)
	if err := p.syncOnce(ctx); err != nil {
		t.Fatal(err)
	}
	writeProgress(t, dir, PhaseSucceeded, 3, 7)
	if err := p.syncOnce(ctx); err != nil {
		t.Fatalf("terminal snapshot rejected: %v", err)
	}
}

func TestInvalidFinalSnapshotIsAProtocolViolationExit(t *testing.T) {
	p, fc, dir := newPublisher(t)
	writeSnapshot(t, dir, []byte(`{"schemaVersion":2}`))
	fc.setState(SweeperState{Exited: true, ExitCode: 0})
	err := runWithTimeout(t, p)
	if ExitCode(err) != ExitProtocolViolation {
		t.Fatalf("err = %v, exit = %d", err, ExitCode(err))
	}
}

func TestParseManifestHandlesCompanionResources(t *testing.T) {
	dgd := "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\nspec: {}\n"
	cm := "apiVersion: v1\nkind: ConfigMap\nmetadata: {}\n"
	for name, manifest := range map[string]string{
		"dgd first":       dgd + "---\n" + cm,
		"configmap first": "# generated\n---\n" + cm + "---\n" + dgd,
	} {
		got, err := ParseManifest(manifest)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got.DGD["kind"] != "DynamoGraphDeployment" || len(got.Companions) != 1 || !strings.Contains(got.Companions[0], "ConfigMap") {
			t.Fatalf("%s: %+v", name, got)
		}
	}
	for name, manifest := range map[string]string{
		"two dgds":               dgd + "---\n" + dgd,
		"no dgd":                 cm,
		"no spec":                "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n",
		"no apiVersion":          "kind: DynamoGraphDeployment\nspec: {}\n",
		"unsupported apiVersion": "apiVersion: nvidia.com/v9\nkind: DynamoGraphDeployment\nspec: {}\n",
	} {
		if _, err := ParseManifest(manifest); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
}

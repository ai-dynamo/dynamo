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
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testNamespace = "ns"
	testRunName   = "run"
	testPodName   = "pod-1"
)

// memWriter is the reconciler's writer for tests. It logs every write attempt (a failed
// attempt is logged with a trailing "!") and, unless lag is set, applies the write to
// the fake client the reconciler reads from, which stands in for the informer cache.
// With lag set, writes are logged but never become visible to the reconciler, like a
// cache that has not caught up; the server map still records them.
type memWriter struct {
	t      *testing.T
	c      client.Client
	ops    []string
	status []RunStatus
	// statusVersions and populateVersions record the resourceVersion each write was
	// constrained to.
	statusVersions   []string
	populateVersions map[string]string
	lag              bool
	// server is what the API server holds, whatever the cache shows: the candidates that
	// exist, and whether each has its status.
	server map[string]bool
	// failOn makes the named operation ("create", "populate", "status", "delete") fail;
	// once, unless sticky.
	failOn  map[string]error
	sticky  bool
	invalid map[string]bool
}

func (w *memWriter) fail(op string) error {
	err, ok := w.failOn[op]
	if !ok {
		return nil
	}
	if !w.sticky {
		delete(w.failOn, op)
	}
	return err
}

func (w *memWriter) record(op string, err error) error {
	if err != nil {
		op += "!"
	}
	w.ops = append(w.ops, op)
	return err
}

func (w *memWriter) CreateCandidate(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, name string, candidate dgdcreconcile.DesiredCandidate) (StoredCandidate, error) {
	if err := w.record("create:"+name, w.fail("create")); err != nil {
		return StoredCandidate{}, err
	}
	// The apply is idempotent; the read-back reports what the server holds.
	populated := w.server[name]
	w.server[name] = populated
	stored := StoredCandidate{UID: types.UID("uid-" + name), ResourceVersion: createdVersion(name), Populated: populated}
	if w.lag {
		return stored, nil
	}
	object := &v1beta2.DynamoGraphDeploymentCandidate{
		Namespace:       testNamespace,
		Name:            name,
		UID:             types.UID("uid-" + name),
		Labels:          map[string]string{LabelRunUID: testRunUID, LabelCandidateID: candidate.ID},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(run, v1beta2.DynamoGraphDeploymentRunGVK)},
		Spec:            runtime.RawExtension{Raw: []byte(`{}`)},
	}
	return stored, client.IgnoreAlreadyExists(w.c.Create(ctx, object))
}

// createdVersion is the resourceVersion the fake API server "returns" for a new candidate.
func createdVersion(name string) string { return "created-" + name }

func (w *memWriter) PrepareCandidate(_ context.Context, _ *v1beta2.DynamoGraphDeploymentRun, candidate dgdcreconcile.DesiredCandidate) (dgdcreconcile.DesiredCandidate, error) {
	if err := w.fail("prepare:" + candidate.ID); err != nil {
		return candidate, err
	}
	candidate.Validity = metav1.Condition{Type: v1beta2.DGDCConditionValid, Status: metav1.ConditionTrue, Reason: "ValidationSucceeded", LastTransitionTime: metav1.Now()}
	if w.invalid[candidate.ID] {
		candidate.Validity.Status = metav1.ConditionFalse
		candidate.Validity.Reason = "OverrideTargetNotFound"
		candidate.Validity.Message = "override target is absent"
	}
	return candidate, nil
}

func TestInvalidCandidatesRemainRankedAndCountTowardTheLimit(t *testing.T) {
	for _, allInvalid := range []bool{false, true} {
		t.Run(fmt.Sprintf("allInvalid=%t", allInvalid), func(t *testing.T) {
			t.Log("Reconcile a terminal ranked result containing invalid Candidates")
			f := newFixture(t)
			f.w.invalid = map[string]bool{"a": allInvalid, "b": true}
			f.publish(snapshotJSON(t, PhaseSucceeded, cand{id: "a"}, cand{id: "b"}))
			f.mustReconcile()
			f.mustReconcile()

			t.Log("Keep every selected Candidate in search order, including invalid ones")
			status := f.lastStatus()
			if status.Published != 2 || !reflect.DeepEqual(status.CandidateNames, []string{cn("a"), cn("b")}) {
				t.Fatalf("published result = %+v", status)
			}
			if !f.r.Completion.Finished() || errors.Is(f.r.Completion.Err(), ErrRunFailed) != allInvalid {
				t.Fatalf("completion = %v, want Run failure only when all Candidates are invalid", f.r.Completion.Err())
			}
			var invalid v1beta2.DynamoGraphDeploymentCandidate
			if err := f.c.Get(t.Context(), types.NamespacedName{Namespace: testNamespace, Name: cn("b")}, &invalid); err != nil {
				t.Fatal(err)
			}
			if len(invalid.Status.Conditions) != 1 || invalid.Status.Conditions[0].Status != metav1.ConditionFalse {
				t.Fatalf("invalid Candidate was lost: %+v", invalid.Status)
			}
		})
	}
}

func TestPreparationCacheTracksTheProjectionAcrossRetries(t *testing.T) {
	f := newFixture(t)
	run := f.currentRun()
	run.Spec.Recommendation.MaxCandidates = 2
	transient := errors.New("temporary dry-run failure")
	tests := []struct {
		name       string
		ids        []string
		failID     string
		cachedID   string
		wantCached []string
	}{
		{name: "initial successful projection", ids: []string{"a", "b"}, wantCached: []string{"a", "b"}},
		{name: "early failure preserves a later live entry", ids: []string{"c", "b"}, failID: "c", wantCached: []string{"b"}},
		{name: "late failure removes the previous projection", ids: []string{"c", "d"}, failID: "d", wantCached: []string{"c"}},
		{name: "repeated failure cannot accumulate old entries", ids: []string{"e", "f"}, failID: "f", wantCached: []string{"e"}},
		{name: "retry reuses successful preparation", ids: []string{"e", "f"}, cachedID: "e", wantCached: []string{"e", "f"}},
		{name: "empty projection clears the cache"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Prepare the next bounded projection, with a transient failure where requested")
			var candidates []cand
			for _, id := range test.ids {
				candidates = append(candidates, cand{id: id})
			}
			snapshot, err := ParseSnapshot(snapshotJSON(t, PhaseRunning, candidates...))
			if err != nil {
				t.Fatal(err)
			}
			if test.failID != "" {
				f.w.failOn["prepare:"+test.failID] = transient
			}
			if test.cachedID != "" {
				f.w.failOn["prepare:"+test.cachedID] = errors.New("cached Candidate must not be prepared again")
			}
			desired := Desire(run.Name, string(run.UID), snapshot)
			err = f.r.prepareCandidates(t.Context(), run, &desired)
			if (test.failID != "" && !errors.Is(err, transient)) || (test.failID == "" && err != nil) {
				t.Fatalf("prepareCandidates: %v, expected failure on %q", err, test.failID)
			}

			t.Log("Retain only successfully prepared Candidates in the current projection")
			if len(f.r.preparedCandidates) != len(test.wantCached) {
				t.Fatalf("cache has %d entries, want IDs %v", len(f.r.preparedCandidates), test.wantCached)
			}
			for _, id := range test.wantCached {
				if _, found := f.r.preparedCandidates[id]; !found {
					t.Errorf("cache lost current Candidate %q", id)
				}
			}
		})
	}
}

func (w *memWriter) PopulateStatus(ctx context.Context, name, resourceVersion string, candidate dgdcreconcile.DesiredCandidate) error {
	if err := w.record("populate:"+name, w.fail("populate")); err != nil {
		return err
	}
	w.populateVersions[name] = resourceVersion
	w.server[name] = true
	if w.lag {
		return nil
	}
	var object v1beta2.DynamoGraphDeploymentCandidate
	if err := w.c.Get(ctx, types.NamespacedName{Namespace: testNamespace, Name: name}, &object); err != nil {
		return err
	}
	object.Status.Metrics = &runtime.RawExtension{Raw: []byte("{}")}
	object.Status.Parameters = &runtime.RawExtension{Raw: []byte("{}")}
	object.Status.Conditions = []metav1.Condition{candidate.Validity}
	return w.c.Update(ctx, &object)
}

func (w *memWriter) UpdateRunStatus(ctx context.Context, resourceVersion string, status RunStatus) error {
	if err := w.record("status", w.fail("status")); err != nil {
		return err
	}
	w.status = append(w.status, status)
	w.statusVersions = append(w.statusVersions, resourceVersion)
	if w.lag {
		return nil
	}
	var run v1beta2.DynamoGraphDeploymentRun
	if err := w.c.Get(ctx, types.NamespacedName{Namespace: testNamespace, Name: testRunName}, &run); err != nil {
		return err
	}
	run.Status.Progress = &v1beta2.DGDRRunProgress{Candidates: &v1beta2.CandidateProgress{Published: status.Published}}
	if !status.LastProgress.IsZero() {
		progressed := metav1.NewTime(status.LastProgress)
		run.Status.LastProgressTime = &progressed
	}
	run.Status.CandidateRefs = nil
	for _, name := range status.CandidateNames {
		run.Status.CandidateRefs = append(run.Status.CandidateRefs, corev1.LocalObjectReference{Name: name})
	}
	return w.c.Update(ctx, &run)
}

func (w *memWriter) DeleteCandidate(ctx context.Context, name string, _ types.UID) error {
	if err := w.record("delete:"+name, w.fail("delete")); err != nil {
		return err
	}
	if w.lag {
		return nil
	}
	return client.IgnoreNotFound(w.c.Delete(ctx, &v1beta2.DynamoGraphDeploymentCandidate{
		Namespace: testNamespace,
		Name:      name,
	}))
}

type fixture struct {
	t *testing.T
	r *Reconciler
	w *memWriter
	c client.Client
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	scheme, err := NewScheme()
	if err != nil {
		t.Fatal(err)
	}
	run := &v1beta2.DynamoGraphDeploymentRun{Namespace: testNamespace, Name: testRunName, UID: testRunUID}
	run.Spec.Recommendation = &v1beta2.RecommendationSpec{MaxCandidates: 100}
	pod := &corev1.Pod{Namespace: testNamespace, Name: testPodName}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(run, pod).Build()
	store := &InputStore{}
	observer := &Observer{
		Store:       store,
		Path:        filepath.Join(t.TempDir(), SnapshotFileName),
		MinInterval: time.Millisecond,
		MaxInterval: time.Millisecond,
	}
	w := &memWriter{t: t, c: c, failOn: map[string]error{}, populateVersions: map[string]string{}, server: map[string]bool{}}
	return &fixture{t: t, c: c, w: w, r: &Reconciler{
		Client:           c,
		Writer:           w,
		Namespace:        testNamespace,
		RunName:          testRunName,
		RunUID:           testRunUID,
		PodName:          testPodName,
		SweeperContainer: "sweeper",
		Observer:         observer,
		Store:            store,
		Completion:       NewCompletion(),
	}}
}

// publish writes a snapshot the way the Sweeper does and lets the observer see it.
func (f *fixture) publish(data []byte) {
	f.t.Helper()
	writeSnapshot(f.t, filepath.Dir(f.r.Observer.Path), data)
	f.r.Observer.Check()
}

func (f *fixture) reconcile() error {
	f.t.Helper()
	_, err := f.r.Reconcile(context.Background(), ctrl.Request{Namespace: testNamespace, Name: testRunName})
	return err
}

func (f *fixture) mustReconcile() {
	f.t.Helper()
	if err := f.reconcile(); err != nil {
		f.t.Fatalf("reconcile: %v", err)
	}
}

// exitSweeper reports the Sweeper container as terminated, as the pod status does.
func (f *fixture) exitSweeper(code int32) {
	f.t.Helper()
	var pod corev1.Pod
	if err := f.c.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: testPodName}, &pod); err != nil {
		f.t.Fatal(err)
	}
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "sweeper", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: code}}}}
	if err := f.c.Status().Update(context.Background(), &pod); err != nil {
		f.t.Fatal(err)
	}
}

// deleteCandidate removes a candidate behind the publisher's back, from the API server
// and from the cache.
func (f *fixture) deleteCandidate(name string) {
	f.t.Helper()
	delete(f.w.server, name)
	object := &v1beta2.DynamoGraphDeploymentCandidate{Namespace: testNamespace, Name: name}
	if err := f.c.Delete(context.Background(), object); err != nil {
		f.t.Fatal(err)
	}
}

func (f *fixture) takeOps() []string {
	ops := f.w.ops
	f.w.ops = nil
	return ops
}

func (f *fixture) lastStatus() RunStatus {
	f.t.Helper()
	if len(f.w.status) == 0 {
		f.t.Fatal("no run status was written")
	}
	return f.w.status[len(f.w.status)-1]
}

func (f *fixture) currentRun() *v1beta2.DynamoGraphDeploymentRun {
	f.t.Helper()
	var run v1beta2.DynamoGraphDeploymentRun
	if err := f.c.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: testRunName}, &run); err != nil {
		f.t.Fatal(err)
	}
	return &run
}

func candidateForRun(name, id string, run *v1beta2.DynamoGraphDeploymentRun) *v1beta2.DynamoGraphDeploymentCandidate {
	return &v1beta2.DynamoGraphDeploymentCandidate{
		Namespace:       run.Namespace,
		Name:            name,
		Labels:          map[string]string{LabelRunUID: string(run.UID), LabelCandidateID: id},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(run, v1beta2.DynamoGraphDeploymentRunGVK)},
	}
}

// withProgress rewrites a snapshot's progress counters.
func withProgress(t *testing.T, data []byte, round, evaluated int) []byte {
	t.Helper()
	doc := map[string]any{}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	doc["progress"] = map[string]any{"round": round, "evaluated": evaluated}
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestReconcileCreatesPopulatesAndReferencesCandidatesInOrder(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}))
	f.mustReconcile()
	want := []string{"create:" + cn("a"), "create:" + cn("b"), "populate:" + cn("a"), "populate:" + cn("b"), "status"}
	if got := f.takeOps(); !reflect.DeepEqual(got, want) {
		t.Fatalf("ops = %v, want %v", got, want)
	}
	if got, want := f.lastStatus().CandidateNames, []string{cn("a"), cn("b")}; !reflect.DeepEqual(got, want) {
		t.Fatalf("refs = %v, want %v", got, want)
	}
}

func TestSwapOnlyReordersTheReferences(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}))
	f.mustReconcile()
	f.takeOps()
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "b"}, cand{id: "a"}))
	f.mustReconcile()
	if got := f.takeOps(); !reflect.DeepEqual(got, []string{"status"}) {
		t.Fatalf("a swap must not touch any candidate, ops = %v", got)
	}
	if got, want := f.lastStatus().CandidateNames, []string{cn("b"), cn("a")}; !reflect.DeepEqual(got, want) {
		t.Fatalf("refs = %v, want %v", got, want)
	}
}

func TestReconcilingUnchangedStateWritesNothing(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	f.takeOps()
	f.mustReconcile()
	f.mustReconcile()
	if got := f.takeOps(); len(got) != 0 {
		t.Fatalf("unchanged state must cause no writes, got %v", got)
	}
}

func TestCacheLagRepeatsOnlyIdempotentWrites(t *testing.T) {
	f := newFixture(t)
	f.w.lag = true // the cache never shows what was written
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	if got, want := f.takeOps(), []string{"create:" + cn("a"), "populate:" + cn("a"), "status"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("first reconcile ops = %v, want %v", got, want)
	}
	f.mustReconcile()
	// The candidate apply repeats harmlessly and the server reports the candidate populated,
	// so its status is not written a second time. The run status is written again, still
	// constrained to the cached version: against a real API server the first write has
	// moved the run on, so this one is rejected as stale and retried through the workqueue.
	if got, want := f.takeOps(), []string{"create:" + cn("a"), "status"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second reconcile ops = %v, want %v", got, want)
	}
	if len(f.w.statusVersions) != 2 || f.w.statusVersions[0] != f.w.statusVersions[1] {
		t.Fatalf("both status writes must carry the cached run version, got %v", f.w.statusVersions)
	}
}

func TestExternallyDeletedCandidateIsRecreated(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	f.takeOps()
	f.deleteCandidate(cn("a"))
	f.mustReconcile()
	if got, want := f.takeOps(), []string{"create:" + cn("a"), "populate:" + cn("a")}; !reflect.DeepEqual(got, want) {
		t.Fatalf("a candidate deleted behind the publisher's back must be repaired, ops = %v, want %v", got, want)
	}
}

func TestExternallyModifiedRunStatusIsRepaired(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	f.takeOps()
	var run v1beta2.DynamoGraphDeploymentRun
	if err := f.c.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: testRunName}, &run); err != nil {
		t.Fatal(err)
	}
	run.Status.CandidateRefs = []corev1.LocalObjectReference{{Name: "changed-by-someone-else"}}
	if err := f.c.Update(context.Background(), &run); err != nil {
		t.Fatal(err)
	}
	f.mustReconcile()
	if got, want := f.takeOps(), []string{"status"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("a run status modified behind the publisher's back must be repaired, ops = %v, want %v", got, want)
	}
}

func TestPartialSuccessRetryRepeatsOnlyTheUnfinishedWrites(t *testing.T) {
	f := newFixture(t)
	f.w.failOn["status"] = errors.New("api unavailable")
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	if err := f.reconcile(); err == nil {
		t.Fatal("a failed run status write must be returned for the workqueue to retry with backoff")
	}
	f.mustReconcile() // the retry, possibly triggered early by the echo of the candidate create
	want := []string{"create:" + cn("a"), "populate:" + cn("a"), "status!", "status"}
	if got := f.takeOps(); !reflect.DeepEqual(got, want) {
		t.Fatalf("ops = %v, want %v", got, want)
	}
}

func TestCreateFailureSkipsTheRunStatusAndIsRetried(t *testing.T) {
	f := newFixture(t)
	f.w.failOn["create"] = errors.New("boom")
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	if err := f.reconcile(); err == nil {
		t.Fatal("want the create failure returned")
	}
	for _, op := range f.takeOps() {
		if op == "status" {
			t.Fatal("the run must not reference a candidate that was not created")
		}
	}
	f.mustReconcile()
	if got := f.takeOps(); got[len(got)-1] != "status" {
		t.Fatalf("ops after retry = %v", got)
	}
}

func TestDroppedCandidateIsDeletedAfterTheRunStatusUpdate(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}))
	f.mustReconcile()
	f.takeOps()
	f.publish(withProgress(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), 3, 9))
	f.mustReconcile()
	if got, want := f.takeOps(), []string{"status", "delete:" + cn("b")}; !reflect.DeepEqual(got, want) {
		t.Fatalf("ops = %v, want %v", got, want)
	}
}

func TestCandidateWithoutStatusIsCompletedNotRecreated(t *testing.T) {
	f := newFixture(t)
	existing := candidateForRun(cn("a"), "a", f.currentRun())
	if err := f.c.Create(context.Background(), existing); err != nil {
		t.Fatal(err)
	}
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	if got, want := f.takeOps(), []string{"populate:" + cn("a"), "status"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("ops = %v, want %v", got, want)
	}
}

func TestProgressTimeSurvivesASnapshotWithoutATimestamp(t *testing.T) {
	f := newFixture(t)
	first := time.Date(2026, 10, 5, 21, 35, 12, 0, time.UTC)
	f.publish(withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), "2026-10-05T21:35:12Z"))
	f.mustReconcile()
	f.publish(withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}), ""))
	f.mustReconcile()
	if len(f.w.status) != 2 {
		t.Fatalf("want two status writes, got %d", len(f.w.status))
	}
	// The preserved time comes from the cached run, the current observed state; a stale
	// cache would lose the version check instead of the timestamp.
	if got := f.lastStatus().LastProgress; !got.Equal(first) {
		t.Fatalf("the second write must repeat the observed progress time %v, got %v", first, got)
	}
}

func TestTerminalSucceededSnapshotIsReconciledThenAcknowledged(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseSucceeded, cand{id: "a"}))
	f.mustReconcile()
	if f.r.Completion.Finished() {
		t.Fatal("the publisher must wait for its candidate and run-status writes to reach the cache")
	}
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
	if len(f.w.status) != 1 {
		t.Fatal("the terminal snapshot must be reconciled before the publisher exits")
	}
}

func TestTerminalCompletionWaitsForWritesFromThePreviousSnapshot(t *testing.T) {
	f := newFixture(t)
	f.w.lag = true
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	f.takeOps()

	// The terminal snapshot arrives before the cache sees the candidate and Run status
	// created for the previous snapshot. Their absence is not convergence: the API writes
	// have succeeded and their watch echoes are still in flight.
	f.publish(snapshotJSON(t, PhaseSucceeded, cand{id: "b"}))
	f.mustReconcile()
	if f.r.Completion.Finished() {
		t.Fatal("terminal completion raced writes that were not observed through the cache")
	}

	// Deliver those echoes. The terminal projection then removes the now-visible
	// candidate and status reference, and waits for those writes in turn.
	run := f.currentRun()
	run.Status.Progress = &v1beta2.DGDRRunProgress{Candidates: &v1beta2.CandidateProgress{Published: 1}}
	run.Status.CandidateRefs = []corev1.LocalObjectReference{{Name: cn("a")}}
	if err := f.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	candidate := candidateForRun(cn("a"), "a", run)
	candidate.UID = types.UID("uid-" + cn("a"))
	candidate.Status.Metrics = &runtime.RawExtension{Raw: []byte("{}")}
	candidate.Spec = runtime.RawExtension{Raw: []byte(`{}`)}
	candidate.Status.Parameters = &runtime.RawExtension{Raw: []byte(`{}`)}
	candidate.Status.Conditions = []metav1.Condition{{Type: v1beta2.DGDCConditionValid, Status: metav1.ConditionTrue}}
	if err := f.c.Create(context.Background(), candidate); err != nil {
		t.Fatal(err)
	}
	f.w.server[cn("b")] = false
	f.w.lag = false
	f.mustReconcile()
	if f.r.Completion.Finished() {
		t.Fatal("terminal completion raced the candidate deletion and final Run status")
	}

	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
}

func TestTerminalFailedSnapshotIsReconciledThenFails(t *testing.T) {
	f := newFixture(t)
	f.publish(withRunDiagnostics(t, snapshotJSON(t, PhaseFailed, cand{id: "a"}), "search failed", "backend unavailable"))
	f.mustReconcile()
	f.mustReconcile()
	if err := f.r.Completion.Err(); !errors.Is(err, ErrRunFailed) || ExitCode(err) != ExitRunFailed ||
		!strings.Contains(err.Error(), "search failed") || !strings.Contains(err.Error(), "backend unavailable") {
		t.Fatalf("err = %v", err)
	}
	if len(f.w.status) != 1 {
		t.Fatal("the failed snapshot must still be reconciled")
	}
}

func TestTerminalSucceededSnapshotWithoutAMaterializedCandidateFails(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseSucceeded, cand{id: "x", failed: true}))
	f.mustReconcile()

	err := f.r.Completion.Err()
	var failure *RunFailedError
	if !errors.Is(err, ErrRunFailed) || !errors.As(err, &failure) || ExitCode(err) != ExitRunFailed ||
		!strings.Contains(err.Error(), "1 candidate failure") || !strings.HasPrefix(failure.Outcome.Error, "x: ") {
		t.Fatalf("err = %v", err)
	}
	if ops := f.takeOps(); len(ops) != 0 {
		t.Fatalf("a failed materialization must not produce cluster writes, got %v", ops)
	}
}

func TestTerminalSnapshotReconcileFailureIsRetriedUntilItSucceeds(t *testing.T) {
	f := newFixture(t)
	f.w.failOn["status"] = errors.New("api unavailable")
	f.publish(snapshotJSON(t, PhaseSucceeded, cand{id: "a"}))
	if err := f.reconcile(); err == nil || f.r.Completion.Finished() {
		t.Fatalf("a terminal snapshot that could not be reconciled must be retried, err=%v", err)
	}
	f.mustReconcile()
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
}

func TestSweeperCrashWithoutTerminalSnapshotReconcilesTheLastSnapshot(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.exitSweeper(137)
	f.mustReconcile()
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("the Sweeper's own exit code fails the Job, not the publisher: finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
	if len(f.w.status) != 1 {
		t.Fatal("the last snapshot must be reconciled before exit")
	}
}

func TestSweeperCleanExitWithoutTerminalSnapshotIsAProtocolViolation(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.exitSweeper(0)
	f.mustReconcile()
	f.mustReconcile()
	if err := f.r.Completion.Err(); !errors.Is(err, ErrMissingTerminalSnapshot) || ExitCode(err) != ExitProtocolViolation {
		t.Fatalf("err = %v", err)
	}
	if len(f.w.status) != 1 {
		t.Fatal("the last snapshot must be reconciled before exit")
	}
}

func TestSweeperExitBeforeAnySnapshot(t *testing.T) {
	crash := newFixture(t)
	crash.exitSweeper(1)
	crash.mustReconcile()
	if !crash.r.Completion.Finished() || crash.r.Completion.Err() != nil {
		t.Fatalf("crash: finished=%v err=%v", crash.r.Completion.Finished(), crash.r.Completion.Err())
	}
	clean := newFixture(t)
	clean.exitSweeper(0)
	clean.mustReconcile()
	if err := clean.r.Completion.Err(); !errors.Is(err, ErrMissingTerminalSnapshot) {
		t.Fatalf("clean exit with no snapshot at all: err = %v", err)
	}
}

func TestSnapshotWrittenJustBeforeTheSweeperExitIsReadAfterTheExitIsSeen(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	// The Sweeper writes its terminal snapshot and exits; the observer has not looked yet.
	writeSnapshot(t, filepath.Dir(f.r.Observer.Path), snapshotJSON(t, PhaseSucceeded, cand{id: "a"}, cand{id: "b"}))
	f.exitSweeper(0)
	f.mustReconcile()
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("the terminal snapshot must be picked up on exit: finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
	if got := f.lastStatus().CandidateNames; len(got) != 2 {
		t.Fatalf("refs = %v", got)
	}
}

func TestInvalidFinalSnapshotIsAProtocolViolationExit(t *testing.T) {
	f := newFixture(t)
	writeSnapshot(t, filepath.Dir(f.r.Observer.Path), []byte(`{"schemaVersion":2}`))
	f.exitSweeper(0)
	f.mustReconcile()
	if err := f.r.Completion.Err(); ExitCode(err) != ExitProtocolViolation {
		t.Fatalf("err = %v, exit = %d", err, ExitCode(err))
	}
}

func TestFinalSnapshotReadFailureEndsThePublisher(t *testing.T) {
	f := newFixture(t)
	f.r.Store.fail(errors.New("permission denied"))
	f.exitSweeper(1)

	f.mustReconcile()

	err := f.r.Completion.Err()
	if err == nil || ExitCode(err) != ExitReconcileFailed || !strings.Contains(err.Error(), "reading last snapshot") {
		t.Fatalf("err = %v, want terminal snapshot read failure", err)
	}
}

func TestRegressingSnapshotIsNotReconciled(t *testing.T) {
	f := newFixture(t)
	f.publish(withProgress(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), 3, 5))
	f.mustReconcile()
	f.takeOps()
	f.publish(withProgress(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}), 2, 99))
	f.mustReconcile()
	if got := f.takeOps(); len(got) != 0 {
		t.Fatalf("a regressing snapshot must not be reconciled, got %v", got)
	}
}

func TestFinalReconcileRetriesUntilTheAPIRecovers(t *testing.T) {
	f := newFixture(t)
	f.w.failOn["status"] = errors.New("api unavailable")
	f.w.sticky = true
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.exitSweeper(1)
	for attempt := 1; attempt <= 5; attempt++ {
		if err := f.reconcile(); err == nil || f.r.Completion.Finished() {
			t.Fatalf("attempt %d: want a retry, err=%v", attempt, err)
		}
	}
	delete(f.w.failOn, "status")
	f.mustReconcile()
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
}

func TestForeignLabeledCandidateDoesNotSatisfyOrGetDeleted(t *testing.T) {
	f := newFixture(t)
	foreignRun := &v1beta2.DynamoGraphDeploymentRun{
		Namespace: testNamespace,
		Name:      "foreign-run",
		UID:       "foreign-run-uid",
	}
	foreign := candidateForRun("foreign-candidate", "a", foreignRun)
	foreign.Labels[LabelRunUID] = testRunUID
	if err := f.c.Create(context.Background(), foreign); err != nil {
		t.Fatal(err)
	}

	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	if got, want := f.takeOps(), []string{"create:" + cn("a"), "populate:" + cn("a"), "status"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("ops = %v, want %v", got, want)
	}
	if err := f.c.Get(context.Background(), client.ObjectKeyFromObject(foreign), &v1beta2.DynamoGraphDeploymentCandidate{}); err != nil {
		t.Fatalf("foreign candidate must remain untouched: %v", err)
	}
}

func TestOwnedCandidateDriftIsReconciledSafely(t *testing.T) {
	tests := []struct {
		name           string
		existingName   string
		existingID     string
		removeRunLabel bool
		desired        []cand
		want           []string
	}{
		{
			name:         "wrong name is replaced before deletion",
			existingName: "wrong-name",
			existingID:   "a",
			desired:      []cand{{id: "a"}},
			want:         []string{"create:" + cn("a"), "populate:" + cn("a"), "status", "delete:wrong-name"},
		},
		{
			name:           "obsolete candidate without run label is deleted",
			existingName:   cn("obsolete"),
			existingID:     "obsolete",
			removeRunLabel: true,
			want:           []string{"delete:" + cn("obsolete")},
		},
		{
			name:         "wrong identity label is reapplied",
			existingName: cn("a"),
			existingID:   "wrong-id",
			desired:      []cand{{id: "a"}},
			want:         []string{"create:" + cn("a"), "populate:" + cn("a"), "status"},
		},
		{
			name:           "missing run label is reapplied",
			existingName:   cn("a"),
			existingID:     "a",
			removeRunLabel: true,
			desired:        []cand{{id: "a"}},
			want:           []string{"create:" + cn("a"), "populate:" + cn("a"), "status"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			f := newFixture(t)
			existing := candidateForRun(test.existingName, test.existingID, f.currentRun())
			if test.removeRunLabel {
				delete(existing.Labels, LabelRunUID)
			}
			if err := f.c.Create(context.Background(), existing); err != nil {
				t.Fatal(err)
			}

			f.publish(snapshotJSON(t, PhaseRunning, test.desired...))
			f.mustReconcile()
			if got := f.takeOps(); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("ops = %v, want %v", got, test.want)
			}
		})
	}
}

func TestRunRecreatedUnderTheSameNameIsNotTouched(t *testing.T) {
	f := newFixture(t)
	var run v1beta2.DynamoGraphDeploymentRun
	if err := f.c.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: testRunName}, &run); err != nil {
		t.Fatal(err)
	}
	run.UID = "another-uid"
	if err := f.c.Update(context.Background(), &run); err != nil {
		t.Fatal(err)
	}
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() == nil {
		t.Fatal("a publisher serving a previous incarnation must stop")
	}
	if len(f.w.ops) != 0 {
		t.Fatalf("it must not write anything: %v", f.w.ops)
	}
}

func TestReconcileIgnoresOtherKeysAndAFinishedRun(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseSucceeded, cand{id: "a"}))
	if _, err := f.r.Reconcile(context.Background(), ctrl.Request{Namespace: testNamespace, Name: "other"}); err != nil || len(f.w.ops) != 0 {
		t.Fatalf("another key must be ignored: err=%v ops=%v", err, f.w.ops)
	}
	f.mustReconcile()
	f.mustReconcile()
	ops := len(f.w.ops)
	f.mustReconcile()
	if len(f.w.ops) != ops {
		t.Fatal("a finished run must not be reconciled again")
	}
}

func podWith(statuses ...corev1.ContainerStatus) *corev1.Pod {
	return &corev1.Pod{Status: corev1.PodStatus{ContainerStatuses: statuses}}
}

func TestSweeperStateOfReadsOnlyTheSweeperContainer(t *testing.T) {
	terminated := func(name string, code int32) corev1.ContainerStatus {
		return corev1.ContainerStatus{Name: name, State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: code}}}
	}
	running := corev1.ContainerStatus{Name: "sweeper", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}}

	cases := map[string]struct {
		pod  *corev1.Pod
		want SweeperState
	}{
		"no statuses yet":      {podWith(), SweeperState{}},
		"running":              {podWith(running), SweeperState{}},
		"exited non-zero":      {podWith(terminated("sweeper", 3)), SweeperState{Exited: true, ExitCode: 3}},
		"exited zero":          {podWith(terminated("sweeper", 0)), SweeperState{Exited: true}},
		"other container only": {podWith(terminated("publisher", 1)), SweeperState{}},
	}
	for name, tc := range cases {
		if got := sweeperStateOf(tc.pod, "sweeper"); got != tc.want {
			t.Errorf("%s: got %+v, want %+v", name, got, tc.want)
		}
	}
}

func TestObservedStatusMirrorsTheRunStatus(t *testing.T) {
	when := metav1.NewTime(time.Date(2026, 10, 5, 21, 35, 12, 0, time.UTC))
	run := &v1beta2.DynamoGraphDeploymentRun{Status: v1beta2.DynamoGraphDeploymentRunStatus{
		Progress:         &v1beta2.DGDRRunProgress{Candidates: &v1beta2.CandidateProgress{Published: 2}},
		LastProgressTime: &when,
		CandidateRefs:    []corev1.LocalObjectReference{{Name: "x"}, {Name: "y"}},
	}}
	want := RunStatus{Published: 2, LastProgress: when.Time, CandidateNames: []string{"x", "y"}}
	if got := observedStatus(run); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	if got := observedStatus(&v1beta2.DynamoGraphDeploymentRun{}); got.Published != 0 || !got.LastProgress.IsZero() {
		t.Fatalf("an empty status must not look like reported progress: %+v", got)
	}
}

func TestCacheOptionsCoverThisRunPodAndNamespaceCandidates(t *testing.T) {
	cfg := Config{Namespace: "ns", RunName: "run", RunUID: "uid-1", PodName: "pod-1"}
	opts := cacheOptions(cfg)
	if _, ok := opts.DefaultNamespaces["ns"]; !ok || len(opts.DefaultNamespaces) != 1 {
		t.Fatalf("cache must be limited to the run's namespace: %v", opts.DefaultNamespaces)
	}
	if len(opts.ByObject) != 3 {
		t.Fatalf("want selectors for the run, its candidates and the pod, got %d", len(opts.ByObject))
	}
	for object, selector := range opts.ByObject {
		switch object.(type) {
		case *v1beta2.DynamoGraphDeploymentRun:
			if selector.Field.String() != "metadata.name=run" {
				t.Errorf("run selector = %q", selector.Field)
			}
		case *v1beta2.DynamoGraphDeploymentCandidate:
			if selector.Field != nil || selector.Label != nil {
				t.Errorf("candidates must remain visible after mutable label drift: %+v", selector)
			}
		case *corev1.Pod:
			if selector.Field.String() != "metadata.name=pod-1" {
				t.Errorf("pod selector = %q", selector.Field)
			}
		default:
			t.Errorf("unexpected cached type %T", object)
		}
	}
}

func TestRunStatusWriteIsConstrainedToTheObservedRunVersion(t *testing.T) {
	f := newFixture(t)
	var run v1beta2.DynamoGraphDeploymentRun
	if err := f.c.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: testRunName}, &run); err != nil {
		t.Fatal(err)
	}
	if run.ResourceVersion == "" {
		t.Fatal("the fake client should assign a resourceVersion")
	}
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	if len(f.w.statusVersions) != 1 || f.w.statusVersions[0] != run.ResourceVersion {
		t.Fatalf("status versions = %v, want [%s]", f.w.statusVersions, run.ResourceVersion)
	}
}

func TestNewCandidateStatusUsesTheVersionReturnedByTheCreate(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	if got, want := f.w.populateVersions[cn("a")], createdVersion(cn("a")); got != want {
		t.Fatalf("populate version = %q, want the version from the create (%q)", got, want)
	}
}

func TestExistingCandidateStatusUsesItsCachedVersion(t *testing.T) {
	f := newFixture(t)
	existing := candidateForRun(cn("a"), "a", f.currentRun())
	if err := f.c.Create(context.Background(), existing); err != nil {
		t.Fatal(err)
	}
	var cached v1beta2.DynamoGraphDeploymentCandidate
	if err := f.c.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: cn("a")}, &cached); err != nil {
		t.Fatal(err)
	}
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.mustReconcile()
	if got := f.w.populateVersions[cn("a")]; got == "" || got != cached.ResourceVersion {
		t.Fatalf("populate version = %q, want the cached version %q", got, cached.ResourceVersion)
	}
}

func TestConflictRemainsRetryableUntilTheCacheCatchesUp(t *testing.T) {
	f := newFixture(t)
	f.w.failOn["status"] = apierrors.NewConflict(schema.GroupResource{Group: "nvidia.com", Resource: "dynamographdeploymentruns"}, testRunName, errors.New("stale"))
	f.w.sticky = true
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.exitSweeper(1)
	for attempt := range 6 {
		if err := f.reconcile(); !apierrors.IsConflict(err) || f.r.Completion.Finished() {
			t.Fatalf("attempt %d: a conflict from a stale read must be retried, err=%v finished=%v", attempt, err, f.r.Completion.Finished())
		}
	}
	delete(f.w.failOn, "status") // the cache caught up
	f.mustReconcile()
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
}

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
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher/placeholderapi"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
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
	failOn map[string]error
	sticky bool
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

func (w *memWriter) CreateCandidate(ctx context.Context, _ *v1beta2.DynamoGraphDeploymentRun, name string, candidate dgdcreconcile.DesiredCandidate) (StoredCandidate, error) {
	if err := w.record("create:"+name, w.fail("create")); err != nil {
		return StoredCandidate{}, err
	}
	// The apply is idempotent; the read-back reports what the server holds.
	populated := w.server[name]
	w.server[name] = populated
	stored := StoredCandidate{ResourceVersion: createdVersion(name), Populated: populated}
	if w.lag {
		return stored, nil
	}
	object := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{
		Namespace: testNamespace,
		Name:      name,
		UID:       types.UID("uid-" + name),
		Labels:    map[string]string{LabelRunUID: testRunUID, LabelCandidateID: candidate.ID},
	}}
	return stored, client.IgnoreAlreadyExists(w.c.Create(ctx, object))
}

// createdVersion is the resourceVersion the fake API server "returns" for a new candidate.
func createdVersion(name string) string { return "created-" + name }

func (w *memWriter) PopulateStatus(ctx context.Context, name, resourceVersion string, _ dgdcreconcile.DesiredCandidate) error {
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
	meta.SetStatusCondition(&object.Status.Conditions, metav1.Condition{Type: evaluatedCondition, Status: metav1.ConditionTrue, Reason: evaluatedCondition})
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
	run.Status.Message = status.Message
	run.Status.Progress = &v1beta2.RunProgress{Round: status.Round, Evaluated: status.Evaluated}
	if !status.LastProgress.IsZero() {
		progressed := metav1.NewTime(status.LastProgress)
		run.Status.LastProgressTime = &progressed
	}
	run.Status.CandidateRefs = nil
	for _, name := range status.CandidateNames {
		run.Status.CandidateRefs = append(run.Status.CandidateRefs, v1beta2.CandidateRef{Name: name})
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
	return client.IgnoreNotFound(w.c.Delete(ctx, &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: name}}))
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
	run := &v1beta2.DynamoGraphDeploymentRun{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: testRunName, UID: testRunUID}}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: testPodName}}
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
	_, err := f.r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: testRunName}})
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
	object := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: name}}
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
	run.Status.Message = "changed by someone else"
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

func TestFailedMaterializationIsNotCreatedButCounted(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "x", failed: true}))
	f.mustReconcile()
	for _, op := range f.takeOps() {
		if strings.Contains(op, "x") && strings.HasPrefix(op, "create") {
			t.Fatalf("a failed candidate must not be created: %s", op)
		}
	}
	status := f.lastStatus()
	if !reflect.DeepEqual(status.CandidateNames, []string{cn("a")}) || !strings.Contains(status.Message, "1 candidate(s) failed materialization") {
		t.Fatalf("status = %+v", status)
	}
}

func TestCandidateWithoutStatusIsCompletedNotRecreated(t *testing.T) {
	f := newFixture(t)
	existing := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{
		Namespace: testNamespace,
		Name:      cn("a"),
		Labels:    map[string]string{LabelRunUID: testRunUID, LabelCandidateID: "a"},
	}}
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
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
	if len(f.w.status) != 1 {
		t.Fatal("the terminal snapshot must be reconciled before the publisher exits")
	}
}

func TestTerminalFailedSnapshotIsReconciledThenFails(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseFailed, cand{id: "a"}))
	f.mustReconcile()
	if err := f.r.Completion.Err(); !errors.Is(err, ErrRunFailed) || ExitCode(err) != ExitRunFailed {
		t.Fatalf("err = %v", err)
	}
	if len(f.w.status) != 1 {
		t.Fatal("the failed snapshot must still be reconciled")
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
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
}

func TestSweeperCrashWithoutTerminalSnapshotReconcilesTheLastSnapshot(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.exitSweeper(137)
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

func TestFinalReconcileFailureEndsTheProcessAfterBoundedRetries(t *testing.T) {
	f := newFixture(t)
	f.w.failOn["status"] = errors.New("api unavailable")
	f.w.sticky = true
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.exitSweeper(1)
	for attempt := 1; attempt < finalReconcileAttempts; attempt++ {
		if err := f.reconcile(); err == nil || f.r.Completion.Finished() {
			t.Fatalf("attempt %d: want a retry, err=%v", attempt, err)
		}
	}
	f.mustReconcile()
	if err := f.r.Completion.Err(); err == nil || ExitCode(err) != ExitReconcileFailed {
		t.Fatalf("err = %v", err)
	}
}

func TestMalformedProjectionFailsImmediately(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	// Two live candidates claiming the same id make the diff input malformed.
	for _, name := range []string{"x-1", "x-2"} {
		object := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{
			Namespace: testNamespace, Name: name,
			Labels: map[string]string{LabelRunUID: testRunUID, LabelCandidateID: "dup"},
		}}
		if err := f.c.Create(context.Background(), object); err != nil {
			t.Fatal(err)
		}
	}
	f.mustReconcile()
	var inputErr *dgdcreconcile.DiffInputError
	if err := f.r.Completion.Err(); !errors.As(err, &inputErr) || ExitCode(err) != ExitReconcileFailed {
		t.Fatalf("err = %v", err)
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
	if _, err := f.r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: "other"}}); err != nil || len(f.w.ops) != 0 {
		t.Fatalf("another key must be ignored: err=%v ops=%v", err, f.w.ops)
	}
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
		Message:          "m",
		Progress:         &v1beta2.RunProgress{Round: 2, Evaluated: 5},
		LastProgressTime: &when,
		CandidateRefs:    []v1beta2.CandidateRef{{Name: "x"}, {Name: "y"}},
	}}
	want := RunStatus{Message: "m", HasProgress: true, Round: 2, Evaluated: 5, LastProgress: when.Time, CandidateNames: []string{"x", "y"}}
	if got := observedStatus(run); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	if got := observedStatus(&v1beta2.DynamoGraphDeploymentRun{}); got.HasProgress || !got.LastProgress.IsZero() {
		t.Fatalf("an empty status must not look like reported progress: %+v", got)
	}
}

func TestCacheOptionsCoverOnlyThisRunItsCandidatesAndItsPod(t *testing.T) {
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
			if selector.Label.String() != LabelRunUID+"=uid-1" {
				t.Errorf("candidate selector = %q", selector.Label)
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

func nested(t *testing.T, object *unstructured.Unstructured, fields ...string) (any, bool) {
	t.Helper()
	value, found, err := unstructured.NestedFieldNoCopy(object.Object, fields...)
	if err != nil {
		t.Fatal(err)
	}
	return value, found
}

func TestCandidateApplyPayload(t *testing.T) {
	scheme, err := NewScheme()
	if err != nil {
		t.Fatal(err)
	}
	run := &v1beta2.DynamoGraphDeploymentRun{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: testRunName, UID: testRunUID}}
	candidate := dgdcreconcile.DesiredCandidate{ID: "a", Spec: validManifest, Parameters: map[string]any{"tp": 2}}
	apply, err := buildCandidateApply(scheme, testNamespace, testRunUID, run, cn("a"), candidate)
	if err != nil {
		t.Fatal(err)
	}
	if apply.GetAPIVersion() != "nvidia.com/v1beta2" || apply.GetKind() != "DynamoGraphDeploymentCandidate" || apply.GetName() != cn("a") || apply.GetNamespace() != testNamespace {
		t.Fatalf("identity = %s %s %s/%s", apply.GetAPIVersion(), apply.GetKind(), apply.GetNamespace(), apply.GetName())
	}
	if _, found := nested(t, apply, "status"); found {
		t.Fatal("status goes through the status subresource")
	}
	if _, found := nested(t, apply, "metadata", "creationTimestamp"); found {
		t.Fatal("creationTimestamp must not be part of an apply payload")
	}
	if got := apply.GetLabels(); got[LabelRunUID] != testRunUID || got[LabelCandidateID] != "a" {
		t.Fatalf("labels = %v", got)
	}
	if _, found := nested(t, apply, "spec", "parameters"); !found {
		t.Fatal("the resolved parameters belong in the spec")
	}
	owners := apply.GetOwnerReferences()
	if len(owners) != 1 || owners[0].UID != testRunUID || owners[0].Controller == nil || !*owners[0].Controller {
		t.Fatalf("owner references = %+v", owners)
	}
}

func TestCandidateStatusApplyPayloadIsDeterministic(t *testing.T) {
	candidate := dgdcreconcile.DesiredCandidate{ID: "a", Metrics: map[string]any{"score": 2}}
	now := time.Date(2026, 10, 5, 21, 35, 12, 0, time.UTC)
	first, err := buildCandidateStatusApply(testNamespace, cn("a"), "17", candidate, now)
	if err != nil {
		t.Fatal(err)
	}
	second, err := buildCandidateStatusApply(testNamespace, cn("a"), "17", candidate, now)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first.Object, second.Object) {
		t.Fatal("the same input must produce the same payload")
	}
	if got, _ := nested(t, first, "metadata", "resourceVersion"); got != "17" {
		t.Fatalf("the status write must be constrained to the version it was decided on, got %v", got)
	}
	conditions, _ := nested(t, first, "status", "conditions")
	list, ok := conditions.([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("conditions = %v", conditions)
	}
	condition := list[0].(map[string]any)
	if condition["type"] != evaluatedCondition || condition["status"] != "True" || condition["lastTransitionTime"] != "2026-10-05T21:35:12Z" {
		t.Fatalf("condition = %v", condition)
	}
}

func TestRunStatusApplyPayload(t *testing.T) {
	when := time.Date(2026, 10, 5, 21, 35, 12, 0, time.UTC)
	status := RunStatus{Message: "m", HasProgress: true, Round: 2, Evaluated: 5, LastProgress: when, CandidateNames: []string{"x", "y"}}
	apply := buildRunStatusApply(testNamespace, testRunName, "42", status)
	if apply.GetKind() != "DynamoGraphDeploymentRun" || apply.GetName() != testRunName {
		t.Fatalf("identity = %s %s", apply.GetKind(), apply.GetName())
	}
	if got, _ := nested(t, apply, "metadata", "resourceVersion"); got != "42" {
		t.Fatalf("the status write must be constrained to the observed run version, got %v", got)
	}
	if _, found := nested(t, buildRunStatusApply(testNamespace, testRunName, "", status), "metadata", "resourceVersion"); found {
		t.Fatal("no version known, no constraint")
	}
	if got, _ := nested(t, apply, "status", "lastProgressTime"); got != "2026-10-05T21:35:12Z" {
		t.Fatalf("lastProgressTime = %v", got)
	}
	if got, _ := nested(t, apply, "status", "progress", "round"); got != int64(2) {
		t.Fatalf("round = %#v", got)
	}
	refs, _ := nested(t, apply, "status", "candidateRefs")
	if list, ok := refs.([]any); !ok || len(list) != 2 {
		t.Fatalf("candidateRefs = %v", refs)
	}
	// A status without a progress time leaves the field out; the reconciler is what
	// carries the known time into the status it writes.
	status.LastProgress = time.Time{}
	if _, found := nested(t, buildRunStatusApply(testNamespace, testRunName, "42", status), "status", "lastProgressTime"); found {
		t.Fatal("no time known, so none is written")
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
	existing := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{
		Namespace: testNamespace,
		Name:      cn("a"),
		Labels:    map[string]string{LabelRunUID: testRunUID, LabelCandidateID: "a"},
	}}
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

func TestConflictIsRetriedWithoutUsingUpTheFinalAttempts(t *testing.T) {
	f := newFixture(t)
	f.w.failOn["status"] = apierrors.NewConflict(schema.GroupResource{Group: "nvidia.com", Resource: "dynamographdeploymentruns"}, testRunName, errors.New("stale"))
	f.w.sticky = true
	f.publish(snapshotJSON(t, PhaseRunning, cand{id: "a"}))
	f.exitSweeper(1)
	for attempt := 0; attempt < 2*finalReconcileAttempts; attempt++ {
		if err := f.reconcile(); !apierrors.IsConflict(err) || f.r.Completion.Finished() {
			t.Fatalf("attempt %d: a conflict from a stale read must be retried, err=%v finished=%v", attempt, err, f.r.Completion.Finished())
		}
	}
	delete(f.w.failOn, "status") // the cache caught up
	f.mustReconcile()
	if !f.r.Completion.Finished() || f.r.Completion.Err() != nil {
		t.Fatalf("finished=%v err=%v", f.r.Completion.Finished(), f.r.Completion.Err())
	}
}

//go:build !clustertest

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
	"strings"
	"testing"
	"time"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// These tests send the publisher's real server-side-apply payloads to an API server that
// has the v1beta2 CRDs installed. The reconciler tests use a fake writer; this is the
// check that the payloads are accepted by the real schema, that the publisher owns what
// it writes, and that the resourceVersion constraints behave as the reconciler assumes.

type writerFixture struct {
	t      *testing.T
	ctx    context.Context
	client client.Client
	run    *v1beta2.DynamoGraphDeploymentRun
	writer *clientWriter
}

func newWriterFixture(t *testing.T) *writerFixture {
	t.Helper()
	env := operatorenv.New(operatorenv.Options{
		// The publisher has no webhooks; the environment only requires the hook to exist.
		SetupWebhooks: func(ctrl.Manager, operatorenv.WebhookSetupOptions) error { return nil },
	}).RunT(t)
	ctx := context.Background()
	c := env.Client()

	run := &v1beta2.DynamoGraphDeploymentRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: env.Namespace(), Name: "publisher-run"},
		Spec: v1beta2.DynamoGraphDeploymentRunSpec{
			NativeSearchInput: v1beta2.NativeSearchInput{Model: &v1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"}},
		},
	}
	if err := c.Create(ctx, run); err != nil {
		t.Fatalf("create run: %v", err)
	}
	return &writerFixture{
		t:      t,
		ctx:    ctx,
		client: c,
		run:    run,
		writer: &clientWriter{
			Client:    c,
			Reader:    c,
			Namespace: run.Namespace,
			RunName:   run.Name,
			RunUID:    string(run.UID),
		},
	}
}

func (f *writerFixture) name(id string) string {
	return CandidateName(f.run.Name, string(f.run.UID), id)
}

func (f *writerFixture) candidate(name string) *v1beta2.DynamoGraphDeploymentCandidate {
	f.t.Helper()
	var got v1beta2.DynamoGraphDeploymentCandidate
	if err := f.client.Get(f.ctx, types.NamespacedName{Namespace: f.run.Namespace, Name: name}, &got); err != nil {
		f.t.Fatalf("get candidate %s: %v", name, err)
	}
	return &got
}

func (f *writerFixture) currentRun() *v1beta2.DynamoGraphDeploymentRun {
	f.t.Helper()
	var got v1beta2.DynamoGraphDeploymentRun
	if err := f.client.Get(f.ctx, client.ObjectKeyFromObject(f.run), &got); err != nil {
		f.t.Fatalf("get run: %v", err)
	}
	return &got
}

func envtestCandidate() dgdcreconcile.DesiredCandidate {
	return dgdcreconcile.DesiredCandidate{
		ID:         "a",
		Spec:       validManifest + "---\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: companion\n",
		Parameters: map[string]any{"tp": 2},
		Metrics:    map[string]any{"throughput": 12.5},
	}
}

func hasApplyManager(fields []metav1.ManagedFieldsEntry, subresource string) bool {
	for _, entry := range fields {
		if entry.Manager == fieldOwner && entry.Operation == metav1.ManagedFieldsOperationApply && entry.Subresource == subresource {
			return true
		}
	}
	return false
}

func TestEnvtestCandidateApplyIsAcceptedAndIdempotent(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")

	stored, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate())
	if err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	if stored.ResourceVersion == "" || stored.Populated {
		t.Fatalf("stored = %+v, want a resourceVersion and no status yet", stored)
	}

	got := f.candidate(name)
	if got.Labels[LabelRunUID] != string(f.run.UID) || got.Labels[LabelCandidateID] != "a" {
		t.Errorf("labels = %v", got.Labels)
	}
	if owner := metav1.GetControllerOf(got); owner == nil || owner.UID != f.run.UID || owner.Kind != "DynamoGraphDeploymentRun" {
		t.Errorf("controller owner = %+v, want the run", owner)
	}
	if !strings.Contains(got.Annotations[AnnotationAdditionalResources], "kind: ConfigMap") {
		t.Errorf("companion annotation = %q", got.Annotations[AnnotationAdditionalResources])
	}
	if !hasApplyManager(got.ManagedFields, "") {
		t.Errorf("no apply manager %q on the candidate: %+v", fieldOwner, got.ManagedFields)
	}

	again, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate())
	if err != nil {
		t.Fatalf("re-apply candidate: %v", err)
	}
	if again.ResourceVersion != stored.ResourceVersion {
		t.Errorf("re-applying the same candidate changed resourceVersion %s -> %s", stored.ResourceVersion, again.ResourceVersion)
	}
}

func TestEnvtestCandidateStatusIsWrittenOnceAndGuardedByResourceVersion(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	candidate := envtestCandidate()

	stored, err := f.writer.CreateCandidate(f.ctx, f.run, name, candidate)
	if err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	if err := f.writer.PopulateStatus(f.ctx, name, stored.ResourceVersion, candidate); err != nil {
		t.Fatalf("populate status: %v", err)
	}
	got := f.candidate(name)
	if got.Status.Metrics == nil {
		t.Fatal("status.metrics is not set after PopulateStatus")
	}
	if !hasApplyManager(got.ManagedFields, "status") {
		t.Errorf("no status apply manager %q: %+v", fieldOwner, got.ManagedFields)
	}

	// A decision made from the version before the status was written must conflict.
	err = f.writer.PopulateStatus(f.ctx, name, stored.ResourceVersion, candidate)
	if !apierrors.IsConflict(err) {
		t.Fatalf("populate with a stale resourceVersion: err = %v, want a conflict", err)
	}

	// Re-applying the candidate after its status exists reports it as populated.
	after, err := f.writer.CreateCandidate(f.ctx, f.run, name, candidate)
	if err != nil {
		t.Fatalf("re-apply candidate: %v", err)
	}
	if !after.Populated {
		t.Error("a candidate with status was reported as not populated")
	}
}

func TestEnvtestRunStatusIsOwnedByThePublisher(t *testing.T) {
	f := newWriterFixture(t)
	first := time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)

	staleVersion := f.currentRun().ResourceVersion
	if err := f.writer.UpdateRunStatus(f.ctx, staleVersion, RunStatus{
		Published:      2,
		LastProgress:   first,
		CandidateNames: []string{f.name("a"), f.name("b")},
	}); err != nil {
		t.Fatalf("update run status: %v", err)
	}
	run := f.currentRun()
	if len(run.Status.CandidateRefs) != 2 || run.Status.CandidateRefs[0].Name != f.name("a") {
		t.Fatalf("candidateRefs = %+v", run.Status.CandidateRefs)
	}
	if run.Status.Progress == nil || run.Status.Progress.Candidates == nil || run.Status.Progress.Candidates.Published != 2 {
		t.Fatalf("progress = %+v, want 2 published", run.Status.Progress)
	}
	if run.Status.LastProgressTime == nil || !run.Status.LastProgressTime.UTC().Equal(first) {
		t.Fatalf("lastProgressTime = %v, want %v", run.Status.LastProgressTime, first)
	}
	if !hasApplyManager(run.ManagedFields, "status") {
		t.Errorf("no status apply manager %q: %+v", fieldOwner, run.ManagedFields)
	}

	// A write based on the version read before the first write conflicts.
	err := f.writer.UpdateRunStatus(f.ctx, staleVersion, RunStatus{Published: 1, CandidateNames: []string{f.name("a")}})
	if !apierrors.IsConflict(err) {
		t.Fatalf("update with a stale resourceVersion: err = %v, want a conflict", err)
	}

	// Fields the publisher owns and omits are removed: this is why the reconciler carries
	// the existing progress time into every payload.
	if err := f.writer.UpdateRunStatus(f.ctx, run.ResourceVersion, RunStatus{Published: 1, CandidateNames: []string{f.name("a")}}); err != nil {
		t.Fatalf("shrinking update: %v", err)
	}
	run = f.currentRun()
	if len(run.Status.CandidateRefs) != 1 || run.Status.Progress.Candidates.Published != 1 {
		t.Errorf("status after shrinking = %+v", run.Status)
	}
	if run.Status.LastProgressTime != nil {
		t.Errorf("lastProgressTime = %v, want it removed when the payload omits it", run.Status.LastProgressTime)
	}
}

func TestEnvtestDeleteCandidateHonorsTheUIDPrecondition(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	if _, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate()); err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	uid := f.candidate(name).UID

	if err := f.writer.DeleteCandidate(f.ctx, name, types.UID("not-the-uid")); err == nil {
		t.Fatal("deleting with the wrong UID succeeded; a recreated candidate would be lost")
	}
	f.candidate(name) // still there

	if err := f.writer.DeleteCandidate(f.ctx, name, uid); err != nil {
		t.Fatalf("delete with the right UID: %v", err)
	}
	var gone v1beta2.DynamoGraphDeploymentCandidate
	err := f.client.Get(f.ctx, types.NamespacedName{Namespace: f.run.Namespace, Name: name}, &gone)
	if !apierrors.IsNotFound(err) {
		t.Fatalf("candidate still exists after delete: %v", err)
	}
	if err := f.writer.DeleteCandidate(f.ctx, name, uid); err != nil {
		t.Fatalf("deleting a missing candidate must not fail: %v", err)
	}
}

func TestEnvtestForeignObjectAtTheExpectedNameIsNotTakenOver(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	// An object controlled by another run sits on the deterministic name. It carries another
	// run's UID label, so the publisher's cache never lists it and only the writer can notice.
	foreignRun := f.run.DeepCopy()
	foreignRun.UID = "another-run-uid"
	foreign, err := buildCandidateApply(f.client.Scheme(), f.run.Namespace, string(foreignRun.UID), foreignRun, name, envtestCandidate())
	if err != nil {
		t.Fatal(err)
	}
	if err := f.client.Create(f.ctx, foreign); err != nil {
		t.Fatalf("create foreign candidate: %v", err)
	}

	if _, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate()); err == nil {
		t.Fatal("applying over an object this run does not control must be refused")
	}
	got := f.candidate(name)
	owner := metav1.GetControllerOf(got)
	if owner == nil || owner.UID != "another-run-uid" || got.Labels[LabelRunUID] != "another-run-uid" {
		t.Fatalf("the foreign object must be left as it was: owner=%v labels=%v", owner, got.Labels)
	}
	if hasApplyManager(got.ManagedFields, "") {
		t.Fatal("the publisher must not have applied anything to the foreign object")
	}
}

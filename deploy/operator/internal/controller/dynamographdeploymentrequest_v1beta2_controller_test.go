/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

type dgdrV1beta2Fixture struct {
	t          *testing.T
	client     client.Client
	reconciler *DynamoGraphDeploymentRequestV1beta2Reconciler
}

func newDGDRV1beta2Fixture(t *testing.T, objects ...client.Object) *dgdrV1beta2Fixture {
	t.Helper()
	return newDGDRV1beta2FixtureWithInterceptor(t, interceptor.Funcs{}, objects...)
}

func newDGDRV1beta2FixtureWithInterceptor(
	t *testing.T,
	funcs interceptor.Funcs,
	objects ...client.Object,
) *dgdrV1beta2Fixture {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta2.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta2 to scheme: %v", err)
	}
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objects...).
		WithIndex(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}, dgdrRunControllerUIDIndex, dgdrRunControllerUIDs).
		WithStatusSubresource(&nvidiacomv1beta2.DynamoGraphDeploymentRequest{}, &nvidiacomv1beta2.DynamoGraphDeploymentRun{}).
		WithInterceptorFuncs(funcs).
		Build()
	return &dgdrV1beta2Fixture{t: t, client: c, reconciler: &DynamoGraphDeploymentRequestV1beta2Reconciler{Client: c}}
}

func (f *dgdrV1beta2Fixture) reconcile() {
	f.t.Helper()
	if err := f.tryReconcile(); err != nil {
		f.t.Fatalf("reconcile: %v", err)
	}
}

func (f *dgdrV1beta2Fixture) tryReconcile() error {
	f.t.Helper()
	_, err := f.reconciler.Reconcile(f.t.Context(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "search"},
	})
	return err
}

func (f *dgdrV1beta2Fixture) request() *nvidiacomv1beta2.DynamoGraphDeploymentRequest {
	f.t.Helper()
	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}
	if err := f.client.Get(f.t.Context(), types.NamespacedName{Namespace: "default", Name: "search"}, request); err != nil {
		f.t.Fatalf("get request: %v", err)
	}
	return request
}

func (f *dgdrV1beta2Fixture) runNames() []string {
	f.t.Helper()
	list := &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}
	if err := f.client.List(f.t.Context(), list, client.InNamespace("default")); err != nil {
		f.t.Fatalf("list runs: %v", err)
	}
	names := make([]string, 0, len(list.Items))
	for _, run := range list.Items {
		names = append(names, run.Name)
	}
	return names
}

func (f *dgdrV1beta2Fixture) updateRunStatus(name string, mutate func(*nvidiacomv1beta2.DynamoGraphDeploymentRunStatus)) {
	f.t.Helper()
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(f.t.Context(), types.NamespacedName{Namespace: "default", Name: name}, run); err != nil {
		f.t.Fatalf("get run %s: %v", name, err)
	}
	mutate(&run.Status)
	if err := f.client.Status().Update(f.t.Context(), run); err != nil {
		f.t.Fatalf("update status of run %s: %v", name, err)
	}
}

// finish ends a Run the way the Run controller does: it publishes Completed and the outcome phase.
func (f *dgdrV1beta2Fixture) finish(name, phase string) {
	f.t.Helper()
	f.updateRunStatus(name, func(status *nvidiacomv1beta2.DynamoGraphDeploymentRunStatus) {
		status.Phase = nvidiacomv1beta2.DGDRRunPhase(phase)
		setCompletedForTest(status)
	})
}

// setPhaseOnly writes a terminal phase without the Completed condition.
func (f *dgdrV1beta2Fixture) setPhaseOnly(name, phase string) {
	f.t.Helper()
	f.updateRunStatus(name, func(status *nvidiacomv1beta2.DynamoGraphDeploymentRunStatus) {
		status.Phase = nvidiacomv1beta2.DGDRRunPhase(phase)
	})
}

// setCompletedOnly publishes Completed without a terminal phase.
func (f *dgdrV1beta2Fixture) setCompletedOnly(name string) {
	f.t.Helper()
	f.updateRunStatus(name, setCompletedForTest)
}

func setCompletedForTest(status *nvidiacomv1beta2.DynamoGraphDeploymentRunStatus) {
	meta.SetStatusCondition(&status.Conditions, metav1.Condition{
		Type:   nvidiacomv1beta2.DGDRRunConditionCompleted,
		Status: metav1.ConditionTrue,
		Reason: "Finished",
	})
}

func (f *dgdrV1beta2Fixture) bumpGeneration(generation int64) {
	f.t.Helper()
	request := f.request()
	request.Generation = generation
	if err := f.client.Update(f.t.Context(), request); err != nil {
		f.t.Fatalf("bump generation: %v", err)
	}
}

func dgdrV1beta2Request(withHardware bool) *nvidiacomv1beta2.DynamoGraphDeploymentRequest {
	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "search", Namespace: "default", UID: "request-uid", Generation: 1},
	}
	request.Spec.Model = &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"}
	request.Spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{
		Image:           "example.com/aisimulate:latest",
		VersionOverride: "0.13.0",
	}
	if withHardware {
		sku, budget := "h100_sxm", int32(4)
		request.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{
			GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: &sku, Budget: &budget},
		}
	}
	return request
}

func dgdrV1beta2AcceptedCondition(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) metav1.Condition {
	t.Helper()
	condition := meta.FindStatusCondition(request.Status.Conditions, nvidiacomv1beta2.DGDRRequestConditionAccepted)
	if condition == nil {
		t.Fatal("no Accepted condition")
	}
	return *condition
}

func TestV1beta2RequestControllerCreatesTheRunOfTheCurrentGeneration(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()

	t.Log("One Run exists, owned by the Request, with the resolved and frozen input")
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(t.Context(), types.NamespacedName{Namespace: "default", Name: "search-1"}, run); err != nil {
		t.Fatalf("get run: %v", err)
	}
	if !metav1.IsControlledBy(run, f.request()) {
		t.Fatal("run is not controlled by the request")
	}
	if got := run.Spec.AISimulate.VersionOverride; got != "0.13.0" {
		t.Fatalf("run versionOverride = %q, want the resolved 0.13.0", got)
	}
	parameters, err := aisimulate.InputForRun(run)
	if err != nil {
		t.Fatalf("input of the run: %v", err)
	}
	searchSpace, _ := parameters["search_space"].(map[string]any)
	if searchSpace["hardware_sku"] != "h100_sxm" || searchSpace["gpu_budget"] != float64(4) {
		t.Fatalf("frozen search_space = %v, want the hardware of the request", searchSpace)
	}

	t.Log("The status records the accepted generation and the active Run")
	request := f.request()
	if request.Status.ObservedGeneration != 1 {
		t.Fatalf("observedGeneration = %d, want 1", request.Status.ObservedGeneration)
	}
	if request.Status.ActiveRunRef == nil || request.Status.ActiveRunRef.Name != "search-1" {
		t.Fatalf("activeRunRef = %v, want search-1", request.Status.ActiveRunRef)
	}
	if condition := dgdrV1beta2AcceptedCondition(t, request); condition.Status != metav1.ConditionTrue {
		t.Fatalf("Accepted = %v, want True", condition)
	}
}

func TestV1beta2RequestControllerIsIdempotent(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()
	f.reconcile()
	f.reconcile()
	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want only search-1", names)
	}
}

func TestV1beta2RequestControllerWaitsForTheActiveRunBeforeStartingTheNext(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()

	t.Log("A spec change while search-1 runs creates nothing and says why")
	f.bumpGeneration(2)
	f.reconcile()
	if names := f.runNames(); len(names) != 1 {
		t.Fatalf("runs = %v, want only the active run", names)
	}
	request := f.request()
	if request.Status.ObservedGeneration != 1 {
		t.Fatalf("observedGeneration = %d, want 1 until generation 2 is accepted", request.Status.ObservedGeneration)
	}
	condition := dgdrV1beta2AcceptedCondition(t, request)
	if condition.Status != metav1.ConditionFalse || condition.Reason != dgdrReasonWaitingForActiveRun {
		t.Fatalf("condition = %v, want False/%s", condition, dgdrReasonWaitingForActiveRun)
	}

	t.Log("Once search-1 is completed, search-2 is created and search-1 moves to the history")
	f.finish("search-1", "Succeeded")
	f.reconcile()
	request = f.request()
	if request.Status.ObservedGeneration != 2 {
		t.Fatalf("observedGeneration = %d, want 2", request.Status.ObservedGeneration)
	}
	if request.Status.ActiveRunRef == nil || request.Status.ActiveRunRef.Name != "search-2" {
		t.Fatalf("activeRunRef = %v, want search-2", request.Status.ActiveRunRef)
	}
	if len(request.Status.RecentRunRefs) != 1 || request.Status.RecentRunRefs[0].Name != "search-1" {
		t.Fatalf("recentRunRefs = %v, want [search-1]", request.Status.RecentRunRefs)
	}
}

func TestV1beta2RequestControllerCoalescesIntermediateGenerations(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile() // search-1 is active

	f.bumpGeneration(2)
	f.reconcile()
	f.bumpGeneration(3)
	f.reconcile()

	// Completing search-1 admits the current level, generation 3.
	f.finish("search-1", "Succeeded")
	f.reconcile()

	// Generation 2 was an intermediate desired state and is never materialized.
	names := f.runNames()
	if len(names) != 2 {
		t.Fatalf("runs = %v, want search-1 and search-3", names)
	}
	for _, name := range names {
		if name != "search-1" && name != "search-3" {
			t.Fatalf("runs = %v, want search-1 and search-3 only", names)
		}
	}
}

func TestV1beta2RequestControllerIgnoresAPhaseWithoutTheCompletedCondition(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()
	f.bumpGeneration(2)

	f.setPhaseOnly("search-1", "Succeeded")
	f.reconcile()

	if names := f.runNames(); len(names) != 1 {
		t.Fatalf("runs = %v, want the next Run held back until Completed is published", names)
	}
	if condition := dgdrV1beta2AcceptedCondition(t, f.request()); condition.Reason != dgdrReasonWaitingForActiveRun {
		t.Fatalf("condition = %v, want reason %s", condition, dgdrReasonWaitingForActiveRun)
	}
}

func TestV1beta2RequestControllerReleasesOnCompletedWithoutATerminalPhase(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()
	f.bumpGeneration(2)

	f.setCompletedOnly("search-1") // the phase is still empty
	f.reconcile()

	if names := f.runNames(); len(names) != 2 {
		t.Fatalf("runs = %v, want search-2 started once Completed is True", names)
	}
}

func TestV1beta2RequestControllerCountsAnUnlabeledOwnedRunAsActive(t *testing.T) {
	request := dgdrV1beta2Request(true)
	active := dgdrV1beta2OwnedRun(request, 1, false) // correct controller OwnerReference, no labels
	f := newDGDRV1beta2Fixture(t, request, active)
	f.bumpGeneration(2)

	f.reconcile()

	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want no second active Run next to search-1", names)
	}
	if ref := f.request().Status.ActiveRunRef; ref == nil || ref.Name != "search-1" {
		t.Fatalf("activeRunRef = %v, want search-1", ref)
	}
}

func TestV1beta2RequestControllerCreatesNoRunWithoutHardware(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(false))
	f.reconcile()
	if names := f.runNames(); len(names) != 0 {
		t.Fatalf("runs = %v, want none", names)
	}
	request := f.request()
	condition := dgdrV1beta2AcceptedCondition(t, request)
	if condition.Status != metav1.ConditionFalse || condition.Reason != dgdrReasonInvalidSpec {
		t.Fatalf("condition = %v, want False/%s", condition, dgdrReasonInvalidSpec)
	}
	if request.Status.ObservedGeneration != 0 {
		t.Fatalf("observedGeneration = %d, want 0 for a rejected generation", request.Status.ObservedGeneration)
	}
}

func TestV1beta2RequestControllerLeavesForeignCarriersAlone(t *testing.T) {
	request := dgdrV1beta2Request(true)
	carrier := request.DeepCopy()
	carrier.Spec.V1Beta1 = &apiextensionsJSONForTest
	f := newDGDRV1beta2Fixture(t, carrier)
	f.reconcile()
	if names := f.runNames(); len(names) != 0 {
		t.Fatalf("runs = %v, want none for a v1beta1 carrier", names)
	}
	if got := f.request().Status.ObservedGeneration; got != 0 {
		t.Fatalf("observedGeneration = %d, want the carrier's status untouched", got)
	}
}

func dgdrV1beta2RunFixtures(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest, count int) []client.Object {
	objects := []client.Object{request}
	for generation := 1; generation <= count; generation++ {
		objects = append(objects, dgdrV1beta2OwnedRun(request, generation, true))
	}
	return objects
}

func TestV1beta2RequestControllerRetainsFiveTerminalRuns(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Generation = 8
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2RunFixtures(request, 7)...)
	f.reconcile()

	t.Log("The two oldest terminal runs are deleted and the newest five are listed newest first")
	assertRetained(t, f)
}

func assertRetained(t *testing.T, f *dgdrV1beta2Fixture) {
	t.Helper()
	for _, name := range []string{"search-1", "search-2"} {
		err := f.client.Get(t.Context(), types.NamespacedName{Namespace: "default", Name: name}, &nvidiacomv1beta2.DynamoGraphDeploymentRun{})
		if !apierrors.IsNotFound(err) {
			t.Fatalf("run %s: err = %v, want it deleted", name, err)
		}
	}
	refs := f.request().Status.RecentRunRefs
	want := []string{"search-7", "search-6", "search-5", "search-4", "search-3"}
	if len(refs) != len(want) {
		t.Fatalf("recentRunRefs = %v, want %v", refs, want)
	}
	for i := range want {
		if refs[i].Name != want[i] {
			t.Fatalf("recentRunRefs = %v, want %v", refs, want)
		}
	}
}

func TestV1beta2RequestControllerRetentionIgnoresTheGenerationAnnotation(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Generation = 8
	objects := dgdrV1beta2RunFixtures(request, 7)
	for i, object := range objects[1:] {
		if i%2 == 0 {
			delete(object.GetAnnotations(), dgdrV1beta2GenerationAnnotation) // missing
		} else {
			object.GetAnnotations()[dgdrV1beta2GenerationAnnotation] = "garbage" // malformed
		}
	}
	f := newDGDRV1beta2Fixture(t, objects...)
	f.reconcile()

	assertRetained(t, f)
}

func TestV1beta2RequestControllerRejectsAControlledRunWithoutAGenerationInItsName(t *testing.T) {
	request := dgdrV1beta2Request(true)
	odd := dgdrV1beta2OwnedRun(request, 1, true)
	odd.Name = "unrelated"
	f := newDGDRV1beta2Fixture(t, request, odd)

	if err := f.tryReconcile(); err == nil || !strings.Contains(err.Error(), "unrelated") {
		t.Fatalf("err = %v, want an error naming the run", err)
	}
	if names := f.runNames(); len(names) != 1 {
		t.Fatalf("runs = %v, want the odd run left alone and none created", names)
	}
}

func TestV1beta2RequestControllerPublishesStatusBeforeDeletingRetiredRuns(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Generation = 8
	failPatch := true
	f := newDGDRV1beta2FixtureWithInterceptor(t, interceptor.Funcs{
		SubResourcePatch: func(
			ctx context.Context, c client.Client, subResource string, obj client.Object,
			patch client.Patch, opts ...client.SubResourcePatchOption,
		) error {
			if failPatch {
				return errors.New("injected status patch failure")
			}
			return c.SubResource(subResource).Patch(ctx, obj, patch, opts...)
		},
	}, dgdrV1beta2RunFixtures(request, 7)...)

	if err := f.tryReconcile(); err == nil {
		t.Fatal("reconcile succeeded although the status patch failed")
	}
	// search-8 was created; the two oldest terminal Runs must still exist because the status did not
	// move on.
	if names := f.runNames(); len(names) != 8 {
		t.Fatalf("runs = %v, want all eight: nothing may be deleted before the status is published", names)
	}

	failPatch = false
	f.reconcile()
	assertRetained(t, f)
}

func dgdrV1beta2OwnedRun(
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	generation int,
	terminal bool,
) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("%s-%d", request.Name, generation),
			Namespace:   request.Namespace,
			Annotations: map[string]string{dgdrV1beta2GenerationAnnotation: fmt.Sprint(generation)},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta2.GroupVersion.String(),
				Kind:       "DynamoGraphDeploymentRequest",
				Name:       request.Name,
				UID:        request.UID,
				Controller: boolPointerForTest(true),
			}},
		},
	}
	if terminal {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhase("Succeeded")
		setCompletedForTest(&run.Status)
	}
	return run
}

func boolPointerForTest(value bool) *bool { return &value }

// apiextensionsJSONForTest is a non-empty v1beta1 carrier payload.
var apiextensionsJSONForTest = apiextensionsv1.JSON{Raw: []byte(`{"model":"carrier"}`)}

func TestV1beta2RequestControllerKeepsTheReadableRunNameWhenItFits(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Name = strings.Repeat("a", 250)
	request.Generation = 3
	if got, want := dgdrV1beta2RunName(request), request.Name+"-3"; got != want {
		t.Fatalf("run name = %q, want %q", got, want)
	}
}

func TestV1beta2RequestControllerDerivesDistinctValidNamesForOverlengthRequests(t *testing.T) {
	a := dgdrV1beta2Request(true)
	a.Name = strings.Repeat("a", 252)
	b := dgdrV1beta2Request(true)
	b.Name = strings.Repeat("a", 251) + "b" // the same retained prefix as a
	nameA, nameB := dgdrV1beta2RunName(a), dgdrV1beta2RunName(b)
	for _, name := range []string{nameA, nameB} {
		if len(name) > dgdrV1beta2MaxRunNameLength || !strings.HasSuffix(name, "-1") {
			t.Fatalf("run name %q (%d characters) is not a valid name with the generation suffix", name, len(name))
		}
	}
	if nameA == nameB {
		t.Fatalf("two different Request names produced the same Run name %q", nameA)
	}

	t.Log("The name maps back to its generation for its own Request only")
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{ObjectMeta: metav1.ObjectMeta{Name: nameA}}
	if generation, err := dgdrV1beta2RunGeneration(a.Name, run); err != nil || generation != 1 {
		t.Fatalf("generation = %d, err = %v, want 1", generation, err)
	}
	if _, err := dgdrV1beta2RunGeneration(b.Name, run); err == nil {
		t.Fatal("a Run name derived from another Request was accepted")
	}
}

func TestV1beta2RequestControllerReconcilesAnOverlengthRequestToOneRun(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Name = strings.Repeat("a", 252)
	f := newDGDRV1beta2Fixture(t, request)
	key := types.NamespacedName{Namespace: request.Namespace, Name: request.Name}
	for range 3 {
		if _, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: key}); err != nil {
			t.Fatalf("reconcile: %v", err)
		}
	}
	want := dgdrV1beta2RunName(request)
	if names := f.runNames(); len(names) != 1 || names[0] != want {
		t.Fatalf("runs = %v, want only %q", names, want)
	}
	live := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}
	if err := f.client.Get(t.Context(), key, live); err != nil {
		t.Fatalf("get request: %v", err)
	}
	if live.Status.ActiveRunRef == nil || live.Status.ActiveRunRef.Name != want || live.Status.ObservedGeneration != 1 {
		t.Fatalf("status = %+v, want the overlength Run active for generation 1", live.Status)
	}
}

func TestV1beta2RequestControllerDoesNotAdoptARunOfAnotherOwner(t *testing.T) {
	stale := &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "search-1",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta2.GroupVersion.String(),
				Kind:       "DynamoGraphDeploymentRequest",
				Name:       "search",
				UID:        "old-uid",
				Controller: boolPointerForTest(true),
			}},
		},
	}
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true), stale)

	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "search"},
	})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if result.RequeueAfter <= 0 {
		t.Fatal("no retry was scheduled for the Run name held by another owner")
	}
	request := f.request()
	if condition := dgdrV1beta2AcceptedCondition(t, request); condition.Reason != dgdrReasonRunNameConflict {
		t.Fatalf("condition = %v, want reason %s", condition, dgdrReasonRunNameConflict)
	}
	if request.Status.ActiveRunRef != nil {
		t.Fatalf("activeRunRef = %v, want none: the old Run is not ours", request.Status.ActiveRunRef)
	}
	if request.Status.ObservedGeneration != 0 {
		t.Fatalf("observedGeneration = %d, want 0", request.Status.ObservedGeneration)
	}
}

func TestV1beta2RequestControllerKeepsNullDeletedKeysOutOfTheRunInput(t *testing.T) {
	request := dgdrV1beta2Request(true)
	// The Request removes a typed value (workload isl) and a derived default (request_count) with null.
	const spec = `{"model":{"name":"Qwen/Qwen3-0.6B"},"hardware":{"gpu":{"sku":"h100_sxm","budget":4}},` +
		`"workload":{"static":{"isl":1024}},` +
		`"aisimulate":{"image":"example.com/aisimulate:latest","versionOverride":"0.13.0",` +
		`"parameters":{"workload":{"isl":null,"request_count":null}}}}`
	request.Spec = nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
	if err := json.Unmarshal([]byte(spec), &request.Spec); err != nil {
		t.Fatalf("decode spec: %v", err)
	}
	f := newDGDRV1beta2Fixture(t, request)

	run, err := f.reconciler.buildRun(request)
	if err != nil {
		t.Fatalf("build run: %v", err)
	}
	input, err := aisimulate.InputForRun(run)
	if err != nil {
		t.Fatalf("input of the run: %v", err)
	}
	workload, _ := input["workload"].(map[string]any)
	for _, key := range []string{"isl", "request_count"} {
		if value, found := workload[key]; found {
			t.Errorf("workload.%s = %v in the Run input, want it absent: the deletion must stay", key, value)
		}
	}
}

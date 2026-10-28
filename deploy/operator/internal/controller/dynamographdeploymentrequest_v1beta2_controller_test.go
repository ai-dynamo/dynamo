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
	"errors"
	"fmt"
	"strings"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
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
	return &dgdrV1beta2Fixture{t: t, client: c, reconciler: &DynamoGraphDeploymentRequestV1beta2Reconciler{
		Client: c, APIReader: c,
	}}
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
		Namespace: "default", Name: "search",
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

func (f *dgdrV1beta2Fixture) run(name string) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	f.t.Helper()
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(f.t.Context(), types.NamespacedName{Namespace: "default", Name: name}, run); err != nil {
		f.t.Fatalf("get run %s: %v", name, err)
	}
	return run
}

func (f *dgdrV1beta2Fixture) updateRunStatus(mutate func(*nvidiacomv1beta2.DynamoGraphDeploymentRunStatus)) {
	f.t.Helper()
	const name = "search-1"
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(f.t.Context(), types.NamespacedName{Namespace: "default", Name: name}, run); err != nil {
		f.t.Fatalf("get run %s: %v", name, err)
	}
	mutate(&run.Status)
	if err := f.client.Status().Update(f.t.Context(), run); err != nil {
		f.t.Fatalf("update status of run %s: %v", name, err)
	}
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
		Name: "search", Namespace: "default", UID: "request-uid", Generation: 1,
	}
	request.Spec.Model = &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"}
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

	t.Log("One Run exists, owned by the Request, with the accepted input snapshot")
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(t.Context(), types.NamespacedName{Namespace: "default", Name: "search-1"}, run); err != nil {
		t.Fatalf("get run: %v", err)
	}
	if !metav1.IsControlledBy(run, f.request()) {
		t.Fatal("run is not controlled by the request")
	}
	if !apiequality.Semantic.DeepEqual(run.Spec.NativeSearchInput, f.request().Spec.NativeSearchInput) {
		t.Fatalf("run input = %#v, want the accepted Request snapshot %#v", run.Spec.NativeSearchInput, f.request().Spec.NativeSearchInput)
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

func TestV1beta2RequestControllerCancelsTheActiveRunBeforeStartingTheNext(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()

	t.Log("A spec change requests cancellation without deleting search-1")
	f.bumpGeneration(2)
	f.reconcile()
	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want the superseded Run retained", names)
	}
	if got := f.run("search-1").Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]; got != "2" {
		t.Fatalf("cancellation generation = %q, want 2", got)
	}
	request := f.request()
	if request.Status.ObservedGeneration != 1 {
		t.Fatalf("observedGeneration = %d, want 1 until generation 2 is accepted", request.Status.ObservedGeneration)
	}
	condition := dgdrV1beta2AcceptedCondition(t, request)
	if condition.Status != metav1.ConditionFalse || condition.Reason != dgdrReasonCancelingStaleRun {
		t.Fatalf("condition = %v, want False/%s", condition, dgdrReasonCancelingStaleRun)
	}

	t.Log("Retries leave the retained Run active until its controller publishes Completed")
	f.reconcile()
	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want no replacement before cancellation completes", names)
	}

	t.Log("The terminal cancellation outcome releases the latest generation")
	f.updateRunStatus(setCompletedForTest)
	f.reconcile()
	f.reconcile()
	request = f.request()
	if request.Status.ObservedGeneration != 2 {
		t.Fatalf("observedGeneration = %d, want 2", request.Status.ObservedGeneration)
	}
	if request.Status.ActiveRunRef == nil || request.Status.ActiveRunRef.Name != "search-2" {
		t.Fatalf("activeRunRef = %v, want search-2", request.Status.ActiveRunRef)
	}
	if len(request.Status.RecentRunRefs) != 1 || request.Status.RecentRunRefs[0].Name != "search-1" {
		t.Fatalf("recentRunRefs = %v, want the canceled Run retained in terminal history", request.Status.RecentRunRefs)
	}
}

func TestV1beta2RequestControllerDoesNotStartANewGenerationWhileTheRunCacheLags(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Status.ActiveRunRef = &corev1.LocalObjectReference{Name: "search-1"}
	meta.SetStatusCondition(&request.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta2.DGDRRequestConditionAccepted,
		Status:             metav1.ConditionFalse,
		ObservedGeneration: 1,
		Reason:             dgdrReasonRunCreating,
		Message:            "Creating Run search-1 for generation 1",
	})
	hideRuns := true
	f := newDGDRV1beta2FixtureWithInterceptor(t, interceptor.Funcs{
		List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, isRunList := list.(*nvidiacomv1beta2.DynamoGraphDeploymentRunList); hideRuns && isRunList {
				return nil
			}
			return c.List(ctx, list, opts...)
		},
	}, request, dgdrV1beta2OwnedRun(request, 1, false))

	t.Log("Observe generation 2 before the Run cache has observed generation 1's reserved create")
	f.bumpGeneration(2)

	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{
		Namespace: "default",
		Name:      "search"})
	if err != nil {
		t.Fatalf("reconcile with lagging Run cache: %v", err)
	}
	if result.RequeueAfter <= 0 {
		t.Fatal("lagging Run cache did not schedule a bounded retry")
	}

	t.Log("The durable RunCreating reservation prevents generation 2 from being created beside generation 1")
	hideRuns = false
	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want only search-1 while its cache observation lags", names)
	}
	live := f.request()
	if live.Status.ActiveRunRef == nil || live.Status.ActiveRunRef.Name != "search-1" {
		t.Fatalf("activeRunRef = %v, want the durable search-1 reservation", live.Status.ActiveRunRef)
	}
	if live.Status.ObservedGeneration != 0 {
		t.Fatalf("observedGeneration = %d, want 0 until the reserved Run is observed", live.Status.ObservedGeneration)
	}

	t.Log("Once the cache catches up, cancellation is requested and the old Run remains retained")
	f.reconcile()
	if got := f.run("search-1").Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]; got != "2" {
		t.Fatalf("cancellation generation = %q, want 2", got)
	}
	f.updateRunStatus(setCompletedForTest)
	f.reconcile()
	if names := f.runNames(); len(names) != 2 {
		t.Fatalf("runs = %v, want retained search-1 and active search-2", names)
	}
}

func TestV1beta2RequestControllerReleasesAReservationAfterAFailedCreate(t *testing.T) {
	failCreate := true
	f := newDGDRV1beta2FixtureWithInterceptor(t, interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, object client.Object, opts ...client.CreateOption) error {
			if _, isRun := object.(*nvidiacomv1beta2.DynamoGraphDeploymentRun); isRun && failCreate {
				failCreate = false
				return errors.New("unknown create outcome")
			}
			return c.Create(ctx, object, opts...)
		},
	}, dgdrV1beta2Request(true))

	t.Log("Persist generation 1's reservation before its Run create fails")
	if err := f.tryReconcile(); err == nil || !strings.Contains(err.Error(), "unknown create outcome") {
		t.Fatalf("reconcile error = %v, want failed Run create", err)
	}
	reserved := f.request()
	if reserved.Status.ActiveRunRef == nil || reserved.Status.ActiveRunRef.Name != "search-1" {
		t.Fatalf("activeRunRef = %v, want durable search-1 reservation", reserved.Status.ActiveRunRef)
	}

	t.Log("A newer generation replaces the stale reservation after the API confirms no Run exists")
	f.bumpGeneration(2)
	f.reconcile()
	if names := f.runNames(); len(names) != 1 || names[0] != "search-2" {
		t.Fatalf("runs = %v, want only search-2", names)
	}
	accepted := dgdrV1beta2AcceptedCondition(t, f.request())
	if accepted.Status != metav1.ConditionTrue || accepted.ObservedGeneration != 2 {
		t.Fatalf("condition = %v, want generation 2 accepted", accepted)
	}
}

func TestV1beta2RequestControllerCoalescesIntermediateGenerations(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile() // search-1 is active

	f.bumpGeneration(2)
	f.reconcile()
	f.bumpGeneration(3)
	f.reconcile()

	t.Log("The latest generation replaces the causal cancellation marker without creating generation 2")
	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want only the retained active search-1", names)
	}
	if got := f.run("search-1").Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]; got != "3" {
		t.Fatalf("cancellation generation = %q, want the latest generation 3", got)
	}

	t.Log("A stale retry cannot move the causal cancellation generation backwards")
	if err := f.reconciler.requestRunCancellation(t.Context(), f.run("search-1"), 2); err != nil {
		t.Fatalf("retry stale cancellation: %v", err)
	}
	if got := f.run("search-1").Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]; got != "3" {
		t.Fatalf("cancellation generation = %q after stale retry, want 3", got)
	}

	t.Log("Once cancellation completes, generation 3 starts and generation 2 remains coalesced")
	f.updateRunStatus(setCompletedForTest)
	f.reconcile()
	if names := f.runNames(); len(names) != 2 || names[0] != "search-1" || names[1] != "search-3" {
		t.Fatalf("runs = %v, want retained search-1 and active search-3", names)
	}
}

func TestV1beta2RequestControllerRepairsInvalidOrStaleCancellationGeneration(t *testing.T) {
	tests := []struct {
		name     string
		existing string
	}{
		{name: "malformed", existing: "malformed"},
		{name: "zero", existing: "0"},
		{name: "negative", existing: "-1"},
		{name: "stale", existing: "2"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := dgdrV1beta2Request(true)
			run := dgdrV1beta2OwnedRun(request, 1, false)
			run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = tt.existing
			f := newDGDRV1beta2Fixture(t, request, run)
			f.bumpGeneration(3)

			t.Log("Reconcile the newer Request generation")
			f.reconcile()

			if got := f.run(run.Name).Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]; got != "3" {
				t.Fatalf("cancellation generation = %q, want repaired generation 3", got)
			}
		})
	}
}

func TestV1beta2RequestControllerAllowsTheCurrentRunToRemainActiveIndefinitely(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()

	f.updateRunStatus(func(status *nvidiacomv1beta2.DynamoGraphDeploymentRunStatus) {
		status.Phase = nvidiacomv1beta2.DGDRRunPhaseSucceeded
	})
	f.reconcile()

	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want the current Run left alone without a new generation", names)
	}
	if condition := dgdrV1beta2AcceptedCondition(t, f.request()); condition.Reason != dgdrReasonRunCreated {
		t.Fatalf("condition = %v, want reason %s", condition, dgdrReasonRunCreated)
	}
}

func TestV1beta2RequestControllerReleasesOnCompletedWithoutATerminalPhase(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true))
	f.reconcile()
	f.bumpGeneration(2)

	f.updateRunStatus(setCompletedForTest) // the phase is still empty
	f.reconcile()

	if names := f.runNames(); len(names) != 2 {
		t.Fatalf("runs = %v, want search-2 started once Completed is True", names)
	}
}

func TestV1beta2RequestControllerCancelsAnUnlabeledOwnedRun(t *testing.T) {
	request := dgdrV1beta2Request(true)
	active := dgdrV1beta2OwnedRun(request, 1, false) // correct controller OwnerReference, no labels
	f := newDGDRV1beta2Fixture(t, request, active)
	f.bumpGeneration(2)

	f.reconcile()

	if names := f.runNames(); len(names) != 1 || names[0] != "search-1" {
		t.Fatalf("runs = %v, want the superseded owned Run retained", names)
	}
	if got := f.run("search-1").Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]; got != "2" {
		t.Fatalf("cancellation generation = %q, want 2", got)
	}
	if ref := f.request().Status.ActiveRunRef; ref == nil || ref.Name != "search-1" {
		t.Fatalf("activeRunRef = %v, want search-1 retained until cancellation is observed", ref)
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

func TestV1beta2RequestControllerOrdersRetainedRunsByTheirGenerationAnnotation(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Generation = 8
	objects := []client.Object{request}
	for generation := 1; generation <= 7; generation++ {
		run := dgdrV1beta2OwnedRun(request, generation, true)
		run.Name = fmt.Sprintf("arbitrary-%02d", 100-generation) // sorts opposite to the generation
		objects = append(objects, run)
	}
	f := newDGDRV1beta2Fixture(t, objects...)
	f.reconcile()
	f.reconcile()

	refs := f.request().Status.RecentRunRefs
	want := []string{"arbitrary-93", "arbitrary-94", "arbitrary-95", "arbitrary-96", "arbitrary-97"}
	if len(refs) != len(want) {
		t.Fatalf("recentRunRefs = %v, want %v", refs, want)
	}
	for i := range want {
		if refs[i].Name != want[i] {
			t.Fatalf("recentRunRefs = %v, want %v", refs, want)
		}
	}
	for _, name := range f.runNames() {
		if name == "arbitrary-98" || name == "arbitrary-99" {
			t.Fatalf("run %q of generation %s was not retired", name, map[string]string{"arbitrary-98": "2", "arbitrary-99": "1"}[name])
		}
	}
}

func TestV1beta2RequestControllerRejectsAControlledRunWithoutAGenerationAnnotation(t *testing.T) {
	request := dgdrV1beta2Request(true)
	odd := dgdrV1beta2OwnedRun(request, 1, true)
	odd.Name = "unrelated"
	delete(odd.Annotations, dgdrV1beta2GenerationAnnotation)
	f := newDGDRV1beta2Fixture(t, request, odd)

	if err := f.tryReconcile(); err == nil || !strings.Contains(err.Error(), "unrelated") {
		t.Fatalf("err = %v, want an error naming the run", err)
	}
	if names := f.runNames(); len(names) != 1 {
		t.Fatalf("runs = %v, want the odd run left alone and none created", names)
	}
}

func TestV1beta2RequestControllerReusesAnOwnedRunWhateverItsName(t *testing.T) {
	request := dgdrV1beta2Request(true)
	existing := dgdrV1beta2OwnedRun(request, int(request.Generation), false)
	existing.Name = "an-arbitrary-valid-name"
	f := newDGDRV1beta2Fixture(t, request, existing)

	f.reconcile()

	if names := f.runNames(); len(names) != 1 || names[0] != existing.Name {
		t.Fatalf("runs = %v, want only %q: an owned Run of the current generation is reused", names, existing.Name)
	}
	live := f.request()
	if live.Status.ActiveRunRef == nil || live.Status.ActiveRunRef.Name != existing.Name ||
		live.Status.ObservedGeneration != request.Generation {
		t.Fatalf("status = %+v, want %q active for generation %d", live.Status, existing.Name, request.Generation)
	}
	if condition := dgdrV1beta2AcceptedCondition(t, live); condition.Status != metav1.ConditionTrue {
		t.Fatalf("Accepted = %v, want True", condition)
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
	// The durable reservation is published before search-8 is created. A failed status write must
	// leave every existing Run untouched and must not perform the create side effect.
	if names := f.runNames(); len(names) != 7 {
		t.Fatalf("runs = %v, want the seven existing Runs and no unreserved search-8", names)
	}

	failPatch = false
	f.reconcile()
	f.reconcile()
	assertRetained(t, f)
}

func dgdrV1beta2OwnedRun(
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	generation int,
	terminal bool,
) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		Name:        fmt.Sprintf("%s-%d", request.Name, generation),
		Namespace:   request.Namespace,
		Annotations: map[string]string{dgdrV1beta2GenerationAnnotation: fmt.Sprint(generation)},
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: nvidiacomv1beta2.GroupVersion.String(),
			Kind:       "DynamoGraphDeploymentRequest",
			Name:       request.Name,
			UID:        request.UID,
			Controller: new(true),
		}},
	}
	if terminal {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhase("Succeeded")
		setCompletedForTest(&run.Status)
	}
	return run
}

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
		Name:      "search-1",
		Namespace: "default",
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: nvidiacomv1beta2.GroupVersion.String(),
			Kind:       "DynamoGraphDeploymentRequest",
			Name:       "search",
			UID:        "old-uid",
			Controller: new(true),
		}},
	}
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(true), stale)

	result, err := f.reconciler.Reconcile(t.Context(), ctrl.Request{
		Namespace: "default", Name: "search",
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

func TestV1beta2RequestControllerPreservesInputOverridesInTheRunSnapshot(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{InputOverrides: &runtime.RawExtension{
		Raw: []byte(`{"traffic":{"source":{"input_tokens":null}},"optimizer":{"max_trials":null}}`),
	}}
	f := newDGDRV1beta2Fixture(t, request)

	run, err := f.reconciler.buildRun(request)
	if err != nil {
		t.Fatalf("build run: %v", err)
	}
	if run.Spec.AISimulate == nil || run.Spec.AISimulate.InputOverrides == nil {
		t.Fatal("Run did not preserve spec.aisimulate.inputOverrides")
	}
	if got, want := string(run.Spec.AISimulate.InputOverrides.Raw), string(request.Spec.AISimulate.InputOverrides.Raw); got != want {
		t.Fatalf("run inputOverrides = %s, want %s", got, want)
	}
}

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
	"encoding/json"
	"fmt"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

type dgdrV1beta2Fixture struct {
	t          *testing.T
	client     client.Client
	reconciler *DynamoGraphDeploymentRequestV1beta2Reconciler
}

func newDGDRV1beta2Fixture(t *testing.T, objects ...client.Object) *dgdrV1beta2Fixture {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta2.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta2 to scheme: %v", err)
	}
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objects...).
		WithStatusSubresource(&nvidiacomv1beta2.DynamoGraphDeploymentRequest{}, &nvidiacomv1beta2.DynamoGraphDeploymentRun{}).
		Build()
	return &dgdrV1beta2Fixture{t: t, client: c, reconciler: &DynamoGraphDeploymentRequestV1beta2Reconciler{Client: c}}
}

func (f *dgdrV1beta2Fixture) reconcile() {
	f.t.Helper()
	if _, err := f.reconciler.Reconcile(f.t.Context(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "search"},
	}); err != nil {
		f.t.Fatalf("reconcile: %v", err)
	}
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

func (f *dgdrV1beta2Fixture) setPhase(name, phase string) {
	f.t.Helper()
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := f.client.Get(f.t.Context(), types.NamespacedName{Namespace: "default", Name: name}, run); err != nil {
		f.t.Fatalf("get run %s: %v", name, err)
	}
	run.Status.Phase = nvidiacomv1beta2.DGDRRunPhase(phase)
	if err := f.client.Status().Update(f.t.Context(), run); err != nil {
		f.t.Fatalf("set phase of run %s: %v", name, err)
	}
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

func dgdrV1beta2ReconciledCondition(t *testing.T, request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) metav1.Condition {
	t.Helper()
	condition := meta.FindStatusCondition(request.Status.Conditions, DGDRReconciledConditionV1beta2)
	if condition == nil {
		t.Fatal("no Reconciled condition")
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
	var parameters map[string]any
	if err := json.Unmarshal(run.Spec.AISimulate.Parameters.Raw, &parameters); err != nil {
		t.Fatalf("decode frozen parameters: %v", err)
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
	if condition := dgdrV1beta2ReconciledCondition(t, request); condition.Status != metav1.ConditionTrue {
		t.Fatalf("Reconciled = %v, want True", condition)
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
	if condition := dgdrV1beta2ReconciledCondition(t, request); condition.Reason != dgdrReasonWaitingForActiveRun {
		t.Fatalf("condition = %v, want reason %s", condition, dgdrReasonWaitingForActiveRun)
	}

	t.Log("Once search-1 is terminal, search-2 is created and search-1 moves to the history")
	f.setPhase("search-1", dgdrRunPhaseSucceeded)
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

func TestV1beta2RequestControllerCreatesNoRunWithoutHardware(t *testing.T) {
	f := newDGDRV1beta2Fixture(t, dgdrV1beta2Request(false))
	f.reconcile()
	if names := f.runNames(); len(names) != 0 {
		t.Fatalf("runs = %v, want none", names)
	}
	request := f.request()
	condition := dgdrV1beta2ReconciledCondition(t, request)
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

func TestV1beta2RequestControllerRetainsFiveTerminalRuns(t *testing.T) {
	request := dgdrV1beta2Request(true)
	request.Generation = 8
	objects := []client.Object{request}
	for generation := 1; generation <= 7; generation++ {
		objects = append(objects, dgdrV1beta2TerminalRun(request, generation))
	}
	f := newDGDRV1beta2Fixture(t, objects...)
	f.reconcile()

	t.Log("The two oldest terminal runs are deleted and the newest five are listed newest first")
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

func dgdrV1beta2TerminalRun(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest, generation int) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("search-%d", generation),
			Namespace:   "default",
			Labels:      map[string]string{dgdrV1beta2RequestLabel: request.Name},
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
	run.Status.Phase = nvidiacomv1beta2.DGDRRunPhase(dgdrRunPhaseSucceeded)
	return run
}

func boolPointerForTest(value bool) *bool { return &value }

// apiextensionsJSONForTest is a non-empty v1beta1 carrier payload.
var apiextensionsJSONForTest = apiextensionsv1.JSON{Raw: []byte(`{"model":"carrier"}`)}

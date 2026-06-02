/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoGraphDeploymentReconciler_shouldUseDisaggregatedSet(t *testing.T) {
	tests := []struct {
		name       string
		runtime    *commoncontroller.RuntimeConfig
		dgd        *nvidiacomv1beta1.DynamoGraphDeployment
		wantUse    bool
		wantReason string
	}{
		{
			name: "enabled with two eligible roles",
			runtime: &commoncontroller.RuntimeConfig{
				LWSEnabled:              true,
				DisaggregatedSetEnabled: true,
			},
			dgd:     disaggregatedSetTestDGD(disaggregatedSetComponent("Prefill", nvidiacomv1beta1.ComponentTypePrefill, ptr.To(int32(1))), disaggregatedSetComponent("Decode", nvidiacomv1beta1.ComponentTypeDecode, ptr.To(int32(1)))),
			wantUse: true,
		},
		{
			name: "falls back when API is unavailable",
			runtime: &commoncontroller.RuntimeConfig{
				LWSEnabled:              true,
				DisaggregatedSetEnabled: false,
			},
			dgd:        disaggregatedSetTestDGD(disaggregatedSetComponent("Prefill", nvidiacomv1beta1.ComponentTypePrefill, ptr.To(int32(1))), disaggregatedSetComponent("Decode", nvidiacomv1beta1.ComponentTypeDecode, ptr.To(int32(1)))),
			wantReason: "DisaggregatedSet API is not available",
		},
		{
			name: "falls back on mixed zero and positive replicas",
			runtime: &commoncontroller.RuntimeConfig{
				LWSEnabled:              true,
				DisaggregatedSetEnabled: true,
			},
			dgd:        disaggregatedSetTestDGD(disaggregatedSetComponent("Prefill", nvidiacomv1beta1.ComponentTypePrefill, ptr.To(int32(0))), disaggregatedSetComponent("Decode", nvidiacomv1beta1.ComponentTypeDecode, ptr.To(int32(1)))),
			wantReason: "replicas to be zero for all roles or positive for all roles",
		},
		{
			name: "falls back with only one eligible role",
			runtime: &commoncontroller.RuntimeConfig{
				LWSEnabled:              true,
				DisaggregatedSetEnabled: true,
			},
			dgd:        disaggregatedSetTestDGD(disaggregatedSetComponent("Prefill", nvidiacomv1beta1.ComponentTypePrefill, ptr.To(int32(1))), singleNodeComponent("Frontend", nvidiacomv1beta1.ComponentTypeFrontend)),
			wantReason: "requires at least two eligible",
		},
		{
			name: "falls back for scaling adapter roles",
			runtime: &commoncontroller.RuntimeConfig{
				LWSEnabled:              true,
				DisaggregatedSetEnabled: true,
			},
			dgd:        disaggregatedSetTestDGD(scalingAdapterDisaggregatedSetComponent("Prefill"), disaggregatedSetComponent("Decode", nvidiacomv1beta1.ComponentTypeDecode, ptr.To(int32(1)))),
			wantReason: "no scale subresource",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reconciler := &DynamoGraphDeploymentReconciler{RuntimeConfig: tt.runtime}
			gotUse, gotReason := reconciler.shouldUseDisaggregatedSet(tt.dgd)
			if gotUse != tt.wantUse {
				t.Fatalf("shouldUseDisaggregatedSet() use = %v, want %v", gotUse, tt.wantUse)
			}
			if tt.wantReason != "" && !strings.Contains(gotReason, tt.wantReason) {
				t.Fatalf("shouldUseDisaggregatedSet() reason = %q, want to contain %q", gotReason, tt.wantReason)
			}
		})
	}
}

func TestSelectDisaggregatedSetComponentsAssignsStableRoleNames(t *testing.T) {
	dgd := disaggregatedSetTestDGD(
		disaggregatedSetComponent("PrefillWorker", nvidiacomv1beta1.ComponentTypePrefill, nil),
		disaggregatedSetComponent("DecodeWorker", nvidiacomv1beta1.ComponentTypeDecode, nil),
		disaggregatedSetComponent("GenericWorker", nvidiacomv1beta1.ComponentTypeWorker, nil),
		singleNodeComponent("Frontend", nvidiacomv1beta1.ComponentTypeFrontend),
	)

	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		t.Fatalf("selectDisaggregatedSetComponents() reason = %q, want empty", reason)
	}
	wantRoles := map[string]string{
		"PrefillWorker": "prefill",
		"DecodeWorker":  "decode",
		"GenericWorker": "genericworker",
	}
	for component, wantRole := range wantRoles {
		if gotRole := selection.componentToRole[component]; gotRole != wantRole {
			t.Fatalf("role for %s = %q, want %q", component, gotRole, wantRole)
		}
	}
	if _, selected := selection.componentToRole["Frontend"]; selected {
		t.Fatalf("single-node frontend should not be selected for DisaggregatedSet")
	}
}

func TestCheckDisaggregatedSetReadiness(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("test-dgd")
	ds.Object["status"] = map[string]any{
		"roleStatuses": []any{
			map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
			map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(1)},
		},
	}
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"Prefill": "prefill", "Decode": "decode"},
		desiredReplicas: map[string]int32{"Prefill": 2, "Decode": 2},
	}

	ready, reason, statuses := checkDisaggregatedSetReadiness(ds, selection)
	if ready {
		t.Fatalf("checkDisaggregatedSetReadiness() ready = true, want false")
	}
	if !strings.Contains(reason, "Decode") || !strings.Contains(reason, "ready=1") {
		t.Fatalf("checkDisaggregatedSetReadiness() reason = %q, want Decode ready=1 detail", reason)
	}
	if gotReady := ptr.Deref(statuses["Prefill"].ReadyReplicas, 0); gotReady != 2 {
		t.Fatalf("Prefill ready replicas = %d, want 2", gotReady)
	}
	if gotReady := ptr.Deref(statuses["Decode"].ReadyReplicas, 0); gotReady != 1 {
		t.Fatalf("Decode ready replicas = %d, want 1", gotReady)
	}
}

func TestCheckDisaggregatedSetReadinessZeroReplicas(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("test-dgd")
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"Prefill": "prefill", "Decode": "decode"},
		desiredReplicas: map[string]int32{"Prefill": 0, "Decode": 0},
	}

	ready, reason, statuses := checkDisaggregatedSetReadiness(ds, selection)
	if !ready {
		t.Fatalf("checkDisaggregatedSetReadiness() ready = false, reason = %q", reason)
	}
	if statuses["Prefill"].Replicas != 0 || statuses["Decode"].Replicas != 0 {
		t.Fatalf("zero-replica DisaggregatedSet statuses = %#v, want zero replicas", statuses)
	}
}

func TestCheckDisaggregatedSetReadinessWaitsForObservedGeneration(t *testing.T) {
	ds := readyDisaggregatedSetStatus()
	ds.SetGeneration(3)
	ds.Object["status"].(map[string]any)["observedGeneration"] = int64(2)
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"Prefill": "prefill", "Decode": "decode"},
		desiredReplicas: map[string]int32{"Prefill": 2, "Decode": 2},
	}

	ready, reason, _ := checkDisaggregatedSetReadiness(ds, selection)
	if ready {
		t.Fatalf("checkDisaggregatedSetReadiness() ready = true, want false")
	}
	if !strings.Contains(reason, "observedGeneration=2") {
		t.Fatalf("checkDisaggregatedSetReadiness() reason = %q, want observedGeneration detail", reason)
	}

	ds.Object["status"].(map[string]any)["observedGeneration"] = int64(3)
	ready, reason, _ = checkDisaggregatedSetReadiness(ds, selection)
	if !ready {
		t.Fatalf("checkDisaggregatedSetReadiness() ready = false, reason = %q", reason)
	}
}

func TestDisaggregatedSetRenderRollingUpdateContextUsesFullDesiredReplicas(t *testing.T) {
	dgd := disaggregatedSetTestDGD(
		disaggregatedSetComponent("Prefill", nvidiacomv1beta1.ComponentTypePrefill, ptr.To(int32(4))),
		disaggregatedSetComponent("Decode", nvidiacomv1beta1.ComponentTypeDecode, ptr.To(int32(5))),
	)
	rollingCtx := dynamo.RollingUpdateContext{
		NewWorkerHash:     "abcd1234",
		OldWorkerReplicas: map[string]int32{"Prefill": 4, "Decode": 5},
		NewWorkerReplicas: map[string]int32{"Prefill": 1, "Decode": 1},
	}

	got := disaggregatedSetRenderRollingUpdateContext(rollingCtx, dgd)
	if got.NewWorkerHash != "abcd1234" {
		t.Fatalf("NewWorkerHash = %q, want abcd1234", got.NewWorkerHash)
	}
	if got.NewWorkerReplicas["Prefill"] != 4 || got.NewWorkerReplicas["Decode"] != 5 {
		t.Fatalf("NewWorkerReplicas = %#v, want full desired replicas", got.NewWorkerReplicas)
	}
	if rollingCtx.NewWorkerReplicas["Prefill"] != 1 {
		t.Fatalf("input rolling context was mutated: %#v", rollingCtx.NewWorkerReplicas)
	}
}

func TestEnsureControlledByDGDAdoptsDCDOwnedService(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme(corev1) error = %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme(nvidia) error = %v", err)
	}

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: types.UID("dgd-uid")},
	}
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph-prefill-abcd1234",
			Namespace: "default",
			UID:       types.UID("dcd-uid"),
			OwnerReferences: []metav1.OwnerReference{
				*dgdControllerOwnerReference(dgd),
			},
		},
	}
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph-prefill-abcd1234",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion:         nvidiacomv1beta1.GroupVersion.String(),
				Kind:               "DynamoComponentDeployment",
				Name:               dcd.Name,
				UID:                dcd.UID,
				Controller:         ptr.To(true),
				BlockOwnerDeletion: ptr.To(true),
			}},
		},
	}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(dcd, service).Build(),
	}

	if err := reconciler.ensureControlledByDGD(t.Context(), dgd, service); err != nil {
		t.Fatalf("ensureControlledByDGD() error = %v", err)
	}

	got := &corev1.Service{}
	if err := reconciler.Get(t.Context(), types.NamespacedName{Name: service.Name, Namespace: service.Namespace}, got); err != nil {
		t.Fatalf("Get(service) error = %v", err)
	}
	if !metav1.IsControlledBy(got, dgd) {
		t.Fatalf("service ownerReferences = %#v, want DGD controller owner", got.OwnerReferences)
	}
}

func TestListOwnedSelectedDCDsSkipsUserManagedDCDs(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme(nvidia) error = %v", err)
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: types.UID("dgd-uid")},
	}
	owned := selectedDCDForCleanup("graph-prefill-abcd1234", "Prefill", dgd, true)
	userManaged := selectedDCDForCleanup("graph-decode-abcd1234", "Decode", dgd, false)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(owned, userManaged).Build(),
	}
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"Prefill": "prefill", "Decode": "decode"},
	}

	got, err := reconciler.listOwnedSelectedDCDs(t.Context(), dgd, selection)
	if err != nil {
		t.Fatalf("listOwnedSelectedDCDs() error = %v", err)
	}
	if len(got) != 1 || got[0].Name != owned.Name {
		t.Fatalf("listOwnedSelectedDCDs() = %#v, want only %q", got, owned.Name)
	}
}

func TestGetUpdatedInProgressForDisaggregatedSetUsesRoleStatus(t *testing.T) {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(disaggregatedSetGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(disaggregatedSetGVK.GroupVersion().WithKind("DisaggregatedSetList"), &unstructured.UnstructuredList{})

	dgd := disaggregatedSetTestDGD(
		disaggregatedSetComponent("Prefill", nvidiacomv1beta1.ComponentTypePrefill, ptr.To(int32(2))),
		disaggregatedSetComponent("Decode", nvidiacomv1beta1.ComponentTypeDecode, ptr.To(int32(2))),
	)
	ds := readyDisaggregatedSetStatus()
	ds.SetName(disaggregatedSetName(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.Object["status"].(map[string]any)["roleStatuses"].([]any)[1].(map[string]any)["readyReplicas"] = int64(1)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(ds).Build(),
	}

	got := reconciler.getUpdatedInProgressForDisaggregatedSet(t.Context(), dgd, []string{"Prefill", "Decode"})
	if len(got) != 1 || got[0] != "Decode" {
		t.Fatalf("getUpdatedInProgressForDisaggregatedSet() = %#v, want Decode only", got)
	}
}

func readyDisaggregatedSetStatus() *unstructured.Unstructured {
	ds := newDisaggregatedSetObject()
	ds.SetName("test-dgd")
	ds.Object["status"] = map[string]any{
		"roleStatuses": []any{
			map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
			map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
		},
	}
	return ds
}

func selectedDCDForCleanup(name, componentName string, dgd *nvidiacomv1beta1.DynamoGraphDeployment, owned bool) *nvidiacomv1beta1.DynamoComponentDeployment {
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: dgd.Namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			},
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: componentName,
				ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
			},
		},
	}
	if owned {
		dcd.OwnerReferences = []metav1.OwnerReference{*dgdControllerOwnerReference(dgd)}
	}
	return dcd
}

func disaggregatedSetTestDGD(components ...nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationEnableGrove:            consts.KubeLabelValueFalse,
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: components,
		},
	}
}

func disaggregatedSetComponent(name string, componentType nvidiacomv1beta1.ComponentType, replicas *int32) nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: name,
		ComponentType: componentType,
		Replicas:      replicas,
		Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
	}
}

func scalingAdapterDisaggregatedSetComponent(name string) nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	component := disaggregatedSetComponent(name, nvidiacomv1beta1.ComponentTypePrefill, ptr.To(int32(1)))
	component.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
	return component
}

func singleNodeComponent(name string, componentType nvidiacomv1beta1.ComponentType) nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: name,
		ComponentType: componentType,
	}
}

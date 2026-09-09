/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"encoding/json"
	"testing"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

const editedThroughAlphaBuildID = "model/edited-through-alpha"

func testCanonicalLPXConfig() *v1beta1.LPXConfig {
	return &v1beta1.LPXConfig{
		BuildID:  "model/build",
		Settings: &apiextensionsv1.JSON{Raw: []byte(`{"scheduler":{"num_chips":4}}`)},
	}
}

func testLPXRoleTemplate(image string) corev1.PodTemplateSpec {
	return corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"role": image}},
		Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: image}}},
	}
}

func TestDynamoGraphDeploymentRoleShapesRoundTrip(t *testing.T) {
	t.Log("Cover absent, empty, and explicit roles without defaulting optional fields")
	tests := []struct {
		name  string
		alpha []ComponentRoleSpec
		beta  []v1beta1.ComponentRoleSpec
	}{
		{name: "absent"},
		{name: "empty", alpha: []ComponentRoleSpec{}, beta: []v1beta1.ComponentRoleSpec{}},
		{
			name: "native fields",
			alpha: []ComponentRoleSpec{
				{Name: v1beta1.ComponentRoleLeader},
				{
					Name:        v1beta1.ComponentRoleWorker,
					Replicas:    ptr.To[int32](3),
					PodTemplate: ptr.To(testLPXRoleTemplate("agent")),
				},
			},
			beta: []v1beta1.ComponentRoleSpec{
				{Name: v1beta1.ComponentRoleLeader},
				{
					Name:        v1beta1.ComponentRoleWorker,
					Replicas:    ptr.To[int32](3),
					PodTemplate: ptr.To(testLPXRoleTemplate("agent")),
				},
			},
		},
	}

	// Conversion preserves the live representation independently of admission rules.
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Convert authored alpha roles to beta without changing their shape")
			alpha := &DynamoGraphDeployment{
				Spec: DynamoGraphDeploymentSpec{Services: map[string]*DynamoComponentDeploymentSharedSpec{
					"lpx": {ComponentType: string(v1beta1.ComponentTypeLPX), Roles: tt.alpha},
				}},
			}
			beta := &v1beta1.DynamoGraphDeployment{}
			if err := alpha.ConvertTo(beta); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.beta, beta.Spec.Components[0].Roles); diff != "" {
				t.Fatalf("role conversion mismatch (-want +got):\n%s", diff)
			}

			t.Log("Convert beta roles back to alpha with the same order and optional values")
			roundTrip := &DynamoGraphDeployment{}
			if err := roundTrip.ConvertFrom(beta); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.alpha, roundTrip.Spec.Services["lpx"].Roles); diff != "" {
				t.Fatalf("role round-trip mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDynamoGraphDeploymentLPXRoundTrip(t *testing.T) {
	checkedAt := metav1.NewTime(time.Unix(1_800_000_000, 0))
	deadlineAt := metav1.NewTime(time.Unix(1_800_000_900, 0))
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "lpx", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Scheduling: &v1beta1.SchedulingSpec{
				AttemptDeadlineSeconds: ptr.To[int64](900),
			},
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "lpx",
				ComponentType: v1beta1.ComponentTypeLPX,
				LPX:           testCanonicalLPXConfig(),
				Roles: []v1beta1.ComponentRoleSpec{
					{Name: v1beta1.ComponentRoleLeader, Replicas: ptr.To[int32](2)},
					{Name: v1beta1.ComponentRoleWorker, PodTemplate: ptr.To(testLPXRoleTemplate("agent"))},
				},
			}},
		},
		Status: v1beta1.DynamoGraphDeploymentStatus{
			LPX: &v1beta1.DynamoGraphDeploymentLPXStatus{
				ModelDownload: &v1beta1.ModelDownloadStatus{
					Builds:        []string{"gs://models/draft", "gs://models/target"},
					LastCheckedAt: &checkedAt,
				},
				Placement: &v1beta1.PlacementStatus{LPXAttempt: &v1beta1.LPXAttemptStatus{
					ObservedGeneration: 2,
					PodCliqueSetUID:    "pcs-uid",
					DeadlineAt:         &deadlineAt,
					Requests: []v1beta1.LPXAttemptRequestStatus{{
						Name: "request", AttemptDigest: "sha256:attempt", UID: "request-uid",
					}},
				}},
			},
		},
	}

	t.Log("Convert the complete LPX configuration, scheduling, and status to alpha")
	original := src.DeepCopy()
	alpha := &DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	service := alpha.Spec.Services["lpx"]
	if service == nil || service.ComponentType != string(v1beta1.ComponentTypeLPX) {
		t.Fatalf("canonical LPX component did not convert to alpha: %#v", service)
	}
	if diff := cmp.Diff(testCanonicalLPXConfig(), service.LPX, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("LPX config conversion mismatch (-want +got):\n%s", diff)
	}
	wantAlphaRoles := []ComponentRoleSpec{
		{Name: v1beta1.ComponentRoleLeader, Replicas: ptr.To[int32](2)},
		{Name: v1beta1.ComponentRoleWorker, PodTemplate: ptr.To(testLPXRoleTemplate("agent"))},
	}
	if diff := cmp.Diff(wantAlphaRoles, service.Roles); diff != "" {
		t.Fatalf("LPX role order or optional fields changed in alpha (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(src.Spec.Scheduling, alpha.Spec.Scheduling); diff != "" {
		t.Fatalf("LPX scheduling conversion mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(src.Status.LPX, alpha.Status.LPX); diff != "" {
		t.Fatalf("LPX status conversion mismatch (-want +got):\n%s", diff)
	}

	t.Log("Serialize the native LPX payload under one status group in both versions")
	payload, err := json.Marshal(src.Status.LPX)
	if err != nil {
		t.Fatal(err)
	}
	for _, status := range []any{src.Status, alpha.Status} {
		raw, err := json.Marshal(status)
		if err != nil {
			t.Fatal(err)
		}
		if want := `{"state":"","lpx":` + string(payload) + `}`; string(raw) != want {
			t.Fatalf("%T JSON = %s, want %s", status, raw, want)
		}
	}

	t.Log("Edit the alpha configuration and round-trip the complete LPX payload")
	service.LPX = service.LPX.DeepCopy()
	service.LPX.BuildID = editedThroughAlphaBuildID
	service.Roles = []ComponentRoleSpec{
		*service.Roles[1].DeepCopy(),
		{Name: v1beta1.ComponentRoleLeader, PodTemplate: ptr.To(testLPXRoleTemplate("conductor"))},
	}

	got := &v1beta1.DynamoGraphDeployment{}
	if err := alpha.ConvertTo(got); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	component := got.GetComponentByName("lpx")
	if component == nil || component.LPX == nil || component.LPX.BuildID != editedThroughAlphaBuildID {
		t.Fatalf("alpha LPX edit was lost: %#v", component)
	}
	wantBetaRoles := []v1beta1.ComponentRoleSpec{
		{Name: v1beta1.ComponentRoleWorker, PodTemplate: ptr.To(testLPXRoleTemplate("agent"))},
		{Name: v1beta1.ComponentRoleLeader, PodTemplate: ptr.To(testLPXRoleTemplate("conductor"))},
	}
	if diff := cmp.Diff(wantBetaRoles, component.Roles); diff != "" {
		t.Fatalf("authored alpha role edits did not survive conversion (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(original, src); diff != "" {
		t.Fatalf("alpha edit mutated the source LPX deployment (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(src.Spec.Scheduling, got.Spec.Scheduling); diff != "" {
		t.Fatalf("LPX scheduling round-trip mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(src.Status, got.Status); diff != "" {
		t.Fatalf("LPX status round-trip mismatch (-want +got):\n%s", diff)
	}
	if _, exists := got.Annotations[annDGDSpec]; exists {
		t.Fatalf("canonical LPX spec must not use sparse conversion annotations: %v", got.Annotations)
	}
	if _, exists := got.Annotations[annDGDStatus]; exists {
		t.Fatalf("canonical LPX status must not use sparse conversion annotations: %v", got.Annotations)
	}

	t.Log("Preserve an explicitly empty LPX group through native conversion")
	alpha.Status.LPX = &v1beta1.DynamoGraphDeploymentLPXStatus{}
	if err := alpha.ConvertTo(got); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(alpha.Status.LPX, got.Status.LPX); diff != "" {
		t.Fatalf("empty LPX group changed (-want +got):\n%s", diff)
	}

	t.Log("Clearing live alpha LPX status removes the previously populated group")
	alpha.Status.LPX = nil
	if err := alpha.ConvertTo(got); err != nil {
		t.Fatal(err)
	}
	if got.Status.LPX != nil {
		t.Fatalf("cleared LPX status was restored: %#v", got.Status.LPX)
	}
}

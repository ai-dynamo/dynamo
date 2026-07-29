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

package v1alpha1

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// dgdsaRoundTripFromV1beta1 converts a v1beta1 DGDSA to v1alpha1 and back to
// v1beta1, asserting bitwise equality at the v1beta1 (hub) boundary.
func dgdsaRoundTripFromV1beta1(t *testing.T, src *v1beta1.DynamoGraphDeploymentScalingAdapter) *v1beta1.DynamoGraphDeploymentScalingAdapter {
	t.Helper()
	a := &DynamoGraphDeploymentScalingAdapter{}
	if err := a.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	out := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
	if err := a.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	return out
}

// dgdsaRoundTripFromV1alpha1 converts a v1alpha1 DGDSA to v1beta1 and back.
func dgdsaRoundTripFromV1alpha1(t *testing.T, src *DynamoGraphDeploymentScalingAdapter) *DynamoGraphDeploymentScalingAdapter {
	t.Helper()
	b := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	out := &DynamoGraphDeploymentScalingAdapter{}
	if err := out.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	return out
}

func TestDGDSA_RoundTripFromV1beta1_Minimal(t *testing.T) {
	src := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{Name: "adapter", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 3,
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "my-dgd",
				ComponentName: "decode",
			},
		},
	}
	got := dgdsaRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGDSA_RoundTripFromV1beta1_FullStatus(t *testing.T) {
	now := metav1.NewTime(time.Date(2026, 4, 26, 10, 30, 0, 0, time.UTC))
	src := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "adapter-full",
			Namespace:   "ns",
			Labels:      map[string]string{"team": "infra"},
			Annotations: map[string]string{"foo": "bar"},
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 5,
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "another-dgd",
				ComponentName: "prefill",
			},
		},
		Status: v1beta1.DynamoGraphDeploymentScalingAdapterStatus{
			Replicas:      5,
			Selector:      "app=prefill,dgd=another-dgd",
			LastScaleTime: &now,
		},
	}
	got := dgdsaRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGDSA_RoundTripFromV1alpha1_Minimal(t *testing.T) {
	src := &DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{Name: "adapter", Namespace: "ns"},
		Spec: DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 2,
			DGDRef: DynamoGraphDeploymentServiceRef{
				Name:        "my-dgd",
				ServiceName: "worker",
			},
		},
	}
	got := dgdsaRoundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGDSA_RoundTripFromV1alpha1_FullStatus(t *testing.T) {
	now := metav1.NewTime(time.Date(2026, 4, 26, 10, 30, 0, 0, time.UTC))
	src := &DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "adapter-full",
			Namespace:   "ns",
			Labels:      map[string]string{"team": "infra"},
			Annotations: map[string]string{"foo": "bar"},
		},
		Spec: DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 5,
			DGDRef: DynamoGraphDeploymentServiceRef{
				Name:        "another-dgd",
				ServiceName: "prefill",
			},
		},
		Status: DynamoGraphDeploymentScalingAdapterStatus{
			Replicas:      5,
			Selector:      "app=prefill,dgd=another-dgd",
			LastScaleTime: &now,
		},
	}
	got := dgdsaRoundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGDSA_ConvertTo_TypeError verifies the type-assert guard rejects an
// unexpected hub target without panicking.
func TestDGDSA_ConvertTo_TypeError(t *testing.T) {
	src := &DynamoGraphDeploymentScalingAdapter{}
	wrong := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(wrong); err == nil {
		t.Fatal("expected error from ConvertTo on wrong hub type, got nil")
	}
}

// TestDGDSA_ConvertFrom_TypeError mirrors the ConvertTo guard for ConvertFrom.
func TestDGDSA_ConvertFrom_TypeError(t *testing.T) {
	dst := &DynamoGraphDeploymentScalingAdapter{}
	wrong := &v1beta1.DynamoGraphDeployment{}
	if err := dst.ConvertFrom(wrong); err == nil {
		t.Fatal("expected error from ConvertFrom on wrong hub type, got nil")
	}
}

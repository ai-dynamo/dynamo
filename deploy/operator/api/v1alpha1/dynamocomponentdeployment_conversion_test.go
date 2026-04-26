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
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// dcdRoundTripFromV1alpha1 converts a v1alpha1 DCD to v1beta1 and back.
func dcdRoundTripFromV1alpha1(t *testing.T, src *DynamoComponentDeployment) *DynamoComponentDeployment {
	t.Helper()
	b := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	out := &DynamoComponentDeployment{}
	if err := out.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	return out
}

func dcdRoundTripFromV1beta1(t *testing.T, src *v1beta1.DynamoComponentDeployment) *v1beta1.DynamoComponentDeployment {
	t.Helper()
	a := &DynamoComponentDeployment{}
	if err := a.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	out := &v1beta1.DynamoComponentDeployment{}
	if err := a.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	return out
}

func TestDCD_RoundTrip_Empty(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty", Namespace: "ns"},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_RoundTrip_Minimal(t *testing.T) {
	replicas := int32(3)
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "min", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: v1beta1.ComponentTypeWorker,
				Replicas:      &replicas,
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_RoundTrip_PodTemplate(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pt", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "main",
								Image: "dynamo:latest",
								Resources: corev1.ResourceRequirements{
									Requests: corev1.ResourceList{
										corev1.ResourceCPU:                    resource.MustParse("2"),
										corev1.ResourceMemory:                 resource.MustParse("8Gi"),
										corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
									},
								},
							},
							{
								Name:  "logger",
								Image: "fluent/fluent-bit",
							},
						},
					},
				},
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_RoundTrip_Experimental(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "exp", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: v1beta1.ComponentTypeWorker,
				Experimental: &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
						Mode:            v1beta1.GMSModeIntraPod,
						DeviceClassName: "gpu.nvidia.com",
					},
				},
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// -----------------------------------------------------------------------------
// Expanded DCD coverage: status, v1alpha1-only shapes, scrubbing, JSON bytes.
// -----------------------------------------------------------------------------

// TestDCD_RoundTrip_Status exercises the DCD status fields (conditions and
// single-service replica status).
func TestDCD_RoundTrip_Status(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "status", Namespace: "ns"},
		Status: v1beta1.DynamoComponentDeploymentStatus{
			ObservedGeneration: 4,
			Conditions: []metav1.Condition{
				{Type: "Available", Status: metav1.ConditionTrue, Reason: "AllReady", Message: "ok"},
			},
			Service: &v1beta1.ServiceReplicaStatus{
				ComponentKind:   v1beta1.ComponentKindDeployment,
				ComponentName:   "dcd-0",
				Replicas:        3,
				UpdatedReplicas: 3,
				ReadyReplicas:   ptr.To(int32(3)),
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDCD_FromV1alpha1_AnnotationPreservedFields exercises the v1alpha1-only
// fields that are preserved verbatim via origin annotations on the DCD carrier
// (prefix "nvidia.com/dcd-"). Fields that flow through podTemplate
// decomposition (EnvFromSecret, Resources, VolumeMounts, Probes) are not
// bitwise round-trippable v1alpha1-first and are exercised via the
// v1beta1-first round-trip instead.
func TestDCD_FromV1alpha1_AnnotationPreservedFields(t *testing.T) {
	dynNs := "legacy-dyn-ns"
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "full", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ComponentType:    "worker",
				SubComponentType: "prefill",
				ServiceName:      "worker-svc",
				DynamoNamespace:  &dynNs,
				Annotations:      map[string]string{"team": "alpha"},
				Labels:           map[string]string{"tier": "gpu"},
				SharedMemory:     &SharedMemorySpec{Disabled: true},
				Autoscaling:      &Autoscaling{Enabled: true, MinReplicas: 1, MaxReplicas: 5},
				Ingress:          &IngressSpec{Enabled: true, Host: "api.example.com"},
				ScalingAdapter:   &ScalingAdapter{Enabled: false},
				GPUMemoryService: &GPUMemoryServiceSpec{Enabled: false, Mode: GMSModeIntraPod},
			},
		},
	}
	got := dcdRoundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDCD_ConvertFrom_ScrubsLingeringAnnotations checks that a stale
// "nvidia.com/dcd-*" annotation is dropped by ConvertFrom.
func TestDCD_ConvertFrom_ScrubsLingeringAnnotations(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "scrub",
			Namespace: "ns",
			Annotations: map[string]string{
				"nvidia.com/dcd-unknown-suffix": "stale",
				"user/keep":                     "kept",
			},
		},
	}
	a := &DynamoComponentDeployment{}
	if err := a.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if _, stale := a.ObjectMeta.Annotations["nvidia.com/dcd-unknown-suffix"]; stale {
		t.Errorf("stale dcd- annotation was not scrubbed: %v", a.ObjectMeta.Annotations)
	}
	if v, ok := a.ObjectMeta.Annotations["user/keep"]; !ok || v != "kept" {
		t.Errorf("user annotations must be preserved, got %v", a.ObjectMeta.Annotations)
	}
}

// TestDCD_JSONRoundTrip_Bytes asserts byte-identical JSON representation
// across a v1beta1 -> v1alpha1 -> v1beta1 round-trip.
func TestDCD_JSONRoundTrip_Bytes(t *testing.T) {
	shm := resource.MustParse("4Gi")
	replicas := int32(2)
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "json", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType:    v1beta1.ComponentTypeWorker,
				Replicas:         &replicas,
				SharedMemorySize: &shm,
				ScalingAdapter:   &v1beta1.ScalingAdapter{},
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)

	wantBytes, err := json.Marshal(src)
	if err != nil {
		t.Fatalf("marshal src: %v", err)
	}
	gotBytes, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal got: %v", err)
	}
	if string(wantBytes) != string(gotBytes) {
		t.Errorf("JSON byte-level round-trip mismatch:\nwant: %s\ngot:  %s", wantBytes, gotBytes)
	}
}

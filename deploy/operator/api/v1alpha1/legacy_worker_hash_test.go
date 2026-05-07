/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

const currentWorkerHashAnnotation = "nvidia.com/current-worker-hash"

func TestComputeDGDWorkersSpecHashGolden(t *testing.T) {
	tests := []struct {
		name string
		dgd  *DynamoGraphDeployment
		want string
	}{
		{
			name: "worker",
			dgd:  legacyWorkerHashDGD(),
			want: "9b66accc",
		},
		{
			name: "resource metadata and non-workers ignored",
			dgd:  legacyWorkerHashDGDWithResourceMetadataAndNonWorkerChanges(),
			want: "9b66accc",
		},
		{
			name: "pod metadata included",
			dgd:  legacyWorkerHashDGDWithPodMetadataChanges(),
			want: "af8a6c60",
		},
		{
			name: "all worker component types",
			dgd: &DynamoGraphDeployment{
				Spec: DynamoGraphDeploymentSpec{
					Services: map[string]*DynamoComponentDeploymentSharedSpec{
						"decode":  {ComponentType: commonconsts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "ROLE", Value: "decode"}}},
						"prefill": {ComponentType: commonconsts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "ROLE", Value: "prefill"}}},
						"worker":  {ComponentType: commonconsts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "ROLE", Value: "worker"}}},
					},
				},
			},
			want: "b175ee30",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got1, err := ComputeDGDWorkersSpecHash(tt.dgd)
			if err != nil {
				t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
			}
			got2, err := ComputeDGDWorkersSpecHash(tt.dgd)
			if err != nil {
				t.Fatalf("second ComputeDGDWorkersSpecHash: %v", err)
			}
			if got1 != got2 {
				t.Fatalf("hash is not stable: first %q, second %q", got1, got2)
			}
			if got1 != tt.want {
				t.Fatalf("ComputeDGDWorkersSpecHash() = %q, want golden %q", got1, tt.want)
			}
		})
	}
}

func TestComputeDGDWorkersSpecHashNil(t *testing.T) {
	if _, err := ComputeDGDWorkersSpecHash(nil); err == nil {
		t.Fatal("ComputeDGDWorkersSpecHash(nil) error = nil, want error")
	}
}

func TestDGDConvertToPreservesLegacyWorkerHashCompatibleWithControllerHash(t *testing.T) {
	src := legacyWorkerHashDGD()
	src.Annotations = map[string]string{
		currentWorkerHashAnnotation: "stale-controller-hash",
	}

	expected, err := ComputeDGDWorkersSpecHash(src)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
	}
	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	got := hub.Annotations[AnnotationDGDLegacyWorkerHash]
	if got != expected {
		t.Fatalf("%s = %q, want legacy hash %q", AnnotationDGDLegacyWorkerHash, got, expected)
	}
	if got == src.Annotations[currentWorkerHashAnnotation] {
		t.Fatalf("%s copied stale %s value instead of recomputing from spec", AnnotationDGDLegacyWorkerHash, currentWorkerHashAnnotation)
	}
}

func TestDGDConvertToSkipsLegacyWorkerHashWithoutCurrentWorkerHash(t *testing.T) {
	src := legacyWorkerHashDGD()

	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if _, ok := hub.Annotations[AnnotationDGDLegacyWorkerHash]; ok {
		t.Fatalf("unexpected %s annotation without %s trigger: %v", AnnotationDGDLegacyWorkerHash, currentWorkerHashAnnotation, hub.Annotations)
	}
}

func TestDGDLegacyWorkerHashTracksPodMetadata(t *testing.T) {
	baseHash := preservedLegacyWorkerHash(t, legacyWorkerHashDGD())

	mutated := legacyWorkerHashDGDWithPodMetadataChanges()

	if got := preservedLegacyWorkerHash(t, mutated); got == baseHash {
		t.Fatalf("pod metadata change did not change preserved legacy worker hash: %q", got)
	}
}

func TestDGDLegacyWorkerHashIgnoresResourceMetadataAndNonWorkers(t *testing.T) {
	baseHash := preservedLegacyWorkerHash(t, legacyWorkerHashDGD())

	mutated := legacyWorkerHashDGDWithResourceMetadataAndNonWorkerChanges()

	if got := preservedLegacyWorkerHash(t, mutated); got != baseHash {
		t.Fatalf("non-pod-template metadata/non-worker changes changed preserved legacy worker hash: got %q, want %q", got, baseHash)
	}
}

func TestDGDConvertFromDropsLegacyWorkerHash(t *testing.T) {
	hub := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "legacy-worker-hash",
			Namespace: "ns",
			Annotations: map[string]string{
				AnnotationDGDLegacyWorkerHash: "deadbeef",
				"user":                        "kept",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
				},
			},
		},
	}

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if _, ok := spoke.Annotations[AnnotationDGDLegacyWorkerHash]; ok {
		t.Fatalf("%s leaked back to v1alpha1 annotations: %v", AnnotationDGDLegacyWorkerHash, spoke.Annotations)
	}
	if got := spoke.Annotations["user"]; got != "kept" {
		t.Fatalf("user annotation = %q, want kept", got)
	}
}

func preservedLegacyWorkerHash(t *testing.T, src *DynamoGraphDeployment) string {
	t.Helper()
	if src.Annotations == nil {
		src.Annotations = map[string]string{}
	}
	src.Annotations[currentWorkerHashAnnotation] = "existing-controller-hash"

	expected, err := ComputeDGDWorkersSpecHash(src)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
	}
	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	got := hub.Annotations[AnnotationDGDLegacyWorkerHash]
	if got == "" {
		t.Fatalf("missing %s annotation in hub annotations: %v", AnnotationDGDLegacyWorkerHash, hub.Annotations)
	}
	if got != expected {
		t.Fatalf("%s = %q, want legacy hash %q", AnnotationDGDLegacyWorkerHash, got, expected)
	}
	return got
}

func legacyWorkerHashDGD() *DynamoGraphDeployment {
	return &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "legacy-worker-hash",
			Namespace: "ns",
		},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: commonconsts.ComponentTypeFrontend,
					Envs:          []corev1.EnvVar{{Name: "FRONTEND_ONLY", Value: "ignored"}},
				},
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					Annotations:   map[string]string{"resource": "base"},
					Labels:        map[string]string{"resource": "base"},
					Replicas:      ptr.To(int32(2)),
					Envs:          []corev1.EnvVar{{Name: "MODEL", Value: "llama"}},
					ExtraPodMetadata: &ExtraPodMetadata{
						Labels:      map[string]string{"rollout": "base"},
						Annotations: map[string]string{"checksum/config": "base"},
					},
				},
			},
		},
	}
}

func legacyWorkerHashDGDWithPodMetadataChanges() *DynamoGraphDeployment {
	mutated := legacyWorkerHashDGD()
	mutated.Spec.Services["worker"].ExtraPodMetadata = &ExtraPodMetadata{
		Labels:      map[string]string{"rollout": "changed"},
		Annotations: map[string]string{"checksum/config": "changed"},
	}
	return mutated
}

func legacyWorkerHashDGDWithResourceMetadataAndNonWorkerChanges() *DynamoGraphDeployment {
	mutated := legacyWorkerHashDGD()
	mutated.Spec.Services["worker"].Annotations = map[string]string{"resource": "changed"}
	mutated.Spec.Services["worker"].Labels = map[string]string{"resource": "changed"}
	mutated.Spec.Services["worker"].Replicas = ptr.To(int32(99))
	mutated.Spec.Services["worker"].Ingress = &IngressSpec{
		Enabled: true,
		Host:    "changed.example.com",
	}
	mutated.Spec.Services["frontend"].Envs = []corev1.EnvVar{{Name: "FRONTEND_ONLY", Value: "changed"}}
	return mutated
}

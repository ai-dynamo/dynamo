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

package controller

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
)

func TestFindLoRAModelsForWorkloadChangeScopesToLoRAModelsInNamespace(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add DynamoModel scheme: %v", err)
	}
	reconciler := &DynamoModelReconciler{Client: fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(
			&v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{Name: "lora-a", Namespace: "test"},
				Spec:       v1alpha1.DynamoModelSpec{ModelType: "lora"},
			},
			&v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{Name: "base", Namespace: "test"},
				Spec:       v1alpha1.DynamoModelSpec{ModelType: "base"},
			},
			&v1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{Name: "lora-other-ns", Namespace: "other"},
				Spec:       v1alpha1.DynamoModelSpec{ModelType: "lora"},
			},
		).
		Build()}

	requests := reconciler.findLoRAModelsForWorkloadChange(context.Background(), &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test"},
	})
	if len(requests) != 1 || requests[0].Name != "lora-a" || requests[0].Namespace != "test" {
		t.Fatalf("expected only test/lora-a to reconcile, got %#v", requests)
	}
}

func TestMarkLoRAManagementUnavailableFallbackEligible(t *testing.T) {
	candidates := []modelendpoint.Candidate{
		{PodName: "vllm-prefill", WorkloadName: "graph.vllm-prefill", PodIdentityResolved: true},
		{PodName: "vllm-decode", WorkloadName: "graph-vllm-decode", PodIdentityResolved: true},
		{PodName: "sglang-prefill", WorkloadName: "graph-sglang-prefill", PodIdentityResolved: true},
	}
	components := []v1beta1.DynamoComponentDeployment{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "graph.vllm-prefill"},
			Spec: v1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "vllm",
				DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "prefill",
					ComponentType: v1beta1.ComponentTypePrefill,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "graph-vllm-decode"},
			Spec: v1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "vllm",
				DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "decode",
					ComponentType: v1beta1.ComponentTypeDecode,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "graph-sglang-prefill"},
			Spec: v1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "sglang",
				DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "prefill",
					ComponentType: v1beta1.ComponentTypePrefill,
				},
			},
		},
	}

	classified := markLoRAManagementUnavailableFallbackEligible(candidates, components, nil)
	if !classified[0].AllowLoRAManagementUnavailable {
		t.Fatal("vLLM prefill must be eligible for the legacy missing-handler fallback")
	}
	if classified[0].LoRAFallbackGroup != "workload:graph.vllm-prefill" {
		t.Fatalf("unexpected workload fallback group: %#v", classified[0])
	}
	if classified[1].AllowLoRAManagementUnavailable {
		t.Fatal("vLLM decode must remain eligible for LoRA lifecycle calls")
	}
	if classified[2].AllowLoRAManagementUnavailable {
		t.Fatal("SGLang prefill must remain eligible for LoRA lifecycle calls")
	}
	if candidates[0].AllowLoRAManagementUnavailable {
		t.Fatal("classification must not mutate the input candidates")
	}
}

func TestMarkLoRAManagementUnavailableFallbackEligibleForGrove(t *testing.T) {
	candidates := []modelendpoint.Candidate{{
		PodName:             "grove-prefill-0",
		WorkloadName:        "grove-prefill",
		GraphDeploymentName: "grove",
		ComponentName:       "prefill",
	}}
	graphs := []v1beta1.DynamoGraphDeployment{{
		ObjectMeta: metav1.ObjectMeta{Name: "grove"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "prefill",
				ComponentType: v1beta1.ComponentTypePrefill,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:    v1beta1.MainContainerName,
					Command: []string{"python3", "-m", "dynamo.vllm"},
				}}}},
			}},
		},
	}}

	classified := markLoRAManagementUnavailableFallbackEligible(candidates, nil, graphs)
	if !classified[0].AllowLoRAManagementUnavailable {
		t.Fatal("Grove vLLM prefill must be eligible for the legacy missing-handler fallback")
	}
	if classified[0].LoRAFallbackGroup != "graph:grove" {
		t.Fatalf("unexpected graph fallback group: %#v", classified[0])
	}
}

func TestMarkLoRAManagementUnavailableFallbackPreservesExactDCDIdentity(t *testing.T) {
	candidates := []modelendpoint.Candidate{{
		PodName:             "sglang-prefill",
		WorkloadName:        "foo-bar",
		PodIdentityResolved: true,
	}}
	components := []v1beta1.DynamoComponentDeployment{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo.bar"},
			Spec: v1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "vllm",
				DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "prefill",
					ComponentType: v1beta1.ComponentTypePrefill,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-bar"},
			Spec: v1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "sglang",
				DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "prefill",
					ComponentType: v1beta1.ComponentTypePrefill,
				},
			},
		},
	}

	classified := markLoRAManagementUnavailableFallbackEligible(candidates, components, nil)
	if classified[0].AllowLoRAManagementUnavailable {
		t.Fatal("exact SGLang DCD identity must remain eligible for lifecycle probing")
	}
	if classified[0].LoRAFallbackGroup != "" {
		t.Fatalf("non-vLLM prefill must not receive a fallback group: %#v", classified[0])
	}
}

func TestMarkLoRAManagementUnavailableFallbackEligibleForLWSOwner(t *testing.T) {
	candidates := []modelendpoint.Candidate{{
		PodName:             "prefill-0",
		WorkloadName:        "standalone.prefill-0",
		PodIdentityResolved: true,
		ControllerOwnerKind: "LeaderWorkerSet",
	}}
	components := []v1beta1.DynamoComponentDeployment{{
		ObjectMeta: metav1.ObjectMeta{Name: "standalone.prefill"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "prefill",
				ComponentType: v1beta1.ComponentTypePrefill,
			},
		},
	}}

	classified := markLoRAManagementUnavailableFallbackEligible(candidates, components, nil)
	if !classified[0].AllowLoRAManagementUnavailable {
		t.Fatal("LWS vLLM prefill owner identity must allow the legacy fallback")
	}
	if classified[0].LoRAFallbackGroup != "workload:standalone.prefill-0" {
		t.Fatalf("unexpected LWS fallback group: %#v", classified[0])
	}
}

func TestWithPodIdentityFillsSharedModelSliceMetadata(t *testing.T) {
	candidate := modelendpoint.Candidate{PodName: "prefill-0"}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
		consts.KubeLabelDynamoSelector:            "graph-prefill-hash",
		consts.KubeLabelDynamoComponent:           "prefill",
		consts.KubeLabelDynamoGraphDeploymentName: "graph",
	}}}

	identified := withPodIdentity(candidate, pod)
	if identified.WorkloadName != "graph-prefill-hash" || identified.ComponentName != "prefill" ||
		identified.GraphDeploymentName != "graph" || !identified.PodIdentityResolved || identified.ControllerOwnerKind != "" {
		t.Fatalf("expected pod identity fallback, got %#v", identified)
	}
	if candidate.WorkloadName != "" || candidate.ComponentName != "" || candidate.GraphDeploymentName != "" {
		t.Fatal("pod identity fallback must not mutate the input candidate")
	}
}

func TestWithPodIdentityUsesControllerOwnerWhenSelectorIsAbsent(t *testing.T) {
	controller := true
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Labels: map[string]string{
			consts.KubeLabelDynamoComponent:           "prefill",
			consts.KubeLabelDynamoGraphDeploymentName: "graph",
		},
		OwnerReferences: []metav1.OwnerReference{{
			Kind:       "LeaderWorkerSet",
			Name:       "graph-prefill-hash",
			Controller: &controller,
		}},
	}}

	identified := withPodIdentity(modelendpoint.Candidate{PodName: "prefill-0"}, pod)
	if identified.WorkloadName != "graph-prefill-hash" || identified.ControllerOwnerKind != "LeaderWorkerSet" ||
		!identified.PodIdentityResolved {
		t.Fatalf("expected controller owner identity fallback, got %#v", identified)
	}
}

func TestWithPodIdentityUsesActualPodReadiness(t *testing.T) {
	tests := []struct {
		name      string
		condition corev1.ConditionStatus
		ready     bool
	}{
		{name: "ready", condition: corev1.ConditionTrue, ready: true},
		{name: "not ready", condition: corev1.ConditionFalse, ready: false},
		{name: "unknown", condition: corev1.ConditionUnknown, ready: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{
				Type:   corev1.PodReady,
				Status: tt.condition,
			}}}}
			identified := withPodIdentity(modelendpoint.Candidate{KubernetesReady: !tt.ready}, pod)
			if identified.KubernetesReady != tt.ready {
				t.Fatalf("expected readiness %t from Pod condition, got %#v", tt.ready, identified)
			}
		})
	}
}

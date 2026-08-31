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
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
)

func TestWithPodIdentityClassifiesLoRAFallback(t *testing.T) {
	tests := []struct {
		name      string
		labels    map[string]string
		command   []string
		args      []string
		wantAllow bool
		wantGroup string
	}{
		{
			name: "vLLM prefill in graph",
			labels: map[string]string{
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypePrefill,
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
			},
			command:   []string{"python3", "-m", "dynamo.vllm"},
			wantAllow: true,
			wantGroup: "graph:graph",
		},
		{
			name: "legacy vLLM prefill workload",
			labels: map[string]string{
				consts.KubeLabelDynamoComponentType:    consts.ComponentTypeWorker,
				consts.KubeLabelDynamoSubComponentType: consts.ComponentTypePrefill,
				consts.KubeLabelDynamoSelector:         "standalone-prefill",
			},
			command:   []string{"python3"},
			args:      []string{"-m", "dynamo.vllm"},
			wantAllow: true,
			wantGroup: "workload:standalone-prefill",
		},
		{
			name: "vLLM decode",
			labels: map[string]string{
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypeDecode,
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
			},
			command: []string{"python3", "-m", "dynamo.vllm"},
		},
		{
			name: "SGLang prefill",
			labels: map[string]string{
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypePrefill,
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
			},
			command: []string{"python3", "-m", "dynamo.sglang"},
		},
		{
			name: "vLLM prefill without topology",
			labels: map[string]string{
				consts.KubeLabelDynamoComponentType: consts.ComponentTypePrefill,
			},
			command: []string{"python3", "-m", "dynamo.vllm"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Labels: tt.labels},
				Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:    consts.MainContainerName,
					Command: tt.command,
					Args:    tt.args,
				}}},
			}
			identified := withPodIdentity(modelendpoint.Candidate{}, pod)
			if identified.AllowLoRAManagementUnavailable != tt.wantAllow || identified.LoRAFallbackGroup != tt.wantGroup {
				t.Fatalf("unexpected fallback classification: %#v", identified)
			}
		})
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
	if identified.WorkloadName != "graph-prefill-hash" || identified.GraphDeploymentName != "graph" {
		t.Fatalf("expected pod identity fallback, got %#v", identified)
	}
	if candidate.WorkloadName != "" || candidate.GraphDeploymentName != "" {
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
	if identified.WorkloadName != "graph-prefill-hash" {
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

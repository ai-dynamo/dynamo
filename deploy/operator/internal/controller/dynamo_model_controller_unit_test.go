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
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
)

func TestModelEndpointLifecycleUsesDiscoveredSystemPort(t *testing.T) {
	t.Log("Run a model API on an allocated TCP port and publish it through an EndpointSlice")
	requests := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Method + " " + r.URL.Path
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)
	address := server.Listener.Addr().(*net.TCPAddr)
	model := &v1alpha1.DynamoModel{
		Namespace: "inference",
		Spec: v1alpha1.DynamoModelSpec{
			ModelName: "adapter", BaseModelName: "base-model", ModelType: "lora",
			Source: &v1alpha1.ModelSource{URI: "s3://bucket/adapter"},
		},
	}
	pod := &corev1.Pod{
		Name: "worker", Namespace: model.Namespace, UID: "worker-uid",
		Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}},
	}
	slice := &discoveryv1.EndpointSlice{
		Name: "model-endpoints", Namespace: model.Namespace,
		Labels: map[string]string{
			consts.KubeLabelDynamoBaseModelHash: dynamo.HashModelName(model.Spec.BaseModelName),
			discoveryv1.LabelServiceName:        "model-service",
		},
		Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(address.Port))}},
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{address.IP.String()},
			TargetRef: &corev1.ObjectReference{Kind: "Pod", Name: pod.Name, UID: pod.UID},
		}},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, discoveryv1.AddToScheme(scheme))
	reconciler := &DynamoModelReconciler{
		Client:         fake.NewClientBuilder().WithScheme(scheme).WithObjects(slice, pod).Build(),
		EndpointClient: modelendpoint.NewClient(),
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	t.Cleanup(cancel)

	t.Log("Discover the actual listener and keep Pod readiness for LoRA classification")
	candidates, services, err := reconciler.getEndpointCandidates(ctx, model)
	require.NoError(t, err)
	require.Equal(t, map[string]bool{"model-service": true}, services)
	require.Len(t, candidates, 1)
	require.Equal(t, server.URL, candidates[0].Address)
	require.True(t, candidates[0].KubernetesReady)

	t.Log("Load and unload the adapter against the discovered port")
	endpoints, err := reconciler.EndpointClient.LoadLoRA(ctx, candidates, model)
	require.NoError(t, err)
	require.Len(t, endpoints, 1)
	require.True(t, endpoints[0].Ready)
	require.NoError(t, reconciler.EndpointClient.UnloadLoRA(ctx, candidates, model.Spec.ModelName))
	require.Len(t, requests, 2)
	require.Equal(t, "POST /v1/loras", <-requests)
	require.Equal(t, "DELETE /v1/loras/adapter", <-requests)
}

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
				Labels: tt.labels,
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
	pod := &corev1.Pod{Labels: map[string]string{
		consts.KubeLabelDynamoSelector:            "graph-prefill-hash",
		consts.KubeLabelDynamoComponent:           "prefill",
		consts.KubeLabelDynamoGraphDeploymentName: "graph",
	}}

	identified := withPodIdentity(candidate, pod)
	if identified.WorkloadName != "graph-prefill-hash" || identified.GraphDeploymentName != "graph" {
		t.Fatalf("expected pod identity fallback, got %#v", identified)
	}
	if candidate.WorkloadName != "" || candidate.GraphDeploymentName != "" {
		t.Fatal("pod identity fallback must not mutate the input candidate")
	}
}

// TestWithPodIdentityClassifiesNativeSidecarLoRAFallback uses the engine
// launch independently of the runtime to decide whether prefill fallback is
// supported. Only Dynamo launchers are detected; native engine commands such as
// vllm-rs stay undetected (#15463, #17543).
func TestWithPodIdentityClassifiesNativeSidecarLoRAFallback(t *testing.T) {
	cases := []struct {
		name    string
		command []string
		engine  []string
		allow   bool
	}{
		{name: "vLLM launcher prefill", engine: []string{"python3", "-m", "dynamo.vllm"}, command: []string{"dynamo-vllm-sidecar"}, allow: true},
		{name: "native vLLM engine is not detected", engine: []string{"vllm-rs"}, command: []string{"dynamo-vllm-sidecar"}},
		{name: "SGLang engine", engine: []string{"python3", "-m", "dynamo.sglang"}, command: []string{"dynamo-vllm-sidecar"}},
		{name: "opaque engine", command: []string{"dynamo-vllm-sidecar"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Configure a prefill Pod with separate engine and runtime launches")
			pod := &corev1.Pod{
				Labels: map[string]string{
					consts.KubeLabelDynamoComponentType: consts.ComponentTypePrefill, consts.KubeLabelDynamoGraphDeploymentName: "graph",
				},
				Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: consts.MainContainerName, Command: tc.engine}},
					InitContainers: []corev1.Container{{Name: "runtime", Command: tc.command}},
				},
			}
			original := pod.DeepCopy()

			t.Log("Classify LoRA fallback from the engine without mutating the Pod")
			identified := withPodIdentity(modelendpoint.Candidate{}, pod)
			require.Equal(t, tc.allow, identified.AllowLoRAManagementUnavailable)
			if tc.allow {
				require.Equal(t, "graph:graph", identified.LoRAFallbackGroup)
			} else {
				require.Empty(t, identified.LoRAFallbackGroup)
			}
			require.Equal(t, original, pod)
		})
	}
}

func TestWithPodIdentityUsesControllerOwnerWhenSelectorIsAbsent(t *testing.T) {
	controller := true
	pod := &corev1.Pod{
		Labels: map[string]string{
			consts.KubeLabelDynamoComponent:           "prefill",
			consts.KubeLabelDynamoGraphDeploymentName: "graph",
		},
		OwnerReferences: []metav1.OwnerReference{{
			Kind:       "LeaderWorkerSet",
			Name:       "graph-prefill-hash",
			Controller: &controller,
		}}}

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

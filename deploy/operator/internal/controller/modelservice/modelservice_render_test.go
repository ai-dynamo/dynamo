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

package modelservice

import (
	"maps"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestNamePreservesLegacyAndValidGraphNames(t *testing.T) {
	for _, tc := range []struct {
		name     string
		graph    string
		readable bool
		want     string
	}{
		{name: "legacy shared", want: "dynamo-model-e3352e1e"},
		{name: "legacy graph", graph: "graph-a", want: "dynamo-model-e3352e1e-d8213feb"},
		{name: "readable graph", graph: "graph-a", readable: true, want: "graph-a-model-e3352e1e"},
		{name: "maximum readable length", graph: strings.Repeat("a", 48), readable: true, want: strings.Repeat("a", 48) + "-model-e3352e1e"},
		{name: "over maximum readable length", graph: strings.Repeat("a", 49), readable: true, want: "dynamo-model-e3352e1e-8f9bec6a"},
		{name: "dotted graph retains identity", graph: "graph-a.worker", readable: true, want: "dynamo-model-e3352e1e-62f60e06"},
		{name: "digit leading graph", graph: "1graph", readable: true, want: "dynamo-model-e3352e1e-01af4e81"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Derive a model Service name in the selected compatibility mode")
			name := Name("llama-3", tc.graph, tc.readable)

			t.Log("Keep the expected identity and satisfy Service naming validation")
			require.Equal(t, tc.want, name)
			require.Empty(t, validation.IsDNS1035Label(name))
			require.Equal(t, name, Name("llama-3", tc.graph, tc.readable))
		})
	}
}

func TestRenderScopesDiscoveryWithoutChangingItsProtocol(t *testing.T) {
	for _, tc := range []struct {
		name     string
		graph    string
		readable bool
	}{
		{name: "shared legacy"},
		{name: "graph legacy", graph: "graph-a"},
		{name: "graph readable", graph: "graph-a", readable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Render the model system-API Service from resolved identity")
			service := Render(Params{
				Namespace: "inference", ModelName: "llama-3", GraphName: tc.graph, ReadableName: tc.readable,
			})

			t.Log("Retain model discovery and expose only worker system endpoints")
			wantLabels := map[string]string{
				consts.KubeLabelDynamoBaseModelHash: "e3352e1e",
				"nvidia.com/managed-by":             "dynamo-operator",
			}
			wantSelector := map[string]string{consts.KubeLabelDynamoBaseModelHash: "e3352e1e"}
			if tc.graph != "" {
				wantLabels[consts.KubeLabelDynamoGraphDeploymentName] = tc.graph
				wantSelector[consts.KubeLabelDynamoGraphDeploymentName] = tc.graph
			}
			require.Equal(t, "inference", service.Namespace)
			require.Equal(t, wantLabels, service.Labels)
			require.Equal(t, corev1.ServiceSpec{
				ClusterIP: corev1.ClusterIPNone,
				Selector:  wantSelector,
				Ports: []corev1.ServicePort{{
					Name: "system", Port: 9090, TargetPort: intstr.FromInt32(9090), Protocol: corev1.ProtocolTCP,
				}},
			}, service.Spec)
			require.Equal(t, map[string]string{consts.KubeAnnotationDynamoBaseModel: "llama-3"}, service.Annotations)
			require.Empty(t, service.OwnerReferences, "the caller establishes ownership at persistence")
		})
	}
}

func TestRenderKeepsGraphAndModelIdentitiesDistinct(t *testing.T) {
	t.Log("Render two graphs serving one model and a second model in the first graph")
	first := Render(Params{Namespace: "inference", ModelName: "llama-3", GraphName: "graph-a", ReadableName: true})
	second := Render(Params{Namespace: "inference", ModelName: "llama-3", GraphName: "graph-b", ReadableName: true})
	otherModel := Render(Params{Namespace: "inference", ModelName: "Qwen/Qwen3-0.6B", GraphName: "graph-a", ReadableName: true})

	t.Log("Keep Services private to their graph while preserving model-wide discovery labels")
	require.NotEqual(t, first.Name, second.Name)
	require.NotEqual(t, first.Name, otherModel.Name)
	require.NotEqual(t, second.Name, otherModel.Name)
	require.Equal(t, first.Labels[consts.KubeLabelDynamoBaseModelHash], second.Labels[consts.KubeLabelDynamoBaseModelHash])
	require.NotEqual(t, first.Spec.Selector, second.Spec.Selector)
	require.Equal(t, "graph-a", first.Spec.Selector[consts.KubeLabelDynamoGraphDeploymentName])
	require.Equal(t, "graph-b", second.Spec.Selector[consts.KubeLabelDynamoGraphDeploymentName])
	require.Equal(t, "fb68f5a6", otherModel.Spec.Selector[consts.KubeLabelDynamoBaseModelHash])
}

func TestRenderDoesNotMutateOrAliasInputs(t *testing.T) {
	t.Log("Provide authored annotations including a conflicting model identity")
	annotations := map[string]string{
		"example.com/team":                   "serving",
		consts.KubeAnnotationDynamoBaseModel: "wrong-model",
	}
	original := maps.Clone(annotations)
	service := Render(Params{Namespace: "inference", ModelName: "llama-3", GraphName: "graph-a", Annotations: annotations, ReadableName: true})
	require.Equal(t, original, annotations)
	require.Equal(t, "llama-3", service.Annotations[consts.KubeAnnotationDynamoBaseModel])
	require.Equal(t, "serving", service.Annotations["example.com/team"])

	t.Log("Keep rendered annotations independent from the caller's mutable metadata")
	service.Annotations["example.com/team"] = "rendered"
	require.Equal(t, original, annotations)
	annotations["example.com/new"] = "input"
	require.NotContains(t, service.Annotations, "example.com/new")

	t.Log("Keep Service labels independent from the Pod selector")
	service.Labels[consts.KubeLabelDynamoBaseModelHash] = "changed"
	require.Equal(t, "e3352e1e", service.Spec.Selector[consts.KubeLabelDynamoBaseModelHash])
}

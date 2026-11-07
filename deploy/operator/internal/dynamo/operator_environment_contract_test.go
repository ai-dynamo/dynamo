// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"encoding/json"
	"maps"
	"os"
	"slices"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestPlannerOwnedEnvironmentContract(t *testing.T) {
	t.Log("Load the same ownership cases consumed by Python's pure projection tests")
	data, err := os.ReadFile("testdata/operator_environment.json")
	require.NoError(t, err)
	var cases []struct {
		Name      string             `json:"name"`
		Main      json.RawMessage    `json:"main"`
		Component json.RawMessage    `json:"component"`
		Graph     json.RawMessage    `json:"graph"`
		Role      Role               `json:"role"`
		Backend   BackendFramework   `json:"backend"`
		Bindings  map[string]*string `json:"bindings"`
		Omitted   []string           `json:"omitted"`
		Preserved []string           `json:"preserved"`
		Error     string             `json:"error"`
	}
	require.NoError(t, json.Unmarshal(data, &cases))

	t.Log("Every case must preserve every known name it does not explicitly own or omit")
	known := make(map[string]struct{})
	for _, test := range cases {
		for name := range test.Bindings {
			known[name] = struct{}{}
		}
		for _, name := range slices.Concat(test.Omitted, test.Preserved) {
			known[name] = struct{}{}
		}
	}
	names := slices.Sorted(maps.Keys(known))
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			t.Log("Build the authored component without normalizing away its environment sources")
			main := corev1.Container{Name: "main", Image: "engine:1.8.0", Command: []string{"python3", "-m", "dynamo.vllm"},
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
			}
			if test.Main != nil {
				require.NoError(t, json.Unmarshal(test.Main, &main))
			}
			component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker}
			if test.Component != nil {
				require.NoError(t, json.Unmarshal(test.Component, &component))
			}
			if !HasRolePodTemplates(&component) {
				if component.PodTemplate == nil {
					component.PodTemplate = &corev1.PodTemplateSpec{}
				}
				component.PodTemplate.Spec.Containers = []corev1.Container{main}
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
			}
			if test.Graph != nil {
				require.NoError(t, json.Unmarshal(test.Graph, &dgd.Spec))
			}
			dgd.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{component}

			t.Log("Resolve the backend across the complete component, before selecting a role")
			backend, err := getBackendFrameworkFromComponent(&component, dgd)
			if test.Error != "" {
				require.ErrorContains(t, err, test.Error)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.Backend, backend)
			role := test.Role
			if role == "" {
				role = RoleMain
			}
			selected, ownership, native, err := lowerComponentForRole(&component, role)
			require.NoError(t, err)
			authored := GetMainContainer(selected)
			require.NotNil(t, authored)

			t.Log("Seed every known name without replacing authored references or Secrets, then render")
			for _, name := range names {
				if !slices.ContainsFunc(authored.Env, func(variable corev1.EnvVar) bool { return variable.Name == name }) {
					authored.Env = append(authored.Env, corev1.EnvVar{Name: name, Value: "__authored__"})
				}
			}
			original := authored.DeepCopy()
			nodes := int32(1)
			if selected.Multinode != nil {
				nodes = selected.Multinode.NodeCount
			}
			pod, err := generatePodSpecForRole(ServiceRole{Name: "worker", Role: role}, selected, backend, nil, dgd, nodes,
				&configv1alpha1.OperatorConfiguration{}, "worker", ownership, native, nil, staticContainerGPUCount(1))
			require.NoError(t, err)
			environment := pod.Containers[0].Env
			for name, expected := range test.Bindings {
				actual := findEnvVar(environment, name)
				require.NotNil(t, actual, "owned binding %s", name)
				if expected != nil {
					require.Equal(t, *expected, actual.Value, "owned binding %s", name)
					require.Nil(t, actual.ValueFrom)
				} else {
					require.NotEqual(t, "__authored__", actual.Value, "owned binding %s", name)
				}
			}
			for _, name := range test.Omitted {
				require.Nil(t, findEnvVar(environment, name), "omitted binding %s", name)
			}
			for _, name := range names {
				_, owned := test.Bindings[name]
				if !owned && !slices.Contains(test.Omitted, name) {
					require.Equal(t, findEnvVar(original.Env, name), findEnvVar(environment, name), "unowned binding %s", name)
				}
			}
		})
	}
}

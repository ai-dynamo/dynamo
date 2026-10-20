// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"path/filepath"
	"strings"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// Canonical matrix paths must exercise the named backend, topology, and provider.
func checkUpgradeGroveMatrix(t *testing.T, relative string, dgd *api.DynamoGraphDeployment, objects []unstructured.Unstructured) {
	t.Helper()
	parts := strings.Split(filepath.ToSlash(relative), "/")
	if len(parts) != 5 || parts[0] != "latest" || parts[4] != "grove" {
		return
	}
	t.Log("Check the canonical Grove topology independently of the saved golden")
	require.Equal(t, parts[1], string(dgd.Spec.BackendFramework))
	require.Equal(t, "grove", dgd.Annotations["nvidia.com/workload-provider"])
	expected := map[string]int{"frontend": 1}
	workers := []string{"worker"}
	if parts[2] == "disaggregated" {
		workers = []string{"prefill", "decode"}
	}
	for _, role := range workers {
		expected[role] = 1
		if parts[3] == "multi" {
			expected[role] = 2
		}
	}
	require.Len(t, dgd.Spec.Components, len(expected))
	for _, component := range dgd.Spec.Components {
		count, found := expected[component.ComponentName]
		require.True(t, found, "unexpected component %s", component.ComponentName)
		require.Equal(t, component.ComponentName, string(component.ComponentType))
		if count == 2 {
			require.NotNil(t, component.Multinode)
			require.EqualValues(t, 2, component.Multinode.NodeCount)
		} else {
			require.Nil(t, component.Multinode)
		}
	}
	observed := map[string]int{}
	pcs := 0
	for _, object := range objects {
		require.NotContains(t, []string{"DynamoComponentDeployment", "Deployment", "LeaderWorkerSet"}, object.GetKind(), "Grove scenario selected another workload program")
		if object.GetKind() != upgradePodCliqueSetKind {
			continue
		}
		pcs++
		cliques, found, err := unstructured.NestedSlice(object.Object, "spec", "template", "cliques")
		require.NoError(t, err)
		require.True(t, found)
		for _, value := range cliques {
			clique := value.(map[string]any)
			labels := clique["labels"].(map[string]any)
			observed[labels["nvidia.com/dynamo-component"].(string)]++
		}
	}
	require.Equal(t, 1, pcs)
	require.Equal(t, expected, observed, "multinode workers must lower into distinct leader/worker cliques")
}

//go:build clustertest && upgradecontract

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

// Canonical Component entries must exercise the named backend and topology.
func checkUpgradeComponentMatrix(t *testing.T, relative string, dgd *api.DynamoGraphDeployment, objects []unstructured.Unstructured) {
	t.Helper()
	parts := strings.Split(filepath.ToSlash(relative), "/")
	if len(parts) != 5 || parts[0] != "latest" || parts[4] != "component" {
		return
	}

	t.Log("Check the canonical Component program and topology independently of golden updates")
	require.Equal(t, parts[1], dgd.Spec.BackendFramework)
	require.Equal(t, "component", dgd.Annotations["nvidia.com/workload-provider"])
	expected := map[string]int{"frontend": 1, "worker": 1}
	if parts[2] == "disaggregated" {
		expected = map[string]int{"frontend": 1, "prefill": 1, "decode": 1}
	}
	if parts[3] == "multi" {
		for role := range expected {
			if role != "frontend" {
				expected[role] = 2
			}
		}
	}
	require.Len(t, dgd.Spec.Components, len(expected))
	seen := map[string]bool{}
	for _, component := range dgd.Spec.Components {
		nodes, found := expected[component.ComponentName]
		require.True(t, found, "unexpected component %s", component.ComponentName)
		require.NotContains(t, seen, component.ComponentName)
		seen[component.ComponentName] = true
		require.Equal(t, component.ComponentName, string(component.ComponentType))
		if nodes == 2 {
			require.NotNil(t, component.Multinode)
			require.EqualValues(t, nodes, component.Multinode.NodeCount)
		} else {
			require.Nil(t, component.Multinode)
		}
	}

	// The child barrier verifies the DCD ownership boundary and expected workload kind.
	require.NoError(t, upgradeComponentGraphReady(objects, len(expected)))
	children, workloads := map[string]bool{}, map[string]bool{}
	for _, object := range objects {
		require.NotEqual(t, "DisaggregatedSet", object.GetKind())
		role := object.GetLabels()["nvidia.com/dynamo-component"]
		if object.GetKind() == "DynamoComponentDeployment" {
			nodes, found := expected[role]
			require.True(t, found, "unexpected DCD component %s", role)
			require.NotContains(t, children, role)
			children[role] = true
			actual, _, err := unstructured.NestedInt64(object.Object, "spec", "multinode", "nodeCount")
			require.NoError(t, err)
			if nodes == 2 {
				require.EqualValues(t, nodes, actual)
			} else {
				require.LessOrEqual(t, actual, int64(1))
			}
		}
		if object.GetKind() == "Deployment" || object.GetKind() == "LeaderWorkerSet" {
			nodes, found := expected[role]
			require.True(t, found, "unexpected workload component %s", role)
			require.NotContains(t, workloads, role)
			workloads[role] = true
			kind := "Deployment"
			if nodes == 2 {
				kind = "LeaderWorkerSet"
			}
			require.Equal(t, kind, object.GetKind())
		}
		if object.GetKind() != "LeaderWorkerSet" {
			continue
		}
		size, found, err := unstructured.NestedInt64(object.Object, "spec", "leaderWorkerTemplate", "size")
		require.NoError(t, err)
		require.True(t, found)
		require.EqualValues(t, 2, size)
		for _, role := range []string{"leaderTemplate", "workerTemplate"} {
			_, found, err := unstructured.NestedMap(object.Object, "spec", "leaderWorkerTemplate", role, "spec")
			require.NoError(t, err)
			require.True(t, found, "missing LWS %s", role)
		}
	}
	require.Len(t, children, len(expected))
	require.Len(t, workloads, len(expected))
}

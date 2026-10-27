//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"path/filepath"
	"strings"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func upgradeExpectedDCDs(dgd *api.DynamoGraphDeployment) int {
	count := 0
	for _, component := range dgd.Spec.Components {
		if dgd.Annotations["nvidia.com/workload-provider"] != "disaggregatedset" || !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			count++
		}
	}
	return count
}

// Canonical paths must select their provider and topology independently of accepted goldens.
func checkUpgradeWorkloadMatrix(t *testing.T, relative string, dgd *api.DynamoGraphDeployment, objects []unstructured.Unstructured) {
	t.Helper()
	parts := strings.Split(filepath.ToSlash(relative), "/")
	if parts[0] == "history" {
		parts = parts[1:]
	}
	if len(parts) != 5 || parts[4] == "grove" {
		return
	}
	t.Log("Check the canonical Component/DisaggregatedSet topology independently of the golden")
	require.Equal(t, parts[1], dgd.Spec.BackendFramework)
	require.Equal(t, parts[4], dgd.Annotations["nvidia.com/workload-provider"])
	workers := []string{"worker"}
	if parts[2] == "disaggregated" {
		workers = []string{"prefill", "decode"}
	}
	require.Len(t, dgd.Spec.Components, len(workers)+1)
	for _, component := range dgd.Spec.Components {
		if component.ComponentType == api.ComponentTypeFrontend {
			require.Equal(t, "frontend", component.ComponentName)
			require.Nil(t, component.Multinode)
			continue
		}
		require.Contains(t, workers, component.ComponentName)
		require.Equal(t, component.ComponentName, string(component.ComponentType))
		if parts[3] == "multi" {
			require.NotNil(t, component.Multinode)
			require.EqualValues(t, 2, component.Multinode.NodeCount)
		} else {
			require.Nil(t, component.Multinode)
		}
	}

	// Only the operator's immediate workloads belong to this inventory.
	counts := map[string]int{}
	for _, object := range objects {
		counts[object.GetKind()]++
		if object.GetKind() != "DisaggregatedSet" {
			continue
		}
		require.True(t, metav1.IsControlledBy(&object, dgd))
		require.NoError(t, upgradeDSRolesError(dgd, object))
	}
	require.Zero(t, counts["PodCliqueSet"])
	if parts[4] == "disaggregatedset" {
		require.Equal(t, "disaggregated", parts[2])
		require.Equal(t, "multi", parts[3])
		require.Equal(t, 1, counts["DisaggregatedSet"])
		require.Equal(t, 1, counts["DynamoComponentDeployment"])
		require.Equal(t, 1, counts["Deployment"])
		require.Zero(t, counts["LeaderWorkerSet"], "DS descendants must not be actuated or captured")
	} else {
		require.Zero(t, counts["DisaggregatedSet"])
		require.Equal(t, len(workers)+1, counts["DynamoComponentDeployment"])
		if parts[3] == "multi" {
			require.Equal(t, len(workers), counts["LeaderWorkerSet"])
			require.NoError(t, upgradeLWSSizeError(dgd, objects))
			require.Equal(t, 1, counts["Deployment"])
		} else {
			require.Zero(t, counts["LeaderWorkerSet"])
			require.Equal(t, len(workers)+1, counts["Deployment"])
		}
	}
}

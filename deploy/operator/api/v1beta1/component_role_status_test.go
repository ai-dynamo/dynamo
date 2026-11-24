// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestComponentRoleStatusChildGeneration(t *testing.T) {
	for _, current := range []bool{false, true} {
		t.Run(map[bool]string{false: "stale", true: "current"}[current], func(t *testing.T) {
			t.Log("Observe a child with a target snapshot and independent replica state")
			dcd := &DynamoComponentDeployment{
				Name: "child", Generation: 3,
				Status: DynamoComponentDeploymentStatus{ObservedGeneration: 2, Component: &ComponentReplicaStatus{
					Replicas: 4, Roles: []ComponentRoleStatus{{Name: "main"}},
				}},
			}
			if current {
				dcd.Status.ObservedGeneration = dcd.Generation
			}
			original := dcd.DeepCopy()

			t.Log("Relay target roles only after the child has applied this spec, without requiring readiness")
			status := dcd.GetComponentStatuses()["child"]
			require.EqualValues(t, 4, status.Replicas)
			require.Equal(t, current, len(status.Roles) > 0)
			require.Equal(t, original, dcd)
		})
	}
}

func TestComponentRoleStatusNativePodSpec(t *testing.T) {
	t.Log("Serialize native PodSpec lists, known empty values, and unresolved references")
	role := ComponentRoleStatus{Name: "leader", PodSpec: corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Env: []corev1.EnvVar{
			{Name: "EMPTY"}, {Name: "RANK", Value: "$(PROVIDER_RANK)"},
		}}},
		InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:test"}},
	}}
	data, err := json.Marshal(role)
	require.NoError(t, err)
	var decoded ComponentRoleStatus
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, role, decoded)
	require.Contains(t, string(data), `"podSpec":`)
	require.Contains(t, string(data), `"env":[{"name":"EMPTY"},{"name":"RANK","value":"$(PROVIDER_RANK)"}]`)
	require.NotContains(t, string(data), "unresolved")
}

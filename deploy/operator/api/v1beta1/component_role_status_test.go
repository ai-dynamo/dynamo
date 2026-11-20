// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
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

func TestEnvironmentValueStatusKnownEmpty(t *testing.T) {
	t.Log("Preserve the distinction between a known empty string and an unknown binding")
	known, err := json.Marshal(EnvironmentValueStatus{Value: new("")})
	require.NoError(t, err)
	require.JSONEq(t, `{"value":""}`, string(known))
	unknown, err := json.Marshal(EnvironmentValueStatus{Unresolved: true})
	require.NoError(t, err)
	require.JSONEq(t, `{"unresolved":true}`, string(unknown))
}

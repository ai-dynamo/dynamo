/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"fmt"
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVLLMWorkerPlannerSupportsBoundedAbsoluteResizing(t *testing.T) {
	t.Log("use immutable topology identity and a fixed profile rather than a fixed single-step increment")
	planner := ScalePlanner{ProfileFingerprint: "profile", Minimum: 1, Maximum: 4, Layout: ReplicaLayout{NativeMembersPerReplica: 1}}
	status := domain.GroupStatus{Topologies: domain.TopologyHistory{
		CurrentGeneration: 1,
		Snapshots: []domain.MembershipTopology{{Generation: 1, Replicas: []domain.ReplicaMembership{
			{ReplicaID: "replica-0", Members: []domain.NativeMemberIncarnation{{ID: "dp-0", RuntimeIncarnation: "process-0"}}},
			{ReplicaID: "replica-1", Members: []domain.NativeMemberIncarnation{{ID: "dp-1", RuntimeIncarnation: "process-1"}}},
		}}},
	}}
	for _, target := range []int32{0, 1, 2, 3, 4, 5} {
		t.Logf("resolve absolute target %d against the two-replica base", target)
		result, err := planner.ResolveScalePlan(t.Context(), "group", target, status)
		require.NoError(t, err)
		switch {
		case target < 1 || target > 4:
			require.NotNil(t, result.Rejection)
			assert.Equal(t, "TargetOutOfBounds", result.Rejection.Reason)
		case target == 1:
			require.NotNil(t, result.Plan)
			assert.Equal(t, []domain.ReplicaID{"replica-1"}, result.Plan.Change.Retire.Replicas)
			assert.Equal(t, domain.TrafficRequirementEngineManagedDrain, result.Plan.TrafficRequirement)
		case target == 2:
			assert.Nil(t, result.Plan)
			assert.Nil(t, result.Rejection)
		default:
			require.NotNil(t, result.Plan)
			assert.Len(t, result.Plan.Change.Grow.Replicas, int(target-2))
			assert.Equal(t, domain.VerificationRequirementRequired, result.Plan.VerificationRequirement)
		}
	}

	t.Log("a released slot needs restoration rather than silently inheriting its old launch identity")
	status.Registry.Replicas = []domain.ReplicaRecord{{ReplicaID: "replica-2", SlotID: "slot-2"}}
	result, err := planner.ResolveScalePlan(t.Context(), "group", 3, status)
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "RestorationUnsupported", result.Rejection.Reason)

	t.Log("an in-flight operation remains immutable even if the absolute target changes")
	status.Transition = &domain.TransitionStatus{}
	result, err = planner.ResolveScalePlan(t.Context(), "group", 4, status)
	require.NoError(t, err)
	assert.Nil(t, result.Plan)
	assert.Nil(t, result.Rejection)
}

func TestVLLMPlannerKeepsPackedAllocationTargetsSeparateFromNativeMembership(t *testing.T) {
	t.Log("model two allocations with four native members each")
	layout := ReplicaLayout{NativeMembersPerReplica: 4}
	planner := ScalePlanner{ProfileFingerprint: "profile", Minimum: 1, Maximum: 4, Layout: layout}
	topology := domain.MembershipTopology{Generation: 1}
	for index := range int32(2) {
		ids, err := layout.Members(index)
		require.NoError(t, err)
		allocation := domain.ReplicaMembership{ReplicaID: domain.ReplicaID(fmt.Sprintf("replica-%d", index))}
		for _, id := range ids {
			allocation.Members = append(allocation.Members, domain.NativeMemberIncarnation{ID: id, RuntimeIncarnation: domain.RuntimeIncarnationID("process-" + id)})
		}
		topology.Replicas = append(topology.Replicas, allocation)
	}
	status := domain.GroupStatus{Topologies: domain.TopologyHistory{CurrentGeneration: 1, Snapshots: []domain.MembershipTopology{topology}}}

	t.Log("replicas plus one requests one allocation containing dp-8 through dp-11")
	result, err := planner.ResolveScalePlan(t.Context(), "group", 3, status)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NotNil(t, result.Plan)
	require.Len(t, result.Plan.Change.Grow.Replicas, 1)
	require.Equal(t, []domain.NativeMemberID{"dp-8", "dp-9", "dp-10", "dp-11"}, result.Plan.Change.Grow.Replicas[0].NativeMembers)

	t.Log("losing one member keeps the desired two allocations; it must not be mistaken for completed resizing")
	status.Topologies.Snapshots[0].Replicas[1].Members = status.Topologies.Snapshots[0].Replicas[1].Members[:3]
	result, err = planner.ResolveScalePlan(t.Context(), "group", 2, status)
	require.NoError(t, err)
	require.Nil(t, result.Plan)
	require.NotNil(t, result.Rejection)
	require.Equal(t, "RecoveryUnsupported", result.Rejection.Reason)
	require.Equal(t, int32(7), status.Topologies.Snapshots[0].NativeMemberCount())
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"strconv"
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSGLangScalePlannerRetiresOnlyTheHighestCompleteAllocation(t *testing.T) {
	t.Log("start from two committed packed allocations and retain the complete initial allocation")
	planner := ScalePlanner{ProfileFingerprint: "profile-v1", WorldID: "world", MembersPerReplica: 2, RetirementEnabled: true}
	status := domain.GroupStatus{Membership: domain.MembershipStatus{Observed: domain.MembershipObservation{
		CommittedTopology: domain.MembershipTopology{Generation: 2, Replicas: []domain.ReplicaMembership{
			{ReplicaID: "replica-0", Members: []domain.NativeMemberIncarnation{{ID: "dp-0", RuntimeIncarnation: "pod-0/dp-0"}, {ID: "dp-1", RuntimeIncarnation: "pod-0/dp-1"}}},
			{ReplicaID: "replica-1", Members: []domain.NativeMemberIncarnation{{ID: "dp-2", RuntimeIncarnation: "pod-1/dp-2"}, {ID: "dp-3", RuntimeIncarnation: "pod-1/dp-3"}}},
		}},
	}}}

	t.Log("request native graceful retirement for the complete suffix, followed by survivor verification")
	resolution, err := planner.ResolveScalePlan(t.Context(), "group", 1, status)
	require.NoError(t, err)
	require.NotNil(t, resolution.Plan)
	assert.Equal(t, []domain.ReplicaID{"replica-1"}, resolution.Plan.Change.Retire.Replicas)
	assert.Equal(t, domain.TrafficRequirementEngineManagedDrain, resolution.Plan.TrafficRequirement)
	assert.Equal(t, domain.VerificationRequirementRequired, resolution.Plan.VerificationRequirement)

	t.Log("refuse retirement when the engine has not enabled external retiree lifecycle")
	planner.RetirementEnabled = false
	resolution, err = planner.ResolveScalePlan(t.Context(), "group", 1, status)
	require.NoError(t, err)
	require.NotNil(t, resolution.Rejection)
	assert.Equal(t, "SGLangShrinkUnsupported", resolution.Rejection.Reason)
}

func TestSGLangGrowthPlannerBuildsContiguousIdentityPlan(t *testing.T) {
	t.Log("start from a committed two-allocation world")
	planner := ScalePlanner{ProfileFingerprint: "profile-v1", WorldID: "clique-uid", MembersPerReplica: 1}
	status := domain.GroupStatus{Membership: domain.MembershipStatus{Observed: domain.MembershipObservation{
		CommittedTopology: domain.MembershipTopology{
			Generation: 1,
			Replicas: []domain.ReplicaMembership{
				{ReplicaID: "replica-0", Members: []domain.NativeMemberIncarnation{{ID: "dp-0", RuntimeIncarnation: "pod-0"}}},
				{ReplicaID: "replica-1", Members: []domain.NativeMemberIncarnation{{ID: "dp-1", RuntimeIncarnation: "pod-1"}}},
			},
		},
	}}}

	t.Log("reach a larger absolute target through one independently committed append per step")
	previousPlanID := ""
	for rank := 2; rank < 5; rank++ {
		resolution, err := planner.ResolveScalePlan(t.Context(), "group", 5, status)
		require.NoError(t, err)
		require.NotNil(t, resolution.Plan)
		require.Len(t, resolution.Plan.Change.Grow.Replicas, 1)
		joining := resolution.Plan.Change.Grow.Replicas[0]
		assert.Equal(t, domain.ReplicaID("replica-"+strconv.Itoa(rank)), joining.ReplicaID)
		assert.Equal(t, domain.CapacitySlotID("slot-"+strconv.Itoa(rank)), joining.SlotID)
		assert.Equal(t, []domain.NativeMemberID{domain.NativeMemberID("dp-" + strconv.Itoa(rank))}, joining.NativeMembers)
		assert.NotEqual(t, previousPlanID, resolution.Plan.ID)
		previousPlanID = resolution.Plan.ID

		// Retry the same committed base without changing the immutable step, even if the final target grows.
		t.Log("re-resolving the next append preserves its identity and exact payload")
		replayed, err := planner.ResolveScalePlan(t.Context(), "group", 6, status)
		require.NoError(t, err)
		assert.Equal(t, resolution.Plan, replayed.Plan)

		t.Log("observe the appended member as a new committed topology before planning the next step")
		base := &status.Membership.Observed.CommittedTopology
		base.Generation++
		base.Replicas = append(base.Replicas, domain.ReplicaMembership{
			ReplicaID: joining.ReplicaID,
			Members: []domain.NativeMemberIncarnation{{
				ID: joining.NativeMembers[0], RuntimeIncarnation: domain.RuntimeIncarnationID("pod-" + strconv.Itoa(rank)),
			}},
		})
	}

	t.Log("stop planning once committed membership reaches the absolute target")
	resolution, err := planner.ResolveScalePlan(t.Context(), "group", 5, status)
	require.NoError(t, err)
	assert.Nil(t, resolution.Plan)
	assert.Nil(t, resolution.Rejection)

	t.Log("reject shrink as a definitive backend capability boundary")
	resolution, err = planner.ResolveScalePlan(t.Context(), "group", 4, status)
	require.NoError(t, err)
	require.NotNil(t, resolution.Rejection)
	assert.Equal(t, "SGLangShrinkUnsupported", resolution.Rejection.Reason)

	t.Log("whole-world retirement remains explicitly unsupported rather than releasing healthy capacity")
	resolution, err = planner.ResolveScalePlan(t.Context(), "group", 0, status)
	require.NoError(t, err)
	require.NotNil(t, resolution.Rejection)
	assert.Equal(t, "SGLangRetirementUnsupported", resolution.Rejection.Reason)
}

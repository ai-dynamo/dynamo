/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEngineGroupPackedMemberProjectionPreservesDesiredEP8ThroughEP7Recovery(t *testing.T) {
	t.Log("Form an EP8 world from two physically disjoint four-member allocations")
	status := healthyEngineGroupProjectionStatus(2)
	base := status.Membership.Observed.CommittedTopology
	base.Replicas[0].NativeMembers = []enginegroup.NativeMemberID{"dp-0", "dp-1", "dp-2", "dp-3"}
	base.Replicas[1].NativeMembers = []enginegroup.NativeMemberID{"dp-4", "dp-5", "dp-6", "dp-7"}
	status.Membership.Observed.CommittedTopology = base
	status.Topologies.Snapshots = []enginegroup.MembershipTopology{base}
	status.Traffic.Observed.Admitted = base.Replicas
	for i := range status.Capacity.Observed.Allocations {
		status.Capacity.Observed.Allocations[i].Health = enginegroup.AllocationHealthHealthy
	}
	profile := api.EngineGroupProfileStatus{
		Backend: "vllm", Fingerprint: "packed-tp1", GPUsPerReplica: 4, PodsPerReplica: 1,
		NativeMembersPerReplica: 4, MinSafeServingNativeMembers: 4,
		MinSupportedReplicas: 1, MaxSupportedReplicas: 16,
	}
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 1},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	reconciler := &DynamoGraphDeploymentEngineGroupReconciler{}
	reconciler.projectEngineGroupStatus(group, profile, status, nil, nil)
	require.Equal(t, int32(2), group.Status.Replicas)
	require.Equal(t, int32(2), group.Status.AvailableReplicas)
	require.Equal(t, int32(8), group.Status.ActiveNativeMemberCount)
	require.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTargetReached).Status)
	desired := append([]string(nil), group.Status.DesiredNativeMembers...)

	t.Log("The engine commits EP7 after losing dp-5, while the second allocation still serves three ranks")
	survivors := enginegroup.MembershipTopology{
		Generation: 2,
		Replicas: []enginegroup.ReplicaMembership{
			base.Replicas[0],
			{ReplicaID: base.Replicas[1].ReplicaID, RuntimeIncarnation: base.Replicas[1].RuntimeIncarnation,
				NativeMembers: []enginegroup.NativeMemberID{"dp-4", "dp-6", "dp-7"}},
		},
	}
	status.Membership.Observed.CommittedTopology = survivors
	status.Topologies.CurrentGeneration = 2
	status.Topologies.Snapshots = append(status.Topologies.Snapshots, survivors)
	status.Traffic.Observed.Admitted = survivors.Replicas
	status.Traffic.Observed.Drained = []enginegroup.ReplicaMembership{{
		ReplicaID: base.Replicas[1].ReplicaID, RuntimeIncarnation: base.Replicas[1].RuntimeIncarnation,
		NativeMembers: []enginegroup.NativeMemberID{"dp-5"},
	}}
	status.Capacity.Observed.Allocations[1].Health = enginegroup.AllocationHealthDegraded
	reconciler.projectEngineGroupStatus(group, profile, status, nil, nil)

	t.Log("Physical target stays two allocations, desired membership stays eight, and committed membership becomes seven")
	assert.Equal(t, int32(2), group.Spec.Replicas)
	assert.Equal(t, int32(2), group.Status.Replicas)
	assert.Equal(t, int32(1), group.Status.AvailableReplicas)
	assert.Equal(t, desired, group.Status.DesiredNativeMembers)
	assert.Equal(t, int32(8), group.Status.DesiredNativeMemberCount)
	assert.Equal(t, int32(7), group.Status.ActiveNativeMemberCount)
	assert.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable).Status)
	assert.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionDegraded).Status)
	assert.Equal(t, metav1.ConditionFalse, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTargetReached).Status)
	require.Len(t, group.Status.ReplicaStates, 2)
	partial := group.Status.ReplicaStates[1]
	require.NotNil(t, partial.CurrentAllocation)
	assert.Equal(t, api.EngineGroupReplicaAvailabilityAvailable, partial.CurrentAllocation.Availability)
	assert.Equal(t, api.EngineGroupAllocationHealthDegraded, partial.CurrentAllocation.Health)
	assert.Contains(t, partial.NativeMembers, api.EngineGroupNativeMemberStatus{ID: "dp-5", Membership: api.EngineGroupReplicaMembershipMasked, Traffic: api.EngineGroupMemberTrafficDrained})
	assert.Contains(t, partial.NativeMembers, api.EngineGroupNativeMemberStatus{ID: "dp-6", Membership: api.EngineGroupReplicaMembershipActive, Traffic: api.EngineGroupMemberTrafficAdmitted})

	t.Log("Round-trip the degraded resource before restoring the same desired membership")
	group = group.DeepCopy()
	base.Generation = 3
	status.Membership.Observed.CommittedTopology = base
	status.Topologies.CurrentGeneration = 3
	status.Topologies.Snapshots = append(status.Topologies.Snapshots, base)
	status.Traffic.Observed.Admitted = base.Replicas
	status.Traffic.Observed.Drained = nil
	status.Capacity.Observed.Allocations[1].Health = enginegroup.AllocationHealthHealthy
	reconciler.projectEngineGroupStatus(group, profile, status, nil, nil)
	assert.Equal(t, desired, group.Status.DesiredNativeMembers)
	assert.Equal(t, int32(8), group.Status.ActiveNativeMemberCount)
	assert.Equal(t, int32(2), group.Status.AvailableReplicas)
	assert.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTargetReached).Status)
	assert.Equal(t, metav1.ConditionFalse, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionDegraded).Status)
}

func TestEngineGroupPackedTargetRequiresExactMemberIdentities(t *testing.T) {
	t.Log("Create authoritative serving membership with the correct count but the wrong rank identity")
	status := healthyEngineGroupProjectionStatus(1)
	status.Membership.Observed.CommittedTopology.Replicas[0].NativeMembers = []enginegroup.NativeMemberID{"dp-9"}
	status.Topologies.Snapshots = []enginegroup.MembershipTopology{status.Membership.Observed.CommittedTopology}
	status.Traffic.Observed.Admitted = status.Membership.Observed.CommittedTopology.Replicas
	group := &api.DynamoGraphDeploymentEngineGroup{
		Spec:   api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
		Status: api.DynamoGraphDeploymentEngineGroupStatus{DesiredNativeMembers: []string{"dp-0"}},
	}
	profile := api.EngineGroupProfileStatus{Fingerprint: "profile-v1", NativeMembersPerReplica: 1, MinSafeServingNativeMembers: 1}

	t.Log("Projection must not rewrite desired identity or declare convergence from cardinality alone")
	(&DynamoGraphDeploymentEngineGroupReconciler{}).projectEngineGroupStatus(group, profile, status, nil, nil)
	assert.Equal(t, []string{"dp-0"}, group.Status.DesiredNativeMembers)
	assert.Equal(t, int32(1), group.Status.ActiveNativeMemberCount)
	assert.Equal(t, metav1.ConditionFalse, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTargetReached).Status)
}

func TestEngineGroupPlannedRetirementDoesNotHideDegradedSurvivor(t *testing.T) {
	t.Log("Commit the exact planned survivors while a retained allocation reports degraded health")
	status := plannedRetirementProjectionStatus(t)
	status.Capacity.Observed.Allocations[0].Health = enginegroup.AllocationHealthDegraded
	group := &api.DynamoGraphDeploymentEngineGroup{
		Spec:   api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
		Status: api.DynamoGraphDeploymentEngineGroupStatus{LastStableReplicas: 2},
	}
	profile := api.EngineGroupProfileStatus{Fingerprint: "profile-v1", NativeMembersPerReplica: 1, MinSafeServingNativeMembers: 1}

	t.Log("Planned membership loss is suppressed, but independent survivor degradation remains visible")
	(&DynamoGraphDeploymentEngineGroupReconciler{}).projectEngineGroupStatus(group, profile, status, nil, nil)
	assert.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionDegraded).Status)
}

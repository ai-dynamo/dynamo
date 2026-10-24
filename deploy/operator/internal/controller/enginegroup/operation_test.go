/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEngineGroupOperationProjectsResolvedTargets(t *testing.T) {
	tests := []struct {
		name, intent                string
		change                      domain.ResolvedChange
		replicas                    int32
		members, joining, nominated []string
	}{
		{
			name: "growth", intent: "Grow", replicas: 3, members: []string{"dp-0", "dp-1", "dp-2"}, joining: []string{"replica-2"},
			change: domain.ResolvedChange{Kind: domain.PlanKindGrow, Grow: &domain.GrowChange{
				Replicas: []domain.ReplicaTarget{engineGroupTestReplicaTarget("replica-2", "slot-2", "dp-2")},
			}},
		},
		{
			name: "planned shrink", intent: "Shrink", replicas: 1, members: []string{"dp-0"}, nominated: []string{"replica-1"},
			change: domain.ResolvedChange{Kind: domain.PlanKindRetire, Retire: &domain.RetireChange{Replicas: []domain.ReplicaID{"replica-1"}}},
		},
		{
			name: "survivor reduction", intent: "Recover", replicas: 1, members: []string{"dp-0"}, nominated: []string{"replica-1"},
			change: domain.ResolvedChange{Kind: domain.PlanKindReduceToSurvivors, ReduceToSurvivors: &domain.ReduceToSurvivorsChange{
				Survivors: []domain.ReplicaNativeMembership{{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []domain.NativeMemberID{"dp-0"}}},
			}},
		},
		{
			name: "restoration", intent: "Recover", replicas: 2, members: []string{"dp-0", "dp-1"}, joining: []string{"replica-1"},
			change: domain.ResolvedChange{Kind: domain.PlanKindRestore, Restore: &domain.RestoreChange{
				Replicas: []domain.RestorationTarget{{ReplicaTarget: engineGroupTestReplicaTarget("replica-1", "slot-1", "dp-1")}},
			}},
		},
		{
			name: "remapping", intent: "Recover", replicas: 2, members: []string{"dp-4", "dp-5"},
			change: domain.ResolvedChange{Kind: domain.PlanKindRemap, Remap: &domain.RemapChange{
				Membership: []domain.ReplicaNativeMembership{
					{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []domain.NativeMemberID{"dp-4"}},
					{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []domain.NativeMemberID{"dp-5"}},
				},
			}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build an immutable resolved plan from a known base, independently of the latest scale target")
			state := engineGroupTestStatus()
			base := &state.Topologies.Snapshots[0]
			base.Replicas = append(base.Replicas, engineGroupTestMembership("replica-1", "runtime-1", "dp-1"))
			if test.change.Kind == domain.PlanKindRestore {
				base.Replicas = base.Replicas[:1]
			}
			state.Transition.Spec.Plan = engineGroupTestPlan(test.change)
			group := &api.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{Generation: 7},
				Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 6},
			}

			t.Log("project the exact operation, not the latest replica count or observed joining registry")
			require.NoError(t, projectEngineGroupOperation(group, state))
			operation := group.Status.Operation
			require.NotNil(t, operation)
			assert.Equal(t, test.intent, operation.Intent)
			assert.Equal(t, string(test.change.Kind), operation.Shape)
			assert.Equal(t, test.replicas, operation.TargetReplicas)
			assert.Equal(t, test.members, operation.TargetNativeMembers)
			assert.Equal(t, test.joining, operation.JoiningReplicas)
			assert.Equal(t, test.nominated, operation.NominatedReplicas)
			assert.Equal(t, int64(7), operation.SpecGeneration)
			assert.Equal(t, engineGroupTopologyToAPI(*base), operation.BaseTopology)
			frozen := operation.DeepCopy()

			t.Log("queue a later target without changing operation identity, target, base, or timestamps")
			group.Generation++
			group.Spec.Replicas = 8
			require.NoError(t, projectEngineGroupOperation(group, state))
			assert.Equal(t, int32(8), *group.Status.Operation.QueuedTargetReplicas)
			group.Status.Operation.QueuedTargetReplicas = nil
			assert.Equal(t, frozen, group.Status.Operation)
		})
	}
}

func TestEngineGroupOperationProjectsMembershipAndWorkflowPhases(t *testing.T) {
	tests := []struct {
		name, expected               string
		outcome                      domain.TransitionOutcome
		membership                   domain.MembershipTransitionPhase
		noTarget, absent, mismatched bool
	}{
		{name: "prework", expected: "Pending", noTarget: true},
		{name: "unobserved submission", expected: "Submitting", absent: true},
		{name: "accepted collective", expected: "Committing", membership: domain.MembershipTransitionPhasePending},
		{name: "committed membership", expected: "Committed", membership: domain.MembershipTransitionPhaseCommitted},
		{name: "ambiguous outcome", expected: "Unknown", membership: domain.MembershipTransitionPhaseUnknown},
		{name: "definitive rejection", expected: "Failed", membership: domain.MembershipTransitionPhaseRejected},
		{name: "uncorrelated commit", expected: "Unknown", membership: domain.MembershipTransitionPhaseCommitted, mismatched: true},
		{name: "compensation", expected: "Aborting", outcome: domain.TransitionOutcomeReverting},
		{name: "compensated", expected: "Aborted", outcome: domain.TransitionOutcomeRolledBack},
		{name: "blocked serving", expected: "Failed", outcome: domain.TransitionOutcomeBlocked},
		{name: "completed", expected: "Committed", outcome: domain.TransitionOutcomeCompleted},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build independently correlated membership and workflow observations")
			state := engineGroupTestStatus()
			state.Transition.Outcome = domain.TransitionOutcomeProgressing
			if test.outcome != "" {
				state.Transition.Outcome = test.outcome
			}
			if test.noTarget {
				state.Membership.Desired = nil
			}
			if test.absent {
				state.Membership.Observed.Transition = nil
			} else {
				state.Membership.Observed.Transition.Phase = test.membership
				if test.mismatched {
					state.Membership.Observed.Transition.TargetDigest = "unrelated-target"
				}
			}
			group := &api.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{Generation: 1}}

			t.Log("publish a phase without promoting an uncorrelated result into commit evidence")
			require.NoError(t, projectEngineGroupOperation(group, state))
			assert.Equal(t, api.EngineGroupOperationPhase(test.expected), group.Status.Operation.Phase)
			if test.mismatched || test.absent || test.noTarget {
				assert.Nil(t, group.Status.Operation.CommittedTopology)
			}
		})
	}
}

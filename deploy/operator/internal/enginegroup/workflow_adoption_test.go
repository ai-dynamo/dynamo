/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package enginegroup

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWorkflowCoordinatorAdoptsCompletedSurvivorTopologyBeforeExactRelease(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
			workflowReplicaAllocation("replica-3", "pod-3"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(3, "replica-0", "replica-1", "replica-2"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: []ReplicaID{"replica-0", "replica-1", "replica-2", "replica-3"},
		},
		externalMutationHistory: &externalMutations,
	}
	plan := &OperationPlan{
		ID:                "survivor-recovery-plan",
		Intent:            OperationIntentRecover,
		TargetReplicas:    3,
		NominatedReplicas: []ReplicaID{"replica-3"},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
		Plan:            plan,
		Operation:       completedUnknownTopology(),
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist an adopted recovery record before withdrawing traffic or submitting backend work")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, workflowTestOperationID, result.Operation.ID)
	assert.Equal(t, plan.ID, result.Operation.PlanID)
	assert.Equal(t, OperationIntentRecover, result.Operation.Intent)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int64(2), result.Operation.BaseTopologyGeneration)
	assert.Equal(t, int64(3), result.Operation.CommittedTopologyGeneration)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1", "replica-2", "replica-3"}, result.Operation.BaseReplicas)
	assert.Equal(t, []ReplicaID{"replica-3"}, result.Operation.NominatedReplicas)
	assert.True(t, result.Operation.Adopted)
	assert.False(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, traffic.withdrawCalls)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart from the durable adopted record and withdraw the exact missing replica")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, [][]ReplicaID{{"replica-3"}}, traffic.withdrawCalls)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + workflowTestOperationID}, externalMutations)

	t.Log("Observe operation-scoped drain and persist UID-bound release authorization")
	traffic.snapshot = TrafficSnapshot{
		OperationID: workflowTestOperationID,
		Admitted:    []ReplicaID{"replica-0", "replica-1", "replica-2"},
		Drained:     []ReplicaID{"replica-3"},
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, workflowTestReleaseID, result.ReleaseAuthorization.ID)
	assert.Equal(t, workflowTestOperationID, result.ReleaseAuthorization.OperationID)
	assert.Equal(t, int64(3), result.ReleaseAuthorization.TopologyGeneration)
	assert.Equal(t, []AuthorizedReplica{{
		ReplicaID: "replica-3",
		SlotID:    "slot-replica-3",
		CapacityRefs: []CapacityRef{{
			Namespace: "test",
			Name:      "pod-3",
			UID:       "uid-pod-3",
		}},
	}}, result.ReleaseAuthorization.Replicas)
	assert.Empty(t, capacity.releaseCalls)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from the durable authorization and release only its exact Pod UID")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, *input.ReleaseAuthorization, capacity.releaseCalls[0])
	assert.Empty(t, membership.submitCalls)

	t.Log("Clear release state only after the authorized UID is observably absent")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		},
		FencedReplicas: []ReplicaID{"replica-3"},
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, int32(3), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(3), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Persist post-commit completion before replacing the adopted record with queued restoration")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, workflowTestOperationID, result.Operation.ID)
	assert.True(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Plan cardinal restoration only after the completed marker survives another restart")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = func() string { return "restore-operation" }
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, "restore-operation", result.Operation.ID)
	assert.Equal(t, OperationIntentGrow, result.Operation.Intent)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Equal(t, int32(4), result.Operation.TargetReplicas)
	assert.Empty(t, membership.submitCalls)
}

func TestWorkflowCoordinatorFinishesOldReductionBeforeAdoptingLaterSurvivors(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(3, "replica-0"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: "old-shrink-operation",
			Admitted:    []ReplicaID{"replica-0"},
			Drained:     []ReplicaID{"replica-2"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                          "old-shrink-operation",
		Attempt:                     1,
		PlanID:                      "old-shrink-plan",
		Intent:                      OperationIntentShrink,
		SpecGeneration:              2,
		BaseTopologyGeneration:      1,
		BaseReplicas:                []ReplicaID{"replica-0", "replica-1", "replica-2"},
		TargetReplicas:              2,
		NominatedReplicas:           []ReplicaID{"replica-2"},
		Phase:                       OperationPhaseUnknown,
		CommittedTopologyGeneration: 2,
		StartedAt:                   workflowTestTime.Add(-2 * time.Minute),
		LastTransitionTime:          workflowTestTime.Add(-time.Minute),
	}
	plan := &OperationPlan{
		ID:                "later-survivor-plan",
		Intent:            OperationIntentRecover,
		TargetReplicas:    1,
		NominatedReplicas: []ReplicaID{"replica-1"},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 3,
		Plan:            plan,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Preserve the old operation and authorize only its unfinished retired allocation")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, operation, result.Operation)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, int64(3), result.ReleaseAuthorization.TopologyGeneration)
	assert.Equal(t, int32(1), result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, []AuthorizedReplica{{
		ReplicaID: "replica-2",
		SlotID:    "slot-replica-2",
		CapacityRefs: []CapacityRef{{
			Namespace: "test",
			Name:      "pod-2",
			UID:       "uid-pod-2",
		}},
	}}, result.ReleaseAuthorization.Replicas)
	assert.Empty(t, membership.submitCalls)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Revalidate the later topology after restart and issue the old exact release")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, *input.ReleaseAuthorization, capacity.releaseCalls[0])

	t.Log("Complete the old exact release while the authoritative survivor has no physical allocation")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot.Allocations = []ReplicaAllocation{
		workflowReplicaAllocation("replica-1", "pod-1"),
	}
	capacity.snapshot.FencedReplicas = []ReplicaID{"replica-2"}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Equal(t, int32(1), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(3), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, capacity.ensureCalls)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Persist old cleanup completion without repairing the later topology under a stale operation")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, capacity.ensureCalls)
	input.Operation = result.Operation

	t.Log("Adopt only the newly missing survivor before current capacity repair may begin")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.True(t, result.Operation.Adopted)
	assert.Equal(t, plan.ID, result.Operation.PlanID)
	assert.Equal(t, []ReplicaID{"replica-1"}, result.Operation.NominatedReplicas)
	assert.Equal(t, int32Pointer(3), result.Operation.QueuedTargetReplicas)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, membership.submitCalls)
}

func TestWorkflowCoordinatorAdoptsAfterIncompleteGrowthAndFinishesSurvivorTraffic(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
			workflowReplicaAllocation("replica-3", "pod-3"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(3, "replica-0", "replica-1", "replica-2"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: []ReplicaID{"replica-0", "replica-1"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                          "incomplete-operation",
		Attempt:                     1,
		Intent:                      OperationIntentGrow,
		SpecGeneration:              2,
		BaseTopologyGeneration:      1,
		BaseReplicas:                []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:              4,
		JoiningReplicas:             []ReplicaID{"replica-2", "replica-3"},
		Phase:                       OperationPhaseUnknown,
		CommittedTopologyGeneration: 2,
		StartedAt:                   workflowTestTime.Add(-2 * time.Minute),
		LastTransitionTime:          workflowTestTime.Add(-time.Minute),
	}
	plan := &OperationPlan{
		ID:                "survivor-recovery-plan",
		Intent:            OperationIntentRecover,
		TargetReplicas:    3,
		NominatedReplicas: []ReplicaID{"replica-3"},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
		Plan:            plan,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Adopt the authoritative survivor topology without replaying the incomplete growth operation")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.True(t, result.Operation.Adopted)
	assert.Equal(t, plan.ID, result.Operation.PlanID)
	assert.Equal(t, []ReplicaID{"replica-3"}, result.Operation.NominatedReplicas)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Withdraw and drain the missing replica under the durable adopted operation")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, [][]ReplicaID{{"replica-3"}}, traffic.withdrawCalls)
	traffic.snapshot = TrafficSnapshot{
		OperationID: result.Operation.ID,
		Admitted:    []ReplicaID{"replica-0", "replica-1"},
		Drained:     []ReplicaID{"replica-3"},
	}

	t.Log("Persist and issue exact release authorization for the missing replica")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	input.ReleaseAuthorization = result.ReleaseAuthorization
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)

	t.Log("Observe both physical removal and the durable logical fence before clearing authorization")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		},
		FencedReplicas: []ReplicaID{"replica-3"},
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Equal(t, int32(3), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(3), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Admit every available survivor omitted by the incomplete prior growth postwork")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1", "replica-2"}, traffic.admitCalls[len(traffic.admitCalls)-1])
	assert.False(t, result.Operation.PostCommitComplete)
	traffic.snapshot.Admitted = []ReplicaID{"replica-0", "replica-1", "replica-2"}

	t.Log("Persist post-commit completion only after the complete survivor topology is serving")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
}

func TestWorkflowCoordinatorRejectsUnsafeSurvivorAdoption(t *testing.T) {
	tests := []struct {
		name      string
		topology  MembershipTopology
		plan      OperationPlan
		wantError string
	}{
		{
			name:     "nominees do not identify the missing replica",
			topology: workflowTopology(3, "replica-0", "replica-1", "replica-2"),
			plan: OperationPlan{
				ID:                "survivor-recovery-plan",
				Intent:            OperationIntentRecover,
				TargetReplicas:    3,
				NominatedReplicas: []ReplicaID{"replica-2"},
			},
			wantError: "do not match missing replicas",
		},
		{
			name:     "observed topology contains an unknown survivor identity",
			topology: workflowTopology(3, "replica-0", "replica-1", "replica-9"),
			plan: OperationPlan{
				ID:                "survivor-recovery-plan",
				Intent:            OperationIntentRecover,
				TargetReplicas:    3,
				NominatedReplicas: []ReplicaID{"replica-3"},
			},
			wantError: "observed survivor replica \"replica-9\"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-1", "pod-1"),
					workflowReplicaAllocation("replica-2", "pod-2"),
					workflowReplicaAllocation("replica-3", "pod-3"),
				}},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                tt.topology,
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: TrafficSnapshot{
					Admitted: []ReplicaID{"replica-0", "replica-1"},
				},
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Reject an observed topology that cannot be correlated with the completed logical membership")
			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  3,
				DesiredReplicas: 4,
				Plan:            &tt.plan,
				Operation:       completedUnknownTopology(),
			})
			require.ErrorContains(t, err, tt.wantError)
			assert.Equal(t, completedUnknownTopology(), result.Operation)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, traffic.withdrawCalls)
			assert.Empty(t, capacity.releaseCalls)
			assert.Empty(t, externalMutations)
		})
	}
}

func completedUnknownTopology() *Operation {
	return &Operation{
		ID:                          "completed-operation",
		Attempt:                     1,
		PlanID:                      "completed-plan",
		Intent:                      OperationIntentRecover,
		SpecGeneration:              2,
		BaseTopologyGeneration:      1,
		BaseReplicas:                []ReplicaID{"replica-0", "replica-1", "replica-2", "replica-3"},
		TargetReplicas:              4,
		Phase:                       OperationPhaseUnknown,
		CommittedTopologyGeneration: 2,
		PostCommitComplete:          true,
		StartedAt:                   workflowTestTime.Add(-2 * time.Minute),
		LastTransitionTime:          workflowTestTime.Add(-time.Minute),
	}
}

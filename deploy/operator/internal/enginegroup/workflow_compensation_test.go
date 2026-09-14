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
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type commitRaceMembershipAdapter struct {
	baseTopology            MembershipTopology
	committedTopology       MembershipTopology
	committedOperation      BackendOperation
	observeTopologyCalls    int
	observeOperationCalls   int
	submitOperationRequests []MembershipRequest
}

func (f *commitRaceMembershipAdapter) ObserveCapabilities(
	context.Context,
	GroupID,
) (MembershipCapabilities, error) {
	return allTestMembershipCapabilities(), nil
}

func (f *commitRaceMembershipAdapter) ObserveTopology(
	context.Context,
	GroupID,
) (MembershipTopology, error) {
	// Expose the original base to Step, then the raced committed topology to Submit and verification.
	f.observeTopologyCalls++
	if f.observeTopologyCalls == 1 {
		return cloneTopology(f.baseTopology), nil
	}
	return cloneTopology(f.committedTopology), nil
}

func (f *commitRaceMembershipAdapter) ObserveOperation(
	_ context.Context,
	_ GroupID,
	_ string,
	_ int32,
) (BackendOperation, error) {
	// Report Absent during restart recovery, then expose the exact commit that raced the final topology read.
	f.observeOperationCalls++
	if f.observeOperationCalls == 1 {
		return BackendOperation{Phase: BackendOperationPhaseAbsent}, nil
	}
	return f.committedOperation, nil
}

func (f *commitRaceMembershipAdapter) SubmitOperation(
	_ context.Context,
	_ GroupID,
	request MembershipRequest,
) (BackendOperation, error) {
	// Preserve any unexpected submission for a focused assertion before failing the reconciliation.
	cloned := request
	cloned.BaseReplicas = slices.Clone(request.BaseReplicas)
	cloned.JoiningReplicas = slices.Clone(request.JoiningReplicas)
	cloned.NominatedReplicas = slices.Clone(request.NominatedReplicas)
	f.submitOperationRequests = append(f.submitOperationRequests, cloned)
	return BackendOperation{}, errors.New("unexpected membership submission")
}

func TestWorkflowCoordinatorCompensatesPreSubmitShrinkAfterBaseTopologyDrift(t *testing.T) {
	tests := []struct {
		name                      string
		phase                     OperationPhase
		wantObserveOperationCalls []operationAttempt
	}{
		{
			name:  "pending shrink",
			phase: OperationPhasePending,
		},
		{
			name:  "persisted submitting shrink absent from backend",
			phase: OperationPhaseSubmitting,
			wantObserveOperationCalls: []operationAttempt{{
				ID:      workflowTestOperationID,
				Attempt: 1,
			}},
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
				}},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2"),
				operation:               BackendOperation{Phase: BackendOperationPhaseAbsent},
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: TrafficSnapshot{
					OperationID: workflowTestOperationID,
					Admitted:    []ReplicaID{"replica-0", "replica-1"},
					Drained:     []ReplicaID{"replica-2", "replica-3"},
				},
				externalMutationHistory: &externalMutations,
			}
			operation := &Operation{
				ID:                     workflowTestOperationID,
				Attempt:                1,
				PlanID:                 "plan-1",
				Intent:                 OperationIntentShrink,
				SpecGeneration:         2,
				BaseTopologyGeneration: 1,
				BaseReplicas: []ReplicaID{
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				},
				TargetReplicas:     2,
				NominatedReplicas:  []ReplicaID{"replica-2", "replica-3"},
				Phase:              tt.phase,
				StartedAt:          workflowTestTime.Add(-time.Minute),
				LastTransitionTime: workflowTestTime,
			}
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 2,
				Plan: &OperationPlan{
					ID:                "plan-1",
					Intent:            OperationIntentShrink,
					TargetReplicas:    2,
					NominatedReplicas: []ReplicaID{"replica-2", "replica-3"},
				},
				Operation: operation,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Persist Aborting after authoritative topology drift without mutating an external system")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.NotNil(t, result.Operation.Failure)
			assert.Equal(t, FailureClassificationTerminal, result.Operation.Failure.Classification)
			assert.Equal(t, "BaseTopologyChanged", result.Operation.Failure.Reason)
			assert.True(t, result.OperationChanged)
			assert.Equal(t, tt.wantObserveOperationCalls, membership.observeOperationCalls)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, externalMutations)
			input.Operation = result.Operation

			t.Log("Restart from Aborting and durably freeze the base identity excluded from current topology")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			assert.Equal(t, []ReplicaID{"replica-3"}, result.Operation.CleanupReplicas)
			assert.True(t, result.OperationChanged)
			assert.Nil(t, result.ReleaseAuthorization)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, externalMutations)
			input.Operation = result.Operation

			t.Log("Restart from durable cleanup state and persist its exact target barrier before repairing traffic")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.NotNil(t, result.ReleaseAuthorization)
			assert.Equal(t, int32(3), result.ReleaseAuthorization.TargetReplicas)
			assert.Equal(t, int64(2), result.ReleaseAuthorization.TopologyGeneration)
			require.Len(t, result.ReleaseAuthorization.Replicas, 1)
			assert.Equal(t, ReplicaID("replica-3"), result.ReleaseAuthorization.Replicas[0].ReplicaID)
			assert.Empty(t, result.ReleaseAuthorization.Replicas[0].CapacityRefs)
			assert.True(t, result.ReleaseAuthorizationChanged)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, externalMutations)
			input.ReleaseAuthorization = result.ReleaseAuthorization

			t.Log("Restart from durable authorization and issue the target barrier")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.Len(t, capacity.releaseCalls, 1)
			require.Len(t, capacity.releaseCalls[0].Replicas, 1)
			assert.Equal(t, ReplicaID("replica-3"), capacity.releaseCalls[0].Replicas[0].ReplicaID)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)

			t.Log("Observe Applied and durably persist the target proof")
			capacity.releaseObservation = CapacityReleaseObservation{
				ReleaseID: workflowTestReleaseID,
				Phase:     CapacityReleasePhaseApplied,
			}
			capacity.snapshot.FencedReplicas = []ReplicaID{"replica-3"}
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			assert.Equal(t, int32(3), result.Operation.CapacityTargetReplicas)
			assert.Equal(t, int64(2), result.Operation.CapacityTopologyGeneration)
			assert.True(t, result.Operation.CapacityTargetApplied)
			assert.True(t, result.OperationChanged)
			assert.Nil(t, result.ReleaseAuthorization)
			input.Operation = result.Operation
			input.ReleaseAuthorization = nil

			t.Log("Restart from the durable target proof and explicitly admit every authoritative survivor")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Equal(t, []TrafficRequest{{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 2,
				Replicas:           []ReplicaID{"replica-0", "replica-1", "replica-2"},
			}}, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Equal(t, []string{
				"capacity.release:" + workflowTestReleaseID,
				"traffic.admit:" + workflowTestOperationID,
			}, externalMutations)

			t.Log("Restart after traffic converges and persist Aborted")
			traffic.snapshot = TrafficSnapshot{
				OperationID: workflowTestOperationID,
				Admitted:    []ReplicaID{"replica-0", "replica-1", "replica-2"},
				Drained:     []ReplicaID{"replica-3"},
			}
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
			assert.True(t, result.OperationChanged)
			input.Operation = result.Operation

			t.Log("Restart from Aborted without replaying compensation or membership submission")
			input.DesiredReplicas = 3
			input.Plan = nil
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Len(t, traffic.admitRequests, 1)
			assert.Empty(t, membership.submitCalls)
			assert.Equal(t, []string{
				"capacity.release:" + workflowTestReleaseID,
				"traffic.admit:" + workflowTestOperationID,
			}, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorAttributesCommitRacingFinalPreSubmitTopologyCheck(t *testing.T) {
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
	membership := &commitRaceMembershipAdapter{
		baseTopology:      workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3"),
		committedTopology: workflowTopology(2, "replica-0", "replica-1"),
		committedOperation: BackendOperation{
			ID:                          workflowTestOperationID,
			Attempt:                     1,
			BackendID:                   "backend-" + workflowTestOperationID,
			TargetReplicas:              2,
			Phase:                       BackendOperationPhaseCommitted,
			CommittedTopologyGeneration: 2,
		},
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0", "replica-1"},
			Drained:     []ReplicaID{"replica-2", "replica-3"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		PlanID:                 "plan-1",
		Intent:                 OperationIntentShrink,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas: []ReplicaID{
			"replica-0",
			"replica-1",
			"replica-2",
			"replica-3",
		},
		TargetReplicas:     2,
		NominatedReplicas:  []ReplicaID{"replica-2", "replica-3"},
		Phase:              OperationPhaseSubmitting,
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Plan: &OperationPlan{
			ID:                "plan-1",
			Intent:            OperationIntentShrink,
			TargetReplicas:    2,
			NominatedReplicas: []ReplicaID{"replica-2", "replica-3"},
		},
		Operation: operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Attribute a commit that races the final pre-submit topology check to the exact durable operation")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int64(2), result.Operation.CommittedTopologyGeneration)
	assert.Equal(t, "backend-"+workflowTestOperationID, result.Operation.BackendOperationID)
	assert.Nil(t, result.Operation.Failure)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, 3, membership.observeTopologyCalls)
	assert.Equal(t, 2, membership.observeOperationCalls)
	assert.Empty(t, membership.submitOperationRequests)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart from the durable commit proof and persist exact release authorization")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	require.Len(t, result.ReleaseAuthorization.Replicas, 2)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, []ReplicaID{
		result.ReleaseAuthorization.Replicas[0].ReplicaID,
		result.ReleaseAuthorization.Replicas[1].ReplicaID,
	})
	assert.Empty(t, capacity.releaseCalls)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from durable authorization and issue the exact physical release")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, workflowTestReleaseID, capacity.releaseCalls[0].ID)
	assert.Empty(t, membership.submitOperationRequests)

	t.Log("Observe released allocations as absent and fenced before clearing authorization")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		},
		FencedReplicas: []ReplicaID{"replica-2", "replica-3"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, result.Capacity.FencedReplicas)
	assert.Empty(t, membership.submitOperationRequests)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorCompletesZeroSurvivorCompensationWithoutEmptyAdmission(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{externalMutationHistory: &externalMutations}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{externalMutationHistory: &externalMutations}
	operation := &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		PlanID:                 "plan-1",
		Intent:                 OperationIntentShrink,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         1,
		NominatedReplicas:      []ReplicaID{"replica-1"},
		Phase:                  OperationPhaseAborting,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "BaseTopologyChanged",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 1,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Durably freeze every former base identity after the authoritative topology becomes empty")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1"}, result.Operation.CleanupReplicas)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart and withdraw the former base identities without ever admitting an empty set")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []TrafficRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		Replicas:           []ReplicaID{"replica-0", "replica-1"},
	}}, traffic.withdrawRequests)
	assert.Empty(t, traffic.admitRequests)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Equal(t, []string{"traffic.withdraw:" + workflowTestOperationID}, externalMutations)

	t.Log("Observe every former base identity drained and persist the zero-target fence-only barrier")
	traffic.snapshot = TrafficSnapshot{
		OperationID: workflowTestOperationID,
		Drained:     []ReplicaID{"replica-0", "replica-1"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Zero(t, result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, int64(2), result.ReleaseAuthorization.TopologyGeneration)
	require.Len(t, result.ReleaseAuthorization.Replicas, 2)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1"}, []ReplicaID{
		result.ReleaseAuthorization.Replicas[0].ReplicaID,
		result.ReleaseAuthorization.Replicas[1].ReplicaID,
	})
	assert.Empty(t, result.ReleaseAuthorization.Replicas[0].CapacityRefs)
	assert.Empty(t, result.ReleaseAuthorization.Replicas[1].CapacityRefs)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + workflowTestOperationID}, externalMutations)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart and issue the zero-target fence barrier without calling Admit with an empty replica set")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.Len(t, capacity.releaseCalls, 1)
	require.Len(t, capacity.releaseCalls[0].Replicas, 2)
	assert.Empty(t, traffic.admitRequests)

	t.Log("Observe Applied and persist the zero-target proof before completing compensation")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot.FencedReplicas = []ReplicaID{"replica-0", "replica-1"}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Zero(t, result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(2), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Restart from the durable zero-target proof and persist Aborted without traffic mutation")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{
		"traffic.withdraw:" + workflowTestOperationID,
		"capacity.release:" + workflowTestReleaseID,
	}, externalMutations)

}

func TestWorkflowCoordinatorRepairsAbortedServingStateAfterRestart(t *testing.T) {
	tests := []struct {
		name                string
		capacityAvailable   bool
		traffic             TrafficSnapshot
		wantCapacityRequest []CapacityRequest
		wantTrafficRequest  []TrafficRequest
	}{
		{
			name:              "unavailable active allocation",
			capacityAvailable: false,
			traffic: TrafficSnapshot{
				OperationID: workflowTestOperationID,
				Admitted:    []ReplicaID{"replica-0", "replica-1"},
				Drained:     []ReplicaID{"replica-2", "replica-3"},
			},
			wantCapacityRequest: []CapacityRequest{{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 2,
				TargetReplicas:     2,
				RequiredReplicas:   []ReplicaID{"replica-0", "replica-1"},
			}},
		},
		{
			name:              "lost survivor admission",
			capacityAvailable: true,
			traffic: TrafficSnapshot{
				OperationID: workflowTestOperationID,
				Admitted:    []ReplicaID{"replica-0"},
				Drained:     []ReplicaID{"replica-2", "replica-3"},
			},
			wantTrafficRequest: []TrafficRequest{{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 2,
				Replicas:           []ReplicaID{"replica-0", "replica-1"},
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			secondAllocation := workflowReplicaAllocation("replica-1", "pod-1")
			if !tt.capacityAvailable {
				secondAllocation.Availability = ReplicaAvailabilityUnavailable
			}
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{
					Allocations: []ReplicaAllocation{
						workflowReplicaAllocation("replica-0", "pod-0"),
						secondAllocation,
					},
					FencedReplicas: []ReplicaID{"replica-2", "replica-3"},
				},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                workflowTopology(2, "replica-0", "replica-1"),
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot:                tt.traffic,
				externalMutationHistory: &externalMutations,
			}
			operation := &Operation{
				ID:                     workflowTestOperationID,
				Attempt:                1,
				PlanID:                 "plan-1",
				Intent:                 OperationIntentShrink,
				SpecGeneration:         2,
				BaseTopologyGeneration: 1,
				BaseReplicas: []ReplicaID{
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				},
				TargetReplicas:             2,
				NominatedReplicas:          []ReplicaID{"replica-2", "replica-3"},
				CleanupReplicas:            []ReplicaID{"replica-2", "replica-3"},
				CapacityTargetReplicas:     2,
				CapacityTopologyGeneration: 2,
				CapacityTargetApplied:      true,
				Phase:                      OperationPhaseAborted,
				StartedAt:                  workflowTestTime.Add(-time.Minute),
				LastTransitionTime:         workflowTestTime,
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "BaseTopologyChanged",
				},
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Restart from a long-lived Aborted audit record after serving state regresses")
			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 2,
				Operation:       operation,
			})
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Equal(t, tt.wantCapacityRequest, capacity.ensureCalls)
			assert.Equal(t, tt.wantTrafficRequest, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Len(t, externalMutations, 1)
		})
	}
}

func TestWorkflowCoordinatorCompensatesPreparedShrinkAfterCapabilityLoss(t *testing.T) {
	tests := []struct {
		name    string
		phase   OperationPhase
		failure *OperationFailure
	}{
		{
			name:  "pending shrink",
			phase: OperationPhasePending,
		},
		{
			name:  "retryable failed shrink",
			phase: OperationPhaseFailed,
			failure: &OperationFailure{
				Classification: FailureClassificationRetryable,
				Reason:         "BackendBusy",
			},
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
			capabilities := MembershipCapabilities{Intents: []OperationIntent{OperationIntentGrow}}
			membership := &workflowMembershipAdapter{
				capabilities:            &capabilities,
				topology:                workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3"),
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: TrafficSnapshot{
					OperationID: workflowTestOperationID,
					Admitted:    []ReplicaID{"replica-0", "replica-1"},
					Drained:     []ReplicaID{"replica-2", "replica-3"},
				},
				externalMutationHistory: &externalMutations,
			}
			operation := &Operation{
				ID:                     workflowTestOperationID,
				Attempt:                1,
				PlanID:                 "plan-1",
				Intent:                 OperationIntentShrink,
				SpecGeneration:         2,
				BaseTopologyGeneration: 1,
				BaseReplicas: []ReplicaID{
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				},
				TargetReplicas:     2,
				NominatedReplicas:  []ReplicaID{"replica-2", "replica-3"},
				Phase:              tt.phase,
				StartedAt:          workflowTestTime.Add(-time.Minute),
				LastTransitionTime: workflowTestTime,
				Failure:            cloneFailure(tt.failure),
			}
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 2,
				Plan: &OperationPlan{
					ID:                "plan-1",
					Intent:            OperationIntentShrink,
					TargetReplicas:    2,
					NominatedReplicas: []ReplicaID{"replica-2", "replica-3"},
				},
				Operation: operation,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Persist capability-loss compensation without submitting or restoring traffic in the same reconcile")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.NotNil(t, result.Operation.Failure)
			assert.Equal(t, FailureClassificationTerminal, result.Operation.Failure.Classification)
			assert.Equal(t, "CapabilityLost", result.Operation.Failure.Reason)
			assert.True(t, result.OperationChanged)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, externalMutations)
			input.Operation = result.Operation

			t.Log("Restart from Aborting and persist the empty-victim target barrier before restoring traffic")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.NotNil(t, result.ReleaseAuthorization)
			assert.Equal(t, int32(4), result.ReleaseAuthorization.TargetReplicas)
			assert.Equal(t, int64(1), result.ReleaseAuthorization.TopologyGeneration)
			assert.Empty(t, result.ReleaseAuthorization.Replicas)
			assert.True(t, result.ReleaseAuthorizationChanged)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, externalMutations)
			input.ReleaseAuthorization = result.ReleaseAuthorization

			t.Log("Restart from durable authorization and issue the target barrier")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.Len(t, capacity.releaseCalls, 1)
			assert.Empty(t, capacity.releaseCalls[0].Replicas)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)

			t.Log("Observe Applied and persist the target proof before completing capability-loss compensation")
			capacity.releaseObservation = CapacityReleaseObservation{
				ReleaseID: workflowTestReleaseID,
				Phase:     CapacityReleasePhaseApplied,
			}
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			assert.Equal(t, int32(4), result.Operation.CapacityTargetReplicas)
			assert.Equal(t, int64(1), result.Operation.CapacityTopologyGeneration)
			assert.True(t, result.Operation.CapacityTargetApplied)
			assert.True(t, result.OperationChanged)
			assert.Nil(t, result.ReleaseAuthorization)
			input.Operation = result.Operation
			input.ReleaseAuthorization = nil

			t.Log("Restart from the durable target proof and restore every authoritative member")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Equal(t, []TrafficRequest{{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 1,
				Replicas: []ReplicaID{
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				},
			}}, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)

			t.Log("Restart after compensated traffic converges and persist Aborted")
			traffic.snapshot = TrafficSnapshot{
				OperationID: workflowTestOperationID,
				Admitted: []ReplicaID{
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				},
			}
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
			assert.True(t, result.OperationChanged)
			assert.Len(t, traffic.admitRequests, 1)
			assert.Empty(t, membership.submitCalls)
			assert.Equal(t, []string{
				"capacity.release:" + workflowTestReleaseID,
				"traffic.admit:" + workflowTestOperationID,
			}, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorCleansSurplusGrowthCapacityBeforeAborted(t *testing.T) {
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
		topology:                workflowTopology(1, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0", "replica-1"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         4,
		JoiningReplicas:        []ReplicaID{"replica-2", "replica-3"},
		Phase:                  OperationPhaseFailed,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "Rejected",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist Aborting before discovering or releasing surplus growth capacity")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Empty(t, result.Operation.CleanupReplicas)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart and durably freeze every non-active allocation as cleanup work")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, result.Operation.CleanupReplicas)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart and persist exact UID-bound release authorization without deleting capacity")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	require.Len(t, result.ReleaseAuthorization.Replicas, 2)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, int32(2), result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, int64(1), result.ReleaseAuthorization.TopologyGeneration)
	assert.Equal(t, []AuthorizedReplica{
		{
			ReplicaID: "replica-2",
			SlotID:    "slot-replica-2",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-2",
				UID:       "uid-pod-2",
			}},
		},
		{
			ReplicaID: "replica-3",
			SlotID:    "slot-replica-3",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-3",
				UID:       "uid-pod-3",
			}},
		},
	}, result.ReleaseAuthorization.Replicas)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, traffic.withdrawRequests, "never-active joiners require no traffic drain")
	assert.Empty(t, externalMutations)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from durable authorization and request only the exact surplus allocation release")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, *input.ReleaseAuthorization, capacity.releaseCalls[0])
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)

	t.Log("Observe release Applying without declaring cleanup or abort complete")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplying,
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.NotNil(t, result.ReleaseAuthorization)
	assert.Len(t, capacity.releaseCalls, 1)

	t.Log("Observe every surplus allocation absent and fenced before clearing release authorization")
	capacity.releaseObservation.Phase = CapacityReleasePhaseApplied
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		},
		FencedReplicas: []ReplicaID{"replica-2", "replica-3"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, int32(2), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(1), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Restart after durable fences and only then persist Aborted")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, result.Operation.CleanupReplicas)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorAppliesEmptyGrowthCapacityTargetBarrierBeforeAborted(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(1, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0", "replica-1"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         4,
		JoiningReplicas:        []ReplicaID{"replica-2", "replica-3"},
		Phase:                  OperationPhaseAborting,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "Rejected",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist an empty-victim absolute target barrier even though no surplus allocation is currently visible")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.False(t, result.Operation.CapacityTargetApplied)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, workflowTestReleaseID, result.ReleaseAuthorization.ID)
	assert.Equal(t, int32(2), result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, int64(1), result.ReleaseAuthorization.TopologyGeneration)
	assert.Empty(t, result.ReleaseAuthorization.Replicas)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from durable authorization and apply the target barrier without physical victims")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Empty(t, capacity.releaseCalls[0].Replicas)
	assert.Equal(t, int32(2), capacity.releaseCalls[0].TargetReplicas)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)

	t.Log("Observe Applied and durably record the exact target and topology proof before replacement is allowed")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, int32(2), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(1), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Restart from the durable target proof and only then persist Aborted")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
	assert.Equal(t, int32(2), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(1), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorReleasesLateSurplusAllocationAfterAborted(t *testing.T) {
	const lateReleaseID = "workflow-release-2"

	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{
			Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
				workflowReplicaAllocation("replica-4", "pod-4"),
			},
			FencedReplicas: []ReplicaID{"replica-2", "replica-3"},
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(1, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0", "replica-1"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                         workflowTestOperationID,
		Attempt:                    1,
		Intent:                     OperationIntentGrow,
		SpecGeneration:             2,
		BaseTopologyGeneration:     1,
		BaseReplicas:               []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:             4,
		JoiningReplicas:            []ReplicaID{"replica-2", "replica-3"},
		CleanupReplicas:            []ReplicaID{"replica-2", "replica-3"},
		CapacityTargetReplicas:     2,
		CapacityTopologyGeneration: 1,
		CapacityTargetApplied:      true,
		Phase:                      OperationPhaseAborted,
		StartedAt:                  workflowTestTime.Add(-time.Minute),
		LastTransitionTime:         workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "Rejected",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.newReleaseID = func() string { return lateReleaseID }

	t.Log("Discover a late surplus allocation and durably re-enter Aborting before release")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3", "replica-4"}, result.Operation.CleanupReplicas)
	assert.Zero(t, result.Operation.CapacityTargetReplicas)
	assert.Zero(t, result.Operation.CapacityTopologyGeneration)
	assert.False(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart and persist a separate release authorization including the exact late Pod UID")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.newReleaseID = func() string { return lateReleaseID }
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	require.Len(t, result.ReleaseAuthorization.Replicas, 3)
	assert.Equal(t, lateReleaseID, result.ReleaseAuthorization.ID)
	assert.Empty(t, result.ReleaseAuthorization.Replicas[0].CapacityRefs)
	assert.Empty(t, result.ReleaseAuthorization.Replicas[1].CapacityRefs)
	assert.Equal(t, []CapacityRef{{
		Namespace: "test",
		Name:      "pod-4",
		UID:       "uid-pod-4",
	}}, result.ReleaseAuthorization.Replicas[2].CapacityRefs)
	assert.Empty(t, capacity.releaseCalls)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from the separate authorization and release the late allocation")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, lateReleaseID, capacity.releaseCalls[0].ID)

	t.Log("Observe the late allocation absent and fenced before clearing its authorization")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: lateReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		},
		FencedReplicas: []ReplicaID{"replica-2", "replica-3", "replica-4"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, int32(2), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(1), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Restart after the separate cleanup and return to Aborted")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3", "replica-4"}, result.Operation.CleanupReplicas)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"capacity.release:" + lateReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorReleasesSurplusBeforeRepairingMissingAuthoritativeReplica(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-a", "pod-a"),
			workflowReplicaAllocation("replica-x", "pod-x"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-a", "replica-b"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-a"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-a", "replica-b"},
		TargetReplicas:         3,
		JoiningReplicas:        []ReplicaID{"replica-x"},
		Phase:                  OperationPhaseAborting,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "BaseTopologyChanged",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Durably identify the non-authoritative allocation before release or survivor repair")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-x"}, result.Operation.CleanupReplicas)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart and persist the exact target barrier before requesting the missing authoritative replica")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, int32(2), result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, int64(2), result.ReleaseAuthorization.TopologyGeneration)
	require.Len(t, result.ReleaseAuthorization.Replicas, 1)
	assert.Equal(t, ReplicaID("replica-x"), result.ReleaseAuthorization.Replicas[0].ReplicaID)
	assert.Empty(t, traffic.withdrawRequests, "the never-active joiner requires no drain")
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, externalMutations)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from durable authorization and release the surplus before any replacement allocation")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, *input.ReleaseAuthorization, capacity.releaseCalls[0])
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)

	t.Log("Observe the surplus fenced and durably persist the exact target proof")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations:    []ReplicaAllocation{workflowReplicaAllocation("replica-a", "pod-a")},
		FencedReplicas: []ReplicaID{"replica-x"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, int32(2), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(2), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Restart from the target proof and request the full authoritative identity set")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []CapacityRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     2,
		RequiredReplicas:   []ReplicaID{"replica-a", "replica-b"},
	}}, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []string{
		"capacity.release:" + workflowTestReleaseID,
		"capacity.ensure:2",
	}, externalMutations)

	t.Log("Observe both authoritative allocations and only then admit the complete topology")
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-a", "pod-a"),
			workflowReplicaAllocation("replica-b", "pod-b"),
		},
		FencedReplicas: []ReplicaID{"replica-x"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []TrafficRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		Replicas:           []ReplicaID{"replica-a", "replica-b"},
	}}, traffic.admitRequests)
	assert.Equal(t, []string{
		"capacity.release:" + workflowTestReleaseID,
		"capacity.ensure:2",
		"traffic.admit:" + workflowTestOperationID,
	}, externalMutations)

	t.Log("Observe complete admission and persist Aborted without replaying membership work")
	traffic.snapshot = TrafficSnapshot{
		OperationID: workflowTestOperationID,
		Admitted:    []ReplicaID{"replica-a", "replica-b"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
}

func TestWorkflowCoordinatorDrainsLostBaseReplicaBeforeAbortCleanupRelease(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0", "replica-1"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         3,
		JoiningReplicas:        []ReplicaID{"replica-2"},
		Phase:                  OperationPhaseAborting,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "BaseTopologyChanged",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Durably discover both the lost base replica and never-active joiner before cleanup side effects")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-1", "replica-2"}, result.Operation.CleanupReplicas)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, traffic.withdrawRequests)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart and withdraw only the former base replica before authorizing any capacity release")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []TrafficRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		Replicas:           []ReplicaID{"replica-1"},
	}}, traffic.withdrawRequests)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.releaseCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + workflowTestOperationID}, externalMutations)

	t.Log("Observe the former base replica drained and only then persist exact release authorization")
	traffic.snapshot = TrafficSnapshot{
		OperationID: workflowTestOperationID,
		Admitted:    []ReplicaID{"replica-0"},
		Drained:     []ReplicaID{"replica-1"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	require.Len(t, result.ReleaseAuthorization.Replicas, 2)
	assert.Equal(t, int32(1), result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, int64(2), result.ReleaseAuthorization.TopologyGeneration)
	assert.Equal(t, []ReplicaID{"replica-1", "replica-2"}, []ReplicaID{
		result.ReleaseAuthorization.Replicas[0].ReplicaID,
		result.ReleaseAuthorization.Replicas[1].ReplicaID,
	})
	assert.Empty(t, result.ReleaseAuthorization.Replicas[0].CapacityRefs)
	assert.Equal(t, []CapacityRef{{
		Namespace: "test",
		Name:      "pod-2",
		UID:       "uid-pod-2",
	}}, result.ReleaseAuthorization.Replicas[1].CapacityRefs)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.releaseCalls)
	assert.Len(t, traffic.withdrawRequests, 1)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + workflowTestOperationID}, externalMutations)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from the post-drain authorization and only then issue the exact release")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, []ReplicaID{"replica-1", "replica-2"}, []ReplicaID{
		capacity.releaseCalls[0].Replicas[0].ReplicaID,
		capacity.releaseCalls[0].Replicas[1].ReplicaID,
	})
	assert.Equal(t, []string{
		"traffic.withdraw:" + workflowTestOperationID,
		"capacity.release:" + workflowTestReleaseID,
	}, externalMutations)

	t.Log("Observe both cleanup identities absent and fenced before clearing the authorization")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations:    []ReplicaAllocation{workflowReplicaAllocation("replica-0", "pod-0")},
		FencedReplicas: []ReplicaID{"replica-1", "replica-2"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, int32(1), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(2), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Equal(t, []ReplicaID{"replica-1", "replica-2"}, result.Capacity.FencedReplicas)
}

func TestWorkflowCoordinatorRefusesAbortCleanupReplicaBecomingActive(t *testing.T) {
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
		topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0", "replica-1", "replica-2"},
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         4,
		JoiningReplicas:        []ReplicaID{"replica-2", "replica-3"},
		CleanupReplicas:        []ReplicaID{"replica-2", "replica-3"},
		Phase:                  OperationPhaseAborting,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "Rejected",
		},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Fail closed before traffic or release when a frozen cleanup identity becomes active")
	_, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.ErrorContains(t, err, "abort cleanup replicas became active")
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, externalMutations)
}

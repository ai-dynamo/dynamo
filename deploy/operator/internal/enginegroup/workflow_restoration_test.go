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

func TestWorkflowCoordinatorRestoresExactFencedReplicaWithoutSubstitution(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{
			Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
				workflowReplicaAllocation("replica-2", "pod-2"),
			},
			FencedReplicaSlots: workflowReplicaSlotBindings("replica-3"),
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
		Plan: &OperationPlan{
			ID:                 "restore-plan-1",
			Intent:             OperationIntentRecover,
			TargetReplicas:     4,
			RestoredMembership: workflowReplicaNativeMemberships("replica-3"),
		},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist the stable restoration identity before requesting replacement capacity")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Equal(t, OperationShapeReplacementRestoration, result.Operation.Capability.Shape)
	assert.Equal(t, workflowReplicaNativeMemberships("replica-3"), result.Operation.RestoredMembership)
	assert.Empty(t, result.Operation.JoiningReplicas)
	input.Operation = result.Operation

	t.Log("Request the exact fenced slot without substituting another allocation")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	require.Len(t, capacity.ensureCalls, 1)
	assert.Equal(t, CapacityRequest{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     4,
		RequiredReplicas: workflowRequiredReplicaAllocations(
			"replica-0", "replica-1", "replica-2", "replica-3",
		),
	}, capacity.ensureCalls[0])
	assert.Empty(t, result.Operation.JoiningReplicas)
	assert.Empty(t, membership.submitCalls)

	t.Log("After restart, preserve and re-request the same identity")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.ensureCalls, 2)
	assert.Equal(t, capacity.ensureCalls[0], capacity.ensureCalls[1])
	assert.Empty(t, result.Operation.JoiningReplicas)

	t.Log("Submit only after the exact restored slot is available and unfenced")
	replacement := workflowReplicaAllocation("replica-3", "replacement-pod-3")
	replacement.Incarnation.RuntimeID = "replacement-runtime-replica-3"
	capacity.snapshot = CapacitySnapshot{Allocations: []ReplicaAllocation{
		workflowReplicaAllocation("replica-0", "pod-0"),
		workflowReplicaAllocation("replica-1", "pod-1"),
		workflowReplicaAllocation("replica-2", "pod-2"),
		replacement,
	}}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Equal(t, []ReplicaIncarnation{replacement.Incarnation}, result.Operation.JoiningReplicas)
	input.Operation = result.Operation

	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	require.Len(t, membership.submitCalls, 1)
	assert.Equal(t, []ReplicaIncarnation{replacement.Incarnation}, membership.submitCalls[0].JoiningReplicas)
	assert.Equal(t, workflowReplicaNativeMemberships("replica-3"), membership.submitCalls[0].RestoredMembership)
}

func TestWorkflowCoordinatorCleansConflictingSurplusBeforeRetryingRestoration(t *testing.T) {
	externalMutations := make([]string, 0)
	baseAllocations := []ReplicaAllocation{
		workflowReplicaAllocation("replica-0", "pod-0"),
		workflowReplicaAllocation("replica-1", "pod-1"),
		workflowReplicaAllocation("replica-2", "pod-2"),
	}
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{
			Allocations: append(
				cloneCapacitySnapshot(CapacitySnapshot{Allocations: baseAllocations}).Allocations,
				workflowReplicaAllocation("replica-4", "pod-4"),
			),
			FencedReplicaSlots: workflowReplicaSlotBindings("replica-3"),
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	plan := &OperationPlan{
		ID:                 "restore-plan-1",
		Intent:             OperationIntentRecover,
		TargetReplicas:     4,
		RestoredMembership: workflowReplicaNativeMemberships("replica-3"),
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
		Plan:            plan,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist the exact restoration plan before inspecting its capacity prerequisites")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	input.Operation = result.Operation

	t.Log("Abort instead of substituting or leaking an unrelated allocation")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, "ConflictingSurplusCapacity", result.Operation.Failure.Reason)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Persist the restored slot and unrelated allocation as exact abort cleanup")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, workflowReplicaSlotBindings("replica-3", "replica-4"), result.Operation.CleanupReplicaSlots)
	assert.True(t, result.OperationChanged)
	input.Operation = result.Operation

	t.Log("Authorize the historical restoration fence and the unrelated Pod UID together")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, []AuthorizedReplica{
		{ReplicaID: "replica-3", SlotID: "slot-replica-3"},
		{
			ReplicaID: "replica-4",
			SlotID:    "slot-replica-4",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-4",
				UID:       "uid-pod-4",
			}},
		},
	}, result.ReleaseAuthorization.Replicas)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)

	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations:        baseAllocations,
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-3", "replica-4"),
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.Operation.CapacityTargetApplied)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
	input.Operation = result.Operation

	t.Log("A fresh plan may retry only after the conflicting allocation is absent and fenced")
	retryPlan := *plan
	retryPlan.ID = "restore-plan-2"
	input.Plan = &retryPlan
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = func() string { return "restore-operation-2" }
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, "restore-operation-2", result.Operation.ID)
	assert.Equal(t, retryPlan.ID, result.Operation.PlanID)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorRequiresExactHistoricalFenceForRestorationPlan(t *testing.T) {
	for _, tt := range []struct {
		name   string
		fences []ReplicaSlotBinding
		err    string
	}{
		{
			name: "historical fence is absent",
			err:  "has no durable capacity-slot fence",
		},
		{
			name: "historical fence names another slot",
			fences: []ReplicaSlotBinding{{
				ReplicaID: "replica-3",
				SlotID:    "other-slot",
			}},
			err: "must reuse fenced slot",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{
					Allocations: []ReplicaAllocation{
						workflowReplicaAllocation("replica-0", "pod-0"),
						workflowReplicaAllocation("replica-1", "pod-1"),
						workflowReplicaAllocation("replica-2", "pod-2"),
					},
					FencedReplicaSlots: tt.fences,
				},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2"),
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2")},
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  3,
				DesiredReplicas: 4,
				Plan: &OperationPlan{
					ID:                 "restore-plan-1",
					Intent:             OperationIntentRecover,
					TargetReplicas:     4,
					RestoredMembership: workflowReplicaNativeMemberships("replica-3"),
				},
			})
			require.ErrorContains(t, err, tt.err)
			assert.Nil(t, result.Operation)
			assert.Empty(t, capacity.ensureCalls)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestQueuedRestorationPlanCannotPreemptCurrentTrafficOrReleasePostwork(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
		},
		externalMutationHistory: &externalMutations,
	}
	operation := workflowCommittedShrinkOperation()
	durableDrain := workflowAcceptedTrafficCommand(
		TrafficActionWithdraw,
		1,
		workflowTestOperationID,
		2,
		workflowReplicaIncarnations("replica-1"),
	)
	queuedPlan := &OperationPlan{
		ID:                 "queued-restore-plan",
		Intent:             OperationIntentRecover,
		TargetReplicas:     2,
		RestoredMembership: workflowReplicaNativeMemberships("replica-2"),
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 2,
		Plan:            queuedPlan,
		Operation:       operation,
		TrafficRevision: 1,
		TrafficCommand:  durableDrain,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Replay the current operation's durable traffic command before validating a queued plan")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, operation, result.Operation)
	assert.False(t, result.OperationChanged)
	require.Len(t, traffic.withdrawRequests, 1)
	assert.Equal(t, durableDrain.Request, traffic.withdrawRequests[0])
	assert.Empty(t, membership.validatePlanCalls)

	t.Log("Finish the current drain and persist its exact release authorization before consuming the queued plan")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw,
		1,
		workflowTestOperationID,
		2,
		workflowReplicaIncarnations("replica-0"),
		workflowReplicaIncarnations("replica-1"),
	)
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	require.Len(t, result.ReleaseAuthorization.Replicas, 1)
	assert.Equal(t, ReplicaID("replica-1"), result.ReleaseAuthorization.Replicas[0].ReplicaID)
	assert.Empty(t, membership.validatePlanCalls)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Issue and observe the old operation's release without the queued plan blocking either step")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Empty(t, membership.validatePlanCalls)

	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations:        []ReplicaAllocation{workflowReplicaAllocation("replica-0", "pod-0")},
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-1"),
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.Empty(t, membership.validatePlanCalls)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Persist current-operation completion before the queued plan becomes eligible")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.Operation.PostCommitComplete)
	assert.Empty(t, membership.validatePlanCalls)
	input.Operation = result.Operation

	t.Log("Validate the queued restoration plan only when replacing the completed operation")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.ErrorContains(t, err, "restored replica \"replica-2\" has no durable capacity-slot fence")
	assert.Equal(t, input.Operation, result.Operation)
	assert.Empty(t, membership.validatePlanCalls)
	assert.Equal(t, []string{
		"traffic.withdraw:" + workflowTestOperationID,
		"capacity.release:" + workflowTestReleaseID,
	}, externalMutations)
}

func TestWorkflowCoordinatorFencesBackendNativeRestorationBeforeAdoption(t *testing.T) {
	for _, phase := range []OperationPhase{OperationPhasePending, OperationPhaseSubmitting} {
		t.Run(string(phase), func(t *testing.T) {
			externalMutations := make([]string, 0)
			baseTopology := workflowTopology(1, "replica-0", "replica-1", "replica-2")
			restoredTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3")
			capability := ResolvedOperationCapability{
				Shape:                   OperationShapeReplacementRestoration,
				TrafficRequirement:      ReconfigurationTrafficKeepServing,
				VerificationRequirement: ServingVerificationRequired,
			}
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{
					Allocations: []ReplicaAllocation{
						workflowReplicaAllocation("replica-0", "pod-0"),
						workflowReplicaAllocation("replica-1", "pod-1"),
						workflowReplicaAllocation("replica-2", "pod-2"),
						workflowReplicaAllocation("replica-3", "pod-3"),
					},
				},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                             restoredTopology,
				operation:                            BackendOperation{Phase: BackendOperationPhaseAbsent},
				validateObservedTransitionCapability: &capability,
				externalMutationHistory:              &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot:                TrafficSnapshot{Admitted: topologyReplicaIncarnations(restoredTopology)},
				externalMutationHistory: &externalMutations,
			}
			operation := &Operation{
				ID:                 "planned-restoration",
				Attempt:            1,
				PlanID:             "restore-plan-1",
				Intent:             OperationIntentRecover,
				Capability:         testOperationCapability(OperationShapeReplacementRestoration),
				SpecGeneration:     2,
				BaseTopology:       baseTopology,
				TargetReplicas:     4,
				RestoredMembership: workflowReplicaNativeMemberships("replica-3"),
				Phase:              phase,
				StartedAt:          workflowTestTime.Add(-time.Minute),
				LastTransitionTime: workflowTestTime,
			}
			if phase == OperationPhaseSubmitting {
				operation.JoiningReplicas = []ReplicaIncarnation{restoredTopology.Replicas[3].Incarnation}
			}
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 4,
				Plan: &OperationPlan{
					ID:                 "restore-plan-1",
					Intent:             OperationIntentRecover,
					TargetReplicas:     4,
					RestoredMembership: workflowReplicaNativeMemberships("replica-3"),
				},
				Operation: operation,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
			coordinator.operations.newOperationID = func() string { return operationRestorationAdoptedID }
			fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, restoredTopology)
			fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, restoredTopology)

			t.Log("Keep the old durable operation while fencing the externally restored topology after capacity consumed its slot fence")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, operation, result.Operation)
			assert.False(t, result.OperationChanged)
			assert.Empty(t, traffic.withdrawRequests)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			assert.Equal(t, []TrafficRequest{{
				Revision:           1,
				OperationID:        fenceOperationID,
				TopologyGeneration: restoredTopology.Generation,
				Replicas:           fencedIncarnations,
			}}, traffic.withdrawRequests)
			assert.Empty(t, membership.submitCalls)

			t.Log("Persist exact restored membership only after the old-operation fence survives restart")
			traffic.snapshot = workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw, 1, fenceOperationID, restoredTopology.Generation,
				nil, fencedIncarnations,
			)
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			coordinator.operations.newOperationID = func() string { return operationRestorationAdoptedID }
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.True(t, result.Operation.Adopted)
			assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
			assert.Equal(t, workflowReplicaNativeMemberships("replica-3"), result.Operation.RestoredMembership)
			assert.Equal(t, []ReplicaIncarnation{restoredTopology.Replicas[3].Incarnation}, result.Operation.JoiningReplicas)
			assert.Equal(t, restoredTopology, *result.Operation.CommittedTopology)
			assert.True(t, result.OperationChanged)
			assert.Empty(t, membership.submitCalls)
		})
	}
}

func TestWorkflowCoordinatorAdoptsRestorationRacingSubmissionChecks(t *testing.T) {
	baseTopology := workflowTopology(1, "replica-0", "replica-1", "replica-2")
	restoredTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3")
	replacementCapability := ResolvedOperationCapability{
		Shape:                   OperationShapeReplacementRestoration,
		TrafficRequirement:      ReconfigurationTrafficKeepServing,
		VerificationRequirement: ServingVerificationRequired,
	}

	tests := []struct {
		name                 string
		phase                OperationPhase
		topologyObservations []MembershipTopology
		validateRequestErr   error
		wantValidationCalls  int
		wantObservationCalls int
	}{
		{
			name:                 "pending preparation rejection observes native restoration",
			phase:                OperationPhasePending,
			topologyObservations: []MembershipTopology{baseTopology, restoredTopology},
			validateRequestErr:   ErrMembershipOperationUnsupported,
			wantValidationCalls:  1,
		},
		{
			name:                 "submitting operation observes restoration before fresh preflight",
			phase:                OperationPhaseSubmitting,
			topologyObservations: []MembershipTopology{baseTopology, restoredTopology},
			wantObservationCalls: 2,
		},
		{
			name:  "submitting preflight rejection reobserves native restoration",
			phase: OperationPhaseSubmitting,
			topologyObservations: []MembershipTopology{
				baseTopology,
				baseTopology,
				restoredTopology,
			},
			validateRequestErr:   ErrMembershipOperationUnsupported,
			wantValidationCalls:  1,
			wantObservationCalls: 2,
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
				topology:                             baseTopology,
				topologyObservations:                 cloneTopologySequence(tt.topologyObservations),
				operation:                            BackendOperation{Phase: BackendOperationPhaseAbsent},
				validateRequestErr:                   tt.validateRequestErr,
				validateObservedTransitionCapability: &replacementCapability,
				externalMutationHistory:              &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: TrafficSnapshot{
					Admitted: topologyReplicaIncarnations(baseTopology),
				},
				externalMutationHistory: &externalMutations,
			}
			operation := &Operation{
				ID:                 "planned-restoration",
				Attempt:            1,
				PlanID:             "restore-plan-1",
				Intent:             OperationIntentRecover,
				Capability:         testOperationCapability(OperationShapeReplacementRestoration),
				SpecGeneration:     2,
				BaseTopology:       baseTopology,
				TargetReplicas:     4,
				RestoredMembership: workflowReplicaNativeMemberships("replica-3"),
				Phase:              tt.phase,
				StartedAt:          workflowTestTime.Add(-time.Minute),
				LastTransitionTime: workflowTestTime,
			}
			if tt.phase == OperationPhaseSubmitting {
				operation.JoiningReplicas = []ReplicaIncarnation{restoredTopology.Replicas[3].Incarnation}
			}
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  operation.SpecGeneration,
				DesiredReplicas: operation.TargetReplicas,
				Plan: &OperationPlan{
					ID:                 operation.PlanID,
					Intent:             operation.Intent,
					TargetReplicas:     operation.TargetReplicas,
					RestoredMembership: cloneReplicaNativeMemberships(operation.RestoredMembership),
				},
				Operation: operation,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
			coordinator.operations.newOperationID = func() string { return operationRestorationAdoptedID }
			fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, restoredTopology)
			fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, restoredTopology)

			t.Log("Keep the old durable request while persisting a fence for the raced restoration")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, operation, result.Operation)
			assert.False(t, result.OperationChanged)
			assert.True(t, result.TrafficStateChanged)
			require.NotNil(t, result.TrafficCommand)
			assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
			assert.Equal(t, fenceOperationID, result.TrafficCommand.Request.OperationID)
			assert.Equal(t, fencedIncarnations, result.TrafficCommand.Request.Replicas)
			assert.Len(t, membership.validateRequestCalls, tt.wantValidationCalls)
			assert.Len(t, membership.observeOperationCalls, tt.wantObservationCalls)
			assert.Empty(t, membership.submitCalls)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)

			t.Log("Adopt only after the exact raced topology is durably fenced")
			traffic.snapshot = workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw,
				result.TrafficRevision,
				fenceOperationID,
				restoredTopology.Generation,
				nil,
				fencedIncarnations,
			)
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			coordinator.operations.newOperationID = func() string { return operationRestorationAdoptedID }
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.True(t, result.Operation.Adopted)
			assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
			assert.Equal(t, operation.RestoredMembership, result.Operation.RestoredMembership)
			assert.Equal(t, []ReplicaIncarnation{restoredTopology.Replicas[3].Incarnation}, result.Operation.JoiningReplicas)
			assert.Equal(t, restoredTopology, *result.Operation.CommittedTopology)
			assert.True(t, result.OperationChanged)
			assert.Empty(t, membership.submitCalls)
		})
	}
}

func cloneTopologySequence(topologies []MembershipTopology) []MembershipTopology {
	cloned := make([]MembershipTopology, len(topologies))
	for i := range topologies {
		cloned[i] = cloneTopology(topologies[i])
	}
	return cloned
}

func TestValidateAdoptedRestorationRequiresConsistentHistoricalSlot(t *testing.T) {
	previous := &Operation{RestoredMembership: workflowReplicaNativeMemberships("replica-3")}
	adopted := Operation{RestoredMembership: workflowReplicaNativeMemberships("replica-3")}
	replacement := workflowReplicaAllocation("replica-3", "replacement-pod-3")
	replacement.Incarnation.RuntimeID = "replacement-runtime-replica-3"

	t.Log("The old durable plan proves a fence that normal replacement allocation already consumed")
	require.NoError(t, validateAdoptedRestoration(previous, adopted, CapacitySnapshot{
		Allocations: []ReplicaAllocation{replacement},
	}))

	t.Log("Observed restoration without durable history remains invalid")
	require.ErrorContains(t, validateAdoptedRestoration(nil, adopted, CapacitySnapshot{
		Allocations: []ReplicaAllocation{replacement},
	}), "requires durable historical membership")

	t.Log("Current capacity cannot move the restored identity to a different slot")
	replacement.Incarnation.SlotID = "conflicting-slot"
	require.ErrorContains(t, validateAdoptedRestoration(previous, adopted, CapacitySnapshot{
		Allocations: []ReplicaAllocation{replacement},
	}), "instead of historical slot")

	t.Log("Still-observable history must agree with the durable plan")
	require.ErrorContains(t, validateAdoptedRestoration(previous, adopted, CapacitySnapshot{
		FencedReplicaSlots: []ReplicaSlotBinding{{
			ReplicaID: "replica-3",
			SlotID:    "conflicting-slot",
		}},
	}), "conflicting stable capacity slots")
}

func TestPostCommitObservedRestorationRequiresHistoricalSlotEvidence(t *testing.T) {
	restoredTopology := workflowTopology(3, "replica-0", "replica-1", "replica-2", "replica-3")
	recoveryCapability := ResolvedOperationCapability{
		Shape:                   OperationShapeReplacementRestoration,
		TrafficRequirement:      ReconfigurationTrafficKeepServing,
		VerificationRequirement: ServingVerificationRequired,
	}
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Phase = OperationPhaseFailed
	operation.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "ServingVerificationFailed",
		Message:        "the committed topology could not make progress",
	}
	plan := &OperationPlan{
		ID:                 "restore-plan",
		Intent:             OperationIntentRecover,
		TargetReplicas:     4,
		RestoredMembership: workflowReplicaNativeMemberships("replica-3"),
	}
	fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, restoredTopology)
	fenceOperationID := verificationFailureTrafficOperationID(*operation, restoredTopology)

	tests := []struct {
		name      string
		capacity  CapacitySnapshot
		wantError string
	}{
		{
			name: "missing historical slot",
			capacity: CapacitySnapshot{Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
				workflowReplicaAllocation("replica-2", "pod-2"),
				workflowReplicaAllocation("replica-3", "pod-3"),
			}},
			wantError: "no durable historical capacity slot",
		},
		{
			name: "conflicting historical slot",
			capacity: CapacitySnapshot{
				Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-1", "pod-1"),
					workflowReplicaAllocation("replica-2", "pod-2"),
				},
				FencedReplicaSlots: []ReplicaSlotBinding{{
					ReplicaID: "replica-3",
					SlotID:    "conflicting-slot",
				}},
			},
			wantError: "must reuse historical slot \"conflicting-slot\"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			capacity := &workflowCapacityAdapter{
				snapshot:                tt.capacity,
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                             restoredTopology,
				validateObservedTransitionCapability: &recoveryCapability,
				externalMutationHistory:              &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: workflowTrafficSnapshotWithCommand(
					TrafficActionWithdraw,
					1,
					fenceOperationID,
					restoredTopology.Generation,
					nil,
					fencedIncarnations,
				),
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  3,
				DesiredReplicas: 4,
				Plan:            plan,
				Operation:       cloneOperation(operation),
			})
			require.ErrorContains(t, err, tt.wantError)
			assert.Equal(t, operation, result.Operation)
			assert.False(t, result.OperationChanged)
			require.Len(t, membership.validateObservedTransitionCalls, 1)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, capacity.ensureCalls)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, traffic.withdrawRequests)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorFreshGrowthDoesNotReopenFencedReplica(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{
			Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
				workflowReplicaAllocation("replica-2", "pod-2"),
			},
			FencedReplicaSlots: workflowReplicaSlotBindings("replica-3"),
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Create count-only FreshGrowth without claiming a fenced stable identity")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationShapeFreshGrowth, result.Operation.Capability.Shape)
	assert.Empty(t, result.Operation.JoiningReplicas)
	input.Operation = result.Operation

	t.Log("Ask for anonymous-new capacity while preserving the fence")
	_, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.ensureCalls, 1)
	assert.Equal(t, CapacityRequest{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     4,
		RequiredReplicas:   workflowRequiredReplicaAllocations("replica-0", "replica-1", "replica-2"),
	}, capacity.ensureCalls[0])
	assert.NotContains(
		t,
		capacity.ensureCalls[0].RequiredReplicas,
		RequiredReplicaAllocation{ReplicaID: "replica-3", SlotID: "slot-replica-3"},
	)
}

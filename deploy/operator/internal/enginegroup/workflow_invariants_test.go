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

func TestWorkflowCoordinatorRetriesWithANewAttempt(t *testing.T) {
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
		snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-1")},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                 workflowTestOperationID,
		Attempt:            1,
		Intent:             OperationIntentGrow,
		Capability:         testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:     2,
		BaseTopology:       workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:     4,
		JoiningReplicas:    workflowReplicaIncarnations("replica-2", "replica-3"),
		Phase:              OperationPhaseFailed,
		BackendOperationID: "backend-attempt-1",
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime.Add(-time.Second),
		Failure: &OperationFailure{
			Classification: FailureClassificationRetryable,
			Reason:         "TemporarilyUnavailable",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist a distinct retry attempt while preserving the logical operation and frozen payload")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, workflowTestOperationID, result.Operation.ID)
	assert.Equal(t, int32(2), result.Operation.Attempt)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Empty(t, result.Operation.BackendOperationID)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Restart, observe that only the new attempt is absent, and submit it exactly once")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	require.Len(t, membership.submitCalls, 1)
	assert.Equal(t, int32(2), membership.submitCalls[0].Attempt)
	assert.Equal(t, workflowReplicaIncarnations("replica-2", "replica-3"), membership.submitCalls[0].JoiningReplicas)
	assert.Equal(t, []string{"membership.submit:" + workflowTestOperationID}, externalMutations)
}

func TestWorkflowCoordinatorPersistsTerminalFailureCompensation(t *testing.T) {
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
		snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-1")},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                 workflowTestOperationID,
		Attempt:            1,
		Intent:             OperationIntentGrow,
		Capability:         testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:     2,
		BaseTopology:       workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:     3,
		JoiningReplicas:    workflowReplicaIncarnations("replica-2"),
		Phase:              OperationPhaseFailed,
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "Unsupported",
		},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist compensation for a terminal failure before mutating another subsystem")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, operation.Failure, result.Operation.Failure)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorRejectsUnsupportedRecoveryBeforeMutation(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		externalMutationHistory: &externalMutations,
	}
	capabilities := MembershipCapabilities{OperationShapes: []OperationShape{
		OperationShapeFreshGrowth,
	}}
	membership := &workflowMembershipAdapter{
		capabilities:            &capabilities,
		topology:                workflowTopology(1, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-1")},
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Fail closed before traffic withdrawal or membership submission when survivor recovery is unsupported")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Plan: &OperationPlan{
			ID:                "recovery-plan",
			Intent:            OperationIntentRecover,
			TargetReplicas:    1,
			NominatedReplicas: []ReplicaID{"replica-1"},
		},
	})
	require.ErrorContains(t, err, "membership operation does not support plan intent \"Recover\"")
	assert.Nil(t, result.Operation)
	assert.Equal(t, 1, membership.observeCapabilityCalls)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, traffic.withdrawCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorRequestsFrozenCapacityIdentities(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
			workflowReplicaAllocation("replica-4", "pod-4"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(1, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-1")},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                 workflowTestOperationID,
		Attempt:            1,
		Intent:             OperationIntentGrow,
		Capability:         testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:     2,
		BaseTopology:       workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:     4,
		JoiningReplicas:    workflowReplicaIncarnations("replica-2", "replica-3"),
		Phase:              OperationPhaseSubmitting,
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Abort when an unrelated allocation cannot satisfy the frozen missing joiner incarnation")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.CompensationTopology)
	assert.True(t, servingVerificationTopologiesEqual(operation.BaseTopology, *result.Operation.CompensationTopology))
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, "JoiningIncarnationLost", result.Operation.Failure.Reason)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, capacity.ensureCalls)
}

func TestWorkflowCoordinatorRestoresBaseCapacityBeforeSubmittingRetirement(t *testing.T) {
	externalMutations := make([]string, 0)
	unavailable := workflowReplicaAllocation("replica-1", "pod-1")
	unavailable.Availability = ReplicaAvailabilityUnavailable
	baseTopology := workflowTopology(1, "replica-0", "replica-1")
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			unavailable,
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                baseTopology,
		externalMutationHistory: &externalMutations,
	}
	trafficCommand := workflowAcceptedTrafficCommand(
		TrafficActionWithdraw,
		1,
		workflowTestOperationID,
		baseTopology.Generation,
		topologyReplicaIncarnations(baseTopology),
	)
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw,
			trafficCommand.Request.Revision,
			workflowTestOperationID,
			baseTopology.Generation,
			nil,
			topologyReplicaIncarnations(baseTopology),
		),
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                 workflowTestOperationID,
		Attempt:            1,
		PlanID:             "retire-plan-1",
		Intent:             OperationIntentRetire,
		Capability:         testOperationCapability(OperationShapeFullRetirement),
		SpecGeneration:     2,
		BaseTopology:       baseTopology,
		TargetReplicas:     0,
		NominatedReplicas:  []ReplicaID{"replica-0", "replica-1"},
		Phase:              OperationPhaseSubmitting,
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 0,
		Operation:       operation,
		TrafficRevision: trafficCommand.Request.Revision,
		TrafficCommand:  trafficCommand,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Equal(t, []CapacityRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: baseTopology.Generation,
		TargetReplicas:     baseTopology.ReplicaCount(),
		RequiredReplicas:   workflowRequiredReplicaAllocations("replica-0", "replica-1"),
	}}, capacity.ensureCalls)
	assert.Empty(t, membership.submitCalls)
}

func TestWorkflowCoordinatorUsesDistinctReleaseIDsForReplacementCapacity(t *testing.T) {
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, 2,
			workflowReplicaIncarnations("replica-0"), workflowReplicaIncarnations("replica-1"),
		),
		externalMutationHistory: &externalMutations,
	}
	releaseIDs := &fakeOperationIDGenerator{ids: []string{"release-1", "release-2"}}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.newReleaseID = releaseIDs.Next
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 1,
		Operation:       workflowCommittedShrinkOperation(),
	}

	t.Log("Persist the first exact release request under its own idempotency identity")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, "release-1", result.ReleaseAuthorization.ID)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Clear only the first request after its old UID is absent while a concurrent replacement remains")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: "release-1",
		Phase:     CapacityReleasePhaseApplied,
	}
	replacement := workflowReplicaAllocation("replica-1", "pod-1")
	replacement.Incarnation.CapacityRefs[0].UID = "replacement-uid"
	replacement.Incarnation.RuntimeID = operationRestorationReplacementRun
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			replacement,
		},
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-1"),
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	input.ReleaseAuthorization = nil

	t.Log("Give the concurrent replacement a new release identity instead of inheriting the old result")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, "release-2", result.ReleaseAuthorization.ID)
	assert.Equal(t, workflowTestOperationID, result.ReleaseAuthorization.OperationID)
	assert.Equal(t, PodUID("replacement-uid"), result.ReleaseAuthorization.Replicas[0].CapacityRefs[0].UID)
}

func TestWorkflowCoordinatorHandlesExplicitReleaseRefusal(t *testing.T) {
	tests := []struct {
		name              string
		classification    FailureClassification
		wantAuthorization bool
		wantChanged       bool
	}{
		{
			name:              "retryable refusal permits a fresh release request",
			classification:    FailureClassificationRetryable,
			wantAuthorization: false,
			wantChanged:       true,
		},
		{
			name:              "terminal refusal retains evidence for intervention",
			classification:    FailureClassificationTerminal,
			wantAuthorization: true,
			wantChanged:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-1", "pod-1"),
				}},
				releaseObservation: CapacityReleaseObservation{
					ReleaseID: workflowTestReleaseID,
					Phase:     CapacityReleasePhaseRefused,
					Failure: &OperationFailure{
						Classification: tt.classification,
						Reason:         "WorkloadManagerRefused",
					},
				},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                workflowTopology(2, "replica-0"),
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: workflowTrafficSnapshotWithCommand(
					TrafficActionWithdraw, 1, workflowTestOperationID, 2,
					workflowReplicaIncarnations("replica-0"), workflowReplicaIncarnations("replica-1"),
				),
				externalMutationHistory: &externalMutations,
			}
			authorization := &ReleaseAuthorization{
				ID:                 workflowTestReleaseID,
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 2,
				TargetReplicas:     1,
				Replicas: []AuthorizedReplica{{
					ReplicaID:    "replica-1",
					SlotID:       "slot-replica-1",
					CapacityRefs: []CapacityRef{{Namespace: "test", Name: "pod-1", UID: "uid-pod-1"}},
				}},
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Use refusal classification to decide whether the next reconcile may mint a fresh request")
			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:              "group-0",
				SpecGeneration:       2,
				DesiredReplicas:      1,
				Operation:            workflowCommittedShrinkOperation(),
				ReleaseAuthorization: authorization,
			})
			require.ErrorContains(t, err, "was refused")
			assert.Equal(t, tt.wantAuthorization, result.ReleaseAuthorization != nil)
			assert.Equal(t, tt.wantChanged, result.ReleaseAuthorizationChanged)
			assert.Empty(t, capacity.releaseCalls)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorValidatesTrafficIdentitiesBeforeMutation(t *testing.T) {
	tests := []struct {
		name      string
		operation *Operation
		traffic   TrafficSnapshot
		wantError string
	}{
		{
			name:      "admitted identity is not committed",
			operation: workflowPendingGrowthOperation(),
			traffic: TrafficSnapshot{
				Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-9"),
			},
			wantError: "replica-9",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
				snapshot:                tt.traffic,
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Fail closed on a traffic identity set that cannot describe the observed membership operation")
			_, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: tt.operation.TargetReplicas,
				Operation:       tt.operation,
			})
			require.ErrorContains(t, err, tt.wantError)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorRejectsRogueTrafficWhenMembershipIsOtherwiseConverged(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(1, "replica-0"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-9")},
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Reject admitted traffic outside authoritative topology even when no membership operation is needed")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 1,
	})
	require.ErrorContains(t, err, "replica-9")
	assert.Nil(t, result.Operation)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorFencesUncorrelatedNativeRemapWithoutCommitProof(t *testing.T) {
	externalMutations := make([]string, 0)
	currentTopology := workflowTopology(2, "replica-0", "replica-1")
	currentTopology.Replicas[1].NativeMembers = []NativeMemberID{"native-replica-1-replacement"}
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                currentTopology,
		operation:               BackendOperation{Phase: BackendOperationPhaseAbsent},
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-1")},
		externalMutationHistory: &externalMutations,
	}
	operation := workflowPendingGrowthOperation()
	operation.Phase = OperationPhaseUnknown
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Fence every currently observed identity when Unknown has no exact commit proof for a native remap")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Nil(t, result.Operation.CommittedTopology)
	assert.False(t, result.OperationChanged)
	fenceOperationID := unverifiedTopologyTrafficOperationID(*result.Operation, currentTopology)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        fenceOperationID,
		TopologyGeneration: currentTopology.Generation,
		Replicas:           workflowReplicaIncarnations("replica-0", "replica-1"),
	}}, traffic.withdrawRequests)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart preserves the accepted deterministic whole-topology fence until its effect is observable")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	require.Len(t, traffic.withdrawRequests, 1)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, capacity.releaseCalls)
}

func TestWorkflowCoordinatorKeepsUnknownMembershipWhileObservingRelease(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology: workflowTopology(3, "replica-0"),
		operation: BackendOperation{
			ID:             workflowTestOperationID,
			Attempt:        1,
			TargetReplicas: 1,
			Phase:          BackendOperationPhaseAccepted,
		},
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, 2,
			workflowReplicaIncarnations("replica-0"), workflowReplicaIncarnations("replica-1"),
		),
		externalMutationHistory: &externalMutations,
	}
	operation := workflowCommittedShrinkOperation()
	operation.Phase = OperationPhaseUnknown
	authorization := &ReleaseAuthorization{
		ID:                 workflowTestReleaseID,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     1,
		Replicas: []AuthorizedReplica{{
			ReplicaID:    "replica-1",
			SlotID:       "slot-replica-1",
			CapacityRefs: []CapacityRef{{Namespace: "test", Name: "pod-1", UID: "uid-pod-1"}},
		}},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	input := ReconcileInput{
		GroupID:              "group-0",
		SpecGeneration:       2,
		DesiredReplicas:      1,
		Operation:            operation,
		ReleaseAuthorization: authorization,
	}

	t.Log("Fence the later survivor topology before observing or discarding the durable release")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Equal(t, authorization, result.ReleaseAuthorization)
	assert.False(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.observeReleaseCalls)
	assert.Equal(t, 1, membership.observeTopologyCalls)
	assert.Empty(t, membership.observeOperationCalls)
	assert.Empty(t, capacity.releaseCalls)
	fenceOperationID := unverifiedTopologyTrafficOperationID(*result.Operation, membership.topology)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           2,
		OperationID:        fenceOperationID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           workflowReplicaIncarnations("replica-0", "replica-1"),
	}}, traffic.withdrawRequests)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

	t.Log("Restart from the exact fence and discard the unissued stale-generation authorization")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw, 2, fenceOperationID, membership.topology.Generation,
		nil, workflowReplicaIncarnations("replica-0", "replica-1"),
	)
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, []string{workflowTestReleaseID}, capacity.observeReleaseCalls)
	assert.Equal(t, 3, membership.observeTopologyCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Len(t, traffic.withdrawRequests, 1)
}

func TestWorkflowCoordinatorRejectsDrainRegressionDuringUncertainRelease(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		releaseObservation: CapacityReleaseObservation{
			ReleaseID: workflowTestReleaseID,
			Phase:     CapacityReleasePhaseApplying,
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionAdmit, 1, workflowTestOperationID, 2,
			workflowReplicaIncarnations("replica-0", "replica-1"), nil,
		),
		externalMutationHistory: &externalMutations,
	}
	operation := workflowCommittedShrinkOperation()
	operation.Phase = OperationPhaseUnknown
	authorization := &ReleaseAuthorization{
		ID:                 workflowTestReleaseID,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     1,
		Replicas: []AuthorizedReplica{{
			ReplicaID:    "replica-1",
			SlotID:       "slot-replica-1",
			CapacityRefs: []CapacityRef{{Namespace: "test", Name: "pod-1", UID: "uid-pod-1"}},
		}},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Fail closed when a previously drained victim becomes routable during an outstanding deletion")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:              "group-0",
		SpecGeneration:       2,
		DesiredReplicas:      1,
		Operation:            operation,
		ReleaseAuthorization: authorization,
	})
	require.ErrorContains(t, err, "traffic drain")
	assert.Equal(t, authorization, result.ReleaseAuthorization)
	assert.Empty(t, capacity.observeReleaseCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorRetainsUncertainReleaseUntilFenceIsObservable(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
		}},
		releaseObservation: CapacityReleaseObservation{
			ReleaseID: workflowTestReleaseID,
			Phase:     CapacityReleasePhaseApplied,
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, 2,
			workflowReplicaIncarnations("replica-0"), workflowReplicaIncarnations("replica-1"),
		),
		externalMutationHistory: &externalMutations,
	}
	operation := workflowCommittedShrinkOperation()
	operation.Phase = OperationPhaseUnknown
	authorization := &ReleaseAuthorization{
		ID:                 workflowTestReleaseID,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     1,
		Replicas: []AuthorizedReplica{{
			ReplicaID:    "replica-1",
			SlotID:       "slot-replica-1",
			CapacityRefs: []CapacityRef{{Namespace: "test", Name: "pod-1", UID: "uid-pod-1"}},
		}},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	input := ReconcileInput{
		GroupID:              "group-0",
		SpecGeneration:       2,
		DesiredReplicas:      1,
		Operation:            operation,
		ReleaseAuthorization: authorization,
	}

	t.Log("Retain release correlation when the old Pod is gone but its durable fence is not yet visible")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, authorization, result.ReleaseAuthorization)
	assert.False(t, result.ReleaseAuthorizationChanged)

	t.Log("Clear release correlation only after the capacity adapter exposes the matching fence")
	capacity.snapshot.FencedReplicaSlots = workflowReplicaSlotBindings("replica-1")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorHandlesStoppedPartialCapacityRelease(t *testing.T) {
	newFixture := func(phase CapacityReleasePhase, failure *OperationFailure) (
		*workflowCoordinatorTestHarness,
		*workflowCapacityAdapter,
		ReconcileInput,
		*ReleaseAuthorization,
	) {
		externalMutations := make([]string, 0)
		capacity := &workflowCapacityAdapter{
			snapshot: CapacitySnapshot{
				Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-2", "pod-2"),
					workflowReplicaAllocation("replica-3", "pod-3"),
				},
				FencedReplicaSlots: workflowReplicaSlotBindings("replica-1"),
			},
			releaseObservation: CapacityReleaseObservation{
				ReleaseID: workflowTestReleaseID,
				Phase:     phase,
				Failure:   failure,
			},
			externalMutationHistory: &externalMutations,
		}
		membership := &workflowMembershipAdapter{
			topology:                workflowTopology(2, "replica-0", "replica-2"),
			externalMutationHistory: &externalMutations,
		}
		traffic := &workflowTrafficAdapter{
			snapshot: workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw, 1, workflowTestOperationID, 2,
				workflowReplicaIncarnations("replica-0", "replica-2"),
				workflowReplicaIncarnations("replica-1", "replica-3"),
			),
			externalMutationHistory: &externalMutations,
		}
		operation := &Operation{
			ID:                         workflowTestOperationID,
			Attempt:                    1,
			PlanID:                     "plan-1",
			Intent:                     OperationIntentShrink,
			Capability:                 testOperationCapability(OperationShapePlannedSelectedRetirement),
			SpecGeneration:             2,
			BaseTopology:               workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3"),
			TargetReplicas:             2,
			NominatedReplicas:          []ReplicaID{"replica-1", "replica-3"},
			Phase:                      OperationPhaseCommitted,
			CommittedTopology:          topologyPointer(workflowTopology(2, "replica-0", "replica-2")),
			CapacityTargetReplicas:     2,
			CapacityTopologyGeneration: 2,
			CapacityTargetApplied:      true,
			StartedAt:                  workflowTestTime.Add(-time.Minute),
			LastTransitionTime:         workflowTestTime,
		}
		authorization := &ReleaseAuthorization{
			ID:                 workflowTestReleaseID,
			OperationID:        workflowTestOperationID,
			TopologyGeneration: 2,
			TargetReplicas:     2,
			Replicas: []AuthorizedReplica{
				{ReplicaID: "replica-1", SlotID: "slot-replica-1"},
				{
					ReplicaID: "replica-3",
					SlotID:    "slot-replica-3",
					CapacityRefs: []CapacityRef{{
						Namespace: "test",
						Name:      "pod-3",
						UID:       "uid-pod-3",
					}},
				},
			},
		}
		return newWorkflowCoordinatorForTest(capacity, membership, traffic), capacity, ReconcileInput{
			GroupID:              "group-0",
			SpecGeneration:       2,
			DesiredReplicas:      2,
			Operation:            operation,
			ReleaseAuthorization: authorization,
		}, authorization
	}

	t.Run("applying remains nonterminal", func(t *testing.T) {
		coordinator, capacity, input, authorization := newFixture(CapacityReleasePhaseApplying, nil)
		result, err := coordinator.Reconcile(context.Background(), input)
		require.NoError(t, err)
		assert.Equal(t, authorization, result.ReleaseAuthorization)
		assert.False(t, result.ReleaseAuthorizationChanged)
		assert.True(t, result.Operation.CapacityTargetApplied)
		assert.Empty(t, capacity.releaseCalls)
	})

	t.Run("retryable failure replans from partial capacity and fences", func(t *testing.T) {
		coordinator, capacity, input, _ := newFixture(CapacityReleasePhaseFailed, &OperationFailure{
			Classification: FailureClassificationRetryable,
			Reason:         "ReleaseInterrupted",
		})

		result, err := coordinator.Reconcile(context.Background(), input)
		require.ErrorContains(t, err, "ReleaseInterrupted")
		assert.Nil(t, result.ReleaseAuthorization)
		assert.True(t, result.ReleaseAuthorizationChanged)
		assert.False(t, result.Operation.CapacityTargetApplied)
		assert.Zero(t, result.Operation.CapacityTargetReplicas)
		assert.Zero(t, result.Operation.CapacityTopologyGeneration)
		input.Operation = result.Operation
		input.ReleaseAuthorization = nil

		capacity.releaseObservation = CapacityReleaseObservation{}
		coordinator = newWorkflowCoordinatorForTest(capacity, coordinator.membership, coordinator.traffic)
		coordinator.newReleaseID = func() string { return "workflow-release-2" }
		result, err = coordinator.Reconcile(context.Background(), input)
		require.NoError(t, err)
		require.NotNil(t, result.ReleaseAuthorization)
		assert.Equal(t, "workflow-release-2", result.ReleaseAuthorization.ID)
		assert.Equal(t, []AuthorizedReplica{
			{ReplicaID: "replica-1", SlotID: "slot-replica-1"},
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
	})

	t.Run("terminal failure retains authority and fails closed", func(t *testing.T) {
		coordinator, capacity, input, authorization := newFixture(CapacityReleasePhaseFailed, &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "ReleaseStopped",
		})

		result, err := coordinator.Reconcile(context.Background(), input)
		require.ErrorContains(t, err, "ReleaseStopped")
		assert.Equal(t, authorization, result.ReleaseAuthorization)
		assert.False(t, result.ReleaseAuthorizationChanged)
		assert.True(t, result.Operation.CapacityTargetApplied)
		assert.Empty(t, capacity.releaseCalls)
	})
}

func TestWorkflowCoordinatorDoesNotConsumeANewSameShapePlan(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0", "replica-1")},
		externalMutationHistory: &externalMutations,
	}
	completed := &Operation{
		ID:             "completed-operation",
		Attempt:        1,
		PlanID:         "recovery-plan-1",
		Intent:         OperationIntentRecover,
		Capability:     testOperationCapability(OperationShapeNativeMemberRemapping),
		SpecGeneration: 2,
		BaseTopology: MembershipTopology{
			Generation: 1,
			Replicas: []ReplicaMembership{
				workflowReplicaMembership("replica-0", "native-before-remap-replica-0"),
				workflowReplicaMembership("replica-1", "native-before-remap-replica-1"),
			},
		},
		TargetReplicas:     2,
		TargetMembership:   cloneReplicaMemberships(membership.topology.Replicas),
		Phase:              OperationPhaseCommitted,
		CommittedTopology:  topologyPointer(membership.topology),
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
	}
	operationIDs := &fakeOperationIDGenerator{ids: []string{"new-operation"}}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = operationIDs.Next

	t.Log("Persist completion of the previous operation before replacing its durable record")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 2,
		Plan: &OperationPlan{
			ID:             "recovery-plan-2",
			Intent:         OperationIntentRecover,
			TargetReplicas: 2,
			TargetMembership: []ReplicaMembership{
				workflowReplicaMembership("replica-0", "native-remapped-replica-0"),
				workflowReplicaMembership("replica-1", "native-remapped-replica-1"),
			},
		},
		Operation: completed,
	})
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, "completed-operation", result.Operation.ID)
	assert.True(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, externalMutations)

	t.Log("Treat an identical recovery shape with a new durable plan identity as new work")
	result, err = coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 2,
		Plan: &OperationPlan{
			ID:             "recovery-plan-2",
			Intent:         OperationIntentRecover,
			TargetReplicas: 2,
			TargetMembership: []ReplicaMembership{
				workflowReplicaMembership("replica-0", "native-remapped-replica-0"),
				workflowReplicaMembership("replica-1", "native-remapped-replica-1"),
			},
		},
		Operation: result.Operation,
	})
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, "new-operation", result.Operation.ID)
	assert.Equal(t, "recovery-plan-2", result.Operation.PlanID)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, externalMutations)
}

func workflowPendingGrowthOperation() *Operation {
	return &Operation{
		ID:                 workflowTestOperationID,
		Attempt:            1,
		Intent:             OperationIntentGrow,
		Capability:         testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:     2,
		BaseTopology:       workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:     3,
		Phase:              OperationPhasePending,
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
	}
}

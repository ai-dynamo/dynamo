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
		snapshot:                TrafficSnapshot{Admitted: []ReplicaID{"replica-0", "replica-1"}},
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
		BackendOperationID:     "backend-attempt-1",
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime.Add(-time.Second),
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
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, membership.submitCalls[0].JoiningReplicas)
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
		snapshot:                TrafficSnapshot{Admitted: []ReplicaID{"replica-0", "replica-1"}},
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
		Phase:                  OperationPhaseFailed,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
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
	capabilities := MembershipCapabilities{Intents: []OperationIntent{OperationIntentGrow}}
	membership := &workflowMembershipAdapter{
		capabilities:            &capabilities,
		topology:                workflowTopology(1, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot:                TrafficSnapshot{Admitted: []ReplicaID{"replica-0", "replica-1"}},
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
	require.ErrorContains(t, err, "membership operation \"Recover\" is not supported")
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
		snapshot:                TrafficSnapshot{Admitted: []ReplicaID{"replica-0", "replica-1"}},
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
		Phase:                  OperationPhaseSubmitting,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Reject an unrelated fourth allocation as a substitute for the frozen missing joiner")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []CapacityRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 1,
		TargetReplicas:     4,
		RequiredReplicas:   []ReplicaID{"replica-0", "replica-1", "replica-2", "replica-3"},
	}}, capacity.ensureCalls)
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
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0"},
			Drained:     []ReplicaID{"replica-1"},
		},
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
	replacement.CapacityRefs[0].UID = "replacement-uid"
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			replacement,
		},
		FencedReplicas: []ReplicaID{"replica-1"},
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
				snapshot: TrafficSnapshot{
					OperationID: workflowTestOperationID,
					Admitted:    []ReplicaID{"replica-0"},
					Drained:     []ReplicaID{"replica-1"},
				},
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
				Admitted: []ReplicaID{"replica-0", "replica-1", "replica-9"},
			},
			wantError: "replica-9",
		},
		{
			name:      "drained identity was not nominated",
			operation: workflowPendingShrinkOperation(),
			traffic: TrafficSnapshot{
				OperationID: workflowTestOperationID,
				Admitted:    []ReplicaID{"replica-0"},
				Drained:     []ReplicaID{"replica-1", "replica-2"},
			},
			wantError: "replica-2",
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
		snapshot:                TrafficSnapshot{Admitted: []ReplicaID{"replica-0", "replica-9"}},
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
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0"},
			Drained:     []ReplicaID{"replica-1"},
		},
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

	t.Log("Discard an unissued release before resuming Unknown membership observation on the next reconcile")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:              "group-0",
		SpecGeneration:       2,
		DesiredReplicas:      1,
		Operation:            operation,
		ReleaseAuthorization: authorization,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, []string{workflowTestReleaseID}, capacity.observeReleaseCalls)
	assert.Equal(t, 2, membership.observeTopologyCalls)
	assert.Empty(t, membership.observeOperationCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
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
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0", "replica-1"},
		},
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
		snapshot: TrafficSnapshot{
			OperationID: workflowTestOperationID,
			Admitted:    []ReplicaID{"replica-0"},
			Drained:     []ReplicaID{"replica-1"},
		},
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
	capacity.snapshot.FencedReplicas = []ReplicaID{"replica-1"}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
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
		snapshot:                TrafficSnapshot{Admitted: []ReplicaID{"replica-0", "replica-1"}},
		externalMutationHistory: &externalMutations,
	}
	completed := &Operation{
		ID:                          "completed-operation",
		Attempt:                     1,
		PlanID:                      "recovery-plan-1",
		Intent:                      OperationIntentRecover,
		SpecGeneration:              2,
		BaseTopologyGeneration:      1,
		BaseReplicas:                []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:              2,
		Phase:                       OperationPhaseCommitted,
		CommittedTopologyGeneration: 2,
		StartedAt:                   workflowTestTime.Add(-time.Minute),
		LastTransitionTime:          workflowTestTime,
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
		ID:                     workflowTestOperationID,
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         3,
		Phase:                  OperationPhasePending,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
	}
}

func workflowPendingShrinkOperation() *Operation {
	return &Operation{
		ID:                     workflowTestOperationID,
		Attempt:                1,
		PlanID:                 "plan-1",
		Intent:                 OperationIntentShrink,
		SpecGeneration:         2,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         1,
		NominatedReplicas:      []ReplicaID{"replica-1"},
		Phase:                  OperationPhasePending,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
	}
}

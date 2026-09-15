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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateTrafficCommandUsesActionSpecificIncarnationRules(t *testing.T) {
	oldIncarnation := workflowReplicaIncarnation("replica-0")
	replacement := cloneReplicaIncarnation(oldIncarnation)
	replacement.CapacityRefs[0].UID = operationRestorationReplacementUID
	replacement.RuntimeID = operationRestorationReplacementRun

	t.Log("Allow a withdrawal to retain old and replacement tombstones for the same stable slot")
	require.NoError(t, validateTrafficCommand(TrafficCommand{
		Action: TrafficActionWithdraw,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "operation-1",
			TopologyGeneration: 2,
			Replicas:           []ReplicaIncarnation{oldIncarnation, replacement},
		},
	}))

	t.Log("Reject the same logical slot twice on admission")
	err := validateTrafficCommand(TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "operation-1",
			TopologyGeneration: 2,
			Replicas:           []ReplicaIncarnation{oldIncarnation, replacement},
		},
	})
	require.ErrorContains(t, err, "duplicate")

	t.Log("Reject an exact tombstone duplicated within one withdrawal")
	err = validateTrafficCommand(TrafficCommand{
		Action: TrafficActionWithdraw,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "operation-1",
			TopologyGeneration: 2,
			Replicas:           []ReplicaIncarnation{oldIncarnation, cloneReplicaIncarnation(oldIncarnation)},
		},
	})
	require.ErrorContains(t, err, "duplicate")

	t.Log("Reject one runtime identity reused by two physical incarnations")
	replacement.RuntimeID = oldIncarnation.RuntimeID
	err = validateTrafficCommand(TrafficCommand{
		Action: TrafficActionWithdraw,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "operation-1",
			TopologyGeneration: 2,
			Replicas:           []ReplicaIncarnation{oldIncarnation, replacement},
		},
	})
	require.ErrorContains(t, err, "identifies multiple exact incarnations")
}

func TestValidateTrafficCommandObservationFailureSemantics(t *testing.T) {
	command := TrafficCommand{
		Action: TrafficActionWithdraw,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "operation-1",
			TopologyGeneration: 1,
			Replicas:           workflowReplicaIncarnations("replica-0"),
		},
	}
	retryableFailure := &OperationFailure{
		Classification: FailureClassificationRetryable,
		Reason:         "RuntimeUnavailable",
	}
	terminalFailure := &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "PolicyRejected",
	}

	t.Log("A failed accepted command must report a classified failure")
	require.ErrorContains(t, validateTrafficCommandObservation(TrafficCommandObservation{
		Command: command,
		Phase:   TrafficCommandPhaseFailed,
	}), "validate failed traffic command")

	t.Log("Failed commands may be retryable or terminal because their effects have stopped")
	require.NoError(t, validateTrafficCommandObservation(TrafficCommandObservation{
		Command: command,
		Phase:   TrafficCommandPhaseFailed,
		Failure: retryableFailure,
	}))
	require.NoError(t, validateTrafficCommandObservation(TrafficCommandObservation{
		Command: command,
		Phase:   TrafficCommandPhaseFailed,
		Failure: terminalFailure,
	}))

	t.Log("A command refused before mutation must be terminal")
	require.ErrorContains(t, validateTrafficCommandObservation(TrafficCommandObservation{
		Command: command,
		Phase:   TrafficCommandPhaseRefused,
		Failure: retryableFailure,
	}), "requires a terminal failure")
}

func TestWorkflowTrafficAdapterEnforcesMonotonicCommandOrdering(t *testing.T) {
	replica := workflowReplicaIncarnation("replica-0")
	newerWithdraw := TrafficCommand{
		Action: TrafficActionWithdraw,
		Request: TrafficRequest{
			Revision:           2,
			OperationID:        "withdraw-2",
			TopologyGeneration: 2,
			Replicas:           []ReplicaIncarnation{replica},
		},
	}
	adapter := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{LatestCommand: &TrafficCommandObservation{
			Command: newerWithdraw,
			Phase:   TrafficCommandPhaseAccepted,
		}},
	}

	t.Log("Refuse a delayed older admission after a newer withdrawal has been accepted")
	err := adapter.Admit(context.Background(), "group-0", TrafficRequest{
		Revision:           1,
		OperationID:        "admit-1",
		TopologyGeneration: 2,
		Replicas:           []ReplicaIncarnation{replica},
	})
	require.ErrorContains(t, err, "older than accepted revision 2")
	assert.Empty(t, adapter.admitRequests)
	assert.Equal(t, newerWithdraw, adapter.snapshot.LatestCommand.Command)

	t.Log("Refuse a different payload at the same revision")
	err = adapter.Admit(context.Background(), "group-0", TrafficRequest{
		Revision:           2,
		OperationID:        "admit-2",
		TopologyGeneration: 2,
		Replicas:           []ReplicaIncarnation{replica},
	})
	require.ErrorContains(t, err, "conflicts with the accepted command")

	t.Log("Replay the exact latest command idempotently")
	require.NoError(t, adapter.Withdraw(context.Background(), "group-0", newerWithdraw.Request))
	require.NoError(t, adapter.Withdraw(context.Background(), "group-0", newerWithdraw.Request))
	assert.Len(t, adapter.withdrawRequests, 2)
	assert.Equal(t, adapter.withdrawRequests[0], adapter.withdrawRequests[1])

	t.Log("Never replay an equal revision that was durably refused")
	adapter.snapshot.LatestCommand = &TrafficCommandObservation{
		Command: newerWithdraw,
		Phase:   TrafficCommandPhaseRefused,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "PolicyRejected",
		},
	}
	err = adapter.Withdraw(context.Background(), "group-0", newerWithdraw.Request)
	require.ErrorContains(t, err, "definitively refused")
}

func TestWorkflowCoordinatorFailsClosedOnTrafficRevisionDivergence(t *testing.T) {
	replica := workflowReplicaIncarnation("replica-0")
	durableCommand := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "durable-admit",
			TopologyGeneration: 1,
			Replicas:           []ReplicaIncarnation{replica},
		},
	}
	tests := []struct {
		name       string
		observed   TrafficCommand
		wantErr    string
		durableRev int64
	}{
		{
			name: "adapter is ahead",
			observed: TrafficCommand{
				Action: TrafficActionWithdraw,
				Request: TrafficRequest{
					Revision:           2,
					OperationID:        "unknown-withdraw",
					TopologyGeneration: 1,
					Replicas:           []ReplicaIncarnation{replica},
				},
			},
			wantErr:    "ahead of durable revision 1",
			durableRev: 1,
		},
		{
			name: "equal revision has conflicting payload",
			observed: TrafficCommand{
				Action: TrafficActionWithdraw,
				Request: TrafficRequest{
					Revision:           1,
					OperationID:        "conflicting-withdraw",
					TopologyGeneration: 1,
					Replicas:           []ReplicaIncarnation{replica},
				},
			},
			wantErr:    "does not match the durable command",
			durableRev: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
				snapshot: TrafficSnapshot{
					LatestCommand: &TrafficCommandObservation{
						Command: tt.observed,
						Phase:   TrafficCommandPhaseAccepted,
					},
					Admitted: []ReplicaIncarnation{replica},
				},
				externalMutationHistory: &externalMutations,
			}
			coordinator := NewWorkflowCoordinator(capacity, membership, traffic, &workflowServingVerifier{})

			_, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 1,
				TrafficRevision: tt.durableRev,
				TrafficCommand:  cloneTrafficCommand(&durableCommand),
			})
			require.ErrorContains(t, err, tt.wantErr)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestScheduleTrafficCommandChangesPayloadAtTheNextRevision(t *testing.T) {
	replica := workflowReplicaIncarnation("replica-0")
	oldCommand := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           7,
			OperationID:        "admit-7",
			TopologyGeneration: 1,
			Replicas:           []ReplicaIncarnation{replica},
		},
	}
	result := ReconcileResult{
		TrafficRevision: 7,
		TrafficCommand:  cloneTrafficCommand(&oldCommand),
		Traffic: TrafficSnapshot{LatestCommand: &TrafficCommandObservation{
			Command: oldCommand,
			Phase:   TrafficCommandPhaseAccepted,
		}},
	}

	t.Log("Do not reschedule an identical payload after its revision was definitively refused")
	result.Traffic.LatestCommand.Phase = TrafficCommandPhaseRefused
	result.Traffic.LatestCommand.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "PolicyRejected",
		Message:        "the runtime rejected admission",
	}
	_, err := scheduleTrafficCommand(result, TrafficActionAdmit, TrafficRequest{
		OperationID:        oldCommand.Request.OperationID,
		TopologyGeneration: oldCommand.Request.TopologyGeneration,
		Replicas:           oldCommand.Request.Replicas,
	})
	require.ErrorContains(t, err, "definitively refused")

	t.Log("A changed safe payload supersedes the refusal at the next global revision")
	updated, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
		OperationID:        "withdraw-8",
		TopologyGeneration: 1,
		Replicas:           []ReplicaIncarnation{replica},
	})
	require.NoError(t, err)
	assert.True(t, updated.TrafficStateChanged)
	assert.Equal(t, int64(8), updated.TrafficRevision)
	require.NotNil(t, updated.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, updated.TrafficCommand.Action)
	assert.Equal(t, int64(8), updated.TrafficCommand.Request.Revision)
}

func TestWorkflowCoordinatorHandlesStoppedTrafficCommand(t *testing.T) {
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3")
	joiningReplicas := workflowReplicaIncarnations("replica-2", "replica-3")
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Capability.TrafficRequirement = ReconfigurationTrafficKeepServing
	operation.Capability.VerificationRequirement = ServingVerificationNotRequired
	operation.TargetReplicas = 4
	operation.JoiningReplicas = joiningReplicas
	operation.CommittedTopology = topologyPointer(committedTopology)
	operation.ServingVerificationAttempt = 0
	operation.ServingVerificationTarget = nil
	operation.ServingVerificationProof = nil
	durableCommand := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        workflowTestOperationID,
			TopologyGeneration: committedTopology.Generation,
			Replicas:           joiningReplicas,
		},
	}

	tests := []struct {
		name           string
		classification FailureClassification
		wantAction     TrafficAction
	}{
		{
			name:           "retryable failure mints a new revision",
			classification: FailureClassificationRetryable,
			wantAction:     TrafficActionAdmit,
		},
		{
			name:           "terminal failed admission is covered by a withdrawal",
			classification: FailureClassificationTerminal,
			wantAction:     TrafficActionWithdraw,
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
				topology:                committedTopology,
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: TrafficSnapshot{
					LatestCommand: &TrafficCommandObservation{
						Command: durableCommand,
						Phase:   TrafficCommandPhaseFailed,
						Failure: &OperationFailure{
							Classification: tt.classification,
							Reason:         "TrafficMutationStopped",
							Message:        "the runtime stopped the requested effect",
						},
					},
					Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
				},
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 4,
				Operation:       cloneOperation(operation),
				TrafficRevision: 1,
				TrafficCommand:  cloneTrafficCommand(&durableCommand),
			})
			require.NoError(t, err)
			assert.True(t, result.TrafficStateChanged)
			assert.Equal(t, int64(2), result.TrafficRevision)
			require.NotNil(t, result.TrafficCommand)
			assert.Equal(t, tt.wantAction, result.TrafficCommand.Action)
			if tt.classification == FailureClassificationTerminal {
				assert.Equal(
					t,
					unverifiedTopologyTrafficOperationID(*result.Operation, committedTopology),
					result.TrafficCommand.Request.OperationID,
				)
				assert.Equal(
					t,
					unverifiedTopologyDrainReplicas(*result.Operation, committedTopology),
					result.TrafficCommand.Request.Replicas,
				)
				assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
				require.NotNil(t, result.Operation.TerminalAdmissionFailure)
			} else {
				assert.Equal(t, durableCommand.Request.OperationID, result.TrafficCommand.Request.OperationID)
				assert.Equal(t, durableCommand.Request.Replicas, result.TrafficCommand.Request.Replicas)
			}
			assert.Equal(t, durableCommand.Request.TopologyGeneration, result.TrafficCommand.Request.TopologyGeneration)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorRetainsIncompleteAcceptedAbsoluteCommand(t *testing.T) {
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3")
	joiningReplicas := workflowReplicaIncarnations("replica-2", "replica-3")
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Capability.TrafficRequirement = ReconfigurationTrafficKeepServing
	operation.Capability.VerificationRequirement = ServingVerificationNotRequired
	operation.TargetReplicas = 4
	operation.JoiningReplicas = joiningReplicas
	operation.CommittedTopology = topologyPointer(committedTopology)
	operation.ServingVerificationAttempt = 0
	operation.ServingVerificationTarget = nil
	operation.ServingVerificationProof = nil
	durableCommand := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        workflowTestOperationID,
			TopologyGeneration: committedTopology.Generation,
			Replicas:           joiningReplicas,
		},
	}
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
		topology:                committedTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			LatestCommand: &TrafficCommandObservation{
				Command: durableCommand,
				Phase:   TrafficCommandPhaseAccepted,
			},
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
		TrafficRevision: 1,
		TrafficCommand:  cloneTrafficCommand(&durableCommand),
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Do not replace accepted Admit[A,B] with a newly computed Admit[B]")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.TrafficStateChanged)
	assert.Equal(t, int64(1), result.TrafficRevision)
	assert.Equal(t, &durableCommand, result.TrafficCommand)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, externalMutations)

	t.Log("Cover a terminally stopped partial admission with an absolute withdrawal")
	traffic.snapshot.LatestCommand.Phase = TrafficCommandPhaseFailed
	traffic.snapshot.LatestCommand.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "AdmissionStopped",
		Message:        "the runtime could not admit every requested replica",
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.TrafficStateChanged)
	assert.Equal(t, int64(2), result.TrafficRevision)
	require.NotNil(t, result.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
	assert.Equal(t, unverifiedTopologyDrainReplicas(*result.Operation, committedTopology), result.TrafficCommand.Request.Replicas)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	require.NotNil(t, result.Operation.TerminalAdmissionFailure)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorSupersedesUndispatchedAdmissionAfterAuthorityLoss(t *testing.T) {
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	joiningReplica := workflowReplicaIncarnation("replica-2")
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Capability.TrafficRequirement = ReconfigurationTrafficKeepServing
	operation.Capability.VerificationRequirement = ServingVerificationNotRequired
	operation.ServingVerificationAttempt = 0
	operation.ServingVerificationTarget = nil
	operation.ServingVerificationProof = nil
	durableAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        operation.ID,
			TopologyGeneration: committedTopology.Generation,
			Replicas:           []ReplicaIncarnation{joiningReplica},
		},
	}

	tests := []struct {
		name       string
		capacity   CapacitySnapshot
		topology   MembershipTopology
		wantReason string
	}{
		{
			name: "exact capacity became unavailable",
			capacity: func() CapacitySnapshot {
				joiner := workflowReplicaAllocation("replica-2", "pod-2")
				joiner.Availability = ReplicaAvailabilityUnavailable
				return CapacitySnapshot{Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-1", "pod-1"),
					joiner,
				}}
			}(),
			topology:   committedTopology,
			wantReason: "unavailable capacity",
		},
		{
			name: "authoritative topology changed",
			capacity: CapacitySnapshot{Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
				workflowReplicaAllocation("replica-2", "pod-2"),
			}},
			topology:   workflowTopology(3, "replica-0", "replica-1", "replica-2"),
			wantReason: "topology drift",
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
				topology:                tt.topology,
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: TrafficSnapshot{
					Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
				},
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  operation.SpecGeneration,
				DesiredReplicas: operation.TargetReplicas,
				Operation:       cloneOperation(operation),
				TrafficRevision: durableAdmit.Request.Revision,
				TrafficCommand:  cloneTrafficCommand(&durableAdmit),
			})
			require.NoError(t, err)
			assert.True(t, result.TrafficStateChanged, tt.wantReason)
			assert.Equal(t, int64(2), result.TrafficRevision)
			require.NotNil(t, result.TrafficCommand)
			assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
			assert.Equal(
				t,
				unverifiedTopologyTrafficOperationID(*operation, tt.topology),
				result.TrafficCommand.Request.OperationID,
			)
			assert.Equal(t, tt.topology.Generation, result.TrafficCommand.Request.TopologyGeneration)
			assert.Equal(
				t,
				unverifiedTopologyDrainReplicas(*operation, tt.topology),
				result.TrafficCommand.Request.Replicas,
			)
			assert.Contains(t, result.TrafficCommand.Request.Replicas, operation.BaseTopology.Replicas[0].Incarnation)
			assert.Contains(t, result.TrafficCommand.Request.Replicas, operation.BaseTopology.Replicas[1].Incarnation)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, traffic.withdrawRequests)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorFencesOlderTerminalAdmissionBeforeDurableReplay(t *testing.T) {
	replicas := workflowReplicaIncarnations("replica-2", "replica-3")
	failedAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "failed-admit",
			TopologyGeneration: 2,
			Replicas:           replicas,
		},
	}
	nonCoveringDurableAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           2,
			OperationID:        "later-admit",
			TopologyGeneration: 2,
			Replicas:           replicas[1:],
		},
	}
	externalMutations := make([]string, 0)
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			LatestCommand: &TrafficCommandObservation{
				Command: failedAdmit,
				Phase:   TrafficCommandPhaseFailed,
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "AdmissionStopped",
				},
			},
			Admitted: replicas[:1],
		},
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorForTest(
		&workflowCapacityAdapter{
			snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
				workflowReplicaAllocation("replica-2", "pod-2"),
				workflowReplicaAllocation("replica-3", "pod-3"),
			}},
			externalMutationHistory: &externalMutations,
		},
		&workflowMembershipAdapter{
			topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3"),
			externalMutationHistory: &externalMutations,
		},
		traffic,
	)

	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		TrafficRevision: nonCoveringDurableAdmit.Request.Revision,
		TrafficCommand:  cloneTrafficCommand(&nonCoveringDurableAdmit),
	})
	require.NoError(t, err)
	assert.True(t, result.TrafficStateChanged)
	assert.Equal(t, int64(3), result.TrafficRevision)
	require.NotNil(t, result.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
	assert.Equal(t, failedAdmit.Request.OperationID, result.TrafficCommand.Request.OperationID)
	assert.Equal(t, failedAdmit.Request.Replicas, result.TrafficCommand.Request.Replicas)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, traffic.withdrawRequests)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorReplaysCoveringWithdrawalBeforeMembershipReconciliation(t *testing.T) {
	replicas := workflowReplicaIncarnations("replica-0", "replica-1")
	failedAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        "failed-admit",
			TopologyGeneration: 1,
			Replicas:           replicas,
		},
	}
	coveringWithdraw := TrafficCommand{
		Action: TrafficActionWithdraw,
		Request: TrafficRequest{
			Revision:           2,
			OperationID:        "safety-withdraw",
			TopologyGeneration: 1,
			Replicas:           replicas,
		},
	}
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
			LatestCommand: &TrafficCommandObservation{
				Command: failedAdmit,
				Phase:   TrafficCommandPhaseFailed,
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "AdmissionStopped",
				},
			},
			Admitted: replicas[:1],
		},
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 2,
		TrafficRevision: coveringWithdraw.Request.Revision,
		TrafficCommand:  cloneTrafficCommand(&coveringWithdraw),
	})
	require.NoError(t, err)
	assert.False(t, result.TrafficStateChanged)
	assert.Equal(t, &coveringWithdraw, result.TrafficCommand)
	assert.Equal(t, []TrafficRequest{coveringWithdraw.Request}, traffic.withdrawRequests)
	assert.Zero(t, membership.observeTopologyCalls, "durable withdrawal must replay before membership reconciliation")
	assert.Equal(t, []string{"traffic.withdraw:" + coveringWithdraw.Request.OperationID}, externalMutations)
}

func TestTerminalAdmissionFailureRemainsFencedUntilDistinctRecovery(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Capability.TrafficRequirement = ReconfigurationTrafficQuiesceGroup
	operation.Capability.VerificationRequirement = ServingVerificationNotRequired
	operation.ServingVerificationAttempt = 0
	operation.ServingVerificationTarget = nil
	operation.ServingVerificationProof = nil
	allReplicas := topologyReplicaIncarnations(committedTopology)
	failedAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        operation.ID,
			TopologyGeneration: committedTopology.Generation,
			Replicas:           allReplicas,
		},
	}
	terminalFailure := &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "AdmissionStopped",
		Message:        "the runtime stopped before admitting the complete topology",
	}
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                committedTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			LatestCommand: &TrafficCommandObservation{
				Command: failedAdmit,
				Phase:   TrafficCommandPhaseFailed,
				Failure: cloneFailure(terminalFailure),
			},
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  operation.SpecGeneration,
		DesiredReplicas: operation.TargetReplicas,
		Operation:       operation,
		TrafficRevision: failedAdmit.Request.Revision,
		TrafficCommand:  cloneTrafficCommand(&failedAdmit),
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist both the terminal operation outcome and its higher-revision covering fence")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.Equal(t, terminalFailure, result.Operation.Failure)
	require.NotNil(t, result.Operation.TerminalAdmissionFailure)
	assert.Equal(t, failedAdmit, result.Operation.TerminalAdmissionFailure.Command)
	assert.Equal(t, *terminalFailure, result.Operation.TerminalAdmissionFailure.Failure)
	assert.True(t, result.OperationChanged)
	assert.True(t, result.TrafficStateChanged)
	assert.Equal(t, int64(2), result.TrafficRevision)
	require.NotNil(t, result.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
	assert.Equal(t, allReplicas, result.TrafficCommand.Request.Replicas)
	require.NoError(t, validateOperation(*result.Operation))
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, traffic.withdrawRequests)
	input.Operation = result.Operation
	persistWorkflowTrafficState(&input, result)

	t.Log("A restart dispatches only the durable covering withdrawal")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.False(t, result.TrafficStateChanged)
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []TrafficRequest{result.TrafficCommand.Request}, traffic.withdrawRequests)

	t.Log("After the fence completes, repeated reconciliation cannot retry admission")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw,
		result.TrafficRevision,
		result.TrafficCommand.Request.OperationID,
		committedTopology.Generation,
		nil,
		allReplicas,
	)
	for range 2 {
		coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
		result, err = coordinator.Reconcile(context.Background(), input)
		require.NoError(t, err)
		assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
		assert.False(t, result.OperationChanged)
		assert.False(t, result.TrafficStateChanged)
		assert.Empty(t, traffic.admitRequests)
	}

	t.Log("A distinct explicit recovery plan may replace the failed audit record")
	targetMembership := cloneReplicaMemberships(committedTopology.Replicas)
	targetMembership[2].NativeMembers = []NativeMemberID{"native-replica-2-remapped"}
	recoveryCapability := ResolvedOperationCapability{
		Shape:                   OperationShapeNativeMemberRemapping,
		TrafficRequirement:      ReconfigurationTrafficKeepServing,
		VerificationRequirement: ServingVerificationRequired,
	}
	membership.validatePlanCapability = &recoveryCapability
	input.SpecGeneration++
	input.Plan = &OperationPlan{
		ID:               "recovery-plan",
		Intent:           OperationIntentRecover,
		TargetReplicas:   operation.TargetReplicas,
		TargetMembership: targetMembership,
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = func() string { return workflowTestRecoveryOperationID }
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, workflowTestRecoveryOperationID, result.Operation.ID)
	assert.Equal(t, input.Plan.ID, result.Operation.PlanID)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Nil(t, result.Operation.TerminalAdmissionFailure)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
}

func TestRefusedAdmissionBecomesDurableFailureBeforeRecovery(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Capability.TrafficRequirement = ReconfigurationTrafficQuiesceGroup
	operation.Capability.VerificationRequirement = ServingVerificationNotRequired
	operation.ServingVerificationAttempt = 0
	operation.ServingVerificationTarget = nil
	operation.ServingVerificationProof = nil
	allReplicas := topologyReplicaIncarnations(committedTopology)
	refusedAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        operation.ID,
			TopologyGeneration: committedTopology.Generation,
			Replicas:           allReplicas,
		},
	}
	refusal := &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "AdmissionRejected",
		Message:        "the runtime refused the complete admission before mutation",
	}
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                committedTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			LatestCommand: &TrafficCommandObservation{
				Command: refusedAdmit,
				Phase:   TrafficCommandPhaseRefused,
				Failure: cloneFailure(refusal),
			},
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  operation.SpecGeneration,
		DesiredReplicas: operation.TargetReplicas,
		Operation:       operation,
		TrafficRevision: refusedAdmit.Request.Revision,
		TrafficCommand:  cloneTrafficCommand(&refusedAdmit),
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist the refusal as the current operation's terminal outcome before any new side effect")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.Equal(t, refusal, result.Operation.Failure)
	require.NotNil(t, result.Operation.TerminalAdmissionFailure)
	assert.Equal(t, refusedAdmit, result.Operation.TerminalAdmissionFailure.Command)
	assert.True(t, result.OperationChanged)
	assert.False(t, result.TrafficStateChanged)
	assert.Equal(t, &refusedAdmit, result.TrafficCommand)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, traffic.withdrawRequests)
	input.Operation = result.Operation

	t.Log("A restart establishes the failed topology's full recovery fence")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.True(t, result.TrafficStateChanged)
	require.NotNil(t, result.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
	assert.Equal(t, allReplicas, result.TrafficCommand.Request.Replicas)
	assert.Empty(t, traffic.admitRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)

	t.Log("Only a distinct Recover plan may take ownership after the fence completes")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw,
		result.TrafficRevision,
		result.TrafficCommand.Request.OperationID,
		committedTopology.Generation,
		nil,
		allReplicas,
	)
	targetMembership := cloneReplicaMemberships(committedTopology.Replicas)
	targetMembership[2].NativeMembers = []NativeMemberID{"native-replica-2-remapped"}
	recoveryCapability := ResolvedOperationCapability{
		Shape:                   OperationShapeNativeMemberRemapping,
		TrafficRequirement:      ReconfigurationTrafficKeepServing,
		VerificationRequirement: ServingVerificationRequired,
	}
	membership.validatePlanCapability = &recoveryCapability
	input.SpecGeneration++
	input.Plan = &OperationPlan{
		ID:               "recovery-after-refusal",
		Intent:           OperationIntentRecover,
		TargetReplicas:   operation.TargetReplicas,
		TargetMembership: targetMembership,
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = func() string { return workflowTestRecoveryOperationID }
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, workflowTestRecoveryOperationID, result.Operation.ID)
	assert.Equal(t, input.Plan.ID, result.Operation.PlanID)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Nil(t, result.Operation.TerminalAdmissionFailure)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
}

func TestOnlyCoveringWithdrawalSupersedesIncompleteAcceptedCommand(t *testing.T) {
	replicas := workflowReplicaIncarnations("replica-0", "replica-1")
	acceptedAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           4,
			OperationID:        "admit-4",
			TopologyGeneration: 1,
			Replicas:           replicas,
		},
	}
	result := ReconcileResult{
		TrafficRevision: acceptedAdmit.Request.Revision,
		TrafficCommand:  cloneTrafficCommand(&acceptedAdmit),
		Traffic: TrafficSnapshot{
			LatestCommand: &TrafficCommandObservation{
				Command: acceptedAdmit,
				Phase:   TrafficCommandPhaseAccepted,
			},
			Admitted: replicas[:1],
		},
	}

	t.Log("A partial withdrawal cannot supersede an incomplete absolute admission")
	unchanged, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
		OperationID:        "withdraw-partial",
		TopologyGeneration: 1,
		Replicas:           replicas[1:],
	})
	require.NoError(t, err)
	assert.False(t, unchanged.TrafficStateChanged)
	assert.Equal(t, int64(4), unchanged.TrafficRevision)
	assert.Equal(t, &acceptedAdmit, unchanged.TrafficCommand)

	t.Log("A withdrawal covering every possibly admitted incarnation is a safe superseding fence")
	updated, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
		OperationID:        "withdraw-all",
		TopologyGeneration: 1,
		Replicas:           replicas,
	})
	require.NoError(t, err)
	assert.True(t, updated.TrafficStateChanged)
	assert.Equal(t, int64(5), updated.TrafficRevision)
	require.NotNil(t, updated.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, updated.TrafficCommand.Action)
	assert.Equal(t, replicas, updated.TrafficCommand.Request.Replicas)
}

func TestTrafficReadinessRejectsAnAcceptedOppositeCommand(t *testing.T) {
	replica := workflowReplicaIncarnation("replica-0")
	topology := MembershipTopology{
		Generation: 1,
		Replicas: []ReplicaMembership{{
			Incarnation:   replica,
			NativeMembers: []NativeMemberID{"native-0"},
		}},
	}
	operation := Operation{
		BaseTopology:      topology,
		JoiningReplicas:   []ReplicaIncarnation{replica},
		NominatedReplicas: []ReplicaID{"replica-0"},
	}

	t.Log("An old drained effect is not drain-ready while a newer admission can still take effect")
	traffic := TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionAdmit, 2, "admit-2", 1, []ReplicaIncarnation{replica},
		),
		Drained: []ReplicaIncarnation{replica},
	}
	assert.False(t, retirementTrafficReady(operation, topology, traffic))

	t.Log("An old admitted effect is not admission-ready while a newer withdrawal can still take effect")
	traffic = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw, 3, "withdraw-3", 1, []ReplicaIncarnation{replica},
		),
		Admitted: []ReplicaIncarnation{replica},
	}
	assert.False(t, trafficAdmissionComplete(operation, traffic))

	t.Log("A definitively refused opposite command cannot invalidate the observed effect")
	traffic.LatestCommand.Phase = TrafficCommandPhaseRefused
	traffic.LatestCommand.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "PolicyRejected",
	}
	assert.True(t, trafficAdmissionComplete(operation, traffic))
}

func TestTrafficAdmissionRequiresTheExactReplicaIncarnation(t *testing.T) {
	original := workflowReplicaIncarnation("replica-0")
	replacement := cloneReplicaIncarnation(original)
	replacement.CapacityRefs[0].UID = operationRestorationReplacementUID
	replacement.RuntimeID = operationRestorationReplacementRun

	t.Log("Do not let a replacement inherit admission from the prior Pod and process")
	assert.False(t, trafficAdmissionComplete(Operation{
		JoiningReplicas: []ReplicaIncarnation{replacement},
	}, TrafficSnapshot{
		Admitted: []ReplicaIncarnation{original},
	}))

	t.Log("Admit the replacement after its own exact incarnation becomes routable")
	assert.True(t, trafficAdmissionComplete(Operation{
		JoiningReplicas: []ReplicaIncarnation{replacement},
	}, TrafficSnapshot{
		Admitted: []ReplicaIncarnation{replacement},
		Drained:  []ReplicaIncarnation{original},
	}))
}

func TestEmptyTopologyCompensationWaitsForSameGenerationAdmission(t *testing.T) {
	replica := workflowReplicaIncarnation("replica-0")
	emptyTopology := MembershipTopology{Generation: 3}
	acceptedAdmit := workflowAcceptedTrafficObservation(
		TrafficActionAdmit, 4, "admit-4", emptyTopology.Generation, []ReplicaIncarnation{replica},
	)

	t.Log("Do not finish while an accepted same-generation admission can still take effect")
	assert.False(t, trafficCompensationComplete(emptyTopology, TrafficSnapshot{
		LatestCommand: acceptedAdmit,
	}))

	t.Log("A stale-generation admission cannot mutate the authoritative empty topology")
	staleAdmit := cloneTrafficCommandObservation(acceptedAdmit)
	staleAdmit.Command.Request.TopologyGeneration--
	assert.True(t, trafficCompensationComplete(emptyTopology, TrafficSnapshot{
		LatestCommand: staleAdmit,
	}))

	t.Log("A refused admission is terminally unable to mutate traffic")
	refusedAdmit := cloneTrafficCommandObservation(acceptedAdmit)
	refusedAdmit.Phase = TrafficCommandPhaseRefused
	refusedAdmit.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "PolicyRejected",
	}
	assert.True(t, trafficCompensationComplete(emptyTopology, TrafficSnapshot{
		LatestCommand: refusedAdmit,
	}))

	t.Log("A newer accepted withdrawal supersedes admission risk")
	assert.True(t, trafficCompensationComplete(emptyTopology, TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw, 5, "withdraw-5", emptyTopology.Generation, []ReplicaIncarnation{replica},
		),
	}))
}

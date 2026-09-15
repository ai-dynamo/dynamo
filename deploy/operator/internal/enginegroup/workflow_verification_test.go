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

func TestWorkflowCoordinatorRequiresServingVerificationBeforeAdmission(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, committedTopology.Generation,
			nil, workflowReplicaIncarnations("replica-0", "replica-1"),
		),
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       workflowCommittedVerifiedGrowthOperation(),
	}
	request := ServingVerificationRequest{
		OperationID:         workflowTestOperationID,
		Attempt:             1,
		VerificationAttempt: 1,
		Topology:            committedTopology,
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Start verification only after the submitted operation's whole-group drain is durable")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, verifier.ensureRequests, 1)
	assert.True(t, servingVerificationRequestsEqual(request, verifier.ensureRequests[0]))
	assert.Empty(t, traffic.withdrawRequests)
	assert.Empty(t, traffic.admitRequests)
	assert.Nil(t, result.Operation.ServingVerificationProof)

	t.Log("Wait while the exact verification remains restart-observably running")
	verifier.proof = servingVerificationTestProof(request, ServingVerificationPhaseRunning, nil)
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Len(t, verifier.ensureRequests, 1)
	assert.Empty(t, traffic.admitRequests)
	assert.False(t, result.OperationChanged)

	t.Log("Persist the passed proof as a durable boundary before admitting traffic")
	verifier.proof = servingVerificationTestProof(request, ServingVerificationPhasePassed, nil)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation.ServingVerificationProof)
	assert.True(t, servingVerificationPassedForRequest(*result.Operation.ServingVerificationProof, request))
	assert.True(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
	input.Operation = result.Operation

	t.Log("Restart from the durable proof and readmit the exact verified topology")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           2,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		Replicas:           topologyReplicaIncarnations(committedTopology),
	}}, traffic.admitRequests)
}

func TestWorkflowCoordinatorFencesPrematurelyAdmittedJoinerBeforeVerification(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, committedTopology.Generation,
			workflowReplicaIncarnations("replica-2"),
			workflowReplicaIncarnations("replica-0", "replica-1"),
		),
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Durably schedule a whole-topology fence when a joiner bypasses the quiescence contract")
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       workflowCommittedVerifiedGrowthOperation(),
	}
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.withdrawRequests)
	assert.Empty(t, verifier.observeCalls)
	assert.Empty(t, verifier.ensureRequests)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, externalMutations)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           2,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: committedTopology.Generation,
		Replicas:           topologyReplicaIncarnations(committedTopology),
	}}, traffic.withdrawRequests)
}

func TestWorkflowCoordinatorFailsClosedOnPostCommitTrafficRegression(t *testing.T) {
	tests := []struct {
		name      string
		operation func() *Operation
		traffic   TrafficSnapshot
	}{
		{
			name: "QuiesceGroup drain regressed",
			operation: func() *Operation {
				operation := workflowCommittedVerifiedGrowthOperation()
				operation.Capability.TrafficRequirement = ReconfigurationTrafficQuiesceGroup
				return operation
			},
			traffic: workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw, 1, workflowTestOperationID, 2,
				workflowReplicaIncarnations("replica-0"), workflowReplicaIncarnations("replica-1"),
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
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
				snapshot:                tt.traffic,
				externalMutationHistory: &externalMutations,
			}
			verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
			coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

			t.Log("Schedule a durable whole-topology fence when post-commit traffic safety regresses")
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 3,
				Operation:       tt.operation(),
			}
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.False(t, result.OperationChanged)
			assert.Empty(t, verifier.observeCalls)
			assert.Empty(t, verifier.ensureRequests)
			assert.Empty(t, traffic.withdrawRequests)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, capacity.ensureCalls)
			assert.Empty(t, externalMutations)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			assert.Equal(t, []TrafficRequest{{
				Revision:           2,
				OperationID:        workflowTestOperationID,
				TopologyGeneration: committedTopology.Generation,
				Replicas:           topologyReplicaIncarnations(committedTopology),
			}}, traffic.withdrawRequests)
		})
	}
}

func TestWorkflowCoordinatorFailsClosedOnStaleServingVerificationProof(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	staleTopology := cloneTopology(committedTopology)
	staleTopology.Replicas[2].NativeMembers[0] = "native-replica-2-before-remap"
	request := ServingVerificationRequest{
		OperationID:         workflowTestOperationID,
		Attempt:             1,
		VerificationAttempt: 1,
		Topology:            staleTopology,
	}
	staleProof := servingVerificationTestProof(request, ServingVerificationPhasePassed, nil)
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.ServingVerificationProof = cloneServingVerificationProof(&staleProof)
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
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
		},
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{
		proof:                   staleProof,
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Refuse a passed proof whose native-member mapping no longer matches committed topology")
	_, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	})
	require.ErrorContains(t, err, "durable serving verification proof does not match the exact verification request")
	assert.Empty(t, verifier.ensureRequests)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorFencesTerminalServingVerificationOutcomesBeforeFailing(t *testing.T) {
	tests := []struct {
		name            string
		phase           ServingVerificationPhase
		observedFailure *OperationFailure
		storedFailure   *OperationFailure
	}{
		{
			name:  "explicit terminal failure",
			phase: ServingVerificationPhaseFailed,
			observedFailure: &OperationFailure{
				Classification: FailureClassificationTerminal,
				Reason:         "CollectiveStalled",
				Message:        "collective progress barrier timed out",
			},
			storedFailure: &OperationFailure{
				Classification: FailureClassificationTerminal,
				Reason:         "CollectiveStalled",
				Message:        "collective progress barrier timed out",
			},
		},
		{
			name:  "unrecoverable unknown outcome",
			phase: ServingVerificationPhaseUnknown,
			storedFailure: &OperationFailure{
				Classification: FailureClassificationTerminal,
				Reason:         "ServingVerificationUnknown",
				Message:        "the verifier cannot recover the exact serving-verification outcome",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
			request := ServingVerificationRequest{
				OperationID:         workflowTestOperationID,
				Attempt:             1,
				VerificationAttempt: 1,
				Topology:            committedTopology,
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
					Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
				},
				externalMutationHistory: &externalMutations,
			}
			verifier := &workflowServingVerifier{
				proof:                   servingVerificationTestProof(request, tt.phase, tt.observedFailure),
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 3,
				Operation:       workflowCommittedVerifiedGrowthOperation(),
			}
			t.Log("Persist a command to fence the complete topology without admitting committed joining capacity")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
			assert.Equal(t, int32(1), result.Operation.ServingVerificationAttempt)
			assert.False(t, result.OperationChanged)
			assert.Nil(t, result.Operation.Failure)
			assert.Empty(t, verifier.ensureRequests)
			assert.Empty(t, traffic.admitRequests)
			failureTrafficOperationID := workflowTestOperationID
			assert.Empty(t, traffic.withdrawRequests)
			require.True(t, result.TrafficStateChanged)
			assert.Equal(t, &TrafficCommand{
				Action: TrafficActionWithdraw,
				Request: TrafficRequest{
					Revision:           1,
					OperationID:        failureTrafficOperationID,
					TopologyGeneration: committedTopology.Generation,
					Replicas:           topologyReplicaIncarnations(committedTopology),
				},
			}, result.TrafficCommand)
			assert.Empty(t, externalMutations)

			input.Operation = result.Operation
			persistWorkflowTrafficState(&input, result)
			t.Log("Restart before dispatch and replay the exact durable failure fence")
			coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
			assert.Equal(t, int32(1), result.Operation.ServingVerificationAttempt)
			assert.False(t, result.OperationChanged)
			assert.Equal(t, []TrafficRequest{{
				Revision:           1,
				OperationID:        failureTrafficOperationID,
				TopologyGeneration: committedTopology.Generation,
				Replicas:           topologyReplicaIncarnations(committedTopology),
			}}, traffic.withdrawRequests)
			assert.Empty(t, traffic.admitRequests)
			assert.Equal(t, []string{"traffic.withdraw:" + failureTrafficOperationID}, externalMutations)

			t.Log("Restart after command acceptance but before the asynchronous drain effect")
			coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Len(t, traffic.withdrawRequests, 1)

			t.Log("Observe the exact whole-topology drain and only then persist terminal failure")
			traffic.snapshot = workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw, 1, failureTrafficOperationID, committedTopology.Generation,
				nil, topologyReplicaIncarnations(committedTopology),
			)
			coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
			assert.Equal(t, int32(1), result.Operation.ServingVerificationAttempt)
			assert.Equal(t, tt.storedFailure, result.Operation.Failure)
			assert.True(t, result.OperationChanged)
			assert.Len(t, traffic.withdrawRequests, 1)
			assert.Empty(t, traffic.admitRequests)

			t.Log("Restart from the durable failure without replaying verification or traffic mutation")
			input.Operation = result.Operation
			coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Len(t, traffic.withdrawRequests, 1)
			assert.Empty(t, traffic.admitRequests)
			assert.Len(t, verifier.observeCalls, 1)
			assert.Equal(t, []string{
				"traffic.withdraw:" + failureTrafficOperationID,
			}, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorFailsAdoptedOperationOnlyAfterWholeTopologyFence(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(3, "replica-0", "replica-1", "replica-2")
	allAffectedReplicas := workflowReplicaIncarnations("replica-0", "replica-1", "replica-2", "replica-3")
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
			Admitted: allAffectedReplicas,
		},
		externalMutationHistory: &externalMutations,
	}
	request := ServingVerificationRequest{
		OperationID:         workflowTestOperationID,
		Attempt:             1,
		VerificationAttempt: 1,
		Topology:            committedTopology,
	}
	failure := &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "RecoveredCollectiveStalled",
		Message:        "the adopted survivor topology could not make progress",
	}
	verifier := &workflowServingVerifier{
		proof:                   servingVerificationTestProof(request, ServingVerificationPhaseFailed, failure),
		externalMutationHistory: &externalMutations,
	}
	plan := &OperationPlan{
		ID:                "survivor-recovery-plan",
		Intent:            OperationIntentRecover,
		TargetReplicas:    3,
		NominatedReplicas: []ReplicaID{"replica-3"},
	}
	durableOperation := completedUnknownTopology()
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
		Plan:            plan,
		Operation:       durableOperation,
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	oldFenceOperationID := unverifiedTopologyTrafficOperationID(*durableOperation, committedTopology)
	oldFenceReplicas := unverifiedTopologyDrainReplicas(*durableOperation, committedTopology)

	t.Log("Fence the external survivor topology under the old durable operation")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, durableOperation, result.Operation)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, verifier.observeCalls)
	assert.Empty(t, traffic.withdrawRequests)
	require.True(t, result.TrafficStateChanged)
	assert.Equal(t, &TrafficCommand{
		Action: TrafficActionWithdraw,
		Request: TrafficRequest{
			Revision:           1,
			OperationID:        oldFenceOperationID,
			TopologyGeneration: committedTopology.Generation,
			Replicas:           oldFenceReplicas,
		},
	}, result.TrafficCommand)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        oldFenceOperationID,
		TopologyGeneration: committedTopology.Generation,
		Replicas:           oldFenceReplicas,
	}}, traffic.withdrawRequests)
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw, 1, oldFenceOperationID, committedTopology.Generation,
		nil, oldFenceReplicas,
	)

	t.Log("Persist the externally observed survivor topology only after the old-operation fence")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.True(t, result.Operation.Adopted)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.False(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	input.Operation = result.Operation

	t.Log("Persist terminal failure only after the pre-adoption whole-topology fence is observable")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.Equal(t, failure, result.Operation.Failure)
	assert.True(t, result.Operation.Adopted)
	assert.True(t, result.OperationChanged)
	require.NoError(t, validateOperation(*result.Operation))
	input.Operation = result.Operation

	t.Log("Restart from the durable adopted failure without replaying verification or traffic mutation")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.Len(t, traffic.withdrawRequests, 1)
	assert.Len(t, verifier.observeCalls, 1)
	assert.Equal(t, []string{
		"traffic.withdraw:" + oldFenceOperationID,
	}, externalMutations)
}

func TestWorkflowCoordinatorFinishesFailedShrinkReleaseBeforeStartingRecovery(t *testing.T) {
	externalMutations := make([]string, 0)
	baseTopology := workflowTopology(1, "replica-0", "replica-1", "replica-2")
	committedTopology := workflowTopology(2, "replica-0", "replica-1")
	operation := &Operation{
		ID:      workflowTestOperationID,
		Attempt: 1,
		PlanID:  "shrink-plan",
		Intent:  OperationIntentShrink,
		Capability: ResolvedOperationCapability{
			Shape:                   OperationShapePlannedHighRankSuffixShrink,
			TrafficRequirement:      ReconfigurationTrafficQuiesceGroup,
			VerificationRequirement: ServingVerificationRequired,
		},
		SpecGeneration:             2,
		BaseTopology:               baseTopology,
		TargetReplicas:             2,
		NominatedReplicas:          []ReplicaID{"replica-2"},
		Phase:                      OperationPhaseFailed,
		CommittedTopology:          topologyPointer(committedTopology),
		ServingVerificationAttempt: 1,
		ServingVerificationTarget:  topologyPointer(committedTopology),
		StartedAt:                  workflowTestTime.Add(-time.Minute),
		LastTransitionTime:         workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "CollectiveStalled",
			Message:        "the reduced topology could not make progress",
		},
	}
	failureTrafficOperationID := verificationFailureTrafficOperationID(*operation, committedTopology)
	recoveryCapability := ResolvedOperationCapability{
		Shape:                   OperationShapeReplacementRestoration,
		TrafficRequirement:      ReconfigurationTrafficKeepServing,
		VerificationRequirement: ServingVerificationRequired,
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
		validatePlanCapability:  &recoveryCapability,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 2, failureTrafficOperationID, committedTopology.Generation,
			nil, workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		),
		externalMutationHistory: &externalMutations,
	}
	recoveryPlan := &OperationPlan{
		ID:                 "restore-plan",
		Intent:             OperationIntentRecover,
		TargetReplicas:     3,
		RestoredMembership: workflowReplicaNativeMemberships("replica-2"),
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist authorization for the failed shrink's exact old victim before considering recovery")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.Equal(t, "shrink-plan", result.Operation.PlanID)
	assert.False(t, result.OperationChanged)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, ReleaseAuthorization{
		ID:                 workflowTestReleaseID,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: committedTopology.Generation,
		TargetReplicas:     2,
		Replicas: []AuthorizedReplica{{
			ReplicaID: "replica-2",
			SlotID:    "slot-replica-2",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-2",
				UID:       "uid-pod-2",
			}},
		}},
	}, *result.ReleaseAuthorization)
	assert.Empty(t, membership.validatePlanCalls)
	assert.Empty(t, capacity.releaseCalls)
	input.Operation = result.Operation
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from the durable authorization and issue only its exact UID-bound release")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.False(t, result.ReleaseAuthorizationChanged)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, *input.ReleaseAuthorization, capacity.releaseCalls[0])
	assert.Empty(t, membership.validatePlanCalls)

	t.Log("An in-progress release remains owned by the failed shrink across another restart")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplying,
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.False(t, result.ReleaseAuthorizationChanged)
	assert.Len(t, capacity.releaseCalls, 1)
	assert.Empty(t, membership.validatePlanCalls)

	t.Log("Clear authorization only after the exact victim is absent and durably fenced")
	capacity.releaseObservation.Phase = CapacityReleasePhaseApplied
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		},
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-2"),
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.Equal(t, int32(2), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, committedTopology.Generation, result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, membership.validatePlanCalls)
	require.NoError(t, validateOperation(*result.Operation))
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil
	input.Plan = recoveryPlan

	t.Log("Only a later restart may replace the finished shrink record with the distinct recovery plan")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = func() string { return "restore-operation" }
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, "restore-operation", result.Operation.ID)
	assert.Equal(t, "restore-plan", result.Operation.PlanID)
	assert.Equal(t, OperationIntentRecover, result.Operation.Intent)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Equal(t, int32(3), result.Operation.TargetReplicas)
	assert.Equal(t, workflowReplicaNativeMemberships("replica-2"), result.Operation.RestoredMembership)
	assert.Empty(t, result.Operation.JoiningReplicas, "physical restoration is resolved from capacity later")
	assert.True(t, result.OperationChanged)
	require.Len(t, membership.validatePlanCalls, 1)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorRetriesServingVerificationWithDurableAttempt(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	firstRequest := ServingVerificationRequest{
		OperationID:         workflowTestOperationID,
		Attempt:             1,
		VerificationAttempt: 1,
		Topology:            committedTopology,
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, committedTopology.Generation,
			nil, topologyReplicaIncarnations(committedTopology),
		),
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{
		proof: servingVerificationTestProof(
			firstRequest,
			ServingVerificationPhaseFailed,
			&OperationFailure{
				Classification: FailureClassificationRetryable,
				Reason:         "ProbeUnavailable",
				Message:        "serving probe transport was temporarily unavailable",
			},
		),
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       workflowCommittedVerifiedGrowthOperation(),
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Persist the new verification identity because the complete failed topology is already fenced")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int32(2), result.Operation.ServingVerificationAttempt)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.Operation.ServingVerificationProof)
	assert.Empty(t, verifier.ensureRequests)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, traffic.withdrawRequests)
	input.Operation = result.Operation

	t.Log("Restart and observe the durable second attempt without replaying the failed request key")
	verifier.proof = ServingVerificationProof{}
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.Equal(t, int32(2), result.Operation.ServingVerificationAttempt)
	assert.Empty(t, traffic.withdrawRequests)
	assert.Equal(t, []servingVerificationAttempt{
		{
			OperationID:         workflowTestOperationID,
			OperationAttempt:    1,
			VerificationAttempt: 1,
		},
		{
			OperationID:         workflowTestOperationID,
			OperationAttempt:    1,
			VerificationAttempt: 2,
		},
	}, verifier.observeCalls)
	require.Len(t, verifier.ensureRequests, 1)
	assert.Equal(t, int32(2), verifier.ensureRequests[0].VerificationAttempt)
	assert.True(t, servingVerificationTopologiesEqual(committedTopology, verifier.ensureRequests[0].Topology))
	assert.NotContains(t, verifier.requests, servingVerificationAttempt{
		OperationID:         workflowTestOperationID,
		OperationAttempt:    1,
		VerificationAttempt: 1,
	})
	assert.Contains(t, verifier.requests, servingVerificationAttempt{
		OperationID:         workflowTestOperationID,
		OperationAttempt:    1,
		VerificationAttempt: 2,
	})
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []string{
		"serving.verify:" + workflowTestOperationID,
	}, externalMutations)
}

func TestWorkflowCoordinatorReverifiesAfterSameIncarnationCapacityRecovery(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	oldRequest := ServingVerificationRequest{
		OperationID:         workflowTestOperationID,
		Attempt:             1,
		VerificationAttempt: 1,
		Topology:            committedTopology,
	}
	oldProof := servingVerificationTestProof(oldRequest, ServingVerificationPhasePassed, nil)
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Capability.TrafficRequirement = ReconfigurationTrafficKeepServing
	operation.ServingVerificationProof = cloneServingVerificationProof(&oldProof)
	unavailableReplica := workflowReplicaAllocation("replica-2", "pod-2")
	unavailableReplica.Availability = ReplicaAvailabilityUnavailable
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			unavailableReplica,
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                committedTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: topologyReplicaIncarnations(committedTopology),
		},
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Fence every exact active incarnation before treating the availability loss as repairable")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.withdrawRequests)
	require.NotNil(t, result.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
	assert.Equal(t, topologyReplicaIncarnations(committedTopology), result.TrafficCommand.Request.Replicas)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	fenceCommand := cloneTrafficCommand(result.TrafficCommand)
	require.NotNil(t, fenceCommand)
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: &TrafficCommandObservation{
			Command: *fenceCommand,
			Phase:   TrafficCommandPhaseAccepted,
		},
		Drained: topologyReplicaIncarnations(committedTopology),
	}

	t.Log("Invalidate the pre-outage proof and persist a fresh verification identity")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, CapacityRecoveryPhaseRepairing, result.Operation.CapacityRecoveryPhase)
	assert.Equal(t, int32(2), result.Operation.ServingVerificationAttempt)
	assert.Nil(t, result.Operation.ServingVerificationProof)
	assert.Empty(t, capacity.ensureCalls)
	input.Operation = result.Operation

	t.Log("Repair the same stable slot and runtime incarnation while it remains fenced")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []CapacityRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: committedTopology.Generation,
		TargetReplicas:     3,
		RequiredReplicas:   workflowRequiredReplicaAllocations("replica-0", "replica-1", "replica-2"),
	}}, capacity.ensureCalls)
	assert.Empty(t, verifier.ensureRequests)

	t.Log("Persist Verifying after readiness recovers; readiness alone cannot revive the old proof")
	capacity.snapshot.Allocations[2].Availability = ReplicaAvailabilityAvailable
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, CapacityRecoveryPhaseVerifying, result.Operation.CapacityRecoveryPhase)
	assert.Nil(t, result.Operation.ServingVerificationProof)
	assert.Empty(t, verifier.ensureRequests)
	input.Operation = result.Operation

	t.Log("Start only the fresh attempt after its Verifying marker survives restart")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, verifier.ensureRequests, 1)
	newRequest := verifier.ensureRequests[0]
	assert.Equal(t, int32(2), newRequest.VerificationAttempt)
	assert.True(t, servingVerificationTopologiesEqual(committedTopology, newRequest.Topology))
	assert.Empty(t, traffic.admitRequests)

	t.Log("Persist the fresh proof, then clear recovery before restoring admission")
	verifier.proof = servingVerificationTestProof(newRequest, ServingVerificationPhasePassed, nil)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.OperationChanged)
	require.NotNil(t, result.Operation.ServingVerificationProof)
	assert.Empty(t, traffic.admitRequests)
	input.Operation = result.Operation

	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, CapacityRecoveryPhaseNone, result.Operation.CapacityRecoveryPhase)
	assert.Empty(t, traffic.admitRequests)
	input.Operation = result.Operation

	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, topologyReplicaIncarnations(committedTopology), traffic.admitRequests[len(traffic.admitRequests)-1].Replicas)
}

func TestUnknownCapacityRecoveryRequiresTheExactCommittedTopology(t *testing.T) {
	newUnknownOperation := func() *Operation {
		operation := workflowCommittedVerifiedGrowthOperation()
		operation.Phase = OperationPhaseUnknown
		return operation
	}

	t.Run("authoritative drift stays fenced without verifying an unadopted topology", func(t *testing.T) {
		externalMutations := make([]string, 0)
		operation := newUnknownOperation()
		currentTopology := workflowTopology(3, "replica-0", "replica-1")
		capacity := &workflowCapacityAdapter{
			snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
			}},
			externalMutationHistory: &externalMutations,
		}
		membership := &workflowMembershipAdapter{
			topology: currentTopology,
			operation: BackendOperation{
				ID:             workflowTestOperationID,
				Attempt:        1,
				TargetReplicas: 3,
				Phase:          BackendOperationPhaseUnknown,
			},
			externalMutationHistory: &externalMutations,
		}
		traffic := &workflowTrafficAdapter{
			snapshot: TrafficSnapshot{
				Admitted: unverifiedTopologyDrainReplicas(*operation, currentTopology),
			},
			externalMutationHistory: &externalMutations,
		}
		verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
		input := ReconcileInput{
			GroupID:         "group-0",
			SpecGeneration:  2,
			DesiredReplicas: 3,
			Operation:       operation,
		}
		coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
		fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, currentTopology)
		fencedReplicas := unverifiedTopologyDrainReplicas(*operation, currentTopology)

		t.Log("Fence the complete unadopted topology before considering missing capacity")
		result, err := coordinator.Reconcile(context.Background(), input)
		require.NoError(t, err)
		assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
		assert.Equal(t, CapacityRecoveryPhaseNone, result.Operation.CapacityRecoveryPhase)
		assert.Empty(t, capacity.ensureCalls)
		assert.Empty(t, verifier.observeCalls)
		assert.Empty(t, verifier.ensureRequests)
		dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
		require.Len(t, traffic.withdrawRequests, 1)
		assert.Equal(t, TrafficRequest{
			Revision:           1,
			OperationID:        fenceOperationID,
			TopologyGeneration: currentTopology.Generation,
			Replicas:           fencedReplicas,
		}, traffic.withdrawRequests[0])

		t.Log("Remain fenced after restart instead of starting recovery verification for the unadopted generation")
		traffic.snapshot = workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw,
			1,
			fenceOperationID,
			currentTopology.Generation,
			nil,
			fencedReplicas,
		)
		coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
		result, err = coordinator.Reconcile(context.Background(), input)
		require.NoError(t, err)
		assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
		assert.Equal(t, CapacityRecoveryPhaseNone, result.Operation.CapacityRecoveryPhase)
		assert.Equal(t, int32(1), result.Operation.ServingVerificationAttempt)
		assert.False(t, result.OperationChanged)
		assert.Empty(t, capacity.ensureCalls)
		assert.Empty(t, verifier.observeCalls)
		assert.Empty(t, verifier.ensureRequests)
	})

	t.Run("exact committed topology loss enters capacity recovery", func(t *testing.T) {
		externalMutations := make([]string, 0)
		operation := newUnknownOperation()
		committedTopology := cloneTopology(*operation.CommittedTopology)
		capacity := &workflowCapacityAdapter{
			snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
			}},
			externalMutationHistory: &externalMutations,
		}
		membership := &workflowMembershipAdapter{
			topology: committedTopology,
			operation: BackendOperation{
				ID:             workflowTestOperationID,
				Attempt:        1,
				TargetReplicas: 3,
				Phase:          BackendOperationPhaseUnknown,
			},
			externalMutationHistory: &externalMutations,
		}
		traffic := &workflowTrafficAdapter{
			snapshot:                TrafficSnapshot{Admitted: topologyReplicaIncarnations(committedTopology)},
			externalMutationHistory: &externalMutations,
		}
		verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
		input := ReconcileInput{
			GroupID:         "group-0",
			SpecGeneration:  2,
			DesiredReplicas: 3,
			Operation:       operation,
		}
		coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
		fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, committedTopology)
		fencedReplicas := unverifiedTopologyDrainReplicas(*operation, committedTopology)

		result, err := coordinator.Reconcile(context.Background(), input)
		require.NoError(t, err)
		dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
		traffic.snapshot = workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw,
			1,
			fenceOperationID,
			committedTopology.Generation,
			nil,
			fencedReplicas,
		)

		coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
		result, err = coordinator.Reconcile(context.Background(), input)
		require.NoError(t, err)
		assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
		assert.Equal(t, CapacityRecoveryPhaseRepairing, result.Operation.CapacityRecoveryPhase)
		assert.Equal(t, int32(2), result.Operation.ServingVerificationAttempt)
		assert.True(t, result.OperationChanged)
		assert.Empty(t, capacity.ensureCalls, "persist recovery state before repairing capacity")
		assert.Empty(t, verifier.observeCalls)
		assert.Empty(t, verifier.ensureRequests)
	})
}

func TestUnavailableActiveMemberAllowsExplicitRecoveryAfterSafetyFence(t *testing.T) {
	tests := []struct {
		name         string
		plan         *OperationPlan
		wantRecovery bool
	}{
		{
			name: "distinct recovery plan",
			plan: &OperationPlan{
				ID:                "survivor-recovery-plan",
				Intent:            OperationIntentRecover,
				TargetReplicas:    2,
				NominatedReplicas: []ReplicaID{"replica-2"},
			},
			wantRecovery: true,
		},
		{name: "same-incarnation repair without a plan"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			operation := workflowCommittedVerifiedGrowthOperation()
			unavailable := workflowReplicaAllocation("replica-2", "pod-2")
			unavailable.Availability = ReplicaAvailabilityUnavailable
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-1", "pod-1"),
					unavailable,
				}},
				externalMutationHistory: &externalMutations,
			}
			recoveryCapability := ResolvedOperationCapability{
				Shape:                   OperationShapeSurvivorReduction,
				TrafficRequirement:      ReconfigurationTrafficKeepServing,
				VerificationRequirement: ServingVerificationRequired,
			}
			membership := &workflowMembershipAdapter{
				topology:                cloneTopology(*operation.CommittedTopology),
				validatePlanCapability:  &recoveryCapability,
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot:                TrafficSnapshot{Admitted: topologyReplicaIncarnations(membership.topology)},
				externalMutationHistory: &externalMutations,
			}
			verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  3,
				DesiredReplicas: operation.TargetReplicas,
				Plan:            tt.plan,
				Operation:       operation,
			}
			if tt.plan != nil {
				input.DesiredReplicas = tt.plan.TargetReplicas
			}
			coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
			fencedReplicas := unverifiedTopologyDrainReplicas(*operation, membership.topology)

			t.Log("Fence the exact active topology before capacity repair or a replacement operation")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Empty(t, capacity.ensureCalls)
			dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			traffic.snapshot = workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw,
				1,
				fenceOperationID,
				membership.topology.Generation,
				nil,
				fencedReplicas,
			)

			t.Log("Invalidate verification that could have overlapped the availability regression")
			coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, CapacityRecoveryPhaseRepairing, result.Operation.CapacityRecoveryPhase)
			assert.Equal(t, int32(2), result.Operation.ServingVerificationAttempt)
			assert.Empty(t, capacity.ensureCalls)
			input.Operation = result.Operation

			coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			coordinator.operations.newOperationID = func() string { return workflowTestRecoveryOperationID }
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			if tt.wantRecovery {
				require.NotNil(t, result.Operation)
				assert.Equal(t, workflowTestRecoveryOperationID, result.Operation.ID)
				assert.Equal(t, tt.plan.ID, result.Operation.PlanID)
				assert.Equal(t, OperationPhasePending, result.Operation.Phase)
				assert.Equal(t, OperationShapeSurvivorReduction, result.Operation.Capability.Shape)
				assert.Empty(t, capacity.ensureCalls)
				assert.Empty(t, verifier.observeCalls)
				assert.Empty(t, verifier.ensureRequests)
				return
			}

			assert.Equal(t, input.Operation, result.Operation)
			assert.Equal(t, []CapacityRequest{{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: membership.topology.Generation,
				TargetReplicas:     membership.topology.ReplicaCount(),
				RequiredReplicas:   workflowRequiredReplicaAllocations("replica-0", "replica-1", "replica-2"),
			}}, capacity.ensureCalls)
			assert.Empty(t, verifier.observeCalls)
			assert.Empty(t, verifier.ensureRequests)
		})
	}
}

func TestWorkflowCoordinatorClearsCapacityRecoveryOnTerminalVerificationOutcome(t *testing.T) {
	tests := []struct {
		name       string
		phase      ServingVerificationPhase
		failure    *OperationFailure
		wantReason string
	}{
		{
			name:  "failed",
			phase: ServingVerificationPhaseFailed,
			failure: &OperationFailure{
				Classification: FailureClassificationTerminal,
				Reason:         "CollectiveStalled",
			},
			wantReason: "CollectiveStalled",
		},
		{
			name:       "unknown",
			phase:      ServingVerificationPhaseUnknown,
			wantReason: "ServingVerificationUnknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
			operation := workflowCommittedVerifiedGrowthOperation()
			operation.Capability.TrafficRequirement = ReconfigurationTrafficKeepServing
			operation.ServingVerificationAttempt = 2
			operation.ServingVerificationProof = nil
			operation.CapacityRecoveryPhase = CapacityRecoveryPhaseVerifying
			request := ServingVerificationRequest{
				OperationID:         workflowTestOperationID,
				Attempt:             1,
				VerificationAttempt: 2,
				Topology:            committedTopology,
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
				snapshot: workflowTrafficSnapshotWithCommand(
					TrafficActionWithdraw,
					1,
					workflowTestOperationID,
					committedTopology.Generation,
					nil,
					topologyReplicaIncarnations(committedTopology),
				),
				externalMutationHistory: &externalMutations,
			}
			verifier := &workflowServingVerifier{
				proof:                   servingVerificationTestProof(request, tt.phase, tt.failure),
				externalMutationHistory: &externalMutations,
			}
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 3,
				Operation:       operation,
			}
			coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
			assert.Equal(t, CapacityRecoveryPhaseNone, result.Operation.CapacityRecoveryPhase)
			require.NotNil(t, result.Operation.Failure)
			assert.Equal(t, tt.wantReason, result.Operation.Failure.Reason)
			require.NoError(t, validateOperation(*result.Operation))
			input.Operation = result.Operation

			t.Log("Restart from a valid durable terminal record without restoring recovery state")
			coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
			assert.Equal(t, CapacityRecoveryPhaseNone, result.Operation.CapacityRecoveryPhase)
			assert.False(t, result.OperationChanged)
			require.NoError(t, validateOperation(*result.Operation))
		})
	}
}

func TestWorkflowCoordinatorQuiescesGroupAndReadmitsVerifiedTopology(t *testing.T) {
	externalMutations := make([]string, 0)
	baseTopology := workflowTopology(1, "replica-0", "replica-1")
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                baseTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
		},
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       workflowPendingQuiescingGrowthOperation(),
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Withdraw every current member because this operation requires a fully quiesced commit")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Empty(t, traffic.withdrawRequests)
	input.Operation = result.Operation
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 1,
		Replicas:           workflowReplicaIncarnations("replica-0", "replica-1"),
	}}, traffic.withdrawRequests)
	assert.Empty(t, membership.submitCalls)

	t.Log("Persist the exact membership request only after every base member is drained")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw, 1, workflowTestOperationID, membership.topology.Generation,
		nil, workflowReplicaIncarnations("replica-0", "replica-1"),
	)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Submit while the operation-correlated group drain remains observable")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	require.Len(t, membership.submitCalls, 1)
	assert.Empty(t, traffic.admitRequests)
	input.Operation = result.Operation

	t.Log("Fence the independently committed topology before persisting backend progress")
	membership.topology = committedTopology
	membership.operation = BackendOperation{
		ID:                workflowTestOperationID,
		Attempt:           1,
		BackendID:         "backend-" + workflowTestOperationID,
		TargetReplicas:    3,
		Phase:             BackendOperationPhaseCommitted,
		CommittedTopology: topologyPointer(committedTopology),
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, verifier.ensureRequests)
	assert.Empty(t, traffic.admitRequests)
	unverifiedOperationID := workflowTestOperationID
	unverifiedReplicas := workflowReplicaIncarnations("replica-2")
	assert.Len(t, traffic.withdrawRequests, 1)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, TrafficRequest{
		Revision:           2,
		OperationID:        unverifiedOperationID,
		TopologyGeneration: committedTopology.Generation,
		Replicas:           unverifiedReplicas,
	}, traffic.withdrawRequests[len(traffic.withdrawRequests)-1])
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw,
			2,
			unverifiedOperationID,
			membership.topology.Generation,
			unverifiedReplicas,
		),
		Drained: topologyReplicaIncarnations(committedTopology),
	}

	t.Log("Persist the committed expanded topology after the exact fence survives restart")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	input.Operation = result.Operation

	t.Log("Start verification while the complete committed group remains withdrawn")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, verifier.ensureRequests, 1)
	assert.Empty(t, traffic.admitRequests)
	request := verifier.ensureRequests[0]
	assert.True(t, servingVerificationTopologiesEqual(committedTopology, request.Topology))

	t.Log("Persist successful verification without crossing the traffic admission boundary")
	verifier.proof = servingVerificationTestProof(request, ServingVerificationPhasePassed, nil)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
	input.Operation = result.Operation

	t.Log("Readmit the complete verified topology after restart")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.admitRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           3,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		Replicas:           workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
	}}, traffic.admitRequests)
	assert.Equal(t, []string{
		"traffic.withdraw:" + workflowTestOperationID,
		"membership.submit:" + workflowTestOperationID,
		"traffic.withdraw:" + unverifiedOperationID,
		"serving.verify:" + workflowTestOperationID,
		"traffic.admit:" + workflowTestOperationID,
	}, externalMutations)
}

func TestWorkflowCoordinatorReassertsQuiescenceWhileMembershipIsInFlight(t *testing.T) {
	tests := []struct {
		name         string
		operation    OperationPhase
		backendPhase BackendOperationPhase
	}{
		{
			name:         "accepted",
			operation:    OperationPhaseAccepted,
			backendPhase: BackendOperationPhaseAccepted,
		},
		{
			name:         "committing",
			operation:    OperationPhaseCommitting,
			backendPhase: BackendOperationPhaseCommitting,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			baseTopology := workflowTopology(1, "replica-0", "replica-1")
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-1", "pod-1"),
					workflowReplicaAllocation("replica-2", "pod-2"),
				}},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology: baseTopology,
				operation: BackendOperation{
					ID:             workflowTestOperationID,
					Attempt:        1,
					BackendID:      "backend-" + workflowTestOperationID,
					TargetReplicas: 3,
					Phase:          tt.backendPhase,
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
			verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
			operation := workflowPendingQuiescingGrowthOperation()
			operation.Phase = tt.operation
			operation.BackendOperationID = "backend-" + workflowTestOperationID
			operation.JoiningReplicas = workflowReplicaIncarnations("replica-2")
			coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 3,
				Operation:       operation,
			}

			t.Log("Reissue the whole-group withdrawal when an in-flight quiescence observation regresses")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, tt.operation, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Empty(t, traffic.withdrawRequests)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			assert.Equal(t, []TrafficRequest{{
				Revision:           2,
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 1,
				Replicas:           workflowReplicaIncarnations("replica-0", "replica-1"),
			}}, traffic.withdrawRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, traffic.admitRequests)
			assert.Equal(t, []string{"traffic.withdraw:" + workflowTestOperationID}, externalMutations)
		})
	}
}

func TestNativeMemberRemapDrainsChangedReplicaBeforeSubmissionAndVerification(t *testing.T) {
	externalMutations := make([]string, 0)
	baseTopology := workflowTopology(1, "replica-0", "replica-1")
	targetTopology := cloneTopology(baseTopology)
	targetTopology.Generation = 2
	targetTopology.Replicas[0].NativeMembers = []NativeMemberID{"native-replica-0-remapped"}
	operation := &Operation{
		ID:      workflowTestOperationID,
		Attempt: 1,
		PlanID:  "remap-plan",
		Intent:  OperationIntentRecover,
		Capability: ResolvedOperationCapability{
			Shape:                   OperationShapeNativeMemberRemapping,
			TrafficRequirement:      ReconfigurationTrafficKeepServing,
			VerificationRequirement: ServingVerificationRequired,
		},
		SpecGeneration:     2,
		BaseTopology:       baseTopology,
		TargetReplicas:     2,
		TargetMembership:   cloneReplicaMemberships(targetTopology.Replicas),
		Phase:              OperationPhasePending,
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
	}
	changedReplica := []ReplicaIncarnation{baseTopology.Replicas[0].Incarnation}
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
		}},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                baseTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: topologyReplicaIncarnations(baseTopology),
		},
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  operation.SpecGeneration,
		DesiredReplicas: operation.TargetReplicas,
		Plan: &OperationPlan{
			ID:               operation.PlanID,
			Intent:           operation.Intent,
			TargetReplicas:   operation.TargetReplicas,
			TargetMembership: cloneReplicaMemberships(operation.TargetMembership),
		},
		Operation: operation,
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Drain only the base replica whose engine-native membership will change")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.True(t, result.TrafficStateChanged)
	require.NotNil(t, result.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
	assert.Equal(t, changedReplica, result.TrafficCommand.Request.Replicas)
	assert.Empty(t, membership.validateRequestCalls)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, verifier.ensureRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)

	t.Log("Persist the exact request only after the changed replica's drain is observable")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw,
		result.TrafficRevision,
		operation.ID,
		baseTopology.Generation,
		[]ReplicaIncarnation{baseTopology.Replicas[1].Incarnation},
		changedReplica,
	)
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	require.Len(t, membership.validateRequestCalls, 1)
	assert.Equal(t, operation.TargetMembership, membership.validateRequestCalls[0].TargetMembership)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Submit the frozen remap while that exact drain remains stable")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	require.Len(t, membership.submitCalls, 1)
	input.Operation = result.Operation

	t.Log("Persist the committed native mapping before starting its serving verification")
	membership.topology = targetTopology
	membership.operation = BackendOperation{
		ID:                operation.ID,
		Attempt:           operation.Attempt,
		BackendID:         "backend-" + operation.ID,
		TargetReplicas:    operation.TargetReplicas,
		Phase:             BackendOperationPhaseCommitted,
		CommittedTopology: topologyPointer(targetTopology),
	}
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int32(1), result.Operation.ServingVerificationAttempt)
	assert.Empty(t, verifier.ensureRequests)
	input.Operation = result.Operation

	t.Log("A traffic regression restores the changed replica's fence before verification")
	regressedAdmit := TrafficCommand{
		Action: TrafficActionAdmit,
		Request: TrafficRequest{
			Revision:           result.TrafficRevision + 1,
			OperationID:        operation.ID,
			TopologyGeneration: targetTopology.Generation,
			Replicas:           changedReplica,
		},
	}
	input.TrafficRevision = regressedAdmit.Request.Revision
	input.TrafficCommand = cloneTrafficCommand(&regressedAdmit)
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: &TrafficCommandObservation{
			Command: regressedAdmit,
			Phase:   TrafficCommandPhaseAccepted,
		},
		Admitted: topologyReplicaIncarnations(targetTopology),
	}
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.TrafficStateChanged)
	require.NotNil(t, result.TrafficCommand)
	assert.Equal(t, TrafficActionWithdraw, result.TrafficCommand.Action)
	assert.Equal(t, changedReplica, result.TrafficCommand.Request.Replicas)
	assert.Empty(t, verifier.ensureRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)

	t.Log("Verification may start only after the remapped replica is fenced again")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw,
		result.TrafficRevision,
		operation.ID,
		targetTopology.Generation,
		[]ReplicaIncarnation{targetTopology.Replicas[1].Incarnation},
		changedReplica,
	)
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, verifier.ensureRequests, 1)
	assert.Equal(t, targetTopology, verifier.ensureRequests[0].Topology)
	assert.Empty(t, traffic.admitRequests)
}

func TestWorkflowCoordinatorUsesValidatedMonotonicTrafficStateForQuiesceReadmission(t *testing.T) {
	tests := []struct {
		name               string
		trafficOperationID string
		wantError          string
	}{
		{
			name:      "empty traffic operation",
			wantError: "traffic command operation ID must not be empty",
		},
		{
			name:               "different traffic operation",
			trafficOperationID: "previous-operation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			committedTopology := workflowTopology(2, "replica-0", "replica-1")
			request := ServingVerificationRequest{
				OperationID:         workflowTestOperationID,
				Attempt:             1,
				VerificationAttempt: 1,
				Topology:            committedTopology,
			}
			proof := servingVerificationTestProof(request, ServingVerificationPhasePassed, nil)
			operation := &Operation{
				ID:      workflowTestOperationID,
				Attempt: 1,
				PlanID:  "remap-plan-1",
				Intent:  OperationIntentRecover,
				Capability: ResolvedOperationCapability{
					Shape:                   OperationShapeNativeMemberRemapping,
					TrafficRequirement:      ReconfigurationTrafficQuiesceGroup,
					VerificationRequirement: ServingVerificationRequired,
				},
				SpecGeneration: 2,
				BaseTopology: MembershipTopology{
					Generation: 1,
					Replicas: []ReplicaMembership{
						workflowReplicaMembership("replica-0", "native-before-remap-replica-0"),
						workflowReplicaMembership("replica-1", "native-before-remap-replica-1"),
					},
				},
				TargetReplicas:             2,
				TargetMembership:           cloneReplicaMemberships(committedTopology.Replicas),
				Phase:                      OperationPhaseCommitted,
				CommittedTopology:          topologyPointer(committedTopology),
				ServingVerificationAttempt: 1,
				ServingVerificationTarget:  topologyPointer(committedTopology),
				ServingVerificationProof:   cloneServingVerificationProof(&proof),
				StartedAt:                  workflowTestTime,
				LastTransitionTime:         workflowTestTime,
			}
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
					workflowReplicaAllocation("replica-1", "pod-1"),
				}},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                committedTopology,
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: workflowTrafficSnapshotWithCommand(
					TrafficActionAdmit, 1, tt.trafficOperationID, 1,
					workflowReplicaIncarnations("replica-0", "replica-1"), nil,
				),
				externalMutationHistory: &externalMutations,
			}
			verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
			coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 2,
				Operation:       operation,
			}

			t.Log("Validate the latest monotonic traffic command before accepting the exact admitted set")
			result, err := coordinator.Reconcile(context.Background(), input)
			if tt.wantError != "" {
				require.ErrorContains(t, err, tt.wantError)
				assert.Empty(t, externalMutations)
				return
			}
			require.NoError(t, err)
			assert.True(t, result.Operation.PostCommitComplete)
			assert.True(t, result.OperationChanged)
			assert.Empty(t, verifier.observeCalls)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, externalMutations)
		})
	}
}

func workflowCommittedVerifiedGrowthOperation() *Operation {
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
	return &Operation{
		ID:      workflowTestOperationID,
		Attempt: 1,
		Intent:  OperationIntentGrow,
		Capability: ResolvedOperationCapability{
			Shape:                   OperationShapeFreshGrowth,
			TrafficRequirement:      ReconfigurationTrafficQuiesceGroup,
			VerificationRequirement: ServingVerificationRequired,
		},
		SpecGeneration:             2,
		BaseTopology:               workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:             3,
		JoiningReplicas:            workflowReplicaIncarnations("replica-2"),
		Phase:                      OperationPhaseCommitted,
		CommittedTopology:          topologyPointer(committedTopology),
		ServingVerificationAttempt: 1,
		ServingVerificationTarget:  topologyPointer(committedTopology),
		StartedAt:                  workflowTestTime,
		LastTransitionTime:         workflowTestTime,
	}
}

func workflowPendingQuiescingGrowthOperation() *Operation {
	operation := workflowCommittedVerifiedGrowthOperation()
	operation.Capability.TrafficRequirement = ReconfigurationTrafficQuiesceGroup
	operation.Phase = OperationPhasePending
	operation.CommittedTopology = nil
	operation.ServingVerificationAttempt = 0
	operation.ServingVerificationTarget = nil
	operation.ServingVerificationProof = nil
	operation.JoiningReplicas = nil
	return operation
}

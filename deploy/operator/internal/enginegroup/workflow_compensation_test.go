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
	"fmt"
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

func (f *commitRaceMembershipAdapter) ValidatePlan(
	_ context.Context,
	_ GroupID,
	topology MembershipTopology,
	plan OperationPlan,
) (ResolvedOperationCapability, error) {
	if err := validatePlan(plan, topology); err != nil {
		return ResolvedOperationCapability{}, err
	}
	return resolveTestOperationCapability(plan, topology, allTestMembershipCapabilities())
}

func (f *commitRaceMembershipAdapter) ValidateObservedTransition(
	_ context.Context,
	_ GroupID,
	transition ObservedMembershipTransition,
) (ResolvedOperationCapability, error) {
	return resolveTestObservedTransitionCapability(transition, allTestMembershipCapabilities())
}

func (f *commitRaceMembershipAdapter) ValidateRequest(
	_ context.Context,
	_ GroupID,
	request MembershipRequest,
) error {
	return validateCapabilityForTestRequest(request.Capability, request)
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
	cloned.BaseTopology = cloneTopology(request.BaseTopology)
	cloned.JoiningReplicas = slices.Clone(request.JoiningReplicas)
	cloned.NominatedReplicas = slices.Clone(request.NominatedReplicas)
	f.submitOperationRequests = append(f.submitOperationRequests, cloned)
	return BackendOperation{}, errors.New("unexpected membership submission")
}

func TestWorkflowCoordinatorFailsClosedForPreSubmitShrinkAfterBaseTopologyDrift(t *testing.T) {
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
			traffic := &workflowTrafficAdapter{externalMutationHistory: &externalMutations}
			operation := &Operation{
				ID:             workflowTestOperationID,
				Attempt:        1,
				PlanID:         "plan-1",
				Intent:         OperationIntentShrink,
				Capability:     testOperationCapability(OperationShapePlannedHighRankSuffixShrink),
				SpecGeneration: 2,
				BaseTopology: workflowTopology(
					1,
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				),
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

			fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
			fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, membership.topology)
			t.Log("Fence the unvalidated authoritative topology before persisting nested operation progress")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, tt.phase, result.Operation.Phase)
			assert.Nil(t, result.Operation.Failure)
			assert.Nil(t, result.Operation.CompensationTopology)
			assert.False(t, result.OperationChanged)
			assert.Equal(t, tt.wantObserveOperationCalls, membership.observeOperationCalls)
			assert.Empty(t, membership.submitCalls)
			assert.Empty(t, capacity.ensureCalls)
			assert.Empty(t, capacity.releaseCalls)
			assert.Empty(t, traffic.withdrawRequests)
			require.True(t, result.TrafficStateChanged)
			assert.Equal(t, &TrafficCommand{
				Action: TrafficActionWithdraw,
				Request: TrafficRequest{
					Revision:           1,
					OperationID:        fenceOperationID,
					TopologyGeneration: membership.topology.Generation,
					Replicas:           fencedIncarnations,
				},
			}, result.TrafficCommand)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, externalMutations)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			assert.Equal(t, []TrafficRequest{{
				Revision:           1,
				OperationID:        fenceOperationID,
				TopologyGeneration: membership.topology.Generation,
				Replicas:           fencedIncarnations,
			}}, traffic.withdrawRequests)
			assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)
			traffic.snapshot = workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw, 1, fenceOperationID, membership.topology.Generation,
				nil, fencedIncarnations,
			)

			t.Log("Persist Unknown only after the exact fence is observable after restart")
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
			assert.Empty(t, result.Operation.CleanupReplicaSlots)
			assert.True(t, result.OperationChanged)
			assert.Nil(t, result.ReleaseAuthorization)
			assert.Empty(t, capacity.ensureCalls)
			assert.Empty(t, capacity.releaseCalls)
			assert.Equal(t, []TrafficRequest{{
				Revision:           1,
				OperationID:        fenceOperationID,
				TopologyGeneration: membership.topology.Generation,
				Replicas:           fencedIncarnations,
			}}, traffic.withdrawRequests)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)
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
			ID:                workflowTestOperationID,
			Attempt:           1,
			BackendID:         "backend-" + workflowTestOperationID,
			TargetReplicas:    2,
			Phase:             BackendOperationPhaseCommitted,
			CommittedTopology: topologyPointer(workflowTopology(2, "replica-0", "replica-1")),
		},
	}
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, 2,
			workflowReplicaIncarnations("replica-0", "replica-1"),
			workflowReplicaIncarnations("replica-2", "replica-3"),
		),
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:             workflowTestOperationID,
		Attempt:        1,
		PlanID:         "plan-1",
		Intent:         OperationIntentShrink,
		Capability:     testOperationCapability(OperationShapePlannedHighRankSuffixShrink),
		SpecGeneration: 2,
		BaseTopology: workflowTopology(
			1,
			"replica-0",
			"replica-1",
			"replica-2",
			"replica-3",
		),
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

	t.Log("Attribute a commit that races the final pre-submit topology check once its exact retirees are already fenced")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, int64(2), result.Operation.CommittedTopology.Generation)
	assert.Equal(t, "backend-"+workflowTestOperationID, result.Operation.BackendOperationID)
	assert.Nil(t, result.Operation.Failure)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, traffic.withdrawRequests)
	assert.Empty(t, membership.submitOperationRequests)
	assert.GreaterOrEqual(t, membership.observeTopologyCalls, 2)
	assert.GreaterOrEqual(t, membership.observeOperationCalls, 1)
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
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-2", "replica-3"),
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, workflowReplicaSlotBindings("replica-2", "replica-3"), result.Capacity.FencedReplicaSlots)
	assert.Empty(t, membership.submitOperationRequests)
	assert.Equal(t, []string{
		"capacity.release:" + workflowTestReleaseID,
	}, externalMutations)
}

func TestWorkflowCoordinatorInitializesServingVerificationTargetDuringCompensation(t *testing.T) {
	externalMutations := make([]string, 0)
	compensationTopology := workflowTopology(3, "replica-0", "replica-1")
	compensationTopology.Replicas[1].NativeMembers[0] = "native-replica-1-replacement"
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{
			Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
			},
			FencedReplicaSlots: workflowReplicaSlotBindings("replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                compensationTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionWithdraw, 1, workflowTestOperationID, compensationTopology.Generation,
			nil, topologyReplicaIncarnations(compensationTopology),
		),
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
	operation := &Operation{
		ID:      workflowTestOperationID,
		Attempt: 1,
		Intent:  OperationIntentGrow,
		Capability: ResolvedOperationCapability{
			Shape:                   OperationShapeFreshGrowth,
			TrafficRequirement:      ReconfigurationTrafficQuiesceGroup,
			VerificationRequirement: ServingVerificationRequired,
		},
		SpecGeneration:             2,
		BaseTopology:               cloneTopology(compensationTopology),
		TargetReplicas:             3,
		JoiningReplicas:            workflowReplicaIncarnations("replica-2"),
		CleanupReplicaSlots:        workflowReplicaSlotBindings("replica-2"),
		CapacityTargetReplicas:     2,
		CapacityTopologyGeneration: 3,
		CapacityTargetApplied:      true,
		Phase:                      OperationPhaseAborting,
		CompensationTopology:       topologyPointer(compensationTopology),
		StartedAt:                  workflowTestTime.Add(-time.Minute),
		LastTransitionTime:         workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "AtomicRequestRejected",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Persist a new verification identity bound to the exact compensation topology")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation.ServingVerificationTarget)
	assert.Equal(t, int32(1), result.Operation.ServingVerificationAttempt)
	assert.True(t, servingVerificationTopologiesEqual(
		compensationTopology,
		*result.Operation.ServingVerificationTarget,
	))
	assert.True(t, result.OperationChanged)
	assert.Empty(t, verifier.observeCalls)
	assert.Empty(t, verifier.ensureRequests)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart from the durable target and start only its exact verification attempt")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	require.Len(t, verifier.ensureRequests, 1)
	assert.Equal(t, int32(1), verifier.ensureRequests[0].VerificationAttempt)
	assert.True(t, servingVerificationTopologiesEqual(
		compensationTopology,
		verifier.ensureRequests[0].Topology,
	))
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []string{"serving.verify:" + workflowTestOperationID}, externalMutations)
}

func TestWorkflowCoordinatorKeepsTerminalVerificationOutcomeDuringCompensationDurablyAborting(t *testing.T) {
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
				Reason:         "CompensatedCollectiveStalled",
				Message:        "the restored base topology could not make progress",
			},
			storedFailure: &OperationFailure{
				Classification: FailureClassificationTerminal,
				Reason:         "CompensatedCollectiveStalled",
				Message:        "the restored base topology could not make progress",
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
		for _, initialPhase := range []OperationPhase{OperationPhaseAborting, OperationPhaseAborted} {
			t.Run(tt.name+"/"+string(initialPhase), func(t *testing.T) {
				externalMutations := make([]string, 0)
				compensationTopology := workflowTopology(3, "replica-0", "replica-1")
				cleanupReplicas := []ReplicaID{"replica-2"}
				capacity := &workflowCapacityAdapter{
					snapshot: CapacitySnapshot{
						Allocations: []ReplicaAllocation{
							workflowReplicaAllocation("replica-0", "pod-0"),
							workflowReplicaAllocation("replica-1", "pod-1"),
						},
						FencedReplicaSlots: workflowReplicaSlotBindings(cleanupReplicas...),
					},
					externalMutationHistory: &externalMutations,
				}
				membership := &workflowMembershipAdapter{
					topology:                compensationTopology,
					externalMutationHistory: &externalMutations,
				}
				traffic := &workflowTrafficAdapter{
					snapshot: workflowTrafficSnapshotWithCommand(
						TrafficActionWithdraw, 1, workflowTestOperationID, compensationTopology.Generation,
						nil, topologyReplicaIncarnations(compensationTopology),
					),
					externalMutationHistory: &externalMutations,
				}
				request := ServingVerificationRequest{
					OperationID:         workflowTestOperationID,
					Attempt:             1,
					VerificationAttempt: 1,
					Topology:            compensationTopology,
				}
				verifier := &workflowServingVerifier{
					proof: servingVerificationTestProof(
						request,
						tt.phase,
						tt.observedFailure,
					),
					externalMutationHistory: &externalMutations,
				}
				operation := &Operation{
					ID:      workflowTestOperationID,
					Attempt: 1,
					Intent:  OperationIntentGrow,
					Capability: ResolvedOperationCapability{
						Shape:                   OperationShapeFreshGrowth,
						TrafficRequirement:      ReconfigurationTrafficQuiesceGroup,
						VerificationRequirement: ServingVerificationRequired,
					},
					SpecGeneration:             2,
					BaseTopology:               cloneTopology(compensationTopology),
					TargetReplicas:             3,
					JoiningReplicas:            workflowReplicaIncarnations("replica-2"),
					CleanupReplicaSlots:        workflowReplicaSlotBindings(cleanupReplicas...),
					CapacityTargetReplicas:     2,
					CapacityTopologyGeneration: compensationTopology.Generation,
					CapacityTargetApplied:      true,
					Phase:                      initialPhase,
					CompensationTopology:       topologyPointer(compensationTopology),
					ServingVerificationAttempt: 1,
					ServingVerificationTarget:  topologyPointer(compensationTopology),
					StartedAt:                  workflowTestTime.Add(-time.Minute),
					LastTransitionTime:         workflowTestTime,
					Failure: &OperationFailure{
						Classification: FailureClassificationTerminal,
						Reason:         "AtomicRequestRejected",
					},
				}
				input := ReconcileInput{
					GroupID:         "group-0",
					SpecGeneration:  2,
					DesiredReplicas: 2,
					Operation:       operation,
				}
				coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

				t.Log("Record the failed serving check as Aborting because the compensated topology is already fully fenced")
				result, err := coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
				assert.Equal(t, tt.storedFailure, result.Operation.Failure)
				assert.Equal(t, cleanupReplicas, abortCleanupReplicaIDs(*result.Operation))
				assert.True(t, servingVerificationTopologiesEqual(
					compensationTopology,
					*result.Operation.CompensationTopology,
				))
				assert.True(t, result.OperationChanged)
				require.NoError(t, validateOperation(*result.Operation))
				input.Operation = result.Operation

				t.Log("Restart without producing an invalid Failed operation or losing cleanup state")
				coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
				result, err = coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
				assert.Equal(t, tt.storedFailure, result.Operation.Failure)
				assert.Equal(t, cleanupReplicas, abortCleanupReplicaIDs(*result.Operation))
				assert.True(t, servingVerificationTopologiesEqual(
					compensationTopology,
					*result.Operation.CompensationTopology,
				))
				assert.False(t, result.OperationChanged)
				assert.Empty(t, traffic.withdrawRequests)
				assert.Len(t, verifier.observeCalls, 2)
				assert.Empty(t, externalMutations)
			})
		}
	}
}

func TestWorkflowCoordinatorFailsClosedWhenTopologyChangesDuringCompensation(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{externalMutationHistory: &externalMutations}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(2),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{externalMutationHistory: &externalMutations}
	operation := &Operation{
		ID:                   workflowTestOperationID,
		Attempt:              1,
		PlanID:               "plan-1",
		Intent:               OperationIntentShrink,
		Capability:           testOperationCapability(OperationShapePlannedHighRankSuffixShrink),
		SpecGeneration:       2,
		BaseTopology:         workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:       1,
		NominatedReplicas:    []ReplicaID{"replica-1"},
		Phase:                OperationPhaseAborting,
		CompensationTopology: topologyPointer(workflowTopology(1, "replica-0", "replica-1")),
		StartedAt:            workflowTestTime.Add(-time.Minute),
		LastTransitionTime:   workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "AtomicRequestRejected",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 1,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
	fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, membership.topology)
	t.Log("Fence the changed topology before persisting a compensation-drift outcome")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Empty(t, result.Operation.CleanupReplicaSlots)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, FailureClassificationTerminal, result.Operation.Failure.Classification)
	assert.Equal(t, "AtomicRequestRejected", result.Operation.Failure.Reason)
	assert.False(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, traffic.withdrawRequests)
	require.True(t, result.TrafficStateChanged)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        fenceOperationID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           fencedIncarnations,
	}}, traffic.withdrawRequests)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

	t.Log("Persist the fail-closed compensation drift only after observing the exact fence")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw, 1, fenceOperationID, membership.topology.Generation,
		nil, fencedIncarnations,
	)
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, "CompensationTopologyChanged", result.Operation.Failure.Reason)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, capacity.releaseCalls)
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
			traffic: workflowTrafficSnapshotWithCommand(
				TrafficActionAdmit, 1, workflowTestOperationID, 2,
				workflowReplicaIncarnations("replica-0", "replica-1"), nil,
			),
			wantCapacityRequest: []CapacityRequest{{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 2,
				TargetReplicas:     2,
				RequiredReplicas:   workflowRequiredReplicaAllocations("replica-0", "replica-1"),
			}},
		},
		{
			name:              "lost survivor admission",
			capacityAvailable: true,
			traffic: workflowTrafficSnapshotWithCommand(
				TrafficActionAdmit, 1, workflowTestOperationID, 2,
				workflowReplicaIncarnations("replica-0"), nil,
			),
			wantTrafficRequest: []TrafficRequest{{
				Revision:           2,
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 2,
				Replicas:           workflowReplicaIncarnations("replica-0", "replica-1"),
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
					FencedReplicaSlots: workflowReplicaSlotBindings("replica-2", "replica-3"),
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
				ID:                         workflowTestOperationID,
				Attempt:                    1,
				Intent:                     OperationIntentGrow,
				Capability:                 testOperationCapability(OperationShapeFreshGrowth),
				SpecGeneration:             2,
				BaseTopology:               workflowTopology(2, "replica-0", "replica-1"),
				TargetReplicas:             4,
				JoiningReplicas:            workflowReplicaIncarnations("replica-2", "replica-3"),
				CleanupReplicaSlots:        workflowReplicaSlotBindings("replica-2", "replica-3"),
				CapacityTargetReplicas:     2,
				CapacityTopologyGeneration: 2,
				CapacityTargetApplied:      true,
				Phase:                      OperationPhaseAborted,
				CompensationTopology:       topologyPointer(workflowTopology(2, "replica-0", "replica-1")),
				StartedAt:                  workflowTestTime.Add(-time.Minute),
				LastTransitionTime:         workflowTestTime,
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "AtomicRequestRejected",
				},
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 2,
				Operation:       operation,
			}

			t.Log("Restart from a long-lived Aborted audit record after serving state regresses")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			if !tt.capacityAvailable {
				fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
				fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, membership.topology)
				assert.Empty(t, traffic.withdrawRequests)
				result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
				assert.Equal(t, []TrafficRequest{{
					Revision:           2,
					OperationID:        fenceOperationID,
					TopologyGeneration: membership.topology.Generation,
					Replicas:           fencedIncarnations,
				}}, traffic.withdrawRequests)
				assert.Empty(t, capacity.ensureCalls)

				t.Log("Repair readiness only after the exact active-incarnation fence is observable")
				traffic.snapshot = workflowTrafficSnapshotWithCommand(
					TrafficActionWithdraw, 2, fenceOperationID, membership.topology.Generation,
					nil, fencedIncarnations,
				)
				coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
				result, err = coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				assert.Equal(t, tt.wantCapacityRequest, capacity.ensureCalls)
				assert.Len(t, externalMutations, 2)
			} else {
				assert.Empty(t, traffic.admitRequests)
				result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
				assert.Equal(t, tt.wantTrafficRequest, traffic.admitRequests)
				assert.Len(t, externalMutations, 1)
			}
			assert.Empty(t, membership.submitCalls)
		})
	}
}

func TestWorkflowCoordinatorCompensatesPreparedShrinkAfterExactRequestPreflightRejection(t *testing.T) {
	tests := []struct {
		name        string
		phase       OperationPhase
		failure     *OperationFailure
		wantAttempt int32
	}{
		{
			name:        "pending shrink",
			phase:       OperationPhasePending,
			wantAttempt: 1,
		},
		{
			name:        "retryable failed shrink",
			phase:       OperationPhaseFailed,
			wantAttempt: 2,
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
			membership := &workflowMembershipAdapter{
				validateRequestErr: fmt.Errorf(
					"exact prepared request is no longer safe: %w",
					ErrMembershipOperationUnsupported,
				),
				topology:                workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3"),
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: workflowTrafficSnapshotWithCommand(
					TrafficActionWithdraw, 1, workflowTestOperationID, 1,
					workflowReplicaIncarnations("replica-0", "replica-1"),
					workflowReplicaIncarnations("replica-2", "replica-3"),
				),
				externalMutationHistory: &externalMutations,
			}
			operation := &Operation{
				ID:             workflowTestOperationID,
				Attempt:        1,
				PlanID:         "plan-1",
				Intent:         OperationIntentShrink,
				Capability:     testOperationCapability(OperationShapePlannedHighRankSuffixShrink),
				SpecGeneration: 2,
				BaseTopology: workflowTopology(
					1,
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				),
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

			t.Log("Reject the exact frozen request and persist compensation without submitting or restoring traffic")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.NotNil(t, result.Operation.Failure)
			assert.Equal(t, FailureClassificationTerminal, result.Operation.Failure.Classification)
			assert.Equal(t, "CapabilityLost", result.Operation.Failure.Reason)
			require.Len(t, membership.validateRequestCalls, 1)
			request := membership.validateRequestCalls[0]
			assert.Equal(t, operation.ID, request.ID)
			assert.Equal(t, tt.wantAttempt, request.Attempt)
			assert.Equal(t, operation.Capability, request.Capability)
			assert.True(t, servingVerificationTopologiesEqual(operation.BaseTopology, request.BaseTopology))
			assert.Equal(t, operation.TargetReplicas, request.TargetReplicas)
			assert.Equal(t, operation.NominatedReplicas, request.NominatedReplicas)
			assert.Zero(t, membership.observeCapabilityCalls)
			assert.Empty(t, membership.validatePlanCalls)
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
			assert.Empty(t, traffic.admitRequests)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			assert.Equal(t, []TrafficRequest{{
				Revision:           2,
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 1,
				Replicas: workflowReplicaIncarnations(
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				),
			}}, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)

			t.Log("Restart after compensated traffic converges and persist Aborted")
			traffic.snapshot = workflowTrafficSnapshotWithCommand(
				TrafficActionAdmit, 2, workflowTestOperationID, 1,
				workflowReplicaIncarnations(
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				),
				nil,
			)
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

func TestWorkflowCoordinatorDistinguishesSubmissionPreflightFromAmbiguousSubmitFailure(t *testing.T) {
	newFixture := func() (
		[]string,
		*workflowCapacityAdapter,
		*workflowMembershipAdapter,
		*workflowTrafficAdapter,
		*Operation,
	) {
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
			topology:                workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3"),
			externalMutationHistory: &externalMutations,
		}
		traffic := &workflowTrafficAdapter{
			snapshot: workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw, 1, workflowTestOperationID, 1,
				workflowReplicaIncarnations("replica-0", "replica-1"),
				workflowReplicaIncarnations("replica-2", "replica-3"),
			),
			externalMutationHistory: &externalMutations,
		}
		operation := &Operation{
			ID:             workflowTestOperationID,
			Attempt:        1,
			PlanID:         "plan-1",
			Intent:         OperationIntentShrink,
			Capability:     testOperationCapability(OperationShapePlannedHighRankSuffixShrink),
			SpecGeneration: 2,
			BaseTopology: workflowTopology(
				1,
				"replica-0",
				"replica-1",
				"replica-2",
				"replica-3",
			),
			TargetReplicas:     2,
			NominatedReplicas:  []ReplicaID{"replica-2", "replica-3"},
			Phase:              OperationPhaseSubmitting,
			StartedAt:          workflowTestTime.Add(-time.Minute),
			LastTransitionTime: workflowTestTime,
		}
		return externalMutations, capacity, membership, traffic, operation
	}

	t.Run("unsupported second preflight is safe to compensate", func(t *testing.T) {
		externalMutations, capacity, membership, traffic, operation := newFixture()
		membership.validateRequestErr = fmt.Errorf(
			"exact persisted request is no longer supported: %w",
			ErrMembershipOperationUnsupported,
		)
		coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

		result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
			GroupID:         "group-0",
			SpecGeneration:  2,
			DesiredReplicas: 2,
			Operation:       operation,
		})
		require.NoError(t, err)
		assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
		require.NotNil(t, result.Operation.CompensationTopology)
		assert.True(t, servingVerificationTopologiesEqual(
			membership.topology,
			*result.Operation.CompensationTopology,
		))
		require.NotNil(t, result.Operation.Failure)
		assert.Equal(t, "CapabilityLost", result.Operation.Failure.Reason)
		assert.True(t, result.OperationChanged)
		assert.Len(t, membership.observeOperationCalls, 1)
		assert.Len(t, membership.validateRequestCalls, 1)
		assert.Empty(t, membership.submitCalls)
		assert.Empty(t, externalMutations)
	})

	t.Run("unsupported submit failure remains ambiguous", func(t *testing.T) {
		externalMutations, capacity, membership, traffic, operation := newFixture()
		membership.submitErr = fmt.Errorf(
			"backend may have accepted the request: %w",
			ErrMembershipOperationUnsupported,
		)
		coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

		result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
			GroupID:         "group-0",
			SpecGeneration:  2,
			DesiredReplicas: 2,
			Operation:       operation,
		})
		require.ErrorContains(t, err, "submit membership operation")
		assert.ErrorIs(t, err, ErrMembershipOperationUnsupported)
		assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
		assert.Nil(t, result.Operation.CompensationTopology)
		assert.False(t, result.OperationChanged)
		assert.Len(t, membership.validateRequestCalls, 1)
		assert.Len(t, membership.submitCalls, 1)
		assert.Empty(t, externalMutations)
	})
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionAdmit, 1, workflowTestOperationID, 1,
			workflowReplicaIncarnations("replica-0", "replica-1"), nil,
		),
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
		StartedAt:          workflowTestTime.Add(-time.Minute),
		LastTransitionTime: workflowTestTime,
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
	assert.Empty(t, result.Operation.CleanupReplicaSlots)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart and durably freeze every non-active allocation as cleanup work")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, abortCleanupReplicaIDs(*result.Operation))
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
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-2", "replica-3"),
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
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, abortCleanupReplicaIDs(*result.Operation))
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorFencesUnlaunchedGrowthCapacityBeforeAborted(t *testing.T) {
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionAdmit, 1, workflowTestOperationID, 1,
			workflowReplicaIncarnations("replica-0", "replica-1"), nil,
		),
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                   workflowTestOperationID,
		Attempt:              1,
		Intent:               OperationIntentGrow,
		Capability:           testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:       2,
		BaseTopology:         workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:       4,
		JoiningReplicas:      workflowReplicaIncarnations("replica-2", "replica-3"),
		Phase:                OperationPhaseAborting,
		CompensationTopology: topologyPointer(workflowTopology(1, "replica-0", "replica-1")),
		StartedAt:            workflowTestTime.Add(-time.Minute),
		LastTransitionTime:   workflowTestTime,
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

	t.Log("Persist every planned joining slot even though no surplus allocation is currently visible")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Equal(t, workflowReplicaSlotBindings("replica-2", "replica-3"), result.Operation.CleanupReplicaSlots)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Persist a fence-only authorization and the absolute target barrier")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.False(t, result.Operation.CapacityTargetApplied)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, workflowTestReleaseID, result.ReleaseAuthorization.ID)
	assert.Equal(t, int32(2), result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, int64(1), result.ReleaseAuthorization.TopologyGeneration)
	assert.Equal(t, []AuthorizedReplica{
		{ReplicaID: "replica-2", SlotID: "slot-replica-2"},
		{ReplicaID: "replica-3", SlotID: "slot-replica-3"},
	}, result.ReleaseAuthorization.Replicas)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from durable authorization and apply the target barrier without concrete Pods")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, result.ReleaseAuthorization.Replicas, capacity.releaseCalls[0].Replicas)
	assert.Equal(t, int32(2), capacity.releaseCalls[0].TargetReplicas)
	assert.Equal(t, []string{"capacity.release:" + workflowTestReleaseID}, externalMutations)

	t.Log("Observe Applied and durably record the exact target and topology proof before replacement is allowed")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot.FencedReplicaSlots = workflowReplicaSlotBindings("replica-2", "replica-3")
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
			FencedReplicaSlots: workflowReplicaSlotBindings("replica-2", "replica-3"),
		},
		externalMutationHistory: &externalMutations,
	}
	membership := &workflowMembershipAdapter{
		topology:                workflowTopology(1, "replica-0", "replica-1"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionAdmit, 1, workflowTestOperationID, 1,
			workflowReplicaIncarnations("replica-0", "replica-1"), nil,
		),
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                         workflowTestOperationID,
		Attempt:                    1,
		Intent:                     OperationIntentGrow,
		Capability:                 testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:             2,
		BaseTopology:               workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:             4,
		JoiningReplicas:            workflowReplicaIncarnations("replica-2", "replica-3"),
		CleanupReplicaSlots:        workflowReplicaSlotBindings("replica-2", "replica-3"),
		CapacityTargetReplicas:     2,
		CapacityTopologyGeneration: 1,
		CapacityTargetApplied:      true,
		Phase:                      OperationPhaseAborted,
		CompensationTopology:       topologyPointer(workflowTopology(1, "replica-0", "replica-1")),
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
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3", "replica-4"}, abortCleanupReplicaIDs(*result.Operation))
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
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-2", "replica-3", "replica-4"),
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
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3", "replica-4"}, abortCleanupReplicaIDs(*result.Operation))
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"capacity.release:" + lateReleaseID}, externalMutations)
}

func TestWorkflowCoordinatorReleasesSurplusBeforeAbortingForMissingAuthoritativeReplica(t *testing.T) {
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionAdmit, 1, workflowTestOperationID, 2,
			workflowReplicaIncarnations("replica-a"), nil,
		),
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                   workflowTestOperationID,
		Attempt:              1,
		Intent:               OperationIntentGrow,
		Capability:           testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:       2,
		BaseTopology:         workflowTopology(2, "replica-a", "replica-b"),
		TargetReplicas:       3,
		JoiningReplicas:      workflowReplicaIncarnations("replica-x"),
		Phase:                OperationPhaseAborting,
		CompensationTopology: topologyPointer(workflowTopology(2, "replica-a", "replica-b")),
		StartedAt:            workflowTestTime.Add(-time.Minute),
		LastTransitionTime:   workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "AtomicRequestRejected",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
	fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, membership.topology)

	t.Log("Fence the active regression before identifying surplus capacity or repairing survivors")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Empty(t, result.Operation.CleanupReplicaSlots)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           2,
		OperationID:        fenceOperationID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           fencedIncarnations,
	}}, traffic.withdrawRequests)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw, 2, fenceOperationID, membership.topology.Generation,
		nil, fencedIncarnations,
	)

	t.Log("Durably identify the non-authoritative allocation after the fence is observable")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []ReplicaID{"replica-x"}, abortCleanupReplicaIDs(*result.Operation))
	assert.Equal(t, workflowReplicaSlotBindings("replica-x"), result.Operation.CleanupReplicaSlots)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)
	input.Operation = result.Operation
	capacity.snapshot.Allocations = []ReplicaAllocation{
		workflowReplicaAllocation("replica-a", "pod-a"),
	}

	t.Log("Preserve the disappeared surplus allocation's historical slot in the release authorization")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.Equal(t, int32(2), result.ReleaseAuthorization.TargetReplicas)
	assert.Equal(t, int64(2), result.ReleaseAuthorization.TopologyGeneration)
	require.Len(t, result.ReleaseAuthorization.Replicas, 1)
	assert.Equal(t, ReplicaID("replica-x"), result.ReleaseAuthorization.Replicas[0].ReplicaID)
	assert.Equal(t, CapacitySlotID("slot-replica-x"), result.ReleaseAuthorization.Replicas[0].SlotID)
	assert.Empty(t, result.ReleaseAuthorization.Replicas[0].CapacityRefs)
	assert.Len(t, traffic.withdrawRequests, 1, "the never-active joiner requires no additional drain")
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)
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
	assert.Equal(t, []string{
		"traffic.withdraw:" + fenceOperationID,
		"capacity.release:" + workflowTestReleaseID,
	}, externalMutations)

	t.Log("Observe the surplus fenced and durably persist the exact target proof")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplied,
	}
	capacity.snapshot = CapacitySnapshot{
		Allocations:        []ReplicaAllocation{workflowReplicaAllocation("replica-a", "pod-a")},
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-x"),
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

	t.Log("Restart from the target proof and expose a separate recovery boundary")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Equal(t, []string{
		"traffic.withdraw:" + fenceOperationID,
		"capacity.release:" + workflowTestReleaseID,
	}, externalMutations)
	assert.Empty(t, membership.submitCalls)
}

func TestWorkflowCoordinatorRefusesExternallyChangedTopologyDuringAbortCleanup(t *testing.T) {
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionAdmit, 1, workflowTestOperationID, 1,
			workflowReplicaIncarnations("replica-0", "replica-1"), nil,
		),
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                   workflowTestOperationID,
		Attempt:              1,
		Intent:               OperationIntentGrow,
		Capability:           testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:       2,
		BaseTopology:         workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:       3,
		JoiningReplicas:      workflowReplicaIncarnations("replica-2"),
		Phase:                OperationPhaseAborting,
		CompensationTopology: topologyPointer(workflowTopology(1, "replica-0", "replica-1")),
		StartedAt:            workflowTestTime.Add(-time.Minute),
		LastTransitionTime:   workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "AtomicRequestRejected",
		},
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
	fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, membership.topology)
	t.Log("Fence an external transition before changing the durable compensation outcome")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Empty(t, result.Operation.CleanupReplicaSlots)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, FailureClassificationTerminal, result.Operation.Failure.Classification)
	assert.Equal(t, "AtomicRequestRejected", result.Operation.Failure.Reason)
	assert.False(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           2,
		OperationID:        fenceOperationID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           fencedIncarnations,
	}}, traffic.withdrawRequests)
	assert.Empty(t, capacity.releaseCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

	t.Log("Persist CompensationTopologyChanged only after the exact fence is observable")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw, 2, fenceOperationID, membership.topology.Generation,
		nil, fencedIncarnations,
	)
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, "CompensationTopologyChanged", result.Operation.Failure.Reason)
	assert.True(t, result.OperationChanged)
}

func TestWorkflowCoordinatorFencesCompensationTopologyDriftBeforeCapacityWork(t *testing.T) {
	for _, initialPhase := range []OperationPhase{OperationPhaseAborting, OperationPhaseAborted} {
		t.Run(string(initialPhase), func(t *testing.T) {
			externalMutations := make([]string, 0)
			currentTopology := workflowTopology(2, "replica-0")
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
					workflowReplicaAllocation("replica-0", "pod-0"),
				}},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                currentTopology,
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot:                TrafficSnapshot{Admitted: workflowReplicaIncarnations("replica-0")},
				externalMutationHistory: &externalMutations,
			}
			compensationTopology := workflowTopology(1, "replica-0", "replica-1")
			operation := &Operation{
				ID:                         workflowTestOperationID,
				Attempt:                    1,
				Intent:                     OperationIntentGrow,
				Capability:                 testOperationCapability(OperationShapeFreshGrowth),
				SpecGeneration:             2,
				BaseTopology:               cloneTopology(compensationTopology),
				TargetReplicas:             3,
				JoiningReplicas:            workflowReplicaIncarnations("replica-2"),
				Phase:                      initialPhase,
				CompensationTopology:       topologyPointer(compensationTopology),
				CapacityTargetReplicas:     1,
				CapacityTopologyGeneration: 2,
				CapacityTargetApplied:      true,
				StartedAt:                  workflowTestTime.Add(-time.Minute),
				LastTransitionTime:         workflowTestTime,
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "AtomicRequestRejected",
				},
			}
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 3,
				Operation:       operation,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
			fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, currentTopology)
			fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, currentTopology)

			t.Log("Fence the whole changed topology before persisting drift or performing capacity work")
			result, err := coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, initialPhase, result.Operation.Phase)
			require.NotNil(t, result.Operation.Failure)
			assert.Equal(t, FailureClassificationTerminal, result.Operation.Failure.Classification)
			assert.Equal(t, "AtomicRequestRejected", result.Operation.Failure.Reason)
			assert.False(t, result.OperationChanged)
			assert.Empty(t, traffic.withdrawRequests)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			assert.Equal(t, []TrafficRequest{{
				Revision:           1,
				OperationID:        fenceOperationID,
				TopologyGeneration: currentTopology.Generation,
				Replicas:           fencedIncarnations,
			}}, traffic.withdrawRequests)
			assert.Empty(t, capacity.ensureCalls)
			assert.Empty(t, capacity.releaseCalls)
			assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

			t.Log("Observe the exact whole-topology fence after restart")
			traffic.snapshot = workflowTrafficSnapshotWithCommand(
				TrafficActionWithdraw, 1, fenceOperationID, currentTopology.Generation,
				nil, fencedIncarnations,
			)
			coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
			result, err = coordinator.Reconcile(context.Background(), input)
			require.NoError(t, err)
			if initialPhase == OperationPhaseAborting {
				assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
				require.NotNil(t, result.Operation.Failure)
				assert.Equal(t, "CompensationTopologyChanged", result.Operation.Failure.Reason)
				assert.True(t, result.OperationChanged)
				input.Operation = result.Operation

				t.Log("Discover the missing old member as exact cleanup work on a later restart")
				coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
				result, err = coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
				assert.Equal(t, []ReplicaID{"replica-1", "replica-2"}, abortCleanupReplicaIDs(*result.Operation))
				assert.True(t, result.OperationChanged)
				input.Operation = result.Operation

				t.Log("Persist and execute a logical release for the missing old member")
				coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
				result, err = coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				require.NotNil(t, result.ReleaseAuthorization)
				input.ReleaseAuthorization = result.ReleaseAuthorization

				coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
				result, err = coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				require.Len(t, capacity.releaseCalls, 1)

				capacity.releaseObservation = CapacityReleaseObservation{
					ReleaseID: workflowTestReleaseID,
					Phase:     CapacityReleasePhaseApplied,
				}
				capacity.snapshot.FencedReplicaSlots = workflowReplicaSlotBindings("replica-1", "replica-2")
				coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
				result, err = coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				assert.Nil(t, result.ReleaseAuthorization)
				input.ReleaseAuthorization = nil
				input.Operation = result.Operation

				t.Log("Reach Aborted only after the cleanup fence and target proof are durable")
				coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
				result, err = coordinator.Reconcile(context.Background(), input)
				require.NoError(t, err)
				assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
				assert.True(t, result.OperationChanged)
			} else {
				assert.Equal(t, OperationPhaseAborted, result.Operation.Phase)
				assert.Equal(t, "AtomicRequestRejected", result.Operation.Failure.Reason)
				assert.False(t, result.OperationChanged)
			}
			assert.Len(t, traffic.withdrawRequests, 1)
			assert.Empty(t, capacity.ensureCalls)
			if initialPhase == OperationPhaseAborted {
				assert.Empty(t, capacity.releaseCalls)
			}
		})
	}
}

func TestWorkflowCoordinatorRefusesCleanupReplicaRejoiningDuringCompensation(t *testing.T) {
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
		snapshot: workflowTrafficSnapshotWithCommand(
			TrafficActionAdmit, 1, workflowTestOperationID, 2,
			workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"), nil,
		),
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                   workflowTestOperationID,
		Attempt:              1,
		Intent:               OperationIntentGrow,
		Capability:           testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:       2,
		BaseTopology:         workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:       4,
		JoiningReplicas:      workflowReplicaIncarnations("replica-2", "replica-3"),
		CleanupReplicaSlots:  workflowReplicaSlotBindings("replica-2", "replica-3"),
		Phase:                OperationPhaseAborting,
		CompensationTopology: topologyPointer(workflowTopology(1, "replica-0", "replica-1")),
		StartedAt:            workflowTestTime.Add(-time.Minute),
		LastTransitionTime:   workflowTestTime,
		Failure: &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "Rejected",
		},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
	fencedIncarnations := unverifiedTopologyDrainReplicas(*operation, membership.topology)
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	}

	t.Log("Fence the whole current topology before changing the durable compensation outcome")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, FailureClassificationTerminal, result.Operation.Failure.Classification)
	assert.Equal(t, "Rejected", result.Operation.Failure.Reason)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           2,
		OperationID:        fenceOperationID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           fencedIncarnations,
	}}, traffic.withdrawRequests)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, traffic.admitRequests)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

	t.Log("Persist the terminal topology-drift reason after the exact fence is observable")
	traffic.snapshot = workflowTrafficSnapshotWithCommand(
		TrafficActionWithdraw, 2, fenceOperationID, membership.topology.Generation,
		nil, fencedIncarnations,
	)
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, "CompensationTopologyChanged", result.Operation.Failure.Reason)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, capacity.releaseCalls)
}

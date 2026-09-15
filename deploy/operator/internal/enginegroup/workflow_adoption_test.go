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

func TestWorkflowCoordinatorFencesAdoptedTopologyBeforeRepairingCapacity(t *testing.T) {
	externalMutations := make([]string, 0)
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2")
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
		snapshot:                TrafficSnapshot{Admitted: topologyReplicaIncarnations(committedTopology)},
		externalMutationHistory: &externalMutations,
	}
	verifier := &workflowServingVerifier{externalMutationHistory: &externalMutations}
	baseTopology := workflowTopology(1, "replica-0", "replica-1", "replica-2")
	baseTopology.Replicas[2].Incarnation.CapacityRefs[0].UID = "old-uid-pod-2"
	baseTopology.Replicas[2].Incarnation.RuntimeID = "old-runtime-replica-2"
	operation := &Operation{
		ID:      workflowTestOperationID,
		Attempt: 1,
		PlanID:  "adopted-replacement-plan",
		Intent:  OperationIntentRecover,
		Capability: ResolvedOperationCapability{
			Shape:                   OperationShapeFixedSlotReplacement,
			TrafficRequirement:      ReconfigurationTrafficKeepServing,
			VerificationRequirement: ServingVerificationRequired,
		},
		SpecGeneration:             2,
		BaseTopology:               baseTopology,
		TargetReplicas:             3,
		TargetMembership:           cloneReplicaMemberships(committedTopology.Replicas),
		Phase:                      OperationPhaseCommitted,
		CommittedTopology:          topologyPointer(committedTopology),
		ServingVerificationAttempt: 1,
		ServingVerificationTarget:  topologyPointer(committedTopology),
		Adopted:                    true,
		StartedAt:                  workflowTestTime.Add(-time.Minute),
		LastTransitionTime:         workflowTestTime,
	}
	fencedReplicas := unverifiedTopologyDrainReplicas(*operation, committedTopology)
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Operation:       operation,
	}
	coordinator := newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)

	t.Log("Fence the old and replacement incarnations before repairing unavailable adopted capacity")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.withdrawRequests)
	assert.Equal(t, workflowAcceptedTrafficCommand(
		TrafficActionWithdraw, 1,
		unverifiedTopologyTrafficOperationID(*operation, committedTopology),
		committedTopology.Generation,
		fencedReplicas,
	), result.TrafficCommand)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        unverifiedTopologyTrafficOperationID(*operation, committedTopology),
		TopologyGeneration: committedTopology.Generation,
		Replicas:           fencedReplicas,
	}}, traffic.withdrawRequests)
	assert.Empty(t, capacity.ensureCalls)
	assert.Empty(t, verifier.observeCalls)
	fenceOperationID := unverifiedTopologyTrafficOperationID(*operation, committedTopology)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)
	input.Operation = result.Operation

	t.Log("Only after restart with the exact multi-incarnation fence may verification invalidation become durable")
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw, 1, fenceOperationID, committedTopology.Generation, fencedReplicas,
		),
		Drained: fencedReplicas,
	}
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, CapacityRecoveryPhaseRepairing, result.Operation.CapacityRecoveryPhase)
	assert.Equal(t, int32(2), result.Operation.ServingVerificationAttempt)
	assert.Nil(t, result.Operation.ServingVerificationProof)
	assert.Empty(t, capacity.ensureCalls)
	assert.Len(t, traffic.withdrawRequests, 1)
	assert.Empty(t, verifier.observeCalls)
	input.Operation = result.Operation

	t.Log("A later restart may repair capacity under the fresh verification identity")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.Equal(t, []CapacityRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: committedTopology.Generation,
		TargetReplicas:     3,
		RequiredReplicas:   workflowRequiredReplicaAllocations("replica-0", "replica-1", "replica-2"),
	}}, capacity.ensureCalls)
	assert.Len(t, traffic.withdrawRequests, 1)
	assert.Empty(t, verifier.observeCalls)
	assert.Equal(t, []string{
		"traffic.withdraw:" + fenceOperationID,
		"capacity.ensure:3",
	}, externalMutations)
}

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
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2", "replica-3"),
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
	verifier := &workflowServingVerifier{proof: ServingVerificationProof{
		OperationID:         workflowTestOperationID,
		Attempt:             1,
		VerificationAttempt: 1,
		Topology:            workflowTopology(3, "replica-0", "replica-1", "replica-2"),
		Phase:               ServingVerificationPhasePassed,
	}}

	previousOperation := input.Operation
	preAdoptionFenceID := unverifiedTopologyTrafficOperationID(*previousOperation, membership.topology)
	preAdoptionDrains := unverifiedTopologyDrainReplicas(*previousOperation, membership.topology)

	t.Log("Fence both sides of the externally changed topology before persisting its adoption")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, previousOperation, result.Operation)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        preAdoptionFenceID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           preAdoptionDrains,
	}}, traffic.withdrawRequests)
	assert.Empty(t, membership.submitCalls)
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw, 1, preAdoptionFenceID, membership.topology.Generation, preAdoptionDrains,
		),
		Drained: preAdoptionDrains,
	}

	t.Log("Persist the adopted recovery only after the pre-adoption fence is observable after restart")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, workflowTestOperationID, result.Operation.ID)
	assert.Equal(t, plan.ID, result.Operation.PlanID)
	assert.Equal(t, OperationIntentRecover, result.Operation.Intent)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int64(2), result.Operation.BaseTopology.Generation)
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, int64(3), result.Operation.CommittedTopology.Generation)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1", "replica-2", "replica-3"}, topologyReplicaIDs(result.Operation.BaseTopology))
	assert.Equal(t, []ReplicaID{"replica-3"}, result.Operation.NominatedReplicas)
	assert.True(t, result.Operation.Adopted)
	assert.False(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	require.Len(t, traffic.withdrawRequests, 1)
	assert.Equal(t, []string{"traffic.withdraw:" + preAdoptionFenceID}, externalMutations)
	require.Len(t, membership.validateObservedTransitionCalls, 2)
	validatedTransition := membership.validateObservedTransitionCalls[len(membership.validateObservedTransitionCalls)-1]
	assert.True(t, servingVerificationTopologiesEqual(
		workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3"),
		validatedTransition.PreviousTopology,
	))
	assert.True(t, servingVerificationTopologiesEqual(membership.topology, validatedTransition.ObservedTopology))
	assert.Equal(t, normalizePlan(*plan), validatedTransition.Plan)
	input.Operation = result.Operation

	t.Log("Restart from the durable adopted record and verify behind the existing whole-topology fence")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Len(t, traffic.withdrawCalls, 1)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, [][]ReplicaID{{"replica-0", "replica-1", "replica-2", "replica-3"}}, traffic.withdrawCalls)
	assert.Equal(t, []TrafficRequest{
		{
			Revision:           1,
			OperationID:        preAdoptionFenceID,
			TopologyGeneration: 3,
			Replicas:           preAdoptionDrains,
		},
	}, traffic.withdrawRequests)
	require.NotNil(t, result.Operation.ServingVerificationProof)
	assert.Len(t, verifier.observeCalls, 1)
	assert.Empty(t, verifier.ensureRequests)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + preAdoptionFenceID}, externalMutations)
	input.Operation = result.Operation

	t.Log("Restart from the verified adopted record and explicitly admit the survivor topology")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Len(t, traffic.admitCalls, 0)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1", "replica-2"}, traffic.admitCalls[len(traffic.admitCalls)-1])
	assert.Nil(t, result.ReleaseAuthorization)
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionAdmit,
			2,
			workflowTestOperationID,
			membership.topology.Generation,
			workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		),
		Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		Drained:  workflowReplicaIncarnations("replica-3"),
	}

	t.Log("Persist UID-bound release authorization only after verified survivor admission is observable")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
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
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-3"),
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

	t.Log("Persist post-commit completion after release because verified survivor traffic is already serving")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1", "replica-2"}, traffic.admitCalls[len(traffic.admitCalls)-1])
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
			LatestCommand: workflowAcceptedTrafficObservation(
				TrafficActionWithdraw,
				1,
				"old-shrink-operation",
				2,
				workflowReplicaIncarnations("replica-2"),
			),
			Admitted: workflowReplicaIncarnations("replica-0"),
			Drained:  workflowReplicaIncarnations("replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                 "old-shrink-operation",
		Attempt:            1,
		PlanID:             "old-shrink-plan",
		Intent:             OperationIntentShrink,
		Capability:         testOperationCapability(OperationShapePlannedHighRankSuffixShrink),
		SpecGeneration:     2,
		BaseTopology:       workflowTopology(1, "replica-0", "replica-1", "replica-2"),
		TargetReplicas:     2,
		NominatedReplicas:  []ReplicaID{"replica-2"},
		Phase:              OperationPhaseUnknown,
		CommittedTopology:  topologyPointer(workflowTopology(2, "replica-0", "replica-1")),
		StartedAt:          workflowTestTime.Add(-2 * time.Minute),
		LastTransitionTime: workflowTestTime.Add(-time.Minute),
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

	t.Log("Fence the later survivor topology before finishing the old reduction")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, operation, result.Operation)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           2,
		OperationID:        unverifiedTopologyTrafficOperationID(*operation, membership.topology),
		TopologyGeneration: membership.topology.Generation,
		Replicas:           unverifiedTopologyDrainReplicas(*operation, membership.topology),
	}}, traffic.withdrawRequests)
	assert.Empty(t, membership.submitCalls)

	t.Log("Preserve the old operation and authorize only its unfinished retired allocation after the fence is durable")
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw,
			2,
			unverifiedTopologyTrafficOperationID(*operation, membership.topology),
			membership.topology.Generation,
			unverifiedTopologyDrainReplicas(*operation, membership.topology),
		),
		Drained: unverifiedTopologyDrainReplicas(*operation, membership.topology),
	}
	result, err = coordinator.Reconcile(context.Background(), input)
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
	capacity.snapshot.FencedReplicaSlots = workflowReplicaSlotBindings("replica-2")
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

	t.Log("Adopt the newly missing survivor after the old exact release is durably complete")
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
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
		},
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                 "incomplete-operation",
		Attempt:            1,
		Intent:             OperationIntentGrow,
		Capability:         testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:     2,
		BaseTopology:       workflowTopology(1, "replica-0", "replica-1"),
		TargetReplicas:     4,
		JoiningReplicas:    workflowReplicaIncarnations("replica-2", "replica-3"),
		Phase:              OperationPhaseUnknown,
		CommittedTopology:  topologyPointer(workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3")),
		StartedAt:          workflowTestTime.Add(-2 * time.Minute),
		LastTransitionTime: workflowTestTime.Add(-time.Minute),
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
	verifier := &workflowServingVerifier{proof: ServingVerificationProof{
		OperationID:         workflowTestOperationID,
		Attempt:             1,
		VerificationAttempt: 1,
		Topology:            workflowTopology(3, "replica-0", "replica-1", "replica-2"),
		Phase:               ServingVerificationPhasePassed,
	}}

	preAdoptionFenceID := unverifiedTopologyTrafficOperationID(*operation, membership.topology)
	preAdoptionDrains := unverifiedTopologyDrainReplicas(*operation, membership.topology)
	t.Log("Fence the incomplete growth and observed survivor incarnations before adoption")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, operation, result.Operation)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        preAdoptionFenceID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           preAdoptionDrains,
	}}, traffic.withdrawRequests)
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw, 1, preAdoptionFenceID, membership.topology.Generation, preAdoptionDrains,
		),
		Drained: preAdoptionDrains,
	}

	t.Log("Adopt the authoritative survivor topology after observing the durable fence")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.True(t, result.Operation.Adopted)
	assert.Equal(t, plan.ID, result.Operation.PlanID)
	assert.Equal(t, []ReplicaID{"replica-3"}, result.Operation.NominatedReplicas)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Use the pre-adoption whole-topology fence to verify the adopted survivors")
	coordinator = newWorkflowCoordinatorWithVerifierForTest(capacity, membership, traffic, verifier)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Len(t, traffic.withdrawCalls, 1)
	assert.True(t, result.OperationChanged)
	assert.Equal(t, [][]ReplicaID{{"replica-0", "replica-1", "replica-2", "replica-3"}}, traffic.withdrawCalls)
	require.NotNil(t, result.Operation.ServingVerificationProof)
	assert.Len(t, verifier.observeCalls, 1)
	assert.Empty(t, verifier.ensureRequests)
	input.Operation = result.Operation

	t.Log("Explicitly admit the verified survivor topology before releasing the missing replica")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Empty(t, traffic.admitCalls)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1", "replica-2"}, traffic.admitCalls[len(traffic.admitCalls)-1])
	assert.Nil(t, result.ReleaseAuthorization)
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionAdmit,
			2,
			workflowTestOperationID,
			membership.topology.Generation,
			workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		),
		Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		Drained:  workflowReplicaIncarnations("replica-3"),
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
		FencedReplicaSlots: workflowReplicaSlotBindings("replica-3"),
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

	t.Log("Persist completion after release because verified survivor traffic is already serving")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1", "replica-2"}, traffic.admitCalls[len(traffic.admitCalls)-1])
	assert.True(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
}

func TestWorkflowCoordinatorAdoptsExactObservedNativeMemberRemap(t *testing.T) {
	externalMutations := make([]string, 0)
	observedTopology := workflowTopology(
		3,
		"replica-0",
		"replica-1",
		"replica-2",
		"replica-3",
	)
	observedTopology.Replicas[2].NativeMembers[0] = "native-replica-2-replacement"
	plan := &OperationPlan{
		ID:               "native-remap-plan",
		Intent:           OperationIntentRecover,
		TargetReplicas:   4,
		TargetMembership: cloneReplicaMemberships(observedTopology.Replicas),
	}
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
		topology:                observedTopology,
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2", "replica-3"),
		},
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
		Plan:            plan,
		Operation:       completedUnknownTopology(),
	}
	preAdoptionFenceID := unverifiedTopologyTrafficOperationID(*input.Operation, observedTopology)
	preAdoptionDrains := unverifiedTopologyDrainReplicas(*input.Operation, observedTopology)

	t.Log("Fence the old and remapped exact incarnations before adoption")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, input.Operation, result.Operation)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        preAdoptionFenceID,
		TopologyGeneration: observedTopology.Generation,
		Replicas:           preAdoptionDrains,
	}}, traffic.withdrawRequests)
	traffic.snapshot = TrafficSnapshot{
		LatestCommand: workflowAcceptedTrafficObservation(
			TrafficActionWithdraw, 1, preAdoptionFenceID, observedTopology.Generation, preAdoptionDrains,
		),
		Drained: preAdoptionDrains,
	}

	t.Log("Persist the exact native-member remap after observing the fence")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.True(t, result.Operation.Adopted)
	assert.Equal(t, OperationShapeNativeMemberRemapping, result.Operation.Capability.Shape)
	assert.Equal(t, int64(2), result.Operation.BaseTopology.Generation)
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, int64(3), result.Operation.CommittedTopology.Generation)
	assert.Empty(t, result.Operation.NominatedReplicas)
	assert.True(t, sameReplicaMemberships(observedTopology.Replicas, result.Operation.TargetMembership))
	require.Len(t, membership.validateObservedTransitionCalls, 2)
	validatedTransition := membership.validateObservedTransitionCalls[len(membership.validateObservedTransitionCalls)-1]
	assert.Equal(t, normalizePlan(*plan), validatedTransition.Plan)
	assert.True(t, servingVerificationTopologiesEqual(observedTopology, validatedTransition.ObservedTopology))
	assert.Empty(t, membership.submitCalls)
	assert.Len(t, traffic.withdrawCalls, 1)
	assert.Equal(t, []string{"traffic.withdraw:" + preAdoptionFenceID}, externalMutations)
}

func TestWorkflowCoordinatorRejectsObservedTransitionRequiringHistoricalQuiescence(t *testing.T) {
	externalMutations := make([]string, 0)
	quiescingCapability := ResolvedOperationCapability{
		Shape:                   OperationShapeSurvivorReduction,
		TrafficRequirement:      ReconfigurationTrafficQuiesceGroup,
		VerificationRequirement: ServingVerificationRequired,
	}
	membership := &workflowMembershipAdapter{
		topology:                             workflowTopology(3, "replica-0", "replica-1", "replica-2"),
		validateObservedTransitionCapability: &quiescingCapability,
		externalMutationHistory:              &externalMutations,
	}
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
			workflowReplicaAllocation("replica-3", "pod-3"),
		}},
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
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
	result, err := coordinator.Reconcile(context.Background(), input)
	require.ErrorContains(t, err, "historical whole-group quiescence")
	assert.ErrorIs(t, err, ErrMembershipOperationUnsupported)
	assert.Equal(t, completedUnknownTopology(), result.Operation)
	require.Len(t, membership.validateObservedTransitionCalls, 1)
	assert.Empty(t, membership.submitCalls)
	fenceOperationID := unverifiedTopologyTrafficOperationID(*result.Operation, membership.topology)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        fenceOperationID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           unverifiedTopologyDrainReplicas(*result.Operation, membership.topology),
	}}, traffic.withdrawRequests)
	assert.Empty(t, capacity.releaseCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

	traffic.snapshot.Admitted = nil
	traffic.snapshot.Drained = unverifiedTopologyDrainReplicas(*result.Operation, membership.topology)
	_, err = coordinator.Reconcile(context.Background(), input)
	require.ErrorContains(t, err, "historical whole-group quiescence")
}

func TestWorkflowCoordinatorRejectsUnsupportedObservedTransition(t *testing.T) {
	externalMutations := make([]string, 0)
	membership := &workflowMembershipAdapter{
		topology:                      workflowTopology(3, "replica-0", "replica-1", "replica-2"),
		validateObservedTransitionErr: ErrMembershipOperationUnsupported,
		externalMutationHistory:       &externalMutations,
	}
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
			workflowReplicaAllocation("replica-3", "pod-3"),
		}},
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: workflowReplicaIncarnations("replica-0", "replica-1", "replica-2"),
		},
		externalMutationHistory: &externalMutations,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
		Plan: &OperationPlan{
			ID:                "survivor-recovery-plan",
			Intent:            OperationIntentRecover,
			TargetReplicas:    3,
			NominatedReplicas: []ReplicaID{"replica-3"},
		},
		Operation: completedUnknownTopology(),
	}
	result, err := coordinator.Reconcile(context.Background(), input)
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrMembershipOperationUnsupported)
	assert.Equal(t, completedUnknownTopology(), result.Operation)
	require.Len(t, membership.validateObservedTransitionCalls, 1)
	assert.Empty(t, membership.submitCalls)
	fenceOperationID := unverifiedTopologyTrafficOperationID(*result.Operation, membership.topology)
	assert.Empty(t, traffic.withdrawRequests)
	result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
	assert.Equal(t, []TrafficRequest{{
		Revision:           1,
		OperationID:        fenceOperationID,
		TopologyGeneration: membership.topology.Generation,
		Replicas:           unverifiedTopologyDrainReplicas(*result.Operation, membership.topology),
	}}, traffic.withdrawRequests)
	assert.Empty(t, capacity.releaseCalls)
	assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

	t.Log("Retry preserves the validation error without replaying an acknowledged deterministic fence")
	traffic.snapshot.Admitted = nil
	traffic.snapshot.Drained = unverifiedTopologyDrainReplicas(*result.Operation, membership.topology)
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrMembershipOperationUnsupported)
	require.Len(t, traffic.withdrawRequests, 1)
	assert.Empty(t, membership.submitCalls)
	assert.Empty(t, capacity.releaseCalls)
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
					Admitted: workflowReplicaIncarnations("replica-0", "replica-1"),
				},
				externalMutationHistory: &externalMutations,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Reject an observed topology that cannot be correlated with the completed logical membership")
			input := ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  3,
				DesiredReplicas: 4,
				Plan:            &tt.plan,
				Operation:       completedUnknownTopology(),
			}
			result, err := coordinator.Reconcile(context.Background(), input)
			require.ErrorContains(t, err, tt.wantError)
			assert.Equal(t, completedUnknownTopology(), result.Operation)
			assert.Empty(t, membership.submitCalls)
			fenceOperationID := unverifiedTopologyTrafficOperationID(*result.Operation, tt.topology)
			assert.Empty(t, traffic.withdrawRequests)
			result = dispatchScheduledWorkflowTraffic(t, coordinator, &input, result)
			assert.Equal(t, []TrafficRequest{{
				Revision:           1,
				OperationID:        fenceOperationID,
				TopologyGeneration: tt.topology.Generation,
				Replicas:           unverifiedTopologyDrainReplicas(*result.Operation, tt.topology),
			}}, traffic.withdrawRequests)
			assert.Empty(t, capacity.releaseCalls)
			assert.Equal(t, []string{"traffic.withdraw:" + fenceOperationID}, externalMutations)

			traffic.snapshot.Admitted = nil
			traffic.snapshot.Drained = unverifiedTopologyDrainReplicas(*result.Operation, tt.topology)
			_, err = coordinator.Reconcile(context.Background(), input)
			require.ErrorContains(t, err, tt.wantError)
		})
	}
}

func completedUnknownTopology() *Operation {
	committedTopology := workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3")
	baseTopology := workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3")
	baseTopology.Replicas[0].NativeMembers = []NativeMemberID{"native-replica-0-before-remap"}
	return &Operation{
		ID:                 "completed-operation",
		Attempt:            1,
		PlanID:             "completed-plan",
		Intent:             OperationIntentRecover,
		Capability:         testOperationCapability(OperationShapeNativeMemberRemapping),
		SpecGeneration:     2,
		BaseTopology:       baseTopology,
		TargetReplicas:     4,
		TargetMembership:   cloneReplicaMemberships(committedTopology.Replicas),
		Phase:              OperationPhaseUnknown,
		CommittedTopology:  topologyPointer(committedTopology),
		PostCommitComplete: true,
		StartedAt:          workflowTestTime.Add(-2 * time.Minute),
		LastTransitionTime: workflowTestTime.Add(-time.Minute),
	}
}

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
	"fmt"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	workflowTestOperationID = "workflow-operation-1"
	workflowTestReleaseID   = "workflow-release-1"
)

var workflowTestTime = time.Date(2026, time.September, 14, 13, 0, 0, 0, time.UTC)

type workflowCapacityAdapter struct {
	snapshot                CapacitySnapshot
	releaseObservation      CapacityReleaseObservation
	observeCapacityCalls    int
	ensureCalls             []CapacityRequest
	observeReleaseCalls     []string
	releaseCalls            []ReleaseAuthorization
	releaseRequests         map[string]ReleaseAuthorization
	externalMutationHistory *[]string
}

func (f *workflowCapacityAdapter) ObserveCapacity(context.Context, GroupID) (CapacitySnapshot, error) {
	f.observeCapacityCalls++
	return cloneCapacitySnapshot(f.snapshot), nil
}

func (f *workflowCapacityAdapter) EnsureCapacity(_ context.Context, _ GroupID, request CapacityRequest) error {
	cloned := request
	cloned.RequiredReplicas = slices.Clone(request.RequiredReplicas)
	f.ensureCalls = append(f.ensureCalls, cloned)
	*f.externalMutationHistory = append(
		*f.externalMutationHistory,
		fmt.Sprintf("capacity.ensure:%d", request.TargetReplicas),
	)
	return nil
}

func (f *workflowCapacityAdapter) ObserveRelease(
	_ context.Context,
	_ GroupID,
	releaseID string,
) (CapacityReleaseObservation, error) {
	f.observeReleaseCalls = append(f.observeReleaseCalls, releaseID)
	if f.releaseObservation.Phase == "" {
		return CapacityReleaseObservation{Phase: CapacityReleasePhaseAbsent}, nil
	}
	return CapacityReleaseObservation{
		ReleaseID: f.releaseObservation.ReleaseID,
		Phase:     f.releaseObservation.Phase,
		Failure:   cloneFailure(f.releaseObservation.Failure),
	}, nil
}

func (f *workflowCapacityAdapter) Release(
	_ context.Context,
	_ GroupID,
	authorization ReleaseAuthorization,
) error {
	cloned := *cloneReleaseAuthorization(&authorization)
	f.releaseCalls = append(f.releaseCalls, cloned)

	// Model workload-manager idempotency while refusing one release ID for two UID-bound payloads.
	if f.releaseRequests == nil {
		f.releaseRequests = make(map[string]ReleaseAuthorization)
	}
	if existing, released := f.releaseRequests[authorization.ID]; released {
		if !reflect.DeepEqual(existing, authorization) {
			return fmt.Errorf("conflicting payload for capacity release %q", authorization.ID)
		}
	} else {
		f.releaseRequests[authorization.ID] = cloned
	}
	*f.externalMutationHistory = append(
		*f.externalMutationHistory,
		"capacity.release:"+authorization.ID,
	)
	return nil
}

type workflowMembershipAdapter struct {
	capabilities            *MembershipCapabilities
	capabilitiesErr         error
	observeCapabilityCalls  int
	topology                MembershipTopology
	operation               BackendOperation
	observeTopologyCalls    int
	observeOperationCalls   []operationAttempt
	submitCalls             []MembershipRequest
	externalMutationHistory *[]string
}

func (f *workflowMembershipAdapter) ObserveCapabilities(context.Context, GroupID) (MembershipCapabilities, error) {
	f.observeCapabilityCalls++
	if f.capabilitiesErr != nil {
		return MembershipCapabilities{}, f.capabilitiesErr
	}
	if f.capabilities == nil {
		return allTestMembershipCapabilities(), nil
	}
	return MembershipCapabilities{Intents: slices.Clone(f.capabilities.Intents)}, nil
}

func (f *workflowMembershipAdapter) ObserveTopology(context.Context, GroupID) (MembershipTopology, error) {
	f.observeTopologyCalls++
	return cloneTopology(f.topology), nil
}

func (f *workflowMembershipAdapter) ObserveOperation(
	_ context.Context,
	_ GroupID,
	operationID string,
	attempt int32,
) (BackendOperation, error) {
	f.observeOperationCalls = append(
		f.observeOperationCalls,
		operationAttempt{ID: operationID, Attempt: attempt},
	)
	if f.operation.Phase == "" {
		return BackendOperation{Phase: BackendOperationPhaseAbsent}, nil
	}
	observed := f.operation
	if observed.Attempt == 0 {
		observed.Attempt = attempt
	}
	observed.Failure = cloneFailure(f.operation.Failure)
	return observed, nil
}

func (f *workflowMembershipAdapter) SubmitOperation(
	_ context.Context,
	_ GroupID,
	request MembershipRequest,
) (BackendOperation, error) {
	clonedRequest := request
	clonedRequest.BaseReplicas = slices.Clone(request.BaseReplicas)
	clonedRequest.JoiningReplicas = slices.Clone(request.JoiningReplicas)
	clonedRequest.NominatedReplicas = slices.Clone(request.NominatedReplicas)
	f.submitCalls = append(f.submitCalls, clonedRequest)
	*f.externalMutationHistory = append(*f.externalMutationHistory, "membership.submit:"+request.ID)

	return BackendOperation{
		ID:             request.ID,
		Attempt:        request.Attempt,
		BackendID:      "backend-" + request.ID,
		TargetReplicas: request.TargetReplicas,
		Phase:          BackendOperationPhaseAccepted,
	}, nil
}

type workflowTrafficAdapter struct {
	snapshot                TrafficSnapshot
	unresolvedOperationID   string
	observeTrafficCalls     int
	admitRequests           []TrafficRequest
	withdrawRequests        []TrafficRequest
	admitCalls              [][]ReplicaID
	withdrawCalls           [][]ReplicaID
	externalMutationHistory *[]string
}

func (f *workflowTrafficAdapter) ObserveTraffic(context.Context, GroupID) (TrafficSnapshot, error) {
	f.observeTrafficCalls++
	return cloneTrafficSnapshot(f.snapshot), nil
}

func (f *workflowTrafficAdapter) Admit(
	_ context.Context,
	_ GroupID,
	request TrafficRequest,
) error {
	if f.unresolvedOperationID != "" && f.unresolvedOperationID != request.OperationID {
		return fmt.Errorf(
			"traffic operation %q is unresolved",
			f.unresolvedOperationID,
		)
	}
	clonedRequest := request
	clonedRequest.Replicas = slices.Clone(request.Replicas)
	f.admitRequests = append(f.admitRequests, clonedRequest)
	f.admitCalls = append(f.admitCalls, slices.Clone(request.Replicas))
	*f.externalMutationHistory = append(*f.externalMutationHistory, "traffic.admit:"+request.OperationID)
	return nil
}

func (f *workflowTrafficAdapter) Withdraw(
	_ context.Context,
	_ GroupID,
	request TrafficRequest,
) error {
	if f.unresolvedOperationID != "" && f.unresolvedOperationID != request.OperationID {
		return fmt.Errorf(
			"traffic operation %q is unresolved",
			f.unresolvedOperationID,
		)
	}
	clonedRequest := request
	clonedRequest.Replicas = slices.Clone(request.Replicas)
	f.withdrawRequests = append(f.withdrawRequests, clonedRequest)
	f.withdrawCalls = append(f.withdrawCalls, slices.Clone(request.Replicas))
	*f.externalMutationHistory = append(*f.externalMutationHistory, "traffic.withdraw:"+request.OperationID)
	return nil
}

func TestWorkflowCoordinatorOrdersGrowthAcrossDurableRestartBoundaries(t *testing.T) {
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
			Admitted: []ReplicaID{"replica-0", "replica-1"},
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist the planned membership operation before allocating capacity")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Allocate the absolute target while membership remains Pending")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []CapacityRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 1,
		TargetReplicas:     4,
		RequiredReplicas:   []ReplicaID{"replica-0", "replica-1"},
	}}, capacity.ensureCalls)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.Equal(t, []string{"capacity.ensure:4"}, externalMutations)

	t.Log("Observe available capacity after restart and persist exact joiners before submission")
	capacity.snapshot = CapacitySnapshot{Allocations: []ReplicaAllocation{
		workflowReplicaAllocation("replica-0", "pod-0"),
		workflowReplicaAllocation("replica-1", "pod-1"),
		workflowReplicaAllocation("replica-2", "pod-2"),
		workflowReplicaAllocation("replica-3", "pod-3"),
	}}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, result.Operation.JoiningReplicas)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Restart from the durable Submitting marker and submit the exact frozen request")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	require.Len(t, membership.submitCalls, 1)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, membership.submitCalls[0].JoiningReplicas)
	assert.Empty(t, traffic.admitCalls)
	input.Operation = result.Operation

	t.Log("Persist the committed topology before admitting its joining replicas")
	membership.topology = workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3")
	membership.operation = BackendOperation{
		ID:                          workflowTestOperationID,
		BackendID:                   "backend-" + workflowTestOperationID,
		TargetReplicas:              4,
		Phase:                       BackendOperationPhaseCommitted,
		CommittedTopologyGeneration: 2,
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, traffic.admitCalls)
	input.Operation = result.Operation

	t.Log("Restart from committed state and admit only the exact joining replicas")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.OperationChanged)
	assert.Equal(t, [][]ReplicaID{{"replica-2", "replica-3"}}, traffic.admitCalls)
	assert.Equal(t, []TrafficRequest{{
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		Replicas:           []ReplicaID{"replica-2", "replica-3"},
	}}, traffic.admitRequests)
	assert.Equal(t, []string{
		"capacity.ensure:4",
		"membership.submit:" + workflowTestOperationID,
		"traffic.admit:" + workflowTestOperationID,
	}, externalMutations)
}

func TestWorkflowCoordinatorRepairsCommittedJoiningCapacityBeforeTrafficAdmission(t *testing.T) {
	tests := []struct {
		name               string
		joinerMissing      bool
		joinerAvailability ReplicaAvailability
		fencedReplicas     []ReplicaID
	}{
		{
			name:          "missing joining allocation",
			joinerMissing: true,
		},
		{
			name:               "unavailable joining allocation",
			joinerAvailability: ReplicaAvailabilityUnavailable,
		},
		{
			name:               "fenced joining allocation",
			joinerAvailability: ReplicaAvailabilityAvailable,
			fencedReplicas:     []ReplicaID{"replica-3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			allocations := []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				workflowReplicaAllocation("replica-1", "pod-1"),
				workflowReplicaAllocation("replica-2", "pod-2"),
			}
			if !tt.joinerMissing {
				joiner := workflowReplicaAllocation("replica-3", "pod-3")
				joiner.Availability = tt.joinerAvailability
				allocations = append(allocations, joiner)
			}
			capacity := &workflowCapacityAdapter{
				snapshot: CapacitySnapshot{
					Allocations:    allocations,
					FencedReplicas: slices.Clone(tt.fencedReplicas),
				},
				externalMutationHistory: &externalMutations,
			}
			membership := &workflowMembershipAdapter{
				topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3"),
				externalMutationHistory: &externalMutations,
			}
			traffic := &workflowTrafficAdapter{
				snapshot: TrafficSnapshot{
					Admitted: []ReplicaID{"replica-0", "replica-1"},
				},
				externalMutationHistory: &externalMutations,
			}
			operation := &Operation{
				ID:                          workflowTestOperationID,
				Attempt:                     1,
				Intent:                      OperationIntentGrow,
				SpecGeneration:              2,
				BaseTopologyGeneration:      1,
				BaseReplicas:                []ReplicaID{"replica-0", "replica-1"},
				TargetReplicas:              4,
				JoiningReplicas:             []ReplicaID{"replica-2", "replica-3"},
				Phase:                       OperationPhaseCommitted,
				CommittedTopologyGeneration: 2,
				StartedAt:                   workflowTestTime.Add(-time.Minute),
				LastTransitionTime:          workflowTestTime,
			}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Repair every authoritative allocation before admitting committed joining replicas")
			result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 4,
				Operation:       operation,
			})
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
			assert.False(t, result.OperationChanged)
			assert.Equal(t, []CapacityRequest{{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 2,
				TargetReplicas:     4,
				RequiredReplicas: []ReplicaID{
					"replica-0",
					"replica-1",
					"replica-2",
					"replica-3",
				},
			}}, capacity.ensureCalls)
			assert.Empty(t, traffic.admitRequests)
			assert.Empty(t, membership.submitCalls)
			assert.Equal(t, []string{"capacity.ensure:4"}, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorRestartObservesSubmittedOperationBeforeReplay(t *testing.T) {
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
		topology: workflowTopology(1, "replica-0", "replica-1"),
		operation: BackendOperation{
			ID:             workflowTestOperationID,
			BackendID:      "backend-" + workflowTestOperationID,
			TargetReplicas: 4,
			Phase:          BackendOperationPhaseCommitting,
		},
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: []ReplicaID{"replica-0", "replica-1"},
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
		Phase:                  OperationPhaseSubmitting,
		StartedAt:              workflowTestTime.Add(-time.Minute),
		LastTransitionTime:     workflowTestTime,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Recover backend progress after restart before deciding whether the durable request needs replay")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseCommitting, result.Operation.Phase)
	assert.Equal(t, "backend-"+workflowTestOperationID, result.Operation.BackendOperationID)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls, "an observed operation must not be submitted again")
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorFailsClosedOnUnresolvedTrafficOperation(t *testing.T) {
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
		topology:                workflowTopology(2, "replica-0", "replica-1", "replica-2", "replica-3"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			OperationID: "foreign-operation",
			Admitted:    []ReplicaID{"replica-0", "replica-1"},
		},
		unresolvedOperationID:   "foreign-operation",
		externalMutationHistory: &externalMutations,
	}
	operation := &Operation{
		ID:                          workflowTestOperationID,
		Attempt:                     1,
		Intent:                      OperationIntentGrow,
		SpecGeneration:              2,
		BaseTopologyGeneration:      1,
		BaseReplicas:                []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:              4,
		JoiningReplicas:             []ReplicaID{"replica-2", "replica-3"},
		Phase:                       OperationPhaseCommitted,
		CommittedTopologyGeneration: 2,
		StartedAt:                   workflowTestTime.Add(-time.Minute),
		LastTransitionTime:          workflowTestTime,
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Refuse to overwrite a different unresolved traffic mutation after membership commits")
	_, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.ErrorContains(t, err, "foreign-operation")
	assert.Empty(t, traffic.admitCalls)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorRejectsInvalidDurableStateBeforeAdapterAccess(t *testing.T) {
	tests := []struct {
		name  string
		input ReconcileInput
	}{
		{
			name: "empty group ID",
			input: ReconcileInput{
				SpecGeneration:  1,
				DesiredReplicas: 1,
			},
		},
		{
			name: "intent contradicts operation geometry",
			input: ReconcileInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation: &Operation{
					ID:                          workflowTestOperationID,
					Attempt:                     1,
					PlanID:                      "plan-1",
					Intent:                      OperationIntentShrink,
					SpecGeneration:              1,
					BaseTopologyGeneration:      1,
					BaseReplicas:                []ReplicaID{"replica-0", "replica-1"},
					TargetReplicas:              4,
					JoiningReplicas:             []ReplicaID{"replica-2", "replica-3"},
					Phase:                       OperationPhaseCommitted,
					CommittedTopologyGeneration: 2,
					StartedAt:                   workflowTestTime.Add(-time.Minute),
					LastTransitionTime:          workflowTestTime,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			externalMutations := make([]string, 0)
			capacity := &workflowCapacityAdapter{externalMutationHistory: &externalMutations}
			membership := &workflowMembershipAdapter{externalMutationHistory: &externalMutations}
			traffic := &workflowTrafficAdapter{externalMutationHistory: &externalMutations}
			coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

			t.Log("Reject malformed desired or durable state before reading or mutating any adapter")
			_, err := coordinator.Reconcile(context.Background(), tt.input)
			require.Error(t, err)
			assert.Zero(t, capacity.observeCapacityCalls)
			assert.Zero(t, membership.observeTopologyCalls)
			assert.Zero(t, traffic.observeTrafficCalls)
			assert.Empty(t, externalMutations)
		})
	}
}

func TestWorkflowCoordinatorOrdersSelectedShrinkAndExactRelease(t *testing.T) {
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
	capabilities := allTestMembershipCapabilities()
	membership := &workflowMembershipAdapter{
		capabilities:            &capabilities,
		topology:                workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: []ReplicaID{"replica-0", "replica-1", "replica-2", "replica-3"},
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Plan: &OperationPlan{
			ID:                "plan-1",
			Intent:            OperationIntentShrink,
			TargetReplicas:    2,
			NominatedReplicas: []ReplicaID{"replica-1", "replica-3"},
		},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Persist the exact selected-retirement plan before withdrawing traffic")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Empty(t, externalMutations)
	input.Operation = result.Operation

	t.Log("Withdraw only the nominated victims while membership remains unchanged")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, [][]ReplicaID{{"replica-1", "replica-3"}}, traffic.withdrawCalls)
	assert.Empty(t, membership.submitCalls)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)

	t.Log("Observe operation-scoped drain after restart and persist Submitting")
	traffic.snapshot = TrafficSnapshot{
		OperationID: workflowTestOperationID,
		Admitted:    []ReplicaID{"replica-0", "replica-2"},
		Drained:     []ReplicaID{"replica-1", "replica-3"},
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, membership.submitCalls)
	input.Operation = result.Operation

	t.Log("Submit the persisted retirement request and preserve the exact victim identities")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	require.Len(t, membership.submitCalls, 1)
	assert.Equal(t, []ReplicaID{"replica-1", "replica-3"}, membership.submitCalls[0].NominatedReplicas)
	input.Operation = result.Operation

	t.Log("Observe and persist the committed survivor topology without releasing capacity")
	membership.topology = workflowTopology(2, "replica-0", "replica-2")
	membership.operation = BackendOperation{
		ID:                          workflowTestOperationID,
		BackendID:                   "backend-" + workflowTestOperationID,
		TargetReplicas:              2,
		Phase:                       BackendOperationPhaseCommitted,
		CommittedTopologyGeneration: 2,
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Empty(t, capacity.releaseCalls)
	input.Operation = result.Operation

	t.Log("Persist UID-bound release authorization before invoking the workload manager")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.observeReleaseCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Equal(t, []AuthorizedReplica{
		{
			ReplicaID: "replica-1",
			SlotID:    "slot-replica-1",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-1",
				UID:       "uid-pod-1",
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
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Restart from durable authorization and release only the exact authorized Pod UIDs")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.ReleaseAuthorizationChanged)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, *input.ReleaseAuthorization, capacity.releaseCalls[0])

	t.Log("Observe an in-progress release without replaying it")
	capacity.releaseObservation = CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplying,
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.False(t, result.ReleaseAuthorizationChanged)
	assert.Len(t, capacity.releaseCalls, 1)

	t.Log("Clear durable authorization only after every authorized UID is observably absent")
	capacity.releaseObservation.Phase = CapacityReleasePhaseApplied
	capacity.snapshot = CapacitySnapshot{
		Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-2", "pod-2"),
		},
		FencedReplicas: []ReplicaID{"replica-1", "replica-3"},
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, []string{
		"traffic.withdraw:" + workflowTestOperationID,
		"membership.submit:" + workflowTestOperationID,
		"capacity.release:" + workflowTestReleaseID,
	}, externalMutations)
}

func TestWorkflowCoordinatorRefusesStaleReleaseAuthorization(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{
			Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				{
					ID:     "replica-1",
					SlotID: "slot-replica-1",
					CapacityRefs: []CapacityRef{{
						Namespace: "test",
						Name:      "pod-1",
						UID:       "replacement-uid",
					}},
					Availability: ReplicaAvailabilityAvailable,
				},
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
	operation := workflowCommittedShrinkOperation()
	authorization := &ReleaseAuthorization{
		ID:                 workflowTestReleaseID,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     1,
		Replicas: []AuthorizedReplica{{
			ReplicaID: "replica-1",
			SlotID:    "slot-replica-1",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-1",
				UID:       "original-uid",
			}},
		}},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Discard an unapplied authorization after the selected Pod name is reused by a different UID")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:              "group-0",
		SpecGeneration:       2,
		DesiredReplicas:      1,
		Operation:            operation,
		ReleaseAuthorization: authorization,
	})
	require.ErrorContains(t, err, "stale")
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Equal(t, []string{workflowTestReleaseID}, capacity.observeReleaseCalls)
	assert.Empty(t, capacity.releaseCalls)
	assert.Empty(t, externalMutations)
}

func TestWorkflowCoordinatorDoesNotApplyStaleAuthorizationToReplacementAfterTimeout(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{
			Allocations: []ReplicaAllocation{
				workflowReplicaAllocation("replica-0", "pod-0"),
				{
					ID:     "replica-1",
					SlotID: "slot-replica-1",
					CapacityRefs: []CapacityRef{{
						Namespace: "test",
						Name:      "pod-1",
						UID:       "replacement-uid",
					}},
					Availability: ReplicaAvailabilityAvailable,
				},
			},
			FencedReplicas: []ReplicaID{"replica-1"},
		},
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
	authorization := &ReleaseAuthorization{
		ID:                 workflowTestReleaseID,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     1,
		Replicas: []AuthorizedReplica{{
			ReplicaID: "replica-1",
			SlotID:    "slot-replica-1",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-1",
				UID:       "original-uid",
			}},
		}},
	}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Recover the applied release outcome before comparing capacity against the old UID binding")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:              "group-0",
		SpecGeneration:       2,
		DesiredReplicas:      1,
		Operation:            workflowCommittedShrinkOperation(),
		ReleaseAuthorization: authorization,
	})
	require.NoError(t, err)
	assert.Nil(t, result.ReleaseAuthorization)
	assert.True(t, result.ReleaseAuthorizationChanged)
	assert.Empty(t, capacity.releaseCalls, "an applied operation must not be replayed against replacement capacity")
	assert.Empty(t, externalMutations)
}

func TestReleaseAuthorizationBindsReplicaSlotAndPodIdentity(t *testing.T) {
	authorization := ReleaseAuthorization{
		ID:                 workflowTestReleaseID,
		OperationID:        workflowTestOperationID,
		TopologyGeneration: 2,
		TargetReplicas:     1,
		Replicas: []AuthorizedReplica{{
			ReplicaID: "replica-1",
			SlotID:    "slot-replica-1",
			CapacityRefs: []CapacityRef{{
				Namespace: "test",
				Name:      "pod-1",
				UID:       "uid-pod-1",
			}},
		}},
	}
	tests := []struct {
		name     string
		capacity CapacitySnapshot
	}{
		{
			name: "logical replica moves to another slot",
			capacity: CapacitySnapshot{Allocations: []ReplicaAllocation{{
				ID:     "replica-1",
				SlotID: "replacement-slot",
				CapacityRefs: []CapacityRef{{
					Namespace: "test",
					Name:      "pod-1",
					UID:       "uid-pod-1",
				}},
				Availability: ReplicaAvailabilityAvailable,
			}}},
		},
		{
			name: "authorized Pod moves to another logical allocation",
			capacity: CapacitySnapshot{Allocations: []ReplicaAllocation{{
				ID:     "replica-9",
				SlotID: "slot-replica-9",
				CapacityRefs: []CapacityRef{{
					Namespace: "test",
					Name:      "pod-1",
					UID:       "uid-pod-1",
				}},
				Availability: ReplicaAvailabilityAvailable,
			}}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Refuse release when any stable capacity correlation differs from authorization")
			err := validateAuthorizationFresh(authorization, tt.capacity)
			require.ErrorContains(t, err, "stale")
		})
	}
}

func TestWorkflowCoordinatorAuthorizesEveryPodInMultiPodReplica(t *testing.T) {
	externalMutations := make([]string, 0)
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0-a", "pod-0-b"),
			workflowReplicaAllocation("replica-1", "pod-1-b", "pod-1-a"),
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
	operation := workflowCommittedShrinkOperation()
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)

	t.Log("Build one indivisible release authorization containing every Pod in the selected replica")
	result, err := coordinator.Reconcile(context.Background(), ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 1,
		Operation:       operation,
	})
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	require.Len(t, result.ReleaseAuthorization.Replicas, 1)
	assert.Equal(t, "replica-1", string(result.ReleaseAuthorization.Replicas[0].ReplicaID))
	assert.Equal(t, "slot-replica-1", string(result.ReleaseAuthorization.Replicas[0].SlotID))
	assert.Equal(t, []CapacityRef{
		{Namespace: "test", Name: "pod-1-a", UID: "uid-pod-1-a"},
		{Namespace: "test", Name: "pod-1-b", UID: "uid-pod-1-b"},
	}, result.ReleaseAuthorization.Replicas[0].CapacityRefs)
	assert.Empty(t, capacity.releaseCalls)

	t.Log("Apply the persisted multi-Pod authorization as one exact workload-manager request")
	input := ReconcileInput{
		GroupID:              "group-0",
		SpecGeneration:       2,
		DesiredReplicas:      1,
		Operation:            operation,
		ReleaseAuthorization: result.ReleaseAuthorization,
	}
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	_, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.Len(t, capacity.releaseCalls, 1)
	assert.Equal(t, result.ReleaseAuthorization.Replicas, capacity.releaseCalls[0].Replicas)
}

func TestWorkflowCoordinatorDegradesThenRestoresFailedReplica(t *testing.T) {
	const (
		degradeOperationID = "recovery-degrade-1"
		restoreOperationID = "recovery-restore-1"
	)
	externalMutations := make([]string, 0)
	failedAllocation := workflowReplicaAllocation("replica-3", "pod-3")
	failedAllocation.Availability = ReplicaAvailabilityUnavailable
	capacity := &workflowCapacityAdapter{
		snapshot: CapacitySnapshot{Allocations: []ReplicaAllocation{
			workflowReplicaAllocation("replica-0", "pod-0"),
			workflowReplicaAllocation("replica-1", "pod-1"),
			workflowReplicaAllocation("replica-2", "pod-2"),
			failedAllocation,
		}},
		externalMutationHistory: &externalMutations,
	}
	capabilities := MembershipCapabilities{Intents: []OperationIntent{
		OperationIntentRecover,
		OperationIntentGrow,
	}}
	membership := &workflowMembershipAdapter{
		capabilities:            &capabilities,
		topology:                workflowTopology(1, "replica-0", "replica-1", "replica-2", "replica-3"),
		externalMutationHistory: &externalMutations,
	}
	traffic := &workflowTrafficAdapter{
		snapshot: TrafficSnapshot{
			Admitted: []ReplicaID{"replica-0", "replica-1", "replica-2"},
		},
		externalMutationHistory: &externalMutations,
	}
	input := ReconcileInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Plan: &OperationPlan{
			ID:                "degrade-plan-1",
			Intent:            OperationIntentRecover,
			TargetReplicas:    3,
			NominatedReplicas: []ReplicaID{"replica-3"},
		},
	}
	operationIDs := &fakeOperationIDGenerator{ids: []string{degradeOperationID, restoreOperationID}}
	coordinator := newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = operationIDs.Next

	t.Log("Persist a recovery reduction while preserving the desired healthy size as a queued target")
	result, err := coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationIntentRecover, result.Operation.Intent)
	assert.Equal(t, int32(3), result.Operation.TargetReplicas)
	assert.Equal(t, int32Pointer(4), result.Operation.QueuedTargetReplicas)
	input.Operation = result.Operation

	t.Log("Withdraw the failed replica and persist Submitting after operation-scoped drain is observed")
	_, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	traffic.snapshot = TrafficSnapshot{
		OperationID: degradeOperationID,
		Admitted:    []ReplicaID{"replica-0", "replica-1", "replica-2"},
		Drained:     []ReplicaID{"replica-3"},
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	input.Operation = result.Operation

	t.Log("Submit and commit the exact survivor topology before authorizing failed capacity release")
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	input.Operation = result.Operation
	membership.topology = workflowTopology(2, "replica-0", "replica-1", "replica-2")
	membership.operation = BackendOperation{
		ID:                          degradeOperationID,
		BackendID:                   "backend-" + degradeOperationID,
		TargetReplicas:              3,
		Phase:                       BackendOperationPhaseCommitted,
		CommittedTopologyGeneration: 2,
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	input.Operation = result.Operation
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.ReleaseAuthorization)
	input.ReleaseAuthorization = result.ReleaseAuthorization

	t.Log("Release the failed Pod identity and persist its applied capacity-target proof after removal")
	_, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
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
	assert.Equal(t, int32(3), result.Operation.CapacityTargetReplicas)
	assert.Equal(t, int64(2), result.Operation.CapacityTopologyGeneration)
	assert.True(t, result.Operation.CapacityTargetApplied)
	assert.True(t, result.OperationChanged)
	assert.Nil(t, result.ReleaseAuthorization)
	input.Operation = result.Operation
	input.ReleaseAuthorization = nil

	t.Log("Persist post-commit completion as a separate durable level before replacing the recovery operation")
	membership.operation = BackendOperation{Phase: BackendOperationPhaseAbsent}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, degradeOperationID, result.Operation.ID)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.True(t, result.Operation.PostCommitComplete)
	assert.True(t, result.OperationChanged)
	input.Operation = result.Operation

	t.Log("Restart from completed recovery and persist restoration growth for the unchanged desired size")
	coordinator = newWorkflowCoordinatorForTest(capacity, membership, traffic)
	coordinator.operations.newOperationID = operationIDs.Next
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, restoreOperationID, result.Operation.ID)
	assert.Equal(t, OperationIntentGrow, result.Operation.Intent)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	input.Operation = result.Operation
	input.Plan = nil

	t.Log("Allocate a replacement, freeze its stable replica identity, and submit restoration")
	_, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	replacement := workflowReplicaAllocation("replica-3", "pod-3")
	replacement.CapacityRefs[0].UID = "replacement-uid"
	capacity.snapshot.Allocations = append(capacity.snapshot.Allocations, replacement)
	capacity.snapshot.FencedReplicas = nil
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []ReplicaID{"replica-3"}, result.Operation.JoiningReplicas)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	input.Operation = result.Operation
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	input.Operation = result.Operation

	t.Log("Commit the restored topology before readmitting the replacement replica")
	membership.topology = workflowTopology(3, "replica-0", "replica-1", "replica-2", "replica-3")
	membership.operation = BackendOperation{
		ID:                          restoreOperationID,
		BackendID:                   "backend-" + restoreOperationID,
		TargetReplicas:              4,
		Phase:                       BackendOperationPhaseCommitted,
		CommittedTopologyGeneration: 3,
	}
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	input.Operation = result.Operation
	result, err = coordinator.Reconcile(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, [][]ReplicaID{{"replica-3"}}, traffic.admitCalls)
	assert.Equal(t, []string{
		"traffic.withdraw:" + degradeOperationID,
		"membership.submit:" + degradeOperationID,
		"capacity.release:" + workflowTestReleaseID,
		"capacity.ensure:4",
		"membership.submit:" + restoreOperationID,
		"traffic.admit:" + restoreOperationID,
	}, externalMutations)
}

func newWorkflowCoordinatorForTest(
	capacity CapacityAdapter,
	membership MembershipAdapter,
	traffic TrafficAdapter,
) *WorkflowCoordinator {
	coordinator := NewWorkflowCoordinator(capacity, membership, traffic)
	coordinator.operations.now = workflowFixedTime
	coordinator.operations.newOperationID = workflowFixedOperationID
	coordinator.newReleaseID = workflowFixedReleaseID
	return coordinator
}

func workflowFixedTime() time.Time {
	return workflowTestTime
}

func workflowFixedOperationID() string {
	return workflowTestOperationID
}

func workflowFixedReleaseID() string {
	return workflowTestReleaseID
}

func workflowTopology(generation int64, replicaIDs ...ReplicaID) MembershipTopology {
	topology := MembershipTopology{
		Generation: generation,
		Replicas:   make([]ReplicaMembership, len(replicaIDs)),
	}

	// Give every logical replica one unique engine-native member for authoritative correlation.
	for i, replicaID := range replicaIDs {
		topology.Replicas[i] = ReplicaMembership{
			ReplicaID:     replicaID,
			NativeMembers: []NativeMemberID{"native-" + NativeMemberID(replicaID)},
		}
	}
	return topology
}

func workflowReplicaAllocation(replicaID ReplicaID, podNames ...string) ReplicaAllocation {
	allocation := ReplicaAllocation{
		ID:           replicaID,
		SlotID:       "slot-" + CapacitySlotID(replicaID),
		CapacityRefs: make([]CapacityRef, len(podNames)),
		Availability: ReplicaAvailabilityAvailable,
	}

	// Build physically disjoint concrete identities for the requested logical allocation.
	for i, podName := range podNames {
		allocation.CapacityRefs[i] = CapacityRef{
			Namespace: "test",
			Name:      podName,
			UID:       PodUID("uid-" + podName),
		}
	}
	return allocation
}

func workflowCommittedShrinkOperation() *Operation {
	return &Operation{
		ID:                          workflowTestOperationID,
		Attempt:                     1,
		PlanID:                      "plan-1",
		Intent:                      OperationIntentShrink,
		SpecGeneration:              2,
		BaseTopologyGeneration:      1,
		BaseReplicas:                []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:              1,
		NominatedReplicas:           []ReplicaID{"replica-1"},
		Phase:                       OperationPhaseCommitted,
		CommittedTopologyGeneration: 2,
		StartedAt:                   workflowTestTime.Add(-time.Minute),
		LastTransitionTime:          workflowTestTime,
	}
}

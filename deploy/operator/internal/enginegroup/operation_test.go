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
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testBackendOperationID = "backend-1"

var testNow = time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)

type fakeMembershipAdapter struct {
	capabilities         *MembershipCapabilities
	capabilitiesErr      error
	capabilitiesCalls    int
	topology             MembershipTopology
	topologyObservations []MembershipTopology
	topologyErr          error
	topologyCalls        int
	operation            BackendOperation
	operationErr         error
	submitResult         BackendOperation
	submitErr            error
	submitCalls          []MembershipRequest
	observeCalls         []operationAttempt
	submittedRequests    map[operationAttempt]MembershipRequest
}

type operationAttempt struct {
	ID      string
	Attempt int32
}

func (f *fakeMembershipAdapter) ObserveCapabilities(context.Context, GroupID) (MembershipCapabilities, error) {
	f.capabilitiesCalls++
	if f.capabilitiesErr != nil {
		return MembershipCapabilities{}, f.capabilitiesErr
	}
	if f.capabilities == nil {
		return allTestMembershipCapabilities(), nil
	}
	return MembershipCapabilities{Intents: slices.Clone(f.capabilities.Intents)}, nil
}

func (f *fakeMembershipAdapter) ObserveTopology(context.Context, GroupID) (MembershipTopology, error) {
	f.topologyCalls++
	if f.topologyErr != nil {
		return MembershipTopology{}, f.topologyErr
	}
	if len(f.topologyObservations) == 0 {
		return cloneTopology(f.topology), nil
	}

	topology := f.topologyObservations[0]
	f.topologyObservations = f.topologyObservations[1:]
	return cloneTopology(topology), nil
}

func (f *fakeMembershipAdapter) ObserveOperation(
	_ context.Context,
	_ GroupID,
	operationID string,
	attempt int32,
) (BackendOperation, error) {
	f.observeCalls = append(f.observeCalls, operationAttempt{ID: operationID, Attempt: attempt})
	if f.operationErr != nil {
		return BackendOperation{}, f.operationErr
	}
	if f.operation.Phase == "" {
		return BackendOperation{Phase: BackendOperationPhaseAbsent}, nil
	}
	operation := cloneBackendOperation(f.operation)
	if operation.Attempt == 0 {
		operation.Attempt = attempt
	}
	return operation, nil
}

func (f *fakeMembershipAdapter) SubmitOperation(_ context.Context, _ GroupID, request MembershipRequest) (BackendOperation, error) {
	request.BaseReplicas = slices.Clone(request.BaseReplicas)
	request.JoiningReplicas = slices.Clone(request.JoiningReplicas)
	request.NominatedReplicas = slices.Clone(request.NominatedReplicas)
	f.submitCalls = append(f.submitCalls, request)

	// Model backend idempotency by operation and attempt while rejecting payload reuse.
	key := operationAttempt{ID: request.ID, Attempt: request.Attempt}
	if f.submittedRequests == nil {
		f.submittedRequests = make(map[operationAttempt]MembershipRequest)
	}
	if existing, submitted := f.submittedRequests[key]; submitted {
		if !reflect.DeepEqual(existing, request) {
			return BackendOperation{}, errors.New("conflicting payload for membership operation attempt")
		}
	} else {
		f.submittedRequests[key] = request
	}
	if f.submitErr != nil {
		return BackendOperation{}, f.submitErr
	}
	if f.submitResult.Phase != "" {
		operation := cloneBackendOperation(f.submitResult)
		if operation.Attempt == 0 {
			operation.Attempt = request.Attempt
		}
		return operation, nil
	}

	return BackendOperation{
		ID:             request.ID,
		Attempt:        request.Attempt,
		TargetReplicas: request.TargetReplicas,
		Phase:          BackendOperationPhaseAccepted,
	}, nil
}

type fakeOperationIDGenerator struct {
	ids   []string
	calls int
}

func (g *fakeOperationIDGenerator) Next() string {
	if g.calls >= len(g.ids) {
		return ""
	}

	id := g.ids[g.calls]
	g.calls++
	return id
}

func TestOperationCoordinatorPersistsBeforeSubmitting(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
	ids := &fakeOperationIDGenerator{ids: []string{"operation-1"}}
	coordinator := newTestOperationCoordinator(adapter, ids.Next)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 4,
	}

	t.Log("Create a durable Pending operation without calling the membership backend")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Equal(t, "operation-1", result.Operation.ID)
	assert.Equal(t, int32(4), result.Operation.TargetReplicas)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Keep the operation Pending until capacity or traffic preconditions are durable")
	input.Operation = result.Operation
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Freeze exact joining identities in Submitting without calling the membership backend")
	prepared, err := coordinator.PrepareSubmission(result.Operation, []ReplicaID{"replica-3", "replica-2"})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, prepared.Phase)
	assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, prepared.JoiningReplicas)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Observe that the persisted operation is absent before requesting submission")
	input.Operation = prepared
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.True(t, result.SubmissionNeeded)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Submit the exact persisted operation after the caller rechecks prerequisites")
	result, err = coordinator.Submit(context.Background(), input.GroupID, result.Operation)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAccepted, result.Operation.Phase)
	require.Len(t, adapter.submitCalls, 1)
	assert.Equal(t, MembershipRequest{
		ID:                     "operation-1",
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         4,
		JoiningReplicas:        []ReplicaID{"replica-2", "replica-3"},
	}, adapter.submitCalls[0])
}

func TestOperationCoordinatorRechecksCapabilityImmediatelyBeforeSubmit(t *testing.T) {
	capabilities := MembershipCapabilities{Intents: []OperationIntent{OperationIntentShrink}}
	adapter := &fakeMembershipAdapter{
		capabilities: &capabilities,
		topology:     membershipTopology(1, 2),
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

	t.Log("Reject a persisted growth operation when support is absent at the mutation boundary")
	result, err := coordinator.Submit(
		context.Background(),
		"group-0",
		durableOperation(OperationPhaseSubmitting, 4),
	)
	require.ErrorContains(t, err, "membership operation \"Grow\" is not supported")
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Equal(t, 1, adapter.capabilitiesCalls)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorDoesNothingForConvergedMembership(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(3, 4)}
	coordinator := NewOperationCoordinator(adapter)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
	}

	t.Log("Observe matching desired and committed replica counts")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Nil(t, result.Operation)
	assert.False(t, result.OperationChanged)
	assert.Equal(t, int32(4), result.Topology.ReplicaCount())
	assert.Empty(t, adapter.observeCalls)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorQueuesConvergedTargetBehindPendingOperation(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Operation:       durableOperation(OperationPhasePending, 4),
	}

	t.Log("Keep a potentially effectful Pending request frozen and queue the converged target")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, "operation-1", result.Operation.ID)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Equal(t, int32(4), result.Operation.TargetReplicas)
	assert.Equal(t, int32Pointer(2), result.Operation.QueuedTargetReplicas)
	assert.True(t, result.OperationChanged)
	assert.Empty(t, adapter.observeCalls)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorRecoversAmbiguousSubmissionAfterRestart(t *testing.T) {
	requestTimeout := errors.New("request timeout")
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(1, 2),
		operation: BackendOperation{
			Phase: BackendOperationPhaseAbsent,
		},
		submitErr: requestTimeout,
	}
	operation := durableOperation(OperationPhaseSubmitting, 4)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       operation,
	}

	t.Log("Observe that the persisted operation needs submission")
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.True(t, result.SubmissionNeeded)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Leave the durable operation in Submitting when the request outcome is ambiguous")
	result, err = coordinator.Submit(context.Background(), input.GroupID, result.Operation)
	require.ErrorIs(t, err, requestTimeout)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	require.Len(t, adapter.submitCalls, 1)
	assert.Equal(t, "operation-1", adapter.submitCalls[0].ID)

	t.Log("Simulate a controller restart after the backend accepted the timed-out request")
	adapter.submitErr = nil
	adapter.operation = BackendOperation{
		ID:             "operation-1",
		BackendID:      "backend-7",
		TargetReplicas: 4,
		Phase:          BackendOperationPhaseCommitting,
	}
	restarted := newTestOperationCoordinator(adapter, emptyOperationID)
	input.Operation = result.Operation
	result, err = restarted.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitting, result.Operation.Phase)
	assert.Equal(t, "backend-7", result.Operation.BackendOperationID)
	assert.Len(t, adapter.submitCalls, 1, "restart must observe instead of submitting a competitor")
}

func TestOperationCoordinatorKeepsUnobservableSubmissionRetryable(t *testing.T) {
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(1, 2),
		submitResult: BackendOperation{
			Phase: BackendOperationPhaseAbsent,
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       durableOperation(OperationPhaseSubmitting, 4),
	}

	t.Log("Observe that the persisted operation needs submission")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.True(t, result.SubmissionNeeded)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Keep Submitting when the backend acknowledgement is not yet observable")
	result, err = coordinator.Submit(context.Background(), input.GroupID, result.Operation)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	require.Len(t, adapter.submitCalls, 1)
	assert.Equal(t, "operation-1", adapter.submitCalls[0].ID)
}

func TestOperationCoordinatorReplaysOnlyTheExactAttemptPayload(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	original := durableOperation(OperationPhaseSubmitting, 4)

	t.Log("Submit the same durable attempt twice and let backend idempotency return the same outcome")
	first, err := coordinator.Submit(context.Background(), "group-0", original)
	require.NoError(t, err)
	second, err := coordinator.Submit(context.Background(), "group-0", original)
	require.NoError(t, err)
	assert.Equal(t, first.Operation, second.Operation)
	require.Len(t, adapter.submitCalls, 2)
	assert.Equal(t, adapter.submitCalls[0], adapter.submitCalls[1])

	t.Log("Reject reuse of that operation attempt with a different immutable target and joining set")
	conflicting := cloneOperation(original)
	conflicting.TargetReplicas = 3
	conflicting.JoiningReplicas = []ReplicaID{"replica-2"}
	_, err = coordinator.Submit(context.Background(), "group-0", conflicting)
	require.ErrorContains(t, err, "conflicting payload")
}

func TestOperationCoordinatorBeginsCompensationWhenBaseTopologyChangesBeforeSubmit(t *testing.T) {
	tests := []struct {
		name     string
		topology MembershipTopology
	}{
		{
			name:     "topology generation advanced",
			topology: membershipTopology(2, 2),
		},
		{
			name: "replica identity changed at the same generation",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					{ReplicaID: "replica-0", NativeMembers: []NativeMemberID{"dp-0"}},
					{ReplicaID: "replica-9", NativeMembers: []NativeMemberID{"dp-9"}},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{topology: tt.topology}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
			operation := durableOperation(OperationPhaseSubmitting, 4)

			t.Log("Persist compensation after proving the stale request cannot be submitted")
			result, err := coordinator.Submit(context.Background(), "group-0", operation)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
			require.NotNil(t, result.Operation.Failure)
			assert.Equal(t, "BaseTopologyChanged", result.Operation.Failure.Reason)
			assert.True(t, result.OperationChanged)
			assert.Equal(t, []operationAttempt{{ID: operation.ID, Attempt: operation.Attempt}}, adapter.observeCalls)
			assert.Empty(t, adapter.submitCalls)
		})
	}
}

func TestOperationCoordinatorCompensatesSubmittingOperationAfterAbsentBaseDrift(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(2, 3)}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	operation := durableOperation(OperationPhaseSubmitting, 4)

	t.Log("Begin compensation before requesting any prerequisite or backend replay")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, "BaseTopologyChanged", result.Operation.Failure.Reason)
	assert.True(t, result.OperationChanged)
	assert.False(t, result.SubmissionNeeded)
	assert.Equal(t, []operationAttempt{{ID: operation.ID, Attempt: operation.Attempt}}, adapter.observeCalls)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorPersistsCompensationBeforeGrowthIdentitiesAreFrozen(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(2, 2)}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       durableOperation(OperationPhasePending, 4),
	}

	t.Log("Record base-topology compensation even though no joining identities were frozen yet")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Empty(t, result.Operation.JoiningReplicas)
	assert.True(t, result.OperationChanged)

	t.Log("Keep the persisted compensation record stable for the workflow to execute after restart")
	input.Operation = result.Operation
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorRequiresIdentityAwareShrinkPlan(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
	ids := &fakeOperationIDGenerator{ids: []string{"operation-1"}}
	coordinator := newTestOperationCoordinator(adapter, ids.Next)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 1,
	}

	t.Log("Reject a cardinal-only shrink because it cannot identify the exact retiring replica")
	result, err := coordinator.Step(context.Background(), input)
	require.ErrorContains(t, err, "identity-aware operation plan")
	assert.Nil(t, result.Operation)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Create a Pending shrink after the caller nominates the exact retiring replica")
	input.Plan = &OperationPlan{
		ID:                "plan-1",
		Intent:            OperationIntentShrink,
		TargetReplicas:    1,
		NominatedReplicas: []ReplicaID{"replica-1"},
	}
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1"}, result.Operation.BaseReplicas)
	assert.Equal(t, []ReplicaID{"replica-1"}, result.Operation.NominatedReplicas)
}

func TestOperationCoordinatorFreezesEveryDurableOperationTarget(t *testing.T) {
	tests := []struct {
		name            string
		phase           OperationPhase
		desiredReplicas int32
	}{
		{
			name:            "pending preparation may already have effects",
			phase:           OperationPhasePending,
			desiredReplicas: 6,
		},
		{
			name:            "submitted operation may be in flight",
			phase:           OperationPhaseCommitting,
			desiredReplicas: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{
				topology: membershipTopology(1, 2),
				operation: BackendOperation{
					ID:             "operation-1",
					TargetReplicas: 4,
					Phase:          BackendOperationPhaseCommitting,
				},
			}
			ids := &fakeOperationIDGenerator{ids: []string{"unused-operation"}}
			coordinator := newTestOperationCoordinator(adapter, ids.Next)
			input := OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: tt.desiredReplicas,
				Operation:       durableOperation(tt.phase, 4),
			}

			t.Log("Queue a newer desired target without replacing the durable operation")
			original := cloneOperation(input.Operation)
			result, err := coordinator.Step(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, "operation-1", result.Operation.ID)
			assert.Equal(t, int32(4), result.Operation.TargetReplicas)
			assert.Equal(t, int32Pointer(6), result.Operation.QueuedTargetReplicas)
			assert.Zero(t, ids.calls)
			assert.Equal(t, original, input.Operation, "Step must not mutate caller-owned durable state")
		})
	}
}

func TestOperationCoordinatorRetainsCommittedOperationForWorkflowFinalization(t *testing.T) {
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(2, 4),
		operation: BackendOperation{
			ID:                          "operation-1",
			BackendID:                   testBackendOperationID,
			TargetReplicas:              4,
			Phase:                       BackendOperationPhaseCommitted,
			CommittedTopologyGeneration: 2,
		},
	}
	ids := &fakeOperationIDGenerator{ids: []string{"operation-2"}}
	coordinator := newTestOperationCoordinator(adapter, ids.Next)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 6,
		Operation:       durableOperation(OperationPhaseCommitting, 4),
	}

	t.Log("Finish the immutable active operation while retaining the newer desired target")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int32Pointer(6), result.Operation.QueuedTargetReplicas)
	assert.Equal(t, int64(2), result.Operation.CommittedTopologyGeneration)

	t.Log("Retain the committed operation until the workflow finishes traffic and capacity work")
	input.Operation = result.Operation
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, "operation-1", result.Operation.ID)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int32(4), result.Operation.TargetReplicas)
	assert.Equal(t, int32Pointer(6), result.Operation.QueuedTargetReplicas)
	assert.Equal(t, 0, ids.calls)
}

func TestOperationCoordinatorHandlesExplicitBackendFailure(t *testing.T) {
	tests := []struct {
		name              string
		classification    FailureClassification
		wantRetryPrepared bool
	}{
		{
			name:              "retryable failure reuses operation",
			classification:    FailureClassificationRetryable,
			wantRetryPrepared: true,
		},
		{
			name:              "terminal failure cannot be retried",
			classification:    FailureClassificationTerminal,
			wantRetryPrepared: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{
				topology: membershipTopology(1, 2),
				operation: BackendOperation{
					ID:             "operation-1",
					TargetReplicas: 4,
					Phase:          BackendOperationPhaseFailed,
					Failure: &OperationFailure{
						Classification: tt.classification,
						Reason:         "BackendRejected",
						Message:        "membership transition failed",
					},
				},
			}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
			input := OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       durableOperation(OperationPhaseCommitting, 4),
			}

			t.Log("Record the backend's structured failure")
			result, err := coordinator.Step(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
			require.NotNil(t, result.Operation.Failure)
			assert.Equal(t, tt.classification, result.Operation.Failure.Classification)

			t.Log("Keep the persisted failure stable during membership observation")
			input.Operation = result.Operation
			result, err = coordinator.Step(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
			assert.Equal(t, "operation-1", result.Operation.ID)

			t.Log("Prepare only an explicitly retryable failure for idempotent resubmission")
			prepared, prepareErr := coordinator.PrepareSubmission(result.Operation, nil)
			if tt.wantRetryPrepared {
				require.NoError(t, prepareErr)
				assert.Equal(t, OperationPhaseSubmitting, prepared.Phase)
				assert.Equal(t, "operation-1", prepared.ID)
				assert.Equal(t, int32(2), prepared.Attempt)
				assert.Nil(t, prepared.Failure)
				assert.Equal(t, []ReplicaID{"replica-2", "replica-3"}, prepared.JoiningReplicas)
			} else {
				require.Error(t, prepareErr)
				assert.Nil(t, prepared)
			}
		})
	}
}

func TestOperationCoordinatorRejectsChangedBackendIdentityAndStaleProgress(t *testing.T) {
	tests := []struct {
		name            string
		backendID       string
		backendPhase    BackendOperationPhase
		wantBackendID   string
		wantPhase       OperationPhase
		wantStateChange bool
	}{
		{
			name:            "empty backend identity preserves correlation",
			backendPhase:    BackendOperationPhaseCommitting,
			wantBackendID:   testBackendOperationID,
			wantPhase:       OperationPhaseCommitting,
			wantStateChange: false,
		},
		{
			name:            "accepted observation cannot regress committing",
			backendID:       testBackendOperationID,
			backendPhase:    BackendOperationPhaseAccepted,
			wantBackendID:   testBackendOperationID,
			wantPhase:       OperationPhaseCommitting,
			wantStateChange: false,
		},
		{
			name:            "changed backend identity loses correlation",
			backendID:       "backend-2",
			backendPhase:    BackendOperationPhaseCommitting,
			wantBackendID:   testBackendOperationID,
			wantPhase:       OperationPhaseUnknown,
			wantStateChange: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{
				topology: membershipTopology(1, 2),
				operation: BackendOperation{
					ID:             "operation-1",
					Attempt:        1,
					BackendID:      tt.backendID,
					TargetReplicas: 4,
					Phase:          tt.backendPhase,
				},
			}
			operation := durableOperation(OperationPhaseCommitting, 4)
			operation.BackendOperationID = testBackendOperationID
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

			t.Log("Apply only monotonic progress correlated to the immutable backend operation identity")
			result, err := coordinator.Step(context.Background(), OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       operation,
			})
			require.NoError(t, err)
			assert.Equal(t, tt.wantPhase, result.Operation.Phase)
			assert.Equal(t, tt.wantBackendID, result.Operation.BackendOperationID)
			assert.Equal(t, tt.wantStateChange, result.OperationChanged)
		})
	}
}

func TestOperationCoordinatorQueuesNewTargetBehindResolvedFailure(t *testing.T) {
	tests := []struct {
		name           string
		classification FailureClassification
	}{
		{
			name:           "retryable failure",
			classification: FailureClassificationRetryable,
		},
		{
			name:           "terminal failure",
			classification: FailureClassificationTerminal,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
			ids := &fakeOperationIDGenerator{ids: []string{"unused-operation"}}
			coordinator := newTestOperationCoordinator(adapter, ids.Next)
			operation := durableOperation(OperationPhaseFailed, 4)
			operation.Failure = &OperationFailure{
				Classification: tt.classification,
				Reason:         "BackendRejected",
			}

			t.Log("Retain potentially effectful failed state and queue a newer target for later compensation")
			result, err := coordinator.Step(context.Background(), OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 6,
				Operation:       operation,
			})
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, "operation-1", result.Operation.ID)
			assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
			assert.Equal(t, int32(4), result.Operation.TargetReplicas)
			assert.Equal(t, int32Pointer(6), result.Operation.QueuedTargetReplicas)
			assert.True(t, result.OperationChanged)
			assert.Zero(t, ids.calls)
			assert.Empty(t, adapter.submitCalls)
		})
	}
}

func TestOperationCoordinatorFailsClosedAndRecoversCorrelation(t *testing.T) {
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(1, 2),
		operation: BackendOperation{
			Phase: BackendOperationPhaseAbsent,
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       durableOperation(OperationPhaseAccepted, 4),
	}

	t.Log("Mark an acknowledged operation Unknown when the backend loses its correlation")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Recover from Unknown only after the exact committed operation and topology reappear")
	adapter.topology = membershipTopology(2, 4)
	adapter.operation = BackendOperation{
		ID:                          "operation-1",
		BackendID:                   testBackendOperationID,
		TargetReplicas:              4,
		Phase:                       BackendOperationPhaseCommitted,
		CommittedTopologyGeneration: 2,
	}
	input.Operation = result.Operation
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int64(2), result.Operation.CommittedTopologyGeneration)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorDoesNotRegressUnknownCommittedProof(t *testing.T) {
	tests := []struct {
		name        string
		observation BackendOperation
	}{
		{
			name: "stale accepted progress",
			observation: BackendOperation{
				Phase: BackendOperationPhaseAccepted,
			},
		},
		{
			name: "stale committing progress",
			observation: BackendOperation{
				Phase: BackendOperationPhaseCommitting,
			},
		},
		{
			name: "stale backend failure",
			observation: BackendOperation{
				Phase: BackendOperationPhaseFailed,
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "StaleFailure",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			operation := durableOperation(OperationPhaseUnknown, 4)
			operation.BackendOperationID = testBackendOperationID
			operation.CommittedTopologyGeneration = 2
			observation := tt.observation
			observation.ID = operation.ID
			observation.Attempt = operation.Attempt
			observation.BackendID = "stale-backend"
			observation.TargetReplicas = operation.TargetReplicas
			adapter := &fakeMembershipAdapter{
				topology:  membershipTopology(3, 3),
				operation: observation,
			}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

			t.Log("Keep durable commit proof Unknown until an exact committed topology can be reverified")
			result, err := coordinator.Step(context.Background(), OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       operation,
			})
			require.NoError(t, err)
			assert.Equal(t, operation, result.Operation)
			assert.False(t, result.OperationChanged)
			assert.NoError(t, validateOperation(*result.Operation))
		})
	}
}

func TestOperationCoordinatorKeepsUnknownCommitProofWhileTopologyObservationLags(t *testing.T) {
	operation := durableOperation(OperationPhaseUnknown, 4)
	operation.BackendOperationID = testBackendOperationID
	operation.CommittedTopologyGeneration = 2
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(1, 2),
		operation: BackendOperation{
			ID:                          operation.ID,
			Attempt:                     operation.Attempt,
			BackendID:                   operation.BackendOperationID,
			TargetReplicas:              operation.TargetReplicas,
			Phase:                       BackendOperationPhaseCommitted,
			CommittedTopologyGeneration: operation.CommittedTopologyGeneration,
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

	t.Log("Preserve Unknown and its durable commit proof while the topology read remains stale")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, operation, result.Operation)
	assert.False(t, result.OperationChanged)
	assert.NoError(t, validateOperation(*result.Operation))
}

func TestOperationCoordinatorRequiresExactCommittedTopology(t *testing.T) {
	wrongIdentityTopology := membershipTopology(2, 4)
	wrongIdentityTopology.Replicas[3] = ReplicaMembership{
		ReplicaID:     "replica-9",
		NativeMembers: []NativeMemberID{"dp-9"},
	}
	laterUnexpectedTopology := membershipTopology(3, 3)
	laterUnexpectedTopology.Replicas[2] = ReplicaMembership{
		ReplicaID:     "replica-9",
		NativeMembers: []NativeMemberID{"dp-9"},
	}

	tests := []struct {
		name                 string
		topology             MembershipTopology
		committedGeneration  int64
		backendTarget        int32
		wantPhase            OperationPhase
		wantCommitProof      int64
		wantTopologyReplicas int32
	}{
		{
			name:                 "exact topology commits",
			topology:             membershipTopology(2, 4),
			committedGeneration:  2,
			backendTarget:        4,
			wantPhase:            OperationPhaseCommitted,
			wantCommitProof:      2,
			wantTopologyReplicas: 4,
		},
		{
			name:                 "stale topology keeps committing",
			topology:             membershipTopology(1, 2),
			committedGeneration:  2,
			backendTarget:        4,
			wantPhase:            OperationPhaseCommitting,
			wantTopologyReplicas: 2,
		},
		{
			name:                 "wrong topology cardinality is unknown",
			topology:             membershipTopology(2, 3),
			committedGeneration:  2,
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantTopologyReplicas: 3,
		},
		{
			name:                 "wrong replica identity at the expected cardinality is unknown",
			topology:             wrongIdentityTopology,
			committedGeneration:  2,
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantTopologyReplicas: 4,
		},
		{
			name:                 "later topology with unexpected identity is unknown without commit proof",
			topology:             laterUnexpectedTopology,
			committedGeneration:  2,
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantTopologyReplicas: 3,
		},
		{
			name:                 "later topology generation is unknown",
			topology:             membershipTopology(3, 4),
			committedGeneration:  2,
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantCommitProof:      2,
			wantTopologyReplicas: 4,
		},
		{
			name:                 "mismatched backend target is unknown",
			topology:             membershipTopology(2, 4),
			committedGeneration:  2,
			backendTarget:        5,
			wantPhase:            OperationPhaseUnknown,
			wantTopologyReplicas: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{
				topology: tt.topology,
				operation: BackendOperation{
					ID:                          "operation-1",
					TargetReplicas:              tt.backendTarget,
					Phase:                       BackendOperationPhaseCommitted,
					CommittedTopologyGeneration: tt.committedGeneration,
				},
			}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
			input := OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       durableOperation(OperationPhaseCommitting, 4),
			}

			t.Log("Apply the backend commit observation against authoritative topology")
			result, err := coordinator.Step(context.Background(), input)
			require.NoError(t, err)
			assert.Equal(t, tt.wantPhase, result.Operation.Phase)
			assert.Equal(t, tt.wantCommitProof, result.Operation.CommittedTopologyGeneration)
			assert.Equal(t, tt.wantTopologyReplicas, result.Topology.ReplicaCount())
		})
	}
}

func TestOperationCoordinatorPreservesCommitProofWhenTopologyObservationSkipsToSurvivors(t *testing.T) {
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(3, 3),
		operation: BackendOperation{
			ID:                          "operation-1",
			TargetReplicas:              4,
			Phase:                       BackendOperationPhaseCommitted,
			CommittedTopologyGeneration: 2,
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

	t.Log("Record exact backend commit proof while keeping the later survivor topology Unknown")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       durableOperation(OperationPhaseCommitting, 4),
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Equal(t, int64(2), result.Operation.CommittedTopologyGeneration)
	assert.Equal(t, int32(3), result.Topology.ReplicaCount())
	assert.NoError(t, validateOperation(*result.Operation))

	t.Log("Reobserve the same survivor topology without rewriting identical durable proof")
	stable, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       result.Operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, stable.Operation.Phase)
	assert.Equal(t, int64(2), stable.Operation.CommittedTopologyGeneration)
	assert.False(t, stable.OperationChanged)
}

func TestOperationCoordinatorRejectsCommitThatRemovedWrongReplica(t *testing.T) {
	adapter := &fakeMembershipAdapter{
		topology: MembershipTopology{
			Generation: 2,
			Replicas: []ReplicaMembership{
				{ReplicaID: "replica-0", NativeMembers: []NativeMemberID{"dp-0"}},
				{ReplicaID: "replica-1", NativeMembers: []NativeMemberID{"dp-1"}},
			},
		},
		operation: BackendOperation{
			ID:                          "operation-1",
			TargetReplicas:              2,
			Phase:                       BackendOperationPhaseCommitted,
			CommittedTopologyGeneration: 2,
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	operation := &Operation{
		ID:                     "operation-1",
		Attempt:                1,
		PlanID:                 "plan-1",
		Intent:                 OperationIntentShrink,
		SpecGeneration:         1,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1", "replica-2", "replica-3"},
		TargetReplicas:         2,
		NominatedReplicas:      []ReplicaID{"replica-1", "replica-3"},
		Phase:                  OperationPhaseCommitting,
		StartedAt:              testNow.Add(-time.Minute),
		LastTransitionTime:     testNow.Add(-time.Second),
	}

	t.Log("Reject the expected replica count when the backend retired a retained identity")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 2,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
}

func TestOperationCoordinatorPreservesNominatedReplicaIdentities(t *testing.T) {
	nominations := []ReplicaID{"replica-7", "replica-3"}
	adapter := &fakeMembershipAdapter{topology: membershipTopology(7, 8)}
	ids := &fakeOperationIDGenerator{ids: []string{"recovery-1"}}
	coordinator := newTestOperationCoordinator(adapter, ids.Next)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  4,
		DesiredReplicas: 8,
		Plan: &OperationPlan{
			ID:                "plan-1",
			Intent:            OperationIntentRecover,
			TargetReplicas:    6,
			NominatedReplicas: nominations,
		},
	}

	t.Log("Persist a normalized copy of the exact recovery nominations")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, []ReplicaID{"replica-3", "replica-7"}, result.Operation.NominatedReplicas)
	assert.Equal(t, int32Pointer(8), result.Operation.QueuedTargetReplicas)

	t.Log("Mutate the caller-owned plan and verify durable operation state is unchanged")
	nominations[0] = "replica-mutated"
	assert.Equal(t, []ReplicaID{"replica-3", "replica-7"}, result.Operation.NominatedReplicas)
}

func TestOperationCoordinatorRejectsInvalidPlans(t *testing.T) {
	tests := []struct {
		name string
		plan OperationPlan
	}{
		{
			name: "negative target",
			plan: OperationPlan{Intent: OperationIntentRecover, TargetReplicas: -1},
		},
		{
			name: "grow target does not grow",
			plan: OperationPlan{Intent: OperationIntentGrow, TargetReplicas: 2},
		},
		{
			name: "shrink target does not shrink",
			plan: OperationPlan{Intent: OperationIntentShrink, TargetReplicas: 2},
		},
		{
			name: "retire target is nonzero",
			plan: OperationPlan{Intent: OperationIntentRetire, TargetReplicas: 1},
		},
		{
			name: "unknown intent",
			plan: OperationPlan{Intent: "Replace", TargetReplicas: 2},
		},
		{
			name: "empty nominated replica",
			plan: OperationPlan{
				Intent:            OperationIntentRecover,
				TargetReplicas:    2,
				NominatedReplicas: []ReplicaID{""},
			},
		},
		{
			name: "duplicate nominated replica",
			plan: OperationPlan{
				Intent:            OperationIntentRecover,
				TargetReplicas:    2,
				NominatedReplicas: []ReplicaID{"replica-1", "replica-1"},
			},
		},
		{
			name: "shrink omits exact victim",
			plan: OperationPlan{
				Intent:         OperationIntentShrink,
				TargetReplicas: 1,
			},
		},
		{
			name: "shrink nominates an unknown victim",
			plan: OperationPlan{
				Intent:            OperationIntentShrink,
				TargetReplicas:    1,
				NominatedReplicas: []ReplicaID{"replica-9"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
			plan := tt.plan
			plan.ID = "plan-1"
			input := OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 2,
				Plan:            &plan,
			}

			t.Log("Reject an invalid identity-aware operation plan")
			result, err := coordinator.Step(context.Background(), input)
			require.Error(t, err)
			assert.Nil(t, result.Operation)
			assert.Empty(t, adapter.observeCalls)
			assert.Empty(t, adapter.submitCalls)
		})
	}
}

func TestOperationCoordinatorRejectsInvalidTopologies(t *testing.T) {
	tests := []struct {
		name     string
		topology MembershipTopology
	}{
		{
			name:     "negative generation",
			topology: MembershipTopology{Generation: -1},
		},
		{
			name: "empty replica ID",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					{NativeMembers: []NativeMemberID{"dp-0"}},
				},
			},
		},
		{
			name: "duplicate replica ID",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					{ReplicaID: "replica-0", NativeMembers: []NativeMemberID{"dp-0"}},
					{ReplicaID: "replica-0", NativeMembers: []NativeMemberID{"dp-1"}},
				},
			},
		},
		{
			name: "missing native member",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					{ReplicaID: "replica-0"},
				},
			},
		},
		{
			name: "empty native member ID",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					{ReplicaID: "replica-0", NativeMembers: []NativeMemberID{""}},
				},
			},
		},
		{
			name: "duplicate native member ID",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					{ReplicaID: "replica-0", NativeMembers: []NativeMemberID{"dp-0"}},
					{ReplicaID: "replica-1", NativeMembers: []NativeMemberID{"dp-0"}},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{topology: tt.topology}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
			input := OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 2,
			}

			t.Log("Fail closed on malformed authoritative topology")
			result, err := coordinator.Step(context.Background(), input)
			require.Error(t, err)
			assert.Nil(t, result.Operation)
			assert.Empty(t, adapter.observeCalls)
			assert.Empty(t, adapter.submitCalls)
		})
	}
}

func TestOperationCoordinatorRejectsInvalidStateBeforeBackendAccess(t *testing.T) {
	conflictingPlanOperation := durableOperation(OperationPhaseCommitting, 4)
	conflictingPlanOperation.PlanID = "plan-1"
	committedProofWithoutJoiners := durableOperation(OperationPhaseUnknown, 4)
	committedProofWithoutJoiners.JoiningReplicas = nil
	committedProofWithoutJoiners.CommittedTopologyGeneration = 2
	abortingWithoutFailure := durableOperation(OperationPhaseAborting, 4)
	abortingWithRetryableFailure := durableOperation(OperationPhaseAborting, 4)
	abortingWithRetryableFailure.Failure = &OperationFailure{
		Classification: FailureClassificationRetryable,
		Reason:         "Transient",
	}

	tests := []struct {
		name  string
		input OperationInput
	}{
		{
			name: "empty group ID",
			input: OperationInput{
				SpecGeneration:  1,
				DesiredReplicas: 2,
			},
		},
		{
			name: "negative desired replicas",
			input: OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: -1,
			},
		},
		{
			name: "explicit plan without durable identity",
			input: OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 1,
				Plan: &OperationPlan{
					Intent:            OperationIntentShrink,
					TargetReplicas:    1,
					NominatedReplicas: []ReplicaID{"replica-1"},
				},
			},
		},
		{
			name: "plan identity reused with another payload",
			input: OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 4,
				Plan: &OperationPlan{
					ID:             "plan-1",
					Intent:         OperationIntentGrow,
					TargetReplicas: 5,
				},
				Operation: conflictingPlanOperation,
			},
		},
		{
			name: "failed operation without structured failure",
			input: OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 2,
				Operation:       durableOperation(OperationPhaseFailed, 2),
			},
		},
		{
			name: "aborting operation without structured failure",
			input: OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       abortingWithoutFailure,
			},
		},
		{
			name: "aborting operation with retryable failure",
			input: OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       abortingWithRetryableFailure,
			},
		},
		{
			name: "committed growth proof without frozen joiner identities",
			input: OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       committedProofWithoutJoiners,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 1)}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

			t.Log("Reject invalid durable input without observing or mutating the backend")
			_, err := coordinator.Step(context.Background(), tt.input)
			require.Error(t, err)
			assert.Zero(t, adapter.topologyCalls)
			assert.Empty(t, adapter.observeCalls)
			assert.Empty(t, adapter.submitCalls)
		})
	}
}

func membershipTopology(generation int64, replicas int) MembershipTopology {
	topology := MembershipTopology{
		Generation: generation,
		Replicas:   make([]ReplicaMembership, replicas),
	}
	for i := range replicas {
		topology.Replicas[i] = ReplicaMembership{
			ReplicaID:     ReplicaID(fmt.Sprintf("replica-%d", i)),
			NativeMembers: []NativeMemberID{NativeMemberID(fmt.Sprintf("dp-%d", i))},
		}
	}
	return topology
}

func durableOperation(
	phase OperationPhase,
	targetReplicas int32,
) *Operation {
	operation := &Operation{
		ID:                     "operation-1",
		Attempt:                1,
		Intent:                 OperationIntentGrow,
		SpecGeneration:         1,
		BaseTopologyGeneration: 1,
		BaseReplicas:           []ReplicaID{"replica-0", "replica-1"},
		TargetReplicas:         targetReplicas,
		Phase:                  phase,
		StartedAt:              testNow.Add(-time.Minute),
		LastTransitionTime:     testNow.Add(-time.Second),
	}

	// Every submitted growth record freezes the exact logical replica identities being added.
	if targetReplicas == 4 && phase != OperationPhasePending {
		operation.JoiningReplicas = []ReplicaID{"replica-2", "replica-3"}
	}
	return operation
}

func cloneBackendOperation(operation BackendOperation) BackendOperation {
	operation.Failure = cloneFailure(operation.Failure)
	return operation
}

func newTestOperationCoordinator(
	membership MembershipAdapter,
	newOperationID func() string,
) *OperationCoordinator {
	return &OperationCoordinator{
		membership:     membership,
		now:            fixedNow,
		newOperationID: newOperationID,
	}
}

func fixedNow() time.Time {
	return testNow
}

func emptyOperationID() string {
	return ""
}

func int32Pointer(value int32) *int32 {
	return &value
}

func allTestMembershipCapabilities() MembershipCapabilities {
	return MembershipCapabilities{Intents: []OperationIntent{
		OperationIntentGrow,
		OperationIntentShrink,
		OperationIntentRecover,
		OperationIntentRetire,
	}}
}

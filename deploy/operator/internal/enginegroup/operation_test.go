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
	capabilities                         *MembershipCapabilities
	capabilitiesErr                      error
	capabilitiesCalls                    int
	validatePlanErr                      error
	validatePlanCalls                    []OperationPlan
	validateObservedTransitionErr        error
	validateObservedTransitionCapability *ResolvedOperationCapability
	validateObservedTransitionCalls      []ObservedMembershipTransition
	validateRequestErr                   error
	validateRequestCalls                 []MembershipRequest
	acceptedShapes                       map[OperationShape]struct{}
	topology                             MembershipTopology
	topologyObservations                 []MembershipTopology
	topologyErr                          error
	topologyCalls                        int
	operation                            BackendOperation
	operationErr                         error
	submitResult                         BackendOperation
	submitErr                            error
	submitCalls                          []MembershipRequest
	observeCalls                         []operationAttempt
	submittedRequests                    map[operationAttempt]MembershipRequest
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
	return cloneTestMembershipCapabilities(*f.capabilities), nil
}

func (f *fakeMembershipAdapter) ValidatePlan(
	_ context.Context,
	_ GroupID,
	topology MembershipTopology,
	plan OperationPlan,
) (ResolvedOperationCapability, error) {
	f.validatePlanCalls = append(f.validatePlanCalls, normalizePlan(plan))
	if f.validatePlanErr != nil {
		return ResolvedOperationCapability{}, f.validatePlanErr
	}
	if err := validatePlan(plan, topology); err != nil {
		return ResolvedOperationCapability{}, err
	}

	capability, err := resolveTestOperationCapability(plan, topology, f.currentCapabilities())
	if err != nil {
		return ResolvedOperationCapability{}, err
	}
	if f.acceptedShapes == nil {
		f.acceptedShapes = make(map[OperationShape]struct{})
	}
	f.acceptedShapes[capability.Shape] = struct{}{}
	return capability, nil
}

func (f *fakeMembershipAdapter) ValidateObservedTransition(
	_ context.Context,
	_ GroupID,
	transition ObservedMembershipTransition,
) (ResolvedOperationCapability, error) {
	f.validateObservedTransitionCalls = append(
		f.validateObservedTransitionCalls,
		cloneObservedMembershipTransition(transition),
	)
	if f.validateObservedTransitionErr != nil {
		return ResolvedOperationCapability{}, f.validateObservedTransitionErr
	}
	if f.validateObservedTransitionCapability != nil {
		return *f.validateObservedTransitionCapability, nil
	}
	return resolveTestObservedTransitionCapability(transition, f.currentCapabilities())
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
	if operation.Phase == BackendOperationPhaseAbsent {
		return BackendOperation{Phase: BackendOperationPhaseAbsent}, nil
	}
	if operation.Attempt == 0 {
		operation.Attempt = attempt
	}
	return operation, nil
}

func (f *fakeMembershipAdapter) ValidateRequest(
	_ context.Context,
	_ GroupID,
	request MembershipRequest,
) error {
	f.validateRequestCalls = append(f.validateRequestCalls, cloneTestMembershipRequest(request))
	if f.validateRequestErr != nil {
		return f.validateRequestErr
	}
	return f.validateRequest(request)
}

func (f *fakeMembershipAdapter) SubmitOperation(_ context.Context, _ GroupID, request MembershipRequest) (BackendOperation, error) {
	request.BaseTopology = cloneTopology(request.BaseTopology)
	request.JoiningReplicas = slices.Clone(request.JoiningReplicas)
	request.NominatedReplicas = slices.Clone(request.NominatedReplicas)
	request.TargetMembership = cloneReplicaMemberships(request.TargetMembership)
	f.submitCalls = append(f.submitCalls, request)

	// Keep submission authoritative even when every caller-side preflight previously succeeded.
	if err := f.validateRequest(request); err != nil {
		return BackendOperation{
			ID:             request.ID,
			Attempt:        request.Attempt,
			TargetReplicas: request.TargetReplicas,
			Phase:          BackendOperationPhaseFailed,
			Failure: &OperationFailure{
				Classification: FailureClassificationTerminal,
				Reason:         "AtomicRequestRejected",
				Message:        err.Error(),
			},
		}, nil
	}

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
		if operation.Phase == BackendOperationPhaseAbsent {
			return BackendOperation{Phase: BackendOperationPhaseAbsent}, nil
		}
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

func (f *fakeMembershipAdapter) currentCapabilities() MembershipCapabilities {
	if f.capabilities == nil {
		return allTestMembershipCapabilities()
	}
	return cloneTestMembershipCapabilities(*f.capabilities)
}

func (f *fakeMembershipAdapter) validateRequest(request MembershipRequest) error {
	if request.ID == "" {
		return errors.New("membership request ID must not be empty")
	}
	if request.Attempt < 1 {
		return errors.New("membership request attempt must be positive")
	}
	if !servingVerificationTopologiesEqual(request.BaseTopology, f.topology) {
		return errors.New("membership request base topology is stale")
	}
	if err := validateCapabilityForTestRequest(request.Capability, request); err != nil {
		return err
	}
	if _, accepted := f.acceptedShapes[request.Capability.Shape]; accepted {
		return nil
	}
	for _, shape := range f.currentCapabilities().OperationShapes {
		if shape == request.Capability.Shape {
			return nil
		}
	}
	return fmt.Errorf(
		"membership operation shape %q is not supported: %w",
		request.Capability.Shape,
		ErrMembershipOperationUnsupported,
	)
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
	assert.Equal(t, OperationShapeFreshGrowth, result.Operation.Capability.Shape)
	assert.Len(t, adapter.validatePlanCalls, 1)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Keep the operation Pending until capacity or traffic preconditions are durable")
	input.Operation = result.Operation
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.False(t, result.OperationChanged)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Freeze exact joining identities in Submitting without calling the membership backend")
	prepared, err := coordinator.PrepareSubmission(
		context.Background(),
		input.GroupID,
		result.Operation,
		membershipReplicaIncarnations("replica-3", "replica-2"),
	)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseSubmitting, prepared.Phase)
	assert.Equal(t, membershipReplicaIncarnations("replica-2", "replica-3"), prepared.JoiningReplicas)
	assert.Len(t, adapter.validateRequestCalls, 1)
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
	assert.Len(t, adapter.validateRequestCalls, 2)
	require.Len(t, adapter.submitCalls, 1)
	assert.Equal(t, MembershipRequest{
		ID:              "operation-1",
		Attempt:         1,
		Intent:          OperationIntentGrow,
		Capability:      testOperationCapability(OperationShapeFreshGrowth),
		BaseTopology:    membershipTopology(1, 2),
		TargetReplicas:  4,
		JoiningReplicas: membershipReplicaIncarnations("replica-2", "replica-3"),
	}, adapter.submitCalls[0])
}

func TestOperationCoordinatorRechecksExactRequestImmediatelyBeforeSubmit(t *testing.T) {
	capabilities := MembershipCapabilities{OperationShapes: []OperationShape{
		OperationShapePlannedHighRankSuffixShrink,
	}}
	adapter := &fakeMembershipAdapter{
		capabilities:       &capabilities,
		topology:           membershipTopology(1, 2),
		validateRequestErr: errors.New("request capability is no longer valid for this group"),
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

	t.Log("Reject a persisted growth operation when its exact request fails preflight at the mutation boundary")
	result, err := coordinator.Submit(
		context.Background(),
		"group-0",
		durableOperation(OperationPhaseSubmitting, 4),
	)
	require.ErrorContains(t, err, "request capability is no longer valid")
	assert.Equal(t, OperationPhaseSubmitting, result.Operation.Phase)
	assert.Len(t, adapter.validateRequestCalls, 1)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorValidatesFrozenIdentitiesBeforeSubmitting(t *testing.T) {
	adapter := &fakeMembershipAdapter{
		topology:           membershipTopology(1, 2),
		validateRequestErr: errors.New("joining identities are unsupported"),
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	operation := durableOperation(OperationPhasePending, 4)

	t.Log("Freeze joining identities but refuse the request before returning a durable Submitting transition")
	prepared, err := coordinator.PrepareSubmission(
		context.Background(),
		"group-0",
		operation,
		membershipReplicaIncarnations("replica-2", "replica-3"),
	)
	require.ErrorContains(t, err, "joining identities are unsupported")
	assert.Nil(t, prepared)
	assert.Equal(t, OperationPhasePending, operation.Phase)
	assert.Empty(t, operation.JoiningReplicas)
	assert.Len(t, adapter.validateRequestCalls, 1)
	assert.Empty(t, adapter.submitCalls)
}

func TestMembershipSubmissionRemainsAtomicAfterSuccessfulPreflight(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
	request := membershipRequest(*durableOperation(OperationPhaseSubmitting, 4))

	t.Log("Preflight the complete request against its original base topology")
	require.NoError(t, adapter.ValidateRequest(context.Background(), "group-0", request))

	t.Log("Advance membership after preflight and reject the stale request atomically at submission")
	adapter.topology = membershipTopology(2, 2)
	result, err := adapter.SubmitOperation(context.Background(), "group-0", request)
	require.NoError(t, err)
	assert.Equal(t, BackendOperationPhaseFailed, result.Phase)
	require.NotNil(t, result.Failure)
	assert.Equal(t, FailureClassificationTerminal, result.Failure.Classification)
	assert.Contains(t, result.Failure.Message, "base topology is stale")
	assert.Empty(t, adapter.submittedRequests)
}

func TestOperationCoordinatorKeepsAcceptedPlanSupportStable(t *testing.T) {
	capabilities := allTestMembershipCapabilities()
	adapter := &fakeMembershipAdapter{
		capabilities: &capabilities,
		topology:     membershipTopology(1, 2),
	}
	coordinator := newTestOperationCoordinator(adapter, func() string { return "operation-1" })

	t.Log("Resolve fresh growth before making the operation durable")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationShapeFreshGrowth, result.Operation.Capability.Shape)

	t.Log("Remove freshly advertised growth while preserving support accepted by the durable operation")
	adapter.capabilities = &MembershipCapabilities{OperationShapes: []OperationShape{
		OperationShapePlannedHighRankSuffixShrink,
	}}
	prepared, err := coordinator.PrepareSubmission(
		context.Background(),
		"group-0",
		result.Operation,
		membershipReplicaIncarnations("replica-2", "replica-3"),
	)
	require.NoError(t, err)
	_, err = coordinator.Submit(context.Background(), "group-0", prepared)
	require.NoError(t, err)
	assert.Len(t, adapter.submitCalls, 1)
}

func TestOperationCoordinatorValidatesOperationShapeBeforePending(t *testing.T) {
	suffixOnly := MembershipCapabilities{OperationShapes: []OperationShape{
		OperationShapePlannedHighRankSuffixShrink,
	}}
	tests := []struct {
		name                string
		capabilities        MembershipCapabilities
		desiredReplicas     int32
		plan                *OperationPlan
		wantShape           OperationShape
		wantErr             string
		wantValidationCalls int
	}{
		{
			name:            "suffix-only adapter accepts suffix retirement",
			capabilities:    suffixOnly,
			desiredReplicas: 3,
			plan: &OperationPlan{
				ID:                "plan-1",
				Intent:            OperationIntentShrink,
				TargetReplicas:    3,
				NominatedReplicas: []ReplicaID{"replica-3"},
			},
			wantShape:           OperationShapePlannedHighRankSuffixShrink,
			wantValidationCalls: 1,
		},
		{
			name:            "suffix-only adapter rejects selected retirement",
			capabilities:    suffixOnly,
			desiredReplicas: 3,
			plan: &OperationPlan{
				ID:                "plan-1",
				Intent:            OperationIntentShrink,
				TargetReplicas:    3,
				NominatedReplicas: []ReplicaID{"replica-1"},
			},
			wantErr:             "does not support plan",
			wantValidationCalls: 1,
		},
		{
			name: "automatic growth is validated like an explicit plan",
			capabilities: MembershipCapabilities{OperationShapes: []OperationShape{
				OperationShapePlannedHighRankSuffixShrink,
			}},
			desiredReplicas:     5,
			wantErr:             "does not support plan",
			wantValidationCalls: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{
				capabilities: &tt.capabilities,
				topology:     membershipTopology(1, 4),
			}
			coordinator := newTestOperationCoordinator(adapter, func() string { return "operation-1" })

			t.Log("Resolve semantic support before creating a durable operation or allowing prework")
			result, err := coordinator.Step(context.Background(), OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: tt.desiredReplicas,
				Plan:            tt.plan,
			})
			assert.Len(t, adapter.validatePlanCalls, tt.wantValidationCalls)
			assert.Empty(t, adapter.submitCalls)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				assert.Nil(t, result.Operation)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, OperationPhasePending, result.Operation.Phase)
			assert.Equal(t, tt.wantShape, result.Operation.Capability.Shape)
		})
	}
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

func TestOperationCoordinatorRecordsDefinitiveSubmitRejection(t *testing.T) {
	failure := &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "RequestRejected",
		Message:        "the exact request cannot be applied",
	}
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(1, 2),
		submitResult: BackendOperation{
			ID:             "operation-1",
			Attempt:        1,
			TargetReplicas: 4,
			Phase:          BackendOperationPhaseFailed,
			Failure:        failure,
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

	t.Log("Persist a correlated terminal failure when the backend guarantees that submission did not mutate membership")
	result, err := coordinator.Submit(
		context.Background(),
		"group-0",
		durableOperation(OperationPhaseSubmitting, 4),
	)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseFailed, result.Operation.Phase)
	assert.Equal(t, failure, result.Operation.Failure)
	assert.True(t, result.OperationChanged)
	require.Len(t, adapter.submitCalls, 1)
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
	conflicting.JoiningReplicas = membershipReplicaIncarnations("replica-2")
	_, err = coordinator.Submit(context.Background(), "group-0", conflicting)
	require.ErrorContains(t, err, "conflicting payload")
}

func TestOperationCoordinatorFailsClosedWhenBaseTopologyChangesBeforeSubmit(t *testing.T) {
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
					membershipReplicaMembership("replica-0", "dp-0"),
					membershipReplicaMembership("replica-9", "dp-9"),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{topology: tt.topology}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
			operation := durableOperation(OperationPhaseSubmitting, 4)

			t.Log("Fail closed after proving the stale request cannot be submitted")
			result, err := coordinator.Submit(context.Background(), "group-0", operation)
			require.NoError(t, err)
			assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
			assert.Nil(t, result.Operation.Failure)
			assert.Nil(t, result.Operation.CompensationTopology)
			assert.True(t, result.OperationChanged)
			assert.Equal(t, []operationAttempt{{ID: operation.ID, Attempt: operation.Attempt}}, adapter.observeCalls)
			assert.Empty(t, adapter.submitCalls)
		})
	}
}

func TestOperationCoordinatorFailsClosedForSubmittingOperationAfterAbsentBaseDrift(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(2, 3)}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	operation := durableOperation(OperationPhaseSubmitting, 4)

	t.Log("Fail closed before requesting any prerequisite or backend replay")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Nil(t, result.Operation.Failure)
	assert.Nil(t, result.Operation.CompensationTopology)
	assert.True(t, result.OperationChanged)
	assert.False(t, result.SubmissionNeeded)
	assert.Equal(t, []operationAttempt{{ID: operation.ID, Attempt: operation.Attempt}}, adapter.observeCalls)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorFailsClosedBeforeGrowthIdentitiesAreFrozenAfterBaseDrift(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(2, 2)}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Operation:       durableOperation(OperationPhasePending, 4),
	}

	t.Log("Fail closed because the changed base has not been validated as a recovery transition")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Empty(t, result.Operation.JoiningReplicas)
	assert.Nil(t, result.Operation.CompensationTopology)
	assert.True(t, result.OperationChanged)

	t.Log("Turn definitive backend absence into durable compensation after restart")
	input.Operation = result.Operation
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseAborting, result.Operation.Phase)
	require.NotNil(t, result.Operation.CompensationTopology)
	assert.True(t, servingVerificationTopologiesEqual(input.Operation.BaseTopology, *result.Operation.CompensationTopology))
	require.NotNil(t, result.Operation.Failure)
	assert.Equal(t, "OperationAbsentAfterTopologyDrift", result.Operation.Failure.Reason)
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
	assert.Equal(t, []ReplicaID{"replica-0", "replica-1"}, topologyReplicaIDs(result.Operation.BaseTopology))
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
			ID:                "operation-1",
			BackendID:         testBackendOperationID,
			TargetReplicas:    4,
			Phase:             BackendOperationPhaseCommitted,
			CommittedTopology: topologyPointer(membershipTopology(2, 4)),
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
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, membershipTopology(2, 4), *result.Operation.CommittedTopology)

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

func TestOperationCoordinatorFreezesExactServingVerificationTargetAtCommit(t *testing.T) {
	committedTopology := membershipTopology(2, 4)
	committedTopology.Replicas[3].NativeMembers = []NativeMemberID{"native-3-a", "native-3-b"}
	adapter := &fakeMembershipAdapter{
		topology: committedTopology,
		operation: BackendOperation{
			ID:                "operation-1",
			BackendID:         testBackendOperationID,
			TargetReplicas:    4,
			Phase:             BackendOperationPhaseCommitted,
			CommittedTopology: topologyPointer(committedTopology),
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	operation := durableOperation(OperationPhaseCommitting, 4)
	operation.Capability.VerificationRequirement = ServingVerificationRequired
	operation.Capability.TrafficRequirement = ReconfigurationTrafficQuiesceGroup

	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  1,
		DesiredReplicas: 4,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	require.NotNil(t, result.Operation.CommittedTopology)
	require.NotNil(t, result.Operation.ServingVerificationTarget)
	assert.Equal(t, committedTopology, *result.Operation.CommittedTopology)
	assert.Equal(t, committedTopology, *result.Operation.ServingVerificationTarget)
	assert.Equal(t, int32(1), result.Operation.ServingVerificationAttempt)

	committedTopology.Replicas[3].NativeMembers[0] = "mutated-after-observation"
	assert.NotEqual(t, committedTopology, *result.Operation.CommittedTopology)
	assert.Equal(t, *result.Operation.CommittedTopology, *result.Operation.ServingVerificationTarget)
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
			prepared, prepareErr := coordinator.PrepareSubmission(
				context.Background(),
				input.GroupID,
				result.Operation,
				nil,
			)
			if tt.wantRetryPrepared {
				require.NoError(t, prepareErr)
				assert.Equal(t, OperationPhaseSubmitting, prepared.Phase)
				assert.Equal(t, "operation-1", prepared.ID)
				assert.Equal(t, int32(2), prepared.Attempt)
				assert.Nil(t, prepared.Failure)
				assert.Equal(t, membershipReplicaIncarnations("replica-2", "replica-3"), prepared.JoiningReplicas)
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

func TestOperationCoordinatorFailsClosedOnPhaseContradictoryBackendOperation(t *testing.T) {
	tests := []struct {
		name        string
		observation BackendOperation
	}{
		{
			name: "accepted result carries committed topology",
			observation: BackendOperation{
				Phase:             BackendOperationPhaseAccepted,
				CommittedTopology: topologyPointer(membershipTopology(2, 4)),
			},
		},
		{
			name: "committing result carries failure",
			observation: BackendOperation{
				Phase: BackendOperationPhaseCommitting,
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "ContradictoryState",
				},
			},
		},
		{
			name: "committed result also carries failure",
			observation: BackendOperation{
				Phase:             BackendOperationPhaseCommitted,
				CommittedTopology: topologyPointer(membershipTopology(2, 4)),
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "ContradictoryState",
				},
			},
		},
		{
			name: "failed result also carries committed topology",
			observation: BackendOperation{
				Phase:             BackendOperationPhaseFailed,
				CommittedTopology: topologyPointer(membershipTopology(2, 4)),
				Failure: &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "ContradictoryState",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			operation := durableOperation(OperationPhaseCommitting, 4)
			observation := cloneBackendOperation(tt.observation)
			observation.ID = operation.ID
			observation.Attempt = operation.Attempt
			observation.TargetReplicas = operation.TargetReplicas
			adapter := &fakeMembershipAdapter{
				topology:  membershipTopology(1, 2),
				operation: observation,
			}
			coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

			result, err := coordinator.Step(context.Background(), OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  1,
				DesiredReplicas: 4,
				Operation:       operation,
			})
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
			assert.Nil(t, result.Operation.CommittedTopology)
			assert.Nil(t, result.Operation.Failure)
			assert.Empty(t, adapter.submitCalls)
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
		ID:                "operation-1",
		BackendID:         testBackendOperationID,
		TargetReplicas:    4,
		Phase:             BackendOperationPhaseCommitted,
		CommittedTopology: topologyPointer(membershipTopology(2, 4)),
	}
	input.Operation = result.Operation
	result, err = coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, membershipTopology(2, 4), *result.Operation.CommittedTopology)
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
			operation.CommittedTopology = topologyPointer(membershipTopology(2, 4))
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
	operation.CommittedTopology = topologyPointer(membershipTopology(2, 4))
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(1, 2),
		operation: BackendOperation{
			ID:                operation.ID,
			Attempt:           operation.Attempt,
			BackendID:         operation.BackendOperationID,
			TargetReplicas:    operation.TargetReplicas,
			Phase:             BackendOperationPhaseCommitted,
			CommittedTopology: cloneTopologyPointer(operation.CommittedTopology),
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
	wrongIdentityTopology.Replicas[3] = membershipReplicaMembership("replica-9", "dp-9")
	laterUnexpectedTopology := membershipTopology(3, 3)
	laterUnexpectedTopology.Replicas[2] = membershipReplicaMembership("replica-9", "dp-9")

	tests := []struct {
		name                 string
		topology             MembershipTopology
		committedTopology    *MembershipTopology
		backendTarget        int32
		wantPhase            OperationPhase
		wantCommitProof      bool
		wantTopologyReplicas int32
	}{
		{
			name:                 "exact topology commits",
			topology:             membershipTopology(2, 4),
			committedTopology:    topologyPointer(membershipTopology(2, 4)),
			backendTarget:        4,
			wantPhase:            OperationPhaseCommitted,
			wantCommitProof:      true,
			wantTopologyReplicas: 4,
		},
		{
			name:                 "stale topology keeps committing",
			topology:             membershipTopology(1, 2),
			committedTopology:    topologyPointer(membershipTopology(2, 4)),
			backendTarget:        4,
			wantPhase:            OperationPhaseCommitting,
			wantTopologyReplicas: 2,
		},
		{
			name:                 "wrong topology cardinality is unknown",
			topology:             membershipTopology(2, 3),
			committedTopology:    topologyPointer(membershipTopology(2, 4)),
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantTopologyReplicas: 3,
		},
		{
			name:                 "wrong replica identity at the expected cardinality is unknown",
			topology:             wrongIdentityTopology,
			committedTopology:    topologyPointer(membershipTopology(2, 4)),
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantTopologyReplicas: 4,
		},
		{
			name:                 "later topology with unexpected identity is unknown without commit proof",
			topology:             laterUnexpectedTopology,
			committedTopology:    topologyPointer(membershipTopology(2, 4)),
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantTopologyReplicas: 3,
		},
		{
			name:                 "later topology generation is unknown",
			topology:             membershipTopology(3, 4),
			committedTopology:    topologyPointer(membershipTopology(2, 4)),
			backendTarget:        4,
			wantPhase:            OperationPhaseUnknown,
			wantCommitProof:      true,
			wantTopologyReplicas: 4,
		},
		{
			name:                 "mismatched backend target is unknown",
			topology:             membershipTopology(2, 4),
			committedTopology:    topologyPointer(membershipTopology(2, 4)),
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
					ID:                "operation-1",
					TargetReplicas:    tt.backendTarget,
					Phase:             BackendOperationPhaseCommitted,
					CommittedTopology: cloneTopologyPointer(tt.committedTopology),
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
			if tt.wantCommitProof {
				assert.Equal(t, tt.committedTopology, result.Operation.CommittedTopology)
			} else {
				assert.Nil(t, result.Operation.CommittedTopology)
			}
			assert.Equal(t, tt.wantTopologyReplicas, result.Topology.ReplicaCount())
		})
	}
}

func TestOperationCoordinatorPreservesCommitProofWhenTopologyObservationSkipsToSurvivors(t *testing.T) {
	committedTopology := membershipTopology(2, 4)
	adapter := &fakeMembershipAdapter{
		topology: membershipTopology(3, 3),
		operation: BackendOperation{
			ID:                "operation-1",
			TargetReplicas:    4,
			Phase:             BackendOperationPhaseCommitted,
			CommittedTopology: topologyPointer(committedTopology),
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
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, committedTopology, *result.Operation.CommittedTopology)
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
	require.NotNil(t, stable.Operation.CommittedTopology)
	assert.Equal(t, committedTopology, *stable.Operation.CommittedTopology)
	assert.False(t, stable.OperationChanged)
}

func TestOperationCoordinatorAdoptsRecoveryAgainstExactPreviousCommittedTopology(t *testing.T) {
	previousTopology := membershipTopology(2, 4)
	previousTopology.Replicas[0].NativeMembers = []NativeMemberID{"previous-native-0-a", "previous-native-0-b"}
	previous := durableOperation(OperationPhaseUnknown, 4)
	previous.BaseTopology.Replicas = cloneReplicaMemberships(previousTopology.Replicas[:2])
	previous.CommittedTopology = topologyPointer(previousTopology)

	observedTopology := MembershipTopology{
		Generation: 3,
		Replicas:   cloneReplicaMemberships(previousTopology.Replicas[:3]),
	}
	capability := testOperationCapability(OperationShapeSurvivorReduction)
	capability.VerificationRequirement = ServingVerificationRequired
	adapter := &fakeMembershipAdapter{
		topology:                             observedTopology,
		validateObservedTransitionCapability: &capability,
	}
	coordinator := newTestOperationCoordinator(adapter, func() string { return "adopt-operation-1" })
	plan := &OperationPlan{
		ID:                "adopt-plan-1",
		Intent:            OperationIntentRecover,
		TargetReplicas:    3,
		NominatedReplicas: []ReplicaID{"replica-3"},
	}

	t.Log("Validate adoption against the exact durable logical-to-native mapping, not reconstructed replica IDs")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Plan:            plan,
		Operation:       previous,
	})
	require.NoError(t, err)
	require.Len(t, adapter.validateObservedTransitionCalls, 1)
	transition := adapter.validateObservedTransitionCalls[0]
	assert.Equal(t, previousTopology, transition.PreviousTopology)
	assert.Equal(t, observedTopology, transition.ObservedTopology)

	require.NotNil(t, result.Operation)
	assert.True(t, result.Operation.Adopted)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, observedTopology, *result.Operation.CommittedTopology)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorRejectsObservedAdoptionWithRestoredReplicas(t *testing.T) {
	previousTopology := membershipTopology(2, 4)
	previous := durableOperation(OperationPhaseUnknown, 4)
	previous.CommittedTopology = topologyPointer(previousTopology)
	adapter := &fakeMembershipAdapter{topology: membershipTopology(3, 3)}
	coordinator := newTestOperationCoordinator(adapter, func() string { return "unused-operation" })

	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 3,
		Plan: &OperationPlan{
			ID:                 "invalid-adoption-plan",
			Intent:             OperationIntentRecover,
			TargetReplicas:     3,
			NominatedReplicas:  []ReplicaID{"replica-3"},
			RestoredMembership: membershipReplicaNativeMemberships("replica-3"),
		},
		Operation: previous,
	})
	require.ErrorContains(t, err, "restored replicas are only valid for a recovery expansion")
	assert.Equal(t, previous, result.Operation)
	assert.Empty(t, adapter.validateObservedTransitionCalls)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorSkipsServingVerificationForEmptyCommittedTopology(t *testing.T) {
	baseTopology := membershipTopology(1, 2)
	emptyTopology := membershipTopology(2, 0)
	operation := &Operation{
		ID:      "operation-1",
		Attempt: 1,
		PlanID:  "retire-all-plan",
		Intent:  OperationIntentRetire,
		Capability: ResolvedOperationCapability{
			Shape:                   OperationShapeFullRetirement,
			TrafficRequirement:      ReconfigurationTrafficQuiesceGroup,
			VerificationRequirement: ServingVerificationRequired,
		},
		SpecGeneration:     2,
		BaseTopology:       baseTopology,
		TargetReplicas:     0,
		NominatedReplicas:  topologyReplicaIDs(baseTopology),
		Phase:              OperationPhaseCommitting,
		StartedAt:          testNow.Add(-time.Minute),
		LastTransitionTime: testNow.Add(-time.Second),
	}
	adapter := &fakeMembershipAdapter{
		topology: emptyTopology,
		operation: BackendOperation{
			ID:                operation.ID,
			Attempt:           operation.Attempt,
			TargetReplicas:    operation.TargetReplicas,
			Phase:             BackendOperationPhaseCommitted,
			CommittedTopology: topologyPointer(emptyTopology),
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 0,
		Operation:       operation,
	})
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int32(0), result.Operation.ServingVerificationAttempt)
	assert.Nil(t, result.Operation.ServingVerificationTarget)
	assert.Nil(t, result.Operation.ServingVerificationProof)
}

func TestOperationCoordinatorSkipsServingVerificationForEmptyAdoptedTopology(t *testing.T) {
	previousTopology := membershipTopology(2, 2)
	previous := &Operation{
		ID:                 "previous-operation",
		Attempt:            1,
		PlanID:             "previous-plan",
		Intent:             OperationIntentShrink,
		Capability:         testOperationCapability(OperationShapePlannedHighRankSuffixShrink),
		SpecGeneration:     2,
		BaseTopology:       membershipTopology(1, 3),
		TargetReplicas:     2,
		NominatedReplicas:  []ReplicaID{"replica-2"},
		Phase:              OperationPhaseUnknown,
		CommittedTopology:  topologyPointer(previousTopology),
		StartedAt:          testNow.Add(-time.Minute),
		LastTransitionTime: testNow.Add(-time.Second),
	}
	emptyTopology := membershipTopology(3, 0)
	capability := ResolvedOperationCapability{
		Shape:                   OperationShapeSurvivorReduction,
		TrafficRequirement:      ReconfigurationTrafficKeepServing,
		VerificationRequirement: ServingVerificationRequired,
	}
	adapter := &fakeMembershipAdapter{
		topology:                             emptyTopology,
		validateObservedTransitionCapability: &capability,
	}
	coordinator := newTestOperationCoordinator(adapter, func() string { return "adopt-operation-1" })

	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  3,
		DesiredReplicas: 0,
		Plan: &OperationPlan{
			ID:                "adopt-empty-plan",
			Intent:            OperationIntentRecover,
			TargetReplicas:    0,
			NominatedReplicas: topologyReplicaIDs(previousTopology),
		},
		Operation: previous,
	})
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.True(t, result.Operation.Adopted)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	assert.Equal(t, int32(0), result.Operation.ServingVerificationAttempt)
	assert.Nil(t, result.Operation.ServingVerificationTarget)
	assert.Nil(t, result.Operation.ServingVerificationProof)
}

func TestOperationCoordinatorRejectsCommitThatRemovedWrongReplica(t *testing.T) {
	committedTopology := MembershipTopology{
		Generation: 2,
		Replicas: []ReplicaMembership{
			membershipReplicaMembership("replica-0", "dp-0"),
			membershipReplicaMembership("replica-1", "dp-1"),
		},
	}
	adapter := &fakeMembershipAdapter{
		topology: committedTopology,
		operation: BackendOperation{
			ID:                "operation-1",
			TargetReplicas:    2,
			Phase:             BackendOperationPhaseCommitted,
			CommittedTopology: topologyPointer(committedTopology),
		},
	}
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)
	operation := &Operation{
		ID:                 "operation-1",
		Attempt:            1,
		PlanID:             "plan-1",
		Intent:             OperationIntentShrink,
		Capability:         testOperationCapability(OperationShapePlannedSelectedRetirement),
		SpecGeneration:     1,
		BaseTopology:       membershipTopology(1, 4),
		TargetReplicas:     2,
		NominatedReplicas:  []ReplicaID{"replica-1", "replica-3"},
		Phase:              OperationPhaseCommitting,
		StartedAt:          testNow.Add(-time.Minute),
		LastTransitionTime: testNow.Add(-time.Second),
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

func TestOperationCoordinatorPersistsAndReplaysExactNativeMemberRemap(t *testing.T) {
	targetMembership := []ReplicaMembership{
		membershipReplicaMembership("replica-1", "dp-new-1-b", "dp-new-1-a"),
		membershipReplicaMembership("replica-0", "dp-new-0"),
	}
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
	coordinator := newTestOperationCoordinator(adapter, func() string { return "remap-operation-1" })
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Plan: &OperationPlan{
			ID:               "remap-plan-1",
			Intent:           OperationIntentRecover,
			TargetReplicas:   2,
			TargetMembership: targetMembership,
		},
	}

	t.Log("Persist a canonical copy of the exact desired native-member mapping")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationShapeNativeMemberRemapping, result.Operation.Capability.Shape)
	wantTarget := []ReplicaMembership{
		membershipReplicaMembership("replica-0", "dp-new-0"),
		membershipReplicaMembership("replica-1", "dp-new-1-a", "dp-new-1-b"),
	}
	assert.Equal(t, wantTarget, result.Operation.TargetMembership)

	t.Log("Mutate caller-owned input without changing the durable operation")
	targetMembership[0].NativeMembers[0] = "mutated"
	assert.Equal(t, wantTarget, result.Operation.TargetMembership)

	t.Log("Freeze and preflight the exact remap request")
	prepared, err := coordinator.PrepareSubmission(context.Background(), input.GroupID, result.Operation, nil)
	require.NoError(t, err)
	require.Len(t, adapter.validateRequestCalls, 1)
	assert.Equal(t, wantTarget, adapter.validateRequestCalls[0].TargetMembership)

	t.Log("Recover after restart and accept only the exact committed mapping")
	adapter.topology = MembershipTopology{Generation: 2, Replicas: cloneReplicaMemberships(wantTarget)}
	adapter.operation = BackendOperation{
		ID:                prepared.ID,
		Attempt:           prepared.Attempt,
		TargetReplicas:    prepared.TargetReplicas,
		Phase:             BackendOperationPhaseCommitted,
		CommittedTopology: topologyPointer(adapter.topology),
	}
	restarted := newTestOperationCoordinator(adapter, emptyOperationID)
	input.Plan = &OperationPlan{
		ID:               "remap-plan-1",
		Intent:           OperationIntentRecover,
		TargetReplicas:   2,
		TargetMembership: cloneReplicaMemberships(wantTarget),
	}
	input.Operation = prepared
	result, err = restarted.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
	require.NotNil(t, result.Operation.CommittedTopology)
	assert.Equal(t, adapter.topology, *result.Operation.CommittedTopology)
	assert.Equal(t, wantTarget, result.Operation.TargetMembership)
}

func TestOperationCoordinatorRejectsCommittedNativeMemberRemapWithWrongMapping(t *testing.T) {
	targetMembership := []ReplicaMembership{
		membershipReplicaMembership("replica-0", "dp-new-0"),
		membershipReplicaMembership("replica-1", "dp-new-1"),
	}
	operation := &Operation{
		ID:                 "remap-operation-1",
		Attempt:            1,
		PlanID:             "remap-plan-1",
		Intent:             OperationIntentRecover,
		Capability:         testOperationCapability(OperationShapeNativeMemberRemapping),
		SpecGeneration:     2,
		BaseTopology:       membershipTopology(1, 2),
		TargetReplicas:     2,
		TargetMembership:   targetMembership,
		Phase:              OperationPhaseSubmitting,
		StartedAt:          testNow.Add(-time.Minute),
		LastTransitionTime: testNow.Add(-time.Second),
	}
	adapter := &fakeMembershipAdapter{
		topology: MembershipTopology{
			Generation: 2,
			Replicas: []ReplicaMembership{
				membershipReplicaMembership("replica-0", "dp-new-0"),
				membershipReplicaMembership("replica-1", "dp-wrong-1"),
			},
		},
		operation: BackendOperation{
			ID:             operation.ID,
			Attempt:        operation.Attempt,
			TargetReplicas: operation.TargetReplicas,
			Phase:          BackendOperationPhaseCommitted,
		},
	}
	adapter.operation.CommittedTopology = topologyPointer(adapter.topology)
	coordinator := newTestOperationCoordinator(adapter, emptyOperationID)

	t.Log("Fail closed when restart observation has the right logical replicas but a different native mapping")
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Operation:       operation,
	})
	require.NoError(t, err)
	assert.Equal(t, OperationPhaseUnknown, result.Operation.Phase)
	assert.Nil(t, result.Operation.CommittedTopology)
}

func TestOperationCoordinatorKeepsDurableRemapWhenCandidatePayloadChanges(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 2)}
	coordinator := newTestOperationCoordinator(adapter, func() string { return "remap-operation-1" })
	plan := &OperationPlan{
		ID:             "remap-plan-1",
		Intent:         OperationIntentRecover,
		TargetReplicas: 2,
		TargetMembership: []ReplicaMembership{
			membershipReplicaMembership("replica-0", "dp-new-0"),
			membershipReplicaMembership("replica-1", "dp-new-1"),
		},
	}
	result, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Plan:            plan,
	})
	require.NoError(t, err)
	observations := adapter.topologyCalls

	t.Log("Keep the durable mapping authoritative while the candidate cannot be consumed")
	conflictingPlan := *plan
	conflictingPlan.TargetMembership = cloneReplicaMemberships(plan.TargetMembership)
	conflictingPlan.TargetMembership[1].NativeMembers[0] = "dp-other-1"
	stable, err := coordinator.Step(context.Background(), OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 2,
		Plan:            &conflictingPlan,
		Operation:       result.Operation,
	})
	require.NoError(t, err)
	assert.Equal(t, result.Operation, stable.Operation)
	assert.Equal(t, observations+1, adapter.topologyCalls)
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
		{
			name: "stable recovery omits target membership",
			plan: OperationPlan{Intent: OperationIntentRecover, TargetReplicas: 2},
		},
		{
			name: "stable recovery preserves the existing native mapping",
			plan: OperationPlan{
				Intent:           OperationIntentRecover,
				TargetReplicas:   2,
				TargetMembership: membershipTopology(1, 2).Replicas,
			},
		},
		{
			name: "stable recovery changes a logical replica identity",
			plan: OperationPlan{
				Intent:         OperationIntentRecover,
				TargetReplicas: 2,
				TargetMembership: []ReplicaMembership{
					membershipReplicaMembership("replica-0", "dp-new-0"),
					membershipReplicaMembership("replica-9", "dp-new-9"),
				},
			},
		},
		{
			name: "stable recovery duplicates a native member",
			plan: OperationPlan{
				Intent:         OperationIntentRecover,
				TargetReplicas: 2,
				TargetMembership: []ReplicaMembership{
					membershipReplicaMembership("replica-0", "dp-new"),
					membershipReplicaMembership("replica-1", "dp-new"),
				},
			},
		},
		{
			name: "growth carries target membership",
			plan: OperationPlan{
				Intent:         OperationIntentGrow,
				TargetReplicas: 3,
				TargetMembership: []ReplicaMembership{
					membershipReplicaMembership("replica-0", "dp-new-0"),
				},
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

func TestRuntimeIdentityContinuityRejectsIncarnationAliasing(t *testing.T) {
	baseTopology := membershipTopology(1, 2)
	replacementTopology := cloneTopology(baseTopology)
	replacementTopology.Generation = 2
	replacementTopology.Replicas[1].Incarnation.CapacityRefs[0].UID = operationRestorationReplacementUID
	replacementTopology.Replicas[1].Incarnation.RuntimeID = operationRestorationReplacementRun
	reusedRuntimeTopology := cloneTopology(replacementTopology)
	reusedRuntimeTopology.Replicas[1].Incarnation.RuntimeID =
		baseTopology.Replicas[1].Incarnation.RuntimeID

	t.Log("Reject an explicit fixed-slot replacement that gives a new Pod the old process identity")
	err := validatePlan(OperationPlan{
		ID:               "replacement-plan",
		Intent:           OperationIntentRecover,
		TargetReplicas:   2,
		TargetMembership: reusedRuntimeTopology.Replicas,
	}, baseTopology)
	require.ErrorContains(t, err, "runtime identity continuity")

	t.Log("Reject the same alias if it appears in a restored durable operation")
	operation := Operation{
		ID:                 "replacement-operation",
		Attempt:            1,
		PlanID:             "replacement-plan",
		Intent:             OperationIntentRecover,
		Capability:         testOperationCapability(OperationShapeFixedSlotReplacement),
		SpecGeneration:     2,
		BaseTopology:       baseTopology,
		TargetReplicas:     2,
		TargetMembership:   reusedRuntimeTopology.Replicas,
		Phase:              OperationPhasePending,
		StartedAt:          testNow,
		LastTransitionTime: testNow,
	}
	err = validateOperation(operation)
	require.ErrorContains(t, err, "runtime identity continuity")

	t.Log("Reject an externally observed cardinal recovery with the same alias")
	err = validateObservedMembershipTransition(ObservedMembershipTransition{
		PreviousTopology: baseTopology,
		ObservedTopology: reusedRuntimeTopology,
		Plan: OperationPlan{
			ID:               "replacement-plan",
			Intent:           OperationIntentRecover,
			TargetReplicas:   2,
			TargetMembership: reusedRuntimeTopology.Replicas,
		},
	})
	require.ErrorContains(t, err, "runtime identity continuity")

	t.Log("Accept the replacement when the new Pod also has a new runtime identity")
	require.NoError(t, validatePlan(OperationPlan{
		ID:               "replacement-plan",
		Intent:           OperationIntentRecover,
		TargetReplicas:   2,
		TargetMembership: replacementTopology.Replicas,
	}, baseTopology))
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
					membershipReplicaMembership("replica-0", "dp-0"),
					membershipReplicaMembership("replica-0", "dp-1"),
				},
			},
		},
		{
			name: "missing native member",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					{Incarnation: membershipReplicaIncarnation("replica-0")},
				},
			},
		},
		{
			name: "empty native member ID",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					membershipReplicaMembership("replica-0", ""),
				},
			},
		},
		{
			name: "duplicate native member ID",
			topology: MembershipTopology{
				Generation: 1,
				Replicas: []ReplicaMembership{
					membershipReplicaMembership("replica-0", "dp-0"),
					membershipReplicaMembership("replica-1", "dp-0"),
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
	committedProofWithoutJoiners := durableOperation(OperationPhaseUnknown, 4)
	committedProofWithoutJoiners.JoiningReplicas = nil
	committedProofWithoutJoiners.CommittedTopology = topologyPointer(membershipTopology(2, 4))
	abortingWithoutFailure := durableOperation(OperationPhaseAborting, 4)
	abortingWithRetryableFailure := durableOperation(OperationPhaseAborting, 4)
	abortingWithRetryableFailure.Failure = &OperationFailure{
		Classification: FailureClassificationRetryable,
		Reason:         "Transient",
	}

	tests := []struct {
		name              string
		input             OperationInput
		wantTopologyCalls int
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
			wantTopologyCalls: 1,
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
			assert.Equal(t, tt.wantTopologyCalls, adapter.topologyCalls)
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
		replicaID := ReplicaID(fmt.Sprintf("replica-%d", i))
		topology.Replicas[i] = ReplicaMembership{
			Incarnation:   membershipReplicaIncarnation(replicaID),
			NativeMembers: []NativeMemberID{NativeMemberID(fmt.Sprintf("dp-%d", i))},
		}
	}
	return topology
}

func membershipReplicaIncarnation(replicaID ReplicaID) ReplicaIncarnation {
	podName := "pod-" + string(replicaID[len("replica-"):])
	return ReplicaIncarnation{
		ReplicaID: replicaID,
		SlotID:    "slot-" + CapacitySlotID(replicaID),
		CapacityRefs: []CapacityRef{{
			Namespace: "test",
			Name:      podName,
			UID:       PodUID("uid-" + podName),
		}},
		RuntimeID: "runtime-" + RuntimeIncarnationID(replicaID),
	}
}

func membershipReplicaIncarnations(replicaIDs ...ReplicaID) []ReplicaIncarnation {
	incarnations := make([]ReplicaIncarnation, len(replicaIDs))
	for i, replicaID := range replicaIDs {
		incarnations[i] = membershipReplicaIncarnation(replicaID)
	}
	return incarnations
}

func membershipReplicaNativeMemberships(replicaIDs ...ReplicaID) []ReplicaNativeMembership {
	memberships := make([]ReplicaNativeMembership, len(replicaIDs))
	for i, replicaID := range replicaIDs {
		memberships[i] = ReplicaNativeMembership{
			ReplicaID:     replicaID,
			SlotID:        membershipReplicaIncarnation(replicaID).SlotID,
			NativeMembers: []NativeMemberID{"dp-" + NativeMemberID(replicaID[len("replica-"):])},
		}
	}
	return memberships
}

func membershipReplicaMembership(replicaID ReplicaID, nativeMembers ...NativeMemberID) ReplicaMembership {
	return ReplicaMembership{
		Incarnation:   membershipReplicaIncarnation(replicaID),
		NativeMembers: slices.Clone(nativeMembers),
	}
}

func durableOperation(
	phase OperationPhase,
	targetReplicas int32,
) *Operation {
	operation := &Operation{
		ID:                 "operation-1",
		Attempt:            1,
		Intent:             OperationIntentGrow,
		Capability:         testOperationCapability(OperationShapeFreshGrowth),
		SpecGeneration:     1,
		BaseTopology:       membershipTopology(1, 2),
		TargetReplicas:     targetReplicas,
		Phase:              phase,
		StartedAt:          testNow.Add(-time.Minute),
		LastTransitionTime: testNow.Add(-time.Second),
	}

	// Every submitted growth record freezes the exact logical replica identities being added.
	if targetReplicas == 4 && phase != OperationPhasePending {
		operation.JoiningReplicas = membershipReplicaIncarnations("replica-2", "replica-3")
	}
	return operation
}

func cloneBackendOperation(operation BackendOperation) BackendOperation {
	operation.CommittedTopology = cloneTopologyPointer(operation.CommittedTopology)
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
	return MembershipCapabilities{OperationShapes: []OperationShape{
		OperationShapeFreshGrowth,
		OperationShapePlannedHighRankSuffixShrink,
		OperationShapePlannedSelectedRetirement,
		OperationShapeSurvivorReduction,
		OperationShapeReplacementRestoration,
		OperationShapeFixedSlotReplacement,
		OperationShapeNativeMemberRemapping,
		OperationShapeFullRetirement,
	}}
}

func testOperationCapability(shape OperationShape) ResolvedOperationCapability {
	return ResolvedOperationCapability{
		Shape:                   shape,
		TrafficRequirement:      ReconfigurationTrafficKeepServing,
		VerificationRequirement: ServingVerificationNotRequired,
	}
}

func cloneTestMembershipCapabilities(capabilities MembershipCapabilities) MembershipCapabilities {
	return MembershipCapabilities{OperationShapes: slices.Clone(capabilities.OperationShapes)}
}

func cloneTestMembershipRequest(request MembershipRequest) MembershipRequest {
	request.BaseTopology = cloneTopology(request.BaseTopology)
	request.JoiningReplicas = cloneReplicaIncarnations(request.JoiningReplicas)
	request.RestoredMembership = cloneReplicaNativeMemberships(request.RestoredMembership)
	request.NominatedReplicas = slices.Clone(request.NominatedReplicas)
	request.TargetMembership = cloneReplicaMemberships(request.TargetMembership)
	return request
}

func resolveTestOperationCapability(
	plan OperationPlan,
	topology MembershipTopology,
	capabilities MembershipCapabilities,
) (ResolvedOperationCapability, error) {
	var shapes []OperationShape
	switch plan.Intent {
	case OperationIntentGrow:
		shapes = []OperationShape{OperationShapeFreshGrowth}
	case OperationIntentShrink:
		if isTestSuffix(topologyReplicaIDs(topology), plan.NominatedReplicas) {
			shapes = append(shapes, OperationShapePlannedHighRankSuffixShrink)
		}
		shapes = append(shapes, OperationShapePlannedSelectedRetirement)
	case OperationIntentRecover:
		switch {
		case plan.TargetReplicas < topology.ReplicaCount():
			shapes = []OperationShape{OperationShapeSurvivorReduction}
		case plan.TargetReplicas > topology.ReplicaCount():
			shapes = []OperationShape{OperationShapeReplacementRestoration}
		default:
			shapes = []OperationShape{OperationShapeNativeMemberRemapping}
		}
	case OperationIntentRetire:
		shapes = []OperationShape{OperationShapeFullRetirement}
	default:
		return ResolvedOperationCapability{}, fmt.Errorf("invalid operation intent %q", plan.Intent)
	}

	for _, shape := range shapes {
		for _, supportedShape := range capabilities.OperationShapes {
			if supportedShape == shape {
				return testOperationCapability(shape), nil
			}
		}
	}
	return ResolvedOperationCapability{}, fmt.Errorf(
		"membership operation does not support plan intent %q and nominees %v: %w",
		plan.Intent,
		plan.NominatedReplicas,
		ErrMembershipOperationUnsupported,
	)
}

func resolveTestObservedTransitionCapability(
	transition ObservedMembershipTransition,
	capabilities MembershipCapabilities,
) (ResolvedOperationCapability, error) {
	if err := validateObservedMembershipTransition(transition); err != nil {
		return ResolvedOperationCapability{}, err
	}
	shape := OperationShapeSurvivorReduction
	if transition.Plan.TargetReplicas == transition.PreviousTopology.ReplicaCount() {
		shape = OperationShapeNativeMemberRemapping
	}
	for _, supportedShape := range capabilities.OperationShapes {
		if supportedShape == shape {
			return testOperationCapability(shape), nil
		}
	}
	return ResolvedOperationCapability{}, fmt.Errorf(
		"membership operation shape %q is not supported: %w",
		shape,
		ErrMembershipOperationUnsupported,
	)
}

func validateCapabilityForTestRequest(capability ResolvedOperationCapability, request MembershipRequest) error {
	operation := Operation{
		Intent:             request.Intent,
		Capability:         capability,
		BaseTopology:       cloneTopology(request.BaseTopology),
		TargetReplicas:     request.TargetReplicas,
		JoiningReplicas:    cloneReplicaIncarnations(request.JoiningReplicas),
		RestoredMembership: cloneReplicaNativeMemberships(request.RestoredMembership),
		NominatedReplicas:  slices.Clone(request.NominatedReplicas),
		TargetMembership:   cloneReplicaMemberships(request.TargetMembership),
		Phase:              OperationPhaseSubmitting,
	}
	if err := validateCapabilityForOperation(capability, operation); err != nil {
		return err
	}
	if err := validateOperationReplicaGeometry(operation); err != nil {
		return err
	}
	if capability.Shape == OperationShapePlannedHighRankSuffixShrink &&
		!isTestSuffix(topologyReplicaIDs(request.BaseTopology), request.NominatedReplicas) {
		return errors.New("planned suffix shrink request nominates a non-suffix replica")
	}
	return nil
}

func isTestSuffix(baseReplicas, nominatedReplicas []ReplicaID) bool {
	base := normalizeReplicaIDs(baseReplicas)
	nominated := normalizeReplicaIDs(nominatedReplicas)
	if len(nominated) > len(base) {
		return false
	}
	return sameReplicaIDs(nominated, base[len(base)-len(nominated):])
}

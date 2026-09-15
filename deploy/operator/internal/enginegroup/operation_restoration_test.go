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

const (
	operationRestorationAdoptedID      = "adopted-restoration"
	operationRestorationTestID         = "operation-1"
	operationRestorationReplacementUID = "replacement-pod-uid"
	operationRestorationReplacementRun = "replacement-runtime"
)

func TestOperationCoordinatorPersistsExactReplacementRestorationIdentity(t *testing.T) {
	adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 3)}
	coordinator := newTestOperationCoordinator(adapter, func() string { return "restore-operation-1" })
	plan := &OperationPlan{
		ID:                 "restore-plan-1",
		Intent:             OperationIntentRecover,
		TargetReplicas:     4,
		RestoredMembership: membershipReplicaNativeMemberships("replica-3"),
	}
	input := OperationInput{
		GroupID:         "group-0",
		SpecGeneration:  2,
		DesiredReplicas: 4,
		Plan:            plan,
	}

	t.Log("Persist the exact stable identity before any capacity or membership mutation")
	result, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	require.NotNil(t, result.Operation)
	assert.Equal(t, OperationPhasePending, result.Operation.Phase)
	assert.Equal(t, OperationShapeReplacementRestoration, result.Operation.Capability.Shape)
	assert.Equal(t, membershipReplicaNativeMemberships("replica-3"), result.Operation.RestoredMembership)
	assert.Empty(t, result.Operation.JoiningReplicas)
	assert.Empty(t, adapter.submitCalls)

	t.Log("Reject substitution while preparing the exact membership request")
	_, err = coordinator.PrepareSubmission(
		context.Background(),
		input.GroupID,
		result.Operation,
		membershipReplicaIncarnations("replica-4"),
	)
	require.Error(t, err)
	assert.Empty(t, adapter.validateRequestCalls)

	t.Log("Carry the planned stable identity into the exact backend request")
	prepared, err := coordinator.PrepareSubmission(
		context.Background(),
		input.GroupID,
		result.Operation,
		membershipReplicaIncarnations("replica-3"),
	)
	require.NoError(t, err)
	assert.Equal(t, membershipReplicaIncarnations("replica-3"), prepared.JoiningReplicas)
	require.Len(t, adapter.validateRequestCalls, 1)
	assert.Equal(t, membershipReplicaIncarnations("replica-3"), adapter.validateRequestCalls[0].JoiningReplicas)
	assert.Equal(t, membershipReplicaNativeMemberships("replica-3"), adapter.validateRequestCalls[0].RestoredMembership)

	t.Log("Keep the durable restoration payload authoritative while a conflicting candidate waits")
	observations := adapter.topologyCalls
	input.Operation = result.Operation
	input.Plan = &OperationPlan{
		ID:                 plan.ID,
		Intent:             plan.Intent,
		TargetReplicas:     plan.TargetReplicas,
		RestoredMembership: membershipReplicaNativeMemberships("replica-4"),
	}
	stable, err := coordinator.Step(context.Background(), input)
	require.NoError(t, err)
	assert.Equal(t, result.Operation, stable.Operation)
	assert.Equal(t, observations+1, adapter.topologyCalls)
	assert.Empty(t, adapter.submitCalls)
}

func TestOperationCoordinatorAdoptsBackendNativeReplacementRestorationBeforeSubmit(t *testing.T) {
	for _, phase := range []OperationPhase{OperationPhasePending, OperationPhaseSubmitting} {
		t.Run(string(phase), func(t *testing.T) {
			baseTopology := membershipTopology(1, 3)
			restoredTopology := membershipTopology(2, 4)
			plan := &OperationPlan{
				ID:                 "restore-plan-1",
				Intent:             OperationIntentRecover,
				TargetReplicas:     4,
				RestoredMembership: membershipReplicaNativeMemberships("replica-3"),
			}
			observedCapability := ResolvedOperationCapability{
				Shape:                   OperationShapeReplacementRestoration,
				TrafficRequirement:      ReconfigurationTrafficKeepServing,
				VerificationRequirement: ServingVerificationRequired,
			}
			adapter := &fakeMembershipAdapter{
				topology:                             baseTopology,
				validateObservedTransitionCapability: &observedCapability,
			}
			coordinator := newTestOperationCoordinator(adapter, func() string { return operationRestorationAdoptedID })
			input := OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: 4,
				Plan:            plan,
			}

			created, err := coordinator.Step(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, created.Operation)
			operation := created.Operation
			if phase == OperationPhaseSubmitting {
				operation, err = coordinator.PrepareSubmission(
					context.Background(),
					input.GroupID,
					operation,
					[]ReplicaIncarnation{restoredTopology.Replicas[3].Incarnation},
				)
				require.NoError(t, err)
				adapter.operation = BackendOperation{Phase: BackendOperationPhaseAbsent}
			}
			assert.Equal(t, phase, operation.Phase)
			adapter.topology = restoredTopology
			input.Operation = operation

			t.Log("Adopt the exact backend-native restoration without submitting a competing request")
			result, err := coordinator.Step(context.Background(), input)
			require.NoError(t, err)
			require.NotNil(t, result.Operation)
			assert.True(t, result.Operation.Adopted)
			assert.Equal(t, OperationPhaseCommitted, result.Operation.Phase)
			assert.Equal(t, restoredTopology, *result.Operation.CommittedTopology)
			assert.Equal(t, plan.RestoredMembership, result.Operation.RestoredMembership)
			assert.Equal(t, []ReplicaIncarnation{restoredTopology.Replicas[3].Incarnation}, result.Operation.JoiningReplicas)
			assert.Equal(t, ServingVerificationRequired, result.Operation.Capability.VerificationRequirement)
			assert.Empty(t, adapter.submitCalls)
			if phase == OperationPhaseSubmitting {
				assert.Len(t, adapter.observeCalls, 1, "Submitting may adopt only after the exact attempt is proven absent")
			} else {
				assert.Empty(t, adapter.observeCalls)
			}
		})
	}
}

func TestOperationCoordinatorRejectsInvalidReplacementRestorationIdentities(t *testing.T) {
	tests := []struct {
		name string
		plan OperationPlan
	}{
		{
			name: "missing exact identity",
			plan: OperationPlan{
				ID:             "restore-plan-1",
				Intent:         OperationIntentRecover,
				TargetReplicas: 4,
			},
		},
		{
			name: "wrong identity count",
			plan: OperationPlan{
				ID:                 "restore-plan-1",
				Intent:             OperationIntentRecover,
				TargetReplicas:     5,
				RestoredMembership: membershipReplicaNativeMemberships("replica-3"),
			},
		},
		{
			name: "duplicate identity",
			plan: OperationPlan{
				ID:                 "restore-plan-1",
				Intent:             OperationIntentRecover,
				TargetReplicas:     5,
				RestoredMembership: membershipReplicaNativeMemberships("replica-3", "replica-3"),
			},
		},
		{
			name: "identity already active",
			plan: OperationPlan{
				ID:                 "restore-plan-1",
				Intent:             OperationIntentRecover,
				TargetReplicas:     4,
				RestoredMembership: membershipReplicaNativeMemberships("replica-2"),
			},
		},
		{
			name: "fresh growth carries restoration identity",
			plan: OperationPlan{
				ID:                 "grow-plan-1",
				Intent:             OperationIntentGrow,
				TargetReplicas:     4,
				RestoredMembership: membershipReplicaNativeMemberships("replica-3"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := &fakeMembershipAdapter{topology: membershipTopology(1, 3)}
			coordinator := newTestOperationCoordinator(adapter, func() string { return operationRestorationTestID })

			result, err := coordinator.Step(context.Background(), OperationInput{
				GroupID:         "group-0",
				SpecGeneration:  2,
				DesiredReplicas: tt.plan.TargetReplicas,
				Plan:            &tt.plan,
			})
			require.Error(t, err)
			assert.Nil(t, result.Operation)
			assert.Empty(t, adapter.submitCalls)
		})
	}
}

func TestCardinalRecoveryShapeDistinguishesReplacementFromNativeRemapping(t *testing.T) {
	base := membershipTopology(1, 2).Replicas
	fixedSlotReplacement := cloneReplicaMemberships(base)
	fixedSlotReplacement[1].Incarnation.CapacityRefs[0].UID = operationRestorationReplacementUID
	fixedSlotReplacement[1].Incarnation.RuntimeID = operationRestorationReplacementRun

	t.Log("A new physical/runtime incarnation with the same native mapping is a fixed-slot replacement")
	assert.Equal(t, OperationShapeFixedSlotReplacement, cardinalRecoveryShape(base, fixedSlotReplacement))

	nativeRemapping := cloneReplicaMemberships(fixedSlotReplacement)
	nativeRemapping[1].NativeMembers = []NativeMemberID{"replacement-native-member"}

	t.Log("Changing the native member mapping is a native-member remap even when capacity is also replaced")
	assert.Equal(t, OperationShapeNativeMemberRemapping, cardinalRecoveryShape(base, nativeRemapping))
}

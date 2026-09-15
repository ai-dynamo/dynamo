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
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateCapacitySnapshotRejectsInvalidFenceBindings(t *testing.T) {
	tests := []struct {
		name      string
		snapshot  CapacitySnapshot
		wantError string
	}{
		{
			name: "empty replica identity",
			snapshot: CapacitySnapshot{FencedReplicaSlots: []ReplicaSlotBinding{{
				SlotID: "slot-replica-1",
			}}},
			wantError: "replica ID must not be empty",
		},
		{
			name: "empty slot identity",
			snapshot: CapacitySnapshot{FencedReplicaSlots: []ReplicaSlotBinding{{
				ReplicaID: "replica-1",
			}}},
			wantError: "empty slot ID",
		},
		{
			name: "duplicate replica identity",
			snapshot: CapacitySnapshot{FencedReplicaSlots: []ReplicaSlotBinding{
				{ReplicaID: "replica-1", SlotID: "slot-replica-1"},
				{ReplicaID: "replica-1", SlotID: "other-slot"},
			}},
			wantError: "duplicate fenced capacity replica ID",
		},
		{
			name: "duplicate slot identity",
			snapshot: CapacitySnapshot{FencedReplicaSlots: []ReplicaSlotBinding{
				{ReplicaID: "replica-1", SlotID: "shared-slot"},
				{ReplicaID: "replica-2", SlotID: "shared-slot"},
			}},
			wantError: "belongs to replicas",
		},
		{
			name: "replica moved away from its fence",
			snapshot: CapacitySnapshot{
				Allocations: []ReplicaAllocation{workflowReplicaAllocation("replica-1", "pod-1")},
				FencedReplicaSlots: []ReplicaSlotBinding{{
					ReplicaID: "replica-1",
					SlotID:    "other-slot",
				}},
			},
			wantError: "changed slot",
		},
		{
			name: "slot reassigned to another replica",
			snapshot: CapacitySnapshot{
				Allocations: []ReplicaAllocation{workflowReplicaAllocation("replica-1", "pod-1")},
				FencedReplicaSlots: []ReplicaSlotBinding{{
					ReplicaID: "replica-2",
					SlotID:    workflowReplicaIncarnation("replica-1").SlotID,
				}},
			},
			wantError: "is allocated to replica",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.ErrorContains(t, validateCapacitySnapshot(tt.snapshot), tt.wantError)
		})
	}
}

func TestValidateCapacityReleaseObservationSupportsStoppedRelease(t *testing.T) {
	require.NoError(t, validateCapacityReleaseObservation(CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseApplying,
	}, workflowTestReleaseID))

	for _, classification := range []FailureClassification{
		FailureClassificationRetryable,
		FailureClassificationTerminal,
	} {
		require.NoError(t, validateCapacityReleaseObservation(CapacityReleaseObservation{
			ReleaseID: workflowTestReleaseID,
			Phase:     CapacityReleasePhaseFailed,
			Failure: &OperationFailure{
				Classification: classification,
				Reason:         "ReleaseStopped",
			},
		}, workflowTestReleaseID))
	}

	require.ErrorContains(t, validateCapacityReleaseObservation(CapacityReleaseObservation{
		ReleaseID: workflowTestReleaseID,
		Phase:     CapacityReleasePhaseFailed,
	}, workflowTestReleaseID), "structured failure")
}

func TestValidateCapacityRequestRejectsInconsistentRequiredReplicas(t *testing.T) {
	tests := []struct {
		name             string
		targetReplicas   int32
		requiredReplicas []RequiredReplicaAllocation
		wantError        string
	}{
		{
			name:           "required count exceeds target",
			targetReplicas: 1,
			requiredReplicas: []RequiredReplicaAllocation{
				{ReplicaID: "replica-0", SlotID: "slot-0"},
				{ReplicaID: "replica-1", SlotID: "slot-1"},
			},
			wantError: "requires 2 exact replicas above target 1",
		},
		{
			name:           "duplicate replica",
			targetReplicas: 2,
			requiredReplicas: []RequiredReplicaAllocation{
				{ReplicaID: "replica-0", SlotID: "slot-0"},
				{ReplicaID: "replica-0", SlotID: "slot-1"},
			},
			wantError: "repeats required replica",
		},
		{
			name:           "duplicate slot",
			targetReplicas: 2,
			requiredReplicas: []RequiredReplicaAllocation{
				{ReplicaID: "replica-0", SlotID: "slot-0"},
				{ReplicaID: "replica-1", SlotID: "slot-0"},
			},
			wantError: "repeats required slot",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCapacityRequest(CapacityRequest{
				OperationID:        workflowTestOperationID,
				TopologyGeneration: 1,
				TargetReplicas:     tt.targetReplicas,
				RequiredReplicas:   tt.requiredReplicas,
			})
			require.ErrorContains(t, err, tt.wantError)
		})
	}
}

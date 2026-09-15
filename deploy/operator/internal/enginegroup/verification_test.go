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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const remappedNativeMemberID NativeMemberID = "native-remapped"

func TestValidateServingVerificationRequest(t *testing.T) {
	valid := servingVerificationTestRequest()
	tests := []struct {
		name    string
		request ServingVerificationRequest
		wantErr string
	}{
		{
			name:    "valid",
			request: valid,
		},
		{
			name: "empty operation ID",
			request: ServingVerificationRequest{
				Attempt:             1,
				VerificationAttempt: 1,
				Topology:            valid.Topology,
			},
			wantErr: "operation ID must not be empty",
		},
		{
			name: "non-positive attempt",
			request: ServingVerificationRequest{
				OperationID:         valid.OperationID,
				VerificationAttempt: 1,
				Topology:            valid.Topology,
			},
			wantErr: "attempt must be positive",
		},
		{
			name: "non-positive verification attempt",
			request: ServingVerificationRequest{
				OperationID: valid.OperationID,
				Attempt:     valid.Attempt,
				Topology:    valid.Topology,
			},
			wantErr: "retry attempt must be positive",
		},
		{
			name: "missing topology",
			request: ServingVerificationRequest{
				OperationID:         valid.OperationID,
				Attempt:             valid.Attempt,
				VerificationAttempt: valid.VerificationAttempt,
			},
			wantErr: "topology must not be empty",
		},
		{
			name: "empty generated topology",
			request: ServingVerificationRequest{
				OperationID:         valid.OperationID,
				Attempt:             valid.Attempt,
				VerificationAttempt: valid.VerificationAttempt,
				Topology: MembershipTopology{
					Generation: 3,
				},
			},
			wantErr: "topology must not be empty",
		},
		{
			name: "invalid topology",
			request: ServingVerificationRequest{
				OperationID:         valid.OperationID,
				Attempt:             valid.Attempt,
				VerificationAttempt: valid.VerificationAttempt,
				Topology: MembershipTopology{
					Generation: -1,
				},
			},
			wantErr: "generation must not be negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Validate the immutable serving verification request")
			err := validateServingVerificationRequest(tt.request)

			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}

func TestValidateServingVerificationProof(t *testing.T) {
	request := servingVerificationTestRequest()
	failure := &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "CollectiveStalled",
		Message:        "collective progress barrier timed out",
	}
	tests := []struct {
		name    string
		proof   ServingVerificationProof
		wantErr string
	}{
		{
			name: "absent",
			proof: ServingVerificationProof{
				OperationID:         request.OperationID,
				Attempt:             request.Attempt,
				VerificationAttempt: request.VerificationAttempt,
				Phase:               ServingVerificationPhaseAbsent,
			},
		},
		{
			name:  "running",
			proof: servingVerificationTestProof(request, ServingVerificationPhaseRunning, nil),
		},
		{
			name:  "passed",
			proof: servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
		},
		{
			name:  "failed",
			proof: servingVerificationTestProof(request, ServingVerificationPhaseFailed, failure),
		},
		{
			name:  "unknown",
			proof: servingVerificationTestProof(request, ServingVerificationPhaseUnknown, nil),
		},
		{
			name:    "absent with topology",
			proof:   servingVerificationTestProof(request, ServingVerificationPhaseAbsent, nil),
			wantErr: "must not carry a topology",
		},
		{
			name:    "running with failure",
			proof:   servingVerificationTestProof(request, ServingVerificationPhaseRunning, failure),
			wantErr: "must not carry a failure",
		},
		{
			name:    "failed without failure",
			proof:   servingVerificationTestProof(request, ServingVerificationPhaseFailed, nil),
			wantErr: "must carry a structured failure",
		},
		{
			name:    "invalid phase",
			proof:   servingVerificationTestProof(request, ServingVerificationPhase("Waiting"), nil),
			wantErr: "invalid serving verification phase",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Validate phase-specific proof invariants")
			err := validateServingVerificationProof(tt.proof)

			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}

func TestServingVerificationProofMatchesExactRequest(t *testing.T) {
	request := servingVerificationTestRequest()
	reordered := cloneServingVerificationRequest(request)
	reordered.Topology.Replicas[0], reordered.Topology.Replicas[1] =
		reordered.Topology.Replicas[1], reordered.Topology.Replicas[0]
	reordered.Topology.Replicas[1].NativeMembers[0], reordered.Topology.Replicas[1].NativeMembers[1] =
		reordered.Topology.Replicas[1].NativeMembers[1], reordered.Topology.Replicas[1].NativeMembers[0]
	differentOperation := cloneServingVerificationRequest(request)
	differentOperation.OperationID = "operation-2"
	differentAttempt := cloneServingVerificationRequest(request)
	differentAttempt.Attempt++
	differentVerificationAttempt := cloneServingVerificationRequest(request)
	differentVerificationAttempt.VerificationAttempt++
	differentGeneration := cloneServingVerificationRequest(request)
	differentGeneration.Topology.Generation++
	differentMapping := cloneServingVerificationRequest(request)
	differentMapping.Topology.Replicas[0].NativeMembers[0] = remappedNativeMemberID
	differentPodUID := cloneServingVerificationRequest(request)
	differentPodUID.Topology.Replicas[0].Incarnation.CapacityRefs[0].UID = operationRestorationReplacementUID
	differentRuntime := cloneServingVerificationRequest(request)
	differentRuntime.Topology.Replicas[0].Incarnation.RuntimeID = operationRestorationReplacementRun
	differentSlot := cloneServingVerificationRequest(request)
	differentSlot.Topology.Replicas[0].Incarnation.SlotID = "replacement-slot"

	tests := []struct {
		name    string
		proof   ServingVerificationProof
		request ServingVerificationRequest
		want    bool
	}{
		{
			name:    "exact request",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: request,
			want:    true,
		},
		{
			name:    "presentation order differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: reordered,
			want:    true,
		},
		{
			name:    "operation differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentOperation,
		},
		{
			name:    "attempt differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentAttempt,
		},
		{
			name:    "verification attempt differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentVerificationAttempt,
		},
		{
			name:    "topology generation differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentGeneration,
		},
		{
			name:    "logical to native mapping differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentMapping,
		},
		{
			name:    "capacity Pod UID differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentPodUID,
		},
		{
			name:    "runtime incarnation differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentRuntime,
		},
		{
			name:    "capacity slot differs",
			proof:   servingVerificationTestProof(request, ServingVerificationPhasePassed, nil),
			request: differentSlot,
		},
		{
			name: "absent is not proof",
			proof: ServingVerificationProof{
				OperationID:         request.OperationID,
				Attempt:             request.Attempt,
				VerificationAttempt: request.VerificationAttempt,
				Phase:               ServingVerificationPhaseAbsent,
			},
			request: request,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Match the proof against the complete immutable request")
			assert.Equal(t, tt.want, servingVerificationProofMatchesRequest(tt.proof, tt.request))
		})
	}
}

func TestServingVerificationPassedForRequest(t *testing.T) {
	request := servingVerificationTestRequest()
	running := servingVerificationTestProof(request, ServingVerificationPhaseRunning, nil)
	passed := servingVerificationTestProof(request, ServingVerificationPhasePassed, nil)
	changedTopology := cloneServingVerificationRequest(request)
	changedTopology.Topology.Replicas[0].NativeMembers[0] = remappedNativeMemberID

	t.Log("Refuse admission while an exactly correlated verification is still running")
	assert.False(t, servingVerificationPassedForRequest(running, request))

	t.Log("Authorize admission after the exact committed topology passes verification")
	assert.True(t, servingVerificationPassedForRequest(passed, request))

	t.Log("Invalidate the passed proof when the logical-to-native mapping changes")
	assert.False(t, servingVerificationPassedForRequest(passed, changedTopology))
}

func TestServingVerificationRequestEqualityUsesTopologySets(t *testing.T) {
	left := servingVerificationTestRequest()
	right := cloneServingVerificationRequest(left)
	right.Topology.Replicas[0], right.Topology.Replicas[1] = right.Topology.Replicas[1], right.Topology.Replicas[0]
	right.Topology.Replicas[1].NativeMembers[0], right.Topology.Replicas[1].NativeMembers[1] =
		right.Topology.Replicas[1].NativeMembers[1], right.Topology.Replicas[1].NativeMembers[0]

	t.Log("Treat topology slice order as presentation rather than request identity")
	assert.True(t, servingVerificationRequestsEqual(left, right))

	t.Log("Treat a native-member remap as a conflicting request payload")
	right.Topology.Replicas[0].NativeMembers[0] = remappedNativeMemberID
	assert.False(t, servingVerificationRequestsEqual(left, right))
}

func TestCloneServingVerificationState(t *testing.T) {
	request := servingVerificationTestRequest()
	proof := servingVerificationTestProof(request, ServingVerificationPhaseFailed, &OperationFailure{
		Classification: FailureClassificationRetryable,
		Reason:         "ProbeUnavailable",
	})

	t.Log("Clone requests and proofs without sharing nested topology or failure storage")
	requestClone := cloneServingVerificationRequest(request)
	proofClone := cloneServingVerificationProof(&proof)
	require.NotNil(t, proofClone)

	requestClone.Topology.Replicas[0].NativeMembers[0] = "request-mutated"
	proofClone.Topology.Replicas[0].NativeMembers[0] = "proof-mutated"
	proofClone.Failure.Reason = "FailureMutated"
	assert.NotEqual(t, requestClone.Topology, request.Topology)
	assert.NotEqual(t, proofClone.Topology, proof.Topology)
	assert.Equal(t, "ProbeUnavailable", proof.Failure.Reason)

	t.Log("Preserve nil proof semantics")
	assert.Nil(t, cloneServingVerificationProof(nil))
}

func servingVerificationTestRequest() ServingVerificationRequest {
	return ServingVerificationRequest{
		OperationID:         "operation-1",
		Attempt:             1,
		VerificationAttempt: 1,
		Topology: MembershipTopology{
			Generation: 4,
			Replicas: []ReplicaMembership{
				membershipReplicaMembership("replica-0", "native-0-0", "native-0-1"),
				membershipReplicaMembership("replica-1", "native-1-0"),
			},
		},
	}
}

func servingVerificationTestProof(
	request ServingVerificationRequest,
	phase ServingVerificationPhase,
	failure *OperationFailure,
) ServingVerificationProof {
	return ServingVerificationProof{
		OperationID:         request.OperationID,
		Attempt:             request.Attempt,
		VerificationAttempt: request.VerificationAttempt,
		Topology:            cloneTopology(request.Topology),
		Phase:               phase,
		Failure:             cloneFailure(failure),
	}
}

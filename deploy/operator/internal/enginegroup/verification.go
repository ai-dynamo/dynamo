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
)

// ServingVerificationPhase is the restart-observable state of one serving verification.
type ServingVerificationPhase string

const (
	// ServingVerificationPhaseAbsent means the verifier has no request for this operation attempt.
	ServingVerificationPhaseAbsent ServingVerificationPhase = "Absent"
	// ServingVerificationPhaseRunning means the exact request is durable and verification is in progress.
	ServingVerificationPhaseRunning ServingVerificationPhase = "Running"
	// ServingVerificationPhasePassed means the exact committed topology passed its required serving checks.
	ServingVerificationPhasePassed ServingVerificationPhase = "Passed"
	// ServingVerificationPhaseFailed means the exact verification attempt explicitly failed.
	ServingVerificationPhaseFailed ServingVerificationPhase = "Failed"
	// ServingVerificationPhaseUnknown means the verifier cannot establish the exact request's outcome.
	ServingVerificationPhaseUnknown ServingVerificationPhase = "Unknown"
)

// ServingVerificationRequest binds one retryable serving check to a membership operation attempt and its exact
// committed logical-to-native member mapping. The operation and verification attempts form the idempotency identity
// within one Engine Group.
type ServingVerificationRequest struct {
	OperationID         string
	Attempt             int32
	VerificationAttempt int32
	Topology            MembershipTopology
}

// ServingVerificationProof is the durable, restart-observable result of one exact serving verification request.
// A Passed proof applies only while its complete topology still matches the authoritative committed topology. The
// membership contract advances that topology generation for every process or capacity incarnation that may serve.
type ServingVerificationProof struct {
	OperationID         string
	Attempt             int32
	VerificationAttempt int32
	Topology            MembershipTopology
	Phase               ServingVerificationPhase
	Failure             *OperationFailure
}

// ServingVerifier verifies that an exact committed engine topology can make serving progress. For one exact
// (operation ID, membership attempt, verification attempt), observations are monotonic: Absent may advance to
// Running or directly to a terminal phase, Running may advance to a terminal phase, and Passed, Failed, and Unknown
// are immutable terminal results. A retry after Failed or Unknown uses a new verification attempt.
type ServingVerifier interface {
	// EnsureVerification idempotently starts or resumes the exact request. The verifier durably records the complete
	// request before returning success. Reusing the same operation ID, membership attempt, and verification attempt
	// with a different topology is rejected without replacing the original request or its result. A non-nil error
	// leaves the start outcome ambiguous; callers observe the same full retry identity rather than minting a competing
	// one.
	EnsureVerification(ctx context.Context, groupID GroupID, request ServingVerificationRequest) error
	// ObserveVerification recovers the result of one verification attempt after timeout or restart. Running, Passed,
	// Failed, and Unknown observations echo the complete topology accepted by EnsureVerification. Absent echoes only
	// the queried operation identity. Once EnsureVerification succeeds, later observations cannot report Absent or
	// regress from a terminal phase.
	ObserveVerification(
		ctx context.Context,
		groupID GroupID,
		operationID string,
		attempt int32,
		verificationAttempt int32,
	) (ServingVerificationProof, error)
}

func validateServingVerificationRequest(request ServingVerificationRequest) error {
	if request.OperationID == "" {
		return errors.New("serving verification operation ID must not be empty")
	}
	if request.Attempt <= 0 {
		return fmt.Errorf("serving verification attempt must be positive: %d", request.Attempt)
	}
	if request.VerificationAttempt <= 0 {
		return fmt.Errorf(
			"serving verification retry attempt must be positive: %d",
			request.VerificationAttempt,
		)
	}
	if err := validateServingVerificationTopology(request.Topology); err != nil {
		return fmt.Errorf("validate serving verification topology: %w", err)
	}
	return nil
}

func validateServingVerificationProof(proof ServingVerificationProof) error {
	if proof.OperationID == "" {
		return errors.New("serving verification proof operation ID must not be empty")
	}
	if proof.Attempt <= 0 {
		return fmt.Errorf("serving verification proof attempt must be positive: %d", proof.Attempt)
	}
	if proof.VerificationAttempt <= 0 {
		return fmt.Errorf(
			"serving verification proof retry attempt must be positive: %d",
			proof.VerificationAttempt,
		)
	}

	// An absent observation has no accepted request whose topology or failure could be reported.
	if proof.Phase == ServingVerificationPhaseAbsent {
		if !zeroMembershipTopology(proof.Topology) {
			return errors.New("absent serving verification proof must not carry a topology")
		}
		if proof.Failure != nil {
			return errors.New("absent serving verification proof must not carry a failure")
		}
		return nil
	}

	// Every non-absent observation must echo the exact committed topology accepted for verification.
	if err := validateServingVerificationTopology(proof.Topology); err != nil {
		return fmt.Errorf("validate serving verification proof topology: %w", err)
	}

	// Only an explicit failed result carries a structured operation failure.
	switch proof.Phase {
	case ServingVerificationPhaseRunning,
		ServingVerificationPhasePassed,
		ServingVerificationPhaseUnknown:
		if proof.Failure != nil {
			return fmt.Errorf("serving verification phase %q must not carry a failure", proof.Phase)
		}
		return nil
	case ServingVerificationPhaseFailed:
		if err := validateFailure(proof.Failure); err != nil {
			return fmt.Errorf("validate serving verification failure: %w", err)
		}
		return nil
	default:
		return fmt.Errorf("invalid serving verification phase %q", proof.Phase)
	}
}

func validateServingVerificationTopology(topology MembershipTopology) error {
	if err := validateTopology(topology); err != nil {
		return err
	}
	if len(topology.Replicas) == 0 {
		return errors.New("serving verification topology must not be empty")
	}
	return nil
}

// servingVerificationProofMatchesRequest reports whether a non-absent, valid proof is correlated with the exact
// request. Replica and native-member ordering is presentation-only; identities and their mapping compare as sets.
func servingVerificationProofMatchesRequest(
	proof ServingVerificationProof,
	request ServingVerificationRequest,
) bool {
	if proof.Phase == ServingVerificationPhaseAbsent ||
		validateServingVerificationProof(proof) != nil ||
		validateServingVerificationRequest(request) != nil {
		return false
	}
	return proof.OperationID == request.OperationID &&
		proof.Attempt == request.Attempt &&
		proof.VerificationAttempt == request.VerificationAttempt &&
		servingVerificationTopologiesEqual(proof.Topology, request.Topology)
}

// servingVerificationPassedForRequest reports whether an exact, valid proof authorizes admission for the request.
// Any topology generation or logical-to-native member mapping change invalidates an earlier passed proof.
func servingVerificationPassedForRequest(
	proof ServingVerificationProof,
	request ServingVerificationRequest,
) bool {
	return proof.Phase == ServingVerificationPhasePassed &&
		servingVerificationProofMatchesRequest(proof, request)
}

func servingVerificationProofMatchesOperation(proof *ServingVerificationProof, operation Operation) bool {
	return proof != nil &&
		operation.CommittedTopology != nil &&
		proof.Phase == ServingVerificationPhasePassed &&
		proof.OperationID == operation.ID &&
		proof.Attempt == operation.Attempt &&
		proof.VerificationAttempt == operation.ServingVerificationAttempt &&
		servingVerificationTopologiesEqual(proof.Topology, *operation.CommittedTopology)
}

func servingVerificationRequestsEqual(left, right ServingVerificationRequest) bool {
	if validateServingVerificationRequest(left) != nil || validateServingVerificationRequest(right) != nil {
		return false
	}
	return left.OperationID == right.OperationID &&
		left.Attempt == right.Attempt &&
		left.VerificationAttempt == right.VerificationAttempt &&
		servingVerificationTopologiesEqual(left.Topology, right.Topology)
}

func servingVerificationTopologiesEqual(left, right MembershipTopology) bool {
	if left.Generation != right.Generation || len(left.Replicas) != len(right.Replicas) {
		return false
	}

	// Index the right-hand topology by logical replica while preserving its exact physical incarnation and native set.
	rightReplicas := make(map[ReplicaID]ReplicaMembership, len(right.Replicas))
	for _, replica := range right.Replicas {
		rightReplicas[replica.Incarnation.ReplicaID] = replica
	}

	// Require every left logical replica to have the same capacity incarnation and native-member identities.
	for _, replica := range left.Replicas {
		rightReplica, exists := rightReplicas[replica.Incarnation.ReplicaID]
		if !exists ||
			!sameReplicaIncarnation(replica.Incarnation, rightReplica.Incarnation) ||
			!sameNativeMemberIDs(replica.NativeMembers, rightReplica.NativeMembers) {
			return false
		}
	}
	return true
}

func cloneServingVerificationRequest(request ServingVerificationRequest) ServingVerificationRequest {
	cloned := request
	cloned.Topology = cloneTopology(request.Topology)
	return cloned
}

func cloneServingVerificationProof(proof *ServingVerificationProof) *ServingVerificationProof {
	if proof == nil {
		return nil
	}

	cloned := *proof
	cloned.Topology = cloneTopology(proof.Topology)
	cloned.Failure = cloneFailure(proof.Failure)
	return &cloned
}

func zeroMembershipTopology(topology MembershipTopology) bool {
	return topology.Generation == 0 && len(topology.Replicas) == 0
}

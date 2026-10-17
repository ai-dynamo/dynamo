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

func (c *Coordinator) reconcileServingVerification(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	committed MembershipTopology,
) (ready bool, persist bool, err error) {
	if status.Transition.Spec.Plan.VerificationRequirement == VerificationRequirementNone {
		return true, false, nil
	}

	verification := &status.Transition.Verification
	if verification.Phase == VerificationPhasePassed {
		if verification.Proof == nil ||
			verification.Proof.TopologyGeneration != committed.Generation ||
			verification.Proof.RuntimeDigest != TopologyRuntimeDigest(committed) {
			return false, false, errors.New("serving proof does not match the committed topology")
		}
		return true, false, nil
	}
	if verification.Phase == VerificationPhaseFailed {
		return false, false, nil
	}
	if verification.Phase == "" {
		verification.Phase = VerificationPhasePending
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}

	result, verifyErr := c.verifier.Verify(ctx, groupID, committed)
	if verifyErr != nil {
		return false, false, fmt.Errorf("verify committed serving topology: %w", verifyErr)
	}
	if result.Proof != nil && result.Failure != nil {
		return false, false, errors.New("serving verification returned both proof and failure")
	}
	if result.Proof == nil && result.Failure == nil {
		return false, false, errors.New("serving verification returned neither proof nor failure")
	}
	if result.Failure != nil {
		if failureErr := validateFailure(result.Failure); failureErr != nil {
			return false, false, fmt.Errorf("invalid serving verification failure: %w", failureErr)
		}
		// A transient probe failure is retryable evidence, not a terminal verdict on the committed topology.
		if result.Failure.Classification == FailureClassificationRetryable {
			return false, false, &InvocationError{Failure: *result.Failure}
		}
		verification.Phase = VerificationPhaseFailed
		verification.Failure = cloneFailure(result.Failure)
		c.blockTransition(status, *result.Failure)
		return false, true, nil
	}
	if result.Proof.TopologyGeneration != committed.Generation ||
		result.Proof.RuntimeDigest != TopologyRuntimeDigest(committed) {
		return false, false, errors.New("serving verifier returned proof for another topology")
	}

	verification.Phase = VerificationPhasePassed
	verification.Proof = cloneServingProof(result.Proof)
	verification.Failure = nil
	status.Transition.UpdatedAt = c.now()
	return false, true, nil
}

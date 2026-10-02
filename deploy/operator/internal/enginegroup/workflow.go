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

func (c *Coordinator) reconcileRollback(
	ctx context.Context,
	groupID GroupID,
	status GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
) (ReconcileResult, error) {
	if !sameTopology(base, observedTopology) {
		failure := Failure{
			Classification: FailureClassificationTerminal,
			Reason:         "RollbackTopologyChanged",
			Message:        "cannot restore preparatory state because engine membership changed",
		}
		c.blockTransition(&status, failure)
		return ReconcileResult{Status: status}, nil
	}

	// Remove only uncommitted joining capacity, using its exact observed Pod UIDs as deletion fences.
	ready, persist, err := c.reconcileRollbackCapacity(ctx, groupID, &status, base, resolution)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: status, Requeue: err == nil}, err
	}

	// Restore the unchanged canonical base topology as the absolute admitted traffic set.
	ready, persist, err = c.reconcileRollbackTraffic(ctx, groupID, &status, base)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: status, Requeue: err == nil}, err
	}

	status.Transition.Outcome = TransitionOutcomeRolledBack
	status.Transition.UpdatedAt = c.now()
	return ReconcileResult{Status: status}, nil
}

func (c *Coordinator) reconcileMaintainedTargets(
	ctx context.Context,
	groupID GroupID,
	status GroupStatus,
) (ReconcileResult, error) {
	if status.Transition.Outcome != TransitionOutcomeBlocked {
		current, found := status.Topologies.Current()
		if !found || !sameTopology(current, status.Membership.Observed.CommittedTopology) {
			return ReconcileResult{Status: status}, errors.New(
				"terminal transition does not match authoritative membership",
			)
		}
	}

	acceptedTraffic := status.Traffic.Accepted
	if acceptedTraffic != nil && !trafficTargetConverged(*acceptedTraffic, status.Traffic.Observed) {
		result, err := c.traffic.Apply(ctx, groupID, *acceptedTraffic)
		if err != nil {
			return ReconcileResult{Status: status, Requeue: true}, fmt.Errorf(
				"reassert maintained traffic target: %w",
				err,
			)
		}
		if result.Rejection != nil {
			if err := validateRejection(result.Rejection); err != nil {
				return ReconcileResult{Status: status}, fmt.Errorf("invalid maintained traffic rejection: %w", err)
			}
			return ReconcileResult{Status: status}, fmt.Errorf(
				"maintained traffic target was rejected: %s: %s",
				result.Rejection.Reason,
				result.Rejection.Message,
			)
		}
		return ReconcileResult{Status: status, Requeue: true}, nil
	}

	acceptedCapacity := status.Capacity.Accepted
	if acceptedCapacity != nil && !terminalCapacityTargetConverged(*acceptedCapacity, status.Capacity.Observed) {
		// A committed topology must enter explicit recovery instead of recreating or adopting a new incarnation.
		if membershipCommittedForTransition(status) {
			if err := validatePinnedCapacity(
				*acceptedCapacity,
				status.Capacity.Observed,
				status.Membership.Observed.CommittedTopology,
			); err != nil {
				return ReconcileResult{Status: status}, err
			}
		}

		result, err := c.capacity.Apply(ctx, groupID, *acceptedCapacity)
		if err != nil {
			return ReconcileResult{Status: status, Requeue: true}, fmt.Errorf(
				"reassert maintained capacity target: %w",
				err,
			)
		}
		if result.Rejection != nil {
			if err := validateRejection(result.Rejection); err != nil {
				return ReconcileResult{Status: status}, fmt.Errorf("invalid maintained capacity rejection: %w", err)
			}
			return ReconcileResult{Status: status}, fmt.Errorf(
				"maintained capacity target was rejected: %s: %s",
				result.Rejection.Reason,
				result.Rejection.Message,
			)
		}
		return ReconcileResult{Status: status, Requeue: true}, nil
	}

	return ReconcileResult{Status: status}, nil
}

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
	"slices"
	"strings"
)

func (c *Coordinator) reconcileRollbackTraffic(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
) (ready bool, persist bool, err error) {
	target := TrafficTarget{
		TransitionID:       status.Transition.Spec.ID,
		TopologyGeneration: base.Generation,
		Admitted:           normalizeMemberships(base.Replicas),
	}
	if status.Traffic.Desired == nil || !sameTrafficTargetIntent(*status.Traffic.Desired, target) {
		revision, revisionErr := c.nextControlRevision(status)
		if revisionErr != nil {
			return false, false, revisionErr
		}
		target.ControlRevision = revision
		status.Traffic.Desired = &target
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}

	desired := *status.Traffic.Desired
	if !trafficTargetConverged(desired, status.Traffic.Observed) {
		result, applyErr := c.traffic.Apply(ctx, groupID, desired)
		if applyErr != nil {
			return false, false, fmt.Errorf("apply rollback traffic target: %w", applyErr)
		}
		if result.Rejection != nil {
			if rejectionErr := validateRejection(result.Rejection); rejectionErr != nil {
				return false, false, fmt.Errorf("invalid rollback traffic rejection: %w", rejectionErr)
			}
			c.blockTransition(status, *result.Rejection)
			return false, true, nil
		}
		return false, false, nil
	}
	return true, false, nil
}

// reconcileMaintainedTargets keeps the last adapter-acknowledged physical and traffic levels converged after progress
// stops. A newer definitively rejected target remains durable for diagnosis but is never replayed here.

func (c *Coordinator) reconcilePreMembershipTraffic(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	if membershipCommittedForTransition(*status) {
		return true, false, nil
	}

	target := buildPreMembershipTrafficTarget(*status, base, resolution)
	if status.Traffic.Desired == nil ||
		!sameTrafficTargetIntent(*status.Traffic.Desired, target) {
		revision, revisionErr := c.nextControlRevision(status)
		if revisionErr != nil {
			return false, false, revisionErr
		}
		target.ControlRevision = revision
		status.Traffic.Desired = &target
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}

	desired := *status.Traffic.Desired
	if !trafficTargetConverged(desired, status.Traffic.Observed) {
		result, applyErr := c.traffic.Apply(ctx, groupID, desired)
		if applyErr != nil {
			return false, false, fmt.Errorf("apply pre-membership traffic target: %w", applyErr)
		}
		if result.Rejection != nil {
			if rejectionErr := validateRejection(result.Rejection); rejectionErr != nil {
				return false, false, fmt.Errorf("invalid pre-membership traffic rejection: %w", rejectionErr)
			}
			c.beginRollback(status, *result.Rejection)
			return false, true, nil
		}
		return false, false, nil
	}
	return true, false, nil
}

func (c *Coordinator) reconcileCommittedTraffic(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	committed MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	target := buildCommittedTrafficTarget(*status, base, committed, resolution)
	if status.Traffic.Desired == nil ||
		!sameTrafficTargetIntent(*status.Traffic.Desired, target) {
		revision, revisionErr := c.nextControlRevision(status)
		if revisionErr != nil {
			return false, false, revisionErr
		}
		target.ControlRevision = revision
		status.Traffic.Desired = &target
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}

	desired := *status.Traffic.Desired
	if !trafficTargetConverged(desired, status.Traffic.Observed) {
		result, applyErr := c.traffic.Apply(ctx, groupID, desired)
		if applyErr != nil {
			return false, false, fmt.Errorf("apply committed traffic target: %w", applyErr)
		}
		if result.Rejection != nil {
			if rejectionErr := validateRejection(result.Rejection); rejectionErr != nil {
				return false, false, fmt.Errorf("invalid committed traffic rejection: %w", rejectionErr)
			}
			c.blockTransition(status, *result.Rejection)
			return false, true, nil
		}
		return false, false, nil
	}
	return true, false, nil
}

func buildPreMembershipTrafficTarget(
	status GroupStatus,
	base MembershipTopology,
	resolution planResolution,
) TrafficTarget {
	retiring := replicaIDSet(resolution.retiringReplicaIDs)
	admitted := make([]ReplicaMembership, 0, len(base.Replicas))
	if status.Transition.Spec.Plan.TrafficRequirement == TrafficRequirementKeepServing {
		for _, membership := range base.Replicas {
			if _, removed := retiring[membership.ReplicaID]; !removed {
				admitted = append(admitted, cloneReplicaMembership(membership))
			}
		}
	}

	drain := make([]TrafficDrainTarget, 0)
	if status.Transition.Spec.Plan.TrafficRequirement == TrafficRequirementQuiesceGroup {
		for _, membership := range base.Replicas {
			mode := TrafficDrainModeGraceful
			if _, retiringMember := retiring[membership.ReplicaID]; retiringMember &&
				status.Transition.Spec.Plan.Change.Kind == PlanKindReduceToSurvivors {
				mode = TrafficDrainModeConfirmInactive
			}
			drain = append(drain, TrafficDrainTarget{Membership: cloneReplicaMembership(membership), Mode: mode})
		}
	} else {
		for _, replicaID := range resolution.retiringReplicaIDs {
			membership, _ := membershipByID(base, replicaID)
			drain = append(drain, TrafficDrainTarget{
				Membership: membership,
				Mode:       trafficDrainMode(status.Transition.Spec.Plan.Change.Kind),
			})
		}
	}

	return TrafficTarget{
		TransitionID:       status.Transition.Spec.ID,
		TopologyGeneration: base.Generation,
		Admitted:           normalizeMemberships(admitted),
		Drain:              normalizeTrafficDrainTargets(drain),
	}
}

func buildCommittedTrafficTarget(
	status GroupStatus,
	base MembershipTopology,
	committed MembershipTopology,
	resolution planResolution,
) TrafficTarget {
	drain := make([]TrafficDrainTarget, 0, len(resolution.retiringReplicaIDs))
	for _, replicaID := range resolution.retiringReplicaIDs {
		membership, _ := membershipByID(base, replicaID)
		drain = append(drain, TrafficDrainTarget{
			Membership: membership,
			Mode:       trafficDrainMode(status.Transition.Spec.Plan.Change.Kind),
		})
	}
	return TrafficTarget{
		TransitionID:       status.Transition.Spec.ID,
		TopologyGeneration: committed.Generation,
		Admitted:           normalizeMemberships(committed.Replicas),
		Drain:              normalizeTrafficDrainTargets(drain),
	}
}

func trafficDrainMode(kind PlanKind) TrafficDrainMode {
	if kind == PlanKindReduceToSurvivors {
		return TrafficDrainModeConfirmInactive
	}
	return TrafficDrainModeGraceful
}

func sameTrafficTargetIntent(left, right TrafficTarget) bool {
	return left.TransitionID == right.TransitionID &&
		left.TopologyGeneration == right.TopologyGeneration &&
		sameMemberships(left.Admitted, right.Admitted) &&
		sameTrafficDrainTargets(left.Drain, right.Drain)
}

func normalizeTrafficDrainTargets(values []TrafficDrainTarget) []TrafficDrainTarget {
	normalized := make([]TrafficDrainTarget, 0, len(values))
	for _, value := range values {
		value.Membership = cloneReplicaMembership(value.Membership)
		value.Membership.NativeMembers = normalizeNativeMembers(value.Membership.NativeMembers)
		normalized = append(normalized, value)
	}
	slices.SortFunc(normalized, func(left, right TrafficDrainTarget) int {
		return strings.Compare(string(left.Membership.ReplicaID), string(right.Membership.ReplicaID))
	})
	return normalized
}

func sameTrafficDrainTargets(left, right []TrafficDrainTarget) bool {
	left = normalizeTrafficDrainTargets(left)
	right = normalizeTrafficDrainTargets(right)
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].Mode != right[index].Mode ||
			!sameMembership(left[index].Membership, right[index].Membership) {
			return false
		}
	}
	return true
}

func trafficTargetConverged(target TrafficTarget, observation TrafficObservation) bool {
	if observation.AppliedRevision < target.ControlRevision ||
		!sameMemberships(target.Admitted, observation.Admitted) {
		return false
	}
	for _, required := range target.Drain {
		if !containsMembership(observation.Drained, required.Membership) {
			return false
		}
	}
	return true
}

func containsMembership(values []ReplicaMembership, expected ReplicaMembership) bool {
	for _, value := range values {
		if sameMembership(value, expected) {
			return true
		}
	}
	return false
}

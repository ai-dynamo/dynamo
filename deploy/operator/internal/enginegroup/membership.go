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
	"slices"
)

func (c *Coordinator) reconcilePlanPreflight(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
) (ready bool, persist bool, err error) {
	digest, err := canonicalPlanDigest(status.Transition.Spec.Plan)
	if err != nil {
		return false, false, err
	}
	preflight := &status.Transition.PlanPreflight
	if preflight.SubjectDigest != "" {
		if preflight.SubjectDigest != digest {
			return false, false, errors.New("durable plan preflight refers to another plan")
		}
		return preflight.Evidence != nil, false, nil
	}

	result, validationErr := c.membership.ValidatePlan(
		ctx,
		groupID,
		PlanValidationRequest{
			BaseTopology: cloneTopology(base),
			Plan:         cloneResolvedPlan(status.Transition.Spec.Plan),
			PlanDigest:   digest,
		},
	)
	if validationErr != nil {
		return false, false, fmt.Errorf("validate membership plan: %w", validationErr)
	}
	if err := validatePreflightResult(result); err != nil {
		return false, false, fmt.Errorf("invalid plan preflight result: %w", err)
	}
	preflight.TransitionID = status.Transition.Spec.ID
	preflight.SubjectDigest = digest
	if result.Rejection != nil {
		preflight.Rejection = cloneFailure(result.Rejection)
		c.blockTransition(status, *result.Rejection)
		return false, true, nil
	}
	if err := validatePlanEvidence(*result.Evidence, digest, status.Transition.Spec.Plan); err != nil {
		return false, false, fmt.Errorf("invalid plan validation evidence: %w", err)
	}
	preflight.Evidence = cloneValidationEvidence(result.Evidence)
	status.Transition.UpdatedAt = c.now()
	return false, true, nil
}

func (c *Coordinator) reconcileMembership(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	expectedTransitionID := membershipTransitionID(
		status.Transition.Spec.Plan.ID,
		status.Transition.Spec.BaseTopologyGeneration,
	)
	if status.Membership.Desired == nil || status.Membership.Desired.TransitionID != expectedTransitionID {
		return c.prepareMembershipTarget(
			ctx,
			groupID,
			status,
			base,
			observedTopology,
			resolution,
			expectedTransitionID,
		)
	}
	return c.reconcileDesiredMembership(ctx, status, groupID, base, observedTopology, resolution)
}

func (c *Coordinator) prepareMembershipTarget(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
	expectedTransitionID string,
) (ready bool, persist bool, err error) {
	if status.Membership.Desired != nil && !membershipTargetMayBeReplaced(status.Membership) {
		return false, false, errors.New("previous membership target has no terminal result")
	}
	if !sameTopology(base, observedTopology) {
		return false, false, errors.New("engine topology changed before membership target creation")
	}
	joining, err := joiningReplicaIdentities(status.Registry, resolution)
	if err != nil {
		return false, false, err
	}
	planEvidence := status.Transition.PlanPreflight.Evidence
	if planEvidence == nil {
		return false, false, errors.New("membership target lacks durable plan validation")
	}
	revision, err := nextControlRevisionValue(*status)
	if err != nil {
		return false, false, err
	}
	target := normalizeMembershipTarget(MembershipTarget{
		ControlRevision: revision,
		TransitionID:    expectedTransitionID,
		Validation:      *cloneValidationEvidence(planEvidence),
		BaseTopology:    cloneTopology(base),
		Plan:            cloneResolvedPlan(status.Transition.Spec.Plan),
		Joining:         joining,
	})
	target.TargetDigest, err = canonicalMembershipTargetDigest(target)
	if err != nil {
		return false, false, err
	}
	result, err := c.membership.ValidateTarget(ctx, groupID, target)
	if err != nil {
		return false, false, fmt.Errorf("validate exact membership target: %w", err)
	}
	if err := validatePreflightResult(result); err != nil {
		return false, false, fmt.Errorf("invalid target preflight result: %w", err)
	}
	status.Transition.TargetPreflight = PreflightStatus{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		SubjectDigest:   target.TargetDigest,
	}
	if result.Rejection != nil {
		status.Transition.TargetPreflight.Rejection = cloneFailure(result.Rejection)
		c.beginRollback(status, *result.Rejection)
		return false, true, nil
	}
	if err := validateTargetEvidence(*result.Evidence, target, *planEvidence); err != nil {
		return false, false, fmt.Errorf("invalid target validation evidence: %w", err)
	}
	target.Validation = *cloneValidationEvidence(result.Evidence)
	status.ControlRevision = revision
	status.Transition.TargetPreflight.Evidence = cloneValidationEvidence(result.Evidence)
	status.Membership.Desired = &target
	status.Transition.UpdatedAt = c.now()
	return false, true, nil
}

func (c *Coordinator) reconcileDesiredMembership(
	ctx context.Context,
	status *GroupStatus,
	groupID GroupID,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	target := *status.Membership.Desired
	if !membershipTargetMatchesTransition(target, *status.Transition) ||
		!sameTopology(target.BaseTopology, base) {
		return false, false, errors.New("durable membership target does not match the active transition")
	}
	if status.Transition.TargetPreflight.Evidence == nil ||
		status.Transition.TargetPreflight.SubjectDigest != target.TargetDigest ||
		*status.Transition.TargetPreflight.Evidence != target.Validation {
		return false, false, errors.New("durable membership target lacks matching target validation")
	}
	if status.Membership.Observed.RequestedTransitionID != target.TransitionID {
		return false, false, errors.New("membership observation is not scoped to the desired transition")
	}
	observation := status.Membership.Observed.Transition
	if observation == nil {
		if !sameTopology(base, observedTopology) {
			return c.blockUnknownMembership(
				status,
				"TopologyChangedWithoutTransition",
				"engine topology changed while the desired membership transition was absent",
			)
		}
		if applyErr := c.membership.Apply(ctx, groupID, target); applyErr != nil {
			return false, false, fmt.Errorf("apply membership target: %w", applyErr)
		}
		return false, false, nil
	}

	switch observation.Phase {
	case MembershipTransitionPhasePending:
		if !sameTopology(base, observedTopology) {
			return c.blockUnknownMembership(
				status,
				"TopologyChangedWhilePending",
				"engine topology changed before the transition reported commit",
			)
		}
		return false, false, nil
	case MembershipTransitionPhaseCommitted:
		return c.recordCommittedMembership(status, base, observedTopology, resolution, *observation)
	case MembershipTransitionPhaseRejected:
		if !sameTopology(base, observedTopology) {
			return c.blockUnknownMembership(
				status,
				"RejectedAfterTopologyChanged",
				"membership rejection is unsafe to roll back because committed topology changed",
			)
		}
		c.beginRollback(status, *observation.Failure)
		return false, true, nil
	case MembershipTransitionPhaseUnknown:
		message := "adapter cannot establish the membership transition outcome"
		if observation.Failure != nil && observation.Failure.Message != "" {
			message = observation.Failure.Message
		}
		return c.blockUnknownMembership(
			status,
			"UnknownMembershipOutcome",
			message,
		)
	default:
		return false, false, fmt.Errorf("invalid membership transition phase %q", observation.Phase)
	}
}

func (c *Coordinator) recordCommittedMembership(
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
	observation MembershipTransitionObservation,
) (ready bool, persist bool, err error) {
	if sameTopology(observedTopology, base) {
		return false, false, nil
	}
	if !sameTopology(observedTopology, *observation.ResultTopology) {
		return c.blockUnknownMembership(
			status,
			"ConflictingTopologyObservation",
			"authoritative topology matches neither the base nor the correlated transition result",
		)
	}
	if err := validateCommittedTopology(
		base,
		resolution,
		status.Membership.Desired.Joining,
		*observation.ResultTopology,
	); err != nil {
		return c.blockUnknownMembership(status, "InvalidCommittedTopology", err.Error())
	}

	alreadyCurrent := sameTopologyWithCurrent(status.Topologies, *observation.ResultTopology)
	history, err := appendTopology(status.Topologies, *observation.ResultTopology)
	if err != nil {
		return false, false, err
	}
	if alreadyCurrent {
		return true, false, nil
	}
	status.Topologies = history
	status.Transition.UpdatedAt = c.now()
	return false, true, nil
}

func membershipCommittedForTransition(status GroupStatus) bool {
	if status.Transition == nil || status.Membership.Desired == nil ||
		status.Membership.Observed.Transition == nil {
		return false
	}
	return membershipTargetMatchesTransition(*status.Membership.Desired, *status.Transition) &&
		status.Membership.Observed.RequestedTransitionID == status.Membership.Desired.TransitionID &&
		status.Membership.Observed.Transition.Phase == MembershipTransitionPhaseCommitted
}

func membershipTargetMatchesTransition(target MembershipTarget, transition TransitionStatus) bool {
	return target.TransitionID == membershipTransitionID(
		transition.Spec.Plan.ID,
		transition.Spec.BaseTopologyGeneration,
	) && target.BaseTopology.Generation == transition.Spec.BaseTopologyGeneration &&
		sameResolvedPlan(target.Plan, transition.Spec.Plan)
}

func membershipTargetMayBeReplaced(status MembershipStatus) bool {
	if status.Desired == nil {
		return true
	}
	observed := status.Observed.Transition
	if status.Observed.RequestedTransitionID != status.Desired.TransitionID ||
		observed == nil || observed.TransitionID != status.Desired.TransitionID ||
		observed.ControlRevision != status.Desired.ControlRevision ||
		observed.TargetDigest != status.Desired.TargetDigest {
		return false
	}
	return observed.Phase == MembershipTransitionPhaseCommitted ||
		observed.Phase == MembershipTransitionPhaseRejected
}

func (c *Coordinator) blockUnknownMembership(
	status *GroupStatus,
	reason string,
	message string,
) (ready bool, persist bool, err error) {
	failure := Failure{
		Classification: FailureClassificationTerminal,
		Reason:         reason,
		Message:        message,
	}
	c.blockTransition(status, failure)
	return false, true, nil
}

func validateCommittedTopology(
	base MembershipTopology,
	resolution planResolution,
	joining []JoiningReplica,
	committed MembershipTopology,
) error {
	if err := validateTopology(committed); err != nil {
		return err
	}
	if committed.Generation <= base.Generation {
		return fmt.Errorf(
			"committed topology generation %d does not advance base generation %d",
			committed.Generation,
			base.Generation,
		)
	}
	if !sameReplicaIDs(topologyReplicaIDs(committed), resolution.targetReplicaIDs) {
		return fmt.Errorf(
			"committed replica set %v does not match target %v",
			topologyReplicaIDs(committed),
			resolution.targetReplicaIDs,
		)
	}

	joiningByID := make(map[ReplicaID]JoiningReplica, len(joining))
	for _, replica := range joining {
		joiningByID[replica.ReplicaID] = replica
	}
	plannedJoining := make(map[ReplicaID][]NativeMemberID, len(resolution.joiningTargets))
	for _, target := range resolution.joiningTargets {
		plannedJoining[target.ReplicaID] = joiningNativeMembers(resolution, target)
	}
	remappedByID := nativeMembershipByID(resolution.remappedMembership)
	for _, membership := range committed.Replicas {
		replicaID := membership.ReplicaID
		if joiningReplica, found := joiningByID[replicaID]; found {
			if membership.RuntimeIncarnation != joiningReplica.RuntimeIncarnation {
				return fmt.Errorf("joining replica %q committed another runtime incarnation", replicaID)
			}
			if planned := plannedJoining[replicaID]; len(planned) > 0 &&
				!slices.Equal(normalizeNativeMembers(planned), normalizeNativeMembers(membership.NativeMembers)) {
				return fmt.Errorf("joining replica %q committed another native membership", replicaID)
			}
			continue
		}

		baseMembership, retained := membershipByID(base, replicaID)
		if !retained {
			return fmt.Errorf("committed replica %q is neither retained nor joining", replicaID)
		}
		if baseMembership.RuntimeIncarnation != membership.RuntimeIncarnation {
			return fmt.Errorf("retained replica %q changed physical incarnation", replicaID)
		}
		if remapped, remap := remappedByID[replicaID]; remap {
			if !slices.Equal(
				normalizeNativeMembers(remapped.NativeMembers),
				normalizeNativeMembers(membership.NativeMembers),
			) {
				return fmt.Errorf("remapped replica %q committed another native membership", replicaID)
			}
			continue
		}
		if !slices.Equal(
			normalizeNativeMembers(baseMembership.NativeMembers),
			normalizeNativeMembers(membership.NativeMembers),
		) {
			return fmt.Errorf("retained replica %q changed native membership", replicaID)
		}
	}
	return nil
}

func nativeMembershipByID(values []ReplicaNativeMembership) map[ReplicaID]ReplicaNativeMembership {
	byID := make(map[ReplicaID]ReplicaNativeMembership, len(values))
	for _, value := range values {
		value.NativeMembers = slices.Clone(value.NativeMembers)
		byID[value.ReplicaID] = value
	}
	return byID
}

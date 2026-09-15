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
	"strings"
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

func (c *Coordinator) reconcileRollbackCapacity(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	target, err := buildRollbackCapacityTarget(*status, base, resolution)
	if err != nil {
		return false, false, err
	}
	if status.Capacity.Desired == nil || !sameCapacityTargetIntent(*status.Capacity.Desired, target) {
		revision, revisionErr := c.nextControlRevision(status)
		if revisionErr != nil {
			return false, false, revisionErr
		}
		target.ControlRevision = revision
		status.Capacity.Desired = &target
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}

	desired := *status.Capacity.Desired
	if status.Capacity.Observed.AppliedRevision < desired.ControlRevision {
		result, applyErr := c.capacity.Apply(ctx, groupID, desired)
		if applyErr != nil {
			return false, false, fmt.Errorf("apply rollback capacity target: %w", applyErr)
		}
		if result.Rejection != nil {
			if rejectionErr := validateRejection(result.Rejection); rejectionErr != nil {
				return false, false, fmt.Errorf("invalid rollback capacity rejection: %w", rejectionErr)
			}
			c.blockTransition(status, *result.Rejection)
			return false, true, nil
		}
		return false, false, nil
	}
	if !capacityReleaseConverged(desired, status.Capacity.Observed) {
		return false, false, nil
	}

	changed, err := discardUncommittedJoining(&status.Registry, resolution)
	if err != nil {
		return false, false, err
	}
	if changed {
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}
	return true, false, nil
}

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
	if status.Traffic.Observed.AppliedRevision < desired.ControlRevision {
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
	return trafficTargetConverged(desired, status.Traffic.Observed), false, nil
}

func (c *Coordinator) reconcileJoiningCapacity(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	if len(resolution.joiningTargets) == 0 {
		return true, false, nil
	}

	target, err := buildCapacityTarget(*status, resolution, nil)
	if err != nil {
		return false, false, err
	}
	if status.Capacity.Desired == nil ||
		!sameCapacityTargetIntent(*status.Capacity.Desired, target) {
		revision, revisionErr := c.nextControlRevision(status)
		if revisionErr != nil {
			return false, false, revisionErr
		}
		target.ControlRevision = revision
		status.Capacity.Desired = &target
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}

	desired := *status.Capacity.Desired
	if status.Capacity.Observed.AppliedRevision < desired.ControlRevision {
		result, applyErr := c.capacity.Apply(ctx, groupID, desired)
		if applyErr != nil {
			return false, false, fmt.Errorf("apply joining capacity target: %w", applyErr)
		}
		if result.Rejection != nil {
			if rejectionErr := validateRejection(result.Rejection); rejectionErr != nil {
				return false, false, fmt.Errorf("invalid joining capacity rejection: %w", rejectionErr)
			}
			c.beginRollback(status, *result.Rejection)
			return false, true, nil
		}
		return false, false, nil
	}
	if !capacityAllocationsConverged(desired, status.Capacity.Observed, true) {
		return false, false, nil
	}

	// Freeze the workload manager's concrete joining incarnations in the canonical registry before membership submit.
	changed, err := bindJoiningIncarnations(&status.Registry, resolution, status.Capacity.Observed)
	if err != nil {
		return false, false, err
	}
	if changed {
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}
	return true, false, nil
}

func (c *Coordinator) reconcilePreMembershipTraffic(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	if status.Transition.Membership.Phase == MembershipOperationPhaseCommitted {
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
	if status.Traffic.Observed.AppliedRevision < desired.ControlRevision {
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
	return trafficTargetConverged(desired, status.Traffic.Observed), false, nil
}

func (c *Coordinator) reconcileMembership(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	operation := &status.Transition.Membership
	if operation.Phase == MembershipOperationPhaseNotStarted {
		if !sameTopology(base, observedTopology) {
			return false, false, errors.New("engine topology changed before membership submission")
		}
		joining, freezeErr := joiningReplicaIdentities(status.Registry, resolution)
		if freezeErr != nil {
			return false, false, freezeErr
		}
		operation.JoiningReplicas = joining
		operation.Phase = MembershipOperationPhasePrepared
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}
	if operation.Phase == MembershipOperationPhaseCommitted {
		committed, found := status.Topologies.Snapshot(operation.CommittedTopologyGeneration)
		if !found {
			return false, false, errors.New("membership status references an absent committed topology")
		}
		if !sameTopology(committed, observedTopology) {
			return false, false, errors.New("engine topology diverged after membership commit")
		}
		return true, false, nil
	}
	if operation.Phase == MembershipOperationPhaseRejected ||
		operation.Phase == MembershipOperationPhaseUnknown {
		return false, false, nil
	}

	observation, err := c.membership.ObserveOperation(ctx, groupID, operation.ID)
	if err != nil {
		return false, false, fmt.Errorf("observe membership operation: %w", err)
	}
	if observation.ID != "" && observation.ID != operation.ID {
		return false, false, fmt.Errorf(
			"membership observation ID %q does not match operation %q",
			observation.ID,
			operation.ID,
		)
	}
	if observationErr := validateMembershipObservation(observation); observationErr != nil {
		return c.blockUnknownMembership(status, "InvalidMembershipObservation", observationErr.Error())
	}

	return c.reconcileMembershipObservation(ctx, groupID, status, base, observedTopology, resolution, observation)
}

func (c *Coordinator) reconcileMembershipObservation(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
	observation MembershipOperationObservation,
) (ready bool, persist bool, err error) {
	switch observation.Phase {
	case MembershipBackendPhaseAbsent:
		return c.submitPreparedMembership(ctx, groupID, status, base, observedTopology)
	case MembershipBackendPhaseRunning:
		return c.recordRunningMembership(status, base, observedTopology)
	case MembershipBackendPhaseCommitted:
		return c.recordCommittedMembership(status, base, observedTopology, resolution, observation)
	case MembershipBackendPhaseRejected:
		return c.recordRejectedMembership(status, observation)
	case MembershipBackendPhaseUnknown:
		message := "engine cannot establish the membership operation outcome"
		if observation.Failure != nil && observation.Failure.Message != "" {
			message = observation.Failure.Message
		}
		return c.blockUnknownMembership(status, "UnknownMembershipOutcome", message)
	default:
		return false, false, fmt.Errorf("invalid membership backend phase %q", observation.Phase)
	}
}

func (c *Coordinator) submitPreparedMembership(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
) (ready bool, persist bool, err error) {
	if !sameTopology(base, observedTopology) {
		return c.blockUnknownMembership(
			status,
			"TopologyChangedWithoutOperation",
			"engine topology changed while the membership operation was absent",
		)
	}
	request := MembershipRequest{
		ID:              status.Transition.Membership.ID,
		BaseTopology:    cloneTopology(base),
		Plan:            cloneResolvedPlan(status.Transition.Spec.Plan),
		JoiningReplicas: slices.Clone(status.Transition.Membership.JoiningReplicas),
	}
	result, submitErr := c.membership.Submit(ctx, groupID, request)
	if submitErr != nil {
		return false, false, fmt.Errorf("submit membership operation: %w", submitErr)
	}
	if result.Rejection == nil {
		return false, false, nil
	}
	if err := validateRejection(result.Rejection); err != nil {
		return false, false, fmt.Errorf("invalid membership rejection: %w", err)
	}

	status.Transition.Membership.Phase = MembershipOperationPhaseRejected
	status.Transition.Membership.Failure = cloneFailure(result.Rejection)
	c.beginRollback(status, *result.Rejection)
	return false, true, nil
}

func (c *Coordinator) recordRunningMembership(
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
) (ready bool, persist bool, err error) {
	if !sameTopology(base, observedTopology) {
		return c.blockUnknownMembership(
			status,
			"TopologyChangedWhileRunning",
			"engine topology changed before the operation reported commit",
		)
	}
	if status.Transition.Membership.Phase == MembershipOperationPhaseRunning {
		return false, false, nil
	}

	status.Transition.Membership.Phase = MembershipOperationPhaseRunning
	status.Transition.UpdatedAt = c.now()
	return false, true, nil
}

func (c *Coordinator) recordCommittedMembership(
	status *GroupStatus,
	base MembershipTopology,
	observedTopology MembershipTopology,
	resolution planResolution,
	observation MembershipOperationObservation,
) (ready bool, persist bool, err error) {
	if !sameTopology(observedTopology, base) && !sameTopology(observedTopology, *observation.CommittedTopology) {
		return c.blockUnknownMembership(
			status,
			"ConflictingTopologyObservation",
			"observed topology matches neither the base nor the correlated commit",
		)
	}
	if err := validateCommittedTopology(
		base,
		resolution,
		status.Transition.Membership.JoiningReplicas,
		*observation.CommittedTopology,
	); err != nil {
		return c.blockUnknownMembership(status, "InvalidCommittedTopology", err.Error())
	}

	history, err := appendTopology(status.Topologies, *observation.CommittedTopology)
	if err != nil {
		return false, false, err
	}
	status.Topologies = history
	status.Transition.Membership.Phase = MembershipOperationPhaseCommitted
	status.Transition.Membership.CommittedTopologyGeneration = observation.CommittedTopology.Generation
	status.Transition.Membership.Failure = nil
	status.Transition.UpdatedAt = c.now()
	return false, true, nil
}

func (c *Coordinator) recordRejectedMembership(
	status *GroupStatus,
	observation MembershipOperationObservation,
) (ready bool, persist bool, err error) {
	status.Transition.Membership.Phase = MembershipOperationPhaseRejected
	status.Transition.Membership.Failure = cloneFailure(observation.Failure)
	c.beginRollback(status, *observation.Failure)
	return false, true, nil
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
	status.Transition.Membership.Phase = MembershipOperationPhaseUnknown
	status.Transition.Membership.Failure = cloneFailure(&failure)
	c.blockTransition(status, failure)
	return false, true, nil
}

func (c *Coordinator) reconcileRetiredCapacity(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
	base MembershipTopology,
	committed MembershipTopology,
	resolution planResolution,
) (ready bool, persist bool, err error) {
	if len(resolution.retiringReplicaIDs) == 0 {
		return true, false, nil
	}

	releaseFences, err := releaseFencesFor(base, status.Registry, resolution.retiringReplicaIDs)
	if err != nil {
		return false, false, err
	}
	target, err := buildCapacityTarget(*status, resolution, releaseFences)
	if err != nil {
		return false, false, err
	}
	if status.Capacity.Desired == nil ||
		!sameCapacityTargetIntent(*status.Capacity.Desired, target) {
		revision, revisionErr := c.nextControlRevision(status)
		if revisionErr != nil {
			return false, false, revisionErr
		}
		target.ControlRevision = revision
		status.Capacity.Desired = &target
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}

	desired := *status.Capacity.Desired
	if status.Capacity.Observed.AppliedRevision < desired.ControlRevision {
		result, applyErr := c.capacity.Apply(ctx, groupID, desired)
		if applyErr != nil {
			return false, false, fmt.Errorf("apply retired capacity target: %w", applyErr)
		}
		if result.Rejection != nil {
			if rejectionErr := validateRejection(result.Rejection); rejectionErr != nil {
				return false, false, fmt.Errorf("invalid retired capacity rejection: %w", rejectionErr)
			}
			c.blockTransition(status, *result.Rejection)
			return false, true, nil
		}
		return false, false, nil
	}
	if !capacityReleaseConverged(desired, status.Capacity.Observed) {
		return false, false, nil
	}

	// Archive each retired incarnation only after the exact UID fence and physical absence are observable.
	changed, err := archiveRetiredReplicas(&status.Registry, base, committed, resolution.retiringReplicaIDs)
	if err != nil {
		return false, false, err
	}
	if changed {
		status.Transition.UpdatedAt = c.now()
		return false, true, nil
	}
	return true, false, nil
}

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
			verification.Proof.RuntimeDigest != topologyRuntimeDigest(committed) {
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
		if failureErr := validateRejection(result.Failure); failureErr != nil {
			return false, false, fmt.Errorf("invalid serving verification failure: %w", failureErr)
		}
		verification.Phase = VerificationPhaseFailed
		verification.Failure = cloneFailure(result.Failure)
		c.blockTransition(status, *result.Failure)
		return false, true, nil
	}
	if result.Proof.TopologyGeneration != committed.Generation ||
		result.Proof.RuntimeDigest != topologyRuntimeDigest(committed) {
		return false, false, errors.New("serving verifier returned proof for another topology")
	}

	verification.Phase = VerificationPhasePassed
	verification.Proof = cloneServingProof(result.Proof)
	verification.Failure = nil
	status.Transition.UpdatedAt = c.now()
	return false, true, nil
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
	if status.Traffic.Observed.AppliedRevision < desired.ControlRevision {
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
	return trafficTargetConverged(desired, status.Traffic.Observed), false, nil
}

func buildCapacityTarget(
	status GroupStatus,
	resolution planResolution,
	releaseFences []ReleaseFence,
) (CapacityTarget, error) {
	joiningByID := make(map[ReplicaID]ReplicaTarget, len(resolution.joiningTargets))
	for _, target := range resolution.joiningTargets {
		joiningByID[target.ReplicaID] = target
	}

	replicas := make([]CapacityReplicaTarget, 0, len(resolution.targetReplicaIDs))
	for _, replicaID := range resolution.targetReplicaIDs {
		record, found := status.Registry.Find(replicaID)
		if !found {
			return CapacityTarget{}, fmt.Errorf("target replica %q has no canonical registry record", replicaID)
		}
		target := CapacityReplicaTarget{ReplicaID: replicaID, SlotID: record.SlotID}
		if joining, found := joiningByID[replicaID]; found {
			target.Bootstrap = joining.Bootstrap
		} else if record.Current != nil {
			incarnation := cloneReplicaIncarnation(*record.Current)
			target.Incarnation = &incarnation
		} else {
			return CapacityTarget{}, fmt.Errorf("target replica %q has neither capacity nor bootstrap intent", replicaID)
		}
		replicas = append(replicas, target)
	}
	slices.SortFunc(replicas, func(left, right CapacityReplicaTarget) int {
		return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
	})

	return CapacityTarget{
		TransitionID:          status.Transition.Spec.ID,
		ProfileFingerprint:    status.Transition.Spec.Plan.ProfileFingerprint,
		ProcessLifecycleOwner: status.Transition.Spec.Plan.ProcessLifecycleOwner,
		Replicas:              replicas,
		ReleaseFences:         cloneReleaseFences(releaseFences),
	}, nil
}

func buildRollbackCapacityTarget(
	status GroupStatus,
	base MembershipTopology,
	resolution planResolution,
) (CapacityTarget, error) {
	replicas := make([]CapacityReplicaTarget, 0, len(base.Replicas))
	for _, membership := range base.Replicas {
		record, found := status.Registry.Find(membership.ReplicaID)
		if !found || record.Current == nil {
			return CapacityTarget{}, fmt.Errorf(
				"base replica %q has no canonical capacity for rollback",
				membership.ReplicaID,
			)
		}
		if !membershipMatchesIncarnation(membership, *record.Current) {
			return CapacityTarget{}, fmt.Errorf(
				"base replica %q changed incarnation before rollback",
				membership.ReplicaID,
			)
		}
		incarnation := cloneReplicaIncarnation(*record.Current)
		replicas = append(replicas, CapacityReplicaTarget{
			ReplicaID:   record.ReplicaID,
			SlotID:      record.SlotID,
			Incarnation: &incarnation,
		})
	}

	releaseFences := make([]ReleaseFence, 0, len(resolution.joiningTargets))
	for _, joining := range resolution.joiningTargets {
		record, found := status.Registry.Find(joining.ReplicaID)
		if !found {
			continue
		}
		incarnation := record.Current
		if incarnation == nil {
			allocation, allocated := allocationByID(status.Capacity.Observed, joining.ReplicaID)
			if !allocated {
				continue
			}
			if allocation.Incarnation.SlotID != joining.SlotID {
				return CapacityTarget{}, fmt.Errorf(
					"uncommitted replica %q occupies unexpected slot %q",
					joining.ReplicaID,
					allocation.Incarnation.SlotID,
				)
			}
			observed := cloneReplicaIncarnation(allocation.Incarnation)
			incarnation = &observed
		}
		releaseFences = append(releaseFences, ReleaseFence{
			ReplicaID:    record.ReplicaID,
			SlotID:       record.SlotID,
			CapacityRefs: cloneCapacityRefs(incarnation.CapacityRefs),
		})
	}
	slices.SortFunc(replicas, func(left, right CapacityReplicaTarget) int {
		return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
	})

	return CapacityTarget{
		TransitionID:          status.Transition.Spec.ID,
		ProfileFingerprint:    status.Transition.Spec.Plan.ProfileFingerprint,
		ProcessLifecycleOwner: status.Transition.Spec.Plan.ProcessLifecycleOwner,
		Replicas:              replicas,
		ReleaseFences:         normalizeReleaseFences(releaseFences),
	}, nil
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

	drain := make([]ReplicaMembership, 0)
	if status.Transition.Spec.Plan.TrafficRequirement == TrafficRequirementQuiesceGroup {
		drain = cloneReplicaMemberships(base.Replicas)
	} else if status.Transition.Spec.Plan.RetirementSafety == RetirementSafetyDrained {
		for _, replicaID := range resolution.retiringReplicaIDs {
			membership, _ := membershipByID(base, replicaID)
			drain = append(drain, membership)
		}
	}

	return TrafficTarget{
		TransitionID:       status.Transition.Spec.ID,
		TopologyGeneration: base.Generation,
		Admitted:           normalizeMemberships(admitted),
		Drain:              normalizeMemberships(drain),
	}
}

func buildCommittedTrafficTarget(
	status GroupStatus,
	base MembershipTopology,
	committed MembershipTopology,
	resolution planResolution,
) TrafficTarget {
	drain := make([]ReplicaMembership, 0, len(resolution.retiringReplicaIDs))
	if status.Transition.Spec.Plan.RetirementSafety == RetirementSafetyDrained {
		for _, replicaID := range resolution.retiringReplicaIDs {
			membership, _ := membershipByID(base, replicaID)
			drain = append(drain, membership)
		}
	}
	return TrafficTarget{
		TransitionID:       status.Transition.Spec.ID,
		TopologyGeneration: committed.Generation,
		Admitted:           normalizeMemberships(committed.Replicas),
		Drain:              normalizeMemberships(drain),
	}
}

func cloneReleaseFences(values []ReleaseFence) []ReleaseFence {
	cloned := make([]ReleaseFence, 0, len(values))
	for _, value := range values {
		value.CapacityRefs = cloneCapacityRefs(value.CapacityRefs)
		cloned = append(cloned, value)
	}
	return cloned
}

func sameCapacityTargetIntent(left, right CapacityTarget) bool {
	if left.TransitionID != right.TransitionID ||
		left.ProfileFingerprint != right.ProfileFingerprint ||
		left.ProcessLifecycleOwner != right.ProcessLifecycleOwner ||
		len(left.Replicas) != len(right.Replicas) ||
		len(left.ReleaseFences) != len(right.ReleaseFences) {
		return false
	}

	leftReplicas := normalizeCapacityReplicaTargets(left.Replicas)
	rightReplicas := normalizeCapacityReplicaTargets(right.Replicas)
	for index := range leftReplicas {
		if !sameCapacityReplicaTarget(leftReplicas[index], rightReplicas[index]) {
			return false
		}
	}
	return sameReleaseFences(left.ReleaseFences, right.ReleaseFences)
}

func normalizeCapacityReplicaTargets(values []CapacityReplicaTarget) []CapacityReplicaTarget {
	normalized := make([]CapacityReplicaTarget, 0, len(values))
	for _, value := range values {
		if value.Incarnation != nil {
			incarnation := normalizeIncarnation(*value.Incarnation)
			value.Incarnation = &incarnation
		}
		normalized = append(normalized, value)
	}
	slices.SortFunc(normalized, func(left, right CapacityReplicaTarget) int {
		return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
	})
	return normalized
}

func sameCapacityReplicaTarget(left, right CapacityReplicaTarget) bool {
	if left.ReplicaID != right.ReplicaID || left.SlotID != right.SlotID || left.Bootstrap != right.Bootstrap {
		return false
	}
	if left.Incarnation == nil || right.Incarnation == nil {
		return left.Incarnation == nil && right.Incarnation == nil
	}
	return sameIncarnation(*left.Incarnation, *right.Incarnation)
}

func normalizeReleaseFences(values []ReleaseFence) []ReleaseFence {
	normalized := cloneReleaseFences(values)
	for index := range normalized {
		normalized[index].CapacityRefs = normalizeCapacityRefs(normalized[index].CapacityRefs)
	}
	slices.SortFunc(normalized, func(left, right ReleaseFence) int {
		return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
	})
	return normalized
}

func sameReleaseFences(left, right []ReleaseFence) bool {
	left = normalizeReleaseFences(left)
	right = normalizeReleaseFences(right)
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].ReplicaID != right[index].ReplicaID ||
			left[index].SlotID != right[index].SlotID ||
			!sameCapacityRefs(left[index].CapacityRefs, right[index].CapacityRefs) {
			return false
		}
	}
	return true
}

func capacityAllocationsConverged(
	target CapacityTarget,
	observation CapacityObservation,
	requireAvailable bool,
) bool {
	if observation.AppliedRevision < target.ControlRevision ||
		len(observation.Allocations) != len(target.Replicas) {
		return false
	}
	for _, replica := range target.Replicas {
		allocation, found := allocationByID(observation, replica.ReplicaID)
		if !found || allocation.Incarnation.SlotID != replica.SlotID || requireAvailable && !allocation.Available {
			return false
		}
		if replica.Incarnation != nil && !sameIncarnation(*replica.Incarnation, allocation.Incarnation) {
			return false
		}
	}
	return true
}

func capacityReleaseConverged(target CapacityTarget, observation CapacityObservation) bool {
	if !capacityAllocationsConverged(target, observation, false) {
		return false
	}
	for _, desiredFence := range target.ReleaseFences {
		if _, found := allocationByID(observation, desiredFence.ReplicaID); found {
			return false
		}
		matched := false
		for _, observedFence := range observation.ReleaseFences {
			if sameReleaseFences([]ReleaseFence{desiredFence}, []ReleaseFence{observedFence}) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return true
}

func bindJoiningIncarnations(
	registry *ReplicaRegistry,
	resolution planResolution,
	observation CapacityObservation,
) (bool, error) {
	changed := false
	for _, target := range resolution.joiningTargets {
		allocation, found := allocationByID(observation, target.ReplicaID)
		if !found || !allocation.Available {
			return false, fmt.Errorf("joining replica %q has no available allocation", target.ReplicaID)
		}
		if allocation.Incarnation.SlotID != target.SlotID {
			return false, fmt.Errorf(
				"joining replica %q was allocated in slot %q instead of %q",
				target.ReplicaID,
				allocation.Incarnation.SlotID,
				target.SlotID,
			)
		}

		index, found := registryRecordIndex(*registry, target.ReplicaID)
		if !found {
			return false, fmt.Errorf("joining replica %q has no canonical record", target.ReplicaID)
		}
		if registry.Replicas[index].Current == nil {
			incarnation := cloneReplicaIncarnation(allocation.Incarnation)
			registry.Replicas[index].Current = &incarnation
			changed = true
			continue
		}
		if !sameIncarnation(*registry.Replicas[index].Current, allocation.Incarnation) {
			return false, fmt.Errorf("joining replica %q changed physical incarnation", target.ReplicaID)
		}
	}
	return changed, nil
}

func registryRecordIndex(registry ReplicaRegistry, replicaID ReplicaID) (int, bool) {
	for index := range registry.Replicas {
		if registry.Replicas[index].ReplicaID == replicaID {
			return index, true
		}
	}
	return 0, false
}

func joiningReplicaIdentities(
	registry ReplicaRegistry,
	resolution planResolution,
) ([]JoiningReplica, error) {
	joining := make([]JoiningReplica, 0, len(resolution.joiningTargets))
	for _, target := range resolution.joiningTargets {
		record, found := registry.Find(target.ReplicaID)
		if !found || record.Current == nil {
			return nil, fmt.Errorf("joining replica %q has no frozen physical incarnation", target.ReplicaID)
		}
		joining = append(joining, JoiningReplica{
			ReplicaID:          record.ReplicaID,
			RuntimeIncarnation: record.Current.RuntimeIncarnation,
		})
	}
	slices.SortFunc(joining, func(left, right JoiningReplica) int {
		return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
	})
	return joining, nil
}

func releaseFencesFor(
	base MembershipTopology,
	registry ReplicaRegistry,
	retiringReplicaIDs []ReplicaID,
) ([]ReleaseFence, error) {
	fences := make([]ReleaseFence, 0, len(retiringReplicaIDs))
	for _, replicaID := range retiringReplicaIDs {
		membership, found := membershipByID(base, replicaID)
		if !found {
			return nil, fmt.Errorf("retiring replica %q is absent from base topology", replicaID)
		}
		record, found := registry.Find(replicaID)
		if !found {
			return nil, fmt.Errorf("retiring replica %q has no canonical record", replicaID)
		}
		if record.Current != nil && !membershipMatchesIncarnation(membership, *record.Current) {
			return nil, fmt.Errorf("retiring replica %q canonical incarnation changed", replicaID)
		}
		if record.Current == nil && len(record.History) == 0 {
			return nil, fmt.Errorf("retiring replica %q has no current or historical incarnation", replicaID)
		}
		capacityRefs := []CapacityRef(nil)
		if record.Current != nil {
			capacityRefs = cloneCapacityRefs(record.Current.CapacityRefs)
		} else {
			capacityRefs = cloneCapacityRefs(record.History[len(record.History)-1].Incarnation.CapacityRefs)
		}
		fences = append(fences, ReleaseFence{
			ReplicaID:    replicaID,
			SlotID:       record.SlotID,
			CapacityRefs: capacityRefs,
		})
	}
	return normalizeReleaseFences(fences), nil
}

func archiveRetiredReplicas(
	registry *ReplicaRegistry,
	base MembershipTopology,
	committed MembershipTopology,
	retiringReplicaIDs []ReplicaID,
) (bool, error) {
	changed := false
	for _, replicaID := range retiringReplicaIDs {
		if _, retained := membershipByID(committed, replicaID); retained {
			return false, fmt.Errorf("retired replica %q remains in committed membership", replicaID)
		}
		index, found := registryRecordIndex(*registry, replicaID)
		if !found {
			return false, fmt.Errorf("retired replica %q has no canonical record", replicaID)
		}
		if registry.Replicas[index].Current == nil {
			continue
		}
		baseMembership, _ := membershipByID(base, replicaID)
		if !membershipMatchesIncarnation(baseMembership, *registry.Replicas[index].Current) {
			return false, fmt.Errorf("retired replica %q canonical incarnation changed before archival", replicaID)
		}
		registry.Replicas[index].History = append(
			registry.Replicas[index].History,
			ReplicaHistoryEntry{
				TopologyGeneration: base.Generation,
				Incarnation:        cloneReplicaIncarnation(*registry.Replicas[index].Current),
				NativeMembers:      cloneNativeMembers(baseMembership.NativeMembers),
			},
		)
		registry.Replicas[index].Current = nil
		changed = true
	}
	return changed, nil
}

func discardUncommittedJoining(
	registry *ReplicaRegistry,
	resolution planResolution,
) (bool, error) {
	joining := make(map[ReplicaID]struct{}, len(resolution.joiningTargets))
	for _, target := range resolution.joiningTargets {
		joining[target.ReplicaID] = struct{}{}
	}
	if len(joining) == 0 {
		return false, nil
	}

	changed := false
	retained := make([]ReplicaRecord, 0, len(registry.Replicas))
	for _, record := range registry.Replicas {
		if _, uncommitted := joining[record.ReplicaID]; !uncommitted {
			retained = append(retained, record)
			continue
		}
		if record.Current != nil {
			record.Current = nil
			changed = true
		}
		if resolution.kind == PlanKindGrow {
			changed = true
			continue
		}
		retained = append(retained, record)
	}
	if len(retained) != len(registry.Replicas) {
		changed = true
	}
	registry.Replicas = retained
	if err := validateRegistry(*registry); err != nil {
		return false, err
	}
	return changed, nil
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
	restoredByID := nativeMembershipByID(resolution.restoredMembership)
	remappedByID := nativeMembershipByID(resolution.remappedMembership)
	for _, membership := range committed.Replicas {
		replicaID := membership.ReplicaID
		if joiningReplica, found := joiningByID[replicaID]; found {
			if membership.RuntimeIncarnation != joiningReplica.RuntimeIncarnation {
				return fmt.Errorf("joining replica %q committed another runtime incarnation", replicaID)
			}
			if restored, restore := restoredByID[replicaID]; restore &&
				!slices.Equal(
					normalizeNativeMembers(restored.NativeMembers),
					normalizeNativeMembers(membership.NativeMembers),
				) {
				return fmt.Errorf("restored replica %q committed another native membership", replicaID)
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
		value.NativeMembers = cloneNativeMembers(value.NativeMembers)
		byID[value.ReplicaID] = value
	}
	return byID
}

func sameTrafficTargetIntent(left, right TrafficTarget) bool {
	return left.TransitionID == right.TransitionID &&
		left.TopologyGeneration == right.TopologyGeneration &&
		sameMemberships(left.Admitted, right.Admitted) &&
		sameMemberships(left.Drain, right.Drain)
}

func trafficTargetConverged(target TrafficTarget, observation TrafficObservation) bool {
	if observation.AppliedRevision < target.ControlRevision ||
		!sameMemberships(target.Admitted, observation.Admitted) {
		return false
	}
	for _, required := range target.Drain {
		if !containsMembership(observation.Drained, required) {
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

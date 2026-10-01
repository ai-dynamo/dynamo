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
	"errors"
	"fmt"
)

func validateGroupStatus(status GroupStatus) error {
	if status.ControlRevision < 0 {
		return errors.New("control revision must not be negative")
	}
	if err := validateRegistry(status.Registry); err != nil {
		return fmt.Errorf("validate replica registry: %w", err)
	}
	if err := validateTopologyHistory(status.Topologies); err != nil {
		return fmt.Errorf("validate topology history: %w", err)
	}
	if err := validateCapacityObservation(status.Capacity.Observed); err != nil {
		return fmt.Errorf("validate durable capacity observation: %w", err)
	}
	if err := validateTrafficObservation(status.Traffic.Observed); err != nil {
		return fmt.Errorf("validate durable traffic observation: %w", err)
	}
	if err := validateCapacityTarget(status.ControlRevision, status.Capacity.Desired); err != nil {
		return err
	}
	if err := validateCapacityTarget(status.ControlRevision, status.Capacity.Accepted); err != nil {
		return fmt.Errorf("validate accepted capacity target: %w", err)
	}
	if err := validateTrafficTarget(status.ControlRevision, status.Traffic.Desired); err != nil {
		return err
	}
	if err := validateTrafficTarget(status.ControlRevision, status.Traffic.Accepted); err != nil {
		return fmt.Errorf("validate accepted traffic target: %w", err)
	}
	if err := validateAcceptedTargets(status); err != nil {
		return err
	}
	if err := validateMembershipTarget(status.ControlRevision, status.Membership.Desired); err != nil {
		return err
	}
	if err := validateMembershipObservation(status.Membership.Observed, status.Membership.Desired); err != nil {
		return fmt.Errorf("validate durable membership observation: %w", err)
	}

	current, found := status.Topologies.Current()
	if !found {
		return errors.New("current topology is absent")
	}
	for _, membership := range current.Replicas {
		record, found := status.Registry.Find(membership.ReplicaID)
		if !found || record.Current == nil {
			return fmt.Errorf(
				"current member %q has no canonical current incarnation",
				membership.ReplicaID,
			)
		}
		if !membershipMatchesIncarnation(membership, *record.Current) {
			return fmt.Errorf(
				"current member %q differs from its canonical incarnation",
				membership.ReplicaID,
			)
		}
	}

	if status.Transition != nil {
		if err := validateTransition(status); err != nil {
			return fmt.Errorf("validate transition: %w", err)
		}
	}
	return nil
}

// TerminalDeletionEvidenceReady validates one fresh cross-subsystem observation and reports whether it proves that
// finalizer removal cannot leave serving state behind or allow an accepted non-empty target to converge later.
func TerminalDeletionEvidenceReady(
	status GroupStatus,
	capacity CapacityObservation,
	traffic TrafficObservation,
	membership MembershipObservation,
) (bool, error) {
	if err := validateGroupStatus(status); err != nil {
		return false, fmt.Errorf("validate durable status: %w", err)
	}
	fresh, err := statusWithFreshObservations(status, capacity, traffic, membership)
	if err != nil {
		return false, err
	}

	// Both the newest desired level and the last acknowledged level must be terminally empty. A newer non-empty
	// desired target could have been accepted despite a transport error and converge after finalizer removal.
	if fresh.Capacity.Desired == nil || fresh.Capacity.Accepted == nil ||
		len(fresh.Capacity.Desired.Replicas) != 0 || len(fresh.Capacity.Accepted.Replicas) != 0 ||
		fresh.Traffic.Desired == nil || fresh.Traffic.Accepted == nil ||
		len(fresh.Traffic.Desired.Admitted) != 0 || len(fresh.Traffic.Accepted.Admitted) != 0 {
		return false, nil
	}
	if !terminalCapacityTargetConverged(*fresh.Capacity.Accepted, capacity) ||
		!trafficTargetConverged(*fresh.Traffic.Accepted, traffic) {
		return false, nil
	}
	if len(capacity.Allocations) != 0 || len(traffic.Admitted) != 0 || len(traffic.Draining) != 0 ||
		len(membership.CommittedTopology.Replicas) != 0 {
		return false, nil
	}
	if membership.Transition != nil &&
		(membership.Transition.Phase == MembershipTransitionPhasePending ||
			membership.Transition.Phase == MembershipTransitionPhaseUnknown) {
		return false, nil
	}
	return true, nil
}

// UninitializedDeletionEvidenceReady validates fresh external truth for a group whose durable journal was never
// created. No controller effect can precede that journal, so an authoritatively empty world is safe to delete.
func UninitializedDeletionEvidenceReady(
	capacity CapacityObservation,
	traffic TrafficObservation,
	membership MembershipObservation,
) (bool, error) {
	if err := validateCapacityObservation(capacity); err != nil {
		return false, fmt.Errorf("validate capacity observation: %w", err)
	}
	if err := validateTrafficObservation(traffic); err != nil {
		return false, fmt.Errorf("validate traffic observation: %w", err)
	}
	if membership.RequestedTransitionID != "" {
		return false, errors.New("uninitialized membership observation is scoped to an unexpected transition")
	}
	if err := validateMembershipObservation(membership, nil); err != nil {
		return false, fmt.Errorf("validate membership observation: %w", err)
	}
	return len(capacity.Allocations) == 0 &&
		len(traffic.Admitted) == 0 &&
		len(traffic.Draining) == 0 &&
		len(membership.CommittedTopology.Replicas) == 0, nil
}

func validateAcceptedTargets(status GroupStatus) error {
	// Accepted capacity must be an acknowledged version of the current desired-target lineage.
	if status.Capacity.Accepted != nil {
		if status.Capacity.Desired == nil {
			return errors.New("accepted capacity target has no desired target")
		}
		if status.Capacity.Accepted.ControlRevision > status.Capacity.Desired.ControlRevision {
			return errors.New("accepted capacity target is newer than the desired target")
		}
		if status.Capacity.Accepted.ControlRevision != status.Capacity.Observed.AppliedRevision {
			return errors.New("accepted capacity target does not match the adapter applied revision")
		}
		if status.Capacity.Accepted.ControlRevision == status.Capacity.Desired.ControlRevision &&
			!sameCapacityTargetIntent(*status.Capacity.Accepted, *status.Capacity.Desired) {
			return errors.New("accepted and desired capacity payloads conflict at the same revision")
		}
	}

	// Accepted traffic follows the same revision and payload invariants independently from capacity.
	if status.Traffic.Accepted != nil {
		if status.Traffic.Desired == nil {
			return errors.New("accepted traffic target has no desired target")
		}
		if status.Traffic.Accepted.ControlRevision > status.Traffic.Desired.ControlRevision {
			return errors.New("accepted traffic target is newer than the desired target")
		}
		if status.Traffic.Accepted.ControlRevision != status.Traffic.Observed.AppliedRevision {
			return errors.New("accepted traffic target does not match the adapter applied revision")
		}
		if status.Traffic.Accepted.ControlRevision == status.Traffic.Desired.ControlRevision &&
			!sameTrafficTargetIntent(*status.Traffic.Accepted, *status.Traffic.Desired) {
			return errors.New("accepted and desired traffic payloads conflict at the same revision")
		}
	}
	return nil
}

func validateFailure(failure *Failure) error {
	if failure == nil {
		return errors.New("failure must not be nil")
	}
	if failure.Classification != FailureClassificationRetryable &&
		failure.Classification != FailureClassificationTerminal {
		return fmt.Errorf("invalid failure classification %q", failure.Classification)
	}
	if failure.Reason == "" {
		return errors.New("failure reason must not be empty")
	}
	return nil
}

func validateRejection(rejection *Failure) error {
	if err := validateFailure(rejection); err != nil {
		return err
	}
	if rejection.Classification != FailureClassificationTerminal {
		return errors.New("definitive rejection must be terminal for the same request")
	}
	return nil
}

func validatePreflightResult(result PreflightResult) error {
	if (result.Evidence == nil) == (result.Rejection == nil) {
		return errors.New("preflight must return exactly one of evidence or rejection")
	}
	if result.Rejection != nil {
		return validateRejection(result.Rejection)
	}
	return nil
}

func validatePlanEvidence(evidence ValidationEvidence, digest string, plan ResolvedPlan) error {
	if evidence.PlanDigest != digest {
		return errors.New("validation evidence carries another plan digest")
	}
	if evidence.TargetDigest != "" {
		return errors.New("plan validation evidence must not carry a target digest")
	}
	if evidence.ProfileFingerprint != plan.ProfileFingerprint {
		return errors.New("validation evidence carries another profile fingerprint")
	}
	if evidence.CapabilityGeneration == "" {
		return errors.New("validation evidence lacks a capability generation")
	}
	return nil
}

func validateTargetEvidence(
	evidence ValidationEvidence,
	target MembershipTarget,
	planEvidence ValidationEvidence,
) error {
	if evidence.PlanDigest != planEvidence.PlanDigest ||
		evidence.ProfileFingerprint != planEvidence.ProfileFingerprint ||
		evidence.CapabilityGeneration != planEvidence.CapabilityGeneration {
		return errors.New("target validation changed its plan, profile, or capability generation")
	}
	if evidence.TargetDigest != target.TargetDigest {
		return errors.New("target validation evidence carries another target digest")
	}
	return nil
}

func validateMembershipTarget(controlRevision int64, target *MembershipTarget) error {
	if target == nil {
		return nil
	}
	if target.ControlRevision <= 0 || target.ControlRevision > controlRevision {
		return fmt.Errorf(
			"membership target revision %d is invalid for control revision %d",
			target.ControlRevision,
			controlRevision,
		)
	}
	if target.TransitionID == "" || target.TargetDigest == "" {
		return errors.New("membership target lacks a transition ID or canonical digest")
	}
	if err := validateTopology(target.BaseTopology); err != nil {
		return fmt.Errorf("validate membership target base topology: %w", err)
	}
	if target.Plan.ID == "" || target.Plan.ProfileFingerprint == "" {
		return errors.New("membership target has an incomplete resolved plan")
	}
	if err := validateResolvedChange(target.Plan.Change); err != nil {
		return err
	}
	planDigest, err := canonicalPlanDigest(target.Plan)
	if err != nil {
		return err
	}
	planEvidence := target.Validation
	planEvidence.TargetDigest = ""
	if err := validatePlanEvidence(planEvidence, planDigest, target.Plan); err != nil {
		return err
	}
	wantDigest, err := canonicalMembershipTargetDigest(*target)
	if err != nil {
		return err
	}
	if target.TargetDigest != wantDigest {
		return errors.New("membership target digest does not match its canonical payload")
	}
	if target.Validation.TargetDigest != target.TargetDigest {
		return errors.New("membership target validation does not bind its canonical digest")
	}
	return nil
}

func validateMembershipObservation(observation MembershipObservation, target *MembershipTarget) error {
	if err := validateTopology(observation.CommittedTopology); err != nil {
		return fmt.Errorf("validate authoritative committed topology: %w", err)
	}
	if observation.RequestedTransitionID == "" && observation.Transition != nil {
		return errors.New("topology-only membership observation carries a transition result")
	}
	transition := observation.Transition
	if transition == nil {
		return nil
	}
	if transition.TransitionID != observation.RequestedTransitionID {
		return errors.New("membership transition result does not match its observation scope")
	}
	if target != nil && observation.RequestedTransitionID == target.TransitionID &&
		(transition.ControlRevision != target.ControlRevision ||
			transition.TargetDigest != target.TargetDigest) {
		return errors.New("membership transition observation does not match the desired target")
	}

	switch transition.Phase {
	case MembershipTransitionPhasePending:
		if transition.ResultTopology != nil || transition.Failure != nil {
			return errors.New("pending membership transition carries a terminal result")
		}
	case MembershipTransitionPhaseCommitted:
		if transition.ResultTopology == nil || transition.Failure != nil {
			return errors.New("committed membership transition is not a closed topology result")
		}
		if err := validateTopology(*transition.ResultTopology); err != nil {
			return fmt.Errorf("validate membership result topology: %w", err)
		}
	case MembershipTransitionPhaseRejected:
		if transition.ResultTopology != nil || transition.Failure == nil {
			return errors.New("rejected membership transition is not a closed failure result")
		}
		if err := validateRejection(transition.Failure); err != nil {
			return err
		}
	case MembershipTransitionPhaseUnknown:
		if transition.ResultTopology != nil {
			return errors.New("unknown membership transition carries a result topology")
		}
		if transition.Failure != nil {
			return validateFailure(transition.Failure)
		}
	default:
		return fmt.Errorf("invalid membership transition phase %q", transition.Phase)
	}
	return nil
}

func validateMembershipObservationEvolution(previous, current MembershipObservation) error {
	if previous.RequestedTransitionID != current.RequestedTransitionID {
		return nil
	}
	before := previous.Transition
	after := current.Transition
	if before == nil {
		return nil
	}
	if after == nil {
		return errors.New("membership adapter forgot a previously observed transition")
	}
	if before.TransitionID != after.TransitionID ||
		before.ControlRevision != after.ControlRevision ||
		before.TargetDigest != after.TargetDigest {
		return errors.New("membership adapter changed transition correlation")
	}

	switch before.Phase {
	case MembershipTransitionPhasePending:
		if after.Phase != MembershipTransitionPhasePending &&
			after.Phase != MembershipTransitionPhaseCommitted &&
			after.Phase != MembershipTransitionPhaseRejected &&
			after.Phase != MembershipTransitionPhaseUnknown {
			return fmt.Errorf("invalid membership transition %s to %s", before.Phase, after.Phase)
		}
	case MembershipTransitionPhaseUnknown:
		if after.Phase != MembershipTransitionPhaseUnknown &&
			after.Phase != MembershipTransitionPhasePending &&
			after.Phase != MembershipTransitionPhaseCommitted &&
			after.Phase != MembershipTransitionPhaseRejected {
			return fmt.Errorf("invalid membership transition %s to %s", before.Phase, after.Phase)
		}
	case MembershipTransitionPhaseCommitted, MembershipTransitionPhaseRejected:
		if !sameMembershipTransitionObservation(*before, *after) {
			return errors.New("membership adapter changed an immutable terminal result")
		}
	default:
		return fmt.Errorf("invalid previous membership transition phase %q", before.Phase)
	}
	return nil
}

func validateTopologyHistory(history TopologyHistory) error {
	if history.CurrentGeneration <= 0 {
		return errors.New("current topology generation must be positive")
	}
	seen := make(map[int64]MembershipTopology, len(history.Snapshots))
	for _, snapshot := range history.Snapshots {
		if err := validateTopology(snapshot); err != nil {
			return err
		}
		if existing, duplicate := seen[snapshot.Generation]; duplicate {
			if !sameTopology(existing, snapshot) {
				return fmt.Errorf("topology generation %d has conflicting snapshots", snapshot.Generation)
			}
			return fmt.Errorf("topology generation %d appears more than once", snapshot.Generation)
		}
		seen[snapshot.Generation] = snapshot
	}
	if _, found := seen[history.CurrentGeneration]; !found {
		return fmt.Errorf("current topology generation %d has no snapshot", history.CurrentGeneration)
	}
	return nil
}

func validateCapacityTarget(controlRevision int64, target *CapacityTarget) error {
	if target == nil {
		return nil
	}
	if target.ControlRevision <= 0 || target.ControlRevision > controlRevision {
		return fmt.Errorf(
			"capacity target revision %d is invalid for control revision %d",
			target.ControlRevision,
			controlRevision,
		)
	}
	if target.TransitionID == "" {
		return errors.New("capacity target transition ID must not be empty")
	}
	if target.ProfileFingerprint == "" {
		return errors.New("capacity target profile fingerprint must not be empty")
	}
	if target.ProcessLifecycleOwner != ProcessLifecycleOwnerEngine &&
		target.ProcessLifecycleOwner != ProcessLifecycleOwnerOrchestrator {
		return fmt.Errorf("capacity target has invalid process lifecycle owner %q", target.ProcessLifecycleOwner)
	}

	replicaIDs := make(map[ReplicaID]struct{}, len(target.Replicas))
	slots := make(map[CapacitySlotID]struct{}, len(target.Replicas))
	for _, replica := range target.Replicas {
		if replica.ReplicaID == "" || replica.SlotID == "" {
			return errors.New("capacity target contains an incomplete stable identity")
		}
		if _, duplicate := replicaIDs[replica.ReplicaID]; duplicate {
			return fmt.Errorf("capacity target repeats replica %q", replica.ReplicaID)
		}
		if _, duplicate := slots[replica.SlotID]; duplicate {
			return fmt.Errorf("capacity target repeats slot %q", replica.SlotID)
		}
		if replica.Incarnation == nil {
			if target.ProcessLifecycleOwner == ProcessLifecycleOwnerOrchestrator {
				if err := validateCapacityBootstrap(replica.ReplicaID, replica.Bootstrap); err != nil {
					return err
				}
			} else if replica.Bootstrap != nil {
				return fmt.Errorf("engine-owned capacity target replica %q carries process bootstrap", replica.ReplicaID)
			}
		} else {
			if replica.Bootstrap != nil {
				return fmt.Errorf("capacity target replica %q has both incarnation and bootstrap mode", replica.ReplicaID)
			}
			if err := validateIncarnation(*replica.Incarnation); err != nil {
				return err
			}
			if replica.Incarnation.ReplicaID != replica.ReplicaID ||
				replica.Incarnation.SlotID != replica.SlotID {
				return fmt.Errorf("capacity target replica %q incarnation changes its stable identity", replica.ReplicaID)
			}
		}
		replicaIDs[replica.ReplicaID] = struct{}{}
		slots[replica.SlotID] = struct{}{}
	}
	if err := validateReleaseFences(target.ReleaseFences); err != nil {
		return err
	}
	for _, fence := range target.ReleaseFences {
		if fence.TransitionID != target.TransitionID {
			return fmt.Errorf("release fence for replica %q belongs to another transition", fence.ReplicaID)
		}
		if _, retained := replicaIDs[fence.ReplicaID]; retained {
			return fmt.Errorf("capacity target both retains and releases replica %q", fence.ReplicaID)
		}
	}
	return nil
}

func validateCapacityBootstrap(replicaID ReplicaID, bootstrap *CapacityBootstrap) error {
	if bootstrap == nil ||
		(bootstrap.Mode != BootstrapModeJoin && bootstrap.Mode != BootstrapModeRestoreFixedSlot) ||
		bootstrap.BaseTopologyGeneration <= 0 || len(bootstrap.NativeMembers) == 0 {
		return fmt.Errorf("capacity target replica %q has incomplete process bootstrap", replicaID)
	}
	seen := make(map[NativeMemberID]struct{}, len(bootstrap.NativeMembers))
	for _, nativeMember := range bootstrap.NativeMembers {
		if nativeMember == "" {
			return fmt.Errorf("capacity target replica %q has an empty bootstrap native member", replicaID)
		}
		if _, duplicate := seen[nativeMember]; duplicate {
			return fmt.Errorf("capacity target replica %q repeats bootstrap native member %q", replicaID, nativeMember)
		}
		seen[nativeMember] = struct{}{}
	}
	return nil
}

func validateTrafficTarget(controlRevision int64, target *TrafficTarget) error {
	if target == nil {
		return nil
	}
	if target.ControlRevision <= 0 || target.ControlRevision > controlRevision {
		return fmt.Errorf(
			"traffic target revision %d is invalid for control revision %d",
			target.ControlRevision,
			controlRevision,
		)
	}
	if target.TransitionID == "" || target.TopologyGeneration <= 0 {
		return errors.New("traffic target has no transition identity or topology generation")
	}
	if err := validateMembershipIdentitySet(target.Admitted); err != nil {
		return fmt.Errorf("validate desired admitted membership: %w", err)
	}
	draining := make([]ReplicaMembership, 0, len(target.Drain))
	seenDrains := make(map[ReplicaID]struct{}, len(target.Drain))
	for _, drained := range target.Drain {
		if drained.Mode != TrafficDrainModeGraceful && drained.Mode != TrafficDrainModeConfirmInactive {
			return fmt.Errorf("replica %q has invalid traffic drain mode %q", drained.Membership.ReplicaID, drained.Mode)
		}
		if _, duplicate := seenDrains[drained.Membership.ReplicaID]; duplicate {
			return fmt.Errorf("replica %q appears more than once in desired drain state", drained.Membership.ReplicaID)
		}
		seenDrains[drained.Membership.ReplicaID] = struct{}{}
		draining = append(draining, drained.Membership)
		if containsMembership(target.Admitted, drained.Membership) {
			return fmt.Errorf(
				"traffic target both admits and drains replica %q incarnation %q",
				drained.Membership.ReplicaID,
				drained.Membership.RuntimeIncarnation,
			)
		}
	}
	if err := validateMembershipIdentitySet(draining); err != nil {
		return fmt.Errorf("validate desired drain membership: %w", err)
	}
	return nil
}

func validateTransition(status GroupStatus) error {
	transition := status.Transition
	if transition.Spec.ID == "" || transition.Spec.Plan.ID == "" {
		return errors.New("transition or plan ID must not be empty")
	}
	if transition.Spec.ID != transitionID(transition.Spec.Plan.ID, transition.Spec.BaseTopologyGeneration) {
		return errors.New("transition ID does not match its plan and base topology")
	}
	base, found := status.Topologies.Snapshot(transition.Spec.BaseTopologyGeneration)
	if !found {
		return fmt.Errorf("base topology generation %d is absent", transition.Spec.BaseTopologyGeneration)
	}
	resolution, err := validateResolvedPlan(base, status.Registry, transition.Spec.Plan)
	if err != nil {
		return err
	}
	if err := validateTransitionPreflights(status); err != nil {
		return err
	}
	if err := validateTransitionCapacityEvidence(status); err != nil {
		return err
	}
	if err := validateTransitionMembership(status, resolution); err != nil {
		return err
	}
	if err := validateVerificationStatus(status); err != nil {
		return err
	}

	switch transition.Outcome {
	case TransitionOutcomeProgressing:
		if transition.Failure != nil {
			return errors.New("progressing transition must not carry a terminal failure")
		}
	case TransitionOutcomeReverting, TransitionOutcomeRolledBack, TransitionOutcomeBlocked:
		if transition.Failure == nil {
			return fmt.Errorf("%s transition must carry a failure", transition.Outcome)
		}
		if err := validateFailure(transition.Failure); err != nil {
			return err
		}
		if transition.Outcome == TransitionOutcomeRolledBack &&
			!membershipProvablyUncommitted(status) {
			return errors.New("rolled-back transition lacks a provably uncommitted membership state")
		}
		if transition.Outcome == TransitionOutcomeRolledBack {
			if err := validateTerminalTargetsAccepted(status); err != nil {
				return err
			}
		}
	case TransitionOutcomeCompleted:
		if transition.Failure != nil {
			return errors.New("completed transition must not carry a failure")
		}
		if !membershipCommitConverged(status) {
			return errors.New("completed transition has no committed membership")
		}
		if transition.Spec.Plan.VerificationRequirement == VerificationRequirementRequired &&
			transition.Verification.Phase != VerificationPhasePassed {
			return errors.New("completed transition has no required serving proof")
		}
		if err := validateTerminalTargetsAccepted(status); err != nil {
			return err
		}
	default:
		return fmt.Errorf("invalid transition outcome %q", transition.Outcome)
	}
	return nil
}

func validateTerminalTargetsAccepted(status GroupStatus) error {
	if status.Capacity.Desired != nil &&
		(status.Capacity.Accepted == nil ||
			status.Capacity.Accepted.ControlRevision != status.Capacity.Desired.ControlRevision ||
			!sameCapacityTargetIntent(*status.Capacity.Accepted, *status.Capacity.Desired)) {
		return errors.New("terminal transition has an unaccepted capacity target")
	}
	if status.Traffic.Desired != nil &&
		(status.Traffic.Accepted == nil ||
			status.Traffic.Accepted.ControlRevision != status.Traffic.Desired.ControlRevision ||
			!sameTrafficTargetIntent(*status.Traffic.Accepted, *status.Traffic.Desired)) {
		return errors.New("terminal transition has an unaccepted traffic target")
	}
	return nil
}

func validateTransitionCapacityEvidence(status GroupStatus) error {
	target := status.Capacity.Desired
	transition := status.Transition
	if target == nil || target.TransitionID != transition.Spec.ID {
		return nil
	}
	for _, replica := range target.Replicas {
		if replica.Bootstrap != nil &&
			replica.Bootstrap.BaseTopologyGeneration != transition.Spec.BaseTopologyGeneration {
			return fmt.Errorf("capacity bootstrap for replica %q refers to another base topology", replica.ReplicaID)
		}
		if status.Membership.Desired != nil &&
			membershipTargetMatchesTransition(*status.Membership.Desired, *transition) &&
			replica.Bootstrap != nil {
			return fmt.Errorf(
				"capacity bootstrap for replica %q remains after membership target creation",
				replica.ReplicaID,
			)
		}
	}
	for _, fence := range target.ReleaseFences {
		if _, found := status.Topologies.Snapshot(fence.AuthorizingTopologyGeneration); !found {
			return fmt.Errorf(
				"release fence for replica %q refers to an unknown authorizing topology",
				fence.ReplicaID,
			)
		}
	}
	return nil
}

func validateTransitionPreflights(status GroupStatus) error {
	transition := status.Transition
	planDigest, err := canonicalPlanDigest(transition.Spec.Plan)
	if err != nil {
		return err
	}
	if err := validatePlanPreflight(*transition, planDigest); err != nil {
		return err
	}
	return validateTargetPreflight(status)
}

func validatePlanPreflight(transition TransitionStatus, planDigest string) error {
	plan := transition.PlanPreflight
	if plan.SubjectDigest == "" {
		if plan.TransitionID != "" || plan.ControlRevision != 0 ||
			plan.Evidence != nil || plan.Rejection != nil {
			return errors.New("empty plan preflight carries an outcome")
		}
		if transition.TargetPreflight.SubjectDigest != "" {
			return errors.New("target preflight exists before plan validation")
		}
		return nil
	}
	if plan.SubjectDigest != planDigest {
		return errors.New("plan preflight digest does not match the durable plan")
	}
	if plan.TransitionID != transition.Spec.ID || plan.ControlRevision != 0 {
		return errors.New("plan preflight correlation does not match the transition")
	}
	if err := validatePreflightResult(PreflightResult{
		Evidence: plan.Evidence, Rejection: plan.Rejection,
	}); err != nil {
		return fmt.Errorf("validate durable plan preflight: %w", err)
	}
	if plan.Evidence != nil {
		if err := validatePlanEvidence(*plan.Evidence, planDigest, transition.Spec.Plan); err != nil {
			return err
		}
	} else {
		if transition.Outcome != TransitionOutcomeBlocked {
			return errors.New("rejected plan preflight did not block the transition")
		}
		if !sameFailure(transition.Failure, plan.Rejection) {
			return errors.New("transition failure differs from its plan rejection")
		}
	}
	return nil
}

func validateTargetPreflight(status GroupStatus) error {
	transition := status.Transition
	target := transition.TargetPreflight
	if target.SubjectDigest == "" {
		if target.TransitionID != "" || target.ControlRevision != 0 ||
			target.Evidence != nil || target.Rejection != nil {
			return errors.New("empty target preflight carries an outcome")
		}
		return nil
	}
	if err := validatePreflightResult(PreflightResult{
		Evidence: target.Evidence, Rejection: target.Rejection,
	}); err != nil {
		return fmt.Errorf("validate durable target preflight: %w", err)
	}
	if target.TransitionID != membershipTransitionID(
		transition.Spec.Plan.ID,
		transition.Spec.BaseTopologyGeneration,
	) || target.ControlRevision <= 0 {
		return errors.New("target preflight correlation does not match the transition")
	}
	if target.Evidence == nil {
		if transition.Outcome != TransitionOutcomeReverting &&
			transition.Outcome != TransitionOutcomeRolledBack &&
			transition.Outcome != TransitionOutcomeBlocked {
			return errors.New("rejected target preflight did not stop membership progress")
		}
		if !sameFailure(transition.Failure, target.Rejection) {
			return errors.New("transition failure differs from its target rejection")
		}
		return nil
	}
	if status.Membership.Desired == nil ||
		status.Membership.Desired.TransitionID != membershipTransitionID(
			transition.Spec.Plan.ID,
			transition.Spec.BaseTopologyGeneration,
		) {
		return errors.New("successful target preflight has no matching desired membership")
	}
	if target.SubjectDigest != status.Membership.Desired.TargetDigest {
		return errors.New("target preflight digest differs from desired membership")
	}
	if target.ControlRevision != status.Membership.Desired.ControlRevision {
		return errors.New("target preflight revision differs from desired membership")
	}
	if transition.PlanPreflight.Evidence == nil {
		return errors.New("target preflight lacks successful plan validation")
	}
	if err := validateTargetEvidence(
		*target.Evidence,
		*status.Membership.Desired,
		*transition.PlanPreflight.Evidence,
	); err != nil {
		return err
	}
	return nil
}

func membershipCommitConverged(status GroupStatus) bool {
	if !membershipCommittedForTransition(status) || status.Membership.Observed.Transition.ResultTopology == nil {
		return false
	}
	result := *status.Membership.Observed.Transition.ResultTopology
	return sameTopology(status.Membership.Observed.CommittedTopology, result) &&
		sameTopologyWithCurrent(status.Topologies, result)
}

func validateTransitionMembership(status GroupStatus, resolution planResolution) error {
	target := status.Membership.Desired
	if target == nil || !membershipTargetMatchesTransition(*target, *status.Transition) {
		return nil
	}
	base, _ := status.Topologies.Snapshot(status.Transition.Spec.BaseTopologyGeneration)
	if !sameTopology(target.BaseTopology, base) {
		return errors.New("membership target base topology differs from the transition base")
	}
	observed := status.Membership.Observed.Transition
	observationMatchesTarget := status.Membership.Observed.RequestedTransitionID == target.TransitionID
	requireCurrentRegistry := !observationMatchesTarget ||
		observed == nil || observed.Phase != MembershipTransitionPhaseRejected
	if err := validateJoiningReplicas(
		status.Registry,
		resolution,
		target.Joining,
		requireCurrentRegistry,
	); err != nil {
		return err
	}
	if !observationMatchesTarget {
		return nil
	}
	if observed == nil || observed.Phase != MembershipTransitionPhaseCommitted {
		return nil
	}
	if observed.ResultTopology == nil {
		return errors.New("committed membership transition lacks its immutable result")
	}
	if err := validateCommittedTopology(base, resolution, target.Joining, *observed.ResultTopology); err != nil {
		return fmt.Errorf("validate durable committed topology: %w", err)
	}
	return nil
}

func membershipProvablyUncommitted(status GroupStatus) bool {
	if status.Transition == nil || status.Membership.Desired == nil ||
		!membershipTargetMatchesTransition(*status.Membership.Desired, *status.Transition) {
		return true
	}
	observed := status.Membership.Observed.Transition
	return observed != nil && observed.Phase == MembershipTransitionPhaseRejected
}

func validateJoiningReplicas(
	registry ReplicaRegistry,
	resolution planResolution,
	joining []JoiningReplica,
	requireCurrentRegistry bool,
) error {
	if len(joining) != len(resolution.joiningTargets) {
		return errors.New("membership joining replica count does not match the plan")
	}
	targets := make(map[ReplicaID]struct{}, len(resolution.joiningTargets))
	for _, target := range resolution.joiningTargets {
		targets[target.ReplicaID] = struct{}{}
	}
	seen := make(map[ReplicaID]struct{}, len(joining))
	for _, replica := range joining {
		if replica.ReplicaID == "" || replica.RuntimeIncarnation == "" {
			return errors.New("membership joining replica has an incomplete identity")
		}
		if _, expected := targets[replica.ReplicaID]; !expected {
			return fmt.Errorf("membership joining replica %q is absent from the plan", replica.ReplicaID)
		}
		if _, duplicate := seen[replica.ReplicaID]; duplicate {
			return fmt.Errorf("membership joining replica %q appears more than once", replica.ReplicaID)
		}
		if requireCurrentRegistry {
			record, found := registry.Find(replica.ReplicaID)
			if !found || record.Current == nil || record.Current.RuntimeIncarnation != replica.RuntimeIncarnation {
				return fmt.Errorf("membership joining replica %q differs from the canonical registry", replica.ReplicaID)
			}
		}
		seen[replica.ReplicaID] = struct{}{}
	}
	return nil
}

func validateVerificationStatus(status GroupStatus) error {
	verification := status.Transition.Verification
	switch verification.Phase {
	case "":
		if verification.Proof != nil || verification.Failure != nil {
			return errors.New("empty verification status carries a result")
		}
	case VerificationPhasePending:
		if verification.Proof != nil || verification.Failure != nil {
			return errors.New("pending verification carries a result")
		}
	case VerificationPhasePassed:
		if verification.Proof == nil || verification.Failure != nil {
			return errors.New("passed verification lacks proof or carries failure")
		}
		topology, found := status.Topologies.Snapshot(verification.Proof.TopologyGeneration)
		if !found || verification.Proof.RuntimeDigest != TopologyRuntimeDigest(topology) {
			return errors.New("serving proof does not match retained topology history")
		}
	case VerificationPhaseFailed:
		if verification.Proof != nil || verification.Failure == nil {
			return errors.New("failed verification lacks failure or carries proof")
		}
		if err := validateFailure(verification.Failure); err != nil {
			return err
		}
	default:
		return fmt.Errorf("invalid verification phase %q", verification.Phase)
	}
	return nil
}

func validatePlanCanStart(registry ReplicaRegistry, resolution planResolution) error {
	for _, target := range resolution.joiningTargets {
		record, found := registry.Find(target.ReplicaID)
		switch resolution.kind {
		case PlanKindGrow:
			if found {
				return fmt.Errorf("fresh growth replica %q already has a canonical record", target.ReplicaID)
			}
		case PlanKindRestore:
			if !found {
				return fmt.Errorf("restored replica %q has no canonical record", target.ReplicaID)
			}
			if record.Current != nil {
				return fmt.Errorf("restored replica %q still has current physical capacity", target.ReplicaID)
			}
		}
	}
	return nil
}

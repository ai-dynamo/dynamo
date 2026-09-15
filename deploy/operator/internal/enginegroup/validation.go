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
	if err := validateTrafficTarget(status.ControlRevision, status.Traffic.Desired); err != nil {
		return err
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

func validateMembershipObservation(observation MembershipOperationObservation) error {
	switch observation.Phase {
	case MembershipBackendPhaseAbsent, MembershipBackendPhaseRunning:
		if observation.CommittedTopology != nil || observation.Failure != nil {
			return fmt.Errorf("%s membership observation carries a result", observation.Phase)
		}
	case MembershipBackendPhaseCommitted:
		if observation.CommittedTopology == nil || observation.Failure != nil {
			return errors.New("committed membership observation is not a closed topology result")
		}
	case MembershipBackendPhaseRejected:
		if observation.CommittedTopology != nil || observation.Failure == nil {
			return errors.New("rejected membership observation is not a closed failure result")
		}
		if err := validateRejection(observation.Failure); err != nil {
			return err
		}
	case MembershipBackendPhaseUnknown:
		if observation.CommittedTopology != nil {
			return errors.New("unknown membership observation carries a committed topology")
		}
		if observation.Failure != nil {
			return validateFailure(observation.Failure)
		}
	default:
		return fmt.Errorf("invalid membership backend phase %q", observation.Phase)
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
			if replica.Bootstrap != BootstrapModeJoin && replica.Bootstrap != BootstrapModeRestoreFixedSlot {
				return fmt.Errorf("capacity target replica %q has no valid bootstrap mode", replica.ReplicaID)
			}
		} else {
			if replica.Bootstrap != "" {
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
		if _, retained := replicaIDs[fence.ReplicaID]; retained {
			return fmt.Errorf("capacity target both retains and releases replica %q", fence.ReplicaID)
		}
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
	if err := validateMembershipIdentitySet(target.Drain); err != nil {
		return fmt.Errorf("validate desired drain membership: %w", err)
	}
	for _, drained := range target.Drain {
		if containsMembership(target.Admitted, drained) {
			return fmt.Errorf(
				"traffic target both admits and drains replica %q incarnation %q",
				drained.ReplicaID,
				drained.RuntimeIncarnation,
			)
		}
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
	if transition.Membership.ID != membershipOperationID(
		transition.Spec.Plan.ID,
		transition.Spec.BaseTopologyGeneration,
	) {
		return errors.New("membership operation ID does not match the transition")
	}
	if err := validateMembershipStatus(status, resolution); err != nil {
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
			transition.Membership.Phase != MembershipOperationPhaseRejected &&
			transition.Membership.Phase != MembershipOperationPhaseNotStarted {
			return errors.New("rolled-back transition lacks a provably uncommitted membership state")
		}
	case TransitionOutcomeCompleted:
		if transition.Failure != nil {
			return errors.New("completed transition must not carry a failure")
		}
		if transition.Membership.Phase != MembershipOperationPhaseCommitted {
			return errors.New("completed transition has no committed membership")
		}
		if transition.Spec.Plan.VerificationRequirement == VerificationRequirementRequired &&
			transition.Verification.Phase != VerificationPhasePassed {
			return errors.New("completed transition has no required serving proof")
		}
	default:
		return fmt.Errorf("invalid transition outcome %q", transition.Outcome)
	}
	return nil
}

func validateMembershipStatus(status GroupStatus, resolution planResolution) error {
	membership := status.Transition.Membership
	switch membership.Phase {
	case MembershipOperationPhaseNotStarted:
		if len(membership.JoiningReplicas) != 0 || membership.CommittedTopologyGeneration != 0 ||
			membership.Failure != nil {
			return errors.New("not-started membership contains operation results")
		}
	case MembershipOperationPhasePrepared, MembershipOperationPhaseRunning:
		if membership.CommittedTopologyGeneration != 0 || membership.Failure != nil {
			return errors.New("uncommitted membership contains a commit or failure result")
		}
		if err := validateJoiningReplicas(status.Registry, resolution, membership.JoiningReplicas, true); err != nil {
			return err
		}
	case MembershipOperationPhaseCommitted:
		if membership.CommittedTopologyGeneration <= 0 || membership.Failure != nil {
			return errors.New("committed membership lacks a generation or carries failure")
		}
		committed, found := status.Topologies.Snapshot(membership.CommittedTopologyGeneration)
		if !found {
			return errors.New("committed membership topology is absent")
		}
		if err := validateJoiningReplicas(status.Registry, resolution, membership.JoiningReplicas, true); err != nil {
			return err
		}
		base, _ := status.Topologies.Snapshot(status.Transition.Spec.BaseTopologyGeneration)
		if err := validateCommittedTopology(base, resolution, membership.JoiningReplicas, committed); err != nil {
			return fmt.Errorf("validate durable committed topology: %w", err)
		}
	case MembershipOperationPhaseRejected, MembershipOperationPhaseUnknown:
		if membership.CommittedTopologyGeneration != 0 || membership.Failure == nil {
			return errors.New("failed membership status lacks a failure or carries a commit")
		}
		if err := validateFailure(membership.Failure); err != nil {
			return err
		}
		if err := validateJoiningReplicas(status.Registry, resolution, membership.JoiningReplicas, false); err != nil {
			return err
		}
	default:
		return fmt.Errorf("invalid membership operation phase %q", membership.Phase)
	}
	return nil
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
		if !found || verification.Proof.RuntimeDigest != topologyRuntimeDigest(topology) {
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

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
	"slices"
)

type planResolution struct {
	kind               PlanKind
	targetReplicaIDs   []ReplicaID
	joiningTargets     []ReplicaTarget
	retiringReplicaIDs []ReplicaID
	restoredMembership []ReplicaNativeMembership
	remappedMembership []ReplicaNativeMembership
}

// NewGroupStatus validates and captures one already-running Engine Group as the initial durable status.
func NewGroupStatus(
	topology MembershipTopology,
	capacity CapacityObservation,
	traffic TrafficObservation,
) (GroupStatus, error) {
	if err := validateTopology(topology); err != nil {
		return GroupStatus{}, fmt.Errorf("validate initial topology: %w", err)
	}
	if err := validateCapacityObservation(capacity); err != nil {
		return GroupStatus{}, fmt.Errorf("validate initial capacity: %w", err)
	}
	if err := validateTrafficObservation(traffic); err != nil {
		return GroupStatus{}, fmt.Errorf("validate initial traffic: %w", err)
	}

	// Every committed member receives one canonical registry record backed by the same observed incarnation.
	registry := ReplicaRegistry{Replicas: make([]ReplicaRecord, 0, len(topology.Replicas))}
	for _, membership := range topology.Replicas {
		allocation, found := allocationByID(capacity, membership.ReplicaID)
		if !found {
			return GroupStatus{}, fmt.Errorf(
				"committed replica %q has no physical allocation",
				membership.ReplicaID,
			)
		}
		if !membershipMatchesIncarnation(membership, allocation.Incarnation) {
			return GroupStatus{}, fmt.Errorf(
				"committed replica %q does not match its physical allocation",
				membership.ReplicaID,
			)
		}
		current := cloneReplicaIncarnation(allocation.Incarnation)
		registry.Replicas = append(registry.Replicas, ReplicaRecord{
			ReplicaID: membership.ReplicaID,
			SlotID:    allocation.Incarnation.SlotID,
			Current:   &current,
		})
	}
	if err := validateRegistry(registry); err != nil {
		return GroupStatus{}, fmt.Errorf("validate initial replica registry: %w", err)
	}

	// Initial status owns one immutable current topology and starts after either adapter's observed control revision.
	return GroupStatus{
		ControlRevision: max(capacity.AppliedRevision, traffic.AppliedRevision),
		Registry:        registry,
		Topologies: TopologyHistory{
			CurrentGeneration: topology.Generation,
			Snapshots:         []MembershipTopology{cloneTopology(topology)},
		},
		Capacity: CapacityStatus{Observed: cloneCapacityObservation(capacity)},
		Traffic:  TrafficStatus{Observed: cloneTrafficObservation(traffic)},
		Membership: MembershipStatus{Observed: MembershipObservation{
			CommittedTopology: cloneTopology(topology),
		}},
	}, nil
}

func validateResolvedPlan(
	base MembershipTopology,
	registry ReplicaRegistry,
	plan ResolvedPlan,
) (planResolution, error) {
	if plan.ID == "" {
		return planResolution{}, errors.New("plan ID must not be empty")
	}
	if plan.ProfileFingerprint == "" {
		return planResolution{}, errors.New("profile fingerprint must not be empty")
	}
	if err := validateResolvedChange(plan.Change); err != nil {
		return planResolution{}, err
	}
	if plan.ProcessLifecycleOwner != ProcessLifecycleOwnerEngine &&
		plan.ProcessLifecycleOwner != ProcessLifecycleOwnerOrchestrator {
		return planResolution{}, fmt.Errorf("invalid process lifecycle owner %q", plan.ProcessLifecycleOwner)
	}
	if plan.TrafficRequirement != TrafficRequirementKeepServing &&
		plan.TrafficRequirement != TrafficRequirementQuiesceGroup {
		return planResolution{}, fmt.Errorf("invalid traffic requirement %q", plan.TrafficRequirement)
	}
	if plan.VerificationRequirement != VerificationRequirementNone &&
		plan.VerificationRequirement != VerificationRequirementRequired {
		return planResolution{}, fmt.Errorf("invalid verification requirement %q", plan.VerificationRequirement)
	}
	if err := validateTopology(base); err != nil {
		return planResolution{}, fmt.Errorf("validate base topology: %w", err)
	}
	if err := validateRegistry(registry); err != nil {
		return planResolution{}, fmt.Errorf("validate replica registry: %w", err)
	}

	var resolution planResolution
	var err error
	switch plan.Change.Kind {
	case PlanKindGrow:
		resolution, err = resolveGrowth(base, registry, plan.Change.Grow)
	case PlanKindRetire:
		resolution, err = resolveRetirement(base, plan.Change.Retire)
	case PlanKindReduceToSurvivors:
		resolution, err = resolveSurvivorReduction(base, plan.Change.ReduceToSurvivors)
	case PlanKindRestore:
		resolution, err = resolveRestoration(base, registry, plan.Change.Restore)
	case PlanKindRemap:
		resolution, err = resolveRemap(base, registry, plan.Change.Remap)
	default:
		return planResolution{}, fmt.Errorf("unsupported membership change kind %q", plan.Change.Kind)
	}
	if err != nil {
		return planResolution{}, err
	}
	if err := validateJoiningLifecycle(base, plan.ProcessLifecycleOwner, resolution); err != nil {
		return planResolution{}, err
	}

	return resolution, nil
}

func validateResolvedChange(change ResolvedChange) error {
	variants := 0
	for _, present := range []bool{
		change.Grow != nil,
		change.Retire != nil,
		change.ReduceToSurvivors != nil,
		change.Restore != nil,
		change.Remap != nil,
	} {
		if present {
			variants++
		}
	}
	if variants != 1 {
		return fmt.Errorf("membership change must contain exactly one variant, found %d", variants)
	}

	var matches bool
	switch change.Kind {
	case PlanKindGrow:
		matches = change.Grow != nil
	case PlanKindRetire:
		matches = change.Retire != nil
	case PlanKindReduceToSurvivors:
		matches = change.ReduceToSurvivors != nil
	case PlanKindRestore:
		matches = change.Restore != nil
	case PlanKindRemap:
		matches = change.Remap != nil
	default:
		return fmt.Errorf("unsupported membership change kind %q", change.Kind)
	}
	if !matches {
		return fmt.Errorf("membership change kind %q does not match its populated variant", change.Kind)
	}
	return nil
}

func resolveGrowth(
	base MembershipTopology,
	registry ReplicaRegistry,
	change *GrowChange,
) (planResolution, error) {
	if len(change.Replicas) == 0 {
		return planResolution{}, errors.New("growth must add at least one replica")
	}

	// Growth adds only fresh logical and slot identities; reusing an excluded identity is restoration.
	targetIDs := topologyReplicaIDs(base)
	seenReplicas := replicaIDSet(targetIDs)
	seenSlots := registrySlotSet(registry)
	for _, target := range change.Replicas {
		if err := validateReplicaTarget(target, BootstrapModeJoin); err != nil {
			return planResolution{}, err
		}
		if _, found := seenReplicas[target.ReplicaID]; found {
			return planResolution{}, fmt.Errorf("growth replica %q already exists", target.ReplicaID)
		}
		if record, found := registry.Find(target.ReplicaID); found {
			if record.SlotID != target.SlotID || len(record.History) > 0 {
				return planResolution{}, fmt.Errorf(
					"growth replica %q has historical identity; use restoration",
					target.ReplicaID,
				)
			}
		}
		if existingReplica, found := seenSlots[target.SlotID]; found && existingReplica != target.ReplicaID {
			return planResolution{}, fmt.Errorf(
				"growth slot %q is already bound to replica %q",
				target.SlotID,
				existingReplica,
			)
		}
		seenReplicas[target.ReplicaID] = struct{}{}
		seenSlots[target.SlotID] = target.ReplicaID
		targetIDs = append(targetIDs, target.ReplicaID)
	}

	return planResolution{
		kind:             PlanKindGrow,
		targetReplicaIDs: normalizeReplicaIDs(targetIDs),
		joiningTargets:   slices.Clone(change.Replicas),
	}, nil
}

func resolveRetirement(base MembershipTopology, change *RetireChange) (planResolution, error) {
	if len(change.Replicas) == 0 {
		return planResolution{}, errors.New("retirement must select at least one replica")
	}

	// The exact selected set is removed; every other logical identity remains.
	baseIDs := replicaIDSet(topologyReplicaIDs(base))
	retiring := make(map[ReplicaID]struct{}, len(change.Replicas))
	for _, replicaID := range change.Replicas {
		if replicaID == "" {
			return planResolution{}, errors.New("retiring replica ID must not be empty")
		}
		if _, found := baseIDs[replicaID]; !found {
			return planResolution{}, fmt.Errorf("retiring replica %q is absent from the base topology", replicaID)
		}
		if _, duplicate := retiring[replicaID]; duplicate {
			return planResolution{}, fmt.Errorf("retiring replica %q is duplicated", replicaID)
		}
		retiring[replicaID] = struct{}{}
	}

	targetIDs := make([]ReplicaID, 0, len(base.Replicas)-len(retiring))
	for _, replicaID := range topologyReplicaIDs(base) {
		if _, removed := retiring[replicaID]; !removed {
			targetIDs = append(targetIDs, replicaID)
		}
	}

	return planResolution{
		kind:               PlanKindRetire,
		targetReplicaIDs:   normalizeReplicaIDs(targetIDs),
		retiringReplicaIDs: normalizeReplicaIDs(change.Replicas),
	}, nil
}

func resolveSurvivorReduction(
	base MembershipTopology,
	change *ReduceToSurvivorsChange,
) (planResolution, error) {
	baseIDs := replicaIDSet(topologyReplicaIDs(base))
	survivors := replicaIDSet(change.Survivors)
	if len(survivors) != len(change.Survivors) {
		return planResolution{}, errors.New("survivor set contains duplicate replica IDs")
	}
	for replicaID := range survivors {
		if replicaID == "" {
			return planResolution{}, errors.New("survivor replica ID must not be empty")
		}
		if _, found := baseIDs[replicaID]; !found {
			return planResolution{}, fmt.Errorf("survivor %q is absent from the base topology", replicaID)
		}
	}
	if len(survivors) == len(baseIDs) {
		return planResolution{}, errors.New("survivor reduction must remove at least one replica")
	}

	retiring := make([]ReplicaID, 0, len(baseIDs)-len(survivors))
	for replicaID := range baseIDs {
		if _, retained := survivors[replicaID]; !retained {
			retiring = append(retiring, replicaID)
		}
	}

	return planResolution{
		kind:               PlanKindReduceToSurvivors,
		targetReplicaIDs:   normalizeReplicaIDs(change.Survivors),
		retiringReplicaIDs: normalizeReplicaIDs(retiring),
	}, nil
}

func resolveRestoration(
	base MembershipTopology,
	registry ReplicaRegistry,
	change *RestoreChange,
) (planResolution, error) {
	if len(change.Replicas) == 0 {
		return planResolution{}, errors.New("restoration must restore at least one replica")
	}

	// Restoration reuses excluded canonical IDs and slots with an exact native-member mapping.
	targetIDs := topologyReplicaIDs(base)
	seen := replicaIDSet(targetIDs)
	joining := make([]ReplicaTarget, 0, len(change.Replicas))
	restored := make([]ReplicaNativeMembership, 0, len(change.Replicas))
	for _, target := range change.Replicas {
		if err := validateReplicaTarget(target.ReplicaTarget, BootstrapModeRestoreFixedSlot); err != nil {
			return planResolution{}, err
		}
		if len(target.NativeMembers) == 0 {
			return planResolution{}, fmt.Errorf("restored replica %q has no native members", target.ReplicaID)
		}
		if _, active := seen[target.ReplicaID]; active {
			return planResolution{}, fmt.Errorf("restored replica %q is already active", target.ReplicaID)
		}
		record, found := registry.Find(target.ReplicaID)
		if !found {
			return planResolution{}, fmt.Errorf("restored replica %q has no canonical record", target.ReplicaID)
		}
		if record.SlotID != target.SlotID {
			return planResolution{}, fmt.Errorf(
				"restored replica %q requests slot %q instead of canonical slot %q",
				target.ReplicaID,
				target.SlotID,
				record.SlotID,
			)
		}
		if len(record.History) == 0 {
			return planResolution{}, fmt.Errorf("restored replica %q has no excluded membership history", target.ReplicaID)
		}
		latest := record.History[len(record.History)-1]
		if !slices.Equal(
			normalizeNativeMembers(latest.NativeMembers),
			normalizeNativeMembers(target.NativeMembers),
		) {
			return planResolution{}, fmt.Errorf(
				"restored replica %q native membership does not match its latest excluded incarnation",
				target.ReplicaID,
			)
		}
		seen[target.ReplicaID] = struct{}{}
		targetIDs = append(targetIDs, target.ReplicaID)
		joining = append(joining, target.ReplicaTarget)
		restored = append(restored, ReplicaNativeMembership{
			ReplicaID:     target.ReplicaID,
			SlotID:        target.SlotID,
			NativeMembers: normalizeNativeMembers(target.NativeMembers),
		})
	}

	return planResolution{
		kind:               PlanKindRestore,
		targetReplicaIDs:   normalizeReplicaIDs(targetIDs),
		joiningTargets:     joining,
		restoredMembership: restored,
	}, nil
}

func resolveRemap(
	base MembershipTopology,
	registry ReplicaRegistry,
	change *RemapChange,
) (planResolution, error) {
	if len(change.Membership) != len(base.Replicas) {
		return planResolution{}, fmt.Errorf(
			"remap contains %d replicas for base cardinality %d",
			len(change.Membership),
			len(base.Replicas),
		)
	}

	baseIDs := replicaIDSet(topologyReplicaIDs(base))
	seen := make(map[ReplicaID]struct{}, len(change.Membership))
	remapped := make([]ReplicaNativeMembership, 0, len(change.Membership))
	for _, membership := range change.Membership {
		if _, found := baseIDs[membership.ReplicaID]; !found {
			return planResolution{}, fmt.Errorf("remapped replica %q is absent from base topology", membership.ReplicaID)
		}
		if _, duplicate := seen[membership.ReplicaID]; duplicate {
			return planResolution{}, fmt.Errorf("remapped replica %q is duplicated", membership.ReplicaID)
		}
		if len(membership.NativeMembers) == 0 {
			return planResolution{}, fmt.Errorf("remapped replica %q has no native members", membership.ReplicaID)
		}
		record, _ := registry.Find(membership.ReplicaID)
		if membership.SlotID != record.SlotID {
			return planResolution{}, fmt.Errorf(
				"remapped replica %q requests slot %q instead of stable slot %q",
				membership.ReplicaID,
				membership.SlotID,
				record.SlotID,
			)
		}
		seen[membership.ReplicaID] = struct{}{}
		membership.NativeMembers = normalizeNativeMembers(membership.NativeMembers)
		remapped = append(remapped, membership)
	}

	return planResolution{
		kind:               PlanKindRemap,
		targetReplicaIDs:   normalizeReplicaIDs(topologyReplicaIDs(base)),
		remappedMembership: remapped,
	}, nil
}

func validateReplicaTarget(target ReplicaTarget, expectedBootstrap BootstrapMode) error {
	if target.ReplicaID == "" {
		return errors.New("replica target ID must not be empty")
	}
	if target.SlotID == "" {
		return fmt.Errorf("replica %q target slot must not be empty", target.ReplicaID)
	}
	if target.Bootstrap != expectedBootstrap {
		return fmt.Errorf(
			"replica %q bootstrap %q does not match required mode %q",
			target.ReplicaID,
			target.Bootstrap,
			expectedBootstrap,
		)
	}
	seen := make(map[NativeMemberID]struct{}, len(target.NativeMembers))
	for _, nativeMember := range target.NativeMembers {
		if nativeMember == "" {
			return fmt.Errorf("replica %q has an empty planned native member", target.ReplicaID)
		}
		if _, duplicate := seen[nativeMember]; duplicate {
			return fmt.Errorf("replica %q repeats planned native member %q", target.ReplicaID, nativeMember)
		}
		seen[nativeMember] = struct{}{}
	}
	return nil
}

func validateJoiningLifecycle(
	base MembershipTopology,
	owner ProcessLifecycleOwner,
	resolution planResolution,
) error {
	if len(resolution.joiningTargets) == 0 {
		return nil
	}
	planned := nativeMembershipByID(resolution.restoredMembership)
	for _, target := range resolution.joiningTargets {
		if len(target.NativeMembers) > 0 {
			planned[target.ReplicaID] = ReplicaNativeMembership{
				ReplicaID: target.ReplicaID, SlotID: target.SlotID, NativeMembers: target.NativeMembers,
			}
		}
		if owner == ProcessLifecycleOwnerOrchestrator && len(planned[target.ReplicaID].NativeMembers) == 0 {
			return fmt.Errorf(
				"orchestrator-owned joining replica %q has no resolved native membership",
				target.ReplicaID,
			)
		}
	}

	seen := make(map[NativeMemberID]ReplicaID)
	for _, membership := range base.Replicas {
		for _, nativeMember := range membership.NativeMembers {
			seen[nativeMember] = membership.ReplicaID
		}
	}
	for _, membership := range planned {
		for _, nativeMember := range membership.NativeMembers {
			if replicaID, duplicate := seen[nativeMember]; duplicate {
				return fmt.Errorf(
					"planned native member %q belongs to both replica %q and joining replica %q",
					nativeMember,
					replicaID,
					membership.ReplicaID,
				)
			}
			seen[nativeMember] = membership.ReplicaID
		}
	}
	return nil
}

func topologyReplicaIDs(topology MembershipTopology) []ReplicaID {
	replicaIDs := make([]ReplicaID, 0, len(topology.Replicas))
	for _, membership := range topology.Replicas {
		replicaIDs = append(replicaIDs, membership.ReplicaID)
	}
	return normalizeReplicaIDs(replicaIDs)
}

func registrySlotSet(registry ReplicaRegistry) map[CapacitySlotID]ReplicaID {
	slots := make(map[CapacitySlotID]ReplicaID, len(registry.Replicas))
	for _, record := range registry.Replicas {
		slots[record.SlotID] = record.ReplicaID
	}
	return slots
}

func replicaIDSet(values []ReplicaID) map[ReplicaID]struct{} {
	set := make(map[ReplicaID]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return set
}

func validateTopology(topology MembershipTopology) error {
	if topology.Generation <= 0 {
		return errors.New("topology generation must be positive")
	}

	replicaIDs := make(map[ReplicaID]struct{}, len(topology.Replicas))
	runtimes := make(map[RuntimeIncarnationID]struct{}, len(topology.Replicas))
	nativeMembers := make(map[NativeMemberID]struct{})
	for _, membership := range topology.Replicas {
		replicaID := membership.ReplicaID
		runtimeIncarnation := membership.RuntimeIncarnation
		if replicaID == "" || runtimeIncarnation == "" {
			return errors.New("topology contains an incomplete replica or runtime identity")
		}
		if len(membership.NativeMembers) == 0 {
			return fmt.Errorf("replica %q has no native members", replicaID)
		}
		if _, duplicate := replicaIDs[replicaID]; duplicate {
			return fmt.Errorf("replica %q appears more than once", replicaID)
		}
		if _, duplicate := runtimes[runtimeIncarnation]; duplicate {
			return fmt.Errorf("runtime incarnation %q appears more than once", runtimeIncarnation)
		}
		for _, nativeMember := range membership.NativeMembers {
			if nativeMember == "" {
				return fmt.Errorf("replica %q has an empty native member ID", replicaID)
			}
			if _, duplicate := nativeMembers[nativeMember]; duplicate {
				return fmt.Errorf("native member %q appears more than once", nativeMember)
			}
			nativeMembers[nativeMember] = struct{}{}
		}
		replicaIDs[replicaID] = struct{}{}
		runtimes[runtimeIncarnation] = struct{}{}
	}
	return nil
}

func validateIncarnation(incarnation ReplicaIncarnation) error {
	if incarnation.ReplicaID == "" {
		return errors.New("replica incarnation ID must not be empty")
	}
	if incarnation.SlotID == "" {
		return fmt.Errorf("replica %q slot must not be empty", incarnation.ReplicaID)
	}
	if incarnation.RuntimeIncarnation == "" {
		return fmt.Errorf("replica %q runtime incarnation must not be empty", incarnation.ReplicaID)
	}
	if len(incarnation.CapacityRefs) == 0 {
		return fmt.Errorf("replica %q has no physical capacity references", incarnation.ReplicaID)
	}

	seen := make(map[PodUID]struct{}, len(incarnation.CapacityRefs))
	for _, capacityRef := range incarnation.CapacityRefs {
		if capacityRef.Name == "" || capacityRef.UID == "" {
			return fmt.Errorf("replica %q has an incomplete physical capacity reference", incarnation.ReplicaID)
		}
		if _, duplicate := seen[capacityRef.UID]; duplicate {
			return fmt.Errorf("replica %q repeats Pod UID %q", incarnation.ReplicaID, capacityRef.UID)
		}
		seen[capacityRef.UID] = struct{}{}
	}
	return nil
}

func validateRegistry(registry ReplicaRegistry) error {
	replicaIDs := make(map[ReplicaID]struct{}, len(registry.Replicas))
	slots := make(map[CapacitySlotID]struct{}, len(registry.Replicas))
	runtimes := make(map[RuntimeIncarnationID]ReplicaID)
	podUIDs := make(map[PodUID]ReplicaID)
	for _, record := range registry.Replicas {
		if record.ReplicaID == "" || record.SlotID == "" {
			return errors.New("replica registry contains an incomplete stable identity")
		}
		if _, duplicate := replicaIDs[record.ReplicaID]; duplicate {
			return fmt.Errorf("registry replica %q appears more than once", record.ReplicaID)
		}
		if _, duplicate := slots[record.SlotID]; duplicate {
			return fmt.Errorf("registry slot %q appears more than once", record.SlotID)
		}
		if record.Current != nil {
			if err := validateRecordIncarnation(record, *record.Current); err != nil {
				return err
			}
			if err := registerIncarnationIdentities(*record.Current, runtimes, podUIDs); err != nil {
				return err
			}
		}
		var priorGeneration int64
		for _, entry := range record.History {
			if entry.TopologyGeneration <= 0 {
				return fmt.Errorf("replica %q has history without a topology generation", record.ReplicaID)
			}
			if entry.TopologyGeneration <= priorGeneration {
				return fmt.Errorf("replica %q history is not ordered by topology generation", record.ReplicaID)
			}
			if err := validateRecordIncarnation(record, entry.Incarnation); err != nil {
				return err
			}
			if err := registerIncarnationIdentities(entry.Incarnation, runtimes, podUIDs); err != nil {
				return err
			}
			if len(entry.NativeMembers) == 0 {
				return fmt.Errorf("replica %q has history without native members", record.ReplicaID)
			}
			priorGeneration = entry.TopologyGeneration
		}
		replicaIDs[record.ReplicaID] = struct{}{}
		slots[record.SlotID] = struct{}{}
	}
	return nil
}

func registerIncarnationIdentities(
	incarnation ReplicaIncarnation,
	runtimes map[RuntimeIncarnationID]ReplicaID,
	podUIDs map[PodUID]ReplicaID,
) error {
	if replicaID, duplicate := runtimes[incarnation.RuntimeIncarnation]; duplicate {
		return fmt.Errorf(
			"runtime incarnation %q belongs to both replica %q and replica %q",
			incarnation.RuntimeIncarnation,
			replicaID,
			incarnation.ReplicaID,
		)
	}
	runtimes[incarnation.RuntimeIncarnation] = incarnation.ReplicaID
	for _, capacityRef := range incarnation.CapacityRefs {
		if replicaID, duplicate := podUIDs[capacityRef.UID]; duplicate {
			return fmt.Errorf(
				"Pod UID %q belongs to both replica %q and replica %q",
				capacityRef.UID,
				replicaID,
				incarnation.ReplicaID,
			)
		}
		podUIDs[capacityRef.UID] = incarnation.ReplicaID
	}
	return nil
}

func validateRecordIncarnation(record ReplicaRecord, incarnation ReplicaIncarnation) error {
	if err := validateIncarnation(incarnation); err != nil {
		return err
	}
	if incarnation.ReplicaID != record.ReplicaID || incarnation.SlotID != record.SlotID {
		return fmt.Errorf(
			"replica %q incarnation changes canonical identity to replica %q slot %q",
			record.ReplicaID,
			incarnation.ReplicaID,
			incarnation.SlotID,
		)
	}
	return nil
}

func validateCapacityObservation(observation CapacityObservation) error {
	if observation.AppliedRevision < 0 {
		return errors.New("capacity applied revision must not be negative")
	}
	seen := make(map[ReplicaID]struct{}, len(observation.Allocations))
	slots := make(map[CapacitySlotID]ReplicaID, len(observation.Allocations))
	runtimes := make(map[RuntimeIncarnationID]ReplicaID, len(observation.Allocations))
	podUIDs := make(map[PodUID]ReplicaID)
	for _, allocation := range observation.Allocations {
		if err := validateIncarnation(allocation.Incarnation); err != nil {
			return err
		}
		if _, duplicate := seen[allocation.Incarnation.ReplicaID]; duplicate {
			return fmt.Errorf(
				"capacity contains more than one allocation for replica %q",
				allocation.Incarnation.ReplicaID,
			)
		}
		if replicaID, duplicate := slots[allocation.Incarnation.SlotID]; duplicate {
			return fmt.Errorf(
				"capacity slot %q belongs to both replica %q and replica %q",
				allocation.Incarnation.SlotID,
				replicaID,
				allocation.Incarnation.ReplicaID,
			)
		}
		if err := registerIncarnationIdentities(allocation.Incarnation, runtimes, podUIDs); err != nil {
			return err
		}
		seen[allocation.Incarnation.ReplicaID] = struct{}{}
		slots[allocation.Incarnation.SlotID] = allocation.Incarnation.ReplicaID
	}
	return validateReleaseFences(observation.ReleaseFences)
}

func validateTrafficObservation(observation TrafficObservation) error {
	if observation.AppliedRevision < 0 {
		return errors.New("traffic applied revision must not be negative")
	}
	for name, memberships := range map[string][]ReplicaMembership{
		"admitted": observation.Admitted,
		"draining": observation.Draining,
		"drained":  observation.Drained,
	} {
		if err := validateMembershipIdentitySet(memberships); err != nil {
			return fmt.Errorf("validate %s traffic membership: %w", name, err)
		}
	}
	return nil
}

func validateMembershipIdentitySet(memberships []ReplicaMembership) error {
	seen := make(map[ReplicaID]struct{}, len(memberships))
	for _, membership := range memberships {
		if membership.ReplicaID == "" || membership.RuntimeIncarnation == "" {
			return errors.New("traffic membership contains an incomplete replica or runtime identity")
		}
		if len(membership.NativeMembers) == 0 {
			return fmt.Errorf("replica %q has no native members", membership.ReplicaID)
		}
		if _, duplicate := seen[membership.ReplicaID]; duplicate {
			return fmt.Errorf("replica %q appears more than once", membership.ReplicaID)
		}
		seen[membership.ReplicaID] = struct{}{}
	}
	return nil
}

func validateReleaseFences(fences []ReleaseFence) error {
	seen := make(map[ReplicaID]struct{}, len(fences))
	for _, fence := range fences {
		if fence.TransitionID == "" || fence.AuthorizingTopologyGeneration <= 0 ||
			fence.ReplicaID == "" || fence.SlotID == "" || len(fence.CapacityRefs) == 0 {
			return errors.New("release fence contains incomplete transition, topology, replica, slot, or Pod identity")
		}
		if _, duplicate := seen[fence.ReplicaID]; duplicate {
			return fmt.Errorf("release fence for replica %q appears more than once", fence.ReplicaID)
		}
		for _, capacityRef := range fence.CapacityRefs {
			if capacityRef.Name == "" || capacityRef.UID == "" {
				return fmt.Errorf("release fence for replica %q has an incomplete Pod identity", fence.ReplicaID)
			}
		}
		seen[fence.ReplicaID] = struct{}{}
	}
	return nil
}

func sameReplicaIDs(left, right []ReplicaID) bool {
	return slices.Equal(normalizeReplicaIDs(left), normalizeReplicaIDs(right))
}

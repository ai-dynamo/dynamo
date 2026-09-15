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
	"cmp"
	"errors"
	"fmt"
	"slices"
)

func validateCapacitySnapshot(snapshot CapacitySnapshot) error {
	replicaIDs := make(map[ReplicaID]struct{}, len(snapshot.Allocations))
	slotIDs := make(map[CapacitySlotID]ReplicaID, len(snapshot.Allocations))
	runtimeIDs := make(map[RuntimeIncarnationID]struct{}, len(snapshot.Allocations))
	podUIDs := make(map[PodUID]struct{})
	podNames := make(map[string]struct{})

	// Validate every complete allocation and reject identity overlap across replicas.
	for _, allocation := range snapshot.Allocations {
		if err := validateReplicaIncarnation(
			allocation.Incarnation,
			allocation.Availability == ReplicaAvailabilityAvailable,
		); err != nil {
			return fmt.Errorf("validate capacity allocation: %w", err)
		}
		if _, exists := replicaIDs[allocation.Incarnation.ReplicaID]; exists {
			return fmt.Errorf("duplicate capacity allocation replica ID %q", allocation.Incarnation.ReplicaID)
		}
		replicaIDs[allocation.Incarnation.ReplicaID] = struct{}{}
		if _, exists := slotIDs[allocation.Incarnation.SlotID]; exists {
			return fmt.Errorf("duplicate capacity slot ID %q", allocation.Incarnation.SlotID)
		}
		slotIDs[allocation.Incarnation.SlotID] = allocation.Incarnation.ReplicaID
		if allocation.Incarnation.RuntimeID != "" {
			if _, exists := runtimeIDs[allocation.Incarnation.RuntimeID]; exists {
				return fmt.Errorf("duplicate capacity runtime incarnation ID %q", allocation.Incarnation.RuntimeID)
			}
			runtimeIDs[allocation.Incarnation.RuntimeID] = struct{}{}
		}

		if !validReplicaAvailability(allocation.Availability) {
			return fmt.Errorf(
				"capacity allocation %q has invalid availability %q",
				allocation.Incarnation.ReplicaID,
				allocation.Availability,
			)
		}

		for _, capacityRef := range allocation.Incarnation.CapacityRefs {
			podName := capacityRef.Namespace + "/" + capacityRef.Name
			if _, exists := podNames[podName]; exists {
				return fmt.Errorf("capacity Pod name %q belongs to multiple allocations", podName)
			}
			podNames[podName] = struct{}{}
			if _, exists := podUIDs[capacityRef.UID]; exists {
				return fmt.Errorf("capacity Pod UID %q belongs to multiple allocations", capacityRef.UID)
			}
			podUIDs[capacityRef.UID] = struct{}{}
		}
	}

	// Fences are durable logical-slot state and remain observable after their concrete Pods disappear.
	fencedReplicas := make(map[ReplicaID]CapacitySlotID, len(snapshot.FencedReplicaSlots))
	fencedSlots := make(map[CapacitySlotID]ReplicaID, len(snapshot.FencedReplicaSlots))
	allocations := replicaAllocationByID(snapshot)
	for _, binding := range snapshot.FencedReplicaSlots {
		if binding.ReplicaID == "" {
			return errors.New("fenced capacity replica ID must not be empty")
		}
		if binding.SlotID == "" {
			return fmt.Errorf("fenced capacity replica %q has an empty slot ID", binding.ReplicaID)
		}
		if _, exists := fencedReplicas[binding.ReplicaID]; exists {
			return fmt.Errorf("duplicate fenced capacity replica ID %q", binding.ReplicaID)
		}
		if previous, exists := fencedSlots[binding.SlotID]; exists {
			return fmt.Errorf(
				"fenced capacity slot %q belongs to replicas %q and %q",
				binding.SlotID,
				previous,
				binding.ReplicaID,
			)
		}
		if allocation, exists := allocations[binding.ReplicaID]; exists &&
			allocation.Incarnation.SlotID != binding.SlotID {
			return fmt.Errorf(
				"fenced capacity replica %q changed slot from %q to %q",
				binding.ReplicaID,
				binding.SlotID,
				allocation.Incarnation.SlotID,
			)
		}
		if allocatedReplica, exists := slotIDs[binding.SlotID]; exists && allocatedReplica != binding.ReplicaID {
			return fmt.Errorf(
				"fenced capacity slot %q for replica %q is allocated to replica %q",
				binding.SlotID,
				binding.ReplicaID,
				allocatedReplica,
			)
		}
		fencedReplicas[binding.ReplicaID] = binding.SlotID
		fencedSlots[binding.SlotID] = binding.ReplicaID
	}
	return nil
}

func validateReplicaSlotBindings(label string, bindings []ReplicaSlotBinding) error {
	replicaSlots := make(map[ReplicaID]CapacitySlotID, len(bindings))
	slotReplicas := make(map[CapacitySlotID]ReplicaID, len(bindings))
	for _, binding := range bindings {
		if binding.ReplicaID == "" {
			return fmt.Errorf("%s replica ID must not be empty", label)
		}
		if binding.SlotID == "" {
			return fmt.Errorf("%s replica %q has an empty capacity slot ID", label, binding.ReplicaID)
		}
		if _, exists := replicaSlots[binding.ReplicaID]; exists {
			return fmt.Errorf("duplicate %s replica ID %q", label, binding.ReplicaID)
		}
		if previous, exists := slotReplicas[binding.SlotID]; exists {
			return fmt.Errorf(
				"%s capacity slot %q belongs to replicas %q and %q",
				label,
				binding.SlotID,
				previous,
				binding.ReplicaID,
			)
		}
		replicaSlots[binding.ReplicaID] = binding.SlotID
		slotReplicas[binding.SlotID] = binding.ReplicaID
	}
	return nil
}

func normalizeReplicaSlotBindings(bindings []ReplicaSlotBinding) ([]ReplicaSlotBinding, error) {
	replicaSlots := make(map[ReplicaID]CapacitySlotID, len(bindings))
	slotReplicas := make(map[CapacitySlotID]ReplicaID, len(bindings))
	for _, binding := range bindings {
		if binding.ReplicaID == "" || binding.SlotID == "" {
			return nil, errors.New("replica-slot binding requires non-empty replica and slot IDs")
		}
		if slotID, exists := replicaSlots[binding.ReplicaID]; exists {
			if slotID != binding.SlotID {
				return nil, fmt.Errorf(
					"replica %q has conflicting capacity slots %q and %q",
					binding.ReplicaID,
					slotID,
					binding.SlotID,
				)
			}
			continue
		}
		if replicaID, exists := slotReplicas[binding.SlotID]; exists && replicaID != binding.ReplicaID {
			return nil, fmt.Errorf(
				"capacity slot %q belongs to replicas %q and %q",
				binding.SlotID,
				replicaID,
				binding.ReplicaID,
			)
		}
		replicaSlots[binding.ReplicaID] = binding.SlotID
		slotReplicas[binding.SlotID] = binding.ReplicaID
	}

	normalized := make([]ReplicaSlotBinding, 0, len(replicaSlots))
	for replicaID, slotID := range replicaSlots {
		normalized = append(normalized, ReplicaSlotBinding{ReplicaID: replicaID, SlotID: slotID})
	}
	slices.SortFunc(normalized, func(left, right ReplicaSlotBinding) int {
		if byReplica := cmp.Compare(left.ReplicaID, right.ReplicaID); byReplica != 0 {
			return byReplica
		}
		return cmp.Compare(left.SlotID, right.SlotID)
	})
	return normalized, nil
}

func replicaIDsForSlotBindings(bindings []ReplicaSlotBinding) []ReplicaID {
	replicaIDs := make([]ReplicaID, len(bindings))
	for i, binding := range bindings {
		replicaIDs[i] = binding.ReplicaID
	}
	return normalizeReplicaIDs(replicaIDs)
}

func sameReplicaSlotBindings(left, right []ReplicaSlotBinding) bool {
	normalizedLeft, leftErr := normalizeReplicaSlotBindings(left)
	normalizedRight, rightErr := normalizeReplicaSlotBindings(right)
	return leftErr == nil && rightErr == nil && slices.Equal(normalizedLeft, normalizedRight)
}

func validateCapacityRef(capacityRef CapacityRef) error {
	if capacityRef.Namespace == "" {
		return errors.New("capacity Pod namespace must not be empty")
	}
	if capacityRef.Name == "" {
		return errors.New("capacity Pod name must not be empty")
	}
	if capacityRef.UID == "" {
		return errors.New("capacity Pod UID must not be empty")
	}
	return nil
}

func validateReplicaIncarnation(incarnation ReplicaIncarnation, requireRuntime bool) error {
	if incarnation.ReplicaID == "" {
		return errors.New("replica incarnation ID must not be empty")
	}
	if incarnation.SlotID == "" {
		return fmt.Errorf("replica incarnation %q has an empty capacity slot ID", incarnation.ReplicaID)
	}
	if len(incarnation.CapacityRefs) == 0 {
		return fmt.Errorf("replica incarnation %q has no concrete capacity references", incarnation.ReplicaID)
	}
	if requireRuntime && incarnation.RuntimeID == "" {
		return fmt.Errorf("available replica incarnation %q has an empty runtime ID", incarnation.ReplicaID)
	}

	// One incarnation is an exact set of concrete Pods; duplicate names or UIDs make correlation ambiguous.
	podNames := make(map[string]struct{}, len(incarnation.CapacityRefs))
	podUIDs := make(map[PodUID]struct{}, len(incarnation.CapacityRefs))
	for _, capacityRef := range incarnation.CapacityRefs {
		if err := validateCapacityRef(capacityRef); err != nil {
			return fmt.Errorf("replica incarnation %q: %w", incarnation.ReplicaID, err)
		}
		podName := capacityRef.Namespace + "/" + capacityRef.Name
		if _, exists := podNames[podName]; exists {
			return fmt.Errorf("replica incarnation %q repeats capacity Pod name %q", incarnation.ReplicaID, podName)
		}
		podNames[podName] = struct{}{}
		if _, exists := podUIDs[capacityRef.UID]; exists {
			return fmt.Errorf(
				"replica incarnation %q repeats capacity Pod UID %q",
				incarnation.ReplicaID,
				capacityRef.UID,
			)
		}
		podUIDs[capacityRef.UID] = struct{}{}
	}
	return nil
}

func validateTrafficSnapshot(snapshot TrafficSnapshot) error {
	if snapshot.LatestCommand != nil {
		if err := validateTrafficCommandObservation(*snapshot.LatestCommand); err != nil {
			return fmt.Errorf("validate latest traffic command: %w", err)
		}
	}
	if err := validateReplicaIncarnations("traffic-admitted", snapshot.Admitted, true); err != nil {
		return err
	}
	if err := validateHistoricalReplicaIncarnations("traffic-drained", snapshot.Drained); err != nil {
		return err
	}
	if overlap := intersectReplicaIncarnations(snapshot.Admitted, snapshot.Drained); len(overlap) != 0 {
		return fmt.Errorf("traffic incarnations cannot be both admitted and drained: %v", overlap)
	}
	if err := validateTrafficRuntimeIdentities(snapshot); err != nil {
		return err
	}
	if len(snapshot.Drained) != 0 && snapshot.LatestCommand == nil {
		return errors.New("drained traffic state must carry its latest traffic command")
	}
	return nil
}

func validateTrafficCommandObservation(observation TrafficCommandObservation) error {
	if err := validateTrafficCommand(observation.Command); err != nil {
		return err
	}
	switch observation.Phase {
	case TrafficCommandPhaseAccepted:
		if observation.Failure != nil {
			return errors.New("accepted traffic command must not carry a failure")
		}
	case TrafficCommandPhaseRefused:
		if err := validateFailure(observation.Failure); err != nil {
			return fmt.Errorf("validate refused traffic command: %w", err)
		}
		if observation.Failure.Classification != FailureClassificationTerminal {
			return errors.New("refused traffic command requires a terminal failure")
		}
	case TrafficCommandPhaseFailed:
		if err := validateFailure(observation.Failure); err != nil {
			return fmt.Errorf("validate failed traffic command: %w", err)
		}
	default:
		return fmt.Errorf("invalid traffic command phase %q", observation.Phase)
	}
	return nil
}

func validateTrafficCommand(command TrafficCommand) error {
	if command.Action != TrafficActionAdmit && command.Action != TrafficActionWithdraw {
		return fmt.Errorf("invalid traffic command action %q", command.Action)
	}
	if command.Request.Revision <= 0 {
		return fmt.Errorf("traffic command revision must be positive: %d", command.Request.Revision)
	}
	if command.Request.OperationID == "" {
		return errors.New("traffic command operation ID must not be empty")
	}
	if command.Request.TopologyGeneration <= 0 {
		return fmt.Errorf(
			"traffic command topology generation must be positive: %d",
			command.Request.TopologyGeneration,
		)
	}
	switch command.Action {
	case TrafficActionAdmit:
		if err := validateReplicaIncarnations("traffic-command", command.Request.Replicas, true); err != nil {
			return err
		}
	case TrafficActionWithdraw:
		if err := validateHistoricalReplicaIncarnations("traffic-command", command.Request.Replicas); err != nil {
			return err
		}
		if err := validateReplicaRuntimeIdentities("traffic-command", command.Request.Replicas); err != nil {
			return err
		}
	}
	if len(command.Request.Replicas) == 0 {
		return errors.New("traffic command must name at least one exact replica incarnation")
	}
	return nil
}

func validateHistoricalReplicaIncarnations(label string, incarnations []ReplicaIncarnation) error {
	// Historical drains may share stable or physical identities, but each complete incarnation appears only once.
	seen := make([]ReplicaIncarnation, 0, len(incarnations))
	for _, incarnation := range incarnations {
		if err := validateReplicaIncarnation(incarnation, true); err != nil {
			return fmt.Errorf("validate %s replica: %w", label, err)
		}
		if slices.ContainsFunc(seen, func(candidate ReplicaIncarnation) bool {
			return sameReplicaIncarnation(candidate, incarnation)
		}) {
			return fmt.Errorf("duplicate %s exact incarnation for logical replica %q", label, incarnation.ReplicaID)
		}
		seen = append(seen, incarnation)
	}
	return nil
}

func validateTrafficRuntimeIdentities(snapshot TrafficSnapshot) error {
	incarnations := append(cloneReplicaIncarnations(snapshot.Admitted), snapshot.Drained...)
	return validateReplicaRuntimeIdentities("traffic", incarnations)
}

func validateReplicaRuntimeIdentities(label string, incarnations []ReplicaIncarnation) error {
	// A runtime identity remains globally exact even when old and new logical incarnations coexist.
	runtimeIncarnations := make(map[RuntimeIncarnationID]ReplicaIncarnation)
	for _, incarnation := range incarnations {
		previous, exists := runtimeIncarnations[incarnation.RuntimeID]
		if exists && !sameReplicaIncarnation(previous, incarnation) {
			return fmt.Errorf(
				"%s runtime incarnation ID %q identifies multiple exact incarnations",
				label,
				incarnation.RuntimeID,
			)
		}
		runtimeIncarnations[incarnation.RuntimeID] = incarnation
	}
	return nil
}

func validateReleaseAuthorization(authorization ReleaseAuthorization) error {
	if authorization.ID == "" {
		return errors.New("release authorization ID must not be empty")
	}
	if authorization.OperationID == "" {
		return errors.New("release authorization operation ID must not be empty")
	}
	if authorization.TopologyGeneration < 0 {
		return fmt.Errorf(
			"release authorization topology generation must not be negative: %d",
			authorization.TopologyGeneration,
		)
	}
	if authorization.TargetReplicas < 0 {
		return fmt.Errorf(
			"release authorization target replicas must not be negative: %d",
			authorization.TargetReplicas,
		)
	}
	replicaIDs := make(map[ReplicaID]struct{}, len(authorization.Replicas))
	slotIDs := make(map[CapacitySlotID]struct{}, len(authorization.Replicas))
	podNames := make(map[string]struct{})
	podUIDs := make(map[PodUID]struct{})

	// Require complete, physically disjoint authorization for every named replica.
	for _, replica := range authorization.Replicas {
		if replica.ReplicaID == "" {
			return errors.New("authorized replica ID must not be empty")
		}
		if _, exists := replicaIDs[replica.ReplicaID]; exists {
			return fmt.Errorf("duplicate authorized replica ID %q", replica.ReplicaID)
		}
		replicaIDs[replica.ReplicaID] = struct{}{}
		if replica.SlotID == "" {
			return fmt.Errorf("authorized replica %q has an empty capacity slot", replica.ReplicaID)
		}
		if _, exists := slotIDs[replica.SlotID]; exists {
			return fmt.Errorf("duplicate authorized capacity slot ID %q", replica.SlotID)
		}
		slotIDs[replica.SlotID] = struct{}{}
		if len(replica.CapacityRefs) == 0 {
			continue
		}

		for _, capacityRef := range replica.CapacityRefs {
			if err := validateCapacityRef(capacityRef); err != nil {
				return fmt.Errorf("authorized replica %q: %w", replica.ReplicaID, err)
			}
			podName := capacityRef.Namespace + "/" + capacityRef.Name
			if _, exists := podNames[podName]; exists {
				return fmt.Errorf("authorized capacity Pod name %q appears more than once", podName)
			}
			podNames[podName] = struct{}{}
			if _, exists := podUIDs[capacityRef.UID]; exists {
				return fmt.Errorf("authorized capacity Pod UID %q appears more than once", capacityRef.UID)
			}
			podUIDs[capacityRef.UID] = struct{}{}
		}
	}
	return nil
}

func validateCapacityReleaseObservation(observation CapacityReleaseObservation, releaseID string) error {
	switch observation.Phase {
	case CapacityReleasePhaseAbsent:
		if observation.ReleaseID != "" && observation.ReleaseID != releaseID {
			return fmt.Errorf("release observation ID %q does not match %q", observation.ReleaseID, releaseID)
		}
		if observation.Failure != nil {
			return errors.New("absent release observation must not carry a failure")
		}
	case CapacityReleasePhaseApplying, CapacityReleasePhaseApplied:
		if observation.ReleaseID != releaseID {
			return fmt.Errorf("release observation ID %q does not match %q", observation.ReleaseID, releaseID)
		}
		if observation.Failure != nil {
			return fmt.Errorf("release phase %q must not carry a failure", observation.Phase)
		}
	case CapacityReleasePhaseRefused:
		if observation.ReleaseID != releaseID {
			return fmt.Errorf("release observation ID %q does not match %q", observation.ReleaseID, releaseID)
		}
		if err := validateFailure(observation.Failure); err != nil {
			return fmt.Errorf("validate refused release: %w", err)
		}
	case CapacityReleasePhaseFailed:
		if observation.ReleaseID != releaseID {
			return fmt.Errorf("release observation ID %q does not match %q", observation.ReleaseID, releaseID)
		}
		if err := validateFailure(observation.Failure); err != nil {
			return fmt.Errorf("validate failed release: %w", err)
		}
	default:
		return fmt.Errorf("invalid capacity release phase %q", observation.Phase)
	}
	return nil
}

func validReplicaAvailability(availability ReplicaAvailability) bool {
	return availability == ReplicaAvailabilityAvailable ||
		availability == ReplicaAvailabilityUnavailable ||
		availability == ReplicaAvailabilityUnknown
}

func cloneCapacitySnapshot(snapshot CapacitySnapshot) CapacitySnapshot {
	cloned := CapacitySnapshot{
		Allocations:        make([]ReplicaAllocation, len(snapshot.Allocations)),
		FencedReplicaSlots: slices.Clone(snapshot.FencedReplicaSlots),
	}
	for i, allocation := range snapshot.Allocations {
		cloned.Allocations[i] = allocation
		cloned.Allocations[i].Incarnation = cloneReplicaIncarnation(allocation.Incarnation)
	}
	return cloned
}

func requiredReplicaAllocations(incarnations []ReplicaIncarnation) []RequiredReplicaAllocation {
	required := make([]RequiredReplicaAllocation, len(incarnations))
	for i, incarnation := range incarnations {
		required[i] = RequiredReplicaAllocation{
			ReplicaID: incarnation.ReplicaID,
			SlotID:    incarnation.SlotID,
		}
	}
	return normalizeRequiredReplicaAllocations(required)
}

func restoredReplicaAllocations(memberships []ReplicaNativeMembership) []RequiredReplicaAllocation {
	required := make([]RequiredReplicaAllocation, len(memberships))
	for i, membership := range memberships {
		required[i] = RequiredReplicaAllocation{
			ReplicaID: membership.ReplicaID,
			SlotID:    membership.SlotID,
		}
	}
	return normalizeRequiredReplicaAllocations(required)
}

func normalizeRequiredReplicaAllocations(required []RequiredReplicaAllocation) []RequiredReplicaAllocation {
	normalized := slices.Clone(required)
	slices.SortFunc(normalized, func(left, right RequiredReplicaAllocation) int {
		if comparison := cmp.Compare(left.ReplicaID, right.ReplicaID); comparison != 0 {
			return comparison
		}
		return cmp.Compare(left.SlotID, right.SlotID)
	})
	return normalized
}

func validateCapacityRequest(request CapacityRequest) error {
	if request.OperationID == "" {
		return errors.New("capacity request operation ID must not be empty")
	}
	if request.TopologyGeneration <= 0 {
		return fmt.Errorf(
			"capacity request topology generation must be positive: %d",
			request.TopologyGeneration,
		)
	}
	if request.TargetReplicas < 0 {
		return fmt.Errorf("capacity request target replicas must not be negative: %d", request.TargetReplicas)
	}
	if int32(len(request.RequiredReplicas)) > request.TargetReplicas {
		return fmt.Errorf(
			"capacity request requires %d exact replicas above target %d",
			len(request.RequiredReplicas),
			request.TargetReplicas,
		)
	}

	replicaIDs := make(map[ReplicaID]struct{}, len(request.RequiredReplicas))
	slotIDs := make(map[CapacitySlotID]struct{}, len(request.RequiredReplicas))
	for _, required := range request.RequiredReplicas {
		if required.ReplicaID == "" {
			return errors.New("capacity request required replica ID must not be empty")
		}
		if required.SlotID == "" {
			return fmt.Errorf("capacity request required replica %q has an empty slot ID", required.ReplicaID)
		}
		if _, exists := replicaIDs[required.ReplicaID]; exists {
			return fmt.Errorf("capacity request repeats required replica %q", required.ReplicaID)
		}
		if _, exists := slotIDs[required.SlotID]; exists {
			return fmt.Errorf("capacity request repeats required slot %q", required.SlotID)
		}
		replicaIDs[required.ReplicaID] = struct{}{}
		slotIDs[required.SlotID] = struct{}{}
	}
	return nil
}

func cloneTrafficSnapshot(snapshot TrafficSnapshot) TrafficSnapshot {
	return TrafficSnapshot{
		LatestCommand: cloneTrafficCommandObservation(snapshot.LatestCommand),
		Admitted:      cloneReplicaIncarnations(snapshot.Admitted),
		Drained:       cloneReplicaIncarnations(snapshot.Drained),
	}
}

func cloneTrafficCommandObservation(observation *TrafficCommandObservation) *TrafficCommandObservation {
	if observation == nil {
		return nil
	}
	cloned := *observation
	cloned.Command = *cloneTrafficCommand(&observation.Command)
	cloned.Failure = cloneFailure(observation.Failure)
	return &cloned
}

func cloneTrafficCommand(command *TrafficCommand) *TrafficCommand {
	if command == nil {
		return nil
	}
	cloned := *command
	cloned.Request.Replicas = cloneReplicaIncarnations(command.Request.Replicas)
	return &cloned
}

func sameTrafficCommand(left, right TrafficCommand) bool {
	return left.Action == right.Action &&
		left.Request.Revision == right.Request.Revision &&
		left.Request.OperationID == right.Request.OperationID &&
		left.Request.TopologyGeneration == right.Request.TopologyGeneration &&
		sameReplicaIncarnations(left.Request.Replicas, right.Request.Replicas)
}

func sameTrafficCommandPayload(left, right TrafficCommand) bool {
	left.Request.Revision = 1
	right.Request.Revision = 1
	return sameTrafficCommand(left, right)
}

func cloneReleaseAuthorization(authorization *ReleaseAuthorization) *ReleaseAuthorization {
	if authorization == nil {
		return nil
	}

	cloned := *authorization
	cloned.Replicas = make([]AuthorizedReplica, len(authorization.Replicas))
	for i, replica := range authorization.Replicas {
		cloned.Replicas[i] = AuthorizedReplica{
			ReplicaID:    replica.ReplicaID,
			SlotID:       replica.SlotID,
			CapacityRefs: slices.Clone(replica.CapacityRefs),
		}
	}
	return &cloned
}

func replicaAllocationByID(snapshot CapacitySnapshot) map[ReplicaID]ReplicaAllocation {
	allocations := make(map[ReplicaID]ReplicaAllocation, len(snapshot.Allocations))
	for _, allocation := range snapshot.Allocations {
		allocations[allocation.Incarnation.ReplicaID] = allocation
	}
	return allocations
}

func validateReplicaIncarnations(
	label string,
	incarnations []ReplicaIncarnation,
	requireRuntime bool,
) error {
	replicaIDs := make(map[ReplicaID]struct{}, len(incarnations))
	slotIDs := make(map[CapacitySlotID]struct{}, len(incarnations))
	runtimeIDs := make(map[RuntimeIncarnationID]struct{}, len(incarnations))
	podNames := make(map[string]struct{})
	podUIDs := make(map[PodUID]struct{})
	for _, incarnation := range incarnations {
		if err := validateReplicaIncarnation(incarnation, requireRuntime); err != nil {
			return fmt.Errorf("validate %s replica: %w", label, err)
		}
		if _, exists := replicaIDs[incarnation.ReplicaID]; exists {
			return fmt.Errorf("duplicate %s logical replica ID %q", label, incarnation.ReplicaID)
		}
		replicaIDs[incarnation.ReplicaID] = struct{}{}
		if _, exists := slotIDs[incarnation.SlotID]; exists {
			return fmt.Errorf("duplicate %s capacity slot ID %q", label, incarnation.SlotID)
		}
		slotIDs[incarnation.SlotID] = struct{}{}
		if incarnation.RuntimeID != "" {
			if _, exists := runtimeIDs[incarnation.RuntimeID]; exists {
				return fmt.Errorf("duplicate %s runtime incarnation ID %q", label, incarnation.RuntimeID)
			}
			runtimeIDs[incarnation.RuntimeID] = struct{}{}
		}
		for _, capacityRef := range incarnation.CapacityRefs {
			podName := capacityRef.Namespace + "/" + capacityRef.Name
			if _, exists := podNames[podName]; exists {
				return fmt.Errorf("duplicate %s capacity Pod name %q", label, podName)
			}
			podNames[podName] = struct{}{}
			if _, exists := podUIDs[capacityRef.UID]; exists {
				return fmt.Errorf("duplicate %s capacity Pod UID %q", label, capacityRef.UID)
			}
			podUIDs[capacityRef.UID] = struct{}{}
		}
	}
	return nil
}

func cloneReplicaIncarnation(incarnation ReplicaIncarnation) ReplicaIncarnation {
	cloned := incarnation
	cloned.CapacityRefs = slices.Clone(incarnation.CapacityRefs)
	return cloned
}

func cloneReplicaIncarnations(incarnations []ReplicaIncarnation) []ReplicaIncarnation {
	if incarnations == nil {
		return nil
	}
	cloned := make([]ReplicaIncarnation, len(incarnations))
	for i, incarnation := range incarnations {
		cloned[i] = cloneReplicaIncarnation(incarnation)
	}
	return cloned
}

func normalizeReplicaIncarnations(incarnations []ReplicaIncarnation) []ReplicaIncarnation {
	normalized := cloneReplicaIncarnations(incarnations)
	for i := range normalized {
		slices.SortFunc(normalized[i].CapacityRefs, compareCapacityRefs)
	}
	slices.SortFunc(normalized, compareReplicaIncarnations)
	return slices.CompactFunc(normalized, sameReplicaIncarnation)
}

func intersectReplicaIncarnations(left, right []ReplicaIncarnation) []ReplicaIncarnation {
	intersection := make([]ReplicaIncarnation, 0)
	for _, candidate := range normalizeReplicaIncarnations(left) {
		if slices.ContainsFunc(right, func(other ReplicaIncarnation) bool {
			return sameReplicaIncarnation(candidate, other)
		}) {
			intersection = append(intersection, candidate)
		}
	}
	return intersection
}

func missingReplicaIncarnations(required, observed []ReplicaIncarnation) []ReplicaIncarnation {
	missing := make([]ReplicaIncarnation, 0)
	for _, candidate := range normalizeReplicaIncarnations(required) {
		if !slices.ContainsFunc(observed, func(other ReplicaIncarnation) bool {
			return sameReplicaIncarnation(candidate, other)
		}) {
			missing = append(missing, candidate)
		}
	}
	return missing
}

func containsAllReplicaIncarnations(haystack, needles []ReplicaIncarnation) bool {
	return len(intersectReplicaIncarnations(haystack, needles)) == len(normalizeReplicaIncarnations(needles))
}

func sameReplicaIncarnations(left, right []ReplicaIncarnation) bool {
	return slices.EqualFunc(
		normalizeReplicaIncarnations(left),
		normalizeReplicaIncarnations(right),
		sameReplicaIncarnation,
	)
}

func replicaIncarnationIDs(incarnations []ReplicaIncarnation) []ReplicaID {
	replicaIDs := make([]ReplicaID, len(incarnations))
	for i, incarnation := range incarnations {
		replicaIDs[i] = incarnation.ReplicaID
	}
	return normalizeReplicaIDs(replicaIDs)
}

func sameReplicaIncarnation(left, right ReplicaIncarnation) bool {
	return compareReplicaIncarnations(left, right) == 0
}

func compareReplicaIncarnations(left, right ReplicaIncarnation) int {
	if comparison := cmp.Compare(left.ReplicaID, right.ReplicaID); comparison != 0 {
		return comparison
	}
	if comparison := cmp.Compare(left.SlotID, right.SlotID); comparison != 0 {
		return comparison
	}
	if comparison := cmp.Compare(left.RuntimeID, right.RuntimeID); comparison != 0 {
		return comparison
	}
	leftRefs := slices.Clone(left.CapacityRefs)
	rightRefs := slices.Clone(right.CapacityRefs)
	slices.SortFunc(leftRefs, compareCapacityRefs)
	slices.SortFunc(rightRefs, compareCapacityRefs)
	return slices.CompareFunc(leftRefs, rightRefs, compareCapacityRefs)
}

func compareCapacityRefs(left, right CapacityRef) int {
	if comparison := cmp.Compare(left.Namespace, right.Namespace); comparison != 0 {
		return comparison
	}
	if comparison := cmp.Compare(left.Name, right.Name); comparison != 0 {
		return comparison
	}
	return cmp.Compare(left.UID, right.UID)
}

func containsAllReplicaIDs(haystack []ReplicaID, needles []ReplicaID) bool {
	return len(intersectReplicaIDs(haystack, needles)) == len(needles)
}

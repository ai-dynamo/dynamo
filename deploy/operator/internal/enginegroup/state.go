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

func validateCapacitySnapshot(snapshot CapacitySnapshot) error {
	replicaIDs := make(map[ReplicaID]struct{}, len(snapshot.Allocations))
	slotIDs := make(map[CapacitySlotID]struct{}, len(snapshot.Allocations))
	podUIDs := make(map[PodUID]struct{})
	podNames := make(map[string]struct{})

	// Validate every complete allocation and reject identity overlap across replicas.
	for _, allocation := range snapshot.Allocations {
		if allocation.ID == "" {
			return errors.New("capacity allocation replica ID must not be empty")
		}
		if _, exists := replicaIDs[allocation.ID]; exists {
			return fmt.Errorf("duplicate capacity allocation replica ID %q", allocation.ID)
		}
		replicaIDs[allocation.ID] = struct{}{}

		if allocation.SlotID == "" {
			return fmt.Errorf("capacity allocation %q has an empty slot ID", allocation.ID)
		}
		if _, exists := slotIDs[allocation.SlotID]; exists {
			return fmt.Errorf("duplicate capacity slot ID %q", allocation.SlotID)
		}
		slotIDs[allocation.SlotID] = struct{}{}

		if !validReplicaAvailability(allocation.Availability) {
			return fmt.Errorf(
				"capacity allocation %q has invalid availability %q",
				allocation.ID,
				allocation.Availability,
			)
		}
		if len(allocation.CapacityRefs) == 0 {
			return fmt.Errorf("capacity allocation %q has no concrete capacity references", allocation.ID)
		}

		for _, capacityRef := range allocation.CapacityRefs {
			if err := validateCapacityRef(capacityRef); err != nil {
				return fmt.Errorf("capacity allocation %q: %w", allocation.ID, err)
			}
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

	// Fences are durable logical state and remain observable after their concrete Pods disappear.
	fencedReplicas := make(map[ReplicaID]struct{}, len(snapshot.FencedReplicas))
	for _, replicaID := range snapshot.FencedReplicas {
		if replicaID == "" {
			return errors.New("fenced capacity replica ID must not be empty")
		}
		if _, exists := fencedReplicas[replicaID]; exists {
			return fmt.Errorf("duplicate fenced capacity replica ID %q", replicaID)
		}
		fencedReplicas[replicaID] = struct{}{}
	}
	return nil
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

func validateTrafficSnapshot(snapshot TrafficSnapshot) error {
	if err := validateReplicaIDs("traffic-admitted", snapshot.Admitted); err != nil {
		return err
	}
	if err := validateReplicaIDs("traffic-drained", snapshot.Drained); err != nil {
		return err
	}
	if overlap := intersectReplicaIDs(snapshot.Admitted, snapshot.Drained); len(overlap) != 0 {
		return fmt.Errorf("traffic replicas cannot be both admitted and drained: %v", overlap)
	}
	if len(snapshot.Drained) != 0 && snapshot.OperationID == "" {
		return errors.New("drained traffic state must carry an operation ID")
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
		if len(replica.CapacityRefs) == 0 {
			if replica.SlotID != "" {
				return fmt.Errorf("fence-only authorized replica %q must not carry a capacity slot", replica.ReplicaID)
			}
			continue
		}
		if replica.SlotID == "" {
			return fmt.Errorf("authorized replica %q with capacity references has an empty slot ID", replica.ReplicaID)
		}
		if _, exists := slotIDs[replica.SlotID]; exists {
			return fmt.Errorf("duplicate authorized capacity slot ID %q", replica.SlotID)
		}
		slotIDs[replica.SlotID] = struct{}{}

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
		Allocations:    make([]ReplicaAllocation, len(snapshot.Allocations)),
		FencedReplicas: slices.Clone(snapshot.FencedReplicas),
	}
	for i, allocation := range snapshot.Allocations {
		cloned.Allocations[i] = allocation
		cloned.Allocations[i].CapacityRefs = slices.Clone(allocation.CapacityRefs)
	}
	return cloned
}

func cloneTrafficSnapshot(snapshot TrafficSnapshot) TrafficSnapshot {
	return TrafficSnapshot{
		OperationID: snapshot.OperationID,
		Admitted:    slices.Clone(snapshot.Admitted),
		Drained:     slices.Clone(snapshot.Drained),
	}
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
		allocations[allocation.ID] = allocation
	}
	return allocations
}

func containsAllReplicaIDs(haystack []ReplicaID, needles []ReplicaID) bool {
	return len(intersectReplicaIDs(haystack, needles)) == len(needles)
}

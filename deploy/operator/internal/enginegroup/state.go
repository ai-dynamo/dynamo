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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

func cloneReplicaIncarnation(value ReplicaIncarnation) ReplicaIncarnation {
	value.CapacityRefs = slices.Clone(value.CapacityRefs)
	value.Members = slices.Clone(value.Members)
	return value
}

func cloneReplicaMembership(value ReplicaMembership) ReplicaMembership {
	value.Members = slices.Clone(value.Members)
	return value
}

func cloneReplicaMemberships(values []ReplicaMembership) []ReplicaMembership {
	cloned := make([]ReplicaMembership, 0, len(values))
	for _, value := range values {
		cloned = append(cloned, cloneReplicaMembership(value))
	}
	return cloned
}

func cloneTopology(value MembershipTopology) MembershipTopology {
	value.Replicas = cloneReplicaMemberships(value.Replicas)
	return value
}

func normalizeTopology(value MembershipTopology) MembershipTopology {
	value = cloneTopology(value)
	value.Replicas = normalizeMemberships(value.Replicas)
	return value
}

func cloneResolvedPlan(value ResolvedPlan) ResolvedPlan {
	change := value.Change
	if change.Grow != nil {
		replicas := make([]ReplicaTarget, 0, len(change.Grow.Replicas))
		for _, replica := range change.Grow.Replicas {
			replica.NativeMembers = slices.Clone(replica.NativeMembers)
			replicas = append(replicas, replica)
		}
		value.Change.Grow = &GrowChange{Replicas: replicas}
	}
	if change.Retire != nil {
		value.Change.Retire = &RetireChange{Replicas: slices.Clone(change.Retire.Replicas)}
	}
	if change.ReduceToSurvivors != nil {
		value.Change.ReduceToSurvivors = &ReduceToSurvivorsChange{
			Survivors: normalizeNativeMemberships(change.ReduceToSurvivors.Survivors),
		}
	}
	if change.Restore != nil {
		replicas := make([]RestorationTarget, 0, len(change.Restore.Replicas))
		for _, replica := range change.Restore.Replicas {
			replica.ReplicaTarget.NativeMembers = slices.Clone(replica.ReplicaTarget.NativeMembers)
			replicas = append(replicas, replica)
		}
		value.Change.Restore = &RestoreChange{Replicas: replicas}
	}
	if change.Remap != nil {
		membership := make([]ReplicaNativeMembership, 0, len(change.Remap.Membership))
		for _, replica := range change.Remap.Membership {
			replica.NativeMembers = slices.Clone(replica.NativeMembers)
			membership = append(membership, replica)
		}
		value.Change.Remap = &RemapChange{Membership: membership}
	}
	return value
}

func normalizeResolvedPlan(value ResolvedPlan) ResolvedPlan {
	value = cloneResolvedPlan(value)
	switch value.Change.Kind {
	case PlanKindGrow:
		if value.Change.Grow != nil {
			for index := range value.Change.Grow.Replicas {
				value.Change.Grow.Replicas[index].NativeMembers = normalizeNativeMembers(
					value.Change.Grow.Replicas[index].NativeMembers,
				)
			}
			slices.SortFunc(value.Change.Grow.Replicas, func(left, right ReplicaTarget) int {
				return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
			})
		}
	case PlanKindRetire:
		if value.Change.Retire != nil {
			value.Change.Retire.Replicas = normalizeReplicaIDs(value.Change.Retire.Replicas)
		}
	case PlanKindReduceToSurvivors:
		if value.Change.ReduceToSurvivors != nil {
			value.Change.ReduceToSurvivors.Survivors = normalizeNativeMemberships(
				value.Change.ReduceToSurvivors.Survivors,
			)
		}
	case PlanKindRestore:
		if value.Change.Restore != nil {
			for index := range value.Change.Restore.Replicas {
				value.Change.Restore.Replicas[index].ReplicaTarget.NativeMembers = normalizeNativeMembers(
					value.Change.Restore.Replicas[index].ReplicaTarget.NativeMembers,
				)
			}
			slices.SortFunc(value.Change.Restore.Replicas, func(left, right RestorationTarget) int {
				return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
			})
		}
	case PlanKindRemap:
		if value.Change.Remap != nil {
			for index := range value.Change.Remap.Membership {
				value.Change.Remap.Membership[index].NativeMembers = normalizeNativeMembers(
					value.Change.Remap.Membership[index].NativeMembers,
				)
			}
			slices.SortFunc(value.Change.Remap.Membership, func(left, right ReplicaNativeMembership) int {
				return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
			})
		}
	}
	return value
}

func canonicalPlanDigest(plan ResolvedPlan) (string, error) {
	encoded, err := json.Marshal(normalizeResolvedPlan(plan))
	if err != nil {
		return "", fmt.Errorf("marshal canonical resolved plan: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

func normalizeJoiningReplicas(values []JoiningReplica) []JoiningReplica {
	return normalizeMemberships(values)
}

func cloneMembershipTarget(value *MembershipTarget) *MembershipTarget {
	if value == nil {
		return nil
	}
	cloned := *value
	cloned.BaseTopology = cloneTopology(value.BaseTopology)
	cloned.Plan = cloneResolvedPlan(value.Plan)
	cloned.Joining = cloneReplicaMemberships(value.Joining)
	return &cloned
}

func normalizeMembershipTarget(value MembershipTarget) MembershipTarget {
	value = *cloneMembershipTarget(&value)
	value.BaseTopology = normalizeTopology(value.BaseTopology)
	value.Plan = normalizeResolvedPlan(value.Plan)
	value.Joining = normalizeJoiningReplicas(value.Joining)
	return value
}

func canonicalMembershipTargetDigest(target MembershipTarget) (string, error) {
	target.TargetDigest = ""
	target.Validation.TargetDigest = ""
	encoded, err := json.Marshal(normalizeMembershipTarget(target))
	if err != nil {
		return "", fmt.Errorf("marshal canonical membership target: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

func cloneMembershipTransitionObservation(
	value *MembershipTransitionObservation,
) *MembershipTransitionObservation {
	if value == nil {
		return nil
	}
	cloned := *value
	if value.ResultTopology != nil {
		topology := cloneTopology(*value.ResultTopology)
		cloned.ResultTopology = &topology
	}
	cloned.Failure = cloneFailure(value.Failure)
	return &cloned
}

func cloneMembershipObservation(value MembershipObservation) MembershipObservation {
	value.UnavailableMembers = cloneReplicaMemberships(value.UnavailableMembers)
	value.CommittedTopology = cloneTopology(value.CommittedTopology)
	value.Transition = cloneMembershipTransitionObservation(value.Transition)
	return value
}

func sameMembershipTransitionObservation(left, right MembershipTransitionObservation) bool {
	if left.TransitionID != right.TransitionID ||
		left.ControlRevision != right.ControlRevision ||
		left.TargetDigest != right.TargetDigest ||
		left.Phase != right.Phase ||
		!sameFailure(left.Failure, right.Failure) {
		return false
	}
	if left.ResultTopology == nil || right.ResultTopology == nil {
		return left.ResultTopology == nil && right.ResultTopology == nil
	}
	return sameTopology(*left.ResultTopology, *right.ResultTopology)
}

func sameFailure(left, right *Failure) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return *left == *right
}

func cloneReplicaRecord(value ReplicaRecord) ReplicaRecord {
	value.DesiredNativeMembers = slices.Clone(value.DesiredNativeMembers)
	if value.Current != nil {
		current := cloneReplicaIncarnation(*value.Current)
		value.Current = &current
	}
	history := value.History
	value.History = make([]ReplicaHistoryEntry, 0, len(history))
	for _, entry := range history {
		entry.Incarnation = cloneReplicaIncarnation(entry.Incarnation)
		entry.NativeMembers = slices.Clone(entry.NativeMembers)
		value.History = append(value.History, entry)
	}
	return value
}

func cloneRegistry(value ReplicaRegistry) ReplicaRegistry {
	replicas := value.Replicas
	value.Replicas = make([]ReplicaRecord, 0, len(replicas))
	for _, replica := range replicas {
		value.Replicas = append(value.Replicas, cloneReplicaRecord(replica))
	}
	return value
}

func cloneTopologyHistory(value TopologyHistory) TopologyHistory {
	snapshots := value.Snapshots
	value.Snapshots = make([]MembershipTopology, 0, len(snapshots))
	for _, snapshot := range snapshots {
		value.Snapshots = append(value.Snapshots, cloneTopology(snapshot))
	}
	return value
}

func cloneFailure(value *Failure) *Failure {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneCapacityTarget(value *CapacityTarget) *CapacityTarget {
	if value == nil {
		return nil
	}
	cloned := *value
	cloned.Replicas = make([]CapacityReplicaTarget, 0, len(value.Replicas))
	for _, replica := range value.Replicas {
		if replica.Incarnation != nil {
			incarnation := cloneReplicaIncarnation(*replica.Incarnation)
			replica.Incarnation = &incarnation
		}
		if replica.Bootstrap != nil {
			bootstrap := *replica.Bootstrap
			bootstrap.NativeMembers = slices.Clone(replica.Bootstrap.NativeMembers)
			replica.Bootstrap = &bootstrap
		}
		cloned.Replicas = append(cloned.Replicas, replica)
	}
	cloned.ReleaseFences = make([]ReleaseFence, 0, len(value.ReleaseFences))
	for _, fence := range value.ReleaseFences {
		fence.CapacityRefs = slices.Clone(fence.CapacityRefs)
		cloned.ReleaseFences = append(cloned.ReleaseFences, fence)
	}
	return &cloned
}

func cloneCapacityObservation(value CapacityObservation) CapacityObservation {
	allocations := value.Allocations
	value.Allocations = make([]CapacityAllocation, 0, len(allocations))
	for _, allocation := range allocations {
		allocation.Incarnation = cloneReplicaIncarnation(allocation.Incarnation)
		value.Allocations = append(value.Allocations, allocation)
	}
	value.ReleaseFences = cloneReleaseFences(value.ReleaseFences)
	return value
}

func cloneTrafficTarget(value *TrafficTarget) *TrafficTarget {
	if value == nil {
		return nil
	}
	cloned := *value
	cloned.Admitted = cloneReplicaMemberships(value.Admitted)
	cloned.Drain = make([]TrafficDrainTarget, 0, len(value.Drain))
	for _, drain := range value.Drain {
		drain.Membership = cloneReplicaMembership(drain.Membership)
		cloned.Drain = append(cloned.Drain, drain)
	}
	return &cloned
}

func cloneTrafficObservation(value TrafficObservation) TrafficObservation {
	value.Admitted = cloneReplicaMemberships(value.Admitted)
	value.Draining = cloneReplicaMemberships(value.Draining)
	value.Drained = cloneReplicaMemberships(value.Drained)
	return value
}

func cloneServingProof(value *ServingProof) *ServingProof {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneTransition(value *TransitionStatus) *TransitionStatus {
	if value == nil {
		return nil
	}
	cloned := *value
	cloned.Spec.Plan = cloneResolvedPlan(value.Spec.Plan)
	cloned.PlanPreflight.Evidence = cloneValidationEvidence(value.PlanPreflight.Evidence)
	cloned.PlanPreflight.Rejection = cloneFailure(value.PlanPreflight.Rejection)
	cloned.TargetPreflight.Evidence = cloneValidationEvidence(value.TargetPreflight.Evidence)
	cloned.TargetPreflight.Rejection = cloneFailure(value.TargetPreflight.Rejection)
	cloned.Verification.Proof = cloneServingProof(value.Verification.Proof)
	cloned.Verification.Failure = cloneFailure(value.Verification.Failure)
	cloned.Failure = cloneFailure(value.Failure)
	return &cloned
}

func cloneValidationEvidence(value *ValidationEvidence) *ValidationEvidence {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneStatus(value GroupStatus) GroupStatus {
	value.Registry = cloneRegistry(value.Registry)
	value.Topologies = cloneTopologyHistory(value.Topologies)
	value.Capacity.Desired = cloneCapacityTarget(value.Capacity.Desired)
	value.Capacity.Accepted = cloneCapacityTarget(value.Capacity.Accepted)
	value.Capacity.Observed = cloneCapacityObservation(value.Capacity.Observed)
	value.Traffic.Desired = cloneTrafficTarget(value.Traffic.Desired)
	value.Traffic.Accepted = cloneTrafficTarget(value.Traffic.Accepted)
	value.Traffic.Observed = cloneTrafficObservation(value.Traffic.Observed)
	value.Membership.Desired = cloneMembershipTarget(value.Membership.Desired)
	value.Membership.Observed = cloneMembershipObservation(value.Membership.Observed)
	value.Transition = cloneTransition(value.Transition)
	return value
}

// Find returns a copy of the canonical replica record with the given identity.
func (r ReplicaRegistry) Find(replicaID ReplicaID) (ReplicaRecord, bool) {
	for _, replica := range r.Replicas {
		if replica.ReplicaID == replicaID {
			return cloneReplicaRecord(replica), true
		}
	}
	return ReplicaRecord{}, false
}

// Snapshot returns a copy of the immutable topology with the given generation.
func (h TopologyHistory) Snapshot(generation int64) (MembershipTopology, bool) {
	for _, snapshot := range h.Snapshots {
		if snapshot.Generation == generation {
			return cloneTopology(snapshot), true
		}
	}
	return MembershipTopology{}, false
}

// Current returns the current immutable topology snapshot.
func (h TopologyHistory) Current() (MembershipTopology, bool) {
	return h.Snapshot(h.CurrentGeneration)
}

func normalizeCapacityRefs(values []CapacityRef) []CapacityRef {
	normalized := slices.Clone(values)
	slices.SortFunc(normalized, func(left, right CapacityRef) int {
		if left.Name != right.Name {
			return strings.Compare(left.Name, right.Name)
		}
		return strings.Compare(string(left.UID), string(right.UID))
	})
	return normalized
}

func normalizeNativeMembers(values []NativeMemberID) []NativeMemberID {
	normalized := slices.Clone(values)
	slices.SortFunc(normalized, func(left, right NativeMemberID) int {
		return strings.Compare(string(left), string(right))
	})
	return normalized
}

func normalizeMemberIncarnations(values []NativeMemberIncarnation) []NativeMemberIncarnation {
	values = slices.Clone(values)
	slices.SortFunc(values, func(left, right NativeMemberIncarnation) int {
		if left.ID != right.ID {
			return strings.Compare(string(left.ID), string(right.ID))
		}
		return strings.Compare(string(left.RuntimeIncarnation), string(right.RuntimeIncarnation))
	})
	return values
}

func sameMemberIncarnations(left, right []NativeMemberIncarnation) bool {
	return slices.Equal(normalizeMemberIncarnations(left), normalizeMemberIncarnations(right))
}

func sameRuntimeSet(left, right []NativeMemberIncarnation) bool {
	leftIDs := make([]RuntimeIncarnationID, 0, len(left))
	rightIDs := make([]RuntimeIncarnationID, 0, len(right))
	for _, member := range left {
		leftIDs = append(leftIDs, member.RuntimeIncarnation)
	}
	for _, member := range right {
		rightIDs = append(rightIDs, member.RuntimeIncarnation)
	}
	slices.Sort(leftIDs)
	slices.Sort(rightIDs)
	return slices.Equal(leftIDs, rightIDs)
}

func nativeMemberIDs(members []NativeMemberIncarnation) []NativeMemberID {
	ids := make([]NativeMemberID, 0, len(members))
	for _, member := range members {
		ids = append(ids, member.ID)
	}
	return normalizeNativeMembers(ids)
}

func normalizeNativeMemberships(values []ReplicaNativeMembership) []ReplicaNativeMembership {
	values = slices.Clone(values)
	for index := range values {
		values[index].NativeMembers = normalizeNativeMembers(values[index].NativeMembers)
	}
	slices.SortFunc(values, func(left, right ReplicaNativeMembership) int {
		return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
	})
	return values
}

func normalizeIncarnation(value ReplicaIncarnation) ReplicaIncarnation {
	value.CapacityRefs = normalizeCapacityRefs(value.CapacityRefs)
	value.Members = normalizeMemberIncarnations(value.Members)
	return value
}

func normalizeMemberships(values []ReplicaMembership) []ReplicaMembership {
	normalized := cloneReplicaMemberships(values)
	for index := range normalized {
		normalized[index].Members = normalizeMemberIncarnations(normalized[index].Members)
	}
	slices.SortFunc(normalized, func(left, right ReplicaMembership) int {
		return strings.Compare(string(left.ReplicaID), string(right.ReplicaID))
	})
	merged := make([]ReplicaMembership, 0, len(normalized))
	for _, value := range normalized {
		last := len(merged) - 1
		if last >= 0 && merged[last].ReplicaID == value.ReplicaID {
			merged[last].Members = slices.Compact(normalizeMemberIncarnations(append(merged[last].Members, value.Members...)))
		} else {
			merged = append(merged, value)
		}
	}
	return merged
}

func normalizeReplicaIDs(values []ReplicaID) []ReplicaID {
	normalized := slices.Clone(values)
	slices.SortFunc(normalized, func(left, right ReplicaID) int {
		return strings.Compare(string(left), string(right))
	})
	return normalized
}

func sameCapacityRefs(left, right []CapacityRef) bool {
	return slices.Equal(normalizeCapacityRefs(left), normalizeCapacityRefs(right))
}

func sameIncarnation(left, right ReplicaIncarnation) bool {
	return left.ReplicaID == right.ReplicaID &&
		left.SlotID == right.SlotID &&
		sameRuntimeSet(left.Members, right.Members) &&
		sameCapacityRefs(left.CapacityRefs, right.CapacityRefs)
}

// SameIncarnation reports whether two incarnations name the same logical, runtime, and physical capacity.
func SameIncarnation(left, right ReplicaIncarnation) bool {
	return sameIncarnation(left, right)
}

// MembershipMatchesIncarnation checks that every active process belongs to this exact physical allocation.
func MembershipMatchesIncarnation(membership ReplicaMembership, incarnation ReplicaIncarnation) bool {
	if membership.ReplicaID != incarnation.ReplicaID {
		return false
	}
	// Native IDs may be remapped explicitly; physical provenance is bound to the independent process lifetimes.
	for _, member := range membership.Members {
		if !slices.ContainsFunc(incarnation.Members, func(hosted NativeMemberIncarnation) bool {
			return hosted.RuntimeIncarnation == member.RuntimeIncarnation
		}) {
			return false
		}
	}
	return true
}

func sameMembership(left, right ReplicaMembership) bool {
	return left.ReplicaID == right.ReplicaID &&
		sameMemberIncarnations(left.Members, right.Members)
}

func subtractMemberships(values, excluded []ReplicaMembership) []ReplicaMembership {
	remaining := make([]ReplicaMembership, 0, len(values))
	for _, value := range values {
		members := make([]NativeMemberIncarnation, 0, len(value.Members))
		for _, member := range value.Members {
			present := false
			for _, exclusion := range excluded {
				if exclusion.ReplicaID == value.ReplicaID && slices.Contains(exclusion.Members, member) {
					present = true
					break
				}
			}
			if !present {
				members = append(members, member)
			}
		}
		if len(members) > 0 {
			remaining = append(remaining, ReplicaMembership{ReplicaID: value.ReplicaID, Members: members})
		}
	}
	return normalizeMemberships(remaining)
}

func membershipsForReplicaIDs(topology MembershipTopology, ids []ReplicaID) []ReplicaMembership {
	values := make([]ReplicaMembership, 0, len(ids))
	for _, id := range ids {
		if membership, found := membershipByID(topology, id); found {
			values = append(values, membership)
		}
	}
	return normalizeMemberships(values)
}

func sameMemberships(left, right []ReplicaMembership) bool {
	left = normalizeMemberships(left)
	right = normalizeMemberships(right)
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if !sameMembership(left[index], right[index]) {
			return false
		}
	}
	return true
}

// SameMembershipSet reports whether two membership lists contain the same logical, runtime, and native identities.
// Ordering is not significant.
func SameMembershipSet(left, right []ReplicaMembership) bool {
	return sameMemberships(left, right)
}

func sameTopology(left, right MembershipTopology) bool {
	return left.Generation == right.Generation && sameMemberships(left.Replicas, right.Replicas)
}

// SameTopology reports whether two topologies have the same generation and exact membership identity set.
func SameTopology(left, right MembershipTopology) bool {
	return sameTopology(left, right)
}

func membershipByID(topology MembershipTopology, replicaID ReplicaID) (ReplicaMembership, bool) {
	for _, membership := range topology.Replicas {
		if membership.ReplicaID == replicaID {
			return cloneReplicaMembership(membership), true
		}
	}
	return ReplicaMembership{}, false
}

func allocationByID(observation CapacityObservation, replicaID ReplicaID) (CapacityAllocation, bool) {
	for _, allocation := range observation.Allocations {
		if allocation.Incarnation.ReplicaID == replicaID {
			allocation.Incarnation = cloneReplicaIncarnation(allocation.Incarnation)
			return allocation, true
		}
	}
	return CapacityAllocation{}, false
}

func appendTopology(history TopologyHistory, topology MembershipTopology) (TopologyHistory, error) {
	if topology.Generation <= 0 {
		return history, errors.New("topology generation must be positive")
	}

	// An equal generation is an immutable replay, while a lower unseen generation is stale.
	if existing, found := history.Snapshot(topology.Generation); found {
		if !sameTopology(existing, topology) {
			return history, fmt.Errorf("topology generation %d changed payload", topology.Generation)
		}
		history.CurrentGeneration = topology.Generation
		return history, nil
	}
	if history.CurrentGeneration >= topology.Generation {
		return history, fmt.Errorf(
			"topology generation %d does not advance current generation %d",
			topology.Generation,
			history.CurrentGeneration,
		)
	}

	history.Snapshots = append(history.Snapshots, cloneTopology(topology))
	history.CurrentGeneration = topology.Generation
	return history, nil
}

// TopologyRuntimeDigest returns the canonical identity a serving proof must echo for one topology.
func TopologyRuntimeDigest(topology MembershipTopology) string {
	memberships := normalizeMemberships(topology.Replicas)
	var input strings.Builder
	_, _ = fmt.Fprintf(&input, "generation=%d;", topology.Generation)
	for _, membership := range memberships {
		_, _ = fmt.Fprintf(&input, "replica=%s;", membership.ReplicaID)
		for _, member := range normalizeMemberIncarnations(membership.Members) {
			_, _ = fmt.Fprintf(&input, "native=%s;runtime=%s;", member.ID, member.RuntimeIncarnation)
		}
	}

	digest := sha256.Sum256([]byte(input.String()))
	return hex.EncodeToString(digest[:])
}

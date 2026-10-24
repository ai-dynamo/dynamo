/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"slices"
	"sort"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// reconcileEngineGroupDesiredAssignment owns the durable assignment, not its health projection.
// Recovery retains it; a cancelled, settled resize may return to the complete canonical base.
func reconcileEngineGroupDesiredAssignment(group *api.DynamoGraphDeploymentEngineGroup, profile api.EngineGroupProfileStatus, status domain.GroupStatus, desiredPlan *domain.ResolvedPlan) {
	desired := slices.Clone(group.Status.DesiredNativeMembers)
	if len(desired) == 0 {
		initial, found := status.Topologies.Current()
		if found {
			desired = engineGroupTopologyNativeMembers(initial)
		}
	}

	// A superseded or rejected plan cannot rewrite the new target's assignment.
	transition := status.Transition
	if desiredPlan != nil && transition != nil && desiredPlan.ID == transition.Spec.Plan.ID {
		desired = engineGroupPlanNativeMembers(desired, status, group.Spec.Replicas)
	} else if desiredPlan == nil && (transition == nil ||
		transition.Outcome == domain.TransitionOutcomeRolledBack ||
		transition.Outcome == domain.TransitionOutcomeCompleted) {
		current, found := status.Topologies.Current()
		expected := int64(group.Spec.Replicas) * int64(profile.NativeMembersPerReplica)
		if found && int64(len(desired)) != expected && current.ReplicaCount() == group.Spec.Replicas &&
			int64(current.NativeMemberCount()) == expected {
			desired = engineGroupTopologyNativeMembers(current)
		}
	}

	// Persist the exact assignment once; public counts and health remain projections of it.
	sort.Strings(desired)
	group.Status.DesiredNativeMembers = slices.Compact(desired)
	if int64(len(group.Status.DesiredNativeMembers)) == int64(group.Spec.Replicas)*int64(profile.NativeMembersPerReplica) {
		group.Status.DesiredAssignmentGeneration = group.Generation
	}
}

func engineGroupPlanNativeMembers(desired []string, status domain.GroupStatus, desiredReplicas int32) []string {
	// Derive the exact assignment from the immutable plan, never from an uncorrelated observation.
	desired = slices.Clone(desired)
	change := status.Transition.Spec.Plan.Change
	switch change.Kind {
	case domain.PlanKindGrow:
		for _, target := range change.Grow.Replicas {
			desired = append(desired, engineGroupNativeMembersToAPI(target.NativeMembers)...)
		}
	case domain.PlanKindRetire:
		base, found := status.Topologies.Snapshot(status.Transition.Spec.BaseTopologyGeneration)
		// Temporary retirement during repair preserves the slot's full desired assignment. A scale-down to the spec
		// target removes that assignment, including members already masked out of the base topology.
		if !found || base.ReplicaCount()-int32(len(change.Retire.Replicas)) != desiredReplicas {
			return desired
		}
		for _, record := range status.Registry.Replicas {
			if slices.Contains(change.Retire.Replicas, record.ReplicaID) {
				desired = slices.DeleteFunc(desired, func(id string) bool {
					return slices.Contains(record.DesiredNativeMembers, domain.NativeMemberID(id))
				})
			}
		}
	case domain.PlanKindRemap:
		desired = nil
		for _, replica := range change.Remap.Membership {
			desired = append(desired, engineGroupNativeMembersToAPI(replica.NativeMembers)...)
		}
	}
	sort.Strings(desired)
	return slices.Compact(desired)
}

func engineGroupTopologyNativeMembers(topology domain.MembershipTopology) []string {
	var members []string
	for _, replica := range topology.Replicas {
		for _, member := range replica.Members {
			members = append(members, string(member.ID))
		}
	}
	sort.Strings(members)
	return slices.Compact(members)
}

func projectEngineGroupReplicaStates(status domain.GroupStatus, previous []api.EngineGroupReplicaStatus) []api.EngineGroupReplicaStatus {
	replicas := make([]api.EngineGroupReplicaStatus, 0, len(status.Registry.Replicas))
	for _, record := range status.Registry.Replicas {
		projected := api.EngineGroupReplicaStatus{ReplicaID: string(record.ReplicaID), SlotID: string(record.SlotID)}
		members := make(map[string]struct{})
		for _, old := range previous {
			if old.ReplicaID == projected.ReplicaID {
				for _, member := range old.NativeMembers {
					members[member.ID] = struct{}{}
				}
			}
		}

		// Keep excluded identities visible so a partial allocation cannot become a smaller healthy replica.
		for _, snapshot := range status.Topologies.Snapshots {
			if membership, found := engineGroupTopologyMembership(snapshot, record.ReplicaID); found {
				for _, member := range membership.Members {
					members[string(member.ID)] = struct{}{}
				}
			}
		}
		for _, history := range record.History {
			for _, member := range history.NativeMembers {
				members[string(member)] = struct{}{}
			}
		}
		active, found := engineGroupTopologyMembership(status.Membership.Observed.CommittedTopology, record.ReplicaID)
		if found {
			for _, member := range active.Members {
				members[string(member.ID)] = struct{}{}
			}
		}
		if record.Current != nil {
			projected.CurrentAllocation = projectEngineGroupAllocation(*record.Current, status.Capacity.Observed)
		}

		// Correlate each traffic record with the committed runtime incarnation, not merely its rank ID.
		for id := range members {
			member := api.EngineGroupNativeMemberStatus{ID: id, Membership: api.EngineGroupReplicaMembershipUnknown, Traffic: api.EngineGroupMemberTrafficUnknown}
			if record.Current != nil {
				for _, process := range record.Current.Members {
					if string(process.ID) == id {
						member.RuntimeIncarnation = string(process.RuntimeIncarnation)
					}
				}
			}
			if status.Membership.Observed.CommittedTopology.Generation > 0 {
				member.Membership = api.EngineGroupReplicaMembershipMasked
				index := slices.IndexFunc(active.Members, func(member domain.NativeMemberIncarnation) bool {
					return member.ID == domain.NativeMemberID(id)
				})
				if found && index >= 0 {
					member.Membership = api.EngineGroupReplicaMembershipActive
					member.RuntimeIncarnation = string(active.Members[index].RuntimeIncarnation)
				}
			}
			if record.Current != nil {
				member.Traffic = projectEngineGroupMemberTraffic(record, domain.NativeMemberID(id), status.Traffic.Observed)
			}
			projected.NativeMembers = append(projected.NativeMembers, member)
		}
		applyEngineGroupTransitionIntent(&projected, status.Transition)
		sort.Slice(projected.NativeMembers, func(i, j int) bool { return projected.NativeMembers[i].ID < projected.NativeMembers[j].ID })
		replicas = append(replicas, projected)
	}
	return replicas
}

func projectEngineGroupAllocation(incarnation domain.ReplicaIncarnation, capacity domain.CapacityObservation) *api.EngineGroupReplicaAllocationStatus {
	allocation := &api.EngineGroupReplicaAllocationStatus{
		CapacityRefs: engineGroupCapacityRefsToAPI(incarnation.CapacityRefs),
		Availability: api.EngineGroupReplicaAvailabilityUnknown,
		Health:       api.EngineGroupAllocationHealthUnknown,
	}
	observed, found := engineGroupAllocationByReplica(capacity, incarnation.ReplicaID)
	if !found || !domain.SameIncarnation(observed.Incarnation, incarnation) {
		return allocation
	}
	allocation.Availability = api.EngineGroupReplicaAvailabilityUnavailable
	if observed.Available {
		allocation.Availability = api.EngineGroupReplicaAvailabilityAvailable
	}
	if observed.Health != "" {
		allocation.Health = api.EngineGroupAllocationHealth(observed.Health)
	}
	return allocation
}

func projectEngineGroupMemberTraffic(record domain.ReplicaRecord, id domain.NativeMemberID, traffic domain.TrafficObservation) api.EngineGroupMemberTraffic {
	states := []struct {
		members []domain.ReplicaMembership
		state   api.EngineGroupMemberTraffic
	}{
		{traffic.Draining, api.EngineGroupMemberTrafficDraining},
		{traffic.Drained, api.EngineGroupMemberTrafficDrained},
		{traffic.Admitted, api.EngineGroupMemberTrafficAdmitted},
	}
	for _, state := range states {
		for _, member := range state.members {
			if member.ReplicaID != record.ReplicaID {
				continue
			}
			for _, process := range member.Members {
				if process.ID == id && slices.Contains(record.Current.Members, process) {
					return state.state
				}
			}
		}
	}
	return api.EngineGroupMemberTrafficUnknown
}

func applyEngineGroupTransitionIntent(replica *api.EngineGroupReplicaStatus, transition *domain.TransitionStatus) {
	if transition == nil || transition.Outcome != domain.TransitionOutcomeProgressing {
		return
	}
	change := transition.Spec.Plan.Change
	var targets []domain.ReplicaTarget
	switch change.Kind {
	case domain.PlanKindGrow:
		targets = change.Grow.Replicas
	case domain.PlanKindRestore:
		for _, target := range change.Restore.Replicas {
			targets = append(targets, target.ReplicaTarget)
		}
	case domain.PlanKindRetire:
		if slices.Contains(change.Retire.Replicas, domain.ReplicaID(replica.ReplicaID)) {
			for i := range replica.NativeMembers {
				replica.NativeMembers[i].Membership = api.EngineGroupReplicaMembershipRetiring
			}
		}
	}

	// Preserve committed members during post-commit verification; only missing members remain joining.
	for _, target := range targets {
		if string(target.ReplicaID) != replica.ReplicaID {
			continue
		}
		for _, id := range target.NativeMembers {
			index := slices.IndexFunc(replica.NativeMembers, func(member api.EngineGroupNativeMemberStatus) bool { return member.ID == string(id) })
			if index < 0 {
				replica.NativeMembers = append(replica.NativeMembers, api.EngineGroupNativeMemberStatus{ID: string(id), Membership: api.EngineGroupReplicaMembershipJoining, Traffic: api.EngineGroupMemberTrafficUnknown})
			} else if replica.NativeMembers[index].Membership != api.EngineGroupReplicaMembershipActive {
				replica.NativeMembers[index].Membership = api.EngineGroupReplicaMembershipJoining
			}
		}
	}
}

func countAvailableEngineGroupReplicas(replicas []api.EngineGroupReplicaStatus, width int32) int32 {
	var count int32
	for _, replica := range replicas {
		if replica.CurrentAllocation == nil || replica.CurrentAllocation.Availability != api.EngineGroupReplicaAvailabilityAvailable {
			continue
		}
		available := true
		var active int32
		for _, member := range replica.NativeMembers {
			if member.Membership == api.EngineGroupReplicaMembershipActive {
				active++
				available = available && member.Traffic == api.EngineGroupMemberTrafficAdmitted
			}
		}
		if available && active == width {
			count++
		}
	}
	return count
}

func engineGroupHasDegradedAllocation(replicas []api.EngineGroupReplicaStatus) bool {
	for _, replica := range replicas {
		if replica.CurrentAllocation != nil && replica.CurrentAllocation.Health == api.EngineGroupAllocationHealthDegraded {
			return true
		}
	}
	return false
}

func engineGroupUnexpectedMembershipLoss(group *api.DynamoGraphDeploymentEngineGroup, status domain.GroupStatus) bool {
	if stable, found := status.Topologies.Snapshot(group.Status.LastStableTopologyGeneration); found {
		current := engineGroupTopologyNativeMembers(status.Membership.Observed.CommittedTopology)
		for _, id := range engineGroupTopologyNativeMembers(stable) {
			if !slices.Contains(current, id) {
				return true
			}
		}
		return false
	}
	return group.Status.LastStableReplicas > 0 && int64(group.Status.ActiveNativeMemberCount) < int64(group.Status.LastStableReplicas)*int64(group.Status.Profile.NativeMembersPerReplica)
}

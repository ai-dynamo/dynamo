/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"slices"
	"sort"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// resolveEngineGroupDesiredNativeMembers preserves assignments across survivor observations.
// Normal scaling/remapping changes assignment; recovery never replaces it with surviving membership.
func resolveEngineGroupDesiredNativeMembers(previous []string, status enginegroup.GroupStatus) []string {
	desired := slices.Clone(previous)
	if len(desired) == 0 {
		initial, found := status.Topologies.Current()
		if found {
			desired = engineGroupTopologyNativeMembers(initial)
		}
	}
	if status.Transition == nil || status.Transition.Outcome == enginegroup.TransitionOutcomeRolledBack {
		sort.Strings(desired)
		return slices.Compact(desired)
	}

	// Derive the exact assignment from the immutable plan, never from an uncorrelated observation.
	change := status.Transition.Spec.Plan.Change
	switch change.Kind {
	case enginegroup.PlanKindGrow:
		for _, target := range change.Grow.Replicas {
			desired = append(desired, engineGroupNativeMembersToAPI(target.NativeMembers)...)
		}
	case enginegroup.PlanKindRetire:
		base, found := status.Topologies.Snapshot(status.Transition.Spec.BaseTopologyGeneration)
		if found {
			for _, replica := range base.Replicas {
				if slices.Contains(change.Retire.Replicas, replica.ReplicaID) {
					for _, member := range replica.NativeMembers {
						desired = slices.DeleteFunc(desired, func(id string) bool { return id == string(member) })
					}
				}
			}
		}
	case enginegroup.PlanKindRemap:
		desired = nil
		for _, replica := range change.Remap.Membership {
			desired = append(desired, engineGroupNativeMembersToAPI(replica.NativeMembers)...)
		}
	}
	sort.Strings(desired)
	return slices.Compact(desired)
}

func engineGroupTopologyNativeMembers(topology enginegroup.MembershipTopology) []string {
	var members []string
	for _, replica := range topology.Replicas {
		members = append(members, engineGroupNativeMembersToAPI(replica.NativeMembers)...)
	}
	sort.Strings(members)
	return slices.Compact(members)
}

func projectEngineGroupReplicaStates(status enginegroup.GroupStatus, previous []api.EngineGroupReplicaStatus) []api.EngineGroupReplicaStatus {
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
				for _, member := range membership.NativeMembers {
					members[string(member)] = struct{}{}
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
			for _, member := range active.NativeMembers {
				members[string(member)] = struct{}{}
			}
		}
		if record.Current != nil {
			projected.CurrentAllocation = projectEngineGroupAllocation(*record.Current, status.Capacity.Observed)
		}

		// Correlate each traffic record with the committed runtime incarnation, not merely its rank ID.
		for id := range members {
			member := api.EngineGroupNativeMemberStatus{ID: id, Membership: api.EngineGroupReplicaMembershipUnknown, Traffic: api.EngineGroupMemberTrafficUnknown}
			if status.Membership.Observed.CommittedTopology.Generation > 0 {
				member.Membership = api.EngineGroupReplicaMembershipMasked
				if found && slices.Contains(active.NativeMembers, enginegroup.NativeMemberID(id)) {
					member.Membership = api.EngineGroupReplicaMembershipActive
				}
			}
			if record.Current != nil {
				member.Traffic = projectEngineGroupMemberTraffic(record, enginegroup.NativeMemberID(id), status.Traffic.Observed)
			}
			projected.NativeMembers = append(projected.NativeMembers, member)
		}
		applyEngineGroupTransitionIntent(&projected, status.Transition)
		sort.Slice(projected.NativeMembers, func(i, j int) bool { return projected.NativeMembers[i].ID < projected.NativeMembers[j].ID })
		replicas = append(replicas, projected)
	}
	return replicas
}

func projectEngineGroupAllocation(incarnation enginegroup.ReplicaIncarnation, capacity enginegroup.CapacityObservation) *api.EngineGroupReplicaAllocationStatus {
	allocation := &api.EngineGroupReplicaAllocationStatus{
		RuntimeIncarnation: string(incarnation.RuntimeIncarnation),
		CapacityRefs:       engineGroupIncarnationToAPI(incarnation).CapacityRefs,
		Availability:       api.EngineGroupReplicaAvailabilityUnknown,
		Health:             api.EngineGroupAllocationHealthUnknown,
	}
	observed, found := engineGroupAllocationByReplica(capacity, incarnation.ReplicaID)
	if !found || !enginegroup.SameIncarnation(observed.Incarnation, incarnation) {
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

func projectEngineGroupMemberTraffic(record enginegroup.ReplicaRecord, id enginegroup.NativeMemberID, traffic enginegroup.TrafficObservation) api.EngineGroupMemberTraffic {
	states := []struct {
		members []enginegroup.ReplicaMembership
		state   api.EngineGroupMemberTraffic
	}{
		{traffic.Draining, api.EngineGroupMemberTrafficDraining},
		{traffic.Drained, api.EngineGroupMemberTrafficDrained},
		{traffic.Admitted, api.EngineGroupMemberTrafficAdmitted},
	}
	for _, state := range states {
		for _, member := range state.members {
			if member.ReplicaID == record.ReplicaID && member.RuntimeIncarnation == record.Current.RuntimeIncarnation && slices.Contains(member.NativeMembers, id) {
				return state.state
			}
		}
	}
	return api.EngineGroupMemberTrafficUnknown
}

func applyEngineGroupTransitionIntent(replica *api.EngineGroupReplicaStatus, transition *enginegroup.TransitionStatus) {
	if transition == nil || transition.Outcome != enginegroup.TransitionOutcomeProgressing {
		return
	}
	change := transition.Spec.Plan.Change
	var targets []enginegroup.ReplicaTarget
	switch change.Kind {
	case enginegroup.PlanKindGrow:
		targets = change.Grow.Replicas
	case enginegroup.PlanKindRestore:
		for _, target := range change.Restore.Replicas {
			targets = append(targets, target.ReplicaTarget)
		}
	case enginegroup.PlanKindRetire:
		if slices.Contains(change.Retire.Replicas, enginegroup.ReplicaID(replica.ReplicaID)) {
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

func engineGroupUnexpectedMembershipLoss(group *api.DynamoGraphDeploymentEngineGroup, status enginegroup.GroupStatus) bool {
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

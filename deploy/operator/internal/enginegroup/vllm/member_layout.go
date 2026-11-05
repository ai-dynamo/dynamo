/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// ReplicaLayout maps complete allocation slots to their immutable native-member assignments.
// Committed membership may contain fewer members after recovery; it never changes this layout.
type ReplicaLayout struct {
	NativeMembersPerReplica int32
}

func appendMemberSubset(values []enginegroup.ReplicaMembership, subset enginegroup.ReplicaMembership) []enginegroup.ReplicaMembership {
	// Native observations are per process; coordinator observations remain grouped by allocation.
	for i := range values {
		if values[i].ReplicaID == subset.ReplicaID {
			for _, member := range subset.Members {
				if !slices.Contains(values[i].Members, member) {
					values[i].Members = append(values[i].Members, member)
				}
			}
			return values
		}
	}
	subset.Members = slices.Clone(subset.Members)
	return append(values, subset)
}

func containsMemberSubset(values []enginegroup.ReplicaMembership, subset enginegroup.ReplicaMembership) bool {
	// Callers pass one exact process incarnation, never an allocation-wide health assertion.
	return slices.ContainsFunc(values, func(value enginegroup.ReplicaMembership) bool {
		return value.ReplicaID == subset.ReplicaID && slices.Contains(value.Members, subset.Members[0])
	})
}

// NativeSize converts an allocation target into a native DP target without changing scale units.
func (l ReplicaLayout) NativeSize(replicas int32) (int32, error) {
	// Reject invalid geometry and overflow before generating native IDs or sending a target.
	if l.NativeMembersPerReplica <= 0 || replicas < 0 || int64(replicas)*int64(l.NativeMembersPerReplica) > math.MaxInt32 {
		return 0, fmt.Errorf("invalid allocation count %d or native member width %d", replicas, l.NativeMembersPerReplica)
	}
	return replicas * l.NativeMembersPerReplica, nil
}

// Members returns the fixed rank range for one allocation, not its current serving members.
func (l ReplicaLayout) Members(index int32) ([]enginegroup.NativeMemberID, error) {
	// The end of the range must be representable before allocating its member list.
	if index < 0 || index == math.MaxInt32 {
		return nil, fmt.Errorf("invalid allocation index %d", index)
	}
	end, err := l.NativeSize(index + 1)
	if err != nil {
		return nil, err
	}
	members := make([]enginegroup.NativeMemberID, l.NativeMembersPerReplica)
	for offset := range members {
		members[offset] = enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", end-l.NativeMembersPerReplica+int32(offset)))
	}
	return members, nil
}

// SlotIndex accepts only a canonical physical allocation slot.
func SlotIndex(slot enginegroup.CapacitySlotID) (int32, error) {
	literal := strings.TrimPrefix(string(slot), "slot-")
	index, err := strconv.ParseInt(literal, 10, 32)
	if err != nil || index < 0 || string(slot) != "slot-"+strconv.FormatInt(index, 10) {
		return 0, fmt.Errorf("invalid allocation slot %q", slot)
	}
	return int32(index), nil
}

// MembersForSlot binds canonical physical slot parsing to the checked native rank range.
func (l ReplicaLayout) MembersForSlot(slot enginegroup.CapacitySlotID) ([]enginegroup.NativeMemberID, error) {
	index, err := SlotIndex(slot)
	if err != nil {
		return nil, err
	}
	return l.Members(index)
}

// Target derives one complete joining allocation using the fixed native rank layout.
func (l ReplicaLayout) Target(index int32) (enginegroup.ReplicaTarget, error) {
	members, err := l.Members(index)
	if err != nil {
		return enginegroup.ReplicaTarget{}, err
	}
	return enginegroup.ReplicaTarget{ReplicaID: enginegroup.ReplicaID(fmt.Sprintf("replica-%d", index)),
		SlotID:    enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", index)),
		Bootstrap: enginegroup.BootstrapModeJoin, NativeMembers: members}, nil
}

// MatchesMembers checks an exact rank set independently of slice ordering.
func (l ReplicaLayout) MatchesMembers(index int32, members []enginegroup.NativeMemberIncarnation) bool {
	if index < 0 || index == math.MaxInt32 || int64(len(members)) != int64(l.NativeMembersPerReplica) {
		return false
	}
	end, err := l.NativeSize(index + 1)
	if err != nil {
		return false
	}
	first := end - l.NativeMembersPerReplica
	seen := make(map[int32]bool, len(members))
	for _, member := range members {
		rank, err := NativeRank(member.ID)
		if err != nil || rank < first || rank >= end || seen[rank] {
			return false
		}
		seen[rank] = true
	}
	return true
}

// ValidateComplete applies only to healthy planned resizing, never survivor observation.
func (l ReplicaLayout) ValidateComplete(topology enginegroup.MembershipTopology) error {
	if len(topology.Replicas) == 0 || int64(len(topology.Replicas)) > math.MaxInt32 {
		return fmt.Errorf("planned resizing requires a nonempty complete allocation layout")
	}
	if _, err := l.NativeSize(int32(len(topology.Replicas))); err != nil {
		return err
	}
	// Index allocations without mutating or relying on the observer's slice order.
	byReplica := make(map[enginegroup.ReplicaID]enginegroup.ReplicaMembership, len(topology.Replicas))
	for _, replica := range topology.Replicas {
		if _, duplicate := byReplica[replica.ReplicaID]; duplicate {
			return fmt.Errorf("allocation %q appears more than once", replica.ReplicaID)
		}
		byReplica[replica.ReplicaID] = replica
	}
	// Every logical allocation must retain its own exact native rank range.
	for index := range int32(len(topology.Replicas)) {
		id := enginegroup.ReplicaID(fmt.Sprintf("replica-%d", index))
		replica, found := byReplica[id]
		if !found || !l.MatchesMembers(index, replica.Members) {
			return fmt.Errorf("planned resizing requires the complete fixed member range of %s", id)
		}
	}
	return nil
}

// NativeRank parses one canonical DP rank identifier without accepting aliases.
func NativeRank(id enginegroup.NativeMemberID) (int32, error) {
	rank, err := strconv.ParseInt(strings.TrimPrefix(string(id), "dp-"), 10, 32)
	if err != nil || rank < 0 || string(id) != fmt.Sprintf("dp-%d", rank) {
		return 0, fmt.Errorf("invalid native DP member %q", id)
	}
	return int32(rank), nil
}

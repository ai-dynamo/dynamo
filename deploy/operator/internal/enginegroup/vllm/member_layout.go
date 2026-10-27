/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"fmt"
	"math"
	"slices"

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

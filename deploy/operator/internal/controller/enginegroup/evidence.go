/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"

// engineGroupHasDegradedAllocation inspects exact current allocations, not status projections or unrelated joiners.
func engineGroupHasDegradedAllocation(state domain.GroupStatus) bool {
	for _, record := range state.Registry.Replicas {
		if record.Current == nil {
			continue
		}
		allocation, found := engineGroupAllocationByReplica(state.Capacity.Observed, record.ReplicaID)
		if found && domain.SameIncarnation(allocation.Incarnation, *record.Current) &&
			allocation.Health == domain.AllocationHealthDegraded {
			return true
		}
	}
	return false
}

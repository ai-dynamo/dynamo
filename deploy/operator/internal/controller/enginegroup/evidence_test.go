//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
)

func TestEngineGroupDegradationUsesExactCurrentAllocations(t *testing.T) {
	for _, tc := range []struct {
		name            string
		current         bool
		observedReplica domain.ReplicaID
		observedUID     domain.PodUID
		health          domain.AllocationHealth
		want            bool
	}{
		{"healthy current allocation", true, "replica-0", "uid-0", domain.AllocationHealthHealthy, false},
		{"degraded current allocation", true, "replica-0", "uid-0", domain.AllocationHealthDegraded, true},
		{"unbound allocation", false, "replica-0", "uid-0", domain.AllocationHealthDegraded, false},
		{"unrelated joining allocation", true, "replica-1", "uid-1", domain.AllocationHealthDegraded, false},
		{"replacement cannot describe the old incarnation", true, "replica-0", "replacement", domain.AllocationHealthDegraded, false},
		{"unknown health is not degradation evidence", true, "replica-0", "uid-0", domain.AllocationHealthUnknown, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("keep the canonical current incarnation distinct from physical observations")
			current := domain.ReplicaIncarnation{ReplicaID: "replica-0", SlotID: "slot-0",
				CapacityRefs: []domain.CapacityRef{{Name: "pod", UID: "uid-0"}}}
			record := domain.ReplicaRecord{ReplicaID: current.ReplicaID, SlotID: current.SlotID}
			if tc.current {
				record.Current = &current
			}
			observed := current
			observed.ReplicaID = tc.observedReplica
			observed.CapacityRefs = []domain.CapacityRef{{Name: "pod", UID: tc.observedUID}}
			state := domain.GroupStatus{Registry: domain.ReplicaRegistry{Replicas: []domain.ReplicaRecord{record}},
				Capacity: domain.CapacityStatus{Observed: domain.CapacityObservation{Allocations: []domain.CapacityAllocation{
					{Incarnation: observed, Health: tc.health},
				}}}}

			t.Log("derive degradation directly from matching domain evidence")
			assert.Equal(t, tc.want, engineGroupHasDegradedAllocation(state))
		})
	}
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReplicaLayoutCanonicalSlotsAndBounds(t *testing.T) {
	for _, slot := range []enginegroup.CapacitySlotID{"slot--1", "slot-01", "slot-+1", "1", "slot-2147483648"} {
		t.Run(string(slot), func(t *testing.T) {
			t.Log("reject a noncanonical slot before deriving native rank identities")
			_, err := (ReplicaLayout{NativeMembersPerReplica: 4}).MembersForSlot(slot)
			require.Error(t, err)
		})
	}

	t.Log("bound both the total target and the end of an allocation's native range")
	_, err := (ReplicaLayout{NativeMembersPerReplica: 2}).NativeSize(math.MaxInt32)
	require.Error(t, err)
	_, err = (ReplicaLayout{NativeMembersPerReplica: 4}).Members(math.MaxInt32)
	require.Error(t, err)
	_, err = (ReplicaLayout{}).NativeSize(1)
	require.Error(t, err)
}

func TestReplicaLayoutValidatesCompletePlannedMembership(t *testing.T) {
	for _, width := range []int32{1, 4} {
		for _, scenario := range []string{"complete", "reordered", "missing member", "wrong range", "duplicate rank", "rank alias", "replica alias", "duplicate allocation"} {
			t.Run(strconv.Itoa(int(width))+"/"+scenario, func(t *testing.T) {
				t.Log("derive physical and native identities from one checked layout")
				layout := ReplicaLayout{NativeMembersPerReplica: width}
				topology := enginegroup.MembershipTopology{Generation: 1}
				for index := range int32(3) {
					target, err := layout.Target(index)
					require.NoError(t, err)
					assert.Equal(t, enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", index)), target.SlotID)
					members, err := layout.MembersForSlot(target.SlotID)
					require.NoError(t, err)
					assert.Equal(t, target.NativeMembers, members)
					replica := enginegroup.ReplicaMembership{ReplicaID: target.ReplicaID}
					for _, id := range members {
						replica.Members = append(replica.Members, enginegroup.NativeMemberIncarnation{ID: id, RuntimeIncarnation: "process"})
					}
					topology.Replicas = append(topology.Replicas, replica)
				}

				t.Log("distinguish complete rank ranges from equal counts with incompatible identities")
				switch scenario {
				case "reordered":
					topology.Replicas[0], topology.Replicas[2] = topology.Replicas[2], topology.Replicas[0]
				case "missing member":
					topology.Replicas[1].Members = topology.Replicas[1].Members[:width-1]
				case "wrong range":
					topology.Replicas[1].Members[0].ID = "dp-0"
				case "duplicate rank":
					topology.Replicas[1].Members = append(topology.Replicas[1].Members, topology.Replicas[1].Members[0])
				case "rank alias":
					topology.Replicas[0].Members[0].ID = "dp-00"
				case "replica alias":
					topology.Replicas[1].ReplicaID = "replica-01"
				case "duplicate allocation":
					topology.Replicas[1].ReplicaID = topology.Replicas[0].ReplicaID
				}
				err := layout.ValidateComplete(topology)
				if scenario == "complete" || scenario == "reordered" {
					require.NoError(t, err)
				} else {
					require.Error(t, err)
				}
			})
		}
	}
}

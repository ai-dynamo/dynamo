/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTrafficObservationTerminalEvidence(t *testing.T) {
	tests := []struct {
		name     string
		live     ReplicaMembership
		drained  ReplicaMembership
		draining bool
		want     string
	}{
		{
			name: "unfinished drain invalidates a terminal tombstone",
			live: engineReplica(0), drained: engineReplica(0), draining: true,
			want: "both draining and drained",
		},
		{
			name: "readmission invalidates a terminal tombstone",
			live: engineReplica(0), drained: engineReplica(0),
			want: "both admitted and drained",
		},
		{
			name: "a replacement process does not inherit the previous lifetime's drain",
			live: ReplicaMembership{ReplicaID: "replica-0", Members: []NativeMemberIncarnation{{
				ID: "dp-0", RuntimeIncarnation: "replacement-runtime",
			}}},
			drained: engineReplica(0),
		},
		{
			name: "a packed allocation can serve one member while another is drained",
			live: engineReplica(0),
			drained: ReplicaMembership{ReplicaID: "replica-0", Members: []NativeMemberIncarnation{{
				ID: "dp-1", RuntimeIncarnation: "other-runtime",
			}}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe exact process identities independently of the allocation's aggregate state")
			observation := TrafficObservation{Drained: []ReplicaMembership{test.drained}}
			if test.draining {
				observation.Draining = []ReplicaMembership{test.live}
			} else {
				observation.Admitted = []ReplicaMembership{test.live}
			}

			t.Log("Accept only terminal evidence that does not contradict the same process's current traffic state")
			err := validateTrafficObservation(observation)
			if test.want != "" {
				require.ErrorContains(t, err, test.want)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestTrafficReadmissionRequiresFreshDrainEvidence(t *testing.T) {
	t.Log("Gracefully drain a process and retain its exact terminal evidence")
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	target := TrafficTarget{
		ControlRevision: 1, TransitionID: "first-drain", TopologyGeneration: 1,
		Admitted: cloneReplicaMemberships(base.Replicas[:1]),
		Drain:    []TrafficDrainTarget{{Membership: engineReplica(1), Mode: TrafficDrainModeGraceful}},
	}
	result, err := scenario.traffic.Apply(t.Context(), scenario.groupID, target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.True(t, trafficTargetConverged(target, scenario.traffic.observation))

	t.Log("Readmitting the same incarnation invalidates its previous drain tombstone")
	target = TrafficTarget{
		ControlRevision: 2, TransitionID: "readmit", TopologyGeneration: 1,
		Admitted: cloneReplicaMemberships(base.Replicas),
	}
	result, err = scenario.traffic.Apply(t.Context(), scenario.groupID, target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.Empty(t, scenario.traffic.observation.Drained)
	require.NoError(t, validateTrafficObservation(scenario.traffic.observation))

	t.Log("A new graceful drain must wait for new in-flight work instead of reusing the previous tombstone")
	scenario.traffic.autoDrain = false
	target = TrafficTarget{
		ControlRevision: 3, TransitionID: "second-drain", TopologyGeneration: 1,
		Admitted: cloneReplicaMemberships(base.Replicas[:1]),
		Drain:    []TrafficDrainTarget{{Membership: engineReplica(1), Mode: TrafficDrainModeGraceful}},
	}
	result, err = scenario.traffic.Apply(t.Context(), scenario.groupID, target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.Empty(t, scenario.traffic.observation.Drained)
	require.True(t, sameMemberships(base.Replicas[1:], scenario.traffic.observation.Draining))
	require.False(t, trafficTargetConverged(target, scenario.traffic.observation))
}

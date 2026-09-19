/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import "testing"

func TestTerminalDeletionEvidenceRequiresEmptyAcceptedAndObservedLevels(t *testing.T) {
	status, capacity, traffic, membership := terminalDeletionEvidence(t)
	member := engineReplica(0)

	tests := []struct {
		name   string
		mutate func(*GroupStatus, *CapacityObservation, *TrafficObservation, *MembershipObservation)
	}{
		{
			name: "admitted traffic remains",
			mutate: func(_ *GroupStatus, _ *CapacityObservation, traffic *TrafficObservation, _ *MembershipObservation) {
				traffic.Admitted = []ReplicaMembership{member}
			},
		},
		{
			name: "traffic is still draining",
			mutate: func(_ *GroupStatus, _ *CapacityObservation, traffic *TrafficObservation, _ *MembershipObservation) {
				traffic.Draining = []ReplicaMembership{member}
			},
		},
		{
			name: "accepted capacity can still recreate a replica",
			mutate: func(status *GroupStatus, _ *CapacityObservation, _ *TrafficObservation, _ *MembershipObservation) {
				incarnation := replicaIncarnation(0)
				status.Capacity.Desired.Replicas = []CapacityReplicaTarget{{
					ReplicaID:   incarnation.ReplicaID,
					SlotID:      incarnation.SlotID,
					Incarnation: &incarnation,
				}}
				status.Capacity.Accepted = cloneCapacityTarget(status.Capacity.Desired)
			},
		},
		{
			name: "accepted traffic can still readmit a replica",
			mutate: func(status *GroupStatus, _ *CapacityObservation, _ *TrafficObservation, _ *MembershipObservation) {
				status.Traffic.Desired.Admitted = []ReplicaMembership{member}
				status.Traffic.Desired.Drain = nil
				status.Traffic.Accepted = cloneTrafficTarget(status.Traffic.Desired)
			},
		},
		{
			name: "physical capacity remains",
			mutate: func(_ *GroupStatus, capacity *CapacityObservation, _ *TrafficObservation, _ *MembershipObservation) {
				capacity.Allocations = []CapacityAllocation{{Incarnation: replicaIncarnation(0), Available: true}}
			},
		},
		{
			name: "committed membership remains",
			mutate: func(_ *GroupStatus, _ *CapacityObservation, _ *TrafficObservation, membership *MembershipObservation) {
				membership.CommittedTopology.Replicas = []ReplicaMembership{member}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidateStatus := cloneStatus(status)
			candidateCapacity := cloneCapacityObservation(capacity)
			candidateTraffic := cloneTrafficObservation(traffic)
			candidateMembership := cloneMembershipObservation(membership)
			test.mutate(&candidateStatus, &candidateCapacity, &candidateTraffic, &candidateMembership)

			t.Log("evaluate deletion against one unsafe current or accepted level")
			ready, err := TerminalDeletionEvidenceReady(
				candidateStatus,
				candidateCapacity,
				candidateTraffic,
				candidateMembership,
			)
			if err != nil {
				t.Fatalf("validate deletion evidence: %v", err)
			}
			if ready {
				t.Fatal("unsafe deletion evidence was accepted")
			}
		})
	}
}

func TestTerminalDeletionEvidenceAllowsDurableDrainedTombstones(t *testing.T) {
	status, capacity, traffic, membership := terminalDeletionEvidence(t)

	t.Log("prove the last member is absent while retaining its durable drained tombstone")
	ready, err := TerminalDeletionEvidenceReady(status, capacity, traffic, membership)
	if err != nil {
		t.Fatalf("validate deletion evidence: %v", err)
	}
	if !ready {
		t.Fatal("terminal empty evidence with a drained tombstone should allow deletion")
	}
}

func TestTerminalDeletionEvidenceBlocksUnfinishedOrInvalidMembershipAuthority(t *testing.T) {
	status, capacity, traffic, membership := pendingTerminalDeletionEvidence(t)

	tests := []struct {
		name      string
		phase     MembershipTransitionPhase
		corrupt   bool
		wantError bool
	}{
		{name: "pending membership mutation", phase: MembershipTransitionPhasePending},
		{name: "unknown membership mutation", phase: MembershipTransitionPhaseUnknown},
		{name: "invalid transition correlation", phase: MembershipTransitionPhasePending, corrupt: true, wantError: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidateStatus := cloneStatus(status)
			candidateMembership := cloneMembershipObservation(membership)
			candidateStatus.Membership.Observed.Transition.Phase = test.phase
			candidateMembership.Transition.Phase = test.phase
			if test.corrupt {
				candidateMembership.Transition.TargetDigest = "unrelated-target"
			}

			t.Log("evaluate deletion while membership authority is unfinished or uncorrelated")
			ready, err := TerminalDeletionEvidenceReady(
				candidateStatus,
				capacity,
				traffic,
				candidateMembership,
			)
			if test.wantError {
				if err == nil {
					t.Fatal("invalid membership correlation should fail validation")
				}
				return
			}
			if err != nil {
				t.Fatalf("validate deletion evidence: %v", err)
			}
			if ready {
				t.Fatal("unfinished membership authority should block deletion")
			}
		})
	}
}

func terminalDeletionEvidence(t *testing.T) (
	GroupStatus,
	CapacityObservation,
	TrafficObservation,
	MembershipObservation,
) {
	t.Helper()
	base := engineTopology(1, 1)
	status, err := NewGroupStatus(base, capacityForTopology(base), TrafficObservation{Admitted: base.Replicas})
	if err != nil {
		t.Fatalf("construct initial status: %v", err)
	}
	member := cloneReplicaMembership(base.Replicas[0])
	status.ControlRevision = 2
	status.Capacity.Desired = &CapacityTarget{
		ControlRevision:       1,
		TransitionID:          "terminal-retirement",
		ProfileFingerprint:    "profile-v1",
		ProcessLifecycleOwner: ProcessLifecycleOwnerOrchestrator,
	}
	status.Capacity.Accepted = cloneCapacityTarget(status.Capacity.Desired)
	status.Capacity.Observed = CapacityObservation{AppliedRevision: 1}
	status.Traffic.Desired = &TrafficTarget{
		ControlRevision:    2,
		TransitionID:       "terminal-retirement",
		TopologyGeneration: 1,
		Drain: []TrafficDrainTarget{{
			Membership: member,
			Mode:       TrafficDrainModeGraceful,
		}},
	}
	status.Traffic.Accepted = cloneTrafficTarget(status.Traffic.Desired)
	status.Traffic.Observed = TrafficObservation{AppliedRevision: 2, Drained: []ReplicaMembership{member}}

	capacity := cloneCapacityObservation(status.Capacity.Observed)
	traffic := cloneTrafficObservation(status.Traffic.Observed)
	membership := MembershipObservation{CommittedTopology: MembershipTopology{Generation: 2}}
	return status, capacity, traffic, membership
}

func pendingTerminalDeletionEvidence(t *testing.T) (
	GroupStatus,
	CapacityObservation,
	TrafficObservation,
	MembershipObservation,
) {
	t.Helper()
	scenario := newCoordinatorScenario(t, engineTopology(1, 1))
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("pending-deletion", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("observe pending membership transition", func(s *coordinatorScenario) bool {
		return s.status.Membership.Observed.Transition != nil &&
			s.status.Membership.Observed.Transition.Phase == MembershipTransitionPhasePending
	})

	status := cloneStatus(scenario.status)
	base := status.Membership.Observed.CommittedTopology
	status.ControlRevision++
	capacityRevision := status.ControlRevision
	status.Capacity.Desired = &CapacityTarget{
		ControlRevision:       capacityRevision,
		TransitionID:          status.Transition.Spec.ID,
		ProfileFingerprint:    status.Transition.Spec.Plan.ProfileFingerprint,
		ProcessLifecycleOwner: status.Transition.Spec.Plan.ProcessLifecycleOwner,
	}
	status.Capacity.Accepted = cloneCapacityTarget(status.Capacity.Desired)
	status.Capacity.Observed = CapacityObservation{AppliedRevision: capacityRevision}
	status.ControlRevision++
	trafficRevision := status.ControlRevision
	status.Traffic.Desired = &TrafficTarget{
		ControlRevision:    trafficRevision,
		TransitionID:       status.Transition.Spec.ID,
		TopologyGeneration: base.Generation,
	}
	for _, member := range base.Replicas {
		status.Traffic.Desired.Drain = append(status.Traffic.Desired.Drain, TrafficDrainTarget{
			Membership: member,
			Mode:       TrafficDrainModeGraceful,
		})
	}
	status.Traffic.Accepted = cloneTrafficTarget(status.Traffic.Desired)
	status.Traffic.Observed = TrafficObservation{
		AppliedRevision: trafficRevision,
		Drained:         cloneReplicaMemberships(base.Replicas),
	}
	capacity := cloneCapacityObservation(status.Capacity.Observed)
	traffic := cloneTrafficObservation(status.Traffic.Observed)
	membership := cloneMembershipObservation(status.Membership.Observed)
	membership.CommittedTopology = MembershipTopology{Generation: base.Generation + 1}
	return status, capacity, traffic, membership
}

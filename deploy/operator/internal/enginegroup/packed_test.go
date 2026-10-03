/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"testing"
)

func packedTopology() MembershipTopology {
	topology := engineTopology(1, 2)
	for allocation := range topology.Replicas {
		members := make([]NativeMemberIncarnation, 0, 4)
		for rank := allocation * 4; rank < allocation*4+4; rank++ {
			members = append(members, NativeMemberIncarnation{ID: NativeMemberID(fmt.Sprintf("dp-%d", rank)), RuntimeIncarnation: RuntimeIncarnationID(fmt.Sprintf("process-%d-v1", rank))})
		}
		topology.Replicas[allocation].Members = members
	}
	return topology
}

func TestCoordinatorPackedRecoveryEP8ToEP7ToEP4ToEP8(t *testing.T) {
	t.Log("Start two four-member allocations and freeze the desired eight-member assignment")
	base := packedTopology()
	scenario := newCoordinatorScenario(t, base)
	scenario.capacity.observation.Allocations[1].Health = AllocationHealthDegraded
	failed := ReplicaMembership{ReplicaID: "replica-1", Members: base.Replicas[1].Members[1:2]}
	scenario.membership.unavailable = []ReplicaMembership{failed}
	plan := retirePlan("mask-dp-5", "replica-1", TrafficRequirementKeepServing, VerificationRequirementRequired)
	plan.Change = ResolvedChange{Kind: PlanKindReduceToSurvivors, ReduceToSurvivors: &ReduceToSurvivorsChange{Survivors: []ReplicaNativeMembership{
		{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []NativeMemberID{"dp-0", "dp-1", "dp-2", "dp-3"}},
		{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []NativeMemberID{"dp-4", "dp-6", "dp-7"}},
	}}}
	scenario.desired = &plan
	scenario.runUntil("apply one-member survivor reduction", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
	if len(scenario.capacity.observation.Allocations) != 2 || !containsMembership(scenario.traffic.observation.Drained, failed) {
		t.Fatal("partial recovery released capacity or skipped failed-member withdrawal")
	}

	t.Log("Restart before observing the correlated EP7 commit, retaining the same Pod UIDs")
	scenario.rebuildCoordinator()
	ep7 := cloneTopology(base)
	ep7.Generation = 2
	ep7.Replicas = subtractMemberships(base.Replicas, []ReplicaMembership{failed})
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, ep7)
	scenario.runUntil("complete EP7 recovery", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
	if ep7.NativeMemberCount() != 7 || len(scenario.capacity.observation.Allocations) != 2 {
		t.Fatal("EP7 was not represented independently of allocation count")
	}
	record, _ := scenario.status.Registry.Find("replica-1")
	if !slices.Equal(record.DesiredNativeMembers, []NativeMemberID{"dp-4", "dp-5", "dp-6", "dp-7"}) {
		t.Fatalf("failure rewrote desired assignment: %v", record.DesiredNativeMembers)
	}

	t.Log("Hold EP7 across a restart: the exited process is absent from serving membership, not from allocation identity")
	scenario.rebuildCoordinator()
	capacityCalls := scenario.capacity.applyCalls
	for range 3 {
		scenario.mustReconcile("hold the survivor topology without restarting its exited neighbour")
	}
	if scenario.capacity.applyCalls != capacityCalls || !containsMembership(scenario.traffic.observation.Drained, failed) {
		t.Fatal("steady EP7 recreated a process or dropped the held allocation's withdrawal evidence")
	}

	t.Log("Refuse a non-conforming capacity observation that forgets the exited process's last incarnation")
	retained := cloneCapacityObservation(scenario.capacity.observation)
	scenario.capacity.observation.Allocations[1].Incarnation.Members = slices.DeleteFunc(
		slices.Clone(scenario.capacity.observation.Allocations[1].Incarnation.Members),
		func(member NativeMemberIncarnation) bool { return member.ID == "dp-5" },
	)
	if err := scenario.reconcile("fail closed instead of inventing an identity for the disappeared process"); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("missing frozen process identity was silently accepted: %v", err)
	}
	scenario.capacity.observation = retained
	scenario.mustReconcile("resume EP7 with retained allocation identity metadata")

	t.Log("Gracefully retire the damaged allocation's three survivors, keeping the previous failed-member tombstone")
	retirement := retirePlan("retire-damaged-pod", "replica-1", TrafficRequirementKeepServing, VerificationRequirementRequired)
	scenario.desired = &retirement
	scenario.runUntil("apply EP4 retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 2 })
	if !containsMembership(scenario.traffic.observation.Drained, failed) {
		t.Fatal("the newer retirement target dropped dp-5's terminal withdrawal evidence")
	}
	ep4 := MembershipTopology{Generation: 3, Replicas: cloneReplicaMemberships(base.Replicas[:1])}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, ep4)
	scenario.runUntil("complete exact damaged-Pod release", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
	if len(scenario.capacity.observation.Allocations) != 1 || len(scenario.capacity.lastTarget.ReleaseFences) != 1 ||
		scenario.capacity.lastTarget.ReleaseFences[0].CapacityRefs[0].UID != "pod-uid-1-v1" {
		t.Fatal("retirement did not authorize only the damaged Pod UID")
	}

	t.Log("Create a new allocation in the same slot with four new process incarnations and restore the original assignment")
	replacement := physicalIncarnationForMembership(base.Replicas[1])
	replacement.CapacityRefs[0].UID = "pod-uid-1-v2"
	for i := range replacement.Members {
		replacement.Members[i].RuntimeIncarnation += "-replacement"
	}
	scenario.capacity.planned["replica-1"] = replacement
	restore := retirePlan("restore-packed-slot", "replica-1", TrafficRequirementKeepServing, VerificationRequirementRequired)
	restore.Change = ResolvedChange{Kind: PlanKindRestore, Restore: &RestoreChange{Replicas: []RestorationTarget{{ReplicaTarget: ReplicaTarget{
		ReplicaID: "replica-1", SlotID: "slot-1", Bootstrap: BootstrapModeRestoreFixedSlot,
		NativeMembers: []NativeMemberID{"dp-4", "dp-5", "dp-6", "dp-7"},
	}}}}}
	incomplete := cloneResolvedPlan(restore)
	incomplete.Change.Restore.Replicas[0].NativeMembers = []NativeMemberID{"dp-4", "dp-6", "dp-7"}
	if _, err := validateResolvedPlan(ep4, scenario.status.Registry, incomplete); err == nil {
		t.Fatal("restoration accepted the reduced historical set instead of the full desired assignment")
	}
	scenario.desired = &restore
	scenario.runUntil("apply EP8 restoration", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 3 })
	scenario.rebuildCoordinator()
	restored := cloneTopology(base)
	restored.Generation = 4
	restored.Replicas[1].Members = slices.Clone(replacement.Members)
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, restored)
	scenario.runUntil("verify and admit restored EP8", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
	if !sameMemberships(scenario.traffic.observation.Admitted, restored.Replicas) || restored.NativeMemberCount() != 8 {
		t.Fatal("restoration did not admit exactly the verified eight-member topology")
	}
}

func TestCoordinatorRejectedRecoveryNeverReadmitsFailedMember(t *testing.T) {
	t.Log("Withdraw dp-5 while retaining its three healthy neighbours")
	base := packedTopology()
	scenario := newCoordinatorScenario(t, base)
	failed := ReplicaMembership{ReplicaID: "replica-1", Members: base.Replicas[1].Members[1:2]}
	scenario.membership.unavailable = []ReplicaMembership{failed}
	plan := retirePlan("rejected-survivors", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
	plan.Change = ResolvedChange{Kind: PlanKindReduceToSurvivors, ReduceToSurvivors: &ReduceToSurvivorsChange{Survivors: []ReplicaNativeMembership{
		{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []NativeMemberID{"dp-0", "dp-1", "dp-2", "dp-3"}},
		{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []NativeMemberID{"dp-4", "dp-6", "dp-7"}},
	}}}
	scenario.desired = &plan
	scenario.runUntil("submit recovery", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })

	t.Log("Definitive rejection rolls back preparation without reviving the failed process")
	id := scenario.status.Membership.Desired.TransitionID
	observation := scenario.membership.transitions[id]
	observation.Phase = MembershipTransitionPhaseRejected
	observation.Failure = &Failure{Classification: FailureClassificationTerminal, Reason: "RecoveryRejected"}
	scenario.membership.transitions[id] = observation
	scenario.runUntil("finish safe rollback", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeRolledBack })
	if containsMembership(scenario.traffic.observation.Admitted, failed) ||
		!sameMemberships(scenario.traffic.observation.Admitted, subtractMemberships(base.Replicas, []ReplicaMembership{failed})) {
		t.Fatal("rollback admitted failed membership or lost healthy survivors")
	}
}

func TestCoordinatorInvalidReplacementPlanPreservesDurableTransition(t *testing.T) {
	t.Log("Complete a retirement and retain its authoritative history")
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	plan := retirePlan("first-plan", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("submit retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 1))
	scenario.runUntil("finish retirement", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
	before := cloneStatus(scenario.status)

	t.Log("Reject an invalid new plan without clearing the previous transition or compacting its history")
	invalid := retirePlan("invalid-plan", "missing-replica", TrafficRequirementKeepServing, VerificationRequirementNone)
	result, err := scenario.coordinator.Reconcile(context.Background(), scenario.groupID, &invalid, scenario.status)
	if err == nil || !reflect.DeepEqual(before, result.Status) {
		t.Fatalf("invalid replacement damaged durable state: %v", err)
	}
}

func TestInvocationErrorPreservesClassificationAndCause(t *testing.T) {
	for _, classification := range []FailureClassification{FailureClassificationRetryable, FailureClassificationTerminal} {
		t.Run(string(classification), func(t *testing.T) {
			t.Log("Classify an invocation failure without asserting a definitive membership rejection")
			err := &InvocationError{Failure: Failure{Classification: classification, Reason: "TransportUnavailable"}, Err: context.DeadlineExceeded}
			wrapped := fmt.Errorf("apply target: %w", err)
			var invocation *InvocationError
			if !errors.Is(wrapped, context.DeadlineExceeded) || !errors.As(wrapped, &invocation) || invocation.Failure.Classification != classification {
				t.Fatal("wrapping lost invocation evidence")
			}
		})
	}
}

func TestPackedReleaseRequiresDrainEvidenceForPreviouslyMaskedMember(t *testing.T) {
	t.Log("Keep all four process identities in the allocation after dp-5 is masked out of EP7")
	base := packedTopology()
	scenario := newCoordinatorScenario(t, base)
	failed := ReplicaMembership{ReplicaID: "replica-1", Members: base.Replicas[1].Members[1:2]}
	ep7 := cloneTopology(base)
	ep7.Generation = 2
	ep7.Replicas = subtractMemberships(base.Replicas, []ReplicaMembership{failed})
	ep4 := MembershipTopology{Generation: 3, Replicas: cloneReplicaMemberships(base.Replicas[:1])}
	plan := retirePlan("retire-packed", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
	resolution, err := validateResolvedPlan(ep7, scenario.status.Registry, plan)
	if err != nil {
		t.Fatal(err)
	}
	scenario.status.Transition = &TransitionStatus{Spec: TransitionSpec{ID: "retire-packed", Plan: plan}}
	scenario.status.Traffic.Observed.Drained = cloneReplicaMemberships(ep7.Replicas[1:])

	t.Log("Draining three surviving ranks cannot authorize release while the failed rank's terminal evidence is missing")
	ready, persist, err := scenario.coordinator.reconcileRetiredCapacity(context.Background(), scenario.groupID, &scenario.status, ep7, ep4, resolution)
	if err != nil || ready || persist || scenario.status.Capacity.Desired != nil {
		t.Fatalf("incomplete process evidence authorized Pod release: ready=%v, persist=%v, err=%v", ready, persist, err)
	}

	t.Log("A retained tombstone for the fourth process permits persisting the exact Pod-UID release fence")
	scenario.status.Traffic.Observed.Drained = append(scenario.status.Traffic.Observed.Drained, failed)
	ready, persist, err = scenario.coordinator.reconcileRetiredCapacity(context.Background(), scenario.groupID, &scenario.status, ep7, ep4, resolution)
	if err != nil || ready || !persist || scenario.status.Capacity.Desired == nil || len(scenario.status.Capacity.Desired.ReleaseFences) != 1 {
		t.Fatalf("complete terminal evidence did not persist a release fence: ready=%v, persist=%v, err=%v", ready, persist, err)
	}
}

func TestHeldAllocationWithdrawalEvidence(t *testing.T) {
	base := packedTopology()
	failed := ReplicaMembership{ReplicaID: "replica-1", Members: slices.Clone(base.Replicas[1].Members[1:2])}
	survivors := subtractMemberships(base.Replicas, []ReplicaMembership{failed})
	cases := []struct {
		name     string
		admitted []ReplicaMembership
		drain    []TrafficDrainTarget
		joining  []ReplicaTarget
		released bool
		want     []TrafficDrainTarget
	}{
		{
			name: "keep inactive process evidence while its allocation is held", admitted: survivors,
			want: []TrafficDrainTarget{{Membership: failed, Mode: TrafficDrainModeConfirmInactive}},
		},
		{
			name: "preserve graceful retirement and retain the previously withdrawn process", admitted: base.Replicas[:1],
			drain: []TrafficDrainTarget{{Membership: survivors[1], Mode: TrafficDrainModeGraceful}},
			want: []TrafficDrainTarget{
				{Membership: failed, Mode: TrafficDrainModeConfirmInactive},
				{Membership: survivors[1], Mode: TrafficDrainModeGraceful},
			},
		},
		{
			name: "never demand previous-admission evidence from a fresh joiner", admitted: base.Replicas[:1],
			joining: []ReplicaTarget{{ReplicaID: "replica-1", SlotID: "slot-1"}},
		},
		{
			name: "stop requesting evidence once release has been observed", admitted: base.Replicas[:1], released: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Derive drain retention from canonical held allocations without mutating the input")
			scenario := newCoordinatorScenario(t, base)
			if tc.released {
				scenario.status.Registry.Replicas[1].Current = nil
			}
			before := cloneStatus(scenario.status)
			target := TrafficTarget{Admitted: tc.admitted, Drain: tc.drain}
			beforeTarget := cloneTrafficTarget(&target)
			retained := retainHeldMemberDrains(scenario.status, target, tc.joining)
			if !sameTrafficDrainTargets(retained.Drain, tc.want) {
				t.Fatalf("retained drains: got %#v, want %#v", retained.Drain, tc.want)
			}
			if !reflect.DeepEqual(before, cloneStatus(scenario.status)) || !reflect.DeepEqual(beforeTarget, cloneTrafficTarget(&target)) {
				t.Fatal("deriving a traffic target mutated durable status or its input target")
			}
		})
	}
}

func TestTrafficFakeDropsUntargetedDrainEvidence(t *testing.T) {
	t.Log("Request terminal evidence for a failed process")
	base := packedTopology()
	scenario := newCoordinatorScenario(t, base)
	failed := ReplicaMembership{ReplicaID: "replica-1", Members: slices.Clone(base.Replicas[1].Members[1:2])}
	target := TrafficTarget{
		ControlRevision: 1, TransitionID: "mask-member", TopologyGeneration: 1,
		Admitted: subtractMemberships(base.Replicas, []ReplicaMembership{failed}),
		Drain:    []TrafficDrainTarget{{Membership: failed, Mode: TrafficDrainModeConfirmInactive}},
	}
	if result, err := scenario.traffic.Apply(context.Background(), scenario.groupID, target); err != nil || result.Rejection != nil {
		t.Fatalf("request terminal evidence: %v, %#v", err, result.Rejection)
	}
	if !containsMembership(scenario.traffic.observation.Drained, failed) {
		t.Fatal("fake did not establish requested evidence")
	}

	t.Log("A newer absolute target may discard any proof it no longer requests")
	target.ControlRevision++
	target.Drain = nil
	if result, err := scenario.traffic.Apply(context.Background(), scenario.groupID, target); err != nil || result.Rejection != nil {
		t.Fatalf("replace target: %v, %#v", err, result.Rejection)
	}
	if len(scenario.traffic.observation.Drained) != 0 {
		t.Fatal("fake retained untargeted evidence, masking missing retention in coordinator targets")
	}
}

func TestCommittedMembershipRequiresCanonicalAllocation(t *testing.T) {
	for _, missingRecord := range []bool{true, false} {
		t.Run(fmt.Sprintf("missing-record=%t", missingRecord), func(t *testing.T) {
			t.Log("Submit partial survivor reduction with valid canonical allocation evidence")
			base := packedTopology()
			scenario := newCoordinatorScenario(t, base)
			failed := ReplicaMembership{ReplicaID: "replica-1", Members: base.Replicas[1].Members[1:2]}
			plan := retirePlan("commit-check", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
			plan.Change = ResolvedChange{Kind: PlanKindReduceToSurvivors, ReduceToSurvivors: &ReduceToSurvivorsChange{Survivors: []ReplicaNativeMembership{
				{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []NativeMemberID{"dp-0", "dp-1", "dp-2", "dp-3"}},
				{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []NativeMemberID{"dp-4", "dp-6", "dp-7"}},
			}}}
			scenario.membership.unavailable = []ReplicaMembership{failed}
			scenario.desired = &plan
			scenario.runUntil("submit membership", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
			resolution, err := validateResolvedPlan(base, scenario.status.Registry, plan)
			if err != nil {
				t.Fatal(err)
			}
			result := MembershipTopology{Generation: 2, Replicas: subtractMemberships(base.Replicas, []ReplicaMembership{failed})}
			id := scenario.status.Membership.Desired.TransitionID
			scenario.membership.commit(id, result)

			t.Log("Missing registry or allocation evidence returns an error without publishing partial commit state")
			if missingRecord {
				scenario.status.Registry.Replicas = scenario.status.Registry.Replicas[:1]
			} else {
				scenario.status.Registry.Replicas[1].Current = nil
			}
			scenario.status.Registry.Replicas[0].DesiredNativeMembers = nil
			before := cloneStatus(scenario.status)
			ready, persist, err := scenario.coordinator.recordCommittedMembership(
				&scenario.status, base, result, resolution, scenario.membership.transitions[id],
			)
			if err == nil || ready || persist || !reflect.DeepEqual(before, scenario.status) {
				t.Fatalf("missing canonical evidence partially committed state: ready=%v, persist=%v, err=%v", ready, persist, err)
			}
		})
	}
}

func TestCoordinatorResumesBlockedCommitAfterObservationOrderingSkew(t *testing.T) {
	t.Log("Submit retirement once, then see the new topology before its correlated commit result")
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	plan := retirePlan("ordered-observations", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("submit membership", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
	result := engineTopology(2, 1)
	scenario.membership.topology = result
	scenario.mustReconcile("fail closed on pending result with an advanced topology")
	if scenario.status.Transition.Outcome != TransitionOutcomeBlocked {
		t.Fatal("observation skew did not fail closed")
	}

	t.Log("Restart, obtain the exact transition result, and resume without resubmitting the collective")
	scenario.rebuildCoordinator()
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, result)
	scenario.runUntil("resume and finish correlated retirement", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
	if scenario.membership.applyCalls != 1 {
		t.Fatal("resumption submitted a competing collective")
	}
}

func TestCoordinatorDetectsOneRestartedProcessWithinAnUnchangedPod(t *testing.T) {
	t.Log("Capture a packed allocation whose four members have independent process lifetimes")
	base := packedTopology()
	scenario := newCoordinatorScenario(t, base)
	retire := retirePlan("establish-target", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
	scenario.desired = &retire
	scenario.runUntil("submit retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, MembershipTopology{Generation: 2, Replicas: base.Replicas[:1]})
	scenario.runUntil("complete target", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })

	t.Log("A single process restarts without changing the Pod UID or its three neighbours")
	scenario.capacity.observation.Allocations[0].Incarnation.Members[2].RuntimeIncarnation = "process-2-v2"
	err := scenario.reconcile("refuse to adopt the restarted rank outside membership recovery")
	if !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("process restart was mistaken for converged capacity: %v", err)
	}
}

func TestCoordinatorRetriesTransientServingFailureWithoutReadmission(t *testing.T) {
	t.Log("Commit a retirement and begin topology-bound verification")
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	plan := retirePlan("transient-probe", "replica-1", TrafficRequirementQuiesceGroup, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.runUntil("submit retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 1))
	scenario.runUntil("persist verification intent", func(s *coordinatorScenario) bool {
		return s.status.Transition.Verification.Phase == VerificationPhasePending
	})
	scenario.verifier.failure = &Failure{Classification: FailureClassificationRetryable, Reason: "ProbeUnavailable"}

	t.Log("A retryable probe result leaves verification pending and traffic quiesced")
	err := scenario.reconcile("observe retryable serving failure")
	var invocation *InvocationError
	if !errors.As(err, &invocation) || scenario.status.Transition.Verification.Phase != VerificationPhasePending || len(scenario.traffic.observation.Admitted) != 0 {
		t.Fatalf("transient verification incorrectly admitted or blocked the world: %v", err)
	}

	t.Log("Restart and repeat verification of the same committed process identities before admission")
	scenario.verifier.failure = nil
	scenario.rebuildCoordinator()
	scenario.runUntil("verify and complete", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
}

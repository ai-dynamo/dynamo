/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestInvocationRecoveryFaultResumesFromCurrentAuthority(t *testing.T) {
	for _, authority := range []ObservationAuthority{ObservationAuthorityCapacity, ObservationAuthorityTraffic, ObservationAuthorityMembership} {
		t.Run(string(authority), func(t *testing.T) {
			t.Log("start growth whose invocation reports a recovery fault without a definitive rejection")
			scenario := newCoordinatorScenario(t, engineTopology(1, 2))
			joining := replicaIncarnation(2)
			scenario.capacity.planned[joining.ReplicaID] = joining
			plan := growPlan("recovery-retry", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID,
				NativeMembers: nativeMemberIDs(joining.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
			scenario.desired = &plan
			fault := fmt.Errorf("adapter lost its incarnation fence: %w", ErrRecoveryRequired)
			switch authority {
			case ObservationAuthorityCapacity:
				scenario.capacity.applyErr = fault
			case ObservationAuthorityTraffic:
				scenario.traffic.applyErr = fault
			case ObservationAuthorityMembership:
				scenario.membership.applyErr = fault
				scenario.membership.failAfterFirstAccept = true
			}
			var err error
			for range 30 {
				err = scenario.reconcile("advance to the failed invocation")
				if err != nil {
					break
				}
			}
			if !errors.Is(err, ErrRecoveryRequired) || scenario.status.Transition.Outcome != TransitionOutcomeProgressing {
				t.Fatalf("invocation changed the durable operation: error=%v transition=%#v", err, scenario.status.Transition)
			}
			if err := ValidateGroupStatus(scenario.status); err != nil {
				t.Fatalf("invocation left an invalid checkpoint: %v", err)
			}
			membershipTarget := cloneMembershipTarget(scenario.status.Membership.Desired)

			t.Log("keep refusing effects while fresh observations report the persistent unsafe condition")
			scenario.capacity.observeErr = fault
			calls := [3]int{scenario.capacity.applyCalls, scenario.traffic.applyCalls, scenario.membership.applyCalls}
			scenario.rebuildCoordinator()
			err = scenario.reconcile("observe the unresolved fault after restart")
			var observationErr *ObservationError
			if !errors.Is(err, ErrRecoveryRequired) || !errors.As(err, &observationErr) {
				t.Fatalf("persistent fault lost its observation authority: %v", err)
			}
			if actual := [3]int{scenario.capacity.applyCalls, scenario.traffic.applyCalls, scenario.membership.applyCalls}; actual != calls {
				t.Fatalf("invalid observations allowed effects: before=%v after=%v", calls, actual)
			}

			t.Log("recover authority and finish the same operation without replacing its membership target")
			scenario.capacity.observeErr = nil
			scenario.capacity.applyErr, scenario.traffic.applyErr, scenario.membership.applyErr = nil, nil, nil
			scenario.rebuildCoordinator()
			scenario.runUntil("resume identical growth", func(s *coordinatorScenario) bool { return s.membership.applyCalls > 0 })
			if membershipTarget != nil && !reflect.DeepEqual(membershipTarget, scenario.status.Membership.Desired) {
				t.Fatal("retry changed the accepted membership target")
			}
			scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 3))
			scenario.runUntil("record commit and complete growth", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
			if scenario.status.Topologies.CurrentGeneration != 2 {
				t.Fatal("authoritative commit was not recorded")
			}

			t.Log("a fresh valid resize is not poisoned by the previous invocation error")
			next := replicaIncarnation(3)
			nextPlan := growPlan("next-growth", ReplicaTarget{ReplicaID: next.ReplicaID, SlotID: next.SlotID,
				NativeMembers: nativeMemberIDs(next.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
			scenario.desired = &nextPlan
			scenario.mustReconcile("persist a new plan with a genuinely fresh replica identity")
			if scenario.status.Transition.Spec.Plan.ID != nextPlan.ID {
				t.Fatal("recovered group could not accept a subsequent valid plan")
			}
		})
	}
}

func TestRecoveryBeforeFirstResize(t *testing.T) {
	t.Log("lose an allocation before any resize and request the exact surviving member set")
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	scenario.capacity.observation.Allocations = scenario.capacity.observation.Allocations[:1]
	scenario.membership.unavailable = cloneReplicaMemberships(base.Replicas[1:])
	plan := retirePlan("first-recovery", "replica-1", TrafficRequirementKeepServing, VerificationRequirementRequired)
	plan.Change = ResolvedChange{Kind: PlanKindReduceToSurvivors, ReduceToSurvivors: &ReduceToSurvivorsChange{Survivors: []ReplicaNativeMembership{
		{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []NativeMemberID{"dp-0"}},
	}}}
	scenario.desired = &plan
	scenario.runUntil("validate and submit the initial recovery plan", func(s *coordinatorScenario) bool { return s.membership.applyCalls > 0 })

	t.Log("record the survivor commit and account for the disappeared allocation's exact release")
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 1))
	scenario.runUntil("complete first recovery", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
	if len(scenario.status.Registry.Replicas[1].History) != 1 {
		t.Fatal("recovery forgot the excluded allocation identity")
	}
}

func TestRetirementAfterTransientMembershipInvocationFault(t *testing.T) {
	t.Log("accept growth, lose its acknowledgment, and restart the coordinator")
	scenario := newCoordinatorScenario(t, engineTopology(1, 1))
	joining := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joining
	plan := growPlan("before-retirement", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID,
		NativeMembers: nativeMemberIDs(joining.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.membership.applyErr = ErrRecoveryRequired
	scenario.membership.failAfterFirstAccept = true
	var err error
	for range 30 {
		err = scenario.reconcile("advance until ambiguous submission")
		if err != nil {
			break
		}
	}
	if !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("missing invocation failure: %v", err)
	}
	scenario.membership.applyErr = nil
	scenario.rebuildCoordinator()
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 2))
	scenario.runUntil("finish growth from native truth", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })

	t.Log("retire the complete world and prove finalizer evidence can converge")
	retirement := retirePlan("terminal-retirement", "replica-0", TrafficRequirementQuiesceGroup, VerificationRequirementNone)
	retirement.Change.Retire.Replicas = []ReplicaID{"replica-0", "replica-1"}
	scenario.desired = &retirement
	previousCalls := scenario.membership.applyCalls
	scenario.runUntil("submit terminal retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls > previousCalls })
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(3, 0))
	scenario.runUntil("complete terminal retirement", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
	ready, err := TerminalDeletionEvidenceReady(scenario.status, scenario.capacity.observation, scenario.traffic.observation, scenario.status.Membership.Observed)
	if err != nil || !ready {
		t.Fatalf("transient invocation error prevented terminal deletion: ready=%v error=%v", ready, err)
	}
}

func TestAdapterFailureReasonsDoNotAuthorizeCoordinatorResumption(t *testing.T) {
	for _, reason := range []string{"RecoveryRequired", "UnknownMembershipOutcome", "TopologyChangedWhilePending", "TopologyChangedWithoutTransition", "ConflictingTopologyObservation", "RejectedAfterTopologyChanged"} {
		t.Run(reason, func(t *testing.T) {
			t.Log("complete a resize so retained membership evidence belongs to an older operation")
			scenario := newCoordinatorScenario(t, engineTopology(1, 1))
			joining := replicaIncarnation(1)
			scenario.capacity.planned[joining.ReplicaID] = joining
			plan := growPlan("completed-growth", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID,
				NativeMembers: nativeMemberIDs(joining.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
			scenario.desired = &plan
			scenario.runUntil("submit growth", func(s *coordinatorScenario) bool { return s.membership.applyCalls > 0 })
			scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 2))
			scenario.runUntil("finish growth", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })

			t.Log("reject a fresh plan using an adapter-defined diagnostic reason")
			next := replicaIncarnation(2)
			nextPlan := growPlan("rejected-growth", ReplicaTarget{ReplicaID: next.ReplicaID, SlotID: next.SlotID,
				NativeMembers: nativeMemberIDs(next.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
			scenario.desired = &nextPlan
			scenario.membership.planRejection = &Failure{Classification: FailureClassificationTerminal, Reason: reason}
			scenario.mustReconcile("persist the new plan")
			scenario.mustReconcile("record authoritative preflight rejection")
			calls := scenario.membership.applyCalls

			t.Log("old native success and reason text cannot resume the rejected plan")
			for range 3 {
				scenario.mustReconcile("maintain the blocked rejected plan")
			}
			if scenario.status.Transition.Outcome != TransitionOutcomeBlocked ||
				scenario.status.Transition.BlockKind != TransitionBlockKindSubsystemFailure || scenario.membership.applyCalls != calls {
				t.Fatal("adapter reason changed coordinator-owned block semantics")
			}
		})
	}
}

func TestObservableRecoveryFaultIsRecheckedWithoutLatching(t *testing.T) {
	t.Log("observe a typed recovery failure without inventing a transition or membership rejection")
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	scenario.membership.observeErr = fmt.Errorf("native health: %w", ErrRecoveryRequired)
	err := scenario.reconcile("fail closed on current native authority")
	var observationErr *ObservationError
	if !errors.Is(err, ErrRecoveryRequired) || !errors.As(err, &observationErr) || scenario.status.Transition != nil {
		t.Fatalf("unexpected recovery state: error=%v transition=%#v", err, scenario.status.Transition)
	}

	t.Log("recover observation authority without a permanent latch or membership mutation")
	scenario.membership.observeErr = nil
	scenario.rebuildCoordinator()
	scenario.mustReconcile("observe the unchanged healthy world")
	if scenario.membership.applyCalls != 0 {
		t.Fatal("observation recovery mutated membership")
	}
}

func TestPreflightRecoveryFaultIsRetryable(t *testing.T) {
	t.Log("freeze a plan before read-only native preflight reports a recovery fault")
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	joining := replicaIncarnation(2)
	scenario.capacity.planned[joining.ReplicaID] = joining
	plan := growPlan("preflight-recovery", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID,
		NativeMembers: nativeMemberIDs(joining.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.membership.planValidationErr = fmt.Errorf("native preflight: %w", ErrRecoveryRequired)
	scenario.mustReconcile("persist the plan")
	err := scenario.reconcile("report inconclusive preflight without a durable rejection")
	if !errors.Is(err, ErrRecoveryRequired) || scenario.status.Transition.Outcome != TransitionOutcomeProgressing || scenario.status.Transition.PlanPreflight.Rejection != nil {
		t.Fatalf("preflight error became a terminal outcome: %v", err)
	}
	if scenario.capacity.applyCalls != 0 || scenario.membership.applyCalls != 0 {
		t.Fatal("preflight failure allowed effects")
	}

	t.Log("revalidate current authority after restart and continue the unchanged plan")
	scenario.membership.planValidationErr = nil
	scenario.rebuildCoordinator()
	scenario.runUntil("resume validated growth", func(s *coordinatorScenario) bool { return s.membership.applyCalls > 0 })
}

func TestPreparationRejectionCannotRollBackAcceptedMembership(t *testing.T) {
	for _, authority := range []ObservationAuthority{ObservationAuthorityCapacity, ObservationAuthorityTraffic} {
		t.Run(string(authority), func(t *testing.T) {
			t.Log("accept growth while preserving its exact joining process identities")
			scenario := newCoordinatorScenario(t, engineTopology(1, 2))
			joining := replicaIncarnation(2)
			scenario.capacity.planned[joining.ReplicaID] = joining
			plan := growPlan("pending-growth", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID,
				NativeMembers: nativeMemberIDs(joining.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
			scenario.desired = &plan
			scenario.runUntil("submit growth", func(s *coordinatorScenario) bool { return s.membership.applyCalls > 0 })
			scenario.mustReconcile("observe native Pending")
			originalTarget := cloneMembershipTarget(scenario.status.Membership.Desired)

			t.Log("make preparatory capacity or a newer withdrawal target definitively unacceptable")
			switch authority {
			case ObservationAuthorityCapacity:
				scenario.capacity.observation.Allocations[2].Incarnation.CapacityRefs[0].UID = testReplacementPodUID
			case ObservationAuthorityTraffic:
				scenario.membership.unavailable = []ReplicaMembership{engineReplica(1)}
				scenario.traffic.rejectNext = &Failure{Classification: FailureClassificationTerminal, Reason: "WithdrawalRejected"}
			}
			var err error
			for range 5 {
				err = scenario.reconcile("observe rejection without canceling accepted membership")
				if err != nil {
					break
				}
			}
			var invocationErr *InvocationError
			if !errors.As(err, &invocationErr) || scenario.status.Transition.Outcome != TransitionOutcomeProgressing {
				t.Fatalf("preparatory rejection started rollback or lost its error: %v", err)
			}
			if len(scenario.capacity.observation.ReleaseFences) != 0 || !reflect.DeepEqual(originalTarget, scenario.status.Membership.Desired) {
				t.Fatal("rejection released capacity or replaced the pending membership target")
			}
			if err := ValidateGroupStatus(scenario.status); err != nil {
				t.Fatalf("pending rejection corrupted durable state: %v", err)
			}

			t.Log("record an authoritative commit before retrying the rejected preparatory target")
			scenario.membership.commit(originalTarget.TransitionID, engineTopology(2, 3))
			scenario.rebuildCoordinator()
			scenario.mustReconcile("persist native commit independently of preparatory drift")
			if scenario.status.Topologies.CurrentGeneration != 2 || scenario.status.Transition.Outcome == TransitionOutcomeReverting {
				t.Fatal("preparatory rejection stranded the native commit")
			}
			if authority == ObservationAuthorityCapacity {
				t.Log("recording native truth does not authorize effects against a replacement Pod")
				calls := [2]int{scenario.capacity.applyCalls, scenario.traffic.applyCalls}
				err := scenario.reconcile("refuse effects until exact physical identity is restored")
				if !errors.Is(err, ErrRecoveryRequired) || calls != [2]int{scenario.capacity.applyCalls, scenario.traffic.applyCalls} {
					t.Fatalf("commit bypassed replacement identity checks: %v", err)
				}
			}
		})
	}
}

func TestMaintainedInvocationFaultPreservesMembershipAuthorityBlock(t *testing.T) {
	t.Log("quiesce traffic for retirement, then observe an inconclusive native result")
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	plan := retirePlan("unknown-retirement", "replica-1", TrafficRequirementQuiesceGroup, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.runUntil("submit retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls > 0 })
	id := scenario.status.Membership.Desired.TransitionID
	observation := scenario.membership.transitions[id]
	observation.Phase = MembershipTransitionPhaseUnknown
	scenario.membership.transitions[id] = observation
	scenario.runUntil("block on native authority", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeBlocked })

	t.Log("persist coordinator-owned block semantics independently of the backend reason")
	checkpoint, err := json.Marshal(scenario.status)
	if err != nil {
		t.Fatal(err)
	}
	var restored GroupStatus
	if err := json.Unmarshal(checkpoint, &restored); err != nil {
		t.Fatal(err)
	}
	if err := ValidateGroupStatus(restored); err != nil {
		t.Fatal(err)
	}
	scenario.status = restored

	t.Log("a maintained traffic invocation fails without overwriting the resumable membership block")
	scenario.traffic.observation.Admitted = cloneReplicaMemberships(base.Replicas)
	scenario.traffic.observation.Drained = nil
	scenario.traffic.applyErr = ErrRecoveryRequired
	err = scenario.reconcile("attempt to restore the accepted withdrawal target")
	if !errors.Is(err, ErrRecoveryRequired) || scenario.status.Transition.BlockKind != TransitionBlockKindMembershipAuthority {
		t.Fatalf("invocation replaced the native authority block: %v", err)
	}

	t.Log("recover the traffic adapter and consume the correlated native commit after restart")
	scenario.traffic.applyErr = nil
	scenario.mustReconcile("reassert the existing withdrawal target")
	scenario.membership.commit(id, engineTopology(2, 1))
	scenario.rebuildCoordinator()
	scenario.runUntil("finish the recovered retirement", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
}

func TestMaintainedInvocationFaultDoesNotRewriteCompletedTransition(t *testing.T) {
	t.Log("complete planned retirement with retained negative traffic intent")
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	plan := retirePlan("completed-retirement", "replica-1", TrafficRequirementKeepServing, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.runUntil("submit retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls > 0 })
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 1))
	scenario.runUntil("finish retirement", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })

	t.Log("fail a repair of routing drift without rewriting the completed operation")
	scenario.traffic.observation.Admitted = cloneReplicaMemberships(base.Replicas)
	scenario.traffic.observation.Drained = nil
	scenario.traffic.applyErr = ErrRecoveryRequired
	err := scenario.reconcile("reassert accepted withdrawal")
	if !errors.Is(err, ErrRecoveryRequired) || scenario.status.Transition.Outcome != TransitionOutcomeCompleted || scenario.status.Transition.Failure != nil {
		t.Fatalf("maintained invocation rewrote completion: %v", err)
	}

	t.Log("repair the same accepted target once authority returns")
	scenario.traffic.applyErr = nil
	scenario.rebuildCoordinator()
	scenario.mustReconcile("repair retained withdrawal after restart")
	scenario.mustReconcile("observe repaired withdrawal")
	if !sameMemberships(scenario.traffic.observation.Admitted, engineTopology(2, 1).Replicas) {
		t.Fatal("transient invocation error stranded the accepted withdrawal target")
	}
}

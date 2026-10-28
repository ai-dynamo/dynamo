/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestInvocationRecoveryFaultFencesEffectsAcrossRestart(t *testing.T) {
	for _, authority := range []ObservationAuthority{ObservationAuthorityCapacity, ObservationAuthorityTraffic, ObservationAuthorityMembership} {
		t.Run(string(authority), func(t *testing.T) {
			t.Log("start growth whose adapter discovers a recovery fault only while applying its target")
			scenario := newCoordinatorScenario(t, engineTopology(1, 2))
			joining := replicaIncarnation(2)
			scenario.capacity.planned[joining.ReplicaID] = joining
			plan := growPlan("recovery-fence", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID,
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
				err = scenario.reconcile("advance to the fenced invocation")
				if err != nil {
					break
				}
			}
			if !errors.Is(err, ErrRecoveryRequired) || !scenario.status.Transition.RequiresRecovery() {
				t.Fatalf("fault was not retained: error=%v transition=%#v", err, scenario.status.Transition)
			}
			if err := ValidateGroupStatus(scenario.status); err != nil {
				t.Fatalf("recovery checkpoint is invalid: %v", err)
			}
			membershipTarget := cloneMembershipTarget(scenario.status.Membership.Desired)
			calls := [3]int{scenario.capacity.applyCalls, scenario.traffic.applyCalls, scenario.membership.applyCalls}

			t.Log("fresh observations and a controller restart do not erase an invocation-only failure")
			scenario.capacity.applyErr, scenario.traffic.applyErr, scenario.membership.applyErr = nil, nil, nil
			for range 3 {
				scenario.rebuildCoordinator()
				if err := scenario.reconcile("observe without replaying unsafe effects"); !errors.Is(err, ErrRecoveryRequired) {
					t.Fatalf("lost recovery diagnosis: %v", err)
				}
			}
			if actual := [3]int{scenario.capacity.applyCalls, scenario.traffic.applyCalls, scenario.membership.applyCalls}; actual != calls {
				t.Fatalf("recovery replayed effects: before=%v after=%v", calls, actual)
			}
			if !reflect.DeepEqual(membershipTarget, scenario.status.Membership.Desired) {
				t.Fatal("recovery changed the accepted membership payload")
			}

			t.Log("a subsequent transient observation failure cannot erase the retained recovery diagnosis")
			scenario.capacity.observeErr = errors.New("temporary observation outage")
			err = scenario.reconcile("retain the fault while fresh observations are unavailable")
			var observationErr *ObservationError
			if !errors.Is(err, ErrRecoveryRequired) || !errors.As(err, &observationErr) {
				t.Fatalf("lost recovery or observation evidence: %v", err)
			}
			scenario.capacity.observeErr = nil

			t.Log("a new ordinary resize cannot masquerade as explicit recovery")
			plan.ID = "ordinary-resize"
			if err := scenario.reconcile("refuse a replacement growth plan"); !errors.Is(err, ErrRecoveryRequired) {
				t.Fatalf("ordinary plan bypassed the recovery fence: %v", err)
			}
			if authority == ObservationAuthorityMembership {
				t.Log("continue observing the accepted collective without inventing a rejection or rollback")
				if scenario.status.Membership.Observed.Transition == nil || scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhasePending {
					t.Fatal("accepted membership did not remain pending")
				}
				scenario.membership.commit(membershipTarget.TransitionID, engineTopology(2, 3))
				if err := scenario.reconcile("observe the native commit while retaining the recovery fence"); !errors.Is(err, ErrRecoveryRequired) {
					t.Fatalf("lost recovery after native commit: %v", err)
				}
				if scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhaseCommitted {
					t.Fatal("recovery rewrote the native terminal result")
				}
			}
		})
	}
}

func TestObservableRecoveryFaultIsRecheckedWithoutLatching(t *testing.T) {
	t.Log("observe a typed recovery failure without inventing a new transition or membership rejection")
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

func TestPreflightRecoveryFaultDoesNotBecomeAnInvocationLatch(t *testing.T) {
	t.Log("freeze a new plan before its read-only native preflight discovers a recovery fault")
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	joining := replicaIncarnation(2)
	scenario.capacity.planned[joining.ReplicaID] = joining
	plan := growPlan("preflight-recovery", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID,
		NativeMembers: nativeMemberIDs(joining.Members), Bootstrap: BootstrapModeJoin}, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.membership.planValidationErr = fmt.Errorf("native preflight: %w", ErrRecoveryRequired)
	scenario.mustReconcile("persist the plan")
	err := scenario.reconcile("fail closed without marking an invocation fault")
	var observationErr *ObservationError
	if !errors.Is(err, ErrRecoveryRequired) || !errors.As(err, &observationErr) || scenario.status.Transition.RequiresRecovery() {
		t.Fatalf("preflight failure incorrectly latched: %v", err)
	}
	if scenario.capacity.applyCalls != 0 || scenario.membership.applyCalls != 0 {
		t.Fatal("preflight failure allowed effects")
	}

	t.Log("revalidate current authority after restart instead of permanently retaining an observation fault")
	scenario.membership.planValidationErr = nil
	scenario.rebuildCoordinator()
	scenario.runUntil("resume the validated growth", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
}

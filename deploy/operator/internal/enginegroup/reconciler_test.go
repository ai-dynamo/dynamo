/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package enginegroup

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

func TestCoordinatorGrowthOrdersCapacityCommitVerificationAndAdmission(t *testing.T) {
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	joining := engineReplica(2)
	joiningIncarnation := replicaIncarnation(2)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("grow-to-three", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementRequired)
	scenario.desired = &plan

	t.Log("Converge physical capacity and the pre-membership traffic fence before application")
	scenario.runUntil("prepare and apply growth", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	if eventIndex(scenario.events, "capacity:") == -1 || eventIndex(scenario.events, "traffic:") == -1 {
		t.Fatalf("expected capacity and traffic projections before membership, events: %v", scenario.events)
	}
	if eventIndex(scenario.events, "capacity:") > eventIndex(scenario.events, "membership:") ||
		eventIndex(scenario.events, "traffic:") > eventIndex(scenario.events, "membership:") {
		t.Fatalf("membership was applied before its prerequisites, events: %v", scenario.events)
	}
	if eventIndex(scenario.events, "verify:") != -1 {
		t.Fatalf("serving verification ran before commit, events: %v", scenario.events)
	}
	if !sameMemberships(scenario.traffic.observation.Admitted, engineTopology(1, 2).Replicas) {
		t.Fatalf("joining replica became routable before commit: %#v", scenario.traffic.observation.Admitted)
	}
	bootstrapTarget := scenario.capacity.firstTarget
	capacityTarget := scenario.capacity.lastTarget
	if bootstrapTarget == nil || capacityTarget == nil {
		t.Fatal("joining capacity targets were not retained")
	}
	joiningCapacity := bootstrapTarget.Replicas[len(bootstrapTarget.Replicas)-1]
	if joiningCapacity.Bootstrap == nil ||
		joiningCapacity.Bootstrap.Mode != BootstrapModeJoin ||
		joiningCapacity.Bootstrap.BaseTopologyGeneration != 1 ||
		!slices.Equal(joiningCapacity.Bootstrap.NativeMembers, joining.NativeMembers) {
		t.Fatalf("joining bootstrap lacks resolved topology and native identity: %#v", joiningCapacity.Bootstrap)
	}
	pinnedCapacity := capacityTarget.Replicas[len(capacityTarget.Replicas)-1]
	if pinnedCapacity.Bootstrap != nil || pinnedCapacity.Incarnation == nil ||
		!sameIncarnation(*pinnedCapacity.Incarnation, joiningIncarnation) ||
		capacityTarget.ControlRevision <= bootstrapTarget.ControlRevision {
		t.Fatalf("joining capacity was not pinned before membership: %#v", pinnedCapacity)
	}
	if scenario.status.Capacity.Accepted == nil ||
		scenario.status.Capacity.Accepted.ControlRevision != capacityTarget.ControlRevision {
		t.Fatalf("exact joining capacity was not durably accepted before membership: %#v", scenario.status.Capacity)
	}

	t.Log("Commit the exact topology produced by the engine")
	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(engineTopology(1, 2).Replicas),
			cloneReplicaMembership(joining),
		),
	}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)

	t.Log("Verify serving before explicitly admitting the committed topology")
	scenario.runUntil("finish growth", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
	verificationIndex := eventIndex(scenario.events, "verify:")
	lastTrafficIndex := -1
	for index, event := range scenario.events {
		if strings.HasPrefix(event, "traffic:") {
			lastTrafficIndex = index
		}
	}
	if verificationIndex == -1 || lastTrafficIndex <= verificationIndex {
		t.Fatalf("committed topology was not verified before admission, events: %v", scenario.events)
	}
	if !sameMemberships(scenario.traffic.observation.Admitted, committed.Replicas) {
		t.Fatalf("committed topology was not admitted exactly: %#v", scenario.traffic.observation.Admitted)
	}
	record, found := scenario.status.Registry.Find(joining.ReplicaID)
	if !found || record.Current == nil || !sameIncarnation(*record.Current, joiningIncarnation) {
		t.Fatalf("joining incarnation was not frozen in the canonical registry: %#v", record)
	}
}

func TestCoordinatorShrinkOrdersDrainCommitReleaseVerificationAndAdmission(t *testing.T) {
	const retiringReplicaID ReplicaID = "replica-1"

	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	plan := retirePlan(
		"retire-second",
		retiringReplicaID,
		TrafficRequirementQuiesceGroup,
		VerificationRequirementRequired,
	)
	scenario.desired = &plan

	t.Log("Drain the complete quiescing group before applying selected retirement")
	scenario.runUntil("prepare and apply retirement", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	if len(scenario.membership.targets) != 1 {
		t.Fatalf("expected one immutable membership target, got %d", len(scenario.membership.targets))
	}
	if len(scenario.traffic.observation.Admitted) != 0 ||
		!sameMemberships(scenario.traffic.observation.Drained, base.Replicas) {
		t.Fatalf("whole-group drain did not complete before apply: %#v", scenario.traffic.observation)
	}

	t.Log("Commit the survivor topology and release only the selected Pod UID")
	committed := MembershipTopology{
		Generation: 2,
		Replicas:   cloneReplicaMemberships(base.Replicas[:1]),
	}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("finish retirement", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
	if len(scenario.capacity.observation.ReleaseFences) != 1 {
		t.Fatalf("expected one release fence, got %#v", scenario.capacity.observation.ReleaseFences)
	}
	fence := scenario.capacity.observation.ReleaseFences[0]
	if fence.TransitionID != scenario.status.Transition.Spec.ID ||
		fence.AuthorizingTopologyGeneration != committed.Generation ||
		fence.ReplicaID != retiringReplicaID || fence.CapacityRefs[0].UID != "pod-uid-1-v1" {
		t.Fatalf("release did not retain the exact victim identity: %#v", fence)
	}
	if _, found := allocationByID(scenario.capacity.observation, retiringReplicaID); found {
		t.Fatal("retired physical capacity remains allocated")
	}
	record, found := scenario.status.Registry.Find(retiringReplicaID)
	if !found || record.Current != nil || len(record.History) != 1 {
		t.Fatalf("retired replica history is not canonical: %#v", record)
	}
	if !sameIncarnation(record.History[0].Incarnation, replicaIncarnation(1)) ||
		!slices.Equal(record.History[0].NativeMembers, base.Replicas[1].NativeMembers) {
		t.Fatalf("retired replica lost historical native membership: %#v", record.History)
	}

	t.Log("Confirm release precedes verification and survivors are admitted only afterward")
	capacityIndex := eventIndex(scenario.events, "capacity:")
	verificationIndex := eventIndex(scenario.events, "verify:")
	if capacityIndex == -1 || verificationIndex == -1 || capacityIndex > verificationIndex {
		t.Fatalf("expected release before serving verification, events: %v", scenario.events)
	}
	if !sameMemberships(scenario.traffic.observation.Admitted, committed.Replicas) {
		t.Fatalf("survivor topology was not readmitted exactly: %#v", scenario.traffic.observation.Admitted)
	}
}

func TestCoordinatorReassertsCompletedTargetsAfterObservedDrift(t *testing.T) {
	const retiringReplicaID ReplicaID = "replica-1"

	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	plan := retirePlan(
		"reassert-completed-retirement",
		retiringReplicaID,
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &plan
	scenario.runUntil("apply retirement", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{Generation: 2, Replicas: cloneReplicaMemberships(base.Replicas[:1])}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("complete retirement", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})

	t.Log("Recover accepted payloads from their durable applied revisions after restart")
	scenario.status.Capacity.Accepted = nil
	scenario.status.Traffic.Accepted = nil
	scenario.rebuildCoordinator()
	scenario.mustReconcile("recover accepted targets")
	if scenario.status.Capacity.Accepted == nil || scenario.status.Traffic.Accepted == nil {
		t.Fatalf("durable adapter acknowledgments did not recover accepted targets: %#v", scenario.status)
	}

	t.Log("Remove an unsafe traffic drift by replaying the accepted target at the same revision")
	trafficCalls := scenario.traffic.applyCalls
	scenario.traffic.observation.Admitted = cloneReplicaMemberships(base.Replicas)
	scenario.mustReconcile("reassert completed traffic target")
	if scenario.traffic.applyCalls != trafficCalls+1 ||
		!sameMemberships(scenario.traffic.observation.Admitted, committed.Replicas) {
		t.Fatalf(
			"completed traffic target was not reasserted: calls=%d observation=%#v",
			scenario.traffic.applyCalls,
			scenario.traffic.observation,
		)
	}

	t.Log("Remove reappearing retired capacity only through its retained exact release fence")
	capacityCalls := scenario.capacity.applyCalls
	scenario.capacity.observation.Allocations = append(
		scenario.capacity.observation.Allocations,
		CapacityAllocation{Incarnation: replicaIncarnation(1), Available: true},
	)
	scenario.mustReconcile("reassert completed capacity target")
	if scenario.capacity.applyCalls != capacityCalls+1 {
		t.Fatalf("completed capacity target was not reasserted: %d applications", scenario.capacity.applyCalls)
	}
	if _, found := allocationByID(scenario.capacity.observation, retiringReplicaID); found {
		t.Fatal("reasserted capacity target retained the exactly fenced retired allocation")
	}
	scenario.mustReconcile("observe repaired terminal targets")
	if scenario.status.Transition.Outcome != TransitionOutcomeCompleted {
		t.Fatalf("steady-state repair changed the completed outcome: %#v", scenario.status.Transition)
	}
}

func TestCoordinatorRequiresRecoveryForCommittedCapacityIncarnationDrift(t *testing.T) {
	testCases := []struct {
		name     string
		mutation string
	}{
		{name: "replacement incarnation", mutation: "replace"},
		{name: "missing incarnation", mutation: "remove"},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			base := engineTopology(1, 1)
			scenario := newCoordinatorScenario(t, base)
			joining := engineReplica(1)
			joiningIncarnation := replicaIncarnation(1)
			scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
			plan := growPlan("capacity-drift", ReplicaTarget{
				ReplicaID: joining.ReplicaID,
				SlotID:    joiningIncarnation.SlotID,
				Bootstrap: BootstrapModeJoin,
			}, VerificationRequirementNone)
			scenario.desired = &plan

			t.Log("Commit growth with capacity pinned to the exact joining runtime and Pod UID")
			scenario.runUntil("apply growth", func(s *coordinatorScenario) bool {
				return s.membership.applyCalls == 1
			})
			committed := MembershipTopology{
				Generation: 2,
				Replicas: append(
					cloneReplicaMemberships(base.Replicas),
					cloneReplicaMembership(joining),
				),
			}
			scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
			scenario.runUntil("complete growth", func(s *coordinatorScenario) bool {
				return s.status.Transition.Outcome == TransitionOutcomeCompleted
			})

			t.Log("Change committed physical capacity without an engine membership recovery transition")
			switch testCase.mutation {
			case "replace":
				allocation, found := allocationByID(scenario.capacity.observation, joining.ReplicaID)
				if !found {
					t.Fatal("joining allocation is absent before replacement")
				}
				allocation.Incarnation.RuntimeIncarnation = testReplacementRuntime
				allocation.Incarnation.CapacityRefs[0].UID = testReplacementPodUID
				for index := range scenario.capacity.observation.Allocations {
					if scenario.capacity.observation.Allocations[index].Incarnation.ReplicaID == joining.ReplicaID {
						scenario.capacity.observation.Allocations[index] = allocation
					}
				}
			case "remove":
				allocations := scenario.capacity.observation.Allocations[:0]
				for _, allocation := range scenario.capacity.observation.Allocations {
					if allocation.Incarnation.ReplicaID != joining.ReplicaID {
						allocations = append(allocations, allocation)
					}
				}
				scenario.capacity.observation.Allocations = allocations
			default:
				t.Fatalf("unknown test mutation %q", testCase.mutation)
			}

			capacityCalls := scenario.capacity.applyCalls
			err := scenario.reconcile("detect committed capacity drift")
			if !errors.Is(err, ErrRecoveryRequired) {
				t.Fatalf("expected recovery-required error, got %v", err)
			}
			if scenario.capacity.applyCalls != capacityCalls {
				t.Fatalf("capacity drift replayed bootstrap or exact creation: %d applications", scenario.capacity.applyCalls)
			}
			if scenario.status.Transition.Outcome != TransitionOutcomeCompleted {
				t.Fatalf("capacity drift rewrote the completed transition: %#v", scenario.status.Transition)
			}
		})
	}
}

func TestCoordinatorRestartAfterAmbiguousApplyDoesNotCompete(t *testing.T) {
	scenario := newCoordinatorScenario(t, engineTopology(1, 1))
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	scenario.membership.failAfterFirstAccept = true
	plan := growPlan("restart-safe-growth", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan

	t.Log("Persist the exact membership target, then simulate a timeout after engine acceptance")
	var applyErr error
	for iteration := 1; iteration <= 20 && applyErr == nil; iteration++ {
		applyErr = scenario.reconcile("advance to ambiguous membership application")
	}
	if applyErr == nil || !strings.Contains(applyErr.Error(), "timed out after acceptance") {
		t.Fatalf("expected ambiguous apply error, got %v", applyErr)
	}
	if scenario.membership.applyCalls != 1 || scenario.status.Membership.Desired == nil ||
		scenario.status.Membership.Observed.Transition != nil {
		t.Fatalf("unexpected state after ambiguous apply: calls=%d status=%#v", scenario.membership.applyCalls, scenario.status.Membership)
	}

	t.Log("Rebuild the coordinator and recover the running operation by its durable identity")
	scenario.rebuildCoordinator()
	scenario.mustReconcile("observe the accepted membership operation after restart")
	if scenario.membership.applyCalls != 1 {
		t.Fatalf("restart started a competing membership application: %d calls", scenario.membership.applyCalls)
	}
	if scenario.status.Membership.Observed.Transition == nil ||
		scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhasePending {
		t.Fatalf("pending transition was not recovered: %#v", scenario.status.Membership)
	}

	t.Log("Commit the original operation and finish without changing its request identity")
	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(engineTopology(1, 1).Replicas),
			cloneReplicaMembership(joining),
		),
	}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("finish recovered growth", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
	if scenario.membership.applyCalls != 1 {
		t.Fatalf("committed request was applied more than once: %d", scenario.membership.applyCalls)
	}
}

func TestCoordinatorReplaysIdenticalTargetAfterAuthoritativeAbsence(t *testing.T) {
	scenario := newCoordinatorScenario(t, engineTopology(1, 1))
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	scenario.membership.failBeforeFirstAccept = true
	plan := growPlan("replay-absent-growth", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan

	t.Log("Persist the target, then lose the first invocation before adapter acceptance")
	var applyErr error
	for iteration := 1; iteration <= 30 && applyErr == nil; iteration++ {
		applyErr = scenario.reconcile("advance to pre-acceptance timeout")
	}
	if applyErr == nil || !strings.Contains(applyErr.Error(), "timed out before acceptance") {
		t.Fatalf("expected pre-acceptance timeout, got %v", applyErr)
	}
	digest := scenario.status.Membership.Desired.TargetDigest
	if scenario.membership.applyCalls != 1 || len(scenario.membership.targets) != 0 {
		t.Fatalf("adapter unexpectedly retained the failed invocation: %#v", scenario.membership)
	}

	t.Log("Restart, observe exact-ID absence, and safely replay the unchanged desired level")
	scenario.rebuildCoordinator()
	scenario.mustReconcile("observe absence and replay exact target")
	if scenario.membership.applyCalls != 2 || len(scenario.membership.targets) != 1 ||
		scenario.status.Membership.Desired.TargetDigest != digest {
		t.Fatalf("absence did not replay one identical target: %#v", scenario.membership)
	}
	scenario.mustReconcile("observe replayed target pending")
	if scenario.status.Membership.Observed.Transition == nil ||
		scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhasePending {
		t.Fatalf("replayed target was not recovered: %#v", scenario.status.Membership)
	}
}

func TestCoordinatorWaitsForAuthoritativeTopologyAfterCorrelatedCommit(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("delayed-topology-growth", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("apply growth target", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}

	t.Log("Report the immutable transition result while the general topology observer remains on the base")
	scenario.membership.reportCommit(scenario.status.Membership.Desired.TransitionID, committed)
	for iteration := 1; iteration <= 3; iteration++ {
		scenario.mustReconcile("wait for authoritative topology to catch up")
	}
	if scenario.status.Transition.Outcome != TransitionOutcomeProgressing ||
		!sameTopologyWithCurrent(scenario.status.Topologies, base) {
		t.Fatalf("correlated result committed before authoritative topology agreed: %#v", scenario.status)
	}

	t.Log("Publish the same topology through the general observer and complete convergence")
	scenario.membership.topology = cloneTopology(committed)
	scenario.runUntil("complete delayed topology growth", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
}

func TestCoordinatorFailsClosedWhenCommitAndAuthoritativeTopologyConflict(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("conflicting-topology-growth", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("apply growth target", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	result := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	scenario.membership.reportCommit(scenario.status.Membership.Desired.TransitionID, result)
	scenario.membership.topology = MembershipTopology{Generation: 3, Replicas: cloneReplicaMemberships(base.Replicas)}

	t.Log("Refuse admission when the transition result and authoritative engine topology describe different worlds")
	scenario.runUntil("block conflicting topology", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	if scenario.status.Transition.Failure.Reason != "ConflictingTopologyObservation" ||
		scenario.verifier.calls != 0 {
		t.Fatalf("conflicting topology did not fail closed: status=%#v events=%v", scenario.status, scenario.events)
	}
}

func TestCoordinatorResumesWhenUnknownMembershipAuthorityRecovers(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("recover-authority", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("apply growth target", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	target := *scenario.status.Membership.Desired
	scenario.membership.transitions[target.TransitionID] = MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           MembershipTransitionPhaseUnknown,
		Failure: &Failure{
			Classification: FailureClassificationRetryable,
			Reason:         "AuthorityUnavailable",
			Message:        "adapter temporarily lost authoritative state",
		},
	}

	t.Log("Fail closed while the accepted collective has an unknown outcome")
	scenario.runUntil("block unknown transition", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	if scenario.membership.applyCalls != 1 {
		t.Fatalf("unknown transition was replayed: %d applies", scenario.membership.applyCalls)
	}

	t.Log("Recover authority for the same transition and resume without superseding it")
	scenario.membership.transitions[target.TransitionID] = MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           MembershipTransitionPhasePending,
	}
	scenario.mustReconcile("resume recovered pending transition")
	if scenario.status.Transition.Outcome != TransitionOutcomeProgressing ||
		scenario.membership.applyCalls != 1 {
		t.Fatalf("recovered authority did not resume safely: %#v", scenario.status.Transition)
	}

	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	scenario.membership.commit(target.TransitionID, committed)
	scenario.runUntil("complete recovered transition", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
}

func TestCoordinatorRollsBackRecoveredRejectionBeforeAcceptingReplacementPlan(t *testing.T) {
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	retiring := base.Replicas[1].ReplicaID
	plan := retirePlan(
		"rejected-retirement",
		retiring,
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &plan
	scenario.runUntil("apply retirement target", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	target := *scenario.status.Membership.Desired
	scenario.membership.transitions[target.TransitionID] = MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           MembershipTransitionPhaseUnknown,
		Failure: &Failure{
			Classification: FailureClassificationRetryable,
			Reason:         "AuthorityUnavailable",
			Message:        "adapter temporarily lost authoritative state",
		},
	}
	scenario.runUntil("block unknown retirement", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeBlocked
	})

	replacement := retirePlan(
		"replacement-retirement",
		retiring,
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &replacement
	scenario.membership.transitions[target.TransitionID] = MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           MembershipTransitionPhaseRejected,
		Failure: &Failure{
			Classification: FailureClassificationTerminal,
			Reason:         "Rejected",
			Message:        "the exact retirement target was rejected without mutation",
		},
	}

	t.Log("Interpret and persist the recovered rejection before considering the newer plan")
	scenario.mustReconcile("begin rollback of rejected retirement")
	if scenario.status.Transition.Outcome != TransitionOutcomeReverting ||
		scenario.status.Transition.Spec.Plan.ID != plan.ID {
		t.Fatalf("replacement bypassed rollback of the rejected plan: %#v", scenario.status.Transition)
	}

	t.Log("Restore the old plan's preparatory state before starting the replacement")
	scenario.runUntil("finish rejected retirement rollback", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeRolledBack
	})
	scenario.mustReconcile("start replacement retirement")
	if scenario.status.Transition.Spec.Plan.ID != replacement.ID ||
		scenario.status.Transition.Outcome != TransitionOutcomeProgressing {
		t.Fatalf("replacement did not start after rollback: %#v", scenario.status.Transition)
	}
}

func TestCoordinatorRollsBackPreparatoryStateAfterDefinitiveRejection(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	rejection := Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "UnsupportedTransition",
		Message:        "engine rejected the exact plan without mutation",
	}
	scenario.membership.applyRejection = &rejection
	plan := growPlan("rejected-growth", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementRequired)
	plan.TrafficRequirement = TrafficRequirementQuiesceGroup
	scenario.desired = &plan

	t.Log("Converge preparatory capacity and traffic before a definitive engine rejection")
	scenario.runUntil("observe definitive membership rejection", func(s *coordinatorScenario) bool {
		return s.status.Transition != nil && s.status.Transition.Outcome == TransitionOutcomeReverting
	})
	if scenario.status.Membership.Observed.Transition == nil ||
		scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhaseRejected {
		t.Fatalf("membership rejection was not retained independently: %#v", scenario.status.Membership)
	}
	if _, found := allocationByID(scenario.capacity.observation, joining.ReplicaID); !found {
		t.Fatal("test did not reach allocated preparatory capacity")
	}

	t.Log("Restore the canonical base from its topology reference and release only uncommitted capacity")
	scenario.runUntil("finish rollback", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeRolledBack
	})
	if !sameTopology(scenario.membership.topology, base) {
		t.Fatalf("rollback changed engine membership: %#v", scenario.membership.topology)
	}
	if _, found := allocationByID(scenario.capacity.observation, joining.ReplicaID); found {
		t.Fatal("rollback retained uncommitted joining capacity")
	}
	if _, found := scenario.status.Registry.Find(joining.ReplicaID); found {
		t.Fatal("fresh uncommitted logical identity remained in the canonical registry")
	}
	if !sameMemberships(scenario.traffic.observation.Admitted, base.Replicas) {
		t.Fatalf("rollback did not restore base traffic: %#v", scenario.traffic.observation)
	}
}

func TestCoordinatorDoesNotRollBackRejectedTargetAfterTopologyChanged(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	scenario.membership.applyRejection = &Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "Rejected",
		Message:        "target was rejected without mutation",
	}
	plan := growPlan("rejected-after-change", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("apply rejected growth target", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})

	t.Log("Advance authoritative topology independently before observing the correlated rejection")
	scenario.membership.topology = MembershipTopology{
		Generation: 2,
		Replicas:   cloneReplicaMemberships(base.Replicas),
	}
	scenario.runUntil("block unsafe rejection rollback", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	if scenario.status.Transition.Failure.Reason != "RejectedAfterTopologyChanged" {
		t.Fatalf("changed topology was treated as safely rejected: %#v", scenario.status.Transition)
	}
	if _, found := allocationByID(scenario.capacity.observation, joining.ReplicaID); !found {
		t.Fatal("unsafe rollback released joining capacity")
	}
}

func TestCoordinatorRollsBackPartiallyAllocatedCapacityAfterDefinitiveRejection(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.rejectNext = &Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "CapacityRejected",
		Message:        "capacity target was rejected after one allocation was created",
	}
	scenario.capacity.partial = []CapacityAllocation{{
		Incarnation: joiningIncarnation,
		Available:   false,
	}}
	plan := growPlan("partially-rejected-growth", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan

	t.Log("Observe the definitive rejection and its partially created physical allocation")
	scenario.runUntil("begin partial-capacity rollback", func(s *coordinatorScenario) bool {
		return s.status.Transition != nil && s.status.Transition.Outcome == TransitionOutcomeReverting
	})
	if _, found := allocationByID(scenario.capacity.observation, joining.ReplicaID); !found {
		t.Fatal("test capacity adapter did not expose its partial allocation")
	}

	t.Log("Fence the observed Pod UID and remove the uncommitted allocation during rollback")
	scenario.runUntil("finish partial-capacity rollback", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeRolledBack
	})
	if _, found := allocationByID(scenario.capacity.observation, joining.ReplicaID); found {
		t.Fatal("rollback retained partially allocated capacity")
	}
	if len(scenario.capacity.observation.ReleaseFences) != 1 ||
		scenario.capacity.observation.ReleaseFences[0].AuthorizingTopologyGeneration != base.Generation ||
		scenario.capacity.observation.ReleaseFences[0].CapacityRefs[0].UID != joiningIncarnation.CapacityRefs[0].UID {
		t.Fatalf("rollback did not fence the exact partial allocation: %#v", scenario.capacity.observation.ReleaseFences)
	}
}

func TestCoordinatorServingFailureDoesNotRewriteMembershipCommit(t *testing.T) {
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	failure := Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "CollectiveStalled",
		Message:        "serving progress check failed",
	}
	scenario.verifier.failure = &failure
	plan := retirePlan(
		"failed-verification",
		"replica-1",
		TrafficRequirementQuiesceGroup,
		VerificationRequirementRequired,
	)
	scenario.desired = &plan

	t.Log("Commit a valid reduced membership topology")
	scenario.runUntil("apply retirement", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{Generation: 2, Replicas: cloneReplicaMemberships(base.Replicas[:1])}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)

	t.Log("Record serving failure as workflow health while preserving membership history")
	scenario.runUntil("observe terminal verification failure", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	if scenario.status.Membership.Observed.Transition == nil ||
		scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhaseCommitted {
		t.Fatalf("serving failure rewrote membership outcome: %#v", scenario.status.Membership)
	}
	if scenario.status.Transition.Verification.Phase != VerificationPhaseFailed {
		t.Fatalf("verification failure was not recorded independently: %#v", scenario.status.Transition.Verification)
	}
	if len(scenario.traffic.observation.Admitted) != 0 {
		t.Fatalf("failed topology became routable: %#v", scenario.traffic.observation.Admitted)
	}

	t.Log("Reassert the accepted fail-closed traffic target after routing drifts while blocked")
	trafficCalls := scenario.traffic.applyCalls
	scenario.traffic.observation.Admitted = cloneReplicaMemberships(committed.Replicas)
	scenario.mustReconcile("remove routing drift after verification failure")
	if scenario.traffic.applyCalls != trafficCalls+1 || len(scenario.traffic.observation.Admitted) != 0 {
		t.Fatalf(
			"blocked transition did not preserve fail-closed traffic: calls=%d observation=%#v",
			scenario.traffic.applyCalls,
			scenario.traffic.observation,
		)
	}
}

func TestCoordinatorMaintainsAcceptedTrafficAfterNewerTargetIsRejected(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("rejected-admission", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("apply growth", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.traffic.rejectNext = &Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "AdmissionRejected",
		Message:        "runtime rejected the committed traffic target",
	}

	t.Log("Block after the new committed-traffic revision is definitively rejected")
	scenario.runUntil("reject committed traffic", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	trafficCalls := scenario.traffic.applyCalls
	accepted := scenario.status.Traffic.Accepted
	if accepted == nil ||
		scenario.status.Traffic.Desired.ControlRevision <= accepted.ControlRevision ||
		accepted.ControlRevision != scenario.traffic.observation.AppliedRevision {
		t.Fatalf("rejected traffic target replaced the accepted target: %#v", scenario.status.Traffic)
	}

	t.Log("Keep the rejected target durable for diagnosis without replaying it")
	for iteration := 1; iteration <= 3; iteration++ {
		scenario.mustReconcile("observe blocked rejected target")
	}
	if scenario.traffic.applyCalls != trafficCalls {
		t.Fatalf("definitively rejected traffic target was replayed: %d applications", scenario.traffic.applyCalls)
	}

	t.Log("Restart and repair routing drift by replaying the older accepted fail-closed target")
	scenario.rebuildCoordinator()
	scenario.traffic.observation.Admitted = cloneReplicaMemberships(committed.Replicas)
	scenario.mustReconcile("reassert accepted traffic after restart")
	if scenario.traffic.applyCalls != trafficCalls+1 ||
		!sameMemberships(scenario.traffic.observation.Admitted, base.Replicas) ||
		scenario.traffic.lastTarget.ControlRevision != accepted.ControlRevision {
		t.Fatalf(
			"accepted traffic target did not repair drift: calls=%d observation=%#v target=%#v",
			scenario.traffic.applyCalls,
			scenario.traffic.observation,
			scenario.traffic.lastTarget,
		)
	}
}

func TestCoordinatorMaintainsAcceptedCapacityAfterNewerTargetIsRejected(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	growth := growPlan("establish-capacity", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &growth
	scenario.runUntil("apply growth", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	expanded := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, expanded)
	scenario.runUntil("complete growth", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})

	t.Log("Commit retirement, then reject its newer exact-release capacity target")
	retirement := retirePlan(
		"reject-capacity-release",
		joining.ReplicaID,
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &retirement
	scenario.runUntil("apply retirement", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 2
	})
	scenario.capacity.rejectNext = &Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "ReleaseRejected",
		Message:        "workload manager rejected the release target",
	}
	contracted := MembershipTopology{Generation: 3, Replicas: cloneReplicaMemberships(base.Replicas)}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, contracted)
	scenario.runUntil("reject capacity release", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	accepted := scenario.status.Capacity.Accepted
	if accepted == nil ||
		scenario.status.Capacity.Desired.ControlRevision <= accepted.ControlRevision ||
		accepted.ControlRevision != scenario.capacity.observation.AppliedRevision {
		t.Fatalf("rejected capacity target replaced the accepted target: %#v", scenario.status.Capacity)
	}

	t.Log("Repair retained-capacity drift with the older accepted target")
	for index := range scenario.capacity.observation.Allocations {
		if scenario.capacity.observation.Allocations[index].Incarnation.ReplicaID == base.Replicas[0].ReplicaID {
			scenario.capacity.observation.Allocations[index].Available = false
		}
	}
	capacityCalls := scenario.capacity.applyCalls
	scenario.mustReconcile("reassert accepted capacity")
	allocation, found := allocationByID(scenario.capacity.observation, base.Replicas[0].ReplicaID)
	if !found || !allocation.Available || scenario.capacity.applyCalls != capacityCalls+1 ||
		scenario.capacity.lastTarget.ControlRevision != accepted.ControlRevision {
		t.Fatalf(
			"accepted capacity target did not repair drift: calls=%d observation=%#v target=%#v",
			scenario.capacity.applyCalls,
			scenario.capacity.observation,
			scenario.capacity.lastTarget,
		)
	}
}

func TestCoordinatorFailsClosedOnInvalidCommittedMembership(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("invalid-commit", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.runUntil("apply growth", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})

	t.Log("Return a correlated commit that illegally remaps a retained native member")
	invalid := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	invalid.Replicas[0].NativeMembers = []NativeMemberID{"unexpected-remap"}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, invalid)
	scenario.runUntil("reject invalid committed topology", func(s *coordinatorScenario) bool {
		return s.status.Transition != nil && s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	if scenario.status.Membership.Observed.Transition == nil ||
		scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhaseCommitted ||
		scenario.status.Transition.Failure.Reason != "InvalidCommittedTopology" {
		t.Fatalf("invalid commit did not become unknown and fail closed: %#v", scenario.status.Transition)
	}
	if scenario.verifier.calls != 0 || containsMembership(scenario.traffic.observation.Admitted, joining) {
		t.Fatalf("invalid topology reached verification or traffic: events=%v traffic=%#v", scenario.events, scenario.traffic.observation)
	}
}

func TestCoordinatorRerunsVerificationWhenProofWasNotPersisted(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("rerun-verification", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementRequired)
	scenario.desired = &plan
	scenario.runUntil("apply growth", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("persist pending verification", func(s *coordinatorScenario) bool {
		return s.status.Transition.Verification.Phase == VerificationPhasePending
	})
	persistedBeforeProbe := cloneStatus(scenario.status)

	t.Log("Run a successful check but simulate a controller crash before persisting its proof")
	first, err := scenario.coordinator.Reconcile(
		context.Background(),
		scenario.groupID,
		scenario.desired,
		persistedBeforeProbe,
	)
	if err != nil || first.Status.Transition.Verification.Phase != VerificationPhasePassed {
		t.Fatalf("first serving verification failed: result=%#v error=%v", first, err)
	}

	t.Log("Restart from the old durable status and safely repeat the same topology check")
	second, err := scenario.coordinator.Reconcile(
		context.Background(),
		scenario.groupID,
		scenario.desired,
		persistedBeforeProbe,
	)
	if err != nil || second.Status.Transition.Verification.Phase != VerificationPhasePassed {
		t.Fatalf("repeated serving verification failed: result=%#v error=%v", second, err)
	}
	if scenario.verifier.calls != 2 {
		t.Fatalf("expected safely repeated verification, got %d calls", scenario.verifier.calls)
	}
	if *first.Status.Transition.Verification.Proof != *second.Status.Transition.Verification.Proof {
		t.Fatal("repeated verification produced a different topology-bound proof")
	}
}

func TestCoordinatorRejectsStaleExactReleaseFence(t *testing.T) {
	base := engineTopology(1, 2)
	retiringReplicaID := base.Replicas[1].ReplicaID
	scenario := newCoordinatorScenario(t, base)
	plan := retirePlan(
		"stale-release",
		retiringReplicaID,
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &plan
	scenario.runUntil("apply retirement", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{Generation: 2, Replicas: cloneReplicaMemberships(base.Replicas[:1])}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("persist exact release target", func(s *coordinatorScenario) bool {
		return s.status.Capacity.Desired != nil &&
			len(s.status.Capacity.Desired.ReleaseFences) == 1 &&
			s.capacity.observation.AppliedRevision < s.status.Capacity.Desired.ControlRevision
	})

	t.Log("Replace the selected Pod after authorization but before the workload manager applies it")
	replacement := replicaIncarnation(1)
	replacement.RuntimeIncarnation = testReplacementRuntime
	replacement.CapacityRefs[0].UID = testReplacementPodUID
	for index := range scenario.capacity.observation.Allocations {
		if scenario.capacity.observation.Allocations[index].Incarnation.ReplicaID == retiringReplicaID {
			scenario.capacity.observation.Allocations[index].Incarnation = replacement
		}
	}

	t.Log("Refuse the stale UID fence instead of deleting the healthy replacement")
	scenario.mustReconcile("apply stale release target")
	if scenario.status.Transition.Outcome != TransitionOutcomeBlocked ||
		scenario.status.Transition.Failure.Reason != "StaleReleaseFence" {
		t.Fatalf("stale release did not fail closed: %#v", scenario.status.Transition)
	}
	if _, found := allocationByID(scenario.capacity.observation, retiringReplicaID); !found {
		t.Fatal("stale authorization deleted replacement capacity")
	}
}

func TestCoordinatorReleaseFenceCoversEveryPodInReplicaAllocation(t *testing.T) {
	base := engineTopology(1, 2)
	retiringReplicaID := base.Replicas[1].ReplicaID
	scenario := newCoordinatorScenario(t, base)
	multiPodIncarnation := replicaIncarnation(1)
	multiPodIncarnation.CapacityRefs = append(
		multiPodIncarnation.CapacityRefs,
		CapacityRef{Name: "worker-1-peer", UID: "pod-uid-1-peer-v1"},
	)
	for index := range scenario.capacity.observation.Allocations {
		if scenario.capacity.observation.Allocations[index].Incarnation.ReplicaID == retiringReplicaID {
			scenario.capacity.observation.Allocations[index].Incarnation = multiPodIncarnation
			scenario.status.Capacity.Observed.Allocations[index].Incarnation = multiPodIncarnation
		}
	}
	for index := range scenario.status.Registry.Replicas {
		if scenario.status.Registry.Replicas[index].ReplicaID == retiringReplicaID {
			current := cloneReplicaIncarnation(multiPodIncarnation)
			scenario.status.Registry.Replicas[index].Current = &current
		}
	}
	plan := retirePlan(
		"multi-pod-release",
		retiringReplicaID,
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &plan
	scenario.runUntil("apply multi-Pod retirement", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{Generation: 2, Replicas: cloneReplicaMemberships(base.Replicas[:1])}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)

	t.Log("Persist one indivisible release fence containing every Pod UID in the logical replica")
	scenario.runUntil("derive multi-Pod release fence", func(s *coordinatorScenario) bool {
		return s.status.Capacity.Desired != nil && len(s.status.Capacity.Desired.ReleaseFences) == 1
	})
	fence := scenario.status.Capacity.Desired.ReleaseFences[0]
	if len(fence.CapacityRefs) != 2 || !sameCapacityRefs(fence.CapacityRefs, multiPodIncarnation.CapacityRefs) {
		t.Fatalf("release fence did not cover the complete replica allocation: %#v", fence)
	}
}

func TestCoordinatorFailsClosedOnUncorrelatedTopologyChange(t *testing.T) {
	scenario := newCoordinatorScenario(t, engineTopology(1, 1))
	scenario.membership.topology = engineTopology(2, 1)

	t.Log("Observe an engine topology generation that no durable transition owns")
	err := scenario.reconcile("reconcile unexpected topology")
	if err == nil || !strings.Contains(err.Error(), "without a resolved transition") {
		t.Fatalf("expected fail-closed topology error, got %v", err)
	}
}

func TestCoordinatorRejectsMutationOfTerminalMembershipResult(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("immutable-terminal-result", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("apply growth target", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	transitionID := scenario.status.Membership.Desired.TransitionID
	scenario.membership.commit(transitionID, committed)
	scenario.runUntil("complete growth", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
	target := *scenario.status.Membership.Desired
	scenario.membership.transitions[transitionID] = MembershipTransitionObservation{
		TransitionID:    transitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           MembershipTransitionPhaseRejected,
		Failure: &Failure{
			Classification: FailureClassificationTerminal,
			Reason:         "RewrittenHistory",
		},
	}

	t.Log("Reject an adapter that rewrites an already committed terminal result")
	err := scenario.reconcile("observe mutated terminal result")
	if err == nil || !strings.Contains(err.Error(), "changed an immutable terminal result") {
		t.Fatalf("expected immutable-terminal error, got %v", err)
	}
}

func TestCoordinatorRejectsForgottenTerminalMembershipResult(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	plan := growPlan("retained-terminal-result", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan
	scenario.runUntil("apply growth target", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := MembershipTopology{
		Generation: 2,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			cloneReplicaMembership(joining),
		),
	}
	transitionID := scenario.status.Membership.Desired.TransitionID
	scenario.membership.commit(transitionID, committed)
	scenario.runUntil("complete growth", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
	delete(scenario.membership.transitions, transitionID)

	t.Log("Reject time-based expiry that recreates ambiguity after a long controller outage")
	err := scenario.reconcile("observe forgotten terminal result")
	if err == nil || !strings.Contains(err.Error(), "forgot a previously observed transition") {
		t.Fatalf("expected terminal-retention error, got %v", err)
	}
}

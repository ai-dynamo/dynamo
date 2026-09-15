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
	capacityTarget := scenario.capacity.lastTarget
	if capacityTarget == nil {
		t.Fatal("joining capacity target was not retained")
	}
	joiningCapacity := capacityTarget.Replicas[len(capacityTarget.Replicas)-1]
	if joiningCapacity.Bootstrap == nil ||
		joiningCapacity.Bootstrap.Mode != BootstrapModeJoin ||
		joiningCapacity.Bootstrap.BaseTopologyGeneration != 1 ||
		!slices.Equal(joiningCapacity.Bootstrap.NativeMembers, joining.NativeMembers) {
		t.Fatalf("joining bootstrap lacks resolved topology and native identity: %#v", joiningCapacity.Bootstrap)
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
	replacement.RuntimeIncarnation = "runtime-1-v2"
	replacement.CapacityRefs[0].UID = "pod-uid-1-v2"
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

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
	"slices"
	"testing"
)

func TestCoordinatorPersistsEveryDirectiveBeforeCallingItsAdapter(t *testing.T) {
	scenario := newCoordinatorScenario(t, engineTopology(1, 2))
	plan := retirePlan(
		"persist-before-effects",
		"replica-1",
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &plan

	t.Log("Persist the immutable transition before changing traffic")
	scenario.mustReconcile("start transition")
	if scenario.status.Transition == nil || len(scenario.events) != 0 {
		t.Fatalf("transition was not isolated from external effects: status=%#v events=%v", scenario.status, scenario.events)
	}

	t.Log("Persist authoritative plan validation before capacity or traffic prework")
	scenario.mustReconcile("validate resolved membership plan")
	if scenario.status.Transition.PlanPreflight.Evidence == nil || len(scenario.events) != 0 {
		t.Fatalf("plan validation was not persisted before prework: status=%#v events=%v", scenario.status.Transition.PlanPreflight, scenario.events)
	}
	request := scenario.membership.lastPlanValidation
	if request == nil || request.PlanDigest != scenario.status.Transition.PlanPreflight.SubjectDigest ||
		!sameTopology(request.BaseTopology, engineTopology(1, 2)) || !sameResolvedPlan(request.Plan, plan) {
		t.Fatalf("adapter did not receive the coordinator-owned plan digest: %#v", request)
	}

	t.Log("Persist the absolute traffic target before applying it")
	scenario.mustReconcile("derive traffic target")
	if scenario.status.Traffic.Desired == nil || scenario.traffic.applyCalls != 0 {
		t.Fatalf("traffic target was not persisted first: status=%#v calls=%d", scenario.status.Traffic, scenario.traffic.applyCalls)
	}
	scenario.mustReconcile("apply persisted traffic target")
	if scenario.traffic.applyCalls != 1 {
		t.Fatalf("persisted traffic target was not applied exactly once: %d", scenario.traffic.applyCalls)
	}
	if scenario.status.Traffic.Accepted != nil {
		t.Fatal("traffic target was promoted before its applied revision was observed")
	}

	t.Log("Persist the exact membership target before applying it")
	scenario.mustReconcile("freeze membership target")
	if scenario.status.Membership.Desired == nil || scenario.membership.applyCalls != 0 {
		t.Fatalf("membership target was not persisted first: status=%#v calls=%d", scenario.status.Membership, scenario.membership.applyCalls)
	}
	if scenario.status.Traffic.Accepted == nil ||
		scenario.status.Traffic.Accepted.ControlRevision != scenario.status.Traffic.Desired.ControlRevision {
		t.Fatalf("observed traffic target was not promoted to accepted: %#v", scenario.status.Traffic)
	}
	scenario.mustReconcile("apply persisted membership target")
	if scenario.membership.applyCalls != 1 {
		t.Fatalf("persisted membership target was not applied: %d", scenario.membership.applyCalls)
	}
}

func TestCoordinatorRejectsUnsupportedPlanBeforePrework(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	scenario.capacity.planned[joining.ReplicaID] = replicaIncarnation(1)
	rejection := Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "UnsupportedPlan",
		Message:        "adapter cannot execute the resolved operation shape",
	}
	scenario.membership.planRejection = &rejection
	plan := growPlan("unsupported-growth", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    "slot-1",
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan

	t.Log("Persist a correlated authoritative rejection without reserving or allocating capacity")
	scenario.runUntil("reject unsupported plan", func(s *coordinatorScenario) bool {
		return s.status.Transition != nil && s.status.Transition.Outcome == TransitionOutcomeBlocked
	})
	if scenario.membership.planValidationCalls != 1 ||
		scenario.membership.targetValidationCalls != 0 ||
		scenario.membership.applyCalls != 0 ||
		scenario.capacity.applyCalls != 0 ||
		scenario.traffic.applyCalls != 0 {
		t.Fatalf("unsupported plan caused prework: events=%v", scenario.events)
	}
	if _, found := scenario.status.Registry.Find(joining.ReplicaID); found {
		t.Fatal("unsupported plan reserved a logical replica")
	}
	preflight := scenario.status.Transition.PlanPreflight
	if preflight.TransitionID != scenario.status.Transition.Spec.ID ||
		preflight.SubjectDigest == "" || preflight.Rejection == nil {
		t.Fatalf("plan rejection lacks durable correlation: %#v", preflight)
	}
}

func TestCoordinatorValidatesExactTargetBeforeMembershipApply(t *testing.T) {
	base := engineTopology(1, 1)
	scenario := newCoordinatorScenario(t, base)
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	rejection := Failure{
		Classification: FailureClassificationTerminal,
		Reason:         "UnsupportedRuntimeIdentity",
		Message:        "adapter rejected the frozen joining process",
	}
	scenario.membership.targetRejection = &rejection
	plan := growPlan("rejected-target", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan

	t.Log("Converge joining capacity, then reject the exact runtime-bound target before applying membership")
	scenario.runUntil("reject exact membership target", func(s *coordinatorScenario) bool {
		return s.status.Transition != nil && s.status.Transition.Outcome == TransitionOutcomeReverting
	})
	if scenario.membership.planValidationCalls != 1 ||
		scenario.membership.targetValidationCalls != 1 ||
		scenario.membership.applyCalls != 0 {
		t.Fatalf("target validation boundary was not respected: events=%v", scenario.events)
	}
	preflight := scenario.status.Transition.TargetPreflight
	if preflight.TransitionID == "" || preflight.ControlRevision <= 0 ||
		preflight.SubjectDigest == "" || preflight.Rejection == nil {
		t.Fatalf("target rejection lacks durable correlation: %#v", preflight)
	}
	if scenario.status.Membership.Desired != nil {
		t.Fatal("rejected target became desired membership")
	}

	t.Log("Roll back only the preparatory allocation after definitive target rejection")
	scenario.runUntil("complete target-preflight rollback", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeRolledBack
	})
	if _, found := allocationByID(scenario.capacity.observation, joining.ReplicaID); found {
		t.Fatal("target-preflight rollback retained joining capacity")
	}
}

func TestCoordinatorWaitsForCompleteDrainBeforeMembershipMutation(t *testing.T) {
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	scenario.traffic.autoDrain = false
	scenario.traffic.confirmInactive = false
	plan := retirePlan(
		"wait-for-drain",
		"replica-1",
		TrafficRequirementKeepServing,
		VerificationRequirementNone,
	)
	scenario.desired = &plan

	t.Log("Withdraw the selected replica while its in-flight traffic remains draining")
	scenario.runUntil("apply withdrawal target", func(s *coordinatorScenario) bool {
		return s.traffic.applyCalls == 1
	})
	for iteration := 1; iteration <= 3; iteration++ {
		scenario.mustReconcile("wait for drain completion")
	}
	if scenario.membership.applyCalls != 0 {
		t.Fatalf("membership changed before drain completed: %d applications", scenario.membership.applyCalls)
	}
	if scenario.traffic.applyCalls < 2 {
		t.Fatalf("accepted traffic target was not reasserted while unconverged: %d applications", scenario.traffic.applyCalls)
	}
	if len(scenario.status.Traffic.Desired.Drain) != 1 ||
		scenario.status.Traffic.Desired.Drain[0].Mode != TrafficDrainModeGraceful {
		t.Fatalf("planned retirement did not require graceful drain: %#v", scenario.status.Traffic.Desired.Drain)
	}
	if !sameMemberships(scenario.traffic.observation.Draining, base.Replicas[1:]) {
		t.Fatalf("selected replica is not observably draining: %#v", scenario.traffic.observation)
	}

	t.Log("Publish durable drain completion and allow membership application")
	scenario.traffic.observation.Drained = cloneReplicaMemberships(scenario.traffic.observation.Draining)
	scenario.traffic.observation.Draining = nil
	scenario.runUntil("apply after drain", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
}

func TestCoordinatorWaitsForCompleteReplicaAvailability(t *testing.T) {
	scenario := newCoordinatorScenario(t, engineTopology(1, 1))
	joining := engineReplica(1)
	joiningIncarnation := replicaIncarnation(1)
	scenario.capacity.planned[joining.ReplicaID] = joiningIncarnation
	scenario.capacity.available = false
	plan := growPlan("wait-for-capacity", ReplicaTarget{
		ReplicaID: joining.ReplicaID,
		SlotID:    joiningIncarnation.SlotID,
		Bootstrap: BootstrapModeJoin,
	}, VerificationRequirementNone)
	scenario.desired = &plan

	t.Log("Allocate the complete logical replica while its physical capacity remains unavailable")
	scenario.runUntil("apply joining capacity target", func(s *coordinatorScenario) bool {
		return s.capacity.applyCalls == 1
	})
	for iteration := 1; iteration <= 3; iteration++ {
		scenario.mustReconcile("wait for complete replica availability")
	}
	if scenario.membership.applyCalls != 0 {
		t.Fatalf("membership changed before capacity became available: %d applications", scenario.membership.applyCalls)
	}
	if scenario.capacity.applyCalls < 2 {
		t.Fatalf("accepted capacity target was not reasserted while unconverged: %d applications", scenario.capacity.applyCalls)
	}
	record, found := scenario.status.Registry.Find(joining.ReplicaID)
	if !found || record.Current != nil {
		t.Fatalf("unavailable capacity was frozen as a usable incarnation: %#v", record)
	}

	t.Log("Mark every Pod in the replica allocation available and continue")
	for index := range scenario.capacity.observation.Allocations {
		if scenario.capacity.observation.Allocations[index].Incarnation.ReplicaID == joining.ReplicaID {
			scenario.capacity.observation.Allocations[index].Available = true
		}
	}
	scenario.runUntil("apply after capacity availability", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
}

func TestCoordinatorRestoresStableReplicaWithNewPhysicalIncarnation(t *testing.T) {
	base := engineTopology(2, 1)
	scenario := newCoordinatorScenario(t, base)
	excluded := engineReplica(1)
	excludedIncarnation := replicaIncarnation(1)
	scenario.status.Registry.Replicas = append(scenario.status.Registry.Replicas, ReplicaRecord{
		ReplicaID: excluded.ReplicaID,
		SlotID:    excludedIncarnation.SlotID,
		History: []ReplicaHistoryEntry{{
			TopologyGeneration: 1,
			Incarnation:        cloneReplicaIncarnation(excludedIncarnation),
			NativeMembers:      cloneNativeMembers(excluded.NativeMembers),
		}},
	})
	replacement := cloneReplicaIncarnation(excludedIncarnation)
	replacement.RuntimeIncarnation = testReplacementRuntime
	replacement.CapacityRefs[0].UID = testReplacementPodUID
	scenario.capacity.planned[excluded.ReplicaID] = replacement
	plan := ResolvedPlan{
		ID:                      "restore-fixed-slot",
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      TrafficRequirementQuiesceGroup,
		VerificationRequirement: VerificationRequirementRequired,
		Change: ResolvedChange{
			Kind: PlanKindRestore,
			Restore: &RestoreChange{Replicas: []RestorationTarget{{
				ReplicaTarget: ReplicaTarget{
					ReplicaID:     excluded.ReplicaID,
					SlotID:        excludedIncarnation.SlotID,
					Bootstrap:     BootstrapModeRestoreFixedSlot,
					NativeMembers: cloneNativeMembers(excluded.NativeMembers),
				},
			}}},
		},
	}
	scenario.desired = &plan

	t.Log("Provision replacement capacity in the stable slot and apply exact native-member restoration")
	scenario.runUntil("apply restoration", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committedReplica := cloneReplicaMembership(excluded)
	committedReplica.RuntimeIncarnation = replacement.RuntimeIncarnation
	committed := MembershipTopology{
		Generation: 3,
		Replicas: append(
			cloneReplicaMemberships(base.Replicas),
			committedReplica,
		),
	}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("finish restoration", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})

	t.Log("Keep historical membership while making only the replacement incarnation current")
	record, found := scenario.status.Registry.Find(excluded.ReplicaID)
	if !found || record.Current == nil || !sameIncarnation(*record.Current, replacement) {
		t.Fatalf("replacement did not become canonical: %#v", record)
	}
	if len(record.History) != 1 ||
		!sameIncarnation(record.History[0].Incarnation, excludedIncarnation) ||
		!slices.Equal(record.History[0].NativeMembers, excluded.NativeMembers) {
		t.Fatalf("restoration lost excluded membership history: %#v", record.History)
	}
}

func TestCoordinatorRequiresTerminalTrafficEvidenceForFailedMember(t *testing.T) {
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	scenario.traffic.autoDrain = false
	scenario.traffic.confirmInactive = false
	plan := ResolvedPlan{
		ID:                      "remove-failed-member",
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      TrafficRequirementKeepServing,
		VerificationRequirement: VerificationRequirementRequired,
		Change: ResolvedChange{
			Kind:              PlanKindReduceToSurvivors,
			ReduceToSurvivors: &ReduceToSurvivorsChange{Survivors: []ReplicaID{"replica-0"}},
		},
	}
	scenario.desired = &plan

	t.Log("Withdraw the failed identity, but withhold terminal inactivity evidence")
	scenario.runUntil("apply failed-member withdrawal", func(s *coordinatorScenario) bool {
		return s.traffic.applyCalls == 1
	})
	for iteration := 1; iteration <= 3; iteration++ {
		scenario.mustReconcile("wait for failed-member inactivity confirmation")
	}
	if scenario.membership.applyCalls != 0 || len(scenario.traffic.observation.Draining) != 0 ||
		len(scenario.traffic.observation.Drained) != 0 {
		t.Fatalf("membership advanced without terminal inactive evidence: %#v", scenario.traffic.observation)
	}
	if !sameMemberships(scenario.traffic.observation.Admitted, base.Replicas[:1]) {
		t.Fatalf("failed identity remained routable: %#v", scenario.traffic.observation.Admitted)
	}
	if len(scenario.status.Traffic.Desired.Drain) != 1 ||
		scenario.status.Traffic.Desired.Drain[0].Mode != TrafficDrainModeConfirmInactive {
		t.Fatalf("survivor recovery did not require inactive-member evidence: %#v", scenario.status.Traffic.Desired.Drain)
	}

	t.Log("Publish durable inactivity confirmation and allow survivor reduction")
	scenario.traffic.observation.Drained = cloneReplicaMemberships(base.Replicas[1:])
	scenario.runUntil("apply survivor reduction", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})

	t.Log("Commit the exact survivor topology, release failed capacity, and verify survivors")
	committed := MembershipTopology{Generation: 2, Replicas: cloneReplicaMemberships(base.Replicas[:1])}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("finish survivor reduction", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})
	if scenario.status.Membership.Observed.Transition == nil ||
		scenario.status.Membership.Observed.Transition.Phase != MembershipTransitionPhaseCommitted ||
		scenario.status.Transition.Verification.Phase != VerificationPhasePassed {
		t.Fatalf("survivor reduction lost independent outcomes: %#v", scenario.status.Transition)
	}
}

func TestCoordinatorRemapsNativeMembersWithoutChangingPhysicalIdentity(t *testing.T) {
	base := engineTopology(1, 2)
	scenario := newCoordinatorScenario(t, base)
	originalRegistry := cloneRegistry(scenario.status.Registry)
	plan := ResolvedPlan{
		ID:                      "remap-native-members",
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   ProcessLifecycleOwnerEngine,
		TrafficRequirement:      TrafficRequirementQuiesceGroup,
		VerificationRequirement: VerificationRequirementRequired,
		Change: ResolvedChange{
			Kind: PlanKindRemap,
			Remap: &RemapChange{Membership: []ReplicaNativeMembership{
				{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []NativeMemberID{"remapped-0"}},
				{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []NativeMemberID{"remapped-1"}},
			}},
		},
	}
	scenario.desired = &plan

	t.Log("Quiesce the unchanged physical incarnations before native-member remapping")
	scenario.runUntil("apply remap", func(s *coordinatorScenario) bool {
		return s.membership.applyCalls == 1
	})
	committed := cloneTopology(base)
	committed.Generation = 2
	committed.Replicas[0].NativeMembers = []NativeMemberID{"remapped-0"}
	committed.Replicas[1].NativeMembers = []NativeMemberID{"remapped-1"}
	scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, committed)
	scenario.runUntil("finish remap", func(s *coordinatorScenario) bool {
		return s.status.Transition.Outcome == TransitionOutcomeCompleted
	})

	t.Log("Confirm remapping changed no stable slot, runtime, or Pod UID")
	for _, membership := range committed.Replicas {
		record, found := scenario.status.Registry.Find(membership.ReplicaID)
		original, existed := originalRegistry.Find(membership.ReplicaID)
		if !found || !existed || record.Current == nil || original.Current == nil ||
			!sameIncarnation(*record.Current, *original.Current) {
			t.Fatalf("remap changed physical identity for %q: %#v", membership.ReplicaID, record)
		}
	}
}

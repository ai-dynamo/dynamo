/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"
	"reflect"
	"testing"
)

func TestNewGroupStatusRequiresCoherentInitialObservations(t *testing.T) {
	for _, name := range []string{
		"coherent", "extra allocation", "missing allocation", "extra admitted process",
		"missing admitted process", "wrong admitted incarnation", "draining committed process",
		"packed survivor allocation", "unrelated drained tombstone",
	} {
		t.Run(name, func(t *testing.T) {
			t.Log("Observe one already-running world before adopting it as recovery authority")
			topology := engineTopology(1, 2)
			capacity := capacityForTopology(topology)
			traffic := TrafficObservation{Admitted: cloneReplicaMemberships(topology.Replicas)}
			valid := false
			switch name {
			case "coherent":
				valid = true
			case "extra allocation":
				capacity.Allocations = append(capacity.Allocations, CapacityAllocation{Incarnation: replicaIncarnation(2)})
			case "missing allocation":
				capacity.Allocations = capacity.Allocations[:1]
			case "extra admitted process":
				traffic.Admitted = append(traffic.Admitted, engineReplica(2))
			case "missing admitted process":
				traffic.Admitted = traffic.Admitted[:1]
			case "wrong admitted incarnation":
				traffic.Admitted[1].Members[0].RuntimeIncarnation = "other-runtime"
			case "draining committed process":
				traffic.Draining = cloneReplicaMemberships(topology.Replicas[1:])
			case "packed survivor allocation":
				topology = packedTopology()
				capacity = capacityForTopology(topology)
				topology.Replicas[1].Members = topology.Replicas[1].Members[:3]
				traffic.Admitted = cloneReplicaMemberships(topology.Replicas)
				valid = true
			case "unrelated drained tombstone":
				traffic.Drained = []ReplicaMembership{engineReplica(2)}
				valid = true
			}
			beforeCapacity := cloneCapacityObservation(capacity)
			beforeTraffic := cloneTrafficObservation(traffic)
			beforeTopology := cloneTopology(topology)

			t.Log("Reject cross-subsystem mismatches without treating allocated inactive processes as committed members")
			_, err := NewGroupStatus(topology, capacity, traffic)
			if (err == nil) != valid {
				t.Fatalf("valid=%t, initialization error=%v", valid, err)
			}
			if !reflect.DeepEqual(beforeCapacity, cloneCapacityObservation(capacity)) ||
				!reflect.DeepEqual(beforeTraffic, cloneTrafficObservation(traffic)) ||
				!reflect.DeepEqual(beforeTopology, cloneTopology(topology)) {
				t.Fatal("initial validation mutated its observations")
			}
		})
	}
}

func TestValidateGroupStatusBoundsObservedRevisions(t *testing.T) {
	for _, authority := range []string{"capacity", "traffic"} {
		for _, revision := range []int64{0, 1, 2} {
			t.Run(fmt.Sprintf("%s/%d", authority, revision), func(t *testing.T) {
				t.Log("Persist revision one without acknowledging or promoting a desired target")
				scenario := newCoordinatorScenario(t, engineTopology(1, 1))
				status := cloneStatus(scenario.status)
				status.ControlRevision = 1
				incarnation := replicaIncarnation(0)
				status.Capacity.Desired = &CapacityTarget{
					ControlRevision: 1, TransitionID: "initial-target", ProfileFingerprint: "profile-v1",
					ProcessLifecycleOwner: ProcessLifecycleOwnerOrchestrator,
					Replicas:              []CapacityReplicaTarget{{ReplicaID: incarnation.ReplicaID, SlotID: incarnation.SlotID, Incarnation: &incarnation}},
				}
				status.Traffic.Desired = &TrafficTarget{
					ControlRevision: 1, TransitionID: "initial-target", TopologyGeneration: 1,
				}
				if authority == "capacity" {
					status.Capacity.Observed.AppliedRevision = revision
				} else {
					status.Traffic.Observed.AppliedRevision = revision
				}
				before := cloneStatus(status)

				t.Log("Reject only future applied revisions and leave recovery authority unchanged")
				err := ValidateGroupStatus(status)
				if (err == nil) != (revision <= status.ControlRevision) {
					t.Fatalf("revision=%d, validation error=%v", revision, err)
				}
				if !reflect.DeepEqual(before, cloneStatus(status)) {
					t.Fatal("checkpoint validation mutated state or promoted an acknowledgment")
				}
			})
		}
	}
}

func TestValidateReleaseFencesRejectsDuplicatePodUIDs(t *testing.T) {
	for _, name := range []string{"within one allocation", "across allocations"} {
		t.Run(name, func(t *testing.T) {
			t.Log("Bind two otherwise complete release references to the same concrete Pod UID")
			fence := ReleaseFence{
				TransitionID: "release", AuthorizingTopologyGeneration: 2,
				ReplicaID: "replica-0", SlotID: "slot-0", CapacityRefs: replicaIncarnation(0).CapacityRefs,
			}
			fences := []ReleaseFence{fence}
			if name == "within one allocation" {
				fences[0].CapacityRefs = append(fences[0].CapacityRefs, fence.CapacityRefs[0])
			} else {
				fence.ReplicaID = "replica-1"
				fence.SlotID = "slot-1"
				fences = append(fences, fence)
			}
			before := cloneReleaseFences(fences)

			t.Log("Refuse repeated physical deletion authority without changing the serialized references")
			if err := validateReleaseFences(fences); err == nil {
				t.Fatal("duplicate Pod UID was accepted as release authority")
			}
			if !reflect.DeepEqual(before, cloneReleaseFences(fences)) {
				t.Fatal("release validation mutated its references")
			}
		})
	}
}

func TestValidateGroupStatusRebindsReleaseAuthority(t *testing.T) {
	for _, name := range []string{"pod UID", "pod name", "slot", "replica", "generation", "duplicate Pod UID"} {
		t.Run(name, func(t *testing.T) {
			t.Log("Commit retirement and persist exact release authority before the first capacity application")
			base := engineTopology(1, 3)
			scenario := newCoordinatorScenario(t, base)
			plan := retirePlan("release-checkpoint", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
			plan.Change.Retire.Replicas = append(plan.Change.Retire.Replicas, "replica-2")
			scenario.desired = &plan
			scenario.runUntil("submit retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
			scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 1))
			scenario.runUntil("persist release target", func(s *coordinatorScenario) bool {
				return s.status.Capacity.Desired != nil && len(s.status.Capacity.Desired.ReleaseFences) == 2
			})
			if err := ValidateGroupStatus(scenario.status); err != nil {
				t.Fatalf("valid checkpoint rejected: %v", err)
			}

			t.Log("Corrupt one serialized fence while leaving the immutable plan and canonical allocations intact")
			fence := &scenario.status.Capacity.Desired.ReleaseFences[0]
			switch name {
			case "pod UID":
				fence.CapacityRefs[0].UID = replicaIncarnation(0).CapacityRefs[0].UID
			case "pod name":
				fence.CapacityRefs[0].Name = "worker-0"
			case "slot":
				fence.SlotID = "slot-0"
			case "replica":
				fence.ReplicaID = "other-replica"
			case "generation":
				fence.AuthorizingTopologyGeneration = base.Generation
			case "duplicate Pod UID":
				fence.CapacityRefs[0].UID = scenario.status.Capacity.Desired.ReleaseFences[1].CapacityRefs[0].UID
			}
			before := cloneStatus(scenario.status)
			if err := ValidateGroupStatus(scenario.status); err == nil {
				t.Fatal("checkpoint admitted corrupt release authority")
			}

			t.Log("Restart in the persist-before-effect window and refuse the corrupted target before any adapter call")
			scenario.rebuildCoordinator()
			calls := scenario.capacity.applyCalls
			if err := scenario.reconcile("reject corrupt persisted release fence"); err == nil {
				t.Fatal("restarted coordinator accepted corrupt release authority")
			}
			if scenario.capacity.applyCalls != calls || !reflect.DeepEqual(before, cloneStatus(scenario.status)) {
				t.Fatal("corrupt checkpoint reached capacity or changed durable authority")
			}
		})
	}
}

func TestValidateGroupStatusPreservesHistoricalReleaseAuthority(t *testing.T) {
	for _, name := range []string{"completed retirement", "new transition", "corrupt older desired", "corrupt older accepted"} {
		t.Run(name, func(t *testing.T) {
			t.Log("Finish retirement and archive the released allocation while retaining its exact target and receipt")
			base := engineTopology(1, 2)
			scenario := newCoordinatorScenario(t, base)
			plan := retirePlan("historical-release", "replica-1", TrafficRequirementKeepServing, VerificationRequirementNone)
			scenario.desired = &plan
			scenario.runUntil("submit retirement", func(s *coordinatorScenario) bool { return s.membership.applyCalls == 1 })
			scenario.membership.commit(scenario.status.Membership.Desired.TransitionID, engineTopology(2, 1))
			scenario.runUntil("finish retirement", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeCompleted })
			record, found := scenario.status.Registry.Find("replica-1")
			if !found || record.Current != nil || len(record.History) != 1 {
				t.Fatal("released incarnation was not preserved in canonical history")
			}
			if err := ValidateGroupStatus(scenario.status); err != nil {
				t.Fatalf("archived release authority rejected: %v", err)
			}

			t.Log("Keep the older accepted projection verifiable after replacing the completed operation")
			if name != "completed retirement" {
				growth := growPlan("later-growth", ReplicaTarget{ReplicaID: "replica-2", SlotID: "slot-2", Bootstrap: BootstrapModeJoin}, VerificationRequirementNone)
				scenario.desired = &growth
				scenario.mustReconcile("start later transition without discarding accepted release authority")
			}
			if name == "corrupt older desired" {
				scenario.status.Capacity.Desired.ReleaseFences[0].AuthorizingTopologyGeneration = 1
			} else if name == "corrupt older accepted" {
				scenario.status.Capacity.Desired.ControlRevision++
				scenario.status.ControlRevision++
				scenario.status.Capacity.Accepted.ReleaseFences[0].CapacityRefs[0].UID = replicaIncarnation(0).CapacityRefs[0].UID
			}
			valid := name == "completed retirement" || name == "new transition"
			if err := ValidateGroupStatus(scenario.status); (err == nil) != valid {
				t.Fatalf("valid=%t, release authority validation error=%v", valid, err)
			}
		})
	}
}

func TestValidateGroupStatusRebindsRollbackReleaseAuthority(t *testing.T) {
	for _, partial := range []bool{false, true} {
		t.Run(fmt.Sprintf("partial-allocation=%t", partial), func(t *testing.T) {
			t.Log("Reject growth after creating an uncommitted allocation")
			scenario := newCoordinatorScenario(t, engineTopology(1, 1))
			joining := replicaIncarnation(1)
			scenario.capacity.planned[joining.ReplicaID] = joining
			rejection := Failure{Classification: FailureClassificationTerminal, Reason: "Rejected"}
			if partial {
				scenario.capacity.rejectNext = &rejection
				scenario.capacity.partial = []CapacityAllocation{{Incarnation: joining}}
			} else {
				scenario.membership.applyRejection = &rejection
			}
			plan := growPlan("rollback-release", ReplicaTarget{ReplicaID: joining.ReplicaID, SlotID: joining.SlotID, Bootstrap: BootstrapModeJoin}, VerificationRequirementNone)
			scenario.desired = &plan
			scenario.runUntil("persist rollback release", func(s *coordinatorScenario) bool {
				return s.status.Transition != nil && s.status.Transition.Outcome == TransitionOutcomeReverting &&
					s.status.Capacity.Desired != nil && len(s.status.Capacity.Desired.ReleaseFences) == 1
			})
			if err := ValidateGroupStatus(scenario.status); err != nil {
				t.Fatalf("canonical rollback fence rejected: %v", err)
			}
			valid := cloneStatus(scenario.status)

			t.Log("Refuse a rollback fence forged to name the retained replica's Pod UID before dispatch")
			scenario.status.Capacity.Desired.ReleaseFences[0].CapacityRefs[0].UID = replicaIncarnation(0).CapacityRefs[0].UID
			scenario.rebuildCoordinator()
			calls := scenario.capacity.applyCalls
			if err := scenario.reconcile("reject forged rollback authority"); err == nil || scenario.capacity.applyCalls != calls {
				t.Fatal("forged rollback authority reached the capacity adapter")
			}

			t.Log("Resume the valid checkpoint, release the joiner, and verify replay after its registry record is discarded")
			scenario.status = valid
			scenario.runUntil("finish valid rollback", func(s *coordinatorScenario) bool { return s.status.Transition.Outcome == TransitionOutcomeRolledBack })
			scenario.rebuildCoordinator()
			scenario.mustReconcile("validate released rollback receipt after restart")
			if err := ValidateGroupStatus(scenario.status); err != nil {
				t.Fatalf("released rollback authority became unverifiable: %v", err)
			}
		})
	}
}

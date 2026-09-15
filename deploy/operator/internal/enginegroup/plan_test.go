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
	"strings"
	"testing"
)

func TestResolvedPlanVariantsProduceExactIdentitySets(t *testing.T) {
	base := engineTopology(1, 2)
	status, err := NewGroupStatus(base, capacityForTopology(base), TrafficObservation{
		Admitted: cloneReplicaMemberships(base.Replicas),
	})
	if err != nil {
		t.Fatalf("construct base status: %v", err)
	}

	// Give the restoration case one excluded stable identity with exact historical native membership.
	excluded := engineReplica(2)
	excludedIncarnation := replicaIncarnation(2)
	status.Registry.Replicas = append(status.Registry.Replicas, ReplicaRecord{
		ReplicaID: excludedIncarnation.ReplicaID,
		SlotID:    excludedIncarnation.SlotID,
		History: []ReplicaHistoryEntry{{
			TopologyGeneration: 1,
			Incarnation:        cloneReplicaIncarnation(excludedIncarnation),
			NativeMembers:      cloneNativeMembers(excluded.NativeMembers),
		}},
	})

	tests := []struct {
		name     string
		plan     ResolvedPlan
		kind     PlanKind
		target   []ReplicaID
		joining  []ReplicaID
		retiring []ReplicaID
	}{
		{
			name: "fresh growth",
			plan: growPlan("grow", ReplicaTarget{
				ReplicaID: "replica-3",
				SlotID:    "slot-3",
				Bootstrap: BootstrapModeJoin,
			}, VerificationRequirementRequired),
			kind:    PlanKindGrow,
			target:  []ReplicaID{"replica-0", "replica-1", "replica-3"},
			joining: []ReplicaID{"replica-3"},
		},
		{
			name: "selected retirement",
			plan: retirePlan(
				"retire",
				"replica-1",
				TrafficRequirementKeepServing,
				VerificationRequirementNone,
			),
			kind:     PlanKindRetire,
			target:   []ReplicaID{"replica-0"},
			retiring: []ReplicaID{"replica-1"},
		},
		{
			name: "survivor reduction",
			plan: ResolvedPlan{
				ID:                      "survivors",
				ProfileFingerprint:      "profile-v1",
				ProcessLifecycleOwner:   ProcessLifecycleOwnerOrchestrator,
				TrafficRequirement:      TrafficRequirementKeepServing,
				RetirementSafety:        RetirementSafetyWithdrawn,
				VerificationRequirement: VerificationRequirementRequired,
				Change:                  &ReduceToSurvivorsChange{Survivors: []ReplicaID{"replica-1"}},
			},
			kind:     PlanKindReduceToSurvivors,
			target:   []ReplicaID{"replica-1"},
			retiring: []ReplicaID{"replica-0"},
		},
		{
			name: "fixed-slot restoration",
			plan: ResolvedPlan{
				ID:                      "restore",
				ProfileFingerprint:      "profile-v1",
				ProcessLifecycleOwner:   ProcessLifecycleOwnerOrchestrator,
				TrafficRequirement:      TrafficRequirementQuiesceGroup,
				VerificationRequirement: VerificationRequirementRequired,
				Change: &RestoreChange{Replicas: []RestorationTarget{{
					ReplicaTarget: ReplicaTarget{
						ReplicaID: excludedIncarnation.ReplicaID,
						SlotID:    excludedIncarnation.SlotID,
						Bootstrap: BootstrapModeRestoreFixedSlot,
					},
					NativeMembers: cloneNativeMembers(excluded.NativeMembers),
				}}},
			},
			kind:    PlanKindRestore,
			target:  []ReplicaID{"replica-0", "replica-1", "replica-2"},
			joining: []ReplicaID{"replica-2"},
		},
		{
			name: "native-member remap",
			plan: ResolvedPlan{
				ID:                      "remap",
				ProfileFingerprint:      "profile-v1",
				ProcessLifecycleOwner:   ProcessLifecycleOwnerEngine,
				TrafficRequirement:      TrafficRequirementQuiesceGroup,
				VerificationRequirement: VerificationRequirementRequired,
				Change: &RemapChange{Membership: []ReplicaNativeMembership{
					{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []NativeMemberID{"new-dp-0"}},
					{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []NativeMemberID{"new-dp-1"}},
				}},
			},
			kind:   PlanKindRemap,
			target: []ReplicaID{"replica-0", "replica-1"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the typed membership change against the canonical base state")
			resolution, err := validateResolvedPlan(base, status.Registry, test.plan)
			if err != nil {
				t.Fatalf("resolve plan: %v", err)
			}
			if resolution.kind != test.kind || !sameReplicaIDs(resolution.targetReplicaIDs, test.target) {
				t.Fatalf("unexpected resolution: %#v", resolution)
			}
			joining := make([]ReplicaID, 0, len(resolution.joiningTargets))
			for _, target := range resolution.joiningTargets {
				joining = append(joining, target.ReplicaID)
			}
			if !sameReplicaIDs(joining, test.joining) ||
				!sameReplicaIDs(resolution.retiringReplicaIDs, test.retiring) {
				t.Fatalf("unexpected joining or retiring identities: %#v", resolution)
			}
		})
	}
}

func TestResolvedPlanRejectsInvalidIdentitySemantics(t *testing.T) {
	base := engineTopology(1, 2)
	status, err := NewGroupStatus(base, capacityForTopology(base), TrafficObservation{
		Admitted: cloneReplicaMemberships(base.Replicas),
	})
	if err != nil {
		t.Fatalf("construct base status: %v", err)
	}

	tests := []struct {
		name      string
		plan      ResolvedPlan
		wantError string
	}{
		{
			name: "growth cannot reuse active identity",
			plan: growPlan("reuse", ReplicaTarget{
				ReplicaID: "replica-1",
				SlotID:    "slot-2",
				Bootstrap: BootstrapModeJoin,
			}, VerificationRequirementNone),
			wantError: "already exists",
		},
		{
			name: "retirement requires exact base identity",
			plan: retirePlan(
				"missing",
				"replica-9",
				TrafficRequirementKeepServing,
				VerificationRequirementNone,
			),
			wantError: "absent from the base topology",
		},
		{
			name: "remap preserves cardinality",
			plan: ResolvedPlan{
				ID:                      "short-remap",
				ProfileFingerprint:      "profile-v1",
				ProcessLifecycleOwner:   ProcessLifecycleOwnerEngine,
				TrafficRequirement:      TrafficRequirementQuiesceGroup,
				VerificationRequirement: VerificationRequirementRequired,
				Change: &RemapChange{Membership: []ReplicaNativeMembership{{
					ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []NativeMemberID{"dp-new"},
				}}},
			},
			wantError: "base cardinality",
		},
		{
			name: "untyped change is impossible",
			plan: ResolvedPlan{
				ID:                      "nil",
				ProfileFingerprint:      "profile-v1",
				ProcessLifecycleOwner:   ProcessLifecycleOwnerEngine,
				TrafficRequirement:      TrafficRequirementKeepServing,
				VerificationRequirement: VerificationRequirementNone,
			},
			wantError: "must not be nil",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Reject the malformed typed plan before any external prework")
			_, err := validateResolvedPlan(base, status.Registry, test.plan)
			if err == nil || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("expected error containing %q, got %v", test.wantError, err)
			}
		})
	}
}

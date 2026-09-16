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

package controller

import (
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// engineGroupStatusFromAPI reconstructs the coordinator's durable state from Kubernetes status.
// status must be non-nil and must have been written by engineGroupStatusToAPI.
func engineGroupStatusFromAPI(status *nvidiacomv1beta1.EngineGroupReconciliationStatus) enginegroup.GroupStatus {
	return enginegroup.GroupStatus{
		ControlRevision: status.ControlRevision,
		Registry:        engineGroupRegistryFromAPI(status.Registry),
		Topologies:      engineGroupTopologyHistoryFromAPI(status.TopologyHistory),
		Capacity:        engineGroupCapacityStatusFromAPI(status.Capacity),
		Traffic:         engineGroupTrafficStatusFromAPI(status.Traffic),
		Membership:      engineGroupMembershipStatusFromAPI(status.Membership),
		Transition:      engineGroupTransitionFromAPI(status.Transition),
	}
}

// engineGroupStatusToAPI serializes every coordinator-owned field needed to resume reconciliation.
func engineGroupStatusToAPI(status enginegroup.GroupStatus) *nvidiacomv1beta1.EngineGroupReconciliationStatus {
	return &nvidiacomv1beta1.EngineGroupReconciliationStatus{
		ControlRevision: status.ControlRevision,
		Registry:        engineGroupRegistryToAPI(status.Registry),
		TopologyHistory: engineGroupTopologyHistoryToAPI(status.Topologies),
		Capacity:        engineGroupCapacityStatusToAPI(status.Capacity),
		Traffic:         engineGroupTrafficStatusToAPI(status.Traffic),
		Membership:      engineGroupMembershipStatusToAPI(status.Membership),
		Transition:      engineGroupTransitionToAPI(status.Transition),
	}
}

func engineGroupRegistryFromAPI(values []nvidiacomv1beta1.EngineGroupReplicaRecordStatus) enginegroup.ReplicaRegistry {
	replicas := make([]enginegroup.ReplicaRecord, 0, len(values))
	for _, value := range values {
		record := enginegroup.ReplicaRecord{
			ReplicaID: enginegroup.ReplicaID(value.ReplicaID),
			SlotID:    enginegroup.CapacitySlotID(value.SlotID),
			Current:   engineGroupIncarnationPointerFromAPI(value.Current),
			History:   make([]enginegroup.ReplicaHistoryEntry, 0, len(value.History)),
		}

		// Preserve every excluded incarnation and its historical native-member correlation.
		for _, entry := range value.History {
			record.History = append(record.History, enginegroup.ReplicaHistoryEntry{
				TopologyGeneration: entry.TopologyGeneration,
				Incarnation:        engineGroupIncarnationFromAPI(entry.Incarnation),
				NativeMembers:      engineGroupNativeMembersFromAPI(entry.NativeMembers),
			})
		}
		replicas = append(replicas, record)
	}
	return enginegroup.ReplicaRegistry{Replicas: replicas}
}

func engineGroupRegistryToAPI(value enginegroup.ReplicaRegistry) []nvidiacomv1beta1.EngineGroupReplicaRecordStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupReplicaRecordStatus, 0, len(value.Replicas))
	for _, value := range value.Replicas {
		record := nvidiacomv1beta1.EngineGroupReplicaRecordStatus{
			ReplicaID: string(value.ReplicaID),
			SlotID:    string(value.SlotID),
			Current:   engineGroupIncarnationPointerToAPI(value.Current),
			History:   make([]nvidiacomv1beta1.EngineGroupReplicaHistoryStatus, 0, len(value.History)),
		}

		// Preserve every excluded incarnation and its historical native-member correlation.
		for _, entry := range value.History {
			record.History = append(record.History, nvidiacomv1beta1.EngineGroupReplicaHistoryStatus{
				TopologyGeneration: entry.TopologyGeneration,
				Incarnation:        engineGroupIncarnationToAPI(entry.Incarnation),
				NativeMembers:      engineGroupNativeMembersToAPI(entry.NativeMembers),
			})
		}
		replicas = append(replicas, record)
	}
	return replicas
}

func engineGroupTopologyHistoryFromAPI(
	value nvidiacomv1beta1.EngineGroupTopologyHistoryStatus,
) enginegroup.TopologyHistory {
	snapshots := make([]enginegroup.MembershipTopology, 0, len(value.Snapshots))
	for _, snapshot := range value.Snapshots {
		snapshots = append(snapshots, engineGroupTopologyFromAPI(snapshot))
	}
	return enginegroup.TopologyHistory{
		CurrentGeneration: value.CurrentGeneration,
		Snapshots:         snapshots,
	}
}

func engineGroupTopologyHistoryToAPI(
	value enginegroup.TopologyHistory,
) nvidiacomv1beta1.EngineGroupTopologyHistoryStatus {
	snapshots := make([]nvidiacomv1beta1.EngineGroupTopologyStatus, 0, len(value.Snapshots))
	for _, snapshot := range value.Snapshots {
		snapshots = append(snapshots, engineGroupTopologyToAPI(snapshot))
	}
	return nvidiacomv1beta1.EngineGroupTopologyHistoryStatus{
		CurrentGeneration: value.CurrentGeneration,
		Snapshots:         snapshots,
	}
}

func engineGroupTopologyFromAPI(value nvidiacomv1beta1.EngineGroupTopologyStatus) enginegroup.MembershipTopology {
	replicas := make([]enginegroup.ReplicaMembership, 0, len(value.Replicas))
	for _, replica := range value.Replicas {
		replicas = append(replicas, engineGroupMembershipFromAPI(replica))
	}
	return enginegroup.MembershipTopology{Generation: value.Generation, Replicas: replicas}
}

func engineGroupTopologyToAPI(value enginegroup.MembershipTopology) nvidiacomv1beta1.EngineGroupTopologyStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupMemberStatus, 0, len(value.Replicas))
	for _, replica := range value.Replicas {
		replicas = append(replicas, engineGroupMembershipToAPI(replica))
	}
	return nvidiacomv1beta1.EngineGroupTopologyStatus{Generation: value.Generation, Replicas: replicas}
}

func engineGroupMembershipFromAPI(value nvidiacomv1beta1.EngineGroupMemberStatus) enginegroup.ReplicaMembership {
	return enginegroup.ReplicaMembership{
		ReplicaID:          enginegroup.ReplicaID(value.ReplicaID),
		RuntimeIncarnation: enginegroup.RuntimeIncarnationID(value.RuntimeIncarnation),
		NativeMembers:      engineGroupNativeMembersFromAPI(value.NativeMembers),
	}
}

func engineGroupMembershipToAPI(value enginegroup.ReplicaMembership) nvidiacomv1beta1.EngineGroupMemberStatus {
	return nvidiacomv1beta1.EngineGroupMemberStatus{
		ReplicaID:          string(value.ReplicaID),
		RuntimeIncarnation: string(value.RuntimeIncarnation),
		NativeMembers:      engineGroupNativeMembersToAPI(value.NativeMembers),
	}
}

func engineGroupMembershipsFromAPI(values []nvidiacomv1beta1.EngineGroupMemberStatus) []enginegroup.ReplicaMembership {
	replicas := make([]enginegroup.ReplicaMembership, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, engineGroupMembershipFromAPI(value))
	}
	return replicas
}

func engineGroupMembershipsToAPI(values []enginegroup.ReplicaMembership) []nvidiacomv1beta1.EngineGroupMemberStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupMemberStatus, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, engineGroupMembershipToAPI(value))
	}
	return replicas
}

func engineGroupIncarnationFromAPI(
	value nvidiacomv1beta1.EngineGroupControlIncarnationStatus,
) enginegroup.ReplicaIncarnation {
	capacityRefs := make([]enginegroup.CapacityRef, 0, len(value.CapacityRefs))
	for _, ref := range value.CapacityRefs {
		capacityRefs = append(capacityRefs, enginegroup.CapacityRef{
			Name: ref.Name,
			UID:  enginegroup.PodUID(ref.UID),
		})
	}
	return enginegroup.ReplicaIncarnation{
		ReplicaID:          enginegroup.ReplicaID(value.ReplicaID),
		SlotID:             enginegroup.CapacitySlotID(value.SlotID),
		RuntimeIncarnation: enginegroup.RuntimeIncarnationID(value.RuntimeIncarnation),
		CapacityRefs:       capacityRefs,
	}
}

func engineGroupIncarnationToAPI(
	value enginegroup.ReplicaIncarnation,
) nvidiacomv1beta1.EngineGroupControlIncarnationStatus {
	capacityRefs := make([]nvidiacomv1beta1.EngineGroupCapacityRef, 0, len(value.CapacityRefs))
	for _, ref := range value.CapacityRefs {
		capacityRefs = append(capacityRefs, nvidiacomv1beta1.EngineGroupCapacityRef{
			Name: ref.Name,
			UID:  types.UID(ref.UID),
		})
	}
	return nvidiacomv1beta1.EngineGroupControlIncarnationStatus{
		ReplicaID:          string(value.ReplicaID),
		SlotID:             string(value.SlotID),
		RuntimeIncarnation: string(value.RuntimeIncarnation),
		CapacityRefs:       capacityRefs,
	}
}

func engineGroupIncarnationPointerFromAPI(
	value *nvidiacomv1beta1.EngineGroupControlIncarnationStatus,
) *enginegroup.ReplicaIncarnation {
	if value == nil {
		return nil
	}
	incarnation := engineGroupIncarnationFromAPI(*value)
	return &incarnation
}

func engineGroupIncarnationPointerToAPI(
	value *enginegroup.ReplicaIncarnation,
) *nvidiacomv1beta1.EngineGroupControlIncarnationStatus {
	if value == nil {
		return nil
	}
	incarnation := engineGroupIncarnationToAPI(*value)
	return &incarnation
}

func engineGroupNativeMembersFromAPI(values []string) []enginegroup.NativeMemberID {
	members := make([]enginegroup.NativeMemberID, 0, len(values))
	for _, value := range values {
		members = append(members, enginegroup.NativeMemberID(value))
	}
	return members
}

func engineGroupNativeMembersToAPI(values []enginegroup.NativeMemberID) []string {
	members := make([]string, 0, len(values))
	for _, value := range values {
		members = append(members, string(value))
	}
	return members
}

func engineGroupFailureFromAPI(value *nvidiacomv1beta1.EngineGroupFailureStatus) *enginegroup.Failure {
	if value == nil {
		return nil
	}
	return &enginegroup.Failure{
		Classification: enginegroup.FailureClassification(value.Classification),
		Reason:         value.Reason,
		Message:        value.Message,
	}
}

func engineGroupFailureToAPI(value *enginegroup.Failure) *nvidiacomv1beta1.EngineGroupFailureStatus {
	if value == nil {
		return nil
	}
	return &nvidiacomv1beta1.EngineGroupFailureStatus{
		Classification: nvidiacomv1beta1.EngineGroupFailureClassification(value.Classification),
		Reason:         value.Reason,
		Message:        value.Message,
	}
}

func engineGroupTimeFromAPI(value metav1.Time) time.Time {
	return value.Time
}

func engineGroupTimeToAPI(value time.Time) metav1.Time {
	return metav1.NewTime(value)
}

func engineGroupCapacityStatusFromAPI(
	value nvidiacomv1beta1.EngineGroupCapacityReconciliationStatus,
) enginegroup.CapacityStatus {
	return enginegroup.CapacityStatus{
		Desired:  engineGroupCapacityTargetFromAPI(value.Desired),
		Accepted: engineGroupCapacityTargetFromAPI(value.Accepted),
		Observed: engineGroupCapacityObservationFromAPI(value.Observed),
	}
}

func engineGroupCapacityStatusToAPI(
	value enginegroup.CapacityStatus,
) nvidiacomv1beta1.EngineGroupCapacityReconciliationStatus {
	return nvidiacomv1beta1.EngineGroupCapacityReconciliationStatus{
		Desired:  engineGroupCapacityTargetToAPI(value.Desired),
		Accepted: engineGroupCapacityTargetToAPI(value.Accepted),
		Observed: engineGroupCapacityObservationToAPI(value.Observed),
	}
}

func engineGroupCapacityTargetFromAPI(
	value *nvidiacomv1beta1.EngineGroupCapacityTargetStatus,
) *enginegroup.CapacityTarget {
	if value == nil {
		return nil
	}

	// Reconstruct the complete desired allocation set without inferring omitted identities.
	replicas := make([]enginegroup.CapacityReplicaTarget, 0, len(value.Replicas))
	for _, replica := range value.Replicas {
		replicas = append(replicas, enginegroup.CapacityReplicaTarget{
			ReplicaID:   enginegroup.ReplicaID(replica.ReplicaID),
			SlotID:      enginegroup.CapacitySlotID(replica.SlotID),
			Incarnation: engineGroupIncarnationPointerFromAPI(replica.Incarnation),
			Bootstrap:   engineGroupCapacityBootstrapFromAPI(replica.Bootstrap),
		})
	}

	return &enginegroup.CapacityTarget{
		ControlRevision:       value.ControlRevision,
		TransitionID:          value.TransitionID,
		ProfileFingerprint:    value.ProfileFingerprint,
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwner(value.ProcessLifecycleOwner),
		Replicas:              replicas,
		ReleaseFences:         engineGroupReleaseFencesFromAPI(value.ReleaseFences),
	}
}

func engineGroupCapacityTargetToAPI(
	value *enginegroup.CapacityTarget,
) *nvidiacomv1beta1.EngineGroupCapacityTargetStatus {
	if value == nil {
		return nil
	}

	// Serialize the complete desired allocation set without collapsing identity assertions into bootstrap intent.
	replicas := make([]nvidiacomv1beta1.EngineGroupCapacityReplicaTargetStatus, 0, len(value.Replicas))
	for _, replica := range value.Replicas {
		replicas = append(replicas, nvidiacomv1beta1.EngineGroupCapacityReplicaTargetStatus{
			ReplicaID:   string(replica.ReplicaID),
			SlotID:      string(replica.SlotID),
			Incarnation: engineGroupIncarnationPointerToAPI(replica.Incarnation),
			Bootstrap:   engineGroupCapacityBootstrapToAPI(replica.Bootstrap),
		})
	}

	return &nvidiacomv1beta1.EngineGroupCapacityTargetStatus{
		ControlRevision:       value.ControlRevision,
		TransitionID:          value.TransitionID,
		ProfileFingerprint:    value.ProfileFingerprint,
		ProcessLifecycleOwner: nvidiacomv1beta1.EngineGroupProcessLifecycleOwner(value.ProcessLifecycleOwner),
		Replicas:              replicas,
		ReleaseFences:         engineGroupReleaseFencesToAPI(value.ReleaseFences),
	}
}

func engineGroupCapacityBootstrapFromAPI(
	value *nvidiacomv1beta1.EngineGroupCapacityBootstrapStatus,
) *enginegroup.CapacityBootstrap {
	if value == nil {
		return nil
	}
	return &enginegroup.CapacityBootstrap{
		Mode:                   enginegroup.BootstrapMode(value.Mode),
		BaseTopologyGeneration: value.BaseTopologyGeneration,
		NativeMembers:          engineGroupNativeMembersFromAPI(value.NativeMembers),
	}
}

func engineGroupCapacityBootstrapToAPI(
	value *enginegroup.CapacityBootstrap,
) *nvidiacomv1beta1.EngineGroupCapacityBootstrapStatus {
	if value == nil {
		return nil
	}
	return &nvidiacomv1beta1.EngineGroupCapacityBootstrapStatus{
		Mode:                   nvidiacomv1beta1.EngineGroupBootstrapMode(value.Mode),
		BaseTopologyGeneration: value.BaseTopologyGeneration,
		NativeMembers:          engineGroupNativeMembersToAPI(value.NativeMembers),
	}
}

func engineGroupCapacityObservationFromAPI(
	value nvidiacomv1beta1.EngineGroupCapacityObservationStatus,
) enginegroup.CapacityObservation {
	allocations := make([]enginegroup.CapacityAllocation, 0, len(value.Allocations))
	for _, allocation := range value.Allocations {
		allocations = append(allocations, enginegroup.CapacityAllocation{
			Incarnation: engineGroupIncarnationFromAPI(allocation.Incarnation),
			Available:   allocation.Available,
		})
	}
	return enginegroup.CapacityObservation{
		AppliedRevision: value.AppliedRevision,
		Allocations:     allocations,
		ReleaseFences:   engineGroupReleaseFencesFromAPI(value.ReleaseFences),
	}
}

func engineGroupCapacityObservationToAPI(
	value enginegroup.CapacityObservation,
) nvidiacomv1beta1.EngineGroupCapacityObservationStatus {
	allocations := make([]nvidiacomv1beta1.EngineGroupCapacityAllocationStatus, 0, len(value.Allocations))
	for _, allocation := range value.Allocations {
		allocations = append(allocations, nvidiacomv1beta1.EngineGroupCapacityAllocationStatus{
			ReplicaID:   string(allocation.Incarnation.ReplicaID),
			Incarnation: engineGroupIncarnationToAPI(allocation.Incarnation),
			Available:   allocation.Available,
		})
	}
	return nvidiacomv1beta1.EngineGroupCapacityObservationStatus{
		AppliedRevision: value.AppliedRevision,
		Allocations:     allocations,
		ReleaseFences:   engineGroupReleaseFencesToAPI(value.ReleaseFences),
	}
}

func engineGroupReleaseFencesFromAPI(
	values []nvidiacomv1beta1.EngineGroupReleaseAuthorization,
) []enginegroup.ReleaseFence {
	fences := make([]enginegroup.ReleaseFence, 0, len(values))
	for _, value := range values {
		refs := make([]enginegroup.CapacityRef, 0, len(value.CapacityRefs))
		for _, ref := range value.CapacityRefs {
			refs = append(refs, enginegroup.CapacityRef{Name: ref.Name, UID: enginegroup.PodUID(ref.UID)})
		}
		fences = append(fences, enginegroup.ReleaseFence{
			TransitionID:                  value.OperationID,
			AuthorizingTopologyGeneration: value.TopologyGeneration,
			ReplicaID:                     enginegroup.ReplicaID(value.ReplicaID),
			SlotID:                        enginegroup.CapacitySlotID(value.SlotID),
			CapacityRefs:                  refs,
		})
	}
	return fences
}

func engineGroupReleaseFencesToAPI(
	values []enginegroup.ReleaseFence,
) []nvidiacomv1beta1.EngineGroupReleaseAuthorization {
	fences := make([]nvidiacomv1beta1.EngineGroupReleaseAuthorization, 0, len(values))
	for _, value := range values {
		refs := make([]nvidiacomv1beta1.EngineGroupCapacityRef, 0, len(value.CapacityRefs))
		for _, ref := range value.CapacityRefs {
			refs = append(refs, nvidiacomv1beta1.EngineGroupCapacityRef{Name: ref.Name, UID: types.UID(ref.UID)})
		}
		fences = append(fences, nvidiacomv1beta1.EngineGroupReleaseAuthorization{
			OperationID:        value.TransitionID,
			TopologyGeneration: value.AuthorizingTopologyGeneration,
			ReplicaID:          string(value.ReplicaID),
			SlotID:             string(value.SlotID),
			CapacityRefs:       refs,
		})
	}
	return fences
}

func engineGroupTrafficStatusFromAPI(
	value nvidiacomv1beta1.EngineGroupTrafficReconciliationStatus,
) enginegroup.TrafficStatus {
	return enginegroup.TrafficStatus{
		Desired:  engineGroupTrafficTargetFromAPI(value.Desired),
		Accepted: engineGroupTrafficTargetFromAPI(value.Accepted),
		Observed: engineGroupTrafficObservationFromAPI(value.Observed),
	}
}

func engineGroupTrafficStatusToAPI(
	value enginegroup.TrafficStatus,
) nvidiacomv1beta1.EngineGroupTrafficReconciliationStatus {
	return nvidiacomv1beta1.EngineGroupTrafficReconciliationStatus{
		Desired:  engineGroupTrafficTargetToAPI(value.Desired),
		Accepted: engineGroupTrafficTargetToAPI(value.Accepted),
		Observed: engineGroupTrafficObservationToAPI(value.Observed),
	}
}

func engineGroupTrafficTargetFromAPI(
	value *nvidiacomv1beta1.EngineGroupTrafficTargetStatus,
) *enginegroup.TrafficTarget {
	if value == nil {
		return nil
	}
	drain := make([]enginegroup.TrafficDrainTarget, 0, len(value.Drain))
	for _, target := range value.Drain {
		drain = append(drain, enginegroup.TrafficDrainTarget{
			Membership: engineGroupMembershipFromAPI(target.Membership),
			Mode:       enginegroup.TrafficDrainMode(target.Mode),
		})
	}
	return &enginegroup.TrafficTarget{
		ControlRevision:    value.ControlRevision,
		TransitionID:       value.TransitionID,
		TopologyGeneration: value.TopologyGeneration,
		Admitted:           engineGroupMembershipsFromAPI(value.Admitted),
		Drain:              drain,
	}
}

func engineGroupTrafficTargetToAPI(
	value *enginegroup.TrafficTarget,
) *nvidiacomv1beta1.EngineGroupTrafficTargetStatus {
	if value == nil {
		return nil
	}
	drain := make([]nvidiacomv1beta1.EngineGroupTrafficDrainTargetStatus, 0, len(value.Drain))
	for _, target := range value.Drain {
		drain = append(drain, nvidiacomv1beta1.EngineGroupTrafficDrainTargetStatus{
			Membership: engineGroupMembershipToAPI(target.Membership),
			Mode:       nvidiacomv1beta1.EngineGroupTrafficDrainMode(target.Mode),
		})
	}
	return &nvidiacomv1beta1.EngineGroupTrafficTargetStatus{
		ControlRevision:    value.ControlRevision,
		TransitionID:       value.TransitionID,
		TopologyGeneration: value.TopologyGeneration,
		Admitted:           engineGroupMembershipsToAPI(value.Admitted),
		Drain:              drain,
	}
}

func engineGroupTrafficObservationFromAPI(
	value nvidiacomv1beta1.EngineGroupTrafficObservationStatus,
) enginegroup.TrafficObservation {
	return enginegroup.TrafficObservation{
		AppliedRevision: value.AppliedRevision,
		Admitted:        engineGroupMembershipsFromAPI(value.Admitted),
		Draining:        engineGroupMembershipsFromAPI(value.Draining),
		Drained:         engineGroupMembershipsFromAPI(value.Drained),
	}
}

func engineGroupTrafficObservationToAPI(
	value enginegroup.TrafficObservation,
) nvidiacomv1beta1.EngineGroupTrafficObservationStatus {
	return nvidiacomv1beta1.EngineGroupTrafficObservationStatus{
		AppliedRevision: value.AppliedRevision,
		Admitted:        engineGroupMembershipsToAPI(value.Admitted),
		Draining:        engineGroupMembershipsToAPI(value.Draining),
		Drained:         engineGroupMembershipsToAPI(value.Drained),
	}
}

func engineGroupMembershipStatusFromAPI(
	value nvidiacomv1beta1.EngineGroupMembershipReconciliationStatus,
) enginegroup.MembershipStatus {
	return enginegroup.MembershipStatus{
		Desired:  engineGroupMembershipTargetFromAPI(value.Desired),
		Observed: engineGroupMembershipObservationFromAPI(value.Observed),
	}
}

func engineGroupMembershipStatusToAPI(
	value enginegroup.MembershipStatus,
) nvidiacomv1beta1.EngineGroupMembershipReconciliationStatus {
	return nvidiacomv1beta1.EngineGroupMembershipReconciliationStatus{
		Desired:  engineGroupMembershipTargetToAPI(value.Desired),
		Observed: engineGroupMembershipObservationToAPI(value.Observed),
	}
}

func engineGroupMembershipTargetFromAPI(
	value *nvidiacomv1beta1.EngineGroupMembershipTargetStatus,
) *enginegroup.MembershipTarget {
	if value == nil {
		return nil
	}
	joining := make([]enginegroup.JoiningReplica, 0, len(value.Joining))
	for _, replica := range value.Joining {
		joining = append(joining, enginegroup.JoiningReplica{
			ReplicaID:          enginegroup.ReplicaID(replica.ReplicaID),
			RuntimeIncarnation: enginegroup.RuntimeIncarnationID(replica.RuntimeIncarnation),
		})
	}
	return &enginegroup.MembershipTarget{
		ControlRevision: value.ControlRevision,
		TransitionID:    value.TransitionID,
		TargetDigest:    value.TargetDigest,
		Validation:      engineGroupValidationEvidenceFromAPI(value.Validation),
		BaseTopology:    engineGroupTopologyFromAPI(value.BaseTopology),
		Plan:            engineGroupPlanFromAPI(value.Plan),
		Joining:         joining,
	}
}

func engineGroupMembershipTargetToAPI(
	value *enginegroup.MembershipTarget,
) *nvidiacomv1beta1.EngineGroupMembershipTargetStatus {
	if value == nil {
		return nil
	}
	joining := make([]nvidiacomv1beta1.EngineGroupJoiningReplicaStatus, 0, len(value.Joining))
	for _, replica := range value.Joining {
		joining = append(joining, nvidiacomv1beta1.EngineGroupJoiningReplicaStatus{
			ReplicaID:          string(replica.ReplicaID),
			RuntimeIncarnation: string(replica.RuntimeIncarnation),
		})
	}
	return &nvidiacomv1beta1.EngineGroupMembershipTargetStatus{
		ControlRevision: value.ControlRevision,
		TransitionID:    value.TransitionID,
		TargetDigest:    value.TargetDigest,
		Validation:      engineGroupValidationEvidenceToAPI(value.Validation),
		BaseTopology:    engineGroupTopologyToAPI(value.BaseTopology),
		Plan:            engineGroupPlanToAPI(value.Plan),
		Joining:         joining,
	}
}

func engineGroupMembershipObservationFromAPI(
	value nvidiacomv1beta1.EngineGroupMembershipObservationStatus,
) enginegroup.MembershipObservation {
	return enginegroup.MembershipObservation{
		CommittedTopology:     engineGroupTopologyFromAPI(value.CommittedTopology),
		RequestedTransitionID: value.RequestedTransitionID,
		Transition:            engineGroupMembershipTransitionFromAPI(value.Transition),
	}
}

func engineGroupMembershipObservationToAPI(
	value enginegroup.MembershipObservation,
) nvidiacomv1beta1.EngineGroupMembershipObservationStatus {
	return nvidiacomv1beta1.EngineGroupMembershipObservationStatus{
		CommittedTopology:     engineGroupTopologyToAPI(value.CommittedTopology),
		RequestedTransitionID: value.RequestedTransitionID,
		Transition:            engineGroupMembershipTransitionToAPI(value.Transition),
	}
}

func engineGroupMembershipTransitionFromAPI(
	value *nvidiacomv1beta1.EngineGroupMembershipTransitionObservationStatus,
) *enginegroup.MembershipTransitionObservation {
	if value == nil {
		return nil
	}
	var result *enginegroup.MembershipTopology
	if value.ResultTopology != nil {
		topology := engineGroupTopologyFromAPI(*value.ResultTopology)
		result = &topology
	}
	return &enginegroup.MembershipTransitionObservation{
		TransitionID:    value.TransitionID,
		ControlRevision: value.ControlRevision,
		TargetDigest:    value.TargetDigest,
		Phase:           enginegroup.MembershipTransitionPhase(value.Phase),
		ResultTopology:  result,
		Failure:         engineGroupFailureFromAPI(value.Failure),
	}
}

func engineGroupMembershipTransitionToAPI(
	value *enginegroup.MembershipTransitionObservation,
) *nvidiacomv1beta1.EngineGroupMembershipTransitionObservationStatus {
	if value == nil {
		return nil
	}
	var result *nvidiacomv1beta1.EngineGroupTopologyStatus
	if value.ResultTopology != nil {
		topology := engineGroupTopologyToAPI(*value.ResultTopology)
		result = &topology
	}
	return &nvidiacomv1beta1.EngineGroupMembershipTransitionObservationStatus{
		TransitionID:    value.TransitionID,
		ControlRevision: value.ControlRevision,
		TargetDigest:    value.TargetDigest,
		Phase:           nvidiacomv1beta1.EngineGroupMembershipTransitionPhase(value.Phase),
		ResultTopology:  result,
		Failure:         engineGroupFailureToAPI(value.Failure),
	}
}

func engineGroupTransitionFromAPI(
	value *nvidiacomv1beta1.EngineGroupTransitionStatus,
) *enginegroup.TransitionStatus {
	if value == nil {
		return nil
	}
	return &enginegroup.TransitionStatus{
		Spec: enginegroup.TransitionSpec{
			ID:                     value.Spec.ID,
			BaseTopologyGeneration: value.Spec.BaseTopologyGeneration,
			Plan:                   engineGroupPlanFromAPI(value.Spec.Plan),
		},
		PlanPreflight:   engineGroupPreflightFromAPI(value.PlanPreflight),
		TargetPreflight: engineGroupPreflightFromAPI(value.TargetPreflight),
		Verification:    engineGroupVerificationFromAPI(value.Verification),
		Outcome:         enginegroup.TransitionOutcome(value.Outcome),
		Failure:         engineGroupFailureFromAPI(value.Failure),
		StartedAt:       engineGroupTimeFromAPI(value.StartedAt),
		UpdatedAt:       engineGroupTimeFromAPI(value.UpdatedAt),
	}
}

func engineGroupTransitionToAPI(
	value *enginegroup.TransitionStatus,
) *nvidiacomv1beta1.EngineGroupTransitionStatus {
	if value == nil {
		return nil
	}
	return &nvidiacomv1beta1.EngineGroupTransitionStatus{
		Spec: nvidiacomv1beta1.EngineGroupTransitionSpecStatus{
			ID:                     value.Spec.ID,
			BaseTopologyGeneration: value.Spec.BaseTopologyGeneration,
			Plan:                   engineGroupPlanToAPI(value.Spec.Plan),
		},
		PlanPreflight:   engineGroupPreflightToAPI(value.PlanPreflight),
		TargetPreflight: engineGroupPreflightToAPI(value.TargetPreflight),
		Verification:    engineGroupVerificationToAPI(value.Verification),
		Outcome:         nvidiacomv1beta1.EngineGroupTransitionOutcome(value.Outcome),
		Failure:         engineGroupFailureToAPI(value.Failure),
		StartedAt:       engineGroupTimeToAPI(value.StartedAt),
		UpdatedAt:       engineGroupTimeToAPI(value.UpdatedAt),
	}
}

func engineGroupPreflightFromAPI(value *nvidiacomv1beta1.EngineGroupPreflightStatus) enginegroup.PreflightStatus {
	if value == nil {
		return enginegroup.PreflightStatus{}
	}
	return enginegroup.PreflightStatus{
		TransitionID:    value.TransitionID,
		ControlRevision: value.ControlRevision,
		SubjectDigest:   value.SubjectDigest,
		Evidence:        engineGroupValidationEvidencePointerFromAPI(value.Evidence),
		Rejection:       engineGroupFailureFromAPI(value.Rejection),
	}
}

func engineGroupPreflightToAPI(value enginegroup.PreflightStatus) *nvidiacomv1beta1.EngineGroupPreflightStatus {
	if value.TransitionID == "" && value.ControlRevision == 0 && value.SubjectDigest == "" &&
		value.Evidence == nil && value.Rejection == nil {
		return nil
	}
	return &nvidiacomv1beta1.EngineGroupPreflightStatus{
		TransitionID:    value.TransitionID,
		ControlRevision: value.ControlRevision,
		SubjectDigest:   value.SubjectDigest,
		Evidence:        engineGroupValidationEvidencePointerToAPI(value.Evidence),
		Rejection:       engineGroupFailureToAPI(value.Rejection),
	}
}

func engineGroupValidationEvidenceFromAPI(
	value nvidiacomv1beta1.EngineGroupValidationEvidenceStatus,
) enginegroup.ValidationEvidence {
	return enginegroup.ValidationEvidence{
		PlanDigest:           value.PlanDigest,
		TargetDigest:         value.TargetDigest,
		ProfileFingerprint:   value.ProfileFingerprint,
		CapabilityGeneration: value.CapabilityGeneration,
	}
}

func engineGroupValidationEvidenceToAPI(
	value enginegroup.ValidationEvidence,
) nvidiacomv1beta1.EngineGroupValidationEvidenceStatus {
	return nvidiacomv1beta1.EngineGroupValidationEvidenceStatus{
		PlanDigest:           value.PlanDigest,
		TargetDigest:         value.TargetDigest,
		ProfileFingerprint:   value.ProfileFingerprint,
		CapabilityGeneration: value.CapabilityGeneration,
	}
}

func engineGroupValidationEvidencePointerFromAPI(
	value *nvidiacomv1beta1.EngineGroupValidationEvidenceStatus,
) *enginegroup.ValidationEvidence {
	if value == nil {
		return nil
	}
	evidence := engineGroupValidationEvidenceFromAPI(*value)
	return &evidence
}

func engineGroupValidationEvidencePointerToAPI(
	value *enginegroup.ValidationEvidence,
) *nvidiacomv1beta1.EngineGroupValidationEvidenceStatus {
	if value == nil {
		return nil
	}
	evidence := engineGroupValidationEvidenceToAPI(*value)
	return &evidence
}

func engineGroupVerificationFromAPI(
	value *nvidiacomv1beta1.EngineGroupVerificationStatus,
) enginegroup.VerificationStatus {
	if value == nil {
		return enginegroup.VerificationStatus{}
	}
	var proof *enginegroup.ServingProof
	if value.Proof != nil {
		proof = &enginegroup.ServingProof{
			TopologyGeneration: value.Proof.TopologyGeneration,
			RuntimeDigest:      value.Proof.RuntimeDigest,
			ObservedAt:         engineGroupTimeFromAPI(value.Proof.ObservedAt),
		}
	}
	return enginegroup.VerificationStatus{
		Phase:   enginegroup.VerificationPhase(value.Phase),
		Proof:   proof,
		Failure: engineGroupFailureFromAPI(value.Failure),
	}
}

func engineGroupVerificationToAPI(
	value enginegroup.VerificationStatus,
) *nvidiacomv1beta1.EngineGroupVerificationStatus {
	if value.Phase == "" && value.Proof == nil && value.Failure == nil {
		return nil
	}
	var proof *nvidiacomv1beta1.EngineGroupServingProofStatus
	if value.Proof != nil {
		proof = &nvidiacomv1beta1.EngineGroupServingProofStatus{
			TopologyGeneration: value.Proof.TopologyGeneration,
			RuntimeDigest:      value.Proof.RuntimeDigest,
			ObservedAt:         engineGroupTimeToAPI(value.Proof.ObservedAt),
		}
	}
	return &nvidiacomv1beta1.EngineGroupVerificationStatus{
		Phase:   nvidiacomv1beta1.EngineGroupVerificationPhase(value.Phase),
		Proof:   proof,
		Failure: engineGroupFailureToAPI(value.Failure),
	}
}

func engineGroupPlanFromAPI(value nvidiacomv1beta1.EngineGroupResolvedPlanStatus) enginegroup.ResolvedPlan {
	return enginegroup.ResolvedPlan{
		ID:                      value.ID,
		ProfileFingerprint:      value.ProfileFingerprint,
		ProcessLifecycleOwner:   enginegroup.ProcessLifecycleOwner(value.ProcessLifecycleOwner),
		TrafficRequirement:      enginegroup.TrafficRequirement(value.TrafficRequirement),
		VerificationRequirement: enginegroup.VerificationRequirement(value.VerificationRequirement),
		Change:                  engineGroupChangeFromAPI(value.Change),
	}
}

func engineGroupPlanToAPI(value enginegroup.ResolvedPlan) nvidiacomv1beta1.EngineGroupResolvedPlanStatus {
	return nvidiacomv1beta1.EngineGroupResolvedPlanStatus{
		ID:                      value.ID,
		ProfileFingerprint:      value.ProfileFingerprint,
		ProcessLifecycleOwner:   nvidiacomv1beta1.EngineGroupProcessLifecycleOwner(value.ProcessLifecycleOwner),
		TrafficRequirement:      nvidiacomv1beta1.EngineGroupTrafficRequirement(value.TrafficRequirement),
		VerificationRequirement: nvidiacomv1beta1.EngineGroupVerificationRequirement(value.VerificationRequirement),
		Change:                  engineGroupChangeToAPI(value.Change),
	}
}

func engineGroupChangeFromAPI(value nvidiacomv1beta1.EngineGroupResolvedChangeStatus) enginegroup.ResolvedChange {
	change := enginegroup.ResolvedChange{Kind: enginegroup.PlanKind(value.Kind)}

	// Decode only the explicitly represented tagged-union variants; validation rejects inconsistent shapes later.
	if value.Grow != nil {
		change.Grow = &enginegroup.GrowChange{Replicas: engineGroupReplicaTargetsFromAPI(value.Grow.Replicas)}
	}
	if value.Retire != nil {
		change.Retire = &enginegroup.RetireChange{Replicas: engineGroupReplicaIDsFromAPI(value.Retire.Replicas)}
	}
	if value.ReduceToSurvivors != nil {
		change.ReduceToSurvivors = &enginegroup.ReduceToSurvivorsChange{
			Survivors: engineGroupReplicaIDsFromAPI(value.ReduceToSurvivors.Survivors),
		}
	}
	if value.Restore != nil {
		replicas := engineGroupReplicaTargetsFromAPI(value.Restore.Replicas)
		change.Restore = &enginegroup.RestoreChange{Replicas: make([]enginegroup.RestorationTarget, 0, len(replicas))}
		for _, replica := range replicas {
			change.Restore.Replicas = append(change.Restore.Replicas, enginegroup.RestorationTarget{
				ReplicaTarget: replica,
			})
		}
	}
	if value.Remap != nil {
		change.Remap = &enginegroup.RemapChange{
			Membership: engineGroupNativeMembershipsFromAPI(value.Remap.Membership),
		}
	}
	return change
}

func engineGroupChangeToAPI(value enginegroup.ResolvedChange) nvidiacomv1beta1.EngineGroupResolvedChangeStatus {
	change := nvidiacomv1beta1.EngineGroupResolvedChangeStatus{
		Kind: nvidiacomv1beta1.EngineGroupPlanKind(value.Kind),
	}

	// Encode every populated variant so malformed internal state cannot be silently normalized at persistence.
	if value.Grow != nil {
		change.Grow = &nvidiacomv1beta1.EngineGroupGrowChangeStatus{
			Replicas: engineGroupReplicaTargetsToAPI(value.Grow.Replicas),
		}
	}
	if value.Retire != nil {
		change.Retire = &nvidiacomv1beta1.EngineGroupRetireChangeStatus{
			Replicas: engineGroupReplicaIDsToAPI(value.Retire.Replicas),
		}
	}
	if value.ReduceToSurvivors != nil {
		change.ReduceToSurvivors = &nvidiacomv1beta1.EngineGroupReduceToSurvivorsChangeStatus{
			Survivors: engineGroupReplicaIDsToAPI(value.ReduceToSurvivors.Survivors),
		}
	}
	if value.Restore != nil {
		replicas := make([]enginegroup.ReplicaTarget, 0, len(value.Restore.Replicas))
		for _, replica := range value.Restore.Replicas {
			replicas = append(replicas, replica.ReplicaTarget)
		}
		change.Restore = &nvidiacomv1beta1.EngineGroupRestoreChangeStatus{
			Replicas: engineGroupReplicaTargetsToAPI(replicas),
		}
	}
	if value.Remap != nil {
		change.Remap = &nvidiacomv1beta1.EngineGroupRemapChangeStatus{
			Membership: engineGroupNativeMembershipsToAPI(value.Remap.Membership),
		}
	}
	return change
}

func engineGroupReplicaTargetsFromAPI(
	values []nvidiacomv1beta1.EngineGroupReplicaTargetStatus,
) []enginegroup.ReplicaTarget {
	replicas := make([]enginegroup.ReplicaTarget, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, enginegroup.ReplicaTarget{
			ReplicaID:     enginegroup.ReplicaID(value.ReplicaID),
			SlotID:        enginegroup.CapacitySlotID(value.SlotID),
			Bootstrap:     enginegroup.BootstrapMode(value.Bootstrap),
			NativeMembers: engineGroupNativeMembersFromAPI(value.NativeMembers),
		})
	}
	return replicas
}

func engineGroupReplicaTargetsToAPI(
	values []enginegroup.ReplicaTarget,
) []nvidiacomv1beta1.EngineGroupReplicaTargetStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupReplicaTargetStatus, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, nvidiacomv1beta1.EngineGroupReplicaTargetStatus{
			ReplicaID:     string(value.ReplicaID),
			SlotID:        string(value.SlotID),
			Bootstrap:     nvidiacomv1beta1.EngineGroupBootstrapMode(value.Bootstrap),
			NativeMembers: engineGroupNativeMembersToAPI(value.NativeMembers),
		})
	}
	return replicas
}

func engineGroupNativeMembershipsFromAPI(
	values []nvidiacomv1beta1.EngineGroupNativeMembershipStatus,
) []enginegroup.ReplicaNativeMembership {
	replicas := make([]enginegroup.ReplicaNativeMembership, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, enginegroup.ReplicaNativeMembership{
			ReplicaID:     enginegroup.ReplicaID(value.ReplicaID),
			SlotID:        enginegroup.CapacitySlotID(value.SlotID),
			NativeMembers: engineGroupNativeMembersFromAPI(value.NativeMembers),
		})
	}
	return replicas
}

func engineGroupNativeMembershipsToAPI(
	values []enginegroup.ReplicaNativeMembership,
) []nvidiacomv1beta1.EngineGroupNativeMembershipStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupNativeMembershipStatus, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, nvidiacomv1beta1.EngineGroupNativeMembershipStatus{
			ReplicaID:     string(value.ReplicaID),
			SlotID:        string(value.SlotID),
			NativeMembers: engineGroupNativeMembersToAPI(value.NativeMembers),
		})
	}
	return replicas
}

func engineGroupReplicaIDsFromAPI(values []string) []enginegroup.ReplicaID {
	replicas := make([]enginegroup.ReplicaID, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, enginegroup.ReplicaID(value))
	}
	return replicas
}

func engineGroupReplicaIDsToAPI(values []enginegroup.ReplicaID) []string {
	replicas := make([]string, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, string(value))
	}
	return replicas
}

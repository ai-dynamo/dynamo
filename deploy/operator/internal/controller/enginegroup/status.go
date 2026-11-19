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
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// These one-way projections describe observed state; they never reconstruct recovery authority.
func engineGroupTopologyToAPI(value domain.MembershipTopology) nvidiacomv1beta1.EngineGroupTopologyStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupMemberStatus, 0, len(value.Replicas))
	for _, replica := range value.Replicas {
		replicas = append(replicas, engineGroupMembershipToAPI(replica))
	}
	return nvidiacomv1beta1.EngineGroupTopologyStatus{Generation: value.Generation, Replicas: replicas}
}

func engineGroupMembershipToAPI(value domain.ReplicaMembership) nvidiacomv1beta1.EngineGroupMemberStatus {
	return nvidiacomv1beta1.EngineGroupMemberStatus{
		ReplicaID:     string(value.ReplicaID),
		NativeMembers: engineGroupMemberIncarnationsToAPI(value.Members),
	}
}

func engineGroupMembershipsToAPI(values []domain.ReplicaMembership) []nvidiacomv1beta1.EngineGroupMemberStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupMemberStatus, 0, len(values))
	for _, value := range values {
		replicas = append(replicas, engineGroupMembershipToAPI(value))
	}
	return replicas
}

func engineGroupMemberIncarnationsToAPI(values []domain.NativeMemberIncarnation) []nvidiacomv1beta1.EngineGroupNativeMemberIncarnationStatus {
	members := make([]nvidiacomv1beta1.EngineGroupNativeMemberIncarnationStatus, 0, len(values))
	for _, value := range values {
		members = append(members, nvidiacomv1beta1.EngineGroupNativeMemberIncarnationStatus{ID: string(value.ID), RuntimeIncarnation: string(value.RuntimeIncarnation)})
	}
	return members
}

func engineGroupNativeMembersToAPI(values []domain.NativeMemberID) []string {
	members := make([]string, 0, len(values))
	for _, value := range values {
		members = append(members, string(value))
	}
	return members
}

func engineGroupFailureToAPI(value *domain.Failure) *nvidiacomv1beta1.EngineGroupFailureStatus {
	if value == nil {
		return nil
	}
	return &nvidiacomv1beta1.EngineGroupFailureStatus{
		Classification: nvidiacomv1beta1.EngineGroupFailureClassification(value.Classification),
		Reason:         value.Reason,
		Message:        value.Message,
	}
}

func engineGroupReleaseFencesToAPI(
	values []domain.ReleaseFence,
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

func engineGroupVerificationToAPI(
	value domain.VerificationStatus,
) *nvidiacomv1beta1.EngineGroupVerificationStatus {
	if value.Phase == "" && value.Proof == nil && value.Failure == nil {
		return nil
	}
	var proof *nvidiacomv1beta1.EngineGroupServingProofStatus
	if value.Proof != nil {
		proof = &nvidiacomv1beta1.EngineGroupServingProofStatus{
			TopologyGeneration: value.Proof.TopologyGeneration,
			RuntimeDigest:      value.Proof.RuntimeDigest,
			ObservedAt:         metav1.NewTime(value.Proof.ObservedAt),
		}
	}
	return &nvidiacomv1beta1.EngineGroupVerificationStatus{
		Phase:   nvidiacomv1beta1.EngineGroupVerificationPhase(value.Phase),
		Proof:   proof,
		Failure: engineGroupFailureToAPI(value.Failure),
	}
}

func engineGroupCapacityRefsToAPI(refs []domain.CapacityRef) []nvidiacomv1beta1.EngineGroupCapacityRef {
	result := make([]nvidiacomv1beta1.EngineGroupCapacityRef, 0, len(refs))
	for _, ref := range refs {
		result = append(result, nvidiacomv1beta1.EngineGroupCapacityRef{Name: ref.Name, UID: types.UID(ref.UID)})
	}
	return result
}

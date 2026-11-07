/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"
	"slices"
	"sort"
	"time"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func resolveEngineGroupOperation(group *api.DynamoGraphDeploymentEngineGroup, state domain.GroupStatus, previous *api.EngineGroupOperationStatus) (*api.EngineGroupOperationStatus, error) {
	if state.Transition == nil {
		return nil, nil
	}
	transition := state.Transition
	operation := previous.DeepCopy()
	if operation == nil || operation.ID != transition.Spec.ID {
		base, found := state.Topologies.Snapshot(transition.Spec.BaseTopologyGeneration)
		if !found {
			return nil, fmt.Errorf("operation %q has no base topology", transition.Spec.ID)
		}
		target, joining, nominated := engineGroupOperationTarget(base, transition.Spec.Plan.Change)
		intent := "Recover"
		if transition.Spec.Plan.Change.Kind == domain.PlanKindGrow {
			intent = "Grow"
		} else if transition.Spec.Plan.Change.Kind == domain.PlanKindRetire {
			intent = "Shrink"
			if !group.DeletionTimestamp.IsZero() && len(target) == 0 {
				intent = "Retire"
			}
		}

		// Freeze the source generation and exact resolved target when the plan first becomes durable.
		operation = &api.EngineGroupOperationStatus{
			ID: transition.Spec.ID, Intent: intent, Shape: string(transition.Spec.Plan.Change.Kind),
			SpecGeneration: group.Generation, BaseTopology: engineGroupTopologyToAPI(base),
			TargetReplicas: int32(len(target)), TargetNativeMembers: operationTargetMemberIDs(target),
			JoiningReplicas: joining, NominatedReplicas: nominated,
			TrafficRequirement:  api.EngineGroupTrafficRequirement(transition.Spec.Plan.TrafficRequirement),
			ServingVerification: api.EngineGroupVerificationRequirement(transition.Spec.Plan.VerificationRequirement),
			StartedAt:           metav1.NewTime(transition.StartedAt),
		}
	}

	// Phase is an observation projection: a persisted desired target is not evidence of acceptance.
	phase := engineGroupOperationPhase(state)
	if operation.Phase != phase {
		operation.LastTransitionTime = metav1.NewTime(time.Now().UTC().Truncate(time.Second))
	}
	operation.Phase = phase
	operation.Verification = engineGroupVerificationToAPI(transition.Verification)
	operation.Error = engineGroupFailureToAPI(transition.Failure)
	operation.CommittedTopology = nil
	observed := state.Membership.Observed.Transition
	if engineGroupMembershipTargetIsCurrent(state) && observed != nil &&
		observed.TransitionID == state.Membership.Desired.TransitionID &&
		observed.ControlRevision == state.Membership.Desired.ControlRevision &&
		observed.TargetDigest == state.Membership.Desired.TargetDigest {
		if observed.Phase == domain.MembershipTransitionPhaseCommitted && observed.ResultTopology != nil {
			result := engineGroupTopologyToAPI(*observed.ResultTopology)
			operation.CommittedTopology = &result
		}
		if operation.Error == nil {
			operation.Error = engineGroupFailureToAPI(observed.Failure)
		}
	}

	// A newer target queues behind the immutable operation; it never rewrites its plan.
	operation.QueuedTargetReplicas = nil
	if group.DeletionTimestamp.IsZero() && group.Generation > operation.SpecGeneration && group.Spec.Replicas != operation.TargetReplicas {
		target := group.Spec.Replicas
		operation.QueuedTargetReplicas = &target
	}
	return operation, nil
}

func engineGroupOperationPhase(state domain.GroupStatus) api.EngineGroupOperationPhase {
	switch state.Transition.Outcome {
	case domain.TransitionOutcomeReverting:
		return api.EngineGroupOperationPhaseAborting
	case domain.TransitionOutcomeRolledBack:
		return api.EngineGroupOperationPhaseAborted
	case domain.TransitionOutcomeCompleted:
		return api.EngineGroupOperationPhaseCommitted
	case domain.TransitionOutcomeBlocked:
		if observed := state.Membership.Observed.Transition; engineGroupMembershipTargetIsCurrent(state) && observed != nil &&
			observed.Phase == domain.MembershipTransitionPhaseUnknown {
			return api.EngineGroupOperationPhaseUnknown
		}
		return api.EngineGroupOperationPhaseFailed
	}
	if !engineGroupMembershipTargetIsCurrent(state) {
		return api.EngineGroupOperationPhasePending
	}
	observed := state.Membership.Observed.Transition
	if observed == nil {
		return api.EngineGroupOperationPhaseSubmitting
	}
	if observed.TransitionID != state.Membership.Desired.TransitionID ||
		observed.ControlRevision != state.Membership.Desired.ControlRevision ||
		observed.TargetDigest != state.Membership.Desired.TargetDigest {
		return api.EngineGroupOperationPhaseUnknown
	}
	switch observed.Phase {
	case domain.MembershipTransitionPhasePending:
		return api.EngineGroupOperationPhaseCommitting
	case domain.MembershipTransitionPhaseCommitted:
		return api.EngineGroupOperationPhaseCommitted
	case domain.MembershipTransitionPhaseRejected:
		return api.EngineGroupOperationPhaseFailed
	default:
		return api.EngineGroupOperationPhaseUnknown
	}
}

// engineGroupMembershipTargetIsCurrent binds retained membership evidence to this transition's exact preflight.
// state.Transition must not be nil; durable preflight correlation is validated before projection.
func engineGroupMembershipTargetIsCurrent(state domain.GroupStatus) bool {
	target := state.Membership.Desired
	if target == nil {
		return false
	}
	preflight := state.Transition.TargetPreflight
	return preflight.Evidence != nil && preflight.TransitionID == target.TransitionID &&
		preflight.ControlRevision == target.ControlRevision && preflight.SubjectDigest == target.TargetDigest
}

// engineGroupOperationTarget projects only immutable plan identity, never current registry or Pod state.
// change must be a valid tagged union; the coordinator validates it before persisting a transition.
func engineGroupOperationTarget(
	base domain.MembershipTopology,
	change domain.ResolvedChange,
) (map[domain.ReplicaID][]domain.NativeMemberID, []string, []string) {
	target := make(map[domain.ReplicaID][]domain.NativeMemberID, len(base.Replicas))
	for _, replica := range base.Replicas {
		for _, member := range replica.Members {
			target[replica.ReplicaID] = append(target[replica.ReplicaID], member.ID)
		}
	}
	var joining, nominated []string
	switch change.Kind {
	case domain.PlanKindGrow:
		for _, replica := range change.Grow.Replicas {
			target[replica.ReplicaID] = slices.Clone(replica.NativeMembers)
			joining = append(joining, string(replica.ReplicaID))
		}
	case domain.PlanKindRetire:
		for _, replicaID := range change.Retire.Replicas {
			delete(target, replicaID)
			nominated = append(nominated, string(replicaID))
		}
	case domain.PlanKindReduceToSurvivors:
		clear(target)
		for _, replica := range change.ReduceToSurvivors.Survivors {
			target[replica.ReplicaID] = slices.Clone(replica.NativeMembers)
		}
		for _, replica := range base.Replicas {
			survivors := slices.Clone(target[replica.ReplicaID])
			slices.Sort(survivors)
			original := operationBaseMemberIDs(replica)
			slices.Sort(original)
			if !slices.Equal(survivors, original) {
				nominated = append(nominated, string(replica.ReplicaID))
			}
		}
	case domain.PlanKindRestore:
		for _, replica := range change.Restore.Replicas {
			target[replica.ReplicaID] = slices.Clone(replica.NativeMembers)
			joining = append(joining, string(replica.ReplicaID))
		}
	case domain.PlanKindRemap:
		clear(target)
		for _, replica := range change.Remap.Membership {
			target[replica.ReplicaID] = slices.Clone(replica.NativeMembers)
		}
	}
	sort.Strings(joining)
	sort.Strings(nominated)
	return target, joining, nominated
}

func operationBaseMemberIDs(replica domain.ReplicaMembership) []domain.NativeMemberID {
	members := make([]domain.NativeMemberID, 0, len(replica.Members))
	for _, member := range replica.Members {
		members = append(members, member.ID)
	}
	return members
}

func operationTargetMemberIDs(target map[domain.ReplicaID][]domain.NativeMemberID) []string {
	// Preserve absence while allocating once for every projected target member.
	memberCount := 0
	for _, replica := range target {
		memberCount += len(replica)
	}
	if memberCount == 0 {
		return nil
	}
	members := make([]string, 0, memberCount)
	for _, replica := range target {
		members = append(members, engineGroupNativeMembersToAPI(replica)...)
	}
	sort.Strings(members)
	return slices.Compact(members)
}

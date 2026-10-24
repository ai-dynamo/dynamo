/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// ValidatePlan checks the exact planned resize shape and the still-current base without changing either authority.
func (a *MembershipAdapter) ValidatePlan(
	ctx context.Context,
	groupID enginegroup.GroupID,
	request enginegroup.PlanValidationRequest,
) (enginegroup.PreflightResult, error) {
	// Validate declared capabilities before contacting a runtime that may not support this operation at all.
	if err := a.validate(groupID); err != nil {
		return enginegroup.PreflightResult{}, err
	}
	if _, failure := a.resolvePlan(request.BaseTopology, request.Plan); failure != nil {
		return enginegroup.PreflightResult{Rejection: failure}, nil
	}
	if request.PlanDigest == "" {
		return rejectedPreflight("InvalidPlan", "the coordinator's canonical plan digest is required"), nil
	}
	if err := a.validateBase(ctx, request.BaseTopology); err != nil {
		return enginegroup.PreflightResult{}, err
	}
	// Integration-owned evidence must prove drain support for every old process before any effects.
	if request.Plan.TrafficRequirement == enginegroup.TrafficRequirementEngineManagedDrain {
		if request.Plan.Change.Kind != enginegroup.PlanKindRetire || a.ManagedDrainValidator == nil {
			return rejectedPreflight("UnsupportedDrainPolicy", "planned retirement requires a configured native-drain validator"), nil
		}
		binding, _, err := a.load(ctx)
		if err != nil {
			return enginegroup.PreflightResult{}, err
		}
		members, err := membersByRank(request.BaseTopology)
		if err != nil {
			return enginegroup.PreflightResult{}, err
		}
		for _, member := range members {
			process, err := a.processForMember(member.Members[0].ID)
			if err != nil {
				return enginegroup.PreflightResult{}, err
			}
			failure, err := a.ManagedDrainValidator(ctx, groupID, process, binding.Instances[process.Rank])
			if err != nil {
				return enginegroup.PreflightResult{}, err
			}
			if failure != nil {
				return enginegroup.PreflightResult{Rejection: failure}, nil
			}
		}
	}
	return enginegroup.PreflightResult{Evidence: &enginegroup.ValidationEvidence{
		PlanDigest: request.PlanDigest, ProfileFingerprint: a.ProfileFingerprint,
		CapabilityGeneration: a.capabilityGeneration(),
	}}, nil
}

// ValidateTarget binds the earlier evidence to the exact frozen joining identities and canonical target digest.
func (a *MembershipAdapter) ValidateTarget(
	ctx context.Context,
	groupID enginegroup.GroupID,
	target enginegroup.MembershipTarget,
) (enginegroup.PreflightResult, error) {
	// Repeat semantic validation using the current capability bounds before accepting exact-target evidence.
	preflight, err := a.ValidatePlan(ctx, groupID, enginegroup.PlanValidationRequest{
		BaseTopology: target.BaseTopology, Plan: target.Plan, PlanDigest: target.Validation.PlanDigest,
	})
	if err != nil || preflight.Rejection != nil {
		return preflight, err
	}
	if target.Validation.ProfileFingerprint != a.ProfileFingerprint ||
		target.Validation.CapabilityGeneration != a.capabilityGeneration() {
		return rejectedPreflight("CapabilitiesChanged", "profile or capability bounds changed after plan validation"), nil
	}
	if !validIdentifier(target.TransitionID) || target.ControlRevision <= 0 || target.TargetDigest == "" {
		return rejectedPreflight("InvalidTarget", "a transition ID, positive control revision, and canonical target digest are required"), nil
	}
	if err := a.validateJoining(target); err != nil {
		return rejectedPreflight("InvalidJoiningSet", err.Error()), nil
	}
	evidence := *preflight.Evidence
	evidence.TargetDigest = target.TargetDigest
	return enginegroup.PreflightResult{Evidence: &evidence}, nil
}

func (a *MembershipAdapter) resolvePlan(base enginegroup.MembershipTopology, plan enginegroup.ResolvedPlan) (int32, *enginegroup.Failure) {
	// The profile fixes members per allocation; native resizing changes only a healthy contiguous rank suffix.
	if plan.ID == "" || plan.ProfileFingerprint != a.ProfileFingerprint ||
		plan.ProcessLifecycleOwner != enginegroup.ProcessLifecycleOwnerOrchestrator ||
		plan.VerificationRequirement != enginegroup.VerificationRequirementRequired ||
		(plan.TrafficRequirement != enginegroup.TrafficRequirementKeepServing &&
			plan.TrafficRequirement != enginegroup.TrafficRequirementQuiesceGroup &&
			plan.TrafficRequirement != enginegroup.TrafficRequirementEngineManagedDrain) {
		return 0, terminalFailure("UnsupportedPlan", "vLLM resize requires the resolved profile, external lifecycle, traffic policy, and serving verification")
	}
	if base.Generation <= 0 || base.Generation == math.MaxInt64 {
		return 0, terminalFailure("InvalidBase", "the base requires a positive topology generation with room for a successor")
	}
	byRank, err := membersByRank(base)
	if err != nil {
		return 0, terminalFailure("UnsupportedRankLayout", err.Error())
	}
	for _, replica := range base.Replicas {
		if len(replica.Members) != int(a.Layout.NativeMembersPerReplica) {
			return 0, terminalFailure("PartialMembershipUnsupported", "planned external resize requires complete allocations; survivor recovery is a separate capability")
		}
	}
	change := plan.Change
	variants := 0
	for _, present := range []bool{change.Grow != nil, change.Restore != nil, change.Retire != nil, change.Remap != nil, change.ReduceToSurvivors != nil} {
		if present {
			variants++
		}
	}
	if variants != 1 {
		return 0, terminalFailure("InvalidChange", "a resolved plan must contain exactly one change variant")
	}
	switch change.Kind {
	case enginegroup.PlanKindGrow:
		if change.Grow == nil {
			return 0, terminalFailure("InvalidChange", "the grow tag does not match its change variant")
		}
		return a.resolveAppend(byRank, change.Grow.Replicas, enginegroup.BootstrapModeJoin)
	case enginegroup.PlanKindRestore:
		if change.Restore == nil {
			return 0, terminalFailure("InvalidChange", "the restore tag does not match its change variant")
		}
		// Re-adding a fully retired suffix uses native append, not masked-slot or arbitrary-member recovery.
		targets := make([]enginegroup.ReplicaTarget, len(change.Restore.Replicas))
		for i, replica := range change.Restore.Replicas {
			targets[i] = replica.ReplicaTarget
		}
		return a.resolveAppend(byRank, targets, enginegroup.BootstrapModeRestoreFixedSlot)
	case enginegroup.PlanKindRetire:
		if change.Retire == nil {
			return 0, terminalFailure("InvalidChange", "the retire tag does not match its change variant")
		}
		return a.resolveRetirement(byRank, change.Retire)
	default:
		return 0, terminalFailure("UnsupportedOperation", "vLLM external Elastic EP supports append and suffix retirement, not survivor reduction or remapping")
	}
}

func (a *MembershipAdapter) resolveRetirement(base []enginegroup.ReplicaMembership, retire *enginegroup.RetireChange) (int32, *enginegroup.Failure) {
	// Native planned shrink retires a complete suffix; sparse survivor masking uses a different backend contract.
	if len(retire.Replicas) == 0 {
		return 0, terminalFailure("UnsupportedRetirement", "vLLM planned shrink requires a nonempty suffix and cannot retire the whole world")
	}
	retiring := make(map[enginegroup.ReplicaID]bool, len(retire.Replicas))
	for _, id := range retire.Replicas {
		if id == "" || retiring[id] {
			return 0, terminalFailure("InvalidRetirement", "retirement requires distinct allocation identities")
		}
		retiring[id] = true
	}
	removed := 0
	found := make(map[enginegroup.ReplicaID]bool)
	for _, member := range base {
		if retiring[member.ReplicaID] {
			removed++
			found[member.ReplicaID] = true
		}
	}
	if len(found) != len(retiring) || removed == len(base) {
		return 0, terminalFailure("UnsupportedRetirement", "retirement must identify current allocations without removing the whole world")
	}
	size := int32(len(base) - removed)
	for rank, member := range base {
		if retiring[member.ReplicaID] != (int32(rank) >= size) {
			return 0, terminalFailure("UnsupportedRetirement", "vLLM planned shrink removes only the high-rank suffix")
		}
	}
	return a.validateSize(size)
}

func (a *MembershipAdapter) resolveAppend(base []enginegroup.ReplicaMembership, targets []enginegroup.ReplicaTarget, bootstrap enginegroup.BootstrapMode) (int32, *enginegroup.Failure) {
	// Reject unsafe additions without relying on native count arithmetic to validate identity ownership.
	added, err := a.Layout.NativeSize(int32(len(targets)))
	if err != nil || len(targets) == 0 || int64(len(base))+int64(added) > int64(a.MaxDataParallelSize) {
		return 0, terminalFailure("UnsupportedGrowth", "growth needs a nonempty joining suffix within the profile maximum")
	}
	ids := make(map[enginegroup.ReplicaID]bool, len(base)+len(targets))
	for _, membership := range base {
		ids[membership.ReplicaID] = true
	}
	ranks := make(map[int32]bool, len(targets))
	slots := make(map[enginegroup.CapacitySlotID]bool, len(targets))
	size := int32(len(base)) + added
	for _, replica := range targets {
		if replica.ReplicaID == "" || ids[replica.ReplicaID] || replica.SlotID == "" || slots[replica.SlotID] ||
			replica.Bootstrap != bootstrap || len(replica.NativeMembers) != int(a.Layout.NativeMembersPerReplica) {
			return 0, terminalFailure("UnsupportedRankLayout", "each new allocation needs a unique replica, slot, and its full profile-defined native-member set")
		}
		for _, member := range replica.NativeMembers {
			rank, err := nativeRank(member)
			if err != nil || rank < int32(len(base)) || rank >= size || ranks[rank] {
				return 0, terminalFailure("UnsupportedRankLayout", "joining DP ranks must form the complete contiguous suffix")
			}
			ranks[rank] = true
		}
		ids[replica.ReplicaID], slots[replica.SlotID] = true, true
	}
	return a.validateSize(size)
}

func (a *MembershipAdapter) validateSize(size int32) (int32, *enginegroup.Failure) {
	if size < a.MinDataParallelSize || size > a.MaxDataParallelSize {
		return 0, terminalFailure("TargetOutOfBounds", fmt.Sprintf("DP target %d is outside profile bounds [%d, %d]", size, a.MinDataParallelSize, a.MaxDataParallelSize))
	}
	return size, nil
}

func (a *MembershipAdapter) validateJoining(target enginegroup.MembershipTarget) error {
	// Frozen workload-process identities are not inferred from a successful native count change.
	expected := make(map[enginegroup.ReplicaID][]enginegroup.NativeMemberID)
	if grow := target.Plan.Change.Grow; grow != nil {
		for _, replica := range grow.Replicas {
			expected[replica.ReplicaID] = replica.NativeMembers
		}
	}
	if restore := target.Plan.Change.Restore; restore != nil {
		for _, replica := range restore.Replicas {
			expected[replica.ReplicaID] = replica.NativeMembers
		}
	}
	if len(target.Joining) != len(expected) {
		return fmt.Errorf("joining identities do not match the planned allocation set")
	}
	seen := make(map[enginegroup.ReplicaID]bool, len(target.Joining))
	for _, joining := range target.Joining {
		ids := expected[joining.ReplicaID]
		if seen[joining.ReplicaID] || len(joining.Members) != len(ids) || len(ids) == 0 {
			return fmt.Errorf("joining identities do not match the planned native members")
		}
		members := make(map[enginegroup.NativeMemberID]bool, len(ids))
		for _, member := range joining.Members {
			if member.RuntimeIncarnation == "" || members[member.ID] || !slices.Contains(ids, member.ID) {
				return fmt.Errorf("joining identities do not match the planned native members")
			}
			process, err := a.processForMember(member.ID)
			if err != nil || !reflect.DeepEqual(process.Membership, enginegroup.ReplicaMembership{
				ReplicaID: joining.ReplicaID, Members: []enginegroup.NativeMemberIncarnation{member},
			}) {
				return fmt.Errorf("joining member %s does not match its frozen runtime binding", member.ID)
			}
			members[member.ID] = true
		}
		seen[joining.ReplicaID] = true
	}
	return nil
}

func membersByRank(topology enginegroup.MembershipTopology) ([]enginegroup.ReplicaMembership, error) {
	if len(topology.Replicas) == 0 {
		return nil, fmt.Errorf("vLLM planned resizing requires an already-formed nonempty world")
	}
	byRank := make([]enginegroup.ReplicaMembership, topology.NativeMemberCount())
	ids := make(map[enginegroup.ReplicaID]bool, len(byRank))
	for _, membership := range topology.Replicas {
		if membership.ReplicaID == "" || ids[membership.ReplicaID] || len(membership.Members) == 0 {
			return nil, fmt.Errorf("vLLM topology requires distinct allocations with concrete native members")
		}
		for _, member := range membership.Members {
			rank, err := nativeRank(member.ID)
			if err != nil || member.RuntimeIncarnation == "" || rank >= int32(len(byRank)) || byRank[rank].ReplicaID != "" {
				return nil, fmt.Errorf("committed vLLM DP ranks must be distinct concrete members forming the contiguous prefix")
			}
			byRank[rank] = enginegroup.ReplicaMembership{ReplicaID: membership.ReplicaID, Members: []enginegroup.NativeMemberIncarnation{member}}
		}
		ids[membership.ReplicaID] = true
	}
	return byRank, nil
}

func nativeRank(id enginegroup.NativeMemberID) (int32, error) {
	rank, err := strconv.ParseInt(strings.TrimPrefix(string(id), "dp-"), 10, 32)
	if err != nil || rank < 0 || string(id) != fmt.Sprintf("dp-%d", rank) {
		return 0, fmt.Errorf("invalid native DP member %q", id)
	}
	return int32(rank), nil
}

func rejectedPreflight(reason, message string) enginegroup.PreflightResult {
	return enginegroup.PreflightResult{Rejection: terminalFailure(reason, message)}
}

func terminalFailure(reason, message string) *enginegroup.Failure {
	return &enginegroup.Failure{Classification: enginegroup.FailureClassificationTerminal, Reason: reason, Message: message}
}

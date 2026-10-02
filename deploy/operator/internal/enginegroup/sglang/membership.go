/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
)

const capabilityGeneration = "sglang-elastic-ep-growth/v1"

// CapacityObserver supplies the exact runtime incarnations correlated with
// SGLang's native dp-N ranks.
type CapacityObserver interface {
	Observe(context.Context, enginegroup.GroupID) (enginegroup.CapacityObservation, error)
}

// MembershipAdapter translates the generic compare-and-apply contract to
// SGLang's growth-only API. Its ConfigMap ledger is a compatibility boundary
// for the current SGLang API, which does not yet expose operation identities.
type MembershipAdapter struct {
	Client             *Client
	Capacity           CapacityObserver
	ProfileFingerprint string
	Journal            kubejournal.Store
}

type membershipJournal struct {
	TopologyGeneration int64              `json:"topologyGeneration"`
	EffectiveEPSize    int32              `json:"effectiveEPSize"`
	Transition         *transitionJournal `json:"transition,omitempty"`
}

type transitionJournal struct {
	TransitionID    string                                `json:"transitionID"`
	ControlRevision int64                                 `json:"controlRevision"`
	TargetDigest    string                                `json:"targetDigest"`
	BaseTopology    enginegroup.MembershipTopology        `json:"baseTopology"`
	TargetEPSize    int32                                 `json:"targetEPSize"`
	Phase           enginegroup.MembershipTransitionPhase `json:"phase"`
	ResultTopology  *enginegroup.MembershipTopology       `json:"resultTopology,omitempty"`
	Failure         *enginegroup.Failure                  `json:"failure,omitempty"`
}

// ValidatePlan accepts only the merged SGLang width-one growth semantics.
func (a *MembershipAdapter) ValidatePlan(
	_ context.Context,
	_ enginegroup.GroupID,
	request enginegroup.PlanValidationRequest,
) (enginegroup.PreflightResult, error) {
	failure := a.validatePlan(request.BaseTopology, request.Plan)
	if failure != nil {
		return enginegroup.PreflightResult{Rejection: failure}, nil
	}
	return enginegroup.PreflightResult{Evidence: &enginegroup.ValidationEvidence{
		PlanDigest:           request.PlanDigest,
		ProfileFingerprint:   a.ProfileFingerprint,
		CapabilityGeneration: capabilityGeneration,
	}}, nil
}

// ValidateTarget binds validation to the exact joining runtime identities.
func (a *MembershipAdapter) ValidateTarget(
	_ context.Context,
	_ enginegroup.GroupID,
	target enginegroup.MembershipTarget,
) (enginegroup.PreflightResult, error) {
	if failure := a.validatePlan(target.BaseTopology, target.Plan); failure != nil {
		return enginegroup.PreflightResult{Rejection: failure}, nil
	}
	if target.TargetDigest == "" || target.Validation.PlanDigest == "" {
		return rejectedPreflight("InvalidTarget", "membership target and validated plan digests are required"), nil
	}
	if target.Validation.ProfileFingerprint != a.ProfileFingerprint ||
		target.Validation.CapabilityGeneration != capabilityGeneration {
		return rejectedPreflight("CapabilitiesChanged", "profile or adapter capabilities changed after plan validation"), nil
	}
	if target.Plan.Change.Grow == nil || len(target.Joining) != len(target.Plan.Change.Grow.Replicas) {
		return rejectedPreflight("InvalidJoiningSet", "joining runtime identities do not match the grow plan"), nil
	}
	joining := make(map[enginegroup.ReplicaID]enginegroup.RuntimeIncarnationID, len(target.Joining))
	for _, member := range target.Joining {
		if member.ReplicaID == "" || member.RuntimeIncarnation == "" {
			return rejectedPreflight("InvalidJoiningSet", "joining replica and runtime identities are required"), nil
		}
		joining[member.ReplicaID] = member.RuntimeIncarnation
	}
	for _, replica := range target.Plan.Change.Grow.Replicas {
		if joining[replica.ReplicaID] == "" {
			return rejectedPreflight("InvalidJoiningSet", "every planned replica needs an exact joining incarnation"), nil
		}
	}
	return enginegroup.PreflightResult{Evidence: &enginegroup.ValidationEvidence{
		PlanDigest:           target.Validation.PlanDigest,
		TargetDigest:         target.TargetDigest,
		ProfileFingerprint:   a.ProfileFingerprint,
		CapabilityGeneration: capabilityGeneration,
	}}, nil
}

// Observe correlates SGLang's engine state with one requested transition.
func (a *MembershipAdapter) Observe(
	ctx context.Context,
	groupID enginegroup.GroupID,
	transitionID string,
) (enginegroup.MembershipObservation, error) {
	state, engineState, err := a.observeState(ctx)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
	}
	if state.TopologyGeneration == 0 {
		state.TopologyGeneration = 1
		state.EffectiveEPSize = engineState.EffectiveEPSize
		if err := a.Journal.Save(ctx, state); err != nil {
			return enginegroup.MembershipObservation{}, err
		}
	}

	committedGeneration := state.TopologyGeneration
	if engineState.EffectiveEPSize != state.EffectiveEPSize {
		// Only the exact correlated transition may advance the generation below.
		if state.Transition == nil || engineState.EffectiveEPSize != state.Transition.TargetEPSize {
			return enginegroup.MembershipObservation{}, fmt.Errorf(
				"SGLang effective EP size changed from %d to unrelated size %d",
				state.EffectiveEPSize,
				engineState.EffectiveEPSize,
			)
		}
		committedGeneration = state.Transition.BaseTopology.Generation + 1
	}
	committed, err := a.topology(ctx, groupID, committedGeneration, engineState.EffectiveEPSize)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
	}

	observation := enginegroup.MembershipObservation{
		CommittedTopology:     committed,
		RequestedTransitionID: transitionID,
	}
	if transitionID == "" || state.Transition == nil || state.Transition.TransitionID != transitionID {
		return observation, nil
	}

	record := state.Transition
	changed := false
	if record.Phase == enginegroup.MembershipTransitionPhaseRejected &&
		engineState.EffectiveEPSize == record.TargetEPSize {
		return enginegroup.MembershipObservation{}, fmt.Errorf(
			"SGLang committed transition %q after reporting an immutable rejection",
			record.TransitionID,
		)
	}
	if record.Phase == enginegroup.MembershipTransitionPhaseCommitted &&
		engineState.EffectiveEPSize != record.TargetEPSize {
		return enginegroup.MembershipObservation{}, fmt.Errorf(
			"SGLang topology diverged after committed transition %q",
			record.TransitionID,
		)
	}
	if engineState.EffectiveEPSize == record.TargetEPSize && !engineState.Scaling {
		if record.Phase != enginegroup.MembershipTransitionPhaseCommitted {
			record.Phase = enginegroup.MembershipTransitionPhaseCommitted
			record.ResultTopology = &committed
			record.Failure = nil
			state.TopologyGeneration = committed.Generation
			state.EffectiveEPSize = engineState.EffectiveEPSize
			changed = true
		}
	} else if engineState.Scaling {
		if record.Phase != enginegroup.MembershipTransitionPhasePending {
			record.Phase = enginegroup.MembershipTransitionPhasePending
			changed = true
		}
	} else if record.Phase == enginegroup.MembershipTransitionPhasePending {
		// The request was persisted before dispatch. If SGLang no longer reports
		// it and still exposes the base, current APIs cannot prove non-acceptance.
		record.Phase = enginegroup.MembershipTransitionPhaseUnknown
		record.Failure = &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationRetryable,
			Reason:         "TransitionOutcomeUnknown",
			Message:        "SGLang has no operation identity that can prove whether the persisted request was accepted",
		}
		changed = true
	}
	if changed {
		if err := a.Journal.Save(ctx, state); err != nil {
			return enginegroup.MembershipObservation{}, err
		}
	}
	result := record.ResultTopology
	if result != nil {
		copy := *result
		copy.Replicas = append([]enginegroup.ReplicaMembership(nil), result.Replicas...)
		result = &copy
	}
	observation.Transition = &enginegroup.MembershipTransitionObservation{
		TransitionID:    record.TransitionID,
		ControlRevision: record.ControlRevision,
		TargetDigest:    record.TargetDigest,
		Phase:           record.Phase,
		ResultTopology:  result,
		Failure:         record.Failure,
	}
	return observation, nil
}

// Apply persists the exact target before invoking SGLang. It never reissues a
// Pending or Unknown request because the current backend API cannot correlate it.
func (a *MembershipAdapter) Apply(
	ctx context.Context,
	groupID enginegroup.GroupID,
	target enginegroup.MembershipTarget,
) error {
	state, engineState, err := a.observeState(ctx)
	if err != nil {
		return err
	}
	if state.TopologyGeneration == 0 {
		return fmt.Errorf("membership baseline must be observed before Apply")
	}
	if state.Transition != nil {
		record := state.Transition
		if record.TransitionID == target.TransitionID {
			if record.ControlRevision != target.ControlRevision || record.TargetDigest != target.TargetDigest {
				return fmt.Errorf("membership transition %q was reused with different content", target.TransitionID)
			}
			return nil
		}
		if record.Phase == enginegroup.MembershipTransitionPhasePending ||
			record.Phase == enginegroup.MembershipTransitionPhaseUnknown {
			return fmt.Errorf("membership transition %q is still %s", record.TransitionID, record.Phase)
		}
	}
	current, err := a.topology(ctx, groupID, state.TopologyGeneration, engineState.EffectiveEPSize)
	if err != nil {
		return err
	}
	if !enginegroup.SameTopology(current, target.BaseTopology) {
		return a.persistRejection(ctx, state, target, "BaseTopologyChanged", "SGLang topology no longer matches the validated base")
	}
	targetSize := target.BaseTopology.ReplicaCount() + int32(len(target.Joining))
	state.Transition = &transitionJournal{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		BaseTopology:    target.BaseTopology,
		TargetEPSize:    targetSize,
		Phase:           enginegroup.MembershipTransitionPhasePending,
	}
	if err := a.Journal.Save(ctx, state); err != nil {
		return err
	}

	response, err := a.Client.Scale(ctx, targetSize)
	if err != nil {
		state.Transition.Phase = enginegroup.MembershipTransitionPhaseUnknown
		state.Transition.Failure = &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationRetryable,
			Reason:         "ScaleRequestAmbiguous",
			Message:        err.Error(),
		}
		if saveErr := a.Journal.Save(ctx, state); saveErr != nil {
			return fmt.Errorf("SGLang scale request failed (%v) and persist Unknown failed: %w", err, saveErr)
		}
		return err
	}
	if response.Status != "ok" {
		return a.persistRejection(ctx, state, target, "ScaleRejected", response.Message)
	}
	return nil
}

func (a *MembershipAdapter) observeState(ctx context.Context) (membershipJournal, ScaleState, error) {
	if a.Client == nil || a.Capacity == nil {
		return membershipJournal{}, ScaleState{}, fmt.Errorf("SGLang membership client and capacity observer are required")
	}
	state := membershipJournal{}
	_, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return membershipJournal{}, ScaleState{}, err
	}
	engineState, err := a.Client.Observe(ctx)
	if err != nil {
		return membershipJournal{}, ScaleState{}, err
	}
	return state, engineState, nil
}

func (a *MembershipAdapter) topology(
	ctx context.Context,
	groupID enginegroup.GroupID,
	generation int64,
	size int32,
) (enginegroup.MembershipTopology, error) {
	return topologyFromCapacity(ctx, a.Capacity, groupID, generation, size)
}

func (a *MembershipAdapter) validatePlan(
	base enginegroup.MembershipTopology,
	plan enginegroup.ResolvedPlan,
) *enginegroup.Failure {
	if plan.ProfileFingerprint != a.ProfileFingerprint {
		return terminalFailure("ProfileMismatch", "plan fingerprint does not match the SGLang runtime profile")
	}
	if plan.ProcessLifecycleOwner != enginegroup.ProcessLifecycleOwnerOrchestrator {
		return terminalFailure("UnsupportedLifecycleOwner", "SGLang external joining capacity must be orchestrator-owned")
	}
	if plan.Change.Kind != enginegroup.PlanKindGrow || plan.Change.Grow == nil {
		return terminalFailure("UnsupportedOperation", "the current SGLang adapter supports growth only")
	}
	if plan.TrafficRequirement != enginegroup.TrafficRequirementKeepServing ||
		plan.VerificationRequirement != enginegroup.VerificationRequirementRequired {
		return terminalFailure("UnsafePlan", "SGLang growth must keep the base serving and verify the committed result")
	}
	expected := int(base.ReplicaCount())
	targets := make(map[enginegroup.ReplicaID]enginegroup.ReplicaTarget, len(plan.Change.Grow.Replicas))
	for _, target := range plan.Change.Grow.Replicas {
		if _, duplicate := targets[target.ReplicaID]; duplicate {
			return terminalFailure("UnsupportedRankLayout", "SGLang growth contains a duplicate replica identity")
		}
		targets[target.ReplicaID] = target
	}
	for rank := expected; rank < expected+len(targets); rank++ {
		replicaID := enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank))
		target, found := targets[replicaID]
		if !found {
			return terminalFailure("UnsupportedRankLayout", "SGLang growth requires a contiguous replica suffix")
		}
		if target.ReplicaID != enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)) ||
			target.SlotID != enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)) ||
			target.Bootstrap != enginegroup.BootstrapModeJoin ||
			len(target.NativeMembers) != 1 || target.NativeMembers[0] != enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank)) {
			return terminalFailure("UnsupportedRankLayout", "SGLang growth requires contiguous replica-N, slot-N, and dp-N identities")
		}
	}
	return nil
}

func (a *MembershipAdapter) persistRejection(
	ctx context.Context,
	state membershipJournal,
	target enginegroup.MembershipTarget,
	reason string,
	message string,
) error {
	state.Transition = &transitionJournal{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		BaseTopology:    target.BaseTopology,
		Phase:           enginegroup.MembershipTransitionPhaseRejected,
		Failure:         terminalFailure(reason, message),
	}
	return a.Journal.Save(ctx, state)
}

func rejectedPreflight(reason, message string) enginegroup.PreflightResult {
	return enginegroup.PreflightResult{Rejection: terminalFailure(reason, message)}
}

func terminalFailure(reason, message string) *enginegroup.Failure {
	return &enginegroup.Failure{
		Classification: enginegroup.FailureClassificationTerminal,
		Reason:         reason,
		Message:        message,
	}
}

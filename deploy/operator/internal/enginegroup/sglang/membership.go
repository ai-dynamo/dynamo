/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"fmt"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
)

const capabilityGeneration = "sglang-elastic-ep-correlated-growth/v1"

// CapacityObserver supplies the exact runtime incarnations correlated with
// SGLang's native dp-N ranks.
type CapacityObserver interface {
	Observe(context.Context, enginegroup.GroupID) (enginegroup.CapacityObservation, error)
}

// GrowthAdapter correlates native SGLang operations with the durable control target.
// The journal retains the control revision, digest, and topology projection that
// SGLang does not expose. It never infers an operation commit from size alone.
type GrowthAdapter struct {
	Client             *Client
	Capacity           CapacityObserver
	ProfileFingerprint string
	MembersPerReplica  int32
	InitialEPSize      int32
	Journal            kubejournal.Store
}

type membershipJournal struct {
	InstanceID           string             `json:"instanceID"`
	CommittedOperationID string             `json:"committedOperationID"`
	TopologyGeneration   int64              `json:"topologyGeneration"`
	EffectiveEPSize      int32              `json:"effectiveEPSize"`
	Transition           *transitionJournal `json:"transition,omitempty"`
}

type transitionJournal struct {
	TransitionID         string                                `json:"transitionID"`
	ControlRevision      int64                                 `json:"controlRevision"`
	TargetDigest         string                                `json:"targetDigest"`
	BaseTopology         enginegroup.MembershipTopology        `json:"baseTopology"`
	TargetEPSize         int32                                 `json:"targetEPSize"`
	JoiningAllocationIDs []string                              `json:"joiningAllocationIDs"`
	Phase                enginegroup.MembershipTransitionPhase `json:"phase"`
	ResultTopology       *enginegroup.MembershipTopology       `json:"resultTopology,omitempty"`
	Failure              *enginegroup.Failure                  `json:"failure,omitempty"`
}

// ValidatePlan accepts one complete native append allocation with post-commit verification.
func (a *GrowthAdapter) ValidatePlan(
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
func (a *GrowthAdapter) ValidateTarget(
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
		if member.ReplicaID == "" || int32(len(member.Members)) != a.MembersPerReplica || member.Members[0].RuntimeIncarnation == "" {
			return rejectedPreflight("InvalidJoiningSet", "joining replica and runtime identities are required"), nil
		}
		joining[member.ReplicaID] = member.Members[0].RuntimeIncarnation
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
func (a *GrowthAdapter) Observe(
	ctx context.Context,
	groupID enginegroup.GroupID,
	transitionID string,
) (enginegroup.MembershipObservation, error) {
	state, snapshot, engineState, err := a.observeState(ctx)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
	}
	if state.TopologyGeneration == 0 {
		if engineState.OperationID != "" || engineState.Scaling {
			return enginegroup.MembershipObservation{}, fmt.Errorf("cannot adopt an in-flight or previously resized SGLang world without its control journal")
		}
		state.InstanceID = engineState.InstanceID
		state.TopologyGeneration = 1
		state.EffectiveEPSize = engineState.EffectiveEPSize
		snapshot, err = a.Journal.Save(ctx, snapshot, state)
		if err != nil {
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
	committed, err := topologyFromCapacity(ctx, a.Capacity, groupID, committedGeneration, engineState.EffectiveEPSize)
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
	// A local compare-and-apply rejection never reached the engine and remains valid only on its unchanged base.
	if record.Phase == enginegroup.MembershipTransitionPhaseRejected &&
		engineState.OperationID != record.TransitionID && !engineState.Scaling && enginegroup.SameTopology(committed, record.BaseTopology) {
		observation.Transition = &enginegroup.MembershipTransitionObservation{
			TransitionID: record.TransitionID, ControlRevision: record.ControlRevision, TargetDigest: record.TargetDigest,
			Phase: record.Phase, Failure: record.Failure,
		}
		return observation, nil
	}
	// Native operation and allocation correlation is required even when the effective size looks correct.
	if engineState.OperationID != record.TransitionID || engineState.TargetEPSize != record.TargetEPSize ||
		!slices.Equal(engineState.JoiningAllocationIDs, record.JoiningAllocationIDs) {
		if engineState.OperationID != state.CommittedOperationID || engineState.Scaling {
			return enginegroup.MembershipObservation{}, fmt.Errorf("SGLang reports an uncorrelated operation %q", engineState.OperationID)
		}
		if engineState.EffectiveEPSize != state.EffectiveEPSize {
			return enginegroup.MembershipObservation{}, fmt.Errorf("SGLang changed membership without matching operation and joining identities")
		}
		observation.Transition = &enginegroup.MembershipTransitionObservation{
			TransitionID: record.TransitionID, ControlRevision: record.ControlRevision,
			TargetDigest: record.TargetDigest, Phase: enginegroup.MembershipTransitionPhaseUnknown,
			Failure: &enginegroup.Failure{Classification: enginegroup.FailureClassificationRetryable,
				Reason: "SubmissionNotObserved", Message: "the exact native operation is not yet observable; replay the identical target"},
		}
		return observation, nil
	}

	// A terminal record cannot be repurposed as evidence for an uncorrelated later collective.
	if engineState.Scaling && (record.Phase == enginegroup.MembershipTransitionPhaseCommitted ||
		record.Phase == enginegroup.MembershipTransitionPhaseRejected) {
		return enginegroup.MembershipObservation{}, fmt.Errorf(
			"SGLang reports scaling after terminal transition %q", record.TransitionID,
		)
	}
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
	if engineState.EffectiveEPSize == record.TargetEPSize && !engineState.Scaling &&
		engineState.OperationSucceeded != nil && *engineState.OperationSucceeded {
		if record.Phase != enginegroup.MembershipTransitionPhaseCommitted {
			record.Phase = enginegroup.MembershipTransitionPhaseCommitted
			record.ResultTopology = &committed
			record.Failure = nil
			state.TopologyGeneration = committed.Generation
			state.EffectiveEPSize = engineState.EffectiveEPSize
			state.CommittedOperationID = engineState.OperationID
			changed = true
		}
	} else if engineState.Scaling && engineState.Phase != "submission_unknown" {
		if record.Phase != enginegroup.MembershipTransitionPhasePending || record.Failure != nil {
			record.Phase = enginegroup.MembershipTransitionPhasePending
			record.Failure = nil
			record.ResultTopology = nil
			changed = true
		}
	} else if !engineState.Scaling && engineState.OperationSucceeded != nil && !*engineState.OperationSucceeded &&
		enginegroup.SameTopology(committed, record.BaseTopology) {
		record.Phase = enginegroup.MembershipTransitionPhaseRejected
		record.Failure = terminalFailure("ScaleRejected", engineState.LastError)
		changed = true
	} else if record.Phase == enginegroup.MembershipTransitionPhasePending {
		// An accepted request without a terminal native result remains ambiguous and replayable.
		record.Phase = enginegroup.MembershipTransitionPhaseUnknown
		record.Failure = &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationRetryable,
			Reason:         "TransitionOutcomeUnknown",
			Message:        "the correlated native operation has no definitive terminal result",
		}
		changed = true
	}
	if changed {
		if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
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

// Apply persists the exact target before dispatch and safely replays it using native operation IDs.
func (a *GrowthAdapter) Apply(
	ctx context.Context,
	groupID enginegroup.GroupID,
	target enginegroup.MembershipTarget,
) error {
	state, snapshot, engineState, err := a.observeState(ctx)
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
			if record.Phase == enginegroup.MembershipTransitionPhaseCommitted || record.Phase == enginegroup.MembershipTransitionPhaseRejected {
				return nil
			}
			// An unchanged replay still needs compare-and-swap authority; a stale reader cannot replace a newer operation.
			if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
				return err
			}
			_, err := a.Client.Scale(ctx, ScaleRequest{NewEPSize: record.TargetEPSize, OperationID: record.TransitionID,
				ExpectedInstanceID: state.InstanceID, ExpectedJoiningAllocationIDs: record.JoiningAllocationIDs})
			return err
		}
		if record.Phase == enginegroup.MembershipTransitionPhasePending ||
			record.Phase == enginegroup.MembershipTransitionPhaseUnknown {
			return fmt.Errorf("membership transition %q is still %s", record.TransitionID, record.Phase)
		}
	}
	current, err := topologyFromCapacity(ctx, a.Capacity, groupID, state.TopologyGeneration, engineState.EffectiveEPSize)
	if err != nil {
		return err
	}
	if !enginegroup.SameTopology(current, target.BaseTopology) {
		return a.persistRejection(ctx, state, snapshot, target, "BaseTopologyChanged", "SGLang topology no longer matches the validated base")
	}
	targetSize := (target.BaseTopology.ReplicaCount() + int32(len(target.Joining))) * a.MembersPerReplica
	joiningAllocationIDs, err := a.joiningAllocationIDs(ctx, groupID, target)
	if err != nil {
		return err
	}
	state.Transition = &transitionJournal{
		TransitionID:         target.TransitionID,
		ControlRevision:      target.ControlRevision,
		TargetDigest:         target.TargetDigest,
		BaseTopology:         target.BaseTopology,
		TargetEPSize:         targetSize,
		JoiningAllocationIDs: joiningAllocationIDs,
		Phase:                enginegroup.MembershipTransitionPhasePending,
	}
	snapshot, err = a.Journal.Save(ctx, snapshot, state)
	if err != nil {
		return err
	}

	response, err := a.Client.Scale(ctx, ScaleRequest{NewEPSize: targetSize, OperationID: target.TransitionID,
		ExpectedInstanceID: state.InstanceID, ExpectedJoiningAllocationIDs: joiningAllocationIDs})

	// A backend exception or unsuccessful response is not proof that the dispatched collective cannot commit later.
	if err == nil && response.Status != "ok" {
		err = fmt.Errorf("SGLang scale outcome is ambiguous: status=%q pending_ep_size=%d: %s",
			response.Status, response.PendingEPSize, response.Message)
	}
	if err != nil {
		state.Transition.Phase = enginegroup.MembershipTransitionPhaseUnknown
		state.Transition.Failure = &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationRetryable,
			Reason:         "ScaleRequestAmbiguous",
			Message:        err.Error(),
		}
		if _, saveErr := a.Journal.Save(ctx, snapshot, state); saveErr != nil {
			return fmt.Errorf("SGLang scale request failed (%v) and persist Unknown failed: %w", err, saveErr)
		}
		return err
	}
	return nil
}

func (a *GrowthAdapter) observeState(ctx context.Context) (membershipJournal, kubejournal.Snapshot, ScaleState, error) {
	if a.Client == nil || a.Capacity == nil {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("SGLang membership client and capacity observer are required")
	}
	state := membershipJournal{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, err
	}
	if snapshot.Exists() && (state.InstanceID == "" || state.TopologyGeneration <= 0 || state.EffectiveEPSize <= 0) {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("%w: SGLang journal lost its native runtime or committed topology binding", enginegroup.ErrRecoveryRequired)
	}
	engineState, err := a.Client.Observe(ctx)
	if err != nil {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, err
	}
	if a.MembersPerReplica < 1 || engineState.InstanceID == "" || engineState.AllocationWidth != a.MembersPerReplica {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("SGLang must report its native runtime ID and the profile's allocation width")
	}
	if a.InitialEPSize > 0 && engineState.InitialEPSize != a.InitialEPSize {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("%w: native initial EP size differs from the immutable profile", enginegroup.ErrRecoveryRequired)
	}
	if state.InstanceID != "" && state.InstanceID != engineState.InstanceID {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("%w: SGLang runtime instance changed", enginegroup.ErrRecoveryRequired)
	}
	return state, snapshot, engineState, nil
}

func (a *GrowthAdapter) validatePlan(
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
	if a.MembersPerReplica < 1 {
		return terminalFailure("InvalidProfile", "SGLang member width must be resolved before validating operations")
	}

	// The native protocol admits one allocation per resize; it may contain several members.
	if len(plan.Change.Grow.Replicas) != 1 {
		return terminalFailure("UnsupportedJoiningSet", "SGLang supports one joining allocation per resize")
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
		expectedMembers := make([]enginegroup.NativeMemberID, a.MembersPerReplica)
		for local := range expectedMembers {
			expectedMembers[local] = enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", int32(rank)*a.MembersPerReplica+int32(local)))
		}
		if target.ReplicaID != enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)) ||
			target.SlotID != enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)) ||
			target.Bootstrap != enginegroup.BootstrapModeJoin ||
			!slices.Equal(target.NativeMembers, expectedMembers) {
			return terminalFailure("UnsupportedRankLayout", "SGLang growth requires contiguous replica-N, slot-N, and dp-N identities")
		}
	}
	return nil
}

func (a *GrowthAdapter) joiningAllocationIDs(ctx context.Context, groupID enginegroup.GroupID, target enginegroup.MembershipTarget) ([]string, error) {
	// Native allocation IDs are downward-API Pod UIDs, not mutable names or logical replica IDs.
	capacity, err := a.Capacity.Observe(ctx, groupID)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, joining := range target.Joining {
		for _, allocation := range capacity.Allocations {
			if allocation.Incarnation.ReplicaID != joining.ReplicaID {
				continue
			}
			if !slices.Equal(allocation.Incarnation.Members, joining.Members) || len(allocation.Incarnation.CapacityRefs) != 1 {
				return nil, fmt.Errorf("joining allocation changed before native submission")
			}
			ids = append(ids, string(allocation.Incarnation.CapacityRefs[0].UID))
		}
	}
	if len(ids) != 1 || ids[0] == "" {
		return nil, fmt.Errorf("one exact joining Pod UID is required")
	}
	return ids, nil
}

func (a *GrowthAdapter) persistRejection(
	ctx context.Context,
	state membershipJournal,
	snapshot kubejournal.Snapshot,
	target enginegroup.MembershipTarget,
	reason string,
	message string,
) error {
	state.Transition = &transitionJournal{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		BaseTopology:    target.BaseTopology,
		TargetEPSize:    (target.BaseTopology.ReplicaCount() + int32(len(target.Joining))) * a.MembersPerReplica,
		Phase:           enginegroup.MembershipTransitionPhaseRejected,
		Failure:         terminalFailure(reason, message),
	}
	_, err := a.Journal.Save(ctx, snapshot, state)
	return err
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

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

const capabilityGeneration = "sglang-elastic-ep-correlated-resize/v1"

// CapacityObserver supplies the exact runtime incarnations correlated with
// SGLang's native dp-N ranks.
type CapacityObserver interface {
	Observe(context.Context, enginegroup.GroupID) (enginegroup.CapacityObservation, error)
}

// MembershipAdapter correlates native SGLang operations with the durable control target.
// The journal retains the control revision, digest, and topology projection that
// SGLang does not expose. It never infers an operation commit from size alone.
type MembershipAdapter struct {
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
	TopologyToken        string             `json:"topologyToken"`
	Transition           *transitionJournal `json:"transition,omitempty"`
}

type transitionJournal struct {
	TransitionID         string                                `json:"transitionID"`
	ControlRevision      int64                                 `json:"controlRevision"`
	TargetDigest         string                                `json:"targetDigest"`
	BaseTopology         enginegroup.MembershipTopology        `json:"baseTopology"`
	TargetEPSize         int32                                 `json:"targetEPSize"`
	BaseTopologyToken    string                                `json:"baseTopologyToken"`
	JoiningAllocationIDs []string                              `json:"joiningAllocationIDs"`
	Retiring             []enginegroup.ReplicaMembership       `json:"retiring,omitempty"`
	Phase                enginegroup.MembershipTransitionPhase `json:"phase"`
	ResultTopology       *enginegroup.MembershipTopology       `json:"resultTopology,omitempty"`
	Failure              *enginegroup.Failure                  `json:"failure,omitempty"`
}

// ValidatePlan accepts one complete append or healthy suffix retirement with serving verification.
func (a *MembershipAdapter) ValidatePlan(
	ctx context.Context,
	_ enginegroup.GroupID,
	request enginegroup.PlanValidationRequest,
) (enginegroup.PreflightResult, error) {
	failure := a.validatePlan(request.BaseTopology, request.Plan)
	if failure != nil {
		return enginegroup.PreflightResult{Rejection: failure}, nil
	}
	if failure, err := a.validateNativeCapabilities(ctx, request.Plan); failure != nil || err != nil {
		return enginegroup.PreflightResult{Rejection: failure}, err
	}
	if err := a.validateBase(ctx, request.BaseTopology); err != nil {
		return enginegroup.PreflightResult{}, err
	}
	return enginegroup.PreflightResult{Evidence: &enginegroup.ValidationEvidence{
		PlanDigest:           request.PlanDigest,
		ProfileFingerprint:   a.ProfileFingerprint,
		CapabilityGeneration: capabilityGeneration,
	}}, nil
}

// ValidateTarget binds validation to the exact joining runtime identities.
func (a *MembershipAdapter) ValidateTarget(
	ctx context.Context,
	_ enginegroup.GroupID,
	target enginegroup.MembershipTarget,
) (enginegroup.PreflightResult, error) {
	if failure := a.validatePlan(target.BaseTopology, target.Plan); failure != nil {
		return enginegroup.PreflightResult{Rejection: failure}, nil
	}
	if failure, err := a.validateNativeCapabilities(ctx, target.Plan); failure != nil || err != nil {
		return enginegroup.PreflightResult{Rejection: failure}, err
	}
	if err := a.validateBase(ctx, target.BaseTopology); err != nil {
		return enginegroup.PreflightResult{}, err
	}
	if target.TargetDigest == "" || target.Validation.PlanDigest == "" {
		return rejectedPreflight("InvalidTarget", "membership target and validated plan digests are required"), nil
	}
	if target.Validation.ProfileFingerprint != a.ProfileFingerprint ||
		target.Validation.CapabilityGeneration != capabilityGeneration {
		return rejectedPreflight("CapabilitiesChanged", "profile or adapter capabilities changed after plan validation"), nil
	}
	expectedJoining := 0
	if target.Plan.Change.Grow != nil {
		expectedJoining = len(target.Plan.Change.Grow.Replicas)
	}
	if len(target.Joining) != expectedJoining {
		return rejectedPreflight("InvalidJoiningSet", "joining runtime identities do not match the grow plan"), nil
	}
	joining := make(map[enginegroup.ReplicaID]enginegroup.RuntimeIncarnationID, len(target.Joining))
	for _, member := range target.Joining {
		if member.ReplicaID == "" || int32(len(member.Members)) != a.MembersPerReplica || member.Members[0].RuntimeIncarnation == "" {
			return rejectedPreflight("InvalidJoiningSet", "joining replica and runtime identities are required"), nil
		}
		joining[member.ReplicaID] = member.Members[0].RuntimeIncarnation
	}
	if target.Plan.Change.Grow != nil {
		for _, replica := range target.Plan.Change.Grow.Replicas {
			if joining[replica.ReplicaID] == "" {
				return rejectedPreflight("InvalidJoiningSet", "every planned replica needs an exact joining incarnation"), nil
			}
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
	state, snapshot, engineState, err := a.observeState(ctx)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
	}
	if state.TopologyGeneration == 0 {
		if engineState.OperationID != "" || engineState.Scaling {
			return enginegroup.MembershipObservation{}, fmt.Errorf("cannot adopt an in-flight or previously resized SGLang world without its control journal")
		}
		state.InstanceID = engineState.InstanceID
		state.TopologyToken = engineState.TopologyToken
		state.TopologyGeneration = 1
		state.EffectiveEPSize = engineState.EffectiveEPSize
		snapshot, err = a.Journal.Save(ctx, snapshot, state)
		if err != nil {
			return enginegroup.MembershipObservation{}, err
		}
	}

	committedGeneration, err := observedTopologyGeneration(state, engineState)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
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

	changed, err := recordNativeOutcome(&state, engineState, committed)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
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

// observedTopologyGeneration requires correlated terminal success before advancing membership.
// Effective size can change before survivor warmup succeeds; size alone is not commit proof.
func observedTopologyGeneration(state membershipJournal, native ScaleState) (int64, error) {
	if native.EffectiveEPSize == state.EffectiveEPSize {
		return state.TopologyGeneration, nil
	}
	if state.Transition == nil || native.EffectiveEPSize != state.Transition.TargetEPSize {
		return 0, fmt.Errorf("SGLang effective EP size changed from %d to unrelated size %d",
			state.EffectiveEPSize, native.EffectiveEPSize)
	}

	// A failed or still-running collective cannot establish a new committed topology.
	record := state.Transition
	if native.OperationID != record.TransitionID || native.TargetEPSize != record.TargetEPSize ||
		!slices.Equal(native.JoiningAllocationIDs, record.JoiningAllocationIDs) ||
		native.Scaling || native.OperationSucceeded == nil || !*native.OperationSucceeded {
		return 0, fmt.Errorf("SGLang changed effective EP size without a correlated successful commit")
	}
	return record.BaseTopology.Generation + 1, nil
}

// recordNativeOutcome updates only the already-correlated transition from native evidence.
// state and state.Transition must not be nil. The caller persists any reported change.
func recordNativeOutcome(state *membershipJournal, engineState ScaleState, committed enginegroup.MembershipTopology) (bool, error) {
	// Terminal outcomes cannot be repurposed for a later or contradictory collective.
	record := state.Transition
	if engineState.Scaling && (record.Phase == enginegroup.MembershipTransitionPhaseCommitted ||
		record.Phase == enginegroup.MembershipTransitionPhaseRejected) {
		return false, fmt.Errorf("SGLang reports scaling after terminal transition %q", record.TransitionID)
	}
	if record.Phase == enginegroup.MembershipTransitionPhaseRejected && engineState.EffectiveEPSize == record.TargetEPSize {
		return false, fmt.Errorf("SGLang committed transition %q after reporting an immutable rejection", record.TransitionID)
	}
	if record.Phase == enginegroup.MembershipTransitionPhaseCommitted && engineState.EffectiveEPSize != record.TargetEPSize {
		return false, fmt.Errorf("SGLang topology diverged after committed transition %q", record.TransitionID)
	}

	// Only native terminal success advances the committed topology generation.
	changed := false
	if engineState.EffectiveEPSize == record.TargetEPSize && !engineState.Scaling &&
		engineState.OperationSucceeded != nil && *engineState.OperationSucceeded {
		if record.Phase != enginegroup.MembershipTransitionPhaseCommitted {
			record.Phase = enginegroup.MembershipTransitionPhaseCommitted
			record.ResultTopology = &committed
			record.Failure = nil
			state.TopologyGeneration = committed.Generation
			state.EffectiveEPSize = engineState.EffectiveEPSize
			state.CommittedOperationID = engineState.OperationID
			state.TopologyToken = engineState.TopologyToken
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
		state.TopologyToken = engineState.TopologyToken
		changed = true
	} else if record.Phase == enginegroup.MembershipTransitionPhasePending {
		// Accepted requests without a terminal native result remain ambiguous and replayable.
		record.Phase = enginegroup.MembershipTransitionPhaseUnknown
		record.Failure = &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationRetryable,
			Reason:         "TransitionOutcomeUnknown",
			Message:        "the correlated native operation has no definitive terminal result",
		}
		changed = true
	}
	return changed, nil
}

// Apply persists the exact target before dispatch and safely replays it using native operation IDs.
func (a *MembershipAdapter) Apply(
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
				ExpectedInstanceID: state.InstanceID, ExpectedTopologyToken: record.BaseTopologyToken, ExpectedJoiningAllocationIDs: record.JoiningAllocationIDs})
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
	if engineState.TopologyToken != state.TopologyToken {
		return fmt.Errorf("SGLang base topology token changed before target submission")
	}
	if failure := a.validatePlan(target.BaseTopology, target.Plan); failure != nil {
		return a.persistRejection(ctx, state, snapshot, target, failure.Reason, failure.Message)
	}
	if failure, err := a.validateNativeCapabilities(ctx, target.Plan); err != nil {
		return err
	} else if failure != nil {
		return a.persistRejection(ctx, state, snapshot, target, failure.Reason, failure.Message)
	}
	targetSize := a.targetEPSize(target)
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
		BaseTopologyToken:    state.TopologyToken,
		JoiningAllocationIDs: joiningAllocationIDs,
		Retiring:             retiringMembership(target.BaseTopology, target.Plan),
		Phase:                enginegroup.MembershipTransitionPhasePending,
	}
	snapshot, err = a.Journal.Save(ctx, snapshot, state)
	if err != nil {
		return err
	}

	response, err := a.Client.Scale(ctx, ScaleRequest{NewEPSize: targetSize, OperationID: target.TransitionID,
		ExpectedInstanceID: state.InstanceID, ExpectedTopologyToken: state.TopologyToken, ExpectedJoiningAllocationIDs: joiningAllocationIDs})

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

func (a *MembershipAdapter) observeState(ctx context.Context) (membershipJournal, kubejournal.Snapshot, ScaleState, error) {
	if a.Client == nil || a.Capacity == nil {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("SGLang membership client and capacity observer are required")
	}
	state := membershipJournal{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, err
	}
	if snapshot.Exists() && (state.InstanceID == "" || state.TopologyToken == "" || state.TopologyGeneration <= 0 || state.EffectiveEPSize <= 0) {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("%w: SGLang journal lost its native runtime or committed topology binding", enginegroup.ErrRecoveryRequired)
	}
	engineState, err := a.Client.Observe(ctx)
	if err != nil {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, err
	}
	if engineState.TopologyToken == "" {
		return membershipJournal{}, kubejournal.Snapshot{}, ScaleState{}, fmt.Errorf("SGLang requires opaque native base-topology fencing")
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
	if (plan.Change.Kind != enginegroup.PlanKindGrow || plan.Change.Grow == nil) &&
		(plan.Change.Kind != enginegroup.PlanKindRetire || plan.Change.Retire == nil) {
		return terminalFailure("UnsupportedOperation", "SGLang supports healthy append growth and complete suffix retirement only")
	}
	if a.MembersPerReplica < 1 {
		return terminalFailure("InvalidProfile", "SGLang member width must be resolved before validating operations")
	}
	// Healthy resizing cannot reinterpret a partially active allocation as a smaller complete replica.
	for rank := int32(0); rank < base.ReplicaCount(); rank++ {
		id := enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank))
		if !slices.ContainsFunc(base.Replicas, func(replica enginegroup.ReplicaMembership) bool {
			return replica.ReplicaID == id && hasNativeMemberRange(replica.Members, rank*a.MembersPerReplica, a.MembersPerReplica)
		}) {
			return terminalFailure("PartialMembershipUnsupported", "SGLang planned resize requires healthy complete contiguous allocations")
		}
	}
	if plan.Change.Kind == enginegroup.PlanKindRetire {
		if a.InitialEPSize < 2 {
			return terminalFailure("InvalidProfile", "SGLang retirement requires the immutable initial EP size")
		}
		if len(plan.Change.Retire.Replicas) != 1 || base.ReplicaCount() < 2 ||
			plan.Change.Retire.Replicas[0] != enginegroup.ReplicaID(fmt.Sprintf("replica-%d", base.ReplicaCount()-1)) {
			return terminalFailure("UnsupportedRetirementSet", "SGLang retirement removes one complete highest-rank allocation, never the primary")
		}
		if (base.ReplicaCount()-1)*a.MembersPerReplica < a.InitialEPSize {
			return terminalFailure("InitialWorldMinimum", "SGLang planned shrink must retain its complete initial world")
		}
		if plan.TrafficRequirement != enginegroup.TrafficRequirementEngineManagedDrain ||
			plan.VerificationRequirement != enginegroup.VerificationRequirementRequired {
			return terminalFailure("UnsafePlan", "SGLang retirement requires native graceful drain and verification of the survivors")
		}
		return nil
	}

	// The native protocol admits one allocation per resize; it may contain several members.
	if len(plan.Change.Grow.Replicas) != 1 {
		return terminalFailure("UnsupportedJoiningSet", "SGLang supports one joining allocation per resize")
	}
	if plan.TrafficRequirement != enginegroup.TrafficRequirementKeepServing ||
		plan.VerificationRequirement != enginegroup.VerificationRequirementRequired {
		return terminalFailure("UnsafePlan", "SGLang growth must keep the base serving and verify the committed result")
	}
	// Canonical string ordering must not change which fixed rank range an allocation owns.
	rank := base.ReplicaCount()
	target := plan.Change.Grow.Replicas[0]
	members := slices.Clone(target.NativeMembers)
	slices.Sort(members)
	if target.ReplicaID != enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)) ||
		target.SlotID != enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)) ||
		target.Bootstrap != enginegroup.BootstrapModeJoin ||
		!slices.Equal(members, nativeMemberIDs(rank*a.MembersPerReplica, a.MembersPerReplica)) {
		return terminalFailure("UnsupportedRankLayout", "SGLang growth requires contiguous replica-N, slot-N, and dp-N identities")
	}
	return nil
}

func (a *MembershipAdapter) validateNativeCapabilities(ctx context.Context, plan enginegroup.ResolvedPlan) (*enginegroup.Failure, error) {
	if a.Client == nil {
		return nil, fmt.Errorf("SGLang membership requires a native control client")
	}
	state, err := a.Client.Observe(ctx)
	if err != nil {
		return nil, err
	}
	if plan.Change.Kind == enginegroup.PlanKindGrow {
		if state.MaxCommittedEPSize < state.EffectiveEPSize {
			return nil, fmt.Errorf("SGLang must report a maximum committed EP size covering its current membership")
		}
		// Auto bootstrap appends never-used slots; reusing a retired range requires native recovery.
		if state.MaxCommittedEPSize > state.EffectiveEPSize {
			return terminalFailure("SlotRecoveryUnsupported", "native automatic bootstrap cannot refill previously occupied SGLang ranks"), nil
		}
		return nil, nil
	}
	// A growth-only engine, or self-exiting retirees, cannot satisfy the external release contract.
	if state.RetireeLifecycle != "external" || state.SafeToTerminateRanks == nil {
		return terminalFailure("NativeRetirementUnsupported", "SGLang must advertise external retiree lifecycle and exact safe-to-terminate ranks"), nil
	}
	return nil, nil
}

func (a *MembershipAdapter) validateBase(ctx context.Context, base enginegroup.MembershipTopology) error {
	// Healthy resizing starts only from the bound, settled native world; a plan
	// cannot use a matching count to conceal another accepted operation or epoch.
	state, snapshot, native, err := a.observeState(ctx)
	if err != nil {
		return err
	}
	if !snapshot.Exists() || native.Scaling || state.TopologyGeneration != base.Generation ||
		state.EffectiveEPSize != native.EffectiveEPSize || native.TopologyToken != state.TopologyToken {
		return fmt.Errorf("SGLang preflight requires its unchanged settled membership baseline")
	}
	if state.Transition != nil && (state.Transition.Phase == enginegroup.MembershipTransitionPhasePending ||
		state.Transition.Phase == enginegroup.MembershipTransitionPhaseUnknown) {
		return fmt.Errorf("SGLang preflight cannot supersede an unresolved transition")
	}
	if native.OperationID != state.CommittedOperationID && !(state.Transition != nil &&
		state.Transition.Phase == enginegroup.MembershipTransitionPhaseRejected && native.OperationID == state.Transition.TransitionID) {
		return fmt.Errorf("SGLang preflight observed an uncorrelated native operation")
	}
	current, err := topologyFromCapacity(ctx, a.Capacity, enginegroup.GroupID(a.Journal.Owner.UID), state.TopologyGeneration, native.EffectiveEPSize)
	if err != nil {
		return err
	}
	if !enginegroup.SameTopology(current, base) {
		return fmt.Errorf("SGLang preflight topology no longer matches its exact base")
	}
	return nil
}

func nativeMemberIDs(first, width int32) []enginegroup.NativeMemberID {
	// Return the fixed numeric range in the domain's canonical string order.
	members := make([]enginegroup.NativeMemberID, width)
	for local := range members {
		members[local] = enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", first+int32(local)))
	}
	slices.Sort(members)
	return members
}

func hasNativeMemberRange(members []enginegroup.NativeMemberIncarnation, first, width int32) bool {
	// Exact cardinality and every expected ID establish the complete rank range.
	if int32(len(members)) != width {
		return false
	}
	for offset := range width {
		id := enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", first+offset))
		if !slices.ContainsFunc(members, func(member enginegroup.NativeMemberIncarnation) bool { return member.ID == id }) {
			return false
		}
	}
	return true
}

func (a *MembershipAdapter) targetEPSize(target enginegroup.MembershipTarget) int32 {
	size := (target.BaseTopology.ReplicaCount() + int32(len(target.Joining))) * a.MembersPerReplica
	if target.Plan.Change.Kind == enginegroup.PlanKindRetire && target.Plan.Change.Retire != nil {
		size -= int32(len(target.Plan.Change.Retire.Replicas)) * a.MembersPerReplica
	}
	return size
}

func retiringMembership(base enginegroup.MembershipTopology, plan enginegroup.ResolvedPlan) []enginegroup.ReplicaMembership {
	if plan.Change.Kind != enginegroup.PlanKindRetire || plan.Change.Retire == nil {
		return nil
	}
	var retiring []enginegroup.ReplicaMembership
	for _, replica := range base.Replicas {
		if slices.Contains(plan.Change.Retire.Replicas, replica.ReplicaID) {
			retiring = append(retiring, enginegroup.ReplicaMembership{
				ReplicaID: replica.ReplicaID, Members: slices.Clone(replica.Members),
			})
		}
	}
	return retiring
}

// retiredMembers returns native terminal drain evidence for the exact frozen process set.
// A smaller effective size, including a successful commit, is insufficient by itself.
func (a *MembershipAdapter) retiredMembers(ctx context.Context, native ScaleState) ([]enginegroup.ReplicaMembership, error) {
	state := membershipJournal{}
	_, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return nil, err
	}
	if state.InstanceID != native.InstanceID {
		return nil, fmt.Errorf("%w: native retirement belongs to a different SGLang runtime", enginegroup.ErrRecoveryRequired)
	}
	record := state.Transition
	if record == nil || len(record.Retiring) == 0 || record.Phase != enginegroup.MembershipTransitionPhaseCommitted {
		return nil, nil
	}
	if native.OperationID != record.TransitionID || native.TargetEPSize != record.TargetEPSize ||
		native.EffectiveEPSize != record.TargetEPSize || native.Scaling || native.OperationSucceeded == nil || !*native.OperationSucceeded ||
		native.RetireeLifecycle != "external" || native.Phase != "serving_shrunk" {
		return nil, nil
	}
	for _, replica := range record.Retiring {
		for _, member := range replica.Members {
			if !slices.ContainsFunc(native.SafeToTerminateRanks, func(rank int32) bool {
				return member.ID == enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank))
			}) {
				return nil, nil
			}
		}
	}
	return record.Retiring, nil
}

func (a *MembershipAdapter) joiningAllocationIDs(ctx context.Context, groupID enginegroup.GroupID, target enginegroup.MembershipTarget) ([]string, error) {
	if target.Plan.Change.Kind == enginegroup.PlanKindRetire {
		return nil, nil
	}
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
			physical := enginegroup.ReplicaMembership{ReplicaID: joining.ReplicaID, Members: allocation.Incarnation.Members}
			if !enginegroup.SameMembershipSet([]enginegroup.ReplicaMembership{physical}, []enginegroup.ReplicaMembership{joining}) || len(allocation.Incarnation.CapacityRefs) != 1 {
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

func (a *MembershipAdapter) persistRejection(
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
		TargetEPSize:    a.targetEPSize(target),
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"fmt"
	"maps"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// ReplicaVector is one component-complete set of power-managed replica targets.
type ReplicaVector map[string]int32

// AdmissionDecision is an all-role commit or an unchanged vector with a bounded
// pending reason.
type AdmissionDecision struct {
	Accepted      bool
	Committed     ReplicaVector
	Ledger        Ledger
	PendingReason nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReason
}

// AdmissionState contains fail-closed baseline health that cannot be inferred
// from replica counts alone.
type AdmissionState struct {
	Phase             nvidiacomv1beta1.DynamoGraphPowerBudgetPhase
	TopologySupported bool
	HardwareQualified bool
}

// CompleteRequestedVector coalesces component updates against the current
// committed vector before atomic admission.
func CompleteRequestedVector(committed ReplicaVector, updates ReplicaVector) (ReplicaVector, error) {
	if len(committed) == 0 {
		return nil, fmt.Errorf("committed replica vector is empty")
	}

	complete := maps.Clone(committed)
	for component, replicas := range updates {
		if _, found := committed[component]; !found {
			return nil, fmt.Errorf("component %q is not in the committed vector", component)
		}
		complete[component] = replicas
	}
	return complete, nil
}

// AdmitVector atomically admits a component-complete vector against current and
// projected ledger snapshots.
func AdmitVector(
	spec Spec,
	committed ReplicaVector,
	requested ReplicaVector,
	currentProjection AdmissionProjection,
	requestedProjection AdmissionProjection,
	state AdmissionState,
) AdmissionDecision {
	decision := AdmissionDecision{
		Committed: maps.Clone(committed),
		Ledger:    currentProjection.Ledger(),
	}
	if !sameComponents(committed, requested) {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonInvalidTarget
		return decision
	}

	// InvalidTarget has deterministic priority over BelowMinimum.
	for _, replicas := range requested {
		if replicas < 0 {
			decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonInvalidTarget
			return decision
		}
	}
	for _, replicas := range requested {
		if replicas < spec.MinEndpoint() {
			decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBelowMinimum
			return decision
		}
	}
	if !maps.Equal(currentProjection.requested, committed) || !maps.Equal(requestedProjection.requested, requested) {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonInvalidTarget
		return decision
	}

	// Floor-respecting reductions never wait for stale evidence or a budget fit.
	if !vectorIncreases(committed, requested) {
		decision.Accepted = true
		decision.Committed = maps.Clone(requested)
		return decision
	}

	if !state.TopologySupported {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnsupportedTopology
		return decision
	}
	if !state.HardwareQualified || state.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnqualifiedHardware
		return decision
	}

	current := currentProjection.ledger
	projected := requestedProjection.ledger
	bootstrap := state.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing &&
		allTargetsZero(committed) &&
		current.TotalChargedWatts() == 0
	healthyBaseline := state.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle &&
		current.UnknownWatts() == 0 &&
		current.InGateReservedWatts() == 0 &&
		current.RolloutExtraWatts() == 0 &&
		(allTargetsZero(committed) || current.TotalChargedWatts() > 0) &&
		current.TotalChargedWatts() <= spec.BudgetWatts()
	if !bootstrap && !healthyBaseline {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline
		return decision
	}
	if projected.TotalChargedWatts() > spec.BudgetWatts() {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded
		return decision
	}
	decision.Accepted = true
	decision.Committed = maps.Clone(requested)
	decision.Ledger = requestedProjection.Ledger()
	return decision
}

func vectorIncreases(committed ReplicaVector, requested ReplicaVector) bool {
	for component, replicas := range requested {
		if replicas > committed[component] {
			return true
		}
	}
	return false
}

func allTargetsZero(vector ReplicaVector) bool {
	for _, replicas := range vector {
		if replicas != 0 {
			return false
		}
	}
	return true
}

func sameComponents(left ReplicaVector, right ReplicaVector) bool {
	if len(left) == 0 || len(left) != len(right) {
		return false
	}
	for component := range left {
		if _, found := right[component]; !found {
			return false
		}
	}
	return true
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"fmt"
	"math"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// RolloutReservationDecision is an aggregate rollout-extra reservation result.
type RolloutReservationDecision struct {
	Accepted      bool
	Ledger        Ledger
	PendingReason nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReason
}

// PreserveRolloutExtraFloor keeps a durable pre-child-write reservation while
// refreshing the other ledger classes from watched inventory.
func PreserveRolloutExtraFloor(
	status nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus,
	floorWatts int64,
) (nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus, error) {
	if floorWatts <= status.RolloutExtraWatts {
		return status, nil
	}
	ledger, err := newLedgerFromTotals(ledgerTotals{
		EnforcedWatts:       status.EnforcedWatts,
		UnknownWatts:        status.UnknownWatts,
		InGateReservedWatts: status.InGateReservedWatts,
		RolloutExtraWatts:   floorWatts,
	})
	if err != nil {
		return nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{}, err
	}
	return NewLedgerStatus(ledger), nil
}

// AdmitRolloutExtraReservation adds the conservative charge for newly required
// rollout capacity to the already classified rollout obligations. A proposal
// that adds no physical-GPU obligation remains safe to apply while the fence is
// otherwise closed.
func AdmitRolloutExtraReservation(
	spec Spec,
	current nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus,
	additionalRolloutExtraWatts int64,
	state AdmissionState,
) (RolloutReservationDecision, error) {
	ledger, err := newLedgerFromTotals(ledgerTotals{
		EnforcedWatts:       current.EnforcedWatts,
		UnknownWatts:        current.UnknownWatts,
		InGateReservedWatts: current.InGateReservedWatts,
		RolloutExtraWatts:   current.RolloutExtraWatts,
	})
	if err != nil {
		return RolloutReservationDecision{}, err
	}
	if ledger.TotalChargedWatts() != current.TotalChargedWatts {
		return RolloutReservationDecision{}, fmt.Errorf(
			"rollout ledger totalChargedWatts=%d, computed=%d",
			current.TotalChargedWatts,
			ledger.TotalChargedWatts(),
		)
	}
	if additionalRolloutExtraWatts < 0 {
		return RolloutReservationDecision{}, fmt.Errorf(
			"additional rollout-extra watts must be nonnegative, got %d",
			additionalRolloutExtraWatts,
		)
	}

	decision := RolloutReservationDecision{Ledger: ledger}
	if additionalRolloutExtraWatts == 0 {
		decision.Accepted = true
		return decision, nil
	}
	if !state.TopologySupported {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnsupportedTopology
		return decision, nil
	}
	if !state.HardwareQualified || state.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnqualifiedHardware
		return decision, nil
	}
	if state.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle &&
		state.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline
		return decision, nil
	}
	if ledger.UnknownWatts() != 0 || ledger.TotalChargedWatts() > spec.BudgetWatts() {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline
		return decision, nil
	}

	if current.RolloutExtraWatts > math.MaxInt64-additionalRolloutExtraWatts {
		return RolloutReservationDecision{}, fmt.Errorf("aggregate rollout-extra reservation overflows")
	}
	projected, err := newLedgerFromTotals(ledgerTotals{
		EnforcedWatts:       current.EnforcedWatts,
		UnknownWatts:        current.UnknownWatts,
		InGateReservedWatts: current.InGateReservedWatts,
		RolloutExtraWatts:   current.RolloutExtraWatts + additionalRolloutExtraWatts,
	})
	if err != nil {
		return RolloutReservationDecision{}, err
	}
	if projected.TotalChargedWatts() > spec.BudgetWatts() {
		decision.PendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded
		return decision, nil
	}
	decision.Accepted = true
	decision.Ledger = projected
	return decision, nil
}

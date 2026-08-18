/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"encoding/json"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	// MaxPowerManagedComponents is the supported per-DGD component limit.
	MaxPowerManagedComponents = nvidiacomv1beta1.DynamoGraphPowerBudgetMaxComponents
	// MaxEncodedStatusBytes bounds the operator-produced DGPB status payload.
	MaxEncodedStatusBytes = 64 * 1024
	// MaxStatusConditions bounds the currently defined aggregate condition set.
	MaxStatusConditions = 1
	// MaxStatusConditionMessageBytes bounds operator-authored diagnostic text.
	MaxStatusConditionMessageBytes = 1024
)

// NewLedgerStatus copies checked aggregate values into the durable API status.
func NewLedgerStatus(ledger Ledger) nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus {
	return nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts:       ledger.EnforcedWatts(),
		UnknownWatts:        ledger.UnknownWatts(),
		InGateReservedWatts: ledger.InGateReservedWatts(),
		RolloutExtraWatts:   ledger.RolloutExtraWatts(),
		TotalChargedWatts:   ledger.TotalChargedWatts(),
	}
}

// EncodeStatusSnapshot validates and serializes the durable v1beta1 DGPB status.
func EncodeStatusSnapshot(status nvidiacomv1beta1.DynamoGraphPowerBudgetStatus) ([]byte, error) {
	if len(status.Components) > MaxPowerManagedComponents {
		return nil, fmt.Errorf(
			"DGPB status has %d components, limit is %d",
			len(status.Components),
			MaxPowerManagedComponents,
		)
	}
	if len(status.CommittedReplicaTargets) > MaxPowerManagedComponents {
		return nil, fmt.Errorf(
			"DGPB status has %d committed targets, limit is %d",
			len(status.CommittedReplicaTargets),
			MaxPowerManagedComponents,
		)
	}
	if len(status.Conditions) > MaxStatusConditions {
		return nil, fmt.Errorf("DGPB status has %d conditions, limit is %d", len(status.Conditions), MaxStatusConditions)
	}

	seen := make(map[string]struct{}, len(status.Components))
	for _, component := range status.Components {
		if component.Name == "" {
			return nil, fmt.Errorf("DGPB status contains an empty component name")
		}
		if _, found := seen[component.Name]; found {
			return nil, fmt.Errorf("DGPB status contains duplicate component %q", component.Name)
		}
		seen[component.Name] = struct{}{}
	}
	for _, condition := range status.Conditions {
		if condition.Type != nvidiacomv1beta1.DynamoGraphPowerBudgetConditionTypePowerInfeasible {
			return nil, fmt.Errorf("DGPB status contains unknown condition type %q", condition.Type)
		}
		if len(condition.Message) > MaxStatusConditionMessageBytes {
			return nil, fmt.Errorf(
				"DGPB condition message has %d bytes, limit is %d",
				len(condition.Message),
				MaxStatusConditionMessageBytes,
			)
		}
	}

	ledger, err := newLedgerFromTotals(ledgerTotals{
		EnforcedWatts:       status.Ledger.EnforcedWatts,
		UnknownWatts:        status.Ledger.UnknownWatts,
		InGateReservedWatts: status.Ledger.InGateReservedWatts,
		RolloutExtraWatts:   status.Ledger.RolloutExtraWatts,
	})
	if err != nil {
		return nil, fmt.Errorf("invalid DGPB status ledger: %w", err)
	}
	if ledger.TotalChargedWatts() != status.Ledger.TotalChargedWatts {
		return nil, fmt.Errorf(
			"DGPB status totalChargedWatts=%d, computed=%d",
			status.Ledger.TotalChargedWatts,
			ledger.TotalChargedWatts(),
		)
	}

	encoded, err := json.Marshal(status)
	if err != nil {
		return nil, fmt.Errorf("encode DGPB status: %w", err)
	}
	if len(encoded) > MaxEncodedStatusBytes {
		return nil, fmt.Errorf("DGPB status size %d exceeds %d bytes", len(encoded), MaxEncodedStatusBytes)
	}
	return encoded, nil
}

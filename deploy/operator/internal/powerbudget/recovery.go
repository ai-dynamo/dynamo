/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"fmt"
	"maps"
	"sort"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

var recoveryScaleDownOrder = [...]nvidiacomv1beta1.ComponentType{
	nvidiacomv1beta1.ComponentTypeDecode,
	nvidiacomv1beta1.ComponentTypePrefill,
}

// RecoveryDecision describes one replica-only recovery step.
type RecoveryDecision struct {
	Phase            nvidiacomv1beta1.DynamoGraphPowerBudgetPhase
	Committed        ReplicaVector
	ReducedComponent string
	Fits             bool
	ConditionType    string
	RequiredWatts    int64
	AvailableWatts   int64
}

// NextRecoveryReduction computes at most one decode-first replica reduction.
func NextRecoveryReduction(
	committed ReplicaVector,
	componentRoles map[string]nvidiacomv1beta1.ComponentType,
	minEndpoint int32,
	requiredWatts int64,
	availableWatts int64,
) (RecoveryDecision, error) {
	if minEndpoint < 1 {
		return RecoveryDecision{}, fmt.Errorf("minEndpoint must be at least 1, got %d", minEndpoint)
	}
	if requiredWatts < 0 || availableWatts < 0 {
		return RecoveryDecision{}, fmt.Errorf(
			"recovery watts must be nonnegative, required=%d available=%d",
			requiredWatts,
			availableWatts,
		)
	}
	if len(committed) == 0 || len(componentRoles) != len(committed) {
		return RecoveryDecision{}, fmt.Errorf("recovery component-role map must match the committed vector")
	}
	for component := range committed {
		role, found := componentRoles[component]
		if !found || (role != nvidiacomv1beta1.ComponentTypeDecode && role != nvidiacomv1beta1.ComponentTypePrefill) {
			return RecoveryDecision{}, fmt.Errorf("component %q has unsupported recovery role %q", component, role)
		}
	}

	decision := RecoveryDecision{
		Phase:          nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		Committed:      maps.Clone(committed),
		RequiredWatts:  requiredWatts,
		AvailableWatts: availableWatts,
	}
	if requiredWatts <= availableWatts {
		decision.Fits = true
		return decision, nil
	}

	for _, role := range recoveryScaleDownOrder {
		components := componentsForRole(componentRoles, role)
		for _, component := range components {
			replicas := decision.Committed[component]
			if replicas > minEndpoint {
				decision.Committed[component] = replicas - 1
				decision.ReducedComponent = component
				return decision, nil
			}
		}
	}

	decision.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible
	decision.ConditionType = nvidiacomv1beta1.DynamoGraphPowerBudgetConditionTypePowerInfeasible
	return decision, nil
}

func componentsForRole(
	componentRoles map[string]nvidiacomv1beta1.ComponentType,
	role nvidiacomv1beta1.ComponentType,
) []string {
	components := make([]string, 0, len(componentRoles))
	for component, componentRole := range componentRoles {
		if componentRole == role {
			components = append(components, component)
		}
	}
	sort.Strings(components)
	return components
}

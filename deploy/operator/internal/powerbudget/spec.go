/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package powerbudget implements side-effect-free power accounting and
// replica-admission decisions for DynamoGraphPowerBudget controllers.
package powerbudget

import (
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// Spec is an immutable validated copy of DGPB policy inputs.
type Spec struct {
	budgetWatts int64
	minEndpoint int32
}

// NewSpec validates and copies immutable DGPB policy inputs.
func NewSpec(spec nvidiacomv1beta1.DynamoGraphPowerBudgetSpec) (Spec, error) {
	if spec.BudgetWatts < 1 {
		return Spec{}, fmt.Errorf("budgetWatts must be at least 1, got %d", spec.BudgetWatts)
	}
	if spec.Policy.MinEndpoint < 1 {
		return Spec{}, fmt.Errorf("minEndpoint must be at least 1, got %d", spec.Policy.MinEndpoint)
	}
	return Spec{
		budgetWatts: spec.BudgetWatts,
		minEndpoint: spec.Policy.MinEndpoint,
	}, nil
}

// BudgetWatts returns the aggregate physical-GPU budget.
func (s Spec) BudgetWatts() int64 {
	return s.budgetWatts
}

// MinEndpoint returns the per-component replica floor.
func (s Spec) MinEndpoint() int32 {
	return s.minEndpoint
}

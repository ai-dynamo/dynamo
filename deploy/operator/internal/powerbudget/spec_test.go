/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestDGPBSpecRequiresPositiveBudgetAndFloor(t *testing.T) {
	tests := []struct {
		name    string
		budget  int64
		floor   int32
		wantErr bool
	}{
		{name: "valid", budget: 5200, floor: 2},
		{name: "zero budget", budget: 0, floor: 2, wantErr: true},
		{name: "negative budget", budget: -1, floor: 2, wantErr: true},
		{name: "zero floor", budget: 5200, floor: 0, wantErr: true},
		{name: "negative floor", budget: 5200, floor: -1, wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build the API policy input")
			apiSpec := nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
				BudgetWatts: test.budget,
				Policy: nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{
					MinEndpoint: test.floor,
				},
			}

			t.Log("Validate and copy the immutable inputs")
			spec, err := NewSpec(apiSpec)
			if test.wantErr {
				if err == nil {
					t.Fatal("NewSpec() error = nil, want validation error")
				}
				return
			}
			if err != nil {
				t.Fatalf("NewSpec() error = %v", err)
			}

			t.Log("Expose the validated values without mutable fields")
			if got := spec.BudgetWatts(); got != test.budget {
				t.Errorf("BudgetWatts() = %d, want %d", got, test.budget)
			}
			if got := spec.MinEndpoint(); got != test.floor {
				t.Errorf("MinEndpoint() = %d, want %d", got, test.floor)
			}
		})
	}
}

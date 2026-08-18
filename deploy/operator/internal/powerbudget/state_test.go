/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"reflect"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestPhaseFence(t *testing.T) {
	t.Log("Open the replica fence only for the Idle phase")
	if ReplicaFenceClosed(nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle) {
		t.Fatal("Idle must open the replica fence")
	}

	t.Log("Close the replica fence for every non-Idle phase, including unknown values")
	closedPhases := []nvidiacomv1beta1.DynamoGraphPowerBudgetPhase{
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible,
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified,
		"future-phase",
	}
	for _, phase := range closedPhases {
		if !ReplicaFenceClosed(phase) {
			t.Errorf("phase %q unexpectedly opened the replica fence", phase)
		}
	}

	t.Log("Keep the fence derived instead of persisting a second status boolean")
	statusType := reflect.TypeFor[nvidiacomv1beta1.DynamoGraphPowerBudgetStatus]()
	if _, found := statusType.FieldByName("FenceClosed"); found {
		t.Fatal("DGPB status must not persist FenceClosed")
	}
}

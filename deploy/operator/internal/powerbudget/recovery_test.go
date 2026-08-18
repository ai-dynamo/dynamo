/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"maps"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestRecoveryNeverCrossesMinEndpoint(t *testing.T) {
	t.Log("Reduce exactly one decode replica before prefill")
	committed := ReplicaVector{"VllmPrefillWorker": 3, "VllmDecodeWorker": 3}
	roles := map[string]nvidiacomv1beta1.ComponentType{
		"VllmPrefillWorker": nvidiacomv1beta1.ComponentTypePrefill,
		"VllmDecodeWorker":  nvidiacomv1beta1.ComponentTypeDecode,
	}
	decision, err := NextRecoveryReduction(committed, roles, 2, 2100, 1400)
	if err != nil {
		t.Fatalf("NextRecoveryReduction() error = %v", err)
	}
	want := ReplicaVector{"VllmPrefillWorker": 3, "VllmDecodeWorker": 2}
	if decision.ReducedComponent != "VllmDecodeWorker" || !maps.Equal(decision.Committed, want) {
		t.Fatalf("first recovery decision = %#v, want %v", decision, want)
	}
	if !maps.Equal(committed, ReplicaVector{"VllmPrefillWorker": 3, "VllmDecodeWorker": 3}) {
		t.Fatalf("input committed vector was mutated: %v", committed)
	}

	t.Log("Reduce prefill only after decode reaches the immutable floor")
	decision, err = NextRecoveryReduction(want, roles, 2, 1750, 1400)
	if err != nil {
		t.Fatalf("NextRecoveryReduction() error = %v", err)
	}
	want = ReplicaVector{"VllmPrefillWorker": 2, "VllmDecodeWorker": 2}
	if decision.ReducedComponent != "VllmPrefillWorker" || !maps.Equal(decision.Committed, want) {
		t.Fatalf("second recovery decision = %#v, want %v", decision, want)
	}

	t.Log("Report PowerInfeasible instead of reducing any component below the floor")
	decision, err = NextRecoveryReduction(want, roles, 2, 1500, 1400)
	if err != nil {
		t.Fatalf("NextRecoveryReduction() error = %v", err)
	}
	if decision.ReducedComponent != "" || !maps.Equal(decision.Committed, want) {
		t.Fatalf("floor recovery decision changed replicas: %#v", decision)
	}
	if decision.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible {
		t.Fatalf("phase = %q, want Infeasible", decision.Phase)
	}
	if decision.ConditionType != nvidiacomv1beta1.DynamoGraphPowerBudgetConditionTypePowerInfeasible {
		t.Fatalf("condition = %q, want PowerInfeasible", decision.ConditionType)
	}
	if !ReplicaFenceClosed(decision.Phase) {
		t.Fatal("PowerInfeasible must keep the replica fence closed")
	}
	if decision.RequiredWatts != 1500 || decision.AvailableWatts != 1400 {
		t.Fatalf("infeasible watts = required %d available %d", decision.RequiredWatts, decision.AvailableWatts)
	}
}

func TestRecoveryScaleDownDecodeFirstOneAtATime(t *testing.T) {
	committed := ReplicaVector{"prefill": 4, "decode-b": 4, "decode-a": 4}
	roles := map[string]nvidiacomv1beta1.ComponentType{
		"prefill":  nvidiacomv1beta1.ComponentTypePrefill,
		"decode-b": nvidiacomv1beta1.ComponentTypeDecode,
		"decode-a": nvidiacomv1beta1.ComponentTypeDecode,
	}

	decision, err := NextRecoveryReduction(committed, roles, 2, 3000, 2000)
	if err != nil {
		t.Fatalf("NextRecoveryReduction() error = %v", err)
	}
	want := ReplicaVector{"prefill": 4, "decode-b": 4, "decode-a": 3}
	if decision.ReducedComponent != "decode-a" || !maps.Equal(decision.Committed, want) {
		t.Fatalf("recovery decision = %#v, want one deterministic decode reduction %v", decision, want)
	}
	if !maps.Equal(committed, ReplicaVector{"prefill": 4, "decode-b": 4, "decode-a": 4}) {
		t.Fatalf("input vector mutated: %v", committed)
	}
}

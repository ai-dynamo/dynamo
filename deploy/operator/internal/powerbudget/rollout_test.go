/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestAdmitRolloutExtraReservation(t *testing.T) {
	spec := mustSpec(t, 1000, 1)
	healthy := AdmissionState{
		Phase:             nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying,
		TopologySupported: true,
		HardwareQualified: true,
	}
	current := nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts:     600,
		TotalChargedWatts: 600,
	}

	t.Log("Reserve one safe surge GPU above the committed baseline")
	decision, err := AdmitRolloutExtraReservation(spec, current, 300, healthy)
	if err != nil || !decision.Accepted {
		t.Fatalf("safe rollout reservation = (%+v, %v)", decision, err)
	}
	if got := decision.Ledger.TotalChargedWatts(); got != 900 {
		t.Fatalf("safe rollout total = %d, want 900", got)
	}

	t.Log("Reject an over-budget surge without changing the current ledger")
	decision, err = AdmitRolloutExtraReservation(spec, current, 500, healthy)
	if err != nil {
		t.Fatalf("over-budget rollout reservation: %v", err)
	}
	if decision.Accepted || decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded {
		t.Fatalf("over-budget rollout decision = %+v", decision)
	}
	if got := decision.Ledger.TotalChargedWatts(); got != 600 {
		t.Fatalf("rejected rollout total = %d, want 600", got)
	}

	t.Log("Reuse an existing replacement obligation without increasing reserved watts")
	current.RolloutExtraWatts = 400
	current.TotalChargedWatts = 1000
	stale := healthy
	stale.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale
	decision, err = AdmitRolloutExtraReservation(spec, current, 0, stale)
	if err != nil || !decision.Accepted {
		t.Fatalf("replacement reservation = (%+v, %v)", decision, err)
	}
	if got := decision.Ledger.TotalChargedWatts(); got != 1000 {
		t.Fatalf("replacement total = %d, want 1000", got)
	}

	t.Log("An unknown-charged existing extra does not reserve a second physical GPU")
	spec = mustSpec(t, 1400, 1)
	current = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts:     350,
		RolloutExtraWatts: 700,
		TotalChargedWatts: 1050,
	}
	decision, err = AdmitRolloutExtraReservation(spec, current, 350, healthy)
	if err != nil || !decision.Accepted {
		t.Fatalf("incremental rollout reservation = (%+v, %v)", decision, err)
	}
	if got := decision.Ledger.RolloutExtraWatts(); got != 1050 {
		t.Fatalf("incremental rollout-extra watts = %d, want 1050", got)
	}
	if got := decision.Ledger.TotalChargedWatts(); got != 1400 {
		t.Fatalf("incremental rollout total = %d, want 1400", got)
	}
}

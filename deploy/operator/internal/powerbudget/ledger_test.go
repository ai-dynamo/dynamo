/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"math"
	"testing"
	"testing/quick"
	"time"
)

func TestLedgerChargeOnce(t *testing.T) {
	bounds := ComponentBounds{InGateWatts: 350, UnenforcedWatts: 700}
	now := time.Date(2026, time.August, 15, 12, 0, 0, 0, time.UTC)
	enforced := int64(325)
	healthy := testGPUReport(now.Add(-time.Minute), AgentPolicyOutcomeAnnotated, &enforced)
	safeDefault := testGPUReport(now.Add(-time.Minute), AgentPolicyOutcomeSafeDefaultConflict, &enforced)
	tests := []struct {
		name  string
		input GPUChargeInput
		want  GPUCharge
	}{
		{
			name: "fresh annotated readback uses exact enforced cap",
			input: GPUChargeInput{
				Bounds:                bounds,
				Report:                &healthy,
				ReportExisted:         true,
				ReportIdentityMatches: true,
				ExpectedGPUUUID:       healthy.UUID,
				Now:                   now,
				FreshnessLimit:        5 * time.Minute,
			},
			want: GPUCharge{class: ChargeClassEnforced, watts: 325},
		},
		{
			name: "safe-default readback remains unknown at U_c",
			input: GPUChargeInput{
				Bounds:                bounds,
				Report:                &safeDefault,
				ReportExisted:         true,
				ReportIdentityMatches: true,
				ExpectedGPUUUID:       safeDefault.UUID,
				Now:                   now,
				FreshnessLimit:        5 * time.Minute,
			},
			want: GPUCharge{class: ChargeClassUnknown, watts: 700},
		},
		{
			name: "never-reported injected slot uses B_c",
			input: GPUChargeInput{
				Bounds:       bounds,
				InjectedGate: true,
			},
			want: GPUCharge{class: ChargeClassInGate, watts: 350},
		},
		{
			name: "rollout extra retains its conservative underlying charge",
			input: GPUChargeInput{
				Bounds:                bounds,
				Report:                &healthy,
				ReportExisted:         true,
				ReportIdentityMatches: true,
				ExpectedGPUUUID:       healthy.UUID,
				Now:                   now,
				FreshnessLimit:        5 * time.Minute,
				RolloutExtra:          true,
			},
			want: GPUCharge{class: ChargeClassRolloutExtra, watts: 325},
		},
	}

	var ledger Ledger
	for index, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			charge, err := ClassifyGPUCharge(test.input)
			if err != nil {
				t.Fatalf("ClassifyGPUCharge() error = %v", err)
			}
			if charge != test.want {
				t.Fatalf("ClassifyGPUCharge() = %#v, want %#v", charge, test.want)
			}
			if err := ledger.AddCharge(test.name, charge); err != nil {
				t.Fatalf("AddCharge() error = %v", err)
			}
			if index == 0 {
				if err := ledger.AddCharge(test.name, charge); err == nil {
					t.Fatal("AddCharge() accepted a duplicate physical-GPU identity")
				}
			}
		})
	}

	t.Log("Sum each unique charge through exactly one aggregate class")
	if ledger.EnforcedWatts() != 325 || ledger.UnknownWatts() != 700 || ledger.InGateReservedWatts() != 350 || ledger.RolloutExtraWatts() != 325 {
		t.Fatalf("ledger totals = enforced %d unknown %d inGate %d rollout %d",
			ledger.EnforcedWatts(), ledger.UnknownWatts(), ledger.InGateReservedWatts(), ledger.RolloutExtraWatts())
	}
	if got, want := ledger.TotalChargedWatts(), int64(1700); got != want {
		t.Fatalf("TotalChargedWatts() = %d, want %d", got, want)
	}

	t.Log("Reject negative and overflowing totals without mutating the ledger")
	if _, err := newLedgerFromTotals(ledgerTotals{UnknownWatts: -1}); err == nil {
		t.Fatal("newLedgerFromTotals() accepted negative watts")
	}
	overflow, err := newLedgerFromTotals(ledgerTotals{EnforcedWatts: math.MaxInt64})
	if err != nil {
		t.Fatalf("newLedgerFromTotals() error = %v", err)
	}
	if err := overflow.AddCharge("overflow", GPUCharge{class: ChargeClassUnknown, watts: 1}); err == nil {
		t.Fatal("AddCharge() accepted an overflowing watt total")
	}
	if overflow.TotalChargedWatts() != math.MaxInt64 || overflow.UnknownWatts() != 0 {
		t.Fatal("overflowing AddCharge() mutated the ledger")
	}
}

func TestStaleNeverReturnsToInGate(t *testing.T) {
	bounds := ComponentBounds{InGateWatts: 350, UnenforcedWatts: 700}
	now := time.Date(2026, time.August, 15, 12, 0, 0, 0, time.UTC)
	enforced := int64(350)
	stale := testGPUReport(now.Add(-10*time.Minute), AgentPolicyOutcomeAnnotated, &enforced)

	t.Log("Derive stale evidence from observedAt instead of a caller freshness flag")
	charge, err := ClassifyGPUCharge(GPUChargeInput{
		Bounds:                bounds,
		Report:                &stale,
		ReportExisted:         true,
		ReportIdentityMatches: true,
		ExpectedGPUUUID:       stale.UUID,
		Now:                   now,
		FreshnessLimit:        5 * time.Minute,
		InjectedGate:          true,
	})
	if err != nil {
		t.Fatalf("ClassifyGPUCharge() error = %v", err)
	}
	if charge.Class() != ChargeClassUnknown || charge.Watts() != 700 {
		t.Fatalf("stale charge = %#v, want unknown at U_c", charge)
	}

	t.Log("Keep missing evidence unknown after any report has existed")
	charge, err = ClassifyGPUCharge(GPUChargeInput{
		Bounds:        bounds,
		ReportExisted: true,
		InjectedGate:  true,
	})
	if err != nil {
		t.Fatalf("ClassifyGPUCharge() error = %v", err)
	}
	if charge.Class() != ChargeClassUnknown || charge.Watts() != 700 {
		t.Fatalf("missing historical charge = %#v, want unknown at U_c", charge)
	}
}

func TestLedgerChargeProperties(t *testing.T) {
	t.Log("Check arbitrary bounded class charges sum without omission or double count")
	property := func(enforced uint16, unknown uint16, inGate uint16, rollout uint16) bool {
		watts := [...]int64{int64(enforced) + 1, int64(unknown) + 1, int64(inGate) + 1, int64(rollout) + 1}
		classes := [...]ChargeClass{ChargeClassEnforced, ChargeClassUnknown, ChargeClassInGate, ChargeClassRolloutExtra}
		var ledger Ledger
		for index, class := range classes {
			if err := ledger.AddCharge(string(rune('a'+index)), GPUCharge{class: class, watts: watts[index]}); err != nil {
				return false
			}
		}
		want := watts[0] + watts[1] + watts[2] + watts[3]
		return ledger.TotalChargedWatts() == want &&
			ledger.EnforcedWatts() == watts[0] &&
			ledger.UnknownWatts() == watts[1] &&
			ledger.InGateReservedWatts() == watts[2] &&
			ledger.RolloutExtraWatts() == watts[3]
	}
	if err := quick.Check(property, &quick.Config{MaxCount: 1000}); err != nil {
		t.Fatalf("ledger charge property failed: %v", err)
	}
}

func testGPUReport(observedAt time.Time, policy AgentPolicyOutcome, enforced *int64) AgentGPUReport {
	return AgentGPUReport{
		UUID:               "GPU-1",
		RequestedWatts:     350,
		TargetWatts:        350,
		ConstraintMinWatts: 300,
		ConstraintMaxWatts: 700,
		PolicyOutcome:      policy,
		WriteOutcome:       AgentWriteOutcomeSucceeded,
		ReadbackOutcome:    AgentReadbackOutcomeSucceeded,
		EnforcedCapWatts:   enforced,
		Actuator:           AgentActuatorNVML,
		ObservedAt:         observedAt,
	}
}

func mustLedger(t *testing.T, totals ledgerTotals) Ledger {
	t.Helper()
	ledger, err := newLedgerFromTotals(totals)
	if err != nil {
		t.Fatalf("newLedgerFromTotals() error = %v", err)
	}
	return ledger
}

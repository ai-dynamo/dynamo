/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
)

func TestReportMismatch(t *testing.T) {
	t.Log("Build one healthy two-GPU synthetic report and its watched identity")
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	expected := testPodPowerExpectation(now)
	base := testPodPowerReport(now)

	tests := []struct {
		name   string
		mutate func(*powerbudget.AgentReport)
	}{
		{name: "version", mutate: func(report *powerbudget.AgentReport) { report.Version++ }},
		{name: "DGD UID", mutate: func(report *powerbudget.AgentReport) { report.DGDUID = "other" }},
		{name: "component", mutate: func(report *powerbudget.AgentReport) { report.Component = "other" }},
		{name: "Pod UID", mutate: func(report *powerbudget.AgentReport) { report.PodUID = "other" }},
		{name: "node", mutate: func(report *powerbudget.AgentReport) { report.Node = "other" }},
		{name: "GPU count", mutate: func(report *powerbudget.AgentReport) { report.GPUs = report.GPUs[:1] }},
		{name: "allocation", mutate: func(report *powerbudget.AgentReport) { report.AllocationID = "other" }},
		{name: "requested cap", mutate: func(report *powerbudget.AgentReport) { report.GPUs[0].RequestedWatts = 360 }},
		{name: "clamped target", mutate: func(report *powerbudget.AgentReport) { report.GPUs[0].TargetWatts = 400 }},
		{name: "policy", mutate: func(report *powerbudget.AgentReport) {
			report.GPUs[0].PolicyOutcome = powerbudget.AgentPolicyOutcomeSafeDefaultConflict
		}},
		{name: "bound", mutate: func(report *powerbudget.AgentReport) {
			enforced := int64(450)
			report.GPUs[0].EnforcedCapWatts = &enforced
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Mutate one required report field and require Pod-atomic unknown evidence")
			report := cloneAgentReport(base)
			tc.mutate(&report)
			var encoded string
			if tc.name == "version" {
				// Unsupported versions are intentionally encoded outside the
				// producer helper so the consumer rejection path is exercised.
				encoded = `{"version":2}`
			} else {
				encoded = encodeSyntheticReport(t, report)
			}
			evidence, err := evaluatePodPowerReport(encoded, expected)
			if err != nil {
				t.Fatalf("evaluatePodPowerReport() error = %v", err)
			}
			if evidence.Accepted {
				t.Fatalf("evaluatePodPowerReport() accepted %s mismatch", tc.name)
			}
			assertAtomicUnknownEvidence(t, evidence, 2, 1400)
		})
	}
}

func TestMultiGPUAtomicEvidence(t *testing.T) {
	t.Log("Accept a report only when every assigned GPU has healthy exact readback")
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	expected := testPodPowerExpectation(now)
	report := testPodPowerReport(now)

	evidence, err := evaluatePodPowerReport(encodeSyntheticReport(t, report), expected)
	if err != nil {
		t.Fatalf("evaluatePodPowerReport() healthy error = %v", err)
	}
	if !evidence.Accepted {
		t.Fatalf("healthy multi-GPU evidence rejected: %s", evidence.Reason)
	}
	assertChargeTotals(t, evidence, powerbudget.ChargeClassEnforced, 2, 700)

	t.Log("Reject the entire allocation after one GPU readback fails")
	report = cloneAgentReport(report)
	report.GPUs[1].ReadbackOutcome = powerbudget.AgentReadbackOutcomeFailed
	report.GPUs[1].EnforcedCapWatts = nil
	evidence, err = evaluatePodPowerReport(encodeSyntheticReport(t, report), expected)
	if err != nil {
		t.Fatalf("evaluatePodPowerReport() partial failure error = %v", err)
	}
	if evidence.Accepted {
		t.Fatal("partially healthy multi-GPU report was accepted")
	}
	assertAtomicUnknownEvidence(t, evidence, 2, 1400)

	t.Log("Return the entire allocation to enforced after a fresh repaired report")
	repaired := testPodPowerReport(now.Add(10 * time.Second))
	expected.Now = now.Add(10 * time.Second)
	evidence, err = evaluatePodPowerReport(encodeSyntheticReport(t, repaired), expected)
	if err != nil {
		t.Fatalf("evaluatePodPowerReport() repaired error = %v", err)
	}
	if !evidence.Accepted {
		t.Fatalf("repaired multi-GPU evidence rejected: %s", evidence.Reason)
	}
	assertChargeTotals(t, evidence, powerbudget.ChargeClassEnforced, 2, 700)
}

func TestAssignedMissingReportUsesBoundedHistory(t *testing.T) {
	t.Log("Classify a never-reported assigned Pod at the injected in-gate bound B_c")
	expected := testPodPowerExpectation(time.Now())
	expected.ReportExisted = false
	evidence, err := evaluatePodPowerReport("", expected)
	if err != nil {
		t.Fatalf("evaluatePodPowerReport() missing error = %v", err)
	}
	if evidence.Accepted {
		t.Fatal("missing assigned report was accepted")
	}
	assertChargeTotals(t, evidence, powerbudget.ChargeClassInGate, 2, 800)

	t.Log("Keep missing evidence at U_c after durable history records a report")
	expected.ReportExisted = true
	evidence, err = evaluatePodPowerReport("", expected)
	if err != nil {
		t.Fatalf("evaluatePodPowerReport() historical missing error = %v", err)
	}
	assertAtomicUnknownEvidence(t, evidence, 2, 1400)
}

func testPodPowerExpectation(now time.Time) podPowerReportExpectation {
	uuids := []string{"GPU-a", "GPU-b"}
	return podPowerReportExpectation{
		DGDUID:                    "dgd-uid",
		Component:                 "decode",
		PodUID:                    "pod-uid",
		Node:                      "node-a",
		AllocationID:              expectedPowerAllocationID("pod-uid", "main", uuids),
		ExpectedGPUCount:          len(uuids),
		ExpectedGPUUUIDs:          uuids,
		ExpectedRequestedCapWatts: 350,
		ReportExisted:             true,
		Bounds:                    powerbudget.ComponentBounds{InGateWatts: 400, UnenforcedWatts: 700},
		Now:                       now,
		FreshnessLimit:            time.Minute,
		StructurallyInGated:       true,
	}
}

func testPodPowerReport(observedAt time.Time) powerbudget.AgentReport {
	enforcedA := int64(350)
	enforcedB := int64(350)
	uuids := []string{"GPU-a", "GPU-b"}
	return powerbudget.AgentReport{
		Version:      powerbudget.AgentReportDocumentVersion,
		DGDUID:       "dgd-uid",
		Component:    "decode",
		PodUID:       "pod-uid",
		Node:         "node-a",
		AllocationID: expectedPowerAllocationID("pod-uid", "main", uuids),
		GPUs: []powerbudget.AgentGPUReport{
			{
				UUID:               uuids[0],
				RequestedWatts:     350,
				TargetWatts:        350,
				ConstraintMinWatts: 300,
				ConstraintMaxWatts: 700,
				PolicyOutcome:      powerbudget.AgentPolicyOutcomeAnnotated,
				WriteOutcome:       powerbudget.AgentWriteOutcomeSucceeded,
				ReadbackOutcome:    powerbudget.AgentReadbackOutcomeSucceeded,
				EnforcedCapWatts:   &enforcedA,
				Actuator:           powerbudget.AgentActuatorNVML,
				ObservedAt:         observedAt,
			},
			{
				UUID:               uuids[1],
				RequestedWatts:     350,
				TargetWatts:        350,
				ConstraintMinWatts: 300,
				ConstraintMaxWatts: 700,
				PolicyOutcome:      powerbudget.AgentPolicyOutcomeAnnotated,
				WriteOutcome:       powerbudget.AgentWriteOutcomeSucceeded,
				ReadbackOutcome:    powerbudget.AgentReadbackOutcomeSucceeded,
				EnforcedCapWatts:   &enforcedB,
				Actuator:           powerbudget.AgentActuatorNVML,
				ObservedAt:         observedAt,
			},
		},
	}
}

func cloneAgentReport(report powerbudget.AgentReport) powerbudget.AgentReport {
	cloned := report
	cloned.GPUs = append([]powerbudget.AgentGPUReport(nil), report.GPUs...)
	for i := range cloned.GPUs {
		if report.GPUs[i].EnforcedCapWatts != nil {
			value := *report.GPUs[i].EnforcedCapWatts
			cloned.GPUs[i].EnforcedCapWatts = &value
		}
	}
	return cloned
}

func encodeSyntheticReport(t *testing.T, report powerbudget.AgentReport) string {
	t.Helper()
	encoded, err := powerbudget.EncodeAgentReport(report)
	if err != nil {
		t.Fatalf("EncodeAgentReport() error = %v", err)
	}
	return string(encoded)
}

func assertAtomicUnknownEvidence(t *testing.T, evidence podPowerEvidence, wantCount int, wantWatts int64) {
	t.Helper()
	assertChargeTotals(t, evidence, powerbudget.ChargeClassUnknown, wantCount, wantWatts)
	for _, charge := range evidence.Charges {
		if charge.Class() == powerbudget.ChargeClassEnforced {
			t.Fatal("atomic rejection retained an enforced GPU charge")
		}
	}
}

func assertChargeTotals(
	t *testing.T,
	evidence podPowerEvidence,
	wantClass powerbudget.ChargeClass,
	wantCount int,
	wantWatts int64,
) {
	t.Helper()
	count := 0
	var watts int64
	for _, charge := range evidence.Charges {
		if charge.Class() == wantClass {
			count++
			watts += charge.Watts()
		}
	}
	if count != wantCount || watts != wantWatts {
		t.Fatalf("%s charges = (%d, %dW), want (%d, %dW); evidence=%+v", wantClass, count, watts, wantCount, wantWatts, evidence)
	}
}

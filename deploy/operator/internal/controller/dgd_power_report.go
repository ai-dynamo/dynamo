/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
)

type podPowerReportExpectation struct {
	DGDUID                    string
	Component                 string
	PodUID                    string
	Node                      string
	AllocationID              string
	ExpectedGPUCount          int
	ExpectedGPUUUIDs          []string
	ExpectedRequestedCapWatts int64
	ReportExisted             bool
	Bounds                    powerbudget.ComponentBounds
	Now                       time.Time
	FreshnessLimit            time.Duration
	StructurallyInGated       bool
	RolloutExtra              bool
}

type podPowerEvidence struct {
	Charges  []powerbudget.GPUCharge
	Accepted bool
	Reason   string
}

func evaluatePodPowerReport(encoded string, expected podPowerReportExpectation) (podPowerEvidence, error) {
	if expected.ExpectedGPUCount < 1 {
		return podPowerEvidence{}, fmt.Errorf("expected GPU count must be positive")
	}
	if expected.ExpectedRequestedCapWatts < 1 {
		return podPowerEvidence{}, fmt.Errorf("expected requested cap must be positive")
	}
	if expected.Bounds.InGateWatts < 1 || expected.Bounds.UnenforcedWatts < expected.Bounds.InGateWatts {
		return podPowerEvidence{}, fmt.Errorf("invalid expected component bounds")
	}
	if encoded == "" {
		charges, err := fallbackPodPowerCharges(nil, expected, expected.ReportExisted)
		return podPowerEvidence{Charges: charges, Reason: "report missing"}, err
	}

	report, err := powerbudget.DecodeAgentReport([]byte(encoded))
	if err != nil {
		charges, chargeErr := fallbackPodPowerCharges(nil, expected, true)
		if chargeErr != nil {
			return podPowerEvidence{}, chargeErr
		}
		return podPowerEvidence{Charges: charges, Reason: err.Error()}, nil
	}
	if report.DGDUID != expected.DGDUID ||
		report.Component != expected.Component ||
		report.PodUID != expected.PodUID ||
		report.Node != expected.Node ||
		report.AllocationID != expected.AllocationID {
		charges, chargeErr := fallbackPodPowerCharges(report.GPUs, expected, true)
		if chargeErr != nil {
			return podPowerEvidence{}, chargeErr
		}
		return podPowerEvidence{Charges: charges, Reason: "report identity mismatch"}, nil
	}
	if len(report.GPUs) != expected.ExpectedGPUCount {
		charges, chargeErr := fallbackPodPowerCharges(report.GPUs, expected, true)
		if chargeErr != nil {
			return podPowerEvidence{}, chargeErr
		}
		return podPowerEvidence{Charges: charges, Reason: "report GPU count mismatch"}, nil
	}
	if len(expected.ExpectedGPUUUIDs) != 0 {
		actual := make([]string, 0, len(report.GPUs))
		for _, gpu := range report.GPUs {
			actual = append(actual, gpu.UUID)
		}
		want := append([]string(nil), expected.ExpectedGPUUUIDs...)
		sort.Strings(actual)
		sort.Strings(want)
		if !slices.Equal(actual, want) {
			charges, chargeErr := fallbackPodPowerCharges(report.GPUs, expected, true)
			if chargeErr != nil {
				return podPowerEvidence{}, chargeErr
			}
			return podPowerEvidence{Charges: charges, Reason: "report GPU allocation mismatch"}, nil
		}
	}
	for i := range report.GPUs {
		gpu := &report.GPUs[i]
		expectedTarget := min(max(expected.ExpectedRequestedCapWatts, gpu.ConstraintMinWatts), gpu.ConstraintMaxWatts)
		if gpu.RequestedWatts != expected.ExpectedRequestedCapWatts || gpu.TargetWatts != expectedTarget {
			charges, chargeErr := fallbackPodPowerCharges(report.GPUs, expected, true)
			if chargeErr != nil {
				return podPowerEvidence{}, chargeErr
			}
			return podPowerEvidence{Charges: charges, Reason: "report power intent mismatch"}, nil
		}
	}

	charges := make([]powerbudget.GPUCharge, 0, len(report.GPUs))
	allEnforced := true
	for i := range report.GPUs {
		gpu := &report.GPUs[i]
		charge, err := powerbudget.ClassifyGPUCharge(powerbudget.GPUChargeInput{
			Bounds:                expected.Bounds,
			Report:                gpu,
			ReportExisted:         true,
			ReportIdentityMatches: true,
			ExpectedGPUUUID:       gpu.UUID,
			Now:                   expected.Now,
			FreshnessLimit:        expected.FreshnessLimit,
			InjectedGate:          expected.StructurallyInGated,
			RolloutExtra:          expected.RolloutExtra,
		})
		if err != nil {
			return podPowerEvidence{}, err
		}
		charges = append(charges, charge)
		if charge.Class() != powerbudget.ChargeClassEnforced {
			allEnforced = false
		}
	}
	if allEnforced {
		return podPowerEvidence{Charges: charges, Accepted: true}, nil
	}

	// Evidence is Pod-atomic: one unhealthy assigned GPU makes every GPU in the
	// allocation unknown until one report proves the entire set healthy.
	charges, err = fallbackPodPowerCharges(report.GPUs, expected, true)
	if err != nil {
		return podPowerEvidence{}, err
	}
	return podPowerEvidence{Charges: charges, Reason: "not every assigned GPU has healthy evidence"}, nil
}

func fallbackPodPowerCharges(
	reports []powerbudget.AgentGPUReport,
	expected podPowerReportExpectation,
	reportExisted bool,
) ([]powerbudget.GPUCharge, error) {
	count := max(expected.ExpectedGPUCount, len(reports))
	charges := make([]powerbudget.GPUCharge, 0, count)
	for i := 0; i < count; i++ {
		var report *powerbudget.AgentGPUReport
		expectedUUID := fmt.Sprintf("unobserved-%d", i)
		if i < len(reports) {
			report = &reports[i]
			expectedUUID = report.UUID
		}
		charge, err := powerbudget.ClassifyGPUCharge(powerbudget.GPUChargeInput{
			Bounds:                expected.Bounds,
			Report:                report,
			ReportExisted:         reportExisted,
			ReportIdentityMatches: false,
			ExpectedGPUUUID:       expectedUUID,
			Now:                   expected.Now,
			FreshnessLimit:        expected.FreshnessLimit,
			InjectedGate:          expected.StructurallyInGated,
			RolloutExtra:          expected.RolloutExtra,
		})
		if err != nil {
			return nil, err
		}
		charges = append(charges, charge)
	}
	return charges, nil
}

func expectedPowerAllocationID(podUID, containerName string, gpuUUIDs []string) string {
	uuids := append([]string(nil), gpuUUIDs...)
	sort.Strings(uuids)
	return podUID + "/" + containerName + "/" + strings.Join(uuids, ",")
}

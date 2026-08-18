/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"fmt"
	"math"
	"time"
)

// ChargeClass identifies one disjoint aggregate ledger class.
type ChargeClass string

const (
	// ChargeClassEnforced contains fresh exact annotated-cap readbacks.
	ChargeClassEnforced ChargeClass = "enforced"
	// ChargeClassUnknown contains stale, failed, mismatched, malformed, or
	// safe-default evidence.
	ChargeClassUnknown ChargeClass = "unknown"
	// ChargeClassInGate contains never-reported slots protected by the injected gate.
	ChargeClassInGate ChargeClass = "inGate"
	// ChargeClassRolloutExtra contains old, new, surge, or terminating capacity
	// outside the committed replica slots.
	ChargeClassRolloutExtra ChargeClass = "rolloutExtra"
)

// GPUChargeInput is the watched evidence used to classify one physical GPU or
// one committed physical-GPU slot. Freshness and result health are derived from
// Report rather than asserted by callers.
type GPUChargeInput struct {
	Bounds                ComponentBounds
	Report                *AgentGPUReport
	ReportExisted         bool
	ReportIdentityMatches bool
	ExpectedGPUUUID       string
	Now                   time.Time
	FreshnessLimit        time.Duration
	InjectedGate          bool
	RolloutExtra          bool
}

// GPUCharge is one conservative charge assigned to exactly one ledger class.
type GPUCharge struct {
	class ChargeClass
	watts int64
}

// ClassifyGPUCharge selects one conservative charge from watched evidence.
func ClassifyGPUCharge(input GPUChargeInput) (GPUCharge, error) {
	if input.Bounds.InGateWatts < 1 || input.Bounds.UnenforcedWatts < input.Bounds.InGateWatts {
		return GPUCharge{}, fmt.Errorf(
			"invalid component bounds B_c=%d U_c=%d",
			input.Bounds.InGateWatts,
			input.Bounds.UnenforcedWatts,
		)
	}

	class, watts := baseGPUCharge(input)
	if input.RolloutExtra {
		class = ChargeClassRolloutExtra
	}
	return GPUCharge{class: class, watts: watts}, nil
}

// Class returns the disjoint aggregate class selected for this charge.
func (charge GPUCharge) Class() ChargeClass {
	return charge.class
}

// Watts returns the conservative watts selected for this charge.
func (charge GPUCharge) Watts() int64 {
	return charge.watts
}

// baseGPUCharge derives exact, unknown, or in-gate state before rollout
// capacity is moved into its own aggregate class.
func baseGPUCharge(input GPUChargeInput) (ChargeClass, int64) {
	if input.Report == nil {
		if !input.ReportExisted && input.InjectedGate {
			return ChargeClassInGate, input.Bounds.InGateWatts
		}
		return ChargeClassUnknown, input.Bounds.UnenforcedWatts
	}

	report := input.Report
	unknownWatts := max(input.Bounds.UnenforcedWatts, report.ConstraintMaxWatts)
	reportAge := input.Now.Sub(report.ObservedAt)
	reportValid := validateAgentGPUReport(*report) == nil
	fresh := !input.Now.IsZero() &&
		input.FreshnessLimit > 0 &&
		reportAge >= 0 &&
		reportAge <= input.FreshnessLimit
	healthy := reportValid &&
		input.ReportIdentityMatches &&
		input.ExpectedGPUUUID != "" &&
		report.UUID == input.ExpectedGPUUUID &&
		report.PolicyOutcome == AgentPolicyOutcomeAnnotated &&
		report.WriteOutcome == AgentWriteOutcomeSucceeded &&
		report.ReadbackOutcome == AgentReadbackOutcomeSucceeded &&
		report.EnforcedCapWatts != nil &&
		*report.EnforcedCapWatts > 0 &&
		*report.EnforcedCapWatts <= input.Bounds.InGateWatts &&
		liveRangeMatchesQualification(input.Bounds, report)
	if fresh && healthy {
		return ChargeClassEnforced, *report.EnforcedCapWatts
	}
	return ChargeClassUnknown, unknownWatts
}

func liveRangeMatchesQualification(bounds ComponentBounds, report *AgentGPUReport) bool {
	if bounds.QualifiedMinWatts == 0 && bounds.QualifiedMaxWatts == 0 {
		// Non-production ledger unit fixtures may exercise the generic charge
		// classifier without a product qualification. The DGD inventory always
		// supplies both exact bounds.
		return report.ConstraintMaxWatts <= bounds.UnenforcedWatts
	}
	return report.ConstraintMinWatts == bounds.QualifiedMinWatts &&
		report.ConstraintMaxWatts == bounds.QualifiedMaxWatts
}

// ledgerTotals is an internal aggregate-ledger reconstruction input.
type ledgerTotals struct {
	EnforcedWatts       int64
	UnknownWatts        int64
	InGateReservedWatts int64
	RolloutExtraWatts   int64
}

// Ledger contains checked aggregate watts and in-memory charge identities.
type Ledger struct {
	enforcedWatts       int64
	unknownWatts        int64
	inGateReservedWatts int64
	rolloutExtraWatts   int64
	totalChargedWatts   int64
	charged             map[string]struct{}
}

// newLedgerFromTotals validates a precomputed aggregate ledger snapshot.
func newLedgerFromTotals(totals ledgerTotals) (Ledger, error) {
	values := [...]int64{
		totals.EnforcedWatts,
		totals.UnknownWatts,
		totals.InGateReservedWatts,
		totals.RolloutExtraWatts,
	}
	var total int64
	for _, value := range values {
		if value < 0 {
			return Ledger{}, fmt.Errorf("ledger totals must be nonnegative: %+v", totals)
		}
		var err error
		total, err = checkedAddWatts(total, value)
		if err != nil {
			return Ledger{}, err
		}
	}
	return Ledger{
		enforcedWatts:       totals.EnforcedWatts,
		unknownWatts:        totals.UnknownWatts,
		inGateReservedWatts: totals.InGateReservedWatts,
		rolloutExtraWatts:   totals.RolloutExtraWatts,
		totalChargedWatts:   total,
	}, nil
}

// AddCharge adds one unique physical-GPU or reservation charge exactly once.
func (ledger *Ledger) AddCharge(chargeID string, charge GPUCharge) error {
	if chargeID == "" {
		return fmt.Errorf("charge identity is empty")
	}
	if charge.watts < 1 {
		return fmt.Errorf("charge %q must be at least 1 watt, got %d", chargeID, charge.watts)
	}
	if ledger.charged == nil {
		ledger.charged = make(map[string]struct{})
	}
	if _, found := ledger.charged[chargeID]; found {
		return fmt.Errorf("charge identity %q was already accounted", chargeID)
	}

	newTotal, err := checkedAddWatts(ledger.totalChargedWatts, charge.watts)
	if err != nil {
		return fmt.Errorf("charge %q: %w", chargeID, err)
	}
	switch charge.class {
	case ChargeClassEnforced:
		ledger.enforcedWatts, err = checkedAddWatts(ledger.enforcedWatts, charge.watts)
	case ChargeClassUnknown:
		ledger.unknownWatts, err = checkedAddWatts(ledger.unknownWatts, charge.watts)
	case ChargeClassInGate:
		ledger.inGateReservedWatts, err = checkedAddWatts(ledger.inGateReservedWatts, charge.watts)
	case ChargeClassRolloutExtra:
		ledger.rolloutExtraWatts, err = checkedAddWatts(ledger.rolloutExtraWatts, charge.watts)
	default:
		return fmt.Errorf("charge %q has unknown class %q", chargeID, charge.class)
	}
	if err != nil {
		return fmt.Errorf("charge %q: %w", chargeID, err)
	}

	ledger.totalChargedWatts = newTotal
	ledger.charged[chargeID] = struct{}{}
	return nil
}

// AddAggregateCharge adds a checked number of identical charges without
// materializing one identity per slot. Callers must derive count from a bounded,
// component-complete capacity calculation rather than external evidence IDs.
func (ledger *Ledger) AddAggregateCharge(count int64, charge GPUCharge) error {
	if count < 0 {
		return fmt.Errorf("aggregate charge count must be nonnegative, got %d", count)
	}
	if count == 0 {
		return nil
	}
	if charge.watts < 1 {
		return fmt.Errorf("aggregate charge must be at least 1 watt, got %d", charge.watts)
	}
	if count > math.MaxInt64/charge.watts {
		return fmt.Errorf("aggregate charge watt total overflows int64")
	}
	watts := count * charge.watts
	newTotal, err := checkedAddWatts(ledger.totalChargedWatts, watts)
	if err != nil {
		return fmt.Errorf("aggregate charge: %w", err)
	}
	switch charge.class {
	case ChargeClassEnforced:
		ledger.enforcedWatts, err = checkedAddWatts(ledger.enforcedWatts, watts)
	case ChargeClassUnknown:
		ledger.unknownWatts, err = checkedAddWatts(ledger.unknownWatts, watts)
	case ChargeClassInGate:
		ledger.inGateReservedWatts, err = checkedAddWatts(ledger.inGateReservedWatts, watts)
	case ChargeClassRolloutExtra:
		ledger.rolloutExtraWatts, err = checkedAddWatts(ledger.rolloutExtraWatts, watts)
	default:
		return fmt.Errorf("aggregate charge has unknown class %q", charge.class)
	}
	if err != nil {
		return fmt.Errorf("aggregate charge: %w", err)
	}
	ledger.totalChargedWatts = newTotal
	return nil
}

// EnforcedWatts returns exact-readback aggregate watts.
func (ledger Ledger) EnforcedWatts() int64 {
	return ledger.enforcedWatts
}

// UnknownWatts returns fail-closed aggregate watts.
func (ledger Ledger) UnknownWatts() int64 {
	return ledger.unknownWatts
}

// InGateReservedWatts returns never-reported gated reservation watts.
func (ledger Ledger) InGateReservedWatts() int64 {
	return ledger.inGateReservedWatts
}

// RolloutExtraWatts returns aggregate rollout capacity outside committed slots.
func (ledger Ledger) RolloutExtraWatts() int64 {
	return ledger.rolloutExtraWatts
}

// TotalChargedWatts returns the checked sum of all four disjoint classes.
func (ledger Ledger) TotalChargedWatts() int64 {
	return ledger.totalChargedWatts
}

func checkedAddWatts(left int64, right int64) (int64, error) {
	if left < 0 || right < 0 {
		return 0, fmt.Errorf("cannot add negative ledger watts %d and %d", left, right)
	}
	if left > math.MaxInt64-right {
		return 0, fmt.Errorf("ledger watt total overflows int64")
	}
	return left + right, nil
}

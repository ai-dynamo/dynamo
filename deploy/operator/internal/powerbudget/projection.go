/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"errors"
	"fmt"
	"maps"
	"math"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// ErrInvalidTarget marks a replica request that cannot be represented by the
// bounded aggregate status contract.
var ErrInvalidTarget = errors.New("invalid replica target")

// ProjectionComponent contains immutable physical capacity for one requested role.
type ProjectionComponent struct {
	PhysicalGPUsPerReplica   int32
	RolloutExtraPhysicalGPUs int32
}

// ObligationKind distinguishes committed slots from rollout-only capacity.
type ObligationKind string

const (
	// ObligationKindCommittedSlot is one physical GPU represented by a committed replica.
	ObligationKindCommittedSlot ObligationKind = "committedSlot"
	// ObligationKindRolloutExtra is one old, new, surge, or terminating physical GPU
	// outside the committed replica slots.
	ObligationKindRolloutExtra ObligationKind = "rolloutExtra"
)

// ChargeObligation binds one unique physical-GPU or reserved-slot identity to a
// component and its already classified conservative charge.
type ChargeObligation struct {
	ID        string
	Component string
	Kind      ObligationKind
	Charge    GPUCharge
}

// AdmissionProjection is a component-complete, charge-complete ledger projection.
// Its fields are private so callers must use BuildAdmissionProjection.
type AdmissionProjection struct {
	requested ReplicaVector
	ledger    Ledger
}

// BuildAdmissionProjection proves that every requested physical-GPU slot is
// charged exactly once and that rollout extras use their separate class.
func BuildAdmissionProjection(
	requested ReplicaVector,
	components map[string]ProjectionComponent,
	obligations []ChargeObligation,
) (AdmissionProjection, error) {
	if len(requested) == 0 || len(components) != len(requested) {
		return AdmissionProjection{}, fmt.Errorf("projection components must match the requested vector")
	}

	expectedCommitted := make(map[string]int64, len(requested))
	expectedRolloutExtra := make(map[string]int64, len(requested))
	for component, replicas := range requested {
		capacity, found := components[component]
		if !found {
			return AdmissionProjection{}, fmt.Errorf("projection is missing component %q", component)
		}
		if replicas < 0 || capacity.PhysicalGPUsPerReplica < 1 || capacity.RolloutExtraPhysicalGPUs < 0 {
			return AdmissionProjection{}, fmt.Errorf(
				"component %q has invalid replicas=%d physicalGPUsPerReplica=%d rolloutExtraPhysicalGPUs=%d",
				component,
				replicas,
				capacity.PhysicalGPUsPerReplica,
				capacity.RolloutExtraPhysicalGPUs,
			)
		}
		expectedCommitted[component] = int64(replicas) * int64(capacity.PhysicalGPUsPerReplica)
		expectedRolloutExtra[component] = int64(capacity.RolloutExtraPhysicalGPUs)
	}
	for component := range components {
		if _, found := requested[component]; !found {
			return AdmissionProjection{}, fmt.Errorf("projection contains unknown component %q", component)
		}
	}

	actualCommitted := make(map[string]int64, len(requested))
	actualRolloutExtra := make(map[string]int64, len(requested))
	var ledger Ledger
	for _, obligation := range obligations {
		if _, found := requested[obligation.Component]; !found {
			return AdmissionProjection{}, fmt.Errorf("obligation %q has unknown component %q", obligation.ID, obligation.Component)
		}
		switch obligation.Kind {
		case ObligationKindCommittedSlot:
			if obligation.Charge.class == ChargeClassRolloutExtra {
				return AdmissionProjection{}, fmt.Errorf("committed obligation %q uses rollout-extra charge", obligation.ID)
			}
			actualCommitted[obligation.Component]++
		case ObligationKindRolloutExtra:
			if obligation.Charge.class != ChargeClassRolloutExtra {
				return AdmissionProjection{}, fmt.Errorf("rollout obligation %q lacks rollout-extra charge", obligation.ID)
			}
			actualRolloutExtra[obligation.Component]++
		default:
			return AdmissionProjection{}, fmt.Errorf("obligation %q has unknown kind %q", obligation.ID, obligation.Kind)
		}
		if err := ledger.AddCharge(obligation.ID, obligation.Charge); err != nil {
			return AdmissionProjection{}, err
		}
	}

	for component, expected := range expectedCommitted {
		if actual := actualCommitted[component]; actual != expected {
			return AdmissionProjection{}, fmt.Errorf(
				"component %q has %d committed GPU charges, expected %d",
				component,
				actual,
				expected,
			)
		}
		if actual := actualRolloutExtra[component]; actual != expectedRolloutExtra[component] {
			return AdmissionProjection{}, fmt.Errorf(
				"component %q has %d rollout-extra GPU charges, expected %d",
				component,
				actual,
				expectedRolloutExtra[component],
			)
		}
	}
	return AdmissionProjection{requested: maps.Clone(requested), ledger: ledger}, nil
}

// BuildObservedAdmissionProjection reconstructs the current admission ledger
// from the bounded DGPB status while proving that every committed slot is
// represented by exactly one ordinary charge class. Rollout-extra capacity is
// already disjoint in the aggregate status and never satisfies a committed slot.
func BuildObservedAdmissionProjection(
	committed ReplicaVector,
	components []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus,
	status nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus,
) (AdmissionProjection, error) {
	if len(committed) == 0 || len(components) != len(committed) {
		return AdmissionProjection{}, fmt.Errorf("observed components must match the committed vector")
	}

	seen := make(map[string]struct{}, len(components))
	for _, component := range components {
		replicas, found := committed[component.Name]
		if !found {
			return AdmissionProjection{}, fmt.Errorf("observed component %q is not committed", component.Name)
		}
		if _, duplicate := seen[component.Name]; duplicate {
			return AdmissionProjection{}, fmt.Errorf("observed component %q is duplicated", component.Name)
		}
		seen[component.Name] = struct{}{}
		if replicas < 0 || component.PhysicalGPUsPerReplica < 1 ||
			component.EnforcedPhysicalGPUs < 0 || component.UnknownPhysicalGPUs < 0 ||
			component.InGatePhysicalGPUs < 0 {
			return AdmissionProjection{}, fmt.Errorf("observed component %q has invalid capacity", component.Name)
		}

		expected := int64(replicas) * int64(component.PhysicalGPUsPerReplica)
		observed := int64(component.EnforcedPhysicalGPUs) +
			int64(component.UnknownPhysicalGPUs) + int64(component.InGatePhysicalGPUs)
		if observed != expected {
			return AdmissionProjection{}, fmt.Errorf(
				"observed component %q has %d committed GPU charges, expected %d",
				component.Name,
				observed,
				expected,
			)
		}
	}

	ledger, err := newLedgerFromTotals(ledgerTotals{
		EnforcedWatts:       status.EnforcedWatts,
		UnknownWatts:        status.UnknownWatts,
		InGateReservedWatts: status.InGateReservedWatts,
		RolloutExtraWatts:   status.RolloutExtraWatts,
	})
	if err != nil {
		return AdmissionProjection{}, err
	}
	if ledger.TotalChargedWatts() != status.TotalChargedWatts {
		return AdmissionProjection{}, fmt.Errorf(
			"observed totalChargedWatts=%d, computed=%d",
			status.TotalChargedWatts,
			ledger.TotalChargedWatts(),
		)
	}
	return AdmissionProjection{requested: maps.Clone(committed), ledger: ledger}, nil
}

// BuildIncrementalAdmissionProjection retains every currently observed charge
// and adds an in-gate B_c reservation for each newly committed physical GPU.
// Decreasing targets do not release their old charge here; watched inventory
// later moves surviving capacity into rollout-extra until deletion is observed.
func BuildIncrementalAdmissionProjection(
	committed ReplicaVector,
	requested ReplicaVector,
	components []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus,
	current AdmissionProjection,
) (AdmissionProjection, error) {
	if !sameComponents(committed, requested) || !maps.Equal(current.requested, committed) {
		return AdmissionProjection{}, fmt.Errorf("incremental projection vectors must share the current components")
	}
	byName := make(map[string]nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus, len(components))
	for _, component := range components {
		if _, duplicate := byName[component.Name]; duplicate {
			return AdmissionProjection{}, fmt.Errorf("duplicate power-budget component %q", component.Name)
		}
		byName[component.Name] = component
	}
	if len(byName) != len(requested) {
		return AdmissionProjection{}, fmt.Errorf("incremental projection components must match the requested vector")
	}

	ledger := current.Ledger()
	for componentName, target := range requested {
		component, found := byName[componentName]
		if !found || component.PhysicalGPUsPerReplica < 1 {
			return AdmissionProjection{}, fmt.Errorf("invalid incremental projection component %q", componentName)
		}
		if target < 0 {
			return AdmissionProjection{}, fmt.Errorf("%w: component %q has negative replicas", ErrInvalidTarget, componentName)
		}
		physicalGPUs := int64(target) * int64(component.PhysicalGPUsPerReplica)
		if physicalGPUs > math.MaxInt32 {
			return AdmissionProjection{}, fmt.Errorf(
				"%w: component %q requires %d physical GPUs, maximum representable is %d",
				ErrInvalidTarget,
				componentName,
				physicalGPUs,
				math.MaxInt32,
			)
		}
		addedReplicas := max(int32(0), target-committed[componentName])
		addedGPUs := int64(addedReplicas) * int64(component.PhysicalGPUsPerReplica)
		charge, err := ClassifyGPUCharge(GPUChargeInput{
			Bounds: ComponentBounds{
				InGateWatts:     component.InGateBoundWattsPerGPU,
				UnenforcedWatts: component.UnenforcedBoundWattsPerGPU,
			},
			InjectedGate: true,
		})
		if err != nil {
			return AdmissionProjection{}, err
		}
		if err := ledger.AddAggregateCharge(addedGPUs, charge); err != nil {
			return AdmissionProjection{}, fmt.Errorf(
				"add incremental reservation for component %q: %w",
				componentName,
				err,
			)
		}
	}
	return AdmissionProjection{requested: maps.Clone(requested), ledger: ledger}, nil
}

// TotalChargedWatts returns the checked aggregate projection total.
func (projection AdmissionProjection) TotalChargedWatts() int64 {
	return projection.ledger.TotalChargedWatts()
}

// Ledger returns an isolated checked ledger copy for durable status production.
func (projection AdmissionProjection) Ledger() Ledger {
	ledger := projection.ledger
	ledger.charged = maps.Clone(projection.ledger.charged)
	return ledger
}

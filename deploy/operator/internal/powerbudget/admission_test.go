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
	"testing"
	"testing/quick"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestAdmitVector(t *testing.T) {
	spec := mustSpec(t, 1000, 2)
	committed := ReplicaVector{"prefill-worker": 2, "decode-worker": 2}
	healthy := AdmissionState{
		Phase:             nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle,
		TopologySupported: true,
		HardwareQualified: true,
	}
	current := mustProjection(t, committed, ChargeClassEnforced, 175)

	t.Log("Reject an over-budget complete increase without changing any role")
	requested := ReplicaVector{"prefill-worker": 3, "decode-worker": 2}
	decision := AdmitVector(spec, committed, requested, current, mustProjection(t, requested, ChargeClassInGate, 201), healthy)
	if decision.Accepted {
		t.Fatal("AdmitVector() accepted an over-budget vector")
	}
	if decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded {
		t.Fatalf("pending reason = %q, want BudgetExceeded", decision.PendingReason)
	}
	assertVectorUnchanged(t, decision.Committed, committed)
	assertVectorUnchanged(t, committed, ReplicaVector{"prefill-worker": 2, "decode-worker": 2})

	t.Log("Reject an incomplete vector rather than partially committing one role")
	decision = AdmitVector(spec, committed, ReplicaVector{"prefill-worker": 3}, AdmissionProjection{}, AdmissionProjection{}, healthy)
	if decision.Accepted || decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonInvalidTarget {
		t.Fatalf("incomplete admission decision = %#v", decision)
	}
	assertVectorUnchanged(t, decision.Committed, committed)

	t.Log("Block increases from unknown evidence with UnenforcedBaseline")
	unknownBaseline := mustProjection(t, committed, ChargeClassUnknown, 175)
	decision = AdmitVector(spec, committed, requested, unknownBaseline, mustProjection(t, requested, ChargeClassInGate, 180), healthy)
	if decision.Accepted || decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline {
		t.Fatalf("unknown-baseline admission decision = %#v", decision)
	}
	assertVectorUnchanged(t, decision.Committed, committed)

	t.Log("Accept every role together when a complete healthy projection fits")
	decision = AdmitVector(spec, committed, requested, current, mustProjection(t, requested, ChargeClassInGate, 180), healthy)
	if !decision.Accepted || decision.PendingReason != "" {
		t.Fatalf("safe admission decision = %#v", decision)
	}
	assertVectorUnchanged(t, decision.Committed, requested)

	t.Log("Allow a floor-respecting reduction while stale and still over budget")
	stale := AdmissionState{
		Phase:             nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale,
		TopologySupported: true,
		HardwareQualified: true,
	}
	large := ReplicaVector{"prefill-worker": 3, "decode-worker": 3}
	reduced := ReplicaVector{"prefill-worker": 2, "decode-worker": 2}
	decision = AdmitVector(
		spec,
		large,
		reduced,
		mustProjection(t, large, ChargeClassUnknown, 250),
		mustProjection(t, reduced, ChargeClassUnknown, 300),
		stale,
	)
	if !decision.Accepted || !maps.Equal(decision.Committed, reduced) {
		t.Fatalf("safe reduction decision = %#v", decision)
	}

	t.Log("Allow a complete mixed vector whose larger reduction outweighs its increase")
	mixedSpec := mustSpec(t, 2000, 2)
	mixedCurrent := ReplicaVector{"prefill-worker": 4, "decode-worker": 2}
	mixedRequested := ReplicaVector{"prefill-worker": 2, "decode-worker": 3}
	decision = AdmitVector(
		mixedSpec,
		mixedCurrent,
		mixedRequested,
		mustProjection(t, mixedCurrent, ChargeClassEnforced, 200),
		mustProjection(t, mixedRequested, ChargeClassInGate, 200),
		healthy,
	)
	if !decision.Accepted || !maps.Equal(decision.Committed, mixedRequested) {
		t.Fatalf("mixed-vector decision = %#v", decision)
	}

	t.Log("Give InvalidTarget deterministic priority for mixed invalid values")
	for range 1000 {
		decision = AdmitVector(
			spec,
			committed,
			ReplicaVector{"prefill-worker": -1, "decode-worker": 1},
			AdmissionProjection{},
			AdmissionProjection{},
			healthy,
		)
		if decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonInvalidTarget {
			t.Fatalf("mixed-invalid pending reason = %q, want InvalidTarget", decision.PendingReason)
		}
	}
}

func TestBootstrapFromZero(t *testing.T) {
	spec := mustSpec(t, 1400, 2)
	zero := ReplicaVector{"prefill-worker": 0, "decode-worker": 0}
	seed := ReplicaVector{"prefill-worker": 2, "decode-worker": 2}
	zeroProjection := mustProjection(t, zero, ChargeClassInGate, 350)
	initializing := AdmissionState{
		Phase:             nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing,
		TopologySupported: true,
		HardwareQualified: true,
	}

	t.Log("Evaluate and commit the first component-complete seed from zero")
	decision := AdmitVector(spec, zero, seed, zeroProjection, mustProjection(t, seed, ChargeClassInGate, 350), initializing)
	if !decision.Accepted || !maps.Equal(decision.Committed, seed) {
		t.Fatalf("safe bootstrap decision = %#v", decision)
	}

	t.Log("Keep every committed target at zero when the first seed is unsafe")
	decision = AdmitVector(spec, zero, seed, zeroProjection, mustProjection(t, seed, ChargeClassInGate, 351), initializing)
	if decision.Accepted || !maps.Equal(decision.Committed, zero) {
		t.Fatalf("unsafe bootstrap decision = %#v", decision)
	}
	if decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded {
		t.Fatalf("pending reason = %q, want BudgetExceeded", decision.PendingReason)
	}

	t.Log("Do not reopen bootstrap while a zero vector is stale")
	stale := initializing
	stale.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale
	decision = AdmitVector(spec, zero, seed, zeroProjection, mustProjection(t, seed, ChargeClassInGate, 350), stale)
	if decision.Accepted || decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline {
		t.Fatalf("stale zero-vector decision = %#v", decision)
	}
}

func TestAdmitVectorProperties(t *testing.T) {
	t.Log("Check arbitrary replica increases require every projected GPU obligation")
	property := func(extra uint8, budgetSeed uint16) bool {
		budget := int64(budgetSeed) + 1
		spec := mustSpec(t, budget, 1)
		committed := ReplicaVector{"prefill": 1, "decode": 1}
		requested := ReplicaVector{"prefill": int32(extra) + 2, "decode": 1}
		current := mustProjection(t, committed, ChargeClassEnforced, 1)
		projected := mustProjection(t, requested, ChargeClassInGate, 1)
		decision := AdmitVector(
			spec,
			committed,
			requested,
			current,
			projected,
			AdmissionState{
				Phase:             nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle,
				TopologySupported: true,
				HardwareQualified: true,
			},
		)
		return decision.Accepted == (projected.TotalChargedWatts() <= budget)
	}
	if err := quick.Check(property, &quick.Config{MaxCount: 1000}); err != nil {
		t.Fatalf("admission property failed: %v", err)
	}

	t.Log("Expand replicas by physical GPUs and reject one omitted obligation")
	requested := ReplicaVector{"prefill": 2}
	components := map[string]ProjectionComponent{"prefill": {
		PhysicalGPUsPerReplica:   2,
		RolloutExtraPhysicalGPUs: 1,
	}}
	obligations := make([]ChargeObligation, 0, 5)
	for slot := range 4 {
		obligations = append(obligations, ChargeObligation{
			ID:        fmt.Sprintf("prefill/%d", slot),
			Component: "prefill",
			Kind:      ObligationKindCommittedSlot,
			Charge:    GPUCharge{class: ChargeClassInGate, watts: 350},
		})
	}
	if _, err := BuildAdmissionProjection(requested, components, obligations[:3]); err == nil {
		t.Fatal("BuildAdmissionProjection() accepted an omitted GPU obligation")
	}

	t.Log("Charge rollout capacity separately without satisfying a committed slot")
	if _, err := BuildAdmissionProjection(requested, components, obligations); err == nil {
		t.Fatal("BuildAdmissionProjection() accepted an omitted rollout-extra obligation")
	}
	obligations = append(obligations, ChargeObligation{
		ID:        "prefill/rollout-extra",
		Component: "prefill",
		Kind:      ObligationKindRolloutExtra,
		Charge:    GPUCharge{class: ChargeClassRolloutExtra, watts: 350},
	})
	projection, err := BuildAdmissionProjection(requested, components, obligations)
	if err != nil {
		t.Fatalf("BuildAdmissionProjection() with rollout extra error = %v", err)
	}
	if got, want := projection.TotalChargedWatts(), int64(1750); got != want {
		t.Fatalf("projection total with rollout extra = %d, want %d", got, want)
	}
}

func TestExtremeReplicaRequestIsBounded(t *testing.T) {
	committed := ReplicaVector{"worker": 1}
	requested := ReplicaVector{"worker": math.MaxInt32}
	components := []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{{
		Name:                       "worker",
		PhysicalGPUsPerReplica:     1,
		InGateBoundWattsPerGPU:     300,
		UnenforcedBoundWattsPerGPU: 700,
		EnforcedPhysicalGPUs:       1,
	}}
	current, err := BuildObservedAdmissionProjection(
		committed,
		components,
		nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
			EnforcedWatts: 300, TotalChargedWatts: 300,
		},
	)
	if err != nil {
		t.Fatalf("build current projection: %v", err)
	}

	t.Log("Aggregate a MaxInt32 request without allocating one identity per GPU")
	projected, err := BuildIncrementalAdmissionProjection(committed, requested, components, current)
	if err != nil {
		t.Fatalf("build extreme incremental projection: %v", err)
	}
	wantTotal := int64(math.MaxInt32) * 300
	if projected.TotalChargedWatts() != wantTotal || len(projected.ledger.charged) != 0 {
		t.Fatalf(
			"extreme projection = total %d identities %d, want total %d and no materialized identities",
			projected.TotalChargedWatts(),
			len(projected.ledger.charged),
			wantTotal,
		)
	}

	decision := AdmitVector(
		mustSpec(t, 1000, 1),
		committed,
		requested,
		current,
		projected,
		AdmissionState{
			Phase:             nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle,
			TopologySupported: true,
			HardwareQualified: true,
		},
	)
	if decision.Accepted || decision.PendingReason != nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded {
		t.Fatalf("extreme request decision = %#v, want BudgetExceeded", decision)
	}

	t.Log("Accept the same representable aggregate promptly when its large budget fits")
	decision = AdmitVector(
		mustSpec(t, wantTotal, 1),
		committed,
		requested,
		current,
		projected,
		AdmissionState{
			Phase:             nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle,
			TopologySupported: true,
			HardwareQualified: true,
		},
	)
	if !decision.Accepted || decision.PendingReason != "" {
		t.Fatalf("large-budget extreme request decision = %#v, want accepted", decision)
	}

	t.Log("Reject a target whose physical-GPU count cannot fit bounded int32 status")
	tooWideComponents := append([]nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus(nil), components...)
	tooWideComponents[0].PhysicalGPUsPerReplica = 2
	tooWideComponents[0].EnforcedPhysicalGPUs = 2
	tooWideCurrent, err := BuildObservedAdmissionProjection(
		committed,
		tooWideComponents,
		nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
			EnforcedWatts: 600, TotalChargedWatts: 600,
		},
	)
	if err != nil {
		t.Fatalf("build wider current projection: %v", err)
	}
	if _, err := BuildIncrementalAdmissionProjection(committed, requested, tooWideComponents, tooWideCurrent); !errors.Is(err, ErrInvalidTarget) {
		t.Fatalf("unrepresentable physical-GPU request error = %v, want ErrInvalidTarget", err)
	}
}

func mustSpec(t *testing.T, budgetWatts int64, minEndpoint int32) Spec {
	t.Helper()
	spec, err := NewSpec(nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
		BudgetWatts: budgetWatts,
		Policy:      nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{MinEndpoint: minEndpoint},
	})
	if err != nil {
		t.Fatalf("NewSpec() error = %v", err)
	}
	return spec
}

func mustProjection(t *testing.T, vector ReplicaVector, class ChargeClass, wattsPerGPU int64) AdmissionProjection {
	t.Helper()
	projection, err := BuildAdmissionProjection(
		vector,
		projectionComponents(vector),
		projectionObligations(vector, class, wattsPerGPU),
	)
	if err != nil {
		t.Fatalf("BuildAdmissionProjection() error = %v", err)
	}
	return projection
}

func projectionComponents(vector ReplicaVector) map[string]ProjectionComponent {
	components := make(map[string]ProjectionComponent, len(vector))
	for component := range vector {
		components[component] = ProjectionComponent{PhysicalGPUsPerReplica: 1}
	}
	return components
}

func projectionObligations(vector ReplicaVector, class ChargeClass, wattsPerGPU int64) []ChargeObligation {
	var obligations []ChargeObligation
	for component, replicas := range vector {
		for slot := range replicas {
			obligations = append(obligations, ChargeObligation{
				ID:        fmt.Sprintf("%s/%d", component, slot),
				Component: component,
				Kind:      ObligationKindCommittedSlot,
				Charge:    GPUCharge{class: class, watts: wattsPerGPU},
			})
		}
	}
	return obligations
}

func assertVectorUnchanged(t *testing.T, got ReplicaVector, want ReplicaVector) {
	t.Helper()
	if !maps.Equal(got, want) {
		t.Fatalf("replica vector = %v, want %v", got, want)
	}
}

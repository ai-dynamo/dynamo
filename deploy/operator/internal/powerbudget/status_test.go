/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestGoldenSerialization(t *testing.T) {
	t.Log("Pin the durable bounded v1beta1 DGPB status JSON contract")
	replicas := int32(2)
	ledger := mustLedger(t, ledgerTotals{
		EnforcedWatts:       1400,
		InGateReservedWatts: 700,
	})
	status := nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
		DGDUID:                  "dgd-uid",
		ObservedGeneration:      12,
		InventoryEpoch:          42,
		Phase:                   nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying,
		CommittedReplicaTargets: map[string]int32{"prefill": 3},
		Components: []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{{
			Name: "prefill",
			ReplicaStatus: nvidiacomv1beta1.ComponentReplicaStatus{
				ComponentKind:     nvidiacomv1beta1.ComponentKindDeployment,
				Replicas:          3,
				UpdatedReplicas:   2,
				ReadyReplicas:     &replicas,
				AvailableReplicas: &replicas,
				ScheduledReplicas: &replicas,
			},
			RequestedCapWattsPerGPU:    350,
			InGateBoundWattsPerGPU:     350,
			UnenforcedBoundWattsPerGPU: 700,
			PhysicalGPUsPerReplica:     2,
			InGatePhysicalGPUs:         2,
			EnforcedPhysicalGPUs:       4,
		}},
		Ledger: NewLedgerStatus(ledger),
	}

	encoded, err := EncodeStatusSnapshot(status)
	if err != nil {
		t.Fatalf("EncodeStatusSnapshot() error = %v", err)
	}
	want := `{"dgdUID":"dgd-uid","observedGeneration":12,"inventoryEpoch":42,"phase":"Applying","committedReplicaTargets":{"prefill":3},"components":[{"name":"prefill","replicaStatus":{"componentKind":"Deployment","replicas":3,"updatedReplicas":2,"readyReplicas":2,"availableReplicas":2,"scheduledReplicas":2},"requestedCapWattsPerGPU":350,"inGateBoundWattsPerGPU":350,"unenforcedBoundWattsPerGPU":700,"physicalGPUsPerReplica":2,"inGatePhysicalGPUs":2,"terminatingReplicas":0,"enforcedPhysicalGPUs":4,"unknownPhysicalGPUs":0}],"ledger":{"enforcedWatts":1400,"unknownWatts":0,"inGateReservedWatts":700,"rolloutExtraWatts":0,"totalChargedWatts":2100}}`
	if string(encoded) != want {
		t.Fatalf("encoded DGPB status = %s, want %s", encoded, want)
	}
}

func TestStatusBounded(t *testing.T) {
	t.Log("Keep component rows aggregate-only with no per-GPU identities or reports")
	componentType := reflect.TypeFor[nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus]()
	for index := range componentType.NumField() {
		field := componentType.Field(index)
		lowerName := strings.ToLower(field.Name)
		if lowerName == "gpus" || strings.Contains(lowerName, "uuid") || strings.Contains(lowerName, "report") {
			t.Fatalf("component field %s persists per-GPU evidence", field.Name)
		}
	}

	t.Log("Encode 25 aggregate components plus the bounded condition payload")
	status := nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
		DGDUID:                  strings.Repeat("d", 36),
		ObservedGeneration:      123456789,
		InventoryEpoch:          123456789,
		Phase:                   nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible,
		CommittedReplicaTargets: make(map[string]int32, MaxPowerManagedComponents),
		Components:              make([]nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus, 0, MaxPowerManagedComponents),
		Ledger:                  nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{},
		RequiredWatts:           9_999_999_999,
		AvailableWatts:          9_000_000_000,
		Conditions: []metav1.Condition{{
			Type:               nvidiacomv1beta1.DynamoGraphPowerBudgetConditionTypePowerInfeasible,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: 123456789,
			LastTransitionTime: metav1.NewTime(time.Date(2026, time.August, 15, 12, 0, 0, 0, time.UTC)),
			Reason:             "MinimumEndpointExceedsBudget",
			Message:            strings.Repeat("m", MaxStatusConditionMessageBytes),
		}},
	}
	for index := range MaxPowerManagedComponents {
		name := fmt.Sprintf("component-%02d-%s", index, strings.Repeat("x", 50))
		status.CommittedReplicaTargets[name] = 9999
		status.Components = append(status.Components, nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{
			Name: name,
			ReplicaStatus: nvidiacomv1beta1.ComponentReplicaStatus{
				ComponentKind:   nvidiacomv1beta1.ComponentKindDeployment,
				Replicas:        9999,
				UpdatedReplicas: 9999,
			},
			RequestedCapWattsPerGPU:    9999,
			InGateBoundWattsPerGPU:     9999,
			UnenforcedBoundWattsPerGPU: 9999,
			PhysicalGPUsPerReplica:     9999,
			InGatePhysicalGPUs:         9999,
			TerminatingReplicas:        9999,
			EnforcedPhysicalGPUs:       9999,
			UnknownPhysicalGPUs:        9999,
		})
	}

	encoded, err := EncodeStatusSnapshot(status)
	if err != nil {
		t.Fatalf("EncodeStatusSnapshot() at component limit error = %v", err)
	}
	if len(encoded) >= MaxEncodedStatusBytes {
		t.Fatalf("encoded status size = %d, limit = %d", len(encoded), MaxEncodedStatusBytes)
	}

	t.Log("Reject a twenty-sixth component instead of growing status without bound")
	status.Components = append(status.Components, nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{Name: "component-over-limit"})
	if _, err := EncodeStatusSnapshot(status); err == nil {
		t.Fatal("EncodeStatusSnapshot() accepted more than 25 components")
	}
	status.Components = status.Components[:MaxPowerManagedComponents]

	t.Log("Reject additional or unbounded condition payloads")
	status.Conditions = append(status.Conditions, metav1.Condition{Type: "extra"})
	if _, err := EncodeStatusSnapshot(status); err == nil {
		t.Fatal("EncodeStatusSnapshot() accepted more than one condition")
	}
	status.Conditions = status.Conditions[:1]
	status.Conditions[0].Message += "x"
	if _, err := EncodeStatusSnapshot(status); err == nil {
		t.Fatal("EncodeStatusSnapshot() accepted an oversized condition message")
	}
}

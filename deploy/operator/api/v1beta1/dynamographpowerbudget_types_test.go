/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1beta1

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestTransactionalPowerCRDsAreRegisteredForKustomizeInstall(t *testing.T) {
	t.Log("Locate the Operator CRD kustomization from this source file")
	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve current test source path")
	}
	kustomizationPath := filepath.Join(filepath.Dir(sourceFile), "..", "..", "config", "crd", "kustomization.yaml")

	t.Log("Require each transactional power CRD exactly once in the raw install surface")
	kustomization, err := os.ReadFile(kustomizationPath)
	if err != nil {
		t.Fatalf("read %s: %v", kustomizationPath, err)
	}
	for _, resource := range []string{
		"- bases/nvidia.com_dynamographdeploymentscalingadapters.yaml",
		"- bases/nvidia.com_dynamographpowerbudgets.yaml",
	} {
		if got := strings.Count(string(kustomization), resource); got != 1 {
			t.Errorf("%s registration count = %d, want 1", resource, got)
		}
	}
}

func TestDGPBSpecHasOnlyImmutablePolicyInputs(t *testing.T) {
	t.Log("Inspect the public DGPB spec fields")
	specType := reflect.TypeFor[DynamoGraphPowerBudgetSpec]()
	wantFields := map[string]string{
		"BudgetWatts": "budgetWatts",
		"Policy":      "policy",
	}
	if specType.NumField() != len(wantFields) {
		t.Fatalf("DGPB spec field count = %d, want %d", specType.NumField(), len(wantFields))
	}

	t.Log("Require budget and policy without omitempty defaults")
	for fieldName, wantJSONName := range wantFields {
		field, ok := specType.FieldByName(fieldName)
		if !ok {
			t.Fatalf("DGPB spec missing %s", fieldName)
		}
		if got := field.Tag.Get("json"); got != wantJSONName {
			t.Errorf("%s JSON tag = %q, want %q", fieldName, got, wantJSONName)
		}
	}

	t.Log("Keep request proposals out of the immutable policy resource")
	if _, found := specType.FieldByName("Proposal"); found {
		t.Fatal("DGPB spec must not contain a Planner-writable proposal")
	}
}

func TestDGPBSpecPolicyRequiresScalarMinimum(t *testing.T) {
	t.Log("Inspect the scalar policy contract")
	policyType := reflect.TypeFor[DynamoGraphPowerBudgetPolicy]()
	if policyType.NumField() != 1 {
		t.Fatalf("DGPB policy field count = %d, want 1", policyType.NumField())
	}

	t.Log("Require minEndpoint without an omitempty default")
	field, ok := policyType.FieldByName("MinEndpoint")
	if !ok {
		t.Fatal("DGPB policy missing MinEndpoint")
	}
	if got := field.Tag.Get("json"); got != "minEndpoint" {
		t.Fatalf("MinEndpoint JSON tag = %q, want %q", got, "minEndpoint")
	}
}

func TestDGPBStatusKeepsApprovedAggregateContract(t *testing.T) {
	t.Log("Keep internal convergence metadata out of the public DGPB status schema")
	statusType := reflect.TypeFor[DynamoGraphPowerBudgetStatus]()
	wantFields := map[string]string{
		"DGDUID":                  "dgdUID,omitempty",
		"ObservedGeneration":      "observedGeneration,omitempty",
		"InventoryEpoch":          "inventoryEpoch,omitempty",
		"Phase":                   "phase,omitempty",
		"CommittedReplicaTargets": "committedReplicaTargets,omitempty",
		"Components":              "components,omitempty",
		"Ledger":                  "ledger,omitempty",
		"RolloutInProgress":       "rolloutInProgress,omitempty",
		"RequiredWatts":           "requiredWatts,omitempty",
		"AvailableWatts":          "availableWatts,omitempty",
		"Conditions":              "conditions,omitempty",
	}
	if statusType.NumField() != len(wantFields) {
		t.Fatalf("DGPB status field count = %d, want %d", statusType.NumField(), len(wantFields))
	}
	for fieldName, wantJSONTag := range wantFields {
		field, ok := statusType.FieldByName(fieldName)
		if !ok {
			t.Fatalf("DGPB status missing %s", fieldName)
		}
		if got := field.Tag.Get("json"); got != wantJSONTag {
			t.Errorf("%s JSON tag = %q, want %q", fieldName, got, wantJSONTag)
		}
	}
}

func TestPendingReasons(t *testing.T) {
	t.Log("Keep replica-request pending reasons bounded and stable")
	got := []DynamoGraphPowerBudgetPendingReason{
		DynamoGraphPowerBudgetPendingReasonBudgetExceeded,
		DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline,
		DynamoGraphPowerBudgetPendingReasonUnsupportedTopology,
		DynamoGraphPowerBudgetPendingReasonUnqualifiedHardware,
		DynamoGraphPowerBudgetPendingReasonBelowMinimum,
		DynamoGraphPowerBudgetPendingReasonInvalidTarget,
	}
	want := []DynamoGraphPowerBudgetPendingReason{
		"BudgetExceeded",
		"UnenforcedBaseline",
		"UnsupportedTopology",
		"UnqualifiedHardware",
		"BelowMinimum",
		"InvalidTarget",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("pending reasons = %v, want %v", got, want)
	}

	t.Log("Represent minimum-footprint infeasibility as a condition, not a pending reason")
	if DynamoGraphPowerBudgetConditionTypePowerInfeasible != "PowerInfeasible" {
		t.Fatalf("condition type = %q, want PowerInfeasible", DynamoGraphPowerBudgetConditionTypePowerInfeasible)
	}
	for _, reason := range got {
		if string(reason) == DynamoGraphPowerBudgetConditionTypePowerInfeasible {
			t.Fatal("PowerInfeasible must not be a pending reason")
		}
	}
}

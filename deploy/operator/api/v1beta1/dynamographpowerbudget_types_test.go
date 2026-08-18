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

	apitest "k8s.io/apiextensions-apiserver/pkg/test"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
)

func TestDGPBCRDIsRegisteredForKustomizeInstall(t *testing.T) {
	t.Log("Locate the Operator CRD kustomization from this source file")
	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve current test source path")
	}
	kustomizationPath := filepath.Join(filepath.Dir(sourceFile), "..", "..", "config", "crd", "kustomization.yaml")

	t.Log("Require the DGPB CRD exactly once in the raw install surface")
	kustomization, err := os.ReadFile(kustomizationPath)
	if err != nil {
		t.Fatalf("read %s: %v", kustomizationPath, err)
	}
	resource := "- bases/nvidia.com_dynamographpowerbudgets.yaml"
	if got := strings.Count(string(kustomization), resource); got != 1 {
		t.Errorf("%s registration count = %d, want 1", resource, got)
	}
}

func TestDGPBTypesAreRegisteredWithScheme(t *testing.T) {
	t.Log("Register the v1beta1 API types")
	scheme := k8sruntime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 to scheme: %v", err)
	}

	t.Log("Define the expected object and list registrations")
	tests := []struct {
		name string
		gvk  string
		want any
	}{
		{name: "object", gvk: "DynamoGraphPowerBudget", want: &DynamoGraphPowerBudget{}},
		{name: "list", gvk: "DynamoGraphPowerBudgetList", want: &DynamoGraphPowerBudgetList{}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Construct the registered type by GVK")
			got, err := scheme.New(GroupVersion.WithKind(test.gvk))
			if err != nil {
				t.Fatalf("construct %s from scheme: %v", test.gvk, err)
			}

			t.Log("Require the registered concrete API type")
			if reflect.TypeOf(got) != reflect.TypeOf(test.want) {
				t.Fatalf("scheme type = %T, want %T", got, test.want)
			}
		})
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

func TestPowerInfeasibleConditionType(t *testing.T) {
	t.Log("Keep minimum-footprint infeasibility represented as a condition")
	if DynamoGraphPowerBudgetConditionTypePowerInfeasible != "PowerInfeasible" {
		t.Fatalf("condition type = %q, want PowerInfeasible", DynamoGraphPowerBudgetConditionTypePowerInfeasible)
	}
}

func TestDGPBCommittedReplicaTargetsRejectNegativeValues(t *testing.T) {
	t.Log("Locate the generated DGPB CRD")
	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve current test source path")
	}
	crdPath := filepath.Join(filepath.Dir(sourceFile), "..", "..", "config", "crd", "bases", "nvidia.com_dynamographpowerbudgets.yaml")

	t.Log("Compile the v1beta1 CEL validator")
	validator, found := apitest.VersionValidatorsFromFile(t, crdPath)[GroupVersion.Version]
	if !found {
		t.Fatalf("DGPB CRD has no %s CEL validator", GroupVersion.Version)
	}

	tests := []struct {
		name         string
		targets      map[string]int32
		wantRejected bool
	}{
		{name: "zero target", targets: map[string]int32{"decode": 0}},
		{name: "positive target", targets: map[string]int32{"decode": 2}},
		{name: "negative target", targets: map[string]int32{"decode": -1}, wantRejected: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build a DGPB with the requested committed targets")
			object := &DynamoGraphPowerBudget{
				TypeMeta: metav1.TypeMeta{
					APIVersion: GroupVersion.String(),
					Kind:       "DynamoGraphPowerBudget",
				},
				ObjectMeta: metav1.ObjectMeta{Name: "test-budget", Namespace: "default"},
				Spec: DynamoGraphPowerBudgetSpec{
					BudgetWatts: 1000,
					Policy:      DynamoGraphPowerBudgetPolicy{MinEndpoint: 1},
				},
				Status: DynamoGraphPowerBudgetStatus{CommittedReplicaTargets: test.targets},
			}
			current, err := k8sruntime.DefaultUnstructuredConverter.ToUnstructured(object)
			if err != nil {
				t.Fatalf("convert DGPB to unstructured: %v", err)
			}

			t.Log("Evaluate the generated CEL rules")
			validationErrors := validator(current, nil)
			if gotRejected := len(validationErrors) != 0; gotRejected != test.wantRejected {
				t.Fatalf("CEL rejected = %t, want %t: %v", gotRejected, test.wantRejected, validationErrors)
			}
		})
	}
}

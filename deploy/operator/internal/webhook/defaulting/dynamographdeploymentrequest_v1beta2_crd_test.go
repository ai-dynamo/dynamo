/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/yaml"
)

// v1beta2SpecSchema returns the spec schema of the generated v1beta2 Request CRD.
func v1beta2SpecSchema(t *testing.T) apiextensionsv1.JSONSchemaProps {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(thisFile), "../../../config/crd/bases", "nvidia.com_dynamographdeploymentrequests.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read the generated CRD: %v", err)
	}
	var crd apiextensionsv1.CustomResourceDefinition
	if err := yaml.Unmarshal(raw, &crd); err != nil {
		t.Fatalf("decode the generated CRD: %v", err)
	}
	for _, version := range crd.Spec.Versions {
		if version.Name == "v1beta2" {
			return version.Schema.OpenAPIV3Schema.Properties["spec"]
		}
	}
	t.Fatal("the generated CRD has no v1beta2 version")
	return apiextensionsv1.JSONSchemaProps{}
}

func schemaDefault(t *testing.T, prop apiextensionsv1.JSONSchemaProps) any {
	t.Helper()
	if prop.Default == nil {
		return nil
	}
	var value any
	if err := json.Unmarshal(prop.Default.Raw, &value); err != nil {
		t.Fatalf("decode a schema default: %v", err)
	}
	return value
}

// A schema default on a top-level native field is also applied to an object that carries
// spec.v1beta1, which puts native fields next to the carrier and makes conversion reject it.
func TestV1beta2CRDHasNoTopLevelStructuralDefaults(t *testing.T) {
	spec := v1beta2SpecSchema(t)
	for _, field := range []string{"backends", "workload", "objective", "recommendation"} {
		prop, ok := spec.Properties[field]
		if !ok {
			t.Fatalf("spec.%s is missing from the v1beta2 schema", field)
		}
		if prop.Default != nil {
			t.Errorf("spec.%s has a schema default %s; top-level defaults belong in admission", field, string(prop.Default.Raw))
		}
	}
}

// Admission creates workload, objective and recommendation itself, and the API server then
// defaults the mutated object in the webhook's version, so the nested schema defaults would
// not apply to them. The webhook sets those values explicitly and they must equal the schema.
func TestV1beta2NestedDefaultsMatchTheCRD(t *testing.T) {
	spec := v1beta2SpecSchema(t)

	wantStatic := map[string]any{
		"isl":         float64(defaultStaticISLV1beta2),
		"osl":         float64(defaultStaticOSLV1beta2),
		"concurrency": float64(defaultStaticConcurrencyV1beta2),
	}
	got := schemaDefault(t, spec.Properties["workload"].Properties["static"])
	if !reflect.DeepEqual(got, wantStatic) {
		t.Errorf("workload.static default = %v, webhook constants = %v", got, wantStatic)
	}

	if got := schemaDefault(t, spec.Properties["objective"].Properties["mode"]); got != "Optimize" {
		t.Errorf("objective.mode default = %v, webhook sets Optimize", got)
	}

	if got := schemaDefault(t, spec.Properties["recommendation"].Properties["maxCandidates"]); got != float64(defaultMaxCandidatesV1beta2) {
		t.Errorf("recommendation.maxCandidates default = %v, webhook constant = %d", got, defaultMaxCandidatesV1beta2)
	}
}

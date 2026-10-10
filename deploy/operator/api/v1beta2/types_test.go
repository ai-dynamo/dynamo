/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package v1beta2

import (
	"reflect"
	"strings"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestAddToSchemeRegistersSearchResources(t *testing.T) {
	t.Parallel()

	t.Log("Create a scheme and register v1beta2 search resources.")
	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme() error = %v", err)
	}

	t.Log("Verify that the scheme creates each search resource kind.")
	tests := []struct {
		gvk  string
		kind string
	}{
		{gvk: DynamoGraphDeploymentRequestGVK.String(), kind: "DynamoGraphDeploymentRequest"},
		{gvk: DynamoGraphDeploymentRunGVK.String(), kind: "DynamoGraphDeploymentRun"},
		{gvk: DynamoGraphDeploymentCandidateGVK.String(), kind: "DynamoGraphDeploymentCandidate"},
	}
	for _, test := range tests {
		t.Run(test.kind, func(t *testing.T) {
			t.Logf("Create the registered %s resource.", test.gvk)
			gvk := GroupVersion.WithKind(test.kind)
			if _, err := scheme.New(gvk); err != nil {
				t.Fatalf("scheme.New(%s) error = %v", test.gvk, err)
			}
		})
	}
}

func TestRunSpecIsExactRequestSpec(t *testing.T) {
	t.Parallel()

	t.Log("Compare the run spec type with the request spec type.")
	requestType := reflect.TypeOf(DynamoGraphDeploymentRequestSpec{})
	runType := reflect.TypeOf(DynamoGraphDeploymentRunSpec{})
	if requestType != runType {
		t.Fatalf("run spec type %v differs from request spec type %v", runType, requestType)
	}
}

func TestCandidateSpecUsesV1Beta1DGDContract(t *testing.T) {
	t.Parallel()

	t.Log("Verify that the candidate spec inlines the v1beta1 DGD contract.")
	candidateSpecType := reflect.TypeOf(DynamoGraphDeploymentCandidateSpec{})
	dgdField := candidateSpecType.Field(0)
	dgdType := reflect.TypeOf(v1beta1.DynamoGraphDeploymentSpec{})
	if !dgdField.Anonymous || dgdField.Type != dgdType || dgdField.Tag.Get("json") != ",inline" {
		t.Fatalf("candidate DGD field = %#v, want anonymous inline %v", dgdField, dgdType)
	}

	t.Log("Verify that resolved parameters are part of the immutable candidate spec.")
	parametersField, ok := candidateSpecType.FieldByName("Parameters")
	if !ok {
		t.Fatal("candidate spec has no Parameters field")
	}
	rawExtensionType := reflect.TypeOf((*runtime.RawExtension)(nil))
	if parametersField.Type != rawExtensionType {
		t.Fatalf("candidate parameters type = %v, want %v", parametersField.Type, rawExtensionType)
	}
}

func TestRequestSpecPreservesMVPFieldsWithOptionalPostMVPControls(t *testing.T) {
	t.Parallel()

	t.Log("Verify the request spec preserves MVP search intent and layers post-MVP controls.")
	specType := reflect.TypeOf(DynamoGraphDeploymentRequestSpec{})
	got := make([]string, specType.NumField())
	for i := range specType.NumField() {
		got[i] = specType.Field(i).Name
	}
	want := []string{
		"ModelRef",
		"Backends",
		"Image",
		"Hardware",
		"Workload",
		"Objective",
		"Search",
		"Recommendation",
		"Rerun",
		"Overrides",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("request spec fields = %v, want %v", got, want)
	}

	t.Log("Verify the model reference preserves identity and revision before optional post-MVP controls.")
	modelType := reflect.TypeOf(ModelReference{})
	got = make([]string, modelType.NumField())
	for i := range modelType.NumField() {
		got[i] = modelType.Field(i).Name
	}
	want = []string{"Name", "Revision", "RemoteCode", "Cache"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("model reference fields = %v, want %v", got, want)
	}

	t.Log("Verify MVP recommendation remains pointer-shaped and optional.")
	assertOptionalFieldType(t, specType, "Recommendation", reflect.TypeOf((*RecommendationSpec)(nil)))

	t.Log("Verify post-MVP request controls are optional.")
	assertOptionalFieldType(t, specType, "Image", reflect.TypeOf(""))
	assertOptionalFieldType(t, specType, "Rerun", reflect.TypeOf((*RerunSpec)(nil)))
	assertOptionalFieldType(t, specType, "Overrides", reflect.TypeOf((*OverridesSpec)(nil)))
	assertOptionalFieldType(t, modelType, "RemoteCode", reflect.TypeOf(RemoteCodePolicy("")))
	assertOptionalFieldType(t, modelType, "Cache", reflect.TypeOf((*ModelCacheSpec)(nil)))

	t.Log("Verify post-MVP override controls retain their typed API shapes.")
	overridesType := reflect.TypeOf(OverridesSpec{})
	assertOptionalFieldType(t, overridesType, "ProfilingJob", reflect.TypeOf((*batchv1.JobSpec)(nil)))
	assertOptionalFieldType(t, overridesType, "DGD", reflect.TypeOf((*runtime.RawExtension)(nil)))
}

func assertOptionalFieldType(t *testing.T, owner reflect.Type, name string, want reflect.Type) {
	t.Helper()

	field, ok := owner.FieldByName(name)
	if !ok {
		t.Fatalf("%v has no %s field", owner, name)
	}
	if field.Type != want {
		t.Fatalf("%v.%s type = %v, want %v", owner, name, field.Type, want)
	}
	if !strings.Contains(field.Tag.Get("json"), "omitempty") {
		t.Fatalf("%v.%s json tag = %q, want omitempty", owner, name, field.Tag.Get("json"))
	}
}
